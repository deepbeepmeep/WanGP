"""Prism/MOVA inference architecture; derived from Tencent-Hunyuan/Prism (MIT).

See LICENSE and NOTICE for the original revision and MOVA attribution.
"""
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from shared import attention_kit
from shared.attention import pay_attention
from shared.utils.phase_progress import check_abort
from .sparse_attention import block_layout, sparse_attention


def timestep_embedding(dim, time):
    angles = torch.outer(time.double(), 10000 ** (-torch.arange(dim // 2, device=time.device, dtype=torch.float64) / (dim // 2)))
    return torch.cat((angles.cos(), angles.sin()), -1).float()


def frequencies(dim, positions):
    inverse = 1 / (10000 ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float64) / dim))
    angles = positions.double()[:, None] * inverse
    return torch.polar(torch.ones_like(angles), angles)


def self_rope(grid, device, audio=False):
    if audio:
        return frequencies(128, torch.arange(grid[0], device=device))[:, None]
    t, h, w = grid
    parts = [frequencies(d, torch.arange(n, device=device)) for d, n in zip((44, 42, 42), grid)]
    return torch.cat((parts[0][:, None, None].expand(t, h, w, -1), parts[1][None, :, None].expand(t, h, w, -1),
                      parts[2][None, None].expand(t, h, w, -1)), -1).reshape(t * h * w, 1, 64)


def bridge_rope(grid, audio_tokens, fps, device, dtype):
    inverse = 1 / (10000 ** (torch.arange(0, 128, 2, device=device).float() / 128))
    positions = (torch.arange(grid[0], device=device).float().repeat_interleave(grid[1] * grid[2]) * (200 / fps),
                 torch.arange(audio_tokens, device=device).float())
    result = []
    for position in positions:
        angles = position[:, None] * inverse
        angles = torch.cat((angles, angles), -1)[None, :, None]
        result.append((angles.cos().to(dtype), angles.sin().to(dtype)))
    return result


def rotate_rows(x, rope, first, last, bridge=False):
    if bridge:
        cos, sin = (part[:, first:last] for part in rope)
        half = x.shape[-1] // 2
        # Upstream compiles this expression: its BF16 inputs are evaluated in
        # FP32 registers and rounded only at the output.
        values = x.float()
        return (values * cos.float() + torch.cat((-values[..., half:], values[..., :half]), -1) * sin.float()).to(x.dtype)
    # The released self-attention uses complex128 RoPE, then rounds back to BF16.
    pairs = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * rope[None, first:last]).flatten(-2).to(x.dtype)


def modulate_(x, shift, scale):
    # Preserve the released compiled expression's single BF16 rounding, without
    # forcing compilation or retaining full-size FP32 visual activations.
    multiplier, shift = 1 + scale.float(), shift.float()
    for first in range(0, x.shape[1], 1024):
        part = x[:, first:first + 1024]
        part.copy_(part.float().mul_(multiplier).add_(shift))
    return x


class RMSNorm(nn.RMSNorm):
    def forward(self, x, variance=None, heads=None):
        if variance is None:
            return super().forward(x)
        # Invoke the module even for head slices so MMGP applies norm LoRA deltas.
        normalized = (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return normalized * self.weight[heads.start * 128:heads.stop * 128]


class Attention(nn.Module):
    def __init__(self, dim, heads, eps=1e-6, kv_dim=None):
        super().__init__()
        self.heads, self.dim, self.eps = heads, dim, eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim or dim, dim)
        self.v = nn.Linear(kv_dim or dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def norm_rope(self, rope):
        def apply(query, key, group):
            for index, (tensor, norm, heads) in enumerate(((query, self.norm_q, group.q), (key, self.norm_k, group.kv))):
                if tensor is None:
                    continue
                def transform(part, first, last):
                    if group.mean_squares is None:
                        normalized = norm(part.flatten(2)).view_as(part)
                    else:
                        variance = group.mean_squares[index][:, first:last]
                        normalized = norm(part.flatten(2), variance, heads).view_as(part)
                    return rotate_rows(normalized, rope, first, last)
                attention_kit.rows_(tensor, transform)
        return apply

    def forward(self, x, rope, sparse=None):
        holder = [x]
        del x
        attention_fn = None if sparse is None else partial(sparse_attention, layout=sparse[0], grid=sparse[1])
        return attention_kit.qkv_attention(holder, self.q, self.k, self.v, self.heads, 128, self.norm_rope(rope), norm_spans_heads=True, attention_fn=attention_fn,
                                           out_proj=self.o)

    def cross(self, x, context, q_rope=None, k_rope=None, capture=None, cached=None):
        q = self.q(x).unflatten(-1, (self.heads, 128))
        del x
        if cached is None:
            k = self.k(context).unflatten(-1, (self.heads, 128))
            values = self.v(context).unflatten(-1, (self.heads, 128))
        else:
            k, values = (tensor.to(q.device) for tensor in cached)
        del context
        transforms = [(q, self.norm_q, q_rope)]
        if cached is None:
            transforms.append((k, self.norm_k, k_rope))
        for tensor, norm, rope in transforms:
            def transform(part, first, last):
                part = norm(part.flatten(2)).view_as(part)
                return part if rope is None else rotate_rows(part, rope, first, last, bridge=True)
            attention_kit.rows_(tensor, transform)
        if capture is not None:
            # One step's conditioning cache; keep its many video-token K/V on CPU.
            capture.extend((k.to('cpu'), values.to('cpu')))
        del transforms, tensor
        tensors = [q, k, values]
        del q, k, values
        return self.o(pay_attention(tensors, recycle_q=True).flatten(2))


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, eps=1e-6):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(dim, heads, eps), Attention(dim, heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))
        self.ffn_ratio = math.ceil(ffn_dim / dim)

    def forward(self, x, context, time, rope, sparse=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (self.modulation + time).chunk(6, dim=1)
        x = x + gate_a * self.self_attn(modulate_(self.norm1(x), shift_a, scale_a), rope, sparse)
        x.add_(self.cross_attn.cross(self.norm3(x), context))
        # Bound the expanded visual FFN; small audio/text tensors stay unchunked.
        chunk = max(1024, math.ceil(x.shape[1] / self.ffn_ratio)) if x.shape[1] > 4096 else x.shape[1]
        for first in range(0, x.shape[1], chunk):
            part = x[:, first:first + chunk]
            transformed = modulate_(self.norm2(part), shift_f, scale_f)
            part.add_(gate_f * self.ffn(transformed))
            check_abort()
        return x


class Head(nn.Module):
    def __init__(self, dim, output, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, output)
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x, time):
        shift, scale = (self.modulation + time[:, None]).chunk(2, dim=1)
        return self.head(self.norm(x).mul_(1 + scale).add_(shift))


class TimeLinear(nn.Linear):
    def forward(self, x):
        # Upstream requests FP32 timestep math while storing BF16 weights. Bound
        # temporary FP32 weights without changing checkpoint storage precision.
        output = x.new_empty((*x.shape[:-1], self.out_features), dtype=torch.float32)
        for first in range(0, self.out_features, 1024):
            last = min(first + 1024, self.out_features)
            output[..., first:last] = F.linear(x.float(), self.weight[first:last].float(), self.bias[first:last].float())
        return output


class Tower(nn.Module):
    def __init__(self, cfg, audio=False, blocks=True):
        super().__init__()
        dim = cfg['dim']
        self.dim, self.freq_dim, self.audio = dim, cfg['freq_dim'], audio
        self.patch_size = tuple(cfg['patch_size'])
        conv = nn.Conv1d if audio else nn.Conv3d
        self.patch_embedding = conv(cfg['in_dim'], dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(cfg['text_dim'], dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(TimeLinear(self.freq_dim, dim), nn.SiLU(), TimeLinear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), TimeLinear(dim, 6 * dim))
        self.head = Head(dim, cfg['out_dim'] * math.prod(self.patch_size), cfg['eps'])
        if blocks:
            self.blocks = nn.ModuleList(DiTBlock(dim, cfg['num_heads'], cfg['ffn_dim'], cfg['eps']) for _ in range(cfg['num_layers']))

    def prepare(self, latents, context, timestep):
        time = timestep_embedding(self.freq_dim, timestep)
        time = self.time_embedding(time)
        modulation = self.time_projection(time).unflatten(1, (6, self.dim))
        x = self.patch_embedding(latents)
        grid = x.shape[2:]
        return x.flatten(2).transpose(1, 2).contiguous(), self.text_embedding(context), time.to(latents.dtype), modulation.to(latents.dtype), grid

    def finish(self, x, time, grid):
        x = self.head(x, time)
        if self.audio:
            return x.transpose(1, 2)
        t, h, w = grid
        return x.reshape(x.shape[0], t, h, w, 2, 2, 16).permute(0, 6, 1, 2, 4, 3, 5).reshape(x.shape[0], 16, t, h * 2, w * 2)


class Conditioner(nn.Module):
    def __init__(self, dim, kv_dim):
        super().__init__()
        self.y_norm = nn.LayerNorm(kv_dim, eps=1e-6)
        self.inner = Attention(dim, dim // 128, kv_dim=kv_dim)

    def forward(self, x, y, x_rope, y_rope, capture=None, cached=None):
        return self.inner.cross(x, self.y_norm(y) if cached is None else None, x_rope, y_rope, capture, cached)


class FusionBlock(nn.Module):
    def __init__(self, video_cfg, audio_cfg):
        super().__init__()
        self.video_block = DiTBlock(video_cfg['dim'], video_cfg['num_heads'], video_cfg['ffn_dim'])
        self.audio_block = DiTBlock(audio_cfg['dim'], audio_cfg['num_heads'], audio_cfg['ffn_dim'])
        self.a2v_conditioner = Conditioner(video_cfg['dim'], audio_cfg['dim'])
        self.v2a_conditioner = Conditioner(audio_cfg['dim'], video_cfg['dim'])

    def forward(self, states, contexts, modulations, ropes, cross_ropes, sparse=None, audio_cache=None, audio_only=False):
        outputs = []
        # CFG branches are sequential inside this loaded fusion block.
        for branch, (v_context, a_context) in enumerate(contexts):
            video, audio = states.pop(0)
            if audio_only:
                audio = audio + self.v2a_conditioner(audio, None, cross_ropes[1], None, cached=audio_cache)
                outputs.append((None, self.audio_block(audio, a_context, modulations[1], ropes[1])))
                check_abort()
                continue
            v_residual = self.a2v_conditioner(video, audio, cross_ropes[0], cross_ropes[1])
            a_residual = self.v2a_conditioner(audio, video, cross_ropes[1], cross_ropes[0], capture=audio_cache if branch == 0 else None)
            video = video + v_residual
            audio = audio + a_residual
            del v_residual, a_residual
            video = self.video_block(video, v_context, modulations[0], ropes[0], sparse)
            check_abort()
            audio = self.audio_block(audio, a_context, modulations[1], ropes[1])
            outputs.append((video, audio))
            check_abort()
        return outputs


class PrismTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        video, audio = config['video_dit'], config['audio_dit']
        self.video_dit = Tower(video, blocks=False)
        self.audio_dit = Tower(audio, audio=True, blocks=False)
        self.fusion_blocks = nn.ModuleList(FusionBlock(video, audio) for _ in range(audio['num_layers']))
        self.remaining_video_blocks = nn.ModuleList(DiTBlock(video['dim'], video['num_heads'], video['ffn_dim']) for _ in range(video['num_layers'] - audio['num_layers']))
        self._convertWeightsFloatTo = None

    def preprocess_loras(self, model_type, state):
        from .lora import map_video_lora
        return map_video_lora(state, len(self.fusion_blocks), len(self.remaining_video_blocks))

    def forward(self, video, audio, contexts, timestep, audio_timestep, fps, sparse=True, audio_cache=None, audio_only=False):
        tower = self.video_dit
        states, embedded_contexts = [], []
        for v_context, a_context in contexts:
            if audio_only:
                vx = vc = vm = None
            else:
                vx, vc, vt, vm, grid = tower.prepare(video.to(torch.bfloat16), v_context, timestep)
            ax, ac, at, am, audio_grid = self.audio_dit.prepare(audio.to(torch.bfloat16), a_context, audio_timestep)
            states.append((vx, ax))
            embedded_contexts.append((vc, ac))
        del vx, ax
        ropes = (None if audio_only else self_rope(grid, audio.device), self_rope(audio_grid, audio.device, audio=True))
        cross_ropes = (None, audio_cache['rope']) if audio_only else bridge_rope(grid, audio_grid[0], fps, audio.device, torch.bfloat16)
        if audio_cache is not None and not audio_only:
            audio_cache['rope'] = cross_ropes[1]
            audio_cache['layers'] = [[] for _ in self.fusion_blocks]
        sparse = (block_layout(grid, audio.device), grid) if sparse and not audio_only else None
        for i, block in enumerate(self.fusion_blocks):
            cache = audio_cache['layers'][i] if audio_cache is not None else None
            states = block(states, embedded_contexts, (vm, am), ropes, cross_ropes, sparse, cache, audio_only)
            check_abort()
        if audio_only:
            return [self.audio_dit.finish(ax, at, audio_grid) for _, ax in states]
        for block in self.remaining_video_blocks:
            outputs = []
            for context in embedded_contexts:
                video, audio = states.pop(0)
                outputs.append((block(video, context[0], vm, ropes[0], sparse), audio))
                check_abort()
            states = outputs
        return [(tower.finish(vx, vt, grid), self.audio_dit.finish(ax, at, audio_grid)) for vx, ax in states]
