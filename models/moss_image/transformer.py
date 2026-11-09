# Adapted from OpenMOSS MOSS-Image, Apache-2.0; see LICENSE.
from types import SimpleNamespace

import torch
from torch import nn
from shared import attention_kit
from shared.attention import pay_attention
from shared.kernels import ops
from shared.utils.lora_mapping import convert_lora_keys
from shared.utils.phase_progress import GenerationAborted


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        return ops.rms_norm_rounded([x], self.eps, manual=True) * self.weight


def attend(x_list, projections, q_norm, k_norm, rope, heads, kv_prefix=None, capture=None):
    def normalize(q, k, group):
        ops.split_half_norm_rope_(q, k, rope, None, q_norm, k_norm)
    def reference_attention(qkv, group):
        # References attend only to references. Borrow their normalized K/V for
        # the body (or the run's cache) before attention consumes the list.
        for index, tensor in enumerate(qkv[1:]):
            if group.kv.start == 0 and group.kv.stop == heads:
                capture[index] = tensor
            else:
                if capture[index] is None:
                    capture[index] = tensor.new_empty((*tensor.shape[:2], heads, tensor.shape[-1]))
                capture[index][:, :, group.kv.start:group.kv.stop].copy_(tensor)
        return pay_attention(qkv, recycle_q=True)
    if kv_prefix is not None and not isinstance(kv_prefix, attention_kit.KVPrefix):  # a cached prefix of one item serves the batch
        batch = x_list[0].shape[0]
        kv_prefix = tuple(tensor.expand(batch, -1, -1, -1) for tensor in kv_prefix)
    return attention_kit.qkv_attention(x_list, *projections, heads, q_norm.weight.numel(), normalize,
                                       kv_prefix=kv_prefix, attention_fn=reference_attention if capture is not None else None,
                                       fused_quant=lambda query, key, group, core, settings: ops.split_half_norm_rope_quant_(query, key, rope, q_norm, k_norm, core, settings)).flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        x = x + self.proj(attend([self.norm1(x)], (self.q, self.k, self.v), self.q_norm, self.k_norm, rope, self.heads))
        return ops.mlp([self.norm2(x)], self.ffn_gate, self.ffn_down, 'situ', up=self.ffn_up, residual=x)


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')})
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def update(self, x, rope, modulation, kv_prefix=None, capture=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        shape = x.shape
        x = x.view(-1, shape[-1])
        spans = [(0, x.shape[0], 0)]
        y = ops.norm_modulate([x], self.norm1, shift_a, scale_a, spans).view(shape)
        y_list, y = [y], None
        attended = attend(y_list, tuple(self.proj[name] for name in ('q', 'k', 'v')), self.q_norm, self.k_norm, rope, self.heads, kv_prefix, capture)
        ops.gated_linear(self.proj['o'], [attended.view(-1, shape[-1])], x, gate_a, spans)
        del attended
        y = ops.norm_modulate([x], self.norm2, shift_f, scale_f, spans)
        y_list, y = [y], None
        ops.mlp(y_list, self.proj['ffn_up_gate'], self.proj['ffn_down'], 'situ', residual=x, gate=gate_f[0])
        return x.view(shape)

    def update_reference(self, reference, rope, modulation, cache_layer):
        kv_prefix = None
        if reference is not None:
            kv_prefix = [None, None]
            reference = self.update(reference, rope, modulation, capture=kv_prefix)
            if cache_layer is not None:
                cache_layer.store(*kv_prefix, kit=True)
        if cache_layer is not None:
            kv_prefix = cache_layer.kv_prefix()
        return reference, kv_prefix

    def forward(self, streams, ropes, modulation, skip_branch=-1, reference=None, reference_rope=None, reference_modulation=None, cache_layer=None):
        # HQ references follow each prompt's length and skipped blocks. Keep
        # their states separate while executing both branches in one layer load.
        separate_references = isinstance(reference_rope, list)
        if not separate_references:
            reference, kv_prefix = self.update_reference(reference, reference_rope, reference_modulation, cache_layer)
        for branch in range(len(streams)):
            if branch == skip_branch:
                continue
            if separate_references:
                reference[branch], kv_prefix = self.update_reference(reference[branch], reference_rope[branch], reference_modulation,
                                                                    None if cache_layer is None else cache_layer[branch])
            x, streams[branch] = streams[branch], None
            streams[branch] = self.update(x, ropes[branch], modulation, kv_prefix)
        return streams, reference


def rope_embed(height, width, text_length, device, head_dim=128, base=10000.0, reference_index=0, image_text_position=None):
    bands = 1 / (base ** (torch.arange(head_dim // 8, device='cpu', dtype=torch.float32) / (head_dim // 8)))
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    image = torch.stack((torch.full_like(h, reference_index), torch.full_like(h, text_length if image_text_position is None else image_text_position), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands.to(device)
    return torch.stack((phase.cos().flatten(-2), phase.sin().flatten(-2)), -1)[None, :, None]


class MOSSImageTransformer(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__()
        self.config = SimpleNamespace(dim=dim, freq_dim=freq_dim, num_heads=num_heads, latent_channels=latent_channels, rope_base=rope_base)
        self._interrupt = False
        self.split_linear_modules_map = {'qkv': {'mapped_modules': ['q', 'k', 'v'], 'split_sizes': [dim] * 3}}
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        # Holds no weights itself: the lock keeps MMGP from rounding the float32 timestep to BF16 before the FP64 sinusoid (forward casts
        # the latents to the video_embedder's dtype itself)
        self._lock_dtype = torch.float32

    def preprocess_loras(self, model_type, state_dict):
        return convert_lora_keys(state_dict, dict(self.named_modules()), split_linear_modules_map=self.split_linear_modules_map)

    def time_conditioning(self, timestep):
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=timestep.device, dtype=torch.float64) / half)
        phase = timestep.double().reshape(1, 1) * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        modulation = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(1, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(1, 2, -1)
        return modulation, head

    def forward(self, latents, timestep, contexts, ropes, skip_layer=22, reference_latents=(), reference_rope=None, kv_cache=None):
        """HQ passes reference_rope and kv_cache per branch; Edit SFT shares one.

        Each PrefixCache covers self.layers and is populated at the first call.
        """
        modulation, head = self.time_conditioning(timestep)
        separate_references = isinstance(reference_rope, list)
        reference = [None] * len(contexts) if separate_references else None
        reference_modulation = None
        needs_reference = kv_cache is None or (any(cache.empty() for cache in kv_cache) if separate_references else kv_cache.empty())
        if reference_latents and needs_reference:
            embedded = torch.cat([self.video_embedder(z.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous()) for z in reference_latents], 1)
            if separate_references:
                reference = [embedded.clone() if kv_cache is None or kv_cache[branch].empty() else None for branch in range(len(contexts))]
            else:
                reference = embedded
            del embedded
            reference_modulation, _ = self.time_conditioning(torch.zeros_like(timestep))
        video = self.video_embedder(latents.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous())
        tokens = video.shape[1]
        streams = []
        for context, rope in zip(contexts, ropes):
            text = self.text_embedder.proj(context.to(self.text_embedder.proj.weight.dtype))
            for block in self.text_embedder.blocks:
                text = block(text, rope[:, tokens:])
                if self._interrupt:
                    raise GenerationAborted
            streams.append(torch.cat((video, text.expand(video.shape[0], -1, -1)), 1))
        del video, text
        for index, layer in enumerate(self.layers):
            cache_layer = None
            if kv_cache is not None:
                cache_layer = [cache[index] for cache in kv_cache] if separate_references else kv_cache[index]
            streams, reference = layer(streams, ropes, modulation, 0 if len(contexts) == 2 and index == skip_layer else -1,
                                       reference, reference_rope, reference_modulation, cache_layer)
            if self._interrupt:
                raise GenerationAborted
        results = []
        for x in streams:
            x = self.final_norm_video(x[:, :tokens].to(self.final_norm_video.weight.dtype))
            x = ops.scale_shift([x], head[:, 1, None], head[:, 0, None])
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            results.append(x.transpose(1, 2).reshape_as(latents).float())
        return results
