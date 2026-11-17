# Kandinsky 6 video + audio diffusion transformer, inference port of kandinskylab/kandinsky-6
# (kandinsky/core/components/dit.py and piflow_dit.py, commit bf676db, MIT License, see LICENSE in this folder).
# Module names match the released Diffusers checkpoints. Their FP32 tensors (embeddings, q/k norms) stay FP32, the hidden states are
# BF16, and the AdaLN modulation, scale/shift/gate arithmetic and RoPE are computed in FP32 as upstream does.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgp import cuda_graphs

from shared import attention_kit
from shared.kernels import kernel_policy, ops
from shared.utils.phase_progress import check_abort

try:
    from . import triton_kernels as _fused
except (ImportError, OSError, RuntimeError):
    _fused = None

_CHUNK_ELEMENTS = 1 << 24  # FP32 scratch bound (64 MB) of the row-wise elementwise passes
_FFN_ELEMENTS = 1 << 26  # FFN intermediate bound per chunk of tokens


def get_freqs(dim, max_period=10000.0):
    return torch.exp(-math.log(max_period) * torch.arange(0, dim, dtype=torch.float32, device="cpu") / dim)


def _row_ranges(x, elements=_CHUNK_ELEMENTS):
    step = max(1, elements // (x.shape[0] * x.shape[-1]))
    return [(start, min(start + step, x.shape[1])) for start in range(0, x.shape[1], step)]


def _fused_kernels(x):
    """The fused Triton kernels (Triton installed, CUDA GPU with native BF16) when approximate kernels are allowed: their reductions and
    rounding differ from the PyTorch path at the rounding level. The PyTorch path otherwise."""
    return _fused is not None and x.is_cuda and kernel_policy.allow_approximate() and _fused.supported(x.device)


def scale_shift_norm(x, scale, shift):
    """(layer_norm(x.float()) * (scale + 1) + shift).to(x.dtype) for x (batch, tokens, dim), by chunks of tokens: same values as the
    whole-tensor expression of upstream apply_scale_shift_norm with a bounded FP32 scratch."""
    if _fused_kernels(x):
        return _fused.scale_shift_norm(x, scale, shift)
    scale, shift = scale.float().unsqueeze(1) + 1.0, shift.float().unsqueeze(1)
    out = torch.empty_like(x)
    for start, stop in _row_ranges(x):
        out[:, start:stop] = F.layer_norm(x[:, start:stop].float(), x.shape[-1:]).mul_(scale).add_(shift)
    return out


def gate_add_(x, out, gate):
    """x = (x.float() + gate.float() * out.float()).to(x.dtype) in place, by chunks of tokens (upstream apply_gate_sum)."""
    if _fused_kernels(x):
        return _fused.gate_add_(x, out, gate)
    gate = gate.float().unsqueeze(1)
    for start, stop in _row_ranges(x):
        part = x[:, start:stop]
        part.copy_(out[:, start:stop].float().mul_(gate).add_(part.float()))
    return x


def _rope_rows(rope, first, last):
    return None if rope is None else (rope[0][first:last], rope[1][first:last])


def _norm_rope_rows(x, norm, rope):
    """q/k RMSNorm computed in FP32 and rounded to the activation dtype, then RoPE in FP32, of rows (batch, tokens, heads, head_dim):
    the rounding points of upstream (query_norm, then apply_rotary(...).type_as(q)). rope = (cos, sin) of shape (tokens, 1, head_dim / 2)."""
    dtype = x.dtype
    x = norm(x.float()).to(dtype)
    if rope is None:
        return x
    cos, sin = rope
    pairs = x.float().unflatten(-1, (-1, 2))
    x0, x1 = pairs[..., 0], pairs[..., 1]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2).to(dtype)


def _norm_rope_(tensor, norm, rope):
    if _fused_kernels(tensor):
        _fused.norm_rope_(tensor, norm.weight, rope, torch.finfo(torch.float32).eps if norm.eps is None else norm.eps)
        return
    attention_kit.rows_(tensor, lambda part, first, last: _norm_rope_rows(part, norm, _rope_rows(rope, first, last)))


class _TimestepEmbedder(nn.Module):
    def __init__(self, model_dim, time_dim):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, time_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_dim, time_dim)


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim, time_dim, max_period=10000.0):
        super().__init__()
        self.freqs = get_freqs(model_dim // 2, max_period)  # plain CPU attribute: computed on the CPU as upstream, never a checkpoint weight
        self.timestep_embedder = _TimestepEmbedder(model_dim, time_dim)

    def forward(self, time):
        args = torch.outer(time.float(), self.freqs.to(time.device))
        embedder = self.timestep_embedder
        return embedder.linear_2(embedder.act(embedder.linear_1(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))))


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, x):
        return self.norm(self.in_layer(x.to(self.in_layer.weight.dtype)))


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x, token_embeddings=None):
        """FP32 patch projection plus the token-type embeddings (added in place), rounded to the BF16 hidden states."""
        batch, frames, height, width, channels = x.shape
        pT, pH, pW = self.patch_size
        x = x.view(batch, frames // pT, pT, height // pH, pH, width // pW, pW, channels).permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(4, 7)
        x = self.in_layer(x.to(self.in_layer.weight.dtype))
        if token_embeddings is not None:
            x.add_(token_embeddings[:, :, None, None, :])
        return x.to(torch.bfloat16)


class Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.act = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)

    def forward(self, x):
        # AdaLN parameters in FP32, as upstream F.linear(silu(x.float()), weight.float(), bias.float())
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            return self.out_layer(self.act(x.float()))


class _GELUProjection(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.proj = nn.Linear(dim, ff_dim, bias=False)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.net = nn.ModuleList([_GELUProjection(dim, ff_dim), nn.Dropout(0.0), nn.Linear(ff_dim, dim, bias=False)])

    def forward(self, x):
        return self.net[2](F.gelu(self.net[0].proj(x)))


def modulated_ffn_(x, feed_forward, ff_dim, shift, scale, gate):
    """x = gate_sum(x, feed_forward(scale_shift_norm(x))), in place by chunks of tokens: the norm, the FFN and the gate are row-wise, so
    only one chunk of the FFN intermediate exists at a time."""
    for start, stop in _row_ranges(x, max(_FFN_ELEMENTS // ff_dim * x.shape[-1], 1)):
        part = x[:, start:stop]
        gate_add_(part, feed_forward(scale_shift_norm(part, scale, shift)), gate)
    return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = dim // head_dim, head_dim
        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(dim, dim)

    def _norm_rope_(self, query, key, rope):
        """attention_kit callback: q/k norm and RoPE in place by chunks of tokens."""
        for tensor, norm in ((query, self.query_norm), (key, self.key_norm)):
            if tensor is not None:
                _norm_rope_(tensor, norm, rope)

    def forward(self, x_list, rope):
        return attention_kit.qkv_attention(x_list, self.to_query, self.to_key, self.to_value, self.num_heads, self.head_dim,
                                           lambda query, key, group: self._norm_rope_(query, key, rope),
                                           fused_quant=lambda query, key, group, core, settings: ops.split_half_norm_rope_quant_(
                                               query, key, None if rope is None else rope[2], self.query_norm, self.key_norm, core, settings, interleaved=True),
                                           out_proj=self.out_layer)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, q_dim, head_dim, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or q_dim
        self.num_heads, self.head_dim = q_dim // head_dim, head_dim
        self.to_query = nn.Linear(q_dim, q_dim)
        self.to_key = nn.Linear(kv_dim, q_dim)
        self.to_value = nn.Linear(kv_dim, q_dim)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(q_dim, q_dim)

    def forward(self, x_list, cond_list, q_rope=None, kv_rope=None):
        """x and cond handed off in their lists: each is released after its projections when the caller holds no other reference."""
        query = self.to_query(x_list.pop()).unflatten(-1, (self.num_heads, self.head_dim))
        _norm_rope_(query, self.query_norm, q_rope)
        cond = cond_list.pop()
        key = self.to_key(cond).unflatten(-1, (self.num_heads, self.head_dim))
        value = self.to_value(cond).unflatten(-1, (self.num_heads, self.head_dim))
        del cond
        _norm_rope_(key, self.key_norm, kv_rope)
        qkv_list = [query, key, value]
        del query, key, value
        return self.out_layer(attention_kit.kv_first_attention(qkv_list).flatten(2))


class OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim)

    def forward(self, x, grid_shape, time_embed):
        shift, scale = self.modulation(time_embed).chunk(2, dim=-1)
        x = self.out_layer(scale_shift_norm(x, scale, shift))
        frames, height, width = grid_shape
        pT, pH, pW = self.patch_size
        x = x.view(x.shape[0], frames, height, width, -1, pT, pH, pW).permute(0, 1, 5, 2, 6, 3, 7, 4)
        return x.flatten(1, 2).flatten(2, 3).flatten(3, 4)


class OutLayerAudio(nn.Module):
    def __init__(self, model_dim, time_dim, audio_dim):
        super().__init__()
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.out_layer = nn.Linear(model_dim, audio_dim)

    def forward(self, x, time_embed):
        shift, scale = self.modulation(time_embed).chunk(2, dim=-1)
        x = scale_shift_norm(x, scale, shift)
        return self.out_layer(F.layer_norm(x, x.shape[-1:]))  # second norm of the reference training code


class TransformerEncoderBlock(nn.Module):
    """Text self-attention + FFN block."""

    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.ff_dim = ff_dim
        self.text_modulation = Modulation(time_dim, model_dim, 6)
        self.attn = MultiheadSelfAttention(model_dim, head_dim)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x, time_embed, rope):
        shift, scale, gate, ff_shift, ff_scale, ff_gate = self.text_modulation(time_embed).chunk(6, dim=-1)
        gate_add_(x, self.attn([scale_shift_norm(x, scale, shift)], rope), gate)
        return modulated_ffn_(x, self.feed_forward, self.ff_dim, ff_shift, ff_scale, ff_gate)


class TransformerDecoderBlock(nn.Module):
    """Self-attention, cross-attention to the text and FFN of one modality."""

    def __init__(self, model_dim, time_dim, ff_dim, head_dim):
        super().__init__()
        self.ff_dim = ff_dim
        self.visual_modulation = Modulation(time_dim, model_dim, 9)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def attend(self, x, text, time_embed, rope):
        """Self-attention and text cross-attention: returns the cross-attention input (kept for the cross-modal attention) and the
        FFN parameters."""
        shift, scale, gate, ca_shift, ca_scale, ca_gate, *ff_params = self.visual_modulation(time_embed).chunk(9, dim=-1)
        gate_add_(x, self.self_attention([scale_shift_norm(x, scale, shift)], rope), gate)
        pre_cross = scale_shift_norm(x, ca_scale, ca_shift)
        gate_add_(x, self.cross_attention([pre_cross], [text]), ca_gate)
        return pre_cross, ff_params


class FusedTransformerDecoderBlock(nn.Module):
    """Video and audio blocks joined by bidirectional cross-modal attention (ca_rope, cross_gates, fix_modulation)."""

    def __init__(self, model_dim, time_dim, ff_dim, head_dim, model_dim_a, time_dim_a, ff_dim_a, head_dim_a):
        super().__init__()
        self.model_dim, self.model_dim_a = model_dim, model_dim_a
        self.video_dec_block = TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim)
        self.audio_dec_block = TransformerDecoderBlock(model_dim_a, time_dim_a, ff_dim_a, head_dim_a)
        self.va_cross_attention = MultiheadCrossAttention(model_dim, head_dim, model_dim_a)
        self.av_cross_attention = MultiheadCrossAttention(model_dim_a, head_dim_a, model_dim)
        self.va_modulation = Modulation(time_dim, model_dim * 2 + model_dim_a, 1)
        self.av_modulation = Modulation(time_dim_a, model_dim_a * 2 + model_dim, 1)

    def forward(self, vis, aud, text_v, text_a, time_v, time_a, vis_rope, aud_rope):
        # Upstream order with the independent updates moved forward, which releases each attention output and the video cross-attention
        # input as soon as possible: same values.
        vis_pre_cross, vis_ff = self.video_dec_block.attend(vis, text_v, time_v, vis_rope)
        aud_pre_cross, aud_ff = self.audio_dec_block.attend(aud, text_a, time_a, aud_rope)
        dim, dim_a = self.model_dim, self.model_dim_a
        va_shift, va_scale, va_gate = self.va_modulation(time_v).split([dim, dim, dim_a], dim=-1)
        av_shift, av_scale, av_gate = self.av_modulation(time_a).split([dim_a, dim_a, dim], dim=-1)
        # audio queries attend to the video cross-attention input; with cross_gates the audio update uses the va gate and conversely
        kv_list = [vis_pre_cross]
        vis_pre_cross = None
        aud_from_vis = self.av_cross_attention([scale_shift_norm(aud, av_scale, av_shift)], kv_list, aud_rope, vis_rope)
        vis_from_aud = self.va_cross_attention([scale_shift_norm(vis, va_scale, va_shift)], [aud_pre_cross], vis_rope, aud_rope)
        gate_add_(aud, aud_from_vis, va_gate)
        gate_add_(vis, vis_from_aud, av_gate)
        del aud_from_vis, vis_from_aud, aud_pre_cross
        modulated_ffn_(vis, self.video_dec_block.feed_forward, self.video_dec_block.ff_dim, *vis_ff)
        modulated_ffn_(aud, self.audio_dec_block.feed_forward, self.audio_dec_block.ff_dim, *aud_ff)
        return vis, aud


class Kandinsky6Transformer(nn.Module):
    """Kandinsky 6 multimodal DiT with the PiFlow grid output head of the distilled checkpoints."""

    def __init__(self, model_dim, model_dim_a, time_dim, time_dim_a, ff_dim, ff_dim_a, axes_dims, axes_dims_a, num_text_blocks,
                 num_visual_blocks, in_visual_dim, in_audio_dim, in_text_dim, in_text_dim2, patch_size, audio_freqs_scaling,
                 visual_token_type_num_embeddings, n_grid, out_visual_dim, out_audio_dim):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.axes_dims, self.axes_dims_a = tuple(axes_dims), tuple(axes_dims_a)
        self.head_dim, self.head_dim_a = sum(axes_dims), sum(axes_dims_a)
        self.audio_freqs_scaling = audio_freqs_scaling
        self.in_visual_dim, self.in_audio_dim = in_visual_dim, in_audio_dim
        self.n_grid, self.out_visual_dim, self.out_audio_dim = n_grid, out_visual_dim, out_audio_dim
        self.visual_embeddings = VisualEmbeddings(2 * in_visual_dim + 1, model_dim, patch_size)
        self.visual_token_type_embeddings = nn.Embedding(visual_token_type_num_embeddings, model_dim)
        self.audio_embeddings = TextEmbeddings(in_audio_dim, model_dim_a)
        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim * n_grid, patch_size)
        self.audio_out_layer = OutLayerAudio(model_dim_a, time_dim_a, out_audio_dim * n_grid)
        for prefix, dim, t_dim, f_dim, h_dim in (("video", model_dim, time_dim, ff_dim, self.head_dim), ("audio", model_dim_a, time_dim_a, ff_dim_a, self.head_dim_a)):
            setattr(self, f"{prefix}_time_embeddings", TimeEmbeddings(dim, t_dim))
            setattr(self, f"{prefix}_text_embeddings", TextEmbeddings(in_text_dim, dim))
            setattr(self, f"{prefix}_pooled_text_embeddings", TextEmbeddings(in_text_dim2, t_dim))
            setattr(self, f"{prefix}_text_transformer_blocks", nn.ModuleList([TransformerEncoderBlock(dim, t_dim, f_dim, h_dim) for _ in range(num_text_blocks)]))
        self.visual_transformer_blocks = nn.ModuleList([
            FusedTransformerDecoderBlock(model_dim, time_dim, ff_dim, self.head_dim, model_dim_a, time_dim_a, ff_dim_a, self.head_dim_a)
            for _ in range(num_visual_blocks)])

    def lock_checkpoint_fp32_modules(self):
        """The checkpoints store the embeddings and the q/k norms in FP32: MMGP keeps these modules in FP32 (the rest is BF16)."""
        modules = [self.visual_embeddings.in_layer, self.visual_token_type_embeddings, self.audio_embeddings.in_layer, self.audio_embeddings.norm]
        for prefix in ("video", "audio"):
            time_embedder = getattr(self, f"{prefix}_time_embeddings").timestep_embedder
            modules += [time_embedder.linear_1, time_embedder.linear_2]
            for name in ("text_embeddings", "pooled_text_embeddings"):
                embeddings = getattr(self, f"{prefix}_{name}")
                modules += [embeddings.in_layer, embeddings.norm]
        modules += [module for module in self.modules() if isinstance(module, nn.RMSNorm)]
        for module in modules:
            module._lock_dtype = torch.float32

    def rope_tables(self, frames, height, width, text_tokens, audio_tokens, scale_factor, device, reference_frame=False):
        """RoPE (cos, sin, cos / sin pairs) tables of shapes (tokens, 1, head_dim / 2) and (1, tokens, 1, head_dim / 2, 2) (the layout of
        ops.split_half_norm_rope_quant_), computed once per generation: frequencies on the CPU, positions scaled and rotated on the device,
        as upstream RoPE1D / RoPE3D. A reference frame reuses temporal position 0."""
        def table(args):
            cos, sin = torch.cos(args).unsqueeze(1), torch.sin(args).unsqueeze(1)
            return cos, sin, torch.stack((cos, sin), dim=-1)[None]

        def args_1d(dim, length, scaling=1.0):
            return torch.outer(torch.arange(length, dtype=torch.float32, device=device), get_freqs(dim // 2).to(device) * scaling)

        axes = []
        for dim, length, scale in zip(self.axes_dims, (frames, height, width), scale_factor):
            axes.append(args_1d(dim, length) / scale)
        if reference_frame:
            axes[0] = torch.cat([axes[0], axes[0][:1]])
        frames += int(reference_frame)
        args = torch.cat([axes[0].view(frames, 1, 1, -1).expand(frames, height, width, -1), axes[1].view(1, height, 1, -1).expand(frames, height, width, -1),
                          axes[2].view(1, 1, width, -1).expand(frames, height, width, -1)], dim=-1).flatten(0, 2)
        return {"video": table(args), "audio": table(args_1d(self.head_dim_a, audio_tokens, self.audio_freqs_scaling)),
                "text_video": table(args_1d(self.head_dim, text_tokens)), "text_audio": table(args_1d(self.head_dim_a, text_tokens))}

    def _encode_text(self, prefix, text_embeds, pooled, time, rope):
        time_embed = getattr(self, f"{prefix}_time_embeddings")(time) + getattr(self, f"{prefix}_pooled_text_embeddings")(pooled)
        text = getattr(self, f"{prefix}_text_embeddings")(text_embeds).to(torch.bfloat16)
        for block in getattr(self, f"{prefix}_text_transformer_blocks"):
            text = block(text, time_embed, rope)
            check_abort()
        return text, time_embed

    def forward(self, x_video, x_audio, text_embeds, pooled_embed, time, ropes, token_types=None):
        """x_video (batch, frames, height, width, 2 * C + 1) with the conditioning channels, x_audio (batch, tokens, C). Returns the
        PiFlow grids (batch, n_grid, frames, height, width, C) and (batch, n_grid, tokens, C)."""
        with cuda_graphs.scope():  # the blocks are CUDA graphed when the mode is on
            text_v, time_v = self._encode_text("video", text_embeds, pooled_embed, time, ropes["text_video"])
            text_a, time_a = self._encode_text("audio", text_embeds, pooled_embed, time, ropes["text_audio"])
            vis = self.visual_embeddings(x_video, None if token_types is None else self.visual_token_type_embeddings(token_types))
            grid_shape = vis.shape[1:4]
            vis = vis.flatten(1, 3)
            aud = self.audio_embeddings(x_audio).to(torch.bfloat16)
            for block in self.visual_transformer_blocks:
                vis, aud = block(vis, aud, text_v, text_a, time_v, time_a, ropes["video"], ropes["audio"])
                check_abort()
        video = self.out_layer(vis, grid_shape, time_v)
        del vis
        audio = self.audio_out_layer(aud, time_a)
        video = video.view(*video.shape[:-1], self.n_grid, self.out_visual_dim).movedim(-2, 1)
        audio = audio.view(*audio.shape[:-1], self.n_grid, self.out_audio_dim).movedim(-2, 1)
        return video, audio
