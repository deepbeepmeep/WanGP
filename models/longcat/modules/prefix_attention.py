# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""LongCat's clean-prefix topology, shared by video and avatar transformers."""
import torch
from mmgp import offload
from shared import attention_kit
from shared.attention import pay_attention
from shared.kernels import ops
from .blocks import _take_tensor


def split_map(hidden_size):
    return {"qkv": {"mapped_modules": ["q", "k", "v"], "split_sizes": [hidden_size] * 3}}


def prefix_attention(attn, x, shape, num_cond_latents=0, *, avatar=False, layer_cache=None,
                     build_cache=False, rope=None, num_ref_latents=None, ref_img_index=None,
                     mask_frame_range=None, ref_target_masks=None):
    """Clean frames never attend noisy frames. Cached calls contain only noisy frames.

    Avatar continuation keeps its anchor separate from the overlap. The cache stores
    normalized, rotated keys using PrefixCache, whose MMGP ring owns RAM transfers.
    """
    x = _take_tensor(x)
    batch, tokens, dim = x.shape
    spatial = shape[1] * shape[2]
    cond_frames = int(num_cond_latents or 0)
    cached = layer_cache is not None and not build_cache
    prefix_tokens = cond_frames * spatial
    offset = prefix_tokens if cached else 0
    prefix, prefix_output = None, None
    if not cached and prefix_tokens:
        clean = x[:, :prefix_tokens].contiguous()
        clean_list = [clean]
        clean = None
        prepared = offload.prepare_linear_input(clean_list, (attn.q, attn.k, attn.v))
        projections = [offload.linear_rows(layer, prepared, 0, dim).view(batch, prefix_tokens, attn.num_heads, attn.head_dim)
                       for layer in (attn.q, attn.k, attn.v)]
        prepared = None
        q, k, v = projections
        projections.clear()
        ops.interleaved_norm_rope_(q, k, rope[:, :prefix_tokens], attn.q_norm, attn.k_norm)
        boundaries = [0, spatial, prefix_tokens] if avatar and cond_frames > 1 else [0, prefix_tokens]
        clean_output = torch.empty_like(q)
        for start, stop in zip(boundaries, boundaries[1:]):
            if stop > start:
                clean_output[:, start:stop] = pay_attention(
                    [q[:, start:stop].contiguous(), k[:, start:stop].contiguous(), v[:, start:stop].contiguous()])
        q = None
        prefix_output = attn.proj(clean_output.flatten(2))
        clean_output = None
        if build_cache:
            layer_cache.store(k, v, kit=True)
            prefix = layer_cache.kv_prefix()
        else:
            prefix = (k, v)
        k = v = None
        x = x[:, prefix_tokens:].contiguous()
        offset = prefix_tokens
    elif cached:
        prefix = layer_cache.kv_prefix()

    ref_map = None
    if x.shape[1]:
        noisy_tokens = x.shape[1]
        noisy_rope = rope[:, offset:offset + noisy_tokens]
        start_frame = (ref_img_index or 0) - (mask_frame_range or 0) - cond_frames + (num_ref_latents or 0)
        stop_frame = (ref_img_index or 0) + (mask_frame_range or 0) - cond_frames + (num_ref_latents or 0) + 1
        masked = avatar and (mask_frame_range or 0) > 0 and 0 <= start_frame < stop_frame <= noisy_tokens // spatial
        custom_attention = masked or ref_target_masks is not None
        if ref_target_masks is not None:
            ref_map = x.new_zeros((ref_target_masks.shape[0], noisy_tokens))

        def attend(qkv, group):
            q, k, v = qkv
            qkv.clear()
            if ref_map is not None:
                from ..audio_process.torch_utils import get_attn_map_with_target
                # Read before attention recycles q. Each group contributes its actual head count.
                ref_map.add_(get_attn_map_with_target(q, k, shape, ref_target_masks, split_num=1),
                             alpha=q.shape[2] / attn.num_heads)
            if not masked:
                items = [q, k, v]
                q = k = v = None
                return pay_attention(items, recycle_q=True)
            result = torch.empty_like(q)
            start, stop = start_frame * spatial, stop_frame * spatial
            for left, right, key_start in ((0, start, 0), (start, stop, spatial), (stop, noisy_tokens, 0)):
                if right > left:
                    result[:, left:right] = pay_attention([q[:, left:right].contiguous(),
                                                          k[:, key_start:].contiguous(), v[:, key_start:].contiguous()])
            return result

        items = [x]
        x = None
        output = attention_kit.qkv_attention(
            items, attn.q, attn.k, attn.v, attn.num_heads, attn.head_dim,
            lambda q, k, group: ops.interleaved_norm_rope_(q, k, noisy_rope, attn.q_norm, attn.k_norm),
            fused_quant=lambda q, k, group, core, settings: ops.split_half_norm_rope_quant_(
                q, k, noisy_rope, attn.q_norm, attn.k_norm, core, settings, interleaved=True),
            kv_prefix=prefix, attention_fn=attend if custom_attention else None, out_proj=attn.proj)
        if prefix_output is not None:
            joined = output.new_empty(batch, tokens, dim)
            joined[:, :prefix_tokens].copy_(prefix_output)
            joined[:, prefix_tokens:].copy_(output)
            output = joined
    else:
        output = prefix_output
    return output, ref_map
