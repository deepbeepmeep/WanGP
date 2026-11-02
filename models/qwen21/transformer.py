# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.normalization import RMSNorm
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import logging
from shared import attention_kit
from shared.kernels import ops as kernels
from shared.attention import pay_attention
from shared.utils.phase_progress import check_abort
from mmgp import cuda_graphs


def dispatch_attention_fn(qkv_list, attn_mask=None, recycle_q=False):
    # The highest owning caller clears its Q/K/V names before this handoff.
    return pay_attention(qkv_list, attention_mask=None if attn_mask is None else attn_mask.transpose(1, 2), recycle_q=recycle_q)

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.get_logger(__name__)
_IMG_TOKENS_PER_SLOT = 4
_FLEX_BLOCK_SIZE = 128
QwenImage21KVLayerCache = attention_kit.PrefixLayer  # a dataclass: CUDA graphs take its tensors as arguments


class QwenImage21KVCache(attention_kit.PrefixCache):
    """The prefix cache of the text and condition-image tokens (attention_kit.PrefixCache: INT8 with quantize, in RAM with ram, in VRAM as
    far as it fits beside the weights with dynamic) and the layout of the cached steps. native=False when the cached prefixes are read in
    16 bits (NAG)."""

    def __init__(self, blocks, quantize: bool=False, native: bool=True, ram: bool=False, dynamic: bool=False):
        super().__init__(blocks, quantize, native, ram, "Qwen Image 2.1 KV cache", dynamic)
        self.layout = None

    @property
    def layer_caches(self):
        return self.layers

    def get_layer(self, layer_idx: int) -> QwenImage21KVLayerCache:
        return self.layers[layer_idx]

    def clear(self, keep_ring: bool=False):
        self.layout = None
        super().clear(keep_ring)

class QwenImage21TemporalTimesteps(nn.Module):

    def __init__(self, timestep_dim: int, max_period: int=10000, time_factor: float=1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device="cpu") / half)
        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.freqs[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)

class QwenImage21TimestepProjEmbeddings(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))

class QwenImage21ZeroCenterRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, device="cpu"))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_list, hidden_states = [hidden_states], None
        return kernels.zero_center_rms_norm(x_list, self.weight, self.eps)

class QwenImage21TextProjection(nn.Module):

    def __init__(self, context_in_dim: int, hidden_size: int, eps: float=1e-06):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)

class QwenImage21SwiGLUFeedForward(nn.Module):

    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None=None, segments=None) -> torch.Tensor:
        """In place of the disposable normalized activation, or added to residual with the gates of segments."""
        check_abort()
        x_list = hidden_states if isinstance(hidden_states, list) else [hidden_states]
        return kernels.mlp(x_list, self.gate_layer, self.out, "swiglu", up=self.proj, residual=residual, segments=segments)

class QwenImage21AdaLayerNormContinuous(nn.Module):

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float=1e-06):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=False)

    def forward(self, hidden_states: torch.Tensor, conditioning_embedding: torch.Tensor, target_start: int | None=None) -> torch.Tensor:
        hidden_states = hidden_states.pop() if isinstance(hidden_states, list) else hidden_states
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        hidden_states = self.norm(hidden_states)
        return _apply_modulation_(hidden_states, scale, target_start)


def _apply_modulation_(hidden_states, scale, target_start):
    # Prefix tokens always precede the target. Broadcast one modulation row
    # over each region instead of allocating [B, tokens, hidden] via where.
    if target_start is None:
        kernels.scale_shift([hidden_states], scale.unsqueeze(1))
    else:
        kernels.scale_shift([hidden_states[:, :target_start]], scale[-1:].unsqueeze(1))
        kernels.scale_shift([hidden_states[:, target_start:]], scale[:-1].unsqueeze(1))
    return hidden_states


def _add_gated_residual_(hidden_states, output, gate, target_start):
    if target_start is None:
        kernels.gated_residual_(hidden_states, [output], gate.unsqueeze(1), tanh=True)
    else:
        kernels.gated_residual_(hidden_states[:, :target_start], [output[:, :target_start]], gate[-1:].unsqueeze(1), tanh=True)
        kernels.gated_residual_(hidden_states[:, target_start:], [output[:, target_start:]], gate[:-1].unsqueeze(1), tanh=True)

def _gate_segments(gate, batch, tokens, target_start):
    """kernels.mlp segments of the tanh gates (batch + 1 rows with a prefix, the last one for it) over the flattened tokens."""
    gate = gate.tanh()
    if target_start is None:
        return kernels.gate_segments(gate.unsqueeze(1), batch, tokens)
    return [(b * tokens + start, b * tokens + stop, row) for b in range(batch) for start, stop, row in ((0, target_start, gate[-1]), (target_start, tokens, gate[b]))]

def _select_modulation_rows(params: torch.Tensor, target_token_mask: torch.Tensor | None) -> torch.Tensor:
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = (params[:-1].unsqueeze(1), params[-1:].unsqueeze(0))
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)

def _qwenimage21_prefix_segments(image_ids: torch.Tensor, prefix_len: int) -> list[tuple[int, int, bool]]:
    prefix_ids = image_ids[:prefix_len].tolist()
    segments = []
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or prefix_ids[index] != prefix_ids[start]:
            segments.append((start, index, prefix_ids[start] < 0))
            start = index
    return segments

def _qwenimage21_prepare_qkv(attn: 'QwenImage21Attention', hidden_states: torch.Tensor, rotary_emb: torch.Tensor | None, layer_cache: QwenImage21KVLayerCache | None, kv_cache_mode: str | None, cache_write_slice: slice | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    hidden_states = hidden_states.pop() if isinstance(hidden_states, list) else hidden_states
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    del hidden_states
    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))
    # Native RMSNorm arithmetic, FP32 complex RoPE, bounded scratch, disposable Q/K storage.
    _qwenimage21_norm_rope_(query, key, attn, rotary_emb)
    if layer_cache is not None:
        if kv_cache_mode == 'extract' and cache_write_slice is not None:
            layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
        elif kv_cache_mode == 'cached':
            cached_k, cached_v = layer_cache.get()
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)
    seq_len_q = query.shape[1]
    return (query, key, value, seq_len_q)

def _qwenimage21_prefix_attention(query, key, value, segments, key_valid):
    """Attention of the prefix segments (text: causal, condition images), each segment to the keys up to its end, (batch, tokens,
    heads, head_dim) for each segment."""
    outputs = []
    for start, end, is_text in segments:
        seg_mask = None
        if is_text:
            seg_len = end - start
            seg_mask = torch.cat([torch.ones(seg_len, start, dtype=torch.bool, device=query.device), torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device))], dim=1)[None, None]
        if key_valid is not None:
            seg_key_valid = key_valid[:, None, None, :end]
            seg_mask = seg_key_valid if seg_mask is None else seg_mask & seg_key_valid
        outputs.append(dispatch_attention_fn([query[:, start:end], key[:, :end], value[:, :end]], attn_mask=seg_mask))
    return outputs


def _qwenimage21_norm_rope_(query, key, attn, rotary_emb):
    """attention_kit callback: norm_q/norm_k then RoPE of q and/or k in place, by chunks of tokens, as _qwenimage21_prepare_qkv computes
    them; rotary_emb: the complex frequencies of these tokens."""
    for tensor, norm in ((query, attn.norm_q), (key, attn.norm_k)):
        if tensor is not None:
            attention_kit.rows_(tensor, lambda part, first, last: _qwenimage21_norm_rope_rows(part, norm, None if rotary_emb is None else rotary_emb[first:last]))


def _qwenimage21_norm_rope_rows(x, norm, freqs_cis):
    x_list, x = [x], None
    return kernels.complex_norm_rope(x_list, norm, freqs_cis)


def _qwenimage21_norm_rope_quant(query, key, attn, rotary_emb, core, settings):
    """attention_kit fused_quant: the staged Sage 2 quantization of q and/or k with norm_q / norm_k and the complex RoPE, the frequencies
    (tokens, pairs) viewed as a (1, tokens, 1, pairs, 2) cos / sin table of interleaved pairs. The norm is rounded once (diffusers'
    RMSNorm rounds the normalized rows, then their product with the weight), the rotation runs in FP32 as complex_norm_rope."""
    rope = None if rotary_emb is None else torch.view_as_real(rotary_emb)[None, :, None]
    return kernels.split_half_norm_rope_quant_(query, key, rope, attn.norm_q, attn.norm_k, core, settings, interleaved=True)


def _qwenimage21_kit_attention(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, segments):
    """QwenImage21AttnProcessor without mask nor NAG: the prefix tokens (text and condition images) attended by segments as before,
    then the target tokens, whose attention to all the keys goes through attention_kit with the prefix keys and values first."""
    x = hidden_states.pop() if isinstance(hidden_states, list) else hidden_states
    outputs, kv_prefix = [], None
    if kv_cache_mode == 'cached':
        kv_prefix = layer_cache.kv_prefix()
    else:
        prefix_len = segments[-1][1] if segments else 0
        if prefix_len:
            query, key, value, _ = _qwenimage21_prepare_qkv(attn, x[:, :prefix_len], None if rotary_emb is None else rotary_emb[:prefix_len], None, None, None)
            outputs = _qwenimage21_prefix_attention(query, key, value, segments, None)
            if kv_cache_mode == 'extract':  # the target tokens of this step attend the cached prefix as the next steps will
                layer_cache.store(key, value, kit=True)
                kv_prefix = layer_cache.kv_prefix()
            else:
                kv_prefix = (key, value)
            del query, key, value
            rotary_emb = None if rotary_emb is None else rotary_emb[prefix_len:]
        x = x[:, prefix_len:].contiguous()
    x_list = [x]
    del x
    if x_list[0].shape[1]:  # the target tokens (none in NAG's negative branch: the prefix alone)
        outputs.append(attention_kit.qkv_attention(x_list, attn.to_q, attn.to_k, attn.to_v, attn.heads, attn.inner_dim // attn.heads,
                                                   lambda query, key, group: _qwenimage21_norm_rope_(query, key, attn, rotary_emb), kv_prefix=kv_prefix,
                                                   fused_quant=lambda query, key, group, core, settings: _qwenimage21_norm_rope_quant(query, key, attn, rotary_emb, core, settings)))
    kv_prefix = None
    out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
    outputs.clear()
    return attn.to_out[1](attn.to_out[0](out.flatten(2, 3)))


class QwenImage21AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: 'QwenImage21Attention', hidden_states: torch.Tensor, attention_mask: Any | None=None, rotary_emb: torch.Tensor | None=None, layer_cache: QwenImage21KVLayerCache | None=None, kv_cache_mode: str | None=None, cache_write_slice: slice | None=None, segments: list[tuple[int, int, bool]] | None=None, key_valid: torch.Tensor | None=None, nag_cache=None, nag_parameters=None, nag_target_tokens=None) -> torch.Tensor:
        if nag_cache is None and attention_mask is None and key_valid is None and (kv_cache_mode == 'cached' or segments is not None):
            return _qwenimage21_kit_attention(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, segments)
        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)
        query_dtype = query.dtype
        alternate = None
        if nag_cache is not None:
            negative_k, negative_v = nag_cache.get()
            count = nag_target_tokens
            # NAG still needs Q/K/V here. Consume the main attention inputs only
            # after this alternate branch has completed.
            alternate = dispatch_attention_fn([query[:, -count:], torch.cat([negative_k, key[:, -count:]], dim=1), torch.cat([negative_v, value[:, -count:]], dim=1)])
            del negative_k, negative_v
        if segments is None:
            qkv_list = [query, key, value]
            del query, key, value
            hidden_states = dispatch_attention_fn(qkv_list, attn_mask=attention_mask, recycle_q=True)
        else:
            prefix_len = segments[-1][1] if segments else 0
            outputs = _qwenimage21_prefix_attention(query, key, value, segments, key_valid)
            if prefix_len < seq_len_q:
                qkv_list = [query[:, prefix_len:], key, value]
                del query, key, value
                outputs.append(dispatch_attention_fn(qkv_list, attn_mask=None if key_valid is None else key_valid[:, None, None, :], recycle_q=True))
            else:
                del query, key, value
            hidden_states = torch.cat(outputs, dim=1)
            outputs.clear()
        if nag_cache is not None:
            positive = hidden_states[:, -count:].flatten(2)
            alternate = alternate.flatten(2)
            scale, tau, alpha = nag_parameters
            guided = alternate * (1 - scale) + positive * scale
            norm_positive = positive.float().norm(p=1, dim=-1, keepdim=True)
            norm_guided = guided.float().norm(p=1, dim=-1, keepdim=True)
            factor = (norm_positive * tau / (norm_guided + 1e-7)).clamp(max=1).to(guided.dtype)
            guided.mul_(factor).mul_(alpha).add_(positive * (1 - alpha))
            hidden_states[:, -count:] = guided.unflatten(-1, (attn.heads, -1))
        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).to(query_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)

class QwenImage21Attention(nn.Module):
    _default_processor_cls = QwenImage21AttnProcessor
    _available_processors = [QwenImage21AttnProcessor]

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float=1e-06, processor: Any | None=None):
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head
        self.use_bias = False
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.processor = processor if processor is not None else self._default_processor_cls()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)

class QwenImage21TransformerBlock(nn.Module):

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: int=3, eps: float=1e-06):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def _modulate(self, hidden_states: torch.Tensor, mod_params: torch.Tensor, target_start: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        return (_apply_modulation_(hidden_states, scale, target_start), gate)

    def forward(self, hidden_states: torch.Tensor, modulation: torch.Tensor, rotary_emb: torch.Tensor | None=None, attention_mask: Any | None=None, target_token_mask: torch.Tensor | None=None, layer_cache: QwenImage21KVLayerCache | None=None, kv_cache_mode: str | None=None, cache_write_slice: slice | None=None, segments: list[tuple[int, int, bool]] | None=None, key_valid: torch.Tensor | None=None, nag_cache=None, nag_parameters=None, nag_target_tokens=None, target_start=None) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)
        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_start)
        hidden_holder = [img_modulated]
        del img_modulated
        attn_output = self.attn(hidden_states=hidden_holder, attention_mask=attention_mask, rotary_emb=rotary_emb, layer_cache=layer_cache, kv_cache_mode=kv_cache_mode, cache_write_slice=cache_write_slice, segments=segments, key_valid=key_valid, nag_cache=nag_cache, nag_parameters=nag_parameters, nag_target_tokens=nag_target_tokens)
        _add_gated_residual_(hidden_states, attn_output, img_gate1, target_start)
        del attn_output, img_gate1
        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_start)
        hidden_holder = [img_modulated2]
        del img_modulated2
        self.img_mlp(hidden_holder, hidden_states, _gate_segments(img_gate2, *hidden_states.shape[:2], target_start))
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states

class QwenImage21Rope(nn.Module):

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(8192, device="cpu")
        neg_index = torch.arange(1024, device="cpu").flip(0) * -1 - 1
        self.freqs = [torch.cat([self.rope_params(pos_index, dim, theta), self.rope_params(neg_index, dim, theta)], dim=0) for dim in axes_dim]

    def rope_params(self, index: torch.Tensor, dim: int, theta: int=10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=index.device).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, img_shapes: list[tuple[int, int, int]], image_pad_mask: torch.Tensor, device: torch.device, target_offset: tuple[int, int]=(0, 0)) -> torch.Tensor:
        self.freqs = [freq.to(device) for freq in self.freqs]
        frame_index, height_index, width_index = ([], [], [])
        image_height_index, image_width_index = ([], [])
        cursor, position = (0, 0)
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()
        for image_index, (_, height, width) in enumerate(img_shapes):
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len
            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)
            y_offset, x_offset = target_offset if image_index == len(img_shapes) - 1 else (0, 0)
            image_height_index.extend([h + y_offset for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            image_width_index.extend([w + x_offset for _ in range(height) for w in range(-(width - width // 2), width // 2)])
        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))
        frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_index = frame_index.clone()
        width_index = frame_index.clone()
        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)
        return torch.cat([self.freqs[0][frame_index], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1)

class QwenImage21Transformer2DModel(ModelMixin, ConfigMixin):

    def preprocess_loras(self, model_type, state_dict):
        from shared.utils.lora_mapping import convert_lora_keys
        return convert_lora_keys(state_dict, dict(self.named_modules()), split_linear_modules_map=self.split_linear_modules_map, compose_lokr=True)
    _supports_gradient_checkpointing = True
    _no_split_modules = ['QwenImage21TransformerBlock']
    _skip_layerwise_casting_patterns = ['pos_embed', 'norm']
    _repeated_blocks = ['QwenImage21TransformerBlock']
    _skip_keys = ['kv_cache']

    @register_to_config
    def __init__(self, patch_size: int=1, in_channels: int=64, out_channels: int | None=64, num_layers: int=32, attention_head_dim: int=128, num_attention_heads: int=32, context_in_dim: int=4096, mlp_ratio: int=3, axes_dims_rope: tuple[int, int, int]=(16, 56, 56), eps: float=1e-06, causal_condition: bool=True):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        # AI Toolkit / ComfyUI pack SwiGLU output rows as [gate_layer; proj].
        # MMGP splits LoRA B and shares A, preserving the original rank/scaling.
        self.split_linear_modules_map = {
            "gate_up": {"mapped_modules": ("gate_layer", "proj"), "split_sizes": (self.inner_dim * mlp_ratio,) * 2},
        }
        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList([QwenImage21TransformerBlock(dim=self.inner_dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio, eps=eps) for _ in range(num_layers)])
        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)
        self.gradient_checkpointing = False

    @staticmethod
    def build_token_metadata(image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]], num_target_images: int=1) -> tuple[torch.Tensor, torch.Tensor]:
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        block_lengths = [math.prod(shape) for shape in img_shapes]
        if sum(block_lengths) != image_positions.numel():
            raise ValueError(f'img_shapes accounts for {sum(block_lengths)} image tokens but image_pad_mask marks {image_positions.numel()}.')
        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        block_ids = torch.repeat_interleave(torch.arange(len(block_lengths), device=image_pad_mask.device), torch.tensor(block_lengths, device=image_pad_mask.device))
        image_ids[image_positions] = block_ids
        target_token_mask = torch.zeros_like(image_pad_mask)
        # the last num_target_images blocks are the denoised target (several maps attend jointly, each with its own RoPE block)
        target_token_mask[image_positions[-sum(block_lengths[-num_target_images:]):]] = True
        return (image_ids, target_token_mask)

    def _prepare(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, timestep: torch.Tensor, img_shapes: list[list[tuple[int, int, int]]], img_mask: torch.Tensor, encoder_hidden_states_mask: torch.Tensor | None=None, attention_kwargs: dict[str, Any] | None=None, kv_cache: QwenImage21KVCache | None=None, kv_cache_mode: str | None=None, return_dict: bool=True, target_rope_offset: tuple[int, int]=(0, 0), num_target_images: int=1) -> torch.Tensor | Transformer2DModelOutput:
        if kv_cache_mode == 'cached':
            rotary_emb, attention_mask, target_tokens = kv_cache.layout
            hidden_states = self.img_in(hidden_states[:, -target_tokens:])
            temb = self.time_text_embed(timestep.to(hidden_states.dtype), hidden_states)
            return dict(hidden_states=hidden_states, modulation=self.modulation(temb), rotary_emb=rotary_emb, attention_mask=attention_mask, target_token_mask=None, kv_cache=kv_cache, kv_cache_mode=kv_cache_mode, cache_write_slice=None, segments=None, key_valid=None), temb
        batch_size = hidden_states.shape[0]
        if kv_cache is not None and (not self.config.causal_condition):
            raise ValueError('kv_cache requires `causal_condition=True`. The cache is only valid because text and condition-image tokens modulate from t=0, which makes their activations independent of the denoising step.')
        if kv_cache is not None and kv_cache_mode not in ('extract', 'cached'):
            raise ValueError(f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}.")
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError(f'kv_cache_mode is {kv_cache_mode!r} but no kv_cache was passed to hold the prefix.')
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        target_tokens = sum(math.prod(shape) for shape in img_shapes[0][-num_target_images:])
        joint_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2])], dim=1)
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states
        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device, target_offset=target_rope_offset)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0], num_target_images)
        timestep = timestep.to(hidden_states.dtype)
        if self.config.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)
        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device)
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][:encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]
        prefix_len = int((~target_token_mask).sum())
        if kv_cache_mode == 'cached':
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            rotary_emb = rotary_emb[prefix_len:]
            modulation_mask = modulation_mask[prefix_len:]
            attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
            cache_write_slice = None
            block_segments, block_key_valid = (None, None)
        else:
            attention_mask = None
            block_segments = _qwenimage21_prefix_segments(image_ids, prefix_len)
            cache_write_slice = slice(0, prefix_len) if kv_cache_mode == 'extract' else None
            block_key_valid = joint_key_valid
        if kv_cache_mode == 'extract':
            kv_cache.layout = (rotary_emb[prefix_len:].clone(), None if joint_key_valid is None else joint_key_valid[:, None, None, :], target_tokens)
        return dict(hidden_states=joint_hidden_states, modulation=modulation, rotary_emb=rotary_emb, attention_mask=attention_mask, target_token_mask=modulation_mask, target_start=prefix_len if modulation_mask is not None else None, kv_cache=kv_cache, kv_cache_mode=kv_cache_mode, cache_write_slice=cache_write_slice, segments=block_segments, key_valid=block_key_valid), temb

    def forward(self, hidden_states=None, encoder_hidden_states=None, timestep=None, img_shapes=None, img_mask=None, encoder_hidden_states_mask=None, attention_kwargs=None, kv_cache=None, kv_cache_mode=None, return_dict=True, branches=None, nag_parameters=None, target_rope_offset=(0, 0)):
        inputs = branches if branches is not None else [dict(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, img_shapes=img_shapes, img_mask=img_mask, encoder_hidden_states_mask=encoder_hidden_states_mask, attention_kwargs=attention_kwargs, kv_cache=kv_cache, kv_cache_mode=kv_cache_mode, target_rope_offset=target_rope_offset)]
        negative_state = None
        negative_cache = None
        temporary_negative_cache = False
        if nag_parameters is not None:
            positive, negative = inputs
            negative_cache = negative['kv_cache']
            if negative_cache is None:
                temporary_negative_cache = True
                negative_cache = QwenImage21KVCache(len(self.transformer_blocks))
                negative = {**negative, 'kv_cache': negative_cache, 'kv_cache_mode': 'extract'}
            if negative_cache.layer_caches[0].empty():
                negative_state, _ = self._prepare(**negative)
                prefix = negative_state['cache_write_slice'].stop
                for key in ('hidden_states',):
                    negative_state[key] = negative_state[key][:, :prefix].clone()
                negative_state['rotary_emb'] = negative_state['rotary_emb'][:prefix]
                negative_state['target_token_mask'] = negative_state['target_token_mask'][:prefix]
                if negative_state['key_valid'] is not None:
                    negative_state['key_valid'] = negative_state['key_valid'][:, :prefix]
            inputs = [positive]
        prepared = [self._prepare(**item) for item in inputs]
        # the blocks that store K/V in a cache run eagerly: a capture would store tensors of its own memory
        with cuda_graphs.scope(negative_state is None and all(item['kv_cache_mode'] != 'extract' for item in inputs)):
            for index, block in enumerate(self.transformer_blocks):
                if negative_state is not None:
                    check_abort()
                    negative_args = {key: value for key, value in negative_state.items() if key != 'kv_cache'}
                    negative_args['layer_cache'] = negative_cache.get_layer(index)
                    negative_state['hidden_states'] = block(**negative_args)
                    del negative_args
                for state, temb in prepared:
                    check_abort()
                    cache = state['kv_cache']
                    block_args = {key: value for key, value in state.items() if key != 'kv_cache'}
                    block_args['layer_cache'] = cache.get_layer(index) if cache is not None else None
                    if negative_cache is not None:
                        block_args.update(nag_cache=negative_cache.get_layer(index), nag_parameters=nag_parameters, nag_target_tokens=math.prod(inputs[0]['img_shapes'][0][-1]))
                    state['hidden_states'] = block(**block_args)
                    del block_args
                if temporary_negative_cache:
                    negative_cache.get_layer(index).clear()
        outputs = []
        for state, temb in prepared:
            # Drop state ownership before the disposable normalization call.
            hidden_holder = [state.pop('hidden_states')]
            outputs.append(self.proj_out(self.norm_out(hidden_holder, temb, state.get('target_start'))))
        if branches is not None:
            return outputs
        return Transformer2DModelOutput(sample=outputs[0]) if return_dict else (outputs[0],)
