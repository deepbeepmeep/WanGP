# Copyright (c) 2026 SandAI. All Rights Reserved.
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


import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from shared.attention import pay_attention
from models.magi_human.checkpoint_schema import MODALITY_NAMES, preprocess_magi_lora_state_dict
from inference.common import Modality, VarlenHandler, is_hopper_arch
from magi_compiler import magi_compile
from magi_compiler.api import magi_register_custom_op
from magi_compiler.config import CompileConfig
from torch import Tensor
from torch.nn import Parameter


@dataclass
class FFAHandler:
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: float


class MagiAbortRequested(Exception):
    pass


# Define the MLP activation type
class MLPActivationType(Enum):
    """Enumeration of supported activation functions for MLP"""

    SWIGLU7 = "swiglu7"
    GELU7 = "gelu7"


def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer (from GPT-OSS)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def gelu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu = x
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu.to(out_dtype)


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    match activation_type:
        case MLPActivationType.SWIGLU7:
            return swiglu7
        case MLPActivationType.GELU7:
            return gelu7
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")


class ModalityDispatcher:
    group_size: torch.Tensor
    group_size_cpu: list[int]
    num_modalities: int

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        """
        Initialize dispatcher.
        This runs once during object construction and precomputes all mappings.
        """
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities
        self.group_size = torch.bincount(modality_mapping, minlength=num_modalities).to(torch.int32)
        self.group_size_cpu: list[int] = [int(x) for x in self.group_size.to("cpu").tolist()]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        grouped_tensors = torch.split(x, self.group_size_cpu, dim=0)
        return list(grouped_tensors)

    def undispatch(self, *processed_groups: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(processed_groups, dim=0)


def freq_bands(
    num_bands: int, temperature: float = 10000.0, step: int = 2, device: Optional[torch.device] = None
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exp)
    return bands


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], dim=-1)


class ElementWiseFourierEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
        learnable: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            dim: Output feature dimension, total channels, must be divisible by 6
            max_res: Max pixel-frequency resolution for pixel-domain bands
            temperature: Temperature in inverse-frequency mode
            in_pixels: True -> pixel-frequency bands, False -> inverse-frequency bands
            linear_bands: Whether pixel-frequency bands are linearly spaced
            learnable: Whether frequency bands are trainable
        """
        super().__init__()
        self.dim = dim
        self.in_pixels = in_pixels
        self.learnable = learnable
        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands
        self.device = device
        self.dtype = dtype
        # Make frequency bands trainable or register as buffer
        bands = self.get_default_bands()
        if self.learnable:
            self.bands = nn.Parameter(bands)
        else:
            self.register_buffer("bands", bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [L,9], column order (time, row, col, T, H, W, ref_T, ref_H, ref_W)
        Returns:
            emb: [L, dim] element-wise Fourier embedding
        """
        # Use slicing instead of unbind + stack to reduce intermediates
        coords_xyz = coords[:, :3]  # [L,3] -> (t, h, w)
        sizes = coords[:, 3:6]  # [L,3] -> (T, H, W)
        refs = coords[:, 6:9]  # [L,3] -> (ref_T, ref_H, ref_W)

        # Compute scale factors
        scales = (refs - 1) / (sizes - 1)  # [L,3]

        # NOTE: if both ref and size are 1, scale is fixed to 1; otherwise invalid
        scales[(refs == 1) & (sizes == 1)] = 1
        assert not scales.isnan().any(), "scales has nan"
        assert not scales.isinf().any(), "scales has inf"

        # Center alignment: apply to h,w only (not time)
        centers = (sizes - 1) / 2  # [L,3]
        centers[:, 0] = 0  # Do not center the time dimension
        coords_xyz = coords_xyz - centers  # [L,3]

        # Project to frequency bands in one shot: [L,3,B]
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands

        # Compute sin & cos and concatenate
        sin_proj = proj.sin()  # [L,3,B]
        cos_proj = proj.cos()

        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)

    def reset_parameters(self):
        bands = self.get_default_bands()
        self.bands.copy_(bands)

    def get_default_bands(self):
        if self.in_pixels:
            raise NotImplementedError("in_pixels are not implemented yet")
        else:
            bands = freq_bands(self.dim // 8, temperature=self.temperature, step=1, device=self.device).to(self.dtype)
        return bands


class MultiModalityRMSNorm(nn.Module):
    __constants__ = ["dim", "eps", "num_modality"]
    dim: int
    eps: float
    num_modality: int

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | None = None, num_modality: int = 1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality

        self.weight = torch.nn.Parameter(torch.zeros(dim * num_modality, device=device, dtype=torch.bfloat16))
        if num_modality > 1:
            self.forward = self.forward_multi_experts
        else:
            self.forward = self.forward_single_expert

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward_multi_experts(self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        out = torch.empty_like(x)
        weight = self.weight.view(self.num_modality, self.dim)
        start = 0
        for idx, size in enumerate(modality_dispatcher.group_size_cpu):
            end = start + size
            if size > 0:
                out[start:end] = F.rms_norm(x[start:end], (self.dim,), (weight[idx] + 1).to(x.dtype), self.eps)
            start = end
        return out

    def forward_single_expert(self, x: torch.Tensor, modality_dispatcher: Optional[ModalityDispatcher] = None) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), (self.weight + 1).to(x.dtype), self.eps)


class _BF16ComputeLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        output_dtype: Optional[torch.dtype],
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        # Convert input to specified input data type
        input_cast = input.to(compute_dtype)
        # Convert weight to computation data type
        weight_cast = weight.to(compute_dtype)
        # Perform linear operation
        output = torch.matmul(input_cast, weight_cast.t())

        # Add bias if present
        if bias is not None:
            bias_cast = bias.to(compute_dtype)
            output = output + bias_cast
        else:
            bias_cast = None

        # Convert output to specified output data type
        return output.to(output_dtype)


class BaseLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_layers_for_initialization=1, num_experts=1, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=torch.bfloat16 if dtype is None else dtype)
        self.num_layers_for_initialization = num_layers_for_initialization
        self.num_experts = num_experts


def _prepare_rope_components(rope: torch.Tensor, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    sin, cos = rope.tensor_split(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype, non_blocking=True)
    sin = sin.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype, non_blocking=True)
    return cos.contiguous(), sin.contiguous()


def apply_rope_inplace_(x: torch.Tensor, rope_components: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rope_components
    half_dim = cos.shape[-1]
    x_left = x[..., :half_dim]
    x_right = x[..., half_dim: half_dim * 2]
    x_left_orig = x_left.clone()
    x_left.mul_(cos).addcmul_(x_right, sin, value=-1)
    x_right.mul_(cos).addcmul_(x_left_orig, sin)
    return x


def pay_magi_attention(qkv_list: list[torch.Tensor]) -> torch.Tensor:
    q, k, v = qkv_list
    qkv_list.clear()
    k, v = _expand_kv_heads(q, k, v)
    return pay_attention([q, k, v], recycle_q=True)


def _as_branch_list(value):
    return value if isinstance(value, list) else [value]


def _restore_branches(values, was_list: bool):
    return values if was_list else values[0]


def create_linear(
    in_features, out_features, num_layers=1, num_experts=1, bias=True, device=None, dtype=None
) -> BaseLinear:
    return BaseLinear(in_features, out_features, num_layers, num_experts, bias, device, dtype)


HAS_MAGI_ATTENTION = importlib.util.find_spec("magi_attention") is not None
HAS_FA3 = importlib.util.find_spec("flash_attn_interface") is not None


def _expand_kv_heads(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if query.shape[-2] == key.shape[-2]:
        return key, value
    if query.shape[-2] % key.shape[-2] != 0:
        raise ValueError(
            f"Grouped attention head mismatch: query heads={query.shape[-2]}, key/value heads={key.shape[-2]}"
        )
    repeat_factor = query.shape[-2] // key.shape[-2]
    return key.repeat_interleave(repeat_factor, dim=-2), value.repeat_interleave(repeat_factor, dim=-2)


@magi_register_custom_op(name="infra::flash_attn_func", is_subgraph_boundary=True)
def flash_attn_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    key, value = _expand_kv_heads(query, key, value)
    return pay_attention([query, key, value], force_attention="sdpa")


def _split_q_range_with_no_overlap(
    q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> Tuple[List[List[int]], List[List[List[int]]]]:
    range_boundary = torch.unique(q_ranges, sorted=True).tolist()
    candidates = [[start, end, []] for start, end in zip(range_boundary[:-1], range_boundary[1:])]
    q_ranges = q_ranges.tolist()
    k_ranges = k_ranges.tolist()
    for q_range, k_range in zip(q_ranges, k_ranges):
        q_start, q_end = q_range
        for q_range_cand in candidates:
            if q_start <= q_range_cand[0] and q_range_cand[1] <= q_end:
                q_range_cand[2].append(k_range)
    q_ranges_out = []
    k_ranges_out = []
    for q_range_cand in candidates:
        if len(q_range_cand[2]) > 0:
            q_ranges_out.append(q_range_cand[0:2])
            k_ranges_out.append(q_range_cand[2])
    return q_ranges_out, k_ranges_out


def _flash_attn_with_correction(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: List[List[int]], k_range_list: List[List[List[int]]]
):
    output = torch.zeros_like(query)
    output_lse = torch.zeros((query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device)

    for q_range, k_ranges in zip(q_ranges, k_range_list):
        q_start, q_end = q_range
        q_chunk = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_chunk = key.transpose(0, 1).unsqueeze(0)
        v_chunk = value.transpose(0, 1).unsqueeze(0)
        k_chunk, v_chunk = _expand_kv_heads(q_chunk, k_chunk, v_chunk)
        attn_mask = torch.full(
            (1, 1, q_end - q_start, key.shape[0]),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        for k_start, k_end in k_ranges:
            attn_mask[..., k_start:k_end] = 0.0
        qo_out = torch.nn.functional.scaled_dot_product_attention(
            q_chunk.to(torch.float32),
            k_chunk.to(torch.float32),
            v_chunk.to(torch.float32),
            attn_mask=attn_mask,
            is_causal=False,
        ).squeeze(0).transpose(0, 1).to(query.dtype)
        output[q_start:q_end] = qo_out
    return output, output_lse


def _custom_flex_flash_attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor, **kwargs
):
    q_ranges, k_range_list = _split_q_range_with_no_overlap(q_ranges, k_ranges)
    return _flash_attn_with_correction(query, key, value, q_ranges, k_range_list)


def _flex_flash_attn_func_infer_output_meta(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)
    output_lse = torch.empty((query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device)
    return output, output_lse


@magi_register_custom_op(
    name="infra::flex_flash_attn_func",
    mutates_args=(),
    infer_output_meta_fn=_flex_flash_attn_func_infer_output_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if HAS_MAGI_ATTENTION and is_hopper_arch():
        from magi_attention.api import flex_flash_attn_func as magi_flex_flash_attn_func

        return magi_flex_flash_attn_func(query, key, value, q_ranges, k_ranges)
    else:
        return _custom_flex_flash_attn_func(query, key, value, q_ranges, k_ranges)


def _attention_with_cp_infer_output_meta(q: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return torch.empty_like(q, dtype=torch.bfloat16).squeeze(0)


@magi_register_custom_op(
    name="infra::flash_attn_with_cp",
    mutates_args=(),
    infer_output_meta_fn=_attention_with_cp_infer_output_meta,
    is_subgraph_boundary=True,
)
def flash_attn_with_cp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cp_split_sizes: List[int]) -> torch.Tensor:
    return flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)).squeeze(0)


@magi_register_custom_op(
    name="infra::flex_flash_attn_with_cp",
    mutates_args=(),
    infer_output_meta_fn=_attention_with_cp_infer_output_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_with_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    cp_split_sizes: List[int],
) -> torch.Tensor:
    out, _ = flex_flash_attn_func(
        q.to(torch.bfloat16).squeeze(0),
        k.to(torch.bfloat16).squeeze(0),
        v.to(torch.bfloat16).squeeze(0),
        q_ranges=q_ranges,
        k_ranges=k_ranges,
    )
    return out


@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    params_dtype: torch.dtype
    checkpoint_qk_layernorm_rope: bool
    num_modality: int
    num_layers: int
    use_local_attn: bool = False
    enable_attn_gating: bool = False


class Attention(torch.nn.Module):
    config: AttentionConfig

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.pre_split_qkv = True

        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, eps=1e-6, num_modality=config.num_modality)
        self.gating_size = config.num_heads_q if config.enable_attn_gating else 0

        self.linear_q = self._build_linear("linear_q", config.hidden_size, config.num_heads_q * config.head_dim, bias=False)
        self.linear_k = self._build_linear("linear_k", config.hidden_size, config.num_heads_kv * config.head_dim, bias=False)
        self.linear_v = self._build_linear("linear_v", config.hidden_size, config.num_heads_kv * config.head_dim, bias=False)
        if self.gating_size:
            self.linear_g = self._build_linear("linear_g", config.hidden_size, self.gating_size, bias=False)
        else:
            self.linear_g = None
        self.linear_proj = self._build_linear("linear_proj", config.num_heads_q * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)
        self.k_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)

        self.q_size = config.num_heads_q * config.head_dim
        self.kv_size = config.num_heads_kv * config.head_dim

    def reset_parameters(self):
        pass

    def _build_linear(self, name: str, in_features: int, out_features: int, bias: bool) -> Optional[BaseLinear]:
        if self.config.num_modality == 1:
            linear = create_linear(in_features, out_features, bias=bias, dtype=self.config.params_dtype, num_layers=self.config.num_layers)
            setattr(self, name, linear)
            return linear
        for modality_name in MODALITY_NAMES:
            setattr(self, f"{name}_{modality_name}", create_linear(in_features, out_features, bias=bias, dtype=self.config.params_dtype, num_layers=self.config.num_layers))
        return None

    def _apply_linear(self, linear: BaseLinear, x: torch.Tensor) -> torch.Tensor:
        return linear(x).to(torch.bfloat16)

    def _apply_expert_linear(
        self,
        linear_video: BaseLinear,
        linear_audio: BaseLinear,
        linear_text: BaseLinear,
        out_features: int,
        x: torch.Tensor,
        group_sizes: list[int],
    ) -> torch.Tensor:
        out = torch.empty((x.shape[0], out_features), device=x.device, dtype=torch.bfloat16)
        start = 0
        for expert_linear, group_size in zip((linear_video, linear_audio, linear_text), group_sizes):
            end = start + group_size
            if group_size > 0:
                out.narrow(0, start, group_size).copy_(expert_linear(x.narrow(0, start, group_size)).to(torch.bfloat16))
            start = end
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_components: tuple[torch.Tensor, torch.Tensor],
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        group_sizes = modality_dispatcher.group_size_cpu if self.config.num_modality > 1 else None
        hidden_states = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        if self.config.num_modality == 1:
            q = self._apply_linear(self.linear_q, hidden_states)
            k = self._apply_linear(self.linear_k, hidden_states)
            v = self._apply_linear(self.linear_v, hidden_states)
            g = self._apply_linear(self.linear_g, hidden_states) if self.gating_size else None
        else:
            q = self._apply_expert_linear(self.linear_q_video, self.linear_q_audio, self.linear_q_text, self.q_size, hidden_states, group_sizes)
            k = self._apply_expert_linear(self.linear_k_video, self.linear_k_audio, self.linear_k_text, self.kv_size, hidden_states, group_sizes)
            v = self._apply_expert_linear(self.linear_v_video, self.linear_v_audio, self.linear_v_text, self.kv_size, hidden_states, group_sizes)
            g = self._apply_expert_linear(self.linear_g_video, self.linear_g_audio, self.linear_g_text, self.gating_size, hidden_states, group_sizes) if self.gating_size else None
        hidden_states = None
        q = q.view(-1, self.config.num_heads_q, self.config.head_dim)
        k = k.view(-1, self.config.num_heads_kv, self.config.head_dim)
        v = v.view(-1, self.config.num_heads_kv, self.config.head_dim)
        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)
        if g is not None:
            g = g.view(-1, self.config.num_heads_q, 1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        q = apply_rope_inplace_(q, rope_components)
        k = apply_rope_inplace_(k, rope_components)

        if self.config.use_local_attn:
            self_attn_out = flex_flash_attn_with_cp(
                q, k, v, local_attn_handler.q_ranges, local_attn_handler.k_ranges, cp_split_sizes
            )
        else:
            k, v = _expand_kv_heads(q, k, v)
            qkv_list = [q, k, v]
            q = k = v = None
            self_attn_out = pay_attention(qkv_list, recycle_q=True).squeeze(0)
        if g is not None:
            self_attn_out.mul_(torch.sigmoid(g))
        self_attn_out = self_attn_out.view(-1, self.config.num_heads_q * self.config.head_dim)
        if self.config.num_modality == 1:
            out = self._apply_linear(self.linear_proj, self_attn_out)
        else:
            out = self._apply_expert_linear(self.linear_proj_video, self.linear_proj_audio, self.linear_proj_text, self.config.hidden_size, self_attn_out, group_sizes)
        return out


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(torch.nn.Module):
    config: MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=config.num_modality)
        intermediate_size_up = config.intermediate_size * 2 if config.gated_act else config.intermediate_size

        self.up_gate_proj = self._build_linear("up_gate_proj", config.hidden_size, intermediate_size_up, bias=False)
        self.down_proj = self._build_linear("down_proj", config.intermediate_size, config.hidden_size, bias=False)
        self.activation_func = create_activation_func(config.activation_type)
        self.ffn_expand_ratio = intermediate_size_up / config.hidden_size

    def _build_linear(self, name: str, in_features: int, out_features: int, bias: bool) -> Optional[BaseLinear]:
        if self.config.num_modality == 1:
            linear = create_linear(in_features, out_features, bias=bias, dtype=self.config.params_dtype, num_layers=self.config.num_layers)
            setattr(self, name, linear)
            return linear
        for modality_name in MODALITY_NAMES:
            setattr(self, f"{name}_{modality_name}", create_linear(in_features, out_features, bias=bias, dtype=self.config.params_dtype, num_layers=self.config.num_layers))
        return None

    def _get_chunk_size(self, token_count: int) -> int:
        return max(1, int(token_count / self.ffn_expand_ratio))

    def _run_ffn_inplace(self, x_chunk: torch.Tensor, up_proj: BaseLinear, down_proj: BaseLinear) -> None:
        mlp_chunk = up_proj(x_chunk).to(torch.bfloat16)
        mlp_chunk = self.activation_func(mlp_chunk, out_dtype=torch.bfloat16)
        x_chunk.copy_(down_proj(mlp_chunk).to(torch.bfloat16))
        del mlp_chunk

    def _run_ffn_chunked_inplace(self, x: torch.Tensor, up_proj: BaseLinear, down_proj: BaseLinear) -> None:
        chunk_size = self._get_chunk_size(x.shape[0])
        if chunk_size >= x.shape[0]:
            self._run_ffn_inplace(x, up_proj, down_proj)
            return
        for x_chunk in torch.split(x, chunk_size):
            self._run_ffn_inplace(x_chunk, up_proj, down_proj)

    def forward(self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)
        if self.config.num_modality == 1:
            self._run_ffn_chunked_inplace(x, self.up_gate_proj, self.down_proj)
            return x

        video_size, audio_size, text_size = modality_dispatcher.group_size_cpu
        if video_size > 0:
            self._run_ffn_chunked_inplace(x.narrow(0, 0, video_size), self.up_gate_proj_video, self.down_proj_video)
        if audio_size > 0:
            self._run_ffn_inplace(x.narrow(0, video_size, audio_size), self.up_gate_proj_audio, self.down_proj_audio)
        if text_size > 0:
            self._run_ffn_inplace(x.narrow(0, video_size + audio_size, text_size), self.up_gate_proj_text, self.down_proj_text)
        return x

    def extra_repr(self) -> str:
        return f"{self.config.hidden_size=}, {self.config.intermediate_size=}, {self.config.num_modality=}"


@dataclass
class AdapterConfig:
    hidden_size: int
    num_attention_heads: int
    text_in_channels: int
    video_in_channels: int
    audio_in_channels: int
    params_dtype: torch.dtype


class Adapter(torch.nn.Module):
    config: AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.video_embedder = nn.Linear(config.video_in_channels, config.hidden_size, bias=True, dtype=config.params_dtype)
        self.text_embedder = nn.Linear(config.text_in_channels, config.hidden_size, bias=True, dtype=config.params_dtype)
        self.audio_embedder = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True, dtype=config.params_dtype)
        self.rope = ElementWiseFourierEmbed(config.hidden_size // config.num_attention_heads, in_pixels=False, learnable=False)

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        rope = self.rope(coords_mapping)
        x = x.to(self.config.params_dtype)  # FP32 tokens: the FP32 lock leaves them to this cast, which MMGP made before
        output_x = torch.zeros(x.shape[0], self.config.hidden_size, device=x.device, dtype=self.config.params_dtype)
        output_x[text_mask] = self.text_embedder(x[text_mask, : self.config.text_in_channels])
        output_x[audio_mask] = self.audio_embedder(x[audio_mask, : self.config.audio_in_channels])
        output_x[video_mask] = self.video_embedder(x[video_mask, : self.config.video_in_channels])
        return output_x, rope


class TransFormerLayer(torch.nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        num_modality = 3 if layer_idx in config.mm_layers else 1
        use_local_attn = layer_idx in config.local_attn_layers
        self.post_norm = layer_idx in config.post_norm_layers
        attention_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads_q=config.num_heads_q,
            num_heads_kv=config.num_heads_kv,
            head_dim=config.head_dim,
            params_dtype=config.params_dtype,
            checkpoint_qk_layernorm_rope=config.checkpoint_qk_layernorm_rope,
            num_modality=num_modality,
            num_layers=config.num_layers,
            use_local_attn=use_local_attn,
            enable_attn_gating=config.enable_attn_gating,
        )
        self.attention: Attention = Attention(attention_config)

        activation_type = MLPActivationType.GELU7 if layer_idx in config.gelu7_layers else MLPActivationType.SWIGLU7
        if activation_type == MLPActivationType.SWIGLU7:
            gated_act = True
            intermediate_size = int(config.hidden_size * 4 * 2 / 3) // 4 * 4
        else:
            gated_act = False
            intermediate_size = config.hidden_size * 4
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            params_dtype=config.params_dtype,
            num_modality=num_modality,
            num_layers=config.num_layers,
            gated_act=gated_act,
        )
        self.mlp: MLP = MLP(mlp_config)
        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)
            self.mlp_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        residual = hidden_states
        attn_out = self.attention(
            residual,
            rope,
            varlen_handler,
            local_attn_handler,
            modality_dispatcher,
            cp_split_sizes,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        residual.add_(attn_out)
        attn_out = None
        mlp_out = self.mlp(residual, modality_dispatcher)
        if self.post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        residual.add_(mlp_out)
        return residual


is_base_model = True


def config_patch(compile_config: CompileConfig) -> CompileConfig:
    global is_base_model
    if is_base_model:
        is_base_model = False
    else:
        # Fully offload SR model for memory-constrained GPU
        compile_config.offload_config.gpu_resident_weight_ratio = 0.0
    return compile_config


@magi_compile(config_patch=config_patch)
class TransformerBlock(torch.nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.layers: list[TransFormerLayer] = nn.ModuleList()
        for layer_idx in range(model_config.num_layers):
            self.layers.append(TransFormerLayer(model_config, layer_idx))

    def forward(
        self,
        x_list,
        rope_list,
        varlen_handler_list,
        local_attn_handler_list,
        modality_dispatcher_list,
        cp_split_sizes_list,
    ):
        interrupt_check = getattr(self, "_interrupt_check", None)
        branch_count = len(x_list)
        for layer_idx, layer in enumerate(self.layers):
            if interrupt_check is not None and interrupt_check():
                raise MagiAbortRequested(f"Magi Human generation aborted before transformer layer {layer_idx}.")
            for branch_idx in range(branch_count):
                x_list[branch_idx] = layer(
                    x_list[branch_idx],
                    rope_list[branch_idx],
                    varlen_handler_list[branch_idx],
                    local_attn_handler_list[branch_idx],
                    modality_dispatcher_list[branch_idx],
                    cp_split_sizes_list[branch_idx],
                )
            if interrupt_check is not None and interrupt_check():
                raise MagiAbortRequested(f"Magi Human generation aborted during transformer layer {layer_idx}.")
        return x_list


@dataclass
class TransformerConfig:
    hidden_size: int
    video_in_channels: int
    audio_in_channels: int
    text_in_channels: int
    params_dtype: torch.dtype
    post_process_dtype: torch.dtype


class DiTModel(torch.nn.Module):
    config: TransformerConfig

    def preprocess_loras(self, model_type, sd):
        return preprocess_magi_lora_state_dict(sd)

    def __init__(self, model_config: Any):
        super().__init__()
        self.config = TransformerConfig(
            hidden_size=model_config.hidden_size,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            text_in_channels=model_config.text_in_channels,
            params_dtype=model_config.params_dtype,
            post_process_dtype=torch.float32,
        )
        adapter_config = AdapterConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_heads_q,
            text_in_channels=model_config.text_in_channels,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            params_dtype=model_config.params_dtype,
        )
        self.adapter: Adapter = Adapter(adapter_config)
        self.block: TransformerBlock = TransformerBlock(model_config=model_config)
        self.final_norm_video = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_linear_video = nn.Linear(
            self.config.hidden_size, self.config.video_in_channels, bias=False, dtype=model_config.params_dtype
        )
        self.final_linear_audio = nn.Linear(
            self.config.hidden_size, self.config.audio_in_channels, bias=False, dtype=model_config.params_dtype
        )

    def forward(
        self,
        x,
        coords_mapping,
        modality_mapping,
        varlen_handler,
        local_attn_handler,
    ):
        was_list = isinstance(x, list)
        x_list = _as_branch_list(x)
        coords_mapping_list = _as_branch_list(coords_mapping)
        modality_mapping_list = _as_branch_list(modality_mapping)
        varlen_handler_list = _as_branch_list(varlen_handler)
        local_attn_handler_list = _as_branch_list(local_attn_handler)
        rope_list = []
        modality_dispatcher_list = []
        cp_split_sizes_list = []
        video_mask_list = []
        audio_mask_list = []
        for branch_idx, (x_branch, coords_branch, modality_branch) in enumerate(zip(x_list, coords_mapping_list, modality_mapping_list)):
            cp_split_sizes_list.append([x_branch.shape[0]])
            modality_dispatcher = ModalityDispatcher(modality_branch, 3)
            video_mask = modality_branch == Modality.VIDEO
            audio_mask = modality_branch == Modality.AUDIO
            text_mask = modality_branch == Modality.TEXT
            x_branch, rope = self.adapter(x_branch, coords_branch, video_mask, audio_mask, text_mask)
            x_list[branch_idx] = x_branch
            rope_list.append(_prepare_rope_components(rope, x_branch.device, self.config.params_dtype))
            modality_dispatcher_list.append(modality_dispatcher)
            video_mask_list.append(video_mask)
            audio_mask_list.append(audio_mask)

        x_out_list = self.block(
            x_list,
            rope_list,
            varlen_handler_list,
            local_attn_handler_list,
            modality_dispatcher_list,
            cp_split_sizes_list,
        )

        outputs = []
        for x_branch, video_mask, audio_mask in zip(x_out_list, video_mask_list, audio_mask_list):
            x_video = x_branch[video_mask].to(self.final_norm_video.weight.dtype)
            x_video = self.final_norm_video(x_video)
            x_video = self.final_linear_video(x_video).to(self.config.post_process_dtype)

            x_audio = x_branch[audio_mask].to(self.final_norm_audio.weight.dtype)
            x_audio = self.final_norm_audio(x_audio)
            x_audio = self.final_linear_audio(x_audio).to(self.config.post_process_dtype)

            x_out = torch.zeros(
                x_branch.shape[0],
                max(self.config.video_in_channels, self.config.audio_in_channels),
                device=x_branch.device,
                dtype=self.config.post_process_dtype,
            )
            x_out[video_mask, : self.config.video_in_channels] = x_video
            x_out[audio_mask, : self.config.audio_in_channels] = x_audio
            outputs.append(x_out)
        return _restore_branches(outputs, was_list)
