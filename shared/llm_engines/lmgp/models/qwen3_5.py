from __future__ import annotations

import logging
from copy import copy
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

logging.getLogger("fla.utils").setLevel(logging.ERROR)

try:
    from flash_attn import flash_attn_varlen_func
except Exception:  # pragma: no cover
    flash_attn_varlen_func = None

try:
    # from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    causal_conv1d_fn = None
    causal_conv1d_update = None
except Exception:  # pragma: no cover
    causal_conv1d_fn = None
    causal_conv1d_update = None

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fast_chunk_gated_delta_rule
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fast_recurrent_gated_delta_rule
except Exception:  # pragma: no cover
    fast_chunk_gated_delta_rule = None
    fast_recurrent_gated_delta_rule = None

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.modules import fused_norm_gate as _fla_fused_norm_gate
except Exception:  # pragma: no cover
    FusedRMSNormGated = None
    ShortConvolution = None
    _fla_fused_norm_gate = None

_DEFAULT_FLASH_ATTN_VARLEN_FUNC = flash_attn_varlen_func
_DEFAULT_FAST_CHUNK_GATED_DELTA_RULE = fast_chunk_gated_delta_rule
_DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE = fast_recurrent_gated_delta_rule
_DEFAULT_FUSED_RMSNORM_GATED = FusedRMSNormGated
_DEFAULT_SHORT_CONVOLUTION = ShortConvolution
_FLA_PREFILL_AUTOTUNE_CONFIGURED = False


def configure_qwen35_fla_prefill_autotune(device: torch.device) -> None:
    """Configure the shared FLA inference kernels before the first forward."""
    global _FLA_PREFILL_AUTOTUNE_CONFIGURED
    if _FLA_PREFILL_AUTOTUNE_CONFIGURED or _DEFAULT_FUSED_RMSNORM_GATED is None:
        return
    from shared.kernels.fla_inference import configure_fla_inference_kernels
    configure_fla_inference_kernels()
    _FLA_PREFILL_AUTOTUNE_CONFIGURED = True


def configure_qwen35_safe_legacy_kernels(enabled: bool) -> None:
    global flash_attn_varlen_func, fast_chunk_gated_delta_rule, fast_recurrent_gated_delta_rule
    global FusedRMSNormGated, ShortConvolution
    if enabled:
        flash_attn_varlen_func = None
        fast_chunk_gated_delta_rule = None
        fast_recurrent_gated_delta_rule = None
        FusedRMSNormGated = None
        ShortConvolution = None
        return
    flash_attn_varlen_func = _DEFAULT_FLASH_ATTN_VARLEN_FUNC
    fast_chunk_gated_delta_rule = _DEFAULT_FAST_CHUNK_GATED_DELTA_RULE
    fast_recurrent_gated_delta_rule = _DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE
    FusedRMSNormGated = _DEFAULT_FUSED_RMSNORM_GATED
    ShortConvolution = _DEFAULT_SHORT_CONVOLUTION

from ..layers.activation import SiluAndMul
from ..layers.attention import Attention, _efficient_causal_attention
from shared.kernels.dispatch import capability, choose, triton_dtype_supported
try:
    from shared.kernels import static_cache_attention as _static_cache_attention
except (ImportError, OSError, RuntimeError):
    _static_cache_attention = None
from ..layers.layernorm import RMSNorm
from shared.kernels import bf16_bits
from ..layers.linear import ColumnParallelLinear, RowParallelLinear
from ..utils.context import get_context
from mmgp import cuda_graphs

_CU_SEQLENS_CACHE: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}
_MROPE_FREQ_CACHE: dict[tuple[str, int, int, float], torch.Tensor] = {}


def _triton_kernels(config) -> bool:
    """The LM features' Triton switch (off with WGP_LLM_DISABLE=triton, or for a model keeping its encoder on PyTorch). Each kernel then
    decides per call from its tensors' GPU and dtype: native Triton, BF16 read as bits on Turing (shared.kernels.bf16_bits), or PyTorch."""
    return not getattr(config, "_prompt_enhancer_safe_legacy", False)


def _slow_bf16_gemm(x: torch.Tensor) -> bool:
    """A BF16 GEMM of a few rows on an NVIDIA GPU without BF16 tensor cores (before Ampere) runs on a slow SIMT kernel."""
    found = capability(x.device.index) if x.is_cuda else None
    return x.dtype == torch.bfloat16 and found is not None and found < (8, 0)




def _get_tp_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _build_cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    normalized_lengths = tuple(int(length) for length in lengths)
    cache_key = (normalized_lengths, str(device))
    cached = _CU_SEQLENS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    total = 0
    out = [0]
    for length in normalized_lengths:
        total += length
        out.append(total)
    cached = torch.tensor(out, dtype=torch.int32, device=device)
    _CU_SEQLENS_CACHE[cache_key] = cached
    return cached


def _build_mrope_freq_cache(
    device: torch.device,
    max_position: int,
    rotary_dim: int,
    rope_theta: float,
) -> torch.Tensor:
    cache_key = (str(device), int(max_position), int(rotary_dim), float(rope_theta))
    cached = _MROPE_FREQ_CACHE.get(cache_key)
    if cached is not None:
        return cached

    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    cached = torch.einsum("i,j->ij", positions, inv_freq)
    _MROPE_FREQ_CACHE[cache_key] = cached
    return cached


def _interleave_axis_halves(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or tensor.shape[dim] % 2 != 0:
        return tensor
    first_half, second_half = tensor.chunk(2, dim=dim)
    return torch.stack((first_half, second_half), dim=dim + 1).flatten(dim, dim + 1)


def _inverse_interleave_axis_halves(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or tensor.shape[dim] % 2 != 0:
        return tensor
    half = tensor.shape[dim] // 2
    tensor = tensor.reshape(*tensor.shape[:dim], half, 2, *tensor.shape[dim + 1 :])
    return torch.cat((tensor.select(dim + 1, 0), tensor.select(dim + 1, 1)), dim=dim)


def _reorder_v_heads_grouped_to_tiled(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        return tensor
    if tensor.shape[dim] != num_v_heads * head_dim:
        return tensor
    num_v_per_k = num_v_heads // num_k_heads
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).reshape(*shape)


def _reorder_v_heads_tiled_to_grouped(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        return tensor
    if tensor.shape[dim] != num_v_heads * head_dim:
        return tensor
    num_v_per_k = num_v_heads // num_k_heads
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).reshape(*shape)


def _reorder_v_head_axis_grouped_to_tiled(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        return tensor
    if tensor.shape[dim] != num_v_heads:
        return tensor
    num_v_per_k = num_v_heads // num_k_heads
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).reshape(*shape)


def _reorder_v_head_axis_tiled_to_grouped(
    tensor: torch.Tensor,
    dim: int,
    num_k_heads: int,
    num_v_heads: int,
) -> torch.Tensor:
    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim or num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
        return tensor
    if tensor.shape[dim] != num_v_heads:
        return tensor
    num_v_per_k = num_v_heads // num_k_heads
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads] + shape[dim + 1 :]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).reshape(*shape)


def _maybe_reorder_gguf_ssm_param(
    tensor: torch.Tensor,
    *,
    interleave_halves: bool,
    tiled_to_grouped: bool,
    num_k_heads: int,
    num_v_heads: int,
) -> torch.Tensor:
    if tiled_to_grouped:
        return _reorder_v_heads_tiled_to_grouped(tensor, dim=0, num_k_heads=num_k_heads, num_v_heads=num_v_heads, head_dim=1)
    if interleave_halves:
        return _interleave_axis_halves(tensor, dim=0)
    return tensor


def clear_qwen35_runtime_caches(device: torch.device | None = None) -> None:
    if device is None:
        _CU_SEQLENS_CACHE.clear()
        _MROPE_FREQ_CACHE.clear()
        return

    device_key = str(device)
    for cache_key in [key for key in _CU_SEQLENS_CACHE if key[1] == device_key]:
        _CU_SEQLENS_CACHE.pop(cache_key, None)
    for cache_key in [key for key in _MROPE_FREQ_CACHE if key[0] == device_key]:
        _MROPE_FREQ_CACHE.pop(cache_key, None)


def apply_rotary_pos_emb(
    qk_list: list[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k = qk_list
    qk_list.clear()
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    half = rotary_dim // 2
    for tensor in (q, k):
        first = tensor[..., :half]
        second = tensor[..., half:rotary_dim]
        scratch = torch.empty_like(first)
        scratch.copy_(first)
        first.mul_(cos[..., :half]).sub_(second * sin[..., :half])
        second.mul_(cos[..., half:]).add_(scratch * sin[..., half:])
        scratch = None
    return q, k


def _take_tensor(x_list: list[torch.Tensor]) -> torch.Tensor:
    x = x_list[0]
    x_list.clear()
    return x


_AB_ROWS_GEMV_MAX = 16  # verification passes (at most 9 rows) and small batches; prefill chunks keep the GEMM


def _rows_linear(module, hidden_states):
    """module(hidden_states) one row at a time. On an RTX 2080 Ti the 48 dense BF16 alpha/beta projections of the 27B take 17.6 ms per
    5-row verification as one GEMM and 1.5 ms row by row; each row runs the GEMV of single-token decoding, with the same results."""
    rows = hidden_states.reshape(-1, hidden_states.shape[-1])
    return torch.cat([module(row) for row in rows.split(1)]).reshape(*hidden_states.shape[:-1], -1)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias=bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key)]
    value, beta = [x.transpose(1, 2).contiguous() for x in (value, beta)]
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size
    query = (query * (query.shape[-1] ** -0.5)).reshape(query.shape[0], query.shape[1], -1, chunk_size, query.shape[-1])
    key = key.reshape(key.shape[0], key.shape[1], -1, chunk_size, key.shape[-1])
    value = value.reshape(value.shape[0], value.shape[1], -1, chunk_size, value.shape[-1])
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(dim=-1)
    lower_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    upper_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    outputs = torch.empty(batch_size, num_heads, num_chunks, chunk_size, v_head_dim, device=value.device, dtype=initial_dtype)
    eye = torch.eye(chunk_size, dtype=torch.float32, device=query.device)

    for idx in range(num_chunks):
        q_i = query[:, :, idx]
        k_i = key[:, :, idx]
        value_i = value[:, :, idx].to(torch.float32)
        beta_i = beta[:, :, idx].to(torch.float32)
        g_i = g[:, :, idx]
        beta_i_unsqueezed = beta_i.unsqueeze(-1)
        g_exp_i = g_i.exp().unsqueeze(-1)
        k_beta_i = k_i * beta_i_unsqueezed
        decay_mask_i = ((g_i.unsqueeze(-1) - g_i.unsqueeze(-2)).tril().exp()).tril()
        lower = k_beta_i @ k_i.transpose(-1, -2)
        lower.mul_(decay_mask_i)
        lower.masked_fill_(lower_mask, 0)
        lower.add_(eye)
        value_i.mul_(beta_i_unsqueezed)
        value_i = torch.linalg.solve_triangular(lower, value_i, upper=False, left=True)
        k_beta_i.mul_(g_exp_i)
        k_cumdecay_i = torch.linalg.solve_triangular(lower, k_beta_i, upper=False, left=True)
        attn_i = q_i @ k_i.transpose(-1, -2)
        attn_i.mul_(decay_mask_i)
        attn_i.masked_fill_(upper_mask, 0)
        value_i.sub_(k_cumdecay_i @ recurrent_state)
        outputs[:, :, idx] = (((q_i * g_exp_i) @ recurrent_state) + attn_i @ value_i).to(initial_dtype)
        recurrent_state.mul_(g_i[:, :, -1, None, None].exp())
        recurrent_state.add_((k_i * (g_i[:, :, -1, None] - g_i).exp().unsqueeze(-1)).transpose(-1, -2) @ value_i)

    outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], -1, outputs.shape[-1])
    outputs = outputs[:, :, :sequence_length]
    if not output_final_state:
        recurrent_state = None
    return outputs.transpose(1, 2).contiguous(), recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key, value, beta = [x.transpose(1, 2).contiguous() for x in (query, key, value, beta)]
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = query.shape[-1] ** -0.5
    outputs = torch.empty(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=initial_dtype)
    recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32)
    )

    for idx in range(sequence_length):
        q_t = query[:, :, idx].to(torch.float32)
        k_t = key[:, :, idx].to(torch.float32)
        v_t = value[:, :, idx].to(torch.float32)
        g_t = g[:, :, idx]
        beta_t = beta[:, :, idx].to(torch.float32)
        q_t = q_t * scale
        g_t = g_t.exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_t.unsqueeze(-1)
        recurrent_state = recurrent_state * g_t
        kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        recurrent_state = recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs[:, :, idx] = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2).to(initial_dtype)

    if not output_final_state:
        recurrent_state = None
    return outputs.transpose(1, 2).contiguous(), recurrent_state


class Qwen3_5DynamicCache:
    def __init__(self, config):
        self.layer_types = list(config.layer_types)
        self.transformer_layers = [idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]
        self.last_linear_layer = max((idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"), default=-1)
        self.conv_states = [None for _ in range(int(config.num_hidden_layers))]
        self.recurrent_states = [None for _ in range(int(config.num_hidden_layers))]
        self.key_cache = [None for _ in range(int(config.num_hidden_layers))]
        self.value_cache = [None for _ in range(int(config.num_hidden_layers))]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if not self.transformer_layers:
            return 0
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        cache = self.key_cache[layer_idx]
        if cache is None:
            return 0
        return int(cache.shape[-2])

    @property
    def has_previous_state(self) -> bool:
        return self.last_linear_layer >= 0 and self.conv_states[self.last_linear_layer] is not None


class Qwen3_5StaticCache(Qwen3_5DynamicCache):
    """Fixed-capacity cache used by the single-sequence MTP decoder."""

    def __init__(self, config, max_cache_len: int, device: torch.device, dtype: torch.dtype):
        super().__init__(config)
        self.max_cache_len = int(max_cache_len)
        self._seq_length = 0
        num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        for layer_idx in self.transformer_layers:  # zeros: the slots never written stay finite, so that a mask hides them (graph_attention)
            self.key_cache[layer_idx] = torch.zeros((1, self.max_cache_len, num_kv_heads, head_dim), device=device, dtype=dtype)
            self.value_cache[layer_idx] = torch.zeros((1, self.max_cache_len, num_kv_heads, head_dim), device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(1, dtype=torch.int32, device=device)
        self.cache_positions = torch.arange(self.max_cache_len, device=device)
        self.triton_kernels = not getattr(config, "_prompt_enhancer_safe_legacy", False)  # the legacy and cg engines use no Triton

    def graph_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layer_idx: int, scaling: float) -> torch.Tensor:
        """Decoding attention with GPU cache positions and fixed shapes for CUDA graph replay, on GPUs without FlashAttention."""
        positions = self.cache_seqlens + self.cache_positions[:query.shape[1]]
        key_cache, value_cache = self.key_cache[layer_idx], self.value_cache[layer_idx]
        key_cache.index_copy_(1, positions, key)
        value_cache.index_copy_(1, positions, value)
        if self.triton_kernels and _static_cache_attention is not None and choose("Static Cache Attention", owned_triton=_static_cache_attention.usable(query, key_cache)):
            # reads each row's keys up to its own position only, in FP32: no copy of the cache, no score matrix
            return _static_cache_attention.attention(query.contiguous(), key_cache, value_cache, self.cache_seqlens, scaling)
        # the slots past the written ones are zeros (finite), which the mask hides
        key, value = key_cache.transpose(1, 2), value_cache.transpose(1, 2)
        repeats = query.shape[2] // key.shape[1]
        mask = self.cache_positions[None, :] <= positions[:, None]
        # Fold query heads into rows within each KV head instead of duplicating the full cache for GQA.
        batch, count, heads, dim = query.shape
        query = query.transpose(1, 2).reshape(batch, key.shape[1], repeats * count, dim)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask.repeat(repeats, 1), dropout_p=0.0, scale=scaling)
        return output.reshape(batch, heads, count, dim).transpose(1, 2)

    def prepare_append(self) -> None:
        self.cache_seqlens.fill_(self._seq_length)

    def advance(self, token_count: int) -> None:
        end = self._seq_length + int(token_count)
        if end > self.max_cache_len:
            raise RuntimeError(f"MTP cache capacity exceeded ({end} > {self.max_cache_len}).")
        self._seq_length = end

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = int(key_states.shape[2])
        end = self._seq_length + token_count
        if end > self.max_cache_len:
            raise RuntimeError(f"MTP cache capacity exceeded ({end} > {self.max_cache_len}).")
        self.key_cache[layer_idx][:, self._seq_length:end].copy_(key_states.transpose(1, 2))
        self.value_cache[layer_idx][:, self._seq_length:end].copy_(value_states.transpose(1, 2))
        self._seq_length = end
        return self.key_cache[layer_idx][:, :end].transpose(1, 2), self.value_cache[layer_idx][:, :end].transpose(1, 2)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self._seq_length

    def reset(self) -> None:
        self._seq_length = 0

    def truncate(self, seq_length: int) -> None:
        seq_length = int(seq_length)
        if not 0 <= seq_length <= self._seq_length:
            raise RuntimeError(f"Cannot truncate MTP cache from {self._seq_length} to {seq_length} tokens.")
        self._seq_length = seq_length

    def snapshot(self, previous: dict | None = None, reuse_tokens: int = 0) -> dict:
        from shared.llm_engines.snapshot_cache import snapshot_cache

        reuse = 0 if previous is None else min(reuse_tokens, previous["seq_length"], self._seq_length)
        return {
            "seq_length": self._seq_length,
            "key_cache": [None if cache is None else snapshot_cache(cache, [(0, self._seq_length)], axis=1, previous=None if previous is None else previous["key_cache"][index], reuse=reuse) for index, cache in enumerate(self.key_cache)],
            "value_cache": [None if cache is None else snapshot_cache(cache, [(0, self._seq_length)], axis=1, previous=None if previous is None else previous["value_cache"][index], reuse=reuse) for index, cache in enumerate(self.value_cache)],
        }

    def restore(self, snapshot: dict) -> None:
        from shared.llm_engines.snapshot_cache import restore_cache

        seq_length = int(snapshot["seq_length"])
        if seq_length > self.max_cache_len:
            raise RuntimeError(f"Saved MTP cache exceeds live capacity ({seq_length} > {self.max_cache_len}).")
        for live_key, live_value, saved_key, saved_value in zip(self.key_cache, self.value_cache, snapshot["key_cache"], snapshot["value_cache"]):
            if live_key is None:
                if saved_key is not None or saved_value is not None:
                    raise RuntimeError("Saved MTP cache layout does not match the live model.")
                continue
            if saved_key is None or saved_value is None:
                raise RuntimeError("Saved MTP cache layout does not match the live model.")
            restore_cache(live_key, saved_key)
            restore_cache(live_value, saved_value)
        self._seq_length = seq_length


class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        rotary_dim = int(head_dim * float(rope_parameters.get("partial_rotary_factor", 1.0)))
        rope_theta = float(rope_parameters.get("rope_theta", 1000000))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu") / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = list(rope_parameters.get("mrope_section", [11, 11, 10]))
        self.max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim

    def prepare_sequence_state(self, max_batch_size: int, device: torch.device, dtype: torch.dtype):
        # the table the decoding graphs read, made before their capture and held as engine state while the runtime lives
        self._freq_cache = _build_mrope_freq_cache(device, max_position=self.max_position_embeddings, rotary_dim=self.rotary_dim, rope_theta=self.rope_theta)

    def release_sequence_state(self):
        self._freq_cache = None

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        max_position = self.max_position_embeddings
        if not (position_ids.is_cuda and torch.cuda.is_current_stream_capturing()):
            max_position = max(max_position, int(position_ids.max().item()) + 1)
        freq_cache = _build_mrope_freq_cache(
            position_ids.device,
            max_position=max_position,
            rotary_dim=self.rotary_dim,
            rope_theta=self.rope_theta,
        )
        freqs = torch.stack([freq_cache[position_ids[axis].long()] for axis in range(3)], dim=0)
        freqs = self.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_triton = False  # the LM features' Triton switch; forward_list picks FLA's kernel, the bits one or PyTorch per call

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        hidden_states = hidden_states * self.weight.float()
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(dtype=input_dtype)

    def forward_list(self, state_list: list[torch.Tensor]) -> torch.Tensor:
        hidden_states, gate = state_list
        state_list.clear()
        if self.use_triton and hidden_states.is_cuda and gate.dtype == hidden_states.dtype:
            if _fla_fused_norm_gate is not None and triton_dtype_supported(hidden_states.device, hidden_states.dtype):
                return _fla_fused_norm_gate.rms_norm_gated(hidden_states, gate, self.weight, None, "swish", eps=self.eps)
            if bf16_bits.applies(hidden_states):
                return bf16_bits.gated_rms_norm(hidden_states, gate, self.weight, self.eps)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states.mul_(torch.rsqrt(hidden_states.square().mean(-1, keepdim=True).add_(self.eps)))
        hidden_states.mul_(self.weight.float())
        gate = F.silu(gate.float(), inplace=True)
        hidden_states.mul_(gate)
        return hidden_states.to(dtype=input_dtype)


def _forward_gated_norm_list(norm: nn.Module, state_list: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(norm, Qwen3_5RMSNormGated):
        return norm.forward_list(state_list)
    hidden_states, gate = state_list
    state_list.clear()
    return norm(hidden_states, gate)


def _repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(num_repeats, dim=1)


def _flash_attention(
    qkv_list: list[torch.Tensor],
    query_lengths: list[int],
    key_lengths: list[int],
    scaling: float,
    flash_attention_fn,
    efficient: bool = False,
) -> torch.Tensor:
    query_states, key_states, value_states = qkv_list
    qkv_list.clear()
    if flash_attention_fn is not None and query_states.is_cuda:
        q_chunks = [query_states[idx, : q_len] for idx, q_len in enumerate(query_lengths) if q_len > 0]
        k_chunks = [key_states[idx, : k_len] for idx, k_len in enumerate(key_lengths) if k_len > 0]
        v_chunks = [value_states[idx, : k_len] for idx, k_len in enumerate(key_lengths) if k_len > 0]
        q_flat = torch.cat(q_chunks, dim=0)
        k_flat = torch.cat(k_chunks, dim=0)
        v_flat = torch.cat(v_chunks, dim=0)
        cu_q = _build_cu_seqlens(query_lengths, query_states.device)
        cu_k = _build_cu_seqlens(key_lengths, query_states.device)
        out_flat = flash_attention_fn(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(query_lengths),
            max_seqlen_k=max(key_lengths),
            softmax_scale=scaling,
            causal=True,
        )
        out = torch.zeros_like(query_states)
        offset = 0
        for idx, q_len in enumerate(query_lengths):
            if q_len <= 0:
                continue
            out[idx, :q_len] = out_flat[offset : offset + q_len]
            offset += q_len
        return out

    outputs = []
    for idx, q_len in enumerate(query_lengths):
        q = query_states[idx, :q_len].transpose(0, 1)
        k = key_states[idx, : key_lengths[idx]].transpose(0, 1)
        v = value_states[idx, : key_lengths[idx]].transpose(0, 1)
        if efficient:
            outputs.append(_efficient_causal_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scaling).squeeze(0).transpose(0, 1))
            continue
        k = _repeat_kv(k.unsqueeze(0), q.shape[0] // k.shape[0]).squeeze(0)
        v = _repeat_kv(v.unsqueeze(0), q.shape[0] // v.shape[0]).squeeze(0)
        attention_mask = None
        is_causal = True
        if q_len != key_lengths[idx]:
            query_positions = torch.arange(q_len, device=q.device) + key_lengths[idx] - q_len
            key_positions = torch.arange(key_lengths[idx], device=q.device)
            attention_mask = query_positions[:, None] >= key_positions[None, :]
            is_causal = False
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scaling,
        ).squeeze(0).transpose(0, 1)
        outputs.append(out)
    padded = torch.zeros_like(query_states)
    for idx, out in enumerate(outputs):
        padded[idx, : out.shape[0]] = out
    return padded


class Qwen3_5Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        triton_kernels = _triton_kernels(config)
        self.layer_type = str(config.layer_types[layer_idx])
        self.attn_norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.post_attention_norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.attn_norm.use_triton_rmsnorm = triton_kernels
        self.post_attention_norm.use_triton_rmsnorm = triton_kernels
        self.ffn_gate = ColumnParallelLinear(int(config.hidden_size), int(config.intermediate_size), bias=False)
        self.ffn_up = ColumnParallelLinear(int(config.hidden_size), int(config.intermediate_size), bias=False)
        self.ffn_gate_up = None
        self.ffn_down = RowParallelLinear(int(config.intermediate_size), int(config.hidden_size), bias=False)
        self.mlp_act_fn = SiluAndMul(use_triton=triton_kernels)

        if self.layer_type == "full_attention":
            tp_size = _get_tp_size()
            hidden_size = int(config.hidden_size)
            total_num_heads = int(config.num_attention_heads)
            total_num_kv_heads = int(config.num_key_value_heads)
            assert total_num_heads % tp_size == 0
            assert total_num_kv_heads % tp_size == 0
            self.num_heads = total_num_heads // tp_size
            self.num_kv_heads = total_num_kv_heads // tp_size
            self.head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))
            self.num_key_value_groups = self.num_heads // self.num_kv_heads
            self.scaling = self.head_dim**-0.5
            self.attn_q = ColumnParallelLinear(
                hidden_size,
                total_num_heads * self.head_dim * 2,
                bias=bool(config.attention_bias),
            )
            self.attn_k = ColumnParallelLinear(
                hidden_size,
                total_num_kv_heads * self.head_dim,
                bias=bool(config.attention_bias),
            )
            self.attn_v = ColumnParallelLinear(
                hidden_size,
                total_num_kv_heads * self.head_dim,
                bias=bool(config.attention_bias),
            )
            self.attn_kv = None
            self.attn_qkv_full = None
            self.attn_output = RowParallelLinear(
                total_num_heads * self.head_dim,
                hidden_size,
                bias=bool(config.attention_bias),
            )
            self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
            # FlashAttention goes with the Triton kernels unless the loader decides it from what is installed
            flash_attention = getattr(config, "_prompt_enhancer_flash_attention", triton_kernels)
            if not flash_attention:
                self.attn.flash_attn_varlen_func = None
                self.attn.flash_attn_with_kvcache = None
            if not triton_kernels:
                self.attn.use_triton_kv_cache = False
            self.attn_q_norm = RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
            self.attn_k_norm = RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
            self.attn_q_norm.use_triton_rmsnorm = triton_kernels
            self.attn_k_norm.use_triton_rmsnorm = triton_kernels
            self._flash_attn_varlen_func = _DEFAULT_FLASH_ATTN_VARLEN_FUNC if flash_attention else None
        else:
            # FLA's Triton kernels where the Triton switch allows; each forward then runs them only for dtypes its GPU converts
            self._short_convolution_cls = _DEFAULT_SHORT_CONVOLUTION if triton_kernels else None
            self._fast_recurrent_gated_delta_rule = _DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE if triton_kernels else None
            self._fast_chunk_gated_delta_rule = _DEFAULT_FAST_CHUNK_GATED_DELTA_RULE if triton_kernels else None
            self.num_v_heads = int(config.linear_num_value_heads)
            self.num_k_heads = int(config.linear_num_key_heads)
            self.head_k_dim = int(config.linear_key_head_dim)
            self.head_v_dim = int(config.linear_value_head_dim)
            self.key_dim = self.head_k_dim * self.num_k_heads
            self.value_dim = self.head_v_dim * self.num_v_heads
            self.conv_kernel_size = int(config.linear_conv_kernel_dim)
            hidden_size = int(config.hidden_size)
            self.attn_qkv = ColumnParallelLinear(hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
            self.attn_gate = ColumnParallelLinear(hidden_size, self.value_dim, bias=False)
            self.ssm_alpha = ColumnParallelLinear(hidden_size, self.num_v_heads, bias=False)
            self.ssm_beta = ColumnParallelLinear(hidden_size, self.num_v_heads, bias=False)
            self.attn_gate_ab = None
            self.attn_qkv_gate = None
            self.ssm_ab = None
            self.ssm_dt = nn.Parameter(torch.zeros(self.num_v_heads))
            self.ssm_a = nn.Parameter(-torch.ones(self.num_v_heads))
            if self._short_convolution_cls is not None:
                self.ssm_conv1d = self._short_convolution_cls(
                    hidden_size=self.key_dim * 2 + self.value_dim,
                    kernel_size=self.conv_kernel_size,
                    bias=False,
                    activation="silu",
                )
                self._use_short_convolution = True
            else:
                self.ssm_conv1d = nn.Conv1d(
                    in_channels=self.key_dim * 2 + self.value_dim,
                    out_channels=self.key_dim * 2 + self.value_dim,
                    kernel_size=self.conv_kernel_size,
                    bias=False,
                    groups=self.key_dim * 2 + self.value_dim,
                    padding=self.conv_kernel_size - 1,
                )
                self._use_short_convolution = False
            self.ssm_norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=float(config.rms_norm_eps))
            self.ssm_norm.use_triton = triton_kernels
            self.ssm_out = RowParallelLinear(self.value_dim, hidden_size, bias=False)
            self.conv_state_buffer = torch.empty(0)
            self.recurrent_state_buffer = torch.empty(0)
            self.speculative_conv_state_buffer = torch.empty(0)
            self.speculative_recurrent_state_buffer = torch.empty(0)
            self.speculative_replay = None
            self._gguf_interleave_ssm_ab = False
            self._gguf_v_head_reordered = False
            self._gguf_ssm_param_reordered = False
            self._log_ssm_a = False
            self._gdn_prepare_decode = None
            self._gdn_recurrent_raw = None
            self._gdn_replay_verification = False
            self._gdn_replay_probe = None

    def prepare_sequence_state(self, max_batch_size: int, device: torch.device, dtype: torch.dtype):
        if self.layer_type != "linear_attention":
            return
        conv_shape = (int(max_batch_size), self.key_dim * 2 + self.value_dim, self.conv_kernel_size)
        recurrent_shape = (int(max_batch_size), self.num_v_heads, self.head_k_dim, self.head_v_dim)
        with torch.inference_mode():
            if (
                self.conv_state_buffer.numel() == 0
                or tuple(self.conv_state_buffer.shape) != conv_shape
                or self.conv_state_buffer.device != device
                or self.conv_state_buffer.dtype != dtype
            ):
                self.conv_state_buffer = torch.zeros(conv_shape, device=device, dtype=dtype)
            else:
                self.conv_state_buffer.zero_()
            if (
                self.recurrent_state_buffer.numel() == 0
                or tuple(self.recurrent_state_buffer.shape) != recurrent_shape
                or self.recurrent_state_buffer.device != device
                or self.recurrent_state_buffer.dtype != dtype
            ):
                self.recurrent_state_buffer = torch.zeros(recurrent_shape, device=device, dtype=dtype)
            else:
                self.recurrent_state_buffer.zero_()

    def reset_sequence_state(self):
        if self.layer_type != "linear_attention":
            return
        with torch.inference_mode():
            if self.conv_state_buffer.numel() > 0:
                self.conv_state_buffer.zero_()
            if self.recurrent_state_buffer.numel() > 0:
                self.recurrent_state_buffer.zero_()

    def snapshot_sequence_state(self):
        # the live buffers with their copies: a restore writes into the tensors that graphs read, even if a side request replaced them
        if self.layer_type != "linear_attention":
            return None
        return {name: (getattr(self, name), getattr(self, name).to("cpu", copy=True)) for name in ("conv_state_buffer", "recurrent_state_buffer")}

    def restore_sequence_state(self, snapshot):
        for name, (tensor, copy) in (snapshot or {}).items():
            setattr(self, name, tensor)
            tensor.copy_(copy)

    def prepare_speculative_state(self, max_verify_tokens: int) -> None:
        if self.layer_type != "linear_attention":
            return
        # Final verification states are written to the live buffers. Only
        # prefixes before a rejected suffix need separate snapshots.
        conv_shape = (int(max_verify_tokens) - 1, *self.conv_state_buffer.shape)
        recurrent_shape = (int(max_verify_tokens) - 1, *self.recurrent_state_buffer.shape)
        if tuple(self.speculative_conv_state_buffer.shape) != conv_shape or self.speculative_conv_state_buffer.device != self.conv_state_buffer.device or self.speculative_conv_state_buffer.dtype != self.conv_state_buffer.dtype:
            self.speculative_conv_state_buffer = torch.empty(conv_shape, device=self.conv_state_buffer.device, dtype=self.conv_state_buffer.dtype)
        if self._gdn_replay_verification or (self._gdn_replay_probe is not None and self._gdn_replay_probe(self, int(max_verify_tokens))):
            # Accepted prefixes are replayed from the state before verification: no recurrent snapshot per draft token.
            from shared.kernels.qwen_gdn import RecurrentReplay
            state = self.recurrent_state_buffer
            if self.speculative_replay is None or not self.speculative_replay.fits(state, int(max_verify_tokens), state.dtype, state.dtype):
                self.speculative_replay = RecurrentReplay(state, self.num_k_heads, self.head_k_dim, int(max_verify_tokens), state.dtype, state.dtype)
                # the commit replays run in the engine's own graphs, which must not read weights that MMGP may stream: copies of the two
                # small per-head parameters stay with the replay buffers
                self._replay_ssm_parameters = tuple(t.detach().to(device=state.device, copy=True) for t in self._recurrence_ssm_parameters(getattr(self, "_gdn_direct_layout", True)))
            self.speculative_recurrent_state_buffer = torch.empty(0)
        else:
            self.speculative_replay = None
            if tuple(self.speculative_recurrent_state_buffer.shape) != recurrent_shape or self.speculative_recurrent_state_buffer.device != self.recurrent_state_buffer.device or self.speculative_recurrent_state_buffer.dtype != self.recurrent_state_buffer.dtype:
                self.speculative_recurrent_state_buffer = torch.empty(recurrent_shape, device=self.recurrent_state_buffer.device, dtype=self.recurrent_state_buffer.dtype)

    def commit_speculative_state(self, processed_tokens: int, verified_tokens: int) -> None:
        if self.layer_type != "linear_attention" or processed_tokens == verified_tokens:
            return
        self.conv_state_buffer.copy_(self.speculative_conv_state_buffer[int(processed_tokens) - 1])
        if self.speculative_replay is not None:
            self.replay_speculative_state(processed_tokens)
        else:
            self.recurrent_state_buffer.copy_(self.speculative_recurrent_state_buffer[int(processed_tokens) - 1])

    def replay_speculative_state(self, processed_tokens: int) -> None:
        # The recurrent state the last verification held after its first processed tokens, with the same kernel and arithmetic.
        from shared.kernels.qwen_gdn import recurrent_raw_replay
        direct_recurrent_layout = getattr(self, "_gdn_direct_layout", True)
        ssm_a, ssm_dt = self._replay_ssm_parameters
        recurrent_raw_replay(self.speculative_replay, ssm_a, ssm_dt, int(processed_tokens), self.recurrent_state_buffer, **self._recurrence_layout(direct_recurrent_layout))

    def _recurrence_ssm_parameters(self, direct_recurrent_layout: bool):
        # The raw recurrence addresses checkpoint heads directly. Other paths
        # retain the established materialized execution order.
        ssm_a = self.ssm_a if direct_recurrent_layout else _maybe_reorder_gguf_ssm_param(
            self.ssm_a,
            interleave_halves=self._gguf_interleave_ssm_ab,
            tiled_to_grouped=self._gguf_ssm_param_reordered,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
        )
        ssm_a = -torch.exp(ssm_a.float()) if self._log_ssm_a else ssm_a
        ssm_dt = self.ssm_dt if direct_recurrent_layout else _maybe_reorder_gguf_ssm_param(
            self.ssm_dt,
            interleave_halves=self._gguf_interleave_ssm_ab,
            tiled_to_grouped=self._gguf_ssm_param_reordered,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
        )
        return ssm_a, ssm_dt

    def _recurrence_layout(self, direct_recurrent_layout: bool) -> dict:
        return dict(
            v_heads_tiled=direct_recurrent_layout and self._gguf_v_head_reordered,
            ssm_params_tiled=direct_recurrent_layout and self._gguf_ssm_param_reordered,
            interleave_ab=direct_recurrent_layout and self._gguf_interleave_ssm_ab,
        )

    def release_sequence_state(self):
        if self.layer_type != "linear_attention":
            return
        with torch.inference_mode():
            self.conv_state_buffer = torch.empty(0)
            self.recurrent_state_buffer = torch.empty(0)
            self.speculative_conv_state_buffer = torch.empty(0)
            self.speculative_recurrent_state_buffer = torch.empty(0)
            self.speculative_replay = None
            self._replay_ssm_parameters = None  # its GPU copies of ssm_a / ssm_dt, made again with the next replay

    def _get_runtime_conv_state(self, batch_size: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.conv_state_buffer.numel() == 0
            or self.conv_state_buffer.shape[0] < int(batch_size)
            or self.conv_state_buffer.device != hidden_states.device
            or self.conv_state_buffer.dtype != hidden_states.dtype
        ):
            self.prepare_sequence_state(batch_size, hidden_states.device, hidden_states.dtype)
        return self.conv_state_buffer[:batch_size]

    def _get_runtime_recurrent_state(self, batch_size: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.recurrent_state_buffer.numel() == 0
            or self.recurrent_state_buffer.shape[0] < int(batch_size)
            or self.recurrent_state_buffer.device != hidden_states.device
            or self.recurrent_state_buffer.dtype != hidden_states.dtype
        ):
            self.prepare_sequence_state(batch_size, hidden_states.device, hidden_states.dtype)
        return self.recurrent_state_buffer[:batch_size]

    def _forward_full_attention(
        self,
        x_list: list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        past_key_values: Qwen3_5DynamicCache | None,
        cache_prepared: bool = False,
    ) -> torch.Tensor:
        hidden_states = _take_tensor(x_list)
        batch_size, seq_len, _ = hidden_states.shape
        is_cuda = hidden_states.is_cuda
        if self.attn_qkv_full is None:
            q_and_gate = self.attn_q(hidden_states)
            fused_key = fused_value = None
        else:
            q_and_gate, fused_key, fused_value = self.attn_qkv_full(hidden_states).split(
                [self.num_heads * self.head_dim * 2, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], dim=-1
            )
        q_and_gate = q_and_gate.reshape(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        query_states = self.attn_q_norm(query_states)
        if fused_key is not None:
            key_states = fused_key.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            value_states = fused_value.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        elif self.attn_kv is None:
            key_states = self.attn_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            value_states = self.attn_v(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        else:
            key_value_states = self.attn_kv(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim * 2)
            key_states, value_states = torch.chunk(key_value_states, 2, dim=-1)
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        key_states = self.attn_k_norm(key_states)
        hidden_states = None

        cos, sin = position_embeddings
        cache_written = False
        fused_rope = False
        if past_key_values is None and self.attn.use_triton_kv_cache and is_cuda and torch.version.hip is None:
            from shared.kernels.qwen_rope_cache import apply_rope_cache, supported
            fused_rope = supported(query_states, key_states, value_states, cos, sin)
            if fused_rope:
                cache_written = apply_rope_cache(query_states, key_states, value_states, cos, sin, self.attn, get_context().slot_mapping)
        if not fused_rope:
            qk_list = [query_states, key_states]
            query_states = key_states = None
            query_states, key_states = apply_rotary_pos_emb(qk_list, cos, sin)

        if isinstance(past_key_values, Qwen3_5StaticCache) and self.attn.flash_attn_with_kvcache is not None and is_cuda:
            attn_output = self.attn.flash_attn_with_kvcache(
                query_states,
                past_key_values.key_cache[layer_idx],
                past_key_values.value_cache[layer_idx],
                key_states,
                value_states,
                cache_seqlens=past_key_values.cache_seqlens,
                softmax_scale=self.scaling,
                causal=True,
            ).reshape(batch_size, seq_len, -1)
        elif isinstance(past_key_values, Qwen3_5StaticCache) and cache_prepared:
            attn_output = past_key_values.graph_attention(query_states, key_states, value_states, layer_idx, self.scaling).reshape(batch_size, seq_len, -1)
        elif past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                layer_idx,
            )
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            key_length = int(key_states.shape[1])
            qkv_list = [query_states, key_states, value_states]
            query_states = key_states = value_states = None
            attn_output = _flash_attention(
                qkv_list,
                query_lengths=[int(seq_len)] * batch_size,
                key_lengths=[key_length] * batch_size,
                scaling=self.scaling,
                flash_attention_fn=self._flash_attn_varlen_func,
                efficient=qkv_list[0].is_cuda and not triton_dtype_supported(qkv_list[0].device, qkv_list[0].dtype),  # pre-Ampere BF16: FP32 SDPA
            ).reshape(batch_size, seq_len, -1)
        else:
            qkv_list = [
                query_states.reshape(-1, self.num_heads, self.head_dim),
                key_states.reshape(-1, self.num_kv_heads, self.head_dim),
                value_states.reshape(-1, self.num_kv_heads, self.head_dim),
            ]
            query_states = key_states = value_states = None
            attn_output = self.attn.forward_list(qkv_list, cache_written=cache_written).reshape(batch_size, seq_len, -1)
        query_states = key_states = value_states = None
        gate.sigmoid_()
        attn_output.mul_(gate)
        gate = q_and_gate = None
        return self.attn_output(attn_output)

    def _forward_linear_attention(
        self,
        x_list: list[torch.Tensor],
        layer_idx: int,
        cache_params: Qwen3_5DynamicCache | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = _take_tensor(x_list)
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states.mul_(attention_mask[:, :, None])

        batch_size, seq_len, _ = hidden_states.shape
        hidden_dtype = hidden_states.dtype
        is_cuda = hidden_states.is_cuda
        fla = is_cuda and triton_dtype_supported(hidden_states.device, hidden_dtype)  # FLA's kernels convert this dtype on this GPU
        fast_recurrent = self._fast_recurrent_gated_delta_rule if fla else None
        fast_chunk = self._fast_chunk_gated_delta_rule if fla else None
        context = get_context()
        if cache_params is not None:
            conv_state = cache_params.conv_states[layer_idx]
            recurrent_state = cache_params.recurrent_states[layer_idx]
            has_previous_state = cache_params.has_previous_state
        else:
            conv_state = self._get_runtime_conv_state(batch_size, hidden_states)
            recurrent_state = self._get_runtime_recurrent_state(batch_size, hidden_states)
            has_previous_state = bool(getattr(context, "has_previous_state", False)) if context.is_prefill else True
        use_precomputed_states = has_previous_state and seq_len == 1
        speculative_verify = context.speculative_verify and cache_params is None and has_previous_state and seq_len > 1
        use_raw_recurrent = (self._gdn_recurrent_raw is not None and is_cuda
                             and cache_params is None and (use_precomputed_states or speculative_verify))
        direct_recurrent_layout = use_raw_recurrent and getattr(self, "_gdn_direct_layout", True)
        if speculative_verify:
            recurrent_capacity = self.speculative_replay.tokens - 1 if self.speculative_replay is not None else self.speculative_recurrent_state_buffer.shape[0]
            if self.speculative_conv_state_buffer.shape[0] < seq_len - 1 or recurrent_capacity < seq_len - 1:
                raise RuntimeError(f"Predictive state buffers do not cover a {seq_len}-token verification pass.")

        if self.attn_qkv_gate is not None:
            mixed_qkv_input, z = self.attn_qkv_gate(hidden_states).split([self.key_dim * 2 + self.value_dim, self.value_dim], dim=-1)
            # The convolution verification kernel consumes dense [B, T, C].
            # A multi-token slice of the combined projection has a wider stride.
            mixed_qkv_input = mixed_qkv_input.contiguous()
            z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
            # the dense BF16 alpha/beta projection of a speculative verification row by row through the single-token GEMV (_rows_linear)
            ab_rows = 1 < batch_size * seq_len <= _AB_ROWS_GEMV_MAX and _slow_bf16_gemm(hidden_states)
            a, b = (_rows_linear(self.ssm_ab, hidden_states) if ab_rows else self.ssm_ab(hidden_states)).chunk(2, dim=-1)
        elif self.attn_gate_ab is None:
            mixed_qkv_input = self.attn_qkv(hidden_states)
            z = self.attn_gate(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
            a = self.ssm_alpha(hidden_states)
            b = self.ssm_beta(hidden_states)
        else:
            mixed_qkv_input = self.attn_qkv(hidden_states)
            gate_ab = self.attn_gate_ab(hidden_states)
            gate_proj, a, b = torch.split(gate_ab, [self.value_dim, self.num_v_heads, self.num_v_heads], dim=-1)
            z = gate_proj.reshape(batch_size, seq_len, -1, self.head_v_dim)
        hidden_states = None
        if self._gguf_v_head_reordered and not direct_recurrent_layout:
            z = _reorder_v_head_axis_tiled_to_grouped(
                z,
                dim=2,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
            )
        if self._gguf_v_head_reordered and not direct_recurrent_layout:
            b = _reorder_v_heads_tiled_to_grouped(
                b,
                dim=-1,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_dim=1,
            )
            a = _reorder_v_heads_tiled_to_grouped(
                a,
                dim=-1,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_dim=1,
            )
        elif self._gguf_interleave_ssm_ab and not direct_recurrent_layout:
            b = _interleave_axis_halves(b, dim=-1)
            a = _interleave_axis_halves(a, dim=-1)

        use_short_convolution = (
            fla and self._use_short_convolution
            and self._short_convolution_cls is not None
            and isinstance(self.ssm_conv1d, self._short_convolution_cls)
        )

        if speculative_verify and use_short_convolution and is_cuda:
            from shared.llm_engines.lmgp.layers.speculative_state import conv_verify
            mixed_qkv, last_conv_state = conv_verify(mixed_qkv_input, conv_state, self.ssm_conv1d.weight, self.ssm_conv1d.bias, self.speculative_conv_state_buffer)
        elif speculative_verify and use_short_convolution:
            conv_outputs = []
            for token_idx in range(seq_len):
                conv_output, _ = self.ssm_conv1d(mixed_qkv_input[:, token_idx:token_idx + 1], cache=conv_state, output_final_state=True)
                conv_outputs.append(conv_output)
                if token_idx + 1 < seq_len:
                    self.speculative_conv_state_buffer[token_idx].copy_(conv_state)
            mixed_qkv = torch.cat(conv_outputs, dim=1)
            last_conv_state = conv_state
        elif use_short_convolution:
            short_conv_cache = conv_state if has_previous_state and conv_state is not None else None
            mixed_qkv, last_conv_state = self.ssm_conv1d(
                mixed_qkv_input,
                cache=short_conv_cache,
                output_final_state=True,
            )
        else:
            mixed_qkv = mixed_qkv_input.transpose(1, 2)
            use_fast_causal_conv = causal_conv1d_fn is not None and causal_conv1d_update is not None
            if speculative_verify:
                conv_kernel = self.ssm_conv1d.weight.reshape(self.ssm_conv1d.weight.shape[0], self.ssm_conv1d.weight.shape[-1])
                conv_outputs = []
                for token_idx in range(seq_len):
                    conv_input = mixed_qkv[:, :, token_idx:token_idx + 1]
                    if use_fast_causal_conv:
                        conv_output = causal_conv1d_update(conv_input, conv_state, conv_kernel, self.ssm_conv1d.bias, "silu")
                    else:
                        conv_output = torch_causal_conv1d_update(conv_input, conv_state, conv_kernel, self.ssm_conv1d.bias)
                    conv_outputs.append(conv_output)
                    if token_idx + 1 < seq_len:
                        self.speculative_conv_state_buffer[token_idx].copy_(conv_state)
                mixed_qkv = torch.cat(conv_outputs, dim=-1)
            elif use_precomputed_states:
                if use_fast_causal_conv:
                    conv_kernel = self.ssm_conv1d.weight.squeeze(1)
                    mixed_qkv = causal_conv1d_update(
                        mixed_qkv,
                        conv_state,
                        conv_kernel,
                        self.ssm_conv1d.bias,
                        "silu",
                    )
                else:
                    conv_kernel = self.ssm_conv1d.weight.reshape(
                        self.ssm_conv1d.weight.shape[0],
                        self.ssm_conv1d.weight.shape[-1],
                    )
                    mixed_qkv = torch_causal_conv1d_update(
                        mixed_qkv,
                        conv_state,
                        conv_kernel,
                        self.ssm_conv1d.bias,
                    )
            elif has_previous_state:
                conv_kernel = self.ssm_conv1d.weight.reshape(self.ssm_conv1d.weight.shape[0], self.ssm_conv1d.weight.shape[-1])
                mixed_qkv = torch_causal_conv1d_update(mixed_qkv, conv_state, conv_kernel, self.ssm_conv1d.bias)
            else:
                if cache_params is not None:
                    if mixed_qkv.shape[-1] >= self.conv_kernel_size:
                        cache_params.conv_states[layer_idx] = mixed_qkv[:, :, -self.conv_kernel_size :].contiguous()
                    else:
                        cache_params.conv_states[layer_idx] = F.pad(
                            mixed_qkv,
                            (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
                        )
                else:
                    if mixed_qkv.shape[-1] >= self.conv_kernel_size:
                        conv_state.copy_(mixed_qkv[:, :, -self.conv_kernel_size :].contiguous())
                    else:
                        conv_state.copy_(
                            F.pad(
                                mixed_qkv,
                                (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
                            )
                        )
                conv_input = mixed_qkv.to(self.ssm_conv1d.weight.dtype)
                if use_fast_causal_conv:
                    mixed_qkv = causal_conv1d_fn(
                        x=conv_input,
                        weight=self.ssm_conv1d.weight.squeeze(1),
                        bias=self.ssm_conv1d.bias,
                        activation="silu",
                        seq_idx=None,
                    )
                else:
                    mixed_qkv = F.silu(nn.Conv1d.forward(self.ssm_conv1d, conv_input)[:, :, :seq_len])  # FLA's module as the plain convolution
                mixed_qkv = mixed_qkv.to(hidden_dtype)
            mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_short_convolution:
            if cache_params is not None:
                cache_params.conv_states[layer_idx] = last_conv_state
            elif last_conv_state is not conv_state:
                conv_state.copy_(last_conv_state)

        mixed_qkv_input = None

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if self._gguf_v_head_reordered and not direct_recurrent_layout:
            value = _reorder_v_head_axis_tiled_to_grouped(
                value,
                dim=2,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
            )

        ssm_a, ssm_dt = self._recurrence_ssm_parameters(direct_recurrent_layout)
        if use_raw_recurrent:
            core_attn_out, last_recurrent_state = self._gdn_recurrent_raw(
                query, key, value, a, b, ssm_a, ssm_dt, recurrent_state,
                self.speculative_recurrent_state_buffer if speculative_verify and self.speculative_replay is None else None,
                self.speculative_replay if speculative_verify else None,
                **self._recurrence_layout(direct_recurrent_layout),
            )
            g = beta = None
        elif self._gdn_prepare_decode is not None and use_precomputed_states and is_cuda:
            query, key, g, beta = self._gdn_prepare_decode(query, key, a, b, ssm_a, ssm_dt, self.num_v_heads)
        else:
            beta = b.sigmoid()
            g = ssm_a.float() * F.softplus(a.float() + ssm_dt)
            if self.num_v_heads // self.num_k_heads > 1:
                repeat_factor = self.num_v_heads // self.num_k_heads
                query = query.repeat_interleave(repeat_factor, dim=2)
                key = key.repeat_interleave(repeat_factor, dim=2)
        a = b = None

        if use_raw_recurrent:
            pass
        elif speculative_verify and fast_recurrent is not None:
            from shared.llm_engines.lmgp.layers.speculative_state import recurrent_verify
            core_attn_out, last_recurrent_state = recurrent_verify(query, key, value, g, beta, recurrent_state, self.speculative_recurrent_state_buffer)
        elif speculative_verify:
            recurrent_outputs = []
            current_recurrent_state = recurrent_state
            for token_idx in range(seq_len):
                if fast_recurrent is not None:
                    recurrent_output, current_recurrent_state = fast_recurrent(
                        query[:, token_idx:token_idx + 1],
                        key[:, token_idx:token_idx + 1],
                        value[:, token_idx:token_idx + 1],
                        g=g[:, token_idx:token_idx + 1],
                        beta=beta[:, token_idx:token_idx + 1],
                        initial_state=current_recurrent_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                else:
                    recurrent_output, current_recurrent_state = torch_recurrent_gated_delta_rule(
                        query[:, token_idx:token_idx + 1],
                        key[:, token_idx:token_idx + 1],
                        value[:, token_idx:token_idx + 1],
                        g=g[:, token_idx:token_idx + 1],
                        beta=beta[:, token_idx:token_idx + 1],
                        initial_state=current_recurrent_state,
                        output_final_state=True,
                    )
                recurrent_outputs.append(recurrent_output)
                if token_idx + 1 < seq_len:
                    self.speculative_recurrent_state_buffer[token_idx].copy_(current_recurrent_state)
            core_attn_out = torch.cat(recurrent_outputs, dim=1)
            last_recurrent_state = current_recurrent_state
        elif use_precomputed_states:
            if fast_recurrent is not None:
                core_attn_out, last_recurrent_state = fast_recurrent(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                )
        else:
            recurrent_initial_state = recurrent_state if has_previous_state else None
            if fast_chunk is not None:
                core_attn_out, last_recurrent_state = fast_chunk(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_initial_state,
                    output_final_state=True,
                )

        if cache_params is not None:
            cache_params.recurrent_states[layer_idx] = last_recurrent_state
        elif last_recurrent_state is not recurrent_state:
            recurrent_state.copy_(last_recurrent_state)

        query = key = value = mixed_qkv = beta = g = last_recurrent_state = None

        norm_state_list = [core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)]
        core_attn_out = z = None
        core_attn_out = _forward_gated_norm_list(self.ssm_norm, norm_state_list)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        if self._gguf_v_head_reordered and not direct_recurrent_layout:
            core_attn_out = _reorder_v_heads_grouped_to_tiled(
                core_attn_out,
                dim=-1,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_dim=self.head_v_dim,
            )
        return self.ssm_out(core_attn_out)

    def forward(
        self,
        x_list: list[torch.Tensor | None],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3_5DynamicCache | None = None,
        cache_prepared: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = x_list
        x_list.clear()
        if residual is None:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
        else:
            norm_state_list = [hidden_states, residual]
            hidden_states = residual = None
            hidden_states, residual = self.attn_norm.forward_list(norm_state_list)

        if self.layer_type == "full_attention":
            x_list = [hidden_states]
            hidden_states = None
            hidden_states = self._forward_full_attention(x_list, position_embeddings, layer_idx, past_key_values, cache_prepared)
        else:
            x_list = [hidden_states]
            hidden_states = None
            hidden_states = self._forward_linear_attention(x_list, layer_idx, past_key_values, attention_mask)

        norm_state_list = [hidden_states, residual]
        hidden_states = residual = None
        hidden_states, residual = self.post_attention_norm.forward_list(norm_state_list)
        gate_up = self.ffn_gate_up(hidden_states) if self.ffn_gate_up is not None else torch.cat((self.ffn_gate(hidden_states), self.ffn_up(hidden_states)), dim=-1)
        hidden_states = None
        gate_up_list = [gate_up]
        gate_up = None
        # Decode and speculative verification (up to 8 rows): the package quantizes silu(gate) * up itself, same arithmetic as the
        # Triton kernel (round the SiLU, round the product, then Q8), one launch and a BF16 round trip fewer per layer.
        if self.mlp_act_fn.use_triton and getattr(self.ffn_down, "_fuse_silu_mul", False) and gate_up_list[0].numel() <= 8 * gate_up_list[0].shape[-1]:
            hidden_states = self.ffn_down(gate_up_list)
        else:
            hidden_states = self.ffn_down(self.mlp_act_fn.forward_list(gate_up_list))
        return hidden_states, residual


class Qwen3_5ForCausalLM(nn.Module):
    _lm_engine_tower_prefix = "blk."  # the blocks MMGP streams: the engine's own graphs read the rest of the model

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.token_embd = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.blk = nn.ModuleList([Qwen3_5Block(config, idx) for idx in range(int(config.num_hidden_layers))])
        self.output_norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.output_norm.use_triton_rmsnorm = _triton_kernels(config)
        self.output = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)
        self.mtp = Qwen3_5MTP(config) if bool(getattr(config, "_prompt_enhancer_enable_mtp_speculative", False)) else None
        self._block_draft = False
        self._draft_features = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def embed_tokens(self):
        return self.token_embd

    @property
    def layers(self):
        return self.blk

    @property
    def norm(self):
        return self.output_norm

    @property
    def lm_head(self):
        return self.output

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3_5DynamicCache | None = None,
        use_cache: bool | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        del use_cache
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if position_ids is not None:
            positions = position_ids

        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions is None:
            seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            positions = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                device=(inputs_embeds.device if inputs_embeds is not None else input_ids.device),
                dtype=torch.long,
            ).unsqueeze(0)
        elif positions.ndim == 1:
            positions = positions.unsqueeze(0)
        elif positions.ndim == 3 and positions.shape[0] == 4:
            positions = positions[1:]

        hidden_states = self.token_embd(input_ids) if inputs_embeds is None else inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, positions)
        residual = None
        draft_features = [] if self._block_draft else None
        if draft_features is None and attention_mask is None and past_key_values is None:
            # MMGP: inside a graph scope, each run of blocks kept in VRAM is one graph, the streamed blocks go through their hooks
            carry = [hidden_states, None]
            hidden_states = None
            hidden_states, residual = cuda_graphs.run_blocks(self, self.blk, self._run_block, carry, position_embeddings)
        else:
            for layer_idx, block in enumerate(self.blk):
                x_list = [hidden_states, residual]
                hidden_states = residual = None
                hidden_states, residual = block(
                    x_list,
                    position_embeddings,
                    layer_idx=layer_idx,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                if draft_features is not None and layer_idx in self.mtp.target_layer_ids:
                    draft_features.append(hidden_states + residual)
        if draft_features is not None:
            self._draft_features = torch.cat(draft_features, dim=-1)
        norm_state_list = [hidden_states, residual]
        hidden_states = residual = None
        hidden_states, _ = self.output_norm.forward_list(norm_state_list)
        return hidden_states

    @staticmethod
    def _run_block(layer_idx: int, block: nn.Module, x_list: list, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> list:
        return list(block(x_list, position_embeddings, layer_idx=layer_idx, attention_mask=None, past_key_values=None))

    def lm_engine_graph_forward(self, input_ids: torch.Tensor, positions: torch.Tensor, outputs: torch.Tensor, draft_features: torch.Tensor | None = None) -> None:
        """A decoding pass of the engine (cuda_graphs.run stage): the hidden states of input_ids written into outputs, the block drafters'
        target features into draft_features. The context of the pass reads the engine's registered buffers."""
        outputs.copy_(self(input_ids, positions).view(outputs.shape))
        if draft_features is not None:
            draft_features.copy_(self._draft_features.view(draft_features.shape))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if hidden_states.ndim == 3:
            if context.is_prefill and context.cu_seqlens_q is not None and context.cu_seqlens_q.numel() > 1:
                num_seqs = int(context.cu_seqlens_q.numel() - 1)
                if hidden_states.shape[0] == num_seqs:
                    lengths = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                    row_indices = torch.arange(num_seqs, device=hidden_states.device)
                    token_indices = lengths.to(device=hidden_states.device, dtype=torch.long) - 1
                    hidden_states = hidden_states[row_indices, token_indices]
                elif hidden_states.shape[0] == 1 and num_seqs == 1:
                    last_index = int(context.cu_seqlens_q[-1].item()) - 1
                    hidden_states = hidden_states[:, last_index, :]
            elif hidden_states.shape[-2] == 1:
                hidden_states = hidden_states[:, 0, :]
        return self.output(hidden_states)


class Qwen3_5MTP(nn.Module):
    """The native single-layer Qwen3.5/3.8 next-token predictor."""

    def __init__(self, config):
        super().__init__()
        mtp_config = copy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.layer_types = ["full_attention"]
        hidden_size = int(config.hidden_size)
        self.embed_tokens = nn.Embedding(int(config.vocab_size), hidden_size)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.fc = ColumnParallelLinear(hidden_size * 2, hidden_size, bias=False)
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.block = Qwen3_5Block(mtp_config, 0)
        self.norm = RMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.lm_head = nn.Linear(hidden_size, int(config.vocab_size), bias=False)
        for module in (self.pre_fc_norm_embedding, self.pre_fc_norm_hidden, self.norm):
            module.use_triton_rmsnorm = _triton_kernels(config)
        self.block.attn._exclude_paged_kv_cache = True
        self._cache = None
        self._cache_config = mtp_config

    def prepare_cache(self, max_cache_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            not isinstance(self._cache, Qwen3_5StaticCache)
            or self._cache.max_cache_len != int(max_cache_len)
            or self._cache.key_cache[0].device != device
            or self._cache.key_cache[0].dtype != dtype
        ):
            self._cache = Qwen3_5StaticCache(self._cache_config, max_cache_len, device, dtype)
        else:
            self._cache.reset()

    def reset_sequence_state(self):
        if isinstance(self._cache, Qwen3_5StaticCache):
            self._cache.reset()
        else:
            self._cache = None

    def release_sequence_state(self):
        clear_head_cache = getattr(self.lm_head, "clear_cache", None)
        if callable(clear_head_cache):
            clear_head_cache()
        self._cache = None

    def snapshot_sequence_state(self, previous: dict | None = None, reuse_tokens: int = 0) -> dict | None:
        if self._cache is None:  # no cache: an empty state (speculative decoding off, or released)
            return None
        if not isinstance(self._cache, Qwen3_5StaticCache):
            raise RuntimeError("MTP static cache is not prepared.")
        return self._cache.snapshot(previous=previous, reuse_tokens=reuse_tokens)

    def restore_sequence_state(self, snapshot: dict | None) -> None:
        if snapshot is None:
            self.reset_sequence_state()
            return
        if not isinstance(self._cache, Qwen3_5StaticCache):
            raise RuntimeError("MTP static cache is not prepared.")
        self._cache.restore(snapshot)

    def get_cache_length(self) -> int:
        if not isinstance(self._cache, Qwen3_5StaticCache):
            raise RuntimeError("MTP static cache is not prepared.")
        return self._cache.get_seq_length()

    def truncate_cache(self, seq_length: int) -> None:
        if not isinstance(self._cache, Qwen3_5StaticCache):
            raise RuntimeError("MTP static cache is not prepared.")
        self._cache.truncate(seq_length)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor, inputs_embeds: torch.Tensor | None = None, compute_logits: bool = True, last_logits_only: bool = False, cache_prepared: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if inputs_embeds is None:
            token_embeddings = self.embed_tokens(input_ids)
        elif inputs_embeds.shape[1] == hidden_states.shape[1] - 1:
            token_embeddings = torch.cat((inputs_embeds, self.embed_tokens(input_ids[:, -1:])), dim=1)
        else:
            token_embeddings = inputs_embeds
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        token_embeddings = self.pre_fc_norm_embedding(token_embeddings)
        hidden_states = self.fc(torch.cat((token_embeddings, hidden_states), dim=-1))
        position_embeddings = self.rotary_emb(hidden_states, positions)
        if self._cache is None:
            self._cache = Qwen3_5DynamicCache(self._cache_config)
        static_flash_cache = isinstance(self._cache, Qwen3_5StaticCache) and self.block.attn.flash_attn_with_kvcache is not None and hidden_states.is_cuda
        if static_flash_cache and not cache_prepared:
            self._cache.prepare_append()
        x_list = [hidden_states, None]
        hidden_states = None
        hidden_states, residual = self.block(x_list, position_embeddings, layer_idx=0, past_key_values=self._cache, cache_prepared=cache_prepared)
        if static_flash_cache and not cache_prepared:
            self._cache.advance(hidden_states.shape[1])
        norm_state_list = [hidden_states, residual]
        hidden_states = residual = None
        hidden_states, _ = self.norm.forward_list(norm_state_list)
        logits_hidden = hidden_states[:, -1:] if last_logits_only else hidden_states
        return hidden_states, self.lm_head(logits_hidden) if compute_logits else None


__all__ = ["Qwen3_5DynamicCache", "Qwen3_5StaticCache", "Qwen3_5ForCausalLM", "Qwen3_5MTP", "clear_qwen35_runtime_caches"]
