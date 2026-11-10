# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as tl_libdevice
except Exception:  # pragma: no cover
    triton = None
    tl = None
    tl_libdevice = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except Exception:  # pragma: no cover
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
try:
    import llamacpp_gguf_cuda
except Exception:  # pragma: no cover
    llamacpp_gguf_cuda = None
from shared.kernels import bf16_bits, dispatch
from ..utils.context import get_context
from . import kv_formats as kvf
from . import attention_mixed

_MIXED_KERNELS = True  # mixed layouts use the Triton kernels of attention_mixed where available (else the dense fallbacks)
# The dense prefill (Attention.dense_prefill) serves chunks of this many tokens: below, a prefill lasts what its kernel launches take on the host,
# and the check of the dense output would make each layer wait for the GPU (128 tokens at 8K of context: 129 ms exact, 140 ms dense).
_DENSE_PREFILL_ROWS = 256
# It reads a context of three slices or more by slices of this many tokens, merged by their softmax normalizations: what it takes in VRAM no longer
# follows the context (27B, a 1,024-token chunk: 122 MiB at any length where 61K tokens at once took 282 MiB, for 1.7 % of the chunk's time).
_DENSE_PREFILL_KEYS = 8192

_DEFAULT_FLASH_ATTN_VARLEN_FUNC = flash_attn_varlen_func
_DEFAULT_FLASH_ATTN_WITH_KVCACHE = flash_attn_with_kvcache
_USE_TRITON_KV_CACHE = triton is not None and tl is not None
_DEFAULT_USE_TRITON_KV_CACHE = _USE_TRITON_KV_CACHE
_Q8_KV_BLOCK_SIZE = 32
_LOGGED_KV_ATTENTION_BACKENDS = set()


def _log_kv_attention_backend_once(backend: str, message: str) -> None:
    if backend in _LOGGED_KV_ATTENTION_BACKENDS:
        return
    _LOGGED_KV_ATTENTION_BACKENDS.add(backend)
    print(message)


def reset_attention_backend_logs() -> None:
    _LOGGED_KV_ATTENTION_BACKENDS.clear()


def _get_q8_paged_attention(module):
    kernel = getattr(module, "q8_paged_attention", None)
    cache_format = getattr(module, "q8_paged_attention_format", None)
    return kernel if callable(kernel) and callable(cache_format) and cache_format() == "q8_0_fp16_scales_v1" else None


_Q8_PAGED_ATTENTION = _get_q8_paged_attention(llamacpp_gguf_cuda)


def configure_attention_safe_legacy_kernels(enabled: bool) -> None:
    global flash_attn_varlen_func, flash_attn_with_kvcache, _USE_TRITON_KV_CACHE
    if enabled:
        flash_attn_varlen_func = None
        flash_attn_with_kvcache = None
        _USE_TRITON_KV_CACHE = False
        return
    flash_attn_varlen_func = _DEFAULT_FLASH_ATTN_VARLEN_FUNC
    flash_attn_with_kvcache = _DEFAULT_FLASH_ATTN_WITH_KVCACHE
    _USE_TRITON_KV_CACHE = triton is not None and tl is not None


if triton is not None and tl is not None:
    @triton.jit(do_not_specialize=("key_stride", "value_stride"))
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        offsets = tl.arange(0, triton.next_power_of_2(D))
        key_offsets = idx * key_stride + offsets
        value_offsets = idx * value_stride + offsets
        key = tl.load(key_ptr + key_offsets, mask=offsets < D, other=0)
        value = tl.load(value_ptr + value_offsets, mask=offsets < D, other=0)
        cache_offsets = slot * D + offsets
        tl.store(k_cache_ptr + cache_offsets, key, mask=offsets < D)
        tl.store(v_cache_ptr + cache_offsets, value, mask=offsets < D)


    @triton.jit
    def store_int8_kvcache_kernel(
        key_ptr,
        key_stride_token,
        key_stride_head,
        value_ptr,
        value_stride_token,
        value_stride_head,
        k_cache_ptr,
        v_cache_ptr,
        k_scale_ptr,
        v_scale_ptr,
        slot_mapping_ptr,
        H: tl.constexpr,
        D: tl.constexpr,
        QB: tl.constexpr, BF16_BITS: tl.constexpr = False,
    ):
        idx = tl.program_id(0)
        token_idx = idx // (H * QB)
        head_block_idx = idx % (H * QB)
        head_idx = head_block_idx // QB
        quant_block_idx = head_block_idx % QB
        slot = tl.load(slot_mapping_ptr + token_idx)
        if slot == -1: return
        offsets = quant_block_idx * 32 + tl.arange(0, 32)
        key = _load_fp32(key_ptr + token_idx * key_stride_token + head_idx * key_stride_head + offsets, BF16_BITS)
        value = _load_fp32(value_ptr + token_idx * value_stride_token + head_idx * value_stride_head + offsets, BF16_BITS)
        key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / 127.0, 1e-8)
        value_scale = tl.maximum(tl.max(tl.abs(value), axis=0) / 127.0, 1e-8)
        key_int8 = tl_libdevice.nearbyint(key / key_scale).to(tl.int8)
        value_int8 = tl_libdevice.nearbyint(value / value_scale).to(tl.int8)
        cache_offsets = (slot * H + head_idx) * D + offsets
        scale_offset = (slot * H + head_idx) * QB + quant_block_idx
        tl.store(k_cache_ptr + cache_offsets, key_int8)
        tl.store(v_cache_ptr + cache_offsets, value_int8)
        tl.store(k_scale_ptr + scale_offset, key_scale)
        tl.store(v_scale_ptr + scale_offset, value_scale)


    @triton.jit(do_not_specialize=("bt_stride",), do_not_specialize_on_alignment=("bt_stride",))
    def q8_paged_prefill_kernel(
        q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, block_tables_ptr, cu_q_ptr, cu_k_ptr, out_ptr,
        q_stride_t: tl.constexpr, q_stride_h: tl.constexpr, cache_stride_block: tl.constexpr,
        cache_stride_token: tl.constexpr, cache_stride_head: tl.constexpr, scale_stride_block: tl.constexpr,
        scale_stride_token: tl.constexpr, scale_stride_head: tl.constexpr, bt_stride,
        out_stride_t: tl.constexpr, out_stride_h: tl.constexpr, softmax_scale: tl.constexpr,
        H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        sequence = tl.program_id(0)
        query_head = tl.program_id(1)
        query_block = tl.program_id(2)
        q_start = tl.load(cu_q_ptr + sequence)
        q_end = tl.load(cu_q_ptr + sequence + 1)
        k_end = tl.load(cu_k_ptr + sequence + 1) - tl.load(cu_k_ptr + sequence)
        q_len = q_end - q_start
        prefix_len = k_end - q_len
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, D)
        row_mask = rows < q_len
        q = tl.load(q_ptr + (q_start + rows[:, None]) * q_stride_t + query_head * q_stride_h + dims[None, :], mask=row_mask[:, None], other=0.0)
        maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        denominator = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, D), tl.float32)
        kv_head = query_head // (H_Q // H_KV)

        for key_start in tl.range(0, tl.minimum(k_end, prefix_len + (query_block + 1) * BLOCK_M), BLOCK_N):
            columns = key_start + tl.arange(0, BLOCK_N)
            column_mask = columns < k_end
            physical_block = tl.load(block_tables_ptr + sequence * bt_stride + columns // PAGE, mask=column_mask, other=0)
            physical_token = columns % PAGE
            cache_base = physical_block * cache_stride_block + physical_token * cache_stride_token + kv_head * cache_stride_head
            scale_base = physical_block * scale_stride_block + physical_token * scale_stride_token + kv_head * scale_stride_head
            k = tl.load(k_ptr + cache_base[None, :] + dims[:, None], mask=column_mask[None, :], other=0.0).to(tl.float32)
            k *= tl.load(ks_ptr + scale_base[None, :] + (dims[:, None] // 32), mask=column_mask[None, :], other=0.0).to(tl.float32)
            k_high = k.to(q.dtype)
            # Keep the dequantized cache precision while using tensor cores.
            # A high/low pair avoids rounding Q8 * FP16 scales to one BF16 value.
            k_low = (k - k_high.to(tl.float32)).to(q.dtype)
            scores = tl.dot(q, k_high)
            scores = tl.dot(q, k_low, scores) * softmax_scale
            causal = columns[None, :] <= prefix_len + rows[:, None]
            scores = tl.where(row_mask[:, None] & column_mask[None, :] & causal, scores, -float("inf"))
            tile_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            correction = tl.exp2((maximum - tile_maximum) * 1.4426950408889634)
            probabilities = tl.exp2((scores - tile_maximum[:, None]) * 1.4426950408889634)
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            v = tl.load(v_ptr + cache_base[:, None] + dims[None, :], mask=column_mask[:, None], other=0.0).to(tl.float32)
            v *= tl.load(vs_ptr + scale_base[:, None] + (dims[None, :] // 32), mask=column_mask[:, None], other=0.0).to(tl.float32)
            p_high = probabilities.to(q.dtype)
            p_low = (probabilities - p_high.to(tl.float32)).to(q.dtype)
            v_high = v.to(q.dtype)
            v_low = (v - v_high.to(tl.float32)).to(q.dtype)
            accumulator = accumulator * correction[:, None]
            accumulator = tl.dot(p_high, v_high, accumulator)
            accumulator = tl.dot(p_low, v_high, accumulator)
            accumulator = tl.dot(p_high, v_low, accumulator)
            accumulator = tl.dot(p_low, v_low, accumulator)
            maximum = tile_maximum
        output = accumulator / denominator[:, None]
        tl.store(out_ptr + (q_start + rows[:, None]) * out_stride_t + query_head * out_stride_h + dims[None, :], output, mask=row_mask[:, None])


    @triton.jit(do_not_specialize=("Q_PER_SEQ", "TABLE_WIDTH"), do_not_specialize_on_alignment=("Q_PER_SEQ", "TABLE_WIDTH"))
    def q8_grouped_partials_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tables_ptr, lengths_ptr, partial_ptr, maximum_ptr, sum_ptr, tree_ptr,
                                  Q_PER_SEQ, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                                  TABLE_WIDTH, SPLITS: tl.constexpr, M: tl.constexpr, N: tl.constexpr, SCALE: tl.constexpr, TREE: tl.constexpr):
        sequence, kv_head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        split, query_block = tile % SPLITS, tile // SPLITS
        group: tl.constexpr = H_Q // H_KV
        rows = query_block * M + tl.arange(0, M)
        query_index = sequence * Q_PER_SEQ + rows // group
        query_head = kv_head * group + rows % group
        valid_rows = rows < Q_PER_SEQ * group
        dims = tl.arange(0, D)
        q = tl.load(q_ptr + (query_index[:, None] * H_Q + query_head[:, None]) * D + dims[None, :], mask=valid_rows[:, None], other=0.0)
        context_len = tl.load(lengths_ptr + sequence)
        prefix_len = context_len - Q_PER_SEQ
        if TREE:  # tree verification: each row sees the prefix and the verify rows marked in its ancestor bits (itself included)
            ancestors = tl.load(tree_ptr + sequence * Q_PER_SEQ + rows // group, mask=valid_rows, other=0)
        tiles_per_split = tl.cdiv(tl.cdiv(context_len, N), SPLITS)
        begin = split * tiles_per_split * N
        end = tl.minimum(context_len, (split + 1) * tiles_per_split * N)
        maximum = tl.full((M,), -1.0e20, tl.float32)
        denominator = tl.zeros((M,), tl.float32)
        acc = tl.zeros((M, D), tl.float32)
        for start in tl.range(begin, end, N):
            columns = start + tl.arange(0, N)
            valid_columns = columns < end
            page = tl.load(tables_ptr + sequence * TABLE_WIDTH + columns // PAGE, mask=valid_columns, other=0)
            slot = page * PAGE + columns % PAGE
            offsets = (slot * H_KV + kv_head) * D
            scale_offsets = (slot * H_KV + kv_head) * (D // 32)
            k = tl.load(k_ptr + offsets[None, :] + dims[:, None], mask=valid_columns[None, :], other=0).to(tl.float32)
            k *= tl.load(ks_ptr + scale_offsets[None, :] + dims[:, None] // 32, mask=valid_columns[None, :], other=0).to(tl.float32)
            k_hi = k.to(q.dtype)
            k_lo = (k - k_hi.to(tl.float32)).to(q.dtype)
            scores = tl.dot(q, k_hi)
            scores = tl.dot(q, k_lo, scores) * SCALE
            if TREE:
                local = columns - prefix_len
                visible = (local[None, :] < 0) | (((ancestors[:, None] >> tl.maximum(local, 0)[None, :].to(tl.int64)) & 1) != 0)
                valid = valid_rows[:, None] & valid_columns[None, :] & visible
            else:
                valid = valid_rows[:, None] & valid_columns[None, :] & (columns[None, :] <= prefix_len + rows[:, None] // group)
            scores = tl.where(valid, scores, -1.0e20)
            new_maximum = tl.maximum(maximum, tl.max(scores, 1))
            correction = tl.exp2((maximum - new_maximum) * 1.4426950408889634)
            p = tl.where(valid, tl.exp2((scores - new_maximum[:, None]) * 1.4426950408889634), 0.0)
            denominator = denominator * correction + tl.sum(p, 1)
            v = tl.load(v_ptr + offsets[:, None] + dims[None, :], mask=valid_columns[:, None], other=0).to(tl.float32)
            v *= tl.load(vs_ptr + scale_offsets[:, None] + dims[None, :] // 32, mask=valid_columns[:, None], other=0).to(tl.float32)
            p_hi, v_hi = p.to(q.dtype), v.to(q.dtype)
            p_lo = (p - p_hi.to(tl.float32)).to(q.dtype)
            v_lo = (v - v_hi.to(tl.float32)).to(q.dtype)
            acc *= correction[:, None]
            acc = tl.dot(p_hi, v_hi, acc)
            acc = tl.dot(p_lo, v_hi, acc)
            acc = tl.dot(p_hi, v_lo, acc)
            acc = tl.dot(p_lo, v_lo, acc)
            maximum = new_maximum
        index = (query_index * H_Q + query_head) * SPLITS + split
        tl.store(partial_ptr + index[:, None] * D + dims[None, :], acc, mask=valid_rows[:, None])
        tl.store(maximum_ptr + index, maximum, mask=valid_rows)
        tl.store(sum_ptr + index, denominator, mask=valid_rows)


    @triton.jit
    def q8_grouped_reduce_kernel(partial_ptr, maximum_ptr, sum_ptr, out_ptr, SPLITS: tl.constexpr, D: tl.constexpr):
        row = tl.program_id(0)
        splits = tl.arange(0, SPLITS)
        dims = tl.arange(0, D)
        maximum = tl.load(maximum_ptr + row * SPLITS + splits)
        sums = tl.load(sum_ptr + row * SPLITS + splits)
        weights = tl.exp2((maximum - tl.max(maximum, 0)) * 1.4426950408889634)
        denominator = tl.sum(sums * weights, 0)
        partial = tl.load(partial_ptr + (row * SPLITS + splits[:, None]) * D + dims[None, :])
        result = tl.sum(partial * weights[:, None], 0) / tl.maximum(denominator, 1.0e-20)
        tl.store(out_ptr + row * D + dims, result)


def _q8_grouped_attention(q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, softmax_scale, tree_mask=None):
    """tree_mask: per verify row, int64 bits of the verify rows it sees (tree verification); None: causal rows."""
    queries_per_sequence = q.shape[0] // context_lens.numel()
    groups = q.shape[1] // k_cache.shape[2]
    query_rows = queries_per_sequence * groups
    block_m = 32 if query_rows > 16 else 16
    query_blocks = triton.cdiv(query_rows, block_m)
    # Share K/V reads across more query rows without increasing split buffers.
    blocks = context_lens.numel() * k_cache.shape[2] * triton.cdiv(query_rows, 16)
    # Split the context to keep several thread blocks per SM even at batch one.
    splits = min(128, triton.next_power_of_2(triton.cdiv(4 * torch.cuda.get_device_properties(q.device).multi_processor_count, blocks)))
    partial = torch.empty((*q.shape[:2], splits, q.shape[2]), device=q.device, dtype=torch.float32)
    maximum = torch.empty((*q.shape[:2], splits), device=q.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    output = torch.empty_like(q)
    sm120 = _SM120_Q8 if tree_mask is None else _sm120_tree_kernels()  # the wheel's precompiled SM120 kernel masks chains only
    if sm120 is not None and _use_sm120_q8(q):
        _log_kv_attention_backend_once("sm120_grouped_q8" + ("" if tree_mask is None else "_tree"), f"[Deepy][KV cache] SM120 asynchronous copies enabled for Q8 {'decode/verification' if tree_mask is None else 'tree verification'} ({sm120.__name__}).")
        if tree_mask is None:
            sm120.q8_grouped_partials(q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, partial, maximum, denominator, splits, softmax_scale)
        else:
            sm120.q8_grouped_partials(q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, partial, maximum, denominator, splits, softmax_scale, tree_mask)
    else:
        q8_grouped_partials_kernel[(context_lens.numel(), k_cache.shape[2], query_blocks * splits)](
            q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, partial, maximum, denominator, context_lens if tree_mask is None else tree_mask,
            queries_per_sequence, q.shape[1], k_cache.shape[2], q.shape[2], k_cache.shape[1], block_tables.shape[1], splits, block_m, 32, softmax_scale, tree_mask is not None,
            num_warps=4, num_stages=1)
    q8_grouped_reduce_kernel[(q.shape[0] * q.shape[1],)](partial, maximum, denominator, output, splits, q.shape[2], num_warps=4)
    return output


def _repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(num_repeats, dim=1)


def _sdpa_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    attention_bias: torch.Tensor | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    # Avoid enable_gqa here: on the local Torch 2.10 + CUDA 13 build it can pick
    # a much higher-workspace backend than the explicit-repeat SDPA path.
    key_states = _repeat_kv(key_states, num_key_value_groups)
    value_states = _repeat_kv(value_states, num_key_value_groups)
    if attention_bias is not None and attention_bias.dtype != torch.bool and attention_bias.dtype != query_states.dtype:
        attention_bias = attention_bias.to(dtype=query_states.dtype)
    return F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_bias,
        dropout_p=0.0,
        is_causal=bool(is_causal),
        scale=scaling,
    )


def _build_causal_bias(query_len: int, key_len: int, device: torch.device, query_offset: int = 0) -> torch.Tensor:
    q_pos = torch.arange(query_len, device=device, dtype=torch.long) + int(query_offset)
    k_pos = torch.arange(key_len, device=device, dtype=torch.long)
    valid = q_pos[:, None] >= k_pos[None, :]
    bias = torch.zeros((query_len, key_len), dtype=torch.float32, device=device)
    bias.masked_fill_(~valid, float("-inf"))
    return bias

def _efficient_causal_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """Bottom-right causal attention of one sequence on GPUs without FlashAttention (Turing): PyTorch's memory-efficient kernel in FP32
    in one call, which never materializes the scores and skips the masked tiles. The query heads sharing a KV head are folded into the
    batch against a stride-0 view of that head: no copy of the keys and values per query head. query (1, heads, L, d), key and value
    (1, kv_heads, S, d) in any float dtype; returns query's dtype."""
    _, heads, length, dim = query.shape
    kv_heads, keys = key.shape[1:3]
    groups = heads // kv_heads
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    query = query.to(torch.float32, memory_format=torch.contiguous_format).view(kv_heads, groups, length, dim).transpose(0, 1)
    key, value = key.float().expand(groups, -1, -1, -1), value.float().expand(groups, -1, -1, -1)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=causal_lower_right(length, keys), scale=scale)
    del query, key, value
    output.view(kv_heads, groups, length, dim).transpose(0, 1).copy_(result)
    return output


def _gather_cache_tokens(cache: torch.Tensor, block_table: torch.Tensor, seq_len: int) -> torch.Tensor:
    if seq_len <= 0:
        return cache.new_empty((0, cache.shape[-2], cache.shape[-1]))
    block_size = int(cache.shape[1])
    num_blocks = (int(seq_len) + block_size - 1) // block_size
    valid_block_ids = [int(block_id) for block_id in block_table[:num_blocks].tolist() if int(block_id) >= 0]
    gathered = cache[torch.tensor(valid_block_ids, device=cache.device, dtype=torch.long)]
    return gathered.reshape(-1, cache.shape[-2], cache.shape[-1])[:seq_len]


def _flash_attention_fallback_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    num_key_value_groups: int,
    context,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    efficient: bool = False,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    outputs = []
    use_prefix_cache = context.block_tables is not None
    for idx in range(int(context.cu_seqlens_q.numel()) - 1):
        q_start = int(context.cu_seqlens_q[idx].item())
        q_end = int(context.cu_seqlens_q[idx + 1].item())
        k_start = int(context.cu_seqlens_k[idx].item())
        k_end = int(context.cu_seqlens_k[idx + 1].item())
        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len <= 0:
            continue
        q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        if efficient:
            if use_prefix_cache:  # the occupied pages only, dequantized straight to the FP32 the attention computes in
                table = context.block_tables[idx]
                keys = _gather_cache_tokens(k_cache, table, k_len)
                values = _gather_cache_tokens(v_cache, table, k_len)
                if k_scale is not None:
                    keys = _dequantize_kvcache(keys, _gather_cache_tokens(k_scale, table, k_len), torch.float32)
                    values = _dequantize_kvcache(values, _gather_cache_tokens(v_scale, table, k_len), torch.float32)
            else:
                keys, values = k[k_start:k_end], v[k_start:k_end]
            output = _efficient_causal_attention(q_i, keys.transpose(0, 1).unsqueeze(0), values.transpose(0, 1).unsqueeze(0), scale)
            outputs.append(output.squeeze(0).transpose(0, 1))
            continue
        if use_prefix_cache:
            k_i = _gather_cache_tokens(k_cache, context.block_tables[idx], k_len).transpose(0, 1).unsqueeze(0)
            v_i = _gather_cache_tokens(v_cache, context.block_tables[idx], k_len).transpose(0, 1).unsqueeze(0)
            query_offset = k_len - q_len
            bias = _build_causal_bias(q_len, k_len, q.device, query_offset=query_offset).view(1, 1, q_len, k_len)
            output = _sdpa_attention(q_i, k_i, v_i, scale, num_key_value_groups, attention_bias=bias, is_causal=False)
        else:
            k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
            v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
            output = _sdpa_attention(q_i, k_i, v_i, scale, num_key_value_groups)
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)


def _flash_attention_fallback_decode(
    q: torch.Tensor,
    scale: float,
    num_key_value_groups: int,
    context,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    if q.is_cuda and context.block_tables is not None and context.context_lens is not None:
        batch_size = int(q.shape[0])
        max_num_blocks = int(context.block_tables.shape[1])
        block_size = int(k_cache.shape[1])
        total_cache_tokens = max_num_blocks * block_size
        flat_block_ids = context.block_tables.clamp_min(0).reshape(-1).long()
        k_tokens = k_cache.index_select(0, flat_block_ids).reshape(batch_size, max_num_blocks, block_size, k_cache.shape[-2], k_cache.shape[-1]).reshape(batch_size, total_cache_tokens, k_cache.shape[-2], k_cache.shape[-1])
        v_tokens = v_cache.index_select(0, flat_block_ids).reshape(batch_size, max_num_blocks, block_size, v_cache.shape[-2], v_cache.shape[-1]).reshape(batch_size, total_cache_tokens, v_cache.shape[-2], v_cache.shape[-1])
        block_mask = context.block_tables.ge(0).unsqueeze(-1).expand(-1, -1, block_size).reshape(batch_size, total_cache_tokens)
        token_positions = torch.arange(total_cache_tokens, device=q.device, dtype=context.context_lens.dtype).unsqueeze(0)
        valid_tokens = block_mask & token_positions.lt(context.context_lens.unsqueeze(1))
        has_tokens = valid_tokens.any(dim=1, keepdim=True)
        safe_valid_tokens = valid_tokens | (~has_tokens & token_positions.eq(0))
        kv_valid_mask = safe_valid_tokens.unsqueeze(-1).unsqueeze(-1)
        # The last active block may be only partially written. Zero invalid slots by replacement
        # so unwritten `torch.empty` cache contents cannot leak NaNs into SDPA.
        k_tokens = torch.where(kv_valid_mask, k_tokens, torch.zeros((), dtype=k_tokens.dtype, device=k_tokens.device))
        v_tokens = torch.where(kv_valid_mask, v_tokens, torch.zeros((), dtype=v_tokens.dtype, device=v_tokens.device))
        attention_bias = safe_valid_tokens.unsqueeze(1).unsqueeze(1)
        attn_output = _sdpa_attention(
            q.unsqueeze(2),
            k_tokens.permute(0, 2, 1, 3),
            v_tokens.permute(0, 2, 1, 3),
            scale,
            num_key_value_groups,
            attention_bias=attention_bias,
            is_causal=False,
        ).transpose(1, 2)
        return attn_output * has_tokens.view(batch_size, 1, 1, 1).to(attn_output.dtype)

    outputs = []
    for idx in range(int(q.shape[0])):
        seq_len = int(context.context_lens[idx].item())
        q_i = q[idx:idx + 1].transpose(0, 1).unsqueeze(0)
        k_i = _gather_cache_tokens(k_cache, context.block_tables[idx], seq_len).transpose(0, 1).unsqueeze(0)
        v_i = _gather_cache_tokens(v_cache, context.block_tables[idx], seq_len).transpose(0, 1).unsqueeze(0)
        outputs.append(_sdpa_attention(q_i, k_i, v_i, scale, num_key_value_groups, is_causal=False).transpose(1, 2))
    return torch.cat(outputs, dim=0)


def _flash_attention_fallback_speculative(
    q: torch.Tensor,
    scale: float,
    num_key_value_groups: int,
    context,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    if q.is_cuda:
        query_length = int(q.shape[0])
        block_size = int(k_cache.shape[1])
        max_num_blocks = int(context.block_tables.shape[1])
        total_cache_tokens = max_num_blocks * block_size
        block_ids = context.block_tables[0].clamp_min(0).long()
        keys = k_cache.index_select(0, block_ids).reshape(total_cache_tokens, k_cache.shape[-2], k_cache.shape[-1])
        values = v_cache.index_select(0, block_ids).reshape(total_cache_tokens, v_cache.shape[-2], v_cache.shape[-1])
        token_positions = torch.arange(total_cache_tokens, device=q.device, dtype=context.cu_seqlens_k.dtype)
        sequence_length = context.cu_seqlens_k[-1]
        cache_valid = context.block_tables[0].ge(0).unsqueeze(-1).expand(-1, block_size).reshape(total_cache_tokens) & token_positions.lt(sequence_length)
        kv_valid_mask = cache_valid.unsqueeze(-1).unsqueeze(-1)
        keys = torch.where(kv_valid_mask, keys, torch.zeros((), dtype=keys.dtype, device=keys.device))
        values = torch.where(kv_valid_mask, values, torch.zeros((), dtype=values.dtype, device=values.device))
        query_positions = sequence_length - query_length + torch.arange(query_length, device=q.device, dtype=sequence_length.dtype)
        attention_bias = (cache_valid.unsqueeze(0) & token_positions.unsqueeze(0).le(query_positions.unsqueeze(1))).unsqueeze(0).unsqueeze(0)
        return _sdpa_attention(q.transpose(0, 1).unsqueeze(0), keys.transpose(0, 1).unsqueeze(0), values.transpose(0, 1).unsqueeze(0), scale, num_key_value_groups, attention_bias=attention_bias, is_causal=False).squeeze(0).transpose(0, 1)

    sequence_length = int(context.cu_seqlens_k[-1].item())
    query_length = int(q.shape[0])
    keys = _gather_cache_tokens(k_cache, context.block_tables[0], sequence_length).transpose(0, 1).unsqueeze(0)
    values = _gather_cache_tokens(v_cache, context.block_tables[0], sequence_length).transpose(0, 1).unsqueeze(0)
    bias = _build_causal_bias(query_length, sequence_length, q.device, query_offset=sequence_length - query_length).view(1, 1, query_length, sequence_length)
    return _sdpa_attention(q.transpose(0, 1).unsqueeze(0), keys, values, scale, num_key_value_groups, attention_bias=bias, is_causal=False).squeeze(0).transpose(0, 1)


if triton is not None:
    @triton.jit
    def _store_group(x, data_ptr, scale_ptr, min_ptr, BITS: tl.constexpr, ASYM: tl.constexpr):
        """One group of 32 values quantized to the format (kv_formats) and stored: data at data_ptr, FP16 scale and min at their pointers."""
        if BITS == 8:
            scale = tl.maximum(tl.max(tl.abs(x), axis=0) * (1.0 / 127.0), 1e-8)  # as PyTorch: a scalar divisor multiplies by its reciprocal, elementwise divisions are exact
            tl.store(data_ptr + tl.arange(0, 32), tl_libdevice.nearbyint(tl.div_rn(x, scale)).to(tl.int8))
            tl.store(scale_ptr, scale.to(tl.float16))
        else:
            lanes = tl.arange(0, 32)
            if ASYM:
                lo = tl.min(x, axis=0)
                d = (tl.max(x, axis=0) - lo) * (1.0 / 15.0)
                inverse = tl.where(d > 0, tl.div_rn(1.0, d), 0.0)
                q = tl.minimum(tl.maximum(tl_libdevice.nearbyint((x - lo) * inverse), 0.0), 15.0).to(tl.int32)
                tl.store(scale_ptr, d.to(tl.float16))
                tl.store(min_ptr, lo.to(tl.float16))
            else:
                HALF: tl.constexpr = 1 << (BITS - 1)
                largest = tl.max(tl.abs(x), axis=0)
                first = tl.min(tl.where(tl.abs(x) == largest, lanes, 32), axis=0)  # the first extreme, as llama.cpp and torch.argmax
                extreme = tl.sum(tl.where(lanes == first, x, 0.0), axis=0)
                d = extreme * (-1.0 / HALF)
                inverse = tl.where(d != 0, tl.div_rn(1.0, d), 0.0)
                q = tl.minimum(tl.maximum(tl_libdevice.nearbyint(x * inverse) + HALF, 0.0), 2.0 * HALF - 1.0).to(tl.int32)
                tl.store(scale_ptr, d.to(tl.float16))
            first, second = tl.split(tl.trans(tl.reshape(q, (2, 16))))  # values 0..15 and 16..31
            tl.store(data_ptr + tl.arange(0, 16), ((first & 15) | ((second & 15) << 4)).to(tl.uint8))
            if BITS == 6:
                high = tl.reshape(q >> 4, (2, 4, 4))  # (half, r, b)
                plane = tl.sum(high << (2 * tl.arange(0, 4))[None, :, None], axis=1)  # (half, b)
                tl.store(data_ptr + 16 + tl.arange(0, 8), tl.reshape(plane, (8,)).to(tl.uint8))


    @triton.jit
    def store_mixed_kvcache_kernel(
        key_ptr, key_stride_token, key_stride_head, value_ptr, value_stride_token, value_stride_head,
        k_body_ptr, v_body_ptr, k_scale_ptr, v_scale_ptr, k_min_ptr, v_min_ptr,
        k_overlay_ptr, v_overlay_ptr, k_overlay_scale_ptr, v_overlay_scale_ptr, slot_mapping_ptr, overlay_mapping_ptr,
        H: tl.constexpr, D: tl.constexpr, QB: tl.constexpr, K_BITS: tl.constexpr, K_ASYM: tl.constexpr, V_BITS: tl.constexpr, V_ASYM: tl.constexpr,
        OVERLAY: tl.constexpr, BF16_BITS: tl.constexpr = False,
    ):
        """A token's key and value group into the body in its formats and, for a token with an overlay row, into the exact Q8 overlays."""
        idx = tl.program_id(0)
        token_idx = idx // (H * QB)
        head_block_idx = idx % (H * QB)
        head_idx = head_block_idx // QB
        quant_block_idx = head_block_idx % QB
        slot = tl.load(slot_mapping_ptr + token_idx)
        if slot == -1: return
        offsets = quant_block_idx * 32 + tl.arange(0, 32)
        key = _load_fp32(key_ptr + token_idx * key_stride_token + head_idx * key_stride_head + offsets, BF16_BITS)
        value = _load_fp32(value_ptr + token_idx * value_stride_token + head_idx * value_stride_head + offsets, BF16_BITS)
        K_BYTES: tl.constexpr = 32 * K_BITS // 8
        V_BYTES: tl.constexpr = 32 * V_BITS // 8
        scale_offset = (slot * H + head_idx) * QB + quant_block_idx
        _store_group(key, k_body_ptr + (slot * H + head_idx) * (QB * K_BYTES) + quant_block_idx * K_BYTES, k_scale_ptr + scale_offset, k_min_ptr + scale_offset, K_BITS, K_ASYM)
        _store_group(value, v_body_ptr + (slot * H + head_idx) * (QB * V_BYTES) + quant_block_idx * V_BYTES, v_scale_ptr + scale_offset, v_min_ptr + scale_offset, V_BITS, V_ASYM)
        if OVERLAY:
            row = tl.load(overlay_mapping_ptr + token_idx)
            if row >= 0:
                overlay_scale = (row * H + head_idx) * QB + quant_block_idx
                _store_group(key, k_overlay_ptr + (row * H + head_idx) * D + quant_block_idx * 32, k_overlay_scale_ptr + overlay_scale, k_min_ptr, 8, False)
                _store_group(value, v_overlay_ptr + (row * H + head_idx) * D + quant_block_idx * 32, v_overlay_scale_ptr + overlay_scale, v_min_ptr, 8, False)


def _store_mixed(key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale, k_min, v_min, k_overlay, v_overlay, k_overlay_scale, v_overlay_scale, overlay_mapping, layout, use_triton):
    N, num_heads, head_dim = key.shape
    quant_blocks = head_dim // _Q8_KV_BLOCK_SIZE
    overlay = k_overlay is not None and overlay_mapping is not None
    if use_triton and key.is_cuda:
        bits = key.dtype == torch.bfloat16 and not dispatch.triton_dtype_supported(key.device, torch.bfloat16)
        dummy = k_scale
        store_mixed_kvcache_kernel[(N * num_heads * quant_blocks,)](
            key.view(torch.int16) if bits else key, key.stride(0), key.stride(1), value.view(torch.int16) if bits else value, value.stride(0), value.stride(1),
            k_cache, v_cache, k_scale, v_scale, dummy if k_min is None else k_min, dummy if v_min is None else v_min,
            k_overlay if overlay else k_cache, v_overlay if overlay else v_cache, k_overlay_scale if overlay else k_scale, v_overlay_scale if overlay else v_scale,
            slot_mapping, overlay_mapping if overlay else slot_mapping,
            num_heads, head_dim, quant_blocks, layout.k.bits, layout.k.asymmetric, layout.v.bits, layout.v.asymmetric, overlay, BF16_BITS=bits)
        torch.autograd.graph.increment_version(tuple(t for t in (k_cache, v_cache, k_scale, v_scale, k_min, v_min, k_overlay, v_overlay, k_overlay_scale, v_overlay_scale) if t is not None))
        return
    valid = slot_mapping >= 0
    capturing = slot_mapping.is_cuda and torch.cuda.is_current_stream_capturing()
    if not capturing and not bool(torch.any(valid)):
        return
    slots = slot_mapping.long() if capturing else slot_mapping[valid].long()
    for x, fmt, body, scale, minimum in ((key, layout.k, k_cache, k_scale, k_min), (value, layout.v, v_cache, v_scale, v_min)):
        data, scales, mins = kvf.quantize_heads(x if capturing else x[valid], fmt)
        body.reshape(-1, num_heads, body.shape[-1]).index_copy_(0, slots, data)
        scale.reshape(-1, num_heads, quant_blocks).index_copy_(0, slots, scales)
        if minimum is not None:
            minimum.reshape(-1, num_heads, quant_blocks).index_copy_(0, slots, mins)
    if overlay:
        rows = overlay_mapping if capturing else overlay_mapping[valid]
        rows = torch.where(rows >= 0, rows, torch.full_like(rows, k_overlay.shape[0] - 1)).long()  # the scratch row takes the tokens without overlay
        for x, body, scale in ((key, k_overlay, k_overlay_scale), (value, v_overlay, v_overlay_scale)):
            data, scales, _ = kvf.quantize_heads(x if capturing else x[valid], kvf.Q8)
            body.reshape(-1, num_heads, head_dim).index_copy_(0, rows, data)
            scale.reshape(-1, num_heads, quant_blocks).index_copy_(0, rows, scales)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    use_triton_kv_cache: bool | None = None,
    *,
    k_min: torch.Tensor | None = None,
    v_min: torch.Tensor | None = None,
    k_overlay: torch.Tensor | None = None,
    v_overlay: torch.Tensor | None = None,
    k_overlay_scale: torch.Tensor | None = None,
    v_overlay_scale: torch.Tensor | None = None,
    overlay_mapping: torch.Tensor | None = None,
    layout=None,
):
    assert slot_mapping.numel() == key.shape[0]
    if use_triton_kv_cache is None:
        use_triton_kv_cache = _USE_TRITON_KV_CACHE
    if layout is not None and layout.mixed:
        return _store_mixed(key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale, k_min, v_min, k_overlay, v_overlay, k_overlay_scale, v_overlay_scale, overlay_mapping, layout, use_triton_kv_cache)
    quantized = k_cache.dtype == torch.int8
    if quantized and (k_scale is None or v_scale is None):
        raise RuntimeError("INT8 KV cache requires key and value scales.")
    if use_triton_kv_cache is None:
        use_triton_kv_cache = _USE_TRITON_KV_CACHE
    if use_triton_kv_cache:
        N, num_heads, head_dim = key.shape
        if quantized:
            if head_dim % _Q8_KV_BLOCK_SIZE:
                raise RuntimeError(f"INT8 KV cache head dimension must be divisible by {_Q8_KV_BLOCK_SIZE}; got {head_dim}.")
            quant_blocks = head_dim // _Q8_KV_BLOCK_SIZE
            bits = key.dtype == torch.bfloat16 and not dispatch.triton_dtype_supported(key.device, torch.bfloat16)  # Turing: BF16 read as bits
            store_int8_kvcache_kernel[(N * num_heads * quant_blocks,)](
                key.view(torch.int16) if bits else key, key.stride(0), key.stride(1), value.view(torch.int16) if bits else value, value.stride(0), value.stride(1),
                k_cache, v_cache, k_scale, v_scale, slot_mapping, num_heads, head_dim, quant_blocks, BF16_BITS=bits)
            return
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
        torch.autograd.graph.increment_version((k_cache, v_cache))
        return

    if quantized:
        quant_blocks = key.shape[-1] // _Q8_KV_BLOCK_SIZE
        key_blocks = key.reshape(*key.shape[:-1], quant_blocks, _Q8_KV_BLOCK_SIZE)
        value_blocks = value.reshape(*value.shape[:-1], quant_blocks, _Q8_KV_BLOCK_SIZE)
        # Quantize with FP32 scales, then round stored scales, as the native/Triton format does.
        key_blocks, value_blocks = key_blocks.float(), value_blocks.float()
        key_scale = key_blocks.abs().amax(dim=-1).div_(127).clamp_min_(1e-8)
        value_scale = value_blocks.abs().amax(dim=-1).div_(127).clamp_min_(1e-8)
        key = key_blocks.div(key_scale.unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).reshape_as(key)
        value = value_blocks.div(value_scale.unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).reshape_as(value)
        key_scale, value_scale = key_scale.to(k_scale.dtype), value_scale.to(v_scale.dtype)

    if slot_mapping.is_cuda and torch.cuda.is_current_stream_capturing():
        flat_k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        flat_v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
        slot_ids = slot_mapping.long()
        flat_k_cache.index_copy_(0, slot_ids, key)
        flat_v_cache.index_copy_(0, slot_ids, value)
        if quantized:
            k_scale.reshape(-1, k_scale.shape[-2], k_scale.shape[-1]).index_copy_(0, slot_ids, key_scale)
            v_scale.reshape(-1, v_scale.shape[-2], v_scale.shape[-1]).index_copy_(0, slot_ids, value_scale)
        return

    valid_mask = slot_mapping >= 0
    if not torch.any(valid_mask):
        return
    flat_k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
    flat_v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
    slot_ids = slot_mapping[valid_mask].long()
    flat_k_cache[slot_ids] = key[valid_mask]
    flat_v_cache[slot_ids] = value[valid_mask]
    if quantized:
        k_scale.reshape(-1, k_scale.shape[-2], k_scale.shape[-1])[slot_ids] = key_scale[valid_mask]
        v_scale.reshape(-1, v_scale.shape[-2], v_scale.shape[-1])[slot_ids] = value_scale[valid_mask]


def _dequantize_kvcache(cache: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return cache.reshape(*cache.shape[:-1], scale.shape[-1], _Q8_KV_BLOCK_SIZE).to(dtype).mul_(scale.to(dtype).unsqueeze(-1)).reshape_as(cache)


def _dense_body(cache: torch.Tensor, scale: torch.Tensor, minimum: torch.Tensor | None, fmt, dtype: torch.dtype) -> torch.Tensor:
    """The whole paged body dequantized to [pages, page, H, D] in dtype."""
    return _dequantize_kvcache(cache, scale, dtype) if fmt.q8 else kvf.dequantize_heads(cache, scale, minimum, fmt, dtype)


def _apply_overlays(dense: torch.Tensor, overlay: torch.Tensor, overlay_scale: torch.Tensor, context, layout, dtype: torch.dtype) -> None:
    """The exact Q8 overlay rows of every sequence written over its sink and ring positions of the dense cache (fallback paths only)."""
    page = int(dense.shape[1])
    sink, recent = layout.sink_tokens(page), layout.recent_tokens
    lengths = (context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]) if context.context_lens is None else context.context_lens
    for index, (base, start, length) in enumerate(zip(context.overlay_base.tolist(), context.ring_start.tolist(), lengths.tolist())):
        if base < 0:
            continue
        positions = torch.cat((torch.arange(0, min(length, sink), device=dense.device), torch.arange(min(start, length), length, device=dense.device)))
        if positions.numel() == 0:
            continue
        rows = base + torch.where(positions < sink, positions, sink + (positions & (recent - 1)))
        pages = context.block_tables[index][positions // page].long()
        dense[pages, positions % page] = _dequantize_kvcache(overlay[rows], overlay_scale[rows], dtype)


def _q8_paged_prefill(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor, context, softmax_scale: float) -> torch.Tensor:
    output = torch.empty_like(q)
    query_lengths = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
    grid = (query_lengths.numel(), q.shape[1], triton.cdiv(q.shape[0], 32))
    if torch.version.hip is None:
        _log_kv_attention_backend_once("exact_prefill_q8", "[Deepy][KV cache] Q8 prefix prefill uses exact INT8 tensor-core products with per-block scales.")
        q8_prefill_exact_kernel[grid](
            q, k_cache, v_cache, k_scale, v_scale, context.block_tables, context.cu_seqlens_q, context.cu_seqlens_k, output,
            q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), context.block_tables.stride(0), output.stride(0), output.stride(1), softmax_scale,
            q.shape[1], k_cache.shape[2], q.shape[2], k_cache.shape[1], BLOCK_M=32, BLOCK_N=32, num_warps=4, num_stages=_Q8_EXACT_PREFILL_STAGES,
        )
        return output
    q8_paged_prefill_kernel[grid](
        q, k_cache, v_cache, k_scale, v_scale, context.block_tables, context.cu_seqlens_q, context.cu_seqlens_k, output,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), context.block_tables.stride(0), output.stride(0), output.stride(1), softmax_scale,
        q.shape[1], k_cache.shape[2], q.shape[2], k_cache.shape[1], BLOCK_M=32, BLOCK_N=32, num_warps=4, num_stages=1,
    )
    return output


def _q8_paged_speculative(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    return _Q8_PAGED_ATTENTION(q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, softmax_scale).squeeze(1)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.flash_attn_varlen_func = _DEFAULT_FLASH_ATTN_VARLEN_FUNC
        self.flash_attn_with_kvcache = _DEFAULT_FLASH_ATTN_WITH_KVCACHE
        # the LM features' Triton switch: each call then takes the Triton kernels the GPU runs for q's dtype (native, or BF16 read as bits on Turing)
        self.use_triton_kv_cache = _DEFAULT_USE_TRITON_KV_CACHE
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = torch.tensor([])
        self._q8_speculative_metadata = {}
        self.kv_layout = None  # kv_formats.KVCacheLayout of a quantized cache (set by the runner)
        self.k_min = self.v_min = None  # FP16 minimums of asymmetric body formats
        self.k_overlay = self.v_overlay = None  # exact Q8 overlays (sink and recent ring) of a mixed layout, [rows, H, D]
        self.k_overlay_scale = self.v_overlay_scale = None
        self.dense_prefill = False  # opt-in (set by the runner): prefill by FlashAttention over the quantized cache dequantized to FP16

    def prepare_speculative_metadata(self, max_query_count: int, max_num_blocks: int) -> None:
        if self.k_cache.dtype != torch.int8 or not self.k_cache.is_cuda or _Q8_PAGED_ATTENTION is None:
            return
        if self.use_triton_kv_cache and triton is not None and self.head_dim in (32, 64, 128, 256) and dispatch.triton_dtype_supported(self.k_cache.device, torch.bfloat16):
            return
        # Allocate before graph-state registration; eager and captured calls share fixed-width tables.
        self._q8_speculative_metadata = {
            count: (
                torch.arange(1, count + 1, dtype=torch.int32, device=self.k_cache.device),
                torch.empty(count, dtype=torch.int32, device=self.k_cache.device),
                torch.empty(count, max_num_blocks, dtype=torch.int32, device=self.k_cache.device),
            )
            for count in range(1, max_query_count + 1)  # a verification without drafted tokens left has one query
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, cache_written=False):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if not cache_written and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.k_scale, self.v_scale, use_triton_kv_cache=self.use_triton_kv_cache,
                          k_min=self.k_min, v_min=self.v_min, k_overlay=self.k_overlay, v_overlay=self.v_overlay, k_overlay_scale=self.k_overlay_scale,
                          v_overlay_scale=self.v_overlay_scale, overlay_mapping=context.overlay_mapping, layout=self.kv_layout)
        quantized_cache = k_cache.dtype in (torch.int8, torch.uint8)
        mixed = quantized_cache and self.kv_layout is not None and self.kv_layout.mixed
        q8_kernels = quantized_cache and not mixed  # the Q8 kernels never read a mixed body: attention_mixed or the dense fallbacks do
        reads_cache = context.speculative_verify or not context.is_prefill or context.block_tables is not None
        q8_attention = _Q8_PAGED_ATTENTION if q8_kernels and not mixed and not context.speculative_verify and not context.is_prefill else None
        use_triton = self.use_triton_kv_cache and triton is not None and dispatch.triton_dtype_supported(q.device, q.dtype)
        bits = self.use_triton_kv_cache and triton is not None and bf16_bits.applies(q)  # Turing: the kernels reading BF16 q as bits
        q8_grouped = q8_kernels and use_triton and self.head_dim in (32, 64, 128, 256) and (context.speculative_verify or not context.is_prefill)
        q8_prefill = q8_kernels and context.is_prefill and context.block_tables is not None and use_triton
        q8_speculative = q8_kernels and not mixed and context.speculative_verify and _Q8_PAGED_ATTENTION is not None
        if context.tree_mask is not None and (bits or not q8_grouped or not (q.is_cuda and q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16))):
            raise RuntimeError("Tree verification needs the grouped Q8 paged attention (a Q8 KV cache with Triton): this layer's backend only masks a chain.")
        if bits and quantized_cache and not mixed and (context.speculative_verify or not context.is_prefill) and q.dtype == torch.bfloat16 and self.head_dim == 256 and q.is_contiguous():
            _log_kv_attention_backend_once("bf16_bits_q8", "[Deepy][KV cache] decode/verification backend=grouped Q8 paged attention for Turing (BF16 read as bits, FP16 tensor cores, FP32 accumulation).")
            context_lens = context.cu_seqlens_k[1:] if context.speculative_verify else context.context_lens
            output = bf16_bits.q8_paged_attention(q, k_cache, v_cache, self.k_scale, self.v_scale, context.block_tables, context_lens, self.scale)
            return output if context.speculative_verify else output.unsqueeze(1)
        if (self.dense_prefill and quantized_cache and context.is_prefill and not context.speculative_verify and context.block_tables is not None and q.shape[0] >= _DENSE_PREFILL_ROWS
                and self.flash_attn_varlen_func is not None and use_triton and attention_mixed.available(q, self, prefill=True)):
            output = self._dense_prefill(q, context)
            if output is not None:
                return output
        if mixed and _MIXED_KERNELS and (use_triton or bits) and attention_mixed.available(q, self, prefill=context.is_prefill and not context.speculative_verify):
            if context.speculative_verify or not context.is_prefill:
                _log_kv_attention_backend_once("mixed_grouped", f"[Deepy][KV cache] decode/verification backend=grouped paged attention over the {self.kv_layout.value} cache (Triton, FP32 accumulation).")
                output = attention_mixed.grouped_attention(q, self, context, self.scale)
                return output if context.speculative_verify else output.unsqueeze(1)
            if context.block_tables is not None:
                _log_kv_attention_backend_once("mixed_prefill", f"[Deepy][KV cache] prefix prefill backend=exact integer paged attention over the {self.kv_layout.value} cache (Triton).")
                return attention_mixed.prefill(q, self, context, self.scale)
        if (q8_attention is not None or q8_grouped or q8_prefill or q8_speculative) and q.is_cuda and q.is_contiguous() and q.dtype in (torch.float16, torch.bfloat16) and self.head_dim <= 256 and self.head_dim % _Q8_KV_BLOCK_SIZE == 0:
            dtype_name = "BF16" if q.dtype == torch.bfloat16 else "FP16"
            if q8_grouped:
                _log_kv_attention_backend_once("grouped_triton_q8", f"[Deepy][KV cache] decode/verification backend=grouped Q8 paged attention ({dtype_name} I/O, FP32 accumulation, no dense KV materialization).")
                context_lens = context.cu_seqlens_k[1:] if context.speculative_verify else context.context_lens
                output = _q8_grouped_attention(q, k_cache, v_cache, self.k_scale, self.v_scale, context.block_tables, context_lens, self.scale, context.tree_mask)
                return output if context.speculative_verify else output.unsqueeze(1)
            if context.speculative_verify:
                _log_kv_attention_backend_once("speculative_llamacpp_q8", f"[Deepy][Speculative] verification backend=batched llama.cpp fattn-vec Q8 paged attention ({dtype_name} I/O, FP32 accumulation, no dense KV materialization).")
                query_count = q.shape[0]
                query_offsets, context_lens, block_tables = self._q8_speculative_metadata[query_count]
                context_lens.copy_(query_offsets).add_(context.cu_seqlens_k[1] - query_count)
                block_tables.fill_(-1)
                block_tables[:, :context.block_tables.shape[1]].copy_(context.block_tables.expand(query_count, -1))
                return _q8_paged_speculative(q, k_cache, v_cache, self.k_scale, self.v_scale, context_lens, block_tables, self.scale)
            elif context.is_prefill:
                _log_kv_attention_backend_once("prefill_triton_q8", f"[Deepy][KV cache] prefix prefill backend=tiled Q8 paged attention ({dtype_name} I/O, FP32 accumulation, no dense KV materialization).")
                return _q8_paged_prefill(q, k_cache, v_cache, self.k_scale, self.v_scale, context, self.scale)
            else:
                _log_kv_attention_backend_once("llamacpp_q8", f"[Deepy][KV cache] decode backend=llama.cpp fattn-vec Q8 paged adapter ({dtype_name} I/O, FP32 accumulation).")
            context_lens = context.cu_seqlens_k[1:] if context.speculative_verify else context.context_lens
            output = q8_attention(q, k_cache, v_cache, self.k_scale, self.v_scale, context.block_tables, context_lens, self.scale)
            return output.squeeze(1) if context.speculative_verify else output
        # GPUs without BF16 arithmetic (pre-Ampere NVIDIA) and without FlashAttention: one memory-efficient FP32 SDPA per sequence
        if (context.is_prefill and not context.speculative_verify and self.flash_attn_varlen_func is None
                and q.is_cuda and not dispatch.triton_dtype_supported(q.device, q.dtype)):
            # Gather occupied pages before dequantizing on the hardware-selected
            # SDPA route, rather than expanding the entire reserved Q8 cache.
            if mixed:
                dense_k, dense_v = self._dense_caches(torch.float32, context)
                return _flash_attention_fallback_prefill(q, k, v, self.scale, self.num_heads // self.num_kv_heads, context, dense_k, dense_v, efficient=True)
            return _flash_attention_fallback_prefill(
                q, k, v, self.scale, self.num_heads // self.num_kv_heads, context, k_cache, v_cache,
                efficient=True,
                k_scale=self.k_scale if quantized_cache else None, v_scale=self.v_scale if quantized_cache else None,
            )
        attention_k_cache, attention_v_cache = self._dense_caches(q.dtype, context) if quantized_cache and reads_cache else (k_cache, v_cache)
        # AMD Triton FA2 exposes the FlashAttention functions, but rejects their
        # block_table argument. Import success does not imply paged-cache support.
        # Keep ordinary (non-paged) FlashAttention prefill available on HIP.
        flash_paged_supported = torch.version.hip is None or context.block_tables is None
        if not flash_paged_supported:
            _log_kv_attention_backend_once(
                "hip_paged_sdpa",
                "[Deepy][KV cache] HIP Paged Attention Backend: PyTorch SDPA (FlashAttention Paged API Unsupported).",
            )
        if context.speculative_verify:
            if self.flash_attn_with_kvcache is None or not flash_paged_supported:
                detail = " with on-the-fly Q8 cache dequantization" if quantized_cache else ""
                _log_kv_attention_backend_once("speculative_sdpa", f"[Deepy][Speculative] verification backend=PyTorch SDPA{detail}.")
                return _flash_attention_fallback_speculative(q, self.scale, self.num_heads // self.num_kv_heads, context, attention_k_cache, attention_v_cache)
            detail = " with on-the-fly Q8 cache dequantization" if quantized_cache else ""
            _log_kv_attention_backend_once("speculative_flash", f"[Deepy][Speculative] verification backend=FlashAttention KV cache{detail}.")
            # The q=2 verification queries have already been written to the paged
            # cache. FlashAttention's KV-cache API replays this layout correctly;
            # its varlen prefill API does not preserve the first query on replay.
            return self.flash_attn_with_kvcache(
                q.unsqueeze(0),
                attention_k_cache,
                attention_v_cache,
                cache_seqlens=context.cu_seqlens_k[1:],
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            ).squeeze(0)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = attention_k_cache, attention_v_cache
            if self.flash_attn_varlen_func is not None and flash_paged_supported:
                o = self.flash_attn_varlen_func(q, k, v,
                                                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                                softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = _flash_attention_fallback_prefill(
                    q,
                    k,
                    v,
                    self.scale,
                    self.num_heads // self.num_kv_heads,
                    context,
                    attention_k_cache,
                    attention_v_cache,
                )
        else:    # decode
            if self.flash_attn_with_kvcache is not None and flash_paged_supported:
                if quantized_cache:
                    _log_kv_attention_backend_once("flash_q8_fallback", "[Deepy][KV cache] llama.cpp Q8 decode unavailable; backend=FlashAttention with on-the-fly cache dequantization.")
                o = self.flash_attn_with_kvcache(q.unsqueeze(1), attention_k_cache, attention_v_cache,
                                                 cache_seqlens=context.context_lens, block_table=context.block_tables,
                                                 softmax_scale=self.scale, causal=True)
            else:
                if quantized_cache:
                    _log_kv_attention_backend_once("sdpa_q8_fallback", "[Deepy][KV cache] llama.cpp Q8 decode and FlashAttention unavailable; backend=PyTorch SDPA with on-the-fly cache dequantization.")
                o = _flash_attention_fallback_decode(
                    q,
                    self.scale,
                    self.num_heads // self.num_kv_heads,
                    context,
                    attention_k_cache,
                    attention_v_cache,
                )
        return o

    def _dense_prefill(self, q: torch.Tensor, context) -> torch.Tensor | None:
        """Prefill attention of a chunk by FlashAttention over the keys and values of its context dequantized to FP16, about four times
        faster than the exact kernels over the quantized cache. Costs: 2 x H_KV x D FP16 values per context token while the call lasts, and
        operands rounded to FP16 (error of the output against an FP64 reference 1.73e-3 where the exact kernel's is 1.66e-3, the rounding of
        a BF16 output). None when a value leaves FP16's range: the exact kernel then serves the call."""
        _log_kv_attention_backend_once("dense_prefill", "[Deepy][KV cache] prefix prefill backend=FlashAttention over the cache dequantized to FP16 (faster prefill option).")
        keys, slice_keys = int(context.max_seqlen_k), max(_DENSE_PREFILL_KEYS, q.shape[0])
        if context.cu_seqlens_k.numel() != 2 or keys < 3 * slice_keys:  # several sequences, or a context that takes no more at once than by slices
            dense_k, dense_v = context.dense_kv = attention_mixed.dense(self, context, torch.float16, context.dense_kv)
            output = self.flash_attn_varlen_func(q.to(torch.float16), dense_k, dense_v, max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                                 max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k, softmax_scale=self.scale, causal=True)
            return output.to(q.dtype) if bool(torch.isfinite(output).all()) else None
        if context.dense_slices is None:  # the last slice holds the chunk's own keys (causal); every query sees the earlier ones whole, in even slices
            last = (keys - slice_keys) // 32 * 32
            step = -(-last // (-(-last // slice_keys) * 32)) * 32
            starts = [*range(0, last, step), last]
            context.dense_slices = [(start, stop - start, torch.tensor([0, stop - start], dtype=torch.int32, device=q.device)) for start, stop in zip(starts, [*starts[1:], keys])]
        q16, output, total, capacity = q.to(torch.float16), None, None, max(count for _, count, _ in context.dense_slices)
        for start, count, bounds in context.dense_slices:
            dense_k, dense_v = context.dense_kv = attention_mixed.dense(self, context, torch.float16, context.dense_kv, start, count, capacity)
            part, normalization, _ = self.flash_attn_varlen_func(q16, dense_k[:count], dense_v[:count], max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q, max_seqlen_k=count,
                                                                 cu_seqlens_k=bounds, softmax_scale=self.scale, causal=start + count == keys, return_attn_probs=True)
            if output is None:
                output, total = part.float(), normalization
            else:  # softmax over both sets of keys: each part weighs its share of the summed exponentials
                share = (normalization - total).transpose(0, 1).unsqueeze(-1)
                output.mul_(torch.sigmoid(-share)).addcmul_(part, torch.sigmoid(share))
                total = torch.logaddexp(total, normalization)
        return output.to(q.dtype) if bool(torch.isfinite(output).all()) else None

    def _dense_caches(self, dtype: torch.dtype, context) -> tuple[torch.Tensor, torch.Tensor]:
        """The quantized cache as dense [pages, page, H, D] tensors for the SDPA / FlashAttention paths, exact overlays applied."""
        layout = self.kv_layout
        if layout is None or not layout.mixed:
            return _dequantize_kvcache(self.k_cache, self.k_scale, dtype), _dequantize_kvcache(self.v_cache, self.v_scale, dtype)
        dense_k = _dense_body(self.k_cache, self.k_scale, self.k_min, layout.k, dtype)
        dense_v = _dense_body(self.v_cache, self.v_scale, self.v_min, layout.v, dtype)
        if layout.overlay and self.k_overlay is not None and context.overlay_base is not None and context.block_tables is not None:
            _apply_overlays(dense_k, self.k_overlay, self.k_overlay_scale, context, layout, dtype)
            _apply_overlays(dense_v, self.v_overlay, self.v_overlay_scale, context, layout, dtype)
        return dense_k, dense_v

    def forward_list(self, qkv_list: list[torch.Tensor], *, cache_written=False):
        q, k, v = qkv_list
        qkv_list.clear()
        return self.forward(q, k, v, cache_written=cache_written)


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
_SM120_Q8 = getattr(llamacpp_gguf_cuda, "sm120", None)
if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
    if torch.version.hip is None and _SM120_Q8 is None:
        try:
            from . import attention_sm120 as _SM120_Q8
        except ImportError:  # Older Triton packages do not provide Gluon.
            pass


def _use_sm120_q8(q):
    return torch.version.hip is None and _SM120_Q8 is not None and q.shape[-1] == 256 and torch.cuda.get_device_capability(q.device) == (12, 0)


_SM120_TREE = []


def _sm120_tree_kernels():
    """The Gluon source of the SM120 kernel, which also masks trees (None without Gluon: the portable Triton kernel serves trees)."""
    if not _SM120_TREE:
        try:
            from . import attention_sm120
        except ImportError:
            attention_sm120 = None
        _SM120_TREE.append(attention_sm120 if torch.version.hip is None else None)
    return _SM120_TREE[0]


if triton is not None and tl is not None:
    # INT8 cache values are exact in FP16/BF16, so tensor cores multiply them
    # without the high/low split used above; the per-32-dimension scales are
    # applied to exact FP32-accumulated partial products instead:
    #   QK[i, j] = sum_b ks[j, b] * (Q[i, b] . Kq[j, b])
    #   O[i, b] = sum_j (P[i, j] * vs[j, b]) * Vq[j, b], with P * vs as a high/low pair.
    # This halves the tensor-core work and removes the K/V rounding residual.
    # P * vs is raised by an exact 2**6 so FP16 residuals stay normal; a scale
    # from FP16 values is at most 65504 / 127, so the product cannot overflow.
    @triton.jit(do_not_specialize=("bt_stride",), do_not_specialize_on_alignment=("bt_stride",))
    def q8_prefill_exact_kernel(
        q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, block_tables_ptr, cu_q_ptr, cu_k_ptr, out_ptr,
        q_stride_t: tl.constexpr, q_stride_h: tl.constexpr, cache_stride_block: tl.constexpr,
        cache_stride_token: tl.constexpr, cache_stride_head: tl.constexpr, scale_stride_block: tl.constexpr,
        scale_stride_token: tl.constexpr, scale_stride_head: tl.constexpr, bt_stride,
        out_stride_t: tl.constexpr, out_stride_h: tl.constexpr, softmax_scale: tl.constexpr,
        H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        NB: tl.constexpr = D // 32
        sequence = tl.program_id(0)
        query_head = tl.program_id(1)
        query_block = tl.program_id(2)
        q_start = tl.load(cu_q_ptr + sequence)
        q_end = tl.load(cu_q_ptr + sequence + 1)
        k_end = tl.load(cu_k_ptr + sequence + 1) - tl.load(cu_k_ptr + sequence)
        q_len = q_end - q_start
        prefix_len = k_end - q_len
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < q_len
        blocks = tl.arange(0, NB)
        lanes = tl.arange(0, 32)
        q = tl.load(q_ptr + (q_start + rows[None, :, None]) * q_stride_t + query_head * q_stride_h + blocks[:, None, None] * 32 + lanes[None, None, :],
                    mask=row_mask[None, :, None], other=0.0)
        maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        denominator = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((NB, BLOCK_M, 32), tl.float32)
        kv_head = query_head // (H_Q // H_KV)

        for key_start in tl.range(0, tl.minimum(k_end, prefix_len + (query_block + 1) * BLOCK_M), BLOCK_N):
            columns = key_start + tl.arange(0, BLOCK_N)
            column_mask = columns < k_end
            physical_block = tl.load(block_tables_ptr + sequence * bt_stride + columns // PAGE, mask=column_mask, other=0)
            physical_token = columns % PAGE
            cache_base = physical_block * cache_stride_block + physical_token * cache_stride_token + kv_head * cache_stride_head
            scale_base = physical_block * scale_stride_block + physical_token * scale_stride_token + kv_head * scale_stride_head
            k = tl.load(k_ptr + cache_base[None, None, :] + blocks[:, None, None] * 32 + lanes[None, :, None], mask=column_mask[None, None, :], other=0).to(q.dtype)
            k_scale = tl.load(ks_ptr + scale_base[None, :] + blocks[:, None], mask=column_mask[None, :], other=0.0).to(tl.float32)
            scores = tl.sum(tl.dot(q, k) * k_scale[:, None, :], axis=0) * softmax_scale
            causal = columns[None, :] <= prefix_len + rows[:, None]
            scores = tl.where(row_mask[:, None] & column_mask[None, :] & causal, scores, -float("inf"))
            tile_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            correction = tl.exp2((maximum - tile_maximum) * 1.4426950408889634)
            probabilities = tl.exp2((scores - tile_maximum[:, None]) * 1.4426950408889634)
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            v = tl.load(v_ptr + cache_base[None, :, None] + blocks[:, None, None] * 32 + lanes[None, None, :], mask=column_mask[None, :, None], other=0).to(q.dtype)
            v_scale = tl.load(vs_ptr + scale_base[None, :] + blocks[:, None], mask=column_mask[None, :], other=0.0).to(tl.float32)
            weighted = probabilities[None, :, :] * (v_scale[:, None, :] * 64.0)
            w_high = weighted.to(q.dtype)
            w_low = (weighted - w_high.to(tl.float32)).to(q.dtype)
            accumulator = accumulator * correction[None, :, None]
            accumulator = tl.dot(w_high, v, accumulator)
            accumulator = tl.dot(w_low, v, accumulator)
            maximum = tile_maximum
        output = accumulator / (denominator[None, :, None] * 64.0)
        tl.store(out_ptr + (q_start + rows[None, :, None]) * out_stride_t + query_head * out_stride_h + blocks[:, None, None] * 32 + lanes[None, None, :],
                 output, mask=row_mask[None, :, None])


    @triton.jit
    def _load_fp32(pointer, BF16_BITS: tl.constexpr):
        # GPUs without BF16 conversion instructions (Turing) read BF16 as its bits, an int16 view: exact
        if BF16_BITS:
            return bf16_bits.bf16_bits_to_fp32(tl.load(pointer))
        return tl.load(pointer).to(tl.float32)

    # Triton 3.2 aborts when pipelining the exact Q8 prefill kernel's rank-three loads.
    _Q8_EXACT_PREFILL_STAGES = 1 if triton.__version__.split(".")[:2] == ["3", "2"] else 2
