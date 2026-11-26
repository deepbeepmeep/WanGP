"""SageAttention (1): INT8 q/k per block of tokens and fp16 v, the Triton kernels of sageattn_varlen (SageAttention 1.0.6, or the copy
SageAttention 2 keeps in its triton package), launched without its device sync (the sizes of its scales read on the GPU) so that CUDA
graphs can record them, and in stages for the attention kit: v, k and q quantized as soon as each is computed, the output written into
q. Same values as sageattn_varlen for sequences of the same length (all batch items), head_dim 64 or 128, no mask."""
from functools import lru_cache

import torch

from shared.kernels.token_mean import token_mean
from shared.kernels import sage_block_query

try:  # SageAttention 2
    from sageattention.triton.quant_per_block_varlen import quant_per_block_int8_kernel
    from sageattention.triton.attn_qk_int8_block_varlen import _attn_fwd
    def _sm_scale(head_dim):  # sageattn_varlen's softmax scale, computed as each version does
        return 1.0 / head_dim ** 0.5
except ImportError:  # SageAttention 1.0.6
    from sageattention.quant_per_block_varlen import quant_per_block_int8_kernel
    from sageattention.attn_qk_int8_block_varlen import _attn_fwd
    def _sm_scale(head_dim):
        return head_dim ** -0.5
try:  # compute capability 7.5: the same kernels reading BF16 q, k and v and writing BF16 output as bits (no FP16 copies)
    from .sage1_turing import _attn_fwd as _attn_fwd_bits, _attn_range, quant_per_block_int8_bits_kernel as _quant_bits
except ImportError:
    _attn_fwd_bits = _attn_range = _quant_bits = None
# comparison switch: the query blocks of a part-filled last wave run over ranges of the keys (sage1_turing._attn_range), merged by
# sagegp.split_merge, the full waves by the kernel as before: faster where a launch has few CTAs per wave (few heads)
LAST_WAVE_SPLIT = False

SETTINGS = {"scale_v": False}  # the staged functions take the settings of the staged_* functions of shared.sage2_core: v scaling only here
SCALED_V_SETTINGS = {"scale_v": True}
HEAD_DIMS = (64, 128, 256)  # 256 (not in sageattn_varlen): one pipeline stage, GPUs of compute capability 8.0 or more (head_dims)
_STAGES = {64: 3, 128: 4, 256: 1}
# The Triton attentions of SageAttention (1, and 2 where its CUDA kernels are not used) add P.V up in fp16 by blocks of 64 keys (p <= 1):
# the v of some models (model def key sage_v_scaling) overflows them. Their v is divided per head and channel by a power of two that
# brings it to 512 at most (block sums below 32768) and the output is multiplied back: no rounding added, same values where v stays
# within 512.
_V_LIMIT = 512.0
# Turing kernel: a BF16 v read as bits is converted again by every block of 128 queries; measured on an RTX 2080 Ti against the one-off
# FP16 copy: faster up to about 2,000 tokens (+9% on 1,797 and 1,001 tokens), slower beyond (-3% at 4,096, -2.6% at 15,065).
_V_BITS_MAX_TOKENS = 2048


def v_scales(v, tokens_dims):
    """Powers of two, at least 1, per head and channel of v (its batch and token dimensions: tokens_dims, kept with size 1)."""
    amax = torch.linalg.vector_norm(v, float("inf"), dim=tokens_dims, keepdim=True).float()
    return torch.exp2(torch.ceil(torch.log2(amax.mul_(1 / _V_LIMIT)))).clamp_(min=1)


def fp16_v(v, scale_v, tokens_dims=(0, 1)):
    """fp16 v and, with scale_v, its v_scales (v divided by them), None otherwise."""
    if not scale_v:
        return v.to(torch.float16), None
    scale = v_scales(v, tokens_dims)
    return torch.div(v, scale, out=torch.empty(v.shape, dtype=torch.float16, device=v.device)), scale


def scale_output(out, scale, heads_dim):
    """out multiplied in place by the v scales of its attention (out has more heads with grouped query attention)."""
    groups = out.shape[heads_dim] // scale.shape[heads_dim]
    return out.mul_(scale if groups == 1 else scale.repeat_interleave(groups, heads_dim))


@lru_cache(None)
def _fp16_required(device):
    return torch.cuda.get_device_capability(device)[0] < 8


@lru_cache(None)
def masks_supported(device):
    """Attention masks run on the Turing kernel (compute capability 7.5); the packaged kernel takes none."""
    return _attn_fwd_bits is not None and _fp16_required(device)


@lru_cache(None)
def head_dims(device):
    """Head dims of the kernels on this device: at 256, q blocks of 128 rows need more shared memory than before compute capability 8.0."""
    return HEAD_DIMS[:2] if _fp16_required(device) else HEAD_DIMS


def padded_head_dim(head_dim, device):
    """The head_dim the kernels run a head_dim at: the next of head_dims(device), None beyond. attention pads q, k and v with zeros up to
    it, as SageAttention 2 does: the same q.k products (INT8, summed exactly), INT8 scales and key means, with the softmax scale of the
    real head_dim; the extra output channels (zero) are dropped."""
    return next((dim for dim in head_dims(device) if dim >= head_dim), None)


def _offsets(batch, length, device):
    """[0, length, 2 length, ...] of the batch items, made on the GPU (a host copy would not be allowed in a CUDA graph)."""
    return torch.arange(0, (batch + 1) * length, length, dtype=torch.int32, device=device)


def _bits_input(x):
    """A BF16 tensor on compute capability 7.5: the Turing kernels read it as bits instead of an FP16 copy."""
    return _quant_bits is not None and x.dtype == torch.bfloat16 and _fp16_required(x.device)


def _quantize(x, block, sm_scale, mean=None):
    """x (batch, tokens, heads, head_dim), batch items one after another in memory, as INT8 rows (batch * tokens, heads, head_dim) and
    scales per block of tokens and head, with the offsets of the batch items in both. mean (heads, head_dim), FP32: subtracted first
    inside the Turing kernel (BF16 x read as bits)."""
    batch, tokens, heads, head_dim = x.shape
    blocks = (tokens + block - 1) // block
    x_int8 = torch.empty((batch * tokens, heads, head_dim), dtype=torch.int8, device=x.device)
    scale = torch.empty((batch * blocks, heads), dtype=torch.float32, device=x.device)
    offsets, scale_offsets = _offsets(batch, tokens, x.device), _offsets(batch, blocks, x.device)
    if _bits_input(x):
        _quant_bits[(blocks, heads, batch)](x.view(torch.int16), x_int8, scale, x_int8 if mean is None else mean, offsets, scale_offsets,
                                            x.stride(2), x.stride(1), x_int8.stride(1), x_int8.stride(0),
                                            sm_scale=sm_scale, H=heads, C=head_dim, BLK=block, HAS_MEAN=mean is not None)
        return x_int8, scale, offsets, scale_offsets
    if mean is not None:
        raise ValueError("the key mean is folded into the quantization on the Turing kernel only")
    quant_per_block_int8_kernel[(blocks, heads, batch)](x, x_int8, scale, offsets, scale_offsets, x.stride(2), x.stride(1), x_int8.stride(1), x_int8.stride(0),
                                                        sm_scale=sm_scale, H=heads, C=head_dim, BLK=block)
    return x_int8, scale, offsets, scale_offsets


@torch.compiler.disable
def staged_quantize_v(v_list, settings):
    """fp16 v with its scales (fp16_v, None without settings["scale_v"]); v is released. On compute capability 7.5, an unscaled BF16 v
    of a short sequence stays as it is: the Turing kernel reads its bits (staged_attention)."""
    v = v_list.pop()
    if "padded_dim" in settings:
        v = torch.nn.functional.pad(v, (0, settings["padded_dim"] - v.size(-1)))
    if (_attn_fwd_bits is not None and v.dtype == torch.bfloat16 and not settings["scale_v"] and v.shape[1] <= _V_BITS_MAX_TOKENS
            and _fp16_required(v.device)):
        return v, None, None
    return *fp16_v(v, settings["scale_v"]), None


def _quantize_k(k, in_place, mean=None):
    """INT8 k minus its mean over the tokens of all batch items (or mean, (heads, head_dim) FP32: Turing kernel only), with its scales."""
    rows = k.view(-1, *k.shape[2:])
    if _bits_input(k):  # the FP32 mean is subtracted inside the quantization kernel: no FP16 copy, k untouched
        return _quantize(k, 64, 1.0, token_mean(rows, 0, fp32=True)[0] if mean is None else mean)
    if mean is not None:
        raise ValueError("A given key mean is subtracted by the Turing kernel only")
    if k.dtype == torch.bfloat16 and _fp16_required(k.device):
        k = k.to(torch.float16)  # match pay_attention: convert before smoothing on pre-Ampere GPUs
        rows = k.view(-1, *k.shape[2:])
    mean = token_mean(rows, 0)
    return _quantize((rows.sub_(mean) if in_place else rows - mean).view(k.shape), 64, 1.0)


@torch.compiler.disable
def staged_quantize_k(k_list, settings):
    """INT8 k minus its mean over the tokens of all batch items, with its scales; k is released (the mean is subtracted in place).
    settings["k_mean"] (heads, head_dim) FP32, when present, is the mean subtracted instead (a cached prefix's: shared.attention_kit)."""
    k = k_list.pop()
    if "padded_dim" in settings:
        k = torch.nn.functional.pad(k, (0, settings["padded_dim"] - k.size(-1)))
    return _quantize_k(k, True, settings.get("k_mean"))


@torch.compiler.disable
def staged_quantize_q(q, settings, sm_scale=None):
    """INT8 q with its scales (the softmax scale, in base 2, goes into them: sm_scale, or that of q's head_dim); q is not released
    (staged_attention can write the output into it)."""
    factor = (_sm_scale(q.shape[-1]) if sm_scale is None else sm_scale) * 1.44269504
    if "padded_dim" in settings:
        q = torch.nn.functional.pad(q, (0, settings["padded_dim"] - q.size(-1)))
    batch, tokens, heads, dim = q.shape
    if (settings.get("recycle_query") and _fp16_required(q.device) and (q.dtype == torch.float16 or _bits_input(q))
            and (batch == 1 or q.stride(0) == tokens * q.stride(1)) and sage_block_query.recyclable_query(q)):
        packed, scales = sage_block_query.pack_query(q, 128, factor, varlen=True)
        return (packed.view(-1, heads, dim), scales.view(-1, heads), _offsets(batch, tokens, q.device),
                _offsets(batch, scales.size(1), q.device))
    if q.dtype == torch.bfloat16 and _fp16_required(q.device) and not _bits_input(q):
        q = q.to(torch.float16)
    return _quantize(q, 128, factor)


@torch.compiler.disable
def staged_attention(out, quantized_list, settings, attn_mask=None, return_lse=False):
    """Attention of the staged [q, k, v], written into out and returned: out has q's shape (batch, tokens, heads, head_dim) and its
    batch items one after another in memory: q itself, or a slice of heads of a larger output. Fewer heads in k/v: grouped query
    attention. return_lse (Turing kernel only): (out, lse), lse (batch, heads, tokens) FP32 the base-2 log-sum-exp of each query's
    scores without the keys' smoothing mean."""
    if quantized_list[0][0].size(-1) != out.size(-1):
        padded = out.new_empty((*out.shape[:-1], quantized_list[0][0].size(-1)))
        result = staged_attention(padded, quantized_list, settings, attn_mask, return_lse)
        out.copy_(padded[..., :out.size(-1)])
        return (out, result[1]) if return_lse else out
    (q_int8, q_scale, q_offsets, q_scale_offsets), (k_int8, k_scale, k_offsets, k_scale_offsets), (v, v_scale, _) = quantized_list
    quantized_list.clear()
    batch, tokens, heads, head_dim = out.shape
    rows, v = out.view(-1, heads, head_dim), v.view(-1, *v.shape[2:])
    # compute capability 7.5 with a BF16 output: the Turing kernel writes it as bits (no FP16 scratch and copy) and reads a BF16 v
    # (kept by staged_quantize_v for short sequences) as bits
    bits = out.dtype == torch.bfloat16 and _attn_fwd_bits is not None and _fp16_required(out.device)
    v_bits = v.dtype == torch.bfloat16
    if v_bits and not bits:
        raise ValueError(f"SageAttention on this GPU takes a BF16 v only with a BF16 output, not {out.dtype}")
    fp16_output = out.dtype == torch.bfloat16 and _fp16_required(out.device) and not bits
    if fp16_output:  # Triton cannot write BF16 on pre-Ampere either; scratch covers only this head group
        rows = torch.empty(rows.shape, dtype=torch.float16, device=out.device)
    grid = ((tokens + 127) // 128, heads, batch)
    args = (q_int8, k_int8, v.view(torch.int16) if v_bits else v, q_offsets, k_offsets, q_scale, k_scale, q_scale_offsets, k_scale_offsets,
            rows.view(torch.int16) if bits else rows)
    strides = (q_int8.stride(1), q_int8.stride(0), k_int8.stride(1), k_int8.stride(0), v.stride(1), v.stride(0), rows.stride(1), rows.stride(0))
    launch = dict(BLOCK_M=128, BLOCK_N=64, HEAD_DIM=head_dim, STAGE=1, num_warps=4 if head_dim == 64 else 8, num_stages=_STAGES[head_dim])
    mask, mask_mode = None, 0
    masked = attn_mask is not None and masks_supported(out.device)
    if masked:  # SageAttention's (batch, heads, q, k) layout, broadcastable; boolean as a uint8 view, additive as FP32
        mask = attn_mask.view(torch.uint8) if attn_mask.dtype == torch.bool else attn_mask.to(torch.float32)
        mask = mask.expand(batch, heads, tokens, k_int8.shape[0] // batch)
        mask_mode = 1 if attn_mask.dtype == torch.bool else 2
    mask_strides = mask.stride() if mask is not None else (0, 0, 0, 0)
    main, splits = (_range_plan(out.device, head_dim, launch["num_warps"], launch["num_stages"], grid[0], heads * batch, k_int8.shape[0] // batch)
                    if LAST_WAVE_SPLIT and _attn_range is not None and attn_mask is None and not return_lse else (grid[0], 1))
    grid = (main, heads, batch)
    if bits or masked:
        # without a mask the kernel still takes a pointer (never read): q_int8 with zero strides, as Lse without return_lse
        lse = torch.empty((batch, heads, tokens), dtype=torch.float32, device=out.device) if return_lse else q_int8
        if main:
            _attn_fwd_bits[grid](*args, q_int8 if mask is None else mask, lse, *strides, *mask_strides,
                                 heads, heads // k_int8.shape[1], V_BITS=v_bits, MASK_MODE=mask_mode, RETURN_LSE=return_lse, **launch)
    else:
        if attn_mask is not None or return_lse:
            raise ValueError("SageAttention takes an attention mask or returns the log-sum-exp only on the Turing kernel")
        if main:
            _attn_fwd[grid](*args, *strides, heads, heads // k_int8.shape[1], **launch)
    if splits > 1:  # the last wave's query blocks: ranges of the keys, merged into the output's rows after the full blocks'
        from sagegp import split_merge
        keys, tail_rows = k_int8.shape[0] // batch, tokens - main * 128
        parts = torch.empty((batch * splits, tail_rows, heads, head_dim), dtype=torch.float32, device=out.device)
        range_lse = torch.empty((batch * splits, heads, tail_rows), dtype=torch.float32, device=out.device)
        _attn_range[((tokens + 127) // 128 - main, heads, batch * splits)](*args[:-1], parts, q_int8 if mask is None else mask, range_lse, *strides[:6],
                                                                            *mask_strides, heads, heads // k_int8.shape[1], main, tail_rows,
                                                                            -(-((keys + 63) // 64) // splits) * 64,
                                                                            SPLITS=splits, V_BITS=v_bits, MASK_MODE=mask_mode, **launch)
        target = rows.view(torch.int16) if bits else rows
        split_merge.merge(parts, range_lse, target.view(batch, tokens, heads, head_dim), main * 128, bf16_bits=bits)
    if fp16_output:
        out.copy_(rows.view(out.shape))
    elif settings.get("recycle_query") and q_int8.data_ptr() == out.data_ptr():
        torch.autograd.graph.increment_version(out)
    out = out if v_scale is None else scale_output(out, v_scale, 2)
    return (out, lse) if return_lse else out


@lru_cache(None)
def _range_slots(device, head_dim, num_warps, num_stages):
    """CTAs of the range kernel the GPU runs at once (SMs x its occupancy, from a compilation on placeholder inputs)."""
    keys = 64 * 32
    q = torch.zeros((128, 1, head_dim), dtype=torch.int8, device=device)
    k = torch.zeros((keys, 1, head_dim), dtype=torch.int8, device=device)
    v = torch.zeros((keys, 1, head_dim), dtype=torch.float16, device=device)
    seq_q, seq_k = torch.tensor([0, 128], dtype=torch.int32, device=device), torch.tensor([0, keys], dtype=torch.int32, device=device)
    scale_q, scale_k = torch.tensor([0, 1], dtype=torch.int32, device=device), torch.tensor([0, keys // 64], dtype=torch.int32, device=device)
    scales = torch.ones(keys // 64, dtype=torch.float32, device=device)
    parts, lse = torch.empty((2, 128, 1, head_dim), dtype=torch.float32, device=device), torch.empty((2, 1, 128), dtype=torch.float32, device=device)
    compiled = _attn_range[(1, 1, 2)](q, k, v, seq_q, seq_k, scales, scales, scale_q, scale_k, parts, q, lse, q.stride(1), q.stride(0), k.stride(1),
                                      k.stride(0), v.stride(1), v.stride(0), 0, 0, 0, 0, 1, 1, 0, 128, keys // 2, SPLITS=2, HEAD_DIM=head_dim,
                                      BLOCK_M=128, BLOCK_N=64, STAGE=1, V_BITS=False, MASK_MODE=0, num_warps=num_warps, num_stages=num_stages)
    from sagegp import split_merge
    return split_merge.triton_slots(device, compiled, num_warps)


def _range_plan(device, head_dim, num_warps, num_stages, blocks, programs_per_block, keys):
    from sagegp import split_merge
    return split_merge.tail_plan(_range_slots(device, head_dim, num_warps, num_stages), blocks, programs_per_block, (keys + 63) // 64)


# A cached prefix (shared.attention_kit.cache_prefix) on the Turing kernel: k as staged_quantize_k makes it with the prefix's own mean
# (prefix_mean), which the keys attended with it subtract too; v stored by the kit and made 16-bit again for each attention.
def prefix_supported(settings, device, dtype):
    return _attn_fwd_bits is not None and dtype == torch.bfloat16 and _fp16_required(device)


def prefix_settings(settings, device, scale_v):
    return settings


def prefix_fp8_v(settings):
    return False


def prefix_mean(k):
    """(heads, head_dim) FP32 mean over the tokens of all batch items, rounded to k's dtype as SageAttention 2's prefixes."""
    return token_mean(k.reshape(-1, *k.shape[2:]), 0)[0].float()


def prefix_mean_heads(mean, heads):
    return mean[heads.start:heads.stop]


def prefix_quantize_k(k_list, settings, mean):
    """A prefix's k (batch, tokens, heads, head_dim, released) quantized with mean: k_int8 (batch, tokens, heads, head_dim) and k_scale
    (batch, blocks, heads)."""
    batch, tokens = k_list[0].shape[:2]
    k_int8, k_scale, _, _ = staged_quantize_k(k_list, {**settings, "k_mean": mean})
    return k_int8.view(batch, tokens, *k_int8.shape[1:]), k_scale.view(batch, -1, k_scale.shape[-1])


def prefix_k(k_int8, k_scale, heads, batch):
    """The quantized k of staged_attention for the heads of a cached prefix: k_int8 (batch, tokens, heads, head_dim) and k_scale
    (batch, blocks, heads) as _quantize makes them (batch items one after another), for batch items (one prefix item serves them all)."""
    k_int8, k_scale = k_int8[:, :, heads.start:heads.stop], k_scale[:, :, heads.start:heads.stop]
    if k_int8.shape[0] != batch:
        k_int8, k_scale = k_int8.expand(batch, -1, -1, -1), k_scale.expand(batch, -1, -1)
    tokens, blocks = k_int8.shape[1], k_scale.shape[1]
    return (k_int8.reshape(-1, *k_int8.shape[2:]).contiguous(), k_scale.reshape(-1, k_scale.shape[-1]).contiguous(),
            _offsets(batch, tokens, k_int8.device), _offsets(batch, blocks, k_int8.device))


@torch.compiler.disable
def attention(qkv_list, recycle_q=False, scale_v=False, attention_mask=None, sm_scale=None):
    """sageattn_varlen of q, k and v (batch, tokens, heads, head_dim, batch items one after another in memory) handed off in qkv_list,
    every batch item with all its tokens. k and v are not modified (others may hold them); the output is written into q with
    recycle_q. scale_v: v scaled by powers of two (fp16_v). attention_mask: SageAttention's (batch, heads, q, k) layout
    (shared.attention.sage_attention_mask), Turing kernel only (masks_supported). sm_scale: the softmax scale, that of the head_dim by
    default. A head_dim the kernels do not run is padded with zeros (padded_head_dim)."""
    q, k, v = qkv_list
    qkv_list.clear()
    head_dim = q.shape[-1]
    padded, sm_scale = padded_head_dim(head_dim, q.device), _sm_scale(head_dim) if sm_scale is None else sm_scale
    if padded != head_dim:
        q, k, v = (torch.nn.functional.pad(tensor, (0, padded - head_dim)) for tensor in (q, k, v))
    recycle_q = recycle_q and not (torch._C._overlaps(q, k) or torch._C._overlaps(q, v))
    settings = SCALED_V_SETTINGS if scale_v else SETTINGS
    if recycle_q:
        settings = {**settings, "recycle_query": True}
    quantized = [None, _quantize_k(k, False), staged_quantize_v([v], settings)]
    k = v = None
    quantized[0] = staged_quantize_q(q, settings, sm_scale)
    out = staged_attention(q if recycle_q else torch.empty_like(q, memory_format=torch.contiguous_format), quantized, settings, attention_mask)
    return out if padded == head_dim else out[..., :head_dim]
