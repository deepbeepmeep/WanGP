# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Kernels of the attention kit's cached prefixes (shared.attention_kit): q8 / q4_0 keys / values (the LM engines' formats with
FP16 scales per token, head and 32 channels) dequantized through strides into a 16-bit buffer (a head group of a prefix into the joined keys
or values, a prefix of one batch item broadcast to all), and two attentions of the same queries over separate key sets merged in place by
their base-2 log-sum-exp. The dequantization rounds the FP32 product once, as the PyTorch path; the merge computes in FP32 (precise
division) and rounds once. BITS: BF16 tensors passed as int16 views and converted by the bit helpers of bf16_bits.py, for GPUs whose
Triton has no BF16 conversion instructions (compute capability 7.5); same roundings (to nearest even, as PyTorch's casts)."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .bf16_bits import bf16_bits_to_fp32, fp32_to_bf16_bits
from .dispatch import device_supported
from .triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_kv_prefix", 2, "Attention")
_TUNE_TOKENS = 4096
_RUNTIME = ["TOKENS", "DB", "DT", "DH", "SB", "ST", "SH", "OB", "OT", "OH"]


@triton.jit(do_not_specialize=_RUNTIME)
def _dequantize_kernel(DATA, SCALES, OUT, TOKENS, DB, DT, DH, SB, ST, SH, OB, OT, OH, D: tl.constexpr, BLOCK_T: tl.constexpr,
                      BITS: tl.constexpr, Q4: tl.constexpr):
    tokens = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    head, batch = tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    c = tl.arange(0, D)[None, :]
    valid = tokens < TOKENS
    if Q4:
        packed = tl.load(DATA + batch * DB + tokens * DT + head * DH + c // 32 * 16 + c % 16, valid, 0).to(tl.int32)
        data = (((packed >> tl.where(c % 32 < 16, 0, 4)) & 15) - 8).to(tl.float32)
    else:
        data = tl.load(DATA + batch * DB + tokens * DT + head * DH + c, valid, 0).to(tl.float32)
    scale = tl.load(SCALES + batch * SB + tokens * ST + head * SH + c // 32, valid, 0.0).to(tl.float32)
    value = data * scale
    tl.store(OUT + batch * OB + tokens * OT + head * OH + c, fp32_to_bf16_bits(value) if BITS else value.to(OUT.dtype.element_ty), valid)


@triton.jit(do_not_specialize=["TOKENS", "OB", "OT", "OH", "PB", "PT", "PH", "LB", "LH"])
def _merge_lse_kernel(O, P, LO, LP, TOKENS, OB, OT, OH, PB, PT, PH, LB, LH, D: tl.constexpr, BLOCK_T: tl.constexpr, BITS: tl.constexpr):
    tokens = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    head, batch = tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    valid = tokens < TOKENS
    lo = tl.load(LO + batch * LB + head * LH + tokens, valid, 0.0)
    lp = tl.load(LP + batch * LB + head * LH + tokens, valid, 0.0)
    top = tl.maximum(lo, lp)
    wo, wp = libdevice.exp2(lo - top), libdevice.exp2(lp - top)
    total = wo + wp
    wo, wp = tl.div_rn(wo, total)[:, None], tl.div_rn(wp, total)[:, None]
    c, rows = tl.arange(0, D)[None, :], tokens[:, None]
    o_ptrs = O + batch * OB + rows * OT + head * OH + c
    o = tl.load(o_ptrs, valid[:, None], 0)
    p = tl.load(P + batch * PB + rows * PT + head * PH + c, valid[:, None], 0)
    if BITS:
        o, p = bf16_bits_to_fp32(o), bf16_bits_to_fp32(p)
    merged = o.to(tl.float32) * wo + p.to(tl.float32) * wp
    tl.store(o_ptrs, fp32_to_bf16_bits(merged) if BITS else merged.to(O.dtype.element_ty), valid[:, None])


def _bits(x):
    """A BF16 tensor on a GPU below the BF16 floor (Turing): passed to the kernels as its int16 view."""
    return x.dtype == torch.bfloat16 and not device_supported(x.device, torch.bfloat16)


def dequantize_q8_into(out, data, scales):
    """out (batch, tokens, heads, head_dim) 16-bit = data (batch or 1, tokens, heads, head_dim) int8 times scales (..., head_dim // 32)
    FP16, through their strides (last dimensions contiguous)."""
    return _dequantize_into(out, data, scales, q4=False)


def dequantize_q4_into(out, data, scales):
    """As dequantize_q8_into, with q4_0 packed uint8 data (..., head_dim // 2)."""
    return _dequantize_into(out, data, scales, q4=True)


def _dequantize_into(out, data, scales, q4):
    batch, tokens, heads, dim = out.shape
    broadcast = data.shape[0] == 1 and batch > 1
    bits = _bits(out)

    def launch(target, source, source_scales, config):
        block, warps = config
        _dequantize_kernel[(triton.cdiv(target.shape[1], block), heads, batch)](
            source, source_scales, target.view(torch.int16) if bits else target, target.shape[1], 0 if broadcast else source.stride(0), source.stride(1),
            source.stride(2), 0 if broadcast else source_scales.stride(0), source_scales.stride(1), source_scales.stride(2), *target.stride()[:3], dim, block,
            bits, q4, num_warps=warps)

    def scratch():
        target = torch.empty((batch, _TUNE_TOKENS, heads, dim), dtype=out.dtype, device=out.device)
        return lambda config: launch(target, data[:, :_TUNE_TOKENS], scales[:, :_TUNE_TOKENS], config)

    config = _TUNER.config(f"dequantize_q{4 if q4 else 8}{'_bits' if bits else ''}", dim, out.device, ((32, 4), (16, 4), (64, 4), (64, 8)), scratch,
                           tokens >= _TUNE_TOKENS)
    launch(out, data, scales, config)
    torch.autograd.graph.increment_version(out)
    return out


def merge_lse_(out, lse, other, other_lse):
    """In place: out (batch, tokens, heads, head_dim, through its strides) = the attention over both key sets of the queries whose
    attention over one set is out with lse (batch, heads, tokens) base-2 log-sum-exp, and over the other is other with other_lse."""
    batch, tokens, heads, dim = out.shape
    bits = _bits(out)

    def launch(target, source, target_lse, source_lse, config):
        block, warps = config
        target, source = (target.view(torch.int16), source.view(torch.int16)) if bits else (target, source)
        _merge_lse_kernel[(triton.cdiv(target.shape[1], block), heads, batch)](
            target, source, target_lse, source_lse, target.shape[1], *target.stride()[:3], *source.stride()[:3], target_lse.stride(0),
            target_lse.stride(1), dim, block, bits, num_warps=warps, enable_fp_fusion=False)  # the PyTorch path's roundings

    def scratch():
        target, source = out[:, :_TUNE_TOKENS].clone(), other[:, :_TUNE_TOKENS].contiguous()
        target_lse, source_lse = lse[:, :, :_TUNE_TOKENS].contiguous(), other_lse[:, :, :_TUNE_TOKENS].contiguous()
        return lambda config: launch(target, source, target_lse, source_lse, config)

    config = _TUNER.config(f"merge_lse{'_bits' if bits else ''}", dim, out.device, ((32, 4), (16, 4), (64, 4), (64, 8)), scratch, tokens >= _TUNE_TOKENS)
    launch(out, other, lse, other_lse, config)
    torch.autograd.graph.increment_version(out)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
