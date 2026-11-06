# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The output pass of a low-rank adapter, out += scale * y @ factor, on the tensor cores: FP32 accumulation and one rounding, as cuBLAS, for
every weight format since the base's output is a 16-bit tensor whatever its weights. K is small against the outputs, so that the pass is
bound by its reads and writes of out rather than by its products, which suits smaller tiles than general GEMM configurations. The
SwiGLU that reads a linear's output can share that pass. Compiled once per configuration: only the row and rank counts vary between calls."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_lora_kernels", 1, "LoRA")
BK = 32
# (BLOCK_M, BLOCK_N, warps, stages); the first one is the static configuration
_CANDIDATES = [(128, 64, 8, 3), (128, 64, 4, 3), (64, 64, 4, 3), (128, 128, 8, 3)]
_TUNE_ROWS = 1024


@triton.jit(do_not_specialize=["M", "K", "SCALE"], do_not_specialize_on_alignment=["M", "K"])
def _output_pass(Y, FACTOR, OUT, M, N, K, SY, SFN, SFK, SO, SCALE, RANK_FIRST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask, n_mask = rm < M, rn < N
    y_ptr = Y + rm[:, None] * SY + rk[None, :]
    factor_ptr = FACTOR + rk[:, None] * SFK + rn[None, :] if RANK_FIRST else FACTOR + rn[:, None] * SFN + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        y = tl.load(y_ptr, mask=m_mask[:, None], other=0.0)
        if RANK_FIRST:
            acc = tl.dot(y, tl.load(factor_ptr, mask=(k * BK + rk < K)[:, None] & n_mask[None, :], other=0.0), acc)
        else:
            acc = tl.dot(y, tl.trans(tl.load(factor_ptr, mask=n_mask[:, None], other=0.0)), acc)
        y_ptr += BK
        factor_ptr += BK * (SFK if RANK_FIRST else 1)
    out_ptr = OUT + rm[:, None] * SO + rn[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr, (acc * SCALE + tl.load(out_ptr, mask=mask, other=0.0).to(tl.float32)).to(OUT.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["M", "K", "SCALE"], do_not_specialize_on_alignment=["M", "K"])
def _output_swiglu(Y, FACTOR, OUT, M, N, K, SY, SFN, SFK, SO, SCALE, HAS_LORA: tl.constexpr, RANK_FIRST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Rows [gate | value] of N channels each, plus their LoRA, to silu(gate) * value over the gate half, with the roundings of the separate
    operations: out += scale * y @ factor, then F.silu(gate, inplace=True).mul_(value). One pass over the rows instead of three."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    m_mask, n_mask = rm < M, rn < N
    mask = m_mask[:, None] & n_mask[None, :]
    dtype = OUT.dtype.element_ty
    out_ptr = OUT + rm[:, None] * SO + rn[None, :]
    gate = tl.load(out_ptr, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(out_ptr + N, mask=mask, other=0.0).to(tl.float32)
    if HAS_LORA:
        rk = tl.arange(0, BK)
        acc_gate, acc_value = tl.zeros((BM, BN), dtype=tl.float32), tl.zeros((BM, BN), dtype=tl.float32)
        y_ptr = Y + rm[:, None] * SY + rk[None, :]
        gate_ptr = FACTOR + rk[:, None] * SFK + rn[None, :] if RANK_FIRST else FACTOR + rn[:, None] * SFN + rk[None, :]
        value_ptr = gate_ptr + (N if RANK_FIRST else N * SFN)
        for k in range(0, tl.cdiv(K, BK)):
            y = tl.load(y_ptr, mask=m_mask[:, None], other=0.0)
            if RANK_FIRST:
                k_mask = (k * BK + rk < K)[:, None] & n_mask[None, :]
                acc_gate = tl.dot(y, tl.load(gate_ptr, mask=k_mask, other=0.0), acc_gate)
                acc_value = tl.dot(y, tl.load(value_ptr, mask=k_mask, other=0.0), acc_value)
            else:
                acc_gate = tl.dot(y, tl.trans(tl.load(gate_ptr, mask=n_mask[:, None], other=0.0)), acc_gate)
                acc_value = tl.dot(y, tl.trans(tl.load(value_ptr, mask=n_mask[:, None], other=0.0)), acc_value)
            y_ptr += BK
            gate_ptr += BK * (SFK if RANK_FIRST else 1)
            value_ptr += BK * (SFK if RANK_FIRST else 1)
        gate = (acc_gate * SCALE + gate).to(dtype).to(tl.float32)
        value = (acc_value * SCALE + value).to(dtype).to(tl.float32)
    activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(dtype).to(tl.float32)
    tl.store(out_ptr, (activated * value).to(dtype), mask=mask)


def _launch_swiglu(out, y, factor, scale, rank_first, config):
    bm, bn, warps, stages = config
    m, n = out.shape[0], out.shape[1] // 2
    has_lora = y is not None
    k = 0 if not has_lora else factor.shape[0] if rank_first else factor.shape[1]
    y, factor = (out, out) if not has_lora else (y, factor)
    _output_swiglu[(triton.cdiv(m, bm), triton.cdiv(n, bn))](y, factor, out, m, n, k, y.stride(0), 0 if rank_first or not has_lora else factor.stride(0), factor.stride(0) if rank_first and has_lora else 0, out.stride(0), scale, has_lora, rank_first, bm, bn, BK, num_warps=warps, num_stages=stages, enable_fp_fusion=False)


def _pad(y):
    # rows that end inside a K tile: a small copy holding zeros up to the end of the tile, as the factor's rows there are masked or absent
    if y.shape[1] % BK == 0:
        return y
    padded = y.new_zeros(y.shape[0], -(-y.shape[1] // BK) * BK)
    padded[:, :y.shape[1]] = y
    return padded


def swiglu_(out, y=None, factor=None, scale=0.0, rank_first=False):
    """Rows [gate | value] of out, scale * y @ factor (see add_) added first when given, to silu(gate) * value in place over the gate half; returns that half."""
    if y is not None:
        y = _pad(y)

    def scratch():
        y_rows, out_rows = None if y is None else y[:_TUNE_ROWS], out[:_TUNE_ROWS].clone()
        return lambda config: _launch_swiglu(out_rows, y_rows, factor, 0.0, rank_first, config)

    config = _TUNER.config("output_swiglu", f"lora={y is not None} rank_first={rank_first}", out.device, _CANDIDATES, scratch, out.shape[0] >= _TUNE_ROWS)
    _launch_swiglu(out, y, factor, scale, rank_first, config)
    torch.autograd.graph.increment_version(out)
    return out[:, :out.shape[1] // 2]


def _launch(out, y, factor, scale, rank_first, config):
    bm, bn, warps, stages = config
    m, k, n = y.shape[0], factor.shape[0] if rank_first else factor.shape[1], factor.shape[1] if rank_first else factor.shape[0]
    _output_pass[(triton.cdiv(m, bm), triton.cdiv(n, bn))](y, factor, out, m, n, k, y.stride(0), 0 if rank_first else factor.stride(0), factor.stride(0) if rank_first else 0, out.stride(0), scale, rank_first, bm, bn, BK, num_warps=warps, num_stages=stages, enable_fp_fusion=False)


def supported(out, y, factor, rank_first):
    """out, y and factor of one 16-bit dtype, rows of out and y contiguous. A factor (outputs, rank) has to hold whole K tiles: its rows are
    read past the rank otherwise; a factor (rank, outputs) masks its last rows."""
    if y.dtype != out.dtype or factor.dtype != out.dtype or out.ndim != 2 or out.stride(1) != 1 or y.stride(1) != 1 or factor.stride(1) != 1:
        return False
    return rank_first or factor.shape[1] % BK == 0


def add_(out, y, factor, scale, rank_first):
    """out += scale * y @ factor (rank_first: factor (rank, outputs)), else y @ factor.T (factor (outputs, rank)). y of any width."""
    y = _pad(y)

    def scratch():
        rows = _TUNE_ROWS
        y_rows, out_rows = y[:rows], out[:rows].clone()
        return lambda config: _launch(out_rows, y_rows, factor, 0.0, rank_first, config)

    config = _TUNER.config("output_pass", f"rank_first={rank_first}", out.device, _CANDIDATES, scratch, y.shape[0] >= _TUNE_ROWS)
    _launch(out, y, factor, scale, rank_first, config)
    torch.autograd.graph.increment_version(out)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
