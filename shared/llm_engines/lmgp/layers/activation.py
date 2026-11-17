# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from torch import nn
import torch.nn.functional as F
from shared.kernels import bf16_bits
from shared.kernels.dispatch import triton_dtype_supported

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None


if triton is not None:
    @triton.jit(do_not_specialize=("ROWS",), do_not_specialize_on_alignment=("ROWS",))
    def _silu_mul_kernel(x, out, ROWS, COLUMNS, R: tl.constexpr, BLOCK: tl.constexpr):
        # tiles of R rows x BLOCK columns, the FFN width at runtime: one compile for every model's width
        rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = (rows < ROWS)[:, None] & (columns < COLUMNS)[None, :]
        source = rows[:, None] * (2 * COLUMNS) + columns[None, :]
        gate = tl.load(x + source, valid, other=0).to(tl.float32)
        value = tl.load(x + source + COLUMNS, valid, other=0).to(tl.float32)
        # Preserve the rounding between the existing SiLU and multiply ops.
        activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(out.dtype.element_ty).to(tl.float32)
        tl.store(out + rows[:, None] * COLUMNS + columns[None, :], activated * value, valid)


class SiluAndMul(nn.Module):

    def __init__(self, use_triton: bool = True):
        super().__init__()
        self.use_triton = use_triton  # the LM features' Triton switch; silu_mul picks the kernel the GPU and dtype run

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward_list(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        x = x_list[0]
        x_list.clear()
        return silu_mul(x, self.use_triton)


def silu_mul(x, use_triton=True):
    if use_triton and x.is_cuda and triton is not None and triton_dtype_supported(x.device, x.dtype):
        x = x.contiguous()
        output = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
        columns = output.shape[-1]
        rows = output.numel() // columns
        block = min(256, triton.next_power_of_2(columns))  # 256 elements per program as before: decode rows keep their parallelism
        _silu_mul_kernel[(triton.cdiv(rows, 256 // block), triton.cdiv(columns, block))](x, output, rows, columns, 256 // block, block)
        return output
    if use_triton and triton is not None and bf16_bits.applies(x):  # Turing: the same kernel reading BF16 as bits
        return bf16_bits.silu_mul(x)
    gate, value = x.chunk(2, -1)
    F.silu(gate, inplace=True).mul_(value)
    return gate.contiguous()


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
