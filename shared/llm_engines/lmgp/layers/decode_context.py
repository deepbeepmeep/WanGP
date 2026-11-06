# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The values of one sequence's decoding pass reach the buffers its CUDA graphs read in one launch of a kernel, from mapped RAM that the host writes without a driver call, rather than in one copy each."""
import triton
import triton.language as tl


@triton.jit
def _decode_context(source, input_ids, positions, slot_mapping, context_lens, overlay_mapping, overlay_base, ring_start):
    tl.store(input_ids, tl.load(source))
    tl.store(positions, tl.load(source + 1))
    tl.store(slot_mapping, tl.load(source + 2).to(tl.int32))
    tl.store(context_lens, tl.load(source + 3).to(tl.int32))
    tl.store(overlay_mapping, tl.load(source + 4).to(tl.int32))
    tl.store(overlay_base, tl.load(source + 5).to(tl.int32))
    tl.store(ring_start, tl.load(source + 6).to(tl.int32))


def load_decode_context(source, input_ids, positions, slot_mapping, context_lens, overlay_mapping, overlay_base, ring_start):
    """source: int64 [7] of mapped RAM, a CUDA view. Each value goes to the first row of its buffer."""
    assert source.is_cuda and input_ids.is_cuda and positions.is_cuda and slot_mapping.is_cuda and context_lens.is_cuda and overlay_mapping.is_cuda and overlay_base.is_cuda and ring_start.is_cuda
    _decode_context[(1,)](source, input_ids, positions, slot_mapping, context_lens, overlay_mapping, overlay_base, ring_start, num_warps=1)


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
