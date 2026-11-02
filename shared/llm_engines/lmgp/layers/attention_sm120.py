"""SM120 Q8 attention: asynchronous staging with shared high/low math.

Only the vllm attention dispatcher selects this optional module. Other CUDA
architectures and older Triton versions retain the shared kernels. Use cp.async:
TMA launches on the tested Windows driver add a 3.3 GiB stack reservation.
Query counts, context lengths, page counts and table widths are runtime values.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

# Triton 3.9 renamed the builtin; resolve it once, outside the launch path.
_async_load = getattr(async_copy, "async_load", None)
if _async_load is None:
    _async_load = async_copy.async_copy_global_to_shared
_barrier = getattr(gl, "barrier", None)
if _barrier is None:
    _barrier = gl.thread_barrier


@gluon.jit
def _stage_cache(k_ptr, v_ptr, k_smem, v_smem, slot, head, H_KV: gl.constexpr, D: gl.constexpr, N: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    rows = gl.arange(0, N, layout=gl.SliceLayout(1, layout))
    dims = gl.arange(0, D, layout=gl.SliceLayout(0, layout))
    offsets = ((slot + rows[:, None]) * H_KV + head) * D + dims[None, :]
    _async_load(k_smem, k_ptr + offsets)
    _async_load(v_smem, v_ptr + offsets)
    async_copy.commit_group()


@gluon.jit(do_not_specialize=("bt_stride",), do_not_specialize_on_alignment=("bt_stride",))
def q8_prefill_async_kernel(
    q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tables_ptr, cu_q_ptr, cu_k_ptr, out_ptr,
    q_stride_t: gl.constexpr, q_stride_h: gl.constexpr,
    scale_stride_block: gl.constexpr, scale_stride_token: gl.constexpr, scale_stride_head: gl.constexpr,
    bt_stride, out_stride_t: gl.constexpr, out_stride_h: gl.constexpr, SCALE: gl.constexpr,
    H_Q: gl.constexpr, H_KV: gl.constexpr, D: gl.constexpr, PAGE: gl.constexpr,
):
    M: gl.constexpr = 16
    N: gl.constexpr = 32
    STAGES: gl.constexpr = 2
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    # Match the shared kernel's operand packing and floating-point sum order.
    QA: gl.constexpr = gl.DotOperandLayout(0, MMA, 4)
    KB: gl.constexpr = gl.DotOperandLayout(1, MMA, 4)
    sequence, head, block = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    q_start = gl.load(cu_q_ptr + sequence)
    q_len = gl.load(cu_q_ptr + sequence + 1) - q_start
    k_len = gl.load(cu_k_ptr + sequence + 1) - gl.load(cu_k_ptr + sequence)
    prefix = k_len - q_len
    kv_head = head // (H_Q // H_KV)
    q_rows = block * M + gl.arange(0, M, layout=gl.SliceLayout(1, QA))
    q_dims = gl.arange(0, D, layout=gl.SliceLayout(0, QA))
    q = gl.load(q_ptr + (q_start + q_rows[:, None]) * q_stride_t + head * q_stride_h + q_dims[None, :], mask=q_rows[:, None] < q_len, other=0)
    rows = block * M + gl.arange(0, M, layout=gl.SliceLayout(1, MMA))
    columns = gl.arange(0, N, layout=gl.SliceLayout(0, MMA))
    key_dims = gl.arange(0, D, layout=gl.SliceLayout(1, KB))
    key_cols = gl.arange(0, N, layout=gl.SliceLayout(0, KB))
    value_rows = gl.arange(0, N, layout=gl.SliceLayout(1, KB))
    value_dims = gl.arange(0, D, layout=gl.SliceLayout(0, KB))
    accumulator = gl.full((M, D), 0, gl.float32, layout=MMA)
    maximum = gl.full((M,), -float("inf"), gl.float32, layout=gl.SliceLayout(1, MMA))
    denominator = gl.full((M,), 0, gl.float32, layout=gl.SliceLayout(1, MMA))
    k_buffers = gl.allocate_shared_memory(gl.int8, [STAGES, N, D], gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8))
    v_buffers = gl.allocate_shared_memory(gl.int8, [STAGES, N, D], gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8))
    end = gl.minimum(k_len, prefix + (block + 1) * M)
    first_page = gl.load(tables_ptr + sequence * bt_stride, mask=end > 0, other=0)
    if end > 0:
        _stage_cache(k_ptr, v_ptr, k_buffers.index(0), v_buffers.index(0), first_page * PAGE, kv_head, H_KV, D, N)

    for iteration in range(gl.cdiv(end, N)):
        start = iteration * N
        stage = iteration % STAGES
        k_smem = k_buffers.index(stage)
        v_smem = v_buffers.index(stage)
        async_copy.wait_group(0)
        _barrier()
        page = gl.load(tables_ptr + sequence * bt_stride + start // PAGE)
        token = start % PAGE
        next_start = start + N
        more = next_start < end
        next_stage = (iteration + 1) % STAGES
        next_page = gl.load(tables_ptr + sequence * bt_stride + next_start // PAGE, mask=more, other=0)
        next_slot = next_page * PAGE + next_start % PAGE
        # Transfer the next tile while this tile's tensor-core work executes.
        if more:
            _stage_cache(k_ptr, v_ptr, k_buffers.index(next_stage), v_buffers.index(next_stage), next_slot, kv_head, H_KV, D, N)
        k = k_smem.permute((1, 0)).load(KB).to(gl.float32)
        k_scale = gl.load(ks_ptr + page * scale_stride_block + (token + key_cols[None, :]) * scale_stride_token + kv_head * scale_stride_head + key_dims[:, None] // 32, mask=start + key_cols[None, :] < k_len, other=0).to(gl.float32)
        k *= k_scale
        k_high = k.to(q.dtype)
        k_low = (k - k_high.to(gl.float32)).to(q.dtype)
        scores = mma_v2(q, k_high, gl.full((M, N), 0, gl.float32, layout=MMA))
        scores = mma_v2(q, k_low, scores) * SCALE
        valid = (rows[:, None] < q_len) & (start + columns[None, :] < k_len) & (start + columns[None, :] <= prefix + rows[:, None])
        scores = gl.where(valid, scores, -float("inf"))
        next_maximum = gl.maximum(maximum, gl.max(scores, 1))
        correction = gl.exp2((maximum - next_maximum) * 1.4426950408889634)
        p = gl.exp2((scores - next_maximum[:, None]) * 1.4426950408889634)
        denominator = denominator * correction + gl.sum(p, 1)
        p_high = p.to(q.dtype)
        p_low = (p - p_high.to(gl.float32)).to(q.dtype)
        p_high = gl.convert_layout(p_high, QA)
        p_low = gl.convert_layout(p_low, QA)
        v = v_smem.load(KB).to(gl.float32)
        v_scale = gl.load(vs_ptr + page * scale_stride_block + (token + value_rows[:, None]) * scale_stride_token + kv_head * scale_stride_head + value_dims[None, :] // 32, mask=start + value_rows[:, None] < k_len, other=0).to(gl.float32)
        v *= v_scale
        v_high = v.to(q.dtype)
        v_low = (v - v_high.to(gl.float32)).to(q.dtype)
        accumulator *= correction[:, None]
        accumulator = mma_v2(p_high, v_high, accumulator)
        accumulator = mma_v2(p_low, v_high, accumulator)
        accumulator = mma_v2(p_high, v_low, accumulator)
        accumulator = mma_v2(p_low, v_low, accumulator)
        maximum = next_maximum
        # All readers finish before a pipeline slot can be overwritten.
        _barrier()
    dims = gl.arange(0, D, layout=gl.SliceLayout(0, MMA))
    gl.store(out_ptr + (q_start + rows[:, None]) * out_stride_t + head * out_stride_h + dims[None, :], accumulator / denominator[:, None], mask=rows[:, None] < q_len)


@gluon.jit(do_not_specialize=("Q_PER_SEQ", "TABLE_WIDTH"), do_not_specialize_on_alignment=("Q_PER_SEQ", "TABLE_WIDTH"))
def q8_grouped_async_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tables_ptr, lengths_ptr, partial_ptr, maximum_ptr, sum_ptr,
                         Q_PER_SEQ, H_Q: gl.constexpr, H_KV: gl.constexpr, D: gl.constexpr, PAGE: gl.constexpr,
                         TABLE_WIDTH, SPLITS: gl.constexpr, SCALE: gl.constexpr):
    M: gl.constexpr = 16
    N: gl.constexpr = 32
    STAGES: gl.constexpr = 1
    # Short split-context loops benefit from one buffer and higher occupancy.
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8])
    QA: gl.constexpr = gl.DotOperandLayout(0, MMA, 4)
    KB: gl.constexpr = gl.DotOperandLayout(1, MMA, 4)
    sequence, kv_head, tile = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    split, query_block = tile % SPLITS, tile // SPLITS
    group: gl.constexpr = H_Q // H_KV
    q_rows = query_block * M + gl.arange(0, M, layout=gl.SliceLayout(1, QA))
    q_dims = gl.arange(0, D, layout=gl.SliceLayout(0, QA))
    q_index = sequence * Q_PER_SEQ + q_rows // group
    q_head = kv_head * group + q_rows % group
    q = gl.load(q_ptr + (q_index[:, None] * H_Q + q_head[:, None]) * D + q_dims[None, :], mask=q_rows[:, None] < Q_PER_SEQ * group, other=0)
    rows = query_block * M + gl.arange(0, M, layout=gl.SliceLayout(1, MMA))
    columns = gl.arange(0, N, layout=gl.SliceLayout(0, MMA))
    key_dims = gl.arange(0, D, layout=gl.SliceLayout(1, KB))
    key_cols = gl.arange(0, N, layout=gl.SliceLayout(0, KB))
    value_rows = gl.arange(0, N, layout=gl.SliceLayout(1, KB))
    value_dims = gl.arange(0, D, layout=gl.SliceLayout(0, KB))
    accumulator = gl.full((M, D), 0, gl.float32, layout=MMA)
    maximum = gl.full((M,), -1.0e20, gl.float32, layout=gl.SliceLayout(1, MMA))
    denominator = gl.full((M,), 0, gl.float32, layout=gl.SliceLayout(1, MMA))
    k_buffers = gl.allocate_shared_memory(gl.int8, [STAGES, N, D], gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8))
    v_buffers = gl.allocate_shared_memory(gl.int8, [STAGES, N, D], gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8))
    context_len = gl.load(lengths_ptr + sequence)
    prefix_len = context_len - Q_PER_SEQ
    tiles_per_split = gl.cdiv(gl.cdiv(context_len, N), SPLITS)
    begin = split * tiles_per_split * N
    end = gl.minimum(context_len, (split + 1) * tiles_per_split * N)
    first_page = gl.load(tables_ptr + sequence * TABLE_WIDTH + begin // PAGE, mask=end > begin, other=0)
    first_slot = first_page * PAGE + begin % PAGE
    if end > begin:
        _stage_cache(k_ptr, v_ptr, k_buffers.index(0), v_buffers.index(0), first_slot, kv_head, H_KV, D, N)
    for iteration in range(gl.cdiv(end - begin, N)):
        start = begin + iteration * N
        stage = iteration % STAGES
        k_smem, v_smem = k_buffers.index(stage), v_buffers.index(stage)
        async_copy.wait_group(0)
        _barrier()
        page = gl.load(tables_ptr + sequence * TABLE_WIDTH + start // PAGE)
        token = start % PAGE
        next_start = start + N
        more = next_start < end
        next_page = gl.load(tables_ptr + sequence * TABLE_WIDTH + next_start // PAGE, mask=more, other=0)
        next_slot = next_page * PAGE + next_start % PAGE
        k = k_smem.permute((1, 0)).load(KB).to(gl.float32)
        k *= gl.load(ks_ptr + ((page * PAGE + token + key_cols[None, :]) * H_KV + kv_head) * (D // 32) + key_dims[:, None] // 32, mask=start + key_cols[None, :] < end, other=0).to(gl.float32)
        k_high = k.to(q.dtype)
        k_low = (k - k_high.to(gl.float32)).to(q.dtype)
        scores = mma_v2(q, k_high, gl.full((M, N), 0, gl.float32, layout=MMA))
        scores = mma_v2(q, k_low, scores) * SCALE
        valid = (rows[:, None] < Q_PER_SEQ * group) & (start + columns[None, :] < end) & (start + columns[None, :] <= prefix_len + rows[:, None] // group)
        scores = gl.where(valid, scores, -1.0e20)
        next_maximum = gl.maximum(maximum, gl.max(scores, 1))
        correction = gl.exp2((maximum - next_maximum) * 1.4426950408889634)
        p = gl.where(valid, gl.exp2((scores - next_maximum[:, None]) * 1.4426950408889634), 0.0)
        denominator = denominator * correction + gl.sum(p, 1)
        p_high = p.to(q.dtype)
        p_low = (p - p_high.to(gl.float32)).to(q.dtype)
        p_high = gl.convert_layout(p_high, QA)
        p_low = gl.convert_layout(p_low, QA)
        v = v_smem.load(KB).to(gl.float32)
        v *= gl.load(vs_ptr + ((page * PAGE + token + value_rows[:, None]) * H_KV + kv_head) * (D // 32) + value_dims[None, :] // 32, mask=start + value_rows[:, None] < end, other=0).to(gl.float32)
        v_high = v.to(q.dtype)
        v_low = (v - v_high.to(gl.float32)).to(q.dtype)
        accumulator *= correction[:, None]
        accumulator = mma_v2(p_high, v_high, accumulator)
        accumulator = mma_v2(p_low, v_high, accumulator)
        accumulator = mma_v2(p_high, v_low, accumulator)
        accumulator = mma_v2(p_low, v_low, accumulator)
        maximum = next_maximum
        _barrier()
        if more:
            _stage_cache(k_ptr, v_ptr, k_buffers.index(0), v_buffers.index(0), next_slot, kv_head, H_KV, D, N)
    dims = gl.arange(0, D, layout=gl.SliceLayout(0, MMA))
    index = ((sequence * Q_PER_SEQ + rows // group) * H_Q + kv_head * group + rows % group) * SPLITS + split
    gl.store(partial_ptr + index[:, None] * D + dims[None, :], accumulator, mask=rows[:, None] < Q_PER_SEQ * group)
    gl.store(maximum_ptr + index, maximum, mask=rows < Q_PER_SEQ * group)
    gl.store(sum_ptr + index, denominator, mask=rows < Q_PER_SEQ * group)


def q8_paged_prefill(q, k_cache, v_cache, k_scale, v_scale, context, softmax_scale):
    output = torch.empty_like(q)
    q8_prefill_async_kernel[(context.cu_seqlens_q.numel() - 1, q.shape[1], triton.cdiv(q.shape[0], 16))](
        q, k_cache, v_cache, k_scale, v_scale, context.block_tables, context.cu_seqlens_q, context.cu_seqlens_k, output,
        q.stride(0), q.stride(1), k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), context.block_tables.stride(0),
        output.stride(0), output.stride(1), softmax_scale, q.shape[1], k_cache.shape[2], q.shape[2], k_cache.shape[1], num_warps=4, num_stages=1)
    return output


def q8_grouped_partials(q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, partial, maximum, denominator, splits, softmax_scale):
    queries = q.shape[0] // context_lens.numel()
    query_rows = queries * (q.shape[1] // k_cache.shape[2])
    q8_grouped_async_kernel[(context_lens.numel(), k_cache.shape[2], triton.cdiv(query_rows, 16) * splits)](
        q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, partial, maximum, denominator,
        queries, q.shape[1], k_cache.shape[2], q.shape[2], k_cache.shape[1], block_tables.shape[1], splits, softmax_scale, num_warps=4, num_stages=1)


# Import-time registration also covers direct module use and CUDA graph warmup.
from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
