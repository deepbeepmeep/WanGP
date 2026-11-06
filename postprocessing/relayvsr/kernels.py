# Adapted from RelayVSR (https://github.com/kopperx/RelayVSR, Apache-2.0, see LICENSE in this folder): the windowed attentions of its keyframe
# model and of its decoder, rewritten to compile once per model. Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""RelayVSR's local attentions of one query frame, tensors (1, tokens, heads, head_dim) read through strides:
- shifted_window_attention: the keyframe model's window of (rows, columns) tokens, shifted inside the frame at its borders, in every key frame;
- centered_window_attention: the decoder's windows centered on each query and cut at the frame's borders: a window for one or two condition
  planes, another for the video planes (grouped query attention).
Triton kernels on CUDA (FP32 scores and sums, probabilities rounded to the value dtype as in flash attention), compiled once per model whatever
the frame size; the PyTorch path attends bands of query rows with SDPA and a mask of their keys."""
import torch
import torch.nn.functional as F

from shared.kernels import dispatch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError, RuntimeError):
    triton = None

BAND_ROWS = 4  # query rows of a PyTorch band
TILES = {"shifted": (8, 16, 64, 8, 2), "centered": (8, 16, 32, 4, 2)}  # Triton programs: query rows, query columns, keys per block, warps, stages (static, measured)
LOG2E = 1.4426950408889634


def _triton_eligible(q):
    return triton is not None and q.shape[-1] in (64, 128) and dispatch.cuda_inference(q, dtypes=(torch.bfloat16, torch.float16))


if triton is not None:
    _RUNTIME = ["sq_t", "sq_h", "sk_t", "sk_h", "so_t", "so_h", "height", "width", "key_frames", "wh", "ww", "head_offset"]

    @triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME)
    def _shifted_window(Q, K, V, OUT, sq_t, sq_h, sk_t, sk_h, so_t, so_h, height, width, key_frames, wh, ww, head_offset,
                        SCALE: tl.constexpr, DIM: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
        row_tile, segment, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        lanes = tl.arange(0, BH * BW)
        qy, qx = row_tile * BH + lanes // BW, segment * BW + lanes % BW
        valid_q = (qy < height) & (qx < width)
        d = tl.arange(0, DIM)
        q = tl.load(Q + (qy * width + qx)[:, None].to(tl.int64) * sq_t + head * sq_h + d[None, :], valid_q[:, None], 0)
        top = tl.minimum(tl.maximum(qy - wh // 2, 0), height - wh)
        left = tl.minimum(tl.maximum(qx - ww // 2, 0), width - ww)
        base_top = tl.minimum(tl.maximum(row_tile * BH - wh // 2, 0), height - wh)
        base_left = tl.minimum(tl.maximum(segment * BW - ww // 2, 0), width - ww)
        acc = tl.zeros((BH * BW, DIM), tl.float32)
        maximum = tl.full((BH * BW,), -float("inf"), tl.float32)
        total = tl.zeros((BH * BW,), tl.float32)
        tokens = height * width
        kv_head = head + head_offset
        for frame in range(key_frames):
            for offset in range(0, ROWS * COLS, BN):
                p = offset + tl.arange(0, BN)
                ky, kx = base_top + p // COLS, base_left + p % COLS
                valid_k = (p < ROWS * COLS) & (ky < height) & (kx < width) & (ky < base_top + wh + BH - 1)
                index = (frame * tokens + ky * width + kx).to(tl.int64)
                k = tl.load(K + index[None, :] * sk_t + kv_head * sk_h + d[:, None], valid_k[None, :], 0)
                scores = tl.dot(q, k) * SCALE
                allowed = valid_k[None, :] & (ky[None, :] >= top[:, None]) & (ky[None, :] < top[:, None] + wh) & (kx[None, :] >= left[:, None]) & (kx[None, :] < left[:, None] + ww)
                scores = tl.where(allowed, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                safe = tl.where(new_max == -float("inf"), 0.0, new_max)
                probabilities = tl.exp2(scores - safe[:, None])
                correction = tl.exp2(maximum - safe)
                v = tl.load(V + index[:, None] * sk_t + kv_head * sk_h + d[None, :], valid_k[:, None], 0)
                acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
                total = total * correction + tl.sum(probabilities, 1)
                maximum = new_max
        out = acc / total[:, None]
        tl.store(OUT + (qy * width + qx)[:, None].to(tl.int64) * so_t + head * so_h + d[None, :], out.to(OUT.dtype.element_ty), valid_q[:, None])

    _CENTERED_RUNTIME = ["sq_t", "sq_h", "sk_t", "sk_h", "so_t", "so_h", "height", "width", "planes_a", "video_planes", "prev_slot", "cur_slot", "slot_stride", "head_offset"]

    @triton.jit
    def _centered_plane(acc, maximum, total, q, K, V, plane_offset, sk_t, kv_head, sk_h, qy, qx, tile_top, tile_left, height, width,
                        SCALE: tl.constexpr, DIM: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr, BN: tl.constexpr, WIN: tl.constexpr, COLS: tl.constexpr):
        d = tl.arange(0, DIM)
        top, left = tile_top - WIN // 2, tile_left - WIN // 2
        rows = WIN + BH - 1
        for offset in range(0, rows * COLS, BN):
            p = offset + tl.arange(0, BN)
            ky, kx = top + p // COLS, left + p % COLS
            valid_k = (p < rows * COLS) & (ky >= 0) & (ky < height) & (kx >= 0) & (kx < width)
            index = (ky * width + kx).to(tl.int64)
            k = tl.load(K + plane_offset + index[None, :] * sk_t + kv_head * sk_h + d[:, None], valid_k[None, :], 0)
            scores = tl.dot(q, k) * SCALE
            allowed = (valid_k[None, :] & (ky[None, :] >= qy[:, None] - WIN // 2) & (ky[None, :] < qy[:, None] + WIN // 2)
                       & (kx[None, :] >= qx[:, None] - WIN // 2) & (kx[None, :] < qx[:, None] + WIN // 2))
            scores = tl.where(allowed, scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            safe = tl.where(new_max == -float("inf"), 0.0, new_max)
            probabilities = tl.exp2(scores - safe[:, None])
            correction = tl.exp2(maximum - safe)
            v = tl.load(V + plane_offset + index[:, None] * sk_t + kv_head * sk_h + d[None, :], valid_k[:, None], 0)
            acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
            total = total * correction + tl.sum(probabilities, 1)
            maximum = new_max
        return acc, maximum, total

    @triton.jit(do_not_specialize=_CENTERED_RUNTIME, do_not_specialize_on_alignment=_CENTERED_RUNTIME)
    def _centered_window(Q, KA, VA, KB, VB, KV, VV, OUT, sq_t, sq_h, sk_t, sk_h, so_t, so_h, height, width, planes_a, video_planes, prev_slot, cur_slot,
                         slot_stride, head_offset, SCALE: tl.constexpr, GROUP: tl.constexpr, DIM: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr,
                         BN: tl.constexpr, WIN_A: tl.constexpr, COLS_A: tl.constexpr, WIN_V: tl.constexpr, COLS_V: tl.constexpr):
        row_tile, segment, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        lanes = tl.arange(0, BH * BW)
        qy, qx = row_tile * BH + lanes // BW, segment * BW + lanes % BW
        valid_q = (qy < height) & (qx < width)
        d = tl.arange(0, DIM)
        q = tl.load(Q + (qy * width + qx)[:, None].to(tl.int64) * sq_t + head * sq_h + d[None, :], valid_q[:, None], 0)
        acc = tl.zeros((BH * BW, DIM), tl.float32)
        maximum = tl.full((BH * BW,), -float("inf"), tl.float32)
        total = tl.zeros((BH * BW,), tl.float32)
        kv_head = head // GROUP + head_offset
        tile_top, tile_left = row_tile * BH, segment * BW
        if planes_a > 0:
            acc, maximum, total = _centered_plane(acc, maximum, total, q, KA, VA, 0, sk_t, kv_head, sk_h, qy, qx, tile_top, tile_left, height, width, SCALE, DIM, BH, BW, BN, WIN_A, COLS_A)
        if planes_a > 1:
            acc, maximum, total = _centered_plane(acc, maximum, total, q, KB, VB, 0, sk_t, kv_head, sk_h, qy, qx, tile_top, tile_left, height, width, SCALE, DIM, BH, BW, BN, WIN_A, COLS_A)
        if video_planes > 1:
            acc, maximum, total = _centered_plane(acc, maximum, total, q, KV, VV, prev_slot.to(tl.int64) * slot_stride, sk_t, kv_head, sk_h, qy, qx, tile_top, tile_left, height, width,
                                                  SCALE, DIM, BH, BW, BN, WIN_V, COLS_V)
        if video_planes > 0:
            acc, maximum, total = _centered_plane(acc, maximum, total, q, KV, VV, cur_slot.to(tl.int64) * slot_stride, sk_t, kv_head, sk_h, qy, qx, tile_top, tile_left, height, width,
                                                  SCALE, DIM, BH, BW, BN, WIN_V, COLS_V)
        out = acc / total[:, None]
        tl.store(OUT + (qy * width + qx)[:, None].to(tl.int64) * so_t + head * so_h + d[None, :], out.to(OUT.dtype.element_ty), valid_q[:, None])

    from shared.kernels.triton_compilation_log import install_triton_compilation_logger

    install_triton_compilation_logger()


def _check(*tensors):
    for tensor in tensors:
        if tensor.stride(-1) != 1:
            raise ValueError("RelayVSR windowed attention needs contiguous head dimensions")


def shifted_window_attention(q, k, v, height, width, window, key_frames, head_offset=0):
    """q (1, height * width, heads, head_dim): the queries of the current frame; k, v (1, frames * height * width, kv heads, head_dim), the
    keys of key_frames frames, the query heads attending kv heads head_offset + head. Each query attends in every key frame the window
    (rows, columns) of keys around it, shifted inside the frame at its borders (clamped to the frame). Returns (1, tokens, heads, head_dim)."""
    _check(q, k, v)
    wh, ww = min(window[0], height), min(window[1], width)
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if _triton_eligible(q) and k.stride() == v.stride():
        rows, columns, block, warps, stages = TILES["shifted"]
        cols = window[1] + columns - 1  # the key columns of a program's queries
        grid = (triton.cdiv(height, rows), triton.cdiv(width, columns), q.shape[2])
        _shifted_window[grid](q, k, v, out, q.stride(1), q.stride(2), k.stride(1), k.stride(2), out.stride(1), out.stride(2), height, width, key_frames,
                              wh, ww, head_offset, q.shape[-1] ** -0.5 * LOG2E, q.shape[-1], rows, columns, block, window[0] + rows - 1, cols,
                              num_warps=warps, num_stages=stages)
        return out
    heads, tokens = q.shape[2], height * width
    k, v = (t[:, :key_frames * tokens, head_offset:head_offset + heads].unflatten(1, (key_frames, height, width)) for t in (k, v))
    columns = torch.arange(width, device=q.device)
    left = (columns - ww // 2).clamp_(0, width - ww)
    column_mask = (columns[None, :] >= left[:, None]) & (columns[None, :] < left[:, None] + ww)  # (query column, key column)
    for start in range(0, height, BAND_ROWS):
        stop = min(start + BAND_ROWS, height)
        rows = torch.arange(start, stop, device=q.device)
        top = (rows - wh // 2).clamp_(0, height - wh)
        first, last = int(top[0]), int(top[-1]) + wh
        key_rows = torch.arange(first, last, device=q.device)
        row_mask = (key_rows[None, :] >= top[:, None]) & (key_rows[None, :] < top[:, None] + wh)  # (query row, key row)
        mask = (row_mask[:, None, :, None] & column_mask[None, :, None, :]).flatten(0, 1).flatten(1)  # (band queries, band key rows x columns)
        band_k, band_v = (t[:, :, first:last].flatten(1, 3).transpose(1, 2) for t in (k, v))
        band_q = q[:, start * width:stop * width].transpose(1, 2)
        out[:, start * width:stop * width] = F.scaled_dot_product_attention(band_q, band_k, band_v, attn_mask=mask.repeat(1, key_frames)).transpose(1, 2)
    return out


def centered_window_attention(q, planes_a, window_a, video, window_v, prev_slot, cur_slot, height, width, group, head_offset=0):
    """q (1, height * width, heads, head_dim): the queries of one frame. planes_a: one or two (k, v) planes (1, tokens, kv heads, head_dim)
    attended in the window window_a centered on each query; video: (k, v) slots (slots, tokens, kv heads, head_dim) whose slots prev_slot and
    cur_slot (-1: none) are attended in the window window_v. Windows cut at the frame's borders; query head h attends kv head head_offset +
    h // group (group: query heads per kv head). Returns (1, tokens, heads, head_dim)."""
    k0 = planes_a[0][0]
    _check(q, k0)
    heads = q.shape[2]
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    video_planes = (prev_slot >= 0) + (cur_slot >= 0) if video is not None else 0
    if _triton_eligible(q):
        ka, va = planes_a[0]
        kb, vb = planes_a[1] if len(planes_a) > 1 else planes_a[0]
        kvideo, vvideo = video if video is not None else (ka, va)
        for tensor in (va, kb, vb, kvideo, vvideo):
            if tensor.stride()[1:] != ka.stride()[1:]:
                raise ValueError("RelayVSR decoder planes must share their layout")
        rows, columns, block, warps, stages = TILES["centered"]
        cols_a, cols_v = (window + columns - 1 for window in (window_a, window_v))
        grid = (triton.cdiv(height, rows), triton.cdiv(width, columns), heads)
        _centered_window[grid](q, ka, va, kb, vb, kvideo, vvideo, out, q.stride(1), q.stride(2), ka.stride(1), ka.stride(2), out.stride(1), out.stride(2),
                               height, width, len(planes_a), int(video_planes), max(prev_slot, 0), max(cur_slot, 0), kvideo.stride(0), head_offset,
                               q.shape[-1] ** -0.5 * LOG2E, group, q.shape[-1], rows, columns, block, window_a, cols_a, window_v, cols_v,
                               num_warps=warps, num_stages=stages)
        return out
    planes = [(k, v, window_a) for k, v in planes_a]
    if video is not None:
        planes += [(video[0][slot:slot + 1], video[1][slot:slot + 1], window_v) for slot in (prev_slot, cur_slot) if slot >= 0]
    rows_all, columns = torch.arange(height, device=q.device), torch.arange(width, device=q.device)
    for start in range(0, height, BAND_ROWS):
        stop = min(start + BAND_ROWS, height)
        rows = rows_all[start:stop]
        keys, values, masks = [], [], []
        for k, v, window in planes:
            first, last = max(0, start - window // 2), min(height, stop - 1 + window // 2)
            key_rows = rows_all[first:last]
            row_mask = (key_rows[None, :] >= rows[:, None] - window // 2) & (key_rows[None, :] < rows[:, None] + window // 2)
            column_mask = (columns[None, :] >= columns[:, None] - window // 2) & (columns[None, :] < columns[:, None] + window // 2)
            masks.append((row_mask[:, None, :, None] & column_mask[None, :, None, :]).flatten(0, 1).flatten(1))
            selected = [t.view(1, height, width, *t.shape[2:])[:, first:last, :, head_offset:head_offset + (heads // group)].flatten(1, 2) for t in (k, v)]
            keys.append(selected[0])
            values.append(selected[1])
        band_k = torch.cat(keys, 1).transpose(1, 2).repeat_interleave(group, dim=1)
        band_v = torch.cat(values, 1).transpose(1, 2).repeat_interleave(group, dim=1)
        band_q = q[:, start * width:stop * width].transpose(1, 2)
        out[:, start * width:stop * width] = F.scaled_dot_product_attention(band_q, band_k, band_v, attn_mask=torch.cat(masks, 1)).transpose(1, 2)
    return out


__all__ = ["centered_window_attention", "shifted_window_attention"]
