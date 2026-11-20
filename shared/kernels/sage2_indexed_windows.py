# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prepare independent indexed windows together for Sage2's SM120 INT8/FP8 CUDA forward.

Q, K and V are borrowed strided (token, head, channel) views. Each window keeps its own key mean,
quantization scales and true sequence lengths. Optional global queries read a contiguous range
of the same K/V rows in their original precision, in the window preparation and scatter launches.
The native scale and FP8 layouts follow SageAttention's CUDA kernels (https://github.com/thu-ml/SageAttention, Apache-2.0).
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .sage2_sm120 import scaled_e4m3


@triton.jit
def _global_partial(Q, K, V, PART, group, head, first, length, rows: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                    qt, qh, kt, kh, vt, vh):
    # The queries are the final rows of this key range. Two 128-key tiles share the query load.
    r = tl.arange(0, 16)
    d = tl.arange(0, D)
    q = tl.load(Q + (first + length - rows + r[:, None]) * qt + head * qh + d[None, :], r[:, None] < rows, 0)
    hi = tl.full((16,), -float("inf"), tl.float32)
    total = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, D), 0, tl.float32)
    for tile in tl.static_range(2):
        i = group * 256 + tile * 128 + tl.arange(0, 128)
        k = tl.load(K + (first + i[:, None]) * kt + head * kh + d[None, :], i[:, None] < length, 0)
        v = tl.load(V + (first + i[:, None]) * vt + head * vh + d[None, :], i[:, None] < length, 0)
        scores = tl.dot(q, tl.trans(k)) * (1.4426950408889634 * D**-0.5)
        scores = tl.where(i[None, :] < length, scores, -float("inf"))
        new_hi = tl.maximum(hi, tl.max(scores, 1))
        weights = tl.exp2(scores - new_hi[:, None])
        scale = tl.exp2(hi - new_hi)
        total = total * scale + tl.sum(weights, 1)
        high = weights.to(v.dtype)
        acc = acc * scale[:, None] + tl.dot(high, v)
        acc += tl.dot((weights - high.to(tl.float32)).to(v.dtype), v)
        hi = new_hi
    at = PART + ((group * H + head) * rows + r) * (D + 2)
    tl.store(at[:, None] + d[None, :], acc, r[:, None] < rows)
    tl.store(at + D, hi, r < rows)
    tl.store(at + D + 1, total, r < rows)


@triton.jit(do_not_specialize=["windows", "first", "length", "tiles"])
def _prepare(Q, K, V, QI, KI, META, Q8, QS, SUM, PEAK, PART, windows, first, length,
             H: tl.constexpr, D: tl.constexpr, tiles, ROWS: tl.constexpr, qt, qh, kt, kh, vt, vh):
    task, head = tl.program_id(0), tl.program_id(1)
    if task >= windows * tiles:
        _global_partial(Q, K, V, PART, task - windows * tiles, head, first.to(tl.int64), length, ROWS, H, D, qt, qh, kt, kh, vt, vh)
        return
    win, block = task // tiles, task % tiles
    info = META + win * 7
    m, n = tl.load(info), tl.load(info + 1)
    if block * 64 >= tl.maximum(m, n):
        return
    qo, ko = tl.load(info + 2), tl.load(info + 3)
    r = block * 64 + tl.arange(0, 64)
    d = tl.arange(0, D)
    qi = tl.load(QI + qo + r, r < m, 0).to(tl.int64)
    ki = tl.load(KI + ko + r, r < n, 0).to(tl.int64)
    q = tl.load(Q + qi[:, None] * qt + head * qh + d[None, :], (r < m)[:, None], 0).to(tl.float32)
    k = tl.load(K + ki[:, None] * kt + head * kh + d[None, :], (r < n)[:, None], 0).to(tl.float32)
    v = tl.load(V + ki[:, None] * vt + head * vh + d[None, :], (r < n)[:, None], 0).to(tl.float32)
    at = ((win * H + head) * tiles + block) * D + d
    tl.store(SUM + at, tl.sum(k, 0))
    tl.store(PEAK + at, tl.max(tl.abs(v), 0))
    # Sage2's CUDA per-warp Q quantizer uses one scale per 32 rows, rounded to nearest even.
    q = tl.reshape(q, (2, 32 * D))
    top = tl.maximum(tl.max(tl.abs(q), 1), 1.e-7)
    codes = tl.reshape(libdevice.rint(q * (127. / top[:, None])).to(tl.int8), (64, D))
    tl.store(Q8 + (qo * H + head * m + r[:, None]) * D + d[None, :], codes, (r < m)[:, None])
    tile = block * 2 + tl.arange(0, 2)
    qtiles = tl.cdiv(m, 128) * 4
    tl.store(QS + tl.load(info + 5) * H + head * qtiles + tile, top / 127., tile < qtiles)


@triton.jit(do_not_specialize=["tiles"])
def _pack(K, V, KI, META, K8, V8, KS, VS, SUM, PEAK, H: tl.constexpr, D: tl.constexpr, tiles, kt, kh, vt, vh):
    block, head, win = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    info = META + win * 7
    n = tl.load(info + 1)
    count = tl.cdiv(n, 64)
    if block >= count:
        return
    ko, vo = tl.load(info + 3), tl.load(info + 4)
    # At most 2,048 keys per window. One compiled reduction also serves the shorter border windows.
    b, d = tl.arange(0, 32), tl.arange(0, D)
    at = ((win * H + head) * tiles + b[:, None]) * D + d[None, :]
    mean = (tl.sum(tl.load(SUM + at, (b < count)[:, None], 0), 0) / n).to(K.dtype.element_ty).to(tl.float32)
    peak = tl.max(tl.load(PEAK + at, (b < count)[:, None], 0), 0)
    r = block * 64 + tl.arange(0, 64)
    ki = tl.load(KI + ko + r, r < n, 0).to(tl.int64)
    k = tl.load(K + ki[:, None] * kt + head * kh + d[None, :], (r < n)[:, None], 0).to(tl.float32)
    k = tl.where((r < n)[:, None], k - mean[None, :], 0)
    top = tl.maximum(tl.max(tl.max(tl.abs(k), 1), 0), 1.e-7)
    tl.store(KS + tl.load(info + 6) * H + head * count + block, top / 127.)
    tl.store(K8 + (ko * H + head * n + r[:, None]) * D + d[None, :], libdevice.rint(k * (127. / top)).to(tl.int8), (r < n)[:, None])
    # Permutation within each 16 keys and zero padding through the last 64, as in the CUDA FP8 packer.
    src = r // 16 * 16 + r % 4 // 2 * 8 + r % 16 // 4 * 2 + r % 2
    vi = tl.load(KI + ko + src, src < n, 0).to(tl.int64)
    v = tl.load(V + vi[:, None] * vt + head * vh + d[None, :], (src < n)[:, None], 0).to(tl.float32)
    if block == 0:
        tl.store(VS + (win * H + head) * D + d, peak / 2.25)
    fp8 = scaled_e4m3(v, peak, V.dtype.element_ty == tl.bfloat16)
    tl.store(V8 + vo * H * D + (head * D + d[:, None]) * (count * 64) + r[None, :], tl.trans(fp8))


@triton.jit(do_not_specialize=["windows", "groups", "tiles"])
def _scatter(O, INDEX, META, DST, PART, GLOBAL, windows, groups, H: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
             tiles, REDUCE: tl.constexpr):
    task, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    if task >= windows * tiles:
        row = task - windows * tiles
        group = tl.arange(0, REDUCE)
        at = PART + ((group * H + head) * ROWS + row) * (D + 2)
        hi = tl.load(at + D, group < groups, -float("inf"))
        weights = tl.exp2(hi - tl.max(hi, 0))
        total = tl.sum(weights * tl.load(at + D + 1, group < groups, 0), 0)
        acc = tl.load(at[:, None] + d[None, :], (group < groups)[:, None], 0)
        tl.store(GLOBAL + (row * H + head) * D + d, tl.sum(acc * weights[:, None], 0) / total)
        return
    win, tile = task // tiles, task % tiles
    info = META + win * 7
    m = tl.load(info)
    if tile * 32 >= m:
        return
    base = tl.load(info + 2)
    row = tile * 32 + tl.arange(0, 32)
    dst = tl.load(INDEX + base + row, row < m, 0).to(tl.int64)
    val = tl.load(O + (base * H + head * m + row[:, None]) * D + d[None, :], (row < m)[:, None], 0)
    tl.store(DST + (dst[:, None] * H + head) * D + d[None, :], val, (row < m)[:, None])


class _WindowBatch:
    """One set of windows sharing preparation/scatter launches. Equal lengths share a native forward."""

    def __init__(self, groups, heads, dim, device):
        self.heads, self.dim = heads, dim
        self.windows = sum(q.shape[0] for q, _, _ in groups)
        self.tiles = max(triton.cdiv(k.shape[1], 64) for _, k, _ in groups)
        self.out_tiles = triton.cdiv(max(q.shape[1] for q, _, _ in groups), 32)
        # Per window: true Q and K lengths, offsets in Q, K and padded V rows, then Q and K scale offsets.
        metadata, self.layouts = [], []
        qrows = krows = vrows = qscales = kscales = 0
        width = heads * dim
        for queries, keys, _ in groups:
            batch, m = queries.shape
            n = keys.shape[1]
            qblocks, kblocks = triton.cdiv(m, 128) * 4, triton.cdiv(n, 64)
            padded = kblocks * 64
            # Contiguous HND views for this native launch within a workspace shared across steps.
            self.layouts.append(((qrows * width, (batch, heads, m, dim)), (krows * width, (batch, heads, n, dim)),
                                 (vrows * width, (batch, heads, dim, padded)), (qrows * width, (batch, heads, m, dim)),
                                 (qscales * heads, (batch, heads, qblocks)), (kscales * heads, (batch, heads, kblocks)),
                                 (len(metadata) * width, (batch, heads, dim))))
            for _ in range(batch):
                metadata.append((m, n, qrows, krows, vrows, qscales, kscales))
                qrows += m
                krows += n
                vrows += padded
                qscales += qblocks
                kscales += kblocks
        # Geometry stays on the host until all the windows can be uploaded together.
        indices = [torch.cat([group[i].reshape(-1) for group in groups]).int() for i in range(3)]
        info = torch.tensor(metadata, dtype=torch.int32, device=indices[0].device).flatten()
        packed = torch.cat((*indices, info)).to(device)
        query_ids, key_ids, output_ids, self.meta = packed.split([x.numel() for x in (*indices, info)])
        self.indices = query_ids, key_ids, output_ids
        self.sizes = (qrows * width, krows * width, vrows * width, qrows * width,
                      qscales * heads, kscales * heads, self.windows * width, self.windows * self.tiles * width)

    def bind(self, workspace, peaks, partials, operator, global_rows):
        self.q, self.k, self.v, self.out, self.qs, self.ks, self.vs, self.sums = workspace
        self.peaks, self.partials = peaks, partials
        self.operator, self.global_rows = operator, global_rows
        self.reduce = triton.next_power_of_2(partials.shape[0])
        self.calls = []
        for layout in self.layouts:
            views = [buffer.narrow(0, start, math.prod(shape)).view(shape) for buffer, (start, shape) in zip(workspace, layout)]
            self.calls.append((*views, 1, 0, 2, self.dim**-0.5, 0))  # HND, noncausal, per-warp INT8, no LSE
        del self.layouts

    def run(self, q, k, v, output, first, length, global_out):
        qi, ki, oi = self.indices
        parts = triton.cdiv(length, 256)
        _prepare[(self.windows * self.tiles + parts, self.heads)](
            q, k, v, qi, ki, self.meta, self.q, self.qs, self.sums, self.peaks, self.partials,
            self.windows, first, length, self.heads, self.dim, self.tiles, self.global_rows,
            *q.stride()[:2], *k.stride()[:2], *v.stride()[:2], num_warps=4)
        _pack[(self.tiles, self.heads, self.windows)](
            k, v, ki, self.meta, self.k, self.v, self.ks, self.vs, self.sums, self.peaks, self.heads, self.dim,
            self.tiles, *k.stride()[:2], *v.stride()[:2], num_warps=4)
        for arguments in self.calls:
            self.operator(*arguments)
        _scatter[(self.windows * self.out_tiles + (self.global_rows if parts else 0), self.heads)](
            self.out, oi, self.meta, output, self.partials, global_out, self.windows, parts, self.heads, self.dim,
            self.global_rows, self.out_tiles, self.reduce, num_warps=4)


class IndexedWindows:
    def __init__(self, batches, global_rows, max_global_keys):
        self.batches = batches
        self.global_rows, self.max_global_keys = global_rows, max_global_keys

    def __call__(self, q, k, v, output, first=0, length=0, global_out=None):
        """Read the plan's Q/K/V ring views and scatter into contiguous token-major outputs.
        The optional global output holds the final global_rows queries of [first, first + length).
        """
        if global_out is None:
            length = 0
        elif not (self.global_rows <= length <= self.max_global_keys and 0 <= first <= q.shape[0] - length):
            raise ValueError("Global attention range exceeds the ring or the indexed-window workspace")
        for index, batch in enumerate(self.batches):
            batch.run(q, k, v, output, first, length if index == 0 else 0, output if global_out is None else global_out)
        torch.autograd.graph.increment_version(output)
        if global_out is not None:
            torch.autograd.graph.increment_version(global_out)


def make_plans(steps, heads, dim, dtype, device, global_rows, max_global_keys, group_bytes, operator):
    """A plan per step, used serially on one stream with a decode-local workspace. Each group holds query, key and output row-index matrices.
    Rows within a group have the same lengths. Output windows are disjoint, and each query window is no longer than its keys.
    """
    plans, batches = [], []
    for step in steps:
        pending, step_batches = [], []
        size = 0
        for group in step:
            size_bytes = group[1].numel() * heads * dim * dtype.itemsize
            if pending and size + size_bytes > group_bytes:
                step_batches.append(_WindowBatch(pending, heads, dim, device))
                pending, size = [], 0
            pending.append(group)
            size += size_bytes
        if pending:
            step_batches.append(_WindowBatch(pending, heads, dim, device))
        plans.append(IndexedWindows(step_batches, global_rows, max_global_keys))
        batches.extend(step_batches)
    types = (torch.int8, torch.int8, torch.float8_e4m3fn, dtype, torch.float32, torch.float32, torch.float32, torch.float32)
    sizes = map(max, zip(*(batch.sizes for batch in batches)))
    workspace = [torch.empty(n, dtype=t, device=device) for n, t in zip(sizes, types)]
    peaks = torch.empty_like(workspace[-1])
    partials = torch.empty((triton.cdiv(max_global_keys, 256), heads, global_rows, dim + 2), dtype=torch.float32, device=device)
    for batch in batches:
        batch.bind(workspace, peaks, partials, operator, global_rows)
    return plans
