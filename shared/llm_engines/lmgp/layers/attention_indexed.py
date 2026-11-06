# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Indexed sparse attention over the paged KV cache: per query, an indexer scores blocks of cached tokens, and the query attends to the tokens of its best
blocks and to the incomplete block at its end (Qwen QSA, DeepSeek-V4 lightning indexer). Blocks group `ratio` consecutive tokens: their mean raw key,
normalized and rotated at the block's first position, is kept in a paged cache (a row per block). Attention reads the selected rows of any KV layout
(BF16, Q8, mixed bodies with exact overlays). All shapes are fixed for a pass of a given length: its decoding passes replay in CUDA graphs."""
import math
import os

import torch
from torch import nn

from shared.kernels import kernel_policy

from . import kv_formats as kvf
from .attention import Attention, store_kvcache
from ..utils.context import get_context

try:
    from shared.kernels import indexed_attention_triton as fused_kernels
except ImportError:  # no Triton: the PyTorch path
    fused_kernels = None

TAIL_RING = 32  # the raw keys and rotary rows kept by position: the longest verification pass and the 3 tokens before it, at least
ATTENTION_ROWS = 32  # queries of a pass attended together: their selected keys and values are read into VRAM while they run
PROMPT_ROWS = 256  # from this many queries, each query is one program per KV head, in launches of LAUNCH_QUERIES: fewer launches and no partial softmaxes, the same scratch
LAUNCH_QUERIES = 1024
SCORE_ROWS = 512  # queries of a pass scored together
FUSED = fused_kernels is not None and "qsa" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # the switch of the fused kernels, until the switches go


def rotate_(x, cos, sin):
    """x [..., heads, dim] rotated in place over its first cos.shape[-1] columns; cos, sin [..., rotary]."""
    half = cos.shape[-1] // 2
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    first, second = x[..., :half], x[..., half:half * 2]
    saved = first.clone()
    first.mul_(cos[..., :half]).sub_(second * sin[..., :half])
    second.mul_(cos[..., half:]).add_(saved * sin[..., half:])
    return x


def rms_norm(x, weight, eps):
    """float32 x [..., dim] normalized over its last axis and weighted."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()


_DECODE_SPLIT = "decode_split" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # until the switches are removed

class IndexedAttention(Attention):
    """index_heads query heads of index_dim scoring blocks of ratio tokens with one key head, budget tokens per query, and the indexer's projections
    (index_q_proj, index_k_proj) and norm weights held by the owning block."""
    def __init__(self, num_heads, head_dim, scale, num_kv_heads, index_heads, index_dim, budget, ratio, eps):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
        self.index_heads, self.index_dim, self.budget, self.ratio, self.eps = index_heads, index_dim, budget, ratio, eps
        self.top_blocks = budget // ratio
        self.index_pool = None  # [pages, page // ratio, index_dim] BF16, bound by the runner
        self.raw_ring = self.cos_ring = self.sin_ring = None

    # the keys of the incomplete block, and of the blocks a verification pass may complete: by position, so that a rejected draft is simply overwritten
    def prepare_sequence_state(self, max_batch_size, device, dtype):
        if self.raw_ring is None or self.raw_ring.device != device:
            self.raw_ring = torch.zeros(TAIL_RING, self.index_dim, dtype=torch.float32, device=device)
            self.cos_ring = torch.zeros(TAIL_RING, 64, dtype=torch.float32, device=device)
            self.sin_ring = torch.zeros(TAIL_RING, 64, dtype=torch.float32, device=device)
        else:
            self.reset_sequence_state()

    def reset_sequence_state(self):
        if self.raw_ring is not None:
            self.raw_ring.zero_()
            self.cos_ring.zero_()
            self.sin_ring.zero_()

    def release_sequence_state(self):
        self.raw_ring = self.cos_ring = self.sin_ring = self.index_pool = None

    def snapshot_sequence_state(self):
        return {name: getattr(self, name).to("cpu", copy=True) for name in ("raw_ring", "cos_ring", "sin_ring") if getattr(self, name) is not None}

    def restore_sequence_state(self, snapshot):
        for name, copy in (snapshot or {}).items():
            getattr(self, name).copy_(copy)  # into the live rings, which a new engine has reallocated and its graphs read

    def _append_index(self, raw, cos, sin, sequence, table, k_norm):
        """Writes the rows of the pass into the rings, and the pooled key of every block a token of the pass completes into the paged cache (the tokens of a
        rejected draft are overwritten by the next pass before any query sees them)."""
        tokens, ratio, before = raw.shape[0], self.ratio, self.ratio - 1
        first = sequence[0]
        previous = (first - before + torch.arange(before, device=raw.device)) % TAIL_RING
        raw_all = torch.cat((self.raw_ring[previous], raw))  # [before + tokens, dim]
        cos_all, sin_all = torch.cat((self.cos_ring[previous], cos)), torch.cat((self.sin_ring[previous], sin))
        written = tokens if tokens <= TAIL_RING else TAIL_RING
        rows = sequence[-written:] % TAIL_RING
        self.raw_ring[rows] = raw[-written:]
        self.cos_ring[rows] = cos[-written:]
        self.sin_ring[rows] = sin[-written:]
        # the tokens of the pass that may complete a block: one row per block, the last candidate possibly past the pass (written to a page kept for it)
        per_page = self.index_pool.shape[1]
        rows = (before - first % ratio) % ratio + torch.arange((tokens + before) // ratio, device=raw.device) * ratio
        complete = rows < tokens
        rows = rows.clamp_max(tokens - 1)
        members = raw_all[rows[:, None] + torch.arange(ratio, device=raw.device)]  # [blocks, ratio, dim]
        pooled = rotate_(rms_norm(members.mean(1), k_norm, self.eps)[:, None], cos_all[rows], sin_all[rows])[:, 0].to(self.index_pool.dtype)
        blocks = sequence[rows] // ratio
        pages = table[(blocks // per_page).clamp_max(table.shape[0] - 1)].long()
        complete = complete & (pages >= 0)
        pages = torch.where(complete, pages, self.index_pool.shape[0] - 1)
        self.index_pool[pages, blocks % per_page] = pooled

    def _selected_tokens(self, query, sequence, table):
        """The tokens each query attends to: positions [queries, budget + ratio - 1], and whether each is valid."""
        queries, ratio = query.shape[0], self.ratio
        per_page = self.index_pool.shape[1]
        pages = table.clamp_min(0).long()
        keys = self.index_pool[pages].reshape(-1, self.index_dim).float()  # [blocks, dim]: the unwritten rows are masked below
        count = keys.shape[0]
        visible = (sequence + 1) // ratio
        order = torch.arange(count, device=keys.device)
        scale = math.sqrt(self.index_dim)
        tops, values = [], []
        k = min(self.top_blocks, count)
        for start in range(0, queries, SCORE_ROWS):
            q = query[start:start + SCORE_ROWS]
            scores = torch.matmul(q, keys.T).relu_().sum(1).div_(scale)  # [queries, blocks]
            scores = scores.masked_fill(order[None, :] >= visible[start:start + SCORE_ROWS, None], -math.inf)
            value, top = scores.topk(k, dim=-1, sorted=False)
            scores = None
            tops.append(top)
            values.append(value)
        top = torch.cat(tops) if len(tops) > 1 else tops[0]
        value = torch.cat(values) if len(values) > 1 else values[0]
        member = torch.arange(ratio, device=keys.device)
        positions = (top[:, :, None] * ratio + member).reshape(queries, -1)
        valid = (value > -math.inf)[:, :, None].expand(-1, -1, ratio).reshape(queries, -1)
        # the tokens after the last complete block of each query: ratio - 1 at most
        tail = visible[:, None] * ratio + torch.arange(ratio - 1, device=keys.device)
        positions = torch.cat((positions, tail), 1)
        valid = torch.cat((valid, tail <= sequence[:, None]), 1)
        return positions.clamp_min(0), valid

    def _read_rows(self, cache, scale, minimum, fmt, pages, slots):
        """[..., kv_heads, head_dim] float32 rows of one component of the body."""
        data = cache[pages, slots]
        if fmt is None:
            return data.float()
        scales = scale[pages, slots]
        if fmt.q8:
            shape = data.shape
            return (data.view(*shape[:-1], scales.shape[-1], kvf.GROUP).float() * scales.float().unsqueeze(-1)).view(shape)
        return kvf.dequantize_heads(data, scales, None if minimum is None else minimum[pages, slots], fmt, torch.float32)

    def _read_kv(self, positions, table, row):
        layout = self.kv_layout
        quantized = self.k_cache.dtype in (torch.int8, torch.uint8)
        page = self.k_cache.shape[1]
        pages, slots = table[(positions // page).clamp_max(table.shape[0] - 1)].clamp_min(0).long(), positions % page  # the invalid positions past the last page are masked
        keys = self._read_rows(self.k_cache, self.k_scale, self.k_min, layout.k if quantized else None, pages, slots)
        values = self._read_rows(self.v_cache, self.v_scale, self.v_min, layout.v if quantized else None, pages, slots)
        if quantized and layout.overlay and self.k_overlay is not None and get_context().overlay_base is not None:
            context = get_context()
            base, ring_start = context.overlay_base[row], context.ring_start[row]
            sink, recent = layout.sink_tokens(page), layout.recent_tokens
            exact = (positions < sink) | (positions >= ring_start)
            exact = exact & (base >= 0)
            rows = base.clamp_min(0) + torch.where(positions < sink, positions, sink + (positions & (max(recent, 1) - 1)))
            exact_keys = self._read_overlay(self.k_overlay, self.k_overlay_scale, rows)
            exact_values = self._read_overlay(self.v_overlay, self.v_overlay_scale, rows)
            mask = exact[..., None, None]
            keys, values = torch.where(mask, exact_keys, keys), torch.where(mask, exact_values, values)
        return keys, values

    @staticmethod
    def _read_overlay(overlay, scale, rows):
        data, scales = overlay[rows], scale[rows]
        shape = data.shape
        return (data.view(*shape[:-1], scales.shape[-1], kvf.GROUP).float() * scales.float().unsqueeze(-1)).view(shape)

    def _attend(self, q, positions, valid, table, row):
        """float32 q [queries, heads, dim]; [queries, heads * dim] attention over the rows of positions [queries, tokens]."""
        queries, groups = q.shape[0], self.num_heads // self.num_kv_heads
        keys, values = self._read_kv(positions, table, row)  # [queries, tokens, kv_heads, dim]
        q = q.view(queries, self.num_kv_heads, groups, self.head_dim)
        scores = torch.matmul(q, keys.permute(0, 2, 3, 1)).mul_(self.scale)  # [queries, kv_heads, groups, tokens]
        keys = None
        scores = scores.masked_fill(~valid[:, None, None, :], -math.inf).softmax(-1)
        output = torch.matmul(scores, values.permute(0, 2, 1, 3))  # [queries, kv_heads, groups, dim]
        return output.reshape(queries, -1)

    def forward_indexed(self, q, k, v, index_q, index_raw, cos, sin, sequence, k_norm):
        """float32 q [queries, heads, dim], k, v [queries, kv_heads, dim], index_q [queries, index_heads, index_dim] (normalized, rotated), index_raw [queries,
        index_dim], cos, sin [queries, rotary] and sequence, the positions in the cache [queries]. Writes the cache, returns [queries, heads * dim] in float32."""
        context = get_context()
        if self.k_cache.dtype not in (torch.int8, torch.uint8):
            k, v = k.to(self.k_cache.dtype), v.to(self.k_cache.dtype)
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping, self.k_scale, self.v_scale, use_triton_kv_cache=self.use_triton_kv_cache,
                      k_min=self.k_min, v_min=self.v_min, k_overlay=self.k_overlay, v_overlay=self.v_overlay, k_overlay_scale=self.k_overlay_scale,
                      v_overlay_scale=self.v_overlay_scale, overlay_mapping=context.overlay_mapping, layout=self.kv_layout)
        table = context.block_tables[0]
        self._append_index(index_raw, cos, sin, sequence, table, k_norm)
        positions, valid = self._selected_tokens(index_q, sequence, table)
        queries = q.shape[0]
        if queries <= ATTENTION_ROWS:
            return self._attend(q, positions, valid, table, 0)
        outputs = q.new_empty(queries, self.num_heads * self.head_dim)
        for start in range(0, queries, ATTENTION_ROWS):
            stop = start + ATTENTION_ROWS
            outputs[start:stop] = self._attend(q[start:stop], positions[start:stop], valid[start:stop], table, 0)
        return outputs

    def fused(self, x, *weights):
        """The fused kernels serve a pass reading x and these weights, all in VRAM (MMGP's CPU compute may keep a block in RAM)."""
        return FUSED and fused_kernels.available(x, self) and all(weight.is_cuda for weight in weights)

    def forward_fused(self, q, q_full, k, v, index_q, index_raw, cos, sin, sequence, k_norm):
        """forward_indexed in the fused kernels: q, index_q float32 (normalized, rotated), k float32, v and index_raw in the dtype of the projections, the gate read from q_full [queries, heads, dim * 2].
        Returns [queries, heads * dim] in BF16."""
        context = get_context()
        if self.k_cache.dtype not in (torch.int8, torch.uint8):
            k, v = k.to(self.k_cache.dtype), v.to(self.k_cache.dtype)
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping, self.k_scale, self.v_scale, use_triton_kv_cache=self.use_triton_kv_cache,
                      k_min=self.k_min, v_min=self.v_min, k_overlay=self.k_overlay, v_overlay=self.v_overlay, k_overlay_scale=self.k_overlay_scale,
                      v_overlay_scale=self.v_overlay_scale, overlay_mapping=context.overlay_mapping, layout=self.kv_layout)
        table = context.block_tables[0]
        fused_kernels.append_index(index_raw, cos, sin, sequence, table, self.raw_ring, self.cos_ring, self.sin_ring, self.index_pool, k_norm, self.ratio, self.eps)
        overlay = self.kv_layout is not None and self.kv_layout.overlay and self.k_overlay is not None and context.overlay_base is not None
        base, ring = (context.overlay_base[:1], context.ring_start[:1]) if overlay else (None, None)
        sink = self.kv_layout.sink_tokens(self.k_cache.shape[1]) if overlay else 0
        queries = q.shape[0]
        if queries <= ATTENTION_ROWS:
            ids, counts = fused_kernels.select_blocks(index_q, self.index_pool, table, sequence, self.top_blocks, self.ratio, ring, sink)
            return fused_kernels.attend_selected(q, q_full, ids, counts, sequence, table, self, base, ring, precision=fused_kernels.SPLIT_BF16 if _DECODE_SPLIT and kernel_policy.allow_approximate() else fused_kernels.PRECISION)
        outputs = q.new_empty(queries, self.num_heads * self.head_dim, dtype=torch.bfloat16)
        rows = LAUNCH_QUERIES if queries >= PROMPT_ROWS else ATTENTION_ROWS
        precision = fused_kernels.SPLIT_BF16 if queries >= PROMPT_ROWS and kernel_policy.allow_approximate() else fused_kernels.PRECISION  # what it adds is far below the BF16 rounding of the output
        for start in range(0, queries, rows):
            stop = start + rows
            ids, counts = fused_kernels.select_blocks(index_q[start:stop], self.index_pool, table, sequence[start:stop], self.top_blocks, self.ratio, ring, sink)
            fused_kernels.attend_selected(q[start:stop], q_full[start:stop], ids, counts, sequence[start:stop], table, self, base, ring, precision=precision, chunks=1 if queries - start >= PROMPT_ROWS else None, output=outputs[start:stop])
        return outputs
