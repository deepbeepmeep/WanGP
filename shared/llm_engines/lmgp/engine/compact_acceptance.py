# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The GPU acceptance of a drafted block on the target's top-k support only (dflash_sampling computes it on whole vocabulary rows).

With top-k, the target distribution of a row lives on its k best scores (and those tied with the k-th, up to the capacity). One pass over each
row applies the sampler's rules (dflash_sampling.apply_rules: the same operations, the same rounding to the logits' type) and keeps every
block's k best scores, which contain the row's k best. A row where a block's last kept score reaches the k-th may hide ties beyond the block:
it is flagged unsafe and the established exact sampler takes it, as for the other ambiguous rows. What follows (nucleus, min-p, softmax,
block verification) is the dense code's arithmetic on these candidates, and the replacement is drawn with the same per-token noise, so the
outputs are the dense path's (bar the summation order of a residual's mass, which the dense code adds over the whole vocabulary).
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 1024
MAX_DRAFTS = 16  # the drafts a row may follow, a bound of the kernel's unrolled loops (not specialized per window: one compile per sampler law)
_INF = tl.constexpr(float("inf"))


@triton.jit(do_not_specialize=("vocab", "drafts_count", "penalty", "inv_penalty", "inv_temperature"))
def _rules_top(LOGITS, DRAFTS, HISTORY, EXEMPT, BIAS, SUPPRESSED, THINKING_STOPS, THINKING, VALUES, IDS, vocab, drafts_count, penalty, inv_penalty, inv_temperature,
               PENALIZE: tl.constexpr, HAS_EXEMPT: tl.constexpr, BLOCK_SIZE: tl.constexpr, KEEP: tl.constexpr, MAX_DRAFTS: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE)
    tokens = block * BLOCK_SIZE + offsets
    inside = tokens < vocab
    scores = tl.load(LOGITS + row * vocab + tokens, mask=inside, other=0.0)
    if PENALIZE:  # tokens of the history or drafted before this row, outside the exempt ones; the products rounded to the logits' type
        seen = tl.load(HISTORY + tokens, mask=inside, other=0) != 0
        for index in tl.static_range(MAX_DRAFTS):
            if index < drafts_count:
                seen = seen | ((tokens == tl.load(DRAFTS + index)) & (index < row))
        if HAS_EXEMPT:
            seen = seen & (tl.load(EXEMPT + tokens, mask=inside, other=0) == 0)
        value = scores.to(tl.float32)
        penalized = tl.where(value < 0, value * penalty, value * inv_penalty).to(scores.dtype)  # PyTorch divides by a scalar as a product by its reciprocal
        scores = tl.where(seen, penalized, scores)
    scores = (scores.to(tl.float32) + tl.load(BIAS + tokens, mask=inside, other=0.0).to(tl.float32)).to(scores.dtype)
    value = tl.where(tl.load(SUPPRESSED + tokens, mask=inside, other=0) != 0, -_INF, scores.to(tl.float32))
    close, remaining, active = tl.load(THINKING), tl.load(THINKING + 1), tl.load(THINKING + 2)
    closed = row < 0
    for index in tl.static_range(MAX_DRAFTS):
        if index < drafts_count:
            closed = closed | ((tl.load(DRAFTS + index) == close) & (index < row))
    in_thinking = (active != 0) & (closed == 0)
    value = tl.where(in_thinking & (tl.load(THINKING_STOPS + tokens, mask=inside, other=0) != 0), -_INF, value)
    value = tl.where(in_thinking & (row >= remaining), tl.where(tokens == close, 0.0, -_INF), value)
    value = tl.where(inside, value * inv_temperature, -_INF)
    base = (row * tl.num_programs(1) + block) * KEEP
    for kept in tl.static_range(KEEP):
        position = tl.argmax(value, 0)
        tl.store(VALUES + base + kept, tl.max(value, 0))
        tl.store(IDS + base + kept, block * BLOCK_SIZE + position)
        value = tl.where(offsets == position, -_INF, value)


def _reciprocal(value):
    """The factor PyTorch's CUDA division by a Python scalar multiplies with: its reciprocal in double, rounded to FP32 (checked against
    x / t for t in 0.55 ... 1.3)."""
    return float(np.float32(1.0 / float(value)))


def supported(seq, logits) -> bool:
    return seq.top_k is not None and 2 <= seq.top_k <= 128 and logits.is_cuda and logits.dtype in (torch.bfloat16, torch.float16)


def target_probabilities(logits, drafts, history, bias, suppressed, thinking_stops, thinking, penalty, exempt, top_k, top_p, min_p, temperature):
    """dflash_sampling.apply_rules then target_probabilities, top-k branch, on the candidates: (ids [R, C], probabilities [R, C], unsafe [R]),
    C the capacity; tokens outside a row's candidates have probability 0."""
    rows, vocab = logits.shape
    blocks = triton.cdiv(vocab, BLOCK)
    kept = torch.empty((rows, blocks * top_k), dtype=torch.float32, device=logits.device)
    kept_ids = torch.empty((rows, blocks * top_k), dtype=torch.int32, device=logits.device)
    _rules_top[(rows, blocks)](logits, drafts, history, history if exempt is None else exempt, bias, suppressed, thinking_stops, thinking, kept, kept_ids, vocab, drafts.numel(),
                               float(penalty), _reciprocal(penalty), _reciprocal(temperature), penalty != 1, exempt is not None, BLOCK, top_k, MAX_DRAFTS, num_warps=4)
    capacity = min(vocab, max(64, 2 * top_k))
    values, order = kept.topk(capacity + 1 if capacity < vocab else capacity, dim=-1)
    ids = kept_ids.gather(1, order).long()
    outside = values[:, capacity] if capacity < vocab else None
    values, ids = values[:, :capacity], ids[:, :capacity]
    threshold = values[:, top_k - 1:top_k]
    unsafe = outside >= threshold[:, 0] if outside is not None else torch.zeros(rows, dtype=torch.bool, device=logits.device)
    unsafe = unsafe | (kept.view(rows, blocks, top_k)[:, :, -1] >= threshold).any(dim=1)  # a block holding more scores tied with the k-th than it kept
    order = ids.argsort(dim=-1)
    ids = ids.gather(1, order)
    values = values.gather(1, order)
    values = values.masked_fill(values < threshold, -torch.inf)
    normalizer = torch.logsumexp(values, dim=-1, keepdim=True)
    if min_p is not None and min_p > 0:
        values = values.masked_fill(values < values.amax(dim=-1, keepdim=True) + math.log(min_p), -torch.inf)
    if top_p is not None and 0 < top_p < 1:
        excluded = 1 - torch.exp(values - normalizer).sum(dim=-1, keepdim=True)
        values, order = values.sort(dim=-1)
        ids = ids.gather(1, order)
        keep = excluded + torch.exp(values - normalizer).cumsum(dim=-1) > 1 - top_p
        keep[:, -1] = True
        lowest_kept = values.masked_fill(~keep, torch.inf).amin(dim=-1)
        highest_removed = values.masked_fill(keep, -torch.inf).amax(dim=-1)
        unsafe = unsafe | (torch.isfinite(lowest_kept) & (lowest_kept == highest_removed))
        values = values.masked_fill(~keep, -torch.inf)
    return ids, values.softmax(dim=-1), unsafe


def accept_block(ids, probabilities, unsafe, drafts, draft_probs, valid_length, uniforms, noise, stops):
    """dflash_sampling.accept_block (sampling) with the target rows on their candidates (ids, probabilities [n + 1, C]); draft_probs stays
    a dense [n, V] row per draft."""
    n = drafts.numel()
    positions = torch.arange(n, device=drafts.device)
    at_drafts = torch.where(ids[:n] == drafts[:, None], probabilities[:n], 0.).sum(dim=1)  # a row holds a token once: its probability or 0
    ratios = at_drafts / draft_probs.gather(1, drafts[:, None])[:, 0]
    weights = [torch.ones((), dtype=ratios.dtype, device=drafts.device)]
    for i in range(n):
        weights.append((weights[-1] * ratios[i]).clamp(max=1.0))
    weights = torch.stack(weights)
    residuals = (weights[:n, None] * probabilities[:n] - draft_probs.gather(1, ids[:n])).clamp_min(0)  # on the target's support: elsewhere the dense residual is 0
    masses = residuals.sum(dim=1)
    inner = masses[1:] / (masses[1:] + 1 - weights[1:n]).clamp_min(1e-30)
    h = torch.cat((inner, weights[n:]))
    h = torch.where(positions + 1 == valid_length, weights[1:], h)
    ok = (uniforms < h) & (positions < valid_length)
    first = torch.where(ok, positions + 1, 0).amax()
    stop_at = torch.where(stops[drafts] & (positions < first), positions, n).amin()
    accepted_count = torch.minimum(first, stop_at + 1)
    row = first.clamp(max=n - 1).reshape(1)
    residual = residuals.index_select(0, row)[0]
    target = probabilities.index_select(0, first.reshape(1))[0]
    use_residual = first < valid_length
    support = torch.where(use_residual, ids.index_select(0, row)[0], ids.index_select(0, first.reshape(1))[0])
    scores = torch.where(use_residual, residual, target) / noise[support]
    # the dense argmax: the lowest token id among the best scores, token 0 when every score is 0 (an empty residual)
    best = scores.amax()
    replacement = torch.where(best > 0, torch.where(scores == best, support, torch.iinfo(torch.int64).max).amin(), torch.zeros((), dtype=torch.int64, device=drafts.device))
    tokens = torch.cat((drafts, replacement.reshape(1)))
    tokens.scatter_(0, first.reshape(1), replacement.reshape(1))
    count = torch.minimum(first + 1, stop_at + 1)
    visited = accepted_count + ((first < valid_length) & (stop_at >= first)).long()
    possible_rows = torch.arange(n + 1, device=drafts.device) <= valid_length
    fallback = (unsafe & possible_rows).any().long()
    return torch.cat((torch.stack((count, accepted_count, visited, fallback, valid_length)), tokens))


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
