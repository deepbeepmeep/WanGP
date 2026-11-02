"""Fixed-shape verification operations shared by DFlash2 and native MTP.

Only the completed block crosses to the host. Ambiguous nucleus ties and an
overflowing top-k support are flagged for the established exact sampler.
"""
import math
import torch


# Without top-k, the nucleus is resolved inside this many highest logits. A
# token's nucleus decision depends only on the mass ranked above it, so the
# support is exact whenever the first excluded token is itself rejected.
NUCLEUS_CAPACITY = 1024


def target_probabilities(logits, top_k, top_p, min_p, temperature):
    rows, vocab = logits.shape
    logits = logits.float() / temperature
    capacity = min(vocab, NUCLEUS_CAPACITY if top_k is None else max(64, 2 * top_k))
    values, ids = logits.topk(min(vocab, capacity + 1), dim=-1)
    nucleus = top_p is not None and 0 < top_p < 1
    outside = values[:, capacity] if capacity < vocab else None
    values, ids = values[:, :capacity], ids[:, :capacity]
    if top_k is None:
        # The reference universe is the whole vocabulary.
        normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
        if outside is not None and min_p is not None and min_p > 0:
            outside = outside.masked_fill(outside < values[:, 0] + math.log(min_p), -torch.inf)
        unsafe = torch.zeros(rows, dtype=torch.bool, device=logits.device) if outside is None or nucleus else outside > -torch.inf
    else:
        threshold = values[:, top_k - 1:top_k]
        unsafe = outside >= threshold[:, 0] if outside is not None else torch.zeros(rows, dtype=torch.bool, device=logits.device)
        outside = None
    # Match the ascending vocabulary order of the reference's nonzero().
    order = ids.argsort(dim=-1)
    ids = ids.gather(1, order)
    values = values.gather(1, order)
    if top_k is not None:
        values = values.masked_fill(values < threshold, -torch.inf)
        normalizer = torch.logsumexp(values, dim=-1, keepdim=True)
    if min_p is not None and min_p > 0:
        values = values.masked_fill(values < values.amax(dim=-1, keepdim=True) + math.log(min_p), -torch.inf)
    if nucleus:
        excluded = 1 - torch.exp(values - normalizer).sum(dim=-1, keepdim=True)
        values, order = values.sort(dim=-1)
        ids = ids.gather(1, order)
        keep = excluded + torch.exp(values - normalizer).cumsum(dim=-1) > 1 - top_p
        keep[:, -1] = True
        # Padding changes torch.sort's unstable order for equal values. If a
        # tied score straddles the nucleus boundary, use the exact old path.
        lowest_kept = values.masked_fill(~keep, torch.inf).amin(dim=-1)
        highest_removed = values.masked_fill(keep, -torch.inf).amax(dim=-1)
        unsafe = unsafe | (torch.isfinite(lowest_kept) & (lowest_kept == highest_removed))
        if outside is not None:
            # The first token beyond capacity joins the nucleus unless the
            # mass ranked above it already reaches top-p; keep a rounding margin.
            unsafe = unsafe | ((outside > -torch.inf) & ((1 - excluded[:, 0] < top_p + 1e-4) | (outside == lowest_kept)))
        values = values.masked_fill(~keep, -torch.inf)
    probabilities = torch.zeros_like(logits).scatter(1, ids, values.softmax(dim=-1))
    return probabilities, unsafe


def apply_rules(logits, drafts, history, bias, suppressed, thinking_stops, thinking, penalty, exempt=None):
    rows, vocab = logits.shape
    scores = logits.clone()
    if penalty != 1:
        seen = history.unsqueeze(0).expand(rows, -1).clone()
        for i in range(drafts.numel()):
            seen[i + 1:].scatter_(1, drafts[i:i + 1].expand(rows - i - 1, 1), True)
        if exempt is not None:
            seen &= ~exempt  # drafted tokens the sequence never penalizes (Sequence.repetition_exempt_ids)
        scores = torch.where(seen, torch.where(scores < 0, scores * penalty, scores / penalty), scores)
    scores.add_(bias)
    scores.masked_fill_(suppressed.unsqueeze(0), -torch.inf)
    row = torch.arange(rows, device=logits.device)
    close, remaining, active = thinking.unbind()
    closed = torch.cat((torch.zeros(1, dtype=torch.int64, device=logits.device), (drafts == close).long().cumsum(0))) > 0
    in_thinking = (active != 0) & ~closed
    force = in_thinking & (row >= remaining)
    scores.masked_fill_(in_thinking[:, None] & thinking_stops[None, :], -torch.inf)
    scores = torch.where(force[:, None], torch.where(torch.arange(vocab, device=logits.device)[None, :] == close, 0., -torch.inf).to(scores.dtype), scores)
    return scores


def accept_block(probabilities, unsafe, drafts, draft_probs, valid_length, uniforms, noise, stops, greedy=False):
    """Accepted prefix and replacement token of a drafted block from the target rows (probabilities[i]: after i accepted drafts) and
    the draft rows (draft_probs[i]). Sampling uses block verification (Sun et al., "Block Verification Accelerates Speculative
    Decoding", 2024): the weights p_0 = 1, p_i = min(p_{i-1} P_i(X_i) / Q_i(X_i), 1) are formed along the whole block, the block ends after
    the largest i whose draw eta_i falls under h_i = S_i / (S_i + 1 - p_i), S_i the mass of max(p_i P - Q, 0) after i drafts (h = p at the
    last draft), and the replacement is drawn from that residual (or from the bonus row after a complete block). The output keeps the
    target distribution exactly and accepts at least as many drafts in expectation as the token-by-token rule, which it equals for a
    single draft. Greedy verification accepts the drafts that are the target's argmax."""
    n = drafts.numel()
    positions = torch.arange(n, device=drafts.device)
    if greedy:
        targets = probabilities.argmax(dim=-1)
        accepted = (targets[:-1] == drafts) & (positions < valid_length)
        first = torch.where(accepted, n, positions).amin()
    else:
        ratios = probabilities[:-1].gather(1, drafts[:, None])[:, 0] / draft_probs.gather(1, drafts[:, None])[:, 0]
        weights = [torch.ones((), dtype=ratios.dtype, device=drafts.device)]
        for i in range(n):   # p_0 .. p_n (a fixed-length block: the loop unrolls under graph capture)
            weights.append((weights[-1] * ratios[i]).clamp(max=1.0))
        weights = torch.stack(weights)
        residuals = (weights[:n, None] * probabilities[:n] - draft_probs).clamp_min(0)   # row i: the residual after i accepted drafts
        masses = residuals.sum(dim=1)
        inner = masses[1:] / (masses[1:] + 1 - weights[1:n]).clamp_min(1e-30)   # h_1 .. h_{n-1} (0 when the residual is empty)
        h = torch.cat((inner, weights[n:]))                                       # h_n = p_n
        h = torch.where(positions + 1 == valid_length, weights[1:], h)            # a block cut at valid_length ends there
        ok = (uniforms < h) & (positions < valid_length)
        first = torch.where(ok, positions + 1, 0).amax()                           # accepted drafts
    stop_at = torch.where(stops[drafts] & (positions < first), positions, n).amin()
    accepted_count = torch.minimum(first, stop_at + 1)
    if greedy:
        replacement = targets.gather(0, first.reshape(1))[0]
    else:
        residual = residuals.index_select(0, first.clamp(max=n - 1).reshape(1))[0]
        distribution = torch.where(first < valid_length, residual, probabilities.index_select(0, first.reshape(1))[0])
        replacement = (distribution / noise).argmax()
    tokens = torch.cat((drafts, replacement.reshape(1)))
    tokens.scatter_(0, first.reshape(1), replacement.reshape(1))
    count = torch.minimum(first + 1, stop_at + 1)
    visited = accepted_count + ((first < valid_length) & (stop_at >= first)).long()
    # The fallback decision must not depend on acceptance RNG. Restarting the
    # reference sampler only when an unsafe row was reached would bias earlier
    # acceptance decisions. Check the entire potentially reachable block.
    possible_rows = torch.arange(n + 1, device=drafts.device) <= valid_length
    fallback = (unsafe & possible_rows).any().long()
    return torch.cat((torch.stack((count, accepted_count, visited, fallback, valid_length)), tokens))
