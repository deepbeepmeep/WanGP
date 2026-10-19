"""Bounded math attention for Wan inference on Apple Silicon."""


def bounded_sdpa(sdpa, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, *, scale=None, enable_gqa=False,
                 score_budget_bytes=256 * 1024**2, synchronize=None):
    """Split independent heads/queries, retaining all keys in each softmax.

    The budget covers ONE attention-score tensor, not total process/GPU memory.
    Wan uses matching four-dimensional batch/head layouts and no GQA. Reject
    other layouts instead of silently taking an unbounded path in this trial.
    """
    import torch

    if enable_gqa or any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("Bounded Wan attention requires 4D tensors without GQA")
    if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
        raise ValueError("Bounded Wan attention requires matching batch/head layouts")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query/key embedding dimensions differ")
    if is_causal and attn_mask is not None:
        raise ValueError("Use either a causal flag or an attention mask")

    batch, heads, queries, _ = query.shape
    keys = key.shape[-2]
    # Math SDPA uses float32 intermediates for float16 and bfloat16 inputs.
    score_bytes = max(4, query.element_size())
    if min(batch, heads, queries, keys) < 1 or score_budget_bytes < keys * score_bytes:
        raise ValueError("Attention budget cannot hold one score row")
    head_step = min(heads, max(1, score_budget_bytes // (queries * keys * score_bytes)))
    query_step = min(queries, score_budget_bytes // (head_step * keys * score_bytes))
    mask = None
    if attn_mask is not None:
        # expand is a view, so a broadcast mask is not materialized in full.
        mask = attn_mask.expand(batch, heads, queries, keys)
    output = query.new_empty((batch, heads, queries, value.shape[-1]))

    for b in range(batch):
        for h in range(0, heads, head_step):
            for q in range(0, queries, query_step):
                rows = slice(q, min(q + query_step, queries))
                hs = slice(h, min(h + head_step, heads))
                part_mask = None if mask is None else mask[b:b+1, hs, rows, :]
                if is_causal:
                    # Causality uses the original query offset, including when
                    # the query and key sequence lengths differ.
                    part_mask = (torch.arange(keys, device=query.device)[None, :]
                                 <= torch.arange(q, rows.stop, device=query.device)[:, None])
                part = sdpa(query[b:b+1, hs, rows, :], key[b:b+1, hs, :, :],
                            value[b:b+1, hs, :, :], attn_mask=part_mask,
                            dropout_p=dropout_p, is_causal=False, scale=scale)
                output[b:b+1, hs, rows, :].copy_(part)
                if synchronize is not None:
                    synchronize()
                del part
    return output
