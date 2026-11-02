"""Shared fixed-shape sampling for native MTP and block drafters."""
import time
import torch


def bounded_support(seq):
    """Whether fixed-capacity filtering can reproduce the sampler's support."""
    if seq.top_k is None:
        return seq.top_p is not None and 0 < seq.top_p < 1 or seq.min_p is not None and seq.min_p > 0
    return 1 <= seq.top_k <= 128


def can_batch_acceptance(runner, seq):
    processor = seq.logits_processor
    return (runner.use_triton_sampling and not getattr(runner, "enforce_eager", False)
            and bounded_support(seq)
            and (processor is None and seq.logits_processor_update_state is None
                 or callable(getattr(processor, "_speculative_batch_rules", None))))


def draft_probabilities(runner, seq, logits):
    """Filter native MTP proposals without dynamic candidate extraction."""
    from .dflash_sampling import target_probabilities
    key = ("mtp_draft", logits.device, logits.dtype, logits.numel(),
           seq.top_k, seq.top_p, seq.min_p, seq.temperature)
    graphs = runner._speculative_sampling_graphs
    state = graphs.pop(key, None)
    if state is None:
        if len(graphs) >= 8:
            graphs.pop(next(iter(graphs)))
        values = logits[None].clone()
        top_k = None if seq.top_k is None else min(seq.top_k, logits.numel())
        target_probabilities(values, top_k, seq.top_p, seq.min_p, seq.temperature)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            probabilities, _ = target_probabilities(values, top_k, seq.top_p, seq.min_p, seq.temperature)
        state = graph, values, probabilities
    graphs[key] = state
    graph, values, probabilities = state
    values.copy_(logits[None])
    graph.replay()
    # Proposal support/tie order may differ from the target: rejection uses
    # this exact normalized q. Each draft retains its own copy until verified.
    return probabilities[0].clone()


def sample_verified_block(runner, seq, logits, draft_tokens, draft_distributions, sample_params, profile=None, *, valid_length=None, method="mtp"):
    """Accept on GPU; retain the reference sampler for ambiguous target support."""
    self = runner
    from .dflash_sampling import apply_rules, target_probabilities, accept_block
    started = time.perf_counter()
    if profile is not None:
        profile["slot"]["events"][5].record(torch.cuda.current_stream())
    n, vocab = draft_tokens.numel(), logits.shape[-1]
    processor = seq.logits_processor
    rules = processor._speculative_batch_rules() if processor is not None else {
        "suppressed": (), "thinking_stops": (), "thinking": (-1, 0, 0)}
    stop_ids = set(getattr(seq, "speculative_stop_token_ids", ()))
    if not seq.ignore_eos:
        stop_ids.add(self.config.eos)
    penalty = float(seq.repetition_penalty or 1.)
    if penalty != 1 and self._sync_repetition_token_cache(seq)["span"] != (False, False, False):
        penalty = 1.  # inside code, quotes or backticks: copied text, not repetition (drafted tokens of the block are approximated as outside)
    exempt = seq.repetition_exempt_ids if penalty != 1 else None
    key = (n, vocab, logits.dtype, seq.top_k, seq.top_p, seq.min_p, seq.temperature, penalty, exempt,
           tuple(rules["suppressed"]), tuple(rules["thinking_stops"]), tuple(sorted(stop_ids)))
    graphs = getattr(self, "_speculative_acceptance_graphs", None)
    if graphs is None:
        graphs = self._speculative_acceptance_graphs = {}
    state = graphs.pop(key, None)
    if state is None:
        # Confidence exits and KV page boundaries can exercise every supported
        # draft length in one request. Keep those eight shapes resident, or
        # short blocks repeatedly evict and recapture the common full block.
        if len(graphs) >= 8:
            graphs.pop(next(iter(graphs)))
        def mask(ids):
            value = torch.zeros(vocab, dtype=torch.bool, device=logits.device)
            valid = [int(i) for i in ids if 0 <= int(i) < vocab]
            if valid:
                value.index_fill_(0, torch.tensor(valid, device=logits.device), True)
            return value
        state = dict(logits=logits.clone(), drafts=draft_tokens.clone(),
                     q=torch.zeros((n, vocab), dtype=torch.float32, device=logits.device),
                     length=torch.full((), n, dtype=torch.int64, device=logits.device),
                     history=mask(()), bias=torch.zeros_like(logits[0]),
                     suppressed=mask(rules["suppressed"]), thinking_stops=mask(rules["thinking_stops"]),
                     thinking=torch.tensor(rules["thinking"], dtype=torch.int64, device=logits.device),
                     stops=mask(stop_ids), uniforms=torch.ones(n, device=logits.device),
                     noise=torch.ones(vocab, device=logits.device), exempt=None if exempt is None else mask(exempt))
        def compute():
            scores = apply_rules(state["logits"], state["drafts"], state["history"], state["bias"],
                                 state["suppressed"], state["thinking_stops"], state["thinking"], penalty, state["exempt"])
            if seq.top_k == 1:
                probabilities = scores
                unsafe = torch.zeros(n + 1, dtype=torch.bool, device=logits.device)
            else:
                probabilities, unsafe = target_probabilities(scores, None if seq.top_k is None else min(seq.top_k, vocab), seq.top_p, seq.min_p, seq.temperature)
            return accept_block(probabilities, unsafe, state["drafts"], state["q"], state["length"],
                                state["uniforms"], state["noise"], state["stops"], greedy=seq.top_k == 1)
        compute()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            packed = compute()
        state.update(graph=graph, packed=packed)
    graphs[key] = state
    state["logits"].copy_(logits)
    state["drafts"].copy_(draft_tokens)
    state["length"].fill_(n) if valid_length is None else state["length"].copy_(valid_length)
    if draft_distributions is not None:
        if all(q.numel() == vocab for q in draft_distributions):
            torch.stack(draft_distributions, out=state["q"])
        else:
            # Native MTP may predict only the first vocabulary rows. Clear the
            # reused suffix, including when a thinking rule expanded a prior q.
            state["q"].zero_()
            for i, q in enumerate(draft_distributions):
                state["q"][i, :q.numel()].copy_(q)
    state["history"].zero_()
    if penalty != 1:
        history = self._sync_repetition_token_cache(seq)["token_ids"]
        if history:
            ids = torch.tensor(history, dtype=torch.int64, device="cpu").to(logits.device, non_blocking=True)
            state["history"].index_fill_(0, ids, True)
    bias = self._get_logits_bias(seq, logits)
    state["bias"].zero_() if bias is None else state["bias"].copy_(bias.reshape(-1))
    for i, value in enumerate(rules["thinking"]):
        state["thinking"][i].fill_(value)
    if seq.top_k != 1:
        state["uniforms"].uniform_(generator=self._sampling_generator)
        state["noise"].exponential_(1, generator=self._sampling_generator).clamp_min_(1e-10)
    state["graph"].replay()
    # The sole D2H boundary: scheduling/streaming and GDN commit need the
    # completed block, never an intermediate draft or acceptance decision.
    count, accepted, visited, fallback, valid_length, *tokens = state["packed"].tolist()
    if fallback:
        counter = f"_{method}_acceptance_fallbacks"
        setattr(self, counter, getattr(self, counter, 0) + 1)
        return self._sample_verified_block_reference(seq, logits[:valid_length + 1], draft_tokens[:valid_length],
            None if draft_distributions is None else draft_distributions[:valid_length], sample_params, profile)
    counter = f"_{method}_gpu_acceptance_rounds"
    rounds = getattr(self, counter, 0) + 1
    setattr(self, counter, rounds)
    if rounds == 1:
        label = {"mtp": "MTP", "dflash": "DFlash2", "dspark": "DSpark"}[method]
        print(f"[Deepy][{label}] GPU block acceptance active (one completed-block readback).")
    emitted = tokens[:count]
    for i in range(visited):
        self.speculative_stats["drafted_by_position"][i] += 1
    for i in range(accepted):
        self.speculative_stats["accepted_by_position"][i] += 1
    self.speculative_stats["drafted"] += visited
    self.speculative_stats["accepted"] += accepted
    if seq.logits_processor_update_state is not None:
        for token in emitted:
            seq.logits_processor_update_state(token)
    self._mark_mtp_stage_profile(profile, 6, "sampling", started)
    return emitted, accepted

