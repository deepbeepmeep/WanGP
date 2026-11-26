# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Placement of the routed experts of an MoE model of the LM engine, shared by its models (Flash-Next, Inkling 2).

- First placement: the expert_priority of each bank (MMGP keeps the highest in VRAM first) from a usage corpus, a JSON file
  {"counts": [[evaluations of each expert] per layer]} measured on representative text with the model's eager core. The priority is the
  evaluations per token per byte (an expert in VRAM saves about the same CPU time whatever its format), comparable across the layers.
  Decoding then follows the routing of the text where the model sets _lm_engine_adaptive_experts (mmgp elastic residency).
- Prompt chunks: the longest prompt chunk whose pass fits in the VRAM left, while the experts out of VRAM stream through a ring
  (mmgp.expert_stream): longer chunks read each streamed expert once for more tokens."""
import json

import torch


def load_usage(path):
    """The usage counts of a model's experts, [layers, experts] float64."""
    with open(path, encoding="utf-8") as reader:
        return torch.tensor(json.load(reader)["counts"], dtype=torch.float64)


def set_expert_priorities(banks, usage, tokens=None):
    """The expert_priority of each bank from its layer's counts: per token of the corpus when tokens (the corpus' tokens) is given, else
    per share of its layer's evaluations (a model whose every layer runs once per token)."""
    for bank, counts in zip(banks, usage):
        bank.expert_priority = ((counts + 1) / ((counts + 1).sum() if tokens is None else tokens) * (2 << 20) / bank.expert_nbytes).tolist()


def prompt_chunk(model, tokens, chunks, bytes_per_token, margin):
    """The tokens of a prompt read together: the whole prompt if its pass fits in the free VRAM (with half the VRAM the experts may lend),
    or else the longest of chunks (longest first) that does; bytes_per_token: VRAM of a prompt pass per token beside the ring of its
    streamed experts, margin: left free beside it."""
    from mmgp import expert_stream, runtime
    room = runtime.free_vram() + runtime.lendable_experts(model) // 2 - margin - expert_stream.REGION_BYTES * expert_stream.MIN_RING_REGIONS
    if tokens * bytes_per_token <= room:
        return max(tokens, chunks[-1])
    return next((chunk for chunk in chunks if chunk * bytes_per_token <= room), chunks[-1])
