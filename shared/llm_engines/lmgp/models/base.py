# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""The contract between a model and the LM engine (engine/model_runner.py), and the code its models share to meet it.

A model of the engine returns its hidden states from forward(input_ids, positions, inputs_embeds=None, ...) and its logits from
compute_logits(hidden_states). The engine reads the optional hooks below with getattr, so models outside this package (Fish, YuE2,
ACE-Step, Nemotron Transcribe) meet the contract without deriving from LMEngineModel; the class gives their defaults and their meaning.

Attributes (class or instance):
- _lm_engine_tower_prefix: the parameter-name prefix of the blocks MMGP streams ("blk.", "layers.", "model.layers."); with MMGP graphs
  the engine's own graphs read every other parameter. Required when _lm_engine_mmgp_graphs is set.
- _lm_engine_mmgp: MMGP manages the weights: the engine sizes its KV cache without reading the free VRAM.
- _lm_engine_mmgp_graphs: decoding passes are MMGP graphs (implies _lm_engine_mmgp): the model may stream its layers.
- _lm_engine_prefill_block_tables: the model reads the cache in every pass (prefill passes get block tables, prompts are chunked).
- _lm_engine_ids_with_embeds: the model also reads the token ids of embedded prompts.
- _lm_engine_adaptive_experts: its GPU experts follow the routing (mmgp elastic residency: the engine folds the routing counts).
- _lm_engine_measured_drafts: block drafters choose their draft count per window (MeasuredDrafts / ConfidentDrafts) in Auto.
- _lm_engine_output_size: what forward returns per token, when it is not hidden_size (Flash-Next: its residual streams).
- lm_engine_prompt_graph_rows: padded sizes of short prompt passes replayed as MMGP graphs (with lm_engine_prompt_graph_ready() and
  lm_engine_prompt_forward(input_ids, positions, outputs, real, draft_features)).

Methods, optional unless stated:
- lm_engine_graph_forward(input_ids, positions, outputs, draft_features=None): a decoding pass as a cuda_graphs.run stage, its states
  written into outputs (required with MMGP graphs). An output module with lm_engine_graph_forward(hidden, logits) (OutputStage) has the
  logits of a verification window computed in a graph too.
- lm_engine_segmented_graph_forward(...): a decoding pass split in stages around host work.
- lm_engine_prefill_chunk(tokens): the tokens of a prompt read together (else the runner's fixed chunk).
- lm_engine_prefetch(token_ids, first, offset=0): start the coming pass's reads before the pass.
- lm_engine_read_ahead(token_ids, start, stop): read what a layer takes from a drive for the next prompt pass.
- _lm_engine_tower_prefix of a drafter, _lm_engine_context_saver (Deepy): see their users.
"""
import torch
from torch import nn

from ..utils.context import get_context


def last_rows(hidden_states):
    """The rows of a pass whose logits are needed: the last prompt row of a single-sequence prefill pass, the one row of a decoding pass,
    every row otherwise (a verification window)."""
    context = get_context()
    if hidden_states.ndim == 3:
        if context.is_prefill and context.cu_seqlens_q is not None:
            hidden_states = hidden_states[:, int(context.cu_seqlens_q[-1].item()) - 1, :]
        elif hidden_states.shape[-2] == 1:
            hidden_states = hidden_states[:, 0, :]
    return hidden_states


class LMEngineModel(nn.Module):
    """A model of the LM engine: the defaults of the optional hooks (module docstring) and the code models share."""
    _lm_engine_mmgp = False
    _lm_engine_mmgp_graphs = False
    _lm_engine_prefill_block_tables = False
    _lm_engine_ids_with_embeds = False
    _lm_engine_adaptive_experts = False
    _lm_engine_measured_drafts = False
    _lm_engine_output_size = None
    lm_engine_prompt_graph_rows = ()

    def _sequence_positions(self, tokens, device):
        """The positions in the sequence (and the cache) of the tokens of this pass, on the device."""
        context = get_context()
        if context.speculative_verify:
            first = context.cu_seqlens_k[1] - tokens
        elif context.is_prefill:
            first = context.cu_seqlens_k[1] - context.cu_seqlens_q[1]
        else:
            first = context.context_lens[0] - 1
        # a graph capture's pass on dummy pages has no context: position 0 keeps what it reads and writes finite
        return first.long().clamp_min(0) + torch.arange(tokens, dtype=torch.long, device=device)


class OutputStage:
    """A model's output module (norm / mix and head) whose logits of a verification window the engine computes as a graph stage."""
    def lm_engine_graph_forward(self, hidden, logits):
        """The logits of hidden into their buffer (cuda_graphs.run stage): no Python between the pass and the sampling."""
        logits.copy_(self(hidden))
