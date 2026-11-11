# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from dataclasses import dataclass
import torch
from mmgp import cuda_graphs


@dataclass
class Context:
    is_prefill: bool = False
    has_previous_state: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    speculative_verify: bool = False
    # mixed-precision KV cache (layers.kv_formats): the exact-overlay row of each written token (-1: none), and per sequence the first row
    # of its overlay (-1: none) and the first position read from the ring
    overlay_mapping: torch.Tensor | None = None
    overlay_base: torch.Tensor | None = None
    ring_start: torch.Tensor | None = None
    # tree verification: per verify row, int64 bits of the verify rows it attends to (its ancestors and itself); None for a chain
    tree_mask: torch.Tensor | None = None
    # and the linear-attention layers' view of the tree: its root-to-leaf paths [P, depth + 1] (node indices, padded with the leaf) and, per node,
    # the row of the flattened paths that computes it
    tree_paths: torch.Tensor | None = None
    tree_rows: torch.Tensor | None = None
    # the dense prefill's key and value buffers (layers.attention_mixed.dense): one pair for the layers of a pass, released with its context
    dense_kv: tuple | None = None
    dense_slices: list | None = None  # of a long context read by slices: (first key, keys, the slice's cu_seqlens_k)

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, has_previous_state=False, speculative_verify=False,
                overlay_mapping=None, overlay_base=None, ring_start=None, tree_mask=None, tree_paths=None, tree_rows=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, bool(has_previous_state), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, bool(speculative_verify),
                       overlay_mapping, overlay_base, ring_start, tree_mask, tree_paths, tree_rows)

def use_context(context):
    # an existing context made current as is (its buffers kept), e.g. to switch between the sequences of a pass and back
    global _CONTEXT
    _CONTEXT = context

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


def _graph_key():
    # the Python values of the context that change what the blocks compute: part of the keys of their MMGP graphs
    context = _CONTEXT
    return context.is_prefill, context.has_previous_state, context.speculative_verify, context.block_tables is None, context.max_seqlen_q, context.max_seqlen_k, context.tree_mask is not None


cuda_graphs.register_key(_graph_key)
