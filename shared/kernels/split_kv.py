# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Split-KV for the SageAttention 2 CUDA kernel of fused v scales and FP16 accumulation (shared.sage2_core.staged_attention): the
keys of each (batch item, head) split in a few ranges attended in one launch (the wangp_sage_split extension, built from
SageAttention's sources), the parts merged by their log-sum-exp (split_kv_triton). A launch over few heads and tokens fills a part
of the GPU's waves; splitting its keys gives the GPU more, shorter programs. The number of parts (1 to 4: 1 keeps the stock kernel)
is timed once per shape and GPU outside CUDA graph recording, on copies of the call's inputs, and kept on disk."""
import functools
import importlib.util

import torch

from shared.kernels.split_kv_triton import merge_parts_, triton_split_attention
from shared.kernels.triton_tuner import LaunchTuner

SPLIT_KV = False  # off: no gain inside the chunker's head partitions (they already end on full waves); True to compare
_TUNER = LaunchTuner("wangp_split_kv", 1, "Split-KV")
_PARTS = ((1,), (2,), (3,), (4,))


@functools.cache
def _operator():
    """The split launch of the extension, None when it is not installed."""
    if importlib.util.find_spec("_sage_split") is None:
        return None
    import _sage_split  # noqa: F401 (registers torch.ops.wangp_sage_split)
    return torch.ops.wangp_sage_split.qk_int8_sv_f8_accum_f16_fuse_v_scale_split


def available():
    return SPLIT_KV and _operator() is not None


def _run(operator, out, scratch, q, k, v, q_scale, k_scale, v_scale, gran, sm_scale, parts):
    lse = operator(q, k, v, scratch, q_scale, k_scale, v_scale, gran, sm_scale, parts)
    for item in range(out.shape[0]):  # the parts of batch item b are rows b * parts ... of the scratch
        merge_parts_(out[item:item + 1], scratch[item * parts:(item + 1) * parts], lse[item * parts:(item + 1) * parts])
    return out


def attention(out, q, k, v, q_scale, k_scale, v_scale, gran, sm_scale):
    """out = the attention of the staged (q, k, v) with the keys split where it is faster for this shape (True), else False and
    nothing done (the caller runs the stock kernel). out (batch, tokens, heads, head_dim), through its strides."""
    operator = _operator() if SPLIT_KV else None
    if operator is None:
        return False
    batch, tokens, heads, dim = q.shape
    identity = f"{batch}x{tokens}x{heads}x{dim}|kv{k.shape[1]}x{k.shape[2]}|g{gran}|{str(out.dtype).rsplit('.', 1)[-1]}"

    def prepare():  # copies of the inputs: the candidates never write the call's own output
        copies = [t.clone() for t in (q, k, v, q_scale, k_scale, v_scale)]
        target = torch.empty(out.shape, dtype=out.dtype, device=out.device)
        scratch = {parts: torch.empty((batch * parts, tokens, heads, dim), dtype=out.dtype, device=out.device) for (parts,) in _PARTS}

        def launch(config):
            parts = config[0]
            if parts == 1:
                return operator(*copies[:3], target, *copies[3:], gran, sm_scale, 1)
            return _run(operator, target, scratch[parts], *copies, gran, sm_scale, parts)
        return launch
    (parts,) = _TUNER.config("split_kv", identity, out.device, _PARTS, prepare, True)
    if parts == 1:
        return False
    scratch = torch.empty((batch * parts, tokens, heads, dim), dtype=out.dtype, device=out.device)
    _run(operator, out, scratch, q, k, v, q_scale, k_scale, v_scale, gran, sm_scale, parts)
    return True


def _merge(out, scratch, lse, parts):
    for item in range(out.shape[0]):
        merge_parts_(out[item:item + 1], scratch[item * parts:(item + 1) * parts], lse[item * parts:(item + 1) * parts])
    return out


def attention_triton(out, q, k, v, q_scale, k_scale, block_m, launch, stock):
    """As attention() for the per-block Triton kernel of SageAttention 2 (v in 16 bits, its scale applied by the caller after): its
    keys split by the Triton split kernel where that is faster for this shape (True), else False. stock(target, q, k, v, q_scale,
    k_scale): the caller's own launch, the candidate of one part."""
    if not SPLIT_KV:
        return False
    batch, tokens, heads, dim = q.shape
    identity = f"triton|{batch}x{tokens}x{heads}x{dim}|kv{k.shape[1]}x{k.shape[2]}|m{block_m}|{launch}|{str(out.dtype).rsplit('.', 1)[-1]}"

    def prepare():
        copies = [t.clone() for t in (q, k, v, q_scale, k_scale)]
        target = torch.empty(out.shape, dtype=out.dtype, device=out.device)
        scratch = {parts: torch.empty((batch * parts, tokens, heads, dim), dtype=out.dtype, device=out.device) for (parts,) in _PARTS}

        def launch_parts(config):
            parts = config[0]
            if parts == 1:
                return stock(target, *copies)
            return _merge(target, scratch[parts], triton_split_attention(scratch[parts], *copies, parts, block_m, launch), parts)
        return launch_parts
    (parts,) = _TUNER.config("split_kv_triton", identity, out.device, _PARTS, prepare, True)
    if parts == 1:
        return False
    scratch = torch.empty((batch * parts, tokens, heads, dim), dtype=out.dtype, device=out.device)
    _merge(out, scratch, triton_split_attention(scratch, q, k, v, q_scale, k_scale, parts, block_m, launch), parts)
    return True
