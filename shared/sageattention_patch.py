# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Runtime equivalent of SageAttention PR #387; no installed files or tensor layouts change.

https://github.com/thu-ml/SageAttention/pull/387
Apply before attention runs. These JIT entry points are launched from Python, not called by other JIT functions.
"""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import re


def sageattention_version():
    """Version of the installed SageAttention. SageGP ships SageAttention 2.2 under its own distribution name. Raises
    PackageNotFoundError (an ImportError) when neither is installed."""
    try:
        return version("sageattention")
    except PackageNotFoundError:
        version("sagegp")
        return "2.2.0"


def _int64_offsets(source, names):
    pattern = re.compile(r"^(\s*(?:" + "|".join(names) + r")\s*=\s*)([^\n#]+)", re.MULTILINE)

    def promote(match):
        expression = match[2].rstrip()
        if re.search(r"\.to\(\s*tl\.int64\s*\)", expression):
            return match[0]
        return match[1] + f"({expression}).to(tl.int64)" + match[2][len(expression):]

    return pattern.sub(promote, source)


def patch_sageattention():
    """Promote vulnerable offsets in place, preserving imported aliases; already-fixed kernels are untouched."""
    quant_offsets = ("off_blk", "off_h", "off_b")
    attention_offsets = ("start_m",)
    targets = (
        ("quant_per_thread", quant_offsets),
        ("quant_per_block", quant_offsets),
        ("quant_per_block_varlen", (*quant_offsets, "cu_seqlens_input_start", "cu_seqlens_scale_start")),
        ("attn_qk_int8_per_block", attention_offsets),
        ("attn_qk_int8_per_block_causal", attention_offsets),
        ("attn_qk_int8_block_varlen", (*attention_offsets, "cu_seqlens_q_start", "cu_seqlens_k_start")),
        ("attn_qk_int8_per_block_causal_varlen", (*attention_offsets, "cu_seqlens_q_start", "cu_seqlens_k_start")),
    )
    changes = []
    for module_name, offsets in targets:
        module = import_module(f"sageattention.triton.{module_name}")
        for name, kernel in vars(module).items():
            if name == "_attn_fwd" or name.startswith("quant_") and name.endswith("_kernel"):
                source = _int64_offsets(kernel.src, offsets)
                if source != kernel.src:
                    changes.append((kernel, source))
    for kernel, source in changes:
        kernel._unsafe_update_src(source)  # Resets Triton's source hash: old disk-cache binaries cannot be reused.
        kernel.device_caches.clear()  # Also discard any previously compiled in-process specialization.
    return tuple(kernel.__module__ + "." + kernel.__name__ for kernel, _ in changes)
