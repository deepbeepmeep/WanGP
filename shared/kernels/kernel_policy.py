# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Policy for optional kernel optimizations.
Precision: "strict" keeps kernels whose error against a high-precision reference is no larger than PyTorch's (bit identity not
required); "fast" adds kernels that trade precision for speed.
Families: the INT8 Math Kernels choice also orders the kernel families of the other fused operations. The selected family comes first;
when it has no kernel for an operation, or that kernel is unavailable or excluded by the precision, the operation behaves as with Auto
(RANKING). Auto and Disabled (PyTorch INT8 linears) use RANKING. WanGP's own Triton kernels take the WGPKernels slot of an operation
that has no WGPKernels kernel."""

CHOICES = [("Preserve Precision", "strict"), ("Allow Faster Approximate Kernels", "fast")]
precision = "fast"
sage2_quality_override = None  # the generation's SageGP quality from its attention mode (sage3 -> fast, sageadaptative -> adaptive)
# Adaptive and Fast run the first denoising step in Precise: it sets the composition, and later steps follow it
SAGE2_FIRST_STEP_PRECISE_DEFAULT = True
sage2_first_step_precise = SAGE2_FIRST_STEP_PRECISE_DEFAULT
denoising_step = None  # the denoising step being computed (0 = first), set by the generation's progress callback
denoising_steps = 0  # the steps of the current denoising loop (0: unknown)
RANKING = ("wgpkernels", "kitchen", "triton", "pytorch")
families = RANKING


# Only the first block needs hooks. Every attention/head group it calls sees the
# same precision, including repeated CFG calls, recomputation and graph capture.
from contextvars import ContextVar
from weakref import WeakSet

_first_block = ContextVar("precise_first_block_depth", default=0)
_hooked_transformers = WeakSet()
_precise_modes = ("adaptive", "fast", "sol3")


def install_first_block_policy(transformer):
    import torch
    if transformer in _hooked_transformers:
        return
    towers = []
    for path, module in transformer.named_modules():
        name = path.rsplit(".", 1)[-1]
        if isinstance(module, torch.nn.ModuleList) and len(module) and (name in ("layers", "blocks", "h") or name.endswith("_blocks")):
            towers.append((path.count("."), module))
    if not towers:
        return

    # Some models have several sequential towers, or enter different towers on
    # different forwards. Protect the first one entered in each transformer call.
    entry = []

    def begin(module, args):
        entry.clear()

    def enter(module, args):
        if not entry:
            entry.append(module)
        if module is entry[0]:
            _first_block.set(_first_block.get() + 1)

    def leave(module, args, result):
        if entry and module is entry[0]:
            _first_block.set(_first_block.get() - 1)

    transformer.register_forward_pre_hook(begin, prepend=True)
    depth = min(level for level, _ in towers)
    seen = set()
    for level, tower in towers:
        if level != depth:
            continue
        kept = getattr(transformer, "kept_blocks", None) if tower is getattr(transformer, "blocks", None) else None
        block = tower[kept[0] if kept else 0]
        if block not in seen:
            block.register_forward_pre_hook(enter, prepend=True)
            block.register_forward_hook(leave, always_call=True)
            seen.add(block)
    _hooked_transformers.add(transformer)


def first_step_precision_enabled():
    """Whether this attention mode honors the precise first step and first block setting."""
    return sage2_first_step_precise and sage2_quality_override in _precise_modes


def first_step_precise():
    """Use the default dense attention throughout step zero of each protected mode."""
    return denoising_step == 0 and first_step_precision_enabled()


def precise_attention():
    return first_step_precision_enabled() and (denoising_step == 0 or _first_block.get() > 0)


def sol3_dense():
    # A global attention choice also reaches preprocessing and the VAE after the final step.
    outside_denoising = sage2_quality_override == "sol3" and (
        denoising_step is None or (denoising_steps > 0 and denoising_step >= denoising_steps))
    return outside_denoising or precise_attention()


def configure_sage2_first_step_precise(value):
    global sage2_first_step_precise
    sage2_first_step_precise = bool(value)


def set_denoising_step(step, steps=0):
    global denoising_step, denoising_steps
    denoising_step, denoising_steps = step, steps


def sage2_execution_quality():
    """The SageGP quality of the attention calls running: "regular" (Precise), "adaptive" or "fast"."""
    return "regular" if sage2_quality_override in (None, "sol3") or precise_attention() else sage2_quality_override


def configure_families(selection):
    global families
    families = (selection,) + tuple(family for family in RANKING if family != selection) if selection in RANKING[:3] else RANKING


def first(candidates):
    """The preferred family among candidates ({family: usable}), or None (the PyTorch path)."""
    for family in families:
        if candidates.get(family):
            return family
    return None


def configure(value):
    global precision
    if value not in ("strict", "fast"):
        raise ValueError(f"Unknown non-INT8 kernel precision: {value!r}")
    precision = value


def allow_approximate():
    return precision == "fast"


