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


def first_step_precise():
    """True while the first denoising step of an Adaptive or Fast generation runs in Precise (its block calls are not graphed)."""
    return sage2_first_step_precise and denoising_step == 0 and sage2_quality_override in ("adaptive", "fast")


def configure_sage2_first_step_precise(value):
    global sage2_first_step_precise
    sage2_first_step_precise = bool(value)


def set_denoising_step(step, steps=0):
    global denoising_step, denoising_steps
    denoising_step, denoising_steps = step, steps


def sage2_execution_quality():
    """The SageGP quality of the attention calls running: "regular" (Precise), "adaptive" or "fast"."""
    return "regular" if sage2_quality_override is None or first_step_precise() else sage2_quality_override


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


