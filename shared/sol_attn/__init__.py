# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Reusable Sol-Attn Triton backend."""

from .interface import prepare_kv, sol_attn, sol_attn_prepared, staged_supported, validate_runtime


def __getattr__(name):
    if name in ("qk_rms_norm_rope_", "rms_norm_rope_"):
        from . import qk_norm_rope
        return getattr(qk_norm_rope, name)
    raise AttributeError(name)

__all__ = ["prepare_kv", "qk_rms_norm_rope_", "rms_norm_rope_", "sol_attn", "sol_attn_prepared", "staged_supported", "validate_runtime"]
