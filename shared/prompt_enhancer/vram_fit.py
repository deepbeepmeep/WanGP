# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""VRAM fit of a local LM of the LM engine, for speculative decoding Auto (config.resolve_prompt_enhancer_speculative_decoding(fit=)):
what the model needs beside its weights (its KV cache and activations, the margins of MMGP's dynamic preload), what its drafter adds,
and what this process may hold. Each model supplies its own weight, KV-cache and drafter bytes."""
import json
import struct

import torch


def available_vram() -> int:
    """VRAM this process may hold: what its allocator reserved and can still reserve."""
    from mmgp import allocator
    if allocator.active == "vmm":
        return allocator.stats()["reserved"] + allocator.room()
    return torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()


def lm_fit(weights: int, kv_bytes: int, activations: int, drafter: int, preload_margin: int | None = None) -> tuple[int, int, int]:
    """(VRAM the model needs without its drafter, VRAM the drafter adds, VRAM this process may hold), in bytes. preload_margin replaces
    MMGP's dynamic-preload margin when given."""
    from mmgp import auto_preload, memory_pressure
    work = kv_bytes + activations
    margin = (auto_preload.MARGIN if preload_margin is None else preload_margin) + memory_pressure.HEADROOM + int(auto_preload.MARGIN_SHARE * work)
    return weights + work + margin, drafter, available_vram()


def safetensors_sizes(path: str, prefix: str = "") -> dict[str, int]:
    """Bytes of each tensor of a safetensors file whose name starts with prefix, read from its header."""
    with open(path, "rb") as reader:
        header = json.loads(reader.read(struct.unpack("<Q", reader.read(8))[0]))
    return {name: info["data_offsets"][1] - info["data_offsets"][0] for name, info in header.items() if name.startswith(prefix) and name != "__metadata__"}
