# TAEH3 decoder (https://huggingface.co/Kijai/MiniMax-H3-TAE), Apache License 2.0 (see shared/tinyvae/LICENSES/taeh3-Apache-2.0.txt),
# in the TAESD / TAEHV architecture family by Ollin Boer Bohan (MIT, see shared/tinyvae/LICENSES/taehv-MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from typing import Any

def build_h3_decoder(state_dict: dict[str, Any]) -> Any:
    import torch
    import torch.nn as nn
    from .taesd import Bands

    def conv(n_in: int, n_out: int, **kwargs: Any) -> nn.Conv2d:
        return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

    class Clamp(nn.Module):
        def forward(self, value: Any) -> Any:
            return torch.tanh(value / 3) * 3

    class Block(nn.Module):
        def __init__(self, n_in: int, n_out: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
            self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
            self.fuse = nn.ReLU()

        def forward(self, value: Any) -> Any:
            return self.fuse(self.conv(value) + self.skip(value))

    by_index: dict[int, dict[str, Any]] = {}
    for key, value in state_dict.items():
        head, separator, rest = key.partition(".")
        if not separator or not head.isdigit():
            raise ValueError(f"unexpected taeh3 state key: {key}")
        by_index.setdefault(int(head), {})[rest] = value

    modules: list[nn.Module] = []
    for index in range(max(by_index) + 1):
        entry = by_index.get(index)
        if entry is None:
            modules.append(Clamp() if index == 0 else nn.ReLU() if index == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            weight = entry["conv.0.weight"]
            modules.append(Block(weight.shape[1], weight.shape[0]))
        elif "weight" in entry:
            weight = entry["weight"]
            modules.append(conv(weight.shape[1], weight.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"unexpected taeh3 module keys at {index}: {sorted(entry)}")

    return Bands(*modules)  # large frames by bands of rows (see taesd.Bands)
