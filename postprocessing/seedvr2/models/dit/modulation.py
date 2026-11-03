# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Callable, List, Optional
import torch
from einops import rearrange
from torch import nn

from ...common.cache import Cache

# (dim: int, emb_dim: int)
ada_layer_type = Callable[[int, int], nn.Module]


def get_ada_layer(ada_layer: str) -> ada_layer_type:
    if ada_layer == "single":
        return AdaSingle
    raise NotImplementedError(f"{ada_layer} is not supported")


def expand_dims(x: torch.Tensor, dim: int, ndim: int):
    """
    Expand tensor "x" to "ndim" by adding empty dims at "dim".
    Example: x is (b d), target ndim is 5, add dim at 1, return (b 1 1 1 d).
    """
    shape = x.shape
    shape = shape[:dim] + (1,) * (ndim - len(shape)) + shape[dim:]
    return x.reshape(shape)


class AdaSingle(nn.Module):
    def __init__(
        self,
        dim: int,
        emb_dim: int,
        layers: List[str],
        modes: List[str] = ["in", "out"],
    ):
        assert emb_dim == 6 * dim, "AdaSingle requires emb_dim == 6 * dim"
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = layers
        for l in layers:
            if "in" in modes:
                self.register_parameter(f"{l}_shift", nn.Parameter(torch.randn(dim) / dim**0.5))
                self.register_parameter(
                    f"{l}_scale", nn.Parameter(torch.randn(dim) / dim**0.5 + 1)
                )
            if "out" in modes:
                self.register_parameter(f"{l}_gate", nn.Parameter(torch.randn(dim) / dim**0.5))

    def forward(
        self,
        hid: torch.FloatTensor,  # b ... c
        emb: torch.FloatTensor,  # b d
        layer: str,
        mode: str,
        cache: Cache = Cache(disable=True),
        branch_tag: str = "",
        hid_len: Optional[torch.LongTensor] = None,  # b
    ) -> torch.FloatTensor:
        idx = self.layers.index(layer)
        emb = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., idx, :]
        emb = expand_dims(emb, 1, hid.ndim + 1)

        if hid_len is not None:
            emb = cache(
                f"emb_repeat_{idx}_{branch_tag}",
                lambda: emb if emb.shape[0] == 1 else torch.repeat_interleave(emb, hid_len, dim=0),
            )

        shiftA, scaleA, gateA = emb.unbind(-1)
        shiftB, scaleB, gateB = (
            getattr(self, f"{layer}_shift", None),
            getattr(self, f"{layer}_scale", None),
            getattr(self, f"{layer}_gate", None),
        )

        if mode == "in":
            return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
        if mode == "out":
            if gateB is not None:
                return hid.mul_(gateA + gateB)
            else:
                # If no gate parameter, just use the embedding gate
                return hid.mul_(gateA)
            
        raise NotImplementedError

    def gate(self, emb: torch.FloatTensor, layer: str) -> torch.FloatTensor:
        """The multiplier of the "out" mode of a layer (b, dim), for a gated residual computed elsewhere."""
        gate = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., self.layers.index(layer), 2]
        gate_b = getattr(self, f"{layer}_gate", None)
        return gate if gate_b is None else gate + gate_b

    def extra_repr(self) -> str:
        return f"dim={self.dim}, emb_dim={self.emb_dim}, layers={self.layers}"
