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
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

import math
from contextlib import contextmanager
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Conv3d

from .context_parallel_lib import cache_send_recv, get_cache_size
from .global_config import get_norm_limit
from .types import MemoryState, _inflation_mode_t, _memory_device_t
from ....optimization.memory_manager import retry_on_oom

# Single GPU inference - no distributed processing needed
#print("Warning: Using single GPU inference mode - distributed features disabled in causal_inflation_lib")

@contextmanager
def ignore_padding(model):
    orig_padding = model.padding
    model.padding = (0, 0, 0)
    try:
        yield
    finally:
        model.padding = orig_padding


class InflatedCausalConv3d(Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t,
        memory_device: _memory_device_t = "same",
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.memory_device = memory_device
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.
        self.memory_limit = float("inf")

    def set_memory_limit(self, value: float):
        self.memory_limit = value

    def set_memory_device(self, memory_device: _memory_device_t):
        self.memory_device = memory_device
    
    def memory_limit_conv(self, x, *, padding, prev_cache=None):
        # Compatible with no limit.
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=2)
            return super().forward(x)

        # The blocks of upstream's recursion (rows, then columns of a row block, until a padded block fits in the memory limit; each block
        # with the overlap of the previous one), so the same convolutions; but each block is assembled once from x and prev_cache (the
        # preceding frames) with its zero padding, and its output written into the output tensor: no concatenated, padded or output copies.
        t_cache = 0 if prev_cache is None else prev_cache.size(2)
        rows = self._split_blocks(list(x.size()), 3, padding, t_cache, x.element_size())
        out, top = None, 0
        for h0, h1, h_overlap, row_padding in rows or [(0, x.size(3), 0, padding)]:
            cols = None if rows is None else self._split_blocks([x.size(0), x.size(1), x.size(2) + t_cache, h1 - h0, x.size(4)], 4, row_padding, h_overlap, x.element_size())
            left = 0
            for w0, w1, w_overlap, block_padding in cols or [(0, x.size(4), 0, row_padding)]:
                block = self._padded_block(x, prev_cache, (h0 - h_overlap, h1), (w0 - w_overlap, w1), block_padding)
                with ignore_padding(self):
                    block = Conv3d.forward(self, block)
                if out is None:
                    size = [x.size(3) + padding[2] + padding[3], x.size(4) + padding[0] + padding[1]]
                    size = [(n - self.dilation[d] * (self.kernel_size[d] - 1) - 1) // self.stride[d] + 1 for d, n in zip((1, 2), size)]
                    out = block.new_empty((*block.shape[:3], *size))
                out[:, :, :, top:top + block.size(3), left:left + block.size(4)] = block
                left += block.size(4)
            top += block.size(3)
            block = None
        return out

    def _split_blocks(self, shape, split_dim, padding, cache_len, element_size):
        """Upstream's split of a block along split_dim (3: rows, 4: columns) when its padded size, with the cache_len overlap of the previous
        level along split_dim - 1, exceeds the memory limit: (start, stop, overlap with the previous block, padding) of each part, else None."""
        size = torch.tensor(shape)
        size[split_dim - 1] += cache_len
        size[-3:] += torch.tensor(padding).view(3, 2).sum(-1).flip(0)
        memory_occupy = size.prod() * element_size / 1024**3  # GiB
        if memory_occupy < self.memory_limit:
            return None
        num_splits = math.ceil(memory_occupy / self.memory_limit)
        size_per_split = shape[split_dim] // num_splits
        split_sizes = [size_per_split] * (num_splits - 1) + [shape[split_dim] - size_per_split * (num_splits - 1)]
        lpad_dim = (len(shape) - split_dim - 1) * 2
        parts, start, overlap = [], 0, 0
        for idx, length in enumerate(split_sizes):
            part_padding = list(padding)
            part_padding[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            part_padding[lpad_dim + 1] = self.padding[split_dim - 2] if idx == len(split_sizes) - 1 else 0
            parts.append((start, start + length, overlap, tuple(part_padding)))
            overlap = get_cache_size(conv_module=self, input_len=length + overlap, pad_len=part_padding[lpad_dim] + part_padding[lpad_dim + 1], dim=split_dim - 2)
            assert overlap <= length
            start += length
        return parts

    @staticmethod
    def _padded_block(x, prev_cache, rows, cols, padding):
        """F.pad(torch.cat([prev_cache, x], 2)[..., rows, cols], padding) built in one tensor."""
        t_cache = 0 if prev_cache is None else prev_cache.size(2)
        pw0, pw1, ph0, ph1, pt0, pt1 = padding
        h, w = rows[1] - rows[0], cols[1] - cols[0]
        block = x.new_zeros((x.size(0), x.size(1), t_cache + x.size(2) + pt0 + pt1, h + ph0 + ph1, w + pw0 + pw1))
        inner = block[:, :, pt0:pt0 + t_cache + x.size(2), ph0:ph0 + h, pw0:pw0 + w]
        if prev_cache is not None:
            inner[:, :, :t_cache] = prev_cache[:, :, :, rows[0]:rows[1], cols[0]:cols[1]]
        inner[:, :, t_cache:] = x[:, :, :, rows[0]:rows[1], cols[0]:cols[1]]
        return block

    def forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET
    ) -> Tensor:
        assert memory_state != MemoryState.UNSET
        if memory_state != MemoryState.ACTIVE:
            self.memory = None
        if (
            math.isinf(self.memory_limit)
            and torch.is_tensor(input)
        ):
            return self.basic_forward(input, memory_state)
        return self.slicing_forward(input, memory_state)

    def basic_forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET):
        mem_size = self.stride[0] - self.kernel_size[0]
        if (self.memory is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.memory, times=-1)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        memory = (
            input[:, :, mem_size:].detach()
            if (mem_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if (
            memory_state != MemoryState.DISABLED
            and not self.training
            and (self.memory_device is not None)
        ):
            self.memory = memory
        return super().forward(input)

    def slicing_forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET,
    ) -> Tensor:
        squeeze_out = False
        if torch.is_tensor(input):
            input = [input]
            squeeze_out = True

        cache_size = self.kernel_size[0] - self.stride[0]
        cache = cache_send_recv(
            input, cache_size=cache_size, memory=self.memory, times=self.temporal_padding * 2
        )

        # Single GPU inference - simplified memory management
        if (
            memory_state in [MemoryState.INITIALIZING, MemoryState.ACTIVE]  # use_slicing
            and not self.training
            and (self.memory_device is not None)
            and cache_size != 0
        ):
            if cache_size > input[-1].size(2) and cache is not None and len(input) == 1:
                input[0] = torch.cat([cache, input[0]], dim=2)
                cache = None
            if cache_size <= input[-1].size(2):
                self.memory = input[-1][:, :, -cache_size:].detach().contiguous()

        padding = tuple(x for x in reversed(self.padding) for _ in range(2))
        for i in range(len(input)):
            # Prepare cache for next input slice.
            next_cache = None
            cache_size = 0
            if i < len(input) - 1:
                cache_len = cache.size(2) if cache is not None else 0
                cache_size = get_cache_size(self, input[i].size(2) + cache_len, pad_len=0)
            if cache_size != 0:
                if cache_size > input[i].size(2) and cache is not None:
                    input[i] = torch.cat([cache, input[i]], dim=2)
                    cache = None
                assert cache_size <= input[i].size(2), f"{cache_size} > {input[i].size(2)}"
                next_cache = input[i][:, :, -cache_size:]

            # Conv forward for this input slice.
            input[i] = self.memory_limit_conv(
                input[i],
                padding=padding,
                prev_cache=cache
            )

            # Update cache.
            cache = next_cache

        return input[0] if squeeze_out else input

    def tflops(self, args, kwargs, output) -> float:
        if torch.is_tensor(output):
            output_numel = output.numel()
        elif isinstance(output, list):
            output_numel = sum(o.numel() for o in output)
        else:
            raise NotImplementedError
        return (2 * math.prod(self.kernel_size) * self.in_channels * (output_numel / 1e6)) / 1e6

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode != "none":
            state_dict = modify_state_dict(
                self,
                state_dict,
                prefix,
                inflate_weight_fn=inflate_weight,
                inflate_bias_fn=inflate_bias,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            (strict and self.inflation_mode == "none"),
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def init_causal_conv3d(
    *args,
    inflation_mode: _inflation_mode_t,
    **kwargs,
):
    """
    Initialize a Causal-3D convolution layer.
    Parameters:
        inflation_mode: Listed as below. It's compatible with all the 3D-VAE checkpoints we have.
            - none: No inflation will be conducted.
                    The loading logic of state dict will fall back to default.
            - tail / replicate: Refer to the definition of `InflatedCausalConv3d`.
    """
    return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, **kwargs)


NORM_CHUNK_BYTES = 256 << 20  # frames normalized per call in causal_norm_wrapper (one frame at least)


def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(norm_layer, (nn.LayerNorm, RMSNorm)):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = norm_layer(x)
            x = rearrange(x, "b h w c -> b c h w")
            return x
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = norm_layer(x)
            x = rearrange(x, "b t h w c -> b c t h w")
            return x
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4:
            return norm_layer(x)
        if x.ndim == 5:
            # The norm sees each frame alone: by chunks of frames (and of groups when a frame exceeds the norm memory limit) written
            # into the output, each chunk's values computed as on the whole tensor. In VRAM, the input, the output and one chunk's
            # copies, instead of the frames-first copy of the input, its normalized groups, their concatenation and the copy back.
            out = torch.empty_like(x)
            frame_bytes = x[:, :, :1].numel() * x.element_size()
            frames = max(1, min(x.size(2), NORM_CHUNK_BYTES // frame_bytes))
            group_chunks = 1
            if isinstance(norm_layer, nn.GroupNorm) and frame_bytes / 1024**3 > get_norm_limit():
                group_chunks = min(4 if x.element_size() == 2 else 2, norm_layer.num_groups)
                assert norm_layer.num_groups % group_chunks == 0
            channels = x.size(1) // group_chunks
            for start in range(0, x.size(2), frames):
                part = x[:, :, start:start + frames]
                count = part.size(2)
                part = rearrange(part, "b c t h w -> (b t) c h w")
                for c0 in range(0, x.size(1), channels):
                    if group_chunks == 1:
                        y = norm_layer(part)
                    else:
                        c1 = c0 + channels
                        y = F.group_norm(part[:, c0:c1], norm_layer.num_groups // group_chunks, norm_layer.weight[c0:c1], norm_layer.bias[c0:c1], norm_layer.eps)
                    out[:, c0:c0 + y.size(1), start:start + count] = rearrange(y, "(b t) c h w -> b c t h w", t=count)
                    del y
                del part
            return out
    raise NotImplementedError


def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    """
    Remove duplicated first frame features in the up-sampling process.
    """
    # Single GPU inference - always process
    if times == 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)


def extend_head(tensor: Tensor, times: int = 2, memory: Optional[Tensor] = None) -> Tensor:
    """
    When memory is None:
        - Duplicate first frame features in the down-sampling process.
    When memory is not None:
        - Concatenate memory features with the input features to keep temporal consistency.
    """
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    assert times >= 0, "Invalid input for function 'extend_head'!"
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        return torch.cat(tensors=(torch.tile(tensor[:, :, :1], tile_repeat), tensor), dim=2)


def inflate_weight(weight_2d: torch.Tensor, weight_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution weight matrix to a 3D one.
    Parameters:
        weight_2d:      The weight matrix of 2D conv to be inflated.
        weight_3d:      The weight matrix of 3D conv to be initialized.
        inflation_mode: the mode of inflation
    """
    assert inflation_mode in ["tail", "replicate"]
    assert weight_3d.shape[:2] == weight_2d.shape[:2]
    with torch.no_grad():
        if inflation_mode == "replicate":
            depth = weight_3d.size(2)
            weight_3d.copy_(weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth)
        else:
            weight_3d.fill_(0.0)
            weight_3d[:, :, -1].copy_(weight_2d)
    return weight_3d


def inflate_bias(bias_2d: torch.Tensor, bias_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution bias tensor to a 3D one
    Parameters:
        bias_2d:        The bias tensor of 2D conv to be inflated.
        bias_3d:        The bias tensor of 3D conv to be initialized.
        inflation_mode: Placeholder to align `inflate_weight`.
    """
    assert bias_3d.shape == bias_2d.shape
    with torch.no_grad():
        bias_3d.copy_(bias_2d)
    return bias_3d


def modify_state_dict(layer, state_dict, prefix, inflate_weight_fn, inflate_bias_fn):
    """
    the main function to inflated 2D parameters to 3D.
    """
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if weight_2d.dim() == 4:
            # Assuming the 2D weights are 4D tensors (out_channels, in_channels, h, w)
            weight_3d = inflate_weight_fn(
                weight_2d=weight_2d,
                weight_3d=layer.weight,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[weight_name] = weight_3d
        else:
            return state_dict
            # It's a 3d state dict, should not do inflation on both bias and weight.
    if bias_name in state_dict:
        bias_2d = state_dict[bias_name]
        if bias_2d.dim() == 1:
            # Assuming the 2D biases are 1D tensors (out_channels,)
            bias_3d = inflate_bias_fn(
                bias_2d=bias_2d,
                bias_3d=layer.bias,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[bias_name] = bias_3d
    return state_dict
