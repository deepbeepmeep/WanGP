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

from math import pi
from typing import Optional, Tuple

import torch
from torch import nn

from ...common.cache import Cache


class SeedRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, freqs_for: str, theta: float = 10000, max_freq: float = 10):
        super().__init__()
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        else:
            raise ValueError(f"Unsupported rotary frequency type: {freqs_for}")
        self.register_buffer("freqs", freqs)

    @property
    def device(self):
        return self.freqs.device

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        all_freqs = []
        for axis, size in enumerate(dims):
            positions = torch.linspace(-1, 1, steps=size, device=self.device) if self.freqs_for == "pixel" else torch.arange(size, device=self.device, dtype=self.freqs.dtype)
            freqs = torch.outer(positions, self.freqs).repeat_interleave(2, dim=-1)
            shape = [1] * len(dims) + [freqs.shape[-1]]
            shape[axis] = size
            all_freqs.append(freqs.reshape(shape).expand(*dims, -1))
        return torch.cat(all_freqs, dim=-1)


class RotaryEmbeddingBase(nn.Module):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__()
        self.rope = SeedRotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="pixel",
            max_freq=256,
        )

    def get_axial_freqs(self, *dims):
        return self.rope.get_axial_freqs(*dims)


class NaRotaryEmbedding3d(nn.Module):
    """The 7B's RoPE: video tokens only, pixel frequencies over each window's own coordinates (linspace(-1, 1) on every axis, as in
    rotary-embedding-torch 0.5.3), interleaved pairs, the first 3 * 2 * (dim // 3 // 2) channels rotated. The frequencies are built
    in FP32 here rather than read from the checkpoint."""

    def __init__(self, dim: int):
        super().__init__()
        self.freqs = torch.linspace(1.0, 128.0, dim // 3 // 2, dtype=torch.float32, device="cpu") * pi
        self.mm = False

    def window_tables(self, shapes: list, txt_len: int, cache: Cache, device: torch.device, pairs: int):
        """One table per window shape (f, h, w) over the window's tokens followed by the text's (see _pair_table); the text is not rotated."""
        def make_tables():
            freqs, axes, tables = self.freqs.to(device), {}, {}
            for f, h, w in set(shapes):
                for size in (f, h, w):
                    if size not in axes:
                        axes[size] = torch.outer(torch.linspace(-1, 1, size, device=device, dtype=torch.float32), freqs)
                angles = torch.zeros(f * h * w + txt_len, 3, freqs.numel(), device=device, dtype=torch.float32)
                window = angles[:f * h * w].view(f, h, w, 3, -1)
                window[..., 0, :] = axes[f][:, None, None]
                window[..., 1, :] = axes[h][None, :, None]
                window[..., 2, :] = axes[w][None, None]
                tables[(f, h, w)] = _pair_table(angles.view(f * h * w + txt_len, -1), pairs)
            return tables
        return cache("rope_tables_3d", make_tables)


class MMRotaryEmbeddingBase(RotaryEmbeddingBase):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__(dim, rope_dim)
        self.rope = SeedRotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="lang",
            theta=10000,
        )
        self.mm = True


class NaMMRotaryEmbedding3d(MMRotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)

    def forward(
        self,
        vid_qk_list: list[torch.FloatTensor],  # L h d
        vid_shape: torch.LongTensor,  # B 3
        txt_qk_list: list[torch.FloatTensor],  # L h d
        txt_shape: torch.LongTensor,  # B 1
        cache: Cache,
        vid_txt_shape: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_q, vid_k = vid_qk_list
        txt_q, txt_k = txt_qk_list
        vid_qk_list.clear()
        txt_qk_list.clear()
        vid_cos, vid_sin, txt_cos, txt_sin = self.get_cos_sin(vid_shape, txt_shape, cache, vid_q.device, vid_txt_shape)
        vid_q_list, vid_k_list = [vid_q], [vid_k]
        txt_q_list, txt_k_list = [txt_q], [txt_k]
        vid_q = vid_k = txt_q = txt_k = None
        vid_q = _apply_rotary_inplace(vid_q_list, vid_cos, vid_sin)
        vid_k = _apply_rotary_inplace(vid_k_list, vid_cos, vid_sin)
        txt_q = _apply_rotary_inplace(txt_q_list, txt_cos, txt_sin)
        txt_k = _apply_rotary_inplace(txt_k_list, txt_cos, txt_sin)
        return vid_q, vid_k, txt_q, txt_k

    def get_cos_sin(self, vid_shape: torch.LongTensor, txt_shape: torch.LongTensor, cache: Cache, device: torch.device, vid_txt_shape: Optional[torch.LongTensor] = None):
        def make_cos_sin():
            vid_freqs, txt_freqs = self.get_freqs(vid_shape, txt_shape, vid_txt_shape)
            vid_freqs, txt_freqs = vid_freqs.to(device), txt_freqs.to(device)
            return vid_freqs.cos(), vid_freqs.sin(), txt_freqs.cos(), txt_freqs.sin()
        return cache("mmrope_cos_sin_3d", make_cos_sin)

    def window_tables(self, shapes: list, txt_len: int, cache: Cache, device: torch.device, pairs: int):
        """One table per window shape (f, h, w) over the window's tokens, whose temporal positions follow the text's, then the text's
        (see _pair_table); the same frequencies as get_freqs."""
        def make_tables():
            txt_freqs, tables = self.get_axial_freqs(txt_len).repeat(1, 3), {}
            for f, h, w in set(shapes):
                freqs = self.get_axial_freqs(txt_len + f, h, w)[txt_len:].reshape(f * h * w, -1)
                tables[(f, h, w)] = _pair_table(torch.cat([freqs, txt_freqs])[:, 0::2].to(device), pairs)
            return tables
        return cache("mmrope_tables_3d", make_tables)

    def get_freqs(
        self,
        vid_shape: torch.LongTensor,
        txt_shape: torch.LongTensor,
        vid_txt_shape: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generate RoPE frequencies for variable batch shapes.
        
        The cache wrapper memoizes these data-dependent shape calculations.
        """
        # Calculate actual max dimensions needed for this batch
        max_temporal = 0
        max_height = 0
        max_width = 0
        max_txt_len = 0
        
        vid_txt_shape = txt_shape if vid_txt_shape is None else vid_txt_shape
        for (f, h, w), l in zip(vid_shape.tolist(), vid_txt_shape[:, 0].tolist()):
            max_temporal = max(max_temporal, l + f)  # Need up to l+f for temporal
            max_height = max(max_height, h)
            max_width = max(max_width, w)
            max_txt_len = max(max_txt_len, l)
        
        # Compute frequencies for actual max dimensions needed
        # Add small buffer to improve cache hits across similar batches
        vid_freqs = self.get_axial_freqs(
            min(max_temporal + 16, 1024),  # Cap at 1024, add small buffer
            min(max_height + 4, 128),      # Cap at 128, add small buffer  
            min(max_width + 4, 128)        # Cap at 128, add small buffer
        )
        txt_freqs = self.get_axial_freqs(min(max_txt_len + 16, 1024))
        
        # Now slice as before
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), vid_txt_shape[:, 0].tolist()):
            vid_freq = vid_freqs[l : l + f, :h, :w].reshape(-1, vid_freqs.size(-1))
            vid_freq_list.append(vid_freq)
        for l in txt_shape[:, 0].tolist():
            txt_freq = txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1))
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


def _pair_table(angles: torch.Tensor, pairs: int) -> torch.Tensor:
    """(tokens, 1, pairs, 2) FP32 cos / sin of interleaved channel pairs for ops.interleaved_norm_rope_: angles (tokens, rotated pairs),
    cos and sin taken in the angles' dtype as the former per-token tables were; the pairs past the rotated ones unrotated (cos 1, sin 0)."""
    table = torch.zeros(angles.shape[0], 1, pairs, 2, device=angles.device, dtype=torch.float32)
    table[:, 0, angles.shape[1]:, 0] = 1
    table[:, 0, :angles.shape[1], 0] = angles.cos()
    table[:, 0, :angles.shape[1], 1] = angles.sin()
    return table


def _apply_rotary_inplace(x_list: list[torch.Tensor], cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    x = x_list.pop()
    rotated = x[..., :cosine.shape[-1]].unflatten(-1, (-1, 2))
    cosine = cosine.unsqueeze(1).unflatten(-1, (-1, 2))
    sine = sine.unsqueeze(1).unflatten(-1, (-1, 2))
    scratch = torch.empty_like(rotated, dtype=torch.float32)
    scratch[..., 0].copy_(rotated[..., 0]).mul_(cosine[..., 0]).addcmul_(rotated[..., 1], sine[..., 0], value=-1)
    scratch[..., 1].copy_(rotated[..., 1]).mul_(cosine[..., 1]).addcmul_(rotated[..., 0], sine[..., 1])
    rotated.copy_(scratch)
    return x


def get_na_rope(rope_type: Optional[str], dim: int):
    if rope_type is None:
        return None
    if rope_type == "mmrope3d":
        return NaMMRotaryEmbedding3d(dim=dim)
    if rope_type == "rope3d":
        return NaRotaryEmbedding3d(dim=dim)
    raise NotImplementedError(f"{rope_type} is not supported.")
