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

from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _triple

from ....common.cache import Cache

from .. import na


class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(in_channels * t * h * w, dim)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if t > 1:
            assert vid.size(2) % t == 1
            vid = torch.cat([vid[:, :, :1]] * (t - 1) + [vid], dim=2)
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)
        return vid


class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(dim, out_channels * t * h * w)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = self.proj(vid)
        vid = rearrange(vid, "b T H W (t h w c) -> b c (T t) (H h) (W w)", t=t, h=h, w=w)
        if t > 1:
            vid = vid[:, :, (t - 1) :]
        return vid


class NaPatchIn(PatchIn):
    def forward(
        self,
        vid_list: list[torch.Tensor],  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),  # for test
    ) -> torch.Tensor:
        cache = cache.namespace("patch")
        vid = vid_list.pop()
        vid_shape_before_patchify = cache("vid_shape_before_patchify", lambda: vid_shape)
        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid = na.unflatten(vid, vid_shape)
            for i in range(len(vid)):
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid[i] = torch.cat([vid[i][:1]] * (t - vid[i].size(0) % t) + [vid[i]], dim=0)
                vid[i] = rearrange(vid[i], "(T t) (H h) (W w) c -> T H W (t h w c)", t=t, h=h, w=w)
            vid, vid_shape = na.flatten(vid)

        vid = self.proj(vid)
        return vid, vid_shape


class NaPatchOut(PatchOut):
    def forward(
        self,
        vid_list: list[torch.FloatTensor],  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),  # for test
    ) -> Tuple[
        torch.FloatTensor,
        torch.LongTensor,
    ]:
        return self.unpatchify([self.proj(vid_list.pop())], vid_shape, cache)

    def unpatchify(self, vid_list: list[torch.FloatTensor], vid_shape: torch.LongTensor, cache: Cache) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """The projected tokens (forward without the projection) back to frames."""
        cache = cache.namespace("patch")
        vid = vid_list.pop()
        vid_shape_before_patchify = cache.get("vid_shape_before_patchify")

        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid = na.unflatten(vid, vid_shape)
            for i in range(len(vid)):
                vid[i] = rearrange(vid[i], "T H W (t h w c) -> (T t) (H h) (W w) c", t=t, h=h, w=w)
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid[i] = vid[i][(t - vid_shape_before_patchify[i, 0] % t) :]
            vid, vid_shape = na.flatten(vid)

        return vid, vid_shape
