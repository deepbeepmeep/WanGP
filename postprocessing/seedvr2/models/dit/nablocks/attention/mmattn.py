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
# Modified for WanGP.

from itertools import accumulate
from typing import Callable, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn.modules.utils import _triple
from shared import attention_kit
from shared.kernels import ops

from .....common.cache import Cache

from ... import na
from ...mm import MMArg, MMModule
from ...normalization import norm_layer_type
from ...rope import get_na_rope
from ...window import get_window_op


def qkv_split_map(heads: int, head_dim: int) -> dict:
    """mmgp fused_split_map of the checkpoints' fused proj_qkv (rows q | k | v) into the proj_q, proj_k and proj_v modules."""
    return {f"proj_qkv.{branch}": {"mapped_modules": [f"proj_{name}.{branch}" for name in "qkv"], "split_sizes": [heads * head_dim] * 3} for branch in ("vid", "txt", "all")}


def _norm_rope_(query, key, rope, q_norms, k_norms):
    """attention_kit's norm_rope: the q/k RMSNorm of each stream's segment of the joined tokens and the rotation of interleaved pairs."""
    for (start, stop, q_norm), (_, _, k_norm) in zip(q_norms, k_norms):
        ops.interleaved_norm_rope_(None if query is None else query[:, start:stop], None if key is None else key[:, start:stop], rope[:, start:stop], q_norm, k_norm)


class NaMMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        # the checkpoints' fused proj_qkv is split at load (qkv_split_map): the attention kit projects and quantizes q, k and v one by one
        self.proj_q = MMModule(nn.Linear, dim, inner_dim, bias=qk_bias, shared_weights=shared_weights)
        self.proj_k = MMModule(nn.Linear, dim, inner_dim, bias=qk_bias, shared_weights=shared_weights)
        self.proj_v = MMModule(nn.Linear, dim, inner_dim, bias=qk_bias, shared_weights=shared_weights)
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            qk_norm,
            dim=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        self.window_batch_size = kwargs.pop("attention_window_batch_size", 0)

    def projections(self, branch: str):
        return self.proj_q.module(branch), self.proj_k.module(branch), self.proj_v.module(branch)


class NaSwinAttention(NaMMAttention):
    def __init__(
        self,
        *args,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__(*args, attention_mode=attention_mode, **kwargs)
        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.window_op = get_window_op(window_method)

    def _window_batches(self, window_rows: torch.Tensor, shapes: list, offsets: list) -> list:
        """(window indices, their rows, shape) of each batch of at most window_batch_size windows of one shape, in order of first
        appearance of the shapes: batches of equal windows, attended without padding."""
        groups = {}
        for index, shape in enumerate(shapes):
            groups.setdefault(shape, []).append(index)
        size, batches = self.window_batch_size or len(shapes), []
        for shape, indices in groups.items():
            for start in range(0, len(indices), size):
                windows = indices[start:start + size]
                rows = torch.cat([window_rows[offsets[index]:offsets[index + 1]] for index in windows])
                batches.append((torch.tensor(windows, device=window_rows.device), rows, shape))
        return batches

    def forward(
        self,
        vid: torch.FloatTensor,  # l c, the video residual, updated in place
        txt_list: list[torch.FloatTensor],  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
        vid_in: Callable[[list[torch.Tensor]], torch.Tensor],
        vid_out: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.FloatTensor:
        """Window attention added to the video residual: each batch of equal windows gathers its rows, vid_in([rows]) normalizes and
        modulates them, and vid_out gates their projected attention output, added back to the same rows (the windows partition the
        tokens, so no window reads rows another one has updated). The text joins every window as a second stream of the attention kit
        (q, k and v projected one by one, each quantized as soon as it exists with the staged SageAttention 2 core); its outputs,
        averaged over the windows, are returned projected. q and k are normalized and rotated in their projection, by one RoPE table
        per window shape."""

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_rows, window_shape, window_count = cache_win("win_transform", lambda: na.window_idx(vid_shape, make_window))
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        offsets = cache_win("vid_offsets_py", lambda: list(accumulate(vid_len_win.tolist(), initial=0)))
        shapes = cache_win("vid_shapes_py", lambda: [tuple(shape) for shape in window_shape.tolist()])
        batches = cache_win("window_batches", lambda: self._window_batches(window_rows, shapes, offsets))
        txt_len_py = cache_win("txt_len_py", lambda: int(txt_len[0].item()))

        txt = txt_list.pop()
        tables = self.rope.window_tables(shapes, txt_len_py, cache_win, txt.device, self.head_dim // 2)
        vid_projections, txt_projections = self.projections("vid"), self.projections("txt")
        q_norms, k_norms = (self.norm_q.module("vid"), self.norm_q.module("txt")), (self.norm_k.module("vid"), self.norm_k.module("txt"))
        txt_windows = txt.new_empty(txt_len_py, len(shapes), self.heads, self.head_dim)
        for windows, rows, shape in batches:
            count, length = len(windows), rows.shape[0] // len(windows)
            q_segments, k_segments = [[(0, length, norms[0]), (length, length + txt_len_py, norms[1])] for norms in (q_norms, k_norms)]
            rope = tables[shape].unsqueeze(0)
            x_list = [vid_in([vid.index_select(0, rows)]).view(count, length, -1)]
            streams = [(x_list, vid_projections), ([txt.unsqueeze(0).repeat(count, 1, 1)], txt_projections)]  # a copy: the kit consumes its inputs
            out = attention_kit.joint_qkv_attention(streams, self.heads, self.head_dim, lambda query, key, group: _norm_rope_(query, key, rope, q_segments, k_segments),
                                                    fused_quant=lambda query, key, group, core, settings: ops.split_half_norm_rope_quant_(query, key, rope, q_segments, k_segments, core, settings, interleaved=True))
            txt_windows.index_copy_(1, windows, out[:, length:].transpose(0, 1))
            vid_attn_list = [out[:, :length].reshape(count * length, -1)]
            out = None
            vid.index_add_(0, rows, vid_out(self.proj_out.forward_vid_disposable(vid_attn_list)))

        txt = None
        txt_out = txt_windows.mean(1).flatten(1)
        txt_windows = None
        txt_out_list = [txt_out]
        txt_out = None
        return self.proj_out.forward_txt_disposable(txt_out_list)
