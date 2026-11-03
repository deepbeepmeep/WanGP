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

from typing import Tuple
import torch
import torch.nn as nn

# from ..cache import Cache
from ....common.cache import Cache

from .attention.mmattn import NaSwinAttention
from ..mm import MMArg
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..mm import MMArg, MMModule
from ..mlp import get_mlp

MLP_INPUT_CHUNK_BYTES = 64 << 20  # the normalized rows of the video residual handed to the MLP at a time (its own chunking follows)



class NaMMSRTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        mlp_type: str,
        shared_weights: bool,
        rope_type: str,
        rope_dim: int,
        is_last_layer: bool,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights,)

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            rope_type=rope_type,
            rope_dim=rope_dim,
            shared_weights=shared_weights,
            attention_mode=attention_mode,
            window=kwargs.pop("window", None),
            window_method=kwargs.pop("window_method", None),
            attention_window_batch_size=kwargs.pop("attention_window_batch_size", 0),
        )

        self.mlp_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights, vid_only=is_last_layer)
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            vid_only=is_last_layer
        )
        self.ada = MMModule(ada, dim=dim, emb_dim=emb_dim, layers=["attn", "mlp"], shared_weights=shared_weights, vid_only=is_last_layer)
        self.is_last_layer = is_last_layer

    def forward(
        self,
        vid_list: list[torch.FloatTensor],  # l c
        txt_list: list[torch.FloatTensor],  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        # The video residual is updated in place: the attention by batches of windows and the MLP by chunks of rows normalize, modulate,
        # process, gate and add back their own rows, so no full-size branch tensor exists beside the residual. The text is processed whole.
        vid, txt = vid_list.pop(), txt_list.pop()
        vid_ada, txt_ada = self.ada.module("vid"), self.ada.module("txt")
        # the cached modulation rows also serve the 3B's output modulation (as upstream: it reads this block layout's "attn" entry)
        vid_kwargs = {"emb": emb, "hid_len": cache("vid_len", lambda: vid_shape.prod(-1)), "cache": cache, "branch_tag": "vid"}
        txt_kwargs = {"emb": emb, "hid_len": cache("txt_len", lambda: txt_shape.prod(-1)), "cache": cache, "branch_tag": "txt"}
        attn_norm, mlp_norm, mlp = self.attn_norm.module("vid"), self.mlp_norm.module("vid"), self.mlp.module("vid")

        def vid_attn_in(x_list):
            x = attn_norm(x_list.pop())
            return vid_ada(x, layer="attn", mode="in", **vid_kwargs)

        txt_attn = self.attn_norm.module("txt")(txt)
        if txt_ada is not None:
            txt_attn = txt_ada(txt_attn, layer="attn", mode="in", **txt_kwargs)
        txt_attn_list = [txt_attn]
        txt_attn = None
        txt_attn = self.attn(vid, txt_attn_list, vid_shape, txt_shape, cache, vid_attn_in, lambda x: vid_ada(x, layer="attn", mode="out", **vid_kwargs))
        if txt_ada is not None:
            txt_attn = txt_ada(txt_attn, layer="attn", mode="out", **txt_kwargs)
        txt_attn.add_(txt)
        txt = None

        gate, rows = vid_ada.gate(emb, "mlp").reshape(-1), max(1, MLP_INPUT_CHUNK_BYTES // (vid.shape[-1] * vid.element_size()))
        for start in range(0, vid.shape[0], rows):
            residual = vid[start:start + rows]
            x_list = [vid_ada(mlp_norm(residual), layer="mlp", mode="in", **vid_kwargs)]
            mlp.forward_residual(x_list, residual, gate)
        if txt_ada is None:  # the 3B's last block has no text MLP: its text output is not used
            return vid, txt_attn, vid_shape, txt_shape

        txt_mlp = txt_ada(self.mlp_norm.module("txt")(txt_attn), layer="mlp", mode="in", **txt_kwargs)
        txt_mlp = txt_ada(self.mlp.module("txt")(txt_mlp), layer="mlp", mode="out", **txt_kwargs)
        txt_mlp.add_(txt_attn)
        return vid, txt_mlp, vid_shape, txt_shape
