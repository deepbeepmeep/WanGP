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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable
import torch
from torch import nn
from tqdm import tqdm

from ...common.cache import Cache

from . import na
from .embedding import TimeEmbedding
from .modulation import get_ada_layer
from .nablocks import get_nablock
from .normalization import get_norm_layer
from .patch import get_na_patch_layers

OUTPUT_CHUNK_BYTES = 64 << 20  # rows of the final hidden states normalized at a time

# Fake func, no checkpointing is required for inference
def gradient_checkpointing(module: Union[Callable, nn.Module], *args, enabled: bool, **kwargs):
    return module(*args, **kwargs)

@dataclass
class NaDiTOutput:
    vid_sample: torch.Tensor


class NaDiT(nn.Module):
    """
    Native Resolution Diffusion Transformer (NaDiT)
    """

    gradient_checkpointing = False

    def __init__(
        self,
        vid_in_channels: int,
        vid_out_channels: int,
        vid_dim: int,
        txt_in_dim: Union[int, List[int]],
        txt_dim: Optional[int],
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: Optional[str],
        norm_eps: float,
        ada: str,
        qk_bias: bool,
        qk_norm: Optional[str],
        patch_size: Union[int, Tuple[int, int, int]],
        num_layers: int,
        block_type: Union[str, Tuple[str]],
        mm_layers: Union[int, Tuple[bool]],
        mlp_type: str = "normal",
        patch_type: str = "v1",
        rope_type: Optional[str] = "rope3d",
        rope_dim: Optional[int] = None,
        window: Optional[Tuple] = None,
        window_method: Optional[Tuple[str]] = None,
        msa_type: Optional[Tuple[str]] = None,
        mca_type: Optional[Tuple[str]] = None,
        txt_in_norm: Optional[str] = None,
        txt_in_norm_scale_factor: int = 0.01,
        txt_proj_type: Optional[str] = "linear",
        vid_out_norm: Optional[str] = None,
        vid_only_last_layer: bool = True,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        ada = get_ada_layer(ada)
        norm = get_norm_layer(norm)
        qk_norm = get_norm_layer(qk_norm)
        rope_dim = rope_dim if rope_dim is not None else head_dim // 2
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        NaPatchIn, NaPatchOut = get_na_patch_layers(patch_type)
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )
        if not isinstance(txt_in_dim, int):
            self.txt_in = nn.ModuleList([])
            for in_dim in txt_in_dim:
                txt_norm_layer = get_norm_layer(txt_in_norm)(txt_dim, norm_eps, True)
                if txt_proj_type == "linear":
                    txt_proj_layer = nn.Linear(in_dim, txt_dim)
                else:
                    txt_proj_layer = nn.Sequential(
                        nn.Linear(in_dim, in_dim), nn.GELU("tanh"), nn.Linear(in_dim, txt_dim)
                    )
                torch.nn.init.constant_(txt_norm_layer.weight, txt_in_norm_scale_factor)
                self.txt_in.append(
                    nn.Sequential(
                        txt_proj_layer,
                        txt_norm_layer,
                    )
                )
        else:
            self.txt_in = (
                nn.Linear(txt_in_dim, txt_dim)
                if txt_in_dim and txt_in_dim != txt_dim
                else nn.Identity()
            )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
        )

        if window is None or isinstance(window[0], int):
            window = [window] * num_layers
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * num_layers

        if msa_type is None or isinstance(msa_type, str):
            msa_type = [msa_type] * num_layers
        if mca_type is None or isinstance(mca_type, str):
            mca_type = [mca_type] * num_layers

        self.blocks = nn.ModuleList(
            [
                get_nablock(block_type[i])(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_norm=qk_norm,
                    shared_weights=not (
                        (i < mm_layers) if isinstance(mm_layers, int) else mm_layers[i]
                    ),
                    mlp_type=mlp_type,
                    window=window[i],
                    window_method=window_method[i],
                    msa_type=msa_type[i],
                    mca_type=mca_type[i],
                    rope_type=rope_type,
                    rope_dim=rope_dim,
                    is_last_layer=vid_only_last_layer and i == num_layers - 1,  # the 3B has no text MLP / modulation in its last block
                    attention_mode=attention_mode,
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.vid_out_norm = None
        if vid_out_norm is not None:
            self.vid_out_norm = get_norm_layer(vid_out_norm)(
                dim=vid_dim,
                eps=norm_eps,
                elementwise_affine=True,
            )
            self.vid_out_ada = ada(
                dim=vid_dim,
                emb_dim=emb_dim,
                layers=["out"],
                modes=["in"],
            )

        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def forward(
        self,
        vid_list: List[torch.FloatTensor],  # l c
        txt_list: List[torch.FloatTensor],  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: Union[torch.LongTensor, List[torch.LongTensor]],  # b 1
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],  # b
        disable_cache: bool = False,  # for test
        abort_callback=None,
        progress_callback=None,
    ):
        cache = Cache(disable=disable_cache)

        # slice vid after patching in when using sequence parallelism
        txt = txt_list.pop()
        if isinstance(txt, list):
            assert isinstance(self.txt_in, nn.ModuleList)
            txt = [
                na.unflatten(fc(i), s) for fc, i, s in zip(self.txt_in, txt, txt_shape)
            ]  # B L D
            txt, txt_shape = na.flatten([torch.cat(t, dim=0) for t in zip(*txt)])
        else:
            txt = self.txt_in(txt)

        # Video input.
        # Sequence parallel slicing is done inside patching class.
        vid, vid_shape = self.vid_in(vid_list, vid_shape, cache)

        # Embedding input.
        emb = self.emb_in(timestep, device=vid.device, target_dtype=vid.dtype)

        # Body
        for i, block in enumerate(tqdm(self.blocks, desc="SeedVR2 DiT", leave=False)):
            if callable(abort_callback) and abort_callback():
                raise InterruptedError("SeedVR2 upscaling aborted")
            vid_list, txt_list = [vid], [txt]
            vid = txt = None
            vid, txt, vid_shape, txt_shape = gradient_checkpointing(
                enabled=(self.gradient_checkpointing and self.training),
                module=block,
                vid_list=vid_list,
                txt_list=txt_list,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                emb=emb,
                cache=cache,
            )
            if callable(progress_callback):
                progress_callback("SeedVR2 denoising layers", i + 1, len(self.blocks))
            if callable(abort_callback) and abort_callback():
                raise InterruptedError("SeedVR2 upscaling aborted")

        # Video output norm and projection, by chunks of rows when there is a norm: no normalized copy of the whole video.
        if self.vid_out_norm is None:
            out = self.vid_out.proj(vid)
        else:
            out = vid.new_empty(vid.shape[0], self.vid_out.proj.out_features)
            rows = max(1, OUTPUT_CHUNK_BYTES // (vid.shape[-1] * vid.element_size()))
            # the cache returns the blocks' "attn" modulation rows for this key, as upstream's inference does
            ada_kwargs = {"emb": emb, "hid_len": cache("vid_len", lambda: vid_shape.prod(-1)), "cache": cache, "branch_tag": "vid"}
            for start in range(0, vid.shape[0], rows):
                x = self.vid_out_ada(self.vid_out_norm(vid[start:start + rows]), layer="out", mode="in", **ada_kwargs)
                out[start:start + rows] = self.vid_out.proj(x)
        vid = None
        vid, vid_shape = self.vid_out.unpatchify([out], vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)
