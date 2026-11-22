# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)


import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mmgp import cuda_graphs
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from shared import attention_kit
from shared.kernels import ops
import functools

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen_inplace(x_list: list, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings in-place using real arithmetic (no complex numbers).

    Args:
        x_list: Single-element list containing query or key tensor [B, S, H, D].
                The list is cleared after use to free memory early.
        freqs_cis: Precomputed frequency tensor with cos/sin stacked [S, D//2, 2]

    Returns:
        Tensor with rotary embeddings applied [B, S, H, D]
    """
    return ops.interleaved_rope(x_list, freqs_cis)


def _norm_rope_rows(x, norm, freqs):
    """norm then RoPE of rows of q or k (batch, tokens, heads, head_dim), as QwenDoubleStreamAttnProcessor2_0 computes them."""
    if norm is not None:
        x = norm(x)
    if freqs is None:
        return x
    x_list, x = [x], None
    return apply_rotary_emb_qwen_inplace(x_list, freqs)


def _norm_rope_(query, key, segments):
    """attention_kit callback: per segment of tokens (start, stop, q norm, k norm, freqs), the norms and RoPE of q and/or k in place, by
    chunks of tokens; freqs (segment tokens, head_dim // 2, 2) or None."""
    for tensor, index in ((query, 2), (key, 3)):
        if tensor is not None:
            for segment in segments:
                start, norm, freqs = segment[0], segment[index], segment[4]
                if norm is not None or freqs is not None:
                    attention_kit.rows_(tensor, lambda part, first, last: _norm_rope_rows(part, norm, None if freqs is None else freqs[first - start:last - start]), start, segment[1])


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            conditioning = conditioning + addition_t_emb.to(dtype=hidden_states.dtype)

        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        Returns:
            Tensor of shape [len(index), dim//2, 2] with cos/sin stacked in last dim
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        # Stack cos and sin instead of using complex numbers (better for torch.compile)
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling() and False:
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        # freqs_pos/neg now have shape [4096, axes_dim[i]//2, 2] with cos/sin in last dim
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        # Keep the cos/sin dimension (2) separate throughout
        freqs_frame = freqs_pos[0][idx : idx + frame]  # [frame, d0//2, 2]
        freqs_frame = freqs_frame.view(frame, 1, 1, -1, 2).expand(frame, height, width, -1, 2)

        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1, 2).expand(frame, height, width, -1, 2)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1, 2).expand(frame, height, width, -1, 2)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1, 2).expand(frame, height, width, -1, 2)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1, 2).expand(frame, height, width, -1, 2)

        # Concatenate on freq dimension (dim -2), keep cos/sin dimension (dim -1) intact
        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-2)  # [F, H, W, total//2, 2]
        freqs = freqs.reshape(seq_lens, -1, 2)  # [seq_lens, total//2, 2]
        return freqs.clone().contiguous()


class QwenEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        Returns:
            Tensor of shape [len(index), dim//2, 2] with cos/sin stacked in last dim
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            else:
                # For the condition image, we set the layer index to -1.
                video_freq = self._compute_condition_freqs(frame, height, width)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame]  # [frame, d0//2, 2]
        freqs_frame = freqs_frame.view(frame, 1, 1, -1, 2).expand(frame, height, width, -1, 2)

        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1, 2).expand(frame, height, width, -1, 2)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1, 2).expand(frame, height, width, -1, 2)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1, 2).expand(frame, height, width, -1, 2)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1, 2).expand(frame, height, width, -1, 2)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-2)  # [F, H, W, total//2, 2]
        freqs = freqs.reshape(seq_lens, -1, 2)
        return freqs.clone().contiguous()

    @functools.lru_cache(maxsize=None)
    def _compute_condition_freqs(self, frame, height, width):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:]  # [1, d0//2, 2]
        freqs_frame = freqs_frame.view(frame, 1, 1, -1, 2).expand(frame, height, width, -1, 2)

        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1, 2).expand(frame, height, width, -1, 2)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1, 2).expand(frame, height, width, -1, 2)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1, 2).expand(frame, height, width, -1, 2)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1, 2).expand(frame, height, width, -1, 2)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-2)  # [F, H, W, total//2, 2]
        freqs = freqs.reshape(seq_lens, -1, 2)
        return freqs.clone().contiguous()


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states_list = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        residual=None,
    ) -> torch.FloatTensor:
        """residual (extreme low VRAM, the image stream an mmgp RowSource): (the image residual, its gate table, segments (start, stop,
        row) over its rows), into which the kit adds the image's projected and gated attention output: returns (None, text output)."""

        hidden_states, encoder_hidden_states = hidden_states_list
        hidden_states_list.clear()

        seq_txt, seq_img = encoder_hidden_states.shape[1], hidden_states.shape[1]
        img_freqs, txt_freqs = (None, None) if image_rotary_emb is None else image_rotary_emb
        # joint attention of the text then image tokens, each stream with its projections, QK norms and RoPE
        streams = [([encoder_hidden_states], (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)), ([hidden_states], (attn.to_q, attn.to_k, attn.to_v))]
        del hidden_states, encoder_hidden_states
        segments = ((0, seq_txt, attn.norm_added_q, attn.norm_added_k, txt_freqs), (seq_txt, seq_txt + seq_img, attn.norm_q, attn.norm_k, img_freqs))
        # the fused norm + RoPE + Sage quantization: the norms of both streams by segments of the joined tokens, one (1, tokens, 1, pairs, 2)
        # cos / sin table of the joined tokens (interleaved pairs, as apply_rotary_emb_qwen_inplace)
        q_segments, k_segments = ([(start, stop, segment[index]) for start, stop, *segment in segments] for index in (0, 1))
        table = None if image_rotary_emb is None else torch.cat((txt_freqs[:seq_txt], img_freqs[:seq_img]), dim=0)[None, :, None]
        options = {} if residual is None else dict(out_proj=attn.to_out[0], residual=residual[0], gate=residual[1], segments=residual[2], out_stream=1,
                                                         by_items=True)  # a batch: the items one after another (batch-wide tables: one row)
        joint_hidden_states = attention_kit.joint_qkv_attention(streams, attn.heads, attn.to_q.out_features // attn.heads,
                                                                lambda query, key, group: _norm_rope_(query, key, segments),
                                                                fused_quant=lambda query, key, group, core, settings: ops.split_half_norm_rope_quant_(query, key, table, q_segments, k_segments, core, settings, interleaved=True),
                                                                **options).flatten(2, 3)
        if residual is not None:  # the image's output already in its residual
            return None, attn.to_add_out(joint_hidden_states)

        # Apply output projections
        img_attn_output = attn.to_out[0](joint_hidden_states[:, seq_txt:])
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout
        txt_attn_output = attn.to_add_out(joint_hidden_states[:, :seq_txt])
        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.zero_cond_t = zero_cond_t

    @staticmethod
    def _apply_ffn_chunked(ffn_in: torch.Tensor, mlp: nn.Module, residual=None, segments=None) -> None:
        """ffn_in <- mlp(ffn_in) in place through the shared chunked FFN, or residual += gate * mlp(ffn_in) with the gates of segments."""
        ops.mlp([ffn_in], mlp.net[0].proj, mlp.net[2], "gelu", residual=residual, segments=segments)

    @staticmethod
    def _norm_modulate(x, norm, shift, scale, split=None):
        """norm(x) * (1 + scale) + shift through the shared segmented modulation, x (batch, tokens, dim) borrowed, shift / scale (rows,
        dim). Without split one row per batch item; with split (zero_cond_t) the tokens before it (the generated image) take the rows of
        the real timestep, the others (conditioning images) those of timestep 0, rows batch + b. Returns the modulated tensor and its
        segments (start, stop, row) over the flattened tokens."""
        batch, tokens = x.shape[:2]
        regions = ((0, tokens, 0),) if split is None else ((0, split, 0), (split, tokens, batch))
        segments = [(b * tokens + start, b * tokens + stop, b + offset) for b in range(batch) for start, stop, offset in regions]
        return ops.norm_modulate([x.view(-1, x.shape[-1])], norm, shift, scale, segments).view(x.shape), segments

    @staticmethod
    def _norm_modulate_rows(x, norm, shift, scale, split=None):
        """_norm_modulate as an mmgp RowSource made by ranges of rows (extreme low VRAM: the attention makes it again for each head
        group, the FFN chunk by chunk), never whole beside x; with its segments."""
        batch, tokens = x.shape[:2]
        regions = ((0, tokens, 0),) if split is None else ((0, split, 0), (split, tokens, batch))
        segments = [(b * tokens + start, b * tokens + stop, b + offset) for b in range(batch) for start, stop, offset in regions]
        return ops.norm_modulate_rows(x.view(-1, x.shape[-1]), norm, shift, scale, segments, x.shape), segments

    def _extreme_forward(self, hidden_states, encoder_hidden_states, img_mod_params, txt_mod_params, image_rotary_emb, joint_attention_kwargs, modulate_index):
        """forward in extreme low VRAM (attention_kit.extreme_low_vram()): the image stream's attention and FFN inputs given by ranges
        of rows (never whole beside the residual), its attention output projected and gated into hidden_states by the kit."""
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = img_mod_params.chunk(6, dim=-1)
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = txt_mod_params.chunk(6, dim=-1)
        img_rows, img_segments = self._norm_modulate_rows(hidden_states, self.img_norm1, img_shift1, img_scale1, modulate_index)
        txt_normed, _ = self._norm_modulate(encoder_hidden_states, self.txt_norm1, txt_shift1, txt_scale1)
        hidden_states_list = [img_rows, txt_normed]
        img_rows = txt_normed = None
        _, txt_attn_output = self.attn.processor(self.attn, hidden_states_list=hidden_states_list, image_rotary_emb=image_rotary_emb,
                                                 residual=(hidden_states, img_gate1, img_segments), **(joint_attention_kwargs or {}))
        output_list, txt_attn_output = [txt_attn_output], None
        ops.gated_residual_(encoder_hidden_states, output_list, txt_gate1.unsqueeze(1), fused=True)
        img_rows, segments = self._norm_modulate_rows(hidden_states, self.img_norm2, img_shift2, img_scale2, modulate_index)
        self._apply_ffn_chunked(img_rows, self.img_mlp, hidden_states, [(start, stop, img_gate2[row]) for start, stop, row in segments])
        img_rows = None
        txt_normed2, _ = self._norm_modulate(encoder_hidden_states, self.txt_norm2, txt_shift2, txt_scale2)
        self._apply_ffn_chunked(txt_normed2, self.txt_mlp, encoder_hidden_states, ops.gate_segments(txt_gate2.unsqueeze(1), *encoder_hidden_states.shape[:2]))
        return encoder_hidden_states, hidden_states

    @staticmethod
    def _gated_residual_segments_(residual, x_list, segments):
        """residual += gate * x per segment (start, stop, gate) of the flattened tokens; x handed off in x_list."""
        x, rows = ops.take(x_list).flatten(0, 1), residual.view(-1, residual.shape[-1])
        for start, stop, gate in segments:
            ops.gated_residual_(rows[start:stop], [x[start:stop]], gate, fused=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        if self.zero_cond_t:
            temb_txt = temb.chunk(2, dim=0)[0]
        else:
            temb_txt = temb
        txt_mod_params = self.txt_mod(temb_txt)

        if hidden_states.dtype != torch.float16 and attention_kit.extreme_low_vram():
            return self._extreme_forward(hidden_states, encoder_hidden_states, img_mod_params, txt_mod_params, image_rotary_emb, joint_attention_kwargs, modulate_index)
        if modulate_index is None:
            # Get modulation parameters for both streams and split into shift, scale, gate
            img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = img_mod_params.chunk(6, dim=-1)
            txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = txt_mod_params.chunk(6, dim=-1)

            # Process image stream - norm1 + modulation (in-place)
            img_normed, _ = self._norm_modulate(hidden_states, self.img_norm1, img_shift1, img_scale1)
            img_gate1 = img_gate1.unsqueeze(1)

            # Process text stream - norm1 + modulation (in-place)
            txt_normed, _ = self._norm_modulate(encoder_hidden_states, self.txt_norm1, txt_shift1, txt_scale1)
            txt_gate1 = txt_gate1.unsqueeze(1)

            hidden_states_list = [img_normed, txt_normed]
            del img_normed, txt_normed
            joint_attention_kwargs = joint_attention_kwargs or {}
            img_attn_output, txt_attn_output = self.attn.processor(
                self.attn,
                hidden_states_list=hidden_states_list,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

            # Apply attention gates and add residual in-place
            output_list, img_attn_output = [img_attn_output], None
            ops.gated_residual_(hidden_states, output_list, img_gate1, fused=True)
            output_list, txt_attn_output = [txt_attn_output], None
            ops.gated_residual_(encoder_hidden_states, output_list, txt_gate1, fused=True)

            img_normed2, _ = self._norm_modulate(hidden_states, self.img_norm2, img_shift2, img_scale2)
            self._apply_ffn_chunked(img_normed2, self.img_mlp, hidden_states, ops.gate_segments(img_gate2.unsqueeze(1), *hidden_states.shape[:2]))
            img_normed2 = None

            txt_normed2, _ = self._norm_modulate(encoder_hidden_states, self.txt_norm2, txt_shift2, txt_scale2)
            self._apply_ffn_chunked(txt_normed2, self.txt_mlp, encoder_hidden_states, ops.gate_segments(txt_gate2.unsqueeze(1), *encoder_hidden_states.shape[:2]))
            txt_normed2 = None
        else:
            img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = img_mod_params.chunk(6, dim=-1)
            txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = txt_mod_params.chunk(6, dim=-1)

            img_modulated, img_segments1 = self._norm_modulate(hidden_states, self.img_norm1, img_shift1, img_scale1, modulate_index)
            txt_modulated, _ = self._norm_modulate(encoder_hidden_states, self.txt_norm1, txt_shift1, txt_scale1)

            hidden_states_list = [img_modulated, txt_modulated]
            del img_modulated, txt_modulated
            joint_attention_kwargs = joint_attention_kwargs or {}
            img_attn_output, txt_attn_output = self.attn.processor(
                self.attn,
                hidden_states_list=hidden_states_list,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

            output_list, img_attn_output = [img_attn_output], None
            self._gated_residual_segments_(hidden_states, output_list, [(start, stop, img_gate1[row]) for start, stop, row in img_segments1])
            output_list, txt_attn_output = [txt_attn_output], None
            ops.gated_residual_(encoder_hidden_states, output_list, txt_gate1.unsqueeze(1), fused=True)

            img_modulated2, segments = self._norm_modulate(hidden_states, self.img_norm2, img_shift2, img_scale2, modulate_index)
            self._apply_ffn_chunked(img_modulated2, self.img_mlp, hidden_states, [(start, stop, img_gate2[row]) for start, stop, row in segments])
            img_modulated2 = None

            txt_normed2, _ = self._norm_modulate(encoder_hidden_states, self.txt_norm2, txt_shift2, txt_scale2)
            self._apply_ffn_chunked(txt_normed2, self.txt_mlp, encoder_hidden_states, ops.gate_segments(txt_gate2.unsqueeze(1), *encoder_hidden_states.shape[:2]))
            txt_normed2 = None

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(nn.Module): 
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]


    def preprocess_loras(self, model_type, sd):
        from shared.utils.lora_mapping import convert_lora_keys
        from .qwen_main import _QWEN_FUSED_SPLIT_MAP
        return convert_lora_keys(sd, dict(self.named_modules()), fused_split_map=_QWEN_FUSED_SPLIT_MAP, compose_lokr=True)

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds
        if use_layer3d_rope:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        else:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, use_additional_t_cond=use_additional_t_cond
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t
        self.use_additional_t_cond = use_additional_t_cond

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states_list  = None,
        encoder_hidden_states_mask_list = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens_list  = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback= None,
        pipeline =None,
        additional_t_cond: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        modulate_index = None
        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            if additional_t_cond is not None:
                additional_t_cond = torch.cat([additional_t_cond, additional_t_cond * 0], dim=0)
            if img_shapes is None:
                raise ValueError("img_shapes must be provided when zero_cond_t is enabled.")
            base_shapes = img_shapes
            if isinstance(img_shapes, list) and len(img_shapes) > 0 and isinstance(img_shapes[0], list):
                base_shapes = img_shapes[0]
            # tokens of the generated image, modulated with the real timestep; the conditioning images after them take timestep 0
            modulate_index = math.prod(base_shapes[0])
        hidden_states_list = [hidden_states if i == 0 else hidden_states.clone() for i, _ in enumerate(encoder_hidden_states_list)]
        
        new_encoder_hidden_states_list = []
        for encoder_hidden_states in encoder_hidden_states_list:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)
            new_encoder_hidden_states_list.append(encoder_hidden_states)
        encoder_hidden_states_list = new_encoder_hidden_states_list
        new_encoder_hidden_states_list = encoder_hidden_states = None
        
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, additional_t_cond)
        )

        image_rotary_emb_list = [ self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device) for txt_seq_lens in txt_seq_lens_list] 

        hidden_states = None

        # Smart Inputs Chunker Auto: the extreme low VRAM mode where the chunked attention would not fit (formats probed on the first block)
        attn, first = self.transformer_blocks[0].attn, hidden_states_list[0]
        rows = first.shape[0] * (first.shape[1] + encoder_hidden_states_list[0].shape[1])
        extreme = attention_kit.plan_extreme(attn, first.device, first.dtype, rows, first.shape[-1], attn.heads, attn.heads, attn.to_q.out_features // attn.heads,
                                             (attn.to_q, attn.to_k, attn.to_v), attn.to_out[0])
        first = None
        with extreme, cuda_graphs.scope():
            for index_block, block in enumerate(self.transformer_blocks):
                if callback != None:
                    callback(-1, None, False, True)
                if pipeline._interrupt:
                    return [None] * len(hidden_states_list)
                for hidden_states, encoder_hidden_states, encoder_hidden_states_mask, image_rotary_emb in zip(hidden_states_list, encoder_hidden_states_list, encoder_hidden_states_mask_list, image_rotary_emb_list):
                    encoder_hidden_states[...], hidden_states[...] = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                        modulate_index=modulate_index,
                    )

        # Use only the image part (hidden_states) from the dual-stream blocks
        temb_out = temb
        if self.zero_cond_t:
            temb_out = temb.chunk(2, dim=0)[0]
        output_list = []
        for i in range(len(hidden_states_list)):
            scale, shift = self.norm_out.linear(self.norm_out.silu(temb_out).to(hidden_states_list[i].dtype)).chunk(2, dim=1)
            normed_list, hidden_states_list[i] = [hidden_states_list[i]], None
            hidden_states = ops.norm_scale_shift(normed_list, self.norm_out.norm, scale[:, None], shift[:, None])
            output_list.append(self.proj_out(hidden_states))

        return output_list
