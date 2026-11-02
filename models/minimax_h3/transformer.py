# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from shared.utils.media_control import inference_checkpoint
# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
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
#
# Modified for WanGP: raw-checkpoint names, MMGP QKV splitting/offload, pruned
# AdaLN curves, shared attention backends, chunked FFNs, and early tensor release.

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from shared import attention_kit
from shared.attention import pay_attention

from .interrupt import GenerationInterrupted
from shared.kernels import ops as kernels
from shared.utils.lora_mapping import convert_lora_keys
from .pdd import MiniMaxH3ParallelHead
from .sol_attention import MiniMaxH3SolAttention
from mmgp import cuda_graphs
from .components.packing import (
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PreparedReference,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)


VISUAL_COND_TIMESTEP = MINIMAX_H3_KEYFRAME_NOISE_AUG
AUDIO_COND_TIMESTEP = 1.0
CONTROL_ROW_CHUNK = 4096
VSA_HEAD_SPLIT_LEVEL = 2  # VSA attends by groups of 7 heads: the group's projections and tiled copies never hold the whole sequence for all heads


def patchify_video(latent, patch_size=(1, 2, 2)):
    return patchify_video_latents(latent, patch_size)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    return unpatchify_video_tokens(rows, t, h * patch_size[1], w * patch_size[2], c, patch_size)


def _keyframe_timesteps(keyframes, rows_per_frame):
    """Condition-row spans of keyframes with their own noise level (softened end images / injected frames); keyframe rows come first."""
    spans, offset = [], 0
    for keyframe in keyframes or ():
        rows = keyframe["latent_frame_count"] * rows_per_frame
        if "timestep" in keyframe:
            spans.append((offset, offset + rows, keyframe["timestep"]))
        offset += rows
    return spans


def _grouped_video_timestep_rows(timestep, timestep_indices, video_start, target_video_rows, fixed_rows, active):
    current_row = int(timestep_indices[video_start])
    if not active or not fixed_rows:
        return timestep, timestep_indices, current_row, [(video_start, video_start + target_video_rows,
                                                          current_row * 3 + MINIMAX_H3_VIDEO_TAG)]
    timestep_count = timestep.shape[0]
    timestep, remap = torch.unique(torch.cat((timestep, timestep.new_tensor([VISUAL_COND_TIMESTEP]))), sorted=True, return_inverse=True)
    timestep_indices = remap[:timestep_count][timestep_indices]
    current_row, fixed_row = int(timestep_indices[video_start]), int(remap[-1])
    head_rows = [(0, fixed_rows, fixed_row)]
    block_rows = [(video_start, video_start + fixed_rows, fixed_row * 3 + MINIMAX_H3_VIDEO_TAG)]
    if fixed_rows < target_video_rows:
        head_rows.append((fixed_rows, target_video_rows, current_row))
        block_rows.append((video_start + fixed_rows, video_start + target_video_rows,
                           current_row * 3 + MINIMAX_H3_VIDEO_TAG))
    return timestep, timestep_indices, head_rows, block_rows


def _frame_video_timestep_rows(timestep, timestep_indices, sigma_video, target_video_rows):
    # Embed one timestep per latent frame and broadcast it over that frame's spatial rows.
    timestep_count = timestep.shape[0]
    frame_timesteps = 1.0 - sigma_video.to(device=timestep.device, dtype=torch.float32).flatten()
    timestep, remap = torch.unique(torch.cat((timestep, frame_timesteps)), sorted=True, return_inverse=True)
    timestep_indices = remap[:timestep_count][timestep_indices]
    frame_rows = remap[-frame_timesteps.numel():]
    rows_per_frame = target_video_rows // frame_timesteps.numel()
    head_rows = [(index * rows_per_frame, (index + 1) * rows_per_frame, int(row)) for index, row in enumerate(frame_rows)]
    return timestep, timestep_indices, head_rows, frame_rows


def pack_audio(latent):
    return latent[0].permute(1, 2, 0).reshape(-1, latent.shape[1]).contiguous()


def unpack_audio(rows, ch=2):
    return unpack_audio_tokens(rows, rows.shape[0] // ch).permute(1, 0, 2).unsqueeze(0)


def get_linear_split_map(inner_size, num_attention_heads=56, attention_head_dim=128, qkv_layout="interleaved"):
    # interleaved rows (heads, q/k/v, head_dim): MMGP keeps the fused weight and computes q, k and v from it in place (row views)
    if qkv_layout not in ("interleaved", "grouped"):
        raise ValueError(f"Unsupported MiniMax H3 QKV layout {qkv_layout!r}")
    return {"qkv_proj": {"mapped_modules": ["q_proj", "k_proj", "v_proj"], "split_sizes": [inner_size] * 3, "num_attention_heads": num_attention_heads,
                         "attention_head_dim": attention_head_dim, "interleaved": qkv_layout == "interleaved"}}


def _take(x_list):
    x = x_list[0]
    x_list.clear()
    return x


def _to_dtype(x_list, dtype):
    x = _take(x_list)
    return x if x.dtype == dtype else x.to(dtype)


def _rope_table(angles, dtype):
    return torch.stack((angles.cos(), angles.sin()), dim=-1).unsqueeze(0).unsqueeze(2).to(dtype)


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out, dtype=None, device=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True, dtype=dtype, device=device)
        self.proj_out = nn.Linear(hidden, out, bias=True, dtype=dtype, device=device)

    def forward(self, timestep):
        half = self.freq_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
        angles = timestep.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.proj_out(F.silu(self.proj_in(embedding)))


class RotaryEmbedding(nn.Module):
    def __init__(self, freq_dim, theta=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32, device=device) / (2 * freq_dim)))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, position_ids, device):
        positions = position_ids.to(device=device, dtype=torch.float32)
        frequencies = positions.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        return torch.cat(frequencies.unbind(dim=1), dim=-1)


class MLP(nn.Module):
    def __init__(self, hidden, ffn, dtype=None, device=None):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 2 * ffn, bias=False, dtype=dtype, device=device)
        self.fc2 = nn.Linear(ffn, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x_list, residual=None, gate=None, segments=None):
        """fc2(swiglu(fc1(x))) by chunks (kernels.mlp), written over x; with a residual, added to it per segment (start, stop, row)
        scaled by gate[row] and written there."""
        if residual is None:
            return kernels.mlp(x_list, self.fc1, self.fc2, "swiglu")
        return kernels.mlp(x_list, self.fc1, self.fc2, "swiglu", residual=residual, segments=[(start, stop, gate[row]) for start, stop, row in segments])


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, sol_attention=None, vdn=False, vsa=False, dtype=None, device=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.sol_attention = sol_attention
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, 3 * inner, bias=False, dtype=dtype, device=device)
        self.qkv_interleaved = False  # rows of an unsplit qkv_proj (heads, q/k/v, head_dim) rather than (q/k/v, heads, head_dim)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=dtype, device=device)
        if vsa:  # FastH3's trained VSA compression gate: one value per token and head channel
            self.gate_proj = nn.Linear(hidden, inner, bias=False, dtype=dtype, device=device)
        if vdn:
            from .vdn_attention import VDNHybridAttention
            self.vdn = VDNHybridAttention(hidden, heads, head_dim, dtype=dtype, device=device)

    def _norm_rope_(self, query, key, rope, packed_rope, allow_fusion):
        kernels.split_half_norm_rope_(query, key, rope, packed_rope, self.q_norm, self.k_norm, allow_fusion=allow_fusion)

    def _out_projection(self, attention_list, seq_len, residual, gate, segments):
        output = _take(attention_list).reshape(seq_len, -1)
        if residual is not None and gate is None:
            output_list, output = [output], None
            return kernels.linear_residual_(self.out_proj, output_list, residual)
        if residual is not None:
            output_list, output = [output], None
            return kernels.gated_linear(self.out_proj, output_list, residual, gate, segments)
        return self.out_proj(output)

    def forward(self, x_list, rope=None, transformer_options=None, residual=None, gate=None, segments=None, input_again=None):
        x = _take(x_list)
        vdn_input = x if hasattr(self, "vdn") else None
        seq_len = x.shape[0]
        use_sol = not hasattr(self, "vdn") and self.sol_attention is not None and self.sol_attention.use_for_layer(seq_len)
        use_veda = hasattr(self, "veda") and self.veda.policy.enabled
        use_vsa = hasattr(self, "vsa") and self.vsa.policy.enabled
        packed_rope = None
        if isinstance(rope, tuple):
            rope, packed_rope = rope
        allow_fusion = not use_sol and not hasattr(self, "vdn")
        split_qkv = hasattr(self, "q_proj")
        if split_qkv and not use_sol and not hasattr(self, "vdn"):
            x_list, x = [x], None
            # VSA: q, k, v and the gate are projected by head groups from one prepared input, so the tiled copies stay bounded
            attention = attention_kit.qkv_attention(x_list, self.q_proj, self.k_proj, self.v_proj, self.heads, self.head_dim,
                                                    lambda query, key, group: self._norm_rope_(query, key, rope, packed_rope, allow_fusion),
                                                    fused_quant=(lambda query, key, group, core, settings:
                                                                 kernels.split_half_norm_rope_quant_(query, key, rope, self.q_norm, self.k_norm, core, settings)) if allow_fusion else None,
                                                    attention_fn=self.vsa if use_vsa else self.veda if use_veda else None,
                                                    extra_proj=self.gate_proj if use_vsa else None, head_split_level=VSA_HEAD_SPLIT_LEVEL if use_vsa else None)
            attention_list, attention = [attention], None
            return self._out_projection(attention_list, seq_len, residual, gate, segments)
        vsa_gate = [self.gate_proj(x).view(1, seq_len, self.heads, self.head_dim)] if use_vsa else None
        if split_qkv and hasattr(self, "vdn"):  # VDN projects q, k and v when it needs them: its linear branch reads them raw
            x_list, x, vdn_input = [x], None, None
            return self.vdn(x_list, (self.q_proj, self.k_proj, self.v_proj), lambda query, key: self._norm_rope_(query, key, rope, packed_rope, False),
                            self.out_proj, input_again)
        if split_qkv and use_sol and self.sol_attention.staged(x, (self.q_proj, self.k_proj, self.v_proj)):
            x_list, x = [x], None
            attention = self.sol_attention.attention(x_list, (self.q_proj, self.k_proj, self.v_proj), (self.q_norm, self.k_norm), rope,
                                                     self.q_norm.eps, self.heads, self.head_dim)
            attention_list, attention = [attention], None
            return self._out_projection(attention_list, seq_len, residual, gate, segments)
        if split_qkv:
            projections = (self.q_proj, self.k_proj, self.v_proj)
            x_list, x = [x], None
            query, key, value = kernels.project_many(projections, x_list)
            x = None  # only the projections read it: released before the norms, at the peak of the block
            query = query.view(1, seq_len, self.heads, self.head_dim)
            key = key.view(1, seq_len, self.heads, self.head_dim)
            value = value.view(1, seq_len, self.heads, self.head_dim)
        else:
            qkv = self.qkv_proj(x)
            x = None
            if self.qkv_interleaved:
                query, key, value = qkv.view(seq_len, self.heads, 3, self.head_dim).unbind(2)
            else:
                query, key, value = qkv.split(self.heads * self.head_dim, dim=-1)
                query = query.view(seq_len, self.heads, self.head_dim)
                key = key.view(seq_len, self.heads, self.head_dim)
                value = value.view(seq_len, self.heads, self.head_dim)
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
            if not use_sol:
                value = value.clone()
            del qkv
        if hasattr(self, "vdn"):  # fused q/k/v projection: k gets storage of its own, normalized in place
            x_handoff, qkv = [vdn_input], [query, key.contiguous(), value]
            vdn_input = query = key = value = None
            return self.vdn(x_handoff, qkv, lambda query, key: self._norm_rope_(query, key, rope, packed_rope, False), self.out_proj)
        if use_sol:
            kernels.sol_rms_rope_(query, self.q_norm.weight, rope, self.q_norm.eps)
            kernels.sol_rms_rope_(key, self.k_norm.weight, rope, self.k_norm.eps)
        else:
            self._norm_rope_(query, key, rope, packed_rope, allow_fusion)
        qkv_list = [query, key, value]
        del query, key, value
        if use_vsa:
            attention = self.vsa(qkv_list, attention_kit.HeadGroup(range(self.heads), range(self.heads), None), vsa_gate)
            vsa_gate = None
        elif use_veda:
            attention = self.veda(qkv_list, attention_kit.HeadGroup(range(self.heads), range(self.heads), None))
        else:
            attention = pay_attention(qkv_list, recycle_q=True) if self.sol_attention is None else self.sol_attention(qkv_list, use_sol)
        attention_list, attention = [attention], None
        return self._out_projection(attention_list, seq_len, residual, gate, segments)


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, dtype=None, device=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device)

    def forward(self, x_list):
        hidden = _take(x_list)
        hidden = self.attn([self.norm1(hidden)], residual=hidden)
        return kernels.mlp([self.norm2(hidden)], self.mlp.fc1, self.mlp.fc2, "swiglu", residual=hidden)


class TokenRefiner(nn.Module):
    def __init__(self, layers, hidden, heads, head_dim, ffn, eps, qk_eps, out_eps, dtype=None, device=None):
        super().__init__()
        self.blocks = nn.ModuleList([RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps,
                                                   dtype=dtype, device=device) for _ in range(layers)])
        self.final_norm = nn.RMSNorm(hidden, eps=out_eps, dtype=dtype, device=device)
        self._interrupt = False

    def forward(self, x_list):
        hidden = _take(x_list)
        for block in self.blocks:
            if self._interrupt:
                raise GenerationInterrupted
            hidden = block([hidden])
        return self.final_norm(hidden)


class AdalnProj(nn.Module):
    def __init__(self, time_dim, hidden, expand, modalities=3, apply_silu=True, dtype=None, device=None):
        super().__init__()
        self.hidden = hidden
        self.expand = expand
        self.modalities = modalities
        self.apply_silu = apply_silu
        self.linear = nn.Linear(time_dim, expand * modalities * hidden, bias=True, dtype=dtype, device=device)

    def forward(self, temb):
        if self.apply_silu:
            temb = F.silu(temb)
        output = self.linear(temb.to(self.linear.weight.dtype)).view(-1, self.expand * self.hidden)
        return output.chunk(self.expand, dim=-1)


def _norm_modulate(norm, hidden, shift_scale, segments):
    # Hidden is the borrowed residual; only the normalized result is disposable.
    shift, scale = shift_scale
    shift_scale.clear()
    return kernels.norm_modulate([hidden], norm, shift, scale, segments)


def _gated_residual(hidden_list, gate_list, branch_list, segments):
    hidden = _take(hidden_list)
    gate, branch = gate_list[0].to(hidden.dtype), _take(branch_list)
    gate_list.clear()
    for start, stop, row in segments:
        kernels.gated_residual_(hidden[start:stop], [branch[start:stop]], gate[row])
    return hidden


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, time_dim, eps, qk_eps, apply_silu=True,
                 adaln_dtype=None, sol_attention=None, vdn=False, vsa=False, dtype=None, device=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, sol_attention=sol_attention, vdn=vdn, vsa=vsa, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(time_dim, hidden, 6, apply_silu=apply_silu,
                                    dtype=adaln_dtype or dtype, device=device)

    @staticmethod
    def _gated_branch(hidden, gate, branch, segments, signature_stride, signature=None):
        for start, stop, row in segments:
            kernels.gated_residual_(hidden[start:stop], [branch[start:stop]], gate[row])
        # Sample the gated branch (gated in place) before it is released; never materialize a full block residual copy.
        sampled = branch.reshape(-1)[::signature_stride]
        if signature is None:
            signature = sampled.clone()
        else:
            signature.add_(sampled)
        return hidden, signature

    def forward(self, x_list, temb, segments, rope, residual_signature_elements=0):
        residual_list = [_take(x_list)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        h_list = [_norm_modulate(self.norm1, residual_list[0], [shift_msa, scale_msa], segments)]
        if (not residual_signature_elements and not hasattr(self.attn, 'vdn')
                and kernels.can_linear(self.attn.out_proj, residual_list[0])
                and kernels.can_linear(self.mlp.fc2, residual_list[0])):
            hidden = self.attn(h_list, rope=rope, residual=residual_list[0], gate=gate_msa, segments=segments)
            h_list = [_norm_modulate(self.norm2, hidden, [shift_mlp, scale_mlp], segments)]
            return self.mlp(h_list, residual=hidden, gate=gate_mlp, segments=segments)
        if not residual_signature_elements:  # VDN computes its input again rather than keeping it (input_again)
            input_again = (lambda: _norm_modulate(self.norm1, residual_list[0], [shift_msa, scale_msa], segments)) if hasattr(self.attn, 'vdn') else None
            residual_list = [_gated_residual(residual_list, [gate_msa], [self.attn(h_list, rope=rope, input_again=input_again)], segments)]
            h_list = [_norm_modulate(self.norm2, residual_list[0], [shift_mlp, scale_mlp], segments)]
            return _gated_residual(residual_list, [gate_mlp], [self.mlp(h_list)], segments)

        hidden = _take(residual_list)
        signature_stride = max(1, math.ceil(hidden.numel() / residual_signature_elements))
        input_again = (lambda: _norm_modulate(self.norm1, hidden, [shift_msa, scale_msa], segments)) if hasattr(self.attn, 'vdn') else None
        branch = self.attn(h_list, rope=rope, input_again=input_again)
        hidden, signature = self._gated_branch(hidden, gate_msa.to(hidden.dtype), branch, segments, signature_stride)
        del branch
        h_list = [_norm_modulate(self.norm2, hidden, [shift_mlp, scale_mlp], segments)]
        branch = self.mlp(h_list)
        hidden, signature = self._gated_branch(hidden, gate_mlp.to(hidden.dtype), branch, segments, signature_stride, signature)
        return hidden, signature


class ControlBlock(DiTBlock):
    """VideoX-Fun ControlNet-Union block attached to the main block it feeds (only one control stream is kept alive)."""

    def __init__(self, hidden, *args, first=False, dtype=None, device=None, **kwargs):
        super().__init__(hidden, *args, dtype=dtype, device=device, **kwargs)
        if first:
            self.before_proj = nn.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)
        self.after_proj = nn.Linear(hidden, hidden, bias=True, dtype=dtype, device=device)


def _chunk_rows(start, stop, chunk=CONTROL_ROW_CHUNK):
    return ((offset, min(stop, offset + chunk)) for offset in range(start, stop, chunk))


class InputPreparation(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Reference the hooked projections without registering this stage on the model:
        # checkpoint paths, LoRA names and MMGP weight groups stay unchanged.
        self.video_patch_proj = model.video_patch_proj
        self.audio_patch_proj = model.audio_patch_proj
        self.patch_size = model.patch_size
        self.hidden_size = model.hidden_size

    def forward(self, video_x, audio_x, text_embeds, text_indices, video_indices,
                audio_indices, target_video_order, cond_video, cond_audio, sequence_length, dtype):
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        if target_video_order is not None:
            video_rows = video_rows.index_select(0, target_video_order)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        if cond_video is not None:
            video_rows = torch.cat((cond_video, video_rows))
        if cond_audio is not None:
            audio_rows = torch.cat((cond_audio, audio_rows))
        video_embeds = self.video_patch_proj(video_rows).to(dtype)
        del video_rows
        audio_embeds = self.audio_patch_proj(audio_rows).to(dtype)
        del audio_rows
        hidden = torch.empty((sequence_length, self.hidden_size), device=video_x.device, dtype=dtype)
        hidden.index_copy_(0, text_indices, text_embeds)
        hidden.index_copy_(0, video_indices, video_embeds)
        hidden.index_copy_(0, audio_indices, audio_embeds)
        return hidden


class FinalLayer(nn.Module):
    def __init__(self, hidden, time_dim, video_dim, audio_dim, eps, apply_silu=True,
                 adaln_dtype=None, pdd_num_steps=None, dtype=None, device=None):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(time_dim, hidden, 2, modalities=1, apply_silu=apply_silu,
                                    dtype=adaln_dtype or dtype, device=device)
        if pdd_num_steps is None:
            self.video_out = nn.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)
            self.audio_out = nn.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)
        else:
            self.video_out = MiniMaxH3ParallelHead(hidden, video_dim, pdd_num_steps, device=device)
            self.audio_out = MiniMaxH3ParallelHead(hidden, audio_dim, pdd_num_steps, device=device)

    def _head(self, h_list, shift, scale, row, output):
        rows = row if isinstance(row, list) else [(0, h_list[0].shape[0], row)]
        value = kernels.norm_modulate(h_list, self.norm, shift, scale, rows, dtype=scale.dtype).to(torch.float32)
        return output(value)

    def forward(self, x_list, temb, video_segment, audio_segment):
        # PDD heads read a sampling plan changed by the scheduler outside this call.
        with cuda_graphs.scope():
            return cuda_graphs.run(self, x_list, temb, video_segment, audio_segment, method="_forward",
                                   enabled=not isinstance(self.video_out, MiniMaxH3ParallelHead))

    def _forward(self, x_list, temb, video_segment, audio_segment):
        hidden = _take(x_list)
        a0, a1, arow = audio_segment
        v0, v1, vrow = video_segment
        shift, scale = self.adaln_proj(temb)
        h_list = [hidden]
        del hidden
        audio_list = [h_list[0][a0:a1]]
        video_list = [_take(h_list)[v0:v1]]
        audio = self._head(audio_list, shift, scale, arow, self.audio_out)
        video = self._head(video_list, shift, scale, vrow, self.video_out)
        return video, audio


def _prepared_references(refs):
    prepared = []
    for ref in refs or ():
        kind = ref["kind"]
        if kind == "video_audio":
            kind = "video"
        prepared.append(MiniMaxH3PreparedReference(
            kind=kind,
            has_audio=ref.get("ref_audio_t", 0) > 0,
            num_latent_frames=ref.get("latent_t", 1),
            latent_height=ref.get("latent_h", 0),
            latent_width=ref.get("latent_w", 0),
            num_audio_latents=ref.get("ref_audio_t", 0),
        ))
    return prepared


class MiniMaxH3Model(nn.Module):
    def preprocess_loras(self, model_type, state_dict):
        from .lora_tables import pop_source_tag

        source_package = pop_source_tag(state_dict)
        attention = self.blocks[0].attn
        split_map = get_linear_split_map(self.attention_inner_size, attention.heads, attention.head_dim)
        modules = dict(self.named_modules())
        state_dict = convert_lora_keys(state_dict, modules,
                                      fused_split_map=split_map, split_linear_modules_map=split_map)
        state_dict = {key.removeprefix("diffusion_model."): value for key, value in state_dict.items()}
        # ComfyUI bundles per-layer backend preferences as uint8 JSON tensors.
        # Keep WanGP's attention selection and leave other unknown keys for MMGP to reject.
        config_suffix = ".comfy_attention.config"
        ignored_configs = [key for key in state_dict if key.endswith(config_suffix)
                           and isinstance(modules.get(key[:-len(config_suffix)]), Attention)]
        for key in ignored_configs:
            state_dict.pop(key)
        if ignored_configs:
            tqdm.write(f"MiniMax H3 LoRA: ignored {len(ignored_configs)} ComfyUI attention configurations; using WanGP's selected attention backend.")
        diffusers_format = any(key.startswith(("transformer_blocks.", "token_refiner.refiner_blocks.",
                                               "time_embedder.linear_", "audio_proj_in.", "proj_in.",
                                               "context_embedder.", "norm_out.", "audio_proj_out.", "proj_out."))
                               for key in state_dict)
        converted = {}
        for key, value in state_dict.items():
            key = key.replace(".lora_A.turbo.weight", ".lora_A.weight").replace(".lora_B.turbo.weight", ".lora_B.weight")
            if key.startswith("lora_unet_"):
                path, suffix = key[len("lora_unet_"):].split(".", 1)
                key = path.replace("blocks_", "blocks.", 1).replace("_attn_", ".attn.").replace("_mlp_", ".mlp.") + "." + suffix
            if key.startswith("token_refiner_blocks."):
                key = "token_refiner.blocks." + key[len("token_refiner_blocks."):]
            if diffusers_format:
                diffusers_fc1 = ".ff.net.0.proj." in key
                for source, target in (("token_refiner.refiner_blocks.", "token_refiner.blocks."),
                                       ("transformer_blocks.", "blocks."),
                                       ("time_embedder.linear_1.", "time_embedder.proj_in."),
                                       ("time_embedder.linear_2.", "time_embedder.proj_out."),
                                       ("audio_proj_in.", "audio_patch_proj."), ("proj_in.", "video_patch_proj."),
                                       ("context_embedder.", "condition_proj."),
                                       ("norm_out.norm.", "final_layer.norm."),
                                       ("norm_out.linear.", "final_layer.adaln_proj.linear."),
                                       ("audio_proj_out.", "final_layer.audio_out."), ("proj_out.", "final_layer.video_out.")):
                    if key.startswith(source):
                        key = target + key[len(source):]
                        break
                for source, target in ((".attn.norm_q.", ".attn.q_norm."), (".attn.norm_k.", ".attn.k_norm."),
                                       (".attn.orig.to_out.0.", ".attn.out_proj."), (".attn.orig.to_q.", ".attn.q_proj."),
                                       (".attn.orig.to_k.", ".attn.k_proj."), (".attn.orig.to_v.", ".attn.v_proj."),
                                       (".attn.to_out.0.", ".attn.out_proj."), (".attn.to_q.", ".attn.q_proj."),
                                       (".attn.to_k.", ".attn.k_proj."), (".attn.to_v.", ".attn.v_proj."),
                                       (".attn.to_gate_compress.", ".attn.gate_proj."),
                                       (".ff.net.0.proj.", ".mlp.fc1."), (".ff.net.2.", ".mlp.fc2.")):
                    key = key.replace(source, target)
                if diffusers_fc1 and key.endswith((".lora_B.weight", ".lora_B.default.weight", ".lora_up.weight",
                                                    ".lora_up.default.weight", ".lora.B.weight", ".lora.B.default.weight",
                                                    ".lora.up.weight", ".lora.up.default.weight")):
                    value = torch.cat(value.chunk(2, dim=0)[::-1], dim=0).contiguous()
            converted[key] = value
        if self.pdd_num_steps:  # PDD: a base-model head LoRA applies to every head through the base projection
            for wrapper in ("", "diffusion_model.", "transformer."):
                for head in ("final_layer.video_out.", "final_layer.audio_out."):
                    prefix = wrapper + head
                    for key in [key for key in converted if key.startswith(prefix) and not key.startswith(prefix + "linear.")]:
                        converted[prefix + "linear." + key[len(prefix):]] = converted.pop(key)
        from .lora_affine import convert_adaln_loras

        if self.use_adaln_curves:
            ignored = sorted(key for key in converted if key.startswith("time_embedder."))
            for key in ignored:
                converted.pop(key)
            if ignored:
                tqdm.write("MiniMax H3 LoRA: ignored unsupported pruned timestep tensors: " + ", ".join(ignored))

        start = time.perf_counter()
        count, architecture, source_width, target_width = convert_adaln_loras(
            model_type, converted, self.adaln_t_table if self.use_adaln_curves else None,
            self.hybrid_ref2va_blocks, self.ref2va_adaln_t_table if self.hybrid_ref2va_blocks is not None else None,
            source_package=source_package, target_package=self.h3_lora_affine_package, checkpoint_checked=self.h3_lora_map_checked)
        if count:
            source_architecture = source_package.name if source_package is not None else architecture.upper()
            source = f"full AdaLN width {source_width}" if source_width == 2688 else f"{source_architecture} pruned AdaLN width {source_width}"
            target_architecture = "hybrid FL2VA/REF2VA" if self.hybrid_ref2va_blocks is not None else (self.h3_lora_affine_package.name if self.h3_lora_affine_package is not None else architecture.upper())
            target = f"full AdaLN width {target_width}" if target_width == 2688 else f"{target_architecture} pruned AdaLN width {target_width}"
            tqdm.write(f"MiniMax H3 LoRA: converted {count} AdaLN adapters from {source} to {target} in {time.perf_counter() - start:.2f}s")
        if hasattr(self.blocks[0].attn, "q_proj"):
            return converted
        for down_suffix, up_suffix in (("lora_A.weight", "lora_B.weight"), ("lora_down.weight", "lora_up.weight"),
                                       ("lora_A.default.weight", "lora_B.default.weight"),
                                       ("lora_down.default.weight", "lora_up.default.weight"),
                                       ("lora.A.weight", "lora.B.weight"), ("lora.down.weight", "lora.up.weight"),
                                       ("lora.A.default.weight", "lora.B.default.weight"),
                                       ("lora.down.default.weight", "lora.up.default.weight")):
            marker = "q_proj." + down_suffix
            for key in [key for key in converted if key.endswith(marker)]:
                prefix = key[:-len(marker)]
                down, up, scales = [], [], []
                for projection in ("q_proj", "k_proj", "v_proj"):
                    down.append(converted.pop(prefix + projection + "." + down_suffix))
                    up.append(converted.pop(prefix + projection + "." + up_suffix))
                    alpha = converted.pop(prefix + projection + ".alpha", None)
                    scales.append(1.0 if alpha is None else float(alpha) / down[-1].shape[0])
                converted[prefix + "qkv_proj." + down_suffix] = torch.cat(down)
                up = torch.block_diag(*(weight * scale for weight, scale in zip(up, scales)))
                attention = self.get_submodule(prefix.rstrip("."))
                if attention.qkv_interleaved:
                    up = up.reshape(3, attention.heads, attention.head_dim, -1).permute(1, 0, 2, 3).reshape(3 * self.attention_inner_size, -1).contiguous()
                converted[prefix + "qkv_proj." + up_suffix] = up
        return converted

    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5,
                 final_norm_eps=1e-5, sigma_shift_video=12.0, sigma_shift_audio=3.0,
                 adaln_curve_grid=None, adaln_dtype=torch.float32, image_model=None,
                 hybrid_ref2va_blocks=None, pdd_num_steps=None, pdd_block_size=None, vdn=False, vsa=False,
                 control_layers=(), control_in_dim=0, dtype=None, device=None, **kwargs):
        super().__init__()
        self._interrupt = False
        self.cache = None
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.attention_inner_size = num_attention_heads * attention_head_dim
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.use_adaln_curves = adaln_curve_grid is not None
        self.h3_lora_affine_package = None
        self.h3_lora_map_checked = False
        self.hybrid_ref2va_blocks = hybrid_ref2va_blocks
        self.pdd_num_steps = pdd_num_steps
        self.pdd_block_size = pdd_block_size
        video_dim = latents_dim * math.prod(self.patch_size)
        self.video_patch_proj = nn.Linear(video_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=adaln_dtype, device=device))
            if hybrid_ref2va_blocks is not None:
                self.register_buffer("ref2va_adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=adaln_dtype, device=device))
        else:
            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,
                                              dtype=torch.float32, device=device)
        self.rope = RotaryEmbedding(rope_inv_freq_len, rope_theta, device=device)
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                          final_norm_eps, dtype=dtype, device=device)
        self.sol_attention = MiniMaxH3SolAttention()
        curve = {"apply_silu": not self.use_adaln_curves,
                 "adaln_dtype": adaln_dtype if self.use_adaln_curves else dtype}
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                                               time_embed_dim, norm_eps, qk_norm_eps, **curve,
                                               sol_attention=self.sol_attention, vdn=vdn, vsa=vsa,
                                               dtype=dtype, device=device) for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_dim, audio_latents_dim,
                                      final_norm_eps, **curve, pdd_num_steps=pdd_num_steps, dtype=dtype, device=device)
        self.control_layers = tuple(control_layers)
        for index, layer in enumerate(self.control_layers):
            self.blocks[layer].control = ControlBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                                                      time_embed_dim, norm_eps, qk_norm_eps, **curve, first=index == 0,
                                                      sol_attention=self.sol_attention,
                                                      dtype=dtype, device=device)
        fp32_modules = [self.video_patch_proj, self.audio_patch_proj, self.final_layer.video_out, self.final_layer.audio_out]
        if self.control_layers:
            self.control_patch_proj = nn.Linear(control_in_dim * math.prod(self.patch_size), hidden_size, bias=True, dtype=torch.float32, device=device)
            fp32_modules.append(self.control_patch_proj)
        if self.use_adaln_curves:
            adaln_modules = [block.adaln_proj.linear for block in self.blocks] + [self.blocks[layer].control.adaln_proj.linear for layer in self.control_layers]
            for module in (*adaln_modules, self.final_layer.adaln_proj.linear):
                module._lock_dtype = adaln_dtype
        else:
            fp32_modules.extend((self.time_embedder.proj_in, self.time_embedder.proj_out))
        for module in fp32_modules:
            for submodule in module.modules():
                submodule._lock_dtype = torch.float32

    def preprocess_text_embeds(self, text_states):
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        self.token_refiner._interrupt = self._interrupt
        return self.token_refiner([self.condition_proj(text_states[0])]).unsqueeze(0)

    def _check_interrupt(self):
        inference_checkpoint()
        if self._interrupt:
            raise GenerationInterrupted

    def _layout(self, text_tags, latent_t, latent_h, latent_w, audio_t, payload):
        target_spatial_context = payload.get("target_spatial_context")
        signature = (text_tags.numel(), latent_t, latent_h, latent_w, audio_t,
                     payload["fps"], payload.get("audio_time_offset", 0.0),
                     target_spatial_context,
                     payload.get("target_audio_condition_latents", 0),
                     payload.get("target_video_condition_frames", 0),
                     tuple((k["anchor"], k["latent_frame_count"], k.get("frame_index")) for k in payload.get("keyframes") or ()),
                     tuple((k["anchor"], k["latent_frame_count"]) for k in payload.get("audio_keyframes") or ()),
                     tuple((r["kind"], r.get("latent_t"), r.get("latent_h"), r.get("latent_w"), r.get("ref_audio_t"))
                           for r in payload.get("refs") or ()))
        if payload.get("layout_signature") == signature:
            return payload["layout"]
        with torch.device("cpu"):
            video_time_scale = 24.0 / payload["fps"]
            anchors = tuple((keyframe["anchor"], keyframe["latent_frame_count"], keyframe.get("frame_index"))
                            for keyframe in payload.get("keyframes") or ())
            audio_anchors = tuple((keyframe["anchor"], keyframe["latent_frame_count"])
                                  for keyframe in payload.get("audio_keyframes") or ())
            target_audio_condition_latents = payload.get("target_audio_condition_latents", 0)
            target_video_condition_frames = payload.get("target_video_condition_frames", 0)
            if payload.get("refs"):
                layout = build_ref2va_packed_sequence(text_tags, _prepared_references(payload["refs"]), latent_t,
                                                      latent_h, latent_w, audio_t, self.patch_size, video_time_scale,
                                                      keyframe_anchors=anchors, audio_condition_anchors=audio_anchors,
                                                      target_condition_audio_latents=target_audio_condition_latents,
                                                      target_condition_video_frames=target_video_condition_frames,
                                                       target_spatial_context=target_spatial_context, audio_time_offset=payload.get("audio_time_offset", 0.0))
            else:
                layout = build_packed_sequence(text_tags, latent_t, latent_h, latent_w, audio_t, self.patch_size,
                                               anchors, video_time_scale, audio_condition_anchors=audio_anchors,
                                               target_condition_audio_latents=target_audio_condition_latents,
                                               target_condition_video_frames=target_video_condition_frames,
                                               target_spatial_context=target_spatial_context, audio_time_offset=payload.get("audio_time_offset", 0.0))
        payload["layout_signature"], payload["layout"] = signature, layout
        return layout

    def _control_stream(self, hidden, control_rows, video_indices):
        # The control stream is the packed sequence with the control embeddings on the video rows, re-based on the main input.
        if control_rows.shape[0] != video_indices.numel():
            raise ValueError(f"MiniMax H3 control rows ({control_rows.shape[0]}) must match the packed video rows ({video_indices.numel()})")
        stream = hidden.clone()
        for start, stop in _chunk_rows(0, video_indices.numel()):
            rows = control_rows[start:stop].to(device=hidden.device, dtype=torch.float32)
            stream.index_copy_(0, video_indices[start:stop], self.control_patch_proj(rows).to(hidden.dtype))
        before_proj = self.blocks[self.control_layers[0]].control.before_proj
        for start, stop in _chunk_rows(0, stream.shape[0]):
            stream[start:stop] = before_proj(stream[start:stop]).add_(hidden[start:stop])
        return stream

    @staticmethod
    def _add_control_skip(hidden, stream, after_proj, row_ranges, scale):
        # Skips are applied in row chunks so no full-sequence skip tensor is ever materialized.
        for first, last in row_ranges:
            for start, stop in _chunk_rows(first, last):
                kernels.linear_residual_(after_proj, [stream[start:stop]], hidden[start:stop], scale)

    def _time_embedding(self, timesteps, table=None):
        if not self.use_adaln_curves:
            return self.time_embedder(timesteps)
        table = self.adaln_t_table if table is None else table
        position = timesteps.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1).to(table.dtype))

    def _prepare_graph_input(self, video_x, audio_x, context, payload, layout, dtype):
        with cuda_graphs.scope():
            if not cuda_graphs.is_enabled(self):
                return None
            stage = self.__dict__.get("_input_preparation")
            if stage is None:
                stage = self.__dict__["_input_preparation"] = InputPreparation(self)
            text_embeds = context[0]
            if text_embeds.shape[-1] != self.hidden_size:
                text_embeds = self.preprocess_text_embeds(context)[0]
            device = video_x.device
            indices = payload.get("graph_input_indices")
            if indices is None or indices[0] is not layout:
                indices = (layout, *(index.to(device) for index in (layout.text_indices, layout.video_indices, layout.audio_indices)))
                payload["graph_input_indices"] = indices
            optional = [payload.get(name) for name in ("target_video_order", "cond_video_rows", "cond_audio_rows")]
            optional = [None if value is None else value.to(device) for value in optional]
            return cuda_graphs.run(stage, video_x, audio_x, text_embeds, *indices[1:], *optional, layout.sequence_length, dtype, owner=self)

    def forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload, spectrum=None, first_block_cache=None):
        device, dtype = video_x.device, self.dtype or next(self.blocks.parameters()).dtype
        video_dtype, audio_dtype = video_x.dtype, audio_x.dtype
        _, _, latent_t, latent_h, latent_w = video_x.shape
        sigma_video = sigma_video.flatten()
        if sigma_video.numel() not in (1, latent_t):
            raise ValueError(f"MiniMax H3 received {sigma_video.numel()} video sigmas for {latent_t} latent frames")
        if not self.control_layers and sigma_video.numel() > 1 and not bool((sigma_video == sigma_video[0]).all()):
            raise ValueError("MiniMax H3 requires one uniform video sigma")
        # A clean first-frame ControlNet anchor must not set the text/audio prefix's noise level.
        base_video_sigma = sigma_video.max() if self.control_layers else sigma_video[0]
        audio_t = audio_x.shape[-1]
        text_tags = payload["text_token_tags"].view(-1).cpu()
        layout = self._layout(text_tags, latent_t, latent_h, latent_w, audio_t, payload)
        target_video_order = payload.get("target_video_order")
        target_video_inverse_order = payload.get("target_video_inverse_order")
        target_video_fixed_rows = payload.get("target_video_fixed_rows", 0)
        target_video_mask_active = payload.get("target_video_mask_active", False)

        if spectrum is not None and spectrum.forecasting:
            timestep, timestep_indices = build_row_timesteps(
                layout,
                float(1.0 - base_video_sigma),
                float(1.0 - sigma_audio.flatten()[0]),
                max(float(1.0 - base_video_sigma), VISUAL_COND_TIMESTEP),
                AUDIO_COND_TIMESTEP,
                keyframe_timesteps=_keyframe_timesteps(payload.get("keyframes"), latent_h // self.patch_size[1] * (latent_w // self.patch_size[2])),
            )
            timestep, timestep_indices = timestep.to(device), timestep_indices.to(device)
            target_video_rows = latent_t * (latent_h // self.patch_size[1]) * (latent_w // self.patch_size[2])
            target_audio_rows = audio_t * 2
            video_start = layout.sequence_length - target_video_rows
            audio_start = video_start - target_audio_rows
            if target_video_order is None:
                video_row = int(timestep_indices[video_start])
            else:
                timestep, timestep_indices, video_row, _ = _grouped_video_timestep_rows(
                    timestep, timestep_indices, video_start, target_video_rows,
                    target_video_fixed_rows, target_video_mask_active)
            if self.control_layers and target_video_order is None and sigma_video.numel() > 1:
                timestep, timestep_indices, video_row, _ = _frame_video_timestep_rows(timestep, timestep_indices, sigma_video, target_video_rows)
            temb = self._time_embedding(timestep)
            audio_row = int(timestep_indices[audio_start + min(layout.num_target_condition_audio_latents,
                                                               max(audio_t - 1, 0))])
            hidden = spectrum.predict(device, dtype, self._check_interrupt)
            video, audio = self.final_layer([hidden], temb, (target_audio_rows, target_audio_rows + target_video_rows, video_row),
                                            (0, target_audio_rows, audio_row))
            del temb, timestep_indices
            if target_video_inverse_order is not None:
                video = video.index_select(0, target_video_inverse_order)
            video = _to_dtype([video], video_dtype)
            audio = _to_dtype([audio], audio_dtype)
            return (unpatchify_video_tokens(video, latent_t, latent_h, latent_w, self.latents_dim, self.patch_size),
                    unpack_audio(audio))
        hidden = self._prepare_graph_input(video_x, audio_x, context, payload, layout, dtype)
        if hidden is None:
            video_x = video_x.to(torch.float32)
            video_rows = patchify_video(video_x, self.patch_size)
            video_x= None
            if target_video_order is not None:
                video_rows = video_rows.index_select(0, target_video_order)
            audio_x = audio_x.to(torch.float32)
            audio_rows = pack_audio(audio_x)
            audio_x = None
            cond_video, cond_audio = payload.get("cond_video_rows"), payload.get("cond_audio_rows")
            if cond_video is not None:
                video_rows = torch.cat((cond_video.to(device), video_rows))
            if cond_audio is not None:
                audio_rows = torch.cat((cond_audio.to(device), audio_rows))
            video_embeds = self.video_patch_proj(video_rows).to(dtype)
            del video_rows
            audio_embeds = self.audio_patch_proj(audio_rows).to(dtype)
            del audio_rows

            text_embeds = context[0]
            if text_embeds.shape[-1] != self.hidden_size:
                text_embeds = self.preprocess_text_embeds(context)[0]
            hidden = torch.empty(layout.sequence_length, self.hidden_size, dtype=dtype, device=device)
            hidden.index_copy_(0, layout.text_indices.to(device), text_embeds)
            hidden.index_copy_(0, layout.video_indices.to(device), video_embeds)
            hidden.index_copy_(0, layout.audio_indices.to(device), audio_embeds)
            del text_embeds, video_embeds, audio_embeds
        else:
            video_x = audio_x = None

        timestep, timestep_indices = build_row_timesteps(
            layout,
            float(1.0 - base_video_sigma),
            float(1.0 - sigma_audio.flatten()[0]),
            max(float(1.0 - base_video_sigma), VISUAL_COND_TIMESTEP),
            AUDIO_COND_TIMESTEP,
            keyframe_timesteps=_keyframe_timesteps(payload.get("keyframes"), latent_h // self.patch_size[1] * (latent_w // self.patch_size[2])),
        )
        timestep, timestep_indices = timestep.to(device), timestep_indices.to(device)
        target_video_rows = latent_t * (latent_h // self.patch_size[1]) * (latent_w // self.patch_size[2])
        video_start = layout.sequence_length - target_video_rows
        video_head_row = int(timestep_indices[video_start])
        frame_rows = grouped_segments = None
        if target_video_order is not None:
            timestep, timestep_indices, video_head_row, grouped_segments = _grouped_video_timestep_rows(
                timestep, timestep_indices, video_start, target_video_rows,
                target_video_fixed_rows, target_video_mask_active)
        elif sigma_video.numel() == latent_t:
            timestep, timestep_indices, video_head_row, frame_rows = _frame_video_timestep_rows(timestep, timestep_indices, sigma_video, target_video_rows)
        adaln_indices = timestep_indices * 3 + layout.token_tags.to(device).clamp_min(0)
        changes = torch.cat((torch.ones(1, dtype=torch.bool, device=device), adaln_indices[1:] != adaln_indices[:-1],
                             torch.ones(1, dtype=torch.bool, device=device))).nonzero().flatten()
        segments = [(int(changes[index]), int(changes[index + 1]), int(adaln_indices[changes[index]]))
                    for index in range(changes.numel() - 1)]
        if frame_rows is not None:
            segments = [segment for segment in segments if segment[0] < video_start]
            rows_per_frame = target_video_rows // latent_t
            segments.extend((video_start + index * rows_per_frame, video_start + (index + 1) * rows_per_frame,
                             int(row) * 3 + MINIMAX_H3_VIDEO_TAG) for index, row in enumerate(frame_rows))
        elif grouped_segments is not None:
            segments = [segment for segment in segments if segment[0] < video_start]
            segments.extend(grouped_segments)
        temb = self._time_embedding(timestep)
        ref2va_temb = self._time_embedding(timestep, self.ref2va_adaln_t_table) if self.hybrid_ref2va_blocks is not None else None
        rope = payload.get("rope")
        if rope is None:
            positions = layout.position_ids.to(torch.float32)
            frequencies = positions.unsqueeze(-1) * self.rope.inv_freq.detach().cpu().view(1, 1, -1)
            rope = _rope_table(torch.cat(frequencies.unbind(dim=1), dim=-1), dtype).to(device)
            if target_video_order is not None:
                target_rope = rope[:, video_start:].index_select(1, target_video_order)
                rope[:, video_start:].copy_(target_rope)
                del target_rope
            payload["rope"] = rope
            del positions, frequencies
        del adaln_indices, changes
        target_audio_rows = audio_t * 2
        audio_start = video_start - target_audio_rows
        self.sol_attention.begin_forward(layout, device, dtype, payload["attention_sparsity"], target_video_order is not None)
        if hasattr(self, "veda_attention"):
            self.veda_attention.begin_forward(layout, (latent_t, latent_h // self.patch_size[1], latent_w // self.patch_size[2]), payload, device, dtype)
        if hasattr(self, "vsa_attention"):
            self.vsa_attention.begin_forward(layout, (latent_t, latent_h // self.patch_size[1], latent_w // self.patch_size[2]), payload, device, dtype)
        if not hasattr(self.blocks[0].attn, "vdn"):  # Sol layers ignore the Kitchen table; shorter sequences use it
            rope = kernels.prepare_split_half_rope(rope)
        if vdn := getattr(self.blocks[0].attn, "vdn", None):
            for block in self.blocks:
                block.attn.vdn.begin_forward(layout, latent_t, latent_h, latent_w, self.patch_size)

        control_rows = payload.get("control_rows")
        control_list = None
        if control_rows is not None:
            control_scale = float(payload["control_scale"])
            # Keep appended conditions, audio, and anchored target frames free of direct control skips. Restoring anchors
            # between denoising steps cannot undo pose features already shared with adjacent frames inside the transformer.
            # The encoder prefix can also contain supplied image/audio embeddings.
            # Apply prefix skips only to text, while retaining the full sequence in control attention.
            control_row_ranges = [(start, stop) for start, stop, row in segments
                                  if start < text_tags.numel() and row % 3 == MINIMAX_H3_TEXT_TAG]
            rows_per_frame, first_frame = target_video_rows // latent_t, 0
            for frame in payload.get("control_anchor_frames", ()):
                if first_frame < frame:
                    control_row_ranges.append((video_start + first_frame * rows_per_frame, video_start + frame * rows_per_frame))
                first_frame = frame + 1
            if first_frame < latent_t:
                control_row_ranges.append((video_start + first_frame * rows_per_frame, layout.sequence_length))
            control_list = [self._control_stream(hidden, control_rows, layout.video_indices.to(device))]

        def run_block(block_index, h_list, residual_signature_elements=0):
            nonlocal control_list
            block = self.blocks[block_index]
            block_temb = ref2va_temb if ref2va_temb is not None and self.hybrid_ref2va_blocks[0] <= block_index <= self.hybrid_ref2va_blocks[1] else temb
            control = block.control if control_list is not None and hasattr(block, "control") else None
            if control is not None:
                control_list = [control(control_list, block_temb, segments, rope)]
                self._check_interrupt()
            output = block(h_list, block_temb, segments, rope, residual_signature_elements) if residual_signature_elements else block(h_list, block_temb, segments, rope)
            if control is not None:
                self._add_control_skip(output[0] if residual_signature_elements else output, control_list[0], control.after_proj, control_row_ranges, control_scale)
                if block_index == self.control_layers[-1]:
                    control_list = None
            return output

        with cuda_graphs.scope():
            if first_block_cache is None:
                for block_index in range(len(self.blocks)):
                    self._check_interrupt()
                    h_list = [hidden]
                    hidden = None
                    hidden = run_block(block_index, h_list)
            else:
                self._check_interrupt()
                hidden, signature = run_block(0, [hidden], first_block_cache.MAX_SIGNATURE_ELEMENTS)
                if first_block_cache.should_compute(signature):
                    head_output = first_block_cache.capture_head_output(hidden[audio_start:])
                    for block_index in range(1, len(self.blocks)):
                        self._check_interrupt()
                        h_list = [hidden]
                        hidden = None
                        hidden = run_block(block_index, h_list)
                    first_block_cache.store_tail_residual(hidden[audio_start:], head_output)
                else:
                    first_block_cache.apply_tail_residual(hidden[audio_start:])
                control_list = None

        audio_row = int(timestep_indices[audio_start + min(layout.num_target_condition_audio_latents,
                                                           max(audio_t - 1, 0))])
        if spectrum is not None:
            spectrum.observe(hidden[audio_start:], target_audio_rows, self._check_interrupt)
        h_list = [hidden]
        hidden = None
        video, audio = self.final_layer(h_list, temb, (video_start, layout.sequence_length, video_head_row),
                                        (audio_start, video_start, audio_row))
        del temb, ref2va_temb, rope, timestep_indices
        if target_video_inverse_order is not None:
            video = video.index_select(0, target_video_inverse_order)
        video = _to_dtype([video], video_dtype)
        audio = _to_dtype([audio], audio_dtype)
        return (unpatchify_video_tokens(video, latent_t, latent_h, latent_w, self.latents_dim, self.patch_size),
                unpack_audio(audio))


__all__ = ["AUDIO_COND_TIMESTEP", "Attention", "DiTBlock", "MLP", "MiniMaxH3Model", "TimeEmbedder",
           "VISUAL_COND_TIMESTEP", "get_linear_split_map", "pack_audio", "patchify_video", "unpack_audio",
           "unpatchify_video"]
