# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math
from einops import rearrange
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import numpy as np
from typing import Union,Optional
from mmgp import offload
from mmgp.offload import get_cache, clear_caches
from shared import attention_kit
from shared.kernels import ops as kernels
from shared.attention import pay_attention
from torch.backends.cuda import sdp_kernel
from ..multitalk.multitalk_utils import get_attn_map_with_target
from ..animate.motion_encoder import Generator
from ..animate.face_blocks import FaceAdapter, FaceEncoder 
from ..animate.model_animate import after_patch_embedding
from ..scail.model_scail import build_scail_pose_tokens
from ..scail2 import build_scail2_pose_tokens
from ..steadydancer.small_archs import FactorConv3d, PoseRefNetNoBNV3
from ..steadydancer.mobilenetv2_dcd import DYModule
from ..shotplan import inject_shotplan_tokens
from ..animate2 import animate2_attention_block, animate2_cached_attention_block, load_cached as animate2_load_cached, store_kv as animate2_store_kv
from ..taomate import audio_attention as taomate_audio_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def reshape_latent(latent, latent_frames):
    return latent.reshape(latent.shape[0], latent_frames, -1, latent.shape[-1] )

def restore_latent_shape(latent):
    return latent.reshape(latent.shape[0], -1, latent.shape[-1] )

def frame_modulation(modulation, e):
    """modulation (1, n, dim) + e (frames, n or 1, dim) as n contiguous (1, frames, dim) rows: per latent frame shifts, scales and gates
    for modulate_frames, gated_frames_, the output projection's gate and kernels.gate_segments; the same sums as before."""
    return (modulation + e).transpose(0, 1).contiguous().chunk(modulation.shape[1])

def frame_segments(batch, tokens, frames):
    """(start, stop, frame) row ranges of the flattened (batch * tokens) rows, one per latent frame: segments of kernels.norm_modulate
    and kernels.gated_linear for (frames, dim) tables."""
    span = tokens // frames
    return [(b * tokens + f * span, b * tokens + (f + 1) * span, f) for b in range(batch) for f in range(frames)]

def modulate_frames(x, norm, shift, scale, dtype=None):
    """norm(x) * (1 + scale) + shift per latent frame (shift, scale: (1, frames, dim) rows of frame_modulation), x (batch, tokens, dim)
    borrowed; the norm output cast to dtype first when given."""
    segments = frame_segments(x.shape[0], x.shape[1], shift.shape[1])
    return kernels.norm_modulate([x.view(-1, x.shape[-1])], norm, shift[0], scale[0], segments, dtype=dtype).view(x.shape)

def gated_frames_(x, y_list, gate):
    """x += y * gate per latent frame, in place (gate (1, frames, dim), y handed off in y_list); returns x."""
    frames = gate.shape[1]
    y_list[0] = reshape_latent(y_list[0], frames)
    return restore_latent_shape(kernels.gated_residual_(reshape_latent(x, frames), y_list, gate.unsqueeze(2), fused=True))


def identify_k( b: float, d: int, N: int):
    """
    This function identifies the index of the intrinsic frequency component in a RoPE-based pre-trained diffusion transformer.

    Args:
        b (`float`): The base frequency for RoPE.
        d (`int`): Dimension of the frequency tensor
        N (`int`): the first observed repetition frame in latent space
    Returns:
        k (`int`): the index of intrinsic frequency component
        N_k (`int`): the period of intrinsic frequency component in latent space
    Example:
        In HunyuanVideo, b=256 and d=16, the repetition occurs approximately 8s (N=48 in latent space).
        k, N_k = identify_k(b=256, d=16, N=48)
        In this case, the intrinsic frequency index k is 4, and the period N_k is 50.
    """

    # Compute the period of each frequency in RoPE according to Eq.(4)
    periods = []
    for j in range(1, d // 2 + 1):
        theta_j = 1.0 / (b ** (2 * (j - 1) / d))
        N_j = round(2 * torch.pi / theta_j)
        periods.append(N_j)

    # Identify the intrinsic frequency whose period is closed to N（see Eq.(7)）
    diffs = [abs(N_j - N) for N_j in periods]
    k = diffs.index(min(diffs)) + 1
    N_k = periods[k-1]
    return k, N_k

def rope_params_riflex(max_seq_len, dim, theta=10000, L_test=30, k=6):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(max_seq_len), inv_theta_pow)
    if True:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return (freqs_cos, freqs_sin)
    else:
        freqs = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
    return freqs



def relative_l1_distance(last_tensor, current_tensor):
    l1_distance = torch.abs(last_tensor - current_tensor).mean()
    norm = torch.abs(last_tensor).mean()
    relative_l1_distance = l1_distance / norm
    return relative_l1_distance.to(torch.float32)

def trim_image_ref(y, ref_images_count, grid_sizes):
    y_shape = y.shape
    y = y.reshape(y_shape[0], grid_sizes[0], -1)
    y = y[:, ref_images_count:]
    y = y.reshape(y_shape[0], -1, y_shape[-1])
    grid_sizes_alt = [grid_sizes[0]-ref_images_count, *grid_sizes[1:]]
    return y, grid_sizes_alt

def fuse_with_image_ref(x, y, ref_images_count, grid_sizes, alpha = 1):
    y_shape = x.shape
    y = y.reshape(y_shape[0], grid_sizes[0]-ref_images_count, -1)
    x = x.reshape(y_shape[0], grid_sizes[0], -1)
    if alpha == 1:
        x[:, ref_images_count:] += y
    else:
        x[:, ref_images_count:].add_(y, alpha= alpha) 

    x = x.reshape(*y_shape)
    return x

class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False,  dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False,  dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, in_place= True):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        if in_place and x.is_contiguous():  # float32 statistics by chunks of rows: the same values without a float32 copy of x
            rows = x.view(-1, x.shape[-1])
            step = max(1, (64 << 20) // (rows.shape[-1] * 4))
            for start in range(0, rows.shape[0], step):
                part = rows[start:start + step]
                y = part.float()
                y.pow_(2)
                y = y.mean(dim=-1, keepdim=True)
                y += self.eps
                y.rsqrt_()
                part *= y
            x *= self.weight
            return x
        y = x.float()
        y.pow_(2)
        y = y.mean(dim=-1, keepdim=True)
        y += self.eps
        y.rsqrt_()
        if in_place:
            x *=  y
        else:
            x = x * y
        x *= self.weight
        return x
        # return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def my_LayerNorm(norm, x):
        y = x.float()
        y_m = y.mean(dim=-1, keepdim=True)
        y -= y_m 
        del y_m
        y.pow_(2)
        y = y.mean(dim=-1, keepdim=True)
        y += norm.eps
        y.rsqrt_()
        x = x *  y
        return x


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # return F.layer_norm(
        #     input, self.normalized_shape, self.weight, self.bias, self.eps
        # )
        if self.weight is not None:
            y = super().forward(x.to(self.weight.dtype))
        else:
            y = super().forward(x)
        x = y.type_as(x)
        return x
        # return super().forward(x).type_as(x)

from .posemb_layers import apply_rotary_emb, apply_rotary_emb_single, get_rotary_pos_embed, apply_rotary_source_id

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 block_no=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.block_no = block_no

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


    def text_cross_attention(self, xlist, context, return_q = False):
        x = xlist[0]
        xlist.clear()
        b, n, d = x.size(0), self.num_heads, self.head_dim
        nag_scale = offload.shared_state.get("_nag_scale",0)
        # compute query, key, value
        q = self.q(x)
        del x
        self.norm_q(q)
        q= q.view(b, -1, n, d)
        k = self.k(context)
        self.norm_k(k)
        k = k.view(context.shape[0], -1, n, d)
        v = self.v(context).view(context.shape[0], -1, n, d)

        if nag_scale <= 1 or len(k)==1:
            qvl_list=[q, k, v]
            if not return_q: del q
            del k, v
            x = pay_attention(qvl_list, recycle_q = not return_q)
            x = x.flatten(2, 3)
        else:
            nag_tau = offload.shared_state["_nag_tau"]
            nag_alpha = offload.shared_state["_nag_alpha"]
            qvl_list=[q, k[:1], v[:1]]
            x_pos = pay_attention(qvl_list)
            qvl_list=[q, k[1:], v[1:]]
            if not return_q: del q
            del k, v
            x_neg = pay_attention(qvl_list, recycle_q = not return_q)

            x_pos = x_pos.flatten(2, 3)
            x_neg = x_neg.flatten(2, 3)
            # Behold DeepBeepMeep as the NAG Butcher !: reduce highly VRAM consumption while at the same time turn the source in gibberish
            x_neg.mul_(1-nag_scale)
            x_neg.add_(x_pos, alpha= nag_scale)
            x_guidance = x_neg
            del x_neg
            norm_positive = torch.norm(x_pos, p=1, dim=-1, keepdim=True)
            norm_guidance = torch.norm(x_guidance, p=1, dim=-1, keepdim=True)
            scale = norm_guidance / norm_positive
            scale = torch.nan_to_num(scale, 10)
            factor = 1 / (norm_guidance + 1e-7) * norm_positive * nag_tau
            x_guidance = torch.where(scale > nag_tau, x_guidance * factor, x_guidance )
            del norm_positive, norm_guidance 
            x_pos.mul_(1 - nag_alpha)
            x_guidance.mul_(nag_alpha)
            x_guidance.add_(x_pos)
            x = x_guidance

            # x_guidance = x_pos * nag_scale - x_neg * (nag_scale - 1)
            # norm_positive = torch.norm(x_pos, p=1, dim=-1, keepdim=True).expand(*x_pos.shape)
            # norm_guidance = torch.norm(x_guidance, p=1, dim=-1, keepdim=True).expand(*x_guidance.shape)

            # scale = norm_guidance / norm_positive
            # scale = torch.nan_to_num(scale, 10)
            # x_guidance[scale > nag_tau] =  x_guidance[scale > nag_tau] / (norm_guidance[scale > nag_tau] + 1e-7) * norm_positive[scale > nag_tau] * nag_tau

            # x = x_guidance * nag_alpha + x_pos * (1 - nag_alpha)
        if return_q:
            return x, q
        else:
            return x, None

    def _project(self, x, residual, gate):
        """o(x), or residual += gate * o(x) per latent frame (gate (frames, dim)) written into residual and returned."""
        if residual is None:
            return self.o(x)
        x_list, x = [x.reshape(-1, x.shape[-1])], None
        rows = residual.view(-1, residual.shape[-1])
        return kernels.gated_linear(self.o, x_list, rows, gate, frame_segments(residual.shape[0], residual.shape[1], gate.shape[0])).view(residual.shape)

    def _norm_rope_(self, query, key, group, freqs):
        """attention_kit callback: norm_q/norm_k (over all heads) and RoPE of q and/or k in place."""
        for tensor, norm, heads, index in ((query, self.norm_q, group.q, 0), (key, self.norm_k, group.kv, 1)):
            if tensor is None:
                continue
            rows = tensor.flatten(2)
            if group.mean_squares is None:
                norm(rows)
            else:
                rows.mul_(group.mean_squares[index].add(norm.eps).rsqrt_())
                rows.mul_(norm.weight[heads.start * self.head_dim:heads.stop * self.head_dim])
            apply_rotary_emb_single([tensor], freqs, head_first=False)

    def forward(self, xlist, grid_sizes, freqs, block_mask = None, ref_target_masks = None, ref_images_count = 0, standin_phase =-1, lynx_ref_buffer = None, lynx_ref_scale = 0, sub_x_no=0, residual=None, gate=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if (block_mask is None and ref_target_masks is None and standin_phase < 1 and lynx_ref_buffer is None and isinstance(freqs, tuple)
                and not offload.shared_state.get("_chipmunk", False) and not offload.shared_state.get("_radial", False)):
            x = attention_kit.qkv_attention(xlist, self.q, self.k, self.v, self.num_heads, self.head_dim,
                                            lambda query, key, group: self._norm_rope_(query, key, group, freqs),
                                            norm_spans_heads=isinstance(self.norm_q, WanRMSNorm))
            return self._project(x.flatten(2), residual, gate), None
        x = xlist[0]
        xlist.clear()

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        q = self.q(x)
        ref_hidden_states = None
        if not lynx_ref_buffer is None:
            lynx_ref_features = lynx_ref_buffer[self.block_no]
            if self.norm_q is not None: ref_query = self.norm_q(q, in_place = False)
            if ref_images_count > 0:
                ref_query, _ = trim_image_ref(ref_query, ref_images_count, grid_sizes)
            ref_key = self.to_k_ref(lynx_ref_features)
            ref_value = self.to_v_ref(lynx_ref_features)
            if self.norm_k is not None: ref_key = self.norm_k(ref_key)
            ref_query, ref_key, ref_value = ref_query.unflatten(2, (self.num_heads, -1)), ref_key.unflatten(2, (self.num_heads, -1)), ref_value.unflatten(2, (self.num_heads, -1))
            qkv_list = [ref_query, ref_key, ref_value ]
            del ref_query, ref_key, ref_value
            ref_hidden_states = pay_attention(qkv_list)

        k, v = self.k(x), self.v(x)

        if standin_phase == 1:
            q += self.q_loras(x)
            k += self.k_loras(x)
            v += self.v_loras(x)
        self.norm_q(q)
        self.norm_k(k)
        q,k,v = q.view(b, s, n, d), k.view(b, s, n, d), v.view(b, s, n, d)
        del x

        qklist = [q,k]
        del q,k
        q,k = apply_rotary_emb(qklist, freqs, head_first=False)

        if standin_phase >= 1:
            standin_cache = get_cache("standin")
            if standin_phase == 1:
                standin_cache[self.block_no] = (k,v)
            elif standin_phase == 2:
                k_ip, v_ip = standin_cache[self.block_no]
                k, v = torch.concat([k, k_ip], dim=1), torch.concat([v, v_ip], dim=1)
                del k_ip, v_ip
        if ref_target_masks != None:
            x_ref_attn_map = get_attn_map_with_target(q, k , grid_sizes, ref_target_masks=ref_target_masks, ref_images_count = ref_images_count)
        else:
            x_ref_attn_map = None

        chipmunk = offload.shared_state.get("_chipmunk", False) 
        radial = offload.shared_state.get("_radial", False) 

        if chipmunk and self.__class__ == WanSelfAttention:
            q = q.transpose(1,2)
            k = k.transpose(1,2)
            v = v.transpose(1,2)
            attn_layers = offload.shared_state["_chipmunk_layers"]
            x = attn_layers[self.block_no](q, k, v)
            x = x.transpose(1,2)
        elif radial and self.__class__ == WanSelfAttention:
            qkv_list = [q,k,v]
            del q,k,v
            radial_cache = get_cache("radial")
            no_step_no = offload.shared_state["step_no"] 
            x = radial_cache[self.block_no](qkv_list=qkv_list, timestep_no=no_step_no)
        elif block_mask == None:
            qkv_list = [q,k,v]
            del q,k,v

            x = pay_attention( qkv_list, recycle_q=True)

        else:
            with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                x = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=block_mask
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
                del q,k,v

        if ref_hidden_states is not None:

            if ref_images_count > 0:
                x = fuse_with_image_ref(x, ref_hidden_states, ref_images_count, grid_sizes, alpha = lynx_ref_scale)
            else:
                x.add_(ref_hidden_states, alpha= lynx_ref_scale) 

        return self._project(x.flatten(2), residual, gate), x_ref_attn_map


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, xlist, context, grid_sizes, lynx_ip_embeds = None, lynx_ip_scale = 0, *args, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        x, q = self.text_cross_attention( xlist, context, return_q=lynx_ip_embeds is not None)
        if lynx_ip_embeds is not None and self.to_k_ip is not None:
            if self.registers is not None:
                from ..lynx.navit_utils import vector_to_list, merge_token_lists, list_to_vector
                ip_hidden_states_list = vector_to_list(lynx_ip_embeds, lynx_ip_embeds.shape[1], 1)
                ip_hidden_states_list = merge_token_lists(ip_hidden_states_list, [self.registers] * len(ip_hidden_states_list), 1)
                lynx_ip_embeds, ip_lens = list_to_vector(ip_hidden_states_list, 1)
                ip_hidden_states_list = None
            ip_key = self.to_k_ip(lynx_ip_embeds)
            ip_value = self.to_v_ip(lynx_ip_embeds)
            # if self.norm_q is not None: ip_query = self.norm_q(q)
            if self.norm_rms_k is None:
                ip_key = self.norm_k(ip_key)
            else:
                ip_key = self.norm_rms_k(ip_key)
            ip_inner_dim = ip_key.shape[-1]
            ip_head_dim = ip_inner_dim // self.num_heads
            batch_size = q.shape[0]
            # ip_query = ip_query.view(batch_size, -1, attn.heads, ip_head_dim)
            ip_key = ip_key.view(batch_size, -1, self.num_heads, ip_head_dim)
            ip_value = ip_value.view(batch_size, -1, self.num_heads, ip_head_dim)
            qkv_list = [q, ip_key, ip_value]
            del q, ip_key, ip_value
            ip_hidden_states = pay_attention(qkv_list, recycle_q= True).reshape(*x.shape)
            x.add_(ip_hidden_states, alpha= lynx_ip_scale)

        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 block_no=0):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, block_no)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, xlist, context, grid_sizes, audio_proj, audio_scale, audio_context_lens, *args, **kwargs ):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """


        context_img = context[:, :257]
        context = context[:, 257:]
        
        x, q = self.text_cross_attention( xlist, context, return_q = True)
        if len(q) != len(context_img):
            context_img = context_img[:len(q)]
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        if audio_scale != None:
            audio_x = self.processor(q, audio_proj, grid_sizes[0], audio_context_lens)
        k_img = self.k_img(context_img)
        self.norm_k_img(k_img)
        k_img = k_img.view(1, -1, n, d)
        v_img = self.v_img(context_img).view(1, -1, n, d)
        if b > 1:
            k_img, v_img = k_img.expand(b, -1, -1, -1), v_img.expand(b, -1, -1, -1)
        qkv_list = [q, k_img, v_img]
        del q, k_img, v_img
        img_x = pay_attention(qkv_list, recycle_q = True)
        img_x = img_x.flatten(2)

        # output
        x += img_x
        del img_x
        if audio_scale != None:
            x.add_(audio_x, alpha= audio_scale)
        x = self.o(x)
        return x



WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None,
                 block_no = 0,                 
                 output_dim=0,
                 norm_input_visual=True,
                 class_range=24,
                 class_interval=4,
                 ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.block_no = block_no

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, block_no= block_no)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps, 
                                                                      block_no)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.block_id = block_id

        if output_dim > 0:
            from ..multitalk.attention import SingleStreamMutiAttention 
            # init audio module
            self.audio_cross_attn = SingleStreamMutiAttention(
                    dim=dim,
                    encoder_hidden_states_dim=output_dim,
                    num_heads=num_heads,
                    qk_norm=False,
                    qkv_bias=True,
                    eps=eps,
                    norm_layer=WanRMSNorm,
                    class_range=class_range,
                    class_interval=class_interval
                )
            self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True)  if norm_input_visual else nn.Identity()
    
    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        context,
        hints= None, 
        context_scale=[1.0],
        cam_emb= None,
        block_mask = None,
        audio_proj= None,
        audio_context_lens= None,
        audio_scale=None,
        multitalk_audio=None,
        multitalk_masks=None,
        ref_images_count=0,
        standin_phase=-1,
        motion_vec = None,
        lynx_ip_embeds = None,
        lynx_ip_scale = 0,
        lynx_ref_scale = 0,
        lynx_feature_extractor = False,
        lynx_ref_buffer = None,
        sub_x_no =0,         
        animate2_generation = None,
        animate2_cached = False,
        taomate_audio = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if animate2_generation is not None:
            return animate2_cached_attention_block(self, x, e, grid_sizes, freqs, animate2_generation) if animate2_cached else animate2_attention_block(self, x, e, context, grid_sizes, freqs, animate2_generation)

        hints_processed = None
        attention_dtype =  self.self_attn.q.weight.dtype 
        dtype = x.dtype

        if self.block_id is not None and hints is not None:
            kwargs = { 
                "grid_sizes" : grid_sizes,
                "freqs" :freqs, 
                "context" : context,
                "e" : e,
            }
            hints_processed= []
            for scale, hint in zip(context_scale, hints):
                if scale == 0:
                    hints_processed.append(None)
                else:
                    hints_processed.append(self.vace(hint, x, **kwargs) if self.block_id == 0 else self.vace(hint, None, **kwargs))
                     
        e = frame_modulation(self.modulation.weight, e)
        # self-attention
        x_mod = modulate_frames(x, self.norm1, e[0], e[1])

        if cam_emb != None:
            cam_emb = self.cam_encoder(cam_emb)
            if cam_emb.ndim == 3 and cam_emb.shape[1] == x_mod.shape[1]:
                x_mod += cam_emb
            else:
                cam_emb = cam_emb.repeat(1, 2, 1)
                cam_emb = cam_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, grid_sizes[1], grid_sizes[2], 1)
                cam_emb = rearrange(cam_emb, 'b f h w d -> b (f h w) d')
                x_mod += cam_emb

        xlist = [x_mod.to(attention_dtype)]
        if lynx_feature_extractor: get_cache("lynx_ref_buffer")[sub_x_no][self.block_no] = xlist[0]
        del x_mod
        gated = cam_emb is None  # the output projection adds its output, gated per latent frame, into x
        y, x_ref_attn_map = self.self_attn( xlist, grid_sizes, freqs, block_mask = block_mask, ref_target_masks = multitalk_masks, ref_images_count = ref_images_count, standin_phase= standin_phase, lynx_ref_buffer = lynx_ref_buffer, lynx_ref_scale = lynx_ref_scale, sub_x_no = sub_x_no,
                                            residual = x if gated else None, gate = e[2][0])
        if gated:
            x = y
        else:
            y_list = [self.projector(y.to(dtype))]
            x = gated_frames_(x, y_list, e[2])
        del y

        if context is not None:
            y = self.norm3(x)
            y = y.to(attention_dtype)
            ylist= [y]
            del y
            x += self.cross_attn(ylist, context, grid_sizes, audio_proj = audio_proj, audio_scale = audio_scale, audio_context_lens = audio_context_lens, lynx_ip_embeds=lynx_ip_embeds, lynx_ip_scale=lynx_ip_scale).to(dtype)

        if taomate_audio is not None:
            x = taomate_audio_attention(self, x, taomate_audio, attention_dtype)

        if multitalk_audio != None:
            # cross attn of multitalk audio
            y = self.norm_x(x)
            y = y.to(attention_dtype)
            if ref_images_count == 0:
                ylist= [y]
                del y
                x += self.audio_cross_attn(ylist, encoder_hidden_states=multitalk_audio, shape=grid_sizes, x_ref_attn_map=x_ref_attn_map)
            else:
                y, grid_sizes_alt = trim_image_ref(y, ref_images_count, grid_sizes)
                ylist= [y]
                y = None
                y = self.audio_cross_attn(ylist, encoder_hidden_states=multitalk_audio, shape=grid_sizes_alt, x_ref_attn_map=x_ref_attn_map)
                x = fuse_with_image_ref(x, y, ref_images_count, grid_sizes)
                del y

        y = modulate_frames(x, self.norm2, e[3], e[4]).to(attention_dtype)

        y_list, y = [y], None  # the FFN by chunks, its output gated per latent frame into x
        kernels.mlp(y_list, self.ffn[0], self.ffn[2], "gelu", residual=x, segments=kernels.gate_segments(e[5], x.shape[0], x.shape[1]))

        if hints_processed is not None:
            for hint, scale in zip(hints_processed, context_scale):
                if scale != 0:
                    if scale == 1:
                        x.add_(hint)
                    else:
                        x.add_(hint, alpha= scale)

        if motion_vec is not None and self.block_no % 5 == 0:
            x += self.face_adapter_fuser_blocks(x.to(self.face_adapter_fuser_blocks.linear1_kv.weight.dtype), motion_vec, None)

        return x 

class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,  
        channels=768, 
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds)) 
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf)) 
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1) 
        audio_embeds_vf = audio_embeds = None
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)
        audio_embeds_c = None
        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens



class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, hints, x, **kwargs):
        # behold dbm magic !
        c = hints[0]
        hints[0] = None
        if self.block_id == 0:
            c = self.before_proj(c)
            bz = x.shape[0]
            if bz > c.shape[0]: c = c.repeat(bz, 1, 1 )
            c += x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        hints[0] = c
        return c_skip

    
class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x_list, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        x = x_list[0]  # handed off: the hidden states are released once normalized, before the float32 copy
        x_list.clear()

        e = frame_modulation(self.modulation.weight, e.unsqueeze(1))
        x = modulate_frames(x, self.norm, e[0], e[1])
        x= x.to(self.head.weight.dtype)
        x = self.head(x)
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds.to(self.emb_pos.dtype) + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class WanModel(ModelMixin, ConfigMixin):
    _offload_hooks = ["project_audio"]

    def setup_chipmunk(self):
        # from chipmunk.util import LayerCounter
        # from chipmunk.modules import SparseDiffMlp, SparseDiffAttn
        seq_shape = (21, 45, 80)
        chipmunk_layers =[]
        for i in range(self.num_layers):
            layer_num, layer_counter = LayerCounter.build_for_layer(is_attn_sparse=True, is_mlp_sparse=False)            
            chipmunk_layers.append( SparseDiffAttn(layer_num, layer_counter))
        offload.shared_state["_chipmunk_layers"] = chipmunk_layers

        chipmunk_layers[0].initialize_static_mask(
            seq_shape=seq_shape,
            txt_len=0,
            local_heads_num=self.num_heads,
            device='cuda'
        )
        chipmunk_layers[0].layer_counter.reset()

    def release_chipmunk(self):
        offload.shared_state["_chipmunk_layers"] = None

    @staticmethod
    def preprocess_sd_with_dtype(dtype, sd):
        new_sd = {}
        prefix_list = ["model.diffusion_model"]
        end_list = [".norm3.bias", ".norm3.weight", ".norm_q.bias", ".norm_q.weight", ".norm_k.bias", ".norm_k.weight" ]
        for k,v in sd.items():
            for prefix in prefix_list:
                if k.startswith(prefix): 
                    k = k[len(prefix)+1:]
                    break
            if ".attn2.norm_added_q." in k:
                continue
            if v.dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
                for endfix in end_list:
                    if k.endswith(endfix):
                        v = v.to(dtype)
                        break
            if k.startswith("patch_embedding_pose."):
                k = k.replace("patch_embedding_pose.", "pose_patch_embedding.", 1)
            if k.startswith("patch_embedding_mask."):
                k = k.replace("patch_embedding_mask.", "mask_patch_embedding.", 1)
            if k.startswith("blocks."):
                parts = k.split(".")
                if len(parts) > 2 and parts[2] == "block":
                    del parts[2]
                    k = ".".join(parts)
            if not k.startswith("vae."):
                new_sd[k] = v
        return new_sd
    def preprocess_loras(self, base_model_type, sd):

        first = next(iter(sd), None)
        if first == None:
            return sd

        if base_model_type in ["scail"]:
            sd.pop("diffusion_model.patch_embedding.diff", None)
            sd.pop("diffusion_model.patch_embedding.diff_b", None)
            return sd
        
        new_sd = {}
        for k,v in sd.items():
            if k.endswith("modulation.diff"):
                pass
            else:
                new_sd[ k] = v
        sd = new_sd

        # if first.startswith("blocks."):
        #     new_sd = {}
        #     for k,v in sd.items():
        #         new_sd["diffusion_model." + k] = v
        #     sd = new_sd
        if ".default." in first:
            new_sd = {}
            for k,v in sd.items():
                k = k.replace(".default.", ".")
                new_sd[k] = v 
            sd = new_sd

        if first.startswith("vace_blocks."):
            new_sd = {}
            for k,v in sd.items():
                if k.startswith("vace_blocks."):
                    l = k.split(".")
                    block_no = self.vace_layers[int(l[1])]
                    l[0] = "blocks." + str(block_no)
                    l[1] = "vace"
                    k = ".".join(l)
                    print(k)
                new_sd[k] = v 
            sd = new_sd

        if first.startswith("lora_unet_"):
            new_sd = {}
            print("Converting Lora Safetensors format to Lora Diffusers format")
            repl_list = ["cross_attn", "self_attn", "ffn"]
            src_list = ["_" + k + "_" for k in repl_list]
            tgt_list = ["." + k + "." for k in repl_list]
            top_level_repl_list = [
                ("lora_unet__head_head", "diffusion_model.head.head"),
                ("lora_unet_head_head", "diffusion_model.head.head"),
                ("lora_unet__img_emb_proj_", "diffusion_model.img_emb.proj."),
                ("lora_unet_img_emb_proj_", "diffusion_model.img_emb.proj."),
                ("lora_unet__text_embedding_", "diffusion_model.text_embedding."),
                ("lora_unet_text_embedding_", "diffusion_model.text_embedding."),
                ("lora_unet__time_embedding_", "diffusion_model.time_embedding."),
                ("lora_unet_time_embedding_", "diffusion_model.time_embedding."),
                ("lora_unet__time_projection_", "diffusion_model.time_projection."),
                ("lora_unet_time_projection_", "diffusion_model.time_projection."),
            ]

            for k,v in sd.items():
                k = k.replace("lora_unet_blocks_","diffusion_model.blocks.")
                k = k.replace("lora_unet__blocks_","diffusion_model.blocks.")
                for src, tgt in top_level_repl_list:
                    k = k.replace(src, tgt)

                for s,t in zip(src_list, tgt_list):
                    k = k.replace(s,t)

                k = k.replace("lora_up","lora_B")
                k = k.replace("lora_down","lora_A")

                new_sd[k] = v
            sd = new_sd
        if base_model_type in ["scail2_14B", "scail2_1.3B"]:
            new_sd = {}
            for k, v in sd.items():
                if "patch_embedding.diff" in k and torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] != 20: continue
                new_sd[k] = v

            sd = new_sd
        from wgp import test_class_i2v 
        if not test_class_i2v(base_model_type) or base_model_type in ["i2v_2_2"]:
            new_sd = {}
            # convert loras for i2v to t2v
            for k,v in sd.items():
                if  any(layer in k for layer in ["cross_attn.k_img", "cross_attn.v_img", "img_emb."]):
                    continue
                new_sd[k] = v
            sd = new_sd

        return sd    
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,                 
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 mask_dim=28,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf = False,
                 recammaster = False,
                 inject_sample_info = False,
                 fantasytalking_dim = 0,
                 multitalk_output_dim = 0,
                 audio_window=5,
                 intermediate_dim=512,
                 context_tokens=32,
                 vae_scale=4, # vae timedownsample scale
                 norm_input_visual=True,
                 norm_output_audio=True,
                 standin= False,
                 motion_encoder_dim=0,
                 lynx=None,
                 steadydancer = False,
                 scail = False,
                 scail2 = False,
                 any_kiwi_source = False,
                 any_kiwi_ref = False,
                 vista4d = False,
                 vista4d_positional_embedding_offset = 31,
                 shotplan = False,
                 add_control_adapter = False,
                 in_dim_control_adapter = 24,
                 downscale_factor_control_adapter = 8,
                 uni3c_controlnet = None,
                 taomate = False,
                 ):

        super().__init__()

        self.model_type = model_type
        self.taomate = taomate

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.mask_dim = mask_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = 1
        self.flag_causal_attention = False
        self.block_mask = None
        self.cache = None
        self.inject_sample_info = inject_sample_info
        self.motion_encoder_dim = motion_encoder_dim
        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale
        self.any_kiwi_source = any_kiwi_source
        self.any_kiwi_ref = any_kiwi_ref
        self.shotplan = shotplan

        multitalk = multitalk_output_dim > 0
        self.multitalk = multitalk
        self.steadydancer = steadydancer
        self.scail = scail
        self.scail2 = scail2
        animate = motion_encoder_dim > 0

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        if shotplan:
            self.hardcut_embedding = nn.Parameter(torch.zeros(1, 1, dim))

        # blocks
        if vace_layers == None:
            cross_attn_type = 't2v_cross_attn' if model_type in ['t2v','i2v2_2', 'ti2v2_2'] else 'i2v_cross_attn'
            self.blocks = nn.ModuleList([
                WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                window_size, qk_norm, cross_attn_norm, eps, block_no =i, output_dim=multitalk_output_dim, norm_input_visual=norm_input_visual)
                for i in range(num_layers)
            ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb = flf)

        if multitalk :
            # init audio adapter
            self.audio_proj = AudioProjModel(
                        seq_len=audio_window,
                        seq_len_vf=audio_window+vae_scale-1,
                        intermediate_dim=intermediate_dim,
                        output_dim=multitalk_output_dim,
                        context_tokens=context_tokens,
                        norm_output_audio=norm_output_audio,
                    )

        # initialize weights
        self.init_weights()

        if vace_layers != None:            
            self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
            self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

            assert 0 in self.vace_layers
            self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

            # blocks
            self.blocks = nn.ModuleList([
                WanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                    self.cross_attn_norm, self.eps, block_no =i,
                                    block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None,
                                    output_dim=multitalk_output_dim,
                                    norm_input_visual=norm_input_visual,
                                    )
                for i in range(self.num_layers)
            ])

            # vace blocks
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                        self.cross_attn_norm, self.eps, block_id=i)
                for i in self.vace_layers
            ])

            # vace patch embeddings
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
            )
        if recammaster :
            dim=self.blocks[0].self_attn.q.weight.shape[0]
            for block in self.blocks:
                block.cam_encoder = nn.Linear(12, dim)
                block.projector = nn.Linear(dim, dim)
                block.cam_encoder.weight.data.zero_()
                block.cam_encoder.bias.data.zero_()
                block.projector.weight = nn.Parameter(torch.eye(dim))
                block.projector.bias = nn.Parameter(torch.zeros(dim))            

        if vista4d:
            from ..vista4d.runtime import add_vista4d_modules
            add_vista4d_modules(self)

        if taomate:
            from ..taomate import add_modules as add_taomate_modules
            add_taomate_modules(self)

        if add_control_adapter: # Wan Fun Control-Camera
            from ..camera_control.fun_camera import SimpleAdapter
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], downscale_factor=downscale_factor_control_adapter)

        if uni3c_controlnet is not None:
            from ..camera_control.uni3c import Uni3CControlNet
            self.uni3c = Uni3CControlNet(in_dim, dim, patch_size=patch_size, **uni3c_controlnet)

        if fantasytalking_dim > 0:
            from ..fantasytalking.model import WanCrossAttentionProcessor
            for block in self.blocks:
                block.cross_attn.processor = WanCrossAttentionProcessor(fantasytalking_dim, dim)

        if standin:
            for block in self.blocks:
                block.self_attn.q_loras = LoRALinearLayer(dim, dim, rank=128)
                block.self_attn.k_loras = LoRALinearLayer(dim, dim, rank=128)
                block.self_attn.v_loras = LoRALinearLayer(dim, dim, rank=128)

        if lynx is not None:
            from ..lynx.attention_processor import setup_lynx_attention_layers
            lynx_full = lynx=="full"
            setup_lynx_attention_layers(self.blocks, lynx_full, dim)

        if animate:
            self.pose_patch_embedding = nn.Conv3d(
                16, dim, kernel_size=patch_size, stride=patch_size
            )

            self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=20)
            self.face_adapter = FaceAdapter(
                heads_num=self.num_heads,
                hidden_dim=self.dim,
                num_adapter_layers=self.num_layers // 5,
            )

            self.face_encoder = FaceEncoder(
                in_dim=motion_encoder_dim,
                hidden_dim=self.dim,
                num_heads=4,
            )

        if scail or scail2:
            self.pose_patch_embedding = nn.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size
            )
        if scail2:
            self.mask_patch_embedding = nn.Conv3d(
                mask_dim, dim, kernel_size=patch_size, stride=patch_size
            )

        if steadydancer:
            self.in_dim_c = 16

        ############### Condition-Reconciliation Mechanism ###############
            self.patch_embedding_fuse = nn.Conv3d(      # x, fused pose, aligned pose
                in_dim + self.in_dim_c + self.in_dim_c, dim, kernel_size=patch_size, stride=patch_size)
            self.patch_embedding_ref_c = nn.Conv3d(    # ref_c
                self.in_dim_c, dim, kernel_size=patch_size, stride=patch_size)

            ############### Synergistic Pose Modulation Modules ###############
            # Spatial Structure Adaptive Extractor
            self.condition_embedding_spatial = DYModule(inp=self.in_dim_c, oup=self.in_dim_c)
            # Temporal Motion Coherence Module
            self.condition_embedding_temporal = nn.Sequential(
                FactorConv3d(in_channels=self.in_dim_c, out_channels=self.in_dim_c, kernel_size=(3, 3, 3), stride=1),
                nn.SiLU(),
                FactorConv3d(in_channels=self.in_dim_c, out_channels=self.in_dim_c, kernel_size=(3, 3, 3), stride=1),
                nn.SiLU(),
                FactorConv3d(in_channels=self.in_dim_c, out_channels=self.in_dim_c, kernel_size=(3, 3, 3), stride=1),
                nn.SiLU()
            )
            # Frame-wise Attention Alignment Unit
            self.condition_embedding_align = PoseRefNetNoBNV3(in_channels_x=16,
                                                    in_channels_c=16,
                                                    hidden_dim=128,
                                                    num_heads=8)

    def project_audio(self, features):
        """TaoMate-FVSR: audio tokens (frames, window, dim) of hidden-state windows (frames, window, features). A hooked entry point of its own: they stay the same for the whole window."""
        return self.audio_embedding(features.to(self.audio_embedding.proj_in.bias.dtype))

    def adapt_modulation(self, block_name ='blocks'):
        def move(v, param_name = "modulation"):
            module = torch.nn.Module()
            module.weight = getattr(v, param_name)
            delattr(v, param_name)
            setattr(v, param_name, module)

        modules_dict= { k: m for k, m in self.named_modules()}
        for k,v in modules_dict[block_name]._modules.items():
            move(v)

        if block_name != "blocks": return
        move(modules_dict["head"])

    def adapt_vace_model(self):
        self.adapt_modulation("vace_blocks")

        modules_dict= { k: m for k, m in self.named_modules()}
        for model_layer, vace_layer in self.vace_layers_mapping.items():
            module = modules_dict[f"vace_blocks.{vace_layer}"]
            target = modules_dict[f"blocks.{model_layer}"]
            setattr(target, "vace", module )
        delattr(self, "vace_blocks")


    def adapt_animate_model(self):
        modules_dict= { k: m for k, m in self.named_modules()}
        for animate_layer in range(8):
            module = modules_dict[f"face_adapter.fuser_blocks.{animate_layer}"]
            model_layer = animate_layer * 5
            target = modules_dict[f"blocks.{model_layer}"]
            setattr(target, "face_adapter_fuser_blocks", module )
        delattr(self, "face_adapter")

    def adapt_uni3c_model(self):
        # each Uni3C ControlNet block runs with the main block it feeds, so that mmgp loads them together
        for block, uni3c_block in zip(self.blocks, self.uni3c.blocks):
            block.uni3c = uni3c_block
        self.uni3c_layers = len(self.uni3c.blocks)
        del self.uni3c.blocks

    def apply_post_init_changes(self):
        self.adapt_modulation()
        if hasattr(self, "vace_blocks"): self.adapt_vace_model()
        if hasattr(self, "face_adapter"): self.adapt_animate_model()
        if hasattr(self, "uni3c"): self.adapt_uni3c_model()

    def lock_layers_dtypes(self, hybrid_dtype = None, dtype = torch.float32):
        layer_list = [self.head, self.head.head, self.head.modulation, self.patch_embedding]
        if self.scail or self.scail2:
            layer_list += [self.pose_patch_embedding]
        if self.scail2:
            layer_list += [self.mask_patch_embedding]
        target_dype= dtype
        
        layer_list2 = [ self.time_embedding, self.time_embedding[0], self.time_embedding[2],
                    self.time_projection, self.time_projection[1]] #, self.text_embedding, self.text_embedding[0], self.text_embedding[2] ]

        for block in self.blocks:
            layer_list2 += [block.norm4 if self.taomate else block.norm3]

        if self.taomate:
            layer_list += [self.spark_ref_proj]

        if hasattr(self, "audio_proj"):
            for block in self.blocks:
                layer_list2 += [block.norm_x]

        if hasattr(self, "fps_embedding"):
            layer_list2 += [self.fps_embedding, self.fps_projection, self.fps_projection[0], self.fps_projection[2]]

        if hasattr(self, "vace_patch_embedding"):
            layer_list2 += [self.vace_patch_embedding]
            layer_list2 += [self.vace_blocks[0].before_proj]
            for block in self.vace_blocks:
                layer_list2 += [block.after_proj, block.norm3]

        if hasattr(self, "latent_encoder"):
            layer_list += [module for module in self.latent_encoder.modules() if isinstance(module, (nn.Conv3d, nn.Linear))]

        if hasattr(self, "uni3c"):
            layer_list += self.uni3c.fp32_layers()

        target_dype2 = hybrid_dtype if hybrid_dtype != None else dtype 

        # cam master
        if hasattr(self.blocks[0], "projector"):
            for block in self.blocks:
                layer_list2 += [block.projector, block.cam_encoder]

        for current_layer_list, current_dtype in zip([layer_list, layer_list2], [target_dype, target_dype2]):
            for layer in current_layer_list:
                layer._lock_dtype = current_dtype

        self._lock_dtype = dtype

    def compute_magcache_threshold(self, start_step, timesteps = None, speed_factor =0):
        skips_step_cache = self.cache
        def nearest_interp(src_array, target_length):
            src_length = len(src_array)
            if target_length == 1: return np.array([src_array[-1]])
            scale = (src_length - 1) / (target_length - 1)
            mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
            return src_array[mapped_indices]
        num_inference_steps = len(timesteps)
        def_mag_ratios = np.array([1.0]*2+ skips_step_cache.def_mag_ratios)
        if len(def_mag_ratios) != num_inference_steps*2:
            mag_ratio_con = nearest_interp(def_mag_ratios[0::2], num_inference_steps)
            mag_ratio_ucon = nearest_interp(def_mag_ratios[1::2], num_inference_steps)
            interpolated_mag_ratios = np.concatenate([mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
            skips_step_cache.mag_ratios = interpolated_mag_ratios
        else:
            skips_step_cache.mag_ratios = def_mag_ratios


        best_deltas = None
        best_threshold = 0.01
        best_diff = 1000
        best_signed_diff = 1000
        target_nb_steps= int(len(timesteps) / speed_factor)
        threshold = 0.01
        x_id_max = 1
        while threshold <= 0.6:
            nb_steps = 0
            diff = 1000
            accumulated_err, accumulated_steps, accumulated_ratio = [0] * x_id_max , [0] * x_id_max, [1.0] * x_id_max
            for i, t in enumerate(timesteps):
                if i<=start_step:
                    skip  = False
                    x_should_calc = [True] * x_id_max
                else:
                    x_should_calc = []
                    for cur_x_id in range(x_id_max):
                        cur_mag_ratio = skips_step_cache.mag_ratios[i * 2 + cur_x_id] # conditional and unconditional in one list
                        accumulated_ratio[cur_x_id] *= cur_mag_ratio # magnitude ratio between current step and the cached step
                        accumulated_steps[cur_x_id] += 1 # skip steps plus 1
                        cur_skip_err = np.abs(1-accumulated_ratio[cur_x_id]) # skip error of current steps
                        accumulated_err[cur_x_id] += cur_skip_err # accumulated error of multiple steps
                        if accumulated_err[cur_x_id]<threshold and accumulated_steps[cur_x_id]<=skips_step_cache.magcache_K:
                            skip  = True
                        else:
                            skip  = False
                            accumulated_err[cur_x_id], accumulated_steps[cur_x_id], accumulated_ratio[cur_x_id] = 0, 0, 1.0
                        x_should_calc.append(not skip)
                if not skip:
                    nb_steps += 1
                    signed_diff = target_nb_steps - nb_steps               
                    diff = abs(signed_diff)  
            if diff < best_diff:
                best_threshold = threshold
                best_diff = diff
                best_signed_diff = signed_diff
            elif diff > best_diff:
                break
            threshold += 0.01
        skips_step_cache.magcache_thresh = best_threshold
        print(f"Mag Cache, best threshold found:{best_threshold:0.2f} with gain x{len(timesteps)/(target_nb_steps - best_signed_diff):0.2f} for a target of x{speed_factor}")
        return best_threshold

    def compute_teacache_threshold(self, start_step, timesteps = None, speed_factor =0): 
        skips_step_cache = self.cache
        modulation_dtype = self.time_projection[1].weight.dtype
        rescale_func = np.poly1d(skips_step_cache.coefficients)
        e_list = []
        for t in timesteps:
            t = torch.stack([t])
            time_emb =  self.time_embedding( sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(modulation_dtype) )  # b, dim   
            e_list.append(time_emb)
        best_deltas = None
        best_threshold = 0.01
        best_diff = 1000
        best_signed_diff = 1000
        target_nb_steps= int(len(timesteps) / speed_factor)
        threshold = 0.01
        while threshold <= 0.6:
            accumulated_rel_l1_distance =0
            nb_steps = 0
            diff = 1000
            deltas = []
            for i, t in enumerate(timesteps):
                skip = False    
                if not (i<=start_step or i== len(timesteps)-1):
                    delta = abs(rescale_func(((e_list[i]-e_list[i-1]).abs().mean() / e_list[i-1].abs().mean()).cpu().item()))
                    # deltas.append(delta)
                    accumulated_rel_l1_distance += delta
                    if accumulated_rel_l1_distance < threshold:
                        skip = True
                        # deltas.append("SKIP")
                    else:
                        accumulated_rel_l1_distance = 0
                if not skip:
                    nb_steps += 1
                    signed_diff = target_nb_steps - nb_steps               
                    diff = abs(signed_diff)  
            if diff < best_diff:
                best_threshold = threshold
                best_deltas = deltas
                best_diff = diff
                best_signed_diff = signed_diff
            elif diff > best_diff:
                break
            threshold += 0.01
        skips_step_cache.rel_l1_thresh = best_threshold
        print(f"Tea Cache, best threshold found:{best_threshold:0.2f} with gain x{len(timesteps)/(target_nb_steps - best_signed_diff):0.2f} for a target of x{speed_factor}")
        # print(f"deltas:{best_deltas}")
        return best_threshold

    
    def forward(
        self,
        x,
        t,
        context,
        vace_context = None,
        vace_context_scale=[1.0],        
        clip_fea=None,
        y=None,
        freqs = None,
        pipeline = None,
        current_step_no = 0,
        real_step_no = 0,
        x_id= 0,
        max_steps = 0, 
        perturbation_layers=None,
        callback = None,
        cam_emb: torch.Tensor = None,
        fps = None,
        causal_block_size = 1,
        causal_attention = False,
        audio_proj=None,
        audio_context_lens=None,
        audio_scale=None,
        multitalk_audio = None,
        multitalk_masks = None,
        ref_images_count = 0,
        standin_freqs = None,
        standin_ref = None,
        pose_latents=None, 
        face_pixel_values=None,
        lynx_ip_embeds = None,
        lynx_ip_scale = 0,
        lynx_ref_scale = 0,
        lynx_feature_extractor = False,
        lynx_ref_buffer = None,
        steadydancer_condition = None,
        steadydancer_ref_x = None,
        steadydancer_ref_c = None,
        steadydancer_clip_fea_c = None,
        scail_pose_latents = None,
        scail2_ref_latents = None,
        scail2_pose_latents = None,
        scail2_driving_masks = None,
        scail2_ref_masks = None,
        kiwi_source_condition = None,
        kiwi_ref_condition = None,
        kiwi_ref_pad_first = False,
        bernini_sources = None,
        vista = None,
        shotplan_cut_frames = None,
        animate2_ref_x = None,
        animate2_ref_y = None,
        animate2_ref_context = None,
        animate2_ref_clip_fea = None,
        animate2_ref_freqs = None,
        animate2_log_scale = 0.0,
        animate2_kv_cache = "Disabled",  # GPU (exact), INT8 (K/V on the GPU), RAM (INT8 K/V in RAM) (animate2.py)
        camera_adapter_features = None,
        uni3c_condition = None,
        taomate_reference = None,  # per branch: the reference canvas, one tensor shared by the branches using it, or None for the all-zero canvas
        taomate_audio = None,  # per branch: project_audio's tokens
    ):
        # patch_dtype =  self.patch_embedding.weight.dtype
        modulation_dtype = self.time_projection[1].weight.dtype
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if torch.is_tensor(freqs) and freqs.device != device:
            freqs = freqs.to(device)

        chipmunk = offload.shared_state.get("_chipmunk", False) 
        if chipmunk:
            # from chipmunk.ops.voxel import voxel_chunk_no_padding, reverse_voxel_chunk_no_padding
            voxel_shape = (4, 6, 8)
        real_seq = 0
        x_list = list(x)
        x.clear()
        x = None
        output_slice = None
        output_grid_sizes = None
        uni3c_x = None
        joint_pass = len(x_list) > 1
        is_source_x = [False] * len(x_list) if bernini_sources is not None else [ x.data_ptr() == x_list[0].data_ptr() and i > 0 for i, x in enumerate(x_list) ]
        last_x_idx  = 0
        steadydancer = steadydancer_condition is not None
        vista_enabled = isinstance(vista, dict) and vista.get("source_latents") is not None
        if steadydancer: # steady dancer
            x_noise_clone = x_list[0].clone()
        if isinstance(y, list):
            y_list = y
        else:
            y_list = [y] * len(x_list)
        if isinstance(scail2_ref_masks, list):
            scail2_ref_masks_list = scail2_ref_masks
        else:
            scail2_ref_masks_list = [scail2_ref_masks] * len(x_list)
        if isinstance(scail2_ref_latents, list):
            scail2_ref_latents_list = scail2_ref_latents
        else:
            scail2_ref_latents_list = [scail2_ref_latents] * len(x_list)
        bernini_enabled = bernini_sources is not None
        animate2_enabled = animate2_ref_x is not None
        animate2_cache = get_cache("animate2_kv") if animate2_enabled and animate2_kv_cache != "Disabled" else None
        animate2_cached = animate2_cache is not None and len(animate2_cache.get("blocks", [])) == len(self.blocks)
        bernini_freqs_list = []
        bernini_output_slices = []

        for i, (is_source, x, y, scail2_ref_mask, scail2_ref_latent) in enumerate(zip(is_source_x, x_list, y_list, scail2_ref_masks_list, scail2_ref_latents_list)):
            if is_source:
                x_list[i] = x_list[0].clone()
                last_x_idx = i
            else:
                # image source
                bz = len(x)
                real_latent_frames = int(x.shape[2])
                if y is not None:
                    y = y.unsqueeze(0)        
                    if bz > 1: y = y.expand(bz, -1, -1, -1, -1)
                    x = torch.cat([x, y], dim=1)
                if scail2_ref_latent is not None:
                    scail2_ref_latent = scail2_ref_latent.unsqueeze(0) if scail2_ref_latent.ndim == 4 else scail2_ref_latent
                    if scail2_ref_latent.shape[0] != bz:
                        scail2_ref_latent = scail2_ref_latent.expand(bz, -1, -1, -1, -1)
                    ref_mask = torch.ones(scail2_ref_latent.shape[0], 4, *scail2_ref_latent.shape[2:], device=scail2_ref_latent.device, dtype=scail2_ref_latent.dtype)
                    scail2_ref_latent = torch.cat([scail2_ref_latent, ref_mask], dim=1)
                    x = torch.cat([scail2_ref_latent.to(device=x.device, dtype=x.dtype), x], dim=2)
                    output_slice = slice(scail2_ref_latent.shape[2], scail2_ref_latent.shape[2] + real_latent_frames)
                if uni3c_condition is not None and i == 0: # Uni3C ControlNet input: noisy latents and i2v mask, then the render latents
                    render_latents = uni3c_condition["render_latents"]
                    uni3c_x = self.uni3c.embed(torch.cat([x[:1, :x.shape[1] - render_latents.shape[1]], render_latents.to(x)], dim=1), uni3c_condition["condition_features"])
                    render_latents = None
                # embeddings
                if bernini_enabled:
                    source_tokens, cos_parts, sin_parts = [], [], []
                    for source_latent, source_id in bernini_sources[i]:
                        source_latent = source_latent.to(device=device, dtype=self.patch_embedding.weight.dtype)
                        source_tokens_i = self.patch_embedding(source_latent).to(modulation_dtype).flatten(2).transpose(1, 2)
                        source_freqs = apply_rotary_source_id(get_rotary_pos_embed(tuple(source_latent.shape[2:]), device=source_latent.device), source_id, self.dim // self.num_heads)
                        source_freqs = (source_freqs[0].to(device), source_freqs[1].to(device))
                        source_tokens.append(source_tokens_i)
                        cos_parts.append(source_freqs[0])
                        sin_parts.append(source_freqs[1])
                    x = self.patch_embedding(x).to(modulation_dtype)
                    grid_sizes = x.shape[2:]
                    x = x.flatten(2).transpose(1, 2)
                    target_freqs = freqs if freqs is not None else get_rotary_pos_embed(tuple(grid_sizes), device=x.device)
                    target_freqs = (target_freqs[0].to(device), target_freqs[1].to(device))
                    bernini_freqs_list.append((torch.cat([target_freqs[0]] + cos_parts, dim=0), torch.cat([target_freqs[1]] + sin_parts, dim=0)))
                    bernini_output_slices.append(slice(0, x.shape[1]))
                    if source_tokens:
                        x = torch.cat([x] + source_tokens, dim=1)
                    source_tokens = cos_parts = sin_parts = None
                elif not steadydancer:
                    x = self.patch_embedding(x).to(modulation_dtype)
                    if camera_adapter_features is not None:
                        x += camera_adapter_features
                    if scail2_ref_mask is not None:
                        scail2_ref_mask = scail2_ref_mask.unsqueeze(0) if scail2_ref_mask.ndim == 4 else scail2_ref_mask
                        if scail2_ref_mask.shape[0] != bz:
                            scail2_ref_mask = scail2_ref_mask.expand(bz, -1, -1, -1, -1)
                        x += self.mask_patch_embedding(scail2_ref_mask.to(device=device, dtype=self.mask_patch_embedding.weight.dtype)).to(modulation_dtype)
                    if kiwi_source_condition is not None:
                        source_cond = kiwi_source_condition.to(modulation_dtype)
                        if source_cond.shape[2:] != x.shape[2:]:
                            source_cond_full = torch.zeros_like(x)
                            t_len = min(source_cond.shape[2], x.shape[2])
                            source_cond_full[:, :, :t_len] = source_cond[:, :, :t_len] 
                            source_cond = source_cond_full
                        sigma = (t.flatten()[0] if t.numel() > 0 else 1000.0) / 1000.0
                        sigma = sigma.to(device=x.device, dtype=modulation_dtype)
                        x += source_cond * sigma
                    if kiwi_ref_condition is not None:
                        ref_cond = kiwi_ref_condition.to(modulation_dtype)
                        real_latent_frames = int(x.shape[2])
                        ref_latent_frames = int(ref_cond.shape[2])
                        if kiwi_ref_pad_first:
                            output_slice = slice(ref_latent_frames, ref_latent_frames + real_latent_frames)
                            x = torch.cat([ref_cond, x], dim=2)
                        else:
                            output_slice = slice(0, real_latent_frames)
                            x = torch.cat([x, ref_cond], dim=2)
                    grid_sizes = x.shape[2:]
                x_list[i] = x
        y = y_list = None
        
        if steadydancer: # steady dancer
            # Spatial Structure Adaptive Extractor.
            time_steps = steadydancer_condition[0].shape[2]
            for i, (x, condition) in enumerate(zip(x_list, steadydancer_condition)):
                # Temporal Motion Coherence Module.
                condition_temporal =self.condition_embedding_temporal(condition)                
                condition_spatial = rearrange(self.condition_embedding_spatial(rearrange(condition, 'b c t h w -> (b t) c h w')), '(b t) c h w -> b c t h w', t=time_steps, b=1)
                # Hierarchical Aggregation (1): condition, temporal condition, spatial condition
                condition_fused = condition + condition_temporal + condition_spatial
                # Frame-wise Attention Alignment Unit.
                condition_aligned = self.condition_embedding_align(condition_fused, x_noise_clone)                
                # Condition Fusion/Injection, Hierarchical Aggregation (2): x, fused condition, aligned condition
                x = self.patch_embedding_fuse(torch.cat([x, condition_fused, condition_aligned], 1).to(self.patch_embedding_fuse.weight.dtype))
                real_seq = math.prod(x.shape[2:])
                output_grid_sizes = x.shape[2:]
                x = torch.cat([x, self.patch_embedding(steadydancer_ref_x.unsqueeze(0).to(self.patch_embedding.weight.dtype )),
                                self.patch_embedding_ref_c(steadydancer_ref_c[:16].unsqueeze(0).to(self.patch_embedding_ref_c.weight.dtype ))], dim=2)
                grid_sizes = x.shape[2:]
                x_list[i] = x
                x = condition = condition_fused = condition_aligned = condition_temporal = condition_spatial = None
            x_noise_clone = x = None

        vista_condition_tokens = vista_cam_emb = None
        if vista_enabled:
            vista_grid_sizes = tuple(int(v) for v in grid_sizes)

            def vista_tensor(name):
                tensor = vista.get(name)
                if tensor is not None and tensor.device != device:
                    tensor = tensor.to(device)
                return tensor

            source_latents = vista_tensor("source_latents")
            source_masks = vista_tensor("source_mask_latents")
            point_latents = vista_tensor("point_latents")
            point_masks = vista_tensor("point_mask_latents")
            vista_source_tokens, _ = self.latent_encoder.source_patch_embedding(source_latents, source_masks, self.patch_embedding, vista_grid_sizes, "Vista4D source patch embedding")
            vista_point_tokens, _ = self.latent_encoder.point_cloud_patch_embedding(point_latents, point_masks, self.patch_embedding, vista_grid_sizes, "Vista4D point cloud patch embedding")
            vista_condition_tokens = torch.cat((vista_point_tokens.to(modulation_dtype), vista_source_tokens.to(modulation_dtype)), dim=1)
            if vista_condition_tokens.shape[0] != x_list[0].shape[0]:
                vista_condition_tokens = vista_condition_tokens.expand(x_list[0].shape[0], -1, -1)
            vista_cam_emb = vista_tensor("cam_emb")
            if vista_cam_emb is not None:
                vista_cam_emb = rearrange(vista_cam_emb, "b f h w d -> b (f h w) d").repeat(1, 3, 1).to(modulation_dtype)
            real_seq = math.prod(vista_grid_sizes)
            output_grid_sizes = grid_sizes
            source_latents = source_masks = point_latents = point_masks = vista_source_tokens = vista_point_tokens = None

        if taomate_reference is not None:
            reference_tokens = self.spark_ref_proj(next(reference for reference in taomate_reference if reference is not None)).to(modulation_dtype)
            for i, reference in enumerate(taomate_reference):
                if reference is not None:
                    x_list[i] += reference_tokens
            reference_tokens = None

        motion_vec_list = []
        pose_tokens = None
        scail2_pose_tokens = None
        if scail_pose_latents is not None:
            pose_tokens = build_scail_pose_tokens(self, scail_pose_latents, modulation_dtype)
        if scail2_pose_latents is not None:
            scail2_pose_tokens = build_scail2_pose_tokens(self, scail2_pose_latents, scail2_driving_masks, modulation_dtype)
        
        if face_pixel_values is None: face_pixel_values =  [None] * len(x_list)
        for i, (x, one_face_pixel_values) in enumerate(zip(x_list, face_pixel_values)):
                # animate/scail embeddings
                motion_vec = None
                if pose_latents is not None: 
                    x, motion_vec = after_patch_embedding(self, x, pose_latents, torch.zeros_like(face_pixel_values[0]) if one_face_pixel_values is None else one_face_pixel_values)
                motion_vec_list.append(motion_vec)
                if bernini_enabled:
                    pass
                elif chipmunk:
                    x = x.unsqueeze(-1)
                    x_og_shape = x.shape
                    x = voxel_chunk_no_padding(x, voxel_shape).squeeze(-1).transpose(1, 2)
                else:
                    x = x.flatten(2).transpose(1, 2)
                    if self.taomate:
                        x = x.contiguous()  # the rows in order: the fused INT8 kernels write into them

                if vista_condition_tokens is not None:
                    x = torch.cat((x, vista_condition_tokens), dim=1)

                if scail_pose_latents is not None:
                    if pose_tokens.shape[0] != x.shape[0]: pose_tokens = pose_tokens.repeat(x.shape[0], 1, 1)
                    x = torch.cat([x, pose_tokens], dim=1)
                if scail2_pose_tokens is not None:
                    current_pose_tokens = scail2_pose_tokens if scail2_pose_tokens.shape[0] == x.shape[0] else scail2_pose_tokens.repeat(x.shape[0], 1, 1)
                    x = torch.cat([x, current_pose_tokens], dim=1)

                x_list[i] = x
        x = None
        vista_condition_tokens = None

        animate2_ref_hidden = animate2_ref_grid_sizes = None
        if animate2_enabled:
            if animate2_cached:
                animate2_ref_x = animate2_ref_y = None
                animate2_ref_grid_sizes = animate2_cache["grid_sizes"]
            else:
                animate2_ref_input = torch.cat([animate2_ref_x, animate2_ref_y], dim=1)
                animate2_ref_x = animate2_ref_y = None
                animate2_ref_hidden = self.patch_embedding(animate2_ref_input.to(self.patch_embedding.weight.dtype)).to(modulation_dtype)
                animate2_ref_input = None
                animate2_ref_grid_sizes = animate2_ref_hidden.shape[2:]
                animate2_ref_hidden = animate2_ref_hidden.flatten(2).transpose(1, 2)
                if animate2_cache is not None:
                    animate2_cache.update({"grid_sizes": animate2_ref_grid_sizes, "blocks": []})

        shotplan_keep_mask = None
        if self.shotplan and shotplan_cut_frames:
            if offload.shared_state.get("_radial", False):
                raise RuntimeError("ShotPlan planning tokens are not compatible with radial attention.")
            base_freqs = freqs
            for index, hidden_states in enumerate(x_list):
                x_list[index], shotplan_freqs, keep_mask = inject_shotplan_tokens(hidden_states, base_freqs, self.hardcut_embedding, shotplan_cut_frames, grid_sizes, self.vae_scale)
                if index == 0:
                    freqs = shotplan_freqs
                    shotplan_keep_mask = keep_mask



        block_mask = None
        if causal_attention and causal_block_size > 0 and False: # NEVER WORKED
            frame_num = grid_sizes[0]
            height = grid_sizes[1]
            width = grid_sizes[2]
            block_num = frame_num // causal_block_size
            range_tensor = torch.arange(block_num).view(-1, 1)
            range_tensor = range_tensor.repeat(1, causal_block_size).flatten()
            causal_mask = range_tensor.unsqueeze(0) <= range_tensor.unsqueeze(1)  # f, f
            causal_mask = causal_mask.view(frame_num, 1, 1, frame_num, 1, 1).to(x[0].device)
            causal_mask = causal_mask.repeat(1, height, width, 1, height, width)
            causal_mask = causal_mask.reshape(frame_num * height * width, frame_num * height * width)
            block_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            del causal_mask

        attention_grid_sizes = (int(grid_sizes[0]) * 3, int(grid_sizes[1]), int(grid_sizes[2])) if vista_enabled else grid_sizes
        if vista_cam_emb is not None:
            cam_emb = vista_cam_emb

        offload.shared_state["embed_sizes"] = attention_grid_sizes 
        offload.shared_state["step_no"] = current_step_no 
        offload.shared_state["max_steps"] = max_steps
        # arguments

        kwargs = dict(
            grid_sizes=attention_grid_sizes,
            freqs=freqs,
            cam_emb = cam_emb,
            block_mask = block_mask,
            audio_proj=audio_proj,
            audio_context_lens=audio_context_lens,
            ref_images_count=ref_images_count,
            lynx_ip_scale= lynx_ip_scale,
            lynx_ref_scale = lynx_ref_scale,
            lynx_feature_extractor = lynx_feature_extractor,            
            )

        _flag_df = t.dim() == 2

        time_positions = torch.cat([t.flatten(), torch.ones_like(t.flatten()[:1])]) if animate2_enabled else t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, time_positions).to(modulation_dtype)  # self.patch_embedding.weight.dtype)
        )  # b, dim
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).to(e.dtype)

        animate2_ref_e0 = None
        if animate2_enabled:
            animate2_ref_e0 = e0[-1:]
            e, e0 = e[:-1], e0[:-1]

        standin_x = None
        if standin_ref is not None:
            standin_cache_enabled = False
            kwargs["standin_phase"] = 2
            if current_step_no == 0 or not standin_cache_enabled :
                standin_x = self.patch_embedding(standin_ref).to(modulation_dtype).flatten(2).transpose(1, 2)
                standin_e = self.time_embedding( sinusoidal_embedding_1d(self.freq_dim, torch.zeros_like(t)).to(modulation_dtype) )
                standin_e0 = self.time_projection(standin_e).unflatten(1, (6, self.dim)).to(e.dtype)
                standin_e = standin_ref = None
        
        if lynx_ip_embeds is None:
            lynx_ip_embeds_list = [None] * len(x_list)
        else:
            lynx_ip_embeds_list = lynx_ip_embeds

        if lynx_ref_buffer is None:
            lynx_ref_buffer_list = [None] * len(x_list)
        else:
            lynx_ref_buffer_list = lynx_ref_buffer


        if self.inject_sample_info and fps!=None:
            fps = torch.tensor(fps, dtype=torch.long, device=device)

            fps_emb = self.fps_embedding(fps).to(e.dtype) 
            if _flag_df:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)
            else:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

        if not (self.any_kiwi_source or self.any_kiwi_ref or self.taomate):
            context = [self.text_embedding(u) for u in context]

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            if steadydancer_clip_fea_c is not None:
                context_clip += self.img_emb(steadydancer_clip_fea_c)  # bs x 257 x dim

            context_list = []
            for one_context in context: 
                if len(one_context) != len(context_clip):
                    context_list.append( torch.cat( [context_clip.repeat(len(one_context), 1, 1), one_context ], dim=1 ))
                else:
                    context_list.append( torch.cat( [context_clip, one_context ], dim=1 ))
        else:
            context_list = context

        animate2_ref_context_emb = None
        if animate2_enabled and not animate2_cached:
            animate2_ref_context_emb = self.text_embedding(animate2_ref_context)
            animate2_ref_context = None
            animate2_ref_clip_emb = self.img_emb(animate2_ref_clip_fea)
            animate2_ref_clip_fea = None
            animate2_ref_context_emb = torch.cat([animate2_ref_clip_emb, animate2_ref_context_emb], dim=1)
            animate2_ref_clip_emb = None
        elif animate2_enabled:
            animate2_ref_context = animate2_ref_clip_fea = None

        if multitalk_audio != None:
            multitalk_audio_list = []
            for audio in multitalk_audio:
                if audio is not None:
                    audio = self.audio_proj(*audio) 
                    audio = torch.concat(audio.split(1), dim=2).to(context[0])
                multitalk_audio_list.append(audio)
            audio = None
        else:
            multitalk_audio_list = [None] * len(x_list)

        if multitalk_masks != None:
            multitalk_masks_list = multitalk_masks
        else:
            multitalk_masks_list = [None] * len(x_list)

        if audio_scale != None: 
            audio_scale_list = audio_scale
        else:
            audio_scale_list = [None] * len(x_list)


        if vace_context == None:
            hints_list = [None ] *len(x_list)
        else:
            # Vace embeddings
            c = [self.vace_patch_embedding(u.to(self.vace_patch_embedding.weight.dtype).unsqueeze(0)) for u in vace_context]
            c = [u.flatten(2).transpose(1, 2) for u in c]
            kwargs['context_scale'] = vace_context_scale
            hints_list = [ [ [sub_c] for sub_c in c] for _ in range(len(x_list)) ] 
            del c
        should_calc = True
        x_should_calc = None
        skips_steps_cache = self.cache
        if skips_steps_cache != None: 
            if skips_steps_cache.cache_type == "mag":
                if real_step_no <= skips_steps_cache.start_step:
                    should_calc = True
                elif skips_steps_cache.one_for_all and x_id != 0: # not joint pass, not main pas, one for all
                    assert len(x_list) == 1
                    should_calc = skips_steps_cache.should_calc
                else:
                    x_should_calc = []
                    for i in range(1 if skips_steps_cache.one_for_all else len(x_list)):
                        cur_x_id = i if joint_pass else x_id  
                        cur_mag_ratio = skips_steps_cache.mag_ratios[real_step_no * 2 + cur_x_id] # conditional and unconditional in one list
                        skips_steps_cache.accumulated_ratio[cur_x_id] *= cur_mag_ratio # magnitude ratio between current step and the cached step
                        skips_steps_cache.accumulated_steps[cur_x_id] += 1 # skip steps plus 1
                        cur_skip_err = np.abs(1-skips_steps_cache.accumulated_ratio[cur_x_id]) # skip error of current steps
                        skips_steps_cache.accumulated_err[cur_x_id] += cur_skip_err # accumulated error of multiple steps
                        if skips_steps_cache.accumulated_err[cur_x_id]<skips_steps_cache.magcache_thresh and skips_steps_cache.accumulated_steps[cur_x_id]<=skips_steps_cache.magcache_K:
                            skip_forward = True
                            if i == 0 and x_id == 0: skips_steps_cache.skipped_steps += 1
                            # print(f"skip: step={current_step} for x_id={cur_x_id}, accum error {skips_step_cache.accumulated_err[cur_x_id]}")
                        else:
                            skip_forward = False
                            skips_steps_cache.accumulated_err[cur_x_id], skips_steps_cache.accumulated_steps[cur_x_id], skips_steps_cache.accumulated_ratio[cur_x_id] = 0, 0, 1.0
                        x_should_calc.append(not skip_forward)
                    if skips_steps_cache.one_for_all:
                        should_calc = skips_steps_cache.should_calc = x_should_calc[0] 
                        x_should_calc = None
            else:
                if x_id != 0:
                    should_calc = skips_steps_cache.should_calc
                else:
                    if real_step_no <= skips_steps_cache.start_step or real_step_no == skips_steps_cache.num_steps-1 or skips_steps_cache.previous_modulated_input is None:
                        should_calc = True
                        skips_steps_cache.accumulated_rel_l1_distance = 0
                    else:
                        rescale_func = np.poly1d(skips_steps_cache.coefficients)
                        delta = abs(rescale_func(((e-skips_steps_cache.previous_modulated_input).abs().mean() / skips_steps_cache.previous_modulated_input.abs().mean()).cpu().item()))
                        skips_steps_cache.accumulated_rel_l1_distance += delta
                        if skips_steps_cache.accumulated_rel_l1_distance < skips_steps_cache.rel_l1_thresh:
                            should_calc = False
                            skips_steps_cache.skipped_steps += 1
                            # print(f"Teacache Skipped Step no {current_step} ({skips_step_cache.cache_skipped_steps}/{current_step}), delta={delta}" )
                        else:
                            should_calc = True
                            skips_steps_cache.accumulated_rel_l1_distance = 0
                    skips_steps_cache.previous_modulated_input = e 
                    skips_steps_cache.should_calc = should_calc

        if x_should_calc  == None: x_should_calc = [should_calc] * len(x_list) 

        if joint_pass:
            for i, x in enumerate(x_list):
                if not x_should_calc[i]: x += skips_steps_cache.previous_residual[i]
        elif not x_should_calc[0]:
            x = x_list[0]
            x += skips_steps_cache.previous_residual[x_id]
        x = None

        if skips_steps_cache != None:
            if skips_steps_cache.previous_residual == None: skips_steps_cache.previous_residual = [ None ] * len(x_list)
    
            if joint_pass:
                for i, should_calc in enumerate(x_should_calc):
                    if should_calc: skips_steps_cache.previous_residual[i] = None
            elif x_should_calc[0]:
                skips_steps_cache.previous_residual[x_id] = None
            ori_hidden_states = [ None ] * len(x_list)
            if all(x_should_calc):
                ori_hidden_states[0] = x_list[0].clone()
                for i in range(1, len(x_list)):
                    ori_hidden_states[i] = ori_hidden_states[0] if is_source_x[i] else x_list[i].clone()
            else:
                for i in range(len(x_list)):
                    if x_should_calc[i]: ori_hidden_states[i] = x_list[i].clone()

        if any(x_should_calc):
            for block_idx, block in enumerate(self.blocks):
                offload.shared_state["layer"] = block_idx
                if callback != None:
                    callback(-1, None, False, True)
                if pipeline._interrupt:
                    return [None] * len(x_list)

                if standin_x is not None:
                    if not standin_cache_enabled: get_cache("standin").clear()
                    standin_x = block(standin_x, context = None, grid_sizes = None, e= standin_e0, freqs = standin_freqs, standin_phase = 1)

                uni3c_block = block.uni3c if uni3c_x is not None and block_idx < self.uni3c_layers else None
                if uni3c_block is not None: # shared by all the guidance branches: the ControlNet does not see the text
                    uni3c_x = uni3c_block(uni3c_x, e, uni3c_condition["freqs"])

                if animate2_enabled:
                    animate2_generation = {
                        "x_list": x_list,
                        "contexts": context_list,
                        "should_calc": x_should_calc,
                        "unconditional": [i > 0 if joint_pass else x_id > 0 for i in range(len(x_list))],
                        "e": e0,
                        "grid_sizes": attention_grid_sizes,
                        "freqs": freqs,
                        "log_scale": animate2_log_scale,
                    }
                    if animate2_cached:
                        animate2_ref_handoff = [animate2_load_cached(animate2_cache["blocks"][block_idx], x_list[0].device, modulation_dtype)]
                        x_list = block(animate2_ref_handoff, e=animate2_ref_e0, context=None, grid_sizes=animate2_ref_grid_sizes, freqs=animate2_ref_freqs, animate2_generation=animate2_generation, animate2_cached=True)
                    else:
                        if animate2_cache is not None and animate2_kv_cache in ("INT8", "RAM"):  # the block stores its reference K/V
                            cache_device = "cpu" if animate2_kv_cache == "RAM" else animate2_ref_hidden.device
                            animate2_generation["store_kv"] = lambda ref_kv_list: animate2_cache["blocks"].append(animate2_store_kv(ref_kv_list, cache_device))
                        elif animate2_cache is not None:
                            animate2_cache["blocks"].append(animate2_ref_hidden.detach().to(animate2_ref_hidden.device, copy=True))
                        animate2_ref_handoff = [animate2_ref_hidden]
                        animate2_ref_hidden = None
                        animate2_ref_hidden, x_list = block(animate2_ref_handoff, e=animate2_ref_e0, context=animate2_ref_context_emb, grid_sizes=animate2_ref_grid_sizes, freqs=animate2_ref_freqs, animate2_generation=animate2_generation)
                    animate2_generation = None
                elif perturbation_layers is not None and block_idx in perturbation_layers:
                    if x_id != 0 or not x_should_calc[0]:
                        continue
                    x_list[0] = block(x_list[0], context = context_list[0], audio_scale= audio_scale_list[0], e= e0, **kwargs)
                else:
                    for i, (x, context, hints, audio_scale, multitalk_audio, multitalk_masks, should_calc, motion_vec, lynx_ip_embeds,lynx_ref_buffer) in enumerate(zip(x_list, context_list, hints_list, audio_scale_list, multitalk_audio_list, multitalk_masks_list, x_should_calc,motion_vec_list, lynx_ip_embeds_list,lynx_ref_buffer_list)):
                        if should_calc:
                            block_kwargs = kwargs
                            if bernini_enabled:
                                block_kwargs = dict(kwargs)
                                block_kwargs["freqs"] = bernini_freqs_list[i]
                            elif taomate_audio is not None:
                                block_kwargs = dict(kwargs, taomate_audio=taomate_audio[i])
                            x_list[i] = block(x, context = context, hints= hints, audio_scale= audio_scale, multitalk_audio = multitalk_audio, multitalk_masks =multitalk_masks, e= e0,  motion_vec = motion_vec, lynx_ip_embeds= lynx_ip_embeds, lynx_ref_buffer = lynx_ref_buffer, sub_x_no =i,  **block_kwargs)
                            del x
                    context = hints = None

                if uni3c_block is not None:
                    uni3c_hint = uni3c_block.proj_out(uni3c_x)
                    for i, should_calc in enumerate(x_should_calc):
                        if should_calc: x_list[i] += uni3c_hint
                    uni3c_hint = None

        if animate2_enabled:
            animate2_ref_hidden = animate2_ref_context_emb = animate2_ref_e0 = animate2_ref_freqs = None

        if skips_steps_cache != None:
            if joint_pass:
                if all(x_should_calc):                        
                    for i, (x, ori, is_source) in enumerate(zip(x_list, ori_hidden_states, is_source_x)) :
                        if i == 0 or is_source and i != last_x_idx  :
                            skips_steps_cache.previous_residual[i] = torch.sub(x, ori) 
                        else:
                            skips_steps_cache.previous_residual[i] = ori
                            torch.sub(x, ori, out=skips_steps_cache.previous_residual[i]) 
                        ori_hidden_states[i] = None
                else:
                    for i, (x, ori, is_source, should_calc) in enumerate(zip(x_list, ori_hidden_states, is_source_x, x_should_calc)) :
                        if should_calc:
                            skips_steps_cache.previous_residual[i] = ori
                            torch.sub(x, ori, out=skips_steps_cache.previous_residual[i]) 
                        ori_hidden_states[i] = None
                x , ori = None, None
            elif x_should_calc[0]:
                residual = ori_hidden_states[0] # just to have a readable code
                torch.sub(x_list[0], ori_hidden_states[0], out=residual)
                skips_steps_cache.previous_residual[x_id] = residual
            residual, ori_hidden_states = None, None
        if lynx_feature_extractor:
            return get_cache("lynx_ref_buffer")
        
        for i in range(len(x_list)):
            x = x_list[i]
            x_list[i] = None
            if chipmunk:
                x = reverse_voxel_chunk_no_padding(x.transpose(1, 2).unsqueeze(-1), x_og_shape, voxel_shape).squeeze(-1)
                x = x.flatten(2).transpose(1, 2)

            if shotplan_keep_mask is not None:
                x = x[:, shotplan_keep_mask]
            if bernini_enabled:
                x = x[:, bernini_output_slices[i]]
            elif real_seq > 0:
                x = x[:, :real_seq]

            # head
            head_input, x = [x], None
            x = self.head(head_input, e)

            # unpatchify
            x = self.unpatchify(x, output_grid_sizes if output_grid_sizes is not None else grid_sizes)
            if output_slice is not None:
                x = x[:, :, output_slice]
            x_list[i] = x
            x = None

        outputs = []
        for index in range(len(x_list)):
            x = x_list[index]
            x_list[index] = None
            outputs.append(x.float())
            x = None
        return outputs

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u in x:
            u = u[:math.prod(grid_sizes)].view(*grid_sizes, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
            out.append(u)
        if len(x) == 1:
            return out[0].unsqueeze(0)
        else:
            return torch.stack(out, 0)

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
