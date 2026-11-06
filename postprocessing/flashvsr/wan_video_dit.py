# Adapted from FlashVSR (https://github.com/OpenImagingLab/FlashVSR), itself built on DiffSynth-Studio
# (https://github.com/modelscope/DiffSynth-Studio), Copyright (c) their authors, Apache License 2.0 (see docs/third_party_licenses/Apache-2.0.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn as nn
import math
import random
import os
import time
from typing import Tuple, Optional, List
from einops import rearrange
from .utils import hash_state_dict_keys

try:
    from block_sparse_attn import block_sparse_attn_func
    BLOCK_ATTN_AVAILABLE = True
except:
    BLOCK_ATTN_AVAILABLE = False

from .attention_backend import sparse_attention, staged_sparge_available
from shared import attention_kit
from shared.kernels import ops as kernels
from shared.attention import get_supported_attention_modes, pay_attention
from mmgp import auto_preload, offload
from PIL import Image
import numpy as np

USE_BLOCK_ATTN = False
STAGED_SPARGE = os.environ.get("WGP_FLASHVSR_STAGED", "1") != "0"  # debug comparison switch: 0 runs the SpargeAttn backend on whole q/k/v
_FLASHVSR_ATTENTION_MODE = None


def get_flashvsr_attention_mode():
    global _FLASHVSR_ATTENTION_MODE
    selected = offload.shared_state.get("_attention")
    if selected not in (None, "auto"):
        return selected
    if _FLASHVSR_ATTENTION_MODE is None:
        modes = get_supported_attention_modes()
        _FLASHVSR_ATTENTION_MODE = "sage2" if "sage2" in modes else "sage" if "sage" in modes else "sdpa"
        print(f"[FlashVSR] WanGP dense attention backend: {_FLASHVSR_ATTENTION_MODE}")
    return _FLASHVSR_ATTENTION_MODE

# ----------------------------
# Local / window masks
# ----------------------------
@torch.no_grad()
def build_local_block_mask_shifted_vec(block_h: int,
                                       block_w: int,
                                       win_h: int = 6,
                                       win_w: int = 6,
                                       include_self: bool = True,
                                       device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = torch.clamp(r_all - r_half, 0, H - win_h)
    end_r   = start_r + win_h - 1
    start_c = torch.clamp(c_all - c_half, 0, W - win_w)
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask

@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(block_h: int,
                                                   block_w: int,
                                                   win_h: int = 6,
                                                   win_w: int = 6,
                                                   include_self: bool = True,
                                                   device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r   = start_r + win_h - 1
    start_c = c_all - c_half
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""
    @staticmethod
    def partition(x: torch.Tensor | list[torch.Tensor], win: Tuple[int, int, int]):
        if isinstance(x, list):
            x_list = x
            x = x_list[0]
            x_list.clear()
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.reshape(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor | list[torch.Tensor], win: Tuple[int, int, int], orig: Tuple[int, int, int]):
        if isinstance(windows, list):
            windows_list = windows
            windows = windows_list[0]
            windows_list.clear()
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.reshape(B, F, H, W, -1)


@torch.no_grad()
def _topk_threshold_mask(flat: torch.Tensor, apply_topk: int) -> torch.Tensor:
    if apply_topk <= 0:
        return torch.zeros_like(flat, dtype=torch.bool)
    threshold_index = flat.shape[1] - apply_topk
    thresholds = flat.kthvalue(threshold_index, dim=1).values.unsqueeze_(1)
    mask = flat > thresholds
    del thresholds
    return mask


@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                              qk_list, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    q_w, k_w = qk_list
    qk_list.clear()
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_k = torch.mean(k_w, dim=1)
    del q_w, k_w
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads)
    scores.mul_(1 / math.sqrt(D))

    repeat_head = scores.shape[0]
    local_h, local_w = local_attn_mask.shape
    repeat_len = scores.shape[1] // local_h
    repeat_num = scores.shape[2] // local_w
    scores = scores.reshape(repeat_head, repeat_len, local_h, repeat_num, local_w)
    scores.masked_fill_(~local_attn_mask.view(1, 1, local_h, 1, local_w), -float('inf'))
    scores = scores.reshape(repeat_head, repeat_len * local_h, repeat_num * local_w)

    attn_map = torch.softmax(scores, dim=-1)
    del scores
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1]-1, topk)
    mask_new = _topk_threshold_mask(flat, apply_topk)
    del flat, attn_map
    mask_new = mask_new.reshape(nheads, seqlen * s1, s2)
    return mask_new.unsqueeze(0)


@torch.no_grad()
def _draft_block_mask(q_heads, k_heads_doubled, seqlen, topk, local_attn_mask, groups=None):
    """Block map (1, heads, query windows, key half-windows) int8 of the window-pooled q_heads (heads, windows, d) against the
    half-window-pooled k_heads_doubled (heads, 2 x key windows, d): local windows only, the top-k entries of every temporal slab
    (one slab per temporal window, or `groups` slabs of several windows: FlashVSR+ ranks its budget over a 2-window chunk)."""
    nheads, D = q_heads.shape[0], q_heads.shape[-1]
    k_heads_1, k_heads_2 = torch.chunk(k_heads_doubled, 2, dim=1)
    scores_1 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_1)
    scores_1.mul_(1 / math.sqrt(D))
    scores_2 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_2)
    scores_2.mul_(1 / math.sqrt(D))
    scores = torch.cat([scores_1, scores_2], dim=-1)
    del scores_1, scores_2

    repeat_head = scores.shape[0]
    local_h, local_w = local_attn_mask.shape
    repeat_len = scores.shape[1] // local_h
    repeat_num = (scores.shape[2] // 2) // local_w
    scores = scores.reshape(repeat_head, repeat_len, local_h, repeat_num, local_w, 2)
    scores.masked_fill_(~local_attn_mask.view(1, 1, local_h, 1, local_w, 1), -float('inf'))
    scores = scores.reshape(repeat_head, repeat_len * local_h, repeat_num * local_w * 2)

    attn_map = torch.softmax(scores, dim=-1)
    del scores
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen if groups is None else groups)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1]-1, topk)

    mask_new = _topk_threshold_mask(flat, apply_topk)
    del flat, attn_map
    mask_new = mask_new.reshape(nheads, -1, s2)
    mask_new = mask_new.to(torch.int8)
    return mask_new.unsqueeze(0)


@torch.no_grad()
def structural_block_candidates(block_h, block_w, seqlen, cached_windows, local_range, device=None):
    """The candidate relation of a local attention layer (FlashVSR+ hybrid attention): every key window of the cached and current
    temporal windows within the local_range x local_range neighborhood ("normal slide", no clamping at the borders) of each query
    window. Returns the (query windows, key windows) bool map, its candidate count and its entry count (upstream's window-level
    top-k budget applies to the entries of the whole chunk)."""
    local = build_local_block_mask_shifted_vec_normal_slide(block_h, block_w, local_range, local_range, include_self=True, device=device)
    key_windows = cached_windows + seqlen
    candidates = local.repeat(seqlen, key_windows)
    return candidates, int(local.sum().item()) * seqlen * key_windows, candidates.numel()


@torch.no_grad()
def structural_block_mask(block_h, block_w, seqlen, cached_windows, local_range, nheads, device):
    """The int8 block map (1, heads, query windows, key half-windows) of structural_block_candidates, the layout the sparse kernels
    consume (two 64-token key blocks per window), the same for every head."""
    candidates, _, _ = structural_block_candidates(block_h, block_w, seqlen, cached_windows, local_range, device)
    return candidates.repeat_interleave(2, dim=1).to(torch.int8).unsqueeze(0).expand(nheads, -1, -1).contiguous().unsqueeze(0)


@torch.no_grad()
def generate_draft_block_mask_sage(batch_size, nheads, seqlen,
                                      qk_list, topk=10, local_attn_mask=None, groups=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"

    q_w, k_w = qk_list
    qk_list.clear()
    avgpool_q = torch.mean(q_w, dim=1)
    del q_w
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)

    k_w_split = k_w.view(k_w.shape[0], 2, 64, k_w.shape[2])
    avgpool_k_split = torch.mean(k_w_split, dim=2)
    del k_w, k_w_split
    avgpool_k_refined = rearrange(avgpool_k_split, 's two d -> (s two) d', two=2) # shape: (s*2, C)
    avgpool_k_refined = rearrange(avgpool_k_refined, 's (h d) -> s h d', h=nheads) # shape: (s*2, h, d)
    k_heads_doubled = avgpool_k_refined.permute(1, 0, 2) # shape: (h, s*2, d)
    return _draft_block_mask(q_heads, k_heads_doubled, seqlen, topk, local_attn_mask, groups=groups)


# ----------------------------
# Attention kernels
# ----------------------------
def flash_attention(qkv_list: list[torch.Tensor], num_heads: int, compatibility_mode=False, attention_mask_list=None, return_KV=False):
    q, k, v = qkv_list
    qkv_list.clear()
    if isinstance(attention_mask_list, list):
        attention_mask = attention_mask_list[0]
        attention_mask_list.clear()
    else:
        attention_mask = None
    if attention_mask is not None:
        seqlen = q.shape[1]
        seqlen_kv = k.shape[1]
        if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        max_seqlen_q_ = seqlen
        max_seqlen_k_ = seqlen_kv
        p_dropout = 0.0
        if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
            head_mask_type = torch.tensor([1]*num_heads, device=q.device, dtype=torch.int32)
            streaming_info = None
            x = block_sparse_attn_func(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                head_mask_type,
                streaming_info,
                attention_mask,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                deterministic=False,
                softmax_scale=None,
                is_causal=False,
                exact_streaming=False,
                return_attn_probs=False,
            ).unsqueeze(0)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            qkv_list = [q, k, v]
            mask_list = [attention_mask]
            del q, k, v, attention_mask
            x = sparse_attention(qkv_list, mask_list, recycle_q=True)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        qkv_list = [q, k, v]
        del q, k, v
        x = pay_attention(qkv_list, force_attention="sdpa" if compatibility_mode else get_flashvsr_attention_mode(), recycle_q=True)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x.mul_(scale.add_(1)).add_(shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def _rope_axis_inplace(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    scratch = x_even.clone()
    x_even.mul_(cos).addcmul_(x_odd, sin, value=-1)
    x_odd.mul_(cos).addcmul_(scratch, sin)
    del scratch


def rope_apply(x_list, freqs, num_heads, f: int, h: int, w: int):
    x = x_list[0]
    x_list.clear()
    b = x.shape[0]
    x = x.view(b, f, h, w, num_heads, -1)
    f_freqs, h_freqs, w_freqs = freqs
    f_dim = f_freqs[0].shape[-1] * 2
    h_dim = h_freqs[0].shape[-1] * 2
    _rope_axis_inplace(x[..., :f_dim], f_freqs[0].view(1, f, 1, 1, 1, -1), f_freqs[1].view(1, f, 1, 1, 1, -1))
    _rope_axis_inplace(x[..., f_dim:f_dim + h_dim], h_freqs[0].view(1, 1, h, 1, 1, -1), h_freqs[1].view(1, 1, h, 1, 1, -1))
    _rope_axis_inplace(x[..., f_dim + h_dim:], w_freqs[0].view(1, 1, 1, w, 1, -1), w_freqs[1].view(1, 1, 1, w, 1, -1))
    return x.reshape(b, f * h * w, -1)


def rope_apply_windowed(x_list, freqs, num_heads, f: int, h: int, w: int, win: Tuple[int, int, int], batch_size: int):
    x = x_list[0]
    x_list.clear()
    wf, wh, ww = win
    nf, nh, nw = f // wf, h // wh, w // ww
    x = x.view(batch_size, nf, nh, nw, wf, wh, ww, num_heads, -1)
    f_freqs, h_freqs, w_freqs = freqs
    f_dim = f_freqs[0].shape[-1] * 2
    h_dim = h_freqs[0].shape[-1] * 2
    _rope_axis_inplace(x[..., :f_dim], f_freqs[0].view(nf, wf, -1).view(1, nf, 1, 1, wf, 1, 1, 1, -1), f_freqs[1].view(nf, wf, -1).view(1, nf, 1, 1, wf, 1, 1, 1, -1))
    _rope_axis_inplace(x[..., f_dim:f_dim + h_dim], h_freqs[0].view(nh, wh, -1).view(1, 1, nh, 1, 1, wh, 1, 1, -1), h_freqs[1].view(nh, wh, -1).view(1, 1, nh, 1, 1, wh, 1, 1, -1))
    _rope_axis_inplace(x[..., f_dim + h_dim:], w_freqs[0].view(nw, ww, -1).view(1, 1, 1, nw, 1, 1, ww, 1, -1), w_freqs[1].view(nw, ww, -1).view(1, 1, 1, nw, 1, 1, ww, 1, -1))
    return x.view(batch_size * nf * nh * nw, wf * wh * ww, -1)


# ----------------------------
# Norms & Blocks
# ----------------------------
def rms_norm_inplace(x: torch.Tensor | list[torch.Tensor], norm: nn.RMSNorm) -> torch.Tensor:
    if isinstance(x, list):
        x_list = x
        x = x_list[0]
        x_list.clear()
    inv_rms = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True, dtype=torch.float32)
    inv_rms.pow_(2).div_(x.shape[-1]).add_(norm.eps).rsqrt_()
    x.mul_(inv_rms.to(dtype=x.dtype))
    weight = norm.weight if norm.weight.dtype == x.dtype else norm.weight.to(dtype=x.dtype)
    x.mul_(weight)
    return x


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, qkv_list, attention_mask_list=None):
        x = flash_attention(qkv_list, num_heads=self.num_heads, attention_mask_list=attention_mask_list)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def forward(self, x, freqs, f=None, h=None, w=None, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, stream_cache=None, local_range = 9, cache_next=True,
                allow_short_start=False, topk_groups=None, block_mask=None):
        """topk_groups: top-k slabs per chunk (None: one per temporal window). block_mask: a structural (1, heads, query windows, key
        half-windows) int8 map that replaces the draft top-k (FlashVSR+ local layers). stream_cache: the stream's StreamCache, whose keys
        and values of the block (from the previous chunk) are attended before the chunk's own, and which keeps the block's last kv_len
        windows for the next chunk (cache_next)."""
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        prefix = None if stream_cache is None else stream_cache.prefix(block_id)
        B, L, D = x.shape
        win = (2, 8, 8)
        assert f % win[0] == 0, "stream chunks hold whole temporal windows"
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        x_w = WindowPartition3D.partition([x.view(B, f, h, w, D)], win)
        x = None
        if block_mask is None:
            self._local_mask(h, w, local_range, x_w.device)
        if STAGED_SPARGE and staged_sparge_available():
            x_list = [x_w]
            x_w = None
            return self._forward_staged(x_list, freqs, f, h, w, win, B, topk, is_stream, stream_cache, block_id, prefix, kv_len, cache_next, topk_groups, block_mask)

        v_w = self.v(x_w)

        q_w = self.q(x_w)        
        q_w = rms_norm_inplace(q_w, self.norm_q)
        q_list = [q_w]
        q_w = None
        q_w = rope_apply_windowed(q_list, freqs, self.num_heads, f, h, w, win, B)

        k_w = self.k(x_w)        
        del x_w
        k_w = rms_norm_inplace(k_w, self.norm_k)
        k_list = [k_w]
        k_w = None
        k_w = rope_apply_windowed(k_list, freqs, self.num_heads, f, h, w, win, B)

        seqlen = f//win[0]
        one_len = k_w.shape[0] // B // seqlen
        if prefix is not None:
            k_w = torch.cat([prefix[0].view(-1, k_w.shape[1], D), k_w], dim=0)
            v_w = torch.cat([prefix[1].view(-1, v_w.shape[1], D), v_w], dim=0)
            prefix = None

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        if is_stream and cache_next:  # the last windows of the joined keys and values (the prefix copied into them), kept for the next chunk
            cache_blocks = min(block_n_kv, one_len * max(1, int(kv_len)))
            for target, joined in zip(stream_cache.window(block_id, cache_blocks * block_s), (k_w, v_w)):
                target[0].copy_(joined.view(B, block_n_kv, block_s, D)[0, -cache_blocks:].view(cache_blocks * block_s, self.num_heads, self.head_dim))
            stream_cache.commit(block_id, cache_blocks * block_s)

        reorder_q = rearrange(q_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n, block_s=block_s)
        reorder_k = rearrange(k_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)
        reorder_v = rearrange(v_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)

        del v_w

        if block_mask is not None:  # structural map: no pooled scores, no top-k
            q_w = k_w = None
            attention_mask = block_mask
        elif USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:
            qk_list = [q_w, k_w]
            q_w = k_w = None
            attention_mask = generate_draft_block_mask(B, self.num_heads, seqlen, qk_list, topk=topk, local_attn_mask=self.local_attn_mask)
        else:
            qk_list = [q_w, k_w]
            q_w = k_w = None
            attention_mask = generate_draft_block_mask_sage(B, self.num_heads, seqlen, qk_list, topk=topk, local_attn_mask=self.local_attn_mask, groups=topk_groups)
        qkv_list = [reorder_q, reorder_k, reorder_v]
        attention_mask_list = [attention_mask]
        del reorder_q, reorder_k, reorder_v, attention_mask
        x = self.attn(qkv_list, attention_mask_list)

        x = rearrange(x, 'b (block_n block_s) d -> (b block_n) (block_s) d', block_n=block_n, block_s=block_s)
        x = WindowPartition3D.reverse([x], win, (f, h, w))
        x = x.view(B, f*h*w, D)
        return self.o(x)

    def _local_mask(self, h, w, local_range, device):
        if self.local_attn_mask is None or self.local_attn_mask_h!=h//8 or self.local_attn_mask_w!=w//8 or self.local_range!=local_range:
            self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(h//8, w//8, local_range, local_range, include_self=True, device=device)
            self.local_attn_mask_h = h//8
            self.local_attn_mask_w = w//8
            self.local_range = local_range

    def _forward_staged(self, x_list, freqs, f, h, w, win, B, topk, is_stream, stream_cache, block_id, prefix, kv_len, cache_next, topk_groups=None, structural_mask=None):
        """The attention through the attention kit: q, k and v projected by head groups from one prepared input (the windows handed
        off in x_list), each quantized for the SpargeAttn block-sparse kernel as soon as it is ready (k before q is projected), the
        block map, the window RoPE and the stream cache slices computed per group. Same math as the whole-tensor path: RMS norm over
        all heads, windowed RoPE, window-pooled block map (or the head slice of a structural map)."""
        assert B == 1, "Only batch_size=1 supported for now"
        heads, head_dim, D = self.num_heads, self.head_dim, self.dim
        block_s = win[0] * win[1] * win[2]
        x_w = x_list[0]
        x_list.clear()
        block_n = x_w.shape[0] // B
        seqlen = f // win[0]
        one_len = block_n // seqlen
        new_tokens = block_n * block_s
        block_n_kv = block_n + (0 if prefix is None else prefix[0].shape[1] // block_s)
        window = None
        if is_stream and cache_next:  # the last cache_blocks windows of the (prefixed) k and v, written per head group by the attention kit
            cache_blocks = min(block_n_kv, one_len * max(1, int(kv_len)))
            cache_tokens = cache_blocks * block_s
            window = (*stream_cache.window(block_id, cache_tokens), max(0, cache_tokens - new_tokens))
        norms = (self.norm_q, self.norm_k)

        def norm_rope(query, key, group):
            for which, tensor in enumerate((query, key)):
                if tensor is None:
                    continue
                norm = norms[which]
                if group.mean_squares is None:  # all heads: the row's mean square from the tensor itself
                    inv_rms = torch.linalg.vector_norm(tensor, ord=2, dim=(-2, -1), keepdim=True, dtype=torch.float32)
                    inv_rms.pow_(2).div_(D).add_(norm.eps).rsqrt_()
                else:
                    inv_rms = group.mean_squares[which].add(norm.eps).rsqrt_().unsqueeze(-1)
                tensor.mul_(inv_rms.to(dtype=tensor.dtype))
                tensor.mul_(norm.weight.view(heads, head_dim)[group.q].to(dtype=tensor.dtype))
                rope_apply_windowed([tensor.view(B * block_n, block_s, -1)], freqs, len(group.q), f, h, w, win, B)

        pooled = {}

        def block_mask(query, key, group):  # the key (with its prefix) comes first and is pooled per half window, then the query
            if structural_mask is not None:
                return None if query is None else structural_mask[:, group.q].contiguous()
            if key is not None:
                pooled["k"] = key.view(B * block_n_kv, 2, block_s // 2, -1, head_dim).mean(dim=2).permute(2, 0, 1, 3).reshape(len(group.kv), -1, head_dim)
            if query is None:
                return None
            q_heads = query.view(B * block_n, block_s, -1, head_dim).mean(dim=1).permute(1, 0, 2)
            return _draft_block_mask(q_heads, pooled.pop("k"), seqlen, topk, self.local_attn_mask, groups=topk_groups)

        x_list = [x_w.view(B, new_tokens, D)]
        x_w = None
        x = attention_kit.qkv_attention(x_list, self.q, self.k, self.v, heads, head_dim, norm_rope, norm_spans_heads=True,
                                        kv_prefix=prefix, attention_mask=block_mask, force_attention="sparge", kv_window=window)
        prefix = None
        if window is not None:  # the block's state for the next chunk written over the prefix it attended
            stream_cache.commit(block_id, window[0].shape[1])
            window = None
        x = WindowPartition3D.reverse([x.view(B * block_n, block_s, D)], win, (f, h, w)).view(B, f * h * w, D)
        return self.o(x)


class CrossAttention(nn.Module):
    """
    仅考虑文本 context；提供持久 KV 缓存。
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

        # 持久缓存
        self.cache_k = None
        self.cache_v = None

    @torch.no_grad()
    def init_cache(self, ctx: torch.Tensor):
        """ctx: [B, S_ctx, dim] —— 经过 text_embedding 之后的上下文"""
        self.cache_k = rms_norm_inplace([self.k(ctx)], self.norm_k)
        self.cache_v = self.v(ctx)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor | list[torch.Tensor], y: torch.Tensor, is_stream: bool = False):
        """
        y 即文本上下文（未做其他分支）。
        """
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        q = rms_norm_inplace([self.q(x)], self.norm_q)
        del x
        assert self.cache_k is not None and self.cache_v is not None
        k = self.cache_k
        v = self.cache_v

        x = self.attn([q, k, v])
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x.add_(residual.mul_(gate))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Training: the FFN by chunks of tokens * dim / ffn_dim tokens (inference uses the shared chunked kernels.mlp)."""
        chunk_size = max(1, min(x.shape[1], int(x.shape[1] * self.dim / self.ffn_dim)))
        return self.ffn(x) if chunk_size >= x.shape[1] else torch.cat([self.ffn(chunk) for chunk in x.split(chunk_size, dim=1)], dim=1)

    def forward(self, x, context, t_mod, freqs, f, h, w, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, stream_cache=None, local_range = 9, cache_next=True,
                allow_short_start=False, topk_groups=None, block_mask=None):
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x_list = [input_x]
        input_x = None
        self_attn_output = self.self_attn(
            x_list , freqs, f, h, w, local_num, topk, train_img, block_id,
            kv_len=kv_len, is_full_block=is_full_block, is_stream=is_stream,
            stream_cache=stream_cache, local_range = local_range, cache_next=cache_next,
            allow_short_start=allow_short_start, topk_groups=topk_groups, block_mask=block_mask)

        x = self.gate(x, gate_msa, self_attn_output)
        del self_attn_output
        cross_input = self.norm3(x)
        cross = self.cross_attn([cross_input], context, is_stream=is_stream)
        x.add_(cross)
        del cross, cross_input
        x_list = [modulate(self.norm2(x), shift_mlp, scale_mlp)]
        if self.training:
            x = self.gate(x, gate_mlp, self.forward_ffn(x_list.pop()))
        else:  # the FFN by chunks, its gated output added to x
            kernels.mlp(x_list, self.ffn[0], self.ffn[2], "gelu", residual=x, segments=kernels.gate_segments(gate_mlp, x.shape[0], x.shape[1]))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x.add_(self.emb_pos.to(dtype=x.dtype, device=x.device))
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.head(x)


# ----------------------------
# WanModel (no image branch) — init 时即产生 KV 缓存
# ----------------------------
class WanModel(torch.nn.Module):
    _stream_storing = False  # the stream's first chunk runs: it stores every block's cache

    def _auto_preload_workload(self, args, kwargs):
        """Workload of MMGP's automatic VRAM preload: by input size, the first chunk of a stream apart (the cache it stores would count as
        activations of every chunk)."""
        return auto_preload.workload(args, kwargs), self._stream_storing

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        # init_context: torch.Tensor,     # <<<< 必填：在 __init__ 里用它生成 cross-attn KV 缓存
        has_image_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size

        # patch embed
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # text / time embed
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self._cross_kv_initialized = False

    # 可选:手动清空 / 重新初始化
    def clear_cross_kv(self):
        for blk in self.blocks:
            blk.cross_attn.clear_cache()
        self._cross_kv_initialized = False

    @torch.no_grad()
    def reinit_cross_kv(self, new_context: torch.Tensor):
        ctx_txt = self.text_embedding(new_context)
        for blk in self.blocks:
            blk.cross_attn.init_cache(ctx_txt)
        self._cross_kv_initialized = True

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor | list[torch.Tensor], grid_size: torch.Tensor):
        if isinstance(x, list):
            x_ref = x
            x = x_ref[0]
            x_ref.clear()
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                LQ_latents: Optional[List[torch.Tensor]] = None,
                train_img: bool = False,
                topk_ratio: Optional[float] = None,
                kv_ratio: Optional[float] = None,
                local_num: Optional[int] = None,
                is_full_block: bool = False,
                causal_idx: Optional[int] = None,
                **kwargs,
                ):
        # time / text embeds
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 这里仍会嵌入 text（CrossAttention 若已有缓存会忽略它）
        # context = self.text_embedding(context)

        # 输入打补丁
        x, (f, h, w) = self.patchify(x)
        B = x.shape[0]

        # window / masks 超参
        win = (2, 8, 8)
        seqlen = f//win[0]
        if local_num is None:
            local_random = random.random()
            if local_random < 0.3:
                local_num = seqlen - 3
            elif local_random < 0.4:
                local_num = seqlen - 4
            elif local_random < 0.5:
                local_num = seqlen - 2
            else:
                local_num = seqlen

        window_size = win[0]*h*w//128
        square_num = window_size*window_size
        topk_ratio = 2.0
        topk = min(max(int(square_num*topk_ratio), 1), int(square_num*seqlen)-1)

        if kv_ratio is None:
            kv_ratio = (random.uniform(0., 1.0)**2)*(local_num-2-2)+2
        kv_len = min(max(int(window_size*kv_ratio), 1), int(window_size*seqlen)-1)

        decay_ratio = random.uniform(0.7, 1.0)

        freqs = tuple((freq.real.to(device=x.device, dtype=x.dtype), freq.imag.to(device=x.device, dtype=x.dtype)) for freq in (self.freqs[0][:f], self.freqs[1][:h], self.freqs[2][:w]))

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # blocks
        for block_id, block in enumerate(self.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                x.add_(LQ_latents[block_id])

            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, f, h, w, local_num, topk,
                            train_img, block_id, kv_len, is_full_block, False,
                            None,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, f, h, w, local_num, topk,
                        train_img, block_id, kv_len, is_full_block, False,
                        None,
                        use_reentrant=False,
                    )
            else:
                x = block([x], context, t_mod, freqs, f, h, w, local_num, topk,
                          train_img, block_id, kv_len, is_full_block, False,
                          None)

        x = self.head([x], t)
        x = self.unpatchify([x], (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    

# ----------------------------
# State dict converter（保持原映射；已忽略 has_image_input 使用）
# ----------------------------
class WanModelStateDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        # 保留原有哈希匹配返回的 config；实现本身不使用 has_image_input 分支
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 16,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 16,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 48,"dim": 1536,"ffn_dim": 8960,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 12,"num_layers": 30,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 48,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6}
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {"has_image_input": False,"patch_size": [1, 2, 2],"in_dim": 36,"dim": 5120,"ffn_dim": 13824,"freq_dim": 256,"text_dim": 4096,"out_dim": 16,"num_heads": 40,"num_layers": 40,"eps": 1e-6,"has_image_pos_emb": False}
        else:
            config = {}
        return state_dict, config
