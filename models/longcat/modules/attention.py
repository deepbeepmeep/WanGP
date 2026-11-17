from typing import List, Optional

import torch
import torch.nn as nn

from shared.attention import pay_attention
from .rope_3d import RotaryPositionalEmbedding
from .blocks import RMSNorm_FP32, _take_tensor


def _run_attention(x_list, out_dtype, **attn_kwargs):
    q, k, v = x_list
    if out_dtype in (torch.float16, torch.bfloat16):
        attn_dtype = out_dtype
    else:
        attn_dtype = torch.bfloat16
    if q.dtype != attn_dtype:
        q = q.to(attn_dtype)
        k = k.to(attn_dtype)
        v = v.to(attn_dtype)
    x_list[:] = [q, k, v]
    del q, k, v
    attn_kwargs.setdefault("recycle_q", True)
    x = pay_attention(x_list, **attn_kwargs)
    if x.dtype != out_dtype:
        x = x.to(out_dtype)
    return x


def _run_sparse_attention(x_list, out_dtype, shape, bsa_params, **attn_kwargs):
    raise NotImplementedError("LongCat sparse/BSA attention is not wired to WanGP shared attention.")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(
            self.head_dim,
            cp_split_hw=cp_split_hw
        )

    def _process_attn(self, q, k, v, shape, out_dtype):
        """
            function wrapper to do attention with q, k, v
        """
        if self.enable_bsa:
            return _run_sparse_attention([q, k, v], out_dtype, shape, self.bsa_params)
        return _run_attention([q, k, v], out_dtype)

    def forward(self, x, shape=None, num_cond_latents=0, return_kv=False, layer_cache=None, rope=None):
        from .prefix_attention import prefix_attention
        output, ref_map = prefix_attention(
            self, x, shape, num_cond_latents, avatar=False, layer_cache=layer_cache,
            build_cache=return_kv, rope=rope)
        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
        ):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

    def _process_cross_attn(self, x, cond, kv_seqlen):
        x = _take_tensor(x)
        cond = _take_tensor(cond)
        B, N, C = x.shape
        assert C == self.dim and cond.shape[2] == self.dim
        out_dtype = x.dtype
        # Lengths are prepared as Python integers before the block graph scope.
        # Trim the usual single prompt before projecting: no device-to-host sync
        # or host tensor construction is allowed during CUDA graph capture.
        if B == 1 and kv_seqlen is not None:
            cond = cond[:, :kv_seqlen[0]]

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
        x = None
        if q.dtype != out_dtype:
            q = q.to(out_dtype)
        kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
        cond = None
        if kv.dtype != out_dtype:
            kv = kv.to(out_dtype)
        k, v = kv.unbind(2)
        v = v.contiguous()
        del kv

        q, k = self.q_norm(q), self.k_norm(k)

        if B > 1 and kv_seqlen is not None:
            x = torch.empty_like(q)
            for index, length in enumerate(kv_seqlen):
                x[index:index + 1] = _run_attention(
                    [q[index:index + 1], k[index:index + 1, :length], v[index:index + 1, :length]], out_dtype)
            q = k = v = None
        else:
            qkv_list = [q, k, v]
            del q, k, v
            x = _run_attention(qkv_list, out_dtype)
        x = x.view(B, N, C)
        x = self.proj(x)
        return x

    def forward_noise(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        x = _take_tensor(x)
        if num_cond_latents is None or num_cond_latents == 0:
            x_list = [x]
            x = None
            return 0, self._process_cross_attn(x_list, cond, kv_seqlen)
        assert shape is not None, "SHOULD pass in the shape"
        B, N, C = x.shape
        num_cond_latents_thw = num_cond_latents * (N // shape[0])
        x_noise = x[:, num_cond_latents_thw:]
        x = None
        x_list = [x_noise]
        x_noise = None
        return num_cond_latents_thw, self._process_cross_attn(x_list, cond, kv_seqlen)

    def forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        """
            x: [B, N, C]
            cond: [B, M, C]
        """
        x = _take_tensor(x)
        B, N, C = x.shape
        x_list = [x]
        x = None
        cond_tokens, output_noise = self.forward_noise(x_list, cond, kv_seqlen, num_cond_latents=num_cond_latents, shape=shape)
        if cond_tokens == 0:
            return output_noise
        output = output_noise.new_zeros(B, N, C)
        output[:, cond_tokens:].copy_(output_noise)
        return output
