# Adapted from FLUX (https://github.com/black-forest-labs/flux), Copyright (c) Black Forest Labs,
# Apache License 2.0 (see docs/third_party_licenses/Apache-2.0.txt). Modified for WanGP.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from shared import attention_kit
from shared.attention import pay_attention
from ..math import apply_rope_, attention, rope
from shared.kernels import ops

def get_linear_split_map(
    hidden_size: int = 3072,
    mlp_ratio: float = 4.0,
    single_linear1_mlp_ratio: float | None = None,
    linear1_mlp_ratio: float | None = None,
):
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    if linear1_mlp_ratio is None and mlp_ratio == 3.0:
        lin1_ratio = mlp_ratio * 2
    else:
        lin1_ratio = linear1_mlp_ratio if linear1_mlp_ratio is not None else mlp_ratio
    lin1_mlp = int(hidden_size * (single_linear1_mlp_ratio if single_linear1_mlp_ratio is not None else lin1_ratio))
    split_linear_modules_map = {
        "qkv": {"mapped_modules": ["q", "k", "v"], "split_sizes": [hidden_size, hidden_size, hidden_size]},
        "linear1": {
            "mapped_modules": ["linear1_attn_q", "linear1_attn_k", "linear1_attn_v", "linear1_mlp"],
            "split_sizes": [hidden_size, hidden_size, hidden_size, lin1_mlp],
        },
        "linear1_qkv": {
            "mapped_modules": ["linear1_attn_q", "linear1_attn_k", "linear1_attn_v"],
            "split_sizes": [hidden_size, hidden_size, hidden_size],
        },
    }
    return split_linear_modules_map


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)

class EmbedNDFlux2(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,
        )

        return emb.unsqueeze(1)
       

def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    @property
    def weight(self):
        return self.scale

    eps = 1e-6

    def forward(self, x: Tensor):
        x_list, x = [x], None
        return ops.rms_norm_rounded(x_list, manual=True) * self.scale



class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if k != None:
            return self.key_norm(k).to(v)
        else: 
            return self.query_norm(q).to(v)
        # q = self.query_norm(q)
        # k = self.key_norm(k)
        # return q.to(v), k.to(v)


def _norm_rope_(query, key, pe, segments):
    """attention_kit callback: the QKNorm of each segment of tokens (start, stop, QKNorm), then RoPE with pe (batch, 1, tokens, ...), of
    q and/or k (batch, tokens, heads, head_dim) in place, by chunks of tokens: the values of QKNorm and math.attention's RoPE."""
    pe = pe.transpose(1, 2)
    for tensor, is_key in ((query, False), (key, True)):
        if tensor is None:
            continue
        for start, stop, norm in segments:
            normalize = (lambda part: norm(None, part, part)) if is_key else (lambda part: norm(part, None, part))
            attention_kit.rows_(tensor, lambda part, first, last: apply_rope_([normalize(part)], pe[:, first:last]), start, stop)


def _fused_norm_rope_quant_(query, key, pe, segments, core, settings):
    """attention_kit's fused_quant: the QKNorm weights of each segment of tokens (RMSNorm: FP32 statistics, eps 1e-6, one rounding) and
    pe's rotation matrices (interleaved channel pairs) fused with the staged SageAttention 2 quantization of q and / or k."""
    return ops.split_half_norm_rope_quant_(query, key, pe.transpose(1, 2), [(start, stop, (norm.query_norm.scale, 1e-6)) for start, stop, norm in segments],
                                           [(start, stop, (norm.key_norm.scale, 1e-6)) for start, stop, norm in segments], core, settings, interleaved=True)


def _kv_norm_rope_(query, key, pe, segments):
    # FLUX rotation matrices' first column is the compact (cos, sin) table.
    rope = pe.transpose(1, 2)[..., 0]
    for start, stop, norm in segments:
        ops.interleaved_norm_rope_(None if query is None else query[:, start:stop], None if key is None else key[:, start:stop],
                                  rope[:, start:stop], norm.query_norm, norm.key_norm)


def _reference_attention(x_list, projections, norm, pe, heads, capture):
    """Klein KV references attend only to references, with their post-RoPE K/V borrowed by the body."""
    segments = ((0, x_list[0].shape[1], norm),)

    def attend(qkv, group):
        for index, tensor in enumerate(qkv[1:]):
            if group.kv.start == 0 and group.kv.stop == heads:
                capture[index] = tensor
            else:
                if capture[index] is None:
                    capture[index] = tensor.new_empty((*tensor.shape[:2], heads, tensor.shape[-1]))
                capture[index][:, :, group.kv.start:group.kv.stop].copy_(tensor)
        return pay_attention(qkv, recycle_q=True)

    return attention_kit.qkv_attention(x_list, *projections, heads, norm.query_norm.scale.numel(),
                                       lambda q, k, group: _kv_norm_rope_(q, k, pe, segments), attention_fn=attend).flatten(2)


def _norm_modulate(x, norm, mod):
    """norm(x) * (1 + mod.scale) + mod.shift through the shared segmented modulation, one row per batch item (mod tensors (batch or 1,
    1, dim)); x is borrowed."""
    batch, tokens = x.shape[:2]
    rows = mod.scale.shape[0]
    segments = [(b * tokens, (b + 1) * tokens, b if rows > 1 else 0) for b in range(batch)]
    return ops.norm_modulate([x.view(-1, x.shape[-1])], norm, mod.shift[:, 0], mod.scale[:, 0], segments).view(x.shape)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, proj_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        raise Exception("not implemented")

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

class ChromaModulationOut(ModulationOut):
    @classmethod
    def from_offset(cls, tensor: torch.Tensor, offset: int = 0):
        return cls(
            shift=tensor[:, offset : offset + 1, :],
            scale=tensor[:, offset + 1 : offset + 2, :],
            gate=tensor[:, offset + 2 : offset + 3, :],
        )


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, bias: bool = True):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=bias)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x_list, x = [x], None
        return ops.swiglu(x_list, in_place=False)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, shared_modulation = False, double_mlp_ratio: float | None = None, double_linear1_mlp_ratio: float | None = None, mod_bias: bool = True, mlp_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        lin1_ratio = double_linear1_mlp_ratio
        if lin1_ratio is None:
            base_ratio = double_mlp_ratio if double_mlp_ratio is not None else mlp_ratio
            lin1_ratio = base_ratio * 2 if base_ratio == 3.0 else base_ratio
        mlp_hidden_dim = int(hidden_size * (double_mlp_ratio if double_mlp_ratio is not None else mlp_ratio))
        lin1_mlp_dim = int(hidden_size * lin1_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.shared_modulation = shared_modulation
        self.mlp_activation = "swiglu" if double_linear1_mlp_ratio is not None else "gelu"  # of both MLPs, for the shared chunked FFN
        if not shared_modulation:
            self.img_mod = Modulation(hidden_size, double=True, bias=mod_bias)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if double_linear1_mlp_ratio is not None:
            self.img_mlp = nn.Sequential(
                nn.Linear(hidden_size, lin1_mlp_dim, bias=mlp_bias),
                SiLUActivation(),
                nn.Linear(lin1_mlp_dim // 2, hidden_size, bias=mlp_bias),
            )
        else:
            self.img_mlp = nn.Sequential(
                nn.Linear(hidden_size, lin1_mlp_dim, bias=mlp_bias),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_size, bias=mlp_bias),
            )

        if not shared_modulation:
            self.txt_mod = Modulation(hidden_size, double=True, bias=mod_bias)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if double_linear1_mlp_ratio is not None:
            self.txt_mlp = nn.Sequential(
                nn.Linear(hidden_size, lin1_mlp_dim, bias=mlp_bias),
                SiLUActivation(),
                nn.Linear(lin1_mlp_dim // 2, hidden_size, bias=mlp_bias),
            )
        else:
            self.txt_mlp = nn.Sequential(
                nn.Linear(hidden_size, lin1_mlp_dim, bias=mlp_bias),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_size, bias=mlp_bias),
            )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, *, NAG: dict | None = None,
                reference=None, reference_pe=None, reference_vec=None, cache_layer=None):
        kv_prefix = None
        if cache_layer is not None:
            if reference is not None:
                ref_mod1, ref_mod2 = reference_vec
                capture = [None, None]
                attended = _reference_attention([_norm_modulate(reference, self.img_norm1, ref_mod1)],
                                                (self.img_attn.q, self.img_attn.k, self.img_attn.v), self.img_attn.norm,
                                                reference_pe, self.num_heads, capture)
                cache_layer.store(*capture, kit=True)
                capture.clear()
                ops.gated_residual_(reference, [self.img_attn.proj(attended)], ref_mod1.gate, fused=True)
                del attended
                ops.mlp([_norm_modulate(reference, self.img_norm2, ref_mod2)], self.img_mlp[0], self.img_mlp[2], self.mlp_activation,
                        residual=reference, segments=ops.gate_segments(ref_mod2.gate, reference.shape[0], reference.shape[1]))
            kv_prefix = cache_layer.kv_prefix()
        if self.shared_modulation:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = _norm_modulate(img, self.img_norm1, img_mod1)

        if NAG is None:
            txt_modulated = _norm_modulate(txt, self.txt_norm1, txt_mod1)
            txt_len, head_dim = txt.shape[1], self.hidden_size // self.num_heads
            streams = [([txt_modulated], (self.txt_attn.q, self.txt_attn.k, self.txt_attn.v)), ([img_modulated], (self.img_attn.q, self.img_attn.k, self.img_attn.v))]
            del txt_modulated, img_modulated
            segments = ((0, txt_len, self.txt_attn.norm), (txt_len, txt_len + img.shape[1], self.img_attn.norm))
            normalize = _kv_norm_rope_ if cache_layer is not None else _norm_rope_
            attn = attention_kit.joint_qkv_attention(streams, self.num_heads, head_dim, lambda query, key, group: normalize(query, key, pe, segments), kv_prefix=kv_prefix,
                                                     fused_quant=lambda query, key, group, core, settings: _fused_norm_rope_quant_(query, key, pe, segments, core, settings)).flatten(2)
        else:
            img_list, img_modulated = [img_modulated], None
            attn = self._attention_nag(img_list, txt, txt_mod1, pe, NAG)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        ops.gated_residual_(img, [self.img_attn.proj(img_attn)], img_mod1.gate, fused=True)
        ops.mlp([_norm_modulate(img, self.img_norm2, img_mod2)], self.img_mlp[0], self.img_mlp[2], self.mlp_activation,
                residual=img, segments=ops.gate_segments(img_mod2.gate, img.shape[0], img.shape[1]))

        # calculate the txt blocks
        ops.gated_residual_(txt, [self.txt_attn.proj(txt_attn)], txt_mod1.gate, fused=True)
        ops.mlp([_norm_modulate(txt, self.txt_norm2, txt_mod2)], self.txt_mlp[0], self.txt_mlp[2], self.mlp_activation,
                residual=txt, segments=ops.gate_segments(txt_mod2.gate, txt.shape[0], txt.shape[1]))
        return (img, txt, reference) if cache_layer is not None else (img, txt)

    def _attention_nag(self, img_list, txt, txt_mod1, pe, NAG):
        img_modulated = img_list.pop()
        shape = (*img_modulated.shape[:2], self.num_heads, int(img_modulated.shape[-1] / self.num_heads) )
        img_q = self.img_attn.q(img_modulated).view(*shape).transpose(1,2)
        img_k = self.img_attn.k(img_modulated).view(*shape).transpose(1,2) 
        img_v = self.img_attn.v(img_modulated).view(*shape).transpose(1,2)
        del img_modulated


        img_q= self.img_attn.norm(img_q, None, img_v)
        img_k= self.img_attn.norm(None, img_k, img_v)

        # prepare txt for attention
        txt_modulated = _norm_modulate(txt, self.txt_norm1, txt_mod1)

        shape = (*txt_modulated.shape[:2], self.num_heads, int(txt_modulated.shape[-1] / self.num_heads) )
        txt_q = self.txt_attn.q(txt_modulated).view(*shape).transpose(1,2)
        txt_k = self.txt_attn.k(txt_modulated).view(*shape).transpose(1,2) 
        txt_v = self.txt_attn.v(txt_modulated).view(*shape).transpose(1,2)
        del txt_modulated


        txt_q = self.txt_attn.norm(txt_q, None, txt_v)
        txt_k = self.txt_attn.norm(None, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v

        qkv_list = [q, k, v]
        del q, k, v
        return attention(qkv_list, pe=pe, txt_len=txt.shape[1], NAG=NAG)


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        shared_modulation = False,
        single_linear1_mlp_ratio: float | None = None,
        single_mlp_hidden_ratio: float | None = None,
        linear_bias: bool = True,
        modulation_bias: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.shared_modulation = shared_modulation
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        lin1_mlp_dim = int(hidden_size * (single_linear1_mlp_ratio if single_linear1_mlp_ratio is not None else mlp_ratio))
        self.use_silu = single_linear1_mlp_ratio is not None
        if self.use_silu:
            self.mlp_hidden_dim = lin1_mlp_dim // 2
        else:
            self.mlp_hidden_dim = int(hidden_size * (single_mlp_hidden_ratio if single_mlp_hidden_ratio is not None else mlp_ratio))
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + lin1_mlp_dim, bias=linear_bias)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=linear_bias)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if not shared_modulation:
            self.modulation = Modulation(hidden_size, double=False, bias=modulation_bias)
        else:
            self.modulation = None

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, *, txt_len: int | None = None, NAG: dict | None = None,
                reference=None, reference_pe=None, reference_vec=None, cache_layer=None):
        kv_prefix = None
        if cache_layer is not None:
            if reference is not None:
                ref_modulated = _norm_modulate(reference, self.pre_norm, reference_vec)
                capture = [None, None]
                attended = _reference_attention([ref_modulated], (self.linear1_attn_q, self.linear1_attn_k, self.linear1_attn_v),
                                                self.norm, reference_pe, self.num_heads, capture)
                cache_layer.store(*capture, kit=True)
                capture.clear()
                ref_list, ref_modulated = [ref_modulated], None
                ops.mlp(ref_list, self.linear1_mlp, self.linear2, "swiglu", prefix=attended, residual=reference,
                        segments=ops.gate_segments(reference_vec.gate, reference.shape[0], reference.shape[1]))
                del attended
            kv_prefix = cache_layer.kv_prefix()
        if self.shared_modulation:
            mod = vec
        elif self.modulation is not None:
            mod, _ = self.modulation(vec)
        x_mod = _norm_modulate(x, self.pre_norm, mod)


        # x_mod = (1 + mod.scale) * x + mod.shift

        if NAG is None:  # x_mod stays: the MLP reads it after the attention
            segments = ((0, x_mod.shape[1], self.norm),)
            normalize = _kv_norm_rope_ if cache_layer is not None else _norm_rope_
            attn = attention_kit.qkv_attention([x_mod], self.linear1_attn_q, self.linear1_attn_k, self.linear1_attn_v, self.num_heads,
                                               self.hidden_size // self.num_heads, lambda query, key, group: normalize(query, key, pe, segments), kv_prefix=kv_prefix,
                                               fused_quant=lambda query, key, group, core, settings: _fused_norm_rope_quant_(query, key, pe, segments, core, settings)).flatten(2)
        else:
            shape = (*x_mod.shape[:2], self.num_heads, int(x_mod.shape[-1] / self.num_heads) )
            q = self.linear1_attn_q(x_mod).view(*shape).transpose(1,2)
            k = self.linear1_attn_k(x_mod).view(*shape).transpose(1,2)
            v = self.linear1_attn_v(x_mod).view(*shape).transpose(1,2)

            q = self.norm(q, None, v)
            k = self.norm(None, k, v)

            # compute attention
            qkv_list = [q, k, v]
            del q, k, v
            attn = attention(qkv_list, pe=pe, txt_len=txt_len, NAG=NAG)
        # linear2 reads the attention output and the activated MLP stream together: the shared chunked FFN with the attention as prefix
        x_mod_list, x_mod = [x_mod], None
        ops.mlp(x_mod_list, self.linear1_mlp, self.linear2, "swiglu" if self.use_silu else "gelu", prefix=attn, residual=x,
                segments=ops.gate_segments(mod.gate, x.shape[0], x.shape[1]))
        return (x, reference) if cache_layer is not None else x


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        chroma_modulation: bool = False,
        use_linear: bool = True,
        linear_bias: bool = True,
        modulation_bias: bool = True,
    ):
        super().__init__()
        self.chroma_modulation = chroma_modulation
        self.use_linear = use_linear
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=linear_bias)
            if use_linear
            else None
        )
        if not chroma_modulation:        
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=modulation_bias)
            )


    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if self.chroma_modulation:
            shift, scale = vec
            shift = shift.squeeze(1)
            scale = scale.squeeze(1)            
        else:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        # x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x_list, x = [x], None
        x = ops.norm_scale_shift(x_list, self.norm_final, scale[:, None, :], shift[:, None, :], in_place=False, fused_add=True)
        x = self.linear(x)
        return x


class DistilledGuidance(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers = 5):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for x in range( n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for x in range( n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)


    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class SigLIPMultiFeatProjModel(torch.nn.Module):
    """
    SigLIP Multi-Feature Projection Model for processing style features from different layers 
    and projecting them into a unified hidden space.
    
    Args:
        siglip_token_nums (int): Number of SigLIP tokens, default 257
        style_token_nums (int): Number of style tokens, default 256  
        siglip_token_dims (int): Dimension of SigLIP tokens, default 1536
        hidden_size (int): Hidden layer size, default 3072
        context_layer_norm (bool): Whether to use context layer normalization, default False
    """
    
    def __init__(
        self,
        siglip_token_nums: int = 257,
        style_token_nums: int = 256,
        siglip_token_dims: int = 1536,
        hidden_size: int = 3072,
        context_layer_norm: bool = False,
    ):
        super().__init__()
        
        # High-level feature processing (layer -2)
        self.high_embedding_linear = nn.Sequential(
            nn.Linear(siglip_token_nums, style_token_nums), 
            nn.SiLU()
        )
        self.high_layer_norm = (
            nn.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.high_projection = nn.Linear(siglip_token_dims, hidden_size, bias=True)
        
        # Mid-level feature processing (layer -11)
        self.mid_embedding_linear = nn.Sequential(
            nn.Linear(siglip_token_nums, style_token_nums), 
            nn.SiLU()
        )
        self.mid_layer_norm = (
            nn.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.mid_projection = nn.Linear(siglip_token_dims, hidden_size, bias=True)
        
        # Low-level feature processing (layer -20)
        self.low_embedding_linear = nn.Sequential(
            nn.Linear(siglip_token_nums, style_token_nums), 
            nn.SiLU()
        )
        self.low_layer_norm = (
            nn.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.low_projection = nn.Linear(siglip_token_dims, hidden_size, bias=True)

    def forward(self, siglip_outputs):
        """
        Forward pass function
        
        Args:
            siglip_outputs: Output from SigLIP model, containing hidden_states
            
        Returns:
            torch.Tensor: Concatenated multi-layer features with shape [bs, 3*style_token_nums, hidden_size]
        """
        dtype = next(self.high_embedding_linear.parameters()).dtype
        
        # Process high-level features (layer -2)
        high_embedding = self._process_layer_features(
            siglip_outputs.hidden_states[-2],
            self.high_embedding_linear,
            self.high_layer_norm,
            self.high_projection,
            dtype
        )
        
        # Process mid-level features (layer -11)
        mid_embedding = self._process_layer_features(
            siglip_outputs.hidden_states[-11],
            self.mid_embedding_linear,
            self.mid_layer_norm,
            self.mid_projection,
            dtype
        )
        
        # Process low-level features (layer -20)
        low_embedding = self._process_layer_features(
            siglip_outputs.hidden_states[-20],
            self.low_embedding_linear,
            self.low_layer_norm,
            self.low_projection,
            dtype
        )
        
        # Concatenate features from all layers
        return torch.cat((high_embedding, mid_embedding, low_embedding), dim=1)
    
    def _process_layer_features(
        self, 
        hidden_states: torch.Tensor,
        embedding_linear: nn.Module,
        layer_norm: nn.Module,
        projection: nn.Module,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Helper function to process features from a single layer
        
        Args:
            hidden_states: Input hidden states [bs, seq_len, dim]
            embedding_linear: Embedding linear layer
            layer_norm: Layer normalization
            projection: Projection layer
            dtype: Target data type
            
        Returns:
            torch.Tensor: Processed features [bs, style_token_nums, hidden_size]
        """
        # Transform dimensions: [bs, seq_len, dim] -> [bs, dim, seq_len] -> [bs, dim, style_token_nums] -> [bs, style_token_nums, dim]
        embedding = embedding_linear(
            hidden_states.to(dtype).transpose(1, 2)
        ).transpose(1, 2)
        
        # Apply layer normalization
        embedding = layer_norm(embedding)
        
        # Project to target hidden space
        embedding = projection(embedding)
        
        return embedding
