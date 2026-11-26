import math
from dataclasses import dataclass

import torch
from mmgp import cuda_graphs
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from shared import attention_kit
from shared.attention import pay_attention
from shared.kernels import ops
from shared.utils.lora_mapping import convert_lora_keys


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def _apply_rope_inplace(x: Tensor, freqs: Tensor) -> Tensor:
    """RoPE of x (batch, heads, tokens, head_dim) in place, channels paired (2i, 2i + 1), cos / sin of freqs in x's dtype."""
    freqs = freqs[:, None]
    return ops.paired_rope_(x, freqs[..., 0, 0].to(x.dtype), freqs[..., 1, 0].to(x.dtype), interleaved=True)


def _norm_rope_(query, key, qknorm, freqs):
    """attention_kit callback: QKNorm and RoPE of q and/or k (batch, tokens, heads, head_dim) in place, by chunks of tokens: the values
    of Attention.forward, whose tensors are (batch, heads, tokens, head_dim)."""
    for tensor, norm in ((query, qknorm.qnorm), (key, qknorm.knorm)):
        if tensor is not None:
            attention_kit.rows_(tensor, lambda part, first, last: _norm_rope_rows(norm.forward(part), None if freqs is None else freqs[:, first:last]))


def _norm_rope_quant(query, key, qknorm, freqs, core, settings):
    """attention_kit fused_quant: the staged Sage 2 quantization of q and/or k with their zero-centered QKNorm (scale + 1 in the scale's
    dtype, as RMSNorm.forward) and RoPE (freqs (batch, tokens, pairs, 2, 2) viewed with a head axis)."""
    norms = [None if tensor is None else (norm.scale + 1, norm.eps) for tensor, norm in ((query, qknorm.qnorm), (key, qknorm.knorm))]
    return ops.split_half_norm_rope_quant_(query, key, None if freqs is None else freqs.unsqueeze(2), *norms, core, settings, interleaved=True)


def _norm_rope_rows(x: Tensor, freqs: Tensor | None) -> Tensor:
    """_apply_rope_inplace of x (batch, tokens, heads, head_dim)."""
    return x if freqs is None else _apply_rope_inplace(x.transpose(1, 2), freqs).transpose(1, 2)


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    return _apply_rope_inplace(xq, freqs), _apply_rope_inplace(xk, freqs)


def _attention_from_blh(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None, scale: float | None = None, gqa: bool = False) -> Tensor:
    if gqa:
        batch = q.shape[0]
        groups = k.shape[2]
        repeat = q.shape[2] // groups
        q = rearrange(q, "B L (G R) D -> (B G R) L 1 D", G=groups, R=repeat)
        k = rearrange(k, "B L G D -> B G 1 L D").expand(batch, groups, repeat, -1, -1)
        v = rearrange(v, "B L G D -> B G 1 L D").expand(batch, groups, repeat, -1, -1)
        k = rearrange(k, "B G R L D -> (B G R) L 1 D")
        v = rearrange(v, "B G R L D -> (B G R) L 1 D")
        if mask is not None:
            mask = mask.repeat_interleave(groups * repeat, dim=0).contiguous() if batch > 1 else mask.expand(groups * repeat, -1, -1, -1)
        qkv_list = [q, k, v]
        q = k = v = None
        out = pay_attention(qkv_list, attention_mask=mask, softmax_scale=scale, recycle_q=True)
        return rearrange(out, "(B G R) L 1 D -> B L (G R D)", B=batch, G=groups, R=repeat)
    qkv_list = [q, k, v]
    q = k = v = None
    return rearrange(pay_attention(qkv_list, attention_mask=mask, softmax_scale=scale, recycle_q=True), "B L H D -> B L (H D)")


def attention(
    qkv_list: list[Tensor],
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
    txt_len: int | None = None,
    NAG: dict | None = None,
    neg_k: Tensor | None = None,
    neg_v: Tensor | None = None,
    neg_mask: Tensor | None = None,
) -> Tensor:
    q, k, v = qkv_list
    qkv_list.clear()
    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    if mask is not None:
        mask = mask.transpose(1, 2)
    if NAG is not None:
        if txt_len is None or neg_k is None or neg_v is None or neg_mask is None:
            raise ValueError("Krea 2 NAG requires positive stream tokens plus negative text K/V side inputs.")
        cap_len = int(NAG.get("cap_embed_len", 0) or 0)
        if cap_len <= 0 or cap_len != txt_len or neg_k.shape[2] != cap_len:
            raise ValueError("Krea 2 NAG expected matching positive and negative text lengths.")
        neg_k = rearrange(neg_k, "B H L D -> B L H D")
        neg_v = rearrange(neg_v, "B H L D -> B L H D")
        neg_mask = neg_mask[:, None, None, :]
        img_start = txt_len
        query_start = int(NAG["query_start"])
        query_end = int(NAG["query_end"])
        if not img_start <= query_start < query_end <= q.shape[1]:
            raise ValueError(f"Krea 2 NAG target query range [{query_start}, {query_end}) is invalid for sequence length {q.shape[1]}.")
        guidance_q = q[:, query_start:query_end].clone()
        x_pos = _attention_from_blh(q, k, v, mask=mask, scale=scale, gqa=gqa)
        neg_full_mask = None if mask is None else torch.cat((neg_mask, mask[..., img_start:]), dim=-1)
        k_neg = torch.cat((neg_k, k[:, img_start:]), dim=1)
        v_neg = torch.cat((neg_v, v[:, img_start:]), dim=1)
        x_guidance = _attention_from_blh(guidance_q, k_neg, v_neg, mask=neg_full_mask, scale=scale, gqa=gqa)
        q = k = v = neg_k = neg_v = k_neg = v_neg = neg_mask = neg_full_mask = None

        x_pos_img = x_pos[:, query_start:query_end]
        nag_scale = float(NAG["scale"])
        nag_alpha = float(NAG["alpha"])
        nag_tau = float(NAG["tau"])
        dtype = x_pos_img.dtype
        x_guidance.mul_(1 - nag_scale)
        x_guidance.add_(x_pos_img, alpha=nag_scale)
        norm_positive = torch.norm(x_pos_img, p=1, dim=-1, keepdim=True)
        norm_guidance = torch.norm(x_guidance, p=1, dim=-1, keepdim=True)
        norm_scale = norm_guidance / norm_positive
        torch.nan_to_num(norm_scale, nan=10.0, posinf=10.0, neginf=10.0, out=norm_scale)
        factor = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
        x_guidance = torch.where(norm_scale > nag_tau, x_guidance * factor, x_guidance).to(dtype)
        del norm_positive, norm_guidance, norm_scale, factor

        x_guidance.mul_(nag_alpha)
        x_guidance.add_(x_pos_img, alpha=1 - nag_alpha)
        x_pos_img.copy_(x_guidance)
        x_guidance = x_pos_img = None
        return x_pos
    out = _attention_from_blh(q, k, v, mask=mask, scale=scale, gqa=gqa)
    q = k = v = None
    return out


def key_padding_mask(mask: Tensor | None) -> Tensor | None:
    return None if mask is None else mask[:, None, None, :]


def temb(t: Tensor, dim: int, period: float = 1e4, tfactor: float = 1e3, device: torch.device = None, dtype: torch.dtype = None) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half)
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = 12
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20


def config_from_diffusers(data: dict) -> SingleMMDiTConfig:
    features = data["attention_head_dim"] * data["num_attention_heads"]
    base_mlp = int(2 * features / 3)
    return SingleMMDiTConfig(
        features=features,
        tdim=data["timestep_embed_dim"],
        txtdim=data["text_hidden_dim"],
        heads=data["num_attention_heads"],
        kvheads=data["num_key_value_heads"],
        multiplier=data["intermediate_size"] // base_mlp,
        layers=data["num_layers"],
        patch=2,
        channels=data["in_channels"] // 4,
        txtheads=data["text_num_attention_heads"],
        txtkvheads=data["text_num_key_value_heads"],
        txtlayers=data["num_text_layers"],
        theta=data["rope_theta"],
    )


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor):
        out = vec + self.lin
        return out.chunk(6, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat([rope(pos[..., i], d, self.theta, self.ntk) for i, d in enumerate(self.axdims)], dim=-3)


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, device=device, dtype=torch.float32))

    def forward(self, x_list: Tensor) -> Tensor:
        if isinstance(x_list,list):
            x = x_list[0]
            x_list.clear()
        else:
            x = x_list
            del x_list
        x_list, x = [x], None
        return ops.zero_center_rms_norm(x_list, self.scale, self.eps, offset_in_weight_dtype=True)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, qkv_list: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        q_list = [qkv_list[0]]
        k_list = [qkv_list[1]]
        v = qkv_list[2]
        qkv_list.clear()
        return self.qnorm.forward(q_list), self.knorm.forward(k_list), v


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)
        self.features = features
        self.mlpdim = mlpdim

    def forward(self, x_list: list[Tensor], residual: Tensor | None = None, gate: Tensor | None = None) -> Tensor:
        """The shared chunked FFN of the activation handed off in x_list: in place, or added to residual scaled by gate (batch, 1,
        features) when given."""
        segments = None if gate is None else ops.gate_segments(gate, residual.shape[0], residual.shape[1])
        return ops.mlp(x_list, self.gate, self.down, "swiglu", up=self.up, residual=residual, segments=segments)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        qkv: Tensor | list[Tensor],
        freqs: Tensor | None = None,
        mask: Tensor | None = None,
        txt_len: int | None = None,
        NAG: dict | None = None,
        neg_context: Tensor | None = None,
        neg_mask: Tensor | None = None,
    ) -> Tensor:
        if NAG is None:
            # the gate reads the attention input: computed first so that the attention releases the input after its projections
            x_list = qkv if isinstance(qkv, list) else [qkv]
            gate = self.gate(x_list[0])
            qkv = None
            out = attention_kit.qkv_attention(x_list, self.wq, self.wk, self.wv, self.heads, self.headdim,
                                              lambda query, key, group: _norm_rope_(query, key, self.qknorm, freqs),
                                              fused_quant=lambda query, key, group, core, settings: _norm_rope_quant(query, key, self.qknorm, freqs, core, settings),
                                              kv_heads=self.kvheads, attention_mask=None if mask is None else mask.transpose(1, 2),
                                              sol3_prefix_tokens=txt_len or 0).flatten(2)
            ops.sigmoid_gate_(out, [gate])
            del gate
            return self.wo(out)
        if isinstance(qkv, list):
            qkv_ = qkv[0]
            qkv.clear()
            qkv = qkv_
        neg_k = neg_v = None
        if NAG is not None:
            if neg_context is None or neg_mask is None:
                raise ValueError("Krea 2 NAG requires negative text context and mask.")
            neg_k, neg_v = self.wk(neg_context), self.wv(neg_context)
            neg_k = rearrange(neg_k, "B L (H D) -> B H L D", H=self.kvheads)
            neg_v = rearrange(neg_v, "B L (H D) -> B H L D", H=self.kvheads)
            neg_k = self.qknorm.knorm.forward([neg_k])
            if freqs is not None:
                neg_k = _apply_rope_inplace(neg_k, freqs[:, : neg_context.shape[1]])
            neg_context = None
        q, k, v = self.wq(qkv), self.wk(qkv), self.wv(qkv)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        qkv_list = [q, k, v]
        q = k = v = None
        q, k, v = self.qknorm(qkv_list)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        qkv_list = [q, k, v]
        q = k = v = None
        out = attention(qkv_list, mask=mask, gqa=self.gqa, txt_len=txt_len, NAG=NAG, neg_k=neg_k, neg_v=neg_v, neg_mask=neg_mask)
        neg_k = neg_v = neg_mask = None
        gate_list = [self.gate(qkv)]
        del qkv
        ops.sigmoid_gate_(out, gate_list)
        return self.wo(out)


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor | list[Tensor], tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x_list = x if isinstance(x, list) else [x]
        return self.linear(ops.norm_scale_shift(x_list, self.norm, scale, shift))


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x_list = [self.prenorm(x)]
        attn_out = self.attn(x_list, mask=mask)
        x.add_(attn_out)
        del attn_out
        self.mlp([self.postnorm(x)], residual=x)
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers: int, txt_dim: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
            if getattr(self, "_interrupt", False):
                return None
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
            if getattr(self, "_interrupt", False):
                return None
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
        txt_len: int | None = None,
        NAG: dict | None = None,
        neg_context: Tensor | None = None,
        neg_mask: Tensor | None = None,
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x_list = [ops.norm_scale_shift([x], self.prenorm, prescale, preshift)]
        if NAG is not None:
            neg_context = ops.norm_scale_shift([neg_context], self.prenorm, prescale, preshift)
        attn_out = self.attn(x_list, freqs, mask, txt_len=txt_len, NAG=NAG, neg_context=neg_context, neg_mask=neg_mask)
        neg_context = neg_mask = None
        ops.gated_residual_(x, [attn_out], pregate)
        del attn_out
        self.mlp([ops.norm_scale_shift([x], self.postnorm, postscale, postshift)], residual=x, gate=postgate)
        return x


class SingleStreamDiT(nn.Module):
    def preprocess_loras(self, model_type, sd):
        """Map adapter names, DoRA magnitudes and linear LoKr factors to MMGP."""
        replacements = (
            ("final_layer", "last"),
            ("img_in", "first"),
            ("time_embed.linear_1", "tmlp.0"),
            ("time_embed.linear_2", "tmlp.2"),
            ("time_mod_proj", "tproj.1"),
            ("txt_in.linear_1", "txtmlp.1"),
            ("txt_in.linear_2", "txtmlp.3"),
            ("transformer_blocks", "blocks"),
            ("text_fusion", "txtfusion"),
            ("feed_forward.", "mlp."),
            ("ffn.", "mlp."),
            ("ff.", "mlp."),
            ("to_out.0", "wo"),
            ("to_out", "wo"),
            ("to_gate", "gate"),
            ("to_q", "wq"),
            ("to_k", "wk"),
            ("to_v", "wv"),
        )

        def map_key(key):
            prefix = next((p for p in ("lora_unet_", "lora_te_") if key.startswith(p)), None)
            if prefix:
                path, dot, suffix = key[len(prefix):].partition(".")
                if not dot:
                    return key
                for source, target in (
                    ("final_layer", "last"),
                    ("img_in", "first"),
                    ("time_embed_linear_1", "tmlp_0"),
                    ("time_embed_linear_2", "tmlp_2"),
                    ("time_mod_proj", "tproj_1"),
                    ("txt_in_linear_1", "txtmlp_1"),
                    ("txt_in_linear_2", "txtmlp_3"),
                    ("transformer_blocks", "blocks"),
                    ("text_fusion", "txtfusion"),
                    ("_feed_forward_", "_mlp_"),
                    ("_ffn_", "_mlp_"),
                    ("_ff_", "_mlp_"),
                    ("to_out_0", "wo"),
                    ("to_out", "wo"),
                    ("to_gate", "gate"),
                    ("to_q", "wq"),
                    ("to_k", "wk"),
                    ("to_v", "wv"),
                ):
                    path = path.replace(source, target)
                for name in ("layerwise_blocks", "refiner_blocks", "language_model"):
                    path = path.replace(name, name.replace("_", "-"))
                path = path.replace("_", ".").replace("-", "_")
                return f"{path}.{suffix}"

            for source, target in replacements:
                key = key.replace(source, target)
            return key

        mapped = {}
        for key, value in sd.items():
            key = map_key(key)
            if key.endswith(".magnitude"):
                # AI Toolkit stores output-row magnitudes as [out]; MMGP uses [out, 1].
                key = key.removesuffix(".magnitude") + ".dora_scale"
                value = value.reshape(-1, 1)
            mapped[key] = value
        return convert_lora_keys(mapped, dict(self.named_modules()), compose_lokr=True)

    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        headdim = config.features // config.heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        self.posemb = PositionalEncoding(config.features, axes, theta=config.theta, ntk=1.0)
        self.first = nn.Linear(config.channels * config.patch**2, config.features, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(config.features, config.heads, config.multiplier, config.bias, config.kvheads) for _ in range(config.layers)])
        self.tmlp = nn.Sequential(nn.Linear(config.tdim, config.features), nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features))
        self.txtfusion = TextFusionTransformer(config.txtlayers, config.txtdim, config.txtheads, config.multiplier, config.bias, config.txtkvheads)
        self.txtmlp = nn.Sequential(RMSNorm(config.txtdim), nn.Linear(config.txtdim, config.features), nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features))
        self.last = LastLayer(config.features, config.patch, config.channels)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6))

    def prepare_context(self, context: Tensor | list[Tensor], mask: Tensor | None, output_len: int | None = None) -> Tensor | None:
        self.txtfusion._interrupt = getattr(self, "_interrupt", False)
        if isinstance(context, list):
            context_ = context[0]
            context.clear()
            context = context_
        else:
            context = context.clone()
        valid_mask = None if mask is None else mask[:, : context.shape[1]]
        txtmask = key_padding_mask(valid_mask)
        context = self.txtfusion(context, mask=txtmask)
        if context is None:
            return None
        context = self.txtmlp(context)
        if valid_mask is not None:
            context.masked_fill_(~valid_mask.unsqueeze(-1), 0)
        if output_len is not None and context.shape[1] < output_len:
            context = F.pad(context, (0, 0, 0, output_len - context.shape[1]))
        return context

    def prepare_timestep(self, t: Tensor) -> tuple[Tensor, Tensor]:
        t = self.tmlp(temb(t, self.config.tdim, device=t.device, dtype=t.dtype))
        return t, self.tproj(t)

    def _build_stream(self, img: Tensor, context: Tensor, pos: Tensor, mask: Tensor | None, freqs: Tensor | None = None):
        txtlen, imglen = context.shape[1], img.shape[1]
        combined = img.new_empty(img.shape[0], txtlen + imglen, img.shape[-1])
        combined[:, :txtlen].copy_(context)
        combined[:, txtlen:].copy_(img)
        del context
        mask = key_padding_mask(mask)
        if freqs is None:
            freqs = self.posemb(pos)
            freqs = freqs.to(combined.dtype)
        return combined, txtlen, imglen, freqs, mask

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        tvec: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        NAG: dict | None = None,
        neg_context: Tensor | None = None,
        neg_mask: Tensor | None = None,
        target_len: int | None = None,
    ) -> Tensor:
        img = self.first(img)
        combined, txtlen, imglen, freqs, mask = self._build_stream(img, context, pos, mask)
        del img, context, pos
        with cuda_graphs.scope():
            for block in self.blocks:
                combined = block(combined, tvec, freqs, mask, txt_len=txtlen, NAG=NAG, neg_context=neg_context, neg_mask=neg_mask)
                if getattr(self, "_interrupt", False):
                    return None
                self.txtfusion._interrupt = getattr(self, "_interrupt", False)
        target_len = imglen if target_len is None else target_len
        return self.last([combined[:, txtlen + imglen - target_len : txtlen + imglen]], t)

    def forward_cfg(self, img: Tensor, context: Tensor, uncond_context: Tensor, t: Tensor, tvec: Tensor, pos: Tensor, uncond_pos: Tensor, mask: Tensor | None, uncond_mask: Tensor | None, target_len: int | None = None) -> tuple[Tensor | None, Tensor | None]:
        img = self.first(img)
        share_freqs = pos.shape == uncond_pos.shape
        combined, txtlen, imglen, freqs, mask = self._build_stream(img, context, pos, mask)
        uncond_combined, uncond_txtlen, uncond_imglen, uncond_freqs, uncond_mask = self._build_stream(img, uncond_context, uncond_pos, uncond_mask, freqs=freqs if share_freqs else None)
        del img, context, uncond_context, pos, uncond_pos
        with cuda_graphs.scope():
            for block in self.blocks:
                combined = block(combined, tvec, freqs, mask, txt_len=txtlen)
                if getattr(self, "_interrupt", False):
                    return None, None
                uncond_combined = block(uncond_combined, tvec, uncond_freqs, uncond_mask, txt_len=uncond_txtlen)
                if getattr(self, "_interrupt", False):
                    return None, None
        target_len = imglen if target_len is None else target_len
        return self.last([combined[:, txtlen + imglen - target_len : txtlen + imglen]], t), self.last([uncond_combined[:, uncond_txtlen + uncond_imglen - target_len : uncond_txtlen + uncond_imglen]], t)
