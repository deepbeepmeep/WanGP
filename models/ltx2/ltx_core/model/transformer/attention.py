from enum import Enum
from typing import Protocol

import torch

from ...utils import rms_norm
from shared import attention_kit
from shared.attention import pay_attention
from .rope import LTXRopeType, RopeCache, apply_rotary_emb_inplace, apply_rope_cache_heads_inplace
from .sol_attention import SOL_ATTN_HEAD_DIM, sol_attention
from shared.kernels.ops import project_many

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return (
                XFormersAttention()(q, k, v, heads, mask)
                if memory_efficient_attention is not None
                else PytorchAttention()(q, k, v, heads, mask)
            )


class DBMRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x, in_place=True):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return rms_norm(x, weight=self.weight, eps=self.eps, in_place=in_place)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = DBMRMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = DBMRMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True) if apply_gated_attention else None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def _resolve_attention_override(self) -> tuple[str | None, int | None]:
        if isinstance(self.attention_function, AttentionFunction):
            if self.attention_function is AttentionFunction.PYTORCH:
                return "sdpa", None
            if self.attention_function is AttentionFunction.XFORMERS:
                return "xformers", None
            if self.attention_function is AttentionFunction.FLASH_ATTENTION_3:
                return "flash", 3
        return None, None

    def _head_mean_squares(self, project):
        # LTX uses vector_norm followed by square, including that rounding point. Compute each full-width statistic once;
        # retain only per-token FP32 scalars while projecting/attending groups of heads.
        result = []
        for which in (0, 1):
            full = project(which, range(self.heads)).flatten(2)
            norm = torch.linalg.vector_norm(full, ord=2, dim=-1, keepdim=True, dtype=torch.float32)
            del full
            result.append(norm.square_().div_(self.heads * self.dim_head))
        return tuple(result)

    def _norm_rope_(self, query, key, pe, k_pe, group):
        """attention_kit callback: q_norm/k_norm (over all heads) and RoPE of q and/or k in place."""
        for which, (tensor, norm, freqs, heads) in enumerate(((query, self.q_norm, pe, group.q), (key, self.k_norm, pe if k_pe is None else k_pe, group.kv))):
            if tensor is not None:
                rows = tensor.flatten(2)
                if group.mean_squares is None:
                    norm(rows)
                else:
                    rows.mul_(group.mean_squares[which].add(norm.eps).rsqrt_())
                    rows.mul_(norm.weight[heads.start * self.dim_head:heads.stop * self.dim_head])
                if freqs is not None:
                    if len(heads) == self.heads:
                        apply_rotary_emb_inplace(rows, freqs, self.rope_type)
                    else:
                        apply_rope_cache_heads_inplace(rows, freqs, heads)

    def forward(
        self,
        x_list: torch.Tensor,
        context_list: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        NAG: dict | None = None,
    ) -> torch.Tensor:
        split_heads = all(freqs is None or isinstance(freqs, RopeCache) and freqs.rope_type == LTXRopeType.SPLIT
                          and freqs.num_attention_heads == self.heads for freqs in (pe, k_pe))
        split_cross = (context_list is not None and NAG is None and split_heads and context_list[0].shape[0] == x_list[0].shape[0] and
                       attention_kit.head_groups(self.heads, x_list[0].shape[0] * max(x_list[0].shape[1], context_list[0].shape[1])) > 1)
        if ((context_list is None and mask is None or split_cross) and self._resolve_attention_override()[0] is None):
            # the gates read the attention input: computed first so that the input is released by the q/k/v projections
            gate_logits = None if self.to_gate_logits is None else self.to_gate_logits(x_list[0])
            callback = lambda query, key, group: self._norm_rope_(query, key, pe, k_pe, group)
            options = dict(split_heads=split_heads, norm_spans_heads=True, norm_stats=self._head_mean_squares, attention_mask=mask)
            if context_list is None:
                out = attention_kit.qkv_attention(x_list, self.to_q, self.to_k, self.to_v, self.heads, self.dim_head, callback, **options)
            else:
                out = attention_kit.cross_qkv_attention(x_list, context_list, self.to_q, self.to_k, self.to_v, self.heads, self.dim_head, callback, **options)
            if gate_logits is not None:
                out.mul_((2.0 * torch.sigmoid(gate_logits).to(dtype=out.dtype)).unsqueeze(-1))
            return self.to_out(out.flatten(2, 3))
        x = x_list[0]
        gate_logits = None if self.to_gate_logits is None else self.to_gate_logits(x)  # first: x is released after the projections
        x_list.clear()
        context = None
        if context_list is not None:
            context = context_list[0]
            context_list.clear()
        cross_attn = context is not None
        if cross_attn:
            q = self.to_q(x)
            context_list, context, x = [context], None, None
            k, v = project_many((self.to_k, self.to_v), context_list)
        else:
            x_list, x = [x], None
            q, k, v = project_many((self.to_q, self.to_k, self.to_v), x_list)
        x = None
        context = None
        self.q_norm(q)
        self.k_norm(k)

        if pe is not None:
            apply_rotary_emb_inplace(q, pe, self.rope_type)
            apply_rotary_emb_inplace(k, pe if k_pe is None else k_pe, self.rope_type)

        q = q.view(q.shape[0], -1, self.heads, self.dim_head)
        k = k.view(k.shape[0], -1, self.heads, self.dim_head)
        v = v.view(v.shape[0], -1, self.heads, self.dim_head)
        force_attention, attention_version = self._resolve_attention_override()

        if cross_attn and NAG is not None:
            cap_len = int(NAG.get("cap_embed_len", 0) or 0)
            if cap_len > 0 and k.shape[1] == cap_len * 2:
                pos_mask = None if mask is None else mask[..., :cap_len]
                neg_mask = None if mask is None else mask[..., cap_len : cap_len * 2]
                qkv_list = [q, k[:, :cap_len], v[:, :cap_len]]
                # Keep the merge in attention-output space with in-place ops, following the Wan low-allocation path.
                x_pos = pay_attention(
                    qkv_list,
                    attention_mask=pos_mask,
                    force_attention=force_attention,
                    version=attention_version,
                )
                qkv_list = [q, k[:, cap_len : cap_len * 2], v[:, cap_len : cap_len * 2]]
                q = k = v = None
                out = pay_attention(
                    qkv_list,
                    attention_mask=neg_mask,
                    force_attention=force_attention,
                    version=attention_version,
                    recycle_q=True,
                )
                nag_scale = float(NAG["scale"])
                nag_alpha = float(NAG["alpha"])
                nag_tau = float(NAG["tau"])
                out.mul_(1 - nag_scale)
                out.add_(x_pos, alpha=nag_scale)
                norm_positive = torch.sum(torch.abs(x_pos), dim=(2, 3), keepdim=True)
                norm_guidance = torch.sum(torch.abs(out), dim=(2, 3), keepdim=True)
                scale = norm_guidance / norm_positive
                torch.nan_to_num(scale, nan=10.0, posinf=10.0, neginf=10.0, out=scale)
                factor = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
                out = torch.where(scale > nag_tau, out * factor, out)
                del norm_positive, norm_guidance, scale, factor
                x_pos.mul_(1 - nag_alpha)
                out.mul_(nag_alpha)
                out.add_(x_pos)
                x_pos = None
                if gate_logits is not None:
                    gates = 2.0 * torch.sigmoid(gate_logits).to(dtype=out.dtype)
                    out.mul_(gates.unsqueeze(-1))
                gate_logits = None
                out = out.flatten(2, 3)
                out = self.to_out(out)
                return out

        use_sol = (
            not cross_attn
            and mask is None
            and NAG is None
            and self.dim_head == SOL_ATTN_HEAD_DIM
            and sol_attention.use_for_layer(q.shape[1], self.dim_head)
        )
        qkv_list = [q, k, v]
        q = k = v = None
        if use_sol:
            out = sol_attention(qkv_list)
        else:
            out = pay_attention(
                qkv_list,
                attention_mask=mask,
                force_attention=force_attention,
                version=attention_version,
                recycle_q=True,
            )
        if gate_logits is not None:
            gates = 2.0 * torch.sigmoid(gate_logits).to(dtype=out.dtype)
            out.mul_(gates.unsqueeze(-1))
        gate_logits = None
        out = out.flatten(2, 3)
        out = self.to_out(out)
        return out
