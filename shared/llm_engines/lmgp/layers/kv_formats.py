# Part of WanGP - WanGP Community License 2.0 (see LICENSE.txt), Copyright (c) 2026 DeepBeepMeep
"""Storage formats of the paged KV cache and the layout of a mixed-precision cache.

Keys and values are quantized per token, head and group of 32 head dimensions. The body of the cache holds every token in the
format chosen for keys and for values; a layout below Q8 keeps two exact Q8 overlays beside it: the sink (the first pages of each
sequence) and the ring of the most recent tokens, both indexed by position. Formats:

- q8: int8, FP16 scale = max|x| / 127 (the cache WanGP has always used).
- q6_0 / q4_0: symmetric, llama.cpp style: d = x_extreme / -2^(bits-1) stored as FP16, q = round(x / d) + 2^(bits-1) unsigned.
- q4_1: asymmetric, d = (max - min) / 15 and the FP16 min stored, q = round((x - min) / d).

Packing of a group: byte j (0..15) holds the low nibble of value j and, in its high nibble, the low nibble of value j + 16; the two high
bits of 6-bit values follow as two little-endian 32-bit words, word h for values h*16 .. h*16+15 with value h*16 + r*4 + b in bits
b*8 + r*2 and b*8 + r*2 + 1. The kernels therefore read a group as two halves of 16 lanes (values 0..15 from the low nibbles, 16..31
from the high nibbles) with contiguous loads and no reordering, and one word per group and half for the high bits. Byte widths per
group: 32 (q8), 24 (q6_0), 16 (q4_0, q4_1); a block of the body stores the groups of one head one after the other.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

GROUP = 32
TILE = 32  # columns a kernel tile covers: region boundaries are aligned to it


@dataclass(frozen=True)
class KVFormat:
    name: str
    bits: int
    asymmetric: bool

    @property
    def q8(self) -> bool:
        return self.bits == 8

    @property
    def data_bytes(self) -> int:
        return GROUP * self.bits // 8

    @property
    def dtype(self) -> torch.dtype:
        return torch.int8 if self.q8 else torch.uint8

    @property
    def group_bytes(self) -> int:
        """Data, FP16 scale and FP16 min of one group."""
        return self.data_bytes + 2 + (2 if self.asymmetric else 0)

    def width(self, head_dim: int) -> int:
        """Bytes of one head's data at one token."""
        return head_dim // GROUP * self.data_bytes


FORMATS = {f.name: f for f in (KVFormat("q8", 8, False), KVFormat("q6_0", 6, False), KVFormat("q4_1", 4, True), KVFormat("q4_0", 4, False))}
Q8 = FORMATS["q8"]

# config values of the body (keys, values) and of the exact overlays (sink pages, recent tokens)
BODY_PRESETS = {"int8": ("q8", "q8"), "q6": ("q6_0", "q6_0"), "int8_q4": ("q8", "q4_0"), "q6_q4": ("q6_0", "q4_0"), "q4_1": ("q4_1", "q4_1"), "q4": ("q4_0", "q4_0")}
EXACT_PRESETS = {"s1r1024": (1, 1024), "s1r256": (1, 256), "s0r1024": (0, 1024), "s0r0": (0, 0)}
DEFAULT_EXACT = "s1r1024"


@dataclass(frozen=True)
class KVCacheLayout:
    body: str
    exact: str
    k: KVFormat
    v: KVFormat
    sink_pages: int
    recent_tokens: int

    @property
    def mixed(self) -> bool:
        return not (self.k.q8 and self.v.q8)

    @property
    def overlay(self) -> bool:
        return self.mixed and (self.sink_pages > 0 or self.recent_tokens > 0)

    @property
    def value(self) -> str:
        return f"{self.body}:{self.exact}" if self.mixed else self.body

    def sink_tokens(self, page_size: int) -> int:
        return self.sink_pages * page_size

    def overlay_rows(self, page_size: int) -> int:
        """Rows of one sequence's overlay: the sink positions, then the ring."""
        return self.sink_tokens(page_size) + self.recent_tokens

    def ring_start(self, ring_high: int, page_size: int) -> int:
        """First position read from the ring for a sequence whose highest written position + 1 is ring_high (TILE aligned, upwards)."""
        if not self.overlay:
            return 1 << 30
        start = max(ring_high - self.recent_tokens, self.sink_tokens(page_size))
        return -(-start // TILE) * TILE

    def overlay_row(self, position: int, page_size: int) -> int:
        sink = self.sink_tokens(page_size)
        return position if position < sink else sink + (position & (self.recent_tokens - 1))

    def body_bytes_per_token(self, layers: int, kv_heads: int, head_dim: int) -> int:
        return layers * kv_heads * head_dim // GROUP * (self.k.group_bytes + self.v.group_bytes)

    def overlay_bytes(self, layers: int, kv_heads: int, head_dim: int, page_size: int, sequences: int) -> int:
        return 0 if not self.overlay else sequences * self.overlay_rows(page_size) * layers * kv_heads * head_dim // GROUP * 2 * Q8.group_bytes


def parse_layout(value: str) -> KVCacheLayout:
    """A quantized cache layout from its config value: a body preset, optionally ':' and an exact-overlay preset (ignored for int8)."""
    body, _, exact = str(value).strip().lower().partition(":")
    if body not in BODY_PRESETS:
        raise ValueError(f"Unknown KV cache quantization {value!r}; expected one of {', '.join(BODY_PRESETS)}.")
    k, v = (FORMATS[name] for name in BODY_PRESETS[body])
    if k.q8 and v.q8:
        return KVCacheLayout(body, "", k, v, 0, 0)
    exact = exact or DEFAULT_EXACT
    if exact not in EXACT_PRESETS:
        raise ValueError(f"Unknown KV cache exact-token preset {exact!r}; expected one of {', '.join(EXACT_PRESETS)}.")
    sink_pages, recent = EXACT_PRESETS[exact]
    assert recent == 0 or (recent & (recent - 1) == 0 and recent % TILE == 0), "the ring size must be a power of two multiple of 32"
    return KVCacheLayout(body, exact, k, v, sink_pages, recent)


def paged_kv_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int, layout: str | KVCacheLayout | None = None, page_size: int = 256,
                   dtype_bytes: int = 2, overlays: bool = True) -> int:
    """VRAM of the keys and values of tokens in the paged attention layers: 16-bit (dtype_bytes) without a layout, else the body of the
    layout (a config value or a KVCacheLayout) and, unless overlays is False, its exact overlays of one sequence."""
    if not layout:
        return layers * 2 * kv_heads * head_dim * dtype_bytes * tokens
    layout = parse_layout(layout) if isinstance(layout, str) else layout
    return layout.body_bytes_per_token(layers, kv_heads, head_dim) * tokens + (layout.overlay_bytes(layers, kv_heads, head_dim, page_size, 1) if overlays else 0)


def layout_of(config) -> KVCacheLayout | None:
    """The layout of an engine Config's cache: None for a BF16 cache."""
    return parse_layout(config.kv_cache_layout or "int8") if config.kv_cache_int8 else None


def overlay_slots(config) -> int:
    """Exact-overlay slots the block manager hands out: one per sequence when the layout has overlays."""
    layout = layout_of(config)
    return int(config.max_num_seqs) if layout is not None and layout.overlay else 0


def is_layout_value(value) -> bool:
    try:
        parse_layout(value)
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------------------------------------------------------------------
# PyTorch reference of the formats (any device): the store kernel, the dense fallbacks and the tests use it.

def quantize(x: torch.Tensor, fmt: KVFormat) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """x [..., GROUP] floating: (packed data [..., data_bytes] in fmt.dtype, scale fp16 [...], min fp16 [...] or None)."""
    x = x.float()
    half = 1 << (fmt.bits - 1)
    if fmt.q8:
        scale = x.abs().amax(-1).div_(127).clamp_min_(1e-8)
        q = x.div(scale.unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8)
        return q, scale.to(torch.float16), None
    if fmt.asymmetric:
        lo, hi = x.amin(-1), x.amax(-1)
        d = (hi - lo) / (2 ** fmt.bits - 1)
        inverse = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
        q = (x - lo.unsqueeze(-1)).mul_(inverse.unsqueeze(-1)).round_().clamp_(0, 2 ** fmt.bits - 1).to(torch.uint8)
        scale, minimum = d.to(torch.float16), lo.to(torch.float16)
    else:
        extreme = x.gather(-1, x.abs().argmax(-1, keepdim=True)).squeeze(-1)
        d = extreme / -half
        inverse = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
        q = x.mul(inverse.unsqueeze(-1)).round_().add_(half).clamp_(0, 2 ** fmt.bits - 1).to(torch.uint8)
        scale, minimum = d.to(torch.float16), None
    low = q & 15
    packed = low[..., :16] | (low[..., 16:] << 4)
    if fmt.bits == 6:
        high = (q >> 4).reshape(*q.shape[:-1], 2, 4, 4)  # (half, r, b): value half * 16 + r * 4 + b
        plane = high[..., 0, :] | (high[..., 1, :] << 2) | (high[..., 2, :] << 4) | (high[..., 3, :] << 6)  # (half, b): byte half * 4 + b
        packed = torch.cat((packed, plane.reshape(*q.shape[:-1], 8)), -1)
    return packed, scale, minimum


def dequantize(packed: torch.Tensor, scale: torch.Tensor, minimum: torch.Tensor | None, fmt: KVFormat, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The inverse of quantize: [..., GROUP] values in dtype."""
    if fmt.q8:
        return packed.to(dtype) * scale.to(dtype).unsqueeze(-1)
    nibbles = packed[..., :16]
    q = torch.cat((nibbles & 15, nibbles >> 4), -1)
    if fmt.bits == 6:
        plane = packed[..., 16:24].reshape(*packed.shape[:-1], 2, 4)  # (half, b)
        bits = torch.stack((plane & 3, (plane >> 2) & 3, (plane >> 4) & 3, plane >> 6), -2).reshape(*packed.shape[:-1], GROUP)  # (half, r, b)
        q = q | (bits << 4)
    if fmt.asymmetric:
        return q.to(dtype) * scale.to(dtype).unsqueeze(-1) + minimum.to(dtype).unsqueeze(-1)
    return (q.to(dtype) - (1 << (fmt.bits - 1))) * scale.to(dtype).unsqueeze(-1)


def quantize_heads(x: torch.Tensor, fmt: KVFormat) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """x [..., head_dim]: the body record of each head, (data [..., width], scales [..., head_dim // 32], mins or None)."""
    groups = x.reshape(*x.shape[:-1], x.shape[-1] // GROUP, GROUP)
    packed, scale, minimum = quantize(groups, fmt)
    return packed.reshape(*x.shape[:-1], -1), scale, minimum


def dequantize_heads(data: torch.Tensor, scale: torch.Tensor, minimum: torch.Tensor | None, fmt: KVFormat, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The inverse of quantize_heads: [..., head_dim] values."""
    groups = data.reshape(*data.shape[:-1], scale.shape[-1], fmt.data_bytes)
    return dequantize(groups, scale, minimum, fmt, dtype).reshape(*data.shape[:-1], scale.shape[-1] * GROUP)
