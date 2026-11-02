# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""VRAM savings shared by model attentions: a model calls qkv_attention() (joint_qkv_attention() for several token streams) instead of
projecting q/k/v and calling pay_attention.

- SageAttention 2 (sm86/sm89/sm120, and its masked Triton path on every architecture) and SageAttention (1, without a mask): v, k and q
  are quantized as soon as each is projected (staged functions of shared.sage2_core / shared.sage1_core), so at most one of them exists
  in 16 bits (with the Triton kernels v stays in fp16); same results as pay_attention.
- The three projections share one input prepared by mmgp (offload.prepare_linear_input) when their weight format supports it: the
  input is quantized once instead of once per projection, and the 16-bit input is released before the projections.
- Attention Head Split (Performance tab, off by default): long sequences (all batch items together) are attended one group of heads
  at a time into one output. q/k/v rows, per-head norms, RoPE and attention are per head; norms over all heads get per token
  statistics computed beforehand.
- Joint attentions: the projections of several streams (text and image tokens) are joined along the tokens in their order, and keys
  and values computed beforehand (a cached prefix) can be put before the projected ones.
- Fused quantization (fused_quant): with a staged SageAttention 2 core, a model whose q/k norm and RoPE have a fused kernel
  (shared.kernels.ops.split_half_norm_rope_quant_) quantizes q and k straight from the projection output, without the in-place
  16-bit pass and the core's separate quantizers (and, for k, its mean and subtraction passes).
- Cached prefixes (cache_prefix, q8_prefix; PrefixCache for a model's layers, in VRAM or streamed from RAM): keys and values computed
  once and attended at every step (reference images, a history) kept quantized, about half of their 16-bit memory. With a staged SageAttention core they keep its own quantized layout and are
  attended in a pass of their own, merged with the projected keys by log-sum-exp: the error of attending the joined keys, and no joined
  16-bit copy at each step. Otherwise they are q8 (int8 per token, head and 32 channels): dequantized into the joined keys and values.
"""
import dataclasses
from collections import namedtuple

import torch
from tqdm import tqdm
from mmgp import offload, runtime

from shared.attention import pay_attention, sage_staged, sage_attention_mask
from shared.kernels import dispatch

try:
    from shared.kernels import kv_prefix_triton as _kv_triton
except ImportError:  # Triton not installed: the PyTorch paths
    _kv_triton = None

HEAD_SPLIT_CHOICES = [("Off", 0), ("Low: saves some VRAM", 1), ("Medium (good balance): saves more VRAM", 2), ("High: saves the most VRAM", 3)]
_TARGET_GROUPS = (1, 4, 8, 16)
MIN_SPLIT_TOKENS = 8192  # fewer query tokens (all batch items together) are never split: little VRAM to save
_ROW_ELEMENTS = 1 << 23  # rows_ chunk size, in elements of the tensor
head_split = 0
_unsplit_notice = False

# Head ranges of a group for q and for k/v; mean_squares: per token mean of the squares of q and of k over all heads, (batch, tokens, 1)
# in float32, when a norm spans all heads and the heads are split, otherwise None.
HeadGroup = namedtuple("HeadGroup", "q kv mean_squares")
# A cached prefix of keys and values (cache_prefix, q8_prefix), tensors only (CUDA graphs take it as a block argument), batch first:
# - q8: k, v (batch, tokens, kv_heads, head_dim) int8 with k_scale, v_scale (..., head_dim // 32) FP16; k_mean, v_mean None.
# - native (k_mean set): k, k_scale as the staged core quantizes keys smoothed by k_mean, the prefix's mean, which the keys attended with
#   it subtract too (so that the two passes' log-sum-exps share it); v, v_scale, v_mean as the core quantizes values (FP8 of the
#   SageAttention 2 CUDA kernels), or q8 v (16-bit v of the Triton kernels).
KVPrefix = namedtuple("KVPrefix", "k k_scale k_mean v v_scale v_mean")
GROUP = 32  # channels per q8 scale (the q8 format of the LM engines' KV cache, shared.llm_engines.lmgp.layers.kv_formats)
_Q8_ROWS = 4096  # tokens quantized / dequantized at a time by the PyTorch paths (FP32 scratch)
_TRITON_DIMS = (64, 128, 256)  # last dimensions of the Triton kernels (power-of-two tiles)


def configure(level):
    global head_split
    head_split = int(level)


@torch.compiler.disable()
def kv_first_attention(qkv_list, force_attention=None):
    """pay_attention(qkv_list, force_attention=force_attention, recycle_q=True) when nothing else holds k and v: with SageAttention 2
    (sm86/sm89/sm120) or SageAttention, k is quantized and released before v is copied for its quantization, a peak lower by half of k;
    same results."""
    staged = sage_staged(qkv_list[0].device, qkv_list[0].shape[-1], force_attention)
    if staged is None:
        return pay_attention(qkv_list, force_attention=force_attention, recycle_q=True)
    core, settings = staged
    query, key_list, value_list = qkv_list[0], qkv_list[1:2], qkv_list[2:3]
    qkv_list.clear()
    quantized = [None, core.staged_quantize_k(key_list, settings), core.staged_quantize_v(value_list, settings)]
    quantized[0] = core.staged_quantize_q(query, settings)
    return core.staged_attention(query, quantized, settings)


def head_groups(kv_heads, tokens, level=None):
    """Groups of heads of an attention over tokens (of all batch items): the divisor of kv_heads closest to the level's target, 1 when
    off."""
    level = head_split if level is None else level
    if level == 0 or tokens < MIN_SPLIT_TOKENS:
        return 1
    return min((groups for groups in range(1, kv_heads + 1) if kv_heads % groups == 0), key=lambda groups: (abs(groups - _TARGET_GROUPS[level]), groups))


def rows_(tensor, fn, start=0, stop=None):
    """tensor[:, first:last] = fn(tensor[:, first:last], first, last) by chunks of tokens of [start, stop), in place: fn computes each
    row on its own (norm, RoPE, possibly in place), so the result is that of fn on the whole range, with a bounded scratch."""
    stop = tensor.shape[1] if stop is None else stop
    step = max(1, _ROW_ELEMENTS // max(1, tensor[:, :1].numel()))
    for first in range(start, stop, step):
        last = min(first + step, stop)
        part = tensor[:, first:last]
        part.copy_(fn(part, first, last))
        del part


def _kernels_eligible(out):
    """The Triton kernels of kv_prefix_triton for this 16-bit output: compute capability 7.5 and up (BF16 as bits below 8.0)."""
    return (_kv_triton is not None and out.shape[-1] in _TRITON_DIMS
            and dispatch.cuda_inference(out, dtypes=(torch.bfloat16, torch.float16), capability_floor=(7, 5)))


def quantize_q8(x):
    """x (batch, tokens, ..., channels) in the q8 format: (int8 data, FP16 scales (..., channels // 32)), each group of 32 channels
    scaled by max|x| / 127 (that of kv_formats.Q8)."""
    data = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((*x.shape[:-1], x.shape[-1] // GROUP), dtype=torch.float16, device=x.device)
    for start in range(0, x.shape[1], _Q8_ROWS):
        stop = start + _Q8_ROWS
        part = x[:, start:stop].float().unflatten(-1, (-1, GROUP))
        scale = part.abs().amax(-1).div_(127).clamp_min_(1e-8)
        data[:, start:stop] = torch.div(part, scale.unsqueeze(-1)).round_().clamp_(-127, 127).flatten(-2)  # part may be x itself (FP32)
        scales[:, start:stop] = scale
        del part, scale
    return data, scales


_q8 = quantize_q8


def dequantize_q8_into(out, data, scales):
    """out (batch, tokens, heads, head_dim) 16-bit, written: q8 data (batch or 1, ...) with its scales, through strides."""
    if _kernels_eligible(out):
        return _kv_triton.dequantize_q8_into(out, data, scales)
    for start in range(0, out.shape[1], _Q8_ROWS):
        stop = start + _Q8_ROWS
        out[:, start:stop] = (data[:, start:stop].float().unflatten(-1, (-1, GROUP)) * scales[:, start:stop].float().unsqueeze(-1)).flatten(-2)
    return out


def _merge_lse_(out, lse, other, other_lse):
    """In place: out = the attention over both key sets of queries attended over one (out, lse: base-2 log-sum-exp (batch, heads,
    tokens)) and over the other (other, other_lse)."""
    if _kernels_eligible(out):
        return _kv_triton.merge_lse_(out, lse, other, other_lse)
    for start in range(0, out.shape[1], _Q8_ROWS):
        stop = start + _Q8_ROWS
        a, b = lse[:, :, start:stop], other_lse[:, :, start:stop]
        top = torch.maximum(a, b)
        a, b = torch.exp2(a - top), torch.exp2(b - top)
        total = a + b
        a, b = (a / total).transpose(1, 2).unsqueeze(-1), (b / total).transpose(1, 2).unsqueeze(-1)
        out[:, start:stop] = out[:, start:stop].float() * a + other[:, start:stop].float() * b
    return out


def cache_prefix(kv_list, native=True):
    """A cached prefix (KVPrefix) for joint_qkv_attention's kv_prefix, of keys and values (batch, tokens, kv_heads, head_dim) normalized
    and rotated, handed off in kv_list ([key, value], cleared; both are consumed). native: in the quantized layout of the staged
    SageAttention core that will attend it where that core supports it (unmasked attentions without attention_fn: pass False for the
    others), attended in a pass of its own and merged by log-sum-exp, the error of attending the joined keys; otherwise q8, dequantized
    into the joined keys and values at each attention (a small error on 16-bit backends). About half of the 16-bit memory either way.
    A block recorded for a CUDA graph keeps the settings of an eager one, so the cached steps can be recorded."""
    key, value = kv_list
    kv_list.clear()
    staged = sage_staged(key.device, key.shape[-1]) if native else None
    if staged is not None and hasattr(staged[0], "prefix_supported") and staged[0].prefix_supported(staged[1], key.device, key.dtype):
        core = staged[0]
        settings = core.prefix_settings(staged[1], key.device, offload.shared_state.get("_sage_v_scaling", False))
        mean = core.prefix_mean(key)
        k, k_scale = core.prefix_quantize_k([key], settings, mean)
        if core.prefix_fp8_v(settings):
            v, v_scale, v_mean = core.staged_quantize_v([value], settings)
        else:
            (v, v_scale), v_mean = _q8(value), None
        return KVPrefix(k, k_scale, mean, v, v_scale, v_mean)
    k, k_scale = _q8(key)
    del key
    v, v_scale = _q8(value)
    return KVPrefix(k, k_scale, None, v, v_scale, None)


def q8_prefix(k, k_scale, v, v_scale):
    """A cached prefix of keys and values already in the q8 format (shared.llm_engines.lmgp.layers.kv_formats.Q8: int8 (batch, tokens,
    kv_heads, head_dim), or without batch, with FP16 scales per 32 channels), as kept by an archive: not copied."""
    k, k_scale, v, v_scale = (t if t.dim() == 4 else t.unsqueeze(0) for t in (k, k_scale, v, v_scale))
    return KVPrefix(k, k_scale, None, v, v_scale, None)


def dequantize_prefix(prefix, dtype=torch.bfloat16):
    """(keys, values) in dtype of a q8 cached prefix."""
    return tuple(dequantize_q8_into(torch.empty(data.shape, dtype=dtype, device=data.device), data, scales)
                 for data, scales in ((prefix.k, prefix.k_scale), (prefix.v, prefix.v_scale)))


DYNAMIC_PREFIX = True  # comparison switch: PrefixCache(dynamic=True) places the cache dynamically; False keeps it on the GPU


@dataclasses.dataclass
class PrefixLayer:  # a dataclass: CUDA graphs take its tensors as block arguments
    """One layer of a PrefixCache: its cached keys / values, 16-bit (k, v) or quantized (prefix), or with a state ring the list of tensors
    the ring returned (stored), whose items it rebinds as the layer moves between VRAM and its streaming slot: read at each use."""
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    prefix: KVPrefix | None = None
    stored: list | None = None
    present: tuple = ()  # with a ring: which of the values (k, v or the fields of a KVPrefix) are stored, the others None
    index: int = 0
    quantize: bool = False
    native: bool = True  # a quantized prefix may keep the layout of the kit's staged attention (never read back in 16 bits then)

    def store(self, k, v, kit=False):
        """k / v (batch, tokens, kv_heads, head_dim), normalized and rotated (consumed when quantized); kit: attended through the
        unmasked attention of joint_qkv_attention (a native prefix then). With a state ring the layer goes to the ring (RAM, or VRAM of
        its own with a dynamic ring), its tensors being those the ring binds (holding them for this pass already)."""
        values = cache_prefix([k, v], native=kit and self.native) if self.quantize else (k, v)
        ring = self.owner.ring
        if ring is not None:
            self.present = tuple(t is not None for t in values)
            self.stored = ring.store(self.index, [t for t in values if t is not None], current=True, adopt=True)  # values made for the cache
        elif self.quantize:
            self.prefix = values
        else:
            self.k, self.v = values

    def _values(self):
        if self.stored is None:
            return self.prefix if self.quantize else (self.k, self.v)
        items = iter(self.stored)
        values = [next(items) if present else None for present in self.present]
        return KVPrefix(*values) if self.quantize else tuple(values)

    def empty(self):
        return self.k is None and self.prefix is None and self.stored is None

    def kv_prefix(self):
        """The kv_prefix of joint_qkv_attention."""
        if self.empty():
            raise RuntimeError("KV cache has not been populated yet.")
        return self._values()

    def get(self):
        """The 16-bit keys / values (attentions outside the kit: masks, NAG)."""
        if self.empty():
            raise RuntimeError("KV cache has not been populated yet.")
        return dequantize_prefix(self._values()) if self.quantize else self._values()

    def clear(self):
        self.k = self.v = self.prefix = self.stored = None


class PrefixCache:
    """Cached keys / values of a model's attention layers: a prefix (reference images, a history) computed once and attended at the
    following passes. One PrefixLayer per block, which the model passes to its block: layer.store(k, v) where it computes them,
    layer.kv_prefix() for joint_qkv_attention afterwards.
    - quantize: INT8 (cache_prefix: about half the memory, the layout of SageAttention where it attends them).
    - ram: kept in RAM and streamed to the GPU block by block with the weights (mmgp.runtime.state_ring, locked RAM as far as the reserved
      RAM allows): the VRAM of two layers. blocks: the model's blocks (a ModuleList called one by one), whose forward then fetches the
      layer's cache; without ram, blocks may be a number of layers.
    - dynamic (without ram): kept in VRAM as far as the model's VRAM allows, planned with its weights by its dynamic preload, the layers that
      do not stay there kept in RAM and streamed as with ram (a state ring with resident="dynamic"), where mmgp.runtime.dynamic_state(blocks);
      otherwise kept on the GPU.
    clear() at the end of the generation (clear(keep_ring=True) to store the layers again, e.g. when step-dependent LoRAs change them)."""

    def __init__(self, blocks, quantize=False, native=True, ram=False, name="KV cache", dynamic=False):
        if ram:
            self.ring = runtime.state_ring(blocks, name)
        elif dynamic and DYNAMIC_PREFIX and not isinstance(blocks, int) and runtime.dynamic_state(blocks):
            self.ring = runtime.state_ring(blocks, name, resident="dynamic")
        else:
            self.ring = None
        self.layers = [PrefixLayer(index=index, quantize=quantize, native=native) for index in range(blocks if isinstance(blocks, int) else len(blocks))]
        for layer in self.layers:
            layer.owner = self  # not a field: not a block argument

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]

    def empty(self):
        return self.layers[0].empty()

    def clear(self, keep_ring=False):
        for layer in self.layers:
            layer.clear()
        if self.ring is not None:
            self.ring.reset() if keep_ring else self.ring.release()


def _attend_with_prefix(core, settings, out, quantized, prefix, heads):
    """The staged attention of the quantized [q, k, v] written into out, merged with the queries' attention over the heads of a native
    cached prefix (two passes sharing the prefix's key mean)."""
    batch = out.shape[0]
    q_quantized = quantized[0]
    k = core.prefix_k(prefix.k, prefix.k_scale, heads, batch)
    if prefix.v.dtype == torch.int8:  # q8 v of a core reading 16-bit v
        # FP16, which these kernels read (the v of the joined path is made FP16 too), unless v is scaled for FP16 overflows
        value = out.new_empty((batch, prefix.v.shape[1], heads.stop - heads.start, prefix.v.shape[-1]), dtype=out.dtype if settings.get("scale_v") else torch.float16)
        dequantize_q8_into(value, prefix.v[:, :, heads.start:heads.stop], prefix.v_scale[:, :, heads.start:heads.stop])
        v = core.staged_quantize_v([value], settings)
        del value
    else:
        v = core.prefix_v(prefix.v, prefix.v_scale, prefix.v_mean, heads, batch)
    prefix_out, prefix_lse = core.staged_attention(torch.empty(out.shape, dtype=out.dtype, device=out.device), [q_quantized, k, v], settings, return_lse=True)
    del k, v
    out, lse = core.staged_attention(out, quantized, settings, return_lse=True)
    return _merge_lse_(out, lse, prefix_out, prefix_lse)


def _head_slice(mask, dim, heads):
    """mask restricted to the range heads along its head dimension dim, when it has one row per head."""
    return mask if mask is None or mask.dim() < 4 or mask.shape[dim] == 1 else mask.narrow(dim, heads.start, heads.stop - heads.start)


def qkv_attention(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, extra_proj=None, **kwargs):
    """Attention of the q/k/v projections of x (handed off in x_list, shape (batch, tokens, features) or (tokens, features)), returned
    as (batch, tokens, heads, head_dim), batch 1 for a 2D x. See joint_qkv_attention for norm_rope and the options; extra_proj is a
    fourth per-head projection of x (heads x head_dim wide) computed by head groups from the same prepared input and handed to
    attention_fn as a third argument."""
    projections = (q_proj, k_proj, v_proj) + ((extra_proj,) if extra_proj is not None else ())
    return joint_qkv_attention([(x_list, projections)], heads, head_dim, norm_rope, **kwargs)


def cross_qkv_attention(x_list, context_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, **kwargs):
    """The same projection/attention path with queries from x and keys/values from context, both handed off in lists.
    Head splitting considers the longer sequence, including a long video used as keys for short audio queries."""
    return joint_qkv_attention([(x_list, (q_proj, None, None)), (context_list, (None, k_proj, v_proj))],
                              heads, head_dim, norm_rope, _cross=True, **kwargs)


@torch.compiler.disable()
def joint_qkv_attention(streams, heads, head_dim, norm_rope, kv_heads=None, norm_spans_heads=False, split_heads=True, kv_prefix=None,
                        attention_mask=None, force_attention=None, attention_fn=None, norm_stats=None, _cross=False, norm_v=None, softmax_scale=None,
                        head_split_level=None, fused_quant=None):
    """Attention of the q/k/v projections of several streams joined along the tokens in their order: streams is a list of
    (x_list, (q_proj, k_proj, v_proj)), each x handed off in its list, of shape (batch, tokens, features) or (tokens, features).
    Returns (batch, all tokens, heads, head_dim).

    norm_rope(query, key, group) normalizes and rotates in place query and/or key, the other one may be None: the joined projections,
    (batch, all tokens, heads of the group, head_dim), for the head ranges group.q and group.kv. With norm_spans_heads, a split group
    gets group.mean_squares to scale its rows; otherwise the tensors hold all heads. split_heads=False keeps all heads together.
    attention_fn: optional model-specific sparse attention accepting the disposable qkv_list and HeadGroup. It owns attention
    computation; projection preparation, head splitting and normalization still use this kit. Staged Sage quantization is disabled.
    kv_prefix: (key, value) of shape (batch, prefix tokens, kv_heads, head_dim), already normalized and rotated, put before the
    projected keys and values; or a cached prefix (cache_prefix, q8_prefix: KVPrefix), q8 dequantized into the joined keys and values,
    native attended in a pass of its own (a native prefix needs the staged core it was made for: no mask, no attention_fn), a prefix
    of one batch item serving all. attention_mask: pay_attention's mask, or on a staged core a callable mask(query, key, group) building
    the group's mask from its normalized and rotated query and key (the key with its prefix): called with the key alone first (it
    keeps what it needs, the key is then quantized and released), then with the query alone, returning the mask.
    kv_heads < heads: grouped query attention.
    norm_stats(project): optional model-native mean-square calculation, used only when splitting a norm spanning all heads;
    project(which, head_range) supplies disposable projections. Otherwise statistics are accumulated by groups below.
    norm_v(value, group): optional in-place value normalization, before prefix concatenation or quantization.
    When every K/V stream shares its k and v module, reuse its projection with independent tensors for normalization.
    softmax_scale: explicit SDPA scale; requires force_attention="sdpa".
    head_split_level: Attention Head Split level of this call instead of the Performance tab's (a model-specific sparse attention
    bounding its working set). A stream may carry a fourth projection: attention_fn then receives its head group's rows in a list.
    fused_quant(query, key, group, core, settings): optional norm + RoPE fused with a staged core's quantization (for example
    shared.kernels.ops.split_half_norm_rope_quant_), returning (q_quantized, k_quantized) for the tensors given (the other None) as
    the core's staged_quantize_q / staged_quantize_k do, or None to run norm_rope and the core's quantizers instead; used by the
    staged cores unless the attention_mask is a callable (it reads the normalized 16-bit q and k); with a kv_prefix only q is fused
    (the prefix joins the normalized 16-bit k), except with a native prefix (k fused with the prefix's mean: settings["k_mean"])."""
    global _unsplit_notice
    if softmax_scale is not None and (force_attention != "sdpa" or attention_fn is not None):
        raise ValueError("An explicit softmax_scale requires force_attention='sdpa' and no attention_fn")
    inputs, projections = [], []
    for x_list, stream_projections in streams:
        x = x_list[0]
        x_list.clear()
        inputs.append(x if x.dim() > 2 else x.unsqueeze(0))
        projections.append(stream_projections)
    streams = x = None
    shared_kv = projections[1][1] is projections[1][2] if _cross else all(modules[1] is modules[2] for modules in projections)
    extra = not _cross and len(projections[0]) > 3
    batch, lengths, dtype = inputs[0].shape[0], [x.shape[1] for x in inputs], inputs[0].dtype
    tokens = lengths[0] if _cross else sum(lengths)
    kv_heads = kv_heads or heads
    groups = head_groups(kv_heads, batch * (max(lengths) if _cross else tokens), head_split_level) if split_heads else 1
    staged = None if attention_fn is not None else sage_staged(inputs[0].device, head_dim, force_attention, attention_mask is not None)
    if callable(attention_mask) and staged is None:
        raise ValueError("A per-group attention mask (a callable) needs a staged attention core")
    cached = isinstance(kv_prefix, KVPrefix)
    native = cached and kv_prefix.k_mean is not None
    if native and (staged is None or attention_mask is not None or not hasattr(staged[0], "prefix_k")):
        raise ValueError("A native cached prefix is attended by the unmasked staged attention it was made for (cache_prefix native=False otherwise)")
    linear_inputs = [None] * len(inputs)
    for index, modules in enumerate(projections):
        if _cross:
            modules = tuple(module for module in modules if module is not None)
        if shared_kv:
            modules = tuple(dict.fromkeys(modules))
        if offload.linear_rows_supported(modules, inputs[index]):
            x_holder, inputs[index] = [inputs[index]], None
            linear_inputs[index] = offload.prepare_linear_input(x_holder, modules)
    # a stream whose projections cannot be computed by groups of heads is projected whole once and sliced, if it is too short to be split
    if groups > 1 and batch * sum(length for length, prepared in zip(lengths, linear_inputs) if prepared is None) >= MIN_SPLIT_TOKENS:
        if not _unsplit_notice:
            tqdm.write("[Attention] Head Split is not applied: these q/k/v projections cannot be computed by groups of heads (weight format or LoRA type).")
            _unsplit_notice = True
        groups = 1
    whole = {}

    def project_stream(index, which, head_range):
        module, linear_input = projections[index][which], linear_inputs[index]
        if linear_input is not None:
            return offload.linear_rows(module, linear_input, head_range.start * head_dim, head_range.stop * head_dim).view(batch, lengths[index], -1, head_dim)
        if groups == 1:
            return module(inputs[index]).view(batch, lengths[index], -1, head_dim)
        if (index, which) not in whole:
            whole[index, which] = module(inputs[index]).view(batch, lengths[index], -1, head_dim)
        return whole[index, which][:, :, head_range.start:head_range.stop]

    def project(which, head_range):
        """Projection which (0 q, 1 k, 2 v) of the heads head_range, the streams joined along the tokens."""
        if _cross:
            return project_stream(0 if which == 0 else 1, which, head_range)
        if len(lengths) == 1:
            return project_stream(0, which, head_range)
        joined, start = None, 0
        for index, length in enumerate(lengths):
            part = project_stream(index, which, head_range)
            if joined is None:
                joined = part.new_empty((batch, tokens, *part.shape[2:]))
            joined[:, start:start + length].copy_(part)
            start += length
            del part
        return joined

    def release():
        inputs.clear()
        linear_inputs.clear()
        whole.clear()

    def with_prefix(tensor, which, head_range):
        if kv_prefix is None or native:  # a native prefix is attended in a pass of its own
            return tensor
        if not cached:
            return torch.cat([kv_prefix[which - 1][:, :, head_range.start:head_range.stop], tensor], dim=1)
        data, scales = (kv_prefix.k, kv_prefix.k_scale) if which == 1 else (kv_prefix.v, kv_prefix.v_scale)
        joined = tensor.new_empty((tensor.shape[0], data.shape[1] + tensor.shape[1], *tensor.shape[2:]))
        dequantize_q8_into(joined[:, :data.shape[1]], data[:, :, head_range.start:head_range.stop], scales[:, :, head_range.start:head_range.stop])
        joined[:, data.shape[1]:].copy_(tensor)
        return joined

    mean_squares = None
    if norm_spans_heads and groups > 1 and norm_stats is not None:
        mean_squares = norm_stats(project)
    elif norm_spans_heads and groups > 1:
        mean_squares = []
        for which, total in ((0, heads), (1, kv_heads)):
            step, squares = total // groups, None
            for group in range(groups):
                part = project(which, range(group * step, (group + 1) * step)).float().square_().sum(dim=(-2, -1))
                squares = part if squares is None else squares.add_(part)
                del part
            mean_squares.append(squares.div_(total * head_dim).unsqueeze(-1))
        mean_squares = tuple(mean_squares)

    if staged is not None:
        core, settings = staged
        if native:  # the layout the prefix was kept in (cache_prefix)
            settings = core.prefix_settings(settings, kv_prefix.k.device, offload.shared_state.get("_sage_v_scaling", False))
        if attention_mask is not None and not callable(attention_mask):  # in SageAttention's layout (batch, heads, q, k)
            attention_mask = sage_attention_mask(attention_mask, dtype)
    # a callable mask reads the normalized 16-bit q and k; a prefix joins the normalized 16-bit k (q alone is still fused), not a native one
    fusing_q = staged is not None and fused_quant is not None and not callable(attention_mask)
    fusing_k = fusing_q and (kv_prefix is None or native)
    q_heads, kv_group_heads = heads // groups, kv_heads // groups
    output = None
    for index in range(groups):
        group = HeadGroup(range(index * q_heads, (index + 1) * q_heads), range(index * kv_group_heads, (index + 1) * kv_group_heads), mean_squares)
        last = index == groups - 1
        if staged is None:
            qkv_list = [project(0, group.q), project(1, group.kv)]
            qkv_list.append(qkv_list[1].clone() if shared_kv else project(2, group.kv))
            extra_list = [project(3, group.q)] if extra else None
            if last:
                release()
            if norm_v is not None:
                norm_v(qkv_list[2], group)
            norm_rope(qkv_list[0], qkv_list[1], group)
            qkv_list[1:] = [with_prefix(qkv_list[which], which, group.kv) for which in (1, 2)]
            if q_heads != kv_group_heads:  # pay_attention's backends do not all take fewer key/value heads
                qkv_list[1:] = [tensor.repeat_interleave(q_heads // kv_group_heads, dim=2) for tensor in qkv_list[1:]]
            if attention_fn is None:
                attention = pay_attention(qkv_list, attention_mask=_head_slice(attention_mask, 2, group.q), force_attention=force_attention, recycle_q=True, softmax_scale=softmax_scale)
            else:
                attention = attention_fn(qkv_list, group) if extra_list is None else attention_fn(qkv_list, group, extra_list)
            if groups == 1:
                return attention
            if output is None:
                output = attention.new_empty((batch, tokens, heads, head_dim))
            output[..., group.q.start:group.q.stop, :].copy_(attention)
            del attention
            continue
        key = project(1, group.kv) if shared_kv else None
        value = key.clone() if shared_kv else project(2, group.kv)
        if norm_v is not None:
            norm_v(value, group)
        value_list = [with_prefix(value, 2, group.kv)]
        del value
        quantized = [None, None, core.staged_quantize_v(value_list, settings)]
        # the keys of a native prefix's passes share its mean
        k_settings = {**settings, "k_mean": core.prefix_mean_heads(kv_prefix.k_mean, group.kv)} if native else settings
        if groups == 1:  # all heads: k is quantized and released before q is projected
            if not shared_kv:
                key = project(1, group.kv)
            fused = fused_quant(None, key, group, core, k_settings) if fusing_k else None
            if fused is None:
                norm_rope(None, key, group)
                key = with_prefix(key, 1, group.kv)
                if callable(attention_mask):
                    attention_mask(None, key, group)
                quantized[1] = core.staged_quantize_k([key], k_settings)
            else:
                quantized[1] = fused[1]
            del key, fused
            query = project(0, group.q)
            release()
            fused = fused_quant(query, None, group, core, settings) if fusing_q else None
            if fused is None:
                norm_rope(query, None, group)
                mask = attention_mask(query, None, group) if callable(attention_mask) else attention_mask
                quantized[0] = core.staged_quantize_q(query, settings)
            else:
                mask, quantized[0] = attention_mask, fused[0]
            del fused
            if native:
                return _attend_with_prefix(core, settings, query, quantized, kv_prefix, group.kv)
            return core.staged_attention(query, quantized, settings, mask)
        query = project(0, group.q)
        if not shared_kv:
            key = project(1, group.kv)
        if last:
            release()
        fused = fused_quant(query, key, group, core, k_settings) if fusing_k else fused_quant(query, None, group, core, settings) if fusing_q else None
        if fused is None or fused[1] is None:
            norm_rope(query if fused is None else None, key, group)
            key = with_prefix(key, 1, group.kv)
            if callable(attention_mask):
                attention_mask(None, key, group)
            quantized[1] = core.staged_quantize_k([key], k_settings)
        else:
            quantized[1] = fused[1]
        del key
        if fused is None:
            mask = attention_mask(query, None, group) if callable(attention_mask) else _head_slice(attention_mask, 1, group.q)
            quantized[0] = core.staged_quantize_q(query, settings)
        else:
            mask, quantized[0] = _head_slice(attention_mask, 1, group.q), fused[0]
        del fused
        if output is None:
            output = query.new_empty((batch, tokens, heads, head_dim))
        if native:
            _attend_with_prefix(core, settings, output[..., group.q.start:group.q.stop, :], quantized, kv_prefix, group.kv)
        else:
            core.staged_attention(output[..., group.q.start:group.q.stop, :], quantized, settings, mask)
        del query
    return output
