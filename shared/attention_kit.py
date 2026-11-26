# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""VRAM savings shared by model attentions: a model calls qkv_attention() (joint_qkv_attention() for several token streams) instead of
projecting q/k/v and calling pay_attention.

- SageAttention 2 (sm86/sm89/sm120, and its masked Triton path on every architecture) and SageAttention (1, without a mask): v, k and q
  are quantized as soon as each is projected (staged functions of shared.sage2_core / shared.sage1_core), so at most one of them exists
  in 16 bits (with the Triton kernels v stays in fp16); same results as pay_attention.
- The three projections share one input prepared by mmgp (offload.prepare_linear_input) when their weight format supports it: the
  input is quantized once instead of once per projection, and the 16-bit input is released before the projections.
- Smart Inputs Chunker (Performance tab: Off, Auto by default, Standard, Extreme): long sequences (all batch items together) are attended one group
  of heads at a time into one output, sized by the Peak Fit analysis (head_partition). q/k/v rows, per-head norms, RoPE and attention
  are per head; norms over all heads get per token statistics computed beforehand. Auto splits only the calls whose whole attention
  would not fit in the card (chunking_needed).
- Joint attentions: the projections of several streams (text and image tokens) are joined along the tokens in their order, and keys
  and values computed beforehand (a cached prefix) can be put before the projected ones.
- Fused quantization (fused_quant): with a staged SageAttention 2 core, a model whose q/k norm and RoPE have a fused kernel
  (shared.kernels.ops.split_half_norm_rope_quant_) quantizes q and k straight from the projection output, without the in-place
  16-bit pass and the core's separate quantizers (and, for k, its mean and subtraction passes).
- Cached prefixes (cache_prefix, q8_prefix; PrefixCache for a model's layers, in VRAM or streamed from RAM): keys and values computed
  once and attended at every step (reference images, a history) kept quantized, about half of their 16-bit memory. With a staged SageAttention core they keep its own quantized layout and are
  attended in a pass of their own, merged with the projected keys by log-sum-exp: the error of attending the joined keys, and no joined
  16-bit copy at each step. Otherwise they are q8 (int8 per token, head and 32 channels): dequantized into the joined keys and values.
  Optional bits=4 uses the LM engines' packed q4_0 format, about half of q8's storage, dequantized per head group with every backend.
"""
import contextlib
import dataclasses
import time
from collections import namedtuple

import torch
from tqdm import tqdm
from mmgp import allocator, cuda_graphs, offload, runtime

from shared import sparge_core
from shared.attention import get_default_attention_mode, pay_attention, sage_staged, sage_attention_mask, launch_waves, fp4_geometry, dense_staged, kv_first_staged, owned_query_staged
from shared.kernels import dispatch, ops
from shared.llm_engines.lmgp.layers import kv_formats

try:
    from shared.kernels import kv_prefix_triton as _kv_triton
except ImportError:  # Triton not installed: the PyTorch paths
    _kv_triton = None

CHUNKER_CONFIG_KEY = "inputs_chunker"  # Smart Inputs Chunker (Performance tab): "off", "auto", "on", "extreme"
CHUNKER_CHOICES = [("Off", "off"), ("Auto: only when the allocated VRAM would run short (Extreme if it must)", "auto"), ("Standard: always, at the best speed for the VRAM saved", "on"),
                   ("Extreme: the lowest VRAM, a few heads at a time (slower)", "extreme")]
CHUNKER_DEFAULT = "auto"
_TARGET_GROUPS = (1, 4, 8, 16)
MIN_SPLIT_TOKENS = 8192  # fewer query tokens (all batch items together) are never split: little VRAM to save
_ROW_ELEMENTS = 1 << 23  # rows_ chunk size, in elements of the tensor
chunker = "off"
head_split = 0  # the level of the Peak Fit analysis (head_partition): 3 when the chunker may split (auto or on), 0 off
_unsplit_notice = False

# Head ranges of a group for q and for k/v; mean_squares: per token mean of the squares of q and of k over all heads, (batch, tokens, 1)
# in float32, when a norm spans all heads and the heads are split, otherwise None; rows: the range of tokens the tensors hold (of their
# stream: q's or the keys'), None for all of them (only with norm_rows: all heads by chunks of tokens).
HeadGroup = namedtuple("HeadGroup", "q kv mean_squares rows", defaults=(None,))
TOKEN_QK_BYTES = 64 * 1024 * 1024  # 16-bit q or k of all heads per chunk of tokens (norm_rows): a few large GEMMs, a bounded scratch
# norm_rows: which groups keep the q and k of the chunk pass (the others project theirs again with its statistics). "room": the first
# groups whose INT8 q and k fit in the measured free room (_room_keep: never above the peaks already reached); comparison switches
# "all" (every group: no second projection, all heads' q and k held), "none" (the statistics pass of norm_stats / STATS_WIDTH)
TOKEN_QK_POLICY = "room"
# A cached prefix of keys and values (cache_prefix, q8_prefix), tensors only (CUDA graphs take it as a block argument), batch first:
# - q8: k, v (batch, tokens, kv_heads, head_dim) int8 with k_scale, v_scale (..., head_dim // 32) FP16; k_mean, v_mean None.
# - q4_0: k, v uint8 (..., head_dim // 2), two channels per byte; the same FP16 scales and None means as q8.
# - native (k_mean set): k, k_scale as the staged core quantizes keys smoothed by k_mean, the prefix's mean, which the keys attended with
#   it subtract too (so that the two passes' log-sum-exps share it); v, v_scale, v_mean as the core quantizes values (FP8 of the
#   SageAttention 2 CUDA kernels), or q8 v (16-bit v of the Triton kernels).
KVPrefix = namedtuple("KVPrefix", "k k_scale k_mean v v_scale v_mean")
GROUP = kv_formats.GROUP  # channels per q8 / q4_0 scale, shared with the LM engines' KV cache
_Q8_ROWS = 4096  # tokens quantized / dequantized at a time by the PyTorch paths (FP32 scratch)
_TRITON_DIMS = (64, 128, 256)  # last dimensions of the Triton kernels (power-of-two tiles)


def configure(mode):
    """Smart Inputs Chunker: "off", "on" (inputs always chunked, by the Peak Fit analysis: head_partition, the equal groups of the
    former High where no launch probe applies), "auto" (chunked only where the whole input would not fit: chunking_needed) or
    "extreme" (always chunked, with EXTREME_LOW_VRAM and EXTREME_LORA_PRODUCTS, and the attention kernels' last waves split: the
    lowest VRAM, slower)."""
    global chunker, head_split, EXTREME_LOW_VRAM, EXTREME_LORA_PRODUCTS, EXTREME_GROUP_HEADS
    if mode not in ("off", "auto", "on", "extreme"):
        raise ValueError(f"Unknown Smart Inputs Chunker mode: {mode!r}")
    if mode != chunker:  # peaks reached in another mode say nothing about this one's (H3's final layer is chunked only with it)
        _Room.ceiling, _Room.last_end, _Room.pending = {}, None, {}
        _needed.clear()
        _extreme_levels.clear()
    chunker, head_split = mode, 0 if mode == "off" else 3
    EXTREME_LOW_VRAM = EXTREME_LORA_PRODUCTS = mode == "extreme"
    EXTREME_GROUP_HEADS = 1
    _last_wave_split(mode == "extreme")


# Extreme: the models' blocks run without CUDA graphs. Each block keeps the GPU busy far longer than the host takes to launch it (H3 1080p,
# 2.5 s per block: the same speed with graphs, which add their capture and 0.13 GiB of reserved VRAM). LM decoders' scopes are not gated.
cuda_graphs.register_gate(lambda: "the Smart Inputs Chunker is Extreme (blocks long enough to need none)" if extreme_low_vram() else None)


def _last_wave_split(on):
    """The attention kernels' last-wave splits (SageGP Precise, Fast and Adaptive on RTX 50xx, SageAttention 2 sm89 / Triton, the Turing kernel), with the
    extreme mode. chunker, part of the CUDA graph keys, tells the graphs of either apart."""
    import importlib
    for name, flag in (("shared.sage1_core", "LAST_WAVE_SPLIT"), ("shared.sage2_core", "LAST_WAVE_SPLIT"), ("sagegp.sm120.dense", "SPLIT_KEYS")):
        try:  # a module whose attention library is not installed has nothing to split
            setattr(importlib.import_module(name), flag, on)
        except ImportError:
            pass


def reset_chunking_measurements():
    """A new attention implementation needs its own allocation peaks and warmup partitions."""
    _Room.ceiling, _Room.pending, _Room.own, _Room.chosen = {}, {}, {}, {}
    _Room.last_end = None
    _needed.clear()
    _extreme_levels.clear()


AUTO_MARGIN = (512 << 20, 0.10)  # Auto leaves at least this much of the card free (bytes, share): the CUDA context, workspaces, fragmentation
_needed = {}  # Auto: kind of call -> whether it is chunked (decided at its first call, kept: a CUDA graph records that decision)


def _auto_room(device):
    """Auto: the bytes new allocations may take: the card (the GPU's VRAM, or the --max-vram card) less AUTO_MARGIN and the VRAM allocated
    now (weights included), and with the MMGP VRAM allocator never more than what it may still take (its room: the VRAM other programs
    leave, the --max-vram limit) less AUTO_MARGIN's bytes. Its reserved VRAM not allocated does not count: chunks of earlier phases (the
    text encoders, the VAE) that a large allocation cannot use (H3 1080p, 0.85 GiB of them: the 1.47 GiB of out_proj's kept input failed)."""
    card, allocated = torch.cuda.get_device_properties(device).total_memory, torch.cuda.memory_allocated(device)
    room = card - max(AUTO_MARGIN[0], AUTO_MARGIN[1] * card) - allocated
    if allocator.active == "vmm":
        room = min(room, allocator.room(device) - AUTO_MARGIN[0])
    return room


def chunking_needed(device, extra_bytes, kind):
    """Whether a call chunks its inputs (splits its heads, cuts its rows): always with "on", never with "off". With "auto", only where
    the whole input would not fit: extra_bytes, what the unchunked path adds, above _auto_room. Decided once per kind of call (shape,
    tokens), at its first call."""
    if chunker != "auto":
        return chunker in ("on", "extreme")
    needed = _needed.get(kind)
    if needed is None:
        needed = _needed[kind] = device.type == "cuda" and extra_bytes > _auto_room(device)
    return needed


_extreme_levels = {}  # Auto: (shapes, LoRA ranks) -> the extreme level plan_extreme chose (0: none), at the first pass of these shapes


@contextlib.contextmanager
def plan_extreme(owner, device, dtype, rows, hidden, heads, kv_heads, head_dim, projections, out_proj):
    """Auto: the blocks of a model that gives its attentions' inputs by ranges of rows (an mmgp offload.RowSource: EXTREME_LOW_VRAM) run
    in the extreme mode where the chunked attention ("on") would not fit, for the blocks run inside this context: groups of 2 key / value
    heads where they fit, else of 1 (the lowest VRAM). Decided at the first extreme_low_vram() inside it (the graph gate at the first
    graphed block call, or the model's block), once MMGP has loaded that block's weights: their format says what the attention keeps.
    The estimates follow the shapes: the VRAM allocated then and what a pass over the blocks adds for its weights (mmgp
    runtime.pass_weight_bytes: streamed blocks' slots, less the preloaded blocks the VRAM pressure releases), plus the attention's peak
    of each mode: chunked, its normalized input whole (rows x hidden, 16-bit), prepared in place, beside out_proj's kept input (INT8
    where its format keeps it so: column_input) and the q / k / v of its largest group (MOST_GROUPS groups); extreme, out_proj's kept
    input, a group's q / k / v, half a range of rows (EXTREME_RANGE_BYTES), the LoRA products of every row (q / k / v, and out_proj's
    in FP32). The chunked estimate is an upper bound for weight formats that prepare their input beside it (no prepare_in_place:
    Triton, Comfy Kitchen; H3 1080p: 7.9 GiB above the blocks' start with Triton, 5.1 with Kitchen, 3.2 with WGPKernels).
    Against _auto_room, as chunking_needed. Decided once per shapes and LoRA ranks.
    Extreme (forced): the same estimates decide whether the extreme mode pays for these blocks: where it would save less than
    EXTREME_MIN_SAVING over the chunked attention (images, short videos: inputs of a few hundred MB), they run as Standard (CUDA graphs,
    the Peak Fit groups) at its speed.
    owner: a module of the model (its MMGP manager); projections: the q / k / v linears, out_proj the output projection (None: the
    attention's output is kept whole in 16 bits, read with another stream: Flux's single blocks)."""
    global _extreme_plan, EXTREME_LOW_VRAM, EXTREME_GROUP_HEADS, EXTREME_LORA_PRODUCTS
    if chunker not in ("auto", "extreme") or rows < MIN_SPLIT_TOKENS or device.type != "cuda" or kv_heads < 2:
        yield
        return
    saved = EXTREME_LOW_VRAM, EXTREME_GROUP_HEADS, EXTREME_LORA_PRODUCTS
    _extreme_plan = (owner, device, dtype, rows, hidden, heads, kv_heads, head_dim, projections, out_proj)
    cuda_graphs.reset_gates()  # an enclosing scope's graphed calls (a model's preparation) read the gate before this decision
    try:
        yield
    finally:
        _extreme_plan = None
        if (EXTREME_LOW_VRAM, EXTREME_GROUP_HEADS, EXTREME_LORA_PRODUCTS) != saved:
            EXTREME_LOW_VRAM, EXTREME_GROUP_HEADS, EXTREME_LORA_PRODUCTS = saved
            _last_wave_split(EXTREME_LOW_VRAM)


_extreme_plan = None  # plan_extreme's arguments until extreme_low_vram() decides


def extreme_low_vram():
    """EXTREME_LOW_VRAM, after plan_extreme's pending decision (if any) is taken."""
    global _extreme_plan, EXTREME_LOW_VRAM, EXTREME_GROUP_HEADS, EXTREME_LORA_PRODUCTS
    if _extreme_plan is None:
        return EXTREME_LOW_VRAM
    owner, device, dtype, rows, hidden, heads, kv_heads, head_dim, projections, out_proj = _extreme_plan
    _extreme_plan = None
    rank_qkv, rank_out = offload.lora_rank(projections), 0 if out_proj is None else offload.lora_rank([out_proj])
    key = (rows, hidden, heads, kv_heads, head_dim, rank_qkv, rank_out, dtype)
    level = _extreme_levels.get(key)
    if level is None:
        budget = _auto_room(device) - runtime.pass_weight_bytes(owner)
        inner = rows * heads * head_dim  # out_proj's input: kept in INT8 by its weight format where it can (column_input), else in 16 bits
        kept = inner + inner // 128 * 4 if out_proj is not None and ops.column_input(out_proj, rows, dtype, device) is not None else inner * 2
        group = lambda h: rows * h * head_dim * 2 * (heads // kv_heads + 2)
        norm = rows * hidden * 2
        if offload.prepares_in_place(projections, dtype, device):
            chunked = norm + max(0, kept - norm // 2) + group(-(-kv_heads // MOST_GROUPS))
        else:  # prepared beside its input (Triton, Comfy Kitchen): an upper bound, the input, a rotated copy and the prepared rows
            chunked = 3 * norm + kept + group(-(-kv_heads // MOST_GROUPS))
        extreme = lambda h: (kept + group(h) + EXTREME_RANGE_BYTES // 2 + rows * rank_qkv * 2 + rows * rank_out * 4
                             + (rows * head_dim * 2 if kv_heads % h else 0))
        if chunker == "extreme":  # forced: only where it saves EXTREME_MIN_SAVING (its groups: the floor's)
            floor = extreme_group_heads(kv_heads, group(1))
            level = _extreme_levels[key] = floor if chunked - extreme(floor) >= EXTREME_MIN_SAVING else 0
            if not level:
                tqdm.write(f"Smart Inputs Chunker Extreme: Standard chunking for these blocks, the extreme mode would save "
                           f"{max(0, chunked - extreme(floor)) / 2**20:.0f} MB of attention peak")
            if not level:
                EXTREME_LOW_VRAM = EXTREME_LORA_PRODUCTS = False
                _last_wave_split(False)
            return EXTREME_LOW_VRAM
        # the largest group up to the floor (at least 2 heads, as before) whose peak fits the room, else 1 head
        most = extreme_group_heads(kv_heads, group(1), max(2, kv_heads))
        level = _extreme_levels[key] = 0 if chunked <= budget else next((h for h in range(max(2, most), 1, -1) if (h == 2 or kv_heads % h == 0) and extreme(h) <= budget), 1)
        if level:
            tqdm.write(f"Smart Inputs Chunker Auto: extreme mode, groups of {level} head{'s' if level > 1 else ''} (estimated attention peak "
                       f"{extreme(level) / 2**30:.2f} GiB, chunked {chunked / 2**30:.2f} GiB, room {budget / 2**30:.2f} GiB)")
    if chunker == "extreme":
        if not level:
            EXTREME_LOW_VRAM = EXTREME_LORA_PRODUCTS = False
            _last_wave_split(False)
        return EXTREME_LOW_VRAM
    if level:
        EXTREME_LOW_VRAM = EXTREME_LORA_PRODUCTS = True
        EXTREME_GROUP_HEADS = level
        _last_wave_split(True)
    return EXTREME_LOW_VRAM


def _graph_key():  # a block recorded in one mode runs other kernels than in another: part of every CUDA graph key
    return chunker, WAVE_SPLIT, UNEQUAL_GROUPS, WAVE_POLICY, FIT_VRAM, FIT_HEADS, FIT_ROOM


@torch.compiler.disable()
def kv_first_attention(qkv_list, force_attention=None):
    """pay_attention(qkv_list, force_attention=force_attention, recycle_q=True) when nothing else holds k and v: with SageAttention 2
    (sm86/sm89/sm120) or SageAttention, k is quantized and released before v is copied for its quantization, a peak lower by half of k;
    same results."""
    q, k = qkv_list[:2]
    use_fp4 = fp4_geometry(q.device, q.dtype, q.shape[0], q.shape[1], k.shape[1], q.shape[2], k.shape[2], q.shape[3], force_attention)
    del q, k
    staged = None if use_fp4 else sage_staged(qkv_list[0].device, qkv_list[0].shape[-1], force_attention)
    if staged is None:
        return pay_attention(qkv_list, force_attention=force_attention, recycle_q=True)
    staged = kv_first_staged(staged, qkv_list, force_attention)
    core, settings = staged
    query, key_list, value_list = qkv_list[0], qkv_list[1:2], qkv_list[2:3]
    qkv_list.clear()
    quantized = [None, core.staged_quantize_k(key_list, settings), core.staged_quantize_v(value_list, settings)]
    quantized[0] = core.staged_quantize_q(query, settings)
    return core.staged_attention(query, quantized, settings)


WAVE_SPLIT = True  # comparison switch: False keeps the level's groups whatever the launches cost
WAVE_TOLERANCE = 0.045  # of a launch's time, lost to a last wave of attention programs that leaves the device part empty


def head_groups(kv_heads, tokens, level=None, waves=None):
    """Groups of heads of an attention over tokens (of all batch items): the divisor of kv_heads closest to the level's target, 1 when
    off. waves: (programs of the attention kernel for all heads together, the kernel's shared.kernels.launch_waves.LaunchWaves on this
    device), where the attention core has them (its launch_waves, calibrated once per GPU): the most groups, up to that divisor,
    whose launches fill the device as well as all heads together would, within WAVE_TOLERANCE. A launch over few heads and few
    tokens ends on a part-empty device, and launches do not overlap: on an RTX 5090, up to 30 % of the attention's time for a few
    tens of MB."""
    level = head_split if level is None else level
    if level == 0 or tokens < MIN_SPLIT_TOKENS:
        return 1
    divisors = [groups for groups in range(1, kv_heads + 1) if kv_heads % groups == 0]
    target = min(divisors, key=lambda groups: (abs(groups - _TARGET_GROUPS[level]), groups))
    if waves is None or not WAVE_SPLIT:
        return target
    programs, model = waves
    limit = model.cost(programs) + WAVE_TOLERANCE
    return max(groups for groups in divisors if groups <= target and programs % groups == 0 and (groups == 1 or model.cost(programs // groups) <= limit))


UNEQUAL_GROUPS = True  # comparison switch: False keeps groups of equal sizes (head_groups)
WAVE_POLICY = "fit"  # "fit" (Peak Fit): the fastest groups within FIT_VRAM of the finest split; comparison switches: "vram" (the level sets the largest group), "speed" (the finest split within WAVE_TOLERANCE)
FIT_VRAM = (100 * 1024 * 1024, 150 * 1024 * 1024)  # what the largest group may add to the finest split's peak: at least, at most
STATS_WIDTH = 2.5  # heads of a 16-bit projection for the statistics of norms over all heads, per head of the largest group (5 bytes per element there)
FIT_HEADS = 1.5  # between them: this many heads' worth of q, k and v, whatever the model's head width, batch and precision
FIT_ROOM = True  # comparison switch: Peak Fit also takes the measured free room (_Room), with the MMGP VRAM allocator
ADD_AFTER_ROWS = 16384  # rows of out_proj's output per part with add_after: the INT8 kept input's GEMM parts (int8_backend.COLUMN_ROWS)
CHUNK_STATISTICS = True  # comparison switch: False takes the statistics of norms over all heads from norm_stats or by groups of heads when no q / k is kept
ROW_SOURCES = True  # comparison switch: False makes an input given by ranges of rows (mmgp offload.RowSource) whole before it is prepared
# Extreme low VRAM: every attention that can be split runs one head per group, whatever the chunker decides: the least VRAM of the
# attention's own tensors, for more and smaller launches (H3 1080p 15 s: about 8 % slower blocks). Models forcing a level keep it.
EXTREME_LOW_VRAM = False
EXTREME_GROUP_HEADS = 1  # key / value heads per group with it
# With it, an input given by ranges of rows (an mmgp offload.RowSource, a norm of the residual) is never prepared whole: each head group
# makes it again range by range, each range prepared, projected through the group's q / k / v columns and dropped. Bytes of 16-bit
# rows per range (H3 1080p: 128 MiB, ~12.5k rows, -80 MB at the peak against 256 MiB at the same speed; the ranges are turns of a graph loop).
EXTREME_RANGE_BYTES = 128 << 20
# The group floor: a group of fewer bytes of q, k and v than this takes more heads (a divisor of the key / value heads). Each group makes
# its input again and launches its own small kernels: below it (short calls: LTX-2's half-resolution stage, images) these costs
# dominate; at the sizes that need the extreme mode a head is above it. A generation whose largest call is short pays at most this.
# Forced Extreme, 2 blocks, RTX 5090 (ms per call for 1 / 2 / 4 / 8 heads): LTX-2.5 half-resolution 160 / 93 / 82 / 79, Wan 480p 81
# frames 270 / 231 / 216 / 207, Qwen Image 2.1 2048x2048 94 / 77 / 72 / 67 (+20 / +44 / +108 MiB), Flux 2 2048x2048 560 / 290 /
# 224 / 211: most of the gain by ~40 MB per group.
EXTREME_GROUP_BYTES = 48 << 20
# Forced Extreme runs the blocks of a plan_extreme as Standard where its estimated saving over the chunked attention is below this:
# 2048x2048 images (Qwen Image 2.1, Flux 2: estimated 35 MB, measured 0.1-0.2 GiB, mostly Standard's CUDA graph pool, for 1.5-3x the
# time) and LTX-2's half-resolution stage (0); long videos keep it (LTX-2.5 1080p 241 frames: estimated 229 MB, measured -0.63 GiB
# with its FFN input by rows, +29 %). The estimate counts the attention alone.
EXTREME_MIN_SAVING = 128 << 20


def extreme_group_heads(kv_heads, head_bytes, most=None):
    """Key / value heads per group of the extreme mode: the largest divisor of kv_heads (at most most) whose group of head_bytes per
    head stays within EXTREME_GROUP_BYTES, EXTREME_GROUP_HEADS at least."""
    limit = max(EXTREME_GROUP_HEADS, EXTREME_GROUP_BYTES // max(1, head_bytes))
    limit = limit if most is None else min(limit, most)
    return max([size for size in range(1, kv_heads + 1) if kv_heads % size == 0 and size <= limit] + [min(EXTREME_GROUP_HEADS, kv_heads)])
# With it, the LoRA products x @ A of every row (rows x rank per factor: about 170 MB for q / k / v at rank 128 and 220k tokens) made once
# before the groups (mmgp offload.row_lora_products), while out_proj's kept input is not allocated yet: the groups then read them instead of
# making the 16-bit rows of each range and x @ A again.
EXTREME_LORA_PRODUCTS = False
ROOM_PAUSE_S = 10.0  # this long without an attention call: a new generation, whose peaks the earlier ones say nothing about
MOST_GROUPS = 16  # every group adds its kernels to a block (17 to 20 on H3) and to its CUDA graph
WAVE_GAIN = 0.005  # of the attention's time: what one more group has to save
FIT_TIE = 0.005  # of the attention's time: what larger groups paid with Peak Fit's headroom must save over the fastest split within the measured room
GROUP_OVERHEAD_MS = 0.07  # a group's own cost until the projection probe has run (_group_ms; RTX 5090, CUDA graphs: H3 at 13k tokens, 12 groups, +40 ms a step)
WAVE_ALLOWANCE_GAIN = 0.01  # of the attention's time: what a largest group above the level's equal groups has to save
WAVE_VRAM = 0.10  # of the attention's working set with the level's equal groups: what a faster partition's largest group may add
_partitions = {}


class _Room:
    """The free room of Peak Fit, measured (MMGP VRAM allocator only). The allocator's probe phase is reset at the end of every attention
    call and read at the start of the next: its peak is that of everything between two attentions (an FFN, a block of another kind, a
    final layer), and the highest of a generation is its ceiling, a peak the generation has reached already. Each call also records,
    per shape, its peaks above what was allocated at its start before its groups (inputs beside their prepared form, statistics) and
    during them, with its largest group. A later call of that shape may then grow its largest group while the groups' peak, plus
    head_bytes per head more (the kit's count of a head, never below what one costs), stays under the ceiling or the call's own peak
    before its groups: the generation's peak cannot rise. Blocks of different kinds share the ceiling. The text encoders and the VAE run before
    the first attention call of a generation or after a pause (ROOM_PAUSE_S), which starts a new ceiling: their peaks never count.
    A kind's ceiling (tokens, heads, head_dim: the blocks of one transformer, the steps of its denoising) takes the intervals between two
    of its calls, and the peaks of the calls of other kinds between them (LTX-2 interleaves video, audio and cross attentions; its cross
    attention's output projection may hold the block's peak): an encoder's calls before the transformer's first (its weights unloaded
    between them) set none of it. A kind never takes its own calls' peaks: its growth cannot justify itself.
    A CUDA graph capture repeats the groups of its warmup, the eager call before it, and measures nothing; a recorded block uses no
    room (its graph pool keeps memory the eager allocator reuses): Peak Fit's headroom alone.
    Every figure leaves the models' weights out (the allocator's probe counts the allocations MMGP's loads make as weights apart): the
    blocks loaded, prefetched or released between two calls never read as room."""
    ceiling, last_end, own, chosen, pending = {}, None, {}, {}, {}  # per kind of call: its ceiling, the peak since its last call


def _room_begin(device, kind):
    """The call's measurement: (device, memory allocated but the weights at its start, [shape, sizes] once chosen, capturing, kind), None without the allocator."""
    if not FIT_ROOM or device.type != "cuda" or allocator.active != "vmm":
        return None
    capturing = cuda_graphs.capturing()
    if not capturing:
        now, peak = time.perf_counter(), allocator.probe_peak(device)
        if _Room.last_end is None or now - _Room.last_end > ROOM_PAUSE_S:
            _Room.ceiling, _Room.pending = {}, {}
        for seen in _Room.pending:  # the interval lies after the last call of every kind seen: calls of other kinds may come between two of one kind
            _Room.pending[seen] = max(_Room.pending[seen], peak)
        if kind in _Room.pending:
            _Room.ceiling[kind] = max(_Room.ceiling.get(kind, 0), _Room.pending[kind])
        _Room.pending[kind] = 0
    allocator.probe_reset(device)
    return device, allocator.probe_now(device), [], capturing, kind


def _room_slack(room, shape, equal, head_bytes):
    """head_partition's slack and last_slack, the free room in bytes beyond the finest split: what lets the groups before the last, and
    the last, reach the highest peak reached (the
    ceiling, and the shape's first call: made with Peak Fit's headroom alone, before its groups and during each group), from the first
    call's sizes and peaks at the shape's cost of a head (_room_end). (0, 0) when nothing was measured: Peak Fit's headroom alone."""
    if room is None or shape not in _Room.own or not head_bytes or cuda_graphs.recording():  # a graph's pool keeps what the eager allocator
        return 0, 0  # reuses: H3 1080p 15 s, the room's 10x5 + 6 took 96 MiB more pool than 2 + 4 + 10x5 at the same eager peak
    sizes, before, peaks, cost = _Room.own[shape][:4]
    reached = max(_Room.ceiling.get(room[4], 0) - room[1], before, *peaks)
    rest = max(range(len(sizes) - 1), key=lambda index: peaks[index], default=None)  # the group before the last with the highest peak
    caps = (equal if rest is None else sizes[rest] + int((reached - peaks[rest]) // cost), sizes[-1] + int((reached - peaks[-1]) // cost))
    return tuple(max(0, (cap - equal) * head_bytes) for cap in caps)


def _head_cost(sizes, peaks, kept_head, head_bytes):
    """What one more head in a group adds to its peak, from the groups of one call before the last (the last runs once the prepared
    input is released) with different sizes, the outputs kept by the groups between them deducted; head_bytes, the kit's count,
    at least."""
    cost, before = head_bytes, [sum(sizes[:index]) for index in range(len(sizes))]
    for i in range(len(sizes) - 1):
        for j in range(i + 1, len(sizes) - 1):
            if sizes[i] != sizes[j]:
                cost = max(cost, (peaks[j] - peaks[i] - (before[j] - before[i]) * kept_head) / (sizes[j] - sizes[i]))
    return cost


def _room_end(room, failed=False):
    device, start, chosen, capturing, kind = room
    if not failed and not capturing:  # the call's own peak is one the calls of every other kind see between two of theirs
        peak = start + max([allocator.probe_peak(device) - start, *chosen[4:]])
        for seen in _Room.pending:
            if seen != kind:
                _Room.pending[seen] = max(_Room.pending[seen], peak)
    if len(chosen) >= 7 and not failed and not capturing:  # an interrupted call's peak would be short of what the shape takes
        # peaks above the call's start: while its inputs were prepared (and what they then hold), before its groups, during each group
        shape, sizes, head_bytes, kept_head, prepared, level, before, *peaks = chosen
        peaks.append(allocator.probe_peak(device) - start)
        if shape not in _Room.own:  # the reference: its sizes and peaks, and the cost of a head between its groups
            _Room.own[shape] = [sizes, before, peaks, _head_cost(sizes, peaks, kept_head, head_bytes), prepared, level]
        else:  # the cost of a head only rises with what grown groups measure against the reference
            entry = _Room.own[shape]
            first, _, first_peaks, cost = entry[:4]
            for index_first, index in ((-1, -1), (max(range(len(first) - 1), key=lambda i: first_peaks[i], default=None), max(range(len(sizes) - 1), key=lambda i: peaks[i], default=None))):
                if index_first is not None and index is not None and sizes[index] > first[index_first]:
                    cost = max(cost, (peaks[index] - first_peaks[index_first]) / (sizes[index] - first[index_first]))
            entry[3] = cost
        _Room.chosen[shape] = sizes
    allocator.probe_reset(device)
    _Room.last_end = time.perf_counter()


def _wave_partition(heads, most, per_head, model, largest=None, equal=None, progressive=False, group_cost=0.0, last=None):
    """Sizes of at most `most` groups of `heads` heads (per_head attention programs each), the largest last. With `largest`: the
    fastest partition into groups of at most `equal` heads (a group more only when it saves WAVE_GAIN), or of at most `largest`
    heads when that saves WAVE_ALLOWANCE_GAIN more. Without: the smallest largest group whose
    partition costs no more than one launch of all heads plus WAVE_TOLERANCE, in at most one group more than that size needs
    (the largest group sets the peak: more, smaller groups beside it would only add kernels), the fewest that fit.
    progressive with `last`: the last group may have up to `last` heads, the others up to `largest` (the last one runs once the
    prepared input is released); the groups before it in ascending order."""
    times = [0.0] + [(1 + model.cost(size * per_head)) * size for size in range(1, heads + 1)]  # of a launch of `size` heads, in heads on full waves
    limit = times[heads] + WAVE_TOLERANCE * heads + 1e-9

    def search(largest, limit=None, most=most, gain=WAVE_GAIN):
        """The fastest partition into groups of at most `largest` heads, or with a limit the one of the fewest groups within it (None: none)."""
        best, cover, chosen = None, [0.0] + [float("inf")] * heads, []
        for count in range(1, most + 1):  # cover[n]: the least time of n heads in `count` groups of at most `largest` heads
            longer, sizes = [float("inf")] * (heads + 1), [0] * (heads + 1)
            for n in range(count, min(heads, count * largest) + 1):
                for size in range(1, min(largest, n - count + 1) + 1):
                    time = cover[n - size] + times[size] + group_cost  # group_cost: each group's own kernels, in heads
                    if time < longer[n]:
                        longer[n], sizes[n] = time, size
            cover = longer
            chosen.append(sizes)
            if cover[heads] <= (limit if limit is not None else float("inf") if best is None else best[0] - gain * heads):
                partition, n = [], heads
                for sizes in reversed(chosen):
                    partition.append(sizes[n])
                    n -= sizes[n]
                best = cover[heads], tuple(sorted(partition))
                if limit is not None:
                    break
        return best

    if progressive and last is not None and last > largest:
        return _last_partition(heads, most, times, group_cost, equal, largest, last)
    if progressive:  # every partition into groups of at most `largest` heads: the fastest; on a tie the smallest largest group, then the fewest groups
        found = [search(size, gain=1e-12) for size in range(equal, largest + 1)]
        return min(found, key=lambda partition: (round(partition[0], 9), partition[1][-1], len(partition[1])))[1]
    if largest is not None:
        within, beyond = search(equal), search(largest)
        return (beyond if beyond[0] < within[0] - WAVE_ALLOWANCE_GAIN * heads else within)[1]
    for largest in range(-(-heads // most), heads + 1):  # one group of all heads always qualifies
        partition = search(largest, limit, min(most, -(-heads // largest) + 1))
        if partition is not None:
            return partition[1]


def _last_partition(heads, most, times, group_cost, equal, largest, last):
    """The fastest partition whose groups before the last have at most `largest` heads (from `equal` up), the last at most `last`; on a
    tie the smallest groups before the last, the smallest last group, then the fewest groups. The groups before the last ascending."""
    best = None
    for cap in range(equal, largest + 1):
        cover, chosen, rows = [0.0] + [float("inf")] * heads, [], []
        for count in range(1, most):  # rows[count - 1][n]: the least time of n heads in `count` groups of at most `cap` heads
            longer, sizes = [float("inf")] * (heads + 1), [0] * (heads + 1)
            for n in range(count, min(heads, count * cap) + 1):
                for size in range(1, min(cap, n - count + 1) + 1):
                    time = cover[n - size] + times[size] + group_cost
                    if time < longer[n]:
                        longer[n], sizes[n] = time, size
            cover = longer
            chosen.append(sizes)
            rows.append(cover)
        for tail in range(1, min(heads, last) + 1):
            rest = heads - tail
            counts = [0] if rest == 0 else [count for count in range(1, most) if rows[count - 1][rest] < float("inf")]
            for count in counts:
                time = (rows[count - 1][rest] if count else 0.0) + times[tail] + group_cost
                partition, n = [], rest
                for sizes in reversed(chosen[:count]):
                    partition.append(sizes[n])
                    n -= sizes[n]
                partition = tuple(sorted(partition)) + (tail,)
                rank = (round(time, 9), max(partition[:-1], default=0), tail, len(partition))
                if best is None or rank < best[0]:
                    best = rank, partition
    return list(best[1])


def fit_headroom(head_bytes):
    """What Peak Fit lets the largest group add to the peak of the finest split: FIT_HEADS heads' worth (head_bytes each), within FIT_VRAM."""
    return int(min(FIT_VRAM[1], max(FIT_VRAM[0], FIT_HEADS * head_bytes)))


def head_partition(kv_heads, tokens, level=None, waves=None, fixed_heads=0.0, head_bytes=0, slack=0, keys=None, last_slack=0, group_ms=None):
    """Sizes, in key / value heads, of the groups of heads of an attention over tokens (of all batch items), in the order they are
    attended. Equal groups (head_groups) unless the attention core gives its kernel's launch waves (see head_groups): equal groups
    can only be as many as a divisor of the heads, and at many token counts all of those end their launches on part-empty waves.
    Peak Fit (WAVE_POLICY "fit", any level but 0: the chunker on, or Auto where it chunks): starting from the finest equal groups (56 heads: 14 groups of 4), the largest group
    may grow by what fit_headroom(head_bytes) buys (head_bytes: what one more head in a group holds, its q, k and v) or by `slack`,
    the measured free room where larger groups cost nothing (_room_slack; last_slack for the last group), whichever is more. A split using the headroom
    must save FIT_TIE over the fastest split within the free room, else that one, the lowest VRAM. Of all the partitions into at most MOST_GROUPS groups of at most that size, the fastest; on a
    tie the smallest largest group. A head weighs little over few tokens, where small launches fill the device badly: the groups
    are then large and fast for a few tens of MB; it weighs much over many tokens, where the finest groups cost little time.
    WAVE_POLICY "vram": the level sets the largest group (its equal groups' size; one step above, within WAVE_VRAM of the
    attention's working set, fixed_heads being what it holds whatever the groups, only for WAVE_ALLOWANCE_GAIN). "speed": the finest
    split that costs no more than one launch of all heads plus WAVE_TOLERANCE.
    The largest group comes last, once the prepared input is released. group_ms: what one group adds whatever its size, its
    projections reading their input again and its small kernels (_group_ms, probed per GPU and weight format), GROUP_OVERHEAD_MS
    where nothing is probed yet."""
    target = head_groups(kv_heads, tokens, level)
    if target == 1:
        return [kv_heads]
    if waves is None or not WAVE_SPLIT:
        return [kv_heads // target] * target
    if not UNEQUAL_GROUPS:
        groups = head_groups(kv_heads, tokens, level, waves)
        return [kv_heads // groups] * groups
    programs, model = waves
    equal, most = kv_heads // target, max(target, min(kv_heads, MOST_GROUPS))
    last = free = None
    if WAVE_POLICY == "fit":
        headroom = fit_headroom(head_bytes) if head_bytes else 0
        # the headroom counts from the finest split, the free room from the peaks reached: the larger of both, never their sum (a
        # reference call already grew by the headroom)
        caps = [min(kv_heads, equal + (room // head_bytes if head_bytes else 0)) for room in (max(headroom, slack), max(headroom, last_slack), slack, last_slack)]
        largest, last, free = caps[0], caps[1], caps[2:]
    else:
        largest = None if WAVE_POLICY == "speed" else int((equal + fixed_heads) * (1 + WAVE_VRAM) - fixed_heads + 1e-9)
    group_ms = GROUP_OVERHEAD_MS if group_ms is None else group_ms
    key = (kv_heads, programs, id(model), largest, last, None if free is None else tuple(free), WAVE_POLICY, keys, round(group_ms, 4))
    if key not in _partitions:
        per_head = programs // kv_heads
        # each group's own cost, in heads of attention: group_ms against a head's time (its programs in waves of the probe's)
        head_ms = per_head / model.slots * model.wave_ms * (keys or tokens) / model.keys if getattr(model, "wave_ms", None) and getattr(model, "keys", None) else 0
        group_cost = group_ms / head_ms if head_ms else 0.0
        sizes = _wave_partition(kv_heads, most, per_head, model, largest, equal, WAVE_POLICY == "fit", group_cost, last)
        if free is not None and (free[0], free[1]) != (largest, last):  # what the headroom buys must be faster: else the split within the free room
            within = _wave_partition(kv_heads, most, per_head, model, free[0], equal, True, group_cost, free[1])
            time = lambda partition: sum((1 + model.cost(size * per_head)) * size + group_cost for size in partition)
            sizes = sizes if time(sizes) < time(within) - FIT_TIE * kv_heads else within
        _partitions[key] = sizes
    return list(_partitions[key])


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


def quantize_q4(x):
    """The LM engines' symmetric q4_0 format: two channels per byte and one FP16 scale per 32 channels."""
    if x.shape[-1] % GROUP:
        raise ValueError("Q4 KV cache requires a head dimension divisible by 32.")
    fmt = kv_formats.FORMATS["q4_0"]
    data = torch.empty((*x.shape[:-1], fmt.width(x.shape[-1])), dtype=fmt.dtype, device=x.device)
    scales = torch.empty((*x.shape[:-1], x.shape[-1] // GROUP), dtype=torch.float16, device=x.device)
    for start in range(0, x.shape[1], _Q8_ROWS):
        stop = start + _Q8_ROWS
        part, scale, _ = kv_formats.quantize_heads(x[:, start:stop], fmt)
        data[:, start:stop], scales[:, start:stop] = part, scale
        del part, scale
    return data, scales


def dequantize_q8_into(out, data, scales):
    """out (batch, tokens, heads, head_dim) 16-bit, written: q8 data (batch or 1, ...) with its scales, through strides."""
    if _kernels_eligible(out):
        return _kv_triton.dequantize_q8_into(out, data, scales)
    for start in range(0, out.shape[1], _Q8_ROWS):
        stop = start + _Q8_ROWS
        out[:, start:stop] = (data[:, start:stop].float().unflatten(-1, (-1, GROUP)) * scales[:, start:stop].float().unsqueeze(-1)).flatten(-2)
    return out


def dequantize_q4_into(out, data, scales):
    """Write q4_0 data into a 16-bit buffer through strides, rounding the FP32 product once."""
    if _kernels_eligible(out):
        return _kv_triton.dequantize_q4_into(out, data, scales)
    for start in range(0, out.shape[1], _Q8_ROWS):
        stop = start + _Q8_ROWS
        out[:, start:stop] = kv_formats.dequantize_heads(data[:, start:stop], scales[:, start:stop], None,
                                                        kv_formats.FORMATS["q4_0"])
    return out


def dequantize_prefix_into(out, data, scales):
    """Write a non-native prefix (q8 int8 or packed q4_0 uint8) into out."""
    return (dequantize_q4_into if data.dtype == torch.uint8 else dequantize_q8_into)(out, data, scales)


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


def cache_prefix(kv_list, native=True, bits=8):
    """A cached prefix (KVPrefix) for joint_qkv_attention's kv_prefix, of keys and values (batch, tokens, kv_heads, head_dim) normalized
    and rotated, handed off in kv_list ([key, value], cleared; both are consumed). native: in the quantized layout of the staged
    SageAttention core that will attend it where that core supports it (unmasked attentions without attention_fn: pass False for the
    others), attended in a pass of its own and merged by log-sum-exp, the error of attending the joined keys; otherwise q8, dequantized
    into the joined keys and values at each attention (a small error on 16-bit backends). About half of the 16-bit memory either way.
    bits=4 uses the LM engines' q4_0 format instead (18 bytes per 32 channels, versus 34 for q8); it is dequantized into the joined
    keys and values even with a staged core. A block recorded for a CUDA graph keeps the settings of an eager one, so the cached steps can be recorded."""
    if bits not in (4, 8):
        raise ValueError("KV prefix precision must be 4 or 8 bits.")
    key, value = kv_list
    kv_list.clear()
    staged = sage_staged(key.device, key.shape[-1]) if native and bits == 8 else None
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
    quantize = quantize_q4 if bits == 4 else _q8
    k, k_scale = quantize(key)
    del key
    v, v_scale = quantize(value)
    return KVPrefix(k, k_scale, None, v, v_scale, None)


def q8_prefix(k, k_scale, v, v_scale):
    """A cached prefix of keys and values already in the q8 format (shared.llm_engines.lmgp.layers.kv_formats.Q8: int8 (batch, tokens,
    kv_heads, head_dim), or without batch, with FP16 scales per 32 channels), as kept by an archive: not copied."""
    k, k_scale, v, v_scale = (t if t.dim() == 4 else t.unsqueeze(0) for t in (k, k_scale, v, v_scale))
    return KVPrefix(k, k_scale, None, v, v_scale, None)


def dequantize_prefix(prefix, dtype=torch.bfloat16):
    """(keys, values) in dtype of a non-native q8 or q4_0 cached prefix."""
    return tuple(dequantize_prefix_into(torch.empty((*data.shape[:-1], scales.shape[-1] * GROUP), dtype=dtype, device=data.device), data, scales)
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
    bits: int = 8

    def store(self, k, v, kit=False):
        """k / v (batch, tokens, kv_heads, head_dim), normalized and rotated (consumed when quantized); kit: attended through the
        unmasked attention of joint_qkv_attention (a native prefix then). With a state ring the layer goes to the ring (RAM, or VRAM of
        its own with a dynamic ring), its tensors being those the ring binds (holding them for this pass already)."""
        values = cache_prefix([k, v], native=kit and self.native, bits=self.bits) if self.quantize else (k, v)
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
    - quantize: INT8 by default (cache_prefix: about half the memory, the layout of SageAttention where it attends them).
      bits=4 selects packed Q4 with FP16 scales, about 28% of 16-bit storage, dequantized per head group when attended.
    - ram: kept in RAM and streamed to the GPU block by block with the weights (mmgp.runtime.state_ring, locked RAM as far as the reserved
      RAM allows): the VRAM of two layers. blocks: the model's blocks (a ModuleList called one by one), whose forward then fetches the
      layer's cache; without ram, blocks may be a number of layers.
    - dynamic (without ram): kept in VRAM as far as the model's VRAM allows, planned with its weights by its dynamic preload, the layers that
      do not stay there kept in RAM and streamed as with ram (a state ring with resident="dynamic"), where mmgp.runtime.dynamic_state(blocks);
      otherwise kept on the GPU.
    clear() at the end of the generation (clear(keep_ring=True) to store the layers again, e.g. when step-dependent LoRAs change them)."""

    def __init__(self, blocks, quantize=False, native=True, ram=False, name="KV cache", dynamic=False, bits=8):
        if bits not in (4, 8):
            raise ValueError("KV prefix precision must be 4 or 8 bits.")
        if ram:
            self.ring = runtime.state_ring(blocks, name)
        elif dynamic and DYNAMIC_PREFIX and not isinstance(blocks, int) and runtime.dynamic_state(blocks):
            self.ring = runtime.state_ring(blocks, name, resident="dynamic")
        else:
            self.ring = None
        self.layers = [PrefixLayer(index=index, quantize=quantize, native=native, bits=bits) for index in range(blocks if isinstance(blocks, int) else len(blocks))]
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


def compact_prompt_padding(tokens, mask):
    """Pack real conditioning rows in their original order, once before denoising.
    The caller has already applied positional encoding, or supplies positions for the packed rows.
    Return a key mask only when a rectangular batch still needs padding."""
    if mask is None:
        return tokens, None
    lengths = mask.sum(1).tolist()
    width = max(lengths)
    if width == 0:
        return tokens, mask
    if min(lengths) == mask.size(1):
        return tokens, None
    indices = mask.to(torch.int32).argsort(dim=1, descending=True, stable=True)[:, :width]
    batch = torch.arange(tokens.size(0), device=tokens.device)[:, None]
    return tokens[batch, indices], None if min(lengths) == width else mask[batch, indices]


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
def _token_qk(projections, linear_inputs, lengths, batch, heads, kv_heads, head_dim, dtype, device, norm_rope, staged, cross, kept, statistics, any_heads=False, *, split_dense=None):
    """The chunk pass of norm_rows: k then q of all heads projected by chunks of tokens of each batch item (TOKEN_QK_BYTES, a multiple of
    128 rows: the quantizers' blocks). Returns, for each of the `kept` groups ((q heads, kv heads) ranges), its q and k normalized and
    rotated by norm_rope (group.rows), quantized for the staged core, chunks joined: [((q_int8, q_scale), (k_int8, k_scale)), ...] (k's
    smoothing mean is the group's first chunk's, per batch item); with `statistics`, the per token mean squares of q and of k over all
    heads, (batch, tokens, 1) FP32, for the groups that project theirs again (HeadGroup.mean_squares), else None. any_heads (norm_rows
    "heads"): with the statistics, only the kept heads are normalized (by them, as the groups projected again are), rotated and
    quantized; otherwise all heads, by the model's own norm. For a split dense core the same retention budget holds BF16 K alone.
    Pack it in its final basis before the Q statistics pass; the caller then projects and packs Q one head group at a time."""
    core, settings = staged
    keep_key_only = split_dense is not None
    result, squares_of = {}, {}
    for which, total in ((1, kv_heads), (0, heads)):
        index = 1 if cross and which else 0
        module, prepared, length = projections[index][which], linear_inputs[index], lengths[index]
        ranges = [] if keep_key_only and which == 0 else [group[which] for group in kept]
        chunk = max(128, TOKEN_QK_BYTES // (total * head_dim * dtype.itemsize) // 128 * 128)
        wholes, scales = [None] * len(ranges), [[] for _ in ranges]
        squares = torch.empty((batch, length, 1), dtype=torch.float32, device=device) if statistics else None
        for item in range(batch):
            mean, item_scales = None, [[] for _ in ranges]
            for start in range(0, length, chunk):
                stop = min(length, start + chunk)
                rows = offload.linear_input_rows(prepared, item * length + start, item * length + stop)
                part = offload.linear_rows(module, rows, 0, total * head_dim).view(1, stop - start, total, head_dim)
                del rows
                if statistics:  # as the statistics pass computes them: one FP32 reduction of the 16-bit projection
                    norms = squares[item, start:stop, 0]
                    torch.linalg.vector_norm(part[0].flatten(1), dim=-1, dtype=torch.float32, out=norms)
                    ops.square_div_(norms, total * head_dim)
                    del norms
                if ranges and statistics and any_heads:  # the kept heads only (the first groups): normalized by the statistics
                    stop_head = ranges[-1].stop
                    part = part[:, :, :stop_head].contiguous()
                    sub_squares = squares[item:item + 1, start:stop]
                    group = HeadGroup(range(stop_head * heads // total) if which else range(stop_head), range(stop_head) if which else range(stop_head * kv_heads // total),
                                      (sub_squares, sub_squares), range(start, stop))
                    norm_rope(part, None, group) if which == 0 else norm_rope(None, part, group)
                elif ranges:  # apply the model's norm to all heads before taking the retained groups
                    group = HeadGroup(range(heads), range(kv_heads), None, range(start, stop))
                    norm_rope(part, None, group) if which == 0 else norm_rope(None, part, group)
                if ranges and keep_key_only:
                    for number, head_range in enumerate(ranges):
                        if wholes[number] is None:
                            wholes[number] = part.new_empty((batch, length, len(head_range), head_dim))
                        wholes[number][item, start:stop].copy_(part[0, :, head_range.start:head_range.stop])
                elif ranges:
                    if which == 0:
                        quantized, scale = core.staged_quantize_q(part, settings)
                    else:
                        mean = core.prefix_mean(part) if mean is None else mean
                        quantized, scale = core.staged_quantize_k([part], {**settings, "k_mean": mean})
                    for number, head_range in enumerate(ranges):
                        if wholes[number] is None:
                            wholes[number] = quantized.new_empty((batch, length, len(head_range), quantized.shape[-1]))
                        wholes[number][item, start:stop].copy_(quantized[0, :, head_range.start:head_range.stop])
                        item_scales[number].append(scale[:, head_range.start:head_range.stop])
                    del quantized, scale
                del part
            if not keep_key_only:
                for group_scales, chunk_scales in zip(scales, item_scales):
                    group_scales.append(torch.cat(chunk_scales, dim=-1))
            del item_scales
        if not keep_key_only:
            result[which] = [(whole, torch.cat(group_scales)) for whole, group_scales in zip(wholes, scales)]
        elif which == 1:
            result[1] = split_dense[0].pack_split_keys(wholes, split_dense[1])
        squares_of[which] = squares
        del wholes, scales, squares
    packed_groups = result[1] if keep_key_only else list(zip(result[0], result[1]))
    return packed_groups, (squares_of[0], squares_of[1]) if statistics else None


class LoopRows:
    """norm_rope's group.rows for a CUDA graph loop's turn (mmgp.cuda_graphs.loop): the tokens start + counter * step to stop + counter *
    step, the counter on the device (start, stop: the first turn's). Given only to a norm_rope with a true `loop_rows` attribute."""
    __slots__ = ("start", "stop", "counter", "step")

    def __init__(self, start, stop, counter, step):
        self.start, self.stop, self.counter, self.step = start, stop, counter, step

    def __len__(self):
        return self.stop - self.start


QUERY_CHUNK_LOOP_COST = 0.001  # share of a turn's chunks' time its pause may cost (cuda_graphs.loop_turn_size)


def _query_chunk_loop(q_proj, out_proj, linear_input, norm_rope, core, settings, item_kv, heads, head_dim, length, chunk, head_gate, residual, gate,
                      segments, all_heads):
    """While a block is captured, the query chunks of a cross attention (batch 1) as one CUDA graph loop: a turn projects its chunks' q
    from the prepared input's rows the loop's counter gives (int8_backend.prepared_rows_linear_at), normalizes and rotates them (norm_rope
    with LoopRows), attends the context's quantized k / v, gates them (head_gate's rows) and adds their gated projection into the
    residual's rows (segment_gated_linear_at): the kernels of the eager chunks, at the counter's rows. Chunks of whole frames, or frames of
    whole chunks (each turn then takes its eager chunk's INT8 path). Returns the first token it did not do (0: no loop)."""
    from shared.kernels import int8_backend, row_loop_triton
    full = length // chunk
    if (full < 3 or not getattr(norm_rope, "loop_rows", False) or residual is None or segments is None or not cuda_graphs.loop_available(residual.device)
            or not cuda_graphs.loop_allocations()):
        return 0
    span = segments[0][1] - segments[0][0]
    if any(segment != (index * span, (index + 1) * span, index) for index, segment in enumerate(segments)) or chunk % span and span % chunk:
        return 0
    provider = linear_input.providers.get(q_proj)
    sample = torch.empty((chunk, heads * head_dim), dtype=residual.dtype, device=residual.device)  # shapes for the kernels' choices (no kernel runs)
    if (provider is not int8_backend._WGP_ROW_PROJECTION or any(key[0] is q_proj for key in linear_input.lora)
            or not int8_backend.segment_gated_linear_supported(out_proj, sample) or head_gate is not None and not head_gate.is_contiguous()):
        return 0
    per_turn = cuda_graphs.loop_turn_size(residual.device, cuda_graphs.loop_turn_ms(out_proj, ("query chunk", chunk)), full, QUERY_CHUNK_LOOP_COST)
    turns = full // per_turn
    if turns < 2:
        return 0
    prepared, step = linear_input.prepared[provider.key(q_proj)], per_turn * chunk
    path = int8_backend.segment_run_path(out_proj, sample, 0, chunk, 0, span if chunk % span == 0 else chunk)  # the eager chunks' run
    rows_of, gates = residual.view(-1, residual.shape[-1]), None if head_gate is None else head_gate.view(-1, heads)
    counter = torch.empty(1, dtype=torch.int64, device=residual.device)
    with cuda_graphs.loop(turns, counter):
        for index in range(per_turn):
            first = index * chunk
            query = int8_backend.prepared_rows_linear_at(q_proj, prepared, chunk, counter, first, step).view(1, chunk, heads, head_dim)
            norm_rope(query, None, HeadGroup(all_heads.q, all_heads.kv, None, LoopRows(first, first + chunk, counter, step)))
            core.staged_attention(query, [core.staged_quantize_q(query, settings), *item_kv], settings, None)
            if gates is not None:
                row_loop_triton.gate_heads_at(query[0], gates, counter, first, step)
            int8_backend.segment_gated_linear_at(out_proj, query.view(chunk, -1), rows_of, gate, span, path, counter, first, step)
            del query
    return turns * step


def _room_keep(room, shape, sizes, group_bytes):
    """How many of the first groups keep the chunk pass's INT8 q and k (TOKEN_QK_POLICY): with "room", the most whose kept q and k,
    each freed once its group has attended, keep every moment of the call (measured on the shape's first call: before its groups,
    during each group) under the highest peak reached (the ceiling of its kind, those moments). None measured yet, or a block being
    recorded for a CUDA graph (its pool keeps memory the eager allocator reuses): none."""
    if TOKEN_QK_POLICY in ("all", "none"):
        return len(sizes) if TOKEN_QK_POLICY == "all" else 0
    if room is None or shape not in _Room.own or cuda_graphs.recording():
        return 0
    first, before, peaks, _, prepared, level = _Room.own[shape]
    peaks = list(peaks) if tuple(first) == tuple(sizes) else [max(peaks)] * len(sizes)
    reached = max(_Room.ceiling.get(room[4], 0) - room[1], prepared, before, *peaks)
    # the chunk pass replaces the statistics pass (the reference's `before`, which projects q and then k of all heads): it holds what
    # the prepared inputs hold, a chunk of q or k of all heads in 16 bits with its quantized form, and the kept groups
    keep = 0
    for count in range(1, len(sizes) + 1):
        pending = [sum(group_bytes[index + 1:count]) for index in range(count)]  # kept, not yet attended, during each kept group
        passing = level + 2 * TOKEN_QK_BYTES + sum(group_bytes[:count])
        if max(prepared, passing) > reached or any(peaks[index] + pending[index] > reached for index in range(count)):
            break
        keep = count
    return keep


FIT_REPROJECT = 0.03  # of the attention's work: what projecting again the q and k of the groups that do not keep the chunk pass's may cost (about 1.5 % of a step)


def _in_width(module):
    return getattr(module, "in_features", None) or module.weight.shape[-1]


def _speed_keep(sizes, ratio, batch, q_tokens, k_tokens, q_width, k_width, head_dim):
    """How many of the first groups keep the chunk pass's q and k whatever the room, so that projecting the others' q and k again (a
    norm over all heads: norm_rows) costs at most FIT_REPROJECT of the attention: per key / value head, the projections of its q and k
    against its attention, 2 * (q_tokens * q_width * ratio + k_tokens * k_width) * head_dim against 4 * q_tokens * k_tokens * ratio *
    head_dim multiply-adds counted alike (INT8 GEMMs and the INT8 / FP8 attention run at similar rates). About K / T of the attention
    (T tokens, K the projections' input width): high resolutions project again every group for a few percent, those of a few tens of
    thousands of tokens keep most groups' INT8 q and k (2 bytes a token per head channel: a third of the unsplit q, k and v in 16 bits)
    instead of losing 5 to 10 % of a step. Deterministic: a CUDA graph's warmup and capture keep the same groups."""
    reproject = 2 * batch * (q_tokens * q_width * ratio + k_tokens * k_width) * head_dim
    allowed = FIT_REPROJECT * sum(sizes) * 4 * batch * q_tokens * k_tokens * ratio * head_dim / reproject  # key / value heads projected again
    keep = len(sizes)
    while keep and sum(sizes[keep - 1:]) <= allowed:
        keep -= 1
    return keep


class _ProbeProjection:
    """project(columns, first): module's output columns [first, first + columns) for `rows` rows of a prepared input (the projection
    probe, shared.kernels.projection_cost), the result dropped."""
    def __init__(self, module, linear_input, rows):
        self.module, self.input, self.rows, self.in_features = module, linear_input, rows, _in_width(module)

    def __call__(self, columns, first):
        offload.linear_rows(self.module, self.input, first, first + columns)


def _head_bytes(cross, batch, lengths, tokens, heads, kv_heads, head_dim, dtype, staged):
    """What one more key / value head in a group holds: its q (16 bits, then INT8 too), k and v (INT8) with a staged core, all three
    in 16 bits without; a cross attention's q over the query tokens, k and v over the keys' (a long video as keys of short audio queries)."""
    if not cross and staged is not None and hasattr(staged[0], "head_bytes"):
        return staged[0].head_bytes(staged[1], batch, tokens, head_dim, dtype, heads // kv_heads)
    if cross:
        return batch * head_dim * ((heads // kv_heads) * lengths[0] * (dtype.itemsize + 1 if staged is not None else dtype.itemsize)
                                   + lengths[1] * (2 if staged is not None else 2 * dtype.itemsize))
    return batch * tokens * (heads // kv_heads) * head_dim * (dtype.itemsize + 3 if staged is not None else 3 * dtype.itemsize)


def _format_identity(key):
    """A row projection's key as the name of its weight format: the per-layer values in it (scale tensors, the module itself) as their
    kind, so that every layer of a format shares one probed cost."""
    named = lambda value: "tensor" if torch.is_tensor(value) else type(value).__name__ if isinstance(value, torch.nn.Module) else value
    return tuple(map(named, key)) if isinstance(key, tuple) else named(key)


def _group_ms(projections, linear_inputs, lengths, batch, head_dim, device, shared_kv):
    """What one group of heads adds whatever its size (head_partition's group_ms), in ms on this GPU: each stream's projections of the
    group (q, k, v and an extra one: one GEMM each, every one reading the stream's prepared input again) at the launch cost probed for
    its weight format and input width (shared.kernels.projection_cost), and the group's small kernels. None where a stream is not
    prepared for row projections or nothing is probed yet (GROUP_OVERHEAD_MS then)."""
    from shared.kernels import projection_cost
    total, cost = 0.0, None
    for modules, linear_input, length in zip(projections, linear_inputs, lengths):
        present = [module for module in dict.fromkeys(modules) if module is not None] if shared_kv else [module for module in modules if module is not None]
        if linear_input is None or not present:
            return None
        module, rows = present[0], batch * length
        identity = f"{_format_identity(linear_input.providers[module].key(module))} K{_in_width(module)}"

        def rows_of(count, module=module, linear_input=linear_input, rows=rows):
            count = min(count, rows)
            if module.out_features < projection_cost.NARROW * head_dim:
                return None
            part = linear_input if count == rows else offload.linear_input_rows(linear_input, 0, count)
            return None if part is None else _ProbeProjection(module, part, count)
        cost = projection_cost.get(identity, device, rows_of, head_dim)
        if cost is None:
            return None
        total += len(present) * cost.launch_ms(rows * _in_width(module))
    return total + projection_cost.KERNELS_PER_GROUP * cost.kernel_ms


BY_ITEMS = True  # comparison switch: False attends a batch whole in the extreme mode (by_items ignored)


def _item_rows(x, item):
    """Batch item `item` of x (batch, tokens, features), a tensor or an mmgp RowSource (its rows read from the whole one's)."""
    if not isinstance(x, offload.RowSource):
        return x[item:item + 1]
    tokens = x.shape[1]
    first = item * tokens
    fill_prepared = None if x.fill_prepared is None else lambda start, stop, form, want_rows: x.fill_prepared(first + start, first + stop, form, want_rows)
    return offload.RowSource((1, *x.shape[1:]), x.dtype, x.device, lambda start, stop: x.fill(first + start, first + stop), fill_prepared=fill_prepared)


def _by_items(streams, args, kwargs, by_items):
    """joint_qkv_attention one batch item after another (EXTREME_LOW_VRAM, out_proj with a residual: what each item keeps of its
    attention output for out_proj, its LoRA products and its head groups are one item's, the inputs made again by ranges as before).
    by_items: True, or item -> keyword arguments of that item (norm_rope, fused_quant reading the item's rows of batch-wide tables)."""
    inputs = []
    for x_list, projections in streams:
        inputs.append((x_list[0], projections))
        x_list.clear()
    streams = None
    batch = inputs[0][0].shape[0]
    residual, segments = kwargs["residual"], kwargs.get("segments")
    out_stream = kwargs.get("out_stream")
    lengths = [x.shape[1] for x, _ in inputs]
    out_tokens = sum(lengths) if out_stream is None else lengths[out_stream]
    rows = residual.view(-1, residual.shape[-1])
    rests = []
    for item in range(batch):
        options = dict(kwargs, **({} if by_items is True else by_items(item)))
        first, last = item * out_tokens, (item + 1) * out_tokens
        options["residual"] = rows[first:last]
        if segments is not None:
            options["segments"] = [(max(start, first) - first, min(stop, last) - first, row) for start, stop, row in segments if stop > first and start < last]
        for name in ("attention_mask", "head_gate"):
            value = kwargs.get(name)
            if torch.is_tensor(value) and (name == "head_gate" or value.ndim >= 3) and value.shape[0] > 1:
                options[name] = value[item:item + 1]
        prefix = kwargs.get("kv_prefix")
        if prefix is not None and not isinstance(prefix, KVPrefix) and prefix[0].shape[0] > 1:
            options["kv_prefix"] = tuple(tensor[item:item + 1] for tensor in prefix)
        item_args = list(args)
        if "norm_rope" in options and len(item_args) > 2:
            item_args[2] = options.pop("norm_rope")
        result = joint_qkv_attention([([_item_rows(x, item)], projections) for x, projections in inputs], *item_args, **options)
        if out_stream is not None:
            rests.append(result)
        result = None
    inputs.clear()
    return torch.cat(rests) if out_stream is not None else residual


def joint_qkv_attention(streams, *args, by_items=None, **kwargs):
    """See _joint_qkv_attention; also measures Peak Fit's free room around the call (_Room). by_items (models whose callbacks can
    take one batch item: True, or item -> its keyword arguments): with EXTREME_LOW_VRAM, a batch of several items, an input given by
    ranges of rows and out_proj with a residual, the items attended one after another (_by_items)."""
    if (by_items is not None and BY_ITEMS and EXTREME_LOW_VRAM and kwargs.get("out_proj") is not None and kwargs.get("residual") is not None
            and streams[0][0][0].dim() > 2 and streams[0][0][0].shape[0] > 1 and any(isinstance(x_list[0], offload.RowSource) for x_list, _ in streams)
            and not isinstance(kwargs.get("kv_prefix"), KVPrefix) and kwargs.get("kv_window") is None and not callable(kwargs.get("attention_mask"))):
        return _by_items(streams, args, kwargs, by_items)
    # no reference to the inputs here: the call hands them off
    room = _room_begin(streams[0][0][0].device, (sum(x_list[0].shape[-2] for x_list, _ in streams), args[0] if args else kwargs["heads"], args[1] if len(args) > 1 else kwargs["head_dim"]))
    try:
        result = _joint_qkv_attention(streams, *args, _room=room, **kwargs)
    except BaseException:
        if room is not None:
            _room_end(room, failed=True)
        raise
    if room is not None:
        _room_end(room)
    return result


def _joint_qkv_attention(streams, heads, head_dim, norm_rope, kv_heads=None, norm_spans_heads=False, split_heads=True, kv_prefix=None,
                        attention_mask=None, force_attention=None, attention_fn=None, norm_stats=None, _cross=False, norm_v=None, softmax_scale=None,
                        head_split_level=None, fused_quant=None, kv_window=None, out_proj=None, residual=None, gate=None, segments=None, norm_rows=False, head_gate=None,
                        add_after=False, _room=None, sample_query=None, out_stream=None, sol3_prefix_tokens=0, sol3_suffix_tokens=0, sol_tau=None):
    """Attention of the q/k/v projections of several streams joined along the tokens in their order: streams is a list of
    (x_list, (q_proj, k_proj, v_proj)), each x handed off in its list, of shape (batch, tokens, features) or (tokens, features), or
    an mmgp offload.RowSource of shape (batch, tokens, features): an input only its projections read, prepared by ranges of tokens
    where its weight format allows it, so that it is never whole in 16 bits. Returns (batch, all tokens, heads, head_dim).

    norm_rope(query, key, group) normalizes and rotates in place query and/or key, the other one may be None: the joined projections,
    (batch, all tokens, heads of the group, head_dim), for the head ranges group.q and group.kv. With norm_spans_heads, a split group
    gets group.mean_squares to scale its rows; otherwise the tensors hold all heads. split_heads=False keeps all heads together.
    attention_fn: optional model-specific sparse attention accepting the disposable qkv_list and HeadGroup. It owns attention
    computation; projection preparation, head splitting and normalization still use this kit. Staged Sage quantization is disabled.
    sample_query: optional (projected_rows, token_ids, core) callback reproducing the fused query norm and RoPE at the requested positions.
    A V-first core can then select a key format from the same prepared projection before it packs the full keys.
    kv_prefix: (key, value) of shape (batch, prefix tokens, kv_heads, head_dim), already normalized and rotated, put before the
    projected keys and values; or a cached prefix (cache_prefix, q8_prefix: KVPrefix), q8 / q4_0 dequantized into the joined keys and values,
    native attended in a pass of its own (a native prefix needs the staged core it was made for: no mask, no attention_fn), a prefix
    of one batch item serving all. attention_mask: pay_attention's mask, or a callable mask(query, key, group) building
    the group's mask from its normalized and rotated query and key (the key with its prefix): called with the key alone first (it
    keeps what it needs, the key is then quantized and released), then with the query alone, returning the mask.
    kv_heads < heads: grouped query attention.
    norm_stats(project): optional model-native mean-square calculation, used only when splitting a norm spanning all heads;
    project(which, head_range) supplies disposable projections. Otherwise statistics are accumulated by groups below.
    norm_v(value, group): optional in-place value normalization, before prefix concatenation or quantization.
    When every K/V stream shares its k and v module, reuse its projection with independent tensors for normalization.
    softmax_scale: explicit SDPA scale; requires force_attention="sdpa".
    head_gate: (batch, all tokens, heads) in the output's dtype, each head's output multiplied by it (LTX-2.5's gated attention) as
    each group's output is made, before out_proj: the same products as on the whole output.
    norm_rows: norm_rope also takes all heads of a range of tokens (group.rows, of the tensor's stream: RoPE tables sliced by it);
    "heads": any range of heads of a range of tokens, with group.mean_squares (only the kept heads are then normalized). With
    norm_spans_heads and split heads on a staged core, q and k are then projected once, all heads by chunks of tokens (mmgp
    offload.linear_input_rows), normalized, rotated and quantized into whole INT8 q and k that the groups attend by ranges of heads,
    instead of a second projection for the statistics; k's smoothing mean is its first chunk's (any mean gives the same attention).
    head_split_level: Peak Fit level of this call instead of the Smart Inputs Chunker's (a model-specific sparse attention
    bounding its working set). A stream may carry a fourth projection: attention_fn then receives its head group's rows in a list.
    fused_quant(query, key, group, core, settings): optional norm + RoPE fused with a staged core's quantization (for example
    shared.kernels.ops.split_half_norm_rope_quant_), returning (q_quantized, k_quantized) for the tensors given (the other None) as
    the core's staged_quantize_q / staged_quantize_k do, or None to run norm_rope and the core's quantizers instead; used by the
    staged cores unless the attention_mask is a callable (it reads the normalized 16-bit q and k); with a kv_prefix only q is fused
    (the prefix joins the normalized 16-bit k), except with a native prefix (k fused with the prefix's mean: settings["k_mean"]).
    kv_window: (key, value, keep), the next window of a sliding cache, (batch, keep + new rows, kv_heads, head_dim) each, that this call
    fills by groups of heads, once a group's prefix has joined its keys and values: the last keep rows of kv_prefix, then the last new
    rows of the projected keys (normalized and rotated) and values. They may be kv_prefix's own tensors: the next call's prefix needs no
    tensors of its own. kv_prefix must then be (key, value) tensors or None (keep 0); k is not fused with its quantization.
    out_proj: the layer that projects the attention output, applied here (shared.kernels.ops.project): the call returns its output
    (batch, all tokens, out_features), or with residual, (batch * all tokens, out_features) or (batch, all tokens, out_features)
    borrowed, residual after the output was added to it, scaled by gate (None, a number, one value per channel, or with segments
    [(start, stop, row)] over the rows a table of gate rows). Results are those of projecting the returned attention, and where the
    heads are split and the layer's weight format allows it (ops.column_input: INT8 layers, their LoRAs included), each group's
    output is kept as that format reads it instead of one 16-bit (batch, tokens, heads, head_dim) tensor.
    add_after (with residual): the projection's output rounded to its dtype, then added to residual scaled by gate, as projecting the
    returned attention and adding it would (two roundings instead of the projection's residual epilogue), by ADD_AFTER_ROWS rows at a
    time: neither the whole output nor the per-segment paths of the layer (segments of a few thousand rows: frames).
    out_stream (with out_proj and residual): out_proj, residual, gate and segments apply to that stream's tokens only (residual and
    segments over its batch * tokens rows); the call returns the other streams' attention (batch, their tokens, heads, head_dim), in
    their order (MMDiT blocks: the image stream projected into its residual here, the text returned for its own projection).
    With EXTREME_LOW_VRAM one stream (of norms per head when there are several) may be a RowSource made again by every head group."""
    global _unsplit_notice
    mode = force_attention or offload.shared_state.get("_attention")
    token_mask = attention_mask is not None and mode not in sparge_core.MODES.values()
    if softmax_scale is not None and (attention_fn is not None or not token_mask and force_attention != "sdpa"):
        raise ValueError("An explicit softmax_scale requires a token mask or force_attention='sdpa', and no attention_fn")
    inputs, projections = [], []
    for x_list, stream_projections in streams:
        x = x_list[0]
        x_list.clear()
        inputs.append(x if x.dim() > 2 else x.unsqueeze(0))
        projections.append(stream_projections)
    streams = x = None
    shared_kv = projections[1][1] is projections[1][2] if _cross else all(modules[1] is modules[2] for modules in projections)
    extra = not _cross and len(projections[0]) > 3
    batch, lengths, dtype, device = inputs[0].shape[0], [x.shape[1] for x in inputs], inputs[0].dtype, inputs[0].device
    tokens = lengths[0] if _cross else sum(lengths)
    kv_heads = kv_heads or heads
    if attention_mask is None and (force_attention or offload.shared_state.get("_attention")) in ("sol3", "sol") and (
            _cross or kv_prefix is not None or kv_window is not None or (mode == "sol" and kv_heads != heads) or
            (len(inputs) > 1 and not (sol3_prefix_tokens or sol3_suffix_tokens))):
        force_attention = get_default_attention_mode()
    if token_mask and torch.is_tensor(attention_mask) and attention_mask.dtype != torch.bool and not attention_mask.is_floating_point():
        raise TypeError("Token attention masks must be boolean or floating point.")
    # Projections can introduce gradients even when the model input requires none.
    # The direct dispatcher checks the projected tensors before selecting an inference kernel.
    staged = None if attention_fn is not None or token_mask and (torch.is_grad_enabled() or softmax_scale is not None) else sage_staged(
        device, head_dim, force_attention, attention_mask is not None, dtype=dtype, tokens=tokens, batch=batch,
        sol3_prefix_tokens=sol3_prefix_tokens, sol3_suffix_tokens=sol3_suffix_tokens, sol_tau=sol_tau)
    split_tokens = batch * (max(lengths) if _cross else tokens)
    if out_stream is not None and (out_proj is None or residual is None or _cross):
        raise ValueError("out_stream needs out_proj and residual, and joined streams")
    out_start = 0 if out_stream is None else sum(lengths[:out_stream])  # the tokens out_proj projects
    out_tokens = tokens if out_stream is None else lengths[out_stream]
    if split_heads and head_split_level is None and split_tokens >= MIN_SPLIT_TOKENS:  # Auto: all heads at once where they fit (their q, k, v and output)
        whole = kv_heads * _head_bytes(_cross, batch, lengths, tokens, heads, kv_heads, head_dim, dtype, staged) + batch * (lengths[0] if _cross else tokens) * heads * head_dim * dtype.itemsize
        split_heads = chunking_needed(device, whole, ("attention", heads, kv_heads, head_dim, batch, tuple(lengths), dtype, _cross))
    group_heads = EXTREME_GROUP_HEADS
    if EXTREME_LOW_VRAM and chunker == "extreme":  # the floor: groups of at least EXTREME_GROUP_BYTES (Auto chose its level with it)
        group_heads = extreme_group_heads(kv_heads, _head_bytes(_cross, batch, lengths, tokens, heads, kv_heads, head_dim, dtype, staged))
    extreme = EXTREME_LOW_VRAM and head_split_level is None and split_tokens >= MIN_SPLIT_TOKENS and kv_heads > group_heads
    extreme_sizes = [min(group_heads, kv_heads - start) for start in range(0, kv_heads, group_heads)]
    sizes = extreme_sizes if extreme else head_partition(kv_heads, split_tokens, head_split_level) if split_heads else [kv_heads]  # the sizes follow the launch probe below, once the inputs are prepared
    groups = len(sizes)
    if (staged is not None and groups == 1 and kv_prefix is None and kv_window is None and attention_mask is None
            and heads == kv_heads and fp4_geometry(device, dtype, batch, tokens, lengths[1] if _cross else tokens,
                                                    heads, kv_heads, head_dim, force_attention)):
        # The FP4 kernel reads the long side directly after norm/RoPE. Account
        # for the unquantized projections before taking that path under Auto.
        direct_bytes = kv_heads * _head_bytes(_cross, batch, lengths, tokens, heads, kv_heads, head_dim, dtype, None)
        direct_bytes += batch * tokens * heads * head_dim * dtype.itemsize
        if chunker != "auto" or not chunking_needed(device, direct_bytes, ("nvfp4", heads, batch, tuple(lengths), dtype, _cross)):
            staged = None
    cached = isinstance(kv_prefix, KVPrefix)
    native = cached and kv_prefix.k_mean is not None
    if kv_window is not None and cached:
        raise ValueError("kv_window takes its kept rows from a prefix of (key, value) tensors, not from a cached prefix")
    prefix_rows = 0 if kv_prefix is None or cached else kv_prefix[0].shape[1]

    def window(which, joined, head_range):
        """kv_window's rows of these heads from the joined keys (which 1) or values (2), the prefix copied into them already."""
        target, keep = kv_window[which - 1][:, :, head_range.start:head_range.stop], kv_window[2]
        if keep:
            target[:, :keep].copy_(joined[:, prefix_rows - keep:prefix_rows])
        target[:, keep:].copy_(joined[:, joined.shape[1] - (target.shape[1] - keep):])
    if native and (staged is None or attention_mask is not None or not hasattr(staged[0], "prefix_k")):
        raise ValueError("A native cached prefix is attended by the unmasked staged attention it was made for (cache_prefix native=False otherwise)")
    linear_inputs = [None] * len(inputs)
    # EXTREME_LOW_VRAM: the input given by ranges of rows, made again by each group range by range (grouped_rows), never whole
    # (one stream of several: the others prepared whole as usual, norms over all heads only with a single stream)
    ranged_source, lora_products = None, None
    ranged_index = next((index for index, x in enumerate(inputs) if isinstance(x, offload.RowSource)), None)
    if (EXTREME_LOW_VRAM and ROW_SOURCES and groups > 1 and not _cross and ranged_index is not None and (len(inputs) == 1 or not norm_spans_heads)
            and offload.linear_rows_supported(projections[ranged_index], inputs[ranged_index])):
        ranged_source, inputs[ranged_index] = inputs[ranged_index], None
        range_rows = max(128, EXTREME_RANGE_BYTES // (ranged_source.shape[-1] * ranged_source.dtype.itemsize) // 128 * 128)
        if EXTREME_LORA_PRODUCTS:
            lora_products = offload.row_lora_products(ranged_source, tuple(module for module in projections[ranged_index] if module is not None), range_rows)
    else:
        ranged_index = None
    out_columns = (ops.column_input(out_proj, batch * out_tokens, dtype, device, None if add_after else segments, None if add_after else residual)
                   if out_proj is not None and groups > 1 else None)  # add_after reads it by spans of rows (linear(first, last))
    # one stream projected by groups and read back by out_proj's kept input: its input is prepared in its own memory, which then
    # holds that kept input too (the bytes the preparation frees: out_proj's input is as large as INT8, where its width is the input's)
    in_place = out_columns is not None and hasattr(out_columns, "reuse") and not _cross and len(inputs) == 1 and ranged_source is None
    for index, modules in enumerate(projections):
        if index == ranged_index:
            continue
        if _cross:
            modules = tuple(module for module in modules if module is not None)
        if shared_kv:
            modules = tuple(dict.fromkeys(modules))
        if isinstance(inputs[index], offload.RowSource) and not ROW_SOURCES:
            inputs[index] = inputs[index].materialize()
        if offload.linear_rows_supported(modules, inputs[index]):
            x_holder, inputs[index] = [inputs[index]], None
            linear_inputs[index] = offload.prepare_linear_input(x_holder, modules, in_place)
        elif isinstance(inputs[index], offload.RowSource):
            inputs[index] = inputs[index].materialize()
    if in_place and linear_inputs[0] is not None:
        out_columns.reuse(linear_inputs[0].spare)
    prepared_room = (0, 0)  # the call's peak while its inputs were prepared, and what they hold then (_room_keep)
    if _room is not None and not _room[3]:
        prepared_room = (allocator.probe_peak(device) - _room[1], allocator.probe_now(device) - _room[1])
        allocator.probe_reset(device)
    # a stream whose projections cannot be computed by groups of heads is projected whole once and sliced, if it is too short to be split
    if groups > 1 and batch * sum(length for index, (length, prepared) in enumerate(zip(lengths, linear_inputs)) if prepared is None and index != ranged_index) >= MIN_SPLIT_TOKENS:
        if not _unsplit_notice:
            tqdm.write("[Attention] Smart Inputs Chunker: heads not split here, these q/k/v projections cannot be computed by groups of heads (weight format or LoRA type).")
            _unsplit_notice = True
        groups, sizes, out_columns = 1, [kv_heads], None
    shape, token_keep = None, 0
    # The token cache joins NHD packed tensors and batch-major scales. Sage1 uses flattened varlen rows and offsets.
    batch_packed = staged is not None and "qk_quant_gran" in staged[1] and staged[1].get("padded_dim") is None
    token_capable = (norm_rows and norm_spans_heads and batch_packed and attention_fn is None and kv_prefix is None
                     and kv_window is None and not callable(attention_mask) and not extra and not shared_kv and norm_v is None and (len(lengths) == 1 or _cross)
                     and TOKEN_QK_POLICY != "none" and all(prepared is not None and offload.linear_input_rows(prepared, 0, 1) is not None for prepared in linear_inputs))
    if groups > 1 and WAVE_SPLIT and attention_fn is None and not callable(attention_mask) and device.type == "cuda":  # the launch probe, asked only where the heads are split
        model = launch_waves(device, head_dim, force_attention, attention_mask is not None)
        group_ms = _group_ms(projections, linear_inputs, lengths, batch, head_dim, device, shared_kv) if model is not None else None
        if model is not None:
            # a group holds its q (16 bits, then INT8 too), k and v (INT8), or all three in 16 bits without a staged core; the call also
            # holds its output (a byte per element when kept for out_proj). Larger groups also take the measured free room (_Room).
            head_bytes = _head_bytes(_cross, batch, lengths, tokens, heads, kv_heads, head_dim, dtype, staged)
            keys = max(lengths) if _cross else tokens
            shape = (heads, kv_heads, head_dim, batch, tuple(lengths), dtype, staged is not None, out_columns is not None, kv_prefix is not None)
            q_width, k_width = _in_width(projections[0][0]), _in_width(projections[1][1] if _cross else projections[0][1])
            speed_keep = lambda partition: _speed_keep(partition, heads // kv_heads, batch, max(lengths) if _cross else tokens, lengths[1] if _cross else tokens,
                                                       q_width, k_width, head_dim) if token_capable else 0
            if _room is not None and _room[3] and shape in _Room.chosen:  # a capture: its warmup's groups
                sizes = list(_Room.chosen[shape])
                token_keep = speed_keep(sizes)  # as the warmup kept them (the room's keep is the warmup's eager measure: none recorded)
            else:
                waves = (batch * heads * -(-tokens // model.rows), model)
                if token_capable:  # the measured room goes to kept q / k where they fit (no second projection), else to larger groups
                    base = head_partition(kv_heads, split_tokens, head_split_level, waves, kv_heads / (dtype.itemsize + 3), head_bytes, 0, keys, 0, group_ms)
                    q_rows, k_rows = batch * (max(lengths) if _cross else tokens) * head_dim, batch * (lengths[1] if _cross else tokens) * head_dim
                    token_keep = _room_keep(_room, shape, base, [int((size * (heads // kv_heads) * q_rows + size * k_rows) * 1.02) for size in base])  # INT8, scales
                    if cuda_graphs.recording() or cuda_graphs.capturing():
                        token_keep = 0  # a block warmed up and recorded without its eager room measure: the speed keep alone, as its capture
                    token_keep = max(token_keep, speed_keep(base))  # never project again more than FIT_REPROJECT of the attention
                if token_keep:
                    sizes = base
                else:
                    slack, last_slack = _room_slack(_room, shape, kv_heads // groups, head_bytes)
                    sizes = head_partition(kv_heads, split_tokens, head_split_level, waves, kv_heads / (dtype.itemsize + 3), head_bytes, slack, keys, last_slack, group_ms)
            if extreme:
                sizes, token_keep = extreme_sizes, 0
            groups = len(sizes)
            if _room is not None:
                _room[2][:] = [shape, tuple(sizes), head_bytes, batch * tokens * (heads // kv_heads) * head_dim * (1 if out_columns is not None else dtype.itemsize), *prepared_room]
            out_columns = out_columns if groups > 1 else None

    def gate_(attention, head_range):
        """attention (batch, tokens, heads of head_range, head_dim) multiplied in place by head_gate's heads."""
        if head_gate is not None:
            attention.mul_(head_gate[..., head_range.start:head_range.stop].unsqueeze(-1))
        return attention

    rest = []  # out_stream: the other streams' attention (batch, their tokens, heads, head_dim)

    def stream_rows(attention, head_range):
        """attention (batch, tokens, heads of head_range, head_dim): with out_stream, the other streams' tokens copied into rest and
        out_stream's tokens returned (a view where they are contiguous: one batch item); attention itself otherwise."""
        if out_stream is None:
            return attention
        if not rest:
            rest.append(attention.new_empty((batch, tokens - out_tokens, heads, head_dim)))
        target, stop = rest[0][:, :, head_range.start:head_range.stop], out_start + out_tokens
        target[:, :out_start].copy_(attention[:, :out_start])
        target[:, out_start:].copy_(attention[:, stop:])
        part = attention[:, out_start:stop]
        return part if batch == 1 else part.contiguous()

    def finish(attention):
        """The attention output (a tensor, or out_proj's column input) through out_proj when it is given."""
        if out_proj is None:
            return attention
        if out_stream is not None:
            if torch.is_tensor(attention):
                attention = stream_rows(attention, range(heads))
            finish_stream(attention)
            return rest.pop()
        return finish_stream(attention)

    def finish_stream(attention):
        if residual is not None and add_after:  # projected by spans of rows, each rounded, then added: never the whole output
            total, column, rows = batch * out_tokens, not torch.is_tensor(attention), residual.view(-1, residual.shape[-1])
            flat = None if column else attention.reshape(total, -1)
            for first in range(0, total, ADD_AFTER_ROWS):
                last = min(total, first + ADD_AFTER_ROWS)
                part = attention.linear(first, last) if column else out_proj(flat[first:last])
                for start, stop, row in segments or [(0, total, None)]:
                    low, high = max(start, first), min(stop, last)
                    if low < high:
                        scale = gate if segments is None else gate[row]
                        target, source = rows[low:high], part[low - first:high - first]
                        target.add_(source) if scale is None else target.addcmul_(source, scale.to(target.dtype))
                part = None
            return residual
        projected = ops.project(out_proj, [attention.flatten(2) if torch.is_tensor(attention) else attention], residual, gate, segments)
        return projected if residual is not None else projected.view(batch, out_tokens, -1)
    whole = {}

    ranged = {}  # EXTREME_LOW_VRAM: the projections of the current group, made in one pass over the ranges of rows (grouped_rows)

    def ranged_statistics():
        """The per token mean squares of q and of k over all heads (HeadGroup.mean_squares) from one pass over the ranges of rows of the
        input: each range made and prepared (offload.prepare_rows), projected through all of q's and k's columns, reduced and dropped
        (re-projecting chunks of heads would take a pass over the input per chunk)."""
        modules, total = projections[ranged_index], ranged_source.rows()
        present = tuple(module for module in modules if module is not None)
        squares = [torch.empty(total, dtype=torch.float32, device=device) for _ in range(2)]
        for first in range(0, total, range_rows):
            last = min(total, first + range_rows)
            prepared = offload.prepare_rows(ranged_source, first, last, present, lora_products)
            for which, width in ((0, heads), (1, kv_heads)):
                part = offload.linear_rows(modules[which], prepared, 0, width * head_dim)
                squares[which][first:last] = torch.linalg.vector_norm(part.reshape(last - first, -1), dim=-1, dtype=torch.float32).square_()
                del part
            prepared = None
        return tuple(s.view(batch, -1).div_(width * head_dim).unsqueeze(-1) for s, width in zip(squares, (heads, kv_heads)))

    def grouped_rows(wanted):
        """{(which, start, stop): output columns of projection which for these heads} from one pass over the ranges of rows of the input:
        each range made and prepared (offload.prepare_rows: in one pass where the source can), projected and released, so that the
        input is never whole."""
        modules, total = projections[ranged_index], ranged_source.rows()
        present = tuple(module for module in modules if module is not None)
        step = range_rows
        full = total // step
        # while a block is captured, the full ranges after the first in one CUDA graph loop: a turn per range, not its kernels per range
        group = offload.rows_loop_group(ranged_source, present, step) if full >= 3 else None
        outs = {}

        def project(prepared, store):
            for which, head_range in wanted:
                part = offload.linear_rows(modules[which], prepared, head_range.start * head_dim, head_range.stop * head_dim)
                part = part.reshape(-1, part.shape[-1])
                key = (which, head_range.start, head_range.stop)
                if key not in outs:
                    outs[key] = part.new_empty((total, part.shape[-1]))
                store(outs[key], part)
                del part

        firsts = [0, *range(full * step, total, step)] if group is not None else range(0, total, step)
        for first in firsts:
            last = min(total, first + step)
            project(offload.prepare_rows(ranged_source, first, last, present, lora_products), lambda target, part: target[first:last].copy_(part))
            if group is not None and first == 0:
                counter = torch.empty(1, dtype=torch.int64, device=ranged_source.device)
                with cuda_graphs.loop(full - 1, counter):
                    project(offload.prepare_rows_at(ranged_source, counter, step, step, present, group, lora_products),
                            lambda target, part: offload.store_rows_at(target, part, counter, step, step))
        return outs

    def project_stream(index, which, head_range):
        module, linear_input = projections[index][which], linear_inputs[index]
        if index == ranged_index:
            key = (which, head_range.start, head_range.stop)
            if key not in ranged:  # the group's projections at its first one; a projection outside the groups on its own
                current = ranged.pop("group", None)
                wanted = [(which, head_range)] if current is None else [(w, current.q if w in (0, 3) else current.kv) for w in range(len(projections[index]))
                                                                       if projections[index][w] is not None and not (shared_kv and w == 2)]
                if (which, head_range) not in wanted:
                    wanted = [(which, head_range)]
                ranged.update(grouped_rows(wanted))
            return ranged.pop(key).view(batch, lengths[index], -1, head_dim)
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
        nonlocal lora_products
        inputs.clear()
        linear_inputs.clear()
        whole.clear()
        lora_products = None

    def with_prefix(tensor, which, head_range):
        if kv_prefix is None or native:  # a native prefix is attended in a pass of its own
            return tensor
        if not cached:
            return torch.cat([kv_prefix[which - 1][:, :, head_range.start:head_range.stop], tensor], dim=1)
        data, scales = (kv_prefix.k, kv_prefix.k_scale) if which == 1 else (kv_prefix.v, kv_prefix.v_scale)
        joined = tensor.new_empty((tensor.shape[0], data.shape[1] + tensor.shape[1], *tensor.shape[2:]))
        dequantize_prefix_into(joined[:, :data.shape[1]], data[:, :, head_range.start:head_range.stop], scales[:, :, head_range.start:head_range.stop])
        joined[:, data.shape[1]:].copy_(tensor)
        return joined

    ratio, kv_ranges = heads // kv_heads, []  # the groups' key / value heads, and their query heads
    for size in sizes:
        kv_ranges.append(range(kv_ranges[-1].stop if kv_ranges else 0, (kv_ranges[-1].stop if kv_ranges else 0) + size))
    q_ranges = [range(head_range.start * ratio, head_range.stop * ratio) for head_range in kv_ranges]
    mean_squares, kept = None, []
    split_dense = None
    if (not _cross and kv_prefix is None and kv_window is None and attention_mask is None
            and attention_fn is None and softmax_scale is None):
        candidate = dense_staged(staged, device, dtype, batch, tokens, max(sizes) * ratio, max(sizes), head_dim, force_attention,
                                 recycle_query=groups == 1 or out_columns is not None or token_capable,
                                 split_keys=groups > 1 and token_capable)
        if candidate is not None and "recycle_value" in candidate[1]:
            # Prepared projections, joined streams and cloned shared K/V all produce disposable V rows.
            candidate[1]["recycle_value"] = shared_kv or len(lengths) > 1 or all(v is not None for v in linear_inputs)
        if groups == 1 or not token_capable:
            staged = candidate
        elif candidate is not None:
            # A core offering pack_split_keys declares where its extra projection and group calibration pay off.
            # Its BF16 pilot samples must be read before the ordinary chunk pass converts Q/K to INT8.
            eligible = getattr(candidate[0], "split_keys_eligible", None)
            if eligible is not None and eligible(tokens, candidate[1]):
                split_dense = candidate
    # a cross attention over a short context, split: by chunks of query tokens with all heads instead (each chunk's norm over all
    # heads is its own: one projection of q; k and v of the whole context once), each chunk gated and projected at once
    query_chunks = (norm_rows and _cross and groups > 1 and batch_packed and attention_fn is None and kv_prefix is None
                    and kv_window is None and not callable(attention_mask) and not shared_kv and norm_v is None and (residual is None or out_proj is not None)
                    and linear_inputs[0] is not None
                    and batch * lengths[1] * kv_heads * head_dim * dtype.itemsize * 2 <= TOKEN_QK_BYTES and lengths[1] * 4 <= lengths[0]
                    and offload.linear_input_rows(linear_inputs[0], 0, 1) is not None)
    count = token_keep if token_capable and groups > 1 and not query_chunks else 0  # chosen with the partition
    if query_chunks:
        pass
    elif count or token_capable and groups > 1 and CHUNK_STATISTICS:  # q and k of all heads by chunks of tokens, never whole in 16 bits
        kept, mean_squares = _token_qk(projections, linear_inputs, lengths, batch, heads, kv_heads, head_dim, dtype, device, norm_rope, staged, _cross,
                                       list(zip(q_ranges[:count], kv_ranges[:count])), count < groups or split_dense is not None, norm_rows == "heads",
                                       split_dense=split_dense)
    elif norm_spans_heads and groups > 1 and ranged_source is not None:
        mean_squares = ranged_statistics()
    elif norm_spans_heads and groups > 1 and norm_stats is not None:
        mean_squares = norm_stats(project)
    elif norm_spans_heads and groups > 1:
        # q and k projected once more for their statistics, by chunks of heads as wide as the largest group's working set (its
        # q, k and v, 16-bit and quantized: STATS_WIDTH times its heads in one 16-bit projection): fewer, wider GEMMs
        mean_squares, chunk = [], max(1, int(max(sizes) * STATS_WIDTH))
        for which, total in ((0, heads), (1, kv_heads)):
            step, squares = max(1, chunk * total // kv_heads), None
            for start in range(0, total, step):
                part = project(which, range(start, min(total, start + step))).flatten(2)
                norm = torch.linalg.vector_norm(part, dim=-1, dtype=torch.float32).square_()  # one fp32 reduction, no fp32 copy
                del part
                squares = norm if squares is None else squares.add_(norm)
                del norm
            mean_squares.append(squares.div_(total * head_dim).unsqueeze(-1))
        mean_squares = tuple(mean_squares)

    if (staged is not None and not native and not callable(attention_mask) and (groups == 1 or all(i is not None for i in linear_inputs))
            and staged[1].get("qk_quant_gran") in (None, "per_block")):
        staged = owned_query_staged(staged)
    if staged is not None:
        core, settings = staged
        if native:  # the layout the prefix was kept in (cache_prefix)
            settings = core.prefix_settings(settings, kv_prefix.k.device, offload.shared_state.get("_sage_v_scaling", False))
        if attention_mask is not None and not callable(attention_mask):
            # Sparge supplies a head-major block map. Dense cores take a token mask in SageAttention's layout.
            attention_mask = getattr(core, "prepare_mask", sage_attention_mask)(attention_mask, dtype)
    if query_chunks:
        core, settings = staged
        all_heads = HeadGroup(range(heads), range(kv_heads), None)
        key, value_list = project(1, all_heads.kv), [project(2, all_heads.kv)]
        norm_rope(None, key, all_heads)
        quantized_kv = [core.staged_quantize_k([key], settings), core.staged_quantize_v(value_list, settings)]
        del key
        length = lengths[0]
        chunk = max(128, TOKEN_QK_BYTES // (heads * head_dim * dtype.itemsize) // 128 * 128)
        result, done = None, 0
        if batch == 1 and attention_mask is None and out_proj is not None and torch.cuda.is_current_stream_capturing():  # a block's capture
            done = _query_chunk_loop(projections[0][0], out_proj, linear_inputs[0], norm_rope, core, settings, [tuple(None if t is None else t[:1] for t in quantized)
                                     for quantized in quantized_kv], heads, head_dim, length, chunk, head_gate, residual, gate, segments, all_heads)
        for item in range(batch):
            item_kv = [tuple(None if t is None else t[item:item + 1] for t in quantized) for quantized in quantized_kv]
            for start in range(done, length, chunk):
                stop = min(length, start + chunk)
                # a block's warmup: one chunk timed for the loop of its capture
                timing = cuda_graphs.loop_turn_begin(out_proj, ("query chunk", chunk)) if start == chunk and stop - start == chunk and out_proj is not None else None
                rows = offload.linear_input_rows(linear_inputs[0], item * length + start, item * length + stop)
                query = offload.linear_rows(projections[0][0], rows, 0, heads * head_dim).view(1, stop - start, heads, head_dim)
                del rows
                norm_rope(query, None, HeadGroup(all_heads.q, all_heads.kv, None, range(start, stop)))
                mask = None if attention_mask is None else attention_mask[min(item, attention_mask.shape[0] - 1):][:1]
                mask = mask if mask is None or mask.dim() < 4 or mask.shape[-2] == 1 else mask[..., start:stop, :]
                core.staged_attention(query, [core.staged_quantize_q(query, settings), *item_kv], settings, mask)
                if head_gate is not None:
                    query.mul_(head_gate[item:item + 1, start:stop].unsqueeze(-1))
                if residual is not None:  # the chunk's projection added into its rows of the residual: no output of all tokens
                    first, rows_of = item * length + start, residual.view(-1, residual.shape[-1])
                    local = None if segments is None else [(max(low, first) - first, min(high, first + stop - start) - first, row)
                                                           for low, high, row in segments if high > first and low < first + stop - start]
                    ops.project(out_proj, [query.view(stop - start, -1)], rows_of[first:first + stop - start], gate, local)
                    del query
                    if timing is not None:
                        timing()
                    continue
                part = query if out_proj is None else ops.project(out_proj, [query.flatten(2)])
                del query
                if result is None:
                    result = part.new_empty((batch, length, *part.shape[2:]))
                result[item, start:stop].copy_(part[0])
                del part
        release()
        return residual if residual is not None else result
    # a callable mask reads the normalized 16-bit q and k; a prefix joins the normalized 16-bit k (q alone is still fused), not a native one
    fusing_q = staged is not None and fused_quant is not None and not callable(attention_mask)
    fusing_k = fusing_q and (kv_prefix is None or native) and kv_window is None
    output = None
    if _room is not None and _room[2] and not _room[3]:  # the call's own peak before its groups (inputs beside their prepared form, statistics), then each group's
        _room[2].append(allocator.probe_peak(device) - _room[1])
        allocator.probe_reset(device)
    for index in range(groups):
        group = HeadGroup(q_ranges[index], kv_ranges[index], mean_squares)
        last = index == groups - 1
        if ranged_source is not None:
            ranged.clear()
            ranged["group"] = group
        if split_dense is not None:
            core = split_dense[0]
            settings = kept[index][1] if index < len(kept) else split_dense[1].copy()
        if index and _room is not None and len(_room[2]) > 6:
            _room[2].append(allocator.probe_peak(device) - _room[1])
            allocator.probe_reset(device)
        if index < len(kept) and split_dense is None:  # regular core: both q and k have already been packed
            value_list = [project(2, group.kv)]
            quantized = [None, None, core.staged_quantize_v(value_list, settings)]
            if last:
                release()
            quantized[:2], kept[index] = kept[index], None
            mask = _head_slice(attention_mask, 1, group.q)
            if out_columns is not None:
                attention = torch.empty((batch, tokens, len(group.q), head_dim), dtype=dtype, device=device)
                gate_(core.staged_attention(attention, quantized, settings, mask), group.q)
                attention_list, attention, output = [stream_rows(attention, group.q)], None, out_columns
                out_columns.append(attention_list)
                continue
            if output is None:
                output = torch.empty((batch, tokens, heads, head_dim), dtype=dtype, device=device)
            gate_(core.staged_attention(output[..., group.q.start:group.q.stop, :], quantized, settings, mask), group.q)
            continue
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
            if kv_window is not None:
                window(1, qkv_list[1], group.kv)
                window(2, qkv_list[2], group.kv)
            if callable(attention_mask):
                attention_mask(None, qkv_list[1], group)
                mask = attention_mask(qkv_list[0], None, group)
                if mask is not None and token_mask:
                    mask = mask.transpose(1, 2) if mask.ndim == 4 else mask  # callable token masks return kernel layout
            else:
                mask = _head_slice(attention_mask, 2, group.q)
            if ratio != 1:  # preserve each unstaged backend's projection layout and arithmetic
                qkv_list[1:] = [tensor.repeat_interleave(ratio, dim=2) for tensor in qkv_list[1:]]
            if attention_fn is None:
                attention = pay_attention(qkv_list, attention_mask=mask, force_attention=force_attention, recycle_q=True, softmax_scale=softmax_scale, sol3_prefix_tokens=sol3_prefix_tokens, sol3_suffix_tokens=sol3_suffix_tokens, sol_tau=sol_tau)
            else:
                attention = attention_fn(qkv_list, group) if extra_list is None else attention_fn(qkv_list, group, extra_list)
            gate_(attention, group.q)
            if groups == 1:
                return finish(attention)
            if out_columns is not None:
                attention_list, attention, output = [stream_rows(attention, group.q)], None, out_columns
                out_columns.append(attention_list)
                continue
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
        if kv_window is not None:
            window(2, value_list[0], group.kv)
        quantized = [None, None, core.staged_quantize_v(value_list, settings)]
        if (sample_query is not None and fusing_k and fusing_q and batch == 1 and len(lengths) == 1
                and not _cross and not norm_spans_heads and kv_prefix is None and kv_window is None
                and linear_inputs[0] is not None and hasattr(core, "early_query_ids")):
            ids = core.early_query_ids(settings, tokens, len(group.q), device)
            if ids is not None:
                # Use the original GEMM dimensions, including LoRAs. Projecting only the pilot rows
                # can change a BF16 rounding and, through the fitted value decoder, the generated clip.
                pilot = project(0, group.q)[0].index_select(0, ids).unsqueeze(0)
                settings["early_q"] = sample_query(pilot, ids, core)
                del pilot
        # the keys of a native prefix's passes share its mean
        k_settings = {**settings, "k_mean": core.prefix_mean_heads(kv_prefix.k_mean, group.kv)} if native else settings
        # k is quantized and released before q is projected: one 16-bit tensor of the group (or of all heads) at a time
        if split_dense is not None and index < len(kept):
            quantized[1], kept[index] = kept[index][0], None
        else:
            if not shared_kv:
                key = project(1, group.kv)
            fused = fused_quant(None, key, group, core, k_settings) if fusing_k else None
            if fused is None:
                norm_rope(None, key, group)
                key = with_prefix(key, 1, group.kv)
                if kv_window is not None:
                    window(1, key, group.kv)
                if callable(attention_mask):
                    attention_mask(None, key, group)
                if hasattr(core, "staged_quantize_k_with_v"):
                    quantized[1] = core.staged_quantize_k_with_v([key], quantized[2], k_settings)
                else:
                    quantized[1] = core.staged_quantize_k([key], k_settings)
            else:
                quantized[1] = fused[1]
            del key, fused
        query = project(0, group.q)
        if last:
            release()
        fused = fused_quant(query, None, group, core, settings) if fusing_q else None
        if fused is None:
            norm_rope(query, None, group)
            mask = attention_mask(query, None, group) if callable(attention_mask) else _head_slice(attention_mask, 1, group.q)
            quantized[0] = core.staged_quantize_q(query, settings)
        else:
            mask, quantized[0] = _head_slice(attention_mask, 1, group.q), fused[0]
        del fused
        if groups == 1:
            return finish(gate_(_attend_with_prefix(core, settings, query, quantized, kv_prefix, group.kv) if native else core.staged_attention(query, quantized, settings, mask), group.q))
        if out_columns is not None:  # attended into q's storage, as all heads together are
            if native:
                _attend_with_prefix(core, settings, query, quantized, kv_prefix, group.kv)
            else:
                core.staged_attention(query, quantized, settings, mask)
            gate_(query, group.q)
            query_list, query, output = [stream_rows(query, group.q)], None, out_columns
            out_columns.append(query_list)
            continue
        if output is None:
            output = query.new_empty((batch, tokens, heads, head_dim))
        if native:
            _attend_with_prefix(core, settings, output[..., group.q.start:group.q.stop, :], quantized, kv_prefix, group.kv)
        else:
            core.staged_attention(output[..., group.q.start:group.q.stop, :], quantized, settings, mask)
        gate_(output[..., group.q.start:group.q.stop, :], group.q)
        del query
    return finish(output)


cuda_graphs.register_key(_graph_key)
