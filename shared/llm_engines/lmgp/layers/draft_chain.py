# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# With ideas of Strata (MIT License, https://github.com/Niko1221/Strata): draft tokens read by the host from mapped RAM, as a kernel writes them.
"""A chain of draft tokens without a synchronization or a tensor per draft: after each pass of the drafter two kernels write its likeliest tokens and their probabilities to mapped RAM, the host reads them as they arrive,
chooses, and one kernel gives the next pass its inputs. Sampling: any proposal keeps verification exact (the rejection step corrects it, see dflash_sampling.accept_block), and a proposal over the few likeliest tokens has nearly the overlap of the
whole distribution, at the cost of a few words instead of a row of the vocabulary per draft."""
import random

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from mmgp import mapped

BLOCK, KEEP = 1024, 4  # tokens per program of the first kernel, and those it keeps: a block rarely holds more than a few tokens that matter
CANDIDATES = 1024  # the tile holding what the blocks keep, for the largest vocabulary below
MAX_VOCAB = CANDIDATES * BLOCK // KEEP
MAX_OUT = 16  # tokens of a proposal, the likeliest first
HIDDEN_TILE = 512
_LOWEST = tl.constexpr(-1e30)


@triton.jit
def _fence(value):
    """The stores before it have reached system memory before those after it."""
    return tl.inline_asm_elementwise("membar.sys; mov.u32 $0, $1;", "=r,r", [value], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit(do_not_specialize=("temperature",), do_not_specialize_on_alignment=("LOGITS",))
def _block_top(LOGITS, VALUES, IDS, SUMS, temperature, VOCAB: tl.constexpr, BLOCK_SIZE: tl.constexpr, KEPT: tl.constexpr):
    """Per block of tokens: its KEPT largest logits and the mass of its tokens relative to its largest."""
    block = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE)
    tokens = block * BLOCK_SIZE + rows
    inside = tokens < VOCAB
    x = tl.where(inside, tl.load(LOGITS + tokens, mask=inside, other=0.0).to(tl.float32), _LOWEST)
    top = tl.max(x)
    tl.store(SUMS + block, tl.sum(libdevice.exp((x - top) / temperature)))
    for kept in tl.static_range(KEPT):
        index = tl.argmax(x, 0)
        tl.store(VALUES + block * KEPT + kept, tl.max(x))
        tl.store(IDS + block * KEPT + kept, block * BLOCK_SIZE + index)
        x = tl.where(rows == index, _LOWEST, x)


@triton.jit(do_not_specialize=("temperature", "top_p", "stamp"), do_not_specialize_on_alignment=("OUT_IDS", "OUT_PROBS", "OUT_COUNT", "OUT_STAMP"))
def _select(VALUES, IDS, SUMS, OUT_IDS, OUT_PROBS, OUT_COUNT, OUT_STAMP, temperature, top_p, stamp, BLOCKS: tl.constexpr, KEPT: tl.constexpr, TILE: tl.constexpr, OUT: tl.constexpr):
    """The likeliest tokens of all the blocks with their probabilities, until they hold top_p of the mass or OUT tokens, to mapped RAM; the stamp comes last."""
    index = tl.arange(0, TILE)
    present = index < BLOCKS * KEPT
    values = tl.load(VALUES + index, mask=present, other=_LOWEST)
    ids = tl.load(IDS + index, mask=present, other=0)
    top = tl.max(values)
    heads = present & (index % KEPT == 0)
    total = tl.sum(tl.load(SUMS + index // KEPT, mask=heads, other=0.0) * libdevice.exp((values - top) / temperature))
    mass = top * 0.0
    count = tl.program_id(0) * 0
    for out in range(OUT):
        if mass < top_p:
            position = tl.argmax(values, 0)
            chosen = index == position
            tl.store(OUT_IDS + out, tl.sum(tl.where(chosen, ids, 0)))
            probability = tl.div_rn(libdevice.exp((tl.max(values) - top) / temperature), total)
            tl.store(OUT_PROBS + out, probability)
            values = tl.where(chosen, _LOWEST, values)
            mass = mass + probability
            count = count + 1
    tl.store(OUT_COUNT, count)
    tl.store(OUT_STAMP, _fence(stamp))
    _fence(stamp)  # on its way to the host now


@triton.jit(do_not_specialize_on_alignment=("CONTEXT", "SOURCE"))
def _feed(CONTEXT, INPUT_IDS, POSITIONS, SEQLENS, SOURCE, HIDDEN, WIDTH: tl.constexpr, TILE: tl.constexpr):
    """The inputs of the drafter's next pass: its token, position and place in its cache from mapped RAM, the streams of the pass before."""
    program = tl.program_id(0)
    if program == 0:
        tl.store(INPUT_IDS, tl.load(CONTEXT).to(INPUT_IDS.dtype.element_ty))
        tl.store(POSITIONS, tl.load(CONTEXT + 1).to(POSITIONS.dtype.element_ty))
        tl.store(SEQLENS, tl.load(CONTEXT + 2).to(SEQLENS.dtype.element_ty))
    offsets = program * TILE + tl.arange(0, TILE)
    inside = offsets < WIDTH
    tl.store(HIDDEN + offsets, tl.load(SOURCE + offsets, mask=inside, other=0.0).to(HIDDEN.dtype.element_ty), mask=inside)


@triton.jit
def _load_q(Q, IDS, PROBS, COUNTS, OLD_IDS, OLD_COUNTS, VOCAB: tl.constexpr, OUT: tl.constexpr):
    """Rows of the draft distributions that the acceptance reads: the entries of the last proposal cleared, those of this one written."""
    row = tl.program_id(0)
    index = tl.arange(0, OUT)
    tl.store(Q + row * VOCAB + tl.load(OLD_IDS + row * OUT + index), 0.0, mask=index < tl.load(OLD_COUNTS + row))
    tl.debug_barrier()  # a token of both proposals keeps its new value
    ids = tl.load(IDS + row * OUT + index)
    fresh = index < tl.load(COUNTS + row)
    tl.store(Q + row * VOCAB + ids, tl.load(PROBS + row * OUT + index), mask=fresh)
    tl.store(OLD_IDS + row * OUT + index, ids)
    tl.store(OLD_COUNTS + row, tl.load(COUNTS + row))


@triton.jit(do_not_specialize=("temperature", "top_p", "slot"))
def _select_draw(VALUES, IDS, SUMS, OUT_IDS, OUT_PROBS, OUT_COUNT, DRAFT_IDS, UNIFORMS, BASES, NEXT_INPUT, NEXT_POSITION, NEXT_SEQLEN, temperature, top_p, slot,
                 BLOCKS: tl.constexpr, KEPT: tl.constexpr, TILE: tl.constexpr, OUT: tl.constexpr, FEED: tl.constexpr):
    """_select into device memory, then the draw of the host chain (DraftChain.draw) on the GPU: the proposal normalized in place, the token
    the first whose cumulative probability reaches the slot's uniform. FEED: the drafter's next pass reads its token, position (BASES[0] + slot)
    and cache length (BASES[1] + slot)."""
    index = tl.arange(0, TILE)
    present = index < BLOCKS * KEPT
    values = tl.load(VALUES + index, mask=present, other=_LOWEST)
    ids = tl.load(IDS + index, mask=present, other=0)
    top = tl.max(values)
    heads = present & (index % KEPT == 0)
    total = tl.sum(tl.load(SUMS + index // KEPT, mask=heads, other=0.0) * libdevice.exp((values - top) / temperature))
    outs = tl.arange(0, OUT)
    out_ids = tl.zeros((OUT,), dtype=tl.int32)
    out_probs = tl.zeros((OUT,), dtype=tl.float32)
    mass = top * 0.0
    count = tl.program_id(0) * 0
    for out in range(OUT):
        if mass < top_p:
            position = tl.argmax(values, 0)
            chosen = index == position
            probability = tl.div_rn(libdevice.exp((tl.max(values) - top) / temperature), total)
            out_ids = tl.where(outs == out, tl.sum(tl.where(chosen, ids, 0)), out_ids)
            out_probs = tl.where(outs == out, probability, out_probs)
            values = tl.where(chosen, _LOWEST, values)
            mass = mass + probability
            count = count + 1
    kept = outs < count
    out_probs = tl.where(kept, tl.div_rn(out_probs, tl.sum(tl.where(kept, out_probs, 0.0))), 0.0)
    cdf = tl.cumsum(out_probs, 0)
    choice = tl.minimum(tl.sum((kept & (cdf < tl.load(UNIFORMS + slot))).to(tl.int32)), count - 1)
    token = tl.sum(tl.where(outs == choice, out_ids, 0)).to(tl.int64)
    tl.store(OUT_IDS + slot * OUT + outs, out_ids)
    tl.store(OUT_PROBS + slot * OUT + outs, out_probs)
    tl.store(OUT_COUNT + slot, count)
    tl.store(DRAFT_IDS + slot, token)
    if FEED:
        tl.store(NEXT_INPUT, token)
        tl.store(NEXT_POSITION, tl.load(BASES) + slot)
        tl.store(NEXT_SEQLEN, (tl.load(BASES + 1) + slot).to(NEXT_SEQLEN.dtype.element_ty))


def supported(device, vocab):
    """Mapped RAM and the PTX fence: CUDA only. Larger vocabularies do not fit the kernels' tiles."""
    return device.type == "cuda" and mapped.supported() and vocab <= MAX_VOCAB


class DraftChain:
    """The mapped RAM and the scratch of a runner's chains."""
    def __init__(self, device, vocab, target_vocab, max_drafts):
        self.vocab, self.target_vocab, self.max_drafts, self.blocks = vocab, target_vocab, max_drafts, -(-vocab // BLOCK)
        sizes = (("context", 3, torch.int64), ("draft_ids", max_drafts, torch.int64), ("ids", max_drafts * MAX_OUT, torch.int32), ("counts", max_drafts, torch.int32), ("stamps", max_drafts, torch.int32), ("probs", max_drafts * MAX_OUT, torch.float32))
        host, gpu = mapped.empty(sum(count * torch.empty(0, dtype=dtype, device="cpu").element_size() for _, count, dtype in sizes), device, "lm")
        offset, views = 0, {}
        for name, count, dtype in sizes:
            nbytes = count * torch.empty(0, dtype=dtype, device="cpu").element_size()
            views[name] = host[offset:offset + nbytes].view(dtype).numpy(), gpu[offset:offset + nbytes].view(dtype)
            offset += nbytes
        self._host_memory, self._gpu_memory = host, gpu  # hold the mapping
        self.context, self.context_gpu = views["context"]
        self.draft_ids, self.draft_ids_gpu = views["draft_ids"]
        ids, self.ids_gpu = views["ids"]
        probs, self.probs_gpu = views["probs"]
        self.ids, self.probs = ids.reshape(max_drafts, MAX_OUT), probs.reshape(max_drafts, MAX_OUT)
        self.counts, self.counts_gpu = views["counts"]
        self.stamps, self.stamps_gpu = views["stamps"]
        self.slots = [(self.ids_gpu[slot * MAX_OUT:], self.probs_gpu[slot * MAX_OUT:], self.counts_gpu[slot:], self.stamps_gpu[slot:]) for slot in range(max_drafts)]
        self.stamps[:] = 0
        self.stamp = 0
        self.values = torch.empty(self.blocks * KEEP, dtype=torch.float32, device=device)
        self.block_ids = torch.empty(self.blocks * KEEP, dtype=torch.int32, device=device)
        self.sums = torch.empty(self.blocks, dtype=torch.float32, device=device)
        self.q = torch.zeros(max_drafts, target_vocab, dtype=torch.float32, device=device)
        self.old_ids = torch.zeros(max_drafts, MAX_OUT, dtype=torch.int32, device=device)
        self.old_counts = torch.zeros(max_drafts, dtype=torch.int32, device=device)
        self.rng, self.rng_seed = None, None
        # the GPU chain (sampling without a confidence exit): its proposals in device memory, its CUDA graphs per drafter graph and law
        self.device_ids = torch.zeros(max_drafts, MAX_OUT, dtype=torch.int32, device=device)
        self.device_probs = torch.zeros(max_drafts, MAX_OUT, dtype=torch.float32, device=device)
        self.device_counts = torch.zeros(max_drafts, dtype=torch.int32, device=device)
        self.device_draft_ids = torch.zeros(max_drafts, dtype=torch.int64, device=device)
        self.graphs = {}

    def top(self, logits, slot, temperature=1.0, top_p=1.0):
        """Queues the kernels and waits for the words of this proposal in mapped RAM: the number of its tokens."""
        self.stamp = stamp = self.stamp % 2_000_000_000 + 1
        ids, probs, count, stamps = self.slots[slot]
        _block_top[(self.blocks,)](logits, self.values, self.block_ids, self.sums, temperature, self.vocab, BLOCK, KEEP, num_warps=2)
        _select[(1,)](self.values, self.block_ids, self.sums, ids, probs, count, stamps, temperature, top_p, stamp, self.blocks, KEEP, CANDIDATES, MAX_OUT, num_warps=8)
        words, spins = self.stamps, 0
        while words[slot] != stamp:
            spins += 1
            if not spins & 8191 and torch.cuda.current_stream().query() and words[slot] != stamp:
                raise RuntimeError("The stream finished without the draft's proposal")
        return int(self.counts[slot])

    def draw(self, slot, count, generator):
        """A token of the proposal, its probabilities normalized in place (what load() gives the acceptance)."""
        seed = generator.initial_seed()
        if self.rng_seed != seed:
            self.rng, self.rng_seed = random.Random(seed), seed
        values = self.probs[slot, :count]
        values /= values.sum()
        return int(self.ids[slot, min(int(np.searchsorted(np.cumsum(values), self.rng.random())), count - 1)])

    def feed(self, token, position, seqlen, graph_vars, source):
        context = self.context
        context[0], context[1], context[2] = token, position, seqlen
        hidden = graph_vars["hidden_states"]
        _feed[(-(-hidden.numel() // HIDDEN_TILE),)](self.context_gpu, graph_vars["input_ids"], graph_vars["positions"], self.seqlens, source, hidden, hidden.numel(), HIDDEN_TILE, num_warps=2)

    def load(self, rows):
        _load_q[(rows,)](self.q, self.ids_gpu, self.probs_gpu, self.counts_gpu, self.old_ids, self.old_counts, self.target_vocab, MAX_OUT, num_warps=1)

    def select_draw(self, logits, slot, temperature, top_p, uniforms, bases, next_input, next_position, next_seqlen, feed):
        """A draft of the GPU chain: no host wait (see _select_draw)."""
        _block_top[(self.blocks,)](logits, self.values, self.block_ids, self.sums, temperature, self.vocab, BLOCK, KEEP, num_warps=2)
        _select_draw[(1,)](self.values, self.block_ids, self.sums, self.device_ids, self.device_probs, self.device_counts, self.device_draft_ids, uniforms, bases,
                           next_input, next_position, next_seqlen, temperature, top_p, slot, self.blocks, KEEP, CANDIDATES, MAX_OUT, bool(feed), num_warps=8)

    def load_device(self, rows):
        _load_q[(rows,)](self.q, self.device_ids, self.device_probs, self.device_counts, self.old_ids, self.old_counts, self.target_vocab, MAX_OUT, num_warps=1)


class CompactDrafts:
    """The draft distributions of a window: entries of the chain's proposals, dense rows only when asked (the reference sampler)."""
    def __init__(self, chain, count):
        self.chain, self.count = chain, count

    def __len__(self):
        return self.count

    def _dense(self, row):
        chain = self.chain
        count = int(chain.counts[row])
        q = torch.zeros(chain.target_vocab, dtype=torch.float32, device=chain.q.device)
        q[torch.from_numpy(chain.ids[row, :count].copy()).long().to(q.device)] = torch.from_numpy(chain.probs[row, :count].copy()).to(q.device)
        return q

    def __getitem__(self, index):
        return [self._dense(row) for row in range(*index.indices(self.count))] if isinstance(index, slice) else self._dense(index)

    def load(self):
        self.chain.load(self.count)


class DeviceDrafts(CompactDrafts):
    """The draft distributions of a GPU chain: its proposals stay in device memory (read back only for the reference sampler)."""
    def _dense(self, row):
        chain = self.chain
        count = int(chain.device_counts[row])
        q = torch.zeros(chain.target_vocab, dtype=torch.float32, device=chain.q.device)
        q[chain.device_ids[row, :count].long()] = chain.device_probs[row, :count]
        return q

    def load(self):
        self.chain.load_device(self.count)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
