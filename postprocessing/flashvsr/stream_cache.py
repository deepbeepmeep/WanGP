# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The stream cache of FlashVSR's DiT: the keys and values (normalized and rotated) of the last windows of each block, which its next chunk
attends before its own, (tokens, heads, head_dim) per block. The blocks keeping as many windows share a state ring of MMGP
(mmgp.runtime.state_ring): in RAM with in_ram (the Minimum VRAM usage: a minimal ring, the GPU holding the state of the block running
only, as the cache did before it used a ring); otherwise, with MMGP's VRAM allocator, in VRAM as far as it fits
and in RAM beyond (resident="dynamic"), without it in RAM but for the blocks whose weights MMGP keeps in VRAM (resident=True); the blocks in
RAM are fetched ahead with the weights. Each chunk replaces a block's whole state, in the tensors the block reads: the windows it keeps from
the previous state moved to the front (in their order), then the newest windows of the chunk, written by the attention once it has read the
previous state (attention_kit kv_window); the first chunk writes into the tensors the ring reserves (no tensors of its own)."""
from mmgp import runtime


class StreamCache:
    def __init__(self, blocks, windows, window_tokens, heads, head_dim, dtype, device, in_ram=False):
        """blocks: the DiT's blocks; windows: per block, the temporal windows it keeps (window_tokens tokens each); in_ram: the whole cache in
        RAM (minimal VRAM)."""
        self.shape, self.dtype, self.device = (heads, head_dim), dtype, device
        self.rings, self.capacity, self.handles, self.lengths = {}, {}, {}, {}
        self.position = {}  # block -> (its ring, its index in the ring)
        for kept in sorted(set(windows)):
            members = [block for block, count in enumerate(windows) if count == kept]
            modules = [blocks[block] for block in members]
            resident = False if in_ram else "dynamic" if runtime.dynamic_state(modules) else True
            ring = runtime.state_ring(modules, f"FlashVSR stream cache ({kept} window{'s' if kept > 1 else ''})", resident=resident, minimal=in_ram)
            self.rings[kept] = ring
            for index, block in enumerate(members):
                self.position[block], self.capacity[block] = (ring, index), kept * window_tokens

    @property
    def storing(self):
        """Nothing stored yet: the first chunk, which stores every block's state."""
        return not self.lengths

    def prefix(self, block):
        """The keys and values (1, rows, heads, head_dim) the block attends before its chunk's own, None at the first chunk."""
        rows = self.lengths.get(block)
        return None if rows is None else tuple(tensor[:rows].unsqueeze(0) for tensor in self.handles[block])

    def window(self, block, rows):
        """The keys and values (1, rows, heads, head_dim) of the block's next state, which its attention writes (kv_window): the tensors
        holding its state (those of the prefix), or at the first chunk those the ring reserves for it."""
        if block not in self.handles:
            ring, index = self.position[block]
            self.handles[block] = ring.reserve(index, [((self.capacity[block], *self.shape), self.dtype)] * 2)
        return tuple(tensor[:rows].unsqueeze(0) for tensor in self.handles[block])

    def commit(self, block, rows):
        """During the block's pass, once its window (rows) is written: its state stored (the first chunk) or updated."""
        ring, index = self.position[block]
        if block in self.lengths:
            ring.update(index, slice(0, rows))
        else:
            ring.store(index, list(self.handles[block]), current=True, rows=slice(0, rows))
        self.lengths[block] = rows

    def release(self):
        for ring in self.rings.values():
            ring.release()
        self.rings, self.handles, self.lengths = {}, {}, {}
