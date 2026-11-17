# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Copies between the GPU and RAM of views with gaps, without a temporary of their size.

A view has gaps when its elements do not fill one range of memory: a range of an axis after others (the pages of a [layers, pages, ...]
cache, a window of frames of a [batch, channels, frames, ...] video), a step slice. PyTorch moves bytes between devices in one linear
transfer: a side with gaps, or two sides whose layouts differ, is first made contiguous in a temporary of the whole view on its own device
(the GPU for a GPU view, in both directions), a GPU copy of GBs for a long context's KV cache. Here a view is copied by parts: its
contiguous strips directly when they are large, otherwise chunks of at most STAGING_BYTES that PyTorch stages, so that no temporary
exceeds STAGING_BYTES whatever the layout. A whole tensor that is only permuted (no gaps) is moved by PyTorch as it is, in one transfer.

RamWriter: results that leave the GPU piece by piece (a decoder's tiles, chunks of frames) written to their RAM buffer without the
caller waiting for the GPU. A copy from the GPU to ordinary RAM blocks until the GPU has finished everything queued before it, and the
GPU then idles while the next operations are prepared. Only locked RAM receives such a copy asynchronously: each piece is queued into
one of a few buffers of locked RAM (mmgp.pinned: exact sizes, counted in the reserved RAM budget, unlocked at a quiet point) and moved
to its destination once the GPU has passed it, at a later write (without waiting unless every buffer is still pending) or at close().
The source stays referenced until then: its memory is not handed to anything else while the copy is pending. The locked buffers
cost a few tens of milliseconds to set up: for a result of one small piece per call, one blocking copy is faster."""
from collections import deque
import threading
import time

import torch

from . import pinned

STAGING_BYTES = 16 << 20  # the largest temporary of a copy: parts of a view below it are staged by PyTorch (fewer, larger transfers)


def _dense(tensor):
    """Its elements fill one range of memory, in some order of its axes: no gaps, no overlaps."""
    expected = 1
    for size, stride in sorted(((size, stride) for size, stride in zip(tensor.shape, tensor.stride()) if size != 1), key=lambda item: item[1]):
        if stride != expected:
            return False
        expected *= size
    return True


def copy_(dst, src, non_blocking=False):
    """dst.copy_(src) for tensors of the same shape, without a temporary above STAGING_BYTES."""
    size = dst.numel() * max(dst.element_size(), src.element_size())  # a temporary of either side (dtype conversions stage in the source's)
    if dst.device == src.device or size <= STAGING_BYTES or dst.stride() == src.stride() and dst.dtype == src.dtype and _dense(dst):
        return dst.copy_(src, non_blocking=non_blocking)  # no temporary, or one below STAGING_BYTES
    rows = dst.shape[0]
    step = STAGING_BYTES * rows // size  # rows of the first axis that fit in STAGING_BYTES together
    for start in range(0, rows, max(step, 1)):
        part = slice(start, start + step) if step > 1 else start  # a chunk that PyTorch stages, or a row copied by its own parts
        copy_(dst[part], src[part], non_blocking)
    return dst


def move(tensor, device, dtype=None, non_blocking=False):
    """tensor.to(device, dtype, copy=True) without a temporary above STAGING_BYTES (a dtype conversion converts the whole tensor in one
    otherwise): a tensor without gaps keeps its layout if its dtype is kept, the result is contiguous otherwise."""
    dtype = tensor.dtype if dtype is None else dtype
    if dtype == tensor.dtype and _dense(tensor):
        return tensor.to(device, copy=True, non_blocking=non_blocking)
    return copy_(torch.empty(tensor.shape, dtype=dtype, device=device), tensor, non_blocking)


_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def ram_copy_(destination, source):
    """destination.copy_(source) between CPU tensors of one dtype, on the calling thread alone. PyTorch copies large tensors with its
    pool of threads, which then spin for a while: pieces copied one after the other while the GPU works (tiles, chunks of frames)
    would keep every core busy for little work."""
    if destination.dtype != source.dtype or destination.device.type != "cpu" or source.device.type != "cpu" or destination.element_size() not in _BITS:
        return destination.copy_(source)
    bits = _BITS[destination.element_size()]  # (their bytes: NumPy has no bfloat16)
    destination.detach().view(bits).numpy()[...] = source.detach().view(bits).numpy()
    return destination


def from_ram(tensor, device):
    """tensor.to(device) for a tensor in RAM; a view with gaps is first made contiguous on the calling thread alone (ram_copy_)."""
    if tensor.device.type != "cpu" or _dense(tensor):
        return tensor.to(device)
    return ram_copy_(torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu"), tensor).to(device)


class RamWriter:
    """write(destination, source, done=None): destination (a CPU tensor, or a view of one) receives source (a CUDA tensor of the same
    shape and dtype) without the caller waiting for the GPU nor copying: a helper thread lands each piece once the GPU has passed it (one
    thread, asleep while it waits, copying on its own: PyTorch's pool of threads would spin between the pieces), then calls done(), on
    that thread (on the caller's for a blocking copy; not for a piece abandoned). close() waits for
    what is pending and retires the buffers. depth: pieces that may be pending at once; max_bytes: locked RAM they may hold (None:
    no limit but depth). The caller waits only when every buffer is pending: a writer whose caller must not wait in the middle of its
    work (the GPU runs dry while the next operations are issued) takes a large depth and bounds the bytes. Where locked RAM is
    refused (budget, driver) or source is not on a CUDA device, the copy is the ordinary blocking one. Use as a context manager: an
    exception in its block abandons what is pending."""

    POLL = 0.0005  # seconds between two looks at the GPU's progress by the helper thread

    def __init__(self, kind="decoded output", depth=4, max_bytes=None):
        self.free, self.pending, self.max_bytes, self.bytes = [pinned.Buffer(kind) for _ in range(depth)], deque(), max_bytes, 0
        self.state, self.thread, self.closing, self.abandoned, self.error = threading.Condition(), None, False, False, None

    def __enter__(self):
        return self

    def __exit__(self, kind, *exc):
        self.abandoned = kind is not None
        self.close()

    def _land(self):
        """The helper thread: the oldest pending piece once the GPU has passed it, then its buffer back to the caller."""
        while True:
            with self.state:
                while not self.pending and not self.closing:
                    self.state.wait()
                if not self.pending:
                    return
                target, staged, event, _, buffer, nbytes, done = self.pending[0]
            try:
                while not self.abandoned and not event.query():
                    time.sleep(RamWriter.POLL)  # (asleep, not spinning as a synchronization would)
                if not self.abandoned and self.error is None:
                    target[...] = staged
                    if done is not None:
                        done()
            except BaseException as error:  # reported by close()
                self.error = error
            with self.state:
                self.pending.popleft()
                self.bytes -= nbytes
                self.free.append(buffer)
                self.state.notify_all()

    def _wait(self, ready):
        with self.state:
            while self.pending and not ready():
                self.state.wait()

    def write(self, destination, source, done=None):
        nbytes = source.numel() * source.element_size()
        self._wait(lambda: self.free and (self.max_bytes is None or self.bytes + nbytes <= self.max_bytes))
        with self.state:
            buffer = self.free.pop()
        bits = _BITS.get(source.element_size())
        staging = buffer.get(nbytes) if source.is_cuda and nbytes and bits is not None and destination.dtype == source.dtype else None
        if staging is None:
            with self.state:
                self.free.append(buffer)
            self._wait(lambda: False)
            destination.copy_(source)
            if done is not None:
                done()
            return
        source = source.contiguous()
        staged = staging[:nbytes].view(source.dtype).view(source.shape)
        staged.copy_(source, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        with self.state:  # (their bytes as NumPy arrays, made here: the helper thread only copies)
            self.pending.append((destination.detach().view(bits).numpy(), staged.view(bits).numpy(), event, source, buffer, nbytes, done))
            self.bytes += nbytes
            if self.thread is None:
                self.thread = threading.Thread(target=self._land, name="RamWriter", daemon=True)
                self.thread.start()
            self.state.notify_all()

    def close(self, abandon=False):
        """Waits for the pending pieces and retires the buffers; abandon: the pieces still pending are dropped (their done() not called)."""
        self.abandoned = self.abandoned or abandon
        with self.state:
            self.closing = True
            self.state.notify_all()
        if self.thread is not None:
            self.thread.join()
        for buffer in self.free:
            buffer.release()
        if self.error is not None and not self.abandoned:
            raise self.error
