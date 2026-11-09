# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A hashed n-gram embedding table read from its GGUF file, with a few rows read per token: only the rows a pass asks for, read past the OS cache, grouped by
page, and a small cache keeps the rows in use (shared.kernels.cpu_linear, rows.inc). Each token's rows give its embedding: heads_per_ngram rows per n-gram
order, in float32."""
import ctypes
import os
import threading

import torch

from mmgp import mapped, pinned
from mmgp.expert_compute import DOORBELL, launch_host_job
from shared.kernels import cpu_linear

if DOORBELL:
    from shared.kernels import host_doorbell

ROW_CACHE_BYTES = 512 << 20
QUEUED_ROWS = 16  # decoding and verification passes: a host function in the stream (and in the graphs); larger passes run eagerly


class NgramJob(ctypes.Structure):
    """rows.inc NgramJob"""
    _fields_ = [("table", ctypes.c_void_p), ("context", ctypes.c_void_p), ("out", ctypes.c_void_p), ("multipliers", ctypes.c_void_p), ("offsets", ctypes.c_void_p), ("vocab_sizes", ctypes.c_void_p),
                ("tokens", ctypes.c_int32), ("ngram_size", ctypes.c_int32), ("heads_per_ngram", ctypes.c_int32), ("values", ctypes.c_int32), ("status", ctypes.c_int32), ("reserved", ctypes.c_int32)]


class NgramTable:
    """path: the file; offset, rows, row_bytes: the IQ4_NL table in it; values: floats per row. multipliers, head_offsets, vocab_sizes: the hash."""
    def __init__(self, path, offset, rows, row_bytes, values, ngram_size, heads_per_ngram, multipliers, head_offsets, vocab_sizes, cache_bytes=ROW_CACHE_BYTES):
        self.library = cpu_linear.library()
        # locked: under memory pressure a hit stays a hit, instead of a read of the page file. Refused by the reserved RAM: ordinary RAM
        self.cache = pinned.empty(cache_bytes, "n-gram rows") if cache_bytes else None
        encoded = os.fsencode(os.fspath(path)) if os.name != "nt" else os.fspath(path).encode("utf-8")
        self.table = self.library.row_table_open(encoded, offset, row_bytes, rows, cache_bytes, None if self.cache is None else self.cache.data_ptr())
        if not self.table:
            raise FileNotFoundError(f"The n-gram table could not be opened: {path}")
        self.values, self.ngram_size, self.heads_per_ngram = values, ngram_size, heads_per_ngram
        self.heads = (ngram_size - 1) * heads_per_ngram
        self.width = self.heads * values
        self.multipliers = (ctypes.c_uint64 * len(multipliers))(*multipliers)
        self.head_offsets = (ctypes.c_uint64 * len(head_offsets))(*head_offsets)
        self.vocab_sizes = (ctypes.c_uint64 * len(vocab_sizes))(*vocab_sizes)
        self._jobs = {}  # tokens -> (job, pinned context, pinned result), at fixed addresses for graph replays
        self._ahead = None  # the thread reading the rows of the next pass of a prompt
        self._doorbell_jobs = {}  # tokens -> (job, its number, CUDA views of its mapped context and result, the host view of their RAM)
        self._prefetch_job = self._prefetch_context = None

    def close(self):
        if self.table:
            if self._ahead is not None:
                self._ahead.join()
            self.release_jobs()
            self.library.row_table_close(self.table)
            self.table = None
            if self.cache is not None:
                pinned.retire(self.cache)  # unlocked at the next quiet point
                self.cache = None

    def __del__(self):
        self.close()

    def release_jobs(self):
        if self._doorbell_jobs:
            torch.cuda.synchronize()
            for _, ticket, inputs, result, host in self._doorbell_jobs.values():
                host_doorbell.release(ticket)
                pinned.retire((inputs, result, host))  # unlocked at the next quiet point
            self._doorbell_jobs.clear()
        if self._jobs:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # a queued host job may still read or write its buffers
            for _, context, result in self._jobs.values():
                pinned.retire(context)
                pinned.retire(result)
            self._jobs.clear()

    def stats(self):
        out = (ctypes.c_uint64 * 6)()
        self.library.row_table_stats(self.table, out)
        return dict(zip(("lookups", "hits", "reads", "bytes", "read_us", "calls"), out))

    def _job(self, context, out, tokens):
        return NgramJob(self.table, context, out, ctypes.addressof(self.multipliers), ctypes.addressof(self.head_offsets), ctypes.addressof(self.vocab_sizes), tokens, self.ngram_size, self.heads_per_ngram, self.values, 0, 0)

    def read_ahead(self, context):
        """The rows of the next pass of a prompt (context: a CPU tensor, int32 [tokens, ngram_size]) into the row cache, on a thread: on a new prompt the drive is the slowest part, and its reads overlap the GPU's work on the pass before."""
        def read(previous):
            if previous is not None:
                previous.join()
            scratch = torch.empty(context.shape[0], self.width, dtype=torch.float32, device="cpu")
            self.library.ngram_rows_forward(ctypes.byref(self._job(context.data_ptr(), scratch.data_ptr(), context.shape[0])))

        self._ahead = threading.Thread(target=read, args=(self._ahead,), name="ngram_read_ahead", daemon=True)
        self._ahead.start()

    def prefetch(self, context, tokens):
        """context: int [tokens * ngram_size], as queue_embed's: their rows start reaching the cache now, on another thread. A pass asking for them meanwhile waits for what remains only."""
        job = self._prefetch_job
        if job is None:
            job = self._prefetch_job = self._job(None, None, 0)
        buffer = self._prefetch_context
        if buffer is None or len(buffer) < len(context):
            buffer = self._prefetch_context = (ctypes.c_int32 * max(len(context), 64))()
        buffer[:len(context)] = context
        job.context, job.tokens = ctypes.addressof(buffer), tokens
        self.library.ngram_rows_prefetch(ctypes.byref(job))

    def embed(self, context, device):
        """context: int32 [tokens, ngram_size] on any device. float32 [tokens, heads * values] on device, read now (the host waits for the context)."""
        tokens = context.shape[0]
        host = context.to("cpu").contiguous()
        out = torch.empty(tokens, self.width, dtype=torch.float32, device="cpu")
        job = self._job(host.data_ptr(), out.data_ptr(), tokens)
        status = self.library.ngram_rows_forward(ctypes.byref(job))
        if status:
            raise OSError(status, "Reading the n-gram table failed")
        return out.to(device)

    def queue_embed(self, context):
        """As embed() on the current CUDA stream, graph-safe: the context reaches pinned memory, a host function reads the rows into pinned memory, the result is
        copied back. Failures are reported by raise_job_errors() once the stream has been synchronized."""
        tokens = context.shape[0]
        if DOORBELL and host_doorbell.supported(context.device):
            if tokens not in self._doorbell_jobs:
                count, offset = tokens * self.ngram_size, tokens * self.ngram_size * 4
                offset = -(-offset // 64) * 64
                host, view = mapped.empty(offset + tokens * self.width * 4, context.device, "experts")
                job = self._job(host.data_ptr(), host.data_ptr() + offset, tokens)
                ticket = host_doorbell.register(context.device, None, ctypes.cast(self.library.ngram_rows_host_job, ctypes.c_void_p).value, job)
                self._doorbell_jobs[tokens] = job, ticket, view[:count * 4].view(torch.int32), view[offset:].view(torch.float32), host
            job, ticket, inputs, result, _ = self._doorbell_jobs[tokens]
            host_doorbell.ask(context.to(torch.int32).contiguous().reshape(-1), inputs, ticket)
            output = torch.empty(tokens, self.width, dtype=torch.float32, device=context.device)
            host_doorbell.wait(ticket, result, output, False)
            return output
        if tokens not in self._jobs:
            host = pinned.empty(tokens * self.ngram_size * 4, "experts").view(torch.int32).view(tokens, self.ngram_size)
            result = pinned.empty(tokens * self.width * 4, "experts").view(torch.float32).view(tokens, self.width)
            self._jobs[tokens] = self._job(host.data_ptr(), result.data_ptr(), tokens), host, result
        job, host, result = self._jobs[tokens]
        stream = torch.cuda.current_stream()
        host.copy_(context, non_blocking=True)  # read only by the host function queued after it on this stream
        launch_host_job(ctypes.cast(self.library.ngram_rows_host_job, ctypes.c_void_p).value, job, stream)
        return torch.empty(result.shape, dtype=torch.float32, device=context.device).copy_(result, non_blocking=True)

    def raise_job_errors(self):
        for job in [entry[0] for entry in self._jobs.values()] + [entry[0] for entry in self._doorbell_jobs.values()]:
            if job.status:
                status, job.status = job.status, 0
                raise OSError(status, "Reading the n-gram table failed")
