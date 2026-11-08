# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
# With an idea of Strata (MIT License, https://github.com/Niko1221/Strata).
"""Host jobs inside a CUDA graph, or eagerly, without stopping the GPU: a small kernel writes the inputs of a job to mapped RAM and then its number to a word that a thread of the library spins on (cpu_linear,
host_server.inc); the GPU goes on with its own work, and a later kernel waits for the job's answer in mapped RAM and reads the result. Unlike a host function, whose successors wait for it, the CPU's
work overlaps the GPU's; no copy engine and no driver call are involved. A job is any function of the library or another: this module knows nothing of what it computes. CUDA only: the volatile accesses and
the system memory fence are PTX, and callers keep their host functions elsewhere."""
import atexit
import ctypes
import os

import torch

from mmgp import allocator, mapped, pinned
from shared.kernels import cpu_linear
from shared.kernels.dispatch import triton_dtype_supported

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

READ_WORDS = 1024  # of each of the programs reading a result (4 warps): reads of mapped RAM wait for the bus, so they go side by side

if triton is not None:
    # of the program writing the inputs (1 warp), measured: 10 KB in 14 us as tiles of 32 words, one packet per word on the bus, 2.5 us in tiles of 1024, 32 contiguous words per thread
    WORDS = tl.constexpr(1024)
    _READ_WORDS = tl.constexpr(READ_WORDS)


    @triton.jit
    def _load_word(address):
        """Read from mapped RAM by each thread of the program: not from its caches."""
        return tl.inline_asm_elementwise("ld.volatile.global.u32 $0, [$1];", "=r,l", [address], dtype=tl.int32, is_pure=False, pack=1)


    @triton.jit
    def _fence(value):
        """The stores before it have reached system memory, in order, before those after it."""
        return tl.inline_asm_elementwise("membar.sys; mov.u32 $0, $1;", "=r,r", [value], dtype=tl.int32, is_pure=False, pack=1)


    @triton.jit(do_not_specialize=("count", "ticket"), do_not_specialize_on_alignment=("source", "target", "answer", "mailbox"))
    def _ask(source, target, answer, mailbox, count, ticket):
        """Clears the job's answer, writes its inputs, then rings. A scalar tl.store is one store, not one per thread: a word written twice would run the job twice."""
        tl.store(answer, ticket * 0)
        for start in range(0, count, WORDS):
            offsets = start + tl.arange(0, WORDS)
            mask = offsets < count
            tl.store(target + offsets, tl.load(source + offsets, mask=mask), mask=mask)
        tl.store(mailbox, _fence(ticket))
        _fence(ticket)


    @triton.jit(do_not_specialize=("count", "ticket"), do_not_specialize_on_alignment=("source", "target", "answer", "mailbox"))
    def _ask_floats(source, target, answer, mailbox, count, ticket):
        """As _ask, the inputs of any float type written as float32."""
        tl.store(answer, ticket * 0)
        for start in range(0, count, WORDS):
            offsets = start + tl.arange(0, WORDS)
            mask = offsets < count
            tl.store(target + offsets, tl.load(source + offsets, mask=mask, other=0.0).to(tl.float32), mask=mask)
        tl.store(mailbox, _fence(ticket))
        _fence(ticket)


    @triton.jit(do_not_specialize=("first_count", "count", "ticket"), do_not_specialize_on_alignment=("first", "second", "third", "target", "answer", "mailbox"))
    def _ask_parts(first, second, third, target, answer, mailbox, first_count, count, ticket):
        """As _ask, the inputs in three parts, each converted on the way: float32, float32 and int32, without kernels to make them one tensor first."""
        tl.store(answer, ticket * 0)
        floats = target.to(tl.pointer_type(tl.float32))
        for start in range(0, first_count, WORDS):
            offsets = start + tl.arange(0, WORDS)
            mask = offsets < first_count
            tl.store(floats + offsets, tl.load(first + offsets, mask=mask, other=0.0).to(tl.float32), mask=mask)
        for start in range(0, count, WORDS):
            offsets = start + tl.arange(0, WORDS)
            mask = offsets < count
            tl.store(floats + first_count + offsets, tl.load(second + offsets, mask=mask, other=0.0).to(tl.float32), mask=mask)
            tl.store(target + first_count + count + offsets, tl.load(third + offsets, mask=mask, other=0).to(tl.int32), mask=mask)
        tl.store(mailbox, _fence(ticket))
        _fence(ticket)


    @triton.jit(do_not_specialize_on_alignment=("answer", "state"))
    def _wait(answer, state):
        """Spins until the job answers, leaving the answer in VRAM for _read."""
        value = _load_word(answer.to(tl.int64))
        while value == 0:
            value = _load_word(answer.to(tl.int64))
        tl.store(state, value)


    @triton.jit(do_not_specialize=("count",), do_not_specialize_on_alignment=("state", "source", "target"))
    def _read(state, source, target, count, ADD: tl.constexpr):
        """A result to read joins target (added when ADD, else copied): the reads of mapped RAM wait for the bus, so they go side by side in programs."""
        if tl.load(state) == 2:
            offsets = tl.program_id(0) * _READ_WORDS + tl.arange(0, _READ_WORDS)
            mask = offsets < count
            values = tl.load(source + offsets, mask=mask, other=0.0)
            if ADD:
                values = values + tl.load(target + offsets, mask=mask, other=0.0)
            tl.store(target + offsets, values, mask=mask)


    @triton.jit
    def _load_long(address):
        """An int64 of mapped RAM, not from the caches."""
        return tl.inline_asm_elementwise("ld.volatile.global.u64 $0, [$1];", "=l,l", [address], dtype=tl.int64, is_pure=False, pack=1)


    @triton.jit
    def _now(dummy):
        """The GPU's clock, in ns."""
        return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1)


    @triton.jit(do_not_specialize=("numel", "errors"), do_not_specialize_on_alignment=("rows", "table", "base"))
    def _read_rows(rows, table, base, numel, errors, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EARLY: tl.constexpr, COPY: tl.constexpr, TIMEOUT: tl.constexpr):
        """table (int64 [numel // WIDTH, WIDTH]) zeroed (COPY: a copy of base), then the rows a host job wrote to mapped RAM: their count, then (index, WIDTH values) each.
        EARLY: the job publishes them while it runs (the count is -1 until then), each row followed by the address of a word that a copy sets once it has landed: both
        waited for (TIMEOUT ns at most: a row then left out, counted at rows[errors]), and the count set back to -1 for the next job."""
        for start in range(0, numel, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            values = tl.load(base + offsets, mask=offsets < numel, other=0) if COPY else tl.zeros([BLOCK], dtype=tl.int64)
            tl.store(table + offsets, values, mask=offsets < numel)
        tl.debug_barrier()
        count = _load_long(rows.to(tl.int64))
        if EARLY:
            started = _now(count.to(tl.int32) * 0)
            while count < 0 and _now(count.to(tl.int32) * 0) - started < TIMEOUT:
                count = _load_long(rows.to(tl.int64))
            failed = (count < 0).to(tl.int64)
            count = tl.maximum(count, 0)
            for n in range(0, count):
                entry = rows + 1 + n * (WIDTH + 2)
                done = _load_long((entry + WIDTH + 1).to(tl.int64))
                landed = (done == 0).to(tl.int32)  # 0: read in place, nothing to wait for
                if done != 0:
                    landed = _load_word(done)
                    while landed == 0 and _now(landed * 0) - started < TIMEOUT:
                        landed = _load_word(done)
                if landed != 0:
                    index = _load_long(entry.to(tl.int64))
                    for k in tl.static_range(WIDTH):
                        tl.store(table + index * WIDTH + k, _load_long((entry + 1 + k).to(tl.int64)))
                else:
                    failed += 1
            if failed > 0:
                tl.store(rows + errors, _load_long((rows + errors).to(tl.int64)) + failed)
            tl.store(rows, count * 0 - 1)
        else:
            for n in range(0, count):
                entry = rows + 1 + n * (WIDTH + 1)
                index = _load_long(entry.to(tl.int64))
                for k in tl.static_range(WIDTH):
                    tl.store(table + index * WIDTH + k, _load_long((entry + 1 + k).to(tl.int64)))


    @triton.jit(do_not_specialize=("index",), do_not_specialize_on_alignment=("words",))
    def _mark(words, index):
        tl.store(words + index, index * 0 + 1)


class _Mailbox:
    """The words of the server, one set per process."""
    def __init__(self, device):
        self.library = cpu_linear.library()
        self.capacity = self.library.host_server_capacity()
        self.host, self.gpu = mapped.empty(64 + self.capacity * 4, device, "host jobs")
        self.host.zero_()
        self.library.host_server_open.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.host_server_open.restype = None
        self.library.host_server_close.restype = None
        self.library.host_server_set_call.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.library.host_server_set_call.restype = None
        self.library.host_server_open(self.host.data_ptr(), self.host.data_ptr() + 64)
        self.free = list(range(self.capacity - 1, -1, -1))
        self.views = {}  # the answer word of each job, as a kernel argument: nothing is computed from these tensors during a graph capture
        words = self.gpu.view(torch.int32)
        self.mailbox, self.answers = words[0], words[16:]
        with allocator.normal_allocations():  # read by graphs, not graph scratch
            self.state = torch.zeros(1, dtype=torch.int32, device=device)

    def close(self):
        self.library.host_server_close()
        pinned.retire((self.host, self.gpu, self.mailbox, self.answers))  # unlocked at the next quiet point
        self.host = self.gpu = self.mailbox = self.answers = self.state = None


_mailbox = None
_preemption = {}
_COMPUTE_PREEMPTION = 90  # CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED


def _close_at_exit():
    global _mailbox
    if _mailbox is not None:  # the thread stops reading the words before their RAM goes
        _mailbox.close()
        _mailbox = None


atexit.register(_close_at_exit)


def _preemptible(index):
    """A kernel may wait for a job for seconds: only where the system can preempt it (no driver reset, measured here to 6 s beyond a 2 s limit)."""
    if index not in _preemption:
        driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        handle, value = ctypes.c_int(), ctypes.c_int()
        _preemption[index] = driver.cuDeviceGet(ctypes.byref(handle), index) == 0 and driver.cuDeviceGetAttribute(ctypes.byref(value), _COMPUTE_PREEMPTION, handle) == 0 and value.value == 1
    return _preemption[index]


def supported(device):
    """Whether this device runs these kernels: CUDA with compute preemption, and the library serving their jobs."""
    return triton is not None and device.type == "cuda" and torch.version.hip is None and mapped.supported() and cpu_linear.library() is not None and _preemptible(device.index or 0)


def register(device, function, void_function, job):
    """A number for a job (a ctypes structure kept alive by the caller until release): function(job) returns nonzero when it wrote a result to read, void_function(job) always wrote one."""
    global _mailbox
    if _mailbox is None:
        _mailbox = _Mailbox(device)
    ticket = _mailbox.free.pop()
    _mailbox.library.host_server_set_call(ticket, function, void_function, ctypes.addressof(job))
    _mailbox.views[ticket] = _mailbox.answers[ticket]
    return ticket


def release(ticket):
    """Nothing in flight may ask for this job any more. The server stops with the last job."""
    global _mailbox
    if _mailbox is None:  # the process is leaving
        return
    _mailbox.free.append(ticket)
    del _mailbox.views[ticket]
    if len(_mailbox.free) == _mailbox.capacity and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
        _mailbox.close()
        _mailbox = None


def ask(inputs, target, ticket):
    """Graph-safe, on the current stream: int32 inputs [count] to the job's mapped input buffer (target, a CUDA view), then its number to the server."""
    assert inputs.is_cuda and target.is_cuda and inputs.dtype == target.dtype == torch.int32 and inputs.is_contiguous()
    _ask[(1,)](inputs, target, _mailbox.views[ticket], _mailbox.mailbox, inputs.numel(), ticket + 1, num_warps=1)


def ask_floats(inputs, target, ticket):
    """As ask() for inputs of any float type (parts_supported) as float32."""
    assert inputs.is_cuda and target.is_cuda and target.dtype == torch.float32 and inputs.is_contiguous()
    _ask_floats[(1,)](inputs, target, _mailbox.views[ticket], _mailbox.mailbox, inputs.numel(), ticket + 1, num_warps=1)


def ask_parts(first, second, third, target, ticket):
    """As ask() for inputs in three contiguous parts: first (any float type) as float32, second as float32, third as int32, one after the other in target. second and third have the same size."""
    assert first.is_cuda and second.is_cuda and third.is_cuda and target.is_cuda and target.dtype == torch.int32 and second.numel() == third.numel() and first.is_contiguous() and second.is_contiguous() and third.is_contiguous()
    _ask_parts[(1,)](first, second, third, target, _mailbox.views[ticket], _mailbox.mailbox, first.numel(), third.numel(), ticket + 1, num_warps=1)


def parts_supported(tensor):
    """Whether ask_parts() reads this float type on its device."""
    return triton_dtype_supported(tensor.device, tensor.dtype)


def wait(ticket, result, output, add):
    """Graph-safe, on the current stream: waits for the job, then adds or copies its float32 result (a CUDA view of mapped RAM) into output if it wrote one."""
    assert result.is_cuda and output.is_cuda and result.dtype == output.dtype == torch.float32 and result.is_contiguous() and output.is_contiguous()
    count = output.numel()
    _wait[(1,)](_mailbox.views[ticket], _mailbox.state, num_warps=1)
    _read[((count + READ_WORDS - 1) // READ_WORDS,)](_mailbox.state, result, output, count, ADD=bool(add), num_warps=4)
    torch.autograd.graph.increment_version(output)


ROWS_TIMEOUT = 2_000_000_000  # ns a row of read_rows(early=True) is waited for: a copy that never lands (its helper gone) leaves the row out instead of stopping the GPU


def mark(words, index):
    """Graph-safe, on the current stream: words[index] = 1 (an int64 CUDA view of mapped RAM), once the work queued before has run: a host job sees it."""
    _mark[(1,)](words, index, num_warps=1)


def read_rows(rows, table, early=False, errors=0, base=None):
    """Graph-safe, on the current stream: table (int64 [entries, width] in VRAM) zeroed (or a copy of base, of its shape), then the rows a job wrote to rows (an int64 CUDA view of
    mapped RAM: their count, then (index, width values) each), after wait() of the job. early: before wait(), the job publishing them while it runs, each row followed by the address
    of a word set once what it points to has landed (both waited for; rows[errors] counts the rows left out after ROWS_TIMEOUT)."""
    assert rows.is_cuda and table.is_cuda and rows.dtype == table.dtype == torch.int64 and table.is_contiguous() and table.dim() == 2
    assert base is None or base.is_cuda and base.dtype == torch.int64 and base.is_contiguous() and base.shape == table.shape
    _read_rows[(1,)](rows, table, table if base is None else base, table.numel(), errors, WIDTH=table.shape[1], BLOCK=1024, EARLY=bool(early), COPY=base is not None, TIMEOUT=ROWS_TIMEOUT,
                     num_warps=4)
    torch.autograd.graph.increment_version(table)


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger

    install_triton_compilation_logger()
