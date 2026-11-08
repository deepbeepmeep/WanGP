# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""CPU products of quantized linear weights (linear.cpp): the linears of the blocks that MMGP leaves on the CPU (cpuCompute) compute in the
weights' own storage (mmap or pinned), with GGUF (Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL, Q2_0, Prism PTQ1_0), INT8
(Quanto qint8, INT8 ConvRot), scaled FP8 (e4m3, e5m2) and NVFP4 weights. The modules apply their own input transforms first (ConvRot and Prism
rotations, NVFP4 pre-quantization scale). Activations cross pinned buffers that GPU kernels read and write in place. The library also holds the
routed experts in GGUF storage (experts_gguf.inc) and the reader of table rows from a file (rows.inc). Importing or using this module never
invokes a compiler: python -m shared.kernels.cpu_linear.build rebuilds the libraries."""
import ctypes
import os
import platform
import weakref
from pathlib import Path

import torch

from mmgp import cuda_graphs, mapped
from mmgp.expert_compute import cpu_threads

FORMAT_GGUF, FORMAT_INT8_ROWS, FORMAT_FP8_ROWS, FORMAT_NVFP4 = 0, 1, 2, 3  # linear.cpp formats
FP8_E4M3, FP8_E5M2 = 0, 1
NVFP4_LEGACY, NVFP4_TENSORCORE = 0, 1

CPU_DOORBELL = "cpu_doorbell" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # CPU blocks replayed in CUDA graphs, proven: the switch goes with the others at the end
_library = None
_attempted = False
_buffers = {}
_job_buffers = {}  # bytes -> (host view, CUDA view) of mapped RAM shared by the jobs of linears, which run one at a time
JOB_BUFFER_BYTES = 1 << 20
_jobs_supported = {}


class Weight(ctypes.Structure):
    """One weight of a call (linear.cpp Weight)."""
    _fields_ = [("data", ctypes.c_void_p), ("scales", ctypes.c_void_p), ("out", ctypes.c_void_p), ("n", ctypes.c_int64), ("format", ctypes.c_int32),
                ("type", ctypes.c_int32), ("scale", ctypes.c_float), ("reserved", ctypes.c_int32)]


def library_path():
    return Path(__file__).with_name("cpu_linear_" + ("win_amd64.dll" if os.name == "nt" else "linux_x86_64.so"))


def library():
    """The native library, or None where it is not built for this platform."""
    global _library, _attempted
    if not _attempted:
        _attempted = True
        path = library_path()
        if platform.machine().lower() in ("amd64", "x86_64") and path.is_file():
            lib = ctypes.CDLL(str(path))
            lib.cpu_linear_supported.argtypes = [ctypes.c_int, ctypes.c_int]
            lib.cpu_linear_supported.restype = ctypes.c_int
            lib.cpu_linear.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.POINTER(Weight), ctypes.c_int]
            lib.cpu_linear.restype = ctypes.c_int
            lib.gguf_moe_forward.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 5 + [ctypes.c_int] * 2
            lib.gguf_moe_forward.restype = ctypes.c_int64
            lib.gguf_moe_host_job.argtypes = [ctypes.c_void_p]
            lib.gguf_moe_host_job.restype = None
            lib.row_table_open.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]
            lib.row_table_open.restype = ctypes.c_void_p
            lib.row_table_close.argtypes = [ctypes.c_void_p]
            lib.row_table_close.restype = None
            lib.row_table_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            lib.row_table_stats.restype = None
            lib.row_table_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
            lib.row_table_read.restype = ctypes.c_int
            lib.ngram_rows_forward.argtypes = [ctypes.c_void_p]
            lib.ngram_rows_forward.restype = ctypes.c_int
            lib.ngram_rows_host_job.argtypes = [ctypes.c_void_p]
            lib.ngram_rows_host_job.restype = None
            lib.ngram_rows_prefetch.argtypes = [ctypes.c_void_p]
            lib.ngram_rows_prefetch.restype = None
            _library = lib
    return _library


def moe_stats():
    """What the host jobs of routed calls in GGUF storage have done so far."""
    out = (ctypes.c_uint64 * 3)()
    library().gguf_moe_stats(out)
    return dict(zip(("jobs", "pairs", "microseconds"), out))


SHARE_ROWS = 64  # experts of a job given to the GPU at most (experts_gguf.inc MOE_SHARE_MAX)


class MoeShare(ctypes.Structure):
    """The GPU's share of the experts of a bank that miss VRAM, and its experts in locked RAM (experts_gguf.inc MoeShare): read by the host jobs while they run."""
    _fields_ = [("sources", ctypes.c_void_p), ("offsets", ctypes.c_uint64 * 3), ("slot_bytes", ctypes.c_uint64), ("ring", ctypes.c_void_p), ("fraction", ctypes.c_float)] + \
               [(name, ctypes.c_int32) for name in ("experts", "status", "pad")] + [(name, ctypes.c_uint64) for name in ("asked", "landed", "missing", "jobs")] + [("remote_sources", ctypes.c_void_p), ("device_sources", ctypes.c_void_p),
               ("live", ctypes.c_float), ("pad2", ctypes.c_int32)]


def moe_share_supported():
    """Whether the host jobs can give the GPU a share (the CUDA driver found)."""
    return bool(library().gguf_moe_share_supported())


REMOTE_COPIES = os.name == "nt" and "share_thread" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # the copies of the share asked of a helper process (Windows: a copy call of this process waits for its page faults)
_HELPER = """import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.gguf_moe_copy_helper.argtypes = [ctypes.c_uint32] + [ctypes.c_uint64] * 4 + [ctypes.c_int32]
sys.exit(lib.gguf_moe_copy_helper(*[int(value) for value in sys.argv[2:8]]))"""
_FILL_HELPER = """import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.gguf_moe_fill_helper.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
sys.exit(lib.gguf_moe_fill_helper(int(sys.argv[2]), int(sys.argv[3])))"""


class MoeRemoteMove(ctypes.Structure):
    _fields_ = [("file", ctypes.c_uint64), ("source", ctypes.c_uint64), ("offset", ctypes.c_uint64), ("bytes", ctypes.c_uint64)]


class MoeRemoteFill(ctypes.Structure):  # experts_gguf.inc
    _fields_ = [("section", ctypes.c_uint64), ("target", ctypes.c_uint64), ("moves", MoeRemoteMove * 3), ("count", ctypes.c_int32), ("drop", ctypes.c_int32), ("result", ctypes.c_int32),
                ("pad", ctypes.c_int32)]


class ShareRing:
    """The VRAM slots of the GPU's share of the experts out of VRAM (experts_gguf.inc MoeRing), and what asks the copy engine to fill them: on Windows a helper
    process of its own (the driver's calls of this process wait while its page faults hold its memory, up to hundreds of ms when RAM is short), elsewhere a
    thread of the library. handle: for MoeShare.ring; base: the device address of the first slot; remote: copies of sections (MoeShare.remote_sources)."""
    def __init__(self, slot_bytes, slots, device):
        lib, self.remote, self.helper, self.filler, self.handle = library(), REMOTE_COPIES, None, None, None
        index = torch.device(device).index or 0
        if self.remote:
            lib.gguf_moe_ring_open_remote.argtypes = [ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
            lib.gguf_moe_ring_open_remote.restype = ctypes.c_void_p
            lib.gguf_moe_ring_remote_state.argtypes = lib.gguf_moe_ring_remote_stop.argtypes = lib.gguf_moe_ring_fill_state.argtypes = [ctypes.c_void_p]
            lib.gguf_moe_ring_forget.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32]
            lib.gguf_moe_ring_fill.argtypes, lib.gguf_moe_ring_fill.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32], ctypes.c_int64
            lib.gguf_moe_ring_filled.argtypes, lib.gguf_moe_ring_filled.restype = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int64
            out = (ctypes.c_uint64 * 5)()
            self.handle = lib.gguf_moe_ring_open_remote(slot_bytes, slots, index, out)
            if not self.handle:
                raise RuntimeError("The VRAM of the GPU's share of the experts could not be made")
            self.base = out[0]
            import subprocess
            import sys
            import time
            self.helper = subprocess.Popen([sys.executable, "-I", "-c", _HELPER, str(library_path()), str(os.getpid()), *(str(value) for value in out[1:5]), str(index)],
                                           creationflags=0x08000000 | 0x00008000)  # CREATE_NO_WINDOW, ABOVE_NORMAL_PRIORITY_CLASS: it wakes up before the CPU's workers
            # the fills of level 2 in a process of their own, which makes no driver call: its page faults hold none (MoeRemoteFill)
            self.filler = subprocess.Popen([sys.executable, "-I", "-c", _FILL_HELPER, str(library_path()), str(os.getpid()), str(out[1])],
                                           creationflags=0x08000000 | 0x00004000)  # CREATE_NO_WINDOW, BELOW_NORMAL_PRIORITY_CLASS
            deadline = time.monotonic() + 60
            for name, process, state in (("copy", self.helper, lib.gguf_moe_ring_remote_state), ("fill", self.filler, lib.gguf_moe_ring_fill_state)):
                while state(self.handle) != 1:
                    if process.poll() is not None or time.monotonic() > deadline:
                        code = process.poll()
                        self.close()
                        raise RuntimeError(f"The {name} helper of the GPU's share of the experts did not start ({code})")
                    time.sleep(0.01)
        else:
            from mmgp import allocator, pinned
            lib.gguf_moe_ring_open.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
            lib.gguf_moe_ring_open.restype = ctypes.c_void_p
            with torch.inference_mode(False):
                self.done = pinned.empty(4 * slots, "experts")
                with allocator.normal_allocations():  # read by the graphs, not graph scratch
                    self.ring = torch.empty(slots * slot_bytes, dtype=torch.uint8, device=device)
                    self.one = torch.ones(16, dtype=torch.int32, device=device)
            context = ctypes.c_void_p()
            ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1").cuCtxGetCurrent(ctypes.byref(context))
            self.handle = None if self.done is None else lib.gguf_moe_ring_open(self.ring.data_ptr(), slot_bytes, slots, context, self.done.data_ptr(), self.one.data_ptr())
            if not self.handle:
                raise RuntimeError("The copy thread of the GPU's share of the experts could not start")
            self.base = self.ring.data_ptr()

    def pause(self, paused):
        """While paused, the jobs ask for no copies (the placement's own copies hold the copy engine)."""
        lib = library()
        lib.gguf_moe_ring_pause.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.gguf_moe_ring_pause(self.handle, int(paused))

    def prepare(self, sections):
        """The sections (handles) of experts' copies the helper maps ahead of their first copies. A thread's ring needs nothing."""
        if self.remote and len(sections):
            import numpy as np
            sections = np.ascontiguousarray(sections, dtype=np.uint64)
            lib = library()
            lib.gguf_moe_ring_prepare.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
            lib.gguf_moe_ring_prepare.restype = ctypes.c_int32
            lib.gguf_moe_ring_prepare(self.handle, sections.ctypes.data, len(sections))

    def fill(self, fills):
        """Copies of experts from their checkpoint files into slots of level 2, made by the helper: (section, target, [(file, source, offset, bytes)], drop) each (see MoeRemoteFill).
        The sequence number of the first, None when its queue lacks room for them all."""
        array = (MoeRemoteFill * len(fills))()
        for record, (section, target, moves, drop) in zip(array, fills):
            record.section, record.target, record.count, record.drop = section, target, len(moves), int(drop)
            for move, values in zip(record.moves, moves):
                move.file, move.source, move.offset, move.bytes = values
        first = library().gguf_moe_ring_fill(self.handle, array, len(fills))
        return None if first < 0 else first

    def filled(self, first, count):
        """The fills done so far (the sequence number after the last), the results of those of [first, first + count) done (0: copied), and the helper's (seconds, bytes) of fills."""
        results, stats = (ctypes.c_int32 * max(count, 1))(), (ctypes.c_uint64 * 2)()
        done = library().gguf_moe_ring_filled(self.handle, first, count, results, stats)
        return done, list(results[:max(0, min(count, done - first))]), (stats[0] / 1e9, stats[1])

    def forget(self, section):
        """A section of experts' copies leaves: the helper lets it go first. False if it did not answer."""
        return not self.remote or self.helper is None or bool(library().gguf_moe_ring_forget(self.handle, section, 5000))

    def stats(self):
        return moe_ring_stats(self.handle)

    def close(self):
        """Once nothing in flight asks for copies."""
        lib = library()
        lib.gguf_moe_ring_close.argtypes = [ctypes.c_void_p]
        lib.gguf_moe_ring_close.restype = None
        if self.handle and self.remote and self.helper is not None:
            import subprocess
            lib.gguf_moe_ring_remote_stop(self.handle)  # both helpers leave
            for process in (self.helper, self.filler):
                if process is None:
                    continue
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if self.handle:
            lib.gguf_moe_ring_close(self.handle)
        if not self.remote and getattr(self, "done", None) is not None:
            from mmgp import pinned
            pinned.retire(self.done)
        self.handle = self.helper = self.filler = None


def moe_ring_stats(handle):
    """copies issued, the copy thread's time issuing them, first failure, the copies' time in its queue, jobs with copies and their CPU parts' time."""
    out = (ctypes.c_int64 * 16)()
    library().gguf_moe_ring_stats(ctypes.c_void_p(handle), out)
    names = ("issued", "issue_microseconds", "status", "wait_nanoseconds", "parts", "part_nanoseconds", "call_nanoseconds", "call_max_nanoseconds", "wakes", "wake_nanoseconds")
    return dict(zip(names, out), calls_under=dict(zip(("50us", "200us", "1ms", "10ms", "more"), out[10:15])), fraction=out[15] / 1000)


class MoeJob(ctypes.Structure):
    """One routed call in GGUF storage, run by a CUDA host function (experts_gguf.inc MoeJob): the addresses stay fixed while graphs replay it."""
    _fields_ = [(name, ctypes.c_void_p) for name in ("gate", "up", "down", "payload", "resident", "usage", "out")] + \
               [(name, ctypes.c_int32) for name in ("gate_type", "up_type", "down_type", "tokens", "h", "i", "experts", "topk", "threads", "tail_rank", "status")] + \
               [("share", ctypes.POINTER(MoeShare)), ("share_rows", ctypes.c_void_p)]


def moe_host_function():
    return ctypes.cast(library().gguf_moe_host_job, ctypes.c_void_p).value


def moe_server_function():
    """The job function of the host server (host_server.inc, shared.kernels.host_doorbell): returns whether it wrote a result."""
    return ctypes.cast(library().gguf_moe_server_job, ctypes.c_void_p).value


def moe_forward(gate, up, down, types, x, ids, probabilities, resident, out, tokens, h, i, experts, topk, threads):
    """Addresses of contiguous CPU buffers: the bank's rows, float32 x/probabilities/out, int32 ids and a uint8 resident flag per expert (or 0).
    Returns how many pairs were computed (out is left unwritten when there are none)."""
    count = library().gguf_moe_forward(gate, up, down, *types, x, probabilities, ids, resident, out, tokens, h, i, experts, topk, max(1, threads), 0)
    if count < 0:
        raise RuntimeError(f"CPU GGUF expert kernel failed ({count}), types {types}")
    return count


def _float_rows(scale, n):
    """A per-row scale as contiguous float32 [n] (kept with the weight), or a per-tensor scale as a float; (False, None) otherwise."""
    if scale.numel() == 1:
        return None, float(scale.float().item())
    if scale.numel() != n:
        return False, None
    return scale.reshape(n).to(torch.float32).contiguous(), 1.0


def _describe(weight):
    """(format, type, data, scales, scale, out_features) of a CPU weight, or None without a CPU kernel."""
    from shared.qtypes.gguf import GGUFWeightTensor
    data = getattr(weight, "_data", None)
    if not torch.is_tensor(data) or data.device.type != "cpu" or not data.is_contiguous():
        return None
    if isinstance(weight, GGUFWeightTensor):
        if len(weight._tensor_shape) != 2:
            return None
        return FORMAT_GGUF, int(weight._tensor_type), data, None, 1.0, int(weight._tensor_shape[0])
    n, k = map(int, weight.shape)
    if hasattr(weight, "_layout") and hasattr(weight, "_alpha"):  # NVFP4
        from shared.qtypes.nvfp4 import _NVFP4_LAYOUT_TENSORCORE
        scales = weight._scale
        if (k % 64 or tuple(data.shape) != (n, k // 2) or scales.dim() != 2 or tuple(scales.shape) != ((n + 127) // 128 * 128, k // 16)
                or not scales.is_contiguous() or scales.device.type != "cpu"):
            return None
        tensorcore = weight._layout == _NVFP4_LAYOUT_TENSORCORE
        factor = weight._alpha.float() if tensorcore else weight._alpha.float() * weight._input_global_scale.float()
        return FORMAT_NVFP4, NVFP4_TENSORCORE if tensorcore else NVFP4_LEGACY, data, scales, float(factor.item()), n
    scale = getattr(weight, "_scale", None)
    if not torch.is_tensor(scale) or tuple(data.shape) != (n, k):
        return None
    scales, value = _float_rows(scale, n)
    if scales is False:
        return None
    if data.dtype == torch.int8:
        return FORMAT_INT8_ROWS, 0, data, scales, value, n
    if data.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return FORMAT_FP8_ROWS, FP8_E4M3 if data.dtype == torch.float8_e4m3fn else FP8_E5M2, data, scales, value, n
    return None


def _spec(weight):
    spec = getattr(weight, "_cpu_linear_spec", None)
    if spec is None:
        spec = weight._cpu_linear_spec = _describe(weight) or ()
    return spec


def supported(weight):
    """Whether this CPU weight computes here (its format and this CPU)."""
    spec = _spec(weight)
    lib = library()
    return bool(spec) and lib is not None and bool(lib.cpu_linear_supported(spec[0], spec[1]))


class _Mapped:
    """A CUDA view of pinned host memory: GPU kernels read and write it in place, without the copy engines, whose queues the transfers of
    streamed blocks fill."""
    def __init__(self, host):
        self.__cuda_array_interface__ = {"shape": tuple(host.shape), "typestr": "<f4", "data": (host.data_ptr(), False), "version": 2, "strides": None}


def _buffer(name, numel, device):
    """(pinned host buffer, its CUDA view) of at least numel floats."""
    current = _buffers.get(name)
    if current is None or current[0].numel() < numel:
        if current is not None:
            torch.cuda.synchronize(device)  # GPU kernels may still read the old buffer
        host = torch.empty(max(numel, 2 * (0 if current is None else current[0].numel())), dtype=torch.float32, device="cpu", pin_memory=True)
        current = _buffers[name] = (host, torch.as_tensor(_Mapped(host), device=device))
    return current


class LinearJob(ctypes.Structure):
    """One linear as a job of the host server (linear.cpp LinearJob)."""
    _fields_ = [("x", ctypes.c_void_p), ("rows", ctypes.c_int64), ("k", ctypes.c_int64), ("weight", Weight), ("threads", ctypes.c_int32), ("status", ctypes.c_int32)]


def jobs_supported(device):
    """Whether blocks computing on the CPU may be captured in CUDA graphs on this device: their linears are then jobs that kernels ask for (shared.kernels.host_doorbell)."""
    index = device if isinstance(device, int) else device.index or 0
    if index not in _jobs_supported:
        from shared.kernels import host_doorbell
        _jobs_supported[index] = CPU_DOORBELL and host_doorbell.supported(torch.device("cuda", index))
    return _jobs_supported[index]


def _job_buffer(nbytes, device):
    size = max(JOB_BUFFER_BYTES, nbytes)
    current = _job_buffers.get(size)
    if current is None:
        current = _job_buffers[size] = mapped.empty(size, device, "host jobs")
    return current[0], current[1]


def _release_tickets(tickets):
    from shared.kernels import host_doorbell
    for ticket in tickets:
        host_doorbell.release(ticket)


def _linear_job(weight, spec, rows, k, device):
    """(job, its number, CUDA views of its mapped input and result) of this weight at this row count, kept with the weight. The views are made here: nothing is computed from them during a graph capture."""
    jobs = weight.__dict__.get("_cpu_linear_jobs")
    if jobs is None:
        jobs = weight._cpu_linear_jobs = {}
        tickets = weight._cpu_linear_tickets = []
        weakref.finalize(weight, _release_tickets, tickets).atexit = False
    entry = jobs.get(rows)
    if entry is None:
        from shared.kernels import host_doorbell
        format, type, data, scales, scale, n = spec
        input_host, input_gpu = _job_buffer(rows * k * 4, device)
        output_host, output_gpu = _job_buffer(max(rows * n * 4, JOB_BUFFER_BYTES) + 4096, device)  # another buffer than the input's
        call = Weight(data.data_ptr(), None if scales is None else scales.data_ptr(), output_host.data_ptr(), n, format, type, scale, 0)
        job = LinearJob(input_host.data_ptr(), rows, k, call, cpu_threads(), 0)
        ticket = host_doorbell.register(device, ctypes.cast(library().cpu_linear_server_job, ctypes.c_void_p).value, None, job)
        weight._cpu_linear_tickets.append(ticket)
        entry = jobs[rows] = job, ticket, input_gpu[:rows * k * 4].view(torch.float32), output_gpu[:rows * n * 4].view(torch.float32)
    return entry


def _queued_linear(weight, spec, x, bias, dtype, shape, rows, k, device):
    """linear() inside a CUDA graph, or during the eager call that precedes its capture: no synchronization, the CPU's result joins the stream."""
    from shared.kernels import host_doorbell
    n = spec[5]
    _, ticket, inputs, result = _linear_job(weight, spec, rows, k, device)
    flat = x.reshape(rows * k).contiguous()
    host_doorbell.ask_floats(flat if host_doorbell.parts_supported(flat) else flat.float(), inputs, ticket)
    output = torch.empty(rows * n, dtype=torch.float32, device=device)
    host_doorbell.wait(ticket, result, output, False)
    y = output.view(*shape[:-1], n).to(dtype)
    return y if bias is None else y.add_(bias.to(device=device, dtype=dtype))


def linear(weight, x, bias, dtype):
    """x [..., k] on the GPU (already transformed by the module), weight on the CPU (supported). Returns x . W^T (+ bias) on x's device in
    dtype. The host waits for x: the CPU computes between a GPU kernel writing x to pinned memory and one reading the result from it. While a CUDA graph is prepared or captured: a job of the host server."""
    format, type, data, scales, scale, n = _spec(weight)
    shape, k, device = x.shape, x.shape[-1], x.device
    rows = x.numel() // k
    if CPU_DOORBELL and cuda_graphs.recording() and jobs_supported(device):
        return _queued_linear(weight, (format, type, data, scales, scale, n), x, bias, dtype, shape, rows, k, device)
    (host_in, mapped_in), (host_out, mapped_out) = _buffer("x", rows * k, device), _buffer("y", rows * n, device)
    mapped_in[:rows * k].view(rows, k).copy_(x.reshape(rows, k))
    x = None
    torch.cuda.current_stream(device).synchronize()  # also ends the last reads of host_out before it is written again
    call = Weight(data.data_ptr(), None if scales is None else scales.data_ptr(), host_out.data_ptr(), n, format, type, scale, 0)
    if library().cpu_linear(host_in.data_ptr(), rows, k, 1, ctypes.byref(call), cpu_threads()):
        raise RuntimeError(f"CPU linear: unsupported weight (format {format}, type {type}, {n}x{k})")
    y = mapped_out[:rows * n].view(*shape[:-1], n).to(dtype, copy=True)
    return y if bias is None else y.add_(bias.to(device=device, dtype=dtype))


if CPU_DOORBELL:
    cuda_graphs.register_cpu_jobs(jobs_supported)
