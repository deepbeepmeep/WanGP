# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Shared routed-expert execution (SwiGLU, or squared ReLU without a gate). Architecture code supplies routing, not residency, and
quantization handlers supply the storage formats and their kernels (ExpertFormat)."""
import ctypes
import os

import torch
from torch.nn import functional as F

from .experts import ExpertBank
from . import allocator, pinned


_side_streams = {}


def _side_stream(device):
    """The stream of the GPU expert branch of queue_forward, one per device."""
    if device not in _side_streams:
        _side_streams[device] = torch.cuda.Stream(device)
    return _side_streams[device]


def cpu_threads():
    return int(os.environ.get("MMGP_CPU_EXPERT_THREADS", min(8, torch.get_num_threads())))


_host_launch = None


def launch_host_job(function, job, stream):
    """Queue function(&job) on a CUDA stream (captured as a host node inside a graph capture): it runs when the work queued before it is
    done, and the work queued after it waits for it. The job (a ctypes structure) must stay alive while it may run or be replayed."""
    global _host_launch
    if _host_launch is None:
        driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        driver.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        driver.cuLaunchHostFunc.restype = ctypes.c_int
        _host_launch = driver.cuLaunchHostFunc
    status = _host_launch(stream.cuda_stream, function, ctypes.addressof(job))
    if status:
        raise RuntimeError(f"cuLaunchHostFunc failed ({status})")


class ExpertFormat:
    """Storage and kernels of routed expert banks in one checkpoint format. Quantization handlers supply formats through their
    EXPERT_FORMATS (mmgp.quant_router.expert_formats()); a bank takes the format of the checkpoint tensors it loads. Every tensor
    holds one entry per expert along its first dimension. bank.rotation is the ConvRot group of the activations (0: none), bank.gated
    whether the input projection holds gate then up rows (SwiGLU) or up rows only (squared ReLU), bank.blocks the scale blocks of the
    input and output projections."""
    name = ""

    def tensors(self, experts, hidden, intermediate, gated=True, blocks=None):
        """Checkpoint name -> (shape, dtype) of a bank's tensors, in the order of the rows of the GPU pointer table. blocks: the scale
        blocks of the two projections (None: the format's default)."""
        raise NotImplementedError

    def matches(self, tensors):
        """Whether a bank's checkpoint tensors (name -> tensor, the names of tensors()) are stored in this format."""
        raise NotImplementedError

    def blocks(self, hidden, intermediate, tensors):
        """The scale blocks (input projection, output projection) of a bank's loaded tensors (name -> tensor)."""
        raise NotImplementedError

    def cpu_job(self, bank, payload, resident, out, tokens, topk, threads):
        """(job, host function address) of a CPU computation from the addresses of a pinned routing payload (bank.pack()), of resident
        flags (or 0) and of a pinned float32 [tokens, hidden] result (zeros when every pair is computed elsewhere). The job is a ctypes
        structure passed by address; a nonzero status field reports a failure."""
        raise NotImplementedError

    def cpu_forward(self, bank, x, ids, probabilities, resident, out, tokens, topk, threads):
        """The same computation from the addresses of float32 activations and probabilities and int32 ids: returns the number of pairs
        computed (out is left unwritten when there are none)."""
        raise NotImplementedError

    def gpu_forward(self, bank, x, ids, probabilities, pointers, out=None, mid=None):
        """float32 [tokens, hidden] sum of the selected experts having GPU copies (pointers: int64 [experts, tensors] table of their
        addresses, zero rows contributing nothing); out and the float32 [tokens * topk, intermediate] scratch mid may be preallocated."""
        raise NotImplementedError

    def dequantize(self, bank, tensors):
        """float32 input projection (gate and up [2 * intermediate, hidden], or up [intermediate, hidden]) and down [hidden,
        intermediate] weights of one expert (name -> tensor), as applied to unrotated activations."""
        raise NotImplementedError


def _formats():
    from . import quant_router
    return quant_router.expert_formats()


def _locked(numel, dtype):
    """numel elements of locked RAM of their own (mmgp.pinned: exact size, counted as "experts"), kept while a view lives."""
    tensor = pinned.empty(numel * dtype.itemsize, "experts")
    if tensor is None:
        raise RuntimeError(f"Locked RAM refused for {numel * dtype.itemsize} bytes of expert staging")
    return tensor.view(dtype)


class _HostStaging:
    """Pinned buffers shared by the banks of the process, which run one after the other: the routing payload comes back in one
    blocking copy, the CPU result goes up from pinned memory. `uploaded` orders the next write of the result buffer after its
    transfer."""
    inputs = outputs = uploaded = None

    @classmethod
    def buffer(cls, name, numel, dtype):
        current = getattr(cls, name)
        if current is None or current.numel() < numel:
            pinned.retire(current)  # an earlier transfer may still read the outgrown buffer: unlocked at the next quiet point
            current = _locked(max(numel, 2 * (0 if current is None else current.numel())), dtype)
            setattr(cls, name, current)
        return current

    @classmethod
    def release(cls):
        if cls.uploaded is not None:
            cls.uploaded.synchronize()
        pinned.retire(cls.inputs)
        pinned.retire(cls.outputs)
        cls.inputs = cls.outputs = cls.uploaded = None


ACTIVATIONS = ("swiglu", "relu2")
TRITON_PAIRS = 64  # eager calls of more (token, slot) pairs compute their resident experts with the PyTorch path


class RoutedExperts(ExpertBank):
    """Routed experts: SwiGLU (activation "swiglu", input projection = gate then up rows) or squared ReLU without a gate ("relu2").
    Pageable CPU storage in their checkpoint format, plus optional GPU copies. The format is that of the loaded checkpoint tensors (its
    scale blocks read from their shapes), or the registered format named at construction (the bank's tensors are then allocated on
    device, with the given scale blocks or the format's default)."""
    def __init__(self, num_experts, hidden, intermediate, *, activation="swiglu", format=None, blocks=None, device="cpu", use_triton=False):
        super().__init__(num_experts, ())
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported expert activation {activation}")
        self.hidden, self.intermediate, self.activation, self.gated = hidden, intermediate, activation, activation == "swiglu"
        self.use_triton = use_triton
        self.format, self.rotation, self.blocks = None, 0, None
        # Eager CUDA calls of at least this many (token, slot) pairs (prompt processing) run every selected expert on the GPU, copied
        # over the bus when it has no GPU copy (_streamed_forward); None keeps them on the CPU kernels. Set by the model.
        self.streamed_pairs = None
        self._pointer_table = None
        self._pointer_version = -1
        self._jobs = {}  # tokens -> (host job, host function, pinned payload, pinned result) of queue_forward, at fixed addresses for graph replays
        if format is not None:
            format = next(candidate for candidate in _formats() if candidate.name == format)
            self._set_format(format, device, blocks)

    def _set_format(self, format, device, blocks):
        for name in self.expert_tensor_names:
            del self._buffers[name]
        declared = format.tensors(self.num_experts, self.hidden, self.intermediate, self.gated, blocks)
        for name, (shape, dtype) in declared.items():
            self.register_buffer(name, torch.empty(shape, dtype=dtype, device=device))
        self.expert_tensor_names, self.format = tuple(declared), format
        self.blocks = format.blocks(self.hidden, self.intermediate, {name: getattr(self, name) for name in declared})

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if (group := state_dict.pop(prefix + "convrot_group_size", None)) is not None:
            self.rotation = int(group.item())
        for format in _formats():
            names = format.tensors(self.num_experts, self.hidden, self.intermediate, self.gated)
            if all(prefix + name in state_dict for name in names) and format.matches({name: state_dict[prefix + name] for name in names}):
                blocks = format.blocks(self.hidden, self.intermediate, {name: state_dict[prefix + name] for name in names})
                if format is not self.format or blocks != self.blocks:
                    self._set_format(format, "meta", blocks)
                break
        else:
            raise RuntimeError(f"No registered expert format stores the tensors of {prefix[:-1]}")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def release_execution_cache(self):
        self._pointer_table = None
        self._pointer_version = -1
        _HostStaging.release()
        if self._jobs:
            torch.cuda.synchronize()  # a queued host job may still read or write its buffers
            for _, _, payload, result in self._jobs.values():
                pinned.retire(payload)
                pinned.retire(result)
            self._jobs.clear()

    def queue_forward(self, x, topk_ids, topk_weights, packed=None):
        """Graph-safe forward on the current CUDA stream: the CPU computes the selected experts in a host function queued between the
        copy of the routing payload to pinned memory and the copy of its result back, so a captured graph replays the whole call
        without returning to Python. With the Triton backend, the experts having GPU copies are computed on the GPU instead, in a
        branch running beside the CPU job; a placement change releases the graphs that read these copies. Failures are reported by
        raise_job_errors() once the stream has been synchronized."""
        shape = x.shape
        x = x.reshape(-1, self.hidden).contiguous()
        tokens, topk = topk_ids.shape
        if packed is None:
            packed = self.pack(x, topk_ids, topk_weights)
        if tokens not in self._jobs:
            payload = _locked(packed.numel(), torch.int32)
            result = _locked(tokens * self.hidden, torch.float32).view(tokens, self.hidden)
            job, function = self.format.cpu_job(self, payload.data_ptr(), self.resident_flags.data_ptr() if self.use_triton else 0, result.data_ptr(), tokens, topk, cpu_threads())
            self._jobs[tokens] = job, function, payload, result
        job, function, payload, result = self._jobs[tokens]
        stream = torch.cuda.current_stream()
        if gpu := self.use_triton and bool(self._resident):
            pointers, side = self._pointers(x.device), _side_stream(x.device)
            topk_ids, topk_weights = topk_ids.contiguous(), topk_weights.contiguous()
            # Allocated on this stream, which joins the branch before anything can reuse them.
            gpu_output = torch.empty(tokens, self.hidden, dtype=torch.float32, device=x.device)
            mid = torch.empty(tokens * topk, self.intermediate, dtype=torch.float32, device=x.device)
            side.wait_stream(stream)
            with torch.cuda.stream(side):
                self.format.gpu_forward(self, x, topk_ids, topk_weights, pointers, out=gpu_output, mid=mid)
        payload.copy_(packed, non_blocking=True)  # read only by the host function queued after it on this stream
        launch_host_job(function, job, stream)
        output = torch.empty(result.shape, dtype=torch.float32, device=x.device).copy_(result, non_blocking=True)
        if gpu:
            stream.wait_stream(side)
            output.add_(gpu_output)
        return output.to(dtype=x.dtype).reshape(shape)

    def _streamed_forward(self, x, ids, weights):
        """Passes of many rows (prompt processing) on CUDA: every selected expert runs on the GPU for all its rows, its resident copy or
        its storage copied over the bus once, decoded to float32 (the PyTorch path's arithmetic) and released after use. The CPU would
        compute each row of an expert again; the bus reads it once."""
        out = torch.zeros(x.shape[0], self.hidden, dtype=torch.float32, device=x.device)
        flat_ids, flat_weights = ids.reshape(-1), weights.reshape(-1).float()
        order = torch.argsort(flat_ids)
        counts = torch.bincount(flat_ids, minlength=self.num_experts).tolist()
        topk = ids.shape[-1]
        with self.lease() as resident:
            start = 0
            for expert, count in enumerate(counts):
                if not count:
                    continue
                pairs = order[start:start + count]
                start += count
                rows = pairs // topk
                tensors = resident.get(expert) or {name: getattr(self, name)[expert].to(x.device) for name in self.expert_tensor_names}
                w13, w2 = self.format.dequantize(self, tensors)
                tensors = None
                hidden = F.linear(x[rows].float(), w13)
                if self.gated:
                    gate, up = hidden.chunk(2, dim=-1)
                    hidden = F.silu(gate) * up
                    gate = up = None
                else:
                    hidden = F.relu(hidden).square()
                out.index_add_(0, rows, F.linear(hidden, w2) * flat_weights[pairs, None])
                hidden = w13 = w2 = None
        return out

    def raise_job_errors(self):
        """After a stream synchronization: raise the failure of a host job queued by queue_forward, if any."""
        for job, _, _, _ in self._jobs.values():
            if job.status:
                status, job.status = job.status, 0
                raise RuntimeError(f"CPU expert kernel failed ({status})")

    def _pointers(self, device):
        if self._pointer_version != self.residency_version or self._pointer_table is None:
            rows = [[resident[name].data_ptr() for name in self.expert_tensor_names] if (resident := self._resident.get(expert)) else [0] * len(self.expert_tensor_names)
                    for expert in range(self.num_experts)]
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("The GPU expert pointer table must be prepared before graph capture")
            with allocator.normal_allocations():  # read by the graphs of the placement, not graph scratch
                self._pointer_table = torch.tensor(rows, dtype=torch.int64, device=device)
            self._pointer_version = self.residency_version
        return self._pointer_table

    @staticmethod
    def pack(x, topk_ids, topk_weights):
        """forward()'s routing payload as one int32 tensor: float32 activations, float32 probabilities, int32 ids. Graph-safe:
        computing it in the graphed stage that routes saves its launches."""
        return torch.cat((x.reshape(-1).float().view(torch.int32), topk_weights.reshape(-1).float().view(torch.int32), topk_ids.reshape(-1).to(torch.int32)))

    @torch.inference_mode()
    def forward(self, x, topk_ids, topk_weights, packed=None):
        shape = x.shape
        x = x.reshape(-1, self.hidden)
        tokens, topk = topk_ids.shape
        cuda = x.device.type == "cuda"
        if cuda and self.streamed_pairs is not None and tokens * topk >= self.streamed_pairs:
            return self._streamed_forward(x, topk_ids, topk_weights).to(dtype=x.dtype).reshape(shape)
        if packed is None:
            packed = self.pack(x, topk_ids, topk_weights)
        if cuda:
            host = _HostStaging.buffer("inputs", packed.numel(), torch.int32)
            host[:packed.numel()].copy_(packed)  # blocking: the CPU reads it next
        else:
            host = packed
        x_offset, ids_offset = tokens * self.hidden, tokens * (self.hidden + topk)
        ids = host[ids_offset:ids_offset + tokens * topk].numpy()
        with self.lease() as resident:
            gpu_output = None
            if resident and self._flags[ids].any():
                if not cuda:
                    raise ValueError("A managed expert bank with GPU residents expects CUDA activations")
                if self.use_triton and tokens * topk <= TRITON_PAIRS:
                    gpu_output = self.format.gpu_forward(self, x.contiguous(), topk_ids.contiguous(), topk_weights.contiguous(), self._pointers(x.device))
                else:
                    # PyTorch path (legacy/cg, and passes of many rows: each expert is decoded once for all its rows, while the Triton
                    # kernels read an expert's weights again for every (token, slot) pair). No dependency on Triton or FlashAttention.
                    gpu_output = torch.zeros_like(x, dtype=torch.float32)
                    cpu_ids = torch.from_numpy(ids).view(tokens, topk)
                    for expert in sorted(set(ids.tolist()).intersection(resident)):
                        rows, slots = torch.where(cpu_ids == expert)
                        rows, slots = rows.to(x.device), slots.to(x.device)
                        w13, w2 = self.format.dequantize(self, resident[expert])
                        if self.gated:
                            gate, up = F.linear(x[rows].float(), w13).chunk(2, dim=-1)
                            output = F.linear(F.silu(gate) * up, w2)
                        else:
                            output = F.linear(F.relu(F.linear(x[rows].float(), w13)).square(), w2)
                        gpu_output.index_add_(0, rows, output * topk_weights[rows, slots, None].float())
            if cuda:
                result = _HostStaging.buffer("outputs", x_offset, torch.float32)
                if _HostStaging.uploaded is not None:
                    _HostStaging.uploaded.synchronize()
            else:
                result = torch.empty(x_offset, dtype=torch.float32, device="cpu")
            base = host.data_ptr()
            count = self.format.cpu_forward(self, base, base + 4 * ids_offset, base + 4 * x_offset, self.resident_flags.data_ptr(), result.data_ptr(), tokens, topk, cpu_threads())
            if count:
                output = result[:x_offset].view(tokens, self.hidden)
                if cuda:
                    output = torch.empty_like(output, device=x.device).copy_(output, non_blocking=True)
                    _HostStaging.uploaded = _HostStaging.uploaded or torch.cuda.Event()
                    _HostStaging.uploaded.record()
                if gpu_output is not None:
                    output.add_(gpu_output)
            else:
                output = gpu_output if gpu_output is not None else torch.zeros_like(x, dtype=torch.float32)
            return output.to(dtype=x.dtype).reshape(shape)


class SwiGLUExperts(RoutedExperts):
    """Routed SwiGLU experts (RoutedExperts with the "swiglu" activation)."""
    def __init__(self, num_experts, hidden, intermediate, **kwargs):
        super().__init__(num_experts, hidden, intermediate, activation="swiglu", **kwargs)
