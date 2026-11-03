# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

#

import torch
import gc
import bisect
import collections
import time
import functools
import sys
import os
import json
import inspect
import psutil
import builtins
import math
import re
import fnmatch
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
import ctypes
import threading
import itertools
import weakref
import numpy
from accelerate import init_empty_weights
from tqdm import tqdm
from functools import wraps
import functools
import types
import inspect

from mmgp import safetensors2, runtime, row_views, pinned, memory_pressure
from .read_ahead import MAX_RANGES, COLD_PIECE, cold_ranges, prefetch_ranges
from mmgp import __version__, profile_type
from .quant_router import (
    apply_pre_quantization,
    cache_quantization_for_file,
    detect_and_convert,
    detect_safetensors_format,
    sd_split_linear,
    split_linear_modules,
    split_fused_weights,
    get_extension_handler,
    normalize_extension_path,
)
from optimum.quanto import freeze,  qfloat8, qint2, qint4 , qint8, quantize, QModuleMixin, QLinear, QTensor,  quantize_module, register_qmodule
# support for Embedding module quantization that is not supported by default by quanto
@register_qmodule(torch.nn.Embedding)
class QEmbedding(QModuleMixin, torch.nn.Embedding):
    bias = None
    @classmethod
    def qcreate(cls, module, weights, activations = None, optimizer = None, device = None):
        module.bias = None
        return cls( module.num_embeddings, module.embedding_dim, module.padding_idx , module.max_norm, module.norm_type, module.scale_grad_by_freq, module.sparse, dtype=module.weight.dtype, device=device, weights=weights,
                    activations=activations, optimizer=optimizer, quantize_input=True)      
    @classmethod
    def from_module(cls, module, weights = None, activations = None, optimizer = None):
        qmodule = super().from_module(module, weights, activations, optimizer)
        if "embed_scale" in module._buffers:
            qmodule.register_buffer("embed_scale", module._buffers["embed_scale"], persistent="embed_scale" not in module._non_persistent_buffers_set)
        elif hasattr(module, "embed_scale"):
            qmodule.embed_scale = module.embed_scale
        return qmodule
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.embedding( input, self.qweight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse )
        embed_scale = getattr(self, "embed_scale", None)
        if torch.is_tensor(embed_scale):
            embed_scale = embed_scale.to(device=output.device, dtype=output.dtype)
        return output if embed_scale is None else output * embed_scale



def cudacontext(device):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with torch.device(device):
                return func(*args, **kwargs)
        return wrapper
    return decorator

    return _orig_get_parameter_device(parameter)

try:
    import transformers.modeling_utils as mu # Transfomers v4
    _orig_get_parameter_device = mu.get_parameter_device
    def _patched_get_parameter_device(parameter):
        forced = getattr(parameter, "_force_device", None)
        if forced is not None:
            return torch.device(forced)
        return _orig_get_parameter_device(parameter)    
    mu.get_parameter_device = _patched_get_parameter_device	
except:
    pass

try:
    import transformers.modeling_utils as mu # Transfomers v5
    _orig_device = mu.ModuleUtilsMixin.device.fget
    def _device(self):
        forced = getattr(self, "_force_device", None)
        if forced is not None:
            return torch.device(forced)
        return _orig_device(self)    
    mu.ModuleUtilsMixin.device = property(_device)
except:
    pass


shared_state = {}

def get_cache(cache_name):
    all_cache = shared_state.get("_cache",  None)
    if all_cache is None:
        all_cache = {}
        shared_state["_cache"]=  all_cache
    cache = all_cache.get(cache_name, None)
    if cache is None:
        cache = {}
        all_cache[cache_name] = cache
    return cache

def clear_caches():
    all_cache = shared_state.get("_cache",  None)
    if all_cache is not None:
        all_cache.clear()


mmm = safetensors2.mmm

default_verboseLevel = 1

ONE_MB =  1048576
sizeofhalffloat = torch.bfloat16.itemsize
sizeofint8 = torch.int8.itemsize

physical_memory= psutil.virtual_memory().total # the RAM MMGP plans with: a RAM budget's when one is set (virtual_gpu.install_ram)
from . import virtual_gpu
machine_memory = virtual_gpu.ram_state.get("machine", physical_memory) # the machine's RAM, of which the reserved RAM may take a share

class clock:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    @classmethod
    def start(cls):
        self = cls()        
        self.start_time =time.time()
        return self        

    def stop(self):
        self.stop_time =time.time()  

    def time_gap(self):
        return self.stop_time - self.start_time
    
    def format_time_gap(self):
        return f"{self.stop_time - self.start_time:.2f}s"

# useful functions to move a group of tensors (to design custom offload patches)
def move_tensors(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        _dict = {}
        for k, v in obj.items():
            _dict[k] = move_tensors(v, device)
        return _dict
    elif isinstance(obj, list):
        _list = []
        for v in obj:
            _list.append(move_tensors(v, device))
        return _list
    else:
        raise TypeError("Tensor or list / dict of tensors expected")
def _get_module_name(v):
    return v.__module__.lower()


def _compute_verbose_level(level):
    if level <0:        
        level = safetensors2.verboseLevel = default_verboseLevel
    safetensors2.verboseLevel = level
    return level

def _get_perc_reserved_mem_max(perc_reserved_mem_max = 0):
    perc_reserved_mem_max = float(perc_reserved_mem_max or 0)
    if perc_reserved_mem_max <=0:
        perc_reserved_mem_max = float(os.getenv("perc_reserved_mem_max", 0) or 0)

    if perc_reserved_mem_max <= 0:             
        perc_reserved_mem_max = 0.40 if os.name == 'nt' else 0.6
    return perc_reserved_mem_max
    
def _get_max_reservable_memory(perc_reserved_mem_max = 0):
    # a share of the machine's RAM (a safeguard: the system needs the rest), within a RAM budget
    return min(perc_reserved_mem_max * machine_memory, physical_memory)

def _detect_main_towers(model, min_floors = 5, extra_towers = ()):
    cur_blocks_prefix = None
    towers_modules= []
    towers_names= []

    floors_modules= []
    tower_name = None


    for submodule_name, submodule in model.named_modules():  

        if submodule_name=='':
            continue

        if cur_blocks_prefix != None:
            if submodule_name.startswith(cur_blocks_prefix):
                depth_prefix = cur_blocks_prefix.split(".")
                depth_name = submodule_name.split(".")
                level  =  depth_name[len(depth_prefix)-1]                        
                pre , num = _extract_num_from_str(level)

                if num != cur_blocks_seq: 
                    floors_modules.append(submodule)

                cur_blocks_seq = num
            else:
                if len(floors_modules) >= min_floors or floors_modules and tower_name.rstrip(".") in extra_towers:
                    towers_modules += floors_modules
                    towers_names.append(tower_name)
                tower_name = None
                floors_modules= []
                cur_blocks_prefix, cur_blocks_seq = None, -1

        if cur_blocks_prefix == None:
            pre , num = _extract_num_from_str(submodule_name)
            if isinstance(submodule, (torch.nn.ModuleList)):  
                cur_blocks_prefix, cur_blocks_seq = pre + ".",  -1
                tower_name = submodule_name + "." 
            elif num >=0:
                cur_blocks_prefix, cur_blocks_seq = pre, num
                tower_name = submodule_name[ :-1]  
                floors_modules.append(submodule)

    if len(floors_modules) >= min_floors or floors_modules and tower_name.rstrip(".") in extra_towers:
        towers_modules += floors_modules
        towers_names.append(tower_name)

    return towers_names, towers_modules



def _get_model(model_path):
    if os.path.isfile(model_path):
        return model_path
    
    from pathlib import Path
    _path = Path(model_path).parts
    _filename = _path[-1]
    _path = _path[:-1]
    if len(_path)<=1:
        raise Exception("file not found")
    else:
        try:
            from huggingface_hub import  hf_hub_download #snapshot_download,    
            repoId=  os.path.join(*_path[0:2] ).replace("\\", "/")

            if len(_path) > 2:
                _subfolder = os.path.join(*_path[2:] )
                model_path = hf_hub_download(repo_id=repoId,  filename=_filename,  subfolder=_subfolder)
            else:
                model_path = hf_hub_download(repo_id=repoId,  filename=_filename)
        except:
           model_path = None 
    return model_path



def _remove_model_wrapper(model):
    if not model._modules is None:
        if len(model._modules)!=1:
            return model
    sub_module = model._modules[next(iter(model._modules))]
    if hasattr(sub_module,"config") or hasattr(sub_module,"base_model"):
        return sub_module
    return model  

 

def _move_to_pinned_tensor(source_tensor, big_tensor, offset, length, copier=None):
    dtype= source_tensor.dtype
    shape = source_tensor.shape
    if not source_tensor.is_contiguous(): # copied in place, without the contiguous intermediate a reshape would make
        t = big_tensor[offset: offset + length].view(dtype).view(shape)
        t.copy_(source_tensor)
        return t
    if len(shape) > 0 :
        t = source_tensor.view(torch.uint8)
        t = torch.reshape(t, (length,))
    else:
        # Preserve raw bytes for 0-dim tensors (scalar buffers like embed_scale).
        t = source_tensor.view(1).view(torch.uint8)
        t = torch.reshape(t, (length,))
    # magic swap !
    if copier is None:
        big_tensor[offset: offset + length] = t 
    else:
        copier.copy(big_tensor.data_ptr() + offset, t, length)
    t = big_tensor[offset: offset + length]
    t = t.view(dtype)
    t = torch.reshape(t, shape)
    return t

class _PinningMemoryError(RuntimeError):
    """Allocation or host-registration failure, distinct from loading callbacks and copy errors."""


class _PinningCopier:
    # Copies a model into its pinned blocks on several threads while the next blocks are allocated. Each block's copies are awaited
    # before the block two allocations later, which reports the block as pinned. A source is released as soon as it is copied, which
    # frees its RAM if it is not mapped from a file.
    def __init__(self):
        self.device = torch.cuda.current_device()
        self.executor = ThreadPoolExecutor(max_workers=_staging_copy_threads(), thread_name_prefix="mmgp_pinning", initializer=torch.cuda.set_device, initargs=(self.device,))
        self.blocks = collections.OrderedDict()
        self.lock_guard = threading.Lock()

    def copy(self, dst, source, length):
        futures = self.blocks[next(reversed(self.blocks))]
        holder, left = [source], [(length + STAGING_CHUNK_SIZE - 1) // STAGING_CHUNK_SIZE]
        # a source mapped from a checkpoint: its pages leave the working set once copied, rather than when its whole view of the file is
        # released (the end of the pinning, or never if a part of the view is not pinned). A page it shares with a source copied later comes
        # back and leaves again with that one
        mapped = _mapped_pages([(source.data_ptr(), length)])
        def copied(_):
            with self.lock_guard:
                left[0] -= 1
                done = left[0] == 0
            if done:
                holder.clear()
                if mapped:
                    _drop_mapped_pages(mapped)
        for offset in range(0, length, STAGING_CHUNK_SIZE):
            future = self.executor.submit(ctypes.memmove, dst + offset, source.data_ptr() + offset, min(STAGING_CHUNK_SIZE, length - offset))
            future.add_done_callback(copied)
            futures.append(future)

    def start_block(self, block_no):
        self.blocks[block_no] = []

    def wait(self, keep=0):
        done = []
        while len(self.blocks) > keep:
            block_no, futures = self.blocks.popitem(last=False)
            for future in futures:
                future.result()
            done.append(block_no)
        return done

    def lock(self, block, owner):
        # locks a filled block of normal RAM in place, from a copy thread so that a failure leaves no CUDA error pending in the main thread
        owner.registration = pinned.Registration(owner.base, block.data_ptr(), block.numel(), self.device, kind="model")
        self.lock_error = self.executor.submit(owner.registration.register).result()
        return owner.registration.registered

    def close(self):
        try:
            self.wait()
        finally:
            self.executor.shutdown()


class _PinnedBuffer:
    """A buffer of a model pinned for good (_pin_to_memory, with ranks): the tensors of one block (its rank), either all copied from a
    checkpoint mapping or all made at load. While its model does not run, the pinned pool of another offload object can take its room
    (_PinnedPool._reclaim_pinned):
    - checkpoint copies are freed, their tensors back on their views of the file (kept: their pages, out of the working set, cost no RAM),
      and filled again from them when the model loads (refill);
    - tensors made at load cannot be read again: they are only unlocked, their RAM staying the model's, and locked again (lock).
    items: [parameter, names of its inner tensors (None: a plain tensor), offsets in the buffer, sources (views of the file, mapped only),
    tied parameters that follow it]."""
    __slots__ = ("rank", "mapped", "tensor", "owner", "items", "nbytes", "state")

    def __init__(self, key, tensor, owner):
        (self.rank, self.mapped), self.tensor, self.owner, self.items, self.nbytes, self.state = key, tensor, owner, [], tensor.numel(), "locked"

    @staticmethod
    def _repoint(item, tensors):
        p, names, _, _, followers = item
        for target in [p, *followers]:
            if names is None:
                target.data = tensors[0]
            else:
                _set_quantized_subtensors(target, dict(zip(names, tensors)))

    def free(self):
        # at a quiet point: unlocking waits for the GPU
        for item in self.items:
            self._repoint(item, item[3])
        self.owner.registration.close()
        self.tensor = self.owner = None
        self.state = "freed"

    def unlock(self):
        self.owner.registration.close()
        self.state = "unlocked"

    def lock(self):
        registration = pinned.lock(self.owner.base, self.tensor.data_ptr(), self.nbytes, "model")
        if registration is None:
            return False
        self.owner.registration, self.state = registration, "locked"
        return True

    def refill(self):
        tensor, owner = pinned.block(self.nbytes)
        moved = [[_move_to_pinned_tensor(source, tensor, offset, source.numel() * source.element_size()) for source, offset in zip(item[3], item[2])] for item in self.items]
        owner.registration = pinned.lock(owner.base, tensor.data_ptr(), self.nbytes, "model")
        if owner.registration is None:
            return False # its tensors stay on their views of the file
        for item, tensors in zip(self.items, moved):
            self._repoint(item, tensors)
        _drop_mapped_pages(_mapped_pages([(source.data_ptr(), source.numel() * source.element_size()) for item in self.items for source in item[3]]))
        self.tensor, self.owner, self.state = tensor, owner, "locked"
        return True


class _ReadAheadPinningCopier(_PinningCopier):
    # Copies in batches of COLD_PIECE (small tensors included): the parts of a batch that a drive must read are read ahead in large requests
    # before its copies are submitted, at most two batches ahead of the copy threads, which would fault them in 64 KiB reads. A batch found in
    # RAM is submitted at once.
    def __init__(self):
        super().__init__()
        self.buffered, self.outstanding = collections.deque(), collections.deque()
        self.buffered_bytes = 0

    def _flush(self):
        if not self.buffered:
            return
        cold = cold_ranges([(source.data_ptr(), count) for _, source, count in self.buffered])
        if cold:
            while len(self.outstanding) > 1:
                for future in self.outstanding.popleft():
                    future.result()
            prefetch_ranges(cold)
        futures = self.blocks[next(reversed(self.blocks))]
        first = len(futures)
        while self.buffered:
            dst, source, count = self.buffered.popleft()
            super().copy(dst, source, count)
            del source
        if cold:
            self.outstanding.append(futures[first:])
        self.buffered_bytes = 0

    def copy(self, dst, source, length):
        offset = 0
        while offset < length:
            count = min(COLD_PIECE - self.buffered_bytes, length - offset)
            self.buffered.append((dst + offset, source[offset:offset + count], count))
            self.buffered_bytes += count
            offset += count
            if self.buffered_bytes == COLD_PIECE or len(self.buffered) >= MAX_RANGES:
                self._flush()

    def start_block(self, block_no):
        self._flush()
        super().start_block(block_no)

    def wait(self, keep=0):
        self._flush()
        done = super().wait(keep)
        while self.outstanding and builtins.all(future.done() for future in self.outstanding[0]):
            self.outstanding.popleft()
        return done

    def close(self):
        try:
            super().close()
        finally:
            self.outstanding.clear()


def _safetensors_load_file(file_path, writable_tensors = True):
    from collections import OrderedDict
    sd = OrderedDict()    

    with safetensors2.safe_open(file_path, framework="pt", device="cpu", writable_tensors =writable_tensors) as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
        metadata = f.metadata()

    return sd, metadata

def _force_load_buffer(p):
    # To do : check if buffer was persistent and transfer state, or maybe swap keep already this property ?
    q = _make_buffer(p.clone())
    torch.utils.swap_tensors(p, q)
    del q

def _force_load_parameter(p):
    q = _make_parameter(p.clone(), requires_grad=p.requires_grad)
    torch.utils.swap_tensors(p, q)
    del q

def _make_buffer(tensor):
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False):
            return torch.nn.Buffer(tensor)
    return torch.nn.Buffer(tensor)

def _make_parameter(tensor, requires_grad=False):
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False):
            return torch.nn.Parameter(tensor, requires_grad=requires_grad)
    return torch.nn.Parameter(tensor, requires_grad=requires_grad)

_graph_serials = itertools.count(1)

def _graph_serial(tensor):
    # a number given once to a tensor object, never reused: the CUDA graph keys of the blocks name their tensors by it
    serial = tensor.__dict__.get("_mm_graph_serial", None)
    if serial is None:
        serial = tensor._mm_graph_serial = next(_graph_serials)
    return serial

_placement_generation = 0  # counts the tensors MMGP installed in modules (blocks loaded or unloaded): weights keys that depend only on them

def _install_tensor(module, name, tensor):
    # registers the way nn.Module._apply does: no wrapper object, no __setattr__ checks, and a buffer keeps its persistence
    global _placement_generation
    _placement_generation += 1
    if name in module._parameters:
        module._parameters[name] = tensor
    else:
        module._buffers[name] = tensor

def _unwrap_quantized_tensor(tensor):
    if hasattr(tensor, "_data") and torch.is_tensor(tensor._data):
        return tensor._data
    return tensor

def _qtensor_get_quantized_subtensors(self):
    subtensors = []
    if getattr(self, "_qtype", None) in (qint2, qint4):
        data = _unwrap_quantized_tensor(self._data)
        subtensors.append(("data", data))
        if hasattr(self, "_scale_shift") and self._scale_shift is not None:
            subtensors.append(("scale_shift", self._scale_shift))
        else:
            if hasattr(self, "_scale") and self._scale is not None:
                subtensors.append(("scale", self._scale))
            if hasattr(self, "_shift") and self._shift is not None:
                subtensors.append(("shift", self._shift))
        return subtensors

    if hasattr(self, "_data"):
        data = _unwrap_quantized_tensor(self._data)
        subtensors.append(("data", data))
    if hasattr(self, "_scale") and self._scale is not None:
        subtensors.append(("scale", self._scale))
    return subtensors

def _qtensor_set_quantized_subtensors(self, sub_tensors):
    if isinstance(sub_tensors, dict):
        sub_map = sub_tensors
    else:
        sub_map = {name: tensor for name, tensor in sub_tensors}

    data = sub_map.get("data", None)
    if data is not None:
        if hasattr(self, "_data") and hasattr(self._data, "_data") and torch.is_tensor(self._data._data):
            self._data._data = data
        else:
            self._data = data

    if getattr(self, "_qtype", None) in (qint2, qint4):
        if "scale_shift" in sub_map and sub_map["scale_shift"] is not None:
            self._scale_shift = sub_map["scale_shift"]
        else:
            if "scale" in sub_map and sub_map["scale"] is not None:
                self._scale = sub_map["scale"]
            if "shift" in sub_map and sub_map["shift"] is not None:
                self._shift = sub_map["shift"]
    else:
        if "scale" in sub_map and sub_map["scale"] is not None:
            self._scale = sub_map["scale"]

if not hasattr(QTensor, "get_quantized_subtensors"):
    QTensor.get_quantized_subtensors = _qtensor_get_quantized_subtensors
if not hasattr(QTensor, "set_quantized_subtensors"):
    QTensor.set_quantized_subtensors = _qtensor_set_quantized_subtensors

def _is_mapped(tensor):
    # a parameter kept in mapped RAM (runtime.map_in_ram, mapped.py): outside the blocks, the base, the budgets and the pinning
    return getattr(tensor, "_mm_mapped", False)

def _to_cpu(model):
    # model.to("cpu"), the mapped parameters (CUDA views of their RAM) left in place
    held = [(module, name, p) for module in model.modules() for name, p in module._parameters.items() if p is not None and _is_mapped(p)]
    for module, name, _ in held:
        module._parameters[name] = None
    try:
        return model.to("cpu")
    finally:
        for module, name, p in held:
            module._parameters[name] = p

def _cpu_modules(model):
    # The modules kept in RAM: those computed on the CPU (runtime.lookup_on_cpu) whose tensors are plain, a quantized table keeping the
    # GPU path of its qtype, whose kernels and rotations need not run on the CPU; and those that never run (runtime.unused). A tensor they
    # share with a module computed on the GPU would be moved there by its loads: an error, not a silent table in VRAM or a head run on the CPU.
    plain = lambda tensor: type(tensor.data) is torch.Tensor
    names = tuple(name for name, module in model.named_modules() if getattr(module, "_mm_unused", False)
                  or getattr(module, "_mm_on_cpu", False) and builtins.all(map(plain, itertools.chain(module.parameters(), module.buffers()))))
    if not names:
        return names
    cpu_objects = {id(module) for name in names for module in model.get_submodule(name).modules()}
    shared = {p.data_ptr() for name in names for p in model.get_submodule(name).parameters() if p.numel()}
    tied = [name for name, module in model.named_modules() if id(module) not in cpu_objects and any(plain(p) and p.data_ptr() in shared for p in module.parameters(recurse=False))]
    if tied:
        raise ValueError(f"Module(s) {tied} share the weights of a module kept in RAM: mark them with runtime.lookup_on_cpu or runtime.unused as well")
    return names

def _get_quantized_subtensors(p):
    getter = getattr(p, "get_quantized_subtensors", None)
    if getter is None:
        return None
    sub_tensors = getter()
    if not sub_tensors:
        return None
    if isinstance(sub_tensors, dict):
        sub_tensors = list(sub_tensors.items())
    out = []
    for name, tensor in sub_tensors:
        if tensor is None:
            continue
        if torch.is_tensor(tensor):
            out.append((name, tensor))
    return out if out else None

def _set_quantized_subtensors(p, sub_tensors):
    setter = getattr(p, "set_quantized_subtensors", None)
    if setter is None:
        return False
    setter(sub_tensors)
    return True

def _subtensors_nbytes(sub_tensors):
    return sum(torch.numel(t) * t.element_size() for _, t in sub_tensors)

def _get_tensor_ref(p):
    sub_tensors = _get_quantized_subtensors(p)
    if sub_tensors:
        for _, t in sub_tensors:
            ref = t.data_ptr()
            del sub_tensors
            return ref
        del sub_tensors
    return p.data_ptr()

BIG_TENSOR_MAX_SIZE = 2**28 # 256 MB
BLOCK_ALIGN = 512 # alignment of the tensors packed in pinned memory and in the VRAM buffer of a block: that of the CUDA caching allocator

def _block_aligned(nbytes):
    return (nbytes + BLOCK_ALIGN - 1) // BLOCK_ALIGN * BLOCK_ALIGN
BIG_TENSOR_MIN_SIZE = 2**26 # 64 MB
RESERVED_RAM_MIN_AVAILABLE = BIG_TENSOR_MAX_SIZE # 2**27 # 128 MB
STAGING_CHUNK_SIZE = 2**24 # 16 MB: plain tensors cross the pinned staging slots in chunks of this size
STAGING_MIN_SIZE = 2**20 # 1 MB: smaller pageable tensors keep the driver copy
STAGING_MAX_SIZE = BIG_TENSOR_MAX_SIZE # quantized tensors are staged whole (their inner tensors must stay complete), larger ones keep the driver copy
STAGING_PARALLEL_MIN_SIZE = 2**22 # 4 MB: smaller copies stay on one thread
PREFETCH_VRAM_SHARE = 0.05 # blocks prefetched beyond the next one must fit in this share of the VRAM


def _staging_copy_threads():
    # copy threads while the main thread waits: a third of the CPUs this process may use (first touch of mapped weights is fault bound, warm copies RAM bound)
    cpus = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process, "cpu_affinity") else os.cpu_count() or 1
    return max(1, min(8, cpus // 3))


def _staging_leaves(t):
    # plain inner tensors of a (possibly nested) traceable tensor subclass, or None if it cannot be rebuilt from them
    if type(t) is torch.Tensor:
        return [t] if t.is_contiguous() else None
    if not hasattr(t, "__tensor_flatten__"):
        return None
    leaves = []
    for name in t.__tensor_flatten__()[0]:
        inner = _staging_leaves(getattr(t, name))
        if inner is None:
            return None
        leaves += inner
    return leaves

def _staging_nbytes(t):
    return (t.numel() * t.element_size() + 63) // 64 * 64

def _same_moved_tensor(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is torch.Tensor:
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    names, meta = a.__tensor_flatten__()
    names_b, meta_b = b.__tensor_flatten__()
    try:
        same_meta = names == names_b and meta == meta_b and a.size() == b.size() and a.stride() == b.stride()
    except Exception:
        return False
    return same_meta and builtins.all(_same_moved_tensor(getattr(a, name), getattr(b, name)) for name in names)


class _PinnedStager:
    """Moves pageable weights to the GPU through two small reusable pinned slots.
    A CUDA copy from pageable RAM goes through a driver staging buffer on the calling thread and cannot overlap GPU work.
    Staging here (CPU copy into pinned memory, then an asynchronous DMA) is faster, and overlaps compute when run by the prefetch thread.
    The CPU copies are memmoves on plain threads: torch's parallel copy_ occupies every core (and keeps its threads spinning), which
    starves the main thread that launches the model kernels. A copy runs on one thread while the main thread is busy (checking again every
    STAGING_CHUNK_SIZE bytes) and its remainder is split across the copy threads as soon as the main thread waits for it, since copying
    mapped weights touched for the first time is bound by page faults (much faster in parallel).
    Quantized tensors are rebuilt around pinned copies of their inner tensors and moved with their own .to(); the first tensor of
    each class is checked against the driver copy and classes that do not match keep the driver copy."""

    def __init__(self):
        self.slots = [pinned.Buffer("staging", STAGING_CHUNK_SIZE) for _ in range(2)]
        self.events, self.slot_no, self.checked_classes, self.enabled, self.helpers = [None, None], 0, {}, True, None
        self.copy_threads = _staging_copy_threads()

    def _memmove(self, dst, src, nbytes, parallel):
        offset = 0
        while offset < nbytes:
            remaining = nbytes - offset
            if self.copy_threads > 1 and remaining >= STAGING_PARALLEL_MIN_SIZE and parallel():
                if self.helpers is None:
                    self.helpers = ThreadPoolExecutor(max_workers=self.copy_threads - 1, thread_name_prefix="mmgp_staging")
                # what a drive must read is read ahead in pieces, the next one while the threads copy the current one (they would fault it in
                # 64 KiB reads)
                cold = cold_ranges([(src + offset, remaining)])
                step = COLD_PIECE if cold else remaining
                if cold:
                    prefetch_ranges([(src + offset, src + offset + min(step, remaining))])
                for piece in range(offset, nbytes, step):
                    length = min(step, nbytes - piece)
                    part = (length + self.copy_threads - 1) // self.copy_threads
                    futures = [self.helpers.submit(ctypes.memmove, dst + piece + start, src + piece + start, min(part, length - start)) for start in range(part, length, part)]
                    if cold and piece + length < nbytes:
                        prefetch_ranges([(src + piece + length, src + min(nbytes, piece + length + step))])
                    ctypes.memmove(dst + piece, src + piece, part)
                    for future in futures:
                        future.result()
                return
            length = min(STAGING_CHUNK_SIZE, remaining)
            ctypes.memmove(dst + offset, src + offset, length)
            offset += length

    def release(self):
        if self.helpers is not None:
            self.helpers.shutdown(wait=True)
            self.helpers = None
        for slot in self.slots:
            slot.release()
        self.events = [None, None]

    def _acquire(self, nbytes):
        slot_no = self.slot_no = self.slot_no ^ 1
        if self.events[slot_no] is not None:
            self.events[slot_no].synchronize()
            self.events[slot_no] = None
        with torch.inference_mode(False): # the slots are written by the main thread and the prefetch thread, whatever their inference mode
            slot = self.slots[slot_no].get(nbytes) # exact multiples of STAGING_CHUNK_SIZE, an outgrown slot unlocked at the next quiet point
        if slot is None:
            self.enabled = False
            print("Unable to lock RAM for staged transfers, pageable weights will be copied with slower transfers")
        return slot

    def _release(self):
        self.events[self.slot_no] = torch.cuda.Event()
        self.events[self.slot_no].record()

    def _pinned_copy(self, t, slot, offset, parallel):
        if type(t) is torch.Tensor:
            nbytes = t.numel() * t.element_size()
            view = slot[offset:offset + nbytes].view(t.dtype).view(t.shape)
            self._memmove(view.data_ptr(), t.data_ptr(), nbytes, parallel)
            return view, offset + _staging_nbytes(t)
        names, meta = t.__tensor_flatten__()
        inner = {}
        for name in names:
            inner[name], offset = self._pinned_copy(getattr(t, name), slot, offset, parallel)
        return type(t).__tensor_unflatten__(inner, meta, t.size(), t.stride()), offset

    def to_gpu(self, p, parallel=lambda: True, trust_pinned=True):
        # trust_pinned False: a model of the pinned pool, which locks its blocks in place by the whole pages of their tensors. A tensor may
        # then report pinned while its last page is not, and the driver refuses its asynchronous copy ("invalid argument"): every tensor
        # crosses the slots
        t = p.data if type(p) is torch.nn.Parameter else p
        if not self.enabled:
            return p.to("cuda", non_blocking=True)
        if type(t) is torch.Tensor:
            nbytes = t.numel() * t.element_size()
            if not t.is_contiguous() or trust_pinned and (nbytes < STAGING_MIN_SIZE or t.is_pinned()):
                return p.to("cuda", non_blocking=True)
            dst = torch.empty(t.shape, dtype=t.dtype, device="cuda")
            dst_bytes = dst.reshape(-1).view(torch.uint8)
            for offset in range(0, nbytes, STAGING_CHUNK_SIZE):
                length = min(STAGING_CHUNK_SIZE, nbytes - offset)
                slot = self._acquire(length)
                if slot is None:
                    return p.to("cuda", non_blocking=True)
                self._memmove(slot.data_ptr(), t.data_ptr() + offset, length, parallel)
                dst_bytes[offset:offset + length].copy_(slot[:length], non_blocking=True)
                self._release()
            return dst
        leaves = _staging_leaves(t)
        cls = type(t)
        if leaves is None or self.checked_classes.get(cls) is False or trust_pinned and builtins.all(leaf.is_pinned() for leaf in leaves):
            return p.to("cuda", non_blocking=True)
        nbytes = sum(_staging_nbytes(leaf) for leaf in leaves)
        if trust_pinned and nbytes < STAGING_MIN_SIZE or nbytes > STAGING_MAX_SIZE:
            return p.to("cuda", non_blocking=True)
        slot = self._acquire(nbytes)
        if slot is None:
            return p.to("cuda", non_blocking=True)
        q = self._pinned_copy(t, slot, 0, parallel)[0].to("cuda", non_blocking=True)
        self._release()
        return self.checked(p, q)

    def checked(self, p, q):
        # q: p moved from a pinned copy. The first tensor of each class is compared with the driver copy, which replaces it if they differ.
        cls = type(p.data if type(p) is torch.nn.Parameter else p)
        same = self.checked_classes.get(cls, None)
        if cls is torch.Tensor or same:
            return q
        if same is None:
            q_ref = p.to("cuda", non_blocking=True)
            same = self.checked_classes[cls] = _same_moved_tensor(q, q_ref)
            if not same:
                print(f"Staged transfers are disabled for tensors of type '{cls.__name__}' as they do not match a direct copy")
                return q_ref
        return q if same else p.to("cuda", non_blocking=True)


CACHED_BLOCK_VIEWS = True # swapped blocks move into fixed VRAM slots whose tensor views are built once per block and slot
RING_COPY_THREADS = 3 # copies into the staging ring: 2-4 threads reach the RAM bandwidth, more do not help
RING_DEFAULT_BLOCKS = 3 # default size of the staging ring, in largest blocks to stage
CPU_COMPUTE_MAX_ROWS = 8 # cpuCompute: the longest block calls (rows of their input) computed on the CPU, beyond which the block is streamed
CPU_COMPUTE_MIN_SIZE = 2**20 # cpuCompute: smaller weights stay in VRAM with the base of the model
CPU_COMPUTE_PINNING_RAM_SHARE = 0.25 # cpuComputePinning "auto": never more than this share of the RAM pinned for a model
CPU_COMPUTE_SAMPLES = 16 # cpuCompute: blocks computed on the CPU, at a row count, whose median time decides (after a pass of warm-up per model)
CPU_COMPUTE_TIMED = 4 # cpuCompute: one block computed on the CPU in this many is timed alone (the others start while the GPU ends the work before)
CPU_COMPUTE_CLIMB_PASSES = 5 # cpuCompute: short calls timed per count of streamed blocks tried (their median), after one left to settle
CPU_COMPUTE_CLIMB_GAIN = 0.03 # cpuCompute: a count of streamed blocks replaces the current one when its passes are this much shorter, and by twice their noise
CPU_COMPUTE_MIN_SHARE = 0.2 # cpuCompute: a CPU slower than the link is not tried at the rates' balance when that gives it less than this share
                            # of the blocks out of VRAM: overlapped perfectly, it would shorten their transfers by that share at most, less than
                            # the trial costs (MTP verification on a 4-core DDR4 PC: 1.4 against 12.5 GB/s, 10%)
CPU_COMPUTE_RECHECK = 256 # cpuCompute: short calls after which a settled choice is tried again (a count of streamed blocks against its neighbours, the CPU after it was slower than the transfers)
BASE_SMALL_MODULE_SIZE = STAGING_MIN_SIZE # modules of the blocks of the towers with fewer bytes of their own join the base of their model (0: none)
BASE_SMALL_MODULES_SHARE = 0.05 # unless they weigh more than this share of the towers
BASE_BLOCKS_OUTSIDE_TOWERS = False # in a model with main towers, the blocks outside them (a Sequential, a list of fewer layers than a tower) join its base
COVERING_BUDGET_BLOCKS = True # a budget covering a model with towers preloads every block, released under pressure (False: loaded whole, as a budget of 0)
PRELOAD_BLOCKS_OUTSIDE_TOWERS = True # a fixed budget preloads the blocks outside the towers first, as the dynamic preload ranks them (False: streamed at every pass)
_pcie_bytes_per_second = None


def _pcie_probe():
    # host to device bandwidth from pinned memory, measured once (best of 3 copies of 32 MB): low while blocks stream on the link, or when it
    # runs at a bad moment (3.4 instead of 26 GB/s at a load under VRAM pressure on the 4090, 3.3 to 12.3 GB/s on a PCIe 3 PC)
    global _pcie_bytes_per_second
    if _pcie_bytes_per_second is None:
        source, owner = pinned.block(32 << 20) # locked whatever the budget: unlocked once measured
        if owner is not None:
            owner.registration = pinned.lock(owner.base, source.data_ptr(), source.numel(), "probe")
            if owner.registration is None:
                raise RuntimeError("Unable to lock 32 MB of RAM to measure the transfer speed to the GPU")
        target = torch.empty_like(source, device="cuda")
        start, end, best = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), None
        for _ in range(3):
            start.record()
            target.copy_(source, non_blocking=True)
            end.record()
            end.synchronize()
            best = start.elapsed_time(end) if best is None else min(best, start.elapsed_time(end))
        _pcie_bytes_per_second = source.numel() / (best / 1000)
        del source, owner, target
    return _pcie_bytes_per_second

PCIE_TIMED_TRANSFERS = 8 # one block transfer in this many is timed on the transfer stream: the best recent ones give the link's rate
_pcie_timed = collections.deque(maxlen=64) # (bytes, start event, end event) of timed block transfers, the oldest dropped
_pcie_observed = 0.0


def _pcie_rate():
    # bytes per second from host to device: the probe's figure, or the best block transfer timed so far if faster (blocks cross one at a
    # time on the transfer stream, from pinned memory or the staging ring)
    global _pcie_observed
    while _pcie_timed and _pcie_timed[0][2].query():
        nbytes, start, end = _pcie_timed.popleft()
        _pcie_observed = builtins.max(_pcie_observed, nbytes / (start.elapsed_time(end) / 1000))
    return builtins.max(_pcie_probe(), _pcie_observed)


def _call_rows(args, kwargs):
    # rows of a block call: those of its first tensor argument (or the first tensor of a list argument), its last dimension apart
    for value in (*args, *kwargs.values()):
        if isinstance(value, (list, tuple)):
            value = next((item for item in value if torch.is_tensor(item)), None)
        if torch.is_tensor(value):
            return value.numel() // max(1, value.shape[-1])
    return 0

class _PassClimb:
    # cpuCompute: how many of the blocks out of VRAM the CPU computes during short calls. The rates' balance leaves out the GPU work of the
    # other blocks, which also lies on the CPU's path (a block computed on the CPU waits for the work queued before it): the count moves to a
    # neighbour while the median of the timed passes drops, tries both directions, settles, and tries again later. It is kept when the blocks
    # out of VRAM change (the dynamic preload's plans), only its passes timed again.
    def __init__(self):
        self.value, self.low, self.high = 0, 0, 0
        self.trial, self.base, self.direction, self.failed = None, None, -1, set()  # streaming more first: the balance leaves the CPU's path short
        self.jump, self.jumping = 0, False  # a first trial this far (all streamed from the balance, or the balance from all streamed), before the neighbours
        self.samples, self.skip, self.idle, self.noise = [], 1, 0, 0
        self.anchor = None  # (blocks out of VRAM, candidates to stream) when its passes were last timed

    def rebase(self):
        # the blocks out of VRAM changed: a trial running is dropped, the jump included (it runs once per row count, a jump not started yet
        # stays), and the passes are timed again
        self.trial, self.jumping, self.samples, self.skip = None, False, [], 1

    def current(self):
        return self.value if self.trial is None else self.trial

    def bound(self, low, high):
        self.low, self.high = low, high
        self.value = min(max(self.value, low), high)
        if self.trial is not None and not low <= self.trial <= high:
            self.trial = None
            self.failed.add(self.direction)

    def add(self, seconds):
        # a pass timed at current()
        if self.skip:  # the first pass of a count still uses blocks prefetched for the previous one
            self.skip -= 1
            return False
        if self.trial is None and len(self.failed) == 2:
            self.idle += 1
            if self.idle < CPU_COMPUTE_RECHECK:
                return False
            self.idle, self.failed = 0, set()
        self.samples.append(seconds)
        if len(self.samples) < CPU_COMPUTE_CLIMB_PASSES:
            return False
        median, before = sorted(self.samples)[len(self.samples) // 2], self.current()
        noise = sorted(abs(seconds - median) for seconds in self.samples)[len(self.samples) // 2] / median  # median absolute deviation, relative
        self.samples = []
        if self.trial is None:
            self.base, self.noise = median, noise
        elif median < self.base * (1 - builtins.max(CPU_COMPUTE_CLIMB_GAIN, 2 * builtins.max(noise, self.noise))):
            self.value, self.base, self.noise, self.failed = self.trial, median, noise, set() if self.jumping else {-self.direction}
        elif not self.jumping: # a jump that loses leaves both neighbours to try
            self.failed.add(self.direction)
            self.direction = -self.direction
        self.trial, self.jumping = None, False
        if self.jump and self.low <= self.value + self.jump <= self.high:
            self.trial, self.jump, self.jumping, self.skip = self.value + self.jump, 0, True, 1
            return True
        self.jump = 0
        while self.direction not in self.failed:
            if self.low <= self.value + self.direction <= self.high:
                self.trial = self.value + self.direction
                break
            self.failed.add(self.direction)
            self.direction = -self.direction
        self.skip = 1
        return self.current() != before


class _RingCancelled(Exception):
    pass

RING_RUN_GAP = 4095 # tensors at most this apart in memory are copied together with the bytes between them, which lie in their pages

def _ring_layout(items, by_storage=False):
    # Places the plain inner tensors of items in one buffer, each at a BLOCK_ALIGN boundary like a separate CUDA allocation (GPU kernels
    # rely on it). Tensors stored at such distances from each other, as in the pinned tensors of mmgp, form runs that one copy moves. With
    # by_storage, a run stays within one storage, which a tensor can then view. Returns the runs (source, size, offset in the buffer, a
    # tensor of the run), the offset of each tensor by address and the size of the buffer.
    leaves = {}
    for _, t in items:
        for leaf in _staging_leaves(t):
            leaves.setdefault((leaf.untyped_storage().data_ptr() if by_storage else 0, leaf.data_ptr(), leaf.numel() * leaf.element_size()), leaf)
    return _packed_layout(leaves)

def _packed_layout(leaves):
    # _ring_layout of {(storage key, address, bytes): tensor or None}
    runs, offsets, end = [], {}, 0
    for (storage, ptr, nbytes), leaf in sorted(leaves.items(), key=lambda entry: entry[0]):
        if not runs or runs[-1][4] != storage or (ptr - runs[-1][0]) % BLOCK_ALIGN or ptr > runs[-1][0] + runs[-1][1] + RING_RUN_GAP:
            runs.append([ptr, 0, _block_aligned(end), leaf, storage])
        run = runs[-1]
        run[1] = max(run[1], ptr + nbytes - run[0])
        offsets[ptr] = run[2] + ptr - run[0]
        end = run[2] + run[1]
    return [tuple(run[:4]) for run in runs], offsets, _block_aligned(end)

def _leaf_sizes(p):
    # sizes of the plain tensors of a parameter or buffer, aligned as pinning packs them
    sub_tensors = _get_quantized_subtensors(p)
    return [_block_aligned(t.numel() * t.element_size()) for _, t in sub_tensors] if sub_tensors else [_block_aligned(p.data.numel() * p.data.element_size())]

def _copy_runs(base, runs, helpers):
    # copies the runs of a ring layout to base, a large run split evenly across the RING_COPY_THREADS helpers; weights that a drive must read
    # (RAM cannot keep the model) are read ahead in pieces, the next piece while the threads copy the current one from RAM, instead of being
    # faulted in 64 KiB reads by the threads
    threads = RING_COPY_THREADS
    if not cold_ranges([(src, size) for src, size, _, _ in runs if size >= STAGING_PARALLEL_MIN_SIZE]):
        futures = []
        for src, size, offset, _ in runs:
            if size < STAGING_PARALLEL_MIN_SIZE:
                ctypes.memmove(base + offset, src, size)
            else:
                part = (size + threads - 1) // threads
                futures += [helpers.submit(ctypes.memmove, base + offset + start, src + start, min(part, size - start)) for start in range(0, size, part)]
        for future in futures:
            future.result()
        return
    pieces = [(base + offset + start, src + start, min(COLD_PIECE, size - start)) for src, size, offset, _ in runs for start in range(0, size, COLD_PIECE)]
    prefetch_ranges([(pieces[0][1], pieces[0][1] + pieces[0][2])])
    for index, (dst, src, size) in enumerate(pieces):
        part = size if size < STAGING_PARALLEL_MIN_SIZE else (size + threads - 1) // threads
        futures = [helpers.submit(ctypes.memmove, dst + start, src + start, min(part, size - start)) for start in range(0, size, part)]
        if index + 1 < len(pieces):
            _, next_src, next_size = pieces[index + 1]
            prefetch_ranges([(next_src, next_src + next_size)])
        for future in futures:
            future.result()

PRELOAD_PRIORITY = ("transformer", "transformer2") # the co-tenants that take the VRAM preload first (then the others in their order)


def _cotenant_groups(model_ids, cotenants):
    """The groups of models kept in VRAM together: the maximal sets whose models are all co-tenants of each other (both ways; a model that
    may stay beside any other, "*", joins none), of two models or more."""
    friends = {a: {b for b in model_ids if b != a and isinstance(cotenants.get(a), (list, tuple)) and b in cotenants[a]
                   and isinstance(cotenants.get(b), (list, tuple)) and a in cotenants[b]} for a in model_ids}
    groups = []

    def extend(group, candidates, excluded): # Bron-Kerbosch: a handful of models
        if not candidates and not excluded:
            if len(group) > 1:
                groups.append(group)
            return
        for model_id in list(candidates):
            extend(group | {model_id}, candidates & friends[model_id], excluded & friends[model_id])
            candidates = candidates - {model_id}
            excluded = excluded | {model_id}

    extend(set(), set(model_ids), set())
    return groups


def _preload_floors(model_ids, sizes, requested, preload, cotenants, priority):
    """The budget of each model once the VRAM preload applies (bytes; 0: the whole model). A model alone gets the larger of its budget
    and the preload (0 stays 0). The models of a group kept in VRAM together (_cotenant_groups) share the preload as a total instead: each
    keeps its own budget, and what the preload leaves beyond those goes to them in priority order (priority, else PRELOAD_PRIORITY, then
    their order), each up to its size; the first one also gets what remains beyond its size (every block preloaded: room for state kept
    in VRAM, mmgp.state_ring). A model in several groups gets the smallest of its shares.
    Also returns the share of the preload of each whole model (budget 0): its expert copies take what it leaves beyond the model."""
    floors = {model_id: requested[model_id] if requested[model_id] == 0 or preload == 0 else max(requested[model_id], preload) for model_id in model_ids}
    whole = {model_id: preload for model_id in model_ids if requested[model_id] == 0}
    if preload == 0:
        return floors, whole
    order = list(priority or PRELOAD_PRIORITY)
    shares = {}
    for group in _cotenant_groups(model_ids, cotenants):
        members = sorted(group, key=lambda model_id: (order.index(model_id) if model_id in order else len(order), model_ids.index(model_id)))
        held = {model_id: sizes[model_id] if requested[model_id] == 0 else min(requested[model_id], sizes[model_id]) for model_id in members}
        remaining = preload - builtins.sum(held.values())
        for model_id in members:
            extra = max(0, min(remaining, sizes[model_id] - held[model_id]))
            held[model_id] += extra
            remaining -= extra
        if remaining > 0:
            held[members[0]] += remaining
        for model_id in members:
            shares[model_id] = min(shares.get(model_id, held[model_id]), held[model_id])
    for model_id, share in shares.items():
        floors[model_id] = 0 if requested[model_id] == 0 else max(requested[model_id], share)
        if requested[model_id] == 0:
            whole[model_id] = share
    return floors, whole


def _evenly(entries):
    # entries reordered so that every prefix of them is spread evenly (bit-reversed positions): a pinning that stops where the reserved RAM
    # runs out, and the pinned pool that takes back the last ones first while their model does not run, both leave the pinned blocks spread
    bits = builtins.max(1, (len(entries) - 1).bit_length())
    return [entry for _, entry in sorted((int(format(i, f"0{bits}b")[::-1], 2), entry) for i, entry in enumerate(entries))]

def _spread(entries, ratio):
    # an even share of each run of numbered blocks: the n-th block of a run is kept iff int((n + 1) * ratio) > int(n * ratio)
    runs = {}
    for entry in entries:
        pre, num = _extract_num_from_str(entry)
        runs.setdefault(pre, []).append((num, entry))
    return [entry for run in runs.values() for n, (_, entry) in enumerate(sorted(run)) if int((n + 1) * ratio) > int(n * ratio)]

def _ring_view(t, base, offsets, start):
    # t rebuilt on views of base (a ring region or a VRAM buffer) from start, at the offsets of its plain inner tensors
    if type(t) is torch.Tensor:
        offset = start + offsets[t.data_ptr()]
        return base[offset:offset + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
    names, meta = t.__tensor_flatten__()
    return type(t).__tensor_unflatten__({name: _ring_view(getattr(t, name), base, offsets, start) for name in names}, meta, t.size(), t.stride())

class _StagingRing:
    """Pinned ring through which the blocks that are not pinned reach the GPU. A copier thread copies the next blocks into it, in their
    order of use and as far ahead as the ring holds them: this CPU copy, bound by the RAM bandwidth, overlaps the processing of the blocks
    before them, and their transfers then run at pinned speed. Each block takes one region of the ring, freed in block order once the
    transfer that read it has completed. The ring is normal RAM locked in place, much faster than allocating pinned memory."""

    def __init__(self, size):
        self.size = size
        self.regions = collections.deque() # [start, end, event of the transfer that reads it, None until issued], in block order
        self.cond = threading.Condition()
        self.jobs = collections.OrderedDict() # entry name -> future of (region, parameters, layout), in block order
        self.cancelled = False
        device = torch.cuda.current_device()
        self.copier = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmgp_ring", initializer=torch.cuda.set_device, initargs=(device,))
        self.helpers = ThreadPoolExecutor(max_workers=RING_COPY_THREADS, thread_name_prefix="mmgp_ring_copy")
        with torch.inference_mode(False):
            # pages of its own (a large numpy allocation), back to the system when the ring goes: the RAM allocator would keep the block in its
            # cache (rings come and go with the pinned pool), and PyTorch's allocator may put other tensors on its last page
            ram = numpy.empty(size + 4096, dtype=numpy.uint8)
            start = -ram.ctypes.data % 4096
            self.buffer = torch.from_numpy(ram[start:start + size])
        # locked by the copier thread, so that a failure leaves its CUDA error pending in that thread and not in the main thread
        self.registration = pinned.Registration(self.buffer, self.buffer.data_ptr(), size, device, kind="ring")
        try:
            error = self.copier.submit(self.registration.register).result()
            if int(error) != 0:
                raise RuntimeError(torch.cuda.cudart().cudaGetErrorString(error))
        except BaseException:
            self.copier.shutdown()
            self.helpers.shutdown()
            raise

    def _fit(self, nbytes):
        # start of a free range of nbytes after the last region, or None
        if not self.regions:
            return 0
        first, last_start, last_end = self.regions[0][0], self.regions[-1][0], self.regions[-1][1]
        if last_start >= first: # the regions occupy [first, last_end)
            return last_end if last_end + nbytes <= self.size else 0 if nbytes <= first else None
        return last_end if last_end + nbytes <= first else None # they occupy [first, size) and [0, last_end)

    def _reserve(self, nbytes):
        with self.cond:
            while True:
                if self.cancelled:
                    raise _RingCancelled()
                while self.regions and self.regions[0][2] is not None and self.regions[0][2].query():
                    self.regions.popleft()
                start = self._fit(nbytes)
                if start is not None:
                    region = [start, start + nbytes, None]
                    self.regions.append(region)
                    return region
                transfer = self.regions[0][2]
                if transfer is None:
                    self.cond.wait() # the transfer of the oldest region is not issued yet
                else:
                    self.cond.release()
                    try:
                        transfer.synchronize()
                    finally:
                        self.cond.acquire()

    def _stage(self, items, layout, pooled=None, state=()):
        if pooled is not None and (pooled.result() or layout[2] > self.size):
            return None, items, layout # it entered the pinned pool instead (its transfers read it there), or, refused, does not fit in the ring: moved as a block that is not staged
        nbytes = 0 if layout is None else layout[2]
        # the pageable state records attached to the block (mmgp.state_ring) follow its parameters in its region: region[3] says where
        runs = []
        for ring, block, source, generation in state:
            runs.append((source.data_ptr(), source.numel(), -(-nbytes // BLOCK_ALIGN) * BLOCK_ALIGN, (ring, block, generation)))
            nbytes = runs[-1][2] + source.numel()
        region = self._reserve(nbytes)
        if layout is not None:
            _copy_runs(self.buffer.data_ptr() + region[0], layout[0], self.helpers)
        if runs:
            _copy_runs(self.buffer.data_ptr() + region[0], runs, self.helpers)
            region.append([(ring, block, generation, offset, size) for _, size, offset, (ring, block, generation) in runs])
        return region, items, layout

    def schedule(self, entry_name, items, layout, pooled=None, state=()):
        # pooled: the future of the block's entry into the pinned pool, which replaces its copy into the ring (made if it could not enter);
        # state: the pageable state records attached to the block, staged after its parameters (layout None: the state alone, its
        # parameters being pinned)
        if entry_name not in self.jobs:
            self.jobs[entry_name] = self.copier.submit(self._stage, items, layout, pooled, state)

    def take(self, entry_name):
        # the region (None for a block now in the pinned pool), parameters and layout of a block about to be transferred, None if it was not
        # scheduled; the blocks scheduled before it are skipped and their regions freed
        if entry_name not in self.jobs:
            return None
        while True:
            name, future = self.jobs.popitem(last=False)
            if name == entry_name:
                return future.result()
            self.release(future.result()[0], torch.cuda.Event()) # an event never recorded counts as completed

    def release(self, region, transfer):
        if region is None:
            return
        with self.cond:
            region[2] = transfer
            self.cond.notify_all()

    def reset(self):
        # drops the staged blocks; the caller has synchronized the device, so no transfer reads the ring anymore
        with self.cond:
            self.cancelled = True
            self.cond.notify_all()
        for future in self.jobs.values():
            future.cancel()
        for future in self.jobs.values():
            try:
                future.result()
            except Exception:
                pass
        self.jobs.clear()
        with self.cond:
            self.regions.clear()
            self.cancelled = False

    def close(self):
        if self.buffer is None:
            return
        self.reset()
        try:
            if not self.registration.close():
                raise RuntimeError(f"Unable to release the staging ring: {self.registration.error}")
        finally:
            self.copier.shutdown(wait=True)
            self.helpers.shutdown(wait=True)
        self.buffer = None


PAGE_SIZE = 4096
POOL_RAM_FLOOR = max(4 << 30, physical_memory // 10) # the pinned pool grows while the system keeps this much RAM available (as the VRAM allocator before it spills)
POOL_DROPS_IDLE_PAGES = True # comparison switch: below that floor, when a model loads, is pinned or enters the pool, the checkpoint pages of the models of the other offload objects that do not run leave the working set (_drop_idle_pages)
POOL_RECLAIMS_PINNED = True # comparison switch: what is still short then comes from the RAM pinned for good of the idle models of other offload objects (_reclaim_pinned)
DROP_ONCE_PER_GENERATION_PAGES = True # after each transfer of a block of a model that runs once or twice per generation, the checkpoint pages it read leave the working set (when the RAM is short)
ONCE_PER_GENERATION_MODELS = ("text_encoder", "text_encoder_2", "vae") # unless the model sets _mm_pinned_pool (e.g. a language model registered as a text encoder)
DROP_UNREFERENCED_PAGES = True # comparison switch: once the models are profiled, the checkpoint pages that nothing reads any more (sources of the copies made at load) leave the working set

if os.name == "nt":
    class _MemoryInformation(ctypes.Structure):
        _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p), ("AllocationProtect", ctypes.c_ulong), ("PartitionId", ctypes.c_ushort),
                    ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong), ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong)]
    _virtual_query = ctypes.WinDLL("kernel32", use_last_error=True).VirtualQuery
    _virtual_query.argtypes = (ctypes.c_void_p, ctypes.POINTER(_MemoryInformation), ctypes.c_size_t)
    _virtual_query.restype = ctypes.c_size_t
    _virtual_unlock = ctypes.WinDLL("kernel32", use_last_error=True).VirtualUnlock
    _virtual_unlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    _mapped_file_name = ctypes.WinDLL("psapi", use_last_error=True).GetMappedFileNameW
    _mapped_file_name.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong)

def _drop_mapped_pages(ranges):
    # the pages of a checkpoint mapping that a copy has read stay in the process' working set beside the copy, counted as used RAM, until
    # Windows trims them under pressure (a read-only mapping included): dropped at once (VirtualUnlock on pages that are not locked), they
    # wait on the standby list, available, and come back without reading the drive if the mapping is read again soon. On Linux the available
    # memory already counts the pages of mapped files. About 75 ms per GiB of pages read, whatever the call (unmapping the view costs as
    # much): the system takes each page out of the working set. On a thread of its own it was slower, the copy threads reading the next
    # blocks waiting for it
    if os.name == "nt":
        for start, end in ranges:
            _virtual_unlock(start, end - start)

def _mapped_pages(leaves):
    # the pages of the tensors (address, bytes) mapped from a checkpoint file, to drop once copied: rounded out to whole pages, a range per
    # run of tensors that follow each other in a view of the file (one call each). A page shared with a tensor not read yet, dropped, comes
    # back from the standby list without reading the drive when that tensor is read (and is dropped again with it). Windows only
    if os.name != "nt":
        return []
    info, ranges, region = _MemoryInformation(), [], (0, 0, None)
    for pointer, nbytes in sorted(leaves):
        if not region[0] <= pointer < region[1]:
            found = _virtual_query(pointer, ctypes.byref(info), ctypes.sizeof(info)) != 0
            region = (info.BaseAddress, info.BaseAddress + info.RegionSize, info.AllocationBase if info.Type != 0x20000 else None) if found else (0, 0, None) # not MEM_PRIVATE: its view
        if region[2] is None or nbytes == 0:
            continue
        start, end = pointer // PAGE_SIZE * PAGE_SIZE, (pointer + nbytes + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
        if ranges and ranges[-1][2] == region[2] and start <= ranges[-1][1]: # one call never spans two views
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end, region[2]])
    return [(start, end) for start, end, _ in ranges]

def _tensor_spans(t):
    # (address, bytes) of the memory a CPU tensor reads, through the inner tensors of a traceable tensor subclass
    if type(t) is torch.Tensor:
        if t.device.type != "cpu" or t.numel() == 0:
            return []
        return [(t.data_ptr(), (1 + builtins.sum((size - 1) * stride for size, stride in builtins.zip(t.shape, t.stride()))) * t.element_size())]
    if not hasattr(t, "__tensor_flatten__"):
        return []
    return [span for name in t.__tensor_flatten__()[0] for span in _tensor_spans(getattr(t, name))]

def _unreferenced_pages(spans):
    # the pages of the checkpoint views that the tensors (address, bytes) read, outside all of them: read at load for copies (fused
    # projections, conversions, tables copied into locked RAM, pinned copies), never read again, yet in the working set until Windows trims
    # them. A page shared with a tensor stays. Dropping pages that are not in the working set costs almost nothing. Windows only
    if os.name != "nt":
        return []
    info, views, region = _MemoryInformation(), {}, (0, 0, None)
    for pointer, nbytes in sorted(spans):
        if not region[0] <= pointer < region[1]:
            found = _virtual_query(pointer, ctypes.byref(info), ctypes.sizeof(info)) != 0
            region = (info.BaseAddress, info.BaseAddress + info.RegionSize, info.AllocationBase if info.Type == 0x40000 else None) if found else (0, 0, None) # MEM_MAPPED: its view
        if region[2] is not None and nbytes:
            views.setdefault(region[2], []).append((pointer // PAGE_SIZE * PAGE_SIZE, (pointer + nbytes + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE))
    ranges, name = [], ctypes.create_unicode_buffer(260)
    for base, used in views.items():
        if not _mapped_file_name(ctypes.c_void_p(-1), base, name, 260): # a view of a file, not of shared memory
            continue
        end = base
        while _virtual_query(end, ctypes.byref(info), ctypes.sizeof(info)) and info.AllocationBase == base:
            end = info.BaseAddress + info.RegionSize
        position = base
        for start, stop in sorted(used):
            if start > position:
                ranges.append((position, start))
            position = builtins.max(position, stop)
        if position < end:
            ranges.append((position, end))
    return ranges

def _private_memory(pointers):
    # whether each address (in increasing order) lies in RAM the process allocated (PyTorch's allocators, the heap), rather than in a mapping
    # of a file (a checkpoint). One query per region of memory: each call from a thread waits for the GIL on return, a few ms while the main
    # thread runs a model
    if os.name == "nt":
        info, private, region = _MemoryInformation(), [], (0, 0, False)
        for pointer in pointers:
            if not region[0] <= pointer < region[1]:
                found = _virtual_query(pointer, ctypes.byref(info), ctypes.sizeof(info)) != 0
                region = (info.BaseAddress, info.BaseAddress + info.RegionSize, info.Type == 0x20000) if found else (0, 0, False) # MEM_PRIVATE
            private.append(region[2])
        return private
    regions = []
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split()
            start, end = (int(value, 16) for value in fields[0].split("-"))
            regions.append((start, end, fields[4] == "0")) # no inode: anonymous memory
    regions.sort()
    starts = [region[0] for region in regions]
    found = [bisect.bisect_right(starts, pointer) - 1 for pointer in pointers]
    return [index >= 0 and pointer < regions[index][1] and regions[index][2] for index, pointer in zip(found, pointers)]


class _PoolEntry:
    # a block in the pinned pool: its locked RAM (registrations), the pinned sources of its VRAM buffer as [(offset, uint8 tensor)] for the
    # layout it was made for, the bytes it locks, its page ranges locked in place and its plain tensors as (address, bytes)
    __slots__ = ("model_id", "registrations", "sources", "layout", "nbytes", "in_place", "leaves")

    def __init__(self, model_id, registrations, sources, layout, nbytes, in_place, leaves):
        self.model_id, self.registrations, self.sources, self.layout, self.nbytes, self.in_place, self.leaves = model_id, registrations, sources, layout, nbytes, in_place, leaves

    def close(self):
        # unlocks (waits for the work queued on the GPU) and frees its buffer; tensors locked in place stay the model's, in pageable RAM
        for registration in self.registrations:
            registration.close()
        self.registrations, self.sources = [], None


# How a block enters the pinned pool, made once for its layout: the page ranges to lock in place [(start, end, storages)] and their sources
# [(offset in the block's VRAM buffer, uint8 tensor)], the runs copied into a buffer of locked RAM of its own (_copy_runs), the size of that
# buffer, its pieces [(offset in the VRAM buffer, offset in the buffer, bytes)], the bytes locked and the plain tensors as (address, bytes)
_PoolPlan = collections.namedtuple("_PoolPlan", "layout ranges sources runs copy_nbytes pieces nbytes leaves mapped") # mapped: the pages of the
# tensors mapped from a checkpoint file [(start, end)] (_mapped_pages), dropped from the working set once copied


class _PinnedPool:
    """Reserved RAM shared by the blocks that the running models stream at each pass (offload.all(pinPool=...)), instead of RAM pinned for
    good for each model, which a model keeps even while it does not run. A block enters when it is first transferred (or, during the short
    calls of cpuCompute, when the balance of the CPU and the PCIe link wants it streamed): the pages that its tensors in private RAM (made at
    load: concatenations, quantization) fill are locked in place, the rest (tensors mapped from a checkpoint file, whose locking in place is
    slow) copied into locked RAM of its own. Its transfers then read pinned memory: full speed, no copy through the staging ring. The pool takes the
    blocks out of VRAM spread evenly, up to the reserved RAM that the pinned models and the staging rings leave, while the system keeps
    POOL_RAM_FLOOR available; the blocks it cannot take cross the staging ring. Entries leave only at quiet points (unlocking waits for all
    the work queued on the GPU): when a model is loaded, the entries of the models that do not run go if its own needs and theirs exceed the
    room or the available RAM is short, and those of the other models of its offload object as far as they would make the object's entries
    larger than those of its largest model (pinning that model for good would lock as much); the models unloaded with the RAM short; an
    offload object released. Shared by the offload objects of the process (e.g. a language model and an image model)."""

    def __init__(self):
        self.entries = weakref.WeakKeyDictionary() # offload object -> {entry name: _PoolEntry}
        self.used = weakref.WeakKeyDictionary() # offload object -> {model id: when it was last loaded}
        self.wholes = weakref.WeakKeyDictionary() # offload object -> {model id: bytes of all its blocks the pool takes, when it was last loaded}
        self.pending = {} # (id of the offload object, entry name) -> (future of its preparation, bytes it may lock)
        self.guard = threading.Lock()
        self.executor = self.helpers = None
        self.version = 0 # changes when an entry is added or leaves
        self.epoch = 0 # changes at each quiet point
        self.full = False # a preparation found no room (reserved RAM, available RAM, a lock refused): no growth until the next quiet point
        self.clock = 0
        self.dropped_epoch = -1 # the quiet point after which the idle models of other offload objects last dropped their pages (_drop_idle_pages)
        self.shortfalls = weakref.WeakKeyDictionary() # offload object -> bytes of blocks of its running models refused since its last quiet point (reclaim_for)

    def entry(self, manager, entry_name):
        entries = self.entries.get(manager)
        return None if entries is None else entries.get(entry_name)

    def involved(self, manager, entry_name):
        return self.entry(manager, entry_name) is not None or (id(manager), entry_name) in self.pending

    def reserved(self):
        with self.guard:
            return builtins.sum(nbytes for _, nbytes in self.pending.values())

    def stats(self):
        # {model id: (blocks, bytes locked, of which in place)} of the entries (any thread)
        result = {}
        with self.guard:
            entries = [entry for entries in list(self.entries.values()) for entry in entries.values()]
        for entry in entries:
            blocks, nbytes, in_place = result.get(entry.model_id, (0, 0, 0))
            result[entry.model_id] = (blocks + 1, nbytes + entry.nbytes, in_place + builtins.sum(end - start for start, end in entry.in_place))
        return result

    @staticmethod
    def plan(items, layout):
        """How a block enters the pool (made once, at profile time: on the pool's thread or while a model runs, each Python or torch call waits
        for the GIL, a few ms). In place, the whole pages that a tensor in private RAM fills: a page it shares with other objects is never
        locked (its later tenants, e.g. short-lived tensors, would pass for pinned memory, and their copies to the GPU would no longer wait
        for the transfer before the memory is reused). Copied into locked RAM of its own: the tensors mapped from a checkpoint file (locking
        them in place is slow) and the ends of the others. Parts that lie the same distance apart in their source and in the block's VRAM
        buffer (layout) are copied together."""
        leaves = {}
        for _, t in items:
            for leaf in _staging_leaves(t):
                leaves.setdefault((leaf.data_ptr(), leaf.numel() * leaf.element_size()), leaf)
        keys = sorted(leaves)
        ranges, in_place, copied, mapped = [], [], [], [] # [start, end, storages]; [offset in the VRAM buffer, address, bytes, storage]; (address of the tensor, of the part, bytes)
        for key, private in builtins.zip(keys, _private_memory([pointer for pointer, _ in keys])):
            start, end = (key[0] + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE, (key[0] + key[1]) // PAGE_SIZE * PAGE_SIZE
            if not private or end <= start:
                copied.append((key[0], key[0], key[1]))
                if not private:
                    mapped.append(key)
                continue
            storage = leaves[key].untyped_storage()
            if ranges and start == ranges[-1][1]:
                ranges[-1][1] = end
                ranges[-1][2].append(storage)
            else:
                ranges.append([start, end, [storage]])
            in_place.append([layout[1][key[0]] + start - key[0], start, end - start, storage])
            copied += [part for part in ((key[0], key[0], start - key[0]), (key[0], end, key[0] + key[1] - end)) if part[2]]
        runs, offsets, copy_nbytes = _packed_layout({(0, part, nbytes): None for _, part, nbytes in copied})
        def merged(pieces, same):  # pieces sorted by offset in the VRAM buffer: [offset, source, bytes, owner]
            out = []
            for piece in sorted(pieces, key=lambda piece: piece[:3]):
                last = out[-1] if out else None
                if last is not None and same(last, piece) and 0 <= piece[1] - last[1] - last[2] == piece[0] - last[0] - last[2] <= RING_RUN_GAP:
                    last[2] = piece[1] + piece[2] - last[1]
                else:
                    out.append(piece)
            return out
        with torch.inference_mode(False):
            sources = [(offset, torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage, src - storage.data_ptr(), (nbytes,)))
                       for offset, src, nbytes, storage in merged(in_place, lambda a, b: a[3].data_ptr() == b[3].data_ptr())]
        pieces = merged([[layout[1][pointer] + part - pointer, offsets[part], nbytes, None] for pointer, part, nbytes in copied], lambda a, b: True)
        copy_nbytes = (copy_nbytes + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE if copied else 0
        return _PoolPlan(layout, [(start, end, storages) for start, end, storages in ranges], sources, runs, copy_nbytes, [tuple(piece[:3]) for piece in pieces],
                         builtins.sum(end - start for start, end, _ in ranges) + copy_nbytes, tuple(keys), _mapped_pages(mapped))

    def prepare(self, manager, model_id, entry_name, plan):
        """The future of a block's entry, made on the pool's thread from its plan: True once it is in, False if it could not enter."""
        key = (id(manager), entry_name)
        with self.guard:
            found = self.pending.get(key)
            if found is None:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmgp_pool", initializer=torch.cuda.set_device, initargs=(torch.cuda.current_device(),))
                    self.helpers = ThreadPoolExecutor(max_workers=RING_COPY_THREADS, thread_name_prefix="mmgp_pool_copy")
                found = self.pending[key] = (self.executor.submit(self._prepare, manager, model_id, entry_name, plan, manager.max_reservable_memory), plan.nbytes)
        return found[0]

    def _prepare(self, manager, model_id, entry_name, plan, cap):
        # on the pool's thread: the copy and the locks of a block, refused if the reserved RAM, the available RAM or a lock refuses it (e.g. 712,
        # a page already locked by someone else)
        registrations, device = [], torch.cuda.current_device()
        try:
            reserved = pinned.fits(plan.nbytes, cap)
            if reserved and psutil.virtual_memory().available - plan.copy_nbytes < POOL_RAM_FLOOR:
                self._drop_idle_pages(manager, POOL_RAM_FLOOR + plan.copy_nbytes)
            if reserved and psutil.virtual_memory().available - plan.copy_nbytes >= POOL_RAM_FLOOR:
                for start, end, storages in plan.ranges:
                    registration = pinned.Registration(storages, start, end - start, device, kind="pool")
                    if int(registration.register()) != 0:
                        break
                    registrations.append(registration)
                else:
                    if plan.copy_nbytes:
                        buffer, owner = pinned.block(plan.copy_nbytes)
                        _copy_runs(buffer.data_ptr(), plan.runs, self.helpers)
                        _drop_mapped_pages(plan.mapped)
                        registration = pinned.Registration(owner, buffer.data_ptr(), plan.copy_nbytes, device, kind="pool")
                        if int(registration.register()) == 0:
                            registrations.append(registration)
            if len(registrations) == len(plan.ranges) + bool(plan.copy_nbytes):
                with torch.inference_mode(False):
                    sources = plan.sources + [(offset, buffer[start:start + nbytes]) for offset, start, nbytes in plan.pieces]
                entry = _PoolEntry(model_id, registrations, sources, plan.layout, plan.nbytes, [(start, end) for start, end, _ in plan.ranges], plan.leaves)
                with self.guard:
                    self.entries.setdefault(manager, {})[entry_name] = entry
                    self.version += 1
                return True
            for registration in registrations: # refused: nothing of the block stays locked (a rare wait for the GPU)
                registration.close()
            if not self.full and manager.verboseLevel >= 1:
                tqdm.write(f"Pinned pool full ({builtins.sum(value[1] for value in self.stats().values())/ONE_MB:0.0f} MB): the other blocks of '{model_id}' cross the staging ring")
            with self.guard:
                self.shortfalls[manager] = self.shortfalls.get(manager, 0) + plan.nbytes
            self.full = True
            return False
        finally:
            with self.guard:
                del self.pending[(id(manager), entry_name)]

    def release(self, manager, entry_names):
        # at a quiet point: the entries leave (each unlock waits for the work queued on the GPU)
        with self.guard:
            entries = self.entries.get(manager, {})
            leaving = [entries.pop(entry_name) for entry_name in entry_names]
        for entry_name, entry in builtins.zip(entry_names, leaving):
            entry.close()
            manager.staged_entries.pop(entry_name, None)
        self.version += 1

    def make_room(self, manager, model_id=None, need=0, whole=0, ring=0, ram=0):
        """A quiet point (the GPU has finished the work of the models unloaded), before model_id runs if not None: the entries of the models
        that do not run leave, those of the models used least recently first and their last entered first (those left stay spread evenly),
        while the room left is short of need (the bytes the pool may still lock for model_id) and ring (those of the staging ring its blocks
        cross until they enter), the system's available RAM is under POOL_RAM_FLOOR or, for the other models of manager, its entries and need
        exceed the largest whole (the bytes of all the blocks the pool takes for a model, model_id's being whole): a text encoder's entries
        leave when its transformer runs, a few of the transformer's when the text encoder runs again. Entries whose block's tensors were
        replaced leave too. Then, if the available RAM is short of need, ring and ram (the checkpoint pages its transfers will read), whatever
        the memory profile, the idle models of the other offload objects drop their checkpoint pages (_drop_idle_pages)."""
        self.full, self.epoch = False, self.epoch + 1
        pinned.collect() # the locked buffers retired since the last quiet point (outgrown staging slots) are unlocked
        largest = math.inf
        if model_id is not None:
            self.clock += 1
            self.used.setdefault(manager, {})[model_id] = self.clock
            wholes = self.wholes.setdefault(manager, {})
            wholes[model_id] = whole
            largest = builtins.max(wholes.values())
            stale = [entry_name for entry_name, entry in self.entries.get(manager, {}).items() if entry.model_id == model_id and not set(entry.leaves) <= manager._pool_leaves(entry_name)]
            if stale:
                self.release(manager, stale)
                manager.block_items = {key: found for key, found in manager.block_items.items() if key[0] not in stale}
        idle = {} # the models that do not run: not loaded, or loaded in a parked offload object (offload.park), which gives way to the others
        for other, entries in list(self.entries.items()):
            for entry_name, entry in entries.items():
                if (entry.model_id not in other.active_models_ids or other.parked) and (other is not manager or entry.model_id != model_id):
                    idle.setdefault((self.used.get(other, {}).get(entry.model_id, 0), id(other)), (other, entry.model_id, []))[2].append(entry_name)
        held = builtins.sum(entry.nbytes for entry in self.entries.get(manager, {}).values())
        released = {}
        for key in sorted(idle):
            other, other_model, entry_names = idle[key]
            for entry_name in reversed(entry_names):
                room = manager.max_reservable_memory - pinned.total_bytes - self.reserved()
                if room >= need + ring and psutil.virtual_memory().available >= POOL_RAM_FLOOR and (other is not manager or held + need <= largest):
                    break
                nbytes = self.entries[other][entry_name].nbytes
                self.release(other, [entry_name])
                released[(id(other), other_model)] = released.get((id(other), other_model), 0) + nbytes
                held -= nbytes if other is manager else 0
        if released and manager.verboseLevel >= 1:
            tqdm.write(f"Pinned pool: released {', '.join(f'{nbytes/ONE_MB:0.0f} MB of {other_model}' for (_, other_model), nbytes in released.items())}" + (f" for '{model_id}'" if model_id is not None else " (RAM short)"))
        if need + ring + ram and psutil.virtual_memory().available < POOL_RAM_FLOOR + need + ring + ram:
            self._drop_idle_pages(manager, POOL_RAM_FLOOR + need + ring + ram)
        if model_id is not None:
            with self.guard:
                refused = self.shortfalls.pop(manager, 0) # blocks refused while it last ran: the preload decides those out of VRAM during the run
            if need or refused:
                self._reclaim_pinned(manager, builtins.max(need + ring, refused))
        elif ring and not torch.cuda.is_current_stream_capturing(): # a staging ring made for blocks that stream (_create_ring)
            self._reclaim_pinned(manager, ring)

    def reclaim_for(self, manager):
        # On the main thread, at the transfer of a block of a running model of manager after blocks of it were refused for room (between two
        # quiet points: a model kept loaded from one request to the next has none): the reclaim of a quiet point, once (unlocking waits for
        # the work queued on the GPU), then the pool may grow again
        with self.guard:
            refused = self.shortfalls.pop(manager, 0)
        if refused and self._reclaim_pinned(manager, refused):
            self.full, self.epoch = False, self.epoch + 1

    def _reclaim_pinned(self, manager, need):
        # Still short of need after the idle entries and pages, at this quiet point: the buffers pinned for good of the idle models of the
        # other offload objects (e.g. a video model's transformer while a language model runs) give the rest, their last ranked first (the
        # blocks left pinned stay spread evenly). A buffer copied from a checkpoint is freed (RAM and reserved RAM); one made at load, only
        # unlocked (reserved RAM: its RAM stays the model's), while the reserved RAM alone is short. They come back when their model loads.
        if not POOL_RECLAIMS_PINNED:
            return 0
        budget_short = need - (manager.max_reservable_memory - pinned.total_bytes - self.reserved()) if manager.max_reservable_memory > 0 else 0
        ram_short = POOL_RAM_FLOOR + need - psutil.virtual_memory().available
        if budget_short <= 0 and ram_short <= 0:
            return 0
        victims = [(record.rank, other, model_id, record) for other in list(_managers) if other is not manager and other.models is not None
                   for model_id, model in other.models.items() if model_id not in other.active_models_ids
                   for record in getattr(model, "_mm_pinned_buffers", ()) if record.state == "locked"]
        reclaimed = {}
        with torch.inference_mode(False):
            for _, other, model_id, record in sorted(victims, key=lambda victim: victim[0], reverse=True):
                if budget_short <= 0 and ram_short <= 0:
                    break
                if ram_short > 0 and record.mapped:
                    record.free()
                    ram_short -= record.nbytes
                elif budget_short > 0:
                    record.unlock() # locked again in place at its next load, without a copy
                else:
                    continue
                budget_short -= record.nbytes
                reclaimed[(other, model_id)] = reclaimed.get((other, model_id), 0) + record.nbytes
        for other, model_id in reclaimed:
            other._pinning_changed(model_id)
        if reclaimed and manager.verboseLevel >= 1:
            tqdm.write(f"Pinned pool: {', '.join(f'{nbytes/ONE_MB:0.0f} MB pinned for {model_id}' for (_, model_id), nbytes in reclaimed.items())} given back while it does not run")
        return builtins.sum(reclaimed.values())

    def _drop_idle_pages(self, manager, target):
        # The available RAM is short for a model of manager (when it loads or is pinned, in any memory profile, or when its block is refused
        # by the pool): the checkpoint pages read by the models of the other offload objects that do not run, or are parked (e.g. a language
        # model kept loaded beside a video model, whose pages stay in the working set until Windows trims them, which it hardly does) leave
        # it, until target bytes are available, once per quiet point. On the standby list they count as available and come back without
        # reading the drive while they stay there. The idle models of manager are left to its own policy (_drops_pages). Windows only:
        # elsewhere the available RAM already counts these pages
        if not POOL_DROPS_IDLE_PAGES or self.dropped_epoch == self.epoch:
            return
        self.dropped_epoch = self.epoch
        before, dropped = psutil.virtual_memory().available, set()
        for other in [other for other in list(_managers) if other is not manager]:
            for entry in list(other.blocks_of_modules):
                model_id = entry.split("/", 1)[0]
                if model_id in other.active_models_ids and not other.parked or model_id in other.pages_dropped:  # dropped already, since it last ran
                    continue
                ranges = other._mapped_pages(entry)
                if ranges:
                    _drop_mapped_pages(ranges)
                    dropped.add((other, model_id))
                    if psutil.virtual_memory().available >= target:
                        break
            else:
                for model_id in {model_id for owner, model_id in dropped if owner is other}:
                    other.pages_dropped.add(model_id)  # all of its pages: none left to drop until it runs again (gpu_load)
                continue
            break
        if dropped and manager.verboseLevel >= 1:
            available = psutil.virtual_memory().available
            tqdm.write(f"The RAM is short: the checkpoint pages read by {', '.join(sorted(repr(model_id) for _, model_id in dropped))}, which do not run, "
                       f"left the working set: {(available - before) / ONE_MB:0.0f} MB more available ({available / ONE_MB:0.0f} MB)")

    def close(self, manager):
        # an offload object is released: its preparations end, its entries leave
        with self.guard:
            futures = [future for (owner, _), (future, _) in self.pending.items() if owner == id(manager)]
        for future in futures:
            future.result()
        with self.guard:
            leaving = self.entries.pop(manager, {})
        for entry in leaving.values():
            entry.close()
        self.used.pop(manager, None)
        self.wholes.pop(manager, None)
        self.version += 1

_pool = _PinnedPool()
_managers = weakref.WeakSet() # the offload objects of the process (_PinnedPool._drop_idle_pages)


def _plain_lora(lora_data):
    # LoRA and diff adapters have no effect at scaling 0 (their forwards skip them); DoRA and LoKr adapters steer the forward path
    return lora_data[3] is None and (len(lora_data) < 6 or lora_data[5].get("type", "lora") != "lokr")


def _prefetch_thread(device):
    # the prefetch thread: its VRAM shortages are released by the thread waiting for its transfers (memory_pressure.wait), which runs the blocks
    torch.cuda.set_device(device)
    memory_pressure.helper_thread()

def _run_in_grad_modes(inference_mode, grad_enabled, fn, *args):
    with torch.inference_mode(inference_mode), torch.set_grad_enabled(grad_enabled):
        return fn(*args)

def _pageable_nbytes(p):
    sub_tensors = _get_quantized_subtensors(p)
    tensors = [t for _, t in sub_tensors] if sub_tensors else [p.data if type(p) is torch.nn.Parameter else p]
    return sum(t.numel() * t.element_size() for t in tensors if not t.is_pinned())

def _extract_tie_weights_from_sd(sd , sd_name, verboseLevel =1):
    tied_weights = {}
    tied_weights_count = 0
    tied_weights_total = 0
    tied_weights_last = None
    ref_cache = {}

    for n, p in sd.items():
        ref = _get_tensor_ref(p)
        match = ref_cache.get(ref, None)
        if match != None:
            match_name, match_size = match
            tied_weights_count += 1
            tied_weights_total += match_size
            if verboseLevel >=1:
                tied_weights_last = f"{match_name} <-> {n}"
            tied_weights[n] = match_name
        else:
            length = torch.numel(p.data) * p.data.element_size() 
            ref_cache[ref] = (n, length)
        
    if verboseLevel >=2 and tied_weights_count > 0:
        if tied_weights_count == 1:
            print(f"Tied weights of {tied_weights_total/ONE_MB:0.2f} MB detected: {tied_weights_last}")
        else:
            print(f"Found {tied_weights_count} tied weights for a total of {tied_weights_total/ONE_MB:0.2f} MB, last : {tied_weights_last}")
    return tied_weights

def _pin_sd_to_memory(sd, sd_name, tied_weights = None, gig_tensor_size = BIG_TENSOR_MAX_SIZE, verboseLevel = 1):
    # the tensors of the state dicts (LoRAs) moved into buffers of locked RAM of their own, as far as the reserved RAM budget allows: each
    # tensor is swapped in place with its pinned copy (its tied names follow it), which frees its source
    names_list = sd_name if isinstance(sd, list) else [sd_name]

    if 0 < pinned.limit_bytes <= pinned.total_bytes:
        if  verboseLevel>=1 :
            print(f"Unable to pin data of '{','.join(names_list)}' to reserved RAM as there is no reserved RAM left. Transfer speed from RAM to VRAM will may be slower.")
        return

    
    if isinstance(sd, list):
        new_sd = {}
        for i, sub_sd,  in enumerate(sd):
            for k, v in sub_sd.items():
                new_sd[str(i) + "#" + k] =v
        sd = new_sd
        del new_sd
        sub_sd = None

    if isinstance(tied_weights, list):
        new_tied_weights = {}
        for i, sub_tied_weights,  in enumerate(tied_weights):
            for k, v in sub_tied_weights.items():
                new_tied_weights[str(i) + "#" + k] = str(i) + "#" + v
        tied_weights = new_tied_weights
        del new_tied_weights
        sub_tied_weights = None

    current_big_tensor_size = 0
    big_tensor_no  = 0
    big_tensors_sizes = []
    tensor_map_indexes = []
    total_tensor_bytes = 0

    for n, p in sd.items():
        if tied_weights == None or not n in tied_weights :
            length = torch.numel(p.data) * p.data.element_size() 

            if current_big_tensor_size + length > gig_tensor_size :
                big_tensors_sizes.append(current_big_tensor_size)
                current_big_tensor_size = 0
                big_tensor_no += 1

            itemsize = p.data.dtype.itemsize
            if current_big_tensor_size % itemsize:
                current_big_tensor_size += itemsize - current_big_tensor_size % itemsize
            tensor_map_indexes.append((big_tensor_no, current_big_tensor_size, length  ))
            current_big_tensor_size += length

            total_tensor_bytes += length
  
    big_tensors_sizes.append(current_big_tensor_size)

    big_tensors = []
    last_big_tensor = 0
    total = 0  
    incomplete_pinning = False
    cap = _get_max_reservable_memory(_get_perc_reserved_mem_max())
    for size in big_tensors_sizes:
        current_big_tensor = pinned.empty(size, "lora", cap)
        if current_big_tensor is None:
            incomplete_pinning = True
            print(f"Unable to pin more tensors for '{sd_name}' as the maximum reservable memory has been reached ({total/ONE_MB:.2f} MB). Transfer speed from RAM to VRAM may be slower.")
            break
        big_tensors.append(current_big_tensor)
        last_big_tensor += 1
        total += size

    tensor_no = 0
    tied = {} if tied_weights is None else tied_weights
    for n, p  in sd.items():
        if n not in tied:
            big_tensor_no, offset, length = tensor_map_indexes[tensor_no]
            if big_tensor_no>=0 and big_tensor_no < last_big_tensor:
                current_big_tensor = big_tensors[big_tensor_no]
                length = torch.numel(p.data) * p.data.element_size()
                q = _move_to_pinned_tensor(p.data, current_big_tensor, offset, length)
                torch.utils.swap_tensors(p, q)
                del q
            tensor_no += 1
        del p
    for n, q_name in tied.items(): # once their targets are pinned (or, beyond the budget, left where they were), wherever they come in sd
        sd[n].data = sd[q_name].data
    gc.collect()
    torch.cuda.empty_cache()


    if verboseLevel >=1:
        if incomplete_pinning :
            if len(names_list) > 1:
                print(f"'{','.join(names_list)}' were partially pinned to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")
            else:
                print(f"'{','.join(names_list)}' was partially pinned to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")
        else:
            if len(names_list) > 1:
                print(f"'{','.join(names_list)}' were pinned entirely to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")
            else:
                print(f"'{','.join(names_list)}' was pinned entirely to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")


    return 


def _pin_to_memory(model, model_id, partialPinning = False, pinnedPEFTLora = True, big_tensor_size = BIG_TENSOR_MAX_SIZE, perc_reserved_mem_max = 0,verboseLevel = 1, loading_callback=None, ranks = None):
    # ranks: id of each tensor to pin -> its priority (lowest first, pinning stops where the reserved RAM runs out); all of them if None

    if 0 < pinned.limit_bytes <= pinned.total_bytes:

        if  verboseLevel>=1 :
            print(f"Unable to pin data of '{model_id}' to reserved RAM as there is no reserved RAM left. Transfer speed from RAM to VRAM may be slower.")
        return
    
    if partialPinning and ranks is None:
        towers_names, _ = _detect_main_towers(model)


    perc_reserved_mem_max = _get_perc_reserved_mem_max(perc_reserved_mem_max)
    max_reservable_memory = _get_max_reservable_memory(perc_reserved_mem_max) 

    current_big_tensor_size = 0
    big_tensor_no  = 0
    big_tensors_sizes, big_tensors_keys, current_key = [], [], None # keys: (rank, copied from a checkpoint mapping) of each buffer, with ranks
    tensor_map_indexes = []
    total_tensor_bytes = 0

    params_dict = {} #  OrderedDict
    for k, sub_module in model.named_modules():
        include = True
        if partialPinning and ranks is None:
            include = any(k.startswith(pre) for pre in towers_names)
        if include and not pinnedPEFTLora and ".lora_" in k:
            include = False

        if include:
            params_dict.update( { k + '.' + n : (p,  False) for n, p in sub_module.named_parameters(recurse=False) if (ranks is None or id(p) in ranks) and not _is_mapped(p) }  )
            params_dict.update( { k + '.' + n : (b,  True) for n, b in sub_module.named_buffers(recurse=False) if ranks is None or id(b) in ranks }  )
    origins = {}
    if ranks is not None:
        # blocks of their own (a rank is an entry): those copied from a checkpoint mapping first, then those made at load, in buffers apart
        # (reclaim: the first are freed and copied again from their views, the others only unlocked)
        leaves = {id(p): [tensor.data_ptr() for _, tensor in (_get_quantized_subtensors(p) or [(None, p.data)]) if tensor.numel()] for _, (p, _) in params_dict.items()}
        pointers = sorted({pointer for values in leaves.values() for pointer in values})
        private = dict(zip(pointers, _private_memory(pointers)))
        origins = {key: not builtins.all(private[pointer] for pointer in values) for key, values in leaves.items()}
        del leaves, pointers, private
        params_dict = dict(sorted(params_dict.items(), key=lambda item: (ranks[id(item[1][0])], not origins[id(item[1][0])])))

    if  verboseLevel>=1 :
        if partialPinning:
            if len(params_dict) == 0:
                print(f"Unable to apply Partial of '{model_id}' as no isolated main structures were found")
            else:
                print(f"Partial pinning of data of '{model_id}' to reserved RAM")
        else:
            print(f"Pinning data of '{model_id}' to reserved RAM")

    if len(params_dict) == 0:
        return

    ref_cache = {}
    tied_weights = {}
    tied_weights_count = 0
    tied_weights_total = 0
    tied_weights_last = None

    for n, (p, _) in params_dict.items():
        ref = _get_tensor_ref(p)
        match = ref_cache.get(ref, None)
        if match != None:
            match_name, match_size = match
            tied_weights_count += 1
            tied_weights_total += match_size
            if verboseLevel >=1:
                tied_weights_last = f"{match_name} <-> {n}"
            tied_weights[n] = match_name
        else:
            sub_tensors = _get_quantized_subtensors(p)
            if sub_tensors:
                if builtins.all(t.is_pinned() for _, t in sub_tensors):
                    params_dict[n] = (None, False)
                    del sub_tensors
                    continue
                length = builtins.sum(_block_aligned(t.numel() * t.element_size()) for _, t in sub_tensors) # each inner tensor aligned
            else:
                if p.data.is_pinned():
                    params_dict[n] = (None, False)
                    continue
                length = torch.numel(p.data) * p.data.element_size()

            ref_cache[ref] = (n, length)
            key = None if ranks is None else (ranks[id(p)], origins[id(p)])
            if current_big_tensor_size != 0 and (current_big_tensor_size + length > big_tensor_size or key != current_key):
                big_tensors_sizes.append(current_big_tensor_size)
                big_tensors_keys.append(current_key)
                current_big_tensor_size = 0
                big_tensor_no += 1
            current_key = key

            del sub_tensors
            current_big_tensor_size = _block_aligned(current_big_tensor_size) # aligned as a CUDA allocation: a block can move as one buffer
            tensor_map_indexes.append((big_tensor_no, current_big_tensor_size, length  ))
            current_big_tensor_size += length

            total_tensor_bytes += length
    if verboseLevel >=1 and tied_weights_count > 0:
        if  tied_weights_count == 1:
            print(f"Tied weights of {tied_weights_total/ONE_MB:0.2f} MB detected: {tied_weights_last}")
        else:
            print(f"Found {tied_weights_count} tied weights for a total of {tied_weights_total/ONE_MB:0.2f} MB, last : {tied_weights_last}")
              

    big_tensors_sizes.append(current_big_tensor_size)
    big_tensors_keys.append(current_key)
    if not pinned.LOCKS_RAM: # driver pinned memory: blocks of at least BIG_TENSOR_MIN_SIZE (locked RAM is exact)
        big_tensors_sizes = [max(size, BIG_TENSOR_MIN_SIZE) for size in big_tensors_sizes]

    if total_tensor_bytes == 0:
        return

    pinning_total = 0
    planned_bytes = pinned.total_bytes
    for size in big_tensors_sizes:
        planned_bytes += size
        if max_reservable_memory > 0 and planned_bytes >= max_reservable_memory:
            break
        pinning_total += 1

    big_tensors = []
    total = 0
    registered = 0
    

    failed_planned_allocation = False
    if loading_callback is not None: loading_callback.check_abort()
    gc.collect()
    dummy_pinned_tensor = None
    if not pinned.LOCKS_RAM: # driver pinned memory: RESERVED_RAM_MIN_AVAILABLE of it kept free for the pinned allocations made after
        try:
            dummy_pinned_tensor = torch.empty( RESERVED_RAM_MIN_AVAILABLE, dtype= torch.uint8, pin_memory=True, device="cpu")
        except (MemoryError, RuntimeError) as error:
            flush_torch_caches()
            print(f"Unable to allocate the reserved RAM probe: {error}")
            return

    last_allocated_big_tensor = -1
    tensor_no = 0
    copier = _ReadAheadPinningCopier()
    owners = []
    records, items = [], {} # with ranks: a _PinnedBuffer per buffer, and the item of each pinned name (its tied names follow it)
    def report_pinned(blocks):
        nonlocal total, registered
        for block_no in blocks:
            if owners[block_no] is not None:
                if not copier.lock(big_tensors[block_no], owners[block_no]): # its tensors stay in normal RAM
                    total -= big_tensors[block_no].numel()
                    raise _PinningMemoryError(f"cudaHostRegister failed: {copier.lock_error}")
                registered += big_tensors[block_no].numel()
            if loading_callback is not None: loading_callback.report("Pinning", block_no + 1, pinning_total, model_id)
    try:
        for n, (p, is_buffer) in params_dict.items():
            if p is None: continue
            q_name = tied_weights.get(n,None)
            if q_name != None:
                q , _ = params_dict[q_name]
                sub_tensors = _get_quantized_subtensors(q)
                if sub_tensors:
                    sub_map = {name: tensor for name, tensor in sub_tensors}
                    _set_quantized_subtensors(p, sub_map)
                    del sub_map, sub_tensors
                else:
                    p.data = q.data
                if q_name in items:
                    items[q_name][4].append(p)
                q = None
            else:

                big_tensor_no, offset, length = tensor_map_indexes[tensor_no]
                if last_allocated_big_tensor <  big_tensor_no:
                    if loading_callback is not None: loading_callback.check_abort()
                    last_allocated_big_tensor += 1
                    size = big_tensors_sizes[last_allocated_big_tensor]
                    try:
                        if max_reservable_memory > 0 and ( (pinned.total_bytes + total - registered + size) >= max_reservable_memory):
                            dummy_pinned_tensor = None
                            failed_planned_allocation = True
                            pinned.limit_bytes = pinned.total_bytes + total - registered
                            break

                        report_pinned(copier.wait(keep=1)) # the copies of the block two allocations back are done: it can be locked
                        try:
                            current_big_tensor, owner = pinned.block(size)
                        except (MemoryError, RuntimeError) as error:
                            raise _PinningMemoryError(f"allocation of {size/ONE_MB:.2f} MB failed: {error}") from error
                        big_tensors.append(current_big_tensor)
                        owners.append(owner)
                        if ranks is not None and owner is not None:
                            records.append(_PinnedBuffer(big_tensors_keys[last_allocated_big_tensor], current_big_tensor, owner))
                        copier.start_block(last_allocated_big_tensor)
                        if len(big_tensors) == 1 and loading_callback is not None and pinning_total: # the phase starts with its first block: a model none of whose blocks can be allocated reports nothing
                            loading_callback.report("Pinning", 0, pinning_total, model_id)
                    except _PinningMemoryError as error:
                        print(f"Unable to pin more tensors for this model ({total/ONE_MB:.2f} MB): {error}")
                        dummy_pinned_tensor = None
                        failed_planned_allocation = True
                        pinned.limit_bytes = pinned.total_bytes + total - registered
                        flush_torch_caches()
                        break

                    total += size

                current_big_tensor = big_tensors[big_tensor_no]

                if is_buffer :
                    _force_load_buffer(p) # otherwise potential memory leak
                record = records[big_tensor_no] if records else None
                sub_tensors = _get_quantized_subtensors(p)
                if sub_tensors:
                    sub_offset = offset
                    new_subs, offsets = {}, []
                    sources = [tensor for _, tensor in sub_tensors] if record is not None and record.mapped else None # views of the file, kept
                    for name, tensor in sub_tensors:
                        length = torch.numel(tensor) * tensor.element_size()
                        new_subs[name] = _move_to_pinned_tensor(tensor, current_big_tensor, sub_offset, length, copier)
                        offsets.append(sub_offset)
                        sub_offset += _block_aligned(length)
                        tensor = None
                    _set_quantized_subtensors(p, new_subs)
                    if record is not None:
                        record.items.append(items.setdefault(n, [p, list(new_subs), offsets, sources or [], []]))
                    del new_subs, sub_tensors, sources
                else:
                    length = torch.numel(p.data) * p.data.element_size()
                    source = p.data if record is not None and record.mapped else None # a view of the file, kept; a private source is freed
                    p.data = _move_to_pinned_tensor(p.data, current_big_tensor, offset, length, copier)
                    if record is not None:
                        record.items.append(items.setdefault(n, [p, None, [offset], [] if source is None else [source], []]))
                    source = None

                tensor_no += 1
            del p
        try:
            report_pinned(copier.wait())
        except _PinningMemoryError as error:
            print(f"Unable to lock more RAM for this model ({total/ONE_MB:.2f} MB): {error}")
            failed_planned_allocation = True
            pinned.limit_bytes = pinned.total_bytes + total - registered
    finally:
        copier.close() # also on cancellation: no copy may still write into blocks being freed
    del dummy_pinned_tensor,tied_weights, ref_cache
    for record in records:
        registration = getattr(record.owner, "registration", None)
        record.state = "locked" if registration is not None and registration.registered else "unlocked"
    if records:
        model._mm_pinned_buffers = getattr(model, "_mm_pinned_buffers", []) + records
    del records, items
    model._pinned_bytes = registered if pinned.LOCKS_RAM else total
    model._pinned_bytes_by_owner = pinned.LOCKS_RAM
    if not pinned.LOCKS_RAM: # legacy driver-allocated blocks retain their model-level accounting
        pinned.account(total, "model")
    del params_dict
    gc.collect()

    if verboseLevel >=1:
        if partialPinning or failed_planned_allocation:
            print(f"The model was partially pinned to reserved RAM: {last_allocated_big_tensor + 1} large blocks spread across {total/ONE_MB:.2f} MB")
        else:
            print(f"The whole model was pinned to reserved RAM: {last_allocated_big_tensor + 1} large blocks spread across {total/ONE_MB:.2f} MB")

    model._already_pinned = True


    return 
def change_dtype(model, new_dtype, exclude_buffers = False):
    for submodule_name, submodule in model.named_modules():  
        if hasattr(submodule, "_lock_dtype"):
            continue
        for n, p in submodule.named_parameters(recurse = False):
            if isinstance(p, QTensor) or _is_mapped(p):
                continue
            if p.data.dtype != new_dtype:
                p.data = p.data.to(new_dtype)

        if not exclude_buffers:
            for p in submodule.buffers(recurse=False):
                if isinstance(p, QTensor):
                    continue
                if p.data.dtype != new_dtype:
                    p.data = p.data.to(new_dtype)

    return model
            
def _extract_num_from_str(num_in_str):
    size = len(num_in_str)
    for i in range(size):
        if not num_in_str[-i-1:].isnumeric():
            if i == 0:
                return num_in_str, -1
            else:             
                return num_in_str[: -i],  int(num_in_str[-i:])                    
    return  "", -1 if size == 0 else int(num_in_str)

def  _quantize_dirty_hack(model):
    # dirty hack: add a hook on state_dict() to return a fake non quantized state_dict if called by Lora Diffusers initialization functions
    setattr( model, "_real_state_dict", model.state_dict)
    from collections import OrderedDict
    import traceback

    def state_dict_for_lora(self):
        real_sd = self._real_state_dict()
        fakeit = False
        stack = traceback.extract_stack(f=None, limit=5)
        for frame in stack:
            if "_lora_" in frame.name:
                fakeit = True
                break

        if not fakeit:
            return real_sd
        sd = OrderedDict()
        for k in real_sd:
            v = real_sd[k]
            if k.endswith("._data"):
                k = k[:len(k)-6]
            sd[k] = v
        return sd

    setattr(model, "state_dict", functools.update_wrapper(functools.partial(state_dict_for_lora, model), model.state_dict) )

def _quantization_map(model):
    from optimum.quanto import quantization_map
    return quantization_map(model)

def _set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        parent_module_name = name[: name.rindex(".")]
        parent_module = parent_module.get_submodule(parent_module_name)
        setattr(parent_module, module_names[-1], child_module)

def _quantize_submodule(
    model: torch.nn.Module,
    name: str,
    module: torch.nn.Module,
    weights = None,
    activations = None,
    optimizer = None,
):
    
    qmodule = quantize_module(module, weights=weights, activations=activations, optimizer=optimizer)
    if qmodule is not None:
        _set_module_by_name(model, name, qmodule)
        qmodule.name = name
        for name, param in module.named_parameters():
            # Save device memory by clearing parameters
            setattr(module, name, None)
            del param

def _requantize(model: torch.nn.Module, state_dict: dict, quantization_map: dict, default_dtype=None):
    quantized_names = set(quantization_map.keys())

    def _is_quantized_param(param_name):
        if param_name in quantized_names:
            return True
        if "." in param_name:
            return param_name.rsplit(".", 1)[0] in quantized_names
        return False

    # change dtype of current meta model parameters because 'requantize' won't update the dtype on non quantized parameters
    for k, p in model.named_parameters():
        if _is_quantized_param(k) or k not in state_dict:
            continue
        p_in_file = state_dict[k]
        if not (p_in_file.data.dtype.is_floating_point or p_in_file.data.dtype.is_complex):
            continue
        if p.data.dtype != p_in_file.data.dtype:
            p.data = p.data.to(p_in_file.data.dtype)

    # rebuild quanto objects
    for name, m in model.named_modules():
        qconfig = quantization_map.get(name, None)
        if qconfig is not None:
            weights = qconfig["weights"]
            if weights == "none":
                weights = None
            activations = qconfig["activations"]
            if activations == "none":
                activations = None
            _quantize_submodule(model, name, m, weights=weights, activations=activations)
            if default_dtype is not None:
                new_module = model.get_submodule(name)
                setter = getattr(new_module, "set_default_dtype", None)
                if callable(setter):
                    setter(default_dtype)

    model._quanto_map = quantization_map

    _quantize_dirty_hack(model)



def _quantize_exclude_modules(model_to_quantize, quantize_exclude, verboseLevel=1):
    if quantize_exclude is None:
        return []
    if isinstance(quantize_exclude, str):
        quantize_exclude = [quantize_exclude]

    matched = []
    pattern_matches = {pattern: False for pattern in quantize_exclude}
    for submodule_name, submodule in model_to_quantize.named_modules():
        if submodule_name == "":
            continue
        candidates = [submodule_name]
        candidates += [f"{submodule_name}.{name}" for name, _ in submodule.named_parameters(recurse=False)]
        candidates += [f"{submodule_name}.{name}" for name, _ in submodule.named_buffers(recurse=False)]
        module_matches = False
        for pattern in quantize_exclude:
            if any(fnmatch.fnmatchcase(candidate, pattern) for candidate in candidates):
                pattern_matches[pattern] = True
                module_matches = True
        if module_matches:
            matched.append(submodule_name)

    if verboseLevel >= 1 and len(matched) > 0:
        print(f"User excluded {len(matched)} modules from quantization")
    if verboseLevel >= 2:
        missing = [pattern for pattern, did_match in pattern_matches.items() if not did_match]
        if len(missing) > 0:
            print(f"User quantization exclude patterns not matched: {missing}")
    return matched


def _unique_list(values):
    unique = []
    seen = set()
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _quantize(model_to_quantize, weights=qint8, verboseLevel = 1, threshold = 2**31, model_id = 'Unknown', quantize_exclude = None):
    
    total_size =0
    total_excluded = 0
    exclude_list = []
    submodule_size = 0
    submodule_names = []
    cur_blocks_prefix = None
    prev_blocks_prefix = None

    if hasattr(model_to_quantize, "_quanto_map"):
        for k, entry in model_to_quantize._quanto_map.items():
            weights  =  entry["weights"]
            print(f"Model '{model_id}' is already quantized to format '{weights}'")
            return False
        print(f"Model '{model_id}' is already quantized")
        return False

    print(f"Quantization of model '{model_id}' started to format '{weights}'")

    tower_names ,_  = _detect_main_towers(model_to_quantize)
    tower_names = [ n[:-1] for n in tower_names]


    cache_ref = {}
    tied_weights= {}
    reversed_tied_weights= {}

    for submodule_name, submodule in model_to_quantize.named_modules():  
        if isinstance(submodule, QModuleMixin):
            if verboseLevel>=1:
                print("No quantization to do as model is already quantized")
            return False

        size = 0
        for n, p in submodule.named_parameters(recurse = False):
            ref = _get_tensor_ref(p)
            match = cache_ref.get(ref, None)
            if match != None:
                tied_weights[submodule_name]=  (n, ) + match
                entries = reversed_tied_weights.get( match, [])
                reversed_tied_weights[match] = entries + [ (p, submodule_name,n)]
            else:
                cache_ref[ref] = (submodule_name, n)
                size  += torch.numel(p.data) * sizeofhalffloat

        for p in submodule.buffers(recurse=False):
            size  += torch.numel(p.data) * sizeofhalffloat

        already_added = False
        if hasattr(submodule, "_lock_dtype"):
            submodule_size += size
            submodule_names.append(submodule_name)
            already_added = True

        if not any(submodule_name.startswith(pre) for pre in tower_names):
            flush = False
            if cur_blocks_prefix == None or not submodule_name.startswith(cur_blocks_prefix):
                cur_blocks_prefix = submodule_name + "."
                flush = True                    

            if flush :
                if submodule_size <= threshold :
                    exclude_list += submodule_names
                    if verboseLevel >=2 and submodule_size >0:
                        print(f"Excluded size {submodule_size/ONE_MB:.1f} MB: {prev_blocks_prefix} : {submodule_names}")
                    total_excluded += submodule_size

                submodule_size = 0
                submodule_names = []
            prev_blocks_prefix = cur_blocks_prefix
            if not already_added:
                submodule_size += size
                submodule_names.append(submodule_name)
        total_size += size

    if submodule_size >0  : 
        exclude_list += submodule_names
        if verboseLevel >=2:
            print(f"Excluded size {submodule_size/ONE_MB:.1f} MB: {prev_blocks_prefix} : {submodule_names}")
        total_excluded += submodule_size


    perc_excluded =total_excluded/ total_size if total_size >0 else 1
    if verboseLevel >=2:
        if total_excluded == 0:
            print(f"Can't find any module to exclude from quantization, full model ({total_size/ONE_MB:.1f} MB) will be quantized")
        else:
            print(f"Total Excluded {total_excluded/ONE_MB:.1f} MB of {total_size/ONE_MB:.1f} that is {perc_excluded*100:.2f}%")
    if perc_excluded >= 0.20:
        if verboseLevel >=2:
            print(f"Too many modules are excluded, there is something wrong with the selection, switch back to full quantization.")
        exclude_list = None


    user_exclude_list = _quantize_exclude_modules(model_to_quantize, quantize_exclude, verboseLevel=verboseLevel)
    # the modules read in place from mapped RAM keep their tensors
    user_exclude_list += [name for name, module in model_to_quantize.named_modules() if builtins.any(_is_mapped(p) for p in module.parameters(recurse=False))]
    if exclude_list is None:
        exclude_list = list(tied_weights) + user_exclude_list
    else:
        exclude_list += list(tied_weights) + user_exclude_list
    exclude_list = _unique_list(exclude_list)
    quantize(model_to_quantize, weights= weights, exclude= exclude_list)


    # quantize(model_to_quantize,weights, include= [ "*1.block.attn.to_out*"]) #" 

    # for name, m in model_to_quantize.named_modules():
    #     if exclude_list is None or not any( name == module_name for module_name in exclude_list):
    #         _quantize_submodule(model_to_quantize, name, m, weights=weights, activations=None, optimizer=None)


    # force to read non quantized parameters so that their lazy tensors and corresponding mmap are released
    # otherwise we may end up keeping in memory both the quantized and the non quantize model
    named_modules = {n:m for n,m in model_to_quantize.named_modules()}

    for module_name, module in named_modules.items():
        # do not read quantized weights (detected them directly or behind an adapter)
        if isinstance(module, QModuleMixin) or hasattr(module, "base_layer") and  isinstance(module.base_layer, QModuleMixin): 
            if hasattr(module, "bias") and module.bias is not None:
                _force_load_parameter(module.bias)
        else:
            tied_w = tied_weights.get(module_name, None)
            for n, p in module.named_parameters(recurse = False):
                if _is_mapped(p): # already in its own RAM (a copy would end in VRAM)
                    continue
                if tied_w != None and n == tied_w[0]:
                    if isinstance( named_modules[tied_w[1]], QModuleMixin) :
                        setattr(module, n, None) # release refs of tied weights if source is going to be quantized
                    # otherwise don't force load as it will be loaded in the source anyway
                else:
                    _force_load_parameter(p)
                    entries =  reversed_tied_weights.get( (module_name, n), [])
                    for tied_weight, tied_module_name, tied_weight_name in entries:
                        if n == tied_weight_name:
                             tied_weight.data = p.data

                del p #  del p if not it will still contain a ref to a tensor when leaving the loop
        for b in module.buffers(recurse = False):
            _force_load_buffer(b) 
            del b


    freeze(model_to_quantize)
    torch.cuda.empty_cache()
    gc.collect()       

    for tied_module, (tied_weight, src_module, src_weight) in tied_weights.items():  
        p = getattr(named_modules[src_module], src_weight)
        if isinstance(p, QTensor):
            setattr(named_modules[tied_module], tied_weight, p ) # copy refs to quantized sources

    del named_modules

    quantization_map = _quantization_map(model_to_quantize)

    model_to_quantize._quanto_map = quantization_map

    if hasattr(model_to_quantize, "_already_pinned"):
        delattr(model_to_quantize, "_already_pinned")

    _quantize_dirty_hack(model_to_quantize)

    print(f"Quantization of model '{model_id}' done")

    return True
def load_loras_into_model(model, lora_path, lora_multi = None, activate_all_loras = True, check_only = False, ignore_model_variations = False, pinnedLora = False, maxReservedLoras = -1, split_linear_modules_map = None, preprocess_sd = None, verboseLevel = -1,):
    verboseLevel = _compute_verbose_level(verboseLevel)

    loras_model_data = getattr(model, "_loras_model_data", None)
    if loras_model_data == None:
        merged_loras_model_data = {}
        merged_loras_shortcuts = {}
        sub_loras = {}
        for submodule_name, submodule in model.named_modules():
            if submodule is model:
                continue
            sub_model_data = getattr(submodule, "_loras_model_data", None)
            if sub_model_data:
                submodule._lora_owner = model
                sub_loras[submodule_name] = submodule
                for k, v in sub_model_data.items():
                    if k not in merged_loras_model_data:
                        merged_loras_model_data[k] = v
            sub_shortcuts = getattr(submodule, "_loras_model_shortcuts", None)
            if sub_shortcuts:
                prefix = f"{submodule_name}." if submodule_name else ""
                for k, v in sub_shortcuts.items():
                    merged_key = k
                    if prefix:
                        if k:
                            merged_key = f"{prefix}{k}"
                        else:
                            merged_key = submodule_name
                    if merged_key not in merged_loras_shortcuts:
                        merged_loras_shortcuts[merged_key] = v
        if merged_loras_model_data:
            model._loras_model_data = merged_loras_model_data
            if merged_loras_shortcuts:
                model._loras_model_shortcuts = merged_loras_shortcuts
            model._subloras = sub_loras
            loras_model_data = merged_loras_model_data
        else:
            raise Exception(f"No Loras has been declared for this model while creating the corresponding offload object")
    
    if not check_only:
        unload_loras_from_model(model)

    modules_dict = {k: v for k,v in model.named_modules()}
    module_names = {v: k for k, v in modules_dict.items()}

    CrLf = '\r\n'
    error_msg = ""
    def append(source, text ):
        if len(source) == 0:
            return text
        else:
            return source + CrLf + text
    
    def trunc(text, sz):
        text = str(text)
        if len(text) < sz:
            return text
        else:
            return text[0:sz] + '...'
    
    def _state_dict_size_mb(state_dict):
        total_bytes = 0
        for v in state_dict.values():
            if torch.is_tensor(v):
                total_bytes += v.numel() * v.element_size()
        return total_bytes / (1024 * 1024)

    def _split_lokr_output_sizes(split_sizes, w1, w2):
        if w1 is None or w2 is None:
            return None
        rows2 = int(w2.shape[0])
        if rows2 <= 0:
            return None
        out_rows = int(w1.shape[0]) * rows2
        if sum(split_sizes) != out_rows:
            return None
        reduced = []
        for sz in split_sizes:
            if sz % rows2 != 0:
                return None
            reduced.append(sz // rows2)
        if sum(reduced) != int(w1.shape[0]):
            return None
        return reduced

    def _split_lokr_chunked_if_needed(parent_prefix, mapped_modules, split_sizes, w1, w2, new_state_dict, lokr_split_chunks):
        if w1 is None or w2 is None:
            return False
        out_rows = int(w1.shape[0]) * int(w2.shape[0])
        if sum(split_sizes) != out_rows:
            return False
        start = 0
        for sub_name, sub_size in zip(mapped_modules, split_sizes):
            target_base = parent_prefix + sub_name
            if target_base + ".lokr_w1" not in new_state_dict:
                new_state_dict[target_base + ".lokr_w1"] = w1
            if target_base + ".lokr_w2" not in new_state_dict:
                new_state_dict[target_base + ".lokr_w2"] = w2
            lokr_split_chunks[target_base] = (start, start + sub_size)
            start += sub_size
        return True

    def _split_lokr_key(module_name, module_data, state_dict, new_state_dict, lokr_split_chunks):
        name_parts = module_name.split(".")
        if len(name_parts) < 2:
            return module_name
        split_map = split_linear_modules_map.get(name_parts[-2], None)
        if split_map is None:
            return module_name
        mapped_modules = split_map.get("mapped_modules", None)
        split_sizes = split_map.get("split_sizes", None)
        if not mapped_modules or not split_sizes:
            return module_name

        parent_module_name = ".".join(name_parts[:-1])
        is_w1 = module_name.endswith(".lokr_w1")
        sibling_suffix = ".lokr_w2" if is_w1 else ".lokr_w1"
        sibling_tensor = state_dict.get(parent_module_name + sibling_suffix, None)
        w1 = module_data if is_w1 else sibling_tensor
        w2 = sibling_tensor if is_w1 else module_data
        reduced_split = _split_lokr_output_sizes(split_sizes, w1, w2)
        parent_prefix = parent_module_name[:-len(name_parts[-2])]
        if reduced_split is None:
            if _split_lokr_chunked_if_needed(parent_prefix, mapped_modules, split_sizes, w1, w2, new_state_dict, lokr_split_chunks):
                return None
            return module_name

        if is_w1:
            chunks = torch.split(module_data, reduced_split, dim=0)
            suffix = ".lokr_w1"
        else:
            chunks = [module_data] * len(mapped_modules)
            suffix = ".lokr_w2"
        for sub_name, sub_data in zip(mapped_modules, chunks):
            new_state_dict[parent_prefix + sub_name + suffix] = sub_data
        return None

    def _build_lokr_chunk_specs(chunk, m2):
        start, end = int(chunk[0]), int(chunk[1])
        if end <= start or m2 <= 0:
            return []
        s_row, s_col = divmod(start, m2)
        e_row, e_col = divmod(end, m2)
        if s_row == e_row:
            return [(s_row, s_row + 1, s_col, e_col)]

        specs = []
        mid_start_row = s_row
        if s_col > 0:
            specs.append((s_row, s_row + 1, s_col, m2))
            mid_start_row += 1
        if mid_start_row <= e_row - 1:
            specs.append((mid_start_row, e_row, 0, m2))
        if e_col > 0:
            specs.append((e_row, e_row + 1, 0, e_col))
        return specs

    def _ensure_adapter_meta(loras_module_data, adapter_name):
        data = loras_module_data.get(adapter_name, None)
        if data is None:
            data = [None, None, None, None, 1., {}]
            loras_module_data[adapter_name] = data
        return data, data[5] 

    def _to_dtype_cached(tensor, dtype, cast_cache):
        if tensor is None or tensor.dtype == dtype:
            return tensor
        key = (_get_tensor_ref(tensor), dtype)
        cached = cast_cache.get(key, None)
        if cached is None:
            cached = tensor.to(dtype)
            cast_cache[key] = cached
        return cached

    if not isinstance(lora_path, list):
        lora_path = [lora_path]
    
    if lora_multi is None:
        lora_multi = [1. for _ in lora_path]
    try:
        max_reserved_loras_mb = float(maxReservedLoras)
    except Exception:
        max_reserved_loras_mb = -1
    if max_reserved_loras_mb is None:
        max_reserved_loras_mb = -1
    pinned_total_mb = 0.0
    loras_nos = []
    loras_multi = []
    new_lora_path = []
    errors  = []
    adapters = {}
    adapter_no = 0
    pinned_sd_list = []
    pinned_names_list = []
    pinned_tied_weights_list = []
    for i, path in enumerate(lora_path):
        adapter_name = str(adapter_no)
        error_msg = ""
        if not os.path.isfile(path):
            error_msg = f"Lora '{path}' was not found"
            errors.append((path, error_msg))
            print(error_msg)
            continue
        fail = False
        skip = False
        tied_weights = None
        state_dict, metadata = _safetensors_load_file(path, writable_tensors=False)
        if metadata is not None and "alpha" in metadata:
            try:
                header_alpha = float(metadata["alpha"])
                if not math.isfinite(header_alpha):
                    raise ValueError("Non-finite alpha")
            except (TypeError, ValueError, OverflowError):
                tqdm.write(f"Ignoring invalid LoRA header alpha in '{path}'.")
            else:
                # Preprocessors may fuse projections and bake alpha into their factors.
                # Supply missing scalars first, preserving explicit per-module values.
                for key in list(state_dict):
                    match = re.fullmatch(r"(.*)\.lora[._](?:[AB]|down|up)(?:(?:\.[^.]+)?\.weight)?", key)
                    if match is not None:
                        alpha_key = match[1] + ".alpha"
                        if alpha_key not in state_dict:
                            state_dict[alpha_key] = torch.tensor(header_alpha, dtype=torch.float64, device="cpu")
        if preprocess_sd != None:
            state_dict = preprocess_sd(state_dict)

        clean_up = False
        first_key = next(iter(state_dict), None)
        if first_key is  None:
            msg = f"Empty Lora '{path}'"
            error_msg = append(error_msg, msg) 
            fail = True
        else:
            prefixes = ("diffusion_model.", "transformer.")
            pos = first_key.find(".")
            prefix = first_key[0:pos+1]
            if prefix in prefixes:
                new_state_dict = {}
                for k, v in state_dict.items():
                    for candidate in prefixes:
                        if k.startswith(candidate):
                            k = k[len(candidate) :]
                            break
                    new_state_dict[k] = v
                state_dict = new_state_dict
                del new_state_dict

            src_suffixes = (".default.weight", ".lora.A.weight", ".lora.B.weight", ".lora.down.weight", ".lora.up.weight")
            tgt_suffixes = (".weight", ".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight")
            new_sd = {}
            for k,v in state_dict.items():
                for src, tgt in zip(src_suffixes, tgt_suffixes):
                    if k.endswith(src):
                        k = k[:-len(src)] + tgt
                        break
                new_sd[k] = v
            state_dict = new_sd
            del new_sd,k,v

        if not fail:
            lokr_split_chunks = {}
            if split_linear_modules_map != None:
                new_state_dict = {}
                suffixes = [(".alpha", -2, False), (".lora_B.weight", -3, True), (".lora_A.weight", -3, False), (".lora_up.weight", -3, True), (".lora_down.weight", -3, False),(".dora_scale", -2, False),]
                for module_name, module_data in state_dict.items():
                    if module_name.endswith(".lokr_w1") or module_name.endswith(".lokr_w2"):
                        module_name = _split_lokr_key(module_name, module_data, state_dict, new_state_dict, lokr_split_chunks)
                        if module_name is not None: new_state_dict[module_name] = module_data
                    else:
                        name_parts = module_name.split(".")
                        for suffix, pos, any_split in suffixes:
                            if not module_name.endswith(suffix) or (map := split_linear_modules_map.get(name_parts[pos], None)) is None:
                                continue
                            parent_module_name = ".".join(name_parts[:pos])
                            sub_data = torch.split(module_data, map["split_sizes"], dim=0) if any_split else (module_data,) * len(map["mapped_modules"])
                            for sub_name, subdata in zip(map["mapped_modules"], sub_data):
                                new_state_dict[parent_module_name + "." + sub_name + suffix] = subdata
                            break
                        else:
                            new_state_dict[module_name] = module_data
                state_dict = new_state_dict
                del new_state_dict
            clean_up = True

            keys = list(state_dict.keys())
            dtype_cast_cache = {}

            lora_alphas = {}
            for k in keys:
                if k.endswith(".alpha"):
                    alpha_value = state_dict.pop(k)
                    if torch.is_tensor(alpha_value):
                        alpha_value = float(alpha_value.item())
                    lora_alphas[k] = alpha_value

            invalid_keys = []
            unexpected_keys = []
            new_state_dict = {}
            dora_found = False
            lokr_found = False
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                lora_A = lora_B = diff_b = diff = lora_key = dora_scale = lokr_w1 = lokr_w2 = None
                adapter_type = "lora"
                if k.endswith(".diff"):
                    diff = v
                    module_name = k[ : -5]
                    adapter_type = "diff"
                elif k.endswith(".diff_b"):
                    diff_b = v
                    module_name = k[ : -7]
                    adapter_type = "lora"
                elif k.endswith(".dora_scale"):
                    dora_scale = v
                    module_name = k[ : -11]
                    adapter_type = "dora"
                elif k.endswith(".lokr_w1"):
                    lokr_w1 = v
                    module_name = k[ : -8]
                    adapter_type = "lokr"
                elif k.endswith(".lokr_w2"):
                    lokr_w2 = v
                    module_name = k[ : -8]
                    adapter_type = "lokr"
                else:
                    pos = k.rfind(".lora_")
                    if pos <=0:
                        invalid_keys.append(k)
                        continue
                    module_name = k[ : pos]
                    lora_key = k[ pos+1:]
                    if lora_key in ("lora_A.weight", "lora_down.weight"):
                        lora_A = v
                    elif lora_key in ("lora_B.weight", "lora_up.weight"):
                        lora_B = v
                    else:
                        invalid_keys.append(k)
                        continue

                module =  modules_dict.get(module_name, None)
                if module == None:
                    unexpected_keys.append(k)
                    continue
                module_shape = module.weight.shape
                rank = None
                if lora_A != None:
                    rank = lora_A.shape[0] 
                    if module_shape[1] != v.shape[1]:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': Lora A dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape[1]}, lora A = {v.shape[1]}). It is likely this Lora has been made for another version of this model."
                            error_msg = append(error_msg, msg) 
                            fail = True
                        break
                    v = lora_A = _to_dtype_cached(lora_A, module.weight.dtype, dtype_cast_cache)
                elif lora_B != None:
                    rank = lora_B.shape[1] 
                    if module_shape[0] != v.shape[0]:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': Lora B dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape[0]}, lora B = {v.shape[0]}). It is likely this Lora has been made for another version of this model."
                            error_msg = append(error_msg, msg) 
                            fail = True
                        break
                    v = lora_B = _to_dtype_cached(lora_B, module.weight.dtype, dtype_cast_cache)
                elif diff != None:
                    lora_B = diff
                    if module_shape != v.shape:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': Lora shape is not compatible with model '{_get_module_name(model)}' (model = {module_shape}, lora = {v.shape}). It is likely this Lora has been made for another version of this model."
                            error_msg = append(error_msg, msg) 
                            fail = True
                        break
                    v = lora_B = _to_dtype_cached(lora_B, module.weight.dtype, dtype_cast_cache)
                elif diff_b != None:
                    rank = diff_b.shape[0] 
                    if not hasattr(module, "bias"):
                        pass
                    if module.bias == None:
                        msg = f"Lora '{path}': Lora Basis is defined while it doesnt exist in model '{_get_module_name(model)}'. It is likely this Lora has been made for another version of this model."
                        fail = True
                        break
                    else:
                        module_shape = module.bias.shape
                        if module_shape != v.shape:
                            if ignore_model_variations:
                                skip = True
                            else:
                                msg = f"Lora '{path}': Lora Basis dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape[0]}, lora Basis = {v.shape[0]}). It is likely this Lora has been made for another version of this model."
                                error_msg = append(error_msg, msg) 
                                fail = True
                            break
                    v = diff_b = _to_dtype_cached(diff_b, module.weight.dtype, dtype_cast_cache)
                elif dora_scale != None:
                    rank = dora_scale.shape[1] 
                    if module_shape[0] != v.shape[0]:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}': Dora Scale dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape[0]}, dora scale = {v.shape[0]}). It is likely this Dora has been made for another version of this model."
                            error_msg = append(error_msg, msg) 
                            fail = True
                        break
                    v = dora_scale = _to_dtype_cached(dora_scale, module.weight.dtype, dtype_cast_cache)
                elif lokr_w1 is not None:
                    out_dim, in_dim = int(module_shape[0]), int(module_shape[1])
                    m1, n1 = int(lokr_w1.shape[0]), int(lokr_w1.shape[1])
                    lokr_chunk = lokr_split_chunks.get(module_name, None) 
                    if lokr_chunk is None:
                        invalid = m1 <= 0 or n1 <= 0 or out_dim % m1 != 0 or in_dim % n1 != 0
                    else:
                        invalid = m1 <= 0 or n1 <= 0 or (lokr_chunk[1] - lokr_chunk[0]) != out_dim or in_dim % n1 != 0
                    if invalid:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': LoKr W1 dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape}, lokr_w1 = {v.shape})."
                            error_msg = append(error_msg, msg)
                            fail = True
                        break
                    v = lokr_w1 = _to_dtype_cached(lokr_w1, module.weight.dtype, dtype_cast_cache)
                elif lokr_w2 is not None:
                    out_dim, in_dim = int(module_shape[0]), int(module_shape[1])
                    m2, n2 = int(lokr_w2.shape[0]), int(lokr_w2.shape[1])
                    lokr_chunk = lokr_split_chunks.get(module_name, None) 
                    if lokr_chunk is None:
                        invalid = m2 <= 0 or n2 <= 0 or out_dim % m2 != 0 or in_dim % n2 != 0
                    else:
                        invalid = m2 <= 0 or n2 <= 0 or (lokr_chunk[1] - lokr_chunk[0]) != out_dim or in_dim % n2 != 0
                    if invalid:
                        if ignore_model_variations:
                            skip = True
                        else:
                            msg = f"Lora '{path}/{module_name}': LoKr W2 dimension is not compatible with model '{_get_module_name(model)}' (model = {module_shape}, lokr_w2 = {v.shape})."
                            error_msg = append(error_msg, msg)
                            fail = True
                        break
                    v = lokr_w2 = _to_dtype_cached(lokr_w2, module.weight.dtype, dtype_cast_cache)
                if not check_only:
                    new_state_dict[k] = v
                    v = None
                    loras_module_data = loras_model_data.get(module, None)
                    assert loras_module_data is not None
                    loras_adapter_data, meta = _ensure_adapter_meta(loras_module_data, adapter_name)
                    if adapter_type in ("dora", "lokr", "diff"):
                        meta["type"] = adapter_type
                    elif "type" not in meta:
                        meta["type"] = "lora"
                    if adapter_type == "lokr":
                        lokr_found = True
                        if lokr_chunk is not None: meta["lokr_chunk"] = lokr_chunk
                    elif adapter_type == "dora":
                        dora_found = True
                    if lora_A != None:
                        loras_adapter_data[0] = lora_A
                    elif lora_B != None:
                        loras_adapter_data[1] = lora_B 
                    elif dora_scale != None:
                        loras_adapter_data[3] = dora_scale 
                    elif lokr_w1 is not None:
                        loras_adapter_data[0] = lokr_w1
                    elif lokr_w2 is not None:
                        loras_adapter_data[1] = lokr_w2
                    else:
                        loras_adapter_data[2] = diff_b 
                    if rank != None and lora_key is not None and "lora" in lora_key:
                        alpha_key = k[:-len(lora_key)] + "alpha"
                        alpha = lora_alphas.get(alpha_key, None)
                        if alpha is not None: loras_adapter_data[4] = alpha / rank 
            lora_A = lora_B = diff = diff_b = v = loras_module_data = loras_adapter_data = lora_alphas = dora_scale = lokr_w1 = lokr_w2 = dtype_cast_cache = None

            if len(invalid_keys)  > 0:
                msg = f"Lora '{path}' contains non Lora keys '{trunc(invalid_keys,200)}'"
                error_msg = append(error_msg, msg) 
                fail = True
            if len(unexpected_keys)  > 0:
                msg = f"Lora '{path}' contains unexpected module keys, it is likely that this Lora is for a different model : '{trunc(unexpected_keys,200)}'"
                error_msg = append(error_msg, msg) 
                fail = True
            if (not fail) and (not check_only) and dora_found and lokr_found:
                msg = f"Lora '{path}': LoKr cannot be mixed with DoRA on the same module."
                error_msg = append(error_msg, msg)
                fail = True
            if (not fail) and lokr_found:
                for module, loras_module_data in loras_model_data.items():
                    if adapter_name not in loras_module_data: continue
                    loras_adapter_data, meta = _ensure_adapter_meta(loras_module_data, adapter_name)
                    adapter_type = meta.get("type", "lora")
                    if adapter_type != "lokr": continue
                    w1, w2 = loras_adapter_data[0], loras_adapter_data[1]
                    if w1 is not None and w2 is not None:
                        lokr_chunk = meta.get("lokr_chunk", None)
                        effective_chunk = lokr_chunk if lokr_chunk is not None else (0, int(w1.shape[0]) * int(w2.shape[0]))
                        meta["lokr_specs"] = _build_lokr_chunk_specs(effective_chunk, int(w2.shape[0]))
                        module_shape = module.weight.shape
                        if lokr_chunk is None:
                            invalid = int(module_shape[0]) != int(w1.shape[0]) * int(w2.shape[0]) or int(module_shape[1]) != int(w1.shape[1]) * int(w2.shape[1])
                        else:
                            invalid = int(module_shape[0]) != (lokr_chunk[1] - lokr_chunk[0]) or int(module_shape[1]) != int(w1.shape[1]) * int(w2.shape[1])
                        if invalid:
                            module_name = module_names.get(module, type(module).__name__)
                            msg = f"Lora '{path}/{module_name}': LoKr factors are not compatible with model shape (model = {module_shape}, lokr_w1 = {tuple(w1.shape)}, lokr_w2 = {tuple(w2.shape)})."
                            error_msg = append(error_msg, msg)
                            fail = True
                            break
            if (not fail) and (not check_only):
                alias_owners = {}
                for module, loras_module_data in loras_model_data.items():
                    loras_adapter_data = loras_module_data.get(adapter_name, None)
                    if loras_adapter_data is None:
                        continue
                    module_name = module_names.get(module, None)
                    if module_name is None:
                        continue
                    for slot in range(min(4, len(loras_adapter_data))):
                        tensor = loras_adapter_data[slot]
                        if not torch.is_tensor(tensor):
                            continue
                        key = (slot, _get_tensor_ref(tensor))
                        owner_name = alias_owners.get(key, None)
                        if owner_name is None:
                            alias_owners[key] = module_name
                        else:
                            loras_adapter_data[slot] = owner_name + "#" + str(slot)
            if (not fail) and (not check_only) and pinnedLora:
                tied_weights = _extract_tie_weights_from_sd(new_state_dict, path, verboseLevel=max(0, verboseLevel))
        if fail or skip:
            if fail:
                errors.append((path, error_msg))
                print(error_msg)
            if clean_up and not check_only:
                for _, loras_module_data in loras_model_data.items():
                    if adapter_name in loras_module_data:
                        del loras_module_data[adapter_name]

        else:
            if not check_only:
                # model._loras_tied_weights[adapter_name] = tied_weights
                if pinnedLora:
                    if max_reserved_loras_mb < 0:
                        pinned_sd_list.append(new_state_dict)
                        pinned_names_list.append(path)
                        pinned_tied_weights_list.append(tied_weights)
                    else:
                        lora_size_mb = _state_dict_size_mb(new_state_dict)
                        if pinned_total_mb + lora_size_mb <= max_reserved_loras_mb:
                            pinned_sd_list.append(new_state_dict)
                            pinned_names_list.append(path)
                            pinned_tied_weights_list.append(tied_weights)
                            pinned_total_mb += lora_size_mb

            del state_dict 


            adapters[adapter_name] = path
            loras_nos.append(adapter_name)
            new_lora_path.append(path)        
            loras_multi.append(1.0 if i > (len(lora_multi) -1) else lora_multi[i])
            adapter_no += 1
            if verboseLevel >=1:
                if check_only:
                    print(f"Lora '{path}' was found for model '{_get_module_name(model)}'")
                else:
                    print(f"Lora '{path}' was loaded in model '{_get_module_name(model)}'")
    
    model._loras_errors = errors
    if not check_only:
        if pinnedLora and len(pinned_sd_list) > 0:
            _pin_sd_to_memory(pinned_sd_list, pinned_names_list, pinned_tied_weights_list)
        model._loras_adapters = adapters
    if activate_all_loras:
        activate_loras(model, loras_nos, loras_multi)
    return new_lora_path


def merge_dicts(A, B):
    for key, value in A.items():
        if isinstance(value, dict):
            if key not in B or not isinstance(B[key], dict):
                B[key] = value  # Copy entire dict reference from A
            else:
                merge_dicts(value, B[key])  # Recurse into both dicts
        else:
            B[key] = value  # Copy non-dict value from A to B


def sync_models_loras(model, model2):
    merge_dicts(model._loras_model_shortcuts , model2._loras_model_shortcuts)
    model2._loras_active_adapters = model._loras_active_adapters 
    model2._loras_adapters = model._loras_adapters
    model2._loras_scaling = model._loras_scaling 

def unload_loras_from_model(model):
    if model is None: return
    if not hasattr(model, "_loras_model_data"): return
    for _, v in model._loras_model_data.items():
        v.clear()
    for _, v in model._loras_model_shortcuts.items():
        v.clear()

    model._loras_active_adapters = []
    model._loras_scaling = dict()
    model._loras_tied_weights = dict()
    model._loras_errors = None
    model._loras_adapters = None
    model._loras_scaling = None


def set_step_no_for_lora(model, step_no):
    target = getattr(model, "_lora_owner", None)
    while target is not None and target is not model:
        model = target
        target = getattr(model, "_lora_owner", None)
    model._lora_step_no = step_no
    sub_loras = getattr(model, "_subloras", None)
    if sub_loras:
        submodules = sub_loras.values() if isinstance(sub_loras, dict) else sub_loras
        for submodule in submodules:
            if submodule is model:
                continue
            submodule._lora_step_no = step_no

def activate_loras(model, lora_nos, lora_multi = None):
    target = getattr(model, "_lora_owner", None)
    while target is not None and target is not model:
        model = target
        target = getattr(model, "_lora_owner", None)

    if not isinstance(lora_nos, list):
        lora_nos = [lora_nos]
    lora_nos = [str(l) for l in lora_nos]

    if lora_multi is None:
        lora_multi = [1. for _ in lora_nos]

    lora_scaling_dict = {}
    for no, multi in zip(lora_nos, lora_multi):
        lora_scaling_dict[no] = multi

    model._lora_step_no = 0    
    model._loras_active_adapters = lora_nos
    model._loras_scaling = lora_scaling_dict 
    sub_loras = getattr(model, "_subloras", None)
    if sub_loras:
        submodules = sub_loras.values() if isinstance(sub_loras, dict) else sub_loras
        for submodule in submodules:
            if submodule is model:
                continue
            submodule._lora_step_no = 0
            submodule._loras_active_adapters = lora_nos
            submodule._loras_scaling = lora_scaling_dict


def move_loras_to_device(model, device="cpu" ):
    if hasattr( model, "_lora_loadable_modules"):
        for k in model._lora_loadable_modules:
            move_loras_to_device(getattr(model,k), device)
        return
    
    for k, m in model.named_modules():
        if ".lora_" in k:
            m.to(device)

def fast_load_transformers_model(model_path: str,  do_quantize = False, quantizationType =  qint8, pinToMemory = False, partialPinning = False, forcedConfigPath = None, defaultConfigPath = None, modelClass=None, modelPrefix = None, writable_tensors = True, verboseLevel = -1, preprocess_sd  = None, fused_split_map = None, modules = None,  return_shared_modules = None, default_dtype = torch.bfloat16, ignore_unused_weights = False, configKwargs ={}, quantize_exclude = None, pre_load_callback = None, fp32_dtype = None):
    """
    quick version of .LoadfromPretrained of  the transformers library
    used to build a model and load the corresponding weights (quantized or not)
    """       

    
    _report_loading("Reading Configuration", 0, 2, model_path)
    import os.path
    if not isinstance(model_path, list):
        model_path = [model_path]


    if not builtins.all(file_name.endswith(".sft") or file_name.endswith(".safetensors") or file_name.endswith(".pt") or file_name.endswith(".bin") or file_name.endswith(".ckpt") or get_extension_handler(file_name) is not None for file_name in model_path):
        raise Exception(f"File Extension of file {model_path} is not supported")

    model_path = [ _get_model(file) for file in model_path] 
    if any( file == None for file in model_path):
        raise Exception(f"Unable to find file {model_path}")
    
    verboseLevel = _compute_verbose_level(verboseLevel)
    if model_path[-1].endswith(".sft") or model_path[-1].endswith(".safetensors"):
        with safetensors2.safe_open(model_path[-1], writable_tensors =writable_tensors) as f:
            metadata = f.metadata() 
    else:
        metadata = None

    if metadata is None:
        transformer_config = None
    else:
        transformer_config = metadata.get("config", None)

    if transformer_config == None or forcedConfigPath != None:
        if forcedConfigPath != None:
            config_fullpath = forcedConfigPath
        else:
            config_fullpath =  os.path.join(os.path.dirname(model_path[-1]), "config.json") if defaultConfigPath == None else defaultConfigPath

        if not os.path.isfile(config_fullpath):
            raise Exception("a 'config.json' that describes the model is required in the directory of the model or inside the safetensor file")

        with open(config_fullpath, "r", encoding="utf-8") as reader:
            text = reader.read()
        transformer_config= json.loads(text)

    transformer_config.update( configKwargs )

    if "architectures" in transformer_config: 
        architectures = transformer_config["architectures"]
        class_name = architectures[0] 
        if modelClass !=None:
            transfomer_class = modelClass
        else:
            module = __import__("transformers")
            map = {  "T5WithLMHeadModel" : "T5EncoderModel"}
            class_name = map.get(class_name, class_name)
            transfomer_class = getattr(module, class_name)
        from transformers import AutoConfig

        import tempfile
        with tempfile.NamedTemporaryFile("w", delete = False,  encoding ="utf-8") as fp: 
            fp.write(json.dumps(transformer_config))
            fp.close()
            config_obj = AutoConfig.from_pretrained(fp.name)     
        os.remove(fp.name)
        #needed to keep inits of non persistent buffers
        _report_loading("Building", 1, 2, model_path)
        with init_empty_weights():
            model = transfomer_class(config_obj)

    else:
        if modelClass !=None:
            transfomer_class = modelClass
        elif "_class_name" in transformer_config:
            class_name  = 'Transformer3DModel'
            module = __import__("diffusers")
            transfomer_class = getattr(module, class_name)
        else:
            raise Exception("class not defined")                

        _report_loading("Building", 1, 2, model_path)
        with init_empty_weights():
            model = transfomer_class.from_config(transformer_config )


    model.eval().requires_grad_(False)

    model._config = transformer_config

    load_model_data(model,model_path, do_quantize = do_quantize, quantizationType = quantizationType, quantize_exclude = quantize_exclude, pinToMemory= pinToMemory, partialPinning= partialPinning, modelPrefix = modelPrefix, writable_tensors =writable_tensors, preprocess_sd = preprocess_sd, fused_split_map = fused_split_map, modules = modules, return_shared_modules =  return_shared_modules, default_dtype = default_dtype, ignore_unused_weights = ignore_unused_weights, verboseLevel=verboseLevel, pre_load_callback=pre_load_callback, fp32_dtype = fp32_dtype )
    model.eval().requires_grad_(False)

    return model

def flush_torch_caches():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except torch.cuda.CudaError:
            pass
        pinned.collect()
        # for idx in range(torch.cuda.device_count()):
        #     with torch.cuda.device(idx):
        torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        # torch.cuda.reset_peak_memory_stats()
    try:
        torch._C._host_emptyCache()
    except AttributeError:
        pass
    if os.name == "nt" and False: # suspicion of crash
        try:
            import ctypes, ctypes.wintypes as wintypes, os as _os
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_QUERY_INFORMATION = 0x0400
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, _os.getpid())
            if handle:
                psapi.EmptyWorkingSet(handle)
                kernel32.CloseHandle(handle)
        except Exception:
            pass
    from .allocator import ram
    if not ram.active:  # with PyTorch's CPU allocator: large dummy allocations push it to give freed memory back
        from accelerate import init_empty_weights
        with init_empty_weights():
            try:
                for _ in range(3):
                    dummy_tensor = torch.nn.Embedding(256384, 1024)
                    dummy_tensor = None
            except Exception:
                pass
            dummy_tensor = None

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    ram.release()  # the MMGP RAM allocator's freed blocks go back to the system


def map_state_dict(state_dict, rules):

    def map_one_sd(sd):
        if sd is None: return None
        for rule, repl in rules.items():
            new_sd= {}
            new_start = len(rule)
            prefix = rule + "."
            for k,v in sd.items():
                if k.startswith(prefix):
                    if repl is not None:
                        if len(repl) == 0:
                            k= k[new_start+1:]
                        else:            
                            k = repl + k[new_start:]
                        if isinstance(v, list):
                            new_v = []
                            for sub in v:
                                if sub.startswith(prefix):
                                    if len(repl) == 0:
                                        sub= sub[new_start+1:]
                                    else:            
                                        sub = repl + sub[new_start:]
                                new_v.append(sub)
                            v = new_v

                        new_sd[k] = v
                else:
                    new_sd[k] = v
            sd = new_sd
        return sd
    
    if isinstance(state_dict, list):
        return [map_one_sd(sd) for sd in state_dict]
    else:
        return map_one_sd(state_dict)

def filter_state_dict_basic(state_dict, base_model_prefix, keep_prefix = False):
    new_state_dict= {}
    start = -1
    if keep_prefix:
        for k,v in state_dict.items():
            if k.startswith(base_model_prefix):
                new_state_dict[k] = v
    else:
        for k,v in state_dict.items():
            if k.startswith(base_model_prefix):

                new_start = len(base_model_prefix)
            else:
                pos = k.find("." + base_model_prefix)
                if pos < 0:
                    continue
                new_start = pos + len(base_model_prefix)  +1
            if start != -1 and start != new_start:
                new_state_dict  = state_dict
                break
            start = new_start  
            new_state_dict[k[ start:]] = v
    return new_state_dict

def load_sd(file_path, filters = None, keep_prefixes = False, writable_tensors = True):
    state_dict, metadata = _safetensors_load_file(file_path, writable_tensors =writable_tensors)
    quantization_map = None
    tied_weights_map = None
    if metadata !=  None:
        quantization_map = metadata.get("quantization_map", None)
        tied_weights_map = metadata.get("tied_weights_map", None)

    if filters is not None:
        if not isinstance(filters, list): filters = [filters]
        new_sd = {}
        new_quantization_map = {}
        new_tied_weights_map = {}
        for one_filter in filters:
            new_sd.update(filter_state_dict_basic(state_dict, one_filter, keep_prefixes))
            if quantization_map is not None:
                new_quantization_map.update(filter_state_dict_basic(quantization_map, one_filter, keep_prefixes))
            if tied_weights_map is not None:
                new_tied_weights_map.update(filter_state_dict_basic(tied_weights_map, one_filter, keep_prefixes))
        state_dict = new_sd
        quantization_map = new_quantization_map if len(new_quantization_map) else None
        tied_weights_map = new_tied_weights_map if len(new_tied_weights_map) else None
    return state_dict, quantization_map, tied_weights_map


@cudacontext("cpu")
def load_model_data(model, file_path, do_quantize = False, quantizationType = qint8, pinToMemory = False, partialPinning = False, modelPrefix = None, writable_tensors = True,  preprocess_sd = None, postprocess_sd = None, fused_split_map = None, modules = None, return_shared_modules = None, default_dtype = torch.bfloat16, ignore_unused_weights = False, verboseLevel = -1, ignore_missing_keys = False, quantize_exclude = None, pre_load_callback = None, fp32_dtype = None):
    """
    Load a model, detect if it has been previously quantized using quanto and do the extra setup if necessary
    fp32_dtype: only with default_dtype=None (checkpoint-native dtypes). FP32 parameters are converted to this dtype,
    which is also the compute dtype of quantized layers. None (default) keeps the previous behavior.
    """
    quant_dtype = fp32_dtype if default_dtype is None and fp32_dtype is not None else default_dtype
    _report_loading("Reading Weights", 0, 3, file_path)
    if isinstance(preprocess_sd, dict):
        preprocess_fn = lambda sd, qm, twm: map_state_dict([sd, qm, twm], rules=preprocess_sd)
    else:
        preprocess_fn = preprocess_sd 
    if not isinstance(file_path, list):
        file_path = [file_path]

    file_count =  len(file_path)
    if isinstance(modules, (list,str)):
        if isinstance(modules, str): modules = [modules]
        file_path += modules
        modules = None

    normalized_paths = []
    for file in file_path:
        if isinstance(file, (dict, tuple)):
            normalized_paths.append(file)
        else:
            resolved = _get_model(file)
            if isinstance(resolved, str):
                resolved = normalize_extension_path(resolved)
            normalized_paths.append(resolved)
    file_path = normalized_paths
    if any(file is None for file in file_path):
        raise Exception(f"Unable to find file {file_path}")
    verboseLevel = _compute_verbose_level(verboseLevel)

    model = _remove_model_wrapper(model)

    if return_shared_modules is not None:
        return_state_dict ={}
        return_quantization_map ={}
        return_shared_modules["state_dict"] = return_state_dict 
        return_shared_modules["quantization_map"] = return_quantization_map 

    full_quantization_map = {}
    full_tied_weights_map = {}
    full_state_dict = {}
    for no, file in enumerate(file_path):
        if no:
            _report_loading("Reading Weights", 0, 3, file)
        quantization_map = None
        hybrid_quantization_map = False
        tied_weights_map = None
        metadata = None
        detected_kind = None
        if isinstance(file, tuple):
            if len(file)==2:
                state_dict, quantization_map = file
            elif len(file)==3:
                state_dict, quantization_map, tied_weights_map = file
            else:
                raise Exception("Expected a tuple of (state_dict, quantization_map, tied_weights_map)")
        elif isinstance(file, dict):
            state_dict = file
        elif isinstance(file, str) and get_extension_handler(file) is not None:
            ext_handler = get_extension_handler(file)
            load_fn = getattr(ext_handler, "load_state_dict", None)
            if not callable(load_fn):
                ext = os.path.splitext(file)[1].lower().lstrip(".")
                raise Exception(f"Missing load_state_dict for *.{ext} handler")
            result = load_fn( file, writable_tensors=writable_tensors, verboseLevel=verboseLevel, default_dtype=quant_dtype, pin_to_memory=pinToMemory, )
            if isinstance(result, tuple):
                if len(result) == 2:
                    state_dict, quantization_map = result
                elif len(result) == 3:
                    state_dict, quantization_map, tied_weights_map = result
                else:
                    raise Exception("Expected a tuple of (state_dict, quantization_map, tied_weights_map)")
            else:
                state_dict = result
        elif not (".safetensors" in file or ".sft" in file):
            if pinToMemory:
                raise Exception("Pinning to memory while loading only supported for safe tensors files")
            state_dict = torch.load(file, weights_only=False, map_location="cpu")
            if "module" in state_dict:
                state_dict = state_dict["module"]
        else:
            basename = os.path.basename(file)

            if "-of-" in basename:
                file_parts= basename.split("-")
                parts_max = int(file_parts[-1][:5])
                state_dict = {}
                for i in range(1, parts_max + 1):
                    file_parts[1] = ("0000" + str(i))[:5]
                    sd, _ = _safetensors_load_file( os.path.join( os.path.dirname(file), "-".join(file_parts) ) , writable_tensors =writable_tensors)
                    state_dict.update(sd)
            else:
                state_dict, metadata = _safetensors_load_file(file, writable_tensors =writable_tensors)

        if metadata !=  None:
            if quantization_map is None:
                quantization_map = metadata.get("quantization_map", None)
            config = metadata.get("config", None)
            if config is not None:
                model._config = config
            if tied_weights_map is None:
                tied_weights_map = metadata.get("tied_weights_map", None)

        if quantization_map is None and isinstance(file, str):
            pos = str.rfind(file, ".")
            if pos > 0:
                quantization_map_path = file[:pos]
            quantization_map_path += "_map.json"

            if os.path.isfile(quantization_map_path):
                with open(quantization_map_path, 'r') as f:
                    quantization_map = json.load(f)

        _report_loading("Preparing Weights", 1, 3, file)
        if preprocess_fn != None:
            num_params = len(inspect.signature(preprocess_fn).parameters)
            state_dict = preprocess_fn(*[state_dict, quantization_map, tied_weights_map][:num_params])
            if isinstance(state_dict, (tuple,list)):
                if len(state_dict)==2: 
                    state_dict, quantization_map = state_dict
                else:
                    state_dict, quantization_map, tied_weights_map = state_dict
            hybrid_quantization_map = quantization_map is not None

        if tied_weights_map != None:
            for name, tied_weights_list in tied_weights_map.items():
                mapped_weight = state_dict[name]
                for tied_weights in tied_weights_list:
                    state_dict[tied_weights] = mapped_weight


        if quantization_map is None or hybrid_quantization_map :
            conv_result = detect_and_convert(state_dict, default_dtype=quant_dtype, verboseLevel=verboseLevel, metadata=metadata)
            detected_kind = conv_result.get("kind")
            if conv_result.get("kind") not in ("none", "quanto"):
                state_dict = conv_result["state_dict"]
                quantization_map = quantization_map or {}
                quantization_map.update(conv_result["quant_map"])
                conv_result = None

            if detected_kind in (None, "none") and isinstance(file, str) and (".safetensors" in file or ".sft" in file):
                try:
                    info = detect_safetensors_format(state_dict, verboseLevel=verboseLevel, metadata=metadata)
                    detected_kind = info.get("kind")
                except Exception:
                    detected_kind = detected_kind or None
            if detected_kind not in (None, "none") and isinstance(file, str):
                cache_quantization_for_file(file, detected_kind or "none")
        
        full_state_dict.update(state_dict)
        if quantization_map != None:
            full_quantization_map.update(quantization_map)
        if tied_weights_map != None:
            full_tied_weights_map.update(tied_weights_map)
        if return_shared_modules is not None and no >= file_count:
            return_state_dict.update(state_dict)
            if quantization_map is not None: return_quantization_map.update(quantization_map)

    if isinstance(modules, dict) :
        full_state_dict.update(modules["state_dict"])
        full_quantization_map.update(modules["quantization_map"])

    state_dict, quantization_map, tied_weights_map  = full_state_dict, full_quantization_map, full_tied_weights_map
    full_state_dict, full_quantization_map, full_tied_weights_map = None, None, None

    # deal if we are trying to load just a sub part of a larger model
    if postprocess_sd != None:
        num_params = len(inspect.signature(postprocess_sd).parameters)
        state_dict = postprocess_sd(*[state_dict, quantization_map, tied_weights_map][:num_params])
        if isinstance(state_dict, (tuple,list)):
            if len(state_dict)==2: 
                state_dict, quantization_map = state_dict
            else:
                state_dict, quantization_map, tied_weights_map = state_dict
        
    if modelPrefix != None:
        base_model_prefix = modelPrefix + "."
        state_dict = filter_state_dict_basic(state_dict,base_model_prefix)
        if quantization_map != None:
            quantization_map = filter_state_dict_basic(quantization_map,base_model_prefix)
        if tied_weights_map != None:
            tied_weights_map = filter_state_dict_basic(tied_weights_map,base_model_prefix)

    if fused_split_map:
        row_view_bases = []  # fused weights whose projections interleave head by head, kept fused: row views compute them in place
        state_dict, quantization_map = split_fused_weights(
            state_dict,
            quantization_map,
            fused_split_map,
            default_dtype=quant_dtype,
            verboseLevel=verboseLevel,
            row_view_bases=row_view_bases,
        )
        for base, info in row_view_bases:
            row_views.install(model, base, info, state_dict)

    _report_loading("Applying Weights", 2, 3, file_path)
    post_load_hooks = []
    if quantization_map:
        quantization_map, post_load_hooks = apply_pre_quantization(
            model,
            state_dict,
            quantization_map,
            default_dtype=quant_dtype,
            verboseLevel=verboseLevel,
        )
    if pre_load_callback is not None:
        pre_load_callback(model)
    parameter_dtypes = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        module_dtype = getattr(module, "_lock_dtype", default_dtype)
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            parameter_dtypes[f"{module_name}.{name}" if module_name else name] = getattr(parameter, "_lock_dtype", module_dtype)
    quantized_weights = {f"{name}.weight" for name, qconfig in quantization_map.items() if qconfig["weights"] != "none"}
    converted = {}
    for key in parameter_dtypes.keys() & state_dict.keys():
        tensor = state_dict[key]
        target_dtype = parameter_dtypes[key]
        if target_dtype is None and tensor.dtype == torch.float32:
            target_dtype = fp32_dtype
        if target_dtype is None or key in quantized_weights or isinstance(tensor, QTensor) or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32) or tensor.dtype == target_dtype:
            continue
        ref = id(tensor)
        if ref not in converted:
            converted[ref] = tensor.to(target_dtype)
        state_dict[key] = converted[ref]
    converted = None

    if len(quantization_map) == 0:
        if any(isinstance(file, str) and "quanto" in file for file in file_path) and not do_quantize:
            print("Model seems to be quantized by quanto but no quantization map was found whether inside the model or in a separate '{file_path[:json]}_map.json' file")
    else:
        _requantize(model, state_dict, quantization_map, default_dtype=quant_dtype)    

    missing_keys , unexpected_keys = model.load_state_dict(state_dict, False,  assign = True )
    if len(missing_keys) > 0  :
        # if there is a key mismatch maybe we forgot to remove some prefix
        base_model_prefix = None
        for k,v in state_dict.items():
            if k.endswith(missing_keys[0]):
                base_model_prefix = k[:-len(missing_keys[0])]
                break
        if base_model_prefix == None:
            if not ignore_missing_keys:
                raise Exception(f"Missing keys: {missing_keys}")
        else:
            state_dict = filter_state_dict_basic(state_dict, base_model_prefix)
            missing_keys , unexpected_keys = model.load_state_dict(state_dict, False,  assign = True )
            if len(missing_keys) > 0 and not ignore_missing_keys:
                raise Exception(f"Missing keys: {missing_keys}")
        
    del state_dict

    if post_load_hooks:
        for hook in post_load_hooks:
            try:
                hook(model)
            except Exception as e:
                if verboseLevel >= 2:
                    print(f"Post-load hook skipped: {e}")

    if len(unexpected_keys) > 0 and verboseLevel >=2 and not ignore_unused_weights:
        print(f"Unexpected keys while loading '{file_path}': {unexpected_keys}")

    for k,p in model.named_parameters():
        if p.is_meta :
            txt  = f"Incompatible State Dictionary or 'Init_Empty_Weights' not set since parameter '{k}' has no data"
            raise Exception(txt)
    for k,b in model.named_buffers():
        if b.is_meta :
            txt  = f"Incompatible State Dictionary or 'Init_Empty_Weights' not set since buffer '{k}' has no data"
            raise Exception(txt)
        
    if return_shared_modules is not None:
        mods = { k : v for k,v in model.named_modules()}
        return_parameters = {}
        return_shared_modules["parameters"] = return_parameters
        for k in return_state_dict:
            if k.endswith("._data"):
                k = k[:-6]
            pos = k.rfind(".")
            mod_name = k[:pos]
            param_name =  k[pos +1:]
            mod = mods.get(mod_name, None)
            if mod is not None:
                p =  mod._parameters.get(param_name, None)
                if p is None: p =  mod._buffers.get(param_name, None)
                if p is not None:
                    return_parameters[k] = p
        del mods
        
    if isinstance(modules, dict) :
        mods = { k : v for k,v in model.named_modules()}
        # replace Parameter outer shell so that both models parameters are tied
        for k, rep_p in modules["parameters"].items():
            pos = k.rfind(".")
            mod_name = k[:pos]
            param_name =  k[pos +1:]
            mod = mods.get(mod_name, None)
            if mod is not None:
                setattr(mod, param_name, rep_p)
        del mods 
        modules["parameters"].clear()
        modules["state_dict"].clear()
        rep_p = p = None

    if do_quantize:
        if quantization_map != None and len(quantization_map) > 0 :
            if verboseLevel >=1:
                print("Model already quantized")
        else:
            if _quantize(model, quantizationType, verboseLevel=verboseLevel, model_id=file_path, quantize_exclude=quantize_exclude):
                quantization_map = model._quanto_map  

    if pinToMemory:
        context = _loading_context.get()
        pinning_callback = None if context is None else LoadingCallback(context[0].abort_requested, lambda phase, completed, total, _: _report_loading(phase, completed, total, file_path))
        _pin_to_memory(model, file_path, partialPinning = partialPinning, verboseLevel = verboseLevel, loading_callback=pinning_callback)

    _report_loading("Weights Loaded", 3, 3, file_path)
    return

def save_model(model, file_path, do_quantize = False, quantizationType = qint8, verboseLevel = -1, config_file_path = None, filter_sd =None, quantize_exclude = None):
    """save the weights of a model and quantize them if requested
    These weights can be loaded again using 'load_model_data'
    """       
    
    config = None
    extra_meta = None
    verboseLevel = _compute_verbose_level(verboseLevel)
    if config_file_path !=None:
        with open(config_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
            config= json.loads(text)
    elif hasattr(model, "_config"):
        config = model._config
    elif hasattr(model, "config"):
        config_fullpath = None
        config_obj = getattr(model,"config")
        config_path = getattr(config_obj,"_name_or_path", None)
        if config_path != None:
            config_fullpath = os.path.join(config_path, "config.json")      
            config_fullpath = _get_model(config_fullpath)

            # if not os.path.isfile(config_fullpath):
            #     config_fullpath = None
        if config_fullpath is None:                            
            config_fullpath =  os.path.join(os.path.dirname(file_path), "config.json")
        if os.path.isfile(config_fullpath):
            with open(config_fullpath, "r", encoding="utf-8") as reader:
                text = reader.read()
                config= json.loads(text)

    if do_quantize:
        _quantize(model, weights=quantizationType, model_id=file_path, verboseLevel=verboseLevel, quantize_exclude=quantize_exclude)
    
    quantization_map = getattr(model, "_quanto_map", None)

    from collections import OrderedDict

    cache_ref = {}
    tied_weights_map = {}
    sd = model.state_dict()
    if filter_sd  != None:
        new_sd = {}
        new_quantization_map = {}
        for k_k in filter_sd:
            for s in [".weight", ".bias", ".weight._data", ".weight._scale"]:                
                if k_k.endswith(s): 
                    k_k= k_k[:-len(s)]
                    break
            for k,v in sd.items():
                if k.startswith(k_k):
                    new_sd[k] = v
            if quantization_map != None:
                for k,v in quantization_map.items():
                    if k.startswith(k_k):
                        new_quantization_map[k] = v
        sd = new_sd
        if quantization_map != None: quantization_map = new_quantization_map

    out_sd = OrderedDict()


    for name, weight  in sd.items():
        ref = _get_tensor_ref(weight)
        match = cache_ref.get(ref, None)
        if match != None:
            tied_list = tied_weights_map.get(match, [])
            tied_list.append(name)
            tied_weights_map[match] = tied_list 
        else:
            out_sd[name] = weight 
            cache_ref[ref] = name

    if len(tied_weights_map) > 0:
        extra_meta = { "tied_weights_map" : tied_weights_map }

    if verboseLevel >=1:
        print(f"Saving file '{file_path}")

    safetensors2.torch_write_file(out_sd,  file_path , quantization_map = quantization_map, config = config, extra_meta= extra_meta)
    if verboseLevel >=1:
        print(f"File '{file_path}' saved")


def extract_models(obj = None, prefix = None):
    if isinstance(obj, str): # for compatibility as the two args were switched
        bkp = prefix
        prefix = obj
        obj = bkp

    pipe = {}
    if obj == None:
        raise Exception("an object to analyze must be provided")
    if prefix==None or len(prefix)==0:
        prefix = ""
    elif prefix[ -1:] != "/":
        prefix  + "/"        
    
    for name in dir(obj):
        if name in ["_execution_device"]:
            continue            
        element = getattr(obj,name)
        if name  in ("pipeline", "pipe"):
            pipeline = element
            if  hasattr(pipeline , "components") and isinstance(pipeline.components, dict):
                for k, model in pipeline.components.items():
                    if model != None:
                        pipe[prefix  + k ] = model
        elif isinstance(element, torch.nn.Module) and name!="base_model": 
            if prefix + name in pipe:
                pipe[prefix + "_" + name ] = element
            else:
                pipe[prefix + name ] = element
        elif isinstance(element, dict):
            for k, element in element.items():
                if  hasattr(element , "pipeline"):
                    pipe.update( extract_models(prefix + k,element ))


    return pipe

def get_model_name(model):
    return model.name

class HfHook:
    def __init__(self):
        self.execution_device = "cuda"

    def init_hook(self, module):
        return module

    def detach_hook(self, module):
        return module
    
def _mm_lora_linear_forward(module, *args, **kwargs):
    loras_data = getattr(module, "_mm_lora_data", None)
    if args:
        inp = args[0]
    else:
        inp = kwargs.get("input", None)
    weight = getattr(module, "weight", None)
    if torch.is_tensor(inp) and torch.is_tensor(weight):
        weight_dtype = module.__dict__.get("_mm_weight_dtype", None)
        if weight_dtype is None: # constant once hooked, and reading it from a quantized weight costs a Python dispatch
            weight_dtype = module._mm_weight_dtype = weight.dtype
        if inp.dtype != weight_dtype and inp.dtype.is_floating_point and weight_dtype.is_floating_point:
            inp = inp.to(weight.dtype)
            if args:
                args = (inp,) + args[1:]
            else:
                kwargs = dict(kwargs)
                kwargs["input"] = inp
    loras_data = getattr(module, "_mm_lora_data", None)
    if not loras_data:
        return module._mm_lora_old_forward(*args, **kwargs)
    if not hasattr(module, "_mm_manager"):
        pass
    return module._mm_manager._lora_linear_forward(
        module._mm_lora_model,
        module,
        loras_data,
        *args,
        **kwargs,
    )


def _mm_lora_generic_forward(module, *args, **kwargs):
    loras_data = getattr(module, "_mm_lora_data", None)
    if not loras_data:
        return module._mm_lora_old_forward(*args, **kwargs)
    return module._mm_manager._lora_generic_forward(
        module._mm_lora_model,
        module,
        loras_data,
        module._mm_lora_old_forward,
        *args,
        **kwargs,
    )


# Linear modules computed by ranges of output rows from an input prepared once (for instance attention one group of heads at a
# time): a row projection provider prepares the activation for its weight format once (e.g. quantized rows) and multiplies it by a
# range of weight rows; LoRAs reuse one x @ A per adapter. The result equals the same rows of module(x).
_ROW_PROJECTIONS = []


def register_row_projection(provider):
    """provider: supports(module, x) -> bool for x of shape (tokens, in_features); key(module) -> the hashable format of the prepared
    activation, shared by the modules that use it, or None to use x itself; prepare(module, x) -> prepared activation;
    rows(module, prepared, start, stop) -> output rows [start, stop) with their bias, shape (tokens, stop - start)."""
    if provider not in _ROW_PROJECTIONS:
        _ROW_PROJECTIONS.insert(0, provider)


def unregister_row_projection(provider):
    if provider in _ROW_PROJECTIONS:
        _ROW_PROJECTIONS.remove(provider)


class _PlainRowProjection:
    """Floating point torch.nn.Linear weights."""
    @staticmethod
    def supports(module, x):
        return type(module) is torch.nn.Linear and module.weight.dtype == x.dtype

    @staticmethod
    def key(module):
        return None

    @staticmethod
    def prepare(module, x):
        return None

    @staticmethod
    def rows(module, x, start, stop):
        bias = module.bias
        return torch.nn.functional.linear(x, module.weight[start:stop], None if bias is None else bias[start:stop])


_PLAIN_ROW_PROJECTION = _PlainRowProjection()


def _row_projection(module, x):
    return next((provider for provider in _ROW_PROJECTIONS + [_PLAIN_ROW_PROJECTION] if provider.supports(module, x)), None)


def _row_lora_adapters(module):
    """(adapter, data, scaling) of the module's active LoRAs as _lora_linear_forward applies them, or None when one of them has no
    row equivalent (DoRA merges weights, LoKr)."""
    loras_data = getattr(module, "_mm_lora_data", None)
    if not loras_data:
        return []
    model = module._mm_lora_model
    adapters = []
    for adapter in model._loras_active_adapters:
        data = loras_data.get(adapter + '_GPU', None)
        if data is None:
            continue
        if data[3] is not None or data[5].get("type", "lora") == "lokr":
            return None
        scaling = module._mm_manager._get_lora_scaling(model._loras_scaling, model, adapter) * data[4]
        if scaling != 0:
            adapters.append((adapter, data, scaling))
    return adapters


class LinearInput:
    """Input prepared once for linear modules computed by ranges of output rows (prepare_linear_input, linear_rows)."""
    def __init__(self, shape, x, prepared, providers, lora):
        self.shape, self.x, self.prepared, self.providers, self.lora = shape, x, prepared, providers, lora


def linear_rows_supported(modules, x):
    """True when every module can be computed by ranges of output rows from x (*, in_features), LoRAs included."""
    x = x.reshape(-1, x.shape[-1])
    return builtins.all(_row_projection(module, x) is not None and _row_lora_adapters(module) is not None for module in modules)


def prepare_linear_input(x_list, modules):
    """Prepares x, handed off in x_list, for modules computed by ranges of rows (check linear_rows_supported first): the activation of
    each weight format once, x @ A once per LoRA adapter. x itself is kept only for plain weights and diff adapters."""
    x = x_list[0]
    x_list.clear()
    shape, x = x.shape[:-1], x.reshape(-1, x.shape[-1])
    prepared, providers, lora, keep_x = {}, {}, {}, False
    for module in modules:
        provider = providers[module] = _row_projection(module, x)
        key = provider.key(module)
        if key is None:
            keep_x = True
        elif key not in prepared:
            prepared[key] = provider.prepare(module, x)
        for adapter, data, scaling in _row_lora_adapters(module):
            if data[5].get("type", "lora") == "diff":
                keep_x = True
            elif data[0] is not None:
                lora[(module, adapter)] = x @ data[0].to(x.dtype).T
    return LinearInput(shape, x if keep_x else None, prepared, providers, lora)


def linear_rows(module, linear_input, start, stop):
    """Output rows [start, stop) of module(x) for a prepared input, LoRAs included, shape (*x.shape[:-1], stop - start)."""
    provider = linear_input.providers[module]
    key = provider.key(module)
    out = provider.rows(module, linear_input.x if key is None else linear_input.prepared[key], start, stop)
    for adapter, data, scaling in _row_lora_adapters(module):
        lora_A, lora_B, diff_b = data[:3]
        if data[5].get("type", "lora") == "diff":
            out.addmm_(linear_input.x, lora_B[start:stop].to(out.dtype).T, beta=1, alpha=scaling)
        elif lora_A is not None:
            out.addmm_(linear_input.lora[(module, adapter)], lora_B[start:stop].to(out.dtype).T, beta=1, alpha=scaling)
        if diff_b is not None:
            out.add_(diff_b[start:stop].to(out.dtype), alpha=scaling)
    return out.view(*linear_input.shape, stop - start)


last_offload_obj = None
class offload:
    supports_cotenant_wildcards = True
    def __init__(self):
        _managers.add(self)
        self.pages_dropped = set() # its models whose checkpoint pages left the working set for another offload object's models, since they last ran
        self.active_models = []
        self.active_models_ids = []
        self.models = {}
        self.cotenants_map = { 
                            "text_encoder": ["vae", "text_encoder_2"],
                            "text_encoder_2": ["vae", "text_encoder"],                             
                        }
        self.verboseLevel = 0
        self.blocks_of_modules = {}
        self.blocks_of_modules_sizes = {}
        self.anyCompiledModule = False
        self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory
        self.last_reserved_mem_check =0
        self.loaded_blocks = {}
        self.prev_blocks_names = {}
        self.next_blocks_names = {}
        self.preloaded_blocks_per_model = {}
        self.default_stream = torch.cuda.default_stream(torch.device("cuda")) # torch.cuda.current_stream()
        self.transfer_stream = torch.cuda.Stream()
        self.async_transfers = False
        self.parameters_ref  = {}
        self.max_reservable_memory = 0
        self.stager = _PinnedStager()
        self.staged_entries = {}
        self.state_records = {} # block entry -> [(state ring, block index)]: pageable state records staged in the ring with the block (mmgp.state_ring)
        self.state_rings = weakref.WeakSet() # the state rings that may keep state in VRAM (made with resident=True)
        self.state_rooms = {} # model id -> VRAM its budget leaves beyond the blocks it preloads (tune_preloading): room for state kept in VRAM
        self.prefetch_executor = None
        self.prefetch_futures = {}
        self.main_waiting = False
        self.prefetch_window = 0
        self.prefetch_windows = {}
        self.planned_windows = {} # model id -> prefetch window planned with its budget
        self.inflight = {}
        self.ring = None
        self.ring_check = None # (state, whether the staging ring has nothing left to stage), see _ring_unused
        self.ring_bounds = {} # model id -> bytes of the largest block of its towers in the staging ring
        self.ring_on_demand = None # (reservable RAM, verbose level) while a ring may still be made for blocks streamed later (smartPinning)
        self.ring_refused = None # the pinned pool's epoch at which that ring was refused: tried again after the next quiet point
        self.block_items = {}
        self.block_slots = {} # model id -> [VRAM slot (None until used), entry name using it or None, slot size] for its swapped blocks
        self.slot_views = {} # (entry name, slot number) -> the parameters of the block as views of the slot
        self.towers_per_model = {} # model id -> names of its towers, the recurrent blocks for which the VRAM slots are sized
        self.shuttle_sizes = {} # model id -> size of the largest block of its towers
        self.learned_links = set() # blocks whose next block is the one that came next when they last ran: the ends of the towers
        self.auto_preload = None # with autoPreload, the plans of the models whose preload follows the VRAM left by their workloads
        self.auto_preload_requested = self.preload_plans = False
        self.cuda_graphs = {} # model id -> the CUDA graphs of the blocks of its towers (cudaGraphs)
        self.expert_residency = {}
        self.expert_hooks = []
        self.cpu_compute = {} # model id -> the weights its blocks compute on the CPU when they are not in VRAM (cpuCompute)
        self.cpu_compute_forced = False # cpuComputeForce: the CPU computes the short calls whatever the measured times
        self.cpu_blocks = {} # model id -> its block computing on the CPU now
        self.cpu_block_calls = {} # model id -> its blocks computed on the CPU so far (one in CPU_COMPUTE_TIMED is timed)
        self.cpu_short = {} # model id -> whether its blocks run a short call that the CPU computes (cpuCompute): only its streamed blocks are transferred ahead
        self.cpu_seconds = {} # (model id, rows) -> seconds per byte of its blocks computed on the CPU, measured
        self.cpu_samples = {} # (model id, rows) -> seconds per byte of the blocks measured so far; model id -> the blocks of warm-up left
        self.cpu_plans = {} # (model id, rows) -> (key, whether the CPU computes, the pinned blocks streamed meanwhile, first block out of VRAM) of short calls
        self.cpu_climbs = {} # (model id, rows) -> _PassClimb of the count of blocks out of VRAM streamed during its short calls
        self.cpu_measuring = {} # model id -> whether its current short call measures the CPU (one block out of VRAM in CPU_COMPUTE_TIMED)
        self.transfers = 0 # blocks moved as one buffer (_move_block): one in PCIE_TIMED_TRANSFERS is timed for _pcie_rate
        self.cpu_pass_start = {} # model id -> (rows, start time, streamed offset) of its short call running, from its first block out of VRAM
        self.cpu_streamed = {} # model id -> the pinned blocks streamed during its short call running
        self.cpu_compute_entries = {} # model id -> (block entry, its weights computed on the CPU) of its towers, in order
        self.pool_models = () # models whose streamed blocks enter the pinned pool (pinPool)
        self.pool_limits = {} # model id -> its blocks out of VRAM that cpuCompute's short calls stream: the most the pool takes for it
        self.pool_ranked = {} # model id -> (count of its blocks kept in VRAM, its blocks out of VRAM in the order the pool takes them)
        self.pool_others = {} # model id -> (count of its blocks kept in VRAM, its blocks out of VRAM outside its towers)
        self.pool_wanted = {} # model id -> (key, the blocks the pool may take for it)
        self.pool_plans = {} # entry name -> how its block enters the pool (_PoolPlan), made at profile time for the blocks of pool models
        self.mapped_pages = {} # entry name -> (layout, the pages of its tensors mapped from a checkpoint file), of the models that run once or twice per generation
        self.mapped_bytes = {} # model id -> bytes of these pages, of all its blocks
        self.used = {} # model id -> stamp of its last use (memory_pressure.stamp): the least recently used models give their VRAM first
        self.parked = False # park(): its models stay loaded while others use the GPU, their VRAM reclaimed from any thread as it is needed
        self.park_release = None # with park(): "models", "blocks" or the owner's release callable (see park)
        self.park_state_bytes = 0 # with a release callable: the VRAM the owner holds beside the models, freed by its release
        self.park_lock = threading.RLock() # reclaims of the parked models (any thread) against unpark and unload_all
        self.parked_size = None # parked_bytes() while it is unchanged

        global last_offload_obj
        last_offload_obj = self

        self._type_wrappers = {}
        
    def add_module_to_blocks(self, model_id, blocks_name, submodule, prev_block_name, submodule_name):

        if blocks_name!=None and ".lora_" in blocks_name:
            blocks_name = None
        entry_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        if entry_name in self.blocks_of_modules:
            blocks_params = self.blocks_of_modules[entry_name]
            blocks_params_size = self.blocks_of_modules_sizes[entry_name]
        else:
            blocks_params = []
            self.blocks_of_modules[entry_name] = blocks_params
            blocks_params_size = 0
            if blocks_name !=None:
                prev_entry_name = None if prev_block_name == None else  model_id + "/" + prev_block_name
                self.prev_blocks_names[entry_name] =  prev_entry_name
                if not prev_block_name == None:
                    self.next_blocks_names[prev_entry_name] = entry_name        
        bef = blocks_params_size

        for k,p in submodule.named_parameters(recurse=False):
            if _is_mapped(p): # never moved: the GPU reads it in mapped RAM
                continue
            param_size = 0
            ref = _get_tensor_ref(p)
            tied_param =  self.parameters_ref.get(ref, None)
            blocks_params.append((submodule, k, p, False, tied_param))
            sub_tensors = _get_quantized_subtensors(p)
            if sub_tensors:
                param_size += _subtensors_nbytes(sub_tensors)
                del sub_tensors
            else:
                param_size += torch.numel(p.data) * p.data.element_size()


            if tied_param is None:
                blocks_params_size +=  param_size
                self.parameters_ref[ref] = (submodule, k)

        for k, p in submodule.named_buffers(recurse=False):
            ref = _get_tensor_ref(p)
            tied_param =  self.parameters_ref.get(ref, None)
            blocks_params.append( (submodule, k, p, True, tied_param) )
            if tied_param is None:
                blocks_params_size += p.data.nbytes
                self.parameters_ref[ref] = (submodule, k)

        aft = blocks_params_size

        # if blocks_name is None:
        #     print(f"Default: {model_id}/{submodule_name} : {(aft-bef)/ONE_MB:0.2f} MB")
        #     pass


        self.blocks_of_modules_sizes[entry_name] = blocks_params_size


        return blocks_params_size


    def needs_staging(self, entry_name):
        if _pool.involved(self, entry_name): # in the pinned pool or entering it (its tensors locked in place may already report pinned)
            return _pool.entry(self, entry_name) is None or self._block_items(entry_name)[2] is None
        needs = self.staged_entries.get(entry_name, None)
        if needs is None:
            needs = self.staged_entries[entry_name] = builtins.any(tied_param is None and _pageable_nbytes(p) >= STAGING_MIN_SIZE for _, _, p, _, tied_param in self.blocks_of_modules[entry_name])
        return needs

    def _pinning_changed(self, model_id):
        # some of the model's buffers pinned for good were reclaimed or came back: its blocks' cached layouts and staging needs are made
        # again, and a staging ring for the blocks no longer pinned when they first stream
        prefix = model_id + "/"
        self.block_items = {key: found for key, found in self.block_items.items() if not key[0].startswith(prefix)}
        self.staged_entries = {entry: needs for entry, needs in self.staged_entries.items() if not entry.startswith(prefix)}
        if self.ring is None and self.ring_on_demand is None:
            self.ring_on_demand = (self.max_reservable_memory, self.verboseLevel)

    def _reclaimed_bytes(self, model_id):
        return builtins.sum(record.nbytes for record in getattr(self.models[model_id], "_mm_pinned_buffers", ()) if record.state != "locked")

    def _refill_pinned(self, model_id):
        # at its load, after the quiet point that made room for them: the model's reclaimed buffers come back, first ranked first, as far as
        # the reserved RAM and the pinned pool's RAM floor allow (the others cross the staging ring)
        records = sorted((record for record in getattr(self.models[model_id], "_mm_pinned_buffers", ()) if record.state != "locked"), key=lambda record: record.rank)
        refilled = 0
        with torch.inference_mode(False):
            for record in records:
                if not pinned.fits(record.nbytes, self.max_reservable_memory) or record.state == "freed" and psutil.virtual_memory().available - record.nbytes < POOL_RAM_FLOOR:
                    break
                if not (record.refill() if record.state == "freed" else record.lock()):
                    break
                refilled += record.nbytes
        if refilled:
            self._pinning_changed(model_id)
            if self.verboseLevel >= 1:
                tqdm.write(f"Pinned again {refilled/ONE_MB:0.0f} MB of '{model_id}' given back while it did not run" + ("" if refilled == builtins.sum(record.nbytes for record in records) else " (the rest crosses the staging ring)"))

    def state_room(self, model_id):
        # the VRAM the model's preload leaves to the state its rings keep in VRAM (mmgp.state_ring), weights first: with a dynamic preload,
        # what its plan leaves once every block is kept (nothing until measured, or while blocks stream); with a budget, what it leaves beyond
        # the blocks it preloads; nothing without a preload
        plan = self.auto_preload.plans.get(model_id) if self.auto_preload is not None else None
        return plan.state_room if plan is not None else self.state_rooms.get(model_id, 0)

    def state_bytes(self, model_id):
        # the VRAM of the state the model's rings keep in VRAM (and the slots of its dynamic rings): counted with its weights by the dynamic
        # preload
        return builtins.sum(ring.vram_bytes for ring in self.state_rings if ring.model_id == model_id)

    def state_changed(self, model_id):
        # the state the model's rings keep in VRAM changed: the dynamic preload's cycle running does not measure the activations
        if self.auto_preload is not None:
            self.auto_preload.on_state_change(model_id)

    def state_pressure(self, model_id, released, dynamic=False):
        # the state the model's rings kept in VRAM was released under VRAM pressure: the room of the next plans shrinks by as much (the
        # dynamic preload's state penalty of the workload, or its penalty for a dynamic ring, planned with the weights; a budget's room,
        # until the model is profiled again), not the weights' room
        plan = self.auto_preload.plans.get(model_id) if self.auto_preload is not None else None
        if plan is not None:
            self.auto_preload.on_state_pressure(plan, released, dynamic)
        else:
            self.state_rooms[model_id] = max(0, self.state_rooms.get(model_id, 0) - released)

    def model_prefetch_window(self, model_id):
        # pipelining applies to the models with pinned blocks or a staging ring for the others (without a ring, the blocks of a partially
        # pinned model that are not pinned are staged by the prefetch thread); one block ahead takes no more VRAM than the shuttle
        window = self.prefetch_windows.get(model_id, None)
        if window is None:
            entries = [entry for entry in self.blocks_of_modules if entry.startswith(model_id + "/")]
            largest = max((self.blocks_of_modules_sizes[entry] for entry in entries), default=0)
            pipelined = len(entries) > 0 and (self.ring is not None or not builtins.all(self.needs_staging(entry) for entry in entries))
            planned = self.planned_windows.get(model_id, 1 if self.prefetch_window is None else self.prefetch_window)
            window = 0 if not pipelined else planned if (planned - 1) * largest <= PREFETCH_VRAM_SHARE * self.device_mem_capacity else 1
            self.prefetch_windows[model_id] = window
        return window

    def submit_prefetch(self, entry_name, fn, *args):
        if self.prefetch_executor is None:
            self.prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmgp_prefetch", initializer=_prefetch_thread, initargs=(torch.cuda.current_device(),))
        self.prefetch_futures[entry_name] = self.prefetch_executor.submit(_run_in_grad_modes, torch.is_inference_mode_enabled(), torch.is_grad_enabled(), fn, *args)

    def wait_prefetch(self, entry_name=None):
        entries = list(self.prefetch_futures) if entry_name is None else [entry_name] if entry_name in self.prefetch_futures else []
        if entries:
            self.main_waiting = True # the prefetch thread may now copy with several threads
            try:
                for entry in entries:
                    memory_pressure.wait(self.prefetch_futures.pop(entry))
            finally:
                self.main_waiting = False

    def _load_pipelined(self, model_id, entry_name, cpu_to_gpu):
        # The host queues the copies of the next blocks without waiting for the GPU: they reuse the memory of the unloaded blocks once the
        # transfer stream has waited for the compute queued before (gpu_load_blocks), so VRAM holds window + 1 blocks like the one block shuttle.
        # The blocks that are not pinned are copied further ahead into the staging ring, if any.
        short = self.cpu_short.get(model_id, False) # the CPU computes the blocks out of VRAM that are not streamed: neither staged nor transferred
        streamed = self.cpu_streamed.get(model_id, ()) if short else ()
        if self.ring is not None: # short calls stage their streamed blocks only (outside the pool, they would be copied by this thread)
            self._schedule_ring(entry_name, streamed if short else None)
        consumer = torch.cuda.current_stream()
        if entry_name in self.inflight:
            transfer = self.inflight.pop(entry_name)
            consumer.wait_event(transfer if isinstance(transfer, torch.cuda.Event) else memory_pressure.wait(transfer))
            self._ensure_loras(model_id, entry_name)
        else:
            if self.shared_weight_stream:
                # Demand copies share the prefetch buffers' allocator heap. Order reuse after earlier compute and wait only for the
                # copy needed by this consumer; neither dependency synchronizes the host or the whole device.
                self.transfer_stream.wait_stream(consumer)
                transfer = self._transfer(self.transfer_stream, entry_name, cpu_to_gpu)
                consumer.wait_event(transfer if isinstance(transfer, torch.cuda.Event) else memory_pressure.wait(transfer))
            else:
                self._transfer(consumer, entry_name, cpu_to_gpu)
        next_entry, window, seen, wanted = entry_name, self.prefetch_windows[model_id], {entry_name}, []
        while window > 0:
            prev_entry, next_entry = next_entry, self.next_blocks_names.get(next_entry, None)
            if next_entry is None or next_entry in seen or prev_entry in self.learned_links and self.blocks_of_modules_sizes[next_entry] > self.shuttle_sizes[model_id]:
                break # a learned link is followed towards a block that is not larger than those of the towers, as the VRAM plan assumes
            seen.add(next_entry)
            if short and next_entry not in streamed:
                continue
            window -= 1
            wanted.append(next_entry)
        # a block transferred ahead that the calls did not take (another block came next) would keep its slot from the next blocks, whose
        # copies would then get buffers of their own (new tensors at each transfer: their CUDA graphs captured again): it goes back
        prefix = model_id + "/"
        for stale in [entry for entry in self.inflight if entry.startswith(prefix) and entry not in wanted]:
            self.gpu_unload_blocks(model_id, stale[len(prefix):])
        for next_entry in wanted:
            if next_entry not in self.inflight:
                if self.verboseLevel >= 2:
                    tqdm.write(f"Prefetching model {next_entry} ({self.models[model_id]._get_name()}) in GPU")
                self.inflight[next_entry] = self._transfer(self.transfer_stream, next_entry, cpu_to_gpu)

    def _transfer(self, stream, entry_name, cpu_to_gpu):
        # Queues the transfer of a block on stream and returns the event of its completion, or the future of this event for a block that is
        # neither pinned nor in a staging ring: the prefetch thread stages it while the GPU computes, as for the one block shuttle
        staged = None if self.ring is None else self.ring.take(entry_name)
        if staged is not None and staged[0] is None: # it entered the pinned pool: moved from there
            staged = None
        moved = self._move_block(stream, entry_name, staged if staged is not None and staged[1] else None) # no items: its state alone was staged
        if moved is None and self.ring is None and stream is self.transfer_stream and self.needs_staging(entry_name):
            blocks_params = self.blocks_of_modules[entry_name]
            def staged_transfer():
                cpu_to_gpu(stream, blocks_params, self.is_main_waiting)
                return stream.record_event()
            self.submit_prefetch(entry_name, staged_transfer)
            return self.prefetch_futures[entry_name]
        cpu_to_gpu(stream, self.blocks_of_modules[entry_name], moved=moved)
        if staged is None:
            return stream.record_event()
        region = staged[0]
        for ring, block, generation, offset, size in region[3] if len(region) > 3 else (): # the block's staged state into its slots
            ring.transfer_staged(block, generation, self.ring.buffer[region[0] + offset:region[0] + offset + size], stream)
        event = torch.cuda.Event(blocking=True) # the copier thread sleeps while it waits for the region to be free
        event.record(stream)
        self.ring.release(staged[0], event)
        return event

    def _move_block(self, stream, entry_name, staged):
        # One VRAM buffer for the parameters of a block, filled by a single copy from the region of the block in the staging ring or by one
        # copy per run of pinned memory, the parameters being rebuilt as views of it: an allocation and a copy per tensor would cost the
        # host more than the GPU spends on a short block. Returns the moved parameters by index, None if the block is not moved so.
        if staged is None:
            items, (_, offsets, nbytes), sources = self._block_items(entry_name)
            if not items or sources is None:
                return None
        else:
            region, items, (_, offsets, nbytes) = staged
        slot = self._take_slot(entry_name, nbytes) if CACHED_BLOCK_VIEWS else None
        model_id, block_name = entry_name.split("/", 1)
        adopted = self.adopt_slot(model_id, nbytes) if slot is None and self.auto_preload is not None and self.auto_preload.keeps_all(model_id, block_name) else None
        self.transfers = self.transfers + 1
        timed = self.transfers % PCIE_TIMED_TRANSFERS == 0 and model_id in self.cpu_compute and not torch.cuda.is_current_stream_capturing()
        with torch.cuda.stream(stream):
            buffer = self.block_slots[model_id][slot][0] if slot is not None else adopted if adopted is not None else torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            if timed:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream)
            if staged is None:
                for offset, source in sources:
                    buffer[offset:offset + source.numel()].copy_(source, non_blocking=True)
            else:
                buffer[:nbytes].copy_(self.ring.buffer[region[0]:region[0] + nbytes], non_blocking=True)
            if timed:
                end.record(stream)
                _pcie_timed.append((nbytes, start, end))
        if slot is None:
            return {index: _ring_view(t, buffer, offsets, 0) for index, t in items}
        views = self.slot_views.get((entry_name, slot), None)
        if views is None or views[0] is not items: # built once per slot, unless the block was laid out again
            views = self.slot_views[(entry_name, slot)] = (items, {index: _ring_view(t, buffer, offsets, 0) for index, t in items})
        return views[1]

    def _take_slot(self, entry_name, nbytes):
        # a free VRAM slot of the model of a swapped block, None if it is a resident block or does not fit: freed slots are reused in the
        # order of the transfer stream, which waits for the computation queued before (gpu_load_blocks), like the caching allocator
        model_id, block_name = entry_name.split("/", 1)
        if block_name in self.preloaded_blocks_per_model[model_id] or self.auto_preload is not None and self.auto_preload.own_buffer(model_id, block_name):
            return None # a resident block, or one the automatic preload keeps once transferred or streams without slots
        slots = self.block_slots.get(model_id, None)
        if slots is None: # [buffer, entry name using it, size], sized for the swapped blocks of the towers, the buffer allocated when used
            towers = tuple(model_id + "/" + name for name in self.towers_per_model[model_id])
            swapped = [entry for entry in self.prev_blocks_names if entry.startswith(towers) and entry.split("/", 1)[1] not in self.preloaded_blocks_per_model[model_id]]
            size = max((self._block_items(entry)[1][2] for entry in swapped), default=0)
            slots = self.block_slots[model_id] = [[None, None, size] for _ in range(self.model_prefetch_window(model_id) + 1)]
        if nbytes > slots[0][2]: # a larger block outside the towers keeps a buffer of its own, which the idle slots make room for
            for slot_no, slot in enumerate(slots):
                if slot[1] is None and slot[0] is not None:
                    slot[0] = None
                    self.slot_views = {key: views for key, views in self.slot_views.items() if key[1] != slot_no or not key[0].startswith(model_id + "/")}
            return None
        for slot_no, slot in enumerate(slots):
            if slot[1] is None:
                if slot[0] is None:
                    # in the pool of the block buffers, whose memory it can then reuse; not an inference tensor, as blocks are also loaded
                    # outside inference mode
                    with torch.inference_mode(False), torch.cuda.stream(self.transfer_stream):
                        slot[0] = torch.empty(slot[2], dtype=torch.uint8, device="cuda")
                slot[1] = entry_name
                return slot_no
        return None

    def adopt_slot(self, model_id, nbytes, entry_name=None):
        """A slot of the streamed blocks of a model that leaves the rotation to become the buffer of a block kept in VRAM, while every block
        of the model is to be kept: the slots would serve no streamed block any more, the kept blocks take their memory instead of new
        buffers beside them. entry_name: the slot holding that block, otherwise (None) an idle one, for a block of nbytes at least 3/4 of the
        slot's size (the slots are sized for the largest block: a much smaller block would leave most of it unused). Returns its buffer,
        None if there is none."""
        slots = self.block_slots.get(model_id)
        if not slots:
            return None
        fits = lambda slot: slot[0] is not None and 3 * slot[2] <= 4 * nbytes <= 4 * slot[2]
        slot_no = next((i for i, slot in enumerate(slots) if slot[1] == entry_name and fits(slot)), None)
        if slot_no is None:
            return None
        buffer = slots.pop(slot_no)[0]
        # the numbers of the slots that follow change: their views are rebuilt when used
        self.slot_views = {key: views for key, views in self.slot_views.items() if not key[0].startswith(model_id + "/")}
        if not slots:
            del self.block_slots[model_id]  # made again, sized for the streamed blocks, if a later plan streams some
        return buffer

    def _free_slot(self, entry_name):
        slots = self.block_slots.get(entry_name.split("/", 1)[0], None)
        if slots is not None:
            for slot in slots:
                if slot[1] == entry_name:
                    slot[1] = None

    def _block_items(self, entry_name):
        # The parameters of a block moved as one buffer, their layout and, if they are all pinned, views of their runs of pinned memory;
        # otherwise they cross the staging ring, all of them, as a copy from pageable memory, even a small one, makes the host wait until the
        # transfer stream reaches it, i.e. for the computation of the previous block. Classes that differ from their driver copy are left out.
        disabled = builtins.sum(1 for same in self.stager.checked_classes.values() if same is False)
        found = self.block_items.get((entry_name, disabled), None)
        if found is None:
            items, all_pinned = [], not _pool.involved(self, entry_name) # a block of the pinned pool keeps the layout of its pageable tensors
            for i, (_, _, p, _, tied_param) in enumerate(self.blocks_of_modules[entry_name]):
                t = p.data if type(p) is torch.nn.Parameter else p
                leaves = _staging_leaves(t) if tied_param is None and self.stager.checked_classes.get(type(t), None) is not False else None
                if leaves:
                    items.append((i, t))
                    all_pinned = all_pinned and builtins.all(leaf.is_pinned() for leaf in leaves)
            layout, sources = _ring_layout(items, by_storage=all_pinned), None
            if all_pinned:
                with torch.inference_mode(False):
                    sources = [(offset, torch.empty(0, dtype=torch.uint8, device="cpu").set_(leaf.untyped_storage(), src - leaf.untyped_storage().data_ptr(), (size,))) for src, size, offset, leaf in layout[0]]
            found = self.block_items[(entry_name, disabled)] = (items, layout, sources)
        if found[2] is None:
            pooled = _pool.entry(self, entry_name)
            if pooled is not None and pooled.layout is found[1]: # its pinned sources, made for this layout
                return found[0], found[1], pooled.sources
        return found

    def _schedule_ring(self, entry_name, only=None):
        # the copier stages the blocks that are not pinned (of only, if not None), from this one on in their order of use, as far ahead as the
        # ring holds them (a block used once, before the blocks of each step, leads into their cycle without being part of it: stop at any block seen). A
        # block that the pinned pool takes enters it from the copier, in this order, instead of being copied into the ring, while no other
        # block is being prepared for the pool: its transfer never waits for a queue of preparations (the others enter at a later pass).
        budget, entry, seen = self.ring.size, entry_name, set()
        model_id = entry_name.split("/", 1)[0]
        wanted = self._pool_wanted(model_id)
        while entry is not None and entry not in seen:
            seen.add(entry)
            if entry not in self.inflight and (only is None or entry in only):
                items, layout, sources = self._block_items(entry)
                # the pageable state attached to the block (mmgp.state_ring) is staged with it, or alone for a block whose parameters are pinned
                state = [(ring, block, ring.records[block][0], ring.generation) for ring, block in self.state_records.get(entry, ()) if ring.stages(block)]
                state_bytes = builtins.sum(-(-source.numel() // BLOCK_ALIGN) * BLOCK_ALIGN for _, _, source, _ in state)
                if items and sources is None and entry in wanted and not _pool.pending:
                    self.ring.schedule(entry, items, layout, _pool.prepare(self, model_id, entry, self._pool_plan(entry)))
                elif items and sources is None and layout[2] <= self.ring.size:
                    state = state if layout[2] + state_bytes <= self.ring.size else ()
                    budget -= layout[2] + (state_bytes if state else 0)
                    if budget < 0:
                        break
                    self.ring.schedule(entry, items, layout, state=state)
                elif state and state_bytes <= self.ring.size:
                    budget -= state_bytes
                    if budget < 0:
                        break
                    self.ring.schedule(entry, [], None, state=state)
            entry = self.next_blocks_names.get(entry, None)

    def _pool_ranked(self, model_id):
        # The blocks of the towers of a model out of VRAM, in the order the pinned pool takes them: spread evenly, across the towers too. It is
        # the ranking of the dynamic preload reversed (that preload keeps a prefix of its ranking, which changes with the workloads, e.g. the
        # prompts and the decoding of an LLM): the first blocks of the pool stay the same whatever the blocks kept in VRAM.
        from .auto_preload import ranked_blocks
        preloaded = self.preloaded_blocks_per_model[model_id]
        cached = self.pool_ranked.get(model_id)
        if cached is None or cached[0] != len(preloaded): # the blocks kept in VRAM: those of a fixed budget, or a prefix of the dynamic preload's ranking
            plan = None if self.auto_preload is None else self.auto_preload.plans.get(model_id)
            if plan is None:
                towers = []
                for tower_name in self.towers_per_model.get(model_id) or ():
                    prefix = model_id + "/" + tower_name
                    floors = sorted((int(entry[len(prefix):]), entry[len(model_id) + 1:]) for entry in self.blocks_of_modules_sizes if entry.startswith(prefix) and entry[len(prefix):].isdigit())
                    towers.append([(name, self.blocks_of_modules_sizes[model_id + "/" + name]) for _, name in floors])
                ranking = ranked_blocks(towers)
            else:
                ranking = plan.ranking[len(plan.others):]  # the blocks outside the towers come first in _pool_others
            cached = self.pool_ranked[model_id] = (len(preloaded), [model_id + "/" + name for name in reversed(ranking) if name not in preloaded])
        return cached[1]

    def _pool_others(self, model_id):
        # the blocks of a model out of VRAM outside its towers (e.g. a final layer), transferred at each pass like those of the towers: the
        # pinned pool takes them first, outside the limit of cpuCompute (which computes blocks of the towers only)
        preloaded = self.preloaded_blocks_per_model[model_id]
        cached = self.pool_others.get(model_id)
        if cached is None or cached[0] != len(preloaded):
            towers = builtins.tuple(model_id + "/" + name for name in self.towers_per_model.get(model_id) or ())
            cached = self.pool_others[model_id] = (len(preloaded), [entry for entry in self.blocks_of_modules if entry.startswith(model_id + "/") and not entry.startswith(towers)
                                                                     and entry.split("/", 1)[1] not in preloaded])
        return cached[1]

    def _pool_limit(self, model_id):
        # how many of its blocks out of VRAM the pinned pool may take for a model: all (None), or those cpuCompute's short calls stream
        return self.pool_limits.get(model_id, 0) if model_id in self.cpu_compute else None

    def _pool_wanted(self, model_id):
        """The blocks of a running model that the pinned pool may take: those it has, and those of its towers out of VRAM, spread evenly, as
        many as the reserved RAM left holds (at most _pool_limit of them). None while the dynamic preload changes the blocks kept in VRAM."""
        if model_id not in self.pool_models:
            return ()
        auto = None if self.auto_preload is None else self.auto_preload.plans.get(model_id)
        if auto is not None and (auto.measuring or auto.pending):
            return ()
        limit, admits = self._pool_limit(model_id), self._pool_admits(model_id)
        key = (len(self.preloaded_blocks_per_model[model_id]), limit, _pool.epoch, _pool.full, admits)
        cached = self.pool_wanted.get(model_id)
        if cached is None or cached[0] != key:
            ranked, others = self._pool_ranked(model_id), self._pool_others(model_id)
            wanted = {entry for entry in others + ranked if _pool.involved(self, entry)}
            room, short = self.max_reservable_memory - pinned.total_bytes - _pool.reserved(), 0
            for entry in others + ranked[:limit]:
                if not admits or _pool.full or entry in wanted or not self.needs_staging(entry): # a block pinned for good needs no entry
                    continue
                size = self._pool_plan(entry).nbytes
                if size <= room:
                    wanted.add(entry)
                    room -= size
                else:
                    short += size
            if short: # left out for room: reclaimed at the next block transfer if idle models of other objects hold some (_pool.reclaim_for)
                with _pool.guard:
                    _pool.shortfalls[self] = _pool.shortfalls.get(self, 0) + short
            cached = self.pool_wanted[model_id] = (key, wanted)
        return cached[1]

    def _pool_admits(self, model_id):
        # whether new blocks of a model may enter the pinned pool: its _mm_pinned_pool, read at each load and block so that it may change at
        # run time (profile() sets the defaults of its models)
        return getattr(self.models[model_id], "_mm_pinned_pool", True)

    def _pool_need(self, model_id):
        # bytes the pinned pool may lock for the blocks of a model out of VRAM: those it does not hold yet, and all of them (only those it
        # holds when it may not take more)
        need = whole = 0
        if model_id in self.pool_models:
            admits = self._pool_admits(model_id)
            for entry in self._pool_others(model_id) + self._pool_ranked(model_id)[:self._pool_limit(model_id)]:
                held = _pool.entry(self, entry)
                if held is not None:
                    whole += held.nbytes
                elif admits and self.needs_staging(entry): # not pinned for good
                    size = self._pool_plan(entry).nbytes
                    whole, need = whole + size, need + (0 if _pool.involved(self, entry) else size)
        return need, whole

    def _pool_plan(self, entry_name):
        # how a block enters the pinned pool, for its current layout
        items, layout, _ = self._block_items(entry_name)
        plan = self.pool_plans.get(entry_name)
        if plan is None or plan.layout is not layout:
            plan = self.pool_plans[entry_name] = _PinnedPool.plan(items, layout)
        return plan

    def _drops_pages(self, model_id):
        # whether the checkpoint pages read for a model's transfers leave the working set at once, available to the other models rather than
        # counted as used until Windows trims them: a model that runs once or twice per generation (a text encoder, a VAE; kept out of the
        # pinned pool; pinned memory has none), when keeping all its pages could bring the available RAM under POOL_RAM_FLOOR. Dropping costs
        # about 75 ms per GiB, as much as Windows' own trimming under pressure: with RAM to spare the pages stay, for nothing
        if not (DROP_ONCE_PER_GENERATION_PAGES and not getattr(self.models[model_id], "_mm_pinned_pool", model_id not in ONCE_PER_GENERATION_MODELS)):
            return False
        mapped = self._mapped_bytes(model_id)
        return mapped > 0 and psutil.virtual_memory().available < POOL_RAM_FLOOR + mapped

    def _read_later(self):
        # the CPU tensors it may still read: its models' weights, the CPU copies its transfers read (those of a model in VRAM included) and
        # the views of the file that its freed pinned buffers are filled again from
        for model in self.models.values():
            yield from model.parameters()
            yield from model.buffers()
            yield from (source for record in getattr(model, "_mm_pinned_buffers", ()) for item in record.items for source in item[3])
        yield from (p for params in self.blocks_of_modules.values() for _, _, p, _, _ in params)

    def _mapped_bytes(self, model_id):
        # the bytes of the checkpoint pages of all the blocks of a model
        mapped = self.mapped_bytes.get(model_id)
        if mapped is None:
            mapped = self.mapped_bytes[model_id] = builtins.sum(end - start for entry in self.blocks_of_modules if entry.split("/", 1)[0] == model_id for start, end in self._mapped_pages(entry))
        return mapped

    def _mapped_pages(self, entry_name):
        # the pages of a block's tensors mapped from a checkpoint file, for its current layout
        items, layout, _ = self._block_items(entry_name)
        found = self.mapped_pages.get(entry_name)
        if found is None or found[0] is not layout:
            found = self.mapped_pages[entry_name] = (layout, _mapped_pages([(leaf.data_ptr(), leaf.numel() * leaf.element_size()) for _, t in items for leaf in _staging_leaves(t)]))
        return found[1]

    def _pool_leaves(self, entry_name):
        # the plain tensors of a block as they are now, as (address, bytes)
        return {(leaf.data_ptr(), leaf.numel() * leaf.element_size()) for _, _, p, _, tied in self.blocks_of_modules[entry_name] if tied is None
                for leaf in _staging_leaves(p.data if type(p) is torch.nn.Parameter else p) or ()}

    def _pin_models(self, listed, smart_pinning, partial_pinning, pinned_peft_lora, perc_reserved_mem_max, max_reservable_memory, verboseLevel, loading_callback, cpu_pinning=None):
        # Reserved RAM is shared by the models: the staging ring first if swapped blocks will not be pinned, then the models to pin in the order
        # of the pipe: first the blocks they swap at each step (spread evenly if they do not all fit), then the blocks they preload in VRAM,
        # then their parameters out of blocks, which cross once per load; last, with smart_pinning n > 0, 1/n of the swapped blocks of the
        # other models, a minimum for models that are not to be pinned. A model loaded whole is pinned up to the reserved RAM left.
        todo = [model_id for model_id in self.models if not hasattr(self.models[model_id], "_already_pinned")]
        tensors = {entry: [p for _, _, p, _, tied in params if tied is None] for entry, params in self.blocks_of_modules.items() if entry.split("/", 1)[0] in todo}
        sizes = {entry: builtins.sum(builtins.sum(_leaf_sizes(p)) for p in ts) for entry, ts in tensors.items()}
        _pool.make_room(self, None, builtins.sum(size for entry, size in sizes.items() if listed[entry.split("/", 1)[0]])) # pinning for good comes first
        def entries(model_id, kind):
            blocks = [entry for entry in tensors if entry.startswith(model_id + "/")]
            if kind == "base":
                return [model_id] if model_id in tensors and not (partial_pinning and blocks) else []
            if kind == "towers": # cpuCompute: the blocks of its towers, kept in VRAM or not (the automatic preload keeps the others first)
                return [entry for entry, _ in self.cpu_compute_entries.get(model_id, ()) if entry in tensors]
            return [entry for entry in blocks if (entry.split("/", 1)[1] in self.preloaded_blocks_per_model[model_id]) == (kind == "preloaded")]
        steps = [(model_id, kind, 1) for kind in ("swapped", "preloaded", "base") for model_id in todo if listed[model_id]]
        if smart_pinning:
            steps += [(model_id, "swapped", 1 / smart_pinning) for model_id in todo if not listed[model_id]]
        steps += [(model_id, "towers", share) for model_id, share in (cpu_pinning or {}).items() if model_id in todo and not listed[model_id] and share > 0]
        def plan(budget):
            ranks, taken = {model_id: {} for model_id in todo}, set()
            for rank, (model_id, kind, share) in enumerate(steps):
                kind_entries = entries(model_id, kind)
                need = builtins.sum(sizes[entry] for entry in kind_entries) * share
                whole = kind == "base" and not any(entry.startswith(model_id + "/") for entry in tensors) # pinned as far as it fits, in order
                ratio = share if need == 0 or whole or need <= budget else max(0, share * budget / need)
                chosen = []
                for entry in _spread(kind_entries, ratio):
                    if not whole and sizes[entry] > budget:
                        continue
                    budget = max(0, budget - sizes[entry])
                    taken.add(entry)
                    chosen.append(entry)
                for position, entry in enumerate(_evenly(chosen)): # a rank per block: (step, position)
                    ranks[model_id].update((id(p), (rank, position)) for p in tensors[entry])
            return ranks, taken
        ranks, taken = plan(max_reservable_memory - pinned.total_bytes)
        # the blocks of the models of the pinned pool cross a staging ring only until they enter it: made when they need one (ring_on_demand)
        left = [entry for model_id in todo if (listed[model_id] or smart_pinning is not None) and model_id not in self.pool_models for entry in entries(model_id, "swapped") if entry not in taken]
        if left and self.async_transfers: # sized for the blocks of the towers, the ring takes RAM first so that pinning cannot leave it none
            towers = tuple(model_id + "/" + name for model_id in todo for name in self.towers_per_model[model_id])
            bounds = [builtins.sum(size + RING_RUN_GAP + 1 for p in tensors[entry] for size in _leaf_sizes(p)) for entry in ([entry for entry in left if entry.startswith(towers)] or left)]
            self._create_ring(max(bounds), max_reservable_memory, verboseLevel)
            ranks, taken = plan(max_reservable_memory - pinned.total_bytes)
        for model_id in self.models:
            if model_id not in todo and listed[model_id] and verboseLevel >= 1:
                print(f"Model '{model_id}' already pinned to reserved memory")
        first_rank = {model_id: min(model_ranks.values()) for model_id, model_ranks in ranks.items() if model_ranks}
        for model_id in sorted(first_rank, key=first_rank.get):
            swapped = entries(model_id, "swapped")
            pinned_swapped = [entry for entry in swapped if entry in taken]
            if listed[model_id] and len(pinned_swapped) < len(swapped) and verboseLevel >= 1:
                print(f"Reserved RAM is short for '{model_id}': {len(pinned_swapped)} of its {len(swapped)} blocks processed at each step are pinned, spread evenly, the staging ring takes the others. You may increase 'perc_reserved_mem_max' above {perc_reserved_mem_max:0.2f} to pin more.")
            partial = any(entry not in taken for entry in swapped + entries(model_id, "preloaded") + entries(model_id, "base"))
            _pin_to_memory(self.models[model_id], model_id, partialPinning = partial, pinnedPEFTLora = pinned_peft_lora, perc_reserved_mem_max = perc_reserved_mem_max, verboseLevel = verboseLevel, loading_callback = loading_callback, ranks = ranks[model_id])

    def _create_ring(self, largest, max_reservable_memory, verboseLevel):
        # the staging ring of the swapped blocks that are not pinned: RING_DEFAULT_BLOCKS of the largest of them, two in a shortage of reserved RAM
        _pool.make_room(self, None, ring=RING_DEFAULT_BLOCKS * largest) # the pinned pool's entries and buffers of models that do not run make room for it
        for blocks in (RING_DEFAULT_BLOCKS, 2):
            size = blocks * largest
            if max_reservable_memory > 0 and pinned.total_bytes + size > max_reservable_memory:
                continue
            try:
                self.ring = _StagingRing(size)
            except RuntimeError:
                continue
            self.ring_check = None
            if verboseLevel >= 1:
                tqdm.write(f"Staging ring of {size/ONE_MB:0.1f} MB for the blocks that are not pinned (largest {largest/ONE_MB:0.1f} MB)")
            return
        tqdm.write(f"Unable to lock {2 * largest/ONE_MB:0.1f} MB of RAM for a staging ring: the blocks that are not pinned will use the slower staged transfers")

    def _ring_for_streaming(self, model_id):
        # The blocks of a model were all kept in VRAM when its profile was made, and some of them stream now (released by a dynamic
        # preload), or a block of a model of the pinned pool is not in it: the staging ring that smartPinning gives the blocks left unpinned,
        # made at their first transfer. Refused (the reserved RAM held by models that run), it is tried again after the next quiet point,
        # where the idle models give their reserved RAM (_create_ring)
        if self.ring_refused == _pool.epoch:
            return
        reservable, verbose = self.ring_on_demand
        largest = self._ring_bound(model_id)
        if largest:
            self._create_ring(largest, reservable, verbose)
            if self.ring is None:
                self.ring_refused = _pool.epoch
                return
            self.prefetch_windows.pop(model_id, None)
        self.ring_on_demand = None

    def _ring_bound(self, model_id):
        # the room the largest block of a model's towers takes in the staging ring, 0 without towers
        bound = self.ring_bounds.get(model_id)
        if bound is None:
            towers = builtins.tuple(model_id + "/" + name for name in self.towers_per_model.get(model_id) or ())
            bound = self.ring_bounds[model_id] = builtins.max((builtins.sum(size + RING_RUN_GAP + 1 for _, _, p, _, tied in self.blocks_of_modules[entry] if tied is None for size in _leaf_sizes(p))
                                                                 for entry in self.blocks_of_modules if towers and entry.startswith(towers)), default=0)
        return bound

    def _ring_unused(self):
        # every block out of VRAM of the running models is pinned for good or in the pinned pool, none being prepared: the staging ring has
        # nothing left to stage (checked again when an entry comes or leaves, or the blocks kept in VRAM change)
        if self.ring.jobs or _pool.pending:
            return False
        key = (_pool.version, builtins.tuple((model_id, len(self.preloaded_blocks_per_model[model_id])) for model_id in self.active_models_ids))
        if self.ring_check is None or self.ring_check[0] != key:
            active = builtins.tuple(model_id + "/" for model_id in self.active_models_ids)
            self.ring_check = (key, not builtins.any(self.needs_staging(entry) and self._block_items(entry)[2] is None for entry in self.blocks_of_modules
                                                     if entry.startswith(active) and entry.split("/", 1)[1] not in self.preloaded_blocks_per_model[entry.split("/", 1)[0]]))
        return self.ring_check[1]

    def _release_ring(self, on_demand=False):
        # on_demand: a staging ring is made again when a block needs one (the blocks of the pinned pool, out of it)
        if self.ring is not None:
            self.ring.close()
            self.ring = None
            if on_demand:
                self.ring_on_demand = (self.max_reservable_memory, self.verboseLevel)

    def is_main_waiting(self):
        return self.main_waiting

    def can_model_be_cotenant(self, model_id):
        potential_cotenants = self.cotenants_map.get(model_id) or ()
        if potential_cotenants == "*":
            return True
        for existing in self.active_models_ids:
            if existing not in potential_cotenants and self.cotenants_map.get(existing) != "*":
                return False
        return True

    def _move_loras(self, loras_active_adapters, loras_modules, to_GPU, model=None, skip_idle=False):
        # skip_idle: a swapped block leaves out the adapters scaled to 0 for the current step; _ensure_loras completes it before it runs
        if not to_GPU:
            for lora_module in loras_modules.values():
                for key in [key for key in lora_module if key.endswith("_GPU")]:
                    del lora_module[key]
            return
        idle = {adapter for adapter in loras_active_adapters if self._get_lora_scaling(model._loras_scaling, model, adapter) == 0} if skip_idle else ()
        shortcuts = getattr(model, "_loras_model_shortcuts", None) if model is not None else None
        tensor_cache = {}
        alias_cache = {}

        def _move_tensor(item):
            if item is None:
                return None
            if torch.is_tensor(item):
                ref = _get_tensor_ref(item)
                moved = tensor_cache.get(ref, None)
                if moved is None:
                    moved = item.cuda(non_blocking=True)
                    tensor_cache[ref] = moved
                return moved
            return item

        def _resolve_alias(value, adapter):
            if not isinstance(value, str):
                return _move_tensor(value)
            cache_key = (adapter, value)
            cached = alias_cache.get(cache_key, None)
            if cached is not None:
                return cached
            while isinstance(value, str):
                target, slot = value.rsplit("#", 1)
                value = shortcuts[target][adapter][int(slot)]
            resolved = _move_tensor(value)
            alias_cache[cache_key] = resolved
            return resolved

        for _, lora_module in loras_modules.items():
            for adapter in loras_active_adapters:
                lora_data = lora_module.get(adapter, None)
                key = adapter + '_GPU'
                if lora_data is None or key in lora_module or adapter in idle and _plain_lora(lora_data):
                    continue
                moved_data = list(lora_data)
                for i in range(min(4, len(moved_data))):
                    moved_data[i] = _resolve_alias(moved_data[i], adapter)
                lora_module[key] = moved_data

    def _graph_weight_tensors(self, model, module):
        """The CUDA tensors a block reads besides its arguments and LoRA copies: its parameters and buffers (their quantized parts and the
        tensors cached on them, e.g. converted scales)."""
        for tensor in builtins.list(module.parameters()) + builtins.list(module.buffers()):
            subtensors = _get_quantized_subtensors(tensor)
            yield from (t for _, t in subtensors) if subtensors else (tensor,)
            yield from (_unwrap_quantized_tensor(value) for value in tensor.__dict__.values() if torch.is_tensor(value))

    def _graph_state(self, model, module):
        # per block, for its active adapters: the holders of its tensors (parameter and buffer dictionaries), the LoRA dictionaries with the
        # key of their GPU copies, then the weak references and the key of _graph_weights_key
        adapters = builtins.tuple(getattr(model, "_loras_active_adapters", None) or ())
        state = module.__dict__.get("_mm_graph_state", None)
        if state is None or state[0] != adapters:
            submodules = builtins.list(module.modules())
            holders = [(sub._parameters, name) for sub in submodules for name in sub._parameters] + [(sub._buffers, name) for sub in submodules for name in sub._buffers]
            loras = [(sub._mm_lora_data, adapter + "_GPU") for sub in submodules if getattr(sub, "_mm_lora_data", None) for adapter in adapters]
            state = module._mm_graph_state = [adapters, holders, loras, None, None]
        return state

    def _graph_forget(self, model):
        # the graphs of the model are gone: so are the holders of the tensors of its blocks, kept for their keys. A block may have dropped
        # submodules since they were made (e.g. the KV caches of a generation), whose memory they would otherwise keep
        for module in model.modules():
            module.__dict__.pop("_mm_graph_state", None)
            module.__dict__.pop("_mm_graph_placement", None)

    def _graph_weights_key(self, model, module):
        """What a block's graph depends on besides its arguments and LoRA copies: its parameters and buffers, the active adapters and their
        scalings. A block loaded elsewhere gets new tensor objects (and the caches made on them, e.g. converted scales): the key is their serial
        numbers, never reused, recomputed only when a weak reference shows another object, which keeps the check of each call short."""
        state = self._graph_state(model, module)
        adapters, holders, _, refs, key = state
        # a module whose parameters only MMGP moves (_mm_graph_placement_key, e.g. a whole LLM replayed as one graph) skips the walk over
        # its tensors while MMGP installed none since the last call
        placement = module.__dict__.get("_mm_graph_placement") if module.__dict__.get("_mm_graph_placement_key", False) else None
        if placement is None or placement[0] != _placement_generation or placement[1] is not state:
            current = [holder[name] for holder, name in holders]
            if refs is None or len(refs) != len(current) or any((ref() if ref is not None else None) is not t for ref, t in zip(refs, current)):
                state[3] = [None if t is None else weakref.ref(t) for t in current]
                key = state[4] = builtins.tuple(None if t is None else _graph_serial(t) for t in current)
            if module.__dict__.get("_mm_graph_placement_key", False):
                module._mm_graph_placement = (_placement_generation, state)
        return key, adapters, builtins.tuple(self._get_lora_scaling(model._loras_scaling, model, adapter) for adapter in adapters)

    def _graph_ready(self, model_id):
        # the weights are where they stay: the automatic preload does not measure a new workload (it may keep other blocks after it). Asked
        # at each graph call, which replays without the hooks: the model is in use (and no longer parked)
        self.used[model_id] = memory_pressure.stamp()
        if self.parked:
            self.unpark()
        return (self.auto_preload is None or self.auto_preload.settled(model_id)) and self.cpu_blocks.get(model_id) is None

    def _graph_lora_inputs(self, model, module):
        """The LoRA copies that a block reads, as (list, index): its graphs read them from buffers filled before each replay, so that they come
        and go with their block as without CUDA graphs."""
        return [(data, index) for holder, name in self._graph_state(model, module)[2] for data in (holder.get(name, None),) if data is not None
                for index in range(min(4, len(data))) if torch.is_tensor(data[index])]

    def _ensure_loras(self, model_id, entry_name):
        # A prefetched block got the LoRA copies of the step and adapters current when it was transferred: the last block of a forward
        # prefetches the first block of the next forward, which may use other multipliers. Copy whatever the current step needs.
        model = self.models[model_id]
        loras_active_adapters = getattr(model, "_loras_active_adapters", None)
        loras_model_data = getattr(model, "_loras_model_data", None)
        if not loras_active_adapters or loras_model_data is None:
            return
        loras_modules = {parent_module: loras_model_data[parent_module] for parent_module, _, _, _, _ in self.blocks_of_modules[entry_name] if parent_module in loras_model_data}
        if loras_modules:
            self._move_loras(loras_active_adapters, loras_modules, True, model, skip_idle=True)

    def refresh_loras(self, model):
        """LoRAs loaded into a model (or into a module of a managed model) whose blocks stay in VRAM, e.g. a model kept loaded between
        generations: the GPU copies a load of these blocks would have made, for every active adapter (one idle at this step may be scaled up
        at a later one, and a block kept in VRAM is not loaded again). Blocks loaded later get theirs as usual."""
        loras_active_adapters = getattr(model, "_loras_active_adapters", None)
        loras_model_data = getattr(model, "_loras_model_data", None)
        if not loras_active_adapters or loras_model_data is None:
            return
        for entry_name, modules in self.blocks_of_modules.items():
            if entry_name.split("/", 1)[0] not in self.active_models_ids or not modules or not builtins.all(getattr(parent_module, n).is_cuda for parent_module, n, *_ in modules[-1:]):
                continue
            loras_modules = {parent_module: loras_model_data[parent_module] for parent_module, *_ in modules if parent_module in loras_model_data}
            if loras_modules:
                self._move_loras(loras_active_adapters, loras_modules, True, model)
            
    @torch.compiler.disable()
    def gpu_load_blocks(self, model_id, blocks_name, preload = False):
        # cl = clock.start()


        entry_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        
        def copy_block(stream_to_use, blocks_params, parallel_staging, moved): # moved: parameters already on the GPU, by index
            model = self.models[model_id]
            loras_modules = {}
            loras_active_adapters =  getattr(model ,"_loras_active_adapters", None)
            if loras_active_adapters == None or len(loras_active_adapters) == 0:
                loras_model_data = None
            else:
                loras_model_data =  getattr(model, "_loras_model_data", None)

            with torch.cuda.stream(stream_to_use):
                for i, param in enumerate(blocks_params):
                    parent_module, n, p, is_buffer, tied_param = param

                    if tied_param != None:
                        tied_p = getattr( tied_param[0], tied_param[1]) 
                        if tied_p.is_cuda:
                            _install_tensor(parent_module, n , tied_p)
                            continue
                    # if hasattr(p,'_data'):
                    #     if not p._data.is_pinned() or not p._scale.is_pinned():
                    #         pass
                    # else:
                    #     if  not p.data.is_pinned():
                    #         pass

                    if moved is not None and i in moved:
                        q = self.stager.checked(p, moved[i])
                    else:
                        q = self.stager.to_gpu(p, parallel_staging, model_id not in self.pool_models) # the pool's blocks enter it locked in place
                    if not is_buffer:
                        q._is_param = True # parameter-ness without a wrapper, as nn.Parameter does for tensor subclasses: the cpu parameter is reinstalled on unload
                    _install_tensor(parent_module, n , q)

                    if tied_param != None:
                        _install_tensor(tied_param[0], tied_param[1], q)
                    del p, q
                    if loras_model_data != None:
                        lora_data =  loras_model_data.get(parent_module, None)
                        if lora_data != None:
                            loras_modules[parent_module]= lora_data
                if len(loras_modules) > 0:
                    self._move_loras(loras_active_adapters, loras_modules, True, model, skip_idle=not preload)

        def cpu_to_gpu(stream_to_use, blocks_params, parallel_staging=lambda: True, moved=None):
            return copy_block(stream_to_use, blocks_params, parallel_staging, moved)

        if self.ring is not None and self.pool_models and not preload and blocks_name is not None and self._ring_unused():
            self._release_ring(on_demand=True) # once, when the pinned pool holds every block streamed: unlocking the ring waits for the GPU
        if self.pool_models and not preload and blocks_name is not None and self in _pool.shortfalls and not torch.cuda.is_current_stream_capturing():
            _pool.reclaim_for(self) # blocks of its running models were refused for room: the idle models of other objects give it, once
        if self.ring is None and self.ring_on_demand is not None and not preload and blocks_name is not None and self.needs_staging(entry_name) and self._block_items(entry_name)[2] is None:
            self._ring_for_streaming(model_id)
        pipelined = self.async_transfers and blocks_name != None and self.model_prefetch_window(model_id) > 0
        self.wait_prefetch() # the staging thread must not allocate memory freed below while it may still be read
        loaded_block = self.loaded_blocks[model_id]

        if not preload and loaded_block != None and self.auto_preload is not None and self.auto_preload.keep(model_id, loaded_block):
            loaded_block = self.loaded_blocks[model_id] = None # transferred into a buffer of its own, it stays in VRAM: preloaded from now on
        if not preload and loaded_block != None:
            if self.async_transfers:
                # The block may still be read by queued compute. Its memory, mostly allocated on the transfer stream, is then reused or
                # released in that stream's order: the stream waits for the compute queued so far, the GPU and not the host.
                self.transfer_stream.wait_stream(torch.cuda.current_stream())
            self.gpu_unload_blocks(model_id, loaded_block)
            if self.ready_to_check_mem():
                self.empty_cache_if_needed()


        if self.verboseLevel >=2:
            model = self.models[model_id]
            model_name = model._get_name()
            # if not preload:
            #     print(f"Request to load model {entry_name} ({model_name}) in GPU")
                

        if pipelined:
            prev_entry = None if preload or loaded_block is None else model_id + "/" + loaded_block
            if prev_entry is not None and (self.next_blocks_names.get(prev_entry, None) is None or prev_entry in self.learned_links):
                self.next_blocks_names[prev_entry] = entry_name # the end of a tower is followed by the block that came next when it last ran
                self.learned_links.add(prev_entry)
            if self.verboseLevel >= 2 and entry_name not in self.inflight:
                tqdm.write(f"{'Preloading' if preload else 'Loading'} model {entry_name} ({model_name}) in GPU")
            self._load_pipelined(model_id, entry_name, cpu_to_gpu)
        elif self.async_transfers and blocks_name != None:
            prev = self.prev_blocks_names[entry_name]
            # the block prefetched at its predecessor's load, unless that transfer was sent back since (_retire_inflight)
            first = prev == None or loaded_block == None or prev != model_id + "/" + loaded_block or not self.block_in_vram(model_id, blocks_name)
            next_blocks_entry = self.next_blocks_names[entry_name] if entry_name in self.next_blocks_names else None
            if first:
                if self.verboseLevel >=2:
                    if preload:
                        tqdm.write(f"Preloading model {entry_name} ({model_name}) in GPU")
                    else:
                        tqdm.write(f"Loading model {entry_name} ({model_name}) in GPU")
                cpu_to_gpu(torch.cuda.current_stream(), self.blocks_of_modules[entry_name])

            torch.cuda.synchronize()
            if not first:
                self._ensure_loras(model_id, entry_name)

            if next_blocks_entry != None:
                if self.verboseLevel >=2:
                    tqdm.write(f"Prefetching model {next_blocks_entry} ({model_name}) in GPU")
                if self.needs_staging(next_blocks_entry):
                    self.submit_prefetch(next_blocks_entry, cpu_to_gpu, self.transfer_stream, self.blocks_of_modules[next_blocks_entry], self.is_main_waiting)
                else:
                    cpu_to_gpu(self.transfer_stream, self.blocks_of_modules[next_blocks_entry]) #, self.default_stream

        else:
            if self.verboseLevel >=2:
                tqdm.write(f"Loading model {entry_name} ({model_name}) in GPU")
            cpu_to_gpu(self.default_stream, self.blocks_of_modules[entry_name])
            torch.cuda.synchronize()
        if not preload:
            self.loaded_blocks[model_id] = blocks_name
        if self._drops_pages(model_id):
            _drop_mapped_pages(self._mapped_pages(entry_name))

        # cl.stop()
        # print(f"load time: {cl.format_time_gap()}")

    def block_in_vram(self, model_id, blocks_name):
        # whether a block's tensors are all in VRAM: they are installed one by one, its last one last (a block being transferred is not; a
        # block whose small modules all joined the base has none)
        return builtins.all(getattr(parent_module, n).is_cuda for parent_module, n, *_ in self.blocks_of_modules[model_id + "/" + blocks_name][-1:])

    @torch.compiler.disable()
    def gpu_unload_blocks(self, model_id, blocks_name):
        # cl = clock.start()
        entry_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        self.wait_prefetch(entry_name) # a prefetch of this block may still be setting its parameters
        self.inflight.pop(entry_name, None) # an unloaded block must be transferred again
        self._free_slot(entry_name)
        if blocks_name != None and blocks_name == self.loaded_blocks[model_id]:
            self.loaded_blocks[model_id] = None 


        blocks_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        if self.verboseLevel >=2:
            model = self.models[model_id]
            model_name = model._get_name()
            tqdm.write(f"Unloading model {blocks_name} ({model_name}) from GPU")
 
        blocks_params = self.blocks_of_modules[blocks_name]
        model = self.models[model_id]
        loras_modules = {}
        loras_active_adapters =  getattr(model ,"_loras_active_adapters", None)
        loras_model_data =  getattr(model, "_loras_model_data", None) # every LoRA copy goes, including those of adapters deactivated meanwhile

        for param in blocks_params:
            parent_module, n, p, is_buffer, _  = param
            _install_tensor(parent_module, n , p) # the original cpu tensor
            del p

            if loras_model_data != None:
                lora_data =  loras_model_data.get(parent_module, None)
                if lora_data != None:
                    loras_modules[parent_module]= lora_data

        if len(loras_modules) > 0:
            self._move_loras(loras_active_adapters, loras_modules, False, model)

        # cl.stop()
        # print(f"unload time: {cl.format_time_gap()}")

    # @torch.compiler.disable()
    def gpu_load(self, model_id):
        from .auto_preload import end_open_cycles
        end_open_cycles(self, model_id)  # the VRAM this model takes is not the activations of the cycles other models left open
        self.pages_dropped.discard(model_id)
        need, whole = self._pool_need(model_id)
        if whole and self.ring is not None and not self.active_models_ids and not self.ring.jobs and self.ring.size < 2 * self._ring_bound(model_id):
            self._release_ring(on_demand=True) # made for the smaller blocks of another model: made again for this one's if it needs one
        ring = RING_DEFAULT_BLOCKS * self._ring_bound(model_id) if need and self.ring is None and self.ring_on_demand is not None else 0
        reclaimed = self._reclaimed_bytes(model_id) # its buffers pinned for good that the pinned pool took while it did not run
        ram = 0 if model_id in self.pool_models else self._mapped_bytes(model_id) # the checkpoint pages its transfers read (a pool model's: in need)
        _pool.make_room(self, model_id, need + reclaimed, whole, ring, ram) # a quiet point: the models unloaded before have finished
        if reclaimed:
            self._refill_pinned(model_id)
        model = self.models[model_id]
        self.active_models.append(model)
        self.active_models_ids.append(model_id)
        runtime._load_resources(self, model_id)
        if self.auto_preload is not None:
            self.auto_preload.on_load(model_id)
        for block_name in [None, *self.preloaded_blocks_per_model[model_id]]:
            # Allocation pressure can shrink the automatic preload even while this loop is copying it.
            if block_name is not None and block_name not in self.preloaded_blocks_per_model[model_id]:
                continue
            self.gpu_load_blocks(model_id, block_name, True)
            if block_name is not None and block_name not in self.preloaded_blocks_per_model[model_id]:
                self.gpu_unload_blocks(model_id, block_name)  # left out of the preload by an allocation of its own transfer

    def unload_all(self, *, keep=()):
        with self.park_lock:
            self._unpark()
            self._unload(keep)

    def park(self, release="models", state_bytes=0):
        """Its models stay loaded while other models or code use the GPU (another offload object, a model outside MMGP): their VRAM is
        reclaimed, from the allocating thread, only as allocations need it (mmgp.memory_pressure), before the VRAM of the models running:
        expert copies and state kept in VRAM, the preloaded blocks (the least recently used models first), the disposable assets its models'
        code declared (runtime.disposable, e.g. a language model's KV cache), then, if that is not enough, release="models": whole models
        (they come back at their next forward); "blocks": nothing more (their bases stay in VRAM); a callable: release() is called once, to
        free the owner's state (state_bytes of VRAM) and unload the models. Call it on the thread running the models, between their calls; unpark() before they run again (their hooks unpark too).
        Needs the MMGP VRAM allocator, which asks for VRAM it misses: returns False without it, the models keep their VRAM."""
        from . import allocator
        if allocator.active != "vmm" or not self.active_models_ids:
            return False
        with self.park_lock:
            self.wait_prefetch()
            torch.cuda.synchronize() # nothing of these models runs any more: other threads may release them
            if self.auto_preload is not None:
                self.auto_preload.on_park()
            self.parked, self.park_release, self.park_state_bytes, self.parked_size = True, release, state_bytes, None
            memory_pressure.park(self)
        return True

    def unpark(self):
        """The models run again (park), on the calling thread: what was released comes back as they run."""
        with self.park_lock:
            if self.parked and self.auto_preload is not None:
                self.auto_preload.thread = threading.get_ident()
            self._unpark()

    def _unpark(self):
        if self.parked:
            self.parked, self.park_release, self.park_state_bytes, self.parked_size = False, None, 0, None
            self.pages_dropped.clear() # its models run again without a gpu_load: their pages may come back
            memory_pressure.unpark(self)

    def last_used(self):
        return builtins.max((self.used.get(model_id, 0) for model_id in self.active_models_ids), default=0)

    def parked_bytes(self):
        """VRAM the parked models would give back (park), for the plans of the other models: their preloaded blocks, with release "models" or
        a callable also their bases, expert copies and state kept in VRAM, and the owner's state with a callable (not their disposable
        assets, which go only when an allocation finds the VRAM short)."""
        if self.parked_size is None:
            total = 0
            for model_id in self.active_models_ids:
                total += builtins.sum(size for name, size in self.preloaded_blocks_per_model[model_id].items() if self.block_in_vram(model_id, name))
                if self.park_release != "blocks":
                    residency = self.expert_residency.get(model_id)
                    total += self.blocks_of_modules_sizes.get(model_id, 0) + self.state_bytes(model_id) + (0 if residency is None else residency.nbytes)
            self.parked_size = total + (self.park_state_bytes if callable(self.park_release) else 0)
        return self.parked_size

    def reclaim_parked(self, need):
        """While parked, on the allocating thread (mmgp.memory_pressure): VRAM of these models for an allocation that misses need bytes."""
        from . import allocator
        if need <= 0 or not self.park_lock.acquire(blocking=False):
            return 0
        try:
            if not self.parked:
                return 0
            self.parked_size, freed = None, 0
            for client in memory_pressure.clients_of(self): # expert copies and state kept in VRAM, then the preloaded blocks
                freed += client.on_pressure(need - freed, parked=True)
                if freed >= need:
                    return freed
            if self.park_release == "blocks":
                return freed
            before = allocator.stats()["allocated"]
            if callable(self.park_release):
                release, self.park_release = self.park_release, "blocks" # once: what the owner keeps afterwards is its own
                try:
                    release()
                except Exception as exc: # the owner's code, inside an allocation: the allocation goes on with what was freed
                    tqdm.write(f"VRAM short: the release of parked models failed: {exc}")
            else:
                for model_id in sorted(self.active_models_ids, key=lambda model_id: self.used.get(model_id, 0)):
                    self._unload([name for name in self.active_models_ids if name != model_id], reclaim=True)
                    if freed + before - allocator.stats()["allocated"] >= need:
                        break
            released = before - allocator.stats()["allocated"]
            if released > 0:
                tqdm.write(f"VRAM short, {released / ONE_MB:0.0f} MB of parked models released")
            return freed + builtins.max(0, released)
        finally:
            self.parked_size = None
            self.park_lock.release()

    def _unload(self, keep=(), reclaim=False):
        # reclaim: models of a parked object, unloaded on another thread inside an allocation (reclaim_parked): their work is finished, and
        # the quiet-point work (pinned pool, caches) waits for their next load
        self.wait_prefetch()
        if not reclaim:
            torch.cuda.synchronize() # blocks may still be read or written by queued work
        for ring in list(self.state_rings): # the state kept in VRAM leaves with the weights of its model (streamed from RAM when it comes back)
            if ring.model_id in self.active_models_ids and ring.model_id not in keep:
                ring.drop_resident()
        for model_id, residency in self.expert_residency.items():
            if model_id not in keep:
                residency.unload()
        if self.ring is not None:
            self.ring.reset()
            if self.pool_models and not keep and self._ring_unused():
                self._release_ring(on_demand=True) # the pinned pool holds every block the models streamed: no ring until a block needs one
        for model_id in self.active_models_ids:
            if model_id in keep:
                continue
            self.gpu_unload_blocks(model_id, None)
            for block_name in self.preloaded_blocks_per_model[model_id]:
                self.gpu_unload_blocks(model_id, block_name)

            loaded_block = self.loaded_blocks[model_id]
            if loaded_block != None:
                self.gpu_unload_blocks(model_id, loaded_block)
                entry_name = model_id + "/" + loaded_block
                next_blocks_entry = self.next_blocks_names[entry_name] if entry_name in self.next_blocks_names else None
                if next_blocks_entry != None:
                    pos = next_blocks_entry.rfind("/")
                    self.gpu_unload_blocks(model_id, next_blocks_entry[pos+1:])
                self.loaded_blocks[model_id] = None
            for entry in [entry for entry in self.inflight if entry.startswith(model_id + "/")]:
                self.gpu_unload_blocks(model_id, entry[len(model_id) + 1:])
            if self._drops_pages(model_id):
                # also the blocks that the staging ring copied ahead for a pass that did not come (the first ones, after the last one); the
                # pages already dropped cost almost nothing
                for entry in [entry for entry in self.blocks_of_modules if entry.split("/", 1)[0] == model_id]:
                    _drop_mapped_pages(self._mapped_pages(entry))

        if self.auto_preload is not None:
            self.auto_preload.on_unload([model_id for model_id in self.active_models_ids if model_id not in keep])
        for model_id in [model_id for model_id in self.block_slots if model_id not in keep]:
            del self.block_slots[model_id]
        for model_id, graphs in self.cuda_graphs.items():
            if model_id not in keep:
                graphs.clear()
        runtime._unload_resources(self, [model_id for model_id in self.active_models_ids if model_id not in keep])
        self.slot_views = {key: views for key, views in self.slot_views.items() if key[0].split("/", 1)[0] in self.block_slots}
        self.active_models_ids = [model_id for model_id in self.active_models_ids if model_id in keep]
        self.active_models = [self.models[model_id] for model_id in self.active_models_ids]
        if reclaim:
            return
        _pool.make_room(self) # the entries of the pinned pool of the models that do not run leave if the RAM is short
        torch.cuda.empty_cache()
        gc.collect()
        self.last_reserved_mem_check = time.time()

    def move_args_to_gpu(self, dtype, *args, **kwargs):
        new_args= []
        new_kwargs={}

        for arg in args:
            if torch.is_tensor(arg):    
                if arg.dtype == torch.float32:
                    arg = arg.to(dtype).cuda(non_blocking=True)
                elif not arg.is_cuda:
                    arg = arg.cuda(non_blocking=True)
            new_args.append(arg)
        for k in kwargs:
            arg = kwargs[k]
            if torch.is_tensor(arg):
                if arg.dtype == torch.float32:
                    arg = arg.to(dtype).cuda(non_blocking=True)             
                elif not arg.is_cuda:
                    arg = arg.cuda(non_blocking=True)             
            new_kwargs[k]= arg
        
        return new_args, new_kwargs

    def ready_to_check_mem(self):
        if self.anyCompiledModule:
             return
        cur_clock = time.time()
        # can't check at each call if we can empty the cuda cache as quering the reserved memory value is a time consuming operation
        if (cur_clock - self.last_reserved_mem_check)<0.200:
            return False
        self.last_reserved_mem_check = cur_clock
        return True        


    def empty_cache_if_needed(self):
        mem_reserved = torch.cuda.memory_reserved()
        mem_threshold = 0.9*self.device_mem_capacity
        if mem_reserved >= mem_threshold:            
            mem_allocated = torch.cuda.memory_allocated()
            if mem_allocated <= 0.70 * mem_reserved: 
                # print(f"Cuda empty cache triggered as Allocated Memory ({mem_allocated/1024000:0f} MB) is lot less than Cached Memory ({mem_reserved/1024000:0f} MB)  ")
                torch.cuda.empty_cache()
                tm= time.time()
                if self.verboseLevel >=2:
                    tqdm.write(f"Empty Cuda cache at {tm}")
                # print(f"New cached memory after purge is {torch.cuda.memory_reserved()/1024000:0f} MB)  ")


    def any_param_or_buffer(self, target_module: torch.nn.Module):
        
        for _ in target_module.parameters(recurse= False):
            return True
        
        for _ in target_module.buffers(recurse= False):
            return True
        
        return False

    def _get_lora_scaling(self, loras_scaling, model, active_adapter):
        scaling_list = loras_scaling[active_adapter]
        if isinstance(scaling_list, list):
            step_no =getattr(model, "_lora_step_no", 0)
            return scaling_list[step_no]
        else:
            return float(scaling_list)



    def _lora_generic_forward(self, model, submodule, loras_data, func, *args, **kwargs) -> torch.Tensor:

        weight = submodule.weight 
        bias =  getattr(submodule, "bias", None) 
        original_weight = None 
        original_bias = None
        active_adapters = model._loras_active_adapters
        loras_scaling = model._loras_scaling
        first_weight =  True
        first_bias =  True
        for active_adapter in active_adapters:
            data = loras_data.get(active_adapter + '_GPU', None)
            if data == None:
                continue
            diff_w, _, diff_b, _, alpha = data[:5]
            scaling = self._get_lora_scaling( loras_scaling, model, active_adapter) * alpha
            if scaling == 0:
                continue
            if first_weight:
                original_weight= weight.clone() if weight is not None else None
                first_weight = False
            if first_bias:
                original_bias= bias.clone() if bias is not None else None
                first_bias = False

            if diff_w is not None:
                weight.add_(diff_w, alpha= scaling)
                diff_w = None
            if diff_b is not None:
                bias.add_(diff_b, alpha= scaling)
                diff_b = None

        ret = func(*args, **kwargs )

        if original_weight is not None: weight.data  = original_weight    
        if original_bias is not None: bias.data = original_bias

        return ret


    def _dora_linear_forward(
        self,
        model,
        submodule,
        adapters_data,                # dict: name+"_GPU" -> [A, B, diff_b, g_abs, alpha, meta?]; g_abs=None means LoRA
        weight= None,
        bias = None,
        original_bias = True,
        dora_mode: str = "blend",     # "ref_exact" | "blend"
    ):
        active_adapters = getattr(model, "_loras_active_adapters", [])
        loras_scaling   = getattr(model, "_loras_scaling", {})
        # Snapshot base weight (safe for quantized modules)
        if weight is None:
            bias = submodule.bias
            original_bias = True
            if isinstance(submodule, QModuleMixin):
                weight = submodule.weight.dequantize()
            else:
                weight = submodule.weight.clone()

        base_dtype = weight.dtype
        eps = 1e-8
        W0 = weight.float()
        g0 = torch.linalg.vector_norm(W0, dim=1, keepdim=True, dtype=torch.float32).clamp_min(eps)  # [out,1]

        # Keep big mats in low precision
        # Wc = W0 if W0.dtype == compute_dtype else W0.to(compute_dtype)
        W0 /= g0
        weight[...]  = W0.to(base_dtype) 
        W0 = None

        dir_update = None          # Σ s * ((B@A)/g0)  in compute_dtype
        g = None                   # final magnitude: set absolute (ref_exact) or blended (blend)
        bias_delta = None          # Σ s * diff_b

        # Accumulate DoRA adapters only (g_abs != None)
        for name in active_adapters:
            data = adapters_data.get(name + "_GPU", None)
            if data is None: continue
            A, B, diff_b, g_abs, alpha = data[:5]
            if g_abs is None: continue  

            s = self._get_lora_scaling(loras_scaling, model, name) * float(alpha)
            if s == 0: continue

            # Direction update in V-space with row-wise 1/g0
            if (A is not None) and (B is not None):
                dV = torch.mm(B, A)      # [out,in], compute_dtype
                dV /= g0               # row-wise divide
                dV.mul_(s)
                dir_update = dV if dir_update is None else dir_update.add_(dV)


            if dora_mode == "ref_exact":
                # absolute magnitude (last one wins if multiple DoRAs present)
                g = g_abs
            elif dora_mode == "blend":
                # blend towards absolute magnitude proportional to s
                if g is None:
                    g = g0.clone()
                g.add_(g_abs.sub(g0), alpha=s)
            else:
                raise ValueError(f"Unknown dora_mode: {dora_mode}")

            # Optional bias deltas (not in reference, but harmless if present)
            if diff_b is not None:
                db = diff_b.mul(s)
                bias_delta = db if bias_delta is None else bias_delta.add_(db)
                db = None

        if g is None:
            g = g0  # no magnitude provided -> keep original

        # Re-normalize rows if we changed direction
        if dir_update is not None:
            weight.add_(dir_update)
            V = weight.float()
            Vn = torch.linalg.vector_norm(V, dim=1, keepdim=True, dtype=torch.float32).clamp_min(eps)
            V /= Vn
            V *= g
            weight[...] = V.to(base_dtype)
            V = None
        else:
            weight *= g
        # Recompose adapted weight; cast back to module dtype

        # Merge DoRA bias delta safely
        if bias_delta is not None:
            if bias is None:
                bias = bias_delta 
            else:
                bias = bias.clone() if original_bias else bias
                bias.add_(bias_delta)

        return weight, bias

    def _lokr_chunk_forward(self, x_2d, lokr_w1, lokr_w2, specs=None):
        n1 = int(lokr_w1.shape[1])
        n2 = int(lokr_w2.shape[1])
        if len(specs) == 0:
            return x_2d.new_zeros((x_2d.shape[0], 0))

        x_3d = x_2d.view(-1, n1, n2)
        pieces = []
        for r0, r1, c0, c1 in specs:
            y = torch.matmul(lokr_w1[r0:r1].unsqueeze(0), x_3d)
            pieces.append(torch.matmul(y, lokr_w2[c0:c1].T).reshape(-1, (r1 - r0) * (c1 - c0)))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)



    def _lora_linear_forward(self, model, submodule, loras_data, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weight = submodule.weight
        bias = submodule.bias
        active_adapters = model._loras_active_adapters
        loras_scaling = model._loras_scaling
        any_dora = False
        any_lokr = False
        for active_adapter in active_adapters:
            data = loras_data.get(active_adapter + '_GPU', None)
            if data is None:
                continue
            if data[3] is not None:
                any_dora = True
            meta = data[5]
            if meta.get("type", "lora") == "lokr":
                any_lokr = True
            if any_dora and any_lokr:
                break

        dtype = submodule.__dict__.get("_mm_weight_dtype", None) or weight.dtype # cached by _mm_lora_linear_forward: a quantized weight's dtype costs a Python dispatch
        if any_dora and not any_lokr: # sum base weight and lora matrices instead of applying input on each sub lora matrice if input is too large. This will save a lot VRAM and compute
            original_bias = True
            if isinstance(submodule, QModuleMixin):
                weight = weight.dequantize() # materialize weights without quantized view dispatch under inference mode
            else:
                weight = weight.clone()
            for active_adapter in active_adapters:
                data = loras_data.get(active_adapter + '_GPU', None)
                if data is None:
                    continue
                lora_A_weight, lora_B_weight, diff_b, g_abs, alpha = data[:5]
                scaling = self._get_lora_scaling(loras_scaling, model, active_adapter) * alpha
                if scaling == 0 or g_abs is not None:
                    continue
                target_dtype = weight.dtype
                if lora_A_weight is not None and lora_A_weight.dtype != target_dtype:
                    lora_A_weight = lora_A_weight.to(target_dtype)
                if lora_B_weight is not None and lora_B_weight.dtype != target_dtype:
                    lora_B_weight = lora_B_weight.to(target_dtype)
                if diff_b is not None and diff_b.dtype != target_dtype:
                    diff_b = diff_b.to(target_dtype)
                if lora_A_weight is not None:
                    weight.addmm_(lora_B_weight, lora_A_weight, alpha=scaling)
                if diff_b is not None:
                    if bias is None:
                        bias = diff_b.clone()
                        original_bias = False
                    elif original_bias:
                        bias = bias.clone()
                        original_bias = False
                    bias.add_(diff_b, alpha=scaling)
            weight, bias = self._dora_linear_forward(model, submodule, loras_data, weight, bias, original_bias)
            base_bias = bias
            if base_bias is not None and base_bias.dtype != x.dtype:
                base_bias = base_bias.to(x.dtype)
            result = torch.nn.functional.linear(x, weight, bias=base_bias)

        else:
            base_bias = bias
            if base_bias is not None and base_bias.dtype != x.dtype:
                base_bias = base_bias.to(x.dtype)
            if getattr(submodule, "_mm_requires_native_linear_forward", False):
                result = submodule._mm_lora_old_forward(x, *args, **kwargs)
            else:
                result = torch.nn.functional.linear(x, weight, bias=base_bias)

            if active_adapters:
                compute_dtype = result.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)
                x_2d = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
                result_2d = result if result.dim() == 2 else result.reshape(-1, result.shape[-1])

                for active_adapter in active_adapters:
                    data = loras_data.get(active_adapter + '_GPU', None)
                    if data is None:
                        continue
                    lora_A, lora_B, diff_b, g_abs, alpha, adapter_meta = data
                    adapter_type = adapter_meta.get("type", "lora")
                    # dropout = self.lora_dropout[active_adapter]
                    scaling = self._get_lora_scaling(loras_scaling, model, active_adapter) * alpha
                    if scaling == 0 or g_abs is not None:
                        continue
                    target_dtype = result.dtype
                    if lora_A is not None and lora_A.dtype != target_dtype:
                        lora_A = lora_A.to(target_dtype)
                    if lora_B is not None and lora_B.dtype != target_dtype:
                        lora_B = lora_B.to(target_dtype)
                    if diff_b is not None and diff_b.dtype != target_dtype:
                        diff_b = diff_b.to(target_dtype)

                    if adapter_type == "lokr":
                        if lora_A is None or lora_B is None:
                            continue
                        n1 = int(lora_A.shape[1])
                        n2 = int(lora_B.shape[1])
                        out_dim = int(lora_A.shape[0]) * int(lora_B.shape[0])
                        chunk = adapter_meta.get("lokr_chunk", None)
                        if chunk is None:
                            x_3d = x_2d.view(-1, n1, n2)
                            y = torch.matmul(lora_A.unsqueeze(0), x_3d)
                            y = torch.matmul(y, lora_B.T)
                            result_2d.add_(y.reshape(-1, out_dim), alpha=scaling)
                        else:
                            specs = adapter_meta.get("lokr_specs", [])
                            y = self._lokr_chunk_forward(x_2d, lora_A, lora_B, specs=specs)
                            result_2d.add_(y, alpha=scaling)
                        if diff_b is not None:
                            result_2d.add_(diff_b, alpha=scaling)
                        del y
                        continue
                    if adapter_type == "diff":
                        result_2d.addmm_(x_2d, lora_B.T, beta=1, alpha=scaling)
                        if diff_b is not None:
                            result_2d.add_(diff_b, alpha=scaling)
                        continue

                    if lora_A is None:
                        result_2d.add_(diff_b, alpha=scaling)
                    else:
                        y = x_2d @ lora_A.T
                        result_2d.addmm_(y, lora_B.T, beta=1, alpha=scaling)
                        if diff_b is not None:
                            result_2d.add_(diff_b, alpha=scaling)
                        del y
                target_dtype = dtype
                if result.dtype != target_dtype:
                    result = result.to(target_dtype)

        return result


    def hook_lora(self, submodule, current_model, model_id, loras_model_data, loras_model_shortcuts, submodule_name):
        old_forward = submodule.forward

        loras_data = {}
        assert submodule_name not in loras_model_shortcuts 
        loras_model_shortcuts[submodule_name] = loras_data
        loras_model_data[submodule] = loras_data
        submodule._mm_lora_data = loras_data
        submodule.__dict__["_mm_lora_model"] = current_model # not registered as a submodule, which would put the whole model under each LoRA linear
        submodule._mm_lora_old_forward = old_forward

        if isinstance(submodule,  torch.nn.Linear) or getattr(submodule, "is_nvfp4", False):
            target_fn = _mm_lora_linear_forward
        else:
            target_fn = _mm_lora_generic_forward
        return functools.update_wrapper(functools.partial(target_fn, submodule), old_forward)

    def ensure_model_loaded(self, model_id):
        self.used[model_id] = memory_pressure.stamp()
        if self.parked:
            self.unpark()
        if model_id in self.active_models_ids:
            return
        # new_model_id = getattr(module, "_mm_id") 
        # do not always unload existing models if it is more efficient to keep in them in the GPU 
        # (e.g: small modules whose calls are text encoders) 
        if not self.can_model_be_cotenant(model_id):
            self.unload_all(keep=[name for name in self.active_models_ids if self.cotenants_map.get(name) == "*"])
        self.gpu_load(model_id)

    def _cpu_compute_call(self, model_id, blocks_name, args, kwargs):
        """cpuCompute: the rows of this call of a block left out of VRAM if the CPU computes its large weights (0 otherwise): a short call
        (decoding) whose plan gives this block to the CPU (_cpu_plan: the share that timed passes find faster, the others streamed); a block
        already transferred (or on its way) or about to be kept by the automatic preload runs in VRAM."""
        if model_id not in self.cpu_compute or blocks_name in self.preloaded_blocks_per_model[model_id]:
            return 0
        model = self.models[model_id]
        rows = (getattr(model, "_mm_cpu_compute_rows", None) or _call_rows)(args, kwargs)
        entry_name = model_id + "/" + blocks_name
        plan = self._cpu_plan(model_id, rows, entry_name) if rows <= getattr(model, "_mm_cpu_compute_max_rows", CPU_COMPUTE_MAX_ROWS) else (False, (), False)
        short = plan is None or plan[0]  # forced and not measured yet at this row count: the CPU computes, which measures it
        self.cpu_streamed[model_id] = () if plan is None else plan[1]
        self.cpu_measuring[model_id] = plan is not None and plan[2] # each of its blocks the CPU computes is timed
        if short != self.cpu_short.get(model_id, False):
            self.cpu_short[model_id] = short
            self.cpu_pass_start.pop(model_id, None)
            self._retire_inflight(model_id, short)
        if not short or blocks_name == self.loaded_blocks[model_id] or entry_name in self.inflight or entry_name in self.prefetch_futures:
            return 0
        if plan is not None and entry_name in plan[1]:
            return 0
        auto = None if self.auto_preload is None else self.auto_preload.plans.get(model_id)
        return 0 if auto is not None and blocks_name in auto.pending else rows

    def _cpu_plan(self, model_id, rows, entry_name=None):
        """(whether the CPU computes the short calls of this row count, the blocks out of VRAM streamed meanwhile, whether this call measures
        the CPU), None while the CPU is measured for a forced cpuCompute. The CPU takes a share of the blocks out of VRAM only as far as timed
        passes are faster with it (_PassClimb, from none to all of them):
        - a CPU timed alone at this row count at least as fast as the link starts where both would end together, and first tries all of
          them streamed; a slower one starts with all streamed and first tries that balance. The faster of the two is kept, then its
          neighbours are tried. Working beside the transfers, a CPU about as fast as the link shortens the pass (35 instead of 28 tok/s
          for a 27B LLM with 7 blocks out of VRAM on a DDR5 PC); a slow CPU, whose RAM the transfers also use, lengthens it (2.47 -> 1.95
          tok/s on a 4-core DDR4 PC);
        - the blocks streamed are the first of the pinned pool's order (pinned, or crossing the staging ring until the pool holds them), or
          the pinned ones spread evenly, then those that cross the ring;
        - an unmeasured row count is measured on one block out of VRAM in CPU_COMPUTE_TIMED (the others stream). It is not measured, all
          its blocks streamed, when a smaller one found the CPU slower than the link: the CPU's time grows with the rows, a transfer's not.
        A call starts at its first block out of VRAM (entry_name)."""
        preloaded, prefix = self.preloaded_blocks_per_model[model_id], model_id + "/"
        cpu = self.cpu_seconds.get((model_id, rows))
        if cpu is None:
            if self.cpu_compute_forced:
                return None
            if builtins.any(seconds > 1 / _pcie_rate() for (other, measured), seconds in self.cpu_seconds.items() if other == model_id and measured < rows):
                return False, (), False
            out = [entry for entry, _ in self.cpu_compute_entries[model_id] if entry[len(prefix):] not in preloaded]
            return True, {entry for i, entry in enumerate(out) if i % CPU_COMPUTE_TIMED}, True
        cached, climb = self.cpu_plans.get((model_id, rows)), self.cpu_climbs.get((model_id, rows))
        if cached is not None and entry_name == cached[3] and climb is not None:  # a short call starts
            now, previous = time.perf_counter(), self.cpu_pass_start.get(model_id)
            if previous is not None and previous[0] == rows and previous[2] == climb.current() and not _pool.pending: # nor while the pinned pool copies blocks
                climb.add(now - previous[1])
            self.cpu_pass_start[model_id] = (rows, now, climb.current())
        pooled = model_id in self.pool_models
        pcie = 1 / _pcie_rate()
        key = (int(math.log2(cpu) * 8), int(math.log2(pcie) * 8), len(preloaded), None if climb is None else climb.current(), _pool.version if pooled else None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2], False
        out = [entry for entry, _ in self.cpu_compute_entries[model_id] if entry[len(prefix):] not in preloaded]
        streamed, count = set(), 0
        if out:
            if pooled:
                candidates, pinned_count = self._pool_ranked(model_id), None
            else:
                pinned_blocks = [entry for entry in out if not self.needs_staging(entry)]
                candidates, pinned_count = pinned_blocks + [entry for entry in out if entry not in set(pinned_blocks)], len(pinned_blocks)
            climb = self.cpu_climbs.setdefault((model_id, rows), _PassClimb())
            if climb.anchor is None:  # the first plan of this row count: the CPU's share at the rates' balance
                shared = len(candidates) - round(len(candidates) * cpu / (cpu + pcie))
                climb.value = shared if cpu <= pcie or self.cpu_compute_forced else 0
                trial = 0 if climb.value else shared if shared >= CPU_COMPUTE_MIN_SHARE * len(candidates) else 0
                climb.jump = 0 if self.cpu_compute_forced else trial - climb.value
            if climb.anchor != (len(out), len(candidates)):
                climb.anchor = (len(out), len(candidates))
                climb.rebase()
            climb.bound(0, len(candidates))
            count = len(candidates) - climb.current()
            key = key[:3] + (climb.current(), key[4])
            if pooled:
                self.pool_limits[model_id] = count
                for entry in self._pool_wanted(model_id):
                    if not _pool.involved(self, entry):
                        _pool.prepare(self, model_id, entry, self._pool_plan(entry))
                streamed.update(candidates[:count])
            else:
                spread = lambda group, k: [entry for i, entry in enumerate(group) if (i + 1) * k // len(group) > i * k // len(group)] if group else []
                pinned_streamed = builtins.min(count, pinned_count)
                streamed.update(spread(candidates[:pinned_count], pinned_streamed) + spread(candidates[pinned_count:], count - pinned_streamed))
        self.cpu_plans[(model_id, rows)] = (key, bool(out), streamed, out[0] if out else None)
        if self.verboseLevel >= 1 and (cached is None or len(cached[2]) != len(streamed)) and (climb is None or climb.trial is None or self.verboseLevel >= 2):
            tqdm.write(f"CPU compute of '{model_id}' ({rows} rows, {1 / cpu / 1e9:0.1f} GB/s on the CPU, {1 / pcie / 1e9:0.1f} GB/s over PCIe): "
                       f"{len(out) - len(streamed)} of its {len(out)} blocks out of VRAM computed on the CPU, {len(streamed)} streamed")
        return bool(out), streamed, False

    def _cpu_measure(self, model_id, blocks_name, rows, seconds):
        # the first pass of a model pays one-time costs (threads, buffers, clocks of an idle GPU rising); then a row count follows the median of
        # its last blocks (at least CPU_COMPUTE_SAMPLES, twice as many kept), which a burst of slow blocks (another program running) barely moves
        if _pool.pending: # the pinned pool copies blocks: they compete with the CPU for the RAM
            return
        warmup = self.cpu_samples.get(model_id, len(self.cpu_compute_entries[model_id]) // CPU_COMPUTE_TIMED)
        if warmup:
            self.cpu_samples[model_id] = warmup - 1
            return
        key = (model_id, rows)
        samples = self.cpu_samples.setdefault(key, [])
        samples.append(seconds / self.blocks_of_modules_sizes[model_id + "/" + blocks_name])
        del samples[:-2 * CPU_COMPUTE_SAMPLES]
        if len(samples) >= CPU_COMPUTE_SAMPLES:
            self.cpu_seconds[key] = sorted(samples)[len(samples) // 2]

    def _retire_inflight(self, model_id, short):
        # at a switch between short and long calls, the blocks of a model transferred ahead for the other kind (short calls skip the blocks
        # computed on the CPU) go back to the CPU: VRAM never holds more than the window. Before short calls, the staging ring is emptied: its
        # blocks are taken in their order of use, which the next long call starts anew
        prefix = model_id + "/"
        entries = [entry for entry in (*self.inflight, *self.prefetch_futures) if entry.startswith(prefix)]
        if not entries and not (short and self.ring is not None and self.ring.jobs):
            return
        self.wait_prefetch()
        torch.cuda.synchronize() # no copy reads or writes their memory or the ring any more
        for entry in dict.fromkeys(entries):
            self.gpu_unload_blocks(model_id, entry[len(prefix):])
        if short and self.ring is not None:
            self.ring.reset()

    def _cpu_pinning_share(self, model_id):
        """cpuComputePinning "auto": half of the bytes of the towers expected out of VRAM (the model with its _mm_cpu_compute_reserve for its KV
        cache and activations beyond 90% of the VRAM, or the model beyond its _auto_preload_max), as a share of its towers, and never more than
        CPU_COMPUTE_PINNING_RAM_SHARE of the RAM (the reserved RAM left bounds it too). Nothing is pinned for a model that fits in VRAM."""
        model = self.models[model_id]
        towers = builtins.sum(self.blocks_of_modules_sizes[entry] for entry, _ in self.cpu_compute_entries[model_id])
        weights = towers + self.blocks_of_modules_sizes.get(model_id, 0)
        cap = getattr(model, "_auto_preload_max", None)
        out = weights - cap if cap else weights + getattr(model, "_mm_cpu_compute_reserve", 0) - 0.9 * torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        return min(0.5, 0.5 * max(0, out) / towers, CPU_COMPUTE_PINNING_RAM_SHARE * physical_memory / towers) if towers else 0

    def _small_modules_to_base(self, model_id, towers_names):
        """The modules of the blocks of the towers whose own parameters and buffers weigh less than BASE_SMALL_MODULE_SIZE (norms, short
        convolutions, small projections) join the base of the model, in VRAM whenever it is loaded, unless they weigh more than
        BASE_SMALL_MODULES_SHARE of the towers. Streamed with their block, those that cannot join the block's buffer (tensor classes not
        rebuilt from their inner tensors, tensors shared by several blocks) are copied one by one: new tensors at each transfer, which change
        the key of the block's CUDA graphs (captured again instead of replayed), and copies from pageable memory. Tied tensors count at
        their full size, so that a module sharing a large tensor stays in its block."""
        nbytes = lambda p: _subtensors_nbytes(sub_tensors) if (sub_tensors := _get_quantized_subtensors(p)) else p.data.nbytes
        plan, towers_size, moved_size = [], 0, 0
        for tower_name in towers_names:
            prefix = model_id + "/" + tower_name
            for entry in [entry for entry in self.blocks_of_modules if entry.startswith(prefix) and entry[len(prefix):].isdigit()]:
                modules = {}
                for item in self.blocks_of_modules[entry]:
                    modules.setdefault(item[0], []).append(item)
                small = {module for module, items in modules.items() if builtins.sum(nbytes(item[2]) for item in items) < BASE_SMALL_MODULE_SIZE}
                moved = [(item, 0 if item[4] is not None else nbytes(item[2])) for item in self.blocks_of_modules[entry] if item[0] in small]
                plan.append((entry, small, moved))
                towers_size += self.blocks_of_modules_sizes[entry]
                moved_size += builtins.sum(size for _, size in moved)
        if not builtins.any(moved for _, _, moved in plan) or moved_size > BASE_SMALL_MODULES_SHARE * towers_size:
            return
        base = self.blocks_of_modules[model_id]
        for entry, small, moved in plan:
            base += [item for item, _ in moved]
            self.blocks_of_modules[entry] = [item for item in self.blocks_of_modules[entry] if item[0] not in small]
            self.blocks_of_modules_sizes[entry] -= builtins.sum(size for _, size in moved)
        self.blocks_of_modules_sizes[model_id] += moved_size
        if self.verboseLevel >= 2:
            print(f"Model '{model_id}': {builtins.sum(len(small) for _, small, _ in plan)} small modules of its blocks ({moved_size/ONE_MB:0.1f} MB) kept in VRAM with the model")

    def _shared_tensors_to_base(self, model_id):
        """The tensors that modules of different blocks share (e.g. one rotary table for every layer) join the base of the model: streamed
        with the first block that holds them, the other blocks would keep reading that block's copy once its memory goes to the next
        streamed block."""
        nbytes = lambda p: _subtensors_nbytes(sub_tensors) if (sub_tensors := _get_quantized_subtensors(p)) else p.data.nbytes
        entries = [entry for entry in self.blocks_of_modules if entry.startswith(model_id + "/")]
        owners = {(item[0], item[1]): entry for entry in entries for item in self.blocks_of_modules[entry] if item[4] is None}
        shared = {item[4] for entry in entries for item in self.blocks_of_modules[entry] if item[4] is not None and owners.get(item[4], entry) != entry}
        if not shared:
            return
        base, moved_size = self.blocks_of_modules.setdefault(model_id, []), 0
        for entry in entries:  # the owner of a tensor comes first: its block precedes the blocks that share it
            moved = [item for item in self.blocks_of_modules[entry] if (item[0], item[1]) in shared or item[4] in shared]
            if moved:
                size = builtins.sum(nbytes(item[2]) for item in moved if item[4] is None)
                base += moved
                self.blocks_of_modules[entry] = [item for item in self.blocks_of_modules[entry] if (item[0], item[1]) not in shared and item[4] not in shared]
                self.blocks_of_modules_sizes[entry] -= size
                moved_size += size
        self.blocks_of_modules_sizes[model_id] = self.blocks_of_modules_sizes.get(model_id, 0) + moved_size
        if self.verboseLevel >= 2:
            print(f"Model '{model_id}': {len(shared)} tensors shared by several of its blocks ({moved_size/ONE_MB:0.1f} MB) kept in VRAM with the model")

    def _split_cpu_compute(self, model_id, towers_names):
        """cpuCompute: the blocks of the towers keep the large weights that a CPU kernel computes in their storage; their other parameters and
        buffers join the base of the model, in VRAM whenever it is loaded. A model with a large tensor in its blocks that no CPU kernel computes
        (e.g. unquantized weights) keeps its blocks as they are, without CPU compute."""
        from .quant_router import cpu_compute_supported
        nbytes = lambda p: _subtensors_nbytes(sub_tensors) if (sub_tensors := _get_quantized_subtensors(p)) else p.data.nbytes
        plan, unsupported = [], []
        for tower_name in towers_names:
            prefix = model_id + "/" + tower_name
            for entry in [entry for entry in self.blocks_of_modules if entry.startswith(prefix) and entry[len(prefix):].isdigit()]:
                kept, moved = [], []
                for item in self.blocks_of_modules[entry]:
                    parent, name, p, is_buffer, tied_param = item
                    size = 0 if tied_param is not None else nbytes(p)
                    if not is_buffer and tied_param is None and name == "weight" and size >= CPU_COMPUTE_MIN_SIZE and cpu_compute_supported(p):
                        kept.append(item)
                    elif size >= CPU_COMPUTE_MIN_SIZE:
                        unsupported.append(item)
                    else:
                        moved.append((item, size))
                plan.append((entry, kept, moved))
        if unsupported or not plan:
            if unsupported and self.verboseLevel >= 1:
                parent, name, p = unsupported[0][:3]
                print(f"No CPU compute for model '{model_id}': {len(unsupported)} large tensors of its blocks have no CPU kernel (e.g. {type(parent).__name__}.{name}, {type(p).__name__} {p.dtype})")
            return
        base = self.blocks_of_modules.setdefault(model_id, [])
        computed = self.cpu_compute[model_id] = []
        blocks = self.cpu_compute_entries[model_id] = []  # (entry, its computed weights), in tower order
        moved_size = 0
        for entry, kept, moved in plan:
            for item, size in moved:
                base.append(item)
                self.blocks_of_modules_sizes[entry] -= size
                moved_size += size
            self.blocks_of_modules[entry] = kept
            computed += kept
            blocks.append((entry, kept))
        self.blocks_of_modules_sizes[model_id] = self.blocks_of_modules_sizes.get(model_id, 0) + moved_size
        if self.verboseLevel >= 2:
            print(f"CPU compute of model '{model_id}': {len(computed)} weights ({builtins.sum(nbytes(p) for _, _, p, _, _ in computed)/ONE_MB:0.1f} MB) computed on the CPU "
                  f"for short calls of the blocks not in VRAM, {moved_size/ONE_MB:0.1f} MB of other block parameters kept in VRAM with the model")

    def hook_preload_blocks_for_compilation(self, target_module, model_id,blocks_name, context):

        # @torch.compiler.disable()
        def preload_blocks_for_compile(module,  *args, **kwargs):
            # some_context = context #for debugging
            if blocks_name != None and blocks_name != self.loaded_blocks[model_id] and blocks_name not in self.preloaded_blocks_per_model[model_id]:
                self.gpu_load_blocks(model_id, blocks_name)

        # need to be registered before the forward not to be break the efficiency of the compilation chain
        # it should be at the top of the compilation as this type of hook in the middle of a chain seems to break memory performance
        target_module.register_forward_pre_hook(preload_blocks_for_compile)




    @torch._dynamo.disable
    def _pre_check(self, module):
        model_id    = getattr(module, "_mm_model_id", None)
        blocks_name = getattr(module, "_mm_blocks_name", None)

        self.ensure_model_loaded(model_id)
        if blocks_name is None:
            if self.ready_to_check_mem():
                self.empty_cache_if_needed()
        elif blocks_name != self.loaded_blocks[model_id] and \
             blocks_name not in self.preloaded_blocks_per_model[model_id]:
            self.gpu_load_blocks(model_id, blocks_name)

    def _get_wrapper_for_type(self, mod_cls):
        fn = self._type_wrappers.get(mod_cls)
        if fn is not None:
            return fn

        # Unique function name per class -> unique compiled code object
        fname = f"_mm_wrap_{mod_cls.__module__.replace('.', '_')}_{mod_cls.__name__}"

        # Keep body minimal; all heavy/offload logic runs out-of-graph in _pre_check
        # Include __TYPE_CONST in the code so the bytecode/consts differ per class.
        src = f"""
def {fname}(module, *args, **kwargs):
    _ = __TYPE_CONST  # anchor type as a constant to make code object unique per class
    nada = "{fname}"
    mgr = module._mm_manager
    mgr._pre_check(module)
    return module._mm_forward(*args, **kwargs) #{fname}
"""
        ns = {"__TYPE_CONST": mod_cls}
        exec(src, ns)                   # compile a new function object/code object for this class
        fn = ns[fname]
        self._type_wrappers[mod_cls] = fn
        return fn

    def hook_check_load_into_GPU_if_needed(
        self, target_module, model, model_id, blocks_name, previous_method, context
    ):
        # store instance data on the module (not captured by the wrapper)
        target_module._mm_manager     = self
        target_module._mm_model_id    = model_id
        target_module._mm_blocks_name = blocks_name
        target_module._mm_forward     = previous_method

        # per-TYPE wrapper (unique bytecode per class, reused across instances of that class)
        wrapper_fn = self._get_wrapper_for_type(type(target_module))

        # bind as a bound method (no partial/closures)
        # target_module.forward = types.MethodType(wrapper_fn, target_module)
        target_module.forward = functools.update_wrapper(functools.partial(wrapper_fn, target_module), previous_method) 

    def hook_check_load_into_GPU_if_needed_default(self, target_module, model, model_id, blocks_name, previous_method,  context, cpu_root=False):
        dtype = model._dtype
        weight = getattr(target_module, "weight", None)
        weight_qtype = getattr(weight, "qtype", None) if weight is not None else None
        qint4quantization = isinstance(target_module, QModuleMixin) and weight_qtype == qint4
        if qint4quantization:
            pass

        if hasattr(target_module, "_mm_id"):
            # no hook for a shared module with no weights (otherwise this will cause models loading / unloading for nothing)
            orig_model_id = getattr(target_module, "_mm_id")
            if self.verboseLevel >=2:
                print(f"Model '{model_id}' shares module '{target_module._get_name()}' with module(s) '{orig_model_id}' ")
            assert not self.any_param_or_buffer(target_module)
            if not isinstance(orig_model_id, list):
                orig_model_id = [orig_model_id]
            orig_model_id.append(model_id)
            setattr(target_module, "_mm_id", orig_model_id)
            target_module.forward = target_module._mm_forward
            return

        def check_load_into_GPU_needed():
            self.ensure_model_loaded(model_id)
            if blocks_name == None:
                if self.ready_to_check_mem():
                    self.empty_cache_if_needed()
            elif blocks_name != self.loaded_blocks[model_id] and blocks_name not in self.preloaded_blocks_per_model[model_id] and blocks_name != self.cpu_blocks.get(model_id):
                self.gpu_load_blocks(model_id, blocks_name)
            # if qint4quantization and dtype !=None:
            #     args, kwargs = self.move_args_to_gpu(dtype, *args, **kwargs)

        if isinstance(target_module, torch.nn.Linear):
            def check_load_into_GPU_needed_linear(module, *args, **kwargs):
                check_load_into_GPU_needed()
                return previous_method(*args, **kwargs) # linear
            check_load_into_GPU_needed_module = check_load_into_GPU_needed_linear
        else:
            def check_load_into_GPU_needed_other(module, *args, **kwargs):
                check_load_into_GPU_needed()
                return previous_method(*args, **kwargs) # other
            check_load_into_GPU_needed_module = check_load_into_GPU_needed_other
        if cpu_root:
            check_load_into_GPU_needed_gpu = check_load_into_GPU_needed_module
            def check_load_into_GPU_needed_cpu(module, *args, **kwargs):
                # a short call of a block left out of VRAM computes its large weights on the CPU (cpuCompute): it is not transferred
                rows = self._cpu_compute_call(model_id, blocks_name, args, kwargs)
                if not rows:
                    return check_load_into_GPU_needed_gpu(module, *args, **kwargs)
                self.cpu_blocks[model_id] = blocks_name
                calls = self.cpu_block_calls[model_id] = self.cpu_block_calls.get(model_id, 0) + 1
                timed = self.cpu_measuring.get(model_id) or calls % CPU_COMPUTE_TIMED == 0
                if timed:
                    torch.cuda.current_stream().synchronize() # the time measured is that of this block
                start = time.perf_counter()
                try:
                    return check_load_into_GPU_needed_gpu(module, *args, **kwargs)
                finally:
                    self.cpu_blocks[model_id] = None
                    if timed:
                        self._cpu_measure(model_id, blocks_name, rows, time.perf_counter() - start)
            check_load_into_GPU_needed_module = check_load_into_GPU_needed_cpu
        if self.preload_plans and blocks_name is not None:
            check_load_into_GPU_needed_plain = check_load_into_GPU_needed_module
            def check_load_into_GPU_needed_auto(module, *args, **kwargs):
                if self.auto_preload is not None: # the plan of a cycle is set as it starts, before its first block is loaded
                    self.auto_preload.on_block(model_id, blocks_name, args, kwargs)
                try:
                    return check_load_into_GPU_needed_plain(module, *args, **kwargs)
                finally:
                    if self.auto_preload is not None: # after the last block of its towers, a model is between its passes
                        self.auto_preload.after_block(model_id, blocks_name)
            check_load_into_GPU_needed_module = check_load_into_GPU_needed_auto

        setattr(target_module, "_mm_id", model_id)
        setattr(target_module, "_mm_manager", self)
        setattr(target_module, "_mm_forward", previous_method)

        setattr(target_module, "forward", functools.update_wrapper(functools.partial(check_load_into_GPU_needed_module, target_module), previous_method) )
        # target_module.register_forward_pre_hook(check_empty_cuda_cache)

        
    def hook_cpu_module(self, module):
        """A module marked by runtime.lookup_on_cpu: its forward runs on the CPU (tensor inputs copied there, blocking) and its tensor output goes
        to the execution device. One marked by runtime.unused raises if it runs."""
        if hasattr(module, "_mm_cpu_forward"):
            return
        previous = module._mm_cpu_forward = module.forward
        to_cpu = lambda value: value.to("cpu") if torch.is_tensor(value) else value

        def forward(*args, **kwargs):
            if getattr(module, "_mm_unused", False):
                raise RuntimeError(f"{type(module).__name__} was marked by runtime.unused as never running: its weights are not loaded to the GPU")
            out = previous(*[to_cpu(arg) for arg in args], **{key: to_cpu(value) for key, value in kwargs.items()})
            return out.to(torch.device("cuda", torch.cuda.current_device()))

        module.forward = functools.update_wrapper(forward, previous)

    def hook_change_module(self, target_module, model, model_id, module_id, previous_method, previous_method_name ):
        if hasattr(target_module, "_lock_dtype"):
            dtype = target_module._lock_dtype 
        else:
            dtype = model._dtype

        def check_change_module(module, *args, **kwargs):      
            self.ensure_model_loaded(model_id)
            # transfer leftovers inputs that were incorrectly created in the RAM (mostly due to some .device tests that returned incorrectly "cpu")
            if dtype != None:
                args, kwargs = self.move_args_to_gpu(dtype, *args, **kwargs)
            return previous_method(*args, **kwargs) 
  
        if hasattr(target_module, "_mm_" + previous_method_name):
            return
        setattr(target_module, "_mm_Id", model_id)
        setattr(target_module, "_mm_" + previous_method_name, previous_method)

        setattr(target_module, previous_method_name, functools.update_wrapper(functools.partial(check_change_module, target_module), previous_method) )

        if not self.verboseLevel >=1:
            return

        if previous_method_name =="forward" and (module_id == None or module_id ==''):
            model_name = model._get_name()
            print(f"Hooked to model '{model_id}' ({model_name})")



    def tune_preloading(self, model_id, current_budget, towers_names, covers=False):
        # covers: the budget covers the model (by its size): every block is preloaded
        preloaded_blocks = {}
        preload_total = 0
        max_blocks_fetch = 0

        self.preloaded_blocks_per_model[model_id] = preloaded_blocks
        self.towers_per_model[model_id] = towers_names
        self.shuttle_sizes[model_id] = 0

        if current_budget == 0 or towers_names is None or len(towers_names) == 0 or not self.async_transfers:
            return
        base_size = self.blocks_of_modules_sizes[model_id] 
        current_budget -= base_size
        current_budget = max(0, current_budget)
        
        towers = []
        total_size = 0
        for tower_name in towers_names:
            max_floor_size = 0
            tower_size = 0
            floors = []
            prefix = model_id + "/" + tower_name
            for name, size in self.blocks_of_modules_sizes.items():
                if name.startswith(prefix):
                    tower_size += size
                    floor_no = int(  name[len(prefix): ] )
                    floors.append( (name, floor_no, size))
                    max_floor_size = max(max_floor_size, size)

            towers.append( (floors, max_floor_size, tower_size) )
            total_size += tower_size

        # the shuttle takes window + 1 of the largest blocks from the budget: by default a window of 2 when blocks are also preloaded, which
        # smooths the transfers between the preloaded blocks without VRAM beyond the budget, 1 otherwise
        largest = max(max_floor_size for _, max_floor_size, _ in towers)
        def preload_counts(budget):
            # a parameter-free tower (e.g. a ModuleList of Identity modules) has nothing to preload; a budget beyond the model preloads all the blocks
            return [min(len(floors), int(tower_size / total_size * max(0, budget) / max_floor_size)) if tower_size > 0 else 0 for floors, max_floor_size, tower_size in towers]
        window = self.prefetch_window
        if window is None:
            window = 2 if builtins.sum(preload_counts(current_budget - 3 * largest)) > 0 else 1
        self.planned_windows[model_id] = window
        # the other blocks of the model (a Sequential, a list of fewer layers than a tower: refiners, embeddings, heads), but those it loads
        # only while they run: each runs at every pass as a block of the towers does, and keeping it frees no slot. Preloaded first, as the
        # dynamic preload ranks them: the longest run of them that the budget holds beside the shuttle, then the blocks of the towers. A budget
        # covering the model preloads every block (no shuttle until the VRAM pressure releases some)
        separate = tuple(model_id + "/" + name for name in getattr(self.models[model_id], "_offload_separate_blocks", ()))
        others = [(name, size) for name, size in self.blocks_of_modules_sizes.items() if size and name.startswith(model_id + "/")
                  and not name.startswith(tuple(model_id + "/" + tower_name for tower_name in towers_names)) and not name.startswith(separate)]
        if covers or current_budget >= total_size + builtins.sum(size for _, size in others):
            counts = [len(floors) for floors, _, _ in towers]
        else:
            room, kept = current_budget - (window + 1) * largest, 0
            while PRELOAD_BLOCKS_OUTSIDE_TOWERS and kept < len(others) and others[kept][1] <= room:
                room -= others[kept][1]
                kept += 1
            others, counts = others[:kept], preload_counts(room)
        for name, size in others:
            self.next_blocks_names[name] = None
            self.prev_blocks_names[name] = None
            preloaded_blocks[name[len(model_id) + 1:]] = size
            preload_total += size
        others_total = preload_total

        for (floors, max_floor_size, tower_size), preload_blocks_count in zip(towers, counts):
            max_blocks_fetch = max(max_floor_size, max_blocks_fetch)
            self.shuttle_sizes[model_id] = max_blocks_fetch
            
            nb_blocks= len(floors)
            if preload_blocks_count == 0:
                space_between = 0
                cursor = len(floors)
            else:
                space_between =  (nb_blocks - preload_blocks_count) / preload_blocks_count 
                cursor = space_between
            first_non_preloaded = None
            prev_non_preloaded = None
            for block in floors:
                name, i, size = block
                if i < cursor:
                    if prev_non_preloaded == None:
                        first_non_preloaded = name
                    else:
                        self.next_blocks_names[prev_non_preloaded] = name
                        self.prev_blocks_names[name] = prev_non_preloaded
                    prev_non_preloaded = name
                else:
                    self.next_blocks_names[name] = None
                    self.prev_blocks_names[name] = None
                    preloaded_blocks[name[ len(model_id) + 1 : ] ] = size
                    preload_total += size
                    cursor += 1 + space_between

            if prev_non_preloaded != None and len(towers) == 1 : 
                self.next_blocks_names[prev_non_preloaded] = first_non_preloaded
                self.prev_blocks_names[first_non_preloaded] = prev_non_preloaded
                self.learned_links.add(prev_non_preloaded) # a guess, until the block that comes next is known
            else:
                self.next_blocks_names[prev_non_preloaded] = None

        towers_total = preload_total - others_total
        for entry, target in self.next_blocks_names.items(): # no link towards a preloaded block: it would be transferred again while it runs
            if target is not None and target.startswith(model_id + "/") and target[len(model_id) + 1:] in preloaded_blocks:
                self.next_blocks_names[entry] = None
        self.preloaded_blocks_per_model[model_id] = preloaded_blocks
        # what the budget leaves beyond the base, the preloaded blocks and the shuttle of the streamed ones: room for state kept in VRAM
        self.state_rooms[model_id] = max(0, current_budget - preload_total - (0 if towers_total == total_size else (window + 1) * largest))

        if self.verboseLevel >=1:
            if preload_total == 0:
                tqdm.write(f"Async loading plan for model '{model_id}' : base size of {(preload_total+base_size)/ONE_MB:0.2f} MB will be preloaded with a {max_blocks_fetch/ONE_MB:0.2f} MB async" + (" circular" if len(towers) == 1 else "") + f" shuttle, prefetch window {window}")
            else:
                tqdm.write(f"Async loading plan for model '{model_id}' : {(preload_total+base_size)/ONE_MB:0.2f} MB will be preloaded (base size of {base_size/ONE_MB:0.2f} MB + {towers_total/total_size*100 if total_size else 0:0.1f}% of recurrent layers data" + (f" + {others_total/ONE_MB:0.2f} MB of other blocks" if others_total else "") + f") with a {max_blocks_fetch/ONE_MB:0.2f} MB async" + (" circular" if len(towers) == 1 else "") + f" shuttle, prefetch window {window}")

    def release(self):
        global last_offload_obj

        if self.models is None:
            return

        self.unload_all()
        for handle in self.expert_hooks:
            handle.remove()
        self.expert_hooks.clear()
        for residency in self.expert_residency.values():
            residency.close()
        self.expert_residency.clear()
        if self.cuda_graphs:
            from .cuda_graphs import release
            release(self.cuda_graphs.values())
            self.cuda_graphs = {}
        if self.auto_preload is not None:
            self.auto_preload.close()
            self.auto_preload = None
        if self.prefetch_executor is not None:
            self.prefetch_executor.shutdown(wait=True)
            self.prefetch_executor = None
        self.inflight = {}
        self._release_ring()
        _pool.close(self)
        _managers.discard(self)
        self.stager.release()
        self.stager = None
        self.block_items.clear() # cached tensors and bulk-copy views also own the registered RAM
        self.active_models = None
        self.default_stream = None
        self.transfer_stream = None
        self.parameters_ref = None
        keys= [k for k in self.blocks_of_modules.keys()]
        for k in keys:
            del self.blocks_of_modules[k]

        self.blocks_of_modules = None

        for model_id, model in self.models.items():
            move_loras_to_device(model, "cpu")
            if hasattr(model, "_pinned_bytes") and not getattr(model, "_pinned_bytes_by_owner", False):
                pinned.account(-model._pinned_bytes, "model")
                model._pinned_bytes = 0
            if hasattr(model, "_loras_model_data"):
                unload_loras_from_model(model)
            model = None

        self.models = None            
        runtime._unregister(self)

        if last_offload_obj == self:
            last_offload_obj = None

        gc.collect()
        pinned.collect()
        torch.cuda.empty_cache()




class LoadingCancelled(Exception):
    """Loading was cancelled; discard the interrupted pipeline."""


_loading_context = ContextVar("mmgp_loading_context", default=None)


@contextmanager
def loading_context(callback, model_ids=None):
    """Forward optional progress through model construction and checkpoint loading."""
    if callback is None:
        yield
        return
    model_ids = {} if model_ids is None else {os.path.normcase(os.path.abspath(path)): name for path, name in model_ids.items()}
    token = _loading_context.set((callback, model_ids))
    try:
        callback.report("Preparing Models", 0, 1, "")
        yield
    finally:
        _loading_context.reset(token)


def _report_loading(phase, completed, total, paths):
    context = _loading_context.get()
    if context is None:
        return
    callback, model_ids = context
    paths = paths if isinstance(paths, list) else [paths]
    model_id = next((model_ids[os.path.normcase(os.path.abspath(path))] for path in paths if isinstance(path, str) and os.path.normcase(os.path.abspath(path)) in model_ids), "")
    callback.report(phase, completed, total, model_id)


class LoadingCallback:
    def __init__(self, abort_requested=None, progress=None):
        self.abort_requested = abort_requested
        self.progress = progress

    def check_abort(self):
        if self.abort_requested is not None and self.abort_requested():
            raise LoadingCancelled("Model loading cancelled")

    def report(self, phase, completed, total, model_id):
        self.check_abort()
        if self.progress is not None:
            self.progress(phase, completed, total, model_id)
        self.check_abort()


def all(pipe_or_dict_of_modules, pinnedMemory = False, pinnedPEFTLora = False, partialPinning = False, loras = None, quantizeTransformer = True,  extraModelsToQuantize = None, quantizationType = qint8, budgets= 0, minimumBudget = 0, workingVRAM = None, asyncTransfers = True, compile = False, convertWeightsFloatTo = torch.bfloat16, perc_reserved_mem_max = 0, coTenantsMap = None, vram_safety_coefficient = 0.85, compile_mode ="default", verboseLevel = -1, loading_callback=None, prefetchWindow = None, smartPinning = None, autoPreload = False, cudaGraphs = False, preloadExperts = True, cpuCompute = False, cpuComputePinning = 0, pinPool = False, cpuComputeForce = False, preloadPriority = None):
    """Configure offloading, optionally reporting phases and accepting cancellation."""
    global last_offload_obj
    kwargs = locals().copy()
    previous_pinnable_limit = pinned.limit_bytes
    device_context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
    previous_device, previous_offload = None if device_context is None else device_context.device, last_offload_obj
    self = offload()
    try:
        return _configure(self, **kwargs)
    except LoadingCancelled as error:
        # Drop interrupted pinning frames before releasing their model references.
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
        self.release()
        pinned.limit_bytes = previous_pinnable_limit # live registrations keep their actual charge, even after cancellation
        last_offload_obj = previous_offload
        torch.set_default_device(previous_device)
        raise error from None


def _configure(self, pipe_or_dict_of_modules, pinnedMemory = False, pinnedPEFTLora = False, partialPinning = False, loras = None, quantizeTransformer = True,  extraModelsToQuantize = None, quantizationType = qint8, budgets= 0, minimumBudget = 0, workingVRAM = None, asyncTransfers = True, compile = False, convertWeightsFloatTo = torch.bfloat16, perc_reserved_mem_max = 0, coTenantsMap = None, vram_safety_coefficient = 0.85, compile_mode ="default", verboseLevel = -1, loading_callback=None, prefetchWindow = None, smartPinning = None, autoPreload = False, cudaGraphs = False, preloadExperts = True, cpuCompute = False, cpuComputePinning = 0, pinPool = False, cpuComputeForce = False, preloadPriority = None):
    """Hook to a pipeline or a group of modules in order to reduce their VRAM requirements:
    pipe_or_dict_of_modules : the pipeline object or a dictionary of modules of the model
    quantizeTransformer: set True by default will quantize on the fly the video / image model
    pinnedMemory: move models in reserved memor. This allows very fast performance but requires 50% extra RAM (usually >=64 GB)
    extraModelsToQuantize: a list of models to be also quantized on the fly (e.g the text_encoder), useful to reduce bith RAM and VRAM consumption
    convertWeightsFloatTo: default dtype for remaining FP32 parameters. A model-level _convertWeightsFloatTo overrides it; None preserves FP32.
    budgets: 0 by default (unlimited). If non 0, it corresponds to the maximum size in MB that every model will occupy at any moment.
        0 loads a model whole; a budget at least its size preloads every block of its towers, which leave the VRAM under pressure as any
        preload instead of spilling (a model without towers is loaded whole)
    minimumBudget: 0 by default (none). The least budget in MB of every model, also of those given one in budgets or by a _budget
        attribute (in bytes): a smaller budget is raised to it, 0 (the whole model) stays. Models kept in VRAM together (co-tenants of
        each other, coTenantsMap) share it as a total instead: each keeps its own budget, and what it leaves beyond those goes to them in
        the order of preloadPriority (default: transformer, transformer2, then the others), each up to its size; what remains then is room
        for state kept in VRAM (mmgp.state_ring). At most vram_safety_coefficient of the VRAM
    preloadExperts: with an explicit budget, use its remaining space for individual GPU experts after ordinary blocks; a model loaded
        whole (budget 0) gives them what its share of minimumBudget leaves beyond its other weights. Dynamic preload applies the same
        priority independently. False keeps experts on CPU for fixed-budget execution. With pinnedMemory=True,
        the reserved RAM the models leave also holds pinned copies of the experts that can go to the GPU, most used first, so that
        placing them is a plain transfer; the experts that stay on the CPU are never pinned.
        (in fact the real usage is twice this number). It is very efficient to reduce VRAM consumption but this feature may be very slow
        if pinnedMemory is not enabled
    vram_safety_coefficient: float between 0 and 1 (exclusive), default 0.85. Sets the maximum portion of VRAM that can be used for models.
        Lower values provide more safety margin but may reduce performance.        
    prefetchWindow: with asyncTransfers, number of blocks transferred ahead of the block being computed. None (default): 2 for the models whose
        budget also preloads blocks, 1 for the others. The host queues the copies without waiting for the GPU, which orders them after the
        compute of the blocks whose memory they reuse: the shuttle holds window + 1 blocks, taken from the budget of the model. Windows
        beyond 1 fall back to 1 when their extra blocks exceed PREFETCH_VRAM_SHARE of the VRAM. 0 keeps the shuttle synchronized by the host
        at every block. Windows apply to models whose blocks are pinned or go through the staging ring.
        Hints never block copies; other platforms keep their existing path.
    autoPreload: False by default. With the MMGP VRAM allocator (mmgp.allocator), the models processed block by block (with a budget, async
        transfers, not compiled) keep in VRAM as many of their blocks as the VRAM left by the activations of each workload allows, measured
        during its first cycle; the preload of their budget applies to the workloads not measured yet. True keeps the measures for the
        process, a dictionary keeps them in it (e.g. tied to a model definition, measured again when it changes). See mmgp.auto_preload.
    cudaGraphs: False by default. True or model ids: the blocks of the towers of these models replay CUDA graphs in the scopes their code
        opens (mmgp.cuda_graphs.scope, e.g. around the denoising loop), which removes most of the host work of short steps. A block has a
        graph per location of its weights (kept in VRAM, or each slot of the swapped blocks); its LoRA copies come and go with it as without
        graphs, copied before each replay into buffers that its graphs read.
        See mmgp.cuda_graphs for the contract of a block call.
    cpuCompute: False by default. True, "auto" (the models that declare _mm_cpu_compute_reserve, e.g. language model decoders) or model ids:
        the blocks of the towers of these models that are not kept in VRAM compute their large
        linears on the CPU, in their storage, for short calls (decoding: at most the model's _mm_cpu_compute_max_rows rows, CPU_COMPUTE_MAX_ROWS
        by default) instead of being transferred; longer calls (prompts) stream them as before. A linear computes on the CPU when its quantization
        handler has a CPU kernel for its weight (quant_router.cpu_compute_supported); the rest of each block (norms, small weights) joins the
        model's base, always in VRAM with it. The model may count the rows of a block call itself (_mm_cpu_compute_rows(args, kwargs)). The CPU
        computes the short calls of a row count only when its measured time beats the transfers through the staging ring, which keeps the formats
        whose decoding costs more than their bytes (2-bit or ternary weights) streamed.
    cpuComputeForce: False by default. True: the CPU computes the short calls of the cpuCompute models whatever the measured times (to test
        it; the measured rates still balance the blocks streamed meanwhile). Weights without a CPU kernel keep streaming.
    cpuComputePinning: 0 by default. A share (or model id -> share) of the blocks of the towers of the cpuCompute models pinned to reserved RAM,
        spread evenly: in short calls, as many pinned blocks out of VRAM as end their transfers with the CPU computing the others are streamed
        meanwhile (the CPU does not cross the PCIe link). About half of the blocks out of VRAM is enough on a PCIe 4.0 x16 desktop: "auto" pins
        that, from the VRAM, the size of the model and its _mm_cpu_compute_reserve (bytes its runtime needs beside its weights), nothing when it
        fits in VRAM, and at most CPU_COMPUTE_PINNING_RAM_SHARE of the RAM.
    smartPinning: None by default. Minimum pinning of the models processed block by block (with a budget) that pinnedMemory does not pin:
        with asyncTransfers, 0 copies their blocks ahead through a staging ring of reserved RAM (RING_DEFAULT_BLOCKS of the largest such
        blocks, copied by a background thread, then transferred like pinned blocks), n > 0 also pins 1 block in n of them, spread evenly.
        Reserved RAM is shared in this order: the staging ring if blocks are left unpinned, the models of pinnedMemory in the order of the
        pipe (first their blocks processed at each step, spread evenly if they do not all fit, the ring taking the others, then their blocks
        preloaded in VRAM, then their parameters outside the blocks), then this minimum.
    pinPool: False by default. True or model ids, with asyncTransfers: the blocks these models stream are pinned while they run, into one
        pool of reserved RAM shared by every model of the process, instead of RAM pinned for good for each model. A block enters at its first
        transfer (cpuCompute: when its short calls stream it): the pages its tensors in private RAM fill are locked in place, the rest (tensors
        mapped from a checkpoint file) copied into locked RAM; its transfers then read pinned memory. The pool takes the blocks out of VRAM spread evenly, up to the
        reserved RAM that pinnedMemory and the staging ring leave (perc_reserved_mem_max), while the system keeps POOL_RAM_FLOOR available;
        the running model comes first: the entries of models that do not run leave when it needs their room or the RAM runs short, at quiet
        points only (models loaded or unloaded). The blocks it cannot take cross the staging ring (made as with smartPinning 0). Blocks pinned
        for good (pinnedMemory, smartPinning n) keep their pinning. For cpuCompute models, the pool holds the blocks that the balance of the CPU
        and the PCIe link streams (cpuComputePinning does not apply).
    """
    self.verboseLevel = verboseLevel
    safetensors2.verboseLevel = verboseLevel
    self.modules_data = {}
    
    model_budgets = {}

    windows_os =  os.name == 'nt'

    def get_parsed_budget(b):
        if isinstance(b , str) and b.endswith("%"):
            return float(b[:-1]) / 100 * self.device_mem_capacity
        else:
            return b * ONE_MB

    # Validate vram_safety_coefficient
    if not isinstance(vram_safety_coefficient, float) or vram_safety_coefficient <= 0 or vram_safety_coefficient >= 1:
        raise ValueError("vram_safety_coefficient must be a float between 0 and 1 (exclusive)")

    budget = 0
    if not budgets is None:
        if isinstance(budgets , dict):
            model_budgets = { k : get_parsed_budget(b) for k , b in budgets.items() } 
            budget = model_budgets.get("*", 0)
        else:
            budget = get_parsed_budget(budgets)
    minimum_budget = min(get_parsed_budget(minimumBudget), vram_safety_coefficient * self.device_mem_capacity)
    model_sizes = {} # model id -> bytes of its weights (beyond its expert copies): the co-tenants' shares of the preload need them all

    self.async_transfers = asyncTransfers
    self.prefetch_window = prefetchWindow
    from . import allocator as vram_allocator
    self.shared_weight_stream = vram_allocator.active == "vmm"
    auto_measures = autoPreload if isinstance(autoPreload, dict) else None
    autoPreload = autoPreload is True or auto_measures is not None
    if autoPreload and vram_allocator.active != "vmm":
        print("Automatic VRAM preload needs the MMGP VRAM allocator (mmgp.allocator.install()): the budgets apply")
        autoPreload = False
    elif autoPreload and not asyncTransfers:
        print("Automatic VRAM preload needs asynchronous transfers (the blocks are kept as they are transferred ahead): the budgets apply")
    self.auto_preload_requested = bool(autoPreload) and asyncTransfers # the plans are made of blocks transferred ahead
    # without it, the blocks of a budget's preload leave the VRAM under pressure instead of the allocation spilling into RAM (MMGP allocator)
    self.preload_plans = self.auto_preload_requested or vram_allocator.active == "vmm" and asyncTransfers
    auto_models, fixed_models = [], []



    torch.set_default_device(None) # cpu, without the DeviceContext mode that routes every torch call through Python

    ignored_models = getattr(pipe_or_dict_of_modules, "_mmgp_ignore_models", ())
    if hasattr(pipe_or_dict_of_modules, "components"):
        # create a fake Accelerate parameter so that lora loading doesn't change the device
        pipe_or_dict_of_modules.hf_device_map = torch.device("cuda")
        pipe_or_dict_of_modules= pipe_or_dict_of_modules.components 

    
    models = {k: _remove_model_wrapper(v) for k, v in pipe_or_dict_of_modules.items() if isinstance(v, torch.nn.Module) and k not in ignored_models}

    
    verboseLevel = _compute_verbose_level(verboseLevel)

    if coTenantsMap != None:
        self.cotenants_map = coTenantsMap 
    if loras != None and isinstance(loras, str):
        loras = [loras]
    self.models = models

    extraModelsToQuantize =  extraModelsToQuantize if extraModelsToQuantize is not None else []
    if not isinstance(extraModelsToQuantize, list):
        extraModelsToQuantize= [extraModelsToQuantize]
    if quantizeTransformer:
        extraModelsToQuantize.append("transformer")            
    models_to_quantize = extraModelsToQuantize

    modelsToPin = []
    pinAllModels = False
    if isinstance(pinnedMemory, bool):
        pinAllModels = pinnedMemory
    elif isinstance(pinnedMemory, list):            
        modelsToPin = pinnedMemory
    else:
        modelsToPin = [pinnedMemory]

    modelsToCompile = []
    compileAllModels = False
    if isinstance(compile, bool):
        compileAllModels = compile
    elif isinstance(compile, list):            
        modelsToCompile = compile
    else:
        modelsToCompile = [compile]

    self.anyCompiledModule = compileAllModels or len(modelsToCompile)>0
    graphed_models = [] if isinstance(cudaGraphs, bool) else cudaGraphs if isinstance(cudaGraphs, list) else [cudaGraphs]
    cpu_models = list(models) if cpuCompute is True else [] if not cpuCompute else [model_id for model_id, model in models.items() if hasattr(model, "_mm_cpu_compute_reserve")] if cpuCompute == "auto" \
                 else cpuCompute if isinstance(cpuCompute, list) else [cpuCompute]
    self.pool_models = [] if not pinPool or not asyncTransfers else list(models) if pinPool is True else pinPool if isinstance(pinPool, list) else [pinPool]
    self.cpu_compute_forced = cpuComputeForce
    if self.anyCompiledModule:
        torch.compiler.reset()
        torch._dynamo.config.cache_size_limit = 10000
    #dynamic=True

      #  torch._logging.set_logs(recompiles=True)
      #  torch._inductor.config.realize_opcount_threshold = 100 # workaround bug "AssertionError: increase TRITON_MAX_BLOCK['X'] to 4096."

    perc_reserved_mem_max = _get_perc_reserved_mem_max(perc_reserved_mem_max)
    max_reservable_memory = self.max_reservable_memory = _get_max_reservable_memory(perc_reserved_mem_max)

    phase_no = 0
    phase_total = 4 * len(models) + sum(model_id in models_to_quantize for model_id in models)

    def loading_phase(phase, model_id):
        nonlocal phase_no
        if loading_callback is not None: loading_callback.report(phase, phase_no, phase_total, model_id)
        phase_no += 1

    from . import experts
    expert_banks = {model_id: experts.banks(model) for model_id, model in models.items()}
    expert_exclusions = {model_id: experts.excluded_names(model) for model_id, model in models.items() if expert_banks[model_id]}
    expert_budgets = {}
    listed = {model_id: pinAllModels or model_id in modelsToPin for model_id in models} # the models to pin, as far as the reserved RAM allows
    self.pinned_models = {model_id for model_id, pin in listed.items() if pin} | set(self.pool_models) # their state rings may lock RAM too
    if not asyncTransfers:
        smartPinning = None # the staging ring needs asynchronous transfers
    elif self.pool_models and smartPinning is None:
        smartPinning = 0 # the blocks the pinned pool cannot take cross a staging ring
    self.ring_on_demand = None if smartPinning is None else (max_reservable_memory, verboseLevel)
    # the modules marked by runtime.map_in_ram (and their aliases in any of these models) go to mapped RAM first: in reserved RAM before the
    # pinning, and out of the models' blocks and base whatever their order
    from . import mapped
    mapped.map_parameters(models, max_reservable_memory, self.verboseLevel)
    for model_id in models: 
        if loading_callback is not None: loading_callback.check_abort()
        current_model: torch.nn.Module = models[model_id] 
        loading_phase("Preparing", model_id)
        model_convertWeightsFloatTo = getattr(current_model, "_convertWeightsFloatTo", convertWeightsFloatTo)
        # make sure that no RAM or GPU memory is not allocated for gradiant / training
        _to_cpu(current_model).eval()
        
        # if the model has just been quantized so there is no need to quantize it again
        if model_id in models_to_quantize:
            loading_phase("Quantization", model_id)
            _quantize(current_model, weights=quantizationType, verboseLevel = self.verboseLevel, model_id=model_id)

        loading_phase("Scanning Sizes", model_id)

        current_model_size = 0
        model_dtype = getattr(current_model, "_model_dtype", None)
        # if model_dtype == None:
        #     model_dtype = getattr(current_model, "dtype", None)
        counted = set() # a tensor held by several parameters (e.g. an embedding shared by two heads) is counted once
        cpu_modules = _cpu_modules(current_model)
        for module_name, m in current_model.named_modules():
            ignore_dtype = hasattr(m, "_lock_dtype")
            on_cpu = any(module_name == name or module_name.startswith(name + ".") for name in cpu_modules) # never in VRAM
            for n, p in m.named_parameters(recurse = False):
                p.requires_grad = False
                if _is_mapped(p): # read in place from mapped RAM: no VRAM, no conversion
                    continue
                sub_tensors = _get_quantized_subtensors(p)
                if sub_tensors:
                    ref = _get_tensor_ref(p)
                    if ref not in counted and not on_cpu:
                        counted.add(ref)
                        current_model_size += _subtensors_nbytes(sub_tensors)
                    del sub_tensors
                else:
                    if not ignore_dtype:
                        dtype = p.data.dtype
                        if model_convertWeightsFloatTo != None and dtype == torch.float32 :
                            # convert any left overs float32 weight to bfloat16 / float16 to divide by 2 the model memory footprint
                            dtype = model_convertWeightsFloatTo if model_dtype == None else model_dtype
                            if dtype != torch.float32:
                                p.data = p.data.to(dtype)
                        if model_dtype is None:
                            model_dtype = dtype
                        else:
                            if model_dtype != dtype:
                                pass
                            assert model_dtype == dtype
                    if p.data_ptr() not in counted and not on_cpu: # after a conversion, which gives the parameter a copy of its own
                        counted.add(p.data_ptr())
                        current_model_size +=  torch.numel(p.data) * p.data.element_size()
        if model_dtype is None:
            model_dtype = model_convertWeightsFloatTo if model_convertWeightsFloatTo is not None else torch.bfloat16
        current_model._dtype = model_dtype
        cpu_buffers = {id(b) for name in cpu_modules for b in current_model.get_submodule(name).buffers()}
        for b in current_model.buffers():
            # do not convert 32 bits float to 16 bits since buffers are few (and potential gain low) and usually they are needed for precision calculation (for instance Rope)
            if id(b) not in cpu_buffers:
                current_model_size +=  torch.numel(b.data) * b.data.element_size()

        # Optional GPU expert copies have their own, lower-priority residency budget.
        current_model_size -= builtins.sum(bank.num_experts * bank.expert_nbytes for bank in expert_banks[model_id].values())
        model_sizes[model_id] = current_model_size

    # the VRAM preload: the least budget of each model (its own: budgets, or a _budget attribute in bytes), shared by the models kept in VRAM
    # together
    requested = {model_id: models[model_id]._budget if hasattr(models[model_id], "_budget") else model_budgets[model_id] if model_id in model_budgets else budget
                 for model_id in models}
    preload_floors, whole_shares = _preload_floors(list(models), model_sizes, requested, minimum_budget, self.cotenants_map, preloadPriority)
    for model_id in models:
        current_model_size = model_sizes[model_id]
        model_budget = preload_floors[model_id]
        if workingVRAM != None:
            model_minimumVRAM = -1
            if isinstance(workingVRAM, dict):
                if model_id in workingVRAM:
                    model_minimumVRAM = get_parsed_budget(workingVRAM[model_id])
                elif "*" in model_id in workingVRAM:
                    model_minimumVRAM = get_parsed_budget(workingVRAM["*"])
            else:
                model_minimumVRAM = get_parsed_budget(workingVRAM)

            if model_minimumVRAM > 0:
                new_budget = self.device_mem_capacity -  model_minimumVRAM
                new_budget = 1 if new_budget  < 0 else new_budget
                model_budget =  new_budget if model_budget == 0 or new_budget < model_budget else model_budget
        # a model loaded whole (budget 0) leaves its expert copies what its share of the preload holds beyond its other weights
        expert_budgets[model_id] = (model_budget or whole_shares.get(model_id, 0)) if preloadExperts else 0
        coef =vram_safety_coefficient
        self.state_rooms[model_id] = 0 # room for state kept in VRAM (mmgp.state_ring): what a budget leaves (here, or tune_preloading)
        if  model_budget > 0 and current_model_size <= model_budget:
            # the budget covers the model: every block preloaded, released under VRAM pressure as any preload (a model without towers is
            # loaded whole); the rest, within the VRAM the models may use, is room for state (a model larger than that streams the rest)
            model_budget = min(model_budget, coef * self.device_mem_capacity)
            self.state_rooms[model_id] = max(0, model_budget - current_model_size)
        if current_model_size > coef * self.device_mem_capacity and model_budget == 0 or model_budget > coef * self.device_mem_capacity:
            if verboseLevel >= 1:
                if model_budget == 0:
                    print(f"Model '{model_id}' is too large ({current_model_size/ONE_MB:0.1f} MB) to fit entirely in {coef * 100:.0f}% of the VRAM (max capacity is {coef * self.device_mem_capacity/ONE_MB:0.1f}) MB)")
                else:
                    print(f"Budget ({model_budget/ONE_MB:0.1f} MB) for Model '{model_id}' is too important so that this model can fit in the VRAM (max capacity is {self.device_mem_capacity/ONE_MB}) MB)")
                print(f"Budget allocation for this model has been consequently reduced to the {coef * 100:.0f}% of max GPU Memory ({coef * self.device_mem_capacity/ONE_MB:0.1f} MB). This may not leave enough working VRAM and you will probably need to define manually a lower budget for this model.")
                model_budget = coef * self.device_mem_capacity 
                
        
        model_budgets[model_id] = model_budget

    runtime._register(self, models)

    #  Hook forward methods of modules 
    for model_id in models: 
        if loading_callback is not None: loading_callback.check_abort()
        current_model: torch.nn.Module = models[model_id] 
        current_model._force_device= "cuda"
        # Short recurrent stacks can explicitly join the graph/offload plan so their larger blocks do not discard the fixed weight slots.
        graph_towers = getattr(current_model, "_cuda_graph_towers", ()) if cudaGraphs is True or model_id in graphed_models else ()
        towers_names, towers_modules = _detect_main_towers(current_model, extra_towers=graph_towers)
        # modules kept with the base of the model, never streamed as blocks: their weights are read outside their forward (e.g. the
        # codebooks of a vector quantizer looked up by index)
        base_modules = tuple(getattr(current_model, "_offload_base_modules", ()))
        in_base = lambda name: builtins.any(name == base or name.startswith(base + ".") for base in base_modules)
        # modules used by several modules (e.g. one rotary embedding for all the layers) join the base too: streamed with the first block
        # that holds them, their hooks would load that block again from the others. Not the lists and the modules holding lists: a list of
        # layers held twice, a layer in two lists, a model held by a wrapper too (an alias) keep their blocks.
        containers = (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)
        parents = collections.Counter(child for module in current_model.modules() if not isinstance(module, containers)
                                      for child in module._modules.values() if child is not None and not isinstance(child, containers))
        shared_modules = {sub for module, count in parents.items() if count > 1 and not builtins.any(isinstance(sub, containers) for sub in module.modules())
                          for sub in module.modules()}
        if base_modules:
            names_of = {module: name for name, module in current_model.named_modules()}
            towers_names, towers_modules = [name for name in towers_names if not in_base(name.rstrip("."))], [module for module in towers_modules if not in_base(names_of[module])]
        compilationInThisOne = compileAllModels or model_id in modelsToCompile
        graphs = None
        if (cudaGraphs is True or model_id in graphed_models) and towers_modules:
            from .cuda_graphs import BlockGraphs
            graphs = self.cuda_graphs[model_id] = BlockGraphs(model_id, functools.partial(self._graph_weights_key, current_model), functools.partial(self._graph_weight_tensors, current_model),
                                                              functools.partial(self._graph_lora_inputs, current_model), functools.partial(self._graph_ready, model_id),
                                                              functools.partial(self._graph_forget, current_model))
            towers_set = set(towers_modules)

        loading_phase("Hooks and LoRA Slots", model_id)
        current_budget = model_budgets[model_id] # with the preload (its _budget attribute included)
        covers = current_budget >= model_sizes[model_id] > 0
        if covers and (not towers_names or not COVERING_BUDGET_BLOCKS):
            current_budget = 0 # a budget covering a model without towers (or COVERING_BUDGET_BLOCKS off): loaded whole
        cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq, is_mod_seq = None, None, None, -1, False
        # BASE_BLOCKS_OUTSIDE_TOWERS: the blocks outside the main towers of a model that has some (a Sequential, a list of fewer layers than a
        # tower: embeddings, heads, refiners) join its base, loaded with the model rather than streamed at each pass
        floors, in_tower = set(towers_modules) if BASE_BLOCKS_OUTSIDE_TOWERS else (), False
        self.loaded_blocks[model_id] = None
        any_lora =  loras !=None and model_id in loras
        if any_lora: 
            loras_model_data, loras_model_shortcuts = {}, {}
            current_model._loras_model_data = loras_model_data 
            current_model._loras_model_shortcuts = loras_model_shortcuts
        modules_to_be_compiled, modules_names_to_be_compiled = [], []
        # modules that the model loads only while they run (e.g. an LLM's token embedding and output head, used for a prompt but not to denoise)
        separate_blocks = tuple(getattr(current_model, "_offload_separate_blocks", ())) if current_budget > 0 else ()
        # modules computed on the CPU (runtime.lookup_on_cpu), whose weights never go to VRAM (e.g. a large token table looked up for a
        # prompt: a row gather, the same values): outside the blocks and the base, never pinned nor counted in budgets
        cpu_modules = _cpu_modules(current_model)
        for name in cpu_modules: # the modules that hold one report the execution device, as the model does, not their table's
            for depth in range(1, name.count(".") + 1):
                current_model.get_submodule(name.rsplit(".", depth)[0])._force_device = "cuda"
        cpu_roots = set(towers_modules) if model_id in cpu_models and current_budget > 0 else set()
        for submodule_name, submodule in current_model.named_modules():
            if submodule_name in expert_exclusions.get(model_id, ()):
                continue
            cpu_module = next((name for name in cpu_modules if submodule_name == name or submodule_name.startswith(name + ".")), None)
            if cpu_module is not None:
                if submodule_name == cpu_module:
                    self.hook_cpu_module(submodule)
                continue
            # create a fake 'accelerate' parameter so that the _execution_device property returns always "cuda" 
            # (it is queried in many pipelines even if offloading is not properly implemented)  
            if not hasattr(submodule, "_hf_hook"):
                setattr(submodule, "_hf_hook", HfHook())
            if current_budget > 0 and len(submodule_name) > 0:
                if cur_blocks_prefix != None:
                    if submodule_name.startswith(cur_blocks_prefix):
                        depth_prefix = cur_blocks_prefix.split(".")
                        depth_name = submodule_name.split(".")
                        level  =  depth_name[len(depth_prefix)-1]                        
                        pre , num = _extract_num_from_str(level)
                        if num != cur_blocks_seq and not (is_mod_seq and cur_blocks_seq>=0):
                            prev_blocks_name = cur_blocks_name
                            cur_blocks_name, in_tower =  cur_blocks_prefix + str(num), submodule in floors
                            # print(f"new block: {model_id}/{cur_blocks_name} - {submodule_name}")
                        cur_blocks_seq = num
                    else:
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq, is_mod_seq = None, None, None, -1, False

                if cur_blocks_prefix == None:
                    pre , num = _extract_num_from_str(submodule_name)
                    if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):  
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_seq, is_mod_seq = pre + ".", None, -1, isinstance(submodule, torch.nn.Sequential)
                    elif num >=0:
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_seq, is_mod_seq = pre, None, num, False
                        cur_blocks_name, in_tower = submodule_name, submodule in floors
                        # print(f"new block: {model_id}/{cur_blocks_name} - {submodule_name}")
            separate = next((name for name in separate_blocks if submodule_name == name or submodule_name.startswith(name + ".")), None)
            blocks_name, previous_name = (separate, None) if separate is not None else (None, None) if submodule in shared_modules or base_modules and in_base(submodule_name) \
                or floors and cur_blocks_name is not None and not in_tower else (cur_blocks_name, prev_blocks_name)
            top_submodule = len(submodule_name.split("."))==1
            offload_hooks = submodule._offload_hooks if hasattr(submodule, "_offload_hooks") else []
            assert top_submodule or len(offload_hooks) == 0, "custom offload hooks can only be set at the of the module"
            submodule_method_names = ["forward"] +  offload_hooks
            compile_me = getattr(submodule, "_compile_me", None)
            for submodule_method_name in submodule_method_names:
                if not hasattr(submodule, submodule_method_name ): continue
                if submodule_method_name == "forward" and any_lora and hasattr(submodule,"weight"):
                    submodule_method = self.hook_lora(submodule, current_model, model_id, loras_model_data, loras_model_shortcuts, submodule_name)                
                else:
                    submodule_method = getattr(submodule, submodule_method_name)
                if graphs is not None and submodule_method_name == "forward" and submodule in towers_set and not getattr(submodule, "_cuda_graph_segments", False):
                    submodule_method = graphs.wrap(submodule, submodule_method)
                if callable(submodule_method):
                    if top_submodule and blocks_name is None and any_lora and len(submodule._parameters):
                        pass
                    if top_submodule and blocks_name is None and not (any_lora and len(submodule._parameters)):
                        self.hook_change_module(submodule, current_model, model_id, submodule_name, submodule_method, submodule_method_name)
                    elif compilationInThisOne and submodule in towers_modules and not compile_me == False or compile_me == True:
                        self.hook_preload_blocks_for_compilation(submodule, model_id, blocks_name, context = submodule_name )
                        compile_item_name= ".".join(submodule_name.split(".")[:-1]) if submodule_name[-1].isdigit() else submodule_name
                        if compile_item_name not in modules_names_to_be_compiled: modules_names_to_be_compiled.append(compile_item_name)
                        modules_to_be_compiled.append(submodule)
                    else:
                        if compilationInThisOne: #and False
                            self.hook_check_load_into_GPU_if_needed(submodule, current_model, model_id, blocks_name, submodule_method, context = submodule_name )
                        else:
                            self.hook_check_load_into_GPU_if_needed_default(submodule, current_model, model_id, blocks_name, submodule_method, context = submodule_name, cpu_root = submodule in cpu_roots)

                    self.add_module_to_blocks(model_id, blocks_name, submodule, previous_name, submodule_name)


        loading_phase("Finalization", model_id)
        # compile main iterative modules stacks ("towers")
        if compilationInThisOne:
            if self.verboseLevel>=1:
                if len(modules_names_to_be_compiled)>0:
                    formated_compiled_names = [name + '.*' for name in modules_names_to_be_compiled]
                    print(f"Pytorch compilation of '{model_id}' is scheduled for these modules : {formated_compiled_names}.")
                else:
                    print(f"Pytorch compilation of model '{model_id}' is not yet supported.")

            for submodel in modules_to_be_compiled:
                submodel.forward= torch.compile(submodel.forward,  backend= "inductor", mode= compile_mode) # , fullgraph= True, mode= "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs",  
                    #dynamic=True,

        if towers_names:
            self._shared_tensors_to_base(model_id)
        if towers_names and BASE_SMALL_MODULE_SIZE:
            self._small_modules_to_base(model_id, towers_names)
        if cpu_roots:
            self._split_cpu_compute(model_id, towers_names)
        self.tune_preloading(model_id, current_budget, towers_names, covers)
        if self.preload_plans and current_budget > 0 and towers_names and self.async_transfers and not compilationInThisOne:
            (auto_models if self.auto_preload_requested else fixed_models).append(model_id)
        self.parameters_ref  = {} 


    if self.verboseLevel >=2:
        # consecutive blocks of a tower in one line while their sizes stay within 20% of the middle of their range
        groups = [] # [name prefix, first number, last number, sizes]
        for n, size in self.blocks_of_modules_sizes.items():
            pre, num = _extract_num_from_str(n) if "/" in n else (n, -1)
            group = groups[-1] if groups else None
            if group is not None and num >= 0 and group[0] == pre and num == group[2] + 1 and max(*group[3], size) <= 1.5 * min(*group[3], size):
                group[2] = num
                group[3].append(size)
            else:
                groups.append([pre, num, num, [size]])
        for pre, first, last, sizes in groups:
            low, high = f"{min(sizes)/ONE_MB:.1f}", f"{max(sizes)/ONE_MB:.1f}"
            count = f" ({low if low == high else low + '-' + high} MB x {len(sizes)})" if len(sizes) > 1 else ""
            print(f"Size of submodel '{pre if first < 0 else pre + str(first) + ('' if first == last else f'-{last}')}': {builtins.sum(sizes)/ONE_MB:.1f} MB{count}")


    if auto_models or fixed_models: # before pinning, which pins first the blocks that are not preloaded
        from .auto_preload import AutoPreload
        self.auto_preload = AutoPreload(self, auto_models, verboseLevel, auto_measures, fixed_models)
    for model_id, banks in expert_banks.items():
        if not banks:
            continue
        residency = experts.Residency(self, model_id, banks, expert_budgets[model_id], autoPreload)
        self.expert_residency[model_id] = residency
        def prepare_experts(module, args, model_id=model_id, residency=residency):
            self.ensure_model_loaded(model_id)
            # Dynamic block plans coordinate experts after their ordinary blocks.
            if self.auto_preload is None or model_id not in self.auto_preload.plans:
                residency.prepare()
        self.expert_hooks.append(self.models[model_id].register_forward_pre_hook(prepare_experts))
    cpu_pinning = {model_id: cpuComputePinning.get(model_id, 0) if isinstance(cpuComputePinning, dict) else cpuComputePinning or 0 for model_id in self.cpu_compute if model_id not in self.pool_models}
    for model_id, share in cpu_pinning.items():
        if share == "auto":
            cpu_pinning[model_id] = self._cpu_pinning_share(model_id)
    self._pin_models(listed, smartPinning, partialPinning, pinnedPEFTLora, perc_reserved_mem_max, max_reservable_memory, verboseLevel, loading_callback, cpu_pinning)
    if pinAllModels: # after the models' own pinning, which has priority for the reserved RAM
        for model_id, residency in self.expert_residency.items():
            residency.pin(lambda module, name: _pin_to_memory(module, name, perc_reserved_mem_max=perc_reserved_mem_max, verboseLevel=verboseLevel,
                                                              loading_callback=loading_callback))
    if self.auto_preload is not None:
        self.auto_preload.after_pinning() # the blocks left unpinned are the first to keep in VRAM
    if asyncTransfers:
        # once pinned: the ring goes if every swapped block got pinned, and a pinned model that ran out of reserved RAM gets one
        swapped_entries = [entry for entry in self.prev_blocks_names if entry.split("/", 1)[1] not in self.preloaded_blocks_per_model[entry.split("/", 1)[0]]]
        unpinned = [entry for entry in swapped_entries if self._block_items(entry)[2] is None]
        if not unpinned:
            self._release_ring()
        elif self.ring is None and smartPinning is None and builtins.any(listed[entry.split("/", 1)[0]] for entry in unpinned):
            self._create_ring(max(self._block_items(entry)[1][2] for entry in unpinned), max_reservable_memory, verboseLevel)
    for model_id in self.pool_models: # the plans of the blocks of the pinned pool, made now: while a model runs, each call waits for the GIL
        for entry in [entry for entry in self.blocks_of_modules if entry.startswith(model_id + "/")]:
            self._pool_plan(entry)
    for computed in self.cpu_compute.values(): # after pinning, which may give them other storage
        for parent, name, p, _, _ in computed:
            p._mm_cpu_compute = getattr(parent, name)._mm_cpu_compute = True
    if DROP_UNREFERENCED_PAGES: # what every offload object may still read: a view may serve several
        _drop_mapped_pages(_unreferenced_pages([span for manager in list(_managers) if manager.models is not None for t in manager._read_later()
                                                for span in _tensor_spans(t.data if isinstance(t, torch.nn.Parameter) else t)]))
    if verboseLevel >= 2: # weights in private RAM go to the page file under memory pressure; those mapped from checkpoint files are reread instead
        from .allocator import ram_debug
        for model_id, info in ram_debug.weight_backing(self.models).items():
            tqdm.write(ram_debug.weight_backing_text(model_id, info))
    if loading_callback is not None: loading_callback.report("Finalization", phase_total, phase_total, "")
    torch.set_default_device('cuda')
    torch.cuda.empty_cache()
    gc.collect()         

    return self


all.__doc__ = _configure.__doc__


def profile(pipe_or_dict_of_modules, profile_no: profile_type =  profile_type.VerylowRAM_LowVRAM, verboseLevel = -1, **overrideKwargs):
    """Apply a configuration profile that depends on your hardware:
    pipe_or_dict_of_modules : the pipeline object or a dictionary of modules of the model
    profile_name : num of the profile:
        AutoRAM_AutoVRAM (=0): profile 4 without RAM pinned for good: the parts of the models streamed at each step are pinned while
            their model runs, into reserved RAM shared by all the models (pinPool), and language model decoders compute the layers left
            in RAM on the CPU when it is faster (cpuCompute "auto"). Add autoPreload=True for the automatic VRAM preload
        HighRAM_HighVRAM (=1): profile 2 with automatic VRAM preload. All models kept in reserved RAM; keep as many blocks as fit
            beside the measured workload in VRAM, streaming the rest
        HighRAM_LowVRAM (=2): all models kept in reserved RAM, sent to the GPU part by part. Runs models larger than the VRAM, leaves VRAM
            for long videos or large images and switches models fast, needs a lot of RAM
        LowRAM_HighVRAM (=3): profile 4 with automatic VRAM preload. Only the main model ('transformer') kept in reserved RAM;
            keep as many blocks as fit beside the measured workload in VRAM, streaming the rest
        LowRAM_LowVRAM (=4): only the main model ('transformer') kept in reserved RAM, sent to the GPU part by part. The most versatile:
            runs models larger than the VRAM and leaves VRAM for long videos or large images
        VerylowRAM_LowVRAM (=5): no reserved RAM (only a small staging ring with smartPinning), all models sent to the GPU part by
            part. For PCs short of RAM and VRAM, slower with short steps such as images
        Every profile quantizes the main model to 8 bits by default
    overrideKwargs: every parameter accepted by Offload.All can be added here to override the profile choice
        For instance set quantizeTransformer = False to disable transformer quantization which is by default in every profile
    """      

    message = f" Applying MMGP Memory Profile {int(profile_no)} - {profile_type.tostr(profile_no)} (v{__version__}) "
    tqdm.write(f"\033[1m\033[95m{message:*^100}\033[0m")

    verboseLevel = _compute_verbose_level(verboseLevel)

    modules = pipe_or_dict_of_modules
    ignored_models = getattr(modules, "_mmgp_ignore_models", ())

    if hasattr(modules, "components"):
        modules= modules.components 

    modules = {k: _remove_model_wrapper(v) for k, v in modules.items() if isinstance(v, torch.nn.Module) and k not in ignored_models}
    module_names = {k: _get_module_name(v) for k, v in modules.items() }

    default_extraModelsToQuantize = []
    quantizeTransformer = True
    
    models_to_scan = ("text_encoder", "text_encoder_2")
    candidates_to_quantize = ("t5", "llama", "llm")
    loading_callback = overrideKwargs.get("loading_callback")
    for model_id  in models_to_scan:
        if loading_callback is not None: loading_callback.check_abort()
        if model_id in module_names: 
            name = module_names[model_id]
            for candidate in candidates_to_quantize:
                if candidate in name:
                    default_extraModelsToQuantize.append(model_id)
                    break


    # transformer (video or image generator) should be as small as possible not to occupy space that could be used by actual image data
    # on the other hand the text encoder should be quite large (as long as it fits in 10 GB of VRAM) to reduce sequence offloading

    budgets = {}
    if "transformer" in modules:
        budgets["transformer"] = 1200    

    extraModelsToQuantize = None
    asyncTransfers = True
    prefetchWindow = None

    if profile_no == profile_type.AutoRAM_AutoVRAM:
        pinnedMemory= False
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets["*"] = 3000
    elif profile_no == profile_type.HighRAM_HighVRAM:
        pinnedMemory= True
        budgets["*"] = 3000
        # info = "You have chosen a profile that may require 48 GB of RAM and up to 24 GB of VRAM on some applications."
    elif profile_no == profile_type.HighRAM_LowVRAM:
        pinnedMemory= True
        budgets["*"] =  3000
        # info = "You have chosen a profile that may require 48 GB of RAM and up to 12 GB of VRAM on some applications."
    elif profile_no == profile_type.LowRAM_HighVRAM:
        pinnedMemory= "transformer"
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets["*"] = 3000
        # info = "You have chosen a Medium speed profile that may require 32 GB of RAM and up to 24 GB of VRAM on some applications."
    elif profile_no == profile_type.LowRAM_LowVRAM:
        pinnedMemory= "transformer"
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets["*"] =  3000
        # info = "You have chosen a profile that usually may require 32 GB of RAM and up to 12 GB of VRAM on some applications."
    elif profile_no == profile_type.VerylowRAM_LowVRAM:
        pinnedMemory= False
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets["*"] =  3000
        if "transformer" in modules:
            budgets["transformer"] = 400    
        #asyncTransfers = False
        # info = "You have chosen the slowest profile that may require 24 GB of RAM and up to 10 GB of VRAM on some applications."
    else:
        raise Exception("Unknown profile")
    # info += " Actual requirements may varry depending on the application or on the tuning done to the profile."
    info =""    
    if budgets != None and len(budgets) == 0:
        budgets = None

    CrLf = '\r\n'
    kwargs = { "pinnedMemory": pinnedMemory,  "extraModelsToQuantize" : extraModelsToQuantize, "budgets": budgets, "asyncTransfers" : asyncTransfers, "quantizeTransformer": quantizeTransformer, "prefetchWindow": prefetchWindow }
    kwargs["preloadExperts"] = "budgets" in overrideKwargs
    if profile_no in (profile_type.HighRAM_HighVRAM, profile_type.LowRAM_HighVRAM):
        kwargs["autoPreload"] = True
    if profile_no == profile_type.AutoRAM_AutoVRAM:
        kwargs.update(pinPool=True, cpuCompute="auto")
        # a text encoder or a VAE runs once or twice per generation: entering the pinned pool would cost as much as staging its blocks
        # through the ring, and push out the entries of the models that run many passes (a transformer, a language model), which then copy
        # them again. A model sets _mm_pinned_pool itself to choose (e.g. an autoregressive language model registered as a text encoder),
        # also at run time
        for model_id in ONCE_PER_GENERATION_MODELS:
            if model_id in modules and getattr(modules[model_id], "_mm_pinned_pool", None) is None:
                modules[model_id]._mm_pinned_pool = False

    if verboseLevel>=2:
        info = info  + f"Profile '{profile_type.tostr(profile_no)}' sets the following options:" #CrLf 
        for k,v in kwargs.items():
            if k in overrideKwargs: 
                info = info + CrLf + f"- '{k}':  '{kwargs[k]}' overriden with value '{overrideKwargs[k]}'"
            else:
                info = info + CrLf + f"- '{k}':  '{kwargs[k]}'"

    for k,v in overrideKwargs.items():
        kwargs[k] = overrideKwargs[k]

    if info:
        print(info)

    return all(pipe_or_dict_of_modules, verboseLevel = verboseLevel, **kwargs)
