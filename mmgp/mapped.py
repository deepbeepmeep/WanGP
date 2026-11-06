# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Parameters kept in page-locked RAM and read by the GPU in place (zero-copy, runtime.map_in_ram).

The RAM of a mapped tensor is registered in the GPU's address space: CUDA views of it go to any kernel, CUDA graphs included, and each read
crosses the bus (about a microsecond, PCIe bandwidth). It suits tensors read by rows that the GPU picks, such as a token table, whose lookup
reads a few KB of a table of hundreds of MB, never weights read whole at each call. Their RAM is reserved RAM: charged to MMGP's count of
locked bytes when it is registered, given back when the last view of it dies.
"""
import ctypes
import mmap
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

from . import offload, pinned

_REGISTER_MAPPED = 2  # cudaHostRegisterMapped
_driver = None
_reported = set()


def supported():
    """Mapped RAM needs CUDA's driver (HIP's equivalent is not validated)."""
    return torch.cuda.is_available() and torch.version.hip is None


def is_mapped(tensor):
    """True for a parameter that MMGP keeps in mapped RAM."""
    return getattr(tensor, "_mm_mapped", False)


def _device_pointer(host_pointer):
    # the device address of registered RAM (on Windows it is not its host address)
    global _driver
    if _driver is None:
        _driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        _driver.cuMemHostGetDevicePointer_v2.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint)
    pointer = ctypes.c_uint64()
    error = _driver.cuMemHostGetDevicePointer_v2(ctypes.byref(pointer), ctypes.c_void_p(host_pointer), 0)
    if error:
        raise RuntimeError(f"cuMemHostGetDevicePointer returned {error}")
    return pointer.value


class _Mapping:
    """The CUDA array interface of mapped RAM: its CUDA views hold it, and with it the RAM and its registration."""

    def __init__(self, ram, owner, nbytes, device_pointer):
        self.ram, self.owner = ram, owner
        self.__cuda_array_interface__ = dict(shape=(nbytes,), typestr="|u1", data=(device_pointer, False), version=3)


def empty(nbytes, device, kind="mapped"):
    """(CPU view, CUDA view) of nbytes of mapped RAM of its own, charged to MMGP's locked bytes until both views are gone: the host and GPU kernels share it, CUDA graphs included. On Windows their
    addresses differ. Raises MemoryError if the RAM cannot be locked."""
    size = -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE
    ram, owner = pinned.block(size)
    owner.registration = pinned.lock(owner.base, ram.data_ptr(), size, kind, _REGISTER_MAPPED)
    if owner.registration is None:
        raise MemoryError(f"Unable to lock {size} bytes of mapped RAM")
    return ram[:nbytes], torch.as_tensor(_Mapping(ram, owner, nbytes, _device_pointer(ram.data_ptr())), device=device)


def _mapped_copy(tensor, device, executor):
    """A CUDA view of a copy of tensor in mapped RAM, whole pages of its own (registration rounds to pages)."""
    nbytes = tensor.numel() * tensor.element_size()
    size = -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE
    ram, owner = pinned.block(size)
    ram[:nbytes].view(tensor.dtype).view(tensor.shape).copy_(tensor.as_subclass(torch.Tensor))
    owner.registration = pinned.Registration(owner.base, ram.data_ptr(), size, device.index, flags=_REGISTER_MAPPED, kind="mapped")
    error = executor.submit(owner.registration.register).result()  # a failure leaves no CUDA error pending on this thread
    if int(error) != 0:
        raise MemoryError(f"cudaHostRegister returned {error}")
    return torch.as_tensor(_Mapping(ram, owner, nbytes, _device_pointer(ram.data_ptr())), device=device).view(tensor.dtype).view(tensor.shape)


def _tensors(parameter):
    subtensors = offload._get_quantized_subtensors(parameter)
    return subtensors if subtensors else [(None, parameter.data)]


def map_parameters(models, max_reservable_memory, verboseLevel=1):
    """offload.profile's pass over its models, before their own: the parameters of the modules marked by runtime.map_in_ram, and those of any
    of these models sharing their tensors (an alias of a table), move to mapped RAM, within the reserved RAM left. Mapped parameters stay
    the modules' parameters, outside the blocks, the base and the budgets of MMGP."""
    marked = {}
    for model_id, model in models.items():
        for name, module in model.named_modules():
            if getattr(module, "_mm_mapped", False):
                marked.setdefault(model_id, []).append(name)
                for parameter in module.parameters():
                    if not is_mapped(parameter):
                        marked.setdefault(None, {})[id(parameter)] = parameter
    parameters = list(marked.pop(None, {}).values())
    if not parameters:
        return
    names = ", ".join(f"'{model_id}': {', '.join(names)}" for model_id, names in marked.items())
    if not supported():
        if names not in _reported:
            _reported.add(names)
            tqdm.write(f"[MMGP] RAM mapping needs CUDA: {names} keep their usual placement.")
        return
    sources = {tensor.data_ptr(): tensor.numel() * tensor.element_size() for parameter in parameters for _, tensor in _tensors(parameter)}
    nbytes = sum(-(-size // mmap.PAGESIZE) * mmap.PAGESIZE for size in sources.values())
    if not pinned.fits(nbytes, max_reservable_memory):
        tqdm.write(f"[MMGP] Not enough reserved RAM left to map {names} ({nbytes / offload.ONE_MB:.0f} MB): they keep their usual placement.")
        return
    # the parameters of the other modules sharing these tensors (e.g. an MTP drafter's alias of a table) follow them
    seen = {id(parameter) for parameter in parameters}
    for model in models.values():
        for parameter in model.parameters():
            if id(parameter) not in seen and not is_mapped(parameter) and any(tensor.data_ptr() in sources for _, tensor in _tensors(parameter)):
                seen.add(id(parameter))
                parameters.append(parameter)
    device = torch.device("cuda", torch.cuda.current_device())
    views = {}  # (data pointer, bytes) of a source tensor -> its mapped view
    targets = set()  # data pointers of the mapped views: a tensor shared by several parameters is swapped once

    def mapped(tensor):
        key = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
        view = views.get(key)
        if view is None:
            view = views[key] = _mapped_copy(tensor, device, executor)
            targets.add(view.data_ptr())
        return view.view(view.dtype)  # a view of its own: the qtypes swap the tensors they are given

    try:
        with ThreadPoolExecutor(max_workers=1, initializer=torch.cuda.set_device, initargs=(device.index,)) as executor:
            for parameter in parameters:
                subtensors = offload._get_quantized_subtensors(parameter)
                if subtensors:
                    swapped = {name: mapped(tensor) for name, tensor in subtensors if tensor.data_ptr() not in targets}
                    if swapped:
                        offload._set_quantized_subtensors(parameter, swapped)
                elif parameter.data.data_ptr() not in targets:
                    parameter.data = mapped(parameter.data)
                parameter._mm_mapped = True
    except (MemoryError, RuntimeError) as error:
        tqdm.write(f"[MMGP] Unable to map {names} in reserved RAM ({error}): the parameters not mapped keep their usual placement.")
        return
    if verboseLevel >= 1:
        tqdm.write(f"[MMGP] {names} mapped in reserved RAM ({nbytes / offload.ONE_MB:.0f} MB): the GPU reads their rows in place, out of VRAM.")
