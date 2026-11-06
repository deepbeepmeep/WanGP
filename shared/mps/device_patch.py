"""Apple Silicon MPS compatibility patch for Wan2GP.

Import EARLY at startup — BEFORE any mmgp import. Patches torch.cuda functions
to redirect to MPS, disables torch.compile (CUDA-less PyTorch build), and adds
stub attributes for CUDA-only code paths.
"""
import os
import sys
import types

# CRITICAL: Disable torch.compile / dynamo before torch is imported.
# PyTorch 2.11 on macOS is built with USE_CUDA=OFF. torch.compile traces into
# functions like torch.manual_seed, follows the CUDA call chain, and hits
# C++-level "not linked with cuda" errors that Python patches cannot intercept.
os.environ.setdefault('TORCH_COMPILE', '0')
os.environ.setdefault('TORCHINDUCTOR', '0')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

def apply_mps_patch():
    """Patch torch.cuda functions for MPS compatibility."""
    import torch as _torch

    chip_name = _get_chip_name()
    system_ram_gb = _get_system_memory_gb()
    total_memory_bytes = int(system_ram_gb * 1024 ** 3)

    if 'M1' in chip_name or 'M2' in chip_name:
        dev_cap = (7, 0)
        bfloat16_supported = False
    else:
        dev_cap = (11, 0)
        bfloat16_supported = True

    print(f"[MPS Patch] Detected: {chip_name}, {system_ram_gb:.0f}GB RAM")
    print(f"[MPS Patch] Device capability: {dev_cap}, BF16: {bfloat16_supported}")

    # Dummy objects
    _dummy_stream = types.SimpleNamespace(
        synchronize=_torch.mps.synchronize,
        wait_stream=lambda *a, **kw: None,
        query=lambda: True,
        priority=0,
    )

    class _CudaDeviceProperties:
        def __init__(self):
            self.total_memory = total_memory_bytes
            self.name = chip_name
            self.major = dev_cap[0]
            self.minor = dev_cap[1]
            _torch.cuda._device_props_cache = self
        multi_processor_count = 0
        warp_size = 32
        L2_cache_size = 32 << 20  # read by shared/kernels to size MLP chunks; no CUDA L2 on MPS

    class _DummyEvent:
        def __init__(self, *a, **kw): pass
        def record(self, *a, **kw): pass
        def elapsed_time(self, *a, **kw): return 0.0
        def synchronize(self, *a, **kw): pass
        def query(self): return True

    class _DummyDeviceContext:
        def __init__(self, device=None): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class _DummyStreamContext:
        def __init__(self, s): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class _DummyGraph:
        replay = lambda s, *a, **kw: None
        capture_begin = lambda s, *a, **kw: None
        capture_end = lambda s, *a, **kw: None

    # AMP stub
    class _MpsAutocast:
        def __init__(self, enabled=True, dtype=None, device_type='mps', cache_enabled=None):
            self._autocast = _torch.autocast('mps', enabled=enabled, dtype=dtype)
        def __enter__(self): return self._autocast.__enter__()
        def __exit__(self, *a): return self._autocast.__exit__(*a)

    class _autocast_mode_mod:
        autocast = _MpsAutocast

    class _amp_common:
        @staticmethod
        def amp_definitely_not_available():
            return True

    class _PatchedAMP:
        autocast = _MpsAutocast
        autocast_mode = _autocast_mode_mod
        common = _amp_common
        class GradScaler:
            def __init__(self, *a, **kw): pass
            def step(self, *a, **kw): return a[0] if a else None
            def update(self, *a, **kw): pass
            def unscale_(self, *a, **kw): pass
            def get_scale(self): return 1.0
            def state_dict(self): return {}
            def load_state_dict(self, *a): pass

    _cuda = _torch.cuda

    # Core function patches
    _cuda.is_available = lambda: False
    _cuda._is_compiled = lambda: False
    _cuda.empty_cache = _torch.mps.empty_cache
    _cuda.synchronize = lambda device=None: _torch.mps.synchronize()  # CUDA's takes a device, MPS's takes none
    _cuda.get_device_capability = lambda device=None: dev_cap
    _cuda.manual_seed_all = lambda seed: None
    _cuda.manual_seed = lambda device_or_seed, seed=None: None
    _cuda.current_stream = lambda device=None: _dummy_stream
    _cuda.get_device_properties = lambda device=None: _CudaDeviceProperties()
    _cuda._CudaDeviceProperties = _CudaDeviceProperties
    _cuda.default_stream = lambda device=None: _dummy_stream
    _cuda.set_device = lambda device: None
    _cuda.current_device = lambda: 0
    _cuda.device_count = lambda: 1
    _cuda.ipc_collect = lambda: None
    _cuda.device = _DummyDeviceContext
    # mmgp/pinned.py locks RAM with cudaHostRegister; Apple Silicon memory is unified, so plain RAM needs no locking
    _dummy_cudart = types.SimpleNamespace(
        cudaHostRegister=lambda *a, **kw: 0,
        cudaHostUnregister=lambda *a, **kw: 0,
        cudaGetErrorString=lambda error: f"CUDA error {error} (MPS)",
    )
    _cuda.cudart = lambda: _dummy_cudart

    class _PatchedStream:
        priority = 0
        def __init__(self, *a, **kw): pass
        def synchronize(self, *a, **kw): pass
        def wait_stream(self, *a, **kw): pass
        def query(self): return True

    _cuda.Stream = _PatchedStream
    _cuda.stream = lambda s: _DummyStreamContext(s)
    _cuda.Event = _DummyEvent
    _cuda.is_bf16_supported = lambda device=None: bfloat16_supported
    _cuda.bfloat16_supported = lambda device=None: bfloat16_supported
    _cuda.is_current_stream_capturing = lambda: False
    _cuda.graph = lambda *a, **kw: _DummyGraph()
    _cuda.CUDAGraph = _DummyGraph
    _cuda.graph_pool_handle = lambda: None
    _cuda.mem_get_info = lambda device=None: (total_memory_bytes, total_memory_bytes)
    _cuda.memory_allocated = lambda device=None: 0
    _cuda.memory_reserved = lambda device=None: 0
    _cuda.max_memory_allocated = lambda device=None: 0
    _cuda.max_memory_reserved = lambda device=None: 0
    _cuda.reset_peak_memory_stats = lambda device=None: None
    _cuda.memory_stats = lambda device=None: {}
    try:
        import torch.cuda.amp as _cuda_amp
        _cuda_amp.autocast = _MpsAutocast
        _cuda_amp.GradScaler = _PatchedAMP.GradScaler
        _cuda.amp = _cuda_amp
    except Exception:
        _cuda.amp = _PatchedAMP
    _cuda.is_initialized = lambda: True
    _cuda._lazy_init = lambda: None

    # CRITICAL: Patch torch.manual_seed to avoid internal CUDA calls.
    # torch.manual_seed calls torch.cuda.manual_seed_all internally. Even with
    # cuda.manual_seed_all patched to no-op, torch.compile tracing into manual_seed
    # can trigger C++-level CUDA failures. Replace it with MPS-only version.
    _orig_manual_seed = _torch.manual_seed
    def _mps_manual_seed(seed):
        seed = int(seed)
        _orig_manual_seed(seed)  # CPU seed
        _torch.mps.manual_seed(seed)  # MPS seed
        return _torch._C.Generator()
    _torch.manual_seed = _mps_manual_seed

    # CRITICAL: Replace torch.compile with a true no-op.
    # PyTorch 2.11 on macOS is built with USE_CUDA=OFF. Even the 'eager' backend
    # involves dynamo tracing which can trigger C++-level CUDA failures.
    # Simply return the original function unchanged.
    def _patched_compile(fn=None, *args, **kwargs):
        if fn is not None:
            return fn
        def decorator(f):
            return f
        return decorator
    _torch.compile = _patched_compile

    # CRITICAL: Patch torch.autocast to redirect 'cuda' -> 'mps'
    # Code uses torch.autocast('cuda', ...) or torch.autocast(device_type='cuda', ...)
    _orig_autocast = _torch.autocast
    def _patched_autocast(device_type=None, *args, **kwargs):
        if device_type == 'cuda':
            device_type = 'mps'
        if kwargs.get('device_type') == 'cuda':
            kwargs['device_type'] = 'mps'
        # Handle torch.cuda.amp.autocast which calls with device_type=None initially
        if device_type is None and 'device_type' not in kwargs:
            device_type = 'mps'
        return _orig_autocast(device_type, *args, **kwargs)
    _torch.autocast = _patched_autocast
    # Also patch torch.amp.autocast
    _torch.amp.autocast = _patched_autocast

    # Disable torch._dynamo entirely to avoid any traced CUDA calls
    try:
        _torch._dynamo.config.suppress_errors = True
        _torch._dynamo.config.cache_size_limit = 128
    except Exception:
        pass

    # Tensor and Module .cuda() redirects
    def _patched_tensor_cuda(self, device=None, *args, **kwargs):
        return self.to("mps")
    _torch.Tensor.cuda = _patched_tensor_cuda

    def _patched_module_cuda(self, device=None):
        return self.to("mps")
    _torch.nn.Module.cuda = _patched_module_cuda

    def _replace_cuda_device(val):
        if isinstance(val, str) and val.startswith("cuda"):
            return "mps"
        if isinstance(val, _torch.device) and val.type == "cuda":
            return _torch.device("mps")
        return val

    def _replace_map_location(map_location):
        if isinstance(map_location, dict):
            return {key: _replace_cuda_device(value) for key, value in map_location.items()}
        return _replace_cuda_device(map_location)

    # CRITICAL: Patch Tensor.to and Module.to to intercept cuda device strings.
    # This catches code that passes device="cuda" as a string to .to() calls.
    _orig_tensor_to = _torch.Tensor.to
    def _patched_tensor_to(self, *args, **kwargs):
        # Handle positional args: .to("cuda"), .to(device), .to(dtype, device)
        new_args = [_replace_cuda_device(a) for a in args]
        # Handle keyword device arg
        if "device" in kwargs:
            kwargs["device"] = _replace_cuda_device(kwargs["device"])
        return _orig_tensor_to(self, *new_args, **kwargs)
    _torch.Tensor.to = _patched_tensor_to

    _orig_module_to = _torch.nn.Module.to
    def _patched_module_to(self, *args, **kwargs):
        new_args = [_replace_cuda_device(a) for a in args]
        if "device" in kwargs:
            kwargs["device"] = _replace_cuda_device(kwargs["device"])
        return _orig_module_to(self, *new_args, **kwargs)
    _torch.nn.Module.to = _patched_module_to

    def _patched_pin_memory(self, *args, **kwargs):
        return self
    _torch.Tensor.pin_memory = _patched_pin_memory

    _orig_load = _torch.load
    def _patched_load(*args, **kwargs):
        if "map_location" in kwargs:
            kwargs["map_location"] = _replace_map_location(kwargs["map_location"])
        elif len(args) >= 2:
            args = (args[0], _replace_map_location(args[1]), *args[2:])
        return _orig_load(*args, **kwargs)
    _torch.load = _patched_load

    # Generator patch
    _Gen = _torch.Generator
    class _PatchedGen(_Gen):
        def __new__(cls, device=None):
            device = _replace_cuda_device(device)
            if device:
                return super().__new__(cls, device=device)
            return super().__new__(cls)
    _torch.Generator = _PatchedGen

    # Tensor creation patch — redirect cuda->mps, fix pin_memory bug
    for fn_name in ['zeros', 'ones', 'randn', 'rand', 'tensor', 'arange',
                     'linspace', 'empty', 'full', 'eye', 'zeros_like', 'ones_like',
                     'randn_like', 'rand_like', 'empty_like', 'full_like',
                     'as_tensor', 'from_numpy']:
        if hasattr(_torch, fn_name):
            orig = getattr(_torch, fn_name)
            def make_patcher(o):
                def patched(*args, **kwargs):
                    dev = kwargs.get('device')
                    new_dev = _replace_cuda_device(dev)
                    if new_dev is not dev:
                        kwargs['device'] = new_dev
                    if 'pin_memory' in kwargs:
                        kwargs.pop('pin_memory', None)
                    return o(*args, **kwargs)
                return patched
            setattr(_torch, fn_name, make_patcher(orig))

    print(f"[MPS Patch] Applied successfully")
    print(f"[MPS Patch] BF16 supported: {bfloat16_supported}")
    print(f"[MPS Patch] Available system RAM: {system_ram_gb:.0f}GB")

    # Fix: Some Wan model loading paths call .weight on an nn.Parameter,
    # which is a Tensor subclass, not a Module. On MPS this fails because
    # nn.Parameter doesn't have a .weight attribute. Duck-type it to return self.
    # Reference: https://github.com/deepbeepmeep/Wan2GP/pull/1750#issuecomment-4387455446
    if not hasattr(_torch.nn.Parameter, "weight"):
        _torch.nn.Parameter.weight = property(lambda self: self)

    return True  # signal success

def _get_chip_name():
    try:
        import subprocess
        out = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'], encoding='utf-8', stderr=subprocess.DEVNULL)
        for line in out.split('\n'):
            if 'Chip' in line:
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return "Unknown Apple Silicon"

def _get_system_memory_gb():
    try:
        import subprocess
        out = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], encoding='utf-8').strip()
        return int(out) / (1024 ** 3)
    except Exception:
        return 16.0

# Auto-apply on import if on macOS with MPS
import torch as _torch
_is_mps = sys.platform == 'darwin' and hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available()

# Patch torch._C missing C extension functions
_C = _torch._C
if not hasattr(_C, '_cuda_getDefaultStream'):
    def _cuda_getDefaultStream_stub(device_index=0):
        return (0, device_index, 0)
    _C._cuda_getDefaultStream = _cuda_getDefaultStream_stub

# Add missing torch.mps attributes
if not hasattr(_torch.mps, 'current_device'):
    _torch.mps.current_device = lambda: 0
if not hasattr(_torch.mps, 'device_count'):
    _torch.mps.device_count = lambda: 1
if not hasattr(_torch.mps, 'set_device'):
    _torch.mps.set_device = lambda device: None

# Force SDPA math backend on MPS to avoid Metal command buffer double-commit crash
# Reference: [IOGPUMetalCommandBuffer validate]:214: failed assertion `commit an already committed command buffer'
#
# Root cause: MPS fallback ops (CPU fallback from PYTORCH_ENABLE_MPS_FALLBACK=1)
# corrupt Metal command buffers when mixed with native MPS SDPA. This affects:
#   - Wan 2.2 5B (quanto mbf16 quantization)
#   - Wan 2.1 1.3B (standard safetensors, but ops still fallback)
#   - Any model where CPU-fallback ops precede an SDPA call
#
# Fix strategy (defense in depth):
#   1. Synchronize MPS before SDPA to flush pending fallback ops
#   2. Force MATH backend (avoids MPS-native SDPA bugs on some macOS versions)
#   3. If SDPA fails with Metal error, fall back to manual attention (matmul + softmax)
#   4. Periodic empty_cache to prevent memory fragmentation
if _is_mps:
    _orig_sdpa = _torch.nn.functional.scaled_dot_product_attention
    _sdpa_call_count = [0]

    def _manual_sdpa_fallback(query, key, value, attn_mask=None, is_causal=False, scale=None):
        """Manual attention fallback: matmul + softmax, no Metal SDPA."""
        # query: (B, H, L, D) or (B, L, H, D) after sdpa_kernel wrap
        L = query.size(-2)
        D = query.size(-1)
        if scale is None:
            scale = D ** -0.5

        attn_weights = _torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if is_causal:
            causal_mask = _torch.triu(
                _torch.ones(L, L, device=query.device, dtype=_torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = _torch.nn.functional.softmax(attn_weights, dim=-1)
        return _torch.matmul(attn_weights, value)

    def _patched_sdpa(*args, **kwargs):
        # Flush pending MPS fallback ops before SDPA
        _torch.mps.synchronize()

        # Periodic cache cleanup every 256 SDPA calls
        _sdpa_call_count[0] += 1
        if _sdpa_call_count[0] % 256 == 0:
            _torch.mps.empty_cache()

        try:
            with _torch.nn.attention.sdpa_kernel([_torch.nn.attention.SDPBackend.MATH]):
                return _orig_sdpa(*args, **kwargs)
        except Exception:
            # Metal command buffer corruption caught — fall back to manual attention
            # This handles cases where synchronize isn't sufficient (e.g. macOS 26.x bugs)
            return _manual_sdpa_fallback(*args, **kwargs)

    _torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

if _is_mps:
    try:
        apply_mps_patch()
    except Exception as e:
        print(f"[MPS Patch] Failed to apply patch: {e}")
        import traceback
        traceback.print_exc()
