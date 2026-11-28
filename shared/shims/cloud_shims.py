# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Lightweight shims for Wan2GP Cloud Client mode.

When running Wan2GP in Cloud Client mode (--mode cloud or --cloud), heavy local dependencies
like PyTorch CUDA, Triton, FlashAttention, and heavy ML frameworks are NOT required on the client machine.
This module installs a universal metapath finder and minimal stubs into sys.modules if they are not
already installed, allowing the Wan2GP Gradio UI, FastAPI endpoints, and cloud dispatchers to run with
only requirements-cloud.txt.
"""
from __future__ import annotations

from enum import Enum
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
import os
import sys
import types
from typing import Any, Callable, Dict, Optional


class _DummyContextManager:
    """No-op context manager and decorator for torch.no_grad, autocast, etc."""
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    def __call__(self, fn_or_cls=None, *args, **kwargs):
        if fn_or_cls is None or not callable(fn_or_cls):
            return self
        return fn_or_cls


class _DummyDevice:
    """Mock torch.device."""
    def __init__(self, type_str="cpu", index=None):
        self.type = str(type_str).split(":")[0]
        self.index = index
    def __str__(self):
        return f"{self.type}:{self.index}" if self.index is not None else self.type
    def __repr__(self):
        return f"device(type='{self.type}')"


class _DummyTensor:
    """NumPy-backed mock torch.Tensor for import-time metadata math in cloud client mode."""
    def __init__(self, data=None, dtype=None, device=None):
        import numpy as np
        if isinstance(data, _DummyTensor):
            data = data._array
        if data is None:
            self._array = np.array([], dtype=np.float32)
        else:
            try:
                self._array = np.asarray(data)
            except Exception:
                self._array = np.array([], dtype=np.float32)
        self.data = self._array
        self.dtype = dtype or str(self._array.dtype)
        self.device = _DummyDevice("cpu")

    @property
    def shape(self):
        return self._array.shape

    def __array__(self, dtype=None):
        return self._array.astype(dtype) if dtype is not None else self._array

    def _wrap(self, value):
        return _DummyTensor(value)

    def _binary(self, other, op):
        other = other._array if isinstance(other, _DummyTensor) else other
        return self._wrap(op(self._array, other))

    def __add__(self, other): return self._binary(other, lambda a, b: a + b)
    def __radd__(self, other): return self._binary(other, lambda a, b: b + a)
    def __sub__(self, other): return self._binary(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._binary(other, lambda a, b: b - a)
    def __mul__(self, other): return self._binary(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._binary(other, lambda a, b: b * a)
    def __truediv__(self, other): return self._binary(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._binary(other, lambda a, b: b / a)
    def __floordiv__(self, other): return self._binary(other, lambda a, b: a // b)
    def __mod__(self, other): return self._binary(other, lambda a, b: a % b)
    def __pow__(self, other): return self._binary(other, lambda a, b: a ** b)
    def __neg__(self): return self._wrap(-self._array)
    def __matmul__(self, other): return self._binary(other, lambda a, b: a @ b)
    def __getitem__(self, key): return self._wrap(self._array[key])
    def __len__(self): return len(self._array)
    def __iter__(self): return iter(self._array)
    def __float__(self): return float(self.item())
    def __int__(self): return int(self.item())

    def cpu(self): return self
    def cuda(self, device=None): return self
    def to(self, *args, **kwargs): return self
    def detach(self): return self
    def clone(self): return self._wrap(self._array.copy())
    def contiguous(self): return self
    def numpy(self): return self._array
    def tolist(self): return self._array.tolist()
    def item(self):
        if self._array.size == 1:
            return self._array.reshape(-1)[0].item()
        return 0.0
    def numel(self): return int(self._array.size)
    def dim(self): return int(self._array.ndim)
    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]
    def squeeze(self, dim=None): return self._wrap(self._array.squeeze(dim))
    def unsqueeze(self, dim): return self._wrap(__import__("numpy").expand_dims(self._array, dim))
    def reshape(self, *shape): return self._wrap(self._array.reshape(*shape))
    def view(self, *shape): return self.reshape(*shape)
    def flatten(self, start_dim=0, end_dim=-1): return self._wrap(self._array.flatten())
    def float(self): return self
    def int(self): return self
    def long(self): return self
    def half(self): return self
    def bfloat16(self): return self
    def bool(self): return self
    def type(self, *args, **kwargs): return self
    def __repr__(self):
        return f"DummyTensor(shape={self.shape}, dtype={self.dtype})"


class MockMeta(type):
    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: cls()
    def __getitem__(cls, item: Any) -> Any:
        return 0
    def __int__(cls) -> int:
        return 0


class MockClass(metaclass=MockMeta):
    """Universal base class for dynamically mocked classes."""
    def __init__(self, *args, **kwargs):
        pass
    def __init_subclass__(cls, **kwargs):
        pass
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None
    @classmethod
    def register(cls, *args, **kwargs):
        pass
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls()
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class AutoConfigMock:
    """Mock transformers.AutoConfig supporting registration and lookup."""
    _registry: Dict[str, Any] = {}

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in cls._registry:
            return cls._registry[model_type]
        raise ValueError(f"Unrecognized model type {model_type}")

    @classmethod
    def register(cls, model_type: str, config_class: Any, *args, **kwargs):
        cls._registry[model_type] = config_class


class ConfigMixin:
    """Mock diffusers.configuration_utils.ConfigMixin."""
    pass


class SchedulerMixin:
    """Mock diffusers.schedulers.scheduling_utils.SchedulerMixin."""
    pass


class KarrasDiffusionSchedulers(Enum):
    """Mock diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers."""
    DPMSolverMultistep = 1
    Euler = 2
    EulerAncestral = 3


class MockModule(types.ModuleType):
    """Dynamic mock module that safely creates missing attributes and submodules."""
    def __getattr__(self, name: str) -> Any:
        if name == "__file__":
            return f"/mock/{self.__name__.replace('.', '/')}.py"
        if name == "__path__":
            return []
        if name == "__version__":
            return "3.0.0"
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name == "AutoConfig":
            return AutoConfigMock
        if name == "init_empty_weights":
            return _DummyContextManager
        if name == "qtypes":
            return {}
        if name == "fix_text":
            return lambda text: text
        if name[0].isupper():
            cls = type(name, (MockClass,), {})
            setattr(self, name, cls)
            return cls
        val = MockModule(f"{self.__name__}.{name}")
        val.__file__ = f"/mock/{self.__name__.replace('.', '/')}/{name}.py"
        val.__version__ = "3.0.0"
        setattr(self, name, val)
        return val

    def __call__(self, *args, **kwargs) -> Any:
        # If decorating a class or callable, preserve it
        if args and (isinstance(args[0], type) or callable(args[0])):
            return args[0]
        return self


class MockLoader(Loader):
    """Loader for UniversalMockFinder."""
    def create_module(self, spec: ModuleSpec) -> types.ModuleType:
        mod = MockModule(spec.name)
        mod.__path__ = []
        mod.__file__ = f"/mock/{spec.name.replace('.', '/')}.py"
        mod.ConfigMixin = ConfigMixin
        mod.SchedulerMixin = SchedulerMixin
        mod.KarrasDiffusionSchedulers = KarrasDiffusionSchedulers
        mod.SchedulerOutput = object
        mod.register_to_config = lambda f: f
        mod.remove = lambda *a, **k: None
        mod.new_session = lambda *a, **k: None
        mod.TorchDispatchMode = object
        mod.Qwen3Config = type("Qwen3Config", (), {})
        mod.QModuleMixin = type("QModuleMixin", (), {})
        mod.register_qmodule = lambda *a, **k: (lambda f: f)
        mod.qtypes = {}
        mod.QEmbedding = type("QEmbedding", (), {})
        mod.QLinear = type("QLinear", (), {})
        mod.QConv2d = type("QConv2d", (), {})
        mod.QTensor = type("QTensor", (), {})
        mod.AutoConfig = AutoConfigMock
        mod.init_empty_weights = _DummyContextManager
        mod.fix_text = lambda text: text

        class TensorType(str, Enum):
            PYTORCH = "pt"
            TENSORFLOW = "tf"
            NUMPY = "np"
            JAX = "jax"
        mod.TensorType = TensorType

        # Wire whisper LANGUAGES and TO_LANGUAGE_CODE
        whisper_langs = {
            "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
            "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
            "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
            "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
            "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
            "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
            "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
            "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
            "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
        }
        mod.LANGUAGES = whisper_langs
        mod.TO_LANGUAGE_CODE = {v: k for k, v in whisper_langs.items()}

        # Wire typing_extensions Unpack if available
        try:
            from typing_extensions import Unpack
            mod.Unpack = Unpack
        except ImportError:
            mod.Unpack = Any

        return mod

    def exec_module(self, module: types.ModuleType) -> None:
        pass


MOCK_HEAVY_PACKAGES = {
    "diffusers",
    "torchvision",
    "scipy",
    "torchaudio",
    "einops",
    "rembg",
    "whisper",
    "transformers",
    "optimum",
    "accelerate",
    "skimage",
    "smplfitter",
    "triton",
    "flash_attn",
    "onnxruntime",
    "matplotlib",
    "ftfy",
    "librosa",
    "av",
    "segment_anything",
    "timm",
    "iopath",
    "decord",
    "gguf",
    "tokenizers",
    "insightface",
    "facexlib",
    "pycocotools",
}

KNOWN_SYSTEM_PACKAGES = {
    "os", "sys", "io", "re", "math", "time", "json", "random", "threading", "asyncio",
    "datetime", "pathlib", "typing", "typing_extensions", "dataclasses", "collections",
    "functools", "itertools", "traceback", "shutil", "tempfile", "copy", "hashlib",
    "importlib", "inspect", "subprocess", "weakref", "gc", "uuid", "base64", "urllib",
    "http", "socket", "warnings", "logging", "argparse", "enum", "contextlib", "struct",
    "platform", "glob",
}


class UniversalMockFinder(MetaPathFinder):
    """MetaPathFinder that intercepts heavy ML packages in cloud client mode."""
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Optional[ModuleSpec]:
        top = fullname.split(".")[0]
        if top in KNOWN_SYSTEM_PACKAGES:
            return None
        if top in MOCK_HEAVY_PACKAGES:
            return ModuleSpec(fullname, MockLoader(), is_package=True)
        if top == "torch" and fullname != "torch":
            return ModuleSpec(fullname, MockLoader(), is_package=True)
        return None


def _create_mock_torch() -> types.ModuleType:
    """Build a comprehensive lightweight mock torch module for cloud client mode."""
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.6.0+cloud.client"
    torch_mod.__file__ = "/mock/torch/__init__.py"
    torch_mod.__path__ = ["/mock/torch"]
    torch_mod.Tensor = _DummyTensor
    torch_mod.Size = tuple
    torch_mod.device = _DummyDevice
    torch_mod.dtype = type
    torch_mod.Generator = type("Generator", (), {})

    # Data types
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.float64 = "float64"
    torch_mod.int8 = "int8"
    torch_mod.int16 = "int16"
    torch_mod.int32 = "int32"
    torch_mod.int64 = "int64"
    torch_mod.uint8 = "uint8"
    torch_mod.uint16 = "uint16"
    torch_mod.uint32 = "uint32"
    torch_mod.uint64 = "uint64"
    torch_mod.float8_e4m3fn = "float8_e4m3fn"
    torch_mod.float8_e5m2 = "float8_e5m2"
    torch_mod.bool = "bool"

    # Aliases
    torch_mod.float = torch_mod.float32
    torch_mod.double = torch_mod.float64
    torch_mod.half = torch_mod.float16
    torch_mod.int = torch_mod.int32
    torch_mod.long = torch_mod.int64
    torch_mod.short = torch_mod.int16
    torch_mod.complex64 = "complex64"
    torch_mod.complex128 = "complex128"

    # Tensor class aliases
    torch_mod.IntTensor = _DummyTensor
    torch_mod.FloatTensor = _DummyTensor
    torch_mod.LongTensor = _DummyTensor
    torch_mod.BoolTensor = _DummyTensor

    # Context managers
    torch_mod.no_grad = _DummyContextManager
    torch_mod.inference_mode = _DummyContextManager
    torch_mod.enable_grad = _DummyContextManager

    # CUDA submodule
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: False
    cuda_mod.device_count = lambda: 0
    cuda_mod.current_device = lambda: 0
    cuda_mod.get_device_name = lambda dev=None: "Cloud Worker"
    cuda_mod.get_device_capability = lambda dev=None: (0, 0)
    cuda_mod.empty_cache = lambda: None
    cuda_mod.synchronize = lambda dev=None: None
    cuda_mod.memory_allocated = lambda dev=None: 0
    cuda_mod.max_memory_allocated = lambda dev=None: 0
    cuda_mod.reset_peak_memory_stats = lambda dev=None: None
    cuda_mod.set_device = lambda dev=None: None
    cuda_mod.manual_seed = lambda s: None
    cuda_mod.manual_seed_all = lambda s: None

    amp_mod = types.ModuleType("torch.cuda.amp")
    amp_mod.autocast = _DummyContextManager
    cuda_mod.amp = amp_mod
    torch_mod.cuda = cuda_mod
    torch_mod.amp = amp_mod

    # Backends submodule
    backends_mod = types.ModuleType("torch.backends")
    mps_mod = types.ModuleType("torch.backends.mps")
    mps_mod.is_available = lambda: False
    mps_mod.is_built = lambda: False
    backends_mod.mps = mps_mod

    cuda_backends_mod = types.ModuleType("torch.backends.cuda")
    cuda_backends_mod.matmul = types.SimpleNamespace(allow_fp16_accumulation=True, allow_tf32=True)
    cuda_backends_mod.sdp_kernel = _DummyContextManager
    backends_mod.cuda = cuda_backends_mod

    cudnn_mod = types.ModuleType("torch.backends.cudnn")
    cudnn_mod.benchmark = False
    cudnn_mod.deterministic = False
    cudnn_mod.enabled = False
    backends_mod.cudnn = cudnn_mod
    torch_mod.backends = backends_mod

    # Version submodule
    version_mod = types.ModuleType("torch.version")
    version_mod.cuda = None
    version_mod.hip = None
    torch_mod.version = version_mod

    # _logging submodule
    logging_mod = types.ModuleType("torch._logging")
    logging_mod.set_logs = lambda **kwargs: None
    torch_mod._logging = logging_mod

    # compiler submodule
    compiler_mod = types.ModuleType("torch.compiler")
    compiler_mod.reset = lambda: None
    compiler_mod.is_compiling = lambda: False
    compiler_mod.disable = lambda fn=None, recursive=True: (lambda f: f) if fn is None else fn
    torch_mod.compiler = compiler_mod

    # jit submodule
    jit_mod = types.ModuleType("torch.jit")
    jit_mod.export = lambda f: f
    jit_mod.ignore = lambda f: f
    jit_mod.script = lambda f: f
    jit_mod.unused = lambda f: f
    try:
        from typing import Final
        jit_mod.Final = Final
    except ImportError:
        jit_mod.Final = Any
    torch_mod.jit = jit_mod

    # fx submodule
    fx_mod = types.ModuleType("torch.fx")
    fx_mod.wrap = lambda f: f
    torch_mod.fx = fx_mod

    # autograd submodule
    autograd_mod = types.ModuleType("torch.autograd")
    class _Function:
        @classmethod
        def apply(cls, *args, **kwargs):
            return None
    autograd_mod.Function = _Function
    autograd_mod.no_grad = _DummyContextManager
    autograd_mod.enable_grad = _DummyContextManager
    torch_mod.autograd = autograd_mod

    # ops submodule
    torch_mod.ops = MockModule("torch.ops")

    # NN and Functional submodules
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.__path__ = []
    class _NNModuleBase:
        def __init__(self, *args, **kwargs):
            pass
        def to(self, *args, **kwargs):
            return self
        def eval(self):
            return self
        def train(self, mode: bool = True):
            return self
        def __call__(self, *args, **kwargs):
            return None

    nn_mod.Module = _NNModuleBase
    nn_mod.Parameter = lambda x: x

    def _nn_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (MockClass,), {})
    nn_mod.__getattr__ = _nn_getattr

    functional_mod = types.ModuleType("torch.nn.functional")
    functional_mod.scaled_dot_product_attention = lambda *a, **k: None
    functional_mod.relu = lambda x, *a, **k: x
    functional_mod.gelu = lambda x, *a, **k: x
    functional_mod.silu = lambda x, *a, **k: x
    functional_mod.sigmoid = lambda x, *a, **k: x
    functional_mod.tanh = lambda x, *a, **k: x
    functional_mod.softmax = lambda x, *a, **k: x

    def _functional_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: a[0] if a else None
    functional_mod.__getattr__ = _functional_getattr
    nn_mod.functional = functional_mod

    init_mod = MockModule("torch.nn.init")
    nn_mod.init = init_mod

    nn_utils_mod = types.ModuleType("torch.nn.utils")
    nn_utils_mod.__path__ = []
    param_mod = types.ModuleType("torch.nn.utils.parametrizations")
    param_mod.weight_norm = lambda m, *a, **k: m
    nn_utils_mod.parametrizations = param_mod
    nn_utils_mod.weight_norm = lambda m, *a, **k: m
    nn_mod.utils = nn_utils_mod

    torch_mod.nn = nn_mod

    # library submodule (torch.library.custom_op)
    lib_mod = types.ModuleType("torch.library")
    def _dummy_custom_op(*a, **k):
        def decorator(fn):
            fn.register_fake = lambda *fa, **fk: (lambda f: f)
            return fn
        return decorator
    lib_mod.custom_op = _dummy_custom_op
    torch_mod.library = lib_mod

    # utils submodule (torch.utils._pytree, checkpoint, _python_dispatch)
    utils_mod = MockModule("torch.utils")
    utils_mod.__path__ = []
    disp_mod = MockModule("torch.utils._python_dispatch")
    disp_mod.TorchDispatchMode = object
    utils_mod._python_dispatch = disp_mod

    pytree_mod = MockModule("torch.utils._pytree")
    pytree_mod.tree_flatten = lambda x: ([x], None)
    pytree_mod.tree_unflatten = lambda v, s: v[0]
    pytree_mod.tree_map = lambda f, tree: f(tree)
    pytree_mod.tree_map_only = lambda ty, f, tree: f(tree) if isinstance(tree, ty) else tree
    pytree_mod.register_pytree_node = lambda *a, **k: None
    utils_mod._pytree = pytree_mod

    checkpoint_mod = MockModule("torch.utils.checkpoint")
    checkpoint_mod.checkpoint = lambda f, *a, **k: f(*a, **k)
    utils_mod.checkpoint = checkpoint_mod

    model_zoo_mod = MockModule("torch.utils.model_zoo")
    model_zoo_mod.load_url = lambda *a, **k: {}
    utils_mod.model_zoo = model_zoo_mod

    device_mod = MockModule("torch.utils._device")
    device_mod._device_constructors = lambda: []
    utils_mod._device = device_mod

    torch_mod.utils = utils_mod

    # distributed submodule
    dist_mod = MockModule("torch.distributed")
    dist_mod.is_initialized = lambda: False
    dist_mod.get_rank = lambda: 0
    dist_mod.get_world_size = lambda: 1
    dist_mod.ProcessGroup = type("ProcessGroup", (), {})
    torch_mod.distributed = dist_mod

    # overrides submodule
    overrides_mod = MockModule("torch.overrides")
    class TorchFunctionMode:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    overrides_mod.TorchFunctionMode = TorchFunctionMode
    overrides_mod.has_torch_function = lambda *a, **k: False
    overrides_mod.handle_torch_function = lambda func, relevant_args, *args, **kwargs: func(*args, **kwargs)
    torch_mod.overrides = overrides_mod

    # Memory formats
    torch_mod.preserve_format = "preserve_format"
    torch_mod.contiguous_format = "contiguous_format"
    torch_mod.channels_last = "channels_last"
    torch_mod.channels_last_3d = "channels_last_3d"

    # Basic tensor creation stubs (NumPy-backed so model metadata math works)
    def _shape_of(args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        return tuple(args)

    def _dtype_of(kwargs):
        dtype = kwargs.get("dtype")
        if isinstance(dtype, str) and dtype in ("float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"):
            return {"bfloat16": "float32"}.get(dtype, dtype)
        return None

    def _make_tensor(data=None, **kwargs):
        return _DummyTensor(data=data, dtype=kwargs.get("dtype"))

    torch_mod.from_numpy = lambda arr: _DummyTensor(data=arr)
    torch_mod.tensor = lambda data, **k: _DummyTensor(data=data, dtype=k.get("dtype"))
    torch_mod.as_tensor = torch_mod.tensor
    torch_mod.zeros = lambda *a, **k: _DummyTensor(__import__("numpy").zeros(_shape_of(a), dtype=_dtype_of(k)))
    torch_mod.ones = lambda *a, **k: _DummyTensor(__import__("numpy").ones(_shape_of(a), dtype=_dtype_of(k)))
    torch_mod.empty = lambda *a, **k: _DummyTensor(__import__("numpy").empty(_shape_of(a), dtype=_dtype_of(k)))
    torch_mod.full = lambda shape, fill_value, **k: _DummyTensor(__import__("numpy").full(tuple(shape) if isinstance(shape, (tuple, list)) else (shape,), fill_value, dtype=_dtype_of(k)))
    torch_mod.zeros_like = lambda t, **k: _DummyTensor(__import__("numpy").zeros_like(t._array if isinstance(t, _DummyTensor) else t))
    torch_mod.ones_like = lambda t, **k: _DummyTensor(__import__("numpy").ones_like(t._array if isinstance(t, _DummyTensor) else t))
    torch_mod.arange = lambda *a, **k: _DummyTensor(__import__("numpy").arange(*a, dtype=_dtype_of(k)))
    torch_mod.linspace = lambda *a, **k: _DummyTensor(__import__("numpy").linspace(*a, dtype=_dtype_of(k)))
    torch_mod.stack = lambda tensors, dim=0, **k: _DummyTensor(__import__("numpy").stack([t._array if isinstance(t, _DummyTensor) else t for t in tensors], axis=dim))
    torch_mod.cat = lambda tensors, dim=0, **k: _DummyTensor(__import__("numpy").concatenate([t._array if isinstance(t, _DummyTensor) else t for t in tensors], axis=dim))
    torch_mod.concat = torch_mod.cat
    torch_mod.save = lambda obj, f, **k: None
    torch_mod.load = lambda f, **k: {}
    torch_mod.manual_seed = lambda s: None
    torch_mod.seed = lambda: 0

    def _tensor_op(fn, *args, **kwargs):
        vals = [a._array if isinstance(a, _DummyTensor) else a for a in args]
        return _DummyTensor(fn(*vals, **kwargs))
    for _name, _fn in {
        "max": lambda *a, **k: __import__("numpy").max(*[x._array if isinstance(x, _DummyTensor) else x for x in a], **k),
        "min": lambda *a, **k: __import__("numpy").min(*[x._array if isinstance(x, _DummyTensor) else x for x in a], **k),
        "sum": lambda *a, **k: __import__("numpy").sum(*[x._array if isinstance(x, _DummyTensor) else x for x in a], **k),
        "mean": lambda *a, **k: __import__("numpy").mean(*[x._array if isinstance(x, _DummyTensor) else x for x in a], **k),
        "sqrt": lambda x, **k: __import__("numpy").sqrt(x._array if isinstance(x, _DummyTensor) else x),
        "exp": lambda x, **k: __import__("numpy").exp(x._array if isinstance(x, _DummyTensor) else x),
        "log": lambda x, **k: __import__("numpy").log(x._array if isinstance(x, _DummyTensor) else x),
        "abs": lambda x, **k: __import__("numpy").abs(x._array if isinstance(x, _DummyTensor) else x),
        "round": lambda x, **k: __import__("numpy").round(x._array if isinstance(x, _DummyTensor) else x),
        "floor": lambda x, **k: __import__("numpy").floor(x._array if isinstance(x, _DummyTensor) else x),
        "ceil": lambda x, **k: __import__("numpy").ceil(x._array if isinstance(x, _DummyTensor) else x),
        "sign": lambda x, **k: __import__("numpy").sign(x._array if isinstance(x, _DummyTensor) else x),
    }.items():
        setattr(torch_mod, _name, (lambda fn: (lambda *a, **k: _DummyTensor(fn(*a, **k))))(_fn))

    def _torch_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return MockModule(f"torch.{name}")
    torch_mod.__getattr__ = _torch_getattr

    return torch_mod


def _create_mock_mmgp() -> types.ModuleType:
    """Build a lightweight mock mmgp module for cloud client mode."""
    mmgp_mod = types.ModuleType("mmgp")
    mmgp_mod.__version__ = "3.8.2"

    class ProfileType(int, Enum):
        HighRAM_HighVRAM = 1
        HighRAM_LowVRAM = 2
        LowRAM_HighVRAM = 3
        LowRAM_LowVRAM = 4
        VerylowRAM_LowVRAM = 5
        default = 4
        fast = 2
        off = 0

        @staticmethod
        def tostr(v):
            mapping = {
                1: "HighRAM_HighVRAM",
                2: "HighRAM_LowVRAM",
                3: "LowRAM_HighVRAM",
                4: "LowRAM_LowVRAM",
                5: "VerylowRAM_LowVRAM",
            }
            return mapping.get(int(v) if isinstance(v, (int, float)) else 4, "LowRAM_LowVRAM")

    mmgp_mod.__file__ = "/mock/mmgp/__init__.py"
    mmgp_mod.__path__ = []

    p_type = ProfileType
    mmgp_mod.profile_type = p_type

    # offload
    offload_mod = types.ModuleType("mmgp.offload")
    offload_mod.__file__ = "/mock/mmgp/offload.py"
    offload_mod.__path__ = []
    offload_mod.PROFILE_DEFAULT = "default"
    offload_mod.PROFILE_FAST = "fast"
    offload_mod.PROFILE_OFF = "off"
    offload_mod.profile_type = p_type
    offload_mod.qtypes = {}
    offload_mod.shared_state = {}
    offload_mod.QEmbedding = type("QEmbedding", (), {})
    offload_mod.QLinear = type("QLinear", (), {})
    offload_mod.QConv2d = type("QConv2d", (), {})
    offload_mod.offload_model = lambda *a, **k: None
    offload_mod.clean_memory = lambda *a, **k: None
    offload_mod.is_main_process = lambda: True
    offload_mod.get_cache = lambda *a, **k: None
    offload_mod.clear_caches = lambda *a, **k: None
    def _safe_offload_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None
    offload_mod.__getattr__ = _safe_offload_getattr
    mmgp_mod.offload = offload_mod

    # safetensors2
    st2_mod = types.ModuleType("mmgp.safetensors2")
    st2_mod.__file__ = "/mock/mmgp/safetensors2.py"
    st2_mod.__path__ = []
    st2_mod.safe_open = lambda *a, **k: _DummyContextManager()
    mmgp_mod.safetensors2 = st2_mod

    # quant_router
    qr_mod = types.ModuleType("mmgp.quant_router")
    qr_mod.__file__ = "/mock/mmgp/quant_router.py"
    qr_mod.__path__ = []
    qr_mod.unregister_handler = lambda *a, **k: None
    qr_mod.register_handler = lambda *a, **k: None
    qr_mod.register_file_extension = lambda *a, **k: None
    qr_mod.get_quantization_tokens = lambda *a, **k: []
    qr_mod.get_handler = lambda *a, **k: None
    qr_mod.get_handlers = lambda *a, **k: []
    qr_mod.detect_quantization_label_from_filename = lambda *a, **k: ""
    qr_mod.get_quantization_label = lambda *a, **k: ""
    qr_mod.detect_quantization_kind_for_file = lambda *a, **k: ""
    qr_mod.get_cached_quantization_for_file = lambda *a, **k: None
    qr_mod.cache_quantization_for_file = lambda *a, **k: None
    qr_mod.detect_and_convert = lambda *a, **k: None
    qr_mod.get_available_qtype_aliases = lambda *a, **k: []
    qr_mod.get_available_qtypes = lambda *a, **k: []
    def _safe_qr_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if "token" in name or "list" in name or "aliases" in name or "types" in name or "handlers" in name:
            return lambda *a, **k: []
        if "label" in name or "kind" in name:
            return lambda *a, **k: ""
        return lambda *a, **k: None
    qr_mod.__getattr__ = _safe_qr_getattr
    mmgp_mod.quant_router = qr_mod

    # cuda_graphs
    cg_mod = types.ModuleType("mmgp.cuda_graphs")
    cg_mod.__file__ = "/mock/mmgp/cuda_graphs.py"
    cg_mod.__path__ = []
    cg_mod.scope = lambda *a, **k: _DummyContextManager()
    cg_mod.ready = lambda *a, **k: False
    cg_mod.invalidate = lambda *a, **k: None
    cg_mod.register_key_provider = lambda *a, **k: None
    cg_mod.register_gate = lambda *a, **k: None
    cg_mod.register_capture_setup = lambda *a, **k: None
    cg_mod.register_constant = lambda *a, **k: None
    def _safe_cg_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in ("scope", "capture_scope"):
            return lambda *a, **k: _DummyContextManager()
        return lambda *a, **k: None
    cg_mod.__getattr__ = _safe_cg_getattr
    mmgp_mod.cuda_graphs = cg_mod

    # allocator
    alloc_mod = types.ModuleType("mmgp.allocator")
    alloc_mod.__file__ = "/mock/mmgp/allocator/__init__.py"
    alloc_mod.__path__ = []
    ram_mod = types.ModuleType("mmgp.allocator.ram")
    ram_mod.__file__ = "/mock/mmgp/allocator/ram.py"
    ram_mod.install = lambda *a, **k: None
    ram_mod.uninstall = lambda *a, **k: None
    ram_mod.release = lambda *a, **k: None
    ram_mod.installed = lambda: False
    ram_mod.__getattr__ = lambda name: (lambda *a, **k: None)
    alloc_mod.ram = ram_mod

    ram_debug_mod = types.ModuleType("mmgp.allocator.ram_debug")
    ram_debug_mod.__file__ = "/mock/mmgp/allocator/ram_debug.py"
    ram_debug_mod.start = lambda *a, **k: None
    ram_debug_mod.stop = lambda *a, **k: None
    ram_debug_mod.write_report = lambda *a, **k: None
    ram_debug_mod.__getattr__ = lambda name: (lambda *a, **k: None)
    alloc_mod.ram_debug = ram_debug_mod
    mmgp_mod.allocator = alloc_mod

    # read_ahead & auto_preload
    ra_mod = types.ModuleType("mmgp.read_ahead")
    ra_mod.__file__ = "/mock/mmgp/read_ahead.py"
    ra_mod.__getattr__ = lambda name: (lambda *a, **k: None)
    mmgp_mod.read_ahead = ra_mod

    ap_mod = types.ModuleType("mmgp.auto_preload")
    ap_mod.__file__ = "/mock/mmgp/auto_preload.py"
    ap_mod.__getattr__ = lambda name: (lambda *a, **k: None)
    mmgp_mod.auto_preload = ap_mod

    # expert_compute (imported directly by shared.qtypes.gguf and LLM MoE layers)
    experts_mod = types.ModuleType("mmgp.expert_compute")
    experts_mod.__file__ = "/mock/mmgp/expert_compute.py"
    experts_mod.__path__ = []
    experts_mod.ExpertFormat = type("ExpertFormat", (MockClass,), {"name": "mock", "reports_usage": False})
    experts_mod.RoutedExperts = type("RoutedExperts", (MockClass,), {})
    experts_mod.SwiGLUExperts = type("SwiGLUExperts", (MockClass,), {})
    experts_mod.ExpertBank = type("ExpertBank", (MockClass,), {})
    experts_mod.DOORBELL = object()
    experts_mod.launch_host_job = lambda *a, **k: None
    experts_mod.cpu_threads = lambda *a, **k: 1
    experts_mod.__getattr__ = lambda name: (MockClass if name[:1].isupper() else (lambda *a, **k: None))
    mmgp_mod.expert_compute = experts_mod

    def _safe_mmgp_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        dummy = types.ModuleType(f"mmgp.{name}")
        dummy.__getattr__ = lambda n: (lambda *a, **k: None)
        return dummy
    mmgp_mod.__getattr__ = _safe_mmgp_getattr

    return mmgp_mod


def is_cloud_mode_active() -> bool:
    """Check if Cloud mode is currently requested or active."""
    if os.environ.get("WAN2GP_MODE", "").strip().lower() == "cloud":
        return True
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg in ("--cloud", "-cloud"):
            return True
        if arg.startswith("--mode=") and arg.split("=", 1)[1].strip().lower() == "cloud":
            return True
        if arg == "--mode" and i + 1 < len(argv) and argv[i + 1].strip().lower() == "cloud":
            return True
    return False


def install_cloud_shims(force: bool = False) -> bool:
    """
    Install lightweight shims and import interceptors into sys.modules.
    Returns True if shims were installed, False if local stack is preserved.
    """
    if not (force or is_cloud_mode_active()):
        return False

    # 1. Install universal MetaPathFinder for heavy ML packages
    if not any(isinstance(finder, UniversalMockFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, UniversalMockFinder())

    # 2. Torch shim (only if real torch is not installed or import fails)
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch_mock = _create_mock_torch()
            sys.modules["torch"] = torch_mock
            sys.modules["torch.cuda"] = torch_mock.cuda
            sys.modules["torch.cuda.amp"] = torch_mock.cuda.amp
            sys.modules["torch.amp"] = torch_mock.amp
            sys.modules["torch.backends"] = torch_mock.backends
            sys.modules["torch.backends.mps"] = torch_mock.backends.mps
            sys.modules["torch.backends.cuda"] = torch_mock.backends.cuda
            sys.modules["torch.backends.cudnn"] = torch_mock.backends.cudnn
            sys.modules["torch.version"] = torch_mock.version
            sys.modules["torch._logging"] = torch_mock._logging
            sys.modules["torch.compiler"] = torch_mock.compiler
            sys.modules["torch.jit"] = torch_mock.jit
            sys.modules["torch.ops"] = torch_mock.ops
            sys.modules["torch.nn"] = torch_mock.nn
            sys.modules["torch.nn.functional"] = torch_mock.nn.functional
            sys.modules["torch.nn.init"] = torch_mock.nn.init
            sys.modules["torch.nn.utils"] = torch_mock.nn.utils
            sys.modules["torch.nn.utils.parametrizations"] = torch_mock.nn.utils.parametrizations
            sys.modules["torch.library"] = torch_mock.library
            sys.modules["torch.utils"] = torch_mock.utils
            sys.modules["torch.utils._python_dispatch"] = torch_mock.utils._python_dispatch
            sys.modules["torch.utils._pytree"] = torch_mock.utils._pytree
            sys.modules["torch.utils.checkpoint"] = torch_mock.utils.checkpoint
            sys.modules["torch.utils.model_zoo"] = torch_mock.utils.model_zoo
            sys.modules["torch.distributed"] = torch_mock.distributed
            sys.modules["torch.utils._device"] = torch_mock.utils._device
            sys.modules["torch.overrides"] = torch_mock.overrides

    # 2b. Shield real torch.cuda so local client never crashes on CUDA calls in cloud mode
    try:
        import torch
        if hasattr(torch, "cuda"):
            _orig_get_device_capability = getattr(torch.cuda, "get_device_capability", None)
            def _safe_get_device_capability(device=None):
                if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
                    try:
                        return _orig_get_device_capability(device)
                    except Exception:
                        pass
                return (8, 0)
            torch.cuda.get_device_capability = _safe_get_device_capability

            _orig_current_device = getattr(torch.cuda, "current_device", None)
            def _safe_current_device():
                if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
                    try:
                        return _orig_current_device()
                    except Exception:
                        pass
                return 0
            torch.cuda.current_device = _safe_current_device

            class _CloudDummyDeviceProperties:
                total_memory = 24 * (1024 ** 3)
                multi_processor_count = 80
                major = 8
                minor = 0
                name = "Cloud GPU Client"
                L2_cache_size = 32 * 1024 * 1024

            _orig_get_device_properties = getattr(torch.cuda, "get_device_properties", None)
            def _safe_get_device_properties(device=None):
                if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
                    try:
                        return _orig_get_device_properties(device)
                    except Exception:
                        pass
                return _CloudDummyDeviceProperties()
            torch.cuda.get_device_properties = _safe_get_device_properties
            torch.cuda.get_device_name = lambda device=None: "Cloud GPU Client"
            torch.cuda.empty_cache = lambda: None
            torch.cuda.ipc_collect = lambda: None
            torch.cuda.synchronize = lambda *a, **k: None
            torch.cuda.set_device = lambda *a, **k: None
            torch.cuda.memory_allocated = lambda *a, **k: 0
            torch.cuda.memory_reserved = lambda *a, **k: 0
            torch.cuda.max_memory_allocated = lambda *a, **k: 0
    except Exception:
        pass

    # Safetensors torch shim
    try:
        import safetensors
        try:
            import safetensors.torch as st_torch
            if not hasattr(st_torch, "load_model"):
                st_torch.load_model = lambda *a, **k: None
            if not hasattr(st_torch, "save_model"):
                st_torch.save_model = lambda *a, **k: None
        except Exception:
            st_torch = types.ModuleType("safetensors.torch")
            st_torch.__file__ = "/mock/safetensors/torch.py"
            st_torch.__path__ = []
            st_torch.load_file = lambda *a, **k: {}
            st_torch.save_file = lambda *a, **k: None
            st_torch.load_model = lambda *a, **k: None
            st_torch.save_model = lambda *a, **k: None
            st_torch.safe_open = lambda *a, **k: _DummyContextManager()
            safetensors.torch = st_torch
            sys.modules["safetensors.torch"] = st_torch
        if not hasattr(safetensors, "safe_open"):
            safetensors.safe_open = lambda *a, **k: _DummyContextManager()
    except Exception:
        pass

    # 3. MMGP shim (always use lightweight mock in cloud mode)
    mmgp_mock = _create_mock_mmgp()
    sys.modules["mmgp"] = mmgp_mock
    sys.modules["mmgp.offload"] = mmgp_mock.offload
    sys.modules["mmgp.safetensors2"] = mmgp_mock.safetensors2
    sys.modules["mmgp.quant_router"] = mmgp_mock.quant_router
    sys.modules["mmgp.cuda_graphs"] = mmgp_mock.cuda_graphs
    sys.modules["mmgp.allocator"] = mmgp_mock.allocator
    sys.modules["mmgp.allocator.ram"] = mmgp_mock.allocator.ram
    sys.modules["mmgp.allocator.ram_debug"] = mmgp_mock.allocator.ram_debug
    sys.modules["mmgp.read_ahead"] = mmgp_mock.read_ahead
    sys.modules["mmgp.auto_preload"] = mmgp_mock.auto_preload
    sys.modules["mmgp.expert_compute"] = mmgp_mock.expert_compute
    return True
