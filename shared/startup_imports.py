# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Leaner library imports at WanGP's startup.

Several libraries load at import parts that WanGP does not use, or only uses once a model or a feature runs: optimum-quanto's model
wrappers (which pull in diffusers' model classes, bitsandbytes and peft), torchvision's models, ops and datasets, PyTorch's compiler
(torch._dynamo with sympy and networkx, pulled in for decorators), gradio's IPython support and its writing of type-stub files for every
component, and repeated scans of the installed packages' entry points. `install()` hooks the import system before those libraries
load, so that each part loads when it is first used and behaves as before once loaded; `end_startup()` ends what only holds while
WanGP starts. Each entry below says why the part is not needed at startup.
"""
from __future__ import annotations

import importlib.abc
import sys
import threading

# Executed on first attribute access (importing their submodules does not execute them).
_DEFERRED = ("optimum.quanto.models",  # model wrappers WanGP does not use: they import diffusers' and transformers' model classes
             "torchvision.models", "torchvision.ops", "torchvision.datasets",  # used by model code when it runs (the ops load torch._dynamo)
             "fla.ops",  # imports all 22 operator families: WanGP imports fla.ops.gated_delta_rule only
             "whisper",  # the package loads the model and transcription code (and numba): Deepy's settings only read whisper.tokenizer
             "bitsandbytes",  # diffusers imports it for its own bitsandbytes quantization, which WanGP does not use
             "pandas",  # gradio's plot and dataframe components (unused by WanGP) import it: they use it only in their methods
             "boto3",  # accelerate imports it for its SageMaker configuration commands
             "torchvision.io",  # (with PyAV) video reading of a few models, used when they run
             "git",  # GitPython (it looks for the git executable at import): plugin installs and updates
             # diffusers' model families: importing one module of a family (autoencoders.vae for its output classes) ran the family's
             # __init__, which imports every model of it, and through them most of diffusers' UNets, transformers and loaders
             "diffusers.models.autoencoders", "diffusers.models.unets", "diffusers.models.transformers",
             # Deepy's LLM loader (the LM engines, Qwen 3.5 modeling and transformers' generation code) and its transcription: imported
             # by Deepy at startup, used when Deepy loads its model or transcribes
             "shared.prompt_enhancer.qwen35_text", "shared.deepy.transcription")

# Imports refused while a module executes: the module handles their absence (try / except ImportError).
_BLOCKED_DURING = {"gradio.ipython_ext": ("IPython",),  # the %%blocks notebook magic
                   "diffusers.utils.torch_utils": ("torch._dynamo",)}  # maybe_allow_in_graph, made lazy below

# Exact source edits: a module whose source no longer contains the text loads unchanged.
_SOURCE_EDITS = {"comfy_kitchen.tensor.base": (("import torch._dynamo\n", ""),),  # unused: nothing in comfy_kitchen reads torch._dynamo
                 # torch.utils.cpp_extension (setuptools, Cython) builds quanto's own C++ kernels, which WanGP does not use: imported when one is built
                 "optimum.quanto.library.extensions.extension": (("from torch.utils.cpp_extension import load\n", ""),
                                                                ("            self._lib = load(\n", "            from torch.utils.cpp_extension import load\n            self._lib = load(\n")),
                 # torch._dynamo (with sympy and torch's compiler stack) for a context manager entered when a flex-attention mask is made
                 "transformers.masking_utils": (("if _is_torch_greater_or_equal_than_2_6:\n    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex\n",
                                                 "if _is_torch_greater_or_equal_than_2_6:\n    def TransformGetItemToIndex():\n"
                                                 "        from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex\n"
                                                 "        return TransformGetItemToIndex()\n"),),
                 # sklearn (and scipy.stats) for the confidence threshold of assisted generation, which WanGP does not use
                 "transformers.generation.candidate_generator": (("if is_sklearn_available():\n    from sklearn.metrics import roc_curve\n",
                                                                  "if is_sklearn_available():\n    def roc_curve(*args, **kwargs):\n"
                                                                  "        from sklearn.metrics import roc_curve\n"
                                                                  "        return roc_curve(*args, **kwargs)\n"),),
                 # scipy.optimize for the box matching of object-detection losses (DETR-like models), which WanGP does not use
                 **{name: (("if is_scipy_available():\n    from scipy.optimize import linear_sum_assignment\n",
                            "if is_scipy_available():\n    def linear_sum_assignment(*args, **kwargs):\n"
                            "        from scipy.optimize import linear_sum_assignment\n"
                            "        return linear_sum_assignment(*args, **kwargs)\n"),)
                    for name in ("transformers.loss.loss_for_object_detection", "transformers.loss.loss_rt_detr")}}

_after = {}  # module name -> callables run with the module right after it executes
_blocking = []  # (thread id, prefix): imports refused in that thread at the moment
_installed = False


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        thread = threading.get_ident()
        if any(thread == owner and (name == prefix or name.startswith(prefix + ".")) for owner, prefix in _blocking):
            raise ModuleNotFoundError(f"No module named {name!r} (not needed while WanGP starts)", name=name)
        if name not in _after and name not in _DEFERRED and name not in _BLOCKED_DURING and name not in _SOURCE_EDITS:
            return None
        spec = next((spec for finder in sys.meta_path if finder is not self and hasattr(finder, "find_spec") for spec in (finder.find_spec(name, path, target),) if spec is not None), None)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _Loader(spec.loader, name)
        return spec


class _Loader(importlib.abc.Loader):
    def __init__(self, loader, name):
        self.loader, self.name = loader, name

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        if self.name in _DEFERRED:
            _defer(module, self.loader)
            return
        blocked = [(threading.get_ident(), prefix) for prefix in _BLOCKED_DURING.get(self.name, ())]
        _blocking.extend(blocked)
        try:
            self._execute(module)
        finally:
            for entry in blocked:
                _blocking.remove(entry)
        for callback in _after.pop(self.name, ()):
            callback(module)

    def _execute(self, module):
        edits = _SOURCE_EDITS.get(self.name)
        source = self.loader.get_source(self.name) if edits else None
        if source is None or not all(old in source for old, _ in edits):
            self.loader.exec_module(module)
            return
        for old, new in edits:
            source = source.replace(old, new, 1)
        exec(compile(source, self.loader.get_filename(self.name), "exec"), module.__dict__)

    def __getattr__(self, name):  # get_source, get_filename, is_package... of the wrapped loader
        return getattr(self.loader, name)


def _defer(module, loader):
    """The module stays empty (an empty __all__ for star imports) until one of its attributes is read: then it executes in place."""
    empty = module.__all__ = []
    lock, state = threading.Lock(), {"done": False, "thread": None}  # threads reading it first: one executes it, the others wait

    def __getattr__(name):
        if name.startswith("__") and name.endswith("__") and name != "__version__":  # probes of module attributes, not uses
            raise AttributeError(name)
        if state["thread"] == threading.get_ident():  # read by its own code while it executes, as in any import
            raise AttributeError(f"partially initialized module {module.__name__!r} has no attribute {name!r}")
        with lock:
            if not state["done"]:
                if module.__dict__.get("__all__") is empty:
                    del module.__all__
                state["thread"] = threading.get_ident()
                try:
                    loader.exec_module(module)
                finally:
                    state["thread"] = None
                state["done"] = True
                if module.__dict__.get("__getattr__") is __getattr__:  # unless the module defines its own
                    del module.__getattr__
        return getattr(module, name)
    module.__getattr__ = __getattr__


def lazy_exports(package, exports):
    """A package's __getattr__ (PEP 562) for the names its __init__ would import from heavy modules: exports maps each name to the module
    it comes from (relative to the package), imported on first use; a name that is that module's own name (configs: ".configs") is the
    module itself. Usage in an __init__: `__getattr__ = lazy_exports(__name__, {...})`."""
    import importlib

    def __getattr__(name):
        if name not in exports:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module = importlib.import_module(exports[name], package)
        return module if exports[name].rsplit(".", 1)[-1] == name else getattr(module, name)
    return __getattr__


def after_import(name, callback):
    """Runs callback(module) once the module has executed: at once if it already has."""
    module = sys.modules.get(name)
    if module is not None and name not in _after:
        callback(module)
    else:
        _after.setdefault(name, []).append(callback)


def _dynamo_loaded():
    return "torch._dynamo" in sys.modules and "torch._dynamo" not in _after


def _no_type_stubs(component_meta):
    # Writes or rewrites a .pyi file beside each component class's source, for code editors, by parsing that source at every import.
    component_meta.create_or_modify_pyi = lambda component_class, class_name, events: None


def _no_notebook_checks(utils):
    # gradio checks at each Blocks whether it runs in IPython or Colab by importing IPython: a notebook has imported IPython before any
    # of WanGP's code runs, so without it the answer is no.
    if "IPython" not in sys.modules:
        utils.ipython_check = lambda: False
        utils.colab_check = lambda: False


def _lazy_allow_in_graph(torch_utils):
    # diffusers imports torch._dynamo for this class decorator, which matters once torch.compile traces the class: the classes decorated
    # before the compiler loads are registered when it does.
    pending = []

    def maybe_allow_in_graph(cls):
        if _dynamo_loaded():
            return sys.modules["torch._dynamo"].allow_in_graph(cls)
        pending.append(cls)
        return cls

    def register(dynamo):
        for cls in pending:
            dynamo.allow_in_graph(cls)
        pending.clear()
    torch_utils.maybe_allow_in_graph = maybe_allow_in_graph
    after_import("torch._dynamo", register)


def _lazy_compiler_disable(compiler):
    # torch.compiler.disable() loads torch._dynamo when it decorates a function: modules that decorate theirs at import (WanGP's kernels at
    # startup, transformers' flex attention with a model's first load) would load it. A function decorated while torch._dynamo is not
    # loaded runs as it is until it has loaded (nothing can be compiling before), then through torch.compiler.disable as before: their
    # torch.compiler.disable versions are made as torch._dynamo loads, outside any compiled code.
    import functools
    import weakref
    original = compiler.disable
    pending = weakref.WeakSet()

    def disable(fn=None, recursive=True, *, reason=None):
        if _dynamo_loaded():
            return original(fn, recursive, reason=reason)
        if fn is None:
            return lambda fn: disable(fn, recursive, reason=reason)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            disabled = wrapper._wangp_disabled
            return fn(*args, **kwargs) if disabled is None else disabled(*args, **kwargs)
        wrapper._wangp_disabled = None
        wrapper._wangp_disable = lambda: original(fn, recursive, reason=reason)
        pending.add(wrapper)
        return wrapper

    def make_disabled(dynamo):
        for wrapper in list(pending):
            wrapper._wangp_disabled = wrapper._wangp_disable()
        pending.clear()
    disable.__wrapped__ = original
    compiler.disable = disable
    after_import("torch._dynamo", make_disabled)


def _lazy_disable_dynamo(compile_module):
    # torch's own functions marked "do not trace" (the Python versions of its ops, which run for tensors on the meta device: a model
    # built before its weights load) load torch._dynamo at their first call, only to make that mark. While torch._dynamo is not loaded
    # nothing can be compiling, so they run as they are; once it has loaded, through torch's own wrapper as before.
    import functools
    original = compile_module._disable_dynamo

    def _disable_dynamo(fn=None, recursive=True):
        if fn is None:
            return functools.partial(_disable_dynamo, recursive=recursive)
        disabled = original(fn, recursive)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if "torch._dynamo" not in sys.modules:
                return fn(*args, **kwargs)
            return disabled(*args, **kwargs)
        return inner
    _disable_dynamo.__wrapped__ = original
    compile_module._disable_dynamo = _disable_dynamo


_entry_points_cache = {}


def _cache_entry_points():
    # Ten libraries scan the entry points of every installed package at import (about 440 files each): one scan serves them all while
    # WanGP starts; end_startup() restores the live scan, so that packages installed later are seen.
    import importlib.metadata as metadata
    original = metadata.entry_points

    def entry_points(**params):
        if "all" not in _entry_points_cache:
            _entry_points_cache["all"] = original()
        return _entry_points_cache["all"].select(**params) if params else _entry_points_cache["all"]
    entry_points.__wrapped__ = original
    metadata.entry_points = entry_points


def _cache_packages_distributions():
    # importlib.metadata.packages_distributions() reads the metadata of every installed package (their RECORD files without a
    # top_level.txt): diffusers calls it once at import, 1-2 s here. Kept on disk, keyed by the packages' metadata folders, which change
    # with every package installed, removed or upgraded.
    import hashlib
    import importlib.metadata as metadata
    import json
    import os
    original = metadata.packages_distributions
    path = os.path.join(os.path.expanduser("~"), ".cache", "wangp", "packages_distributions.json")

    def packages_distributions():
        folders = []
        for entry in sys.path:
            if os.path.isdir(entry):
                folders += sorted(name for name in os.listdir(entry) if name.endswith((".dist-info", ".egg-info")))
        key = hashlib.sha1("\n".join(folders).encode("utf-8")).hexdigest()
        try:
            with open(path, encoding="utf-8") as reader:
                cached = json.load(reader)
            if cached["key"] == key:
                return cached["packages"]
        except (OSError, ValueError, KeyError):  # no cache yet, or an unreadable one: computed again below
            pass
        packages = original()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as writer:
            json.dump({"key": key, "packages": packages}, writer)
        return packages
    packages_distributions.__wrapped__ = original
    metadata.packages_distributions = packages_distributions


def install():
    global _installed
    if _installed:
        return
    _installed = True
    # in a process that has already imported some of these (WanGP imported by a test or a tool), what they would defer is loaded already
    global _DEFERRED, _BLOCKED_DURING, _SOURCE_EDITS
    _DEFERRED = tuple(name for name in _DEFERRED if name not in sys.modules)
    _BLOCKED_DURING = {name: blocked for name, blocked in _BLOCKED_DURING.items() if name not in sys.modules}
    _SOURCE_EDITS = {name: edits for name, edits in _SOURCE_EDITS.items() if name not in sys.modules}
    sys.meta_path.insert(0, _Finder())
    after_import("gradio.component_meta", _no_type_stubs)
    after_import("gradio.utils", _no_notebook_checks)
    after_import("diffusers.utils.torch_utils", _lazy_allow_in_graph)
    after_import("torch.compiler", _lazy_compiler_disable)
    after_import("torch._compile", _lazy_disable_dynamo)
    # torch.compile skips these wrappers as it skips torch's own: a call to one is a graph break that runs as written, not a frame to compile
    after_import("torch._dynamo", lambda dynamo: dynamo.trace_rules.add(__name__))
    _cache_entry_points()
    _cache_packages_distributions()


def end_startup():
    """Entry points are scanned live again, so that packages installed later are seen."""
    import importlib.metadata as metadata
    if hasattr(metadata.entry_points, "__wrapped__"):
        metadata.entry_points = metadata.entry_points.__wrapped__
    _entry_points_cache.clear()
