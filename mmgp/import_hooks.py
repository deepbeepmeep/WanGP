# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Code run right after a module is first imported: mmgp's patches to optional libraries apply when the library loads its patched
module, instead of importing it (and what it imports) when mmgp loads."""
import importlib.abc
import sys

_after = {}  # module name -> callables run with the module once it has executed


class _Loader(importlib.abc.Loader):
    def __init__(self, loader, name):
        self.loader, self.name = loader, name

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        for callback in _after.pop(self.name, ()):
            callback(module)

    def __getattr__(self, name):  # get_source, get_filename, is_package... of the wrapped loader
        return getattr(self.loader, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _after:
            return None
        spec = next((spec for finder in sys.meta_path if finder is not self and hasattr(finder, "find_spec") for spec in (finder.find_spec(name, path, target),) if spec is not None), None)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _Loader(spec.loader, name)
        return spec


_finder = _Finder()


def after_import(name, callback):
    """Runs callback(module) once the module has executed: at once if it already has."""
    module = sys.modules.get(name)
    if module is not None and name not in _after:
        callback(module)
        return
    _after.setdefault(name, []).append(callback)
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
