from shared.startup_imports import lazy_exports

from . import ltxv_handler

__getattr__ = lazy_exports(__name__, {"LTXV": ".ltxv"})  # with diffusers' pipelines and transformers' generation code: on first use, not with the handler
