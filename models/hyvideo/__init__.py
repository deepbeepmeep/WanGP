from shared.startup_imports import lazy_exports

from . import hunyuan_handler

__getattr__ = lazy_exports(__name__, {"HunyuanVideoSampler": ".hunyuan"})  # with diffusers' pipelines: on first use, not with the handler
