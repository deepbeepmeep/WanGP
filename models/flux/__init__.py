from shared.startup_imports import lazy_exports

from . import flux_handler

__getattr__ = lazy_exports(__name__, {"model_factory": ".flux_main"})  # the model code (and transformers' generation code): on first use, not with the handler
