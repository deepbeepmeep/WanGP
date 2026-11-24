from shared.startup_imports import lazy_exports

from . import wan_handler, df_handler, ovi_handler

# the model code (diffusers, transformers' tokenizers, the SCAIL tools) loads on first use, not whenever a handler is imported
__getattr__ = lazy_exports(__name__, {"configs": ".configs", "modules": ".modules", "DTT2V": ".diffusion_forcing", "WanAny2V": ".any2video"})
