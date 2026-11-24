"""SAM3 preprocessing wrapper."""

from shared.startup_imports import lazy_exports

__all__ = ["build_sam3_image_model", "build_sam3_predictor", "run_sam3_video"]

# on first use: importing a submodule (the logger) would otherwise load the model code (timm, torchvision's detection ops)
__getattr__ = lazy_exports(__name__, {"build_sam3_image_model": ".model_builder", "build_sam3_predictor": ".model_builder", "run_sam3_video": ".preprocessor"})
