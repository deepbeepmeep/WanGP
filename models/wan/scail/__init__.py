from shared.startup_imports import lazy_exports

__all__ = ["ScailPoseProcessor"]

# the pose tools (NLF, DWPose, matplotlib drawing) load when a SCAIL control video is processed: importing the
# package for a submodule (model_scail, read by every Wan model) would otherwise load them
__getattr__ = lazy_exports(__name__, {"ScailPoseProcessor": ".processor"})
