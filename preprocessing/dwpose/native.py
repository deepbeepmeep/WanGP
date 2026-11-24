# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""DWPose models in PyTorch: the YOLOX-L person detector and the RTMPose-L whole-body keypoints, loaded once and placed by MMGP.

`pose_models()` is the entry point: a context manager that loads the models when needed and returns their weights to RAM when the
outermost user leaves. Threads may share the models: their GPU calls take turns."""
import threading
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from mmgp import offload, runtime
from shared.utils import offload_registry
from shared.utils.media_control import controlled_model_loading, inference_checkpoint
from .assets import ensure_assets

_lock = threading.RLock()
_loaded = None
_users = 0
_owner = "dwpose"


class PoseModels:
    def __init__(self, detector, estimator, manager):
        self.detector, self.estimator, self.manager = detector, estimator, manager
        self.pose_input_size = estimator.input_size

    def _run(self, model, batch):
        inference_checkpoint()
        weight = next(model.parameters())
        with _lock, torch.inference_mode():
            inputs = torch.as_tensor(batch).to(device=runtime.execution_device(model), dtype=weight.dtype)
            return model(inputs)

    def detect_tensor(self, images):
        """(B, 3, 640, 640) pixels in 0-255 on any device -> (B, 8400, 85) float32 raw YOLOX outputs on the execution device."""
        return self._run(self.detector, images).float()

    def detect(self, images: np.ndarray) -> np.ndarray:
        return self.detect_tensor(images).cpu().numpy()

    def estimate(self, crops: np.ndarray):
        """(N, 3, 384, 288) normalized crops -> SimCC x / y logits as float32 arrays."""
        simcc_x, simcc_y = self._run(self.estimator, crops)
        return simcc_x.float().cpu().numpy(), simcc_y.float().cpu().numpy()


def release():
    global _loaded
    with _lock:
        entry, _loaded = _loaded, None
        if entry is not None:
            entry.manager.release()
            offload_registry.unregister_offloadobj(_owner, entry.manager)


@controlled_model_loading
def _load(detector_path, estimator_path, loading_callback=None):
    global _loaded
    if _loaded is not None:
        return _loaded
    if not torch.cuda.is_available():
        raise RuntimeError("Pose Estimation Requires a CUDA GPU.")
    tqdm.write("Loading - Pose Estimation Models (DWPose)")
    from accelerate import init_empty_weights
    from .rtmpose import RTMPose
    from .yolox import YOLOX
    with init_empty_weights(include_buffers=True):
        detector, estimator = YOLOX(), RTMPose()
    modules = {"pose_detector": detector, "pose_estimator": estimator}
    for module, path in ((detector, detector_path), (estimator, estimator_path)):
        offload.load_model_data(module, path, writable_tensors=False, default_dtype=None)
        module._convertWeightsFloatTo = None
        module.eval().requires_grad_(False)
    previous_device = torch.get_default_device()
    try:
        # Detector and estimator alternate on every frame: co-tenants, so that neither unloads the other.
        manager = offload.profile(modules, 0, budgets={name: 0 for name in modules}, coTenantsMap={"pose_detector": ["pose_estimator"], "pose_estimator": ["pose_detector"]},
                                  quantizeTransformer=False, loading_callback=loading_callback)
    finally:
        torch.set_default_device(None if previous_device.type == "cpu" else previous_device)
    _loaded = PoseModels(detector, estimator, manager)
    offload_registry.register_offloadobj(_owner, manager, release)
    return _loaded


@contextmanager
def pose_models(gen=None):
    global _users
    with _lock:
        models = _loaded or _load(*ensure_assets(gen))
        _users += 1
    try:
        yield models
    finally:
        with _lock:
            _users -= 1
            if _users == 0 and _loaded is models:
                models.manager.unload_all()
