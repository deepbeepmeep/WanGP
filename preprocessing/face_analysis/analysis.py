# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""InsightFace's FaceAnalysis on PyTorch: face boxes and 5 keypoints (SCRFD), 106 landmarks, identity embeddings (ArcFace), as the
face refiners and Lynx use them. MMGP owns the weights; release() frees them. Images are HxWx3 uint8 arrays in BGR order, as InsightFace's."""
import cv2
import numpy as np
import torch
from tqdm import tqdm

from mmgp import offload, runtime
from shared.utils import offload_registry
from . import insightface_ops as ops
from .assets import ensure_assets

_OWNER = "face_analysis"


class Face:
    """bbox (x0, y0, x1, y1), kps (5, 2), det_score; landmark_2d_106 (106, 2) and embedding (512,) when their task ran."""

    def __init__(self, bbox, kps, det_score):
        self.bbox, self.kps, self.det_score = bbox, kps, det_score
        self.landmark_2d_106 = self.embedding = None

    @property
    def normed_embedding(self):
        return None if self.embedding is None else self.embedding / np.linalg.norm(self.embedding)


def _networks():
    from .ir_se import Backbone
    from .nets import SCRFD, IResNet50, Landmark106
    return {"detection": SCRFD, "landmark_2d_106": Landmark106, "recognition": IResNet50, "recognition_ir_se50": Backbone}


class FaceAnalysis:
    def __init__(self, tasks=("detection", "recognition"), det_size=(640, 640), det_thresh=0.5, nms_thresh=0.4, gen=None):
        from accelerate import init_empty_weights
        if not torch.cuda.is_available():
            raise RuntimeError("Face Analysis Requires a CUDA GPU.")
        paths = ensure_assets(tasks, gen)
        tqdm.write("Loading - Face Analysis Models")
        networks, self.models = _networks(), {}
        for task in tasks:
            with init_empty_weights(include_buffers=True):
                model = networks[task]()
            offload.load_model_data(model, paths[task], writable_tensors=False, default_dtype=None)
            model._convertWeightsFloatTo = None
            self.models[task] = model.eval().requires_grad_(False)
        previous_device = torch.get_default_device()
        try:
            self.manager = offload.profile(self.models, 0, budgets={task: 0 for task in tasks}, quantizeTransformer=False)
        finally:
            torch.set_default_device(None if previous_device.type == "cpu" else previous_device)
        offload_registry.register_offloadobj(_OWNER, self.manager, self.release)
        self.det_size, self.det_thresh, self.nms_thresh = det_size, det_thresh, nms_thresh

    def release(self) -> bool:
        manager, self.manager, self.models = self.manager, None, {}
        if manager is None:
            return False
        offload_registry.unregister_offloadobj(_OWNER, manager)
        manager.release()
        return True

    def run(self, task, images, rgb=False):
        """A task's network on a list of equally sized uint8 images (BGR unless `rgb`)."""
        model = self.models[task]
        batch = np.stack(images) if rgb else np.stack(images)[..., ::-1]
        return model(torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).to(runtime.execution_device(model)))

    @torch.inference_mode()
    def detect(self, img):
        """Boxes (n, 5: x0, y0, x1, y1, score) and keypoints (n, 5, 2), in decreasing score order after NMS."""
        from .nets import SCRFD
        det_img, det_scale = ops.letterbox(img, self.det_size)
        height, width = det_img.shape[:2]
        levels = [(scores.cpu().numpy(), boxes.cpu().numpy(), kps.cpu().numpy(), (height // stride, width // stride))
                  for scores, boxes, kps, stride in zip(*self.run("detection", [det_img]), SCRFD.strides)]
        return ops.decode_detections(levels, SCRFD.strides, det_scale, self.det_thresh, self.nms_thresh)

    @torch.inference_mode()
    def get(self, img):
        bboxes, kpss = self.detect(img)
        faces = [Face(bbox[:4], kps, bbox[4]) for bbox, kps in zip(bboxes, kpss)]
        if faces and "landmark_2d_106" in self.models:
            crops = [ops.transform(img, ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2), 192,
                                   192 / (max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) * 1.5), 0) for face in faces]
            landmarks = self.run("landmark_2d_106", [crop for crop, _ in crops]).cpu().numpy()
            for face, points, (_, matrix) in zip(faces, landmarks, crops):
                face.landmark_2d_106 = ops.trans_points2d((points + 1) * 96, cv2.invertAffineTransform(matrix))
        if faces and "recognition" in self.models:
            embeddings = self.run("recognition", [ops.norm_crop(img, face.kps) for face in faces]).cpu().numpy()
            for face, embedding in zip(faces, embeddings):
                face.embedding = embedding
        return faces
