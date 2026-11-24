"""
DWpose (YOLOX + RTMPose ONNX) wrapper from the official SCAIL-Pose repo.

Upstream: https://github.com/zai-org/SCAIL-Pose (DWPoseProcess/dwpose/__init__.py)

Minimal adaptations for WanGP:
- Weights are resolved from `ckpts/pose` via `shared.utils.files_locator`.
- Removes the `controlnet_aux` dependency by inlining `HWC3`.
- GPU-only ONNXRuntime providers (no CPU fallback).
"""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
from PIL import Image

from preprocessing.dwpose.wholebody import Wholebody


def HWC3(x: np.ndarray) -> np.ndarray:
    """Ensure image is HxWx3 uint8 (ControlNet-style utility)."""
    if x.ndim != 3:
        raise ValueError(f"HWC3 expects 3D array, got shape {x.shape}")
    h, w, c = x.shape
    if c == 3:
        return x
    if c == 1:
        return np.concatenate([x, x, x], axis=2)
    if c == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        return np.clip(y, 0, 255).astype(np.uint8)
    raise ValueError(f"HWC3 unsupported channel count: {c}")


class DWposeDetector:
    def __init__(self, use_batch: bool = False):
        self.use_batch = use_batch

    def to(self, device: int):
        self.pose_estimation = Wholebody()
        return self

    def _get_multi_result_from_est(self, candidate, score_result, det_result, H, W):
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        subset_score = score_result[:, :24]
        face_score = score_result[:, 24:92]
        hand_score = score_result[:, 92:113]
        hand_score = np.vstack([hand_score, score_result[:, 113:]])

        body_candidate = candidate[:, :24].copy()
        for i in range(len(subset_score)):
            for j in range(len(subset_score[i])):
                if subset_score[i][j] > 0.3:
                    subset_score[i][j] = j
                else:
                    subset_score[i][j] = -1

        un_visible = score_result < 0.3
        candidate[un_visible] = -1

        faces = candidate[:, 24:92]
        hands = candidate[:, 92:113]
        hands = np.vstack([hands, candidate[:, 113:]])

        bodies = dict(candidate=body_candidate, subset=subset_score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)
        score = dict(body_score=subset_score, hand_score=hand_score, face_score=face_score)

        new_det_result = []
        for bbox in det_result:
            x1, y1, x2, y2 = bbox
            new_x1 = x1 / W
            new_y1 = y1 / H
            new_x2 = x2 / W
            new_y2 = y2 / H
            new_bbox = [new_x1, new_y1, new_x2, new_y2]
            new_det_result.append(new_bbox)

        return pose, score, new_det_result

    def __call__(self, input, **kwargs):
        if self.use_batch:
            raise NotImplementedError("DWposeDetector does not support batch mode")

        if isinstance(input, Image.Image):
            rgb = np.array(input, dtype=np.uint8)
        else:
            rgb = np.array(input, dtype=np.uint8)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = HWC3(bgr)
        H, W, C = bgr.shape

        with torch.no_grad():
            candidate, subset, det_result = self.pose_estimation(bgr)
            return self._get_multi_result_from_est(candidate, subset, det_result, H, W)

