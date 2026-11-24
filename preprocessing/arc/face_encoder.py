# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# WanGP: InsightFace and facexlib replaced by preprocessing.face_analysis (same networks in PyTorch, WanGP-managed checkpoints).

import numpy as np
import torch

from preprocessing.face_analysis import insightface_ops as face_align
from preprocessing.face_analysis.analysis import FaceAnalysis


__all__ = [
    "FaceEncoderArcFace",
    "get_landmarks_from_image",
]


def get_landmarks_from_image(image):
    """
    Detect landmarks with insightface.

    Args:
        image (np.ndarray or PIL.Image):
            The input image in RGB format.

    Returns:
        5 2D keypoints, only one face will be returned.
    """
    detector = FaceAnalysis(("detection",))
    try:
        # As upstream, the RGB image goes to the detector unchanged (InsightFace expects BGR).
        faces = detector.get(np.array(image).copy())
    finally:
        detector.release()
    if len(faces) == 0:
        raise ValueError("No face detected in the image")

    # Get the largest face
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    # Return the 5 keypoints directly
    return face.kps  # 5 x 2


class FaceEncoderArcFace():
    """ Official ArcFace, no_grad-only """

    def __repr__(self):
        return "ArcFace"


    def init_encoder_model(self, device, eval_mode=True):
        self.device = device


    @torch.no_grad()
    def __call__(self, in_image, need_proc=False, landmarks=None, image_size=112):
        """The normalized [B, 512] embedding of an RGB image aligned on its 5 `landmarks` (need_proc), or of a uint8 RGB [B, 112, 112, 3] batch of aligned faces."""
        if need_proc:
            assert landmarks is not None, "landmarks are not provided!"
            in_image = face_align.norm_crop(np.array(in_image), landmark=np.array(landmarks), image_size=image_size)[None]
        encoder = FaceAnalysis(("recognition_ir_se50",))
        try:
            return encoder.run("recognition_ir_se50", list(in_image), rgb=True)
        finally:
            encoder.release()
