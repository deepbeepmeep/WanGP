"""
SCAIL NLFPose Extractor - 3D pose extraction using NLFPose.
Reuses WanGP's existing DWPose/YOLOX for detection, adds NLFPose for 3D lifting.
"""

import os
import math
import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Optional, Tuple, Union, Dict, Any

from shared.utils import files_locator as fl
from preprocessing.dwpose.wholebody import HWC3, Wholebody, resize_image


ArrayImage = Union[np.ndarray, Image.Image, torch.Tensor]


def _to_rgb_array(image: ArrayImage) -> np.ndarray:
    """Convert various image formats to RGB uint8 numpy array."""
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.dim() == 4:
            img = img[0]  # Take first batch
        if img.shape[0] in (1, 3, 4):
            img = img.permute(1, 2, 0)
        if img.min() < 0:
            img = (img + 1.0) * 127.5
        elif img.max() <= 1.0:
            img = img * 255.0
        arr = img.clamp(0, 255).byte().numpy()
        if arr.shape[2] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            return image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


class NLFPoseExtractor:
    """
    NLFPose 3D keypoint extraction wrapper.

    Uses YOLOX for person detection, DWPose for 2D keypoint estimation (hands/face),
    and NLFPose (isarandi/nlf) for true 3D lifting from images.
    """

    # Expected NLFPose model filename
    # NLFPOSE_MODEL = "nlf_l_multi_0.3.2.torchscript"
    NLFPOSE_MODEL = "nlf_l_multi_0.3.2_torch2.7.1.torchscript"

    def __init__(
        self,
        device: str = None,
        detect_resolution: int = 1024,
    ):
        """
        Initialize the extractor.

        Args:
            device: Device to run models on (default: auto-detect)
            detect_resolution: Resolution for pose detection
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detect_resolution = detect_resolution

        # Locate model files
        from preprocessing.dwpose.assets import query_download_def
        from shared.utils.download import process_files_def_if_needed

        process_files_def_if_needed(query_download_def())

        # Try to locate NLFPose model
        self.nlfpose_model_path = self._locate_nlfpose_model()
        if self.nlfpose_model_path is None:
            print(
                f"[SCAIL] Warning: NLFPose model '{self.NLFPOSE_MODEL}' not found; using 2D pose + heuristic depth. "
                "Place it under `ckpts/pose/` or `ckpts/scail/` to enable true 3D lifting."
            )

        # Lazy load models
        self._wholebody = None
        self._nlfpose = None

    def _locate_nlfpose_model(self) -> Optional[str]:
        """Try to locate NLFPose model file."""
        # Try common locations
        try:
            path = fl.locate_file(f"pose/{self.NLFPOSE_MODEL}")
            if path and os.path.exists(path):
                return path
        except:
            pass

        try:
            path = fl.locate_file(f"scail/{self.NLFPOSE_MODEL}")
            if path and os.path.exists(path):
                return path
        except:
            pass

        # Try direct paths
        for path in [
            os.path.join("ckpts", self.NLFPOSE_MODEL),
            os.path.join("models", "pose", self.NLFPOSE_MODEL),
        ]:
            if os.path.exists(path):
                return path

        return None

    @property
    def wholebody(self) -> Wholebody:
        """Lazy load DWPose/YOLOX detector."""
        if self._wholebody is None:
            self._wholebody = Wholebody()
        return self._wholebody

    @property
    def nlfpose(self):
        """Lazy load NLFPose model."""
        if self._nlfpose is None and self.nlfpose_model_path:
            try:
                self._nlfpose = torch.jit.load(self.nlfpose_model_path)
                self._nlfpose.to(self.device)
                self._nlfpose.eval()
            except Exception as e:
                print(f"[SCAIL] Warning: Could not load NLFPose model: {e}")
                print("[SCAIL] Falling back to 2D pose with estimated depth")
                self._nlfpose = None
        return self._nlfpose

    def extract_2d_keypoints(
        self,
        image: ArrayImage,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract 2D keypoints from an image using DWPose.

        Args:
            image: Input image
            mask: Optional mask to filter detected persons

        Returns:
            Tuple of (body_keypoints, hand_keypoints, face_keypoints, bbox)
            Each with shape (num_persons, num_joints, 2/3) for body, hands, face
        """
        rgb = _to_rgb_array(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = HWC3(bgr)
        resized = resize_image(bgr, self.detect_resolution)
        H, W = resized.shape[:2]

        # Run DWPose detection
        candidate, subset, det_result = self.wholebody(resized)

        if len(candidate) == 0:
            return np.array([]), np.array([]), np.array([]), None

        # Normalize coordinates to [0, 1]
        candidate = candidate.copy()
        subset = subset.copy() if hasattr(subset, "copy") else subset
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        bbox = None
        if det_result is not None and len(det_result) > 0:
            det_result = np.asarray(det_result).copy()
            det_result[:, [0, 2]] /= float(W)
            det_result[:, [1, 3]] /= float(H)

        # Filter by mask if provided
        if mask is not None:
            mask_resized = cv2.resize(mask, (W, H))
            candidate, subset, det_result = self._filter_by_mask(candidate, subset, det_result, mask_resized)

        # Extract first person's keypoints
        if len(candidate) == 0:
            return np.array([]), np.array([]), np.array([]), None

        # Pick the most confident person.
        # subset contains per-keypoint confidence scores (n, K).
        body_scores_all = subset[:, :18] if subset is not None and len(subset) > 0 else None
        if body_scores_all is None:
            max_ind = 0
        else:
            max_ind = int(np.nanargmax(np.nanmean(body_scores_all, axis=1)))

        # Body keypoints (18 OpenPose/COCO keypoints) with scores
        body_xy = candidate[max_ind, :18].copy()
        body_scores = (
            subset[max_ind, :18].copy()
            if subset is not None and len(subset) > 0
            else np.ones(18, dtype=np.float32)
        )
        body = np.concatenate([body_xy, body_scores[:, None]], axis=-1).astype(np.float32)
        low_conf_body = body_scores <= 0.3
        body[low_conf_body] = -1.0

        # Hands (21 keypoints each, right then left)
        hands = np.zeros((2, 21, 3), dtype=np.float32)
        if candidate.shape[1] >= 134:
            rh_xy = candidate[max_ind, 92:113].copy()
            lh_xy = candidate[max_ind, 113:134].copy()
            rh_scores = (
                subset[max_ind, 92:113].copy()
                if subset is not None and len(subset) > 0
                else np.ones(21, dtype=np.float32)
            )
            lh_scores = (
                subset[max_ind, 113:134].copy()
                if subset is not None and len(subset) > 0
                else np.ones(21, dtype=np.float32)
            )
            hands[0] = np.concatenate([rh_xy, rh_scores[:, None]], axis=-1)
            hands[1] = np.concatenate([lh_xy, lh_scores[:, None]], axis=-1)
            hands[hands[:, :, 2] <= 0.3] = -1.0

        # Face (68 keypoints)
        faces = np.zeros((1, 68, 3), dtype=np.float32)
        if candidate.shape[1] >= 92:
            face_xy = candidate[max_ind, 24:92].copy()
            face_scores = (
                subset[max_ind, 24:92].copy()
                if subset is not None and len(subset) > 0
                else np.ones(68, dtype=np.float32)
            )
            faces[0] = np.concatenate([face_xy, face_scores[:, None]], axis=-1)
            faces[0][faces[0, :, 2] <= 0.3] = -1.0

        if det_result is not None and len(det_result) > 0 and max_ind < len(det_result):
            bbox = det_result[max_ind].astype(np.float32)
        else:
            bbox = self._bbox_from_body(body)

        return body, hands, faces, bbox

    def _filter_by_mask(
        self,
        candidate: np.ndarray,
        subset: np.ndarray,
        det_result: Optional[np.ndarray],
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Filter detected persons by mask overlap (returns candidate, scores, and bboxes)."""
        H, W = mask.shape[:2]
        filtered_idx = []

        for i in range(len(candidate)):
            # Check if person's keypoints fall within mask
            body = candidate[i, :18]
            valid_pts = body[body[:, 0] >= 0]
            if len(valid_pts) == 0:
                continue

            # Convert normalized coords to pixel coords
            px = (valid_pts[:, 0] * W).astype(int).clip(0, W - 1)
            py = (valid_pts[:, 1] * H).astype(int).clip(0, H - 1)

            # Check mask overlap
            mask_vals = mask[py, px]
            overlap = np.mean(mask_vals > 0.5)

            if overlap > 0.3:  # At least 30% overlap
                filtered_idx.append(i)

        if not filtered_idx:
            return np.array([]), np.array([]), None

        filtered_candidate = candidate[filtered_idx]
        filtered_subset = subset[filtered_idx] if subset is not None and len(subset) > 0 else subset
        filtered_det = det_result[filtered_idx] if det_result is not None and len(det_result) > 0 else det_result
        return filtered_candidate, filtered_subset, filtered_det

    def _bbox_from_body(self, body: np.ndarray) -> Optional[np.ndarray]:
        """Approximate person bbox from body keypoints (normalized xy)."""
        if body is None or not isinstance(body, np.ndarray) or body.size == 0:
            return None
        valid = (body[:, 0] >= 0) & (body[:, 1] >= 0)
        if not valid.any():
            return None
        xs = body[valid, 0]
        ys = body[valid, 1]
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        # Expand slightly.
        w = max(x2 - x1, 1e-3)
        h = max(y2 - y1, 1e-3)
        x1 -= 0.05 * w
        x2 += 0.05 * w
        y1 -= 0.08 * h
        y2 += 0.08 * h
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _estimate_3d_from_2d(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Fallback "3D" from 2D keypoints (no NLFPose available).
        Returns normalized x/y plus heuristic depth in [0.5, 1.0].
        """
        if keypoints_2d is None or not isinstance(keypoints_2d, np.ndarray) or keypoints_2d.size == 0:
            return np.full((18, 3), -1.0, dtype=np.float32)

        result = np.zeros((len(keypoints_2d), 3), dtype=np.float32)
        result[:, :2] = keypoints_2d[:, :2].astype(np.float32)

        # Estimate depth based on vertical position and body structure
        # Higher Y (lower in image) typically means closer to camera
        valid_mask = keypoints_2d[:, 0] >= 0

        if valid_mask.any():
            y_coords = keypoints_2d[valid_mask, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            y_range = max(y_max - y_min, 0.1)

            # Normalize depth to [0, 1] range
            for i in range(len(keypoints_2d)):
                if valid_mask[i]:
                    # Invert so lower Y (higher in image) = larger depth
                    normalized_y = (keypoints_2d[i, 1] - y_min) / y_range
                    result[i, 2] = 1.0 - normalized_y * 0.5  # Depth range 0.5-1.0

        # Mark invalid keypoints
        result[~valid_mask, :] = -1

        return result

    def _run_nlfpose_batched(
        self,
        frames_rgb: List[np.ndarray],
        bboxes_norm: List[Optional[np.ndarray]],
        batch_size: int = 32,
    ) -> List[Optional[np.ndarray]]:
        """
        Run NLFPose torchscript on a list of frames using one bbox per frame.

        Returns:
            List[Optional[np.ndarray]]: per-frame SMPL-24 joints in camera space (24, 3) or None.
        """
        model = self.nlfpose
        if model is None:
            return [None] * len(frames_rgb)

        device = torch.device(self.device)
        out: List[Optional[np.ndarray]] = [None] * len(frames_rgb)

        def _call_model(img_batch: torch.Tensor) -> Any:
            # SCAIL-Pose uses `detect_smpl_batched` on this weight. Keep fallback for other wrappers.
            if hasattr(model, "detect_smpl_batched"):
                return model.detect_smpl_batched(img_batch)
            return model(img_batch)

        def _extract_joints(pred: Any, expected: int) -> List[Optional[torch.Tensor]]:
            if isinstance(pred, dict):
                val = None
                for k in ("joints3d_nonparam", "joints3d"):
                    if k in pred:
                        val = pred[k]
                        break
                if val is None:
                    return [None] * expected
            else:
                val = pred

            if isinstance(val, torch.Tensor):
                # (B, 24, 3) or (B, 1, 24, 3)
                if val.ndim == 4 and val.shape[1] == 1:
                    val = val[:, 0]
                if val.ndim == 3:
                    return [val[i] for i in range(min(expected, val.shape[0]))] + [None] * max(0, expected - val.shape[0])
                return [None] * expected

            if isinstance(val, (list, tuple)):
                items: List[Optional[torch.Tensor]] = []
                for i in range(expected):
                    if i >= len(val):
                        items.append(None)
                        continue
                    item = val[i]
                    if isinstance(item, torch.Tensor):
                        items.append(item)
                    elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                        items.append(item[0])
                    else:
                        items.append(None)
                return items

            return [None] * expected

        with torch.inference_mode():
            for start in range(0, len(frames_rgb), batch_size):
                end = min(start + batch_size, len(frames_rgb))
                chunk_frames = frames_rgb[start:end]
                chunk_boxes = bboxes_norm[start:end]
                if not chunk_frames:
                    continue

                H, W = chunk_frames[0].shape[:2]
                buf = torch.zeros((len(chunk_frames), H, W, 3), dtype=torch.uint8, device=device)

                for bi, (frame_np, bbox) in enumerate(zip(chunk_frames, chunk_boxes)):
                    if bbox is None or not np.isfinite(bbox).all():
                        continue
                    frame_t = torch.from_numpy(frame_np).to(device=device, dtype=torch.uint8)
                    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
                    # Expand bbox (matches SCAIL-Pose defaults).
                    x1_px = max(0, math.floor(x1 * W - W * 0.025))
                    y1_px = max(0, math.floor(y1 * H - H * 0.05))
                    x2_px = min(W, math.ceil(x2 * W + W * 0.025))
                    y2_px = min(H, math.ceil(y2 * H + H * 0.05))
                    if x2_px <= x1_px or y2_px <= y1_px:
                        continue
                    buf[bi, y1_px:y2_px, x1_px:x2_px, :] = frame_t[y1_px:y2_px, x1_px:x2_px, :]

                img_batch = buf.permute(0, 3, 1, 2)  # (B, 3, H, W)
                pred = _call_model(img_batch)
                joints_items = _extract_joints(pred, expected=len(chunk_frames))

                for bi, joints_t in enumerate(joints_items):
                    if joints_t is None:
                        continue
                    jt = joints_t
                    if jt.ndim == 3 and jt.shape[0] == 1:
                        jt = jt[0]
                    if jt.ndim != 2 or jt.shape[-1] != 3:
                        continue
                    out[start + bi] = jt.detach().float().cpu().numpy().astype(np.float32)

        return out

    def _smpl24_to_openpose18(self, joints24: np.ndarray) -> np.ndarray:
        """
        Map NLFPose SMPL-24 joints to OpenPose/COCO 18-joint order used by SCAIL-Pose rendering.

        Note: this mapping mirrors the one from zai-org/SCAIL-Pose (NLFPoseExtract/nlf_draw.py).
        """
        out = np.full((18, 3), -1.0, dtype=np.float32)
        if joints24 is None or not isinstance(joints24, np.ndarray) or joints24.shape[0] < 22:
            return out
        mapping = {
            15: 0,   # head
            12: 1,   # neck
            17: 2,   # (left shoulder -> openpose R_SHOULDER in upstream mapping)
            19: 3,   # left elbow
            21: 4,   # left hand
            16: 5,   # right shoulder
            18: 6,   # right elbow
            20: 7,   # right hand
            2: 8,    # left pelvis
            5: 9,    # left knee
            8: 10,   # left feet
            1: 11,   # right pelvis
            4: 12,   # right knee
            7: 13,   # right feet
        }
        for src, dst in mapping.items():
            pt = joints24[src]
            if not np.isfinite(pt).all():
                continue
            out[dst] = pt.astype(np.float32)
        # Invalidate non-positive depth (camera Z).
        invalid = out[:, 2] <= 0.0
        out[invalid] = -1.0
        return out

    def extract_3d_keypoints(
        self,
        frames: List[ArrayImage],
        masks: Optional[List[np.ndarray]] = None,
        return_details: bool = False,
    ):
        """
        Extract 3D keypoints from a sequence of frames.

        Args:
            frames: List of input frames
            masks: Optional list of masks for each frame

        Returns:
            List of 3D keypoints arrays, each shape (num_joints, 3)
        """
        # Collect DWPose details first (hands/face + bbox).
        frames_rgb: List[np.ndarray] = []
        details_2d: List[Dict[str, Any]] = []
        bboxes: List[Optional[np.ndarray]] = []

        for i, frame in enumerate(frames):
            mask = masks[i] if masks is not None and i < len(masks) else None
            rgb = _to_rgb_array(frame)
            frames_rgb.append(rgb)
            body_2d, hands_2d, faces_2d, bbox = self.extract_2d_keypoints(rgb, mask)
            details_2d.append({"body_2d": body_2d, "hands_2d": hands_2d, "face_2d": faces_2d})
            bboxes.append(bbox)

        # Run NLFPose (image-based) if available; otherwise fall back to heuristic depth.
        nlf_joints24 = self._run_nlfpose_batched(frames_rgb, bboxes) if self.nlfpose is not None else [None] * len(frames_rgb)

        results: List[Any] = []
        for d2d, joints24 in zip(details_2d, nlf_joints24):
            body_2d = d2d["body_2d"]
            hands_2d = d2d["hands_2d"]
            faces_2d = d2d["face_2d"]

            if joints24 is not None:
                body_3d = self._smpl24_to_openpose18(joints24)
                body_space = "camera"
            elif isinstance(body_2d, np.ndarray) and body_2d.size > 0:
                body_3d = self._estimate_3d_from_2d(body_2d)
                body_space = "normalized"
            else:
                body_3d = np.full((18, 3), -1.0, dtype=np.float32)
                body_space = "normalized"

            if return_details:
                results.append(
                    {
                        "body_3d": body_3d,
                        "body_space": body_space,
                        "hands_2d": hands_2d,
                        "face_2d": faces_2d,
                    }
                )
            else:
                results.append(body_3d)

        return results

    def extract_single(self, image: ArrayImage, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract 3D keypoints from a single image.

        Args:
            image: Input image
            mask: Optional mask

        Returns:
            3D keypoints, shape (num_joints, 3)
        """
        result = self.extract_3d_keypoints([image], [mask] if mask is not None else None)
        return result[0] if result else np.full((18, 3), -1.0)
