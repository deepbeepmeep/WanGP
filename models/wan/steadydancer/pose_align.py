import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union
import itertools
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from preprocessing.dwpose.pose import convert_to_numpy, draw_pose
from preprocessing.dwpose.native import pose_models
from preprocessing.dwpose.wholebody import HWC3, Wholebody, resize_image
from shared.utils.utils import process_images_multithread
from shared.utils.utils import convert_tensor_to_image, convert_image_to_tensor

ArrayImage = Union[np.ndarray, Image.Image, torch.Tensor]


def _to_bgr_image(image: ArrayImage) -> np.ndarray:
    """Convert supported image types to uint8 BGR HWC."""
    if isinstance(image, torch.Tensor):
        # Expect CHW or CTHW in [-1,1] or [0,1] or 0-255
        if image.dim() == 4:
            image = image[:, 0]  # take first frame
        img = image.detach().cpu()
        if img.shape[0] in (1, 3, 4):
            pass
        elif img.shape[-1] in (1, 3, 4):
            img = img.permute(2, 0, 1)
        # normalize to uint8
        if img.min() < 0:
            img = (img + 1.0) * 127.5
        elif img.max() <= 1.0:
            img = img * 255.0
        arr = img.clamp(0, 255).byte().permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return HWC3(arr)
    was_pil = isinstance(image, Image.Image)
    arr = convert_to_numpy(image)

    # Handle CHW tensors
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif was_pil:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return HWC3(arr)


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize while preserving aspect ratio to a specific height."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(round(w * target_h / float(h)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)


def _frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """Stack RGB uint8 frames as a uint8 tensor shaped 3,F,H,W (model_def "uint8_guides": read as the [-1, 1] values of WanGP's guides)."""
    if not frames:
        return torch.empty(0)
    return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)


def _tensor_to_frames(tensor: torch.Tensor) -> List[np.ndarray]:
    """Convert a tensor in [-1, 1] shaped 3,F,H,W back to RGB uint8 frames."""
    if tensor.numel() == 0:
        return []
    arr = tensor.permute(1, 2, 3, 0).cpu().numpy()
    if arr.dtype != np.uint8:
        arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return [frame for frame in arr]


def _ensure_xyz(arr: np.ndarray) -> np.ndarray:
    """Ensure last dimension is 3 by padding zeros if needed."""
    if arr.shape[-1] >= 3:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, 3 - arr.shape[-1])
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def _xy_only(arr: np.ndarray) -> np.ndarray:
    """Return only x,y channels to satisfy drawing utils."""
    if arr.shape[-1] <= 2:
        return arr
    return arr[..., :2]


def _mask_to_float01(mask: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
    """Normalize any mask type to float32 single-channel in [0,1] with 0.5 threshold."""
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    elif isinstance(mask, torch.Tensor):
        m = mask.detach().cpu()
        if m.dim() == 4:
            m = m[0]  # assume 1,C,H,W or C,H,W
        if m.shape[0] in (1, 3, 4):
            m = m.permute(1, 2, 0)
        mask = m.numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[2] > 1:
        mask = mask.mean(axis=2)
    mask = mask.astype(np.float32)
    mask_min, mask_max = float(mask.min()), float(mask.max())
    if mask_max > 1.0:
        mask = mask / 255.0
    elif mask_min < 0.0:
        mask = (mask + 1.0) / 2.0
    mask = np.clip(mask, 0.0, 1.0)
    mask = (mask > 0.5).astype(np.float32)
    return mask


def _safe_ratio(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return 1.0
    val = num / den
    return float(val) if np.isfinite(val) else 1.0


def _nan_to_one(val: float) -> float:
    return 1.0 if not np.isfinite(val) else float(val)


def _augment_pose(
    pose: Dict,
    offset_x: Tuple[float, float],
    offset_y: Tuple[float, float],
    scale_range: Tuple[float, float],
    aspect_ratio_range: Tuple[float, float],
    fixed_params: Tuple[float, float, float, float] | None = None,
) -> Dict:
    """Lightweight pose jitter used by the diff-aug variant."""
    pose_aug = {
        "bodies": {"candidate": pose["bodies"]["candidate"].copy(), "subset": pose["bodies"]["subset"].copy()},
        "hands": pose["hands"].copy(),
        "faces": pose["faces"].copy(),
    }

    if fixed_params is None:
        sx = np.random.uniform(scale_range[0], scale_range[1])
        aspect = np.random.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
        scale_x = sx * aspect
        scale_y = sx / max(aspect, 1e-6)
        dx = np.random.uniform(offset_x[0], offset_x[1])
        dy = np.random.uniform(offset_y[0], offset_y[1])
    else:
        dx, dy, scale_x, scale_y = fixed_params

    def _apply(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        mask = arr[..., 0] >= 0
        arr[..., 0] = np.where(mask, arr[..., 0] * scale_x + dx, arr[..., 0])
        arr[..., 1] = np.where(mask, arr[..., 1] * scale_y + dy, arr[..., 1])
        return arr

    pose_aug["bodies"]["candidate"] = _apply(pose_aug["bodies"]["candidate"])
    pose_aug["hands"] = _apply(pose_aug["hands"])
    pose_aug["faces"] = _apply(pose_aug["faces"])
    return pose_aug


def _save_video(path: str, frames: List[np.ndarray], fps: float) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def align_img(img: np.ndarray, pose_ori: Dict, scales: Dict[str, float]) -> Dict:
    """Align a single pose dictionary using pre-computed scale factors."""
    body_pose = pose_ori["bodies"]["candidate"].copy()
    hands = pose_ori["hands"].copy()
    faces = pose_ori["faces"].copy()

    H_in, W_in, _ = img.shape
    video_ratio = W_in / H_in
    body_pose[:, 0] *= video_ratio
    hands[:, :, 0] *= video_ratio
    faces[:, :, 0] *= video_ratio

    scale_neck = scales["scale_neck"]
    scale_face_left = scales["scale_face_left"]
    scale_face_right = scales["scale_face_right"]
    scale_shoulder = scales["scale_shoulder"]
    scale_arm_upper = scales["scale_arm_upper"]
    scale_arm_lower = scales["scale_arm_lower"]
    scale_hand = scales["scale_hand"]
    scale_body_len = scales["scale_body_len"]
    scale_leg_upper = scales["scale_leg_upper"]
    scale_leg_lower = scales["scale_leg_lower"]

    scale_list = [
        scale_neck,
        scale_face_left,
        scale_face_right,
        scale_shoulder,
        scale_arm_upper,
        scale_arm_lower,
        scale_hand,
        scale_body_len,
        scale_leg_upper,
        scale_leg_lower,
    ]
    finite_vals = [v for v in scale_list if np.isfinite(v)]
    mean_scale = np.mean(finite_vals) if finite_vals else 1.0
    scale_list = [mean_scale if np.isinf(v) else v for v in scale_list]
    (
        scale_neck,
        scale_face_left,
        scale_face_right,
        scale_shoulder,
        scale_arm_upper,
        scale_arm_lower,
        scale_hand,
        scale_body_len,
        scale_leg_upper,
        scale_leg_lower,
    ) = scale_list

    offset = {
        "14_16_to_0": body_pose[[14, 16], :] - body_pose[[0], :],
        "15_17_to_0": body_pose[[15, 17], :] - body_pose[[0], :],
        "3_to_2": body_pose[[3], :] - body_pose[[2], :],
        "4_to_3": body_pose[[4], :] - body_pose[[3], :],
        "6_to_5": body_pose[[6], :] - body_pose[[5], :],
        "7_to_6": body_pose[[7], :] - body_pose[[6], :],
        "9_to_8": body_pose[[9], :] - body_pose[[8], :],
        "10_to_9": body_pose[[10], :] - body_pose[[9], :],
        "12_to_11": body_pose[[12], :] - body_pose[[11], :],
        "13_to_12": body_pose[[13], :] - body_pose[[12], :],
        "hand_left_to_4": hands[1, :, :] - body_pose[[4], :],
        "hand_right_to_7": hands[0, :, :] - body_pose[[7], :],
    }

    def _warp(target: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
        M = cv2.getRotationMatrix2D((center[0], center[1]), 0, scale)
        return warpAffine_kps(target, M)

    body_pose[[0], :] = _warp(body_pose[[0], :], body_pose[1], scale_neck)
    body_pose[[14, 16], :] = _warp(offset["14_16_to_0"] + body_pose[[0], :], body_pose[0], scale_face_left)
    body_pose[[15, 17], :] = _warp(offset["15_17_to_0"] + body_pose[[0], :], body_pose[0], scale_face_right)
    body_pose[[2, 5], :] = _warp(body_pose[[2, 5], :], body_pose[1], scale_shoulder)

    body_pose[[3], :] = _warp(offset["3_to_2"] + body_pose[[2], :], body_pose[2], scale_arm_upper)
    body_pose[[4], :] = _warp(offset["4_to_3"] + body_pose[[3], :], body_pose[3], scale_arm_lower)
    hands[1, :, :] = _warp(offset["hand_left_to_4"] + body_pose[[4], :], body_pose[4], scale_hand)

    body_pose[[6], :] = _warp(offset["6_to_5"] + body_pose[[5], :], body_pose[5], scale_arm_upper)
    body_pose[[7], :] = _warp(offset["7_to_6"] + body_pose[[6], :], body_pose[6], scale_arm_lower)
    hands[0, :, :] = _warp(offset["hand_right_to_7"] + body_pose[[7], :], body_pose[7], scale_hand)

    body_pose[[8, 11], :] = _warp(body_pose[[8, 11], :], body_pose[1], scale_body_len)
    body_pose[[9], :] = _warp(offset["9_to_8"] + body_pose[[8], :], body_pose[8], scale_leg_upper)
    body_pose[[10], :] = _warp(offset["10_to_9"] + body_pose[[9], :], body_pose[9], scale_leg_lower)
    body_pose[[12], :] = _warp(offset["12_to_11"] + body_pose[[11], :], body_pose[11], scale_leg_upper)
    body_pose[[13], :] = _warp(offset["13_to_12"] + body_pose[[12], :], body_pose[12], scale_leg_lower)

    body_pose_none = pose_ori["bodies"]["candidate"] == -1.0
    hands_none = pose_ori["hands"] == -1.0
    faces_none = pose_ori["faces"] == -1.0

    body_pose[body_pose_none] = -1.0
    hands[hands_none] = -1.0
    faces[faces_none] = -1.0

    body_pose = np.nan_to_num(body_pose, nan=-1.0)
    hands = np.nan_to_num(hands, nan=-1.0)
    faces = np.nan_to_num(faces, nan=-1.0)

    pose_align = dict(
        bodies={"candidate": body_pose, "subset": pose_ori["bodies"]["subset"]},
        hands=hands,
        faces=faces,
    )
    return pose_align


def warpAffine_kps(kps: np.ndarray, M: np.ndarray) -> np.ndarray:
    kps_t = kps.copy()
    kps_t[..., 0] = kps[..., 0] * M[0, 0] + kps[..., 1] * M[0, 1] + M[0, 2]
    kps_t[..., 1] = kps[..., 0] * M[1, 0] + kps[..., 1] * M[1, 1] + M[1, 2]
    return kps_t


@dataclass
class PoseDetection:
    pose: Dict
    pose_map_rgb: np.ndarray
    frame_rgb: np.ndarray
    orig_hw: Tuple[int, int]


class PoseAligner:
    def __init__(self, detect_resolution: int = 1024, detection_workers: int = 1) -> None:
        self.pose_estimation = Wholebody()
        self.detection_workers = max(1, int(detection_workers))
        self.detect_resolution = detect_resolution

    def _detect_pose_session(self, image: ArrayImage, pose_estimation: Wholebody) -> PoseDetection | None:
        bgr_orig = _to_bgr_image(image)
        resized = resize_image(bgr_orig, self.detect_resolution)
        H, W, _ = resized.shape
        candidate, subset, _ = pose_estimation(resized)
        if len(candidate) == 0:
            return None

        nums, keys, locs = candidate.shape
        if keys < 18:
            return None  # not enough keypoints detected
        
        candidate = _ensure_xyz(candidate.copy())
        subset = subset.copy()
        
        # Normalize coordinates to [0, 1]
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)

        locs = candidate.shape[2]
        
        # Extract first person's body keypoints (18 keypoints)
        body = candidate[0, :18].copy()  # shape (18, locs)
        body_scores = subset[0, :18].copy()  # shape (18,)
        
        # Build score array and mask invisible keypoints
        score = np.zeros((1, 18), dtype=np.float64)
        for j in range(18):
            if body_scores[j] <= 0.3:
                body[j] = -1
                score[0, j] = -1
            else:
                score[0, j] = j  # Index into body array
        
        # Extract faces (keypoints 24-92)
        faces = candidate[0:1, 24:92].copy()  # shape (1, 68, locs)
        if subset.shape[1] > 24:
            face_scores = subset[0, 24:92]
            for j in range(faces.shape[1]):
                if j < len(face_scores) and face_scores[j] <= 0.3:
                    faces[0, j] = -1
        faces = _ensure_xyz(faces)
        
        # Extract hands (keypoints 92-113 for right hand, 113-134 for left hand)
        # hands array should be shape (2, 21, locs) - [right_hand, left_hand]
        right_hand = candidate[0, 92:113].copy() if candidate.shape[1] > 92 else np.zeros((21, locs))
        left_hand = candidate[0, 113:134].copy() if candidate.shape[1] > 113 else np.zeros((21, locs))
        
        # Mask invalid hand keypoints
        if subset.shape[1] > 92:
            hand_scores = subset[0, 92:134]
            for j in range(21):
                if j < len(hand_scores) and hand_scores[j] <= 0.3:
                    right_hand[j] = -1
                if j + 21 < len(hand_scores) and hand_scores[j + 21] <= 0.3:
                    left_hand[j] = -1
        
        hands = _ensure_xyz(np.stack([right_hand, left_hand]))  # shape (2, 21, locs)
        
        pose = {"bodies": {"candidate": body, "subset": score}, "hands": hands, "faces": faces}

        # Draw expects xy only
        pose_for_draw = {
            "bodies": {"candidate": _xy_only(body), "subset": score},
            "hands": _xy_only(hands),
            "faces": _xy_only(faces),
        }
        pose_map = draw_pose(pose_for_draw, H, W, use_body=True, use_hand=True, use_face=False)
        pose_map_rgb = cv2.cvtColor(pose_map, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return PoseDetection(pose=pose, pose_map_rgb=pose_map_rgb, frame_rgb=frame_rgb, orig_hw=bgr_orig.shape[:2])

    def _detect_pose(self, image: ArrayImage) -> PoseDetection | None:
        return self._detect_pose_session(image, self.pose_estimation)

    def _compute_alignment(self, ref_pose: Dict, first_pose: Dict, ref_ratio: float, video_ratio: float) -> Tuple[Dict, np.ndarray]:
        body_ref = ref_pose["bodies"]["candidate"].copy()
        hands_ref = ref_pose["hands"].copy()
        faces_ref = ref_pose["faces"].copy()

        body_first = first_pose["bodies"]["candidate"].copy()
        hands_first = first_pose["hands"].copy()
        faces_first = first_pose["faces"].copy()

        body_ref[:, 0] *= ref_ratio
        hands_ref[:, :, 0] *= ref_ratio
        faces_ref[:, :, 0] *= ref_ratio

        body_first[:, 0] *= video_ratio
        hands_first[:, :, 0] *= video_ratio
        faces_first[:, :, 0] *= video_ratio

        def dist(body: np.ndarray, a: int, b: int) -> float:
            pa, pb = body[a, :2], body[b, :2]
            if (pa < 0).any() or (pb < 0).any():
                return np.nan
            return float(np.linalg.norm(pa - pb))

        def hand_dist(hand: np.ndarray, idx_a: int, idx_b: int) -> float:
            pa, pb = hand[idx_a, :2], hand[idx_b, :2]
            if (pa < 0).any() or (pb < 0).any():
                return np.nan
            return float(np.linalg.norm(pa - pb))

        align_args = {
            "scale_neck": _safe_ratio(dist(body_ref, 0, 1), dist(body_first, 0, 1)),
            "scale_face_left": _safe_ratio(
                dist(body_ref, 16, 14) + dist(body_ref, 14, 0),
                dist(body_first, 16, 14) + dist(body_first, 14, 0),
            ),
            "scale_face_right": _safe_ratio(
                dist(body_ref, 17, 15) + dist(body_ref, 15, 0),
                dist(body_first, 17, 15) + dist(body_first, 15, 0),
            ),
            "scale_shoulder": _safe_ratio(dist(body_ref, 2, 5), dist(body_first, 2, 5)),
            "scale_arm_upper": np.nanmean(
                [
                    _safe_ratio(dist(body_ref, 2, 3), dist(body_first, 2, 3)),
                    _safe_ratio(dist(body_ref, 5, 6), dist(body_first, 5, 6)),
                ]
            ),
            "scale_arm_lower": np.nanmean(
                [
                    _safe_ratio(dist(body_ref, 3, 4), dist(body_first, 3, 4)),
                    _safe_ratio(dist(body_ref, 6, 7), dist(body_first, 6, 7)),
                ]
            ),
            "scale_body_len": _safe_ratio(
                dist(body_ref, 1, 8) if not np.isnan(dist(body_ref, 1, 8)) else dist(body_ref, 1, 11),
                dist(body_first, 1, 8) if not np.isnan(dist(body_first, 1, 8)) else dist(body_first, 1, 11),
            ),
            "scale_leg_upper": np.nanmean(
                [
                    _safe_ratio(dist(body_ref, 8, 9), dist(body_first, 8, 9)),
                    _safe_ratio(dist(body_ref, 11, 12), dist(body_first, 11, 12)),
                ]
            ),
            "scale_leg_lower": np.nanmean(
                [
                    _safe_ratio(dist(body_ref, 9, 10), dist(body_first, 9, 10)),
                    _safe_ratio(dist(body_ref, 12, 13), dist(body_first, 12, 13)),
                ]
            ),
        }

        hand_pairs = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
        hand_ratios = []
        for idx_a, idx_b in hand_pairs:
            hand_ratios.append(
                _safe_ratio(
                    hand_dist(hands_ref[0], idx_a, idx_b),
                    hand_dist(hands_first[0], idx_a, idx_b),
                )
            )
            hand_ratios.append(
                _safe_ratio(
                    hand_dist(hands_ref[1], idx_a, idx_b),
                    hand_dist(hands_first[1], idx_a, idx_b),
                )
            )
        hand_ratios = [v for v in hand_ratios if np.isfinite(v)]
        align_args["scale_hand"] = np.mean(hand_ratios) if hand_ratios else (
            align_args["scale_arm_upper"] + align_args["scale_arm_lower"]
        ) / 2

        align_args = {k: _nan_to_one(v) for k, v in align_args.items()}
        offset = np.array(
            [body_ref[1, 0] - body_first[1, 0], body_ref[1, 1] - body_first[1, 1], 0.0],
            dtype=np.float32,
        )
        return align_args, offset

    def align(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        with pose_models():
            return self._align(*args, **kwargs)

    def _align(
        self,
        ref_video_frames: List[ArrayImage],
        ref_image: ArrayImage,
        ref_video_mask: List[ArrayImage] | None = None,
        align_frame: int = 0,
        max_frames: int | None = None,
        include_composite: bool = False,
        augment: bool = True,
        augment_mode: str = "per_frame",  # "per_frame" (original jitter), "fixed"
        augment_params: Dict | None = None,
        cpu_resize_workers: int = None,
        resize_ref_video: bool = False,
        detection_chunk_size: int = 8,
        expand_scale = 0,
        verbose: int = 0,
    ) -> Dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        ref_detection = self._detect_pose(ref_image)
        if ref_detection is None:
            raise ValueError("Unable to detect pose in the reference image.")

        target_hw = ref_detection.frame_rgb.shape[:2]

        def _tensor_video_to_list(t: torch.Tensor) -> List[np.ndarray]:
            if t.dim() != 4 or t.shape[0] not in (1, 3, 4):
                raise ValueError("Video tensor must be 4D CTHW or BGRHWC list")
            vid = t.detach().cpu()
            if vid.min() < 0:
                vid = (vid + 1.0) * 127.5
            elif vid.max() <= 1.0:
                vid = vid * 255.0
            vid = vid.clamp(0, 255).byte()
            # Return BGR to match cv2.VideoCapture format
            frames = []
            for i in range(vid.shape[1]):
                rgb = vid[:, i].permute(1, 2, 0).numpy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                frames.append(bgr)
            return frames
        
        if isinstance(ref_video_frames, torch.Tensor):
            ref_video_frames = _tensor_video_to_list(ref_video_frames)
        def _preprocess_item(idx_frame_mask):
            idx, frame, mask = idx_frame_mask
            # normalize mask here to leverage multithreading
            if mask is not None:
                mask = _mask_to_float01(mask)
            if resize_ref_video and frame.shape[:2] != target_hw:
                frame = cv2.resize(frame, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LANCZOS4)

            if mask is not None:
                mask_f = mask
                if mask_f.shape != frame.shape[:2]:
                    mask_f = cv2.resize(mask_f, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                if expand_scale != 0:
                    kernel_size = abs(expand_scale)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    op_expand = cv2.dilate if expand_scale > 0 else cv2.erode
                    mask = op_expand(mask, kernel, iterations=3)
                frame = (frame.astype(np.float32) * mask_f[..., None]).clip(0, 255).astype(np.uint8)
            return idx, frame

        # prepare frames (mask + optional resize) using CPU multithreading
        items = []
        for idx, frame in enumerate(ref_video_frames):
            mask = None
            if ref_video_mask is not None:
                mask =  ref_video_mask[:, idx] if torch.is_tensor(ref_video_mask) else ref_video_mask[idx]
            items.append((idx, frame, mask))
        if cpu_resize_workers is None:
            cpu_resize_workers = max(1, int(os.cpu_count() / 2))
        preprocessed = process_images_multithread(
            _preprocess_item,
            items,
            process_type="prephase",
            wrap_in_list=False,
            max_workers=cpu_resize_workers,
            in_place=False,
        )
        idx_to_frame = {idx: frame for idx, frame in preprocessed}
        t_preprocess = time.perf_counter() - t0
        if verbose:
            print(f"[pose_align] preprocess done in {t_preprocess:.2f}s")

        ref_ratio = ref_detection.frame_rgb.shape[1] / ref_detection.frame_rgb.shape[0]
        pose_list: List[Dict] = []
        video_frames: List[np.ndarray] = [] if include_composite else []
        video_pose_frames: List[np.ndarray] = [] if include_composite else []

        align_args = None
        offset = None

        # instantiate additional detectors for parallel processing
        detectors = [self.pose_estimation]
        for _ in range(max(1, self.detection_workers) - 1):
            detectors.append(Wholebody())

        sequential_done = False
        pending = []
        t_first_stage = 0.0

        for idx in sorted(idx_to_frame.keys()):
            if idx < align_frame:
                continue
            if max_frames is not None and len(pose_list) >= max_frames:
                break

            frame = idx_to_frame[idx]
            t_detect_start = time.perf_counter()
            detection = self._detect_pose(frame) if not sequential_done else None
            if detection is None:
                # queue for parallel path once we have align_args
                pending.append((idx, frame))
                continue

            if align_args is None:
                video_ratio = detection.frame_rgb.shape[1] / detection.frame_rgb.shape[0]
                align_args, offset = self._compute_alignment(ref_detection.pose, detection.pose, ref_ratio, video_ratio)

            pose_aligned = align_img(detection.frame_rgb, detection.pose, align_args)
            mask_body = pose_aligned["bodies"]["candidate"][..., 0] < 0
            mask_hands = pose_aligned["hands"][..., 0] < 0
            mask_faces = pose_aligned["faces"][..., 0] < 0

            pose_aligned["bodies"]["candidate"] = _ensure_xyz(pose_aligned["bodies"]["candidate"]) + offset
            pose_aligned["hands"] = _ensure_xyz(pose_aligned["hands"]) + offset
            pose_aligned["faces"] = _ensure_xyz(pose_aligned["faces"]) + offset

            pose_aligned["bodies"]["candidate"][mask_body] = -1
            pose_aligned["hands"][mask_hands] = -1
            pose_aligned["faces"][mask_faces] = -1

            pose_aligned["bodies"]["candidate"][:, 0] /= ref_ratio
            pose_aligned["hands"][:, :, 0] /= ref_ratio
            pose_aligned["faces"][:, :, 0] /= ref_ratio

            pose_list.append(pose_aligned)
            if include_composite:
                video_frames.append(detection.frame_rgb)
                video_pose_frames.append(detection.pose_map_rgb)

            sequential_done = True
            start_idx = idx + 1
            pending.extend([(j, idx_to_frame[j]) for j in sorted(idx_to_frame.keys()) if j >= start_idx])
            t_first_stage = time.perf_counter() - t_detect_start
            if verbose:
                print(f"[pose_align] first frame detection+align done in {t_first_stage:.2f}s")
            break

        if align_args is None:
            return {"composite": torch.empty(0), "pose_only": torch.empty(0), "pose_aug": torch.empty(0)}

        # Parallel processing for remaining frames using a single persistent executor
        def _process_pending(item):
            idx, frame, det_idx = item
            det = self._detect_pose_session(frame, detectors[det_idx])
            if det is None:
                return idx, None, None, None

            pa = align_img(det.frame_rgb, det.pose, align_args)
            mb = pa["bodies"]["candidate"][..., 0] < 0
            mh = pa["hands"][..., 0] < 0
            mf = pa["faces"][..., 0] < 0

            pa["bodies"]["candidate"] = _ensure_xyz(pa["bodies"]["candidate"]) + offset
            pa["hands"] = _ensure_xyz(pa["hands"]) + offset
            pa["faces"] = _ensure_xyz(pa["faces"]) + offset

            pa["bodies"]["candidate"][mb] = -1
            pa["hands"][mh] = -1
            pa["faces"][mf] = -1

            pa["bodies"]["candidate"][:, 0] /= ref_ratio
            pa["hands"][:, :, 0] /= ref_ratio
            pa["faces"][:, :, 0] /= ref_ratio
            return idx, pa, det.frame_rgb, det.pose_map_rgb

        if pending:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results = []
            det_cycle = itertools.cycle(range(self.detection_workers))
            with ThreadPoolExecutor(max_workers=self.detection_workers) as executor:
                for start in range(0, len(pending), detection_chunk_size * self.detection_workers):
                    chunk = pending[start : start + detection_chunk_size * self.detection_workers]
                    tasks = []
                    for idx, frame in chunk:
                        tasks.append((idx, frame, next(det_cycle)))
                    futures = {executor.submit(_process_pending, t): t[0] for t in tasks}
                    for future in as_completed(futures):
                        res = future.result()
                        if res[1] is None:
                            continue
                        results.append(res)
            for idx, pa, fr, pm in sorted(results, key=lambda x: x[0]):
                if max_frames is not None and len(pose_list) >= max_frames:
                    break
                pose_list.append(pa)
                if include_composite:
                    video_frames.append(fr)
                    video_pose_frames.append(pm)
            t_parallel = time.perf_counter() - t0 - t_preprocess - t_first_stage
            if verbose:
                print(f"[pose_align] parallel detection done in {t_parallel:.2f}s")
        else:
            t_parallel = 0.0

        if not pose_list:
            return {"composite": torch.empty(0), "pose_only": torch.empty(0), "pose_aug": torch.empty(0)}

        body_seq = [_ensure_xyz(pose["bodies"]["candidate"][:18]) for pose in pose_list]
        body_seq_subset = [pose["bodies"]["subset"][:1] for pose in pose_list]
        hands_seq = [_ensure_xyz(pose["hands"][:2]) for pose in pose_list]
        faces_seq = [_ensure_xyz(pose["faces"][:1]) for pose in pose_list]

        ref_H, ref_W = ref_detection.frame_rgb.shape[:2]
        ref_target_H, ref_target_W = ref_detection.orig_hw if hasattr(ref_detection, "orig_hw") else (ref_H, ref_W)
        composite_frames: List[np.ndarray] = [] if include_composite else []
        pose_only_frames: List[np.ndarray] = []
        pose_aug_frames: List[np.ndarray] = []

        aug_cfg = augment_params or {}
        offset_x = aug_cfg.get("offset_x", (-0.2, 0.2))
        offset_y = aug_cfg.get("offset_y", (-0.2, 0.2))
        scale_rng = aug_cfg.get("scale", (0.7, 1.3))
        aspect_rng = aug_cfg.get("aspect_ratio_range", (0.6, 1.4))
        fixed_aug = None
        if augment and augment_mode == "fixed":
            sx = np.random.uniform(scale_rng[0], scale_rng[1])
            aspect = np.random.uniform(aspect_rng[0], aspect_rng[1])
            scale_x = sx * aspect
            scale_y = sx / max(aspect, 1e-6)
            dx = np.random.uniform(offset_x[0], offset_x[1])
            dy = np.random.uniform(offset_y[0], offset_y[1])
            fixed_aug = (dx, dy, scale_x, scale_y)

        for i in range(len(body_seq)):
            pose_t = {
                "bodies": {"candidate": body_seq[i], "subset": body_seq_subset[i]},
                "hands": hands_seq[i],
                "faces": faces_seq[i],
            }

            pose_for_draw = {
                "bodies": {"candidate": _xy_only(body_seq[i]), "subset": body_seq_subset[i]},
                "hands": _xy_only(hands_seq[i]),
                "faces": _xy_only(faces_seq[i]),
            }

            aligned_pose = draw_pose(pose_for_draw, ref_H, ref_W, use_body=True, use_hand=True, use_face=False)
            aligned_pose = cv2.cvtColor(aligned_pose, cv2.COLOR_BGR2RGB)
            pose_only_frames.append(aligned_pose)

            aug_frame = None
            if augment:
                params = None if augment_mode == "per_frame" else fixed_aug
                pose_t_aug = _augment_pose(pose_t, offset_x, offset_y, scale_rng, aspect_rng, fixed_params=params)
                pose_t_aug_draw = {
                    "bodies": {"candidate": _xy_only(pose_t_aug["bodies"]["candidate"]), "subset": pose_t_aug["bodies"]["subset"]},
                    "hands": _xy_only(pose_t_aug["hands"]),
                    "faces": _xy_only(pose_t_aug["faces"]),
                }
                aug_pose = draw_pose(pose_t_aug_draw, ref_H, ref_W, use_body=True, use_hand=True, use_face=False)
                aug_frame = cv2.cvtColor(aug_pose, cv2.COLOR_BGR2RGB)
                pose_aug_frames.append(aug_frame)

            if include_composite:
                video_frame = _resize_to_height(video_frames[i], ref_H)
                video_pose = _resize_to_height(video_pose_frames[i], ref_H)
                ref_pose_resized = _resize_to_height(ref_detection.pose_map_rgb, ref_H)
                ref_img_resized = _resize_to_height(ref_detection.frame_rgb, ref_H)
                parts = [
                    ref_img_resized,
                    ref_pose_resized,
                    aligned_pose,
                ]
                if augment and aug_frame is not None:
                    parts.append(aug_frame)
                parts.extend([video_frame, video_pose])
                comp = np.concatenate(parts, axis=1)
                composite_frames.append(comp)

        # Resize outputs to reference original size if needed
        if (ref_target_H, ref_target_W) != (ref_H, ref_W):
            def _resize_list(frames: List[np.ndarray]) -> List[np.ndarray]:
                return [cv2.resize(f, (ref_target_W, ref_target_H), interpolation=cv2.INTER_CUBIC) for f in frames]
            pose_only_frames = _resize_list(pose_only_frames)
            pose_aug_frames = _resize_list(pose_aug_frames)
        t_render = time.perf_counter() - t0 - t_preprocess - t_first_stage - t_parallel
        t_total = time.perf_counter() - t0
        if verbose:
            print(f"[pose_align] render done in {t_render:.2f}s; total {t_total:.2f}s")

        return {
            "composite": _frames_to_tensor(composite_frames) if include_composite else torch.empty(0),
            "pose_only": _frames_to_tensor(pose_only_frames),
            "pose_aug": _frames_to_tensor(pose_aug_frames) if augment else torch.empty(0),
        }


def load_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()
    return frames, fps


def load_mask_frames(path: str) -> List[np.ndarray]:
    frames, _ = load_video_frames(path)
    return frames


def demo(
    ref_image_path: str,
    ref_video_path: str,
    ref_video_mask_path: str | None,
    output_dir: str,
    include_composite: bool = True,
    augment: bool = True,
    augment_mode: str = "per_frame",
    align_frame: int = 0,
    max_frames: int | None = None,
    detection_workers: int = 2,
    cpu_resize_workers: int = None,
    resize_ref_video: bool = False,
    detection_chunk_size: int = 8,
    verbose: int = 0,
) -> None:
    frames, fps = load_video_frames(ref_video_path)
    mask_frames = load_mask_frames(ref_video_mask_path) if ref_video_mask_path else None
    aligner = PoseAligner(detection_workers=detection_workers)
    outputs = aligner.align(
        frames,
        Image.open(ref_image_path),
        ref_video_mask=mask_frames,
        align_frame=align_frame,
        max_frames=max_frames,
        include_composite=include_composite,
        augment=augment,
        augment_mode=augment_mode,
        cpu_resize_workers=cpu_resize_workers,
        resize_ref_video=resize_ref_video,
        detection_chunk_size=detection_chunk_size,
        verbose=verbose,
    )

    composite_frames = _tensor_to_frames(outputs["composite"])
    pose_frames = _tensor_to_frames(outputs["pose_only"])
    pose_aug_frames = _tensor_to_frames(outputs["pose_aug"])

    if include_composite and composite_frames:
        _save_video(os.path.join(output_dir, "composite.mp4"), composite_frames, fps)
    if pose_frames:
        _save_video(os.path.join(output_dir, "pose_only.mp4"), pose_frames, fps)
    if pose_aug_frames:
        _save_video(os.path.join(output_dir, "pose_only_aug.mp4"), pose_aug_frames, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose alignment demo")
    parser.add_argument("--ref_image", type=str, default=r"e:/steffete.png", help="Path to the reference image")
    parser.add_argument("--ref_video", type=str, default=r"e:\stage.mp4", help="Path to the reference pose video (mp4)")
    parser.add_argument("--ref_video_mask", type=str, default='e:/stagemasked_alpha.mp4', help="Optional path to a mask video (same length as ref video)")
    # parser.add_argument("--ref_video_mask", type=str, default=None, help="Optional path to a mask video (same length as ref video)")
    parser.add_argument("--output_dir", type=str, default="pose_align_demo", help="Directory to store demo mp4s")
    parser.add_argument("--no_composite", action="store_true", help="Skip saving comparison/composite video")
    parser.add_argument("--no_augment", action="store_true", help="Disable augmented pose output")
    parser.add_argument("--augment_mode", type=str, default="per_frame", choices=["fixed", "per_frame"], help="Augmentation mode (fixed per video or per frame jitter)")
    parser.add_argument("--align_frame", type=int, default=0, help="Frame index to start alignment from")
    parser.add_argument("--max_frames", type=int, default=48, help="Maximum frames to process")
    parser.add_argument("--detection_workers", type=int, default=2, help="Number of parallel detection workers (GPU-backed)")
    parser.add_argument("--cpu_resize_workers", type=int, default=None, help="CPU workers for pre-mask/resize")
    parser.add_argument("--resize_ref_video", action="store_true", help="Resize ref video to ref image size (Lanczos)")
    parser.add_argument("--detection_chunk_size", type=int, default=8, help="Chunk size for parallel detection batches")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for timing/debug")
    args = parser.parse_args()

    demo(
        args.ref_image,
        args.ref_video,
        args.ref_video_mask,
        args.output_dir,
        include_composite=not args.no_composite,
        augment=not args.no_augment,
        augment_mode=args.augment_mode,
        align_frame=args.align_frame,
        max_frames=args.max_frames,
        detection_workers=args.detection_workers,
        cpu_resize_workers=args.cpu_resize_workers,
        resize_ref_video=args.resize_ref_video,
        detection_chunk_size=args.detection_chunk_size,
        verbose=args.verbose,
    )
