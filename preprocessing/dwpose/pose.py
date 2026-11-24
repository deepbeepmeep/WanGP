# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

import os
import cv2
import torch
import numpy as np
from . import util
from .wholebody import Wholebody, HWC3, resize_image
from .native import pose_models
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def convert_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
    return image

def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if use_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = util.draw_handpose(canvas, hands)
    if use_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


def _to_uint8_rgb(image):
    image = convert_to_numpy(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[0] != image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape for pose extraction: {image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size:
            if image.min() < 0.0:
                image = (image + 1.0) * 127.5
            elif image.max() <= 1.0:
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _valid_xy(points):
    points = np.asarray(points)
    return np.all(np.isfinite(points), axis=-1) & np.all(points >= 0, axis=-1)


def _safe_ratio(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return 1.0
    val = num / den
    return float(val) if np.isfinite(val) else 1.0


def _nan_to_one(val: float) -> float:
    return 1.0 if not np.isfinite(val) else float(val)


def _pose_point_mask(points):
    points = np.asarray(points)
    return np.all(np.isfinite(points), axis=-1) & ~np.all(points == -1, axis=-1)


def _point_in_unit_frame(point):
    point = np.asarray(point)
    return np.all(np.isfinite(point)) and 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0


def _transform_points(points: np.ndarray, orig_center: np.ndarray, new_center: np.ndarray, scale: float, point_mask=None, orig_center_valid=True, new_center_valid=True) -> np.ndarray:
    out = points.copy()
    if not orig_center_valid or not new_center_valid:
        return out
    if not np.all(np.isfinite(orig_center)) or not np.all(np.isfinite(new_center)):
        return out
    mask = _pose_point_mask(points) if point_mask is None else (np.asarray(point_mask) & np.all(np.isfinite(points), axis=-1))
    if np.any(mask):
        out[mask] = new_center + (points[mask] - orig_center) * scale
    return out


def _scail_face_scale(ref_face: np.ndarray, drive_face: np.ndarray, center_idx: int = 30) -> float:
    if ref_face.shape[0] <= center_idx or drive_face.shape[0] <= center_idx:
        return 1.0
    ref_center = ref_face[center_idx]
    drive_center = drive_face[center_idx]
    if not _valid_xy(ref_center) or not _valid_xy(drive_center):
        return 1.0

    valid = _valid_xy(ref_face) & _valid_xy(drive_face)
    valid[center_idx] = False
    if not np.any(valid):
        return 1.0

    ref_dist = np.linalg.norm(ref_face[valid] - ref_center, axis=1)
    drive_dist = np.linalg.norm(drive_face[valid] - drive_center, axis=1)
    if ref_dist.size == 0 or drive_dist.size == 0:
        return 1.0

    scale = _safe_ratio(float(np.mean(ref_dist)), float(np.mean(drive_dist)))
    return float(np.clip(scale, 0.8, 1.5))


def _body_dist(body: np.ndarray, a: int, b: int) -> float:
    pa, pb = body[a], body[b]
    if not _valid_xy(pa) or not _valid_xy(pb):
        return np.nan
    return float(np.linalg.norm(pa - pb))


def _hand_dist(hand: np.ndarray, idx_a: int, idx_b: int) -> float:
    pa, pb = hand[idx_a], hand[idx_b]
    if not _valid_xy(pa) or not _valid_xy(pb):
        return np.nan
    return float(np.linalg.norm(pa - pb))


def _compute_alignment_scales(ref_pose: dict, drive_pose: dict, ref_ratio: float, drive_ratio: float):
    body_ref = ref_pose["bodies"]["candidate"].copy()
    body_drive = drive_pose["bodies"]["candidate"].copy()
    hands_ref = ref_pose["hands"].copy()
    hands_drive = drive_pose["hands"].copy()
    faces_ref = ref_pose["faces"].copy()
    faces_drive = drive_pose["faces"].copy()

    body_ref[:, 0] *= ref_ratio
    body_drive[:, 0] *= drive_ratio
    hands_ref[:, :, 0] *= ref_ratio
    hands_drive[:, :, 0] *= drive_ratio
    faces_ref[:, :, 0] *= ref_ratio
    faces_drive[:, :, 0] *= drive_ratio

    scales = {
        "scale_neck": _safe_ratio(_body_dist(body_ref, 0, 1), _body_dist(body_drive, 0, 1)),
        "scale_face_left": _safe_ratio(
            _body_dist(body_ref, 16, 14) + _body_dist(body_ref, 14, 0),
            _body_dist(body_drive, 16, 14) + _body_dist(body_drive, 14, 0),
        ),
        "scale_face_right": _safe_ratio(
            _body_dist(body_ref, 17, 15) + _body_dist(body_ref, 15, 0),
            _body_dist(body_drive, 17, 15) + _body_dist(body_drive, 15, 0),
        ),
        "scale_shoulder": _safe_ratio(_body_dist(body_ref, 2, 5), _body_dist(body_drive, 2, 5)),
        "scale_arm_upper": np.nanmean(
            [
                _safe_ratio(_body_dist(body_ref, 2, 3), _body_dist(body_drive, 2, 3)),
                _safe_ratio(_body_dist(body_ref, 5, 6), _body_dist(body_drive, 5, 6)),
            ]
        ),
        "scale_arm_lower": np.nanmean(
            [
                _safe_ratio(_body_dist(body_ref, 3, 4), _body_dist(body_drive, 3, 4)),
                _safe_ratio(_body_dist(body_ref, 6, 7), _body_dist(body_drive, 6, 7)),
            ]
        ),
        "scale_body_len": _safe_ratio(
            _body_dist(body_ref, 1, 8) if not np.isnan(_body_dist(body_ref, 1, 8)) else _body_dist(body_ref, 1, 11),
            _body_dist(body_drive, 1, 8) if not np.isnan(_body_dist(body_drive, 1, 8)) else _body_dist(body_drive, 1, 11),
        ),
        "scale_leg_upper": np.nanmean(
            [
                _safe_ratio(_body_dist(body_ref, 8, 9), _body_dist(body_drive, 8, 9)),
                _safe_ratio(_body_dist(body_ref, 11, 12), _body_dist(body_drive, 11, 12)),
            ]
        ),
        "scale_leg_lower": np.nanmean(
            [
                _safe_ratio(_body_dist(body_ref, 9, 10), _body_dist(body_drive, 9, 10)),
                _safe_ratio(_body_dist(body_ref, 12, 13), _body_dist(body_drive, 12, 13)),
            ]
        ),
        "scale_face": _scail_face_scale(faces_ref[0], faces_drive[0]) if len(faces_ref) and len(faces_drive) else 1.0,
    }

    hand_pairs = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
    hand_ratios = []
    for idx_a, idx_b in hand_pairs:
        if len(hands_ref) > 0 and len(hands_drive) > 0:
            hand_ratios.append(_safe_ratio(_hand_dist(hands_ref[0], idx_a, idx_b), _hand_dist(hands_drive[0], idx_a, idx_b)))
        if len(hands_ref) > 1 and len(hands_drive) > 1:
            hand_ratios.append(_safe_ratio(_hand_dist(hands_ref[1], idx_a, idx_b), _hand_dist(hands_drive[1], idx_a, idx_b)))
    hand_ratios = [v for v in hand_ratios if np.isfinite(v)]
    scales["scale_hand"] = np.mean(hand_ratios) if hand_ratios else (scales["scale_arm_upper"] + scales["scale_arm_lower"]) / 2
    scales = {k: _nan_to_one(v) for k, v in scales.items()}

    ref_neck = body_ref[1]
    drive_neck = body_drive[1]
    offset = ref_neck - drive_neck if _valid_xy(ref_neck) and _valid_xy(drive_neck) else np.zeros(2, dtype=np.float32)
    return scales, offset.astype(np.float32)


def _apply_pose_alignment(pose: dict, scales: dict, offset: np.ndarray, ref_ratio: float, drive_ratio: float):
    body_orig = pose["bodies"]["candidate"].astype(np.float32).copy()
    hands_orig = pose["hands"].astype(np.float32).copy()
    faces_orig = pose["faces"].astype(np.float32).copy()
    body_valid = _valid_xy(body_orig)
    hands_valid = _valid_xy(hands_orig)
    faces_valid = _valid_xy(faces_orig)

    body_orig[:, 0] *= drive_ratio
    hands_orig[:, :, 0] *= drive_ratio
    faces_orig[:, :, 0] *= drive_ratio

    body = body_orig.copy()
    hands = hands_orig.copy()
    faces = faces_orig.copy()

    body[0:1] = _transform_points(body_orig[0:1], body_orig[1], body[1], scales["scale_neck"], point_mask=body_valid[0:1], orig_center_valid=body_valid[1], new_center_valid=body_valid[1])
    body[[14, 16]] = _transform_points(body_orig[[14, 16]], body_orig[0], body[0], scales["scale_face_left"], point_mask=body_valid[[14, 16]], orig_center_valid=body_valid[0], new_center_valid=body_valid[0])
    body[[15, 17]] = _transform_points(body_orig[[15, 17]], body_orig[0], body[0], scales["scale_face_right"], point_mask=body_valid[[15, 17]], orig_center_valid=body_valid[0], new_center_valid=body_valid[0])
    body[[2, 5]] = _transform_points(body_orig[[2, 5]], body_orig[1], body[1], scales["scale_shoulder"], point_mask=body_valid[[2, 5]], orig_center_valid=body_valid[1], new_center_valid=body_valid[1])

    body[[3]] = _transform_points(body_orig[[3]], body_orig[2], body[2], scales["scale_arm_upper"], point_mask=body_valid[[3]], orig_center_valid=body_valid[2], new_center_valid=body_valid[2])
    body[[4]] = _transform_points(body_orig[[4]], body_orig[3], body[3], scales["scale_arm_lower"], point_mask=body_valid[[4]], orig_center_valid=body_valid[3], new_center_valid=body_valid[3])
    hands[1] = _transform_points(hands_orig[1], body_orig[4], body[4], scales["scale_hand"], point_mask=hands_valid[1], orig_center_valid=body_valid[4], new_center_valid=body_valid[4])

    body[[6]] = _transform_points(body_orig[[6]], body_orig[5], body[5], scales["scale_arm_upper"], point_mask=body_valid[[6]], orig_center_valid=body_valid[5], new_center_valid=body_valid[5])
    body[[7]] = _transform_points(body_orig[[7]], body_orig[6], body[6], scales["scale_arm_lower"], point_mask=body_valid[[7]], orig_center_valid=body_valid[6], new_center_valid=body_valid[6])
    hands[0] = _transform_points(hands_orig[0], body_orig[7], body[7], scales["scale_hand"], point_mask=hands_valid[0], orig_center_valid=body_valid[7], new_center_valid=body_valid[7])

    body[[8, 11]] = _transform_points(body_orig[[8, 11]], body_orig[1], body[1], scales["scale_body_len"], point_mask=body_valid[[8, 11]], orig_center_valid=body_valid[1], new_center_valid=body_valid[1])
    body[[9]] = _transform_points(body_orig[[9]], body_orig[8], body[8], scales["scale_leg_upper"], point_mask=body_valid[[9]], orig_center_valid=body_valid[8], new_center_valid=body_valid[8])
    body[[10]] = _transform_points(body_orig[[10]], body_orig[9], body[9], scales["scale_leg_lower"], point_mask=body_valid[[10]], orig_center_valid=body_valid[9], new_center_valid=body_valid[9])
    body[[12]] = _transform_points(body_orig[[12]], body_orig[11], body[11], scales["scale_leg_upper"], point_mask=body_valid[[12]], orig_center_valid=body_valid[11], new_center_valid=body_valid[11])
    body[[13]] = _transform_points(body_orig[[13]], body_orig[12], body[12], scales["scale_leg_lower"], point_mask=body_valid[[13]], orig_center_valid=body_valid[12], new_center_valid=body_valid[12])

    if len(faces):
        face = faces_orig[0]
        if face.shape[0] > 30:
            face_center = face[30]
            drive_nose = body_orig[0]
            aligned_nose = body[0]
            face_center_valid = faces_valid[0, 30] if faces_valid.shape[1] > 30 else False
            if face_center_valid and body_valid[0]:
                new_center = aligned_nose + (face_center - drive_nose) * scales["scale_face"]
                faces[0] = _transform_points(face, face_center, new_center, scales["scale_face"], point_mask=faces_valid[0], orig_center_valid=face_center_valid, new_center_valid=body_valid[0])

    if np.any(body_valid):
        body[body_valid] += offset
        body[..., 0][body_valid] /= max(ref_ratio, 1e-6)
    if np.any(hands_valid):
        hands[hands_valid] += offset
        hands[..., 0][hands_valid] /= max(ref_ratio, 1e-6)
    if np.any(faces_valid):
        faces[faces_valid] += offset
        faces[..., 0][faces_valid] /= max(ref_ratio, 1e-6)

    body[~body_valid] = -1
    hands[~hands_valid] = -1
    faces[~faces_valid] = -1

    for hand_idx, wrist_idx in ((0, 7), (1, 4)):
        if not body_valid[wrist_idx] or not _point_in_unit_frame(body[wrist_idx]):
            hands[hand_idx] = -1

    body = np.nan_to_num(body, nan=-1.0)
    hands = np.nan_to_num(hands, nan=-1.0)
    faces = np.nan_to_num(faces, nan=-1.0)

    return {
        "bodies": {"candidate": body, "subset": pose["bodies"]["subset"].copy()},
        "hands": hands,
        "faces": faces,
    }


def _render_pose_map(pose: dict, render_shape, orig_shape, *, use_face=True):
    render_h, render_w = render_shape
    orig_h, orig_w = orig_shape
    canvas = draw_pose(pose, render_h, render_w, use_hand=True, use_body=True, use_face=use_face)
    interpolation = cv2.INTER_LANCZOS4 if orig_h * orig_w > render_h * render_w else cv2.INTER_AREA
    return cv2.resize(canvas[..., ::-1], (orig_w, orig_h), interpolation=interpolation)


def _render_pose_map_overscan(pose: dict, render_shape, orig_shape, overscan: float, *, use_face=True):
    if overscan <= 1.0:
        return _render_pose_map(pose, render_shape, orig_shape, use_face=use_face)

    render_h, render_w = render_shape
    pad_y = max(0, int(round((overscan - 1.0) * render_h / 2.0)))
    pad_x = max(0, int(round((overscan - 1.0) * render_w / 2.0)))
    expanded_h = render_h + 2 * pad_y
    expanded_w = render_w + 2 * pad_x

    pose_for_draw = {
        "bodies": {
            "candidate": pose["bodies"]["candidate"].copy(),
            "subset": pose["bodies"]["subset"].copy(),
        },
        "hands": pose["hands"].copy(),
        "faces": pose["faces"].copy(),
    }

    def _remap_points(points):
        mask = _pose_point_mask(points)
        if np.any(mask):
            points = points.copy()
            points[mask, 0] = (points[mask, 0] * render_w + pad_x) / expanded_w
            points[mask, 1] = (points[mask, 1] * render_h + pad_y) / expanded_h
        return points

    pose_for_draw["bodies"]["candidate"] = _remap_points(pose_for_draw["bodies"]["candidate"])
    pose_for_draw["hands"] = _remap_points(pose_for_draw["hands"])
    pose_for_draw["faces"] = _remap_points(pose_for_draw["faces"])

    canvas = draw_pose(pose_for_draw, expanded_h, expanded_w, use_hand=True, use_body=True, use_face=use_face)
    canvas = canvas[pad_y:pad_y + render_h, pad_x:pad_x + render_w]
    orig_h, orig_w = orig_shape
    interpolation = cv2.INTER_LANCZOS4 if orig_h * orig_w > render_h * render_w else cv2.INTER_AREA
    return cv2.resize(canvas[..., ::-1], (orig_w, orig_h), interpolation=interpolation)


def _build_single_person_pose(candidate, subset):
    if len(candidate) == 0:
        return None

    candidate = np.asarray(candidate).copy()
    subset = np.asarray(subset).copy()
    if candidate.ndim != 3 or subset.ndim != 2:
        return None

    if candidate.shape[1] == 0:
        return None

    if subset.shape[0] == 0:
        person_idx = 0
    else:
        body_scores = subset[:, :18] if subset.shape[1] >= 18 else subset
        body_scores = np.where(np.isfinite(body_scores), body_scores, -1)
        mean_scores = np.mean(body_scores, axis=1)
        person_idx = int(np.argmax(mean_scores))

    if candidate.shape[1] < 18:
        return None

    visible = np.zeros(candidate.shape[1], dtype=bool)
    if subset.shape[0] > person_idx:
        visible_len = min(candidate.shape[1], subset.shape[1])
        visible[:visible_len] = subset[person_idx, :visible_len] > 0.3
    candidate[person_idx, ~visible] = -1

    body = candidate[person_idx, :18].astype(np.float32).copy()
    subset_out = np.full((1, 18), -1, dtype=np.float32)
    for idx in range(18):
        if _valid_xy(body[idx]):
            subset_out[0, idx] = idx

    faces = np.full((1, 68, body.shape[-1]), -1, dtype=np.float32)
    if candidate.shape[1] >= 92:
        face_slice = candidate[person_idx, 24:92].astype(np.float32).copy()
        faces[0, : face_slice.shape[0]] = face_slice[:68]

    hands = np.full((2, 21, body.shape[-1]), -1, dtype=np.float32)
    if candidate.shape[1] >= 113:
        right_hand = candidate[person_idx, 92:113].astype(np.float32).copy()
        hands[0, : right_hand.shape[0]] = right_hand[:21]
    if candidate.shape[1] >= 134:
        left_hand = candidate[person_idx, 113:134].astype(np.float32).copy()
        hands[1, : left_hand.shape[0]] = left_hand[:21]

    return {"bodies": {"candidate": body, "subset": subset_out}, "hands": hands, "faces": faces}


class PoseAnnotator:
    def __init__(self, cfg):
        self.pose_estimation = Wholebody()
        self.resize_size = cfg.get("RESIZE_SIZE", 1024)
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', True)

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        image = convert_to_numpy(image)
        input_image = HWC3(image[..., ::-1])
        return self.process(resize_image(input_image, self.resize_size), image.shape[:2])

    def process(self, ori_img, ori_shape):
        ori_h, ori_w = ori_shape
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape
        with torch.no_grad():
            candidate, subset, det_result = self.pose_estimation(ori_img)
            
            if len(candidate) == 0:
                # No detections - return empty results
                empty_ret_data = {}
                if self.use_body:
                    empty_ret_data["detected_map_body"] = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                if self.use_face:
                    empty_ret_data["detected_map_face"] = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                if self.use_body and self.use_face:
                    empty_ret_data["detected_map_bodyface"] = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                if self.use_hand and self.use_body and self.use_face:
                    empty_ret_data["detected_map_handbodyface"] = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                return empty_ret_data, np.array([])
            
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]
            faces = candidate[:, 24:92]
            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            ret_data = {}
            if self.use_body:
                detected_map_body = draw_pose(pose, H, W, use_body=True)
                detected_map_body = cv2.resize(detected_map_body[..., ::-1], (ori_w, ori_h),
                                               interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_body"] = detected_map_body

            if self.use_face:
                detected_map_face = draw_pose(pose, H, W, use_face=True)
                detected_map_face = cv2.resize(detected_map_face[..., ::-1], (ori_w, ori_h),
                                               interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_face"] = detected_map_face

            if self.use_body and self.use_face:
                detected_map_bodyface = draw_pose(pose, H, W, use_body=True, use_face=True)
                detected_map_bodyface = cv2.resize(detected_map_bodyface[..., ::-1], (ori_w, ori_h),
                                                   interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_bodyface"] = detected_map_bodyface

            if self.use_hand and self.use_body and self.use_face:
                detected_map_handbodyface = draw_pose(pose, H, W, use_hand=True, use_body=True, use_face=True)
                detected_map_handbodyface = cv2.resize(detected_map_handbodyface[..., ::-1], (ori_w, ori_h),
                                                       interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_handbodyface"] = detected_map_handbodyface

            # convert_size
            if det_result.shape[0] > 0:
                w_ratio, h_ratio = ori_w / W, ori_h / H
                det_result[..., ::2] *= h_ratio
                det_result[..., 1::2] *= w_ratio
                det_result = det_result.astype(np.int32)
            return ret_data, det_result


class OptimizedPoseAnnotator(PoseAnnotator):
    """Kept name: the annotators of the video classes."""


class PoseBodyFaceAnnotator(PoseAnnotator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_body, self.use_face, self.use_hand = True, True, False
    
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        ret_data, det_result = super().forward(image)
        return ret_data['detected_map_bodyface']


class OptimizedPoseBodyFaceVideoAnnotator:
    """Hands, body and face map of each frame. Frames run one after the other in the calling thread: worker threads only slow the
    GPU pose models down (per-thread CUDA handles, and the drawing code holds the GIL between their kernel launches)."""
    def __init__(self, cfg):
        self.annotator = OptimizedPoseAnnotator(cfg)
        self.use_body, self.use_face, self.use_hand = True, True, True

    def _process_frame(self, frame):
        frame = convert_to_numpy(frame)
        ret_data, _ = self.annotator.process(resize_image(HWC3(frame[..., ::-1]), self.annotator.resize_size), frame.shape[:2])
        return ret_data.get('detected_map_handbodyface', np.zeros((*frame.shape[:2], 3), dtype=np.uint8))

    def forward(self, frames):
        with pose_models():
            return [self._process_frame(frame) for frame in frames]


class OptimizedPoseBodyFaceHandVideoAnnotator(OptimizedPoseBodyFaceVideoAnnotator):
    """Video annotator that includes hands, body, and face"""


class AlignedPoseBodyFaceVideoAnnotator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ref_image = cfg.get("REF_IMAGE")
        self.resize_size = cfg.get("RESIZE_SIZE", 1024)
        self.render_overscan = max(1.0, float(cfg.get("ALIGN_RENDER_OVERSCAN", 2.0)))
        self.annotator = OptimizedPoseAnnotator(cfg) if self.ref_image is not None else None
        self._fallback = None

    def _fallback_forward(self, frames):
        if self._fallback is None:
            self._fallback = OptimizedPoseBodyFaceVideoAnnotator(self.cfg)
        return self._fallback.forward(frames)

    def _detect_pose(self, frame):
        try:
            frame_rgb = _to_uint8_rgb(frame)
            input_image = HWC3(frame_rgb[..., ::-1])
            resized_image = resize_image(input_image, self.resize_size)
            render_shape = resized_image.shape[:2]

            candidate, subset, _ = self.annotator.pose_estimation(resized_image)
            pose = _build_single_person_pose(candidate, subset)
            if pose is None:
                return None

            render_h, render_w = render_shape
            pose["bodies"]["candidate"][:, 0] /= float(render_w)
            pose["bodies"]["candidate"][:, 1] /= float(render_h)
            pose["hands"][:, :, 0] /= float(render_w)
            pose["hands"][:, :, 1] /= float(render_h)
            pose["faces"][:, :, 0] /= float(render_w)
            pose["faces"][:, :, 1] /= float(render_h)
            return {
                "pose": pose,
                "orig_shape": frame_rgb.shape[:2],
                "render_shape": render_shape,
            }
        except Exception as e:
            print(f"Error aligning pose frame: {e}")
            return None

    def forward(self, frames):
        if len(frames) == 0:
            return []
        with pose_models():
            return self._forward(frames)

    def _forward(self, frames):
        if self.ref_image is None:
            return self._fallback_forward(frames)

        try:
            first_frame_rgb = _to_uint8_rgb(frames[0])
        except Exception:
            return self._fallback_forward(frames)

        ref_rgb = _to_uint8_rgb(self.ref_image)
        if ref_rgb.shape[:2] != first_frame_rgb.shape[:2]:
            ref_rgb = cv2.resize(ref_rgb, (first_frame_rgb.shape[1], first_frame_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        ref_detection = self._detect_pose(ref_rgb)
        if ref_detection is None:
            return self._fallback_forward(frames)

        detections = [None] * len(frames)
        first_pose_idx = None
        for frame_idx, frame in enumerate(frames):
            detection = self._detect_pose(frame)
            detections[frame_idx] = detection
            if detection is not None:
                first_pose_idx = frame_idx
                break

        if first_pose_idx is None:
            return self._fallback_forward(frames)

        first_detection = detections[first_pose_idx]
        ref_ratio = ref_detection["render_shape"][1] / max(ref_detection["render_shape"][0], 1)
        drive_ratio = first_detection["render_shape"][1] / max(first_detection["render_shape"][0], 1)
        scales, offset = _compute_alignment_scales(ref_detection["pose"], first_detection["pose"], ref_ratio, drive_ratio)

        ret_frames = []
        for frame_idx, frame in enumerate(frames):
            detection = detections[frame_idx]
            if detection is None and frame_idx > first_pose_idx:
                detection = self._detect_pose(frame)
                detections[frame_idx] = detection

            if detection is None:
                frame_rgb = _to_uint8_rgb(frame)
                ret_frames.append(np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 3), dtype=np.uint8))
                continue

            cur_ratio = detection["render_shape"][1] / max(detection["render_shape"][0], 1)
            aligned_pose = _apply_pose_alignment(detection["pose"], scales, offset, ref_ratio, cur_ratio)
            ret_frames.append(_render_pose_map_overscan(aligned_pose, detection["render_shape"], detection["orig_shape"], self.render_overscan, use_face=True))
        return ret_frames


# Choose which version you want to use:

# Option 1: Body + Face only (original behavior)
class PoseBodyFaceVideoAnnotator(AlignedPoseBodyFaceVideoAnnotator):
    """Backward compatible class name - Body and Face only"""
# Option 2: Body + Face + Hands (if you want hands)
class PoseBodyFaceHandVideoAnnotator(OptimizedPoseBodyFaceHandVideoAnnotator):
    """Video annotator with hands, body, and face"""


# Keep the existing utility functions
import imageio

def save_one_video(file_path, videos, fps=8, quality=8, macro_block_size=None):
    try:
        video_writer = imageio.get_writer(file_path, fps=fps, codec='libx264', quality=quality, macro_block_size=macro_block_size)
        for frame in videos:
            video_writer.append_data(frame)
        video_writer.close()
        return True
    except Exception as e:
        print(f"Video save error: {e}")
        return False
    
def get_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return frames, fps
