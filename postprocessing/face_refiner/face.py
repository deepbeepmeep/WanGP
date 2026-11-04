"""Face tracking, normalized crops, and stitch-back for video refinement.

Adapted from Carasibana/ComfyUI-H3-FaceRefine at commit
79a97ce5ee4b393ce26313bd1280b706fe8b4f2c (MIT).
"""

from __future__ import annotations

import gc
import heapq
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from shared.utils.utils import get_default_workers, process_images_multithread


def sample_to_frames(sample: torch.Tensor) -> tuple[torch.Tensor, str]:
    if sample.ndim != 4 or sample.shape[0] != 3:
        raise ValueError(f"Face Refiner expects a CTHW video tensor, got {tuple(sample.shape)}")
    if sample.dtype == torch.uint8:
        return sample.permute(1, 2, 3, 0).cpu().contiguous(), "uint8"
    if not torch.is_floating_point(sample):
        raise ValueError(f"Face Refiner expects uint8 or floating-point video, got {sample.dtype}")
    frames = torch.empty((sample.shape[1], sample.shape[2], sample.shape[3], 3), dtype=torch.uint8, device="cpu")
    for index in range(sample.shape[1]):
        frame = sample[:, index].to(dtype=torch.float32, device="cpu", copy=True).clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().byte().permute(1, 2, 0)
        frames[index].copy_(frame)
    return frames, "signed"


def frames_to_sample(frames: torch.Tensor, source_format: str, source_dtype: torch.dtype) -> torch.Tensor:
    if source_format == "uint8":
        return frames.permute(3, 0, 1, 2).contiguous()
    sample = torch.empty((3, frames.shape[0], frames.shape[1], frames.shape[2]), dtype=source_dtype, device="cpu")
    for index in range(frames.shape[0]):
        frame = frames[index].permute(2, 0, 1).float().div_(127.5).sub_(1.0).to(source_dtype)
        sample[:, index].copy_(frame)
    return sample


def _bgr(frame) -> np.ndarray:
    if torch.is_tensor(frame):
        frame = frame[..., :3].cpu().numpy() if frame.dtype == torch.uint8 else frame[..., :3].clamp(0, 1).mul(255).byte().cpu().numpy()
    else:
        frame = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    return frame[..., ::-1].copy()


def _smooth(values: np.ndarray, window: int, method: str = "gaussian") -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    window = min(int(window), len(values))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode="reflect")
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter

            polyorder = 2 if window > 3 else 1
            return np.asarray(savgol_filter(padded, window, polyorder))[pad:pad + len(values)]
        except Exception:
            method = "gaussian"
    if method == "gaussian":
        x = np.arange(window, dtype=np.float64) - pad
        sigma = max(window / 6.0, 0.5)
        kernel = np.exp(-(x**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()
    else:
        kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def _interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.arange(len(values))
    return np.interp(indices, indices[valid], values[valid]) if valid.any() else np.zeros(len(values), dtype=np.float64)


def _track_presence(valid: np.ndarray, max_gap: int = 12) -> np.ndarray:
    active = valid.copy()
    indices = np.flatnonzero(valid)
    for left, right in zip(indices[:-1], indices[1:]):
        if right - left - 1 <= int(max_gap):
            active[left:right + 1] = True
    return active


def _track_segments(active: np.ndarray, boxes, max_continuity: float = 1.0) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    active = active.copy()
    cuts = []
    discontinuities = 0
    indices = [index for index, box in enumerate(boxes) if box is not None]
    for left, right in zip(indices[:-1], indices[1:]):
        if not active[left:right + 1].all() or _normalized_continuity_cost(boxes[right], boxes[left]) < float(max_continuity):
            continue
        discontinuities += 1
        if right - left > 1:
            active[left + 1:right] = False
        else:
            cuts.append(right)
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    presence = list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))
    segments = []
    for start, stop in presence:
        boundaries = [start, *(cut for cut in cuts if start < cut < stop), stop]
        segments.extend(zip(boundaries[:-1], boundaries[1:]))
    return active, segments, discontinuities


def _interpolate_track(values: np.ndarray, valid: np.ndarray, active: np.ndarray) -> np.ndarray:
    result = _interpolate(values, valid)
    indices = np.flatnonzero(valid)
    for left, right in zip(indices[:-1], indices[1:]):
        if active[left + 1:right].any():
            continue
        midpoint = (left + right) // 2
        result[left + 1:midpoint + 1] = values[left]
        result[midpoint + 1:right] = values[right]
    return result


def _smooth_track(values: np.ndarray, segments: list[tuple[int, int]], window: int, method: str) -> np.ndarray:
    result = values.copy()
    for start, stop in segments:
        result[start:stop] = _smooth(values[start:stop], window, method)
    return result


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _continuity_cost(box, previous) -> float:
    cx, cy, size = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, box[3] - box[1]
    return math.hypot(cx - previous[0], cy - previous[1]) + abs(size - previous[2]) * 2.0


def _estimate_insightface_norm(lmk, image_size=112, mode="arcface"):
    from insightface.utils import face_align

    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    ratio = image_size / (112.0 if image_size % 112 == 0 else 128.0)
    dst = face_align.arcface_dst * ratio
    if image_size % 112 != 0:
        dst[:, 0] += 8.0 * ratio
    transform = face_align.trans.SimilarityTransform
    if hasattr(transform, "from_estimate"):
        return transform.from_estimate(lmk, dst).params[:2]
    legacy = transform()
    legacy.estimate(lmk, dst)
    return legacy.params[:2]


def _insightface_app(allowed_modules=None, model_dir=None):
    import onnxruntime
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
    from insightface.utils import face_align

    available = set(onnxruntime.get_available_providers())
    providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
    onnxruntime.set_default_logger_severity(3)
    face_align.estimate_norm = _estimate_insightface_norm
    required = {"det_10g.onnx": "detection", "2d106det.onnx": "landmark_2d_106", "w600k_r50.onnx": "recognition"}
    missing = [filename for filename in required if not os.path.isfile(os.path.join(model_dir, filename))]
    if missing:
        raise FileNotFoundError(f"Missing WanGP-managed InsightFace assets: {', '.join(missing)}")
    allowed_modules = allowed_modules or ["detection", "recognition"]
    app = FaceAnalysis.__new__(FaceAnalysis)
    app.models = {}
    app.model_dir = model_dir
    for filename, expected_task in required.items():
        if expected_task not in allowed_modules:
            continue
        model = model_zoo.get_model(os.path.join(model_dir, filename), providers=providers)
        if model is None or model.taskname != expected_task:
            raise RuntimeError(f"Unexpected InsightFace model in {filename}")
        app.models[model.taskname] = model
    if "detection" not in app.models:
        raise RuntimeError("WanGP-managed InsightFace assets do not contain a face detector")
    app.det_model = app.models["detection"]
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))
    return app


def _embeddings(app, frame: torch.Tensor):
    found = []
    for face in app.get(_bgr(frame)):
        embedding = getattr(face, "normed_embedding", None)
        if embedding is not None:
            found.append((face.bbox.tolist(), np.asarray(embedding, dtype=np.float32)))
    return found


def _release_insightface(app) -> bool:
    if app is None:
        return False
    models = list(app.models.values())
    app.det_model = None
    app.models.clear()
    for model in models:
        model.session = None
        center_cache = getattr(model, "center_cache", None)
        if center_cache is not None:
            center_cache.clear()
    models.clear()
    return True


def select_reference_frame(frames, face_boxes, max_candidates=120, insightface_model_dir=None) -> int:
    import cv2

    detected = [(index, box) for index, box in enumerate(face_boxes) if box is not None]
    if not detected:
        raise ValueError("Cannot select a reference frame without a detected face")
    largest_index, largest_box = max(detected, key=lambda item: item[1][3] - item[1][1])
    largest_height = largest_box[3] - largest_box[1]
    candidates = [(index, box) for index, box in detected if box[3] - box[1] >= largest_height * 0.5]
    if len(candidates) > max_candidates:
        candidates = [candidates[index] for index in np.linspace(0, len(candidates) - 1, max_candidates).round().astype(int)]

    app = None
    observations = []
    try:
        app = _insightface_app(["detection", "landmark_2d_106"], insightface_model_dir)
        for index, box in candidates:
            faces = app.get(_bgr(frames[index]))
            if not faces:
                continue
            matched = max(faces, key=lambda item: _iou(item.bbox, box))
            overlap = _iou(matched.bbox, box)
            if overlap < 0.3:
                continue
            keypoints = np.asarray(matched.kps, dtype=np.float64)
            landmarks = np.asarray(matched.landmark_2d_106, dtype=np.float64)
            eye_vector = keypoints[1] - keypoints[0]
            eye_distance = float(np.linalg.norm(eye_vector))
            if eye_distance < 1e-6:
                continue
            eye_midpoint = (keypoints[0] + keypoints[1]) * 0.5
            yaw = abs(float((keypoints[2, 0] - eye_midpoint[0]) / eye_distance))
            roll = abs(math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0]))))
            mouth_open = abs(float(landmarks[53, 1] - landmarks[62, 1])) / eye_distance
            x0, y0, x1, y1 = np.round(matched.bbox).astype(int)
            face_crop = _bgr(frames[index])[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
            sharpness = float(cv2.Laplacian(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            observations.append({"index": index, "height": box[3] - box[1], "yaw": yaw, "roll": roll,
                                 "mouth": mouth_open, "sharpness": sharpness, "confidence": float(matched.det_score)})
    except Exception as error:
        print(f"[FaceRefine] frontal reference selection unavailable ({error}); using largest detected face")
    finally:
        released = _release_insightface(app)
        app = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if released:
            print("[FaceRefine] Unloaded InsightFace ONNX sessions before decoding")

    if not observations:
        return largest_index
    sharpness = np.asarray([item["sharpness"] for item in observations])
    sharp_low, sharp_high = np.percentile(sharpness, (10, 90))
    for item in observations:
        sharp_score = np.clip((item["sharpness"] - sharp_low) / max(sharp_high - sharp_low, 1e-6), 0.0, 1.0)
        item["score"] = (item["height"] * math.exp(-0.5 * (item["yaw"] / 0.08) ** 2)
                         * math.exp(-0.5 * (item["roll"] / 10.0) ** 2)
                         * math.exp(-0.5 * (item["mouth"] / 0.20) ** 2)
                         * (0.9 + 0.1 * sharp_score) * item["confidence"])
    best = max(observations, key=lambda item: item["score"])
    print(f"[FaceRefine] frontal reference frame {best['index'] + 1}: face={best['height']:.0f}px yaw-proxy={best['yaw']:.3f} mouth={best['mouth']:.3f}")
    return int(best["index"])


def _clip_anchor(app, frames, detections, max_samples=24):
    embeddings = []
    step = max(1, len(frames) // max_samples)
    for index in range(0, len(frames), step):
        boxes = detections[index]
        if not boxes:
            continue
        heights = sorted((box[3] - box[1] for box in boxes), reverse=True)
        if len(heights) > 1 and heights[0] < heights[1] * 1.6:
            continue
        candidates = _embeddings(app, frames[index])
        if candidates:
            embeddings.append(max(candidates, key=lambda item: item[0][3] - item[0][1])[1])
    if not embeddings:
        return None, 0
    anchor = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(anchor)
    return (anchor / norm if norm > 0 else anchor), len(embeddings)


def _release_detector(model):
    if model is None:
        return
    inner = getattr(model, "model", None)
    if inner is not None:
        inner.to("cpu")
    model.predictor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_yolo(path):
    from ultralytics import YOLO

    if not str(path).lower().endswith(".safetensors"):
        return YOLO(path)

    from safetensors import safe_open
    from safetensors.torch import load_file
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import callbacks

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata.get("format") != "ultralytics_yolov8_segmentation" or metadata.get("task") != "segment":
        raise ValueError(f"Unsupported safe Ultralytics checkpoint: {path}")
    config = json.loads(metadata["config"])
    network = SegmentationModel(cfg=config, ch=int(config.get("ch", 3)), nc=int(config["nc"]), verbose=False)
    network.load_state_dict(load_file(path, device="cpu"), strict=True)
    network.names = {int(key): value for key, value in json.loads(metadata["names"]).items()}
    network.args = {**json.loads(metadata["args"]), "task": "segment", "model": str(path)}
    network.task = "segment"
    network.pt_path = str(path)
    network.eval()

    model = YOLO.__new__(YOLO)
    torch.nn.Module.__init__(model)
    model.callbacks = callbacks.get_default_callbacks()
    model.predictor = model.trainer = model.metrics = None
    model.model = network
    model.ckpt = {}
    model.cfg = None
    model.ckpt_path = str(path)
    model.task = "segment"
    model.overrides = model._reset_ckpt_args(network.args)
    model.overrides.update(model=str(path), task="segment")
    model.model_name = str(path)
    return model


def _detect(frames, detector_path, confidence, abort_callback=None, progress_callback=None):
    model = _load_yolo(detector_path)
    detections = []
    try:
        for index, frame in enumerate(frames):
            if callable(abort_callback) and abort_callback():
                return None
            result = model.predict(_bgr(frame), conf=float(confidence), verbose=False)[0]
            detections.append([[float(value) for value in box] for box in result.boxes.xyxy.tolist()] if len(result.boxes) else [])
            if callable(progress_callback):
                progress_callback("Detecting and tracking faces", index + 1, len(frames))
    finally:
        _release_detector(model)
    return detections


def _select_track(frames, detections, identity_track=True, identity_threshold=0.28, identity_reference=None, select="largest", insightface_model_dir=None):
    use_identity = identity_track and bool(detections and (len(detections[0]) > 1 or identity_reference is not None))
    app = anchor = None
    identity_matches = continuity_matches = conflicts = 0
    if use_identity:
        try:
            app = _insightface_app(model_dir=insightface_model_dir)
            if identity_reference is not None:
                candidates = _embeddings(app, identity_reference)
                if candidates:
                    anchor = max(candidates, key=lambda item: item[0][3] - item[0][1])[1]
                    print("[FaceRefine] identity anchor from the supplied reference")
            if anchor is None:
                anchor, anchor_frames = _clip_anchor(app, frames, detections)
                if anchor is not None:
                    print(f"[FaceRefine] identity anchor built from the clip itself ({anchor_frames} unambiguous frames)")
        except Exception as error:
            print(f"[FaceRefine] identity matching unavailable ({error})")
            anchor = None

    selected, previous = [], None
    frame_h, frame_w = frames.shape[1:3]
    try:
        for index, boxes in enumerate(detections):
            if not boxes:
                selected.append(None)
                continue
            box = None
            if len(boxes) == 1:
                box = boxes[0]
                continuity_matches += 1
            elif previous is None:
                if anchor is not None:
                    candidates = _embeddings(app, frames[index])
                    if candidates:
                        similarities = [float(np.dot(embedding, anchor)) for _, embedding in candidates]
                        box = candidates[int(np.argmax(similarities))][0]
                        identity_matches += 1
                if box is None:
                    if select == "most_central":
                        box = min(boxes, key=lambda item: ((item[0] + item[2]) / 2.0 - frame_w / 2.0) ** 2 + ((item[1] + item[3]) / 2.0 - frame_h / 2.0) ** 2)
                    else:
                        box = max(boxes, key=lambda item: item[3] - item[1])
            else:
                ranked = sorted(boxes, key=lambda item: _continuity_cost(item, previous))
                best, second = ranked[:2]
                first_cost, second_cost = _continuity_cost(best, previous), _continuity_cost(second, previous)
                conflict = second_cost < first_cost * 2.0 or _iou(best, second) > 0.2
                if conflict and anchor is not None:
                    conflicts += 1
                    nearby = [item for item in boxes if _continuity_cost(item, previous) < first_cost * 3.0] or boxes
                    candidates = [candidate for candidate in _embeddings(app, frames[index]) if any(_iou(candidate[0], item) > 0.3 for item in nearby)]
                    if candidates:
                        similarities = [float(np.dot(embedding, anchor)) for _, embedding in candidates]
                        match = int(np.argmax(similarities))
                        if similarities[match] >= identity_threshold:
                            box = candidates[match][0]
                            identity_matches += 1
                if box is None:
                    box = best
                    continuity_matches += 1
            previous = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, box[3] - box[1])
            selected.append(box)
    finally:
        released = _release_insightface(app)
        app = anchor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if released:
            print("[FaceRefine] Unloaded InsightFace ONNX sessions before refinement")
    return selected, identity_matches, continuity_matches, conflicts


def _aligned_embeddings(app, frame, boxes):
    candidates = _embeddings(app, frame)
    aligned, used = [None] * len(boxes), set()
    for box_index, box in enumerate(boxes):
        matches = sorted(((float(_iou(box, candidate_box)), index, embedding) for index, (candidate_box, embedding) in enumerate(candidates) if index not in used), reverse=True)
        if matches and matches[0][0] >= 0.3:
            _overlap, candidate_index, embedding = matches[0]
            aligned[box_index] = embedding
            used.add(candidate_index)
    return aligned


def _track_anchor(track):
    embeddings = track["embeddings"]
    if not embeddings:
        return None
    anchor = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(anchor)
    return anchor / norm if norm > 0 else None


def _track_similarity(first, second):
    if not first["embeddings"] or not second["embeddings"]:
        return None
    first_embeddings = np.stack(first["embeddings"])
    second_embeddings = np.stack(second["embeddings"])
    if len(first_embeddings) > 64:
        first_embeddings = first_embeddings[np.linspace(0, len(first_embeddings) - 1, 64, dtype=int)]
    if len(second_embeddings) > 64:
        second_embeddings = second_embeddings[np.linspace(0, len(second_embeddings) - 1, 64, dtype=int)]
    similarities = (first_embeddings @ second_embeddings.T).ravel()
    top_count = min(20, len(similarities))
    return float(np.partition(similarities, -top_count)[-top_count:].mean())


def _normalized_continuity_cost(box, previous_box) -> float:
    cx, cy, size = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, max(box[3] - box[1], 1.0)
    previous_cx, previous_cy = (previous_box[0] + previous_box[2]) * 0.5, (previous_box[1] + previous_box[3]) * 0.5
    previous_size = max(previous_box[3] - previous_box[1], 1.0)
    return math.hypot(cx - previous_cx, cy - previous_cy) / max(size, previous_size) + abs(math.log(size / previous_size)) * 2.0


def _shot_boundaries(frames, threshold=0.3):
    if len(frames) < 2:
        return set()
    reduced = F.interpolate(frames[..., :3].movedim(-1, 1).float(), size=(36, 64), mode="area")
    if frames.dtype == torch.uint8:
        reduced.div_(255.0)
    changes = (reduced[1:] - reduced[:-1]).abs().mean(dim=(1, 2, 3))
    return {index + 1 for index in torch.nonzero(changes >= float(threshold), as_tuple=False).flatten().tolist()}


def _flow_missing_boxes(frames, boxes, max_gap=18):
    import cv2

    result = list(boxes)
    propagated = np.zeros(len(result), dtype=bool)
    shot_boundaries = _shot_boundaries(frames)
    for start, box in enumerate(result):
        if box is None or start + 1 >= len(result) or result[start + 1] is not None:
            continue
        if any(item is not None for item in result[start + 1:]):
            continue
        stop = start + 1
        while stop < len(result) and result[stop] is None and stop - start <= int(max_gap) and stop not in shot_boundaries:
            stop += 1
        previous_gray = cv2.cvtColor(_bgr(frames[start]), cv2.COLOR_BGR2GRAY)
        previous_box = np.asarray(box, dtype=np.float32)
        height = max(float(previous_box[3] - previous_box[1]), 1.0)
        if previous_box[2] - previous_box[0] < height * 0.65:
            center_x = float(previous_box[0] + previous_box[2]) * 0.5
            previous_box[[0, 2]] = center_x + np.asarray([-0.325, 0.325], dtype=np.float32) * height
        history = [item for item in result[max(0, start - 6):start + 1] if item is not None]
        centers = np.asarray([((item[0] + item[2]) * 0.5, (item[1] + item[3]) * 0.5) for item in history], dtype=np.float32)
        velocity = np.median(np.diff(centers, axis=0), axis=0) if len(centers) > 1 else np.zeros(2, dtype=np.float32)
        mask = np.zeros_like(previous_gray)
        x0, y0, x1, y1 = previous_box.round().astype(int)
        mask[max(0, y0):min(mask.shape[0], y1), max(0, x0):min(mask.shape[1], x1)] = 255
        points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=80, qualityLevel=0.01, minDistance=3, mask=mask)
        for index in range(start + 1, stop):
            current_gray = cv2.cvtColor(_bgr(frames[index]), cv2.COLOR_BGR2GRAY)
            current_points = np.empty((0, 2), dtype=np.float32)
            motion = velocity
            if points is not None:
                moved, status, _error = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, winSize=(21, 21), maxLevel=3)
                if moved is not None:
                    good = status.reshape(-1).astype(bool)
                    previous_points, current_points = points.reshape(-1, 2)[good], moved.reshape(-1, 2)[good]
                    if len(current_points) >= 4:
                        flow_motion = np.median(current_points - previous_points, axis=0)
                        if float(np.linalg.norm(flow_motion)) <= height * 0.5:
                            motion = flow_motion
            height = max(float(previous_box[3] - previous_box[1]), 1.0)
            if float(np.linalg.norm(motion)) > height * 0.5:
                motion = velocity * (height * 0.5 / max(float(np.linalg.norm(velocity)), 1e-6))
            previous_box = previous_box + np.tile(motion, 2)
            result[index] = previous_box.tolist()
            propagated[index] = True
            velocity = velocity * 0.75 + motion * 0.25
            previous_gray = current_gray
            if len(current_points) >= 8:
                points = current_points.reshape(-1, 1, 2)
            else:
                mask.fill(0)
                x0, y0, x1, y1 = previous_box.round().astype(int)
                mask[max(0, y0):min(mask.shape[0], y1), max(0, x0):min(mask.shape[1], x1)] = 255
                points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=80, qualityLevel=0.01, minDistance=3, mask=mask)
    return result, propagated


def _associate_tracks(detections, frame_embeddings, identity_threshold=0.28, shot_boundaries=(), progress_callback=None):
    frame_count = len(detections)
    shot_boundaries = set(shot_boundaries)
    shot_index = 0
    tracks = []
    for frame_index, boxes in enumerate(detections):
        if frame_index in shot_boundaries:
            shot_index += 1
        embeddings = frame_embeddings[frame_index]
        pairs = []
        for track_index, track in enumerate(tracks):
            gap = frame_index - track["last_frame"]
            anchor = _track_anchor(track)
            for box_index, (box, embedding) in enumerate(zip(boxes, embeddings)):
                continuity = _normalized_continuity_cost(box, track["last_box"])
                similarity = None if anchor is None or embedding is None else float(np.dot(anchor, embedding))
                if track["shot"] != shot_index:
                    if similarity is None or similarity < float(identity_threshold):
                        continue
                    score = 20.0 + similarity * 10.0 - gap * 0.001
                elif gap == 1:
                    if continuity > 3.0:
                        continue
                    score = 30.0 - continuity * 4.0 + (0.0 if similarity is None else similarity * 3.0)
                elif similarity is not None:
                    if similarity < float(identity_threshold):
                        continue
                    score = 15.0 + similarity * 10.0 - continuity * 0.5 - gap * 0.05
                else:
                    if gap > 12 or continuity > 4.0:
                        continue
                    score = 10.0 - continuity - gap * 0.05
                pairs.append((score, track_index, box_index))
        used_tracks, used_boxes = set(), set()
        for _score, track_index, box_index in sorted(pairs, reverse=True):
            if track_index in used_tracks or box_index in used_boxes:
                continue
            track, box, embedding = tracks[track_index], boxes[box_index], embeddings[box_index]
            track["boxes"][frame_index] = box
            track["last_box"], track["last_frame"], track["shot"] = box, frame_index, shot_index
            if embedding is not None:
                track["embeddings"].append(embedding)
            used_tracks.add(track_index)
            used_boxes.add(box_index)
        for box_index, (box, embedding) in enumerate(zip(boxes, embeddings)):
            if box_index in used_boxes:
                continue
            tracks.append({"boxes": [None] * frame_count, "embeddings": [] if embedding is None else [embedding],
                           "last_box": box, "last_frame": frame_index, "shot": shot_index})
            tracks[-1]["boxes"][frame_index] = box

    def merge_score(first, second):
        first_track, second_track = tracks[first], tracks[second]
        if any(a is not None and b is not None for a, b in zip(first_track["boxes"], second_track["boxes"])):
            return None
        similarity = _track_similarity(first_track, second_track)
        if similarity is not None and similarity >= float(identity_threshold):
            return 10.0 + similarity
        first_frames = [index for index, box in enumerate(first_track["boxes"]) if box is not None]
        second_frames = [index for index, box in enumerate(second_track["boxes"]) if box is not None]
        earlier, later = (first, second) if first_frames[-1] < second_frames[0] else (second, first)
        earlier_frames = first_frames if earlier == first else second_frames
        later_frames = second_frames if later == second else first_frames
        gap = later_frames[0] - earlier_frames[-1]
        continuity = _normalized_continuity_cost(tracks[later]["boxes"][later_frames[0]], tracks[earlier]["boxes"][earlier_frames[-1]])
        empty_gap = not any(detections[earlier_frames[-1] + 1:later_frames[0]])
        if gap <= 60 and continuity <= 1.5 and empty_gap and (similarity is None or similarity >= float(identity_threshold) * 0.5):
            return 5.0 - continuity - gap * 0.001
        return None

    track_ids = list(range(len(tracks)))
    track_positions = {track_id: track_id for track_id in track_ids}
    versions = [0] * len(tracks)
    alive = [True] * len(tracks)
    candidates = []
    comparison_count = 0
    comparison_total = max((len(tracks) - 1) ** 2, 1)
    last_progress = -1

    def add_candidate(first_id, second_id):
        nonlocal comparison_count, last_progress
        first, second = track_positions[first_id], track_positions[second_id]
        score = merge_score(first, second)
        if score is not None:
            heapq.heappush(candidates, (-score, -first_id, -second_id, versions[first_id], versions[second_id]))
        comparison_count += 1
        progress = min(comparison_count * 100 // comparison_total, 99)
        if callable(progress_callback) and progress != last_progress:
            last_progress = progress
            progress_callback("Associating face tracks", comparison_count, comparison_total)

    for first in range(len(tracks)):
        for second in range(first + 1, len(tracks)):
            add_candidate(first, second)

    while candidates:
        _negative_score, negative_first_id, negative_second_id, first_version, second_version = heapq.heappop(candidates)
        first_id, second_id = -negative_first_id, -negative_second_id
        if not alive[first_id] or not alive[second_id] or versions[first_id] != first_version or versions[second_id] != second_version:
            continue
        first, second = track_positions[first_id], track_positions[second_id]
        for frame_index, box in enumerate(tracks[second]["boxes"]):
            if box is not None:
                tracks[first]["boxes"][frame_index] = box
        tracks[first]["embeddings"] += tracks[second]["embeddings"]
        tracks[first]["last_frame"] = max(tracks[first]["last_frame"], tracks[second]["last_frame"])
        tracks[first]["last_box"] = tracks[first]["boxes"][tracks[first]["last_frame"]]
        del tracks[second]
        del track_ids[second]
        track_positions.update((track_id, index) for index, track_id in enumerate(track_ids[second:], start=second))
        versions[first_id] += 1
        alive[second_id] = False
        for other_id in track_ids:
            if other_id != first_id:
                add_candidate(min(first_id, other_id), max(first_id, other_id))
    if callable(progress_callback):
        progress_callback("Associating face tracks", comparison_total, comparison_total)
    for track in tracks:
        heights = [box[3] - box[1] for box in track["boxes"] if box is not None]
        track["anchor"] = _track_anchor(track)
        track["rank"] = sum(heights)
        track["max_height"] = max(heights)
        track["presence"] = len(heights) / frame_count
    return tracks


def _select_tracks(frames, detections, face_count=1, identity_threshold=0.28, reference_images=None,
                   auto_min_face_height=32, auto_min_presence=0.2, insightface_model_dir=None,
                   abort_callback=None, progress_callback=None):
    app = None
    shot_boundaries = _shot_boundaries(frames)
    if shot_boundaries:
        print(f"[FaceRefine] detected {len(shot_boundaries)} shot boundary/boundaries; resetting spatial track continuity")
    frame_embeddings = [[None] * len(boxes) for boxes in detections]
    references = list(reference_images or [])
    try:
        app = _insightface_app(model_dir=insightface_model_dir)
        for frame_index, boxes in enumerate(detections):
            if callable(abort_callback) and abort_callback():
                return None
            if boxes:
                frame_embeddings[frame_index] = _aligned_embeddings(app, frames[frame_index], boxes)
            if callable(progress_callback):
                progress_callback("Matching face identities", frame_index + 1, len(detections))
        tracks = _associate_tracks(detections, frame_embeddings, identity_threshold, shot_boundaries, progress_callback)
        reference_embeddings = []
        for reference in references:
            candidates = _embeddings(app, reference)
            reference_embeddings.append(None if not candidates else max(candidates, key=lambda item: item[0][3] - item[0][1])[1])
    except Exception as error:
        print(f"[FaceRefine] multi-face identity matching unavailable ({error}); using geometric tracks")
        tracks = _associate_tracks(detections, frame_embeddings, identity_threshold, shot_boundaries, progress_callback)
        reference_embeddings = [None] * len(references)
    finally:
        released = _release_insightface(app)
        app = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if released:
            print("[FaceRefine] Unloaded InsightFace ONNX sessions before refinement")

    matches = []
    for reference_index, embedding in enumerate(reference_embeddings):
        if embedding is None:
            print(f"[FaceRefine] reference image {reference_index + 1} has no recognizable face and will be ignored")
            continue
        for track_index, track in enumerate(tracks):
            if track["anchor"] is not None:
                similarity = float(np.dot(embedding, track["anchor"]))
                if similarity >= float(identity_threshold):
                    matches.append((similarity, reference_index, track_index))
    assigned_references, assigned_tracks = set(), set()
    for similarity, reference_index, track_index in sorted(matches, reverse=True):
        if reference_index in assigned_references or track_index in assigned_tracks:
            continue
        tracks[track_index]["reference_image"] = references[reference_index]
        tracks[track_index]["reference_similarity"] = similarity
        assigned_references.add(reference_index)
        assigned_tracks.add(track_index)
        print(f"[FaceRefine] matched reference image {reference_index + 1} to face track {track_index + 1} (similarity {similarity:.3f})")
    for reference_index in range(len(references)):
        if reference_index not in assigned_references and reference_embeddings[reference_index] is not None:
            print(f"[FaceRefine] reference image {reference_index + 1} did not match a detected identity and will be ignored")

    rank_key = lambda index: (tracks[index]["anchor"] is not None, tracks[index]["rank"])
    ranked = sorted(range(len(tracks)), key=rank_key, reverse=True)
    prioritized = sorted(assigned_tracks, key=rank_key, reverse=True)
    if int(face_count) == 0:
        relevant = [index for index in ranked if tracks[index]["max_height"] >= float(auto_min_face_height) and tracks[index]["presence"] >= float(auto_min_presence)]
        selected_indices = (prioritized + [index for index in relevant if index not in assigned_tracks])[:5]
        too_small = sum(track["max_height"] < float(auto_min_face_height) for track in tracks)
        too_brief = sum(track["presence"] < float(auto_min_presence) for track in tracks)
        print(f"[FaceRefine] Auto face selection: {len(relevant)}/{len(tracks)} relevant track(s), thresholds max face height >= {float(auto_min_face_height):g}px and presence >= {float(auto_min_presence):.0%}; rejected {too_small} too small and {too_brief} too brief")
    else:
        selected_indices = (prioritized + [index for index in ranked if index not in assigned_tracks])[:int(face_count)]
    print(f"[FaceRefine] detected {len(tracks)} distinct face track(s); refining {len(selected_indices)}")
    return [tracks[index] for index in selected_indices]


def _body_fallback(frames, missing, fallback_path, confidence, abort_callback=None):
    model = _load_yolo(fallback_path)
    found = {}
    try:
        for index in np.nonzero(missing)[0]:
            if callable(abort_callback) and abort_callback():
                return None
            result = model.predict(_bgr(frames[index]), conf=float(confidence), verbose=False)[0]
            if not len(result.boxes):
                continue
            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist() if getattr(result.boxes, "cls", None) is not None else [0] * len(boxes)
            people = [box for box, cls in zip(boxes, classes) if int(cls) == 0] or boxes
            found[index] = max(people, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    finally:
        _release_detector(model)
    return found


def _affine_crop(frame: torch.Tensor, box, width: int, height: int, reduce_large=False) -> torch.Tensor:
    x, y, box_width, box_height = box
    source_h, source_w = frame.shape[:2]
    source = frame[..., :3].movedim(-1, 0).unsqueeze(0).float()
    if frame.dtype == torch.uint8:
        source.div_(255.0)
    theta = source.new_tensor([[[box_width / source_w, 0.0, (2.0 * x + box_width) / source_w - 1.0],
                                [0.0, box_height / source_h, (2.0 * y + box_height) / source_h - 1.0]]])
    oversampling = max(1, math.ceil(box_height / height)) if reduce_large else 1  # a box larger than the canvas: read at its resolution, then averaged
    grid = F.affine_grid(theta, (1, 3, height * oversampling, width * oversampling), align_corners=False)
    crop = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return (F.avg_pool2d(crop, oversampling) if oversampling > 1 else crop)[0]


def crop_face_track(frames, transform, start=0, stop=None, *, uint8_storage=False):
    stop = len(transform["boxes"]) if stop is None else int(stop)
    start = int(start)
    canvas_width, canvas_height = transform["canvas"]
    dtype = torch.uint8 if uint8_storage else torch.float32
    crops = torch.empty((3, stop - start, canvas_height, canvas_width), dtype=dtype, device="cpu")
    for target, index in enumerate(range(start, stop)):
        crop = _affine_crop(frames[index], transform["boxes"][index], canvas_width, canvas_height, reduce_large=transform.get("reduce_large", False))
        if uint8_storage:
            crop = crop.mul_(255.0).round_().byte()
        else:
            crop = crop.mul_(2.0).sub_(1.0)
        crops[:, target].copy_(crop)
    return crops


def best_reference_crop(frames, transform, insightface_model_dir=None):
    """The track's best face as its stabilized crop: (frame index, uint8 (3, height, width)), chosen by select_reference_frame (large, frontal, mouth closed, sharp)."""
    frame = select_reference_frame(frames, transform["raw_face_boxes"], insightface_model_dir=insightface_model_dir)
    return frame, crop_face_track(frames, transform, frame, frame + 1, uint8_storage=True)[:, 0]


def reference_image_crop(image, detector_path, crop_factor, size, confidence=0.25):
    """A supplied picture framed like the crops of a track: (uint8 (3, size, size), the crop's side in the picture), or None without a detected face."""
    frame = torch.from_numpy(np.array(image.convert("RGB"), copy=True))
    boxes = _detect(frame.unsqueeze(0), detector_path, confidence)[0]
    if not boxes:
        return None
    left, top, right, bottom = max(boxes, key=lambda box: box[3] - box[1])
    side = (bottom - top) * float(crop_factor)
    crop = _affine_crop(frame, ((left + right - side) / 2.0, (top + bottom - side) / 2.0, side, side), size, size, reduce_large=True)
    return crop.mul_(255.0).round_().byte(), side


def _crop_track(frames, selected, *, confidence=0.35, crop_factor=2.5, canvas_width=512, canvas_height=512,
                canvas_mode="manual", canvas_tiers=None, smooth_window=21, size_smooth_window=51, smooth_method="gaussian",
                size_mode="per_frame", fallback_detector_path=None, fallback_head_frac=0.5, strength_small_face=0.85,
                strength_large_face=0.85, scale_mode="absolute_px", face_px_small=30.0, face_px_large=120.0,
                strength_gamma=1.0, strength_smooth_frames=9, abort_callback=None, progress_callback=None, track_label="face",
                include_crops=True):
    detected_boxes = list(selected)
    valid = np.asarray([box is not None for box in detected_boxes], dtype=bool)
    if not valid.any():
        raise ValueError("No face detected in any frame. Lower detector confidence, or skip this clip")
    selected, via_flow = _flow_missing_boxes(frames, detected_boxes)

    frame_count, source_h, source_w, _ = frames.shape
    cx = np.asarray([0.0 if box is None else (box[0] + box[2]) / 2.0 for box in selected])
    cy = np.asarray([0.0 if box is None else (box[1] + box[3]) / 2.0 for box in selected])
    face_h = np.asarray([0.0 if box is None else box[3] - box[1] for box in selected])
    face_w = np.asarray([0.0 if box is None else box[2] - box[0] for box in selected])
    detected_indices = np.flatnonzero(valid)
    largest_face_frame = int(detected_indices[np.argmax(face_h[detected_indices])])
    face_h_seed = _interpolate(face_h, valid)
    via_body = np.zeros(frame_count, dtype=bool)
    if fallback_detector_path is not None and (~(valid | via_flow)).any():
        bodies = _body_fallback(frames, ~(valid | via_flow), fallback_detector_path, confidence, abort_callback)
        if bodies is None:
            return None
        for index, body in bodies.items():
            cx[index] = (body[0] + body[2]) / 2.0
            cy[index] = body[1] + float(fallback_head_frac) * max(face_h_seed[index], 8.0)
            face_h[index] = face_h_seed[index]
            via_body[index] = True

    known = valid | via_flow | via_body
    active, segments, discontinuities = _track_segments(_track_presence(known), selected)
    raw_cx, raw_cy = _interpolate_track(cx, known, active), _interpolate_track(cy, known, active)
    raw_face_h, raw_face_w = _interpolate_track(face_h, known, active), _interpolate_track(face_w, known, active)
    cx = _smooth_track(raw_cx, segments, smooth_window, smooth_method)
    cy = _smooth_track(raw_cy, segments, smooth_window, smooth_method)
    face_h = _smooth_track(raw_face_h, segments, size_smooth_window, smooth_method)
    face_w = _smooth_track(raw_face_w, segments, size_smooth_window, smooth_method)
    if size_mode == "max_of_clip":
        face_h[:] = face_h.max()

    if canvas_mode != "manual":
        needed = min(float(face_h.max()) * float(crop_factor), source_h)
        auto_size = max(512, math.ceil(needed / 32) * 32)
        canvas_width = canvas_height = max(128, min(1344, auto_size))
        if canvas_mode == "auto_capped_768":
            canvas_width = canvas_height = min(canvas_width, 768)
        if canvas_tiers:  # a model with fixed working sizes: the smallest that keeps the face's resolution, or the largest
            canvas_width = canvas_height = next((tier for tier in canvas_tiers if tier >= needed), canvas_tiers[-1])
    canvas_width = max(128, min(1344, round(int(canvas_width) / 32) * 32))
    canvas_height = max(128, min(1344, round(int(canvas_height) / 32) * 32))
    aspect = canvas_width / float(canvas_height)
    boxes, face_rects = [], []
    for index in range(frame_count):
        box_h, box_w = float(face_h[index]) * crop_factor, float(face_h[index]) * crop_factor * aspect
        if box_w > source_w:
            box_w, box_h = float(source_w), float(source_w) / aspect
        if box_h > source_h:
            box_h, box_w = float(source_h), float(source_h) * aspect
        x = float(cx[index]) - box_w / 2.0
        y = float(cy[index]) - box_h / 2.0
        box = (x, y, box_w, box_h)
        boxes.append(box)
        face_rects.append(((float(cx[index]) - 0.5 * float(face_w[index]) - x) / max(box_w, 1e-6) * canvas_width,
                           (float(cy[index]) - 0.5 * float(face_h[index]) - y) / max(box_h, 1e-6) * canvas_height,
                           float(face_w[index]) / max(box_w, 1e-6) * canvas_width,
                           float(face_h[index]) / max(box_h, 1e-6) * canvas_height))

    weights = np.clip(_smooth(active.astype(np.float64), max(9, smooth_window // 2), "gaussian"), 0.0, 1.0)
    weights[~active] = 0.0
    if scale_mode == "relative_to_clip":
        low, high = float(face_h.min()), float(face_h.max())
    else:
        low, high = float(face_px_small), float(face_px_large)
    scale = np.zeros_like(face_h) if high - low < 1e-6 else np.clip((face_h - low) / (high - low), 0.0, 1.0)
    scale = scale ** float(strength_gamma)
    strengths = float(strength_small_face) + (float(strength_large_face) - float(strength_small_face)) * scale
    strengths = np.clip(_smooth_track(strengths, segments, strength_smooth_frames, "gaussian"), 0.0, 1.0)
    strengths[via_flow] = np.maximum(strengths[via_flow], float(strength_small_face))
    strengths *= weights
    transform = {"boxes": boxes, "face_rect": face_rects, "raw_face_boxes": detected_boxes, "canvas": (canvas_width, canvas_height),
                 "src_size": (source_w, source_h), "frames": frame_count, "weights": weights.tolist(),
                 "detected": valid.tolist(), "active": active.tolist(), "segments": segments,
                 "crop_factor": float(crop_factor), "largest_face_frame": largest_face_frame, "reduce_large": bool(canvas_tiers)}
    magnifications = [canvas_height / max(box[3], 1e-6) for box in boxes]
    padded = sum(x < 0.0 or y < 0.0 or x + width > source_w or y + height > source_h for x, y, width, height in boxes)
    print(f"[FaceRefine] {track_label}: frames={frame_count} detected={int(valid.sum())} ({valid.mean() * 100:.0f}%) optical-flow={int(via_flow.sum())} body-fallback={int(via_body.sum())} short-gap={int(active.sum() - known.sum())} inactive={int((~active).sum())} discontinuities={discontinuities}")
    print(f"[FaceRefine] kept face centered with border padding in {padded}/{frame_count} edge frames")
    print(f"[FaceRefine] face height min={face_h.min():.0f}px mean={face_h.mean():.0f}px max={face_h.max():.0f}px; magnification into {canvas_width}x{canvas_height}: min={min(magnifications):.2f}x mean={np.mean(magnifications):.2f}x max={max(magnifications):.2f}x")
    print(f"[FaceRefine] per-frame denoise strength {strengths.max():.2f} (smallest) .. {strengths.min():.2f} (largest), mean={strengths.mean():.2f}")
    strengths = torch.from_numpy(strengths).float()
    return (crop_face_track(frames, transform), transform, strengths) if include_crops else (transform, strengths)


def track_and_crop(frames, detector_path, *, confidence=0.35, crop_factor=2.5, canvas_width=512, canvas_height=512,
                   canvas_mode="manual", smooth_window=21, size_smooth_window=51, smooth_method="gaussian",
                   size_mode="per_frame", identity_track=True, identity_threshold=0.28, identity_reference=None, select="largest",
                   fallback_detector_path=None, fallback_head_frac=0.5, strength_small_face=0.85,
                   strength_large_face=0.85, scale_mode="absolute_px", face_px_small=30.0, face_px_large=120.0,
                   strength_gamma=1.0, strength_smooth_frames=9, insightface_model_dir=None, abort_callback=None, progress_callback=None):
    detections = _detect(frames, detector_path, confidence, abort_callback, progress_callback)
    if detections is None:
        return None
    selected, identity_matches, continuity_matches, conflicts = _select_track(frames, detections, identity_track, identity_threshold, identity_reference, select, insightface_model_dir)
    print(f"[FaceRefine] tracking: {continuity_matches} by continuity, {conflicts} ambiguous ({identity_matches} resolved by face identity)")
    return _crop_track(frames, selected, confidence=confidence, crop_factor=crop_factor, canvas_width=canvas_width, canvas_height=canvas_height,
                       canvas_mode=canvas_mode, smooth_window=smooth_window, size_smooth_window=size_smooth_window, smooth_method=smooth_method,
                       size_mode=size_mode, fallback_detector_path=fallback_detector_path, fallback_head_frac=fallback_head_frac,
                       strength_small_face=strength_small_face, strength_large_face=strength_large_face, scale_mode=scale_mode,
                       face_px_small=face_px_small, face_px_large=face_px_large, strength_gamma=strength_gamma,
                       strength_smooth_frames=strength_smooth_frames, abort_callback=abort_callback, progress_callback=progress_callback)


def track_faces(frames, detector_path, *, face_count=1, reference_images=None, confidence=0.35, crop_factor=2.5,
                canvas_width=512, canvas_height=512, canvas_mode="manual", canvas_tiers=None, smooth_window=21, size_smooth_window=51,
                smooth_method="gaussian", size_mode="per_frame", identity_threshold=0.28, fallback_detector_path=None,
                auto_min_face_height=32, auto_min_presence=0.2, fallback_head_frac=0.5, strength_small_face=0.85, strength_large_face=0.85, scale_mode="absolute_px",
                face_px_small=30.0, face_px_large=120.0, strength_gamma=1.0, strength_smooth_frames=9,
                insightface_model_dir=None, abort_callback=None, progress_callback=None):
    detections = _detect(frames, detector_path, confidence, abort_callback, progress_callback)
    if detections is None:
        return None
    tracks = _select_tracks(frames, detections, face_count, identity_threshold, reference_images, auto_min_face_height, auto_min_presence, insightface_model_dir,
                            abort_callback, progress_callback)
    if tracks is None:
        return None
    if not tracks:
        if int(face_count) == 0:
            return []
        raise ValueError("No face detected in any frame. Lower detector confidence, or skip this clip")
    output = []
    for track_index, track in enumerate(tracks):
        prepared = _crop_track(frames, track["boxes"], confidence=confidence, crop_factor=crop_factor, canvas_width=canvas_width,
                               canvas_height=canvas_height, canvas_mode=canvas_mode, canvas_tiers=canvas_tiers, smooth_window=smooth_window,
                               size_smooth_window=size_smooth_window, smooth_method=smooth_method, size_mode=size_mode,
                               fallback_detector_path=fallback_detector_path if len(tracks) == 1 else None,
                               fallback_head_frac=fallback_head_frac, strength_small_face=strength_small_face,
                               strength_large_face=strength_large_face, scale_mode=scale_mode, face_px_small=face_px_small,
                               face_px_large=face_px_large, strength_gamma=strength_gamma, strength_smooth_frames=strength_smooth_frames,
                               abort_callback=abort_callback, progress_callback=progress_callback,
                               track_label=f"face track {track_index + 1}/{len(tracks)}", include_crops=False)
        if prepared is None:
            return None
        output.append((*prepared, track.get("reference_image")))
    return output


def track_and_crop_faces(frames, detector_path, **kwargs):
    tracks = track_faces(frames, detector_path, **kwargs)
    if tracks is None:
        return None
    return [(crop_face_track(frames, transform), transform, strengths, reference_image)
            for transform, strengths, reference_image in tracks]


def _blur(mask, radius: int):
    if radius <= 0:
        return mask
    kernel_size = min(2 * int(radius) + 1, max(3, (min(mask.shape[-2:]) // 2) | 1))
    sigma = max(kernel_size / 6.0, 0.5)
    x = torch.arange(kernel_size, device=mask.device, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    pad = kernel_size // 2
    mask = F.conv2d(F.pad(mask, (pad, pad, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(mask, (0, 0, pad, pad), mode="replicate"), kernel.view(1, 1, -1, 1))


def _face_mask(height, width, rect, dilation, feather, ellipse, device):
    mask = torch.zeros((1, 1, height, width), device=device)
    x, y, rect_w, rect_h = rect
    x, y, rect_w, rect_h = x - dilation, y - dilation, rect_w + 2 * dilation, rect_h + 2 * dilation
    if ellipse:
        yy = torch.arange(height, device=device).view(-1, 1)
        xx = torch.arange(width, device=device).view(1, -1)
        rx, ry = max(rect_w / 2.0, 1.0), max(rect_h / 2.0, 1.0)
        mask[0, 0] = (((xx - x - rx) / rx) ** 2 + ((yy - y - ry) / ry) ** 2 <= 1.0).float()
    else:
        x0, y0 = max(0, round(x)), max(0, round(y))
        x1, y1 = min(width, round(x + rect_w)), min(height, round(y + rect_h))
        mask[0, 0, y0:y1, x0:x1] = 1.0
    return _blur(mask, feather).clamp_(0.0, 1.0)


def _feather_mask(height, width, feather, device):
    mask = torch.ones((height, width), device=device)
    feather = int(max(0, min(feather, min(height, width) // 2 - 1)))
    if feather <= 0:
        return mask
    ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, feather + 2, device=device)[1:-1])
    mask[:feather] *= ramp.view(-1, 1)
    mask[height - feather:] *= ramp.flip(0).view(-1, 1)
    mask[:, :feather] *= ramp.view(1, -1)
    mask[:, width - feather:] *= ramp.flip(0).view(1, -1)
    return mask


def stitch(frames, refined, transform, *, paste_region="face_only", mask_dilation=16, feather=6, feather_fraction=None,
           colour_match=1.0, blend=1.0, undetected_frames="fade_out", feather_scales_with_crop=False,
           abort_callback=None, progress_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else frames.device)
    output = frames[..., :3]
    boxes, rects = transform["boxes"], transform["face_rect"]
    canvas_w, canvas_h = transform["canvas"]
    source_w, source_h = transform["src_size"]
    if undetected_frames == "composite_anyway":
        weights = None
    elif undetected_frames == "skip":
        weights = [1.0 if detected else 0.0 for detected in transform["detected"]]
    else:
        weights = transform["weights"]
    per_frame_mb = source_h * source_w * 3 * 4 / 2**20
    chunk_size = max(1, min(32, int(1024 / max(per_frame_mb, 1e-6))))
    for start in range(0, len(boxes), chunk_size):
        if callable(abort_callback) and abort_callback():
            return None
        stop = min(start + chunk_size, len(boxes))
        count = stop - start
        middle_box_h = boxes[(start + stop - 1) // 2][3]
        if feather_fraction is not None:
            canvas_feather = round(feather_fraction * canvas_h)
        else:
            canvas_feather = int(feather) if feather_scales_with_crop else max(1, min(round(feather * canvas_h / max(middle_box_h, 1.0)), canvas_h // 3))
        if paste_region == "full_crop":
            mask = _feather_mask(canvas_h, canvas_w, canvas_feather, device)
            masks = mask.view(1, 1, canvas_h, canvas_w).expand(count, 1, canvas_h, canvas_w)
        else:
            masks = torch.cat([_face_mask(canvas_h, canvas_w, rects[index], int(mask_dilation), canvas_feather,
                                          paste_region == "face_ellipse", device) for index in range(start, stop)])
        theta = torch.empty((count, 2, 3), device=device)
        for local, index in enumerate(range(start, stop)):
            x, y, box_w, box_h = boxes[index]
            theta[local] = theta.new_tensor(((source_w / box_w, 0.0, (source_w - 2.0 * x) / box_w - 1.0),
                                             (0.0, source_h / box_h, (source_h - 2.0 * y) / box_h - 1.0)))
        grid = F.affine_grid(theta, (count, 3, source_h, source_w), align_corners=False)
        refined_batch = refined[:, start:stop].permute(1, 2, 3, 0).detach().cpu().contiguous()
        resize_items = [(refined_batch[local], (min(canvas_h, max(1, round(boxes[index][3]))), min(canvas_w, max(1, round(boxes[index][2]))))) for local, index in enumerate(range(start, stop))]

        def downsample_lanczos(item):
            patch, target = item
            if target == (canvas_h, canvas_w):
                result = patch.movedim(-1, 0).float()
                return result.div_(255.0) if patch.dtype == torch.uint8 else result
            array = patch.numpy() if patch.dtype == torch.uint8 else patch.clamp(0.0, 1.0).mul(255).round().byte().numpy()
            image = Image.fromarray(array)
            image = image.resize((target[1], target[0]), resample=Image.Resampling.LANCZOS)
            return torch.from_numpy(np.array(image, copy=True)).movedim(-1, 0).float().div_(255.0)

        filtered = process_images_multithread(downsample_lanczos, resize_items, "upsample", wrap_in_list=False, max_workers=max(1, int(get_default_workers())))
        filtered = [F.interpolate(patch.unsqueeze(0).to(device), (canvas_h, canvas_w), mode="bilinear", align_corners=False) if patch.shape[-2:] != (canvas_h, canvas_w) else patch.unsqueeze(0).to(device) for patch in filtered]
        patches = F.grid_sample(torch.cat(filtered), grid, mode="bilinear", padding_mode="zeros", align_corners=False).movedim(1, -1)
        masks = F.grid_sample(masks, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp_(0.0, 1.0).movedim(1, -1)
        base = output[start:stop].to(device).float()
        if output.dtype == torch.uint8:
            base.div_(255.0)
        if colour_match > 0.0:
            weight_sum = masks.sum(dim=(1, 2), keepdim=True).clamp_min_(1e-6)
            base_mean = (base * masks).sum(dim=(1, 2), keepdim=True) / weight_sum
            patch_mean = (patches * masks).sum(dim=(1, 2), keepdim=True) / weight_sum
            base_std = (((base - base_mean) ** 2 * masks).sum(dim=(1, 2), keepdim=True) / weight_sum).sqrt().clamp_min_(1e-6)
            patch_std = (((patches - patch_mean) ** 2 * masks).sum(dim=(1, 2), keepdim=True) / weight_sum).sqrt().clamp_min_(1e-6)
            patches = (patches + ((patches - patch_mean) * (base_std / patch_std) + base_mean - patches) * float(colour_match)).clamp_(0.0, 1.0)
        opacity = masks.mul(float(blend))
        if weights is not None:
            opacity.mul_(torch.as_tensor(weights[start:stop], device=device).view(-1, 1, 1, 1))
        composite = (1.0 - opacity) * base + opacity * patches
        if output.dtype == torch.uint8:
            composite = composite.mul_(255.0).round_().byte()
        output[start:stop].copy_(composite.to(device=output.device, dtype=output.dtype))
        if callable(progress_callback):
            progress_callback("Stitching refined faces", stop, len(boxes))
    return output


__all__ = ["best_reference_crop", "crop_face_track", "frames_to_sample", "reference_image_crop", "sample_to_frames", "select_reference_frame", "stitch", "track_and_crop", "track_and_crop_faces", "track_faces"]
