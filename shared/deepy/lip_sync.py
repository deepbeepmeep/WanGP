# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Do the lips of a shot follow its voice? Mouth opening per frame (68-point face from the local pose model, biggest face) against the voice's loudness at the same moment.
It compares timing, not mouth shapes, and is noisy on small faces: a result to look at, not a proof."""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import numpy as np

RATE = 16000
SHORTEST_FRAMES = 36  # the lag is searched up to 12 frames either way
LONGEST_S = 60.0  # a shot: correlating across cuts says nothing


def decode_voice(path) -> np.ndarray:
    from shared.deepy.video_tools import _get_ffmpeg_path

    raw = subprocess.run([_get_ffmpeg_path(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def load_models() -> tuple:
    """The pose model's detector and landmarks, downloaded the first time, on the GPU where there is one (about 1.2 GB, released with the tuple)."""
    import onnxruntime as ort
    from preprocessing.dwpose.assets import query_download_def
    from shared.utils import files_locator as fl
    from shared.utils.download import process_files_def_if_needed

    process_files_def_if_needed(query_download_def())
    providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in ort.get_available_providers()]
    return tuple(ort.InferenceSession(fl.locate_file(f"pose/{name}"), providers=providers) for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"))


def measure(video, start_s: float, frames: int, fps: float, voice: np.ndarray, voice_at: float, models) -> dict[str, Any] | None:
    """r_at_0: the correlation of what the mouth and the voice do faster than a second; best_r, lag_s: where it is best, positive when the mouth comes after the voice. None without a face or below SHORTEST_FRAMES."""
    import cv2
    from preprocessing.dwpose.onnxdet import inference_detector
    from preprocessing.dwpose.onnxpose import inference_pose

    detector, pose = models
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    opening, heights = [], []
    for _ in range(frames):
        success, frame = capture.read()
        if not success:
            break
        best = None
        boxes = inference_detector(detector, frame)
        if len(boxes):
            boxes = np.array(sorted(boxes, key=lambda box: -(box[2] - box[0]) * (box[3] - box[1]))[:2])
            points, scores = inference_pose(pose, boxes, frame)
            for person, score in zip(points, scores):
                face, sure = person[23:91], score[23:91]
                height = float(np.linalg.norm(face[27] - face[8]))
                if sure[48:68].mean() > 0.3 and (best is None or height > best[1]):
                    best = (float((np.linalg.norm(face[66] - face[62]) + np.linalg.norm(face[57] - face[51])) / 2 / max(height, 1.0)), height)
        opening.append(best[0] if best else np.nan)
        if best:
            heights.append(best[1])
    capture.release()
    count = len(opening)
    if count < SHORTEST_FRAMES or not heights:
        return None
    step = RATE / fps
    loud = np.array([np.sqrt(np.mean(voice[int(voice_at * RATE + index * step):int(voice_at * RATE + (index + 1) * step)] ** 2) + 1e-12) for index in range(count)])
    voiced = float((loud > 0.25 * (np.percentile(voice[::40] ** 2, 99) ** 0.5 + 1e-9)).mean())
    opening = np.array(opening)

    def wiggle(values: np.ndarray) -> np.ndarray:  # what moves faster than about a second
        filled = np.nan_to_num(values, nan=float(np.nanmean(values)))
        return filled - np.convolve(np.pad(filled, 12, mode="edge"), np.ones(25) / 25, "valid")

    moved, heard = wiggle(opening), wiggle(loud)
    scores = {lag: float(np.nan_to_num(np.corrcoef(moved[lag:] if lag >= 0 else moved[:lag], heard[:count - lag] if lag >= 0 else heard[-lag:])[0, 1])) for lag in range(-12, 13)}
    lag = max(scores, key=scores.get)
    return {"face_px": float(np.median(heights)), "seen": len(heights) / count, "voiced": voiced, "r_at_0": scores[0], "best_r": scores[lag], "lag_s": lag / fps}


def verdict(measured: dict[str, Any] | None) -> str:
    """follows, may_not_follow, or not_measurable: no clear face, or too little voice."""
    if not measured or measured["voiced"] < 0.15 or measured["seen"] < 0.5:
        return "not_measurable"
    return "follows" if measured["r_at_0"] >= 0.15 or (measured["best_r"] >= 0.25 and abs(measured["lag_s"]) <= 0.3) else "may_not_follow"  # up to 0.3 s away still reads as in sync, by eye on real shots


def check(video, start_s: float = 0.0, duration_s: float | None = None, voice=None, voice_start_s: float | None = None) -> dict[str, Any]:
    """One shot of a video against its own sound or against a separate recording of the voice."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps, total = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    if not fps or not total:
        raise ValueError(f"{Path(video).name} has no readable video stream.")
    length = max(0.0, total / fps - start_s) if duration_s is None else float(duration_s)
    if length > LONGEST_S:
        raise ValueError(f"This covers {length:.0f} s: check one shot at a time, with start_s and duration_s (at most {LONGEST_S:.0f} s).")
    frames = round(length * fps)
    measured = measure(video, start_s, frames, fps, decode_voice(voice or video), start_s if voice_start_s is None else voice_start_s, load_models())
    result = {"verdict": verdict(measured), "frames": frames}
    if measured:
        result.update(face_height_px=round(measured["face_px"]), face_seen=round(measured["seen"], 2), voice_present=round(measured["voiced"], 2), correlation_at_0=round(measured["r_at_0"], 3),
                      best_correlation=round(measured["best_r"], 3), best_lag_s=round(measured["lag_s"], 3))
    return result
