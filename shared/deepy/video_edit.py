# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# The cut detector, the window plan and the take checks follow the Viggle Remake MCP server (Copyright Viggle AI, Apache License 2.0),
# with the thresholds measured there.
"""Measured help for editing a video shot by shot: where its cuts are, how to split it into render windows, and whether a take kept its cuts and picture."""
from __future__ import annotations

import subprocess

import numpy as np

from shared.deepy.video_tools import _get_ffmpeg_path
from shared.utils.utils import get_video_info

EDIT_FPS = 24
WIDTH = 832  # the detector's thresholds were measured on frames of this size, read one pixel in four
MIN_FRAMES = 90
MAX_FRAMES = 345
DEFAULT_MAX_FRAMES = 294
MAX_CUTS = 3  # a window with more cuts drops and reorders its short shots
HOP_FRAMES = 17
SHORTEST_SHOT = 12  # frames: a flash this short stays with the camera shot before it


def read_thumbnails(path: str, *, start: int = 0, count: int | None = None, fps: float | None = EDIT_FPS) -> np.ndarray:
    """The frames of a video at `fps`, or its own frames with fps None, as small thumbnails (frames, height, width, 3), read as a stream: two minutes hold about 250 MB."""
    _, width, height, _ = get_video_info(path)
    long_side = max(width, height)
    frame_width = round(width * WIDTH / long_side / 2) * 2
    frame_height = round(height * WIDTH / long_side / 2) * 2
    filters = [*([f"fps={fps:g}"] if fps else []), f"scale={frame_width}:{frame_height}"]
    if start or count is not None:
        filters.insert(len(filters) - 1, f"select=between(n\\,{start}\\,{start + (count or 10**9) - 1})")
    command = [_get_ffmpeg_path(), "-v", "error", "-i", str(path), "-vf", ",".join(filters), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = frame_width * frame_height * 3
    thumbnails = []
    while len(data := process.stdout.read(size)) == size:
        thumbnails.append(np.frombuffer(data, np.uint8).reshape(frame_height, frame_width, 3)[::4, ::4].copy())
    error = process.stderr.read().decode("utf-8", "replace")
    if process.wait() != 0 or not thumbnails:
        raise RuntimeError(f"Could not read the frames of {path}: {error.strip() or 'no frames'}")
    return np.stack(thumbnails)


def find_cuts(thumbnails: np.ndarray, threshold: float = 22, spike: float = 3) -> list[int]:
    """Hard cuts, as the first frame of each new shot: a jump in the picture together with a jump in its colours, or a jump standing out from the frames around it
    (a cut measures 3-45x its neighbours, motion inside a shot at most 1.8x). Dark shots get a lower threshold. Candidates closer than 7 frames merge unless the later one is a hard jump."""
    if len(thumbnails) < 2:
        return []
    small = thumbnails.astype(np.int16)
    jump = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
    histogram = np.stack([np.concatenate([np.histogram(frame[..., channel], 16, (0, 256))[0] for channel in range(3)]) for frame in small])
    histogram = histogram / histogram.sum(1, keepdims=True)
    colours = np.abs(histogram[1:] - histogram[:-1]).sum(1)
    around = np.array([np.median(np.r_[jump[max(0, index - 4):index], jump[index + 1:index + 5]]) if len(jump) > 1 else 0.0 for index in range(len(jump))])
    ratio = jump / np.maximum(around, 1)
    luminance = small.mean(axis=(1, 2, 3))
    cut = (jump > threshold) & (colours > 0.25)
    cut |= (jump > threshold) & (ratio > spike)
    cut |= (jump > 8) & (ratio > 8)  # between two dark shots a hard cut stays under the threshold
    cut |= (jump > 12) & (ratio > 4) & (np.maximum(luminance[1:], luminance[:-1]) < 30)  # low-key shots, too many false cuts on bright clips without the luminance gate
    cuts, last = [], -10
    for index in np.where(cut)[0]:
        if index + 1 - last > 6 or (index + 1 - last > 3 and jump[index] > 2 * threshold):  # a run of candidates (a whip, a strobe) stays one cut
            cuts.append(int(index) + 1)
        last = int(index) + 1
    return cuts


def detect_cuts(path: str, fps: float = EDIT_FPS, *, own_frames: bool = False) -> dict:
    """The cuts and shots of a video, in frames at `fps`; own_frames: in the video's own frames, whose rate `fps` is."""
    thumbnails = read_thumbnails(path, fps=None if own_frames else fps)
    cuts = find_cuts(thumbnails)
    frames = len(thumbnails)
    edges = [0, *cuts, frames]
    return {"fps": fps, "frames": frames, "duration_s": round(frames / fps, 3), "cuts": cuts, "cuts_s": [round(cut / fps, 3) for cut in cuts],
            "shots": [{"start": start, "end": end, "seconds": round((end - start) / fps, 3)} for start, end in zip(edges, edges[1:])]}


def window_lengths(minimum: int = MIN_FRAMES, maximum: int = MAX_FRAMES, step: int = 17, offset: int = 5) -> list[int]:
    return [length for length in range(offset, maximum + 1, step) if length >= minimum]


def plan_windows(cuts: list[int], frames: int, *, max_frames: int = DEFAULT_MAX_FRAMES, max_cuts: int = MAX_CUTS, min_frames: int = MIN_FRAMES, hop: int = HOP_FRAMES, step: int = 17, offset: int = 5) -> list[dict]:
    """Split a video into render windows of valid lengths. Windows end on cuts, which hide the seam, with at most `max_cuts` cuts inside where the cuts allow. `render` is the range of source
    frames the window renders and may run past `own`, the frames it delivers. A shot longer than a window is not split into independent windows, where the seam jumps a face, but continued:
    the next window is a hop (`continues` = the window before it) whose first `hop` frames are the last ones of that window's take. The last window is aligned to the end of the video."""
    lengths = window_lengths(min_frames, max_frames, step, offset)
    fit = lambda needed: min(length for length in lengths if length >= needed)  # noqa: E731
    windows, start, carried = [], 0, 0
    while True:
        render_start = start - carried
        candidates = [end for end in cuts if render_start + lengths[0] <= end <= render_start + lengths[-1] and start < end < frames and render_start + fit(end - render_start) <= frames]
        if frames - render_start <= lengths[-1] and (not candidates or sum(start < cut < frames for cut in cuts) <= max_cuts):
            break
        allowed = [end for end in candidates if sum(render_start < cut < render_start + fit(end - render_start) for cut in cuts) <= max_cuts]
        end = max(allowed) if allowed else min(candidates) if candidates else render_start + lengths[-1]
        windows.append({"render": [render_start, render_start + fit(end - render_start)], "own": [start, end], **({"continues": len(windows) - 1} if carried else {})})
        start, carried = end, 0 if end in cuts else hop
    length = fit(max(frames - start + carried, lengths[0]))
    first = max(0, frames - length)  # a video shorter than the next valid length renders past its end, from a clip padded by its last frames: the trim drops them
    if carried:  # the hop starts `hop` frames into its render: its own start moves back, out of the window before it
        start = windows[-1]["own"][1] = first + hop
        windows.append({"render": [first, first + length], "own": [start, frames], "continues": len(windows) - 1})
    else:
        windows.append({"render": [first, first + length], "own": [start, frames]})
    return windows


def plan_camera_shots(cuts: list[int], frames: int, *, max_frames: int = DEFAULT_MAX_FRAMES, min_frames: int = MIN_FRAMES, hop: int = HOP_FRAMES, step: int = 17, offset: int = 5) -> list[dict]:
    """One render per camera shot, delivering all of it and nothing else: no render holds a cut, so each cut falls exactly where the edit joins two shots. A shot of a length the model cannot render is
    WanGP's to fit: a virtual control video pads itself. One longer than a window continues by hops (`windows`)."""
    lengths, edges = window_lengths(min_frames, max_frames, step, offset), [0]
    for cut in cuts:
        if cut - edges[-1] >= SHORTEST_SHOT:
            edges.append(cut)
        elif len(edges) > 1:
            edges[-1] = cut
    if len(edges) > 1 and frames - edges[-1] < SHORTEST_SHOT:
        edges.pop()
    return [{"own": [start, end], "windows": 1 if end - start <= lengths[-1] else len(plan_windows([], end - start, max_frames=max_frames, min_frames=min_frames, hop=hop, step=step, offset=offset))} for start, end in zip(edges, [*edges[1:], frames])]


def render_groups(windows: list[dict], frames: int, *, step: int = 17, offset: int = 5) -> list[dict]:
    """Windows chained by hops as one render, which WanGP's sliding window continues by itself: its `render` range has a valid length, `windows` says how many windows it spans
    and `pad` how many frames that range runs past the end of the video."""
    groups = []
    for window in windows:
        if "continues" in window:
            groups[-1]["own"][1] = window["own"][1]
            groups[-1]["windows"] += 1
        else:
            groups.append({"render": list(window["render"]), "own": list(window["own"]), "windows": 1})
    for group in groups:
        needed = group["own"][1] - group["render"][0] if group["windows"] > 1 else group["render"][1] - group["render"][0]
        length = next(offset + step * count for count in range(10**6) if offset + step * count >= needed)
        start = group["render"][0] if group["render"][0] + length <= frames else max(0, frames - length)
        group["render"], group["pad"] = [start, start + length], max(0, start + length - frames)
    return groups


def _jump(video: np.ndarray, frame: int) -> float:
    part = video[max(frame - 3, 0):frame + 3].astype(np.int16)
    return float(np.abs(np.diff(part, axis=0)).mean(axis=(1, 2, 3)).max()) if len(part) > 1 else 0.0


def cut_faults(source: np.ndarray, take: np.ndarray, start: int = 0, end: int | None = None, tolerance: int = 8, known: tuple[int, ...] = ()) -> list[dict]:
    """Cuts of the source inside [start, end) that the take puts `tolerance` or more frames off or drops, and cuts it adds there. A cut the take really moved leaves it still where the source
    cuts, under 0.4x the source's jump; an added cut is one the source does not answer, moving under 0.65x the take's jump, 3 or more frames from every source cut."""
    count = min(len(source), len(take))
    end = count if end is None else min(end, count)
    source_cuts = sorted({*find_cuts(source[:count]), *(cut for cut in known if 0 < cut < count)})
    take_cuts = find_cuts(take[:count])
    answered = {}  # each take cut answers one source cut, closest pairs first, so that a rapid pair is not matched crosswise
    for gap, source_cut, take_cut in sorted((abs(take_cut - source_cut), source_cut, take_cut) for source_cut in source_cuts for take_cut in take_cuts):
        if gap <= 24 and source_cut not in answered and take_cut not in answered.values():
            answered[source_cut] = take_cut
    faults = []
    for source_cut, take_cut in sorted(answered.items()):
        if (start <= source_cut < end or source_cut == end > take_cut) and abs(take_cut - source_cut) >= tolerance and _jump(take, source_cut) < 0.4 * _jump(source, source_cut):
            faults.append({"fault": "moved", "source_frame": source_cut, "take_frame": take_cut, "frames_off": take_cut - source_cut})
    faults += [{"fault": "dropped", "source_frame": cut} for cut in source_cuts if start <= cut < end and cut >= 3 and cut not in answered and _jump(take, cut) < 0.4 * _jump(source, cut)]
    for take_cut in take_cuts:
        if start <= take_cut < end and take_cut not in answered.values() and all(abs(take_cut - cut) >= 3 for cut in source_cuts) and _jump(source, take_cut) < 0.65 * _jump(take, take_cut):
            faults.append({"fault": "added", "take_frame": take_cut})
    return faults


def edge_bands(source: np.ndarray, take: np.ndarray, start: int = 0, end: int | None = None, minimum_run: int = 6) -> list[dict]:
    """Runs of frames where an edge of the take grows a sharp-edged black band the source lacks: more than 15 % of the line black from the edge, then under 5 % at once, within 35 % of the frame.
    So far always caused by a painted frame carrying that band, so another seed does not remove it: repaint the frame. Frames within 4 of a source cut are skipped."""
    count = min(len(source), len(take))
    end = count if end is None else min(end, count)
    cuts = find_cuts(source[:count])
    bands = []
    for name, axis, flip in (("top", 1, False), ("bottom", 1, True), ("left", 2, False), ("right", 2, True)):
        dark, dark_source = take[:count].max(axis=3) < 40, source[:count].max(axis=3) < 40
        if axis == 2:
            dark, dark_source = dark.transpose(0, 2, 1), dark_source.transpose(0, 2, 1)
        if flip:
            dark, dark_source = dark[:, ::-1], dark_source[:, ::-1]
        lines, lines_source = dark.mean(2), dark_source.mean(2)
        depth = lines.shape[1]
        hits = []
        for frame in range(count):
            clear = np.nonzero(lines[frame, :int(0.35 * depth) + 1] < 0.05)[0]
            if len(clear) and clear[0] >= 1 and (lines[frame, :clear[0]] > 0.15).all() and lines_source[frame, :clear[0]].mean() < 0.05 and start <= frame < end and all(abs(frame - cut) > 4 for cut in cuts):
                hits.append(frame)
        first = None
        for frame, following in zip(hits, [*hits[1:], None]):
            first = frame if first is None else first
            if following != frame + 1:
                if frame - first + 1 >= minimum_run:
                    bands.append({"edge": name, "start_frame": first, "end_frame": frame})
                first = None
    return bands


def check_take(source_path: str, take_path: str, *, source_start: int = 0, own_start: int = 0, own_end: int | None = None, known_cuts: tuple[int, ...] = (), fps: float | None = EDIT_FPS) -> dict:
    """Measure a take against the source frames it was rendered from: frames count from the take's first frame, the source's from `source_start`. fps None: frame for frame, whatever the rates say."""
    take = read_thumbnails(take_path, fps=fps)
    source = read_thumbnails(source_path, start=source_start, count=len(take), fps=fps)
    if source.shape[1:] != take.shape[1:]:
        import cv2

        source = np.stack([cv2.resize(frame, take.shape[2:0:-1], interpolation=cv2.INTER_AREA) for frame in source])
    cuts, bands = cut_faults(source, take, own_start, own_end, known=known_cuts), edge_bands(source, take, own_start, own_end)
    return {"ok": not cuts and not bands, "frames": len(take), "source_frames": len(source), "cut_faults": cuts, "edge_bands": bands}
