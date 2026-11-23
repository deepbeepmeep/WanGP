# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Gathering the windows of a generation in a file, after a Source Video when one is continued.

The model only reads the last frames of what comes before a window. The frames before them stay in a file: the new frames of each
window are encoded alone and joined to the video made so far by copying its pictures, so nothing is encoded twice and the video is
never held in memory. A Source Video encoded like the new frames opens that file as it is; another one (other size, rate, codec, or shortened) is
encoded once, a few frames at a time, going through the upsamplers and Film Grain of the generation on the way when they are on.

The whole Source Video is still loaded when something has to process its frames with the new ones (loading_reasons)."""
from __future__ import annotations

import os
import uuid

from .audio_video import FrameChunks, join_video_streams, video_stream_facts
from .video_codecs import hardware_video_encoding_enabled
from .video_decode import _read_rate, _resample_frame_indices, probe_video_stream_metadata, video_needs_corrected_decode
from .virtual_media import get_virtual_media_entry, parse_virtual_media_path

CHUNK_BYTES = 512 * 2**20  # decoded frames of a Source Video held at once while it is encoded again


def _listed(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def source_frames_count(video_source, keep_frames, fps, max_frames=None):
    """Frames a Source Video gives when read at fps. keep_frames: empty keeps them all (the first max_frames when given), a positive
    number keeps the first ones, a negative one removes the last ones."""
    metadata = probe_video_stream_metadata(video_source)
    rate = _read_rate(metadata, fps)
    keep = (int(metadata["frame_count"] * float(fps) / rate) + 2 if max_frames is None else max_frames) if keep_frames in (None, "") else int(keep_frames)
    if keep < 0:
        keep = int(max(metadata["frame_count"] / metadata["fps_float"] * fps + keep, 0)) if metadata["fps_float"] > 0 else 0
    return len(_resample_frame_indices(rate, metadata["frame_count"], keep, float(fps), 0)) if keep > 0 else 0


def loading_reasons(video_source, *, processing=(), hdr=False, frames_returned=False):
    """Why every frame of a Source Video has to be loaded in memory: what processes them with the new frames, or reads them back.
    processing: why its upsamplers cannot take a video a chunk at a time (postprocessing.chunking.whole_video_reasons).
    Empty when the frames before the ones the next window reads can stay in their file."""
    reasons = list(processing)
    if hdr:
        reasons.append("the output is HDR")
    if get_virtual_media_entry(video_source) is not None:
        reasons.append("the Source Video is not a file")
    if frames_returned:
        reasons.append("the output frames are returned to the caller")
    return reasons


def loaded_source_plan(video_source, keep_frames, fps, new_frames, max_frames, first_window_frames=0, **criteria):
    """What a generation that loads its whole Source Video keeps in memory: the reasons it is loaded (the criteria of loading_reasons),
    its frames read at fps (0 for a file that cannot be read: its own error follows), the new frames kept with them and the most frames
    that may be kept. None when the Source Video stays in its file.

    The loaded Source Video leaves memory once it is saved with the first window, of first_window_frames at most (0: one window makes
    the whole video), each later window being saved alone: only that window counts. An HDR output and frames returned to the caller
    stay in memory to the end ("kept"): every new frame counts."""
    reasons = loading_reasons(video_source, **criteria)
    if not reasons:
        return None
    source_frames, kept = source_frames_count(video_source, keep_frames, fps) if probe_video_stream_metadata(video_source) is not None else 0, bool(criteria.get("hdr") or criteria.get("frames_returned"))
    held_frames = int(new_frames) if kept or first_window_frames <= 0 else min(int(new_frames), int(first_window_frames))
    return {"reasons": reasons, "source_frames": source_frames, "new_frames": held_frames, "max_frames": int(max_frames), "window_frames": 0, "kept": kept}


def frames_limit_error(plan, extra_windows=0):
    """The message refusing a generation, or extra_windows more windows of window_frames each, when the Source Video loaded in memory
    and the new frames pass the limit together; empty when they fit or when the Source Video stays in its file (no plan)."""
    if plan is None:
        return ""
    new_frames = plan["new_frames"] + extra_windows * plan["window_frames"]
    if plan["source_frames"] + new_frames <= plan["max_frames"]:
        return ""
    return (f"The whole Source Video has to be loaded in memory because {_listed(plan['reasons'])}: its {plan['source_frames']} frames and the {new_frames} "
            f"{'new ones' if plan['kept'] else 'of the first window'} are more than the {plan['max_frames']} frames that can be kept in memory. "
            f"Keep fewer frames of the Source Video, {'generate fewer frames' if plan['kept'] else 'use a shorter window'}, or turn that off.")


def loading_notice(plan, video_source, width, height):
    """The console line telling that a Source Video is loaded whole, why, and the memory it takes: its frames at the output size until
    they are saved with the first window (to the end of a generation whose output stays in memory), and more while they are read (the
    decoded frames, then two copies of the resized ones)."""
    metadata = probe_video_stream_metadata(video_source)
    resized, decoded = plan["source_frames"] * width * height * 3, plan["source_frames"] * metadata["display_width"] * metadata["display_height"] * 3
    peak = max(2 * decoded, decoded + resized, 2 * resized)
    return (f"[Continue Video] The whole Source Video ({plan['source_frames']} frames) is loaded in memory because {_listed(plan['reasons'])}: "
            f"about {resized / 2**30:.1f} GB of RAM held {'during the generation' if plan['kept'] else 'until the first window is saved'}, up to {peak / 2**30:.1f} GB while it is loaded.")


def read_source_tail(read_frames, frames_count, tail_frames):
    """The last tail_frames of a Source Video of frames_count frames, read alone with read_frames(start, count), and the number of frames
    its file really gives: a file can hold fewer frames than it announces, and its last frames then start earlier."""
    tail_frames = min(tail_frames, frames_count)
    frames = read_frames(frames_count - tail_frames, tail_frames)
    read = 0 if frames is None else frames.shape[0]
    if read < tail_frames:
        frames_count -= tail_frames - read
        tail_frames = min(tail_frames, frames_count)
        frames = read_frames(frames_count - tail_frames, tail_frames)
    return frames, frames_count


def copy_obstacles(video_source, source_frames, fps, width, height, segment_facts):
    """Why a Source Video cannot open the output as it is, the new frames (encoded in a file whose video_stream_facts are segment_facts)
    following its pictures by copy; empty when it can. It is encoded again otherwise."""
    if parse_virtual_media_path(video_source) is not None:
        return ["only a part of its file is continued"]
    metadata, obstacles = probe_video_stream_metadata(video_source), []
    if video_needs_corrected_decode(video_source) or metadata.get("rotation"):
        obstacles.append("its picture is corrected when read (aspect ratio, rotation or HDR)")
    if (metadata["display_width"], metadata["display_height"]) != (width, height):
        obstacles.append(f"its size {metadata['display_width']}x{metadata['display_height']} is not the output's {width}x{height}")
    if abs(metadata["fps_float"] - fps) > 0.001:
        obstacles.append(f"its rate {metadata['fps_float']:g} fps is not the output's {fps:g} fps")
    if obstacles:
        return obstacles
    facts = video_stream_facts(video_source)
    if facts["frames"] != source_frames:
        obstacles.append(f"{source_frames} of its {facts['frames']} frames are continued")
    if facts["encoding"] != segment_facts["encoding"]:
        obstacles.append("it was not encoded with the current video output settings")
    return obstacles


class ContinuedVideo:
    """The file the windows of a generation are gathered in, after its Source Video when it continues one kept in its file
    (video_source None otherwise: the first window opens the file).

    read_frames(start, count) gives frames_count frames of the Source Video at the size and rate of the generation, as a (frames, H, W, C)
    uint8 tensor; encode(frames, path, hardware) writes frames (a list of tensors or FrameChunks) to a video file without sound and returns
    its path, on NVENC when hardware is True and it is available; fps is the rate of the output. Working files go to work_dir and are
    listed in temp_files until they are removed.

    The encoder is chosen once (the Hardware Video Encoding config) so that every window joins by copy, and turned to the other one when
    only that keeps the pictures of the Source Video copied (a video encoded on the CPU continued with NVENC on, or the reverse).

    With post-processing on, process(read_frames, frames_count) gives the frames of the Source Video gone through it, as FrameChunks
    whose aborted tells that the user stopped it, and copied_frames is the number of pictures a Source Video that already went
    through it holds: such a file opens the output as it is."""

    def __init__(self, video_source, frames_count, fps, read_frames, encode, work_dir, container, temp_files, notify=print, process=None, copied_frames=None):
        self.video_source, self.frames_count, self.fps, self.process = video_source, frames_count, fps, process
        self.copied_frames = frames_count if copied_frames is None else copied_frames
        self.read_frames, self.encode, self.notify, self.temp_files = read_frames, encode, notify, temp_files
        self.hardware = hardware_video_encoding_enabled()
        self.work_prefix, self.container, self.work_no = os.path.join(work_dir, f"tmp{uuid.uuid4().hex[:8]}_continue"), container, 0
        self.source_file, self.source_frames = None, 0  # the Source Video as the new frames can follow it (itself or its encoded copy) and its pictures, kept for every sample
        self.file, self.file_frames = None, 0  # the video made so far for the current sample

    def _work_path(self):
        self.work_no += 1
        path = f"{self.work_prefix}{self.work_no}.{self.container}"
        self.temp_files.append(path)
        return path

    def _remove(self, path):
        if path in self.temp_files:
            self.temp_files.remove(path)
            os.remove(path)

    def _resolve_source_file(self, segment_facts):
        """The file the new frames follow and the number of its pictures; None when the user stopped its post-processing."""
        width, height = segment_facts["encoding"][3:5]
        obstacles = copy_obstacles(self.video_source, self.copied_frames, self.fps, width, height, segment_facts)
        if not obstacles:
            return self.video_source, self.copied_frames
        self.notify(f"[Continue Video] The Source Video is {'post-processed and ' if self.process is not None else ''}encoded again ({self.frames_count} frames, read a few at a time) because {_listed(obstacles)}.")
        if self.process is None:
            metadata = probe_video_stream_metadata(self.video_source)
            frames = FrameChunks(self.read_frames, self.frames_count, max(8, min(256, CHUNK_BYTES // (metadata["display_width"] * metadata["display_height"] * 3))))
        else:
            frames = self.process(self.read_frames, self.frames_count)
        path = self.encode(frames, self._work_path(), self.hardware)
        if getattr(frames, "aborted", False):
            self._remove(path)
            return None
        return path, video_stream_facts(path)["frames"]

    def _encoder_of_source(self, frames, segment, segment_facts):
        """The first window encoded by the other encoder when only that lets the Source Video open the output by copy; the encoder of the
        generation then becomes that one."""
        width, height = segment_facts["encoding"][3:5]
        if not copy_obstacles(self.video_source, self.copied_frames, self.fps, width, height, segment_facts):
            return segment, segment_facts
        other = self.encode(frames, self._work_path(), not self.hardware)
        other_facts = video_stream_facts(other)
        if other_facts["encoding"] == segment_facts["encoding"] or copy_obstacles(self.video_source, self.copied_frames, self.fps, width, height, other_facts):
            self._remove(other)
            return segment, segment_facts
        self._remove(segment)
        self.hardware = not self.hardware
        self.notify(f"[Continue Video] The Source Video was encoded {'on NVENC' if self.hardware else 'on the CPU'}: the new frames are encoded the same way, so that its pictures are copied.")
        return other, other_facts

    def start_sample(self):
        """A new sample continues the Source Video again: the video of the previous one is dropped."""
        if self.file is not None:
            self._remove(self.file)
        self.file, self.file_frames = None, 0

    def append(self, frames):
        """Add the new frames of a window and return the file holding the whole video so far (pictures only), valid until the next call.
        None when the user stopped the post-processing of the Source Video."""
        segment = self.encode(frames, self._work_path(), self.hardware)
        if self.file is None and self.video_source is None:  # nothing before the first window: its own file opens the video
            self.file, self.file_frames = segment, video_stream_facts(segment)["frames"]
            return segment
        segment_facts = video_stream_facts(segment)
        if self.file is None:
            if self.source_file is None:
                segment, segment_facts = self._encoder_of_source(frames, segment, segment_facts)
                resolved = self._resolve_source_file(segment_facts)
                if resolved is None:
                    self._remove(segment)
                    return None
                self.source_file, self.source_frames = resolved
            previous, previous_frames = self.source_file, self.source_frames
        else:
            previous, previous_frames = self.file, self.file_frames
        joined = join_video_streams([previous, segment], [previous_frames, segment_facts["frames"]], self.fps, self._work_path())
        self._remove(segment)
        if previous is self.file:
            self._remove(previous)
        self.file, self.file_frames = joined, previous_frames + segment_facts["frames"]
        return joined
