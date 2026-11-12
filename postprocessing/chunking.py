# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Upsampling a long video a chunk at a time.

Nothing here knows an upsampler: each one declares its own chunk rules through its optional ``query_chunking(value)`` method, which
returns an object read with the fields below (every one optional). An upsampler without that method takes a video in one piece.

    get_chunk_frames(frames_count)        frames of a chunk when the model has a window of its own; without it WanGP sizes the chunks
                                          by memory
    chunk_frame_step                      chunks sized by WanGP hold n * chunk_frame_step + 1 frames (default: frame_step, 1)
    minimum_requested_frames              fewest frames of a chunk (default 1)
    overlap_frames, or get_overlap_frames(chunk_frames), or get_overlap_frames_for_target(chunk_frames, value)
                                          source frames a chunk shares with the one before it (default 0)
    crossfade_overlap_outputs             True: the shared frames fade from the previous chunk's output to this one's; False (default):
                                          they are this chunk's
    overlap_output_split(overlap_frames)  with crossfade_overlap_outputs, how many shared frames are kept from the previous chunk, the
                                          others being this chunk's; None fades
    output_write_range(returned_frames, source_write_start=, source_write_end=, next_overlap_frames=)
                                          the output frames kept from a chunk when it does not return one frame per source frame
    carries_cache(value)                  True: the upsampler is called with return_continue_cache=True and what it returns beside the
                                          frames is handed to the next chunk as continue_cache
    release_continue_cache()              called once the last chunk is done, or the video stopped

Every chunk also receives frame_offset, the place of its first frame in the video."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import torch

from shared.utils.audio_video import FrameChunks

CHUNK_BYTES = 2 * 2**30  # output frames of one chunk: a video that fits goes through its upsamplers in one piece
OVERLAP_SHARE = 4  # a chunk sized by memory holds at least this many times the frames it shares, which are computed twice
FRAME_BY_FRAME = SimpleNamespace()  # the rules of an upsampler whose frames do not depend on their neighbours


@dataclass(frozen=True)
class ChunkPlan:
    control_start_frame: int
    requested_frames: int
    overlap_frames: int


class FramePlanningError(RuntimeError):
    pass


def align_requested_frames(frame_count: int, *, frame_step: int, round_up: bool, frame_offset: int = 1) -> int:
    if frame_count <= frame_offset:
        return frame_offset
    if round_up:
        return int(math.ceil((frame_count - frame_offset) / float(frame_step)) * frame_step + frame_offset)
    return int(math.floor((frame_count - frame_offset) / float(frame_step)) * frame_step + frame_offset)


def build_chunk_plan(
    start_frame: int,
    end_frame_exclusive: int,
    total_source_frames: int,
    chunk_frames: int,
    *,
    frame_step: int,
    minimum_requested_frames: int,
    frame_offset: int = 1,
    overlap_frames: int,
    initial_overlap_frames: int = 0,
) -> list[ChunkPlan]:
    if chunk_frames < minimum_requested_frames:
        raise FramePlanningError("Chunk size is below the model minimum frame count.")
    if overlap_frames < 0:
        raise FramePlanningError("Sliding Window Overlap cannot be negative.")
    if initial_overlap_frames < 0:
        raise FramePlanningError("Initial overlap cannot be negative.")
    if overlap_frames >= chunk_frames:
        raise FramePlanningError("Sliding Window Overlap must stay below the computed chunk size.")
    if initial_overlap_frames >= chunk_frames:
        raise FramePlanningError("Initial overlap must stay below the computed chunk size.")
    available_unique_frames = end_frame_exclusive - start_frame
    if available_unique_frames <= 0:
        raise FramePlanningError("The selected range ends too close to the source video end to build a valid chunk for the current model.")

    def unique_bounds(plan_overlap_frames: int) -> tuple[int, int]:
        minimum_request = align_requested_frames(max(minimum_requested_frames, plan_overlap_frames + 1), frame_step=frame_step, frame_offset=frame_offset, round_up=True)
        return minimum_request - plan_overlap_frames, chunk_frames - plan_overlap_frames

    best_unique_frames = best_chunk_count = 0
    chunk_count = 1
    while True:
        first_minimum, first_maximum = unique_bounds(initial_overlap_frames)
        later_minimum, later_maximum = unique_bounds(overlap_frames)
        minimum_total = first_minimum + (chunk_count - 1) * later_minimum
        maximum_total = first_maximum + (chunk_count - 1) * later_maximum
        residue = (frame_offset - initial_overlap_frames + (chunk_count - 1) * (frame_offset - overlap_frames)) % frame_step if frame_step > 1 else 0
        candidate = min(available_unique_frames, maximum_total)
        if frame_step > 1:
            candidate -= (candidate - residue) % frame_step
        if candidate >= minimum_total and candidate > best_unique_frames:
            best_unique_frames, best_chunk_count = candidate, chunk_count
        if maximum_total >= available_unique_frames:
            break
        chunk_count += 1
    if best_unique_frames <= 0:
        raise FramePlanningError("The selected range ends too close to the source video end to build a valid chunk for the current model.")

    plans: list[ChunkPlan] = []
    cursor = start_frame
    remaining_unique = best_unique_frames
    for chunk_index in range(best_chunk_count):
        plan_overlap_frames = initial_overlap_frames if chunk_index == 0 else overlap_frames
        _, maximum_unique = unique_bounds(plan_overlap_frames)
        later_minimum, _ = unique_bounds(overlap_frames)
        remaining_minimum = (best_chunk_count - chunk_index - 1) * later_minimum
        requested_frames = align_requested_frames(min(maximum_unique, remaining_unique - remaining_minimum) + plan_overlap_frames,
                                                  frame_step=frame_step, frame_offset=frame_offset, round_up=False)
        control_start_frame = cursor - plan_overlap_frames
        max_available_frames = total_source_frames - control_start_frame
        if max_available_frames < requested_frames:
            raise FramePlanningError("The selected range ends too close to the source video end to build a valid chunk for the current model.")
        plans.append(ChunkPlan(control_start_frame=control_start_frame, requested_frames=requested_frames, overlap_frames=plan_overlap_frames))
        unique_frames = requested_frames - plan_overlap_frames
        remaining_unique -= unique_frames
        cursor += unique_frames
    if remaining_unique != 0:
        raise FramePlanningError("Unable to align the selected range to the current model frame shape.")
    return plans


def chunk_write_range(*, plan_overlap_frames: int, plan_requested_frames: int, next_overlap_frames: int, leading_overlap_already_written: bool) -> tuple[int, int]:
    """The source frames a chunk writes: the frames the next chunk shares with it are left to that chunk."""
    return (plan_overlap_frames if leading_overlap_already_written else 0), plan_requested_frames - next_overlap_frames


def chunk_output_write_range(rules, returned_frame_count: int, *, source_write_start: int, source_write_end: int, next_overlap_frames: int) -> tuple[int, int]:
    output_write_range = getattr(rules, "output_write_range", None)
    if callable(output_write_range):
        return output_write_range(returned_frame_count, source_write_start=source_write_start, source_write_end=source_write_end, next_overlap_frames=next_overlap_frames)
    return source_write_start, source_write_end


@torch.inference_mode()
def crossfade_video_overlap(previous_tail: torch.Tensor, current_head: torch.Tensor) -> torch.Tensor:
    overlap_frames = int(previous_tail.shape[1])
    weights = torch.linspace(0.0, torch.pi, overlap_frames, device=previous_tail.device, dtype=torch.float32).cos_().mul_(-0.5).add_(0.5).view(1, overlap_frames, 1, 1)
    return torch.lerp(previous_tail.float(), current_head.float(), weights).round_().to(current_head.dtype)


class JoinedChunk(NamedTuple):
    pieces: list  # the output frames to write, one piece after the other
    left: torch.Tensor | None  # the output frames left to the next chunk, which shares their source frames
    source_frames: int  # the source frames the pieces stand for


def join_chunk(rules, frames, left, *, overlap_frames: int, source_frames: int, next_overlap_frames: int, leading_overlap_already_written: bool = False) -> JoinedChunk:
    """A chunk's output joined to the previous chunk's, as the upsampler's rules declare it.

    frames: the output (C, frames, H, W) of source_frames source frames, whose first overlap_frames are shared with the previous chunk
    and last next_overlap_frames with the next one; left: what the previous chunk left (None for a first chunk, or one that resumes
    a video whose shared frames are written: leading_overlap_already_written)."""
    source_start, source_end = chunk_write_range(plan_overlap_frames=overlap_frames, plan_requested_frames=source_frames, next_overlap_frames=next_overlap_frames, leading_overlap_already_written=leading_overlap_already_written)
    start, end = chunk_output_write_range(rules, int(frames.shape[1]), source_write_start=source_start, source_write_end=source_end, next_overlap_frames=next_overlap_frames)
    if source_end <= source_start or end <= start:
        raise FramePlanningError("has no new frame to write after applying its overlap.")
    if start < 0 or end > frames.shape[1]:
        raise FramePlanningError(f"returned {int(frames.shape[1])} frame(s), but output range [{start}..{end}) was required.")
    pieces = [frames[:, start:end]]
    if left is not None and getattr(rules, "crossfade_overlap_outputs", False):  # the shared frames: faded from the previous chunk's to this one's, or split between them
        split_of = getattr(rules, "overlap_output_split", None)
        split = split_of(overlap_frames) if callable(split_of) else None
        head = crossfade_video_overlap(left, frames[:, :overlap_frames]) if split is None else torch.cat((left[:, :split], frames[:, split:overlap_frames]), dim=1)
        pieces = [head, frames[:, overlap_frames:end]]
    return JoinedChunk(pieces, frames[:, end:].clone() if next_overlap_frames > 0 else None, source_end - source_start)


def overlap_frames(rules, chunk_frames: int, value) -> int:
    """The source frames a chunk of chunk_frames shares with the one before it, as the upsampler's rules declare them for value."""
    for_target, for_chunk = getattr(rules, "get_overlap_frames_for_target", None), getattr(rules, "get_overlap_frames", None)
    if callable(for_target):
        return int(for_target(chunk_frames, value))
    return int(for_chunk(chunk_frames)) if callable(for_chunk) else int(getattr(rules, "overlap_frames", 0))


def chunk_shape(rules, value, frames_count: int, own_chunk_frames) -> tuple[int, int]:
    """The frames of a chunk and those it shares with the one before it: the model's window when its rules give one, else the size the
    caller chooses (own_chunk_frames())."""
    window = getattr(rules, "get_chunk_frames", None)
    chunk_frames = int(window(frames_count)) if callable(window) else int(own_chunk_frames())
    return chunk_frames, overlap_frames(rules, chunk_frames, value)


def carries_cache(rules, value) -> bool:
    query = getattr(rules, "carries_cache", None)
    return bool(query(value)) if callable(query) else False


def chunk_rules(value, temporal=False):
    """The chunk rules the upsampler of a Temporal or Spatial Upsampling value declares (query_chunking), None when it has none."""
    if temporal:
        from postprocessing import temporal_upsamplers
        upsampler = temporal_upsamplers.find_temporal_upsampler(value)
    else:
        from postprocessing import spatial_upsamplers
        upsampler = spatial_upsamplers.find_postprocessing_upsampler(value)
    query = getattr(upsampler, "query_chunking", None)
    return query(value) if callable(query) else None


def works_frame_by_frame(rules, value) -> bool:
    return not callable(getattr(rules, "get_chunk_frames", None)) and overlap_frames(rules, 2**30, value) == 0 and not carries_cache(rules, value)


def whole_video_reasons(temporal_upsampling="", spatial_upsampling=""):
    """Why a video cannot go through these upsamplers a chunk at a time; empty when it can."""
    reasons = []
    temporal = chunk_rules(temporal_upsampling, temporal=True) if temporal_upsampling else None
    if temporal_upsampling and temporal is None:
        reasons.append("the Temporal Upsampling method takes a video in one piece")
    if spatial_upsampling:
        from postprocessing import spatial_upsamplers
        upsampler, spatial = spatial_upsamplers.find_postprocessing_upsampler(spatial_upsampling), chunk_rules(spatial_upsampling)
        if spatial is None:
            reasons.append("the Spatial Upsampling method takes a video in one piece")
        elif upsampler.query_upsampler_def().get("source_audio_conditioning", False):
            reasons.append("the Spatial Upsampling method follows the sound track")
        elif temporal is not None and not works_frame_by_frame(spatial, spatial_upsampling):
            reasons.append("Temporal and Spatial Upsampling are both on")
    return reasons


def growth(temporal_upsampling="", spatial_upsampling=""):
    """How many frames an upsampled video holds for one of its source, and how many pixels a frame holds for one of its source."""
    from postprocessing import spatial_upsamplers, temporal_upsamplers
    temporal = temporal_upsamplers.split_temporal_upsampling_value(temporal_upsampling) if temporal_upsampling else None
    spatial = spatial_upsamplers.split_upsampling_value(spatial_upsampling) if spatial_upsampling else None
    return (float(temporal[1]) if temporal else 1.0), (float(spatial[1]) ** 2 if spatial else 1.0)


def chunk_plans(rules, value, frames_count: int, chunk_frame_bytes: float) -> list[ChunkPlan]:
    """The chunks of a video of frames_count frames for an upsampler's rules; chunk_frame_bytes: what a source frame weighs once
    processed. One chunk when the whole video fits CHUNK_BYTES."""
    budget = max(1, int(CHUNK_BYTES // chunk_frame_bytes))
    if frames_count <= budget:
        return [ChunkPlan(0, frames_count, 0)]
    minimum = int(getattr(rules, "minimum_requested_frames", 1))

    def by_memory():
        step = int(getattr(rules, "chunk_frame_step", getattr(rules, "frame_step", 1)))
        floor = max(minimum, OVERLAP_SHARE * overlap_frames(rules, frames_count, value) + 1)
        return max(align_requested_frames(budget, frame_step=step, round_up=False), align_requested_frames(floor, frame_step=step, round_up=True))
    chunk_frames, overlap = chunk_shape(rules, value, frames_count, by_memory)
    return build_chunk_plan(0, frames_count, frames_count, chunk_frames, frame_step=int(getattr(rules, "frame_step", 1)), minimum_requested_frames=minimum, overlap_frames=overlap)


class ProcessedVideo(FrameChunks):
    """A video gone through its upsamplers a chunk at a time, as save_video reads it: the chunks follow the rules of the upsampler that
    sets them, are joined as it declares and are never held together in memory. Only for upsamplers without whole_video_reasons.

    read_frames(start, count): frames of the source as a (frames, H, W, C) uint8 tensor, fewer when its file ends earlier.
    width, height: their size.
    temporal, spatial: (value, run) of each upsampler in use; run(frames, start_frame, cache, return_cache) returns the (C, frames, H, W)
    uint8 result and the cache for the next chunk, or (None, None) once the user stopped it.
    per_frame(frames): what is applied last to the frames given (Film Grain). progress(chunk_no, chunks): called before each chunk.
    frames: the output frames given so far. aborted: a run stopped the video before its end."""

    def __init__(self, read_frames, frames_count, width, height, temporal=None, spatial=None, per_frame=None, progress=None):
        self.read, self.frames_count, self.per_frame, self.progress = read_frames, int(frames_count), per_frame, progress
        self.driver, self.follower = (temporal, spatial) if temporal is not None else (spatial, None)  # the upsampler setting the chunks, the one after it
        self.value = None if self.driver is None else self.driver[0]
        self.rules = None if self.driver is None else chunk_rules(self.value, temporal=temporal is not None)
        frames_growth, pixels_growth = growth("" if temporal is None else temporal[0], "" if spatial is None else spatial[0])
        self.plans = chunk_plans(self.rules, self.value, self.frames_count, width * height * 3 * frames_growth * pixels_growth)
        self.frames, self.aborted = 0, False

    def _give(self, pieces):
        """Yield pieces of the output once through what follows the chunks; False when the user stopped it."""
        for piece in pieces:
            if piece is None or piece.shape[1] == 0:
                continue
            if self.follower is not None:
                piece, _ = self.follower[1](piece, self.frames, None, False)
                if piece is None:
                    self.aborted = True
                    return False
            if self.per_frame is not None:
                piece = self.per_frame(piece)
            self.frames += int(piece.shape[1])
            yield piece
        return True

    def __iter__(self):
        self.frames, self.aborted = 0, False
        rules, plans, minimum, return_cache = self.rules, self.plans, int(getattr(self.rules, "minimum_requested_frames", 1)), carries_cache(self.rules, self.value)
        cache = left = None  # left: the output frames a chunk leaves to the next one, which shares their source frames
        try:
            for no, plan in enumerate(plans):
                if self.progress is not None:
                    self.progress(no + 1, len(plans))
                frames = self.read(plan.control_start_frame, plan.requested_frames)
                read = 0 if frames is None else int(frames.shape[0])
                if read <= plan.overlap_frames or read < minimum:  # the file holds fewer frames than it announces: no chunk follows the previous one
                    yield from self._give([left])
                    return
                frames, last = frames.permute(3, 0, 1, 2), read < plan.requested_frames or no + 1 == len(plans)
                pieces = [frames]
                if self.driver is not None:
                    frames, cache = self.driver[1](frames, plan.control_start_frame, cache, return_cache)
                    if frames is None:
                        self.aborted = True
                        return
                    pieces, left, _ = join_chunk(rules, frames, left, overlap_frames=plan.overlap_frames, source_frames=read, next_overlap_frames=0 if last else plans[no + 1].overlap_frames)
                given = yield from self._give(pieces)
                frames = pieces = None
                if last or not given:
                    return
        finally:
            release = getattr(rules, "release_continue_cache", None)
            if callable(release):
                release()
