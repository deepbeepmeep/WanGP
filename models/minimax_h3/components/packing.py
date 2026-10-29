# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MiniMax H3 native packed-token layouts for WanGP single-GPU inference.

The geometry follows MiniMax's raw-checkpoint SGLang implementation. WanGP omits
the isolated alignment tail because it cannot attend to live rows and would only
increase attention work.
"""

from dataclasses import dataclass

import numpy as np
import torch


MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999

_INTERP = 32
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0


@dataclass
class MiniMaxH3PackedSequence:
    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_target_condition_audio_latents: int = 0
    num_target_condition_video_rows: int = 0
    dense_spans: tuple = ()  # (start, stop) of every non-video-region row range in sequence order: text, keyframes, image references, audio
    reference_video_spans: tuple = ()  # (start, stop, (t, h, w)) of every reference video's rows, in sequence order
    target_condition_video_anchor: str = "tail"


@dataclass
class MiniMaxH3PreparedReference:
    kind: str
    has_audio: bool = False
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    @property
    def num_video_rows(self):
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self):
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS


def patchify_video_latents(latent, patch_size):
    patch_t, patch_h, patch_w = patch_size
    batch, channels, full_t, full_h, full_w = latent.shape
    t, h, w = full_t // patch_t, full_h // patch_h, full_w // patch_w
    packed = latent.reshape(batch, channels, t, patch_t, h, patch_h, w, patch_w)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(batch * t * h * w, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(rows, num_latent_frames, latent_height, latent_width, channels, patch_size):
    patch_t, patch_h, patch_w = patch_size
    t, h, w = num_latent_frames // patch_t, latent_height // patch_h, latent_width // patch_w
    packed = rows.reshape(-1, t, h, w, channels, patch_t, patch_h, patch_w)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_tokens(rows, num_audio_latents):
    return rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1]).permute(0, 2, 1).contiguous()


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return torch.from_numpy(np.linspace(left, left + ratio, dim // patch, endpoint=False) * _INTERP).to(torch.float64)


def _video_t_grid(length, origin, time_scale=1.0):
    spans = torch.tensor(
        [_FRAME_RESCALE * time_scale * _FRAME_PER_TOKEN[index % len(_FRAME_PER_TOKEN)] for index in range(length)],
        dtype=torch.float64,
    )
    return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def _keyframe_t_span(length, time_scale=1.0):
    spans = np.ones(length, dtype=np.float64) * _FRAME_RESCALE * time_scale
    for index, frames in enumerate(_FRAME_PER_TOKEN):
        spans[index::len(_FRAME_PER_TOKEN)] *= frames
    return float(spans.sum())


def _unpack_keyframe_anchor(entry):
    if not isinstance(entry, tuple):
        return entry, 1, None
    return (*entry, None) if len(entry) == 2 else entry


def _reference_t_span(length, time_scale=1.0):
    return sum(_FRAME_RESCALE * time_scale * _FRAME_PER_TOKEN[index % len(_FRAME_PER_TOKEN)] for index in range(length))


def _frame_grid(latent_height, latent_width, patch_h, patch_w, spatial_context=None):
    canvas_height, canvas_width, top, left = ((latent_height, latent_width, 0, 0)
                                                if spatial_context is None else spatial_context)
    sqrt_area = np.sqrt(canvas_height * canvas_width)
    full_height = _axis_from_sqrt_area(canvas_height, patch_h, sqrt_area)
    full_width = _axis_from_sqrt_area(canvas_width, patch_w, sqrt_area)
    height_start, width_start = top // patch_h, left // patch_w
    height = full_height[height_start:height_start + latent_height // patch_h]
    width = full_width[width_start:width_start + latent_width // patch_w]
    grids = torch.meshgrid(height, width, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), full_width


def _fill_audio_positions(position_ids, rows, length, origin, width_grid):
    time = origin + torch.arange(length, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        (torch.full((length,), float(width_grid[0]), dtype=torch.float64),
         torch.full((length,), float(width_grid[-1]), dtype=torch.float64))
    )


def _fill_audio_condition_positions(position_ids, start, anchors, history_origin, target_origin, width_grid):
    cursor, history_time = start, history_origin
    for entry in anchors:
        anchor, length = entry if isinstance(entry, tuple) else (entry, 1)
        rows = slice(cursor, cursor + length * MINIMAX_H3_AUDIO_CHANNELS)
        if anchor == "history":
            origin = history_time
            history_time += length
        elif anchor == "first":
            origin = target_origin
        else:
            raise ValueError(f"Unknown MiniMax H3 audio condition anchor {anchor!r}")
        _fill_audio_positions(position_ids, rows, length, origin, width_grid)
        cursor = rows.stop


def build_packed_sequence(text_token_tags, num_latent_frames, latent_height, latent_width, num_audio_latents,
                          patch_size, keyframe_anchors=(), video_time_scale=1.0, audio_condition_anchors=(),
                          target_condition_audio_latents=0, target_condition_video_frames=0,
                          target_condition_video_anchor="tail", target_spatial_context=None, audio_time_offset=0.0):
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    text_len = int(text_token_tags.shape[0])
    condition_video_rows = sum(_unpack_keyframe_anchor(anchor)[1] for anchor in keyframe_anchors) * rows_per_frame
    condition_audio_rows = sum(anchor[1] if isinstance(anchor, tuple) else 1 for anchor in audio_condition_anchors) * MINIMAX_H3_AUDIO_CHANNELS
    target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    video_rows = num_latent_frames * rows_per_frame
    sequence_length = text_len + condition_video_rows + condition_audio_rows + target_audio_rows + video_rows
    condition_video_start = text_len
    condition_audio_start = condition_video_start + condition_video_rows
    audio_start = condition_audio_start + condition_audio_rows
    video_start = audio_start + target_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    frame_grid, width_grid = _frame_grid(latent_height, latent_width, patch_h, patch_w, target_spatial_context)
    history_frames = sum(_unpack_keyframe_anchor(entry)[1] for entry in keyframe_anchors if _unpack_keyframe_anchor(entry)[0] == "history")
    target_origin = float(text_len) + _reference_t_span(history_frames, video_time_scale)
    target_times = _video_t_grid(num_latent_frames, target_origin, video_time_scale)
    condition_cursor = condition_video_start
    history_time = float(text_len)
    for entry in keyframe_anchors:
        anchor, condition_frames, frame_index = _unpack_keyframe_anchor(entry)
        rows = slice(condition_cursor, condition_cursor + condition_frames * rows_per_frame)
        condition = position_ids[rows].view(condition_frames, rows_per_frame, 3)
        if anchor == "history":
            condition[:, :, 0] = _video_t_grid(condition_frames, history_time, video_time_scale)[:, None]
            history_time += _reference_t_span(condition_frames, video_time_scale)
        elif anchor == "first":
            condition[:, :, 0] = target_times[:condition_frames, None]
        elif anchor == "last":
            condition[:, :, 0] = target_origin + _keyframe_t_span(num_latent_frames, video_time_scale) - _FRAME_RESCALE * video_time_scale
        elif anchor == "frame":
            condition[:, :, 0] = target_origin + frame_index * _FRAME_RESCALE * video_time_scale
        else:
            raise ValueError(f"Unknown MiniMax H3 keyframe anchor {anchor!r}")
        condition[:, :, 1:] = frame_grid[None]
        condition_cursor = rows.stop

    _fill_audio_condition_positions(position_ids, condition_audio_start, audio_condition_anchors,
                                    float(text_len), target_origin + audio_time_offset, width_grid)
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, target_origin + audio_time_offset, width_grid)
    target = position_ids[video_start:].view(num_latent_frames, rows_per_frame, 3)
    target[:, :, 0] = target_times[:, None]
    target[:, :, 1:] = frame_grid[None]

    text_indices = torch.arange(text_len)
    audio_indices = torch.arange(condition_audio_start, video_start)
    video_indices = torch.cat((torch.arange(condition_video_start, condition_audio_start), torch.arange(video_start, sequence_length)))
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    return MiniMaxH3PackedSequence(sequence_length, position_ids, token_tags, video_indices, audio_indices,
                                   text_indices, condition_video_rows, condition_audio_rows,
                                   target_condition_audio_latents, target_condition_video_frames * rows_per_frame,
                                   tuple(span for span in ((0, text_len), (condition_video_start, condition_audio_start), (condition_audio_start, audio_start), (audio_start, video_start)) if span[1] > span[0]),
                                   (), target_condition_video_anchor)


def build_ref2va_packed_sequence(text_token_tags, references, num_latent_frames, latent_height, latent_width,
                                 num_audio_latents, patch_size, video_time_scale=1.0, keyframe_anchors=(),
                                 audio_condition_anchors=(), target_condition_audio_latents=0,
                                 target_condition_video_frames=0, target_condition_video_anchor="tail",
                                 target_spatial_context=None, audio_time_offset=0.0):
    _, patch_h, patch_w = patch_size
    text_len = int(text_token_tags.shape[0])
    target_frame_grid, target_width_grid = _frame_grid(latent_height, latent_width, patch_h, patch_w,
                                                       target_spatial_context)
    rows_per_target_frame = target_frame_grid.shape[0]
    target_video_rows = num_latent_frames * target_frame_grid.shape[0]
    target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    keyframe_frames = sum(_unpack_keyframe_anchor(anchor)[1] for anchor in keyframe_anchors)
    keyframe_rows = keyframe_frames * rows_per_target_frame
    keyframe_audio_latents = sum(anchor[1] if isinstance(anchor, tuple) else 1 for anchor in audio_condition_anchors)
    keyframe_audio_rows = keyframe_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    condition_video_rows = keyframe_rows + sum(ref.num_video_rows for ref in references if ref.kind != "audio")
    condition_audio_rows = keyframe_audio_rows + sum(ref.num_audio_rows for ref in references)
    sequence_length = text_len + condition_video_rows + condition_audio_rows + target_audio_rows + target_video_rows
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)

    keyframe_start = text_len
    keyframe_audio_start = keyframe_start + keyframe_rows
    reference_start = keyframe_audio_start + keyframe_audio_rows
    video_indices = [torch.arange(keyframe_start, keyframe_audio_start)] if keyframe_rows else []
    audio_indices = [torch.arange(keyframe_audio_start, reference_start)] if keyframe_audio_rows else []
    cursor, time_cursor = reference_start, float(text_len)
    dense_spans = [(0, text_len)] + ([(keyframe_start, keyframe_audio_start)] if keyframe_rows else []) + ([(keyframe_audio_start, reference_start)] if keyframe_audio_rows else [])
    reference_video_spans = []
    for ref in references:
        if ref.kind == "image":
            rows = slice(cursor, cursor + ref.num_video_rows)
            cursor = rows.stop
            dense_spans.append((rows.start, rows.stop))
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_grid(ref.latent_height, ref.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = time_cursor
            position_ids[rows, 1:] = frame_grid
            time_cursor += 1.0
        elif ref.kind == "audio":
            rows = slice(cursor, cursor + ref.num_audio_rows)
            cursor = rows.stop
            dense_spans.append((rows.start, rows.stop))
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, ref.num_audio_latents, time_cursor, target_width_grid)
            time_cursor += float(ref.num_audio_latents)
        elif ref.kind == "video":
            audio_rows = slice(cursor, cursor + ref.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + ref.num_video_rows)
            cursor = video_rows.stop
            if ref.num_audio_rows:
                dense_spans.append((audio_rows.start, audio_rows.stop))
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_grid(ref.latent_height, ref.latent_width, patch_h, patch_w)
            reference_video_spans.append((video_rows.start, video_rows.stop, (ref.num_latent_frames, ref.latent_height // patch_h, ref.latent_width // patch_w)))
            _fill_audio_positions(position_ids, audio_rows, ref.num_audio_latents, time_cursor, width_grid)
            video_grid = position_ids[video_rows].view(ref.num_latent_frames, frame_grid.shape[0], 3)
            video_grid[:, :, 0] = _video_t_grid(ref.num_latent_frames, time_cursor, video_time_scale)[:, None]
            video_grid[:, :, 1:] = frame_grid[None]
            time_cursor += max(float(ref.num_audio_latents), _reference_t_span(ref.num_latent_frames, video_time_scale))
        else:
            raise ValueError(f"Unknown MiniMax H3 reference kind {ref.kind!r}")

    history_frames = sum(_unpack_keyframe_anchor(entry)[1] for entry in keyframe_anchors if _unpack_keyframe_anchor(entry)[0] == "history")
    target_origin = time_cursor + _reference_t_span(history_frames, video_time_scale)
    target_times = _video_t_grid(num_latent_frames, target_origin, video_time_scale)
    condition_cursor = keyframe_start
    history_time = time_cursor
    for entry in keyframe_anchors:
        anchor, condition_frames, frame_index = _unpack_keyframe_anchor(entry)
        rows = slice(condition_cursor, condition_cursor + condition_frames * rows_per_target_frame)
        condition = position_ids[rows].view(condition_frames, rows_per_target_frame, 3)
        if anchor == "history":
            condition[:, :, 0] = _video_t_grid(condition_frames, history_time, video_time_scale)[:, None]
            history_time += _reference_t_span(condition_frames, video_time_scale)
        elif anchor == "first":
            condition[:, :, 0] = target_times[:condition_frames, None]
        elif anchor == "last":
            condition[:, :, 0] = target_origin + _keyframe_t_span(num_latent_frames, video_time_scale) - _FRAME_RESCALE * video_time_scale
        elif anchor == "frame":
            condition[:, :, 0] = target_origin + frame_index * _FRAME_RESCALE * video_time_scale
        else:
            raise ValueError(f"Unknown MiniMax H3 keyframe anchor {anchor!r}")
        condition[:, :, 1:] = target_frame_grid[None]
        condition_cursor = rows.stop

    _fill_audio_condition_positions(position_ids, keyframe_audio_start, audio_condition_anchors,
                                    time_cursor, target_origin + audio_time_offset, target_width_grid)
    audio_start = cursor
    video_start = audio_start + target_audio_rows
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, target_origin + audio_time_offset,
                          target_width_grid)
    target = position_ids[video_start:].view(num_latent_frames, rows_per_target_frame, 3)
    target[:, :, 0] = target_times[:, None]
    target[:, :, 1:] = target_frame_grid[None]

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(text_len)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG
    return MiniMaxH3PackedSequence(sequence_length, position_ids, token_tags, video_indices, audio_indices,
                                   text_indices, condition_video_rows, condition_audio_rows,
                                   target_condition_audio_latents,
                                   target_condition_video_frames * rows_per_target_frame,
                                   tuple(dense_spans + [(audio_start, video_start)]), tuple(reference_video_spans), target_condition_video_anchor)


def build_row_timesteps(layout, video_timestep, audio_timestep, condition_video_timestep,
                        condition_audio_timestep, target_condition_timestep=1.0, keyframe_timesteps=()):
    timesteps = torch.full((layout.sequence_length,), video_timestep, dtype=torch.float32,
                           device=layout.token_tags.device)
    timesteps[layout.video_indices[:layout.num_condition_video_rows]] = condition_video_timestep
    for start, stop, timestep in keyframe_timesteps:  # softened keyframes keep their own noise level
        timesteps[layout.video_indices[start:stop]] = timestep
    if layout.num_target_condition_video_rows:
        # The frozen target rows sit at the clip's tail (the frozen-control
        # video mode) or at its head (the latent-branch Case A prefix);
        # everything else in the target block stays on the solver schedule.
        target_video_start = layout.num_condition_video_rows if layout.target_condition_video_anchor == "head" else \
            layout.video_indices.numel() - layout.num_target_condition_video_rows
        timesteps[layout.video_indices[target_video_start:target_video_start + layout.num_target_condition_video_rows]] = target_condition_timestep
    timesteps[layout.audio_indices[layout.num_condition_audio_rows:]] = audio_timestep
    timesteps[layout.audio_indices[:layout.num_condition_audio_rows]] = condition_audio_timestep
    condition_latents = layout.num_target_condition_audio_latents
    if condition_latents:
        target_audio_indices = layout.audio_indices[layout.num_condition_audio_rows:]
        channel_latents = target_audio_indices.numel() // MINIMAX_H3_AUDIO_CHANNELS
        timesteps[target_audio_indices[:condition_latents]] = condition_audio_timestep
        timesteps[target_audio_indices[channel_latents:channel_latents + condition_latents]] = condition_audio_timestep
    return torch.unique(timesteps, sorted=True, return_inverse=True)


__all__ = ["MINIMAX_H3_AUDIO_TAG", "MINIMAX_H3_KEYFRAME_NOISE_AUG", "MINIMAX_H3_TEXT_TAG",
           "MINIMAX_H3_VIDEO_TAG", "MiniMaxH3PackedSequence", "MiniMaxH3PreparedReference",
           "build_packed_sequence", "build_ref2va_packed_sequence", "build_row_timesteps",
           "patchify_video_latents", "unpack_audio_tokens", "unpatchify_video_tokens"]
