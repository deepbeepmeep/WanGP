"""Qwen3-VL conditioning used by MiniMax H3."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig
from models.ideogram4.qwen3_vl_transformers import Qwen3VLTextModel, Qwen3VLVisionModel

from .interrupt import GenerationInterrupted
from shared.utils.phase_progress import text_encoding_progress

VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
VIDEO_PAD = 151656
TEXT_PAD = 151643


class _RawHiddenState(nn.Module):
    def forward(self, hidden_states):
        return hidden_states[0] if isinstance(hidden_states, list) else hidden_states


def load_h3_qwen_config(config_path, num_hidden_layers=50):
    config = Qwen3VLConfig.from_json_file(config_path)
    config.text_config.num_hidden_layers = num_hidden_layers or config.text_config.num_hidden_layers
    config.text_config.use_cache = False
    config.vision_config.out_hidden_size = config.text_config.hidden_size
    return config


def _visual_patches(frames, video=False, patch_size=16, temporal_patch_size=2, merge_size=2):
    """Convert exactly two THWC RGB frames into the Qwen3-VL patch layout."""
    if frames.shape[0] == 1:
        frames = frames.repeat(2, 1, 1, 1)
    if frames.shape[0] != temporal_patch_size:
        raise ValueError(f"Qwen visual blocks require {temporal_patch_size} frames, got {frames.shape[0]}")
    _, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Qwen visual input must be RGB, got {channels} channels")
    factor = patch_size * merge_size
    min_pixels, max_pixels = ((4096, 25165824) if video else (65536, 16777216))
    target_h = max(factor, round(height / factor) * factor)
    target_w = max(factor, round(width / factor) * factor)
    if target_h * target_w > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        target_h = max(factor, math.floor(height / scale / factor) * factor)
        target_w = max(factor, math.floor(width / scale / factor) * factor)
    elif target_h * target_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        target_h = math.ceil(height * scale / factor) * factor
        target_w = math.ceil(width * scale / factor) * factor

    images = F.interpolate(frames.permute(0, 3, 1, 2), size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
    images = images.mul(2.0).sub_(1.0)
    grid_h, grid_w = target_h // patch_size, target_w // patch_size
    patches = images.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size, patch_size,
                             grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flattened = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device=frames.device)
    return flattened, grid


def _mrope_positions(visuals, seq_len, device):
    if not visuals:
        return None
    positions = torch.zeros((3, seq_len), dtype=torch.long, device=device)
    cursor = offset = 0
    for visual in visuals:
        start, size, grid = visual["index"], visual["size"], visual["grid"]
        if cursor < start:
            positions[:, cursor:start] = torch.arange(cursor + offset, start + offset, device=device)
        end = start + size
        max_grid = int(grid.max()) // 2
        positions[0, start:end] = start + offset
        rows = int(grid[0, 1]) // 2
        cols = int(grid[0, 2]) // 2
        positions[1, start:end] = torch.arange(start + offset, start + rows + offset, device=device).unsqueeze(1).expand(rows, cols).reshape(-1)[:size]
        positions[2, start:end] = torch.arange(start + offset, start + cols + offset, device=device).unsqueeze(0).expand(rows, cols).reshape(-1)[:size]
        offset += max_grid - size
        cursor = end
    if cursor < seq_len:
        positions[:, cursor:] = torch.arange(cursor + offset, seq_len + offset, device=device)
    return positions.unsqueeze(1)


class MiniMaxH3TextEncoder(nn.Module):
    """Qwen3-VL-32B truncated after language layer 50, without final RMSNorm."""

    truncated = True
    mid_result_layers = None
    tokenizer_options = {}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        if self.truncated:
            self.language_model.norm = _RawHiddenState()
        self.tokenizer = None
        self._interrupt = False

    @property
    def _interrupt(self):
        return getattr(self, "_abort", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self._abort = bool(value)
        if hasattr(self, "visual"):
            self.visual._interrupt = self._abort
        if hasattr(self, "language_model"):
            self.language_model._interrupt = self._abort

    def set_tokenizer(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, **self.tokenizer_options)

    def _token_ids(self, text):
        if self.tokenizer is None:
            raise RuntimeError("The Qwen tokenizer has not been initialized")
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _presentation(self, prompt, items, picture_label=None):
        entries = []
        picture_label = picture_label or "<Picture {}>: "
        counters = {"image": 0, "audio": 0, "video": 0}

        def add_text(text):
            entries.extend(self._token_ids(text))

        def add_visual(frames, video=False):
            entries.extend((VISION_START, {"frames": frames, "video": video}, VISION_END))

        for item in items:
            kind = item["type"]
            counters[kind] += 1
            if kind == "image":
                add_text(picture_label.format(counters[kind]))
                if "frames" in item:  # a text-only presentation has the labels alone
                    add_visual(item["frames"])
            elif kind == "audio":
                add_text(f"<Audio {counters[kind]}>: ")
            elif kind == "video":
                add_text(f"<Video {counters[kind]}>: ")
                frames = item.get("frames")
                timestamps = list(item["timestamps"]) if "timestamps" in item else [i / 2.0 for i in range(frames.shape[0])]
                if len(timestamps) % 2:
                    if frames is not None:
                        frames = torch.cat((frames, frames[-1:]), dim=0)
                    timestamps.append(timestamps[-1])
                for index in range(0, len(timestamps), 2):
                    add_text(f"<{(timestamps[index] + timestamps[index + 1]) / 2.0:.1f} seconds>")
                    if frames is not None:
                        add_visual(frames[index:index + 2], video=True)
            else:
                raise ValueError(f"Unsupported H3 presentation item: {kind}")
        add_text(prompt)
        return entries or [TEXT_PAD]

    def _conditioning(self, output, dtype):
        return output.last_hidden_state.to(dtype)

    def encode(self, prompt, items, device, dtype, picture_label=None):
        entries = self._presentation(prompt, items, picture_label)
        token_ids = []
        visual_specs = []
        for entry in entries:
            if isinstance(entry, int):
                token_ids.append(entry)
            else:
                visual_specs.append({"placeholder": len(token_ids), "frames": entry["frames"], "video": entry["video"]})
                token_ids.append(VIDEO_PAD if entry["video"] else IMAGE_PAD)

        encoded_visuals = []
        for spec in visual_specs:
            if self._interrupt:
                raise GenerationInterrupted
            frames = spec["frames"].to(device=device, dtype=torch.float32)
            flattened, grid = _visual_patches(frames, video=spec["video"])
            self.visual._interrupt = self._interrupt
            merged, deepstack = self.visual(flattened, grid)
            if merged is None:
                raise GenerationInterrupted
            encoded_visuals.append({"placeholder": spec["placeholder"], "merged": merged, "deepstack": deepstack,
                                    "grid": grid, "video": spec["video"]})

        expanded_ids = []
        visuals = []
        visual_by_placeholder = {v["placeholder"]: v for v in encoded_visuals}
        for index, token_id in enumerate(token_ids):
            visual = visual_by_placeholder.get(index)
            if visual is None:
                expanded_ids.append(token_id)
                continue
            start = len(expanded_ids)
            size = visual["merged"].shape[0]
            expanded_ids.extend([VIDEO_PAD if visual["video"] else IMAGE_PAD] * size)
            visual["index"] = start
            visual["size"] = size
            visuals.append(visual)

        input_ids = torch.tensor(expanded_ids, dtype=torch.long, device=device).unsqueeze(0)
        embeds = self.language_model.embed_tokens(input_ids)
        visual_mask = torch.zeros((1, embeds.shape[1]), dtype=torch.bool, device=device)
        tags = torch.ones(embeds.shape[1], dtype=torch.long, device=device)
        deepstack = None
        for visual in visuals:
            start, end = visual["index"], visual["index"] + visual["size"]
            embeds[0, start:end] = visual["merged"].to(dtype=embeds.dtype)
            visual_mask[0, start:end] = True
            tags[max(0, start - 1):min(tags.shape[0], end + 1)] = 0
            deepstack = visual["deepstack"] if deepstack is None else [torch.cat((deepstack[i], visual["deepstack"][i]), dim=0) for i in range(len(deepstack))]

        positions = _mrope_positions(visuals, embeds.shape[1], device)
        self.language_model._interrupt = self._interrupt
        with text_encoding_progress(self.language_model.layers):
            output = self.language_model(inputs_embeds=embeds, position_ids=positions, visual_pos_masks=visual_mask if visuals else None,
                                         deepstack_visual_embeds=deepstack, use_cache=False, return_mid_results_layers=self.mid_result_layers)
        if output.last_hidden_state is None:
            raise GenerationInterrupted
        return self._conditioning(output, dtype), tags


class MiniMaxH3TextConnector(nn.Module):
    """Concatenated, standardized hidden states projected to the conditioning H3 reads; the first token, an attention sink, gets a fixed embedding.
    Follows PH3ImageTextConnector in diffusers (Copyright 2026 The HuggingFace Team, Apache License 2.0)."""

    def __init__(self, in_channels=12800, out_channels=5120):
        super().__init__()
        self.input_mean = nn.Parameter(torch.zeros(in_channels))
        self.input_std = nn.Parameter(torch.ones(in_channels))
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.output_mean = nn.Parameter(torch.zeros(out_channels))
        self.output_std = nn.Parameter(torch.ones(out_channels))
        self.sink_embedding = nn.Parameter(torch.zeros(out_channels))

    def forward(self, hidden_states):
        states = torch.cat(hidden_states, dim=-1).to(self.input_mean)
        hidden_states.clear()
        states = self.proj((states - self.input_mean) / self.input_std) * self.output_std + self.output_mean
        return torch.cat((self.sink_embedding.expand(states.shape[0], 1, -1), states[:, 1:]), dim=1)


class MiniMaxH3DistilledTextEncoder(MiniMaxH3TextEncoder):
    """Pruna's Qwen3-VL-4B, distilled to produce the signal of the 32B encoder: hidden states 13, 19, 25, 31 and the final normed one, through the connector."""

    truncated = False
    mid_result_layers = [12, 18, 24, 30]
    tokenizer_options = {"extra_special_tokens": {}}

    def __init__(self, config):
        super().__init__(config)
        self.text_connector = MiniMaxH3TextConnector()

    def _conditioning(self, output, dtype):
        states = [*output.mid_results, output.last_hidden_state]
        output.mid_results = output.last_hidden_state = None
        return self.text_connector(states).to(dtype)


__all__ = ["MiniMaxH3DistilledTextEncoder", "MiniMaxH3TextEncoder", "load_h3_qwen_config"]
