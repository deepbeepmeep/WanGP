# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# The 4 / 6 / 8-step sigma grids and the (t, r) rule of the rows follow HyperFlow H3 Flash's schedule.py (Video Rebirth,
# https://github.com/Video-Rebirth/hyperflow-h3-flash, Apache License 2.0).
"""HyperFlow H3 Flash: MiniMax H3 with removed blocks, low-rank linears and two-time (t, r) conditioning for 4 to 8 steps.

Every AdaLN reads temb(t, r) = emb_t(t) + gate * (emb_r(r) - emb_t(t)), r being the time the step lands on. Generated video and audio
rows step to r; text rows follow the video; conditioning rows (keyframes, references, known audio) and rows pinned at their own noise
level go nowhere: r = t."""

import functools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from tqdm import tqdm

from mmgp import offload
from safetensors import safe_open

from .components.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG, build_row_timesteps

# raw (unshifted) grids, steps + 1 points: 4 favours speed, 6 (the default) balances speed and quality, 8 favours quality
SIGMA_GRIDS = {
    4: (1.0, 0.681872, 0.205019, 0.036496, 0.0),
    6: (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0),
    8: (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0),
}
HYPERFLOW_STEPS = tuple(sorted(SIGMA_GRIDS))
HYPERFLOW_DEFAULT_STEPS = 6
HYPERFLOW_VIDEO_SHIFT = 12.0
_AUDIO_SHIFT = 3.0


def _shift(sigma, shift):
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def video_sigmas(steps):
    """The shifted video noise levels of a step count, without the final 0 (the pipeline's explicit phase schedule)."""
    return [_shift(sigma, HYPERFLOW_VIDEO_SHIFT) for sigma in SIGMA_GRIDS[int(steps)][:-1]]


def grid_tail_sigmas(steps):
    """The last `steps` noise levels of the 8-step grid: a refinement phase from an upscaled draft (3 steps from 0.835, 4 from 0.923,
    8 from pure noise)."""
    if not 1 <= int(steps) <= 8:
        raise ValueError(f"A HyperFlow refinement phase runs 1 to 8 steps of the 8-step grid, got {steps}")
    return video_sigmas(8)[-int(steps):]


LORA_ADALN_PREFIX = "hyperflow_adaln."  # tensors of WanGP's HyperFlow 8-step LoRA repack that are not LoRA factors
_PAIR_TOLERANCE = 1e-4


class LoraAdaLNOffsets:
    """HyperFlow 8-step LoRA on a pruned model (WanGP's repack): pruned AdaLN curves cannot hold its time-embedder LoRAs, so the file
    stores the change of every block's and the final layer's modulation they cause at each (t, r) pair of its 8-step schedules, computed
    from the full model's AdaLN weights. The pruned AdaLN keeps reading the base curve at t; the offsets of the rows' pairs are added."""

    def __init__(self, path, architecture):
        prefix = f"{LORA_ADALN_PREFIX}{architecture}."
        with safe_open(path, framework="pt", device="cpu") as stream:
            if prefix + "pairs" not in stream.keys():
                raise ValueError(f"HyperFlow 8-step LoRA '{os.path.basename(path)}' has no AdaLN offsets for {architecture} "
                                 f"(it has {stream.metadata().get('adaln_architectures', 'none')})")
            self.pairs = stream.get_tensor(prefix + "pairs").double()
            self.blocks = stream.get_tensor(prefix + "blocks")  # [blocks, pairs, 18 * hidden]
            self.final = stream.get_tensor(prefix + "final")  # [pairs, 2 * hidden]

    def rows(self, t, r):
        """Index of each (t, r) pair in the table."""
        query = torch.stack((t.detach().double().cpu(), r.detach().double().cpu()), dim=1)
        distance, rows = (query[:, None] - self.pairs[None]).abs().amax(-1).min(1)
        if bool((distance > _PAIR_TOLERANCE).any()):
            t_value, r_value = query[int(distance.argmax())].tolist()
            raise ValueError(f"The HyperFlow 8-step LoRA has no AdaLN offsets for (t, r) = ({t_value:.6f}, {r_value:.6f}): it runs the 8-step grid "
                             "(a refinement phase: its last steps), with conditions at full strength")
        return rows

    def gather(self, rows, scale, device):
        """Offsets of the pairs `rows` for every block ([blocks, rows, 18 * hidden]) and for the final layer, on `device`."""
        blocks, final = self.blocks[:, rows].to(device), self.final[rows].to(device)
        if scale != 1.0:
            blocks.mul_(scale)
            final.mul_(scale)
        return blocks, final


@functools.lru_cache(maxsize=1)
def load_lora_offsets(path, architecture, mtime):
    return LoraAdaLNOffsets(path, architecture)


def checkpoint_lora_offsets(path, architecture):
    """The offsets of `architecture` (the checkpoint's map name, None when unknown) or, when the file has none for it, the official ones
    it has (WanGP's repack holds Ref2VA's only), with a warning."""
    with safe_open(path, framework="pt", device="cpu") as stream:
        available = sorted(key[len(LORA_ADALN_PREFIX):-len(".pairs")] for key in stream.keys() if key.startswith(LORA_ADALN_PREFIX) and key.endswith(".pairs"))
    if available and architecture not in available:
        tqdm.write(f"MiniMax H3 HyperFlow 8-step LoRA warning: it has no AdaLN offsets for this checkpoint ({architecture or 'unknown table'}). "
                   f"Using its official {available[0]} ones: its timestep changes will be approximate.")
        architecture = available[0]
    return load_lora_offsets(path, architecture, os.path.getmtime(path))


class LowRankWeight(torch.Tensor):
    """The weight of a low-rank linear as code inspecting linear modules sees it (shape, dtype, device); any operation on it works
    on the dense up @ down (a DoRA merge, for instance)."""
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, linear):
        up = linear.up.weight
        return torch.Tensor._make_wrapper_subclass(cls, (linear.out_features, linear.in_features), dtype=up.dtype, device=up.device)

    def __init__(self, linear):
        self.linear = linear

    def __repr__(self):
        return f"LowRankWeight(rank {self.linear.rank}, {tuple(self.shape)}, {self.dtype})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        materialize = lambda value: value.linear.dense_weight() if isinstance(value, LowRankWeight) else value
        return func(*tree_map(materialize, args), **tree_map(materialize, kwargs or {}))


class LowRankLinear(nn.Linear):
    """up(down(x)), the factorized form of a bias-free linear: seen as that linear by LoRAs, hooks and row projections
    (offload.linear_rows), so a base-model LoRA of the projection applies unchanged. The parameters are its down and up layers
    (down INT8 ConvRot in WanGP's checkpoints, up BF16)."""
    _mm_requires_native_linear_forward = True  # LoRA forwards call forward(), not F.linear on the weight

    def __init__(self, in_features, out_features, rank, dtype=None, device=None):
        nn.Module.__init__(self)
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.down = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(rank, out_features, bias=False, dtype=dtype, device=device)

    @property
    def weight(self):
        return LowRankWeight(self)

    @property
    def bias(self):
        return None

    def forward(self, x):
        return self.up(self.down(x))

    def dense_weight(self):
        """up @ down as one (out, in) matrix. down runs on the identity, so that each weight format gives its true values: an INT8 ConvRot
        down stores rotated rows, and one-hot inputs quantize exactly."""
        eye = torch.eye(self.in_features, dtype=self.up.weight.dtype, device=self.up.weight.device)
        return self.up.weight @ self.down(eye).T

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


class _LowRankRows:
    """offload.linear_rows of low-rank linears: down(x) once per linear, then a range of up's output rows."""
    @staticmethod
    def supports(module, x):
        return isinstance(module, LowRankLinear) and module.up.weight.dtype == x.dtype

    @staticmethod
    def key(module):
        return module

    @staticmethod
    def prepare(module, x):
        return module.down(x)

    @staticmethod
    def rows(module, h, start, stop):
        return F.linear(h, module.up.weight[start:stop])


offload.register_row_projection(_LowRankRows())


class DroppedBlock(nn.Module):
    """A block HyperFlow removed: never run, it keeps the indices of the next blocks (checkpoint keys, LoRAs)."""


def replace_low_rank_linears(model, ranks, dtype):
    """Put a LowRankLinear of the given rank in place of each named linear (checkpoint keys <name>.down.weight / <name>.up.weight)."""
    for name, rank in ranks.items():
        owner_name, _, attr = name.rpartition(".")
        owner = model.get_submodule(owner_name)
        dense = getattr(owner, attr)
        setattr(owner, attr, LowRankLinear(dense.in_features, dense.out_features, rank, dtype=dtype, device="meta"))


def two_time_rows(layout, timestep, timestep_indices, endpoints, condition_video_timestep, keyframe_timesteps, pinned=()):
    """(timestep, endpoint, indices): the distinct (t, r) pairs of the rows and each row's pair. timestep / timestep_indices are the
    rows' times of the base layout; pinned (start, stop, t) gives rows their own noise level (fixed or anchored video rows), with r = t.
    endpoints: (video r, audio r) of the step. Pairs are distinct as pairs: video and audio can share t and land on different r."""
    rows_t = timestep[timestep_indices]
    video_r, audio_r = endpoints
    end, end_indices = build_row_timesteps(layout, video_r, audio_r, condition_video_timestep, 1.0, keyframe_timesteps=keyframe_timesteps)
    rows_r = end.to(rows_t.device)[end_indices.to(rows_t.device)]
    for start, stop, value in pinned:
        rows_t[start:stop] = value
        rows_r[start:stop] = value
    unique, indices = torch.unique(torch.stack((rows_t, rows_r), dim=1), dim=0, sorted=True, return_inverse=True)
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), indices


def row_runs(indices, start, stop):
    """Runs (first, last, value) of equal values of indices[start:stop], relative to start."""
    part = indices[start:stop]
    changes = torch.cat((part.new_ones(1, dtype=torch.bool), part[1:] != part[:-1], part.new_ones(1, dtype=torch.bool))).nonzero().flatten().tolist()
    values = part[changes[:-1]].tolist()
    return [(first, last, value) for first, last, value in zip(changes[:-1], changes[1:], values)]


def _cpu_embedding(embedder, timesteps):
    """embedder(timesteps) (models.minimax_h3.transformer.TimeEmbedder) computed on the CPU from its weights, wherever MMGP holds them."""
    half = embedder.freq_dim // 2
    angles = timesteps.float().cpu().unsqueeze(1) * torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device="cpu") / half).unsqueeze(0)
    weights = lambda layer: (layer.weight.detach().float().cpu(), layer.bias.detach().float().cpu())
    return F.linear(F.silu(F.linear(torch.cat((angles.cos(), angles.sin()), dim=-1), *weights(embedder.proj_in))), *weights(embedder.proj_out))


def _sample_pairs():
    """(t, r) pairs of every HyperFlow schedule: generated video and audio rows, conditioning rows (r = t)."""
    pairs = set()
    for grid in SIGMA_GRIDS.values():
        for shift in (HYPERFLOW_VIDEO_SHIFT, _AUDIO_SHIFT):
            sigmas = [_shift(sigma, shift) for sigma in grid]
            for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
                pairs.add((1.0 - sigma, 1.0 - next_sigma))
                if shift == HYPERFLOW_VIDEO_SHIFT:
                    condition = max(1.0 - sigma, MINIMAX_H3_KEYFRAME_NOISE_AUG)
                    pairs.add((condition, condition))
    pairs.add((1.0, 1.0))
    return torch.tensor(sorted(pairs), dtype=torch.float32, device="cpu")


@torch.no_grad()
def _adaln_input_map(model, architecture):
    """M (17, 8): [silu(temb(t, r)), 1] @ M ~= the base model's rank-8 AdaLN curve at t, fitted at every (t, r) pair the schedules
    sample. A base-model AdaLN LoRA reads that curve; through M it reads HyperFlow's AdaLN input instead."""
    from .lora_affine import _load_affine_package

    cached = model.__dict__.get("_hyperflow_adaln_map")
    if cached is not None and cached[0] == architecture:
        return cached[1]
    table = _load_affine_package(architecture, 8)[0].double()
    pairs = _sample_pairs()
    t, r = pairs[:, 0], pairs[:, 1]
    current = _cpu_embedding(model.time_embedder, t)
    inputs = F.silu(current + model.endpoint_gate * (_cpu_embedding(model.endpoint_time_embedder, r) - current)).double()
    position = t.double().clamp(0, 1) * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    curve = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    design = torch.cat((inputs, inputs.new_ones(inputs.shape[0], 1)), dim=1)
    solution = torch.linalg.lstsq(design, curve, rcond=1e-12).solution
    error = (torch.linalg.vector_norm(design @ solution - curve) / torch.linalg.vector_norm(curve - curve.mean(0))).item()
    tqdm.write(f"MiniMax H3 HyperFlow: AdaLN LoRA map fitted on {pairs.shape[0]} (t, r) pairs, relative error {error:.3g}")
    model.__dict__["_hyperflow_adaln_map"] = (architecture, solution.float())
    return solution.float()


def convert_adaln_loras(model, model_type, state_dict, source_package=None):
    """AdaLN LoRAs of the base model (full width or pruned rank 8) moved onto HyperFlow's 16-wide AdaLN input: first onto the
    canonical rank-8 curve (lora_affine), then through the fitted map, the constant term going to the bias delta."""
    from .lora_affine import _LORA_SUFFIXES, _add_bias_delta, _architecture, _load_affine_package
    from .lora_affine import convert_adaln_loras as to_curve

    architecture = _architecture(model_type)
    table = _load_affine_package(architecture, 8)[0]
    to_curve(model_type, state_dict, table, source_package=source_package)
    mapping = None
    count = 0
    for down_suffix, up_suffix in _LORA_SUFFIXES:
        marker = "." + down_suffix
        for down_key in [key for key in state_dict if key.endswith(marker) and ".adaln_proj.linear." in key]:
            module_name = down_key[:-len(marker)]
            if mapping is None:
                mapping = _adaln_input_map(model, architecture)
            down = state_dict[down_key].float() @ mapping.T
            state_dict[down_key] = down[:, :-1].contiguous()
            _add_bias_delta(state_dict, module_name + ".diff_b", state_dict[module_name + "." + up_suffix].float() @ down[:, -1])
            count += 1
    for diff_key in [key for key in state_dict if key.endswith(".adaln_proj.linear.diff")]:
        if mapping is None:
            mapping = _adaln_input_map(model, architecture)
        diff = state_dict[diff_key].float() @ mapping.T
        state_dict[diff_key] = diff[:, :-1].contiguous()
        _add_bias_delta(state_dict, diff_key[:-len(".diff")] + ".diff_b", diff[:, -1])
        count += 1
    return count


__all__ = ["DroppedBlock", "HYPERFLOW_DEFAULT_STEPS", "HYPERFLOW_STEPS", "HYPERFLOW_VIDEO_SHIFT", "LORA_ADALN_PREFIX", "LoraAdaLNOffsets",
           "LowRankLinear", "SIGMA_GRIDS", "checkpoint_lora_offsets", "convert_adaln_loras", "grid_tail_sigmas", "load_lora_offsets", "replace_low_rank_linears",
           "row_runs", "two_time_rows", "video_sigmas"]
