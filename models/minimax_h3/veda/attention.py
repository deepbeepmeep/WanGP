# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Learned H3 tile selection with layer-owned, MMGP-managed predictor modules."""

import functools
import json
import math

import torch
from torch import nn
from safetensors import safe_open

from mmgp import offload
from shared.kernels import int8_backend, kernel_policy
from . import plans, selection, tiling


FORMAT = "wangp-h3-veda-v1"


def read_metadata(filename):
    with safe_open(filename, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
    if metadata.get("format") != FORMAT:
        raise ValueError(f"Expected a WanGP H3 Veda predictor module: {filename}")
    if tuple(int(metadata[key]) for key in ("num_layers", "num_heads", "head_dim")) != (50, 56, 128):
        raise ValueError("H3 Veda requires a 50-layer, 56-head, 128-channel predictor")
    return metadata


def _budget(metadata, prefix):
    kind = metadata.get(f"{prefix}_budget_kind", "ratio")
    value = float(metadata.get(f"{prefix}_budget_value", metadata.get("keep_ratio", "0.1")))
    return selection.Budget(**{kind: value})


def pool_tiles(x, layout):
    """FP32 TripPool over real token rows, excluding all padding from extrema."""
    tiles = x[:layout.n_video_tiles * tiling.TILE_SIZE].view(layout.n_video_tiles, tiling.TILE_SIZE, *x.shape[1:])
    count = layout.valid_count[:layout.n_video_tiles].clamp_min(1).float()
    mean = tiles.sum(1, dtype=torch.float32) / count[:, None, None]
    low, high = torch.aminmax(tiles, dim=1)
    partial = layout.partial_video_tiles
    if partial.numel():
        sub = tiles.index_select(0, partial)
        valid = torch.arange(tiling.TILE_SIZE, device=x.device)[None, :] < layout.valid_count[partial, None]
        high.index_copy_(0, partial, sub.masked_fill(~valid[:, :, None, None], -torch.inf).amax(1))
        low.index_copy_(0, partial, sub.masked_fill(~valid[:, :, None, None], torch.inf).amin(1))
    features = torch.cat((mean, high.float(), low.float()), -1)
    return torch.where(layout.kv_ok[:layout.n_video_tiles, None, None], features, 0).permute(1, 0, 2).contiguous()


class PredictorHead(nn.Module):
    def __init__(self, dim, dtype, device):
        super().__init__()
        self.proj_q = nn.Linear(3 * dim, dim, bias=False, dtype=dtype, device=device)
        self.proj_k = nn.Linear(3 * dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, q, k):
        dim = q.shape[-1] // 3
        # The shared Linear/qtype path owns weight precision and quantization.
        q_hat = self.proj_q(q.to(torch.bfloat16)).float() + q[:, :dim]
        k_hat = self.proj_k(k.to(torch.bfloat16)).float() + k[:, :dim]
        return q_hat @ k_hat.T / math.sqrt(dim)


class VedaAttention(nn.Module):
    def __init__(self, policy, layer, heads=56, dim=128, dtype=torch.bfloat16, device="cpu"):
        super().__init__()
        self.heads = nn.ModuleList(PredictorHead(dim, dtype, device) for _ in range(heads))
        self.policy = policy
        self.layer = layer

    def forward(self, qkv_list, group):
        from .sparse_int8 import attend
        from . import preprocess

        query, key, value = (tensor.squeeze(0) for tensor in qkv_list)
        qkv_list.clear()
        seq_len, heads, dim = query.shape
        out = query.new_empty(seq_len + 1, heads, dim)
        for layout, head_ids, local_ids in self.policy.groups(self.layer, group.q, query.device):
            # Bound tiled Q/K/V copies independently of the user's projection head split.
            chunk = max(1, min(8, (64 << 20) // (layout.num_slots * dim * query.element_size())))
            blocks = selection.column_blocks(layout, self.policy.generated, self.policy.reference)
            for start in range(0, len(head_ids), chunk):
                ids = head_ids[start:start + chunk]
                local = local_ids[start:start + chunk]
                if kernel_policy.allow_approximate():
                    q, qp = preprocess.gather_pool(query, layout, local)
                    k, kp = preprocess.gather_pool(key, layout, local)
                    q16, k16 = qp.to(torch.bfloat16), kp.to(torch.bfloat16)
                    q_hat = preprocess.residual(int8_backend.linear_grouped([self.heads[head].proj_q for head in ids], q16), qp)
                    k_hat = preprocess.residual(int8_backend.linear_grouped([self.heads[head].proj_k for head in ids], k16), kp)
                    scores = torch.bmm(q_hat, k_hat.transpose(1, 2)).div_(math.sqrt(dim))
                    q16 = k16 = q_hat = k_hat = None
                else:
                    q = tiling.gather_tiles(query, layout, local)
                    k = tiling.gather_tiles(key, layout, local)
                    qp, kp = pool_tiles(q, layout), pool_tiles(k, layout)
                    scores = torch.stack([self.heads[head](qp[i], kp[i]) for i, head in enumerate(ids)])
                qp = kp = None
                index, count = preprocess.indices(scores, layout, blocks)
                scores = None
                v = preprocess.gather(value, layout, local)
                output = attend(q, k, v, index, count, layout.valid_count)
                q = k = v = index = count = None
                preprocess.scatter(out, output, layout, local)
                output = None
        return out[:seq_len].unsqueeze(0)


class VedaPolicy:
    def __init__(self, metadata):
        self.plans = plans.PlanTable([plans.TilePlan.from_json(plan) for plan in json.loads(metadata["plans"]).values()])
        if any((p.num_layers, p.num_heads) != (50, 56) for p in self.plans.plans.values()):
            raise ValueError("H3 Veda tile plans do not match the predictor's layers and heads")
        self.generated, self.reference = _budget(metadata, "target"), _budget(metadata, "ref")
        self.enabled = False
        self.full_layers = frozenset()  # layers kept on full attention, set by each generation
        self.reset()

    def reset(self):
        self.signature = None
        self.plan = None
        self.layouts = {}
        self.head_groups = {}

    def begin_forward(self, layout, grid, payload, device, dtype):
        self.enabled = offload.shared_state.get("_attention") == "veda"
        if not self.enabled:
            return
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8 or dtype != torch.bfloat16:
            raise ValueError("H3 Veda requires BF16 on an NVIDIA SM80+ GPU with Triton")
        inverse = payload.get("target_video_inverse_order")
        signature = (payload["layout_signature"], device, id(inverse))
        if signature == self.signature:
            return
        self.reset()
        self.signature = signature
        choice = self.plans.select(grid)
        self.plan = choice.plan
        reference_grids = [(keyframe["latent_frame_count"], grid[1], grid[2]) for keyframe in payload.get("keyframes") or ()]
        reference_grids += [(ref.get("latent_t", 1), ref["latent_h"] // 2, ref["latent_w"] // 2)
                            for ref in payload.get("refs") or () if ref["kind"] != "audio"]
        cursor, spans = 0, []
        for ref_grid in reference_grids:
            spans.append(tiling.TiledSpan(int(layout.video_indices[cursor]), ref_grid, tiling.least_padding_shape(ref_grid)))
            cursor += math.prod(ref_grid)
        assert cursor == layout.num_condition_video_rows
        target_start = layout.sequence_length - math.prod(grid)
        for shape in set(self.plan.shapes):
            tile_spans = ([] if self.reference.keeps_all else spans) + [tiling.TiledSpan(target_start, grid, shape)]
            tiled = tiling.build_tile_layout(tile_spans, layout.sequence_length, device)
            if inverse is not None:
                # Natural 3D tiles address the grouped target rows through the inverse permutation.
                target = tiled.perm >= target_start
                mapped = target_start + inverse[(tiled.perm - target_start).clamp_min(0)]
                tiled.perm = torch.where(target, mapped, tiled.perm)
                tiled.gather_index = tiled.perm.clamp_min(0)
                tiled.scatter_index = torch.where(tiled.perm < 0, layout.sequence_length, tiled.perm)
            self.layouts[shape] = tiled
        full = f"; full attention on layers {', '.join(map(str, sorted(self.full_layers)))}" if self.full_layers else ""
        print(f"[MiniMax H3] Veda Triton Attention - {self.generated.describe()}, references {self.reference.describe()}; {choice.how}{full}")

    def groups(self, layer, heads, device):
        key = (layer, heads.start, heads.stop)
        if key not in self.head_groups:
            shapes = self.plan.head_shape[layer]
            groups = []
            for shape in sorted(set(shapes[h] for h in heads)):
                ids = tuple(h for h in heads if shapes[h] == shape)
                local = torch.tensor([h - heads.start for h in ids], device=device)
                groups.append((self.layouts[self.plan.shapes[shape]], ids, local))
            self.head_groups[key] = groups
        return self.head_groups[key]


def generation_context(method):
    """Release geometry buffers after every generation, including cancellation."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            policy = getattr(self.transformer, "veda_attention", None)
            if policy is not None:
                policy.reset()
    return wrapped
