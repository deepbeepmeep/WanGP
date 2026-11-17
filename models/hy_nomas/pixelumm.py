# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, hymm/models/vae/pixelumm_decoder: model.py, autoencoder.py,
# streaming.py), Copyright (C) 2026 Tencent, Apache License 2.0, itself a port of nv-tlabs/PixelUMM (code Apache License 2.0). The released
# PixelUMM weights and their derivatives are licensed under the NVIDIA One-Way Noncommercial License. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""PixelUMM decoder of Hy-Nomas: DCAE-V latents (32 channels, 32x spatial, 4x temporal) to pixels in one denoising step of a 36-layer
pixel-space transformer (Qwen3-8B shape, GQA 32 / 8 heads, T / H / W MRoPE) over RGB tubelets of 4 x 16 x 16, conditioned on the latents.
A clip is decoded by windows of 10 latent frames overlapping by 5, cross-faded; attention inside a window is local in space (each tile
of 8 x 8 tokens sees the 3 x 3 tiles around it, shifted inward at the borders) and full in time."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import sage_staged
from shared.attention_kit import kv_first_attention
from shared.kernels import dispatch, ops
from mmgp import transfer
from shared.utils.phase_progress import check_abort, vae_decoding_progress

try:
    from . import triton_kernels as _fused
except ImportError:  # Triton not installed: the PyTorch q / k norms and MRoPE
    _fused = None

TUBELET, TUBE_DIM, LATENT_CHANNELS, LATENT_GRID = (4, 16, 16), 3072, 32, 2
HIDDEN, INTERMEDIATE, LAYERS, HEADS, KV_HEADS, HEAD_DIM, EPS = 4096, 12288, 36, 32, 8, 128, 1e-6
ROPE_THETAS, ROPE_SECTIONS = (1_000_000.0, 10_000.0, 10_000.0), (64, 32, 32)
WINDOW, OVERLAP, LOCAL_TILE, NOISE_SEED = 10, 5, 8, 42  # latent frames per window / overlap, tokens per local attention tile, decoder noise
_TILE_BATCH_BYTES = 512 << 20  # keys and values (all heads) gathered for one batched local attention
_QK_ROWS_BYTES = 16 << 20  # FP32 q / k rows normalized and rotated at a time


class RMSNorm(nn.Module):
    """Qwen3RMSNorm: normalized in FP32 (F.rms_norm, without FP32 copies of x), rounded to the input dtype, then scaled."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), eps=EPS).mul_(self.weight)


class PixelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        for name in ("q_norm_t", "q_norm_hw", "k_norm_t", "k_norm_hw"):
            setattr(self, name, RMSNorm(HEAD_DIM // 2))

    @staticmethod
    def _norm_rope_(x, norm_t, norm_hw, rope):
        """In place, by chunks of tokens: the T and HW halves of each head normalized by their own RMSNorm in FP32, rotated by the MRoPE
        sections, rounded once to x's dtype (the source keeps q / k in FP32 through norm and RoPE). The fused kernel makes it one pass,
        with the PyTorch path's error per head, but the decoder turns any rounding change into other fine textures: approximate."""
        if _fused is not None and dispatch.cuda_inference(x, approximate=True) and dispatch.choose("PixelUMM Q/K Norm + MRoPE", owned_triton=True):
            _fused.norm_mrope_(x, norm_t.weight, norm_hw.weight, rope, EPS)
            return
        cos, sin = rope
        step = max(1, _QK_ROWS_BYTES // (x[0, 0].numel() * 4))
        for start in range(0, x.shape[1], step):
            part = x[:, start:start + step].float()
            t_part, hw_part = part.chunk(2, dim=-1)
            part = torch.cat((norm_t(t_part), norm_hw(hw_part)), dim=-1)
            rotated = torch.cat([torch.cat((-piece[..., piece.shape[-1] // 2:], piece[..., :piece.shape[-1] // 2]), dim=-1)
                                 for piece in part.split(ROPE_SECTIONS, dim=-1)], dim=-1)
            x[:, start:start + step] = part.mul_(cos[:, start:start + step]).addcmul_(rotated, sin[:, start:start + step]).to(x.dtype)

    @staticmethod
    def _attend(q, k, v):
        """Grouped query attention: the staged SageAttention core takes the 8 key / value heads as they are, other attentions get them
        repeated for the 32 query heads. The output is written into q."""
        if sage_staged(q.device, HEAD_DIM) is None:
            k, v = (t.repeat_interleave(HEADS // KV_HEADS, dim=2) for t in (k, v))
        return kv_first_attention([q, k, v])

    def forward(self, x_list, rope, tiles):
        x = x_list.pop()
        tokens = x.shape[0]
        q = self.q_proj(x).view(1, tokens, HEADS, HEAD_DIM)
        k = self.k_proj(x).view(1, tokens, KV_HEADS, HEAD_DIM)
        v = self.v_proj(x).view(1, tokens, KV_HEADS, HEAD_DIM)
        del x
        self._norm_rope_(q, self.q_norm_t, self.q_norm_hw, rope)
        self._norm_rope_(k, self.k_norm_t, self.k_norm_hw, rope)
        if tiles is None:
            out = self._attend(q, k, v)
        else:
            # the tiles of equal shapes attend together, a batch at a time, their keys and values gathered with their own heads
            out = torch.empty_like(q)
            frames, height, width = tiles.grid
            grid = lambda t, heads: t.view(frames, height, width, heads, HEAD_DIM)
            q_grid, k_grid, v_grid, out_grid = grid(q, HEADS), grid(k, KV_HEADS), grid(v, KV_HEADS), grid(out, HEADS)
            for batch in tiles.batches:
                gather = lambda source, heads, spans: torch.stack([source[:, a:b, c:d] for a, b, c, d in spans]).view(len(spans), -1, heads, HEAD_DIM)
                queries = gather(q_grid, HEADS, [query for query, _ in batch])
                keys, values = (gather(t, KV_HEADS, [key for _, key in batch]) for t in (k_grid, v_grid))
                (r0, r1, c0, c1), _ = batch[0]
                result = self._attend(queries, keys, values).view(len(batch), frames, r1 - r0, c1 - c0, HEADS, HEAD_DIM)
                del queries, keys, values
                for index, ((r0, r1, c0, c1), _) in enumerate(batch):
                    out_grid[:, r0:r1, c0:c1] = result[index]
                del result
        del q, k, v
        return self.o_proj(out.view(tokens, HEADS * HEAD_DIM))


class PixelMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)


class PixelLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm(HIDDEN)
        self.attn = PixelAttention()
        self.post_attn_norm = RMSNorm(HIDDEN)
        self.mlp = PixelMLP()

    def forward(self, x, rope, tiles):
        x.add_(self.attn([self.input_norm(x)], rope, tiles))
        ops.mlp([self.post_attn_norm(x)], self.mlp.gate_proj, self.mlp.down_proj, "swiglu", up=self.mlp.up_proj, residual=x)


class PixelHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(HIDDEN, eps=EPS)
        self.linear = nn.Linear(HIDDEN, TUBE_DIM, bias=True)


class _LocalTiles:
    """Query tiles of the sliding-tile attention and the 3 x 3 tiles their keys come from, centred tiles clamped inward at the borders, in
    token rows / columns: ((query rows, query columns), (key rows, key columns)); full in time."""

    def __init__(self, grid):
        self.grid = grid
        _, height, width = grid
        tiles_h, tiles_w = -(-height // LOCAL_TILE), -(-width // LOCAL_TILE)
        span = lambda tile, count, size: (tile * LOCAL_TILE, min((tile + 1) * LOCAL_TILE, size))
        keys = lambda tile, count, size: ((min(max(tile, 1), max(count - 2, 1)) - 1) * LOCAL_TILE, min((min(max(tile, 1), max(count - 2, 1)) + 2) * LOCAL_TILE, size))
        self.pairs = [((*span(th, tiles_h, height), *span(tw, tiles_w, width)), (*keys(th, tiles_h, height), *keys(tw, tiles_w, width)))
                      for th in range(tiles_h) for tw in range(tiles_w)]
        # batches of tiles with equal query and key shapes, their gathered keys and values within _TILE_BATCH_BYTES
        shapes = {}
        for pair in self.pairs:
            (r0, r1, c0, c1), (n0, n1, m0, m1) = pair
            shapes.setdefault((r1 - r0, c1 - c0, n1 - n0, m1 - m0), []).append(pair)
        self.batches = []
        for (_, _, key_rows, key_columns), pairs in shapes.items():
            size = max(1, _TILE_BATCH_BYTES // (2 * grid[0] * key_rows * key_columns * HEADS * HEAD_DIM * 2))
            self.batches += [pairs[start:start + size] for start in range(0, len(pairs), size)]

    @staticmethod
    def of(grid):
        _, height, width = grid
        return None if -(-height // LOCAL_TILE) <= 3 and -(-width // LOCAL_TILE) <= 3 else _LocalTiles(grid)


def mrope_tables(grid, device):
    """cos / sin (1, T * H * W, 1, 128) for a T, H, W ordered grid, laid out [T | H | W] sections, each [angles | angles]."""
    positions = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
    cos, sin = [], []
    for position, dim, theta in zip(positions, ROPE_SECTIONS, ROPE_THETAS):
        inv_freq = theta ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        angles = position.flatten().float()[:, None] * inv_freq[None]
        angles = torch.cat((angles, angles), dim=-1)
        cos.append(angles.cos())
        sin.append(angles.sin())
    return torch.cat(cos, dim=-1)[None, :, None], torch.cat(sin, dim=-1)[None, :, None]


class PixelUMMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(TUBE_DIM, HIDDEN)
        self.cond_proj = nn.Linear(LATENT_CHANNELS, LATENT_GRID ** 2 * HIDDEN)
        self.layers = nn.ModuleList(PixelLayer() for _ in range(LAYERS))
        self.final_norm = RMSNorm(HIDDEN)
        self.head = PixelHead()

    def _predict(self, latent, xt):
        """x0 of one window: latent (1, 32, L, h, w), xt (1, 3, 4L, 32h, 32w) noise -> (1, 3, 4L, 32h, 32w) in the decoder dtype."""
        dtype = self.embed.weight.dtype
        _, _, frames, height, width = latent.shape
        grid = (frames, height * LATENT_GRID, width * LATENT_GRID)
        pt, ph, pw = TUBELET
        tubes = xt.to(dtype).reshape(3, frames, pt, grid[1], ph, grid[2], pw).permute(1, 3, 5, 2, 4, 6, 0).reshape(-1, TUBE_DIM)
        x = self.embed(tubes)
        del tubes
        condition = self.cond_proj(latent[0].to(dtype).permute(1, 2, 3, 0)).view(frames, height, width, LATENT_GRID, LATENT_GRID, HIDDEN)
        x.add_(condition.permute(0, 1, 3, 2, 4, 5).reshape(-1, HIDDEN))
        del condition
        rope, tiles = mrope_tables(grid, latent.device), _LocalTiles.of(grid)
        for layer in self.layers:
            layer(x, rope, tiles)
            check_abort()
        x = self.head.linear(self.head.norm(self.final_norm(x)))
        return x.view(frames, grid[1], grid[2], pt, ph, pw, 3).permute(6, 0, 3, 1, 4, 2, 5).reshape(1, 3, frames * pt, grid[1] * ph, grid[2] * pw)

    @staticmethod
    def _noise(latent, start, stop):
        """FP32 decoder noise of latent frames [start, stop): one generator per latent frame, as the source."""
        _, _, _, height, width = latent.shape
        frames = []
        for index in range(start, stop):
            generator = torch.Generator(device=latent.device).manual_seed(NOISE_SEED * 1_000_003 + index)
            frames.append(torch.randn((1, 3, 4, height * 32, width * 32), generator=generator, device=latent.device, dtype=torch.float32))
        return torch.cat(frames, dim=2)

    @staticmethod
    def window_starts(length):
        if length <= WINDOW:
            return [0]
        starts = list(range(0, length - WINDOW + 1, WINDOW - OVERLAP))
        if starts[-1] != length - WINDOW:
            starts.append(length - WINDOW)
        return starts

    @torch.inference_mode()
    def decode_to_ram(self, latent, to_ram, offset=0):
        """decode(latent, offset), clamped, returned in RAM in its final form (to_ram: a function of a piece of the video on the GPU,
        bytes for instance): the frames no later window overlaps go through to_ram and to their place in RAM as soon as their window is
        blended, without waiting for the GPU; the GPU holds a window and the overlap it shares with the next one, not the video."""
        frames = latent.shape[2]
        starts, length = self.window_starts(frames), min(WINDOW, frames)
        out, pending, first = None, None, 0  # pending: the pixels of frames [first, end) not yet final
        with vae_decoding_progress(len(starts), self.head.linear), transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
            for index, start in enumerate(starts):
                part = self._predict(latent[:, :, start:start + length], self._noise(latent, offset + start, offset + start + length))[0]
                pixel_start = start * TUBELET[0]
                if pending is None:
                    pending = part
                else:
                    end = first + pending.shape[1]
                    overlap = end - pixel_start
                    weight = (0.5 - 0.5 * torch.cos(math.pi * torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32) / (overlap + 1)))
                    weight = weight.view(1, -1, 1, 1).to(part.dtype)
                    region = pending[:, pixel_start - first:]
                    region.copy_(region * (1.0 - weight) + part[:, :overlap] * weight)
                    pending = torch.cat([pending, part[:, overlap:]], dim=1)
                del part
                final = (starts[index + 1] if index + 1 < len(starts) else frames) * TUBELET[0]  # (the frames the next window does not touch)
                piece = to_ram(pending[:, : final - first].clamp(-1, 1))
                if out is None:
                    out = torch.empty((piece.shape[0], frames * TUBELET[0], *piece.shape[2:]), dtype=piece.dtype, device="cpu")
                writer.write(out[:, first:final], piece)
                pending, first = pending[:, final - first:].clone(), final
                del piece
        return out

    @torch.inference_mode()
    def decode(self, latent, offset=0, head=None, clamp=True):
        """latent (1, 32, L, h, w) -> video (3, 4L, 32h, 32w) in [-1, 1] (unclamped without clamp): windows cross-faded with a raised
        cosine over their overlap. A part of a longer timeline (the source's streaming) gives offset, the timeline index of its first
        latent frame (each frame's decoder noise follows its index), and head, the pixels a previous window decoded over its first
        OVERLAP latent frames, cross-faded with the first window as in a decode of the whole timeline."""
        frames = latent.shape[2]
        starts, length = self.window_starts(frames), min(WINDOW, frames)
        out, end = None, 0
        if head is not None:
            out = head.new_empty((3, frames * TUBELET[0], *head.shape[2:]))
            out[:, :head.shape[1]] = head
            end = head.shape[1]
        with vae_decoding_progress(len(starts), self.head.linear):
            for start in starts:
                part = self._predict(latent[:, :, start:start + length], self._noise(latent, offset + start, offset + start + length))[0]
                pixel_start = start * TUBELET[0]
                if out is None:
                    out = part.new_empty((3, frames * TUBELET[0], *part.shape[2:]))
                    out[:, :part.shape[1]] = part
                else:
                    overlap = end - pixel_start
                    weight = (0.5 - 0.5 * torch.cos(math.pi * torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32) / (overlap + 1)))
                    weight = weight.view(1, -1, 1, 1).to(part.dtype)
                    out[:, pixel_start:end] = out[:, pixel_start:end] * (1.0 - weight) + part[:, :overlap] * weight
                    out[:, end:pixel_start + part.shape[1]] = part[:, overlap:]
                end = pixel_start + part.shape[1]
                del part
        return out.clamp_(-1, 1) if clamp else out
