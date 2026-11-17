# Hunyuan3D 2.1 ShapeVAE decoder and FlashVDM volume decoding, ported from Tencent Hunyuan3D 2.1
# hy3dshape/models/autoencoders/{model.py, attention_blocks.py, attention_processors.py, volume_decoders.py} (Copyright (C) 2025
# Tencent; Tencent Hunyuan 3D 2.1 Community License, see models/hunyuan3d/LICENSE and Notice.txt).
# Modified by DeepBeepMeep for WanGP: decoder only (the point-cloud encoder is not used); self-attention through shared.attention_kit and
# cross-attention through pay_attention on per-head q/k/v weights (rows reordered in the WanGP checkpoint, same products); the
# geometry decoder keys and values are projected once per mesh instead of once per query chunk (same values); the FlashVDM decoding
# of upstream FlashVDMVolumeDecoding(topk_mode="mean") kept operation for operation, with abort checks and progress per chunk.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from shared import attention_kit
from shared.attention import pay_attention
from shared.utils.phase_progress import PhaseProgress


def _layer_norm_(tensor, norm):
    """norm (LayerNorm over head_dim) of q or k (batch, tokens, heads, head_dim) in place, by chunks of tokens."""
    attention_kit.rows_(tensor, lambda part, first, last: F.layer_norm(part, norm.normalized_shape, norm.weight, norm.bias, norm.eps))


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=8):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_freqs, dtype=torch.float32), persistent=False)
        self.out_dim = 3 * (num_freqs * 2 + 1)

    def forward(self, x):
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)


class MLP(nn.Module):
    def __init__(self, width, expand_ratio=4):
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, width)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class SelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.c_proj = nn.Linear(width, width)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def _norm_(self, query, key, group):
        for tensor, norm in ((query, self.q_norm), (key, self.k_norm)):
            if tensor is not None:
                _layer_norm_(tensor, norm)

    def forward(self, x_list):
        return attention_kit.qkv_attention(x_list, self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, self._norm_, out_proj=self.c_proj)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = SelfAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLP(width)
        self.ln_2 = nn.LayerNorm(width, eps=1e-6)

    def forward(self, x):
        x = x + self.attn([self.ln_1(x)])
        return x.add_(self.mlp(self.ln_2(x)))


class CrossAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.c_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.c_proj = nn.Linear(width, width)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)


class CrossAttentionDecoder(nn.Module):
    """Occupancy logits of query points from the decoded latents (geo_decoder)."""

    def __init__(self, width, heads, num_freqs=8):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs)
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        self.cross_attn_decoder = nn.Module()
        block = self.cross_attn_decoder
        block.attn = CrossAttention(width, heads)
        block.ln_1 = nn.LayerNorm(width, eps=1e-6)
        block.ln_2 = nn.LayerNorm(width, eps=1e-6)
        block.ln_3 = nn.LayerNorm(width, eps=1e-6)
        block.mlp = MLP(width)
        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, 1)

    def keys_values(self, latents):
        """Normalized keys and values (batch, heads, latents, head_dim) of the latents: upstream projects them again for every chunk."""
        attn = self.cross_attn_decoder.attn
        data = self.cross_attn_decoder.ln_2(latents)
        key = attn.to_k(data).unflatten(-1, (attn.heads, attn.head_dim))
        _layer_norm_(key, attn.k_norm)
        return key.transpose(1, 2), attn.to_v(data).unflatten(-1, (attn.heads, attn.head_dim)).transpose(1, 2)

    def forward(self, queries, keys_values, select):
        """queries (batch, points, 3); select(query (batch, heads, points, head_dim), key, value) -> attention output in that layout."""
        block, attn = self.cross_attn_decoder, self.cross_attn_decoder.attn
        x = self.query_proj(self.fourier_embedder(queries).to(keys_values[0].dtype))
        query = attn.c_q(block.ln_1(x)).unflatten(-1, (attn.heads, attn.head_dim))
        _layer_norm_(query, attn.q_norm)
        out = select(query.transpose(1, 2), *keys_values)
        del query
        x = x + attn.c_proj(out.transpose(1, 2).flatten(2))
        del out
        x = x + block.mlp(block.ln_3(x))
        return self.output_proj(self.ln_post(x))


def _attend(query, key, value):
    """pay_attention in its (batch, tokens, heads, head_dim) layout, from and to (batch, heads, tokens, head_dim)."""
    qkv_list = [query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)]
    query = key = value = None
    return pay_attention(qkv_list).transpose(1, 2)


def _topk_kv(query, key, value, stride, topk):
    """FlashVDM adaptive KV selection: the latents most similar on average to every stride-th query."""
    sim = (query[:, :, ::stride, :] @ key.transpose(-1, -2)).mean(-2)
    index = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1).expand(-1, -1, -1, value.shape[-1])
    return torch.gather(key, dim=-2, index=index), torch.gather(value, dim=-2, index=index)


def _topk_count(latents):
    return 1024 if latents == 3072 else 256 if latents == 512 else latents // 3


def _dense_grid(bbox_min, bbox_max, resolution):
    axes = [np.linspace(bbox_min[i], bbox_max[i], int(resolution) + 1, dtype=np.float32) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _near_surface(volume, level):
    """Upstream extract_near_surface_volume_fn: voxels whose sign differs from one of their 6 neighbours (edges replicated)."""
    val = volume + level
    padded = F.pad(val[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    neighbors = [padded[2:, 1:-1, 1:-1], padded[:-2, 1:-1, 1:-1], padded[1:-1, 2:, 1:-1], padded[1:-1, :-2, 1:-1], padded[1:-1, 1:-1, 2:], padded[1:-1, 1:-1, :-2]]
    sign = torch.sign(val.to(torch.float32))
    same = torch.ones_like(val, dtype=torch.bool)
    for neighbor in neighbors:
        same &= torch.sign(torch.where(neighbor > -9000, neighbor, val).to(torch.float32)) == sign
    return (~same).to(torch.int32) * (val > -9000).to(torch.int32)


class ShapeVAEDecoder(nn.Module):
    def __init__(self, num_latents=4096, embed_dim=64, width=1024, heads=16, num_decoder_layers=16, num_freqs=8, scale_factor=1.0039506158752403, **_):
        super().__init__()
        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)
        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(num_decoder_layers)])
        self.geo_decoder = CrossAttentionDecoder(width, heads, num_freqs)

    def forward(self, latents):
        latents = self.post_kl(latents)
        for block in self.transformer.resblocks:
            latents = block(latents)
        return latents

    @torch.no_grad()
    def decode_volume(self, latents, bounds=1.01, num_chunks=8000, mc_level=0.0, octree_resolution=256, min_resolution=63, mini_grid_num=4):
        """Upstream FlashVDMVolumeDecoding.__call__ (topk_mode "mean") for one mesh: a coarse grid decoded by mini-grids with adaptive
        KV selection, then the voxels near the surface refined level by level. Returns the occupancy logits, NaN where not decoded."""
        device, dtype = latents.device, latents.dtype
        resolutions = [octree_resolution] if octree_resolution < min_resolution else []
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i in range(1, len(resolutions)):
            resolutions[i] = resolutions[0] * 2 ** i
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds] if isinstance(bounds, float) else bounds
        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        geo = self.geo_decoder
        key, value = geo.keys_values(latents)
        topk = _topk_count(key.shape[-2])

        xyz = torch.from_numpy(_dense_grid(bbox_min, bbox_max, resolutions[0])).to(device, dtype=dtype)
        grid_size = xyz.shape[0]
        mini = grid_size // mini_grid_num
        xyz = xyz.view(mini_grid_num, mini, mini_grid_num, mini, mini_grid_num, mini, 3).permute(0, 2, 4, 1, 3, 5, 6).reshape(-1, mini ** 3, 3)
        batch = max(num_chunks // xyz.shape[1], 1)
        # one progress unit per geometry decoder call; the refinement levels add their calls once their sizes are known
        progress_total = (xyz.shape[0] + batch - 1) // batch

        def coarse(query, k, v):
            k0, v0 = _topk_kv(query, k.expand(query.shape[0], -1, -1, -1), v.expand(query.shape[0], -1, -1, -1), 100, topk)
            return _attend(query, k0, v0)

        with PhaseProgress(progress_total, "Decoding Volume", "chunks") as progress:
            logits = []
            for start in range(0, xyz.shape[0], batch):
                logits.append(geo(xyz[start:start + batch], (key, value), coarse))
                progress.advance()
            grid_logits = torch.cat(logits, dim=0).reshape(mini_grid_num, mini_grid_num, mini_grid_num, mini, mini, mini).permute(0, 3, 1, 4, 2, 5).contiguous().view(1, grid_size, grid_size, grid_size)
            del logits, xyz

            dilate_weight = torch.ones((1, 1, 3, 3, 3), dtype=dtype, device=device)
            for resolution_now in resolutions[1:]:
                size = resolution_now + 1
                step = bbox_size / resolution_now
                next_index = torch.zeros((size, size, size), dtype=dtype, device=device)
                next_logits = torch.full((size, size, size), -10000.0, dtype=dtype, device=device)
                curr_points = _near_surface(grid_logits[0], mc_level)
                curr_points += grid_logits[0].abs() < 0.95
                expand_num = 0 if resolution_now == resolutions[-1] else 1
                for _ in range(expand_num):
                    curr_points = F.conv3d(curr_points[None, None].to(dtype), dilate_weight, padding=1)[0, 0]
                cidx_x, cidx_y, cidx_z = torch.where(curr_points > 0)
                next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
                del curr_points, cidx_x, cidx_y, cidx_z
                for _ in range(2 - expand_num):
                    next_index = F.conv3d(next_index[None, None], dilate_weight, padding=1)[0, 0]
                nidx = torch.where(next_index > 0)
                del next_index
                next_points = torch.stack(nidx, dim=1)
                next_points = next_points * torch.tensor(step, dtype=torch.float32, device=device) + torch.tensor(bbox_min, dtype=torch.float32, device=device)
                query_grid_num = 6
                min_val, max_val = next_points.min(axis=0).values, next_points.max(axis=0).values
                cell = torch.floor((next_points - min_val) / (max_val - min_val) * (query_grid_num - 0.001)).long()
                cell = (cell[..., 0] * query_grid_num ** 2 + cell[..., 1] * query_grid_num + cell[..., 2]).sort()
                next_points = next_points[cell.indices].unsqueeze(0).contiguous()
                cells, counts = torch.unique(cell.values, return_counts=True)
                chunks, current, total = [], [[], []], 0
                for cell_index, count in zip(cells.cpu().tolist(), counts.cpu().tolist()):
                    if total + count < num_chunks or total == 0:
                        total += count
                        current[0].append(cell_index)
                        current[1].append(count)
                    else:
                        chunks.append(current)
                        current, total = [[cell_index], [count]], count
                if total > 0:
                    chunks.append(current)
                progress.total += len(chunks)
                level_logits, start = [], 0
                for _, counts_list in chunks:
                    def refine(query, k, v, counts_list=counts_list):
                        outs, first = [], 0
                        for count in counts_list:
                            q_chunk = query[:, :, first:first + count, :]
                            k0, v0 = _topk_kv(q_chunk, k, v, 50, topk)
                            outs.append(_attend(q_chunk, k0, v0))
                            first += count
                        return torch.cat(outs, dim=-2)

                    chunk_size = sum(counts_list)
                    level_logits.append(geo(next_points[:, start:start + chunk_size], (key, value), refine))
                    start += chunk_size
                    progress.advance()
                values = torch.zeros((next_points.shape[1],), dtype=dtype, device=device)
                values[cell.indices] = torch.cat(level_logits, dim=1).squeeze(0).squeeze(-1)
                del level_logits, next_points, cell
                next_logits[nidx] = values
                grid_logits = next_logits.unsqueeze(0)
                del values, nidx, next_logits
        grid_logits[grid_logits == -10000.0] = float("nan")
        return grid_logits[0]
