# SUPRA-Tex observation-guided completion transformer, inference port of supra3d/supra-tex (supratex/models/
# interleaved_flow_transformer.py, observation_attention.py, interleaved_backbone.py, observation_stream.py, efficient_attention.py,
# supratex/pipelines/observation_guided_completion.py and samplers/flow_euler.py, MIT License, see LICENSE in this folder), with the
# TRELLIS.2 attention and feed-forward modules of WanGP's port (models/trellis2/dit.py).
# 60 blocks over the material latent cells X of the object; the observed cells O (the encoded visible surface) are a static stream:
# each block projects O's keys and values only (O modulated at time 0), the even blocks update the unobserved cells of X, the odd
# blocks all of X, their queries attending to O and all of X. Precision follows upstream's FP32 model under BF16 autocast: an FP32
# residual stream with FP32 norms and modulations, BF16 matmuls and attention (q/k norms and RoPE in FP32, rounded once).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import attention_kit
from shared.kernels import ops
from shared.utils.phase_progress import check_abort
from models.trellis2.dit import FeedForward, SelfAttention, TimestepEmbedder, rope_phases
from models.trellis2.sparse import SparseTensor


class Block(nn.Module):
    """_PSABlock: weightless norms, q/k RMS-normed self-attention with 3D RoPE, GELU feed-forward, a modulation offset per block."""

    def __init__(self, channels, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.self_attn = SelfAttention(channels, heads)
        self.mlp = FeedForward(channels, int(channels * mlp_ratio))
        self.modulation = nn.Parameter(torch.zeros(6 * channels))

    def modulations(self, mod):
        return (self.modulation.float() + mod).chunk(6, dim=1)

    def _modulated(self, rows, norm, shift, scale):
        """norm(rows) * (1 + scale) + shift in FP32 (FP32 rows), rounded to BF16 for the matmuls (autocast's cast)."""
        return ops.norm_modulate([rows], norm, shift, scale, [(0, rows.shape[0], 0)]).to(torch.bfloat16)

    def observation_kv(self, o, mod_o, rope_o):
        """Keys and values (1, observed, heads, head_dim) BF16 of the static observation rows (k normalized and rotated)."""
        shift, scale = self.modulations(mod_o)[:2]
        oo = self._modulated(o, self.norm1, shift, scale)
        attn = self.self_attn
        key, value = attn.to_k(oo).view(1, -1, attn.heads, attn.head_dim), attn.to_v(oo).view(1, -1, attn.heads, attn.head_dim)
        del oo
        ops.interleaved_norm_rope_(None, key, rope_o, None, attn.k_rms_norm.heads(range(attn.heads)))
        return key, value

    def _attention(self, queries, keys, kv_prefix, rope_q, rope_k):
        """Attention (1, queries, heads, head_dim) of the queries rows (handed off in a list, None: the key rows) over the observation
        prefix and the key rows (handed off in a list), q and k rotated by their own tables."""
        attn = self.self_attn

        def norm_rope(query, key, group):
            ops.interleaved_norm_rope_(query, None, rope_q, attn.q_rms_norm.heads(group.q), None)
            ops.interleaved_norm_rope_(None, key, rope_k, None, attn.k_rms_norm.heads(group.kv))

        def fused_quant(query, key, group, core, settings):
            if query is not None and key is not None:
                return None
            if query is not None:
                return ops.split_half_norm_rope_quant_(query, None, rope_q, attn.q_rms_norm.heads(group.q), None, core, settings, interleaved=True)
            return ops.split_half_norm_rope_quant_(None, key, rope_k, None, attn.k_rms_norm.heads(group.kv), core, settings, interleaved=True)

        if queries is None:
            return attention_kit.qkv_attention(keys, attn.to_q, attn.to_k, attn.to_v, attn.heads, attn.head_dim, norm_rope, fused_quant=fused_quant,
                                               kv_prefix=kv_prefix)
        return attention_kit.joint_qkv_attention([(queries, (attn.to_q, None, None)), (keys, (None, attn.to_k, attn.to_v))], attn.heads, attn.head_dim,
                                                 norm_rope, _cross=True, kv_prefix=kv_prefix, fused_quant=fused_quant)

    def forward(self, h, mod, observation, ropes, urows=None):
        """h (tokens, channels) FP32, updated in place: all its rows (urows None), else the unobserved rows urows only (their queries
        still attend to every row's keys). observation: (rows, time-0 modulation, rope table) of the static observation stream, whose
        keys and values (computed here, where MMGP has loaded the block) come before the rows' own."""
        kv_prefix = self.observation_kv(*observation) if len(observation[0]) else None
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulations(mod)
        xo = self._modulated(h, self.norm1, shift_msa, scale_msa).unsqueeze(0)
        rope_x, rope_u = ropes
        if urows is None:
            y = self._attention(None, [xo], kv_prefix, rope_x, rope_x)
            rows, spans = h, [(0, h.shape[0], 0)]
        else:
            uo = xo[:, urows]
            y = self._attention([uo], [xo], kv_prefix, rope_u, rope_x)
            rows, spans = h[urows], [(0, len(urows), 0)]
        del xo, kv_prefix
        ops.gated_linear(self.self_attn.to_out, [y.view(rows.shape[0], -1)], rows, gate_msa, spans)
        del y
        ops.mlp([self._modulated(rows, self.norm2, shift_mlp, scale_mlp)], self.mlp.mlp[0], self.mlp.mlp[2], "gelu", residual=rows, gate=gate_mlp[0])
        if urows is not None:
            h[urows] = rows
        return h


class CompletionTransformer(nn.Module):
    """InterleavedFlowTransformer (alternating_u_x, static observations, no refiners) of the released completion model."""

    def __init__(self, resolution=64, in_channels=64, out_channels=32, model_channels=1536, num_blocks=60, num_heads=12, mlp_ratio=5.3334,
                 rope_freq=(1.0, 10000.0), **_):
        super().__init__()
        self.model_channels, self.out_channels = model_channels, out_channels
        self.head_dim, self.rope_freq = model_channels // num_heads, tuple(rope_freq)
        for name in ("partial_mean", "partial_std", "pbr_mean", "pbr_std"):
            self.register_buffer(name, torch.zeros(32), persistent=True)
        self.t_embedder = TimestepEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.observation_metadata = nn.Linear(2, model_channels)
        self.observation_role = nn.Parameter(torch.zeros(model_channels))
        self.blocks = nn.ModuleList([Block(model_channels, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.out_layer = nn.Linear(model_channels, out_channels)

    def _bf16(self, layer, x):
        """A linear of the autocast region: BF16 input and output."""
        return layer(x.to(torch.bfloat16))

    def _time(self, t):
        """adaLN_modulation(t_embedder(t)) (1, 6 * channels) FP32 from its BF16 autocast computation."""
        embedder = self.t_embedder
        half = embedder.frequency_embedding_size // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device="cpu") / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = self._bf16(embedder.mlp[0], torch.cat([torch.cos(args), torch.sin(args)], dim=-1))
        emb = self._bf16(embedder.mlp[2], F.silu(emb))
        return self._bf16(self.adaLN_modulation[1], F.silu(emb)).float()

    def prepare(self, coords, shape, observation):
        """Static inputs of the sampling: coords (N, 3) of the latent cells, shape (N, 32) normalized shape latents, observation
        (SparseTensor, 34 channels) of the observed cells. Returns the observation stream, its rope table, the unobserved rows and the
        rope tables of all rows and of the unobserved ones."""
        rows = SparseTensor(None, coords).find(observation.coords).long()
        if bool((rows < 0).any()):
            raise RuntimeError("observed cells missing from the latent cells")
        feats = observation.feats.float()
        raw = feats[:, :32] * self.partial_std + self.partial_mean
        inputs = torch.cat([(raw - self.pbr_mean) / self.pbr_std, shape[rows]], 1)
        o = (self._bf16(self.input_layer, inputs) + self._bf16(self.observation_metadata, feats[:, 32:34])).float() + self.observation_role.float()
        observed = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
        observed[rows] = True
        urows = (~observed).nonzero().squeeze(1)
        table = lambda c: torch.view_as_real(rope_phases(c, self.head_dim, self.rope_freq))[None, :, None]
        rope_x = table(coords)
        mod_o = self._time(torch.zeros(1, device=coords.device))
        return {"o": o, "rope_o": table(observation.coords), "urows": urows, "ropes": (rope_x, rope_x[:, urows].contiguous()), "mod_o": mod_o}

    def forward(self, x, t, shape, prepared):
        """Velocity (N, 32) FP32 of the noisy material latents x (N, 32) FP32 at time t (1,) in [0, 1000]."""
        h = self._bf16(self.input_layer, torch.cat([x, shape], 1)).float()
        mod = self._time(t)
        observation = (prepared["o"], prepared["mod_o"], prepared["rope_o"])
        for index, block in enumerate(self.blocks):
            block(h, mod, observation, prepared["ropes"], prepared["urows"] if index % 2 == 0 else None)
            check_abort()
        h = F.layer_norm(h, h.shape[-1:])
        return self._bf16(self.out_layer, h).float()


def time_steps(steps, rescale_t=3.0):
    """FlowEulerSampler times from 1 to 0, rescaled towards 1."""
    t = np.linspace(1, 0, steps + 1)
    return (rescale_t * t / (1 + (rescale_t - 1) * t)).tolist()


def sample(model, noise, shape, prepared, steps=25, rescale_t=3.0, callback=None):
    """Euler sampling of the material latents (N, 32) FP32 from noise (upstream sample_material_latents, sigma_min 1e-5 only
    enters the x0 prediction it does not use)."""
    x = noise
    times = time_steps(steps, rescale_t)
    if callback is not None:
        callback(-1, None, True)
    for index, (t, t_prev) in enumerate(zip(times[:-1], times[1:])):
        velocity = model(x, torch.tensor([1000 * t], dtype=torch.float32, device=x.device), shape, prepared)
        x = x - (t - t_prev) * velocity
        del velocity
        if callback is not None:
            callback(index, None, False)
    return x
