# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from diffusers ab96e2b58c89adb9df418af74d81cc8b1e1be074.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from diffusers.models.normalization import RMSNorm
from shared import attention_kit
from shared.kernels import ops
from shared.utils.phase_progress import GenerationAborted
from shared.utils.lora_mapping import convert_lora_keys
ADALN_EMBED_DIM = 256

class TimestepEmbedder(nn.Module):

    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, mid_size, bias=True), nn.SiLU(), nn.Linear(mid_size, out_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast('cuda', enabled=False):
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

def modulation_segments(batch: int, seq_len: int, noisy: tuple[int, int] | None=None) -> list[tuple[int, int, int]]:
    """(start, stop, row) over the rows of (batch * seq_len) for ops.norm_modulate and ops.gated_residual_: the modulation row of each
    span of tokens, row b for batch item b; with noisy = (start, stop), the span of noisy tokens, those tokens take row b and the clean
    ones row batch + b (the modulation rows of the noisy inputs first, then those of the clean inputs)."""
    if noisy is None:
        return [(b * seq_len, (b + 1) * seq_len, b) for b in range(batch)]
    start, stop = noisy
    segments = []
    for b in range(batch):
        base = b * seq_len
        if start > 0:
            segments.append((base, base + start, batch + b))
        segments.append((base + start, base + stop, b))
        if stop < seq_len:
            segments.append((base + stop, base + seq_len, batch + b))
    return segments

def _ffn_width(dim):
    return dim * 8 // 3 // 64 * 64


class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        inputs, x = [x], None
        return ops.gated_mlp(inputs, self.w1, self.w3, self.w2, lambda value: F.silu(value, inplace=True))

class FinalLayer(nn.Module):

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True))

    def forward(self, x, adaln, segments) -> torch.Tensor:
        """adaln: the modulation inputs, one row per batch item (or noisy then clean rows); segments: their spans of tokens
        (modulation_segments). norm_final(x) * (1 + scale[row]) in one fused pass."""
        scale = self.adaLN_modulation(adaln)
        batch, tokens, dim = x.shape
        x_list, x = [x.view(-1, dim)], None
        return self.linear(ops.norm_modulate(x_list, self.norm_final, torch.zeros_like(scale), scale, segments)).view(batch, tokens, -1)

class RopeEmbedder:

    def __init__(self, theta: float=256.0, axes_dims: list[int]=(16, 56, 56), axes_lens: list[int]=(64, 128, 128)):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), 'axes_dims and axes_lens must have the same length'
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: list[int], end: list[int], theta: float=256.0):
        with torch.device('cpu'):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / theta ** (torch.arange(0, d, 2, dtype=torch.float64, device='cpu') / d)
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)

def compute_gist_count(text_lengths: list[int], min_gist: int, max_gist: int, gist_step: int, min_text_len: int, max_text_len: int) -> list[int]:
    """Number of gist tokens for each prompt, from its text length in tokens.

    Lengths between `min_text_len` and `max_text_len` map linearly onto `[min_gist, max_gist]`, rounded up to a
    multiple of `gist_step`.
    """
    span = max_text_len - min_text_len
    counts = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        quantized = int(math.ceil(raw / gist_step)) * gist_step
        counts.append(max(min_gist, min(max_gist, quantized)))
    return counts

class BriaFibo2Attention(nn.Module):

    def __init__(self, query_dim: int, heads: int, dim_head: int, kv_dim: int | None=None, eps: float=1e-05):
        super().__init__()
        self.heads = heads
        kv_dim = kv_dim or query_dim
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, query_dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.head_dim = dim_head

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None=None, attention_mask: torch.Tensor | None=None, freqs_cis: torch.Tensor | None=None) -> torch.Tensor:
        # the complex frequencies (1 | batch, tokens, pairs) viewed as real are the (cos, sin) pairs of the rotation of channels (2p, 2p + 1)
        table = None if freqs_cis is None else torch.view_as_real(freqs_cis).unsqueeze(2)
        mask = None if attention_mask is None else attention_mask[:, None, None, :]
        hidden_list, hidden_states = ([hidden_states], None)
        options = dict(attention_mask=mask,
                       fused_quant=lambda query, key, group, core, settings: ops.split_half_norm_rope_quant_(query, key, table, self.norm_q, self.norm_k, core, settings, interleaved=True))
        norm_rope = lambda query, key, group: ops.interleaved_norm_rope_(query, key, table, self.norm_q, self.norm_k)
        if encoder_hidden_states is None:
            result = attention_kit.qkv_attention(hidden_list, self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, norm_rope, **options)
        else:
            result = attention_kit.cross_qkv_attention(hidden_list, [encoder_hidden_states], self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, norm_rope, **options)
        return self.to_out[0](result.flatten(2))

class BriaFibo2TransformerBlock(nn.Module):
    """Z-Image's block. With `cross_attention_dim` it becomes an injection block that also reads the text encoder."""

    def __init__(self, dim: int, n_heads: int, norm_eps: float, cross_attention_dim: int | None=None):
        super().__init__()
        self.attention = BriaFibo2Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=_ffn_width(dim))
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))
        if cross_attention_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps=norm_eps)
            self.cross_attention = BriaFibo2Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
            self.cross_attn_norm2 = RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attention = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None, freqs_cis: torch.Tensor, adaln: torch.Tensor, segments: list,
                encoder_hidden_states: torch.Tensor | None=None, encoder_attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        """adaln: the modulation inputs, one row per batch item, or the rows of the noisy inputs then those of the clean inputs;
        segments: the modulation row of each span of tokens (modulation_segments), so no per-token scale or gate tensor exists."""
        scale_msa, gate_msa, scale_mlp, gate_mlp = (t.contiguous() for t in self.adaLN_modulation(adaln).chunk(4, dim=1))
        gate_msa, gate_mlp = (gate_msa.tanh(), gate_mlp.tanh())
        shift = torch.zeros_like(scale_msa)  # scale-only modulation: norm(x) * (1 + scale[row]) + shift[row]
        batch, tokens, dim = x.shape  # the modulation works on the rows of (batch * tokens)
        x_list = [x.view(-1, dim)]
        attn_out = self.attention(ops.norm_modulate(x_list, self.attention_norm1, shift, scale_msa, segments).view(batch, tokens, dim), attention_mask=attn_mask, freqs_cis=freqs_cis)
        normalized, attn_out = (self.attention_norm2(attn_out), None)
        values, normalized = ([normalized], None)
        x = ops.gated_residual_(x, values, gate_msa, segments=segments)
        del scale_msa, gate_msa
        if self.cross_attention is not None:
            cross_out = self.cross_attention(self.cross_attn_norm1(x), encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask)
            normalized, cross_out = (self.cross_attn_norm2(cross_out), None)
            values, normalized = ([normalized], None)
            x = ops.gated_residual_(x, values, self.cross_attn_gate, tanh=True)
        x_list = [x.view(-1, dim)]
        values = [ops.norm_modulate(x_list, self.ffn_norm1, shift, scale_mlp, segments).view(batch, tokens, dim)]
        del scale_mlp, shift
        ffn_out = ops.mlp(values, self.feed_forward.w1, self.feed_forward.w2, "swiglu", up=self.feed_forward.w3)
        normalized, ffn_out = (self.ffn_norm2(ffn_out), None)
        values, normalized = ([normalized], None)
        return ops.gated_residual_(x, values, gate_mlp, segments=segments)

class BriaFibo2PerceiverLayer(nn.Module):
    """Gist tokens read the text encoder (cross-attention), then each other (self-attention), then an FFN."""

    def __init__(self, dim: int, n_heads: int, cross_attention_dim: int, norm_eps: float):
        super().__init__()
        self.cross_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.cross_attention = BriaFibo2Attention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
        self.self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.self_attention = BriaFibo2Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim=dim, hidden_dim=_ffn_width(dim))

    def forward(self, latents: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor | None=None, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        latents = latents + self.cross_attention(self.cross_attn_norm(latents), encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask)
        latents = latents + self.self_attention(self.self_attn_norm(latents), attention_mask=attention_mask)
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents

class BriaFibo2PerceiverResampler(nn.Module):
    """Squeezes the text encoder's hidden states into `num_gist_tokens` gist tokens."""

    def __init__(self, dim: int, n_heads: int, cross_attention_dim: int, num_queries: int, num_layers: int, norm_eps: float):
        super().__init__()
        self._interrupt = False
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([BriaFibo2PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps) for _ in range(num_layers)])
        self.norm_out = RMSNorm(dim, eps=norm_eps)

    def forward(self, encoder_hidden_states: torch.Tensor, num_gist_tokens: int, encoder_attention_mask: torch.Tensor | None=None, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        batch_size = encoder_hidden_states.shape[0]
        latents = self.latent_queries[:num_gist_tokens].unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            latents = layer(latents, encoder_hidden_states, encoder_attention_mask, attention_mask)
            if self._interrupt:
                raise GenerationAborted
        return self.norm_out(latents)

class BriaFibo2Transformer2DModel(nn.Module):
    _no_split_modules = ['BriaFibo2TransformerBlock', 'BriaFibo2PerceiverLayer']
    # Small per-pass projections stay with the base; nested Sequential modules
    # would otherwise become streamed blocks between the transformer towers.
    _offload_base_modules = ("t_embedder", "final_layer")

    def __init__(self, in_channels: int=32, patch_size: int=2, dim: int=3584, n_layers: int=44, n_refiner_layers: int=2, n_heads: int=28, norm_eps: float=1e-05, cap_feat_dim: int=7680, injection_layer_ids: tuple[int, ...]=(6, 14, 22, 30, 38), perceiver_num_layers: int=2, min_num_gist_tokens: int=256, max_num_gist_tokens: int=512, gist_step: int=64, gist_min_text_len: int=256, gist_max_text_len: int=2048, rope_theta: float=256.0, t_scale: float=1000.0, axes_dims: tuple[int, ...]=(32, 48, 48), axes_lens: tuple[int, ...]=(2048, 512, 512)):
        super().__init__()
        self.config = SimpleNamespace(**{k: v for k, v in locals().items() if k not in ('self', '__class__')})
        self.split_linear_modules_map = {
            "qkv": {"mapped_modules": ["to_q", "to_k", "to_v"], "split_sizes": [dim] * 3},
            "to_kv": {"mapped_modules": ["to_k", "to_v"], "split_sizes": [dim] * 2},
            "out": {"mapped_modules": ["to_out.0"], "split_sizes": [dim]},
            "to_out": {"mapped_modules": ["to_out.0"], "split_sizes": [dim]},
            "w13": {"mapped_modules": ["w1", "w3"], "split_sizes": [_ffn_width(dim)] * 2},
        }
        self._interrupt = False
        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)
        self.noise_refiner = nn.ModuleList([BriaFibo2TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_refiner_layers)])
        self.perceiver = BriaFibo2PerceiverResampler(dim, n_heads, cap_feat_dim, max_num_gist_tokens, perceiver_num_layers, norm_eps)
        self.layers = nn.ModuleList([BriaFibo2TransformerBlock(dim, n_heads, norm_eps, cross_attention_dim=cap_feat_dim if i in injection_layer_ids else None) for i in range(n_layers)])
        self.final_layer = FinalLayer(dim, patch_size * patch_size * in_channels)
        # These hold no weights themselves: their locks keep MMGP from rounding the float32 timestep to BF16 before t_scale and the FP32
        # sinusoid (the pipeline passes BF16 latents and text features)
        self.t_embedder._lock_dtype = self._lock_dtype = torch.float32

    def preprocess_loras(self, model_type, state_dict):
        state_dict = convert_lora_keys(state_dict, dict(self.named_modules()), split_linear_modules_map=self.split_linear_modules_map)
        width = _ffn_width(self.config.dim)
        padded = (width + 255) // 256 * 256  # ComfyUI pads each SwiGLU projection separately.
        if width != padded:
            for key, value in state_dict.items():
                if getattr(value, "ndim", None) != 2:
                    continue
                if key.endswith(".w13.lora_B.weight") and value.shape[0] == 2 * padded:
                    gate, up = value.split(padded)
                    if gate[width:].any() or up[width:].any():
                        raise ValueError(f"Fibo 2 LoRA {key} has nonzero rows in the padded FFN channels")
                    state_dict[key] = torch.cat((gate[:width], up[:width]))
                elif key.endswith(".w2.lora_A.weight") and value.shape[1] == padded:
                    # These columns multiply zeros because the padded gate and up rows are zero.
                    state_dict[key] = value[:, :width].contiguous()
        return state_dict

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing='ij')
        return torch.stack(grids, dim=-1)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        patch_size = self.config.patch_size
        latents = latents.view(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size)
        return latents.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, height // patch_size * (width // patch_size), -1)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor | None=None, context_latents: list[torch.Tensor] | None=None) -> tuple[torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, height, width)`):
                Normalized VAE latents of the image being generated.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                Noise level in `[0, 1]`: 1 is pure noise, 0 is a clean image.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, 1 + len(injection_layer_ids), text_len, cap_feat_dim)`):
                Text encoder features: the Perceiver's bundle first, then one bundle per injection block.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_len)`, *optional*):
                Bool mask of the real text tokens, for a batch of prompts padded to the same length. Leave it as `None`
                when nothing is padded.
            context_latents (`list[torch.Tensor]`, *optional*):
                Normalized VAE latents of the images to edit, one tensor of shape `(batch_size, in_channels, height,
                width)` per image, in order. Each image can have its own size. Their tokens condition the generation as
                clean tokens and are not predicted.
        Returns:
            A tuple containing the predicted velocity, with the shape of `hidden_states`.
        """
        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.config.patch_size
        height_tokens, width_tokens = (height // patch_size, width // patch_size)
        num_image_tokens = height_tokens * width_tokens
        device = hidden_states.device
        adaln_noisy = self.t_embedder(timestep * self.config.t_scale)
        adaln_clean = self.t_embedder(torch.zeros_like(timestep))
        image_plane = self.config.max_num_gist_tokens + 1
        tokens = [self._patchify(hidden_states)]
        ids = [self.create_coordinate_grid((1, height_tokens, width_tokens), (image_plane, 0, 0), device)]
        for k, latents in enumerate(context_latents or []):
            tokens.append(self._patchify(latents))
            context_size = (1, latents.shape[2] // patch_size, latents.shape[3] // patch_size)
            ids.append(self.create_coordinate_grid(context_size, (image_plane + k + 1, 0, 0), device))
        hidden_states = self.x_embedder(torch.cat(tokens, dim=1))
        image_freqs = self.rope_embedder(torch.cat([grid.flatten(0, 2) for grid in ids])).unsqueeze(0)
        # modulation rows: the noisy inputs (batch rows), then the clean ones; the refiner's context tokens are clean
        adaln = torch.cat([adaln_noisy, adaln_clean])
        refiner_segments = modulation_segments(batch_size, hidden_states.shape[1], (0, num_image_tokens) if context_latents else None)
        for block in self.noise_refiner:
            if self._interrupt:
                raise GenerationAborted
            hidden_states = block(hidden_states, None, image_freqs, adaln if context_latents else adaln_noisy, refiner_segments)
            if self._interrupt:
                raise GenerationAborted
        if encoder_attention_mask is not None:
            text_lengths = encoder_attention_mask.sum(dim=1).tolist()
        else:
            text_lengths = [encoder_hidden_states.shape[2]] * batch_size
        gist_counts = compute_gist_count(text_lengths, min_gist=self.config.min_num_gist_tokens, max_gist=self.config.max_num_gist_tokens, gist_step=self.config.gist_step, min_text_len=self.config.gist_min_text_len, max_text_len=self.config.gist_max_text_len)
        num_gist_tokens = max(gist_counts)
        gist_mask = None
        if min(gist_counts) < num_gist_tokens:
            gist_mask = torch.arange(num_gist_tokens, device=device) < torch.tensor(gist_counts, device=device)[:, None]
        gist = self.perceiver(encoder_hidden_states[:, 0], num_gist_tokens, encoder_attention_mask, gist_mask)
        gist_ids = self.create_coordinate_grid((num_gist_tokens, 1, 1), (1, 0, 0), device).flatten(0, 2)
        num_context_tokens = hidden_states.shape[1] - num_image_tokens
        gist_freqs = self.rope_embedder(gist_ids).unsqueeze(0)
        attention_mask = None
        if gist_mask is None:
            image_start = num_gist_tokens
            hidden_states = torch.cat([gist, hidden_states], dim=1)
            freqs_cis = torch.cat([gist_freqs, image_freqs], dim=1)
        else:
            image_start = 0
            hidden_states = torch.cat([hidden_states, gist], dim=1)
            freqs_cis = torch.cat([image_freqs, gist_freqs], dim=1)
            image_mask = torch.ones(batch_size, num_image_tokens + num_context_tokens, dtype=torch.bool, device=device)
            attention_mask = torch.cat([image_mask, gist_mask], dim=1)
        segments = modulation_segments(batch_size, hidden_states.shape[1], (image_start, image_start + num_image_tokens))
        bundle_index = {layer_id: index + 1 for index, layer_id in enumerate(self.config.injection_layer_ids)}
        for layer_id, block in enumerate(self.layers):
            if self._interrupt:
                raise GenerationAborted
            text = encoder_hidden_states[:, bundle_index[layer_id]] if layer_id in bundle_index else None
            hidden_states = block(hidden_states, attention_mask, freqs_cis, adaln, segments, encoder_hidden_states=text, encoder_attention_mask=encoder_attention_mask)
            if self._interrupt:
                raise GenerationAborted
        hidden_states = self.final_layer(hidden_states[:, image_start:image_start + num_image_tokens], adaln_noisy, modulation_segments(batch_size, num_image_tokens))
        hidden_states = hidden_states.view(batch_size, height_tokens, width_tokens, channels, patch_size, patch_size)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
        return (output,)
