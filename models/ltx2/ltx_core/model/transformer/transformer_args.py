from dataclasses import dataclass
import os
from collections.abc import Callable

import torch
from tqdm import tqdm
import torch.nn.functional as F
from mmgp import cuda_graphs

from .adaln import AdaLayerNormSingle
from .modality import Modality, PreparedConditioning
from .rope import (
    LTXRopeType,
    RopeCache,
    build_rope_cache,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
)
from .text_projection import PixArtAlphaTextProjection
from .timestep_embedding import get_timestep_embedding


KeyframesEmbeddingProvider = Callable[[], torch.Tensor | None]


def apply_keyframes_absolute_embedding(hidden_states: torch.Tensor, keyframes_mask: torch.Tensor | None, embedding_provider: KeyframesEmbeddingProvider | None) -> torch.Tensor:
    if keyframes_mask is None or embedding_provider is None:
        return hidden_states
    embedding = embedding_provider()
    if embedding is None:
        return hidden_states
    mask = (keyframes_mask > 0).to(device=hidden_states.device, dtype=hidden_states.dtype)
    # in place, without two temporaries of the hidden state's size: the mask is 0 or 1, so the values are those of hidden + mask * embedding
    return hidden_states.addcmul_(mask, embedding.to(device=hidden_states.device, dtype=hidden_states.dtype))


@dataclass(frozen=True, eq=False)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: RopeCache
    cross_positional_embeddings: RopeCache | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    cross_attention_mask: torch.Tensor | None = None
    enabled: bool = True
    nag: dict | None = None
    prompt_timestep: torch.Tensor | None = None
    self_attention_mask: torch.Tensor | None = None
    ref_context: torch.Tensor | None = None


def _can_broadcast_timesteps(frame_indices: torch.Tensor | None, frame_count: int) -> bool:
    """Whether the tokens come by frames of equal length in order (timesteps then one per frame): checked on the device once per
    frame_indices tensor (one per latent state's plan, reused at every step) and frame count, not at every call (the check synchronizes)."""
    if frame_indices is None or frame_indices.ndim != 2:
        return False
    verdicts = frame_indices.__dict__.setdefault("_broadcast_verdicts", {})
    if frame_count not in verdicts:
        verdicts[frame_count] = _check_broadcast_timesteps(frame_indices, frame_count)
    return verdicts[frame_count]


def _check_broadcast_timesteps(frame_indices: torch.Tensor, frame_count: int) -> bool:
    token_count = frame_indices.shape[1]
    if frame_count <= 0 or token_count % frame_count != 0:
        return False
    tokens_per_frame = token_count // frame_count
    if tokens_per_frame == 0:
        return False
    try:
        frame_view = frame_indices.reshape(frame_indices.shape[0], frame_count, tokens_per_frame)
    except RuntimeError:
        return False
    expected = torch.arange(frame_count, device=frame_indices.device).view(1, frame_count, 1)
    return torch.equal(frame_view, expected.expand(frame_indices.shape[0], -1, tokens_per_frame))


def _maybe_compress_timesteps(
    timestep: torch.Tensor, frame_indices: torch.Tensor | None, batch_size: int
) -> torch.Tensor:
    if frame_indices is None or timestep.ndim != 2:
        return timestep
    token_count = frame_indices.shape[1]
    if timestep.shape[1] != token_count:
        return timestep
    frame_count = int(frame_indices.max().item()) + 1
    if not _can_broadcast_timesteps(frame_indices, frame_count):
        return timestep
    tokens_per_frame = token_count // frame_count
    if tokens_per_frame <= 1:
        return timestep
    return timestep.reshape(batch_size, frame_count, tokens_per_frame)[:, :, 0]


def _embed_timestep_functional(
    adaln: AdaLayerNormSingle,
    timestep: torch.Tensor,
    hidden_dtype: torch.dtype,
    out_device: torch.device,
) -> torch.Tensor:
    # the weights are read outside the embedder's forward: MMGP keeps them with the model's base (X0Model._offload_base_modules)
    time_proj, embedder = adaln.emb.time_proj, adaln.emb.timestep_embedder
    linear_1, linear_2 = embedder.linear_1, embedder.linear_2
    timesteps_proj = get_timestep_embedding(
        timestep.to(device=linear_1.weight.device),
        time_proj.num_channels,
        flip_sin_to_cos=time_proj.flip_sin_to_cos,
        downscale_freq_shift=time_proj.downscale_freq_shift,
        scale=time_proj.scale,
    ).to(dtype=linear_1.weight.dtype)
    hidden_states = F.linear(timesteps_proj, linear_1.weight, linear_1.bias)
    hidden_states = F.silu(hidden_states)
    hidden_states = F.linear(hidden_states, linear_2.weight, linear_2.bias)
    return hidden_states.to(device=out_device, dtype=hidden_dtype)


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection | None,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        prompt_adaln: AdaLayerNormSingle | None = None,
        keyframes_embedding_provider: KeyframesEmbeddingProvider | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln
        self.keyframes_embedding_provider = keyframes_embedding_provider
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type

    def _rope_cache_key(
        self,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
        rope_axes: tuple[int, ...] | None,
        rope_max_pos: list[int] | None,
    ) -> tuple:
        resolved_axes = tuple(range(len(max_pos))) if rope_axes is None else rope_axes
        resolved_max = tuple(max_pos[axis] for axis in resolved_axes) if rope_max_pos is None else tuple(rope_max_pos)
        return (
            inner_dim,
            tuple(max_pos),
            use_middle_indices_grid,
            num_attention_heads,
            x_dtype,
            resolved_axes,
            resolved_max,
        )

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
        frame_indices: torch.Tensor | None = None,
        scale_multiplier: float | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""
        timestep = _maybe_compress_timesteps(timestep, frame_indices, batch_size)
        if scale_multiplier is None:
            scale_multiplier = self.timestep_scale_multiplier
        timestep = timestep * scale_multiplier
        embedded_timestep = adaln.emb(timestep.flatten(), hidden_dtype=hidden_dtype)
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return self._finalize_timestep_outputs(embedded_timestep, adaln, batch_size, frame_indices)

    def _normalize_sigma_batch(self, sigma: torch.Tensor, reference: torch.Tensor, batch_size: int) -> torch.Tensor:
        sigma_batch = sigma.to(device=reference.device, dtype=reference.dtype)
        if sigma_batch.ndim == 0:
            return sigma_batch.expand(batch_size)
        return sigma_batch.reshape(batch_size, -1)[:, 0]

    def _build_base_timestep(
        self,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        batch_size: int,
        frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        compressed_timestep = _maybe_compress_timesteps(timestep, frame_indices, batch_size)
        if compressed_timestep.ndim < 1:
            return None
        sigma_batch = self._normalize_sigma_batch(sigma, compressed_timestep, batch_size)
        if torch.any(sigma_batch == 0):
            return None
        sigma_view = sigma_batch.view(batch_size, *([1] * (compressed_timestep.ndim - 1)))
        return compressed_timestep / sigma_view

    def _prepare_timestep_from_base(
        self,
        base_timestep: torch.Tensor | None,
        sigma: torch.Tensor,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
        frame_indices: torch.Tensor | None = None,
        scale_multiplier: float | int | None = None,
        include_embedded: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if base_timestep is None:
            return None, None
        if scale_multiplier is None:
            scale_multiplier = self.timestep_scale_multiplier
        sigma_batch = self._normalize_sigma_batch(sigma, base_timestep, batch_size)
        sigma_view = sigma_batch.view(batch_size, *([1] * (base_timestep.ndim - 1)))
        step_timestep = base_timestep * sigma_view * scale_multiplier
        embedded_timestep = _embed_timestep_functional(
            adaln,
            step_timestep.flatten(),
            hidden_dtype,
            base_timestep.device,
        )
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        timestep, embedded = self._finalize_timestep_outputs(embedded_timestep, adaln, batch_size, frame_indices)
        if include_embedded:
            return timestep, embedded
        return timestep, None

    def _finalize_timestep_outputs(
        self,
        embedded_timestep: torch.Tensor,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = adaln.linear(adaln.silu(embedded_timestep.reshape(-1, embedded_timestep.shape[-1])))
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])

        # Batch-level timesteps (for example prompt sigma in cross-attn AdaLN) are shared
        # across all tokens and must not be gathered with per-token frame indices.
        if timestep.shape[1] == 1:
            return timestep, embedded_timestep

        if frame_indices is None or _can_broadcast_timesteps(frame_indices, timestep.shape[1]):
            return timestep, embedded_timestep

        gather_index = frame_indices.unsqueeze(-1)
        timestep = timestep.gather(1, gather_index.expand(-1, -1, timestep.shape[-1]))
        embedded_timestep = embedded_timestep.gather(1, gather_index.expand(-1, -1, embedded_timestep.shape[-1]))
        return timestep, embedded_timestep

    def _resolve_hidden_dtype(self, latent: torch.Tensor) -> torch.dtype:
        latent_dtype = latent.dtype
        weight = getattr(self.patchify_proj, "weight", None)
        weight_dtype = getattr(weight, "dtype", None)
        return weight_dtype if weight_dtype is not None else latent_dtype

    def _get_prepared_conditioning(self, context: torch.Tensor | PreparedConditioning) -> PreparedConditioning | None:
        return context if isinstance(context, PreparedConditioning) else None

    def _get_runtime_timestep_bases(self, modality: Modality) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        batch_size = modality.latent.shape[0]
        runtime_cache = modality.runtime_cache
        base_timestep = None if runtime_cache is None else runtime_cache.base_timestep
        if base_timestep is None:
            base_timestep = self._build_base_timestep(
                modality.timesteps, modality.sigma, batch_size, frame_indices=modality.frame_indices
            )
            if runtime_cache is not None:
                runtime_cache.base_timestep = base_timestep
        prompt_base_timestep = None
        if self.prompt_adaln is not None:
            prompt_base_timestep = None if runtime_cache is None else runtime_cache.prompt_base_timestep
            if prompt_base_timestep is None:
                prompt_base_timestep = self._build_base_timestep(modality.sigma, modality.sigma, batch_size)
                if runtime_cache is not None:
                    runtime_cache.prompt_base_timestep = prompt_base_timestep
        return base_timestep, prompt_base_timestep

    def _prepare_context(
        self,
        context: torch.Tensor | PreparedConditioning,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare context for transformer blocks."""
        prepared = self._get_prepared_conditioning(context)
        context_tensor = prepared.projected_context if prepared is not None and prepared.projected_context is not None else (
            prepared.context if prepared is not None else context
        )
        batch_size = x.shape[0]
        if self.caption_projection is not None and prepared is None:
            context_tensor = self.caption_projection(context_tensor)
        context_tensor = context_tensor.view(batch_size, -1, x.shape[-1])

        return context_tensor, attention_mask

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_self_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        finfo = torch.finfo(x_dtype)
        eps = finfo.tiny
        bias = torch.full_like(attention_mask, finfo.min, dtype=x_dtype)
        positive = attention_mask > 0
        if positive.any():
            bias[positive] = torch.log(attention_mask[positive].clamp(min=eps)).to(x_dtype)
        return bias.unsqueeze(2)

    def _prepare_cross_attention_mask(self, cross_attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        if cross_attention_mask is None:
            return None
        mask = cross_attention_mask.to(x_dtype)
        if mask.ndim == 2:
            return (mask - 1).reshape((mask.shape[0], 1, 1, mask.shape[-1])) * torch.finfo(x_dtype).max
        if mask.ndim == 3:
            return (mask - 1).unsqueeze(2) * torch.finfo(x_dtype).max
        raise ValueError(f"Expected cross_attention_mask shape (B, K) or (B, Q, K), got {tuple(mask.shape)}")

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
        rope_axes: tuple[int, ...] | None = None,
        rope_max_pos: list[int] | None = None,
        runtime_cache=None,
    ) -> RopeCache:
        """Prepare positional embeddings."""
        rope_caches = None if runtime_cache is None else runtime_cache.rope_caches
        key = self._rope_cache_key(inner_dim, max_pos, use_middle_indices_grid, num_attention_heads, x_dtype, rope_axes, rope_max_pos)
        if rope_caches is not None:
            cached = rope_caches.get(key)
            if cached is not None:
                return cached
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        cache = build_rope_cache(
            positions=positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            rope_axes=rope_axes,
            rope_max_pos=rope_max_pos,
            freq_grid_generator=freq_grid_generator,
        )
        if rope_caches is not None:
            rope_caches[key] = cache
        return cache

    def build_prepared_conditioning(self, modality: Modality) -> PreparedConditioning:
        prepared = self._get_prepared_conditioning(modality.context)
        if prepared is not None:
            return prepared
        self._get_runtime_timestep_bases(modality)
        projected_context = self.caption_projection(modality.context) if self.caption_projection is not None else None
        return PreparedConditioning(context=modality.context, projected_context=projected_context)

    def _hidden_states(self, latent: torch.Tensor, keyframes_mask: torch.Tensor | None) -> torch.Tensor:
        return apply_keyframes_absolute_embedding(self.patchify_proj(latent), keyframes_mask, self.keyframes_embedding_provider)

    def prepare(
        self,
        modality: Modality,
    ) -> TransformerArgs:
        latent = modality.latent
        latent_dtype = self._resolve_hidden_dtype(latent)
        if latent.dtype != latent_dtype:
            latent = latent.to(latent_dtype)
        if os.environ.get("WAN2GP_GGUF_TRACE_LATENT", "") in ("1", "true", "yes") and not hasattr(
            self, "_gguf_latent_logged"
        ):
            self._gguf_latent_logged = True
            tqdm.write(f"[GGUF][patchify_proj] latent={latent_dtype} weight={getattr(getattr(self.patchify_proj, 'weight', None), 'dtype', None)}")
        # with CUDA graphs, built in the graph pool's idle memory: the hidden state then moves into the blocks' static buffer
        x = cuda_graphs.pooled(self.patchify_proj, self._hidden_states, latent, modality.keyframes_mask)
        base_timestep, prompt_base_timestep = self._get_runtime_timestep_bases(modality)
        prepared = self._get_prepared_conditioning(modality.context)
        timestep, embedded_timestep = self._prepare_timestep_from_base(
            base_timestep,
            modality.sigma,
            self.adaln,
            x.shape[0],
            latent_dtype,
            frame_indices=modality.frame_indices,
        )
        if timestep is None or embedded_timestep is None:
            timestep, embedded_timestep = self._prepare_timestep(
                modality.timesteps, self.adaln, x.shape[0], latent_dtype, frame_indices=modality.frame_indices
            )
        if modality.ref_adaln is not None:
            ref_adaln = modality.ref_adaln.to(device=timestep.device, dtype=timestep.dtype)
            if ref_adaln.ndim == 2:
                ref_adaln = ref_adaln.unsqueeze(1)
            timestep = timestep + ref_adaln
        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._prepare_timestep_from_base(
                prompt_base_timestep,
                modality.sigma,
                self.prompt_adaln,
                x.shape[0],
                latent_dtype,
                include_embedded=False,
            )
            if prompt_timestep is None:
                prompt_timestep, _ = self._prepare_timestep(modality.sigma, self.prompt_adaln, x.shape[0], latent_dtype)
        context, attention_mask = self._prepare_context(modality.context, x, modality.context_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, latent_dtype)
        self_attention_mask = self._prepare_self_attention_mask(modality.attention_mask, latent_dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=latent_dtype,
            runtime_cache=modality.runtime_cache,
        )
        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            cross_attention_mask=None,
            enabled=modality.enabled,
            nag=modality.nag,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self_attention_mask,
            ref_context=None if modality.ref_context is None else modality.ref_context.to(device=x.device, dtype=latent_dtype),
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection | None,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        prompt_adaln: AdaLayerNormSingle | None = None,
        keyframes_embedding_provider: KeyframesEmbeddingProvider | None = None,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            prompt_adaln=prompt_adaln,
            keyframes_embedding_provider=keyframes_embedding_provider,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def build_prepared_conditioning(self, modality: Modality) -> PreparedConditioning:
        return self.simple_preprocessor.build_prepared_conditioning(modality)

    def prepare(
        self,
        modality: Modality,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        base_timestep, _ = self.simple_preprocessor._get_runtime_timestep_bases(modality)
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.audio_cross_attention_dim,
            max_pos=self.simple_preprocessor.max_pos,
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
            rope_axes=(0,),
            rope_max_pos=[self.cross_pe_max_pos],
            runtime_cache=modality.runtime_cache,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            modality=modality,
            timestep=modality.timesteps,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=self.simple_preprocessor._resolve_hidden_dtype(modality.latent),
            frame_indices=modality.frame_indices,
            base_timestep=base_timestep,
        )
        cross_attention_mask = self.simple_preprocessor._prepare_cross_attention_mask(modality.cross_attention_mask, modality.latent.dtype)
        return TransformerArgs(
            x=transformer_args.x,
            context=transformer_args.context,
            context_mask=transformer_args.context_mask,
            timesteps=transformer_args.timesteps,
            embedded_timestep=transformer_args.embedded_timestep,
            positional_embeddings=transformer_args.positional_embeddings,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
            cross_attention_mask=cross_attention_mask,
            enabled=transformer_args.enabled,
            nag=transformer_args.nag,
            prompt_timestep=transformer_args.prompt_timestep,
            self_attention_mask=transformer_args.self_attention_mask,
            ref_context=transformer_args.ref_context,
        )

    def _prepare_cross_attention_timestep(
        self,
        modality: Modality,
        timestep: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
        frame_indices: torch.Tensor | None = None,
        base_timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        scale_shift_timestep, _ = self.simple_preprocessor._prepare_timestep_from_base(
            base_timestep,
            modality.sigma,
            self.cross_scale_shift_adaln,
            batch_size,
            hidden_dtype,
            frame_indices=frame_indices,
            scale_multiplier=timestep_scale_multiplier,
            include_embedded=False,
        )
        if scale_shift_timestep is None:
            scale_shift_timestep, _ = self.simple_preprocessor._prepare_timestep(
                timestep,
                self.cross_scale_shift_adaln,
                batch_size,
                hidden_dtype,
                frame_indices=frame_indices,
                scale_multiplier=timestep_scale_multiplier,
            )
        gate_noise_timestep, _ = self.simple_preprocessor._prepare_timestep_from_base(
            base_timestep,
            modality.sigma,
            self.cross_gate_adaln,
            batch_size,
            hidden_dtype,
            frame_indices=frame_indices,
            scale_multiplier=self.av_ca_timestep_scale_multiplier,
            include_embedded=False,
        )
        if gate_noise_timestep is None:
            gate_noise_timestep, _ = self.simple_preprocessor._prepare_timestep(
                timestep,
                self.cross_gate_adaln,
                batch_size,
                hidden_dtype,
                frame_indices=frame_indices,
                scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
        return scale_shift_timestep, gate_noise_timestep
