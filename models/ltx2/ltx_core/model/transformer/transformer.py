# Original Code by LightTricks (https://github.com/Lightricks/LTX-2)
# VRAM Optimizations by DeepBeepMeep (c) 2026. Please quote DeepBeepMeep / WanGP if reused

from dataclasses import dataclass, replace

import torch
from mmgp import offload

from ...guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from .adaln import adaln_embedding_coefficient
from .attention import Attention, AttentionCallable, AttentionFunction
from .feed_forward import FeedForward
from .rope import LTXRopeType
from .transformer_args import TransformerArgs
from ...utils import rms_norm
from shared.kernels import ops as kernels


SELF_ATTENTION_ADD_AFTER = True  # comparison switch: False returns the video self attention's output, added afterwards


def _reshape_hidden_states(hidden_states: torch.Tensor, frames: int) -> torch.Tensor:
    return hidden_states.reshape(hidden_states.shape[0], frames, -1, hidden_states.shape[-1])


def _restore_hidden_states_shape(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states.reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])


def _apply_scale_shift(hidden_states: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, in_place: bool = True) -> torch.Tensor:
    x_list, hidden_states = [hidden_states], None
    return kernels.scale_shift(x_list, scale, shift, in_place=in_place, frames=True)


class _WeightlessRMSNorm:
    """rms_norm(x, eps) as a norm for the shared segmented modulation (no weight)."""
    weight = None

    def __init__(self, eps: float) -> None:
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, eps=self.eps)


def _norm_scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    """rms_norm(x) * (1 + scale) + shift, x (batch, tokens, dim) borrowed: one modulation per batch row or frame through the shared
    segmented norm modulation (rows (batch * tokens, dim)); one per token, or a modulation of another dtype (the text cross-attention
    AdaLN's, multiplied in its own precision), through the shared scale/shift."""
    batch, tokens, dim = x.shape
    groups = scale.shape[1]
    if groups == tokens and groups > 1 or scale.dtype != x.dtype:
        return _apply_scale_shift(rms_norm(x, eps=eps), scale, shift)
    span, rows, shift, scale = tokens // groups, x.view(-1, dim), shift.reshape(-1, dim), scale.reshape(-1, dim)
    if kernels.rms_norm_modulate_supported(rows, shift, scale):  # one pass, rounded once (as precise as the two passes or more)
        return kernels.rms_norm_modulate(rows, eps, batch * tokens, 0, shift, scale, span).view(x.shape)
    segments = [(index * span, (index + 1) * span, index) for index in range(batch * groups)]
    return kernels.norm_modulate([rows], _WeightlessRMSNorm(eps), shift, scale, segments).view(x.shape)


def _norm_scale_shift_source(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float):
    """_norm_scale_shift(x, scale, shift, eps) as an mmgp RowSource made by ranges of rows of (batch * tokens, dim), for an attention that
    only projects it: never whole in 16 bits beside its prepared form. The tensor itself for a modulation per token or of another dtype."""
    batch, tokens, dim = x.shape
    groups = scale.shape[1]
    if groups == tokens and groups > 1 or scale.dtype != x.dtype:
        return _norm_scale_shift(x, scale, shift, eps)
    span, rows, norm = tokens // groups, x.view(-1, dim), _WeightlessRMSNorm(eps)
    shift, scale = shift.reshape(-1, dim), scale.reshape(-1, dim)

    fused = kernels.rms_norm_modulate_supported(rows, shift, scale)

    def fill(start: int, stop: int) -> torch.Tensor:
        if fused:
            return kernels.rms_norm_modulate(rows, eps, stop - start, start, shift, scale, span)
        segments = [(max(index * span, start) - start, min((index + 1) * span, stop) - start, index) for index in range(start // span, (stop - 1) // span + 1)]
        return kernels.norm_modulate([rows[start:stop]], norm, shift, scale, segments)

    def fill_at(first: int, step: int, count: int):  # fill inside a CUDA graph loop, rows from its counter (the fused kernel reads them)
        return (lambda counter: kernels.rms_norm_modulate(rows, eps, count, first, shift, scale, span, counter, step)) if fused else None
    return offload.RowSource(x.shape, x.dtype, x.device, fill, fill_at)


def _rms_norm_source(x: torch.Tensor, eps: float):
    """rms_norm(x, eps) as an mmgp RowSource made by ranges of rows of (batch * tokens, dim): an attention input only its projections read."""
    rows = x.view(-1, x.shape[-1])
    fused = kernels.rms_norm_modulate_supported(rows)

    def fill(start: int, stop: int) -> torch.Tensor:
        return kernels.rms_norm_modulate(rows, eps, stop - start, start) if fused else rms_norm(rows[start:stop], eps=eps)

    def fill_at(first: int, step: int, count: int):  # inside a CUDA graph loop, rows from its counter (the fused kernel reads them)
        return (lambda counter: kernels.rms_norm_modulate(rows, eps, count, first, counter=counter, step=step)) if fused else None
    return offload.RowSource(x.shape, x.dtype, x.device, fill, fill_at)


def _apply_gate(hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if gate.shape[1] == hidden_states.shape[1]:
        hidden_states.mul_(gate)
        return hidden_states
    hidden_states = _reshape_hidden_states(hidden_states, gate.shape[1])
    hidden_states.mul_(gate.unsqueeze(2))
    return _restore_hidden_states_shape(hidden_states)


def _gated_add_(residual: torch.Tensor, x_list: list, gate: torch.Tensor) -> None:
    """residual += x * gate, x handed off in x_list, with _apply_gate's rounding: a gate per token, per frame or for all tokens."""
    if gate.shape[1] not in (1, residual.shape[1]):
        x_list[0], residual, gate = _reshape_hidden_states(x_list[0], gate.shape[1]), residual.view(residual.shape[0], gate.shape[1], -1, residual.shape[-1]), gate.unsqueeze(2)
    kernels.gated_residual_(residual, x_list, gate)


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False
    ff_bias: bool = True


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim, bias=video.ff_bias)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(adaln_embedding_coefficient(video.cross_attention_adaln), video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim, bias=audio.ff_bias)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(adaln_embedding_coefficient(audio.cross_attention_adaln), audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )
        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None)
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    @staticmethod
    def _gated_ffn(ffn: FeedForward, x_list: list, gate: torch.Tensor, residual: torch.Tensor) -> None:
        """residual += gate * ffn(x), x handed off in x_list, through the shared chunked FFN; a gate per token is applied after it."""
        fc1, fc2 = ffn.net[0].proj, ffn.net[2]
        if gate.shape[1] == residual.shape[1]:
            residual.add_(kernels.mlp(x_list, fc1, fc2, "gelu").mul_(gate))
        else:
            kernels.mlp(x_list, fc1, fc2, "gelu", residual=residual, segments=kernels.gate_segments(gate, residual.shape[0], residual.shape[1]))

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        nag: dict | None = None,
        cross_attention_adaln: bool = False,
    ) -> None:
        """Adds the text cross-attention to x."""
        if not cross_attention_adaln:
            out = attn([_rms_norm_source(x, self.norm_eps)], context_list=[context], mask=context_mask, NAG=nag, residual=x)
            if out is not None:
                x.add_(out)
            return
        q_shift, q_scale, q_gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
        apply_cross_attention_adaln(
            x,
            context,
            attn,
            q_shift,
            q_scale,
            q_gate,
            prompt_scale_shift_table,
            prompt_timestep,
            context_mask,
            nag=nag,
            norm_eps=self.norm_eps,
        )

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        source = video if video is not None else audio
        if source is None:
            raise ValueError("BasicAVTransformerBlock requires at least one modality.")
        batch_size = source.x.shape[0]
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and audio.enabled and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and video.enabled and vx.numel() > 0)
        run_a2v = run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx)
        run_v2a = run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            vshift_msa = vshift_msa.to(vx.dtype)
            vscale_msa = vscale_msa.to(vx.dtype)
            vgate_msa = vgate_msa.to(vx.dtype)
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                norm_vx = _norm_scale_shift_source(vx, vscale_msa, vshift_msa, self.norm_eps)
                x_list = [norm_vx]
                del norm_vx
                masked = perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)  # 0 / 1 per batch row, all ones otherwise
                # unmasked: the attention adds its gated output into vx by spans of rows (no output of all tokens)
                into = SELF_ATTENTION_ADD_AFTER and not masked
                attn_out = self.attn1(x_list, mask=video.self_attention_mask, pe=video.positional_embeddings, residual=vx if into else None, gate=vgate_msa)
                if attn_out is not None:
                    if masked:
                        attn_out.mul_(perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx))
                    attn_list, attn_out = [attn_out], None
                    _gated_add_(vx, attn_list, vgate_msa)
            self._apply_text_cross_attention(
                vx,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                nag=video.nag,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            ref_context = getattr(video, "ref_context", None)
            if ref_context is not None and hasattr(self, "ref_attn"):
                ref_start_block = int(getattr(self, "editanything_ref_start_block", 12))
                ref_end_block = int(getattr(self, "editanything_ref_end_block", 35))
                if ref_start_block <= self.idx <= ref_end_block:
                    ref_context_scale = float(getattr(self, "editanything_ref_context_scale", 0.01))
                    if ref_context_scale != 0:
                        attn_out = self.ref_attn([rms_norm(vx, eps=self.norm_eps)], context_list=[ref_context])
                        vx.add_(attn_out, alpha=ref_context_scale)
                        attn_out = None
            del vshift_msa, vscale_msa, vgate_msa

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            ashift_msa = ashift_msa.to(ax.dtype)
            ascale_msa = ascale_msa.to(ax.dtype)
            agate_msa = agate_msa.to(ax.dtype)

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = _norm_scale_shift(ax, ascale_msa, ashift_msa, self.norm_eps)
                x_list = [norm_ax]
                del norm_ax
                attn_out = self.audio_attn1(x_list, mask=audio.self_attention_mask, pe=audio.positional_embeddings)
                if perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                    attn_out.mul_(perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax))
                attn_list, attn_out = [attn_out], None
                _gated_add_(ax, attn_list, agate_msa)
            self._apply_text_cross_attention(
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                nag=audio.nag if audio.nag is not None and audio.nag.get("enable_audio_text_nag", False) else None,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            del ashift_msa, ascale_msa, agate_msa

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            # Both directions consume the original states. Compute the small audio result first, but add it only after a2v.
            # Recomputing the video norm avoids retaining a full normalized video alongside its modulated copy.
            pending_v2a = None

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )
            scale_ca_audio_hidden_states_a2v = scale_ca_audio_hidden_states_a2v.to(ax.dtype)
            shift_ca_audio_hidden_states_a2v = shift_ca_audio_hidden_states_a2v.to(ax.dtype)
            scale_ca_audio_hidden_states_v2a = scale_ca_audio_hidden_states_v2a.to(ax.dtype)
            shift_ca_audio_hidden_states_v2a = shift_ca_audio_hidden_states_v2a.to(ax.dtype)
            gate_out_v2a = gate_out_v2a.to(ax.dtype)

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )
            scale_ca_video_hidden_states_a2v = scale_ca_video_hidden_states_a2v.to(vx.dtype)
            shift_ca_video_hidden_states_a2v = shift_ca_video_hidden_states_a2v.to(vx.dtype)
            scale_ca_video_hidden_states_v2a = scale_ca_video_hidden_states_v2a.to(vx.dtype)
            shift_ca_video_hidden_states_v2a = shift_ca_video_hidden_states_v2a.to(vx.dtype)
            gate_out_a2v = gate_out_a2v.to(vx.dtype)

            if run_v2a:
                ax_scaled = _norm_scale_shift(ax, scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a, self.norm_eps)
                vx_scaled = _norm_scale_shift_source(vx, scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a, self.norm_eps)
                x_list, context_list = [ax_scaled], [vx_scaled]
                del ax_scaled, vx_scaled
                attn_out = self.video_to_audio_attn(
                    x_list,
                    context_list=context_list,
                    mask=audio.cross_attention_mask,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                )
                attn_out = _apply_gate(attn_out, gate_out_v2a)
                if perturbations.any_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
                    attn_out.mul_(perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax))
                pending_v2a = attn_out
                attn_out = ax_scaled = vx_scaled = None

            if run_a2v:
                vx_scaled = _norm_scale_shift_source(vx, scale_ca_video_hidden_states_a2v, shift_ca_video_hidden_states_a2v, self.norm_eps)
                ax_scaled = _norm_scale_shift(ax, scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v, self.norm_eps)
                x_list, context_list  = [vx_scaled], [ax_scaled]
                del vx_scaled, ax_scaled
                masked = perturbations.any_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx)
                attn_out = self.audio_to_video_attn(  # vx itself is read no more: its scaled copies are the inputs
                    x_list,
                    context_list=context_list,
                    mask=video.cross_attention_mask,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                    residual=None if masked else vx,
                    gate=gate_out_a2v,
                )
                if attn_out is not None:  # not added by the attention itself
                    if masked:
                        attn_out.mul_(perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx))
                    attn_list, attn_out = [attn_out], None
                    _gated_add_(vx, attn_list, gate_out_a2v)
                vx_scaled = ax_scaled = None

            if pending_v2a is not None:
                ax.add_(pending_v2a)
                pending_v2a = None

            del gate_out_a2v, gate_out_v2a
            del (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
            )

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vshift_mlp = vshift_mlp.to(vx.dtype)
            vscale_mlp = vscale_mlp.to(vx.dtype)
            vgate_mlp = vgate_mlp.to(vx.dtype)
            vx_scaled = _norm_scale_shift(vx, vscale_mlp, vshift_mlp, self.norm_eps)
            vx_list, vx_scaled = [vx_scaled], None
            self._gated_ffn(self.ff, vx_list, vgate_mlp, vx)

            del vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ashift_mlp = ashift_mlp.to(ax.dtype)
            ascale_mlp = ascale_mlp.to(ax.dtype)
            agate_mlp = agate_mlp.to(ax.dtype)
            ax_scaled = _norm_scale_shift(ax, ascale_mlp, ashift_mlp, self.norm_eps)
            ax_list, ax_scaled = [ax_scaled], None
            self._gated_ffn(self.audio_ff, ax_list, agate_mlp, ax)

            del ashift_mlp, ascale_mlp, agate_mlp

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor | None,
    prompt_timestep: torch.Tensor | None,
    context_mask: torch.Tensor | None = None,
    nag: dict | None = None,
    norm_eps: float = 1e-6,
) -> None:
    """Adds the gated text cross-attention to x."""
    if prompt_scale_shift_table is not None and prompt_timestep is not None:
        batch_size = x.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        # Context is reused across blocks in LTX 2.3 prompt AdaLN, so this call must stay out-of-place.
        context = _apply_scale_shift(context, scale_kv, shift_kv, in_place=False)
    # handed off: the attention releases its normalized input after the q projection
    gate = q_gate.squeeze(2)
    out = attn([_norm_scale_shift_source(x, q_scale.squeeze(2), q_shift.squeeze(2), norm_eps)], context_list=[context], mask=context_mask, NAG=nag, residual=x, gate=gate)
    if out is not None:  # not added by the attention itself
        _gated_add_(x, [out], gate)
