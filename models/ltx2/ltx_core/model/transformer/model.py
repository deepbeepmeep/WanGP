from collections.abc import Callable
from enum import Enum

import torch
from mmgp import cuda_graphs

from ...guidance.perturbations import BatchedPerturbationConfig
from .adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from .attention import AttentionCallable, AttentionFunction
from .modality import Modality
from .rope import LTXRopeType
from .sol_attention import first_modality, sol_attention
from .transformer import BasicAVTransformerBlock, TransformerConfig
from shared.kernels import ops as kernels
from .transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ...utils import to_denoised


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
        use_prompt_adaln_single: bool = True,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
        use_keyframes_abs_pos_embedding: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_prompt_adaln_single = use_prompt_adaln_single
        self.use_keyframes_abs_pos_embedding = use_keyframes_abs_pos_embedding
        self.interrupt_check: Callable[[], bool] | None = None
        self.interrupted = False
        self.interrupt_check: Callable[[], bool] | None = None
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
            ff_bias=ff_bias,
            audio_ff_bias=audio_ff_bias,
        )

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        self.keyframes_abs_pos_embedding = (
            torch.nn.Parameter(torch.zeros(1, self.inner_dim)) if self.use_keyframes_abs_pos_embedding else None
        )
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=adaln_embedding_coefficient(self.cross_attention_adaln),
        )

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2)
            if self.cross_attention_adaln and self.use_prompt_adaln_single
            else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _keyframes_embedding(self) -> torch.Tensor | None:
        return getattr(self, "keyframes_abs_pos_embedding", None)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=adaln_embedding_coefficient(self.cross_attention_adaln),
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2)
            if self.cross_attention_adaln and self.use_prompt_adaln_single
            else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=getattr(self, "caption_projection", None),
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                keyframes_embedding_provider=self._keyframes_embedding,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=getattr(self, "audio_caption_projection", None),
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=getattr(self, "caption_projection", None),
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                keyframes_embedding_provider=self._keyframes_embedding,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=getattr(self, "audio_caption_projection", None),
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _refresh_preprocessor_refs(self) -> None:
        if self.model_type.is_video_enabled():
            if isinstance(self.video_args_preprocessor, MultiModalTransformerArgsPreprocessor):
                self.video_args_preprocessor.simple_preprocessor.patchify_proj = self.patchify_proj
            else:
                self.video_args_preprocessor.patchify_proj = self.patchify_proj
        if self.model_type.is_audio_enabled():
            if isinstance(self.audio_args_preprocessor, MultiModalTransformerArgsPreprocessor):
                self.audio_args_preprocessor.simple_preprocessor.patchify_proj = self.audio_patchify_proj
            else:
                self.audio_args_preprocessor.patchify_proj = self.audio_patchify_proj

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
                ff_bias=ff_bias,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
                ff_bias=audio_ff_bias,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        with cuda_graphs.scope():
            for block in self.transformer_blocks:
                if self.interrupt_check is not None and self.interrupt_check():
                    self.interrupted = True
                    return None, None
                if self._enable_gradient_checkpointing and self.training:
                    # Use gradient checkpointing to save memory during training.
                    # With use_reentrant=False, we can pass dataclasses directly -
                    # PyTorch will track all tensor leaves in the computation graph.
                    video, audio = torch.utils.checkpoint.checkpoint(
                        block,
                        video,
                        audio,
                        perturbations,
                        use_reentrant=False,
                    )
                else:
                    video, audio = block(
                        video=video,
                        audio=audio,
                        perturbations=perturbations,
                    )

        return video, audio

    def _process_transformer_blocks_joint(
        self,
        video_list: list[TransformerArgs | None],
        audio_list: list[TransformerArgs | None],
        perturbations: BatchedPerturbationConfig | list[BatchedPerturbationConfig] | None,
    ) -> tuple[list[TransformerArgs | None], list[TransformerArgs | None]]:
        """Process transformer blocks for joint-pass CFG."""
        perturbations_list = perturbations if isinstance(perturbations, list) else None
        if perturbations_list is not None and len(perturbations_list) != len(video_list):
            raise ValueError("Perturbations list must match number of joint-pass conditions.")
        with cuda_graphs.scope():
            for block in self.transformer_blocks:
                if self.interrupt_check is not None and self.interrupt_check():
                    self.interrupted = True
                    return [None] * len(video_list), [None] * len(audio_list)

                for idx in range(len(video_list)):
                    condition_perturbations = (
                        perturbations_list[idx] if perturbations_list is not None else perturbations
                    )
                    video_list[idx], audio_list[idx] = block(
                        video=video_list[idx],
                        audio=audio_list[idx],
                        perturbations=condition_perturbations,
                    )

        return video_list, audio_list

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x_list, x = [x], None
        x = kernels.layer_norm_modulate(x_list, shift, scale, norm_out.eps, frames=True)
        x = proj_out(x)
        return x

    def forward(
        self,
        video: Modality | list[Modality] | None,
        audio: Modality | list[Modality] | None,
        perturbations: BatchedPerturbationConfig | list[BatchedPerturbationConfig] | None,
    ) -> tuple[torch.Tensor | None | list[torch.Tensor | None], torch.Tensor | None | list[torch.Tensor | None]]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        with cuda_graphs.scope():  # with CUDA graphs, the preparation and the head run in the graph pool's idle memory (cuda_graphs.pooled)
            return self._forward(video, audio, perturbations)

    def _forward(self, video, audio, perturbations):
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        self._refresh_preprocessor_refs()
        self.interrupted = False
        # Refresh the Sol-Attn policy (enabled state, per-step tau, one-time runtime validation).
        sol_source = first_modality(video) or first_modality(audio)
        if sol_source is not None:
            sol_attention.begin_forward(sol_source.latent.device, sol_source.latent.dtype, video, audio)
        joint_pass = isinstance(video, (list, tuple)) or isinstance(audio, (list, tuple))
        if joint_pass:
            video_list = list(video) if isinstance(video, (list, tuple)) else [None] * len(audio)
            audio_list = list(audio) if isinstance(audio, (list, tuple)) else [None] * len(video)
            if len(video_list) != len(audio_list):
                raise ValueError("Joint-pass inputs must have the same number of video and audio entries.")
            if isinstance(perturbations, list) and len(perturbations) != len(video_list):
                raise ValueError("Perturbations list must match number of joint-pass conditions.")

            if self.model_type.is_video_enabled():
                video_args_list = [
                    self.video_args_preprocessor.prepare(v) if v is not None else None for v in video_list
                ]
            else:
                video_args_list = [None] * len(video_list)
            if self.model_type.is_audio_enabled():
                audio_args_list = [
                    self.audio_args_preprocessor.prepare(a) if a is not None else None for a in audio_list
                ]
            else:
                audio_args_list = [None] * len(audio_list)
            video_out_list, audio_out_list = self._process_transformer_blocks_joint(
                video_list=video_args_list,
                audio_list=audio_args_list,
                perturbations=perturbations,
            )
            if self.interrupted:
                return [None] * len(video_out_list), [None] * len(audio_out_list)

            vx_list = [
                cuda_graphs.pooled(
                    self, self._process_output, self.scale_shift_table, self.norm_out, self.proj_out, v_out.x, v_out.embedded_timestep
                )
                if v_out is not None
                else None
                for v_out in video_out_list
            ]
            ax_list = [
                cuda_graphs.pooled(
                    self,
                    self._process_output,
                    self.audio_scale_shift_table,
                    self.audio_norm_out,
                    self.audio_proj_out,
                    a_out.x,
                    a_out.embedded_timestep,
                )
                if a_out is not None
                else None
                for a_out in audio_out_list
            ]
            return vx_list, ax_list

        if isinstance(perturbations, list):
            if len(perturbations) != 1:
                raise ValueError("Non-joint pass expects at most one perturbation config.")
            perturbations = perturbations[0]

        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )
        if self.interrupted or (video_out is None and audio_out is None):
            return None, None

        vx = (
            cuda_graphs.pooled(
                self, self._process_output, self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            cuda_graphs.pooled(
                self,
                self._process_output,
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | list[Modality] | None,
        audio: Modality | list[Modality] | None,
        perturbations: BatchedPerturbationConfig | list[BatchedPerturbationConfig] | None,
        sigma: float,
    ) -> tuple[torch.Tensor | None | list[torch.Tensor | None], torch.Tensor | None | list[torch.Tensor | None]]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        if vx is None and ax is None:
            return None, None
        if isinstance(vx, list) or isinstance(ax, list):
            video_list = video if isinstance(video, (list, tuple)) else [None] * len(vx)
            audio_list = audio if isinstance(audio, (list, tuple)) else [None] * len(ax)
            denoised_video = [
                to_denoised(v.latent, v_pred, sigma) if v is not None and v_pred is not None else None
                for v, v_pred in zip(video_list, vx)
            ]
            denoised_audio = [
                to_denoised(a.latent, a_pred, sigma) if a is not None and a_pred is not None else None
                for a, a_pred in zip(audio_list, ax)
            ]
            return denoised_video, denoised_audio

        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | list[Modality] | None,
        audio: Modality | list[Modality] | None,
        perturbations: BatchedPerturbationConfig | list[BatchedPerturbationConfig] | None,
    ) -> tuple[torch.Tensor | None | list[torch.Tensor | None], torch.Tensor | None | list[torch.Tensor | None]]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        if vx is None and ax is None:
            return None, None
        if isinstance(vx, list) or isinstance(ax, list):
            video_list = video if isinstance(video, (list, tuple)) else [None] * len(vx)
            audio_list = audio if isinstance(audio, (list, tuple)) else [None] * len(ax)
            denoised_video = []
            denoised_audio = []
            for v, v_pred in zip(video_list, vx):
                if v is None or v_pred is None:
                    denoised_video.append(None)
                    continue
                v_timesteps = v.timesteps
                if v.frame_indices is not None:
                    v_timesteps = v_timesteps.gather(1, v.frame_indices)
                if v_timesteps is not None and v_timesteps.ndim == 2:
                    v_timesteps = v_timesteps.unsqueeze(-1)
                denoised_video.append(to_denoised(v.latent, v_pred, v_timesteps))
            for a, a_pred in zip(audio_list, ax):
                if a is None or a_pred is None:
                    denoised_audio.append(None)
                    continue
                a_timesteps = a.timesteps
                if a_timesteps is not None and a_timesteps.ndim == 2:
                    a_timesteps = a_timesteps.unsqueeze(-1)
                denoised_audio.append(to_denoised(a.latent, a_pred, a_timesteps))
            return denoised_video, denoised_audio

        if video is not None and video.frame_indices is not None:
            video_timesteps = video.timesteps.gather(1, video.frame_indices)
        else:
            video_timesteps = video.timesteps if video is not None else None
        if video_timesteps is not None and video_timesteps.ndim == 2:
            video_timesteps = video_timesteps.unsqueeze(-1)
        audio_timesteps = audio.timesteps if audio is not None else None
        if audio_timesteps is not None and audio_timesteps.ndim == 2:
            audio_timesteps = audio_timesteps.unsqueeze(-1)

        denoised_video = to_denoised(video.latent, vx, video_timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio_timesteps) if ax is not None else None
        return denoised_video, denoised_audio
