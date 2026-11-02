import logging
from collections.abc import Callable, Iterator

import torch

from ..ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep, EulerDiffusionStep, Res2sDiffusionStep
from ..ltx_core.components.guiders import CFGGuider, CFGStarRescalingGuider, LtxAPGGuider, MultiModalGuider, MultiModalGuiderParams
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.components.schedulers import LTX2Scheduler
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.upsampler import upsample_video
from ..ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ..ltx_core.model.video_vae import decode_video_to_tensor as vae_decode_video_to_tensor
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import LatentState, VideoLatentShape, VideoPixelShape
from shared.prompt_relay import encode_prompt_relay
from .utils import ModelLedger
from .utils.args import default_2_stage_arg_parser
from .utils.constants import (
    AUDIO_SAMPLE_RATE,
    DISTILLED_SIGMA_VALUES,
    DISTILLED_8_STEPS_STAGE_2_SIGMA_VALUES,
    LTX23_USE_DISTILLED_8_STEPS_STAGE_2_SIGMAS,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from .utils.spatial_tiling import spatially_tiled_denoising_func
from .utils.helpers import (
    PERTURBATION_SKIP_SELF_ATTENTION,
    assert_resolution,
    bind_interrupt_check,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    latent_conditionings_by_latent_sequence,
    prepare_mask_injection,
    multi_modal_guider_denoising_func,
    res2s_audio_video_denoising_loop,
    simple_denoising_func,
    video_conditionings_by_frozen_video,
    video_conditionings_by_control_video,
)
from .utils.types import PipelineComponents
from ..editanything import build_editanything_reference_conditioning
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.self_refiner import create_self_refiner_handler, normalize_self_refiner_plan
from shared.utils.text_encoder_cache import TextEncoderCache

device = get_device()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        distilled_lora: list[LoraPathStrengthAndSDOps] | None = None,
        spatial_upsampler_path: str | None = None,
        gemma_root: str | None = None,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: str = device,
        fp8transformer: bool = False,
        model_device: torch.device | None = None,
        stage_1_models: object | None = None,
        stage_2_models: object | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.stage_1_models = stage_1_models
        self.stage_2_models = stage_2_models or stage_1_models
        if self.stage_1_models is None:
            if checkpoint_path is None or gemma_root is None or spatial_upsampler_path is None:
                raise ValueError("checkpoint_path, gemma_root, and spatial_upsampler_path are required.")
            ledger_device = model_device or device
            distilled_lora = distilled_lora or []
            self.stage_1_model_ledger = ModelLedger(
                dtype=self.dtype,
                device=ledger_device,
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
                spatial_upsampler_path=spatial_upsampler_path,
                loras=loras or [],
                fp8transformer=fp8transformer,
            )

            if distilled_lora:
                self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
                    loras=distilled_lora,
                )
            else:
                self.stage_2_model_ledger = self.stage_1_model_ledger
        else:
            self.stage_1_model_ledger = None
            self.stage_2_model_ledger = None

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.text_encoder_cache = TextEncoderCache()

    def _get_stage_model(self, stage: int, name: str):
        if stage == 1:
            models = self.stage_1_models
            ledger = self.stage_1_model_ledger
        else:
            models = self.stage_2_models
            ledger = self.stage_2_model_ledger
        if models is not None:
            return getattr(models, name)
        if ledger is None:
            raise ValueError(f"Missing model source for stage {stage} '{name}'.")
        return getattr(ledger, name)()

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        prompt_relay_frame_offset: int = 0,
        prompt_relay_epsilon: float = 1e-3,
        audio_cfg_guidance_scale: float | None = None,
        cfg_star_switch: int = 0,
        apg_switch: int = 0,
        perturbation_switch: int = 0,
        perturbation_layers: list[int] | None = None,
        perturbation_start: float = 0.0,
        perturbation_end: float = 1.0,
        alt_guidance_scale: float = 1.0,
        alt_scale: float = 0.0,
        sample_solver: str = "euler",
        custom_sigmas=None,
        guiding_images: list[tuple[str, int, float]] | None = None,
        guiding_images_stage2: list[tuple] | None = None,
        images_stage2: list[tuple[str, int, float]] | None = None,
        video_conditioning: list[tuple[str, float]] | None = None,
        video_conditioning_downscale_factor: int = 1,
        video_conditioning_stage2: list[tuple[str, float]] | None = None,
        latent_conditioning_stage2: torch.Tensor | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        audio_conditionings: list | None = None,
        audio_conditionings_stage2: list | None = None,
        audio_identity_guidance_scale: float = 0.0,
        callback: Callable[..., None] | None = None,
        set_progress_status: Callable[[str], None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
        loras_slists: dict | None = None,
        text_connectors: dict | None = None,
        masking_source: dict | None = None,
        masking_strength: float | None = None,
        return_latent_slice: slice | None = None,
        continuous_conditioning_and_guide: bool = False,
        skip_stage_2: bool = False,
        tiled_stage_2: bool = False,
        frozen_video_conditioning: torch.Tensor | None = None,
        frozen_output_video: torch.Tensor | None = None,
        self_refiner_setting: int = 0,
        self_refiner_plan: str = "",
        self_refiner_f_uncertainty: float = 0.1,
        self_refiner_certain_percentage: float = 0.999,
        self_refiner_max_plans: int = 1,
        editanything_ref_images=None,
        ltx2_22B_class: bool = False,
        hdr_transform: str | None = None,
        skip_audio: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=not skip_stage_2)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mask_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        noiser = GaussianNoiser(generator=generator)
        sample_solver = (sample_solver or "euler").lower()
        use_hq_sampler = sample_solver == "res2s"
        use_ancestral_sampler = sample_solver == "distilled_8_steps_ancestral"
        use_distilled_8_steps = sample_solver in {"distilled_8_steps", "distilled_8_steps_ancestral"}
        if sample_solver not in {"euler", "res2s", "distilled_8_steps", "distilled_8_steps_ancestral"}:
            raise ValueError(f"Unsupported LTX2 sampler '{sample_solver}'.")
        skip_stage_2 = bool(skip_stage_2)
        stage_1_pass_no = 0 if skip_stage_2 else 1
        stage_1_stepper = Res2sDiffusionStep() if use_hq_sampler else EulerAncestralDiffusionStep() if use_ancestral_sampler else EulerDiffusionStep()
        stage_2_stepper = Res2sDiffusionStep() if use_hq_sampler else EulerDiffusionStep()
        ancestral_noise_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 10000) if use_ancestral_sampler else None
        self_refiner_handler = None
        self_refiner_handler_audio = None
        self_refiner_handler_stage2 = None
        self_refiner_handler_audio_stage2 = None
        if self_refiner_setting and self_refiner_setting > 0 and not use_hq_sampler:
            plans, _ = normalize_self_refiner_plan(self_refiner_plan or "", max_plans=self_refiner_max_plans)
            plan_stage1 = plans[0] if plans else []
            plan_stage2 = plans[1] if len(plans) > 1 else []
            self_refiner_handler = create_self_refiner_handler(
                plan_stage1,
                self_refiner_f_uncertainty,
                self_refiner_setting,
                self_refiner_certain_percentage,
                channel_dim=-1,
            )
            self_refiner_handler_audio = create_self_refiner_handler(
                plan_stage1,
                self_refiner_f_uncertainty,
                self_refiner_setting,
                self_refiner_certain_percentage,
                channel_dim=-1,
            )
            if plan_stage2:
                self_refiner_handler_stage2 = create_self_refiner_handler(
                    plan_stage2,
                    self_refiner_f_uncertainty,
                    self_refiner_setting,
                    self_refiner_certain_percentage,
                    channel_dim=-1,
                )
                self_refiner_handler_audio_stage2 = create_self_refiner_handler(
                    plan_stage2,
                    self_refiner_f_uncertainty,
                    self_refiner_setting,
                    self_refiner_certain_percentage,
                    channel_dim=-1,
                )
        audio_cfg_guidance_scale = cfg_guidance_scale if audio_cfg_guidance_scale is None else audio_cfg_guidance_scale
        guider_cls = CFGGuider
        if apg_switch:
            guider_cls = LtxAPGGuider
        elif cfg_star_switch:
            guider_cls = CFGStarRescalingGuider
        video_cfg_guider = guider_cls(cfg_guidance_scale)
        audio_cfg_guider = guider_cls(audio_cfg_guidance_scale)
        dtype = torch.bfloat16

        text_encoder = self._get_stage_model(1, "text_encoder")
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        # Codex: needs to return only the text embeddings from the text encoder for all the prompts
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(
            text_encoder, text_connectors
        )
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        encode_fn_with_masks = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
            return_attention_masks=True,
        )
        relay_conditioning = encode_prompt_relay(prompt, encode_fn_with_masks, self.text_encoder_cache, self.device, num_frames, frame_rate, text_encoder.tokenizer, visible_frame_offset=prompt_relay_frame_offset, epsilon=prompt_relay_epsilon)
        if relay_conditioning is None:
            contexts = self.text_encoder_cache.encode(
                encode_fn,
                [prompt, negative_prompt],
                device=self.device,
                parallel=True,
            )
            context_p, context_n = contexts
            v_context_p, a_context_p = context_p
            v_context_p_mask_builder = None
            a_context_p_mask_builder = None
        else:
            v_context_p = relay_conditioning.video_context
            a_context_p = relay_conditioning.audio_context
            context_n = self.text_encoder_cache.encode(encode_fn, [negative_prompt], device=self.device, parallel=True)[0]
            v_context_p_mask_builder = relay_conditioning.video_mask_builder
            a_context_p_mask_builder = relay_conditioning.audio_mask_builder

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        # Codex: now that the text encoder has been released, compute the text_embedding_projection,
        # audio_embeddings_connector, video_embeddings_connector in order to get v_context_p, a_context_p
        # and v_context_n, a_context_n
        v_context_n, a_context_n = context_n

        # Stage 1: Initial low resolution video generation.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width if skip_stage_2 else width // 2,
            height=height if skip_stage_2 else height // 2,
            fps=frame_rate,
        )
        if interrupt_check is not None and interrupt_check():
            return None, None
        video_encoder = self._get_stage_model(1, "video_encoder")
        transformer = self._get_stage_model(1, "transformer")
        bind_interrupt_check(transformer, interrupt_check)
        stage_1_ref_conditionings, stage_1_ref_context, stage_1_ref_adaln = build_editanything_reference_conditioning(
            transformer,
            editanything_ref_images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
        )
        if custom_sigmas and custom_sigmas[0] is not None:
            sigmas = torch.tensor([*custom_sigmas[0], 0.0], dtype=torch.float32, device=self.device)
        elif use_distilled_8_steps:
            sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(dtype=torch.float32, device=self.device)
        elif use_hq_sampler:
            empty_latent = torch.empty(
                VideoLatentShape.from_pixel_shape(
                    stage_1_output_shape,
                    latent_channels=self.pipeline_components.video_latent_channels,
                    scale_factors=self.pipeline_components.video_scale_factors,
                ).to_torch_shape(),
                device="meta",
            )
            sigmas = LTX2Scheduler().execute(latent=empty_latent, steps=num_inference_steps, device=self.device)
        else:
            sigmas = LTX2Scheduler().execute(steps=num_inference_steps, device=self.device)
        if loras_slists is not None:
            stage_1_steps = len(sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_1_steps,
                phase_switch_step=stage_1_steps,
                phase_switch_step2=stage_1_steps,
            )

        if set_progress_status is not None:
            set_progress_status("VAE Encoding")

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            preview_tools: VideoLatentTools | None = None,
            mask_context=None,
        ) -> tuple[LatentState, LatentState]:
            if use_hq_sampler:
                hq_stg_blocks = list(perturbation_layers or [])
                hq_stg_scale = 1.0 if perturbation_switch == PERTURBATION_SKIP_SELF_ATTENTION else 0.0
                return res2s_audio_video_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=multi_modal_guider_denoising_func(
                        video_guider=MultiModalGuider(
                            params=MultiModalGuiderParams(
                                cfg_scale=cfg_guidance_scale,
                                stg_scale=hq_stg_scale,
                                stg_blocks=hq_stg_blocks,
                                rescale_scale=alt_scale,
                                modality_scale=alt_guidance_scale,
                            ),
                            negative_context=v_context_n,
                        ),
                        audio_guider=MultiModalGuider(
                            params=MultiModalGuiderParams(
                                cfg_scale=audio_cfg_guidance_scale,
                                stg_scale=hq_stg_scale,
                                stg_blocks=hq_stg_blocks,
                                rescale_scale=1.0,
                                modality_scale=alt_guidance_scale,
                            ),
                            negative_context=a_context_n,
                        ),
                        v_context_p=v_context_p,
                        a_context_p=a_context_p,
                        transformer=transformer,  # noqa: F821
                        audio_identity_guidance_scale=audio_identity_guidance_scale,
                        ref_context=stage_1_ref_context,
                        ref_adaln=stage_1_ref_adaln,
                        v_context_p_mask_builder=v_context_p_mask_builder,
                        a_context_p_mask_builder=a_context_p_mask_builder,
                        skip_audio_to_video=frozen_video_conditioning is not None,
                    ),
                    noise_seed=seed,
                    mask_context=mask_context,
                    interrupt_check=interrupt_check,
                    callback=callback,
                    preview_tools=preview_tools,
                    pass_no=stage_1_pass_no,
                    transformer=transformer,
                )
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=guider_denoising_func(
                    video_cfg_guider,
                    audio_cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=transformer,  # noqa: F821
                    alt_guidance_scale=alt_guidance_scale,
                    alt_scale=alt_scale,
                    perturbation_switch=perturbation_switch,
                    perturbation_layers=perturbation_layers,
                    perturbation_start=perturbation_start,
                    perturbation_end=perturbation_end,
                    audio_identity_guidance_scale=audio_identity_guidance_scale,
                    ref_context=stage_1_ref_context,
                    ref_adaln=stage_1_ref_adaln,
                    v_context_p_mask_builder=v_context_p_mask_builder,
                    a_context_p_mask_builder=a_context_p_mask_builder,
                    skip_audio_to_video=frozen_video_conditioning is not None,
                ),
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=stage_1_pass_no,
                transformer=transformer,
                self_refiner_handler=self_refiner_handler,
                self_refiner_handler_audio=self_refiner_handler_audio,
                self_refiner_generator=generator,
                ancestral_noise_generator=ancestral_noise_generator,
            )
        if interrupt_check is not None and interrupt_check():
            return None, None
        if frozen_video_conditioning is not None:
            stage_1_conditionings = video_conditionings_by_frozen_video(
                video=frozen_video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        else:
            stage_1_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        stage_1_conditionings += stage_1_ref_conditionings
        if frozen_video_conditioning is None and guiding_images:
            stage_1_conditionings += image_conditionings_by_adding_guiding_latent(
                images=guiding_images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if frozen_video_conditioning is None and video_conditioning:
            stage_1_conditionings += video_conditionings_by_control_video(
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                downscale_factor=video_conditioning_downscale_factor,
                tiling_config=tiling_config,
                continuous_conditioning_and_guide=continuous_conditioning_and_guide,
            )
        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_1_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
            generator=mask_generator,
            num_steps=len(sigmas) - 1,
        )
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(sigmas) - 1, pass_no=stage_1_pass_no)
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stage_1_stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=sigmas[0],
            mask_context=mask_context,
        )
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None
        if skip_stage_2:
            torch.cuda.synchronize()
            del transformer
            del video_encoder
            cleanup_memory()

            latent_slice = None
            if return_latent_slice is not None:
                latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
            if frozen_output_video is None:
                video_latent = [video_state.latent]
                video_state = None
                decoded_video = vae_decode_video_to_tensor(
                    video_latent,
                    self._get_stage_model(1, "video_decoder"),
                    tiling_config,
                    expected_frames=int(stage_1_output_shape.frames),
                    expected_height=int(stage_1_output_shape.height),
                    expected_width=int(stage_1_output_shape.width),
                    interrupt_check=interrupt_check,
                    hdr_transform=hdr_transform,
                    output_dtype=torch.float16 if hdr_transform is not None else None,
                )
            else:
                decoded_video = frozen_output_video[:, :num_frames, :height, :width].permute(1, 2, 3, 0)
            decoded_audio = None if skip_audio else vae_decode_audio(audio_state.latent, self._get_stage_model(1, "audio_decoder"), self._get_stage_model(1, "vocoder"))
            if latent_slice is not None:
                return decoded_video, decoded_audio, latent_slice
            return decoded_video, decoded_audio

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self._get_stage_model(2, "spatial_upsampler"),
        )

        torch.cuda.synchronize()
        cleanup_memory()

        transformer = self._get_stage_model(2, "transformer")
        bind_interrupt_check(transformer, interrupt_check)
        stage_2_ref_context = stage_2_ref_adaln = None
        stage_2_ref_conditionings = []
        use_ltx23_stage_2_sigmas = LTX23_USE_DISTILLED_8_STEPS_STAGE_2_SIGMAS and ltx2_22B_class
        stage_2_sigma_values = DISTILLED_8_STEPS_STAGE_2_SIGMA_VALUES if use_distilled_8_steps or use_ltx23_stage_2_sigmas else STAGE_2_DISTILLED_SIGMA_VALUES
        distilled_sigmas = torch.Tensor(stage_2_sigma_values).to(self.device)
        if custom_sigmas and len(custom_sigmas) > 1 and custom_sigmas[1] is not None:
            distilled_sigmas = torch.tensor([*custom_sigmas[1], 0.0], dtype=torch.float32, device=self.device)
        if loras_slists is not None:
            stage_2_steps = len(distilled_sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_2_steps,
                phase_switch_step=0,
                phase_switch_step2=stage_2_steps,
            )

        if set_progress_status is not None:
            set_progress_status("VAE Encoding")

        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            preview_tools: VideoLatentTools | None = None,
            mask_context=None,
        ) -> tuple[LatentState, LatentState]:
            denoise_fn = simple_denoising_func(
                video_context=v_context_p,
                audio_context=a_context_p,
                transformer=transformer,  # noqa: F821
                alt_guidance_scale=1.0,
                manage_lora_step=not use_hq_sampler,
                ref_context=stage_2_ref_context,
                ref_adaln=stage_2_ref_adaln,
                video_context_mask_builder=v_context_p_mask_builder,
                audio_context_mask_builder=a_context_p_mask_builder,
                skip_audio_to_video=frozen_video_conditioning is not None,
            )
            if tiled_stage_2:
                denoise_fn = spatially_tiled_denoising_func(denoise_fn, stage_2_output_shape, self.pipeline_components, interrupt_check, callback)
            if use_hq_sampler:
                return res2s_audio_video_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=denoise_fn,
                    noise_seed=seed + 20000,
                    mask_context=mask_context,
                    interrupt_check=interrupt_check,
                    callback=callback,
                    preview_tools=preview_tools,
                    pass_no=2,
                    transformer=transformer,
                )
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=denoise_fn,
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=2,
                transformer=transformer,
                self_refiner_handler=self_refiner_handler_stage2,
                self_refiner_handler_audio=self_refiner_handler_audio_stage2,
                self_refiner_generator=generator,
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_ref_conditionings, stage_2_ref_context, stage_2_ref_adaln = build_editanything_reference_conditioning(
            transformer,
            editanything_ref_images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
        )
        if interrupt_check is not None and interrupt_check():
            return None, None
        stage_2_images = images if images_stage2 is None else images_stage2
        if frozen_video_conditioning is not None:
            stage_2_conditionings = video_conditionings_by_frozen_video(
                video=frozen_video_conditioning,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        else:
            stage_2_conditionings = image_conditionings_by_replacing_latent(
                images=stage_2_images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        stage_2_conditionings += stage_2_ref_conditionings
        if frozen_video_conditioning is None and guiding_images_stage2:
            stage_2_conditionings += image_conditionings_by_adding_guiding_latent(
                images=guiding_images_stage2,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if frozen_video_conditioning is None and latent_conditioning_stage2 is not None:
            stage_2_conditionings += latent_conditionings_by_latent_sequence(
                latent_conditioning_stage2,
                strength=1.0,
                start_index=0,
            )
        if frozen_video_conditioning is None and video_conditioning_stage2:
            stage_2_conditionings += video_conditionings_by_control_video(
                video_conditioning=video_conditioning_stage2,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                downscale_factor=video_conditioning_downscale_factor,
                tiling_config=tiling_config,
                continuous_conditioning_and_guide=continuous_conditioning_and_guide,
            )
        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_2_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
            generator=mask_generator,
            num_steps=len(distilled_sigmas) - 1,
        )
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(distilled_sigmas) - 1, pass_no=2)
        freeze_audio_stage2 = audio_identity_guidance_scale > 0.0
        stage_2_audio_conditionings = audio_conditionings if audio_conditionings_stage2 is None else audio_conditionings_stage2
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            audio_conditionings=stage_2_audio_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stage_2_stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            audio_noise_scale=0.0 if freeze_audio_stage2 else distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            mask_context=mask_context,
            freeze_audio=freeze_audio_stage2,
        )
        if video_state is None or audio_state is None:
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        latent_slice = None
        if return_latent_slice is not None:
            latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
        if frozen_output_video is None:
            video_latent = [video_state.latent]
            video_state = None
            decoded_video = vae_decode_video_to_tensor(
                video_latent,
                self._get_stage_model(2, "video_decoder"),
                tiling_config,
                expected_frames=int(stage_2_output_shape.frames),
                expected_height=int(stage_2_output_shape.height),
                expected_width=int(stage_2_output_shape.width),
                interrupt_check=interrupt_check,
                hdr_transform=hdr_transform,
                output_dtype=torch.float16 if hdr_transform is not None else None,
            )
        else:
            decoded_video = frozen_output_video[:, :num_frames, :height, :width].permute(1, 2, 3, 0)
        decoded_audio = None if skip_audio else vae_decode_audio(audio_state.latent, self._get_stage_model(2, "audio_decoder"), self._get_stage_model(2, "vocoder"))
        if latent_slice is not None:
            return decoded_video, decoded_audio, latent_slice
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    from .utils.media_io import encode_video

    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
