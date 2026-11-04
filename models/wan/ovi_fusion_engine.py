import os

import torch
from shared.utils.phase_progress import generation_progress
import logging
from textwrap import indent
import torch.nn as nn
from tqdm import tqdm
from .ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from .ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from shared.utils import files_locator as fl
from shared.utils import latent_io
from .modules.vae2_2 import Wan2_2_VAE
from .modules.t5 import T5EncoderModel
from .ovi.modules.mmaudio.features_utils import FeaturesUtils
from .ovi.modules.fusion import FusionModel
import json
from mmgp import offload
from shared.utils.loras_mutipliers import update_loras_slists, get_model_switch_steps

def init_fusion_score_model_ovi():
    config_root = os.path.join("models", "wan", "ovi", "configs")
    video_config_path = os.path.join(config_root , "video.json")
    audio_config_path = os.path.join(config_root , "audio.json")

    with open(video_config_path, encoding="utf-8") as f:
        video_config = json.load(f)

    with open(audio_config_path, encoding="utf-8") as f:
        audio_config = json.load(f)

    with torch.device("meta"):
        fusion_model = FusionModel(video_config, audio_config)
        
    return fusion_model, video_config, audio_config

def init_mmaudio_vae():
    tod_path = fl.locate_file( os.path.join("mmaudio", "v1-16.pth"))
    bigvgan_path = fl.locate_file(os.path.join("mmaudio", "best_netG.pt"))
 
    vae_config = {
        "mode": "16k",
        "need_vae_encoder": True,
        "tod_vae_ckpt": str(tod_path),
        "bigvgan_vocoder_ckpt": str(bigvgan_path),
    }

    return FeaturesUtils(**vae_config).to("cpu")

class OviFusionEngine:
    def __init__(self,  
                device="cuda",
                model_filename = None, 
                text_encoder_filename = None, 
                VAE_dtype = torch.bfloat16,
                dtype = torch.bfloat16,
                model_def = None,
                **any):
        
        self.device = "cpu"
        self.dtype = dtype
        self.sr = 16000
        self.fps = model_def.get("fps", 24)
        self._interrupt = False
        self.last_audio = None

        # Load fusion model
        self.device = device
        self.target_dtype = torch.bfloat16 # dtype, wont work with torch.float16
        model, video_config, audio_config = init_fusion_score_model_ovi()
        # offload.load_model_data(model, "c:/temp/model_960x960.safetensors")
        offload.load_model_data(model.video_model, model_filename[0], writable_tensors=False, default_dtype=None)
        offload.load_model_data(model.audio_model, model_filename[1], writable_tensors=False, default_dtype=None)
        offload.change_dtype(model, dtype, True)
        model = model.eval()
        # model.set_rope_params()
        self.model = model


        # offload.save_model(model.video_model, "wan2.2_ovi1_1_video_10B_bf16.safetensors")
        # offload.save_model(model.video_model, "wan2.2_ovi1_1_video_10B_quanto_bf16_int8.safetensors", do_quantize=True)

        # offload.save_model(model.audio_model, "wan2.2_ovi1_1_audio_10B_bf16.safetensors")
        # offload.save_model(model.audio_model, "wan2.2_ovi1_1_audio_10B_quanto_bf16_int8.safetensors", do_quantize=True)


        self.vae_stride = (4, 16, 16)
        vae_checkpoint = "Wan2.2_VAE.safetensors"
        self.vae = Wan2_2_VAE( vae_pth=fl.locate_file(vae_checkpoint), dtype= VAE_dtype, device="cpu")
        self.vae.device = self.device # need to set to cuda so that vae buffers are properly moved (although the rest will stay in the CPU)
        self.vae.model.requires_grad_(False).eval()

        vae_model_audio = init_mmaudio_vae()
        vae_model_audio.requires_grad_(False).eval()
        self.audio_vae = vae_model_audio.bfloat16()
        # The mmaudio VAE's encoder checkpoint, for the latent companion's
        # audio entry (the BigVGAN vocoder travels with it in the same
        # folder).
        self.audio_vae_file = "v1-16.pth"
        self.audio_vae_pth = fl.locate_file(os.path.join("mmaudio", self.audio_vae_file), error_if_none=False)
        # Load T5 text model
        text_encoder_folder = model_def.get("text_encoder_folder")
        if text_encoder_folder:
            tokenizer_path = fl.locate_folder(text_encoder_folder)
        else:
            tokenizer_path = os.path.dirname(text_encoder_filename)
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device('cpu'),
            checkpoint_path=text_encoder_filename,
            tokenizer_path=tokenizer_path)
        


        ## Load t2i as part of pipeline
        self.image_model = None
        
        # if config.get("mode") == "t2i2v":
        #     logging.info(f"Loading Flux Krea for first frame generation...")
        #     self.image_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
        #     self.image_model.enable_model_cpu_offload(gpu_id=self.device) #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

        # Fixed attributes, non-configurable
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        self.audio_latent_length = 157
        self.video_latent_length = 31

        logging.info(f"OVI Fusion Engine initialized, GPU VRAM allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")


    @torch.no_grad()
    @generation_progress
    def generate(self,
                    input_prompt, 
                    image_start=None,
                    input_video = None,
                    width = 1280,
                    height = 720,
                    frame_num = 121,
                    seed=100,
                    solver_name="unipc",
                    sampling_steps=50,
                    shift=5.0,
                    guide_scale=5.0,
                    audio_cfg_scalecale=4.0,
                    perturbation_layers=[11],
                    perturbation_start = 0.0,
                    perturbation_end = 1.0,
                    n_prompt="",
                    audio_negative_prompt="",
                    loras_slists = None,
                    callback = None,
                    set_progress_status = None,
                    block_size = 0,                    
                    VAE_tile_size = 0,
                    joint_pass = False,
                    save_latents=False,
                    prefix_latents=None,
                    prefix_audio_latents=None,
                    prefix_kept_windows=None,
                    prefix_kept_frames=0,
                    prefix_is_i2v=False,
                    **bbkwargs,
                ):

        if len(n_prompt) == 0: 
            n_prompt = "jitter, bad hands, blur, distortion"  # Artifacts to avoid in video
        if len(audio_negative_prompt) == 0:
            audio_negative_prompt= "robotic, muffled, echo, distorted"    # Artifacts to avoid in audio

        perturbation_layer = None
        if isinstance(perturbation_layers, (list, tuple)) and perturbation_layers:
            perturbation_layer = int(perturbation_layers[0])
        elif isinstance(perturbation_layers, (int, float)):
            perturbation_layer = int(perturbation_layers)
        if perturbation_layer is None:
            perturbation_layer = 11

        video_frame_height_width=(height, width)

        scheduler_video, timesteps_video = self.get_scheduler_time_steps(
            sampling_steps=sampling_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift
        )
        scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
            sampling_steps=sampling_steps,
            device=self.device,
            solver_name=solver_name,
            shift=shift
        )

        if self._interrupt:
            return None


        if input_video is not None:
            first_frame = input_video #image_start.unsqueeze(1) if is_i2v else None
            is_i2v = True
        else:
            first_frame = None
            is_i2v = False

        if callback != None:
            callback(-1, None, True)

        text_embeddings = self.text_encoder([input_prompt, n_prompt, audio_negative_prompt], device= self.device)
        text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]
        if set_progress_status is not None:
            set_progress_status("Preparing Audio and Video Conditioning")
        # Split embeddings
        text_embeddings_audio_pos = text_embeddings[0]
        text_embeddings_video_pos = text_embeddings[0] 

        text_embeddings_video_neg = text_embeddings[1]
        text_embeddings_audio_neg = text_embeddings[2]

        if is_i2v:
            with torch.no_grad():
                latents_images = self.vae.encode([first_frame], VAE_tile_size)[0].to(self.target_dtype) # c 1 h w 
            latents_images = latents_images.to(self.target_dtype)
            video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]
        else:
            video_h, video_w = video_frame_height_width
            video_latent_h, video_latent_w = video_h // 16, video_w // 16

        if frame_num == 121:
            video_latent_length = 31
            audio_latent_length = 157
        else:
            video_latent_length = 61
            audio_latent_length = 314

        # Latent Branch (Phase 1): the head window's frozen prefixes are
        # the sidecar's kept latents (the video prefix verbatim, the audio
        # prefix the flat [20, t] block sliced by time ratio), so the
        # branch does not re-encode the part it keeps; they are re-frozen
        # at every denoise step and captured into the companion file with
        # the rest of the window. Best effort: a prefix that does not
        # match this window's latent grid is refused with a notice and the
        # head window runs without it.
        branch_video_prefix = None
        branch_audio_prefix = None
        if prefix_latents is not None:
            _prefix = torch.as_tensor(prefix_latents)
            if _prefix.dim() == 5:
                _prefix = _prefix[0]
            _prefix_shape = tuple(int(u) for u in _prefix.shape)
            if _prefix.dim() == 4 and int(_prefix.shape[0]) == self.video_latent_channel \
                    and tuple(int(u) for u in _prefix.shape[2:]) == (video_latent_h, video_latent_w) \
                    and 0 < int(_prefix.shape[1]) <= video_latent_length \
                    and (int(_prefix.shape[1]) - 1) % 4 == 0:
                branch_video_prefix = _prefix.detach().to(dtype=self.target_dtype, device=self.device)
            else:
                print(f"[WanGP][Ovi] The branch prefix {_prefix_shape} does not match this window's video latent grid ([{self.video_latent_channel}, <{video_latent_length}>, {video_latent_h}, {video_latent_w}]); the branch prefix is not used for this run.")
        if prefix_audio_latents is not None:
            _audio_prefix = torch.as_tensor(prefix_audio_latents)
            _audio_prefix_shape = tuple(int(u) for u in _audio_prefix.shape)
            if _audio_prefix.dim() == 2 and int(_audio_prefix.shape[0]) == self.audio_latent_channel \
                    and 0 < int(_audio_prefix.shape[1]) <= audio_latent_length:
                # The sidecar stores [c, t]; the window's audio state is [t, c].
                branch_audio_prefix = _audio_prefix.detach().t().to(dtype=self.target_dtype, device=self.device)
            else:
                print(f"[WanGP][Ovi] The branch audio prefix {_audio_prefix_shape} does not match this window's audio latent grid ([{self.audio_latent_channel}, <{audio_latent_length}>]); the branch audio prefix is not used for this run.")


		
        from .modules.posemb_layers import get_rotary_pos_embed, get_nd_rotary_pos_embed

        video_freqs = get_nd_rotary_pos_embed((0, 0, 0 ), (video_latent_length, video_latent_h//2, video_latent_w//2 ), device=self.device)
        # audio_freqs = get_nd_rotary_pos_embed((0,), (audio_latent_length, ), interpolation_factor= self.model.audio_model.temporal_rope_scaling_factor, rope_dim_list= [44])	
        audio_freqs = self.model.audio_model.get_audio_rope_params(self.device)		
        video_noise = torch.randn((self.video_latent_channel, video_latent_length, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # c, f, h, w
        audio_noise = torch.randn((audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # 1, l c -> l, c
        def ret():
            return None
        
        # Calculate sequence lengths from actual latents
        max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
        _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
        max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]

        update_loras_slists(self.model.video_model, loras_slists, len(timesteps_video))
        kwargs = {
                    'vid_seq_len': max_seq_len_video,
                    'audio_seq_len': max_seq_len_audio,
                    'first_frame_is_clean': is_i2v,
                    'callback' : callback,
                    'pipeline': self,
                    'video_freqs': video_freqs,
                    'audio_freqs': audio_freqs,
        }

        # Sampling loop
        with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
            for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio)), total=min(len(timesteps_video), len(timesteps_audio))):
                timestep_input = torch.full((1,), t_v, device=self.device)
                kwargs.update({
                    "vid": video_noise,
                    "audio" : audio_noise,
                    "t": timestep_input,
                })
                offload.set_step_no_for_lora(self.model.video_model, i)
                if is_i2v:
                    video_noise[:, :1] = latents_images
                if branch_video_prefix is not None:
                    video_noise[:, :branch_video_prefix.shape[1]] = branch_video_prefix
                if branch_audio_prefix is not None:
                    audio_noise[:branch_audio_prefix.shape[0], :] = branch_audio_prefix
                computed_perturbation_layers = perturbation_layers if int(perturbation_start * sampling_steps) <= i < int(perturbation_end * sampling_steps) else None
                any_guidance = not (guide_scale == 1 and audio_cfg_scalecale ==1)

                if any_guidance and not joint_pass:
                    pred_vid_pos, pred_audio_pos = self.model(
                        audio_context= [text_embeddings_audio_pos],
                        vid_context= [text_embeddings_video_pos],
                        x_id_list =[0],
                        **kwargs
                    )
                    if pred_vid_pos is None: 
                        return ret()
                    
                    pred_vid_neg, pred_audio_neg = self.model(
                        audio_context= [text_embeddings_audio_neg],
                        vid_context =[text_embeddings_video_neg],
                        x_id_list =[1],
                        computed_perturbation_layers = computed_perturbation_layers,
                        **kwargs
                    )
                    if pred_vid_neg is None: 
                        return ret()
                else:
                    vid, audio = self.model(
                        audio_context= [text_embeddings_audio_pos, text_embeddings_audio_neg],
                        vid_context= [text_embeddings_video_pos, text_embeddings_video_neg],
                        computed_perturbation_layers = computed_perturbation_layers,
                        x_id_list =[0,1],
                        **kwargs
                    )
                    if vid is None: 
                        return ret()
                    pred_vid_pos, pred_vid_neg = vid
                    pred_audio_pos, pred_audio_neg = audio
                    vid = audio = None
                # Apply classifier-free guidance
                pred_video_guided = pred_vid_neg + guide_scale * (pred_vid_pos - pred_vid_neg)
                pred_audio_guided = pred_audio_neg + audio_cfg_scalecale * (pred_audio_pos - pred_audio_neg)
                pred_audio_neg = pred_audio_pos = pred_vid_neg = pred_vid_pos = None
                # Update noise using scheduler
                video_noise = scheduler_video.step(
                    pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)
                pred_video_guided = None
                audio_noise = scheduler_audio.step(
                    pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)
                pred_audio_guided = None
                if callback is not None:
                    latents_preview = video_noise
                    callback(i, latents_preview, False )
                    latents_preview = None
            ret()

            if is_i2v:
                video_noise[:, :1] = latents_images
            if branch_video_prefix is not None:
                video_noise[:, :branch_video_prefix.shape[1]] = branch_video_prefix
            if branch_audio_prefix is not None:
                audio_noise[:branch_audio_prefix.shape[0], :] = branch_audio_prefix

            # Capture the final pre-decode latents for the optional latent
            # companion file ('Save Latents'): the video latent is the
            # final denoised [48, t, h, w] state and the audio latent the
            # final [20, t] state; both are captured right before their
            # VAE decodes.
            latent_file_obj = None
            if save_latents:
                try:
                    latent_file_obj = self._build_latent_file_obj(
                        video=video_noise,
                        audio=audio_noise,
                        frame_num=frame_num,
                        fps=float(bbkwargs.get("fps") or self.fps or 24),
                        seed=int(seed),
                        input_prompt=input_prompt,
                        sampling_steps=int(sampling_steps),
                        denoising_strength=float(bbkwargs.get("denoising_strength") or 1.0),
                        window_no=int(bbkwargs.get("window_no") or 1),
                        window_start_frame_no=int(bbkwargs.get("window_start_frame_no") or 0),
                        loras_slists=loras_slists,
                        kwargs=bbkwargs)
                    print(f"Latent companion payload built (video {tuple(int(u) for u in video_noise.shape)}, audio {tuple(int(u) for u in audio_noise.shape)}).")
                except Exception as exc:
                    print(f"Warning: the latent companion file was not saved: {exc}")
                    latent_file_obj = None

            # Decode audio
            audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
            generated_audio = self.audio_vae.wrapped_decode(audio_latents_for_vae)
            generated_audio = generated_audio.squeeze().cpu().float().numpy()
            
            # Decode video
            generated_video = self.vae.decode_to_cpu_uint8([video_noise], VAE_tile_size, target_frames=frame_num, target_height=height, target_width=width)[0]

        # self.last_audio = audio
        output = {"x": generated_video, "audio": generated_audio}
        if latent_file_obj is not None:
            # Hand the payload to the caller instead of the side-files dict: the
            # caller owns the companion file (it must accumulate the windows of
            # a sliding-window job) and the raw bytes are never echoed to API
            # responses.
            output["latent_file_obj"] = latent_file_obj
        if prefix_kept_windows is not None and int(prefix_kept_frames or 0) > 0:
            # The branch job's kept region is a fresh decode of the source
            # sidecar's kept windows (the same reconstruction the Latent
            # Decode job performs), handed back as decoded_prefix so the job
            # uses it as the output video's prefix instead of a
            # re-extraction of the source video. Best effort: a decode that
            # fails keeps the source-pixel prefix the job set up before the
            # model call.
            _vae_dtype = getattr(self.vae, "dtype", None) or torch.float32
            def _decode_kept_window(spec):
                _lat = spec["latents"].to(device=self.device, dtype=_vae_dtype)
                _videos = self.vae.decode_to_cpu_uint8([_lat], VAE_tile_size)
                if not _videos or _videos[0] is None:
                    return None
                return _videos[0].to("cpu")
            try:
                decoded_prefix = latent_io.decode_kept_windows(
                    _decode_kept_window, prefix_kept_windows, int(prefix_kept_frames),
                    is_i2v=bool(prefix_is_i2v))
                if decoded_prefix is not None:
                    output["decoded_prefix"] = decoded_prefix
            except Exception as exc:
                print(f"Latent branch: the kept region could not be decoded ({exc}); the output keeps the source-pixel prefix.")
        return output


                
    def _build_latent_file_obj(self, *, video, audio, frame_num, fps, seed, input_prompt, sampling_steps,
                               denoising_strength, window_no, window_start_frame_no, loras_slists, kwargs):
        """Build the ``*_latent.pt`` companion payload from the final
        pre-decode latents.

        Called right before the video/audio VAE decodes (see ``generate``)
        so the latent can later be re-decoded without the diffusion model
        (Media Flow Latent Decode). The video entry is a standard
        ``wgp_latent/1`` entry carrying the [48, t, h, w] latent of the
        Wan 2.2 5B-class VAE (the linear 4/1 layout the Wan family
        records); the audio entry carries the [20, t] latent of the
        mmaudio VAE, decoded to 16 kHz mono.
        The file itself is written by the caller (the all-latent sidecar:
        the job's windows stacked on one time dimension, the sidecar the
        sole content source of the video).
        """
        def vae_sha(path_or_name):
            """SHA-256 of a VAE file on disk. The engine keeps the resolved
            path of its video VAE, so a name that is not a path on its own
            is resolved through the app's files locator (the ckpts/
            folder) before hashing; an unlocatable file contributes no
            hash (the readers treat an empty hash as unrecorded)."""
            if not path_or_name:
                return ""
            path = path_or_name if os.path.isabs(str(path_or_name)) else fl.locate_file(str(path_or_name), error_if_none=False)
            if not path or not os.path.isfile(str(path)):
                return ""
            return latent_io.sha256_file(str(path))

        video_latent = video.detach().to(dtype=torch.float32, device="cpu")
        if video_latent.dim() == 5:
            video_latent = video_latent[0]
        lora_names = []
        if loras_slists:
            for lora_entry in loras_slists:
                name = lora_entry if isinstance(lora_entry, str) else (lora_entry[0] if isinstance(lora_entry, (list, tuple)) and len(lora_entry) > 0 else "")
                if len(str(name).strip()) > 0:
                    lora_names.append(str(name))
        video_vae_pth = getattr(self.vae, "vae_pth", None)
        video_payload = latent_io.build_latent_payload(
            video_latent,
            app_version=str(kwargs.get("app_version") or ""),
            model_type=str(kwargs.get("latent_model_type") or ""),
            base_model_type="ovi",
            model_filename=str(kwargs.get("model_filename") or ""),
            loras=lora_names,
            vae_file=os.path.basename(str(video_vae_pth)) if video_vae_pth else "",
            vae_sha256=vae_sha(str(video_vae_pth)) if video_vae_pth else "",
            vae_z_dim=int(video_latent.shape[0]),
            vae_dtype=str(getattr(self.vae, "dtype", "torch.float32")).replace("torch.", ""),
            frames=int(frame_num),
            fps=float(fps),
            seed=int(seed),
            prompt=str(input_prompt)[:1000],
            denoising_strength=float(denoising_strength),
            steps=int(sampling_steps),
        )
        payload = dict(video_payload)
        audio_latent = audio.detach().to(dtype=torch.float32, device="cpu")
        if audio_latent.dim() == 3:
            audio_latent = audio_latent[0]
        audio_samples = int(round(frame_num / fps * self.sr)) if fps > 0 else int(audio_latent.shape[-1])
        audio_payload = latent_io.build_audio_latent_payload(
            audio_latent,
            [{
                "window_no": int(window_no or 1),
                "start_sample": int(round(window_start_frame_no / fps * self.sr)) if fps > 0 else 0,
                "samples": audio_samples,
            }],
            app_version=str(kwargs.get("app_version") or ""),
            model_type=str(kwargs.get("latent_model_type") or ""),
            base_model_type="ovi",
            model_filename=str(kwargs.get("model_filename") or ""),
            loras=lora_names,
            audio_vae_file=self.audio_vae_file,
            audio_vae_sha256=vae_sha(os.path.join("mmaudio", self.audio_vae_file)),
            audio_vae_dtype="bfloat16",
            sample_rate=self.sr,
            channels=1,
            samples=audio_samples,
            seed=int(seed),
            prompt=str(input_prompt)[:1000],
            denoising_strength=float(denoising_strength),
            steps=int(sampling_steps),
            total_windows=1,
        )
        payload.update(audio_payload)
        return payload

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps
    
    def custom_compile(self, **compile_kwargs):
        self.model.custom_compile(compile_kwargs)

    def get_trans_lora(self):
        return self.model.video_model, None
