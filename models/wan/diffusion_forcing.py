import math
import os
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import logging
import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .modules.posemb_layers import get_rotary_pos_embed
from shared.utils.utils import calculate_new_dimensions
from shared.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from shared.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils import files_locator as fl
class DTT2V:


    def __init__(
        self,
        config,
        checkpoint_dir,
        model_filename = None,
        model_type = None,
        model_def = None,
        base_model_type = None,
        save_quantized = False,
        text_encoder_filename = None,
        quantizeTransformer = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False,
    ):
        self.device = torch.device(f"cuda")
        self.config = config
        self.dtype = dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.text_len = config.text_len 
        text_encoder_folder = (model_def or {}).get("text_encoder_folder")
        if text_encoder_folder:
            tokenizer_path = fl.locate_folder(text_encoder_folder)
        else:
            tokenizer_path = os.path.dirname(text_encoder_filename)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=text_encoder_filename,
            tokenizer_path=tokenizer_path)
        self.model_def = model_def
        self.image_outputs = model_def.get("image_outputs", False)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size 

        self.vae = WanVAE(
            vae_pth= fl.locate_file(config.vae_checkpoint.replace(".pth", ".safetensors")), dtype= VAE_dtype,
            device=self.device)

        logging.info(f"Creating WanModel from {model_filename[-1]}")
        from mmgp import offload
        # model_filename = "model.safetensors"
        # model_filename = "c:/temp/diffusion_pytorch_model-00001-of-00006.safetensors"
        base_config_file = f"configs/{base_model_type}.json"
        forcedConfigPath = base_config_file if len(model_filename) > 1 else None
        def pre_load_callback(model):
            model.lock_layers_dtypes(torch.float32 if mixed_precision_transformer else dtype)
        self.model = offload.fast_load_transformers_model(model_filename, modelClass=WanModel,do_quantize= quantizeTransformer, writable_tensors= False, default_dtype=dtype, pre_load_callback=pre_load_callback, forcedConfigPath=forcedConfigPath)
        # offload.load_model_data(self.model, "recam.ckpt")
        # self.model.cpu()
        # dtype = torch.float16
        # offload.save_model(self.model, "sky_reels2_diffusion_forcing_1.3B_mbf16.safetensors", config_file_path="config.json") 
        # offload.save_model(self.model, "sky_reels2_diffusion_forcing_720p_14B_quanto_mbf16_int8.safetensors", do_quantize= True, config_file_path="c:/temp/config _df720.json") 
        # offload.save_model(self.model, "rtfp16_int8.safetensors", do_quantize= "config.json") 

        self.model.eval().requires_grad_(False)
        if save_quantized:            
            from wgp import save_quantized_model
            save_quantized_model(self.model, model_type, model_filename[0], dtype, base_config_file)

        self.scheduler = FlowUniPCMultistepScheduler()

        self.model.apply_post_init_changes()

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1

    def encode_image(
        self, image_start: PipelineImageInput, height: int, width: int, num_frames: int, tile_size = 0, causal_block_size = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # prefix_video
        prefix_video = np.array(image_start.resize((width, height))).transpose(2, 0, 1)
        prefix_video = torch.tensor(prefix_video).unsqueeze(1)  # .to(image_embeds.dtype).unsqueeze(1)
        if prefix_video.dtype == torch.uint8:
            prefix_video = (prefix_video.float() / (255.0 / 2.0)) - 1.0
        prefix_video = prefix_video.to(self.device)
        prefix_video = [self.vae.encode(prefix_video.unsqueeze(0), tile_size = tile_size)[0]]  # [(c, f, h, w)]
        if prefix_video[0].shape[1] % causal_block_size != 0:
            truncate_len = prefix_video[0].shape[1] % causal_block_size
            print("the length of prefix video is truncated for the casual block size alignment.")
            prefix_video[0] = prefix_video[0][:, : prefix_video[0].shape[1] - truncate_len]
        predix_video_latent_length = prefix_video[0].shape[1]
        return prefix_video, predix_video_latent_length

    def prepare_latents(
        self,
        shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        return randn_tensor(shape, generator, device=device, dtype=dtype)

    def generate_timestep_matrix(
        self,
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert ar_step >= min_ar_step, f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, casual_block_size, num_frames_block, base_num_frames_block)
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            valid_interval = [(s * casual_block_size, e * casual_block_size) for s, e in valid_interval]

        return step_matrix, step_index, step_update_mask, valid_interval

    @torch.no_grad()
    def generate(
        self,
        input_prompt: Union[str, List[str]],
        n_prompt: Union[str, List[str]] = "",
        input_video = None,
        height: int = 480,
        width: int = 832,
        fit_into_canvas = True,
        frame_num: int = 97,
        batch_size = 1,
        sampling_steps: int = 50,
        shift: float = 1.0,
        guide_scale: float = 5.0,
        seed: float = 0.0,
        overlap_noise: int = 0,
        model_mode: int = 5,
        causal_block_size: int = 5,
        causal_attention: bool = True,
        fps: int = 24,
        VAE_tile_size = 0,
        joint_pass = False,
        perturbation_layers = None,
        perturbation_start = 0.0,
        perturbation_end = 1.0,
        callback = None,
        loras_slists = None,
        **bbargs
    ):
        self._interrupt = False
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self._guidance_scale = guide_scale
        if frame_num > 1:
            frame_num = max(17, frame_num) # must match causal_block_size for value of 5
            frame_num = int( round( (frame_num - 17) / 20)* 20 + 17 )
        ar_step = model_mode
        if ar_step == 0: 
            causal_block_size = 1
            causal_attention = False

        i2v_extra_kwrags = {}
        prefix_video = None
        predix_video_latent_length = 0

        if input_video != None:
            _ , _ , height, width  = input_video.shape


        latent_length = (frame_num - 1) // 4 + 1 
        latent_height = height // 8
        latent_width = width // 8

        if self._interrupt:
            return None
        text_len = self.text_len
        prompt_embeds = self.text_encoder([input_prompt], self.device)[0]
        prompt_embeds  = prompt_embeds.to(self.dtype).to(self.device)
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds.new_zeros(text_len -prompt_embeds.size(0), prompt_embeds.size(1)) ]).unsqueeze(0) 

        if self.do_classifier_free_guidance:
            negative_prompt_embeds = self.text_encoder([n_prompt], self.device)[0]
            negative_prompt_embeds  = negative_prompt_embeds.to(self.dtype).to(self.device)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds.new_zeros(text_len -negative_prompt_embeds.size(0), negative_prompt_embeds.size(1)) ]).unsqueeze(0) 

        if self._interrupt:
            return None

        self.scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        init_timesteps = self.scheduler.timesteps
        fps_embeds = [fps] #* prompt_embeds[0].shape[0]
        fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]


        output_video = input_video

        if output_video is not None:  # i !=0
            prefix_video = output_video.to(self.device)
            prefix_video = self.vae.encode(prefix_video.unsqueeze(0))[0]  # [(c, f, h, w)]
            predix_video_latent_length = prefix_video.shape[1]
            truncate_len = predix_video_latent_length % causal_block_size
            if truncate_len != 0:
                if truncate_len == predix_video_latent_length:
                    causal_block_size = 1
                    causal_attention = False
                    ar_step = 0
                else:
                    print("the length of prefix video is truncated for the casual block size alignment.")
                    predix_video_latent_length -= truncate_len
                    prefix_video = prefix_video[:, : predix_video_latent_length]

        base_num_frames_iter = latent_length
        latent_shape = [batch_size, 16, base_num_frames_iter, latent_height, latent_width]
        latents = self.prepare_latents(
            latent_shape, dtype=torch.float32, device=self.device, generator=generator
        )
        if prefix_video is not None:
            latents[:, :, :predix_video_latent_length] = prefix_video.to(torch.float32)
        step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
            base_num_frames_iter,
            init_timesteps,
            base_num_frames_iter,
            ar_step,
            predix_video_latent_length,
            causal_block_size,
        )
        sample_schedulers = []
        for _ in range(base_num_frames_iter):
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            sample_schedulers.append(sample_scheduler)
        sample_schedulers_counter = [0] * base_num_frames_iter

        updated_num_steps=  len(step_matrix)
        if callback != None:
            update_loras_slists(self.model, loras_slists, updated_num_steps)
            callback(-1, None, True, override_num_inference_steps = updated_num_steps)
        skip_steps_cache = self.model.cache
        if skip_steps_cache != None:
            skip_steps_cache.num_steps = updated_num_steps
            if skip_steps_cache.cache_type == "tea":
                x_count = 2 if self.do_classifier_free_guidance else 1
                skip_steps_cache.previous_residual = [None] * x_count 
                time_steps_comb = []
                skip_steps_cache.steps = updated_num_steps
                for i, timestep_i in enumerate(step_matrix):
                    valid_interval_start, valid_interval_end = valid_interval[i]
                    timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                    if overlap_noise > 0 and valid_interval_start < predix_video_latent_length:
                        timestep[:, valid_interval_start:predix_video_latent_length] = overlap_noise
                    time_steps_comb.append(timestep)
                self.model.compute_teacache_threshold(skip_steps_cache.start_step, time_steps_comb, skip_steps_cache.multiplier)
                del time_steps_comb
            else:
                self.model.cache = None
        from mmgp import offload
        freqs = get_rotary_pos_embed(latents.shape[2 :], enable_RIFLEx= False, device=latents.device) 
        kwrags = {
            "freqs" :freqs,
            "fps" : fps_embeds,
            "causal_block_size" : causal_block_size,
            "causal_attention" : causal_attention,
            "callback" : callback,
            "pipeline" : self,
        }   
        kwrags.update(i2v_extra_kwrags)

        for i, timestep_i in enumerate(tqdm(step_matrix)):
            kwrags["perturbation_layers"] = perturbation_layers if int(perturbation_start * updated_num_steps) <= i < int(perturbation_end * updated_num_steps) else None

            offload.set_step_no_for_lora(self.model, i)
            update_mask_i = step_update_mask[i]
            valid_interval_start, valid_interval_end = valid_interval[i]
            timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
            latent_model_input = latents[:, :, valid_interval_start:valid_interval_end, :, :].clone()
            if overlap_noise > 0 and valid_interval_start < predix_video_latent_length:
                noise_factor = 0.001 * overlap_noise
                timestep_for_noised_condition = overlap_noise
                latent_model_input[:, :, valid_interval_start:predix_video_latent_length] = (
                    latent_model_input[:, :, valid_interval_start:predix_video_latent_length]
                    * (1.0 - noise_factor)
                    + torch.randn_like(
                        latent_model_input[:, :, valid_interval_start:predix_video_latent_length]
                    )
                    * noise_factor
                )
                timestep[:, valid_interval_start:predix_video_latent_length] = timestep_for_noised_condition
            kwrags.update({
                "t" : timestep,
                "current_step_no" : i,                 
                })

            # with torch.autocast(device_type="cuda"):                
            if True:
                if not self.do_classifier_free_guidance:
                    noise_pred = self.model(
                        x=[latent_model_input],
                        context=[prompt_embeds],
                        **kwrags,
                    )[0]
                    if self._interrupt:
                        return None
                    noise_pred= noise_pred.to(torch.float32)                                                                  
                else:
                    if joint_pass:
                        noise_pred_cond, noise_pred_uncond = self.model(
                            x=[latent_model_input, latent_model_input],
                            context= [prompt_embeds, negative_prompt_embeds],
                            **kwrags,
                        )
                        if self._interrupt:
                            return None                
                    else:
                        noise_pred_cond = self.model(
                            x=[latent_model_input],
                            x_id=0,
                            context=[prompt_embeds],
                            **kwrags,
                        )[0]
                        if self._interrupt:
                            return None                
                        noise_pred_uncond = self.model(
                            x=[latent_model_input],
                            x_id=1,
                            context=[negative_prompt_embeds],
                            **kwrags,
                        )[0]
                        if self._interrupt:
                            return None
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                    del noise_pred_cond, noise_pred_uncond
            preview_latents = latents.clone() if getattr(callback, "tiny_vae", False) else None
            for idx in range(valid_interval_start, valid_interval_end):
                if update_mask_i[idx].item():
                    if preview_latents is not None:
                        preview_step_index = sample_schedulers[idx].step_index
                        if preview_step_index is None:
                            preview_step_index = sample_schedulers[idx].index_for_timestep(timestep_i[idx])
                        preview_latents[:, :, idx] = latents[:, :, idx] - sample_schedulers[idx].sigmas[preview_step_index] * noise_pred[:, :, idx - valid_interval_start]
                    latents[:, :, idx] = sample_schedulers[idx].step(
                        noise_pred[:, :, idx - valid_interval_start],
                        timestep_i[idx],
                        latents[:, :, idx],
                        return_dict=False,
                        generator=generator,
                    )[0]
                    sample_schedulers_counter[idx] += 1
            if callback is not None:
                latents_preview = preview_latents if preview_latents is not None else latents
                if len(latents_preview) > 1: latents_preview = latents_preview.transpose(0,2)
                callback(i, latents_preview[0], False)
                latents_preview = preview_latents = None

        x0 =latents.unbind(dim=0)

        videos = self.vae.decode(x0, VAE_tile_size)

        if self.image_outputs:
            videos = torch.cat(videos, dim=1) if len(videos) > 1 else videos[0]
        else:
            videos = videos[0] # return only first video     

        return videos

