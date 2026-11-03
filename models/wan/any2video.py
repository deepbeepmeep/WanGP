# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import gc
import logging
import math
import os
import random
import time
import sys
import types
import math
from contextlib import contextmanager
from functools import partial
from mmgp import offload, transfer
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from .modules.model import WanModel
from mmgp.offload import get_cache, clear_caches
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .modules.vae2_2 import Wan2_2_VAE

from .modules.clip import CLIPModel
from shared.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from shared.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .modules.posemb_layers import (
    get_rotary_pos_embed,
    get_nd_rotary_pos_embed,
    set_rope_freqs_dtype,
    set_use_fp32_rope_freqs,
)
from shared.utils.vace_preprocessor import VaceVideoProcessor
from shared.utils.basic_flowmatch import FlowMatchScheduler
from shared.utils.euler_scheduler import EulerScheduler
from shared.utils.lcm_scheduler import LCMScheduler
from shared.utils.utils import get_outpainting_frame_location, resize_lanczos, calculate_new_dimensions, convert_image_to_tensor, convert_tensor_to_image, fit_image_into_canvas, guide_mask_to_float, guide_to_float
from .multitalk.multitalk_utils import MomentumBuffer, adaptive_projected_guidance, match_and_blend_colors, match_and_blend_colors_with_mask
from .wanmove.trajectory import replace_feature, create_pos_feature_map
from .alpha.utils import load_gauss_mask, apply_alpha_shift
from shared.utils.audio_video import save_video
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.phase_progress import control_video_encoding, generation_progress, text_encoding_prompts
from shared.utils.self_refiner import PnPHandler, create_self_refiner_handler
from mmgp import safetensors2
from shared.utils import files_locator as fl 
from .scail2 import prepare_scail2_conditioning, test_scail2_replace
from .shotplan import compile_shotplan_prompt

WAN_USE_FP32_ROPE_FREQS = True

def get_vista4d_rotary_pos_embed(latents_size, device):
    lat_t, lat_h, lat_w = latents_size
    grid_t, grid_h, grid_w = lat_t, lat_h // 2, lat_w // 2
    offset = max(31, grid_t)
    cos_parts, sin_parts = [], []
    for start in (0, offset, offset * 2):
        cos, sin = get_nd_rotary_pos_embed((start, 0, 0), (start + grid_t, grid_h, grid_w), (grid_t, grid_h, grid_w), L_test=grid_t, device=device)
        cos_parts.append(cos)
        sin_parts.append(sin)
    return torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0)

def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    
    return st_star

def timestep_transform(t, shift=5.0, num_timesteps=1000 ):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t
    

class WanAny2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        model_filename = None,
        submodel_no_list = None,
        model_type = None, 
        model_def = None,
        base_model_type = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        save_quantized = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False,
        VAE_upsampling = None,
    ):
        self.device = torch.device(f"cuda")
        self.config = config
        self.VAE_dtype = VAE_dtype
        self.dtype = dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.model_def = model_def
        self.model2 = None
        self.transformer_switch = model_def.get("URLs2", None) is not None
        self.is_mocha = model_def.get("mocha_mode", False)
        self.text_encoder = None
        self.text_encoder_cache = None
        if base_model_type != "kiwi_edit":
            text_encoder_folder = model_def.get("text_encoder_folder")
            if text_encoder_folder:
                tokenizer_path = os.path.dirname(fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json")))
            else:
                tokenizer_path = os.path.dirname(text_encoder_filename)
            self.text_encoder = T5EncoderModel(text_len=config.text_len, dtype=config.t5_dtype, device=torch.device('cpu'), checkpoint_path=text_encoder_filename, tokenizer_path=tokenizer_path)
            self.text_encoder_cache = TextEncoderCache()
        if hasattr(config, "clip_checkpoint") and not model_def.get("i2v_2_2", False) or base_model_type in ["animate"]:
            self.clip = CLIPModel(
                dtype=config.clip_dtype,
                device=self.device,
                checkpoint_path=fl.locate_file(config.clip_checkpoint),
                tokenizer_path=fl.locate_folder("xlm-roberta-large"))

        ignore_unused_weights = model_def.get("ignore_unused_weights", False)
        vae_upsampler_factor = 1
        vae_checkpoint2 = None
        vae_checkpoint = model_def.get("VAE_URLs", None )
        vae = WanVAE
        self.vae_stride = config.vae_stride            
        if isinstance(vae_checkpoint, str):
            pass
        elif isinstance(vae_checkpoint, list) and len(vae_checkpoint):
            vae_checkpoint = fl.locate_file(vae_checkpoint[0])
        elif model_def.get("wan_5B_class", False):
            self.vae_stride = (4, 16, 16)
            vae_checkpoint = "Wan2.2_VAE.safetensors"
            vae = Wan2_2_VAE
        else:
            if VAE_upsampling is not None:
                vae_upsampler_factor = 2
                vae_checkpoint ="Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"
            elif model_def.get("alpha_class", False):
                if base_model_type == "alpha2":
                    vae_checkpoint = "wan_alpha_2.1_vae_rgb_channel_v2.safetensors"
                    vae_checkpoint2 = "wan_alpha_2.1_vae_alpha_channel_v2.safetensors"
                else:
                    vae_checkpoint ="wan_alpha_2.1_vae_rgb_channel.safetensors"
                    vae_checkpoint2 ="wan_alpha_2.1_vae_alpha_channel.safetensors"
            else:
                vae_checkpoint = "Wan2.1_VAE.safetensors"                
        self.patch_size = config.patch_size 
        
        self.vae = vae( vae_pth=fl.locate_file(vae_checkpoint), dtype= VAE_dtype, upsampler_factor = vae_upsampler_factor, device="cpu")
        self.vae.upsampling_set = VAE_upsampling
        self.vae.device = self.device # need to set to cuda so that vae buffers are properly moved (although the rest will stay in the CPU)
        self.vae2 = None
        if vae_checkpoint2 is not None:
            self.vae2 = vae( vae_pth=fl.locate_file(vae_checkpoint2), dtype= VAE_dtype, device="cpu")
            self.vae2.device = self.device
        
        # config_filename= "configs/t2v_1.3B.json"
        # import json
        # with open(config_filename, 'r', encoding='utf-8') as f:
        #     config = json.load(f)
        # sd = safetensors2.torch_load_file(xmodel_filename)
        # model_filename = "c:/temp/wan2.2i2v/low/diffusion_pytorch_model-00001-of-00006.safetensors"
        base_config_file = model_def.get("config_file", f"models/wan/configs/{base_model_type}.json")
        base_config_file2 = model_def.get("config_file2", base_config_file)
        forcedConfigPath = base_config_file if len(model_filename) > 1 else None
        # forcedConfigPath = base_config_file = f"configs/flf2v_720p.json"
        # model_filename[1] = xmodel_filename
        self.model = self.model2 = None
        source =  model_def.get("source", None)
        source2 = model_def.get("source2", None)
        module_source =  model_def.get("module_source", None)
        module_source2 =  model_def.get("module_source2", None)
        def preprocess_sd(sd):
            return WanModel.preprocess_sd_with_dtype(dtype, sd)
        def pre_load_callback(model):
            model.lock_layers_dtypes(torch.float32 if mixed_precision_transformer else dtype)
        kwargs= { "modelClass": WanModel,"do_quantize": quantizeTransformer and not save_quantized, "defaultConfigPath": base_config_file , "ignore_unused_weights": ignore_unused_weights, "writable_tensors": False, "default_dtype": dtype, "preprocess_sd": preprocess_sd, "pre_load_callback": pre_load_callback, "forcedConfigPath": forcedConfigPath, }
        kwargs2 = {**kwargs, "defaultConfigPath": base_config_file2, "forcedConfigPath": base_config_file2}
        kwargs_light= { "modelClass": WanModel,"writable_tensors": False, "default_dtype": dtype, "preprocess_sd": preprocess_sd, "pre_load_callback": pre_load_callback, "forcedConfigPath" : base_config_file}
        kwargs_light2 = {**kwargs_light, "forcedConfigPath": base_config_file2}
        if module_source is not None:
            self.model = offload.fast_load_transformers_model(model_filename[:1] + [fl.locate_file(module_source)], **kwargs)
        if module_source2 is not None:
            self.model2 = offload.fast_load_transformers_model(model_filename[1:2] + [fl.locate_file(module_source2)], **kwargs2)
        if source is not None:
            self.model = offload.fast_load_transformers_model(fl.locate_file(source),  **kwargs_light)
        if source2 is not None:
            self.model2 = offload.fast_load_transformers_model(fl.locate_file(source2), **kwargs_light2)

        if self.model is not None or self.model2 is not None:
            from wgp import save_model
            from mmgp.safetensors2 import torch_load_file
        else:
            if self.transformer_switch:
                if 0 in submodel_no_list[2:] and 1 in submodel_no_list[2:]:
                    raise Exception("Shared and non shared modules at the same time across multipe models is not supported")
                
                if 0 in submodel_no_list[2:]:
                    shared_modules= {}
                    self.model = offload.fast_load_transformers_model(model_filename[:1], modules = model_filename[2:], return_shared_modules= shared_modules, **kwargs)
                    self.model2 = offload.fast_load_transformers_model(model_filename[1:2], modules = shared_modules, **kwargs2)
                    shared_modules = None
                else:
                    modules_for_1 =[ file_name for file_name, submodel_no in zip(model_filename[2:],submodel_no_list[2:] ) if submodel_no ==1 ]
                    modules_for_2 =[ file_name for file_name, submodel_no in zip(model_filename[2:],submodel_no_list[2:] ) if submodel_no ==2 ]
                    self.model = offload.fast_load_transformers_model(model_filename[:1], modules = modules_for_1, **kwargs)
                    self.model2 = offload.fast_load_transformers_model(model_filename[1:2], modules = modules_for_2, **kwargs2)

            else:
                self.model = offload.fast_load_transformers_model(model_filename,  **kwargs)
        

        if module_source is not None:
            save_model(self.model, model_type, dtype, None, is_module=True, filter=list(torch_load_file(module_source)), module_source_no=1)
        if module_source2 is not None:
            save_model(self.model2, model_type, dtype, None, is_module=True, filter=list(torch_load_file(module_source2)), module_source_no=2)
        if not source is None:
            save_model(self.model, model_type, dtype, None, submodel_no= 1)
        if not source2 is None:
            save_model(self.model2, model_type, dtype, None, submodel_no= 2)

        if save_quantized:
            from wgp import save_quantized_model
            if self.model is not None:
                save_quantized_model(self.model, model_type, model_filename[0], dtype, base_config_file)
            if self.model2 is not None and model_def.get("save_quantized_submodel2", True):
                save_quantized_model(self.model2, model_type, model_filename[1], dtype, base_config_file2, submodel_no=2)
        self.sample_neg_prompt = config.sample_neg_prompt

        self.use_fp32_rope_freqs = bool(model_def.get("wan_rope_freqs_fp32", WAN_USE_FP32_ROPE_FREQS))
        set_use_fp32_rope_freqs(self.use_fp32_rope_freqs)
        set_rope_freqs_dtype(self.dtype)

        self.model.apply_post_init_changes()
        if self.model2 is not None: self.model2.apply_post_init_changes()
        
        self.kiwi_mllm = None
        self.kiwi_source_embedder_file = None
        self.kiwi_ref_embedder_file = None
        if base_model_type == "kiwi_edit":
            from .kiwi.mllm import KiwiMLLMContextEncoder
            self.kiwi_mllm = KiwiMLLMContextEncoder(
                mllm_root_folder=model_def.get("kiwi_mllm_folder", "kiwi_mllm_encoder_instruct_reference"),
                qwen_weights_path=text_encoder_filename,
                any_ref=model_def.get("any_kiwi_ref", True),
                device=self.device,
                dtype=self.dtype,
                offload_after_encode=True,
            )
            self.kiwi_source_embedder_file = model_def.get("kiwi_source_embedder_file", None)
            self.kiwi_ref_embedder_file = model_def.get("kiwi_ref_embedder_file", None)
        
        self.num_timesteps = 1000 
        self.use_timestep_transform = True 

    def vace_encode_frames(self, frames, ref_images, masks=None, tile_size = 0, overlapped_latents = None):
        ref_images = [ref_images] * len(frames)

        if masks is None:
            latents = self.vae.encode(frames, tile_size = tile_size)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive, tile_size = tile_size)

            if overlapped_latents  != None and False : # disabled as quality seems worse
                # inactive[0][:, 0:1] = self.vae.encode([frames[0][:, 0:1]], tile_size = tile_size)[0] # redundant
                for t in inactive:
                    t[:, 1:overlapped_latents.shape[1] + 1] = overlapped_latents
                overlapped_latents[: 0:1] = inactive[0][: 0:1]

            reactive = self.vae.encode(reactive, tile_size = tile_size)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = self.vae.encode(refs, tile_size = tile_size)
                else:
                    ref_latent = self.vae.encode(refs, tile_size = tile_size)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        ref_images = [ref_images] * len(masks)
        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.vae_stride[0]) # nb latents token without (ref tokens not included)
            height = 2 * (int(height) // (self.vae_stride[1] * 2))
            width = 2 * (int(width) // (self.vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, self.vae_stride[1], width, self.vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                self.vae_stride[1] * self.vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros(mask.shape[0], length, *mask.shape[-2:], dtype=mask.dtype, device=mask.device)
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks


    def get_vae_latents(self, ref_images, device, tile_size= 0):
        ref_vae_latents = []
        for ref_image in ref_images:
            ref_image = TF.to_tensor(ref_image).sub_(0.5).div_(0.5).to(self.device)
            img_vae_latent = self.vae.encode([ref_image.unsqueeze(1)], tile_size= tile_size)
            ref_vae_latents.append(img_vae_latent[0])
                    
        return torch.cat(ref_vae_latents, dim=1)

    def get_i2v_mask(self, lat_h, lat_w, nb_frames_unchanged=0, mask_pixel_values=None, lat_t =0,  device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = F.interpolate(mask_pixel_values.to(device), size=(lat_h, lat_w), mode='nearest')

        if nb_frames_unchanged >0:
            msk[:, :nb_frames_unchanged] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1,2)[0]
        return msk

    def encode_reference_images(self, ref_images, ref_prompt="image of a face", any_guidance= False, tile_size = None, enable_loras = True):
        ref_images = [convert_image_to_tensor(img).unsqueeze(1).to(device=self.device, dtype=self.dtype) for img in ref_images]
        shape = ref_images[0].shape
        freqs = get_rotary_pos_embed( (len(ref_images) , shape[-2] // 8, shape[-1] // 8 ), device=self.device) 
        # batch_ref_image: [B, C, F, H, W]
        vae_feat = self.vae.encode(ref_images, tile_size = tile_size)
        vae_feat = torch.cat( vae_feat, dim=1).unsqueeze(0)
        if any_guidance:
            vae_feat_uncond = self.vae.encode([ref_images[0] * 0], tile_size = tile_size) * len(ref_images)
            vae_feat_uncond = torch.cat( vae_feat_uncond, dim=1).unsqueeze(0)
        encode_fn = lambda prompts: self.text_encoder(prompts, self.device)
        context = self.text_encoder_cache.encode(encode_fn, [ref_prompt], device=self.device)[0].to(self.dtype)
        context = torch.cat([context, context.new_zeros(self.model.text_len -context.size(0), context.size(1)) ]).unsqueeze(0) 
        clear_caches()
        get_cache("lynx_ref_buffer").update({ 0: {}, 1: {} })
        loras_active_adapters = loras_scaling = None
        if not enable_loras and getattr(self.model, "_loras_scaling", None) is not None:
            loras_active_adapters, loras_scaling = self.model._loras_active_adapters, self.model._loras_scaling
            offload.activate_loras(self.model, loras_active_adapters, [0.0] * len(loras_active_adapters))
        ref_buffer = self.model(
            pipeline =self,
            x = [vae_feat, vae_feat_uncond] if any_guidance else [vae_feat],
            context = [context, context] if any_guidance else [context], 
            freqs= freqs,
            t=torch.zeros(1, dtype=torch.float, device=self.device),
            lynx_feature_extractor = True,
        )
        if loras_scaling is not None:
            offload.activate_loras(self.model, loras_active_adapters, [loras_scaling[adapter] for adapter in loras_active_adapters])

        clear_caches()
        return ref_buffer[0], (ref_buffer[1] if any_guidance else None)

    def _build_mocha_latents(self, source_video, mask_tensor, ref_images, frame_num, lat_frames, lat_h, lat_w, tile_size):
        video = source_video.to(device=self.device, dtype=self.VAE_dtype)
        source_latents = self.vae.encode([video], tile_size=tile_size)[0].unsqueeze(0).to(self.dtype)
        mask = mask_tensor[:, :1].to(device=self.device, dtype=self.dtype)
        mask_latents = F.interpolate(mask, size=(lat_h, lat_w), mode="nearest").unsqueeze(2).repeat(1, self.vae.model.z_dim, 1, 1, 1)

        ref_latents = [self.vae.encode([convert_image_to_tensor(img).unsqueeze(1).to(device=self.device, dtype=self.VAE_dtype)], tile_size=tile_size)[0].unsqueeze(0).to(self.dtype) for img in ref_images[:2]]
        ref_latents = torch.cat(ref_latents, dim=2)

        mocha_latents = torch.cat([source_latents, mask_latents, ref_latents], dim=2)

        base_len, source_len, mask_len = lat_frames, source_latents.shape[2], mask_latents.shape[2]
        cos_parts, sin_parts = [], []

        def append_freq(start_t, length, h_offset=1, w_offset=1):
            cos, sin = get_nd_rotary_pos_embed( (start_t, h_offset, w_offset), (start_t + length, h_offset + lat_h // 2, w_offset + lat_w // 2), device=self.device)
            cos_parts.append(cos)
            sin_parts.append(sin)
            
        append_freq(1, base_len)
        append_freq(1, source_len)
        append_freq(1, mask_len)
        append_freq(0, 1)
        if ref_latents.shape[2] > 1: append_freq(0, 1, 1 + lat_h // 2, 1 + lat_w // 2)

        return mocha_latents, (torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0))

    @generation_progress
    def generate(self,
        input_prompt,
        alt_prompt="",
        input_frames= None,
        input_frames2= None,
        input_masks = None,
        input_masks2 = None,
        input_ref_images = None,
        input_ref_masks = None,
        input_faces = None,
        input_video = None,
        image_start = None,
        image_end = None,
        input_custom = None,
        denoising_strength = 1.0,
        masking_strength = 1.0,
        target_camera=None,                  
        context_scale=None,
        width = 1280,
        height = 720,
        fit_into_canvas = True,
        frame_num=81,
        batch_size = 1,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=50,
        guide_scale=5.0,
        guide2_scale = 5.0,
        guide3_scale = 5.0,
        switch_threshold = 0,
        switch2_threshold = 0,
        guide_phases= 1 ,
        model_switch_phase = 1,
        n_prompt="",
        seed=-1,
        callback = None,
        enable_RIFLEx = None,
        VAE_tile_size = 0,
        joint_pass = False,
        perturbation_layers = None,
        perturbation_start = 0.0,
        perturbation_end = 1.0,
        cfg_star_switch = True,
        cfg_zero_step = 5,
        audio_scale=None,
        audio_cfg_scale=None,
        audio_proj=None,
        audio_context_lens=None,
        alt_guide_scale = 1.0,
        overlapped_latents  = None,
        return_latent_slice = None,
        overlap_noise = 0,
        overlap_size = 0,
        sub_parallel_window_size=0,
        sub_parallel_window_overlap=0,
        conditioning_latents_size = 0,
        keep_frames_parsed = [],
        model_type = None,
        model_mode = None,
        loras_slists = None,
        NAG_scale = 0,
        NAG_tau = 3.5,
        NAG_alpha = 0.5,
        offloadobj = None,
        apg_switch = False,
        speakers_bboxes = None,
        color_correction_strength = 1,
        prefix_frames_count = 0,
        image_mode = 0,
        window_no = 0,
        set_header_text = None,
        pre_video_frame = None,
        prefix_video = None,
        video_prompt_type= "",
        original_input_ref_images = [],
        face_arc_embeds = None,
        control_scale_alt = 1.,
        motion_amplitude = 1.,
        window_start_frame_no = 0,
        self_refiner_setting=0,
        self_refiner_plan="",
        self_refiner_f_uncertainty = 0.0,
        self_refiner_certain_percentage = 0.999,
        custom_settings=None,
        custom_sigmas=None,
        save_masks=False,
        vae_upsampler=None,
        set_progress_status=None,
        fps=16,
        **bbargs
                ):
        
        model_def = self.model_def
        if sample_solver =="euler":
            sample_scheduler = EulerScheduler(
                num_train_timesteps=self.num_timesteps,
                use_timestep_transform=self.use_timestep_transform,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'causvid':
            sample_scheduler = FlowMatchScheduler(num_inference_steps=sampling_steps, shift=shift, sigma_min=0, extra_one_step=True)
            timesteps = torch.tensor([1000, 934, 862, 756, 603, 410, 250, 140, 74])[:sampling_steps].to(self.device)
            sample_scheduler.timesteps =timesteps
            sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.], device=self.device)])
        elif sample_solver == 'unipc' or sample_solver == "":
            sample_scheduler = FlowUniPCMultistepScheduler( num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps( sampling_steps, device=self.device, shift=shift)
            
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        elif sample_solver == 'lcm':
            # LCM + LTX scheduler: Latent Consistency Model with RectifiedFlow
            # Optimized for Lightning LoRAs with ultra-fast 2-8 step inference
            effective_steps = min(sampling_steps, 8)  # LCM works best with few steps
            sample_scheduler = LCMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                num_inference_steps=effective_steps,
                shift=shift
            )
            sample_scheduler.set_timesteps(effective_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        else:
            raise NotImplementedError(f"Unsupported Scheduler {sample_solver}")
        if custom_sigmas and any(schedule is not None for schedule in custom_sigmas):
            default_sigmas = (sample_scheduler.sigmas[:-1].tolist() if sample_solver != "euler"
                              else (timesteps / self.num_train_timesteps).tolist())
            thresholds = [1.0] + [float(value) / self.num_train_timesteps for value in (switch_threshold, switch2_threshold)[:len(custom_sigmas) - 1]] + [0.0]
            combined_sigmas = []
            for phase, schedule in enumerate(custom_sigmas):
                values = schedule if schedule is not None else [sigma for sigma in default_sigmas if thresholds[phase + 1] < sigma <= thresholds[phase]]
                if not values:
                    raise ValueError(f"Custom Sigmas phase {phase + 1} has no steps in the automatic schedule; increase Steps, adjust the phase switches, or provide that phase's sigmas.")
                combined_sigmas.extend(values)
            if sample_solver == "euler":
                sample_scheduler.num_inference_steps = len(combined_sigmas)
            else:
                # set_timesteps resets multistep history. These sigmas already include
                # any desired shift; passing shift=1 avoids applying it a second time.
                sample_scheduler.set_timesteps(sigmas=np.asarray(combined_sigmas, dtype=np.float32), device=self.device, shift=1.0)
            # Scale before rounding to FP32, just like the phase-switch controls.
            # Rounding sigmas first can move a boundary one step into the next phase.
            timesteps = torch.tensor([sigma * self.num_train_timesteps for sigma in combined_sigmas], dtype=torch.float32, device=self.device)
            sample_scheduler.timesteps = timesteps
            sampling_steps = len(combined_sigmas)
        original_timesteps = timesteps

        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        image_outputs = image_mode == 1
        kwargs = {'pipeline': self, 'callback': callback}
        color_reference_frame = None
        if self._interrupt:
            return None
        # Text Encoder
        kiwi_edit = model_type in ["kiwi_edit"]
        animate2 = model_def.get("animate2", False)
        animate2_kv_cache = custom_settings.get("animate2_kv_cache", "Disabled") if animate2 and isinstance(custom_settings, dict) else "Disabled"
        if animate2_kv_cache == "Enabled": animate2_kv_cache = "GPU"
        bernini = model_def.get("bernini_class", False)
        shotplan = model_def.get("shotplan", False)
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        text_len = self.model.text_len
        bernini_omega_v = context_scale[0] if bernini and context_scale is not None and len(context_scale) > 0 else 1.0
        bernini_omega_i = alt_guide_scale
        any_guidance_at_all = guide_scale > 1 or guide2_scale > 1 and guide_phases >=2 or guide3_scale > 1 and guide_phases >=3
        if bernini:
            any_guidance_at_all = any_guidance_at_all or "I" in video_prompt_type and bernini_omega_i != 1 or "V" in video_prompt_type and "I" in video_prompt_type and bernini_omega_v != 1
        context_null = context = None
        if input_video is not None: height, width = input_video.shape[-2:]

        if shotplan:
            shotplan_prompt = compile_shotplan_prompt(input_prompt, frame_num, fps)
            input_prompt = shotplan_prompt.prompt

        if kiwi_edit:
            from .kiwi.embedders import build_kiwi_conditions
            kiwi_ref_images = original_input_ref_images[0] if original_input_ref_images is not None and len(original_input_ref_images) else None
            kiwi_state = build_kiwi_conditions(vae=self.vae, source_frames=input_frames, ref_images=kiwi_ref_images, width=width, height=height, batch_size=batch_size, device=self.device, dtype=self.dtype, source_embedder_file=self.kiwi_source_embedder_file, ref_embedder_file=self.kiwi_ref_embedder_file, vae_tile_size=VAE_tile_size)
            # the Qwen2.5-VL processor and vision code create their tensors on the default device
            with text_encoding_prompts(2 if any_guidance_at_all or NAG_scale > 1 else 1), torch.device(self.device):
                context = self.kiwi_mllm.encode_from_inputs(input_prompt, input_frames, kiwi_ref_images, use_ref_image=self.kiwi_ref_embedder_file is not None, max_frames=16)
                context = [context]
                if any_guidance_at_all or NAG_scale > 1:
                    context_null = self.kiwi_mllm.encode_from_inputs(n_prompt, input_frames, kiwi_ref_images, use_ref_image=self.kiwi_ref_embedder_file is not None, max_frames=16)
                    context_null = [context_null]
        else:
            text_len = self.model.text_len
            encode_fn = lambda prompts: self.text_encoder(prompts, self.device)
            prompts = [input_prompt]
            if NAG_scale > 1 or any_guidance_at_all:
                prompts.append(n_prompt)
            if animate2:
                prompts.append(alt_prompt.strip() or model_def["animate2_ref_prompt"])
            contexts = iter(self.text_encoder_cache.encode(encode_fn, prompts, device=self.device))
            context = next(contexts).to(self.dtype)
            context = torch.cat([context, context.new_zeros(text_len -context.size(0), context.size(1)) ]).unsqueeze(0)
            if NAG_scale > 1 or any_guidance_at_all:      
                context_null = next(contexts).to(self.dtype)
                context_null = torch.cat([context_null, context_null.new_zeros(text_len -context_null.size(0), context_null.size(1)) ]).unsqueeze(0)
            if animate2:
                animate2_ref_context = next(contexts).to(self.dtype)
                animate2_ref_context = torch.cat([animate2_ref_context, animate2_ref_context.new_zeros(text_len - animate2_ref_context.size(0), animate2_ref_context.size(1))]).unsqueeze(0)
            del contexts

        if set_progress_status is not None:
            set_progress_status("Preparing Image and Video Conditioning")

        # NAG_prompt =  "static, low resolution, blurry"
        # context_NAG = self.text_encoder([NAG_prompt], self.device)[0]
        # context_NAG = context_NAG.to(self.dtype)
        # context_NAG = torch.cat([context_NAG, context_NAG.new_zeros(text_len -context_NAG.size(0), context_NAG.size(1)) ]).unsqueeze(0) 
        
        from mmgp import offload
        offloadobj.unload_all()

        offload.shared_state.update({"_nag_scale" : NAG_scale, "_nag_tau" : NAG_tau, "_nag_alpha":  NAG_alpha })
        if NAG_scale > 1: context = torch.cat([context, context_null], dim=0)
        # if NAG_scale > 1: context = torch.cat([context, context_NAG], dim=0)
        if self._interrupt: return None
        vace = model_def.get("vace_class", False)
        svi_dance = model_def.get("svi_dance", False)
        phantom = model_type in ["phantom_1.3B", "phantom_14B"]
        fantasy = model_type in ["fantasy"]
        multitalk =  model_def.get("multitalk_class", False)
        infinitetalk = model_type in ["infinitetalk"]
        standin = model_def.get("standin_class", False)
        lynx = model_def.get("lynx_class", False)
        recam = model_type in ["recam_1.3B"]
        ti2v = model_def.get("wan_5B_class", False)
        alpha_class = model_def.get("alpha_class", False)
        alpha2 = model_type in ["alpha2"]
        lucy_edit=  model_type in ["lucy_edit"]
        animate=  model_type in ["animate"]
        chrono_edit = model_type in ["chrono_edit"]
        mocha = model_type in ["mocha"]
        steadydancer = model_type in ["steadydancer"]
        wanmove = model_type in ["wanmove"]
        scail = model_type in ["scail"] 
        scail2 = model_def.get("scail2", False) or model_type in ["scail2_14B", "scail2_1.3B"]
        vista4d = model_type in ["vista4d"]
        fun_camera, uni3c = model_def.get("fun_camera", False), model_def.get("uni3c", False)
        svi_pro = model_def.get("svi2pro", False)
        svi_mode = 2 if svi_pro  else 0 
        svi_ref_pad_num = 0
        start_step_no = 0
        ref_images_count = inner_latent_frames = 0
        trim_frames = 0
        post_decode_pre_trim = 0
        last_latent_preview = False
        extended_overlapped_latents = clip_image_start = clip_image_end = animate2_clip_image_ref = image_mask_latents = latent_slice = freqs = post_freqs = None
        use_extended_overlapped_latents = True
        # SCAIL uses a fixed ref latent frame that should not be noised.
        no_noise_latents_injection = infinitetalk or scail or scail2
        timestep_injection = False
        ps_t, ps_h, ps_w = self.model.patch_size

        lat_frames = int((frame_num - 1) // self.vae_stride[0]) + 1
        extended_input_dim = 0
        ref_images_before = False            
        # image2video 
        if model_def.get("i2v_class", False) and not (animate or animate2 or scail or scail2 or fun_camera):
            any_end_frame = False
            if infinitetalk:
                new_shot = "0" in video_prompt_type
                if input_frames is not None:
                    image_ref = input_frames[:, 0]
                else:
                    if input_ref_images is None:                        
                        if pre_video_frame is None: raise Exception("Missing Reference Image")
                        input_ref_images, new_shot = [pre_video_frame], False
                    new_shot = new_shot and window_no <= len(input_ref_images)
                    image_ref = convert_image_to_tensor(input_ref_images[ min(window_no, len(input_ref_images))-1 ])
                if new_shot or input_video is None:  
                    input_video = image_ref.unsqueeze(1)
                else:
                    color_correction_strength = 0 #disable color correction as transition frames between shots may have a complete different color level than the colors of the new shot
            if input_video is None: 
                input_video = torch.full((3, 1, height, width), -1)
                color_correction_strength = 0
                                                                                                  
            _ , preframes_count, height, width = input_video.shape
            input_video = input_video.to(device=self.device).to(dtype= self.VAE_dtype)
            if infinitetalk:
                image_start = image_ref.to(input_video)
                control_pre_frames_count = 1 
                control_video = image_start.unsqueeze(1)
            else:
                image_start = input_video[:, -1]
                control_pre_frames_count = preframes_count
                control_video = input_video

            color_reference_frame = image_start.unsqueeze(1).clone()

            any_end_frame = image_end is not None 
            add_frames_for_end_image = any_end_frame and model_type == "i2v"
            if any_end_frame:
                color_correction_strength = 0 #disable color correction as transition frames between shots may have a complete different color level than the colors of the new shot
                if add_frames_for_end_image:
                    frame_num +=1
                    lat_frames = int((frame_num - 2) // self.vae_stride[0] + 2)
                    trim_frames = 1

            lat_h, lat_w = height // self.vae_stride[1], width // self.vae_stride[2]

            if image_end is not None:
                img_end_frame = image_end.unsqueeze(1).to(self.device)
            clip_image_start, clip_image_end = image_start, image_end

            remaining_frames = frame_num - control_pre_frames_count
            if any_end_frame and not svi_pro:
                enc= torch.concat([
                        control_video,
                        torch.zeros( (3, frame_num-control_pre_frames_count-1,  height, width), device=self.device, dtype= self.VAE_dtype),
                        img_end_frame,
                ], dim=1).to(self.device)
            else:
                if svi_pro or svi_mode and svi_ref_pad_num != 0:
                    use_extended_overlapped_latents = False
                    if input_ref_images is None or len(input_ref_images)==0:                        
                        if pre_video_frame is None: raise Exception("Missing Reference Image")
                        image_ref = pre_video_frame
                    else:
                        image_ref = input_ref_images[ min(window_no, len(input_ref_images))-1 ]
                    image_ref = convert_image_to_tensor(image_ref).unsqueeze(1).to(device=self.device, dtype=self.VAE_dtype)
                    if svi_pro:
                        if overlapped_latents is not None:
                            post_decode_pre_trim = 1
                        elif prefix_video is not None and prefix_video.shape[1] >= (5 + overlap_size):
                            overlapped_latents = self.vae.encode([torch.cat( [prefix_video[:, -(5 + overlap_size):]], dim=1)], VAE_tile_size)[0][:, -overlap_size//4: ].unsqueeze(0)
                            post_decode_pre_trim = 1
                            
                        image_ref_latents = self.vae.encode([image_ref], VAE_tile_size)[0]
                        pad_len = lat_frames + ref_images_count - image_ref_latents.shape[1] - (overlapped_latents.shape[2] if overlapped_latents is not None else 0)
                        pad_latents = torch.zeros(image_ref_latents.shape[0], pad_len, lat_h, lat_w, device=image_ref_latents.device, dtype=image_ref_latents.dtype)
                        if overlapped_latents is None:
                            lat_y = torch.concat([image_ref_latents, pad_latents], dim=1).to(self.device)
                        else:
                            lat_y = torch.concat([image_ref_latents, overlapped_latents.squeeze(0), pad_latents], dim=1).to(self.device)
                        if any_end_frame:
                            lat_y[:, -1:] = self.vae.encode([img_end_frame], VAE_tile_size)[0][:, -1:]
                        image_ref_latents = None
                    else:
                        svi_ref_pad_num = remaining_frames if svi_ref_pad_num == -1 else min(svi_ref_pad_num, remaining_frames)  
                        padded_frames = image_ref.expand(-1, svi_ref_pad_num, -1, -1)
                        if remaining_frames > svi_ref_pad_num:
                            padded_frames = torch.cat([padded_frames, torch.zeros((3, remaining_frames - svi_ref_pad_num, height, width), device=self.device, dtype=self.VAE_dtype)], dim=1)
                        enc = torch.concat([control_video, padded_frames], dim=1).to(self.device)
                else:
                    enc= torch.concat([ control_video, torch.zeros( (3, remaining_frames, height, width), device=self.device, dtype= self.VAE_dtype) ], dim=1).to(self.device)
                padded_frames = None

            if not svi_pro:
                lat_y = self.vae.encode([enc], VAE_tile_size, any_end_frame= any_end_frame and add_frames_for_end_image)[0]


            msk = torch.ones(1, frame_num + ref_images_count * 4, lat_h, lat_w, device=self.device)
            if any_end_frame:
                msk[:, (1 if svi_mode else control_pre_frames_count):-1] = 0
                if add_frames_for_end_image:
                    msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:-1], torch.repeat_interleave(msk[:, -1:], repeats=4, dim=1) ], dim=1)
                else:
                    msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:] ], dim=1)
            else:
                msk[:, 1 if svi_mode else control_pre_frames_count:] = 0
                msk = torch.concat([ torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:] ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]

            image_start = image_end = img_end_frame = image_ref = control_video = None

            if motion_amplitude > 1:
                base_latent = lat_y[:, :1]
                diff = lat_y[:, control_pre_frames_count:] - base_latent
                diff_mean = diff.mean(dim=(0, 2, 3), keepdim=True)
                diff_centered = diff - diff_mean
                scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                scaled_latent = torch.clamp(scaled_latent, -6, 6)
                if any_end_frame:
                    lat_y = torch.cat([lat_y[:, :control_pre_frames_count], scaled_latent[:, :-1], lat_y[:, -1:]], dim=1)
                else:
                    lat_y = torch.cat([lat_y[:, :control_pre_frames_count], scaled_latent], dim=1)
                base_latent = scaled_latent = diff_mean = diff = diff_centered = None
                
            y = torch.concat([msk, lat_y])
            overlapped_latents_frames_num = int(1 + (preframes_count-1) // 4)
            # if overlapped_latents != None:
            if overlapped_latents_frames_num > 0 and use_extended_overlapped_latents:
                # disabled because looks worse
                if False and overlapped_latents_frames_num > 1: lat_y[:, :, 1:overlapped_latents_frames_num]  = overlapped_latents[:, 1:]
                if infinitetalk:
                    lat_y = self.vae.encode([input_video], VAE_tile_size)[0]
                extended_overlapped_latents = lat_y[:, :overlapped_latents_frames_num].clone().unsqueeze(0)

            lat_y = None
            kwargs.update({ 'y': y})

        # Wan-Move
        if wanmove:
            track = np.load(input_custom)
            if track.ndim == 4: track = track.squeeze(0)
            if track.max() <= 1:
                track = np.round(track * [width, height]).astype(np.int64)
            control_video_pos= 0 if "T" in video_prompt_type else window_start_frame_no
            track = torch.from_numpy(track[control_video_pos:control_video_pos+frame_num]).to(self.device)
            track_feats, track_pos = create_pos_feature_map(track, None, [4, 8, 8], height, width, 16, device=y.device)
            track_feats = None #track_feats.permute(3, 0, 1, 2)
            y_cond = kwargs.pop("y")
            y_uncond = y_cond.clone()
            y_cond[4:20] = replace_feature(y[4:20].unsqueeze(0), track_pos.unsqueeze(0))[0]

        # Steady Dancer
        if steadydancer:
            condition_guide_scale = alt_guide_scale # 2.0
            # ref img_x
            ref_x = self.vae.encode([input_video[:, :1]], VAE_tile_size)[0]
            msk_ref = torch.ones(4, 1, lat_h, lat_w, device=self.device)
            ref_x = torch.concat([ref_x, msk_ref, ref_x])
            # ref img_c
            ref_c = self.vae.encode([input_frames[:, :1]], VAE_tile_size)[0]
            msk_c = torch.zeros(4, 1, lat_h, lat_w, device=self.device)
            ref_c = torch.concat([ref_c, msk_c, ref_c])
            kwargs.update({ 'steadydancer_ref_x': ref_x, 'steadydancer_ref_c': ref_c})
            # conditions, w/o msk
            with control_video_encoding("V" in video_prompt_type):
                conditions = self.vae.encode([input_frames])[0].unsqueeze(0)
            # conditions_null, w/o msk
            with control_video_encoding("V" in video_prompt_type):
                conditions_null = self.vae.encode([input_frames2])[0].unsqueeze(0)
            inner_latent_frames = 2

        # Chrono Edit
        if chrono_edit:
            if frame_num == 5:
                freq0, freq7 = get_nd_rotary_pos_embed( (0, 0, 0), (1, lat_h // 2, lat_w // 2), device=self.device), get_nd_rotary_pos_embed( (7, 0, 0), (8, lat_h // 2, lat_w // 2), device=self.device)
                freqs = ( torch.cat([freq0[0], freq7[0]]), torch.cat([freq0[1],freq7[1]]))
                freq0 = freq7 = None
            last_latent_preview = image_outputs

        # Animate
        if animate:
            pose_pixels = input_frames * input_masks
            input_masks = 1. - input_masks
            pose_pixels -= input_masks
            with control_video_encoding("V" in video_prompt_type):
                pose_latents = self.vae.encode([pose_pixels], VAE_tile_size)[0].unsqueeze(0)
            input_frames = input_frames * input_masks
            if not "X" in video_prompt_type: input_frames += input_masks - 1 # masked area should black (-1) in background frames
            # input_frames = input_frames[:, :1].expand(-1, input_frames.shape[1], -1, -1)
            if prefix_frames_count > 0:
                 input_frames[:, :prefix_frames_count] = input_video 
                 input_masks[:, :prefix_frames_count] = 1 
            # save_video(pose_pixels, "pose.mp4")
            # save_video(input_frames, "input_frames.mp4")
            # save_video(input_masks, "input_masks.mp4", value_range=(0,1))
            lat_h, lat_w = height // self.vae_stride[1], width // self.vae_stride[2]
            msk_ref = self.get_i2v_mask(lat_h, lat_w, nb_frames_unchanged=1,lat_t=1, device=self.device) 
            msk_control =  self.get_i2v_mask(lat_h, lat_w, nb_frames_unchanged=0, mask_pixel_values=input_masks, device=self.device)
            msk = torch.concat([msk_ref, msk_control], dim=1)
            image_ref = input_ref_images[0].to(self.device)
            clip_image_start = image_ref.squeeze(1)
            lat_y = self.vae.encode([image_ref], VAE_tile_size)[0]
            with control_video_encoding("V" in video_prompt_type):
                lat_y = torch.cat([lat_y, self.vae.encode([input_frames.to(self.device)], VAE_tile_size)[0]], dim=1)
            y = torch.concat([msk, lat_y])
            kwargs.update({ 'y': y, 'pose_latents': pose_latents})
            face_pixel_values = input_faces.unsqueeze(0)
            lat_y = msk = msk_control = msk_ref = pose_pixels = None
            ref_images_before = True
            ref_images_count = 1
            lat_frames = int((input_frames.shape[1] - 1) // self.vae_stride[0]) + 1

        # Animate 2
        if animate2:
            input_frames = guide_to_float(transfer.move(input_frames[:, :frame_num], self.device)).to(self.VAE_dtype)  # uint8 guides converted on the GPU
            if not input_ref_images:
                input_ref_images = [image_start if image_start is not None else convert_image_to_tensor(pre_video_frame)]
            image_ref = input_ref_images[0].to(device=self.device, dtype=self.VAE_dtype)
            image_ref = image_ref.unsqueeze(1) if image_ref.ndim == 3 else image_ref
            color_reference_frame = image_ref.clone()
            clip_image_start = image_ref[:, 0]
            animate2_clip_image_ref = input_frames[:, 0]
            with control_video_encoding("V" in video_prompt_type):
                driving_latents = self.vae.encode([input_frames], VAE_tile_size)[0]
            lat_frames, lat_h, lat_w = driving_latents.shape[1:]
            identity_latents = self.vae.encode([image_ref], VAE_tile_size)[0]
            output_pixels = torch.zeros_like(input_frames)
            if prefix_frames_count > 0: output_pixels[:, :prefix_frames_count] = input_video[:, :prefix_frames_count].to(output_pixels)
            output_latents = self.vae.encode([output_pixels], VAE_tile_size)[0]
            y = torch.cat([torch.cat([self.get_i2v_mask(lat_h, lat_w, 1, lat_t=1, device=self.device), self.get_i2v_mask(lat_h, lat_w, prefix_frames_count, lat_t=lat_frames, device=self.device)], dim=1), torch.cat([identity_latents, output_latents], dim=1)])
            grid_h, grid_w = lat_h // ps_h, lat_w // ps_w
            animate2_ref_freqs = get_nd_rotary_pos_embed((1, 0, grid_w), (1 + lat_frames, grid_h, 2 * grid_w), (lat_frames, grid_h, grid_w), L_test=lat_frames, enable_riflex=False, device=self.device)
            kwargs.update({"y": y, "animate2_ref_x": driving_latents.unsqueeze(0), "animate2_ref_y": torch.cat([self.get_i2v_mask(lat_h, lat_w, frame_num, lat_t=lat_frames, device=self.device), driving_latents]).unsqueeze(0), "animate2_ref_context": animate2_ref_context, "animate2_ref_freqs": animate2_ref_freqs, "animate2_log_scale": model_def["animate2_log_scale"], "animate2_kv_cache": animate2_kv_cache})
            ref_images_before, ref_images_count = True, 1
            identity_latents = output_latents = output_pixels = None

        # SCAIL - 3D pose-guided character animation
        if scail:
            pose_pixels = input_frames
            image_ref = input_ref_images[0].to(self.device) if input_ref_images is not None else convert_image_to_tensor(pre_video_frame).unsqueeze(1).to(self.device)
            insert_start_frames = window_start_frame_no + prefix_frames_count > 1
            if insert_start_frames:
                ref_latents = self.vae.encode([image_ref], VAE_tile_size)[0].unsqueeze(0)
                start_frames = input_video.to(self.device)
                color_reference_frame = input_video[:, :1].to(self.device)
                start_latents = self.vae.encode([start_frames], VAE_tile_size)[0].unsqueeze(0)
                extended_overlapped_latents = torch.cat([ref_latents, start_latents], dim=2)
                start_latents = None
            else:
                # sigma = torch.exp(torch.normal(mean=-2.5, std=0.5, size=(1,), device=self.device)).to(image_ref.dtype)
                sigma = torch.exp(torch.normal(mean=-5.0, std=0.5, size=(1,), device=self.device)).to(image_ref.dtype)
                noisy_ref = image_ref + torch.randn_like(image_ref) * sigma
                ref_latents = self.vae.encode([noisy_ref], VAE_tile_size)[0].unsqueeze(0)
                extended_overlapped_latents = ref_latents

            lat_h, lat_w = height // self.vae_stride[1], width // self.vae_stride[2]
            pose_frames = pose_pixels.shape[1]
            lat_t = int((pose_frames - 1) // self.vae_stride[0]) + 1
            msk_ref = self.get_i2v_mask(lat_h, lat_w, nb_frames_unchanged=1, lat_t=1, device=self.device)
            msk_control = self.get_i2v_mask(lat_h, lat_w, nb_frames_unchanged=prefix_frames_count if insert_start_frames else 0, lat_t=lat_t, device=self.device)
            y = torch.concat([msk_ref, msk_control], dim=1)
            # Downsample pose video by 0.5x before VAE encoding (matches `smpl_downsample` in upstream configs)
            pose_pixels_ds = pose_pixels.permute(1, 0, 2, 3)
            pose_pixels_ds = F.interpolate( pose_pixels_ds, size=(max(1, pose_pixels.shape[-2] // 2), max(1, pose_pixels.shape[-1] // 2)), mode="bilinear", align_corners=False, ).permute(1, 0, 2, 3)
            with control_video_encoding("V" in video_prompt_type):
                pose_latents = self.vae.encode([pose_pixels_ds], VAE_tile_size)[0].unsqueeze(0)

            clip_image_start = image_ref.squeeze(1)
            kwargs.update({"y": y, "scail_pose_latents": pose_latents, "ref_images_count": 1})

            pose_grid_t = pose_latents.shape[2] // ps_t
            pose_rope_h = lat_h // ps_h
            pose_rope_w = lat_w // ps_w
            pose_freqs_cos, pose_freqs_sin = get_nd_rotary_pos_embed( (ref_images_count, 0, 120), (ref_images_count + pose_grid_t, pose_rope_h, 120 + pose_rope_w), (pose_grid_t, pose_rope_h, pose_rope_w), L_test = lat_t, enable_riflex = enable_RIFLEx, device=self.device)

            head_dim = pose_freqs_cos.shape[1]
            pose_freqs_cos = pose_freqs_cos.view(pose_grid_t, pose_rope_h, pose_rope_w, head_dim).permute(0, 3, 1, 2)
            pose_freqs_sin = pose_freqs_sin.view(pose_grid_t, pose_rope_h, pose_rope_w, head_dim).permute(0, 3, 1, 2)

            pose_freqs_cos = F.avg_pool2d(pose_freqs_cos, kernel_size=2, stride=2).permute(0, 2, 3, 1).reshape(-1, head_dim)
            pose_freqs_sin = F.avg_pool2d(pose_freqs_sin, kernel_size=2, stride=2).permute(0, 2, 3, 1).reshape(-1, head_dim)
            post_freqs = (pose_freqs_cos, pose_freqs_sin)

            pose_pixels = pose_pixels_ds = pose_freqs_cos_full =  None
            ref_images_before = True
            ref_images_count = 1
            lat_frames = lat_t

        # SCAIL-2 - reference-driven character animation with mask-token conditioning
        if scail2:
            scail2_conditioning = prepare_scail2_conditioning(self, input_frames=input_frames, input_masks=input_masks, input_ref_images=input_ref_images, input_ref_masks=input_ref_masks, input_video=input_video, pre_video_frame=pre_video_frame, prefix_frames_count=prefix_frames_count, overlapped_latents=overlapped_latents, height=height, width=width, VAE_tile_size=VAE_tile_size, enable_RIFLEx=enable_RIFLEx, video_prompt_type=video_prompt_type, custom_settings=custom_settings, model_def=model_def, ps_t=ps_t, ps_h=ps_h, ps_w=ps_w, save_masks=save_masks)
            kwargs.update(scail2_conditioning["kwargs"])
            freqs = scail2_conditioning["freqs"]
            clip_image_start = scail2_conditioning["clip_image_start"]
            extended_overlapped_latents = scail2_conditioning["extended_overlapped_latents"]
            if scail2_conditioning["color_reference_frame"] is not None:
                color_reference_frame = scail2_conditioning["color_reference_frame"]
            ref_images_before = False
            ref_images_count = 0
            lat_frames = scail2_conditioning["lat_frames"]

        # Wan Fun Control-Camera: start image latent in the first latent frame (no i2v mask) and the camera path adapter features
        if fun_camera:
            from .camera_control.fun_camera import prepare_fun_camera
            lat_h, lat_w = height // self.vae_stride[1], width // self.vae_stride[2]
            y = torch.zeros(self.vae.model.z_dim, lat_frames, lat_h, lat_w, device=self.device)
            if input_video is not None:
                clip_image_start = input_video[:, -1].to(self.device)
                y[:, :1] = self.vae.encode([clip_image_start.unsqueeze(1).to(self.VAE_dtype)], VAE_tile_size)[0]
            else:
                kwargs["clip_fea"] = torch.zeros(1, 257, 1280, device=self.device)  # upstream zeroes the image features without start image
            kwargs.update({"y": y, "camera_adapter_features": prepare_fun_camera(self, model_mode, input_custom, custom_settings, lat_frames, height, width)})
            y = None

        # Clip image
        if hasattr(self, "clip") and clip_image_start is not None:                                   
            clip_image_size = self.clip.model.image_size
            clip_image_start = resize_lanczos(clip_image_start, clip_image_size, clip_image_size)
            clip_image_end = resize_lanczos(clip_image_end, clip_image_size, clip_image_size) if clip_image_end is not None else clip_image_start
            if model_type == "flf2v_720p":                    
                clip_context = self.clip.visual([clip_image_start[:, None, :, :], clip_image_end[:, None, :, :] if clip_image_end is not None else clip_image_start[:, None, :, :]])
            else:
                clip_context = self.clip.visual([clip_image_start[:, None, :, :]])
            clip_image_start = clip_image_end = None
            kwargs.update({'clip_fea': clip_context})
            if animate2:
                animate2_clip_image_ref = resize_lanczos(animate2_clip_image_ref, clip_image_size, clip_image_size)
                kwargs["animate2_ref_clip_fea"] = self.clip.visual([animate2_clip_image_ref[:, None, :, :]])
                animate2_clip_image_ref = None
            if steadydancer:
                kwargs['steadydancer_clip_fea_c'] = self.clip.visual([guide_to_float(input_frames[:, :1])])

        # Recam Master & Lucy Edit
        if recam or lucy_edit:
            frame_num, height,width = input_frames.shape[-3:]
            lat_frames = int((frame_num - 1) // self.vae_stride[0]) + 1
            frame_num = (lat_frames -1) * self.vae_stride[0] + 1
            input_frames = input_frames[:, :frame_num].to(dtype=self.dtype , device=self.device)
            with control_video_encoding("V" in video_prompt_type):
                extended_latents = self.vae.encode([input_frames])[0].unsqueeze(0) #.to(dtype=self.dtype, device=self.device)
            extended_input_dim = 2 if recam else 1
            del input_frames

        if recam:
            # Process target camera (recammaster)
            target_camera = model_mode
            from shared.utils.cammmaster_tools import get_camera_embedding
            cam_emb = get_camera_embedding(target_camera)       
            cam_emb = cam_emb.to(dtype=self.dtype, device=self.device)
            kwargs['cam_emb'] = cam_emb

        if vista4d:
            from .vista4d.preprocess import prepare_vista4d_condition
            kwargs.update(prepare_vista4d_condition(self, input_frames, input_custom, frame_num, height, width, VAE_tile_size, fps=bbargs.get("fps", model_def.get("fps", 16)), custom_settings=custom_settings, model_mode=model_mode))
            freqs = get_vista4d_rotary_pos_embed((lat_frames, height // self.vae_stride[1], width // self.vae_stride[2]), self.device)

        if uni3c: # the start image lifted to a point cloud and rendered along the camera path
            from .camera_control.uni3c import prepare_uni3c
            kwargs["uni3c_condition"] = prepare_uni3c(self, input_video[:, -1], model_mode, input_custom, custom_settings, lat_frames, height, width, VAE_tile_size)
            if kwargs["uni3c_condition"] is None: return None

        bernini_sources_by_key = {"": []}
        if bernini:
            bernini_video_latents, bernini_image_latents = [], []
            if input_frames is not None and "V" in video_prompt_type:
                height, width = input_frames.shape[-2:]
                color_reference_frame = input_frames[:, :1].clone()
                with control_video_encoding("V" in video_prompt_type):
                    video_latents = self.vae.encode([input_frames.to(self.device)], VAE_tile_size)[0].unsqueeze(0)
                bernini_video_latents = [video_latents]
                if prefix_frames_count > 0:
                    overlapped_latents_frames_num = int(1 + (prefix_frames_count - 1) // self.vae_stride[0])
                    extended_overlapped_latents = video_latents[:, :, :overlapped_latents_frames_num].clone()
            if input_ref_images is not None and "I" in video_prompt_type:
                bernini_image_latents = [self.vae.encode([u.to(self.device)], VAE_tile_size)[0].unsqueeze(0) for u in input_ref_images]
            bernini_v_sources = [(source_latent, source_id + 1) for source_id, source_latent in enumerate(bernini_video_latents)]
            bernini_i_sources = [(source_latent, source_id + 1) for source_id, source_latent in enumerate(bernini_image_latents)]
            bernini_vi_sources = bernini_v_sources + [(source_latent, len(bernini_v_sources) + source_id + 1) for source_id, source_latent in enumerate(bernini_image_latents)]
            bernini_sources_by_key.update({"V": bernini_v_sources, "I": bernini_i_sources, "VI": bernini_vi_sources})

        # Video 2 Video
        if "G" in video_prompt_type and input_frames != None:
            height, width = input_frames.shape[-2:]
            with control_video_encoding("V" in video_prompt_type):
                source_latents = self.vae.encode([input_frames])[0].unsqueeze(0)
            injection_denoising_step = 0
            inject_from_start = False
            if input_frames != None and denoising_strength < 1 :
                color_reference_frame = guide_to_float(input_frames[:, -1:]).clone()
                if prefix_frames_count > 0:
                    overlapped_frames_num = prefix_frames_count
                    overlapped_latents_frames_num = (overlapped_frames_num -1 // 4) + 1 
                    # overlapped_latents_frames_num = overlapped_latents.shape[2]
                    # overlapped_frames_num = (overlapped_latents_frames_num-1) * 4 + 1
                else: 
                    overlapped_latents_frames_num = overlapped_frames_num  = 0
                if len(keep_frames_parsed) == 0  or image_outputs or  (overlapped_frames_num + len(keep_frames_parsed)) == input_frames.shape[1] and all(keep_frames_parsed) : keep_frames_parsed = [] 
                injection_denoising_step = int( round(sampling_steps * (1. - denoising_strength),4) )
                latent_keep_frames = []
                if source_latents.shape[2] < lat_frames or len(keep_frames_parsed) > 0:
                    inject_from_start = True
                    if len(keep_frames_parsed) >0 :
                        if overlapped_frames_num > 0: keep_frames_parsed = [True] * overlapped_frames_num + keep_frames_parsed
                        latent_keep_frames =[keep_frames_parsed[0]]
                        for i in range(1, len(keep_frames_parsed), 4):
                            latent_keep_frames.append(all(keep_frames_parsed[i:i+4]))
                else:
                    timesteps = timesteps[injection_denoising_step:]
                    start_step_no = injection_denoising_step
                    if hasattr(sample_scheduler, "timesteps"): sample_scheduler.timesteps = timesteps
                    if hasattr(sample_scheduler, "sigmas"): sample_scheduler.sigmas= sample_scheduler.sigmas[injection_denoising_step:]
                    injection_denoising_step = 0

            if input_masks is not None and not "U" in video_prompt_type:
                image_mask_latents = torch.nn.functional.interpolate(guide_mask_to_float(input_masks), size= source_latents.shape[-2:], mode="nearest").unsqueeze(0)
                if image_mask_latents.shape[2] !=1:
                    image_mask_latents = torch.cat([ image_mask_latents[:,:, :1], torch.nn.functional.interpolate(image_mask_latents, size= (source_latents.shape[-3]-1, *source_latents.shape[-2:]), mode="nearest") ], dim=2)
                image_mask_latents = torch.where(image_mask_latents>=0.5, 1., 0. )[:1].to(self.device)
                # save_video(image_mask_latents.squeeze(0), "mama.mp4", value_range=(0,1) )
                # image_mask_rebuilt = image_mask_latents.repeat_interleave(8, dim=-1).repeat_interleave(8, dim=-2).unsqueeze(0)
                masked_steps = math.ceil(sampling_steps * masking_strength)
        else:
            denoising_strength = 1
        # Phantom
        if phantom:
            lat_input_ref_images_neg = None
            if input_ref_images is not None: # Phantom Ref images
                lat_input_ref_images = self.get_vae_latents(input_ref_images, self.device)
                lat_input_ref_images_neg = torch.zeros_like(lat_input_ref_images)
                ref_images_count = trim_frames = lat_input_ref_images.shape[1]

        # Kiwi Edit
        if kiwi_edit:
            if kiwi_state["source_condition"] is not None:
                kwargs["kiwi_source_condition"] = kiwi_state["source_condition"]
            if kiwi_state["ref_condition"] is not None:
                kwargs["kiwi_ref_condition"] = kiwi_state["ref_condition"]
                kwargs["kiwi_ref_pad_first"] = self.model_def.get("kiwi_ref_pad_first", False) 
                inner_latent_frames = 1
            input_video = None

        if ti2v:
            if input_video is None:
                height, width = (height // 32) * 32, (width // 32) * 32 
            else:
                height, width = input_video.shape[-2:]
                source_latents = self.vae.encode([input_video], tile_size = VAE_tile_size)[0].unsqueeze(0)
                timestep_injection = True
                if extended_input_dim > 0:
                    extended_latents[:, :, :source_latents.shape[2]] = source_latents

        # Lynx
        if lynx :
            if original_input_ref_images is None or len(original_input_ref_images) == 0:
                lynx = False
            elif "K" in video_prompt_type and len(input_ref_images) <= 1:
                print("Warning: Missing Lynx Ref Image, make sure 'Inject only People / Objets' is selected or if there is 'Landscape and then People or Objects' there are at least two ref images (one Landscape image followed by face).")
                lynx = False
            else:
                from  .lynx.resampler import Resampler
                lynx_lite = model_type in ["lynx_lite", "vace_lynx_lite_14B"]
                ip_hidden_states = ip_hidden_states_uncond = None
                if True:
                    with torch.device("meta"):
                        arc_resampler = Resampler( depth=4, dim=1280, dim_head=64, embedding_dim=512, ff_mult=4, heads=20, num_queries=16, output_dim=2048 if lynx_lite else 5120 )
                    offload.load_model_data(arc_resampler, fl.locate_file("wan2.1_lynx_lite_arc_resampler.safetensors" if lynx_lite else "wan2.1_lynx_full_arc_resampler.safetensors"), writable_tensors=False, default_dtype=None)
                    arc_resampler.to(self.device)
                    arcface_embed = face_arc_embeds[None,None,:].to(device=self.device, dtype=torch.float) 
                    ip_hidden_states = arc_resampler(arcface_embed).to(self.dtype)
                    ip_hidden_states_uncond = arc_resampler(torch.zeros_like(arcface_embed)).to(self.dtype)
                arc_resampler = None
                if not lynx_lite:
                    image_ref = original_input_ref_images[-1]
                    from preprocessing.face_preprocessor  import FaceProcessor 
                    face_processor = FaceProcessor()
                    lynx_ref = face_processor.process(image_ref, resize_to = 256)
                    lynx_ref_buffer, lynx_ref_buffer_uncond = self.encode_reference_images([lynx_ref], tile_size=VAE_tile_size, any_guidance= any_guidance_at_all, enable_loras = False)
                    lynx_ref = None
                gc.collect()
                torch.cuda.empty_cache()
                kwargs["lynx_ip_scale"] = control_scale_alt
                kwargs["lynx_ref_scale"] = control_scale_alt

        #Standin
        if standin:
            from preprocessing.face_preprocessor  import FaceProcessor 
            standin_ref_pos = 1 if "K" in video_prompt_type else 0
            if len(original_input_ref_images) < standin_ref_pos + 1: 
                if "I" in video_prompt_type and vace:
                    print("Warning: Missing Standin ref image, make sure 'Inject only People / Objets' is selected or if there is 'Landscape and then People or Objects' there are at least two ref images.")
            else: 
                standin_ref_pos = -1
                image_ref = original_input_ref_images[standin_ref_pos]
                face_processor = FaceProcessor()
                standin_ref = face_processor.process(image_ref, remove_bg = vace)
                face_processor = None
                gc.collect()
                torch.cuda.empty_cache()
                standin_freqs = get_nd_rotary_pos_embed((-1, int(height/16), int(width/16) ), (-1, int(height/16 + standin_ref.height/16), int(width/16 + standin_ref.width/16) ), device=self.device) 
                standin_ref = self.vae.encode([ convert_image_to_tensor(standin_ref).unsqueeze(1) ], VAE_tile_size)[0].unsqueeze(0)
                kwargs.update({ "standin_freqs": standin_freqs, "standin_ref": standin_ref, }) 


        # Vace
        if vace :
            # vace context encode
            # uint8 guides and masks (model_def "uint8_guides") are converted on the GPU: a quarter of the transfer, no float copy in RAM
            input_frames = [guide_to_float(input_frames.to(self.device))] +([] if input_frames2 is None else [guide_to_float(input_frames2.to(self.device))])            
            input_masks = [guide_mask_to_float(input_masks.to(self.device))] + ([] if input_masks2 is None else [guide_mask_to_float(input_masks2.to(self.device))])
            if lynx and input_ref_images is not None:
                input_ref_images,input_ref_masks = input_ref_images[:-1], input_ref_masks[:-1]
            input_ref_images = None if input_ref_images is None else [ u.to(self.device) for u in input_ref_images]
            input_ref_masks = None if input_ref_masks is None else [ None if u is None else u.to(self.device) for u in input_ref_masks]
            ref_images_before = True
            with control_video_encoding("V" in video_prompt_type):
                z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks, tile_size = VAE_tile_size, overlapped_latents = overlapped_latents )
            m0 = self.vace_encode_masks(input_masks, input_ref_images)
            if input_ref_masks is not None and len(input_ref_masks) > 0 and input_ref_masks[0] is not None:
                color_reference_frame = input_ref_images[0].clone()
                zbg = self.vace_encode_frames( input_ref_images[:1] * len(input_frames), None, masks=input_ref_masks[0], tile_size = VAE_tile_size )
                mbg = self.vace_encode_masks(input_ref_masks[:1] * len(input_frames), None)
                for zz0, mm0, zzbg, mmbg in zip(z0, m0, zbg, mbg):
                    zz0[:, 0:1] = zzbg
                    mm0[:, 0:1] = mmbg
                zz0 = mm0 = zzbg = mmbg = None
            z = [torch.cat([zz, mm], dim=0) for zz, mm in zip(z0, m0)]
            ref_images_count = len(input_ref_images) if input_ref_images is not None and input_ref_images is not None else 0
            context_scale = context_scale if context_scale != None else [1.0] * len(z)
            kwargs.update({'vace_context' : z, 'vace_context_scale' : context_scale, "ref_images_count": ref_images_count })
            if overlapped_latents != None :
                overlapped_latents_size = overlapped_latents.shape[2]
                extended_overlapped_latents = z[0][:16, :overlapped_latents_size + ref_images_count].clone().unsqueeze(0)
            if prefix_frames_count > 0:
                color_reference_frame = input_frames[0][:, prefix_frames_count -1:prefix_frames_count].clone()
        lat_h, lat_w = height // self.vae_stride[1], width // self.vae_stride[2]

        # Mocha
        if mocha:
            with control_video_encoding("V" in video_prompt_type):
                extended_latents, freqs = self._build_mocha_latents( input_frames, input_masks,  input_ref_images[:2], frame_num, lat_frames, lat_h, lat_w, VAE_tile_size )
            extended_input_dim = 2

        # shotplan
        if shotplan:
            kwargs["shotplan_cut_frames"] = shotplan_prompt.cut_frames

        target_shape = (self.vae.model.z_dim, lat_frames + ref_images_count, lat_h, lat_w)

        if multitalk:
            if audio_proj is None:
                audio_proj = [ torch.zeros( (1, 1, 5, 12, 768 ), dtype=self.dtype, device=self.device), torch.zeros( (1, (frame_num - 1) // 4, 8, 12, 768 ), dtype=self.dtype, device=self.device) ] 
            from .multitalk.multitalk import get_target_masks
            audio_proj = [audio.to(self.dtype) for audio in audio_proj]
            human_no = len(audio_proj[0])
            token_ref_target_masks = get_target_masks(human_no, lat_h, lat_w, height, width, face_scale = 0.05, bbox = speakers_bboxes).to(device=self.device, dtype=self.dtype) if human_no > 1 else None

        if fantasy and audio_proj != None:
            kwargs.update({ "audio_proj": audio_proj.to(self.dtype), "audio_context_lens": audio_context_lens, }) 


        if self._interrupt:
            return None

        expand_shape = [batch_size] + [-1] * len(target_shape)
        # Ropes
        if freqs is not None:
            pass
        elif extended_input_dim>=2:
            shape = list(target_shape[1:])
            shape[extended_input_dim-2] *= 2
            freqs = get_rotary_pos_embed(shape, enable_RIFLEx= False, device=self.device) 
        else:
            freqs = get_rotary_pos_embed( (target_shape[1]+ inner_latent_frames ,) + target_shape[2:] , enable_RIFLEx= enable_RIFLEx, device=self.device) 

        if post_freqs is not None:
            freqs = ( torch.cat([freqs[0], post_freqs[0]]), torch.cat([freqs[1], post_freqs[1]]) )

        kwargs["freqs"] = freqs

        def _build_sub_parallel_windows(total, size, overlap):
            if size <= 0 or size >= total:
                return None
            overlap = min(max(0, overlap), size - 1)
            windows, step, start = [], size - overlap, 0
            while True:
                end = start + size
                if end >= total:
                    start = max(0, total - size)
                    if len(windows) == 0 or windows[-1][0] != start:
                        windows.append((start, total))
                    break
                windows.append((start, end))
                start += step
            return windows

        sub_parallel_window_size = max(0, int(sub_parallel_window_size or 0))
        sub_parallel_window_overlap = max(0, int(sub_parallel_window_overlap or 0))
        sub_parallel_window_latents = min(lat_frames, max(1, int((sub_parallel_window_size - 1) // self.vae_stride[0]) + 1)) if sub_parallel_window_size > 0 else 0
        sub_parallel_overlap_latents = max(0, int((sub_parallel_window_overlap - 1) // self.vae_stride[0]) + 1) if sub_parallel_window_overlap > 0 else 0
        if sub_parallel_window_latents > 0:
            sub_parallel_overlap_latents = min(sub_parallel_overlap_latents, sub_parallel_window_latents - 1)
        sub_parallel_windows = _build_sub_parallel_windows(lat_frames, sub_parallel_window_latents, sub_parallel_overlap_latents)
        sub_parallel_prefix_latents = ref_images_count if ref_images_before and ref_images_count > 0 else 0
        sub_parallel_history_latents = extended_overlapped_latents.shape[2] if scail2 and extended_overlapped_latents is not None else 0

        def _sub_parallel_select(tensor, dim, indices):
            return tensor.index_select(dim, indices.to(device=tensor.device))

        def _sub_parallel_narrow(tensor, dim, start, end):
            return tensor.narrow(dim, start, end - start)

        def _sub_parallel_output_indices(start, end, device):
            return torch.arange(start, end, device=device, dtype=torch.long)

        def _sub_parallel_model_indices(start, end, device):
            indices = torch.arange(sub_parallel_prefix_latents + start, sub_parallel_prefix_latents + end, device=device, dtype=torch.long)
            if sub_parallel_prefix_latents > 0:
                indices = torch.cat([torch.arange(sub_parallel_prefix_latents, device=device, dtype=torch.long), indices])
            return indices

        def _sub_parallel_slice_time(tensor, dim, start, end, include_prefix=False, history_latents=0):
            if not torch.is_tensor(tensor) or tensor.ndim <= dim:
                return tensor
            if include_prefix and tensor.shape[dim] == target_shape[1]:
                if sub_parallel_prefix_latents > 0:
                    sliced = torch.cat([tensor.narrow(dim, 0, sub_parallel_prefix_latents), _sub_parallel_narrow(tensor, dim, sub_parallel_prefix_latents + start, sub_parallel_prefix_latents + end)], dim=dim)
                else:
                    sliced = _sub_parallel_narrow(tensor, dim, start, end)
            elif tensor.shape[dim] == lat_frames:
                sliced = _sub_parallel_narrow(tensor, dim, start, end)
            else:
                sliced = tensor
            if history_latents > 0 and sliced is not tensor:
                history = tensor.narrow(dim, 0, history_latents).to(device=sliced.device, dtype=sliced.dtype)
                sliced = torch.cat([history, sliced], dim=dim)
            return sliced

        def _sub_parallel_token_indices(frames, tokens_per_frame, offset=0):
            token_offsets = torch.arange(tokens_per_frame, device=frames.device, dtype=torch.long)
            return (frames[:, None] * tokens_per_frame + token_offsets).reshape(-1) + offset

        def _sub_parallel_scail2_freqs(start, end, history_latents=0):
            ref_latent_count = kwargs["scail2_ref_latents"].shape[2]
            main_len = history_latents + end - start
            pose_len = main_len
            main_grid_h, main_grid_w = target_shape[2] // ps_h, target_shape[3] // ps_w
            if test_scail2_replace(video_prompt_type):
                ref_freqs_cos, ref_freqs_sin = get_nd_rotary_pos_embed((0, 120, 0), (ref_latent_count, 120 + main_grid_h, main_grid_w), (ref_latent_count, main_grid_h, main_grid_w), L_test=main_len, enable_riflex=False, device=self.device)
                video_freqs_cos, video_freqs_sin = get_nd_rotary_pos_embed((ref_latent_count - 1, 0, 0), (ref_latent_count - 1 + main_len, main_grid_h, main_grid_w), (main_len, main_grid_h, main_grid_w), L_test=main_len, enable_riflex=False, device=self.device)
                main_freqs_cos, main_freqs_sin = torch.cat([ref_freqs_cos, video_freqs_cos]), torch.cat([ref_freqs_sin, video_freqs_sin])
                pose_freqs_cos, pose_freqs_sin = get_nd_rotary_pos_embed((ref_latent_count - 1, 0, 120), (ref_latent_count - 1 + pose_len, main_grid_h, 120 + main_grid_w), (pose_len, main_grid_h, main_grid_w), L_test=main_len, enable_riflex=False, device=self.device)
            else:
                main_freqs_cos, main_freqs_sin = get_nd_rotary_pos_embed((0, 0, 0), (ref_latent_count + main_len, main_grid_h, main_grid_w), (ref_latent_count + main_len, main_grid_h, main_grid_w), L_test=main_len, enable_riflex=False, device=self.device)
                pose_freqs_cos, pose_freqs_sin = get_nd_rotary_pos_embed((ref_latent_count, 0, 120), (ref_latent_count + pose_len, main_grid_h, 120 + main_grid_w), (pose_len, main_grid_h, main_grid_w), L_test=main_len, enable_riflex=False, device=self.device)
            head_dim = pose_freqs_cos.shape[1]
            pose_freqs_cos = F.avg_pool2d(pose_freqs_cos.view(pose_len, main_grid_h, main_grid_w, head_dim).permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1).reshape(-1, head_dim)
            pose_freqs_sin = F.avg_pool2d(pose_freqs_sin.view(pose_len, main_grid_h, main_grid_w, head_dim).permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1).reshape(-1, head_dim)
            return torch.cat([main_freqs_cos, pose_freqs_cos]), torch.cat([main_freqs_sin, pose_freqs_sin])

        def _sub_parallel_slice_freqs(freqs, start, end, history_latents=0):
            if not isinstance(freqs, tuple):
                return freqs
            if scail2:
                return _sub_parallel_scail2_freqs(start, end, history_latents)
            cos, sin = freqs
            main_tokens = (target_shape[2] // ps_h) * (target_shape[3] // ps_w)
            if vista4d:
                main_count = lat_frames * main_tokens
                frames = _sub_parallel_output_indices(start, end, cos.device)
                indices = torch.cat([_sub_parallel_token_indices(frames, main_tokens, main_count * group) for group in range(3)])
            else:
                main_count = target_shape[1] * main_tokens
                indices = [_sub_parallel_token_indices(_sub_parallel_model_indices(start, end, cos.device), main_tokens)]
                pose_tokens = (target_shape[2] // ps_h // 2) * (target_shape[3] // ps_w // 2)
                if scail and cos.shape[0] > main_count and pose_tokens > 0:
                    indices.append(_sub_parallel_token_indices(_sub_parallel_output_indices(start, end, cos.device), pose_tokens, main_count))
                elif cos.shape[0] > main_count:
                    indices.append(torch.arange(main_count, cos.shape[0], device=cos.device, dtype=torch.long))
                indices = torch.cat(indices)
            return _sub_parallel_select(cos, 0, indices), _sub_parallel_select(sin, 0, indices)

        def _sub_parallel_kwargs(start, end, history_latents=0):
            sub_kwargs = {"freqs": _sub_parallel_slice_freqs(kwargs["freqs"], start, end, history_latents)}
            if "t" in kwargs:
                sub_kwargs["t"] = _sub_parallel_slice_time(kwargs["t"], 0, start, end, True, history_latents)
            if "y" in kwargs:
                sub_kwargs["y"] = _sub_parallel_slice_time(kwargs["y"], 1, start, end, True, history_latents)
            for key in ("pose_latents", "scail_pose_latents"):
                if key in kwargs:
                    sub_kwargs[key] = _sub_parallel_slice_time(kwargs[key], 2, start, end, history_latents=history_latents)
            for key in ("scail2_pose_latents", "scail2_driving_masks"):
                if key in kwargs:
                    sub_kwargs[key] = _sub_parallel_slice_time(kwargs[key], 2, start, end, history_latents=history_latents)
            if "scail2_ref_masks" in kwargs:
                ref_latent_count = kwargs["scail2_ref_latents"].shape[2]
                sub_kwargs["scail2_ref_masks"] = kwargs["scail2_ref_masks"][:, :, :ref_latent_count + history_latents + end - start]
            if "vace_context" in kwargs:
                sub_kwargs["vace_context"] = [_sub_parallel_slice_time(u, 1, start, end, True) for u in kwargs["vace_context"]]
            for key in ("kiwi_source_condition", "kiwi_ref_condition"):
                if key in kwargs:
                    sub_kwargs[key] = _sub_parallel_slice_time(kwargs[key], 2, start, end, True)
            if "vista" in kwargs:
                sub_kwargs["vista"] = {key: _sub_parallel_slice_time(value, 2, start, end) if torch.is_tensor(value) and value.ndim == 5 else value for key, value in kwargs["vista"].items()}
            return sub_kwargs

        def _sub_parallel_weight(start, end, dtype, device):
            weight = torch.ones(end - start, dtype=dtype, device=device)
            ramp = min(sub_parallel_overlap_latents, end - start)
            if ramp > 0 and start > 0:
                weight[:ramp] = torch.linspace(1e-6, 1, ramp, dtype=dtype, device=device)
            if ramp > 0 and end < lat_frames:
                weight[-ramp:] = torch.linspace(1, 1e-6, ramp, dtype=dtype, device=device)
            return weight.view(1, 1, -1, 1, 1)

        def _sub_parallel_denoise(latents, denoise_fn):
            nonlocal extended_latents, y_cond, y_uncond, conditions, conditions_null
            noise_pred = torch.zeros_like(latents)
            weight_sum = torch.zeros(1, 1, latents.shape[2], 1, 1, dtype=latents.dtype, device=latents.device)
            for start, end in sub_parallel_windows:
                anchor_latents = 1 if start > 0 else 0
                context_start = start - anchor_latents
                history_latents = sub_parallel_history_latents if context_start > 0 else 0
                latent_window = latents[:, :, sub_parallel_prefix_latents + context_start:sub_parallel_prefix_latents + end]
                if history_latents > 0:
                    latent_window = torch.cat([latents[:, :, :history_latents], latent_window], dim=2)
                if sub_parallel_prefix_latents > 0:
                    latent_window = torch.cat([latents[:, :, :sub_parallel_prefix_latents], latent_window], dim=2)
                saved_kwargs, saved_extended_latents = dict(kwargs), extended_latents if extended_input_dim > 0 else None
                if wanmove:
                    saved_y_cond, saved_y_uncond = y_cond, y_uncond
                if steadydancer:
                    saved_conditions, saved_conditions_null = conditions, conditions_null
                try:
                    kwargs.update(_sub_parallel_kwargs(context_start, end, history_latents))
                    if extended_input_dim > 0:
                        extended_latents = _sub_parallel_slice_time(extended_latents, 2, context_start, end, True)
                    if wanmove:
                        y_cond, y_uncond = _sub_parallel_slice_time(y_cond, 1, context_start, end), _sub_parallel_slice_time(y_uncond, 1, context_start, end)
                    if steadydancer:
                        conditions = [_sub_parallel_slice_time(u, 2, context_start, end) for u in conditions]
                        conditions_null = [_sub_parallel_slice_time(u, 2, context_start, end) for u in conditions_null]
                    pred = denoise_fn(latent_window)
                finally:
                    kwargs.clear()
                    kwargs.update(saved_kwargs)
                    if extended_input_dim > 0:
                        extended_latents = saved_extended_latents
                    if wanmove:
                        y_cond, y_uncond = saved_y_cond, saved_y_uncond
                    if steadydancer:
                        conditions, conditions_null = saved_conditions, saved_conditions_null
                if pred is None:
                    return None
                pred_start = sub_parallel_prefix_latents if sub_parallel_prefix_latents > 0 and pred.shape[2] == sub_parallel_prefix_latents + history_latents + end - context_start else 0
                if pred_start > 0:
                    noise_pred[:, :, :sub_parallel_prefix_latents] += pred[:, :, :sub_parallel_prefix_latents]
                    weight_sum[:, :, :sub_parallel_prefix_latents] += 1
                weight = _sub_parallel_weight(start, end, pred.dtype, pred.device)
                pred_start += history_latents + anchor_latents
                pred_slice = pred[:, :, pred_start:pred_start + end - start]
                pred_slice.mul_(weight)
                noise_pred[:, :, sub_parallel_prefix_latents + start:sub_parallel_prefix_latents + end] += pred_slice
                weight_sum[:, :, sub_parallel_prefix_latents + start:sub_parallel_prefix_latents + end] += weight
                pred = pred_slice = latent_window = weight = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            noise_pred.div_(weight_sum.clamp_min_(1e-6))
            return noise_pred


        # Steps Skipping
        sub_parallel_cache = sub_parallel_cache2 = None
        if sub_parallel_windows is not None:
            sub_parallel_cache = self.model.cache
            self.model.cache = None
            if self.model2 is not None:
                sub_parallel_cache2 = self.model2.cache
                self.model2.cache = None
        skip_steps_cache = self.model.cache
        if skip_steps_cache != None:
            cache_type = skip_steps_cache.cache_type
            x_count = 3 if phantom or fantasy or multitalk else 2
            skip_steps_cache.previous_residual = [None] * x_count
            if cache_type == "tea":
                self.model.compute_teacache_threshold(max(skip_steps_cache.start_step, start_step_no), original_timesteps, skip_steps_cache.multiplier)
            else: 
                self.model.compute_magcache_threshold(max(skip_steps_cache.start_step, start_step_no), original_timesteps, skip_steps_cache.multiplier)
                skip_steps_cache.accumulated_err, skip_steps_cache.accumulated_steps, skip_steps_cache.accumulated_ratio  = [0.0] * x_count, [0] * x_count, [1.0] * x_count
                skip_steps_cache.one_for_all = x_count > 2

        if callback != None:
            callback(-1, None, True)


        clear_caches()
        offload.shared_state["_chipmunk"] =  False
        chipmunk = offload.shared_state.get("_chipmunk", False)        
        if chipmunk:
            self.model.setup_chipmunk()

        offload.shared_state["_radial"] =  offload.shared_state["_attention"]=="radial"
        radial = offload.shared_state.get("_radial", False)        
        if radial and not animate2:
            radial_cache = get_cache("radial")
            from shared.radial_attention.attention import fill_radial_cache
            fill_radial_cache(radial_cache, len(self.model.blocks), *target_shape[1:])

        # init denoising
        updated_num_steps= len(timesteps)

        denoising_extra = ""
        from shared.utils.loras_mutipliers import update_loras_slists, get_model_switch_steps

        phase_switch_step, phase_switch_step2, phases_description = get_model_switch_steps(original_timesteps,guide_phases, 0 if self.model2 is None else model_switch_phase, switch_threshold, switch2_threshold )
        if len(phases_description) > 0:  set_header_text(phases_description)
        guidance_switch_done =  guidance_switch2_done = False
        if guide_phases > 1: denoising_extra = f"Phase 1/{guide_phases} High Noise" if self.model2 is not None else f"Phase 1/{guide_phases}"
        def update_guidance(step_no, t, guide_scale, new_guide_scale, guidance_switch_done, switch_threshold, trans, phase_no, denoising_extra):
            if guide_phases >= phase_no and not guidance_switch_done and t <= switch_threshold:
                if model_switch_phase == phase_no-1 and self.model2 is not None: trans = self.model2
                guide_scale, guidance_switch_done = new_guide_scale, True
                denoising_extra = f"Phase {phase_no}/{guide_phases} {'Low Noise' if trans == self.model2 else 'High Noise'}" if self.model2 is not None else f"Phase {phase_no}/{guide_phases}"
                callback(step_no-1, denoising_extra = denoising_extra)
            return guide_scale, guidance_switch_done, trans, denoising_extra
        update_loras_slists(self.model, loras_slists, len(original_timesteps), phase_switch_step= phase_switch_step, phase_switch_step2= phase_switch_step2)
        if self.model2 is not None: update_loras_slists(self.model2, loras_slists, len(original_timesteps), phase_switch_step= phase_switch_step, phase_switch_step2= phase_switch_step2)
        callback(-1, None, True, override_num_inference_steps = updated_num_steps, denoising_extra = denoising_extra)

        def clear():
            if sub_parallel_windows is not None:
                if sub_parallel_cache is not None:
                    sub_parallel_cache.previous_residual = None
                    sub_parallel_cache.previous_modulated_input = None
                self.model.cache = sub_parallel_cache
                if self.model2 is not None:
                    if sub_parallel_cache2 is not None:
                        sub_parallel_cache2.previous_residual = None
                        sub_parallel_cache2.previous_modulated_input = None
                    self.model2.cache = sub_parallel_cache2
            clear_caches()
            gc.collect()
            torch.cuda.empty_cache()
            return None

        if sample_scheduler != None:
            if isinstance(sample_scheduler, FlowMatchScheduler) or sample_solver == 'unipc_hf':
                scheduler_kwargs = {}
            else:
                scheduler_kwargs = {"generator": seed_g}
        # b, c, lat_f, lat_h, lat_w
        latents = torch.randn(batch_size, *target_shape, dtype=torch.float32, device=self.device, generator=seed_g)
        if alpha_class and alpha2:
            gauss_mask = load_gauss_mask(fl.locate_file("gauss_mask"))
            latents = apply_alpha_shift(latents, gauss_mask, 0.03)
        if "G" in video_prompt_type: randn = latents
        if custom_sigmas and custom_sigmas[0] is not None and "G" not in video_prompt_type:
            latents.mul_(original_timesteps[0] / self.num_train_timesteps)
        if apg_switch != 0:  
            apg_momentum = -0.75
            apg_norm_threshold = 55
            text_momentumbuffer  = MomentumBuffer(apg_momentum) 
            audio_momentumbuffer = MomentumBuffer(apg_momentum) 
        input_frames = input_frames2 = input_masks =input_masks2 = input_video = input_ref_images = input_ref_masks = pre_video_frame = None
        gc.collect()
        torch.cuda.empty_cache()
        # denoising
        trans = self.model
        if self_refiner_setting > 0:
            self_refiner_handler = create_self_refiner_handler(self_refiner_plan, self_refiner_f_uncertainty, self_refiner_setting, self_refiner_certain_percentage)
        else:
            self_refiner_handler = None

        for i, t in enumerate(tqdm(timesteps)):
            guide_scale, guidance_switch_done, trans, denoising_extra = update_guidance(i, t, guide_scale, guide2_scale, guidance_switch_done, switch_threshold, trans, 2, denoising_extra)
            guide_scale, guidance_switch2_done, trans, denoising_extra = update_guidance(i, t, guide_scale, guide3_scale, guidance_switch2_done, switch2_threshold, trans, 3, denoising_extra)
            offload.set_step_no_for_lora(trans, start_step_no + i)
            timestep = torch.stack([t])

            if timestep_injection:
                latents[:, :, :source_latents.shape[2]] = source_latents
                timestep = torch.full((target_shape[-3],), t, dtype=torch.int64, device=latents.device)
                timestep[:source_latents.shape[2]] = 0
                        
            kwargs.update({"t": timestep, "current_step_no": i, "real_step_no": start_step_no + i })  
            kwargs["perturbation_layers"] = perturbation_layers if int(perturbation_start * sampling_steps) <= i < int(perturbation_end * sampling_steps) else None

            if denoising_strength < 1 and i <= injection_denoising_step:
                sigma = t / 1000
                if inject_from_start:
                    noisy_image = latents.clone()
                    noisy_image[:,:, :source_latents.shape[2] ] = randn[:, :, :source_latents.shape[2] ] * sigma + (1 - sigma) * source_latents
                    for latent_no, keep_latent in enumerate(latent_keep_frames):
                        if not keep_latent:
                            noisy_image[:, :, latent_no:latent_no+1 ] = latents[:, :, latent_no:latent_no+1]
                    latents = noisy_image
                    noisy_image = None
                else:
                    latents[...] = randn * sigma + (1 - sigma) * source_latents

            if extended_overlapped_latents != None:
                if no_noise_latents_injection:
                    latents[:, :, :extended_overlapped_latents.shape[2]]   = extended_overlapped_latents 
                else:
                    latent_noise_factor = t / 1000
                    latents[:, :, :extended_overlapped_latents.shape[2]]   = extended_overlapped_latents  * (1.0 - latent_noise_factor) + torch.randn_like(extended_overlapped_latents ) * latent_noise_factor 
                if vace:
                    overlap_noise_factor = overlap_noise / 1000 
                    for zz in z:
                        zz[0:16, ref_images_count:extended_overlapped_latents.shape[2] ]   = extended_overlapped_latents[0, :, ref_images_count:]  * (1.0 - overlap_noise_factor) + torch.randn_like(extended_overlapped_latents[0, :, ref_images_count:] ) * overlap_noise_factor 

            def denoise_with_cfg_fn(latents):

                if extended_input_dim > 0:
                    latent_model_input = torch.cat([latents, extended_latents.expand(*expand_shape)], dim=extended_input_dim)
                else:
                    latent_model_input = latents

                any_guidance = guide_scale != 1
                if phantom:
                    gen_args = {
                        "x" : ([ torch.cat([latent_model_input[:,:, :-ref_images_count], lat_input_ref_images.unsqueeze(0).expand(*expand_shape)], dim=2) ] * 2 + 
                            [ torch.cat([latent_model_input[:,:, :-ref_images_count], lat_input_ref_images_neg.unsqueeze(0).expand(*expand_shape)], dim=2)]),
                        "context": [context, context_null, context_null] ,
                    }
                elif fantasy:
                    gen_args = {
                        "x" : [latent_model_input, latent_model_input, latent_model_input],
                        "context" : [context, context_null, context_null],
                        "audio_scale": [audio_scale, None, None ]
                    }
                elif animate:
                    gen_args = {
                        "x" : [latent_model_input, latent_model_input],
                        "context" : [context, context_null],
                        # "face_pixel_values": [face_pixel_values, None]
                        "face_pixel_values": [face_pixel_values, face_pixel_values] # seems to look better this way
                    }
                elif animate2:
                    gen_args = {
                        "x": [latent_model_input, latent_model_input],
                        "context": [context, context_null],
                    }
                elif wanmove:
                    gen_args = {
                        "x" : [latent_model_input, latent_model_input],
                        "context" : [context, context_null],
                        "y" : [y_cond, y_uncond],
                    }
                elif lynx:
                    gen_args = {
                        "x" : [latent_model_input, latent_model_input],
                        "context" : [context, context_null],
                        "lynx_ip_embeds": [ip_hidden_states, ip_hidden_states_uncond]
                    }
                    if model_type in ["lynx", "vace_lynx_14B"]:
                        gen_args["lynx_ref_buffer"] = [lynx_ref_buffer, lynx_ref_buffer_uncond]
                        
                elif steadydancer:
                    # DC-CFG: pose guidance only in [10%, 50%] of denoising steps
                    apply_cond_cfg = 0.1 <= i / sampling_steps < 0.5 and condition_guide_scale != 1
                    x_list, ctx_list, cond_list = [latent_model_input], [context], [conditions]
                    if guide_scale != 1:
                        x_list.append(latent_model_input); ctx_list.append(context_null); cond_list.append(conditions)
                    if apply_cond_cfg:
                        x_list.append(latent_model_input); ctx_list.append(context); cond_list.append(conditions_null)
                    gen_args = {"x": x_list, "context": ctx_list, "steadydancer_condition": cond_list}
                    any_guidance = len(x_list) > 1
                elif multitalk and audio_proj != None:
                    if guide_scale == 1:
                        gen_args = {
                            "x" : [latent_model_input, latent_model_input],
                            "context" : [context, context],
                            "multitalk_audio": [audio_proj, [torch.zeros_like(audio_proj[0][-1:]), torch.zeros_like(audio_proj[1][-1:])]],
                            "multitalk_masks": [token_ref_target_masks, None]
                        }
                        any_guidance = audio_cfg_scale != 1
                    else:
                        gen_args = {
                            "x" : [latent_model_input, latent_model_input, latent_model_input],
                            "context" : [context, context_null, context_null],
                            "multitalk_audio": [audio_proj, audio_proj, [torch.zeros_like(audio_proj[0][-1:]), torch.zeros_like(audio_proj[1][-1:])]],
                            "multitalk_masks": [token_ref_target_masks, token_ref_target_masks, None]
                        }
                elif kiwi_edit:
                    if guide_scale == 1:
                        any_guidance = False
                        gen_args = {"x": [latent_model_input], "context": context}
                    else:
                        gen_args = {"x": [latent_model_input, latent_model_input], "context": context + context_null}
                elif bernini:
                    omega_v, omega_i, omega_ti = bernini_omega_v, bernini_omega_i, guide_scale
                    if bernini_sources_by_key["V"] and bernini_sources_by_key["I"]:
                        branch_defs = [(1 - omega_v, "", context_null), (omega_v - omega_i, "V", context_null), (omega_i - omega_ti, "VI", context_null), (omega_ti, "VI", context)]
                    elif bernini_sources_by_key["V"]:
                        branch_defs = [(1 - omega_ti, "V", context_null), (omega_ti, "V", context)]
                    elif bernini_sources_by_key["I"]:
                        branch_defs = [(1 - omega_i, "", context_null), (omega_i - omega_ti, "I", context_null), (omega_ti, "I", context)]
                    else:
                        branch_defs = [(1 - omega_ti, "", context_null), (omega_ti, "", context)]
                    branch_defs = [branch_def for branch_def in branch_defs if branch_def[0] != 0]
                    bernini_coeffs = [branch_def[0] for branch_def in branch_defs]
                    gen_args = {
                        "x": [latent_model_input] * len(branch_defs),
                        "context": [branch_def[2] for branch_def in branch_defs],
                        "bernini_sources": [bernini_sources_by_key[branch_def[1]] for branch_def in branch_defs],
                    }
                    any_guidance = len(gen_args["x"]) > 1
                else:
                    gen_args = {
                        "x" : [latent_model_input, latent_model_input],
                        "context": [context, context_null]
                    }

                latent_model_input = None
                run_joint_pass = joint_pass or animate2 and any_guidance
                if run_joint_pass and any_guidance:
                    ret_values = trans( **gen_args , **kwargs)
                    if self._interrupt:
                        return clear()               
                else:
                    size = len(gen_args["x"]) if any_guidance else 1 
                    ret_values = [None] * size
                    for x_id in range(size):
                        sub_gen_args = {k : [v[x_id]] for k, v in gen_args.items() }
                        gen_args["x"][x_id] = None
                        ret_values[x_id] = trans( **sub_gen_args, x_id= x_id , **kwargs)[0]
                        if self._interrupt:
                            return clear()         
                    sub_gen_args = None
                gen_args = None
                if bernini:
                    noise_pred = ret_values[0] if bernini_coeffs[0] == 1 else ret_values[0] * bernini_coeffs[0]
                    for pred, coeff in zip(ret_values[1:], bernini_coeffs[1:]):
                        noise_pred = noise_pred + pred if coeff == 1 else noise_pred + pred * coeff
                elif not any_guidance:
                    noise_pred = ret_values[0]       
                elif phantom:
                    guide_scale_img= 5.0
                    guide_scale_text= guide_scale #7.5
                    pos_it, pos_i, neg = ret_values
                    noise_pred = neg + guide_scale_img * (pos_i - neg) + guide_scale_text * (pos_it - pos_i)
                    pos_it = pos_i = neg = None
                elif fantasy:
                    noise_pred_cond, noise_pred_noaudio, noise_pred_uncond = ret_values
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_noaudio - noise_pred_uncond) + audio_cfg_scale * (noise_pred_cond  - noise_pred_noaudio) 
                    noise_pred_noaudio = None
                elif steadydancer:
                    noise_pred_cond = ret_values[0]
                    if guide_scale == 1:  # only condition CFG (ret_values[1] = uncond_condition)
                        noise_pred = ret_values[1] + condition_guide_scale * (noise_pred_cond - ret_values[1])
                    else:  # text CFG + optionally condition CFG (ret_values[1] = uncond_context)
                        noise_pred = ret_values[1] + guide_scale * (noise_pred_cond - ret_values[1])
                        if apply_cond_cfg:
                            noise_pred = noise_pred + condition_guide_scale * (noise_pred_cond - ret_values[2])
                    noise_pred_cond = None

                elif multitalk and audio_proj != None:
                    if apg_switch != 0:
                        if guide_scale == 1:
                            noise_pred_cond, noise_pred_drop_audio  = ret_values
                            noise_pred = noise_pred_cond + (audio_cfg_scale - 1)* adaptive_projected_guidance(noise_pred_cond - noise_pred_drop_audio, 
                                                                                            noise_pred_cond, 
                                                                                            momentum_buffer=audio_momentumbuffer, 
                                                                                            norm_threshold=apg_norm_threshold)

                        else:
                            noise_pred_cond, noise_pred_drop_text, noise_pred_uncond = ret_values
                            noise_pred = noise_pred_cond + (guide_scale - 1) * adaptive_projected_guidance(noise_pred_cond - noise_pred_drop_text, 
                                                                                                                noise_pred_cond, 
                                                                                                                momentum_buffer=text_momentumbuffer, 
                                                                                                                norm_threshold=apg_norm_threshold) \
                                    + (audio_cfg_scale - 1) * adaptive_projected_guidance(noise_pred_drop_text - noise_pred_uncond, 
                                                                                            noise_pred_cond, 
                                                                                            momentum_buffer=audio_momentumbuffer, 
                                                                                            norm_threshold=apg_norm_threshold)
                    else:
                        if guide_scale == 1:
                            noise_pred_cond, noise_pred_drop_audio  = ret_values
                            noise_pred = noise_pred_drop_audio + audio_cfg_scale* (noise_pred_cond - noise_pred_drop_audio)  
                        else:
                            noise_pred_cond, noise_pred_drop_text, noise_pred_uncond = ret_values
                            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_drop_text) + audio_cfg_scale * (noise_pred_drop_text - noise_pred_uncond)  
                        noise_pred_uncond = noise_pred_cond = noise_pred_drop_text = noise_pred_drop_audio = None
                else:
                    noise_pred_cond, noise_pred_uncond = ret_values
                    if apg_switch != 0:
                        noise_pred = noise_pred_cond + (guide_scale - 1) * adaptive_projected_guidance(noise_pred_cond - noise_pred_uncond, 
                                                                                                            noise_pred_cond, 
                                                                                                            momentum_buffer=text_momentumbuffer, 
                                                                                                            norm_threshold=apg_norm_threshold)
                    else:
                        noise_pred_text = noise_pred_cond
                        if cfg_star_switch:
                            # CFG Zero *. Thanks to https://github.com/WeichenFan/CFG-Zero-star/
                            positive_flat = noise_pred_text.view(batch_size, -1)  
                            negative_flat = noise_pred_uncond.view(batch_size, -1)  

                            alpha = optimized_scale(positive_flat,negative_flat)
                            alpha = alpha.view(batch_size, 1, 1, 1)

                            if (i <= cfg_zero_step):
                                noise_pred = noise_pred_text*0. # it would be faster not to compute noise_pred...
                            else:
                                noise_pred_uncond *= alpha
                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_text - noise_pred_uncond)            
                ret_values = noise_pred_uncond = noise_pred_cond = noise_pred_text = neg  = None
                return noise_pred

            denoise_fn = partial(_sub_parallel_denoise, denoise_fn=denoise_with_cfg_fn) if sub_parallel_windows is not None else denoise_with_cfg_fn
            noise_pred = denoise_fn(latents)
            if noise_pred is None: return clear()
            if self_refiner_handler:
                latents, sample_scheduler = self_refiner_handler.step(i, latents, noise_pred, t, timesteps, target_shape, seed_g, sample_scheduler, scheduler_kwargs, denoise_fn)
                if latents is None: return clear()
            else:
                latents = sample_scheduler.step( noise_pred[:, :, :target_shape[1]], t, latents, **scheduler_kwargs)[0]
            noise_pred = denoise_fn = None


            if image_mask_latents is not None and i< masked_steps:
                sigma = 0 if i == len(timesteps)-1 else timesteps[i+1]/1000
                noisy_image = randn[:, :, :source_latents.shape[2]] * sigma + (1 - sigma) * source_latents
                latents[:, :, :source_latents.shape[2]] = noisy_image * (1-image_mask_latents) + image_mask_latents * latents[:, :, :source_latents.shape[2]]  


            if callback is not None:
                latents_preview = latents
                if ref_images_before and ref_images_count > 0: latents_preview = latents_preview[:, :, ref_images_count: ] 
                if trim_frames > 0:  latents_preview=  latents_preview[:, :,:-trim_frames]
                if image_outputs: latents_preview= latents_preview[:, :,-1:] if last_latent_preview else latents_preview[:, :,:1]
                if len(latents_preview) > 1: latents_preview = latents_preview.transpose(0,2)
                callback(i, latents_preview[0], False, denoising_extra =denoising_extra )
                latents_preview = None

        clear()
        if timestep_injection:
            latents[:, :, :source_latents.shape[2]] = source_latents
        if extended_overlapped_latents != None:
            latents[:, :, :extended_overlapped_latents.shape[2]]   = extended_overlapped_latents 

        if ref_images_before and ref_images_count > 0: latents = latents[:, :, ref_images_count:]
        if trim_frames > 0:  latents=  latents[:, :,:-trim_frames]
        if return_latent_slice != None:
            latent_slice = latents[:, :, return_latent_slice].clone()

        x0 =latents.unbind(dim=0)

        if chipmunk:
            self.model.release_chipmunk() # need to add it at every exit when in prod

        if chrono_edit:
            if frame_num == 5 :
                videos = self.vae.decode_to_cpu_uint8(x0, VAE_tile_size)
            else:
                videos_edit = self.vae.decode_to_cpu_uint8([x[:, [0,-1]] for x in x0 ], VAE_tile_size)
                videos = self.vae.decode_to_cpu_uint8([x[:, :-1] for x in x0 ], VAE_tile_size)
                videos = [ torch.cat([video, video_edit[:, 1:]], dim=1) for video, video_edit in zip(videos, videos_edit)]
            if image_outputs:
                return torch.cat([video[:,-1:] for video in videos], dim=1) if len(videos) > 1 else videos[0][:,-1:]
            else:
                return videos[0]
        if image_outputs:
            x0 = [x[:,:1] for x in x0 ]

        any_vae2= self.vae2 is not None
        needs_color_correction = color_correction_strength > 0 and (window_start_frame_no + prefix_frames_count) > 1
        videos = self.vae.decode_to_cpu_uint8(x0, VAE_tile_size)
        if any_vae2:
            videos2 = self.vae2.decode_to_cpu_uint8(x0, VAE_tile_size)

        if image_outputs:
            videos = torch.cat([video[:,:1] for video in videos], dim=1) if len(videos) > 1 else videos[0][:,:1]
            if any_vae2: videos2 = torch.cat([video[:,:1] for video in videos2], dim=1) if len(videos2) > 1 else videos2[0][:,:1]
            if vae_upsampler is not None:
                if callable(set_progress_status):
                    set_progress_status(f"{getattr(vae_upsampler, 'progress_label', 'VAE Spatial Upsampling')} in progress")
                lq_image_ref = [videos.transpose(0, 1).contiguous()]
                lq_latent_ref = [torch.stack([latent[:, 0] for latent in x0])]
                videos = vae_upsampler.decode_inputs(lq_image_ref, lq_latent_ref, prompt=input_prompt, seed=seed, abort_callback=lambda: self._interrupt, progress_callback=lambda *_args: None).transpose(0, 1).contiguous()
        else:
            videos = videos[0] # return only first video
            if any_vae2: videos2 = videos2[0] # return only first video
        if needs_color_correction:
            videos = videos.float().div_(127.5).sub_(1.0) if videos.dtype == torch.uint8 else videos
            if vace and False:
                # videos = match_and_blend_colors_with_mask(videos.unsqueeze(0), input_frames[0].unsqueeze(0), input_masks[0][:1].unsqueeze(0), color_correction_strength,copy_mode= "progressive_blend").squeeze(0)
                videos = match_and_blend_colors_with_mask(videos.unsqueeze(0), input_frames[0].unsqueeze(0), input_masks[0][:1].unsqueeze(0), color_correction_strength,copy_mode= "reference").squeeze(0)
                # videos = match_and_blend_colors_with_mask(videos.unsqueeze(0), videos.unsqueeze(0), input_masks[0][:1].unsqueeze(0), color_correction_strength,copy_mode= "reference").squeeze(0)
            elif color_reference_frame is not None:
                videos = match_and_blend_colors(videos.unsqueeze(0), color_reference_frame.unsqueeze(0), color_correction_strength).squeeze(0)
            videos = videos.clamp_(-1, 1).add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)

        ret = { "x" : videos, "latent_slice" : latent_slice}
        if post_decode_pre_trim > 0:
            ret["post_decode_pre_trim"] = post_decode_pre_trim

        if alpha_class:
            BGRA_frames = None
            from .alpha.utils import render_video, from_BRGA_numpy_to_RGBA_torch
            videos_for_alpha = videos.float().div_(127.5).sub_(1.0) if videos.dtype == torch.uint8 else videos
            videos2_for_alpha = videos2.float().div_(127.5).sub_(1.0) if videos2.dtype == torch.uint8 else videos2
            videos, BGRA_frames = render_video(videos_for_alpha[None], videos2_for_alpha[None])
            if image_outputs: 
                videos = from_BRGA_numpy_to_RGBA_torch(BGRA_frames) 
                BGRA_frames = None
            if videos.dtype != torch.uint8:
                videos = videos.clamp_(-1, 1).add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
            ret["x"] = videos
            if BGRA_frames is not None:
                from shared.utils.rgba_video import rgba_video_side_files
                output_format = sys.modules["wgp"].server_config.get("rgba_video_output", "png_zip")
                if callable(set_progress_status):
                    set_progress_status("Exporting RGBA Video")
                side_files = rgba_video_side_files(BGRA_frames, fps, output_format, interrupt_check=lambda: self._interrupt)
                if side_files is None:
                    return None
                ret["side_files"] = side_files
        return ret

    def get_loras_transformer(self, get_model_recursive_prop, base_model_type, model_type, video_prompt_type, model_mode, **kwargs):
        if base_model_type == "animate":
            if "#" in video_prompt_type and "1" in video_prompt_type:
                preloadURLs = get_model_recursive_prop(model_type,  "preload_URLs")
                if len(preloadURLs) > 0: 
                    return [os.path.abspath(fl.locate_file(os.path.basename(preloadURLs[0])))] , [1]
        elif base_model_type == "vace_ditto_14B":
            preloadURLs = get_model_recursive_prop(model_type,  "preload_URLs")
            model_mode = int(model_mode)
            if len(preloadURLs) > model_mode: 
                return [os.path.abspath(fl.locate_file(os.path.basename(preloadURLs[model_mode])))] , [1]
        return [], []
