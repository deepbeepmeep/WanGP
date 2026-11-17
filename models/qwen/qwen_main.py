
from shared.utils.phase_progress import generation_progress
from mmgp import offload, runtime

import torch, json, os
import math

from .transformer_qwenimage import QwenImageTransformer2DModel

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, Qwen2VLProcessor
from .autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from .vae_variants import load_vae
from diffusers import FlowMatchEulerDiscreteScheduler
from .pipeline_qwenimage import QwenImagePipeline
from PIL import Image
from shared.utils.utils import calculate_new_dimensions, convert_tensor_to_image
from shared.utils import files_locator as fl 

_QWEN_FUSED_SPLIT_MAP = {
    "attn.to_qkv": {"mapped_modules": ("attn.to_q", "attn.to_k", "attn.to_v")},
    "attn.add_qkv_proj": {"mapped_modules": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")},
}


def stitch_images(img1, img2):
    # Resize img2 to match img1's height
    width1, height1 = img1.size
    width2, height2 = img2.size
    new_width2 = int(width2 * height1 / height2)
    img2_resized = img2.resize((new_width2, height1), Image.Resampling.LANCZOS)
    
    stitched = Image.new('RGB', (width1 + new_width2, height1))
    stitched.paste(img1, (0, 0))
    stitched.paste(img2_resized, (width1, 0))
    return stitched

class model_factory():
    def __init__(
        self,
        checkpoint_dir,
        model_filename = None,
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
    

        transformer_filename = model_filename[0]
        processor = None
        tokenizer = None
        text_encoder_folder = model_def.get("text_encoder_folder")
        if text_encoder_folder:
            tokenizer_path = fl.locate_folder(text_encoder_folder)
        else:
            tokenizer_path = os.path.dirname(text_encoder_filename)
        if base_model_type in ["qwen_image_edit_20B", "qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B", "qwen_image_layered_20B"]:
            processor = Qwen2VLProcessor.from_pretrained(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.base_model_type = base_model_type

        if base_model_type == "qwen_image_layered_20B":
            base_config_file = "models/qwen/configs/qwen_image_layered_20B.json"
        elif base_model_type == "qwen_image_edit_plus2_20B":
            base_config_file = "models/qwen/configs/qwen_image_edit_plus2_20B.json"
        else:
            base_config_file = "models/qwen/configs/qwen_image_20B.json"
        with open(base_config_file, 'r', encoding='utf-8') as f:
            transformer_config = json.load(f)
        transformer_config.pop("_diffusers_version", None)
        transformer_config.pop("_class_name", None)
        transformer_config.pop("pooled_projection_dim", None)
        
        from accelerate import init_empty_weights
        with init_empty_weights():
            transformer = QwenImageTransformer2DModel(**transformer_config)
        source =  model_def.get("source", None)

        if source is not None:
            offload.load_model_data(transformer, source, fused_split_map=_QWEN_FUSED_SPLIT_MAP, writable_tensors=False, default_dtype=torch.bfloat16)
        else:
            offload.load_model_data(transformer, transformer_filename, fused_split_map=_QWEN_FUSED_SPLIT_MAP, writable_tensors=False, default_dtype=torch.bfloat16)
        # transformer = offload.fast_load_transformers_model("transformer_quanto.safetensors", writable_tensors= True , modelClass=QwenImageTransformer2DModel, defaultConfigPath="transformer_config.json")

        if not source is None:
            from wgp import save_model
            save_model(transformer, model_type, dtype, None)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename[0], dtype, base_config_file)

        from shared.kernels.int8_backend import require_blockwise_int8
        # All Qwen 20B variants share these timestep shifts, scales and gates.
        # Protect them from row-wise activation quantization; other linears
        # can still use Kitchen. The separate Qwen 2.1 loader is unaffected.
        require_blockwise_int8(transformer, "Qwen Image 20B Modulation", ("transformer_blocks.*.img_mod.1", "transformer_blocks.*.txt_mod.1"))

        text_encoder = offload.fast_load_transformers_model(text_encoder_filename, writable_tensors=False, modelClass=Qwen2_5_VLForConditionalGeneration, defaultConfigPath=os.path.join(tokenizer_path, "config.json"))
        # the 1 GB token table stays in RAM (mmgp looks up the prompt's rows on the CPU) with the 1 GB output head, which never runs: the
        # hidden states come from the base model
        runtime.lookup_on_cpu(text_encoder.get_input_embeddings())
        runtime.unused(text_encoder.lm_head)
        # text_encoder = offload.fast_load_transformers_model(text_encoder_filename, do_quantize=True,  writable_tensors= True , modelClass=Qwen2_5_VLForConditionalGeneration, defaultConfigPath="text_encoder_config.json", verboseLevel=2)
        # text_encoder.to(torch.float16)
        # offload.save_model(text_encoder, "text_encoder_quanto_fp16.safetensors", do_quantize= True)
        if base_model_type == "qwen_image_layered_20B":
            VAE_upsampling = None
            VAE_upsampler_factor = 1
            preprocess_vae_sd = None
            vae_checkpoint = "qwen_image_layered_vae_bf16.safetensors"
            vae_config_file = "models/qwen/configs/qwen_image_layered_vae_config.json"
            vae_override = model_def.get("vae_URL", None) or model_def.get("vae_URLs", None)
            if isinstance(vae_override, list):
                vae_override = vae_override[0] if len(vae_override) > 0 else None
            if isinstance(vae_override, dict):
                vae_override = vae_override.get("URLs", None)
            if vae_override:
                vae_checkpoint = vae_override
            vae = offload.fast_load_transformers_model(fl.locate_file(vae_checkpoint), writable_tensors=False, modelClass=AutoencoderKLQwenImage, defaultConfigPath=fl.locate_file(vae_config_file), configKwargs={"upsampler_factor": VAE_upsampler_factor}, preprocess_sd=preprocess_vae_sd)
        else:
            vae = load_vae(model_def, VAE_upsampling)
        vae.upsampling_set = VAE_upsampling
        self.pipeline = QwenImagePipeline(vae, text_encoder, tokenizer, transformer, processor)
        self.vae=vae
        self.text_encoder=text_encoder
        self.tokenizer=tokenizer
        self.transformer=transformer
        self.processor = processor

    @generation_progress
    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "replace the logo with the text 'Black Forest Labs'",
        n_prompt = None,
        sampling_steps: int = 20,
        input_ref_images = None,
        input_frames= None,
        input_masks= None,
        width= 832,
        height=480,
        guide_scale: float = 4,
        fit_into_canvas = None,
        callback = None,
        loras_slists = None,
        batch_size = 1,
        video_prompt_type = "",
        VAE_tile_size = None, 
        joint_pass = True,
        sample_solver='default',
        denoising_strength = 1.,
        masking_strength = 1.,
        model_mode = 0,
        outpainting_dims = None,
        vae_upsampler=None,
        set_progress_status=None,
        **bbargs
    ):
        # Generate with different aspect ratios
        aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472)
        }
        

        if sample_solver =='lightning':
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        else:
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": 0.5,
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": 0.9,
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": 0.02,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False
            }

        self.scheduler=FlowMatchEulerDiscreteScheduler(**scheduler_config)
        self.pipeline.scheduler = self.scheduler 
        if VAE_tile_size is not None:
            if isinstance(VAE_tile_size, int):
                tiling_type = VAE_tile_size
                VAE_tile_size = [False, 0] if tiling_type == 0 else [True, 256]  
            tile_size = VAE_tile_size[1]
            if VAE_tile_size[0]:  # tiles of that many pixels, a quarter of them shared with the next tile
                self.vae.enable_tiling(tile_size, tile_size, tile_size * 3 // 4, tile_size * 3 // 4)
            self.vae.use_tiling = VAE_tile_size[0]
        # tile_size = 256
        qwen_edit_plus = self.base_model_type in ["qwen_image_edit_plus_20B", "qwen_image_edit_plus2_20B"]
        qwen_layered = self.base_model_type in ["qwen_image_layered_20B"]
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        # width, height = aspect_ratios["16:9"]

        if n_prompt is None or len(n_prompt) == 0:
            if qwen_layered:
                n_prompt = " "
            else:
                n_prompt=  "text, watermark, copyright, blurry, low resolution"

        image_mask = None if input_masks is None else convert_tensor_to_image(input_masks, mask_levels= True) 
        if input_frames is not None:
            input_ref_images = [convert_tensor_to_image(input_frames) ] +  ([] if input_ref_images  is None else input_ref_images )

        if input_ref_images is not None:
            # image stiching method
            if qwen_layered:
                input_ref_images = [input_ref_images[0]]
            else:
                stiched = input_ref_images[0]
                if "K" in video_prompt_type :
                    w, h = input_ref_images[0].size
                    height, width = calculate_new_dimensions(height, width, h, w, fit_into_canvas)

                if not qwen_edit_plus:
                    for new_img in input_ref_images[1:]:
                        stiched = stitch_images(stiched, new_img)
                    input_ref_images  = [stiched]

        num_images_per_prompt = 1 if qwen_layered else batch_size
        layers = batch_size if qwen_layered else 1
        def _vae_upsampler_progress(_phase, current_step=None, total_steps=None):
            if callable(set_progress_status):
                label = getattr(vae_upsampler, "progress_label", "VAE Spatial Upsampling")
                set_progress_status(f"{label} in Progress" if current_step is None or total_steps is None else f"{label} in Progress ({int(current_step) + 1}/{int(total_steps)})")

        image = self.pipeline(
            prompt=input_prompt,
            negative_prompt=n_prompt,
            image = input_ref_images,
            image_mask = image_mask,
            width=width,
            height=height,
            num_inference_steps=sampling_steps,
            num_images_per_prompt = num_images_per_prompt,
            layers = layers,
            cfg_normalize = True,
            true_cfg_scale=guide_scale,
            callback = callback,
            pipeline=self,
            loras_slists=loras_slists,
            joint_pass = joint_pass,
            denoising_strength=denoising_strength,
            masking_strength=masking_strength,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            model_mode = model_mode,
            outpainting_dims = outpainting_dims,
            qwen_edit_plus = qwen_edit_plus,
            VAE_tile_size = tile_size,
            vae_upsampler=vae_upsampler,
            vae_upsampler_seed=seed,
            vae_upsampler_progress_callback=_vae_upsampler_progress,
        )      
        if image is None: return None
        return image.transpose(0, 1)

    def get_loras_transformer(self, get_model_recursive_prop, model_type, model_mode, image_mode, **kwargs):
        if image_mode !=2 or model_mode != 1: return [], []
        preloadURLs = get_model_recursive_prop(model_type,  "preload_URLs")
        if len(preloadURLs) == 0: return [], []
        return [ os.path.abspath(fl.locate_file(os.path.basename(preloadURLs[0])))] , [1]


