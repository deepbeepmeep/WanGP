from shared.utils.phase_progress import set_phase_status
import torch
import einops
from mmgp import runtime
import numpy as np
import tqdm
from PIL import Image
import torchvision.transforms.v2 as transforms
# FlowUniPCMultistepScheduler generates more details than FlowMatchEulerDiscreteScheduler
from .fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402

from .flash_scheduler import FlashFlowMatchEulerDiscreteScheduler
from .utils import resize_pilimage, calculate_dimensions, find_closest_resolution, get_rope_index_fix_point

TIMESTEP_TOKEN_NUM = 1
NOISE_SCALE = 8.0
T_EPS = 0.001
CONDITION_IMAGE_SIZE = 384
PATCH_SIZE = 32

TENSOR_TRANSFORM = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.5], [0.5]),
])

DEFAULT_TIMESTEPS = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
]


def resample_timesteps(timesteps, num_steps):
    if num_steps == len(timesteps):
        return list(timesteps)
    positions = np.linspace(0, len(timesteps) - 1, num_steps)
    values = np.interp(positions, np.arange(len(timesteps)), timesteps)
    return [int(round(value)) for value in values]


def build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config):
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + boi_token
            + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64, device=input_ids.device
    ).unsqueeze(0)

    vision_tokens = torch.full((1, image_len), image_token_id, dtype=input_ids.dtype, device=input_ids.device)
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1, image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=None, skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype, device=input_ids.device)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

    vinput_mask = (token_types == 1)
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'token_types': token_types_bin,
        'vinput_mask': vinput_mask,
    }

def build_scheduler(num_inference_steps, timesteps_list, shift, device, scheduler_name="default"):
    if timesteps_list is not None:
        num_inference_steps = len(timesteps_list)
    if scheduler_name == "flash":
        sched = FlashFlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    elif scheduler_name == "default":
        sched = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)
    else:
        raise ValueError(f"Unknown scheduler_name={scheduler_name!r}")
    sched.set_timesteps(num_inference_steps, device=device)
    if timesteps_list is not None:
        sched.timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in sched.timesteps]
        sigmas.append(0.0)
        sched.sigmas = torch.tensor(sigmas, device=device)
    return sched


def clamp_tensor(tensor, percentage = 0.1):
    lower_bound = torch.quantile(tensor.float(), percentage)
    upper_bound = torch.quantile(tensor.float(), 1 - percentage)
    src_dtype = tensor.dtype
    return torch.clamp(tensor.float(), min=lower_bound, max=upper_bound).to(src_dtype)


@torch.no_grad()
def generate_image(
        model,
        processor,
        prompt: str,
        ref_image_paths: list = None,
        ref_images: list = None,
        height: int = 1440,
        width: int = 2560,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 3.0,
        timesteps_list=None,
        scheduler_name: str = "default",
        seed: int = 42,
        noise_scale_start: float = NOISE_SCALE,
        noise_scale_end: float = NOISE_SCALE,
        noise_clip_std: float = 0.0,
        keep_original_aspect: bool = False,
        batch_size: int = 1,
        joint_pass: bool = True,
        callback=None,
        abort_callback=None,
) -> Image.Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    model_config = model.config
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    ref_image_paths = [] if ref_image_paths is None else ref_image_paths
    ref_images = [] if ref_images is None else ref_images
    requested_width, requested_height = width, height
    resize_to_requested = False
    ref_count = len(ref_images) + len(ref_image_paths)
    batch_size = max(1, int(batch_size))

    # Single-reference edits are unstable on low/non-native canvases. Use the
    # upstream reference canvas internally, then resize to WanGP's requested size.
    preresized_ref_pil = None
    if ref_count == 1:
        pil_orig = (ref_images[0] if ref_images else Image.open(ref_image_paths[0])).convert("RGB")
        preresized_ref_pil = resize_pilimage(pil_orig, 2048, PATCH_SIZE)
        width, height = preresized_ref_pil.size
        resize_to_requested = not keep_original_aspect and (width != requested_width or height != requested_height)
        if keep_original_aspect:
            print(f"[info] keep_original_aspect: target size set to {width}x{height} from reference image")
    elif keep_original_aspect:
        print("[warning] keep_original_aspect requires exactly one reference image; using requested dimensions.")
    elif ref_count:
        native_width, native_height = find_closest_resolution(width, height)
        resize_to_requested = native_width != width or native_height != height
        width, height = native_width, native_height

    h_patches = height // PATCH_SIZE
    w_patches = width // PATCH_SIZE

    if not ref_images and not ref_image_paths:
        cond_sample = build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config)
        cond_sample["prediction_mask"] = cond_sample["vinput_mask"]
        uncond_sample = None
        if guidance_scale > 1.0:
            uncond_sample = build_t2i_text_sample(" ", height, width, tokenizer, processor, model_config)
            uncond_sample["prediction_mask"] = uncond_sample["vinput_mask"]
        
        def to_device(s):
            return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s.items()}
        
        cond_sample = to_device(cond_sample)
        if uncond_sample is not None:
            uncond_sample = to_device(uncond_sample)
            
        ref_patches = None
        tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        samples = [cond_sample]
        if uncond_sample:
            samples.append(uncond_sample)
    else:
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        spatial_merge_size = model_config.vision_config.spatial_merge_size
        
        if preresized_ref_pil is not None:
            ref_pils = [preresized_ref_pil]
        else:
            ref_pils = [img.convert("RGB") for img in ref_images] + [Image.open(p).convert("RGB") for p in ref_image_paths]
        K = len(ref_pils)

        if K == 1: max_size = max(height, width)
        elif K == 2: max_size = max(height, width) * 48 // 64
        elif K <= 4: max_size = max(height, width) // 2
        elif K <= 8: max_size = max(height, width) * 24 // 64
        else: max_size = max(height, width) // 4

        ref_pils_resized, ref_images = [], []
        for pil in ref_pils:
            # Skip resizing when caller already produced a patch-aligned ref via
            # `keep_original_aspect` — re-running resize_pilimage on it would
            # upscale (since max_size == max(width, height) of the resized ref).
            if preresized_ref_pil is not None and pil is preresized_ref_pil:
                pil_r = pil
            else:
                pil_r = resize_pilimage(pil, max_size, PATCH_SIZE)
            ref_pils_resized.append(pil_r)
            x = TENSOR_TRANSFORM(pil_r)
            x = einops.rearrange(x, "C (H p1) (W p2) -> (H W) (C p1 p2)", p1=PATCH_SIZE, p2=PATCH_SIZE)
            ref_images.append(x)

        ref_image_lens = [img.shape[0] for img in ref_images]
        total_ref_len = sum(ref_image_lens)
        ref_patches = torch.cat(ref_images, dim=0).unsqueeze(0).to(device, dtype)

        tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE

        if K <= 4: cond_img_size = CONDITION_IMAGE_SIZE
        elif K <= 8: cond_img_size = CONDITION_IMAGE_SIZE * 48 // 64
        else: cond_img_size = CONDITION_IMAGE_SIZE // 2

        ref_pils_vlm = []
        for pil_r in ref_pils_resized:
            cond_w, cond_h = calculate_dimensions(cond_img_size, pil_r.width / pil_r.height)
            ref_pils_vlm.append(pil_r.resize((cond_w, cond_h), resample=Image.LANCZOS))

        image_grid_thw_tgt = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64, device="cpu").unsqueeze(0)
        image_grid_thw_ref = torch.zeros((K, 3), dtype=torch.int64, device="cpu")
        for i, pil_r in enumerate(ref_pils_resized):
            rw, rh = pil_r.size
            image_grid_thw_ref[i] = torch.tensor([1, rh // PATCH_SIZE, rw // PATCH_SIZE], dtype=torch.int64, device=image_grid_thw_ref.device)

        samples = []
        captions = [prompt]
        if guidance_scale > 1.0:
            captions.append(" ")
            
        for caption in captions:
            boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
            tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")
            
            content = [{"type": "image"} for _ in range(K)]
            content.append({"type": "text", "text": caption})
            messages = [{"role": "user", "content": content}]
            template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            proc = processor(text=[template_caption], images=ref_pils_vlm, padding="longest", return_tensors="pt")
            proc_input_ids = proc.input_ids.to("cpu")
            proc_image_grid_thw = proc.image_grid_thw.to("cpu")
            proc_pixel_values = proc.pixel_values.to("cpu")
            input_ids_2 = tokenizer.encode(boi_token + tms_token * TIMESTEP_TOKEN_NUM, return_tensors="pt", add_special_tokens=False)
            input_ids = torch.cat([proc_input_ids, input_ids_2.to(proc_input_ids.device)], dim=-1)

            igthw_cond = proc_image_grid_thw.clone()
            for i in range(K):
                igthw_cond[i, 1] //= spatial_merge_size
                igthw_cond[i, 2] //= spatial_merge_size
            igthw_all = torch.cat([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)

            vision_tokens_list = []
            vt_tgt = torch.full((1, tgt_image_len), image_token_id, dtype=input_ids.dtype, device=input_ids.device)
            vt_tgt[0, 0] = vision_start_token_id
            vision_tokens_list.append(vt_tgt)
            for rl in ref_image_lens:
                vt_ref = torch.full((1, rl), image_token_id, dtype=input_ids.dtype, device=input_ids.device)
                vt_ref[0, 0] = vision_start_token_id
                vision_tokens_list.append(vt_ref)
            vision_tokens = torch.cat(vision_tokens_list, dim=1)
            input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

            position_ids, _ = get_rope_index_fix_point(
                1, image_token_id, video_token_id, vision_start_token_id,
                input_ids=input_ids_pad, image_grid_thw=igthw_all,
                video_grid_thw=None, attention_mask=None,
                skip_vision_start_token=[0] * K + [1] + [1] * K,
            )
            txt_seq_len = input_ids.shape[-1]
            all_seq_len = position_ids.shape[-1]

            token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype, device=input_ids.device)
            bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
            end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
            token_types_raw[0, bgn:end] = 1
            token_types_raw[0, end: end + total_ref_len] = 2
            token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

            vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
            token_types_bin = (token_types_raw > 0).to(token_types_raw.dtype)
            prediction_mask = token_types_raw == 1

            samples.append({
                "input_ids": input_ids.to(device),
                "position_ids": position_ids.to(device),
                "token_types": token_types_bin.to(device),
                "vinput_mask": vinput_mask.to(device),
                "prediction_mask": prediction_mask.to(device),
                "pixel_values_cpu": proc_pixel_values,
                "image_grid_thw": proc_image_grid_thw.to(device),
            })

        set_phase_status("Encoding Image Features")
        for sample in samples:
            with torch.autocast(device.type, dtype=dtype, cache_enabled=False):
                pixel_values = sample.pop("pixel_values_cpu").to(device, dtype)
                image_embeds, _ = model.get_image_features(pixel_values, sample["image_grid_thw"])
                if image_embeds is None:
                    return None
                sample["image_embeds"] = torch.cat(image_embeds, dim=0).to(device, dtype)
                del pixel_values, image_embeds

    set_phase_status("Preparing Denoising")
    noise = torch.empty((batch_size, 3, height, width), device="cpu", dtype=torch.float32)
    for batch_idx in range(batch_size):
        noise[batch_idx].normal_(generator=torch.Generator("cpu").manual_seed(seed + batch_idx + 1))
    noise = noise.mul_(noise_scale_start).to(device, dtype)
    z = einops.rearrange(noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)', p1=PATCH_SIZE, p2=PATCH_SIZE)
    del noise

    sched = build_scheduler(num_inference_steps, timesteps_list, shift, device, scheduler_name)
    if callback is not None:
        callback(-1, None, True, override_num_inference_steps=len(sched.timesteps))

    num_steps = len(sched.timesteps)
    if num_steps > 1:
        noise_scale_schedule = [
            noise_scale_start + (noise_scale_end - noise_scale_start) * i / (num_steps - 1)
            for i in range(num_steps)
        ]
    else:
        noise_scale_schedule = [noise_scale_start]

    torch.manual_seed(seed + 1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + 1)

    def forward_once(sample, z_in, t_pixeldit):
        with torch.autocast(device.type, dtype=dtype, cache_enabled=False):
            kwargs = {
                "input_ids": sample['input_ids'],
                "position_ids": sample['position_ids'],
                "vinputs": z_in,
                "timestep": t_pixeldit.reshape(1).expand(z_in.shape[0]).to(device),
                "token_types": sample['token_types'],
            }
            if "pixel_values" in sample: kwargs["pixel_values"] = sample["pixel_values"]
            if "image_grid_thw" in sample: kwargs["image_grid_thw"] = sample["image_grid_thw"]
            if "image_embeds" in sample: kwargs["image_embeds"] = sample["image_embeds"]
            if "prediction_mask" in sample: kwargs["prediction_mask"] = sample["prediction_mask"]

            outputs = model(**kwargs)
            if outputs.x_pred is None:
                return None
            
        x_pred = outputs.x_pred
        del outputs
        # x_pred = clamp_tensor(x_pred, percentage = 0.01)
        if "prediction_mask" in sample:
            return x_pred
        if ref_patches is None:
            return x_pred[0, sample['vinput_mask'][0]].unsqueeze(0)
        else:
            return x_pred[0, sample['vinput_mask'][0]][:tgt_image_len].unsqueeze(0)

    def forward_samples(sample_list, z_in, t_pixeldit):
        if not (joint_pass and runtime.joint_pass_useful(model, sample_list[0]["input_ids"].shape[-1], model_config.text_config.hidden_size, len(sample_list))):
            return [forward_once(sample, z_in, t_pixeldit) for sample in sample_list]

        timestep = t_pixeldit.reshape(1).expand(z_in.shape[0]).to(device)
        with torch.autocast(device.type, dtype=dtype, cache_enabled=False):
            kwargs = {
                "input_ids": [sample["input_ids"] for sample in sample_list],
                "position_ids": [sample["position_ids"] for sample in sample_list],
                "vinputs": [z_in] * len(sample_list),
                "timestep": [timestep] * len(sample_list),
                "token_types": [sample["token_types"] for sample in sample_list],
            }
            if all("image_grid_thw" in sample for sample in sample_list):
                kwargs["image_grid_thw"] = [sample["image_grid_thw"] for sample in sample_list]
            if all("image_embeds" in sample for sample in sample_list):
                kwargs["image_embeds"] = [sample["image_embeds"] for sample in sample_list]
            if all("prediction_mask" in sample for sample in sample_list):
                kwargs["prediction_mask"] = [sample["prediction_mask"] for sample in sample_list]

            outputs = model(**kwargs)
            if outputs.x_pred is None:
                return [None] * len(sample_list)
            x_preds = list(outputs.x_pred)
            del outputs
            return x_preds

    def _preview_tensor(z_in):
        return einops.rearrange(
            z_in.detach(),
            'B (H W) (C p1 p2) -> C B (H p1) (W p2)',
            H=h_patches, W=w_patches, p1=PATCH_SIZE, p2=PATCH_SIZE,
        )

    for step_idx, step_t in enumerate(tqdm.tqdm(sched.timesteps, desc="Generating")):
        if abort_callback is not None and abort_callback():
            return None

        t_pixeldit = 1.0 - step_t.float() / 1000.0
        sigma = (step_t.float() / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)

        if ref_patches is None:
            z_float = z.to(dtype=torch.float32)
            x_preds = forward_samples(samples, z, t_pixeldit)
            x_pred_cond = x_preds[0]
            if x_pred_cond is None:
                return None
            v_cond = x_pred_cond.to(dtype=torch.float32)
            del x_pred_cond
            v_cond.sub_(z_float).div_(sigma)

            if len(samples) > 1:
                x_pred_uncond = x_preds[1]
                if x_pred_uncond is None:
                    return None
                v_uncond = x_pred_uncond.to(dtype=torch.float32)
                del x_pred_uncond
                v_uncond.sub_(z_float).div_(sigma)
                v_cond.sub_(v_uncond).mul_(guidance_scale).add_(v_uncond)
                del v_uncond
            else:
                pass
            x_preds.clear()
            v_guided = v_cond
        else:
            vinputs = torch.cat([z, ref_patches.expand(z.shape[0], -1, -1)], dim=1)
            x_vis_list = forward_samples(samples, vinputs, t_pixeldit)
            del vinputs
            if any(x is None for x in x_vis_list):
                return None
            z_float = z.to(dtype=torch.float32)
            v_cond = x_vis_list[0].to(dtype=torch.float32)
            v_cond.sub_(z_float).div_(sigma)
            if len(samples) > 1:
                v_uncond = x_vis_list[1].to(dtype=torch.float32)
                v_uncond.sub_(z_float).div_(sigma)
                v_cond.sub_(v_uncond).mul_(guidance_scale).add_(v_uncond)
                v_guided = v_cond
                del v_uncond
            else:
                v_guided = v_cond
            x_vis_list.clear()

        model_output = v_guided.neg_()
        del v_guided
        # model_output = clamp_tensor(model_output, percentage = 0.05)
        step_t_float = step_t.to(dtype=torch.float32)
        if scheduler_name == "flash":
            new_z = sched.step(model_output, step_t_float, z_float, s_noise=noise_scale_schedule[step_idx], noise_clip_std=noise_clip_std, return_dict=False)[0].to(dtype)
        else:
            new_z = sched.step(model_output, step_t_float, z_float, return_dict=False)[0].to(dtype)
        del model_output, z, z_float, step_t_float
        z = new_z
        del new_z

        if callback is not None:
            callback(step_idx, _preview_tensor(z), False)

    img = z.to(device="cpu", dtype=torch.float32)
    del z
    img.add_(1).div_(2)
    img = einops.rearrange(img, 'B (H W) (C p1 p2) -> B C (H p1) (W p2)', H=h_patches, W=w_patches, p1=PATCH_SIZE, p2=PATCH_SIZE)
    images = []
    for batch_idx in range(img.shape[0]):
        arr = np.round(np.clip(img[batch_idx].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        image = Image.fromarray(arr).convert("RGB")
        if resize_to_requested:
            image = image.resize((requested_width, requested_height), resample=Image.LANCZOS)
        images.append(image)
    del img
    return images[0] if batch_size == 1 else images
