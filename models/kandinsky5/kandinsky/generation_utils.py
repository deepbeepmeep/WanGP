from shared.utils.phase_progress import text_encoding_prompts
import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import torch
from tqdm import tqdm 

from .models.utils import fast_sta_nabla
import torchvision.transforms.functional as F


def get_sparse_params(conf, batch_embeds, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, _ = batch_embeds["visual"].shape
    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )
    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(T, H // 8, W // 8, conf.model.attention.wT,
                                  conf.model.attention.wH, conf.model.attention.wW, device=device)
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
            "wT": conf.model.attention.wT,
            "wW": conf.model.attention.wW,
            "wH": conf.model.attention.wH,
            "add_sta": conf.model.attention.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(conf.model.attention, "method", "topcdf"),
        }
    else:
        sparse_params = None

    return sparse_params

def adaptive_mean_std_normalization(source, reference):
    source_mean = source.mean(dim=(1,2,3),keepdim=True)
    source_std = source.std(dim=(1,2,3),keepdim=True)
    #magic constants - limit changes in latents
    clump_mean_low = 0.05
    clump_mean_high = 0.1
    clump_std_low = 0.1
    clump_std_high = 0.25

    reference_mean = torch.clamp(reference.mean(), source_mean - clump_mean_low, source_mean + clump_mean_high)
    reference_std = torch.clamp(reference.std(), source_std - clump_std_low, source_std + clump_std_high)

    # normalization
    normalized = (source - source_mean) / source_std
    normalized = normalized * reference_std + reference_mean
    
    return normalized

def normalize_first_frame(latents, reference_frames=5, clump_values=False):
    latents_copy = latents.clone()
    samples = latents_copy
    
    if samples.shape[0] <= 1:
        return (latents, "Only one frame, no normalization needed")
    nFr = 4
    first_frames = samples[:nFr]
    reference_frames_data = samples[nFr:nFr+min(reference_frames, samples.shape[0]-1)]
    
    # print("First frame stats - Mean:", first_frames.mean(dim=(1,2,3)), "Std: ", first_frames.std(dim=(1,2,3)))
    # print(f"Reference frames stats - Mean: {reference_frames_data.mean().item():.4f}, Std: {reference_frames_data.std().item():.4f}")
    
    normalized_first = adaptive_mean_std_normalization(first_frames, reference_frames_data)
    if clump_values:
        min_val = reference_frames_data.min()
        max_val = reference_frames_data.max()
        normalized_first = torch.clamp(normalized_first, min_val, max_val)
    
    samples[:nFr] = normalized_first
    
    return samples

@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
    attention_mask=None,
    null_attention_mask=None,
    joint_pass=False,
):
    with torch._dynamo.utils.disable_cache_limit():
        if joint_pass and abs(guidance_weight - 1.0) > 1e-6:
            outputs = dit(
                [x, x],
                [text_embeds["text_embeds"], null_text_embeds["text_embeds"]],
                [text_embeds["pooled_embed"], null_text_embeds["pooled_embed"]],
                t * 1000,
                [visual_rope_pos, visual_rope_pos],
                [text_rope_pos, null_text_rope_pos],
                scale_factor=[conf.metrics.scale_factor, conf.metrics.scale_factor],
                sparse_params=[sparse_params, sparse_params],
                attention_mask=[attention_mask, null_attention_mask],
            )
            if outputs is None:
                return None
            pred_velocity, uncond_pred_velocity = outputs
            pred_velocity = uncond_pred_velocity + guidance_weight * (
                pred_velocity - uncond_pred_velocity
            )
        else:
            pred_velocity = dit(
                x,
                text_embeds["text_embeds"],
                text_embeds["pooled_embed"],
                t * 1000,
                visual_rope_pos,
                text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=attention_mask,
            )
            if pred_velocity is None:
                return None
            if abs(guidance_weight - 1.0) > 1e-6:
                uncond_pred_velocity = dit(
                    x,
                    null_text_embeds["text_embeds"],
                    null_text_embeds["pooled_embed"],
                    t * 1000,
                    visual_rope_pos,
                    null_text_rope_pos,
                    scale_factor=conf.metrics.scale_factor,
                    sparse_params=sparse_params,
                    attention_mask=null_attention_mask,
                )
                if uncond_pred_velocity is None:
                    return None
                pred_velocity = uncond_pred_velocity + guidance_weight * (
                    pred_velocity - uncond_pred_velocity
                )
    return pred_velocity


@torch.no_grad()
def generate(
    model,
    device,
    img,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    first_frames,
    conf,
    progress=False,
    seed=6554,
    attention_mask=None,
    null_attention_mask=None,
    callback=None,
    interrupt_check=None,
    joint_pass=False,
):
    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    if callback is not None:
        callback(-1, None, True, override_num_inference_steps=num_steps)

    step_iter = zip(timesteps[:-1], torch.diff(timesteps))
    if progress:
        step_iter = tqdm(step_iter, total=num_steps)

    for step_idx, (timestep, timestep_diff) in enumerate(step_iter):
        if interrupt_check is not None and interrupt_check():
            return None
        time = timestep.unsqueeze(0)
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            if first_frames is not None:
                first_frames = first_frames.to(device=visual_cond.device, dtype=visual_cond.dtype)
                img[:1] = first_frames
                visual_cond_mask[:1] = 1
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        pred_velocity = get_velocity(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
            joint_pass=joint_pass,
        )
        if pred_velocity is None:
            return None
        preview_img = None
        if callback is not None and getattr(callback, "tiny_vae", False):
            preview_img = img[..., :pred_velocity.shape[-1]] - timestep * pred_velocity
        img[..., :pred_velocity.shape[-1]] += timestep_diff * pred_velocity
        if callback is not None:
            if preview_img is None:
                preview_img = img
            latents_preview = None
            if visual_rope_pos is not None and len(visual_rope_pos) > 0:
                duration = int(visual_rope_pos[0].numel())
                if duration > 0 and img.shape[0] % duration == 0:
                    batch = img.shape[0] // duration
                    latents_preview = (
                        preview_img.reshape(batch, duration, *preview_img.shape[1:])
                        .permute(0, 4, 1, 2, 3)
                        .detach()
                    )[0]
            callback(step_idx, latents_preview, False)
            del latents_preview
        del preview_img
        # NOTE: remove extra channels that can be added in Image Editing (I2I)
    return img[..., :pred_velocity.shape[-1]]


def resize_video(video, visual_size):
    height, width = video.shape[-2:]
    nearest_height, nearest_width = visual_size

    scale_factor = min(height / nearest_height, width / nearest_width)
    video = F.resize(video, (int(height / scale_factor), int(width / scale_factor)))

    height, width = video.shape[-2:]
    video = F.crop(
        video,
        (height - nearest_height) // 2,
        (width - nearest_width) // 2,
        nearest_height,
        nearest_width,
    )
    return video


def encode_video(data, vae, image_vae): # batch, channels, time, h, w
    if image_vae:
        assert data.shape[2] == 1
        data = vae.encode(data[:, :, 0]).latent_dist.sample()[:, :, None]
    else:
        data = vae.encode(data)[0]
    data *= vae.config.scaling_factor
    return data.permute(0, 2, 3, 4, 1) # batch, time, h, w, channels


def generate_sample(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    joint_pass=False,
    callback=None,
    interrupt_check=None,
):
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)

    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad(), text_encoding_prompts(sum((type_of_content, p) not in text_embedder.text_encoder_cache._entries for p in dict.fromkeys([caption, negative_caption]))):
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    if null_attention_mask is not None:
        null_attention_mask = null_attention_mask.to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                callback=callback,
                interrupt_check=interrupt_check,
                joint_pass=joint_pass,
            )

    if latent_visual is None:
        return None
            

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)


    return images

def generate_sample_ti2i(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    image_vae=False,
    image=None,
    joint_pass=False,
    callback=None,
    interrupt_check=None
):
    bs, duration, height, width, dim = shape
    
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)
    
    if duration == 1:
        if image is None:
            type_of_content = "image"
        else:
            type_of_content = 'image_edit'
    else:
        type_of_content = "video"

    
    if image is not None:
        image = [resize_video(image, (height * 8, width * 8))]

    if dit.instruct_type == 'channel':
        if image is not None:
            edit_latent = [(i.to(device=vae_device, dtype=torch.bfloat16) / 127.5 - 1.0) for i in image]
            edit_latent = torch.cat([encode_video(i[:,:,None], vae, image_vae).squeeze(0) for i in edit_latent], 0)
            edit_latent = torch.cat([edit_latent, torch.ones_like(img[...,:1])],-1)
        else:
            edit_latent = torch.cat([torch.zeros_like(img), torch.zeros_like(img[...,:1])],-1)
        img = torch.cat([img, edit_latent],dim=-1)
    
    with torch.no_grad(), text_encoding_prompts(2 if image is not None else sum((type_of_content, p) not in text_embedder.text_encoder_cache._entries for p in dict.fromkeys([caption, negative_caption]))):
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, images=image
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content, images=image
        )

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device,dtype=torch.bfloat16)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device,dtype=torch.bfloat16)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    if null_attention_mask is not None:
        null_attention_mask = null_attention_mask.to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                callback=callback,
                interrupt_check=interrupt_check,
                joint_pass=joint_pass,
            )

    if latent_visual is None:
        return None
            

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            if image_vae:
                images = images[:,:,0]
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)


    return images

def generate_sample_i2v(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    images,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
    callback=None,
    joint_pass=False,
    interrupt_check=None
):
    text_embedder.embedder.mode = "i2v"
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)
    
    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"
        
    with torch.no_grad(), text_encoding_prompts(sum((type_of_content, p) not in text_embedder.text_encoder_cache._entries for p in dict.fromkeys([caption, negative_caption]))):
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    if null_attention_mask is not None:
        null_attention_mask = null_attention_mask.to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    first_frames = images

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                first_frames,
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                callback=callback,
                interrupt_check=interrupt_check,
                joint_pass=joint_pass,
            )
    if latent_visual is None:
        return None

    if images is not None:
        images = images.to(device=latent_visual.device, dtype=latent_visual.dtype)
        latent_visual[:1] = images
    latent_visual = normalize_first_frame(latent_visual)


    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)


    return images
