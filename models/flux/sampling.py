from shared.utils.phase_progress import text_encoding_progress
import math
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from .model import Flux
from .modules.autoencoder import AutoEncoder
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import CannyImageEncoder, DepthImageEncoder, ReduxImageEncoder
from .util import PREFERED_KONTEXT_RESOLUTIONS
import torchvision.transforms.functional as TVF


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    *,
    channels: int = 16,
    patch_size: int = 2,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    if channels == 16 and patch_size == 2:
        return torch.randn(
            num_samples,
            channels,
            # allow for packing
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
            dtype=dtype,
            device=device,
            generator=generator,
        )

    return torch.randn(
        num_samples,
        channels,
        height,
        width,
        dtype=dtype,
        device=device,
        generator=generator,
    )


def prepare_prompt(t5: HFEmbedder, clip: HFEmbedder, bs: int, prompt: str | list[str], neg: bool = False, device: str = "cuda") -> dict[str, Tensor]:
    with text_encoding_progress(list(t5.hf_module.encoder.block) + list(clip.hf_module.text_model.encoder.layers), prompt_count=1 if isinstance(prompt, str) else len(prompt)):
        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)

        if isinstance(prompt, str):
            prompt = [prompt]
        txt = t5(prompt)
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3, device=device)

        vec = clip(prompt)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)

        return {
            "neg_txt" if neg else "txt": txt.to(device),
            "neg_txt_ids" if neg else "txt_ids": txt_ids.to(device),
            "neg_vec" if neg else "vec": vec.to(device),
        }


def prepare_img(img: Tensor, patch_size: int = 2) -> dict[str, Tensor]:
    bs, c, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // patch_size, w // patch_size, 3, device=img.device)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // patch_size, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // patch_size, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
    }





def prepare_redux(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    encoder: ReduxImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    with torch.no_grad():
        img_cond = encoder(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    txt = torch.cat((txt, img_cond.to(txt)), dim=-2)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=txt.device)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def resizeinput(img):
    multiple_of = 16
    image_height, image_width = img.height, img.width
    aspect_ratio = image_width / image_height
    _, image_width, image_height = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS
    )
    image_width = image_width // multiple_of * multiple_of
    image_height = image_height // multiple_of * multiple_of
    if (image_width, image_height) != img.size:
        img = img.resize((image_width, image_height), Image.LANCZOS)
    return img


def build_mask(target_width, target_height, img_mask, device):
    from shared.utils.utils import convert_image_to_tensor, convert_tensor_to_image
    # image_height, image_width = calculate_new_dimensions(ref_height, ref_width, image_height, image_width, False, block_size=multiple_of)
    image_mask_latents = convert_image_to_tensor(img_mask.resize((target_width // 16, target_height // 16), resample=Image.Resampling.LANCZOS))
    image_mask_latents = torch.where(image_mask_latents>-0.5, 1., 0. )[0:1]
    image_mask_rebuilt = image_mask_latents.repeat_interleave(16, dim=-1).repeat_interleave(16, dim=-2).unsqueeze(0)
    # convert_tensor_to_image( image_mask_rebuilt.squeeze(0).repeat(3,1,1)).save("mmm.png")
    image_mask_latents = image_mask_latents.reshape(1, -1, 1).to(device)        
    return {
        "img_msk_latents": image_mask_latents,
        "img_msk_rebuilt": image_mask_rebuilt,
    }


def prepare_kontext(
    ae: AutoEncoder | None,
    img_cond_list: list,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
    img_mask = None,
    *,
    patch_size: int = 2,
    noise_channels: int = 16,
) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image

    res_match_output = img_mask is not None

    img_cond_seq = None
    img_cond_seq_ids = None
    if img_cond_list == None: img_cond_list = []
    height_offset = 0
    width_offset = 0
    for cond_no, img_cond in enumerate(img_cond_list): 
        if res_match_output:
            if img_cond.size != (target_width, target_height):
                img_cond = img_cond.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            img_cond = resizeinput(img_cond)
        width, height = img_cond.size
        width, height = width // 8, height // 8

        img_cond = np.array(img_cond)
        img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
        img_cond = rearrange(img_cond, "h w c -> 1 c h w")
        if ae is None:
            raise ValueError("Image conditioning is not supported for this model.")
        with torch.no_grad():
            img_cond_latents = ae.encode(img_cond.to(device))

        img_cond_latents = img_cond_latents.to(torch.bfloat16)
        img_cond_latents = rearrange(img_cond_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img_cond.shape[0] == 1 and bs > 1:
            img_cond_latents = repeat(img_cond_latents, "1 ... -> bs ...", bs=bs)
        img_cond = None

        # image ids are the same as base image with the first dimension set to 1
        # instead of 0
        img_cond_ids = torch.zeros(height // 2, width // 2, 3, device=img_cond_latents.device)
        img_cond_ids[..., 0] = 1
        img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2, device=img_cond_latents.device)[:, None] + height_offset
        img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2, device=img_cond_latents.device)[None, :] + width_offset
        img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)
        height_offset +=  height // 2 
        width_offset +=  width // 2

        if target_width is None:
            target_width = 8 * width
        if target_height is None:
            target_height = 8 * height
        img_cond_ids = img_cond_ids.to(device)
        if cond_no == 0:
            img_cond_seq, img_cond_seq_ids  = img_cond_latents, img_cond_ids
        else:
            img_cond_seq, img_cond_seq_ids  =  torch.cat([img_cond_seq, img_cond_latents], dim=1), torch.cat([img_cond_seq_ids, img_cond_ids], dim=1)
        
    return_dict = {
        "img_cond_seq": img_cond_seq,
        "img_cond_seq_ids": img_cond_seq_ids,
    }
    if img_mask is not None:
        return_dict.update(build_mask(target_width, target_height, img_mask, device))

    img = get_noise(
        bs,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
        channels=noise_channels,
        patch_size=patch_size,
    )
    return_dict.update(prepare_img(img, patch_size=patch_size))

    return return_dict, target_height, target_width


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b



def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule_flux2(num_steps: int, image_seq_len: int, device: torch.device) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def get_schedule_piflux2(num_steps: int, image_seq_len: int, device: torch.device) -> list[float]:
    """
    pi-FLUX.2 FlowMapSDE schedule with shift=3.2 and final_step_size_scale=0.5.
    """
    if num_steps <= 0:
        return [0.0]
    shift = 3.2
    final_step_size_scale = 0.5
    end = (final_step_size_scale - 1.0) / (num_steps + final_step_size_scale - 1.0)
    step = (end - 1.0) / num_steps
    raw_timesteps = 1.0 + step * torch.arange(num_steps, dtype=torch.float32, device=device)
    raw_timesteps = raw_timesteps.clamp(min=0)
    sigmas = shift * raw_timesteps / (1 + (shift - 1) * raw_timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    return sigmas.tolist()


def _flow_map_sde_warp_t(raw_t: Tensor, shift: float) -> Tensor:
    return shift * raw_t / (1 + (shift - 1) * raw_t)


def _flow_map_sde_unwarp_t(sigma_t: Tensor, shift: float) -> Tensor:
    return sigma_t / (shift + (1 - shift) * sigma_t)


def _flow_map_sde_calculate_sigmas_dst(
    sigmas: Tensor, h: float = 0.0, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    sigmas_src = sigmas[:-1]
    sigmas_to = sigmas[1:]
    alphas_src = 1 - sigmas_src
    alphas_to = 1 - sigmas_to

    if h <= 0.0:
        m_vals = torch.ones_like(sigmas_src)
    else:
        h2 = h * h
        m_vals = (sigmas_to * alphas_src / (sigmas_src * alphas_to).clamp(min=eps)) ** h2

    sigmas_to_mul_m = sigmas_to * m_vals
    sigmas_dst = sigmas_to_mul_m / (alphas_to + sigmas_to_mul_m).clamp(min=eps)
    return sigmas_dst, m_vals


def _gmflow_posterior_mean(
    sigma_t_src: Tensor,
    sigma_t: Tensor,
    x_t_src: Tensor,
    x_t: Tensor,
    gm_means: Tensor,
    gm_vars: Tensor,
    gm_logweights: Tensor,
    *,
    eps: float = 1e-6,
    gm_dim: int = -4,
    channel_dim: int = -3,
) -> Tensor:
    sigma_t_src = sigma_t_src.clamp(min=eps)
    sigma_t = sigma_t.clamp(min=eps)

    alpha_t_src = 1 - sigma_t_src
    alpha_t = 1 - sigma_t

    alpha_over_sigma_t_src = alpha_t_src / sigma_t_src
    alpha_over_sigma_t = alpha_t / sigma_t

    zeta = alpha_over_sigma_t.square() - alpha_over_sigma_t_src.square()
    nu = alpha_over_sigma_t * x_t / sigma_t - alpha_over_sigma_t_src * x_t_src / sigma_t_src

    nu = nu.unsqueeze(gm_dim)
    zeta = zeta.unsqueeze(gm_dim)
    denom = (gm_vars * zeta + 1).clamp(min=eps)

    out_means = (gm_vars * nu + gm_means) / denom
    logweights_delta = (gm_means * (nu - 0.5 * zeta * gm_means)).sum(dim=channel_dim, keepdim=True) / denom
    out_weights = (gm_logweights + logweights_delta).softmax(dim=gm_dim)

    return (out_means * out_weights).sum(dim=gm_dim)


class _GMFlowPolicy:
    def __init__(
        self,
        denoising_output: dict[str, Tensor],
        x_t_src: Tensor,
        sigma_t_src: Tensor,
        eps: float = 1e-4,
    ) -> None:
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.eps = eps
        self.sigma_t_src = sigma_t_src.reshape(
            *sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1])
        )
        self.denoising_output_x_0 = self._u_to_x_0(denoising_output, self.x_t_src, self.sigma_t_src)

    @staticmethod
    def _u_to_x_0(denoising_output: dict[str, Tensor], x_t: Tensor, sigma_t: Tensor) -> dict[str, Tensor]:
        x_t = x_t.unsqueeze(1)
        sigma_t = sigma_t.unsqueeze(1)
        means_x_0 = x_t - sigma_t * denoising_output["means"]
        gm_vars = (denoising_output["logstds"] * 2).exp() * sigma_t.square()
        return {"means": means_x_0, "gm_vars": gm_vars, "logweights": denoising_output["logweights"]}

    def pi(self, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        means = self.denoising_output_x_0["means"]
        gm_vars = self.denoising_output_x_0["gm_vars"]
        logweights = self.denoising_output_x_0["logweights"]

        if (sigma_t == self.sigma_t_src).all() and (x_t == self.x_t_src).all():
            x_0 = (logweights.softmax(dim=1) * means).sum(dim=1)
        else:
            x_0 = _gmflow_posterior_mean(
                self.sigma_t_src,
                sigma_t,
                self.x_t_src,
                x_t,
                means,
                gm_vars,
                logweights,
                eps=self.eps,
            )
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)

    def temperature_(self, temperature: float) -> None:
        if temperature >= 1.0:
            return
        temperature = max(temperature, self.eps)
        gm = self.denoising_output_x_0
        gm["logweights"] = (gm["logweights"] / temperature).log_softmax(dim=1)
        gm["gm_vars"] = gm["gm_vars"] * temperature


def _policy_rollout(
    x_t_start: Tensor,
    sigma_t_start: Tensor,
    sigma_t_end: Tensor,
    total_substeps: int,
    policy: _GMFlowPolicy,
    *,
    shift: float = 3.2,
) -> Tensor:
    num_batches = x_t_start.size(0)
    ndim = x_t_start.dim()
    sigma_t_start = sigma_t_start.reshape(num_batches, *((ndim - 1) * [1]))
    sigma_t_end = sigma_t_end.reshape(num_batches, *((ndim - 1) * [1]))

    raw_t_start = _flow_map_sde_unwarp_t(sigma_t_start, shift)
    raw_t_end = _flow_map_sde_unwarp_t(sigma_t_end, shift)
    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps
    max_num_substeps = num_substeps.max()

    raw_t = raw_t_start
    sigma_t = sigma_t_start
    x_t = x_t_start

    for substep_id in range(max_num_substeps.item()):
        u = policy.pi(x_t, sigma_t)
        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = _flow_map_sde_warp_t(raw_t_minus, shift)
        x_t_minus = x_t + u * (sigma_t_minus - sigma_t)
        active_mask = num_substeps > substep_id
        x_t = torch.where(active_mask, x_t_minus, x_t)
        sigma_t = torch.where(active_mask, sigma_t_minus, sigma_t)
        raw_t = torch.where(active_mask, raw_t_minus, raw_t)

    return x_t


def _unpack_latent_piflux2(x: Tensor, patch_size: int = 2) -> Tensor:
    bsz, packed_channels, h, w = x.shape
    channels = packed_channels // (patch_size * patch_size)
    x = x.view(bsz, channels, patch_size, patch_size, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(bsz, channels, h * patch_size, w * patch_size)
    return x


def _pack_latent_piflux2(x: Tensor, patch_size: int = 2) -> Tensor:
    bsz, channels, h, w = x.shape
    h_packed = h // patch_size
    w_packed = w // patch_size
    x = x.view(bsz, channels, h_packed, patch_size, w_packed, patch_size)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(
        bsz, channels * patch_size * patch_size, h_packed, w_packed
    )
    return x


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
    device: torch.device = None,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    real_guidance_scale = None,
    final_step_size_scale: float | None = None,
    # extra img tokens (channel-wise)
    neg_txt: Tensor = None,
    neg_txt_ids: Tensor= None,
    neg_vec: Tensor = None,
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
    siglip_embedding = None,
    siglip_embedding_ids = None,
    NAG: dict | None = None,
    callback=None,
    pipeline=None,
    loras_slists=None,
    unpack_latent = None,
    joint_pass= False,
    img_msk_latents = None,
    img_msk_rebuilt = None,
    denoising_strength = 1,
    masking_strength = 1,
    model_mode = None,
    height: int | None = None,
    width: int | None = None,
    vae_scale_factor: int = 8,
    preview_meta = None,
    original_image_latents = None,
    # pi-Flow settings
    piflow_substeps: int = 128,
    piflow_generator: torch.Generator | None = None,
    piflow_gm_temperature: float | None = None,
    kv_cache=None,
):

    kwargs = {
        'pipeline': pipeline,
        'callback': callback,
        "img_len": img.shape[1],
        "siglip_embedding": siglip_embedding,   
        "siglip_embedding_ids": siglip_embedding_ids,
    }
    if NAG is not None:
        kwargs["NAG"] = NAG

    original_timesteps = timesteps
    is_piflow = getattr(model, "piflow", False)
    piflow_spatial_h = piflow_spatial_w = None
    piflow_sigmas = piflow_sigmas_dst = piflow_m_vals = None
    
    morph, first_step = False, 0
    model_mode_int = None
    if model_mode is not None:
        try:
            model_mode_int = int(model_mode)
        except (TypeError, ValueError):
            model_mode_int = None
    lanpaint_proc = None
    lanpaint_mask = None
    true_cfg_scale = 1.0 if real_guidance_scale is None else real_guidance_scale
    if img_msk_latents is not None:
        if not is_piflow and model_mode_int in (2, 3, 4, 5):
            if img_cond_seq is not None:
                if img_cond_seq_ids is not None and img_cond_seq_ids.shape[-1] == 4:
                    first_t = img_cond_seq_ids[:, :1, 0]
                    first_mask = img_cond_seq_ids[..., 0] == first_t
                    if original_image_latents is None:
                        original_image_latents = img_cond_seq[first_mask].view(
                            img_cond_seq.shape[0], -1, img_cond_seq.shape[-1]
                        ).clone()
                    keep_mask = ~first_mask
                    if keep_mask.any():
                        img_cond_seq = img_cond_seq[keep_mask].view(
                            img_cond_seq.shape[0], -1, img_cond_seq.shape[-1]
                        )
                        img_cond_seq_ids = img_cond_seq_ids[keep_mask].view(
                            img_cond_seq_ids.shape[0], -1, img_cond_seq_ids.shape[-1]
                        )
                    else:
                        img_cond_seq = None
                        img_cond_seq_ids = None
                else:
                    base_len = img.shape[1]
                    if original_image_latents is None:
                        original_image_latents = img_cond_seq[:, :base_len].clone()
                    if img_cond_seq.shape[1] <= base_len:
                        img_cond_seq = None
                        img_cond_seq_ids = None
                    else:
                        img_cond_seq = img_cond_seq[:, base_len:]
                        if img_cond_seq_ids is not None:
                            img_cond_seq_ids = img_cond_seq_ids[:, base_len:]
            from shared.lanpaint import LanPaint
            lanpaint_steps = {2: 2, 3: 5, 4: 10, 5: 15}.get(model_mode_int, 5)
            lanpaint_proc = LanPaint(NSteps=lanpaint_steps)
            denoising_strength = 1.0
            masking_strength = 1.0
            if img_msk_latents.shape[-1] != img.shape[-1]:
                lanpaint_mask = img_msk_latents.expand(img.shape[0], img.shape[1], img.shape[2]).contiguous()
            else:
                lanpaint_mask = img_msk_latents
            lanpaint_proc is not None 
        if original_image_latents is None and img_cond_seq is not None:
            original_image_latents = img_cond_seq.clone()
        randn = torch.randn_like(original_image_latents)
        if denoising_strength < 1.:
            first_step = int(len(timesteps[:-1]) * (1. - denoising_strength))
        masked_steps = math.ceil(len(timesteps[:-1]) * masking_strength)
        if not morph:
            latent_noise_factor = timesteps[first_step]
            latents  = original_image_latents  * (1.0 - latent_noise_factor) + randn * latent_noise_factor
            img = latents.to(img)
            latents = None
            timesteps = timesteps[first_step:]


    if is_piflow:
        base_img_ids = img_ids[:, :img.shape[1]]
        piflow_spatial_h = int(base_img_ids[..., 1].max().item() + 1)
        piflow_spatial_w = int(base_img_ids[..., 2].max().item() + 1)

    updated_num_steps= len(timesteps) -1
    if loras_slists is not None:
        from shared.utils.loras_mutipliers import update_loras_slists
        update_loras_slists(model, loras_slists, len(original_timesteps))
    from mmgp import offload
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    if is_piflow and len(timesteps) > 1:
        piflow_sigmas = torch.tensor(timesteps, device=img.device, dtype=torch.float32)
        piflow_sigmas_dst, piflow_m_vals = _flow_map_sde_calculate_sigmas_dst(
            piflow_sigmas, h=0.0
        )
        if piflow_gm_temperature is None:
            nfe = len(timesteps) - 1
            piflow_gm_temperature = min(max(0.1 * (nfe - 1), 0.0), 1.0)

    dynamic_loras = loras_slists is not None and any(isinstance(value, list) or value != loras_slists["phase1"][index]
        for phase in ("phase1", "phase2", "phase3") for index, value in enumerate(loras_slists[phase]))
    if callback is not None:
        callback(-1, None, True, override_num_inference_steps=updated_num_steps)
    for i, (t_curr, t_prev) in enumerate(tqdm(zip(timesteps[:-1], timesteps[1:]), total=updated_num_steps, desc="Denoising", mininterval=1 / 3)):
        offload.set_step_no_for_lora(model, first_step  + i)
        if kv_cache is not None and dynamic_loras and i > 0:
            for cache in kv_cache:
                cache.clear(keep_ring=True)
        if pipeline._interrupt:
            return None

        if img_msk_latents is not None and denoising_strength <1. and i == first_step and morph:
            latent_noise_factor = t_curr/1000
            img  = original_image_latents  * (1.0 - latent_noise_factor) + img * latent_noise_factor 

        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        def run_model(latents, cfg_scale):
            img_input = latents
            img_input_ids = img_ids
            if img_cond is not None:
                img_input = torch.cat((img_input, img_cond), dim=-1)
            if img_cond_seq is not None:
                if kv_cache is None:
                    img_input = torch.cat((img_input, img_cond_seq), dim=1)
                    img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
                else:
                    kwargs.update(reference_latents=img_cond_seq, reference_ids=img_cond_seq_ids, kv_cache=kv_cache)
            if not joint_pass or cfg_scale == 1:
                noise_pred = model(
                    img=img_input,
                    img_ids=img_input_ids,
                    txt_list=[txt],
                    txt_ids_list=[txt_ids],
                    y_list=[vec],
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    **kwargs
                )[0]
                if noise_pred == None:
                    return None, None
                neg_noise_pred = None
                if cfg_scale > 1:
                    neg_noise_pred = model(
                        img=img_input,
                        img_ids=img_input_ids,
                        txt_list=[neg_txt],
                        txt_ids_list=[neg_txt_ids],
                        y_list=[neg_vec],
                        timesteps=t_vec,
                        guidance=guidance_vec,
                        **kwargs
                    )[0]
                    if neg_noise_pred == None:
                        return None, None
            else:
                noise_pred, neg_noise_pred = model(
                    img=img_input,
                    img_ids=img_input_ids,
                    txt_list=[txt, neg_txt],
                    txt_ids_list=[txt_ids, neg_txt_ids],
                    y_list=[vec, neg_vec],
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    **kwargs
                )
                if noise_pred == None:
                    return None, None
            return noise_pred, neg_noise_pred

        def cfg_predictions(noise_pred, neg_noise_pred, cfg_scale, t):
            if cfg_scale > 1:
                return neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)
            return noise_pred

        if lanpaint_proc is not None and height is not None and width is not None and i < updated_num_steps - 1:
            img = lanpaint_proc(
                run_model,
                cfg_predictions,
                true_cfg_scale,
                1.0,
                img,
                original_image_latents,
                randn,
                t_vec,
                lanpaint_mask,
                height=height,
                width=width,
                vae_scale_factor=vae_scale_factor,
            )
            if img is None:
                return None

        pred, neg_pred = run_model(img, true_cfg_scale)
        if pred == None: return None

        preview_x0 = None
        if is_piflow and isinstance(pred, dict):
            if true_cfg_scale > 1:
                pred = {k: neg_pred[k] + true_cfg_scale * (pred[k] - neg_pred[k]) for k in pred}

            patch_size = getattr(model, "piflow_patch_size", 2)
            img_packed = rearrange(
                img, "b (h w) c -> b c h w", h=piflow_spatial_h, w=piflow_spatial_w
            )
            img_unpacked = _unpack_latent_piflux2(img_packed, patch_size=patch_size).float()
            pred = {k: v.float() for k, v in pred.items()}

            sigma_t_src = piflow_sigmas[i].expand(img.shape[0])
            sigma_t_dst = piflow_sigmas_dst[i].expand(img.shape[0])
            policy = _GMFlowPolicy(pred, img_unpacked, sigma_t_src)
            if (
                piflow_gm_temperature is not None
                and i != len(timesteps) - 2
                and piflow_gm_temperature < 1.0
            ):
                policy.temperature_(piflow_gm_temperature)
            if callback is not None and getattr(callback, "tiny_vae", False):
                preview_x0 = (policy.denoising_output_x_0["logweights"].softmax(dim=1) * policy.denoising_output_x_0["means"]).sum(dim=1)
                preview_x0 = _pack_latent_piflux2(preview_x0, patch_size=patch_size)
                preview_x0 = rearrange(preview_x0, "b c h w -> b (h w) c").to(img.dtype)
            img_unpacked = _policy_rollout(
                img_unpacked,
                sigma_t_src,
                sigma_t_dst,
                total_substeps=piflow_substeps,
                policy=policy,
                shift=3.2,
            )
            del policy

            sigma_t_to = piflow_sigmas[i + 1]
            m = piflow_m_vals[i]
            alpha_t_to = 1 - sigma_t_to
            if not torch.allclose(m, torch.ones_like(m)):
                if piflow_generator is not None and img_unpacked.device.type == "cpu":
                    noise = torch.randn_like(img_unpacked, generator=piflow_generator)
                else:
                    noise = torch.randn(img_unpacked.shape, device=img_unpacked.device, dtype=img_unpacked.dtype)
                img_unpacked = (alpha_t_to + sigma_t_to * m) * img_unpacked + sigma_t_to * (
                    1 - m.square()
                ).clamp(min=0).sqrt() * noise

            img_packed = _pack_latent_piflux2(img_unpacked, patch_size=patch_size)
            img = rearrange(img_packed, "b c h w -> b (h w) c").to(img.dtype)
        else:
            if true_cfg_scale > 1:
                pred = cfg_predictions(pred, neg_pred, true_cfg_scale, t_vec)

            if callback is not None and getattr(callback, "tiny_vae", False):
                preview_x0 = img - t_curr * pred
            step_size = t_prev - t_curr
            if final_step_size_scale is not None and i == len(timesteps) - 2:
                step_size = step_size * final_step_size_scale
            img += step_size * pred

        if img_msk_latents is not None and i < masked_steps:
            latent_noise_factor = t_prev
            # noisy_image  = original_image_latents  * (1.0 - latent_noise_factor) + torch.randn_like(original_image_latents) * latent_noise_factor 
            noisy_image  = original_image_latents  * (1.0 - latent_noise_factor) + randn * latent_noise_factor 
            img  =  noisy_image * (1-img_msk_latents)  + img_msk_latents * img
            noisy_image = None
            if preview_x0 is not None:
                preview_x0 = original_image_latents * (1-img_msk_latents) + img_msk_latents * preview_x0

        if callback is not None:
            preview = unpack_latent(preview_x0 if preview_x0 is not None else img).transpose(0,1)
            callback(i, preview, False, preview_meta=preview_meta)
            del preview
        del preview_x0


    return img

def prepare_multi_ip(
    ae: AutoEncoder,
    img_cond_list: list,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
    pe: Literal["d", "h", "w", "o"] = "d",
    conditions_zero_start = False,
    set_cond_index = False,
    res_match_output = True,
    patch_size: int = 2    
    
) -> dict[str, Tensor]:

    assert pe in ["d", "h", "w", "o"]

    if img_cond_list == None: img_cond_list = []

    if not res_match_output:
        for i, img_cond in enumerate(img_cond_list):
            img_cond_list[i]= resizeinput(img_cond)

    ref_imgs = [
        ae.encode(
            (TVF.to_tensor(ref_img) * 2.0 - 1.0)
            .unsqueeze(0)
            .to(device, torch.float32)
        ).to(torch.bfloat16)
        for ref_img in img_cond_list
    ]

    img = get_noise( bs, target_height, target_width, device=device, dtype=torch.bfloat16, seed=seed)
    bs, c, h, w = img.shape
    # tgt img
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // patch_size, w // patch_size, 3, device=img.device)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // patch_size, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // patch_size, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    img_cond_seq = img_cond_seq_ids = None
    if conditions_zero_start:
        pe_shift_w = pe_shift_h = 0
    else:
        pe_shift_w, pe_shift_h = w // patch_size, h // patch_size
    for cond_no, ref_img in enumerate(ref_imgs):
        _, _, ref_h1, ref_w1 = ref_img.shape
        ref_img = rearrange(
            ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
        )
        if ref_img.shape[0] == 1 and bs > 1:
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        ref_img_ids1 = torch.zeros(ref_h1 // 2, ref_w1 // 2, 3, device=ref_img.device)
        if set_cond_index:
            ref_img_ids1[..., 0] = cond_no + 1
        h_offset = pe_shift_h if pe in {"d", "h"} else 0
        w_offset = pe_shift_w if pe in {"d", "w"} else 0
        ref_img_ids1[..., 1] = (
            ref_img_ids1[..., 1] + torch.arange(ref_h1 // 2, device=ref_img.device)[:, None] + h_offset
        )
        ref_img_ids1[..., 2] = (
            ref_img_ids1[..., 2] + torch.arange(ref_w1 // 2, device=ref_img.device)[None, :] + w_offset
        )
        ref_img_ids1 = repeat(ref_img_ids1, "h w c -> b (h w) c", b=bs)

        if target_width is None:
            target_width = 8 * ref_w1
        if target_height is None:
            target_height = 8 * ref_h1
        ref_img_ids1 = ref_img_ids1.to(device)
        if cond_no == 0:
            img_cond_seq, img_cond_seq_ids  = ref_img, ref_img_ids1
        else:
            img_cond_seq, img_cond_seq_ids  =  torch.cat([img_cond_seq, ref_img], dim=1), torch.cat([img_cond_seq_ids, ref_img_ids1], dim=1)


        # 更新pe shift
        pe_shift_h += ref_h1 // 2
        pe_shift_w += ref_w1 // 2

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "img_cond_seq": img_cond_seq,
        "img_cond_seq_ids": img_cond_seq_ids,
    }, target_height, target_width


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def patches_to_image(x: Tensor, height: int, width: int, patch_size: int) -> Tensor:
    tokens = x.transpose(1, 2)
    return F.fold(
        tokens,
        output_size=(height, width),
        kernel_size=patch_size,
        stride=patch_size,
    )
