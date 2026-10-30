import html
import json
import math
import re
from pathlib import Path

import ftfy
import torch
from accelerate import init_empty_weights
from mmgp import offload
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel

from models.wan.modules.vae import WanVAE
from shared.utils.phase_progress import check_abort, generation_progress, set_phase_status, text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.loras_mutipliers import get_model_switch_steps, update_loras_slists
from .audio_vae import AudioVAE
from .transformer import PrismTransformer


def preserve_precision(model):
    model._convertWeightsFloatTo = None
    model._model_dtype = torch.bfloat16
    for module in model.modules():
        module._lock_dtype = None
    return model.eval().requires_grad_(False)


def tie_encoder_embeddings(state, quantization_map, tied_weights_map):
    state['encoder.embed_tokens.weight'] = state['shared.weight']
    return state, quantization_map, tied_weights_map


class PrismPipeline:
    def __init__(self, checkpoints, encoder_checkpoint, tokenizer_folder, video_vae_checkpoint, audio_vae_checkpoint):
        self.device = torch.device('cuda')
        self.config = json.loads(Path(__file__).with_name('prism.json').read_text(encoding='utf-8'))
        with init_empty_weights(include_buffers=True):
            transformer = PrismTransformer(self.config)
            transformer2 = PrismTransformer(self.config)
            audio_vae = AudioVAE(self.config['audio_vae'])
        shared_modules = {}
        offload.load_model_data(transformer, checkpoints[0], modules=checkpoints[2:], return_shared_modules=shared_modules, writable_tensors=False, default_dtype=None, fp32_dtype=torch.bfloat16)
        offload.load_model_data(transformer2, checkpoints[1], modules=shared_modules, writable_tensors=False, default_dtype=None, fp32_dtype=torch.bfloat16)
        self.model, self.model2 = preserve_precision(transformer), preserve_precision(transformer2)
        self.text_encoder = preserve_precision(offload.fast_load_transformers_model(encoder_checkpoint, modelClass=UMT5EncoderModel,
            forcedConfigPath=str(Path(tokenizer_folder) / 'config.json'), writable_tensors=False, default_dtype=None, fp32_dtype=torch.bfloat16, preprocess_sd=tie_encoder_embeddings))
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_folder, local_files_only=True)
        self.vae = WanVAE(vae_pth=video_vae_checkpoint, dtype=torch.bfloat16, device=self.device)
        # Prism normalizes raw BF16 VAE latents in FP32, like its released pipeline.
        self.vae.mean = torch.tensor(self.config['video_vae']['latents_mean'], device=self.device, dtype=torch.float32)
        self.vae.std = torch.tensor(self.config['video_vae']['latents_std'], device=self.device, dtype=torch.float32)
        self.vae.scale = [self.vae.mean, self.vae.std.reciprocal()]
        preserve_precision(self.vae.model)
        offload.load_model_data(audio_vae, audio_vae_checkpoint, writable_tensors=False, default_dtype=None)
        self.audio_vae = preserve_precision(audio_vae)
        self.text_cache = TextEncoderCache()
        self._interrupt = False

    def _encode_text(self, prompts):
        cleaned = [re.sub(r'\s+', ' ', html.unescape(html.unescape(ftfy.fix_text(text))).strip()).strip() for text in prompts]
        tokens = self.tokenizer(cleaned, padding='max_length', max_length=512, truncation=True, return_attention_mask=True, return_tensors='pt').to(self.device)
        with text_encoding_progress(self.text_encoder.encoder.block, prompt_count=len(prompts)):
            encoded = self.text_encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state
        for i, length in enumerate(tokens.attention_mask.sum(-1).tolist()):
            encoded[i, length:] = 0
        check_abort()
        return list(encoded.unbind(0))

    def _condition(self, image, frames, tile_size):
        set_phase_status('Preparing Image Conditioning')
        # Padded image conditioning is encoded causally just as the released pipeline.
        # Its temporal chunks are implementation details, never a control-video progress bar.
        video = torch.zeros(3, frames, *image.shape[-2:], device=self.device, dtype=torch.bfloat16)
        video[:, 0] = image.to(device=self.device, dtype=torch.bfloat16)
        try:
            latent = self.vae.encode([video], tile_size=tile_size)[0]
        finally:
            self.vae.model.clear_cache()
        del video
        mask = latent.new_zeros((4, *latent.shape[1:]))
        mask[:, 0] = 1
        return torch.cat((mask, latent), 0)[None]

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, n_prompt, seed, sampling_steps, frame_num, height, width, fps, guide_scale,
                 image_start, alt_prompt='', callback=None, VAE_tile_size=0, set_progress_status=None, model_mode=0, shift=7.0,
                 sample_solver='euler', loras_slists=None, guide2_scale=5.0, guide3_scale=1.0, guide_phases=2,
                 switch_threshold=900, switch2_threshold=800, model_switch_phase=1, **kwargs):
        device = self.device
        lightx2v = sample_solver == 'lightx2v'
        guidance = (guide_scale, guide2_scale, guide3_scale)[:guide_phases]
        use_negative = lightx2v or any(scale != 1 for scale in guidance)
        prompts = [input_prompt, alt_prompt or input_prompt]
        if use_negative:
            prompts.append(n_prompt)
        embeddings = self.text_cache.encode(self._encode_text, prompts, device=device)
        contexts = [(embeddings[0][None], embeddings[1][None])]
        if use_negative:
            contexts.append((embeddings[2][None], embeddings[2][None]))
        generator = torch.Generator(device=device).manual_seed(seed)
        latent_frames = (frame_num - 1) // 4 + 1
        video = torch.randn((1, 16, latent_frames, height // 8, width // 8), generator=generator, device=device, dtype=torch.float32)
        condition = self._condition(image_start, frame_num, VAE_tile_size)
        audio_samples = int(self.audio_vae.sample_rate * frame_num / fps)
        audio_tokens = math.ceil(audio_samples / self.audio_vae.hop_length)
        audio = torch.randn((1, 128, audio_tokens), generator=generator, device=device, dtype=torch.float32)
        # Upstream pairs separately shifted schedules, then snaps Euler endpoints to
        # the scheduler's original 1000-step (shift=5) training grid.
        raw = torch.linspace(1, 0, sampling_steps + 1, device='cpu')[:-1]
        visual_sigma = shift * raw / (1 + (shift - 1) * raw)
        audio_sigma = 7 * raw / (1 + 6 * raw)
        training = torch.linspace(1, 0, 1001, device='cpu')[:-1]
        training = 5 * training / (1 + 4 * training)
        visual_steps = training[(training[:, None] - visual_sigma).abs().argmin(0)]
        audio_steps = training[(training[:, None] - audio_sigma).abs().argmin(0)]
        visual_steps = torch.cat((visual_steps, torch.zeros(1, device='cpu')))
        audio_steps = torch.cat((audio_steps, torch.zeros(1, device='cpu')))
        phase_switch, phase_switch2, _ = get_model_switch_steps(visual_sigma * 1000, guide_phases, model_switch_phase, switch_threshold, switch2_threshold)
        expert_switch = phase_switch2 if model_switch_phase == 2 else phase_switch
        if guide_phases == 1:
            expert_switch = int((visual_sigma * 1000 >= switch_threshold).sum())
            phase_switch = expert_switch
        if loras_slists is not None:
            for transformer in (self.model, self.model2):
                update_loras_slists(transformer, loras_slists, sampling_steps, phase_switch_step=phase_switch, phase_switch_step2=phase_switch2)
        set_phase_status('Denoising')
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=sampling_steps)
        audio_cache = {}
        try:
            for step in tqdm(range(sampling_steps), desc='Denoising'):
                check_abort()
                transformer = self.model2 if step >= expert_switch else self.model
                offload.set_step_no_for_lora(transformer, step)
                phase = int(step >= phase_switch) + int(step >= phase_switch2)
                scale = guidance[min(phase, len(guidance) - 1)]
                do_cfg = scale != 1 and (not lightx2v or guide_phases > 1 or step == 0)
                video_input = torch.cat((video, condition), 1)
                predictions = transformer(video_input, audio, contexts if do_cfg else contexts[:1], (visual_sigma[step:step+1] * 1000).to(device),
                    (audio_sigma[step:step+1] * 1000).to(device), fps, model_mode == 1, audio_cache if lightx2v else None)
                del video_input
                vp, ap = predictions[0]
                vp, ap = vp.float(), ap.float()
                if do_cfg:
                    vn, an = predictions[1]
                    vp = vn.float() + scale * (vp - vn.float())
                    ap = an.float() + scale * (ap - an.float())
                preview_video = None
                if callback is not None and getattr(callback, "tiny_vae", False):
                    # Predict x0 at model time; Euler endpoints use a separate snapped grid.
                    preview_video = video - visual_sigma[step].to(device) * vp
                video.add_(vp, alpha=float(visual_steps[step + 1] - visual_steps[step]))
                if lightx2v:
                    # FreeVideo Light recipe: conditional student's video K/V,
                    # four audio Euler updates with CFG 5, no extra video pass.
                    from .sampling import audio_substeps
                    next_sigma = audio_sigma[step + 1] if step + 1 < sampling_steps else audio_sigma.new_zeros(())
                    audio_substeps(transformer, audio, contexts, audio_sigma[step], next_sigma, training, audio_cache, fps)
                    audio_cache.clear()
                else:
                    audio.add_(ap, alpha=float(audio_steps[step + 1] - audio_steps[step]))
                del predictions, vp, ap
                if callback is not None:
                    callback(step, (preview_video if preview_video is not None else video)[0], False)
                del preview_video
                check_abort()
            del condition, contexts, embeddings
            video.mul_(self.vae.std[None, :, None, None, None]).add_(self.vae.mean[None, :, None, None, None])
            frames = self.vae.model.decode_to_cpu_uint8(video, scale=None, tile_size=VAE_tile_size, target_frames=frame_num, target_height=height, target_width=width, device=device)[0]
            del video
            set_phase_status('Decoding Audio')
            waveform = self.audio_vae.decode(audio.to(torch.bfloat16))[0, 0, :audio_samples].float()
            waveform = waveform.cpu().numpy()
            check_abort()
            return {'x': frames, 'audio': waveform, 'audio_sampling_rate': self.audio_vae.sample_rate}
        finally:
            audio_cache.clear()
            self.vae.model.clear_cache()
