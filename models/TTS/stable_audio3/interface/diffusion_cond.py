import gc
import numpy as np
import gradio as gr
import re
import subprocess
import torch
import torchaudio
import threading
import os, time, math

from einops import rearrange

from models.TTS.stable_audio3.interface.aeiou import audio_spectrogram_image
from models.TTS.stable_audio3.inference.distribution_shift import LogSNRShift, FluxDistributionShift, DistributionShift, IdentityDistributionShift
from models.TTS.stable_audio3.models.lora import has_lora
from models.TTS.stable_audio3.interface.reprompt import reprompt as _reprompt_fn, get_model as _reprompt_get_model, is_model_cached as _reprompt_is_model_cached
from shared.utils.subprocess_stdin import child_kwargs

stable_audio_3_model = None
sample_size = 5324800
sample_rate = 44100
n_loras = 0
_LENGTH_EXTRACT_RE = re.compile(r' Length: (\d+) seconds\.?\s*$')


# when using a prompt in a filename
def condense_prompt(prompt):
    pattern = r'[\\/:*?"<>|]'
    # Replace special characters with hyphens
    prompt = re.sub(pattern, '-', prompt)
    # set a character limit
    prompt = prompt[:150]
    # zero length prompts may lead to filenames (ie ".wav") which seem cause problems with gradio
    if len(prompt)==0:
        prompt = "_"
    return prompt

def generate_cond(
        prompt,
        negative_prompt=None,
        seconds_total=30,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_max=1000,
        cfg_interval_min=0.0,
        cfg_interval_max=1.0,
        cfg_rescale=0.0,
        cfg_norm_threshold=0.0,
        apg_scale=1.0,
        file_format="wav",
        file_naming="verbose",
        cut_to_seconds_total=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_maskstart=None,
        mask_maskend=None,
        inpaint_audio=None,
        init_audio_type="Init audio",
        inversion_steps=100,
        inversion_gamma=0.3,
        inversion_unconditional=False,
        duration_padding_sec=6.0,
        batch_size=1,
        dist_shift=None,
        *lora_args
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Parse per-LoRA controls from trailing args
    # Each LoRA has 5 controls: strength, interval_min, interval_max, layer_filter
    lora_configs = None
    if n_loras > 0 and len(lora_args) >= n_loras * 4:
        lora_configs = []
        for i in range(n_loras):
            off = i * 4
            strength = lora_args[off]
            interval_min = lora_args[off + 1]
            interval_max = lora_args[off + 2]
            layer_filter = lora_args[off + 3]
            stable_audio_3_model.set_lora_strength(strength, lora_index=i)
            lora_configs.append({
                "lora_index": i,
                "interval": (interval_min, interval_max),
                "layer_filter": layer_filter,
            })

    input_sample_size = sample_size

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        t = callback_info["t"]
        sigma = callback_info["sigma"]

        # Extract scalar from tensor if needed (samplers pass tensors to avoid GPU sync)
        if isinstance(t, torch.Tensor):
            t = t[0].item() if t.dim() > 0 else t.item()
        if isinstance(sigma, torch.Tensor):
            sigma = sigma[0].item() if sigma.dim() > 0 else sigma.item()

        log_snr = math.log(((1 - sigma) / sigma) + 1e-6)

        if (current_step - 1) % preview_every == 0:
            if stable_audio_3_model.model.pretransform is not None:
                denoised = stable_audio_3_model.model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f} logSNR={log_snr:.3f}"))

    if init_audio_type == "RF-Inversion":
        inversion_params = {
            "inversion_steps": inversion_steps,
            "inversion_gamma": inversion_gamma,
            "inversion_unconditional": inversion_unconditional,
            "inversion_cfg_scale": 1.0,
            "inversion_sigma_max": 1.0
        }
    else:
        inversion_params = None

    generate_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "duration": seconds_total,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "cfg_interval": (cfg_interval_min, cfg_interval_max),
        "lora_configs": lora_configs,
        "batch_size": batch_size,
        "sample_size": input_sample_size,
        "seed": seed,
        "sampler_type": sampler_type,
        "sigma_max": sigma_max,
        "init_audio": init_audio,
        "init_noise_level": init_noise_level,
        "callback": progress_callback if preview_every is not None else None,
        "scale_phi": cfg_rescale,
        "cfg_norm_threshold": cfg_norm_threshold,
        "apg_scale": apg_scale,
        "duration_padding_sec": duration_padding_sec,
        "dist_shift": dist_shift,
    }

    # If inpainting, send mask args
    # This will definitely change in the future
    if inpaint_audio is not None:
        generate_args.update({
            "inpaint_audio": inpaint_audio,
            "inpaint_mask_start_seconds": mask_maskstart,
            "inpaint_mask_end_seconds": mask_maskend,
        })

    audio = stable_audio_3_model.generate(**generate_args)

    # Filenaming convention
    prompt_condensed = condense_prompt(prompt)
    if file_naming=="verbose":
        basename = prompt_condensed
        if negative_prompt:
            basename += ".neg-%s" % condense_prompt(negative_prompt)
        basename += ".cfg%s" % (cfg_scale)
        if sigma_max not in [1.0, 100.0]:
            # this is a common parameter to tweak, if it's not a default value, put it in the verbose filename
            basename += ".smx%s" % sigma_max
        basename += ".%s" % seed
    elif file_naming=="prompt":
        basename = prompt_condensed
    else:
        # simple e.g. "output.wav"
        basename = "output"

    if file_format:
        filename_extension = file_format.split(" ")[0].lower()
    else:
        filename_extension = "wav"
    output_filename = "%s.%s" % (basename, filename_extension)
    output_wav = "%s.wav" % basename

    # Cut the extra silence off the end, if the user requested a smaller seconds_total
    if cut_to_seconds_total:
        audio = audio[:,:,:seconds_total*sample_rate]

    # Encode the audio to WAV format
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # save as wav file
    torchaudio.save(output_wav, audio, sample_rate)

    # If file_format is other than wav, convert to other file format
    cmd = ""
    if file_format == "m4a aac_he_v2 32k":
        # note: need to compile ffmpeg with --enable-libfdk_aac
        cmd = f"ffmpeg -i \"{output_wav}\" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 32k -y \"{output_filename}\""
    elif file_format == "m4a aac_he_v2 64k":
        cmd = f"ffmpeg -i \"{output_wav}\" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 64k -y \"{output_filename}\""
    elif file_format == "flac":
        cmd = f"ffmpeg -i \"{output_wav}\" -y \"{output_filename}\""
    elif file_format == "mp3 320k":
        cmd = f"ffmpeg -i \"{output_wav}\" -b:a 320k -y \"{output_filename}\""
    elif file_format == "mp3 128k":
        cmd = f"ffmpeg -i \"{output_wav}\" -b:a 128k -y \"{output_filename}\""
    elif file_format == "mp3 v0":
        cmd = f"ffmpeg -i \"{output_wav}\" -q:a 0 -y \"{output_filename}\""
    else: # wav
        pass
    if cmd:
        cmd += " -loglevel error" # make output less verbose in the cmd window
        subprocess.run(cmd, shell=True, check=True, **child_kwargs())

    # Let's look at a nice spectrogram too
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    # Asynchronously delete the files after returning the output file, so as to prevent clutter in the directory
    delete_files_async([output_wav, output_filename], 30)

    return (output_filename, [audio_spectrogram, *preview_images])

#  Asynchronously delete the given list of filenames after delay seconds. Sets up thread that sleeps for delay then deletes.
def delete_files_async(filenames, delay):
    def delete_files_after_delay(filenames, delay):
        time.sleep(delay)  # Wait for the specified delay
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)  # Delete the file
    threading.Thread(target=delete_files_after_delay, args=(filenames, delay)).start()

def create_sampling_ui(stable_audio_3_model, default_prompt=None):
    global n_loras
    diffusion_objective = stable_audio_3_model.model.diffusion_objective
    is_rf = diffusion_objective == "rectified_flow"
    is_rf_denoiser = diffusion_objective == "rf_denoiser" # includes ARC models

    # Extract default dist_shift params from model's sampling_dist_shift
    default_sampling_dist_shift = getattr(stable_audio_3_model.model, 'sampling_dist_shift', None)
    default_dist_shift_type = "LogSNR"
    default_logsnr_params = {"anchor_length": 2000, "anchor_logsnr": -6.2, "rate": 0.0, "logsnr_end": 2.0}
    default_flux_params = {"min_length": 256, "max_length": 4096, "alpha_min": 6.93, "alpha_max": 6.93}
    default_full_params = {"base_shift": 0.5, "max_shift": 1.15, "min_length": 256, "max_length": 4096}

    if isinstance(default_sampling_dist_shift, LogSNRShift):
        default_dist_shift_type = "LogSNR"
        default_logsnr_params = {
            "anchor_length": getattr(default_sampling_dist_shift, 'anchor_length', 2000),
            "anchor_logsnr": getattr(default_sampling_dist_shift, 'anchor_logsnr', -6.2),
            "rate": getattr(default_sampling_dist_shift, 'rate', 0.0),
            "logsnr_end": getattr(default_sampling_dist_shift, 'logsnr_end', 2.0),
        }
    elif isinstance(default_sampling_dist_shift, FluxDistributionShift):
        default_dist_shift_type = "Flux"
        default_flux_params = {
            "min_length": default_sampling_dist_shift.min_length,
            "max_length": default_sampling_dist_shift.max_length,
            "alpha_min": default_sampling_dist_shift.alpha_min,
            "alpha_max": default_sampling_dist_shift.alpha_max,
        }
    elif isinstance(default_sampling_dist_shift, DistributionShift):
        default_dist_shift_type = "Full"
        default_full_params = {
            "base_shift": default_sampling_dist_shift.base_shift,
            "max_shift": default_sampling_dist_shift.max_shift,
            "min_length": default_sampling_dist_shift.min_length,
            "max_length": default_sampling_dist_shift.max_length,
        }
    elif default_sampling_dist_shift is None:
        default_dist_shift_type = "None"

    has_seconds_total = True

    use_lora = has_lora(stable_audio_3_model.model)
    lora_names = getattr(stable_audio_3_model.model, 'lora_names', [])
    n_loras = len(lora_names)

    if default_prompt is None:
        default_prompt = ""

    _reprompt_model_id = "Qwen/Qwen3.5-2B"
    _reprompt_cached = _reprompt_is_model_cached(_reprompt_model_id)

    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(show_label=False, placeholder="Prompt", value=default_prompt)
            negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
        prompt_assistant_button = gr.Button(
            "Prompt Assistant" if _reprompt_cached else "Download Prompt Assistant (~4.2 GB)",
            scale=1
        )
        generate_button = gr.Button("Generate", variant='primary', scale=1)

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row(visible = True):
                # Timing controls
                seconds_total_slider = gr.Slider(minimum=0, maximum=sample_size//sample_rate, step=1, value=sample_size//sample_rate, label="Seconds total", visible=has_seconds_total)

            with gr.Row():
                # Steps slider
                if is_rf:
                    default_steps = 50
                elif is_rf_denoiser:
                    default_steps = 8

                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=default_steps, label="Steps")
                # CFG scale
                default_cfg_scale = 1.0 if is_rf_denoiser else 7.0
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=default_cfg_scale, label="CFG scale")

            # Per-LoRA controls (dynamic based on number of loaded LoRAs)
            lora_ui_inputs = []
            if use_lora and lora_names:
                for i, lora_name in enumerate(lora_names):
                    with gr.Accordion("LoRA {}: {}".format(i + 1, lora_name), open=(i == 0)):
                        with gr.Row():
                            strength = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=1.0, label="strength")
                        with gr.Row():
                            int_min = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Interval min")
                            int_max = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Interval max")
                            lyr_filt = gr.Textbox(label="Layer filter", placeholder="")
                        lora_ui_inputs.extend([strength, int_min, int_max, lyr_filt])

            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    # Seed
                    seed_textbox = gr.Number(label="Seed (set to -1 for random seed)", value=-1, precision=0)

                    cfg_interval_min_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG interval min")
                    cfg_interval_max_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=1.0, label="CFG interval max")

                with gr.Row():
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG rescale amount")
                    cfg_norm_threshold = gr.Slider(minimum=0.0, maximum=100, step=0.1, value=0.0, label="CFG norm threshold")
                    apg_scale_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1.0, label="APG scale", info="1.0=full APG, 0.0=vanilla CFG")

                with gr.Row():
                    # Sampler params
                    if is_rf:
                        sampler_types = ["euler", "rk4", "dpmpp"]
                        default_sampler_type = "euler"
                        sigma_max_max = 1.0
                        sigma_max_default = 1.0
                    elif is_rf_denoiser:
                        sampler_types = ["pingpong"]
                        default_sampler_type = "pingpong"
                        sigma_max_max = 1.0
                        sigma_max_default = 1.0
                    else:
                        sampler_types = ["dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast", "v-ddim", "v-ddim-cfgpp"]
                        default_sampler_type = "dpmpp-3m-sde"
                        sigma_max_max = 1000.0
                        sigma_max_default = 100.0

                    sampler_type_dropdown = gr.Dropdown(sampler_types, label="Sampler type", value=default_sampler_type)
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=sigma_max_max, step=0.1, value=sigma_max_default, label="Sigma max", visible=True)

                with gr.Row():
                    duration_padding_slider = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, value=6.0, label="Duration padding (sec)")

                def build_dist_shift(shift_type, p1, p2, p3, p4):
                    """Build dist_shift from type + 4 params (meaning depends on type)."""
                    if shift_type == "LogSNR":
                        return LogSNRShift(anchor_length=int(p1), anchor_logsnr=p2, rate=p3, logsnr_end=p4)
                    elif shift_type == "Flux":
                        return FluxDistributionShift(min_length=int(p1), max_length=int(p2), alpha_min=p3, alpha_max=p4)
                    elif shift_type == "Full":
                        return DistributionShift(base_shift=p1, max_shift=p2, min_length=int(p3), max_length=int(p4))
                    return IdentityDistributionShift()  # "None" = no shift

                dist_shift_state = gr.State(value=default_sampling_dist_shift)

                with gr.Row(visible=is_rf or is_rf_denoiser):
                    dist_shift_type_dropdown = gr.Dropdown(
                        ["LogSNR", "Flux", "Full", "None"],
                        label="Sampling schedule shift",
                        value=default_dist_shift_type,
                        info="Distribution shift applied to sampling timesteps"
                    )
                with gr.Row(visible=(is_rf or is_rf_denoiser) and default_dist_shift_type == "LogSNR") as logsnr_params_row:
                    logsnr_anchor_length_slider = gr.Slider(minimum=100, maximum=10000, step=100, value=default_logsnr_params["anchor_length"], label="Anchor length")
                    logsnr_anchor_logsnr_slider = gr.Slider(minimum=-12.0, maximum=0.0, step=0.1, value=default_logsnr_params["anchor_logsnr"], label="Anchor log-SNR")
                    logsnr_rate_slider = gr.Slider(minimum=-2.0, maximum=2.0, step=0.1, value=default_logsnr_params["rate"], label="Rate")
                    logsnr_end_slider = gr.Slider(minimum=-2.0, maximum=6.0, step=0.1, value=default_logsnr_params["logsnr_end"], label="log-SNR end")
                with gr.Row(visible=(is_rf or is_rf_denoiser) and default_dist_shift_type == "Flux") as flux_params_row:
                    flux_min_length_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=default_flux_params["min_length"], label="Min seq len")
                    flux_max_length_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=default_flux_params["max_length"], label="Max seq len")
                    flux_alpha_min_slider = gr.Slider(minimum=0.1, maximum=20.0, step=0.1, value=default_flux_params["alpha_min"], label="Alpha min")
                    flux_alpha_max_slider = gr.Slider(minimum=0.1, maximum=20.0, step=0.1, value=default_flux_params["alpha_max"], label="Alpha max")
                with gr.Row(visible=(is_rf or is_rf_denoiser) and default_dist_shift_type == "Full") as full_params_row:
                    full_base_shift_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.05, value=default_full_params["base_shift"], label="Base shift")
                    full_max_shift_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.05, value=default_full_params["max_shift"], label="Max shift")
                    full_min_length_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=default_full_params["min_length"], label="Min length")
                    full_max_length_slider = gr.Slider(minimum=1, maximum=10000, step=1, value=default_full_params["max_length"], label="Max length")

                # Per-type slider groups for wiring to state
                logsnr_sliders = [logsnr_anchor_length_slider, logsnr_anchor_logsnr_slider, logsnr_rate_slider, logsnr_end_slider]
                flux_sliders = [flux_min_length_slider, flux_max_length_slider, flux_alpha_min_slider, flux_alpha_max_slider]
                full_sliders = [full_base_shift_slider, full_max_shift_slider, full_min_length_slider, full_max_length_slider]
                all_dist_shift_inputs = [dist_shift_type_dropdown] + logsnr_sliders + flux_sliders + full_sliders

                def update_dist_shift_state(shift_type, *params):
                    """Route the 4 relevant params to build_dist_shift based on type."""
                    type_to_slice = {"LogSNR": params[0:4], "Flux": params[4:8], "Full": params[8:12]}
                    p = type_to_slice.get(shift_type, (0, 0, 0, 0))
                    return (
                        build_dist_shift(shift_type, *p),
                        gr.update(visible=((is_rf or is_rf_denoiser) and (shift_type == "LogSNR"))),
                        gr.update(visible=((is_rf or is_rf_denoiser) and (shift_type == "Flux"))),
                        gr.update(visible=((is_rf or is_rf_denoiser) and (shift_type == "Full"))),
                    )

                for component in all_dist_shift_inputs:
                    component.change(
                        update_dist_shift_state,
                        inputs=all_dist_shift_inputs,
                        outputs=[dist_shift_state, logsnr_params_row, flux_params_row, full_params_row],
                    )

            # Hidden state for batch_size (no UI control, but needed for function signature)
            batch_size_state = gr.State(value=1)

            with gr.Accordion("Output params", open=False):
                # Output params
                with gr.Row():
                    file_format_dropdown = gr.Dropdown(["wav", "flac", "mp3 320k", "mp3 v0", "mp3 128k", "m4a aac_he_v2 64k", "m4a aac_he_v2 32k"], label="File format", value="wav")
                    file_naming_dropdown = gr.Dropdown(["verbose", "prompt", "output.wav"], label="File naming", value="verbose") # ,"prompt","verbose"
                    preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Spec Preview Every")

                    cut_to_seconds_total_checkbox = gr.Checkbox(label="Cut to seconds total", value=True)
                    autoplay_checkbox = gr.Checkbox(label="Autoplay", value=False, elem_id="autoplay")
                    infinite_radio_checkbox = gr.Checkbox(label="Infinite Radio", value=False, elem_id="infinite-radio")
                    automatic_download_checkbox = gr.Checkbox(label="Auto Download", value=False, elem_id="automatic-download")

            # Default generation tab
            with gr.Accordion("Init audio", open=False):
                init_audio_input = gr.Audio(label="Init audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
                min_noise_level = 0.01
                max_noise_level = 1.0
                default_noise_level = 0.9 # roughly halfway style transfer values
                if is_rf:
                    choices = ["Init audio","RF-Inversion"]
                else:
                    choices = ["Init audio"]

                init_audio_type_radio = gr.Radio(label="Techniques", choices=choices, value=choices[0], visible=len(choices)>1)
                with gr.Column(visible=True) as interface_a:
                    init_noise_level_slider = gr.Slider(minimum=min_noise_level, maximum=max_noise_level, step=0.01, value=default_noise_level, label="Init noise level")
                with gr.Column(visible=False) as interface_b:
                    inversion_steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Inversion Steps")
                    inversion_gamma_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0, label="Gamma", visible=True)
                    inversion_unconditional_checkbox = gr.Checkbox(label="Unconditional", value=False)
                    gr.HTML("<div style='opacity: 0.5; padding: 0px'>For reproduction, try empty prompt, cfg 1, gamma .3<br>\
                        For prompt re-stylization, try cfg 1-7, gamma 0-.15, unconditional</div>")
                def init_audio_type_switch(choice):
                    return (
                        gr.update(visible=(choice == "Init audio")),
                        gr.update(visible=(choice == "RF-Inversion"))
                    )
                init_audio_type_radio.change(init_audio_type_switch, inputs=init_audio_type_radio, outputs=[interface_a, interface_b])

            with gr.Accordion("Inpainting", open=False):
                inpaint_audio_input = gr.Audio(label="Inpaint audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
                mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=sample_size//sample_rate, step=0.1, value=0, label="Mask Start (sec)")
                mask_maskend_slider = gr.Slider(minimum=0.0, maximum=sample_size//sample_rate, step=0.1, value=sample_size//sample_rate, label="Mask End (sec)")

                # Update inpainting slider ranges when seconds_total changes.
                # Only seconds_total is an input â€” reading the mask sliders here would cause
                # validation errors since their values may exceed the about-to-be-reduced maximum.
                def update_inpaint_sliders(seconds_total):
                    max_val = max(seconds_total, 1)
                    return (
                        gr.update(maximum=max_val),
                        gr.update(maximum=max_val, value=max_val),
                    )
                seconds_total_slider.change(update_inpaint_sliders, inputs=[seconds_total_slider], outputs=[mask_maskstart_slider, mask_maskend_slider])

            inputs = [
                prompt,
                negative_prompt,
                seconds_total_slider,
                cfg_scale_slider,
                steps_slider,
                preview_every_slider,
                seed_textbox,
                sampler_type_dropdown,
                sigma_max_slider,
                cfg_interval_min_slider,
                cfg_interval_max_slider,
                cfg_rescale_slider,
                cfg_norm_threshold,
                apg_scale_slider,
                file_format_dropdown,
                file_naming_dropdown,
                cut_to_seconds_total_checkbox,
                init_audio_input,
                init_noise_level_slider,
                mask_maskstart_slider,
                mask_maskend_slider,
                inpaint_audio_input,
                init_audio_type_radio,
                inversion_steps_slider,
                inversion_gamma_slider,
                inversion_unconditional_checkbox,
                duration_padding_slider,
                batch_size_state,
                dist_shift_state,
            ] + lora_ui_inputs

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False))
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

            send_to_inpaint_button = gr.Button("Send to inpaint audio", scale=1)
            send_to_inpaint_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[inpaint_audio_input])

    generate_button.click(fn=generate_cond,
        inputs=inputs,
        outputs=[
            audio_output,
            audio_spectrogram_output
        ],
        api_name="generate")

    def _prompt_assistant_or_download(text, progress=gr.Progress(track_tqdm=True)):
        if not _reprompt_is_model_cached(_reprompt_model_id):
            _reprompt_get_model(_reprompt_model_id)
            return text, gr.update(), gr.update(value="Prompt Assistant")
        _, result, _ = _reprompt_fn(text, "Auto", "", _reprompt_model_id, 128, 1.11)
        m = _LENGTH_EXTRACT_RE.search(result)
        if m:
            max_seconds = sample_size // sample_rate
            seconds = min(int(m.group(1)), max_seconds)
            result = result[:m.start()]
        else:
            seconds = gr.update()
        return result, seconds, gr.update()

    prompt_assistant_button.click(
        fn=_prompt_assistant_or_download,
        inputs=[prompt],
        outputs=[prompt, seconds_total_slider, prompt_assistant_button],
        concurrency_limit=1,
    )

def create_diffusion_cond_ui(model, gradio_title="", default_prompt=None):
    global sample_size, sample_rate, stable_audio_3_model

    sample_size = model.model_config["sample_size"]
    sample_rate = model.model_config["sample_rate"]
    stable_audio_3_model = model


    js ="""function run_javascript_on_page_load(){
        const generateBtn = Array.from(document.querySelectorAll('button'))
            .find(btn => btn.innerText.trim() === 'Generate');
        function getAudioOutputPlayer () {
            return [...document.querySelectorAll('label')].find(label => label.textContent.trim() === 'Output audio')?.parentElement.querySelector('audio');
        }
        const infiniteRadio = document.querySelector('#infinite-radio input[type="checkbox"]');
        const autoplay = document.querySelector('#autoplay input[type="checkbox"]');
        const automaticDownload = document.querySelector('#automatic-download input[type="checkbox"]');
        let radioAutoStart = false;
        let listenersSetup = false;
        const setupListeners = () => {
            const audioEl = getAudioOutputPlayer();
            if (!audioEl) return;
            audioEl.addEventListener('loadedmetadata', () => {
                if(automaticDownload.checked){
                    downloadAudio(audioEl);
                }
                if(autoplay.checked || radioAutoStart){
                    audioEl.play();
                    radioAutoStart = false;
                }
                if(infiniteRadio.checked){
                    audioEl.addEventListener('timeupdate', function checkAudioEnd() {
                        // Can set window.headstart (seconds) in the dev console if you want to start generating before the song is over
                        let headstart = 1;
                        if(window.headstart) headstart = window.headstart;
                        if (audioEl.duration - audioEl.currentTime <= headstart) {
                            generateBtn.click();
                            radioAutoStart = true;
                            audioEl.removeEventListener('timeupdate', checkAudioEnd);
                        }
                    });
                }
            });
            listenersSetup = true;
        };
        generateBtn.addEventListener('click', () => {
            if(listenersSetup) return;
            const interval = setInterval(() => {
                console.log("...")
                const audioEl = document.querySelector('audio');
                if (audioEl?.src && audioEl.src !== window.location.href) {
                    setupListeners();
                    clearInterval(interval);
                }
            }, 100);
        });
        // Respond to >> button on MacBookPro and on steering wheel during CarPlay
        if ('mediaSession' in navigator) {
            navigator.mediaSession.setActionHandler('nexttrack', () => generateBtn.click());
            navigator.mediaSession.setActionHandler('play', () => getAudioOutputPlayer()?.play());
            navigator.mediaSession.setActionHandler('pause', () => getAudioOutputPlayer()?.pause());
        }
        // Automatic Download
        function downloadAudio(audioEl) {
            const audioSrc = audioEl.src;
            const link = document.createElement('a');
            link.href = audioSrc;
            link.download = audioSrc.substring(audioSrc.lastIndexOf('/') + 1);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
    """

    with gr.Blocks() as ui:
        ui._sao_js = js
        ui._sao_theme = gr.themes.Base()
        if gradio_title:
            gr.Markdown("### %s" % gradio_title)
        with gr.Tab("Generation"):
            create_sampling_ui(model, default_prompt=default_prompt)

        # JavaScript to autoplay audio immediately after generation (if autoplay enabled)
    return ui

