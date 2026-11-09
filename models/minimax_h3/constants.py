H3_PHASE_2_NOISE_LEVEL_START_DEFAULT = 0.9035
H3_REGENERATE_LORA = "open_h3_regenerate_2k_lora.safetensors"
# Open-H3-Regenerate-2K-IC: a Ref2VA LoRA that regenerates the phase-1 draft, kept in context as a reference video, at the phase-2 size
H3_REGENERATE_IC_LORA = "open_h3_regenerate_2k_ic_lora_bf16.safetensors"
H3_REGENERATE_IC_TASK = ("Enhance <Video 1> into a high-quality version of the same video. Preserve the scene, objects, camera motion, timing, "
                         "and composition of every frame. Restore fine details and remove degradation. Do not continue or change the scene.")
# Video Rebirth's HyperFlow 8-step LoRA, repacked by WanGP with per-(t, r) AdaLN offsets in place of its time-embedder LoRAs
H3_HYPERFLOW_LORA = "minimax_h3_hyperflow_8step_lora_bf16.safetensors"
H3_AUDIO_REFINEMENT_SETTING = "audio_refinement"
H3_AUDIO_REFINEMENT_DENOISE = 0.5
H3_AUDIO_REFINEMENT_STEPS = 6

# Retain the latent-handoff experiment for further investigation. It worsened the
# first outpainting join; False restores pixel conditioning and 1 + 17n overlaps.
H3_CONTROL_LATENT_CONTINUATION = False

H3_MASK_MODE_SETTING = "h3_mask_mode"
H3_MASK_MODE_SHARED_TIMESTEP = "shared_timestep"
H3_MASK_MODE_GROUPED_ROWS = "grouped_rows"
H3_MASK_MODE_DEFAULT = H3_MASK_MODE_GROUPED_ROWS
H3_MASK_MODES = (H3_MASK_MODE_SHARED_TIMESTEP, H3_MASK_MODE_GROUPED_ROWS)


def h3_grouped_masking_enabled(custom_settings):
    mode = H3_MASK_MODE_DEFAULT if custom_settings is None else custom_settings.get(H3_MASK_MODE_SETTING, H3_MASK_MODE_DEFAULT)
    if mode not in H3_MASK_MODES:
        raise ValueError(f"Unsupported MiniMax H3 mask denoising mode {mode!r}")
    return mode == H3_MASK_MODE_GROUPED_ROWS
