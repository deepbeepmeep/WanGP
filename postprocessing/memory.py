# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared profile 0 VRAM policy for postprocessors."""

MODE_KEY = "postprocessor_preload_mode"
PRELOAD_KEY = "postprocessor_preload_in_VRAM"
CHOICES = [("Minimum", "minimum"), ("Manual Choice", "manual"), ("Dynamic", "dynamic")]


def config_values(config):
    mode = config.get(MODE_KEY, "minimum")
    if mode not in {value for _, value in CHOICES}:
        mode = "minimum"
    try:
        preload = max(0, int(config.get(PRELOAD_KEY, 0)))
    except (ValueError, TypeError):
        preload = 0
    return mode, preload


def migrate_config(config):
    mode, preload = config_values(config)
    changed = config.get(MODE_KEY) != mode or config.get(PRELOAD_KEY) != preload
    config.update({MODE_KEY: mode, PRELOAD_KEY: preload})
    for key in ("audio_processors", "spatial_upsamplers", "temporal_upsamplers"):
        sections = config.get(key, {})
        if isinstance(sections, dict) and "persistence" in sections:
            sections.pop("persistence")
            changed = True
    return changed


def configure(kwargs, config):
    mode, preload = config_values(config)
    # MMGP combines this floor with each model's budget (including budgets set
    # after init_pipe and module _budget declarations), preserving whole models.
    kwargs["minimumBudget"] = max(kwargs.get("minimumBudget", 0), preload if mode == "manual" else 0)
    kwargs["preloadExperts"] = kwargs["minimumBudget"] > 0
    kwargs["autoPreload"] = mode == "dynamic"
    return 0


def minimal_vram(kwargs):
    """Whether the profile kwargs that init_pipe filled ask for the Minimum VRAM usage (no automatic preload, no VRAM preload): the state a
    processor keeps beside its weights (a KV cache) then stays in RAM, the GPU holding only what its current layers need."""
    return not kwargs["autoPreload"] and not kwargs["minimumBudget"]


def init_pipe(pipe, kwargs, profile=0, *, server_config):
    """Standalone postprocessor initialization when no app callback is supplied."""
    return configure(kwargs, server_config)


def create_config_ui(gr, config, *, lock_config=False):
    mode, preload = config_values(config)
    with gr.Row():
        usage = gr.Dropdown(choices=CHOICES, value=mode, label="Postprocessor VRAM Usage", interactive=not lock_config,
                            info="Minimum uses each processor's minimum. Manual Choice set maximum VRAM. Dynamic adapts preload to available VRAM.")
        amount = gr.Slider(0, max(40000, preload), value=preload, step=100, label="Postprocessor VRAM Preload (MB)", visible=mode == "manual", interactive=not lock_config)
    usage.change(fn=lambda value: gr.update(visible=value == "manual"), inputs=usage, outputs=amount, show_progress="hidden")
    return usage, amount
