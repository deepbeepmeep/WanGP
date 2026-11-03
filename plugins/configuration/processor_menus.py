# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Refresh the processor menus of an open generation page when its shared extension settings change."""
import copy

import gradio as gr

from postprocessing import audio_processors as audio, spatial_upsamplers as spatial, temporal_upsamplers as temporal
from shared.utils.gallery_sources import GALLERY_SOURCES


CONFIG_KEYS = (spatial.UPSAMPLER_CONFIG_KEY, temporal.TEMPORAL_UPSAMPLER_CONFIG_KEY, audio.AUDIO_PROCESSOR_CONFIG_KEY)
GENERATION_VALUES = ("spatial_upsampling", "temporal_upsampling", "spatial_postprocessing_parameters", "temporal_postprocessing_parameters", "postprocess_audio", "replace_voice_method")
FORM_CONTEXT = ("image_mode", "video_prompt_type", "image_prompt_type")
GENERATION_MENUS = ("spatial_upsampling_method", "spatial_upsampling_ratio", "temporal_upsampling_method", "temporal_upsampling_multiplier", "spatial_upsampler_extra", "postprocessing_parameter_media_outputs",
                    "replace_voice_col", "replace_voice_sample_row", "replace_voice_sample2_row", "postprocess_audio_prompt_col", "postprocess_audio_control_col", "postprocess_audio_source_col")
LATE_AUDIO = ("PP_late_audio_postprocess", "PP_late_audio_replace_voice_sample_row", "PP_late_audio_replace_voice_sample2_row",
              "PP_postprocess_audio", "PP_postprocess_audio_prompt_row", "PP_audio_source_row", "PP_replace_voice_sample_row", "PP_replace_voice_sample2_row")
GALLERY_VALUES = ("PP_spatial_upsampling", "PP_spatial_postprocessing_parameters", "PP_temporal_postprocessing_parameters", "PP_spatial_upsampler_help_target", "PP_temporal_upsampling", "video_info_tab")


def request_components(plugin):
    for name in (*GENERATION_VALUES, *FORM_CONTEXT, *GENERATION_MENUS, *LATE_AUDIO, *GALLERY_VALUES, "postprocessing_reference_images", "current_gallery_tab", "refresh_outputs", "select_media_outputs"):
        plugin.request_component(name)
    for name in ("get_current_model_settings", "generate_media_tab", "select_media"):
        plugin.request_global(name)


def bind(plugin, config_event):
    def snapshot():
        return {key: plugin.server_config.get(key, {}) for key in CONFIG_KEYS}

    previous_config = gr.State(copy.deepcopy(snapshot()))
    components = lambda names: [getattr(plugin, name) for name in names]
    # Use the same form builder as Extract Settings and model switching, sending only processor
    # controls to the browser. Other windows may have a prompt, media or other unsaved edits.
    menu_components = {control for item in components((*GENERATION_VALUES, *GENERATION_MENUS)) for control in (item if isinstance(item, list) else [item])}
    menu_components.discard(plugin.postprocessing_reference_images)
    indices = [i for i, control in enumerate(plugin.refresh_outputs) if control in menu_components]
    outputs = [previous_config, *(plugin.refresh_outputs[i] for i in indices), *components(LATE_AUDIO), *plugin.select_media_outputs]

    def refresh(previous, state, gallery_tab, audio_edit, remux, *values):
        current = snapshot()
        if current == previous:
            return (gr.skip(),) * len(outputs)
        replacements = spatial.method_preference_changes(previous, current)
        live = dict(zip((*GENERATION_VALUES, *FORM_CONTEXT), values))
        live["spatial_upsampling"] = spatial.apply_method_preference(live["spatial_upsampling"], replacements)
        model_settings = plugin.get_current_model_settings(state)
        if live["spatial_upsampling"] != values[0]:
            model_settings["spatial_upsampling"] = live["spatial_upsampling"]
        settings = {**model_settings, **live}
        form = plugin.generate_media_tab(update_form=True, state_dict=state, ui_defaults=settings)

        edit_choices, audio_edit = audio.late_audio_edit_menu(audio_edit)
        remux_choices, remux = audio.late_remux_menu(remux)
        source = GALLERY_SOURCES[int(gallery_tab)]
        late_spatial, *gallery_values = values[len(live):]
        gallery = plugin.select_media(state, gallery_tab, None, state["gen"].get(source.selected_key, -1), -1, source.id,
                                      spatial.apply_method_preference(late_spatial, replacements), *gallery_values, None)
        return (copy.deepcopy(current), *(form[i] for i in indices),
                gr.update(choices=edit_choices, value=audio_edit), *audio.audio_edit_refresh_updates(audio_edit),
                gr.update(choices=remux_choices, value=remux), *audio.late_remux_refresh_updates(remux), *gallery)

    config_event(refresh, inputs=[previous_config, plugin.state, plugin.current_gallery_tab, plugin.PP_late_audio_postprocess, plugin.PP_postprocess_audio,
                                  *components((*GENERATION_VALUES, *FORM_CONTEXT)), *components(GALLERY_VALUES)], outputs=outputs, show_progress="hidden")
