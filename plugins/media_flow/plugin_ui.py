# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import html
import io
from pathlib import Path

import gradio as gr
from PIL import Image

from postprocessing import parameter_ui
from shared.gradio.local_file_picker import IMAGE_FILE_EXTENSIONS, LocalFilePickerTextbox, VIDEO_FILE_EXTENSIONS

from . import common
from . import constants as ui_constants
from . import process_catalog as catalog
from . import prompt_schedule as prompts
from . import status_ui
from . import ui_help
from .form_controller import FormComponentValues, create_form_controller
from .process_library import ProcessLibrary
from .process_runner import ProcessRunner, plugin_active_job, remember_status, request_stop, reset_live_chunk_status


def create_config_ui(self, api_session):
    if catalog.PROCESS_DEFINITIONS_ERROR is not None:
        with gr.Blocks() as plugin_blocks:
            gr.Markdown(f"Process settings configuration error: {html.escape(catalog.PROCESS_DEFINITIONS_ERROR)}")
        return plugin_blocks
    get_model_def = self.get_model_def
    get_lora_dir = self.get_lora_dir
    get_base_model_type = self.get_base_model_type
    library = ProcessLibrary(get_model_def=get_model_def, get_lora_dir=get_lora_dir, get_base_model_type=get_base_model_type)
    output_resolution_choices = ui_constants.output_resolution_choices(self.server_config)
    source_audio_track_choices = ui_constants.SOURCE_AUDIO_TRACK_CHOICES
    form_controller = create_form_controller(library, get_model_def, self.server_config)
    form_event_options = {"queue": True, "concurrency_limit": 1, "concurrency_id": f"media_flow_form_{id(self)}", "trigger_mode": "always_last", "show_progress": "hidden"}
    saved_mediaflow_settings = catalog.ensure_mediaflow_settings_migrated(catalog.load_saved_mediaflow_settings(), library.media_kind_for_user_ref)
    default_media_kind = catalog.current_media_kind(saved_mediaflow_settings)
    default_batch_mode = catalog.current_batch_mode(saved_mediaflow_settings)
    saved_ui_settings = catalog.get_last_ui_settings(saved_mediaflow_settings, default_media_kind, default_batch_mode)
    initial_user_refs = catalog.get_saved_user_settings_refs(saved_mediaflow_settings, default_media_kind)

    def _slot_settings_with_active_source(slot_settings: dict, batch_mode_value: str) -> dict:
        active_settings = dict(slot_settings or {})
        if catalog.normalize_batch_mode(batch_mode_value) == "batch":
            active_settings["source_path"] = str(active_settings.get("batch_source_path") or active_settings.get("source_path") or "").strip()
        return active_settings

    initial_form = form_controller.build_initial_form(_slot_settings_with_active_source(saved_ui_settings, default_batch_mode), default_media_kind, self.state.value, initial_user_refs)
    default_model_type = initial_form.model_type
    default_process_choices = initial_form.process_choices
    default_process_name = initial_form.process_name
    default_state = initial_form.form_state
    default_batch_name = str(saved_ui_settings.get("batch_name") or "").strip()
    default_batch_source_path = str(saved_ui_settings.get("batch_source_path") or "").strip()
    default_preserve_artifact_count = catalog.normalize_preserve_artifact_count(saved_ui_settings.get(catalog.PRESERVE_ARTIFACTS_STORAGE_KEY))
    default_preview_enabled = catalog.normalize_preview_enabled(saved_ui_settings.get(catalog.PREVIEW_OUTPUT_STORAGE_KEY))
    initial_spatial_parameters = library.spatial_upsampler_parameters(default_process_name, self.state.value, initial_user_refs)
    initial_spatial_parameter_values = {parameter_ui.LEGACY_NAMES.get(str(name), str(name)): value for name, value in dict(saved_ui_settings.get("spatial_upsampler_parameters") or {}).items()}
    initial_image_process = default_media_kind == "image"
    active_job = plugin_active_job(self)  # shared with the plugin's agent actions: one process at a time, stoppable from either side
    active_job.update({catalog.PRESERVE_ARTIFACTS_STORAGE_KEY: default_preserve_artifact_count, catalog.PREVIEW_OUTPUT_STORAGE_KEY: default_preview_enabled})
    preview_state = {"image": None}
    ui_skip = object()
    initial_status_html = status_ui.render_process_status_html("Idle", "Waiting to start...") if initial_image_process else status_ui.render_chunk_status_html(0, 0, 0, "Idle", "Waiting to start...")
    initial_output_html = status_ui.render_output_file_html("")
    last_html = {"status": initial_status_html, "batch_status": "", "output": initial_output_html}

    def _copy_preview_image(image):
        if image is None:
            return None
        if isinstance(image, dict):
            image = image.get("path") or image.get("name") or image.get("value")
        if isinstance(image, (str, Path)):
            with Image.open(image) as preview:
                image = preview.copy()
        elif isinstance(image, Image.Image):
            image = image.copy()
        else:
            return None
        try:
            image.filename = ""
        except (AttributeError, TypeError):
            pass
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        with Image.open(buffer) as preview:
            return preview.copy()

    def refresh_preview(_refresh_id):
        if not catalog.normalize_preview_enabled(active_job.get(catalog.PREVIEW_OUTPUT_STORAGE_KEY)):
            return None
        return _copy_preview_image(preview_state["image"])

    def _button_update(label: str, enabled: bool | None):
        return gr.skip() if enabled is None else gr.update(value=label, interactive=enabled)

    def _html_value(value, fallback: str = "") -> str:
        if isinstance(value, dict) and value.get("__type__") == "update":
            value = value.get("value", fallback)
        return str(value or fallback or "")

    def _html_update(value, fallback: str = ""):
        return gr.update(value=_html_value(value, fallback))

    def _stopping_status_update():
        last_html["status"] = status_ui.render_process_status_html("Stopping", "Stopping current processing job...")
        return _html_update(last_html["status"])

    def _ui_update(status=ui_skip, output=ui_skip, preview_refresh=ui_skip, *, batch_status=ui_skip, start_enabled: bool | None = None, abort_enabled: bool | None = None, batch_name_value=ui_skip, gallery_refresh=ui_skip):
        remember_status(active_job, status, batch_status)
        if status is not ui_skip:
            last_html["status"] = _html_value(status)
        if batch_status is not ui_skip:
            last_html["batch_status"] = _html_value(batch_status)
        if output is not ui_skip:
            last_html["output"] = status_ui.render_output_file_html(_html_value(output))
        status_update = last_html["status"]
        batch_status_update = last_html["batch_status"]
        output_update = last_html["output"]
        preview_update = gr.skip() if preview_refresh is ui_skip or not catalog.normalize_preview_enabled(active_job.get(catalog.PREVIEW_OUTPUT_STORAGE_KEY)) else preview_refresh
        start_update = _button_update("Start Process", start_enabled)
        abort_update = _button_update("Stop", abort_enabled)
        batch_name_update = gr.skip() if batch_name_value is ui_skip else gr.update(value=batch_name_value)
        gallery_refresh_update = gr.skip() if gallery_refresh is ui_skip else gallery_refresh
        return status_update, batch_status_update, output_update, preview_update, start_update, abort_update, batch_name_update, gallery_refresh_update

    def _info_exit(message: str, *, output=ui_skip, total_chunks: int = 1, completed_chunks: int = 0, current_chunk: int = 1, continued: bool = False):
        gr.Info(str(message or "").strip())
        return _ui_update(status_ui.render_chunk_status_html(total_chunks, completed_chunks, current_chunk, "Info", str(message or "").strip(), continued=continued), output, ui_skip, start_enabled=True, abort_enabled=False)

    def _add_user_process_link(process_value: str, media_kind: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None):
        media_kind = catalog.normalize_media_kind(media_kind)
        if str(process_value or "").strip() == ui_constants.NO_USER_SETTINGS_VALUE:
            raise gr.Error("No user settings are available to add.")
        process_definition = library.process_definition(process_value, main_state, user_refs)
        if process_definition is None:
            raise gr.Error("The selected user settings file could not be found.")
        if catalog.process_definition_media_kind(process_definition) != media_kind:
            raise gr.Error(f"The selected user settings file is not a {media_kind} process.")
        problems = library.validate_user_process_definition(process_definition)
        if len(problems) > 0:
            gr.Info(library.format_user_process_validation_error(process_definition, problems))
            return (gr.skip(),) * 9
        ref = catalog.normalize_user_settings_ref(process_definition.get("ref"))
        if len(ref) == 0:
            raise gr.Error("The selected user settings file could not be linked.")
        refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs}, media_kind)
        model_label = library.model_type_label(library.process_definition_model_type(process_definition))
        if ref.casefold() not in {item.casefold() for item in refs}:
            refs.append(ref)
            catalog.store_user_settings_refs(refs, media_kind)
            gr.Info(f'User settings "{process_definition.get("name")}" have been added for {model_label}.')
        else:
            gr.Info(f'User settings "{process_definition.get("name")}" are already linked for {model_label}.')
        process_choices, selected = library.current_user_settings_choices(media_kind, main_state, main_lset_name)
        if str(process_value or "").strip() in {value for _label, value in process_choices}:
            selected = str(process_value or "").strip()
        return (
            refs,
            gr.update(choices=library.model_type_choices(media_kind, refs), value=ui_constants.ADD_USER_SETTINGS_MODEL_TYPE),
            gr.update(choices=process_choices, value=selected),
            form_controller.user_settings_hint_update(process_choices),
            selected,
            *form_controller.settings_action_updates(ui_constants.ADD_USER_SETTINGS_MODEL_TYPE, selected),
        )

    def _delete_user_process_link(memory_state: dict | None, media_kind: str, batch_mode_value: str, process_value: str, main_state: dict | None, user_refs: list[str] | None, source_path: str, source_image_path: str, batch_source_path: str, batch_image_source_path: str):
        media_kind = catalog.normalize_media_kind(media_kind)
        batch_mode_value = catalog.normalize_batch_mode(batch_mode_value)
        process_value = str(process_value or "").strip()
        ref = catalog.user_process_ref_from_value(process_value)
        if len(ref) == 0:
            raise gr.Error("Choose a linked user settings process to remove.")
        old_refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs}, media_kind)
        deleted_definition = library.build_user_process_definition(ref)
        deleted_model_type = library.process_definition_model_type(deleted_definition)
        new_refs = [item for item in old_refs if item.casefold() != ref.casefold()]
        catalog.store_user_settings_refs(new_refs, media_kind)
        model_type, next_process_name, process_choices = library.select_after_user_process_delete(media_kind, process_value, deleted_model_type, old_refs, new_refs)
        active_source_path = _active_source_value(media_kind, batch_mode_value, source_path, source_image_path, batch_source_path, batch_image_source_path)
        restored = form_controller.restore_state(memory_state, next_process_name, active_source_path, main_state, new_refs, media_kind, batch_mode_value)
        gr.Info(f'Removed user settings "{Path(ref).stem}".')
        action_updates = form_controller.settings_action_updates(model_type, next_process_name)
        return (
            new_refs,
            gr.update(choices=library.model_type_choices(media_kind, new_refs), value=model_type),
            gr.update(choices=process_choices, value=next_process_name),
            form_controller.user_settings_hint_update(process_choices),
            next_process_name,
            *action_updates,
            *_active_source_outputs(media_kind, batch_mode_value, restored[0]),
            *restored[1:],
        )

    model_type_choices = initial_form.model_type_choices
    initial_process_form_memory = {}
    for media_kind in catalog.VALID_MEDIA_KINDS:
        slot_refs = initial_user_refs if media_kind == default_media_kind else catalog.get_saved_user_settings_refs(saved_mediaflow_settings, media_kind)
        for batch_mode_value in catalog.VALID_BATCH_MODES:
            slot_settings = catalog.get_last_ui_settings(saved_mediaflow_settings, media_kind, batch_mode_value)
            slot_form = form_controller.build_initial_form(_slot_settings_with_active_source(slot_settings, batch_mode_value), media_kind, self.state.value, slot_refs)
            initial_process_form_memory[form_controller.memory_key(slot_form.process_name, media_kind, batch_mode_value)] = slot_form.form_state.to_dict()
            initial_process_form_memory[form_controller.memory_selection_key(media_kind, batch_mode_value)] = slot_form.process_name

    process_runner = ProcessRunner(
        plugin=self,
        api_session=api_session,
        library=library,
        get_model_def=get_model_def,
        active_job=active_job,
        preview_state=preview_state,
        ui_skip=ui_skip,
        ui_update=_ui_update,
        info_exit=_info_exit,
        reset_live_chunk_status=reset_live_chunk_status,
    )

    def stop_process():
        job = active_job.get("job")
        job_running = job is not None and not job.done
        if request_stop(active_job):
            if job_running:
                common.plugin_info("Stopping current processing job...")
            return _stopping_status_update(), _html_update(last_html["batch_status"]), gr.update(value="Start Process", interactive=False), gr.update(value="Stop", interactive=False)
        return _html_update(last_html["status"]), _html_update(last_html["batch_status"]), gr.update(value="Start Process", interactive=True), gr.update(value="Stop", interactive=False)

    def _set_preserve_artifact_count(value):
        normalized = catalog.normalize_preserve_artifact_count(value)
        active_job[catalog.PRESERVE_ARTIFACTS_STORAGE_KEY] = normalized
        return gr.update(value=normalized)

    def _set_preview_enabled(value):
        enabled = catalog.normalize_preview_enabled(value)
        active_job[catalog.PREVIEW_OUTPUT_STORAGE_KEY] = enabled
        if not enabled:
            preview_state["image"] = None
        return gr.update(value=enabled), gr.update(visible=enabled, value=None) if not enabled else gr.update(visible=True)

    def _active_source_value(media_kind, batch_mode_value, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value):
        media_kind = catalog.normalize_media_kind(media_kind)
        batch_mode_value = catalog.normalize_batch_mode(batch_mode_value)
        if batch_mode_value == "batch":
            return batch_image_source_path_value if media_kind == "image" else batch_source_path_value
        return source_image_path_value if media_kind == "image" else source_path_value

    def _active_source_outputs(media_kind, batch_mode_value, source_value):
        media_kind = catalog.normalize_media_kind(media_kind)
        batch_mode_value = catalog.normalize_batch_mode(batch_mode_value)
        source_path_update = source_value if batch_mode_value == "single" and media_kind == "video" else gr.skip()
        source_image_path_update = source_value if batch_mode_value == "single" and media_kind == "image" else gr.skip()
        batch_source_path_update = source_value if batch_mode_value == "batch" and media_kind == "video" else gr.skip()
        batch_image_source_path_update = source_value if batch_mode_value == "batch" and media_kind == "image" else gr.skip()
        return source_path_update, source_image_path_update, batch_source_path_update, batch_image_source_path_update

    def _expand_active_source_result(result, media_kind, batch_mode_value, source_index):
        result = tuple(result)
        return (*result[:source_index], *_active_source_outputs(media_kind, batch_mode_value, result[source_index]), *result[source_index + 1:])

    def _form_values(media_kind, batch_mode_value, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        source_value = _active_source_value(media_kind, batch_mode_value, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value)
        return FormComponentValues(source_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)

    def _batch_name_memory_key(media_kind, batch_mode_value):
        return f"{catalog.normalize_media_kind(media_kind)}/{catalog.normalize_batch_mode(batch_mode_value)}/__batch_name__"

    def _store_batch_name(memory_state, media_kind, batch_mode_value, batch_name_value):
        updated_memory = dict(memory_state) if isinstance(memory_state, dict) else {}
        updated_memory[_batch_name_memory_key(media_kind, batch_mode_value)] = str(batch_name_value or "").strip()
        return updated_memory

    def _remembered_batch_name(memory_state, media_kind, batch_mode_value, saved_ui_settings):
        if isinstance(memory_state, dict):
            key = _batch_name_memory_key(media_kind, batch_mode_value)
            if key in memory_state:
                return str(memory_state.get(key) or "").strip()
        return str((saved_ui_settings or {}).get("batch_name") or "").strip()

    def _store_memory(memory_state, current_process_name, media_kind, batch_mode_value, main_state, refs, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, batch_name_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        values = _form_values(media_kind, batch_mode_value, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        updated_memory = form_controller.store_memory(memory_state, current_process_name, main_state, refs, values, media_kind, batch_mode_value)
        return _store_batch_name(updated_memory, media_kind, batch_mode_value, batch_name_value)

    def _change_process_model_type(memory_state, current_process_name, media_kind, batch_mode_value, next_model_type, main_state, main_lset_name, refs, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        values = _form_values(media_kind, batch_mode_value, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        return _expand_active_source_result(form_controller.change_process_model_type(memory_state, current_process_name, media_kind, batch_mode_value, next_model_type, main_state, main_lset_name, refs, values), media_kind, batch_mode_value, 8)

    def _change_process_name(memory_state, current_process_name, media_kind, batch_mode_value, next_process_name, process_model_type_value, main_state, refs, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        values = _form_values(media_kind, batch_mode_value, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        return _expand_active_source_result(form_controller.change_process_name(memory_state, current_process_name, media_kind, batch_mode_value, next_process_name, process_model_type_value, main_state, refs, values), media_kind, batch_mode_value, 5)

    def _refresh_from_main(refresh_id, memory_state, current_process_name, media_kind, batch_mode_value, process_model_type_value, main_state, main_lset_name, refs, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value):
        values = _form_values(media_kind, batch_mode_value, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        return _expand_active_source_result(form_controller.refresh_from_main(refresh_id, memory_state, current_process_name, media_kind, batch_mode_value, process_model_type_value, main_state, main_lset_name, refs, values), media_kind, batch_mode_value, 9)

    def _slot_form(memory_state, media_kind, batch_mode_value, main_state, refs):
        saved_settings = catalog.load_saved_mediaflow_settings()
        media_kind = catalog.normalize_media_kind(media_kind)
        batch_mode_value = catalog.normalize_batch_mode(batch_mode_value)
        slot_refs = catalog.get_saved_user_settings_refs(saved_settings, media_kind) if refs is None else list(refs or [])
        saved_ui = catalog.get_last_ui_settings(saved_settings, media_kind, batch_mode_value)
        remembered_process = form_controller.remembered_process_name(memory_state, media_kind, batch_mode_value, main_state, slot_refs)
        if len(remembered_process) > 0:
            remembered_definition = library.process_definition(remembered_process, main_state, slot_refs)
            saved_ui = {**saved_ui, "process_name": remembered_process, "process_model_type": library.process_definition_model_type(remembered_definition)}
        form = form_controller.build_initial_form(_slot_settings_with_active_source(saved_ui, batch_mode_value), media_kind, main_state, slot_refs)
        return slot_refs, saved_ui, form, form_controller.restore_state(memory_state, form.process_name, "", main_state, slot_refs, media_kind, batch_mode_value)

    def _change_batch_mode(memory_state, current_process_name, media_kind, current_batch_mode, main_state, main_lset_name, refs, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, batch_name_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value, evt: gr.SelectData):
        next_batch_mode = "batch" if evt.index == 1 else "single"
        values = _form_values(media_kind, current_batch_mode, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, target_ratio_value, chunk_size_value, overlap_value, start_value, end_value)
        updated_memory = form_controller.store_memory(memory_state, current_process_name, main_state, refs, values, media_kind, current_batch_mode)
        updated_memory = _store_batch_name(updated_memory, media_kind, current_batch_mode, batch_name_value)
        _slot_refs, slot_settings, next_form, restored = _slot_form(updated_memory, media_kind, next_batch_mode, main_state, refs)
        return (
            updated_memory,
            next_batch_mode,
            gr.update(choices=next_form.model_type_choices, value=next_form.model_type),
            gr.update(choices=next_form.process_choices, value=next_form.process_name),
            form_controller.user_settings_hint_update(next_form.process_choices),
            next_form.process_name,
            *form_controller.settings_action_updates(next_form.model_type, next_form.process_name),
            *_active_source_outputs(media_kind, next_batch_mode, restored[0]),
            *restored[1:],
            gr.update(value=_remembered_batch_name(updated_memory, media_kind, next_batch_mode, slot_settings)),
            gr.update(visible=next_batch_mode == "batch"),
            "",
        )

    def _change_media_kind(memory_state, current_process_name, next_media_kind, batch_mode_value, main_state, refs, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, batch_name_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, chunk_size_value, overlap_value, start_value, end_value):
        next_media_kind = catalog.normalize_media_kind(next_media_kind)
        current_process_media_kind = library.media_kind_for_process(current_process_name, main_state, refs)
        next_batch_mode = catalog.normalize_batch_mode(batch_mode_value)
        current_memory = memory_state.get(form_controller.memory_key(current_process_name, current_process_media_kind, next_batch_mode)) if isinstance(memory_state, dict) else None
        remembered_target_ratio = current_memory.get("target_ratio") if isinstance(current_memory, dict) else None
        values = _form_values(current_process_media_kind, next_batch_mode, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value, process_strength_value, output_path_value, prompt_value, continue_value, source_audio_track_value, output_resolution_value, remembered_target_ratio, chunk_size_value, overlap_value, start_value, end_value)
        updated_memory = form_controller.store_memory(memory_state, current_process_name, main_state, refs, values, current_process_media_kind, next_batch_mode)
        updated_memory = _store_batch_name(updated_memory, current_process_media_kind, next_batch_mode, batch_name_value)
        saved_settings = catalog.load_saved_mediaflow_settings()
        next_refs = catalog.get_saved_user_settings_refs(saved_settings, next_media_kind)
        next_saved_ui, next_form, restored = _slot_form(updated_memory, next_media_kind, next_batch_mode, main_state, next_refs)[1:]
        return (
            next_refs,
            gr.update(choices=next_form.model_type_choices, value=next_form.model_type),
            gr.update(choices=next_form.process_choices, value=next_form.process_name),
            form_controller.user_settings_hint_update(next_form.process_choices),
            updated_memory,
            next_form.process_name,
            *form_controller.settings_action_updates(next_form.model_type, next_form.process_name),
            *_active_source_outputs(next_media_kind, next_batch_mode, restored[0]),
            *restored[1:],
            gr.update(value=_remembered_batch_name(updated_memory, next_media_kind, next_batch_mode, next_saved_ui)),
            gr.update(visible=next_batch_mode == "batch"),
        )

    def _media_visibility_updates(process_name_value, main_state, refs, batch_mode_value):
        image_process = library.is_image_process(process_name_value, main_state, refs)
        is_batch = str(batch_mode_value or "").strip() == "batch"
        return (
            gr.update(visible=(not is_batch and not image_process)),
            gr.update(visible=(not is_batch and image_process)),
            gr.update(visible=(is_batch and not image_process)),
            gr.update(visible=(is_batch and image_process)),
            gr.update(visible=(not image_process or is_batch)),
            gr.update(visible=not image_process),
            gr.update(visible=not library.hides_chunk_size(process_name_value, main_state, refs)),
            gr.update(visible=not image_process),
            gr.update(visible=not image_process),
            gr.update(value=status_ui.render_process_status_html("Idle", "Waiting to start...") if image_process else status_ui.render_chunk_status_html(0, 0, 0, "Idle", "Waiting to start...")),
            gr.update(value=ui_help.process_info_html(library, process_name_value, main_state, refs)),
        )

    def _restore_active_form(memory_state, process_name_value, media_kind, batch_mode_value, main_state, refs, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value):
        active_source_path = _active_source_value(media_kind, batch_mode_value, source_path_value, source_image_path_value, batch_source_path_value, batch_image_source_path_value)
        restored = form_controller.restore_state(memory_state, process_name_value, active_source_path, main_state, refs, media_kind, batch_mode_value)
        return (*_active_source_outputs(media_kind, batch_mode_value, restored[0]), *restored[1:])

    process_form_memory = gr.State(initial_process_form_memory)
    active_process_name_state = gr.State(default_process_name)
    user_process_refs = gr.State(initial_user_refs)
    batch_mode = gr.State(default_batch_mode)
    with gr.Column(elem_id="mediaflow-app"):
        gr.HTML(ui_help.stylesheet(), padding=False)
        with gr.Row(elem_id="mediaflow-topbar"):
            gr.HTML(ui_help.brand_html(), elem_id="mediaflow-brand", padding=False)
            batch_mode_choice = gr.Radio([("One Item", "single"), ("Batch", "batch")], value=default_batch_mode, show_label=False, container=False, interactive=True, elem_id="mediaflow-batch-mode", elem_classes="mf-seg")
            process_media_kind = gr.Radio([("Video", "video"), ("Image", "image")], value=default_media_kind, show_label=False, container=False, elem_id="mediaflow-media-kind", elem_classes="mf-seg")
            guide_button = gr.Button("Guide", scale=0, min_width=0, elem_id=ui_help.GUIDE_BUTTON_ID)
        with gr.Row(elem_id="mediaflow-body"):
            with gr.Column(scale=3, min_width=440, elem_id="mediaflow-setup"):
                with gr.Column(elem_classes="mf-card"):
                    gr.HTML(ui_help.card_head("Process", "What to do with your media: pick a model, then one of its processes.", section="Choosing a Process"), padding=False)
                    with gr.Row(elem_id="mediaflow-process-row"):
                        process_model_type = gr.Dropdown(model_type_choices, value=default_model_type, label="Model", scale=2)
                        process_name = gr.Dropdown(default_process_choices, value=default_process_name, label="Process", scale=3)
                        with gr.Column(scale=0, min_width=41, visible=default_model_type == ui_constants.ADD_USER_SETTINGS_MODEL_TYPE or catalog.is_user_process_value(default_process_name), elem_id="mediaflow-settings-actions") as settings_actions_column:
                            add_user_settings_btn = gr.Button("\u2795", size="sm", min_width=1, visible=default_model_type == ui_constants.ADD_USER_SETTINGS_MODEL_TYPE, elem_classes=["chat__template-tool-icon-btn"])
                            delete_user_settings_btn = gr.Button("\U0001F5D1\uFE0F", size="sm", min_width=1, visible=catalog.is_user_process_value(default_process_name), elem_classes=["chat__template-tool-icon-btn", "chat__template-tool-icon-btn--danger"])
                            settings_actions_placeholder = gr.HTML("<div class='mediaflow-settings-action-placeholder'></div>", visible=False)
                    with gr.Row(visible=library.process_choices_have_user_settings(default_process_choices), elem_id="mediaflow-user-settings-hint-row") as process_user_settings_hint_row:
                        gr.HTML(value=ui_constants.USER_SETTINGS_HINT_HTML, padding=False)
                    process_info = gr.HTML(ui_help.process_info_html(library, default_process_name, self.state.value, initial_user_refs), padding=False)
                with gr.Column(elem_classes="mf-card"):
                    gr.HTML(ui_help.card_head("Source", "The media to process: one file, or a batch of files, folders and patterns.", section="Sources and Batches"), padding=False)
                    with gr.Row(visible=default_batch_mode == "batch") as batch_name_row:
                        batch_name = gr.Textbox(label="Batch Name", value=default_batch_name, placeholder="Names the batch and its progress files, so that it can be resumed")
                    with gr.Column(visible=default_batch_mode == "batch" and not initial_image_process) as batch_video_source_column:
                        batch_source_path = LocalFilePickerTextbox(label="Source Videos", value=default_batch_source_path, placeholder="One per line: a video, a folder or a pattern such as D:\\clips\\*.mp4", file_extensions=VIDEO_FILE_EXTENSIONS, multiselect=True, popup_title="Browse Local Source Videos").mount()
                    with gr.Column(visible=default_batch_mode == "batch" and initial_image_process) as batch_image_source_column:
                        batch_image_source_path = LocalFilePickerTextbox(label="Source Images", value=default_batch_source_path, placeholder="One per line: an image, a folder or a pattern such as D:\\photos\\*.jpg", file_extensions=IMAGE_FILE_EXTENSIONS, multiselect=True, popup_title="Browse Local Source Images").mount()
                    with gr.Column(visible=default_batch_mode != "batch" and not initial_image_process) as source_video_column:
                        source_path = LocalFilePickerTextbox(label="Source Video", value=default_state.source_path, placeholder="The path of a video file", file_extensions=VIDEO_FILE_EXTENSIONS, popup_title="Browse Local Source Video").mount()
                    with gr.Column(visible=default_batch_mode != "batch" and initial_image_process) as source_image_column:
                        source_image_path = LocalFilePickerTextbox(label="Source Image", value=default_state.source_path, placeholder="The path of an image file", file_extensions=IMAGE_FILE_EXTENSIONS, popup_title="Browse Local Source Image").mount()
                with gr.Column(elem_classes="mf-card"):
                    gr.HTML(ui_help.card_head("Output", "Where the result goes and how it looks.", section="Output and Quality"), padding=False)
                    with gr.Row():
                        output_path = gr.Textbox(label="Output", value=default_state.output_path, placeholder="Automatic (next to the source)", scale=3, min_width=240)
                        continue_enabled = gr.Checkbox(label="Continue", value=default_state.continue_enabled, elem_classes="mf-toggle", scale=0, min_width=130, visible=not initial_image_process or default_batch_mode == "batch")
                        preview_enabled = gr.Checkbox(label="Live Preview", value=default_preview_enabled, elem_classes="mf-toggle", scale=0, min_width=150)
                    with gr.Row():
                        initial_output_resolution_choices = list(output_resolution_choices)
                        if default_state.output_resolution not in {value for _, value in initial_output_resolution_choices}:
                            initial_output_resolution_choices.insert(0, (default_state.output_resolution, default_state.output_resolution))
                        output_resolution = gr.Dropdown(initial_output_resolution_choices, value=default_state.output_resolution, label="Output Resolution", visible=initial_form.output_resolution_visible)
                        target_ratio = gr.Dropdown(initial_form.target_ratio_choices if initial_form.target_ratio_visible else ui_constants.RATIO_CHOICES_WITH_EMPTY, value=default_state.target_ratio if initial_form.target_ratio_visible else "", label=initial_form.target_ratio_label, visible=initial_form.target_ratio_visible)
                        default_process_strength = 1.0 if initial_form.target_ratio_visible else default_state.process_strength
                        process_strength = gr.Slider(label="Process Strength", minimum=min(0.0, default_process_strength), maximum=max(3.0, default_process_strength), step=0.01, value=default_process_strength, visible=initial_form.process_strength_visible)
                    # the spatial processor's parameters, in the shared post-processing inputs, refreshed with the process
                    spatial_parameter_ui = parameter_ui.create(gr, initial_spatial_parameters, initial_spatial_parameter_values, triggers=[process_name.change], refresh_inputs=[process_name, self.state, user_process_refs],
                                                               parameters_for=library.spatial_upsampler_parameters, late_postprocessing=True, event_options=form_event_options)
                    spatial_upsampler_parameter_state = spatial_parameter_ui["parameters"]
                with gr.Column(elem_classes=["mf-card", "mf-autohide"]):
                    gr.HTML(ui_help.card_head("Prompt", "What the process should create or how it should render. Timed blocks change it along the video.", section="Prompts"), padding=False)
                    prompt_text = gr.Textbox(label="Prompt", value=default_state.prompt, lines=3, placeholder=prompts.TIMED_PROMPT_EXAMPLE, visible=initial_form.prompt_visible)
                with gr.Column(elem_classes=["mf-card", "mf-autohide"]):
                    gr.HTML(ui_help.card_head("Timing", "The part of the video to process, its soundtrack and the chunks of long videos.", section="Long Videos"), padding=False)
                    with gr.Row():
                        start_seconds = gr.Textbox(label="Start", value=default_state.start_seconds, placeholder="From the start, or 01:30", visible=not initial_image_process)
                        end_seconds = gr.Textbox(label="End", value=default_state.end_seconds, placeholder="To the end, or 02:45.5", visible=not initial_image_process)
                        source_audio_track = gr.Dropdown(source_audio_track_choices, value=default_state.source_audio_track, label="Audio Track", visible=not initial_image_process)
                    with gr.Row():
                        chunk_size_seconds = gr.Number(label="Chunk Size (seconds)", value=default_state.chunk_size_seconds, precision=2, visible=initial_form.chunk_size_visible)
                        sliding_window_overlap = gr.Slider(label="Sliding Window Overlap (frames)", minimum=0 if not initial_form.overlap_visible else 1, maximum=initial_form.overlap_max, step=initial_form.overlap_step, value=default_state.sliding_window_overlap, visible=initial_form.overlap_visible)
            with gr.Column(scale=2, min_width=360, elem_id="mediaflow-monitor"):
                with gr.Column(elem_classes="mf-card"):
                    gr.HTML(ui_help.card_head("Run", "Start, follow and stop the process. A stopped run resumes where it stopped.", section="Stop and Resume", icon_name="run"), padding=False)
                    with gr.Row(elem_id="mediaflow-run-buttons"):
                        start_btn = gr.Button("Start Process", elem_id="mediaflow-start-btn")
                        abort_btn = gr.Button("Stop", interactive=False, elem_id="mediaflow-stop-btn")
                    batch_status_html = gr.HTML(value="", padding=False)
                    status_html = gr.HTML(value=initial_status_html, padding=False)
                with gr.Column(elem_classes=["mf-card", "mf-autohide"]):
                    gr.HTML(ui_help.card_head("Live Preview", "The last frame of the latest finished chunk.", icon_name="preview"), padding=False)
                    preview_image = gr.Image(label="Live Preview", show_label=False, type="pil", visible=default_preview_enabled, elem_id="mediaflow-preview-image")
                with gr.Column(elem_classes="mf-card"):
                    gr.HTML(ui_help.card_head("Result", "The output file, and how many generations WanGP keeps while processing.", icon_name="result"), padding=False)
                    output_file = gr.HTML(value=initial_output_html, elem_id="mediaflow-output-file-html", padding=False)
                    preserve_artifact_count = gr.Dropdown(
                        [(f"The Last {count} Generations", count) for count in range(catalog.PRESERVE_ARTIFACTS_MIN, catalog.PRESERVE_ARTIFACTS_MAX + 1)],
                        value=default_preserve_artifact_count,
                        label="Keep in Galleries & Output Folders",
                        elem_id="mediaflow-preserve-artifacts-dropdown",
                    )
        guide_window = gr.HTML("", padding=False)
        guide_loaded = gr.State(False)
        preview_refresh = gr.Textbox(value="", visible=False)
        tab_refresh_trigger = gr.Textbox(value="", visible=False)

    self.on_tab_outputs = [tab_refresh_trigger]
    visibility_outputs = [source_video_column, source_image_column, batch_video_source_column, batch_image_source_column, continue_enabled, source_audio_track, chunk_size_seconds, start_seconds, end_seconds, status_html, process_info]

    preserve_artifact_count.change(
        fn=_set_preserve_artifact_count,
        inputs=[preserve_artifact_count],
        outputs=[preserve_artifact_count],
        queue=False,
        show_progress="hidden",
    )
    preview_enabled.change(
        fn=_set_preview_enabled,
        inputs=[preview_enabled],
        outputs=[preview_enabled, preview_image],
        queue=False,
        show_progress="hidden",
    )

    gr.on(
        [
            source_path.input,
            source_image_path.input,
            process_strength.input,
            output_path.input,
            prompt_text.input,
            continue_enabled.input,
            source_audio_track.input,
            output_resolution.input,
            target_ratio.input,
            chunk_size_seconds.input,
            sliding_window_overlap.input,
            start_seconds.input,
            end_seconds.input,
            batch_source_path.input,
            batch_image_source_path.input,
            batch_name.input,
        ],
        fn=_store_memory,
        inputs=[process_form_memory, active_process_name_state, process_media_kind, batch_mode, self.state, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path, batch_name, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_form_memory],
        **form_event_options,
    )
    process_mode_event = batch_mode_choice.select(
        fn=_change_batch_mode,
        inputs=[process_form_memory, active_process_name_state, process_media_kind, batch_mode, self.state, self.lset_name, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path, batch_name, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_form_memory, batch_mode, process_model_type, process_name, process_user_settings_hint_row, active_process_name_state, settings_actions_column, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds, batch_name, batch_name_row, batch_status_html],
        **form_event_options,
    )
    process_mode_event.then(fn=_media_visibility_updates, inputs=[active_process_name_state, self.state, user_process_refs, batch_mode], outputs=visibility_outputs, **form_event_options)
    process_media_kind_event = process_media_kind.input(
        fn=_change_media_kind,
        inputs=[process_form_memory, active_process_name_state, process_media_kind, batch_mode, self.state, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path, batch_name, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[user_process_refs, process_model_type, process_name, process_user_settings_hint_row, process_form_memory, active_process_name_state, settings_actions_column, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds, batch_name, batch_name_row],
        **form_event_options,
    )
    process_media_kind_event.then(fn=_media_visibility_updates, inputs=[active_process_name_state, self.state, user_process_refs, batch_mode], outputs=visibility_outputs, **form_event_options)
    process_model_type_event = process_model_type.input(
        fn=_change_process_model_type,
        inputs=[process_form_memory, active_process_name_state, process_media_kind, batch_mode, process_model_type, self.state, self.lset_name, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_form_memory, active_process_name_state, process_name, process_user_settings_hint_row, settings_actions_column, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        **form_event_options,
    )
    process_model_type_event.then(fn=_media_visibility_updates, inputs=[active_process_name_state, self.state, user_process_refs, batch_mode], outputs=visibility_outputs, **form_event_options)
    process_name_event = process_name.input(
        fn=_change_process_name,
        inputs=[process_form_memory, active_process_name_state, process_media_kind, batch_mode, process_name, process_model_type, self.state, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_form_memory, active_process_name_state, settings_actions_column, delete_user_settings_btn, settings_actions_placeholder, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        **form_event_options,
    )
    process_name_event.then(fn=_media_visibility_updates, inputs=[active_process_name_state, self.state, user_process_refs, batch_mode], outputs=visibility_outputs, **form_event_options)
    add_user_settings_event = add_user_settings_btn.click(
        fn=_add_user_process_link,
        inputs=[process_name, process_media_kind, self.state, self.lset_name, user_process_refs],
        outputs=[user_process_refs, process_model_type, process_name, process_user_settings_hint_row, active_process_name_state, settings_actions_column, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder],
        **form_event_options,
    )
    add_user_settings_restore_event = add_user_settings_event.then(fn=_restore_active_form, inputs=[process_form_memory, active_process_name_state, process_media_kind, batch_mode, self.state, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path], outputs=[source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds], **form_event_options)
    add_user_settings_restore_event.then(fn=_media_visibility_updates, inputs=[active_process_name_state, self.state, user_process_refs, batch_mode], outputs=visibility_outputs, **form_event_options)
    delete_user_settings_event = delete_user_settings_btn.click(
        fn=_delete_user_process_link,
        inputs=[process_form_memory, process_media_kind, batch_mode, process_name, self.state, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path],
        outputs=[user_process_refs, process_model_type, process_name, process_user_settings_hint_row, active_process_name_state, settings_actions_column, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        **form_event_options,
    )
    delete_user_settings_event.then(fn=_media_visibility_updates, inputs=[active_process_name_state, self.state, user_process_refs, batch_mode], outputs=visibility_outputs, **form_event_options)
    refresh_form_event = self.refresh_form_trigger.change(
        fn=_refresh_from_main,
        inputs=[self.refresh_form_trigger, process_form_memory, active_process_name_state, process_media_kind, batch_mode, process_model_type, self.state, self.lset_name, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_model_type, process_name, process_user_settings_hint_row, process_form_memory, active_process_name_state, settings_actions_column, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        **form_event_options,
    )
    refresh_form_event.then(fn=_media_visibility_updates, inputs=[active_process_name_state, self.state, user_process_refs, batch_mode], outputs=visibility_outputs, **form_event_options)
    tab_refresh_event = tab_refresh_trigger.change(
        fn=_refresh_from_main,
        inputs=[tab_refresh_trigger, process_form_memory, active_process_name_state, process_media_kind, batch_mode, process_model_type, self.state, self.lset_name, user_process_refs, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        outputs=[process_model_type, process_name, process_user_settings_hint_row, process_form_memory, active_process_name_state, settings_actions_column, add_user_settings_btn, delete_user_settings_btn, settings_actions_placeholder, source_path, source_image_path, batch_source_path, batch_image_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds],
        **form_event_options,
    )
    tab_refresh_event.then(fn=_media_visibility_updates, inputs=[active_process_name_state, self.state, user_process_refs, batch_mode], outputs=visibility_outputs, **form_event_options)
    start_btn.click(
        fn=process_runner.start_process,
        inputs=[self.state, process_name, user_process_refs, source_path, source_image_path, process_strength, output_path, prompt_text, continue_enabled, preview_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds, batch_mode, batch_name, batch_source_path, batch_image_source_path, spatial_upsampler_parameter_state],
        outputs=[status_html, batch_status_html, output_file, preview_refresh, start_btn, abort_btn, batch_name, self.output_trigger],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    guide_button.click(fn=lambda loaded: (gr.skip(), True) if loaded else (ui_help.guide_html(), True), inputs=[guide_loaded], outputs=[guide_window, guide_loaded], queue=False, show_progress="hidden").then(fn=None, js=ui_help.GUIDE_OPEN_JS)
    preview_refresh.change(fn=refresh_preview, inputs=[preview_refresh], outputs=[preview_image], queue=False, show_progress="hidden")
    abort_btn.click(fn=stop_process, outputs=[status_html, batch_status_html, start_btn, abort_btn], queue=False, show_progress="hidden")
