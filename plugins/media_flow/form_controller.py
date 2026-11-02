# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from . import common
from . import constants as ui_constants
from . import frame_planning as frames
from . import process_catalog as catalog


@dataclass(frozen=True)
class ProcessFormState:
    process_model_type: str
    process_name: str
    source_path: str
    process_strength: float
    output_path: str
    prompt: str
    continue_enabled: bool
    source_audio_track: str
    output_resolution: str
    target_ratio: str
    chunk_size_seconds: float
    sliding_window_overlap: int
    start_seconds: str
    end_seconds: str

    def to_dict(self) -> dict:
        return {
            "process_model_type": self.process_model_type,
            "process_name": self.process_name,
            "source_path": self.source_path,
            "process_strength": self.process_strength,
            "output_path": self.output_path,
            "prompt": self.prompt,
            "continue_enabled": self.continue_enabled,
            "source_audio_track": self.source_audio_track,
            "output_resolution": self.output_resolution,
            "target_ratio": self.target_ratio,
            "chunk_size_seconds": self.chunk_size_seconds,
            "sliding_window_overlap": self.sliding_window_overlap,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


@dataclass(frozen=True)
class FormComponentValues:
    source_path: object
    process_strength: object
    output_path: object
    prompt_text: object
    continue_enabled: object
    source_audio_track: object
    output_resolution: object
    target_ratio: object
    chunk_size_seconds: object
    sliding_window_overlap: object
    start_seconds: object
    end_seconds: object

    def to_raw_state(self) -> dict:
        return {
            "source_path": self.source_path,
            "process_strength": self.process_strength,
            "output_path": self.output_path,
            "prompt": self.prompt_text,
            "continue_enabled": self.continue_enabled,
            "source_audio_track": self.source_audio_track,
            "output_resolution": self.output_resolution,
            "target_ratio": self.target_ratio,
            "chunk_size_seconds": self.chunk_size_seconds,
            "sliding_window_overlap": self.sliding_window_overlap,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


@dataclass(frozen=True)
class InitialFormPatch:
    model_type_choices: list[tuple[str, str]]
    process_choices: list[tuple[str, str]]
    model_type: str
    process_name: str
    form_state: ProcessFormState
    overlap_step: int
    overlap_max: int
    overlap_visible: bool
    chunk_size_visible: bool
    output_resolution_visible: bool
    prompt_visible: bool
    process_strength_visible: bool
    target_ratio_visible: bool
    target_ratio_label: str
    target_ratio_choices: list[tuple[str, str]]


RESTORED_FORM_OUTPUT_COUNT = 12


def create_form_controller(library, get_model_def, server_config: dict) -> "ProcessFormController":
    """The form rules of the tab, also used for the defaults of agent runs."""
    all_resolutions = ui_constants.HIGH_OUTPUT_RESOLUTION_CHOICES + ui_constants.STANDARD_OUTPUT_RESOLUTION_CHOICES
    return ProcessFormController(library=library, get_model_def=get_model_def, output_resolution_values={value for _, value in all_resolutions}, output_resolution_choices=ui_constants.output_resolution_choices(server_config),
                                 source_audio_track_values={value for _, value in ui_constants.SOURCE_AUDIO_TRACK_CHOICES}, ratio_values={value for _, value in ui_constants.RATIO_CHOICES})


def skipped_restored_form_outputs() -> tuple:
    return tuple(gr.skip() for _ in range(RESTORED_FORM_OUTPUT_COUNT))


class ProcessFormController:
    def __init__(
        self,
        *,
        library,
        get_model_def,
        output_resolution_values: set[str],
        output_resolution_choices: list[tuple[str, str]],
        source_audio_track_values: set[str],
        ratio_values: set[str],
        default_model_type: str | None = None,
    ) -> None:
        self.library = library
        self.get_model_def = get_model_def
        self.output_resolution_values = output_resolution_values
        self.output_resolution_choices = output_resolution_choices
        self.source_audio_track_values = source_audio_track_values
        self.ratio_values = ratio_values
        self.default_model_type = default_model_type or catalog.DEFAULT_MODEL_TYPE

    @staticmethod
    def _process_strength_slider_bounds(value: float) -> tuple[float, float]:
        lower = value if value < 0.0 else 0.0
        upper = value if value > 3.0 else 3.0
        return lower, upper

    @staticmethod
    def _fit_overlap_slider_value(value: int, maximum: int) -> int:
        if value < 1:
            return 1
        if value > maximum:
            return maximum
        return value

    def build_initial_form(self, saved_ui_settings: dict, media_kind: str, main_state: dict | None, initial_user_refs: list[str]) -> InitialFormPatch:
        media_kind = catalog.normalize_media_kind(media_kind)
        saved_process_name = str(saved_ui_settings.get("process_name") or "").strip()
        saved_model_type = str(saved_ui_settings.get("process_model_type") or "").strip()
        saved_process_definition = self.library.process_definition(saved_process_name, main_state, initial_user_refs)
        if saved_process_definition is not None and catalog.process_definition_media_kind(saved_process_definition) == media_kind:
            saved_model_type = self.library.process_definition_model_type(saved_process_definition) or saved_model_type
        else:
            saved_process_name = ""
        process_names_by_model_type = self.library.process_values_by_model_type(media_kind, initial_user_refs)
        default_catalog_model_type = catalog.default_model_type(media_kind)
        default_catalog_process_name = catalog.default_process_name(media_kind)
        default_model_type = (
            saved_model_type
            if saved_model_type in process_names_by_model_type
            else default_catalog_model_type
            if default_catalog_model_type in process_names_by_model_type
            else next(iter(process_names_by_model_type), default_catalog_model_type)
        )
        default_process_choices = self.library.normal_process_choices(media_kind, default_model_type, initial_user_refs)
        default_process_values = [value for _label, value in default_process_choices]
        default_process_name = (
            saved_process_name
            if saved_process_name in default_process_values
            else default_catalog_process_name
            if default_catalog_process_name in default_process_values
            else (default_process_values[0] if default_process_values else default_catalog_process_name)
        )
        form_state = self.build_form_state(default_process_name, saved_ui_settings, main_state, initial_user_refs, media_kind)
        default_rules = self.library.process_frame_rules(default_process_name, main_state, initial_user_refs)
        target_control_choices = self.library.target_control_choices(default_process_name, main_state, initial_user_refs)
        has_target_control = len(target_control_choices) > 0
        overlap_visible = not self.library.hides_sliding_window_overlap(default_process_name, main_state, initial_user_refs)
        self.default_model_type = default_model_type
        return InitialFormPatch(
            model_type_choices=self.library.model_type_choices(media_kind, initial_user_refs),
            process_choices=default_process_choices,
            model_type=default_model_type,
            process_name=default_process_name,
            form_state=form_state,
            overlap_step=default_rules.frame_step,
            overlap_max=frames.get_overlap_slider_max(default_model_type, self.get_model_def) if overlap_visible else 1,
            overlap_visible=overlap_visible,
            chunk_size_visible=not self.library.hides_chunk_size(default_process_name, main_state, initial_user_refs),
            output_resolution_visible=not self.library.hides_output_resolution(default_process_name, main_state, initial_user_refs),
            prompt_visible=not self.library.hides_prompt(default_process_name, main_state, initial_user_refs),
            process_strength_visible=self.library.is_process_strength_visible(default_process_name, main_state, initial_user_refs),
            target_ratio_visible=has_target_control or self.library.has_process_outpaint(default_process_name, main_state, initial_user_refs),
            target_ratio_label=self.library.target_control_label(default_process_name, main_state, initial_user_refs) if has_target_control else "Target Ratio",
            target_ratio_choices=target_control_choices if has_target_control else ui_constants.RATIO_CHOICES,
        )

    def user_settings_hint_update(self, process_choices: list[tuple[str, str]]):
        return gr.update(visible=self.library.process_choices_have_user_settings(process_choices))

    @staticmethod
    def settings_action_updates(process_model_type_value: str, process_value: str) -> tuple:
        add_visible = process_model_type_value == ui_constants.ADD_USER_SETTINGS_MODEL_TYPE
        delete_visible = catalog.is_user_process_value(process_value)
        actions_visible = add_visible or delete_visible
        return gr.update(visible=actions_visible), gr.update(visible=add_visible), gr.update(visible=delete_visible), gr.update(visible=False)

    def target_ratio_update(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, target_ratio: str | None = None):
        process_definition = self.library.process_definition_or_default(process_name, main_state, user_refs)
        process_settings = process_definition.get("settings", {})
        target_control_choices = self.library.target_control_choices(process_name, main_state, user_refs)
        if len(target_control_choices) > 0:
            values = {value for _label, value in target_control_choices}
            value = str(target_ratio or "").strip()
            if value not in values:
                value = self.library.target_control_default(process_name, main_state, user_refs, saved_value=value)
            return gr.update(label=self.library.target_control_label(process_name, main_state, user_refs), value=value, visible=True, choices=target_control_choices)
        visible = self.library.has_process_outpaint(process_name, main_state, user_refs)
        value = self.library.normalize_outpaint_target_ratio(process_settings, target_ratio) if visible else ""
        return gr.update(label="Target Ratio", value=value, visible=visible, choices=ui_constants.RATIO_CHOICES if visible else ui_constants.RATIO_CHOICES_WITH_EMPTY)

    def process_strength_update(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, process_strength: float | None = None):
        process_definition = self.library.process_definition(process_name, main_state, user_refs)
        visible = self.library.is_process_strength_visible(process_name, main_state, user_refs)
        default_value = common.get_default_process_strength((process_definition or {}).get("settings", {}))
        if isinstance(process_definition, dict) and process_definition.get("source") == "user":
            user_default = self.library.user_lora_strength_override_default(process_definition)
            if user_default is not None:
                default_value = user_default
        value = common.coerce_float(process_strength, default_value) if visible else default_value
        minimum, maximum = self._process_strength_slider_bounds(value)
        return gr.update(value=value, visible=visible, minimum=minimum, maximum=maximum)

    def prompt_update(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, prompt: str):
        return gr.update(value=prompt, visible=not self.library.hides_prompt(process_name, main_state, user_refs))

    def output_resolution_update(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, output_resolution: str):
        choices = list(self.output_resolution_choices)
        if output_resolution in self.output_resolution_values and not any(value == output_resolution for _, value in choices):
            choices.insert(0, (output_resolution, output_resolution))
        return gr.update(choices=choices, value=output_resolution, visible=not self.library.hides_output_resolution(process_name, main_state, user_refs))

    def overlap_control_updates(self, process_name: str, main_state: dict | None, user_refs: list[str] | None):
        if self.library.hides_sliding_window_overlap(process_name, main_state, user_refs):
            return gr.update(minimum=0, maximum=1, step=1, value=0, visible=False)
        process_definition = self.library.process_definition_or_default(process_name, main_state, user_refs)
        settings = process_definition.get("settings", {})
        model_type = str(settings.get("model_type") or "")
        step = frames.get_vae_temporal_latent_size(model_type, self.get_model_def)
        maximum = frames.get_overlap_slider_max(model_type, self.get_model_def)
        value = common.coerce_int(settings.get("sliding_window_overlap"), 1, minimum=1)
        value = self._fit_overlap_slider_value(frames.normalize_overlap_frames(value, frame_step=step), maximum)
        return gr.update(minimum=1, maximum=maximum, step=step, value=value, visible=True)

    def build_form_state(self, process_name: str, raw_state: dict | None = None, main_state: dict | None = None, user_refs: list[str] | None = None, media_kind: str = catalog.DEFAULT_MEDIA_KIND) -> ProcessFormState:
        process_definition = self.library.process_definition_or_default(process_name, main_state, user_refs, media_kind)
        process_settings = process_definition.get("settings", {})
        model_type = str(process_settings.get("model_type") or catalog.DEFAULT_MODEL_TYPE)
        frame_rules = self.library.process_frame_rules(process_name, main_state, user_refs)
        step = int(frame_rules.frame_step)
        maximum = frames.get_overlap_slider_max(model_type, self.get_model_def) if not self.library.hides_sliding_window_overlap(process_name, main_state, user_refs) else 1
        raw_state = raw_state if isinstance(raw_state, dict) else {}

        default_strength = common.get_default_process_strength(process_settings)
        saved_process_strength = raw_state.get("process_strength", raw_state.get("control_video_strength"))
        process_strength = default_strength if saved_process_strength is None else common.coerce_float(saved_process_strength, default_strength)
        source_audio_track = str(raw_state.get("source_audio_track") or "").strip()
        output_resolution = str(raw_state.get("output_resolution") or "").strip()
        target_control_choices = self.library.target_control_choices(process_name, main_state, user_refs)
        if len(target_control_choices) > 0:
            target_values = {value for _label, value in target_control_choices}
            target_ratio = str(raw_state.get("target_ratio") or process_settings.get("target_ratio") or self.library.target_control_default(process_name, main_state, user_refs)).strip()
            if target_ratio not in target_values:
                target_ratio = self.library.target_control_default(process_name, main_state, user_refs, saved_value=target_ratio)
        else:
            target_ratio = self.library.normalize_outpaint_target_ratio(process_settings, raw_state.get("target_ratio"))
        if self.library.hides_sliding_window_overlap(process_name, main_state, user_refs):
            sliding_window_overlap = 0
        else:
            overlap_default = self._fit_overlap_slider_value(frames.normalize_overlap_frames(common.coerce_int(process_settings.get("sliding_window_overlap"), 1, minimum=1), frame_step=step), maximum)
            overlap_value = common.coerce_int(raw_state.get("sliding_window_overlap"), overlap_default, minimum=1)
            sliding_window_overlap = self._fit_overlap_slider_value(frames.normalize_overlap_frames(overlap_value, frame_step=step), maximum)
        default_chunk_size_seconds = self.library.default_chunk_size_seconds(process_name, main_state, user_refs)

        return ProcessFormState(
            process_model_type=model_type,
            process_name=process_name,
            source_path=str(raw_state.get("source_path") or ui_constants.DEFAULT_SOURCE_PATH),
            process_strength=process_strength,
            output_path=str(raw_state.get("output_path") or ui_constants.DEFAULT_OUTPUT_PATH),
            prompt=str(raw_state.get("prompt") or "") if "prompt" in raw_state else str(process_settings.get("prompt") or ""),
            continue_enabled=common.coerce_bool(raw_state.get("continue_enabled"), True),
            source_audio_track=source_audio_track if source_audio_track in self.source_audio_track_values else "",
            output_resolution=output_resolution if output_resolution in self.output_resolution_values else "720p",
            target_ratio=target_ratio if len(target_control_choices) > 0 or target_ratio in self.ratio_values else ui_constants.DEFAULT_TARGET_RATIO,
            chunk_size_seconds=common.coerce_float(raw_state.get("chunk_size_seconds"), default_chunk_size_seconds, minimum=0.1),
            sliding_window_overlap=sliding_window_overlap,
            start_seconds="" if raw_state.get("start_seconds") in (None, "") else str(raw_state.get("start_seconds")),
            end_seconds="" if raw_state.get("end_seconds") in (None, "") else str(raw_state.get("end_seconds")),
        )

    def build_state(self, process_name: str, raw_state: dict | None = None, main_state: dict | None = None, user_refs: list[str] | None = None, media_kind: str = catalog.DEFAULT_MEDIA_KIND) -> dict:
        return self.build_form_state(process_name, raw_state, main_state, user_refs, media_kind).to_dict()

    def snapshot_state(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, values: FormComponentValues, media_kind: str = catalog.DEFAULT_MEDIA_KIND) -> dict:
        return self.build_state(process_name, values.to_raw_state(), main_state, user_refs, media_kind)

    @staticmethod
    def memory_key(process_name: str, media_kind: str = catalog.DEFAULT_MEDIA_KIND, batch_mode: str = catalog.DEFAULT_BATCH_MODE) -> str:
        return f"{catalog.normalize_media_kind(media_kind)}/{catalog.normalize_batch_mode(batch_mode)}/{str(process_name or '').strip()}"

    @staticmethod
    def memory_selection_key(media_kind: str = catalog.DEFAULT_MEDIA_KIND, batch_mode: str = catalog.DEFAULT_BATCH_MODE) -> str:
        return f"{catalog.normalize_media_kind(media_kind)}/{catalog.normalize_batch_mode(batch_mode)}/__selected_process__"

    @classmethod
    def select_memory_process(cls, memory_state: dict, process_name: str, media_kind: str = catalog.DEFAULT_MEDIA_KIND, batch_mode: str = catalog.DEFAULT_BATCH_MODE) -> dict:
        memory_state[cls.memory_selection_key(media_kind, batch_mode)] = str(process_name or "").strip()
        return memory_state

    def remembered_process_name(self, memory_state: dict | None, media_kind: str, batch_mode: str, main_state: dict | None, user_refs: list[str] | None) -> str:
        if not isinstance(memory_state, dict):
            return ""
        process_name = str(memory_state.get(self.memory_selection_key(media_kind, batch_mode)) or "").strip()
        return process_name if self.library.process_definition(process_name, main_state, user_refs) is not None else ""

    def _memory_state_for_process(self, memory_state: dict | None, process_name: str, media_kind: str, batch_mode: str) -> dict | None:
        if not isinstance(memory_state, dict):
            return None
        key = self.memory_key(process_name, media_kind, batch_mode)
        state = memory_state.get(key)
        if isinstance(state, dict):
            return state
        state = memory_state.get(str(process_name or "").strip())
        return state if isinstance(state, dict) else None

    def store_memory(self, memory_state: dict | None, current_process_name: str, main_state: dict | None, user_refs: list[str] | None, values: FormComponentValues, media_kind: str = catalog.DEFAULT_MEDIA_KIND, batch_mode: str = catalog.DEFAULT_BATCH_MODE):
        updated_memory = dict(memory_state) if isinstance(memory_state, dict) else {}
        current_process_name = str(current_process_name or "").strip()
        if self.library.process_definition(current_process_name, main_state, user_refs) is not None:
            updated_memory[self.memory_key(current_process_name, media_kind, batch_mode)] = self.snapshot_state(current_process_name, main_state, user_refs, values, media_kind)
            self.select_memory_process(updated_memory, current_process_name, media_kind, batch_mode)
        return updated_memory

    def restore_state(self, memory_state: dict | None, process_name: str, current_source_path: str, main_state: dict | None, user_refs: list[str] | None, media_kind: str = catalog.DEFAULT_MEDIA_KIND, batch_mode: str = catalog.DEFAULT_BATCH_MODE) -> tuple:
        state = self.build_form_state(process_name, self._memory_state_for_process(memory_state, process_name, media_kind, batch_mode), main_state, user_refs, media_kind)
        source_path_value = current_source_path.strip() or state.source_path
        return (
            source_path_value,
            self.process_strength_update(process_name, main_state, user_refs, state.process_strength),
            state.output_path,
            self.prompt_update(process_name, main_state, user_refs, state.prompt),
            state.continue_enabled,
            state.source_audio_track,
            self.output_resolution_update(process_name, main_state, user_refs, state.output_resolution),
            self.target_ratio_update(process_name, main_state, user_refs, state.target_ratio),
            state.chunk_size_seconds,
            self.overlap_control_updates(process_name, main_state, user_refs),
            state.start_seconds,
            state.end_seconds,
        )

    def change_process_model_type(self, memory_state: dict | None, current_process_name: str, media_kind: str, batch_mode: str, next_model_type: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None, values: FormComponentValues) -> tuple:
        media_kind = catalog.normalize_media_kind(media_kind)
        batch_mode = catalog.normalize_batch_mode(batch_mode)
        refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})
        updated_memory = self.store_memory(memory_state, current_process_name, main_state, refs, values, media_kind, batch_mode)
        process_choices, next_process_name = self.library.process_choices(media_kind, next_model_type, main_state, main_lset_name, refs)
        next_process_name = str(next_process_name or ui_constants.NO_USER_SETTINGS_VALUE).strip()
        self.select_memory_process(updated_memory, next_process_name, media_kind, batch_mode)
        return (
            updated_memory,
            next_process_name,
            gr.update(choices=process_choices, value=next_process_name),
            self.user_settings_hint_update(process_choices),
            *self.settings_action_updates(next_model_type, next_process_name),
            *self.restore_state(updated_memory, next_process_name, values.source_path, main_state, refs, media_kind, batch_mode),
        )

    def change_process_name(self, memory_state: dict | None, current_process_name: str, media_kind: str, batch_mode: str, next_process_name: str, process_model_type_value: str, main_state: dict | None, user_refs: list[str] | None, values: FormComponentValues) -> tuple:
        media_kind = catalog.normalize_media_kind(media_kind)
        batch_mode = catalog.normalize_batch_mode(batch_mode)
        refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})
        updated_memory = self.store_memory(memory_state, current_process_name, main_state, refs, values, media_kind, batch_mode)
        next_process_name = str(next_process_name or "").strip()
        self.select_memory_process(updated_memory, next_process_name, media_kind, batch_mode)
        actions_update, _add_update, delete_update, placeholder_update = self.settings_action_updates(process_model_type_value, next_process_name)
        return (
            updated_memory,
            next_process_name,
            actions_update,
            delete_update,
            placeholder_update,
            *self.restore_state(updated_memory, next_process_name, values.source_path, main_state, refs, media_kind, batch_mode),
        )

    def refresh_from_main(self, _refresh_id, memory_state: dict | None, current_process_name: str, media_kind: str, batch_mode: str, process_model_type_value: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str] | None, values: FormComponentValues) -> tuple:
        media_kind = catalog.normalize_media_kind(media_kind)
        batch_mode = catalog.normalize_batch_mode(batch_mode)
        refs = catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})
        model_choices = self.library.model_type_choices(media_kind, refs)
        process_model_type_value = str(process_model_type_value or "").strip()
        if process_model_type_value != ui_constants.ADD_USER_SETTINGS_MODEL_TYPE:
            valid_model_values = {value for _label, value in model_choices}
            model_value = process_model_type_value if process_model_type_value in valid_model_values else catalog.default_model_type(media_kind)
            return (
                gr.update(choices=model_choices, value=model_value),
                gr.update(),
                gr.update(),
                memory_state,
                current_process_name,
                *self.settings_action_updates(model_value, current_process_name),
                *skipped_restored_form_outputs(),
            )
        updated_memory = self.store_memory(memory_state, current_process_name, main_state, refs, values, media_kind, batch_mode)
        process_choices, next_process_name = self.library.current_user_settings_choices(media_kind, main_state, main_lset_name)
        return (
            gr.update(choices=model_choices, value=ui_constants.ADD_USER_SETTINGS_MODEL_TYPE),
            gr.update(choices=process_choices, value=next_process_name),
            self.user_settings_hint_update(process_choices),
            updated_memory,
            next_process_name,
            *self.settings_action_updates(ui_constants.ADD_USER_SETTINGS_MODEL_TYPE, next_process_name),
            *self.restore_state(updated_memory, next_process_name, values.source_path, main_state, refs, media_kind, batch_mode),
        )
