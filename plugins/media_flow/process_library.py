# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import gradio as gr

from shared.utils.utils import get_outpainting_dims
from shared.utils.settings_bundle import SETTINGS_BUNDLE_ATTACHMENT_KEYS, WAN_GP_SETTINGS_SUFFIXES, is_wangp_settings_filename, load_first_settings_from_queue_zip
from postprocessing import spatial_upsamplers as spatial_upsampler_api

from . import common
from . import constants
from . import frame_planning as frames
from . import process_catalog as catalog
from . import process_validation
from . import system_handlers


class ProcessLibrary:
    def __init__(self, *, get_model_def, get_lora_dir, get_base_model_type) -> None:
        self.get_model_def = get_model_def
        self.get_lora_dir = get_lora_dir
        self.get_base_model_type = get_base_model_type

    def model_type_label(self, model_type: str) -> str:
        if len(str(model_type or "").strip()) == 0:
            return "Unknown Model"
        for process_definition in catalog.PROCESS_DEFINITIONS.values():
            settings = process_definition.get("settings", {})
            if str(settings.get("model_type") or "").strip() != str(model_type):
                continue
            model_label = str(settings.get("model_label") or "").strip()
            if len(model_label) > 0:
                return model_label
        handler = self.system_handler_for_model_type(str(model_type))
        if handler is not None:
            return str(getattr(handler, "model_label", str(model_type)))
        try:
            model_def = frames.require_model_def(str(model_type), self.get_model_def)
        except gr.Error:
            return str(model_type)
        model_block = model_def.get("model")
        if isinstance(model_block, dict):
            model_name = str(model_block.get("name") or "").strip()
            if len(model_name) > 0:
                return model_name
        model_name = str(model_def.get("name") or "").strip()
        return model_name if len(model_name) > 0 else str(model_type)

    def base_model_type_for_ref(self, model_type: str) -> str:
        model_type = str(model_type or "").strip()
        base_model_type = str(self.get_base_model_type(model_type) or "").strip()
        if len(base_model_type) > 0:
            return base_model_type
        return model_type

    def resolve_user_settings_ref(self, ref: str) -> Path | None:
        ref = catalog.normalize_user_settings_ref(ref)
        if len(ref) == 0:
            return None
        base_model_type, filename = ref.split("/", 1)
        lora_dir = Path(self.get_lora_dir(base_model_type))
        settings_path = (lora_dir / Path(filename).name).resolve()
        if settings_path.is_file() and settings_path.suffix.lower() in WAN_GP_SETTINGS_SUFFIXES:
            return settings_path
        return None

    @staticmethod
    def load_settings_payload(settings_path: Path) -> dict | None:
        try:
            if settings_path.suffix.lower() == ".zip":
                payload, source_task_count = load_first_settings_from_queue_zip(settings_path, SETTINGS_BUNDLE_ATTACHMENT_KEYS)
                if source_task_count > 1:
                    print(f"[MediaFlow] Settings bundle {settings_path.name} contains {source_task_count} tasks; only the first task was extracted.")
            else:
                payload = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile):
            return None
        return payload if isinstance(payload, dict) else None

    def build_user_process_definition(self, ref: str) -> dict | None:
        normalized_ref = catalog.normalize_user_settings_ref(ref)
        settings_path = self.resolve_user_settings_ref(normalized_ref)
        if settings_path is None:
            return None
        payload = self.load_settings_payload(settings_path)
        if not isinstance(payload, dict):
            return None
        model_type = str(payload.get("model_type") or "").strip()
        if len(model_type) == 0:
            return None
        return {
            "settings": payload,
            "path": str(settings_path),
            "source": "user",
            "ref": normalized_ref,
            "name": settings_path.stem,
            "value": catalog.user_process_value(normalized_ref),
        }

    def user_process_definitions(self, user_refs: list[str], media_kind: str | None = None) -> dict[str, dict]:
        definitions: dict[str, dict] = {}
        for ref in catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs}):
            definition = self.build_user_process_definition(ref)
            if definition is None:
                continue
            if media_kind is not None and catalog.process_definition_media_kind(definition) != catalog.normalize_media_kind(media_kind):
                continue
            value = str(definition.get("value") or "")
            if len(value) > 0:
                definitions[value] = definition
        return definitions

    @staticmethod
    def system_process_definition(process_name: str) -> dict | None:
        process_definition = catalog.PROCESS_DEFINITIONS.get(str(process_name or "").strip())
        if not isinstance(process_definition, dict):
            return None
        settings = process_definition["settings"]
        if process_definition["info"].get("use_configured_model"):
            handler = system_handlers.get_system_handler(settings["system_handler"])
            method = handler.configured_method(settings["spatial_upsampling_method"])
            _, _, target = spatial_upsampler_api.normalize_upsampling_value_for_method(method, settings["target_ratio"])
            settings = {**settings, "spatial_upsampling_method": method, "target_ratio": target}
        return {
            "settings": settings,
            "path": process_definition.get("path", ""),
            "info": process_definition["info"],
            "source": "system",
            "name": str(process_name or "").strip(),
            "value": str(process_name or "").strip(),
        }

    @staticmethod
    def main_model_type(main_state: dict | None) -> str:
        if not isinstance(main_state, dict):
            return ""
        key = "model_type" if main_state.get("active_form", "add") == "add" else "edit_model_type"
        return str(main_state.get(key) or main_state.get("model_type") or "").strip()

    @staticmethod
    def current_user_settings_filenames(main_state: dict | None) -> list[str]:
        if not isinstance(main_state, dict):
            return []
        loras_presets = main_state.get("loras_presets", [])
        filenames: list[str] = []
        seen: set[str] = set()
        for item in loras_presets:
            filename = str(item or "").strip()
            if "/" in filename or "\\" in filename or not is_wangp_settings_filename(filename):
                continue
            if filename.casefold() in seen:
                continue
            filenames.append(filename)
            seen.add(filename.casefold())
        return sorted(filenames, key=lambda name: Path(name).stem.casefold())

    def normalize_main_lset_selection(self, main_state: dict | None, main_lset_name: str | None) -> str:
        selection = str(main_lset_name or "").strip()
        filenames = self.current_user_settings_filenames(main_state)
        if selection in filenames:
            return selection
        normalized_label = selection.replace("\u2500", "").replace(chr(160), " ").strip().casefold()
        for filename in filenames:
            if Path(filename).stem.casefold() == normalized_label:
                return filename
        return ""

    def resolve_current_user_settings_file(self, main_state: dict | None, settings_filename: str) -> Path | None:
        filename = str(settings_filename or "").strip()
        if filename not in self.current_user_settings_filenames(main_state):
            return None
        model_type = self.main_model_type(main_state)
        if len(model_type) == 0:
            return None
        lora_dir = Path(self.get_lora_dir(model_type))
        settings_path = (lora_dir / Path(filename).name).resolve()
        if settings_path.is_file() and settings_path.suffix.lower() in WAN_GP_SETTINGS_SUFFIXES:
            return settings_path
        return None

    def build_current_user_settings_ref(self, main_state: dict | None, settings_filename: str) -> str:
        model_type = self.main_model_type(main_state)
        base_model_type = self.base_model_type_for_ref(model_type)
        filename = Path(str(settings_filename or "").strip()).name
        return catalog.normalize_user_settings_ref(f"{base_model_type}/{filename}")

    def build_candidate_user_process_definition(self, main_state: dict | None, settings_filename: str) -> dict | None:
        settings_path = self.resolve_current_user_settings_file(main_state, settings_filename)
        if settings_path is None:
            return None
        payload = self.load_settings_payload(settings_path)
        if not isinstance(payload, dict):
            return None
        model_type = str(payload.get("model_type") or self.main_model_type(main_state)).strip()
        if len(model_type) == 0:
            return None
        payload = payload.copy()
        payload["model_type"] = model_type
        ref = self.build_current_user_settings_ref(main_state, settings_path.name)
        return {
            "settings": payload,
            "path": str(settings_path),
            "source": "user",
            "ref": ref,
            "name": settings_path.stem,
            "value": settings_path.name,
        }

    def process_definition(self, process_value: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> dict | None:
        process_value = str(process_value or "").strip()
        if len(process_value) == 0 or process_value == constants.NO_USER_SETTINGS_VALUE:
            return None
        system_definition = self.system_process_definition(process_value)
        if system_definition is not None:
            return system_definition
        if catalog.is_user_process_value(process_value):
            ref = catalog.user_process_ref_from_value(process_value)
            if len(ref) == 0:
                return None
            if user_refs is not None and ref.casefold() not in {item.casefold() for item in catalog.get_saved_user_settings_refs({catalog.USER_SETTINGS_STORAGE_KEY: user_refs})}:
                return None
            return self.build_user_process_definition(ref)
        if is_wangp_settings_filename(process_value):
            return self.build_candidate_user_process_definition(main_state, process_value)
        return None

    @staticmethod
    def process_definition_model_type(process_definition: dict | None) -> str:
        settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
        return str(settings.get("model_type") or "").strip() if isinstance(settings, dict) else ""

    @staticmethod
    def process_definition_is_image(process_definition: dict | None) -> bool:
        return catalog.process_definition_media_kind(process_definition) == "image"

    def is_image_process(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
        return self.process_definition_is_image(self.process_definition(process_name, main_state, user_refs))

    def media_kind_for_process(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> str:
        return catalog.process_definition_media_kind(self.process_definition(process_name, main_state, user_refs))

    def media_kind_for_user_ref(self, ref: str) -> str:
        definition = self.build_user_process_definition(ref)
        return catalog.process_definition_media_kind(definition) if definition is not None else ""

    @staticmethod
    def system_handler_for_definition(process_definition: dict | None):
        settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
        if not isinstance(settings, dict):
            return None
        return system_handlers.get_system_handler(settings.get("system_handler"))

    @staticmethod
    def system_handler_for_model_type(model_type: str):
        model_type = str(model_type or "").strip()
        for process_definition in catalog.PROCESS_DEFINITIONS.values():
            settings = process_definition.get("settings", {})
            if str(settings.get("model_type") or "").strip() != model_type:
                continue
            handler = system_handlers.get_system_handler(settings.get("system_handler"))
            if handler is not None:
                return handler
        return None

    def system_handler_for_process(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None):
        return self.system_handler_for_definition(self.process_definition(process_name, main_state, user_refs))

    def spatial_upsampler_method(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> str:
        process_definition = self.process_definition(process_name, main_state, user_refs)
        handler = self.system_handler_for_definition(process_definition)
        if handler is not None:
            return "" if bool(getattr(handler, "temporal", False)) else str(getattr(handler, "method", "") or "")
        settings = process_definition.get("settings", {}) if isinstance(process_definition, dict) else {}
        value = str(settings.get("spatial_upsampling") or settings.get("spatial_upsampling_method") or "").strip()
        split = spatial_upsampler_api.split_upsampling_value(value)
        return split[0] if split is not None else value if spatial_upsampler_api.find_upsampler_by_method(value) is not None else ""

    def spatial_upsampler_parameters(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> list[dict]:
        """The parameters the process's spatial processor shows in Media Flow (postprocessing.parameter_ui)."""
        return spatial_upsampler_api.method_parameters(self.spatial_upsampler_method(process_name, main_state, user_refs), ui_context=spatial_upsampler_api.PARAMETER_UI_MEDIA_FLOW)

    def target_control_choices(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> list[tuple[str, str]]:
        process_definition = self.process_definition(process_name, main_state, user_refs)
        handler = self.system_handler_for_definition(process_definition)
        if handler is not None and callable(getattr(handler, "target_control_choices_for_process", None)):
            return handler.target_control_choices_for_process(process_definition.get("settings", {}))
        return list(getattr(handler, "target_control_choices", [])) if handler is not None else []

    def target_control_default(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None, *, saved_value: str = "") -> str:
        process_definition = self.process_definition(process_name, main_state, user_refs)
        handler = self.system_handler_for_definition(process_definition)
        if saved_value and process_definition and process_definition.get("info", {}).get("use_configured_model"):
            return handler.normalize_target_control_for_process(saved_value, process_definition["settings"])
        if handler is not None and callable(getattr(handler, "target_control_default_for_process", None)):
            return handler.target_control_default_for_process(process_definition.get("settings", {}))
        return str(getattr(handler, "default_target_control", "")) if handler is not None else ""

    def has_target_control(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
        return len(self.target_control_choices(process_name, main_state, user_refs)) > 0

    def target_control_label(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> str:
        definition = self.process_definition(process_name, main_state, user_refs)
        handler = self.system_handler_for_definition(definition)
        if definition and definition.get("info", {}).get("use_configured_model"):
            method = definition["settings"]["spatial_upsampling_method"]
            _, upsampler, _ = spatial_upsampler_api.method_definition(method)
            name = next(label for label, key in upsampler["methods"] if key == method)
            return f"Upsampling ({name})"
        return str(getattr(handler, "target_control_label", "Target")) if handler is not None else "Target"

    def default_chunk_size_seconds(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> float:
        handler = self.system_handler_for_process(process_name, main_state, user_refs)
        if handler is not None:
            return float(getattr(handler, "default_chunk_size_seconds", 10.0))
        return 10.0

    def hides_chunk_size(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
        if self.is_image_process(process_name, main_state, user_refs):
            return True
        handler = self.system_handler_for_process(process_name, main_state, user_refs)
        return bool(getattr(handler, "hide_chunk_size", False)) if handler is not None else False

    def hides_sliding_window_overlap(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
        if self.is_image_process(process_name, main_state, user_refs):
            return True
        handler = self.system_handler_for_process(process_name, main_state, user_refs)
        return bool(getattr(handler, "hide_sliding_window_overlap", False)) if handler is not None else False

    def hides_output_resolution(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
        handler = self.system_handler_for_process(process_name, main_state, user_refs)
        return bool(getattr(handler, "hide_output_resolution", False)) if handler is not None else False

    def hides_prompt(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
        handler = self.system_handler_for_process(process_name, main_state, user_refs)
        return bool(getattr(handler, "hide_prompt", False)) if handler is not None else False

    def process_frame_rules(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> frames.FramePlanRules:
        process_definition = self.process_definition_or_default(process_name, main_state, user_refs)
        handler = self.system_handler_for_definition(process_definition)
        if handler is not None:
            return frames.FramePlanRules(frame_step=int(getattr(handler, "frame_step", 1)), minimum_requested_frames=int(getattr(handler, "minimum_requested_frames", 1)), frame_offset=int(getattr(handler, "frame_offset", 1)))
        model_type = self.process_definition_model_type(process_definition)
        return frames.get_frame_plan_rules(model_type, self.get_model_def)

    def process_values_by_model_type(self, media_kind: str, user_refs: list[str]) -> dict[str, list[str]]:
        media_kind = catalog.normalize_media_kind(media_kind)
        values_by_model_type: dict[str, list[str]] = {}
        for process_name, process_definition in catalog.PROCESS_DEFINITIONS.items():
            if catalog.process_definition_media_kind(process_definition) != media_kind:
                continue
            model_type = str(process_definition.get("settings", {}).get("model_type") or "").strip()
            if len(model_type) > 0:
                values_by_model_type.setdefault(model_type, []).append(process_name)
        for value, definition in self.user_process_definitions(user_refs, media_kind).items():
            model_type = self.process_definition_model_type(definition)
            if len(model_type) > 0:
                values_by_model_type.setdefault(model_type, []).append(value)
        return values_by_model_type

    def model_type_choices(self, media_kind: str, user_refs: list[str]) -> list[tuple[str, str]]:
        model_types = sorted(self.process_values_by_model_type(media_kind, user_refs), key=lambda item: self.model_type_label(item).casefold())
        choices = [(self.model_type_label(model_type), model_type) for model_type in model_types]
        choices.append((constants.ADD_USER_SETTINGS_LABEL, constants.ADD_USER_SETTINGS_MODEL_TYPE))
        return choices

    def normal_process_choices(self, media_kind: str, model_type: str, user_refs: list[str]) -> list[tuple[str, str]]:
        media_kind = catalog.normalize_media_kind(media_kind)
        model_type = str(model_type or "").strip()
        entries: list[tuple[str, str, str]] = []
        for process_name, process_definition in catalog.PROCESS_DEFINITIONS.items():
            if catalog.process_definition_media_kind(process_definition) == media_kind and str(process_definition.get("settings", {}).get("model_type") or "").strip() == model_type:
                entries.append((process_name, process_name, "system"))
        for value, definition in self.user_process_definitions(user_refs, media_kind).items():
            if self.process_definition_model_type(definition) == model_type:
                label_name = str(definition.get("name") or "").strip()
                if len(label_name) == 0:
                    ref = catalog.user_process_ref_from_value(value)
                    label_name = Path(ref or value).stem
                label = f"{label_name} *"
                entries.append((label, value, "user"))
        entries.sort(key=lambda item: (item[0].removesuffix(" *").casefold(), 0 if item[2] == "system" else 1))
        return [(label, value) for label, value, _source in entries]

    def current_user_settings_choices(self, media_kind: str, main_state: dict | None, main_lset_name: str | None) -> tuple[list[tuple[str, str]], str]:
        media_kind = catalog.normalize_media_kind(media_kind)
        filenames = self.current_user_settings_filenames(main_state)
        filenames = [
            filename
            for filename in filenames
            if catalog.process_definition_media_kind(self.build_candidate_user_process_definition(main_state, filename)) == media_kind
        ]
        if len(filenames) == 0:
            return [(constants.NO_USER_SETTINGS_LABEL, constants.NO_USER_SETTINGS_VALUE)], constants.NO_USER_SETTINGS_VALUE
        choices = [(Path(filename).stem, filename) for filename in filenames]
        selected = self.normalize_main_lset_selection(main_state, main_lset_name)
        value = selected if selected in filenames else filenames[0]
        return choices, value

    def process_choices(self, media_kind: str, process_model_type: str, main_state: dict | None, main_lset_name: str | None, user_refs: list[str]) -> tuple[list[tuple[str, str]], str | None]:
        if str(process_model_type or "").strip() == constants.ADD_USER_SETTINGS_MODEL_TYPE:
            return self.current_user_settings_choices(media_kind, main_state, main_lset_name)
        choices = self.normal_process_choices(media_kind, str(process_model_type or "").strip(), user_refs)
        return choices, choices[0][1] if len(choices) > 0 else None

    @staticmethod
    def process_choices_have_user_settings(process_choices: list[tuple[str, str]]) -> bool:
        return any(catalog.is_user_process_value(value) for _label, value in list(process_choices or []))

    def default_process_definition(self, media_kind: str = catalog.DEFAULT_MEDIA_KIND) -> dict:
        media_kind = catalog.normalize_media_kind(media_kind)
        definition = self.system_process_definition(catalog.default_process_name(media_kind))
        if definition is not None:
            return definition
        for process_name in catalog.PROCESS_DEFINITIONS:
            definition = self.system_process_definition(process_name)
            if definition is not None and catalog.process_definition_media_kind(definition) == media_kind:
                return definition
        return {"settings": {}, "path": "", "source": "system", "name": "", "value": ""}

    def process_definition_or_default(self, process_name: str, main_state: dict | None, user_refs: list[str] | None, media_kind: str = catalog.DEFAULT_MEDIA_KIND) -> dict:
        return self.process_definition(process_name, main_state, user_refs) or self.default_process_definition(media_kind)

    @staticmethod
    def settings_enable_outpaint(settings: dict | None) -> bool:
        if not isinstance(settings, dict) or "video_guide_outpainting" not in settings:
            return False
        return get_outpainting_dims(settings.get("video_guide_outpainting"), str(settings.get("video_guide_outpainting_ratio") or "").strip()) is not None

    @staticmethod
    def default_outpaint_target_ratio(settings: dict | None) -> str:
        value = str((settings or {}).get("video_guide_outpainting_ratio") or constants.DEFAULT_TARGET_RATIO).strip()
        values = {choice_value for _label, choice_value in constants.RATIO_CHOICES}
        return value if value in values else constants.DEFAULT_TARGET_RATIO

    @classmethod
    def normalize_outpaint_target_ratio(cls, settings: dict | None, target_ratio: str | None) -> str:
        value = str(target_ratio or "").strip()
        values = {choice_value for _label, choice_value in constants.RATIO_CHOICES}
        return value if value in values else cls.default_outpaint_target_ratio(settings)

    @classmethod
    def uses_builtin_outpaint_ui(cls, process_definition: dict | None) -> bool:
        settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
        if not isinstance(process_definition, dict) or not isinstance(settings, dict):
            return False
        if process_definition.get("source") != "user":
            return "video_guide_outpainting" in settings
        return cls.settings_enable_outpaint(settings)

    def has_process_outpaint(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
        if self.has_target_control(process_name, main_state, user_refs):
            return False
        process_definition = self.process_definition(process_name, main_state, user_refs)
        return self.uses_builtin_outpaint_ui(process_definition)

    @staticmethod
    def user_lora_strength_override_default(process_definition: dict | None) -> float | None:
        if not isinstance(process_definition, dict) or process_definition.get("source") != "user":
            return None
        settings = process_definition.get("settings")
        if not isinstance(settings, dict):
            return None
        return common.get_single_lora_simple_multiplier(settings)

    def is_process_strength_visible(self, process_name: str, main_state: dict | None = None, user_refs: list[str] | None = None) -> bool:
        process_definition = self.process_definition(process_name, main_state, user_refs)
        if self.system_handler_for_definition(process_definition) is not None:
            return False
        settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
        if not isinstance(settings, dict) or self.uses_builtin_outpaint_ui(process_definition):
            return False
        if process_definition.get("source") == "user":
            return self.user_lora_strength_override_default(process_definition) is not None
        return True

    def validate_user_process_definition(self, process_definition: dict | None) -> list[str]:
        return process_validation.validate_user_process_definition(process_definition, self.get_model_def)

    def format_user_process_validation_error(self, process_definition: dict | None, problems: list[str]) -> str:
        return process_validation.format_user_process_validation_error(process_definition, problems)

    def select_after_user_process_delete(self, media_kind: str, deleted_process_value: str, deleted_model_type: str, old_refs: list[str], new_refs: list[str]) -> tuple[str, str, list[tuple[str, str]]]:
        media_kind = catalog.normalize_media_kind(media_kind)
        old_choices = self.normal_process_choices(media_kind, deleted_model_type, old_refs)
        deleted_index = next((index for index, (_label, value) in enumerate(old_choices) if value == deleted_process_value), -1)
        new_choices = self.normal_process_choices(media_kind, deleted_model_type, new_refs)
        new_values = {value for _label, value in new_choices}
        if deleted_index >= 0:
            for _label, value in old_choices[deleted_index + 1:]:
                if catalog.is_user_process_value(value) and value in new_values:
                    return deleted_model_type, value, new_choices
        first_system_value = next((value for _label, value in new_choices if not catalog.is_user_process_value(value)), None)
        if first_system_value is not None:
            return deleted_model_type, first_system_value, new_choices
        for model_type in sorted(self.process_values_by_model_type(media_kind, new_refs), key=lambda item: self.model_type_label(item).casefold()):
            choices = self.normal_process_choices(media_kind, model_type, new_refs)
            first_system_value = next((value for _label, value in choices if not catalog.is_user_process_value(value)), None)
            if first_system_value is not None:
                return model_type, first_system_value, choices
        if len(new_choices) > 0:
            return deleted_model_type, new_choices[0][1], new_choices
        default_model_type = catalog.default_model_type(media_kind)
        default_process_name = catalog.default_process_name(media_kind)
        return default_model_type, default_process_name, self.normal_process_choices(media_kind, default_model_type, new_refs)
