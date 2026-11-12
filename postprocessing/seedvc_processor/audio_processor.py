# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os
from typing import Any, Callable

from postprocessing.audio_processors import (
    AUDIO_PROCESSOR_TYPE_AUDIO_EDIT,
    AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT,
    AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING,
    SEEDVC_ONE_SPEAKER_METHOD,
    SEEDVC_TWO_SPEAKERS_METHOD,
    method_metadata,
    read_config_section,
)


class SeedVCProcessor:
    def __init__(self, server_config: dict[str, Any], files_locator):
        from postprocessing.seedvc_processor.bridge import SeedVCBridge

        self.bridge_config = {}
        self.bridge = SeedVCBridge(self.bridge_config, files_locator)
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def query_audio_processor_def(cls) -> dict[str, Any]:
        return {
            "name": "SeedVC",
            "processor_types": (AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, AUDIO_PROCESSOR_TYPE_AUDIO_EDIT),
            "methods": [("SeedVC - One Speaker", SEEDVC_ONE_SPEAKER_METHOD), ("SeedVC - Two Speakers", SEEDVC_TWO_SPEAKERS_METHOD)],
            "method_types": {
                SEEDVC_ONE_SPEAKER_METHOD: (AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, AUDIO_PROCESSOR_TYPE_AUDIO_EDIT),
                SEEDVC_TWO_SPEAKERS_METHOD: (AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, AUDIO_PROCESSOR_TYPE_AUDIO_EDIT),
            },
            "needs_voice_sample": {SEEDVC_ONE_SPEAKER_METHOD: True, SEEDVC_TWO_SPEAKERS_METHOD: True},
            "needs_voice_sample2": {SEEDVC_TWO_SPEAKERS_METHOD: True},
            "speaker_count": {SEEDVC_ONE_SPEAKER_METHOD: 1, SEEDVC_TWO_SPEAKERS_METHOD: 2},
            "status": {SEEDVC_ONE_SPEAKER_METHOD: "Replace Voice", SEEDVC_TWO_SPEAKERS_METHOD: "Replace Voice"},
            "method_context_labels": {
                AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING: {
                    SEEDVC_ONE_SPEAKER_METHOD: "Replace voice (SeedVC) - One Speaker",
                    SEEDVC_TWO_SPEAKERS_METHOD: "Replace voice (SeedVC) - Two Speakers",
                },
            },
            "config_key": "seedvc",
            "pos": 30,
            "method_descriptions": {
                SEEDVC_ONE_SPEAKER_METHOD: "Replace one speaker's voice using a target voice sample.",
                SEEDVC_TWO_SPEAKERS_METHOD: "Replace two speakers' voices using two target voice samples.",
            },
        }

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        from postprocessing.seedvc_processor.bridge import SeedVCBridge

        return {"mode": SeedVCBridge.MODE_SINGING}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        from postprocessing.seedvc_processor.bridge import SeedVCBridge

        mode = config.get("mode", SeedVCBridge.MODE_SINGING)
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            mode = SeedVCBridge.MODE_SINGING
        if mode not in SeedVCBridge._VERSIONS:
            mode = SeedVCBridge.MODE_SINGING
        return {"mode": mode}

    def _sync_bridge_config(self, config: dict[str, Any] | None = None) -> None:
        from postprocessing import audio_processors as audio_processor_api

        values = read_config_section(self.server_config, self) if config is None else self.normalize_config_section(config)
        self.bridge_config.clear()
        persistence = self.bridge.PERSIST_RAM if audio_processor_api.persistent_models(self.server_config) else self.bridge.PERSIST_UNLOAD
        self.bridge_config.update({"seedvc_mode": values["mode"], "seedvc_persistence": persistence})
        self.bridge.normalize_config()

    def enabled(self) -> bool:
        self._sync_bridge_config()
        return self.bridge.enabled()

    def format_choice_label(self, label, method) -> str:
        version = self.bridge._VERSIONS.get(read_config_section(self.server_config, self)["mode"])
        return label.replace("SeedVC", f"SeedVC {version}") if version else label

    def validate_method(self, method, voice_sample=None, voice_sample2=None, **_kwargs) -> str:
        if voice_sample is None:
            return "You must provide a voice sample"
        metadata = method_metadata(method)
        if metadata["needs_voice_sample2"] and voice_sample2 is None:
            return "You must provide a second voice sample"
        return ""

    def query_checkpoint_inventory(self):
        from .assets import query_download_def
        from .bridge import SEEDVC_RESTORE_BACKGROUND_STEM
        from shared.utils.download import query_audio_background_replacement_download_def
        shared = [query_audio_background_replacement_download_def()] if SEEDVC_RESTORE_BACKGROUND_STEM else []
        return [{"name": "SeedVC", "variant": label, "download_defs": query_download_def(mode) + shared} for mode, label in self.bridge._VERSIONS.items()]

    def query_download_defs(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        self._sync_bridge_config()
        return self.bridge.query_download_def(enabled_only=enabled_only)

    def download(self, method, process_files: Callable[..., Any], send_cmd=None, status_text=None, **_kwargs) -> bool:
        self._sync_bridge_config()
        downloaded = self.bridge.download(process_files, send_cmd=send_cmd, status_text=status_text or "Downloading SeedVC model files...")
        if method_metadata(method)["needs_voice_sample2"]:
            from preprocessing.speaker_separator.assets import download_speaker_separator

            download_speaker_separator(send_cmd, process_files=process_files)
            downloaded = True
        return downloaded

    def replace_voice_tracks(self, method, audio_tracks: list[str], voice_sample=None, output_dir="", prefix="seedvc", process_files=None, profile_no=4, verbose_level=1, init_pipe=None, voice_sample2=None, **_kwargs) -> tuple[list[str], list[str]]:
        self._sync_bridge_config()
        return self.bridge.replace_audio_tracks(audio_tracks, voice_sample, output_dir, prefix, process_files=process_files, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, voice_sample2_path=voice_sample2, speaker_count=method_metadata(method)["speaker_count"])

    def process_audio_file(self, method, audio_source, voice_sample=None, output_path=None, process_files=None, profile_no=4, verbose_level=1, init_pipe=None, voice_sample2=None, status_callback=None, **_kwargs) -> str:
        self._sync_bridge_config()
        if status_callback is not None:
            status_callback(method_metadata(method)["status"])
        return self.bridge.replace_audio_file(audio_source, voice_sample, output_path, process_files=process_files, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, voice_sample2_path=voice_sample2, speaker_count=method_metadata(method)["speaker_count"], prefix=f"tmp_{os.path.splitext(os.path.basename(audio_source))[0]}")

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        from postprocessing.seedvc_processor.bridge import SeedVCBridge
        config = self.normalize_config_section(config)
        mode = gr.Dropdown(choices=SeedVCBridge.mode_choices(), value=config["mode"], label="SeedVC Voice Replacement", info="Showrunner's singer replacement requires v1.0 Singing. Speech modes are for spoken audio.", interactive=not lock_config)
        return [("mode", mode)]

    def config_requires_release(self, old_config: dict[str, Any], new_config: dict[str, Any], changed_keys) -> bool:
        return old_config != new_config

    def release_vram(self) -> None:
        self.bridge.release_vram()
