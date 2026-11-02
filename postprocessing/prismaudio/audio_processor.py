from __future__ import annotations

import os
from typing import Any, Callable

import torch
from safetensors.torch import save_file

from postprocessing.audio_processors import AUDIO_PROCESSOR_TYPE_SOUNDTRACK, method_metadata, read_config_section
from postprocessing.prismaudio import (
    PRISMAUDIO_CONFIG_KEY,
    PRISMAUDIO_FOLDER,
    PRISMAUDIO_METHOD,
    PRISMAUDIO_MODEL_FILENAME,
    PRISMAUDIO_REPO_ID,
    PRISMAUDIO_SYNCHFORMER_FILENAME,
    PRISMAUDIO_VAE_FILENAME,
)


_UPSTREAM_REPO_ID = "FunAudioLLM/PrismAudio"
_UPSTREAM_MODEL_FILENAME = "prismaudio.ckpt"
_UPSTREAM_VAE_FILENAME = "vae.ckpt"
_UPSTREAM_SYNCHFORMER_FILENAME = "synchformer_state_dict.pth"
_T5_FOLDER = "t5gemma-l-l-ul2-it"
_T5_FILES = ["config.json", "model.safetensors", "generation_config.json", "tokenizer_config.json", "tokenizer.model", "tokenizer.json", "special_tokens_map.json", "chat_template.jinja"]
_VIDEOPRISM_FOLDER = "videoprism-lvt-large-f8r288"
_VIDEOPRISM_FILENAME = "flax_lvt_large_f8r288_repeated.npz"
_VIDEOPRISM_TOKENIZER_FOLDER = "videoprism_c4_en"
_VIDEOPRISM_TOKENIZER_FILENAME = "sentencepiece.model"
_PACKAGED_FILES = (
    (PRISMAUDIO_MODEL_FILENAME, _UPSTREAM_MODEL_FILENAME),
    (PRISMAUDIO_VAE_FILENAME, _UPSTREAM_VAE_FILENAME),
    (PRISMAUDIO_SYNCHFORMER_FILENAME, _UPSTREAM_SYNCHFORMER_FILENAME),
)


def _load_checkpoint_tensors(path: str) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
    return {key: value.detach().cpu().to(torch.bfloat16).contiguous() if torch.is_floating_point(value) else value.detach().cpu().contiguous() for key, value in state_dict.items()}


def _save_bf16_safetensors(source: str, target: str) -> None:
    temp_target = target + ".tmp"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    save_file(_load_checkpoint_tensors(source), temp_target)
    os.replace(temp_target, target)


class PrismAudioProcessor:
    MODE_OFF = 0
    MODE_PRISMAUDIO = 1
    DEFAULT_STEPS = 24
    DEFAULT_CFG_SCALE = 5.0
    MODE_CHOICES = [("PrismAudio", MODE_PRISMAUDIO)]

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def query_audio_processor_def(cls) -> dict[str, Any]:
        return {
            "name": "PrismAudio",
            "processor_types": (AUDIO_PROCESSOR_TYPE_SOUNDTRACK,),
            "methods": [("PrismAudio", PRISMAUDIO_METHOD)],
            "method_types": {PRISMAUDIO_METHOD: (AUDIO_PROCESSOR_TYPE_SOUNDTRACK,)},
            "needs_prompt": {PRISMAUDIO_METHOD: True},
            "supports_repeat": {PRISMAUDIO_METHOD: True},
            "status": {PRISMAUDIO_METHOD: "PrismAudio Soundtrack Generation"},
            "config_key": PRISMAUDIO_CONFIG_KEY,
            "pos": 21,
            "description": "Generate a prompt-guided soundtrack for the video with PrismAudio.",
        }

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"mode": cls.MODE_PRISMAUDIO, "steps": cls.DEFAULT_STEPS, "cfg_scale": cls.DEFAULT_CFG_SCALE}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        normalized.update(config or {})
        try:
            normalized["mode"] = int(normalized["mode"])
        except (TypeError, ValueError):
            normalized["mode"] = cls.MODE_PRISMAUDIO
        try:
            normalized["steps"] = int(normalized["steps"])
        except (TypeError, ValueError):
            normalized["steps"] = cls.DEFAULT_STEPS
        try:
            normalized["cfg_scale"] = float(normalized["cfg_scale"])
        except (TypeError, ValueError):
            normalized["cfg_scale"] = cls.DEFAULT_CFG_SCALE
        if normalized["mode"] != cls.MODE_PRISMAUDIO:
            normalized["mode"] = cls.MODE_PRISMAUDIO
        normalized["steps"] = max(1, min(200, normalized["steps"]))
        normalized["cfg_scale"] = max(0.0, min(25.0, normalized["cfg_scale"]))
        return normalized

    def config(self, server_config: dict[str, Any] | None = None) -> dict[str, Any]:
        return read_config_section(self.server_config if server_config is None else server_config, self)

    def enabled(self) -> bool:
        return True

    def persistent_models(self) -> bool:
        from postprocessing import audio_processors as audio_processor_api

        return audio_processor_api.persistent_models(self.server_config)

    def _locate_file(self, filename: str) -> str:
        return self.files_locator.locate_file(os.path.join(PRISMAUDIO_FOLDER, filename))

    def _locate_folder(self, folder: str) -> str:
        return self.files_locator.locate_folder(os.path.join(PRISMAUDIO_FOLDER, folder))

    def _checkpoint_path(self, filename: str) -> str:
        return self.files_locator.locate_file(os.path.join(PRISMAUDIO_FOLDER, filename), create_path_if_none=True)

    def _required_files(self) -> list[str]:
        return [
            os.path.join(PRISMAUDIO_FOLDER, PRISMAUDIO_MODEL_FILENAME),
            os.path.join(PRISMAUDIO_FOLDER, PRISMAUDIO_VAE_FILENAME),
            os.path.join(PRISMAUDIO_FOLDER, PRISMAUDIO_SYNCHFORMER_FILENAME),
            *[os.path.join(PRISMAUDIO_FOLDER, _T5_FOLDER, name) for name in _T5_FILES],
            os.path.join(PRISMAUDIO_FOLDER, _VIDEOPRISM_FOLDER, _VIDEOPRISM_FILENAME),
            os.path.join(PRISMAUDIO_FOLDER, _VIDEOPRISM_TOKENIZER_FOLDER, _VIDEOPRISM_TOKENIZER_FILENAME),
        ]

    def _has_required_file(self, path: str) -> bool:
        located = self.files_locator.locate_file(path, error_if_none=False)
        return located is not None and os.path.getsize(located) > 0

    def paths(self):
        from postprocessing.prismaudio.runtime import PrismAudioPaths

        return PrismAudioPaths(
            model=self._locate_file(PRISMAUDIO_MODEL_FILENAME),
            vae=self._locate_file(PRISMAUDIO_VAE_FILENAME),
            synchformer=self._locate_file(PRISMAUDIO_SYNCHFORMER_FILENAME),
            t5_model_dir=self._locate_folder(_T5_FOLDER),
            videoprism=self._locate_file(os.path.join(_VIDEOPRISM_FOLDER, _VIDEOPRISM_FILENAME)),
            videoprism_tokenizer=self._locate_file(os.path.join(_VIDEOPRISM_TOKENIZER_FOLDER, _VIDEOPRISM_TOKENIZER_FILENAME)),
        )

    def validate_method(self, method, video_source=None, frames_count=None, fps=None, duration=None, **_kwargs) -> str:
        del method
        if not self.enabled():
            return "PrismAudio is disabled in Configuration > Extensions"
        from postprocessing.prismaudio.runtime import requirement_message

        missing = requirement_message()
        if missing:
            return missing
        if video_source is not None:
            from shared.utils.utils import get_video_info

            fps, _, _, frames_count = get_video_info(video_source)
        elif duration is not None:
            try:
                return "" if float(duration) >= 1 else "PrismAudio can generate an audio track only if the video is at least 1s long"
            except (TypeError, ValueError):
                return ""
        if frames_count is None or fps is None:
            return ""
        return "" if frames_count >= round(float(fps)) else "PrismAudio can generate an audio track only if the video is at least 1s long"

    def query_download_defs(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        if enabled_only and not self.enabled():
            return []
        return [{
            "repoId": PRISMAUDIO_REPO_ID,
            "sourceFolderList": [PRISMAUDIO_FOLDER],
            "fileList": [[PRISMAUDIO_MODEL_FILENAME, PRISMAUDIO_VAE_FILENAME, PRISMAUDIO_SYNCHFORMER_FILENAME]],
        }, {
            "repoId": PRISMAUDIO_REPO_ID,
            "sourceFolderList": [f"{PRISMAUDIO_FOLDER}/{_T5_FOLDER}"],
            "fileList": [_T5_FILES],
        }, {
            "repoId": PRISMAUDIO_REPO_ID,
            "sourceFolderList": [f"{PRISMAUDIO_FOLDER}/{_VIDEOPRISM_FOLDER}"],
            "fileList": [[_VIDEOPRISM_FILENAME]],
        }, {
            "repoId": PRISMAUDIO_REPO_ID,
            "sourceFolderList": [f"{PRISMAUDIO_FOLDER}/{_VIDEOPRISM_TOKENIZER_FOLDER}"],
            "fileList": [[_VIDEOPRISM_TOKENIZER_FILENAME]],
        }]

    @staticmethod
    def _upstream_download_def():
        return {"repoId": _UPSTREAM_REPO_ID, "sourceFolderList": [""], "targetFolderList": [PRISMAUDIO_FOLDER],
                "fileList": [[source for _, source in _PACKAGED_FILES]]}

    def query_checkpoint_inventory(self):
        return [{"name": "PrismAudio", "variant": "BF16", "download_defs": self.query_download_defs(enabled_only=False)},
                {"name": "PrismAudio", "variant": "Upstream Conversion Files", "download_defs": [self._upstream_download_def()]}]

    def _package_from_upstream(self, process_files: Callable[..., Any], send_cmd=None) -> None:
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, "Downloading upstream PrismAudio checkpoints for local safetensors packaging...")
        process_files(**self._upstream_download_def())
        send_download_status(send_cmd, "Packaging PrismAudio checkpoints as bf16 safetensors...")
        for target_name, source_name in _PACKAGED_FILES:
            target_path = self._checkpoint_path(target_name)
            if os.path.isfile(target_path):
                continue
            _save_bf16_safetensors(self._checkpoint_path(source_name), target_path)
        for _, source_name in _PACKAGED_FILES:
            source_path = self.files_locator.locate_file(os.path.join(PRISMAUDIO_FOLDER, source_name), error_if_none=False)
            if source_path is not None:
                os.remove(source_path)

    def _download_auxiliary_assets(self, process_files: Callable[..., Any]) -> None:
        for download_def in self.query_download_defs(enabled_only=True)[1:]:
            process_files(**download_def)

    def download(self, method, process_files: Callable[..., Any], send_cmd=None, status_text=None, **_kwargs) -> bool:
        del method
        required = self._required_files()
        if all(self._has_required_file(path) for path in required):
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text or "Downloading PrismAudio model files...")
        audio_files = [os.path.join(PRISMAUDIO_FOLDER, name) for name in (PRISMAUDIO_MODEL_FILENAME, PRISMAUDIO_VAE_FILENAME, PRISMAUDIO_SYNCHFORMER_FILENAME)]
        if not all(self._has_required_file(path) for path in audio_files):
            try:
                process_files(**self.query_download_defs(enabled_only=True)[0])
            except Exception as exc:
                from huggingface_hub.errors import EntryNotFoundError

                from shared.utils.download import DownloadError
                cause = exc.__cause__ if isinstance(exc, DownloadError) else exc
                if not isinstance(cause, EntryNotFoundError):
                    raise
                self._package_from_upstream(process_files, send_cmd=send_cmd)
                if not all(self._has_required_file(path) for path in audio_files):
                    expected = ", ".join(audio_files)
                    raise RuntimeError(f"PrismAudio checkpoint packaging did not create expected files: {expected}") from exc
        self._download_auxiliary_assets(process_files)
        missing = [path for path in required if not self._has_required_file(path)]
        if missing:
            raise RuntimeError("PrismAudio download did not create expected files: " + ", ".join(missing))
        return True

    def generate_soundtrack(
        self,
        method,
        video_path,
        prompt="",
        negative_prompt="",
        seed=-1,
        duration=0,
        output_path=None,
        send_cmd=None,
        status_callback=None,
        verbose_level=1,
        audio_codec_key="aac_128",
        process_files=None,
        init_pipe=None,
        profile=4,
        abort_callback=None,
        progress_callback=None,
        **_kwargs,
    ) -> str:
        del negative_prompt, send_cmd, audio_codec_key
        config = self.config()
        if int(config["mode"]) != self.MODE_PRISMAUDIO:
            raise RuntimeError("PrismAudio is disabled in Configuration > Extensions")
        if status_callback is not None:
            status_callback(method_metadata(method)["status"])
        if process_files is not None:
            self.download(method, process_files)
        from postprocessing.prismaudio.runtime import generate_audio

        return generate_audio(
            self.paths(),
            video_path,
            output_path,
            prompt=prompt,
            seed=seed,
            duration=duration,
            steps=config["steps"],
            cfg_scale=config["cfg_scale"],
            persistent_models=self.persistent_models(),
            init_pipe=init_pipe,
            profile=profile,
            abort_callback=abort_callback,
            progress_callback=progress_callback,
            verbose_level=verbose_level,
        )

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        config = self.normalize_config_section(config)
        with gr.Group():
            with gr.Row():
                mode = gr.Dropdown(choices=self.MODE_CHOICES, value=config["mode"], label="PrismAudio Soundtrack Generation (optional PrismAudio dependencies required)", interactive=not lock_config)
            with gr.Row():
                steps = gr.Slider(1, 200, value=config["steps"], step=1, label="PrismAudio Steps", interactive=not lock_config)
                cfg_scale = gr.Slider(0.0, 25.0, value=config["cfg_scale"], step=0.1, label="PrismAudio CFG Scale", interactive=not lock_config)
        return [("mode", mode), ("steps", steps), ("cfg_scale", cfg_scale)]

    def validate_config_section(self, config: dict[str, Any]) -> str:
        if int(config["mode"]) == self.MODE_OFF:
            return ""
        from postprocessing.prismaudio.runtime import requirement_message

        return requirement_message() or ""

    def release_vram(self) -> None:
        from postprocessing.prismaudio.runtime import release_models

        release_models()
