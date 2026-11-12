# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations
from shared.utils.media_control import controlled_model_loading, loading_callback, checkpoint_modules

import gc
import os
import tempfile
from typing import Any, Callable


_persistent_converter = None
_persistent_offloadobj = None
_persistent_profile = None
KEEP_ORIGINAL_AUDIO_OUTSIDE_TWO_SPEAKERS = True
SEEDVC_RESTORE_BACKGROUND_STEM = True


def _release_runtime_objects(converter=None, offloadobj=None) -> None:
    import torch
    from shared.utils import offload_registry

    if offloadobj is not None:
        offload_registry.unregister_offloadobj("SeedVC", offloadobj)
        offloadobj.unload_all()
        offloadobj.release()
    del converter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def release_models() -> None:
    global _persistent_converter, _persistent_offloadobj, _persistent_profile

    _release_runtime_objects(_persistent_converter, _persistent_offloadobj)
    _persistent_converter = None
    _persistent_offloadobj = None
    _persistent_profile = None


@controlled_model_loading
def _get_runtime(persistent_models: bool, profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, mode: int = 1):
    import torch
    from mmgp import offload
    from postprocessing import seedvc
    from postprocessing.seedvc_processor.assets import normalize_mode

    global _persistent_converter, _persistent_offloadobj, _persistent_profile

    mode = normalize_mode(mode)
    profile_key = (profile_no, mode)
    if _persistent_offloadobj is not None and _persistent_profile != profile_key:
        release_models()

    keep_alive = persistent_models
    if _persistent_offloadobj is None:
        converter = seedvc.get_model(dtype=torch.float16, mode=mode)
        pipe = seedvc.get_pipe(profile_no=profile_no, model=converter, mode=mode)
        offload_kwargs = {"coTenantsMap": seedvc.get_cotenants_map(pipe)}
        if init_pipe is not None:
            profile_no = init_pipe(pipe, offload_kwargs, profile_no)
        offload_kwargs["pinnedMemory"] = False
        offloadobj = offload.profile(pipe, loading_callback=loading_callback(), profile_no=profile_no, quantizeTransformer=False, convertWeightsFloatTo=torch.float16, verboseLevel=verbose_level, **offload_kwargs)
        from shared.utils import offload_registry
        offload_registry.register_offloadobj("SeedVC", offloadobj, release_models)
        if persistent_models:
            _persistent_converter = converter
            _persistent_offloadobj = offloadobj
            _persistent_profile = profile_key
    else:
        converter = _persistent_converter
        offloadobj = _persistent_offloadobj
        keep_alive = True

    return converter, offloadobj, keep_alive


def convert_audio_file(source_audio_path: str, voice_sample_path: str, output_path: str, *, persistent_models: bool = False, profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, diffusion_steps: int | None = None, cfg_rate: float | None = None, mode: int = 1, amplitude_match_audio_path: str | None = None) -> str:
    import torch
    from postprocessing.seedvc_processor.assets import get_default_cfg_rate, get_default_steps, normalize_mode
    from shared.utils.audio_video import write_wav_file

    mode = normalize_mode(mode)
    converter, offloadobj, keep_alive = _get_runtime(persistent_models, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, mode=mode)
    try:
        source_audio, source_rate = _load_audio_tensor(source_audio_path)
        reference_audio, reference_rate = _load_audio_tensor(voice_sample_path)
        amplitude_audio = source_audio if amplitude_match_audio_path is None else _load_audio_tensor(amplitude_match_audio_path)[0]
        reference_audio = _match_reference_amplitude(amplitude_audio, reference_audio)
        modules = (module for model in converter.pipe_modules().values() for module in model.modules() if not module._modules)
        with torch.inference_mode(), checkpoint_modules(modules):
            converted = converter.convert_tensor(
                source_audio,
                source_rate,
                reference_audio,
                reference_rate,
                output_rate=source_rate,
                diffusion_steps=get_default_steps(mode) if diffusion_steps is None else diffusion_steps,
                cfg_rate=get_default_cfg_rate(mode) if cfg_rate is None else cfg_rate,
            )
        write_wav_file(output_path, converted, source_rate)
    finally:
        if offloadobj is not None:
            offloadobj.unload_all()
        if not keep_alive:
            _release_runtime_objects(converter, offloadobj)
    return output_path


def _make_temp_wav(output_dir: str, prefix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=output_dir)
    os.close(fd)
    return path


def _load_audio_tensor(audio_path: str):
    import librosa
    import numpy as np
    import torch

    audio_data, sample_rate = librosa.load(os.fspath(audio_path), sr=None, mono=False)
    audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.ndim == 1:
        audio_data = audio_data[None, :]
    return torch.from_numpy(audio_data), int(sample_rate)


def _audio_to_frames(audio, *, channels_first: bool):
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio[:, None]
    return audio.T if channels_first else audio


def _reference_active_mask(reference_audio, base_mask=None):
    import numpy as np

    frame_abs = np.mean(np.abs(reference_audio), axis=1)
    active_mask = np.ones(reference_audio.shape[0], dtype=bool) if base_mask is None else np.asarray(base_mask, dtype=np.float32).reshape(-1)[:reference_audio.shape[0]] > 0.5
    active_abs = frame_abs[active_mask]
    if active_abs.size == 0:
        return active_mask
    threshold = 0.1 * float(active_abs.mean())
    return active_mask & (frame_abs > threshold)


def _active_rms_amplitude(audio, active_mask=None, *, channels_first: bool = False) -> float:
    import numpy as np

    audio = _audio_to_frames(audio, channels_first=channels_first)
    if active_mask is None:
        active_mask = _reference_active_mask(audio)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)[:audio.shape[0]]
    active_audio = audio[:active_mask.shape[0]][active_mask]
    return float(np.sqrt(np.mean(np.square(active_audio)))) if active_audio.size else 0.0


def _match_reference_amplitude(source_audio, reference_audio):
    source_active = _active_rms_amplitude(source_audio, channels_first=True)
    reference_active = _active_rms_amplitude(reference_audio, channels_first=True)
    if source_active <= 1e-8 or reference_active <= 1e-8:
        return reference_audio
    gain = source_active / reference_active
    peak = float(reference_audio.detach().abs().max().item())
    if peak > 0.0:
        gain = min(gain, 1.0 / peak)
    return reference_audio * float(gain)


def _match_audio_file_amplitude(reference_path, audio_path, active_mask=None, mask_sample_rate=None) -> str:
    import numpy as np
    import soundfile as sf
    from shared.utils.audio_video import write_wav_file

    reference_audio, reference_rate = sf.read(os.fspath(reference_path), dtype="float32", always_2d=True)
    audio, audio_rate = sf.read(os.fspath(audio_path), dtype="float32", always_2d=True)
    reference_mask = _fit_audio_mask_to_audio(active_mask, mask_sample_rate, reference_rate, reference_audio.shape[0]) if active_mask is not None else None
    reference_activity = _reference_active_mask(reference_audio, reference_mask)
    audio_activity = _fit_audio_mask_to_audio(reference_activity.astype(np.float32), reference_rate, audio_rate, audio.shape[0]) > 0.5
    reference_active = _active_rms_amplitude(reference_audio, reference_activity)
    audio_active = _active_rms_amplitude(audio, audio_activity)
    if reference_active <= 1e-8 or audio_active <= 1e-8:
        return audio_path
    gain = reference_active / audio_active
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        gain = min(gain, 1.0 / peak)
    return write_wav_file(audio_path, audio * float(gain), audio_rate)


def _fit_audio_mask_to_audio(mask, mask_sample_rate, target_sample_rate, target_length):
    import numpy as np
    from shared.utils.audio_video import resample_audio_array

    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if int(mask_sample_rate) != int(target_sample_rate):
        mask = resample_audio_array(mask, int(mask_sample_rate), int(target_sample_rate))
    mask = np.clip(mask, 0.0, 1.0)
    if mask.shape[0] < target_length:
        mask = np.pad(mask, (0, target_length - mask.shape[0]))
    return mask[:target_length]


def _merge_audio_files_to_wav(audio_paths, output_path, masks=None, mask_sample_rate=None):
    import numpy as np
    import soundfile as sf
    from shared.utils.audio_video import resample_audio_array, write_wav_file

    mixed_audio = None
    target_rate = 0
    target_channels = 1
    for track_no, audio_path in enumerate(audio_paths):
        audio_data, sample_rate = sf.read(os.fspath(audio_path), dtype="float32", always_2d=True)
        if mixed_audio is None:
            target_rate = int(sample_rate)
            target_channels = audio_data.shape[1]
            mixed_audio = np.zeros((audio_data.shape[0], target_channels), dtype=np.float32)
        elif int(sample_rate) != target_rate:
            audio_data = resample_audio_array(audio_data, int(sample_rate), target_rate)
            if audio_data.ndim == 1:
                audio_data = audio_data[:, None]
        if audio_data.shape[1] != target_channels:
            audio_data = np.repeat(audio_data[:, :1], target_channels, axis=1) if audio_data.shape[1] == 1 else audio_data[:, :target_channels]
        if masks is not None:
            audio_data = audio_data * _fit_audio_mask_to_audio(masks[track_no], mask_sample_rate, target_rate, audio_data.shape[0])[:, None]
        if audio_data.shape[0] > mixed_audio.shape[0]:
            mixed_audio = np.pad(mixed_audio, ((0, audio_data.shape[0] - mixed_audio.shape[0]), (0, 0)))
        mixed_audio[:audio_data.shape[0]] += audio_data
    return write_wav_file(output_path, np.clip(mixed_audio, -1.0, 1.0), target_rate)


class SeedVCBridge:
    MODE_V1 = 1
    MODE_SINGING = 2
    MODE_ACCENT = 3
    PERSIST_UNLOAD = 1
    PERSIST_RAM = 2
    CURRENT_VERSION_LABEL = "SeedVC"

    _VERSIONS = {
        MODE_SINGING: "v1.0 Singing",
        MODE_V1: "v1.0 Speech",
        MODE_ACCENT: "v2 Speech",
    }

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def mode_choices(cls) -> list[tuple[str, int]]:
        return [(label, mode) for mode, label in cls._VERSIONS.items()]

    def normalize_config(self, config: dict[str, Any] | None = None) -> tuple[int, int]:
        config = self.server_config if config is None else config
        mode = config.get("seedvc_mode", self.MODE_SINGING)
        persistence = config.get("seedvc_persistence", self.PERSIST_UNLOAD)
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            mode = self.MODE_SINGING
        try:
            persistence = int(persistence)
        except (TypeError, ValueError):
            persistence = self.PERSIST_UNLOAD
        if mode not in self._VERSIONS:
            mode = self.MODE_SINGING
        if persistence not in (self.PERSIST_UNLOAD, self.PERSIST_RAM):
            persistence = self.PERSIST_UNLOAD
        config["seedvc_mode"] = mode
        config["seedvc_persistence"] = persistence
        return mode, persistence

    def settings(self, config: dict[str, Any] | None = None) -> tuple[bool, str | None, int]:
        mode, persistence = self.normalize_config(config)
        return True, self._VERSIONS[mode], persistence

    def enabled(self) -> bool:
        return self.settings()[0]

    def query_download_def(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        mode, _ = self.normalize_config()
        from postprocessing.seedvc_processor.assets import query_download_def
        return query_download_def(mode=mode)

    def _assets_available(self) -> bool:
        from postprocessing.seedvc_processor.assets import query_required_files

        mode, _ = self.normalize_config()
        required_files = query_required_files(mode)
        return all(self.files_locator.locate_file(path, error_if_none=False) is not None for path in required_files)

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None) -> bool:
        download_defs = self.query_download_def()
        if not download_defs:
            return False
        downloaded = False
        from shared.utils.download import download_def_missing_files, query_audio_background_replacement_download_def, send_download_status

        from postprocessing.seedvc_processor import component
        if not component.installed():
            send_download_status(send_cmd, "Downloading SeedVC Component...")
            component.ensure_installed()
            downloaded = True
        if download_defs and not self._assets_available():
            send_download_status(send_cmd, status_text)
            for download_def in download_defs:
                process_files(**download_def)
            downloaded = True
        if SEEDVC_RESTORE_BACKGROUND_STEM:
            stem_download_def = query_audio_background_replacement_download_def()
            if download_def_missing_files(stem_download_def):
                send_download_status(send_cmd, "Downloading audio background replacement model files...")
                process_files(**stem_download_def)
                downloaded = True
        return downloaded

    def _replace_two_speaker_audio_file(self, source_audio_path: str, voice_sample_path: str, output_path: str, *, voice_sample2_path: str, process_files: Callable[..., Any], profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, prefix: str = "seedvc") -> str:
        import numpy as np
        from preprocessing.speaker_separator import extract_dual_audio
        from shared.utils.audio_video import cleanup_temp_audio_files

        output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        split_track1 = _make_temp_wav(output_dir, f"{prefix}_speaker1_")
        split_track2 = _make_temp_wav(output_dir, f"{prefix}_speaker2_")
        converted_track1 = _make_temp_wav(output_dir, f"{prefix}_speaker1_seedvc_")
        converted_track2 = _make_temp_wav(output_dir, f"{prefix}_speaker2_seedvc_")
        temp_tracks = [split_track1, split_track2, converted_track1, converted_track2]
        try:
            _, speaker_masks, mask_sample_rate = extract_dual_audio(source_audio_path, split_track1, split_track2, verbose=verbose_level >= 2, return_masks=True, speech_masks_only=True)
            mask_values = list(speaker_masks.values())
            mode, persistence = self.normalize_config()
            convert_audio_file(split_track1, voice_sample_path, converted_track1, persistent_models=persistence == self.PERSIST_RAM, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, mode=mode)
            convert_audio_file(split_track2, voice_sample2_path, converted_track2, persistent_models=persistence == self.PERSIST_RAM, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, mode=mode)
            _match_audio_file_amplitude(split_track1, converted_track1, active_mask=mask_values[0], mask_sample_rate=mask_sample_rate)
            _match_audio_file_amplitude(split_track2, converted_track2, active_mask=mask_values[1], mask_sample_rate=mask_sample_rate)
            merge_tracks, merge_masks = [converted_track1, converted_track2], mask_values
            if KEEP_ORIGINAL_AUDIO_OUTSIDE_TWO_SPEAKERS:
                merge_tracks.append(source_audio_path)
                merge_masks.append(np.clip(1.0 - np.maximum(mask_values[0], mask_values[1]), 0.0, 1.0))
            return _merge_audio_files_to_wav(merge_tracks, output_path, masks=merge_masks, mask_sample_rate=mask_sample_rate)
        finally:
            cleanup_temp_audio_files(temp_tracks)

    def replace_audio_file(self, source_audio_path: str, voice_sample_path: str, output_path: str, *, process_files: Callable[..., Any], profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, voice_sample2_path: str | None = None, speaker_count: int = 1, prefix: str = "seedvc") -> str:
        mode, persistence = self.normalize_config()
        self.download(process_files)
        output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        temp_tracks = []
        conversion_source_path = source_audio_path
        background_path = None
        conversion_output_path = output_path
        try:
            if SEEDVC_RESTORE_BACKGROUND_STEM:
                from preprocessing.extract_vocals import extract_vocal_and_background_stems

                requested_vocals_path = _make_temp_wav(output_dir, "seedvc_vocals_")
                requested_background_path = _make_temp_wav(output_dir, "seedvc_background_")
                temp_tracks += [requested_vocals_path, requested_background_path]
                conversion_source_path, background_path = extract_vocal_and_background_stems(source_audio_path, requested_vocals_path, requested_background_path)
                temp_tracks += [conversion_source_path, background_path]
                conversion_output_path = _make_temp_wav(output_dir, "seedvc_voice_")
                temp_tracks.append(conversion_output_path)

            if int(speaker_count) == 2:
                if voice_sample2_path is None:
                    raise RuntimeError("Two-speaker SeedVC voice replacement requires a second voice sample.")
                converted_path = self._replace_two_speaker_audio_file(conversion_source_path, voice_sample_path, conversion_output_path, voice_sample2_path=voice_sample2_path, process_files=process_files, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, prefix=prefix)
            else:
                converted_path = convert_audio_file(conversion_source_path, voice_sample_path, conversion_output_path, persistent_models=persistence == self.PERSIST_RAM, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, mode=mode)
                _match_audio_file_amplitude(conversion_source_path, converted_path)
            if background_path is not None:
                return _merge_audio_files_to_wav([converted_path, background_path], output_path)
            return converted_path
        finally:
            if temp_tracks:
                from shared.utils.audio_video import cleanup_temp_audio_files

                cleanup_temp_audio_files(temp_tracks)

    def replace_audio_tracks(self, audio_tracks: list[str], voice_sample_path: str | None, output_dir: str, prefix: str, *, process_files: Callable[..., Any], profile_no=4, verbose_level: int = 1, init_pipe: Callable[..., int] | None = None, voice_sample2_path: str | None = None, speaker_count: int = 1) -> tuple[list[str], list[str]]:
        if voice_sample_path is None or len(audio_tracks) == 0:
            return audio_tracks, []
        converted_tracks = []
        for track_no, audio_track in enumerate(audio_tracks):
            output_path = _make_temp_wav(output_dir, f"{prefix}_seedvc_track{track_no}_")
            converted_tracks.append(self.replace_audio_file(audio_track, voice_sample_path, output_path, process_files=process_files, profile_no=profile_no, verbose_level=verbose_level, init_pipe=init_pipe, voice_sample2_path=voice_sample2_path, speaker_count=speaker_count, prefix=f"{prefix}_track{track_no}"))
        return converted_tracks, converted_tracks

    def release_vram(self) -> None:
        release_models()
