# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Vocal / instrumental separation with BS-RoFormer (native PyTorch, MMGP-managed weights)."""
import threading
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from mmgp import offload, runtime
from shared.utils import files_locator as fl
from shared.utils import offload_registry
from shared.utils.download import process_files_def_if_needed
from shared.utils.media_control import controlled_model_loading, inference_checkpoint
from preprocessing.roformer.assets import FILENAME, query_download_def
from preprocessing.roformer import model as roformer

_lock = threading.RLock()
_loaded = None
_owner = "vocal_separation"


def release():
    global _loaded
    with _lock:
        entry, _loaded = _loaded, None
        if entry is not None:
            entry["manager"].release()
            offload_registry.unregister_offloadobj(_owner, entry["manager"])


@controlled_model_loading
def _load(loading_callback=None):
    global _loaded
    if _loaded is not None:
        return _loaded
    from accelerate import init_empty_weights
    tqdm.write("Loading - Vocal Separation Model (BS-RoFormer)")
    with init_empty_weights(include_buffers=True):
        model = roformer.BSRoformer()
    offload.load_model_data(model, fl.locate_file(f"roformer/{FILENAME}"), writable_tensors=False, default_dtype=torch.bfloat16)
    model._convertWeightsFloatTo = None
    model.eval().requires_grad_(False)
    previous_device = torch.get_default_device()
    try:
        manager = offload.profile({"vocal_separator": model}, 0, budgets={"vocal_separator": 0}, quantizeTransformer=False, loading_callback=loading_callback)
    finally:
        torch.set_default_device(None if previous_device.type == "cpu" else previous_device)
    _loaded = {"model": model, "manager": manager}
    offload_registry.register_offloadobj(_owner, manager, release)
    return _loaded


def _separate(src_path: str, min_seconds: float):
    """(vocals, instrumental, mix length): (samples, 2) FP32 stems at 44.1 kHz, as audio-separator's Roformer path produced them; inputs
    shorter than min_seconds are padded with silence (the separator's window is 8 s)."""
    from shared.attention import attention_shared_state
    process_files_def_if_needed(query_download_def())
    mix, _ = librosa.load(src_path, mono=False, sr=roformer.SAMPLE_RATE)
    if not np.any(mix):
        raise ValueError(f"Audio file {src_path} is empty or not valid")
    mix = np.asfortranarray([mix, mix]) if mix.ndim == 1 else mix
    length = mix.shape[1]
    padded = max(round(min_seconds * roformer.SAMPLE_RATE), roformer.CHUNK)
    if length < padded:
        mix = np.pad(mix, ((0, 0), (0, padded - length)))
    mix = roformer.normalize(mix)
    with _lock, runtime.execution_scope(inference_checkpoint), attention_shared_state(), torch.inference_mode():
        entry = _load()
        model, manager = entry["model"], entry["manager"]
        try:
            progress = lambda starts: tqdm(starts, desc="Separating Vocals")
            vocals = roformer.demix(model, torch.from_numpy(np.ascontiguousarray(mix)), runtime.execution_device(model), inference_checkpoint, progress)
        finally:
            manager.unload_all()
    instrumental = mix - vocals
    return roformer.normalize(vocals).T, roformer.normalize(instrumental).T, length


def _write(path: Path, stem: np.ndarray, frames=None):
    """audio-separator's output: peak-normalized, 16-bit PCM, 44.1 kHz stereo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = roformer.normalize(stem[:frames])
    sf.write(str(path), (stem * 32767).astype(np.int16), roformer.SAMPLE_RATE, subtype="PCM_16")
    return str(path)


def _output_path(dst_path: str) -> Path:
    dst = Path(dst_path)
    return dst if dst.suffix else dst.with_suffix(".wav")


def extract_vocal_and_background_stems(src_path: str, vocals_dst_path: str, background_dst_path: str, min_seconds: float = 8) -> tuple[str, str]:
    vocals, instrumental, _ = _separate(src_path, min_seconds)
    return _write(_output_path(vocals_dst_path), vocals), _write(_output_path(background_dst_path), instrumental)


def get_vocals(src_path: str, dst_path: str, min_seconds: float = 8) -> str:
    """Save the vocal stem, padding short inputs for the separator if needed."""
    vocals, _, length = _separate(src_path, min_seconds)
    return _write(_output_path(dst_path), vocals, length)


def get_instrumental(src_path: str, dst_path: str, min_seconds: float = 8) -> str:
    """Save the instrumental stem, removing the separated vocals."""
    _, instrumental, length = _separate(src_path, min_seconds)
    return _write(_output_path(dst_path), instrumental, length)
