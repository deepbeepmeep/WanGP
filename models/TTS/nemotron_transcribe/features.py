# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. Ported from Hugging Face Transformers (commit a37e25d9,
# models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py, models/nemotron3_5_transcribe/processing_nemotron3_5_transcribe.py).
# The mel filter bank follows librosa.filters.mel (ISC License, Copyright (c) 2013--2023, librosa development team).
# Modified for WanGP.
"""Audio decoding and log-mel features of Nemotron 3.5 Transcribe, as `NemotronAsrStreamingFeatureExtractor` and the processor's padding."""
import subprocess

import numpy as np
import torch

from shared.utils.video_decode import resolve_media_binary

LOG_ZERO_GUARD_VALUE = 2**-24
STFT_CHUNK_FRAMES = 6000  # spectrogram frames per STFT pass (60 s): bounds the transient memory of long recordings


def load_audio(path, sampling_rate=16000, start=None, duration=None) -> torch.Tensor:
    """Decodes the first audio stream of any audio or video file to mono float32 samples at `sampling_rate` with WanGP's ffmpeg."""
    command = [resolve_media_binary("ffmpeg"), "-v", "error", "-nostdin", *(["-ss", str(start)] if start else []), "-i", str(path), *(["-t", str(duration)] if duration else []),
               "-vn", "-ac", "1", "-ar", str(sampling_rate), "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise ValueError(f"Cannot decode the audio of {path}: " + result.stderr.decode(errors="replace")[-1200:])
    return torch.from_numpy(np.frombuffer(result.stdout, dtype="<f4").copy())


def _hz_to_mel(frequencies):
    # Slaney mel scale: linear below 1 kHz, logarithmic above
    mels = frequencies / (200.0 / 3)
    log_t = frequencies >= 1000.0
    mels[log_t] = 1000.0 / (200.0 / 3) + np.log(frequencies[log_t] / 1000.0) / (np.log(6.4) / 27.0)
    return mels


def _mel_to_hz(mels):
    freqs = (200.0 / 3) * mels
    log_t = mels >= 1000.0 / (200.0 / 3)
    freqs[log_t] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mels[log_t] - 1000.0 / (200.0 / 3)))
    return freqs


def mel_filter_bank(sampling_rate, n_fft, n_mels) -> np.ndarray:
    """`librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr / 2, norm="slaney")`, float32, with librosa's operation order and roundings."""
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sampling_rate / 2.0]))[0], n_mels + 2))
    fdiff, ramps = np.diff(mel_f), np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights


class NemotronFeatureExtractor:
    """Log-mel spectrogram of `NemotronAsrStreamingFeatureExtractor` (centered STFT, no dither, no normalization), padded on the right to a
    multiple of the encoders' frame stacking (`subsampling_factor` of processor_config.json) as the processor does. `config` is the
    `feature_extractor` entry of processor_config.json."""

    def __init__(self, config: dict, subsampling_factor=8):
        self.sampling_rate, self.hop_length, self.n_fft, self.win_length = config["sampling_rate"], config["hop_length"], config["n_fft"], config["win_length"]
        self.preemphasis = config["preemphasis"]
        self.subsampling_factor = subsampling_factor
        # the checkpoint computes its features with a filter bank and a window rounded to `mel_filters_dtype`; the spectrogram is float32
        dtype = getattr(torch, config["mel_filters_dtype"])
        self.mel_filters = torch.from_numpy(mel_filter_bank(self.sampling_rate, self.n_fft, config["feature_size"])).to(dtype).float()
        self.window = torch.hann_window(self.win_length, periodic=False, dtype=torch.float32, device="cpu").to(dtype).float()

    def __call__(self, waveform: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
        """`waveform` [samples] mono at `sampling_rate` -> (features [1, frames, mels] float32, mask [1, frames] long), `frames` a multiple of
        `subsampling_factor`, the frames past `num_samples // hop_length` zero and masked."""
        num_samples = waveform.shape[0]
        waveform = waveform.to(device=device, dtype=torch.float32)
        waveform = torch.cat([waveform[:1], waveform[1:] - self.preemphasis * waveform[:-1]])
        # centered STFT: n_fft // 2 zeros on each side, 1 + num_samples // hop frames, computed by chunks of frames
        waveform = torch.nn.functional.pad(waveform, (self.n_fft // 2, self.n_fft // 2))
        num_frames = 1 + (waveform.shape[0] - self.n_fft) // self.hop_length
        window, mel_filters = self.window.to(device), self.mel_filters.to(device)
        features = []
        for start in range(0, num_frames, STFT_CHUNK_FRAMES):
            end = min(start + STFT_CHUNK_FRAMES, num_frames)
            stft = torch.stft(waveform[None, start * self.hop_length : (end - 1) * self.hop_length + self.n_fft], self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=window, return_complex=True, center=False)
            magnitudes = torch.sqrt(torch.view_as_real(stft).pow(2).sum(-1)).pow(2)
            del stft
            features.append(torch.log(mel_filters @ magnitudes + LOG_ZERO_GUARD_VALUE).transpose(1, 2))
            del magnitudes
        features = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        num_valid = num_samples // self.hop_length
        padding = -num_frames % self.subsampling_factor
        mask = torch.arange(num_frames + padding, device=device)[None] < num_valid
        features = torch.nn.functional.pad(features, (0, 0, 0, padding)) * mask[..., None]
        return features, mask.long()
