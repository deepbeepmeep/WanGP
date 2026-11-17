# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Windowed Avatar 1.6 Whisper conditioning on the upstream 30-second grid."""
import os
import librosa
import numpy as np
import torch
from shared.utils.phase_progress import check_abort


def audio_windows(model, path, frame_count, fps, start_frame):
    stat = os.stat(path)
    duration = librosa.get_duration(path=path)
    total_frames = max(int(duration * fps), start_frame + frame_count, 1)
    # Upstream truncates to 2 * video_length encoder rows, then interpolates with align_corners=True.
    encoder_frames = total_frames * 2
    window = model.transformer.audio_window
    indices = torch.arange(start_frame, start_frame + frame_count, device='cpu')[:, None]
    indices = (indices + torch.arange(window, device='cpu')[None] - window // 2).clamp(0, total_frames - 1)
    positions = indices.to(torch.float64) * ((encoder_frames - 1) / max(total_frames - 1, 1))
    left = positions.floor().long()
    right = (left + 1).clamp(max=encoder_frames - 1)
    chunks = sorted(set((left // 1500).flatten().tolist() + (right // 1500).flatten().tolist()))
    identity = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    cache = model._audio_chunks
    needed = {(identity, index) for index in chunks}
    # Keep one current interval per driving voice, never an entire long recording.
    for key in list(cache):
        if key[0] == identity and key not in needed:
            del cache[key]
    for index in chunks:
        check_abort()
        key = (identity, index)
        if key in cache:
            continue
        speech, _ = librosa.load(path, sr=16000, offset=index * 30, duration=30)
        if not len(speech):
            speech = np.zeros(1, dtype=np.float32)
        features = model.audio_feature_extractor(speech, sampling_rate=16000, return_tensors='pt')
        inputs = features.input_features.to(model.device, dtype=model.dtype)
        from shared.utils.phase_progress import text_encoding_progress
        with text_encoding_progress(model.audio_encoder.encoder.layers, title="Encoding Driving Audio", prompt_count=0):
            states = model.audio_encoder.encoder(inputs, output_hidden_states=True).hidden_states
        grouped = [torch.stack(states[a:b], dim=0).mean(dim=0).squeeze(0)
                   for a, b in ((0, 8), (8, 16), (16, 24), (24, 32), (32, 33))]
        value = torch.stack(grouped, dim=1).to('cpu')
        check_abort()
        cache[key] = value
        del states, grouped, features, inputs
    def gather(rows):
        output = torch.empty((*rows.shape, 5, 1280), dtype=model.dtype, device='cpu')
        for index in chunks:
            selected = rows // 1500 == index
            output[selected] = cache[(identity, index)][rows[selected] % 1500]
        return output
    low, high = gather(left), gather(right)
    fraction = (positions - left).float()[..., None, None]
    # Interpolate in float, as PyTorch's low-precision linear interpolation kernel does.
    result = (low.float() * (1 - fraction) + high.float() * fraction).to(model.dtype)
    while len(cache) > 4:
        del cache[next(iter(cache))]
    return result.unsqueeze(0)
