# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""HuBERT Large (facebook/hubert-large-ll60k, Apache-2.0) hidden layers as audio conditioning of a video model, and the 16 kHz mono audio it
reads. Only the layers that produce the requested hidden states are kept: the file holds them with the weight-normalized positional
convolution baked in, config.json and preprocessor_config.json stay those of the released model."""
import os
import subprocess

import numpy as np
import torch

from mmgp import offload, runtime
from shared.utils import files_locator as fl
from shared.utils.audio_video import _ffmpeg_binary

HUBERT_FOLDER = "hubert-large-ll60k"
SAMPLE_RATE = 16000
FEATURES_PER_SECOND = 50  # one hidden state every 320 samples


def audio_16k_mono(source_path=None, waveform=None, sample_rate=0) -> np.ndarray:
    """A file, or a waveform (samples[, channels]) at sample_rate, as float32 mono at 16 kHz, through FFmpeg's resampler in both cases."""
    command = [_ffmpeg_binary(), "-nostdin", "-loglevel", "error"]
    payload = None
    if source_path:
        command += ["-i", os.fspath(source_path), "-map", "0:a:0"]
    else:
        audio = waveform.detach().cpu().numpy() if torch.is_tensor(waveform) else np.asarray(waveform)
        audio = audio[:, None] if audio.ndim == 1 else audio
        command += ["-f", "f32le", "-ar", str(int(sample_rate)), "-ac", str(audio.shape[1]), "-i", "pipe:0"]
        payload = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    command += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    decoded = subprocess.run(command, input=payload, check=True, capture_output=True).stdout
    return np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32768.0


class HubertEncoder:
    def __init__(self, weights_path, layers=(9, 10, 11, 12), dtype=torch.bfloat16):
        from accelerate import init_empty_weights
        from torch.nn.utils import parametrize
        from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

        folder = os.path.dirname(fl.locate_file(os.path.join(HUBERT_FOLDER, "config.json")))
        config = HubertConfig.from_pretrained(folder, local_files_only=True)
        config.num_hidden_layers = max(layers) + 1  # the hidden state n is the input of layer n: the last layer's own output is not read
        with init_empty_weights():
            self.model = HubertModel(config)
        parametrize.remove_parametrizations(self.model.encoder.pos_conv_embed.conv, "weight", leave_parametrized=True)
        offload.load_model_data(self.model, weights_path, writable_tensors=False, default_dtype=dtype)
        self.model.eval().requires_grad_(False)
        self.layers = tuple(layers)
        self.dtype = dtype
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(folder, local_files_only=True)

    @torch.inference_mode()
    def __call__(self, pcm: np.ndarray) -> torch.Tensor:
        """Hidden states (features, layers, width) of float32 mono audio at 16 kHz, on the GPU in float32."""
        values = self.extractor(pcm, sampling_rate=SAMPLE_RATE, return_tensors="np").input_values
        values = torch.from_numpy(values).to(device=runtime.execution_device(self.model), dtype=self.dtype)
        hidden_states = self.model(values, output_hidden_states=True).hidden_states
        return torch.stack([hidden_states[index][0].float() for index in self.layers], dim=1)


__all__ = ["FEATURES_PER_SECOND", "HUBERT_FOLDER", "HubertEncoder", "SAMPLE_RATE", "audio_16k_mono"]
