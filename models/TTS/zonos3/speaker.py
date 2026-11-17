# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Speaker embedding of a voice reference for ZONOS3-Mini: Zyphra's own MECT-B2 weights (ownmect_s2) in AntSpeaker's VPR model, on a
16 kHz mono waveform (at most 60 s) as an 80-bin Kaldi filterbank, as released."""
import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio.functional as audio_functional

from mmgp import runtime
from .mect import VPR


class SpeakerEncoder(VPR):
    def __init__(self, c):
        super().__init__({"backbone": {"config": c["backbone_config"]}, "embed_dim": c["embed_dim"]})
        self.sample_rate, self.max_seconds = c["sample_rate"], c["max_seconds"]
        fbank = dict(c["fbank"])
        self.scale = float(fbank.pop("scale"))
        self.fbank = {**fbank, "sample_frequency": float(self.sample_rate)}

    def embed(self, waveform, sample_rate):
        """waveform: mono float32 [samples] -> float32 [embed_dim] on the CPU."""
        runtime.ensure_loaded(self)
        device = runtime.execution_device(self)
        waveform = waveform.to(device=device, dtype=torch.float32)[None]
        with torch.device(device):  # torchaudio's resampling kernel and filterbank tables are built on the default device
            if sample_rate != self.sample_rate:
                waveform = audio_functional.resample(waveform, sample_rate, self.sample_rate)
            waveform = waveform[:, :int(self.max_seconds * self.sample_rate)]
            features = kaldi.fbank(waveform * self.scale, **self.fbank)[None].transpose(1, 2)
        return self(features)[0].float().cpu()
