# Adapted from Zyphra/ZONOS3-Mini (model/src/zonos3/tts/sampler.py, reference/synth.py), commit 9fd14e51.
# Copyright (c) 2026 Zyphra, Inc. MIT License, see LICENSE_ZONOS3.
"""Sampling of one frame of ZONOS3 codes: every codebook from its own logits, with the released defaults (temperature 0.75, top-k 80,
top-p 0.69, repetition penalty 2.4 over the last 96 generated frames of the first 6 codebooks). Audio padding is never sampled; of
the special ids only the end-of-audio id is allowed. Draws use a generator seeded per take, as the reference does."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.75
    top_k: int = 80
    top_p: float = 0.69
    min_p: float = 0.0
    repetition_penalty: float = 2.4
    repetition_window: int = 96
    repetition_codebooks: int = 6

    def clamped(self, codebook_size, n_codebooks):
        return Sampling(max(float(self.temperature), 1e-6), int(self.top_k) if int(self.top_k) >= 1 else codebook_size, min(max(float(self.top_p), 0.0), 1.0),
                        max(float(self.min_p), 0.0), max(float(self.repetition_penalty), 1.0), max(int(self.repetition_window), 0),
                        n_codebooks if int(self.repetition_codebooks) < 0 else min(max(int(self.repetition_codebooks), 0), n_codebooks))


class FrameSampler:
    def __init__(self, sampling, *, n_codebooks, codebook_size, audio_pad_id, eoa_id, device, seed):
        self.s = sampling.clamped(codebook_size, n_codebooks)
        self.n_codebooks, self.codebook_size, self.pad = n_codebooks, codebook_size, audio_pad_id
        vocab = audio_pad_id + 1
        ids = torch.arange(vocab, device=device)
        self.invalid = ((ids == audio_pad_id) | ((ids >= codebook_size) & (ids != eoa_id)))[None]
        self.penalized = (torch.arange(n_codebooks, device=device) < self.s.repetition_codebooks)[:, None]
        self.generator = torch.Generator(device=device).manual_seed(int(seed))

    def __call__(self, logits, recent):
        """logits [n_codebooks, vocab]; recent [rows, n_codebooks]: the last generated frames the repetition penalty looks at."""
        s = self.s
        logits = logits.float().masked_fill(self.invalid, float("-inf"))
        if len(recent) and s.repetition_window > 0 and s.repetition_penalty > 1.0 and s.repetition_codebooks > 0:
            recent = recent[-s.repetition_window:].T.long()
            valid = (recent >= 0) & (recent < self.codebook_size) & self.penalized
            counts = torch.zeros(logits.shape, dtype=torch.int32, device=logits.device).scatter_add_(-1, recent.clamp(0, logits.shape[-1] - 1), valid.int())
            logits = torch.where(counts > 0, torch.where(logits > 0, logits / s.repetition_penalty, logits * s.repetition_penalty), logits)
        logits = logits / s.temperature
        if 0 < s.top_k < logits.shape[-1]:
            logits = logits.masked_fill(logits < torch.topk(logits, s.top_k, dim=-1).values[..., -1:], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        if 0.0 < s.top_p < 1.0:
            ordered, order = torch.sort(probs, dim=-1, descending=True)
            ordered = ordered.masked_fill(ordered.cumsum(-1) - ordered > s.top_p, 0.0)
            probs = probs.scatter(-1, order, ordered)
            probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-8)
        if s.min_p > 0.0:
            probs = probs.masked_fill(probs < s.min_p * probs.max(-1, keepdim=True).values, 0.0)
            probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-8)
        # a row emptied by the filters falls back to its best logit
        greedy = F.one_hot(logits.argmax(-1), probs.shape[-1]).to(probs.dtype)
        probs = torch.where((probs.sum(-1) <= 0)[:, None], greedy, probs)
        return torch.multinomial(probs, 1, generator=self.generator)[:, 0]
