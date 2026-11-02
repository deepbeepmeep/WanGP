# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. Ported from Hugging Face Transformers (commit a37e25d9, models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py
# and models/nemotron3_diarization/modeling_nemotron3_diarization.py), with NVIDIA NeMo's module names.
# Modified for WanGP.
"""Audio side of Nemotron 3.5 Transcribe: the speech recognition encoder and the Sortformer diarization model run on the same spectrogram, the
speaker activities added to the encoder states, the projection to the language model, and the word timestamp (CTC) encoder and classifier.

Module names are NVIDIA NeMo's, so that `NemotronPerception().load_state_dict(tensors, strict=True)` takes the checkpoint tensors under
`perception.` and `ctc_timestamp.` with their original keys, minus `NON_MODULE_KEYS`: constants the module recomputes (the front-end filter
banks and windows, the timestamp head's residual scale and upsampling factor), the unused Sortformer `hidden_to_spks` head, and the alignment
tokenizer (`ctc_timestamp.tokenizer.model_proto`, which `load_perception` turns into `ctc_tokenizer`).

Weights may be stored in a lower precision than `compute_dtype`: they are then cast at use, so FP32 compute (the reference's precision for the
audio tower and the projection) works from BF16 weights."""
import math

import sentencepiece
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from mmgp import offload, runtime
from shared.attention import pay_attention

NON_MODULE_KEYS = (
    "perception.preprocessor.featurizer.fb", "perception.preprocessor.featurizer.window",
    "perception.encoder.diarization_model.preprocessor.featurizer.fb", "perception.encoder.diarization_model.preprocessor.featurizer.window",
    "perception.encoder.diarization_model.sortformer_modules.hidden_to_spks.weight", "perception.encoder.diarization_model.sortformer_modules.hidden_to_spks.bias",
    "ctc_timestamp.decoder.residual_scale", "ctc_timestamp.decoder.upsample_factor", "ctc_timestamp.tokenizer.model_proto",
)
LOG_PROB_CHUNK_FRAMES = 2048  # head frames per classifier pass when only some classes are returned


def _cast(tensor, dtype):
    return None if tensor is None else tensor.to(dtype)


def _linear(module, x):
    return module(x) if module.weight.dtype == x.dtype else F.linear(x, module.weight.to(x.dtype), _cast(module.bias, x.dtype))


def _norm(module, x):
    return module(x) if module.weight.dtype == x.dtype else F.layer_norm(x, module.normalized_shape, module.weight.to(x.dtype), module.bias.to(x.dtype), module.eps)


def _conv(module, x):
    if module.weight.dtype == x.dtype:
        return module(x)
    conv = F.conv_transpose1d if isinstance(module, nn.ConvTranspose1d) else F.conv1d
    return conv(x, module.weight.to(x.dtype), _cast(module.bias, x.dtype), stride=module.stride, padding=module.padding)


def _rotary_inverse_frequencies(rotary_dim, base):
    return 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu") / rotary_dim))


def _rotary_table(inverse_frequencies, num_positions, device, dtype):
    # positions restart at 0 for every encoder run, as in the reference
    freqs = torch.arange(num_positions, device=device)[None, :, None].float() * inverse_frequencies.to(device)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[:, :, None], emb.sin().to(dtype)[:, :, None]


def _apply_rotary(x, cos, sin):
    # split-half rotation of the first `rotary_dim` channels of every head, the others passed through
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot[..., : rotary_dim // 2], x_rot[..., rotary_dim // 2 :]
    return torch.cat([x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin, x_pass], dim=-1)


def normalize_per_feature(features, mask):
    """Each mel bin to zero mean and unit variance over the valid frames (unbiased variance, epsilon added to the standard deviation)."""
    mask = mask.bool()[..., None]
    features = features.float().masked_fill(~mask, 0.0)
    num_frames = mask.sum(1)
    mean = features.sum(1, keepdim=True) / num_frames[:, None].clamp(min=1)
    centered = (features - mean).masked_fill(~mask, 0.0)
    std = (centered.pow(2).sum(1, keepdim=True) / (num_frames[:, None] - 1).clamp(min=1)).sqrt() + 1e-5
    return centered / std


def speaker_embeddings(num_speakers, hidden_size):
    """Fixed sinusoidal embedding of each speaker slot (float32)."""
    position = torch.arange(num_speakers, dtype=torch.float32, device="cpu")[:, None]
    inv_freq = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32, device="cpu") * -(math.log(10000.0) / hidden_size))
    embeddings = torch.zeros(num_speakers, hidden_size, dtype=torch.float32, device="cpu")
    embeddings[:, 0::2] = torch.sin(position * inv_freq)
    embeddings[:, 1::2] = torch.cos(position * inv_freq)
    return embeddings


class FeatureStacking(nn.Module):
    """NeMo `pre_encode`: `factor` consecutive spectrogram frames stacked and projected (the last incomplete group zero-padded)."""

    def __init__(self, feat_in, d_model, factor):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(feat_in * factor, d_model, bias=False)

    def forward(self, features):
        batch_size, num_frames, num_bins = features.shape
        padding = -num_frames % self.factor
        features = F.pad(features, (0, 0, 0, padding))
        return _linear(self.proj, features.reshape(batch_size, (num_frames + padding) // self.factor, num_bins * self.factor))


class SelfAttention(nn.Module):
    """NeMo `attn`: fused bias-free query/key/value projection (rows `[query | key | value]`), optional per-head query/key LayerNorm, RoPE on
    the first channels of each head, biased output projection."""

    def __init__(self, d_model, n_heads, qk_norm):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, rotary, num_keys):
        batch_size, num_tokens, _ = x.shape
        q, k, v = _linear(self.w_qkv, x).view(batch_size, num_tokens, 3, self.n_heads, self.head_dim).unbind(2)
        if hasattr(self, "q_norm"):
            q, k = _norm(self.q_norm, q), _norm(self.k_norm, k)
        q, k = _apply_rotary(q, *rotary), _apply_rotary(k, *rotary)
        # padding is on the right: the keys past `num_keys` are dropped, every query attends to the valid ones
        k_lens = None if num_keys is None else torch.tensor([num_keys], dtype=torch.int32, device="cpu")
        x = pay_attention([q, k, v], force_attention="sdpa" if q.dtype == torch.float32 else None, k_lens=k_lens)
        return _linear(self.out_proj, x.flatten(2))


class FeedForward(nn.Module):
    """NeMo `ffn`: `net` = Linear, GELU (erf), Dropout, Linear."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(0.0), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return _linear(self.net[3], F.gelu(_linear(self.net[0], x)))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, qk_norm):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, qk_norm)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, rotary, num_keys):
        x = x + self.attn(_norm(self.norm1, x), rotary, num_keys)
        return x + self.ffn(_norm(self.norm2, x))


class TransformerEncoder(nn.Module):
    """NeMo `TransformerEncoder` with pre-block norms and RoPE: `pre_encode` (feature stacking, applied by the caller), `embed_norm`, `layers`,
    `final_norm`. `forward` takes the embedded frames."""

    def __init__(self, d_model, n_heads, n_layers, d_ff, qk_norm, rotary_fraction, rope_base=10000.0, feat_in=None, subsampling_factor=None):
        super().__init__()
        if feat_in is not None:
            self.pre_encode = FeatureStacking(feat_in, d_model, subsampling_factor)
        self.embed_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(TransformerLayer(d_model, n_heads, d_ff, qk_norm) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)
        self.inverse_frequencies = _rotary_inverse_frequencies(int(d_model // n_heads * rotary_fraction), rope_base)

    def forward(self, x, num_keys=None, check_abort=None):
        """`x` [1, frames, d_model]; `num_keys` the valid frames when the last ones are padding."""
        rotary = _rotary_table(self.inverse_frequencies, x.shape[1], x.device, x.dtype)
        x = _norm(self.embed_norm, x)
        for layer in self.layers:
            x = layer(x, rotary, num_keys)
            if check_abort is not None:
                check_abort()
        return _norm(self.final_norm, x)


class SpeakerCache:
    """Offline streaming state of the Sortformer: the Arrival-Order Speaker Cache and the FIFO queue of the most recent encoder input frames
    that every chunk attends to. Frame scores are computed in float32 whatever the compute dtype."""

    def __init__(self, config, fifo_length, update_period, embeds):
        self.fifo_length, self.update_period = fifo_length, update_period
        self.cache_length = config["speaker_cache_length"]
        self.num_silence_frames = config["speaker_cache_silence_frames_per_speaker"]
        self.score_threshold = config["prediction_score_threshold"]
        self.latest_boost = config["latest_frames_score_boost"]
        self.num_speakers, self.subsampling_factor = config["num_speakers"], config["subsampling_factor"]
        budget = self.cache_length // self.num_speakers - self.num_silence_frames
        self.min_positive_scores = math.floor(budget * config["min_positive_scores_rate"])
        self.num_strong_boosted = math.floor(budget * config["strong_boost_rate"])
        self.num_weak_boosted = math.floor(budget * config["weak_boost_rate"])
        batch_size, _, hidden_size = embeds.shape
        self.embeds = embeds.new_zeros(batch_size, self.cache_length, hidden_size)
        self.probs = torch.zeros(batch_size, self.cache_length, self.num_speakers, dtype=torch.float32, device=embeds.device)
        self.fifo = embeds.new_zeros(batch_size, fifo_length, hidden_size)
        self.num_cache_frames = self.num_fifo_frames = 0
        self.is_compressed = False

    def get_embeds(self):
        return torch.cat([self.embeds[:, : self.num_cache_frames], self.fifo[:, : self.num_fifo_frames]], dim=1)

    def _pool_probs(self, logits, num_valid):
        probs = F.avg_pool1d(logits.float().sigmoid().transpose(1, 2), self.subsampling_factor, self.subsampling_factor).transpose(1, 2)
        if num_valid < probs.shape[1]:
            probs = probs * (torch.arange(probs.shape[1], device=probs.device) < num_valid)[None, :, None]
        return probs

    def update(self, input_embeds, logits, silence_embeds, num_chunk_frames, num_valid, num_lookback_frames):
        """Pushes the chunk frames of a step (after the cached and look-back frames of `input_embeds`) to the FIFO queue, moving its oldest frames
        to the speaker cache, compressed by frame scores, when it overflows."""
        num_cache_frames, num_fifo_frames = self.num_cache_frames, self.num_fifo_frames
        probs = self._pool_probs(logits, num_valid)
        chunk_start = num_cache_frames + num_fifo_frames + num_lookback_frames
        fifo_embeds = torch.cat([self.fifo[:, :num_fifo_frames], input_embeds[:, chunk_start : chunk_start + num_chunk_frames]], dim=1)
        num_popped = 0 if fifo_embeds.shape[1] <= self.fifo_length else min(max(self.update_period, fifo_embeds.shape[1] - self.fifo_length), fifo_embeds.shape[1])
        if num_popped:
            # the look-back frames, between the FIFO frames and the chunk frames, are skipped
            fifo_probs = torch.cat([probs[:, num_cache_frames : num_cache_frames + num_fifo_frames], probs[:, chunk_start : chunk_start + num_chunk_frames]], dim=1)
            # an uncompressed cache holds plain chunk frames whose probabilities this step re-estimates, a compressed one keeps its stored ones
            stored_probs = self.probs[:, :num_cache_frames] if self.is_compressed else probs[:, :num_cache_frames]
            cache_embeds = torch.cat([self.embeds[:, :num_cache_frames], fifo_embeds[:, :num_popped]], dim=1)
            cache_probs = torch.cat([stored_probs, fifo_probs[:, :num_popped]], dim=1)
            fifo_embeds = fifo_embeds[:, num_popped:]
            if cache_embeds.shape[1] > self.cache_length:
                cache_embeds, cache_probs = self._compress(cache_embeds, cache_probs, silence_embeds)
                self.is_compressed = True
            self.num_cache_frames = cache_embeds.shape[1]
            self.embeds[:, : self.num_cache_frames] = cache_embeds
            self.probs[:, : self.num_cache_frames] = cache_probs
        self.num_fifo_frames = fifo_embeds.shape[1]
        self.fifo[:, : self.num_fifo_frames] = fifo_embeds

    def _frame_scores(self, probs):
        log_probs = torch.log(probs.clamp(min=self.score_threshold))
        log_complements = torch.log((1.0 - probs).clamp(min=self.score_threshold))
        scores = log_probs - log_complements + log_complements.sum(dim=-1, keepdim=True) - math.log(0.5)
        is_speech = probs > 0.5
        scores = scores.masked_fill(~is_speech, float("-inf"))
        is_positive = scores > 0
        has_enough_positive = is_positive.sum(dim=1, keepdim=True) >= self.min_positive_scores
        return scores.masked_fill(~is_positive & is_speech & has_enough_positive, float("-inf"))

    @staticmethod
    def _boost_scores(scores, num_boosted, boost):
        _, indices = torch.topk(scores, num_boosted, dim=1, sorted=False)
        return scores.scatter_add(1, indices, scores.new_full(indices.shape, boost))

    def _compress(self, embeds, probs, silence_embeds):
        """Keeps the `cache_length` most important frames, grouped by speaker in their original order, `num_silence_frames` slots per speaker
        filled with the learned silence embedding."""
        batch_size, num_frames, num_speakers = probs.shape
        scores = self._frame_scores(probs)
        scores[:, self.cache_length :] += self.latest_boost  # the frames popped from the FIFO queue
        scores = self._boost_scores(scores, self.num_strong_boosted, -2.0 * math.log(0.5))
        scores = self._boost_scores(scores, self.num_weak_boosted, -math.log(0.5))
        scores = F.pad(scores, (0, 0, 0, self.num_silence_frames), value=float("inf"))
        embeds = torch.cat([embeds, silence_embeds.to(embeds).view(1, 1, -1).expand(batch_size, 1, -1)], dim=1)
        probs = F.pad(probs, (0, 0, 0, 1))
        num_scored_frames = num_frames + self.num_silence_frames
        sentinel = num_scored_frames * num_speakers
        top_scores, top_indices = torch.topk(scores.transpose(1, 2).reshape(batch_size, -1), self.cache_length, dim=1, sorted=False)
        top_indices, _ = torch.sort(top_indices.masked_fill(top_scores == float("-inf"), sentinel), dim=1)
        frame_indices = torch.where(top_indices == sentinel, num_frames, (top_indices % num_scored_frames).clamp(max=num_frames))
        batch_indices = torch.arange(batch_size, device=probs.device)[:, None]
        return embeds[batch_indices, frame_indices], probs[batch_indices, frame_indices]


class SortformerModules(nn.Module):
    """NeMo `sortformer_modules`: projection of the encoder output, sub-pixel upsampling back to the spectrogram frame rate, speaker classifier."""

    def __init__(self, fc_d_model, tf_d_model, num_speakers, subsampling_factor):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.encoder_proj = nn.Linear(fc_d_model, tf_d_model)
        self.subpixel_upsample = nn.Conv1d(tf_d_model, tf_d_model * subsampling_factor, kernel_size=3, padding=1)
        self.first_hidden_to_hidden = nn.Linear(tf_d_model, tf_d_model)
        self.single_hidden_to_spks = nn.Linear(tf_d_model, num_speakers)
        self.learnable_sil_emb = nn.Parameter(torch.empty(fc_d_model))

    def forward(self, x):
        batch_size, num_frames, _ = x.shape
        x = _linear(self.encoder_proj, x)
        x = _conv(self.subpixel_upsample, x.transpose(1, 2)).transpose(1, 2).reshape(batch_size, num_frames * self.subsampling_factor, -1)
        return _linear(self.single_hidden_to_spks, F.relu(_linear(self.first_hidden_to_hidden, F.relu(x))))


class SortformerDiarizer(nn.Module):
    """NeMo `diarization_model`: streaming Sortformer run offline, chunk by chunk with look-back and look-ahead frames, every chunk attending to
    the speaker cache and FIFO queue. Speaker logits at the spectrogram frame rate, speakers in order of first arrival."""

    def __init__(self, config):
        super().__init__()
        audio, head = config["audio_config"], config["head_config"]
        self.config = config
        self.encoder = TransformerEncoder(audio["hidden_size"], audio["num_attention_heads"], audio["num_hidden_layers"], audio["intermediate_size"], False,
                                          audio["rope_parameters"]["partial_rotary_factor"], audio["rope_parameters"]["rope_theta"], audio["num_mel_bins"], audio["subsampling_factor"])
        self.sortformer_modules = SortformerModules(head["audio_hidden_size"], head["hidden_size"], head["num_speakers"], head["subsampling_factor"])

    def forward(self, features, num_valid_frames, check_abort=None):
        """`features` [1, frames, mels] log-mel (frames a multiple of the subsampling factor), the first `num_valid_frames` valid."""
        config, factor = self.config, self.sortformer_modules.subsampling_factor
        embeds = self.encoder.pre_encode(features)
        num_embeds, num_valid_embeds = embeds.shape[1], -(-num_valid_frames // factor)
        cache = SpeakerCache(config["streaming_config"], config["fifo_length"], config["speaker_cache_update_period"], embeds)
        logits = []
        for start in range(0, num_embeds, config["chunk_length"]):
            end = min(start + config["chunk_length"], num_embeds)
            context_start = max(start - config["chunk_left_context"], 0)
            chunk_embeds = embeds[:, context_start : min(end + config["chunk_right_context"], num_embeds)]
            cached_embeds = cache.get_embeds()
            input_embeds = torch.cat([cached_embeds, chunk_embeds], dim=1)
            num_valid = cached_embeds.shape[1] + min(max(num_valid_embeds - context_start, 0), chunk_embeds.shape[1])
            states = self.encoder(input_embeds, num_valid if num_valid < input_embeds.shape[1] else None, check_abort)
            chunk_logits = self.sortformer_modules(states)
            del states
            cache.update(input_embeds, chunk_logits, self.sortformer_modules.learnable_sil_emb, end - start, num_valid, start - context_start)
            first = (cached_embeds.shape[1] + start - context_start) * factor
            logits.append(chunk_logits[:, first : first + (end - start) * factor])
        return torch.cat(logits, dim=1)[:, : features.shape[1]]


class ParallelExpertEncoder(nn.Module):
    """NeMo `perception.encoder`: the speech recognition encoder and the diarization model, with their output norms."""

    def __init__(self, audio_config):
        super().__init__()
        asr, diarization = audio_config["asr_config"], audio_config["diarization_config"]
        self.asr_encoder = TransformerEncoder(asr["hidden_size"], asr["num_attention_heads"], asr["num_hidden_layers"], asr["intermediate_size"], True,
                                              asr["rope_parameters"]["partial_rotary_factor"], asr["rope_parameters"]["rope_theta"], asr["num_mel_bins"], asr["subsampling_factor"])
        self.diarization_model = SortformerDiarizer(diarization)
        self.asr_norm = nn.LayerNorm(asr["hidden_size"])
        self.diar_norm = nn.LayerNorm(diarization["head_config"]["num_speakers"])


class AudioPerceptionModule(nn.Module):
    """NeMo `perception`: the parallel encoders and the projection to the language model's hidden size."""

    def __init__(self, audio_config, output_dim):
        super().__init__()
        self.encoder = ParallelExpertEncoder(audio_config)
        self.proj = nn.Linear(audio_config["asr_config"]["hidden_size"], output_dim)


class TimestampDecoder(nn.Module):
    """NeMo `ctc_timestamp.decoder`: transposed-convolution upsampling of the speech recognition encoder states, transformer with a scaled
    residual, and the classifier over the alignment tokenizer pieces and the CTC blank (`decoder_layers.0`)."""

    def __init__(self, config):
        super().__init__()
        hidden_size, factor = config["hidden_size"], config["upsampling_factor"]
        self.upsampling_factor, self.residual_scale = factor, config["residual_scale"]
        self.upsample = nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=factor, stride=factor)
        self.transformer = TransformerEncoder(hidden_size, config["num_attention_heads"], config["num_hidden_layers"], config["intermediate_size"], True,
                                              config["rope_parameters"]["partial_rotary_factor"], config["rope_parameters"]["rope_theta"])
        self.decoder_layers = nn.Sequential(nn.Conv1d(hidden_size, config["vocab_size"], kernel_size=1))

    def forward(self, states, num_valid_frames, chunks, check_abort=None):
        """Head states [1, frames * upsampling_factor, hidden] of `states` [1, frames, hidden] (the first `num_valid_frames` valid), each chunk
        encoded with its context."""
        factor = self.upsampling_factor
        hidden_states = states.new_empty(1, states.shape[1] * factor, states.shape[2])
        for context, chunk in chunks:
            x = _conv(self.upsample, states[:, context].transpose(1, 2)).transpose(1, 2)
            num_valid = factor * min(max(num_valid_frames - context.start, 0), context.stop - context.start)
            out = x + x.new_tensor(self.residual_scale) * self.transformer(x, num_valid if num_valid < x.shape[1] else None, check_abort)
            hidden_states[:, (context.start + chunk.start) * factor : (context.start + chunk.stop) * factor] = out[:, chunk.start * factor : chunk.stop * factor]
        return hidden_states

    def classify(self, hidden_states):
        return _conv(self.decoder_layers[0], hidden_states.transpose(1, 2)).transpose(1, 2)


class CtcTimestamp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = TimestampDecoder(config)


class NemotronPerception(nn.Module):
    """`config`: the checkpoint's config.json (its Transformers keys `audio_config`, `text_config`, `word_timestamp_config`)."""

    def __init__(self, config, compute_dtype=torch.float32):
        super().__init__()
        audio_config = config["audio_config"]
        self.compute_dtype = compute_dtype
        self.perception = AudioPerceptionModule(audio_config, config["text_config"]["hidden_size"])
        self.ctc_timestamp = CtcTimestamp(config["word_timestamp_config"])
        asr, diarization = audio_config["asr_config"], audio_config["diarization_config"]
        self.subsampling_factor = asr["subsampling_factor"]
        self.speaker_activity_threshold = audio_config["speaker_activity_threshold"]
        self.chunk_length, self.chunk_left_context, self.chunk_right_context = diarization["chunk_length"], diarization["chunk_left_context"], diarization["chunk_right_context"]
        self.speaker_embedding = speaker_embeddings(diarization["head_config"]["num_speakers"], asr["hidden_size"])

    def get_chunks(self, num_frames):
        """`(context, chunk)` slices of the encoder frames per chunk of `chunk_length` frames: `context` the frames an encoder run sees (the chunk
        with up to `chunk_left_context` and `chunk_right_context` more), `chunk` the chunk's frames within that run."""
        chunks = []
        for start in range(0, num_frames, self.chunk_length):
            end = min(start + self.chunk_length, num_frames)
            context = slice(max(start - self.chunk_left_context, 0), min(end + self.chunk_right_context, num_frames))
            chunks.append((context, slice(start - context.start, end - context.start)))
        return chunks

    def forward(self, features, mask, check_abort=None):
        """`features` [1, frames, mels] and `mask` [1, frames] from `NemotronFeatureExtractor` (frames a multiple of the subsampling factor).
        Returns `audio_embeds` [num_audio_tokens, lm_hidden] (the valid encoder frames, the embeddings of the prompt's audio tokens),
        `asr_hidden` [1, frames / 8, hidden] (encoder states before `asr_norm`, the input of `word_timestamp_logits`), `speaker_logits`
        [1, frames, num_speakers] and `num_audio_tokens`, all in `compute_dtype`."""
        dtype, factor, encoder = self.compute_dtype, self.subsampling_factor, self.perception.encoder
        num_valid = int(mask.sum())
        num_frames, num_audio_tokens = features.shape[1] // factor, -(-num_valid // factor)
        asr_input = normalize_per_feature(features, mask).to(dtype)
        asr_hidden = asr_input.new_empty(1, num_frames, encoder.asr_norm.normalized_shape[0])
        for context, chunk in self.get_chunks(num_frames):
            x = encoder.asr_encoder.pre_encode(asr_input[:, context.start * factor : context.stop * factor])
            num_keys = min(max(num_audio_tokens - context.start, 0), x.shape[1])
            states = encoder.asr_encoder(x, num_keys if num_keys < x.shape[1] else None, check_abort)
            asr_hidden[:, context.start + chunk.start : context.start + chunk.stop] = states[:, chunk]
            del x, states
        del asr_input
        speaker_logits = encoder.diarization_model(features.to(dtype), num_valid, check_abort)
        speaker_probabilities = speaker_logits.sigmoid().unflatten(1, (num_frames, -1)).mean(2)
        speaker_activity = (speaker_probabilities > self.speaker_activity_threshold).to(dtype)
        speaker_states = _norm(encoder.diar_norm, speaker_activity) @ self.speaker_embedding.to(device=speaker_activity.device, dtype=dtype)
        audio_embeds = _linear(self.perception.proj, (_norm(encoder.asr_norm, asr_hidden) + speaker_states)[0, :num_audio_tokens])
        return {"audio_embeds": audio_embeds, "asr_hidden": asr_hidden, "speaker_logits": speaker_logits, "num_audio_tokens": num_audio_tokens}

    def word_timestamp_logits(self, asr_hidden, mask, check_abort=None, columns=None):
        """Word timestamp head on `asr_hidden` (from `forward`), in the encoder's chunks: raw logits [1, frames / 8 * upsampling_factor, vocab]
        (the reference's `get_word_timestamp_logits`, the last class the CTC blank). With `columns` (class ids, see
        `NemotronAligner.classes`), float32 log-probabilities [valid head frames, len(columns)] of those classes only, computed by chunks of
        frames so that the full logits never exist at once."""
        runtime.ensure_loaded(self)  # a custom entry point: MMGP loads the base weights a hooked forward would
        decoder = self.ctc_timestamp.decoder
        num_audio_tokens = -(-int(mask.sum()) // self.subsampling_factor)
        hidden_states = decoder(asr_hidden.to(self.compute_dtype), num_audio_tokens, self.get_chunks(asr_hidden.shape[1]), check_abort)
        if columns is None:
            return decoder.classify(hidden_states)
        columns = torch.as_tensor(columns, device=hidden_states.device)
        num_head_frames = num_audio_tokens * decoder.upsampling_factor
        log_probs = [decoder.classify(hidden_states[:, start : min(start + LOG_PROB_CHUNK_FRAMES, num_head_frames)])[0].log_softmax(-1)[:, columns].float()
                     for start in range(0, num_head_frames, LOG_PROB_CHUNK_FRAMES)]
        return torch.cat(log_probs)


def load_perception(perception_file, config, compute_dtype=torch.float32):
    """Builds the perception on the meta device and maps the checkpoint (`perception.*` and `ctc_timestamp.*` keys, stored dtypes kept: the
    weights are cast to `compute_dtype` at use). `ctc_tokenizer` is the alignment SentencePiece model stored in the checkpoint."""
    with torch.device("meta"):
        perception = NemotronPerception(config, compute_dtype)
    offload.load_model_data(perception, perception_file, writable_tensors=False, default_dtype=None)
    perception._convertWeightsFloatTo = None
    for module in perception.modules():  # precision locks: the word timestamp head is stored in float32, the encoders in BF16
        if any(parameter.dtype == torch.float32 for parameter in module.parameters(recurse=False)):
            module._lock_dtype = torch.float32
    with safe_open(perception_file, framework="pt", device="cpu") as weights:
        perception.ctc_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=weights.get_tensor("ctc_timestamp.tokenizer.model_proto").numpy().tobytes())
    return perception.eval()
