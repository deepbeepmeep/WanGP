# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from mmgp import cuda_graphs, runtime
from shared.utils.phase_progress import generation_progress, PhaseProgress, set_phase_status
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.speech_dialogue import SPEAKER_LINE_PATTERN, parse_dialogue
from .loading import CONFIG, load_decoder, load_lm, load_speaker_encoder
from .lm import GenerationCancelled
from .prompt import CUE_MARKER, TextEncoder, condition_bucket, prompt_rows, split_cues, undelay
from .sampler import FrameSampler, Sampling
from .steering import Steering, cue_values, doses, is_raw, parse_controls
from .voices import DEFAULT_VOICE, VOICES

PREFILL_ROWS = 512  # prompt rows per prefill pass
RAMP_FRAMES = 3  # an inline cue fades in over 0.2 s, as released
READ_LEAD, READ_SMOOTH = 1, 3  # a cue starts one token before its words; the reading is the median of the last 3 readouts (released)
# The alignment heads of each language: [layer, head] pairs whose attention follows the text being read (released with the code).
READING_HEADS = json.loads(Path(__file__).with_name("reading_heads.json").read_text(encoding="utf-8"))
SPEAKING_RATES = {"very_slow": 9.0, "slow": 12.0, "normal": 15.0, "fast": 18.0, "very_fast": 22.0}  # UTF-8 bytes per second of speech
DECODE_CONTEXT_FRAMES = 450  # left context of a decoding tile (30 s, about the decoder's attention reach)


class Zonos3Pipeline:
    def __init__(self, lm_path, decoder_path, speaker_path, voices_path, tokenizer_path, steering_path, features):
        self.features = features
        lm, codec = CONFIG["lm"], CONFIG["codec"]
        self.n_codebooks, self.codebook_size, self.pad, self.eoa, self.text_vocab = lm["n_codebooks"], lm["codebook_size"], lm["audio_pad_id"], lm["eoa_id"], lm["text_vocab"]
        self.sample_rate, self.hop = codec["sample_rate"], codec["hop"]
        self.gain = 10.0 ** (-codec["headroom_db"] / 20.0)
        self.limit = 10.0 ** (codec["output_limit_dbfs"] / 20.0)
        self.transformer = load_lm(lm_path, features)
        self.vae = load_decoder(decoder_path)
        self.speaker_encoder = load_speaker_encoder(speaker_path)
        self.voices = load_file(str(voices_path))
        self.text = TextEncoder(tokenizer_path, CONFIG["text"], CONFIG["languages"])
        self.steering = Steering(steering_path)
        self.transformer.steered_sources = tuple(self.steering.sources)
        self.transformer.abort_requested = self._abort_requested
        self.vae.get_VAE_tile_size = self.get_vae_tile_size
        self.reference_cache = TextEncoderCache()  # speaker embeddings of voice references, by file
        self._interrupt = self._early_stop = False

    @staticmethod
    def get_vae_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=None, output_width=None):
        """Codec frames per decoding tile (15 per second of speech), 0 for one pass; preset mapping, not yet calibrated."""
        return {0: 0 if device_mem_capacity >= 12000 else 2400, 1: 0, 2: 2400, 3: 1200}[vae_config]

    def _abort_requested(self):
        return self._interrupt

    def _check_abort(self, *args):
        if self._interrupt:
            raise GenerationCancelled()

    def request_early_stop(self):
        self._early_stop = True

    def release_caches(self):
        if torch.cuda.is_initialized():
            torch.cuda.current_stream().synchronize()
        cuda_graphs.invalidate()
        self.transformer.release_cache()

    # -- voices -------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _file_key(path):
        stat = os.stat(path)
        return os.path.abspath(path), stat.st_size, stat.st_mtime_ns

    def _reference_embedding(self, path, label):
        def encode(paths):
            set_phase_status(f"Encoding Voice Reference{label}")
            audio, sample_rate = sf.read(paths[0], dtype="float32", always_2d=True)
            return [self.speaker_encoder.embed(torch.from_numpy(audio.mean(axis=1).copy()), sample_rate)]
        return self.reference_cache.encode(encode, [path], cache_keys=[("embedding", *self._file_key(path))])[0]

    def _voice(self, name, conditioning):
        """A built-in voice: its reference codes (cut to the model's 60 s) or its speaker embedding."""
        if conditioning == "prefix":
            return {"codes": self.voices[name + ".codes"][:CONFIG["generation"]["reference_max_frames"]]}
        return {"embedding": self.voices[name + ".embedding"]}

    # -- generation ---------------------------------------------------------------------------------------------------------------------

    def _prompt(self, spoken, voice, rate):
        silence = CONFIG["generation"]["silence_prefill_frames"]
        ids = self.text.spoken_ids(spoken)
        rows, speaker_row, condition_start = prompt_rows(ids, n_codebooks=self.n_codebooks, audio_pad_id=self.pad, text_vocab=self.text_vocab,
            condition_count=len(CONFIG["lm"]["condition_names"]), speaker_row="embedding" in voice, reference_codes=voice.get("codes"),
            silence_frame=CONFIG["codec"]["silence_frame"], silence_frames=silence)
        names, counts = CONFIG["lm"]["condition_names"], CONFIG["lm"]["condition_bucket_counts"]
        buckets = [count for count in counts]  # unknown loudness and bandwidth
        if rate is not None:
            buckets[names.index("speaking_rate")] = condition_bucket(rate, CONFIG["lm"]["condition_edges"]["speaking_rate"])
        conditions = [(condition_start + table, table, bucket) for table, bucket in enumerate(buckets)]
        return rows, voice.get("embedding"), speaker_row, conditions, silence

    def _stretches(self, spoken, cues, values):
        """The steered stretches of a take: [(its first text token, (constant, released) deltas kept from the Voice Controls, those of
        its cue)]: the Voice Controls from the start, then each inline cue from the first token of its words (0 is BOS). None when
        nothing steers the hidden states."""
        parts = spoken.split(CUE_MARKER)
        stretches = [(0 if index == 0 else len(self.text.spoken_ids(CUE_MARKER.join(parts[:index]))) - 1,
                      *(self.steering.hidden(doses(controls)) for controls in ((values, {}) if index == 0 else cue_values(values, cues[index - 1]))))
                     for index in range(len(parts))]
        if all(delta is None for _, kept, cue in stretches for delta in (*kept, *cue)):
            return None
        return stretches

    def _generate_codes(self, rows, speaker, speaker_row, conditions, max_frames, sampling, seed, callback, offset, total, part, stretches=None, text=None):
        """Generated delayed rows [n, n_codebooks] (CPU) and the end-of-speech frame among them (None: the budget ran out). stretches:
        the steering of the take (see _stretches), added on the generated frames (the prompt is not steered); with inline cues, text =
        (first text row, text rows, language): each cue starts once the model reads its words."""
        lm, n_codebooks = self.transformer, self.n_codebooks
        prompt_length = len(rows)
        budget = max_frames + n_codebooks  # the last frames complete n_codebooks - 1 rows later
        lm.setup_caches(prompt_length + budget + 1)
        device = runtime.execution_device(lm)
        positions = torch.arange(prompt_length + budget + 1, device=device)
        chunks = math.ceil(prompt_length / PREFILL_ROWS)
        encoding = "Encoding Text Prompt" + part.replace(" - Turn", "")
        set_phase_status(encoding)
        with cuda_graphs.scope(enabled=False), ExitStack() as stack:
            progress = stack.enter_context(PhaseProgress(chunks * len(lm.layers), encoding, "layers", next_status="Generating Speech" + part))
            bar = stack.enter_context(tqdm(total=chunks * len(lm.layers), desc=encoding, unit="layer", mininterval=1 / 3, leave=False))

            def completed(*args):
                self._check_abort()
                bar.update()
                progress.advance()
            for layer in lm.layers:
                stack.callback(layer.register_forward_hook(completed).remove)
            for start in range(0, prompt_length, PREFILL_ROWS):
                end = min(start + PREFILL_ROWS, prompt_length)
                logits = lm.prefill(rows[start:end], positions[start:end], start, speaker, speaker_row, conditions)
        sampler = FrameSampler(sampling, n_codebooks=n_codebooks, codebook_size=self.codebook_size, audio_pad_id=self.pad, eoa_id=self.eoa, device=device, seed=seed)
        generated = torch.empty(budget, n_codebooks, dtype=torch.long, device=device)
        text_column = torch.full((1, 1), self.text_vocab, dtype=torch.long, device=device)
        audio_prefix = bool((rows[-1, :n_codebooks] != self.pad).any())
        cued = stretches is not None and len(stretches) > 1
        host = torch.empty(n_codebooks + cued, dtype=torch.long, device="cpu", pin_memory=device.type == "cuda")
        done = torch.cuda.Event(blocking=True) if device.type == "cuda" else None
        eos, countdown, produced, updated = None, -1, 0, 0.0
        steer = [tuple(None if delta is None else delta.to(device) for delta in (*kept, *cue)) for _, kept, cue in stretches or ()]
        frame_rate = CONFIG["codec"]["frame_rate"]
        if cued:
            # The alignment heads of the language give the text token read at each step (smoothed: median of the last readouts, never
            # back, at most one token further per step); a cue fades in over RAMP_FRAMES once the reading reaches its first token
            # minus READ_LEAD, from the next frame decided. A cue at the start of the text steers from the first frame.
            start, length, language = text
            lm.set_probe(start, length, READING_HEADS[language])
            anchors = [anchor for anchor, *_ in stretches[1:]]
            onsets = [1 if anchor <= READ_LEAD else None for anchor in anchors]  # the first fully steered frame of each cue
            readings, reading = [], 0
            # zeros where nothing steers: one decoding graph, the same results as no delta
            zero = torch.zeros_like(next(delta for pair in steer for delta in pair if delta is not None))

        def stretch_delta(frame, onset, constant, released, cue_constant, cue_released):
            """The deltas of a stretch: its constant ones and its released ones at their weight, the cue's counted from its onset."""
            delta = None if released is None else released * self.steering.release_weight(frame, frame_rate)
            delta = constant if delta is None else delta if constant is None else constant + delta
            for extra in (None if cue_released is None else cue_released * self.steering.release_weight(frame - onset + 1, frame_rate), cue_constant):
                delta = extra if delta is None else delta if extra is None else delta + extra
            return delta

        def deltas(frame):
            """The steering deltas of generated frame `frame` (from 1): the Voice Controls, then each cue reached, faded in."""
            if not cued:
                return None if not steer else stretch_delta(frame, 1, *steer[0])
            reached = [1.0] + [0.0 if onset is None else min(max((frame - onset + RAMP_FRAMES) / RAMP_FRAMES, 0.0), 1.0) for onset in onsets] + [0.0]
            summed = None
            for index, pairs in enumerate(steer):
                weight = reached[index] - reached[index + 1]
                delta = stretch_delta(frame, ([1] + onsets)[index], *pairs) if weight > 0 else None
                if delta is not None:
                    delta = delta if weight == 1.0 else delta * weight
                    summed = delta if summed is None else summed + delta
            return zero if summed is None else summed
        if callback is not None:
            callback(step_idx=offset - 1, override_num_inference_steps=total, denoising_extra="Generating Speech" + part, progress_unit="frames")
        started = time.perf_counter()
        with tqdm(total=max_frames, desc="Generating Speech" + part, unit="frame", mininterval=1 / 3) as bar, cuda_graphs.scope(enabled=self.features.graphs):
            for step in range(budget):
                self._check_abort()
                if self._early_stop:
                    break
                token = sampler(logits, generated[:step])
                if not audio_prefix and step < n_codebooks - 1:
                    token[step + 1:] = self.pad  # codebooks whose delayed stream has not started yet
                generated[step] = token
                del logits
                host[:n_codebooks].copy_(token, non_blocking=True)
                if cued:
                    host[n_codebooks:].copy_(lm.reading, non_blocking=True)  # the text token read by the step that gave these logits
                if done is not None:
                    done.record()
                if step + 1 < budget:
                    # queued before this frame is read: the GPU goes on with the next step meanwhile
                    position = prompt_length + step
                    logits = lm.decode(torch.cat((token[None], text_column), 1), positions[position:position + 1], position, deltas(step + 1))
                if done is not None:
                    done.synchronize()
                codes = host.tolist()
                if cued:
                    readings.append(codes.pop())
                    reading = max(reading, min(sorted(readings[-READ_SMOOTH:])[len(readings[-READ_SMOOTH:]) // 2], reading + 1))
                    onsets = [step + 1 + RAMP_FRAMES if onset is None and reading >= anchor - READ_LEAD else onset for onset, anchor in zip(onsets, anchors)]
                produced = step + 1
                if eos is None and self.eoa in codes:
                    # every codebook stops in turn: the frame ends where the stop entered its own (delayed) codebook
                    eos = max(0, step - max(index for index, code in enumerate(codes) if code == self.eoa))
                    countdown = n_codebooks + 1
                if countdown > 0:
                    countdown -= 1
                    if countdown == 0:
                        break
                if eos is None:
                    bar.update()
                    now = time.monotonic()
                    if callback is not None and now - updated >= 1 / 3:
                        frames = min(offset + step + 1, total)
                        callback(step_idx=frames - 1, override_num_inference_steps=total, progress_unit="frames",
                                 denoising_extra=f"Generating Speech{part} - {frames / CONFIG['codec']['frame_rate']:.1f}s")
                        updated = now
        elapsed = time.perf_counter() - started
        finish = "eos" if eos is not None else "early stop" if self._early_stop else "length"
        cue_frames = f"; cues fully in at frames {onsets}" if cued else ""
        print(f"[ZONOS3] {produced} rows in {elapsed:.2f}s ({produced / max(elapsed, 1e-6):.1f} rows/s); finish: {finish}{cue_frames}")
        return generated[:produced].cpu(), eos

    def _decode(self, prefill_rows, generated, eos, tile_frames):
        """The audio of the silence prefill and generated frames up to the end of speech: frame f takes codebook c from row f + c."""
        buffer = torch.cat((prefill_rows[:, :self.n_codebooks], generated))
        target = len(buffer) - (self.n_codebooks - 1)
        if eos is not None:
            target = min(target, len(prefill_rows) + eos)
        if target <= 0:
            return torch.zeros(0, device="cpu")
        frames = undelay(buffer[:target + self.n_codebooks - 1], self.pad).T.contiguous()  # [n_codebooks, frames]
        if bool((frames >= self.codebook_size).any()):
            raise RuntimeError("ZONOS3 produced special tokens inside the speech frames")
        device = runtime.execution_device(self.vae)
        tile = tile_frames or target
        parts = []
        total = math.ceil(target / tile)
        with PhaseProgress(total, "Decoding Speech", "tiles") as progress, tqdm(total=total, desc="Decoding Speech", unit="tile", mininterval=1 / 3) as bar:
            for start in range(0, target, tile):
                self._check_abort()
                end = min(start + tile, target)
                left = max(0, start - DECODE_CONTEXT_FRAMES)
                audio = self.vae(frames[None, :, left:end].to(device))
                parts.append(audio[0, (start - left) * self.hop:].float().cpu())
                del audio
                bar.update()
                progress.advance()
        audio = torch.cat(parts) * self.gain
        # soft limiter above -1 dBFS: slope 1 at the knee, bounded by full scale
        over = audio.abs() > self.limit
        audio[over] = audio[over].sign() * (self.limit + (1 - self.limit) * torch.tanh((audio[over].abs() - self.limit) / (1 - self.limit)))
        return audio

    @torch.inference_mode()
    @generation_progress
    def generate(self, input_prompt, *, alt_prompt="", audio_guide=None, audio_guide2=None, audio_guide3=None, audio_prompt_type="",
                 duration_seconds=60, pause_seconds=0.3, temperature=0.75, top_k=80, top_p=0.69, seed=-1, VAE_tile_size=0, custom_settings=None,
                 callback=None, **kwargs):
        self._interrupt = self._early_stop = False
        settings = custom_settings or {}
        language = settings.get("language", "en_us")
        rate = SPEAKING_RATES.get(settings.get("speaking_rate", "auto"))
        values, conditioning, normalize = voice_controls(alt_prompt)
        dose_map = doses(values)
        sampling = Sampling(temperature=temperature, top_k=top_k, top_p=top_p)
        seed = int(seed) if seed is not None and seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,), device="cpu"))
        try:
            with ExitStack() as stack:
                stack.enter_context(runtime.execution_scope(self._check_abort))
                text = input_prompt.strip()
                speakers = len({int(match.group(2)) for match in SPEAKER_LINE_PATTERN.finditer(text)})
                turns = parse_dialogue(text, speakers, line_labels=True, canonical_ids=True) if speakers else [(0, text)]
                guides = [guide for letter, guide in zip("ABD", (audio_guide, audio_guide2, audio_guide3)) if letter in audio_prompt_type]
                if guides:
                    voices = [{"embedding": self._reference_embedding(guide, f" {index}/{len(guides)}" if len(guides) > 1 else "")} for index, guide in enumerate(guides, 1)]
                else:
                    names = speaker_voices(settings.get("voice", DEFAULT_VOICE), settings.get("dialogue_voices", ""), speakers)
                    if speakers > 1:
                        tqdm.write(f"[ZONOS3] Speaker voices: {', '.join(VOICES[name][0] for name in names)}")
                    voices = [self._voice(name, conditioning) for name in names]
                voices = [{"embedding": self.steering.move_embedding(voice["embedding"], dose_map)} if "embedding" in voice else voice for voice in voices]
                max_frames = max(1, math.floor(duration_seconds * CONFIG["codec"]["frame_rate"]))
                pause = torch.zeros(int(round(pause_seconds * self.sample_rate)), device="cpu")
                tile = VAE_tile_size if isinstance(VAE_tile_size, int) else 0
                audio, produced = [], 0
                for index, (speaker, words) in enumerate(turns, 1):
                    part = f" - Turn {index}/{len(turns)}" if len(turns) > 1 else ""
                    spoken, cues = self.text.spoken(words, language, normalize)
                    rows, embedding, speaker_row, conditions, silence = self._prompt(spoken, voices[speaker], rate)
                    stretches = self._stretches(spoken, [parse_controls(cue, cue=True)[0] for cue in cues], values)
                    length = len(self.text.spoken_ids(spoken))
                    generated, eos = self._generate_codes(rows, embedding, speaker_row, conditions, max_frames - produced, sampling, seed + index - 1,
                                                          callback, produced, max_frames, part, stretches, (len(rows) - silence - length, length, language))
                    self.release_caches()
                    if audio:
                        audio.append(pause)
                    audio.append(self._decode(rows[len(rows) - silence:], generated, eos, tile))
                    produced += len(audio[-1]) // self.hop
                    if self._early_stop or produced >= max_frames:
                        break
                audio = torch.cat(audio)
                if audio.numel() == 0:
                    return None
                if not torch.isfinite(audio).all():
                    raise FloatingPointError("ZONOS3 decoded non-finite audio")
                return {"x": audio[None], "audio_sampling_rate": self.sample_rate}
        except GenerationCancelled:
            return None
        finally:
            self.release_caches()


DEFAULT_CONDITIONING = "prefix"


def voice_controls(text):
    """Voice Controls -> (control values, conditioning of built-in voices, text normalization). Voice-design controls move the speaker
    embedding: built-in voices then use speaker-embedding conditioning."""
    values, options = parse_controls(text)
    raw = sorted(name for name, value in values.items() if any(is_raw(direction) for direction in doses({name: value})))
    conditioning = options.get("conditioning") or ("embedding" if raw else DEFAULT_CONDITIONING)
    if raw and conditioning == "prefix":
        raise ValueError(f"Voice Controls that change the voice itself ({', '.join(raw)}) need conditioning=embedding")
    return values, conditioning, options.get("normalize", "on") == "on"


def speaker_voices(voice, others, speakers):
    """Built-in voice ids of Speakers 1..speakers: the selected voice, the voices named in Other Speakers' Voices (comma-separated),
    then for the speakers left the voices of the Voice list not used yet, in its order, those of Speaker 1's language first."""
    chosen = [voice_id(name) for name in [voice, *(name.strip() for name in others.split(",") if name.strip())]]
    language = VOICES[chosen[0]][1]
    spare = [key for key in VOICES if VOICES[key][1] == language and key not in chosen] + [key for key in VOICES if VOICES[key][1] != language and key not in chosen]
    return (chosen + spare)[:max(speakers, len(chosen))]


def check_cues(text):
    """Raises ValueError for an unmatched bracket or a cue that is not a style or a control that can change within a take."""
    for cue in split_cues(text)[1]:
        parse_controls(cue, cue=True)


def voice_id(name):
    """A voice id from its id or its name (case-insensitive), e.g. "Imogen"."""
    if name in VOICES:
        return name
    for key, (label, _, _) in VOICES.items():
        if label.lower() == name.lower():
            return key
    raise ValueError(f"Unknown ZONOS3 voice '{name}': use one of {', '.join(label for label, _, _ in VOICES.values())}")
