# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Nemotron 3.5 Transcribe: speaker-attributed speech recognition, translation and word timestamps.

Contract shared by the handler and the Deepy tool:
- NemotronTranscribePipeline(lm_file, experts_file, perception_file, assets_dir, lm_features, draft_file=None)
  .transformer is the language model (dense weights; its routed experts are MMGP expert banks loaded from experts_file),
  .perception the audio encoders, word timestamp head and projection. MMGP manages both (the handler's offload definition).
- generate(...) returns a dict (speakers numbered from 1 everywhere: the model's <spk:0> is Speaker 1):
    text: the transcript (or translation) after the loop guardrail, speaker tags rendered as "Speaker N: ..." lines when speakers=True
    raw_text: the decoded model output, special tags kept
    turns: [{"speaker": int | None, "text": str, "start": float | None, "end": float | None}]
    words: [{"word": str, "start": float, "end": float, "speaker": int | None}] or None (word_timestamps=False or alignment failed)
    A translation with word_timestamps is timed by sentence: the recording is transcribed and aligned, then each sentence is translated from
    its own audio (turns timed, words None); without them, or when the alignment fails, the whole recording is translated at once (untimed).
    diarization: [{"speaker": int, "start": float, "end": float}] or None
    alignment_error: str | None
    duration: float (seconds of audio transcribed), truncated: bool (max_new_tokens reached)
"""
import contextlib
import json
from pathlib import Path

import torch
from mmgp import offload, runtime

TASKS = ("transcribe", "translate")
ASSET_FOLDER = "nemotron_transcribe"
ASSET_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "processor_config.json", "NemotronTranscribe_perception_bf16.safetensors"]
DRAFT_FILE = "NemotronTranscribe_dflash2_bf16.safetensors"
EXPECTED_TOKENS_PER_SECOND = 4.5  # transcript tokens per second of speech (progress estimate; measured 3.9-5.3 on the test clips)
PREFILL_CHUNK_TOKENS = 4096  # prompt processing by chunks: bounded activations (Mamba scan, routed experts) for recordings up to an hour
SEGMENT_PADDING = 0.15  # seconds of audio kept around a sentence that a timed translation translates on its own
SEGMENT_MAX_TOKENS = 512
SEGMENT_MAX_SECONDS, SEGMENT_PAUSE_SECONDS = 20.0, 1.0
SENTENCE_ENDS = ".?!。？！"


def translation_segments(words, speakers):
    """(start, end, speaker) of the stretches that a timed translation translates one by one: a sentence of one speaker, also cut at a pause
    and after SEGMENT_MAX_SECONDS."""
    spans, current = [], []
    for word in words:
        if current and (speakers and word["speaker"] != current[-1]["speaker"] or word["start"] - current[-1]["end"] > SEGMENT_PAUSE_SECONDS
                        or word["end"] - current[0]["start"] > SEGMENT_MAX_SECONDS):
            spans.append(current)
            current = []
        current.append(word)
        if str(word["word"]).strip()[-1:] in SENTENCE_ENDS:
            spans.append(current)
            current = []
    spans += [current] if current else []
    return [(span[0]["start"], span[-1]["end"], span[0]["speaker"] if speakers else None) for span in spans]


@contextlib.contextmanager
def full_fp32():
    """float32 matmuls and convolutions of the perception without TF32 (other models enable it globally when they are imported)."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def _numbered(speaker):
    return None if speaker is None else int(speaker) + 1


def _decode_floats(value):
    # NeMo's JSON encodes infinities as {"__float__": "Infinity"}
    if isinstance(value, dict) and set(value) == {"__float__"}:
        return float(value["__float__"])
    if isinstance(value, list):
        return [_decode_floats(item) for item in value]
    return value


def lm_config(config):
    from transformers import PretrainedConfig
    values = {key: _decode_floats(value) for key, value in config["llm_config"].items() if key not in ("architectures", "dtype", "transformers_version")}
    return PretrainedConfig(**values, torch_dtype="bfloat16")


def load_language_model(files, config, lm_features):
    """The Nemotron-H LM (dense file, then the routed-expert file) on meta, loaded from mmap; float32 SSM parameters and routing bias
    kept as stored."""
    from shared.llm_engines.lmgp.models.nemotron_h import NemotronHForCausalLM
    with torch.device("meta"):
        # BF16 modules: the compute dtype of the quantized linears; the float32 SSM parameters and routing bias keep their stored dtype
        # (precision-locked modules)
        model = NemotronHForCausalLM(lm_config(config), use_triton=lm_features.triton).to(torch.bfloat16)
    offload.load_model_data(model, [str(path) for path in files], writable_tensors=False, default_dtype=None)
    model._convertWeightsFloatTo = None  # the float32 SSM parameters and routing bias stay float32
    model._model_dtype = torch.bfloat16
    model._lm_engine_mmgp = True  # MMGP manages its weights: the engine sizes its KV cache without reading the free VRAM
    model._lm_engine_mmgp_graphs = lm_features.graphs  # the engine's decoding passes are MMGP graphs: the LM may stream its layers
    model._prefill_chunk_tokens = PREFILL_CHUNK_TOKENS
    model._mm_graph_keys = 32  # its decoding passes: one token, and the lengths of speculative verifications
    model._mm_graph_placement_key = True  # only MMGP moves its weights: the key of a whole-model graph is checked in constant time
    for block in model.model.layers:
        block._mm_graph_keys = 32
    runtime.map_in_ram(model.model.embed_tokens)  # the head is not tied to it: decoding reads a few rows over the bus
    return model.eval()


DRAFT_TOKENS = 7  # DFlash2 drafts verified per pass
USE_DRAFT = False  # DFlash2 speculative decoding: off until validated with the hybrid target


def install_draft(model, draft_file, config_file):
    """NVIDIA's DFlash2 block drafter (lmgp BlockDraft): it reads the hidden states of six target layers and shares the LM's embeddings
    and head; the engine verifies its drafts (BlockDraftRunner)."""
    from shared.llm_engines.lmgp.models.block_draft import BlockDraft
    from shared.prompt_enhancer.block_draft import _load_draft_config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    config = _load_draft_config(config_file)
    if config.hidden_size != model.config.hidden_size or config.vocab_size != model.config.vocab_size:
        raise ValueError("The DFlash2 drafter does not match the Nemotron language model")
    with torch.device("meta"):
        draft = BlockDraft(config, "dflash2")
    with torch.device("cpu"):
        draft.rotary_emb = Qwen3RotaryEmbedding(config)
    offload.load_model_data(draft, str(draft_file), writable_tensors=False, default_dtype=torch.bfloat16)
    model.mtp = draft.eval()
    model._block_draft = True
    model._block_draft_max_tokens = DRAFT_TOKENS
    model._prompt_enhancer_speculative_decoding = True
    model._prompt_enhancer_speculative_tokens = model._prompt_enhancer_speculative_sampling_tokens = DRAFT_TOKENS


class NemotronTranscribePipeline:
    def __init__(self, lm_file, experts_file, perception_file, assets_dir, lm_features, draft_file=None):
        from .features import NemotronFeatureExtractor
        from .perception import load_perception
        from .prompting import NemotronPrompt
        self.assets_dir, self.lm_file, self.lm_features, self.draft_file = Path(assets_dir), str(lm_file), lm_features, draft_file
        perception_file, experts_file = str(perception_file), str(experts_file)
        self.config = json.loads((self.assets_dir / "config.json").read_text(encoding="utf-8"))
        processor = json.loads((self.assets_dir / "processor_config.json").read_text(encoding="utf-8"))
        self.features = NemotronFeatureExtractor(processor["feature_extractor"])
        self.prompt = NemotronPrompt(self.assets_dir, processor)
        self.transformer = load_language_model([lm_file, experts_file], self.config, lm_features)
        if draft_file is not None and USE_DRAFT and lm_features.triton and lm_features.flash:
            install_draft(self.transformer, draft_file, self.assets_dir / "dflash2_config.json")
        self.perception = load_perception(perception_file, self.config)
        self.engine = None
        self._interrupt = False

    def _abort_check(self, abort_fn=None, *args):
        # WanGP sets _interrupt on the pipeline; the Deepy tool may pass its own abort_fn
        if self._interrupt or (abort_fn is not None and abort_fn()):
            raise InterruptedError("Transcription interrupted")

    def _engine(self):
        if self.engine is None:
            from shared.llm_engines.lmgp.vllm_support import TextEngine
            self.engine = TextEngine(self.transformer, self.lm_file, self.prompt.tokenizer, enforce_eager=not self.lm_features.graphs)
        return self.engine

    def release(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    @torch.inference_mode()
    def _decode(self, prompt_ids, prompt_embeds, max_new_tokens, seed, callback, abort_fn, expected_tokens):
        from shared.llm_engines.lmgp import SamplingParams
        from shared.llm_engines.lmgp.engine.block_manager import BlockManager
        from shared.llm_engines.lmgp.layers.kv_formats import overlay_slots
        engine = self._engine()
        engine.reserve_runtime(len(prompt_ids), max_new_tokens, 1.0)
        engine._ensure_llm()
        llm = engine._llm
        llm.scheduler.eos = self.prompt.eos_token_id
        llm.model_runner.ensure_runtime_ready()
        # Prompts differ by their audio embeddings, not by their token ids: no prefix may be reused from an earlier request
        llm.scheduler.block_manager = BlockManager(llm.config.num_kvcache_blocks, llm.config.kvcache_block_size, overlay_slots(llm.config))
        # GPU copies of the experts as the budget or the dynamic preload allows: MMGP accepts once the dense blocks are resident (after
        # the first passes), so asked again after each token until it does; a placement change releases the graphs that read them
        placed = [runtime.place_experts(self.transformer)]
        tokens = []

        def update(token):
            self._abort_check(abort_fn)
            tokens.append(token)
            if not placed[0]:
                placed[0] = runtime.place_experts(self.transformer)
            if callback is not None:
                # an estimate from the audio duration (the length is unknown until the end of the transcript), grown when it is passed
                total = min(max_new_tokens, max(expected_tokens, len(tokens) + 16))
                callback(step_idx=len(tokens) - 1, override_num_inference_steps=total, denoising_extra=f"Generating Transcript: {len(tokens)} tokens", progress_unit="tokens")

        def process(input_ids, logits):
            self._abort_check(abort_fn)
            return logits

        process._requires_input_ids = False
        params = SamplingParams(temperature=1.0, top_k=1, max_tokens=max_new_tokens, seed=seed, logits_processor=process, logits_processor_update_state=update)
        # the engine's embedded prefill takes multimodal (3-row) positions; Nemotron-H's attention has no positional encoding
        positions = torch.arange(len(prompt_ids), dtype=torch.long, device="cpu").repeat(3, 1)
        try:
            outputs = llm.generate_embedded([prompt_ids], [prompt_embeds], [positions], params, use_tqdm=False)
            generated = list(outputs[0]["token_ids"])
        finally:
            llm.reset()
        ended = bool(generated) and generated[-1] == self.prompt.eos_token_id
        return generated[:-1] if ended else generated, not ended

    @torch.inference_mode()
    def generate(self, *, audio_path, task="transcribe", speakers=True, word_timestamps=True, verbatim=True, context_terms="", target_language="",
                 instructions="", max_new_tokens=16384, seed=0, callback=None, abort_fn=None, set_progress_status=None, **kwargs):
        """The transcription dict described in the module docstring, None when interrupted."""
        self._interrupt = False
        try:
            with runtime.execution_scope(lambda *args: self._abort_check(abort_fn)):
                return self._generate(audio_path, task, speakers, word_timestamps, verbatim, context_terms, target_language, instructions, max_new_tokens, seed, callback, abort_fn,
                                      set_progress_status or (lambda status: None))
        except InterruptedError:
            return None

    def _encode(self, waveform, check_abort):
        """The perception of mono samples: (its outputs, the feature mask)."""
        features, mask = self.features(waveform, runtime.execution_device(self.perception))
        with full_fp32():
            audio = self.perception(features, mask, check_abort=check_abort)
        check_abort()
        return audio, mask

    def _respond(self, audio, instruction, max_new_tokens, seed, callback, abort_fn, expected_tokens):
        """The model's answer to an instruction about encoded audio: (decoded text with its special tags, truncated)."""
        prompt_ids, audio_positions = self.prompt.encode(instruction, audio["num_audio_tokens"])
        embed_ids = torch.tensor(self.prompt.embedding_ids(prompt_ids), dtype=torch.long, device=runtime.execution_device(self.transformer))
        prompt_embeds = self.transformer.model.embed_tokens(embed_ids)
        prompt_embeds[audio_positions] = audio["audio_embeds"].to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        embed_ids = None
        generated, truncated = self._decode(prompt_ids, prompt_embeds, max_new_tokens, seed, callback, abort_fn, expected_tokens)
        return self.prompt.decode(generated), truncated

    def _translate_segments(self, waveform, segments, context_terms, target_language, instructions, seed, callback, abort_fn, check_abort):
        """Timed translation turns: each segment (start, end, speaker) of the recording translated from its own stretch of audio."""
        from .prompting import build_instruction, cleanup_generation_loops
        instruction = build_instruction("translate", False, True, context_terms, target_language, instructions)
        rate, turns = self.features.sampling_rate, []
        for index, (start, end, speaker) in enumerate(segments):
            piece = waveform[max(0, int((start - SEGMENT_PADDING) * rate)):int((end + SEGMENT_PADDING) * rate)]
            audio, _ = self._encode(piece, check_abort)
            raw_text, _ = self._respond(audio, instruction, SEGMENT_MAX_TOKENS, seed, None, abort_fn, SEGMENT_MAX_TOKENS)
            text = cleanup_generation_loops(self.prompt.strip_special(raw_text)).strip()
            if text:
                turns.append({"speaker": speaker, "text": text, "start": start, "end": end})
            if callback is not None:
                callback(step_idx=index, override_num_inference_steps=len(segments), denoising_extra=f"Translating Sentence {index + 1}/{len(segments)}", progress_unit="sentences")
        return turns

    def _generate(self, audio_path, task, speakers, word_timestamps, verbatim, context_terms, target_language, instructions, max_new_tokens, seed, callback, abort_fn, status):
        from .alignment import align_words, alignment_classes, speaker_turns
        from .features import load_audio
        from .prompting import build_instruction, cleanup_generation_loops, render_speakers
        check_abort = lambda: self._abort_check(abort_fn)
        # a timed translation: the recording is transcribed and aligned first, then each sentence is translated from its own audio
        timed_translation = task == "translate" and word_timestamps
        status("Encoding Audio")
        waveform = load_audio(audio_path)
        duration = waveform.shape[-1] / self.features.sampling_rate
        audio, mask = self._encode(waveform, check_abort)
        instruction = build_instruction("transcribe" if timed_translation else task, speakers, verbatim or timed_translation, context_terms, target_language, "" if timed_translation else instructions)
        status("Generating Transcript")
        expected_tokens = int(duration * EXPECTED_TOKENS_PER_SECOND) + 32
        if callback is not None:  # the start event, right before the first language-model pass
            callback(step_idx=-1, override_num_inference_steps=min(max_new_tokens, expected_tokens), denoising_extra="Generating Transcript", progress_unit="tokens")
        raw_text, truncated = self._respond(audio, instruction, max_new_tokens, seed, callback, abort_fn, expected_tokens)
        text = cleanup_generation_loops(self.prompt.strip_special(raw_text) if not speakers else raw_text)
        turns = speaker_turns(text) if speakers else [{"speaker": None, "text": text.strip(), "start": None, "end": None}]
        words = diarization = alignment_error = None
        if word_timestamps:
            status("Aligning Words")
            # only the classes of the transcript's tokens: bounded memory for recordings of an hour (the alignment is identical)
            columns = alignment_classes(text, self.perception.ctc_tokenizer)
            if columns is None:
                alignment_error = "The transcript cannot be aligned to the audio"
            else:
                with full_fp32():
                    logits = self.perception.word_timestamp_logits(audio["asr_hidden"], mask, check_abort=check_abort, columns=columns)
                alignment = align_words(text, logits, mask, audio["speaker_logits"], self.config, self.perception.ctc_tokenizer, columns=columns)
                logits = None
                words, diarization, alignment_error = alignment["words"], alignment["diarization"], alignment["error"]
                if words and speakers:
                    turns = speaker_turns(text, words)
        if timed_translation and words:
            status("Translating Sentences")
            segments = translation_segments(words, speakers)
            if callback is not None:
                callback(step_idx=-1, override_num_inference_steps=len(segments), denoising_extra="Translating Sentences", progress_unit="sentences")
            audio = mask = None
            turns = self._translate_segments(waveform, segments, context_terms, target_language, instructions, seed, callback, abort_fn, check_abort)
            text, words = None, None  # the words are the transcript's: the translation is timed by its sentences
        elif timed_translation:  # not aligned (alignment_error says why): an untimed translation of the whole recording
            raw_text, truncated = self._respond(audio, build_instruction(task, speakers, verbatim, context_terms, target_language, instructions), max_new_tokens, seed, callback, abort_fn,
                                                expected_tokens)
            text = cleanup_generation_loops(self.prompt.strip_special(raw_text))
            turns = [{"speaker": None, "text": text.strip(), "start": None, "end": None}]
        audio = mask = waveform = None
        turns = [{**turn, "speaker": _numbered(turn["speaker"])} for turn in turns]
        words = None if words is None else [{**word, "speaker": _numbered(word["speaker"])} for word in words]
        diarization = None if diarization is None else [{**segment, "speaker": _numbered(segment["speaker"])} for segment in diarization]
        if text is None:  # a timed translation: a line per sentence
            text = "\n".join(f"Speaker {turn['speaker']}: {turn['text']}" if turn["speaker"] is not None else turn["text"] for turn in turns)
        else:
            text = render_speakers(text) if speakers and not timed_translation else text.strip()
        return {"text": text, "raw_text": raw_text, "turns": turns, "words": words, "diarization": diarization,
                "alignment_error": alignment_error, "duration": duration, "truncated": truncated}
