# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Bounded single-request decoding for the prompt enhancer, independent of Deepy."""
from time import perf_counter

from tqdm import tqdm

from shared.llm_engines.lmgp.engine.block_manager import BlockManager
from shared.llm_engines.lmgp.layers.kv_formats import overlay_slots
from shared.prompt_enhancer.streaming import ThrottledStreamEmitter
from shared.utils.cancellation import check_cancelled


def generate_prompt(llm, prompt, sampling_params, *, stop_token_ids, close_think_token_id, max_thinking_tokens, stop_requested=None, stream_callback=None, decode=None):
    # decode: token ids to turn text, for a chat format whose tokens are not that text (chat_formats)
    # Match the engine's request preparation without changing its shared decode path.
    llm._refresh_int8_backend()
    llm.model_runner.ensure_runtime_ready()
    llm.reset()
    if llm.config.num_kvcache_blocks > 0 and len(llm.scheduler.block_manager.blocks) != llm.config.num_kvcache_blocks:
        llm.scheduler.block_manager = BlockManager(llm.config.num_kvcache_blocks, llm.config.kvcache_block_size, overlay_slots(llm.config))
    llm.model_runner.reset_generation_state()
    llm.model_runner.call("set_sampling_seed", sampling_params.seed)
    llm.add_request(prompt, sampling_params)
    seq = llm.scheduler.waiting[-1]
    stop_ids = set(stop_token_ids) | {llm.config.eos}
    in_thinking = close_think_token_id is not None
    # As in Deepy's action loop, stop speculative blocks at semantic boundaries
    # and check every emitted token, including alternate EOS tokens.
    seq.speculative_stop_token_ids = stop_ids | ({close_think_token_id} if in_thinking else set())
    emitter = ThrottledStreamEmitter(1 / 3)
    prefill_seconds = decode_seconds = 0.0
    decode_tokens = 0
    stop_reason = "max_tokens"
    stop_token_id = None
    output_ids = []

    def emit(force=False):
        if stream_callback is not None and (force or emitter.is_due()):
            emitter.emit(stream_callback, raw_text=decode(output_ids) if decode else llm.tokenizer.decode(output_ids, skip_special_tokens=False), token_count=seq.num_completion_tokens,
                         max_tokens=sampling_params.max_tokens, prefill_seconds=prefill_seconds,
                         tokens_per_second=decode_tokens / decode_seconds if decode_seconds else 0.0,
                         stop_reason=stop_reason if force else None, is_final=force, force=force)

    try:
        with tqdm(total=sampling_params.max_tokens, desc="Enhancing Prompt", dynamic_ncols=True, leave=False) as pbar:
            emit()
            while not seq.is_finished:
                check_cancelled()
                if callable(stop_requested) and stop_requested():
                    stop_reason = "interrupted"
                    emit(force=True)
                    raise InterruptedError("Prompt Enhancement Stopped")
                if len(seq) >= llm.config.max_model_len:
                    stop_reason = "context_limit"
                    break
                remaining = sampling_params.max_tokens - seq.num_completion_tokens
                if in_thinking:
                    # The logits processor forces </think> at the thinking limit.
                    # Reserve that one token, then fail explicitly if it was ignored.
                    remaining = min(remaining, max_thinking_tokens + 1 - seq.num_completion_tokens)
                if remaining <= 0:
                    stop_reason = "thought_budget_exhausted" if in_thinking else "max_tokens"
                    break
                seq.speculative_max_emission = remaining
                previous_count = seq.num_completion_tokens
                started = perf_counter()
                _, num_tokens = llm.step()
                elapsed = perf_counter() - started
                if num_tokens > 0:
                    prefill_seconds += elapsed
                else:
                    decode_seconds += elapsed
                    decode_tokens -= num_tokens
                emitted = seq.completion_token_ids[previous_count:]
                for token_id in emitted:
                    if token_id in stop_ids:
                        stop_reason, stop_token_id = "stop_token", token_id
                        break
                    output_ids.append(token_id)
                    if in_thinking and token_id == close_think_token_id:
                        in_thinking = False
                        seq.speculative_stop_token_ids = stop_ids
                pbar.update(seq.num_completion_tokens - previous_count)
                emit()
                if stop_token_id is not None:
                    break
            emit(force=True)
        return {"text": decode(output_ids) if decode else llm.tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "output_token_ids": list(seq.completion_token_ids), "stop_reason": stop_reason, "stop_token_id": stop_token_id}
    finally:
        # Alternate EOS and cancellation can leave a live sequence in the scheduler.
        llm.reset()
