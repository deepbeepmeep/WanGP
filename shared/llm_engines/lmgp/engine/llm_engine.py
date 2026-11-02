# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import atexit
from contextlib import contextmanager
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from shared.utils.cancellation import check_cancelled
from shared.prompt_enhancer.streaming import ThrottledStreamEmitter
from shared.kernels import int8_backend

from ..config import Config
from ..sampling_params import SamplingParams
from .sequence import Sequence
from .scheduler import Scheduler
from .block_manager import BlockManager
from ..layers.kv_formats import overlay_slots
from .model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        model_object = kwargs.get("model_object", None)
        graph_pool_handle = kwargs.get("graph_pool_handle", None)
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        runner_class = ModelRunner
        if getattr(model_object, "_block_draft", False):
            from .block_draft_runner import BlockDraftRunner
            runner_class = BlockDraftRunner
        self.model_runner = runner_class(config, model_object=model_object, graph_pool_handle=graph_pool_handle)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._int8_backend_revision = int8_backend.revision
        self._exit_registered = False
        self._closed = False
        self._exited = False
        self._atexit_callback = self.exit
        atexit.register(self._atexit_callback)
        self._exit_registered = True

    def exit(self):
        if self._exited:
            return
        self._exited = True
        runner = getattr(self, "model_runner", None)
        if runner is not None:
            try:
                runner.call("exit")
            except Exception:
                pass
            try:
                del self.model_runner
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.reset_runtime_state()
        except Exception:
            pass
        try:
            self.clear_graph_cache()
        except Exception:
            pass
        try:
            self.exit()
        except Exception:
            pass
        if self._exit_registered:
            try:
                atexit.unregister(self._atexit_callback)
            except Exception:
                pass
            self._exit_registered = False
        self._atexit_callback = None
        self.scheduler = None
        self.tokenizer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def reset_runtime_state(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return
        runner.reset_runtime_state()
        # KV cache is invalid after runtime reset/reprepare, so cached prefix block metadata
        # must be dropped as well to prevent stale-cache reuse.
        try:
            self.reset()
        except Exception:
            pass
        if self.scheduler is None:
            return
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.scheduler.block_manager = BlockManager(
            self.config.num_kvcache_blocks,
            self.config.kvcache_block_size,
            overlay_slots(self.config),  # the exact overlays of a mixed KV layout, as the scheduler's own
        )

    def clear_graph_cache(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return
        runner.clear_graph_cache()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        unconditional_prompt: str | list[int] | None = None,
        prompt_embeds=None,
        prompt_position_ids=None,
        position_offset: int = 0,
        unconditional_prompt_embeds=None,
        unconditional_prompt_position_ids=None,
        unconditional_position_offset: int = 0,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # For CFG: if cfg_scale > 1.0, create both conditional and unconditional sequences
        if sampling_params.cfg_scale > 1.0:
            if unconditional_prompt is None:
                # Try to construct unconditional prompt by replacing user input with "NO USER INPUT"
                # This is a fallback - ideally users should provide unconditional_prompt
                if isinstance(prompt, list):
                    # For now, just use the same prompt (user should provide unconditional_prompt)
                    # TODO: Implement automatic "NO USER INPUT" replacement if possible
                    unconditional_prompt = prompt
                else:
                    unconditional_prompt = prompt
            if isinstance(unconditional_prompt, str):
                unconditional_prompt = self.tokenizer.encode(unconditional_prompt)
            # Create unconditional sequence first (so we can reference it from conditional)
            uncond_seq = Sequence(
                unconditional_prompt,
                sampling_params,
                is_unconditional=True,
                prompt_embeds=unconditional_prompt_embeds,
                prompt_position_ids=unconditional_prompt_position_ids,
                position_offset=unconditional_position_offset,
            )
            # Create conditional sequence with reference to unconditional
            cond_seq = Sequence(
                prompt,
                sampling_params,
                is_unconditional=False,
                conditional_seq=uncond_seq,
                prompt_embeds=prompt_embeds,
                prompt_position_ids=prompt_position_ids,
                position_offset=position_offset,
            )
            uncond_seq.paired_seq = cond_seq  # Link them bidirectionally
            # Add both sequences to scheduler
            self.scheduler.add(cond_seq)
            self.scheduler.add(uncond_seq)
        else:
            seq = Sequence(
                prompt,
                sampling_params,
                prompt_embeds=prompt_embeds,
                prompt_position_ids=prompt_position_ids,
                position_offset=position_offset,
            )
            self.scheduler.add(seq)

    def step(self):
        check_cancelled()
        if self.config.kv_cache_initial_tokens:
            blocks = self.scheduler.block_manager
            needed = sum(max(0, seq.num_blocks - len(seq.block_table)) for seq in self.scheduler.running)
            if needed > len(blocks.free_block_ids):
                capacity = self.model_runner.call("grow_kv_cache", len(blocks.blocks) + needed - len(blocks.free_block_ids))
                blocks.grow(capacity)
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        emitted_tokens = self.scheduler.postprocess(seqs, token_ids)
        # Only output conditional sequences (unconditional sequences are just for CFG computation)
        output_seqs = [seq for seq in seqs if seq.is_finished and (seq.cfg_scale <= 1.0 or not seq.is_unconditional)]
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in output_seqs]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -emitted_tokens
        return outputs, num_tokens

    def is_finished(self):
        if self.scheduler is None:
            return True
        return self.scheduler.is_finished()

    def reset(self):
        """
        Reset the scheduler state and release all allocated blocks.
        This should be called when an exception occurs during generation to prevent
        KV cache block leaks that can cause 'deque index out of range' errors.
        """
        # Deallocate all running sequences
        if self.scheduler is None:
            return
        while self.scheduler.running:
            seq = self.scheduler.running.popleft()
            if seq.block_table:  # Only deallocate if blocks are allocated
                self.scheduler.block_manager.deallocate(seq)
        
        # Deallocate all waiting sequences (they might have blocks from preemption)
        while self.scheduler.waiting:
            seq = self.scheduler.waiting.popleft()
            if seq.block_table:
                self.scheduler.block_manager.deallocate(seq)

    def can_aside(self, tokens: int) -> bool:
        """Whether a side request of this many tokens (prompt and generation) fits beside the live sequences (aside): within the context
        length, in the free pages of the cache as it is."""
        # (a runtime about to be prepared again, or an INT8 kernel change, would rebuild the cache under the live sequences)
        if self.scheduler is None or tokens > self.config.max_model_len or not self.model_runner.runtime_current() or self._int8_backend_revision != int8_backend.revision:
            return False
        blocks = self.scheduler.block_manager
        return len(blocks.blocks) == self.config.num_kvcache_blocks and len(blocks.free_block_ids) >= (tokens + blocks.block_size - 1) // blocks.block_size + 1

    @contextmanager
    def aside(self):
        """Runs requests beside the live sequences (an assistant's conversation), which keep their pages: they leave the scheduler meanwhile,
        their per-sequence state (recurrent states, drafter caches, the sampler) is saved and written back afterwards, and an overlay slot
        they hold is lent with its rows saved when none is free. Requests made inside may reset the scheduler and the generation state as
        they do alone. Check can_aside() first."""
        scheduler, blocks = self.scheduler, self.scheduler.block_manager
        running, waiting = list(scheduler.running), list(scheduler.waiting)
        lending = [] if blocks.free_overlay_slots else [seq for seq in (*running, *waiting) if seq.overlay_slot >= 0][:1]
        lent = [seq.overlay_slot for seq in lending]
        saved = self.model_runner.call("save_sequence_state", lending)
        scheduler.running.clear()
        scheduler.waiting.clear()
        blocks.free_overlay_slots.extend(lent)  # its rows are in the saved state
        try:
            yield
        finally:
            self.reset()  # the side requests' pages and overlay slots
            for slot in lent:
                blocks.free_overlay_slots.remove(slot)
            self.model_runner.call("restore_sequence_state", saved)
            scheduler.running.extend(running)
            scheduler.waiting.extend(waiting)

    def _refresh_int8_backend(self):
        # Once per request, never per token. Captured CUDA graphs bypass Python
        # dispatch; cached KV values also belong to the previous kernel's math.
        # Keep the loaded model and weights, rebuild only runtime allocations.
        if self._int8_backend_revision != int8_backend.revision:
            self.reset_runtime_state()
            self._int8_backend_revision = int8_backend.revision

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        unconditional_prompts: list[str] | list[list[int]] | None = None,
        stop_requested=None,
        stream_callback=None,
        stream_interval_seconds=1 / 3,
    ) -> list[str]:
        if self.scheduler is None:
            raise RuntimeError("LLM engine is closed.")
        # Ensure model runtime/KV cache are prepared, and sync scheduler blocks.
        self._refresh_int8_backend()
        self.model_runner.ensure_runtime_ready()
        if (self.config.num_kvcache_blocks > 0 and
                len(self.scheduler.block_manager.blocks) != self.config.num_kvcache_blocks):
            self.scheduler.block_manager = BlockManager(
                self.config.num_kvcache_blocks,
                self.config.kvcache_block_size,
                overlay_slots(self.config),
            )
        # Clean up any residual state from previous interrupted generations
        # This prevents 'deque index out of range' errors from accumulated block leaks
        if not self.is_finished():
            self.reset()
        self.model_runner.reset_generation_state()
        
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # Seed once per request-batch; keeps deterministic decode without per-step overhead.
        seed_to_apply = None
        for sp in sampling_params:
            seed_val = getattr(sp, "seed", None)
            if seed_val is not None:
                try:
                    seed_to_apply = int(seed_val)
                    break
                except Exception:
                    seed_to_apply = None
        self.model_runner.call("set_sampling_seed", seed_to_apply)
        if unconditional_prompts is None:
            unconditional_prompts = [None] * len(prompts)
        for prompt, sp, uncond_prompt in zip(prompts, sampling_params, unconditional_prompts):
            self.add_request(prompt, sp, uncond_prompt)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        sequences = [seq for seq in self.scheduler.waiting if not seq.is_unconditional] if stream_callback is not None else []
        emitter = ThrottledStreamEmitter(stream_interval_seconds)
        prefill_seconds = decode_seconds = 0.0
        decode_tokens = 0

        def emit_progress(stop_reason=None, force=False):
            if stream_callback is not None and (force or emitter.is_due()):
                emitter.emit(stream_callback, raw_text="".join(self.tokenizer.decode(seq.completion_token_ids) for seq in sequences), token_count=sum(seq.num_completion_tokens for seq in sequences), max_tokens=sum(sp.max_tokens for sp in sampling_params), prefill_seconds=prefill_seconds, tokens_per_second=decode_tokens / decode_seconds if decode_seconds else 0.0, stop_reason=stop_reason, is_final=force, force=force)

        try:
            emit_progress()
            while not self.is_finished():
                if callable(stop_requested) and stop_requested():
                    emit_progress("interrupted", force=True)
                    raise InterruptedError("Prompt Enhancement Stopped")
                t = perf_counter()
                output, num_tokens = self.step()
                elapsed = perf_counter() - t
                if num_tokens > 0:
                    prefill_seconds += elapsed
                else:
                    decode_seconds += elapsed
                    decode_tokens -= num_tokens
                emit_progress()
                if use_tqdm:
                    if num_tokens > 0:
                        prefill_throughput = num_tokens / (perf_counter() - t)
                    else:
                        decode_throughput = -num_tokens / (perf_counter() - t)
                    pbar.set_postfix({
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    })
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    if use_tqdm:
                        pbar.update(1)
            emit_progress("completed", force=True)
        except Exception:
            # Clean up on exception to prevent block leaks
            self.reset()
            raise
        finally:
            if use_tqdm:
                pbar.close()
        
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    def generate_embedded(
        self,
        prompts: list[list[int]],
        prompt_embeds: list,
        prompt_position_ids: list | None,
        sampling_params: SamplingParams | list[SamplingParams],
        position_offsets: list[int] | None = None,
        use_tqdm: bool = True,
        stream_callback=None,
    ):
        if self.scheduler is None:
            raise RuntimeError("LLM engine is closed.")
        self._refresh_int8_backend()
        self.model_runner.ensure_runtime_ready()
        if (
            self.config.num_kvcache_blocks > 0
            and len(self.scheduler.block_manager.blocks) != self.config.num_kvcache_blocks
        ):
            self.scheduler.block_manager = BlockManager(
                self.config.num_kvcache_blocks,
                self.config.kvcache_block_size,
                overlay_slots(self.config),
            )
        if not self.is_finished():
            self.reset()
        self.model_runner.reset_generation_state()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if prompt_position_ids is None:
            prompt_position_ids = [None] * len(prompts)
        if position_offsets is None:
            position_offsets = [0] * len(prompts)
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        seed_to_apply = None
        for sp in sampling_params:
            seed_val = getattr(sp, "seed", None)
            if seed_val is not None:
                try:
                    seed_to_apply = int(seed_val)
                    break
                except Exception:
                    seed_to_apply = None
        self.model_runner.call("set_sampling_seed", seed_to_apply)
        for prompt, embeds, pos_ids, pos_offset, sp in zip(prompts, prompt_embeds, prompt_position_ids, position_offsets, sampling_params):
            self.add_request(
                prompt,
                sp,
                prompt_embeds=embeds,
                prompt_position_ids=pos_ids,
                position_offset=pos_offset,
            )
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        sequences = list(self.scheduler.waiting) if stream_callback is not None else []
        emitter = ThrottledStreamEmitter(1 / 3)
        prefill_seconds = decode_seconds = 0.0
        decode_tokens = 0

        def emit_progress(force=False):
            if stream_callback is not None and (force or emitter.is_due()):
                emitter.emit(stream_callback, raw_text="".join(self.tokenizer.decode(seq.completion_token_ids) for seq in sequences), token_count=sum(seq.num_completion_tokens for seq in sequences), max_tokens=sum(sp.max_tokens for sp in sampling_params), prefill_seconds=prefill_seconds, tokens_per_second=decode_tokens / decode_seconds if decode_seconds else 0.0, stop_reason="completed" if force else None, is_final=force, force=force)

        try:
            emit_progress()
            while not self.is_finished():
                t = perf_counter()
                output, num_tokens = self.step()
                elapsed = perf_counter() - t
                if num_tokens > 0:
                    prefill_seconds += elapsed
                else:
                    decode_seconds += elapsed
                    decode_tokens -= num_tokens
                emit_progress()
                if use_tqdm:
                    if num_tokens > 0:
                        prefill_throughput = num_tokens / (perf_counter() - t)
                    else:
                        decode_throughput = -num_tokens / (perf_counter() - t)
                    pbar.set_postfix({
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    })
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    if use_tqdm:
                        pbar.update(1)
            emit_progress(force=True)
        except Exception:
            self.reset()
            raise
        finally:
            if use_tqdm:
                pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
