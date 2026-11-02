# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import gc
import time
import math
import torch

from mmgp import transfer
from ..config import Config
from .sequence import Sequence
from ..layers import kv_formats as kvf
from .speculative_sampling import bounded_support, draft_probabilities
from ..layers.sampler import Sampler, _REPETITION_INCREMENT_LIMIT, apply_min_p_mask_, apply_sparse_repetition_penalty_
from ..utils.context import set_context, get_context, reset_context



def _state_tensors(value, depth=0):
    """The plain tensors held by a module attribute: a cache or state buffer, in containers or in WanGP's own state objects (a speculative
    replay, a static cache); weights (parameters, quantized subclasses), modules and foreign objects (configs) are not state."""
    if type(value) is torch.Tensor:
        return [value]
    if depth > 2:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, dict):
        items = value.values()
    elif type(value).__module__.startswith("shared.") and hasattr(value, "__dict__") and not isinstance(value, torch.nn.Module):
        items = vars(value).values()
    else:
        return []
    return [tensor for item in items for tensor in _state_tensors(item, depth + 1)]


def _kv_page_tensors(runner) -> list[tuple[str, torch.Tensor, int]]:
    """(attribute name, tensor, page axis) of the runner's paged cache tensors."""
    layout = getattr(runner, "kv_layout", None)
    if layout is not None and layout.mixed and getattr(runner, "kv_views", None) is not None:
        tensors = [("kv_cache", runner.kv_views[0], 1), ("kv_cache", runner.kv_views[1], 1), ("kv_cache_scales", runner.kv_cache_scales, 2)]
        if getattr(runner, "kv_cache_mins", None) is not None:
            tensors.append(("kv_cache_mins", runner.kv_cache_mins, 2))
        return tensors
    tensors = [("kv_cache", runner.kv_cache, 2)]
    if runner.config.kv_cache_int8:
        tensors.append(("kv_cache_scales", runner.kv_cache_scales, 2))
    return tensors


_OUTSIDE_SPANS = (False, False, False)  # (code fence, inline backticks, double quotes)


def _advance_spans(span: tuple[bool, bool, bool], text: str) -> tuple[bool, bool, bool]:
    """The span state after a token's text: ``` toggles a code fence, a single backtick inline code, a double quote a quoted string (outside
    code); a newline closes inline code and quotes, which never span lines."""
    fence, inline, quote = span
    index = 0
    while index < len(text):
        character = text[index]
        if character == "`":
            run = len(text) - index - len(text[index:].lstrip("`"))
            if run >= 3:
                fence, inline = not fence, False
            elif not fence and run % 2:
                inline = not inline
            index += run
            continue
        if character == '"' and not fence and not inline:
            quote = not quote
        elif character == "\n":
            inline = quote = False
        index += 1
    return fence, inline, quote


class ModelRunner:

    _MAX_SPECULATIVE_DRAFT_TOKENS = 8
    mmgp = False  # set from the model in __init__: MMGP manages the weights (dynamic preload, streamed blocks), with or without its graphs
    mmgp_graphs = False  # set from the model in __init__ (MMGP owns the weights and the target model's decoding graphs)

    def __init__(self, config: Config, model_object=None, graph_pool_handle=None):
        # Enable capturing scalar outputs to avoid graph breaks from Tensor.item() calls
        torch._dynamo.config.capture_scalar_outputs = True
        
        self.config = config
        hf_config = config.hf_config
        self.use_triton_sampling = not bool(getattr(hf_config, "_prompt_enhancer_safe_legacy", False))
        self.block_size = config.kvcache_block_size
        self.kv_layout = kvf.layout_of(config)  # None: BF16 cache; otherwise the body formats and exact overlays
        self.enforce_eager = config.enforce_eager
        self.world_size, self.rank = 1, 0  # one process on one GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        # Use dtype instead of deprecated torch_dtype
        config_dtype = getattr(hf_config, 'dtype', getattr(hf_config, 'torch_dtype', None))

        # Validate and convert config_dtype to a valid torch floating-point dtype
        # Default to bfloat16 for CUDA (required for Flash Attention 2)
        if config_dtype is None:
            config_dtype = torch.bfloat16
        elif isinstance(config_dtype, str):
            # Convert string dtype to torch dtype
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float64': torch.float64,
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float64': torch.float64,
            }
            config_dtype = dtype_map.get(config_dtype.lower(), torch.bfloat16)
        elif not isinstance(config_dtype, torch.dtype) or not config_dtype.is_floating_point:
            # If not a valid floating-point torch dtype, default to bfloat16
            config_dtype = torch.bfloat16

        self.dtype = config_dtype  # Save for later use
        self._runtime_ready = False
        self._runtime_initializing = False
        self._graph_cache = {}
        self._graph_cache_order = []
        self._logits_bias_cache = {}
        self._repetition_token_cache = {}
        self._speculative_sampling_graphs = {}
        self._sampling_generator = None
        self._runtime_signature = None
        self._model_storage_signature = None
        self._graph_pool_seed = graph_pool_handle
        self._guard_counts = {}
        self._guard_seen_details = set()
        self._speculative_drafts = {}
        self._speculative_pending = {}
        self._speculative_commit_destinations = []
        self._speculative_commit_sources = {}
        self._speculative_replay_modules = []
        self.speculative_commit_graphs = {}
        self._mtp_sample_params = None
        self._mtp_profile_enabled = False
        self._mtp_profile_epoch = 0
        self._mtp_profile_passes = 0
        self._mtp_profile_event_sets = None
        self._mtp_profile_pending = []
        self._mtp_profile_samples = []
        torch.set_default_dtype(config_dtype)
        if model_object is None:
            raise RuntimeError(
                "The LM engine requires a preloaded MMGP model object. "
                "Pass model_object=... when creating LLM."
            )
        self.model = model_object
        # MMGP owns the weights (dynamic preload, streamed blocks): the target model's decoding graphs are MMGP graph stages, the engine's
        # own graphs read only the base block (always in VRAM while the model is loaded) or the engine state.
        self.mmgp_graphs = bool(getattr(model_object, "_lm_engine_mmgp_graphs", False))
        self.mmgp = self.mmgp_graphs or bool(getattr(model_object, "_lm_engine_mmgp", False))
        if self.mmgp_graphs:  # only MMGP moves its weights: the weights key of a whole-model graph is checked in constant time
            model_object._mm_graph_placement_key = True
        self._max_speculative_draft_tokens = min(self._MAX_SPECULATIVE_DRAFT_TOKENS, max(1, int(getattr(self.model, "_prompt_enhancer_speculative_tokens", 2))))
        self.speculative_stats = self._new_speculative_stats()
        self.sampler = Sampler()
        
        # Pre-allocate buffers for sampling (optimization: avoid repeated tensor creation)
        # Must be called before model execution paths that use these buffers.
        self._allocate_sample_buffers()
        
        torch.set_default_device(None if tuple(map(int, torch.__version__.split("+", 1)[0].split(".")[:2])) >= (2, 8) else "cpu")
        torch.set_default_dtype(default_dtype)


    def ensure_runtime_ready(self):
        if self.mmgp_graphs:
            from mmgp import runtime
            runtime.ensure_loaded(self.model)  # the base block's addresses are those the engine's graphs capture
        elif not self.enforce_eager:
            from mmgp import runtime
            if runtime.requires_layer_offload(self.model):
                self.enforce_eager = True
                self._note_guard("layer_offload", "Whole-model CUDA graphs disabled while MMGP can move layers; decoder kernels are unchanged")
        if self._runtime_ready:
            # In eager mode MMGP may move parameter storage between CPU/GPU after prefill.
            # That changes data_ptr() without invalidating the live KV cache or sequence state.
            # Resetting here drops the prompt context and makes legacy decode drift off-topic.
            if self.enforce_eager:
                return
            current_sig = self._get_graph_capture_signature()
            if current_sig == self._runtime_signature:
                return
            self._note_guard("runtime_reprepare_signature_change")
            self.reset_runtime_state()
        if self.model is None:
            raise RuntimeError("LLM model object is not available.")
        self._tie_word_embeddings_if_needed()
        self._runtime_initializing = True
        try:
            self.allocate_kv_cache()
            self._prepare_model_sequence_state()
            if not self.enforce_eager:
                self.capture_cudagraph()
            self._register_graph_state()
            self._runtime_signature = self._get_graph_capture_signature()
            self._runtime_ready = True
        except BaseException:
            self.reset_runtime_state()
            raise
        finally:
            self._runtime_initializing = False

    def _register_graph_state(self):
        """MMGP mode: the CUDA tensors the target model's graphs read besides weights and arguments (KV pages, recurrent and speculative
        states, the context buffers of the decoding passes), read in place at their fixed addresses until the runtime is reset."""
        if not self.mmgp_graphs or self.enforce_eager:
            return
        from mmgp import cuda_graphs
        tensors = [getattr(self, name, None) for name in ("kv_cache", "kv_cache_scales", "kv_cache_mins", "kv_overlay", "kv_overlay_scales")]
        for graph_vars in (getattr(self, "graph_vars", None) or {}, *(getattr(self, "speculative_graph_vars", None) or {}).values()):
            tensors.extend(graph_vars.values())
        for module in self.model.modules():
            for name, value in vars(module).items():
                if name not in ("_parameters", "_buffers", "_modules", "_hf_hook") and not name.startswith("_mm_"):  # MMGP's own records hold the weights
                    tensors.extend(_state_tensors(value))
        cuda_graphs.register_state(self.model, "lmgp", tensors)

    def _run_target_stage(self, input_ids: torch.Tensor, positions: torch.Tensor, outputs: torch.Tensor, draft_features: torch.Tensor | None = None) -> None:
        """MMGP mode: a decoding pass of the target model into outputs, one MMGP graph while its weights are all in VRAM; otherwise its
        tower runs block by block, each run of blocks kept in VRAM replayed as one graph, the others streamed."""
        from mmgp import cuda_graphs, runtime
        with cuda_graphs.scope(gated=False):
            cuda_graphs.run(self.model, input_ids, positions, outputs, draft_features, method="lm_engine_graph_forward", enabled=runtime.weights_resident(self.model))
        if draft_features is not None:
            self.model._draft_features = draft_features

    def _capture_target_stages(self, slot_mapping: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor, overlay_mapping: torch.Tensor, overlay_base: torch.Tensor,
                               ring_start: torch.Tensor) -> None:
        """MMGP mode, the model in VRAM: the decoding passes run once on dummy pages (as the engine's own graphs are captured), which
        records their MMGP graphs before the first request instead of during its decoding."""
        from mmgp import runtime
        for rows, graph_vars in [(self.graph_bs[-1], self.graph_vars), *self.speculative_graph_vars.items()]:
            self._draft_feature_buffer(graph_vars, rows, register=False)  # all made before the state is registered once
        self._register_graph_state()
        if not runtime.weights_resident(self.model):
            return  # the runs of resident blocks change with the dynamic preload: their graphs are recorded as they run
        graph_vars = self.graph_vars
        for bs in reversed(self.graph_bs):
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                        overlay_mapping=overlay_mapping[:bs], overlay_base=overlay_base[:bs], ring_start=ring_start[:bs])
            self._run_target_stage(graph_vars["input_ids"][:bs], graph_vars["positions"][:bs], graph_vars["outputs"][:bs], self._draft_feature_buffer(graph_vars, bs))
            reset_context()
        for verify_length, graph_vars in sorted(self.speculative_graph_vars.items(), reverse=True):
            set_context(True, graph_vars["cu_seqlens_q"], graph_vars["cu_seqlens_k"], verify_length, self.config.max_model_len, graph_vars["slot_mapping"], None, graph_vars["block_tables"], has_previous_state=True, speculative_verify=True,
                        overlay_mapping=graph_vars["overlay_mapping"], overlay_base=graph_vars["overlay_base"], ring_start=graph_vars["ring_start"])
            self._run_target_stage(graph_vars["input_ids"], graph_vars["positions"], graph_vars["outputs"], self._draft_feature_buffer(graph_vars, verify_length))
            reset_context()
        torch.cuda.synchronize()

    def _draft_feature_buffer(self, graph_vars: dict, rows: int, register: bool = True) -> torch.Tensor | None:
        if not getattr(self.model, "_block_draft", False):
            return None
        buffer = graph_vars.get("draft_features")
        if buffer is None:
            width = len(self.model.mtp.target_layer_ids) * self.config.hf_config.hidden_size
            buffer = graph_vars["draft_features"] = torch.zeros((1, rows, width), dtype=self.dtype, device=self._get_runtime_device())
            if register:
                self._register_graph_state()
        return buffer

    def reset_generation_state(self):
        if self.model is None:
            return
        try:
            for module in self.model.modules():
                reset_sequence_state = getattr(module, "reset_sequence_state", None)
                if callable(reset_sequence_state):
                    reset_sequence_state()
                    continue
                if hasattr(module, "conv_state"):
                    module.conv_state = None
                if hasattr(module, "recurrent_state"):
                    module.recurrent_state = None
        except Exception:
            pass
        self._logits_bias_cache.clear()
        self._repetition_token_cache.clear()
        self._speculative_drafts.clear()
        self._speculative_pending.clear()
        self._mtp_sample_params = None
        self.speculative_stats = self._new_speculative_stats()
        reset_context()

    _REQUEST_STATE = ("_logits_bias_cache", "_repetition_token_cache", "_speculative_drafts", "_speculative_pending", "_mtp_sample_params", "speculative_stats", "_sampling_generator")

    def runtime_current(self) -> bool:
        """The runtime is prepared and ensure_runtime_ready() would keep it (its KV cache and the sequences in it stay valid)."""
        return self._runtime_ready and (self.enforce_eager or self._get_graph_capture_signature() == self._runtime_signature)

    @torch.inference_mode()
    def save_sequence_state(self, lending=()) -> list:
        """The per-sequence state of the live sequences, which a side request (LLMEngine.aside) resets or overwrites: what
        reset_generation_state resets (the snapshot_sequence_state of each module with a reset_sequence_state, the conv/recurrent states
        the others drop), the overlay rows of the sequences lending their slot, and the runner's per-request caches and sampler. Pages,
        weights and per-step scratch (speculative snapshots, metadata tables) are not saved."""
        modules, attributes = [], []
        for module in self.model.modules():
            if callable(getattr(module, "reset_sequence_state", None)):
                modules.append((module, module.snapshot_sequence_state()))
            else:
                attributes.extend((module, name, getattr(module, name)) for name in ("conv_state", "recurrent_state") if hasattr(module, name))
        overlays = [(seq, self.snapshot_overlay(seq)) for seq in lending]
        generator = self._sampling_generator
        requests = {name: getattr(self, name) for name in self._REQUEST_STATE}
        requests = {name: (value.copy() if isinstance(value, dict) else value) for name, value in requests.items()}
        return [modules, attributes, overlays, None if generator is None else generator.get_state(), requests]

    @torch.inference_mode()
    def restore_sequence_state(self, saved: list) -> None:
        """Writes back what save_sequence_state saved, into the same tensors (the addresses that graphs read), then registers the graph
        state again."""
        modules, attributes, overlays, generator_state, requests = saved
        for module, snapshot in modules:
            module.restore_sequence_state(snapshot)
        for module, name, value in attributes:
            setattr(module, name, value)
        for seq, overlay in overlays:
            self.restore_overlay(seq, overlay)
        for name, value in requests.items():
            setattr(self, name, value)
        if generator_state is not None:
            self._sampling_generator.set_state(generator_state)
        self._register_graph_state()

    def _new_speculative_stats(self) -> dict:
        return {
            "drafted": 0,
            "accepted": 0,
            "target_passes": 0,
            "emitted_tokens": 0,
            "drafted_by_position": [0] * self._max_speculative_draft_tokens,
            "accepted_by_position": [0] * self._max_speculative_draft_tokens,
        }

    def speculative_telemetry(self, seq_id: int, sequence_tokens: int) -> dict:
        stats = self.speculative_stats
        mtp_cache_tokens = self.model.mtp.get_cache_length()
        return {
            "drafted": int(stats["drafted"]),
            "accepted": int(stats["accepted"]),
            "target_passes": int(stats["target_passes"]),
            "emitted_tokens": int(stats["emitted_tokens"]),
            "drafted_by_position": [int(value) for value in stats["drafted_by_position"]],
            "accepted_by_position": [int(value) for value in stats["accepted_by_position"]],
            "sequence_tokens": int(sequence_tokens),
            "mtp_cache_tokens": int(mtp_cache_tokens),
            "sync_delta": int(mtp_cache_tokens) - max(0, int(sequence_tokens) - 1),
            "pending": seq_id in self._speculative_pending,
            "draft": seq_id in self._speculative_drafts,
        }

    def set_mtp_stage_profile_enabled(self, enabled: bool) -> None:
        if self._mtp_profile_enabled:
            self._collect_mtp_stage_profiles()
        self._mtp_profile_enabled = bool(enabled)
        self._mtp_profile_epoch += 1
        self._mtp_profile_passes = 0
        self._mtp_profile_samples.clear()
        if self._mtp_profile_enabled and self._mtp_profile_event_sets is None:
            self._mtp_profile_event_sets = [
                {
                    "busy": False,
                    "events": [torch.cuda.Event(enable_timing=True) for _ in range(11)],
                    "distribution_counts": torch.empty((5, 2), dtype=torch.int64, device=self._get_runtime_device()),
                    "distribution_masses": torch.empty(5, dtype=torch.float32, device=self._get_runtime_device()),
                    "cpu_distribution_counts": torch.empty((5, 2), dtype=torch.int64, pin_memory=torch.cuda.is_available()),
                    "cpu_distribution_masses": torch.empty(5, dtype=torch.float32, pin_memory=torch.cuda.is_available()),
                }
                for _ in range(2)
            ]

    def _collect_mtp_stage_profiles(self) -> None:
        for profile in tuple(self._mtp_profile_pending):
            events = profile["slot"]["events"]
            if not events[-1].query():
                continue
            profile["gpu_ms"] = {name: events[index].elapsed_time(events[index + 1]) for index, name in enumerate(("draft", "verify_setup", "verify", "output", "pre_sampling_idle", "sampling", "commit", "truncate", "mtp_advance"))}
            for index, distribution in enumerate(profile["distributions"]):
                distribution["min_p_survivors"] = int(profile["slot"]["cpu_distribution_counts"][index, 0])
                distribution["top_p_survivors"] = int(profile["slot"]["cpu_distribution_counts"][index, 1])
                distribution["excluded_mass"] = float(profile["slot"]["cpu_distribution_masses"][index])
            profile["slot"]["busy"] = False
            del profile["slot"]
            self._mtp_profile_samples.append(profile)
            self._mtp_profile_pending.remove(profile)

    def _start_mtp_stage_profile(self) -> dict | None:
        if not self._mtp_profile_enabled:
            return None
        self._collect_mtp_stage_profiles()
        self._mtp_profile_passes += 1
        if self._mtp_profile_passes % 64:
            return None
        slot = next((candidate for candidate in self._mtp_profile_event_sets if not candidate["busy"]), None)
        if slot is None:
            return None
        slot["busy"] = True
        slot["events"][0].record(torch.cuda.current_stream())
        return {"slot": slot, "epoch": self._mtp_profile_epoch, "cpu_ms": {}, "distributions": []}

    @staticmethod
    def _mark_mtp_stage_profile(profile: dict | None, event_index: int, stage: str, started_at: float) -> None:
        if profile is None:
            return
        profile["cpu_ms"][stage] = 1000.0 * (time.perf_counter() - started_at)
        profile["slot"]["events"][event_index].record(torch.cuda.current_stream())

    def _finish_mtp_stage_profile(self, profile: dict | None, accepted_count: int, emitted_count: int, commit_start: int, context_tokens: int) -> None:
        if profile is None:
            return
        profile.update(accepted_count=int(accepted_count), emitted_count=int(emitted_count), commit_start=int(commit_start), context_tokens=int(context_tokens))
        slot = profile["slot"]
        distribution_count = len(profile["distributions"])
        slot["cpu_distribution_counts"][:distribution_count].copy_(slot["distribution_counts"][:distribution_count], non_blocking=True)
        slot["cpu_distribution_masses"][:distribution_count].copy_(slot["distribution_masses"][:distribution_count], non_blocking=True)
        slot["events"][-1].record(torch.cuda.current_stream())
        self._mtp_profile_pending.append(profile)

    @staticmethod
    def _cancel_mtp_stage_profile(profile: dict | None) -> None:
        if profile is not None:
            profile["slot"]["busy"] = False

    def mtp_stage_profile_samples(self) -> list[dict]:
        self._collect_mtp_stage_profiles()
        return [sample for sample in self._mtp_profile_samples if sample["epoch"] == self._mtp_profile_epoch]

    def _prepare_model_sequence_state(self):
        if self.model is None:
            return
        runtime_device = self._get_runtime_device()
        if self.model.__class__.__name__ == "Qwen3_5ForCausalLM" and self.use_triton_sampling:
            from ..models.qwen3_5 import configure_qwen35_fla_prefill_autotune
            configure_qwen35_fla_prefill_autotune(runtime_device)
        for module in self.model.modules():
            prepare = getattr(module, "prepare_sequence_state", None)
            if callable(prepare):
                prepare(self.config.max_num_seqs, runtime_device, self.dtype)
        mtp = getattr(self.model, "mtp", None)
        if mtp is not None and bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            mtp.prepare_cache(self.config.max_model_len, self._get_runtime_device(), self.dtype)
            max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
            for module in self._get_kv_cache_modules():
                prepare = getattr(module, "prepare_speculative_metadata", None)
                if callable(prepare):
                    prepare(self._max_speculative_draft_tokens + 1, max_num_blocks)
            self._prepare_target_speculative_state()

    def _prepare_target_speculative_state(self) -> None:
        for module in self.model.blk:
            prepare_speculative_state = getattr(module, "prepare_speculative_state", None)
            if callable(prepare_speculative_state):
                prepare_speculative_state(self._max_speculative_draft_tokens + 1)
        stateful_modules = [module for module in self.model.blk if getattr(module, "layer_type", None) == "linear_attention"]
        # Recurrent states of replaying layers are recomputed from one saved state; the others are copied from their snapshots.
        snapshot_modules = [module for module in stateful_modules if module.speculative_replay is None]
        self._speculative_replay_modules = [module for module in stateful_modules if module.speculative_replay is not None]
        self._speculative_commit_destinations = [module.conv_state_buffer for module in stateful_modules] + [module.recurrent_state_buffer for module in snapshot_modules]
        self._speculative_commit_sources = {
            processed_tokens: [module.speculative_conv_state_buffer[processed_tokens - 1] for module in stateful_modules] + [module.speculative_recurrent_state_buffer[processed_tokens - 1] for module in snapshot_modules]
            for processed_tokens in range(1, self._max_speculative_draft_tokens + 1)
        }

    def _prepare_speculative_graph_state(self) -> None:
        # MMGP mode: registered again only when the preparation replaced a buffer (its walk over the modules is not free)
        def buffers():  # addresses and sizes: an empty placeholder assigned again changes nothing the graphs read
            return [(module.speculative_conv_state_buffer.data_ptr(), module.speculative_conv_state_buffer.numel(), module.speculative_recurrent_state_buffer.data_ptr(),
                     module.speculative_recurrent_state_buffer.numel(), id(module.speculative_replay)) for module in self.model.blk if getattr(module, "layer_type", None) == "linear_attention"]
        before = buffers() if self.mmgp_graphs else None
        self._prepare_target_speculative_state()
        if self.mmgp_graphs and buffers() != before:
            self._register_graph_state()

    def _get_tied_embeddings(self):
        if self.model is None:
            return None
        hf_config = getattr(self.config, "hf_config", None)
        if not bool(getattr(hf_config, "tie_word_embeddings", False)):
            return None
        lm_head = getattr(self.model, "lm_head", None)
        embed = getattr(self.model, "embed_tokens", None)
        if lm_head is None or embed is None:
            return None
        lm_w = getattr(lm_head, "weight", None)
        emb_w = getattr(embed, "weight", None)
        if lm_w is None or emb_w is None:
            return None
        return lm_head, embed

    def _tie_word_embeddings_if_needed(self):
        tied = self._get_tied_embeddings()
        if tied is None:
            return
        lm_head, embed = tied
        if lm_head.weight is not embed.weight:
            lm_head.register_parameter("weight", embed.weight)
        if lm_head.weight is not embed.weight:
            raise RuntimeError("Failed to retie lm_head.weight with embed_tokens.weight.")

    def reset_runtime_state(self):
        if not self._runtime_ready and not self._runtime_initializing:
            return
        # Steering/interruption may reach teardown while the last CUDA-graph replay
        # is still in flight.  Every object released below can own storage whose raw
        # address is embedded in a decode/speculative/MTP graph, so the device must
        # finish using those addresses before graph/cache/module references are
        # destroyed and their allocations become reusable.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        if self.mmgp_graphs and self.model is not None:
            from mmgp import cuda_graphs
            cuda_graphs.release_state(self.model, "lmgp")  # the model's MMGP graphs read the state released below
        self._speculative_commit_destinations.clear()
        self._speculative_commit_sources.clear()
        self._speculative_replay_modules = []
        reset_context()
        # Clear attention KV cache refs so we don't write into freed storage later.
        try:
            if self.model is not None:
                for module in self.model.modules():
                    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                        module.k_cache = module.v_cache = torch.tensor([])
                        module.k_scale = module.v_scale = torch.tensor([])
                    if hasattr(module, "k_overlay"):  # the mixed layouts' exact overlays and asymmetric minimums, views of the runner's tensors
                        module.k_overlay = module.v_overlay = module.k_overlay_scale = module.v_overlay_scale = module.k_min = module.v_min = None
                    if hasattr(module, "_q8_speculative_metadata"):
                        module._q8_speculative_metadata.clear()
                    release_sequence_state = getattr(module, "release_sequence_state", None)
                    if callable(release_sequence_state):
                        release_sequence_state()
                        continue
                    reset_sequence_state = getattr(module, "reset_sequence_state", None)
                    if callable(reset_sequence_state):
                        reset_sequence_state()
                    else:
                        if hasattr(module, "conv_state"):
                            module.conv_state = None
                        if hasattr(module, "recurrent_state"):
                            module.recurrent_state = None
        except Exception:
            pass
        # the cache and every view of it (kv_views keep the whole cache alive), the minimums and the exact overlays
        for name in ("kv_cache", "kv_cache_scales", "kv_cache_mins", "kv_views", "kv_min_views", "kv_overlay", "kv_overlay_scales"):
            if hasattr(self, name):
                delattr(self, name)
        # CUDA graphs captured against previous model/KV pointers are unsafe after runtime reset.
        # Force recapture on next prepare to avoid stale-pointer illegal memory access.
        try:
            self.clear_graph_cache()
        except Exception:
            pass
        try:
            for attr_name in ("graphs", "graph_vars", "graph_bs", "graph_pool", "speculative_graphs", "speculative_graph_vars", "speculative_commit_graphs", "mtp_graph", "mtp_graph_pool", "mtp_graph_vars", "mtp_refresh_graphs"):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
        except Exception:
            pass
        # The GGUF CUDA scratch arena is process-global, not owned by this runner.
        # Releasing it here can invalidate pointers captured by a newer live runner
        # when this stale runner is reset or finalized later.  It is a single small
        # reusable allocation and is released automatically when the process exits.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        self._logits_bias_cache.clear()
        self._repetition_token_cache.clear()
        self._speculative_drafts.clear()
        self._speculative_pending.clear()
        self._mtp_sample_params = None
        self._sampling_generator = None
        self._runtime_signature = None
        self._model_storage_signature = None
        self._runtime_ready = False
        gc.collect()

    def _get_model_storage_signature(self):
        signature = []
        # MMGP mode: the blocks of the tower move with the dynamic preload under MMGP's own graphs; the engine's graphs read the base block
        tensors = (tensor for name, tensor in (*self.model.named_parameters(), *self.model.named_buffers()) if not name.startswith(self.model._lm_engine_tower_prefix)) if self.mmgp_graphs else (*self.model.parameters(), *self.model.buffers())
        for tensor in tensors:
            getter = getattr(tensor, "get_quantized_subtensors", None)
            subtensors = getter() if callable(getter) else None
            tensors = [value for _name, value in (subtensors.items() if isinstance(subtensors, dict) else subtensors or ()) if torch.is_tensor(value)] or [tensor]
            for value in tensors:
                signature.append((value.device.type, value.device.index, int(value.data_ptr()), int(value.numel())))
        return tuple(signature)

    def invalidate_graphs_if_model_storage_changed(self) -> bool:
        if not self._runtime_ready or self.enforce_eager:
            return False
        current_signature = self._get_model_storage_signature()
        if current_signature == self._model_storage_signature:
            return False
        self._note_guard("runtime_reprepare_model_storage_change")
        self.reset_runtime_state()
        return True

    def _get_graph_capture_signature(self):
        # Note: this only samples the first parameter, so it can miss a partial address
        # reuse after an offloader (e.g. MMGP) evicts and reloads the model. Integrations
        # that interleave other models must explicitly call reset_runtime_state() (engine
        # release_runtime_allocations()) when their phase ends, like Chain-of-Zoom does.
        model_ptr = -1
        kv_ptr = -1
        try:
            first_param = next(self.model.parameters())
            if first_param.is_cuda:
                model_ptr = int(first_param.data_ptr())
        except Exception:
            pass
        try:
            if hasattr(self, "kv_cache") and torch.is_tensor(self.kv_cache) and self.kv_cache.is_cuda:
                kv_ptr = int(self.kv_cache.data_ptr())
        except Exception:
            pass
        return (model_ptr, kv_ptr, int(self.config.max_model_len), int(self.config.max_num_seqs), self._max_speculative_draft_tokens)

    def _get_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _get_runtime_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return self._get_model_device()

    def _pin_memory_enabled(self) -> bool:
        return self._get_runtime_device().type == "cuda"

    def _to_runtime_device(self, tensor: torch.Tensor) -> torch.Tensor:
        device = self._get_runtime_device()
        return tensor.to(device=device, non_blocking=device.type == "cuda")

    def _drop_graph_cache_entry(self, cache_key):
        entry = self._graph_cache.pop(cache_key, None)
        if cache_key in self._graph_cache_order:
            self._graph_cache_order.remove(cache_key)
        if entry is None:
            return
        try:
            del entry["graphs"]
            del entry["pool"]
            del entry["vars"]
            del entry["bs"]
            entry.pop("speculative_graphs", None)
            entry.pop("speculative_vars", None)
            entry.pop("speculative_commit_graphs", None)
            entry.pop("mtp_graph", None)
            entry.pop("mtp_pool", None)
            entry.pop("mtp_vars", None)
            entry.pop("mtp_refresh_graphs", None)
        except Exception:
            pass

    def clear_graph_cache(self):
        self._speculative_sampling_graphs.clear()
        self._speculative_acceptance_graphs = {}
        if self._graph_cache:
            for key in list(self._graph_cache.keys()):
                self._drop_graph_cache_entry(key)
            self._graph_cache.clear()
            self._graph_cache_order.clear()

    def _note_guard(self, name: str, detail: str | None = None):
        count = self._guard_counts.get(name, 0) + 1
        self._guard_counts[name] = count
        if detail:
            detail_key = (name, detail)
            if detail_key not in self._guard_seen_details:
                print(f"[LM Engine][guard] {name}: {detail}")
                self._guard_seen_details.add(detail_key)
            return
        if count == 1:
            print(f"[LM Engine][guard] {name}")

    def _get_logits_bias(self, seq: Sequence, logits: torch.Tensor):
        bias = getattr(seq, "logits_bias", None)
        if bias is None or not torch.is_tensor(bias):
            return None
        key = (id(bias), logits.device, logits.dtype)
        cached = self._logits_bias_cache.get(key)
        if cached is not None:
            return cached
        cached = bias.to(device=logits.device, dtype=logits.dtype)
        self._logits_bias_cache[key] = cached
        return cached

    def set_sampling_seed(self, seed: int | None):
        if seed is None:
            self._sampling_generator = None
            return
        device = self._get_runtime_device()
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            self._sampling_generator = generator
        except Exception:
            self._sampling_generator = None

    @staticmethod
    def _apply_logits_bias(logits_row: torch.Tensor, bias: torch.Tensor):
        logits_row.add_(bias)

    @staticmethod
    def _call_logits_processor(seq: Sequence, processor, logits: torch.Tensor, virtual_tokens=()):
        input_ids = None
        if getattr(processor, "_requires_input_ids", True):
            input_ids = torch.tensor([seq.token_ids + list(virtual_tokens)], dtype=torch.long, device=logits.device)
        return processor(input_ids, logits)

    def _allocate_sample_buffers(self):
        """Pre-allocate reusable buffers for sampling to avoid repeated tensor creation: on CUDA, views of one buffer of locked RAM
        (mmgp.pinned: exact size, counted with MMGP's pinned memory), not one power-of-two block of PyTorch's pinned allocator each."""
        from mmgp import pinned
        max_bs = self.config.max_num_seqs
        max_tokens = self.config.max_num_batched_tokens
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        drafts = self._max_speculative_draft_tokens + 1
        buffers = {  # name: (shape, dtype, initial value)
            # sampling
            "_cpu_temperatures": ((max_bs,), torch.float32, 0), "_cpu_cfg_scales": ((max_bs,), torch.float32, 0), "_cpu_top_ks": ((max_bs,), torch.int32, 0),
            "_cpu_top_ps": ((max_bs,), torch.float32, 0), "_cpu_min_ps": ((max_bs,), torch.float32, 0), "_cpu_repetition_penalties": ((max_bs,), torch.float32, 0),
            # decoding
            "_cpu_input_ids": ((max_bs,), torch.int64, 0), "_cpu_speculative_input_ids": ((drafts,), torch.int64, 0), "_cpu_speculative_overlay_mapping": ((drafts,), torch.int32, -1),
            "_cpu_positions": ((max_bs,), torch.int64, 0), "_cpu_slot_mapping": ((max_bs,), torch.int32, 0), "_cpu_context_lens": ((max_bs,), torch.int32, 0),
            "_cpu_overlay_mapping": ((max_bs,), torch.int32, -1), "_cpu_overlay_base": ((max_bs,), torch.int32, -1), "_cpu_ring_start": ((max_bs,), torch.int32, 1 << 30),
            # prefill (prepare_prefill)
            "_cpu_prefill_input_ids": ((max_tokens,), torch.int64, 0), "_cpu_prefill_positions": ((max_tokens,), torch.int64, 0), "_cpu_prefill_slot_mapping": ((max_tokens,), torch.int32, 0),
            "_cpu_prefill_cu_seqlens": ((max_bs + 1,), torch.int32, 0), "_cpu_prefill_cu_seqlens_k": ((max_bs + 1,), torch.int32, 0),
            # block tables (decoding and prefill), the sequences' token ids (logits processors and sampler)
            "_cpu_block_tables": ((max_bs, max_num_blocks), torch.int32, 0), "_seq_token_ids_buffer": ((max_bs, self.config.max_model_len), torch.int64, 0),
        }
        offsets, total = {}, 0
        for name, (shape, dtype, _) in buffers.items():
            offsets[name] = total = -(-total // 64) * 64
            total += math.prod(shape) * dtype.itemsize
        self._pinned_buffers = pinned.empty(total, "lm") if self._pin_memory_enabled() else None
        # refused by the reserved RAM or the driver: pageable buffers, whose copies synchronize
        storage = self._pinned_buffers if self._pinned_buffers is not None else torch.empty(total, dtype=torch.uint8, device="cpu")
        for name, (shape, dtype, value) in buffers.items():
            setattr(self, name, storage[offsets[name]:offsets[name] + math.prod(shape) * dtype.itemsize].view(dtype).view(shape).fill_(value))
        self._prefill_copied = None  # the event of the last copies from the prefill buffers: chunked prefills queue without synchronizing

    def _release_sample_buffers(self):
        buffer_names = [
            "_cpu_temperatures",
            "_cpu_cfg_scales",
            "_cpu_top_ks",
            "_cpu_top_ps",
            "_cpu_min_ps",
            "_cpu_repetition_penalties",
            "_cpu_input_ids",
            "_cpu_speculative_input_ids",
            "_cpu_speculative_overlay_mapping",
            "_cpu_positions",
            "_cpu_slot_mapping",
            "_cpu_context_lens",
            "_cpu_overlay_mapping",
            "_cpu_overlay_base",
            "_cpu_ring_start",
            "_cpu_prefill_input_ids",
            "_cpu_prefill_positions",
            "_cpu_prefill_cu_seqlens",
            "_cpu_prefill_cu_seqlens_k",
            "_cpu_prefill_slot_mapping",
            "_cpu_block_tables",
            "_seq_token_ids_buffer",
            "_prefill_copied",
        ]
        if getattr(self, "_pinned_buffers", None) is not None:
            from mmgp import pinned
            pinned.retire(self._pinned_buffers)  # unlocked at MMGP's next quiet point, once no copy reads it
            self._pinned_buffers = None
        for name in buffer_names:
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except Exception:
                    pass

    def exit(self):
        try:
            self.reset_runtime_state()
        except Exception:
            pass
        self._release_sample_buffers()
        self._logits_bias_cache.clear()
        self._repetition_token_cache.clear()
        self._guard_counts.clear()
        self._guard_seen_details.clear()
        if hasattr(self, "sampler"):
            self.sampler = None
        if hasattr(self, "model"):
            self.model = None
        if not self.enforce_eager:
            if hasattr(self, "graphs"):
                del self.graphs
            if hasattr(self, "graph_vars"):
                del self.graph_vars
            if hasattr(self, "graph_bs"):
                del self.graph_bs
            if hasattr(self, "graph_pool"):
                del self.graph_pool
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def call(self, method_name, *args):
        return getattr(self, method_name)(*args)

    def _get_kv_cache_modules(self):
        if self.model is None:
            return []
        return [module for module in self.model.modules() if hasattr(module, "k_cache") and hasattr(module, "v_cache") and not bool(getattr(module, "_exclude_paged_kv_cache", False))]

    def _allocate_kv_tensors(self, layers: int, num_kv_heads: int, head_dim: int, scale_elements: int, runtime_device) -> None:
        """The cache tensors (zeros, as vLLM: the attention reads the unwritten slots of a sequence's last page with a zero weight, and 0 * NaN
        is NaN). A BF16 or Q8 cache is one [2, layers, pages, ...] tensor; a mixed layout keeps the two bodies in one byte buffer with a
        view per component (keys then values, their widths differ), plus the exact Q8 overlays allocated once per runtime."""
        config, layout = self.config, self.kv_layout
        blocks = config.num_kvcache_blocks
        if layout is None or not layout.mixed:
            self.kv_cache = torch.zeros(2, layers, blocks, self.block_size, num_kv_heads, head_dim, device=runtime_device, dtype=torch.int8 if config.kv_cache_int8 else self.dtype)
            self.kv_views = (self.kv_cache[0], self.kv_cache[1])
            self.kv_min_views = (None, None)
            if config.kv_cache_int8:
                self.kv_cache_scales = torch.zeros(2, layers, blocks, self.block_size, num_kv_heads, scale_elements, device=runtime_device, dtype=torch.float16)
            return
        shape = lambda width: (layers, blocks, self.block_size, num_kv_heads, width)
        widths = (layout.k.width(head_dim), layout.v.width(head_dim))
        sizes = [math.prod(shape(width)) for width in widths]
        self.kv_cache = torch.zeros(sum(sizes), device=runtime_device, dtype=torch.uint8)
        self.kv_views = tuple(self.kv_cache[offset:offset + size].view(fmt.dtype).view(shape(width))
                              for fmt, width, size, offset in zip((layout.k, layout.v), widths, sizes, (0, sizes[0])))
        self.kv_cache_scales = torch.zeros(2, *shape(scale_elements), device=runtime_device, dtype=torch.float16)
        asymmetric = [fmt.asymmetric for fmt in (layout.k, layout.v)]
        self.kv_cache_mins = torch.zeros(sum(asymmetric), *shape(scale_elements), device=runtime_device, dtype=torch.float16) if any(asymmetric) else None
        mins = iter(self.kv_cache_mins) if self.kv_cache_mins is not None else None
        self.kv_min_views = tuple(next(mins) if flag else None for flag in asymmetric)
        if layout.overlay and getattr(self, "kv_overlay", None) is None:
            rows = config.max_num_seqs * layout.overlay_rows(self.block_size) + 1  # plus a scratch row for tokens without an overlay (PyTorch store path)
            self.kv_overlay = torch.zeros(2, layers, rows, num_kv_heads, head_dim, device=runtime_device, dtype=torch.int8)
            self.kv_overlay_scales = torch.zeros(2, layers, rows, num_kv_heads, scale_elements, device=runtime_device, dtype=torch.float16)

    def _bind_kv_modules(self, modules) -> None:
        config, layout = self.config, self.kv_layout
        for layer_id, module in enumerate(modules):
            module.k_cache, module.v_cache = self.kv_views[0][layer_id], self.kv_views[1][layer_id]
            if not config.kv_cache_int8:
                continue
            module.k_scale, module.v_scale = self.kv_cache_scales[0, layer_id], self.kv_cache_scales[1, layer_id]
            module.kv_layout = layout
            module.k_min = None if self.kv_min_views[0] is None else self.kv_min_views[0][layer_id]
            module.v_min = None if self.kv_min_views[1] is None else self.kv_min_views[1][layer_id]
            if layout.overlay:
                module.k_overlay, module.v_overlay = self.kv_overlay[0, layer_id], self.kv_overlay[1, layer_id]
                module.k_overlay_scale, module.v_overlay_scale = self.kv_overlay_scales[0, layer_id], self.kv_overlay_scales[1, layer_id]

    def kv_snapshot_items(self) -> dict:
        """name -> (tensor, page axis) of the paged cache tensors, for page-range snapshots."""
        items = {}
        if self.kv_layout is not None and self.kv_layout.mixed:
            items["k_body"], items["v_body"] = (self.kv_views[0], 1), (self.kv_views[1], 1)
        else:
            items["kv_cache"] = (self.kv_cache, 2)
        if getattr(self, "kv_cache_scales", None) is not None:
            items["kv_cache_scales"] = (self.kv_cache_scales, 2)
        if getattr(self, "kv_cache_mins", None) is not None:
            items["kv_cache_mins"] = (self.kv_cache_mins, 2)
        return items

    def snapshot_overlay(self, seq: Sequence) -> dict | None:
        if getattr(self, "kv_overlay", None) is None or seq.overlay_slot < 0:
            return None
        rows = self.kv_layout.overlay_rows(self.block_size)
        base = seq.overlay_slot * rows
        return {"data": transfer.move(self.kv_overlay[:, :, base:base + rows], "cpu"), "scales": transfer.move(self.kv_overlay_scales[:, :, base:base + rows], "cpu"), "ring_high": int(seq.ring_high)}

    def restore_overlay(self, seq: Sequence, saved: dict | None) -> None:
        if saved is None or getattr(self, "kv_overlay", None) is None or seq.overlay_slot < 0:
            return
        rows = self.kv_layout.overlay_rows(self.block_size)
        base = seq.overlay_slot * rows
        transfer.copy_(self.kv_overlay[:, :, base:base + rows], saved["data"])
        transfer.copy_(self.kv_overlay_scales[:, :, base:base + rows], saved["scales"])
        seq.ring_high = int(saved["ring_high"])

    def _fill_overlays_from_body(self, seq: Sequence, count: int) -> None:
        """The exact overlays of positions [0, count) that were cached from other sequences (prefix cache): requantized from the body."""
        layout, page = self.kv_layout, self.block_size
        sink = layout.sink_tokens(page)
        positions = torch.cat((torch.arange(0, min(count, sink)), torch.arange(max(count - layout.recent_tokens, sink), count)))
        seq.ring_high = max(seq.ring_high, count)
        if positions.numel() == 0:
            return
        device = self.kv_overlay.device
        table = torch.tensor(seq.block_table, dtype=torch.long)
        pages, slots = table[positions // page].to(device), (positions % page).to(device)
        rows = (seq.overlay_slot * layout.overlay_rows(page) + torch.where(positions < sink, positions, sink + (positions & (layout.recent_tokens - 1)))).to(device)
        for component, fmt in enumerate((layout.k, layout.v)):
            mins = self.kv_min_views[component]
            values = kvf.dequantize_heads(self.kv_views[component][:, pages, slots], self.kv_cache_scales[component][:, pages, slots], None if mins is None else mins[:, pages, slots], fmt)
            data, scale, _ = kvf.quantize_heads(values, kvf.Q8)
            self.kv_overlay[component][:, rows] = data
            self.kv_overlay_scales[component][:, rows] = scale

    def _fill_overlay_buffers(self, seq: Sequence, start: int, count: int) -> bool:
        """The overlay rows of positions [start, start + count) of one sequence into the pinned speculative buffers, its base and ring start
        into the first decode buffers; False without overlays."""
        layout = self.kv_layout
        if layout is None or not layout.overlay:
            return False
        page = self.block_size
        base = seq.overlay_slot * layout.overlay_rows(page) if seq.overlay_slot >= 0 and seq.block_table else -1
        if base >= 0 and seq.ring_high < start:
            self._fill_overlays_from_body(seq, start)
        buffer = self._cpu_speculative_overlay_mapping
        for index in range(count):
            buffer[index] = base + layout.overlay_row(start + index, page) if base >= 0 else -1
        seq.ring_high = max(seq.ring_high, start + count)
        self._cpu_overlay_base[0] = base
        self._cpu_ring_start[0] = layout.ring_start(seq.ring_high, page)
        return True

    def _overlay_context(self, seqs: list[Sequence], ranges: list[tuple[int, int]]):
        """(overlay_mapping, overlay_base, ring_start) tensors of a pass writing the positions [start, end) of each sequence; Nones without
        overlays. Advances each sequence's ring_high."""
        layout = self.kv_layout
        if layout is None or not layout.overlay:
            return None, None, None
        page = self.block_size
        rows, sink = layout.overlay_rows(page), layout.sink_tokens(page)
        mapping, bases, starts = [], [], []
        for seq, (start, end) in zip(seqs, ranges):
            base = seq.overlay_slot * rows if seq.overlay_slot >= 0 and seq.block_table else -1
            if base >= 0:
                if seq.ring_high < start:
                    self._fill_overlays_from_body(seq, start)
                positions = torch.arange(start, end, dtype=torch.int32)
                mapping.append(base + torch.where(positions < sink, positions, sink + (positions & (layout.recent_tokens - 1))))
                seq.ring_high = max(seq.ring_high, end)
            else:
                mapping.append(torch.full((end - start,), -1, dtype=torch.int32))
            bases.append(base)
            starts.append(layout.ring_start(seq.ring_high, page))
        return (self._to_runtime_device(torch.cat(mapping)), self._to_runtime_device(torch.tensor(bases, dtype=torch.int32)), self._to_runtime_device(torch.tensor(starts, dtype=torch.int32)))

    def allocate_kv_cache(self, generation_tokens=None):
        config = self.config
        hf_config = config.hf_config
        runtime_device = self._get_runtime_device()
        is_cuda_runtime = runtime_device.type == "cuda"
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        if config.kv_cache_int8 and head_dim % 32:
            raise RuntimeError(f"INT8 KV cache requires a head dimension divisible by 32; got {head_dim}.")
        kv_cache_modules = self._get_kv_cache_modules()
        kv_cache_layer_count = len(kv_cache_modules)
        cache_element_size = torch.empty((), dtype=torch.int8 if config.kv_cache_int8 else self.dtype).element_size()
        scale_elements = head_dim // 32 if config.kv_cache_int8 else 0
        scale_element_size = torch.float16.itemsize if config.kv_cache_int8 else 0
        if config.kv_cache_int8:
            block_bytes = self.block_size * self.kv_layout.body_bytes_per_token(kv_cache_layer_count, num_kv_heads, head_dim)
        else:
            block_bytes = 2 * kv_cache_layer_count * self.block_size * num_kv_heads * head_dim * cache_element_size

        # Default consumers reserve the full limit; token generators can opt into growth.
        cache_length = config.max_model_len
        if config.kv_cache_initial_tokens:
            self._kv_cache_generation_tokens = min(config.kv_cache_initial_tokens, config.kv_cache_max_tokens) if generation_tokens is None else generation_tokens
            cache_length = config.kv_cache_prompt_tokens + self._kv_cache_generation_tokens
        required_blocks_per_seq = (cache_length + self.block_size - 1) // self.block_size
        required_total_blocks = required_blocks_per_seq * max(1, config.max_num_seqs)
        config.num_kvcache_blocks = max(1, int(required_total_blocks))
        required_kv_bytes = config.num_kvcache_blocks * block_bytes
        # MMGP mode: the blocks kept in VRAM by the dynamic preload leave room for the cache when the allocator needs it
        if is_cuda_runtime and not self.mmgp:
            free, total = torch.cuda.mem_get_info()
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            target_total_usage = total * config.gpu_memory_utilization
            allowed_kv_bytes = max(0, target_total_usage - current)
            if required_kv_bytes > allowed_kv_bytes:
                raise RuntimeError(
                    f"Insufficient GPU memory for strict KV cache sizing under gpu_memory_utilization={config.gpu_memory_utilization:.2f}. "
                    f"Required KV: {required_kv_bytes / 1024**3:.2f} GB "
                    f"(blocks={config.num_kvcache_blocks}, block={block_bytes / 1024**2:.2f} MB), "
                    f"Allowed by limit: {allowed_kv_bytes / 1024**3:.2f} GB, "
                    f"Free: {free / 1024**3:.2f} GB, Current: {current / 1024**3:.2f} GB, "
                    f"Requested max_model_len={config.max_model_len}, max_num_seqs={config.max_num_seqs}."
                )
        try:
            self._allocate_kv_tensors(kv_cache_layer_count, num_kv_heads, head_dim, scale_elements, runtime_device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                extra = ""
                if is_cuda_runtime:
                    free_now, total_now = torch.cuda.mem_get_info()
                    extra = f" Current free VRAM: {free_now / 1024**3:.2f} GB / {total_now / 1024**3:.2f} GB total."
                raise RuntimeError(
                    f"Failed to allocate strict KV cache ({required_kv_bytes / 1024**3:.2f} GB) for "
                    f"max_model_len={config.max_model_len}, max_num_seqs={config.max_num_seqs} on {runtime_device}.{extra}"
                ) from exc
            raise
        self._bind_kv_modules(kv_cache_modules)

        if config.kv_cache_initial_tokens and generation_tokens is None:
            print(f"[LM Engine] KV cache: {self._kv_cache_generation_tokens}/{config.kv_cache_max_tokens} generation tokens + {config.kv_cache_prompt_tokens} prompt tokens per branch; {config.num_kvcache_blocks} pages ({required_kv_bytes / 1024**2:.1f} MiB).")

    @torch.inference_mode()
    def grow_kv_cache(self, minimum_blocks):
        config = self.config
        old_tokens = tokens = self._kv_cache_generation_tokens
        blocks = config.num_kvcache_blocks
        while blocks < minimum_blocks and tokens < config.kv_cache_max_tokens:
            tokens = min((tokens * 3 + 1) // 2, config.kv_cache_max_tokens)
            blocks = ((config.kv_cache_prompt_tokens + tokens + self.block_size - 1) // self.block_size) * config.max_num_seqs
        if blocks < minimum_blocks:
            raise RuntimeError("KV cache growth would exceed the request's maximum token budget.")
        if self._get_runtime_device().type == "cuda":
            torch.cuda.synchronize()
        if not self.enforce_eager:
            self.clear_graph_cache()
            self.graphs.clear()
            self.graph_vars.clear()
            self.graph_pool = None
        # Stage live pages in RAM before allocating their larger GPU replacement (the exact overlays are not paged and stay in place).
        page_tensors = _kv_page_tensors(self)
        staged = [tensor.cpu() for _name, tensor, _axis in page_tensors]
        for module in self._get_kv_cache_modules():
            module.k_cache = module.v_cache = torch.empty(0, device="cpu")
            if config.kv_cache_int8:
                module.k_scale = module.v_scale = torch.empty(0, device="cpu")
                module.k_min = module.v_min = None
        for name in {name for name, _tensor, _axis in page_tensors}:
            delattr(self, name)
        self.kv_views = None
        self.allocate_kv_cache(tokens)
        if not self.enforce_eager:
            # Capture writes dummy KV slots. Restore live pages only AFTER capture.
            self.capture_cudagraph()
        # saved's pages into the first pages of each tensor, without a full-cache GPU staging copy
        for (_name, tensor, axis), saved in zip(_kv_page_tensors(self), staged):
            transfer.copy_(tensor.narrow(axis, 0, saved.shape[axis]), saved)
        del staged
        self._runtime_signature = self._get_graph_capture_signature()
        print(f"[LM Engine] KV cache grew: {old_tokens} -> {tokens}/{config.kv_cache_max_tokens} generation tokens per branch; {config.num_kvcache_blocks} pages.")
        return config.num_kvcache_blocks

    def prepare_block_tables(self, seqs: list[Sequence]):
        bs = len(seqs)
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = self._cpu_block_tables[:bs, :max_len]
        block_tables.fill_(-1)
        for row, seq in enumerate(seqs):
            if not seq.block_table:
                continue
            block_tables[row, :len(seq.block_table)] = torch.tensor(seq.block_table, dtype=torch.int32, device="cpu")
        return self._to_runtime_device(block_tables)

    def prepare_prefill(self, seqs: list[Sequence]):
        use_prompt_embeds = any(getattr(seq, "prompt_embeds", None) is not None for seq in seqs)
        if use_prompt_embeds and not all(getattr(seq, "prompt_embeds", None) is not None for seq in seqs):
            raise RuntimeError("Mixed embedded/non-embedded prefill batches are not supported.")
        input_ids = []
        positions = []
        prompt_embeds = []
        prompt_position_ids = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        has_previous_state = any(int(getattr(seq, "num_cached_tokens", 0) or 0) > 0 for seq in seqs)
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            if use_prompt_embeds:
                seq_prompt_embeds = getattr(seq, "prompt_embeds", None)
                seq_position_ids = getattr(seq, "prompt_position_ids", None)
                if seq_prompt_embeds is None or seq_position_ids is None:
                    raise RuntimeError("Embedded prefill requires both prompt_embeds and prompt_position_ids.")
                seq_prompt_embeds = seq_prompt_embeds[seq.num_cached_tokens:seqlen]
                if seq_prompt_embeds.ndim != 2 or seq_prompt_embeds.shape[0] != seqlen_q:
                    raise RuntimeError("Embedded prefill prompt_embeds shape does not match uncached prompt length.")
                if seq_position_ids.ndim == 3:
                    seq_position_ids = seq_position_ids[:, 0]
                if seq_position_ids.ndim != 2 or seq_position_ids.shape[0] != 3:
                    raise RuntimeError("Embedded prefill prompt_position_ids must have shape [3, seq_len].")
                seq_position_ids = seq_position_ids[:, seq.num_cached_tokens:seqlen]
                if seq_position_ids.shape[1] != seqlen_q:
                    raise RuntimeError("Embedded prefill prompt_position_ids shape does not match uncached prompt length.")
                prompt_embeds.append(seq_prompt_embeds)
                prompt_position_ids.append(seq_position_ids)
            else:
                positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup: no blocks allocated yet
                slot_mapping.extend([-1] * seqlen_q)
                continue
            cached_tokens = max(0, int(seq.num_cached_tokens or 0))
            cached_partial_tokens = cached_tokens % self.block_size
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i == seq.num_cached_blocks and cached_partial_tokens > 0:
                    start += cached_partial_tokens
                if i != seq.num_blocks - 1:
                    end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    end = seq.block_table[i] * self.block_size + seq.last_block_num_tokens
                if end > start:
                    slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        overlay_mapping, overlay_base, ring_start = self._overlay_context(seqs, [(seq.num_cached_tokens, len(seq)) for seq in seqs])
        if self._prefill_copied is not None:
            self._prefill_copied.synchronize()  # the previous prefill's copies have read the buffers refilled below
        staged = lambda buffer, values: self._to_runtime_device(buffer[:len(values)].copy_(torch.tensor(values, dtype=buffer.dtype)))
        if use_prompt_embeds:
            input_ids = None
            positions = self._to_runtime_device(torch.cat(prompt_position_ids, dim=1).unsqueeze(1).contiguous())
            inputs_embeds = self._to_runtime_device(torch.cat(prompt_embeds, dim=0).unsqueeze(0).contiguous())
        else:
            input_ids = staged(self._cpu_prefill_input_ids, input_ids)
            positions = staged(self._cpu_prefill_positions, positions)
            inputs_embeds = None
        cu_seqlens_q = staged(self._cpu_prefill_cu_seqlens, cu_seqlens_q)
        cu_seqlens_k = staged(self._cpu_prefill_cu_seqlens_k, cu_seqlens_k)
        slot_mapping = staged(self._cpu_prefill_slot_mapping, slot_mapping)
        self._prefill_copied = torch.cuda.current_stream().record_event() if self._pin_memory_enabled() else None
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, has_previous_state=has_previous_state,
                    overlay_mapping=overlay_mapping, overlay_base=overlay_base, ring_start=ring_start)
        return input_ids, positions, inputs_embeds

    @torch.inference_mode()
    def prefill_only(self, seqs: list[Sequence]) -> None:
        self.ensure_runtime_ready()
        input_ids, positions, inputs_embeds = self.prepare_prefill(seqs)
        self.run_model(input_ids, positions, True, inputs_embeds=inputs_embeds)
        for seq in seqs:
            seq.clear_prompt_data()
        reset_context()

    @torch.inference_mode()
    def _prefill_prefix_chunks(self, seq: Sequence, chunk_tokens: int) -> None:
        """Prefill all but the final chunk without sampling or retaining logits."""
        num_tokens = seq.num_tokens
        token_ids = seq.token_ids
        try:
            for chunk_end in range(seq.num_cached_tokens + chunk_tokens, num_tokens, chunk_tokens):
                seq.num_tokens = chunk_end
                seq.token_ids = token_ids[:chunk_end]
                input_ids, positions, inputs_embeds = self.prepare_prefill([seq])
                self.model(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds)
                seq.num_cached_tokens = chunk_end
                reset_context()
        finally:
            seq.num_tokens = num_tokens
            seq.token_ids = token_ids

    def prepare_decode(self, seqs: list[Sequence]):
        """Optimized decode preparation using pre-allocated buffers."""
        bs = len(seqs)
        
        # Use pre-allocated CPU buffers
        for i, seq in enumerate(seqs):
            self._cpu_input_ids[i] = seq.last_token
            self._cpu_positions[i] = len(seq) - 1 + int(getattr(seq, "position_offset", 0) or 0)
            self._cpu_context_lens[i] = len(seq)
            self._cpu_slot_mapping[i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        layout = self.kv_layout
        overlay = layout is not None and layout.overlay
        if overlay:
            rows = layout.overlay_rows(self.block_size)
            for i, seq in enumerate(seqs):
                base = seq.overlay_slot * rows if seq.overlay_slot >= 0 else -1
                if base >= 0 and seq.ring_high < len(seq) - 1:
                    self._fill_overlays_from_body(seq, len(seq) - 1)
                seq.ring_high = max(seq.ring_high, len(seq))
                self._cpu_overlay_mapping[i] = base + layout.overlay_row(len(seq) - 1, self.block_size) if base >= 0 else -1
                self._cpu_overlay_base[i] = base
                self._cpu_ring_start[i] = layout.ring_start(seq.ring_high, self.block_size)
        
        # Transfer to the runtime device using sliced views
        input_ids = self._to_runtime_device(self._cpu_input_ids[:bs])
        positions = self._to_runtime_device(self._cpu_positions[:bs])
        slot_mapping = self._to_runtime_device(self._cpu_slot_mapping[:bs])
        context_lens = self._to_runtime_device(self._cpu_context_lens[:bs])
        block_tables = self.prepare_block_tables(seqs)
        overlay_tensors = (self._to_runtime_device(self._cpu_overlay_mapping[:bs]), self._to_runtime_device(self._cpu_overlay_base[:bs]), self._to_runtime_device(self._cpu_ring_start[:bs])) if overlay else (None, None, None)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, overlay_mapping=overlay_tensors[0], overlay_base=overlay_tensors[1], ring_start=overlay_tensors[2])
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_cfg_batch: bool = False):
        """Optimized sample preparation using pre-allocated buffers."""
        if is_cfg_batch:
            num_seqs = len(seqs) // 2
            target_seqs = seqs[:num_seqs]
        else:
            num_seqs = len(seqs)
            target_seqs = seqs
        
        # Fill pre-allocated CPU buffers
        top_ks_is_zero = True
        top_ps_is_one = True
        min_ps_is_zero = True
        repetition_penalties_is_one = True
        for i, seq in enumerate(target_seqs):
            self._cpu_temperatures[i] = seq.temperature
            self._cpu_cfg_scales[i] = seq.cfg_scale
            self._cpu_top_ks[i] = seq.top_k if seq.top_k is not None else 0
            if seq.top_k is not None and seq.top_k > 0:
                top_ks_is_zero = False
            self._cpu_top_ps[i] = seq.top_p if seq.top_p is not None else 1.0
            if seq.top_p is not None and seq.top_p != 1.0:
                top_ps_is_one = False
            self._cpu_min_ps[i] = seq.min_p if seq.min_p is not None else 0.0
            if seq.min_p is not None and seq.min_p > 0.0:
                min_ps_is_zero = False
            self._cpu_repetition_penalties[i] = seq.repetition_penalty if seq.repetition_penalty is not None else 1.0
            if seq.repetition_penalty is not None and seq.repetition_penalty != 1.0:
                repetition_penalties_is_one = False
        
        # Transfer to the runtime device using sliced views (single batched transfer)
        temperatures = self._to_runtime_device(self._cpu_temperatures[:num_seqs])
        cfg_scales = self._to_runtime_device(self._cpu_cfg_scales[:num_seqs])
        top_ks = self._to_runtime_device(self._cpu_top_ks[:num_seqs]) if not top_ks_is_zero else None
        top_ps = self._to_runtime_device(self._cpu_top_ps[:num_seqs]) if not top_ps_is_one else None
        min_ps = self._to_runtime_device(self._cpu_min_ps[:num_seqs]) if not min_ps_is_zero else None
        repetition_penalties = self._cpu_repetition_penalties[:num_seqs] if not repetition_penalties_is_one else None
        
        return temperatures, cfg_scales, top_ks, top_ps, min_ps, repetition_penalties

    def _sync_repetition_token_cache(self, seq: Sequence) -> dict:
        boundary = int(seq.repetition_penalty_start)
        completion_count = int(seq.num_tokens) - boundary
        cache = self._repetition_token_cache.get(seq.seq_id)
        rules = (id(seq.repetition_exempt_ids), id(seq.repetition_span_marks))
        if cache is None or cache["boundary"] != boundary or cache["synced_tokens"] > completion_count or cache["rules"] != rules:
            if cache is None and len(self._repetition_token_cache) >= max(1, int(self.config.max_num_seqs)) * 2:
                self._repetition_token_cache.pop(next(iter(self._repetition_token_cache)))
            if cache is None:
                cache = {"device_states": {}}
                self._repetition_token_cache[seq.seq_id] = cache
            cache.update(boundary=boundary, synced_tokens=0, seen=set(), token_ids=[], span=_OUTSIDE_SPANS, rules=rules)
            for state in cache["device_states"].values():
                state["count"] = 0
                state["cursor"] = 0
        new_tokens = seq.token_ids[boundary + cache["synced_tokens"]:boundary + completion_count]
        full_vocab_size = int(self.config.hf_config.vocab_size)
        exempt, marks, span = seq.repetition_exempt_ids or (), seq.repetition_span_marks, cache["span"]
        for token_id in new_tokens:
            token_id = int(token_id)
            inside = span != _OUTSIDE_SPANS
            if marks is not None and token_id in marks:
                span = _advance_spans(span, marks[token_id])
            if not inside and 0 <= token_id < full_vocab_size and token_id not in exempt and token_id not in cache["seen"]:
                cache["seen"].add(token_id)
                cache["token_ids"].append(token_id)
        cache["span"] = span
        cache["synced_tokens"] = completion_count
        return cache

    @staticmethod
    def _repetition_virtual_tokens(seq: Sequence, cache: dict, virtual_tokens, vocab_size: int) -> tuple[bool, list[int]]:
        """Whether the position after the drafted tokens lies inside a code, quote or backtick span (no penalty there), and the drafted tokens that
        count as repeated so far."""
        exempt, marks, span = seq.repetition_exempt_ids or (), seq.repetition_span_marks, cache["span"]
        extra_ids = []
        extra_seen = set()
        for token_id in virtual_tokens:
            token_id = int(token_id)
            inside = span != _OUTSIDE_SPANS
            if marks is not None and token_id in marks:
                span = _advance_spans(span, marks[token_id])
            if not inside and 0 <= token_id < vocab_size and token_id not in exempt and token_id not in cache["seen"] and token_id not in extra_seen:
                extra_seen.add(token_id)
                extra_ids.append(token_id)
        return span != _OUTSIDE_SPANS, extra_ids

    def _apply_repetition_penalty(self, seq: Sequence, logits: torch.Tensor, penalty: float, virtual_tokens=()) -> None:
        if penalty == 1.0:
            return
        vocab_size = int(logits.shape[-1])
        cache = self._sync_repetition_token_cache(seq)
        suspended, virtual_ids = self._repetition_virtual_tokens(seq, cache, virtual_tokens, vocab_size)
        if suspended:
            return
        device_key = (logits.device.type, logits.device.index, vocab_size)
        state = cache["device_states"].get(device_key)
        if state is None:
            capacity = int(self.config.max_model_len) + _REPETITION_INCREMENT_LIMIT
            state = {
                "indices": torch.empty(capacity, dtype=torch.long, device=logits.device),
                "values": torch.empty(capacity, dtype=logits.dtype, device=logits.device),
                "count": 0,
                "cursor": 0,
            }
            cache["device_states"][device_key] = state
        new_ids = [token_id for token_id in cache["token_ids"][state["cursor"]:] if token_id < vocab_size]
        state["cursor"] = len(cache["token_ids"])
        if len(new_ids) > _REPETITION_INCREMENT_LIMIT:
            valid_ids = [token_id for token_id in cache["token_ids"] if token_id < vocab_size]
            state["indices"][:len(valid_ids)].copy_(torch.tensor(valid_ids, dtype=torch.long, device=logits.device))
            state["count"] = len(valid_ids)
            new_ids = []
        apply_sparse_repetition_penalty_(logits, state["indices"], state["count"], new_ids, virtual_ids, penalty, state["values"], use_triton=self.use_triton_sampling)
        state["count"] += len(new_ids)

    @staticmethod
    def _speculative_logits_processor(seq: Sequence, predictive: bool):
        processor = seq.logits_processor
        if predictive and not seq.predictive_penalty and processor is not None and hasattr(processor, "_without_penalty"):
            processor = processor._without_penalty
        return processor

    def _apply_speculative_logit_rules(self, seq: Sequence, logits: torch.Tensor, repetition_penalty: float, virtual_tokens: list[int], vocab_size: int | None = None, predictive: bool = False) -> torch.Tensor:
        logits = logits.clone()
        logits_processor = self._speculative_logits_processor(seq, predictive)
        if predictive and not seq.predictive_penalty:
            repetition_penalty = 1.0
        supports_partial_vocab = getattr(logits_processor, "_supports_partial_vocab", False)
        if callable(supports_partial_vocab):
            supports_partial_vocab = supports_partial_vocab()
        if vocab_size is not None and logits_processor is not None and not supports_partial_vocab:
            expanded_logits = logits.new_full((int(self.config.hf_config.vocab_size),), float("-inf"))
            expanded_logits[:vocab_size].copy_(logits)
            logits = expanded_logits
            vocab_size = None
        self._apply_repetition_penalty(seq, logits, repetition_penalty, virtual_tokens)
        bias = self._get_logits_bias(seq, logits.unsqueeze(0))
        if bias is not None:
            if vocab_size is not None:
                bias = bias[..., :vocab_size]
            self._apply_logits_bias(logits, bias)
        if logits_processor is not None:
            logits = self._call_logits_processor(seq, logits_processor, logits.unsqueeze(0), virtual_tokens)[0]
        return logits

    def _sample_speculative_target(self, seq: Sequence, logits: torch.Tensor, sample_params, virtual_tokens: list[int], profile: dict | None = None, profile_role: str = "target") -> int:
        if seq.top_k == 1:
            repetition_penalties = sample_params[-1]
            penalty = 1.0 if repetition_penalties is None else float(repetition_penalties[0].item())
            logits = self._apply_speculative_logit_rules(seq, logits, penalty, virtual_tokens)
            token_id = int(torch.argmax(logits).item())
        else:
            token_id = self._sample_distribution(self._speculative_distribution(seq, logits, sample_params, virtual_tokens, profile=profile, profile_role=profile_role))
        if seq.logits_processor_update_state is not None:
            seq.logits_processor_update_state(token_id)
        return token_id

    def _speculative_distribution(self, seq: Sequence, logits: torch.Tensor, sample_params, virtual_tokens: list[int], vocab_size: int | None = None, predictive: bool = False, profile: dict | None = None, profile_role: str = "target", rules_applied: bool = False) -> torch.Tensor:
        temperatures, _cfg_scales, _top_ks, _top_ps, _min_ps, repetition_penalties = sample_params
        penalty = 1.0 if repetition_penalties is None else float(repetition_penalties[0].item())
        if not rules_applied:
            logits = self._apply_speculative_logit_rules(seq, logits, penalty, virtual_tokens, vocab_size, predictive=predictive)
        if (predictive and logits.is_cuda and self.use_triton_sampling and not self.enforce_eager
                and not getattr(self.model, "_block_draft", False)
                and not getattr(self, "_disable_mtp_gpu_draft", False)
                and seq.top_k != 1 and bounded_support(seq)):
            return draft_probabilities(self, seq, logits)
        logits = logits.float().div_(temperatures[0])
        top_k = int(seq.top_k) if seq.top_k is not None and 0 < int(seq.top_k) < logits.numel() else None
        top_p = float(seq.top_p) if seq.top_p is not None and 0.0 < float(seq.top_p) < 1.0 else None
        min_p = float(seq.min_p) if seq.min_p is not None and float(seq.min_p) > 0.0 else None
        if top_k is None and top_p is None and min_p is None:
            return torch.softmax(logits, dim=-1)
        if top_k is None:
            universe_logits = logits
            universe_ids = None
        else:
            threshold = torch.topk(logits, top_k).values[-1]
            universe_ids = torch.nonzero(logits >= threshold, as_tuple=False).flatten()
            universe_logits = logits[universe_ids]
        if self.enforce_eager or not logits.is_cuda or profile is not None:
            return self._filter_speculative_distribution(logits, universe_logits, universe_ids, top_p, min_p, profile, profile_role, predictive)
        # Retain compact candidates, including top-k ties. Sorting a padded full
        # vocabulary changes PyTorch's tie order at the nucleus cutoff.
        key = (logits.device, logits.numel(), universe_logits.numel(), universe_ids is None, top_p, min_p)
        state = self._speculative_sampling_graphs.pop(key, None)
        if state is None:
            if len(self._speculative_sampling_graphs) >= 8:
                self._speculative_sampling_graphs.pop(next(iter(self._speculative_sampling_graphs)))
            values = universe_logits.clone()
            ids = None if universe_ids is None else universe_ids.clone()
            self._filter_speculative_distribution(logits, values, ids, top_p, min_p)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                probabilities = self._filter_speculative_distribution(logits, values, ids, top_p, min_p)
            state = graph, values, ids, probabilities
        self._speculative_sampling_graphs[key] = state
        graph, values, ids, probabilities = state
        values.copy_(universe_logits)
        if ids is not None:
            ids.copy_(universe_ids)
        graph.replay()
        # Draft distributions remain live while later drafts use the same graph.
        return probabilities.clone()

    def _filter_speculative_distribution(self, logits: torch.Tensor, universe_logits: torch.Tensor, universe_ids: torch.Tensor | None, top_p: float | None, min_p: float | None, profile: dict | None = None, profile_role: str = "target", predictive: bool = False) -> torch.Tensor:
        log_normalizer = torch.logsumexp(universe_logits, dim=0) if top_p is not None else None
        candidate_logits = apply_min_p_mask_(universe_logits, min_p, use_triton=self.use_triton_sampling) if min_p is not None else universe_logits
        candidate_ids = universe_ids
        distribution_index = None
        if profile is not None:
            distribution_index = len(profile["distributions"])
            if distribution_index < profile["slot"]["distribution_counts"].size(0):
                profile["distributions"].append({"role": profile_role, "predictive": bool(predictive), "vocab": int(logits.numel()), "top_p": top_p, "min_p": min_p})
                min_p_survivors = profile["slot"]["distribution_counts"][distribution_index, 0]
                if min_p is None:
                    min_p_survivors.fill_(candidate_logits.numel())
                else:
                    torch.sum(torch.isfinite(candidate_logits), dim=(0,), out=min_p_survivors)
            else:
                distribution_index = None
        if top_p is not None:
            excluded_mass = 1.0 - torch.exp(candidate_logits - log_normalizer).sum()
            candidate_logits, order = torch.sort(candidate_logits)
            candidate_ids = order if candidate_ids is None else candidate_ids[order]
            keep = excluded_mass + torch.exp(candidate_logits - log_normalizer).cumsum(dim=0) > 1.0 - top_p
            candidate_logits[:-1].masked_fill_(~keep[:-1], float("-inf"))
            if distribution_index is not None:
                torch.sum(torch.isfinite(candidate_logits), dim=(0,), out=profile["slot"]["distribution_counts"][distribution_index, 1])
                profile["slot"]["distribution_masses"][distribution_index].copy_(excluded_mass)
        elif candidate_ids is None:
            if distribution_index is not None:
                profile["slot"]["distribution_counts"][distribution_index, 1].fill_(candidate_logits.numel())
                profile["slot"]["distribution_masses"][distribution_index].zero_()
            return torch.softmax(candidate_logits, dim=-1)
        elif distribution_index is not None:
            profile["slot"]["distribution_counts"][distribution_index, 1].fill_(candidate_logits.numel())
            profile["slot"]["distribution_masses"][distribution_index].zero_()
        probabilities = torch.zeros_like(logits)
        probabilities.scatter_(0, candidate_ids, torch.softmax(candidate_logits, dim=-1))
        return probabilities

    def _sample_distribution_tensor(self, probabilities: torch.Tensor) -> torch.Tensor:
        noise = torch.empty_like(probabilities).exponential_(1, generator=self._sampling_generator).clamp_min_(1e-10)
        return probabilities.div(noise).argmax()

    def _sample_distribution(self, probabilities: torch.Tensor) -> int:
        return int(self._sample_distribution_tensor(probabilities).item())

    def _commit_speculative_target_state(self, processed_tokens: int, verified_tokens: int) -> None:
        if processed_tokens == verified_tokens:
            return
        graph = self.speculative_commit_graphs.get(int(processed_tokens))
        if graph is not None:
            graph.replay()
        else:
            self._commit_speculative_target_state_eager(int(processed_tokens))

    def _commit_speculative_target_state_eager(self, processed_tokens: int) -> None:
        torch._foreach_copy_(self._speculative_commit_destinations, self._speculative_commit_sources[processed_tokens])
        for module in self._speculative_replay_modules:
            module.replay_speculative_state(processed_tokens)

    def _store_speculative_pending(self, seq: Sequence, target_logits: torch.Tensor, hidden_states: torch.Tensor, positions: torch.Tensor) -> None:
        self._speculative_drafts.pop(seq.seq_id, None)
        pending_position = positions[0, 0, -1:] if positions.ndim == 3 else positions[-1:]
        self._speculative_pending[seq.seq_id] = {
            "target_logits": target_logits.detach().clone(),
            "hidden_states": hidden_states[:, -1:].detach().clone(),
            "positions": pending_position.detach().clone(),
        }

    def _prime_mtp_context(self, seq: Sequence) -> None:
        chunk_tokens = int(getattr(self.model, "_prefill_chunk_tokens", 0))
        if chunk_tokens and seq.prompt_embeds is not None and len(seq) > chunk_tokens:
            self._prime_embedded_mtp_chunks(seq, chunk_tokens)
            return
        self.model.mtp.reset_sequence_state()
        input_ids, positions, inputs_embeds = self.prepare_prefill([seq])
        hidden_states = self.model(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds)
        self._prepare_speculative_graph_state()  # the prefill made the weights resident: the state replay may have replaced snapshots
        target_logits = self.model.compute_logits(hidden_states)[0]
        if len(seq) > 1:
            shifted_ids = torch.tensor(seq.token_ids[1:], dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            shifted_embeds = None if inputs_embeds is None else inputs_embeds[:, 1:]
            mtp_positions = positions[..., :-1] if positions.ndim == 3 else positions[:-1]
            self.model.mtp(shifted_ids, mtp_positions, hidden_states[:, :-1], inputs_embeds=shifted_embeds, compute_logits=False)
        self._store_speculative_pending(seq, target_logits, hidden_states, positions)
        seq.clear_prompt_data()
        reset_context()
        self.speculative_stats["target_passes"] += 1

    def _prime_embedded_mtp_chunks(self, seq: Sequence, chunk_tokens: int) -> None:
        self.model.mtp.reset_sequence_state()
        self._prepare_speculative_graph_state()
        token_ids = seq.token_ids
        num_tokens = seq.num_tokens
        previous_hidden = previous_positions = None
        try:
            for start in range(0, num_tokens, chunk_tokens):
                end = min(start + chunk_tokens, num_tokens)
                seq.token_ids = token_ids[:end]
                seq.num_tokens = end
                seq.num_cached_tokens = start
                input_ids, positions, inputs_embeds = self.prepare_prefill([seq])
                hidden_states = self.model(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds)
                if previous_hidden is None:
                    mtp_hidden = hidden_states[:, :-1]
                    mtp_positions = positions[..., :-1]
                    mtp_embeds = inputs_embeds[:, 1:]
                    shifted_ids = token_ids[start + 1:end]
                else:
                    # Carry the preceding target hidden state and its full 3D position.
                    # Pair each target state with the next token's visual embedding.
                    mtp_hidden = torch.cat((previous_hidden, hidden_states[:, :-1]), dim=1)
                    mtp_positions = torch.cat((previous_positions, positions[..., :-1]), dim=-1)
                    mtp_embeds = inputs_embeds
                    shifted_ids = token_ids[start:end]
                if shifted_ids:
                    shifted_ids = torch.tensor(shifted_ids, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
                    self.model.mtp(shifted_ids, mtp_positions, mtp_hidden, inputs_embeds=mtp_embeds, compute_logits=False)
                previous_hidden = hidden_states[:, -1:].clone()
                previous_positions = positions[..., -1:].clone()
                reset_context()
            self._store_speculative_pending(seq, self.model.compute_logits(hidden_states[:, -1:])[0], hidden_states, positions)
            seq.clear_prompt_data()
            self.speculative_stats["target_passes"] += 1
        finally:
            seq.token_ids = token_ids
            seq.num_tokens = num_tokens
            reset_context()

    @torch.inference_mode()
    def prefill_mtp_only(self, seqs: list[Sequence]) -> None:
        self.ensure_runtime_ready()
        self._prime_mtp_context(seqs[0])

    @torch.inference_mode()
    def prefill_mtp_suffix(self, seqs: list[Sequence], previous_token_count: int) -> None:
        self.ensure_runtime_ready()
        seq = seqs[0]
        previous_token_count = int(previous_token_count)
        previous_pending = self._speculative_pending.pop(seq.seq_id, None)
        seq.num_cached_tokens = previous_token_count if previous_pending is not None else previous_token_count - 1
        input_ids, positions, inputs_embeds = self.prepare_prefill([seq])
        hidden_states = self.model(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds)
        target_logits = self.model.compute_logits(hidden_states)[0]
        suffix_ids = torch.tensor(seq.token_ids[previous_token_count:], dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        if previous_pending is None:
            mtp_hidden = hidden_states[:, :-1]
            mtp_positions = positions[:-1]
        else:
            mtp_hidden = torch.cat((previous_pending["hidden_states"], hidden_states[:, :-1]), dim=1)
            mtp_positions = torch.cat((previous_pending["positions"], positions[:-1]), dim=0)
        self.model.mtp(suffix_ids, mtp_positions, mtp_hidden, compute_logits=False)
        self._store_speculative_pending(seq, target_logits, hidden_states, positions)
        seq.clear_prompt_data()
        reset_context()
        self.speculative_stats["target_passes"] += 1

    def snapshot_speculative_state(self, seq_id: int, previous: dict | None = None, reuse_tokens: int = 0) -> dict:
        pending = self._speculative_pending.get(seq_id)
        draft = self._speculative_drafts.get(seq_id)
        return {
            "mtp_cache": self.model.mtp.snapshot_sequence_state(previous=None if previous is None else previous["mtp_cache"], reuse_tokens=reuse_tokens),
            "draft": None if draft is None else {name: tensor.detach().as_subclass(torch.Tensor).to("cpu", copy=True) for name, tensor in draft.items()},
            "pending": None if pending is None else {name: tensor.detach().as_subclass(torch.Tensor).to("cpu", copy=True) for name, tensor in pending.items()},
        }

    def restore_speculative_state(self, seq_id: int, snapshot: dict) -> None:
        self.model.mtp.restore_sequence_state(snapshot["mtp_cache"])
        self._speculative_drafts.clear()
        self._speculative_pending.clear()
        device = self._get_runtime_device()
        draft = snapshot.get("draft")
        if draft is not None:
            self._speculative_drafts[seq_id] = {name: tensor.to(device=device) for name, tensor in draft.items()}
        pending = snapshot.get("pending")
        if pending is not None:
            self._speculative_pending[seq_id] = {name: tensor.to(device=device) for name, tensor in pending.items()}

    def snapshot_speculative_rewind_state(self, seq_id: int) -> dict:
        pending = self._speculative_pending.get(seq_id)
        draft = self._speculative_drafts.get(seq_id)
        return {
            "mtp_cache_length": self.model.mtp.get_cache_length(),
            "draft": None if draft is None else {name: tensor.detach().as_subclass(torch.Tensor).to("cpu", copy=True) for name, tensor in draft.items()},
            "pending": None if pending is None else {name: tensor.detach().as_subclass(torch.Tensor).to("cpu", copy=True) for name, tensor in pending.items()},
        }

    def restore_speculative_rewind_state(self, seq_id: int, snapshot: dict) -> None:
        self.model.mtp.truncate_cache(int(snapshot["mtp_cache_length"]))
        self._speculative_drafts.clear()
        self._speculative_pending.clear()
        device = self._get_runtime_device()
        draft = snapshot.get("draft")
        if draft is not None:
            self._speculative_drafts[seq_id] = {name: tensor.to(device=device) for name, tensor in draft.items()}
        pending = snapshot.get("pending")
        if pending is not None:
            self._speculative_pending[seq_id] = {name: tensor.to(device=device) for name, tensor in pending.items()}

    def _prepare_speculative_verify(self, seq: Sequence, draft_tokens: list[int] | torch.Tensor):
        verify_length = len(draft_tokens) + 1
        current_slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        start_position = len(seq) - 1 + int(getattr(seq, "position_offset", 0) or 0)
        if self.enforce_eager:
            if torch.is_tensor(draft_tokens):
                input_ids = torch.cat((torch.tensor([seq.last_token], dtype=torch.long, device=draft_tokens.device), draft_tokens))
            else:
                input_ids = torch.tensor([seq.last_token, *draft_tokens], dtype=torch.long, device=self._get_runtime_device())
            positions = torch.arange(start_position, start_position + verify_length, dtype=torch.long, device=self._get_runtime_device())
            slot_mapping = torch.arange(current_slot, current_slot + verify_length, dtype=torch.int32, device=self._get_runtime_device())
            cu_seqlens_q = torch.tensor([0, verify_length], dtype=torch.int32, device=self._get_runtime_device())
            cu_seqlens_k = torch.tensor([0, len(seq) + len(draft_tokens)], dtype=torch.int32, device=self._get_runtime_device())
            block_tables = self.prepare_block_tables([seq])
            overlay_mapping, overlay_base, ring_start = self._overlay_context([seq], [(len(seq) - 1, len(seq) - 1 + verify_length)])
        else:
            graph_vars = self.speculative_graph_vars[verify_length]
            overlay_mapping, overlay_base, ring_start = graph_vars["overlay_mapping"], graph_vars["overlay_base"], graph_vars["ring_start"]
            if self._fill_overlay_buffers(seq, len(seq) - 1, verify_length):  # pinned CPU buffers: no host synchronization in the MTP loop
                overlay_mapping.copy_(self._cpu_speculative_overlay_mapping[:verify_length], non_blocking=True)
                overlay_base.copy_(self._cpu_overlay_base[:1], non_blocking=True)
                ring_start.copy_(self._cpu_ring_start[:1], non_blocking=True)
            input_ids = graph_vars["input_ids"]
            if torch.is_tensor(draft_tokens):
                input_ids[0].fill_(seq.last_token)
                input_ids[1:].copy_(draft_tokens)
            else:
                cpu_input_ids = self._cpu_speculative_input_ids[:verify_length]
                cpu_input_ids[0] = seq.last_token
                for index, token_id in enumerate(draft_tokens, start=1):
                    cpu_input_ids[index] = token_id
                input_ids.copy_(cpu_input_ids, non_blocking=cpu_input_ids.is_pinned())
            positions = graph_vars["positions"]
            torch.arange(start_position, start_position + verify_length, device=positions.device, out=positions)
            slot_mapping = graph_vars["slot_mapping"]
            torch.arange(current_slot, current_slot + verify_length, device=slot_mapping.device, out=slot_mapping)
            cu_seqlens_q = graph_vars["cu_seqlens_q"]
            cu_seqlens_k = graph_vars["cu_seqlens_k"]
            cu_seqlens_k[1].fill_(len(seq) + len(draft_tokens))
            block_tables = graph_vars["block_tables"]
            block_table_signature = tuple(seq.block_table)
            if graph_vars.get("block_table_signature") != block_table_signature:
                cpu_block_table = self._cpu_block_tables[:1, :len(block_table_signature)]
                cpu_block_table[0].copy_(torch.tensor(block_table_signature, dtype=torch.int32))
                block_tables.fill_(-1)
                block_tables[:, :len(block_table_signature)].copy_(cpu_block_table, non_blocking=cpu_block_table.is_pinned())
                graph_vars["block_table_signature"] = block_table_signature
        # an MMGP graph keeps the bound of its capture, as the engine's graphs of the verification do
        max_seqlen_k = self.config.max_model_len if self.mmgp_graphs and not self.enforce_eager else len(seq) + len(draft_tokens)
        set_context(True, cu_seqlens_q, cu_seqlens_k, verify_length, max_seqlen_k, slot_mapping, None, block_tables, has_previous_state=True, speculative_verify=True,
                    overlay_mapping=overlay_mapping, overlay_base=overlay_base, ring_start=ring_start)
        return input_ids, positions

    def _run_mtp_forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor, inputs_embeds: torch.Tensor | None = None, compute_logits: bool = True, last_logits_only: bool = False):
        if self.enforce_eager or inputs_embeds is not None or positions.numel() != input_ids.numel() or getattr(self, "mtp_graph", None) is None:
            return self.model.mtp(input_ids, positions, hidden_states, inputs_embeds=inputs_embeds, compute_logits=compute_logits, last_logits_only=last_logits_only)
        count = input_ids.numel()
        graph_vars = self.mtp_graph_vars if count == 1 else self.mtp_refresh_graphs[count]
        if input_ids is not graph_vars["input_ids"]:
            graph_vars["input_ids"].copy_(input_ids.reshape_as(graph_vars["input_ids"]))
        if positions is not graph_vars["positions"]:
            graph_vars["positions"].copy_(positions.reshape_as(graph_vars["positions"]))
        graph_vars["hidden_states"].copy_(hidden_states.reshape_as(graph_vars["hidden_states"]))
        self.model.mtp._cache.prepare_append()
        graph = self.mtp_graph if count == 1 else graph_vars["graph"]
        graph.replay()
        self.model.mtp._cache.advance(count)
        return graph_vars["outputs"], graph_vars["logits"]

    def _advance_mtp(self, seq: Sequence, token_ids: list[int], positions: torch.Tensor, hidden_states: torch.Tensor) -> None:
        count = len(token_ids)
        graph_vars = None
        if not self.enforce_eager and positions.numel() == count and getattr(self, "mtp_graph", None) is not None:
            graph_vars = self.mtp_graph_vars if count == 1 else self.mtp_refresh_graphs[count]
        if graph_vars is None:
            mtp_input_ids = torch.tensor([token_ids], dtype=torch.long, device=hidden_states.device)
        else:
            mtp_input_ids = graph_vars["input_ids"]
            if count == 1:
                mtp_input_ids.fill_(token_ids[0])
            else:
                mtp_input_ids.copy_(torch.tensor([token_ids], dtype=torch.long, device="cpu"))
        mtp_hidden, mtp_logits = self._run_mtp_forward(mtp_input_ids, positions, hidden_states, last_logits_only=True)
        next_token = graph_vars["next_token"] if graph_vars is not None else torch.argmax(mtp_logits[0, -1]).reshape(1)
        self._speculative_drafts[seq.seq_id] = {"logits": mtp_logits[0, -1].clone(), "hidden_states": mtp_hidden[:, -1:].clone(), "next_token": next_token.clone()}

    def _build_mtp_drafts(self, seq: Sequence, sample_params, draft_count: int, start_position: int, profile: dict | None = None) -> tuple[list[int] | torch.Tensor, int, list[torch.Tensor] | None]:
        draft_state = self._speculative_drafts[seq.seq_id]
        graph_vars = self.mtp_graph_vars if getattr(self, "mtp_graph", None) is not None and not self.enforce_eager else None
        draft_vocab_size = getattr(self.model.mtp, "draft_vocab_size", None)
        cache_length = self.model.mtp.get_cache_length()
        rejection_sampling = seq.top_k != 1
        confidence_threshold = 0.0 if rejection_sampling else float(getattr(self.model, "_prompt_enhancer_speculative_confidence", 0.0))
        predictive_processor = self._speculative_logits_processor(seq, predictive=True)
        predictive_repetition = sample_params[-1] is not None and seq.predictive_penalty
        if not rejection_sampling and draft_count == 2 and confidence_threshold <= 0.0 and not predictive_repetition and predictive_processor is None:
            initial_logits = draft_state["logits"]
            bias = self._get_logits_bias(seq, initial_logits.unsqueeze(0))
            if bias is not None:
                initial_logits = initial_logits + (bias[..., :draft_vocab_size] if draft_vocab_size is not None else bias)
            first_draft = torch.argmax(initial_logits).reshape(1) if bias is not None else draft_state["next_token"]
            if graph_vars is not None:
                draft_position = graph_vars["positions"]
                draft_position.fill_(start_position)
            else:
                draft_position = torch.tensor([start_position], dtype=torch.long, device=draft_state["hidden_states"].device)
            _draft_hidden, draft_logits = self._run_mtp_forward(first_draft.view(1, 1), draft_position, draft_state["hidden_states"], last_logits_only=True)
            if bias is None and getattr(self, "mtp_graph", None) is not None and not self.enforce_eager:
                second_draft = self.mtp_graph_vars["next_token"]
            else:
                second_logits = draft_logits[0, -1]
                if bias is not None:
                    second_logits = second_logits + (bias[..., :draft_vocab_size] if draft_vocab_size is not None else bias)
                second_draft = torch.argmax(second_logits).reshape(1)
            return torch.cat((first_draft, second_draft)), cache_length, None
        sample_drafts_on_device = rejection_sampling and not predictive_repetition and (predictive_processor is None or not getattr(predictive_processor, "_requires_input_ids", True))
        draft_tokens = []
        draft_distributions = [] if rejection_sampling else None
        draft_logits = draft_state["logits"]
        draft_hidden = draft_state["hidden_states"]
        for draft_idx in range(draft_count):
            if rejection_sampling:
                probabilities = self._speculative_distribution(seq, draft_logits, sample_params, () if sample_drafts_on_device else draft_tokens, draft_vocab_size, predictive=True, profile=profile, profile_role=f"draft{draft_idx}")
                draft_index = self._sample_distribution_tensor(probabilities) if sample_drafts_on_device else self._sample_distribution(probabilities)
                draft_distributions.append(probabilities)
            else:
                repetition_penalties = sample_params[-1]
                penalty = 1.0 if repetition_penalties is None else float(repetition_penalties[0].item())
                processed_logits = self._apply_speculative_logit_rules(seq, draft_logits, penalty, draft_tokens, draft_vocab_size, predictive=True)
                top_logit, top_token = torch.max(processed_logits, dim=0)
                if confidence_threshold > 0.0:
                    confidence = torch.exp(top_logit.float() - torch.logsumexp(processed_logits.float(), dim=0))
                    draft_index, confidence = torch.stack((top_token.float(), confidence)).tolist()
                    if confidence < confidence_threshold:
                        break
                else:
                    draft_index = int(top_token.item())
            draft_token = draft_index.reshape(1) if sample_drafts_on_device else int(draft_index)
            draft_tokens.append(draft_token)
            if draft_idx + 1 == draft_count:
                break
            if graph_vars is not None:
                draft_position = graph_vars["positions"]
                draft_position.fill_(start_position + draft_idx)
                mtp_input_ids = graph_vars["input_ids"]
                if sample_drafts_on_device:
                    mtp_input_ids.copy_(draft_token.view_as(mtp_input_ids))
                else:
                    mtp_input_ids.fill_(draft_token)
            else:
                draft_position = torch.tensor([start_position + draft_idx], dtype=torch.long, device=draft_hidden.device)
                mtp_input_ids = draft_token.view(1, 1) if sample_drafts_on_device else torch.tensor([[draft_token]], dtype=torch.long, device=draft_hidden.device)
            draft_hidden, draft_logits = self._run_mtp_forward(mtp_input_ids, draft_position, draft_hidden, last_logits_only=True)
            draft_logits = draft_logits[0, -1]
        return torch.cat(draft_tokens) if sample_drafts_on_device else draft_tokens, cache_length, draft_distributions

    def _run_native_mtp_target_token(self, seq: Sequence, sample_params) -> list[list[int]]:
        input_ids, positions = self.prepare_decode([seq])
        hidden_states = self._run_decode_hidden(input_ids, positions)
        target_token = self._sample_speculative_target(seq, self.model.compute_logits(hidden_states)[0], sample_params, [])
        self._advance_mtp(seq, [target_token], positions, hidden_states)
        reset_context()
        self.speculative_stats["target_passes"] += 1
        self.speculative_stats["emitted_tokens"] += 1
        return [[target_token]]

    def _emit_pending_mtp_token(self, seq: Sequence, sample_params) -> list[list[int]]:
        pending = self._speculative_pending.pop(seq.seq_id)
        sampled_token = self._sample_speculative_target(seq, pending["target_logits"], sample_params, [])
        self._advance_mtp(seq, [sampled_token], pending["positions"], pending["hidden_states"])
        self.speculative_stats["emitted_tokens"] += 1
        return [[sampled_token]]

    @torch.inference_mode()
    def _run_native_mtp(self, seq: Sequence, is_prefill: bool) -> list[list[int]]:
        if not self.model._prompt_enhancer_speculative_method_logged:
            sampling_method = "rejection sampling" if seq.top_k != 1 else "greedy exact-match"
            execution = "eager" if self.enforce_eager else "CUDA graph"
            draft_tokens = self.model._prompt_enhancer_speculative_sampling_tokens if seq.top_k != 1 else self.model._prompt_enhancer_speculative_tokens
            method = self.model.mtp.method if getattr(self.model, "_block_draft", False) else "native MTP"
            print(f"[Deepy][Speculative] method={method} ({sampling_method}, up to {draft_tokens} draft tokens, {execution} verification).")
            self.model._prompt_enhancer_speculative_method_logged = True
        sample_key = (seq.temperature, seq.cfg_scale, seq.top_k, seq.top_p, seq.min_p, seq.repetition_penalty)
        if self._mtp_sample_params is None or self._mtp_sample_params[0] != sample_key:
            sample_params = self.prepare_sample([seq], is_cfg_batch=False)
            if sample_params[-1] is not None:
                sample_params = (*sample_params[:-1], sample_params[-1].clone())
            self._mtp_sample_params = (sample_key, sample_params)
        else:
            sample_params = self._mtp_sample_params[1]
        if is_prefill:
            self._prime_mtp_context(seq)
        if seq.seq_id in self._speculative_pending:
            return self._emit_pending_mtp_token(seq, sample_params)

        max_emission = int(getattr(seq, "speculative_max_emission", seq.max_tokens - seq.num_completion_tokens))
        remaining_tokens = min(seq.max_tokens - seq.num_completion_tokens, max_emission)
        max_drafts = min(self._max_speculative_draft_tokens, max(1, int(getattr(self.model, "_prompt_enhancer_speculative_tokens", 1))))
        if seq.top_k != 1:
            max_drafts = min(max_drafts, max(1, int(getattr(self.model, "_prompt_enhancer_speculative_sampling_tokens", max_drafts))))
        draft_count = min(max_drafts, remaining_tokens - 1, self.block_size - seq.last_block_num_tokens - 1)
        if draft_count < 1:
            return self._run_native_mtp_target_token(seq, sample_params)

        profile = self._start_mtp_stage_profile()
        start_position = len(seq) - 1 + int(getattr(seq, "position_offset", 0) or 0)
        stage_started = time.perf_counter()
        draft_tokens, mtp_cache_length, draft_distributions = self._build_mtp_drafts(seq, sample_params, draft_count, start_position, profile=profile)
        self._mark_mtp_stage_profile(profile, 1, "draft", stage_started)
        if len(draft_tokens) == 0:
            self._cancel_mtp_stage_profile(profile)
            return self._run_native_mtp_target_token(seq, sample_params)
        stage_started = time.perf_counter()
        input_ids, positions = self._prepare_speculative_verify(seq, draft_tokens)
        self._mark_mtp_stage_profile(profile, 2, "verify_setup", stage_started)
        stage_started = time.perf_counter()
        hidden_states = self._run_speculative_verify_graph(input_ids, positions)
        self._mark_mtp_stage_profile(profile, 3, "verify", stage_started)
        stage_started = time.perf_counter()
        logits = self.model.output(hidden_states)[0]
        self._mark_mtp_stage_profile(profile, 4, "output", stage_started)
        reset_context()
        self.speculative_stats["target_passes"] += 1
        emitted, accepted_count = self._sample_verified_block(seq, logits, draft_tokens, draft_distributions, sample_params, profile)
        stage_started = time.perf_counter()
        self._commit_speculative_target_state(len(emitted), len(draft_tokens) + 1)
        self._mark_mtp_stage_profile(profile, 7, "commit", stage_started)
        stage_started = time.perf_counter()
        # Replace all speculative MTP entries with the verified target states.
        self.model.mtp.truncate_cache(mtp_cache_length)
        commit_start = 0
        self._mark_mtp_stage_profile(profile, 8, "truncate", stage_started)
        mtp_positions = positions[..., commit_start:len(emitted)] if positions.ndim == 3 else positions[commit_start:len(emitted)]
        stage_started = time.perf_counter()
        self._advance_mtp(seq, emitted[commit_start:], mtp_positions, hidden_states[:, commit_start:len(emitted)])
        self._mark_mtp_stage_profile(profile, 9, "mtp_advance", stage_started)
        self._finish_mtp_stage_profile(profile, accepted_count, len(emitted), commit_start, len(seq))
        self.speculative_stats["emitted_tokens"] += len(emitted)
        return [emitted]

    def _sample_verified_block(self, seq, logits, draft_tokens, draft_distributions, sample_params, profile=None):
        from .speculative_sampling import can_batch_acceptance, sample_verified_block
        if (torch.is_tensor(draft_tokens) and logits.is_cuda
                and not getattr(self, "_disable_mtp_gpu_acceptance", False)
                and can_batch_acceptance(self, seq)):
            return sample_verified_block(self, seq, logits, draft_tokens, draft_distributions, sample_params, profile)
        return self._sample_verified_block_reference(seq, logits, draft_tokens, draft_distributions, sample_params, profile)

    def _sample_verified_block_reference(self, seq, logits, draft_tokens, draft_distributions, sample_params, profile=None):
        greedy_target_tokens = None
        if seq.top_k == 1 and sample_params[-1] is None and seq.logits_processor is None:
            bias = self._get_logits_bias(seq, logits)
            if bias is not None:
                logits.add_(bias)
            greedy_target_tokens = torch.argmax(logits, dim=-1).tolist()
        emitted = []
        accepted_count = 0
        speculative_stop_token_ids = set(getattr(seq, "speculative_stop_token_ids", ()))
        stopped = False
        sync_started = time.perf_counter()
        draft_token_ids = draft_tokens.tolist() if torch.is_tensor(draft_tokens) else draft_tokens
        if profile is not None:
            profile["cpu_ms"]["host_wait_at_draft_sync"] = 1000.0 * (time.perf_counter() - sync_started)
            self._collect_mtp_stage_profiles()
            profile["slot"]["events"][5].record(torch.cuda.current_stream())
        stage_started = time.perf_counter()
        for draft_idx, draft_token in enumerate(draft_token_ids):
            if draft_distributions is not None:
                target_distribution = self._speculative_distribution(seq, logits[draft_idx], sample_params, emitted, profile=profile, profile_role=f"target{draft_idx}")
                draft_distribution = draft_distributions[draft_idx]
                acceptance = torch.clamp(target_distribution[draft_token] / draft_distribution[draft_token], max=1.0)
                accepted = bool((torch.rand((), device=logits.device, generator=self._sampling_generator) < acceptance).item())
                if accepted:
                    target_token = draft_token
                else:
                    residual = target_distribution.clone()
                    residual[:draft_distribution.numel()].sub_(draft_distribution)
                    target_token = self._sample_distribution(residual.clamp_min_(0))
                if seq.logits_processor_update_state is not None:
                    seq.logits_processor_update_state(target_token)
            else:
                target_token = greedy_target_tokens[draft_idx] if greedy_target_tokens is not None else self._sample_speculative_target(seq, logits[draft_idx], sample_params, emitted, profile=profile, profile_role=f"target{draft_idx}")
            emitted.append(target_token)
            self.speculative_stats["drafted"] += 1
            self.speculative_stats["drafted_by_position"][draft_idx] += 1
            if target_token != draft_token:
                break
            self.speculative_stats["accepted"] += 1
            self.speculative_stats["accepted_by_position"][draft_idx] += 1
            accepted_count += 1
            stopped = ((not seq.ignore_eos and target_token == self.config.eos) or target_token in speculative_stop_token_ids)
            if stopped:
                break
        if accepted_count == len(draft_token_ids) and not stopped:
            emitted.append(greedy_target_tokens[len(draft_token_ids)] if greedy_target_tokens is not None else self._sample_speculative_target(seq, logits[len(draft_token_ids)], sample_params, emitted, profile=profile, profile_role="bonus"))
        self._mark_mtp_stage_profile(profile, 6, "sampling", stage_started)
        return emitted, accepted_count

    def _replay_decode_graph_hidden(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bs = input_ids.size(0)
        context = get_context()
        graph_bs = next((x for x in self.graph_bs if x >= bs), None)
        if graph_bs is None:
            raise RuntimeError(f"CUDA graph batch size {bs} exceeds captured graph sizes {self.graph_bs} (max_num_seqs={self.config.max_num_seqs})")
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:graph_bs].zero_()
        graph_vars["positions"][:graph_bs].zero_()
        graph_vars["slot_mapping"][:graph_bs].fill_(self.config.num_kvcache_blocks * self.block_size - 1)
        graph_vars["context_lens"][:graph_bs].zero_()
        graph_vars["block_tables"][:graph_bs].fill_(-1)
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph_vars["overlay_mapping"][:graph_bs].fill_(-1)
        graph_vars["overlay_base"][:graph_bs].fill_(-1)
        graph_vars["ring_start"][:graph_bs].fill_(1 << 30)
        if context.overlay_mapping is not None:
            graph_vars["overlay_mapping"][:bs] = context.overlay_mapping
            graph_vars["overlay_base"][:bs] = context.overlay_base
            graph_vars["ring_start"][:bs] = context.ring_start
        if self.mmgp_graphs:
            # the graphs read the context from the buffers registered with the model's graph state (whole buffers when they are the size)
            rows = (lambda t: t) if graph_bs == graph_vars["input_ids"].shape[0] else (lambda t: t[:graph_bs])
            set_context(False, slot_mapping=rows(graph_vars["slot_mapping"]), context_lens=rows(graph_vars["context_lens"]), block_tables=rows(graph_vars["block_tables"]),
                        overlay_mapping=rows(graph_vars["overlay_mapping"]), overlay_base=rows(graph_vars["overlay_base"]), ring_start=rows(graph_vars["ring_start"]))
            self._run_target_stage(rows(graph_vars["input_ids"]), rows(graph_vars["positions"]), rows(graph_vars["outputs"]), self._draft_feature_buffer(graph_vars, graph_bs))
            return graph_vars["outputs"][:bs]
        self.graphs[graph_bs].replay()
        if getattr(self.model, "_block_draft", False):
            self.model._draft_features = graph_vars["draft_features"]
        return graph_vars["outputs"][:bs]

    def _run_decode_hidden(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.enforce_eager:
            return self.model(input_ids=input_ids, positions=positions)
        hidden_states = self._replay_decode_graph_hidden(input_ids, positions)
        return hidden_states.unsqueeze(1) if hidden_states.ndim == 2 else hidden_states

    def _run_speculative_verify_graph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.enforce_eager:
            return self.model(input_ids=input_ids, positions=positions)
        verify_length = int(input_ids.numel())
        graph_vars = self.speculative_graph_vars[verify_length]
        if self.mmgp_graphs:
            self._run_target_stage(graph_vars["input_ids"], graph_vars["positions"], graph_vars["outputs"], self._draft_feature_buffer(graph_vars, verify_length))
            return graph_vars["outputs"]
        self.speculative_graphs[verify_length].replay()
        if getattr(self.model, "_block_draft", False):
            self.model._draft_features = graph_vars["draft_features"]
        return graph_vars["outputs"]

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor | None, positions: torch.Tensor, is_prefill: bool, inputs_embeds: torch.Tensor | None = None):
        decode_batch_size = input_ids.size(0) if input_ids is not None else int(inputs_embeds.shape[0])
        model_kwargs = {"input_ids": input_ids, "positions": positions}
        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds
        if is_prefill or self.enforce_eager or decode_batch_size > 512:
            return self.model.compute_logits(self.model(**model_kwargs))
        else:
            bs = decode_batch_size
            context = get_context()
            
            # Check if block_tables size exceeds pre-allocated buffer size
            # This can happen when conditional and unconditional sequences have different lengths
            # in CFG mode, causing block_tables to have more columns than expected
            max_num_blocks = self.graph_vars["block_tables"].size(1)
            if context.block_tables.size(1) > max_num_blocks:
                # Fall back to eager mode when block_tables is too large for CUDA graph
                self._note_guard(
                    "cudagraph_fallback_block_table_cols",
                    f"requested={context.block_tables.size(1)} max={max_num_blocks}",
                )
                return self.model.compute_logits(self.model(**model_kwargs))
            
            # Fix: Also check if block_tables row count matches batch size
            # Dimension mismatch can cause CUDA illegal memory access during graph replay
            if context.block_tables.size(0) != bs:
                # Fall back to eager mode when block_tables row count doesn't match batch size
                self._note_guard(
                    "cudagraph_fallback_block_table_rows",
                    f"rows={context.block_tables.size(0)} bs={bs}",
                )
                return self.model.compute_logits(self.model(**model_kwargs))
            
            # Fix: Verify slot_mapping and context_lens dimensions match batch size
            if context.slot_mapping.size(0) != bs or context.context_lens.size(0) != bs:
                # Fall back to eager mode when dimensions don't match
                self._note_guard(
                    "cudagraph_fallback_context_shape",
                    f"slot={context.slot_mapping.size(0)} ctx={context.context_lens.size(0)} bs={bs}",
                )
                return self.model.compute_logits(self.model(**model_kwargs))
            
            return self.model.compute_logits(self._replay_decode_graph_hidden(input_ids, positions))

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Run model forward and sampling. For CFG sequences, batch is structured as:
        [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
        where uncond_seqi is the paired unconditional sequence of cond_seqi."""
        self.ensure_runtime_ready()
        # Check if this is a CFG batch (contains paired conditional and unconditional sequences)
        is_cfg_batch = seqs[0].cfg_scale > 1.0 and seqs[0].paired_seq is not None
        if not is_cfg_batch and len(seqs) == 1 and getattr(self.model, "mtp", None) is not None and bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            return self._run_native_mtp(seqs[0], is_prefill)
        if is_cfg_batch:
            # CFG batch: seqs = [cond_seq1, cond_seq2, ..., uncond_seq1, uncond_seq2, ...]
            num_cond = len(seqs) // 2
            cond_seqs = seqs[:num_cond]
            # uncond_seqs = seqs[num_cond:]
            
            # Prepare inputs for both conditional and unconditional (they're already in the batch)
            if is_prefill:
                input_ids, positions, inputs_embeds = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)
                inputs_embeds = None
            sample_params = self.prepare_sample(seqs, is_cfg_batch=True) if self.rank == 0 else None
            if sample_params is not None:
                temperatures, cfg_scales, top_ks, top_ps, min_ps, repetition_penalties = sample_params
            else:
                temperatures = cfg_scales = top_ks = top_ps = min_ps = repetition_penalties = None
            
            # Run model forward (processes entire batch: cond + uncond)
            logits_all = self.run_model(input_ids, positions, is_prefill, inputs_embeds=inputs_embeds)
            if is_prefill:
                for seq in seqs:
                    seq.clear_prompt_data()
            reset_context()
            
            if self.rank == 0:
                # Split logits: first half is conditional, second half is unconditional
                logits_cond = logits_all[:num_cond].clone()
                logits_uncond = logits_all[num_cond:]
                
                # Apply repetition penalty to conditional logits (before CFG)
                if repetition_penalties is not None:
                    for i, seq in enumerate(cond_seqs):
                        penalty = repetition_penalties[i].item()
                        self._apply_repetition_penalty(seq, logits_cond[i], float(penalty))
                
                # Apply CFG formula: logits_cfg = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                cfg_scales_tensor = cfg_scales.unsqueeze(1)  # [num_cond, 1]
                logits_cfg = logits_uncond + cfg_scales_tensor * (logits_cond - logits_uncond)

                # Apply optional per-sequence logits bias before processors/sampling.
                for i, seq in enumerate(cond_seqs):
                    bias = self._get_logits_bias(seq, logits_cfg)
                    if bias is not None:
                        self._apply_logits_bias(logits_cfg[i], bias)
                
                # Apply logits processor for constrained decoding (if any sequence has one)
                for i, seq in enumerate(cond_seqs):
                    if seq.logits_processor is not None:
                        logits_cfg[i:i+1] = self._call_logits_processor(seq, seq.logits_processor, logits_cfg[i:i+1])

                # Prepare input_ids for sampler (for repetition penalty, though we already applied it)
                # cond_input_ids = torch.tensor([seq.token_ids for seq in cond_seqs], device=logits_cfg.device)
                
                # Sample from CFG logits
                token_ids_cfg = self.sampler(
                    logits_cfg, 
                    temperatures,
                    top_ks=top_ks if top_ks is not None else None,
                    top_ps=top_ps if top_ps is not None else None,
                    min_ps=min_ps if min_ps is not None else None,
                    repetition_penalties=None,  # Already applied above
                    generator=self._sampling_generator,
                    # input_ids=cond_input_ids,
                ).tolist()
                
                # Update logits processor state after sampling
                # NOTE: Only update for the first sequence since all sequences share the same processor
                # Updating multiple times would cause duplicate state updates (e.g., codes_count += N instead of += 1)
                if cond_seqs and cond_seqs[0].logits_processor_update_state is not None:
                    cond_seqs[0].logits_processor_update_state(token_ids_cfg[0])
                
                # Return token_ids (will be applied to both conditional and unconditional sequences)
                return token_ids_cfg
            else:
                return None
        else:
            # Normal batch (non-CFG)
            if is_prefill:
                chunk_tokens = int(getattr(self.model, "_prefill_chunk_tokens", 0))
                if chunk_tokens and len(seqs) == 1:
                    self._prefill_prefix_chunks(seqs[0], chunk_tokens)
                input_ids, positions, inputs_embeds = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)
                inputs_embeds = None
            sample_params = self.prepare_sample(seqs, is_cfg_batch=False) if self.rank == 0 else None
            if sample_params is not None:
                temperatures, cfg_scales, top_ks, top_ps, min_ps, repetition_penalties = sample_params
            else:
                temperatures = cfg_scales = top_ks = top_ps = min_ps = repetition_penalties = None
            logits = self.run_model(input_ids, positions, is_prefill, inputs_embeds=inputs_embeds)
            if is_prefill:
                for seq in seqs:
                    seq.clear_prompt_data()
            reset_context()
            
            if self.rank == 0:
                logits = logits.clone()
                # Apply repetition penalty to logits
                if repetition_penalties is not None:
                    for i, seq in enumerate(seqs):
                        penalty = repetition_penalties[i].item()
                        self._apply_repetition_penalty(seq, logits[i], float(penalty))
                
                # Apply logits processor for constrained decoding (if any sequence has one)
                for i, seq in enumerate(seqs):
                    bias = self._get_logits_bias(seq, logits)
                    if bias is not None:
                        self._apply_logits_bias(logits[i], bias)
                for i, seq in enumerate(seqs):
                    if seq.logits_processor is not None:
                        # Apply processor to this sequence's logits (clone to avoid inference mode issues)
                        processed = self._call_logits_processor(seq, seq.logits_processor, logits[i:i+1].clone())
                        logits[i] = processed[0]

                # Prepare input_ids for sampler
                # seq_input_ids = torch.tensor([seq.token_ids for seq in seqs], device=logits.device)
                
                token_ids = self.sampler(
                    logits, 
                    temperatures,
                    top_ks=top_ks if top_ks is not None else None,
                    top_ps=top_ps if top_ps is not None else None,
                    min_ps=min_ps if min_ps is not None else None,
                    repetition_penalties=None,  # Already applied above
                    generator=self._sampling_generator,
                    # input_ids=seq_input_ids,
                ).tolist()
                
                # Update logits processor state after sampling
                # NOTE: Only update for the first sequence since all sequences may share the same processor
                # (when using a single SamplingParams for batch generation)
                # Updating multiple times would cause duplicate state updates (e.g., codes_count += N instead of += 1)
                if seqs and seqs[0].logits_processor_update_state is not None:
                    seqs[0].logits_processor_update_state(token_ids[0])
                
                return token_ids
            else:
                return None

    @torch.inference_mode()
    def capture_cudagraph(self):
        if self._get_runtime_device().type != "cuda":
            self.enforce_eager = True
            return
        config = self.config
        cache_key = (config.max_model_len, config.max_num_seqs, self._max_speculative_draft_tokens)
        model_device = torch.device("cuda") if torch.cuda.is_available() else self._get_model_device()
        cached = self._graph_cache.get(cache_key)
        if cached is not None:
            current_sig = self._get_graph_capture_signature()
            if cached.get("sig") == current_sig:
                self.graphs = cached["graphs"]
                self.graph_pool = cached["pool"]
                self.graph_vars = cached["vars"]
                self.graph_bs = cached["bs"]
                self.speculative_graphs = cached.get("speculative_graphs", {})
                self.speculative_graph_vars = cached.get("speculative_vars", {})
                self.speculative_commit_graphs = cached.get("speculative_commit_graphs", {})
                self.mtp_graph = cached.get("mtp_graph")
                self.mtp_graph_pool = cached.get("mtp_pool")
                self.mtp_graph_vars = cached.get("mtp_vars", {})
                self.mtp_refresh_graphs = cached["mtp_refresh_graphs"]
                if cache_key in self._graph_cache_order:
                    self._graph_cache_order.remove(cache_key)
                self._graph_cache_order.append(cache_key)
                return
            # signature mismatch: free the stale graphs before capturing new ones
            self._drop_graph_cache_entry(cache_key)
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=model_device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=model_device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=model_device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=model_device)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_device)
        overlay_mapping = torch.full((max_bs,), -1, dtype=torch.int32, device=model_device)
        overlay_base = torch.full((max_bs,), -1, dtype=torch.int32, device=model_device)
        ring_start = torch.full((max_bs,), 1 << 30, dtype=torch.int32, device=model_device)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, device=model_device, dtype=self.dtype)
        base_graph_bs = [1, 2, 4, 8]
        self.graph_bs = [bs for bs in base_graph_bs if bs <= max_bs]
        if max_bs > 8:
            self.graph_bs.extend(range(16, max_bs + 1, 16))
        if max_bs not in self.graph_bs:
            self.graph_bs.append(max_bs)
            self.graph_bs.sort()
        if not self.graph_bs:
            self.graph_bs = [max_bs]
        self.graphs = {}
        self.graph_pool = self._graph_pool_seed
        # Kitchen's CUTLASS workspace is per stream and must exist before capture.
        capture_stream = torch.cuda.Stream(device=model_device)

        choose_short_batch_kernels = None
        try:
            import llamacpp_gguf_cuda
            prepare_runtime_buffers = getattr(llamacpp_gguf_cuda, "prepare_runtime_buffers", None)
            if prepare_runtime_buffers is not None:
                prepare_runtime_buffers(model_device)
            from shared.kernels import gguf_short_batch
            choose_short_batch_kernels = lambda: gguf_short_batch.prepare(self.model, llamacpp_gguf_cuda)
        except ImportError:
            pass

        if self.mmgp_graphs and choose_short_batch_kernels is not None:
            # The target model's passes are MMGP stages, captured at their first call: decide the kernels of its weights kept in VRAM now.
            torch.cuda.synchronize()
            choose_short_batch_kernels()
            choose_short_batch_kernels = None
        for bs in reversed(self.graph_bs if not self.mmgp_graphs else ()):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                        overlay_mapping=overlay_mapping[:bs], overlay_base=overlay_base[:bs], ring_start=ring_start[:bs])
            capture_stream.wait_stream(torch.cuda.current_stream(model_device))
            with torch.cuda.stream(capture_stream):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            if choose_short_batch_kernels is not None:
                # MMGP makes the weights resident during the first warm-up. Capture bakes the
                # short-batch kernel of each 2-8 row linear, so choose them before any capture.
                torch.cuda.synchronize()
                choose_short_batch_kernels()
                choose_short_batch_kernels = None
            with torch.cuda.graph(graph, self.graph_pool, stream=capture_stream, capture_error_mode="thread_local"):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            overlay_mapping=overlay_mapping,
            overlay_base=overlay_base,
            ring_start=ring_start,
            outputs=outputs,
        )
        if getattr(self.model, "mtp", None) is not None and bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            # The warm-up made the weights resident: GDN layers with checked launches decide on state replay before capture.
            self._prepare_target_speculative_state()
        self.speculative_graphs = {}
        self.speculative_graph_vars = {}
        self.speculative_commit_graphs = {}
        self.mtp_graph = None
        self.mtp_graph_pool = None
        self.mtp_graph_vars = {}
        self.mtp_refresh_graphs = {}
        if getattr(self.model, "_block_draft", False) and not self.mmgp_graphs:
            self.graph_vars["draft_features"] = self.model._draft_features
        if getattr(self.model, "mtp", None) is not None and bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            dummy_block = config.num_kvcache_blocks - 1
            for verify_length in reversed(range(2, self._max_speculative_draft_tokens + 2)):
                speculative_input_ids = torch.zeros(verify_length, dtype=torch.int64, device=model_device)
                speculative_positions = torch.arange(verify_length, dtype=torch.int64, device=model_device)
                speculative_slot_mapping = torch.arange(dummy_block * self.block_size, dummy_block * self.block_size + verify_length, dtype=torch.int32, device=model_device)
                speculative_cu_seqlens_q = torch.tensor([0, verify_length], dtype=torch.int32, device=model_device)
                speculative_cu_seqlens_k = torch.tensor([0, verify_length], dtype=torch.int32, device=model_device)
                speculative_block_tables = torch.full((1, max_num_blocks), -1, dtype=torch.int32, device=model_device)
                speculative_block_tables[0, 0] = dummy_block
                speculative_outputs = torch.zeros(1, verify_length, hf_config.hidden_size, device=model_device, dtype=self.dtype)
                speculative_overlay_mapping = torch.full((verify_length,), -1, dtype=torch.int32, device=model_device)
                speculative_overlay_base = torch.full((1,), -1, dtype=torch.int32, device=model_device)
                speculative_ring_start = torch.full((1,), 1 << 30, dtype=torch.int32, device=model_device)
                if not self.mmgp_graphs:
                    graph = torch.cuda.CUDAGraph()
                    set_context(True, speculative_cu_seqlens_q, speculative_cu_seqlens_k, verify_length, config.max_model_len, speculative_slot_mapping, None, speculative_block_tables, has_previous_state=True, speculative_verify=True,
                                overlay_mapping=speculative_overlay_mapping, overlay_base=speculative_overlay_base, ring_start=speculative_ring_start)
                    capture_stream.wait_stream(torch.cuda.current_stream(model_device))
                    with torch.cuda.stream(capture_stream):
                        speculative_outputs[:] = self.model(speculative_input_ids, speculative_positions)
                    with torch.cuda.graph(graph, self.graph_pool, stream=capture_stream, capture_error_mode="thread_local"):
                        speculative_outputs[:] = self.model(speculative_input_ids, speculative_positions)
                    torch.cuda.synchronize()
                    reset_context()
                    self.speculative_graphs[verify_length] = graph
                self.speculative_graph_vars[verify_length] = {
                    "input_ids": speculative_input_ids,
                    "positions": speculative_positions,
                    "slot_mapping": speculative_slot_mapping,
                    "cu_seqlens_q": speculative_cu_seqlens_q,
                    "cu_seqlens_k": speculative_cu_seqlens_k,
                    "block_tables": speculative_block_tables,
                    "overlay_mapping": speculative_overlay_mapping,
                    "overlay_base": speculative_overlay_base,
                    "ring_start": speculative_ring_start,
                    "outputs": speculative_outputs,
                }
                if getattr(self.model, "_block_draft", False) and not self.mmgp_graphs:
                    self.speculative_graph_vars[verify_length]["draft_features"] = self.model._draft_features
            def capture_commits(pool):
                # A commit replays every layer's accepted prefix: one graph per accepted count instead of a launch per layer.
                for processed_tokens in range(1, self._max_speculative_draft_tokens + 1):
                    capture_stream.wait_stream(torch.cuda.current_stream(model_device))
                    with torch.cuda.stream(capture_stream):
                        self._commit_speculative_target_state_eager(processed_tokens)    # warmup
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool, stream=capture_stream, capture_error_mode="thread_local"):
                        self._commit_speculative_target_state_eager(processed_tokens)
                    self.speculative_commit_graphs[processed_tokens] = graph
                torch.cuda.synchronize()
            # MMGP mode: the decoding passes are MMGP graphs in MMGP's pool; the commits share the MTP graphs' pool instead (one pool fewer)
            defer_commits = self.mmgp_graphs and not getattr(self.model, "_block_draft", False)
            if self._speculative_replay_modules and not defer_commits:
                capture_commits(self.graph_pool)
            if not getattr(self.model, "_block_draft", False):
                mtp_input_ids = torch.zeros((1, 1), dtype=torch.int64, device=model_device)
                mtp_positions = torch.zeros(1, dtype=torch.int64, device=model_device)
                mtp_hidden_states = torch.zeros((1, 1, hf_config.hidden_size), dtype=self.dtype, device=model_device)
                self.model.mtp._cache.cache_seqlens.zero_()
                capture_stream.wait_stream(torch.cuda.current_stream(model_device))
                # the warmups run on the MTP graphs' pool: their temporaries are the memory the graphs then use, not memory beside it
                from mmgp import allocator
                mtp_pool = torch.cuda.graph_pool_handle()
                with torch.cuda.stream(capture_stream), allocator.graph_warmup(mtp_pool):
                    self.model.mtp(mtp_input_ids, mtp_positions, mtp_hidden_states, last_logits_only=True, cache_prepared=True)
                mtp_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(mtp_graph, mtp_pool, stream=capture_stream, capture_error_mode="thread_local"):
                    mtp_outputs, mtp_logits = self.model.mtp(mtp_input_ids, mtp_positions, mtp_hidden_states, last_logits_only=True, cache_prepared=True)
                    mtp_next_token = torch.argmax(mtp_logits[0, -1]).reshape(1)
                torch.cuda.synchronize()
                self.mtp_graph = mtp_graph
                self.mtp_graph_pool = mtp_graph.pool()
                self.mtp_graph_vars = {
                    "input_ids": mtp_input_ids,
                    "positions": mtp_positions,
                    "hidden_states": mtp_hidden_states,
                    "outputs": mtp_outputs,
                    "logits": mtp_logits,
                    "next_token": mtp_next_token,
                }
                for count in reversed(range(2, self._max_speculative_draft_tokens + 2)):
                    refresh_ids = torch.zeros((1, count), dtype=torch.int64, device=model_device)
                    refresh_positions = torch.arange(count, dtype=torch.int64, device=model_device)
                    refresh_hidden = torch.zeros((1, count, hf_config.hidden_size), dtype=self.dtype, device=model_device)
                    self.model.mtp._cache.cache_seqlens.zero_()
                    capture_stream.wait_stream(torch.cuda.current_stream(model_device))
                    with torch.cuda.stream(capture_stream), allocator.graph_warmup(self.mtp_graph_pool):
                        self.model.mtp(refresh_ids, refresh_positions, refresh_hidden, last_logits_only=True, cache_prepared=True)
                    refresh_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(refresh_graph, self.mtp_graph_pool, stream=capture_stream, capture_error_mode="thread_local"):
                        refresh_outputs, refresh_logits = self.model.mtp(refresh_ids, refresh_positions, refresh_hidden, last_logits_only=True, cache_prepared=True)
                        refresh_next_token = refresh_logits[0, -1].argmax().reshape(1)
                    self.mtp_refresh_graphs[count] = {
                        "graph": refresh_graph,
                        "input_ids": refresh_ids,
                        "positions": refresh_positions,
                        "hidden_states": refresh_hidden,
                        "outputs": refresh_outputs,
                        "logits": refresh_logits,
                        "next_token": refresh_next_token,
                    }
                torch.cuda.synchronize()
                self.model.mtp._cache.cache_seqlens.zero_()
                if self._speculative_replay_modules and defer_commits:
                    capture_commits(self.mtp_graph_pool)
        if self.mmgp_graphs:
            self._capture_target_stages(slot_mapping, context_lens, block_tables, overlay_mapping, overlay_base, ring_start)
        self._graph_cache[cache_key] = {
            "graphs": self.graphs,
            "pool": self.graph_pool,
            "vars": self.graph_vars,
            "bs": self.graph_bs,
            "speculative_graphs": self.speculative_graphs,
            "speculative_vars": self.speculative_graph_vars,
            "speculative_commit_graphs": self.speculative_commit_graphs,
            "mtp_graph": self.mtp_graph,
            "mtp_pool": self.mtp_graph_pool,
            "mtp_vars": self.mtp_graph_vars,
            "mtp_refresh_graphs": self.mtp_refresh_graphs,
            "sig": self._get_graph_capture_signature(),
        }
        if cache_key in self._graph_cache_order:
            self._graph_cache_order.remove(cache_key)
        self._graph_cache_order.append(cache_key)
        while len(self._graph_cache_order) > 5:
            old_key = self._graph_cache_order.pop(0)
            self._drop_graph_cache_entry(old_key)
        self._model_storage_signature = self._get_model_storage_signature()
