import contextlib
import copy
import functools
import os
from typing import Any, Callable, NamedTuple, Optional

_TRITON_SMOKE_CACHE = None


def _env_enabled(name, default=True):
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _is_mps_available():
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


def _check_triton_runtime_smoke():
    global _TRITON_SMOKE_CACHE, tl  # Older Triton resolves kernel names from function globals.
    if _TRITON_SMOKE_CACHE is not None:
        return _TRITON_SMOKE_CACHE
    try:
        import torch
        import triton
        import triton.language as tl

        if not torch.cuda.is_available():
            _TRITON_SMOKE_CACHE = (False, "CUDA is not available")
            return _TRITON_SMOKE_CACHE

        @triton.jit
        def _smoke_add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(y_ptr + offs, x + 1.0, mask=mask)

        n_elements = 128
        block_size = 128
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) == (12, 0) and tuple(map(int, triton.__version__.split(".")[:2])) < (3, 3):
            _TRITON_SMOKE_CACHE = (False, "RTX50xx requires Triton 3.3 or newer; the installed compiler cannot compile SM120 reductions")
            return _TRITON_SMOKE_CACHE
        x = torch.arange(n_elements, dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        grid = (triton.cdiv(n_elements, block_size),)
        _smoke_add_one_kernel[grid](x, y, n_elements, BLOCK=block_size)
        torch.cuda.synchronize(device=device)
        if not torch.allclose(y, x + 1.0, atol=1e-5, rtol=1e-5):
            _TRITON_SMOKE_CACHE = (False, "Triton runtime smoke test failed: incorrect output from smoke kernel")
            return _TRITON_SMOKE_CACHE
    except Exception as exc:
        msg = str(exc).replace("\n", " ").strip()
        if len(msg) > 260:
            msg = msg[:260] + "..."
        _TRITON_SMOKE_CACHE = (False, f"Triton runtime smoke test failed: {msg}")
        return _TRITON_SMOKE_CACHE
    _TRITON_SMOKE_CACHE = (True, "ok")
    return _TRITON_SMOKE_CACHE


def _check_triton():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception as exc:
        return False, f"Triton import failed: {exc}"
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
    if _env_enabled("WGP_VLLM_TRITON_SMOKE", default=True):
        smoke_ok, smoke_msg = _check_triton_runtime_smoke()
        if not smoke_ok:
            return False, smoke_msg
    return True, "ok"


def _check_flash_attention_2():
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
        version = str(getattr(flash_attn, "__version__", ""))
    except ModuleNotFoundError:
        return False, "non installed"
    except Exception as exc:
        if "no module named 'flash_attn'" in str(exc).strip().lower():
            return False, "non installed"
        return False, f"FlashAttention import failed: {exc}"

    major = None
    if len(version) > 0:
        try:
            major = int(version.split(".", 1)[0])
        except Exception:
            major = None
    if major is not None and major < 2:
        return False, f"FlashAttention major version is {major}, expected >= 2"
    import torch
    if torch.version.hip is None and torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):  # FlashAttention 2 needs Ampere or newer
        return False, "FlashAttention 2 needs an NVIDIA GPU of compute capability 8.0 or newer (sm_{}{} found)".format(*torch.cuda.get_device_capability())
    return True, "ok"


@functools.cache
def _triton_check():
    return _check_triton()


@functools.cache
def _flash_attention_2_check():
    return _check_flash_attention_2()


class LMFeatures(NamedTuple):
    """The optimizations of the LM decoders. Each one is used wherever it is supported, whatever the others: every combination works.
    CUDA graphs (MMGP's, with any memory profile), the Triton kernels (fused norms and activations, KV cache writes, paged attention,
    DeltaNet), FlashAttention 2. A reason says why one is off. Debug comparisons: WGP_LLM_DISABLE=graphs,triton,flash (any of them)."""
    graphs: bool
    triton: bool
    flash: bool
    graphs_reason: str = ""
    triton_reason: str = ""
    flash_reason: str = ""

    def describe(self):
        items = (("CUDA graphs", self.graphs, self.graphs_reason), ("Triton kernels", self.triton, self.triton_reason), ("FlashAttention 2", self.flash, self.flash_reason))
        return ", ".join(name if used else f"no {name} ({reason})" for name, used, reason in items)


def _disabled_features():
    return {name.strip() for name in os.environ.get("WGP_LLM_DISABLE", "").lower().split(",") if name.strip()}


def resolve_lm_features():
    """The LM decoder optimizations this installation supports (a disabled one is never probed: no Triton launch with triton disabled)."""
    import torch
    disabled, cuda = _disabled_features(), torch.cuda.is_available() and not _is_mps_available()
    def check(name, probe):
        if not cuda:
            return False, "no CUDA GPU"
        if name in disabled:
            return False, "disabled by WGP_LLM_DISABLE"
        return probe()
    graphs, triton, flash = check("graphs", lambda: (True, "")), check("triton", _triton_check), check("flash", _flash_attention_2_check)
    return LMFeatures(graphs[0], triton[0], flash[0], graphs[1], triton[1], flash[1])


@functools.cache
def _prompt_enhancer_triton_check():
    """Check Qwen's block accesses, including their masked lowering on Triton 3.9."""
    global tl
    import torch
    import triton
    import triton.language as tl
    from tqdm import tqdm

    try:
        @triton.jit
        def _smoke_block_pointer_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
            offset = tl.program_id(0) * BLOCK
            source = tl.make_block_ptr(X, (N,), (1,), (offset,), (BLOCK,), (0,))
            target = tl.make_block_ptr(Y, (N,), (1,), (offset,), (BLOCK,), (0,))
            value = tl.load(source, boundary_check=(0,), padding_option="zero")
            tl.store(target, value + 1, boundary_check=(0,))

        if tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 9):
            from shared.kernels.fla_block_pointers import lower_block_pointers
            from shared.kernels.fla_inference import configure_fla_inference_kernels
            lower_block_pointers(_smoke_block_pointer_kernel)
            configure_fla_inference_kernels()
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.arange(129, dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        _smoke_block_pointer_kernel[(2,)](x, y, N=x.numel(), BLOCK=128)
        torch.cuda.synchronize(device)
        if not torch.equal(y, x + 1):
            raise RuntimeError("incorrect output from the block-pointer kernel")
    except Exception as exc:
        message = str(exc).strip().splitlines()
        reason = f"Triton {triton.__version__} block-pointer test failed: {(message[-1] if message else type(exc).__name__)[:260]}"
        tqdm.write(f"[WARNING] {reason}. Triton kernels are disabled for Deepy and the prompt enhancer; using PyTorch kernels (slower). Reinstall the supported Triton version for your GPU/PyTorch profile from the WanGP installation guide.")
        return False, reason
    return True, "ok"


def resolve_prompt_enhancer_lm_features():
    """Apply Qwen's additional compatibility check without disabling Triton for other model families."""
    features = resolve_lm_features()
    if features.triton:
        supported, reason = _prompt_enhancer_triton_check()
        if not supported:
            features = features._replace(triton=False, triton_reason=reason)
    return features


def configure_engine_kernels(model, features):
    """The kernels of the LM engine's layers in model: Triton or PyTorch ones, FlashAttention 2 or PyTorch attention, as features allow
    (CUDA graphs are decided separately)."""
    from .layers.activation import SiluAndMul
    from .layers.attention import Attention
    from .layers.layernorm import RMSNorm
    for module in model.modules():
        if isinstance(module, Attention):
            if not features.flash:
                module.flash_attn_varlen_func = module.flash_attn_with_kvcache = None
            module.use_triton_kv_cache = features.triton
        elif isinstance(module, RMSNorm):
            module.use_triton_rmsnorm = features.triton
        elif isinstance(module, SiluAndMul):
            module.use_triton = features.triton


class TextEngine:
    keep_loaded_for_phase2 = True

    def __init__(self, model, model_path: str, tokenizer, enforce_eager: bool = False, graph_pool_handle=None, kv_cache_int8: bool = False, kv_cache_layout: str = "int8"):
        self.model = model
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.enforce_eager = bool(enforce_eager)
        self.graph_pool_handle = graph_pool_handle
        self.kv_cache_int8 = bool(kv_cache_int8)
        self.kv_cache_layout = str(kv_cache_layout or "int8")
        self._kv_cache_options = {}
        self.hf_config = getattr(model, "config", None)
        self._llm = None
        self._sampling_params_cls = None
        self._max_model_len_hint = None
        self._max_num_seqs_hint = None
        self._max_num_batched_tokens_hint = None
        self._last_failure_reason = ""
        self._disposable = None  # its runtime (KV cache, graphs, state) declared to MMGP as disposable (mmgp.runtime.disposable)

    @staticmethod
    def _compute_runtime_hints(prompt_len: int, max_tokens: int, cfg_scale: float, num_seqs: int = 1):
        max_model_len = max(8, int(prompt_len) + int(max_tokens))
        max_num_seqs = max(1, int(num_seqs)) * (2 if cfg_scale and cfg_scale > 1.0 else 1)
        max_num_batched_tokens = max_model_len * max_num_seqs
        return max_model_len, max_num_seqs, max_num_batched_tokens

    def _get_min_model_len_hint(self):
        min_model_len = getattr(self.model, "_prompt_enhancer_min_model_len_hint", None)
        if min_model_len is None:
            return 8
        try:
            return max(8, int(min_model_len))
        except Exception:
            return 8

    def _ensure_runtime_capacity(self, max_model_len: int, max_num_seqs: int, max_num_batched_tokens: int):
        if self._max_model_len_hint is None:
            self._max_model_len_hint = max_model_len
            self._max_num_seqs_hint = max_num_seqs
            self._max_num_batched_tokens_hint = max_num_batched_tokens
            return

        need_grow = (
            max_model_len > int(self._max_model_len_hint)
            or max_num_seqs > int(self._max_num_seqs_hint)
            or max_num_batched_tokens > int(self._max_num_batched_tokens_hint)
        )
        if not need_grow:
            return

        self._max_model_len_hint = max_model_len
        self._max_num_seqs_hint = max_num_seqs
        self._max_num_batched_tokens_hint = max_num_batched_tokens
        self.close()  # made again larger

    def reserve_runtime(self, prompt_len: int, max_tokens: int, cfg_scale: float, num_seqs: int = 1, min_model_len: int | None = None):
        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            num_seqs=num_seqs,
        )
        req_model_len = max(req_model_len, self._get_min_model_len_hint() if min_model_len is None else int(min_model_len))
        req_num_batched = max(req_num_batched, req_model_len * req_num_seqs)
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)

    def _ensure_llm(self):
        if self._llm is not None:
            return
        try:
            import torch._inductor.config as inductor_config

            if bool(getattr(inductor_config, "split_reductions", False)):
                inductor_config.split_reductions = False
        except Exception:
            pass
        try:
            from shared.llm_engines.lmgp import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"The LM engine is not available: {exc}") from exc
        if not self.model_path:
            raise RuntimeError("The LM engine requires a model_path")
        max_model_len = self._max_model_len_hint or 4096
        max_num_seqs = self._max_num_seqs_hint or 1
        max_num_batched_tokens = self._max_num_batched_tokens_hint or (max_model_len * max_num_seqs)
        hf_config = self.hf_config
        if hf_config is not None and bool(getattr(self.model, "_prompt_enhancer_allow_extended_context", False)):
            requested_max_model_len = int(self._max_model_len_hint or 0)
            configured_max_position_embeddings = int(getattr(hf_config, "max_position_embeddings", 0) or 0)
            if requested_max_model_len > configured_max_position_embeddings:
                hf_config = copy.deepcopy(hf_config)
                hf_config.max_position_embeddings = requested_max_model_len
        self._llm = LLM(
            model=self.model_path,
            enforce_eager=self.enforce_eager,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            hf_config=hf_config,
            tokenizer=self.tokenizer,
            model_object=self.model,
            graph_pool_handle=self.graph_pool_handle,
            kv_cache_int8=self.kv_cache_int8,
            kv_cache_layout=self.kv_cache_layout,
            **self._kv_cache_options,
        )
        self._sampling_params_cls = SamplingParams
        from mmgp import runtime
        # MMGP takes it back only when VRAM runs short while the engine is not in use (its model parked, or idle()): after the blocks of the
        # idle models; the owner of a live context saves it first (close)
        self._disposable = runtime.disposable(self.model, "the language model's KV cache", self.close)

    def has_live_context(self) -> bool:
        """A sequence kept between requests (an assistant's conversation): other requests run beside it (LLMEngine.aside)."""
        scheduler = None if self._llm is None else self._llm.scheduler
        return scheduler is not None and bool(scheduler.running or scheduler.waiting)

    def idle(self):
        """A scope in which the caller's thread runs other models and the engine is not used: MMGP may release its runtime if VRAM runs short."""
        return contextlib.nullcontext() if self._disposable is None else self._disposable.idle()

    def save_context(self) -> None:
        """Before the live context goes (the engine closed, or a request run as alone): its owner, while away (the model's
        _lm_engine_context_saver, e.g. Deepy parked), saves it to RAM, to restore it at its next use."""
        saver = getattr(self.model, "_lm_engine_context_saver", None)
        if saver is not None and self.has_live_context():
            saver()

    def release_runtime_allocations(self):
        if self._llm is None:
            return
        try:
            self._llm.reset_runtime_state()
        except Exception:
            pass

    def runtime_cudagraph_enabled(self) -> bool:
        llm = getattr(self, "_llm", None)
        runner = getattr(llm, "model_runner", None) if llm is not None else None
        enforce_eager = bool(getattr(runner, "enforce_eager", self.enforce_eager))
        return not enforce_eager

    def close(self):
        try:
            self.save_context()  # the live context of an owner away (Deepy parked) goes to RAM first
        finally:
            self._close()

    def _close(self):
        disposable, self._disposable = getattr(self, "_disposable", None), None
        if disposable is not None:
            disposable.close()
        llm = getattr(self, "_llm", None)
        self._llm = None
        if llm is not None:
            try:
                close_fn = getattr(llm, "close", None)
                if callable(close_fn):
                    close_fn()
                else:
                    try:
                        llm.reset_runtime_state()
                    except Exception:
                        pass
                    try:
                        llm.clear_graph_cache()
                    except Exception:
                        pass
                    exit_fn = getattr(llm, "exit", None)
                    if callable(exit_fn):
                        exit_fn()
            except Exception:
                pass
            try:
                del llm
            except Exception:
                pass
        self._sampling_params_cls = None
        try:
            import torch
            # DO NOT add `from torch._inductor import cudagraph_trees` here.
            # Its import registers borrowed Python objects in native TLS on PyTorch
            # 2.10, causing use-after-free and process crashes when a worker exits.
            # Call the same cleanup as its clear_cublass_cache() helper directly.
            # Our CUDA graphs use torch.cuda.CUDAGraph, not Inductor's graph trees.
            torch._C._cuda_clearCublasWorkspaces()
        except Exception:
            pass

    def __del__(self):
        self.close()

    @staticmethod
    def _extract_text_and_tokens(output_obj) -> tuple[str, list[int]]:
        if output_obj is None:
            return "", []
        if isinstance(output_obj, dict):
            text = str(output_obj.get("text", "") or "")
            token_ids = output_obj.get("token_ids", []) or []
            return text, [int(x) for x in token_ids]
        if hasattr(output_obj, "outputs"):
            outputs = getattr(output_obj, "outputs", None)
            if outputs and len(outputs) > 0:
                text = str(getattr(outputs[0], "text", "") or "")
                token_ids = getattr(outputs[0], "token_ids", None)
                if token_ids is None:
                    token_ids = getattr(outputs[0], "token_ids_list", []) or []
                return text, [int(x) for x in token_ids] if token_ids else []
        text = str(getattr(output_obj, "text", "") or "")
        token_ids = getattr(output_obj, "token_ids", None)
        if token_ids is None:
            token_ids = []
        return text, [int(x) for x in token_ids]

    def get_last_failure_reason(self) -> str:
        return self._last_failure_reason

    def generate_text(
        self,
        prompt: str,
        prompt_negative: str,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        callback=None,
        abort_fn: Optional[Callable[[], bool]] = None,
        logits_processor: Optional[Any] = None,
        logits_processor_update_state: Optional[Callable[[int], None]] = None,
        stop_checker: Optional[Callable[[list[int], int], bool]] = None,
        progress_label: str = "LM text",
        release_vram_after: bool = True,
        ignore_eos: bool = False,
    ):
        del stop_checker
        if abort_fn is not None and abort_fn():
            if release_vram_after:
                self.release_runtime_allocations()
            return None
        try:
            prompt_len = len(self.tokenizer.encode(prompt))
        except Exception:
            prompt_len = 0
        if cfg_scale > 1.0 and prompt_negative:
            try:
                prompt_len = max(prompt_len, len(self.tokenizer.encode(prompt_negative)))
            except Exception:
                pass

        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
        )
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)
        self._ensure_llm()
        if self._llm is None:
            return None
        try:
            self._llm.reset()
        except Exception:
            pass

        if callback is not None:
            callback(
                step_idx=-1,
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} 0/{max_tokens}",
                progress_unit="tokens",
            )

        seed_value = None
        if seed is not None:
            try:
                seed_value = int(seed)
            except Exception:
                seed_value = None
            if seed_value is not None and seed_value < 0:
                seed_value = None

        temp = temperature if temperature is not None and temperature > 0 else 1e-5
        sampling_params = self._sampling_params_cls(
            temperature=temp,
            max_tokens=max_tokens,
            cfg_scale=max(cfg_scale, 1.0),
            top_k=top_k if top_k is not None and top_k > 0 else None,
            top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
            ignore_eos=bool(ignore_eos),
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            seed=seed_value,
        )

        text = ""
        token_ids: list[int] = []
        try:
            outputs = self._llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                use_tqdm=True,
                unconditional_prompts=[prompt_negative] if cfg_scale > 1.0 else None,
            )
            if outputs:
                text, token_ids = self._extract_text_and_tokens(outputs[0])
            if (not text) and token_ids:
                try:
                    text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                except Exception:
                    text = ""
            self._last_failure_reason = ""
        except Exception as exc:
            self._last_failure_reason = str(exc)
            raise
        finally:
            if release_vram_after:
                self.release_runtime_allocations()

        if callback is not None:
            callback(
                step_idx=max(0, max_tokens - 1),
                override_num_inference_steps=max_tokens,
                denoising_extra=f"{progress_label} {max_tokens}/{max_tokens}",
                progress_unit="tokens",
            )

        return {"token_ids": token_ids, "text": text}

    def generate_embedded(
        self,
        prompt_token_ids: list[int],
        prompt_embeds,
        prompt_position_ids,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        use_tqdm: bool = True,
        release_vram_after: bool = True,
        ignore_eos: bool = False,
        position_offset: int = 0,
        repetition_penalty: float = 1.0,
        stream_callback=None,
    ):
        results = self.generate_embedded_batch(
            [{"prompt_token_ids": prompt_token_ids, "prompt_embeds": prompt_embeds, "prompt_position_ids": prompt_position_ids, "position_offset": position_offset}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cfg_scale=cfg_scale,
            seed=seed,
            use_tqdm=use_tqdm,
            release_vram_after=release_vram_after,
            ignore_eos=ignore_eos,
            repetition_penalty=repetition_penalty,
            stream_callback=stream_callback,
        )
        return None if not results else results[0]

    def generate_embedded_batch(
        self,
        requests: list[dict],
        *,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        cfg_scale: float,
        seed: Optional[int],
        use_tqdm: bool = True,
        release_vram_after: bool = True,
        ignore_eos: bool = False,
        repetition_penalty: float = 1.0,
        stream_callback=None,
    ):
        """Run several independent embedded prompts concurrently (batched prefill + decode).

        Each request is a dict with `prompt_token_ids`, `prompt_embeds`, `prompt_position_ids`
        and optional `position_offset`. Results are returned in request order.
        """
        if not requests:
            return []
        req_model_len, req_num_seqs, req_num_batched = self._compute_runtime_hints(
            prompt_len=max(len(request["prompt_token_ids"]) for request in requests),
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            num_seqs=len(requests),
        )
        self._ensure_runtime_capacity(req_model_len, req_num_seqs, req_num_batched)
        self._ensure_llm()
        if self._llm is None:
            return None
        try:
            self._llm.reset()
        except Exception:
            pass

        seed_value = None
        if seed is not None:
            try:
                seed_value = int(seed)
            except Exception:
                seed_value = None
            if seed_value is not None and seed_value < 0:
                seed_value = None

        temp = temperature if temperature is not None and temperature > 0 else 1e-5
        sampling_params = self._sampling_params_cls(
            temperature=temp,
            max_tokens=max_tokens,
            cfg_scale=max(cfg_scale, 1.0),
            top_k=top_k if top_k is not None and top_k > 0 else None,
            top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
            ignore_eos=bool(ignore_eos),
            repetition_penalty=float(repetition_penalty or 1.0),
            seed=seed_value,
        )

        results = []
        try:
            outputs = self._llm.generate_embedded(
                prompts=[request["prompt_token_ids"] for request in requests],
                prompt_embeds=[request["prompt_embeds"] for request in requests],
                prompt_position_ids=[request["prompt_position_ids"] for request in requests],
                position_offsets=[int(request.get("position_offset", 0) or 0) for request in requests],
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
                stream_callback=stream_callback,
            )
            for output in outputs or []:
                text, token_ids = self._extract_text_and_tokens(output)
                if (not text) and token_ids:
                    try:
                        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                    except Exception:
                        text = ""
                results.append({"token_ids": token_ids, "text": text})
            self._last_failure_reason = ""
        except Exception as exc:
            self._last_failure_reason = str(exc)
            raise
        finally:
            if release_vram_after:
                self.release_runtime_allocations()

        return results
