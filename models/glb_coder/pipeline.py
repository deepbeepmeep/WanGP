# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from contextlib import ExitStack, contextmanager
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from mmgp import runtime
from tqdm import tqdm

from shared.llm_engines.lmgp import LLM, SamplingParams
from shared.llm_engines.lmgp.utils.context import reset_context
from shared.mesh3d import glb_to_mesh, preview
from shared.prompt_enhancer.qwen35_text import _build_qwen35_gguf_preprocess_sd, _configure_qwen35_text_model, _load_local_text_model
from shared.prompt_enhancer.qwen35_vl import _load_qwen35_tokenizer, _prepare_multimodal_vllm_prompt, _resize_image_for_caption, load_qwen35_vl_prompt_enhancer
from shared.qtypes.gguf import get_gguf_compute_dtype
from shared.utils.cancellation import cancellation_context
from shared.utils.lora_mapping import convert_lora_keys
from shared.utils.phase_progress import GenerationAborted, PhaseProgress, check_abort, generation_progress, set_phase_status
from .export import SceneExportError, _page, export_glb
from .prompts import SYSTEM_PROMPT
from .repair import apply_repair, fix_obvious_syntax, repair_input
from .settings import DEFAULT_SETTINGS, KV_LAYOUTS, MIN_CONTEXT


class GLBCoder:
    def __init__(self, checkpoint, assets_dir, vision_file, lm_features):
        self._interrupt = False
        self.assets_dir = str(assets_dir)
        self.tokenizer = _load_qwen35_tokenizer(self.assets_dir)
        dtype = get_gguf_compute_dtype()
        model = _load_local_text_model(
            checkpoint, str(assets_dir / "config.json"), default_dtype=dtype, materialize_source_tensors=False,
            preprocess_sd=_build_qwen35_gguf_preprocess_sd(tie_output_to_embeddings=True),
            safe_legacy_mode=not lm_features.triton, flash_attention=lm_features.flash,
        )
        # Both CCORE releases use the llama.cpp tiled V-head layout, including the alpha/beta projections and SSM parameters.
        _configure_qwen35_text_model(model, dtype, v_head_reordered=True, ssm_param_reordered=True)
        model._convertWeightsFloatTo = None
        model._lm_engine_mmgp = True
        model._lm_engine_mmgp_graphs = lm_features.graphs
        model._mm_cpu_compute_reserve = 2 << 30
        model._prompt_enhancer_safe_legacy = not lm_features.triton
        module_names = {name for name, _ in model.named_modules()}
        model.preprocess_loras = lambda model_type, sd: convert_lora_keys(sd, module_names)
        self.transformer = model.eval()
        self.vision = None
        if vision_file:
            self.vision, _ = load_qwen35_vl_prompt_enhancer(model_path=vision_file, assets_dir=self.assets_dir, text_model=model, variant="4b")

    def _check_interrupt(self):
        if self._interrupt:
            raise GenerationAborted

    def _prepare(self, prompt, image):
        content = prompt if image is None else [{"type": "image"}, {"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if image is None:
            return self.tokenizer.encode(text, add_special_tokens=False), None, None, 0

        processor = self.vision._prompt_enhancer_processor
        device = runtime.execution_device(self.transformer)
        inputs = processor(text=[text], images=[_resize_image_for_caption(image.convert("RGB"))], return_tensors="pt",
                           return_mm_token_type_ids=True).to(device)
        pixels = inputs.pop("pixel_values")
        tower = self.vision.vision_tower_model
        with PhaseProgress(len(tower.blocks), "Encoding Reference Image", "layers") as progress, ExitStack() as hooks:
            for block in tower.blocks:
                hooks.enter_context(progress.track(block))
            features = self.vision.model.get_image_features(pixels, inputs["image_grid_thw"], return_dict=True).pooler_output
        del pixels
        runtime.ensure_loaded(self.transformer)
        features = [feature.to(dtype=self.transformer.config.dtype) for feature in features]
        return _prepare_multimodal_vllm_prompt(self.vision, inputs, image_features=features)

    @contextmanager
    def _engine(self, context_size, layout):
        runtime.ensure_loaded(self.transformer)
        engine = LLM(self.assets_dir, model_object=self.transformer, hf_config=self.transformer.config, tokenizer=self.tokenizer,
                     enforce_eager=not self.transformer._lm_engine_mmgp_graphs, max_model_len=context_size,
                     max_num_batched_tokens=context_size, max_num_seqs=1, kv_cache_int8=True, kv_cache_layout=layout)
        try:
            with cancellation_context(self._check_interrupt):
                yield engine
        finally:
            engine.close()
            reset_context()
    def _decode(self, engine, inputs, context_size, seed, title):
        ids, embeds, positions, offset = inputs
        output_tokens = context_size - len(ids)
        if output_tokens < 1:
            raise ValueError(f"The prompt uses {len(ids)} tokens. Increase KV Cache Size above that value to generate the asset.")
        params = SamplingParams(seed=seed, temperature=0.8, top_k=40, top_p=0.95, max_tokens=output_tokens)
        with PhaseProgress(output_tokens, title, "tokens") as progress, tqdm(total=output_tokens, desc=title, unit="tok", leave=False) as bar:
            def report(*, token_count, **event):
                progress.advance(token_count - progress.completed)
                bar.update(token_count - bar.n)

            if embeds is None:
                result, = engine.generate([ids], params, use_tqdm=False, stream_callback=report)
            else:
                result, = engine.generate_embedded([ids], [embeds], [positions], params, [offset], use_tqdm=False, stream_callback=report)
        return self.tokenizer.decode(result["token_ids"], skip_special_tokens=True)

    def _export_with_repair(self, page, prompt, seed, context_size, layout):
        diagnostics = []
        for attempt in range(2):
            check_abort()
            try:
                return _export_mesh(page, seed)
            except SceneExportError as exc:
                error = str(exc)
                diagnostics.append(("Original error: " if attempt == 0 else "Model repair error: ") + error)
            fixed = fix_obvious_syntax(page, check_abort)
            if fixed != page:
                page = fixed
                set_phase_status("Checking Local Repair")
                try:
                    return _export_mesh(page, seed)
                except SceneExportError as exc:
                    error = str(exc)
                    diagnostics.append("After local repair: " + error)
            if attempt == 1:
                break
            check_abort()
            set_phase_status("Preparing Scene Repair")
            tqdm.write(f"[GLB Coder] Asking the model to repair the scene: {error}")
            ids, span = repair_input(self.tokenizer, prompt, page, error, context_size)
            with self._engine(context_size, layout) as engine:
                answer = self._decode(engine, (ids, None, None, 0), context_size, seed, "Repairing 3D Asset Code")
            check_abort()
            try:
                page = apply_repair(page, answer, span)
            except SceneExportError as exc:
                diagnostics.append("Model repair error: " + str(exc))
                break
            set_phase_status("Checking Model Repair")
        raise SceneExportError("GLB export failed after one model repair.\n" + "\n".join(diagnostics) + "\nTry a different seed.")

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed, batch_size, original_input_ref_images=None, callback=None, set_progress_status=None, custom_settings=None, **kwargs):
        set_phase_status("Preparing Asset Description")
        settings = DEFAULT_SETTINGS | (custom_settings or {})
        layout, context_size = settings["kv_cache_layout"], int(settings["kv_cache_size"])
        if layout not in KV_LAYOUTS:
            raise ValueError(f"Unknown KV cache quantization: {layout}")
        if not MIN_CONTEXT <= context_size <= self.transformer.config.max_position_embeddings:
            raise ValueError(f"KV cache size must be between {MIN_CONTEXT} and {self.transformer.config.max_position_embeddings} tokens.")
        inputs = self._prepare(input_prompt, original_input_ref_images[0] if original_input_ref_images else None)
        pages = []
        with self._engine(context_size, layout) as engine:
            for index in range(batch_size):
                title = "Generating 3D Asset Code" + (f" {index + 1}/{batch_size}" if batch_size > 1 else "")
                pages.append(self._decode(engine, inputs, context_size, seed + index, title))
        del inputs
        previews, meshes, sources = [], [], []
        for index, page in enumerate(pages):
            check_abort()
            set_phase_status("Exporting GLB")
            page, data, mesh = self._export_with_repair(page, input_prompt, seed + index, context_size, layout)
            check_abort()
            set_phase_status("Rendering 3D Preview")
            previews.append(preview(mesh.to(runtime.execution_device(self.transformer))).cpu())
            meshes.append(data)
            if int(settings["save_source"]):
                sources.append(_page(page, source_seed=seed + index).encode("utf-8"))
        result = {"x": torch.stack(previews, dim=1), "meshes": meshes}
        if len(sources) == 1:
            result["side_files"] = {".html": sources[0]}
        elif sources:
            buffer = BytesIO()
            with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
                for index, source in enumerate(sources):
                    archive.writestr(f"asset_{index + 1}_seed{seed + index}.html", source)
            result["side_files"] = {"_sources.zip": buffer.getvalue()}
        return result


def _export_mesh(page, seed):
    data = export_glb(page, seed, check_abort)
    check_abort()
    try:
        mesh = glb_to_mesh(data)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise SceneExportError(f"Could not read the exported geometry: {exc}") from exc
    if not len(mesh.faces) or not torch.isfinite(mesh.vertices).all() or mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise SceneExportError("The generated GLB contains no usable geometry.")
    return page, data, mesh
