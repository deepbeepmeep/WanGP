from __future__ import annotations

from shared.utils.phase_progress import text_encoding_prompts, generation_progress
import os

import torch
from tqdm import tqdm
from accelerate import init_empty_weights
from PIL import Image
from transformers import AutoTokenizer

from mmgp import offload, runtime
from shared.utils import files_locator as fl
from shared.utils.utils import convert_tensor_to_image

from .neo_unify.configuration_neo_chat import NEOChatConfig
from .neo_unify.modeling_neo_chat import NEOChatModel
from .neo_unify.modeling_neo_vit import NEOVisionEmbeddings
from .neo_unify.modeling_qwen3 import Qwen3RotaryEmbedding


_CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), "configs")
_MAX_IMAGES = 4  # self-correction: the draft and up to 3 fixes (upstream: 10 images; each round is a whole generation)
_TOKENIZER_FOLDER = "sensenova_u1_5"


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        return convert_tensor_to_image(image).convert("RGB")
    raise TypeError(f"Unsupported SenseNova-U1 reference image type: {type(image)}")


class model_factory:
    def __init__(
        self,
        model_filename,
        model_type=None,
        base_model_type=None,
        dtype=torch.bfloat16,
        save_quantized=False,
        vision_rope_cpu_tables=False,
        **kwargs,
    ):
        filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        config_path = os.path.join(_CONFIGS_FOLDER, f"{base_model_type}.json")
        config = NEOChatConfig.from_json_file(config_path)
        with init_empty_weights(include_buffers=True):
            transformer = NEOChatModel(config)
        for module in transformer.modules():
            if isinstance(module, Qwen3RotaryEmbedding):
                module.reset_inv_freq()
            elif isinstance(module, NEOVisionEmbeddings):
                module.cpu_rope_tables = vision_rope_cpu_tables
        # SenseNova's BF16 residual stream can exceed FP16's finite range.
        # Ignore the shared FP16 setting for loading and MMGP weight conversion.
        dtype = torch.bfloat16
        offload.load_model_data(transformer, filename, writable_tensors=False, default_dtype=dtype)
        transformer._convertWeightsFloatTo = dtype
        runtime.lookup_on_cpu(transformer.language_model.get_input_embeddings())  # prompt rows gathered in RAM, the same values: 1.24 GB never in VRAM
        transformer.eval().requires_grad_(False)
        transformer._interrupt = False
        if save_quantized:
            from wgp import save_quantized_model

            save_quantized_model(transformer, model_type, filename, dtype, config_path)

        tokenizer_config = fl.locate_file(os.path.join(_TOKENIZER_FOLDER, "tokenizer_config.json"))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(tokenizer_config))
        self.transformer = transformer
        self.model = transformer
        self.base_model_type = base_model_type

    @generation_progress
    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "",
        sampling_steps: int = 50,
        guide_scale: float = 4.0,
        width: int = 1024,
        height: int = 1024,
        batch_size: int = 1,
        shift: float = 3.0,
        image_start=None,
        input_ref_images=None,
        callback=None,
        loras_slists=None,
        set_progress_status=None,
        custom_settings=None,
        **kwargs,
    ):
        self_correction = (custom_settings or {}).get("self_correction", "Disabled")
        if loras_slists is not None:
            from shared.utils.loras_mutipliers import update_loras_slists

            update_loras_slists(self.transformer, loras_slists, sampling_steps)

        references = list(input_ref_images) if input_ref_images is not None else []
        if image_start is not None and not references:
            references.append(image_start)
        references = [_as_pil(image) for image in references]

        def step_callback(step_idx, image):
            if callback is not None:
                if step_idx == -1:
                    callback(-1, None, True, override_num_inference_steps=sampling_steps)
                    return
                preview_height = min(200, image.shape[-2])
                preview_width = max(1, round(image.shape[-1] * preview_height / image.shape[-2]))
                preview = torch.nn.functional.interpolate(image, size=(preview_height, preview_width), mode="bilinear", align_corners=False) if image.shape[-2:] != (preview_height, preview_width) else image
                callback(step_idx, preview.float().transpose(0, 1), False)  # on the GPU: WanGP moves it to RAM without waiting for it

        critiques, suffixes = [], []
        with text_encoding_prompts(1 + int(guide_scale != 1.0 if references else guide_scale > 1.0)):
            try:
                if self_correction != "Disabled":
                    outputs = []
                    for index in range(batch_size):
                        drawn, segments = self.transformer.self_correction_generate(self.tokenizer, input_prompt, references, cfg_scale=guide_scale, timestep_shift=shift, image_size=(width, height),
                                                                                    num_steps=sampling_steps, seed=(seed or 0) + index, callback=step_callback, always_review=self_correction.startswith("Always"),
                                                                                    max_images=_MAX_IMAGES)
                        # files: the draft, then its fixes "_fixed", "_fixed2"...; without the draft, the last version only
                        base = "" if index == 0 else f"_{index}"
                        kept = range(len(drawn)) if self_correction.endswith("Keep Draft") else [len(drawn) - 1]
                        outputs += [drawn[round_no] for round_no in kept]
                        suffixes += [base if round_no == 0 or len(kept) == 1 else base + "_fixed" + ("" if round_no == 1 else str(round_no)) for round_no in kept]
                        # segments[i] follows drawn[i - 1]: the review of the draft, then of each fix
                        reviews = [(round_no, text) for round_no, text in enumerate(segments[1:]) if text]
                        if reviews and len(drawn) < _MAX_IMAGES and not segments[-1]:  # the last fix was reviewed too
                            reviews.append((len(drawn) - 1, "Nothing more to fix."))
                        if len(reviews) > 1:
                            critique = "\n\n".join(f"Review of {'the draft' if round_no == 0 else 'fix ' + str(round_no)}:\n{text}" for round_no, text in reviews)
                        else:
                            critique = reviews[0][1] if reviews else "No correction: the review of the draft found nothing to fix."
                        if len(drawn) == 1:
                            tqdm.write(f"Self-Correction: {critique}")
                        critiques.append(critique)
                    image = torch.cat(outputs)
                    del outputs, drawn
                elif references:
                    image = self.transformer.it2i_generate(
                        self.tokenizer,
                        input_prompt,
                        references,
                        cfg_scale=guide_scale,
                        img_cfg_scale=1.0,
                        timestep_shift=shift,
                        image_size=(width, height),
                        num_steps=sampling_steps,
                        batch_size=batch_size,
                        seed=seed or 0,
                        callback=step_callback,
                    )
                else:
                    image = self.transformer.t2i_generate(
                        self.tokenizer,
                        input_prompt,
                        cfg_scale=guide_scale,
                        timestep_shift=shift,
                        image_size=(width, height),
                        num_steps=sampling_steps,
                        batch_size=batch_size,
                        seed=seed or 0,
                        callback=step_callback,
                    )
            except InterruptedError:
                return None
        if self._interrupt:
            return None
        image = image.clamp_(-1, 1).to("cpu").transpose(0, 1)
        if not critiques:
            return image
        critique = critiques[0] if len(critiques) == 1 else "\n\n".join(f"Image {index + 1}:\n{text}" for index, text in enumerate(critiques))
        return {"x": image, "side_files": {"_critique.txt": critique.encode("utf-8")}, "image_suffixes": suffixes}

    @property
    def _interrupt(self):
        return bool(getattr(self.transformer, "_interrupt", False))

    @_interrupt.setter
    def _interrupt(self, value):
        value = bool(value)
        self.transformer._interrupt = value
        self.transformer.language_model.model._interrupt = value
