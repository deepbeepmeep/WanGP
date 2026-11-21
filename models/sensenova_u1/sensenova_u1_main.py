from __future__ import annotations

from shared.utils.phase_progress import text_encoding_prompts, generation_progress
import os

import torch
from accelerate import init_empty_weights
from PIL import Image
from transformers import AutoTokenizer

from mmgp import offload
from shared.utils import files_locator as fl
from shared.utils.utils import convert_tensor_to_image

from .neo_unify.configuration_neo_chat import NEOChatConfig
from .neo_unify.modeling_neo_chat import NEOChatModel
from .neo_unify.modeling_qwen3 import Qwen3RotaryEmbedding


_ARCHITECTURE = "sensenova_u1_5_8b_mot"
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", f"{_ARCHITECTURE}.json")
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
        **kwargs,
    ):
        filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        config = NEOChatConfig.from_json_file(_CONFIG_PATH)
        with init_empty_weights(include_buffers=True):
            transformer = NEOChatModel(config)
        for module in transformer.modules():
            if isinstance(module, Qwen3RotaryEmbedding):
                module.reset_inv_freq()
        # SenseNova's BF16 residual stream can exceed FP16's finite range.
        # Ignore the shared FP16 setting for loading and MMGP weight conversion.
        dtype = torch.bfloat16
        offload.load_model_data(transformer, filename, writable_tensors=False, default_dtype=dtype)
        transformer._convertWeightsFloatTo = dtype
        transformer.eval().requires_grad_(False)
        transformer._interrupt = False
        if save_quantized:
            from wgp import save_quantized_model

            save_quantized_model(transformer, model_type, filename, dtype, _CONFIG_PATH)

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
        **kwargs,
    ):
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

        with text_encoding_prompts(1 + int(guide_scale != 1.0 if references else guide_scale > 1.0)):
            try:
                if references:
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
        return image.clamp_(-1, 1).to("cpu").transpose(0, 1)

    @property
    def _interrupt(self):
        return bool(getattr(self.transformer, "_interrupt", False))

    @_interrupt.setter
    def _interrupt(self, value):
        value = bool(value)
        self.transformer._interrupt = value
        self.transformer.language_model.model._interrupt = value
