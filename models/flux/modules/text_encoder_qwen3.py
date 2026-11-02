from shared.utils.phase_progress import text_encoding_progress
import os
from functools import lru_cache
from typing import List

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from einops import rearrange
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

from mmgp import offload, runtime
from shared.utils.gguf_mapping import has_standard_gguf_tensor_names, remap_state_dict_triplet

OUTPUT_LAYERS = [9, 18, 27]
MAX_LENGTH = 512
_HF_TOKEN_EMBEDDING_KEY = "model.embed_tokens.weight"


def _qwen3_gguf_mapping_needed(state_dict) -> bool:
    if not state_dict or _HF_TOKEN_EMBEDDING_KEY in state_dict:
        return False
    return has_standard_gguf_tensor_names(state_dict)


@lru_cache(maxsize=4)
def _qwen3_gguf_name_map(config_path: str) -> dict[str, str]:
    config = AutoConfig.from_pretrained(config_path, local_files_only=True)
    with init_empty_weights():
        model = Qwen3ForCausalLM(config)
    from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

    return get_gguf_hf_weights_map(model)


def _build_qwen3_state_dict_preprocessor(config_path: str):
    config_path = os.path.abspath(config_path)

    def preprocess_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
        if not _qwen3_gguf_mapping_needed(state_dict):
            return state_dict, quantization_map, tied_weights_map

        name_map = _qwen3_gguf_name_map(config_path)
        mapped_state_dict, mapped_quantization_map, mapped_tied_weights_map = remap_state_dict_triplet(
            state_dict, quantization_map, tied_weights_map, name_map
        )
        if _HF_TOKEN_EMBEDDING_KEY not in mapped_state_dict:
            return state_dict, quantization_map, tied_weights_map

        return mapped_state_dict, mapped_quantization_map, mapped_tied_weights_map

    return preprocess_state_dict


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str | None = None,
        tokenizer_path: str | None = None,
        torch_dtype: str = "bfloat16",
    ):
        super().__init__()
        file_path = model_spec
        side_files_root = tokenizer_path or os.path.dirname(file_path)
        default_config = os.path.join(side_files_root, "config.json")
        self.model = offload.fast_load_transformers_model(
            file_path,
            writable_tensors=False,
            modelClass=Qwen3ForCausalLM,
            defaultConfigPath=default_config,
            preprocess_sd=_build_qwen3_state_dict_preprocessor(default_config),
        )
        # the token table stays in RAM (mmgp looks up the prompt's rows on the CPU) with the output head, which never runs: the hidden
        # states come from the base model
        runtime.lookup_on_cpu(self.model.model.embed_tokens)
        runtime.unused(self.model.lm_head)

        tokenizer_root = side_files_root
        tokenizer_subdir = os.path.join(tokenizer_root, "tokenizer")
        if os.path.isdir(tokenizer_subdir):
            tokenizer_root = tokenizer_subdir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, trust_remote_code=True)
        self.max_length = MAX_LENGTH

    @torch.no_grad()
    def forward(self, txt: List[str]):
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            all_input_ids.append(torch.tensor(model_inputs["input_ids"], dtype=torch.long, device="cpu").unsqueeze(0))
            all_attention_masks.append(torch.tensor(model_inputs["attention_mask"], dtype=torch.long, device="cpu").unsqueeze(0))

        device = runtime.execution_device(self.model)
        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        with text_encoding_progress(self.model.model.layers, prompt_count=len(txt)):
            output = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")
