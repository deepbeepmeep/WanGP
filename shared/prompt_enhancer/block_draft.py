# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional parallel draft checkpoints for Qwen3.8/Bonsai 27B."""
import json

from .config import BLOCK_DRAFT_METHODS


BLOCK_DRAFT_ASSETS = {
    "dspark": {"folder": "Qwen3_8_27B_Uncensored", "config": "Qwen3_8_27B_DSpark.config.json", "weights": "Qwen3_8_27B_DSpark_int8_convrot.safetensors", "drafts": 7},
    "dflash2": {"folder": "Qwen3_8_27B_Uncensored", "config": "Qwen3_8_27B_DFlash2.config.json", "weights": "Qwen3_8_27B_DFlash2_int8_convrot.safetensors", "drafts": 7},
    "dflash2_bonsai": {"folder": "Qwen3_8_27B_Uncensored", "config": "Bonsai_2_27B_DFlash2.config.json", "weights": "Bonsai_2_27B_DFlash2_int8_convrot.safetensors", "drafts": 5},
    "dspine": {"folder": "Qwen3_8_27B_Uncensored", "config": "Qwen3_8_27B_DSpine.config.json", "weights": "Qwen3_8_27B_DSpine_int8_convrot.safetensors", "drafts": 7},
}
BLOCK_DRAFT_LABELS = {"dspark": "DSpark", "dflash2": "DFlash2", "dspine": "DSpine"}


def block_draft_spec(method, *, bonsai=False):
    if method == "dspine" and bonsai:
        raise ValueError("DSpine drafts for Qwen3.8 27B only: its refinement reads the target's output rows, which Bonsai stores rotated.")
    return BLOCK_DRAFT_ASSETS["dflash2_bonsai" if method == "dflash2" and bonsai else method]


def draft_download_def(spec):
    return dict(repoId="DeepBeepMeep/Wan2.1", sourceFolderList=[spec["folder"]], fileList=[[spec["config"], spec["weights"]]])


def ensure_block_draft_assets(process_files_def, method, variant, backend):
    if variant != "27b":
        raise ValueError("Parallel draft models require Qwen3.8-27B or Bonsai 2 27B.")
    spec = block_draft_spec(method, bonsai=backend == "gguf_ptq1")
    process_files_def(**draft_download_def(spec))


def _load_draft_config(config_path):
    from transformers import Qwen3Config

    with open(config_path, encoding="utf-8") as f:
        values = json.load(f)
    # Recent exports put RoPE in rope_parameters. Supported Transformers
    # versions also include Qwen3 implementations that read only these legacy
    # fields; otherwise DFlash2 silently uses theta=10,000 instead of 10,000,000.
    if "rope_parameters" in values:
        rope = values["rope_parameters"]
        values["rope_theta"] = rope["rope_theta"]
        values["rope_scaling"] = {key: value for key, value in rope.items() if key != "rope_theta"}
    return Qwen3Config.from_dict(values)


def install_block_draft(model, method, *, bonsai=False):
    import torch
    from mmgp import offload, quant_router
    from shared.utils import files_locator
    from shared.qtypes.prism import preserve_checkpoint_dtypes
    from shared.llm_engines.lmgp.models.block_draft import BlockDraft

    spec = block_draft_spec(method, bonsai=bonsai)
    config_path = files_locator.locate_file(f"{spec['folder']}/{spec['config']}")
    config = _load_draft_config(config_path)
    if config.hidden_size != model.config.hidden_size or config.vocab_size != model.config.vocab_size:
        raise ValueError("Draft checkpoint hidden size/vocabulary does not match the target model.")
    print(f"Loading - {BLOCK_DRAFT_LABELS[method]} {'Bonsai 2' if bonsai and method == 'dflash2' else 'Qwen3.8'} Draft Model")
    if spec["weights"].endswith("_int8_convrot.safetensors"):
        quant_router.register_handler("shared.qtypes.int8_convrot")
    with torch.device("meta"):
        draft = BlockDraft(config, method, block_size=spec["drafts"] + 1 if method == "dspine" else None)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    with torch.device("cpu"):
        draft.rotary_emb = Qwen3RotaryEmbedding(config)
    offload.load_model_data(draft, files_locator.locate_file(f"{spec['folder']}/{spec['weights']}"), writable_tensors=False, default_dtype=torch.bfloat16, pre_load_callback=preserve_checkpoint_dtypes)
    draft.eval()
    model.mtp = draft
    model._block_draft_asset_folder = spec["folder"]
    model._block_draft_max_tokens = spec["drafts"]
    model._block_draft = True
    model._prompt_enhancer_speculative_decoding = True
    model._prompt_enhancer_speculative_tokens = spec["drafts"]
    model._prompt_enhancer_speculative_sampling_tokens = spec["drafts"]
