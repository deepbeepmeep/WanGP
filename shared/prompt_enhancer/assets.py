# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any


FLORENCE2_FOLDER = "Florence2"
LLAMA32_FOLDER = "Llama3_2"
LLAMAJOY_FOLDER = "llama-joycaption-beta-one-hf-llava"
PROMPT_ENHANCER_REPO = "DeepBeepMeep/LTX_Video"

FLORENCE2_FILES = ["config.json", "configuration_florence2.py", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"]
LLAMA32_FILES = ["config.json", "generation_config.json", "Llama3_2_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
LLAMAJOY_FILES = ["config.json", "llama_config.json", "llama_joycaption_quanto_bf16_int8.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]

QWEN35_TEXT_GGUF_FILENAME = "Qwen3.5-9B-Abliterated-text-Q4_K_M_bis.gguf"
QWEN35_9B_ABLITERATED_TEXT_INT8_VERSION = 1
QWEN35_TEXT_INT8_FILENAME_V1 = "Qwen3.5-9B-Abliterated_quanto_bf16_int8.safetensors"
QWEN35_TEXT_INT8_FILENAME_V2 = "Qwen3.5-9B-Abliterated_v2_quanto_bf16_int8.safetensors"
QWEN35_TEXT_INT8_FILENAME = QWEN35_TEXT_INT8_FILENAME_V2 if QWEN35_9B_ABLITERATED_TEXT_INT8_VERSION == 2 else QWEN35_TEXT_INT8_FILENAME_V1
QWEN35_9B_MTP_FILENAME = "Qwen3.5-9B-Abliterated_mtp_int8_convrot.safetensors"
QWEN35_VISION_FILENAME = "Qwen3.5-9B-vision_bf16.safetensors"
QWEN35_ABLITERATED_REPO = "DeepBeepMeep/Wan2.1"
QWEN35_ABLITERATED_TEXT_REQUIRED_FILES = (
    "chat_template.jinja",
    "config.json",
)
QWEN35_4B_TEXT_GGUF_FILENAME = "Qwen3.5-4B-Abliterated-text-Q4_K_M.gguf"
QWEN35_4B_VISION_FILENAME = "Qwen3.5-4B-vision_bf16.safetensors"
QWEN35_4B_TEXT_INT8_FILENAME = "Qwen3.5-4B-Abliterated_quanto_bf16_int8.safetensors"
QWEN35_VARIANT_9B = "9b"
QWEN35_VARIANT_4B = "4b"
QWEN38_VARIANT_27B = "27b"
QWEN38_VARIANT_9B = "qwen38_9b"
QWEN38_27B_ASSETS_REPO = "Qwen/Qwen3.8-27B"
QWEN38_27B_GGUF_REPO = QWEN35_ABLITERATED_REPO
QWEN38_27B_GGUF_REPO_SUBFOLDER = "Qwen3_8_27B_Uncensored"
QWEN38_27B_TEXT_GGUF_FILENAME = "Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
QWEN38_27B_TEXT_GGUF_Q3_FILENAME = "Qwen3.8-27B-Uncensored-noMTP-IQ3_S.gguf"
QWEN38_27B_TEXT_GGUF_Q2_FILENAME = "Qwen3.8-27B-Uncensored-IQ2_M.gguf"
QWEN38_27B_TEXT_GGUF_PTQ1_FILENAME = "Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf"
QWEN38_27B_MTP_GGUF_FILENAME = "Qwen3.8-27B-Uncensored-MTP-Q4_K_M.gguf"
QWEN38_9B_TEXT_GGUF_FILENAME = "Qwen3.8-9B-Uncensored-Heretic-Q4_K_M.gguf"
QWEN38_9B_TEXT_GGUF_Q8_FILENAME = "Qwen3.8-9B-Uncensored-Heretic-Q8_0.gguf"
BONSAI2_27B_MTP_GGUF_FILENAME = "Ternary-Bonsai-2-27B-MTP-Q8_0.gguf"
QWEN38_27B_VISION_FILENAME = "Qwen3.8-27B-Uncensored-vision-f16.gguf"
QWEN38_VARIANT_FLASH_NEXT = "flash_next"
QWEN38_FLASH_NEXT_FOLDER = "Qwen3_8_Flash_Next"
QWEN38_FLASH_NEXT_TEXT_FILENAME = "Qwen3.8-Flash-Next-Uncensored-q8g-main-IQ3_S.gguf"
QWEN38_FLASH_NEXT_NGRAM_FILENAME = "Qwen3.8-Flash-Next-ngram-table-IQ4_NL.gguf"
QWEN38_FLASH_NEXT_VISION_FILENAME = "Qwen3.8-Flash-Next-vision-BF16.gguf"
QWEN38_FLASH_NEXT_MTP_FILENAME = "Qwen3.8-Flash-Next-drafter-MTP-Q2_0.safetensors"
QWEN35_VARIANT_SPECS = {
    QWEN35_VARIANT_9B: {
        "display_name": "Qwen3.5-9B Abliterated",
        "assets_dir_name": "Qwen3_5_9B_Abliterated",
        "root_repo": QWEN35_ABLITERATED_REPO,
        "repo_subfolder": "Qwen3_5_9B_Abliterated",
        "root_files": [
            "chat_template.jinja",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ],
        "text_repo": QWEN35_ABLITERATED_REPO,
        "text_required_files": list(QWEN35_ABLITERATED_TEXT_REQUIRED_FILES),
        "text_int8_filename": QWEN35_TEXT_INT8_FILENAME,
        "text_mtp_filename": QWEN35_9B_MTP_FILENAME,
        "text_int8_tie_word_embeddings": QWEN35_9B_ABLITERATED_TEXT_INT8_VERSION == 2,
        "text_int8_log_ssm_a": QWEN35_9B_ABLITERATED_TEXT_INT8_VERSION == 2,
        "text_int8_log_ssm_a_filenames": [QWEN35_TEXT_INT8_FILENAME_V2],
        "gguf_repo": QWEN35_ABLITERATED_REPO,
        "text_gguf_filename": QWEN35_TEXT_GGUF_FILENAME,
        "vision_filename": QWEN35_VISION_FILENAME,
        "tie_word_embeddings": False,
        "supports_mtp": True,
        "mtp_draft_vocab_size": 98304,  # GGUF checkpoint: the MTP head drafts over the first 98304 tokens, as the 27B
        "mtp_speculative_tokens": 2,
        "mtp_speculative_sampling_tokens": 2,
        "mtp_speculative_confidence": 0.0,
    },
    QWEN35_VARIANT_4B: {
        "display_name": "Qwen3.5-4B Abliterated",
        "assets_dir_name": "Qwen3_5_4B_Abliterated",
        "root_repo": QWEN35_ABLITERATED_REPO,
        "repo_subfolder": "Qwen3_5_4B_Abliterated",
        "root_files": [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ],
        "text_repo": None,
        "text_required_files": [],
        "text_int8_filename": QWEN35_4B_TEXT_INT8_FILENAME,
        "gguf_repo": QWEN35_ABLITERATED_REPO,
        "text_gguf_filename": QWEN35_4B_TEXT_GGUF_FILENAME,
        "vision_filename": QWEN35_4B_VISION_FILENAME,
        "tie_word_embeddings": True,
    },
    QWEN38_VARIANT_9B: {
        # Heretic ablation of the Qwen3.8 9B distill (Qwen3.5-9B architecture). It shares the
        # Qwen3.5-9B tokenizer, config and vision tower; its GGUF embeds the MTP layer.
        "display_name": "Qwen3.8-9B Uncensored",
        "assets_dir_name": "Qwen3_5_9B_Abliterated",
        "root_repo": QWEN35_ABLITERATED_REPO,
        "repo_subfolder": "Qwen3_5_9B_Abliterated",
        "root_files": [
            "chat_template.jinja",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ],
        "text_repo": None,
        "text_required_files": [],
        "gguf_repo": QWEN35_ABLITERATED_REPO,
        "gguf_repo_subfolder": QWEN38_27B_GGUF_REPO_SUBFOLDER,
        "text_gguf_filename": QWEN38_9B_TEXT_GGUF_FILENAME,
        "text_gguf_q8_filename": QWEN38_9B_TEXT_GGUF_Q8_FILENAME,
        "vision_filename": QWEN35_VISION_FILENAME,
        "backend": "gguf",
        "tie_word_embeddings": False,
        "supports_mtp": True,
        "mtp_speculative_tokens": 2,
        "mtp_speculative_sampling_tokens": 2,
        "mtp_speculative_confidence": 0.0,
    },
    QWEN38_VARIANT_27B: {
        "display_name": "Qwen3.8-27B Uncensored",
        "assets_dir_name": "Qwen3_8_27B_Uncensored",
        "root_repo": QWEN38_27B_ASSETS_REPO,
        "repo_subfolder": "",
        "root_files": [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ],
        "text_repo": None,
        "text_required_files": [],
        "gguf_repo": QWEN38_27B_GGUF_REPO,
        "gguf_repo_subfolder": QWEN38_27B_GGUF_REPO_SUBFOLDER,
        "text_gguf_filename": QWEN38_27B_TEXT_GGUF_FILENAME,
        "text_gguf_q3_filename": QWEN38_27B_TEXT_GGUF_Q3_FILENAME,
        "text_gguf_q2_filename": QWEN38_27B_TEXT_GGUF_Q2_FILENAME,
        "text_gguf_ptq1_filename": QWEN38_27B_TEXT_GGUF_PTQ1_FILENAME,
        "text_gguf_q3_mtp_filename": QWEN38_27B_MTP_GGUF_FILENAME,
        "text_gguf_ptq1_mtp_filename": BONSAI2_27B_MTP_GGUF_FILENAME,
        "vision_filename": QWEN38_27B_VISION_FILENAME,
        "backend": "gguf",
        "supports_mtp": True,
        "mtp_draft_vocab_size": 98304,
        "mtp_speculative_tokens": 2,
        "mtp_speculative_sampling_tokens": 2,
        "mtp_speculative_confidence": 0.30,
        "tie_word_embeddings": False,
        "vision_budget": 1500,
        "embedding_budget": 2500,
        "llm_budget": 18000,
    },
    QWEN38_VARIANT_FLASH_NEXT: {
        # 125B parameters, much larger than the VRAM of a PC: MMGP keeps its routed experts in RAM with the most used in VRAM, and its n-gram table stays on the drive
        "display_name": "Qwen3.8 Flash-Next",
        "architecture": "qwen4_exp",
        "assets_dir_name": QWEN38_FLASH_NEXT_FOLDER,
        "root_repo": "Qwen/Qwen3.8-Flash-Next",
        "repo_subfolder": "",
        "root_files": [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ],
        "text_repo": None,
        "text_required_files": [],
        "gguf_repo": QWEN35_ABLITERATED_REPO,
        "gguf_repo_subfolder": QWEN38_FLASH_NEXT_FOLDER,
        "text_gguf_q3_filename": QWEN38_FLASH_NEXT_TEXT_FILENAME,
        "text_gguf_q3_mtp_filename": QWEN38_FLASH_NEXT_MTP_FILENAME,
        "text_gguf_extra_files": [QWEN38_FLASH_NEXT_NGRAM_FILENAME],
        "ngram_table_filename": QWEN38_FLASH_NEXT_NGRAM_FILENAME,
        "vision_filename": QWEN38_FLASH_NEXT_VISION_FILENAME,
        "backend": "gguf_q3",
        "supports_mtp": True,
        "mtp_speculative_tokens": 4,
        "mtp_speculative_sampling_tokens": 4,
        "mtp_speculative_confidence": 0.5,
        "mtp_speculative_sampling_confidence": 0.8,
        "mtp_draft_vocab_size": 98304,  # the MTP head drafts over the first 98304 tokens, as the 27B: Han starts at 96000, so that CJK text is verified without drafts
        "tie_word_embeddings": False,
        "vision_budget": 1500,
        "embedding_budget": 2500,
        "llm_budget": 18000,
    },
}


QWEN_CHECKPOINTS = {
    "quanto_int8": ("text_int8_filename", "Quanto INT8"),
    "gguf": ("text_gguf_filename", "GGUF Q4_K_M"),
    "gguf_q8": ("text_gguf_q8_filename", "GGUF Q8_0"),
    "gguf_q3": ("text_gguf_q3_filename", "GGUF IQ3_S"),
    "gguf_q2": ("text_gguf_q2_filename", "GGUF IQ2_M"),
    "gguf_ptq1": ("text_gguf_ptq1_filename", "GGUF PTQ1_0"),
}


def _files(repo, folder, names, target=None):
    definition = {"repoId": repo, "sourceFolderList": [folder], "fileList": [list(names)]}
    if target is not None:
        definition["targetFolderList"] = [target]
    return definition


def qwen_download_defs(spec, backend, speculative_decoding=False):
    folder = spec.get("repo_subfolder", "")
    shared_files = list(spec["root_files"])
    gguf = backend.startswith("gguf")
    key = QWEN_CHECKPOINTS[backend][0] if gguf else "text_int8_filename"
    filename = spec[key]
    checkpoint_folder = spec.get("gguf_repo_subfolder", folder) if gguf else folder
    draft_key = {"gguf_q3": "text_gguf_q3_mtp_filename", "gguf_ptq1": "text_gguf_ptq1_mtp_filename"}.get(backend, "text_mtp_filename")
    draft = [spec[draft_key]] if speculative_decoding and draft_key in spec else []
    defs = []
    if spec["root_repo"] == spec["gguf_repo"]:
        if checkpoint_folder != folder:
            defs.append(_files(spec["gguf_repo"], checkpoint_folder, [filename], spec["assets_dir_name"]))
        shared_files.append(spec["vision_filename"])
        if checkpoint_folder == folder:
            shared_files.append(filename)
        shared_files.extend(draft)
    defs.append(_files(spec["root_repo"], folder, shared_files, None if folder else spec["assets_dir_name"]))
    if spec["root_repo"] != spec["gguf_repo"]:
        defs.append(_files(spec["gguf_repo"], checkpoint_folder, [spec["vision_filename"], filename, *spec.get("text_gguf_extra_files", ()), *draft]))
    if spec.get("text_required_files"):
        defs.append(_files(spec["text_repo"], folder, spec["text_required_files"]))
    return defs


def florence_llama_download_def(enhancer_id):
    folder, files = (LLAMA32_FOLDER, LLAMA32_FILES) if enhancer_id == 1 else (LLAMAJOY_FOLDER, LLAMAJOY_FILES)
    return {"repoId": PROMPT_ENHANCER_REPO, "sourceFolderList": [FLORENCE2_FOLDER, folder], "fileList": [FLORENCE2_FILES, files]}


def checkpoint_inventory():
    """Local model choices, including unselected formats and optional draft weights."""
    from shared.utils.files_locator import installed_copies
    from .block_draft import BLOCK_DRAFT_ASSETS, BLOCK_DRAFT_LABELS, draft_download_def

    for number, label in [(1, "Llama 3.2 3B"), (2, "Llama Joy 8B")]:
        yield {"name": "Florence 2", "variant": label, "download_defs": [florence_llama_download_def(number)]}
    bonsai = "Bonsai 2 27B"
    for spec in QWEN35_VARIANT_SPECS.values():
        for backend, (key, label) in QWEN_CHECKPOINTS.items():
            if key in spec:
                defs = qwen_download_defs(spec, backend)
                # Manually installed GGUF/vision weights can also be resolved by the runtime at the checkpoint root or beside the tokenizer.
                aliases = {}
                for definition in defs:
                    source = definition["sourceFolderList"][0]
                    target = definition.get("targetFolderList", [""])[0]
                    for filename in definition["fileList"][0]:
                        if filename in (spec[key], spec["vision_filename"]):
                            path = "/".join(part for part in (target, source, filename) if part)
                            aliases[path] = [filename, f"{spec['assets_dir_name']}/{filename}"]
                model = bonsai if backend == "gguf_ptq1" else spec["display_name"]
                yield {"name": model, "variant": label, "download_defs": defs, "alternate_paths": aliases}
        for key, model, label in [("text_mtp_filename", spec["display_name"], "MTP"), ("text_gguf_q3_mtp_filename", spec["display_name"], "Q3 MTP"), ("text_gguf_ptq1_mtp_filename", bonsai, "MTP")]:
            if key in spec:
                yield {"name": model, "variant": label, "download_defs": [_files(spec["gguf_repo"], spec["assets_dir_name"], [spec[key]])]}
    for name, spec in BLOCK_DRAFT_ASSETS.items():
        model = bonsai if name.endswith("_bonsai") else QWEN35_VARIANT_SPECS[QWEN38_VARIANT_27B]["display_name"]
        method = BLOCK_DRAFT_LABELS[name.removesuffix("_bonsai")]
        yield {"name": model, "variant": f"{method} Draft", "download_defs": [draft_download_def(spec)]}

    # Earlier 9B weights can remain beside the current releases after an upgrade.
    spec = QWEN35_VARIANT_SPECS[QWEN35_VARIANT_9B]
    for filename, label in [(QWEN35_TEXT_INT8_FILENAME_V1, "Quanto INT8 v1"), (QWEN35_TEXT_INT8_FILENAME_V2, "Quanto INT8 v2"),
                            ("Qwen3.5-9B-Abliterated-text-Q4_K_M.gguf", "GGUF Q4_K_M (original)")]:
        if filename not in (spec["text_int8_filename"], spec["text_gguf_filename"]):
            paths = [f"{spec['assets_dir_name']}/{filename}", filename]
            installed = [path for path in paths if installed_copies(path)]
            if installed:
                yield {"name": spec["display_name"], "variant": label, "files": installed}


def query_prompt_enhancer_download_defs() -> list[dict[str, Any]]:
    return [definition for entry in checkpoint_inventory() for definition in entry.get("download_defs", ())]
