# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""SeedVC weights, modes and defaults. The SeedVC engine itself is a separate GPL-3.0 component (see component.py)."""
import os

SEEDVC_MODE_SPEECH = 1
SEEDVC_MODE_SINGING = 2
SEEDVC_MODE_ACCENT = 3

SEEDVC_CAMPPLUS_FILENAME = "campplus_cn_common.bin"
SEEDVC_SPEECH_CHECKPOINT_FILENAME = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
SEEDVC_SPEECH_CONFIG_FILENAME = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
SEEDVC_SINGING_CHECKPOINT_FILENAME = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
SEEDVC_SINGING_CONFIG_FILENAME = "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
SEEDVC_RMVPE_FILENAME = "rmvpe.pt"
SEEDVC_V2_AR_CHECKPOINT_FILENAME = "v2/ar_base.pth"
SEEDVC_V2_CFM_CHECKPOINT_FILENAME = "v2/cfm_small.pth"
SEEDVC_V2_NARROW_CHECKPOINT_FILENAME = "bsq32/bsq32_light.pth"
SEEDVC_V2_WIDE_CHECKPOINT_FILENAME = "bsq2048/bsq2048_light.pth"

SEEDVC_REPO_ID = "DeepBeepMeep/LTX-2"
SEEDVC_ROOT = "seed-vc"
SEEDVC_BIGVGAN_DIR = "bigvgan_v2_22khz_80band_256x"
SEEDVC_BIGVGAN_44K_DIR = "bigvgan_v2_44khz_128band_512x"
SEEDVC_WHISPER_DIR = "whisper-small"
SEEDVC_HUBERT_DIR = "hubert-large-ll60k"
SEEDVC_BIGVGAN_FILES = ["config.json", "bigvgan_generator.pt"]
SEEDVC_WHISPER_FILES = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
SEEDVC_HUBERT_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]

_MODE_DEFAULTS = {
    SEEDVC_MODE_SPEECH: {"label": "v1.0 Speech", "steps": 25, "cfg_rate": 0.5},
    SEEDVC_MODE_SINGING: {"label": "v1.0 Singing", "steps": 10, "cfg_rate": 0.7},
    SEEDVC_MODE_ACCENT: {"label": "v2 Speech", "steps": 30, "cfg_rate": 0.7},
}


def normalize_mode(mode: int | str | None) -> int:
    try:
        mode = int(mode or SEEDVC_MODE_SPEECH)
    except (TypeError, ValueError):
        mode = SEEDVC_MODE_SPEECH
    return mode if mode in _MODE_DEFAULTS else SEEDVC_MODE_SPEECH


def get_default_steps(mode: int | str | None = SEEDVC_MODE_SPEECH) -> int:
    return int(_MODE_DEFAULTS[normalize_mode(mode)]["steps"])


def get_default_cfg_rate(mode: int | str | None = SEEDVC_MODE_SPEECH) -> float:
    return float(_MODE_DEFAULTS[normalize_mode(mode)]["cfg_rate"])


def query_required_files(mode: int | str | None = SEEDVC_MODE_SPEECH, root: str = SEEDVC_ROOT) -> list[str]:
    mode = normalize_mode(mode)
    if mode == SEEDVC_MODE_SINGING:
        return [
            os.path.join(root, SEEDVC_SINGING_CHECKPOINT_FILENAME),
            os.path.join(root, SEEDVC_SINGING_CONFIG_FILENAME),
            os.path.join(root, SEEDVC_CAMPPLUS_FILENAME),
            os.path.join(root, SEEDVC_RMVPE_FILENAME),
            *[os.path.join(SEEDVC_BIGVGAN_44K_DIR, filename) for filename in SEEDVC_BIGVGAN_FILES],
            *[os.path.join(SEEDVC_WHISPER_DIR, filename) for filename in SEEDVC_WHISPER_FILES],
        ]
    if mode == SEEDVC_MODE_ACCENT:
        return [
            os.path.join(root, SEEDVC_V2_AR_CHECKPOINT_FILENAME),
            os.path.join(root, SEEDVC_V2_CFM_CHECKPOINT_FILENAME),
            os.path.join(root, SEEDVC_V2_NARROW_CHECKPOINT_FILENAME),
            os.path.join(root, SEEDVC_V2_WIDE_CHECKPOINT_FILENAME),
            os.path.join(root, SEEDVC_CAMPPLUS_FILENAME),
            *[os.path.join(SEEDVC_BIGVGAN_DIR, filename) for filename in SEEDVC_BIGVGAN_FILES],
            *[os.path.join(SEEDVC_WHISPER_DIR, filename) for filename in SEEDVC_WHISPER_FILES],
            *[os.path.join(SEEDVC_HUBERT_DIR, filename) for filename in SEEDVC_HUBERT_FILES],
        ]
    return [
        os.path.join(root, SEEDVC_SPEECH_CHECKPOINT_FILENAME),
        os.path.join(root, SEEDVC_SPEECH_CONFIG_FILENAME),
        os.path.join(root, SEEDVC_CAMPPLUS_FILENAME),
        *[os.path.join(SEEDVC_BIGVGAN_DIR, filename) for filename in SEEDVC_BIGVGAN_FILES],
        *[os.path.join(SEEDVC_WHISPER_DIR, filename) for filename in SEEDVC_WHISPER_FILES],
    ]


def query_download_def(mode: int | str | None = SEEDVC_MODE_SPEECH, root: str = SEEDVC_ROOT) -> list[dict]:
    mode = normalize_mode(mode)
    root_files = [SEEDVC_CAMPPLUS_FILENAME]
    if mode == SEEDVC_MODE_SINGING:
        root_files += [SEEDVC_SINGING_CHECKPOINT_FILENAME, SEEDVC_SINGING_CONFIG_FILENAME, SEEDVC_RMVPE_FILENAME]
        bigvgan_dir = SEEDVC_BIGVGAN_44K_DIR
    elif mode == SEEDVC_MODE_ACCENT:
        root_files += [
            SEEDVC_V2_AR_CHECKPOINT_FILENAME,
            SEEDVC_V2_CFM_CHECKPOINT_FILENAME,
            SEEDVC_V2_NARROW_CHECKPOINT_FILENAME,
            SEEDVC_V2_WIDE_CHECKPOINT_FILENAME,
        ]
        bigvgan_dir = SEEDVC_BIGVGAN_DIR
    else:
        root_files += [SEEDVC_SPEECH_CHECKPOINT_FILENAME, SEEDVC_SPEECH_CONFIG_FILENAME]
        bigvgan_dir = SEEDVC_BIGVGAN_DIR

    download_def = [
        {"repoId": SEEDVC_REPO_ID, "sourceFolderList": [root], "fileList": [root_files]},
        {"repoId": SEEDVC_REPO_ID, "sourceFolderList": [bigvgan_dir], "fileList": [SEEDVC_BIGVGAN_FILES]},
        {"repoId": SEEDVC_REPO_ID, "sourceFolderList": [SEEDVC_WHISPER_DIR], "fileList": [SEEDVC_WHISPER_FILES]},
    ]
    if mode == SEEDVC_MODE_ACCENT:
        download_def.append({"repoId": SEEDVC_REPO_ID, "sourceFolderList": [SEEDVC_HUBERT_DIR], "fileList": [SEEDVC_HUBERT_FILES]})
    return download_def
