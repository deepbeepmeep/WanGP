# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Download declaration shared by all consumers of these weights."""

FILENAME = "model_bs_roformer_ep_317_sdr_12.9755_bf16.safetensors"
RETIRED_FILES = ["model_bs_roformer_ep_317_sdr_12.9755.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.yaml", "download_checks.json"]


def query_download_def():
    return {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": ["roformer"], "fileList": [[FILENAME]]}


def query_checkpoint_inventory():
    return [{"name": "Vocals Separation", "download_defs": [query_download_def()]},
            {"name": "Vocals Separation", "variant": "Older Releases (audio-separator)", "files": [f"roformer/{name}" for name in RETIRED_FILES]}]
