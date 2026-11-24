# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Download declaration shared by all consumers of these weights."""
from shared.utils import files_locator as fl
from shared.utils.download import process_files_def_if_needed

DETECTOR_FILE, ESTIMATOR_FILE = "yolox_l_fp16.safetensors", "dw-ll_ucoco_384_fp16.safetensors"
RETIRED_FILES = ("pose/yolox_l.onnx", "pose/dw-ll_ucoco_384.onnx")


def query_download_def():
    return {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": ['pose'], "fileList": [[ESTIMATOR_FILE, DETECTOR_FILE]]}


def query_checkpoint_inventory():
    return [{"name": "DWPose", "download_defs": [query_download_def()]}, {"name": "DWPose", "variant": "Older Releases (ONNX)", "files": list(RETIRED_FILES)}]


def ensure_assets(gen=None):
    process_files_def_if_needed(query_download_def(), gen=gen)
    return fl.locate_file(f"pose/{DETECTOR_FILE}"), fl.locate_file(f"pose/{ESTIMATOR_FILE}")
