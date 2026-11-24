# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Face analysis checkpoints (InsightFace buffalo_l networks and facexlib's IR-SE50 ArcFace), fetched only for requested tasks."""
from shared.utils import files_locator as fl
from shared.utils.download import process_download_defs
from shared.utils.download_progress import check_download_cancelled, resolve_download_gen

REPO = "DeepBeepMeep/Wan2.1"
FOLDER = "face_analysis"
FILES = {"detection": "scrfd_10g_bf16.safetensors", "landmark_2d_106": "landmark_2d106_bf16.safetensors",
         "recognition": "arcface_w600k_r50_bf16.safetensors", "recognition_ir_se50": "arcface_ir_se50_bf16.safetensors"}
RETIRED_FILES = ["buffalo_l/det_10g.onnx", "buffalo_l/2d106det.onnx", "buffalo_l/w600k_r50.onnx"]


def query_download_def(tasks=tuple(FILES)):
    return {"repoId": REPO, "sourceFolderList": [FOLDER], "fileList": [[FILES[task] for task in tasks]]}


def query_checkpoint_inventory():
    return [{"name": "Face Analysis", "download_defs": [query_download_def()]},
            {"name": "Face Analysis", "variant": "Older Releases (ONNX)", "files": RETIRED_FILES}]


def ensure_assets(tasks, gen=None):
    gen = resolve_download_gen(gen)
    check_download_cancelled(gen)
    process_download_defs(query_download_def(tasks), gen=gen)
    return {task: fl.locate_file(f"{FOLDER}/{FILES[task]}") for task in tasks}
