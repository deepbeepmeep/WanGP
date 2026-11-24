# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Speaker-separation checkpoints, fetched only when a dialogue is split into speakers."""
REPO = "DeepBeepMeep/Wan2.1"
FOLDER = "speaker_separator"
SEGMENTATION_FILE = f"{FOLDER}/pyannote_segmentation_3.0_fp32.safetensors"  # FP32: BF16 flips 0.4-0.9 % of the frame decisions
EMBEDDING_FILE = f"{FOLDER}/eres2net_base_3dspeaker_16k_bf16.safetensors"
# Former pyannote.audio and sherpa-onnx releases, listed for deletion in Models Manager
RETIRED_FILES = ["pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", "pyannote/pytorch_model_segmentation-3.0.bin",
                 "sherpa/sherpa-onnx-pyannote-segmentation-3-0/model.onnx", "sherpa/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"]


def query_speaker_separator_download_def():
    return [{"repoId": REPO, "sourceFolderList": [FOLDER], "fileList": [[SEGMENTATION_FILE.split("/")[-1], EMBEDDING_FILE.split("/")[-1]]]}]


def query_checkpoint_inventory():
    return [{"name": "Speaker Separation", "download_defs": query_speaker_separator_download_def()},
            {"name": "Speaker Separation", "variant": "Older Releases", "files": RETIRED_FILES}]


def download_speaker_separator(send_cmd=None, status_text="Downloading Speaker Separator Model Files", gen=None, process_files=None):
    from shared.utils.download import process_files_def_if_needed
    return process_files_def_if_needed(query_speaker_separator_download_def(), send_cmd=send_cmd, status_text=status_text, gen=gen, process_files=process_files)
