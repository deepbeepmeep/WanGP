# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Background-removal checkpoints, fetched only for requested cutouts."""
from shared.utils import files_locator as fl
from shared.utils.download import process_download_defs
from shared.utils.download_progress import check_download_cancelled, resolve_download_gen
from shared.utils.hf import build_hf_url

OBJECT_MODEL = "birefnet-general"
MODEL_KEY = "background_removal_model"
REPO = "DeepBeepMeep/Wan2.1"
MODEL_CHOICES = [("Default (U²-Net)", "auto"), ("U²-Net", "u2net"),
                 ("BiRefNet General", OBJECT_MODEL), ("Bria RMBG-2.0", "bria-rmbg")]
_server_config = {}
MODELS = {name: {"filename": f"rembg/{filename}", "url": build_hf_url(REPO, "rembg", filename)}
          for name, filename in (("u2net", "u2net_bf16.safetensors"),
                                 (OBJECT_MODEL, "birefnet-general_bf16.safetensors"),
                                 ("bria-rmbg", "bria-rmbg2_bf16.safetensors"))}


def configure(server_config):
    global _server_config
    _server_config = server_config


def selected_model(default="u2net", server_config=None):
    # Object reconstruction requires BiRefNet independently of the image preference.
    if default == OBJECT_MODEL:
        return OBJECT_MODEL
    selection = (_server_config if server_config is None else server_config).get(MODEL_KEY, "auto")
    if selection == "auto":
        return default
    if selection not in MODELS:
        raise ValueError(f"Unknown Background Removal Model: {selection}")
    return selection


def query_download_def(model_name="u2net"):
    return MODELS[model_name]


def query_download_defs(model_name="u2net", precision="bf16"):
    filename = MODELS[model_name]["filename"].split("/")[-1]
    if precision == "int8_convrot" and model_name != "u2net":
        filename = filename.replace("_bf16.safetensors", "_int8_convrot.safetensors")
    elif precision != "bf16":
        raise ValueError(f"Unsupported Background Remover Precision: {model_name} {precision}")
    definitions = [{"repoId": REPO, "sourceFolderList": ["rembg"],
                    "fileList": [[filename]]}]
    return definitions


def query_checkpoint_inventory():
    entries = [{"name": "Background Removal", "variant": label, "download_defs": query_download_defs(name)}
               for label, name in MODEL_CHOICES if name != "auto"]
    entries.extend({"name": "Background Removal", "variant": f"{label} (INT8 ConvRot)",
                    "download_defs": query_download_defs(name, "int8_convrot")}
                   for label, name in MODEL_CHOICES if name in (OBJECT_MODEL, "bria-rmbg"))
    entries.append({"name": "Background Removal", "variant": "Older Releases (ONNX)",
                    "files": [f"rembg/{name}.onnx" for name in MODELS]})
    return entries


def ensure_assets(gen=None, model_name="u2net"):
    gen = resolve_download_gen(gen)
    check_download_cancelled(gen)
    process_download_defs(query_download_defs(model_name), gen=gen)
    return fl.locate_file(MODELS[model_name]["filename"])
