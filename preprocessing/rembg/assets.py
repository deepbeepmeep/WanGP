# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The rembg checkpoints, fetched before reference-image processing: u2net (the default) and BiRefNet General (single objects: the 3D
models, as upstream TRELLIS.2 and TripoSplat; it leaves out cast shadows and keeps thin parts)."""

import hashlib
import os

from shared.utils import files_locator as fl
from shared.utils.download import DownloadError, download_url
from shared.utils.download_progress import check_download_cancelled, resolve_download_gen


OBJECT_MODEL = "birefnet-general"  # model def "background_removal_model" of the models made from one object picture
# rembg's upstream assets and checksums (not mirrored in Wan2.1), saved as rembg names them (<model>.onnx)
MODELS = {
    "u2net": {"url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", "filename": "rembg/u2net.onnx", "md5": "60024c5c889badc19c04ad937298a77b"},
    OBJECT_MODEL: {"url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx", "filename": "rembg/birefnet-general.onnx",
                   "md5": "7a35a0141cbbc80de11d9c9a28f52697"},
}


def query_download_def(model_name="u2net"):
    return MODELS[model_name]


def ensure_assets(gen=None, model_name="u2net"):
    gen = resolve_download_gen(gen)
    check_download_cancelled(gen)
    definition = query_download_def(model_name)
    path = fl.locate_file(definition["filename"], error_if_none=False)
    if path is not None:
        return path
    path = fl.get_download_location(definition["filename"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        download_url(definition["url"], path, gen=gen)
    except OSError as exc:
        raise DownloadError(f"Unable to Download Background Removal Model: {exc}") from exc
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "md5").hexdigest()
    if digest != definition["md5"]:
        os.remove(path)
        raise DownloadError("Background Removal Model Checksum Mismatch; Please Retry")
    return path
