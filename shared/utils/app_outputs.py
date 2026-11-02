# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Apps gallery files (.wgpapp): web apps made with Deepy, as zip packages of their bundle (shared.apps.bundles), with their thumbnail beside
them (<stem>.png). The WanGP settings of the publication live inside the package as wangp_settings.json."""
import json
import os
import zipfile


def preview_path(path):
    return os.path.splitext(path)[0] + ".png"


def extract_thumbnail(path):
    """An imported package's thumbnail, written beside it for the Apps gallery strip."""
    target = preview_path(path)
    with zipfile.ZipFile(path) as archive:
        if "thumbnail.png" in archive.namelist() and not os.path.isfile(target):
            with open(target, "wb") as handle:
                handle.write(archive.read("thumbnail.png"))


def save_app_metadata(path, settings):
    """Stores the publication settings in the package (once)."""
    with zipfile.ZipFile(path) as archive:
        if "wangp_settings.json" in archive.namelist():
            return
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("wangp_settings.json", json.dumps(settings, ensure_ascii=False))


def read_app_metadata(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        settings = json.loads(archive.read("wangp_settings.json")) if "wangp_settings.json" in names else {}
        manifest = json.loads(archive.read("app.json")) if "app.json" in names else {}
        source = json.loads(archive.read("source.json")) if "source.json" in names else {}
    return {**settings, "app_name": manifest.get("name", ""), "app_description": manifest.get("description", ""), "app_version": source.get("version", manifest.get("version")), "app_controls": manifest.get("controls", {}), "app_note": source.get("note", "")}
