# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Output galleries: one GallerySource per gallery tab.

A new kind of output (for instance text) gets its own gallery by adding a GallerySource here. Generation state keys, the gallery
tabs, imports, workspaces and the Deepy views iterate over GALLERY_SOURCES instead of naming the galleries.
"""
from __future__ import annotations

from dataclasses import dataclass

MESH_EXTENSIONS = (".glb", ".ply")  # 3D gallery files: GLB meshes and Gaussian splat PLYs; other mesh formats are converted to GLB on import
MESH_IMPORT_EXTENSIONS = (".glb", ".obj", ".ply", ".stl")
WORLD_EXTENSIONS = (".wgpworld",)  # Worlds gallery files: worlds of interactive world models, explorable again from their stored state
APP_EXTENSIONS = (".wgpapp",)  # Apps gallery files: web apps made with Deepy (shared.apps), played again from their package


@dataclass(frozen=True)
class GallerySource:
    id: str             # internal id, used in gallery views, workspace payloads and element ids
    label: str          # user-facing gallery name
    kinds: tuple        # media kinds (shared.utils.utils.get_media_kind) shown in this gallery
    prefix: str         # prefix of the generation state keys
    tab_id: str         # Gradio tab id
    tab_label: str      # Gradio tab label
    accept: str         # file input "accept" filter for imports
    media_ids: str      # prefix of the Deepy gallery media ids ("<prefix>:<hash>")

    @property
    def files_key(self):
        return self.prefix + "file_list"

    @property
    def settings_key(self):
        return self.prefix + "file_settings_list"

    @property
    def selected_key(self):
        return self.prefix + "selected"

    @property
    def last_selected_key(self):
        return self.prefix + "last_selected"


VIDEO = GallerySource("video", "Images / Videos", ("image", "video"), "", "video_images", "Video / Images", "image/*,video/*", "visual")
# Audio imports filter by extension: iOS Safari treats audio/* as video and greys out audio files.
AUDIO = GallerySource("audio", "Audio", ("audio",), "audio_", "audio", "Audio", ".wav,.mp3,.aac,.flac,.m4a,.ogg,.opus,.wma", "audio")
MESH = GallerySource("mesh", "3D", ("3d",), "mesh_", "mesh", "3D", ".glb,.obj,.ply,.stl", "3d")
WORLD = GallerySource("world", "Worlds", ("world",), "world_", "world", "Worlds", ".wgpworld", "world")
APP = GallerySource("app", "Apps", ("app",), "app_", "apps", "Apps", ".wgpapp", "app")

GALLERY_SOURCES = (VIDEO, AUDIO, MESH, WORLD, APP)
STRIP_SOURCES = GALLERY_SOURCES[1:]  # galleries rendered by shared.gradio.strip_gallery (all but the image/video gallery)
SOURCES = {source.id: source for source in GALLERY_SOURCES}
_KIND_SOURCES = {kind: source for source in GALLERY_SOURCES for kind in source.kinds}


def gallery_source(source_id):
    return SOURCES[source_id]


def source_index(source_id):
    return GALLERY_SOURCES.index(SOURCES[source_id])


def source_for_kind(kind):
    return _KIND_SOURCES.get(kind, VIDEO)


def source_for_path(path):
    from shared.utils.utils import get_media_kind
    return source_for_kind(get_media_kind(path))


def gallery_state_defaults(selected=-1):
    """Gallery keys of a new generation state (lists, selections and the active gallery)."""
    state = {"current_gallery_source": VIDEO.id, "last_gallery_source": VIDEO.id}
    for source in GALLERY_SOURCES:
        state.update({source.files_key: [], source.settings_key: [], source.selected_key: selected, source.last_selected_key: True})
    return state
