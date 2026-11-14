# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from gradio.data_classes import FileData

from shared.utils.video_decode import _build_hdr_tonemap_filter, probe_video_stream_metadata, resolve_media_binary
from shared.utils.subprocess_stdin import child_kwargs


_BROWSER_VIDEO_EXTENSIONS = {".mp4", ".webm", ".ogg", ".ogv"}
_BROWSER_PLAYABLE_VIDEO_CODECS = {
    (".mp4", "h264"),
    (".mp4", "h265"),
    (".mp4", "hevc"),
    (".mkv", "h264"),
    (".mkv", "h265"),
    (".mkv", "hevc"),
    (".mov", "h264"),
    (".mov", "h265"),
    (".mov", "hevc"),
    (".webm", "vp9"),
    (".ogg", "theora"),
    (".ogv", "theora"),
}
_PREVIEW_SECONDS = 20
_PREVIEW_MAX_WIDTH = 1280
_PREVIEW_CRF = "24"
_PREVIEW_PRESET = "veryfast"
_PREVIEW_CACHE_VERSION = 7
ENABLE_VIDEO_PREVIEW_PATCH = True
_CODEC_DISPLAY_NAMES = {
    "av1": "AV1",
    "dnxhd": "DNxHD",
    "ffv1": "FFV1",
    "h264": "H.264",
    "h265": "HEVC",
    "hevc": "HEVC",
    "prores": "ProRes",
    "theora": "Theora",
    "vp8": "VP8",
    "vp9": "VP9",
}

_lock = threading.Lock()
_installed = False
_original_video_format_video = None
_original_video_postprocess = None
_original_video_preprocess = None
_original_file_process_single_file = None
_original_upload_button_process_single_file = None
_original_create_app = None
_original_gallery_postprocess = None
_original_gallery_preprocess = None
_preview_to_source: dict[str, str] = {}


def _is_http_url_like(value):
    return isinstance(value, str) and value.lower().startswith(("http://", "https://"))


def _abs_path(path):
    return str(Path(os.fspath(path)).resolve())


def _cache_dir(cache_dir=None):
    base = cache_dir or getattr(gr.Video(), "GRADIO_CACHE", None) or tempfile.gettempdir()
    directory = Path(base) / "wangp_video_previews"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _source_signature(source_path):
    stat = source_path.stat()
    data = f"{_PREVIEW_CACHE_VERSION}|{source_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def _preview_path_for(source_path, cache_dir=None):
    suffix = _source_signature(source_path)
    stem = source_path.stem[:80] or "video"
    return _cache_dir(cache_dir) / f"{stem}_{suffix}.mp4"


def _sidecar_path(preview_path):
    return preview_path.with_suffix(preview_path.suffix + ".source.json")


def _remember_preview(source_path, preview_path):
    source = _abs_path(source_path)
    preview = _abs_path(preview_path)
    _preview_to_source[preview] = source
    try:
        _sidecar_path(Path(preview)).write_text(json.dumps({"source": source}), encoding="utf-8")
    except OSError:
        pass


def _restore_preview_path(path):
    if not isinstance(path, (str, os.PathLike)):
        return path
    if not os.fspath(path) or _is_http_url_like(path):
        return path
    preview = _abs_path(path)
    if preview in _preview_to_source:
        return _preview_to_source[preview]
    sidecar = _sidecar_path(Path(preview))
    if not sidecar.is_file():
        return path
    try:
        source = json.loads(sidecar.read_text(encoding="utf-8")).get("source")
    except (OSError, json.JSONDecodeError):
        return path
    if source:
        _preview_to_source[preview] = source
        return source
    return path


def _is_hdr_hevc(metadata):
    return str(metadata.get("codec_name") or "").strip().lower() in ("hevc", "h265") and bool(metadata.get("needs_tonemap"))


def _is_browser_playable(path):
    metadata = probe_video_stream_metadata(str(path)) or {}
    codec_name = str(metadata.get("codec_name") or "").strip().lower()
    if len(codec_name) > 0:
        # Many browsers cannot decode 10-bit HDR HEVC and show an empty black video: those get a tone-mapped SDR preview.
        return (path.suffix.lower(), codec_name) in _BROWSER_PLAYABLE_VIDEO_CODECS and not _is_hdr_hevc(metadata)
    return path.suffix.lower() in _BROWSER_VIDEO_EXTENSIONS


def _is_video_path(path):
    mime_type = mimetypes.guess_type(str(path))[0]
    return bool(mime_type and mime_type.startswith("video/"))


def needs_fast_video_preview(video_path):
    if not video_path or _is_http_url_like(video_path):
        return False
    path = Path(os.fspath(video_path))
    if not path.is_file():
        return False
    if not _is_video_path(path):
        return False
    return not _is_browser_playable(path)


def _format_codec_label(metadata):
    codec_name = str((metadata or {}).get("codec_name") or "").strip().lower()
    codec_profile = str((metadata or {}).get("codec_profile") or "").strip()
    if codec_name == "dnxhd" and codec_profile.upper().startswith("DNXHR"):
        return codec_profile.upper().replace("DNXHR", "DNxHR", 1)
    if codec_name in _CODEC_DISPLAY_NAMES:
        return _CODEC_DISPLAY_NAMES[codec_name]
    return codec_name.upper() if len(codec_name) > 0 else "Unknown Codec"


def _format_preview_label(source_path, metadata):
    codec_name = str((metadata or {}).get("codec_name") or "").strip().lower()
    codec_label = _format_codec_label(metadata)
    container_label = Path(source_path).suffix.lstrip(".").upper() or "Container"
    if _is_hdr_hevc(metadata):
        return "HDR Video, SDR Preview"
    if len(codec_name) > 0 and any(known_codec == codec_name for _, known_codec in _BROWSER_PLAYABLE_VIDEO_CODECS):
        return f"Container {container_label} Not Supported for {codec_label}, Low Res Preview"
    return f"Codec {codec_label} Not Supported, Low Res Preview"


def _preview_video_filter(source_path):
    metadata = probe_video_stream_metadata(str(source_path)) or {}
    width = int(metadata.get("display_width") or metadata.get("width") or 0)
    if width > _PREVIEW_MAX_WIDTH:
        scale_filter = f"scale={_PREVIEW_MAX_WIDTH}:-2:flags=fast_bilinear"
    else:
        scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=fast_bilinear"
    text = _format_preview_label(source_path, metadata)
    text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    label_filter = f"drawtext=text='{text}':fontcolor=white@0.9:fontsize=max(11\\,h/44):box=1:boxcolor=black@0.45:boxborderw=6:x=w-tw-12:y=12"
    # HDR sources are tone-mapped to BT.709 SDR, as for the frames WanGP extracts from HDR videos.
    tonemap_filter = ",".join(_build_hdr_tonemap_filter(metadata)) + ",format=yuv420p," if metadata.get("needs_tonemap") else ""
    return f"{scale_filter},{tonemap_filter}{label_filter}"


def ensure_fast_video_preview(video_path, cache_dir=None):
    if not needs_fast_video_preview(video_path):
        return None
    ffmpeg = resolve_media_binary("ffmpeg")
    if ffmpeg is None:
        return None
    source_path = Path(os.fspath(video_path)).resolve()
    preview_path = _preview_path_for(source_path, cache_dir)
    if preview_path.is_file():
        _remember_preview(source_path, preview_path)
        return str(preview_path)
    tmp_path = preview_path.with_name(preview_path.stem + ".tmp" + preview_path.suffix)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-t",
        str(_PREVIEW_SECONDS),
        "-vf",
        _preview_video_filter(source_path),
        "-c:v",
        "libx264",
        "-preset",
        _PREVIEW_PRESET,
        "-crf",
        _PREVIEW_CRF,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        str(tmp_path),
    ]
    with _lock:
        if preview_path.is_file():
            _remember_preview(source_path, preview_path)
            return str(preview_path)
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
        except OSError:
            result = None
        if result is None or result.returncode != 0 or not tmp_path.is_file():
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        os.replace(tmp_path, preview_path)
    _remember_preview(source_path, preview_path)
    return str(preview_path)


def _gradio_file_url(path):
    return "/gradio_api/file=" + quote(_abs_path(path), safe="")


def _attach_preview(file_data, cache_dir=None):
    if file_data is None or not getattr(file_data, "path", None):
        return file_data
    source_path = file_data.path
    preview_path = ensure_fast_video_preview(file_data.path, cache_dir)
    if preview_path is None:
        return file_data
    file_data.path = preview_path
    file_data.url = _gradio_file_url(preview_path)
    file_data.orig_name = file_data.orig_name or Path(source_path).name
    file_data.mime_type = "video/mp4"
    return file_data


def _replace_upload_paths_with_previews(paths):
    if not isinstance(paths, list):
        return paths
    replaced_paths = []
    for path in paths:
        preview_path = ensure_fast_video_preview(path)
        replaced_paths.append(preview_path or path)
    return replaced_paths


def _patched_video_format_video(self, video):
    if video is None or _is_http_url_like(video) or self.format is not None or self.watermark:
        return _original_video_format_video(self, video)
    video_path = Path(os.fspath(video))
    preview_path = ensure_fast_video_preview(video_path, self.GRADIO_CACHE)
    if preview_path is None:
        return _original_video_format_video(self, video)
    return FileData(path=preview_path, url=_gradio_file_url(preview_path), orig_name=video_path.name, mime_type="video/mp4")


def _patched_video_postprocess(self, value):
    data = _original_video_postprocess(self, value)
    if data is not None and getattr(data, "video", None) is not None:
        _attach_preview(data.video, self.GRADIO_CACHE)
    return data


def _patched_video_preprocess(self, payload):
    if payload is not None and getattr(payload, "video", None) is not None and getattr(payload.video, "path", None):
        payload.video.path = _restore_preview_path(payload.video.path)
    result = _original_video_preprocess(self, payload)
    return _restore_preview_path(result)


def _patched_gallery_postprocess(self, value):
    data = _original_gallery_postprocess(self, value)
    for item in getattr(data, "root", []):
        video = getattr(item, "video", None)
        if video is not None:
            _attach_preview(video, self.GRADIO_CACHE)
    return data


def _patched_gallery_preprocess(self, payload):
    if payload is not None:
        for item in getattr(payload, "root", []) or []:
            video = getattr(item, "video", None)
            if video is not None and getattr(video, "path", None):
                video.path = _restore_preview_path(video.path)
    data = _original_gallery_preprocess(self, payload)
    if data is None:
        return data
    return [(_restore_preview_path(media), caption) for media, caption in data]


def _patched_file_process_single_file(self, file_data):
    if file_data is not None and getattr(file_data, "path", None):
        file_data.path = _restore_preview_path(file_data.path)
    return _original_file_process_single_file(self, file_data)


def _patched_upload_button_process_single_file(self, file_data):
    if file_data is not None and getattr(file_data, "path", None):
        file_data.path = _restore_preview_path(file_data.path)
    return _original_upload_button_process_single_file(self, file_data)


def _install_upload_preview_middleware(app):
    if getattr(app, "_wangp_video_preview_upload_middleware", False):
        return

    @app.middleware("http")
    async def _wangp_video_preview_upload_middleware(request, call_next):
        response = await call_next(request)
        if request.method != "POST" or not request.url.path.endswith("/upload") or response.status_code != 200:
            return response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        try:
            paths = json.loads(body)
        except json.JSONDecodeError:
            from fastapi import Response

            return Response(content=body, status_code=response.status_code, headers=dict(response.headers), media_type=response.media_type, background=response.background)
        replaced_paths = _replace_upload_paths_with_previews(paths)
        if replaced_paths == paths:
            from fastapi import Response

            return Response(content=body, status_code=response.status_code, headers=dict(response.headers), media_type=response.media_type, background=response.background)
        from fastapi.responses import JSONResponse

        headers = {key: value for key, value in response.headers.items() if key.lower() not in {"content-length", "content-type"}}
        return JSONResponse(replaced_paths, status_code=response.status_code, headers=headers, background=response.background)

    app._wangp_video_preview_upload_middleware = True


def _patched_create_app(*args, **kwargs):
    app = _original_create_app(*args, **kwargs)
    _install_upload_preview_middleware(app)
    return app


def install():
    global _installed, _original_video_format_video, _original_video_postprocess, _original_video_preprocess, _original_file_process_single_file, _original_upload_button_process_single_file, _original_create_app, _original_gallery_postprocess, _original_gallery_preprocess
    if not ENABLE_VIDEO_PREVIEW_PATCH:
        return
    if _installed:
        return
    mimetypes.add_type("video/quicktime", ".mov")
    mimetypes.add_type("video/x-matroska", ".mkv")
    _original_video_format_video = gr.Video._format_video
    _original_video_postprocess = gr.Video.postprocess
    _original_video_preprocess = gr.Video.preprocess
    _original_file_process_single_file = gr.File._process_single_file
    _original_upload_button_process_single_file = gr.UploadButton._process_single_file
    _original_gallery_postprocess = gr.Gallery.postprocess
    _original_gallery_preprocess = gr.Gallery.preprocess
    from gradio.routes import App

    _original_create_app = App.create_app
    gr.Video._format_video = _patched_video_format_video
    gr.Video.postprocess = _patched_video_postprocess
    gr.Video.preprocess = _patched_video_preprocess
    gr.File._process_single_file = _patched_file_process_single_file
    gr.UploadButton._process_single_file = _patched_upload_button_process_single_file
    gr.Gallery.postprocess = _patched_gallery_postprocess
    gr.Gallery.preprocess = _patched_gallery_preprocess
    App.create_app = staticmethod(_patched_create_app)
    _installed = True


install()


__all__ = ["ensure_fast_video_preview", "install", "needs_fast_video_preview"]
