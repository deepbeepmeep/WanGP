SDR_VIDEO_CODEC_CHOICES = [
    ("x265 CRF 28 (Smaller Files)", "libx265_28"),
    ("x265 CRF 18 (Balanced)", "libx265_18"),
    ("x264 Level 8 (Balanced)", "libx264_8"),
    ("x265 CRF 8 (High Quality)", "libx265_8"),
    ("x264 Level 10 (High Quality)", "libx264_10"),
    ("x264 Lossless", "libx264_lossless"),
    ("x264 MJPEG", "libx264_mjpeg"),
    ("ProRes 422 (editing)", "prores_422"),
    ("DNxHR HQ (editing)", "dnxhr_hq"),
]

VIDEO_CONTAINER_CHOICES = [
    ("MP4", "mp4"),
    ("MOV / QuickTime", "mov"),
    ("MKV / Matroska", "mkv"),
]

RGBA_VIDEO_OUTPUT_CHOICES = [
    ("RGBA PNG Frames (ZIP)", "png_zip"),
    ("ProRes 4444 (MOV)", "prores_4444"),
]

SUPPORTED_VIDEO_CONTAINERS = {"mkv", "mov", "mp4"}
CONFIG_VIDEO_CONTAINERS = {value for _, value in VIDEO_CONTAINER_CHOICES}
PROFESSIONAL_VIDEO_CODECS = {"prores_422", "dnxhr_hq"}
QUICKTIME_AUDIO_CODEC_KEYS = {"aac_128", "aac_192", "aac_256", "aac_320", "alac"}


def normalize_video_container(container: str | None) -> str:
    return str(container or "mp4").strip().lower() or "mp4"


def normalize_video_codec(codec_key: str | None) -> str:
    return str(codec_key or "libx264_8").strip().lower() or "libx264_8"


def normalize_video_audio_codec(codec_key: str | None) -> str:
    return str(codec_key or "aac_128").strip().lower() or "aac_128"


def get_video_container_extension(container: str | None) -> str:
    container = normalize_video_container(container)
    return f".{container}" if container in SUPPORTED_VIDEO_CONTAINERS else ".mp4"


def _get_video_codec_spec(codec_key: str | None, container: str | None) -> tuple[str, str, list[str]]:
    codec_key = normalize_video_codec(codec_key)
    container = normalize_video_container(container)
    if codec_key == "libx264_8":
        return "libx264", "yuv420p", ["-crf", "10"]
    if codec_key == "libx264_10":
        return "libx264", "yuv420p", ["-crf", "0"]
    if codec_key == "libx265_28":
        return "libx265", "yuv420p", ["-crf", "28", "-x265-params", "log-level=none"]
    if codec_key == "libx265_18":
        return "libx265", "yuv420p", ["-crf", "18", "-x265-params", "log-level=none"]
    if codec_key == "libx265_8":
        return "libx265", "yuv420p", ["-crf", "8", "-x265-params", "log-level=none"]
    if codec_key == "libx264_lossless":
        if container == "mkv":
            return "ffv1", "rgb24", []
        return "libx264", "yuv444p", ["-crf", "0"]
    if codec_key == "libx264_mjpeg":
        # Intra-frame-only x264 (every frame an IDR keyframe): an MJPEG-like lossy YUV 4:2:0 encode
        return "libx264", "yuv420p", ["-crf", "10", "-x264-params", "keyint=1:minkeyint=1:scenecut=0"]
    if codec_key == "prores_422":
        return "prores_ks", "yuv422p10le", ["-profile:v", "2"]
    if codec_key == "prores_4444":
        return "prores_ks", "yuva444p10le", ["-profile:v", "4", "-alpha_bits", "16"]
    if codec_key == "dnxhr_hq":
        return "dnxhd", "yuv422p", ["-profile:v", "dnxhr_hq"]
    return "libx264", "yuv420p", ["-crf", "10"]


def get_video_encode_args(codec_key: str | None, container: str | None) -> list[str]:
    codec, pixel_format, output_params = _get_video_codec_spec(codec_key, container)
    return ["-c:v", codec, *output_params, "-pix_fmt", pixel_format]


def get_imageio_codec_params(codec_key: str | None, container: str | None) -> dict:
    codec, pixel_format, output_params = _get_video_codec_spec(codec_key, container)
    # Alpha mattes must stay pixel-aligned with their source after VAE padding
    # is cropped away. ImageIO's default macroblock resizing breaks that alignment.
    return {"codec": codec, "quality": None, "pixelformat": pixel_format, "macro_block_size": 1, "output_params": [*output_params, "-hide_banner", "-nostats"]}


def validate_video_output_settings(video_codec: str | None, video_container: str | None, audio_codec: str | None = None, width: int | None = None, height: int | None = None, *, allowed_containers: set[str] | None = None) -> str | None:
    video_codec = normalize_video_codec(video_codec)
    video_container = normalize_video_container(video_container)
    audio_codec = normalize_video_audio_codec(audio_codec)
    allowed = CONFIG_VIDEO_CONTAINERS if allowed_containers is None else allowed_containers
    if video_container not in allowed:
        return f"Unsupported video container: {video_container}."
    if video_codec in PROFESSIONAL_VIDEO_CODECS and video_container not in {"mkv", "mov"}:
        return "ProRes 422 and DNxHR HQ require the MOV / QuickTime or MKV container."
    if video_container == "mov" and audio_codec == "flac":
        return "FLAC soundtracks require MP4 or MKV; choose AAC or ALAC for MOV."
    if video_container in {"mp4", "mov"} and audio_codec not in QUICKTIME_AUDIO_CODEC_KEYS and not (video_container == "mp4" and audio_codec == "flac"):
        return f"{video_container.upper()} output does not support audio codec setting '{audio_codec}'."
    if video_codec == "dnxhr_hq" and width is not None and height is not None and (int(width) < 256 or int(height) < 120):
        return "DNxHR HQ output requires a resolution of at least 256x120."
    return None
