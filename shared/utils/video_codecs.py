SDR_VIDEO_CODEC_CHOICES = [
    ("x265 CRF 28 (Smaller Files)", "libx265_28"),
    ("x265 CRF 18 (Balanced)", "libx265_18"),
    ("x264 Level 8 (Balanced)", "libx264_8"),
    ("x265 CRF 8 (High Quality)", "libx265_8"),
    ("x264 Level 10 (High Quality)", "libx264_10"),
    ("x264 Lossless", "libx264_lossless"),
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

HARDWARE_VIDEO_ENCODING_CHOICES = [
    ("Auto (NVIDIA NVENC when available)", "auto"),
    ("Off (CPU encoders only)", "off"),
]

# NVENC settings giving the quality and size of the CPU encoder they replace (Y-PSNR on 1920x1056 footage: x264 CRF 10 52.4 dB, NVENC CQ 10
# 52.5 dB, same size, 3x faster; x264 CRF 0 and NVENC lossless both lossless; x265 CRF 8 / 18 / 28 51.2 / 45.3 / 39.6 dB, HEVC NVENC CQ 10 / 21
# / 31 51.3 / 45.6 / 40.5 dB, 5 % smaller to 5 % larger). The other choices (x264 4:4:4 or FFV1 lossless, ProRes, DNxHR) stay on the CPU.
_NVENC_QUALITY = ["-tune", "hq", "-rc", "vbr", "-b:v", "0", "-maxrate", "500M"]
NVENC_VIDEO_CODECS = {
    "libx264_8": ("h264_nvenc", [*_NVENC_QUALITY, "-cq", "10", "-profile:v", "high"]),
    "libx264_10": ("h264_nvenc", ["-tune", "lossless"]),
    "libx265_28": ("hevc_nvenc", [*_NVENC_QUALITY, "-cq", "31"]),
    "libx265_18": ("hevc_nvenc", [*_NVENC_QUALITY, "-cq", "21"]),
    "libx265_8": ("hevc_nvenc", [*_NVENC_QUALITY, "-cq", "10"]),
}
_hardware_video_encoding = "auto"
_nvenc_probes = {}

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


def set_hardware_video_encoding(mode: str | None) -> None:
    """The Hardware Video Encoding config: "auto" encodes on the GPU's NVENC the choices it can take, when the ffmpeg build and the GPU have it."""
    global _hardware_video_encoding
    _hardware_video_encoding = "off" if mode == "off" else "auto"


def hardware_video_encoding_enabled() -> bool:
    return _hardware_video_encoding != "off"


def nvenc_available(ffmpeg_exe: str, encoder: str) -> bool:
    """Whether ffmpeg_exe encodes with encoder on this machine: a two-frame encode, tried once per binary and encoder (a build without NVENC, a
    GPU or driver it does not support fail it)."""
    key = (ffmpeg_exe, encoder)
    if key not in _nvenc_probes:
        import subprocess
        try:
            result = subprocess.run([ffmpeg_exe, "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:r=24", "-frames:v", "2", "-c:v", encoder, "-f", "null", "-"],
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _nvenc_probes[key] = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _nvenc_probes[key] = False
    return _nvenc_probes[key]


def nvenc_in_use(codec_key: str | None) -> bool:
    """Whether the videos WanGP saves with codec_key are encoded on NVENC (the Hardware Video Encoding config, the codec and the machine)."""
    import imageio_ffmpeg
    return _get_video_codec_spec(codec_key, "mp4", None, imageio_ffmpeg.get_ffmpeg_exe())[0].endswith("_nvenc")


def _get_video_codec_spec(codec_key: str | None, container: str | None, hardware: bool | None = None, ffmpeg_exe: str | None = None) -> tuple[str, str, list[str]]:
    """hardware: encode on NVENC when ffmpeg_exe can (None: the Hardware Video Encoding config)."""
    codec_key = normalize_video_codec(codec_key)
    container = normalize_video_container(container)
    nvenc = NVENC_VIDEO_CODECS.get(codec_key)
    if nvenc is not None and (hardware_video_encoding_enabled() if hardware is None else hardware) and nvenc_available(ffmpeg_exe, nvenc[0]):
        return nvenc[0], "yuv420p", ["-preset", "p5", *nvenc[1]]
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
    if codec_key == "prores_422":
        return "prores_ks", "yuv422p10le", ["-profile:v", "2"]
    if codec_key == "prores_4444":
        return "prores_ks", "yuva444p10le", ["-profile:v", "4", "-alpha_bits", "16"]
    if codec_key == "dnxhr_hq":
        return "dnxhd", "yuv422p", ["-profile:v", "dnxhr_hq"]
    return "libx264", "yuv420p", ["-crf", "10"]


def get_video_encode_args(codec_key: str | None, container: str | None, hardware: bool | None = None) -> list[str]:
    """Encoder arguments for WanGP's ffmpeg binary (ffmpeg_bins)."""
    from .video_decode import resolve_media_binary
    codec, pixel_format, output_params = _get_video_codec_spec(codec_key, container, hardware, resolve_media_binary("ffmpeg") or "ffmpeg")
    return ["-c:v", codec, *output_params, "-pix_fmt", pixel_format]


def get_imageio_codec_params(codec_key: str | None, container: str | None, hardware: bool | None = None) -> dict:
    """Writer parameters for imageio, which runs its own ffmpeg binary."""
    import imageio_ffmpeg
    codec, pixel_format, output_params = _get_video_codec_spec(codec_key, container, hardware, imageio_ffmpeg.get_ffmpeg_exe())
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
