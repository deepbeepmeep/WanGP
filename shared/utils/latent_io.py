# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Thin companion-file format for WanGP generated latents.

A companion ``.pt`` file stores the final pre-decode latent(s) per generated
media file, tagged with the identity of the app/model/VAE that produced them,
so the latent can later be decoded or reused as the starting point of a new
generation. The file is a flat, tagged dictionary: the top-level
object carries one entry per format id, so each model family can later attach
its own interpretation to new keys without bumping the format version.

Entry formats:

- ``wgp_latent/1`` (the all-latent layout): the video latents of
  every window of the job stacked on the latent time dimension, with one
  metadata block per window in video order (the per-window prompt, seed,
  steps and denoising strength, the pixel-space head/tail trims and the
  decoded frame count). The entry records the VAE identity (file name,
  SHA-256, dtype, latent channels), the canvas (width, height, fps, HDR
  flag + transform) and the sidecar's total committed frame count, so the
  decoded length is known without re-deriving it from the VAE layout. The
  sidecar is the sole content source of its video: the only non-latent block
  in the whole format is, on i2v sidecars only, the start image embedded as
  lossless PNG bytes (the preprocessed canvas-size frame the production
  assembly splices into frame 0). Decoding the window latents, re-applying
  the recorded trims and splicing the embedded image (i2v) reproduces the
  original video exactly. It is the layout the app writes for every
  latent-writing job — LTX2 was the first family to write it, then Wan,
  LongCat and MiniMax H3 (the H3 VAE's piecewise 5/17 temporal layout
  rides with the window blocks instead of the linear
  ``latent_stride``/``frame_offset`` fields). It is the only video layout
  the app writes or reads: a ``wgp_latent/1`` entry without a ``windows``
  list is not a current file, and the decode and branch flows refuse such
  files with the one generic message (:func:`non_current_latent_message`).
- ``wgp_latent/audio/1``: the pre-decode audio latent(s) of models that
  generate audio, stored with the same per-window stacking as the video
  windows (one block per window, aligned with the video windows). The audio
  entry is all-latent: it carries the audio VAE identity and the vocoder
  name, never a pointer to an external audio source.

Video entries may additionally record the VAE's temporal layout
(``latent_stride``/``frame_offset``, so the pixel frame count can be derived
for any VAE, not only Wan's) and the HDR state of the decoded output
(``is_hdr``/``hdr_transform``).

MiniMax H3's video VAE has a piecewise temporal layout (the first two latent
frames cover five pixel frames, then every five latent frames cover a
17-frame clip) that the linear ``offset + stride * (t - 1)`` model cannot
express, so H3 entries carry explicit per-window metadata instead:
``frame_count`` (the window's output frame count), ``h3_history_frames``
(the input frames the window's generation was conditioned on; the decode
job re-decodes them from the previous window in front of the window's
target run and the recorded head trim drops them again, so they never
count in the window's own committed share), ``h3_target_frames`` (the
pixel length the decoded window is truncated to) and ``h3_anchor_head``
(1 when the window's first decoded frame is a re-anchored shared frame —
the last decoded frame of the window it continues — which the head trim
drops from its contribution, or commits as its first frame when the
window carries a one-frame head share). The helper
:func:`h3_video_latent_pixel_frames` derives the decoded pixel length
from the latent time dimension.

Only ``torch`` and the Python standard library are required. When NumPy is
available it is used for the raw-bytes conversion; otherwise the latent is
stored through torch's native (already zlib-compressed) tensor serialization.
"""
from __future__ import annotations

import hashlib
import io
import lzma
import os
import tempfile
import zlib

import torch

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is optional
    _np = None

LATENT_FORMAT_ID = "wgp_latent/1"
LATENT_FORMAT_ID_AUDIO = "wgp_latent/audio/1"
LATENT_SIDECAR_SUFFIX = "_latent.pt"

LATENT_COMPRESSION_NONE = "none"
LATENT_COMPRESSION_ZLIB = "zlib"
LATENT_COMPRESSION_LZMA = "lzma"
LATENT_COMPRESSION_TORCH = "torch"
DEFAULT_COMPRESSION = LATENT_COMPRESSION_ZLIB
LATENT_STORAGE_DTYPE = "float16"

_FILE_HASH_CACHE: dict = {}
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path) -> str:
    """SHA-256 of a file, cached by (path, size, mtime)."""
    stat = os.stat(path)
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    _FILE_HASH_CACHE[key] = digest.hexdigest()
    return _FILE_HASH_CACHE[key]


def _compress_raw(raw: bytes, scheme: str) -> bytes:
    if scheme == LATENT_COMPRESSION_NONE:
        return raw
    if scheme == LATENT_COMPRESSION_ZLIB:
        return zlib.compress(raw, 6)
    if scheme == LATENT_COMPRESSION_LZMA:
        return lzma.compress(raw, preset=3)
    raise ValueError(f"Unknown latent compression scheme: {scheme!r}")


def _decompress_raw(raw: bytes, scheme: str) -> bytes:
    if scheme == LATENT_COMPRESSION_NONE:
        return raw
    if scheme == LATENT_COMPRESSION_ZLIB:
        return zlib.decompress(raw)
    if scheme == LATENT_COMPRESSION_LZMA:
        return lzma.decompress(raw)
    raise ValueError(f"Unknown latent compression scheme: {scheme!r}")


def _tensor_to_bytes(latents: torch.Tensor, scheme: str = DEFAULT_COMPRESSION) -> tuple[str, bytes, list[int]]:
    """Encode a latent as raw bytes. Returns (effective_scheme, payload, shape)."""
    flat = latents.detach().to(dtype=torch.float16, device="cpu").contiguous()
    shape = list(flat.shape)
    if scheme == LATENT_COMPRESSION_TORCH or _np is None:
        buffer = io.BytesIO()
        torch.save(flat, buffer)
        return LATENT_COMPRESSION_TORCH, buffer.getvalue(), shape
    return scheme, _compress_raw(flat.numpy().tobytes(), scheme), shape


def _bytes_to_tensor(payload: bytes, scheme: str, shape: list[int]) -> torch.Tensor:
    if scheme == LATENT_COMPRESSION_TORCH:
        return torch.load(io.BytesIO(payload), weights_only=True).to(dtype=torch.float16)
    raw = _decompress_raw(payload, scheme)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float16).reshape(shape)


def _record_layout_and_hdr(entry: dict, stride: int, offset: int, is_hdr: bool, hdr_transform: str) -> None:
    """Record the optional VAE layout / HDR fields on a video entry.

    Kept out of the entry when at their defaults so entries stay minimal and
    entries written before these fields existed stay byte-identical.
    """
    if int(stride or 0) > 0:
        entry["latent_stride"] = int(stride)
        entry["frame_offset"] = int(offset or 0)
    if is_hdr:
        entry["is_hdr"] = True
        if str(hdr_transform or ""):
            entry["hdr_transform"] = str(hdr_transform)


def build_latent_payload(
    latents: torch.Tensor,
    *,
    app_version: str,
    model_type: str,
    base_model_type: str = "",
    model_filename: str = "",
    loras=(),
    vae_file: str = "",
    vae_sha256: str = "",
    vae_z_dim: int = 0,
    vae_dtype: str = "",
    frames: int = 0,
    fps: float = 0.0,
    seed: int = 0,
    prompt: str = "",
    denoising_strength: float = 1.0,
    steps: int = 0,
    latent_stride: int = 0,
    frame_offset: int = 0,
    is_hdr: bool = False,
    hdr_transform: str = "",
    compression: str = DEFAULT_COMPRESSION,
    **extra,
) -> dict:
    """Build the tagged ``wgp_latent/1`` payload for one pre-decode latent.

    The result is one window block of the all-latent sidecar: the app
    accumulates the blocks of a sliding-window job and writes them through
    :func:`build_latent_sidecar`. ``extra`` is copied onto the entry for
    family-specific fields (e.g. the MiniMax H3 per-window frame
    bookkeeping); fields that are not given are not recorded.
    """
    if latents.dim() == 5:
        latents = latents[0]
    if latents.dim() != 4:
        raise ValueError(f"Expected a [c, t, h, w] or [b, c, t, h, w] latent, got shape {tuple(latents.shape)}")
    scheme, payload, shape = _tensor_to_bytes(latents, compression)
    entry = {
        "format_id": LATENT_FORMAT_ID,
        "app_version": str(app_version),
        "model_type": str(model_type),
        "base_model_type": str(base_model_type),
        "model_filename": str(model_filename),
        "loras": [str(u) for u in loras],
        "vae_file": str(vae_file),
        "vae_sha256": str(vae_sha256),
        "vae_z_dim": int(vae_z_dim),
        "vae_dtype": str(vae_dtype),
        "frames": int(frames),
        "fps": float(fps),
        "seed": int(seed),
        "prompt": str(prompt),
        "denoising_strength": float(denoising_strength),
        "steps": int(steps),
        "dtype": LATENT_STORAGE_DTYPE,
        "shape": shape,
        "compression": scheme,
        "payload": payload,
    }
    _record_layout_and_hdr(entry, latent_stride, frame_offset, is_hdr, hdr_transform)
    if extra:
        entry.update(extra)
    return {LATENT_FORMAT_ID: entry}




def build_latent_sidecar(
    latents: list[torch.Tensor],
    window_metas: list[dict],
    *,
    app_version: str,
    model_type: str,
    base_model_type: str = "",
    model_filename: str = "",
    loras=(),
    vae_file: str = "",
    vae_sha256: str = "",
    vae_z_dim: int = 0,
    vae_dtype: str = "",
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
    latent_stride: int = 0,
    frame_offset: int = 0,
    is_hdr: bool = False,
    hdr_transform: str = "",
    i2v_start_image: dict | None = None,
    compression: str = DEFAULT_COMPRESSION,
    **extra,
) -> dict:
    """Build the tagged all-latent ``wgp_latent/1`` sidecar payload.

    The all-latent layout is the sidecar format every model family writes
    (LTX2 is the first adopter, then Wan): one block per window, in video
    order, with the window's full latent run plus its own prompt, seed, steps and
    denoising strength, the pixel-space head/tail trims and the decoded
    frame count. The entry records the VAE identity (file name, SHA-256,
    dtype, latent channels), the canvas (width, height, fps, HDR flag and
    transform) and the sidecar's total committed frame count (the sum of
    the windows' committed runs - the decoded length the Latent Decode job
    must reproduce; on i2v sidecars it includes the one spliced start-image
    frame). ``i2v_start_image`` (i2v jobs only) is the preprocessed
    canvas-size start image as ``{"kind": "image_png", "data": <PNG bytes>}``
    - the only non-latent block in the format. No pointer fields of any
    kind: the sidecar is the sole content source of its video.

    ``latents`` is one ``[c, t, h, w]`` tensor per window, in window order
    (a 5D batch of one is unbatched); ``window_metas`` carries the matching
    per-window metadata (``prompt``, ``seed``, ``steps``,
    ``denoising_strength``, ``head_trim``, ``tail_trim``, ``frame_count``).
    The tensors are stacked on the time dimension (shorter windows are
    zero-padded to the longest one; the per-window ``latent_len`` always
    records the true length). ``extra`` is copied onto the entry for
    family-specific fields.
    """
    tensors: list[torch.Tensor] = []
    for item in latents:
        item = item.detach()
        if item.dim() == 5:
            item = item[0]
        if item.dim() != 4:
            raise ValueError(f"Expected a [c, t, h, w] or [b, c, t, h, w] window latent, got shape {tuple(item.shape)}")
        tensors.append(item)
    if len(tensors) == 0 or len(tensors) != len(window_metas):
        raise ValueError(f"An all-latent payload needs as many window latents ({len(tensors)}) as window metadata blocks ({len(window_metas)})")
    max_time = max(int(t.shape[1]) for t in tensors)
    if all(t.shape == tensors[0].shape for t in tensors):
        # The latent time dimension is dim 1, so windows are concatenated
        # along it (never torch.stack, which would insert a new axis).
        padded = list(tensors)
        stacked = torch.cat(tensors, dim=1)
    else:
        padded = []
        for t in tensors:
            if int(t.shape[1]) < max_time:
                # Allocate the pad explicitly: zeros_like(t) has t's own
                # time dimension, so slicing it "up to" the gap clamps and
                # the padding would be shorter than the documented
                # n_windows * max_time layout (e.g. an LTX-2 tail window
                # of 3 latents padded towards a 31-latent max).
                pad = t.new_zeros(t.shape[0], max_time - int(t.shape[1]), *t.shape[2:])
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        stacked = torch.cat(padded, dim=1)
    scheme, payload, shape = _tensor_to_bytes(stacked, compression)
    if i2v_start_image is not None:
        # Validate the block now (the reader is stricter still): a malformed
        # block would make the whole sidecar unreadable later.
        if not (isinstance(i2v_start_image, dict)
                and str(i2v_start_image.get("kind")) == "image_png"
                and isinstance(i2v_start_image.get("data"), (bytes, bytearray))
                and len(i2v_start_image["data"]) > 0):
            raise ValueError("i2v_start_image must be {\"kind\": \"image_png\", \"data\": <PNG bytes>}")
    has_image = i2v_start_image is not None
    windows: list[dict] = []
    total_frames = 0
    offset = 0
    for meta, t, t_padded in zip(window_metas, tensors, padded):
        head_trim = int(meta.get("head_trim") or 0)
        tail_trim = int(meta.get("tail_trim") or 0)
        frame_count = int(meta.get("frame_count") or 0)
        if frame_count < 0 or head_trim < 0 or tail_trim < 0:
            raise ValueError("all-latent window trims and frame counts must not be negative")
        # The window's committed run (decoded frames minus the trims); the
        # spliced i2v frame is not part of the decoded run but is part of
        # the window's committed region, so it counts in the total. The
        # decoded length is the VAE's layout applied to the window's latents
        # (the piecewise 5/17 grid for MiniMax H3, capped by its recorded
        # target; the linear stride/offset layout for the other families),
        # so the total stays correct even when the recorded frame_count is
        # the pipeline's own output length rather than the decoded run. For
        # an H3 window the head trim counts the rebuilt history prefix too
        # (the decode job prepends it before trimming), so only its excess
        # over the history falls on the decoded run.
        decoded = _window_decoded_frames(
            {"latent_stride": latent_stride, "frame_offset": frame_offset},
            {"latents": t, "h3_history_frames": meta.get("h3_history_frames"), "h3_target_frames": meta.get("h3_target_frames")})
        # The recorded head trim stays on the block verbatim (the decode
        # job applies it to its rebuilt [history | decoded] sample); only
        # its excess over the history falls on the decoded run for the
        # committed arithmetic.
        head_share = h3_head_trim_on_decoded_run(
            {"head_trim": head_trim, "h3_history_frames": meta.get("h3_history_frames"), "h3_target_frames": meta.get("h3_target_frames")})
        committed = max(0, decoded - head_share - tail_trim)
        prefix = 1 if (len(windows) == 0 and has_image) else 0
        total_frames += committed + prefix
        entry_window = {
            "prompt": str(meta.get("prompt") or "")[:1000],
            "seed": int(meta.get("seed") or 0),
            "steps": int(meta.get("steps") or 0),
            "denoising_strength": float(meta.get("denoising_strength") or 0),
            "head_trim": head_trim,
            "tail_trim": tail_trim,
            "frame_count": frame_count,
        }
        # Family-specific per-window metadata (MiniMax H3's piecewise
        # bookkeeping) travels with the window block so the decode job
        # can re-derive the decoded length without the original job's
        # model definition.
        for key in ("h3_history_frames", "h3_target_frames", "h3_anchor_head"):
            if key in meta:
                entry_window[key] = meta[key]
        entry_window["latent_start"] = offset
        entry_window["latent_len"] = int(t.shape[1])
        # The stack pads every window to the longest one, so the offsets
        # accumulate the padded lengths (an early window shorter than a
        # later one would otherwise shift every following window out of its
        # padded slot) while latent_len keeps the true length.
        offset += int(t_padded.shape[1])
        windows.append(entry_window)
    entry = {
        "format_id": LATENT_FORMAT_ID,
        "app_version": str(app_version),
        "model_type": str(model_type),
        "base_model_type": str(base_model_type),
        "model_filename": str(model_filename),
        "loras": [str(u) for u in loras],
        "vae_file": str(vae_file),
        "vae_sha256": str(vae_sha256),
        "vae_z_dim": int(vae_z_dim),
        "vae_dtype": str(vae_dtype),
        "width": int(width or 0),
        "height": int(height or 0),
        "fps": float(fps or 0),
        # The sidecar's true committed length: the video the decode must
        # reproduce (the writer validates it against the written video).
        "frames": int(total_frames),
        "windows": windows,
        "dtype": LATENT_STORAGE_DTYPE,
        "shape": shape,
        "compression": scheme,
        "payload": payload,
    }
    _record_layout_and_hdr(entry, latent_stride, frame_offset, is_hdr, hdr_transform)
    if has_image:
        # i2v jobs only: the embedded start image belongs to window 1 (the
        # block carries no per-window pointer field - its presence on the
        # entry implies it).
        entry["i2v_start_image"] = {"kind": "image_png", "data": bytes(i2v_start_image["data"])}
    entry.update(extra)
    return {LATENT_FORMAT_ID: entry}


def build_audio_latent_payload(
    latents,
    window_metas=None,
    *,
    app_version: str,
    model_type: str,
    base_model_type: str = "",
    model_filename: str = "",
    loras=(),
    audio_vae_file: str = "",
    audio_vae_sha256: str = "",
    audio_vae_dtype: str = "",
    sample_rate: int = 0,
    channels: int = 0,
    samples: int = 0,
    seed: int = 0,
    prompt: str = "",
    denoising_strength: float = 1.0,
    steps: int = 0,
    total_windows: int = 0,
    compression: str = DEFAULT_COMPRESSION,
    **extra,
) -> dict:
    """Build the tagged ``wgp_latent/audio/1`` payload for pre-decode audio latents.

    ``latents`` is one ``[c, t]`` tensor per audio window (time on dim 1,
    mirroring the video layout), in window order, or a single tensor for a
    single-window job. ``window_metas`` carries the matching per-window
    metadata (``window_no``, ``start_sample`` — the sample offset of the
    window in the final audio). The tensors are stacked on the time
    dimension (shorter windows are zero-padded to the longest one; the
    per-window ``latent_len`` always records the true length), the same
    layout as the video all-latent entry.
    """
    if isinstance(latents, torch.Tensor):
        latents = [latents]
    tensors: list[torch.Tensor] = []
    for item in latents:
        item = item.detach()
        if item.dim() == 3:
            item = item[0]
        if item.dim() != 2:
            raise ValueError(f"Expected a [c, t] or [1, c, t] audio latent, got shape {tuple(item.shape)}")
        tensors.append(item)
    if len(tensors) == 0:
        raise ValueError("An audio latent payload needs at least one window latent")
    if window_metas is None:
        window_metas = [{} for _ in tensors]
    if len(window_metas) != len(tensors):
        raise ValueError(f"An audio latent payload needs as many window latents ({len(tensors)}) as window metadata blocks ({len(window_metas)})")
    max_time = max(int(t.shape[1]) for t in tensors)
    if all(t.shape == tensors[0].shape for t in tensors):
        padded = list(tensors)
        stacked = torch.cat(tensors, dim=1)
    else:
        padded = []
        for t in tensors:
            if int(t.shape[1]) < max_time:
                # Same explicit-allocation fix as the video builder above.
                pad = t.new_zeros(t.shape[0], max_time - int(t.shape[1]))
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        stacked = torch.cat(padded, dim=1)
    scheme, payload, shape = _tensor_to_bytes(stacked, compression)
    windows: list[dict] = []
    offset = 0
    for meta, t, t_padded in zip(window_metas, tensors, padded):
        entry_window = {
            "window_no": int(meta.get("window_no") or (len(windows) + 1)),
            "start_sample": int(meta.get("start_sample") or 0),
            "samples": int(meta.get("samples") or 0),
        }
        entry_window["latent_start"] = offset
        entry_window["latent_len"] = int(t.shape[1])
        # The stack pads every window to the longest one, so the offsets
        # accumulate the padded lengths (an early window shorter than a
        # later one would otherwise shift every following window out of
        # its padded slot) while latent_len keeps the true length.
        offset += int(t_padded.shape[1])
        windows.append(entry_window)
    entry = {
        "format_id": LATENT_FORMAT_ID_AUDIO,
        "app_version": str(app_version),
        "model_type": str(model_type),
        "base_model_type": str(base_model_type),
        "model_filename": str(model_filename),
        "loras": [str(u) for u in loras],
        "audio_vae_file": str(audio_vae_file),
        "audio_vae_sha256": str(audio_vae_sha256),
        "audio_vae_dtype": str(audio_vae_dtype),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "samples": int(samples),
        "total_windows": int(total_windows or len(windows)),
        "seed": int(seed),
        "prompt": str(prompt)[:1000],
        "denoising_strength": float(denoising_strength),
        "steps": int(steps),
        "windows": windows,
        "dtype": LATENT_STORAGE_DTYPE,
        "shape": shape,
        "compression": scheme,
        "payload": payload,
    }
    entry.update(extra)
    return {LATENT_FORMAT_ID_AUDIO: entry}


def latent_payload(obj) -> dict | None:
    """Return the ``wgp_latent/1`` entry of a loaded side-file object, if present."""
    if not isinstance(obj, dict):
        return None
    tagged = obj.get(LATENT_FORMAT_ID)
    if isinstance(tagged, dict) and tagged.get("format_id") == LATENT_FORMAT_ID:
        return tagged
    if obj.get("format_id") == LATENT_FORMAT_ID:
        return obj
    return None


def audio_latent_payload(obj) -> dict | None:
    """Return the ``wgp_latent/audio/1`` entry of a loaded side-file object, if present."""
    if not isinstance(obj, dict):
        return None
    tagged = obj.get(LATENT_FORMAT_ID_AUDIO)
    if isinstance(tagged, dict) and tagged.get("format_id") == LATENT_FORMAT_ID_AUDIO:
        return tagged
    if obj.get("format_id") == LATENT_FORMAT_ID_AUDIO:
        return obj
    return None


def payload_latents(entry: dict) -> torch.Tensor:
    """Decode the stored latent (float16; ``[c, t, h, w]``)."""
    scheme = str(entry.get("compression") or LATENT_COMPRESSION_TORCH)
    shape = [int(u) for u in entry.get("shape") or []]
    tensor = _bytes_to_tensor(bytes(entry.get("payload") or b""), scheme, shape)
    if tensor.numel() > 0 and shape and list(tensor.shape) != shape:
        raise ValueError(f"Latent shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
    return tensor


def is_all_latent_entry(entry) -> bool:
    """Whether ``entry`` is an all-latent ``wgp_latent/1`` entry.

    The all-latent layout is the only video layout the app writes or
    reads (LTX2 was the first family to write it, then Wan, LongCat and
    MiniMax H3): it stores every window's full latent run plus the
    per-window prompt/seed/steps/denoising strength, head/tail trims and
    decoded frame count; the entry carries the VAE identity, the canvas,
    the total committed frame count and (i2v jobs only) the preprocessed
    start image embedded as lossless PNG bytes. A ``wgp_latent/1`` entry
    without a ``windows`` list is not a current file; the decode and
    branch flows refuse it with the one generic message
    (:func:`non_current_latent_message`).
    """
    if not isinstance(entry, dict) or entry.get("format_id") != LATENT_FORMAT_ID:
        return False
    return isinstance(entry.get("windows"), list) and len(entry["windows"]) > 0


def non_current_latent_message(subject: str = "the latent file") -> str:
    """The one user-facing refusal for a latent file the app no longer
    reads: an older version of the app wrote the sidecar in a layout this
    build does not decode (its kept region was named in the source video
    instead of stored as latents), and a file that is not a WanGP
    sidecar at all is refused the same way. ``subject`` names the
    offending file.
    """
    return (f"{subject} is not a current WanGP latent file (an older version of the app, or a file from another tool). "
            f"Re-run the original job with the current version of the app to produce a compatible latent file.")


def entry_i2v_image(entry) -> dict | None:
    """The i2v start-image block of an all-latent entry, if present and
    well-formed: ``{"kind": "image_png", "data": <PNG bytes>}`` — the
    preprocessed canvas-size frame the production assembly splices into
    frame 0, embedded so the sidecar stays self-contained (it is content,
    not a pointer: no path, no fit-canvas replay bookkeeping)."""
    if not isinstance(entry, dict):
        return None
    block = entry.get("i2v_start_image")
    if isinstance(block, dict) and str(block.get("kind")) == "image_png" and isinstance(block.get("data"), (bytes, bytearray)) and len(block["data"]) > 0:
        return block
    return None


# Family-specific per-window fields carried on entries and window blocks
# (see the module docstring for the MiniMax H3 rationale).
_WINDOW_EXTRA_FIELDS = ("frame_count", "h3_history_frames", "h3_target_frames", "h3_anchor_head")


def _copy_window_extra_fields(spec: dict, source: dict) -> None:
    for key in _WINDOW_EXTRA_FIELDS:
        value = source.get(key)
        if value is not None:
            spec[key] = int(value)


def entry_window_specs(entry: dict) -> list[dict]:
    """Normalize an entry into one spec per window, in window order.

    Each spec has ``window_no``, ``start_frame`` (position in the final
    video timeline), ``head_trim``/``tail_trim`` (pixel frames dropped
    from the decoded window in the original assembly) and ``latents``
    (``[c, t, h, w]``, CPU float16), plus ``frame_count`` (the window's
    output frame count) and any family-specific fields recorded on the
    window block (``h3_history_frames`` / ``h3_target_frames`` for
    MiniMax H3).

    Every file the app reads is all-latent (one block per window), so
    this is a straight walk over the ``windows`` list; a file in any
    other layout is refused up front by the decode and branch flows and
    raises ``ValueError`` here instead.
    """
    if not is_all_latent_entry(entry):
        raise ValueError(non_current_latent_message())
    # One window per block in the ``windows`` list, each with its own
    # prompt/seed/steps/denoising strength, the pixel-space trims and the
    # decoded frame count; the latents are stacked on the time dimension
    # and split back with the recorded per-window offsets/lengths. No
    # pointer fields exist in this layout (the i2v start image is
    # entry-level content, not a per-window prefix), so prefix_frames is
    # always 0 and the window numbers are positional (a branch job
    # renumbers its carried blocks at build time, so the position in the
    # list is the key).
    stacked = payload_latents(entry)
    specs: list[dict] = []
    for index, window in enumerate(entry.get("windows") or []):
        start = int(window.get("latent_start") or 0)
        length = int(window.get("latent_len") or 0)
        spec: dict = {
            "window_no": index + 1,
            "head_trim": int(window.get("head_trim") or 0),
            "tail_trim": int(window.get("tail_trim") or 0),
            "prefix_frames": 0,
            "prompt": str(window.get("prompt") or ""),
        }
        _copy_window_extra_fields(spec, window)
        spec["latents"] = stacked[:, start:start + length].clone()
        specs.append(spec)
    # The committed offsets (the i2v image frame included) are the
    # windows' positions on the final video's timeline.
    offset = 0
    for spec in specs:
        spec["start_frame"] = offset
        offset += _window_committed_frames(entry, spec)
    return specs


def is_audio_multi_window(entry: dict) -> bool:
    """True for an audio entry that carries more than one window."""
    return entry.get("format_id") == LATENT_FORMAT_ID_AUDIO and len(entry.get("windows") or []) > 1


def audio_window_specs(entry: dict) -> list[dict]:
    """Normalize an audio entry into one spec per window, in window order.

    Each spec has ``window_no``, ``start_sample`` (position of the window in
    the final audio, in samples) and ``latents`` (``[c, t]``, CPU float16,
    true length). Audio windows are expected to line up one-to-one with the
    video windows of the companion entry.
    """
    if entry.get("format_id") != LATENT_FORMAT_ID_AUDIO:
        return []
    stacked = payload_latents(entry)
    specs: list[dict] = []
    for window in entry.get("windows") or []:
        start = int(window.get("latent_start") or 0)
        length = int(window.get("latent_len") or 0)
        specs.append({
            "window_no": int(window.get("window_no") or (len(specs) + 1)),
            "start_sample": int(window.get("start_sample") or 0),
            "latents": stacked[:, start:start + length].clone(),
        })
    return specs


def merge_audio_window_chunks(audio_chunks, window_metas=None, total_samples: int = 0, head_trims=None) -> torch.Tensor:
    """Re-assemble per-window decoded audio waveforms into the final track.

    ``audio_chunks`` is one waveform per window, in window order
    (``[t]`` or ``[c, t]``).

    Default (``head_trims is None``): each chunk is written at its recorded
    ``start_sample`` offset from ``window_metas`` into a zero buffer of
    ``total_samples`` length (later windows win the overlap regions); without
    per-window offsets the chunks are concatenated head-to-tail.

    ``head_trims`` (one int sample count per chunk) switches to the model the
    generation assembly actually uses: the writer commits window 1 in full and
    then, for every later window, drops its first ``head_trims[k]`` samples
    (the sliding-window overlap, ``int(reuse_frames * sample_rate / fps)`` in
    the job) and concatenates the remainder at the accumulated offset — the
    written track therefore contains no overlap at all. With this model chunk
    0 keeps its recorded ``start_sample`` (0 for a fresh job) and each later
    chunk is placed at the end of the previous chunk's committed run, so a
    multi-window decode reproduces the generated track instead of shifting
    every window after the first by the overlap amount.
    """
    tensors: list[torch.Tensor] = []
    for chunk in audio_chunks:
        t = chunk.detach() if torch.is_tensor(chunk) else torch.as_tensor(chunk)
        t = t.to(torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() == 3:
            t = t[0]
        if t.dim() != 2:
            raise ValueError(f"Unsupported audio waveform shape {tuple(t.shape)} (expected [t] or [c, t])")
        tensors.append(t)
    if not tensors:
        return torch.zeros(1, 0)
    if head_trims is not None:
        # Writer-mirror model (see the docstring): head-trim each chunk, then
        # place chunk 0 at its recorded offset and the rest head-to-tail.
        trims = [max(0, int(x or 0)) for x in head_trims]
        if len(trims) != len(tensors):
            raise ValueError(f"head_trims length {len(trims)} does not match the {len(tensors)} audio chunks")
        placed = [t[:, trim:] for t, trim in zip(tensors, trims)]
        base = 0
        if window_metas is not None and len(window_metas) == len(tensors) and window_metas[0] is not None:
            base = max(0, int((window_metas[0] or {}).get("start_sample") or 0))
        channels = max(t.shape[0] for t in placed)
        total = base + sum(t.shape[1] for t in placed)
        if total <= 0:
            return torch.zeros(channels, 0)
        out = torch.zeros(channels, total, dtype=torch.float32)
        cursor = base
        for t in placed:
            n = min(t.shape[1], total - cursor)
            if n <= 0:
                continue
            src = t if t.shape[0] == channels else t[:1].repeat(channels, 1)
            out[:, cursor:cursor + n] = src[:, :n]
            cursor += n
        return out
    offsets = None
    if window_metas is not None and len(window_metas) == len(tensors) and all(w is not None for w in window_metas):
        cand = [int(w.get("start_sample") or 0) for w in window_metas]
        if any(cand):
            offsets = cand
    if offsets is None:
        return torch.cat(tensors, dim=1)
    total = int(total_samples or 0)
    if total <= 0:
        samples_list = [int((w or {}).get("samples") or 0) for w in window_metas]
        total = sum(samples_list) if any(samples_list) else sum(t.shape[1] for t in tensors)
    channels = max(t.shape[0] for t in tensors)
    out = torch.zeros(channels, total, dtype=torch.float32)
    for t, off in zip(tensors, offsets):
        off = max(0, min(int(off), total))
        n = min(t.shape[1], total - off)
        if n <= 0:
            continue
        src = t if t.shape[0] == channels else t[:1].repeat(channels, 1)
        out[:, off:off + n] = src[:, :n]
    return out


def latent_layout_for(model_def, base_model_type=None):
    """The VAE's temporal layout as ``{"stride": S, "offset": O}`` for the model
    family - the linear grid :func:`window_pixel_frames` applies to a latent's
    time length. LTX-2 is 8/1 (the first latent covers one pixel frame and every
    later latent eight); every other family, including an unset model def, uses
    the Wan layout 4/1 - the same fallback :func:`window_pixel_frames` applies
    to an unset layout. MiniMax H3 is not a linear grid (it is the piecewise
    5/17 mapping, carried as explicit per-window frame metadata) and never
    reaches this helper. The LTX-2 test mirrors ``wgp._is_ltx2_family`` (the
    base type, then the model def's architecture)."""
    base = str(base_model_type or "")
    arch = str((model_def or {}).get("architecture") or "")
    if base.startswith("ltx2") or base == "joyai_echo" or arch.startswith("ltx2"):
        return {"stride": 8, "offset": 1}
    return {"stride": 4, "offset": 1}


def window_pixel_frames(latent: torch.Tensor, stride: int = 0, offset: int = 0) -> int:
    """Pixel frames a VAE decodes from a latent with the given time length.

    ``stride``/``offset`` describe the VAE's temporal layout: the first
    latent covers ``offset`` pixel frames and every later latent covers
    ``stride`` frames (Wan: 4/1, LTX-2: 8/1, Hailuo-3: 17/1). With
    ``offset == 0`` every latent covers ``stride`` frames (non-causal
    layouts). Unset values assume the Wan layout.
    """
    stride = int(stride or 0)
    offset = int(offset or 0)
    if stride <= 0:
        stride, offset = 4, 1
    length = int(latent.shape[1])
    if length <= 0:
        return 0
    if offset <= 0:
        return stride * length
    return offset + stride * (length - 1)


def h3_video_latent_pixel_frames(latent_t: int) -> int:
    """Pixel frames the MiniMax H3 video VAE decodes from ``latent_t`` latent frames.

    The H3 VAE is causal and piecewise: the first two latent frames cover the
    first five pixel frames, and every additional five latent frames cover a
    17-frame clip, so a latent of length ``2 + 5k`` decodes to ``17k + 5``
    pixel frames (5, 22, 39, 56, ...). The pipeline's
    ``video_latent_frames`` is the exact inverse. This mapping cannot be
    expressed by :func:`window_pixel_frames`' linear layout, which is why H3
    sidecars carry explicit per-window frame metadata instead of
    ``latent_stride``/``frame_offset``.
    """
    latent_t = int(latent_t or 0)
    if latent_t < 2:
        return 0
    return 17 * ((latent_t - 2) // 5) + 5


def _is_h3_spec(spec: dict) -> bool:
    """Whether a window spec is MiniMax H3 (it carries the piecewise 5/17
    frame metadata instead of a linear ``latent_stride``/``frame_offset``
    layout)."""
    return spec.get("h3_history_frames") is not None or spec.get("h3_target_frames") is not None


def h3_head_trim_on_decoded_run(spec: dict) -> int:
    """The part of a window's recorded head trim that falls on its own
    decoded run.

    The decode job applies the recorded head trim to a window's
    reconstructed sample. For the linear families the decoded run includes
    the overlap region (the latents carry the frozen head), so the full
    head trim falls on it. For MiniMax H3 the window's latents cover the
    target region only: the decode job first prepends the recorded history
    frames (the continuation the generation window was conditioned on,
    re-decoded from the previous window), and only the excess of the head
    trim over that history falls on the decoded run itself (typically the
    one shared frame a mid-stream segment re-anchors on). The committed
    arithmetic and the branch-point mapping use this excess so they cannot
    drift from the decode job's own trim.
    """
    head_trim = int(spec.get("head_trim") or 0)
    if _is_h3_spec(spec):
        return max(0, head_trim - int(spec.get("h3_history_frames") or 0))
    return head_trim


# =============================================================================
# Latent branching (Phase 0): branch-point clamping + sidecar carry-over
# =============================================================================

def _window_decoded_frames(entry: dict, spec: dict) -> int:
    """How many pixel frames the window's latents decode to in a fresh run
    (from the first latent of the window): the layout grid (4/1, 8/1) or the
    piecewise H3 mapping, truncated to ``h3_target_frames`` when recorded.
    """
    if _is_h3_spec(spec):
        decoded = h3_video_latent_pixel_frames(int(spec["latents"].shape[1]))
        target = int(spec.get("h3_target_frames") or 0)
        if target > 0:
            decoded = min(decoded, target)
        return decoded
    # The all-latent layout records the decoded frame count per window (the
    # builder validates it against the latents' layout); trust it instead of
    # re-deriving it so the committed arithmetic cannot desync from the
    # layout the job actually used.
    if is_all_latent_entry(entry) and int(spec.get("frame_count") or 0) > 0:
        return int(spec["frame_count"])
    stride = int(entry.get("latent_stride") or 0)
    offset = int(entry.get("frame_offset") or 0)
    if stride <= 0:
        stride, offset = 4, 1
    latent_len = int(spec["latents"].shape[1])
    if latent_len <= 0:
        return 0
    if offset <= 0:
        return stride * latent_len
    return offset + stride * (latent_len - 1)


def _window_committed_frames(entry: dict, spec: dict) -> int:
    """How many final-video frames the window commits (its decoded run
    after the head/tail trims that fall on it; the i2v image frame, if
    any, counts in the first window's committed region). Raises
    ``ValueError`` for a non-all-latent entry.

    For a MiniMax H3 window the recorded head trim also counts the
    rebuilt history prefix (the decode job prepends it from the previous
    window before trimming), so only its excess over the history counts
    here — the same split the decode job's trim applies.
    """
    if not is_all_latent_entry(entry):
        raise ValueError(non_current_latent_message())
    committed = max(0, _window_decoded_frames(entry, spec) - h3_head_trim_on_decoded_run(spec) - int(spec.get("tail_trim") or 0))
    # The all-latent i2v sidecar splices the embedded start image into frame 0
    # of the first window: the spliced frame sits in front of the trimmed
    # decoded run, so it counts in the window's committed region (the
    # entry's total includes it exactly once).
    if int(spec.get("window_no") or 0) == 1 and entry_i2v_image(entry) is not None:
        committed += 1
    return committed


def _window_committed_offsets(entry: dict) -> list[int]:
    """The number of final-video frames committed before each window
    (one count per window, in window order): window ``k``'s committed run
    is final-video frames ``offset[k] + 1 .. offset[k] + committed_k``.

    The Latent Decode job reassembles a sidecar by concatenating each
    window's head-/tail-trimmed decoded run in window order, so the
    committed runs tile the final video back to front and the total is
    their sum. The all-latent sidecar has no source prefix to re-extract,
    so each window's committed run (the i2v image frame included) simply
    follows the previous windows' committed runs from frame 0. Raises
    ``ValueError`` for a non-all-latent entry.
    """
    if not is_all_latent_entry(entry):
        raise ValueError(non_current_latent_message())
    offsets = []
    committed = 0
    for spec in entry_window_specs(entry):
        offsets.append(committed)
        committed += _window_committed_frames(entry, spec)
    return offsets


def sidecar_total_frames(entry: dict) -> int:
    """The total number of final-video frames the sidecar's windows commit
    (a branch point frame is measured against this).

    The committed runs are concatenated in window order (the way the
    Latent Decode job assembles them), so the total is their sum — for an
    all-latent entry that is the plain sum of the windows' committed
    counts (the i2v image frame, if any, inside the first). Raises
    ``ValueError`` for a non-all-latent entry.
    """
    if not is_all_latent_entry(entry):
        raise ValueError(non_current_latent_message())
    # The i2v image frame (if any) is already inside the first window's
    # committed count, so the plain sum is the full video length.
    return sum(_window_committed_frames(entry, spec) for spec in entry_window_specs(entry))


def sidecar_length_mismatch(entry: dict) -> tuple[int, int] | None:
    """The ``(recorded, written)`` frame counts when an all-latent sidecar is
    flagged (its recorded committed length differs from the length of the
    video the writer produced it from), else ``None``.

    The writer's length guard verifies the sidecar's recorded total against
    the written video. When they differ it is a miscount by the app (the
    recorded total is what a *Latent Decode* and a branch cut are measured
    against), not user error, and the file holds the only unrecoverable copy
    of the model's latent view of the video. So the guard writes the file
    anyway, flagged: the entry carries ``length_verified: false`` and the
    actual written frame count (``written_frame_count``) beside the recorded
    total. *Latent Decode* on such a file proceeds but says the decoded video
    has the recorded length, not the neighboring video's; *From latent*
    refuses a branch cut on it (the cut depends on that length).

    ``entry`` is the sidecar's ``wgp_latent/1`` entry. Returns
    ``(recorded, written)`` for a flagged file, ``None`` for a verified file
    (the flag absent, or set to true) or a pre-flag file that recorded no
    check.
    """
    if not isinstance(entry, dict) or entry.get("length_verified") is not False:
        return None
    return (sidecar_total_frames(entry), int(entry.get("written_frame_count") or 0))


def _window_prefix_boundaries(entry: dict, spec: dict) -> list:
    """The ``(latent_index, decoded_pixel_position)`` pairs a self-contained
    latent prefix of this window can terminate at, in increasing order:

    - linear layouts (4/1, 8/1): every latent index 1..t is on the grid;
    - H3 5/17 piecewise: only the grid points 2, 7, 12, ... (a fresh run from
      the first latent decodes 5, 22, 39, 56, ... pixel frames).
    """
    latent_len = int(spec["latents"].shape[1])
    if _is_h3_spec(spec):
        return [(t, h3_video_latent_pixel_frames(t)) for t in range(2, latent_len + 1, 5)]
    stride = int(entry.get("latent_stride") or 0)
    offset = int(entry.get("frame_offset") or 0)
    if stride <= 0:
        stride, offset = 4, 1
    if offset <= 0:
        return [(t, stride * t) for t in range(1, latent_len + 1)]
    return [(t, offset + stride * (t - 1)) for t in range(1, latent_len + 1)]


def branch_point_for_frame(entry: dict, frame, fit=False) -> dict | None:
    """Resolve a branch point picked as a frame of the final video.

    ``frame`` is a 1-based count of final-video frames to keep ("keep this
    many frames"; the last frame when 0/empty, counted from the end when
    negative — the same semantics as the *Truncate* field). A cut can never
    fall inside a latent frame, so the resolved cut lands on a latent
    boundary; which side of the picked frame it lands on depends on the mode:

    - the default (``fit=False``) takes the pick as the branch point: the
      cut lands on the boundary *before* the latent that contains the frame,
      so that whole latent — the branch's first new latent — is regenerated
      in the new video, and every frame of it changes up to the previous
      latent;
    - ``fit=True`` (the *Round branch frame up to a latent boundary*
      checkbox) keeps the latent containing the frame unchanged and advances
      the cut to the boundary *after* it, clamping forward (a window preset,
      the frame exactly at a window's committed end, clamps to that window's
      last boundary).

    A pick that falls exactly on a latent boundary (including a window's
    committed end) is its own cut in both modes. The final video the Latent
    Decode job assembles concatenates each window's head-/tail-trimmed
    decoded run in window order (see :func:`_window_committed_offsets`), so
    the picked frame's position inside its window's decoded run is
    ``head_trim + (frame - offset)``; a pick whose last kept frame still
    sits before the window's committed start (a head-trimmed window's
    committed range can start inside its first decodable prefix) cannot be
    its own cut, and the default falls back to the ``fit`` resolution
    whenever the previous boundary does not exist or falls before the first
    frame of the video.

    Returns ``None`` when the frame falls outside the sidecar's recorded
    range (before the first committed frame, past the last one, or no
    readable windows) or the entry is not a current all-latent file.
    Otherwise a dict with:

    - ``window_no``: the (1-based) window containing the picked frame;
    - ``latent_index``: the last kept latent (1-based, within that window)
      of the resolved cut;
    - ``kept_frames``: the resolved cut (the branch point itself, 1-based in
      the final video);
    - ``decoded_prefix``: the pixel length the cut decodes to inside its
      window (the new head window's head trim derives from it);
    - ``total_frames``: the sidecar's total committed frame count;
    - ``overrun``: ``kept_frames - total_frames`` when the cut passes the
      last committed frame (possible only in the last window, whose tail
      trim can leave a gap between the last latent boundary and the video
      end); 0 otherwise;
    - ``fit``: whether the resolved cut is the rounded (fit) one — true when
      it is requested or when the default fell back to it (false for an
      on-boundary pick and for the mid-latent default);
    - ``mid_latent``: whether the mid-latent default resolution applied,
      i.e. the cut is the boundary before the latent that contains the
      picked frame and that latent (``latent_index + 1``) is the branch's
      first new latent.
    """
    frame = int(frame or 0)
    if frame < 1:
        return None
    if not is_all_latent_entry(entry):
        return None
    specs = entry_window_specs(entry)
    total = sidecar_total_frames(entry)
    if frame > total:
        return None
    offsets = _window_committed_offsets(entry)
    for index, spec in enumerate(specs):
        committed = _window_committed_frames(entry, spec)
        offset = int(offsets[index])
        if committed <= 0 or frame <= offset or frame > offset + committed:
            continue
        # The branch point falls inside this window's committed run.
        # The first committed frame sits at the window's head share inside
        # its decoded run (the overlap region for the linear families, the
        # one re-anchored shared frame for a MiniMax H3 window with a
        # rebuilt history prefix).
        head_share = h3_head_trim_on_decoded_run(spec)
        # The all-latent i2v window splices the embedded start image into
        # frame 0 of the window: the spliced frame sits in the window's
        # decoded-head slot, so the video-to-decoded mapping is shifted by
        # one and a kept count includes the image itself.
        spliced = 1 if (is_all_latent_entry(entry) and int(spec.get("window_no") or 0) == 1
                        and entry_i2v_image(entry) is not None) else 0
        decoded_q = head_share + (frame - offset) - spliced  # decoded pixel position of the picked frame inside the window
        boundaries = _window_prefix_boundaries(entry, spec)
        if not boundaries:
            continue
        # The latent that contains the picked frame: the first boundary at or
        # after its decoded position (the last one when it reaches the end).
        containing_pos = len(boundaries) - 1
        for position, (candidate_index, candidate_position) in enumerate(boundaries):
            if candidate_position >= decoded_q:
                containing_pos = position
                break
        latent_index, decoded_prefix = boundaries[containing_pos]
        mid_latent = False
        if not fit and decoded_q < decoded_prefix and containing_pos > 0:
            previous_index, previous_position = boundaries[containing_pos - 1]
            candidate = int(offset + (previous_position - head_share) + spliced)
            if 1 <= candidate <= total:
                # The cut lands on the boundary before the latent that
                # contains the frame; that latent is the branch's first new
                # latent (the whole latent is regenerated).
                latent_index, decoded_prefix = previous_index, previous_position
                mid_latent = True
        fit_applied = (decoded_q < decoded_prefix) and not mid_latent
        kept = int(offset + (decoded_prefix - head_share) + spliced)
        return {
            "window_no": int(spec.get("window_no") or index + 1),
            # The window's position in the sidecar's window list (0-based).
            # The window number is positional (a branch job renumbers its
            # carried blocks at build time), so the branch consumers select
            # the window by this position.
            "window_index": int(index),
            "latent_index": int(latent_index),
            "kept_frames": kept,
            "decoded_prefix": int(decoded_prefix),
            "total_frames": int(total),
            "overrun": int(max(0, kept - total)),
            "fit": bool(fit_applied),
            "mid_latent": bool(mid_latent),
        }
    return None


def snap_branch_frame(entry: dict, frame) -> int | None:
    """The frame a branch point entered as ``frame`` cuts at with the rounded
    (fit) resolution — the one the *Round branch frame up to a latent
    boundary* checkbox selects.

    ``frame`` uses the branch-point field's semantics: a 1-based count of
    the frames to keep, empty for the last committed frame, negative
    counting from the end. A cut can never fall inside a latent frame, so
    the result is the clamped kept count, i.e. the first latent boundary at
    or after the requested frame (see :func:`branch_point_for_frame`),
    which may overrun the last committed frame of a tail-trimmed last
    window. Returns ``None`` when the frame is not a valid count or falls
    outside the sidecar's recorded range, where no cut can be expressed,
    or the entry is not a current all-latent file.
    """
    if not is_all_latent_entry(entry):
        return None
    if frame is None or (isinstance(frame, str) and len(str(frame).strip()) == 0):
        frame = sidecar_total_frames(entry)
    try:
        frame = int(frame)
    except (TypeError, ValueError):
        return None
    total = sidecar_total_frames(entry)
    if frame < 0:
        frame = total + frame
    point = branch_point_for_frame(entry, frame, fit=True)
    return int(point["kept_frames"]) if point is not None else None


def branch_frame_grid(entry: dict) -> list:
    """The branch-point values that need no clamping under the rounded (fit)
    resolution, in increasing order: the kept frame counts that fall on a
    latent boundary of the window they belong to (a boundary inside the
    window's committed range, including window presets). For a 4/1 layout
    from the first frame that is 1, 5, 9, ...; the values between two of
    them snap up to the next one (see :func:`snap_branch_frame`), and every
    value outside the sidecar's recorded range is invalid. Returns ``[]``
    for a non-all-latent entry.
    """
    if not is_all_latent_entry(entry):
        return []
    total = sidecar_total_frames(entry)
    if total < 1:
        return []
    grid = set()
    specs = entry_window_specs(entry)
    offsets = _window_committed_offsets(entry)
    for index, spec in enumerate(specs):
        committed = _window_committed_frames(entry, spec)
        if committed <= 0:
            continue
        offset = int(offsets[index])
        # Same i2v splice shift as :func:`branch_point_for_frame`.
        spliced = 1 if (int(spec.get("window_no") or 0) == 1 and entry_i2v_image(entry) is not None) else 0
        for _index, position in _window_prefix_boundaries(entry, spec):
            kept = offset + (position - h3_head_trim_on_decoded_run(spec)) + spliced
            if 1 <= kept <= total and offset <= kept <= offset + committed:
                grid.add(kept)
    return sorted(grid)


def entry_window_prompt(entry: dict, window_no, window_index=None) -> str:
    """The prompt recorded for one window of an all-latent sidecar.

    ``window_index`` (optional) selects the window positionally (0-based in
    the sidecar's window list) and wins over ``window_no``: the branch flow
    selects by position because the window blocks carry no recorded
    window numbers (a branch job renumbers its carried blocks at build
    time). Raises ``ValueError`` for a non-all-latent entry (a file the
    decode and branch flows refuse up front).
    """
    if not is_all_latent_entry(entry):
        raise ValueError(non_current_latent_message())
    # The all-latent blocks carry no window numbers, so both the positional
    # index and the window number select positionally.
    windows = entry.get("windows") or []
    if window_index is not None and 0 <= int(window_index) < len(windows):
        return str(windows[int(window_index)].get("prompt") or "")
    window_no = int(window_no or 1)
    if 1 <= window_no <= len(windows):
        return str(windows[window_no - 1].get("prompt") or "")
    return ""


def branch_window_position(entry: dict, branch_point: dict, specs: list) -> int:
    """The 1-based position of the branch window inside the sidecar's window
    list, resolved from a :func:`branch_point_for_frame` result.

    The window blocks carry no recorded window numbers (the destination
    builder renumbers carried blocks positionally), so the positional
    ``window_index`` the resolver recorded wins; a branch point without one
    falls back to the number itself.
    """
    if not specs:
        raise ValueError("the latent file has no readable latent windows")
    index = branch_point.get("window_index")
    if index is not None:
        position = int(index) + 1
        if 1 <= position <= len(specs):
            return position
    position = int(branch_point.get("window_no") or 0)
    if 1 <= position <= len(specs):
        return position
    raise ValueError(f"the branch point does not reference one of the {len(specs)} windows recorded in the latent file (window_no={branch_point.get('window_no')!r})")


def carry_over_latent_windows(sidecar_obj: dict, branch_point: dict) -> dict:
    """Seed a branch job's cumulative sidecar from a source sidecar.

    Given a loaded sidecar object (a current all-latent ``wgp_latent/1``
    entry plus an optional ``wgp_latent/audio/1``) and a branch point as
    returned by :func:`branch_point_for_frame`, return the kept windows'
    content:

    - ``video_windows``: one block per fully kept window (1..K-1), each the
      source window's metadata plus ``latents`` (``[c, t, h, w]``, true
      length) — appended to the branch job's sidecar with a fresh
      sequential ``window_no`` (one number per timeline position, because
      the source's recorded numbers can repeat) and ``prefix_frames``
      forced to 0, so the destination stays all-latent;
    - ``video_prefix``: the containing window K's block (its full metadata)
      plus the latents kept up to the resolved cut (``[c, t_kept, h, w]``)
      and ``kept_decoded`` (the cut's decoded prefix length the head trim
      derives from). With the rounded (fit) resolution the cut is the end
      of the containing latent, which carries over unchanged; with the
      mid-latent default the kept latents stop before the latent that
      contains the picked frame — that latent is the branch's first new
      latent (re-denoised per model family in later phases, a fresh-encode
      stand-in until then) and is not part of the carried prefix;
    - ``audio_windows`` / ``audio_prefix``: the corresponding audio blocks
      (``None`` when the sidecar carries no audio entry). The audio prefix
      spans the containing window's full audio; :func:`branch_audio_prefix_latents`
      trims it to the resolved cut for the model's frozen prefix.
    """
    if not isinstance(branch_point, dict):
        raise ValueError("branch_point must be the dict returned by branch_point_for_frame")
    video_entry = latent_payload(sidecar_obj)
    if video_entry is None:
        raise ValueError("the latent file carries no video latent entry")
    if not is_all_latent_entry(video_entry):
        raise ValueError(non_current_latent_message("the source latent file"))
    specs = entry_window_specs(video_entry)
    k = branch_window_position(video_entry, branch_point, specs)

    def _block(spec: dict, index: int) -> dict:
        # The all-latent layout records no window numbers (the position in
        # the list is the number; the destination builder renumbers
        # positionally), so the raw window block is carried as-is (the
        # destination's latents are attached below).
        windows = video_entry.get("windows") or []
        block = dict(windows[index]) if index < len(windows) else {}
        for key in _WINDOW_EXTRA_FIELDS:
            if key not in block and key in spec:
                block[key] = spec[key]
        block["latents"] = spec["latents"]
        return block

    video_windows = [_block(specs[index], index) for index in range(k - 1)]
    prefix_spec = specs[k - 1]
    prefix_latents = prefix_spec["latents"]
    kept_latents = int(branch_point.get("latent_index") or 0)
    full_len = int(prefix_latents.shape[1])
    if 0 < kept_latents < full_len:
        # The resolved cut falls inside the containing window: the latents
        # after it (starting with the branch's first new latent) are not
        # carried over.
        prefix_latents = prefix_latents[:, :kept_latents]
    video_prefix = {
        "window_no": k,
        "window": _block(prefix_spec, k - 1),
        "latents": prefix_latents,
        "kept_decoded": int(branch_point.get("decoded_prefix") or 0),
    }
    if 0 < kept_latents < full_len:
        # The sliced window is now a partial window: its recorded latent
        # length and decoded frame count shrink to the kept part, and the
        # tail trim is dropped (the cut itself ends the kept region).
        video_prefix["window"]["latent_len"] = int(kept_latents)
        video_prefix["window"]["frame_count"] = int(branch_point.get("decoded_prefix") or 0)
        video_prefix["window"]["tail_trim"] = 0

    audio_windows = None
    audio_prefix = None
    audio_entry = audio_latent_payload(sidecar_obj)
    if audio_entry is not None:
        audio_specs = audio_window_specs(audio_entry)
        if audio_specs:
            audio_blocks = audio_entry.get("windows") or []
            audio_windows = []
            for index, spec in enumerate(audio_specs[:k - 1]):
                block = dict(audio_blocks[index]) if index < len(audio_blocks) else {"window_no": spec.get("window_no")}
                block["window_no"] = index + 1
                block["latents"] = spec["latents"]
                audio_windows.append(block)
            if k - 1 < len(audio_specs):
                prefix_audio = audio_specs[k - 1]
                block = dict(audio_blocks[k - 1]) if k - 1 < len(audio_blocks) else {"window_no": prefix_audio.get("window_no")}
                block["window_no"] = k
                audio_prefix = {"window_no": k, "window": block, "latents": prefix_audio["latents"]}
    return {
        "video_windows": video_windows,
        "video_prefix": video_prefix,
        "audio_windows": audio_windows,
        "audio_prefix": audio_prefix,
        # The all-latent i2v start image is content, not a pointer: it
        # propagates verbatim through the carry-over so a descendant's
        # frame 0 stays the exact image (its decode splices it back in;
        # ``None`` for every non-i2v sidecar).
        "i2v_image": entry_i2v_image(video_entry),
    }


def branch_cut_window_block(sidecar_obj: dict, branch_point: dict) -> dict:
    """The branch's containing window as a stand-alone sidecar window
    ending at the resolved cut.

    For a branch job whose head window is a fresh encode (this run
    rejected the branch prefix, so the kept latents are not part of any
    generated window) the sidecar carries the containing window ``K``
    this way: its full latent run and metadata, with the committed range
    trimmed to end at the cut. A linear layout (Wan / LongCat / LTX2)
    keeps the window's full decoded length and extends the recorded
    tail trim to the cut's decoded position (a cut at the window's
    committed end leaves the block's recorded trims unchanged); the H3
    piecewise layout caps the decoded run at the same position (``h3_target_frames``)
    and drops the tail trim (the cap ends the kept region). A cut at the
    window's committed start yields a block that commits zero frames.

    The Latent Decode job's per-window reconstruction of the block is the
    same decode the branch's kept-region splice performs (the full latent
    run, the cut's position, the window's own head trim), so a sidecar
    carrying it reproduces the written video's kept region pixel for
    pixel — the self-contained file a rejected-prefix branch writes.

    ``sidecar_obj`` is the loaded source sidecar and ``branch_point`` the
    dict returned by :func:`branch_point_for_frame`. Returns the block
    (the window's metadata plus its full ``latents``), or ``None`` when
    the sidecar has no readable window for the branch point. Raises
    :class:`ValueError` when the source is not a current all-latent
    sidecar (a branch job never reaches that state: the carry-over
    rejects it first).
    """
    if not isinstance(branch_point, dict):
        raise ValueError("branch_point must be the dict returned by branch_point_for_frame")
    video_entry = latent_payload(sidecar_obj)
    if video_entry is None:
        return None
    if not is_all_latent_entry(video_entry):
        raise ValueError("the source latent file is not in the current all-latent layout; the window containing the cut cannot be carried")
    specs = entry_window_specs(video_entry)
    if not specs:
        return None
    k = branch_window_position(video_entry, branch_point, specs)
    spec = specs[k - 1]
    windows = video_entry.get("windows") or []
    block = dict(windows[k - 1]) if k - 1 < len(windows) else {}
    _copy_window_extra_fields(spec, block)
    block["latents"] = spec["latents"]
    decoded_prefix = int(branch_point.get("decoded_prefix") or 0)
    if _is_h3_spec(spec):
        # Piecewise layout: cap the decoded run at the cut and drop the
        # tail trim (the cap ends the kept region); the full latents stay
        # (the decode job caps the full decode at the recorded target).
        target = int(block.get("h3_target_frames") or 0)
        if target > 0:
            block["h3_target_frames"] = min(target, decoded_prefix)
        elif decoded_prefix > 0:
            block["h3_target_frames"] = decoded_prefix
        block["tail_trim"] = 0
    else:
        # Linear layout: keep the full decoded length; extend the tail
        # trim so the committed range ends at the cut's decoded position
        # (a cut at the committed end leaves the recorded value).
        decoded = _window_decoded_frames(video_entry, spec)
        if decoded > 0:
            block["tail_trim"] = max(int(block.get("tail_trim") or 0), decoded - decoded_prefix)
    return block


def branch_cut_audio_window_block(sidecar_obj, branch_point):
    """The branch's containing window as a stand-alone audio window ending
    at the resolved cut (the cut block's audio companion).

    For a branch job whose head window is a fresh encode (this run
    rejected the branch prefix, so the kept latents are not part of any
    generated window) the sidecar carries the containing window ``K``'s
    video as the capped cut block (see :func:`branch_cut_window_block`);
    this is its audio companion: the window's own pre-decode audio latents
    sliced at the resolved cut (the same time-ratio splice the frozen
    audio prefix uses), so the new sidecar's audio entry is self-contained
    the way its video entry is. The PDD fallback is the live case: the
    rejected prefix never freezes the window's audio (the fresh head
    window generates its own), and without this block the sidecar's audio
    entry skips the kept region, whose decode is silence (or a track
    shifted by one window).

    The slice keeps the sidecar's stored layout (``[c, count * mel]`` for
    the mel families, ``[64, count]`` for MiniMax H3); ``samples`` records
    the slice's share of the window's recorded sample count (its duration
    at the entry's ``sample_rate``), and ``start_sample`` keeps the
    window's recorded position in the final audio, shifted with the slice
    when it is the window's committed tail (the kept region sits at the
    same timeline position in the branch as in the source). The block
    carries no window number of its own: the destination builder numbers
    it positionally, like the carried windows.

    ``sidecar_obj`` is the loaded source sidecar (its
    ``wgp_latent/1`` and ``wgp_latent/audio/1`` entries) and
    ``branch_point`` the dict returned by :func:`branch_point_for_frame`.
    Returns the block (the slice's metadata plus its ``latents``), or
    ``None`` when the sidecar carries no usable audio for the branch
    point (no audio entry, no window block, a malformed block, or a
    degenerate slice). Raises :class:`ValueError` when the source is not a
    current all-latent sidecar (a branch job never reaches that state: the
    carry-over rejects it first).
    """
    if not isinstance(branch_point, dict):
        raise ValueError("branch_point must be the dict returned by branch_point_for_frame")
    video_entry = latent_payload(sidecar_obj)
    if video_entry is None:
        return None
    if not is_all_latent_entry(video_entry):
        raise ValueError("the source latent file is not in the current all-latent layout; the window containing the cut cannot be carried")
    audio_entry = audio_latent_payload(sidecar_obj)
    if audio_entry is None:
        return None
    slice_info = _branch_audio_prefix_slice(audio_entry, video_entry, branch_point)
    if slice_info is None:
        return None
    block = slice_info["block"]
    samples = int(block.get("samples") or 0)
    t_k = slice_info["t_k"]
    count = slice_info["count"]
    slice_samples = int(round(samples * count / t_k)) if t_k > 0 else 0
    start_sample = int(block.get("start_sample") or 0)
    if slice_info["tail"]:
        # The slice is the block's committed tail: its start is where that
        # tail begins in the window's recorded sample span.
        start_sample += max(0, samples - slice_samples)
    return {
        "start_sample": start_sample,
        "samples": slice_samples,
        "latents": slice_info["flat"],
    }


def branch_head_window(entry: dict, branch_point: dict, requested_new_frames: int, *, stride: int = 0, offset: int = 0) -> dict:
    """The branch-shaped head window for a sidecar's containing window.

    A branch job's head window is the containing window ``K`` of the
    resolved branch point, regenerated from the cut: its denoised tensor
    is ``[the kept latents of K (verbatim from the sidecar) | new
    latents]``. With the cut inside ``K`` (the mid-latent default, or the
    rounded resolution clamping to a boundary strictly inside ``K``) the
    new latents complete ``K`` up to its committed end, so the window's
    geometry (decoded length, committed range) matches ``K``'s own; with
    the cut at ``K``'s committed end the head window's new portion is a
    fresh standard window (the caller supplies its geometry) and the
    prefix is ``K``'s full latent run.

    ``requested_new_frames`` is the new segment the user asked for (the
    job's video length, which excludes the verbatim kept prefix).

    For linear layouts (Wan/LongCat 4/1, LTX2 8/1) the head window
    shrinks to fit a short request: its ``frame_num`` is the decoded
    prefix plus the request rounded up to the VAE's latent multiple.
    MiniMax H3 keeps ``K``'s own geometry instead — the clip-anchored
    transformer needs the identical clip layout ``K`` was generated with,
    and a window never falls below the H3 minimum of 124 frames:
    ``frame_num`` is ``K``'s recorded ``frame_count`` (on the 5+17k grid),
    the new latents run to ``K``'s last latent, the committed new portion
    is the request clamped to the cut's committed tail, and the overshoot
    is absorbed as ``tail_trim_new`` (which also re-applies ``K``'s own
    tail trim, the new block decoding at ``K``'s capped length).

    Returns a dict with ``prefix_latents`` (the kept latents' count in
    window ``K``, always a self-contained first ``P`` latents run so the
    new sidecar decodes cleanly), ``decoded_prefix`` (the pixel frames
    those latents decode to, the head trim of the branch job's window 1),
    the containing window's full ``latent_count``/``decoded_window``/
    ``start_frame``/``head_trim``/``tail_trim`` metadata, ``first_new_frame``
    (the 0-based index of the first regenerated frame), ``natural_new``
    (the new frames from the cut to ``K``'s committed end), ``committed_start``
    / ``committed_end`` (the containing window's committed range in the final
    video, as 0-based committed-frame counts, the end exclusive) and, when the
    cut is inside ``K``, ``frame_num``/``new_frames``/``tail_trim_new`` for the
    head window (the requested length rounded up to the VAE's latent
    multiple, the overshoot absorbed as a tail trim so the output ends
    exactly at the requested boundary).
    """
    if not isinstance(branch_point, dict):
        raise ValueError("branch_point must be the dict returned by branch_point_for_frame")
    specs = entry_window_specs(entry)
    k = branch_window_position(entry, branch_point, specs)
    spec = specs[k - 1]
    is_h3 = _is_h3_spec(spec)
    stride = int(stride or entry.get("latent_stride") or 0)
    offset = int(offset or entry.get("frame_offset") or 0)
    if stride <= 0:
        stride, offset = 4, 1
    prefix_latents = int(branch_point.get("latent_index") or 0)
    if not 0 < prefix_latents <= int(spec["latents"].shape[1]):
        raise ValueError(f"the branch point keeps {prefix_latents} latent(s) of window {k} with {int(spec['latents'].shape[1])}")
    if is_h3 and (prefix_latents < 2 or (prefix_latents - 2) % 5 != 0):
        raise ValueError(f"the H3 branch point keeps {prefix_latents} latent(s) of window {k}; the H3 prefix length sits on the 2+5k latent grid")
    full_latents = int(spec["latents"].shape[1])
    decoded_window = _window_decoded_frames(entry, spec) if is_h3 else window_pixel_frames(spec["latents"], stride, offset)
    head_trim = int(spec.get("head_trim") or 0)
    tail_trim = int(spec.get("tail_trim") or 0)
    start_frame = int(spec.get("start_frame") or 0)
    kept_frames = int(branch_point.get("kept_frames") or 0)
    # The containing window's committed range in the final video (the decode
    # job concatenates the committed runs in window order): it starts at the
    # committed frames of the earlier windows and ends after its own
    # committed length. ``committed_end`` is also the last committed frame
    # of the final video when the cut is at the window's committed end.
    committed_start = int(_window_committed_offsets(entry)[k - 1])
    committed_end = committed_start + _window_committed_frames(entry, spec)
    natural_new = max(0, committed_end - kept_frames)
    if natural_new > 0:
        # The cut falls inside the window's committed range: the new
        # latents complete the window (the decoded length is the window's
        # own, so the new sidecar's block stays a self-contained unit).
        prefix_count = prefix_latents
        target = max(1, min(natural_new, int(requested_new_frames or 0)))
        if is_h3:
            decoded_prefix = h3_video_latent_pixel_frames(prefix_count)
            # The generated window keeps K's own geometry: its frame count
            # is K's recorded one (on the 5+17k grid, at least the H3
            # minimum window of 124 frames), never shrunk to fit a short
            # request the way the linear layouts do. The new latents
            # complete K up to its last latent; the committed new portion
            # is the request clamped to the cut's committed tail, the
            # overshoot (including K's own tail share, the block decoding
            # at K's capped length) absorbed as the new block's tail trim.
            frame_num = int(spec.get("frame_count") or 0) or h3_video_latent_pixel_frames(full_latents)
            new_latents = full_latents - prefix_count
            tail = max(0, int(decoded_window) - decoded_prefix - target)
        else:
            decoded_prefix = window_pixel_frames(spec["latents"][:, :prefix_count], stride, offset)
            new_latents = (target + stride - 1) // stride
            frame_num = decoded_prefix + stride * new_latents
            tail = stride * new_latents - target
    else:
        # The cut is the window's committed end: the whole window is kept
        # and the head window's new portion is a fresh standard window
        # (the caller combines it with the standard first-window geometry).
        prefix_count = full_latents
        decoded_prefix = decoded_window
        target = 0
        new_latents = 0
        frame_num = 0
        tail = 0
    return {
        "window_no": k,
        "prefix_latents": prefix_count,
        "latent_count": full_latents,
        "decoded_prefix": int(decoded_prefix),
        "decoded_window": int(decoded_window),
        "start_frame": int(start_frame),
        "head_trim": head_trim,
        "tail_trim": tail_trim,
        "first_new_frame": kept_frames,
        "natural_new": int(natural_new),
        "cut_at_window_end": natural_new == 0,
        "requested_new_frames": int(requested_new_frames or 0),
        "new_frames": int(target),
        "frame_num": int(frame_num),
        "tail_trim_new": int(tail),
        # The containing window's committed range in the final video (the
        # 0-based committed-frame counts; the end is exclusive): the branch
        # job's sidecar writer records a pure-new (anchor-head) branch
        # window against this range instead of the window's recorded
        # start/trims.
        "committed_start": int(committed_start),
        "committed_end": int(committed_end),
    }


def h3_branch_head_plan(entry: dict, branch_point: dict, requested_new_frames) -> dict:
    """The branch head window plan for an all-latent MiniMax H3 sidecar.

    A pure function over the sidecar's window blocks: given the sidecar
    entry and a resolved branch point (a :func:`branch_point_for_frame`
    result) it returns the geometry the branch job's head window needs.
    It never touches the source video, never loads a model, and never
    mutates its inputs, so the same branch point plans identically from
    any copy of the sidecar — every length the H3 branch flow uses (the
    overlap, the guide, the anchor flag) comes from the plan itself.

    Every length routes through the decode job's own arithmetic
    (:func:`_window_decoded_frames`, :func:`_window_committed_frames`,
    :func:`_window_committed_offsets`), so the branch plan and the
    decode cannot drift: the kept decoded length is capped at the
    containing window's committed end (its capped decoded run minus its
    tail trim, in the window's decoded-run coordinates); ``kept_decoded``
    is that region's length in final-video terms (the cut's committed
    share of the containing window, its full committed run in case B);
    and ``assembly_overlap`` is the prefix of the head window's decoded
    output the assembly drops (also the head trim the branch sidecar
    records on the head window, so the decode reassembles the same
    video the job wrote): in case A the kept latents' decoded length
    (capped at the committed end), in case B the committed kept region
    itself.

    The case follows the cut: inside the containing window's committed
    range it is ``"A"`` — the head window is that window itself,
    regenerated from the cut with the kept latents frozen at its head,
    the new latents completing it up to its last latent, the committed
    new portion the request clamped to the cut's committed tail, and the
    overshoot absorbed as the new block's tail trim; at the window's
    committed end it is ``"B"`` — a fresh window whose kept portion is
    the containing window's full latent run (its geometry comes from
    the job's standard first-window plan), which ``anchor_head`` marks
    as the GAP-3 anchor-head window when the end is cap-driven (the
    recorded ``h3_target_frames`` below the natural decoded length):
    the pure-new window whose first decoded frame re-anchors the
    carried window's last frame.

    ``requested_new_frames`` is the new segment the user asked for (the
    job's video length, which excludes the verbatim kept prefix).
    Returns a dict with ``case`` / ``cut_at_window_end`` /
    ``anchor_head`` / ``clean_window_end`` (the Case B restriction the
    live flow applies: a head trim of at most one, no recorded tail
    trim), ``window_no``, ``prefix_latents`` (on the 2+5k latent grid),
    ``latent_count``, ``new_latents``, ``decoded_prefix`` /
    ``decoded_window`` / ``natural_decoded``, ``committed_start`` /
    ``committed_end`` (the containing window's committed range in the
    final video, 0-based with an exclusive end), ``kept_frames`` (the
    resolved cut, 1-based), ``kept_decoded`` (the committed share up to
    the cut — case A: shorter than the kept decoded run by the window's
    head share for a second-and-later window), ``assembly_overlap``,
    ``guide_frames`` (case A: the kept latents' decoded length — the
    model's conditioning region, which the pipeline's case check compares
    against that decoded length; case B: the committed kept region, the
    continuation the one-shot decode slices), ``head_frame_num`` /
    ``head_output`` / ``head_trim_new`` (case A: the containing window's
    recorded ``frame_count`` (on the 5+17k grid, at least the H3 minimum
    window), the request clamped to the cut's committed tail, and the
    absorbed tail trim; case B: ``None`` / ``None`` / 0),
    ``natural_new`` (case A: the new frames available from the cut to
    the containing window's committed end; case B: 0) and
    ``requested_new_frames``. Raises ``ValueError`` when the sidecar is
    not an all-latent H3 entry, the branch point does not reference one
    of its windows (or cuts outside the window's committed range), or
    the kept latents do not sit on the H3 2+5k latent grid.
    """
    if not isinstance(branch_point, dict):
        raise ValueError("branch_point must be the dict returned by branch_point_for_frame")
    if not is_all_latent_entry(entry):
        raise ValueError("h3_branch_head_plan plans all-latent files only")
    specs = entry_window_specs(entry)
    k = branch_window_position(entry, branch_point, specs)
    spec = specs[k - 1]
    if not _is_h3_spec(spec):
        raise ValueError(f"window {k} is not a MiniMax H3 window (no piecewise 5/17 frame metadata)")
    full_latents = int(spec["latents"].shape[1])
    if full_latents < 2 or (full_latents - 2) % 5 != 0:
        raise ValueError(f"window {k} holds {full_latents} latents, off the H3 2+5k latent grid")
    prefix_latents = int(branch_point.get("latent_index") or 0)
    if prefix_latents < 1:
        raise ValueError(f"the branch point keeps no latents of window {k}")
    if prefix_latents > full_latents:
        raise ValueError(f"the branch point keeps {prefix_latents} latent(s) of window {k} with {full_latents}")
    if prefix_latents < full_latents and (prefix_latents < 2 or (prefix_latents - 2) % 5 != 0):
        raise ValueError(f"the H3 branch point keeps {prefix_latents} latent(s) of window {k}; the H3 prefix length sits on the 2+5k latent grid")

    head_trim = int(spec.get("head_trim") or 0)
    # The part of the head trim that falls on the window's decoded run
    # (for a window with a rebuilt history prefix the excess over that
    # history, the one re-anchored shared frame).
    head_share = h3_head_trim_on_decoded_run(spec)
    tail_trim = int(spec.get("tail_trim") or 0)
    target = int(spec.get("h3_target_frames") or 0)
    committed = _window_committed_frames(entry, spec)
    committed_start = int(_window_committed_offsets(entry)[k - 1])
    committed_end = committed_start + committed
    natural = h3_video_latent_pixel_frames(full_latents)
    decoded_window = _window_decoded_frames(entry, spec)
    kept_frames = int(branch_point.get("kept_frames") or 0)
    # The all-latent i2v window splices the embedded start image into
    # frame 0 of the window: the video-to-decoded mapping shifts by one
    # (the same rule branch_point_for_frame applies to the cut).
    spliced = 1 if (int(spec.get("window_no") or 0) == 1 and entry_i2v_image(entry) is not None) else 0
    # The containing window's committed end in its own decoded-run
    # coordinates (the cuts and the kept latent boundaries live on this
    # grid): the capped decoded run minus the tail trim — the head trim
    # (and the spliced i2v image slot, which counts in the committed
    # total) cancel in the mapping the resolver uses for the cut.
    committed_end_decoded = max(0, decoded_window - tail_trim)

    cut_at_window_end = prefix_latents == full_latents
    if cut_at_window_end:
        # The cut is the window's committed end: the whole window is
        # kept (a fit cut may overrun a tail-trimmed committed end; the
        # kept region ends at the committed end regardless).
        if not committed_start < kept_frames:
            raise ValueError(f"the branch point (frame {kept_frames}) does not fall inside window {k}'s committed range ({committed_start}..{committed_end})")
        decoded_prefix = min(natural, committed_end_decoded)
        kept_decoded = max(0, decoded_prefix - head_share + spliced)
        # The committed kept region (the final-video frames the output
        # already carries up to the cut): the assembly drops it from the
        # head window's decoded output.
        assembly_overlap = max(1, committed)
        # The one-shot decode of the kept run slices this same committed
        # region as the continuation the model conditions on.
        guide_frames = max(1, kept_decoded)
        new_latents = 0
        natural_new = 0
        head_frame_num = None
        head_output = None
        head_trim_new = 0
        # GAP-3: the committed end is cap-driven (the recorded target
        # below the natural decoded length), so the fresh window's
        # first decoded frame re-anchors the carried window's last
        # frame.
        anchor_head = 0 < target < natural
    else:
        # The cut falls inside the window's committed range: the head
        # window is the window itself, regenerated from the cut with
        # the kept latents frozen at its head.
        if not committed_start < kept_frames <= committed_end:
            raise ValueError(f"the branch point (frame {kept_frames}) does not cut inside window {k}'s committed range ({committed_start}..{committed_end})")
        decoded_prefix = min(h3_video_latent_pixel_frames(prefix_latents), committed_end_decoded)
        kept_decoded = max(1, kept_frames - committed_start)
        # The head sample's leading frames decode the kept latents (the
        # window's head-trim share included), so the assembly drops the
        # whole kept decoded run: for a window-1 cut that is the
        # committed kept region itself.
        assembly_overlap = max(1, decoded_prefix)
        # The model's conditioning region is the kept latents' decoded
        # length itself (the pipeline's case check compares it against
        # that decoded length); the committed share up to the cut
        # (``kept_decoded``) is the head share shorter for a window past
        # the first.
        guide_frames = max(1, decoded_prefix)
        new_latents = full_latents - prefix_latents
        natural_new = max(0, committed_end - kept_frames)
        # The new latents complete the window (the block decodes at the
        # window's capped length): the committed new portion is the
        # request clamped to the cut's committed tail, the overshoot
        # (including the window's own tail share) absorbed as the new
        # block's tail trim.
        head_frame_num = int(spec.get("frame_count") or 0) or natural
        head_output = max(1, min(natural_new, int(requested_new_frames or 0)))
        head_trim_new = max(0, decoded_window - decoded_prefix - head_output)
        anchor_head = False

    return {
        "case": "B" if cut_at_window_end else "A",
        "cut_at_window_end": bool(cut_at_window_end),
        "anchor_head": bool(anchor_head),
        "clean_window_end": bool(head_trim <= 1 and tail_trim == 0),
        "window_no": int(k),
        "prefix_latents": int(prefix_latents),
        "latent_count": int(full_latents),
        "new_latents": int(new_latents),
        "decoded_prefix": int(decoded_prefix),
        "decoded_window": int(decoded_window),
        "natural_decoded": int(natural),
        "committed_start": int(committed_start),
        "committed_end": int(committed_end),
        "kept_frames": int(kept_frames),
        "kept_decoded": int(kept_decoded),
        "assembly_overlap": int(assembly_overlap),
        "guide_frames": int(guide_frames),
        "head_frame_num": int(head_frame_num) if head_frame_num is not None else None,
        "head_output": int(head_output) if head_output is not None else None,
        "head_trim_new": int(head_trim_new),
        "natural_new": int(natural_new),
        "requested_new_frames": int(requested_new_frames or 0),
    }


def decode_kept_windows(decode_window, kept_specs, kept_frames, *, is_i2v=False, i2v_image_frames=None):
    """The branch job's kept region as the first ``kept_frames`` frames of the
    sidecar's decoded video, assembled from a per-window decoder.

    ``decode_window`` is the caller's family VAE call for one window: it
    receives one spec (see :func:`entry_window_specs`) and returns that
    window's decoded frames as a ``[3, f, h, w]`` tensor in whatever pixel
    space the caller's Latent Decode job produces (uint8 for SDR, float for
    HDR), on CPU. The assembly mirrors the Latent Decode job exactly: each
    window's recorded head/tail trims are re-applied to its decoded run
    (a continuation window drops its decoded overlap head), the trimmed runs
    are tiled in window order, and the result is sliced to the cut. For an
    LTX2-style all-latent i2v sidecar (``is_i2v`` plus ``i2v_image_frames``,
    the embedded start image as ``[3, 1, h, w]`` in the decoded frames'
    pixel space) the image is spliced in front of window 1's trimmed run,
    the way the decode job writes frame 0. (Unlike the H3 decode, no family
    here exempts a window-1 head trim: the all-latent writers record no
    image slot in a non-LTX2 i2v window's trim.)

    ``kept_specs`` are the sidecar's kept windows (1..K, the branch cut's
    containing window included) in window order. Raises ``ValueError`` when
    a window cannot be decoded or the assembled run falls short of the cut
    (the branch job keeps its source-pixel prefix instead).
    """
    window_frames = []
    for spec in kept_specs:
        window_no = int(spec["window_no"])
        frames = decode_window(spec)
        if frames is None or int(frames.numel()) == 0:
            raise ValueError(f"window {window_no} could not be decoded")
        frames = frames.to("cpu")
        head_trim = int(spec.get("head_trim") or 0)
        tail_trim = int(spec.get("tail_trim") or 0)
        if head_trim > 0 or tail_trim > 0:
            frames = frames[:, head_trim:max(head_trim, frames.shape[1] - tail_trim)]
        if is_i2v and i2v_image_frames is not None and window_no == 1:
            image = i2v_image_frames.to(device=frames.device, dtype=frames.dtype)
            frames = torch.cat([image, frames], dim=1)
        window_frames.append(frames)
    if not window_frames:
        raise ValueError("the latent file has no readable kept windows")
    assembled = torch.cat(window_frames, dim=1) if len(window_frames) > 1 else window_frames[0]
    kept_frames = int(kept_frames)
    if int(assembled.shape[1]) < kept_frames:
        raise ValueError(f"the kept windows of the latent file decode to {int(assembled.shape[1])} frames, fewer than the branch cut ({kept_frames})")
    return assembled[:, :kept_frames]



def _branch_audio_prefix_slice(audio_entry, video_entry, branch_point):
    """The containing window's audio block sliced at the resolved cut, in
    the sidecar's stored layout.

    The one shared time-ratio splice behind the two frozen-prefix splices
    (:func:`branch_audio_prefix_latents`,
    :func:`h3_branch_audio_prefix_latents`) and the cut block's audio
    carry-over (:func:`branch_cut_audio_window_block`). The slice is pure
    time: the kept region covers the first ``length`` decoded frames of the
    containing window (the cut's decoded position with the cut inside the
    window, the window's committed length when the cut ends it), so it
    spans that fraction of the window's audio block, measured with the
    sidecar's recorded duration (the block's ``samples`` at the entry's
    ``sample_rate`` against ``length / fps``, falling back to the window's
    decoded length) and cut on the block's latent-time grid. With the cut
    inside the window (the mid-latent default, or the rounded resolution)
    the slice is the head of the block; with the cut at the window's
    committed end it is the block's committed tail (the kept frames start
    where the source window committed, i.e. after its own head trim).

    The layout follows the video spec: the mel families store the block as
    ``[c, t * mel]`` (the mel bins recorded at save time) and the slice
    keeps a whole number of mel frames; MiniMax H3 stores it flat (``[64,
    t]``, one column per latent time step) and the slice keeps a whole
    number of columns.

    ``audio_entry``/``video_entry`` are the sidecar's
    ``wgp_latent/audio/1`` and ``wgp_latent/1|2`` entries;
    ``branch_point`` is the dict returned by :func:`branch_point_for_frame`
    for the video entry. Returns a dict with ``flat`` (the 2D slice),
    ``count`` (the slice's latent time steps), ``t_k`` (the block's full
    step count), ``tail`` (whether the slice is the committed tail),
    ``layout`` (``"h3"`` / ``"mel"``) and ``block`` (the containing
    window's raw audio block), or ``None`` when the sidecar carries no
    usable audio for the branch point (no audio entry, no window block,
    missing mel/sample-rate/fps bookkeeping, or a degenerate slice).
    """
    if not isinstance(branch_point, dict) or not isinstance(audio_entry, dict) or not isinstance(video_entry, dict):
        return None
    if not is_all_latent_entry(video_entry):
        return None
    video_specs = entry_window_specs(video_entry)
    try:
        k = branch_window_position(video_entry, branch_point, video_specs)
    except ValueError:
        return None
    audio_specs = audio_window_specs(audio_entry)
    if not 1 <= k <= len(audio_specs):
        return None
    audio_spec = audio_specs[k - 1]
    # The block's recorded sample count (its duration at the entry's sample
    # rate) lives on the raw window block, not the normalized spec.
    audio_blocks = audio_entry.get("windows") or []
    audio_block = audio_blocks[k - 1] if k - 1 < len(audio_blocks) else {}
    video_spec = video_specs[k - 1]
    block = audio_spec["latents"]
    is_h3 = _is_h3_spec(video_spec)
    if is_h3:
        if block.dim() != 2 or block.shape[0] < 2 or block.shape[0] % 2 != 0:
            return None
        t_k = int(block.shape[1])
        full_latents = int(video_spec["latents"].shape[1])
        if t_k < 1 or full_latents < 1:
            return None
        decoded_window = _window_decoded_frames(video_entry, video_spec)
        if decoded_window < 1:
            return None
        if int(branch_point.get("latent_index") or 0) >= full_latents:
            # The cut ends the window: the frozen frames are the committed tail.
            length = _window_committed_frames(video_entry, video_spec)
            tail = True
        else:
            length = int(branch_point.get("decoded_prefix") or 0)
            tail = False
        if length < 1:
            return None
        # The block covers the window's target region; the ratio falls back
        # to the window's natural (uncapped) 5/17 decoded length.
        natural_window = h3_video_latent_pixel_frames(full_latents)
    else:
        mel = int(audio_entry.get("mel_bins") or 0)
        raw_len = int(block.shape[1])
        if mel <= 0 or raw_len < mel or raw_len % mel != 0:
            return None
        t_k = raw_len // mel
        full_latents = int(video_spec["latents"].shape[1])
        if t_k < 1 or full_latents < 1:
            return None
        stride = int(video_entry.get("latent_stride") or 0)
        offset = int(video_entry.get("frame_offset") or 0)
        if stride <= 0:
            stride, offset = 4, 1
        decoded_window = window_pixel_frames(video_spec["latents"], stride, offset)
        head_trim = int(video_spec.get("head_trim") or 0)
        tail_trim = int(video_spec.get("tail_trim") or 0)
        if decoded_window < 1:
            return None
        if int(branch_point.get("latent_index") or 0) >= full_latents:
            # The cut ends the window: the frozen frames are the committed tail.
            length = max(0, decoded_window - head_trim - tail_trim)
            tail = True
        else:
            length = int(branch_point.get("decoded_prefix") or 0)
            tail = False
        if length < 1:
            return None
        natural_window = decoded_window
    samples = int(audio_block.get("samples") or 0)
    sample_rate = int(audio_entry.get("sample_rate") or 0)
    fps = float(video_entry.get("fps") or 0)
    if samples > 0 and sample_rate > 0 and fps > 0:
        window_seconds = float(samples) / float(sample_rate)
        if window_seconds > 0:
            count = int(round(t_k * (float(length) / fps) / window_seconds))
        else:
            count = int(round(t_k * float(length) / natural_window))
    else:
        count = int(round(t_k * float(length) / natural_window))
    count = max(1, min(count, t_k))
    if is_h3:
        flat = block[:, (t_k - count):] if tail else block[:, :count]
    else:
        flat = block[:, (t_k - count) * mel:] if tail else block[:, :count * mel]
    return {
        "flat": flat,
        "count": int(count),
        "t_k": int(t_k),
        "tail": bool(tail),
        "layout": "h3" if is_h3 else "mel",
        "block": audio_block,
    }


def branch_audio_prefix_latents(audio_entry, video_entry, branch_point):
    """The sidecar's kept audio latent run, as a frozen prefix for the branch
    job's head window (LTX2 Phase 2).

    Instead of re-encoding the branch job's source audio waveform, the head
    window's audio prefix is the sidecar's own pre-decode latents, sliced by
    a pure time ratio: the frozen prefix covers the first ``length`` decoded
    frames of the containing window, so it spans that fraction of the
    window's audio block, measured with the sidecar's recorded duration
    (the block's ``samples`` at the entry's ``sample_rate`` against
    ``length / fps``). With the cut inside the window (the mid-latent
    default, or the rounded resolution) the prefix is the head of the
    block; with the cut at the window's committed end it is the block's
    committed tail (the branch window's frozen frames start where the
    source window committed, i.e. after its own head trim).

    ``audio_entry``/``video_entry`` are the sidecar's ``wgp_latent/audio/1``
    and ``wgp_latent/1|2`` entries; ``branch_point`` is the dict returned by
    :func:`branch_point_for_frame` for the video entry. Returns a batched
    ``[1, c, t, mel]`` tensor (the audio VAE's pre-decode layout) or ``None``
    when the sidecar carries no usable audio for the branch point (no audio
    entry, no window block, missing mel/sample-rate/fps bookkeeping, or a
    degenerate slice). The shared time-ratio splice behind this function
    and its H3 companion is :func:`_branch_audio_prefix_slice`.
    """
    slice_info = _branch_audio_prefix_slice(audio_entry, video_entry, branch_point)
    if slice_info is None or slice_info["layout"] != "mel":
        return None
    mel = int(audio_entry.get("mel_bins") or 0)
    raw = slice_info["flat"]
    return raw.reshape(1, int(raw.shape[0]), slice_info["count"], mel)


def h3_branch_audio_prefix_latents(audio_entry, video_entry, branch_point):
    """The sidecar's kept MiniMax H3 audio latents, as the branch job's
    frozen audio prefix (Phase 3).

    H3 stores its audio latent flat: the pipeline's batched ``[1, 32, 2, t]``
    state is recorded as ``[64, t]`` (the 32 latent channels times the two
    stereo bins on dimension 0, read row-major), so one audio latent time
    step is one column of 64 values. This is the H3 companion of
    :func:`branch_audio_prefix_latents`: the same head/tail time-ratio
    splice and clamps over that layout. The kept length follows the
    resolved cut: with the cut inside the window (the mid-latent default or
    the rounded resolution) the prefix is the head of the block; at the
    window's committed end it is the block's committed tail. The ratio
    compares the kept frames' duration at the video entry's ``fps`` against
    the block's recorded ``samples`` at the entry's ``sample_rate`` (falling
    back to the window's piecewise 5/17 decoded length when that
    bookkeeping is missing; the fallback uses the window's *natural* length
    — the block covers the full run, capped or not).

    ``audio_entry``/``video_entry`` are the sidecar's ``wgp_latent/audio/1``
    and ``wgp_latent/1`` entries; ``branch_point`` is the dict returned by
    :func:`branch_point_for_frame` for the video entry. Returns a dict with

    - ``batched``: the prefix as the pipeline's ``[1, 32, 2, N]`` audio
      state (Case A: placed at the head of the branch window's audio
      tensor and frozen through the run);
    - ``flat``: the same slice in the sidecar's ``[64, N]`` flat layout
      (Case B: the saved-audio-latent condition rows instead of a
      waveform re-encode);

    or ``None`` when the sidecar carries no usable audio for the branch
    point (no audio entry, no window block, a malformed block shape, or a
    degenerate slice).
    """
    if not isinstance(branch_point, dict) or not isinstance(audio_entry, dict) or not isinstance(video_entry, dict):
        return None
    if not is_all_latent_entry(video_entry):
        return None
    slice_info = _branch_audio_prefix_slice(audio_entry, video_entry, branch_point)
    if slice_info is None or slice_info["layout"] != "h3":
        return None
    flat = slice_info["flat"]
    channels = int(flat.shape[0]) // 2
    count = slice_info["count"]
    return {"batched": flat.reshape(1, channels, 2, count), "flat": flat}


def build_latent_file_bytes(payload: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def write_latent_file_bytes(path, data: bytes) -> str:
    """Atomically write latent file bytes (tmp file + fsync + rename)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".latent_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return path


def write_latent_file(path, payload: dict) -> str:
    return write_latent_file_bytes(path, build_latent_file_bytes(payload))


def load_latent_file(path):
    with open(path, "rb") as handle:
        data = handle.read()
    return torch.load(io.BytesIO(data), weights_only=True)


def latent_sidecar_path(media_path) -> str:
    media_path = str(media_path)
    directory = os.path.dirname(media_path)
    stem = os.path.splitext(os.path.basename(media_path))[0]
    return os.path.join(directory, stem + LATENT_SIDECAR_SUFFIX) if directory else stem + LATENT_SIDECAR_SUFFIX


def name_skeleton(name) -> str:
    """Normalize a filename the way Gradio mangles a browser upload.

    Gradio's upload temp file keeps only the characters its
    ``strip_invalid_filename_characters`` accepts (alphanumerics plus
    ``. _ - ,`` and spaces) and truncates the name to 200 bytes, so a name
    like ``...856_480fps@24fps_scale1.0...mp4`` arrives as
    ``...856_480fps24fps_scale1.0...mp4``. The skeleton (the kept characters
    only) is therefore identical for the original and the mangled name, and
    a truncated name is a leading chunk of the original's skeleton.
    """
    return "".join(char for char in str(name or "") if char.isalnum() or char in "._-, ")


def _skeleton_match(a, b) -> bool:
    """Whether two name skeletons describe the same file (an exact match, or
    a leading-chunk match when one name was truncated to the 200-byte upload
    limit). Short skeletons never match, so small names cannot pair with
    unrelated files."""
    a, b = str(a or ""), str(b or "")
    if len(a) < 8 or len(b) < 8:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    return longer.startswith(shorter) and len(shorter) >= int(0.8 * len(longer))


def find_latent_file_in_dirs(media_path, dirs) -> str | None:
    """Locate the saved latent file of ``media_path`` across candidate
    directories.

    ``find_latent_file`` only looks next to the file it is given. When the
    media was selected from outside the app's save path (or its upload temp
    name was mangled by Gradio) that lookup misses, so this helper
    additionally matches the media name against the ``*_latent.pt`` names in
    each candidate directory by :func:`name_skeleton`. Returns the best
    candidate (the most recently modified one when several match) or
    ``None``.
    """
    if len(str(media_path or "")) == 0:
        return None
    direct = find_latent_file(media_path)
    if direct is not None:
        return direct
    media_name = os.path.basename(str(media_path))
    target = name_skeleton(os.path.splitext(media_name)[0])
    if len(target) == 0:
        return None
    suffix = LATENT_SIDECAR_SUFFIX
    candidates = []
    for directory in dict.fromkeys(str(d or "") for d in (dirs or [])):
        if len(directory) == 0 or not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if not filename.endswith(suffix):
                continue
            base = filename[: -len(suffix)]
            if base.endswith("_latent"):
                base = base[: -len("_latent")]
            if _skeleton_match(target, name_skeleton(base)):
                candidates.append((os.path.getmtime(os.path.join(directory, filename)), os.path.join(directory, filename)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0][1]
    if len(candidates) > 1:
        print(f"Latent file lookup: {len(candidates)} saved latent files match {media_name}; using the most recent one ({os.path.basename(best)}).", flush=True)
    return best


def find_latent_file(media_path) -> str | None:
    """Locate the latent companion file of a media path (or the .pt itself)."""
    media_path = str(media_path or "").strip()
    if len(media_path) == 0:
        return None
    if media_path.lower().endswith(".pt"):
        return media_path if os.path.isfile(media_path) else None
    sidecar = latent_sidecar_path(media_path)
    return sidecar if os.path.isfile(sidecar) else None


# The app's video extensions (the *Continue Last Video* auto-resolve and
# the video pickers accept these): the reverse sidecar → video lookup
# probes them, since the sidecar name (<stem>_latent.pt) loses the
# video's own extension.
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv")


def find_video_for_latent(latent_path, search_dirs=None) -> list[str]:
    """Locate the companion video(s) of a latent file.

    The sidecar name is ``<stem>_latent.pt`` (the video's extension is
    lost), so the reverse lookup probes the app's video extensions
    (``.mp4`` / ``.mov`` / ``.mkv``) in the file's own directory and in
    ``search_dirs`` (the caller passes the same construction
    ``resolve_latent_branch`` uses: the save path, its ``videos``
    subdirectory, and ``branch_latent_search_dirs``). An exact stem
    match wins; otherwise names are matched with :func:`name_skeleton` /
    :func:`_skeleton_match` (the same mangled-upload-name tolerance as
    :func:`find_latent_file_in_dirs`). Returns de-duplicated absolute
    paths: empty, one, or several candidates — the caller decides
    (several are never guessed between).
    """
    latent_path = str(latent_path or "").strip()
    if len(latent_path) == 0 or not latent_path.lower().endswith(LATENT_SIDECAR_SUFFIX):
        return []
    stem = os.path.splitext(os.path.basename(latent_path))[0]
    # The writer names the sidecar <video stem>_latent.pt (
    # latent_sidecar_path): undo that. A doubled _latent (a sidecar of a
    # decode named after its own sidecar) strips one _latent, mirroring
    # the forward matcher (find_latent_file_in_dirs).
    if stem.endswith("_latent"):
        stem = stem[: -len("_latent")]
    if len(stem) == 0:
        return []
    # The file's own directory is always searched (the user can pick a
    # file from anywhere), then the caller's search directories.
    directories = [os.path.abspath(os.path.dirname(latent_path) or ".")]
    for directory in (search_dirs or []):
        directory = str(directory or "").strip()
        if len(directory) == 0:
            continue
        directories.append(os.path.abspath(directory))
    directories = list(dict.fromkeys(directories))
    # An exact stem match (in any of the app's video extensions) wins
    # outright: several can apply (two extensions, several directories),
    # and all of them are reported.
    exact = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for extension in _VIDEO_EXTENSIONS:
            path = os.path.join(directory, stem + extension)
            if os.path.isfile(path):
                exact.append(path)
    if exact:
        return sorted(set(exact))
    target = name_skeleton(stem)
    if len(target) == 0:
        return []
    candidates = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            extension = os.path.splitext(filename)[1].lower()
            if extension not in _VIDEO_EXTENSIONS:
                continue
            if _skeleton_match(target, name_skeleton(os.path.splitext(filename)[0])):
                candidates.append(os.path.join(directory, filename))
    return sorted(set(candidates))


def identity_warnings(entry: dict, current: dict) -> list[str]:
    """Compare a saved latent's identity against the current app state."""
    warnings = []
    for field in ("app_version", "model_type", "base_model_type"):
        saved = str(entry.get(field) or "")
        current_value = str(current.get(field) or "")
        if saved and current_value and saved != current_value:
            warnings.append(f"{field}: latent saved with '{saved}', current value is '{current_value}'")
    saved_hash = str(entry.get("vae_sha256") or "")
    current_hash = str(current.get("vae_sha256") or "")
    if saved_hash and current_hash and saved_hash != current_hash:
        warnings.append("VAE checkpoint: latent saved with a different VAE file than the one currently in use")
    saved_audio_hash = str(entry.get("audio_vae_sha256") or "")
    current_audio_hash = str(current.get("audio_vae_sha256") or "")
    if saved_audio_hash and current_audio_hash and saved_audio_hash != current_audio_hash:
        warnings.append("audio VAE checkpoint: latent saved with a different audio VAE file than the one currently in use")
    return warnings
