"""
Video Metadata Utilities

This module provides functionality to read and write JSON metadata to video files.
- MP4: Uses mutagen to store metadata in ©cmt tag
- MKV: Uses FFmpeg to store metadata in comment/description tags and source images as attached pictures
"""

import json
import mmap
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from shared.utils.video_decode import resolve_media_binary
from shared.utils.subprocess_stdin import child_kwargs

DEFAULT_RESERVED_VIDEO_METADATA_BYTES = 50 * 1024
_RESERVED_METADATA_KEY = "_wangp_metadata_reserved"
_MP4_COMMENT_TAG = "\xa9cmt"
_MP4_COMMENT_BOX = b"\xa9cmt"
_MKV_COMMENT_NAME = b"\x45\xa3\x87COMMENT"
_CONTAINER_COMMENT_TAGS = ("comment", "COMMENT", "description", "DESCRIPTION")
_MP4_METADATA_CONTAINERS = {b"moov", b"udta", b"meta", b"ilst", _MP4_COMMENT_BOX}
_MP4_METADATA_EXTENSIONS = (".mp4", ".mov")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _resolve_media_tool(name):
    return resolve_media_binary(name) or name


def _is_verbose_metadata_debug(verbose_level):
    try:
        return int(verbose_level or 0) >= 2
    except (TypeError, ValueError):
        return False


def _log_metadata_debug(verbose_level, message):
    if _is_verbose_metadata_debug(verbose_level):
        print(f"[Video Metadata] {message}")


def _normalize_metadata_text(text):
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return str(text or "").replace("\ufeff", "").rstrip("\0")


def _parse_metadata_text(text):
    payload = _normalize_metadata_text(text)
    if len(payload.strip()) == 0:
        return None
    try:
        metadata = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return None if isinstance(metadata, dict) and metadata.get(_RESERVED_METADATA_KEY) else metadata


def _encode_metadata_bytes(metadata_dict):
    return json.dumps(metadata_dict, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _pad_metadata_bytes(payload_bytes, reserved_bytes):
    reserved_bytes = int(reserved_bytes)
    if len(payload_bytes) > reserved_bytes:
        return None
    return payload_bytes + (b" " * (reserved_bytes - len(payload_bytes)))


def build_reserved_video_metadata_text(reserved_bytes=DEFAULT_RESERVED_VIDEO_METADATA_BYTES, metadata_dict=None):
    reserved_bytes = max(128, int(reserved_bytes))
    payload = _encode_metadata_bytes({_RESERVED_METADATA_KEY: True} if metadata_dict is None else metadata_dict)
    return _pad_metadata_bytes(payload, max(reserved_bytes, len(payload))).decode("utf-8")


def _escape_ffmetadata_value(value):
    text = str(value or "").replace("\\", "\\\\").replace("\r", "").replace("\n", "\\\n")
    return text.replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")


def _write_ffmetadata_file(file_path, tags):
    with open(file_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(";FFMETADATA1\n")
        for key, value in (tags or {}).items():
            if value is None:
                continue
            handle.write(f"{key}={_escape_ffmetadata_value(value)}\n")
    return file_path


def write_reserved_video_ffmetadata(file_path, reserved_bytes=DEFAULT_RESERVED_VIDEO_METADATA_BYTES, metadata_dict=None):
    return _write_ffmetadata_file(file_path, {"comment": build_reserved_video_metadata_text(reserved_bytes, metadata_dict)})


def _read_container_tags(file_path):
    ffprobe_path = _resolve_media_tool("ffprobe")
    result = subprocess.run([ffprobe_path, "-v", "error", "-show_entries", "format_tags", "-of", "json", file_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if result.returncode != 0:
        return {}
    try:
        tags = ((json.loads(result.stdout) or {}).get("format") or {}).get("tags") or {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {str(key): str(value) for key, value in tags.items() if value is not None}


def _make_temp_output_path(file_path, suffix):
    path = Path(file_path)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def _read_mp4_box(mm, offset, end):
    if offset + 8 > end:
        return None
    box_size = int.from_bytes(mm[offset:offset + 4], "big")
    box_type = bytes(mm[offset + 4:offset + 8])
    header_size = 8
    if box_size == 1:
        if offset + 16 > end:
            return None
        box_size = int.from_bytes(mm[offset + 8:offset + 16], "big")
        header_size = 16
    elif box_size == 0:
        box_size = end - offset
    box_end = offset + box_size
    if box_size < header_size or box_end > end:
        return None
    content_offset = offset + header_size + (4 if box_type == b"meta" else 0)
    return None if content_offset > box_end else (box_type, content_offset, box_end)


def _find_mp4_comment_slot_in_range(mm, start, end):
    cursor = start
    while cursor + 8 <= end:
        box = _read_mp4_box(mm, cursor, end)
        if box is None:
            break
        box_type, content_offset, box_end = box
        if box_type == _MP4_COMMENT_BOX:
            child_cursor = content_offset
            while child_cursor + 8 <= box_end:
                child_box = _read_mp4_box(mm, child_cursor, box_end)
                if child_box is None:
                    break
                child_type, child_content_offset, child_end = child_box
                if child_type == b"data" and child_content_offset + 8 <= child_end:
                    payload_offset = child_content_offset + 8
                    return payload_offset, child_end - payload_offset
                child_cursor = child_end
        elif box_type in _MP4_METADATA_CONTAINERS:
            slot = _find_mp4_comment_slot_in_range(mm, content_offset, box_end)
            if slot is not None:
                return slot
        cursor = box_end
    return None


def _find_mp4_comment_slot(file_path):
    with open(file_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return _find_mp4_comment_slot_in_range(mm, 0, len(mm))


def _find_reserved_metadata_text_slot(file_path):
    placeholder = _encode_metadata_bytes({_RESERVED_METADATA_KEY: True})
    with open(file_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            cursor = 0
            fallback_slot = None
            comment_slot = None
            while True:
                payload_offset = mm.find(placeholder, cursor)
                if payload_offset < 0:
                    break
                payload_end = payload_offset + len(placeholder)
                while payload_end < len(mm) and mm[payload_end] == 0x20:
                    payload_end += 1
                slot = payload_offset, payload_end - payload_offset
                fallback_slot = slot
                if mm.rfind(_MP4_COMMENT_BOX, max(0, payload_offset - 64), payload_offset) >= 0:
                    comment_slot = slot
                cursor = payload_offset + len(placeholder)
            return comment_slot or fallback_slot


def _find_mp4_metadata_slot(file_path):
    return _find_reserved_metadata_text_slot(file_path) or _find_mp4_comment_slot(file_path)


def _read_ebml_size(mm, offset):
    if offset >= len(mm):
        return None
    first = mm[offset]
    mask = 0x80
    length = 1
    while length <= 8 and not (first & mask):
        mask >>= 1
        length += 1
    if length > 8 or offset + length > len(mm):
        return None
    value = first & (mask - 1)
    for index in range(1, length):
        value = (value << 8) | mm[offset + index]
    return value, length


def _find_mkv_comment_slot(file_path):
    with open(file_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            cursor = 0
            while True:
                name_offset = mm.find(_MKV_COMMENT_NAME, cursor)
                if name_offset < 0:
                    return None
                search_start = name_offset + len(_MKV_COMMENT_NAME)
                search_end = min(len(mm), search_start + 8192)
                value_offset = mm.find(b"\x44\x87", search_start, search_end)
                if value_offset >= 0:
                    parsed_size = _read_ebml_size(mm, value_offset + 2)
                    if parsed_size is not None:
                        data_size, size_len = parsed_size
                        data_offset = value_offset + 2 + size_len
                        if data_offset + data_size <= len(mm):
                            return data_offset, data_size
                cursor = name_offset + len(_MKV_COMMENT_NAME)


def _write_metadata_slot(file_path, slot, payload_bytes):
    if slot is None:
        return False
    payload_offset, reserved_bytes = slot
    padded = _pad_metadata_bytes(payload_bytes, reserved_bytes)
    if padded is None:
        return False
    with open(file_path, "r+b") as handle:
        handle.seek(payload_offset)
        handle.write(padded)
    return True


def _maybe_update_metadata_in_place(file_path, payload_bytes, *, container_name, find_slot_fn, allow_inplace_update=False, verbose_level=0):
    if not allow_inplace_update:
        return False
    started_at = time.perf_counter()
    slot = find_slot_fn(file_path)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    if slot is None:
        _log_metadata_debug(verbose_level, f"{container_name}: no reserved metadata slot found in {elapsed_ms:.1f} ms")
        return False
    payload_offset, reserved_bytes = slot
    if len(payload_bytes) > int(reserved_bytes):
        _log_metadata_debug(verbose_level, f"{container_name}: reserved metadata slot too small ({len(payload_bytes)} > {int(reserved_bytes)}) after {elapsed_ms:.1f} ms")
        return False
    ok = _write_metadata_slot(file_path, slot, payload_bytes)
    if ok:
        _log_metadata_debug(verbose_level, f"{container_name}: updated metadata in place in {elapsed_ms:.1f} ms at offset {int(payload_offset)}")
    else:
        _log_metadata_debug(verbose_level, f"{container_name}: in-place update failed after slot lookup in {elapsed_ms:.1f} ms")
    return ok


def _copy_video_with_comment(file_path, metadata_text, source_images=None):
    ffmpeg_path = _resolve_media_tool("ffmpeg")
    temp_output_path = _make_temp_output_path(file_path, "metadata")
    meta_dir = tempfile.mkdtemp(prefix="wangp_metadata_")
    metadata_path = os.path.join(meta_dir, "comment.ffmeta")
    tags = {key: value for key, value in _read_container_tags(file_path).items() if str(key).lower() not in {"comment", "description"}}
    tags["comment"] = metadata_text
    try:
        _write_ffmetadata_file(metadata_path, tags)
        command = [ffmpeg_path, "-y", "-v", "error", "-i", file_path, "-f", "ffmetadata", "-i", metadata_path, "-map", "0", "-map_metadata", "1", "-c", "copy"]
        for attachment_no, (attachment_path, attachment_filename, mimetype) in enumerate(_materialize_mkv_attachments(source_images, meta_dir)):
            command += ["-attach", attachment_path, f"-metadata:s:t:{attachment_no}", f"mimetype={mimetype}", f"-metadata:s:t:{attachment_no}", f"filename={attachment_filename}"]
        command += [temp_output_path]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
        if result.returncode != 0 or not os.path.isfile(temp_output_path):
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            return False, (result.stderr or result.stdout or "").strip()
        os.replace(temp_output_path, file_path)
        return True, ""
    finally:
        shutil.rmtree(meta_dir, ignore_errors=True)

def _convert_image_to_bytes(img):
    """
    Convert various image formats to bytes suitable for MP4 cover art.
    
    Args:
        img: Can be:
            - PIL Image object
            - File path (str)
            - bytes
    
    Returns:
        tuple: (image_bytes, image_format)
            - image_bytes: Binary image data
            - image_format: AtomDataType constant (JPEG or PNG)
    """
    from mutagen.mp4 import AtomDataType
    from PIL import Image
    import io
    import os
    
    try:
        # If it's already bytes, detect format and return
        if isinstance(img, bytes):
            # Detect format from magic numbers
            if img.startswith(b'\x89PNG'):
                return img, AtomDataType.PNG
            else:
                return img, AtomDataType.JPEG
        
        # If it's a file path, read and convert
        if isinstance(img, str):
            if not os.path.exists(img):
                print(f"Warning: Image file not found: {img}")
                return None, None
            
            # Determine format from extension
            ext = os.path.splitext(img)[1].lower()
            
            # Open with PIL for conversion
            pil_img = Image.open(img)
            
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if pil_img.mode not in ('RGB', 'L'):
                if pil_img.mode == 'RGBA':
                    # Create white background for transparency
                    background = Image.new('RGB', pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[3])
                    pil_img = background
                else:
                    pil_img = pil_img.convert('RGB')
            
            # Save to bytes
            img_bytes = io.BytesIO()
            
            # Use PNG for lossless formats, JPEG for others
            if ext in ['.png', '.bmp', '.tiff', '.tif']:
                pil_img.save(img_bytes, format='PNG')
                img_format = AtomDataType.PNG
            else:
                pil_img.save(img_bytes, format='JPEG', quality=95)
                img_format = AtomDataType.JPEG
            
            return img_bytes.getvalue(), img_format
        
        # If it's a PIL Image
        if isinstance(img, Image.Image):
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Save to bytes (prefer PNG for quality)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue(), AtomDataType.PNG
        
        print(f"Warning: Unsupported image type: {type(img)}")
        return None, None
        
    except Exception as e:
        print(f"Error converting image to bytes: {e}")
        return None, None


def _safe_metadata_filename_part(text, default):
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(text or "").strip())
    safe = safe.strip("._")
    return safe or default


def _split_mkv_attachment_filename(filename):
    filename = os.path.basename(str(filename or ""))
    stem, ext = os.path.splitext(filename)
    if "__" in stem:
        tag, original_stem = stem.split("__", 1)
        return _safe_metadata_filename_part(tag, "unknown"), _safe_metadata_filename_part(original_stem, "source") + ext
    return "unknown", _safe_metadata_filename_part(filename, "source.png")


def _normalize_source_images(source_images):
    if not source_images:
        return None
    normalized = {}
    if isinstance(source_images, dict):
        iterable = source_images.items()
    else:
        iterable = ((_split_mkv_attachment_filename(img)[0], img) for img in list(source_images if isinstance(source_images, (list, tuple)) else [source_images]))
    for img_tag, img_data in iterable:
        img_list = img_data if isinstance(img_data, list) else [img_data]
        valid_images = [img for img in img_list if img is not None]
        if valid_images:
            normalized.setdefault(str(img_tag or "unknown"), []).extend(valid_images)
    return normalized or None


def _image_extension_and_mimetype(image_bytes):
    if image_bytes.startswith(b"\x89PNG"):
        return ".png", "image/png"
    return ".jpg", "image/jpeg"


def _materialize_mkv_attachments(source_images, work_dir):
    source_images = _normalize_source_images(source_images)
    if source_images is None:
        return []
    attachments = []
    used_filenames = set()
    for img_tag, img_list in source_images.items():
        safe_tag = _safe_metadata_filename_part(img_tag, "unknown")
        for image_no, img in enumerate(img_list):
            image_bytes, _image_format = _convert_image_to_bytes(img)
            if not image_bytes:
                continue
            extension, mimetype = _image_extension_and_mimetype(image_bytes)
            if isinstance(img, str) and os.path.exists(img):
                _attachment_tag, original_filename = _split_mkv_attachment_filename(os.path.basename(img))
                original_name = os.path.splitext(original_filename)[0]
            else:
                original_name = f"source_{image_no}"
            safe_name = _safe_metadata_filename_part(original_name, f"source_{image_no}")
            attachment_filename = f"{safe_tag}__{safe_name}{extension}"
            base_name, ext = os.path.splitext(attachment_filename)
            counter = 1
            while attachment_filename.lower() in used_filenames:
                attachment_filename = f"{base_name}_{counter}{ext}"
                counter += 1
            used_filenames.add(attachment_filename.lower())
            attachment_path = os.path.join(work_dir, attachment_filename)
            with open(attachment_path, "wb") as handle:
                handle.write(image_bytes)
            attachments.append((attachment_path, attachment_filename, mimetype))
    return attachments


def embed_source_images_metadata_mp4(file, source_images):
    from mutagen.mp4 import MP4, MP4Cover, AtomDataType
    import json
    import os
    
    source_images = _normalize_source_images(source_images)
    if not source_images:
        return file
    
    try:
        if file.tags is None:
            file.add_tags()
        
        # Convert source images to cover art and build metadata
        cover_data = []
        image_metadata = {}  # Maps tag to list of {index, filename, extension}
        
        # Process each source image type
        for img_tag, img_data in source_images.items():
            if img_data is None:
                continue
            
            tag_images = []
            
            # Normalize to list for uniform processing
            img_list = img_data if isinstance(img_data, list) else [img_data]
            
            for img in img_list:
                if img is not None:
                    cover_bytes, image_format = _convert_image_to_bytes(img)
                    if cover_bytes:
                        # Extract filename and extension
                        if isinstance(img, str) and os.path.exists(img):
                            filename = os.path.basename(img)
                            extension = os.path.splitext(filename)[1]
                        else:
                            # PIL Image or unknown - infer from format
                            extension = '.png' if image_format == AtomDataType.PNG else '.jpg'
                            filename = f"{img_tag}{extension}"
                        
                        tag_images.append({
                            'index': len(cover_data),
                            'filename': filename,
                            'extension': extension
                        })
                        cover_data.append(MP4Cover(cover_bytes, image_format))
            
            if tag_images:
                image_metadata[img_tag] = tag_images
        
        if cover_data:
            file.tags['----:com.apple.iTunes:EMBEDDED_IMAGES'] = cover_data
            # Store the complete metadata as JSON
            file.tags['----:com.apple.iTunes:IMAGE_METADATA'] = json.dumps(image_metadata).encode('utf-8')
            # print(f"Successfully embedded {len(cover_data)} cover images")
            # print(f"Image tags: {list(image_metadata.keys())}")
        
    except Exception as e:
        print(f"Failed to embed cover art with mutagen: {e}")
        print(f"This might be due to image format or MP4 file structure issues")
    
    return file


def _legacy_save_metadata_to_mp4(file_path, metadata_dict, source_images = None):
    """
    Legacy MP4 metadata writer kept for reference.
    
    Args:
        file_path (str): Path to MP4 file
        metadata_dict (dict): Metadata dictionary to save
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from mutagen.mp4 import MP4
        file = MP4(file_path)
        file.tags['©cmt'] = [json.dumps(metadata_dict)]
        if source_images is not None:
            embed_source_images_metadata_mp4(file, source_images)
        file.save()
        return True
    except Exception as e:
        print(f"Error saving metadata to MP4 {file_path}: {e}")
        return False


def _legacy_save_metadata_to_mkv(file_path, metadata_dict):
    """
    Legacy MKV metadata writer kept for reference.
    
    Args:
        file_path (str): Path to MKV file
        metadata_dict (dict): Metadata dictionary to save
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create temporary file with metadata
        temp_path = file_path.replace('.mkv', '_temp_with_metadata.mkv')
        
        # Use FFmpeg to add metadata while preserving ALL streams (including attachments)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-metadata', f'comment={json.dumps(metadata_dict)}',
            '-map', '0',  # Map all streams from input (including attachments)
            '-c', 'copy',  # Copy streams without re-encoding
            temp_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, **child_kwargs())
        
        if result.returncode == 0:
            # Replace original with metadata version
            shutil.move(temp_path, file_path)
            return True
        else:
            print(f"Warning: Failed to add metadata to MKV file: {result.stderr}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
                
    except Exception as e:
        print(f"Error saving metadata to MKV {file_path}: {e}")
        return False



def _legacy_save_video_metadata(file_path, metadata_dict, source_images=  None):
    """
    Legacy video metadata writer kept for reference.
    
    Args:
        file_path (str): Path to video file
        metadata_dict (dict): Metadata dictionary to save
    
    Returns:
        bool: True if successful, False otherwise
    """

    if str(file_path).lower().endswith(_MP4_METADATA_EXTENSIONS):
        return save_metadata_to_mp4(file_path, metadata_dict, source_images)
    elif str(file_path).lower().endswith('.mkv'):
        return save_metadata_to_mkv(file_path, metadata_dict)
    else:
        return False


def _legacy_read_metadata_from_mp4(file_path):
    """
    Legacy MP4 metadata reader kept for reference.
    
    Args:
        file_path (str): Path to MP4 file
    
    Returns:
        dict or None: Metadata dictionary if found, None otherwise
    """
    try:
        from mutagen.mp4 import MP4
        file = MP4(file_path)
        tags = file.tags['©cmt'][0]
        return json.loads(tags)
    except Exception:
        return None


def _legacy_read_metadata_from_mkv(file_path):
    """
    Legacy MKV metadata reader kept for reference.
    
    Args:
        file_path (str): Path to MKV file
    
    Returns:
        dict or None: Metadata dictionary if found, None otherwise
    """
    try:
        # Try to get metadata using ffprobe
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', file_path
        ], capture_output=True, text=True, **child_kwargs())
        
        if result.returncode == 0:
            probe_data = json.loads(result.stdout)
            format_tags = probe_data.get('format', {}).get('tags', {})
            
            # Look for our metadata in various possible tag locations
            for tag_key in ['comment', 'COMMENT', 'description', 'DESCRIPTION']:
                if tag_key in format_tags:
                    try:
                        return json.loads(format_tags[tag_key])
                    except:
                        continue
        return None
    except Exception:
        return None


def _legacy_read_metadata_from_video(file_path):
    """
    Legacy video metadata reader kept for reference.
    
    Args:
        file_path (str): Path to video file
    
    Returns:
        dict or None: Metadata dictionary if found, None otherwise
    """
    if str(file_path).lower().endswith(_MP4_METADATA_EXTENSIONS):
        return read_metadata_from_mp4(file_path)
    elif str(file_path).lower().endswith('.mkv'):
        return read_metadata_from_mkv(file_path)
    else:
        return None


def save_metadata_to_mp4(file_path, metadata_dict, source_images = None, allow_inplace_update=False, verbose_level=0):
    metadata_text = json.dumps(metadata_dict, ensure_ascii=False, separators=(",", ":"))
    payload_bytes = metadata_text.encode("utf-8")
    source_images = _normalize_source_images(source_images)
    if source_images is None and _maybe_update_metadata_in_place(file_path, payload_bytes, container_name="MP4", find_slot_fn=_find_mp4_metadata_slot, allow_inplace_update=allow_inplace_update, verbose_level=verbose_level):
        return True
    if source_images is not None:
        _log_metadata_debug(verbose_level, "MP4: skipping in-place update because embedded images are being written too")
    try:
        from mutagen.mp4 import MP4
        file = MP4(file_path)
        if file.tags is None:
            file.add_tags()
        file.tags[_MP4_COMMENT_TAG] = [metadata_text]
        if source_images is not None:
            embed_source_images_metadata_mp4(file, source_images)
        file.save()
        _log_metadata_debug(verbose_level, "MP4: used standard metadata save path (non in-place)")
        return True
    except Exception as e:
        print(f"Error saving metadata to MP4 {file_path}: {e}")
        return False


def save_metadata_to_mkv(file_path, metadata_dict, source_images=None, allow_inplace_update=False, verbose_level=0):
    metadata_text = json.dumps(metadata_dict, ensure_ascii=False, separators=(",", ":"))
    payload_bytes = metadata_text.encode("utf-8")
    source_images = _normalize_source_images(source_images)
    if source_images is None and _maybe_update_metadata_in_place(file_path, payload_bytes, container_name="MKV", find_slot_fn=_find_mkv_comment_slot, allow_inplace_update=allow_inplace_update, verbose_level=verbose_level):
        return True
    if source_images is not None:
        _log_metadata_debug(verbose_level, "MKV: skipping in-place update because embedded images are being written too")
    try:
        ok, error = _copy_video_with_comment(file_path, metadata_text, source_images)
        if ok:
            _log_metadata_debug(verbose_level, "MKV: created a rewritten container copy to store metadata")
            return True
        print(f"Warning: Failed to add metadata to MKV file: {error}")
        return False
    except Exception as e:
        print(f"Error saving metadata to MKV {file_path}: {e}")
        return False


def save_video_metadata(file_path, metadata_dict, source_images=  None, allow_inplace_update=False, verbose_level=0):
    source_images = _normalize_source_images(source_images)
    if str(file_path).lower().endswith(_MP4_METADATA_EXTENSIONS):
        return save_metadata_to_mp4(file_path, metadata_dict, source_images, allow_inplace_update=allow_inplace_update, verbose_level=verbose_level)
    if str(file_path).lower().endswith('.mkv'):
        return save_metadata_to_mkv(file_path, metadata_dict, source_images, allow_inplace_update=allow_inplace_update, verbose_level=verbose_level)
    return False


def read_metadata_from_mp4(file_path):
    try:
        from mutagen.mp4 import MP4
        file = MP4(file_path)
        tag_values = file.tags.get(_MP4_COMMENT_TAG) if file.tags is not None else None
        metadata = None if not tag_values else _parse_metadata_text(tag_values[0])
        if metadata is not None:
            return metadata
    except Exception:
        pass
    tags = _read_container_tags(file_path)
    for tag_key in _CONTAINER_COMMENT_TAGS:
        metadata = _parse_metadata_text(tags.get(tag_key))
        if metadata is not None:
            return metadata
    return None


def read_metadata_from_mkv(file_path):
    try:
        ffprobe_path = _resolve_media_tool("ffprobe")
        result = subprocess.run([ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_format", file_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
        if result.returncode == 0:
            probe_data = json.loads(result.stdout)
            format_tags = probe_data.get("format", {}).get("tags", {})
            for tag_key in _CONTAINER_COMMENT_TAGS:
                metadata = _parse_metadata_text(format_tags.get(tag_key))
                if metadata is not None:
                    return metadata
        return None
    except Exception:
        return None


def read_metadata_from_video(file_path):
    if str(file_path).lower().endswith(_MP4_METADATA_EXTENSIONS):
        return read_metadata_from_mp4(file_path)
    if str(file_path).lower().endswith('.mkv'):
        return read_metadata_from_mkv(file_path)
    return None

def _extract_mp4_cover_art(video_path, output_dir = None):
    """
    Extract cover art from MP4 files using mutagen with proper tag association.
    
    Args:
        video_path (str): Path to the MP4 file
        output_dir (str): Directory to save extracted images
    
    Returns:
        dict: Dictionary mapping tags to lists of extracted image file paths
              Format: {tag_name: [path1, path2, ...], ...}
    """
    try:
        from mutagen.mp4 import MP4
        import json
        
        file = MP4(video_path)
        
        if file.tags is None or '----:com.apple.iTunes:EMBEDDED_IMAGES' not in file.tags:
            return {}
        
        cover_art =  file.tags['----:com.apple.iTunes:EMBEDDED_IMAGES']
        
        # Retrieve the image metadata
        metadata_data = file.tags.get('----:com.apple.iTunes:IMAGE_METADATA')
        
        if metadata_data:
            # Deserialize metadata and extract with original filenames
            image_metadata = json.loads(metadata_data[0].decode('utf-8'))
            extracted_files = {}
            
            for tag, tag_images in image_metadata.items():
                extracted_files[tag] = []
                
                for img_info in tag_images:
                    cover_idx = img_info['index']
                    
                    if cover_idx >= len(cover_art):
                        continue
                    if output_dir is None: output_dir = _create_temp_dir()
                    os.makedirs(output_dir, exist_ok=True)

                    cover = cover_art[cover_idx]
                    
                    # Use original filename
                    filename = img_info['filename']
                    output_file = os.path.join(output_dir, filename)
                    
                    # Handle duplicate filenames by adding suffix
                    if os.path.exists(output_file):
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(output_file):
                            filename = f"{base}_{counter}{ext}"
                            output_file = os.path.join(output_dir, filename)
                            counter += 1


                    # Write cover art to file
                    with open(output_file, 'wb') as f:
                        f.write(cover)
                    
                    if os.path.exists(output_file):
                        extracted_files[tag].append(output_file)
            
            return extracted_files
        
        else:
            # Fallback: Extract all images with generic naming
            print(f"Warning: No IMAGE_METADATA found in {video_path}, using generic extraction")
            extracted_files = {'unknown': []}
            
            for i, cover in enumerate(cover_art):
                if output_dir is None: output_dir = _create_temp_dir()
                os.makedirs(output_dir, exist_ok=True)

                filename = f"cover_art_{i}.jpg"
                output_file = os.path.join(output_dir, filename)
                
                with open(output_file, 'wb') as f:
                    f.write(cover)
                
                if os.path.exists(output_file):
                    extracted_files['unknown'].append(output_file)
            
            return extracted_files
        
    except Exception as e:
        print(f"Error extracting cover art from MP4: {e}")
        return {}

def _create_temp_dir():
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def extract_source_images(video_path, output_dir = None):
    
    # Handle MP4 files with mutagen
    if video_path.lower().endswith(_MP4_METADATA_EXTENSIONS):
        return _extract_mp4_cover_art(video_path, output_dir)
    if output_dir is None:
        output_dir = _create_temp_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Handle MKV files with ffmpeg attached pictures.
    try:
        ffprobe_path = _resolve_media_tool("ffprobe")
        ffmpeg_path = _resolve_media_tool("ffmpeg")
        # First, probe the video to find attachment streams (attached pics)
        probe_cmd = [
            ffprobe_path, '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', video_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, **child_kwargs())
        import json as json_module
        probe_data = json_module.loads(result.stdout)
        
        # Find attachment streams (attached pics)
        attachment_streams = []
        for i, stream in enumerate(probe_data.get('streams', [])):
            # Check for attachment streams in multiple ways:
            # 1. Traditional attached_pic flag
            # 2. Video streams with image-like metadata (filename, mimetype)
            # 3. MJPEG codec which is commonly used for embedded images
            is_attached_pic = stream.get('disposition', {}).get('attached_pic', 0) == 1
            
            # Check for image metadata in video streams (our case after metadata embedding)
            tags = stream.get('tags', {})
            has_image_metadata = (
                'FILENAME' in tags and tags['FILENAME'].lower().endswith(('.jpg', '.jpeg', '.png')) or
                'filename' in tags and tags['filename'].lower().endswith(('.jpg', '.jpeg', '.png')) or
                'MIMETYPE' in tags and tags['MIMETYPE'].startswith('image/') or
                'mimetype' in tags and tags['mimetype'].startswith('image/')
            )
            
            # Check for MJPEG codec (common for embedded images)
            is_mjpeg = stream.get('codec_name') == 'mjpeg'
            
            if (stream.get('codec_type') == 'video' and 
                (is_attached_pic or (has_image_metadata and is_mjpeg))):
                attachment_streams.append(i)
        
        if not attachment_streams:
            return {}
        
        # Extract each attachment stream
        extracted_files = {}
        used_filenames = set()  # Track filenames to avoid collisions
        
        for stream_idx in attachment_streams:
            # Get original filename from metadata if available
            stream_info = probe_data['streams'][stream_idx]
            tags = stream_info.get('tags', {})
            original_filename = (
                tags.get('filename') or 
                tags.get('FILENAME') or 
                f'attachment_{stream_idx}.png'
            )
            
            img_tag, original_filename = _split_mkv_attachment_filename(original_filename)
            
            # Clean filename for filesystem
            safe_filename = os.path.basename(original_filename)
            if not safe_filename.lower().endswith(_IMAGE_EXTENSIONS):
                safe_filename += '.png'
            
            # Handle filename collisions
            base_name, ext = os.path.splitext(safe_filename)
            counter = 0
            final_filename = safe_filename
            while final_filename in used_filenames:
                counter += 1
                final_filename = f"{base_name}_{counter}{ext}"
            used_filenames.add(final_filename)
            
            output_file = os.path.join(output_dir, final_filename)
            
            # Extract the attachment stream
            extract_cmd = [
                ffmpeg_path, '-y', '-v', 'error', '-i', video_path,
                '-map', f'0:{stream_idx}', '-frames:v', '1', '-update', '1',
                output_file
            ]
            
            try:
                subprocess.run(extract_cmd, capture_output=True, text=True, check=True, **child_kwargs())
                if os.path.exists(output_file):
                    extracted_files.setdefault(img_tag, []).append(output_file)
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract attachment {stream_idx} from {os.path.basename(video_path)}: {e.stderr}")
        
        return extracted_files
            
    except subprocess.CalledProcessError as e:
        print(f"Error extracting source images from {os.path.basename(video_path)}: {e.stderr}")
        return {}

