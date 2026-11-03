# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import inspect
import os, shutil, sys, time
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import quote
from tqdm import tqdm
from shared.utils.download_progress import DownloadCancelled, check_download_cancelled, download_context, download_operation, install_hf_download_patch, resolve_download_gen


class DownloadError(Exception):
    """An asset transfer failed; stop the operation before trying to load it."""


def generation_downloads(get_gen_info, get_offloadobj=lambda: None):
    from functools import wraps

    def decorate(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def run(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            arguments = bound.arguments
            gen = get_gen_info(arguments["state"])
            from shared.utils.media_control import MediaControl, MediaProcessingAborted
            with download_operation(gen), MediaControl(gen, arguments["send_cmd"], get_offloadobj) as control:
                arguments["send_cmd"] = control.send_cmd
                try:
                    return fn(*bound.args, **bound.kwargs)
                except (DownloadCancelled, MediaProcessingAborted):
                    gen["abort"] = True
                    return True  # The queue/API reports this as cancellation from gen["abort"].
                except DownloadError as exc:
                    message = str(exc)
                    tqdm.write(message)
                    arguments["send_cmd"]("error", message)
                    return False
        return run
    return decorate


def _offline():
    # --offline sets HF_HUB_OFFLINE before Hugging Face libraries load, which then never download either
    from huggingface_hub import constants
    return constants.HF_HUB_OFFLINE


def _ensure_online(name, url):
    if _offline():
        raise DownloadError(f"{name} is missing and WanGP runs offline (--offline): copy it into a checkpoints folder, or run once without --offline to download it\nDownload URL: {url}")


def _hf_download(*, gen=None, show_filename=True, file_index=1, file_count=1, **kwargs):
    from huggingface_hub import hf_hub_download, hf_hub_url

    url = hf_hub_url(kwargs["repo_id"], kwargs["filename"], **{key: kwargs[key] for key in ("subfolder", "repo_type", "revision", "endpoint") if key in kwargs})
    _ensure_online(kwargs["filename"], url)
    gen = resolve_download_gen(gen)
    install_hf_download_patch()
    try:
        with download_context(gen, kwargs["filename"], show_filename, file_index, file_count):
            return hf_hub_download(**kwargs)
    except DownloadCancelled:
        raise
    except Exception as exc:
        raise DownloadError(f"Unable to Download {kwargs['filename']} From {kwargs['repo_id']}: {exc}\nDownload URL: {url}") from exc

# Global variables to track download progress
_start_time = None
_last_time = None
_last_downloaded = 0
_speed_history = []
_update_interval = 0.5  # Update speed every 0.5 seconds

def progress_hook(block_num, block_size, total_size, filename=None):
    """
    Simple progress bar hook for urlretrieve
    
    Args:
        block_num: Number of blocks downloaded so far
        block_size: Size of each block in bytes
        total_size: Total size of the file in bytes
        filename: Name of the file being downloaded (optional)
    """
    global _start_time, _last_time, _last_downloaded, _speed_history, _update_interval
    
    current_time = time.time()
    downloaded = block_num * block_size
    
    # Initialize timing on first call
    if _start_time is None or block_num == 0:
        _start_time = current_time
        _last_time = current_time
        _last_downloaded = 0
        _speed_history = []
    
    # Calculate download speed only at specified intervals
    speed = 0
    if current_time - _last_time >= _update_interval:
        if _last_time > 0:
            current_speed = (downloaded - _last_downloaded) / (current_time - _last_time)
            _speed_history.append(current_speed)
            # Keep only last 5 speed measurements for smoothing
            if len(_speed_history) > 5:
                _speed_history.pop(0)
            # Average the recent speeds for smoother display
            speed = sum(_speed_history) / len(_speed_history)
        
        _last_time = current_time
        _last_downloaded = downloaded
    elif _speed_history:
        # Use the last calculated average speed
        speed = sum(_speed_history) / len(_speed_history)
    # Format file sizes and speed
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}TB"
    
    file_display = filename if filename else "Unknown file"
    
    if total_size <= 0:
        # If total size is unknown, show downloaded bytes
        speed_str = f" @ {format_bytes(speed)}/s" if speed > 0 else ""
        line = f"\r{file_display}: {format_bytes(downloaded)}{speed_str}"
        # Clear any trailing characters by padding with spaces
        sys.stdout.write(line.ljust(80))
        sys.stdout.flush()
        return
    
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    
    # Create progress bar (40 characters wide to leave room for other info)
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Format file sizes and speed
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}TB"
    
    speed_str = f" @ {format_bytes(speed)}/s" if speed > 0 else ""
    
    # Display progress with filename first
    line = f"\r{file_display}: [{bar}] {percent:.1f}% ({format_bytes(downloaded)}/{format_bytes(total_size)}){speed_str}"
    # Clear any trailing characters by padding with spaces
    sys.stdout.write(line.ljust(100))
    sys.stdout.flush()
    
    # Print newline when complete
    if percent >= 100:
        print()

# Wrapper function to include filename in progress hook
def create_progress_hook(filename):
    """Creates a progress hook with the filename included"""
    global _start_time, _last_time, _last_downloaded, _speed_history
    # Reset timing variables for new download
    _start_time = None
    _last_time = None
    _last_downloaded = 0
    _speed_history = []
    
    def hook(block_num, block_size, total_size):
        return progress_hook(block_num, block_size, total_size, filename)
    return hook


def process_files_def(repoId=None, sourceFolderList=None, fileList=None, targetFolderList=None, gen=None, show_filename=True):
    from huggingface_hub import HfApi, constants, snapshot_download
    from shared.utils import files_locator as fl

    gen = resolve_download_gen(gen)
    check_download_cancelled(gen)
    install_hf_download_patch()
    if targetFolderList is None:
        targetFolderList = [None] * len(sourceFolderList)
    downloads = []
    for targetFolder, sourceFolder, files in zip(targetFolderList, sourceFolderList, fileList):
        if targetFolder is not None and len(targetFolder) == 0:
            targetFolder = None
        explicit_target = targetFolder if targetFolder is not None else (sourceFolder if len(sourceFolder) > 0 else None)
        targetRoot = fl.get_smart_download_root(explicit_target)
        local_dir = os.path.join(targetRoot, targetFolder) if targetFolder is not None else targetRoot
        if len(files) == 0:
            if gen is None or _offline():  # offline: the remote listing is unreachable, a folder present locally is taken as complete
                if fl.locate_folder(sourceFolder if targetFolder is None else os.path.join(targetFolder, sourceFolder), error_if_none=False) is None:
                    _ensure_online(sourceFolder, f"{constants.ENDPOINT}/{repoId}/tree/main/{quote(sourceFolder)}")
                    snapshot_download(repo_id=repoId, allow_patterns=sourceFolder + "/*", local_dir=local_dir)
                continue
            # Explicit iteration keeps the opt-in context in this thread, including folder downloads.
            prefix = sourceFolder.rstrip("/") + "/" if sourceFolder else ""
            files = [name[len(prefix):] for name in HfApi().list_repo_files(repoId) if name.startswith(prefix)]
        for onefile in files:
            check_download_cancelled(gen)
            if fl.locate_file(_download_relpath(sourceFolder, onefile, targetFolder), error_if_none=False) is None:
                downloads.append(dict(repo_id=repoId, filename=onefile, local_dir=local_dir, subfolder=sourceFolder or None))
    for index, download in enumerate(downloads, 1):
        _hf_download(**download, gen=gen, show_filename=show_filename, file_index=index, file_count=len(downloads))


def _download_relpath(source_folder, filename, target_folder=None):
    source_folder = "" if source_folder is None else source_folder
    if target_folder is not None and len(target_folder) == 0:
        target_folder = None
    if target_folder is None:
        return os.path.join(source_folder, filename) if len(source_folder) > 0 else filename
    return os.path.join(target_folder, source_folder, filename) if len(source_folder) > 0 else os.path.join(target_folder, filename)


def download_def_missing_files(download_def):
    from shared.utils import files_locator as fl

    if download_def is None:
        return []
    if isinstance(download_def, list):
        missing = []
        for one_def in download_def:
            missing.extend(download_def_missing_files(one_def))
        return missing
    source_folders = download_def.get("sourceFolderList", [])
    file_lists = download_def.get("fileList", [])
    target_folders = download_def.get("targetFolderList")
    if target_folders is None:
        target_folders = [None] * len(source_folders)
    missing = []
    for source_folder, files, target_folder in zip(source_folders, file_lists, target_folders):
        if len(files) == 0:
            rel_folder = _download_relpath(source_folder, "", target_folder).rstrip("\\/")
            if fl.locate_folder(rel_folder, error_if_none=False) is None:
                missing.append(rel_folder)
            continue
        for filename in files:
            rel_path = _download_relpath(source_folder, filename, target_folder)
            if fl.locate_file(rel_path, error_if_none=False) is None:
                missing.append(rel_path)
    return missing


def send_download_status(send_cmd=None, status_text=None):
    if send_cmd is not None and status_text:
        send_cmd("status", status_text)


def process_files_def_if_needed(download_def, send_cmd=None, status_text=None, gen=None, show_filename=True, process_files=None):
    gen = resolve_download_gen(gen)
    check_download_cancelled(gen)
    if download_def is None or len(download_def_missing_files(download_def)) == 0:
        return False
    send_download_status(send_cmd, status_text)
    if process_files is None:
        from functools import partial
        process_files = partial(process_files_def, gen=gen, show_filename=show_filename)
    if isinstance(download_def, list):
        for one_def in download_def:
            process_files(**one_def)
    else:
        process_files(**download_def)
    return True


def query_audio_background_replacement_download_def():
    from preprocessing.roformer.assets import query_download_def

    return query_download_def()


def download_audio_background_replacement(send_cmd=None, status_text="Downloading audio background replacement model files...", gen=None, process_files=None):
    return process_files_def_if_needed(query_audio_background_replacement_download_def(), send_cmd=send_cmd, status_text=status_text, gen=gen, process_files=process_files)


def process_download_defs(download_defs, gen=None, show_filename=True):
    if isinstance(download_defs, dict):
        process_files_def(**download_defs, gen=gen, show_filename=show_filename)
        return
    for download_def in download_defs or []:
        if download_def is not None:
            process_files_def(**download_def, gen=gen, show_filename=show_filename)


_download_compat_warnings = set()


def download_url(url, filename, gen=None, show_filename=True):
    """Call the active downloader, including older plugin replacements."""
    gen = resolve_download_gen(gen)
    downloader = download_file
    parameters = inspect.signature(downloader, follow_wrapped=False).parameters
    kwargs = {"gen": gen, "show_filename": show_filename}
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in parameters and parameters[key].kind != inspect.Parameter.POSITIONAL_ONLY}
    if len(kwargs) < 2:
        implementation = downloader if inspect.isroutine(downloader) else type(downloader).__call__
        source = inspect.getsourcefile(implementation) or inspect.getfile(implementation)
        parts = source.replace("\\", "/").split("/")
        plugin = next((parts[i + 1] for i, part in enumerate(parts[:-1]) if part.lower() == "plugins"), None)
        owner = f"Plugin '{plugin}' ({source})" if plugin else f"Download Replacement '{source}'"
        if owner not in _download_compat_warnings:
            _download_compat_warnings.add(owner)
            missing = ", ".join(key for key in ("gen", "show_filename") if key not in kwargs)
            print(f"[Downloads] Warning: Unable to Enable the Full Download Progress Bar as {owner} Does Not Support {missing}. Continuing Without These Arguments; Please Update the Plugin. Download Cancellation May Also Be Unavailable During the Transfer.")
    check_download_cancelled(gen)
    result = downloader(url, filename, **kwargs)
    check_download_cancelled(gen)
    return result


def download_file(url, filename, gen=None, show_filename=True):
    from shared.utils import files_locator as fl

    gen = resolve_download_gen(gen)
    check_download_cancelled(gen)
    url = url.split("|")[0]
    _ensure_online(os.path.basename(filename), url)
    if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
        base_dir = os.path.dirname(filename)
        url = url[len("https://huggingface.co/"):]
        url_parts = url.split("/resolve/main/")
        repoId = url_parts[0]
        onefile = os.path.basename(url_parts[-1])
        sourceFolder = os.path.dirname(url_parts[-1])
        if len(sourceFolder) == 0:
            _hf_download(repo_id=repoId, filename=onefile, local_dir=fl.get_download_location() if len(base_dir) == 0 else base_dir, gen=gen, show_filename=show_filename)
        else:
            tgt = fl.get_download_location() if len(base_dir) == 0 else base_dir
            os.makedirs(tgt, exist_ok=True)
            with TemporaryDirectory(prefix="_temp", dir=tgt) as temp_dir_path:
                _hf_download(repo_id=repoId, filename=onefile, local_dir=temp_dir_path, subfolder=sourceFolder, gen=gen, show_filename=show_filename)
                check_download_cancelled(gen)
                shutil.move(os.path.join(temp_dir_path, sourceFolder, onefile), tgt)
    elif gen is not None:
        from huggingface_hub import file_download as hf

        install_hf_download_patch()
        with download_context(gen, filename, show_filename):
            # Write beside the destination so publication is an atomic rename.
            with NamedTemporaryFile(dir=os.path.dirname(filename) or ".", prefix="_download_", suffix=".incomplete", delete=False) as temp_file:
                temp_path = temp_file.name
                try:
                    hf.http_get(url, temp_file, displayed_filename=os.path.basename(filename))
                except BaseException:
                    temp_file.close()
                    os.remove(temp_path)
                    raise
            try:
                check_download_cancelled(gen)
                os.replace(temp_path, filename)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    else:
        from urllib.request import urlretrieve

        urlretrieve(url, filename, create_progress_hook(filename))
