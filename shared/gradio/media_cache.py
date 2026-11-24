# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Inspect and expire Gradio's content-addressed media copies and WanGP video previews."""
import errno
import re
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import anyio
from tqdm import tqdm

from shared.utils.workspace_storage import OUTPUT_KEYS


CONFIG_KEY = "gradio_cache_cleanup"
MANUAL, STARTUP = 0, -1
CHOICES = [("Manual only", MANUAL), ("Clear on startup", STARTUP), ("After 12 hours", 12),
           ("After 1 day", 24), ("After 7 days", 168), ("After 30 days", 720)]
POLICIES = tuple(value for _, value in CHOICES)
_HASH = re.compile(r"[0-9a-f]{64}")
_PREVIEW = re.compile(r".*_[0-9a-f]{16}\.mp4(?:\.source\.json)?", re.DOTALL)
_STARTED = time.time()
_lock = threading.Lock()


def _guard_root(root, blocks, config):
    if root == root.parent:
        raise ValueError("Choose a dedicated Gradio cache folder inside the drive.")
    for base in (Path.home(), Path.cwd(), Path(tempfile.gettempdir())):
        if base.resolve().is_relative_to(root):
            raise ValueError(f"Choose a dedicated Gradio cache folder. {root} contains or is {base}.")
    # allowed_paths includes the live workspace folders and CLI output overrides;
    # the configuration also covers newly saved output paths awaiting restart.
    protected = [*blocks.allowed_paths, *(config.get(key) for key in (*OUTPUT_KEYS, "workspaces_dir"))]
    for path in filter(None, protected):
        directory = Path(path).resolve()
        if directory.is_relative_to(root) or root.is_relative_to(directory):
            raise ValueError(f"The Gradio cache overlaps a protected folder ({directory}). Set GRADIO_TEMP_DIR to a separate folder.")


def _is_link(info):
    return stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT


@dataclass
class Usage:
    files: int = 0
    size: int = 0
    removed: int = 0
    freed: int = 0
    skipped: int = 0
    busy: bool = False

    def describe(self, deleting):
        if self.busy:
            message = "Cleanup paused while generation tasks are queued or running."
        else:
            remaining = tqdm.format_sizeof(self.size - self.freed, suffix="B", divisor=1024)
            count = self.files - self.removed
            message = f"{count:,} cached file{'s' if count != 1 else ''} ({remaining})."
        if deleting and (self.removed or not self.busy):
            freed = tqdm.format_sizeof(self.freed, suffix="B", divisor=1024)
            message = f"Removed {self.removed:,} file{'s' if self.removed != 1 else ''} ({freed}). {message}"
        if self.skipped:
            message += f" Skipped {self.skipped:,} inaccessible entries."
        return message


def _files(root, usage):
    for directory in root.iterdir():
        preview = directory.name == "wangp_video_previews"
        if not preview and not _HASH.fullmatch(directory.name):
            continue
        try:
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or _is_link(info):
                continue
            removed_before = usage.removed
            for path in directory.iterdir():
                if preview and not _PREVIEW.fullmatch(path.name):
                    continue
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                except OSError:
                    usage.skipped += 1
                    continue
                if stat.S_ISREG(info.st_mode) and not _is_link(info):
                    yield path, info
            if usage.removed > removed_before and not preview:
                try:
                    directory.rmdir()
                except OSError as exc:
                    if exc.errno != errno.ENOTEMPTY:
                        raise
        except FileNotFoundError:
            continue
        except OSError:
            usage.skipped += 1


def scan(blocks, config, state, *, cutoff=None):
    """A missing cutoff measures usage. Only direct files in known cache layouts are eligible."""
    cache_path = Path(blocks.GRADIO_CACHE)
    root = cache_path.resolve()
    _guard_root(root, blocks, config)
    usage = Usage()

    def busy():
        return bool(state["gen"]["queue"]) or state.service.generation_running

    with _lock:
        if cutoff is not None and busy():
            usage.busy = True
            return usage
        if not root.exists():
            return usage
        keep = set()
        if cutoff is not None:
            keep.update(Path(p).resolve() for p in blocks.upload_file_set.copy())
            for block in blocks.blocks.values():
                keep.update(Path(p).resolve() for p in block.keep_in_cache.copy())
        for path, info in _files(root, usage):
            usage.files += 1
            usage.size += info.st_size
            # copy2 preserves the source mtime, even for a freshly cached copy.
            if cutoff is None or max(info.st_ctime, info.st_mtime) >= cutoff or path in keep:
                continue
            if busy():
                usage.busy = True
                break
            # Gradio registers uploads using the configured root, which can be relative or a symlink.
            if str(cache_path / path.parent.name / path.name) in blocks.upload_file_set:
                continue
            try:
                path.unlink()
                usage.removed += 1
                usage.freed += info.st_size
            except FileNotFoundError:
                continue
            except OSError:
                usage.skipped += 1
    return usage


async def scheduled_cleanup(blocks, get_config, state):
    startup = True
    while True:
        config = get_config()
        policy = config.get(CONFIG_KEY, MANUAL)
        retry_startup = False
        if policy in POLICIES and (policy > 0 or (startup and policy == STARTUP)):
            cutoff = _STARTED if policy == STARTUP else time.time() - policy * 3600
            try:
                usage = await anyio.to_thread.run_sync(lambda: scan(blocks, config, state, cutoff=cutoff))
                retry_startup = policy == STARTUP and usage.busy
                if usage.removed or usage.skipped:
                    tqdm.write(f"Gradio cache: {usage.describe(True)}")
            except (OSError, ValueError) as exc:
                tqdm.write(f"Gradio cache cleanup: {exc}")
        startup = retry_startup
        await anyio.sleep(30 * 60)


def create_controls(config, state, *, locked):
    import gradio as gr
    from shared.gradio.progress import WangpProgress

    blocks = gr.context.Context.root_block
    with gr.Accordion("Gradio Media Cache", open=False):
        gr.Markdown("Gradio stores temporary copies of uploads and media previews on disk. Large videos can fill the cache quickly. Clearing old copies frees space while keeping your saved outputs.")
        policy = gr.Dropdown(choices=CHOICES, value=config.get(CONFIG_KEY, MANUAL), label="Automatic Cache Cleanup", interactive=not locked,
                             info="Age-based cleanup runs at startup and every 30 minutes. Cleanup waits for an empty generation queue and keeps uploads from this run.")
        usage = gr.Textbox(label="Cache Usage", value="Click Check Cache Size to see how many files are cached and how much disk space they use.",
                           info=f"Gradio cache folder: {blocks.GRADIO_CACHE}", interactive=False)
        with gr.Row(elem_classes=["wangp-cache-actions"]):
            inspect = gr.Button("Check Cache Size", size="sm", scale=0, min_width=0, elem_classes=["wangp-action-pill"])
            clear = gr.Button("Clear Cache Now", size="sm", scale=0, min_width=0, elem_classes=["wangp-action-pill"], interactive=not locked)
            gr.Markdown("Clear Cache Now removes eligible copies of any age. Other apps sharing this cache may need to reload media.", elem_classes=["wangp-cache-help"])
        progress_display = WangpProgress.component()

    def report(cutoff=None):
        try:
            return scan(blocks, config, state, cutoff=cutoff).describe(cutoff is not None)
        except (OSError, ValueError) as exc:
            return str(exc)

    def measure(progress=WangpProgress()):
        progress.status("Checking Cache Size")
        return report()

    def clear_now(progress=WangpProgress()):
        if locked:
            raise gr.Error("Configuration is locked by command-line arguments.")
        progress.status("Clearing Gradio Cache")
        return report(time.time())

    for button, action in ((inspect, measure), (clear, clear_now)):
        WangpProgress.bind(button.click, action, inputs=[], outputs=[usage], component=progress_display, api_name=False)
    return policy
