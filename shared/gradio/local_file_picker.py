# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
import itertools
import json
from pathlib import Path

import gradio as gr
import gradio.component_meta as gradio_component_meta
from gradio.components.base import server

from shared.utils import files_locator as fl


VIDEO_FILE_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".ogv"}
IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
CHECKPOINT_FILE_EXTENSIONS = {".safetensors", ".safetensor", ".sft", ".ckpt", ".bin", ".binn", ".pt", ".pth", ".gguf", ".onnx"}
LAST_DIRECTORY_CONFIG_KEY = "local_file_picker_last_dir"
_server_config: dict | None = None
_tooltip_counter = itertools.count(1)


def configure_last_directory_store(server_config: dict | None) -> None:
    global _server_config
    _server_config = server_config


def _normalize_extensions(file_extensions: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    return {f".{str(ext).strip().lower().lstrip('.')}" for ext in (file_extensions or []) if str(ext).strip()}


def _available_roots() -> list[tuple[str, str]]:
    roots: list[tuple[str, str]] = []
    if os.name == "nt":
        try:
            import ctypes

            drive_mask = ctypes.windll.kernel32.GetLogicalDrives()
            roots = [(f"{chr(ord('A') + index)}:\\", f"{chr(ord('A') + index)}:\\") for index in range(26) if drive_mask & (1 << index)]
        except Exception:
            roots = []
    else:
        roots = [("/", "/")]
    current_root = str(Path.cwd().anchor or "").strip()
    if current_root and current_root.casefold() not in {value.casefold() for _label, value in roots}:
        roots.insert(0, (current_root, current_root))
    return roots or [(".", ".")]


def _is_under_root(path: str, root: str) -> bool:
    path = os.path.abspath(os.path.normpath(path))
    root = os.path.abspath(os.path.normpath(root))
    try:
        return os.path.commonpath([path, root]).casefold() == root.casefold()
    except ValueError:
        return False


def _is_probable_url(value: str) -> bool:
    return "://" in str(value) or str(value).startswith(("mailto:", "urn:"))


def _is_non_local_entry(value: str) -> bool:
    return _is_probable_url(value) or str(value).startswith("=")


_create_or_modify_pyi = gradio_component_meta.create_or_modify_pyi
gradio_component_meta.create_or_modify_pyi = lambda *args, **kwargs: None


class _FilteredFileExplorer(gr.FileExplorer):
    def __init__(
        self,
        glob: str = "**/*",
        *,
        value=None,
        file_count: str = "multiple",
        root_dir: str | Path = ".",
        ignore_glob: str | None = None,
        label: str | None = None,
        every=None,
        inputs=None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        height: int | str | None = None,
        max_height: int | str | None = 500,
        min_height: int | str | None = None,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | None = None,
        file_extensions: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.file_extensions = _normalize_extensions(file_extensions)
        super().__init__(glob=glob, value=value, file_count=file_count, root_dir=root_dir, ignore_glob=ignore_glob, label=label, every=every, inputs=inputs, show_label=show_label, container=container, scale=scale, min_width=min_width, height=height, max_height=max_height, min_height=min_height, interactive=interactive, visible=visible, elem_id=elem_id, elem_classes=elem_classes, render=render, key=key)

    @classmethod
    def get_component_class_id(cls) -> str:
        return gr.FileExplorer.get_component_class_id()

    def get_block_name(self) -> str:
        return "fileexplorer"

    def get_block_class(self) -> str:
        return "fileexplorer"

    def get_config(self):
        config = super().get_config()
        config.pop("file_extensions", None)
        return config

    def _strip_root(self, path: str) -> str:
        root_dir = os.path.abspath(os.path.normpath(self.root_dir))
        normalized_path = os.path.abspath(os.path.normpath(path))
        try:
            relative_path = os.path.relpath(normalized_path, root_dir)
        except ValueError:
            return normalized_path
        if relative_path == "." or relative_path.startswith(f"..{os.path.sep}") or relative_path == "..":
            return normalized_path
        return relative_path

    @server
    def ls(self, subdirectory=None):
        if subdirectory is None:
            subdirectory = []
        try:
            full_subdir_path = self._safe_join(subdirectory)
            subdir_items = sorted(os.listdir(full_subdir_path), key=str.casefold)
        except OSError:
            return []
        files, folders = [], []
        for item in subdir_items:
            full_path = os.path.join(full_subdir_path, item)
            if os.path.isfile(full_path):
                if not self.file_extensions or Path(full_path).suffix.lower() in self.file_extensions:
                    files.append({"name": item, "type": "file", "valid": True})
            else:
                folders.append({"name": item, "type": "folder", "valid": False})
        return folders + files


gradio_component_meta.create_or_modify_pyi = _create_or_modify_pyi


class LocalFilePickerTextbox:
    def __init__(
        self,
        *,
        label: str,
        value: str = "",
        file_extensions: set[str] | list[str] | tuple[str, ...] | None = None,
        multiselect: bool = False,
        popup_title: str = "Browse Local Files",
        browse_label: str = "\U0001F4C2",
        browse_tooltip: str | None = None,
        lines: int | None = None,
        placeholder: str | None = None,
        textbox_scale: int = 8,
        button_min_width: int = 34,
        browser_height: int = 320,
        default_dir: str | Path | None = None,
        default_dir_input=None,
        compress_root: str | Path | None = None,
        compress_root_input=None,
    ) -> None:
        self.label = label
        self.value = value
        self.file_extensions = _normalize_extensions(file_extensions)
        self.multiselect = bool(multiselect)
        self.popup_title = popup_title
        self.browse_label = browse_label
        self.browse_tooltip = browse_tooltip or ("Browse local files" if self.multiselect else "Browse local file")
        self.tooltip_class = f"wangp-local-file-picker-tooltip-{next(_tooltip_counter)}"
        self.lines = lines if lines is not None else (4 if self.multiselect else 1)
        self.placeholder = placeholder
        self.textbox_scale = textbox_scale
        self.button_min_width = button_min_width
        self.browser_height = browser_height
        self.default_dir_input = default_dir_input
        self.compress_root_input = compress_root_input
        self.roots = _available_roots()
        self.static_default_dir = self._resolve_existing_dir(default_dir)
        self.static_compress_root = self._resolve_existing_dir(compress_root)
        paths = self._parse_text(value, self.static_compress_root)
        self.default_dir = self._start_dir_for_paths(paths) or self.static_default_dir or self._remembered_dir() or self.roots[0][1]
        self.default_root = self._root_for_paths(paths) or self._root_for_directory(self.default_dir) or self.roots[0][1]
        self.textbox: gr.Textbox | None = None
        self.browse_button: gr.Button | None = None
        self.popup: gr.Column | None = None
        self.drive: gr.Dropdown | None = None
        self.current_dir: gr.Textbox | None = None
        self.up_button: gr.Button | None = None
        self.close_button: gr.Button | None = None
        self.exit_button: gr.Button | None = None
        self.browser: gr.FileExplorer | None = None

    def mount(self) -> gr.Textbox:
        gr.HTML(f"<style>{self.get_css()}{self._tooltip_css()}</style>", elem_classes=["wangp-local-file-picker-style"])
        with gr.Column(elem_classes=["wangp-local-file-picker-field", self.tooltip_class]):
            self.textbox = gr.Textbox(label=self.label, value=self.value, lines=self.lines, placeholder=self.placeholder, scale=self.textbox_scale)
            self.browse_button = gr.Button(self.browse_label, min_width=1, elem_classes=["wangp-local-file-picker-browse-btn"])
        self._mount_popup()
        self._wire_events()
        return self.textbox

    def _tooltip_css(self) -> str:
        return f"""
.{self.tooltip_class} .wangp-local-file-picker-browse-btn::after {{
    content: {json.dumps(self.browse_tooltip)};
}}
"""

    def _mount_popup(self) -> None:
        with gr.Column(visible=False, elem_classes=["wangp-local-file-picker-popup"]) as self.popup:
            with gr.Column(elem_classes=["wangp-model-info-card", "wangp-local-file-picker-card"]):
                with gr.Row(elem_classes=["wangp-model-info-titlebar", "wangp-local-file-picker-titlebar"]):
                    gr.HTML(f"<div class='wangp-model-info-heading'>{self.popup_title}</div>", elem_classes=["wangp-local-file-picker-heading"])
                    self.close_button = gr.Button("x", elem_classes=["wangp-model-info-close", "wangp-local-file-picker-close"], min_width=26, scale=0)
                with gr.Column(elem_classes=["wangp-model-info-content", "wangp-local-file-picker-content"]):
                    with gr.Row(elem_classes=["wangp-local-file-picker-location-row"]):
                        self.drive = gr.Dropdown(self.roots, value=self.default_root, show_label=False, container=False, scale=1, min_width=90, elem_classes=["wangp-local-file-picker-drive"])
                        self.current_dir = gr.Text(show_label=False, value=self._display_dir(self.default_root, self.default_dir), placeholder="Folder", interactive=True, container=False, scale=8, min_width=240, elem_classes=["wangp-local-file-picker-folder-input"])
                        self.up_button = gr.Button("Up", min_width=58, elem_classes=["wangp-local-file-picker-up-btn"])
                    self.browser = _FilteredFileExplorer(value=self._browser_value(self.default_dir, self._parse_text(self.value, self.static_compress_root)), file_extensions=self.file_extensions, glob="**/*.*", show_label=False, root_dir=self.default_dir, file_count="multiple" if self.multiselect else "single", height=self.browser_height, interactive=True)
                    self.exit_button = gr.Button("Exit", visible=self.multiselect, elem_classes=["wangp-local-file-picker-exit-btn"], min_width=72)

    def _wire_events(self) -> None:
        assert self.textbox is not None
        assert self.browse_button is not None
        assert self.popup is not None
        assert self.drive is not None
        assert self.current_dir is not None
        assert self.up_button is not None
        assert self.close_button is not None
        assert self.exit_button is not None
        assert self.browser is not None

        self.browse_button.click(fn=self._open_popup, inputs=self._event_inputs([self.textbox], default_dir=True, compress_root=True), outputs=[self.popup, self.drive, self.current_dir, self.browser], queue=False, show_progress="hidden")
        self.drive.change(fn=self._change_drive, inputs=self._event_inputs([self.drive, self.textbox], compress_root=True), outputs=[self.current_dir, self.browser], queue=False, show_progress="hidden")
        self.current_dir.blur(fn=self._change_folder, inputs=self._event_inputs([self.drive, self.current_dir, self.textbox], compress_root=True), outputs=[self.drive, self.current_dir, self.browser], queue=False, show_progress="hidden")
        self.current_dir.submit(fn=self._change_folder, inputs=self._event_inputs([self.drive, self.current_dir, self.textbox], compress_root=True), outputs=[self.drive, self.current_dir, self.browser], queue=False, show_progress="hidden")
        self.up_button.click(fn=self._go_up, inputs=self._event_inputs([self.drive, self.current_dir, self.textbox], compress_root=True), outputs=[self.drive, self.current_dir, self.browser], queue=False, show_progress="hidden")
        self.close_button.click(fn=lambda: gr.update(visible=False), outputs=[self.popup], queue=False, show_progress="hidden")
        self.exit_button.click(fn=lambda: gr.update(visible=False), outputs=[self.popup], queue=False, show_progress="hidden")
        self.browser.change(fn=self._select_files, inputs=self._event_inputs([self.browser, self.textbox, self.drive, self.current_dir], compress_root=True), outputs=[self.textbox, self.popup], queue=False, show_progress="hidden")

    def _event_inputs(self, inputs: list, *, default_dir: bool = False, compress_root: bool = False) -> list:
        result = list(inputs)
        if default_dir and self.default_dir_input is not None:
            result.append(self.default_dir_input)
        if compress_root and self.compress_root_input is not None:
            result.append(self.compress_root_input)
        return result

    def _open_popup(self, current_text: str, dynamic_default_dir=None, compress_root=None):
        compress_root = self._active_compress_root(compress_root)
        paths = self._parse_text(current_text, compress_root)
        current_dir = self._start_dir_for_paths(paths) or self._resolve_existing_dir(dynamic_default_dir) or self._remembered_dir() or self.default_dir
        selected_root = self._root_for_directory(current_dir) or self.default_root
        self._remember_dir(current_dir)
        return gr.update(visible=True), gr.update(value=selected_root), gr.update(value=self._display_dir(selected_root, current_dir)), self._browser_update(current_dir, paths)

    def _change_drive(self, selected_root: str, current_text: str, compress_root=None):
        compress_root = self._active_compress_root(compress_root)
        current_dir = str(selected_root or self.default_root)
        self._remember_dir(current_dir)
        return gr.update(value=""), self._browser_update(current_dir, self._parse_text(current_text, compress_root))

    def _change_folder(self, selected_root: str, displayed_dir: str, current_text: str, compress_root=None):
        compress_root = self._active_compress_root(compress_root)
        current_dir = self._resolve_dir_for_root(selected_root, displayed_dir)
        if not os.path.isdir(current_dir):
            return gr.skip(), gr.skip(), gr.skip()
        selected_root = self._root_for_directory(current_dir) or selected_root or self.default_root
        self._remember_dir(current_dir)
        return gr.update(value=selected_root), gr.update(value=self._display_dir(selected_root, current_dir)), self._browser_update(current_dir, self._parse_text(current_text, compress_root))

    def _go_up(self, selected_root: str, displayed_dir: str, current_text: str, compress_root=None):
        compress_root = self._active_compress_root(compress_root)
        current_dir = self._resolve_dir_for_root(selected_root, displayed_dir)
        current_root = self._root_for_directory(current_dir)
        if current_root and os.path.abspath(os.path.normpath(current_root)).casefold() == current_dir.casefold():
            parent_dir = current_root
        else:
            parent_dir = os.path.dirname(current_dir) or current_root or current_dir
        selected_root = self._root_for_directory(parent_dir) or self.default_root
        self._remember_dir(parent_dir)
        return gr.update(value=selected_root), gr.update(value=self._display_dir(selected_root, parent_dir)), self._browser_update(parent_dir, self._parse_text(current_text, compress_root))

    def _select_files(self, selected_paths, current_text: str, selected_root: str, displayed_dir: str, compress_root=None):
        compress_root = self._active_compress_root(compress_root)
        current_dir = self._resolve_dir_for_root(selected_root, displayed_dir)
        selected = self._coerce_selected_paths(selected_paths)
        if self.multiselect:
            current_paths = self._current_entries_outside_dir(current_text, current_dir, compress_root)
            current_paths.extend(selected)
            return self._format_paths(current_paths, compress_root), gr.update()
        selected_text = self._format_paths(selected, compress_root)
        if not selected_text:  # an emptied list (a folder reload answering after the pick) leaves the pop-up as it is
            return gr.skip(), gr.skip()
        current_text = self._format_paths(self._parse_text(current_text, compress_root), compress_root)
        return selected_text, gr.update(visible=selected_text == current_text and bool(selected_text))

    def _browser_update(self, current_dir: str, paths: list[str]):
        self._remember_dir(current_dir)
        return gr.update(root_dir=current_dir, value=self._browser_value(current_dir, paths))

    def _browser_value(self, root: str, paths: list[str]):
        root_paths = [path for path in paths if _is_under_root(path, root)]
        if self.multiselect:
            return root_paths
        return root_paths[0] if root_paths else []

    def _root_for_paths(self, paths: list[str]) -> str | None:
        for path in paths:
            if not self._path_is_allowed(path):
                continue
            for _label, root in self.roots:
                if _is_under_root(path, root):
                    return root
        return None

    def _root_for_directory(self, directory: str) -> str | None:
        for _label, root in self.roots:
            if _is_under_root(directory, root):
                return root
        return None

    def _resolve_dir_for_root(self, root: str | None, displayed_dir: str | None) -> str:
        root = str(root or self.default_root)
        displayed_dir = str(displayed_dir or "").strip().strip('"')
        if os.path.isabs(displayed_dir):
            return os.path.abspath(os.path.normpath(displayed_dir))
        return os.path.abspath(os.path.normpath(os.path.join(root, displayed_dir))) if displayed_dir else os.path.abspath(os.path.normpath(root))

    def _display_dir(self, root: str, directory: str) -> str:
        root = os.path.abspath(os.path.normpath(root or self.default_root))
        directory = os.path.abspath(os.path.normpath(directory or root))
        try:
            relative = os.path.relpath(directory, root)
        except ValueError:
            return directory
        return "" if relative == "." else relative

    def _start_dir_for_paths(self, paths: list[str]) -> str | None:
        for path in paths:
            if self._path_is_allowed(path):
                return os.path.dirname(path)
        return None

    def _remembered_dir(self) -> str | None:
        value = str((_server_config or {}).get(LAST_DIRECTORY_CONFIG_KEY, "") or "").strip()
        return os.path.abspath(os.path.normpath(value)) if value and os.path.isdir(value) else None

    def _remember_dir(self, directory: str) -> None:
        if _server_config is not None and directory and os.path.isdir(directory):
            _server_config[LAST_DIRECTORY_CONFIG_KEY] = os.path.abspath(os.path.normpath(directory))

    def _parse_text(self, value: str | None, compress_root: str | None = None) -> list[str]:
        compress_root = self._active_compress_root(compress_root)
        chunks = str(value or "").splitlines() if self.multiselect else [str(value or "")]
        paths = []
        seen = set()
        for chunk in chunks:
            path = chunk.strip().strip('"')
            if not path:
                continue
            if _is_non_local_entry(path):
                continue
            if compress_root and not os.path.isabs(path):
                path = os.path.join(compress_root, path)
            else:
                path = fl.uncompress_path(path)
            path = os.path.abspath(os.path.normpath(path))
            key = path.casefold()
            if key not in seen:
                paths.append(path)
                seen.add(key)
        return paths

    def _coerce_selected_paths(self, selected_paths) -> list[str]:
        if selected_paths is None:
            return []
        paths = selected_paths if isinstance(selected_paths, list) else [selected_paths]
        return [path for path in self._parse_text("\n".join(str(path) for path in paths)) if self._path_is_allowed(path)]

    def _format_paths(self, paths: list[str], compress_root: str | None = None) -> str:
        compress_root = self._active_compress_root(compress_root)
        if self.multiselect:
            return "\n".join(self._format_path(path, compress_root) for path in paths)
        return self._format_path(paths[0], compress_root) if paths else ""

    def _format_path(self, path: str, compress_root: str | None = None) -> str:
        path = str(path)
        if _is_non_local_entry(path):
            return path
        if compress_root and os.path.isabs(path) and _is_under_root(path, compress_root):
            return os.path.relpath(path, compress_root).replace("\\", "/")
        return fl.compress_path(path)

    def _current_entries_outside_dir(self, current_text: str, current_dir: str, compress_root: str | None = None) -> list[str]:
        compress_root = self._active_compress_root(compress_root)
        entries = []
        seen = set()
        for chunk in str(current_text or "").splitlines():
            entry = chunk.strip().strip('"')
            if not entry:
                continue
            if _is_non_local_entry(entry):
                normalized = entry
            else:
                if compress_root and not os.path.isabs(entry):
                    path = os.path.join(compress_root, entry)
                else:
                    path = fl.uncompress_path(entry)
                path = os.path.abspath(os.path.normpath(path))
                if _is_under_root(path, current_dir):
                    continue
                normalized = path
            key = normalized.casefold()
            if key not in seen:
                seen.add(key)
                entries.append(normalized)
        return entries

    def _path_is_allowed(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False
        return not self.file_extensions or Path(path).suffix.lower() in self.file_extensions

    def _active_compress_root(self, value=None) -> str | None:
        return self._resolve_existing_dir(value) or self.static_compress_root

    def _resolve_existing_dir(self, value=None) -> str | None:
        value = str(value or "").strip()
        if not value:
            return None
        directory = os.path.abspath(os.path.normpath(os.path.expanduser(value)))
        return directory if os.path.isdir(directory) else None

    @staticmethod
    def get_css() -> str:
        return """
.wangp-local-file-picker-style {
    display: none !important;
}
.wangp-local-file-picker-field {
    position: relative !important;
}
.wangp-local-file-picker-field > .form {
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.wangp-local-file-picker-browse-btn {
    position: absolute !important;
    top: 0 !important;
    right: 4px !important;
    z-index: 3 !important;
    width: 24px !important;
    min-width: 24px !important;
    max-width: 24px !important;
    height: 24px !important;
    min-height: 24px !important;
    max-height: 24px !important;
    padding: 0 !important;
    flex: 0 0 24px !important;
    font-size: 14px !important;
    line-height: 1 !important;
    border: 0 !important;
    outline: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wangp-local-file-picker-browse-btn > .form {
    border: 0 !important;
    outline: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.wangp-local-file-picker-browse-btn button,
.wangp-local-file-picker-browse-btn > button,
.wangp-local-file-picker-browse-btn button:hover,
.wangp-local-file-picker-browse-btn > button:hover,
.wangp-local-file-picker-browse-btn button:focus,
.wangp-local-file-picker-browse-btn > button:focus {
    width: 24px !important;
    min-width: 24px !important;
    max-width: 24px !important;
    height: 24px !important;
    min-height: 24px !important;
    max-height: 24px !important;
    padding: 0 !important;
    border: 0 !important;
    outline: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    color: var(--body-text-color, #172b3a) !important;
    font-size: 14px !important;
    line-height: 1 !important;
}
.wangp-local-file-picker-browse-btn *,
.wangp-local-file-picker-browse-btn *::before,
.wangp-local-file-picker-browse-btn *::after {
    border-color: transparent !important;
    outline: 0 !important;
    box-shadow: none !important;
}
.wangp-local-file-picker-browse-btn::after {
    position: absolute;
    top: 28px;
    right: 0;
    width: max-content;
    max-width: 260px;
    padding: 5px 7px;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.18));
    border-radius: 5px;
    background: var(--background-fill-primary, #ffffff);
    color: var(--body-text-color, #172b3a);
    box-shadow: var(--shadow-drop, 0 4px 12px rgba(0, 0, 0, 0.14));
    font-size: 12px;
    line-height: 1.25;
    pointer-events: none;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.12s ease, visibility 0s linear 0.12s;
}
.wangp-local-file-picker-browse-btn:hover::after {
    opacity: 1;
    visibility: visible;
    transition-delay: 0.5s;
}
.wangp-local-file-picker-popup {
    position: fixed !important;
    top: 96px !important;
    right: 32px !important;
    width: min(780px, calc(100vw - 34px)) !important;
    max-height: min(80vh, 760px) !important;
    z-index: 1210 !important;
    pointer-events: none !important;
}
.wangp-local-file-picker-popup > .form {
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wangp-local-file-picker-card {
    pointer-events: auto !important;
    position: relative !important;
    overflow: hidden !important;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16)) !important;
    border-radius: 18px !important;
    background: var(--background-fill-primary, rgba(255, 255, 255, 0.99)) !important;
    box-shadow: 0 28px 62px rgba(7, 31, 48, 0.24) !important;
}
.wangp-local-file-picker-titlebar {
    margin: 0 !important;
}
.wangp-local-file-picker-titlebar > .form {
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}
.wangp-local-file-picker-heading,
.wangp-local-file-picker-heading > .html-container {
    min-height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}
.wangp-local-file-picker-titlebar .wangp-local-file-picker-close {
    flex: 0 0 26px !important;
}
.wangp-local-file-picker-titlebar .wangp-local-file-picker-close button,
.wangp-local-file-picker-close {
    width: 26px !important;
    height: 26px !important;
    min-width: 26px !important;
    min-height: 26px !important;
    padding: 0 !important;
}
.wangp-local-file-picker-content {
    max-height: calc(min(80vh, 760px) - 46px) !important;
    overflow: hidden !important;
    padding-bottom: 14px !important;
}
.wangp-local-file-picker-content .file-wrap {
    max-height: none !important;
}
.wangp-local-file-picker-location-row {
    align-items: flex-end !important;
    gap: 8px !important;
}
.wangp-local-file-picker-location-row .form {
    border: 0 !important;
    background: transparent !important;
}
.wangp-local-file-picker-drive {
    min-width: 90px !important;
}
.wangp-local-file-picker-folder-input {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 6px !important;
    background: var(--input-background-fill, var(--body-background-fill)) !important;
    display: block !important;
    min-width: 240px !important;
}
.wangp-local-file-picker-folder-input textarea,
.wangp-local-file-picker-folder-input input {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 6px !important;
    background: var(--input-background-fill, var(--body-background-fill)) !important;
    display: block !important;
    width: 100% !important;
    min-height: 32px !important;
    height: 32px !important;
    padding: 4px 8px !important;
}
.wangp-local-file-picker-content .file-wrap,
.wangp-local-file-picker-content .file-explorer,
.wangp-local-file-picker-content [data-testid="file-explorer"] {
    overflow-y: auto !important;
}
.wangp-local-file-picker-exit-btn {
    align-self: flex-end !important;
    margin-top: 8px !important;
    width: 72px !important;
    min-width: 72px !important;
}
"""
