# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Galleries shown as a player above a strip of thumbnails (audio files, 3D assets).

StripGallery holds the state components, the thumbnail strip and the selection plumbing shared by every strip gallery; a subclass
only says how a file is played (the player component and its value) and how its thumbnail looks. Element ids derive from the
gallery source id (`<id>_gallery_html`, `<id>_gallery_state_paths`, ...), so each gallery keeps stable anchors for scripts and tests.
"""
import html
import json
import os
import re
import uuid
import zipfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from shared.utils.gallery_sources import APP, AUDIO, MESH, WORLD

SHOW_THUMBNAIL_DATES = False  # True: the 3D and World thumbnails also show their date and time (Audio thumbnails always do: no image)


def _get_selected_idx(infos, selected_idx):
    selected_idx = int(selected_idx) if selected_idx is not None else 0
    if selected_idx >= len(infos):
        selected_idx = len(infos) - 1
    elif selected_idx < 0:
        selected_idx = 0
    if len(infos) == 0:
        selected_idx = -1
    return selected_idx


def gradio_file_url(path):
    return "/gradio_api/file=" + quote(os.path.abspath(path), safe="")


class StripGallery:
    """A strip gallery bound to a gallery source.

    Args:
        paths: initial file paths
        selected_index: initially selected index
        max_thumbnails: maximum number of thumbnails (None: no limit)
        label: player label
        update_only: if True, only render the player and the strip (form refresh)
    """

    source = None
    _timestamp_cache = OrderedDict()
    _timestamp_cache_max_entries = 50
    _gradio_upload_mtime_patch_installed = False

    def __init__(self, paths=None, selected_index=-1, max_thumbnails=10, label=None, update_only=False):
        self.paths = paths or []
        self.selected_index = selected_index
        self.max_thumbnails = max_thumbnails
        self.label = label or self.source.label
        self._render(update_only)

    def element_id(self, suffix):
        return f"{self.source.id}_gallery_{suffix}"

    # -------------------------------
    # Subclass hooks
    # -------------------------------

    def _create_player(self, path):
        raise NotImplementedError

    def _player_value(self, path):
        return path

    def _create_details(self, path):
        """Components under the player that follow the selection (none by default)."""
        return []

    def _details_value(self, path):
        return []

    def _thumbnail_body(self, info):
        raise NotImplementedError

    def _extra_file_info(self, path):
        return {}

    # -------------------------------
    # Public API
    # -------------------------------

    @staticmethod
    def get_javascript():
        """JavaScript for Blocks(js=...): the strip plumbing of every strip gallery, then each gallery's own script."""
        ids = json.dumps(list(STRIP_GALLERY_CLASSES))  # the Apps tab too: App Development can be turned on without reloading the page
        return StripGallery._javascript.replace("__STRIP_GALLERY_IDS__", ids) + "".join(gallery._own_javascript for gallery in STRIP_GALLERY_CLASSES.values())

    _own_javascript = ""
    _javascript = r"""
// ===== StripGallery: global-safe setup (no IIFE) =====
window.__stripGalleryNS = window.__stripGalleryNS || {};
const SG = window.__stripGalleryNS;
SG.state = SG.state || {};  // per gallery id: {manual, prevScroll}
SG.moMap = SG.moMap || {};
SG.uploadPatchInit = SG.uploadPatchInit || false;

SG.galleryState = function (id) { return SG.state[id] = SG.state[id] || { manual: false, prevScroll: 0 }; };
SG.root = function (id) { return document.querySelector('#' + id + '_gallery_html'); };
SG.container = function (id) { const r = SG.root(id); return r ? r.querySelector('.thumbnails-container') : null; };

if (!SG.init) {
  // Track the scroll position of each thumbnail strip
  document.addEventListener('scroll', (e) => {
    const strip = e.target.closest && e.target.closest('.strip-gallery-container');
    if (strip && e.target.classList.contains('thumbnails-container')) SG.galleryState(strip.dataset.gallery).prevScroll = e.target.scrollLeft;
  }, true);
  // Delegate thumbnail clicks (manual selection)
  document.addEventListener('click', function (e) {
    const thumb = e.target.closest('.strip-thumbnail');
    if (!thumb) return;
    const id = thumb.dataset.gallery, c = SG.container(id);
    if (c) SG.galleryState(id).prevScroll = c.scrollLeft;  // snapshot BEFORE backend updates
    window.selectStripThumbnail(id, thumb.getAttribute('data-index'));
  });
  document.addEventListener('dragstart', function (e) {
    const thumb = e.target.closest('.audio-thumbnail[data-wangp-media-url]');
    if (!thumb) return;
    const source = new URL(thumb.dataset.wangpMediaUrl, location.href).href;
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('application/x-wangp-gallery-media', source);
    e.dataTransfer.setData('text/uri-list', source);
    e.dataTransfer.setData('text/plain', source);
  });
  SG.init = true;
}

// Patch upload requests once to forward browser lastModified timestamps.
if (!SG.uploadPatchInit) {
  SG._origFetch = SG._origFetch || window.fetch.bind(window);
  window.fetch = function (input, init) {
    let nextInit = init;
    try {
      const reqUrl = (typeof input === 'string') ? input : (input && input.url ? input.url : '');
      const reqMethod = ((init && init.method) || (input && input.method) || 'GET').toUpperCase();
      const reqBody = init && init.body;
      const isUploadRequest = reqMethod === 'POST' && typeof reqUrl === 'string' && (reqUrl.includes('/upload') || reqUrl.includes('/gradio_api/upload'));
      if (isUploadRequest && reqBody instanceof FormData && reqBody.has('files')) {
        const lastModifiedValues = reqBody.getAll('files').map(oneFile => (typeof File !== 'undefined' && oneFile instanceof File) ? Number(oneFile.lastModified || 0) : 0);
        if (lastModifiedValues.length > 0) {
          const headers = new Headers((init && init.headers) || {});
          headers.set('x-wangp-file-last-modified', JSON.stringify(lastModifiedValues));
          nextInit = Object.assign({}, init || {}, { headers });
        }
      }
    } catch (_) {}
    return SG._origFetch(input, nextInit);
  };
  SG.uploadPatchInit = true;
}

// Manual selection trigger (used by the click handler and callable from elsewhere)
window.selectStripThumbnail = function (id, index) {
  if (window.WanGPGallerySelection && document.querySelector('#wangp-gallery-view')) {
    window.WanGPGallerySelection.select(id, index);
    return;
  }
  const c = SG.container(id), state = SG.galleryState(id);
  if (c) state.prevScroll = c.scrollLeft;  // snapshot BEFORE Gradio re-renders
  state.manual = true;
  const hiddenTextbox = document.querySelector('#' + id + '_gallery_click_data textarea');
  const hiddenButton = document.querySelector('#' + id + '_gallery_click_trigger');
  if (hiddenTextbox && hiddenButton) {
    hiddenTextbox.value = String(index);
    hiddenTextbox.dispatchEvent(new Event('input', { bubbles: true }));
    hiddenButton.click();
  }
  // Brief window marking this as manual to suppress auto-scroll
  setTimeout(() => { state.manual = false; }, 500);
};

// Ensure the selected thumbnail is fully visible (programmatic changes only)
SG.ensureVisibleIfNeeded = function (id) {
  const c = SG.container(id), r = SG.root(id);
  const sel = r && r.querySelector('.strip-thumbnail.selected');
  if (!c || !sel) return;
  const left = sel.offsetLeft, right = left + sel.clientWidth;
  if (left >= c.scrollLeft && right <= c.scrollLeft + c.clientWidth) return;
  const target = left - (c.clientWidth / 2) + (sel.clientWidth / 2), start = c.scrollLeft, dist = target - start, duration = 300;
  let t0 = null;
  function step(ts) {
    if (t0 === null) t0 = ts;
    const p = Math.min((ts - t0) / duration, 1);
    c.scrollLeft = start + dist * (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
};

// Observe Gradio's DOM replacement of a gallery strip and restore its scroll
SG.installObserver = function (id) {
  const rootEl = SG.root(id);
  if (!rootEl) return false;
  if (SG.moMap[id]) { try { SG.moMap[id].disconnect(); } catch (_) {} }
  const mo = new MutationObserver(() => {
    const c = SG.container(id), state = SG.galleryState(id);
    if (!c) return;
    c.scrollLeft = state.prevScroll;  // prevents the jump to 0
    if (!state.manual) requestAnimationFrame(() => SG.ensureVisibleIfNeeded(id));
  });
  mo.observe(rootEl, { childList: true, subtree: true });
  SG.moMap[id] = mo;
  return true;
};

// Call back once an element exists (a gallery tab renders its content when first shown)
SG.whenElement = function (selector, callback, delay = 50) {
  const el = document.querySelector(selector);
  if (el) callback(el);
  else setTimeout(() => SG.whenElement(selector, callback, Math.min(delay * 2, 500)), delay);
};
for (const id of __STRIP_GALLERY_IDS__) SG.whenElement('#' + id + '_gallery_html', () => SG.installObserver(id));
// ===== end StripGallery JS =====
"""

    @classmethod
    def install_gradio_upload_mtime_patch(cls):
        """Monkey-patch Gradio multipart parser to keep browser file lastModified as temp-file mtime."""
        if cls._gradio_upload_mtime_patch_installed:
            return True
        try:
            from gradio.route_utils import GradioMultiPartParser, GradioUploadFile
        except Exception:
            return False

        original_parse = getattr(GradioMultiPartParser, "parse", None)
        if original_parse is None:
            return False
        if getattr(original_parse, "__wangp_upload_mtime_patch__", False):
            cls._gradio_upload_mtime_patch_installed = True
            return True

        header_name = "x-wangp-file-last-modified"

        async def patched_parse(parser_self, *args, **kwargs):
            form = await original_parse(parser_self, *args, **kwargs)
            try:
                header_payload = None
                if hasattr(parser_self, "headers") and parser_self.headers is not None:
                    header_payload = parser_self.headers.get(header_name)
                if not header_payload:
                    return form

                mtime_values = json.loads(header_payload)
                if not isinstance(mtime_values, list) or len(mtime_values) == 0:
                    return form

                uploaded_files = form.getlist("files")
                for idx, file_obj in enumerate(uploaded_files):
                    if idx >= len(mtime_values):
                        break
                    ts_ms = mtime_values[idx]
                    if isinstance(ts_ms, dict):
                        ts_ms = ts_ms.get("lastModified", ts_ms.get("last_modified", 0))
                    try:
                        ts_ms = float(ts_ms)
                    except Exception:
                        continue
                    if ts_ms <= 0:
                        continue
                    ts = ts_ms / 1000.0 if ts_ms > 100_000_000_000 else ts_ms

                    tmp_path = None
                    if isinstance(file_obj, GradioUploadFile):
                        tmp_path = getattr(getattr(file_obj, "file", None), "name", None)
                    elif hasattr(file_obj, "file"):
                        tmp_path = getattr(file_obj.file, "name", None)
                    if tmp_path and os.path.exists(tmp_path):
                        os.utime(tmp_path, (ts, ts))
            except Exception:
                pass
            return form

        patched_parse.__wangp_upload_mtime_patch__ = True
        GradioMultiPartParser.parse = patched_parse
        cls._gradio_upload_mtime_patch_installed = True
        return True

    def get_state(self):
        """State components for other Gradio events: (state_paths, state_selected, refresh_trigger)."""
        return self.state_paths, self.state_selected, self.refresh_trigger

    # -------------------------------
    # Internal plumbing
    # -------------------------------

    def _render(self, update_only):
        with gr.Column() as self.component:
            self.state_paths = gr.Textbox(value=json.dumps(self.paths), visible=False, elem_id=self.element_id("state_paths"))
            selected_index = self.selected_index
            self.state_selected = gr.Number(value=selected_index, visible=False, elem_id=self.element_id("state_selected"))
            # Trigger for refreshing the gallery (programmatic)
            self.refresh_trigger = gr.Textbox(value="", visible=False, elem_id=self.element_id("refresh_trigger"))

            infos = self._process_paths(self.paths)
            default_path = infos[selected_index]["path"] if infos and 0 <= selected_index < len(infos) else (infos[0]["path"] if infos else None)
            with gr.Column(elem_classes=["strip-player-wrapper", f"{self.source.id}-player-wrapper"]):
                self.player = self._create_player(default_path)
            self.details = self._create_details(default_path)

            # The thumbnails strip; its id is the stable anchor of the MutationObserver
            self.gallery_html = gr.HTML(value=self._create_gallery_html(infos, selected_index), elem_id=self.element_id("html"), elem_classes="strip-gallery-root")

            if update_only:
                return

            # Hidden textbox and button capturing thumbnail clicks
            self.click_data = gr.Textbox(value="", visible=False, elem_id=self.element_id("click_data"))
            self.click_trigger = gr.Button(visible=False, elem_id=self.element_id("click_trigger"))
            self.click_trigger.click(fn=self._select, inputs=[self.click_data, self.state_paths, self.state_selected], outputs=[self.player, *self.details, self.gallery_html, self.state_paths, self.state_selected, self.click_data], show_progress="hidden")
            # Programmatic updates
            self.refresh_trigger.change(fn=self._refresh_gallery, inputs=[self.refresh_trigger, self.state_paths, self.state_selected], outputs=[self.player, *self.details, self.gallery_html, self.state_selected], show_progress="hidden")

    def _select(self, click_value, paths_json, current_selected):
        """Handle a thumbnail selection (manual)."""
        if not click_value:
            return self._render_from_state(paths_json, current_selected)
        try:
            paths = json.loads(paths_json) if paths_json else []
            infos = self._process_paths(paths)
            if not infos:
                return *self._player_updates(None), self._create_gallery_html([], 0), paths_json, -1, ""
            new_index = int(click_value)
            if 0 <= new_index < len(infos):
                selected_path = infos[new_index]["path"]
                if not os.path.exists(selected_path):
                    selected_path = None
                return *self._player_updates(selected_path), self._create_gallery_html(infos, new_index), paths_json, new_index, ""
        except Exception:
            pass
        return self._render_from_state(paths_json, current_selected)

    def _refresh_gallery(self, refresh_id, paths_json, selected_idx):
        *players, gallery_html, _paths_json, selected_idx, _click_data = self._render_from_state(paths_json, selected_idx)
        return *players, gallery_html, selected_idx

    def _player_updates(self, path):
        """The player's value and the details under it."""
        return self._player_value(path), *self._details_value(path)

    @classmethod
    def _get_cached_creation_datetime(cls, path):
        try:
            stat = os.stat(path)
            cache_key = (path, int(stat.st_size), int(stat.st_mtime))
        except Exception:
            return None
        cached_dt = cls._timestamp_cache.get(cache_key)
        if cached_dt is not None:
            cls._timestamp_cache.move_to_end(cache_key)
            return cached_dt
        dt = cls._read_creation_datetime(path)
        if dt is None:
            try:
                if os.name == "nt":
                    ts = os.path.getctime(path)
                else:
                    stat = os.stat(path)
                    ts = stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
                dt = datetime.fromtimestamp(ts)
            except Exception:
                return None
        cls._timestamp_cache[cache_key] = dt
        while len(cls._timestamp_cache) > cls._timestamp_cache_max_entries:
            cls._timestamp_cache.popitem(last=False)
        return dt

    @classmethod
    def _read_creation_datetime(cls, path):
        return None

    def _get_file_info(self, path, not_found=False):
        info = {"basename": Path(path).name, "path": path}
        if not_found:
            info.update(date="Deleted", time="00:00:00", timestamp="", deleted=True)
            return info
        dt = self._get_cached_creation_datetime(path) or datetime.fromtimestamp(os.path.getmtime(path))
        info.update(date=dt.strftime("%Y-%m-%d"), time=dt.strftime("%H:%M:%S"), timestamp=dt.timestamp(), deleted=False)
        info.update(self._extra_file_info(path))
        return info

    def _create_thumbnail_html(self, info, index, is_selected):
        selected_class = "selected" if is_selected else ""
        drag_attrs = f' draggable="true" data-wangp-media-url="{html.escape(gradio_file_url(info["path"]))}"' if self.source == AUDIO and not info["deleted"] else ""
        return f"""
        <div class="strip-thumbnail {self.source.id}-thumbnail {selected_class}" data-gallery="{self.source.id}" data-index="{index}" data-path="{html.escape(info['path'])}" title="{html.escape(info['basename'])}"{drag_attrs}>
            {self._thumbnail_body(info)}
        </div>
        """

    def _create_gallery_html(self, infos, selected_index, offset=0):
        thumbnail_width = max(80, min(150, 100 - (len(infos) - 1) * 2)) if infos else 100
        thumbnails_html = "".join(self._create_thumbnail_html(info, i + offset, i == selected_index) for i, info in enumerate(infos))
        selected_basename = infos[selected_index]["basename"] if infos and 0 <= selected_index < len(infos) else ""
        return f"""
        <style>
            .strip-gallery-container {{
                --bg-primary: var(--studio-surface);
                --bg-secondary: var(--studio-input);
                --bg-selected-filename: var(--studio-soft);
                --bg-selected-thumbnail: var(--studio-button);
                --bg-tooltip: var(--body-text-color);
                --text-primary: var(--body-text-color);
                --text-secondary: var(--body-text-color-subdued);
                --text-tooltip: var(--background-fill-primary);
                --border-primary: var(--studio-border);
                --border-secondary: var(--studio-border);
                --accent-color: var(--studio-accent);
                --shadow-color: color-mix(in srgb, var(--studio-accent) 20%, transparent);
                --shadow-color-selected: color-mix(in srgb, var(--studio-accent) 25%, transparent);
                display: flex; flex-direction: column; overflow: hidden; margin-top: 0 !important; padding-top: 0 !important;
            }}
            .strip-player-wrapper {{ margin-bottom: 0 !important; padding-bottom: 0 !important; }}
            .audio-player-wrapper {{ min-height: 200px; max-height: 200px; overflow: hidden; }}
            .audio-player-wrapper .audio {{ height: 100% !important; margin-bottom: 0 !important; }}
            .strip-transcript {{
                display: flex; flex-direction: column; gap: 6px; padding: 8px 10px; border: 1px solid var(--studio-border); border-radius: 8px;
                background: var(--studio-surface); color: var(--body-text-color);
            }}
            .strip-transcript .strip-transcript-bar {{ display: flex; align-items: center; gap: 6px; }}
            .strip-transcript .strip-transcript-title {{ margin-right: auto; font-weight: 600; }}
            .strip-transcript .strip-transcript-bar .strip-transcript-button {{  /* outranks Gradio's .prose a */
                padding: 2px 10px; border-radius: 6px; border: 1px solid var(--studio-border); background: var(--studio-input); color: var(--body-text-color);
                font-size: 13px; line-height: 20px; text-decoration: none; cursor: pointer;
            }}
            .strip-transcript .strip-transcript-bar .strip-transcript-button:hover {{ border-color: var(--studio-accent); }}
            .strip-transcript .strip-transcript-lines {{ max-height: 220px; overflow-y: auto; font-size: 14px; line-height: 1.45; }}
            .strip-transcript .strip-transcript-line {{ display: flex; gap: 10px; padding: 2px 4px; border-radius: 4px; }}
            .strip-transcript .strip-transcript-line[data-start] {{ cursor: pointer; }}
            .strip-transcript .strip-transcript-line[data-start]:hover {{ background: var(--studio-soft); }}
            .strip-transcript .strip-transcript-time {{ flex-shrink: 0; min-width: 42px; color: var(--studio-accent); font-variant-numeric: tabular-nums; }}
            .strip-transcript .strip-transcript-line.selected {{ background: var(--studio-soft); }}
            .strip-transcript b[style] {{ color: var(--speaker); }}
            /* the text or the timeline: the page's view, kept across selections (:root data-transcript-view) */
            .strip-transcript .strip-transcript-timeline {{ display: none; flex-direction: column; gap: 4px; }}
            :root[data-transcript-view="timeline"] .strip-transcript.has-timeline .strip-transcript-lines {{ display: none; }}
            :root[data-transcript-view="timeline"] .strip-transcript.has-timeline .strip-transcript-timeline {{ display: flex; }}
            .strip-transcript .strip-transcript-bar .strip-transcript-icon {{
                display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 26px; padding: 0; border-radius: 6px;
                border: 1px solid var(--studio-border); background: var(--studio-input); color: var(--body-text-color); cursor: pointer;
            }}
            .strip-transcript .strip-transcript-bar .strip-transcript-icon:hover {{ border-color: var(--studio-accent); }}
            .strip-transcript .strip-transcript-switch {{ display: inline-flex; }}
            .strip-transcript .strip-transcript-bar .strip-transcript-switch .strip-transcript-icon:first-child {{ border-radius: 6px 0 0 6px; }}
            .strip-transcript .strip-transcript-bar .strip-transcript-switch .strip-transcript-icon:last-child {{ border-radius: 0 6px 6px 0; border-left: 0; }}
            :root:not([data-transcript-view="timeline"]) .strip-transcript .strip-transcript-bar .strip-transcript-switch [data-view="text"],
            :root[data-transcript-view="timeline"] .strip-transcript .strip-transcript-bar .strip-transcript-switch [data-view="timeline"] {{
                background: var(--studio-accent); border-color: var(--studio-accent); color: var(--button-primary-text-color, #fff);
            }}
            .strip-transcript .strip-transcript-copy .icon-done, .strip-transcript .strip-transcript-copy .icon-failed,
            .strip-transcript .strip-transcript-copy.copied .icon-copy, .strip-transcript .strip-transcript-copy.failed .icon-copy {{ display: none; }}
            .strip-transcript .strip-transcript-copy.copied .icon-done, .strip-transcript .strip-transcript-copy.failed .icon-failed {{ display: inline; }}
            .strip-transcript .strip-transcript-lane, .strip-transcript .strip-transcript-axis {{ display: flex; align-items: center; gap: 8px; }}
            .strip-transcript .strip-transcript-speaker {{
                flex: 0 0 84px; font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--speaker, var(--body-text-color));
            }}
            .strip-transcript .strip-transcript-track {{ position: relative; flex: 1; height: 20px; border-radius: 4px; background: var(--studio-input); }}
            .strip-transcript .strip-transcript-segment {{
                position: absolute; top: 3px; bottom: 3px; min-width: 3px; border-radius: 3px; background: var(--speaker, var(--studio-accent)); opacity: 0.7; cursor: pointer;
            }}
            .strip-transcript .strip-transcript-segment:hover, .strip-transcript .strip-transcript-segment.selected {{ opacity: 1; outline: 2px solid var(--body-text-color); z-index: 1; }}
            .strip-transcript .strip-transcript-axis > div {{
                flex: 1; display: flex; justify-content: space-between; font-size: 11px; color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums;
            }}
            .strip-transcript .strip-transcript-selection {{
                min-height: 44px; max-height: 120px; overflow-y: auto; margin-top: 4px; padding: 4px 6px; border-radius: 6px; background: var(--studio-input);
                font-size: 14px; line-height: 1.45; color: var(--body-text-color-subdued);
            }}
            .strip-transcript .strip-transcript-selection:empty::before {{ content: attr(data-empty); }}
            .strip-transcript .strip-transcript-selection .strip-transcript-line {{ color: var(--body-text-color); }}
            .strip-gallery-container .selected-filename {{
                padding: 4px 12px; background: var(--bg-selected-filename); border-radius: 4px; font-size: 14px; font-weight: 500;
                margin: 0 0 8px 0; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary);
            }}
            .strip-gallery-container .thumbnails-container {{
                display: flex; justify-content: center; gap: 8px; padding: 12px; overflow-x: auto; overflow-y: hidden; flex-direction: row;
                border: 1px solid var(--border-primary); border-radius: 8px; background: var(--bg-primary); min-height: 120px;
            }}
            .strip-thumbnail {{
                position: relative; min-width: {thumbnail_width}px; width: {thumbnail_width}px; flex-shrink: 0; padding: 12px;
                border: 2px solid var(--border-secondary); border-radius: 8px; cursor: pointer; background: var(--bg-secondary);
                transition: all 0.2s ease; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px;
            }}
            .strip-thumbnail:hover {{ border-color: var(--accent-color); box-shadow: 0 2px 8px var(--shadow-color); transform: translateY(-2px); }}
            .strip-thumbnail:hover::after {{
                content: attr(title); position: absolute; bottom: calc(100% + 5px); left: 50%; transform: translateX(-50%);
                background: var(--bg-tooltip); color: var(--text-tooltip); padding: 6px 10px; border-radius: 4px; font-size: 12px;
                white-space: nowrap; z-index: 1000; pointer-events: none; max-width: {thumbnail_width}px; overflow: hidden; text-overflow: ellipsis;
            }}
            .strip-thumbnail.selected {{ border-color: var(--accent-color); background: var(--bg-selected-thumbnail); box-shadow: 0 2px 12px var(--shadow-color-selected); }}
            .strip-thumbnail .thumbnail-date {{ font-size: 13px; font-weight: 600; color: var(--text-primary); white-space: nowrap; }}
            .strip-thumbnail .thumbnail-time {{ font-size: 12px; color: var(--text-secondary); white-space: nowrap; }}
            .strip-thumbnail .thumbnail-detail {{ font-size: 13px; font-weight: bold; color: var(--accent-color); margin-top: 4px; white-space: nowrap; }}
            .strip-thumbnail .thumbnail-image {{ width: 100%; aspect-ratio: 1; object-fit: contain; border-radius: 4px; }}
            .mesh-thumbnail {{ padding: 6px; }}
            .mesh-thumbnail .thumbnail-time {{ font-size: 11px; line-height: 1.2; text-align: center; white-space: nowrap; }}
            .mesh-thumbnail .thumbnail-detail {{ margin-top: 0; font-size: 10.5px; max-width: 100%; overflow: hidden; text-overflow: ellipsis; }}
            .app-thumbnail {{ padding: 6px; }}
            .app-thumbnail .thumbnail-detail {{ margin-top: 0; max-width: 100%; font-size: 10.5px; line-height: 1.25; text-align: center; white-space: normal; overflow-wrap: anywhere;
                display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; overflow: hidden; }}  /* the app's name on up to two lines */
            @keyframes wangp-mesh-ready {{ from {{ opacity: .99; }} to {{ opacity: 1; }} }}
            .mesh-viewer {{ display: block; position: relative; border-radius: 8px; background: var(--studio-input); }}
            wangp-mesh-viewer {{ animation: wangp-mesh-ready 1ms; }}
            .mesh-viewer-status {{ position: absolute; inset: auto 8px 8px; padding: 8px; border-radius: 6px; background: var(--background-fill-primary); color: var(--body-text-color); font-size: 14px; }}
            .mesh-viewer-status[hidden] {{ display: none; }}
            .mesh-viewer-status button {{ margin-left: 8px; text-decoration: underline; }}
            .mesh-viewer-fallback {{ width: 100%; height: 100%; object-fit: contain; }}
            .mesh-viewer:fullscreen {{ height: 100vh !important; width: 100vw; border-radius: 0; }}
            .mesh-viewer-actions {{ position: absolute; top: 8px; right: 8px; display: flex; gap: 6px; }}
            .mesh-viewer-actions > * {{
                display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 6px; cursor: pointer;
                border: 1px solid var(--studio-border); background: var(--studio-surface); color: var(--body-text-color); opacity: 0.85;
            }}
            .mesh-viewer-actions > *:hover {{ opacity: 1; border-color: var(--studio-accent); }}
            .mesh-viewer-actions > [hidden] {{ display: none; }}
            .mesh-viewer-actions button:disabled {{ cursor: progress; opacity: 0.5; }}
            .mesh-snapshot-popup {{
                position: absolute; top: 48px; right: 8px; z-index: 5; display: flex; flex-direction: column; gap: 8px; padding: 10px 12px;
                width: 300px; max-width: calc(100% - 16px); box-sizing: border-box; border-radius: 8px; border: 1px solid var(--studio-border);
                background: var(--studio-surface); color: var(--body-text-color); box-shadow: 0 4px 16px var(--shadow-color); font-size: 13px;
            }}
            .mesh-snapshot-popup .mesh-snapshot-title {{ font-weight: 600; }}
            .mesh-snapshot-popup select {{
                width: 100%; padding: 4px 6px; border-radius: 6px; border: 1px solid var(--studio-border); background: var(--studio-input); color: inherit; font-size: 13px;
            }}
            .mesh-snapshot-popup .mesh-snapshot-hint {{ font-size: 11px; opacity: 0.75; }}
            .mesh-snapshot-popup .mesh-snapshot-buttons {{ display: flex; gap: 6px; justify-content: flex-end; align-items: center; }}
            .mesh-snapshot-popup button {{
                padding: 4px 10px; border-radius: 6px; border: 1px solid var(--studio-border); background: var(--studio-surface); color: inherit; cursor: pointer;
                height: 28px; font-size: 13px; line-height: 18px;
            }}
            .mesh-snapshot-popup button:hover, .mesh-snapshot-popup button[data-snapshot="add"] {{ border-color: var(--studio-accent); }}
            .mesh-snapshot-popup button:disabled {{ cursor: progress; opacity: 0.5; }}
            .mesh-viewer-fallback {{ height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: 12px; box-sizing: border-box; }}
            .world-viewer {{ display: block; position: relative; border-radius: 8px; background: #000; overflow: hidden; }}
            .world-viewer-empty {{ background: var(--studio-input); }}
            .world-viewer .world-video {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; display: block; }}
            .world-viewer .world-video.world-buffer {{ visibility: hidden; }}  /* the next clip, loading */
            .world-viewer:fullscreen {{ height: 100vh !important; width: 100vw; border-radius: 0; }}
            /* controls drawn over the video: light on dark whatever the theme (selectors outrank Gradio's .prose * and button resets) */
            .world-viewer .world-viewer-actions {{
                position: absolute; top: 8px; left: 8px; right: 8px; display: flex; flex-direction: column; align-items: flex-end; gap: 6px;
                pointer-events: none;
            }}
            .world-viewer .world-actions-row {{ display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: center; gap: 6px; }}
            .world-viewer .world-actions-row > * {{
                display: inline-flex; align-items: center; justify-content: center; gap: 6px; min-width: 32px; height: 30px; padding: 0 8px;
                border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.28); background: rgba(0, 0, 0, 0.6); color: #fff; cursor: pointer;
                text-decoration: none; box-sizing: border-box; font-size: 13px; pointer-events: auto;
            }}
            .world-viewer .world-actions-row > [hidden] {{ display: none; }}  /* a menu the world's model does not offer */
            .world-viewer .world-actions-row > a:link, .world-viewer .world-actions-row > a:visited {{ color: #fff; }}
            .world-viewer .world-actions-row > *:hover {{ border-color: var(--studio-accent, var(--color-accent, #2f6fed)); background: rgba(0, 0, 0, 0.8); }}
            .world-viewer .world-actions-row > * svg, .world-viewer .world-actions-row > * svg *, .world-viewer button.world-explore svg,
            .world-viewer button.world-explore svg *, .world-viewer button.world-explore span {{ color: inherit; }}  /* the theme's .prose * colours the paths too */
            .world-viewer .world-viewer-actions select {{ border: 0; background: transparent; color: #fff; font-size: 13px; }}
            .world-viewer .world-viewer-actions select option {{ color: #111; background: #fff; }}
            .world-viewer button.world-explore {{
                position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%); display: inline-flex; align-items: center; gap: 8px;
                padding: 10px 24px; border-radius: 24px; border: 0; background: var(--studio-accent, var(--color-accent, #2f6fed)); color: #fff;
                font-size: 16px; font-weight: 700; cursor: pointer; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
            }}
            .world-viewer button.world-explore:hover {{ filter: brightness(1.12); }}
            .world-viewer button.world-explore svg {{ width: 22px; height: 22px; }}
            .world-hud, .world-event, .world-help {{ display: none; }}
            /* Gradio's icon button styles, stacked as the image gallery's frame actions (its wrapper class shows them: hidden here unless exploring) */
            .world-viewer:not(.exploring) .world-view-actions {{ display: none; }}
            .world-viewer.exploring .world-view-actions {{ position: absolute; top: 12px; right: 12px; z-index: 5; flex-direction: column; gap: 0; }}
            .world-viewer .world-view-actions button div, .world-viewer .world-view-actions button svg, .world-viewer .world-view-actions button svg * {{ color: inherit; }}
            .world-viewer .world-actions-row .world-memory {{ position: relative; cursor: pointer; }}
            .world-viewer .world-actions-row .world-unavailable {{ color: #fff; opacity: 0.8; font-size: 13px; }}
            .world-viewer .world-actions-row .world-memory-button {{ background: transparent; border: 0; margin: 0; padding: 0 2px; color: #fff; font-size: 13px; cursor: pointer; }}  /* no theme margin: aligned with its label */
            .world-viewer .world-actions-row .world-memory-button::after {{ content: " ▾"; opacity: 0.8; }}
            .world-viewer .world-memory-list {{
                position: absolute; top: calc(100% + 4px); right: -1px; z-index: 10; display: flex; flex-direction: column; padding: 4px 0;
                border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.28); background: rgba(16, 16, 16, 0.95); text-align: left;
            }}
            .world-viewer .world-memory-list[hidden] {{ display: none; }}
            .world-viewer .world-memory-list > div {{ padding: 4px 10px; white-space: nowrap; color: #fff; cursor: pointer; }}
            .world-viewer .world-memory-list > .world-memory-note {{ opacity: 0.7; font-size: 12px; cursor: default; }}
            .world-viewer .world-memory-list > div[data-value]:hover {{ background: var(--studio-accent, var(--color-accent, #2f6fed)); }}
            .world-viewer .world-memory-list > div[aria-selected="true"] {{ font-weight: 600; }}
            .world-viewer video.world-replay {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; background: #000; z-index: 4; }}
            .world-viewer.exploring:not(.world-ready) .world-event {{ display: none; }}  /* events once the world is ready */
            .world-viewer .world-stats {{ display: none; }}
            .world-viewer.exploring.world-stats-shown:not(.world-ui-hidden) .world-stats {{
                display: block; position: absolute; top: 12px; left: 12px; padding: 8px 12px; border-radius: 8px; background: rgba(0, 0, 0, 0.6);
                color: #fff; font: 12px/1.5 ui-monospace, Consolas, monospace; white-space: pre; pointer-events: none;
            }}
            .world-viewer.exploring.world-ui-hidden .world-hud, .world-viewer.exploring.world-ui-hidden .world-event,
            .world-viewer.exploring.world-ui-hidden .world-help, .world-viewer.exploring.world-ui-hidden .world-view-actions {{ display: none; }}
            .world-viewer.exploring .world-viewer-actions, .world-viewer.exploring button.world-explore {{ display: none; }}
            .world-viewer.exploring .world-hud {{
                display: flex; gap: 14px; position: absolute; top: 14px; left: 50%; transform: translateX(-50%); padding: 6px 14px; border-radius: 16px;
                background: rgba(0, 0, 0, 0.55); color: #fff; font-size: 15px; pointer-events: none;
            }}
            .world-viewer.exploring .world-hud:has(.world-status:empty):has(.world-keys:empty) {{ display: none; }}
            .world-viewer.exploring .world-hud span {{ color: #fff; }}
            .world-viewer.exploring .world-keys {{ font-weight: 700; letter-spacing: 4px; }}
            .world-viewer.exploring .world-event {{ display: block; position: absolute; left: 50%; bottom: 44px; transform: translateX(-50%); width: min(640px, 80%); }}
            .world-viewer.exploring .world-event input {{
                width: 100%; padding: 10px 14px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.35); background: rgba(0, 0, 0, 0.5);
                color: #fff; font-size: 15px; box-sizing: border-box; outline: none;
            }}
            .world-viewer.exploring .world-help {{ display: block; position: absolute; left: 0; right: 0; bottom: 14px; text-align: center; color: rgba(255, 255, 255, 0.7); font-size: 13px; pointer-events: none; }}
            .mesh-viewer-fallback img {{ min-height: 0; max-width: 100%; flex: 1; object-fit: contain; }}
            .mesh-viewer-fallback div {{ font-size: 0.85em; opacity: 0.75; text-align: center; }}
        </style>
        <div class="strip-gallery-container" data-gallery="{self.source.id}">
            <div class="selected-filename">{html.escape(selected_basename)}</div>
            <div class="thumbnails-container">
                {thumbnails_html}
            </div>
        </div>
        """

    def _process_paths(self, paths):
        """File infos in the same order as the paths."""
        infos = []
        for path in paths or []:
            try:
                infos.append(self._get_file_info(path, not os.path.exists(path)))
            except Exception:
                continue
        return infos[: self.max_thumbnails]

    def _render_from_state(self, paths_json, selected_idx):
        value = json.loads(paths_json) if paths_json else []
        projected = isinstance(value, dict)
        paths, offset = (value['paths'], value['offset']) if projected else (value, 0)
        infos = self._process_paths(paths)
        local_index = selected_idx - offset
        if not projected:
            local_index = _get_selected_idx(infos, local_index)
            selected_idx = local_index
        selected_path = infos[local_index]['path'] if 0 <= local_index < len(infos) else None
        if selected_path and not os.path.exists(selected_path):
            selected_path = None
        return *self._player_updates(selected_path), self._create_gallery_html(infos, local_index, offset), paths_json, selected_idx, ''


class AudioGallery(StripGallery):
    source = AUDIO

    def _create_player(self, path):
        return gr.Audio(value=path, label=self.label, type="filepath")

    @classmethod
    def _read_creation_datetime(cls, path):
        try:
            from shared.utils.audio_metadata import read_audio_metadata, resolve_audio_creation_datetime
            return resolve_audio_creation_datetime(path, wangp_metadata=read_audio_metadata(path))
        except Exception:
            return None

    def _extra_file_info(self, path):
        return {"duration": self._get_audio_duration(path)}

    _own_javascript = r"""
// ===== AudioGallery: the transcript under the player =====
if (!SG.transcriptInit) {
  // The view (text or timeline) is the page's: it stays when another recording is selected, and is remembered by the browser
  try { const view = localStorage.getItem('wangp-transcript-view'); if (view) document.documentElement.dataset.transcriptView = view; } catch (_) {}
  document.addEventListener('click', function (e) {
    const panel = e.target.closest && e.target.closest('.strip-transcript');
    if (!panel) return;
    const copy = e.target.closest('.strip-transcript-copy');
    if (copy) {  // the clipboard icon turns into a check mark (or a cross when the browser refuses)
      const done = (state) => { copy.classList.add(state); setTimeout(() => copy.classList.remove(state), 1200); };
      if (!navigator.clipboard) { done('failed'); return; }
      navigator.clipboard.writeText(panel.querySelector('.strip-transcript-source').value).then(() => done('copied'), () => done('failed'));
      return;
    }
    const view = e.target.closest('.strip-transcript-switch [data-view]');
    if (view) {
      document.documentElement.dataset.transcriptView = view.dataset.view;
      try { localStorage.setItem('wangp-transcript-view', view.dataset.view); } catch (_) {}
      return;
    }
    // A timed line or a timeline segment: shown below the timeline, and the player moves to its start; a click beside them unselects
    const item = e.target.closest('.strip-transcript-line[data-start], .strip-transcript-segment');
    if (!item && e.target.closest('a, button')) return;
    const selection = panel.querySelector('.strip-transcript-selection');
    panel.querySelectorAll('.selected').forEach((el) => el.classList.remove('selected'));
    if (!item) {
      if (selection) selection.replaceChildren();
      return;
    }
    const row = panel.querySelector('.strip-transcript-lines .strip-transcript-line[data-index="' + item.dataset.index + '"]');
    if (selection) selection.innerHTML = row.outerHTML;
    for (const el of [row, panel.querySelector('.strip-transcript-segment[data-index="' + item.dataset.index + '"]')]) if (el) el.classList.add('selected');
    SG.seekAudio(panel, Number(item.dataset.start) / Number(panel.dataset.duration));
  });
  SG.transcriptInit = true;
}
// The waveform player keeps its audio element out of the page: it seeks where its waveform is clicked
SG.seekAudio = function (panel, fraction) {
  const gallery = panel.closest('.column');  // the gallery's column: its player is beside the transcript (each tab has its galleries)
  const host = [...gallery.querySelectorAll('.audio-player-wrapper #waveform > div')].find((el) => el.shadowRoot);
  const wave = host && host.shadowRoot.querySelector('[part="wrapper"]');
  if (!wave) return;
  const r = wave.getBoundingClientRect();
  wave.dispatchEvent(new MouseEvent('click', { clientX: r.left + r.width * Math.min(Math.max(fraction, 0), 1), clientY: r.top + r.height / 2, bubbles: true }));
};
"""

    def _create_details(self, path):
        transcript = self._transcript_html(path)
        return [gr.HTML(value=transcript, visible=bool(transcript), elem_id=self.element_id("transcript"), elem_classes="strip-transcript-root")]

    def _details_value(self, path):
        transcript = self._transcript_html(path)
        return [gr.update(value=transcript, visible=bool(transcript))]

    _timed_line = re.compile(r"\[(\d+):(\d\d):(\d\d(?:\.\d+)?) - (\d+):(\d\d):(\d\d(?:\.\d+)?)\]\s*(.*)")  # a "[start - end] text" line of a transcript
    _speaker_hues = (210, 25, 140, 280, 350, 50, 180, 320)

    _text_icon = '<path d="M4 6h16M4 12h16M4 18h10"/>'
    _timeline_icon = '<rect x="3" y="4" width="9" height="4" rx="1"/><rect x="12" y="10" width="9" height="4" rx="1"/><rect x="6" y="16" width="10" height="4" rx="1"/>'
    _copy_icon = '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'
    _done_icon = '<path d="M20 6 9 17l-5-5"/>'
    _failed_icon = '<path d="M18 6 6 18M6 6l12 12"/>'

    @staticmethod
    def _icon(drawing, classes=""):
        return (f'<svg class="{classes}" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"'
                f' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{drawing}</svg>')

    @staticmethod
    def _clock(seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes // 60}:{minutes % 60:02}:{seconds:02}" if minutes >= 60 else f"{minutes}:{seconds:02}"

    @staticmethod
    def _speaker_names(path):
        """The names that the recording's generation gave its speakers (custom setting speaker_names), in the order of Speaker 1, 2..."""
        from shared.utils.audio_metadata import read_audio_metadata
        from shared.utils.speech_dialogue import parse_speaker_names
        try:
            metadata = read_audio_metadata(path) or {}
        except Exception:  # an imported file without readable WanGP metadata
            return []
        return parse_speaker_names((metadata.get("custom_settings") or {}).get("speaker_names") or "")

    @classmethod
    def _transcript_html(cls, path):
        """The text a model saved beside the audio (its .txt side file), shown under the player as text or as a timeline with one lane per
        speaker; a timed line or segment plays the recording from its start. "" without one."""
        from shared.utils.side_files import saved_side_files
        files = saved_side_files(path) if path is not None else {}
        if ".txt" not in files:
            return ""
        text = files[".txt"].read_text(encoding="utf-8").strip()
        duration = cls._audio_seconds(path)
        names = cls._speaker_names(path)
        labels = sorted({name for name in names if name}, key=len, reverse=True)  # the longest first: a name may begin with another one
        speaker_line = re.compile("(" + "".join(re.escape(name) + "|" for name in labels) + r"Speaker \d+):\s*(.*)")
        rows, lanes = [], {}
        for index, line in enumerate(filter(None, (line.strip() for line in text.splitlines()))):
            timed = cls._timed_line.fullmatch(line)
            body = timed.group(7) if timed else line
            speaker = speaker_line.fullmatch(body)
            number = (names.index(speaker.group(1)) + 1 if speaker.group(1) in names else int(speaker.group(1)[len("Speaker "):])) if speaker else 0
            color = f' style="--speaker: hsl({cls._speaker_hues[(number - 1) % len(cls._speaker_hues)]} 65% 50%)"' if speaker else ""
            body = f"<b{color}>{html.escape(speaker.group(1))}:</b> {html.escape(speaker.group(2))}" if speaker else html.escape(body)
            if timed is None:
                rows.append(f'<div class="strip-transcript-line"><span>{body}</span></div>')
                continue
            start, end = (int(timed.group(i)) * 3600 + int(timed.group(i + 1)) * 60 + float(timed.group(i + 2)) for i in (1, 4))
            rows.append(f'<div class="strip-transcript-line" data-index="{index}" data-start="{start:.2f}" title="Play from {cls._clock(start)}">'
                        f'<span class="strip-transcript-time">{cls._clock(start)}</span><span>{body}</span></div>')
            lane = lanes.setdefault(number, {"label": speaker.group(1) if speaker else "Speech", "color": color, "segments": []})
            left, right = (100 * min(max(seconds, 0), duration) / duration for seconds in (start, end)) if duration > 0 else (0, 0)
            lane["segments"].append(f'<div class="strip-transcript-segment" data-index="{index}" data-start="{start:.2f}" style="left: {left:.3f}%; width: {right - left:.3f}%"'
                                    f' title="{cls._clock(start)} - {cls._clock(end)}"></div>')
        timeline = ""
        if lanes and duration > 0:  # one lane per speaker ("Speech" when the speakers are not labelled)
            lanes_html = "".join(f'<div class="strip-transcript-lane"{lane["color"]}><span class="strip-transcript-speaker" title="{html.escape(lane["label"])}">{html.escape(lane["label"])}</span>'
                                 f'<div class="strip-transcript-track">{"".join(lane["segments"])}</div></div>' for number, lane in sorted(lanes.items()))
            ticks = "".join(f"<span>{cls._clock(duration * step / 4)}</span>" for step in range(5))
            timeline = (f'<div class="strip-transcript-timeline">{lanes_html}<div class="strip-transcript-axis"><span class="strip-transcript-speaker"></span><div>{ticks}</div></div>'
                        f'<div class="strip-transcript-selection" data-empty="Select a turn to read what was said."></div></div>')
        view_switch = ('<span class="strip-transcript-switch" role="group" aria-label="View">'
                       f'<button type="button" class="strip-transcript-icon" data-view="text" title="Text" aria-label="Text">{cls._icon(cls._text_icon)}</button>'
                       f'<button type="button" class="strip-transcript-icon" data-view="timeline" title="Timeline of the Speakers" aria-label="Timeline">{cls._icon(cls._timeline_icon)}</button>'
                       '</span>') if timeline else ""
        copy_button = (f'<button type="button" class="strip-transcript-icon strip-transcript-copy" title="Copy to Clipboard" aria-label="Copy to Clipboard">'
                       f'{cls._icon(cls._copy_icon, "icon-copy")}{cls._icon(cls._done_icon, "icon-done")}{cls._icon(cls._failed_icon, "icon-failed")}</button>')
        downloads = "".join(f'<a class="strip-transcript-button" href="{html.escape(gradio_file_url(files[suffix]))}" download="{html.escape(files[suffix].name)}" title="Download">{suffix}</a>'
                            for suffix in (".txt", ".srt") if suffix in files)
        return (f'<div class="strip-transcript{" has-timeline" if timeline else ""}" data-duration="{duration:.3f}"><div class="strip-transcript-bar"><span class="strip-transcript-title">Transcript</span>'
                f'{view_switch}{copy_button}{downloads}</div>'
                f'<div class="strip-transcript-lines">{"".join(rows)}</div>{timeline}<textarea class="strip-transcript-source" hidden>{html.escape(text)}</textarea></div>')

    def _thumbnail_body(self, info):
        duration = "0:00" if info["deleted"] else info["duration"]
        return f'<div class="thumbnail-date">{info["date"]}</div><div class="thumbnail-time">{info["time"]}</div><div class="thumbnail-detail">{duration}</div>'

    @staticmethod
    def _audio_seconds(audio_path):
        import contextlib
        import wave
        try:
            with contextlib.closing(wave.open(audio_path, "r")) as f:
                return f.getnframes() / float(f.getframerate())
        except Exception:
            pass
        try:
            from mutagen import File
            audio = File(audio_path)
            if audio and getattr(audio, "info", None):
                return audio.info.length
        except Exception:
            pass
        return os.path.getsize(audio_path) / 32000  # rough estimate from the file size

    def _get_audio_duration(self, audio_path):
        """Audio duration formatted as M:SS."""
        try:
            return self._format_duration(self._audio_seconds(audio_path))
        except Exception:
            return "0:00"

    def _format_duration(self, seconds):
        mins, secs = divmod(int(seconds + 0.5), 60)
        return f"{mins}:{secs:02d}"


class MeshGallery(StripGallery):
    """3D assets (GLB meshes, PLY Gaussian splats): an interactive viewer above a strip of preview thumbnails (`<stem>.png` beside each asset)."""

    source = MESH
    viewer_height = 420

    @staticmethod
    def get_viewer_javascript(vendor_base):
        return MeshGallery._own_javascript.replace("'/gradio_api/file=shared/gradio/vendor/'", f"new URL({json.dumps(vendor_base)}, document.baseURI).href")

    def _create_player(self, path):
        return gr.HTML(value=self._player_value(path), elem_id=self.element_id("viewer"))

    def _player_value(self, path):
        if path is None:
            return f'<div class="mesh-viewer mesh-viewer-empty" style="height:{self.viewer_height}px"></div>'
        return self.viewer_html(path, gradio_file_url(path))

    @classmethod
    def viewer_html(cls, path, url):
        """The viewer of the 3D asset `path` served at `url`, with its actions (also shown full screen for Deepy chat assets)."""
        url, name = html.escape(url), html.escape(Path(path).name)
        preview = cls.preview_path(path)
        preview_url = html.escape(gradio_file_url(preview)) if os.path.isfile(preview) else ""
        if Path(path).suffix.lower() == ".ply":
            viewer, kind = f'<wangp-splat-viewer src="{url}" style="display:block;width:100%;height:100%"></wangp-splat-viewer>', "Gaussian Splats (PLY)"
        else:
            viewer, kind = f'<model-viewer src="{url}" alt="{name}" camera-controls touch-action="pan-y" interaction-prompt="none" shadow-intensity="0.6" environment-image="neutral" exposure="1" tone-mapping="neutral" style="width:100%;height:100%;background:transparent"></model-viewer>', "GLB"
        snapshot_icon = '<path d="M4 8h3l2-3h6l2 3h3v11H4V8Z"/><circle cx="12" cy="13" r="3.5"/>'
        download_icon = '<path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19h14"/>'
        fullscreen_icon = '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>'
        play_icon = '<path d="M8 5v14l11-7z"/>'
        icon = lambda drawing: f'<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{drawing}</svg>'
        return f"""<wangp-mesh-viewer class="mesh-viewer" data-path="{html.escape(path)}" data-preview="{preview_url}" style="height:{cls.viewer_height}px">
{viewer}
<div class="mesh-viewer-actions" role="group" aria-label="3D Asset Actions">
<button type="button" data-mesh-action="play" title="Play the Animation" aria-label="Play the Animation" hidden>{icon(play_icon)}</button>
<button type="button" data-mesh-action="snapshot" title="Add a Snapshot of the Current View to the Image Gallery" aria-label="Add Snapshot to Image Gallery">{icon(snapshot_icon)}</button>
<a href="{url}" download="{name}" title="Download the 3D Asset ({kind})" aria-label="Download {name}">{icon(download_icon)}</a>
<button type="button" data-mesh-action="fullscreen" title="Full Screen (Esc to leave)" aria-label="Full Screen">{icon(fullscreen_icon)}</button>
</div>
</wangp-mesh-viewer>"""

    @staticmethod
    def preview_path(path):
        return os.path.splitext(path)[0] + ".png"

    def _extra_file_info(self, path):
        preview = self.preview_path(path)
        info = {"preview": preview if os.path.isfile(preview) else None}
        try:
            from shared.utils.mesh_outputs import asset_stats
            stats = asset_stats(path)
            count, unit = (stats["gaussians"], "splats") if stats["kind"] == "splats" else (stats["faces"], "faces")
            info["detail"] = f"{count / 1e6:.1f}M {unit}" if count >= 1e6 else f"{count / 1000:.0f}K {unit}" if count >= 1000 else f"{count} {unit}"
        except Exception:
            info["detail"] = f"{os.path.getsize(path) / 1048576:.1f} MB"
        return info

    def _thumbnail_body(self, info):
        if info["deleted"]:
            return f'<div class="thumbnail-date">{info["date"]}</div><div class="thumbnail-time">{info["time"]}</div>'
        image = f'<img class="thumbnail-image" src="{html.escape(gradio_file_url(info["preview"]))}" alt="" loading="lazy">' if info["preview"] else '<div class="thumbnail-detail">3D</div>'
        when = f'<div class="thumbnail-time">{info["date"]}<br>{info["time"]}</div>' if SHOW_THUMBNAIL_DATES else ""
        return f'{image}{when}<div class="thumbnail-detail">{info["detail"]}</div>'

    # The vendored viewers load the first time the viewer shows an asset (<model-viewer> for GLB, three.js + Spark for splats):
    # users without 3D outputs pay nothing. Both need WebGL; without it (e.g. browser hardware acceleration disabled) the preview is shown.
    _own_javascript = r"""
// ===== MeshGallery: lazy 3D viewers =====
window.__meshGalleryNS = window.__meshGalleryNS || {};
(function (MG) {
  if (MG.init) return;
  MG.init = true;
  const VENDOR = '/gradio_api/file=shared/gradio/vendor/';
  const SPARK = VENDOR + 'spark/';
  MG.visible = element => {
    const box = element.getBoundingClientRect();
    const modal = document.querySelector('dialog[open]:focus-within') || [...document.querySelectorAll('dialog[open]')].at(-1);
    const full = document.fullscreenElement;
    return element.isConnected && !document.hidden && document.hasFocus() && (!modal || modal.contains(element))
      && (!full || full.contains(element) || element.contains(full))
      && box.width > 0 && box.height > 0 && box.bottom > 0 && box.right > 0
      && box.top < window.innerHeight && box.left < window.innerWidth && getComputedStyle(element).visibility !== 'hidden';
  };
  MG.webgl = function () {
    if (MG.hasWebGL !== true) {
      const gl = document.createElement('canvas').getContext('webgl2');
      MG.hasWebGL = !!gl;
      gl?.getExtension('WEBGL_lose_context')?.loseContext();
    }
    return MG.hasWebGL;
  };
  MG.load = function () {
    if (customElements.get('model-viewer')) return Promise.resolve();
    // The render lifecycle API and its bundle must update together, including
    // browsers which have the former viewer cached from an earlier session.
    const retry = MG.moduleAttempt ? '&retry=' + MG.moduleAttempt : '';
    return MG.loading ||= import(VENDOR + 'model-viewer.min.js?v=wangp-demand-1' + retry).catch(error => {
      MG.loading = null; MG.moduleAttempt = (MG.moduleAttempt || 0) + 1; throw error;
    });
  };
  // Gaussian splats: one WebGL context while the asset is shown, released when the viewer shows another asset.
  class SplatViewer extends HTMLElement {
    async load() {
      const token = this.token = {};
      this.ready = this.inView = false;
      const download = this.download = new AbortController();
      const bytes = fetch(this.getAttribute('src'), {signal: download.signal}).then(response => {
        if (!response.ok) throw new Error('Unable to Load the Splat Asset');
        return response.arrayBuffer();
      });
      const [THREE, {SparkRenderer, SplatMesh}, {OrbitControls}, fileBytes] = await Promise.all([import(SPARK + 'three.module.js'), import(SPARK + 'spark.module.min.js?v=wangp-dispose-1'), import(SPARK + 'OrbitControls.js'), bytes]);
      if (this.token !== token) return;
      this.download = null;
      const renderer = new THREE.WebGLRenderer({antialias: false, alpha: true});
      renderer.domElement.addEventListener('webglcontextlost', () => {
        if (this.token === token) this.dispatchEvent(new Event('error'));
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.domElement.style.cssText = 'width:100%;height:100%;display:block';
      this.appendChild(renderer.domElement);
      const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000), controls = new OrbitControls(camera, renderer.domElement);
      // Redraw for interaction and Spark's asynchronous sorting, then leave static views idle.
      const requestRender = () => {
        if (this.renderFrame || this.token !== token || !this.ready || !this.inView || document.hidden) return;
        this.renderFrame = requestAnimationFrame(() => {
          this.renderFrame = 0;
          if (this.token !== token || !this.inView || document.hidden || !this.clientWidth || !this.clientHeight) return;
          renderer.render(scene, camera);
        });
      };
      const splat = new SplatMesh({fileBytes: new Uint8Array(fileBytes), fileName: 'asset.ply'});
      const spark = new SparkRenderer({renderer, onDirty: requestRender});
      scene.add(spark, splat);
      this.state = {renderer, scene, camera, controls, splat, spark, requestRender};
      controls.addEventListener('change', requestRender);
      const resize = () => {
        const width = this.clientWidth, height = this.clientHeight;
        if (!width || !height) return;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        requestRender();
      };
      this.state.resize = resize;
      await splat.initialized;
      if (this.token !== token) {splat.dispose(); return;}
      // Same 3/4 framing as the preview: the whole asset in view, seen from the front right and slightly above.
      const box = splat.getBoundingBox(true), center = box.getCenter(new THREE.Vector3()), radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
      camera.position.copy(center).addScaledVector(new THREE.Vector3(0.55, 0.35, 0.75).normalize(), radius / Math.sin(THREE.MathUtils.degToRad(20)));
      camera.near = radius / 100;
      camera.far = radius * 100;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
      this.ready = true;
      this.refresh();
    }
    refresh() {
      this.inView = MG.visible(this);
      if (!this.inView) {
        cancelAnimationFrame(this.renderFrame);
        this.renderFrame = 0;
      } else if (this.ready) this.state.resize();
    }
    disconnectedCallback() {
      this.token = null;
      this.download?.abort(); this.download = null;
      cancelAnimationFrame(this.renderFrame);
      this.renderFrame = 0;
      const state = this.state;
      if (!state) return;
      this.state = null;
      state.controls.removeEventListener('change', state.requestRender);
      state.controls.dispose();
      state.splat.dispose();
      state.spark.dispose();
      state.renderer.dispose();
      state.renderer.forceContextLoss();
      state.renderer.domElement.remove();
    }
    toBlob({mimeType = 'image/png', pixelRatio = window.devicePixelRatio} = {}) {
      const {renderer, scene, camera} = this.state;
      const previousRatio = renderer.getPixelRatio();
      renderer.setPixelRatio(pixelRatio);
      renderer.render(scene, camera);  // the canvas is read in the same task as the render
      const blob = new Promise(resolve => renderer.domElement.toBlob(resolve, mimeType));
      renderer.setPixelRatio(previousRatio);
      return blob;
    }
  }
  customElements.define('wangp-splat-viewer', SplatViewer);
  // Animated GLBs behave like a video: paused until Play, then Pause / Resume (the button only shows when the asset has an animation).
  const PLAY_ICON = '<path d="M8 5v14l11-7z"/>', PAUSE_ICON = '<path d="M7 5h4v14H7zM13 5h4v14h-4z"/>';
  const setPlayButton = (button, playing) => {
    button.querySelector('svg').innerHTML = playing ? PAUSE_ICON : PLAY_ICON;
    button.title = button.ariaLabel = playing ? 'Pause the Animation' : 'Play the Animation';
  };
  class MeshViewer extends HTMLElement {
    connectedCallback() {
      const mount = this.mount = {};
      // HTML components insert their children in the same task as the host.
      queueMicrotask(() => { if (this.isConnected && this.mount === mount) this.start(); });
    }
    disconnectedCallback() {
      this.mount = null;
      this.events?.abort();
      this.viewer?.pause?.();
      this.viewer?.setRenderActive?.(false);
    }
    async start() {
      if (!MG.visible(this)) {this.deferred = true; return;}
      this.deferred = false;
      this.events?.abort();
      this.events = new AbortController();
      const viewer = this.viewer = this.querySelector('model-viewer, wangp-splat-viewer');
      this.playing = false;
      const snapshot = this.querySelector('[data-mesh-action="snapshot"]');
      if (snapshot) snapshot.disabled = true;
      const status = this.status ||= this.appendChild(document.createElement('div'));
      status.className = 'mesh-viewer-status'; status.setAttribute('role', 'status');
      status.hidden = false; status.textContent = 'Loading 3D Asset…';
      const fail = message => {
        if (!this.isConnected || this.viewer !== viewer) return;
        const replacement = viewer.cloneNode(false);
        const preview = document.createElement('img'); preview.className = 'mesh-viewer-fallback'; preview.alt = '3D Asset Preview';
        if (this.dataset.preview) preview.src = this.dataset.preview;
        else preview.hidden = true;
        viewer.pause?.();
        viewer.replaceWith(preview);
        this.viewer = null;
        this.querySelector('[data-mesh-action="play"]').hidden = true;
        if (snapshot) snapshot.disabled = true;
        status.hidden = false; status.textContent = message;
        const retry = document.createElement('button'); retry.type = 'button'; retry.textContent = 'Retry 3D';
        retry.onclick = () => {preview.replaceWith(replacement); this.start();};
        status.append(retry);
      };
      const loaded = () => {
        if (!this.isConnected || this.viewer !== viewer) return;
        status.hidden = true;
        if (snapshot) snapshot.disabled = false;
        const play = this.querySelector('[data-mesh-action="play"]');
        play.hidden = !viewer.availableAnimations?.length;
        setPlayButton(play, false);
        this.refresh();
      };
      viewer.addEventListener('error', () => fail('Unable to Load This 3D Asset. You Can Still Download It.'), {once: true, signal: this.events.signal});
      viewer.addEventListener('load', loaded, {once: true, signal: this.events.signal});
      try {
        // A temporary context failure must not disable all later attempts.
        if (!MG.webgl()) {fail('Interactive 3D Is Unavailable in This Browser. You Can Still Download the Asset.'); return;}
        if (viewer.localName === 'model-viewer') {
          await MG.load();
          if (!this.isConnected || this.viewer !== viewer) return;
          this.refresh();
          if (viewer.loaded) loaded();
        } else {
          await viewer.load();
          loaded();
        }
      } catch (error) {
        if (!this.isConnected || this.viewer !== viewer) return;
        console.error('3D viewer', error);
        fail('Unable to Start the 3D Viewer. You Can Still Download the Asset.');
      }
    }
    refresh() {
      const visible = MG.visible(this);
      if (this.deferred && visible) {this.start(); return;}
      const viewer = this.viewer;
      if (!viewer) return;
      if (viewer.localName === 'wangp-splat-viewer') {
        if (!visible) {
          this.events?.abort();
          viewer.replaceWith(viewer.cloneNode(false));
          this.viewer = null; this.deferred = true;
        } else viewer.refresh();
      }
      else if (typeof viewer.pause === 'function') {
        viewer.setRenderActive(visible);
        if (this.playing && visible) { if (viewer.paused) viewer.play(); }
        else if (!viewer.paused) viewer.pause();
      }
    }
  }
  customElements.define('wangp-mesh-viewer', MeshViewer);
  let layoutFrame = 0;
  MG.refresh = () => {
    if (layoutFrame) return;
    layoutFrame = requestAnimationFrame(() => {
      layoutFrame = 0;
      document.querySelectorAll('wangp-mesh-viewer').forEach(viewer => viewer.refresh());
    });
  };
  window.addEventListener('resize', MG.refresh);
  document.addEventListener('scroll', MG.refresh, {capture: true, passive: true});
  document.addEventListener('fullscreenchange', MG.refresh);
  document.addEventListener('visibilitychange', () => document.querySelectorAll('wangp-mesh-viewer').forEach(viewer => viewer.refresh()));
  document.addEventListener('wangp-viewer-scope', () => document.querySelectorAll('wangp-mesh-viewer').forEach(viewer => viewer.refresh()));
  window.addEventListener('blur', () => queueMicrotask(() => document.querySelectorAll('wangp-mesh-viewer').forEach(viewer => viewer.refresh())));
  window.addEventListener('focus', MG.refresh);
  document.addEventListener('animationstart', event => {if (event.animationName === 'wangp-mesh-ready') MG.refresh();});
  document.addEventListener('click', MG.refresh);
  document.addEventListener('click', event => {
    const button = event.target.closest('.mesh-viewer [data-mesh-action="play"]');
    if (!button) return;
    const container = button.closest('wangp-mesh-viewer');
    container.playing = !container.playing;
    container.refresh();
    setPlayButton(button, container.playing);
  });
  // Full screen: the viewer of the selected asset (both viewers follow their container's size)
  document.addEventListener('click', event => {
    const button = event.target.closest('.mesh-viewer [data-mesh-action="fullscreen"]');
    if (!button) return;
    const container = button.closest('.mesh-viewer');
    if (document.fullscreenElement === container) document.exitFullscreen();
    else container.requestFullscreen();
  });
  // Snapshot: the current view of the 3D viewer becomes an image of the image gallery. A popup sets its size: the pixel budget of a
  // resolution (by default the one selected in the settings) with the shape of the viewer, the view being rendered again at that size.
  const SNAPSHOT_SIDES = [512, 768, 1024, 1440, 2048, 2880, 4096], SNAPSHOT_MAX_SIDE = 8192;
  const snapshotRatio = (viewer, budget) => {
    const width = viewer.clientWidth || 1, height = viewer.clientHeight || 1;
    return budget ? Math.min(Math.sqrt(budget / (width * height)), SNAPSHOT_MAX_SIDE / Math.max(width, height)) : window.devicePixelRatio;
  };
  // model-viewer renders at the page's pixel ratio, read at each frame: raised while it renders the snapshot (minimum scale 1, no throttling)
  MG.renderSnapshot = async function (viewer, pixelRatio) {
    if (viewer.localName === 'wangp-splat-viewer') return viewer.toBlob({mimeType: 'image/png', pixelRatio});
    const ModelViewer = customElements.get('model-viewer'), minimum = ModelViewer.minimumRenderScale;
    const own = Object.getOwnPropertyDescriptor(window, 'devicePixelRatio');
    Object.defineProperty(window, 'devicePixelRatio', {configurable: true, get: () => pixelRatio});
    ModelViewer.minimumRenderScale = 1;
    try {
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      return await viewer.toBlob({mimeType: 'image/png'});
    } finally {
      if (own) Object.defineProperty(window, 'devicePixelRatio', own);
      else delete window.devicePixelRatio;
      ModelViewer.minimumRenderScale = minimum;
    }
  };
  const addSnapshot = async (container, pixelRatio) => {
    const viewer = container.querySelector('model-viewer, wangp-splat-viewer');
    const request = document.querySelector('#wangp-gallery-add-snapshot textarea'), view = document.querySelector('#wangp-gallery-view textarea');
    if (!viewer || typeof viewer.toBlob !== 'function' || !request || !view?.value) return;
    const blob = await MG.renderSnapshot(viewer, pixelRatio);
    const image = await new Promise((resolve, reject) => {const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(blob);});
    const payload = {workspace: JSON.parse(view.value).workspace, path: container.dataset.path, image, id: Date.now()};
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(request, JSON.stringify(payload));
    request.dispatchEvent(new Event('input', {bubbles: true}));
  };
  MG.openSnapshotPopup = function (container) {
    const viewer = container.querySelector('model-viewer, wangp-splat-viewer');
    if (!viewer) return;
    const choices = [], settings = document.querySelector('#wangp-resolution input')?.value.match(/(\d+)\s*x\s*(\d+)/);
    if (settings) choices.push([`Settings: ${settings[1]}x${settings[2]}`, settings[1] * settings[2]]);
    choices.push(['Viewer', 0], ...SNAPSHOT_SIDES.map(side => [`as ${side}x${side}`, side * side]));
    const popup = document.createElement('div');
    popup.className = 'mesh-snapshot-popup';
    popup.innerHTML = '<div class="mesh-snapshot-title">Snapshot Size</div><select aria-label="Snapshot Size"></select>'
      + '<div class="mesh-snapshot-hint">As many pixels as the resolution in brackets, with the shape of the viewer.</div>'
      + '<div class="mesh-snapshot-buttons"><button type="button" data-snapshot="add">Add to Image Gallery</button><button type="button" data-snapshot="cancel">Cancel</button></div>';
    const select = popup.querySelector('select');
    for (const [label, budget] of choices) {
      const ratio = snapshotRatio(viewer, budget);
      select.add(new Option(`${Math.ceil(viewer.clientWidth * ratio)} x ${Math.ceil(viewer.clientHeight * ratio)} (${label})`, ratio));
    }
    popup.addEventListener('click', async event => {
      const action = event.target.closest('[data-snapshot]')?.dataset.snapshot;
      if (action === 'cancel') popup.remove();
      if (action !== 'add') return;
      popup.querySelectorAll('button, select').forEach(element => { element.disabled = true; });
      try { await addSnapshot(container, Number(select.value)); } finally { popup.remove(); }
    });
    popup.addEventListener('keydown', event => { if (event.key === 'Escape') popup.remove(); });
    container.appendChild(popup);
    select.focus();
  };
  document.addEventListener('click', event => {
    const button = event.target.closest('.mesh-viewer [data-mesh-action="snapshot"]');
    if (!button) return;
    const container = button.closest('.mesh-viewer'), popup = container.querySelector('.mesh-snapshot-popup');
    if (popup) popup.remove();
    else MG.openSnapshotPopup(container);
  });
})(window.__meshGalleryNS);
// ===== end MeshGallery JS =====
"""


class WorldGallery(StripGallery):
    """Worlds of interactive world models (.wgpworld): the world's video above a strip of thumbnails. Explore opens it full screen and
    continues it from its end, a move at a time, with the keyboard and typed events, until Esc; the next exploration starts again from the
    stored world. The keys, the help and the event box are those of the shared explorer controls the world's model uses (its handler's
    query_world_explorer: W A S D to move, the arrows to look, typed events, a player's actions...). The server side is
    shared.gradio.world_explorer."""

    source = WORLD
    viewer_height = 420
    _icons = {
        "explore": '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5 5-2Z"/>',
        "download": '<path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19h14"/>',
        "record": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 8h4M3 12h18M3 16h4M17 8h4M17 16h4"/>',  # a film strip
        "save-world": '<circle cx="10" cy="10" r="7"/><path d="M3 10h14M10 3c2 2 2.8 4.5 2.8 7S12 15 10 17M10 3c-2 2-2.8 4.5-2.8 7S8 15 10 17M19 15v6M16 18h6"/>',
        "replay": '<path d="M7 5v14l12-7z"/>',
        "save-view": '<path d="M12 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v7M3 17l6-6 4 4M18 14v8m-3-3 3 3 3-3"/><circle cx="15" cy="8" r="1.5"/>',
        "add-view": '<path d="M14 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v9M3 17l6-6 5 5M15 19h8m-4-4v8"/><circle cx="15" cy="8" r="1.5"/>',
    }

    @classmethod
    def _icon(cls, name):
        return f'<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{cls._icons[name]}</svg>'

    @classmethod
    def _view_button(cls, action, label, title, icon):
        """A button with the markup of Gradio's icon buttons (the image gallery's frame actions): the same look in every theme."""
        drawing = cls._icons[icon]
        return (f'<button type="button" aria-label="{label}" title="{title}" class="svelte-vzs2gq padded" data-world-action="{action}" '
                f'style="color: var(--block-label-text-color); --bg-color: var(--block-background-fill);"><div class="svelte-vzs2gq small">'
                f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" '
                f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{drawing}</svg></div></button>')

    @staticmethod
    def video_path(path):
        from shared.utils.world_outputs import video_path
        return video_path(path)

    @staticmethod
    def _poster(path):
        """Data URI of the world's full resolution last frame (its small thumbnail for worlds saved before posters were stored)."""
        import base64
        import zipfile
        with zipfile.ZipFile(path) as archive:
            name = "poster.jpg" if "poster.jpg" in archive.namelist() else "thumbnail.jpg"
            return "data:image/jpeg;base64," + base64.b64encode(archive.read(name)).decode("ascii")

    @staticmethod
    def read_manifest(path):
        from shared.utils.world_outputs import read_manifest
        return read_manifest(path)

    def _create_player(self, path):
        return gr.HTML(value=self._player_value(path), elem_id=self.element_id("viewer"))

    @staticmethod
    def _controls_layout(names):
        """What the explorer script needs of the shared explorer controls `names` (the world model's, CONTROLS of
        shared.utils.world_instructions), read from the viewer's data-controls: its keys (event.code -> key name), their labels (letters as
        on a QWERTY keyboard until the browser gives this keyboard's), the help and the hint."""
        from shared.utils.world_instructions import CONTROLS
        controls = [CONTROLS[name] for name in names]
        keys = {code: key for control in controls for code, key in control.get("keys", {}).items()}
        labels = {**{key: code[3:] for code, key in keys.items() if code.startswith("Key")}, **{key: label for control in controls for key, label in control.get("labels", {}).items()}}
        return {"keys": keys, "labels": labels, "help": [control["help"] for control in controls], "hint": [control["hint"] for control in controls if "hint" in control]}

    def _player_value(self, path):
        if path is None or not os.path.isfile(path):
            return f'<div class="world-viewer world-viewer-empty" style="height:{self.viewer_height}px"></div>'
        return self.viewer_html(path)

    @classmethod
    def viewer_html(cls, path):
        video = cls.video_path(path)
        video_url = html.escape(gradio_file_url(video)) if os.path.isfile(video) else ""
        name = html.escape(Path(path).name)
        manifest = cls.read_manifest(path)
        poster = f' poster="{cls._poster(path)}"'
        from shared.gradio.world_explorer import explorer_choices, explorer_menus, world_model_available
        if not world_model_available(manifest):  # its model was removed or disabled: the world's video plays, without the explorer
            return f"""<wangp-world-viewer class="world-viewer" data-world="{html.escape(path)}" data-video="{video_url}" style="height:{cls.viewer_height}px">
<video class="world-video" src="{video_url}"{poster} autoplay loop muted playsinline></video>
<div class="world-viewer-actions" role="group" aria-label="World Actions">
<div class="world-actions-row">
<span class="world-unavailable">Its model ({html.escape(manifest["model_type"])}) is not available: this world can be watched, not explored</span>
<a href="{html.escape(gradio_file_url(path))}" download="{name}" title="Download the World File" aria-label="Download {name}">{cls._icon("download")}</a>
</div>
</div>
</wangp-world-viewer>"""
        menus = explorer_menus(manifest)  # the world model's own controls and menus
        chosen = explorer_choices(manifest, menus)  # those of the last exploration
        move_frames = {value: frames for value, _, frames in menus["seconds"]}
        seconds = "".join(f'<option value="{value}" data-frames="{frames}"{" selected" if value == chosen["seconds"] else ""}>{label}</option>' for value, label, frames in menus["seconds"])
        steps = "".join(f'<option value="{value}"{" selected" if value == chosen["steps"] else ""}>{label}</option>' for value, label in menus["steps"])
        # the Memory menu: a list drawn inside the viewer (a native menu with these labels leaves the window); the hidden select keeps the
        # value, its options the data the labels are made from (WangpWorld.memoryLabels: moves of the chosen length)
        memory = "".join(f'<option value="{value}" data-seconds="{video_seconds}" data-needs="{needs}"{" selected" if value == chosen["memory"] else ""}></option>'
                         for value, video_seconds, needs in menus["memory"])
        def moves(frames):
            count = max(1, round(frames / move_frames[chosen["seconds"]]))
            return f"{count} move{'' if count == 1 else 's'}"
        memory_items = "".join(f'<div role="option" data-value="{value}" aria-selected="{str(value == chosen["memory"]).lower()}">{moves(value)} ({video_seconds} s) · {needs}</div>'
                               for value, video_seconds, needs in menus["memory"])
        event_box = '<form class="world-event"><input type="text" placeholder="Type something that happens, then Enter" aria-label="World Event"></form>' if "event" in menus["controls"] else ""
        return f"""<wangp-world-viewer class="world-viewer" data-world="{html.escape(path)}" data-video="{video_url}" data-controls="{html.escape(json.dumps(cls._controls_layout(menus["controls"])))}" style="height:{cls.viewer_height}px">
<video class="world-video" src="{video_url}"{poster} autoplay loop muted playsinline></video>
<div class="world-hud" aria-live="polite"><span class="world-status"></span><span class="world-keys"></span></div>
{event_box}
<div class="world-help"></div>
<div class="world-stats"></div>
<div class="icon-button-wrapper svelte-9lsba8 world-view-actions no-border" role="group" aria-label="View Actions">{cls._view_button("download-view", "Save Image", "Save Current View as PNG - Original Resolution, without the Controls", "save-view")}{cls._view_button("snapshot", "Add View to Image Gallery", "Add Current View to Image Gallery, without the Controls", "add-view")}{cls._view_button("record", "Record Video", "Record the World's Video and your Moves as a new Video of the Video Gallery", "record")}{cls._view_button("replay", "Replay", "Replay the World's Video and your Moves - Space: Pause, Esc: Stop", "replay")}{cls._view_button("save-world", "Save World", "Save the World as it is now, with your Moves, as a new World of the Worlds Gallery: Explore it later to resume from here", "save-world")}</div>
<div class="world-viewer-actions" role="group" aria-label="World Actions">
<div class="world-actions-row">
<label class="world-seconds" title="Length of each move">Move <select aria-label="Move Length">{seconds}</select></label>
<label class="world-steps" title="Quality of the moves: fewer denoising steps answer faster"{" hidden" if len(menus["steps"]) < 2 else ""}>Quality <select aria-label="Move Quality">{steps}</select></label>
<div class="world-memory" title="{html.escape(menus["memory_title"])}"{" hidden" if len(menus["memory"]) < 2 else ""}>Memory <select aria-label="Move Memory" hidden>{memory}</select><button type="button" class="world-memory-button" aria-haspopup="listbox">{moves(chosen["memory"])}</button><div class="world-memory-list" role="listbox" hidden><div class="world-memory-note">Memory needs at this world's {manifest["width"]}x{manifest["height"]}, memory profile {menus["memory_profile"]["profile"]}</div>{memory_items}</div></div>
<a href="{html.escape(gradio_file_url(path))}" download="{name}" title="Download the World File" aria-label="Download {name}">{cls._icon("download")}</a>
</div>
</div>
<button type="button" class="world-explore" data-world-action="explore" title="Explore this World Full Screen">{cls._icon("explore")}<span>Explore</span></button>
</wangp-world-viewer>"""

    def _extra_file_info(self, path):
        import base64
        import zipfile
        info = {"thumbnail": None, "detail": ""}
        try:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                info["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(archive.read("thumbnail.jpg")).decode("ascii")
            seconds = sum(segment["video_frames"] for segment in manifest["segments"]) / manifest["fps"]
            info["detail"] = f"{seconds:.0f} s"
        except Exception:
            info["detail"] = f"{os.path.getsize(path) / 1048576:.1f} MB"
        return info

    def _thumbnail_body(self, info):
        if info["deleted"]:
            return f'<div class="thumbnail-date">{info["date"]}</div><div class="thumbnail-time">{info["time"]}</div>'
        image = f'<img class="thumbnail-image" src="{info["thumbnail"]}" alt="" loading="lazy">' if info["thumbnail"] else '<div class="thumbnail-detail">World</div>'
        when = f'<div class="thumbnail-time">{info["date"]}<br>{info["time"]}</div>' if SHOW_THUMBNAIL_DATES else ""
        return f'{image}{when}<div class="thumbnail-detail">{info["detail"]}</div>'

    _own_javascript = r"""
// ===== WorldGallery: full screen exploration =====
window.WangpWorld = window.WangpWorld || (function () {
  // the keys, labels, help and hint of the explorer controls the world's model uses (the viewer's data-controls, from its handler): keys
  // by position (event.code), the same physical keys on every layout, letters labelled as this keyboard prints them (Z Q S D on AZERTY)
  const learned = {};  // key name -> label of this keyboard, kept from one exploration to the next
  let session = null;  // {viewer, path, layout, held, pressed, busy, ready, text, seconds, steps, memory, moves}
  let requests = 0;  // numbers every request: two identical moves in a row must still change the textbox value

  function send(payload) {
    const box = document.querySelector('#wangp_world_request textarea, #wangp_world_request input');
    if (!box) return;
    const proto = box.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(box, JSON.stringify({...payload, request: ++requests}));
    box.dispatchEvent(new Event('input', {bubbles: true}));
  }
  function status(text) { if (session) session.viewer.querySelector('.world-status').textContent = text; }
  function label(key) { return learned[key] || session.layout.labels[key]; }
  function fill(text) { return text.replace(/\{(\w+)\}/g, (_, key) => label(key)); }
  function hold() { return session.layout.hint.map(fill).join(', '); }
  function showHelp() {
    session.viewer.querySelector('.world-help').textContent = [...session.layout.help.map(fill), 'G: GPU Stats', 'H: Hide Controls', 'Esc: Leave'].join(' · ');
  }
  function showStats() {
    // live GPU / RAM (refreshed every second while shown), then the last move's memory and timing
    const live = session.live, move = session.moveStats, lines = [];
    if (live) {
      lines.push(live.vram_used_gb !== undefined ? `GPU      ${live.vram_used_gb} / ${live.vram_total_gb} GB VRAM used · ${live.gpu_load} % load` + (move ? ` · ${move.peak_gb} GB peak of the last move` : '')
                                                 : `GPU      ${move ? move.peak_gb + ' GB VRAM peak of the last move' : 'no statistics available'}`);
      lines.push(`RAM      ${live.ram_used_gb} / ${live.ram_total_gb} GB used · WanGP ${live.process_ram_gb} GB`);
    }
    if (!move) {
      lines.push(session.ready ? 'Move statistics appear after the first move' : 'Preparing the world...');
      session.viewer.querySelector('.world-stats').textContent = lines.join('\n');
      return;
    }
    const stats = move;
    lines.push(`Explored ${stats.explored.seconds} s: the world's own video ${stats.world.seconds} s + your ${stats.moves} move${stats.moves === 1 ? '' : 's'} ${stats.explored_moves.seconds} s`);
    lines.push(stats.memory_line ? stats.memory_line : stats.memory_full ? `Memory   full (${stats.memory.seconds} s, ${stats.memory.frames} frames, ${stats.memory.moves} moves): a sample of the ${stats.explored.seconds} s explored, older parts sparser`
                                 : `Memory   holds all of it: ${stats.explored.seconds} of ${stats.memory.seconds} s (${stats.explored.frames} of ${stats.memory.frames} frames), room for ${stats.room_moves} more moves`);
    lines.push(`Move     ${stats.moves} · ${stats.video_seconds} s of video in ${stats.move_seconds} s · ${stats.steps} steps`);
    session.viewer.querySelector('.world-stats').textContent = lines.join('\n');
  }
  function refreshStats() {
    // Refresh on opening the statistics and on exploration responses, without a background timer.
    if (!session || !session.viewer.classList.contains('world-stats-shown')) return;
    send({action: 'stats', path: session.path});
    showStats();
  }
  function captureView(viewer) {
    // the frame shown, at the video's resolution: the controls drawn over the video are not part of it
    const video = viewer.querySelector('.world-video:not(.world-buffer)');
    if (!video || !video.videoWidth) return null;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas;
  }
  function viewName(viewer) {
    const stem = viewer.dataset.world.split(/[\\/]/).pop().replace(/\.wgpworld$/i, '');
    return stem + '_view_' + (session ? session.moves : 0) + '.png';
  }
  function downloadView(viewer) {
    const canvas = captureView(viewer);
    if (!canvas) return;
    canvas.toBlob((blob) => {
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob); link.download = viewName(viewer);
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 10000);
    }, 'image/png');
  }
  function addViewToGallery(viewer) {
    const canvas = captureView(viewer);
    const request = document.querySelector('#wangp-gallery-add-snapshot textarea'), view = document.querySelector('#wangp-gallery-view textarea');
    if (!canvas || !request || !view || !view.value) return;
    const payload = {workspace: JSON.parse(view.value).workspace, path: viewer.dataset.world, gallery: 'world', image: canvas.toDataURL('image/png'), id: Date.now()};
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(request, JSON.stringify(payload));
    request.dispatchEvent(new Event('input', {bubbles: true}));
  }
  function learnLayout() {
    if (!navigator.keyboard || !navigator.keyboard.getLayoutMap) return;  // other browsers learn the labels from the keys pressed (onKey)
    const current = session;
    navigator.keyboard.getLayoutMap().then((map) => {
      const keys = current.layout.keys;
      for (const code of Object.keys(keys).filter((code) => code.startsWith('Key'))) { const printed = map.get(code); if (printed) learned[keys[code]] = printed.toUpperCase(); }
      if (session === current) showHelp();
    }).catch(() => {});
  }
  function showClip(url) {
    // the clips are shown one after the other, each in full: a clip that arrives while the previous one still waits for its turn (a model
    // faster than playback) queues behind it instead of replacing it
    const viewer = session.viewer;
    session.clips = (session.clips || Promise.resolve()).then(() => presentClip(viewer, url));
  }
  function presentClip(viewer, url) {
    // the clip loads in a hidden second video and replaces the shown one once loaded and once the shown clip has ended: no blank frame.
    // The hidden video never plays before its turn: the world's own video, which serves as the second one from the second move on,
    // autoplays, and would otherwise run the clip while hidden and show it from its middle or its end
    if (!session || session.viewer !== viewer) return;
    const shown = viewer.querySelector('.world-video:not(.world-buffer)');
    let next = viewer.querySelector('.world-video.world-buffer');
    if (!next) {
      next = document.createElement('video');
      next.className = 'world-video world-buffer'; next.muted = true; next.playsInline = true;
      shown.after(next);
    }
    next.autoplay = false; next.loop = false;
    const loaded = new Promise((resolve) => { next.addEventListener('loadeddata', resolve, {once: true}); next.addEventListener('error', resolve, {once: true}); });
    const ended = shown.ended || shown.paused ? Promise.resolve() : new Promise((resolve) => shown.addEventListener('ended', resolve, {once: true}));
    next.src = url;
    return Promise.all([loaded, ended]).then(() => {
      if (!session || session.viewer !== viewer || next.getAttribute('src') !== url || next.error) return;
      shown.pause(); shown.classList.add('world-buffer'); next.classList.remove('world-buffer');
      next.play().catch(() => {});
    });
  }
  function showKeys() {
    if (!session) return;
    session.viewer.querySelector('.world-keys').textContent = [...session.held].map(label).join(' ');
  }
  function maybeMove() {
    if (!session || !session.ready || session.busy || session.replay) return;
    const keys = [...new Set([...session.held, ...session.pressed])], text = session.text;
    if (!keys.length && !text) { status(session.moves ? '' : 'Hold ' + hold()); return; }
    session.busy = true; session.pressed.clear(); session.text = '';
    status(text ? 'Generating: ' + text : 'Generating…');
    send({action: 'move', path: session.path, keys: keys.join('+'), text: text, seconds: session.seconds, steps: session.steps, memory: session.memory});
  }
  function onKey(event) {
    if (!session) return;
    if (session.replay) {  // Space pauses or resumes the replay, Esc stops it
      if (event.code === 'Space' || event.code === 'Escape') {
        event.preventDefault(); event.stopPropagation();
        if (event.type === 'keydown' && !event.repeat) {
          if (event.code === 'Escape') stopReplay();
          else if (session.replay.paused) session.replay.play().catch(() => {});
          else session.replay.pause();
        }
      }
      return;
    }
    if (event.target.closest && event.target.closest('.world-event')) return;  // typing an event
    if (event.code === 'KeyH' || event.code === 'KeyG') {
      event.preventDefault(); event.stopPropagation();
      if (event.type === 'keydown' && !event.repeat) {
        session.viewer.classList.toggle(event.code === 'KeyH' ? 'world-ui-hidden' : 'world-stats-shown');
        if (event.code === 'KeyG') refreshStats();
      }
      return;
    }
    const key = session.layout.keys[event.code];
    if (!key) return;
    event.preventDefault(); event.stopPropagation();
    if (event.type === 'keydown' && event.code.startsWith('Key') && event.key.length === 1 && label(key) !== event.key.toUpperCase()) {
      learned[key] = event.key.toUpperCase();
      showHelp();
    }
    if (event.type === 'keydown') { session.held.add(key); session.pressed.add(key); maybeMove(); }
    else session.held.delete(key);
    showKeys();
  }
  function onFullscreen() { if (session && document.fullscreenElement !== session.viewer) leave(); }
  function start(viewer) {
    if (session) leave();
    const choice = (name) => viewer.querySelector('.world-' + name + ' select').value;
    session = {viewer: viewer, path: viewer.dataset.world, layout: JSON.parse(viewer.dataset.controls), held: new Set(), pressed: new Set(), busy: true, ready: false, text: '',
               seconds: parseFloat(choice('seconds')), steps: parseInt(choice('steps')), memory: parseInt(choice('memory')), moves: 0};
    viewer.classList.add('exploring');
    const video = viewer.querySelector('.world-video');  // the exploration continues from the world's last frame
    video.loop = false; video.pause();
    // once the world's video is known: this element shows the clips later, which must not be moved to their end
    const world = video.getAttribute('src');
    const toEnd = () => { if (session && session.viewer === viewer && video.getAttribute('src') === world) video.currentTime = Math.max(0, video.duration - 0.05); };
    if (isFinite(video.duration)) toEnd();
    else video.addEventListener('loadedmetadata', toEnd, {once: true});
    document.addEventListener('keydown', onKey, true);
    document.addEventListener('keyup', onKey, true);
    document.addEventListener('fullscreenchange', onFullscreen);
    showHelp();
    learnLayout();
    if (viewer.requestFullscreen) viewer.requestFullscreen().catch(() => {});
    status('Preparing the world… (the model loads the first time)');
    send({action: 'start', path: session.path, seconds: session.seconds, steps: session.steps, memory: session.memory});
  }
  function leave(suspend = false) {
    if (!session) return;
    if (session.replay) stopReplay();
    const current = session;
    session = null;
    clearTimeout(current.errorTimer);
    refreshStats();
    document.removeEventListener('keydown', onKey, true);
    document.removeEventListener('keyup', onKey, true);
    document.removeEventListener('fullscreenchange', onFullscreen);
    current.viewer.classList.remove('exploring', 'world-ui-hidden', 'world-ready', 'world-stats-shown');
    const [video, ...buffers] = current.viewer.querySelectorAll('.world-video');
    buffers.forEach((buffer) => buffer.remove());
    video.classList.remove('world-buffer');
    video.loop = true; video.autoplay = !suspend;
    if (!suspend && current.viewer.dataset.video) video.src = current.viewer.dataset.video;
    if (!suspend && window.__meshGalleryNS.visible(current.viewer)) video.play().catch(() => {});
    else video.pause();
    if (document.fullscreenElement === current.viewer) document.exitFullscreen().catch(() => {});
    send({action: 'leave', path: current.path});
  }
  function onResponse(value) {
    let response;
    try { response = JSON.parse(value || '{}'); } catch (error) { return; }
    if (!session || response.path !== session.path) return;
    if (response.action === 'ready') {
      session.ready = true; session.busy = false;
      session.viewer.classList.add('world-ready');
      status('Ready: hold ' + hold());
      refreshStats();
      maybeMove();
    } else if (response.action === 'clip') {
      session.busy = false; session.moves += 1;
      showClip(response.url);
      if (response.stats) { session.moveStats = session.live = response.stats; showStats(); }
      status('');
      maybeMove();
    } else if (response.action === 'saved') {
      session.busy = false;
      status('Saved in the Worlds gallery (shown when you leave): ' + response.recorded);
      maybeMove();
    } else if (response.action === 'rendered') {
      session.busy = false;
      if (response.recorded) { status('Saved in the Video gallery: ' + response.recorded); maybeMove(); }
      else { session.replayUrl = response.url; session.replayMoves = session.moves; startReplay(response.url); }
    } else if (response.action === 'stats') {
      session.live = response.stats;
      showStats();
    } else if (response.action === 'error') {
      session.busy = true;
      status('The exploration could not run: ' + response.message);
      session.errorTimer = setTimeout(leave, 6000);
    } else if (response.action === 'ended') {
      leave();
    }
  }
  function render(record) {
    // the world's own video and the moves made so far, decoded by the model: recorded in the Video gallery, or replayed here
    if (!session || !session.ready || session.busy || session.replay) return;
    if (!record && session.replayUrl && session.replayMoves === session.moves) { startReplay(session.replayUrl); return; }
    session.busy = true;
    status(record ? 'Recording the video of the world and your moves…' : 'Preparing the replay…');
    send({action: 'render', path: session.path, record: record});
  }
  function startReplay(url) {
    const video = document.createElement('video');
    video.className = 'world-replay'; video.muted = true; video.playsInline = true; video.src = url;
    session.viewer.appendChild(video);
    session.replay = video;
    // Esc stops the replay instead of leaving the full screen (a long press still leaves it)
    if (navigator.keyboard && navigator.keyboard.lock) navigator.keyboard.lock(['Escape']).catch(() => {});
    video.addEventListener('ended', stopReplay, {once: true});
    video.play().catch(() => {});
    status('Replay - Space: Pause, Esc: Stop');
  }
  function stopReplay() {
    if (!session || !session.replay) return;
    session.replay.remove();
    session.replay = null;
    if (navigator.keyboard && navigator.keyboard.unlock) navigator.keyboard.unlock();
    status('');
    maybeMove();
  }
  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-world-action]');
    if (!button) return;
    const viewer = button.closest('.world-viewer'), action = button.dataset.worldAction;
    if (action === 'explore') start(viewer);
    else if (action === 'download-view') downloadView(viewer);
    else if (action === 'snapshot') addViewToGallery(viewer);
    else if (action === 'record') render(true);
    else if (action === 'replay') render(false);
    else if (action === 'save-world') saveWorld();
  });
  function saveWorld() {
    // the world and the moves made so far, as a new world of the Worlds gallery (shown there once the viewer leaves)
    if (!session || !session.ready || session.busy || session.replay) return;
    if (!session.moves) { status('Make a move first: the saved world is this world followed by your moves'); return; }
    session.busy = true;
    status('Saving the world and your moves…');
    send({action: 'save', path: session.path});
  }
  function memoryLabels(viewer) {
    // in moves of the chosen Move length: the button shows the choice, the list also the seconds and the memory each one needs
    const moveFrames = parseInt(viewer.querySelector('.world-seconds select').selectedOptions[0].dataset.frames);
    const select = viewer.querySelector('.world-memory select');
    const moves = (option) => { const count = Math.max(1, Math.round(parseInt(option.value) / moveFrames)); return `${count} move${count === 1 ? '' : 's'}`; };
    for (const option of select.options) {
      const item = viewer.querySelector(`.world-memory-list [data-value="${option.value}"]`);
      item.textContent = `${moves(option)} (${option.dataset.seconds} s) · ${option.dataset.needs}`;
      item.setAttribute('aria-selected', option.selected ? 'true' : 'false');
    }
    viewer.querySelector('.world-memory-button').textContent = moves(select.selectedOptions[0]);
  }
  function closeMemoryLists(except) {
    document.querySelectorAll('.world-memory-list:not([hidden])').forEach((list) => { if (list !== except) list.hidden = true; });
  }
  document.addEventListener('change', (event) => {
    const viewer = event.target.closest('.world-viewer');
    if (viewer && event.target.closest('.world-seconds, .world-memory')) memoryLabels(viewer);
  });
  document.addEventListener('click', (event) => {
    const memory = event.target.closest('.world-memory'), item = event.target.closest('.world-memory-list [data-value]');
    if (memory && !event.target.closest('.world-memory-list')) {  // the Memory label or its button opens / closes the list
      const viewer = memory.closest('.world-viewer'), list = viewer.querySelector('.world-memory-list');
      memoryLabels(viewer);
      closeMemoryLists(list);
      list.hidden = !list.hidden;
    } else if (item) {
      event.preventDefault();
      const select = item.closest('.world-memory').querySelector('select');
      select.value = item.dataset.value;
      select.dispatchEvent(new Event('change', {bubbles: true}));
      closeMemoryLists();
    } else if (!event.target.closest('.world-memory-list')) closeMemoryLists();
  });
  document.addEventListener('submit', (event) => {
    const form = event.target.closest('.world-event');
    if (!form) return;
    event.preventDefault();
    const input = form.querySelector('input');
    if (session && input.value.trim()) { session.text = input.value.trim(); input.value = ''; input.blur(); maybeMove(); }
  });
  class WorldViewer extends HTMLElement {
    connectedCallback() {queueMicrotask(() => {if (this.isConnected) this.refresh();});}
    disconnectedCallback() { this.suspend(); }
    suspend() {
      if (session?.viewer === this) leave(true);
      this.querySelectorAll('video').forEach(video => {video.pause(); video.removeAttribute('src'); video.load();});
      this.suspended = true;
    }
    refresh() {
      if (!window.__meshGalleryNS.visible(this)) {if (!this.suspended) this.suspend();}
      else if (this.suspended) {
        this.suspended = false;
        const video = this.querySelector('.world-video');
        if (video && this.dataset.video) {video.autoplay = false; video.controls = true; video.src = this.dataset.video;}
      }
    }
  }
  customElements.define('wangp-world-viewer', WorldViewer);
  const refreshViewers = () => document.querySelectorAll('wangp-world-viewer').forEach(viewer => viewer.refresh());
  document.addEventListener('visibilitychange', refreshViewers);
  document.addEventListener('wangp-viewer-scope', refreshViewers);
  document.addEventListener('fullscreenchange', refreshViewers);
  document.addEventListener('scroll', refreshViewers, {capture: true, passive: true});
  document.addEventListener('click', () => queueMicrotask(refreshViewers));
  window.addEventListener('resize', refreshViewers);
  window.addEventListener('blur', () => queueMicrotask(refreshViewers));
  window.addEventListener('focus', refreshViewers);
  window.addEventListener('pagehide', () => document.querySelectorAll('wangp-world-viewer').forEach(viewer => viewer.suspend()));
  return {onResponse: onResponse, start: start, leave: leave};
})();
// ===== end WorldGallery JS =====
"""


class AppGallery(StripGallery):
    """Web apps made with Deepy (.wgpapp): the selected app's thumbnail, name and controls above a strip of thumbnails. Play opens it in the
    app player (deepy/web/app_player.js), framed isolated from WanGP by shared.apps.server; nothing runs until Play is pressed."""

    source = APP
    viewer_height = 420

    def _create_player(self, path):
        return gr.HTML(value=self._player_value(path), elem_id=self.element_id("viewer"))

    def _player_value(self, path):
        if path is None:
            return f'<div class="app-viewer app-viewer-empty" style="height:{self.viewer_height}px"></div>'
        return self.viewer_html(path)

    @classmethod
    def viewer_html(cls, path):
        from shared.apps import server as apps_server
        from shared.utils.app_outputs import preview_path, read_app_metadata

        try:
            app_id = apps_server.register_package(Path(path))
            app = read_app_metadata(path)
        except Exception as error:
            return f'<div class="app-viewer" style="height:{cls.viewer_height}px"><p class="app-viewer-error">This app cannot be opened: {html.escape(str(error))}</p></div>'
        preview = preview_path(path)
        picture = f'<img src="{html.escape(gradio_file_url(preview))}" alt="" loading="lazy">' if os.path.isfile(preview) else '<span class="app-viewer-icon">▶</span>'
        controls = "".join(f'<span><kbd>{html.escape(str(key))}</kbd> {html.escape(str(action))}</span>' for key, action in (app["app_controls"] or {}).items())
        name = html.escape(app["app_name"] or Path(path).stem)
        version = html.escape(str(int(os.path.getmtime(path))))
        return f"""<div class="app-viewer" style="height:{cls.viewer_height}px">
<a class="app-viewer-play" href="#" role="button" data-app-id="{app_id}" data-app-name="{name}" data-app-version="{version}" data-download-url="{html.escape(gradio_file_url(path))}" data-download="{html.escape(Path(path).name)}" title="Play {name}">
{picture}<span class="app-viewer-badge"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>Play</span></a>
<div class="app-viewer-info"><strong>{name}</strong> <span class="app-viewer-version">version {html.escape(str(app["app_version"]))}</span>{f'<p>{html.escape(app["app_description"])}</p>' if app["app_description"] else ""}{f'<div class="app-viewer-controls">{controls}</div>' if controls else ""}</div>
</div>"""

    def _extra_file_info(self, path):
        from shared.utils.app_outputs import preview_path, read_app_metadata

        preview = preview_path(path)
        try:  # a package dropped in the outputs by hand may be damaged: its file name labels it
            name = read_app_metadata(path)["app_name"]
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            name = ""
        return {"preview": preview if os.path.isfile(preview) else None, "name": name or Path(path).stem}

    def _thumbnail_body(self, info):
        if info["deleted"]:
            return f'<div class="thumbnail-date">{info["date"]}</div><div class="thumbnail-time">{info["time"]}</div>'
        image = f'<img class="thumbnail-image" src="{html.escape(gradio_file_url(info["preview"]))}" alt="" loading="lazy">' if info["preview"] else '<div class="thumbnail-detail">App</div>'
        return f'{image}<div class="thumbnail-detail">{html.escape(info["name"])}</div>'

    _own_javascript = r"""
// ===== AppGallery: Play opens the app player =====
if (!window.__appGalleryInit) {
  window.__appGalleryInit = true;
  document.addEventListener('click', event => {
    const link = event.target.closest('.app-viewer [data-app-id]');
    if (!link || !window.WanGPAppPlayer) return;
    event.preventDefault();
    window.WanGPAppPlayer.open(link);
  });
  const style = document.createElement('style');
  style.textContent = `
.app-viewer { display: flex; gap: 16px; align-items: stretch; padding: 12px; box-sizing: border-box; border-radius: 8px; background: var(--studio-input); color: var(--body-text-color); }
.app-viewer-empty { padding: 0; }
.app-viewer-play { position: relative; flex: 1.6; min-width: 0; display: grid; place-items: center; border-radius: 8px; overflow: hidden; background: #000; text-decoration: none !important; }
.app-viewer-play img { width: 100%; height: 100%; object-fit: contain; }
.app-viewer-icon { color: #fff; font-size: 48px; }
.app-viewer-badge { position: absolute; display: inline-flex; align-items: center; gap: 8px; padding: 10px 20px; border-radius: 999px; background: rgba(0, 0, 0, 0.66); color: #fff !important; font-weight: 600; font-size: 16px; }
.app-viewer-badge svg { width: 18px; height: 18px; fill: currentColor; }
.app-viewer-play:hover .app-viewer-badge { background: var(--studio-accent, #3b82f6); }
.app-viewer-info { flex: 1; min-width: 0; overflow: auto; font-size: 14px; line-height: 1.5; }
.app-viewer-info strong { font-size: 16px; }
.app-viewer-version { color: var(--body-text-color-subdued); }
.app-viewer-controls { display: flex; flex-wrap: wrap; gap: 6px 12px; margin-top: 8px; }
.app-viewer-controls kbd { padding: 1px 6px; border: 1px solid var(--studio-border); border-bottom-width: 2px; border-radius: 5px; font-size: 12px; }
.app-viewer-error { margin: auto; }
@media (max-width: 760px) { .app-viewer { flex-direction: column; } }`;
  document.head.append(style);
}
// ===== end AppGallery JS =====
"""


STRIP_GALLERY_CLASSES = {gallery.source.id: gallery for gallery in (AudioGallery, MeshGallery, WorldGallery, AppGallery)}
