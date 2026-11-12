# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import html
import json
import re
from functools import lru_cache

import markdown as markdown_module
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor

from shared.utils.markdown import EscapeHtmlExtension


class _SafeHelpLinks(Treeprocessor):
    def run(self, root):
        for element in root.iter():
            if element.tag == "a":
                href = element.get("href", "")
                if not re.match(r"^https?://", href, re.IGNORECASE):
                    element.attrib.pop("href", None)
                else:
                    element.set("target", "_blank")
                    element.set("rel", "noopener noreferrer")
            elif element.tag == "img":
                if not re.match(r"^https?://", element.get("src", ""), re.IGNORECASE):
                    element.attrib.pop("src", None)


class _SafeHelpLinksExtension(Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(_SafeHelpLinks(md), "safe_help_links", 1)


def _render_markdown(markdown: str) -> str:
    source = str(markdown or "").strip()
    if not source:
        return ""
    return _render_markdown_source(source)


@lru_cache(maxsize=128)
def _render_markdown_source(source: str) -> str:
    # Cache only rendered help text, never popup IDs, component or session state.
    # Model/plugin edits produce a different source and render immediately.
    return markdown_module.markdown(
        source,
        extensions=["tables", "fenced_code", "sane_lists", EscapeHtmlExtension(), _SafeHelpLinksExtension()],
        output_format="html5",
    )


def _normalize_infos(infos, model_name: str) -> tuple[str, str]:
    return str(model_name or "Model"), str(infos or "")


def _json_script_payload(value: str) -> str:
    return json.dumps(value).replace("</", "<\\/")


def render_info_trigger(popup_id: str, title: str, *, extra_class: str = "") -> str:
    title_attr = html.escape(title, quote=True)
    classes = "wangp-model-info-trigger" + (f" {html.escape(extra_class, quote=True)}" if extra_class else "")
    return f"<button type='button' class='{classes}' title='{title_attr}' aria-label='{title_attr}' data-wangp-model-info-open='{html.escape(popup_id, quote=True)}'>i</button>"


def render_info_popup(popup_id: str, title: str, markdown: str, *, lazy: bool = False) -> str:
    title_attr = html.escape(title, quote=True)
    title_html = html.escape(title)
    if lazy:
        payload_id = f"{popup_id}-payload"
        payload = html.escape(_json_script_payload(_render_markdown(markdown)), quote=False)
        content = (
            f"<div class='wangp-model-info-content' data-wangp-model-info-content data-wangp-model-info-source='{html.escape(payload_id, quote=True)}'>"
            "<div class='wangp-model-info-loading'>Loading...</div>"
            "</div>"
            f"<textarea id='{html.escape(payload_id, quote=True)}' hidden>{payload}</textarea>"
        )
    else:
        content = f"<div class='wangp-model-info-content'>{_render_markdown(markdown)}</div>"
    return (
        f"<div id='{html.escape(popup_id, quote=True)}' class='wangp-model-info-popup' role='dialog' aria-label='{title_attr}' data-wangp-model-info-popup hidden>"
        "<div class='wangp-model-info-card'>"
        "<div class='wangp-model-info-titlebar' data-wangp-model-info-drag>"
        f"<div class='wangp-model-info-heading'>{title_html}</div>"
        "<button type='button' class='wangp-model-info-close' aria-label='Close information' data-wangp-model-info-close>&times;</button>"
        "</div>"
        f"{content}"
        "</div>"
        "</div>"
    )


def render_info_trigger_and_popup(popup_id: str, title: str, markdown: str, *, lazy: bool = False) -> str:
    return render_info_trigger(popup_id, title) + render_info_popup(popup_id, title, markdown, lazy=lazy)


def _render_info_trigger_and_popup(popup_id: str, title: str, markdown: str, *, lazy: bool = False) -> str:
    return render_info_trigger_and_popup(popup_id, title, markdown, lazy=lazy)


def render_model_description(description: str, infos=None, *, model_type: str = "", model_name: str = "Model", height: int = 40) -> str:
    if not infos:
        return f"<div class='wangp-model-info-host' style='min-height:{int(height)}px'><div class='wangp-model-info-description'>{description}</div></div>"
    title, markdown = _normalize_infos(infos, model_name)
    if not markdown.strip():
        return f"<div class='wangp-model-info-host' style='min-height:{int(height)}px'><div class='wangp-model-info-description'>{description}</div></div>"
    popup_id = "wangp-model-info-" + re.sub(r"[^A-Za-z0-9_-]", "-", str(model_type or model_name)).strip("-").lower()
    return (
        f"<div class='wangp-model-info-host' style='min-height:{int(height)}px'>"
        f"{_render_info_trigger_and_popup(popup_id, title, markdown)}"
        f"<div class='wangp-model-info-description'>{description}</div>"
        "</div>"
    )


def render_prompt_label(label: str, infos=None, *, model_type: str = "", prompt_id: str = "prompt", title: str = "Prompt Guidelines", lazy: bool = False) -> str:
    if not infos:
        return ""
    popup_title, markdown = _normalize_infos(infos, title)
    if not markdown.strip():
        return ""
    popup_key = re.sub(r"[^A-Za-z0-9_-]", "-", f"{model_type or 'model'}-{prompt_id or 'prompt'}").strip("-").lower()
    popup_id = f"wangp-prompt-info-{popup_key}"
    return (
        "<div class='wangp-prompt-info-host'>"
        f"{_render_info_trigger_and_popup(popup_id, popup_title, markdown, lazy=lazy)}"
        "</div>"
    )


def get_css() -> str:
    return """
.wangp-model-info-host {
    position: relative;
    padding-right: 0;
    display: flex;
    align-items: center;
    gap: 4px;
}
.wangp-model-info-host > .wangp-model-info-trigger {
    position: static;
    order: 1;
    flex: 0 0 22px;
    margin: 0;
    width: 22px;
    height: 22px;
    min-width: 22px;
    min-height: 22px;
    border: 1px solid currentColor;
    background: transparent;
    color: #175a79;
    font: 500 17px/1 Arial, sans-serif;
}
.header-markdown-group .html-container {
    padding: 0 !important;
}
.wangp-model-info-description {
    flex: 1;
    min-width: 0;
    line-height: 1.35;
}
.wangp-prompt-info-host {
    position: relative;
    width: 100%;
    height: 0;
    margin: 0;
    overflow: visible;
    pointer-events: none;
    z-index: 50;
}
.wangp-prompt-info-stack {
    position: relative;
}
.wangp-prompt-info-stack .wangp-prompt-info-anchor {
    position: absolute !important;
    inset: 0 0 auto 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    z-index: 50;
}
.wangp-prompt-info-anchor,
.wangp-prompt-info-anchor > *,
.wangp-prompt-info-anchor .html-container {
    min-height: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    overflow: visible !important;
    scrollbar-width: none !important;
}
.wangp-prompt-info-anchor::-webkit-scrollbar,
.wangp-prompt-info-anchor *::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}
.wangp-prompt-info-host .wangp-model-info-trigger {
    pointer-events: auto;
    z-index: 60;
    top: 23px;
    right: 8px;
    width: 18px;
    height: 18px;
    min-width: 18px;
    min-height: 18px;
    border: 1px solid var(--button-secondary-border-color, rgba(17, 84, 118, 0.24));
    background: var(--button-secondary-background-fill, rgba(255, 255, 255, 0.86));
    box-shadow: none;
    color: var(--button-secondary-text-color, #155574);
    font-size: 11px;
}
.wangp-prompt-info-host .wangp-model-info-trigger:hover {
    box-shadow: none;
}
.wangp-model-info-trigger {
    position: absolute;
    top: 1px;
    right: 2px;
    width: 26px;
    height: 26px;
    min-width: 26px;
    min-height: 26px;
    padding: 0;
    border: 1px solid var(--button-secondary-border-color, rgba(17, 84, 118, 0.18));
    border-radius: 999px;
    background: var(--button-secondary-background-fill, linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(238, 247, 252, 0.98) 100%));
    color: var(--button-secondary-text-color, #155574);
    box-shadow: none;
    cursor: pointer;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.wangp-model-info-trigger:hover {
    border-color: rgba(16, 86, 121, 0.36);
    box-shadow: none;
}
.wangp-model-info-popup[hidden] {
    display: none !important;
}
.wangp-model-info-popup {
    position: fixed;
    top: 96px;
    right: 32px;
    width: min(680px, calc(100vw - 34px));
    max-height: min(78vh, 720px);
    z-index: 1200;
    pointer-events: none;
}
.wangp-model-info-card {
    pointer-events: auto;
    overflow: hidden;
    border-radius: 18px;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16));
    background: var(--background-fill-primary, rgba(255, 255, 255, 0.99));
    box-shadow: 0 28px 62px rgba(7, 31, 48, 0.24);
    color: var(--body-text-color, #174a67);
}
.wangp-model-info-titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 10px 14px 9px 16px;
    background: var(--button-primary-background-fill, linear-gradient(180deg, rgba(16, 86, 121, 0.98) 0%, rgba(10, 59, 84, 0.98) 100%));
    color: var(--button-primary-text-color, #f3fbff);
    cursor: move;
    user-select: none;
    touch-action: none;
}
.wangp-model-info-titlebar:active {
    cursor: move;
}
.wangp-model-info-heading {
    color: var(--button-primary-text-color, #f3fbff) !important;
    font-size: 0.92rem;
    font-weight: 800;
}
.wangp-model-info-close {
    width: 26px;
    height: 26px;
    min-width: 26px;
    min-height: 26px;
    padding: 0;
    border: 1px solid var(--button-primary-border-color, rgba(255, 255, 255, 0.24));
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    color: var(--button-primary-text-color, #f3fbff);
    cursor: pointer;
    font-size: 20px;
    line-height: 1;
}
.wangp-model-info-content {
    max-height: calc(min(78vh, 720px) - 46px);
    overflow: auto;
    padding: 16px 18px 18px;
    color: var(--body-text-color, #174a67);
    font-size: 0.92rem;
    line-height: 1.5;
}
.wangp-prompt-helper-popup {
    box-sizing: border-box;
    max-height: calc(100vh - 12px);
    min-width: 360px;
    min-height: 320px;
    padding: 0 8px 8px 0;
    border-radius: 18px;
    filter: drop-shadow(0 20px 34px rgba(7, 31, 48, 0.24)) drop-shadow(0 3px 10px rgba(7, 31, 48, 0.14));
    overflow: hidden;
    pointer-events: auto;
    resize: both;
}
.wangp-prompt-helper-popup .wangp-model-info-card {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    max-width: calc(100vw - 24px);
    max-height: 100%;
    box-shadow: none;
    resize: none;
}
.wangp-prompt-helper-popup .wangp-model-info-content {
    flex: 1 1 auto;
    min-height: 0;
    max-height: none;
}
.wangp-model-info-content h1,
.wangp-model-info-content h2,
.wangp-model-info-content h3,
.wangp-model-info-content h4,
.wangp-model-info-content h5,
.wangp-model-info-content h6 {
    margin: 12px 0 7px;
    color: var(--body-text-color, #103f59);
    font-weight: 800;
}
.wangp-model-info-content h1:first-child,
.wangp-model-info-content h2:first-child,
.wangp-model-info-content h3:first-child,
.wangp-model-info-content h4:first-child,
.wangp-model-info-content h5:first-child,
.wangp-model-info-content h6:first-child {
    margin-top: 0;
}
.wangp-model-info-content p,
.wangp-model-info-content ul,
.wangp-model-info-content ol,
.wangp-model-info-content blockquote,
.wangp-model-info-content table {
    margin: 0 0 11px;
}
.wangp-model-info-content ul,
.wangp-model-info-content ol {
    padding-left: 20px;
}
.wangp-model-info-content blockquote {
    padding-left: 13px;
    border-left: 3px solid var(--border-color-primary, rgba(17, 84, 118, 0.24));
}
.wangp-model-info-content table {
    width: 100%;
    border-collapse: collapse;
    overflow-wrap: anywhere;
}
.wangp-model-info-content th,
.wangp-model-info-content td {
    padding: 7px 9px;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.18));
    text-align: left;
    vertical-align: top;
}
.wangp-model-info-content th {
    background: var(--background-fill-secondary, rgba(16, 86, 121, 0.08));
}
.wangp-model-info-content code {
    padding: 1px 4px;
    border-radius: 5px;
    background: var(--background-fill-secondary, rgba(16, 86, 121, 0.08));
    color: var(--body-text-color, #0f4967);
}
.wangp-model-info-content pre {
    margin: 8px 0 13px;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.12));
    background: var(--background-fill-secondary, #f4f9fc);
    overflow: auto;
}
.wangp-model-info-content pre code {
    padding: 0;
    border-radius: 0;
    background: transparent;
    color: var(--body-text-color, #123f58);
}
.wangp-model-info-loading {
    opacity: 0.72;
    font-style: italic;
}
.wangp-confirm-backdrop {
    position: fixed;
    inset: 0;
    z-index: 5000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 18px;
    background: rgba(7, 27, 38, 0.34);
    backdrop-filter: blur(2px);
}
.wangp-confirm-backdrop[hidden] {
    display: none !important;
}
.wangp-confirm-card {
    width: min(520px, 100%);
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.18));
    border-radius: 8px;
    background: var(--background-fill-primary, #fff);
    box-shadow: 0 18px 48px rgba(8, 34, 48, 0.28);
    color: var(--body-text-color, #174a67);
    overflow: hidden;
}
.wangp-confirm-title {
    padding: 15px 16px 5px;
    font-size: 1rem;
    font-weight: 800;
}
.wangp-confirm-message {
    padding: 0 16px 14px;
    color: var(--body-text-color-subdued, #5d7787);
    font-size: 0.86rem;
    line-height: 1.35;
    white-space: pre-line;
}
.wangp-confirm-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.12));
    background: var(--background-fill-secondary, #f4f9fc);
}
.wangp-confirm-actions button {
    min-width: 78px;
    height: 30px;
    padding: 0 12px;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.18));
    border-radius: 6px;
    background: var(--button-secondary-background-fill, #fff);
    color: var(--button-secondary-text-color, #174a67);
    cursor: pointer;
    font: inherit;
    font-weight: 700;
}
.wangp-confirm-actions button.primary {
    border-color: rgba(21, 96, 130, 0.34);
    background: #156082;
    color: #fff;
}
.wangp-confirm-actions button.danger {
    border-color: rgba(180, 35, 24, 0.34);
    background: #b42318;
    color: #fff;
}
"""


def get_javascript() -> str:
    return """
    window.wangpModelInfo = window.wangpModelInfo || {};
    window.wangpConfirm = function(options) {
        const opts = typeof options === "string" ? { message: options } : (options || {});
        return new Promise((resolve) => {
            const previousFocus = document.activeElement;
            const backdrop = document.createElement("div");
            backdrop.className = "wangp-confirm-backdrop";
            backdrop.setAttribute("role", "dialog");
            backdrop.setAttribute("aria-modal", "true");
            const buttonDefs = Array.isArray(opts.buttons) && opts.buttons.length ? opts.buttons : [
                { text: opts.cancelText || "Cancel", value: false, cancel: true },
                { text: opts.confirmText || "OK", value: true, primary: true, danger: !!opts.danger }
            ];
            backdrop.innerHTML = `
                <div class="wangp-confirm-card">
                    <div class="wangp-confirm-title"></div>
                    <div class="wangp-confirm-message"></div>
                    <div class="wangp-confirm-actions"></div>
                </div>`;
            const title = backdrop.querySelector(".wangp-confirm-title");
            const message = backdrop.querySelector(".wangp-confirm-message");
            const actions = backdrop.querySelector(".wangp-confirm-actions");
            title.textContent = opts.title || "Confirm";
            message.textContent = opts.message || "";
            let defaultButton = null;
            function finish(value) {
                document.removeEventListener("keydown", onKeyDown, true);
                backdrop.remove();
                if (previousFocus && document.contains(previousFocus)) previousFocus.focus();
                resolve(value);
            }
            function buttonValue(def) {
                return Object.prototype.hasOwnProperty.call(def, "value") ? def.value : def.text;
            }
            async function choose(def, event) {
                if (typeof def.action === "function") {
                    const result = await def.action(buttonValue(def), event);
                    if (result === false) return;
                }
                finish(buttonValue(def));
            }
            function onKeyDown(event) {
                if (event.key === "Escape") {
                    event.preventDefault();
                    const cancelDef = buttonDefs.find((def) => def.cancel) || { value: false };
                    finish(buttonValue(cancelDef));
                }
            }
            buttonDefs.forEach((def, index) => {
                const button = document.createElement("button");
                button.type = "button";
                button.textContent = def.text || def.label || String(buttonValue(def));
                button.className = def.className || "";
                button.classList.toggle("primary", !!def.primary);
                button.classList.toggle("danger", !!def.danger);
                button.addEventListener("click", (event) => choose(def, event));
                actions.appendChild(button);
                if (def.autofocus || (!defaultButton && !def.cancel && index === buttonDefs.length - 1)) defaultButton = button;
            });
            backdrop.addEventListener("click", (event) => {
                if (event.target !== backdrop) return;
                const cancelDef = buttonDefs.find((def) => def.cancel) || { value: false };
                finish(buttonValue(cancelDef));
            });
            document.addEventListener("keydown", onKeyDown, true);
            document.body.appendChild(backdrop);
            (defaultButton || actions.querySelector("button"))?.focus();
        });
    };
    window.wangpModelInfo.hydrate = function(popup) {
        const content = popup?.querySelector("[data-wangp-model-info-content]");
        if (!content || content.dataset.wangpHydrated === "1") return;
        const sourceId = content.getAttribute("data-wangp-model-info-source");
        const source = sourceId ? document.getElementById(sourceId) : null;
        if (!source) return;
        try {
            content.innerHTML = JSON.parse(source.value || source.textContent || '""');
            content.dataset.wangpHydrated = "1";
        } catch (_err) {
            content.innerHTML = "<p>Unable to load help content.</p>";
            content.dataset.wangpHydrated = "1";
        }
    };
    window.wangpModelInfo.requestClose = function(popup) {
        if (!popup || popup.hidden) return true;
        const event = new CustomEvent("wangp:model-info-before-close", { bubbles: true, cancelable: true });
        popup.dispatchEvent(event);
        if (event.defaultPrevented) return false;
        popup.hidden = true;
        return true;
    };
    window.wangpModelInfo.open = function(button) {
        const popupId = button?.getAttribute("data-wangp-model-info-open");
        const popup = (popupId ? document.getElementById(popupId) : null)
            || button?.parentElement?.querySelector(".wangp-model-info-popup");
        if (!popup) return;
        const wasOpen = !popup.hidden;
        window.wangpModelInfo.hydrate(popup);
        for (const other of document.querySelectorAll("[data-wangp-model-info-popup], .wangp-model-info-popup")) {
            if (other !== popup && !window.wangpModelInfo.requestClose(other)) return;
        }
        if (wasOpen) {
            window.wangpModelInfo.requestClose(popup);
            return;
        }
        popup.hidden = false;
        const isPromptHelper = popup.getAttribute("data-wangp-prompt-helper-popup") === "1";
        if (isPromptHelper) {
            const width = popup.getBoundingClientRect().width;
            popup.style.setProperty("left", Math.max(12, Math.round((window.innerWidth - width) / 2)) + "px");
            popup.style.setProperty("right", "auto");
            popup.style.setProperty("top", "0px");
        } else {
            popup.style.setProperty("left", "auto");
            popup.style.setProperty("right", "32px");
            popup.style.setProperty("top", "96px");
        }
        popup.dispatchEvent(new CustomEvent("wangp:model-info-opened", { bubbles: true }));
    };
    window.wangpModelInfo.close = function(closeButton) {
        const popup = closeButton?.closest("[data-wangp-model-info-popup], .wangp-model-info-popup");
        window.wangpModelInfo.requestClose(popup);
    };
    window.wangpModelInfo.alignPromptInfoButtons = function() {
        document.querySelectorAll(".wangp-prompt-info-anchor").forEach((anchor) => {
            const trigger = anchor.querySelector(".wangp-prompt-info-host .wangp-model-info-trigger");
            if (!trigger || !anchor.classList.contains("block")) return;
            let target = null;
            for (let sibling = anchor.nextElementSibling; sibling && !target; sibling = sibling.nextElementSibling) {
                if (sibling.classList?.contains("wangp-prompt-info-anchor")) continue;
                const label = sibling.querySelector?.("label");
                if (!label) continue;
                target = label.querySelector("span") || label.firstElementChild || label;
            }
            if (!target) return;
            const host = trigger.closest(".wangp-prompt-info-host");
            const hostRect = host.getBoundingClientRect();
            const targetRect = target.getBoundingClientRect();
            const triggerRect = trigger.getBoundingClientRect();
            if (!hostRect.width || !targetRect.height) return;
            trigger.style.top = Math.max(0, targetRect.top - hostRect.top + (targetRect.height - triggerRect.height) / 2) + "px";
        });
    };
    window.wangpModelInfo.schedulePromptInfoAlign = function() {
        if (window.wangpModelInfo.alignPending) return;
        window.wangpModelInfo.alignPending = true;
        requestAnimationFrame(() => {
            window.wangpModelInfo.alignPending = false;
            window.wangpModelInfo.alignPromptInfoButtons();
        });
    };
    let wangpModelInfoDrag = null;
    document.addEventListener("click", (event) => {
        const openButton = event.target.closest("[data-wangp-model-info-open], .wangp-model-info-trigger");
        if (openButton) {
            event.preventDefault();
            event.stopPropagation();
            window.wangpModelInfo.open(openButton);
            return;
        }
        const closeButton = event.target.closest("[data-wangp-model-info-close], .wangp-model-info-close");
        if (closeButton) {
            event.preventDefault();
            event.stopPropagation();
            window.wangpModelInfo.close(closeButton);
        }
    });
    document.addEventListener("pointerdown", (event) => {
        const handle = event.target.closest("[data-wangp-model-info-drag], .wangp-model-info-titlebar, .wangp-local-file-picker-titlebar");
        if (!handle || event.target.closest("[data-wangp-model-info-close], .wangp-model-info-close, .wangp-local-file-picker-close")) return;
        const popup = handle.closest("[data-wangp-model-info-popup], .wangp-model-info-popup, .wangp-local-file-picker-popup");
        if (!popup) return;
        const rect = popup.getBoundingClientRect();
        popup.style.setProperty("left", rect.left + "px", "important");
        popup.style.setProperty("top", rect.top + "px", "important");
        popup.style.setProperty("right", "auto", "important");
        wangpModelInfoDrag = { popup, pointerId: event.pointerId, offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top };
        handle.setPointerCapture?.(event.pointerId);
        event.preventDefault();
    });
    document.addEventListener("pointermove", (event) => {
        if (!wangpModelInfoDrag || wangpModelInfoDrag.pointerId !== event.pointerId) return;
        const margin = 10;
        const popup = wangpModelInfoDrag.popup;
        const rect = popup.getBoundingClientRect();
        const left = Math.min(Math.max(margin, event.clientX - wangpModelInfoDrag.offsetX), Math.max(margin, window.innerWidth - rect.width - margin));
        const top = Math.min(Math.max(margin, event.clientY - wangpModelInfoDrag.offsetY), Math.max(margin, window.innerHeight - 48));
        popup.style.setProperty("left", left + "px", "important");
        popup.style.setProperty("top", top + "px", "important");
        event.preventDefault();
    });
    document.addEventListener("pointerup", (event) => {
        if (wangpModelInfoDrag && wangpModelInfoDrag.pointerId === event.pointerId) wangpModelInfoDrag = null;
    });
    document.addEventListener("pointercancel", (event) => {
        if (wangpModelInfoDrag && wangpModelInfoDrag.pointerId === event.pointerId) wangpModelInfoDrag = null;
    });
    if (document.querySelector(".wangp-prompt-info-anchor")) {
        window.addEventListener("load", window.wangpModelInfo.schedulePromptInfoAlign);
        window.addEventListener("resize", window.wangpModelInfo.schedulePromptInfoAlign);
        setTimeout(window.wangpModelInfo.schedulePromptInfoAlign, 250);
        setTimeout(window.wangpModelInfo.schedulePromptInfoAlign, 1200);
    }
"""
