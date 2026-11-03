# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Media Flow's app chrome and help: the brand bar, the Guide (the sections of docs/MEDIA_FLOW.md, sent when it is first opened), each card's
header with a link to its guide section, and the card describing the selected process."""
from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

import markdown as markdown_module

from . import constants as ui_constants
from . import process_catalog as catalog

CSS_PATH = Path(__file__).with_name("media_flow.css")
GUIDE_PATH = Path(__file__).resolve().parents[2] / "docs" / "MEDIA_FLOW.md"
GUIDE_ID = "mediaflow-guide"
GUIDE_BUTTON_ID = "mediaflow-guide-btn"
# After the Guide button's server step (which sends the guide the first time): select the section a card asked for, then open the window, or
# close it when the button is pressed while it is open. The window arrives with the step's output, a frame or so after this script starts.
GUIDE_OPEN_JS = """() => {
    const section = window.mediaflowGuideSection;
    delete window.mediaflowGuideSection;
    let frames = 0;
    const show = () => {
        const popup = document.getElementById('%s');
        if (!popup) { if (frames++ < 30) requestAnimationFrame(show); return; }
        const radio = section && popup.querySelector(`input[data-section="${CSS.escape(section)}"]`);
        if (radio) { radio.checked = true; popup.querySelector('.mf-guide-pages').scrollTop = 0; }
        if (radio && !popup.hidden) return;
        const opener = document.createElement('button');
        opener.setAttribute('data-wangp-model-info-open', '%s');
        window.wangpModelInfo.open(opener);
    };
    show();
}""" % (GUIDE_ID, GUIDE_ID)

ICONS = {
    "brand": '<path d="M3 15c3.5 0 4-6 7.5-6S14 15 17.5 15 21 11 21 11"/><path d="M3 9c3.5 0 4-4 7.5-4S14 9 17.5 9"/><path d="M3 19.5c3.5 0 4-3 7.5-3s3.5 3 7 3"/>',
    "book": '<path d="M12 6C9 3 5 3 2 4v15c3-1 7-1 10 2 3-3 7-3 10-2V4c-3-1-7-1-10 2Zm0 0v15"/>',
    "overview": '<path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9Z"/><path d="M19 15v4m-2-2h4M5 3v3M3.5 4.5h3"/>',
    "start": '<circle cx="12" cy="12" r="9"/><path d="m10 8 6 4-6 4Z"/>',
    "process": '<path d="m12 3 9 5-9 5-9-5Z"/><path d="m3 13 9 5 9-5"/>',
    "source": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
    "output": '<path d="M4 7h10M4 17h6M18 7h2M14 17h6"/><circle cx="16" cy="7" r="2"/><circle cx="12" cy="17" r="2"/>',
    "long": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9.5h4m10 0h4M3 14.5h4m10 0h4"/>',
    "prompt": '<path d="M4 6h16M4 11h16M4 16h9"/><path d="m17 15 1.2 2.8L21 19l-2.8 1.2L17 23l-1.2-2.8L13 19l2.8-1.2Z"/>',
    "resume": '<path d="M20 12a8 8 0 1 1-2.3-5.6"/><path d="M20 4v5h-5"/>',
    "tips": '<path d="M9 18h6m-5 3h4"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.6.5 1.1 1.2 1.1 2.2h5c0-1 .5-1.7 1.1-2.2A6 6 0 0 0 12 3Z"/>',
    "run": '<path d="M13 2 4 14h7l-1 8 9-12h-7Z"/>',
    "preview": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
    "result": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5M9 14l2 2 4-4"/>',
    "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    "outpaint": '<rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M3 8V5a2 2 0 0 1 2-2h3m8 0h3a2 2 0 0 1 2 2v3m0 8v3a2 2 0 0 1-2 2h-3m-8 0H5a2 2 0 0 1-2-2v-3"/>',
    "refocus": '<circle cx="12" cy="12" r="3"/><path d="M3 8V5a2 2 0 0 1 2-2h3m8 0h3a2 2 0 0 1 2 2v3m0 8v3a2 2 0 0 1-2 2h-3m-8 0H5a2 2 0 0 1-2-2v-3"/>',
    "uncompress": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h6V3m6 18v-6h6M9 21v-6H3m18-6h-6V3"/>',
    "ungrade": '<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18Z" fill="currentColor" stroke="none" opacity=".35"/><path d="M12 3v18"/>',
    "hdr": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "detail": '<path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9Z"/><path d="M19 16v5m-2.5-2.5h5"/>',
    "upscale": '<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>',
    "interpolate": '<rect x="2" y="7" width="5" height="10" rx="1"/><rect x="9.5" y="7" width="5" height="10" rx="1" opacity=".55"/><rect x="17" y="7" width="5" height="10" rx="1"/>',
    "custom": '<path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6m2-6h6m2 8h6"/>',
}
TASKS = {
    "outpaint": ("Outpaint", "Extends the picture to a new aspect ratio, inventing the missing borders."),
    "refocus": ("Refocus", "Removes focus and motion blur."),
    "uncompress": ("Uncompress", "Removes blocking, banding and smearing left by heavy compression."),
    "ungrade": ("Ungrade", "Removes a stylized color grade and restores neutral colors."),
    "hdr": ("SDR to HDR", "Converts standard dynamic range media to HDR."),
    "detail": ("Detail", "Adds fine texture and sharper edges."),
    "upscale": ("Upscale", "Increases the resolution."),
    "interpolate": ("Interpolate", "Adds frames between frames for smoother motion and a higher frame rate."),
    "custom": ("User Settings", "Runs your own WanGP settings on each piece of the source."),
}
GUIDE_ICONS = {"Overview": "overview", "Quick Start": "start", "Choosing a Process": "process", "Sources and Batches": "source", "Output and Quality": "output",
               "Long Videos": "long", "Prompts": "prompt", "Stop and Resume": "resume", "Tips and Troubleshooting": "tips"}


def icon(name: str) -> str:
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'


def stylesheet() -> str:
    return f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"


def _markdown(text: str) -> str:
    return markdown_module.markdown(text, extensions=["tables", "fenced_code", "sane_lists"], output_format="html5")


@lru_cache(maxsize=1)
def guide_sections() -> dict[str, str]:
    """The guide's sections by title (its introduction as Overview), without the documentation scope footer."""
    text = GUIDE_PATH.read_text(encoding="utf-8").split("\n---\n")[0]
    intro, *parts = re.split(r"^## ", text, flags=re.MULTILINE)
    sections = {"Overview": intro.split("\n", 1)[1].strip()}
    for part in parts:
        title, body = part.split("\n", 1)
        sections[title.strip()] = body.strip()
    return sections


def brand_html() -> str:
    return (f"<div class='mf-brand'><div class='mf-mark'>{icon('brand')}</div><div><div class='mf-brand-name'>Media Flow</div>"
            "<div class='mf-brand-tag'><b>Give your media a new life</b> &middot; restore, outpaint, upscale, smooth</div></div></div>")


def _task_tile(task: str) -> str:
    name, description = TASKS[task]
    return f"<div class='mf-tile'><div class='mf-process-icon'>{icon(task)}</div><div><b>{name}</b><span>{description}</span></div></div>"


def guide_html() -> str:
    """The Guide window: a page per section, chosen with radio labels, shown as one of WanGP's help popups."""
    sections = guide_sections()
    radios, links, pages, rules = [], [], [], []
    for index, (title, body) in enumerate(sections.items()):
        radio_id = f"{GUIDE_ID}-{index}"
        radios.append(f"<input type='radio' class='mf-guide-radio' name='{GUIDE_ID}-page' id='{radio_id}' data-section='{html.escape(title, quote=True)}'{' checked' if index == 0 else ''}>")
        links.append(f"<label for='{radio_id}'>{icon(GUIDE_ICONS.get(title, 'book'))}<span>{html.escape(title)}</span></label>")
        if index == 0:
            content = (f"<div class='mf-guide-hero'><h2>Give your media a new life</h2>{_markdown(body)}</div>"
                       f"<div class='mf-tiles'>{''.join(_task_tile(task) for task in TASKS)}</div>")
        else:
            content = f"<h2>{html.escape(title)}</h2>{_markdown(body)}"
        pages.append(f"<article class='mf-guide-page' data-page='{index}'>{content}</article>")
        rules.append(f"#{radio_id}:checked~.mf-guide-pages>[data-page='{index}']{{display:block}}"
                     f"#{radio_id}:checked~.mf-guide-nav>[for='{radio_id}']{{background:var(--mf-accent-soft);color:var(--mf-text)}}"
                     f"#{radio_id}:checked~.mf-guide-nav>[for='{radio_id}'] svg{{color:var(--mf-accent)}}")
    return (f"<style>{''.join(rules)}</style>"
            f"<div id='{GUIDE_ID}' class='wangp-model-info-popup' role='dialog' aria-label='Media Flow Guide' data-wangp-model-info-popup hidden>"
            "<div class='wangp-model-info-card'><div class='wangp-model-info-titlebar' data-wangp-model-info-drag>"
            f"<div class='wangp-model-info-heading'><span class='mf-mark' style='width:28px;height:28px;border-radius:9px'>{icon('brand')}</span>Media Flow Guide</div>"
            "<button type='button' class='wangp-model-info-close' aria-label='Close the guide' data-wangp-model-info-close>&times;</button></div>"
            f"<div class='mf-guide-body'>{''.join(radios)}<nav class='mf-guide-nav'>{''.join(links)}</nav><div class='mf-guide-pages'>{''.join(pages)}</div></div>"
            "</div></div>")


def card_head(title: str, subtitle: str, *, section: str | None = None, icon_name: str | None = None) -> str:
    """A card's title, numbered as a setup step unless it has an icon, with a button opening the Guide at its section."""
    badge = f"<span class='mf-card-icon'>{icon(icon_name)}</span>" if icon_name else "<span class='mf-step'></span>"
    help_html = ""
    if section is not None:
        assert section in guide_sections(), section
        click = f"window.mediaflowGuideSection={json.dumps(section)};document.getElementById('{GUIDE_BUTTON_ID}').click()"
        help_html = f"<button type='button' class='mf-help' title='Help: {html.escape(section, quote=True)}' aria-label='Help: {html.escape(section, quote=True)}' onclick=\"{html.escape(click, quote=True)}\">?</button>"
    return f"<div class='mf-card-head'>{badge}<div class='mf-card-titles'><div class='mf-card-title'>{title}</div><div class='mf-card-sub'>{subtitle}</div></div>{help_html}</div>"


def process_info_html(library, process_name: str, main_state: dict | None, user_refs: list[str] | None) -> str:
    """What the selected process does: its task, the model variant behind it and whether it is the task's recommended one."""
    process_name = str(process_name or "").strip()
    badges, summary = [], ""
    if process_name == ui_constants.NO_USER_SETTINGS_VALUE:
        task, summary = "custom", "No settings to add: save settings for the model selected in WanGP's generator first (they must use a Control Video), then pick them here."
    elif process_name in catalog.PROCESS_DEFINITIONS:
        info = catalog.PROCESS_DEFINITIONS[process_name]["info"]
        task, summary = info.get("task", "custom"), str(info.get("summary", ""))
        badges += ["<span class='mf-badge rec'>Recommended</span>"] if info.get("recommended") else []
        badges += ["<span class='mf-badge legacy'>Legacy</span>"] if info.get("legacy") else []
    else:
        task = "custom"
        definition = library.process_definition(process_name, main_state, user_refs)
        model = library.model_type_label(library.process_definition_model_type(definition)) if definition is not None else "the selected model"
        if catalog.is_user_process_value(process_name):
            badges.append("<span class='mf-badge user'>Linked</span>")
            summary = f"Your settings for {model}."
        else:
            summary = f"Settings saved for {model}. Click + to add them to Media Flow."
    name, description = TASKS.get(task, TASKS["custom"])
    return (f"<div class='mf-process' data-task='{html.escape(task, quote=True)}'><div class='mf-process-icon'>{icon(task if task in TASKS else 'custom')}</div>"
            f"<div class='mf-process-text'><div class='mf-process-top'><span class='mf-process-task'>{name}</span>{''.join(badges)}</div>"
            f"<p class='mf-process-desc'>{description}</p>{f'<p class=mf-process-summary>{html.escape(summary)}</p>' if summary else ''}</div></div>")
