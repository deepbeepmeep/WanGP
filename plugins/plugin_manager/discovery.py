# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional, metadata-defined information cards for any Plugin."""
from html import escape
from functools import cache
from pathlib import Path
from urllib.parse import quote, urlsplit

import gradio as gr
from shared.gradio.app_icons import icon
from shared.utils.bundled_apps import FEATURED_APPS


def card_header(plugin, badges=""):
    name, pid = escape(plugin["name"], quote=True), escape(plugin["id"], quote=True)
    details = f'<button type="button" class="plugin-info-icon" data-app-info="{pid}" title="Details about {name}" aria-label="Details about {name}">{icon("info")}</button>' if plugin.get("info_card") else ""
    featured = '<span class="plugin-featured-badge">Featured</span>' if plugin.get("important") else ""
    return f'<div class="plugin-header"><div class="plugin-title-line"><span class="name">{name}</span>{badges}<span class="plugin-identity"><span>(id: {pid})</span>{details}</span></div>{featured}</div>'


def _image_url(plugin, image):
    if not image:
        return ""
    if urlsplit(image).scheme == "https":
        return image
    # Local images belong to the Plugin, or to WanGP's shared icons directory.
    base = Path("icons") if image.startswith("icons/") else Path(plugin["path"]) if plugin.get("path") else None
    if base is None:
        return ""
    base = base.resolve()
    path = (base / (image[6:] if image.startswith("icons/") else image)).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        return ""
    return _serve_image(path)


@cache
def _serve_image(path):
    gr.set_static_paths(paths=[path])
    return "gradio_api/file=" + quote(path.as_posix(), safe="/:")


def information_card(plugin):
    card = plugin.get("info_card")
    if not card:
        return ""
    pid = escape(plugin["id"], quote=True)
    name = escape(plugin["name"])
    image = _image_url(plugin, card.get("image", ""))
    visual = f'<img src="{escape(image, quote=True)}" alt="{escape(card.get("image_alt", plugin["name"]), quote=True)}" width="1200" height="750" loading="lazy">' if image else f'<div class="app-intro-visual">{icon(plugin["id"])}<span>{name}</span></div>'
    onboarding = ""
    if plugin["id"] in FEATURED_APPS:
        onboarding = '<div data-app-onboarding><div class="app-intro-actions"><button type="button" data-app-decision="keep">Keep It</button><button type="button" data-app-decision="remove">Remove It</button></div><p class="app-intro-detail">Keep it to open the app. You can enable it again from Apps &amp; PlugIn Manager at any time.</p></div>'
    return f"""<article class="app-intro" data-app-intro="{pid}" hidden>{visual}
        <div class="app-intro-body"><h2>{name}</h2>
        <p class="app-intro-tagline">{escape(card.get("tagline", ""))}</p>
        <p>{escape(card.get("description") or plugin.get("description", ""))}</p>
        <p class="app-intro-detail">{escape(card.get("details", ""))}</p>{onboarding}
        <div class="app-intro-actions" data-app-information><button type="button" data-app-close>Close</button></div></div></article>"""
