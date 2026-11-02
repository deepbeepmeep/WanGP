# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import html
import itertools
import re

import gradio as gr

from shared.gradio import model_infos


_COUNTER = itertools.count()
WAND_ICON_SVG = (
    '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M3 19.4 15.4 7 18 9.6 5.6 22 3 19.4Z"/>'
    '<path d="m15.4 7 2.1-2.1 2.6 2.6L18 9.6"/>'
    '<path d="M20 1v3m-1.5-1.5h3M6 3v2M5 4h2m14 11v2m-1-1h2"/></svg>'
)


SPATIAL_UPSAMPLER_HELP_INTRO = """Spatial upsamplers increase resolution. Visual refiners improve targeted content and may leave the dimensions unchanged. Scale is shown only for methods that support multipliers."""
TEMPORAL_UPSAMPLER_HELP_INTRO = """Temporal upsamplers increase frame rate by generating intermediate frames. WanGP multiplies the output frame rate by the selected factor so video duration and audio synchronization are preserved."""

SYSTEM_HELP = {
    "custom_sigmas": {
        "title": "Custom Sigmas",
        "markdown": """Leave this field empty to use the model's normal schedules.

- Separate sigma values with commas, and phases with a semicolon (`;`). Include one slot per selected phase; leave a slot blank to keep that phase's default.
- Values must be positive numbers **less than 1**, in strictly descending order. A final `0` is optional and ignored; the sampler always finishes at zero.
- Each positive value defines one denoising step. An explicit schedule overrides that phase's step count and starting-noise switch. Values are used directly, without another flow shift.

For two phases, `; 0.909375, 0.8571428571428571, 0.6666666666666666` keeps phase 1 automatic and sets three steps for phase 2.

`0.95, 0.7, 0.3;` customizes only phase 1. `0.95, 0.7, 0.3; 0.9, 0.6` customizes both phases. With one phase, omit the semicolon.

When a later phase has an explicit schedule, its first sigma takes precedence over that phase's switch. A notice appears if the values differ.

**Wan:** all phases form one descending schedule. For three phases, use two semicolons, for example `0.95,0.9; 0.8,0.6; 0.4,0.2`. The two switches become 800 and 400. Blank phases keep the normal schedule points within their switch boundaries; Steps controls that normal schedule. Explicit phases replace those points, so the final step count can change.

**H3 and LTX-2:** each phase has its own schedule; phase 2 may restart at a higher sigma than the end of phase 1. A scheduler with a fixed schedule may reject custom values. Clear this field to use its usual schedule.""",
    },
    "loras_multipliers": {
        "title": "LoRA Multipliers",
        "markdown": """Enter one multiplier per selected LoRA, in the same order. Separate LoRAs with spaces or new lines. Missing multipliers default to `1.0`. Lines beginning with `#` are comments.

- `0.8 1.2` applies 0.8 to the first LoRA and 1.2 to the second. `0` disables a LoRA.
- Use `;` **inside one multiplier**, without spaces, for phase-specific strengths: with two phases, `1;0` applies only in phase 1, `0;1` only in phase 2, and `1;0.6` uses 1 then 0.6. For three phases, use three values, for example `0;1;0`. A single value applies to every phase.
- Use `,` inside a phase for a sequence of strengths distributed across its steps, for example `1,0.8,0.6`. Do not put spaces inside that sequence.
- Models that expose separate LoRA branches also accept `:` between branch strengths, in the order given by that model's help. For example, `1:0.5` sets the two branches independently. Other models do not accept this syntax.
- The `|` separator divides accelerator-profile LoRAs on the left from additional LoRAs on the right. Preserve it when editing a profile. For example, `1;0 0;1 | 0.7` uses two accelerators in separate phases and an additional LoRA at 0.7 in both.

Use phase syntax only when the selected model supports multiple LoRA phases. H3's automatic phase-2 Turbo is replaced when an accelerator before `|` has a nonzero phase-2 multiplier.""",
    },
    "spatial_upsampling": {
        "title": "Spatial Upsampler / Visual Refiner",
        "markdown": SPATIAL_UPSAMPLER_HELP_INTRO,
    },
    "temporal_upsampling": {
        "title": "Temporal Upsampler",
        "markdown": TEMPORAL_UPSAMPLER_HELP_INTRO,
    },
}


def get_model_prompt_help(model_def):
    model_def = model_def or {}
    return model_def.get("prompt_infos", model_def.get("promt_infos", None))


def _normalize_classes(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _ensure_component_target(component):
    elem_id = getattr(component, "elem_id", None)
    if not elem_id:
        elem_id = f"wangp-field-help-target-{next(_COUNTER)}"
        component.elem_id = elem_id
    classes = _normalize_classes(getattr(component, "elem_classes", None))
    if "wangp-field-help-target" not in classes:
        classes.append("wangp-field-help-target")
        component.elem_classes = classes
    return elem_id


def _resolve_help(help_id, *, title=None, markdown=None):
    if markdown is not None:
        return title or "Help", str(markdown)
    help_def = SYSTEM_HELP.get(str(help_id or ""))
    if isinstance(help_def, dict):
        return title or help_def.get("title", "Help"), str(help_def.get("markdown", ""))
    return title or "Help", str(help_def or "")


def _tool_button(kind, popup_id, title, icon):
    popup_id = str(popup_id or "").strip()
    title = str(title or "")
    open_attr = f" data-wangp-model-info-open='{html.escape(popup_id, quote=True)}'" if popup_id else ""
    hidden_attr = "" if popup_id else " aria-hidden='true' tabindex='-1'"
    return (
        f"<button type='button' class='wangp-context-tool wangp-context-tool-{html.escape(kind, quote=True)}' "
        f"title='{html.escape(title, quote=True)}' aria-label='{html.escape(title, quote=True)}'{open_attr}{hidden_attr}>{icon}</button>"
    )


def render_marker(elem_id, help_id, *, title=None, markdown=None, helper_popup_id=None, helper_title=None, keep_empty=False):
    title, markdown = _resolve_help(help_id, title=title, markdown=markdown)
    has_markdown = bool(str(markdown or "").strip())
    has_helper = bool(str(helper_popup_id or "").strip())
    if not has_markdown and not has_helper and not keep_empty:
        return ""
    popup_html = ""
    popup_id = ""
    if has_markdown:
        popup_key = re.sub(r"[^A-Za-z0-9_-]", "-", f"{elem_id}-{help_id}").strip("-").lower()
        popup_id = f"wangp-field-help-{popup_key}"
        popup_html = model_infos.render_info_popup(popup_id, title, markdown, lazy=True)
    info_button = _tool_button("info", popup_id, title, "i")
    helper_button = _tool_button("helper", helper_popup_id if has_helper else "", helper_title or "Prompt Helper", WAND_ICON_SVG)
    return (
        f"<span class='wangp-field-help-inline' data-wangp-field-help-for='{html.escape(elem_id, quote=True)}'>"
        f"{info_button}{helper_button}"
        "</span>"
        f"{popup_html}"
    )


def bind(component, help_id, *, title=None, markdown=None):
    elem_id = _ensure_component_target(component)
    marker = render_marker(elem_id, help_id, title=title, markdown=markdown)
    if not marker:
        return gr.HTML("", visible=False)
    return gr.HTML(marker, elem_classes=["wangp-field-help-inline-host"])


def render_model_prompt_marker(elem_id, model_type, model_def, prompt_id, *, helper_popup_id=None, helper_title=None):
    return render_model_prompt_tools("", elem_id, model_type, model_def, prompt_id, helper_popup_id=helper_popup_id, helper_title=helper_title)


def render_model_prompt_tools(label, elem_id, model_type, model_def, prompt_id, *, helper_popup_id=None, helper_title=None):
    infos = get_model_prompt_help(model_def)
    if isinstance(infos, (list, tuple)) and len(infos) >= 2:
        title, markdown = str(infos[0] or "Prompt Guidelines"), str(infos[1] or "")
    else:
        title, markdown = "Prompt Guidelines", str(infos or "")
    popup_html = ""
    popup_id = ""
    if markdown.strip():
        popup_key = re.sub(r"[^A-Za-z0-9_-]", "-", f"{elem_id}-prompt-{model_type or 'model'}-{prompt_id or 'prompt'}").strip("-").lower()
        popup_id = f"wangp-field-help-{popup_key}"
        popup_html = model_infos.render_info_popup(popup_id, title, markdown, lazy=True)
    info_button = _tool_button("info", popup_id, title, "i")
    helper_button = _tool_button("helper", helper_popup_id, helper_title or "Prompt Helper", WAND_ICON_SVG)
    row_class = "wangp-prompt-tools-row" if popup_id or str(helper_popup_id or "").strip() else "wangp-prompt-tools-row wangp-prompt-tools-empty"
    return (
        f"<div class='{row_class}' data-wangp-prompt-tools-for='{html.escape(elem_id, quote=True)}'>"
        f"<span class='wangp-context-tools'>{info_button}{helper_button}</span>"
        "</div>"
        f"{popup_html}"
    )


def get_css():
    return """
.wangp-prompt-tools-stack {
    position: relative;
}
.wangp-prompt-tools-anchor {
    position: absolute !important;
    left: 0 !important;
    top: 0 !important;
    width: 0 !important;
    min-height: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    overflow: visible !important;
}
.wangp-prompt-tools-anchor > :not(.wangp-model-info-popup):not([data-wangp-model-info-popup]),
.wangp-prompt-tools-anchor .html-container {
    min-height: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    overflow: visible !important;
}
.wangp-prompt-tools-anchor .wangp-prompt-tools-row {
    display: none !important;
}
.wangp-prompt-tools-row {
    display: inline-flex !important;
    align-items: center;
    gap: 4px;
    margin: -2px 0 -2px 6px;
    vertical-align: middle;
    line-height: 0 !important;
}
/* Voice controls are mounted in this row even when the model has no help. */
.wangp-prompt-tools-empty:not(:has(.wangp-prompt-microphone:not([hidden]))) {
    display: none !important;
    margin: 0 !important;
}
.wangp-context-tools,
.wangp-field-help-inline {
    display: inline-flex !important;
    align-items: center;
    gap: 4px;
    vertical-align: middle;
    white-space: nowrap;
    margin-block: -2px !important;
    line-height: 0 !important;
}
.wangp-field-help-inline {
    margin-left: 6px !important;
}
.wangp-field-help-inline:not(:has([data-wangp-model-info-open])) {
    display: none !important;
}
.wangp-field-help-inline-host,
.wangp-field-help-inline-host > *,
.wangp-field-help-inline-host .html-container {
    position: absolute !important;
    left: 0 !important;
    top: 0 !important;
    width: 0 !important;
    min-height: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    overflow: visible !important;
}
.wangp-context-tool {
    position: static !important;
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    min-height: 18px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 1px solid var(--button-secondary-border-color, rgba(17, 84, 118, 0.24)) !important;
    border-radius: 999px !important;
    background: var(--button-secondary-background-fill, rgba(255, 255, 255, 0.86)) !important;
    color: var(--button-secondary-text-color, #155574) !important;
    box-shadow: none !important;
    cursor: pointer;
    vertical-align: middle;
    font-size: 11px !important;
    line-height: 1 !important;
    overflow: hidden !important;
    opacity: 1 !important;
    visibility: visible !important;
}
.wangp-context-tool:not([data-wangp-model-info-open]) {
    display: none !important;
}
.wangp-context-tool-info {
    width: 13px !important;
    height: 13px !important;
    min-width: 13px !important;
    min-height: 13px !important;
    border: 1px solid currentColor !important;
    background: transparent !important;
    color: #175a79 !important;
    font-family: Arial, sans-serif !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    font-style: normal !important;
}
.postprocess .wangp-context-tool-info {
    width: 10px !important;
    height: 10px !important;
    min-width: 10px !important;
    min-height: 10px !important;
    font-size: 8px !important;
}
.postprocess .wangp-field-help-inline,
.postprocess .wangp-context-tools {
    margin-block: 0 !important;
    padding-block: 2px !important;
    padding-inline-end: 2px !important;
}
.postprocess .wangp-prompt-tools-row {
    margin-block: 0 !important;
}
.wangp-context-tool-helper {
    width: 22px !important;
    height: 20px !important;
    min-width: 22px !important;
    min-height: 20px !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: #175a79 !important;
}
.wangp-context-tool-helper svg {
    width: 18px;
    height: 18px;
}
"""


def get_javascript():
    return """
    window.wangpPromptTools = window.wangpPromptTools || {};
    window.wangpPromptTools.attach = function(root) {
        const scope = root && root.querySelectorAll ? root : document;
        function movePopups(sourceRoot, buttons) {
            buttons.forEach((button) => {
                const popupId = button.getAttribute("data-wangp-model-info-open");
                if (!popupId) return;
                const popups = Array.from(document.querySelectorAll("#" + CSS.escape(popupId)));
                const popup = popups.find((candidate) => sourceRoot?.contains(candidate)) || popups[popups.length - 1];
                popups.forEach((candidate) => {
                    if (candidate !== popup) candidate.remove();
                });
                if (popup && popup.parentElement !== document.body) document.body.appendChild(popup);
            });
        }
        scope.querySelectorAll(".wangp-prompt-tools-row[data-wangp-prompt-tools-for]").forEach((row) => {
            if (!row.isConnected) return;
            const sourceRoot = row.parentElement;
            const targetId = row.getAttribute("data-wangp-prompt-tools-for") || "";
            const target = document.getElementById(targetId);
            const label = target?.querySelector('[data-testid="block-info"]');
            if (!label) return;
            movePopups(sourceRoot, row.querySelectorAll("[data-wangp-model-info-open]"));
            label.querySelectorAll(".wangp-prompt-tools-row[data-wangp-prompt-tools-for]").forEach((existing) => {
                if (existing !== row && existing.getAttribute("data-wangp-prompt-tools-for") === targetId) existing.remove();
            });
            if (row.isConnected && row.parentElement !== label) label.appendChild(row);
        });
        scope.querySelectorAll(".wangp-field-help-inline[data-wangp-field-help-for]").forEach((inline) => {
            if (!inline.isConnected) return;
            const sourceRoot = inline.parentElement;
            const targetId = inline.getAttribute("data-wangp-field-help-for") || "";
            const target = document.getElementById(targetId);
            const label = target?.querySelector('[data-testid="block-info"]');
            // Only mount incoming markers; an old mounted marker must not discard its replacement popup.
            if (!label || inline.parentElement === label) return;
            movePopups(sourceRoot, inline.querySelectorAll("[data-wangp-model-info-open]"));
            label.querySelectorAll(".wangp-field-help-inline[data-wangp-field-help-for]").forEach((existing) => {
                if (existing !== inline && existing.getAttribute("data-wangp-field-help-for") === targetId) existing.remove();
            });
            if (inline.isConnected && inline.parentElement !== label) label.appendChild(inline);
        });
    };
"""
