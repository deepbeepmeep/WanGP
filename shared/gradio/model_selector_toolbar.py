# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import html
import re
import time
from dataclasses import dataclass
from typing import Callable

import gradio as gr

from shared import model_dropdowns
from shared.deepy.config import deepy_available
from shared.utils.model_unload import model_unload_guard


MAX_SEARCH_RESULTS = 24
SHOW_SEARCH_RESULT_TYPE_LINE = False


@dataclass
class ModelSelectorToolbar:
    search_button: gr.Button
    refresh_button: gr.Button
    unload_button: gr.Button
    finetune_button: gr.Button | None = None
    tool_row: gr.Row | None = None
    search_row: gr.Row | None = None
    search_query: gr.Textbox | None = None
    search_results: gr.HTML | None = None
    search_target: gr.Textbox | None = None
    search_apply_button: gr.Button | None = None
    search_close_button: gr.Button | None = None


def create_toolbar(is_finetune_editor=False):
    with gr.Column(scale=2, min_width=210, elem_classes=["wangp-model-selector-tools"]):
        with gr.Row(elem_classes=["wangp-model-selector-tool-row"]) as tool_row:
            search_button = gr.Button("⌕", elem_id="wangp_model_tool_search", elem_classes=["wangp-model-selector-tool", "wangp-model-selector-tool-search"], size="sm", scale=0)
            finetune_button = gr.Button("✎" if is_finetune_editor else "+", elem_id="wangp_model_tool_finetune", elem_classes=["wangp-model-selector-tool", "wangp-model-selector-tool-finetune"], size="sm", scale=0)
            refresh_button = gr.Button("↻", elem_id="wangp_model_tool_refresh", elem_classes=["wangp-model-selector-tool", "wangp-model-selector-tool-refresh"], size="sm", scale=0)
            unload_button = gr.Button("Unload models and extensions", elem_id="wangp_model_tool_unload", elem_classes=["wangp-model-selector-tool", "wangp-model-selector-tool-unload"], size="sm", scale=0)
        with gr.Row(visible=False, elem_classes=["wangp-model-selector-search-row"]) as search_row:
            with gr.Column(scale=1, min_width=0, elem_classes=["wangp-model-selector-search-box"]):
                search_query = gr.Textbox(value="", show_label=False, placeholder="Search models", elem_id="wangp_model_search_query", elem_classes=["wangp-model-selector-search-input"])
                search_results = gr.HTML(value="", visible=False, elem_id="wangp_model_search_results")
    return ModelSelectorToolbar(search_button, refresh_button, unload_button, finetune_button=finetune_button, tool_row=tool_row, search_row=search_row, search_query=search_query, search_results=search_results)


def create_search_panel(toolbar: ModelSelectorToolbar):
    with gr.Row(visible=False, elem_classes=["wangp-model-selector-hidden-controls"]):
        toolbar.search_target = gr.Textbox(value="", show_label=False, elem_id="wangp_model_search_target")
        toolbar.search_apply_button = gr.Button("Apply model search", elem_id="wangp_model_search_apply")
        toolbar.search_close_button = gr.Button("Close model search", elem_id="wangp_model_search_close")
    return toolbar


def show_search_panel():
    return gr.update(visible=False), gr.update(visible=True), gr.update(value=""), gr.update(value="", visible=False)


def clear_search_panel():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=""), gr.update(value="", visible=False)


def _normalize_search_text(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _display_join(*parts):
    text = " ".join(str(part or "").strip() for part in parts if str(part or "").strip())
    return re.sub(r"\s+", " ", text).strip()


def _family_model_types(deps, dropdown_types, family):
    return [model_type for model_type in dropdown_types if deps.get_model_family(model_type, for_ui=True) == family]


def _default_model_for_family(deps, state, dropdown_types, family):
    family_name = deps.families_infos[family][1]
    rows = sorted([(model_dropdowns.compact_name(family_name, deps.get_model_name(model_type)), model_type) for model_type in _family_model_types(deps, dropdown_types, family)], key=lambda row: row[0].casefold())
    values = [model_type for _label, model_type in rows]
    model_type = (state or {}).get("last_model_per_family", {}).get(family, "")
    return model_type if model_type in values else (values[0] if values else "")


def _family_hierarchy(deps, dropdown_types, family):
    family_name = deps.families_infos[family][1]
    rows = [
        (model_dropdowns.compact_name(family_name, deps.get_model_name(model_type)), model_type, deps.get_parent_model_type(model_type))
        for model_type in _family_model_types(deps, dropdown_types, family)
    ]
    rows.sort(key=lambda row: row[0].casefold())
    return model_dropdowns.create_models_hierarchy(rows)


def _default_model_for_parent(deps, state, children_by_parent, parent_model_type):
    children = children_by_parent.get(parent_model_type, [])
    values = [model_type for _label, model_type in children]
    model_type = (state or {}).get("last_model_per_type", {}).get(parent_model_type, "")
    return model_type if model_type in values else (values[0] if values else "")


def _append_result(results, seen_targets, scope, label, model_type, path):
    if not model_type or model_type in seen_targets:
        return
    seen_targets.add(model_type)
    results.append({"scope": scope, "label": label, "model_type": model_type, "path": path})


def _search_results(deps, state, query):
    needle = _normalize_search_text(query)
    if len(needle) == 0:
        return []
    dropdown_types = model_dropdowns.get_dropdown_model_types(deps)
    family_ids = sorted({deps.get_model_family(model_type, for_ui=True) for model_type in dropdown_types}, key=lambda family: deps.families_infos.get(family, (999, family))[0])
    results, seen_targets = [], set()
    for family in family_ids:
        if family not in deps.families_infos:
            continue
        family_name = deps.families_infos[family][1]
        if needle in _normalize_search_text(family_name):
            _append_result(results, seen_targets, "Family", family_name, _default_model_for_family(deps, state, dropdown_types, family), family_name)
            continue
        parent_choices, children_by_parent = _family_hierarchy(deps, dropdown_types, family)
        for parent_name, parent_model_type in parent_choices:
            parent_label = _display_join(family_name, parent_name)
            if needle in _normalize_search_text(parent_label) or needle in _normalize_search_text(deps.get_model_name(parent_model_type)):
                _append_result(results, seen_targets, "Model", parent_label, _default_model_for_parent(deps, state, children_by_parent, parent_model_type), family_name)
                continue
            for child_name, child_model_type in children_by_parent.get(parent_model_type, []):
                full_name = deps.get_model_name(child_model_type)
                child_label = full_name if needle in _normalize_search_text(full_name) else _display_join(parent_label, child_name)
                if needle in _normalize_search_text(child_label) or needle in _normalize_search_text(full_name):
                    _append_result(results, seen_targets, "Finetune", child_label, child_model_type, _display_join(family_name, parent_name))
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return results
    return results


def render_search_results(deps, state, query):
    query = str(query or "")
    if len(query.strip()) == 0:
        return gr.update(value="", visible=False)
    results = _search_results(deps, state, query)
    if len(results) == 0:
        return gr.update(value="<div class='wangp-model-search-popup'><div class='wangp-model-search-empty'>No matching models</div></div>", visible=True)
    items = []
    for index, result in enumerate(results):
        type_line = "<span class='wangp-model-search-result-meta'>{scope} · {path}</span>".format(scope=html.escape(result["scope"]), path=html.escape(result["path"])) if SHOW_SEARCH_RESULT_TYPE_LINE else ""
        items.append(
            "<button type='button' class='wangp-model-search-result' data-model-type='{model_type}' role='option'>"
            "<span class='wangp-model-search-result-title'>{label}</span>"
            "{type_line}"
            "</button>".format(
                model_type=html.escape(result["model_type"], quote=True),
                label=html.escape(result["label"]),
                type_line=type_line,
            )
        )
    return gr.update(value="<div class='wangp-model-search-popup' role='listbox'>" + "".join(items) + "</div>", visible=True)


def apply_search_selection(model_type):
    model_type = str(model_type or "").strip()
    return (f"{model_type}|{time.time()}" if model_type else gr.update()), *clear_search_panel()


def _prune_orphan_model_settings(state, deps):
    all_settings = (state or {}).get("all_settings", None)
    if not isinstance(all_settings, dict):
        return 0
    orphan_model_types = [model_type for model_type in all_settings if deps.get_model_def(model_type) is None]
    for model_type in orphan_model_types:
        all_settings.pop(model_type, None)
    return len(orphan_model_types)


def refresh_models_with_info(refresh_model_defs, refresh_model_dropdowns, state, deps_factory):
    try:
        parse_errors = refresh_model_defs() or []
    except Exception as e:
        gr.Info(f"Unable to refresh model list: {e}")
        return refresh_model_dropdowns(state)
    pruned_count = _prune_orphan_model_settings(state, deps_factory())
    prune_text = f" Removed {pruned_count} orphan model setting{'s' if pruned_count > 1 else ''}." if pruned_count > 0 else ""
    if len(parse_errors) > 0:
        gr.Info("Model list refreshed, but parsing errors were found: " + parse_errors[0] + prune_text)
    else:
        gr.Info("Model list refreshed." + prune_text)
    return refresh_model_dropdowns(state)


def unload_models_from_ram(state, *, server_config, any_GPU_process_running, release_deepy_vram, reset_prompt_enhancer, reset_prompt_enhancer_if_requested, release_extensions, release_model):
    with model_unload_guard():
        unload_targets = _unload_targets_text(server_config)
        if any_GPU_process_running(state, "configuration"):
            gr.Info(f"Unable to unload {unload_targets} while GPU resources are allocated.")
            return
        if deepy_available(server_config):
            release_deepy_vram(state, clear_session_state=False, discard_runtime_snapshot=True)
        if "Prompt Enhancer" in unload_targets:
            reset_prompt_enhancer()
            reset_prompt_enhancer_if_requested()
        release_extensions()
        release_model()
    gr.Info(f"{unload_targets} unloaded from RAM.")


def _unload_targets_text(server_config):
    from shared.utils.offload_registry import registered_names

    targets = ["Models"]
    try:
        enhancer_enabled = int(server_config.get("enhancer_enabled", 0) or 0) > 0
    except Exception:
        enhancer_enabled = False
    if enhancer_enabled:
        targets.append("Prompt Enhancer")
    targets.extend(registered_names())
    if deepy_available(server_config):
        targets.append("Deepy")
    if len(targets) == 1:
        return targets[0]
    return ", ".join(targets[:-1]) + f", and {targets[-1]}" if len(targets) > 2 else " and ".join(targets)


def bind_toolbar(toolbar: ModelSelectorToolbar, *, deps_factory: Callable, state, model_family, model_base_type_choice, model_choice, model_choice_target, refresh_form_trigger, lset_name, loras_choices, refresh_model_defs: Callable, refresh_model_dropdowns: Callable, refresh_lora_list: Callable, unload_handler: Callable):
    toolbar.search_button.click(
        fn=show_search_panel,
        outputs=[toolbar.tool_row, toolbar.search_row, toolbar.search_query, toolbar.search_results],
        show_progress="hidden",
    ).then(fn=None, js=focus_search_javascript(), inputs=None, outputs=None)
    toolbar.search_query.input(
        fn=lambda state_value, query: render_search_results(deps_factory(), state_value, query),
        inputs=[state, toolbar.search_query],
        outputs=[toolbar.search_results],
        show_progress="hidden",
    )
    toolbar.search_apply_button.click(
        fn=apply_search_selection,
        inputs=[toolbar.search_target],
        outputs=[model_choice_target, toolbar.tool_row, toolbar.search_row, toolbar.search_query, toolbar.search_results],
        show_progress="hidden",
    )
    toolbar.search_close_button.click(
        fn=clear_search_panel,
        outputs=[toolbar.tool_row, toolbar.search_row, toolbar.search_query, toolbar.search_results],
        show_progress="hidden",
    )
    toolbar.refresh_button.click(
        fn=lambda state_value: refresh_models_with_info(refresh_model_defs, refresh_model_dropdowns, state_value, deps_factory),
        inputs=[state],
        outputs=[model_family, model_base_type_choice, model_choice, refresh_form_trigger],
        show_progress="hidden",
    ).then(
        fn=refresh_lora_list,
        inputs=[state, lset_name, loras_choices],
        outputs=[lset_name, loras_choices],
        show_progress="hidden",
    )
    toolbar.unload_button.click(fn=unload_handler, inputs=[state], outputs=None, show_progress="hidden")


def focus_search_javascript():
    return """
    () => {
        const root = window.gradioApp ? window.gradioApp() : (document.querySelector("gradio-app")?.shadowRoot || document);
        setTimeout(() => {
            const input = root.querySelector("#wangp_model_search_query textarea, #wangp_model_search_query input");
            if (input) input.focus();
        }, 50);
    }
    """


def get_javascript():
    return r"""
    (function () {
        let searchResultPointerDown = false;
        let activeSearchIndex = -1;

        function root() {
            if (window.gradioApp) return window.gradioApp();
            const app = document.querySelector("gradio-app");
            return app ? (app.shadowRoot || app) : document;
        }

        function queryInput() {
            return root().querySelector("#wangp_model_search_query textarea, #wangp_model_search_query input");
        }

        function targetInput() {
            return root().querySelector("#wangp_model_search_target textarea, #wangp_model_search_target input");
        }

        function applyButton() {
            const el = root().querySelector("#wangp_model_search_apply");
            return el?.matches("button") ? el : el?.querySelector("button");
        }

        function closeButton() {
            const el = root().querySelector("#wangp_model_search_close");
            return el?.matches("button") ? el : el?.querySelector("button");
        }

        function toolButton(id) {
            const el = root().querySelector(id);
            return el?.matches("button") ? el : el?.querySelector("button");
        }

        function updateFinetuneTooltip() {
            const button = toolButton("#wangp_model_tool_finetune");
            if (!button) return;
            const text = (button.textContent || "").trim();
            button.dataset.wangpTooltip = text.includes("✎") ? "Edit finetune [Alt+F]" : "Create finetune [Alt+F]";
        }

        function resultItems() {
            return Array.from(root().querySelectorAll("#wangp_model_search_results [data-model-type]"));
        }

        function setActive(index) {
            const items = resultItems();
            if (!items.length) return;
            const bounded = Math.max(0, Math.min(index, items.length - 1));
            activeSearchIndex = bounded;
            items.forEach((item, itemIndex) => item.classList.toggle("wangp-model-search-result-active", itemIndex === bounded));
            items[bounded].scrollIntoView({ block: "nearest" });
        }

        function activeIndex() {
            const items = resultItems();
            if (activeSearchIndex >= 0 && activeSearchIndex < items.length) return activeSearchIndex;
            const index = items.findIndex((item) => item.classList.contains("wangp-model-search-result-active"));
            return index;
        }

        function selectModel(modelType) {
            const target = targetInput();
            const button = applyButton();
            if (!target || !button || !modelType) return;
            target.value = modelType;
            target.dispatchEvent(new Event("input", { bubbles: true }));
            button.click();
        }

        function eventSearchItem(event) {
            const path = event.composedPath ? event.composedPath() : [];
            for (const node of path) {
                if (node?.matches?.("#wangp_model_search_results [data-model-type]")) return node;
                const closest = node?.closest?.("#wangp_model_search_results [data-model-type]");
                if (closest) return closest;
            }
            return event.target?.closest?.("#wangp_model_search_results [data-model-type]");
        }

        document.addEventListener("click", (event) => {
            const item = eventSearchItem(event);
            if (!item) return;
            event.preventDefault();
            selectModel(item.dataset.modelType || "");
        });

        document.addEventListener("pointerdown", (event) => {
            if (!eventSearchItem(event)) return;
            searchResultPointerDown = true;
            setTimeout(() => searchResultPointerDown = false, 250);
        });

        function bindSearchKeyboard() {
            const input = queryInput();
            if (!input || input.dataset.wangpModelSearchBound === "1") return;
            input.dataset.wangpModelSearchBound = "1";
            input.addEventListener("input", () => activeSearchIndex = -1);
            input.addEventListener("keydown", (event) => {
                if (event.key === "Escape") {
                    event.preventDefault();
                    closeButton()?.click();
                    return;
                }
                const items = resultItems();
                if (!items.length) return;
                if (event.key === "ArrowDown" || event.key === "Down") {
                    event.preventDefault();
                    const index = activeIndex();
                    setActive(index < 0 ? 0 : index + 1);
                } else if (event.key === "ArrowUp" || event.key === "Up") {
                    event.preventDefault();
                    const index = activeIndex();
                    setActive(index < 0 ? items.length - 1 : index - 1);
                } else if (event.key === "Enter") {
                    event.preventDefault();
                    const index = activeIndex();
                    const item = items[index >= 0 ? index : 0] || items[0];
                    selectModel(item?.dataset?.modelType || "");
                }
            });
            input.addEventListener("blur", () => {
                setTimeout(() => {
                    if (searchResultPointerDown) return;
                    const active = root().activeElement || document.activeElement;
                    if (active === queryInput()) return;
                    closeButton()?.click();
                }, 120);
            });
        }

        document.addEventListener("keydown", (event) => {
            if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.repeat) return;
            const key = event.key.toLowerCase();
            const target = key === "s" ? "#wangp_model_tool_search" : key === "r" ? "#wangp_model_tool_refresh" : key === "u" ? "#wangp_model_tool_unload" : "";
            if (key === "f") {
                event.preventDefault();
                toolButton("#wangp_model_tool_finetune")?.click();
                return;
            }
            if (!target) return;
            event.preventDefault();
            toolButton(target)?.click();
        });

        document.addEventListener("focusin", () => { bindSearchKeyboard(); updateFinetuneTooltip(); });
        document.addEventListener("pointerover", (event) => {
            if (event.target.closest?.("#wangp_model_tool_finetune")) updateFinetuneTooltip();
        });
        bindSearchKeyboard();
        updateFinetuneTooltip();
    })();
    """


def get_css():
    return """
    .wangp-model-selector-tools {
        position: relative;
        align-self: center;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .wangp-model-selector-tool-row {
        position: relative;
        z-index: 2;
        width: max-content;
        margin: 0 auto !important;
        gap: 6px !important;
        padding: 0 4px;
        background: var(--body-background-fill);
    }
    .wangp-model-selector-tool-row:has(.wangp-model-selector-tool:hover) {
        z-index: 61;
    }
    .wangp-model-selector-tool {
        position: relative;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 42px !important;
        width: 42px !important;
        max-width: 42px !important;
        height: 32px !important;
        padding: 0 !important;
        font-size: 0 !important;
        line-height: 1 !important;
        border-radius: 6px !important;
        color: var(--button-secondary-text-color, var(--body-text-color)) !important;
    }
    .wangp-model-selector-tool::before {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--button-secondary-text-color, var(--body-text-color));
        font-family: "Segoe UI Symbol", "Arial Unicode MS", sans-serif;
        font-size: 32px;
        line-height: 1;
    }
    .wangp-model-selector-tool-search::before {
        content: "⌕";
        font-size: 22px;
        transform: translateY(-1px);
    }
    .wangp-model-selector-tool-refresh::before {
        content: "↻";
        font-size: 23px;
        transform: translateY(-1px);
    }
    .wangp-model-selector-tool-unload::before {
        content: "";
        inset: 0;
        margin: auto;
        width: 22px;
        height: 22px;
        background: currentColor;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 3 3 15h18L12 3ZM3 18h18v3H3z'/%3E%3C/svg%3E") center / contain no-repeat;
    }
    .wangp-model-selector-tool-finetune {
        font-size: 24px !important;
        font-family: "Segoe UI Symbol", "Arial Unicode MS", sans-serif !important;
        transform: translateY(-1px);
    }
    .wangp-model-selector-tool-finetune::before {
        content: "";
    }
    .wangp-model-selector-tool::after {
        position: absolute;
        left: 50%;
        bottom: calc(100% + 8px);
        transform: translateX(-50%);
        white-space: nowrap;
        padding: 4px 7px;
        border-radius: 4px;
        background: var(--body-text-color);
        color: var(--body-background-fill);
        font-size: 12px;
        line-height: 1.2;
        opacity: 0;
        pointer-events: none;
        transition: opacity 120ms ease;
        transition-delay: 0s;
        z-index: 50;
    }
    .wangp-model-selector-tool:hover::after {
        opacity: 1;
        transition-delay: 1000ms;
    }
    .wangp-model-selector-tool-search::after { content: "Search models [Alt+S]"; }
    .wangp-model-selector-tool-refresh::after { content: "Refresh model list [Alt+R]"; }
    .wangp-model-selector-tool-finetune::after { content: attr(data-wangp-tooltip); }
    .wangp-model-selector-tool-unload::after { content: "Unload models and extensions [Alt+U]"; left: auto; right: 0; transform: none; }
    .wangp-model-selector-search-row {
        --wangp-model-selector-gap: calc(16px * var(--wangp-ui-scale, 0.9));
        --wangp-model-selector-search-margin: 8px;
        position: relative;
        z-index: 3;
        box-sizing: border-box;
        width: calc(100% + var(--wangp-model-selector-gap) - var(--wangp-model-selector-search-margin));
        margin: 0 0 0 calc(var(--wangp-model-selector-search-margin) - var(--wangp-model-selector-gap)) !important;
        padding: 0;
        background: var(--body-background-fill);
    }
    .wangp-model-selector-search-box {
        position: relative;
        width: 100%;
        min-width: 0 !important;
    }
    .wangp-model-selector-search-input textarea,
    .wangp-model-selector-search-input input {
        text-align: left;
        min-height: 27px !important;
        height: 27px !important;
        padding: 2px 7px !important;
        font-size: 13px !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    .wangp-model-selector-search-input {
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 5px !important;
        background: var(--input-background-fill, var(--body-background-fill)) !important;
        box-sizing: border-box;
        width: 100%;
        padding: 1px !important;
        margin: 0 !important;
    }
    .wangp-model-selector-search-input label,
    .wangp-model-selector-search-input > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    #wangp_model_search_results {
        position: absolute !important;
        left: auto;
        right: 0;
        top: calc(100% - 1px);
        width: min(560px, 90vw);
        z-index: 1000;
    }
    .wangp-model-search-popup {
        margin-top: 0;
        border: 1px solid var(--border-color-primary);
        border-radius: 0 0 6px 6px;
        background: var(--body-background-fill);
        overflow: hidden;
        max-height: 280px;
        overflow-y: auto;
    }
    .wangp-model-search-result {
        width: 100%;
        border: 0;
        border-bottom: """ + ("1px solid var(--border-color-primary)" if SHOW_SEARCH_RESULT_TYPE_LINE else "0") + """;
        background: transparent;
        color: var(--body-text-color);
        display: block;
        text-align: left;
        padding: 8px 10px;
        cursor: pointer;
    }
    .wangp-model-search-result:last-child { border-bottom: 0; }
    .wangp-model-search-result:hover,
    .wangp-model-search-result-active {
        background: var(--background-fill-secondary);
        background: color-mix(in srgb, var(--button-primary-background-fill) 18%, var(--body-background-fill));
        box-shadow: inset 3px 0 0 var(--button-primary-background-fill);
        color: var(--body-text-color);
    }
    .wangp-model-search-result:hover .wangp-model-search-result-title,
    .wangp-model-search-result-active .wangp-model-search-result-title {
        font-weight: 700;
    }
    .wangp-model-search-result-title,
    .wangp-model-search-result-meta {
        display: block;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .wangp-model-search-result-title {
        font-weight: 600;
        font-size: 13px;
    }
    .wangp-model-search-result-meta {
        opacity: 0.75;
        font-size: 11px;
        margin-top: 2px;
    }
    .wangp-model-search-empty {
        padding: 8px 10px;
        font-size: 12px;
        opacity: 0.75;
    }
    .wangp-model-selector-hidden-controls {
        display: none !important;
    }
    """
