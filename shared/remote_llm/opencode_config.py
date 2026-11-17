# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .config import ENGINE_OPENCODE, normalize_opencode_model_selection, normalize_opencode_provider_selection, normalize_reasoning_effort_selection, opencode_model_choices, opencode_provider_choices, opencode_reasoning_effort_choices
from .config_ui_common import cached_model_catalog, save_model_catalog
from .opencode_backend import OpenCodeBackend


@dataclass(slots=True)
class OpenCodeConfigUI:
    group: Any
    executable: Any
    base_url: Any
    provider: Any
    model: Any
    reasoning_effort: Any
    config: Any
    refresh: Any
    status: Any

    @property
    def save_components(self) -> list[Any]:
        return [self.executable, self.base_url, self.provider, self.model, self.reasoning_effort, self.config]


def create_opencode_config_ui(gr, profile: dict[str, Any], *, visible: bool, lock_config: bool) -> OpenCodeConfigUI:
    with gr.Group(visible=visible, elem_classes=["wangp-transparent-group"]) as group:
        gr.Markdown("**OpenCode setup** — install and authenticate providers in OpenCode outside WanGP. [Provider authentication help](https://opencode.ai/docs/providers/). Provider credentials remain in OpenCode; WanGP works with OpenCode 1 and OpenCode 2; the password of a server you start yourself is read at runtime from `OPENCODE_PASSWORD` (OpenCode 2) or `OPENCODE_SERVER_USERNAME` / `OPENCODE_SERVER_PASSWORD` (OpenCode 1) and is not saved. Refresh lists only providers currently configured in OpenCode. Custom configuration below is saved verbatim, so use OpenCode authentication or environment-variable references instead of putting secrets in it.")
        with gr.Row():
            executable = gr.Textbox(value=profile["executable"], label="OpenCode executable")
            base_url = gr.Textbox(value=profile["base_url"], label="OpenCode server URL")
        with gr.Row(elem_classes=["wangp-bottom-aligned-row"]):
            provider = gr.Dropdown(choices=opencode_provider_choices(profile["model_catalog"], profile["provider"]), value=profile["provider"], allow_custom_value=True, label="OpenCode provider")
            model = gr.Dropdown(choices=opencode_model_choices(profile["model_catalog"], profile["provider"], profile["model"]), value=profile["model"], allow_custom_value=True, label="OpenCode model")
            reasoning_effort = gr.Dropdown(choices=opencode_reasoning_effort_choices(profile["model_catalog"], profile["provider"], profile["model"], profile["reasoning_effort"]), value=profile["reasoning_effort"], allow_custom_value=True, label="Reasoning effort")
            refresh = gr.Button("Refresh", min_width=80, scale=0, interactive=not lock_config)
        status = gr.Markdown()
        config = gr.Textbox(value=profile["config"], label="OpenCode configuration (JSON / JSONC, optional)", lines=6, info="Applied through OPENCODE_CONFIG_CONTENT only when WanGP starts the local OpenCode server. Do not enter API keys or tokens here.")
    return OpenCodeConfigUI(group, executable, base_url, provider, model, reasoning_effort, config, refresh, status)


def bind_opencode_config_ui(gr, ui: OpenCodeConfigUI, server_config: dict[str, Any], server_config_filename: str) -> None:
    def refresh_catalog(executable, base_url, config, selected_provider, selected_model, selected_effort):
        selected_provider = normalize_opencode_provider_selection(selected_provider)
        selected_model = normalize_opencode_model_selection(selected_model)
        selected_effort = normalize_reasoning_effort_selection(selected_effort)
        backend = OpenCodeBackend({"executable": str(executable or "opencode").strip() or "opencode", "base_url": str(base_url or "http://127.0.0.1:4096").strip().rstrip("/"), "config": str(config or "").strip()})
        try:
            catalog = backend.list_models()
        except Exception as exc:
            return gr.update(), gr.update(), gr.update(), f"**Could not refresh OpenCode models:** {exc}"
        finally:
            backend.close()
        save_model_catalog(server_config, server_config_filename, ENGINE_OPENCODE, catalog)
        return gr.update(choices=opencode_provider_choices(catalog, selected_provider), value=selected_provider), gr.update(choices=opencode_model_choices(catalog, selected_provider, selected_model), value=selected_model), gr.update(choices=opencode_reasoning_effort_choices(catalog, selected_provider, selected_model, selected_effort), value=selected_effort), f"**Refreshed and cached {len(catalog)} OpenCode models.** The selected provider, model, and reasoning effort were not changed."

    def update_model_and_effort_choices(provider, selected_model, selected_effort):
        provider = normalize_opencode_provider_selection(provider)
        selected_model = normalize_opencode_model_selection(selected_model)
        selected_effort = normalize_reasoning_effort_selection(selected_effort)
        catalog = cached_model_catalog(server_config, ENGINE_OPENCODE)
        return gr.update(choices=opencode_model_choices(catalog, provider, selected_model), value=selected_model), gr.update(choices=opencode_reasoning_effort_choices(catalog, provider, selected_model, selected_effort), value=selected_effort)

    def update_reasoning_efforts(provider, model, selected_effort):
        provider = normalize_opencode_provider_selection(provider)
        model = normalize_opencode_model_selection(model)
        selected_effort = normalize_reasoning_effort_selection(selected_effort)
        catalog = cached_model_catalog(server_config, ENGINE_OPENCODE)
        return gr.update(choices=opencode_reasoning_effort_choices(catalog, provider, model, selected_effort), value=selected_effort)

    ui.provider.change(fn=update_model_and_effort_choices, inputs=[ui.provider, ui.model, ui.reasoning_effort], outputs=[ui.model, ui.reasoning_effort], show_progress="hidden")
    ui.model.change(fn=update_reasoning_efforts, inputs=[ui.provider, ui.model, ui.reasoning_effort], outputs=[ui.reasoning_effort], show_progress="hidden")
    ui.refresh.click(fn=refresh_catalog, inputs=[ui.executable, ui.base_url, ui.config, ui.provider, ui.model, ui.reasoning_effort], outputs=[ui.provider, ui.model, ui.reasoning_effort, ui.status], show_progress="hidden")


def opencode_profile_from_values(executable: Any, base_url: Any, provider: Any, model: Any, reasoning_effort: Any, config: Any, model_catalog: Any) -> dict[str, Any]:
    return {"executable": str(executable or "opencode").strip() or "opencode", "base_url": str(base_url or "http://127.0.0.1:4096").strip().rstrip("/"), "provider": normalize_opencode_provider_selection(provider), "model": normalize_opencode_model_selection(model), "reasoning_effort": normalize_reasoning_effort_selection(reasoning_effort), "config": str(config or "").strip(), "model_catalog": model_catalog}


def validate_opencode_profile(profile: dict[str, Any]) -> None:
    if not profile["executable"]:
        raise ValueError("OpenCode executable is required.")
    parsed = urlparse(profile["base_url"])
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OpenCode server URL must be an absolute http:// or https:// URL.")
