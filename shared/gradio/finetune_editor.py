# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gradio as gr

from shared.gradio.local_file_picker import CHECKPOINT_FILE_EXTENSIONS, LocalFilePickerTextbox
from shared.prompt_enhancer import chaining as prompt_enhancer_chaining
from shared.prompt_enhancer import labels as prompt_enhancer_labels
from shared import resolutions as resolution_utils
from shared.utils import files_locator as fl


FINETUNES_DIR = "finetunes"
FINETUNE_URL_FIELDS = ("URLs", "URLs2", "text_encoder_URLs")
FINETUNE_LORA_FIELDS = ("loras", "loras_multipliers")
FINETUNE_SOURCE_MODEL_KEY = "finetune_source_model"
LORA_FILE_EXTENSIONS = {".safetensors", ".sft"}
MAX_CUSTOM_URL_FIELDS = 3
MAX_PROMPT_ENHANCER_SYSTEMS = 3
MAX_FINETUNE_PARAMS = 3


@dataclass
class FinetuneEditorDeps:
    settings_version: float
    families_infos: dict
    transformer_types: list
    displayed_model_types: list
    get_model_def: Callable
    get_model_name: Callable
    get_base_model_type: Callable
    get_parent_model_type: Callable
    get_model_family: Callable
    get_state_model_type: Callable
    get_model_settings: Callable
    set_model_settings: Callable
    get_default_settings: Callable
    get_settings_file_name: Callable
    get_lora_dir: Callable
    get_lora_URL: Callable
    update_loras_url_cache: Callable
    refresh_model_defs: Callable
    refresh_model_dropdowns: Callable
    change_model: Callable
    request_reload_if_loaded: Callable


@dataclass
class FinetuneEditorUI:
    popup: gr.Column
    title: gr.HTML
    mode: gr.Textbox
    source_model_type: gr.Textbox
    original_id: gr.Textbox
    creator_source_mode: gr.Radio
    form_fields: gr.Column
    import_file: gr.File
    finetunes_info: gr.Markdown
    source_info: gr.Textbox
    id_text: gr.Textbox
    auto_id: gr.Checkbox
    name_text: gr.Textbox
    description_text: gr.Textbox
    urls_group: gr.Column
    urls_text: gr.Textbox
    urls2_group: gr.Column
    urls2_text: gr.Textbox
    text_encoder_group: gr.Column
    text_encoder_text: gr.Textbox
    custom_url_1_group: gr.Column
    custom_url_1_text: gr.Textbox
    custom_url_2_group: gr.Column
    custom_url_2_text: gr.Textbox
    custom_url_3_group: gr.Column
    custom_url_3_text: gr.Textbox
    loras_group: gr.Column
    loras_root_text: gr.Textbox
    loras_text: gr.Textbox
    loras_multipliers_text: gr.Textbox
    lock_resolutions_editor: gr.Checkbox
    resolutions_editor: gr.Textbox
    resolution_budget_editor: gr.Textbox
    resolution_budget_tiled_editor: gr.Textbox
    resolution_min_editor: gr.Textbox
    infos_editor: gr.Textbox
    prompt_infos_editor: gr.Textbox
    finetune_params_tab: gr.Tab
    finetune_param_1_group: gr.Column
    finetune_param_1_dropdown: gr.Dropdown
    finetune_param_1_help: gr.Markdown
    finetune_param_2_group: gr.Column
    finetune_param_2_dropdown: gr.Dropdown
    finetune_param_2_help: gr.Markdown
    finetune_param_3_group: gr.Column
    finetune_param_3_dropdown: gr.Dropdown
    finetune_param_3_help: gr.Markdown
    enhancer_system_1_group: gr.Column
    enhancer_system_1_editor: gr.Textbox
    enhancer_system_1_default: gr.State
    enhancer_system_1_default_button: gr.Button
    enhancer_system_1_tokens: gr.Textbox
    enhancer_system_2_group: gr.Column
    enhancer_system_2_editor: gr.Textbox
    enhancer_system_2_default: gr.State
    enhancer_system_2_default_button: gr.Button
    enhancer_system_2_tokens: gr.Textbox
    enhancer_system_3_group: gr.Column
    enhancer_system_3_editor: gr.Textbox
    enhancer_system_3_default: gr.State
    enhancer_system_3_default_button: gr.Button
    enhancer_system_3_tokens: gr.Textbox
    use_current_settings: gr.Checkbox
    creator_actions: gr.Row
    editor_actions: gr.Row
    create_button: gr.Button
    create_new_button: gr.Button
    cancel_button: gr.Button
    save_button: gr.Button
    export_button: gr.DownloadButton
    delete_button: gr.Button
    close_button: gr.Button
    delete_confirm: gr.Row
    confirm_delete_button: gr.Button
    cancel_delete_button: gr.Button
    create_apply_trigger: gr.Textbox
    create_save_trigger: gr.Textbox
    create_new_apply_trigger: gr.Textbox
    create_new_save_trigger: gr.Textbox
    save_apply_trigger: gr.Textbox
    save_save_trigger: gr.Textbox


def is_finetune_model_def(model_def: dict | None) -> bool:
    path = str((model_def or {}).get("path", "") or "")
    return Path(path).parent.name.casefold() == Path(FINETUNES_DIR).name.casefold()


def is_finetune_model(deps: FinetuneEditorDeps, model_type: str | None) -> bool:
    return is_finetune_model_def(deps.get_model_def(model_type))


def create_editor() -> FinetuneEditorUI:
    with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-popup"]) as popup:
        with gr.Column(elem_classes=["wangp-model-info-card", "wangp-finetune-editor-card"]):
            with gr.Row(elem_classes=["chat__template-modal-titlebar", "wangp-finetune-editor-titlebar"]):
                title = gr.HTML("<div class='chat__template-modal-heading'>Finetune Creator</div>")
                close_button = gr.Button("x", elem_id="wangp_finetune_editor_close", elem_classes=["wangp-model-info-close"], min_width=1, scale=0)
            with gr.Column(elem_classes=["wangp-finetune-editor-content"]):
                gr.HTML(
                    "<div class='wangp-finetune-editor-intro'>A finetune is a lightweight model definition derived from an existing WanGP model. "
                    "It inherits the source model properties, then lets you override its identity, description, weights, text encoder, and default settings.</div>",
                    padding=False,
                    container=True,
                    elem_classes=["wangp-finetune-editor-intro-host"],
                )
                mode = gr.Textbox(value="creator", visible=False)
                source_model_type = gr.Textbox(value="", visible=False)
                original_id = gr.Textbox(value="", visible=False)
                create_apply_trigger = gr.Textbox(value="", visible=False)
                create_save_trigger = gr.Textbox(value="", visible=False)
                create_new_apply_trigger = gr.Textbox(value="", visible=False)
                create_new_save_trigger = gr.Textbox(value="", visible=False)
                save_apply_trigger = gr.Textbox(value="", visible=False)
                save_save_trigger = gr.Textbox(value="", visible=False)
                creator_source_mode = gr.Radio(label="Create New Finetune", choices=[("Using Current Model", "current"), ("By importing a File", "import")], value="current", visible=False)
                with gr.Column(elem_classes=["wangp-finetune-editor-fields"]) as form_fields:
                    finetunes_info = gr.Markdown(value="", visible=False, elem_classes=["wangp-finetune-source-info"])
                    source_info = gr.Textbox(label="Source Model", value="", lines=1, max_lines=1, autoscroll=False, interactive=False, elem_classes=["wangp-finetune-editor-readonly"])
                    with gr.Row(elem_classes=["wangp-finetune-editor-id-row"]):
                        id_text = gr.Textbox(label="Id", value="", scale=7)
                        auto_id = gr.Checkbox(label="auto", value=True, scale=1, min_width=80, elem_classes="cbx_bottom")
                    name_text = gr.Textbox(label="Name", value="")
                    description_text = gr.Textbox(label="Description", value="", lines=3)
                    with gr.Tabs(elem_classes=["wangp-finetune-editor-tabs"]):
                        with gr.Tab("URLs"):
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as urls_group:
                                urls_text = LocalFilePickerTextbox(label="Main Checkpoints", file_extensions=CHECKPOINT_FILE_EXTENSIONS, multiselect=True, popup_title="Select Local Checkpoint Files").mount()
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as urls2_group:
                                urls2_text = LocalFilePickerTextbox(label="Secondary Checkpoints", file_extensions=CHECKPOINT_FILE_EXTENSIONS, multiselect=True, popup_title="Select Local Checkpoint Files").mount()
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as text_encoder_group:
                                text_encoder_text = LocalFilePickerTextbox(label="Text Encoder Checkpoints", file_extensions=CHECKPOINT_FILE_EXTENSIONS, multiselect=True, popup_title="Select Local Text Encoder Checkpoints").mount()
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as custom_url_1_group:
                                custom_url_1_text = LocalFilePickerTextbox(label="custom_url_1", file_extensions=CHECKPOINT_FILE_EXTENSIONS, multiselect=False, popup_title="Select Local Checkpoint File").mount()
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as custom_url_2_group:
                                custom_url_2_text = LocalFilePickerTextbox(label="custom_url_2", file_extensions=CHECKPOINT_FILE_EXTENSIONS, multiselect=False, popup_title="Select Local Checkpoint File").mount()
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as custom_url_3_group:
                                custom_url_3_text = LocalFilePickerTextbox(label="custom_url_3", file_extensions=CHECKPOINT_FILE_EXTENSIONS, multiselect=False, popup_title="Select Local Checkpoint File").mount()
                        with gr.Tab("LoRAs"):
                            with gr.Column(visible=True, elem_classes=["wangp-finetune-editor-field-group"]) as loras_group:
                                loras_root_text = gr.Textbox(value="", visible=False)
                                loras_text = LocalFilePickerTextbox(label="Always Loaded LoRAs", file_extensions=LORA_FILE_EXTENSIONS, multiselect=True, popup_title="Select LoRA Files", default_dir_input=loras_root_text, compress_root_input=loras_root_text).mount()
                                loras_multipliers_text = gr.Textbox(label="LoRAs Multipliers", value="", lines=1, max_lines=4)
                        with gr.Tab("Resolutions"):
                            with gr.Row():
                                resolution_budget_editor = gr.Textbox(label="Maximum Resolution", value="", placeholder="4096x4096 or 16777216", lines=1)
                                resolution_min_editor = gr.Textbox(label="Minimum Resolution", value="", placeholder="512x512 or 262144", lines=1)
                            resolution_budget_tiled_editor = gr.Textbox(label="Maximum With Tiling", value="", placeholder="3840x2176 or 8355840", lines=1, visible=False)
                            lock_resolutions_editor = gr.Checkbox(label="Lock Resolutions", value=False)
                            resolutions_editor = gr.Textbox(label="Custom Resolutions (one WxH value per line)", value="", lines=6, max_lines=10, placeholder="1024x2048")
                            gr.HTML("<div class='wangp-finetune-editor-hint'>Example: <code>1024x2048</code> is saved as <code>1024x2048 (1:2)</code>.</div>", padding=False, container=False)
                        with gr.Tab("Help"):
                            infos_editor = _markdown_editor("Model Infos")
                            prompt_infos_editor = _markdown_editor("Prompt Help")
                        with gr.Tab("Parameters", visible=False) as finetune_params_tab:
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as finetune_param_1_group:
                                finetune_param_1_dropdown = gr.Dropdown(label="Finetune Parameter 1", choices=[], value=None)
                                finetune_param_1_help = gr.Markdown(value="", visible=False, elem_classes=["wangp-finetune-param-help"])
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as finetune_param_2_group:
                                finetune_param_2_dropdown = gr.Dropdown(label="Finetune Parameter 2", choices=[], value=None)
                                finetune_param_2_help = gr.Markdown(value="", visible=False, elem_classes=["wangp-finetune-param-help"])
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as finetune_param_3_group:
                                finetune_param_3_dropdown = gr.Dropdown(label="Finetune Parameter 3", choices=[], value=None)
                                finetune_param_3_help = gr.Markdown(value="", visible=False, elem_classes=["wangp-finetune-param-help"])
                        with gr.Tab("Prompt Enhancer"):
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as enhancer_system_1_group:
                                enhancer_system_1_default = gr.State(value="")
                                with gr.Column(elem_classes=["wangp-finetune-editor-enhancer-field"]):
                                    enhancer_system_1_editor = gr.Textbox(label="System Prompt", value="", lines=6)
                                    enhancer_system_1_default_button = gr.Button("👁", visible=False, min_width=1, scale=0, elem_classes=["wangp-finetune-editor-enhancer-default-btn"])
                                enhancer_system_1_tokens = gr.Textbox(label="Max Tokens (empty = auto)", value="", lines=1, max_lines=1)
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as enhancer_system_2_group:
                                enhancer_system_2_default = gr.State(value="")
                                with gr.Column(elem_classes=["wangp-finetune-editor-enhancer-field"]):
                                    enhancer_system_2_editor = gr.Textbox(label="System Prompt", value="", lines=6)
                                    enhancer_system_2_default_button = gr.Button("👁", visible=False, min_width=1, scale=0, elem_classes=["wangp-finetune-editor-enhancer-default-btn"])
                                enhancer_system_2_tokens = gr.Textbox(label="Max Tokens (empty = auto)", value="", lines=1, max_lines=1)
                            with gr.Column(visible=False, elem_classes=["wangp-finetune-editor-field-group"]) as enhancer_system_3_group:
                                enhancer_system_3_default = gr.State(value="")
                                with gr.Column(elem_classes=["wangp-finetune-editor-enhancer-field"]):
                                    enhancer_system_3_editor = gr.Textbox(label="System Prompt", value="", lines=6)
                                    enhancer_system_3_default_button = gr.Button("👁", visible=False, min_width=1, scale=0, elem_classes=["wangp-finetune-editor-enhancer-default-btn"])
                                enhancer_system_3_tokens = gr.Textbox(label="Max Tokens (empty = auto)", value="", lines=1, max_lines=1)
                import_file = gr.File(label="Import Finetune JSON", file_types=[".json"], type="filepath", visible=False)
            with gr.Column(elem_classes=["wangp-finetune-editor-footer"]):
                use_current_settings = gr.Checkbox(label="Use Current Model Settings as Default Settings", value=False)
                with gr.Row(elem_classes=["wangp-finetune-editor-actions"]) as creator_actions:
                    create_button = gr.Button("Create", variant="primary", size="sm", elem_classes=["chat__template-modal-btn", "chat__template-modal-btn--primary"])
                    create_new_button = gr.Button("Create & New", size="sm", elem_classes=["chat__template-modal-btn"])
                    cancel_button = gr.Button("Cancel", size="sm", elem_classes=["chat__template-modal-btn"])
                with gr.Row(visible=False, elem_classes=["wangp-finetune-editor-actions"]) as editor_actions:
                    save_button = gr.Button("Save", variant="primary", size="sm", elem_classes=["chat__template-modal-btn", "chat__template-modal-btn--primary"])
                    export_button = gr.DownloadButton("Export", value=None, size="sm", elem_classes=["chat__template-modal-btn"])
                    delete_button = gr.Button("Delete", variant="stop", size="sm", elem_classes=["chat__template-modal-btn"])
                with gr.Row(visible=False, elem_classes=["wangp-finetune-editor-actions", "wangp-finetune-editor-delete-confirm"]) as delete_confirm:
                    confirm_delete_button = gr.Button("Confirm Delete", variant="stop", size="sm", elem_classes=["chat__template-modal-btn"])
                    cancel_delete_button = gr.Button("Cancel", size="sm", elem_classes=["chat__template-modal-btn"])
    ui = FinetuneEditorUI(
        popup=popup,
        title=title,
        mode=mode,
        source_model_type=source_model_type,
        original_id=original_id,
        creator_source_mode=creator_source_mode,
        form_fields=form_fields,
        import_file=import_file,
        finetunes_info=finetunes_info,
        source_info=source_info,
        id_text=id_text,
        auto_id=auto_id,
        name_text=name_text,
        description_text=description_text,
        urls_group=urls_group,
        urls_text=urls_text,
        urls2_group=urls2_group,
        urls2_text=urls2_text,
        text_encoder_group=text_encoder_group,
        text_encoder_text=text_encoder_text,
        custom_url_1_group=custom_url_1_group,
        custom_url_1_text=custom_url_1_text,
        custom_url_2_group=custom_url_2_group,
        custom_url_2_text=custom_url_2_text,
        custom_url_3_group=custom_url_3_group,
        custom_url_3_text=custom_url_3_text,
        loras_group=loras_group,
        loras_root_text=loras_root_text,
        loras_text=loras_text,
        loras_multipliers_text=loras_multipliers_text,
        lock_resolutions_editor=lock_resolutions_editor,
        resolutions_editor=resolutions_editor,
        resolution_budget_editor=resolution_budget_editor,
        resolution_budget_tiled_editor=resolution_budget_tiled_editor,
        resolution_min_editor=resolution_min_editor,
        infos_editor=infos_editor,
        prompt_infos_editor=prompt_infos_editor,
        finetune_params_tab=finetune_params_tab,
        finetune_param_1_group=finetune_param_1_group,
        finetune_param_1_dropdown=finetune_param_1_dropdown,
        finetune_param_1_help=finetune_param_1_help,
        finetune_param_2_group=finetune_param_2_group,
        finetune_param_2_dropdown=finetune_param_2_dropdown,
        finetune_param_2_help=finetune_param_2_help,
        finetune_param_3_group=finetune_param_3_group,
        finetune_param_3_dropdown=finetune_param_3_dropdown,
        finetune_param_3_help=finetune_param_3_help,
        enhancer_system_1_group=enhancer_system_1_group,
        enhancer_system_1_editor=enhancer_system_1_editor,
        enhancer_system_1_default=enhancer_system_1_default,
        enhancer_system_1_default_button=enhancer_system_1_default_button,
        enhancer_system_1_tokens=enhancer_system_1_tokens,
        enhancer_system_2_group=enhancer_system_2_group,
        enhancer_system_2_editor=enhancer_system_2_editor,
        enhancer_system_2_default=enhancer_system_2_default,
        enhancer_system_2_default_button=enhancer_system_2_default_button,
        enhancer_system_2_tokens=enhancer_system_2_tokens,
        enhancer_system_3_group=enhancer_system_3_group,
        enhancer_system_3_editor=enhancer_system_3_editor,
        enhancer_system_3_default=enhancer_system_3_default,
        enhancer_system_3_default_button=enhancer_system_3_default_button,
        enhancer_system_3_tokens=enhancer_system_3_tokens,
        use_current_settings=use_current_settings,
        creator_actions=creator_actions,
        editor_actions=editor_actions,
        create_button=create_button,
        create_new_button=create_new_button,
        cancel_button=cancel_button,
        save_button=save_button,
        export_button=export_button,
        delete_button=delete_button,
        close_button=close_button,
        delete_confirm=delete_confirm,
        confirm_delete_button=confirm_delete_button,
        cancel_delete_button=cancel_delete_button,
        create_apply_trigger=create_apply_trigger,
        create_save_trigger=create_save_trigger,
        create_new_apply_trigger=create_new_apply_trigger,
        create_new_save_trigger=create_new_save_trigger,
        save_apply_trigger=save_apply_trigger,
        save_save_trigger=save_save_trigger,
    )
    ui.close_button.click(fn=close_editor, outputs=[ui.popup], queue=False, show_progress="hidden")
    ui.cancel_button.click(fn=close_editor, outputs=[ui.popup], queue=False, show_progress="hidden")
    ui.creator_source_mode.change(fn=toggle_creator_source_mode, inputs=[ui.mode, ui.creator_source_mode], outputs=[ui.form_fields, ui.import_file, ui.use_current_settings], queue=False, show_progress="hidden")
    ui.delete_button.click(fn=lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[ui.editor_actions, ui.delete_confirm], queue=False, show_progress="hidden")
    ui.cancel_delete_button.click(fn=lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[ui.editor_actions, ui.delete_confirm], queue=False, show_progress="hidden")
    ui.enhancer_system_1_default_button.click(fn=_use_model_def_prompt, inputs=[ui.enhancer_system_1_default], outputs=[ui.enhancer_system_1_editor], queue=False, show_progress="hidden")
    ui.enhancer_system_2_default_button.click(fn=_use_model_def_prompt, inputs=[ui.enhancer_system_2_default], outputs=[ui.enhancer_system_2_editor], queue=False, show_progress="hidden")
    ui.enhancer_system_3_default_button.click(fn=_use_model_def_prompt, inputs=[ui.enhancer_system_3_default], outputs=[ui.enhancer_system_3_editor], queue=False, show_progress="hidden")
    return ui


def close_editor():
    return gr.update(visible=False)


def _markdown_editor(label: str) -> gr.Textbox:
    with gr.Column(elem_classes=["wangp-markdown-editor-field"]):
        gr.HTML(
            "<div class='wangp-markdown-editor-toolbar'>"
            "<button type='button' data-wangp-md-action='bold' title='Bold'><b>B</b></button>"
            "<button type='button' data-wangp-md-action='italic' title='Italic'><i>I</i></button>"
            "<button type='button' data-wangp-md-action='heading' title='Heading'>H</button>"
            "<button type='button' data-wangp-md-action='list' title='List'>•</button>"
            "<button type='button' data-wangp-md-action='link' title='Link'>↗</button>"
            "<button type='button' data-wangp-md-action='code' title='Code'>`</button>"
            "</div>",
            padding=False,
            container=False,
        )
        editor = gr.Textbox(label=label, value="", lines=10, max_lines=18, autoscroll=False, elem_classes=["wangp-markdown-editor"])
    return editor


def toggle_creator_source_mode(mode, creator_source_mode):
    importing = str(mode or "") == "creator" and str(creator_source_mode or "") == "import"
    return gr.update(visible=not importing), gr.update(visible=importing), gr.update(value=False, visible=not importing)


def _use_model_def_prompt(value):
    return str(value or "")


def prepare_finetune_action(deps: FinetuneEditorDeps, state, mode, original_id, source_model_type, action):
    if _form_model_is_active(deps, state, mode, original_id, source_model_type):
        token = f"{action}|skip_redirect_save|{time.time()}"
        return gr.update(value=token), gr.update()
    token = f"{action}|save_current_on_redirect|{time.time()}"
    return gr.update(), gr.update(value=token)


def save_finetune_from_action(deps: FinetuneEditorDeps, action_value, *values, create_new_output_count=0):
    parts = str(action_value or "").split("|")
    action = parts[0] if parts else ""
    skip_redirect_save = len(parts) > 1 and parts[1] == "skip_redirect_save"
    return save_finetune(deps, *values, create_new=action == "create_new", create_new_output_count=create_new_output_count, skip_redirect_save=skip_redirect_save)


def _form_model_is_active(deps: FinetuneEditorDeps, state, mode, original_id, source_model_type) -> bool:
    mode = "editor" if str(mode or "") == "editor" else "creator"
    form_model_type = _settings_source_model_type(mode, str(original_id or "").strip(), str(source_model_type or "").strip())
    return bool(form_model_type and deps.get_state_model_type(state) == form_model_type)


def bind_editor(
    ui: FinetuneEditorUI,
    *,
    deps_factory: Callable[[], FinetuneEditorDeps],
    state,
    toolbar_button,
    model_family,
    model_base_type_choice,
    model_choice,
    model_choice_target,
    refresh_form_trigger,
    model_description,
    header,
    validate_wizard_prompt: Callable,
    wizard_inputs: list,
    prompt_output,
    save_inputs_handler: Callable,
    target_state,
    generation_inputs: list,
    bind_model_change: bool = True,
):
    action_outputs = _action_outputs(ui, state, model_choice_target)
    delete_outputs = _delete_outputs(ui, state, model_choice_target)
    create_new_outputs = _create_new_outputs(ui, model_family, model_base_type_choice, model_choice, refresh_form_trigger, toolbar_button, model_description, header)

    gr.on(
        triggers=[toolbar_button.click],
        fn=lambda state_value: open_editor(deps_factory(), state_value),
        inputs=[state],
        outputs=_open_outputs(ui),
        show_progress="hidden",
    )

    def bind_save(trigger, save_trigger, apply_trigger, *, create_new=False):
        trigger(
            fn=lambda state_value, mode_value, original_id_value, source_model_type_value: prepare_finetune_action(deps_factory(), state_value, mode_value, original_id_value, source_model_type_value, "create_new" if create_new else "save"),
            inputs=[state, ui.mode, ui.original_id, ui.source_model_type],
            outputs=[save_trigger, apply_trigger],
            show_progress="hidden",
        )
        save_trigger.change(
            fn=validate_wizard_prompt,
            inputs=wizard_inputs,
            outputs=[prompt_output],
            show_progress="hidden",
        ).then(
            fn=save_inputs_handler,
            inputs=[target_state] + generation_inputs,
            outputs=None,
        ).then(
            fn=lambda action_value, *values: save_finetune_from_action(deps_factory(), action_value, *values, create_new_output_count=len(create_new_outputs)),
            inputs=[save_trigger] + _save_inputs(ui, state),
            outputs=create_new_outputs if create_new else action_outputs,
            show_progress="hidden",
        )
        apply_trigger.change(
            fn=lambda action_value, *values: save_finetune_from_action(deps_factory(), action_value, *values, create_new_output_count=len(create_new_outputs)),
            inputs=[apply_trigger] + _save_inputs(ui, state),
            outputs=create_new_outputs if create_new else action_outputs,
            show_progress="hidden",
        )

    bind_save(ui.create_button.click, ui.create_save_trigger, ui.create_apply_trigger)
    bind_save(ui.create_new_button.click, ui.create_new_save_trigger, ui.create_new_apply_trigger, create_new=True)
    bind_save(ui.save_button.click, ui.save_save_trigger, ui.save_apply_trigger)
    ui.confirm_delete_button.click(
        fn=lambda state_value, original_id_value, source_model_type_value, name_value: delete_finetune(deps_factory(), state_value, original_id_value, source_model_type_value, name_value),
        inputs=[state, ui.original_id, ui.source_model_type, ui.name_text],
        outputs=delete_outputs,
        show_progress="hidden",
    )
    if bind_model_change:
        model_choice_target.change(
            fn=lambda model_target_value: toolbar_button_updates(deps_factory(), str(model_target_value or "").split("|", 1)[0].strip()),
            inputs=[model_choice_target],
            outputs=[toolbar_button],
            show_progress="hidden",
        )
    auto_id_inputs = [state, ui.mode, ui.original_id, ui.source_model_type, ui.id_text, ui.auto_id, ui.name_text, ui.description_text]
    ui.auto_id.change(fn=lambda *values: refresh_auto_id(deps_factory(), *values, update_interactivity=True), inputs=auto_id_inputs, outputs=[ui.id_text], queue=False, show_progress="hidden")
    ui.name_text.input(fn=lambda *values: refresh_auto_id(deps_factory(), *values, event_kind="input"), inputs=auto_id_inputs, outputs=[ui.id_text], queue=False, show_progress="hidden")
    ui.name_text.change(fn=lambda *values: refresh_auto_id(deps_factory(), *values, event_kind="change"), inputs=auto_id_inputs, outputs=[ui.id_text], queue=False, show_progress="hidden")
    ui.name_text.blur(fn=lambda *values: refresh_auto_id(deps_factory(), *values, event_kind="blur"), inputs=auto_id_inputs, outputs=[ui.id_text], queue=False, show_progress="hidden")


def _open_outputs(ui: FinetuneEditorUI) -> list:
    return [
        ui.popup,
        ui.title,
        ui.mode,
        ui.source_model_type,
        ui.original_id,
        ui.creator_source_mode,
        ui.form_fields,
        ui.import_file,
        ui.finetunes_info,
        ui.source_info,
        ui.id_text,
        ui.auto_id,
        ui.name_text,
        ui.description_text,
        ui.urls_group,
        ui.urls_text,
        ui.urls2_group,
        ui.urls2_text,
        ui.text_encoder_group,
        ui.text_encoder_text,
        ui.custom_url_1_group,
        ui.custom_url_1_text,
        ui.custom_url_2_group,
        ui.custom_url_2_text,
        ui.custom_url_3_group,
        ui.custom_url_3_text,
        ui.loras_group,
        ui.loras_root_text,
        ui.loras_text,
        ui.loras_multipliers_text,
        ui.lock_resolutions_editor,
        ui.resolutions_editor,
        ui.infos_editor,
        ui.prompt_infos_editor,
        ui.finetune_params_tab,
        ui.finetune_param_1_group,
        ui.finetune_param_1_dropdown,
        ui.finetune_param_1_help,
        ui.finetune_param_2_group,
        ui.finetune_param_2_dropdown,
        ui.finetune_param_2_help,
        ui.finetune_param_3_group,
        ui.finetune_param_3_dropdown,
        ui.finetune_param_3_help,
        ui.enhancer_system_1_group,
        ui.enhancer_system_1_editor,
        ui.enhancer_system_1_default,
        ui.enhancer_system_1_default_button,
        ui.enhancer_system_1_tokens,
        ui.enhancer_system_2_group,
        ui.enhancer_system_2_editor,
        ui.enhancer_system_2_default,
        ui.enhancer_system_2_default_button,
        ui.enhancer_system_2_tokens,
        ui.enhancer_system_3_group,
        ui.enhancer_system_3_editor,
        ui.enhancer_system_3_default,
        ui.enhancer_system_3_default_button,
        ui.enhancer_system_3_tokens,
        ui.use_current_settings,
        ui.creator_actions,
        ui.editor_actions,
        ui.delete_confirm,
        ui.export_button,
        ui.resolution_budget_editor,
        ui.resolution_budget_tiled_editor,
        ui.resolution_min_editor,
    ]


def _action_outputs(ui: FinetuneEditorUI, state, model_choice_target) -> list:
    return [
        state,
        ui.popup,
        model_choice_target,
        ui.export_button,
    ]


def _delete_outputs(ui: FinetuneEditorUI, state, model_choice_target) -> list:
    return [
        state,
        ui.popup,
        model_choice_target,
        ui.export_button,
    ]


def _create_new_outputs(ui: FinetuneEditorUI, model_family, model_base_type_choice, model_choice, refresh_form_trigger, toolbar_button, model_description, header) -> list:
    return [
        model_family,
        model_base_type_choice,
        model_choice,
        refresh_form_trigger,
        toolbar_button,
        model_description,
        header,
        ui.export_button,
        *_open_outputs(ui),
    ]


def _save_inputs(ui: FinetuneEditorUI, state) -> list:
    return [
        state,
        ui.mode,
        ui.original_id,
        ui.source_model_type,
        ui.creator_source_mode,
        ui.import_file,
        ui.id_text,
        ui.auto_id,
        ui.name_text,
        ui.description_text,
        ui.urls_text,
        ui.urls2_text,
        ui.text_encoder_text,
        ui.custom_url_1_text,
        ui.custom_url_2_text,
        ui.custom_url_3_text,
        ui.loras_text,
        ui.loras_multipliers_text,
        ui.lock_resolutions_editor,
        ui.resolutions_editor,
        ui.infos_editor,
        ui.prompt_infos_editor,
        ui.finetune_param_1_dropdown,
        ui.finetune_param_2_dropdown,
        ui.finetune_param_3_dropdown,
        ui.enhancer_system_1_editor,
        ui.enhancer_system_1_tokens,
        ui.enhancer_system_2_editor,
        ui.enhancer_system_2_tokens,
        ui.enhancer_system_3_editor,
        ui.enhancer_system_3_tokens,
        ui.use_current_settings,
        ui.resolution_budget_editor,
        ui.resolution_budget_tiled_editor,
        ui.resolution_min_editor,
    ]


def toolbar_button_updates(deps: FinetuneEditorDeps, model_type: str | None):
    is_editor = is_finetune_model(deps, model_type)
    return gr.update(value="\u270e" if is_editor else "+")


def refresh_auto_id(deps: FinetuneEditorDeps, state, mode, original_id, source_model_type, id_text, auto_id, name, description, update_interactivity=False, event_kind=""):
    mode = "editor" if str(mode or "") == "editor" else "creator"
    source_model_type = str(source_model_type or "").strip() or _source_model_type(deps, deps.get_state_model_type(state))
    raw_source, _ = _load_model_json(deps, source_model_type)
    source_model = raw_source.get("model", {})
    if mode == "creator":
        if not auto_id:
            return gr.update(interactive=True) if update_interactivity else gr.update()
        return gr.update(value=_unique_model_id(deps, _auto_model_id(source_model_type, name, source_model.get("name", ""))), interactive=False)
    if event_kind in {"change", "blur"} and _editor_should_auto_rename_generic_id(original_id, id_text, source_model_type):
        return gr.update(value=_editor_auto_model_id(deps, original_id, source_model_type, name, source_model.get("name", "")), interactive=True)
    return gr.update(interactive=True) if update_interactivity else gr.update()


def open_editor(deps: FinetuneEditorDeps, state, source_model_type_override=None):
    current_model_type = str(source_model_type_override or "").strip() or deps.get_state_model_type(state)
    editor_mode = is_finetune_model(deps, current_model_type)
    original_id = current_model_type if editor_mode else ""
    raw_current, _ = _load_model_json(deps, current_model_type)
    source_model_type = _source_model_type(deps, current_model_type)
    raw_source, _ = _load_model_json(deps, source_model_type)
    model_for_values = copy.deepcopy((raw_current if editor_mode else raw_source).get("model", {}))
    title = "Finetune Editor" if editor_mode else "Finetune Creator"
    editable_fields = _editable_url_fields(deps, source_model_type, raw_source)
    field_values = {field: _format_list(model_for_values.get(field, [])) for field in FINETUNE_URL_FIELDS}
    source_model = raw_source.get("model", {})
    id_value = original_id if editor_mode else _unique_model_id(deps, _auto_model_id(source_model_type, model_for_values.get("name", ""), source_model.get("name", "")))
    custom_updates = _custom_url_component_updates(deps, source_model_type, model_for_values)
    loras_update = _loras_component_update(deps, source_model_type, model_for_values)
    source_def = deps.get_model_def(source_model_type)
    model_for_values = resolution_utils.migrate_resolution_definition(model_for_values, legacy_lists=(raw_current if editor_mode else raw_source).get("settings_version", 0) < 2.80)
    effective_def = deps.get_model_def(current_model_type)
    for key in ("resolution_budget", "resolution_budget_tiled", "resolution_min", "resolutions", "lock_resolutions"):
        if key not in model_for_values and key in effective_def:
            model_for_values[key] = effective_def[key]
    finetunes_info = str(source_def.get("finetunes_infos", "") or "").strip()
    finetune_param_updates = _finetune_param_component_updates(_finetune_param_specs(deps, source_model_type), model_for_values)
    enhancer_updates = _prompt_enhancer_component_updates(deps, source_model_type, model_for_values)
    current_source_choice = _creator_current_choice(deps, source_model_type)
    return (
        gr.update(visible=True),
        f"<div class='chat__template-modal-heading'>{title}</div>",
        "editor" if editor_mode else "creator",
        source_model_type,
        original_id,
        gr.update(choices=[current_source_choice, ("By importing a File", "import")], value="current", visible=not editor_mode),
        gr.update(visible=True),
        gr.update(value=None, visible=False),
        gr.update(value=finetunes_info, visible=bool(finetunes_info)),
        gr.update(value=_render_source_info(raw_source.get("model", {}).get("name", deps.get_model_name(source_model_type)), source_model_type), visible=editor_mode),
        gr.update(value=id_value, interactive=editor_mode),
        gr.update(value=not editor_mode, visible=not editor_mode),
        model_for_values.get("name", ""),
        model_for_values.get("description", ""),
        gr.update(visible="URLs" in editable_fields),
        field_values["URLs"],
        gr.update(visible="URLs2" in editable_fields),
        field_values["URLs2"],
        gr.update(visible="text_encoder_URLs" in editable_fields),
        field_values["text_encoder_URLs"],
        *custom_updates,
        *loras_update,
        model_for_values.get("lock_resolutions", False),
        _format_resolutions_value(model_for_values.get("resolutions", "")),
        _format_help_value(model_for_values.get("infos", "")),
        _format_help_value(model_for_values.get("prompt_infos", "")),
        *finetune_param_updates,
        *enhancer_updates,
        gr.update(value=False, visible=True),
        gr.update(visible=not editor_mode),
        gr.update(visible=editor_mode),
        gr.update(visible=False),
        gr.update(value=_finetune_json_path(original_id) if editor_mode else None, visible=editor_mode),
        str(model_for_values.get("resolution_budget") or ""),
        gr.update(value=str(model_for_values.get("resolution_budget_tiled") or ""), visible=bool(source_def.get("phase_2_spatial_tiling") and source_def.get("guidance_max_phases", 0) >= 2)),
        str(model_for_values.get("resolution_min") or ""),
    )


def save_finetune(deps: FinetuneEditorDeps, state, mode, original_id, source_model_type, creator_source_mode, import_file, id_text, auto_id, name, description, urls, urls2, text_encoder_urls, custom_url_1, custom_url_2, custom_url_3, loras, loras_multipliers, lock_resolutions, resolutions, infos, prompt_infos, finetune_param_1, finetune_param_2, finetune_param_3, enhancer_system_1, enhancer_system_1_tokens, enhancer_system_2, enhancer_system_2_tokens, enhancer_system_3, enhancer_system_3_tokens, use_current_settings, resolution_budget=None, resolution_budget_tiled=None, resolution_min=None, create_new=False, create_new_output_count=0, skip_redirect_save=False):
    mode = "editor" if str(mode or "") == "editor" else "creator"
    original_id = str(original_id or "").strip()
    source_model_type = str(source_model_type or "").strip()
    if mode == "creator" and str(creator_source_mode or "") == "import":
        if create_new:
            gr.Info("Create & New is not available when importing a finetune JSON.")
            return _no_create_new_action_updates(create_new_output_count)
        return import_finetune(deps, state, import_file, skip_redirect_save=skip_redirect_save)
    if not source_model_type or deps.get_model_def(source_model_type) is None:
        gr.Info("Unable to identify the source model for this finetune.")
        return _no_create_new_action_updates(create_new_output_count) if create_new else _no_action_updates()

    raw_source, _ = _load_model_json(deps, source_model_type)
    raw_existing = None
    if mode == "editor":
        if not is_finetune_model(deps, original_id):
            gr.Info("This model is not a finetune and cannot be edited.")
            return _no_action_updates()
        raw_existing, _ = _load_model_json(deps, original_id)

    url_inputs = {
        "URLs": urls,
        "URLs2": urls2,
        "text_encoder_URLs": text_encoder_urls,
    }
    values = {
        field: _parse_url_value(value) for field, value in url_inputs.items()
    }
    custom_values = _custom_url_values(deps, source_model_type, [custom_url_1, custom_url_2, custom_url_3])
    loras_value = _parse_loras_value(deps, source_model_type, loras)
    loras_multipliers_value = _parse_loras_multipliers_value(loras_multipliers)
    resolutions_value, resolutions_problems = _parse_resolutions_value(resolutions)
    editable_fields = _editable_url_fields(deps, source_model_type, raw_source)
    urls_required = "URLs" in raw_source.get("model", {}) or bool(raw_existing and "URLs" in raw_existing.get("model", {}))
    source_model = raw_source.get("model", {})
    source_name = source_model.get("name", "")
    finetune_param_specs = _finetune_param_specs(deps, source_model_type)
    finetune_param_values = [finetune_param_1, finetune_param_2, finetune_param_3]
    problems = _validate_inputs(deps, mode, original_id, source_model_type, id_text, auto_id, name, description, editable_fields, values, urls_required, source_name)
    problems.extend(_validate_url_values(url_inputs))
    problems.extend(_validate_finetune_param_values(finetune_param_specs, finetune_param_values))
    problems.extend(resolutions_problems)
    bounds = {}
    for key, value in (("resolution_budget", resolution_budget), ("resolution_budget_tiled", resolution_budget_tiled), ("resolution_min", resolution_min)):
        bounds[key], errors = _parse_resolution_budget_value(value)
        problems.extend(errors)
    try:
        resolution_utils.validate_resolution_definition({**deps.get_model_def(source_model_type), **bounds, "resolutions": resolutions_value, "lock_resolutions": bool(lock_resolutions)})
    except ValueError as error:
        problems.append(str(error))
    if problems:
        gr.Info("Finetune not saved:\n- " + "\n- ".join(problems))
        return _no_create_new_action_updates(create_new_output_count) if create_new else _no_action_updates()

    model_id = _resolve_model_id(deps, mode, original_id, source_model_type, id_text, auto_id, name, source_name)
    settings_to_copy = _settings_to_copy(deps, state, _settings_source_model_type(mode, original_id, source_model_type)) if use_current_settings else None
    enhancer_specs = _prompt_enhancer_system_specs(deps, source_model_type)
    enhancer_values = [(enhancer_system_1, enhancer_system_1_tokens), (enhancer_system_2, enhancer_system_2_tokens), (enhancer_system_3, enhancer_system_3_tokens)]
    raw_output = _build_finetune_json(mode, source_model_type, raw_source, raw_existing, name, description, editable_fields, values, custom_values, loras_value, loras_multipliers_value, lock_resolutions, resolutions_value, infos, prompt_infos, finetune_param_specs, finetune_param_values, enhancer_specs, enhancer_values, settings_to_copy)
    for key, value in bounds.items():
        if value is None and key not in deps.get_model_def(source_model_type) and key not in raw_output["model"]:
            raw_output["model"].pop(key, None)
        else:
            # An explicit null clears an inherited handler recommendation.
            raw_output["model"][key] = value
    raw_output["settings_version"] = deps.settings_version
    url_fields_changed = mode == "editor" and _url_fields_changed((raw_existing or {}).get("model", {}), raw_output.get("model", {}), [*FINETUNE_URL_FIELDS, *custom_values.keys(), *FINETUNE_LORA_FIELDS])
    old_path = _finetune_json_path(original_id) if mode == "editor" else None
    new_path = _finetune_json_path(model_id)
    os.makedirs(FINETUNES_DIR, exist_ok=True)
    with open(new_path, "w", encoding="utf-8") as writer:
        json.dump(raw_output, writer, indent=4)
    if mode == "editor" and original_id != model_id and old_path and os.path.exists(old_path):
        os.remove(old_path)
        _rename_settings(deps, original_id, model_id)
        _rename_cached_settings(deps, state, original_id, model_id)
    if settings_to_copy is not None:
        deps.set_model_settings(state, model_id, settings_to_copy)
    if url_fields_changed:
        deps.request_reload_if_loaded(original_id)
    parse_errors = deps.refresh_model_defs() or []
    _warn_parse_errors(parse_errors)
    finetune_label = _finetune_label(model_id, name)
    if create_new and mode == "creator":
        gr.Info(f"Finetune '{finetune_label}' created.")
        return _create_new_updates(deps, state, source_model_type)
    if skip_redirect_save:
        state["ignore_save_form"] = True
    gr.Info(f"Finetune '{finetune_label}' {'saved' if mode == 'editor' else 'created'}.")
    return state, *_redirect_to_model_updates(deps, model_id)


def import_finetune(deps: FinetuneEditorDeps, state, import_file, skip_redirect_save=False):
    import_path = _uploaded_file_path(import_file)
    if not import_path or not os.path.isfile(import_path):
        gr.Info("Please select a finetune JSON file to import.")
        return _no_action_updates()
    try:
        with open(import_path, "r", encoding="utf-8") as reader:
            raw_output = json.load(reader)
    except Exception as exc:
        gr.Info(f"Unable to import finetune JSON: {exc}")
        return _no_action_updates()
    if not isinstance(raw_output, dict) or not isinstance(raw_output.get("model"), dict):
        gr.Info("Finetune not imported: JSON must contain a 'model' object.")
        return _no_action_updates()
    architecture = str(raw_output["model"].get("architecture") or "").strip()
    if not architecture:
        gr.Info("Finetune not imported: model.architecture is required.")
        return _no_action_updates()
    try:
        resolution_utils.fix_resolution_settings(raw_output)
        resolution_utils.validate_resolution_definition(raw_output["model"])
    except ValueError as error:
        gr.Info(f"Finetune not imported: {error}")
        return _no_action_updates()
    raw_output["settings_version"] = deps.settings_version
    source_model_type = str(raw_output["model"].get(FINETUNE_SOURCE_MODEL_KEY) or architecture).strip()
    _compress_model_path_fields(raw_output["model"], [*FINETUNE_URL_FIELDS, "loras", *_custom_url_keys(deps, source_model_type)])
    model_id = _sanitize_model_id(Path(import_path).stem)
    if not model_id:
        gr.Info("Finetune not imported: file name cannot be used as a model id.")
        return _no_action_updates()
    if _model_id_exists(deps, model_id):
        gr.Info(f"Finetune not imported: '{_finetune_label(model_id, raw_output.get('model', {}).get('name', ''))}' already exists.")
        return _no_action_updates()
    os.makedirs(FINETUNES_DIR, exist_ok=True)
    new_path = _finetune_json_path(model_id)
    with open(new_path, "w", encoding="utf-8") as writer:
        json.dump(raw_output, writer, indent=4)
    parse_errors = deps.refresh_model_defs() or []
    _warn_parse_errors(parse_errors)
    if skip_redirect_save:
        state["ignore_save_form"] = True
    gr.Info(f"Finetune '{_finetune_label(model_id, raw_output.get('model', {}).get('name', ''))}' imported.")
    return state, *_redirect_to_model_updates(deps, model_id)


def delete_finetune(deps: FinetuneEditorDeps, state, original_id, source_model_type, name):
    original_id = str(original_id or "").strip()
    source_model_type = str(source_model_type or "").strip() or _source_model_type(deps, original_id)
    if not original_id or not is_finetune_model(deps, original_id):
        gr.Info("This model is not a finetune and cannot be deleted.")
        return _no_delete_action_updates()
    json_path = _finetune_json_path(original_id)
    finetune_label = _finetune_label(original_id, name)
    if os.path.exists(json_path):
        os.remove(json_path)
    settings_path = deps.get_settings_file_name(original_id)
    if os.path.exists(settings_path):
        os.remove(settings_path)
    _delete_cached_settings(state, original_id)
    deps.request_reload_if_loaded(original_id)
    parse_errors = deps.refresh_model_defs() or []
    _warn_parse_errors(parse_errors)
    target_model_type = source_model_type if deps.get_model_def(source_model_type) is not None else next(iter(deps.displayed_model_types), "")
    state["ignore_save_form"] = True
    gr.Info(f"Finetune '{finetune_label}' deleted.")
    return state, *_redirect_to_model_updates(deps, target_model_type)


def _source_model_type(deps: FinetuneEditorDeps, model_type: str) -> str:
    model_def = deps.get_model_def(model_type) or {}
    if is_finetune_model_def(model_def):
        return model_def.get(FINETUNE_SOURCE_MODEL_KEY) or model_def.get("architecture") or deps.get_base_model_type(model_type)
    return model_type


def _load_model_json(deps: FinetuneEditorDeps, model_type: str):
    model_def = deps.get_model_def(model_type) or {}
    path = model_def.get("path", "")
    if not path:
        return {"model": {}}, ""
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader), path


def _uploaded_file_path(uploaded) -> str:
    if isinstance(uploaded, str):
        return uploaded
    if isinstance(uploaded, dict):
        return str(uploaded.get("path") or uploaded.get("name") or "")
    if isinstance(uploaded, list) and uploaded:
        return _uploaded_file_path(uploaded[0])
    return str(getattr(uploaded, "name", "") or "")


def _editable_url_fields(deps: FinetuneEditorDeps, source_model_type: str, raw_source: dict) -> set[str]:
    source_def = deps.get_model_def(source_model_type) or {}
    raw_model = raw_source.get("model", {})
    fields = set()
    for field in ("URLs", "URLs2"):
        if field in source_def or field in raw_model:
            fields.add(field)
    if "text_encoder_URLs" in source_def or "text_encoder" in source_def or "text_encoder_URLs" in raw_model or "text_encoder" in raw_model:
        fields.add("text_encoder_URLs")
    return fields


def _custom_url_keys(deps: FinetuneEditorDeps, source_model_type: str) -> list[str]:
    keys = (deps.get_model_def(source_model_type) or {}).get("finetune_custom_urls", [])
    if isinstance(keys, str):
        keys = [keys]
    return [str(key).strip() for key in keys if str(key).strip()][:MAX_CUSTOM_URL_FIELDS]


def _custom_url_component_updates(deps: FinetuneEditorDeps, source_model_type: str, model_for_values: dict):
    keys = _custom_url_keys(deps, source_model_type)
    updates = []
    for index in range(MAX_CUSTOM_URL_FIELDS):
        key = keys[index] if index < len(keys) else ""
        value = model_for_values.get(key, "") if key else ""
        updates.append(gr.update(visible=bool(key)))
        updates.append(gr.update(label=_friendly_label(key) if key else f"Custom Checkpoint {index + 1}", value=_format_single_value(value)))
    return updates


def _custom_url_values(deps: FinetuneEditorDeps, source_model_type: str, values: list[str]) -> dict[str, str]:
    keys = _custom_url_keys(deps, source_model_type)
    return {key: fl.compress_path(str(values[index] or "").strip()) for index, key in enumerate(keys)}


def _finetune_param_specs(deps: FinetuneEditorDeps, source_model_type: str) -> list[dict]:
    definitions = deps.get_model_def(source_model_type).get("finetunes_params", {})
    if not definitions:
        return []
    if not isinstance(definitions, dict) or len(definitions) > MAX_FINETUNE_PARAMS:
        raise ValueError(f"finetunes_params must be a dictionary containing at most {MAX_FINETUNE_PARAMS} parameters")
    specs = []
    for param_id, definition in definitions.items():
        param_id = str(param_id or "").strip()
        if not param_id or not isinstance(definition, dict):
            raise ValueError("Each finetunes_params entry must have a non-empty id and a definition dictionary")
        choices = definition.get("choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ValueError(f"Finetune parameter {param_id!r} requires non-empty dropdown choices")
        normalized_choices = []
        for choice in choices:
            if not isinstance(choice, (list, tuple)) or len(choice) != 2:
                raise ValueError(f"Finetune parameter {param_id!r} choices must be [label, value] pairs")
            normalized_choices.append((str(choice[0]), choice[1]))
        if "default" not in definition or definition["default"] not in [value for _, value in normalized_choices]:
            raise ValueError(f"Finetune parameter {param_id!r} default must match one dropdown value")
        specs.append({
            "id": param_id,
            "label": str(definition.get("label") or _friendly_label(param_id)),
            "choices": normalized_choices,
            "default": definition["default"],
            "description": str(definition.get("description", definition.get("help", "")) or "").strip(),
        })
    return specs


def _finetune_param_component_updates(specs: list[dict], model_values: dict) -> list:
    updates = [gr.update(visible=bool(specs))]
    for index in range(MAX_FINETUNE_PARAMS):
        spec = specs[index] if index < len(specs) else None
        if spec is None:
            updates.extend((gr.update(visible=False), gr.update(choices=[], value=None), gr.update(value="", visible=False)))
            continue
        value = model_values.get(spec["id"], spec["default"])
        if value not in [choice_value for _, choice_value in spec["choices"]]:
            raise ValueError(f"Finetune parameter {spec['id']!r} value {value!r} is not one of its dropdown choices")
        updates.extend((
            gr.update(visible=True),
            gr.update(label=spec["label"], choices=spec["choices"], value=value),
            gr.update(value=spec["description"], visible=bool(spec["description"])),
        ))
    return updates


def _loras_component_update(deps: FinetuneEditorDeps, source_model_type: str, model_values: dict):
    lora_dir = _lora_dir_for_model(deps, source_model_type)
    loras_value = _format_loras_value(deps, source_model_type, model_values.get("loras", []))
    loras_visible = _model_supports_loras(deps, source_model_type, lora_dir)
    return [
        gr.update(visible=loras_visible),
        gr.update(value=lora_dir or ""),
        gr.update(value=loras_value),
        gr.update(value=_format_loras_multipliers(model_values.get("loras_multipliers", []))),
    ]


def _model_supports_loras(deps: FinetuneEditorDeps, source_model_type: str, lora_dir: str | None = None) -> bool:
    model_def = deps.get_model_def(source_model_type) or {}
    return not bool(model_def.get("no_lora", False)) and (lora_dir if lora_dir is not None else _lora_dir_for_model(deps, source_model_type)) is not None


def _lora_dir_for_model(deps: FinetuneEditorDeps, source_model_type: str) -> str | None:
    try:
        return deps.get_lora_dir(source_model_type)
    except Exception:
        return None


def _format_loras_value(deps: FinetuneEditorDeps, source_model_type: str, value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return f"={value}" if value else ""
    loras = [str(item) for item in value if str(item or "").strip()]
    lora_dir = _lora_dir_for_model(deps, source_model_type)
    if lora_dir is not None and loras:
        loras = deps.update_loras_url_cache(lora_dir, loras) or []
    return "\n".join(_format_lora_path_for_edit(lora_dir, item) for item in loras)


def _format_loras_multipliers(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return f"={value}" if value else ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _parse_loras_value(deps: FinetuneEditorDeps, source_model_type: str, value):
    reference = _parse_reference_value(value)
    if reference is not None:
        return reference
    loras = _parse_lora_entries(value)
    if not loras:
        return []
    lora_dir = _lora_dir_for_model(deps, source_model_type)
    if lora_dir is None:
        return [fl.compress_path(lora) for lora in loras]
    return [_format_lora_path_for_save(deps, lora_dir, lora) for lora in loras]


def _parse_lora_entries(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        entries = [str(item).strip() for item in value]
    else:
        entries = [line.strip() for line in str(value or "").replace("\r", "").split("\n")]
    seen, loras = set(), []
    for entry in entries:
        if not entry:
            continue
        key = entry.casefold()
        if key not in seen:
            seen.add(key)
            loras.append(entry)
    return loras


def _format_lora_path_for_edit(lora_dir: str | None, value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if _is_lora_url(value):
        return _lora_relative_path_from_url(value)
    if lora_dir is None:
        return fl.compress_path(value)
    raw, path = _lora_raw_and_path(lora_dir, value)
    if _is_under_lora_dir(path, lora_dir):
        return os.path.relpath(path, lora_dir).replace("\\", "/")
    return fl.compress_path(raw)


def _format_lora_path_for_save(deps: FinetuneEditorDeps, lora_dir: str, value: str) -> str:
    value = str(value or "").strip()
    if _is_lora_url(value):
        relative_lora = (deps.update_loras_url_cache(lora_dir, [value]) or [value])[0]
        return fl.compress_path(deps.get_lora_URL(lora_dir, relative_lora))
    _raw, path = _lora_raw_and_path(lora_dir, value)
    if _is_under_lora_dir(path, lora_dir):
        relative_lora = os.path.relpath(path, lora_dir).replace("\\", "/")
        return fl.compress_path(deps.get_lora_URL(lora_dir, relative_lora))
    return fl.compress_path(os.path.abspath(os.path.normpath(path)))


def _lora_raw_and_path(lora_dir: str, value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if fl.is_relative_down_path(raw):
        return raw, os.path.join(lora_dir, raw)
    raw = fl.uncompress_path(raw)
    return raw, raw if os.path.isabs(raw) else os.path.join(lora_dir, raw)


def _lora_relative_path_from_url(value: str) -> str:
    parts = str(value).split("|", 1)
    if len(parts) == 1:
        return os.path.basename(parts[0])
    return os.path.join(fl.clean_relative_path(parts[1]), os.path.basename(parts[0])).replace("\\", "/")


def _is_lora_url(value: str) -> bool:
    return str(value).startswith(("http:", "https:"))


def _is_under_lora_dir(path: str, lora_dir: str) -> bool:
    path = os.path.abspath(os.path.normpath(path))
    lora_dir = os.path.abspath(os.path.normpath(lora_dir))
    try:
        return os.path.commonpath([path, lora_dir]).casefold() == lora_dir.casefold()
    except ValueError:
        return False


def _parse_loras_multipliers_value(value):
    reference = _parse_reference_value(value)
    if reference is not None:
        return reference
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value if str(item).strip()]
    else:
        tokens = []
        for line in str(value or "").replace("\r", "").split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.extend(token for token in line.split() if token)
    return [_float_or_string(token) for token in tokens]


def _parse_reference_value(value) -> str | None:
    if isinstance(value, (list, tuple)):
        return None
    lines = _parse_multiline(value)
    if len(lines) == 1 and lines[0].startswith("="):
        return lines[0][1:].strip()
    return None


def _float_or_string(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _prompt_enhancer_component_updates(deps: FinetuneEditorDeps, source_model_type: str, model_values: dict):
    specs = _prompt_enhancer_system_specs(deps, source_model_type)
    model_def = deps.get_model_def(source_model_type) or {}
    updates = []
    for index in range(MAX_PROMPT_ENHANCER_SYSTEMS):
        spec = specs[index] if index < len(specs) else None
        label = spec["label"] if spec else "System Prompt"
        default_prompt = _prompt_enhancer_model_def_system_value(model_def, spec)
        updates.append(gr.update(visible=spec is not None))
        updates.append(gr.update(label=label, value=_prompt_enhancer_system_value(model_values, spec) if spec else ""))
        updates.append(default_prompt)
        updates.append(gr.update(visible=bool(default_prompt)))
        updates.append(gr.update(label=spec.get("tokens_label", f"{label} Max Tokens (empty = auto)") if spec else "Max Tokens (empty = auto)", value=_prompt_enhancer_system_tokens(model_values, spec) if spec else ""))
    return updates


def _supports_image_output_mode(model_def: dict) -> bool:
    model_modes = model_def.get("model_modes", None)
    image_modes = model_modes.get("image_modes", []) if isinstance(model_modes, dict) else []
    return bool(model_def.get("v2i_switch_supported", False) or model_def.get("inpaint_support", False) or 1 in image_modes or 2 in image_modes)


def _prompt_enhancer_system_specs(deps: FinetuneEditorDeps, source_model_type: str) -> list[dict]:
    model_def = deps.get_model_def(source_model_type) or {}
    if not isinstance(model_def.get("prompt_enhancer_def"), dict):
        return _default_prompt_enhancer_system_specs(model_def)
    mode_prefixes = _prompt_enhancer_mode_prefixes(model_def)
    grouped = {}
    for label, mode in _prompt_enhancer_choices(model_def):
        for step_mode in prompt_enhancer_chaining.split_mode(mode):
            suffix = _prompt_enhancer_profile_suffix(step_mode)
            prefixes = mode_prefixes(step_mode)
            if not prefixes:
                continue
            grouped.setdefault(suffix, {"suffix": suffix, "labels": [], "prefixes": []})
            grouped[suffix]["labels"].append(str(label))
            for prefix in prefixes:
                if prefix not in grouped[suffix]["prefixes"]:
                    grouped[suffix]["prefixes"].append(prefix)
    specs = []
    for suffix, spec in sorted(grouped.items(), key=lambda item: int(item[0] or 0)):
        specs.append({
            "suffix": "" if suffix == "0" else suffix,
            "label": _prompt_enhancer_system_label(spec["labels"]),
            "prefixes": spec["prefixes"],
        })
    return specs[:MAX_PROMPT_ENHANCER_SYSTEMS]


def _default_prompt_enhancer_system_specs(model_def: dict) -> list[dict]:
    prefixes = []
    mode_prefixes = _prompt_enhancer_mode_prefixes(model_def)
    for mode in _prompt_enhancer_default_modes(model_def):
        for prefix in mode_prefixes(mode):
            if prefix not in prefixes:
                prefixes.append(prefix)
    labels = {
        "text": ("Text Prompt Enhancer Instructions", "Text Prompt Enhancer Max Tokens (empty = auto)"),
        "video": ("Video Prompt Enhancer Instructions", "Video Prompt Enhancer Max Tokens (empty = auto)"),
        "image": ("Image Prompt Enhancer Instructions", "Image Prompt Enhancer Max Tokens (empty = auto)"),
    }
    return [{"suffix": "", "label": labels[prefix][0], "tokens_label": labels[prefix][1], "prefixes": [prefix]} for prefix in prefixes[:MAX_PROMPT_ENHANCER_SYSTEMS]]


def _prompt_enhancer_default_modes(model_def: dict) -> list[str]:
    selection = model_def.get("prompt_enhancer_choices_allowed", ["T"] if model_def.get("audio_only", False) else ["T", "TI"])
    if isinstance(selection, str):
        selection = [selection]
    if not isinstance(selection, list):
        selection = []
    return [str(mode).strip() for mode in selection if str(mode).strip()]


def _prompt_enhancer_choices(model_def: dict) -> list[tuple[str, str]]:
    default_labels = prompt_enhancer_labels.default_labels(model_def)
    prompt_enhancer_def = prompt_enhancer_labels.resolve_definition(model_def)
    if isinstance(prompt_enhancer_def, dict):
        selection = prompt_enhancer_def.get("selection", [])
        labels = prompt_enhancer_def.get("labels", {})
        if isinstance(selection, str):
            selection = [selection]
        if not isinstance(selection, list):
            selection = []
        if not isinstance(labels, dict):
            labels = {}
        labels = {**labels, **{mode.replace("V", "").replace("P", ""): label for mode, label in labels.items()}}
        return [(str(labels.get(str(mode).strip(), default_labels.get(str(mode).strip(), str(mode).strip()))), str(mode).strip()) for mode in selection if str(mode).strip()]
    selection = model_def.get("prompt_enhancer_choices_allowed", ["T"] if model_def.get("audio_only", False) else ["T", "TI"])
    if isinstance(selection, str):
        selection = [selection]
    if not isinstance(selection, list):
        selection = []
    return [(default_labels.get(str(mode).strip(), str(mode).strip()), str(mode).strip()) for mode in selection if str(mode).strip()]


def _prompt_enhancer_mode_prefixes(model_def: dict):
    image_outputs = bool(model_def.get("image_outputs", False))
    supports_image_mode = image_outputs or _supports_image_output_mode(model_def)
    supports_video_mode = not image_outputs

    def prefixes(mode: str) -> list[str]:
        if "I" not in str(mode or ""):
            return ["text"]
        ret = []
        if supports_video_mode:
            ret.append("video")
        if supports_image_mode:
            ret.append("image")
        return ret

    return prefixes


def _prompt_enhancer_profile_suffix(mode: str) -> str:
    return prompt_enhancer_chaining.profile_suffix(mode)


def _prompt_enhancer_system_label(labels: list[str]) -> str:
    compact_labels = []
    for label in labels:
        compact = re.split(r"\s+(?:using|based on|with)\s+", str(label or "").strip(), maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if compact and compact.casefold() not in {value.casefold() for value in compact_labels}:
            compact_labels.append(compact)
    detail = " / ".join(compact_labels[:2])
    return f"System Prompt \"{detail}\"" if detail else "System Prompt"


def _prompt_enhancer_system_keys(spec: dict, base_key: str) -> list[str]:
    return [f"{prefix}_prompt_enhancer_{base_key}{spec['suffix']}" for prefix in spec["prefixes"]]


def _prompt_enhancer_system_value(model_values: dict, spec: dict | None) -> str:
    if not spec:
        return ""
    for key in _prompt_enhancer_system_keys(spec, "instructions"):
        if key in model_values:
            return _format_help_value(model_values.get(key, ""))
    return ""


def _prompt_enhancer_model_def_system_value(model_def: dict, spec: dict | None) -> str:
    if not spec:
        return ""
    for key in _prompt_enhancer_system_keys(spec, "instructions"):
        value = model_def.get(key, "")
        if value:
            return _format_help_value(value)
    return ""


def _prompt_enhancer_system_tokens(model_values: dict, spec: dict | None) -> str:
    if not spec:
        return ""
    for key in _prompt_enhancer_system_keys(spec, "max_tokens"):
        if key in model_values:
            return str(_format_token_value(model_values.get(key, 0)))
    return ""


def _validate_inputs(deps, mode, original_id, source_model_type, id_text, auto_id, name, description, editable_fields, values, urls_required, source_name=""):
    problems = []
    model_id = _resolve_model_id(deps, mode, original_id, source_model_type, id_text, auto_id, name, source_name)
    if not model_id:
        problems.append("Id is required.")
    if not str(name or "").strip():
        problems.append("Name is required.")
    if not str(description or "").strip():
        problems.append("Description is required.")
    model_exists = _model_id_exists(deps, model_id)
    if mode == "creator" and model_exists:
        problems.append(f"id '{model_id}' already exists.")
    if mode == "editor" and model_id != original_id and model_exists:
        problems.append(f"id '{model_id}' already exists.")
    if urls_required and "URLs" in editable_fields and len(values["URLs"]) == 0:
        problems.append("URLs is required for this source model.")
    return problems


def _validate_finetune_param_values(specs: list[dict], values: list) -> list[str]:
    problems = []
    for spec, value in zip(specs, values):
        if value not in [choice_value for _, choice_value in spec["choices"]]:
            problems.append(f"{spec['label']}: select one of the available values.")
    return problems


def _resolve_model_id(deps, mode, original_id, source_model_type, id_text, auto_id, name=None, source_name=None):
    if mode == "creator" and auto_id:
        return _unique_model_id(deps, _auto_model_id(source_model_type, name, source_name))
    if mode == "editor" and _editor_should_auto_rename_generic_id(original_id, id_text, source_model_type):
        return _editor_auto_model_id(deps, original_id, source_model_type, name, source_name)
    return _sanitize_model_id(id_text)


def _editor_should_auto_rename_generic_id(original_id, id_text, source_model_type) -> bool:
    original_id = _sanitize_model_id(original_id)
    id_text = _sanitize_model_id(id_text)
    generic_id = _sanitize_model_id(f"{source_model_type}_finetune")
    return bool(original_id and id_text == original_id and (original_id == generic_id or original_id.startswith(f"{generic_id}_")))


def _editor_auto_model_id(deps, original_id, source_model_type, name=None, source_name=None) -> str:
    model_id = _auto_model_id(source_model_type, name, source_name)
    if model_id == _sanitize_model_id(original_id):
        return model_id
    return _unique_model_id(deps, model_id)


def _auto_model_id(source_model_type: str, name=None, source_name=None) -> str:
    base_id = _sanitize_model_id(source_model_type)
    name_text = str(name or "").strip()
    source_name = str(source_name or "").strip()
    if not name_text or name_text == source_name:
        return _sanitize_model_id(f"{base_id}_finetune")
    words = _auto_id_words(name_text)
    divergent_words = _divergent_auto_id_words(source_name, name_text)
    suffix = "_".join((divergent_words or words)[:2]).casefold()
    return _sanitize_model_id(f"{base_id}_{suffix}" if suffix else f"{base_id}_finetune")


def _divergent_auto_id_words(source_name, name) -> list[str]:
    source_words = _auto_id_words(source_name)
    words = _auto_id_words(name)
    index = _left_common_word_count(source_words, words)
    return words[index:index + 2]


def _auto_id_words(value) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(value or ""))


def _left_common_word_count(source_words: list[str], words: list[str]) -> int:
    index = 0
    while index < len(source_words) and index < len(words) and source_words[index].casefold() == words[index].casefold():
        index += 1
    return index


def _unique_model_id(deps: FinetuneEditorDeps, base_id: str) -> str:
    base_id = base_id or "finetune"
    candidate = base_id
    index = 1
    while _model_id_exists(deps, candidate):
        candidate = f"{base_id}_{index}"
        index += 1
    return candidate


def _model_id_exists(deps: FinetuneEditorDeps, model_id: str) -> bool:
    return deps.get_model_def(model_id) is not None or os.path.exists(_finetune_json_path(model_id)) or os.path.exists(os.path.join("defaults", _sanitize_model_id(model_id) + ".json"))


def _sanitize_model_id(value) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value


def _settings_source_model_type(mode: str, original_id: str, source_model_type: str) -> str:
    return original_id if mode == "editor" else source_model_type


def _settings_to_copy(deps: FinetuneEditorDeps, state, model_type: str) -> dict:
    settings = _clean_settings(deps.get_model_settings(state, model_type) or deps.get_default_settings(model_type))
    settings["settings_version"] = deps.settings_version
    return settings


def _build_finetune_json(mode, source_model_type, raw_source, raw_existing, name, description, editable_fields, values, custom_values, loras_value, loras_multipliers_value, lock_resolutions, resolutions_value, infos, prompt_infos, finetune_param_specs, finetune_param_values, enhancer_specs, enhancer_values, settings_to_copy):
    raw_output = copy.deepcopy(raw_existing if mode == "editor" and raw_existing else raw_source)
    model_section = copy.deepcopy(raw_output.get("model", {}))
    model_section.pop("finetunes_infos", None)
    model_section.pop("finetunes_params", None)
    if mode != "editor" and not model_section.get("architecture"):
        model_section["architecture"] = raw_source.get("model", {}).get("architecture", source_model_type)
    if source_model_type != model_section.get("architecture"):
        model_section[FINETUNE_SOURCE_MODEL_KEY] = source_model_type
    else:
        model_section.pop(FINETUNE_SOURCE_MODEL_KEY, None)
    model_section["name"] = str(name or "").strip()
    model_section["description"] = str(description or "").strip()
    for field in editable_fields:
        _set_optional_list_field(model_section, field, values[field])
    for key, value in custom_values.items():
        if value:
            model_section[key] = value
        else:
            model_section.pop(key, None)
    _set_optional_list_field(model_section, "loras", loras_value)
    _set_optional_list_field(model_section, "loras_multipliers", loras_multipliers_value)
    model_section.pop("resolutions_categories", None)
    model_section["lock_resolutions"] = bool(lock_resolutions)
    _set_optional_resolutions(model_section, resolutions_value)
    _set_optional_markdown(model_section, "infos", infos)
    _set_optional_markdown(model_section, "prompt_infos", prompt_infos)
    _set_finetune_params(model_section, finetune_param_specs, finetune_param_values)
    _set_optional_prompt_enhancer_systems(model_section, enhancer_specs, enhancer_values)
    _compress_model_path_fields(model_section, [*FINETUNE_URL_FIELDS, *custom_values.keys()])
    raw_output["model"] = model_section
    if settings_to_copy is not None:
        _replace_settings(raw_output, settings_to_copy)
    return raw_output


def _replace_settings(raw_output: dict, settings) -> None:
    settings_source = _clean_settings(settings)
    for key in list(raw_output.keys()):
        if key != "model":
            raw_output.pop(key, None)
    raw_output.update(settings_source)


def _clean_settings(settings):
    cleaned = copy.deepcopy(settings or {})
    cleaned.pop("model", None)
    cleaned.pop("state", None)
    cleaned.pop("model_type", None)
    cleaned.pop("base_model_type", None)
    cleaned.pop("model_filename", None)
    return cleaned


def _parse_url_value(value):
    lines = _parse_multiline(value)
    if len(lines) == 1 and lines[0].startswith("="):
        return lines[0][1:].strip()
    return [fl.compress_path(line) for line in lines]


def _validate_url_values(url_inputs: dict) -> list[str]:
    problems = []
    for field, value in url_inputs.items():
        lines = _parse_multiline(value)
        scalar_lines = [line for line in lines if line.startswith("=")]
        if scalar_lines and len(lines) != 1:
            problems.append(f"{field}: '=value' scalar mode must contain exactly one line.")
        elif len(lines) == 1 and lines[0].startswith("=") and not lines[0][1:].strip():
            problems.append(f"{field}: '=value' scalar mode requires a non-empty value.")
    return problems


def _url_fields_changed(before_model: dict, after_model: dict, keys) -> bool:
    return any(_normalized_url_value(before_model.get(key)) != _normalized_url_value(after_model.get(key)) for key in keys)


def _normalized_url_value(value):
    if isinstance(value, list):
        return [_normalized_url_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalized_url_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized_url_value(item) for key, item in value.items()}
    if isinstance(value, str):
        return fl.compress_path(value)
    return value


def _parse_multiline(value) -> list[str]:
    seen, values = set(), []
    for line in str(value or "").splitlines():
        item = line.strip()
        if not item:
            continue
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            values.append(item)
    return values


def _parse_resolution_budget_value(value):
    if value is None or str(value).strip() == "":
        return None, []
    value = str(value).strip().lower()
    try:
        parsed = int(value) if value.isdecimal() else value
        resolution_utils.resolution_pixels(parsed)
        return parsed, []
    except ValueError:
        return None, ['Resolution bounds must be positive integer pixels or "WIDTHxHEIGHT".']


def _parse_resolutions_value(value):
    resolutions = _parse_multiline(value)
    problems = [
        f"Custom Resolutions: '{resolution}' is not in WxH format."
        for resolution in resolutions
        if not resolution_utils.is_resolution_value(resolution)
    ]
    if problems:
        return [], problems
    return [[_resolution_label(resolution), resolution.lower()] for resolution in resolutions], []


def _format_list(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return f"={value}"
    if isinstance(value, (list, tuple)):
        return "\n".join(fl.compress_path(str(item)) for item in value)
    return fl.compress_path(str(value))


def _format_resolutions_value(value) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    resolutions = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            resolutions.append(str(item[1]))
        elif isinstance(item, str):
            resolutions.append(item)
    return "\n".join(resolutions)


def _resolution_label(resolution) -> str:
    width, height = resolution_utils.parse_resolution(resolution)
    ratio_width, ratio_height = _approximate_ratio(width, height)
    return f"{width}x{height} ({ratio_width}:{ratio_height})"


def _approximate_ratio(width, height, max_value=24):
    divisor = math.gcd(width, height)
    exact_width, exact_height = width // divisor, height // divisor
    if exact_width <= max_value and exact_height <= max_value:
        return exact_width, exact_height

    target_ratio = width / height
    return min(
        ((ratio_width, ratio_height) for ratio_width in range(1, max_value + 1) for ratio_height in range(1, max_value + 1) if math.gcd(ratio_width, ratio_height) == 1),
        key=lambda ratio: (abs((ratio[0] / ratio[1]) - target_ratio), ratio[0] + ratio[1]),
    )


def _format_single_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return fl.compress_path(str(value[0])) if value else ""
    return "" if value is None else fl.compress_path(str(value))


def _friendly_label(value) -> str:
    text = re.sub(r"[_-]+", " ", str(value or "")).strip()
    words = []
    for word in text.split():
        upper = word.upper()
        words.append("URLs" if upper == "URLS" else upper if upper in {"URL", "ID"} else word[:1].upper() + word[1:])
    return " ".join(words)


def _format_help_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=4, ensure_ascii=False)


def _format_token_value(value) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def _set_optional_markdown(model_section: dict, key: str, value) -> None:
    value = str(value or "").strip()
    if value:
        model_section[key] = value
    else:
        model_section.pop(key, None)


def _set_finetune_params(model_section: dict, specs: list[dict], values: list) -> None:
    for spec, value in zip(specs, values):
        model_section[spec["id"]] = value


def _set_optional_list_field(model_section: dict, key: str, values) -> None:
    if isinstance(values, str):
        if values:
            model_section[key] = values
        else:
            model_section.pop(key, None)
        return
    if len(values) > 0 or key in model_section:
        model_section[key] = values
    else:
        model_section.pop(key, None)


def _set_optional_resolutions(model_section: dict, values) -> None:
    # An empty list must also override presets inherited from the handler.
    model_section["resolutions"] = values or []


def _compress_model_path_fields(model_section: dict, keys) -> None:
    for key in keys:
        if key in model_section:
            model_section[key] = _compress_path_value(model_section[key])


def _compress_path_value(value):
    if isinstance(value, (list, tuple)):
        return [fl.compress_path(item) for item in value]
    if isinstance(value, str):
        return fl.compress_path(value)
    return value


def _set_optional_prompt_enhancer_systems(model_section: dict, specs: list[dict], values: list[tuple]) -> None:
    for spec, (instructions, max_tokens) in zip(specs, values):
        instructions = str(instructions or "").strip()
        max_tokens_text = str(max_tokens or "").strip()
        tokens = _format_token_value(max_tokens_text)
        for instructions_key in _prompt_enhancer_system_keys(spec, "instructions"):
            if instructions:
                model_section[instructions_key] = instructions
            else:
                model_section.pop(instructions_key, None)
        for tokens_key in _prompt_enhancer_system_keys(spec, "max_tokens"):
            if instructions and max_tokens_text and tokens > 0:
                model_section[tokens_key] = tokens
            else:
                model_section.pop(tokens_key, None)


def _render_source_info(source_name: str, source_model_type: str) -> str:
    return f"{str(source_name or source_model_type)} ({str(source_model_type or '')})"


def _creator_current_choice(deps: FinetuneEditorDeps, source_model_type: str):
    source_name = deps.get_model_name(source_model_type)
    return f"Using Current Model {source_name} ({source_model_type})", "current"


def _finetune_json_path(model_id: str) -> str:
    return os.path.join(FINETUNES_DIR, _sanitize_model_id(model_id) + ".json") if model_id else ""


def _rename_settings(deps, old_id, new_id):
    old_settings = deps.get_settings_file_name(old_id)
    new_settings = deps.get_settings_file_name(new_id)
    if os.path.exists(old_settings):
        os.makedirs(os.path.dirname(new_settings) or ".", exist_ok=True)
        os.replace(old_settings, new_settings)


def _rename_cached_settings(deps, state, old_id, new_id):
    all_settings = (state or {}).get("all_settings", None)
    if isinstance(all_settings, dict) and old_id in all_settings:
        all_settings[new_id] = all_settings.pop(old_id)


def _delete_cached_settings(state, model_id):
    all_settings = (state or {}).get("all_settings", None)
    if isinstance(all_settings, dict):
        all_settings.pop(model_id, None)


def _redirect_to_model_updates(deps: FinetuneEditorDeps, model_type: str):
    is_finetune = is_finetune_model(deps, model_type)
    return gr.update(visible=False), gr.update(value=f"{model_type}|{time.time()}"), gr.update(value=_finetune_json_path(model_type) if is_finetune else None, visible=is_finetune)


def _create_new_updates(deps: FinetuneEditorDeps, state, source_model_type=None):
    current_model_type = deps.get_state_model_type(state)
    *dropdowns, refresh_update = deps.refresh_model_dropdowns(state)
    return (
        *dropdowns,
        refresh_update,
        toolbar_button_updates(deps, current_model_type),
        gr.update(),
        gr.update(),
        gr.update(value=None, visible=False),
        *open_editor(deps, state, source_model_type_override=source_model_type),
    )


def _warn_parse_errors(parse_errors):
    if parse_errors:
        gr.Info("Model list refreshed, but parsing errors were found: " + parse_errors[0])


def _finetune_label(model_id, name=""):
    model_id = str(model_id or "").strip()
    name = str(name or "").strip()
    return f"{name} ({model_id})" if name else model_id


def _no_action_updates():
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _no_delete_action_updates():
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _no_create_new_action_updates(output_count: int):
    return tuple(gr.update() for _ in range(output_count))


def get_css() -> str:
    return """
.wangp-finetune-editor-popup {
    position: fixed !important;
    top: 84px !important;
    right: 32px !important;
    width: min(820px, calc(100vw - 34px)) !important;
    max-height: min(86vh, 820px) !important;
    z-index: 1200 !important;
    pointer-events: none !important;
}
.wangp-finetune-editor-popup > .form {
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.wangp-finetune-editor-card {
    pointer-events: auto !important;
    max-height: min(86vh, 820px) !important;
}
.wangp-finetune-editor-card > .form {
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    max-height: min(86vh, 820px) !important;
    min-height: 0 !important;
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: hidden !important;
}
.wangp-finetune-editor-titlebar {
    flex: 0 0 auto !important;
    margin: 0 !important;
    justify-content: space-between !important;
}
.wangp-finetune-editor-titlebar > .form {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 12px !important;
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wangp-finetune-editor-titlebar .html-container,
.wangp-finetune-editor-titlebar .prose {
    margin: 0 !important;
    padding: 0 !important;
}
.wangp-finetune-editor-intro-host {
    min-height: 44px !important;
    margin-bottom: 12px !important;
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.wangp-finetune-editor-intro-host .html-container {
    padding: 0 !important;
}
.wangp-finetune-editor-intro {
    margin: 0;
    color: var(--body-text-color, #174a67);
    line-height: 1.45;
}
.wangp-finetune-source-info {
    margin: 0 0 12px !important;
    padding: 10px 12px !important;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16)) !important;
    border-radius: 7px !important;
    background: var(--background-fill-secondary, rgba(17, 84, 118, 0.05)) !important;
}
.wangp-finetune-param-help {
    margin: -4px 0 10px !important;
    color: var(--body-text-color-subdued, #5b7282) !important;
}
.wangp-finetune-editor-content {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 14px 16px !important;
}
.wangp-finetune-editor-content > .form {
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.wangp-finetune-editor-footer {
    flex: 0 0 auto !important;
    border-top: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.16)) !important;
    background: var(--background-fill-primary, rgba(255, 255, 255, 0.99)) !important;
    padding: 10px 14px !important;
}
.wangp-finetune-editor-footer > .form {
    border: 0 !important;
    padding: 0 !important;
    gap: 8px !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wangp-finetune-editor-actions {
    justify-content: flex-end !important;
    gap: 10px !important;
    margin: 0 !important;
}
.wangp-finetune-editor-actions > .form,
.wangp-finetune-editor-id-row > .form {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wangp-finetune-editor-id-row {
    align-items: flex-end !important;
}
.wangp-finetune-editor-enhancer-field {
    position: relative !important;
    margin: 0 !important;
}
.wangp-finetune-editor-enhancer-field > .form {
    position: relative !important;
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.wangp-finetune-editor-enhancer-default-btn {
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
    font-size: 13px !important;
    line-height: 1 !important;
    border: 0 !important;
    outline: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wangp-finetune-editor-enhancer-default-btn > .form {
    border: 0 !important;
    outline: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.wangp-finetune-editor-enhancer-default-btn *,
.wangp-finetune-editor-enhancer-default-btn *::before,
.wangp-finetune-editor-enhancer-default-btn *::after {
    border-color: transparent !important;
    outline: 0 !important;
    box-shadow: none !important;
}
.wangp-finetune-editor-enhancer-default-btn button,
.wangp-finetune-editor-enhancer-default-btn > button,
.wangp-finetune-editor-enhancer-default-btn button:hover,
.wangp-finetune-editor-enhancer-default-btn > button:hover,
.wangp-finetune-editor-enhancer-default-btn button:focus,
.wangp-finetune-editor-enhancer-default-btn > button:focus,
.wangp-finetune-editor-enhancer-default-btn button:focus-visible,
.wangp-finetune-editor-enhancer-default-btn > button:focus-visible {
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
.wangp-finetune-editor-enhancer-default-btn::after {
    content: "Copy the source model system prompt";
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
.wangp-finetune-editor-enhancer-default-btn:hover::after {
    opacity: 1;
    visibility: visible;
    transition-delay: 0.5s;
}
.wangp-finetune-editor-content textarea[rows="1"] {
    overflow-y: hidden !important;
}
.wangp-finetune-editor-readonly textarea {
    background: var(--block-background-fill, rgba(0, 0, 0, 0.035)) !important;
    border-style: dashed !important;
    color: var(--body-text-color-subdued, #5b7282) !important;
    cursor: default !important;
}
#wangp_finetune_editor_close,
#wangp_finetune_editor_close button,
#wangp_finetune_editor_close > button {
    width: 26px !important;
    height: 26px !important;
    min-width: 26px !important;
    min-height: 26px !important;
    max-width: 26px !important;
    max-height: 26px !important;
    padding: 0 !important;
    flex: 0 0 26px !important;
    font-size: 14px !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wangp-finetune-editor-delete-confirm {
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wangp-markdown-editor-field {
    margin-bottom: 14px !important;
}
.wangp-markdown-editor-field > .form {
    border: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    gap: 6px !important;
    overflow: visible !important;
}
.wangp-markdown-editor-toolbar {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 0 0;
}
.wangp-markdown-editor-toolbar button {
    width: 26px;
    height: 24px;
    min-width: 26px;
    padding: 0;
    border: 1px solid var(--border-color-primary, rgba(17, 84, 118, 0.18));
    border-radius: 5px;
    background: var(--button-secondary-background-fill, var(--background-fill-secondary));
    color: var(--button-secondary-text-color, var(--body-text-color));
    font-size: 12px;
    line-height: 1;
    cursor: pointer;
}
.wangp-markdown-editor-toolbar button:hover {
    border-color: var(--button-primary-background-fill);
}
.wangp-markdown-editor {
    margin-bottom: 4px !important;
    overflow: visible !important;
}
.wangp-markdown-editor textarea {
    min-height: 220px !important;
    color: var(--input-text-color, var(--body-text-color)) !important;
    caret-color: var(--input-text-color, var(--body-text-color)) !important;
}
.wangp-markdown-editor textarea::selection {
    color: #ffffff !important;
    background: #176b8f !important;
}
"""


def get_javascript() -> str:
    return r"""
    (function () {
        if (window.__wangpFinetuneEditorSetup) return;
        window.__wangpFinetuneEditorSetup = true;
        function root() {
            if (window.gradioApp) return window.gradioApp();
            const app = document.querySelector("gradio-app");
            return app ? (app.shadowRoot || app) : document;
        }
        function installEnhancerDefaultTooltips(scope = root()) {
            const text = "Copy the system prompt defined by the source model into this field.";
            const selector = ".wangp-finetune-editor-enhancer-default-btn, .wangp-finetune-editor-enhancer-default-btn button";
            const buttons = [...scope.querySelectorAll(selector)];
            if (scope.matches?.(selector)) buttons.push(scope);
            buttons.forEach((button) => {
                button.removeAttribute("title");
                if (button.getAttribute("aria-label") !== text) button.setAttribute("aria-label", text);
            });
        }
        function markdownSnippet(action) {
            if (action === "bold") return { prefix: "**", suffix: "**", sample: "bold text" };
            if (action === "italic") return { prefix: "*", suffix: "*", sample: "italic text" };
            if (action === "heading") return { prefix: "## ", suffix: "", sample: "Heading" };
            if (action === "list") return { prefix: "- ", suffix: "", sample: "item" };
            if (action === "link") return { prefix: "[", suffix: "](https://)", sample: "link text" };
            if (action === "code") return { prefix: "`", suffix: "`", sample: "code" };
            return { prefix: "", suffix: "", sample: "" };
        }
        function closestFromEvent(event, selector) {
            for (const node of event.composedPath ? event.composedPath() : []) {
                if (node?.matches?.(selector)) return node;
                const closest = node?.closest?.(selector);
                if (closest) return closest;
            }
            return event.target?.closest?.(selector) || null;
        }
        function insertMarkdown(toolbar, action) {
            const field = toolbar.closest(".wangp-markdown-editor-field");
            const editable = field?.querySelector("textarea");
            if (!editable) return;
            const { prefix, suffix, sample } = markdownSnippet(action);
            editable.focus();
            const start = editable.selectionStart ?? editable.value.length;
            const end = editable.selectionEnd ?? start;
            const selected = editable.value.slice(start, end) || sample;
            editable.setRangeText(prefix + selected + suffix, start, end, "select");
            editable.dispatchEvent(new Event("input", { bubbles: true }));
            editable.dispatchEvent(new Event("change", { bubbles: true }));
        }
        document.addEventListener("pointerdown", (event) => {
            if (closestFromEvent(event, ".wangp-markdown-editor-toolbar button")) event.preventDefault();
        });
        document.addEventListener("click", (event) => {
            const button = closestFromEvent(event, ".wangp-markdown-editor-toolbar button");
            if (!button || !root().contains(button)) return;
            event.preventDefault();
            insertMarkdown(button.closest(".wangp-markdown-editor-toolbar"), button.getAttribute("data-wangp-md-action") || "");
        });
        installEnhancerDefaultTooltips();
        new MutationObserver(mutations => {
            for (const mutation of mutations) for (const node of mutation.addedNodes) {
                if (node.nodeType === 1) installEnhancerDefaultTooltips(node);
            }
        }).observe(root(), { childList: true, subtree: true });
    })();
    """
