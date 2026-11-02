# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Parameters of the post-processors. A processor declares them in its definition (``method_parameters``: an id in ``name``, the
argument of the processor it feeds in ``setting``, a ``component`` kind and the UI contexts in ``ui``). Each slot keeps the values of
its processor in its own dictionary under those ids, like a model's ``custom_settings`` (``spatial_postprocessing_parameters``,
``temporal_postprocessing_parameters``), shown in a fixed set of inputs per kind below the slot's selector, two per line: processors of
the two slots never share a value. Selecting another processor in a slot resets its values to the new processor's defaults, except the
normalized ids (NORMALIZED_IDS) it also declares. The reference images are one input shared by both slots
(``postprocessing_reference_images``): every processor declaring an images parameter reads it (the Object Image of a 3D refiner and a
texturer)."""

from __future__ import annotations

from typing import Any

SPATIAL, TEMPORAL = "spatial", "temporal"
SETTINGS = {SPATIAL: "spatial_postprocessing_parameters", TEMPORAL: "temporal_postprocessing_parameters"}
IMAGES_SETTING = "postprocessing_reference_images"
IMAGES = "reference_images"  # the shared reference images in a slot's runtime values (runtime_values)
# Inputs per kind of each slot. As for a model's custom settings, a number is a slider (bounded) or a text input.
SLIDERS, DROPDOWNS, TEXTBOXES = 2, 2, 1
INPUT_COUNTS = {"slider": SLIDERS, "dropdown": DROPDOWNS, "textbox": TEXTBOXES}
# Ids with one meaning for every processor declaring them: their value is kept when the slot's processor changes.
NORMALIZED_IDS = {"prompt": "a text description of the content to process"}
# Parameter names and settings before the per-slot dictionaries (settings 2.82): migrated to their ids.
LEGACY_NAMES = {"spatial_upsampler_prompt": "prompt", "spatial_upsampler_reference_images": IMAGES, "spatial_upsampler_param": "face_count",
                "spatial_upsampler_face_count": "face_count", "spatial_upsampler_h3_strength": "face_strength", "spatial_upsampler_dlss_strength": "dlss_strength",
                "spatial_upsampler_hi3d_level": "hi3d_level", "spatial_upsampler_hi3d_max_faces": "hi3d_max_faces", "temporal_upsampler_texture_size": "texture_size",
                "temporal_upsampler_max_faces": "texture_max_faces", "temporal_upsampler_texture_views": "texture_views"}
LEGACY_SETTINGS = ("postprocessing_parameters", "spatial_upsampler_parameters", "temporal_upsampler_parameters", "spatial_upsampler_prompt",
                   "spatial_upsampler_reference_images", "spatial_upsampler_param", "spatial_upsampler_param2", "spatial_upsampler_face_count")


def parameter_component_type(parameter: dict[str, Any]) -> str:
    component_type = str(parameter.get("component", "") or "").strip().lower()
    if component_type:
        return component_type
    kind = str(parameter.get("type", "string"))
    if kind in ("integer", "number"):
        return "slider" if "minimum" in parameter and "maximum" in parameter else "textbox"
    return {"boolean": "checkbox", "array": "images"}.get(kind, "textbox")


def parameter_default(parameter: dict[str, Any]):
    value = parameter.get("default", [] if parameter_component_type(parameter) == "images" else None)
    return list(value) if isinstance(value, list) else value


def _module(slot: str):
    from . import spatial_upsamplers, temporal_upsamplers
    return spatial_upsamplers if slot == SPATIAL else temporal_upsamplers


def method_of(slot: str, value) -> str:
    """The processor (method) of a slot value (``method*scale``), "" without one."""
    module = _module(slot)
    split = (module.split_upsampling_value if slot == SPATIAL else module.split_temporal_upsampling_value)(value)
    return split[0] if split else ""


def slot_parameters(slot: str, method, ui_context: str | None = None) -> list[dict[str, Any]]:
    """The parameters the slot's processor declares in ui_context (all of them without one)."""
    return _module(slot).method_parameters(method, ui_context=ui_context) if method else []


def images_parameter(*parameter_lists) -> dict[str, Any] | None:
    """The images parameter of the selected processors (the spatial slot's first), shown in the shared images input."""
    return next((parameter for parameters in parameter_lists for parameter in parameters if parameter_component_type(parameter) == "images"), None)


def parameter_value(parameter: dict[str, Any], values):
    """A parameter's value in a slot's runtime values: an images parameter reads the shared reference images."""
    return (values or {}).get(IMAGES if parameter_component_type(parameter) == "images" else str(parameter["name"]))


def runtime_values(values, images) -> dict[str, Any]:
    """A slot's values for its processor call: its dictionary and the shared reference images."""
    return {**dict(values or {}), IMAGES: images or []}


def settings_runtime_values(settings: dict, slot: str) -> dict[str, Any]:
    return runtime_values(settings.get(SETTINGS[slot]), settings.get(IMAGES_SETTING))


def migrate_settings(settings: dict) -> None:
    """Old settings: their parameter dictionaries and flat parameter settings moved to the dictionary of the slot whose selected processor
    declares them, the reference images to the shared input."""
    declared = {SPATIAL: slot_parameters(SPATIAL, method_of(SPATIAL, settings.get("spatial_upsampling"))),
                TEMPORAL: slot_parameters(TEMPORAL, method_of(TEMPORAL, settings.get("temporal_upsampling")))}
    declared = {slot: {str(parameter["name"]): parameter for parameter in parameters} for slot, parameters in declared.items()}
    # the former shared prompt belonged to the selected spatial processor: its own text parameter (the LTX Detail Refiner's refine_prompt)
    prompt = next((name for name, parameter in declared[SPATIAL].items() if parameter_component_type(parameter) == "textbox"), "prompt")
    values = {slot: dict(settings.get(setting) or {}) for slot, setting in SETTINGS.items()}
    images = settings.get(IMAGES_SETTING)
    for key in LEGACY_SETTINGS:
        value = settings.pop(key, None)
        for name, one in (value.items() if isinstance(value, dict) else [(key, value)]):
            name = prompt if name == "spatial_upsampler_prompt" else LEGACY_NAMES.get(name, name)
            if one in (None, "", []):
                continue
            if name == IMAGES:
                images = images or one
                continue
            slot = next((slot for slot in (SPATIAL, TEMPORAL) if name in declared[slot]), None)
            if slot is not None and values[slot].get(name) in (None, "", []):
                values[slot][name] = one
    for slot, setting in SETTINGS.items():
        settings[setting] = values[slot] or None
    settings[IMAGES_SETTING] = images or None


def assign(parameters) -> list:
    """The parameter shown in each input of a slot (None: hidden), inputs ordered by kind; images parameters use the shared images input."""
    inputs = {kind: [] for kind in INPUT_COUNTS}
    for parameter in parameters:
        kind = parameter_component_type(parameter)
        if kind != "images":
            inputs[kind].append(parameter)
    for kind, used in inputs.items():
        if len(used) > INPUT_COUNTS[kind]:
            raise ValueError(f"The selected post-processor shows {len(used)} {kind} parameters; a slot has {INPUT_COUNTS[kind]} {kind} inputs")
        used += [None] * (INPUT_COUNTS[kind] - len(used))
    return [parameter for kind in INPUT_COUNTS for parameter in inputs[kind]]


def ui_state(parameters, parameter_values=None) -> dict[str, Any]:
    """The inputs of these parameters, and the slot's values completed with the defaults of the shown parameters."""
    values = dict(parameter_values or {})
    for parameter in parameters:
        if parameter_component_type(parameter) != "images" and values.get(parameter["name"]) is None:
            values[parameter["name"]] = parameter_default(parameter)
    return {"inputs": assign(parameters), "values": values}


def reset(parameters, parameter_values) -> dict[str, Any]:
    """The values of a slot whose processor changed: the normalized ids the new processor declares keep theirs, the others start over."""
    declared = {str(parameter["name"]) for parameter in parameters}
    return {name: value for name, value in dict(parameter_values or {}).items() if name in NORMALIZED_IDS and name in declared}


def _properties(parameter, kind: str, values, late_postprocessing: bool) -> dict[str, Any]:
    """Label, value and kind-specific properties of an input showing parameter."""
    label = parameter.get("label", str(parameter["name"]).replace("_", " ").title())
    properties = {"label": label if late_postprocessing else parameter.get("label_long", label)}
    if kind != "images":
        properties.update(value=values[parameter["name"]], info=str(parameter.get("description", "") or "") or None)
    if kind == "slider":
        properties.update(minimum=parameter["minimum"], maximum=parameter["maximum"], step=parameter.get("step", 1))
    elif kind == "dropdown":
        properties["choices"] = parameter.get("choices", parameter.get("enum", ()))
    elif kind == "textbox":
        properties["lines"] = int(parameter.get("lines", 1))
    return properties


def _kinds() -> list[str]:
    return [kind for kind, count in INPUT_COUNTS.items() for _ in range(count)]


def build(gr, state: dict[str, Any], *, late_postprocessing: bool):
    """The inputs of a slot for a state (ui_state), two per line: (inputs, their row)."""
    inputs = []
    with gr.Row(visible=any(parameter is not None for parameter in state["inputs"]), elem_classes=["wangp-postprocessing-parameters"]) as row:
        for kind, parameter in zip(_kinds(), state["inputs"]):
            properties = {"visible": False, "label": kind.title()} if parameter is None else {"visible": True, **_properties(parameter, kind, state["values"], late_postprocessing)}
            if kind == "slider":
                properties["show_reset_button"] = False  # as the custom settings' sliders
            inputs.append({"slider": gr.Slider, "dropdown": gr.Dropdown, "textbox": gr.Textbox}[kind](elem_classes=["wangp-postprocessing-parameter"], **properties))
    return inputs, row


def updates(gr, state: dict[str, Any], *, late_postprocessing: bool) -> list:
    """Updates of a slot's inputs (build order), then of their row, for a new state."""
    output = [gr.update(visible=parameter is not None, **({} if parameter is None else _properties(parameter, kind, state["values"], late_postprocessing)))
              for kind, parameter in zip(_kinds(), state["inputs"])]
    return [*output, gr.update(visible=any(parameter is not None for parameter in state["inputs"]))]


def collect(parameters, parameter_values, input_values) -> dict[str, Any]:
    """The slot's values, those of the shown parameters read from its inputs (input_values, in build order)."""
    values = dict(parameter_values or {})
    for parameter, value in zip(assign(parameters), input_values):
        if parameter is not None:
            values[parameter["name"]] = value
    return values


def create(gr, parameters, parameter_values=None, *, triggers=(), refresh_inputs=(), parameters_for=None, previous=None, late_postprocessing: bool = False,
           update_form: bool = False, event_options=None) -> dict[str, Any]:
    """The inputs of a slot (built where it is called, below its selector) and its values (a State, or the values in a form refresh). Their
    triggers (the processor selection) refresh them from parameters_for(*refresh_inputs); previous: (the component holding the slot value
    before the selection, method_of for it), which tells a processor change that resets the values (reset)."""
    state = ui_state(parameters, parameter_values)
    inputs, row = build(gr, state, late_postprocessing=late_postprocessing)
    values = state["values"] if update_form else gr.State(state["values"])
    if not update_form:
        options = {"show_progress": "hidden", **(event_options or {})}
        count = len(refresh_inputs)

        def refresh(*args):
            new_parameters, current = parameters_for(*args[:count]), args[-1]
            if previous is not None and args[0] != previous[1](args[count]):
                current = reset(new_parameters, current)
            new_state = ui_state(new_parameters, current)
            return *updates(gr, new_state, late_postprocessing=late_postprocessing), new_state["values"]

        def store(*args):
            return collect(parameters_for(*args[:count]), args[count], args[count + 1:])

        gr.on(triggers=list(triggers), fn=refresh, inputs=[*refresh_inputs, *([previous[0]] if previous is not None else []), values], outputs=[*inputs, row, values], **options)
        gr.on(triggers=[component.input for component in inputs], fn=store, inputs=[*refresh_inputs, values, *inputs], outputs=values, **options)  # user edits only
    return {"parameters": values, "inputs": inputs, "row": row, "extra_components": [*inputs, row, values], "media_outputs": [*inputs, row, values]}


def create_slot(gr, slot: str, method, value, method_value, parameter_values=None, *, ui_context: str, late_postprocessing: bool = False, update_form: bool = False):
    """The inputs of a WanGP slot's processor parameters, below its selector (method; value: its hidden slot value)."""
    parameters_for = lambda method: slot_parameters(slot, method, ui_context)
    return create(gr, parameters_for(method_value), parameter_values, triggers=[] if update_form else [method.input], refresh_inputs=[method], parameters_for=parameters_for,
                  previous=(value, lambda previous_value: method_of(slot, previous_value)), late_postprocessing=late_postprocessing, update_form=update_form)


def images_updates(gr, parameter) -> list:
    """Updates of the shared images input (its gallery, then its row) for the images parameter of the selected processors."""
    return [gr.update(label=parameter.get("label", "Reference Images")) if parameter is not None else gr.update(), gr.update(visible=parameter is not None)]


def create_images(gr, methods, method_values, images=None, *, ui_context: str, update_form: bool = False) -> dict[str, Any]:
    """The reference images input shared by both slots (methods: the spatial and temporal selectors), shown while one of their processors
    declares an images parameter."""
    from shared.gradio.gallery import AdvancedMediaGallery

    images_for = lambda spatial_method, temporal_method: images_parameter(slot_parameters(SPATIAL, spatial_method, ui_context), slot_parameters(TEMPORAL, temporal_method, ui_context))
    parameter = images_for(*method_values)
    with gr.Row(visible=parameter is not None) as row:
        gallery = AdvancedMediaGallery(media_mode="image", height=240, columns=4, label="Reference Images" if parameter is None else parameter.get("label", "Reference Images"),
                                       initial=list(images or []), single_image_mode=False)
        gallery.mount(update_form=update_form)
    if not update_form:
        gr.on(triggers=[method.input for method in methods], fn=lambda *method_values: images_updates(gr, images_for(*method_values)), inputs=list(methods),
              outputs=[gallery.gallery, row], show_progress="hidden")
    return {"images": gallery.gallery, "row": row, "extra_components": [row, *gallery.get_toggable_elements()], "media_outputs": [gallery.gallery, row]}


def runtime_kwargs(parameters: list[dict[str, Any]], parameter_values) -> dict[str, Any]:
    """Keyword arguments of the processor call: each declared parameter with a value in the slot's runtime values (runtime_values), under
    its ``setting`` name."""
    output = {}
    for parameter in parameters:
        value = parameter_value(parameter, parameter_values)
        if value is None or value == "" or value == []:
            continue
        output[str(parameter.get("setting", parameter["name"]))] = value
    return output
