# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL_TYPES = ["t2v", "wan_finetune", "new_wan_finetune", "image", "new_image_finetune", "new_model"]


def parent_model(model_type):
    if model_type in MODEL_TYPES[:3]:
        return "t2v"
    if model_type in MODEL_TYPES[3:5]:
        return "image"
    return model_type


def model_family(model_type, for_ui=False):
    return {"t2v": "wan", "image": "flux", "new_model": "new"}[parent_model(model_type)]


def load_helpers():
    # Load the real selection functions without importing Gradio or starting inference.
    tree = ast.parse((ROOT / "shared/model_dropdowns.py").read_text(encoding="utf-8"))
    names = {"parse_model_selection", "expand_model_selection", "get_dropdown_model_types", "generate_dropdown_model_list"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"INVERT_MODEL_SELECTION_CONFIG_KEY": "invert_model_selection"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "shared/model_dropdowns.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


def make_deps(selection=(), inverted=None, current="image"):
    return SimpleNamespace(
        transformer_types=list(selection), displayed_model_types=list(MODEL_TYPES),
        transformer_type=current, server_config={} if inverted is None else {"invert_model_selection": inverted},
        get_parent_model_type=parent_model, get_base_model_type=parent_model,
        get_model_family=model_family, three_levels_hierarchy=True,
    )


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.helpers = load_helpers()

    def test_default_and_empty_selection_keep_existing_behavior(self):
        for inverted in (None, False, True):
            with self.subTest(inverted=inverted):
                self.assertEqual(self.helpers.get_dropdown_model_types(make_deps(inverted=inverted)), MODEL_TYPES)
        self.assertEqual(self.helpers.get_dropdown_model_types(make_deps(["wan_finetune"])), ["wan_finetune", "image"])

    def test_inversion_hides_selected_models_including_current(self):
        deps = make_deps(["t2v", "wan_finetune"], True, current="t2v")
        self.assertEqual(self.helpers.get_dropdown_model_types(deps), MODEL_TYPES[2:])
        self.assertEqual(deps.transformer_types, ["t2v", "wan_finetune"])
        deps.server_config["invert_model_selection"] = False
        self.assertEqual(self.helpers.get_dropdown_model_types(deps), ["t2v", "wan_finetune"])

    def test_hidden_groups_include_future_finetunes(self):
        for selection in (["1:wan"], ["2:t2v"], ["1:wan", "2:t2v", "wan_finetune"]):
            with self.subTest(selection=selection):
                self.assertEqual(self.helpers.get_dropdown_model_types(make_deps(selection, True, current="t2v")), MODEL_TYPES[3:])

    def test_all_models_can_be_hidden(self):
        self.assertEqual(self.helpers.get_dropdown_model_types(make_deps(MODEL_TYPES, True)), [])

    def test_startup_avoids_hidden_saved_or_default_model(self):
        tree = ast.parse((ROOT / "wgp.py").read_text(encoding="utf-8"))
        start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "transformer_types" for target in node.targets))
        end = next(i for i, node in enumerate(tree.body[start:], start) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "transformer_quantization" for target in node.targets))
        for current in (None, "t2v", "image"):
            with self.subTest(current=current):
                namespace = dict(
                    server_config={"transformer_types": ["1:wan"], "invert_model_selection": True, "last_model_type": current},
                    model_dropdowns=self.helpers, displayed_model_types=MODEL_TYPES,
                    get_parent_model_type=parent_model, get_model_family=model_family,
                    get_model_def=lambda model_type: {} if model_type in MODEL_TYPES else None,
                    model_types=MODEL_TYPES, models_def=dict.fromkeys(MODEL_TYPES, {}),
                    retired_model_types={}, retired_model_selection=lambda entry: entry,
                    args=SimpleNamespace(advanced=False), CONFIG_FILENAME="unused.json",
                )
                exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]), "wgp.py", "exec"), namespace)
                self.assertEqual(namespace["transformer_type"], "image")

    def test_saved_selection_dispatches_real_model_switch(self):
        tree = ast.parse((ROOT / "plugins/configuration/plugin.py").read_text(encoding="utf-8"))
        callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "switch_to_visible_model")
        no_update = object()
        namespace = {"gr": SimpleNamespace(update=lambda: no_update), "self": SimpleNamespace(_model_choice_target_value=lambda model_type: f"target:{model_type}")}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "plugins/configuration/plugin.py", "exec"), namespace)
        switch = namespace["switch_to_visible_model"]
        self.assertEqual(switch({"model_type": "t2v"}, "image"), "target:image")
        self.assertIs(switch({"model_type": "t2v"}, "t2v"), no_update)
        self.assertIs(switch({"model_type": "t2v"}, None), no_update)

    def configure_renderer(self):
        namespace = self.helpers.generate_dropdown_model_list.__globals__
        namespace.update(
            gr=SimpleNamespace(Dropdown=lambda **kwargs: kwargs),
            get_model_output_filter=lambda *args: "video",
            get_filtered_dropdown_model_types=lambda deps, model_filter, types: types,
            get_sorted_dropdown=lambda deps, types, family, current, three_levels: ([(family, family)], [(parent_model(current), parent_model(current))], [(current, current)]),
            get_family_dropdown_model_types=lambda deps, family, types: types,
            get_model_download_status_maps=lambda *args: ({}, {}),
            decorate_dropdown_choices_with_status=lambda choices, statuses: choices,
            decorate_finetune_dropdown_choices=lambda deps, choices: choices,
        )
    def test_rendering_does_not_restore_hidden_active_model(self):
        self.configure_renderer()
        dropdowns = self.helpers.generate_dropdown_model_list(make_deps(["1:wan"], True, current="t2v"), "t2v", select_visible_model=True)
        self.assertEqual(dropdowns[2]["value"], "image")
        dropdowns = self.helpers.generate_dropdown_model_list(make_deps(MODEL_TYPES, True), "image", select_visible_model=True)
        for dropdown in dropdowns:
            self.assertEqual(dropdown["choices"], [])
            self.assertIsNone(dropdown["value"])

    def test_explicit_hidden_target_keeps_selector_state_and_form_consistent(self):
        self.configure_renderer()
        tree = ast.parse((ROOT / "wgp.py").read_text(encoding="utf-8"))
        switch = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "switch_model_form")
        parse_target = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_model_choice_target_model_type")
        for selection in (["1:wan"], MODEL_TYPES):
            with self.subTest(selection=selection):
                deps = make_deps(selection, True)
                state = {"model_type": "image"}

                def change_model(state, model_type, refresh_header=False):
                    state["model_type"] = model_type
                    return model_type, None

                namespace = dict(
                    prompt_vars=[], prompt_save_index=0, finetune_outputs=[],
                    validate_wizard_prompt=lambda *args: "prompt", save_inputs=lambda *args: None,
                    goto_model_type_fn=lambda state, target: (*self.helpers.generate_dropdown_model_list(deps, target.split("|")[0], state), None),
                    change_model=change_model, fill_inputs=lambda state: [state["model_type"]],
                )
                exec(compile(ast.Module(body=[parse_target, switch], type_ignores=[]), "wgp.py", "exec"), namespace)
                updates = namespace["switch_model_form"](state, "t2v|123", False, {}, "prompt", "", state, "prompt")
                self.assertEqual(updates[2]["value"], "t2v")
                self.assertEqual(updates[2]["value"], state["model_type"])
                self.assertEqual(updates[-1], state["model_type"])


if __name__ == "__main__":
    unittest.main()
