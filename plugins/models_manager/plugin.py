# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import gradio as gr
import os
import json
import html
import stat
from collections import defaultdict
from shared.utils.plugins import WAN2GPPlugin
from shared.utils import files_locator as fl
from shared import model_dropdowns
from shared.gradio.app_icons import icon as app_icon
from mmgp import quant_router

VARIANT_LABEL_OTHER_NON_SHARED = "Other Non Shared"
VARIANT_LABEL_OTHER_SHARED = "Other Shared"
VARIANT_LABEL_MISSING = "Missing"
VARIANT_LABEL_RETIRED = "Older Releases"


class modelsManagerPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self._node_map = {}
        self._system_entries = {}
        self._tree_data = []
        self._model_files = {}
        self._model_primary = {}
        self._model_variants = {}
        self._variant_to_model = {}
        self._model_base_label = {}
        self._model_family_label = {}
        self._model_display_label = {}
        self._model_direct_status_map = {}
        self._model_aggregated_status_map = {}
        self._model_usage_ids = {}
        self._usage_files = {}
        self._file_usage = {}
        self._file_usage_models = {}
        self._file_info = {}
        self._basename_index = {}
        self._stats_total = 0
        self._stats_by_drive = []
        self._errors = []
        self._repo_root = os.path.abspath(os.getcwd())
        self._expanded_nodes = set()
        self._current_tree_mode = "full"
        self._current_filter_text = ""
        self._last_rendered_nodes_count = 0

    def setup_ui(self):
        self.request_global("get_model_def")
        self.request_global("get_model_name")
        self.request_global("get_model_family")
        self.request_global("get_parent_model_type")
        self.request_global("get_base_model_type")
        self.request_global("get_model_handler")
        self.request_global("get_model_recursive_prop")
        self.request_global("get_model_filename")
        self.request_global("get_lora_dir")
        self.request_global("get_lora_local_path")
        self.request_global("compact_name")
        self.request_global("create_models_hierarchy")
        self.request_global("families_infos")
        self.request_global("models_def")
        self.request_global("server_config")
        self.request_global("transformer_quantization")
        self.request_global("transformer_dtype_policy")
        self.request_global("text_encoder_quantization")
        self.request_global("displayed_model_types")
        self.request_global("transformer_types")
        self.request_global("get_transformer_dtype")
        self.request_global("app")

        self.add_custom_js(self._get_js())

        self.add_tab(
            tab_id="model_manager",
            label="Models Manager",
            component_constructor=self.create_model_ui,
        )

    def create_model_ui(self):
        with gr.Column():
            gr.HTML(f'<div class="ckpt-heading"><div><h3>Models Manager</h3><p>Shared files are used by multiple models or variants and count once in each total.</p></div><button type="button" class="wangp-model-manager-link" data-app-route="plugin_configuration" data-ckpt-return title="Back to Configuration [Alt+M]">{app_icon("back")} <span>Back to Configuration</span> <kbd>Alt+M</kbd></button></div>')
            with gr.Row():
                with gr.Column(scale=5):
                    self.stats_html = gr.HTML()
                self.refresh_button = gr.Button("Refresh", variant="secondary", elem_id="ckpt_refresh_btn", scale=1, elem_classes="btn_centered")
            with gr.Tabs(visible=False) as self.inventory_tabs:
                with gr.Tab("Models"):
                    with gr.Row():
                        self.filter_input = gr.Textbox(label="Filter Finetunes", show_label=False, placeholder="Type at least 3 characters from a model name", lines=1, scale=5, visible=False, elem_id="ckpt_filter_input")
                        self.filter_button = gr.Button("Filter", variant="primary", scale=1, visible=False, elem_id="ckpt_filter_btn", elem_classes="btn_centered")
                    with gr.Row():
                        with gr.Column(scale=3, min_width=550):
                            self.tree_html = gr.HTML()
                        with gr.Column(scale=2, min_width=350, elem_classes="ckpt-right-column"):
                            self.view_html = gr.HTML()
                with gr.Tab("System"):
                    with gr.Row():
                        with gr.Column(scale=3, min_width=550):
                            self.system_tree = gr.HTML()
                        with gr.Column(scale=2, min_width=350, elem_classes="ckpt-right-column"):
                            self.system_view = gr.HTML()

            with gr.Column(visible=False):
                self.action_input = gr.Textbox(elem_id="model_action_input")

        model_outputs = [self.tree_html, self.view_html, self.refresh_button, self.filter_input, self.filter_button]
        self.on_tab_outputs = [*model_outputs, self.stats_html, self.system_tree, self.system_view, self.inventory_tabs]
        self.refresh_button.click(
            fn=self._show_loading,
            inputs=[],
            outputs=self.on_tab_outputs,
            show_progress="hidden",
        ).then(
            fn=self._build_tree,
            inputs=[],
            outputs=self.on_tab_outputs,
            show_progress="hidden",
        ).then(
            fn=lambda: gr.update(value=""),
            inputs=[],
            outputs=[self.filter_input],
            show_progress="hidden",
        )

        self.filter_button.click(
            fn=self._build_filtered_tree,
            inputs=[self.filter_input],
            outputs=model_outputs,
            show_progress="full",
        )

        self.action_input.change(
            fn=self._handle_action,
            inputs=[self.action_input],
            outputs=self.on_tab_outputs,
            show_progress="hidden",
        )

        return self.tree_html

    def on_tab_select(self, state: dict):
        return self._show_loading()

    def _get_js(self):
        return """
        function modelRoot() {
            if (window.gradioApp) {
                return window.gradioApp();
            }
            const app = document.querySelector('gradio-app');
            return app ? (app.shadowRoot || app) : document;
        }

        window.updatemodelInput = function(elem_id, value) {
            const root = modelRoot();
            const input = root.querySelector(`#${elem_id} textarea, #${elem_id} input`);
            if (input) {
                input.value = value;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                return true;
            }
            return false;
        };

        function ensuremodelModal() {
            let backdrop = document.getElementById('ckpt-delete-modal');
            if (backdrop) return backdrop;
            backdrop = document.createElement('div');
            backdrop.id = 'ckpt-delete-modal';
            backdrop.className = 'ckpt-modal-backdrop';
            backdrop.innerHTML = `
                <div class="ckpt-modal">
                    <div class="ckpt-modal-title">Confirm Delete</div>
                    <div class="ckpt-modal-body" id="ckpt-delete-modal-body"></div>
                    <div class="ckpt-modal-actions">
                        <button class="ckpt-modal-btn" id="ckpt-delete-cancel">Cancel</button>
                        <button class="ckpt-modal-btn danger" id="ckpt-delete-exclusive">Delete Exclusive</button>
                        <button class="ckpt-modal-btn danger" id="ckpt-delete-all">Delete All</button>
                    </div>
                </div>
            `;
            document.body.appendChild(backdrop);
            const cancelBtn = backdrop.querySelector('#ckpt-delete-cancel');
            const exclusiveBtn = backdrop.querySelector('#ckpt-delete-exclusive');
            const allBtn = backdrop.querySelector('#ckpt-delete-all');
            cancelBtn.addEventListener('click', () => {
                backdrop.style.display = 'none';
                backdrop.dataset.payloadExclusive = '';
                backdrop.dataset.payloadAll = '';
            });
            exclusiveBtn.addEventListener('click', () => {
                const payload = backdrop.dataset.payloadExclusive || '';
                if (payload) {
                    window.updatemodelInput("model_action_input", payload);
                }
                backdrop.style.display = 'none';
                backdrop.dataset.payloadExclusive = '';
                backdrop.dataset.payloadAll = '';
            });
            allBtn.addEventListener('click', () => {
                const payload = backdrop.dataset.payloadAll || '';
                if (payload) {
                    window.updatemodelInput("model_action_input", payload);
                }
                backdrop.style.display = 'none';
                backdrop.dataset.payloadExclusive = '';
                backdrop.dataset.payloadAll = '';
            });
            backdrop.addEventListener('click', (evt) => {
                if (evt.target === backdrop) {
                    backdrop.style.display = 'none';
                    backdrop.dataset.payloadExclusive = '';
                    backdrop.dataset.payloadAll = '';
                }
            });
            return backdrop;
        }

        function showmodelModal(message, payloadExclusive, payloadAll, exclusiveSize, totalSize) {
            const backdrop = ensuremodelModal();
            const body = backdrop.querySelector('#ckpt-delete-modal-body');
            const exclusiveBtn = backdrop.querySelector('#ckpt-delete-exclusive');
            const allBtn = backdrop.querySelector('#ckpt-delete-all');
            body.textContent = message;
            backdrop.dataset.payloadExclusive = payloadExclusive || '';
            backdrop.dataset.payloadAll = payloadAll || '';
            exclusiveBtn.textContent = exclusiveSize ? `Delete Exclusive (${exclusiveSize})` : 'Delete Exclusive';
            allBtn.textContent = totalSize ? `Delete All (${totalSize})` : 'Delete All';
            backdrop.style.display = 'flex';
        }

        window.handlemodelAction = function(button, action) {
            if (!button) return;
            const nodeId = button.dataset.nodeId;
            const nodeLabel = button.dataset.nodeLabel || nodeId;
            const exclusiveSize = button.dataset.exclusiveSize || "0 B";
            const totalSize = button.dataset.totalSize || exclusiveSize;
            if (!nodeId) return;
            if (action === "delete") {
                const payloadExclusive = JSON.stringify({
                    action: action,
                    node_id: nodeId,
                    delete_shared: false,
                    ts: Date.now()
                });
                const payloadAll = JSON.stringify({
                    action: action,
                    node_id: nodeId,
                    delete_shared: true,
                    ts: Date.now()
                });
                showmodelModal(
                    `Delete Exclusive keeps files used outside "${nodeLabel}". Delete All removes those files too.`,
                    payloadExclusive,
                    payloadAll,
                    exclusiveSize,
                    totalSize
                );
                return;
            }
            const payload = JSON.stringify({
                action: action,
                node_id: nodeId,
                ts: Date.now()
            });
            window.updatemodelInput("model_action_input", payload);
        };

        window.handlemodelRowClick = function(evt, row) {
            if (evt) {
                evt.stopPropagation();
                evt.preventDefault();
            }
            if (!row) return;
            const nodeId = row.dataset.nodeId;
            if (!nodeId) return;
            const payload = JSON.stringify({
                action: "view",
                node_id: nodeId,
                ts: Date.now()
            });
            window.updatemodelInput("model_action_input", payload);
        };

        window.handlemodelSummaryClick = function(evt, summary) {
            if (!summary) return;
            if (evt) {
                evt.stopPropagation();
                evt.preventDefault();
            }
            const details = summary.closest('details');
            if (!details) return;
            const nodeId = details.dataset.nodeId;
            if (!nodeId) return;
            const row = summary.querySelector('.ckpt-row');
            const nodeType = row ? (row.dataset.nodeType || '') : '';
            const payload = JSON.stringify({
                action: "toggle_node",
                node_id: nodeId,
                open: !details.open,
                view: nodeType === "model",
                ts: Date.now()
            });
            window.updatemodelInput("model_action_input", payload);
        };

        function ismodelTabActive() {
            const root = modelRoot();
            const tabs = root.querySelectorAll('button[role="tab"]');
            for (const tab of tabs) {
                if (tab.getAttribute('aria-selected') === 'true') {
                    const label = (tab.textContent || '').trim();
                    if (label === 'Models Manager') return true;
                }
            }
            return false;
        }

        function triggermodelRefresh() {
            const root = modelRoot();
            const btn = root.querySelector('#ckpt_refresh_btn button') || root.querySelector('#ckpt_refresh_btn');
            if (btn) btn.click();
        }

        function triggermodelFilter() {
            const root = modelRoot();
            const btn = root.querySelector('#ckpt_filter_btn button') || root.querySelector('#ckpt_filter_btn');
            if (btn) btn.click();
        }

        function setupmodelFilterEnter() {
            const root = modelRoot();
            const input = root.querySelector('#ckpt_filter_input textarea, #ckpt_filter_input input');
            if (!input || input.dataset.ckptEnterBound === '1') return;
            input.dataset.ckptEnterBound = '1';
            input.addEventListener('keydown', (evt) => {
                if (evt.key !== 'Enter') return;
                evt.preventDefault();
                evt.stopPropagation();
                triggermodelFilter();
            });
        }


        let ckptTabWasActive = false;
        let ckptRefreshBusy = false;
        let ckptRefreshStartedAt = 0;

        function waitForRefreshReady() {
            const root = modelRoot();
            const treeRoot = root.querySelector('.ckpt-stats');
            if (treeRoot && treeRoot.dataset.status === 'ready') {
                ckptRefreshBusy = false;
                setupmodelFilterEnter();
                return;
            }
            if (ckptRefreshStartedAt && Date.now() - ckptRefreshStartedAt > 60000) {
                ckptRefreshBusy = false;
                setupmodelFilterEnter();
                return;
            }
            setTimeout(waitForRefreshReady, 400);
        }

        function startmodelBuild() {
            if (ckptRefreshBusy) return;
            ckptRefreshBusy = true;
            ckptRefreshStartedAt = Date.now();
            triggermodelRefresh();
            setTimeout(waitForRefreshReady, 200);
        }

        // The screen the Models Manager was opened from, where its return link and Alt+M lead back.
        let ckptReturn = {target: 'plugin_configuration', label: 'Configuration'};

        function updatemodelReturnLink() {
            const link = modelRoot().querySelector('[data-ckpt-return]');
            link.dataset.appRoute = ckptReturn.target;
            link.title = `Back to ${ckptReturn.label} [Alt+M]`;
            link.querySelector('span').textContent = `Back to ${ckptReturn.label}`;
        }

        function handleTabChange() {
            const active = ismodelTabActive();
            if (!active) {
                const root = modelRoot();
                const tab = root.querySelector('.wangp-native-app-tabs > .tab-wrapper button[data-tab-id][aria-selected="true"]');
                if (tab) {
                    const appLink = root.querySelector(`#wangp-app-bar [data-app-route="${CSS.escape(tab.dataset.tabId)}"]`);
                    ckptReturn = {target: tab.dataset.tabId, label: appLink ? appLink.getAttribute('aria-label') : tab.textContent.trim()};
                }
            }
            if (active && !ckptTabWasActive) {
                updatemodelReturnLink();
                startmodelBuild();
            }
            if (active) {
                setTimeout(setupmodelFilterEnter, 280);
            }
            ckptTabWasActive = active;
        }

        function initmodelBridge() {
            const root = modelRoot();
            const tabs = root.querySelectorAll('button[role="tab"]');
            if (!tabs || tabs.length === 0) {
                setTimeout(initmodelBridge, 400);
                return;
            }
            const tabObserver = new MutationObserver(handleTabChange);
            tabObserver.observe(root, { attributes: true, subtree: true, attributeFilter: ['aria-selected'] });
            handleTabChange();
        }

        document.addEventListener('keydown', (evt) => {
            if (!evt.altKey || evt.ctrlKey || evt.metaKey || evt.shiftKey || evt.repeat || evt.key.toLowerCase() !== 'm') return;
            evt.preventDefault();
            window.WanGPAppBar?.navigate(ismodelTabActive() ? ckptReturn.target : 'plugin_model_manager');
        });

        initmodelBridge();
        """

    def _show_loading(self):
        tree_loading = self._build_loading_html()
        view_hidden = gr.update(value="", visible=False)
        refresh_hidden = gr.update(visible=False)
        filter_input_hidden = gr.update(visible=False)
        filter_button_hidden = gr.update(visible=False)
        return "", view_hidden, refresh_hidden, filter_input_hidden, filter_button_hidden, tree_loading, "", "", gr.update(visible=False)

    def _build_tree(self):
        self._build_cache()
        self._expanded_nodes = set()
        self._current_tree_mode = "full"
        self._current_filter_text = ""
        tree_html = self._build_tree_html()
        view_html = self._build_empty_view_html()
        view_visible = gr.update(value=view_html, visible=True)
        refresh_visible = gr.update(visible=True)
        filter_input_visible = gr.update(visible=True)
        filter_button_visible = gr.update(visible=True)
        stats = self._tree_styles() + self._build_stats_html()
        return tree_html, view_visible, refresh_visible, filter_input_visible, filter_button_visible, stats, self._build_system_tree_html(), view_html, gr.update(visible=True)

    def _build_filtered_tree(self, filter_text):
        text = str(filter_text or "").strip()
        if len(text) < 3:
            gr.Info("Please enter at least 3 characters to filter finetunes.")
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        if not self._node_map:
            return gr.skip()
        self._expanded_nodes = set()
        self._current_tree_mode = "filtered"
        self._current_filter_text = text
        tree_html = self._build_filtered_tree_html(text)
        view_html = self._build_empty_view_html()
        view_visible = gr.update(value=view_html, visible=True)
        refresh_visible = gr.update(visible=True)
        filter_input_visible = gr.update(visible=True)
        filter_button_visible = gr.update(visible=True)
        return tree_html, view_visible, refresh_visible, filter_input_visible, filter_button_visible

    def _handle_action(self, payload: str):
        if not payload:
            yield gr.skip()
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            yield gr.skip()
            return

        action = data.get("action")
        node_id = data.get("node_id")
        if node_id not in self._node_map:
            yield gr.skip()
            return

        if action == "view":
            view_html = self._build_view_html(node_id)
            yield {self.system_view if node_id.startswith("system::") else self.view_html: view_html}
        elif action == "toggle_node":
            if data.get("open"):
                self._expanded_nodes.add(node_id)
            else:
                self._expanded_nodes.discard(node_id)
            if self._current_tree_mode == "filtered" and len(self._current_filter_text) >= 3:
                tree_html = self._build_filtered_tree_html(self._current_filter_text)
            else:
                tree_html = self._build_tree_html()
            if data.get("view"):
                view_html = self._build_view_html(node_id)
                yield {self.tree_html: tree_html, self.view_html: view_html}
            else:
                yield {self.tree_html: tree_html}
        elif action == "delete":
            yield self._show_loading()
            delete_shared = bool(data.get("delete_shared"))
            removed, missing, errors = self._delete_files_for_node(
                node_id, delete_shared=delete_shared
            )
            if errors:
                gr.Warning(f"Delete completed with errors: {len(errors)}")
            else:
                gr.Info(f"Deleted {len(removed)} files (missing: {len(missing)})")
            yield self._build_tree()
        else:
            yield gr.skip()

    def _delete_files_for_node(self, node_id, delete_shared=False):
        node = self._node_map.get(node_id)
        if node is None:
            return [], [], []
        removed = []
        missing = []
        errors = []
        if delete_shared:
            target_files = node.get("files", set())
        else:
            target_files = node["exclusive_files"]
        for path in sorted(target_files):
            if not os.path.isfile(path):
                missing.append(path)
                continue
            try:
                os.remove(path)
                removed.append(path)
            except OSError as exc:
                errors.append((path, str(exc)))
        return removed, missing, errors

    def _build_cache(self):
        self._node_map = {}
        self._tree_data = []
        self._model_files = {}
        self._model_primary = {}
        self._model_variants = {}
        self._variant_to_model = {}
        self._model_base_label = {}
        self._model_family_label = {}
        self._model_display_label = {}
        self._model_direct_status_map = {}
        self._model_aggregated_status_map = {}
        self._model_usage_ids = {}
        self._usage_files = {}
        self._file_usage = defaultdict(set)
        self._file_usage_models = defaultdict(set)
        self._file_info = {}
        self._basename_index = {}
        self._errors = []
        self._stats_total = 0
        self._stats_by_drive = []

        copies_by_path = {}
        model_types = self._get_model_types()
        self._model_direct_status_map, self._model_aggregated_status_map = self._compute_model_status_maps(model_types)
        for model_type in model_types:
            try:
                variant_info = self._collect_model_variants(model_type)
            except Exception as exc:
                self._errors.append(f"{model_type}: {exc}")
                variant_info = {"variants": [], "primary_paths": [], "files": set()}
            variants = variant_info.get("variants", [])
            model_files = set()
            for variant in variants:
                copies = set()
                for path in variant.get("files", ()):
                    if path not in copies_by_path:
                        relative = fl.compress_path(path)
                        copies_by_path[path] = {self._normalize_path(p) for p in fl.installed_copies(relative)} or {path}
                    copies.update(copies_by_path[path])
                variant["files"] = copies
                model_files.update(copies)
            self._model_variants[model_type] = variants
            self._model_primary[model_type] = variant_info.get("primary_paths", [])
            self._model_files[model_type] = model_files
            for path in model_files:
                self._file_usage_models[path].add(model_type)

        self._system_entries = self._collect_system_entries()
        linked_paths = defaultdict(list)
        for path in self._file_usage_models:
            try:
                info = os.stat(path)
                exists = stat.S_ISREG(info.st_mode)
            except OSError:
                exists = False
            file_id = (info.st_dev, info.st_ino) if exists and info.st_ino else path
            self._file_info[path] = {"size": info.st_size if exists else 0, "exists": exists, "id": file_id}
            if exists:
                linked_paths[file_id].append(path)
        for paths in linked_paths.values():
            if len(paths) > 1:
                owners = set().union(*(self._file_usage_models[path] for path in paths))
                for path in paths:
                    self._file_usage_models[path].update(owners)
        self._rebalance_other_shared_variants(model_types)

        self._usage_files = {}
        self._file_usage = defaultdict(set)
        self._variant_to_model = {}
        self._model_usage_ids = {}
        for model_type in model_types:
            usage_ids = set()
            for variant in self._model_variants.get(model_type, []):
                variant_id = variant["id"]
                usage_ids.add(variant_id)
                self._variant_to_model[variant_id] = model_type
                files = variant.get("files", set())
                self._usage_files[variant_id] = files
                for path in files:
                    self._file_usage[path].add(variant_id)
            self._model_usage_ids[model_type] = usage_ids

        self._compute_global_stats()
        self._build_tree_structure(model_types)
        self._build_system_tree()

    def _build_dropdown_deps(self, model_types):
        if not model_types:
            return None
        fallback_type = model_types[0]
        return model_dropdowns.DropdownDeps(
            transformer_types=list(model_types),
            displayed_model_types=list(model_types),
            transformer_type=fallback_type,
            families_infos=self.families_infos,
            server_config=self.server_config,
            transformer_quantization=self.transformer_quantization,
            transformer_dtype_policy=self.transformer_dtype_policy,
            text_encoder_quantization=self.text_encoder_quantization,
            get_model_def=self.get_model_def,
            get_model_recursive_prop=self.get_model_recursive_prop,
            get_model_filename=self.get_model_filename,
            get_local_model_filename=self._get_local_model_filename,
            get_lora_dir=self.get_lora_dir,
            get_lora_local_path=self.get_lora_local_path,
            get_parent_model_type=self.get_parent_model_type,
            get_model_family=self.get_model_family,
            get_model_name=self.get_model_name,
            get_transformer_dtype=self.get_transformer_dtype,
        )

    def _resolve_expected_entry_path(self, entry, model_type=None):
        if not isinstance(entry, dict):
            return None
        local_path = entry.get("path", None)
        if isinstance(local_path, str) and len(local_path) > 0:
            return self._normalize_path(local_path)
        filename = entry.get("filename", "")
        extra_paths = entry.get("extra_paths", None)
        return self._resolve_path(filename, force_folder=extra_paths, lora_dir=self._safe_get_lora_dir(model_type))

    def _collect_expected_missing_files(self, model_type):
        deps = self._build_dropdown_deps([model_type])
        if deps is None:
            return set()
        missing_paths = set()
        try:
            expected_entries = []
            expected_entries.extend(model_dropdowns.get_expected_core_file_entries_for_status(deps, model_type))
            expected_entries.extend(model_dropdowns.get_expected_secondary_file_entries_for_status(deps, model_type))
        except Exception as exc:
            self._errors.append(f"Missing file check failed for {model_type}: {exc}")
            return set()

        model_def = self.get_model_def(model_type)
        if isinstance(model_def, dict):
            try:
                for resolved_path in self._collect_handler_file_paths(model_type, model_def):
                    if resolved_path and not os.path.isfile(resolved_path):
                        missing_paths.add(resolved_path)
            except Exception as exc:
                self._errors.append(f"Missing handler file check failed for {model_type}: {exc}")

        for entry in expected_entries:
            resolved_path = self._resolve_expected_entry_path(entry, model_type=model_type)
            if resolved_path and not os.path.isfile(resolved_path):
                missing_paths.add(resolved_path)
        return missing_paths

    def _compute_model_status_maps(self, model_types):
        if not model_types:
            return {}, {}
        try:
            deps = self._build_dropdown_deps(model_types)
            if deps is None:
                return {}, {}
            direct_status_map, aggregated_parent_status_map = model_dropdowns.get_model_download_status_maps(
                deps, dropdown_types=list(model_types)
            )
            return direct_status_map, aggregated_parent_status_map
        except Exception as exc:
            self._errors.append(f"Model status map failed: {exc}")
            return {}, {}

    def _decorate_status_label(self, label, status):
        return model_dropdowns.decorate_model_dropdown_label(label, status)

    def _compute_global_stats(self):
        total = 0
        by_drive = defaultdict(int)
        seen = set()
        for path, info in self._file_info.items():
            size = info.get("size", 0)
            if size <= 0 or info["id"] in seen:
                continue
            seen.add(info["id"])
            total += size
            drive = self._get_drive_label(os.path.realpath(path))
            by_drive[drive] += size
        self._stats_total = total
        self._stats_by_drive = sorted(
            by_drive.items(), key=lambda item: (-item[1], item[0])
        )

    def _get_drive_label(self, path):
        drive, _ = os.path.splitdrive(path)
        if drive:
            return drive.rstrip(":").upper()
        if path.startswith("\\\\"):
            return "UNC"
        return "ROOT"

    def _get_model_types(self):
        types = []
        if isinstance(getattr(self, "models_def", None), dict) and self.models_def:
            types = sorted(self.models_def.keys())
        elif isinstance(self.transformer_types, list) and self.transformer_types:
            types = model_dropdowns.expand_model_selection(self.transformer_types, self.displayed_model_types, self.get_parent_model_type, self.get_model_family)
        elif isinstance(self.displayed_model_types, list):
            types = list(self.displayed_model_types)
        ordered = []
        for model_type in dict.fromkeys(filter(None, types)):
            model_def = self.get_model_def(model_type)
            if model_def is None:
                self._errors.append(f"Model type '{model_type}' was not found.")
                continue
            architecture = model_def.get("architecture")
            source = model_def.get("path", model_type)
            if not architecture:
                self._errors.append(f"Model definition '{source}' does not specify an architecture.")
            elif self.get_model_def(architecture) is None:
                self._errors.append(f"Model type '{architecture}' was not found. Check the architecture in '{source}'.")
            else:
                ordered.append(model_type)
        return ordered

    def _collect_model_variants(self, model_type):
        model_def = self.get_model_def(model_type)
        if model_def is None:
            return {"variants": [], "primary_paths": [], "files": set()}

        default_dtype_policy = self._resolve_default_dtype_policy(model_type, model_def)
        transformer_variants = self._collect_transformer_variants(
            model_type, model_def, default_dtype_policy
        )
        module_variants = self._collect_module_variants(
            model_type, model_def, default_dtype_policy
        )
        text_encoder_variants = self._collect_text_encoder_variants(
            model_type, model_def, default_dtype_policy
        )
        lora_variants = self._collect_lora_variants(model_type)
        transformer_files = set()
        module_files = set()
        text_encoder_files = set()
        lora_files = set()
        model_primary_paths = []
        for variant in transformer_variants:
            transformer_files.update(variant["files"])
            model_primary_paths.extend(variant.get("primary_paths", []))
        for variant in module_variants:
            module_files.update(variant["files"])
            model_primary_paths.extend(variant.get("primary_paths", []))
        for variant in text_encoder_variants:
            text_encoder_files.update(variant["files"])
            model_primary_paths.extend(variant.get("primary_paths", []))
        for variant in lora_variants:
            lora_files.update(variant["files"])
            model_primary_paths.extend(variant.get("primary_paths", []))
        model_primary_paths = self._dedupe_paths(model_primary_paths)

        other_files, shared_files = self._collect_other_files(
            model_type,
            model_def,
            transformer_files.union(module_files),
            text_encoder_files,
            lora_files,
        )
        variants = list(transformer_variants) + list(module_variants) + list(text_encoder_variants) + list(lora_variants)
        if other_files:
            other_id = self._make_variant_id(model_type, "other", "other")
            variants.append(
                {
                    "id": other_id,
                    "label": VARIANT_LABEL_OTHER_NON_SHARED,
                    "type": "variant",
                    "files": other_files,
                    "primary_paths": [],
                }
            )
        if shared_files:
            shared_id = self._make_variant_id(model_type, "shared", "shared")
            variants.append(
                {
                    "id": shared_id,
                    "label": VARIANT_LABEL_OTHER_SHARED,
                    "type": "variant",
                    "files": shared_files,
                    "primary_paths": [],
                }
            )
        retired_files = self._collect_retired_files(model_type)
        if retired_files:
            variants.append({"id": self._make_variant_id(model_type, "retired", "retired"), "label": VARIANT_LABEL_RETIRED, "type": "variant", "files": retired_files, "primary_paths": []})

        model_files = set()
        for variant in variants:
            model_files.update(variant.get("files", set()))
        missing_files = self._collect_expected_missing_files(model_type)
        if missing_files:
            missing_id = self._make_variant_id(model_type, "missing", "missing")
            variants.append(
                {
                    "id": missing_id,
                    "label": VARIANT_LABEL_MISSING,
                    "type": "missing",
                    "files": set(),
                    "missing_files": missing_files,
                    "primary_paths": [],
                }
            )

        return {
            "variants": variants,
            "primary_paths": model_primary_paths,
            "files": model_files,
        }

    def _collect_transformer_variants(
        self, model_type, model_def, default_dtype_policy
    ):
        choice_lists, modules = self._collect_transformer_choice_lists(model_type, model_def)
        base_dtypes, token_dtypes, token_labels = self._collect_variant_candidates(
            choice_lists, default_dtype_policy
        )
        variants = []
        used_labels = set()

        base_default = self._format_dtype_label(default_dtype_policy)
        if base_dtypes:
            for dtype_policy in sorted(base_dtypes):
                label = self._format_variant_label("", dtype_policy, base_default)
                files, primary_paths = self._collect_transformer_files_for_variant(
                    model_type, model_def, modules, "", dtype_policy
                )
                if not files:
                    continue
                file_label = self._detect_quant_label_from_files(files)
                label = file_label or label
                variant_label = f"Transformer {label}"
                variant_id = self._make_variant_id(model_type, "transformer", label)
                variants.append(
                    {
                        "id": variant_id,
                        "label": variant_label,
                        "type": "variant",
                        "files": files,
                        "primary_paths": primary_paths,
                    }
                )
                used_labels.add(variant_label)

        for token, dtypes in sorted(token_dtypes.items()):
            quant_param = self._quant_param_from_token(token)
            quant_label = token_labels.get(token) or self._format_quant_label(token)
            for dtype_policy in sorted(dtypes):
                label_suffix = ""
                if len(dtypes) > 1:
                    label_suffix = f" {self._format_dtype_label(dtype_policy)}"
                label = f"{quant_label}{label_suffix}"
                files, primary_paths = self._collect_transformer_files_for_variant(
                    model_type,
                    model_def,
                    modules,
                    quant_param,
                    dtype_policy,
                    token_match=token,
                )
                if not files:
                    continue
                file_label = self._detect_quant_label_from_files(files)
                quant_label = file_label or quant_label
                if not self._files_match_token(files, token):
                    continue
                label = f"{quant_label}{label_suffix}" if quant_label else self._format_dtype_label(dtype_policy)
                variant_label = f"Transformer {label}"
                if variant_label in used_labels:
                    continue
                variant_id = self._make_variant_id(model_type, "transformer", label)
                variants.append(
                    {
                        "id": variant_id,
                        "label": variant_label,
                        "type": "variant",
                        "files": files,
                        "primary_paths": primary_paths,
                    }
                )
                used_labels.add(variant_label)

        return sorted(variants, key=lambda v: v["label"].lower())

    def _collect_module_variants(self, model_type, model_def, default_dtype_policy):
        module_entries = self._expand_modules_with_ids(model_type)
        if not module_entries:
            return []

        variants = []
        used_labels = set()
        base_default = self._format_dtype_label(default_dtype_policy)
        for idx, entry in enumerate(module_entries):
            module = entry.get("module")
            module_id = entry.get("module_id", "")
            module_name = self._get_module_display_name(model_type, module, module_id, idx)
            choice_lists = self._collect_module_choice_lists(module)
            base_dtypes, token_dtypes, token_labels = self._collect_variant_candidates(
                choice_lists, default_dtype_policy
            )

            if base_dtypes:
                for dtype_policy in sorted(base_dtypes):
                    label = self._format_variant_label("", dtype_policy, base_default)
                    files = self._collect_module_files_for_variant(
                        model_type,
                        module,
                        "",
                        dtype_policy,
                    )
                    if not files:
                        continue
                    file_label = self._detect_quant_label_from_files(files)
                    label = file_label or label
                    variant_label = f"Module {module_name} {label}".strip()
                    if variant_label in used_labels:
                        continue
                    variant_id = self._make_variant_id(model_type, "module", f"{module_name} {label}")
                    variants.append(
                        {
                            "id": variant_id,
                            "label": variant_label,
                            "type": "variant",
                            "files": files,
                            "primary_paths": [],
                        }
                    )
                    used_labels.add(variant_label)

            for token, dtypes in sorted(token_dtypes.items()):
                quant_param = self._quant_param_from_token(token)
                quant_label = token_labels.get(token) or self._format_quant_label(token)
                for dtype_policy in sorted(dtypes):
                    label_suffix = ""
                    if len(dtypes) > 1:
                        label_suffix = f" {self._format_dtype_label(dtype_policy)}"
                    files = self._collect_module_files_for_variant(
                        model_type,
                        module,
                        quant_param,
                        dtype_policy,
                        token_match=token,
                    )
                    if not files:
                        continue
                    if not self._files_match_token(files, token):
                        continue
                    file_label = self._detect_quant_label_from_files(files)
                    effective_quant_label = file_label or quant_label
                    label = f"{effective_quant_label}{label_suffix}" if effective_quant_label else self._format_dtype_label(dtype_policy)
                    variant_label = f"Module {module_name} {label}".strip()
                    if variant_label in used_labels:
                        continue
                    variant_id = self._make_variant_id(model_type, "module", f"{module_name} {label}")
                    variants.append(
                        {
                            "id": variant_id,
                            "label": variant_label,
                            "type": "variant",
                            "files": files,
                            "primary_paths": [],
                        }
                    )
                    used_labels.add(variant_label)

        return sorted(variants, key=lambda v: v["label"].lower())

    def _get_module_display_name(self, model_type, module, module_id, module_index):
        if isinstance(module_id, str) and module_id:
            return module_id
        if isinstance(module, str):
            module_type, _ = self._split_module_ref(module)
            return module_type
        return model_type

    def _collect_module_choice_lists(self, module):
        choice_lists = []
        if isinstance(module, dict):
            urls1 = module.get("URLs", [])
            urls2 = module.get("URLs2", [])
            if urls1:
                choice_lists.append(urls1)
            if urls2:
                choice_lists.append(urls2)
        elif isinstance(module, list):
            if module:
                choice_lists.append(module)
        else:
            module_type = module
            sub_prop_name = "_list"
            if isinstance(module_type, str) and "#" in module_type:
                pos = module_type.rfind("#")
                sub_prop_name = module_type[pos + 1 :]
                module_type = module_type[:pos]
            try:
                mod_urls = self.get_model_recursive_prop(
                    module_type, "modules", sub_prop_name=sub_prop_name, return_list=True
                )
            except Exception:
                mod_urls = []
            if isinstance(mod_urls, list) and mod_urls:
                if all(isinstance(item, list) for item in mod_urls):
                    for entry in mod_urls:
                        if entry:
                            choice_lists.append(entry)
                else:
                    choice_lists.append(mod_urls)
        return choice_lists

    def _collect_module_files_for_variant(
        self,
        model_type,
        module,
        quantization,
        dtype_policy,
        token_match=None,
    ):
        files = set()
        is_exotic = quantization not in ("", "int8", "fp8")
        if isinstance(module, dict):
            for key in ("URLs", "URLs2"):
                urls = module.get(key, [])
                if not urls:
                    continue
                filename = ""
                if token_match and is_exotic:
                    filename = self._select_filename_by_token(urls, token_match)
                if not filename:
                    filename = self.get_model_filename(
                        model_type=model_type,
                        quantization=quantization,
                        dtype_policy=dtype_policy,
                        URLs=urls,
                    )
                if filename:
                    self._add_file(files, filename)
        elif isinstance(module, list):
            filename = ""
            if token_match and is_exotic:
                filename = self._select_filename_by_token(module, token_match)
            if not filename:
                filename = self.get_model_filename(
                    model_type=model_type,
                    quantization=quantization,
                    dtype_policy=dtype_policy,
                    URLs=module,
                )
            if filename:
                self._add_file(files, filename)
        else:
            filename = ""
            if token_match and is_exotic:
                module_type = module
                sub_prop_name = "_list"
                if isinstance(module_type, str) and "#" in module_type:
                    pos = module_type.rfind("#")
                    sub_prop_name = module_type[pos + 1 :]
                    module_type = module_type[:pos]
                try:
                    mod_urls = self.get_model_recursive_prop(
                        module_type,
                        "modules",
                        sub_prop_name=sub_prop_name,
                        return_list=True,
                    )
                except Exception:
                    mod_urls = []
                filename = self._select_filename_by_token(mod_urls, token_match)
            if not filename:
                try:
                    filename = self.get_model_filename(
                        model_type=model_type,
                        quantization=quantization,
                        dtype_policy=dtype_policy,
                        module_type=module,
                    )
                except Exception:
                    filename = ""
            if filename:
                self._add_file(files, filename)
        return files

    def _collect_transformer_choice_lists(self, model_type, model_def):
        choice_lists = []
        modules = self._expand_modules(model_type)

        urls = self.get_model_recursive_prop(model_type, "URLs", return_list=True)
        if urls:
            choice_lists.append(urls)
        if "URLs2" in model_def:
            urls2 = self.get_model_recursive_prop(model_type, "URLs2", return_list=True)
            if urls2:
                choice_lists.append(urls2)

        has_module_source = "module_source" in model_def
        has_module_source2 = "module_source2" in model_def
        for module in modules:
            if isinstance(module, dict):
                urls1 = module.get("URLs", [])
                urls2 = module.get("URLs2", [])
                if urls1 and not has_module_source:
                    choice_lists.append(urls1)
                if urls2 and not has_module_source2:
                    choice_lists.append(urls2)
            elif isinstance(module, list):
                if has_module_source:
                    continue
                if module:
                    choice_lists.append(module)
            else:
                if has_module_source:
                    continue
                module_type = module
                sub_prop_name = "_list"
                if isinstance(module_type, str) and "#" in module_type:
                    pos = module_type.rfind("#")
                    sub_prop_name = module_type[pos + 1 :]
                    module_type = module_type[:pos]
                try:
                    mod_urls = self.get_model_recursive_prop(
                        module_type, "modules", sub_prop_name=sub_prop_name, return_list=True
                    )
                except Exception:
                    mod_urls = []
                if isinstance(mod_urls, list) and mod_urls:
                    if all(isinstance(item, list) for item in mod_urls):
                        for entry in mod_urls:
                            if entry:
                                choice_lists.append(entry)
                    else:
                        choice_lists.append(mod_urls)
        return choice_lists, modules

    def _collect_transformer_files_for_variant(
        self,
        model_type,
        model_def,
        modules,
        quantization,
        dtype_policy,
        token_match=None,
    ):
        files = set()
        primary_paths = []

        model_filename = ""
        is_exotic = quantization not in ("", "int8", "fp8")
        if token_match and is_exotic:
            urls = self.get_model_recursive_prop(model_type, "URLs", return_list=True)
            model_filename = self._select_filename_by_token(urls, token_match)
        if not model_filename:
            model_filename = self.get_model_filename(
                model_type=model_type,
                quantization=quantization,
                dtype_policy=dtype_policy,
            )
        if model_filename:
            self._add_file(files, model_filename)
            primary_path = self._resolve_path(model_filename)
            if primary_path and primary_path in files:
                primary_paths.append(primary_path)

        if "URLs2" in model_def:
            model_filename2 = ""
            if token_match and is_exotic:
                urls2 = self.get_model_recursive_prop(
                    model_type, "URLs2", return_list=True
                )
                model_filename2 = self._select_filename_by_token(urls2, token_match)
            if not model_filename2:
                model_filename2 = self.get_model_filename(
                    model_type=model_type,
                    quantization=quantization,
                    dtype_policy=dtype_policy,
                    submodel_no=2,
                )
            if model_filename2:
                self._add_file(files, model_filename2)
                primary_path2 = self._resolve_path(model_filename2)
                if primary_path2 and primary_path2 in files:
                    primary_paths.append(primary_path2)

        has_module_source = "module_source" in model_def
        has_module_source2 = "module_source2" in model_def
        for module in modules:
            if isinstance(module, dict):
                urls1 = module.get("URLs", [])
                urls2 = module.get("URLs2", [])
                if urls1 and not has_module_source:
                    filename = ""
                    if token_match and is_exotic:
                        filename = self._select_filename_by_token(urls1, token_match)
                    if not filename:
                        filename = self.get_model_filename(
                            model_type=model_type,
                            quantization=quantization,
                            dtype_policy=dtype_policy,
                            URLs=urls1,
                        )
                    if filename:
                        self._add_file(files, filename)
                if urls2 and not has_module_source2:
                    filename = ""
                    if token_match and is_exotic:
                        filename = self._select_filename_by_token(urls2, token_match)
                    if not filename:
                        filename = self.get_model_filename(
                            model_type=model_type,
                            quantization=quantization,
                            dtype_policy=dtype_policy,
                            URLs=urls2,
                        )
                    if filename:
                        self._add_file(files, filename)
            elif isinstance(module, list):
                if has_module_source:
                    continue
                filename = ""
                if token_match and is_exotic:
                    filename = self._select_filename_by_token(module, token_match)
                if not filename:
                    filename = self.get_model_filename(
                        model_type=model_type,
                        quantization=quantization,
                        dtype_policy=dtype_policy,
                        URLs=module,
                    )
                if filename:
                    self._add_file(files, filename)
            else:
                if has_module_source:
                    continue
                if token_match and is_exotic:
                    module_type = module
                    sub_prop_name = "_list"
                    if isinstance(module_type, str) and "#" in module_type:
                        pos = module_type.rfind("#")
                        sub_prop_name = module_type[pos + 1 :]
                        module_type = module_type[:pos]
                    try:
                        mod_urls = self.get_model_recursive_prop(
                            module_type,
                            "modules",
                            sub_prop_name=sub_prop_name,
                            return_list=True,
                        )
                    except Exception:
                        mod_urls = []
                    filename = self._select_filename_by_token(mod_urls, token_match)
                else:
                    filename = ""
                try:
                    if not filename:
                        filename = self.get_model_filename(
                            model_type=model_type,
                            quantization=quantization,
                            dtype_policy=dtype_policy,
                            module_type=module,
                        )
                except Exception:
                    filename = ""
                if filename:
                    self._add_file(files, filename)

        return files, primary_paths

    def _collect_text_encoder_variants(
        self, model_type, model_def, default_dtype_policy
    ):
        variants = []
        text_encoder_urls = self.get_model_recursive_prop(
            model_type, "text_encoder_URLs", return_list=True
        )
        text_encoder_folder = model_def.get("text_encoder_folder", None)
        used_paths = set()
        used_labels = set()
        base_default = self._format_dtype_label(default_dtype_policy)
        if text_encoder_urls:
            base_dtypes, token_dtypes, token_labels = self._collect_variant_candidates(
                [text_encoder_urls], default_dtype_policy
            )
            if base_dtypes:
                for dtype_policy in sorted(base_dtypes):
                    label = self._format_variant_label("", dtype_policy, base_default)
                    filename = self.get_model_filename(
                        model_type=model_type,
                        quantization="",
                        dtype_policy=dtype_policy,
                        URLs=text_encoder_urls,
                    )
                    if not filename:
                        continue
                    files = set()
                    self._add_file(
                        files, filename, force_folder=text_encoder_folder
                    )
                    if not files:
                        continue
                    resolved_path = next(iter(files))
                    if resolved_path in used_paths:
                        continue
                    used_paths.add(resolved_path)
                    variant_label = f"Text Encoder {label}"
                    if variant_label in used_labels:
                        continue
                    variant_id = self._make_variant_id(model_type, "text", label)
                    variants.append(
                        {
                            "id": variant_id,
                            "label": variant_label,
                            "type": "variant",
                            "files": files,
                            "primary_paths": [resolved_path],
                        }
                    )
                    used_labels.add(variant_label)

            for token, dtypes in sorted(token_dtypes.items()):
                quant_param = self._quant_param_from_token(token)
                quant_label = token_labels.get(token) or self._format_quant_label(token)
                for dtype_policy in sorted(dtypes):
                    label_suffix = ""
                    if len(dtypes) > 1:
                        label_suffix = f" {self._format_dtype_label(dtype_policy)}"
                    label = f"{quant_label}{label_suffix}"
                    filename = ""
                    if quant_param not in ("", "int8", "fp8"):
                        filename = self._select_filename_by_token(text_encoder_urls, token)
                    if not filename:
                        filename = self.get_model_filename(
                            model_type=model_type,
                            quantization=quant_param,
                            dtype_policy=dtype_policy,
                            URLs=text_encoder_urls,
                        )
                    if not filename:
                        continue
                    files = set()
                    self._add_file(
                        files, filename, force_folder=text_encoder_folder
                    )
                    if not files:
                        continue
                    if not self._files_match_token(files, token):
                        continue
                    file_label = self._detect_quant_label_from_files(files)
                    quant_label = file_label or quant_label
                    resolved_path = next(iter(files))
                    if resolved_path in used_paths:
                        continue
                    used_paths.add(resolved_path)
                    label = f"{quant_label}{label_suffix}" if quant_label else self._format_dtype_label(dtype_policy)
                    variant_label = f"Text Encoder {label}"
                    if variant_label in used_labels:
                        continue
                    variant_id = self._make_variant_id(model_type, "text", label)
                    variants.append(
                        {
                            "id": variant_id,
                            "label": variant_label,
                            "type": "variant",
                            "files": files,
                            "primary_paths": [resolved_path],
                        }
                    )
                    used_labels.add(variant_label)
        else:
            handler = self.get_model_handler(model_type)
            get_name = getattr(handler, "get_text_encoder_filename", None)
            if get_name is not None:
                for quant_param in self._text_encoder_quant_candidates():
                    try:
                        filename = get_name(quant_param)
                    except Exception:
                        filename = None
                    if not filename:
                        continue
                    files = set()
                    self._add_file(
                        files, filename, force_folder=text_encoder_folder
                    )
                    if not files:
                        continue
                    resolved_path = next(iter(files))
                    if resolved_path in used_paths:
                        continue
                    used_paths.add(resolved_path)
                    quant_label = self._detect_quant_label_from_files(files)
                    label = quant_label or self._format_variant_label(
                        "",
                        self._detect_dtype_policy_from_name(os.path.basename(filename))
                        or default_dtype_policy,
                        base_default,
                    )
                    variant_label = f"Text Encoder {label}"
                    if variant_label in used_labels:
                        continue
                    variant_id = self._make_variant_id(model_type, "text", label)
                    variants.append(
                        {
                            "id": variant_id,
                            "label": variant_label,
                            "type": "variant",
                            "files": files,
                            "primary_paths": [resolved_path],
                        }
                    )
                    used_labels.add(variant_label)

        return sorted(variants, key=lambda v: v["label"].lower())

    def _collect_lora_variants(self, model_type):
        lora_dir = self._safe_get_lora_dir(model_type)
        if not lora_dir:
            return []
        loras = self.get_model_recursive_prop(model_type, "loras", return_list=True)
        if loras is None:
            return []

        variants = []
        used_paths = set()
        used_labels = {}
        for entry in self._ensure_list(loras):
            if not isinstance(entry, str) or len(entry) == 0:
                continue
            source_entry = entry.split("|", 1)[0]
            basename = os.path.basename(source_entry)
            if len(basename) == 0:
                continue
            files = set()
            self._add_file(files, self.get_lora_local_path(lora_dir, entry))
            if not files:
                continue
            resolved_path = next(iter(files))
            if resolved_path in used_paths:
                continue
            used_paths.add(resolved_path)
            stem = os.path.splitext(basename)[0]
            base_label = f"Lora {stem}"
            label_count = used_labels.get(base_label, 0)
            used_labels[base_label] = label_count + 1
            variant_label = base_label if label_count == 0 else f"{base_label} #{label_count + 1}"
            variant_id = self._make_variant_id(model_type, "lora", basename)
            variants.append(
                {
                    "id": variant_id,
                    "label": variant_label,
                    "type": "variant",
                    "files": files,
                    "primary_paths": [resolved_path],
                }
            )
        return sorted(variants, key=lambda v: v["label"].lower())

    def _collect_other_files(
        self,
        model_type,
        model_def,
        transformer_files,
        text_encoder_files,
        lora_files,
    ):
        other_files = set()
        shared_files = set()
        lora_dir = self._safe_get_lora_dir(model_type)

        preload_urls = self.get_model_recursive_prop(
            model_type, "preload_URLs", return_list=True
        )
        for url in self._ensure_list(preload_urls):
            self._add_file(other_files, url, lora_dir=lora_dir)

        vae_urls = model_def.get("VAE_URLs", [])
        for url in self._ensure_list(vae_urls):
            self._add_file(shared_files, url, lora_dir=lora_dir)

        handler_files = self._collect_handler_files(model_type, model_def)
        shared_files.update(handler_files)

        other_files.difference_update(transformer_files)
        other_files.difference_update(text_encoder_files)
        other_files.difference_update(lora_files)
        shared_files.difference_update(transformer_files)
        shared_files.difference_update(text_encoder_files)
        shared_files.difference_update(lora_files)
        shared_files.difference_update(other_files)
        return other_files, shared_files

    def _resolve_default_dtype_policy(self, model_type, model_def):
        dtype = model_def.get("dtype")
        if isinstance(dtype, str) and dtype.lower() in ("fp16", "bf16"):
            return dtype.lower()
        policy = self.transformer_dtype_policy
        if isinstance(policy, str) and policy:
            return policy.lower()
        return "bf16"

    def _format_dtype_label(self, dtype_policy):
        return "FP16" if dtype_policy == "fp16" else "BF16"

    def _format_quant_label(self, token):
        if not token:
            return ""
        label = quant_router.get_quantization_label(token)
        if label:
            return label
        return str(token).replace("_", " ").upper()

    def _format_variant_label(self, token, dtype_policy, default_label):
        if token:
            return self._format_quant_label(token)
        if dtype_policy:
            return self._format_dtype_label(dtype_policy)
        return default_label

    def _detect_quant_info(self, name):
        if not name:
            return "", ""
        label = quant_router.detect_quantization_label_from_filename(name) or ""
        if label:
            return self._label_to_token(label), label
        kind = quant_router.detect_quantization_kind_for_file(name, verboseLevel=0)
        if kind and kind != "none":
            kind = str(kind).lower()
            return kind, kind.upper()
        lower = str(name).lower()
        aliases = self._get_quant_aliases()
        for token in aliases:
            if token in ("bf16", "fp16", "bfloat16", "float16"):
                continue
            if token and token in lower:
                return token, ""
        return "", ""

    def _label_to_token(self, label):
        if not label:
            return ""
        raw = str(label).strip()
        lower = raw.lower()
        if lower.startswith("gguf-"):
            return lower.split("-", 1)[1]
        if lower == "gguf":
            return "gguf"
        return lower

    def _detect_dtype_policy_from_name(self, name):
        if not name:
            return ""
        lower = str(name).lower()
        if "fp16" in lower or "float16" in lower:
            return "fp16"
        if "bf16" in lower or "bfloat16" in lower:
            return "bf16"
        return ""

    def _collect_variant_candidates(self, choice_lists, default_dtype_policy):
        base_dtypes = set()
        token_dtypes = defaultdict(set)
        token_labels = {}
        for choice_list in choice_lists:
            for entry in self._ensure_list(choice_list):
                if not entry:
                    continue
                entry_str = str(entry)
                name = self._locator_basename(entry_str)
                quant_source = self._resolve_path(entry_str) or self._locator_path(entry_str) or entry_str
                token, label = self._detect_quant_info(quant_source)
                dtype_policy = self._detect_dtype_policy_from_name(name) or default_dtype_policy
                if token:
                    token = str(token).lower()
                    token_dtypes[token].add(dtype_policy)
                    if label:
                        token_labels.setdefault(token, label)
                else:
                    base_dtypes.add(dtype_policy)
        if not base_dtypes and not token_dtypes:
            base_dtypes.add(default_dtype_policy)
        return base_dtypes, token_dtypes, token_labels

    def _quant_param_from_token(self, token):
        if not token:
            return ""
        lower = str(token).lower()
        if "int8" in lower:
            return "int8"
        if "fp8" in lower or "float8" in lower:
            return "fp8"
        return lower

    def _files_match_token(self, files, token):
        if not token:
            return True
        token = str(token).lower()
        for path in files:
            base = os.path.basename(path).lower()
            if token in base:
                return True
            label = quant_router.detect_quantization_label_from_filename(path)
            if label and self._label_to_token(label) == token:
                return True
        return False

    def _detect_quant_label_from_files(self, files):
        for path in files:
            label = quant_router.detect_quantization_label_from_filename(path)
            if label:
                return label
        return ""

    def _select_filename_by_token(self, urls, token):
        if not urls or not token:
            return ""
        token = str(token).lower()
        for entry in self._ensure_list(urls):
            if not entry:
                continue
            if isinstance(entry, list):
                for item in entry:
                    if not item:
                        continue
                    base = self._locator_basename(item).lower()
                    if token in base:
                        return item
                    label = quant_router.detect_quantization_label_from_filename(item)
                    if label and self._label_to_token(label) == token:
                        return item
                continue
            base = self._locator_basename(entry).lower()
            if token in base:
                return entry
            label = quant_router.detect_quantization_label_from_filename(entry)
            if label and self._label_to_token(label) == token:
                return entry
        return ""

    def _split_module_ref(self, module_ref, default_sub_prop="_list"):
        module_type = module_ref
        sub_prop_name = default_sub_prop
        if isinstance(module_ref, str) and "#" in module_ref:
            pos = module_ref.rfind("#")
            module_type = module_ref[:pos]
            sub_prop_name = module_ref[pos + 1 :]
        return module_type, sub_prop_name

    def _resolve_recursive_prop_owner(self, model_type, prop, sub_prop_name, stack):
        if not isinstance(model_type, str) or not model_type:
            return ""
        if model_type in stack or len(stack) > 20:
            return model_type
        model_def = self.get_model_def(model_type)
        if not isinstance(model_def, dict):
            return model_type
        prop_value = model_def.get(prop, None)
        if prop_value is None:
            return model_type
        if sub_prop_name is not None:
            if sub_prop_name == "_list":
                if not isinstance(prop_value, list) or len(prop_value) != 1:
                    return model_type
                prop_value = prop_value[0]
            else:
                if not isinstance(prop_value, dict) or sub_prop_name not in prop_value:
                    return model_type
                prop_value = prop_value[sub_prop_name]
        if isinstance(prop_value, str):
            next_model_type, next_sub_prop_name = self._split_module_ref(
                prop_value, default_sub_prop=sub_prop_name
            )
            if not next_model_type:
                return model_type
            return self._resolve_recursive_prop_owner(
                next_model_type, prop, next_sub_prop_name, stack + [model_type]
            )
        return model_type

    def _resolve_final_module_id(self, module_ref):
        if not isinstance(module_ref, str) or not module_ref:
            return ""
        module_type, sub_prop_name = self._split_module_ref(module_ref)
        resolved_id = self._resolve_recursive_prop_owner(
            module_type, "modules", sub_prop_name, []
        )
        return resolved_id or module_type

    def _expand_modules_with_ids(self, model_type):
        modules = self.get_model_recursive_prop(model_type, "modules", return_list=True)
        expanded_modules = []
        modules_count = len(modules)
        for index, module in enumerate(modules):
            if isinstance(module, str):
                module_id = self._resolve_final_module_id(module)
                if "#" in module:
                    expanded_modules.append({"module": module, "module_id": module_id})
                    continue
                try:
                    expanded = self.get_model_recursive_prop(
                        module, "modules", sub_prop_name="_list", return_list=True
                    )
                except Exception:
                    expanded = []
                if expanded:
                    expanded_modules.append({"module": expanded, "module_id": module_id})
                else:
                    expanded_modules.append({"module": module, "module_id": module_id})
            else:
                fallback_id = model_type if modules_count <= 1 else f"{model_type}#{index + 1}"
                expanded_modules.append({"module": module, "module_id": fallback_id})
        return expanded_modules

    def _expand_modules(self, model_type):
        return [entry.get("module") for entry in self._expand_modules_with_ids(model_type)]

    def _make_variant_id(self, model_type, kind, label):
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        return f"variant::{model_type}::{kind}::{slug}"

    def _dedupe_paths(self, paths):
        seen = set()
        ordered = []
        for path in paths:
            if not path or path in seen:
                continue
            ordered.append(path)
            seen.add(path)
        return ordered

    def _get_quant_aliases(self):
        if not hasattr(self, "_quant_alias_cache"):
            aliases = set(quant_router.get_available_qtype_aliases() or [])
            aliases.update({"int8", "int8n", "fp8"})
            self._quant_alias_cache = sorted(
                {alias for alias in aliases if alias}, key=len, reverse=True
            )
        return self._quant_alias_cache

    def _text_encoder_quant_candidates(self):
        candidates = {""}
        aliases = self._get_quant_aliases()
        for token in aliases:
            if "int8" in token:
                candidates.add("int8")
            if "fp8" in token or "float8" in token:
                candidates.add("fp8")
        return sorted(candidates)

    def _collect_text_encoder_files(self, model_type, model_def):
        files = set()
        text_encoder_urls = self.get_model_recursive_prop(
            model_type, "text_encoder_URLs", return_list=True
        )
        text_encoder_folder = model_def.get("text_encoder_folder", None)
        text_encoder_filename = None
        if text_encoder_urls:
            text_encoder_filename = self.get_model_filename(
                model_type=model_type,
                quantization=self.text_encoder_quantization,
                dtype_policy=self.transformer_dtype_policy,
                URLs=text_encoder_urls,
            )
        else:
            handler = self.get_model_handler(model_type)
            get_name = getattr(handler, "get_text_encoder_filename", None)
            if get_name is not None:
                try:
                    text_encoder_filename = get_name(self.text_encoder_quantization)
                except Exception:
                    text_encoder_filename = None
        if text_encoder_filename:
            self._add_file(files, text_encoder_filename, force_folder=text_encoder_folder)
        return files

    def _collect_retired_files(self, model_type):
        """The files of earlier releases of the model type that are still on disk: its handler names them (query_retired_files), as the
        current definitions no longer do. A few file checks, no folder scan."""
        query = getattr(self.get_model_handler(model_type), "query_retired_files", None)
        if query is None:
            return set()
        return {self._normalize_path(copy) for relative in query(model_type) for copy in fl.installed_copies(relative)}

    def _collect_handler_file_paths(self, model_type, model_def):
        handler = self.get_model_handler(model_type)
        query = getattr(handler, "query_model_files", None)
        if query is None:
            return set()
        base_model_type = self.get_base_model_type(model_type)
        try:
            download_defs = query(
                self._compute_list,
                base_model_type,
                model_def,
            )
        except Exception as exc:
            self._errors.append(f"{model_type}: {exc}")
            return set()

        return self._collect_download_def_file_paths(download_defs)

    def _collect_download_def_file_paths(self, download_defs, *, resolve=None):
        if not download_defs:
            return set()
        if isinstance(download_defs, dict):
            download_defs = [download_defs]

        files = set()
        for download_def in download_defs:
            if not isinstance(download_def, dict):
                continue
            source_folders = download_def.get("sourceFolderList", [])
            file_lists = download_def.get("fileList", [])
            target_folders = download_def.get("targetFolderList")
            if target_folders is None:
                target_folders = [None] * len(source_folders)

            for source_folder, file_list, target_folder in zip(
                source_folders, file_lists, target_folders
            ):
                if target_folder == "":
                    target_folder = None
                if source_folder is None:
                    source_folder = ""

                if not file_list:
                    continue

                for filename in file_list:
                    if not filename:
                        continue
                    rel_path = self._combine_download_path(
                        target_folder, source_folder, filename
                    )
                    if resolve is None:
                        files.add(self._resolve_download_relpath(rel_path))
                    else:
                        files.update(resolve(rel_path))
        return {path for path in files if path}

    def _collect_system_entries(self):
        from shared.utils.optional_assets import checkpoint_inventory

        entries = {}
        for entry in checkpoint_inventory(self.app.plugin_manager.get_all_plugins(), self._errors.append):
            family, name = entry["family"], entry["name"]
            labels = (family, name, entry["variant"]) if entry.get("variant") else (family, name)
            owner = "system::asset::" + json.dumps((entry["owner"], *labels), ensure_ascii=False)
            try:
                def resolve(path):
                    alternatives = entry.get("alternate_paths", {}).get(path.replace("\\", "/"), ())
                    installed = {self._normalize_path(copy) for candidate in (path, *alternatives) for copy in fl.installed_copies(candidate)}
                    return installed or {self._resolve_download_relpath(path)}

                paths = self._collect_download_def_file_paths(entry.get("download_defs"), resolve=resolve)
                for path in entry.get("files", ()):
                    paths.update(resolve(path))
                for model_type in entry.get("model_types", ()):
                    paths.update(self._model_files[model_type])
                    for variant in self._model_variants[model_type]:
                        paths.update(variant.get("missing_files", ()))
                for model_type, keys in entry.get("model_lora_keys", {}).items():
                    definition = self.get_model_def(model_type)
                    for key in keys:
                        paths.add(self._normalize_path(self.get_lora_local_path(self.get_lora_dir(model_type), definition[key])))
            except Exception as exc:
                self._errors.append(f"{' / '.join(labels)}: {exc}")
                continue
            _, files = entries.setdefault(owner, (labels, set()))
            files.update(paths)
            for path in paths:
                self._file_usage_models[path].add(owner)
        return entries

    def _build_system_tree(self):
        families = defaultdict(list)
        variants = defaultdict(list)
        for owner, (labels, paths) in self._system_entries.items():
            installed = {path for path in paths if self._file_info[path]["exists"]}
            self._register_node(owner, labels[-1], "system", {owner}, installed, missing_files=paths - installed, path_label=" / ".join(labels))
            leaf = {"id": owner, "label": labels[-1]}
            if len(labels) == 3:
                variants[labels[:2]].append(leaf)
            else:
                families[labels[0]].append(leaf)
        for labels, children in variants.items():
            node_id = "system::group::" + json.dumps(labels, ensure_ascii=False)
            families[labels[0]].append(self._system_branch(node_id, labels, children))

        family_order = ["Deepy / Prompt Enhancer", "Video Postprocessors", "Audio Postprocessors", "Video Preprocessors",
                        "Audio Preprocessors", "Image Postprocessors", "Image Preprocessors", "3D Postprocessors", "Preview Decoders"]
        ordered = sorted(families, key=lambda name: (family_order.index(name) if name in family_order else len(family_order), name))
        self._system_families = [self._system_branch(f"system::{family}", (family,), families[family]) for family in ordered]

    def _system_branch(self, node_id, labels, children):
        owners, files = set(), set()
        for child in children:
            node = self._node_map[child["id"]]
            owners.update(node["models"])
            files.update(node["files"])
        self._register_node(node_id, labels[-1], "family" if len(labels) == 1 else "base", owners, files, path_label=" / ".join(labels))
        return {"id": node_id, "label": labels[-1], "children": sorted(children, key=lambda child: child["label"].casefold())}

    def _build_system_tree_html(self):
        rows = "".join(self._render_node(family, native_expand=True) for family in self._system_families)
        return f"<div class='ckpt-panel ckpt-panel-inner ckpt-tree'>{rows}</div>"

    def _collect_handler_files(self, model_type, model_def):
        files = self._collect_handler_file_paths(model_type, model_def)
        return {path for path in files if os.path.isfile(path)}

    def _build_tree_structure(self, model_types):
        families = defaultdict(list)
        for model_type in model_types:
            family = self.get_model_family(model_type, for_ui=True) or "unknown"
            families[family].append(model_type)

        family_order = sorted(
            families.keys(),
            key=lambda family: self.families_infos.get(family, (999, family))[0],
        )

        for family in family_order:
            family_label = self.families_infos.get(family, (999, family))[1]
            rows = []
            for model_type in families[family]:
                model_name = self.compact_name(
                    family_label, self.get_model_name(model_type)
                )
                parent_id = self.get_parent_model_type(model_type)
                rows.append((model_name, model_type, parent_id))

            rows.sort(key=lambda row: row[0])
            parents_list, children_dict = self.create_models_hierarchy(rows)
            family_node_id = f"family::{family}"
            family_models = set(families[family])
            family_usage_ids = self._collect_usage_ids_for_models(family_models)
            family_files = self._collect_files_for_usage_ids(family_usage_ids)
            self._register_node(
                family_node_id,
                family_label,
                "family",
                family_models,
                family_files,
                path_label=family_label,
            )

            base_nodes = []
            for base_label, base_id in parents_list:
                base_status = self._model_aggregated_status_map.get(
                    base_id, model_dropdowns.MODEL_FILE_STATUS_MISSING
                )
                base_label_decorated = self._decorate_status_label(base_label, base_status)
                child_entries = children_dict.get(base_id, [])
                child_nodes = []
                base_models = set()
                base_usage_ids = set()
                base_path = f"{family_label} / {base_label}"
                for child_label, child_model in child_entries:
                    child_status = self._model_direct_status_map.get(
                        child_model, model_dropdowns.MODEL_FILE_STATUS_MISSING
                    )
                    child_label_decorated = self._decorate_status_label(child_label, child_status)
                    base_models.add(child_model)
                    model_usage_ids = self._model_usage_ids.get(child_model, set())
                    base_usage_ids.update(model_usage_ids)
                    self._model_base_label[child_model] = base_label
                    self._model_family_label[child_model] = family_label
                    model_node_id = f"model::{child_model}"
                    model_files = self._collect_files_for_usage_ids(model_usage_ids)
                    primary_paths = self._model_primary.get(child_model, [])
                    model_path = f"{base_path} / {child_label}"
                    self._model_display_label[child_model] = model_path
                    self._register_node(
                        model_node_id,
                        child_label_decorated,
                        "model",
                        {child_model},
                        model_files,
                        primary_paths=primary_paths,
                        path_label=model_path,
                    )
                    variant_children = []
                    for variant in self._ordered_variants_for_display(child_model):
                        variant_id = variant["id"]
                        variant_label = variant["label"]
                        variant_type = variant.get("type", "variant")
                        variant_path = f"{model_path} / {variant_label}"
                        self._register_node(
                            variant_id,
                            variant_label,
                            variant_type,
                            {child_model},
                            variant.get("files", set()),
                            missing_files=variant.get("missing_files", set()),
                            primary_paths=variant.get("primary_paths", []),
                            path_label=variant_path,
                        )
                        variant_children.append(
                            {"id": variant_id, "label": variant_label, "type": variant_type}
                        )
                    child_nodes.append(
                        {
                            "id": model_node_id,
                            "label": child_label_decorated,
                            "type": "model",
                            "children": variant_children,
                        }
                    )

                base_node_id = f"base::{family}::{base_id}"
                base_files = self._collect_files_for_usage_ids(base_usage_ids)
                self._register_node(
                    base_node_id,
                    base_label_decorated,
                    "base",
                    base_models,
                    base_files,
                    path_label=base_path,
                )
                base_nodes.append(
                    {
                        "id": base_node_id,
                        "label": base_label_decorated,
                        "type": "base",
                        "children": child_nodes,
                    }
                )

            self._tree_data.append(
                {
                    "id": family_node_id,
                    "label": family_label,
                    "type": "family",
                    "children": base_nodes,
                }
            )

    def _collect_usage_ids_for_models(self, model_set):
        usage_ids = set()
        for model_type in model_set:
            usage_ids.update(self._model_usage_ids.get(model_type, set()))
        return usage_ids

    def _collect_files_for_usage_ids(self, usage_ids):
        files = set()
        for usage_id in usage_ids:
            files.update(self._usage_files.get(usage_id, set()))
        return files

    def _ordered_variants_for_display(self, model_type):
        variants = list(self._model_variants.get(model_type, []))
        if not variants:
            return []
        non_missing = [variant for variant in variants if variant.get("type") != "missing"]
        missing = [variant for variant in variants if variant.get("type") == "missing"]
        return non_missing + missing

    def _register_node(
        self,
        node_id,
        label,
        node_type,
        model_set,
        files_set,
        missing_files=None,
        primary_paths=None,
        path_label=None,
    ):
        files_set = set(files_set or set())
        missing_files = set(missing_files or set())
        missing_files.difference_update(files_set)
        missing_files = {path for path in missing_files if path and not os.path.isfile(path)}
        shared_files = {path for path in files_set if len(self._file_usage_models[path]) > 1}
        unique_files = files_set - shared_files
        # Files shared only among siblings are still safe to remove with the whole branch.
        exclusive_files = {path for path in files_set if self._file_usage_models[path].issubset(model_set)}
        unique_size = self._sum_sizes(unique_files)
        shared_size = self._sum_sizes(shared_files)
        self._node_map[node_id] = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "models": set(model_set),
            "files": set(files_set),
            "unique_files": unique_files,
            "shared_files": shared_files,
            "exclusive_files": exclusive_files,
            "exclusive_size": self._sum_sizes(exclusive_files),
            "unique_size": unique_size,
            "shared_size": shared_size,
            "total_size": unique_size + shared_size,
            "missing_files": missing_files,
            "missing_count": len(missing_files),
            "primary_paths": primary_paths or [],
            "path_label": path_label or label,
        }

    def _sum_sizes(self, files_set):
        sizes = {self._file_info[path]["id"]: self._file_info[path]["size"] for path in files_set}
        return sum(sizes.values())

    def _rebalance_other_shared_variants(self, model_types):
        for model_type in model_types:
            variants = self._model_variants.get(model_type, [])
            if not variants:
                continue
            other_variant = None
            shared_variant = None
            for variant in variants:
                variant_id = str(variant.get("id", ""))
                parts = variant_id.split("::")
                variant_kind = parts[2] if len(parts) >= 4 else ""
                label = str(variant.get("label", "")).lower()
                if variant_kind == "other" or label in ("other", "other non shared"):
                    other_variant = variant
                elif variant_kind == "shared" or label in ("shared", "other shared"):
                    shared_variant = variant

            if not other_variant and not shared_variant:
                continue

            combined = set()
            if other_variant:
                combined.update(other_variant.get("files", set()))
            if shared_variant:
                combined.update(shared_variant.get("files", set()))

            new_variants = []
            for variant in variants:
                if variant is other_variant or variant is shared_variant:
                    continue
                new_variants.append(variant)
            if not combined:
                self._model_variants[model_type] = new_variants
                continue

            model_set = {model_type}
            shared_files = {
                path
                for path in combined
                if not self._file_usage_models.get(path, set()).issubset(model_set)
            }
            other_files = combined - shared_files

            if other_files:
                if other_variant is None:
                    other_variant = {
                        "id": self._make_variant_id(model_type, "other", "other"),
                        "label": VARIANT_LABEL_OTHER_NON_SHARED,
                        "type": "variant",
                        "files": other_files,
                        "primary_paths": [],
                    }
                else:
                    other_variant["files"] = other_files
                new_variants.append(other_variant)

            if shared_files:
                if shared_variant is None:
                    shared_variant = {
                        "id": self._make_variant_id(model_type, "shared", "shared"),
                        "label": VARIANT_LABEL_OTHER_SHARED,
                        "type": "variant",
                        "files": shared_files,
                        "primary_paths": [],
                    }
                else:
                    shared_variant["files"] = shared_files
                new_variants.append(shared_variant)

            self._model_variants[model_type] = new_variants

    def _get_model_shared_label(self, model_type):
        if model_type in self._system_entries:
            labels, _ = self._system_entries[model_type]
            return " / ".join(labels)
        model_name = self.get_model_name(model_type)
        if model_name:
            return model_name
        label = self._model_display_label.get(model_type)
        if label:
            return label
        return model_type

    def _shared_model_labels(self, path, exclude_models=None):
        if exclude_models is None:
            exclude_models = set()
        labels = set()
        for model_type in self._file_usage_models.get(path, set()):
            if model_type in exclude_models:
                continue
            labels.add(self._get_model_shared_label(model_type))
        return sorted(labels, key=lambda value: value.lower())

    @staticmethod
    def _tree_styles():
        return """
        <style>
        .ckpt-wrap { display: flex; flex-direction: column; gap: 12px; }
        .ckpt-panel { border: 1px solid var(--border-color-primary); border-radius: 14px; background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02)); box-shadow: var(--shadow-drop-sm, 0 2px 8px rgba(0,0,0,0.08)); }
        .ckpt-panel-inner { padding: 10px; }
        .ckpt-tree { display: flex; flex-direction: column; gap: 8px; }
        .ckpt-node { border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--background-fill-secondary); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
        .ckpt-node summary { list-style: none; cursor: pointer; padding: 8px 10px; }
        .ckpt-node summary::-webkit-details-marker { display: none; }
        .ckpt-row { display: flex; align-items: center; gap: 10px; border-radius: 8px; padding: 6px 6px; }
        .ckpt-row.clickable { cursor: pointer; transition: background 0.15s ease; }
        .ckpt-row.clickable:hover { background: rgba(127,127,127,0.08); }
        .ckpt-label { flex: 1; font-weight: 600; color: inherit; }
        .ckpt-label-missing { color: #c62828; }
        .ckpt-row[data-node-type="missing"] .ckpt-label { color: #c62828 !important; font-weight: 700; }
        .ckpt-sizes { display: flex; flex-direction: column; align-items: stretch; min-width: 128px; width: 128px; font-size: 0.85em; font-variant-numeric: tabular-nums; gap: 4px; }
        .ckpt-size-unique { color: inherit; }
        .ckpt-size-shared { color: inherit; }
        .ckpt-sizes-missing { justify-content: center; }
        .ckpt-size-missing { font-size: 0.82em; color: #c62828; text-align: center; font-weight: 600; }
        .ckpt-badge { padding: 3px 8px; border-radius: 999px; font-weight: 600; font-size: 0.82em; font-variant-numeric: tabular-nums; min-width: 84px; width: 100%; display: inline-flex; align-items: center; justify-content: center; }
        .ckpt-badge-unique { background: rgba(64, 120, 255, 0.18); color: #2f5bd4; border: 1px solid rgba(64, 120, 255, 0.4); }
        .ckpt-badge-shared { background: rgba(240, 200, 40, 0.25); color: #a66a00; border: 1px solid rgba(240, 200, 40, 0.5); }
        .ckpt-badge-zero { background: rgba(120, 120, 120, 0.12); color: #888; border: 1px solid rgba(120, 120, 120, 0.25); }
        .ckpt-badge-missing { background: rgba(220, 60, 60, 0.18); color: #b71c1c; border: 1px solid rgba(220, 60, 60, 0.45); min-width: auto; width: auto; }
        .ckpt-actions { display: flex; gap: 18px; }
        .ckpt-action-btn { padding: 3px 12px; height: 30px; border: 1px solid var(--border-color-primary); border-radius: 6px; cursor: pointer; font-size: 0.85em; display: inline-flex; align-items: center; justify-content: center; font-weight: 600; background: transparent; color: inherit; }
        .ckpt-action-btn:hover { box-shadow: var(--shadow-drop, 0 2px 6px rgba(0,0,0,0.12)); }
        .ckpt-view-btn { background: rgba(60, 120, 255, 0.9) !important; color: #ffffff !important; border-color: rgba(60, 120, 255, 0.95) !important; }
        .ckpt-view-btn:hover { background: rgba(60, 120, 255, 1) !important; }
        .ckpt-delete { background: rgba(220, 60, 60, 0.95) !important; color: #ffffff !important; border-color: rgba(220, 60, 60, 0.95) !important; }
        .ckpt-delete:hover { background: rgba(220, 60, 60, 1) !important; }
        .ckpt-children { padding: 6px 8px 10px 18px; display: flex; flex-direction: column; gap: 6px; }
        .ckpt-leaf { border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--background-fill-primary); padding: 6px 8px; }
        .gradio-container .ckpt-right-column { align-self: flex-start; position: sticky; top: 16px; max-height: calc(100dvh - 32px); overflow-y: auto; }
        .ckpt-view { border: 1px solid var(--border-color-primary); border-radius: 12px; background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02)); padding: 12px; max-height: none; overflow: visible; }
        .ckpt-view h4 { margin: 0 0 6px 0; }
        .ckpt-view-summary { font-size: 0.85em; color: inherit; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 12px; }
        .ckpt-loading-message { font-size: 1.05em; font-weight: 600; color: inherit; padding: 16px 8px; }
        .ckpt-muted { opacity: 0.75; }
        .ckpt-stats { display: flex; flex-direction: column; gap: 6px; }
        .ckpt-stat-main { font-weight: 600; font-size: 0.95em; }
        .ckpt-stat-assumption { font-size: 0.82em; opacity: 0.85; }
        .ckpt-drive-list { display: flex; flex-wrap: wrap; gap: 8px; }
        .ckpt-drive { padding: 4px 8px; border-radius: 999px; background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); font-size: 0.82em; }
        .ckpt-file-tree { display: flex; flex-direction: column; gap: 6px; max-height: none; overflow: visible; }
        .ckpt-folder { border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--background-fill-primary); }
        .ckpt-folder.shared { border-color: rgba(240,200,40,0.45); }
        .ckpt-folder.shared > summary { background: rgba(240,200,40,0.12); }
        .ckpt-folder.unique { border-color: rgba(64, 120, 255, 0.35); }
        .ckpt-folder.unique > summary { background: rgba(64, 120, 255, 0.12); }
        .ckpt-folder summary { list-style: none; cursor: pointer; padding: 6px 8px; display: flex; align-items: center; gap: 8px; position: relative; }
        .ckpt-folder summary::-webkit-details-marker { display: none; }
        .ckpt-folder-name { flex: 1; font-weight: 600; }
        .ckpt-folder-size { font-size: 0.8em; color: inherit; opacity: 0.75; font-variant-numeric: tabular-nums; }
        .ckpt-folder-children { padding: 6px 8px 8px 18px; display: flex; flex-direction: column; gap: 4px; }
        .ckpt-file-row { display: flex; gap: 8px; align-items: center; font-size: 0.85em; padding: 2px 0; }
        .ckpt-file-row.shared { position: relative; }
        .ckpt-file-row.shared { color: #a66a00; background: rgba(240, 200, 40, 0.12); border-radius: 6px; padding: 2px 6px; }
        .ckpt-file-row.unique { background: rgba(64, 120, 255, 0.12); border-radius: 6px; padding: 2px 6px; }
        .ckpt-file-row.missing-only { color: #b71c1c; background: rgba(220, 60, 60, 0.16); border-radius: 6px; padding: 2px 6px; }
        .ckpt-file-row.missing { opacity: 0.6; }
        .ckpt-file-name { flex: 1; word-break: break-all; }
        .ckpt-file-size { min-width: 70px; text-align: right; color: inherit; opacity: 0.75; font-variant-numeric: tabular-nums; }
        .ckpt-shared-tag { font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.04em; border: 1px solid rgba(240,200,40,0.5); color: #a66a00; padding: 2px 6px; border-radius: 999px; background: rgba(240,200,40,0.12); }
        .ckpt-folder-tag { margin-left: 6px; }
        .ckpt-shared-tooltip { opacity: 0; visibility: hidden; pointer-events: none; position: absolute; top: 100%; left: 0; margin-top: 6px; padding: 6px 8px; background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); border-radius: 8px; box-shadow: var(--shadow-drop, 0 2px 6px rgba(0,0,0,0.12)); font-size: 0.8em; z-index: 10; min-width: 160px; transform: translateY(4px); transition: opacity 0.15s ease, transform 0.15s ease; transition-delay: 0s; }
        .ckpt-shared-tooltip-title { font-weight: 600; margin-bottom: 4px; }
        .ckpt-file-row.shared:hover .ckpt-shared-tooltip { opacity: 1; visibility: visible; transform: translateY(0); transition-delay: 2s; }
        .ckpt-folder.shared summary:hover .ckpt-shared-tooltip { opacity: 1; visibility: visible; transform: translateY(0); transition-delay: 2s; }
        .ckpt-path { font-size: 0.82em; color: inherit; opacity: 0.8; margin-bottom: 6px; word-break: break-all; }
        .ckpt-primary { font-size: 0.82em; color: inherit; opacity: 0.8; margin-bottom: 8px; word-break: break-all; }
        .ckpt-missing-summary { margin: 10px 0; }
        .ckpt-errors { padding: 8px 10px; border: 1px solid rgba(220, 80, 80, 0.4); background: var(--background-fill-primary); border-radius: 8px; font-size: 0.85em; }
        .ckpt-modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.45); display: none; align-items: center; justify-content: center; z-index: 9999; }
        .ckpt-modal { width: min(420px, 90vw); background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px; box-shadow: 0 12px 28px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 12px; }
        .ckpt-modal-title { font-weight: 700; font-size: 1.05em; }
        .ckpt-modal-body { font-size: 0.95em; color: inherit; opacity: 0.8; }
        .ckpt-modal-actions { display: flex; gap: 12px; justify-content: flex-end; }
        .ckpt-modal-btn { padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border-color-primary); background: var(--button-secondary-background-fill, #f0f0f0); cursor: pointer; font-weight: 600; }
        .ckpt-modal-btn.danger { background: rgba(220,80,80,0.15); border-color: rgba(220,80,80,0.5); color: #c62828; }
        </style>
        """

    def _build_tree_html(self):
        self._last_rendered_nodes_count = 0
        tree_parts = [
            "<div id='ckpt_tree_root' data-status='ready' class='ckpt-wrap'>",
            self._build_errors_html(),
        ]

        tree_parts.append("<div class='ckpt-panel'><div class='ckpt-panel-inner'>")
        tree_parts.append(self._model_assumption_html())
        rendered_nodes_html = []
        for family in self._tree_data:
            rendered_nodes_html.append(self._render_node(family, is_family=True))
        tree_parts.append("<div class='ckpt-tree'>")
        tree_parts.extend(rendered_nodes_html)
        tree_parts.append("</div></div></div></div>")
        return "".join(tree_parts)

    def _build_errors_html(self):
        if not self._errors:
            return ""
        errors_html = "<br>".join(html.escape(err) for err in self._errors[:10])
        return f"<div class='ckpt-errors'>{errors_html}</div>"

    def _build_filtered_tree_html(self, filter_text):
        query = str(filter_text).strip()
        self._current_tree_mode = "filtered"
        self._current_filter_text = query
        query_lower = query.casefold()
        matches = []
        for node_id, info in self._node_map.items():
            if info.get("type") != "model":
                continue
            model_types = list(info.get("models", set()))
            if len(model_types) != 1:
                continue
            model_type = model_types[0]
            model_name = self.get_model_name(model_type)
            if not isinstance(model_name, str):
                continue
            if query_lower not in model_name.casefold():
                continue
            status = self._model_direct_status_map.get(
                model_type, model_dropdowns.MODEL_FILE_STATUS_MISSING
            )
            display_label = self._decorate_status_label(model_name, status)
            matches.append((model_name, display_label, node_id, model_type))

        matches.sort(key=lambda item: item[0].casefold())

        self._last_rendered_nodes_count = 0
        title = (
            f"Filtered Finetunes for '{html.escape(query)}' "
            f"<span class='ckpt-muted'>({len(matches)} match{'es' if len(matches) != 1 else ''})</span>"
        )
        parts = [
            "<div id='ckpt_tree_root' data-status='ready' class='ckpt-wrap'>",
            self._build_errors_html(),
            "<div class='ckpt-panel'><div class='ckpt-panel-inner'>",
            f"<div class='ckpt-loading-message'>{title}</div>",
            self._model_assumption_html(),
            "<div class='ckpt-tree'>",
        ]
        if not matches:
            parts.append(
                "<div class='ckpt-leaf'><div class='ckpt-row'>"
                "<div class='ckpt-label'>No finetune matches your filter.</div>"
                "</div></div>"
            )
        for _, display_label, node_id, model_type in matches:
            variant_children = [
                {
                    "id": variant["id"],
                    "label": variant["label"],
                    "type": variant.get("type", "variant"),
                    "children": [],
                }
                for variant in self._ordered_variants_for_display(model_type)
            ]
            model_node = {
                "id": node_id,
                "label": display_label,
                "type": "model",
                "children": variant_children,
            }
            parts.append(self._render_node(model_node))
        parts.extend(["</div></div></div></div>"])
        return "".join(parts)

    def _model_assumption_html(self):
        transformer_quantization = self.server_config.get("transformer_quantization", self.transformer_quantization)
        text_encoder_quantization = self.text_encoder_quantization
        cfg_line = (
            "Missing files assume config: "
            f"transformer qtype={html.escape(str(transformer_quantization))}, "
            f"text-encoder qtype={html.escape(str(text_encoder_quantization))}"
        )
        return f"<div class='ckpt-stat-assumption'>{cfg_line}</div>"

    def _build_stats_html(self):
        total_label = self._format_gb(self._stats_total)
        drive_html = ""
        if self._stats_by_drive:
            drive_items = []
            for drive, size in self._stats_by_drive:
                drive_items.append(
                    f"<span class='ckpt-drive'>{html.escape(drive)}: {self._format_gb(size)}</span>"
                )
            drive_html = f"<div class='ckpt-drive-list'>{''.join(drive_items)}</div>"
        return (
            "<div class='ckpt-stats' data-status='ready'>"
            f"<div class='ckpt-stat-main'>Total Used: {total_label}</div>"
            "<div class='ckpt-stat-assumption'>Models and system checkpoints combined. Shared files count once.</div>"
            f"{drive_html}</div>"
        )

    def _render_node(self, node, is_family=False, native_expand=False):
        self._last_rendered_nodes_count += 1
        node_id = node["id"]
        info = self._node_map.get(node_id, {})
        label = node.get("label", info.get("label", node_id))
        unique_size_value = info.get("unique_size", 0)
        shared_size_value = info.get("shared_size", 0)
        unique = self._format_gb(unique_size_value)
        shared = self._format_gb(shared_size_value)
        exclusive_label = self._format_size(info.get("exclusive_size", 0))
        total_label = self._format_size(unique_size_value + shared_size_value)
        missing_count = int(info.get("missing_count", 0) or 0)
        node_type = info.get("type", node.get("type", ""))

        children = node.get("children", [])
        click_view = node_type in ("variant", "missing", "system")
        row_html = self._render_row(
            label,
            node_id,
            unique,
            shared,
            exclusive_label,
            total_label,
            node_type,
            missing_count=missing_count,
            click_view=click_view,
        )
        if children:
            node_id_attr = html.escape(node_id, quote=True)
            is_open = node_id in self._expanded_nodes
            open_attr = " open" if is_open else ""
            child_html = ""
            if is_open or native_expand:
                child_html = "".join(self._render_node(child, native_expand=native_expand) for child in children)
            click = "" if native_expand else " onclick='handlemodelSummaryClick(event, this);'"
            return (
                f"<details class='ckpt-node' data-node-id=\"{node_id_attr}\"{open_attr}>"
                f"<summary{click}>{row_html}</summary>"
                f"<div class='ckpt-children'>{child_html}</div></details>"
            )
        return f"<div class='ckpt-leaf'>{row_html}</div>"

    def _render_row(
        self,
        label,
        node_id,
        unique,
        shared,
        exclusive_label,
        total_label,
        node_type,
        missing_count=0,
        click_view=False,
    ):
        label_display = html.escape(label)
        for status, prefix in model_dropdowns.MODEL_STATUS_PREFIXES.items():
            if label.startswith(prefix + " "):
                status_name, status_title = {
                    model_dropdowns.MODEL_FILE_STATUS_MISSING: ("missing", "Not installed"),
                    model_dropdowns.MODEL_FILE_STATUS_PARTIAL: ("partial", "Partially available"),
                    model_dropdowns.MODEL_FILE_STATUS_EXPECTED: ("available", "Available"),
                }[status]
                label_display = f"<span class='wangp-model-status' data-wangp-availability='{status_name}' role='img' aria-label='{status_title}' title='{status_title}'></span>{html.escape(label[len(prefix) + 1:])}"
                break
        action_label = self._node_map[node_id]["path_label"] if node_id.startswith("system::") else label
        label_attr = html.escape(action_label, quote=True)
        node_id_attr = html.escape(node_id, quote=True)
        type_attr = html.escape(node_type, quote=True)
        row_class = "ckpt-row clickable" if click_view else "ckpt-row"
        row_onclick = (
            "onclick='handlemodelRowClick(event, this);'" if click_view else ""
        )
        label_class = "ckpt-label ckpt-label-missing" if node_type == "missing" else "ckpt-label"
        if node_type == "missing":
            file_count_label = f"{missing_count} file" if missing_count == 1 else f"{missing_count} files"
            sizes_html = (
                "<div class='ckpt-sizes ckpt-sizes-missing'>"
                f"<div class='ckpt-size-missing'>{html.escape(file_count_label)}</div>"
                "</div>"
            )
        else:
            unique_badge_class = "ckpt-badge ckpt-badge-unique"
            shared_badge_class = "ckpt-badge ckpt-badge-shared"

            if unique == "0.00 GB":
                unique_badge_class += " ckpt-badge-zero"
            if shared == "0.00 GB":
                shared_badge_class += " ckpt-badge-zero"

            unique_badge = f"<span class='{unique_badge_class}'>{unique}</span>"
            shared_badge = f"<span class='{shared_badge_class}'>{shared}</span>"
            sizes_html = (
                "<div class='ckpt-sizes'>"
                f"<div class='ckpt-size-unique' title='Unique size'>{unique_badge}</div>"
                f"<div class='ckpt-size-shared' title='Shared size'>{shared_badge}</div>"
                "</div>"
            )
        show_view_button = node_type not in ("variant", "missing")
        show_delete_button = node_type != "missing" and (not node_id.startswith("system::") or bool(self._node_map[node_id]["files"]))
        action_buttons = []
        if show_view_button:
            action_buttons.append(
                f"<button class='ckpt-action-btn ckpt-view-btn' data-node-id=\"{node_id_attr}\" onclick='event.preventDefault(); event.stopPropagation(); handlemodelAction(this, \"view\");'>View</button>"
            )
        if show_delete_button:
            action_buttons.append(
                f"<button class='ckpt-action-btn ckpt-delete' data-node-id=\"{node_id_attr}\" data-node-label=\"{label_attr}\" data-exclusive-size=\"{exclusive_label}\" data-total-size=\"{total_label}\" onclick='event.preventDefault(); event.stopPropagation(); handlemodelAction(this, \"delete\");'>Delete</button>"
            )
        actions_html = f"<div class='ckpt-actions'>{''.join(action_buttons)}</div>"
        return (
            f"<div class='{row_class}' data-node-id=\"{node_id_attr}\" data-node-label=\"{label_attr}\" data-node-type=\"{type_attr}\" {row_onclick}>"
            f"<div class='{label_class}'>{label_display}</div>"
            f"{sizes_html}"
            f"{actions_html}"
            "</div>"
        )

    def _build_empty_view_html(self):
        return (
            "<div class='ckpt-view'>"
            "<h4>Files</h4>"
            "<div class='ckpt-view-summary'>Select a branch to view its files.</div>"
            "</div>"
        )

    def _build_loading_html(self):
        return (
            "<div id='ckpt_tree_root' data-status='loading' class='ckpt-wrap'>"
            "<div class='ckpt-panel'><div class='ckpt-panel-inner'>"
            "<div class='ckpt-loading-message' role='status'>Computing model and system disk usage…</div>"
            "</div></div></div>"
        )

    def _build_view_html(self, node_id):
        node = self._node_map.get(node_id)
        if node is None:
            return self._build_empty_view_html()
        if node.get("type") == "missing":
            return self._build_missing_view_html(node)

        files = sorted(node["files"])
        shared_set = node["shared_files"]
        exclude_models = node["models"] if len(node["models"]) == 1 else ()

        shared_info = {}
        for path in shared_set:
            shared_labels = self._shared_model_labels(path, exclude_models=exclude_models)
            if not shared_labels:
                shared_labels = ["Shared with another model"]
            shared_info[path] = shared_labels
        file_tree = self._build_file_tree(files, shared_set, shared_info)
        file_tree_html = self._render_file_tree(file_tree)

        unique = self._format_gb(node["unique_size"])
        shared = self._format_gb(node["shared_size"])
        label_text = node["label"]
        path_label = node.get("path_label") or label_text
        header = html.escape(path_label)
        unique_badge = f"<span class='ckpt-badge ckpt-badge-unique'>{unique}</span>"
        shared_badge = f"<span class='ckpt-badge ckpt-badge-shared'>{shared}</span>"
        shared_count = len(shared_set)
        unique_count = max(0, len(files) - shared_count)
        summary = (
            f"<span class='ckpt-muted'>{unique_count} files</span> | unique {unique_badge} "
            f"<span class='ckpt-muted'>{shared_count} files</span> | shared {shared_badge}"
        )
        missing_files = self._collect_missing_files_for_node(node)
        primary_paths = node.get("primary_paths", [])
        primary_html = ""
        if primary_paths:
            label = "model path" if len(primary_paths) == 1 else "model paths"
            lines = "<br>".join(html.escape(path) for path in primary_paths)
            primary_html = f"<div class='ckpt-primary'><strong>{label}:</strong><br>{lines}</div>"
        missing_html = self._render_missing_files_block(missing_files) if node.get("type") in ("model", "system") and missing_files else ""
        return (
            "<div class='ckpt-view'>"
            f"<h4><strong>{header}</strong></h4>"
            f"{primary_html}"
            f"<div class='ckpt-view-summary'>{summary}</div>"
            f"{file_tree_html}{missing_html}</div>"
        )

    def _build_missing_view_html(self, node):
        missing_files = self._collect_missing_files_for_node(node)
        label_text = node["label"]
        path_label = node.get("path_label") or label_text
        header = html.escape(path_label)
        file_tree_html = self._render_missing_files_block(missing_files, show_title=True)
        return (
            "<div class='ckpt-view'>"
            f"<h4><strong>{header}</strong></h4>"
            f"{file_tree_html}</div>"
        )

    def _collect_missing_files_for_node(self, node):
        missing_files = set(node.get("missing_files", set()))
        if node.get("type") == "model":
            model_types = list(node.get("models", set()))
            if len(model_types) == 1:
                model_type = model_types[0]
                for variant in self._ordered_variants_for_display(model_type):
                    missing_files.update(variant.get("missing_files", set()))
        return sorted(missing_files, key=lambda path: self._display_path(path).lower())

    def _render_missing_files_block(self, missing_files, show_title=True):
        if not missing_files:
            return "<div class='ckpt-file-tree'><div class='ckpt-muted'>No missing files.</div></div>"
        file_rows = []
        for path in missing_files:
            display_path = html.escape(self._display_path(path))
            full_path = html.escape(path)
            file_rows.append(
                f"<div class='ckpt-file-row missing-only' title='{full_path}'>"
                f"<span class='ckpt-file-name'>{display_path}</span>"
                "</div>"
            )
        title_html = (
            f"<div class='ckpt-view-summary ckpt-missing-summary'><span class='ckpt-muted'>{len(missing_files)} files</span> | missing</div>"
            if show_title
            else ""
        )
        return f"{title_html}<div class='ckpt-file-tree'>{''.join(file_rows)}</div>"

    def _build_file_tree(self, files, shared_set, shared_info=None):
        if shared_info is None:
            shared_info = {}
        root = {
            "children": {},
            "files": [],
            "size": 0,
            "all_shared": False,
            "all_unique": False,
            "has_files": False,
            "shared_with": set(),
            "shared_count": 0,
            "is_root": True,
        }
        for path in files:
            info = self._file_info.get(path, {})
            display = self._display_path(path)
            parts = self._split_display_path(display)
            name = parts[-1] if parts else display
            shared_with = shared_info.get(path, [])
            entry = {
                "path": path,
                "display": display,
                "name": name,
                "size": info.get("size", 0),
                "exists": info.get("exists", False),
                "shared": path in shared_set,
                "shared_with": shared_with,
            }

            node = root
            for part in parts[:-1]:
                node = node["children"].setdefault(
                    part,
                    {
                        "children": {},
                        "files": [],
                        "size": 0,
                        "all_shared": False,
                        "all_unique": False,
                        "has_files": False,
                        "shared_with": set(),
                        "shared_count": 0,
                        "is_root": False,
                    },
                )
            node["files"].append(entry)

        self._compute_folder_sizes(root)
        return root

    def _compute_folder_sizes(self, node):
        total = sum(entry["size"] for entry in node["files"])
        has_files = len(node["files"]) > 0
        all_shared = all(entry["shared"] for entry in node["files"]) if node["files"] else True
        all_unique = all(not entry["shared"] for entry in node["files"]) if node["files"] else True
        shared_with = set()
        shared_count = 0
        for entry in node["files"]:
            if entry.get("shared"):
                shared_with.update(entry.get("shared_with", []))
        for child in node["children"].values():
            (
                child_total,
                child_has_files,
                child_all_shared,
                child_shared_with,
                child_shared_count,
                child_all_unique,
            ) = self._compute_folder_sizes(child)
            total += child_total
            has_files = has_files or child_has_files
            all_shared = all_shared and child_all_shared
            all_unique = all_unique and child_all_unique
            shared_with.update(child_shared_with)
            shared_count += child_shared_count
        node["size"] = total
        node["has_files"] = has_files
        node["all_shared"] = has_files and all_shared
        node["all_unique"] = has_files and all_unique
        node["shared_with"] = shared_with
        if node["all_shared"] and not node.get("is_root"):
            shared_count += 1
        node["shared_count"] = shared_count
        return total, has_files, node["all_shared"], shared_with, shared_count, node["all_unique"]

    def _render_file_tree(self, node):
        open_shared_path = node.get("shared_count", 0) == 1
        parts = []
        for name in sorted(node["children"].keys(), key=lambda n: n.lower()):
            parts.append(
                self._render_folder(
                    name,
                    node["children"][name],
                    level=0,
                    open_shared_path=open_shared_path,
                )
            )

        for entry in sorted(node["files"], key=lambda e: e["name"].lower()):
            parts.append(self._render_file_entry(entry))

        return "<div class='ckpt-file-tree'>" + "".join(parts) + "</div>"

    def _render_folder(self, name, node, level, open_shared_path=False):
        name, node = self._collapse_single_child_folder(name, node)
        folder_name = html.escape(name)
        size_label = self._format_size(node.get("size", 0))
        shared_tag = ""
        shared_tooltip = ""
        folder_class = "ckpt-folder"
        if node.get("all_shared"):
            shared_tag = "<span class='ckpt-shared-tag ckpt-folder-tag'>Shared</span>"
            folder_class += " shared"
            shared_with = sorted(node.get("shared_with", []), key=lambda value: value.lower())
            if shared_with:
                lines = "".join(
                    f"<div>{html.escape(label)}</div>" for label in shared_with
                )
                shared_tooltip = (
                    "<div class='ckpt-shared-tooltip'>"
                    "<div class='ckpt-shared-tooltip-title'>Shared with</div>"
                    f"{lines}</div>"
                )
        elif node.get("all_unique"):
            folder_class += " unique"
        open_attr = ""
        if level < 1 or (open_shared_path and node.get("shared_count", 0) == 1):
            open_attr = " open"
        children_html = []
        for child_name in sorted(node["children"].keys(), key=lambda n: n.lower()):
            children_html.append(
                self._render_folder(
                    child_name,
                    node["children"][child_name],
                    level + 1,
                    open_shared_path=open_shared_path,
                )
            )
        for entry in sorted(node["files"], key=lambda e: e["name"].lower()):
            children_html.append(self._render_file_entry(entry))
        return (
            f"<details class='{folder_class}'{open_attr}>"
            "<summary>"
            f"<span class='ckpt-folder-name'>{folder_name}</span>"
            f"{shared_tag}"
            f"<span class='ckpt-folder-size'>{size_label}</span>"
            f"{shared_tooltip}"
            "</summary>"
            "<div class='ckpt-folder-children'>"
            + "".join(children_html)
            + "</div></details>"
        )

    def _collapse_single_child_folder(self, name, node):
        parts = [name]
        current = node
        while True:
            if current.get("files"):
                break
            children = current.get("children", {})
            if len(children) != 1:
                break
            child_name, child_node = next(iter(children.items()))
            parts.append(child_name)
            current = child_node
        return os.sep.join(parts), current

    def _render_file_entry(self, entry):
        classes = ["ckpt-file-row"]
        if entry.get("shared"):
            classes.append("shared")
        else:
            classes.append("unique")
        if not entry.get("exists"):
            classes.append("missing")
        class_attr = " ".join(classes)
        name = html.escape(entry.get("name", ""))
        size_label = self._format_size(entry.get("size", 0))
        display = html.escape(entry.get("display", ""))
        shared_tag = "<span class='ckpt-shared-tag'>Shared</span>" if entry.get("shared") else ""
        shared_tooltip = ""
        if entry.get("shared"):
            shared_with = entry.get("shared_with", [])
            if shared_with:
                lines = "".join(
                    f"<div>{html.escape(label)}</div>" for label in shared_with
                )
                shared_tooltip = (
                    "<div class='ckpt-shared-tooltip'>"
                    "<div class='ckpt-shared-tooltip-title'>Shared with</div>"
                    f"{lines}</div>"
                )
        return (
            f"<div class='{class_attr}' title='{display}'>"
            f"<span class='ckpt-file-name'>{name}</span>"
            f"{shared_tag}"
            f"<span class='ckpt-file-size'>{size_label}</span>"
            f"{shared_tooltip}"
            "</div>"
        )

    def _split_display_path(self, display_path):
        drive, rest = os.path.splitdrive(display_path)
        if drive:
            drive = drive.rstrip("\\/")
        rest = rest.replace("\\", "/")
        parts = [part for part in rest.split("/") if part]
        if drive:
            return [drive] + parts
        if display_path.startswith("\\\\"):
            return ["UNC"] + parts
        return parts

    def _add_file(self, files_set, value, force_folder=None, lora_dir=None):
        path = self._resolve_path(value, force_folder=force_folder, lora_dir=lora_dir)
        if path and os.path.isfile(path):
            files_set.add(path)

    def _download_root_name(self):
        root = fl.get_download_location()
        if not root or os.path.isabs(root):
            return ""
        norm = os.path.normpath(root)
        if not norm or norm == ".":
            return ""
        return norm.split(os.sep)[0]

    def _resolve_repo_relative_path(self, rel_path):
        if not rel_path:
            return None
        rel_path = str(rel_path)
        if rel_path.startswith("http"):
            return None
        root_name = self._download_root_name()
        if not root_name:
            return None
        rel_norm = os.path.normpath(rel_path)
        parts = rel_norm.split(os.sep)
        if parts and parts[0].lower() == root_name.lower():
            return self._normalize_path(os.path.join(self._repo_root, rel_norm))
        return None

    def _get_local_model_filename(self, model_filename, use_locator=True, extra_paths=None, lora_dir=None):
        try:
            return fl.get_local_model_filename(model_filename, use_locator=use_locator, extra_paths=extra_paths, lora_dir=lora_dir)
        except Exception:
            return None

    def _locator_path(self, value, lora_dir=None):
        if not value:
            return ""
        if not isinstance(value, str):
            value = str(value)
        try:
            return fl.extract_alternate_path(value, lora_dir=lora_dir)
        except Exception:
            source_value = value.split("|", 1)[0]
            return os.path.basename(source_value) if source_value.startswith("http") else source_value

    def _locator_basename(self, value, lora_dir=None):
        path = self._locator_path(value, lora_dir=lora_dir)
        return os.path.basename(path) if path else ""

    def _download_location(self, value, force_folder=None, lora_dir=None):
        try:
            return fl.get_download_location(value, force_path=force_folder, lora_dir=lora_dir)
        except Exception:
            source_value = str(value).split("|", 1)[0]
            filename = os.path.basename(source_value) if source_value.startswith("http") else source_value
            return fl.get_download_location(filename, force_path=force_folder)

    def _resolve_path(self, value, force_folder=None, lora_dir=None):
        if not value:
            return None
        if not isinstance(value, str):
            value = str(value)
        if os.path.isabs(value):
            abs_path = self._normalize_path(value)
            if os.path.isfile(abs_path):
                return abs_path
            basename = os.path.basename(abs_path)
            return abs_path
        repo_rel = self._resolve_repo_relative_path(value)
        if repo_rel:
            return repo_rel

        local_path = self._get_local_model_filename(value, extra_paths=force_folder, lora_dir=lora_dir)
        if local_path:
            return self._normalize_path(local_path)

        expected = self._download_location(value, force_folder=force_folder, lora_dir=lora_dir)
        return self._normalize_path(expected)

    def _resolve_download_relpath(self, rel_path):
        if rel_path and os.path.isabs(rel_path):
            return self._normalize_path(rel_path)
        repo_rel = self._resolve_repo_relative_path(rel_path)
        if repo_rel:
            return repo_rel
        local = fl.locate_file(rel_path, error_if_none=False)
        if local:
            return self._normalize_path(local)
        return self._normalize_path(fl.get_download_location(rel_path))

    def _resolve_folder_path(self, source_folder, target_folder):
        parts = []
        if target_folder:
            parts.append(target_folder)
        if source_folder:
            parts.append(source_folder)
        rel_path = os.path.join(*parts) if parts else ""
        if rel_path and os.path.isabs(rel_path):
            return self._normalize_path(rel_path)
        repo_rel = self._resolve_repo_relative_path(rel_path)
        if repo_rel:
            return repo_rel
        if not rel_path:
            return self._normalize_path(fl.get_download_location())
        folder = fl.locate_folder(rel_path, error_if_none=False)
        if folder:
            return self._normalize_path(folder)
        return self._normalize_path(fl.get_download_location(rel_path))

    def _list_folder_files(self, folder_path):
        files = set()
        if not folder_path or not os.path.isdir(folder_path):
            return files
        for root, _, filenames in os.walk(folder_path):
            for name in filenames:
                files.add(self._normalize_path(os.path.join(root, name)))
        return files

    def _combine_download_path(self, target_folder, source_folder, filename):
        parts = []
        if target_folder:
            parts.append(target_folder)
        if source_folder:
            parts.append(source_folder)
        parts.append(filename)
        return os.path.join(*parts)

    def _compute_list(self, filename):
        if not filename:
            return []
        return [os.path.basename(str(filename))]

    def _ensure_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _safe_get_lora_dir(self, model_type):
        try:
            return self.get_lora_dir(model_type)
        except Exception:
            return None

    def _normalize_path(self, path):
        return os.path.normcase(os.path.abspath(path))

    def _format_gb(self, size_bytes):
        size_gb = size_bytes / (1024 ** 3)
        return f"{size_gb:.2f} GB"

    def _format_size(self, size_bytes):
        if size_bytes >= 1024 ** 3:
            return f"{size_bytes / (1024 ** 3):.2f} GB"
        if size_bytes >= 1024 ** 2:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        if size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes} B"

    def _display_path(self, path):
        try:
            rel = os.path.relpath(path, self._repo_root)
            if not rel.startswith(".."):
                return rel
        except Exception:
            pass
        return path

    def _get_model_roots(self):
        roots = []
        for root in getattr(fl, "_models_paths", []) or []:
            if not root:
                continue
            if os.path.isabs(root):
                abs_root = root
            else:
                abs_root = os.path.join(self._repo_root, root)
            abs_root = self._normalize_path(abs_root)
            if abs_root not in roots:
                roots.append(abs_root)
        return roots

    def _build_basename_index(self, basenames, min_size=1):
        basenames = {name.lower() for name in basenames if name}
        if not basenames:
            return {}
        found = {}
        remaining = set(basenames)
        for root in self._get_model_roots():
            if not remaining:
                break
            if not os.path.isdir(root):
                continue
            for name in list(remaining):
                candidate = os.path.join(root, name)
                if os.path.isfile(candidate):
                    try:
                        if os.path.getsize(candidate) < min_size:
                            continue
                    except OSError:
                        continue
                    found[name] = self._normalize_path(candidate)
                    remaining.discard(name)
            if not remaining:
                break
            if os.path.normcase(root) == os.path.normcase(self._repo_root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    key = filename.lower()
                    if key in remaining:
                        full_path = os.path.join(dirpath, filename)
                        try:
                            if os.path.getsize(full_path) < min_size:
                                continue
                        except OSError:
                            continue
                        found[key] = self._normalize_path(full_path)
                        remaining.discard(key)
                        if not remaining:
                            break
                if not remaining:
                    break
        return found

    def _find_in_model_roots(self, basename, min_size=1):
        if not basename:
            return None
        key = basename.lower()
        if key in self._basename_index:
            return self._basename_index.get(key)
        found = self._build_basename_index({basename}, min_size=min_size)
        if found:
            self._basename_index.update(found)
            return self._basename_index.get(key)
        self._basename_index[key] = None
        return None

    def _repair_missing_paths(self):
        missing = []
        for path in self._file_usage_models.keys():
            try:
                size = os.path.getsize(path) if os.path.isfile(path) else -1
            except OSError:
                size = -1
            if size <= 0:
                missing.append(path)
        if not missing:
            return
        basenames = {os.path.basename(path) for path in missing}
        found = self._build_basename_index(basenames, min_size=1)
        if not found:
            return
        for old_path in list(missing):
            key = os.path.basename(old_path).lower()
            new_path = found.get(key)
            if not new_path or new_path == old_path:
                continue
            self._swap_path(old_path, new_path)

    def _swap_path(self, old_path, new_path):
        usage_ids = self._file_usage.pop(old_path, set())
        if usage_ids:
            self._file_usage[new_path].update(usage_ids)
            for usage_id in usage_ids:
                files = self._usage_files.get(usage_id)
                if files is not None:
                    files.discard(old_path)
                    files.add(new_path)
                model_type = self._variant_to_model.get(usage_id)
                if model_type:
                    for variant in self._model_variants.get(model_type, []):
                        if variant.get("id") != usage_id:
                            continue
                        files_set = variant.get("files")
                        if isinstance(files_set, set):
                            files_set.discard(old_path)
                            files_set.add(new_path)
                        primary_paths = variant.get("primary_paths", [])
                        variant["primary_paths"] = [
                            new_path if path == old_path else path for path in primary_paths
                        ]
                        break

        model_types = self._file_usage_models.pop(old_path, set())
        if model_types:
            self._file_usage_models[new_path].update(model_types)
            for model_type in model_types:
                files = self._model_files.get(model_type)
                if files is not None:
                    files.discard(old_path)
                    files.add(new_path)
                primary_paths = self._model_primary.get(model_type)
                if primary_paths:
                    self._model_primary[model_type] = [
                        new_path if path == old_path else path for path in primary_paths
                    ]
