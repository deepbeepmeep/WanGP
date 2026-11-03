import gradio as gr
from shared.gradio.progress import WangpProgress
import json
import os
import traceback
from shared.utils.config_store import config_lock, write_config, update_config
from shared.utils.plugins import INSTALLED_REMOTE_PLUGINS_KEY, WAN2GPPlugin, compare_release_metadata, is_wangp_compatible, normalize_plugin_types, plugin_id_from_url
from . import discovery
from shared.utils.bundled_apps import FEATURED_APPS, REVIEW_KEY

DISCOVER_PLUGIN_TYPE_TABS = (
    ("all", "All", None),
    ("apps", "Apps", "app"),
    ("extensions", "Extensions", "extension"),
    ("processors", "Processors", "processor"),
    ("models", "Models", "model"),
)

class PluginManagerUIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Apps & PlugIn Manager"
        self.version = "1.8.0"
        self.description = "A built-in UI for managing, installing, and updating Wan2GP plugins"
        self.WanGP_version = ""
        self.quit_application = None
        self.restart_application = None
        self.restart_required = False

    def setup_ui(self):
        self.request_global("app")
        self.request_global("server_config")
        self.request_global("server_config_filename")
        self.request_global("quit_application")
        self.request_global("restart_application")
        self.request_global("WanGP_version")
        self.request_component("main")
        self.request_component("main_tabs")
        
        self.add_tab(
            tab_id="plugin_manager_tab",
            label="Apps & PlugIn Manager",
            component_constructor=self.create_plugin_manager_ui,
        )

    def _build_navigation_html(self):
        return self.app.app_bar.refresh_default()

    def information_cards_html(self):
        catalog = dict(getattr(self, '_community_plugins_cache', None) or {})
        catalog.update({p['id']: p for p in self.app.plugin_manager.get_plugins_info()})
        return ''.join(discovery.information_card(dict(plugin, id=pid)) for pid, plugin in catalog.items())

    def refresh_html(self):
        return self._build_plugins_html(), *self._build_available_plugins_html_outputs()

    def _get_js_script_html(self):
        js_code = """
            () => {
                function pluginRoot() {
                    if (window.gradioApp) {
                        return window.gradioApp();
                    }
                    const app = document.querySelector('gradio-app');
                    return app ? (app.shadowRoot || app) : document;
                }

                function updateGradioInput(elem_id, value) {
                    const root = pluginRoot();
                    const input = root.querySelector(`#${elem_id} textarea, #${elem_id} input`);
                    if (input) {
                        input.value = value;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        input.dispatchEvent(new Event('change', { bubbles: true }));
                        return true;
                    }
                    return false;
                }

                function makePayload(data) {
                    return JSON.stringify(Object.assign({ nonce: `${Date.now()}-${Math.random()}` }, data));
                }

                function makeSortable() {
                    const root = pluginRoot();
                    const userPluginList = root.querySelector('#user-plugin-list');
                    if (!userPluginList) return;
                    if (userPluginList.dataset.sortableBound === '1') return;
                    userPluginList.dataset.sortableBound = '1';

                    let draggedItem = null;
                    let orderBeforeDrag = "";

                    function currentPluginOrder() {
                        return Array.from(userPluginList.querySelectorAll('.plugin-item')).map(item => item.dataset.pluginId || '').join('|');
                    }

                    userPluginList.addEventListener('dragstart', e => {
                        draggedItem = e.target.closest('.plugin-item');
                        if (!draggedItem) return;
                        orderBeforeDrag = currentPluginOrder();
                        draggedItem.classList.add('dragging');
                        if (e.dataTransfer) {
                            e.dataTransfer.effectAllowed = 'move';
                            e.dataTransfer.setData('text/plain', draggedItem.dataset.pluginId || '');
                        }
                        setTimeout(() => {
                            if (draggedItem) draggedItem.style.opacity = '0.5';
                        }, 0);
                    });

                    userPluginList.addEventListener('dragend', e => {
                        setTimeout(() => {
                             if (draggedItem) {
                                draggedItem.style.opacity = '1';
                                draggedItem.classList.remove('dragging');
                                draggedItem = null;
                             }
                             if (orderBeforeDrag && orderBeforeDrag !== currentPluginOrder()) {
                                handlePluginOrderSave(false, true);
                             }
                             orderBeforeDrag = "";
                        }, 0);
                    });

                    userPluginList.addEventListener('dragover', e => {
                        e.preventDefault();
                        if (e.dataTransfer) {
                            e.dataTransfer.dropEffect = 'move';
                        }
                        const afterElement = getDragAfterElement(userPluginList, e.clientY);
                        if (draggedItem) {
                            if (afterElement === draggedItem) return;
                            if (afterElement == null) {
                                userPluginList.appendChild(draggedItem);
                            } else {
                                userPluginList.insertBefore(draggedItem, afterElement);
                            }
                        }
                    });

                    userPluginList.addEventListener('drop', e => {
                        e.preventDefault();
                    });

                    function getDragAfterElement(container, y) {
                        const draggableElements = [...container.querySelectorAll('.plugin-item:not(.dragging)')];
                        return draggableElements.reduce((closest, child) => {
                            const box = child.getBoundingClientRect();
                            const offset = y - box.top - box.height / 2;
                            if (offset < 0 && offset > closest.offset) {
                                return { offset: offset, element: child };
                            } else {
                                return closest;
                            }
                        }, { offset: Number.NEGATIVE_INFINITY }).element;
                    }
                }
                
                function observeUserPluginList() {
                    const root = pluginRoot();
                    if (!root) {
                        setTimeout(observeUserPluginList, 400);
                        return;
                    }
                    if (root.dataset.pluginSortableObserver === '1') {
                        makeSortable();
                        return;
                    }
                    root.dataset.pluginSortableObserver = '1';
                    const observer = new MutationObserver(() => {
                        makeSortable();
                    });
                    observer.observe(root, { childList: true, subtree: true });
                    makeSortable();
                }
                
                setTimeout(observeUserPluginList, 200);
                setTimeout(makeSortable, 500);

                window.handlePluginAction = function(button, action) {
                    const pluginItem = button.closest('.plugin-item');
                    const pluginId = pluginItem.dataset.pluginId;
                    const payload = makePayload({ action: action, plugin_id: pluginId });
                    updateGradioInput('plugin_action_input', payload);
                };
                
                window.handleStoreInstall = function(button, url) {
                    const payload = makePayload({ action: 'install_from_store', url: url });
                    updateGradioInput('plugin_action_input', payload);
                };

                function collectEnabledUserPlugins() {
                    const root = pluginRoot();
                    const user_container = root.querySelector('#user-plugin-list');
                    if (!user_container) return null;
                    
                    const user_plugins = user_container.querySelectorAll('.plugin-item');
                    return Array.from(user_plugins).map(item => item.dataset.pluginId);
                }

                window.handlePluginOrderSave = function(restart, silent) {
                    const enabledUserPlugins = collectEnabledUserPlugins();
                    if (silent && enabledUserPlugins === null) return;
                    
                    if (!restart) window.WanGPAppBar?.order(enabledUserPlugins);
                    const payload = makePayload({ restart: restart, enabled_plugins: enabledUserPlugins });
                    updateGradioInput('save_action_input', payload);

                    if (restart) {
                        setTimeout(() => {
                            document.body.innerHTML = "<div style='display:flex;justify-content:center;align-items:center;height:100vh;background-color:#0b0f19;color:#e5e7eb;font-family:sans-serif;text-align:center;'><h2>WanGP is restarting...<br><br>You can safely close this tab.<br>A new tab will open shortly.</h2></div>";
                            window.open('', '_self', '');
                            window.close();
                        }, 1000);
                    }
                };

                window.handleRestart = function() {
                    handlePluginOrderSave(true, false);
                };
            }
        """
        return f"{js_code}"
    
    def _get_community_plugins_info(self):
        if hasattr(self, '_community_plugins_cache') and self._community_plugins_cache is not None:
            return self._community_plugins_cache
        try:
            self._community_plugins_cache = self.app.plugin_manager.get_merged_catalog_entries(use_remote=True)
            return self._community_plugins_cache
        except Exception as e:
            print(f"[PluginManager] Could not fetch community plugins info: {e}")
            self._community_plugins_cache = {}
            return {}

    def _build_community_plugins_html_outputs(self):
        return [self._build_community_plugins_html(plugin_type) for _, _, plugin_type in DISCOVER_PLUGIN_TYPE_TABS]

    def _build_available_plugins_html_outputs(self):
        return self._build_local_available_plugins_html_outputs() + self._build_community_plugins_html_outputs() + [self._build_navigation_html()]

    def _available_outputs_count(self):
        return len(DISCOVER_PLUGIN_TYPE_TABS) * 2 + 1

    def _build_type_badges_html(self, plugin):
        return "<span class='plugin-type-badges'>" + "".join(f"<span class='plugin-type-badge'>{plugin_type.replace('_', ' ').title()}</span>" for plugin_type in normalize_plugin_types(plugin.get('type'))) + "</span>"

    def _wan2gp_requirement_html(self, plugin):
        required_version = plugin.get('wan2gp_version') or plugin.get('wangp_version') or ''
        required_version = required_version.strip() if isinstance(required_version, str) else str(required_version).strip()
        if not required_version or is_wangp_compatible(required_version, self.WanGP_version):
            return required_version, False, ""
        text = f"Need WanGP v{required_version}"
        return required_version, True, f"<span class='plugin-incompatible-badge' title='{text}'>{text}</span>"

    def _requirement_actions_html(self, required_version):
        return f"<div class='plugin-item-actions'><span class='plugin-requirements-not-met'>Need WanGP v{required_version}</span></div>"

    def _empty_community_message(self, plugin_type):
        if plugin_type is None:
            return "All available community apps and plugins are already installed."
        label = next(label for _, label, tab_type in DISCOVER_PLUGIN_TYPE_TABS if tab_type == plugin_type)
        return f"No available {label.lower()} found."

    def _build_community_plugins_html(self, plugin_type=None):
        try:
            installed_plugin_ids = {p['id'] for p in self.app.plugin_manager.get_plugins_info()}
            remote_plugins = self._get_community_plugins_info()
            community_plugins = [
                dict(p, id=plugin_id) for plugin_id, p in remote_plugins.items()
                if plugin_id not in installed_plugin_ids and (plugin_type is None or plugin_type in normalize_plugin_types(p.get('type')))
            ]
            community_plugins.sort(key=lambda p: (not p.get('important'), (p.get('name') or plugin_id_from_url(p.get('url', '')) or '').lower()))

        except Exception as e:
            gr.Warning(f"Could not process the community apps and plugins list: {e}")
            return "<p style='text-align:center; color: var(--color-accent-soft);'>Failed to load community apps and plugins.</p>"

        if not community_plugins:
            return f"<p style='text-align:center; color: var(--text-color-secondary);'>{self._empty_community_message(plugin_type)}</p>"

        items_html = ""
        for plugin in community_plugins:
            name = plugin.get('name')
            author = plugin.get('author') or "Unknown"
            version = plugin.get('version', 'N/A')
            description = plugin.get('description') or "No description provided."
            type_badges_html = self._build_type_badges_html(plugin)
            url = plugin.get('url')

            if not url:
                continue
            if not name:
                name = plugin_id_from_url(url) or "Unknown Plugin"
            
            safe_url = url.replace("'", "\\'")
            required_version, incompatible, incompat_html = self._wan2gp_requirement_html(plugin)
            actions_container_html = self._requirement_actions_html(required_version) if incompatible else f"""
                <div class="plugin-item-actions">
                    <div class="plugin-action-row">
                        <button class="plugin-action-btn" onclick="handleStoreInstall(this, '{safe_url}')">Download</button>
                    </div>
                </div>
            """

            items_html += f"""
            <div class="plugin-item{' plugin-important' if plugin.get('important') else ''}">
                {discovery.card_header(dict(plugin, name=name), type_badges_html + incompat_html)}
                <div class="plugin-item-info">
                    <span class="version">version {version} by {author}</span>
                    <span class="description">{description}</span>
                </div>
                {actions_container_html}
            </div>
            """

        return f"<div class='plugin-list'>{items_html}</div>"

    def _build_local_available_plugins_html_outputs(self):
        return [self._build_local_available_plugins_html(plugin_type) for _, _, plugin_type in DISCOVER_PLUGIN_TYPE_TABS]

    def _empty_local_available_message(self, plugin_type):
        if plugin_type is None:
            return "All local apps and plugins are enabled."
        label = next(label for _, label, tab_type in DISCOVER_PLUGIN_TYPE_TABS if tab_type == plugin_type)
        return f"No disabled local {label.lower()} available."

    def _build_local_available_plugins_html(self, plugin_type=None):
        plugins_info = self.app.plugin_manager.get_plugins_info()
        enabled_user_plugins = set(self.server_config.get("enabled_plugins", []))
        installed_remote_plugins = set(self.server_config.get(INSTALLED_REMOTE_PLUGINS_KEY, []))
        local_plugins = [
            plugin for plugin in plugins_info
            if not plugin.get('system') and plugin.get('id') not in enabled_user_plugins and (plugin_type is None or plugin_type in normalize_plugin_types(plugin.get('type')))
        ]
        local_plugins.sort(key=lambda p: (not p.get('important'), (p.get('name') or p.get('id') or '').lower()))
        if not local_plugins:
            return f"<p style='text-align:center; color: var(--text-color-secondary);'>{self._empty_local_available_message(plugin_type)}</p>"

        items_html = ""
        for plugin in local_plugins:
            plugin_id = plugin['id']
            author = plugin.get('author') or "Unknown"
            type_badges_html = self._build_type_badges_html(plugin)
            is_bundled = not plugin.get('uninstallable', True)
            bundled_badge_html = '<span class="plugin-bundled-badge" title="Included with WanGP, cannot be uninstalled">Included</span>' if is_bundled else ''
            required_version, incompatible, incompat_html = self._wan2gp_requirement_html(plugin)
            action = "enable_local" if is_bundled or plugin_id in installed_remote_plugins else "install_local"
            label = "Enable" if action == "enable_local" else "Install"
            update_button_html = ""
            if plugin.get('uninstallable', True) and os.path.isdir(os.path.join(plugin.get('path', ''), '.git')):
                update_button_html = '<button class="plugin-action-btn" onclick="handlePluginAction(this, \'update\')">Update</button>'
            actions_container_html = self._requirement_actions_html(required_version) if incompatible else f"""
                <div class="plugin-item-actions">
                    <div class="plugin-action-row">
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, '{action}')">{label}</button>
                        {update_button_html}
                    </div>
                </div>
            """
            item_class = ("plugin-item incompatible" if incompatible else "plugin-item") + (" plugin-important" if plugin.get('important') else "")
            items_html += f"""
            <div class="{item_class}" data-plugin-id="{plugin_id}">
                {discovery.card_header(plugin, bundled_badge_html + type_badges_html + incompat_html)}
                <div class="plugin-item-info">
                    <span class="version">version {plugin['version']} by {author}</span>
                    <span class="description">{plugin.get('description', 'No description provided.')}</span>
                </div>
                {actions_container_html}
            </div>
            """
        return f"<div class='plugin-list'>{items_html}</div>"

    def _build_plugins_html(self):
        plugins_info = self.app.plugin_manager.get_plugins_info()
        enabled_user_plugins = self.server_config.get("enabled_plugins", [])
        all_user_plugins_info = [p for p in plugins_info if not p.get('system')]
        remote_plugins_info = self.app.plugin_manager.get_merged_catalog_entries(use_remote=False)
        
        css = """
        <style>
            .plugin-list { display: flex; flex-direction: column; gap: 12px; }
            .plugin-item { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 12px; align-items: center; padding: 16px; border: 1px solid var(--border-color-primary); border-radius: 12px; background-color: var(--background-fill-secondary); transition: box-shadow 0.2s ease-in-out; }
            .plugin-item.plugin-important { border-color: color-mix(in srgb, var(--color-accent) 45%, var(--border-color-primary)); background: color-mix(in srgb, var(--color-accent) 6%, var(--background-fill-secondary)); border-left: 3px solid var(--color-accent); }
            .plugin-item:hover { box-shadow: var(--shadow-drop-lg); }
            .plugin-item[draggable="true"] { cursor: grab; }
            .plugin-item[draggable="true"]:active { cursor: grabbing; }
            .plugin-item-info { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
            .plugin-header { grid-column: 1 / -1; display: flex; align-items: flex-start; gap: 8px; }
            .plugin-title-line { display: flex; flex: 1; flex-wrap: wrap; align-items: center; gap: 6px; min-width: 0; }
            .plugin-header .name { font-weight: 600; font-size: 1.1em; color: var(--text-color-primary); font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif; }
            .plugin-identity { display: inline-flex; align-items: center; gap: 4px; max-width: 100%; font-size: 0.85em; color: var(--text-color-secondary); }
            .plugin-identity > span { overflow-wrap: anywhere; min-width: 0; }
            .plugin-info-icon { display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; width: 20px; height: 20px; min-width: 0; padding: 1px !important; margin: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; color: var(--text-color-secondary); cursor: pointer; }
            .plugin-info-icon svg { width: 17px; height: 17px; }
            .plugin-info-icon:hover { color: var(--color-accent); }
            .plugin-info-icon:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; border-radius: 50%; }
            .plugin-featured-badge { flex-shrink: 0; padding: 2px 7px; border: 1px solid color-mix(in srgb, #eab308 45%, transparent); border-radius: 4px; background: color-mix(in srgb, #eab308 14%, transparent); color: var(--body-text-color); font-size: 0.78em; font-weight: 600; white-space: nowrap; }
            .plugin-item-info .version { font-size: 0.9em; color: var(--text-color-secondary); }
            .plugin-item-info .description { font-size: 0.95em; color: var(--text-color-secondary); margin-top: 4px; }
            .plugin-type-badges { display: inline-flex; flex-wrap: wrap; gap: 6px; }
            .plugin-type-badge { display: inline-flex; align-items: center; width: max-content; border: 1px solid var(--border-color-primary); border-radius: 4px; padding: 1px 7px; font-size: 0.78em; font-weight: 600; color: var(--text-color-secondary); background-color: var(--background-fill-primary); line-height: 1.5; }
            .plugin-item-actions { display: flex; flex-direction: column; gap: 6px; flex-shrink: 0; align-items: stretch; }
            .plugin-action-row { display: flex; gap: 8px; justify-content: flex-end; }
            .plugin-action-btn { display: inline-flex; align-items: center; justify-content: center; min-width: 88px; margin: 0 !important; border: 1px solid var(--button-secondary-border-color, #ccc) !important; background: var(--button-secondary-background-fill, #f0f0f0) !important; color: var(--button-secondary-text-color) !important; padding: 4px 12px !important; border-radius: 4px !important; cursor: pointer; font-weight: 500; }
            .plugin-action-btn:hover { background: var(--button-secondary-background-fill-hover, #e0e0e0) !important; transform: translateY(-1px); box-shadow: var(--shadow-drop); }
            .plugin-requirements-not-met { display: inline-flex; align-items: center; justify-content: center; min-width: 132px; border: 1px solid rgba(185, 28, 28, 0.35); border-radius: 4px; padding: 4px 12px; color: var(--color-error, #b91c1c); background-color: rgba(185, 28, 28, 0.1); font-weight: 600; white-space: nowrap; }
            .save-buttons-container { justify-content: flex-start; margin-top: 20px !important; gap: 12px; }
            .stylish-save-btn { font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif !important; font-weight: 600 !important; font-size: 1.05em !important; padding: 10px 20px !important; white-space: nowrap; }
            .plugin-instruction { font-size: 0.95em; color: var(--text-color-secondary); margin-bottom: 6px; }
            .plugin-restart-warning {
                border: 1px solid rgba(217, 119, 6, 0.45);
                border-left: 4px solid rgb(217, 119, 6);
                border-radius: 6px;
                background-color: rgba(217, 119, 6, 0.12);
                color: var(--body-text-color);
                padding: 10px 12px;
                font-size: 0.95em;
                font-weight: 600;
            }
            .update-available-notice {
                font-size: 0.9em;
                font-weight: 600;
                color: var(--color-accent);
                background-color: var(--color-accent-soft);
                padding: 2px 8px;
                border-radius: 4px;
                margin-left: 8px;
                white-space: nowrap;
            }
            .plugin-bundled-badge {
                font-size: 0.85em;
                font-weight: 600;
                color: var(--text-color-secondary);
                background-color: var(--background-fill-primary);
                padding: 2px 8px;
                border-radius: 4px;
                margin-left: 8px;
                border: 1px solid var(--border-color-primary);
                white-space: nowrap;
            }
            .plugin-incompatible-badge {
                font-size: 0.85em;
                font-weight: 600;
                color: var(--color-error, #b91c1c);
                background-color: rgba(185, 28, 28, 0.1);
                padding: 2px 8px;
                border-radius: 4px;
                margin-left: 8px;
                border: 1px solid rgba(185, 28, 28, 0.35);
                white-space: nowrap;
            }
            .plugin-item.update-available {
                border-left: 4px solid var(--color-accent);
            }
            .plugin-item.incompatible {
                border-left: 4px solid var(--color-error, #b91c1c);
            }
        </style>
        """

        if not enabled_user_plugins:
            instruction_html = ""
            user_html = "<p style='text-align:center; color: var(--text-color-secondary);'>No apps or plugins enabled yet. Explore the available apps and plugins.</p>"
        else:
            instruction_html = "<div class='plugin-instruction'>Drag enabled apps and plugins to reorder their tabs. Order is saved automatically. Disabled items remain in Available Locally.</div>"
            user_plugins_map = {p['id']: p for p in all_user_plugins_info}
            user_plugins = []
            for plugin_id in enabled_user_plugins:
                if plugin_id in user_plugins_map:
                    user_plugins.append(user_plugins_map.pop(plugin_id))

            user_items_html = ""
            for plugin in user_plugins:
                plugin_id = plugin['id']
                uninstallable = plugin.get('uninstallable', True)
                author = plugin.get('author') or "Unknown"
                type_badges_html = self._build_type_badges_html(plugin)
                
                update_notice_html = ''
                item_classes = []
                if uninstallable and plugin_id in remote_plugins_info:
                    remote_entry = remote_plugins_info[plugin_id]
                    if compare_release_metadata(remote_entry, plugin) > 0:
                        remote_version = remote_entry.get('version') or remote_entry.get('date') or "unknown"
                        update_notice_html = (
                            f'<span class="update-available-notice">New version {remote_version} is available !</span>'
                        )
                        item_classes.append('update-available')

                bundled_badge_html = ''
                if not uninstallable:
                    bundled_badge_html = '<span class="plugin-bundled-badge" title="Included with WanGP, cannot be uninstalled">Included</span>'

                required_version, incompatible, incompat_html = self._wan2gp_requirement_html(plugin)
                if incompatible:
                    item_classes.append('incompatible')
                    actions_container_html = self._requirement_actions_html(required_version)
                else:
                    actions_html = """
                    <div class="plugin-action-row">
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'disable_local')">Disable</button>
                    """
                    if uninstallable:
                        actions_html += """
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'update')">Update</button>
                    </div>
                    <div class="plugin-action-row">
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'reinstall')">Reinstall</button>
                        <button class="plugin-action-btn" onclick="handlePluginAction(this, 'uninstall')">Uninstall</button>
                        """
                    actions_html += "</div>"
                    actions_container_html = f'<div class="plugin-item-actions">{actions_html}</div>'
                
                user_items_html += f"""
                <div class="plugin-item {' '.join(item_classes)}" data-plugin-id="{plugin_id}" draggable="true">
                    {discovery.card_header(plugin, bundled_badge_html + type_badges_html + update_notice_html + incompat_html)}
                    <div class="plugin-item-info">
                        <span class="version">version {plugin['version']} by {author}</span>
                        <span class="description">{plugin.get('description', 'No description provided.')}</span>
                    </div>
                    {actions_container_html}
                </div>
                """
            user_html = f'<div id="user-plugin-list">{user_items_html}</div>'

        restart_warning_html = "<div class='plugin-restart-warning'>Restart required: app and plugin changes were saved. Restart WanGP for them to take effect.</div>" if self.restart_required else ""
        return f"{css}<div class='plugin-list'>{instruction_html}{user_html}{restart_warning_html}</div>"

    def _default_available_tab(self):
        enabled_user_plugins = set(self.server_config.get("enabled_plugins", []))
        plugins_info = self.app.plugin_manager.get_plugins_info()
        return "available_locally" if any(not plugin.get('system') and plugin.get('id') not in enabled_user_plugins for plugin in plugins_info) else "downloadable"

    def create_plugin_manager_ui(self):
        self._startup_enabled = frozenset(self.server_config.get("enabled_plugins", []))
        self.navigation_html_display = self.app.app_bar.component
        with gr.Blocks() as plugin_blocks:
            with gr.Row(equal_height=False, variant='panel'):
                with gr.Column(scale=2, min_width=600):
                    gr.Markdown("### Enabled Apps & Plugins")
                    self.plugins_html_display = gr.HTML()
                    with gr.Row(elem_classes="save-buttons-container"):
                        self.restart_button = gr.Button("Restart", variant="primary", size="sm", scale=0, elem_classes="stylish-save-btn")
                        self.refresh_catalog_button = gr.Button("Check for Updates", variant="secondary", size="sm", scale=0, elem_classes="stylish-save-btn")
                with gr.Column(scale=2, min_width=300):
                    gr.Markdown("### Available Apps & Plugins")
                    self.local_available_plugins_html_outputs = []
                    self.community_plugins_html_outputs = []
                    with gr.Tabs(selected=self._default_available_tab()):
                        with gr.Tab("Available Locally", id="available_locally"):
                            with gr.Tabs(selected="local_all"):
                                for tab_id, label, _ in DISCOVER_PLUGIN_TYPE_TABS:
                                    with gr.Tab(label, id=f"local_{tab_id}"):
                                        self.local_available_plugins_html_outputs.append(gr.HTML())
                        with gr.Tab("Downloadable", id="downloadable"):
                            with gr.Tabs(selected="downloadable_all"):
                                for tab_id, label, _ in DISCOVER_PLUGIN_TYPE_TABS:
                                    with gr.Tab(label, id=f"downloadable_{tab_id}"):
                                        self.community_plugins_html_outputs.append(gr.HTML())

                    with gr.Accordion("Install from URL", open=True):
                        with gr.Group():
                            self.plugin_url_textbox = gr.Textbox(label="GitHub URL", placeholder="https://github.com/user/wan2gp-plugin-repo")
                            self.install_plugin_button = gr.Button("Download and Install from URL")

            self.operation_progress = WangpProgress.component()
            with gr.Column(visible=False):
                self.plugin_action_input = gr.Textbox(elem_id="plugin_action_input")
                self.save_action_input = gr.Textbox(elem_id="save_action_input")
                self.catalog_refresh_input = gr.Textbox(elem_id="app_catalog_refresh")

        js = self._get_js_script_html()
        plugin_blocks.load(fn=None, js=js)

        self.on_tab_outputs = [self.plugins_html_display, *self.local_available_plugins_html_outputs, *self.community_plugins_html_outputs, self.navigation_html_display]
        self.catalog_refresh_input.change(fn=lambda _: self.refresh_html(), inputs=[self.catalog_refresh_input], outputs=self.on_tab_outputs, show_progress="hidden").then(fn=None, js="() => window.WanGPAppBar?.refresh()")
        catalog_changed_js = "() => window.WanGPAppBar?.changed()"
        
        self.restart_button.click(fn=None, js="handleRestart()")
        WangpProgress.bind(self.refresh_catalog_button.click, self._refresh_catalog,
            inputs=[],
            outputs=self.on_tab_outputs,
            component=self.operation_progress,
        )

        # Saving an order changes nothing on this page: the dragged list and the app bar already show it.
        self.save_action_input.change(fn=self._handle_save_action, inputs=[self.save_action_input], show_progress="hidden").then(fn=None, js=catalog_changed_js)
        
        WangpProgress.bind(self.plugin_action_input.change, self._handle_plugin_action_from_json,
            inputs=[self.plugin_action_input],
            outputs=self.on_tab_outputs,
            component=self.operation_progress,
        ).then(fn=None, js=catalog_changed_js)

        WangpProgress.bind(self.install_plugin_button.click, self._install_plugin_and_refresh,
            inputs=[self.plugin_url_textbox],
            outputs=[*self.on_tab_outputs, self.plugin_url_textbox],
            component=self.operation_progress,
        ).then(fn=None, js=catalog_changed_js)

        return plugin_blocks

    def on_tab_select(self, state: dict):
        if hasattr(self, '_community_plugins_cache'):
            del self._community_plugins_cache

        installed_html = self._build_plugins_html()
        available_outputs = [gr.update(value=html) for html in self._build_available_plugins_html_outputs()]
        return gr.update(value=installed_html), *available_outputs

    def _refresh_catalog(self, progress=WangpProgress()):
        self.app.plugin_manager.refresh_catalog(installed_only=True, use_remote=False)
        if hasattr(self, '_community_plugins_cache'):
            del self._community_plugins_cache
        updates_available = self._count_available_updates()
        if updates_available <= 0:
            gr.Info("All apps and plugins are up to date")
        elif updates_available == 1:
            gr.Info("One app or plugin update is available")
        else:
            gr.Info(f"{updates_available} app and plugin updates are available")
        return self._build_plugins_html(), *self._build_available_plugins_html_outputs()

    def _count_available_updates(self) -> int:
        try:
            plugins_info = self.app.plugin_manager.get_plugins_info()
            remote_plugins_info = self.app.plugin_manager.get_merged_catalog_entries(use_remote=False)
            count = 0
            for plugin in plugins_info:
                if plugin.get('system'):
                    continue
                if not plugin.get('uninstallable', True):
                    continue
                plugin_id = plugin.get('id')
                if not plugin_id or plugin_id not in remote_plugins_info:
                    continue
                remote_entry = remote_plugins_info[plugin_id]
                if compare_release_metadata(remote_entry, plugin) > 0:
                    count += 1
            return count
        except Exception:
            return 0

    def _write_server_config(self):
        write_config(self.server_config, self.server_config_filename)

    def _enable_plugin_id(self, plugin_id: str):
        with config_lock:
            enabled_plugins = self.server_config.get("enabled_plugins", [])
            if plugin_id not in enabled_plugins:
                enabled_plugins.append(plugin_id)
                self.server_config["enabled_plugins"] = enabled_plugins
                if plugin_id in FEATURED_APPS:
                    self.server_config.setdefault(REVIEW_KEY, {})[plugin_id] = True
                self._write_server_config()
                return True
            return False

    def _enable_plugin_after_install(self, url: str):
        plugin_id = plugin_id_from_url(url)
        try:
            return self._enable_plugin_id(plugin_id)
        except Exception as e:
            gr.Warning(f"Failed to auto-enable plugin {plugin_id}: {e}")
        return False

    def _finish_install_from_url(self, url: str, result_message: str):
        plugin_id = plugin_id_from_url(url)
        if "[Success]" in result_message:
            self.restart_required = True
            was_enabled = self._enable_plugin_after_install(url)
            if was_enabled:
                result_message = result_message.replace("Please enable it in the list and restart WanGP.", "It has been auto-enabled. Please restart WanGP.")
            elif plugin_id in self.server_config.get("enabled_plugins", []):
                result_message = result_message.replace("Please enable it in the list and restart WanGP.", "It is already enabled. Please restart WanGP.")
            if plugin_id:
                self.app.plugin_manager.record_plugin_metadata(plugin_id, url=url)
            return result_message
        return result_message

    def _save_and_restart(self, enabled_plugins: list):
        update_config(self.server_config, self.server_config_filename, {"enabled_plugins": enabled_plugins})
        gr.Info("Settings saved. Restarting application...")
        if callable(getattr(self, "restart_application", None)):
            self.restart_application()
            return
        elif callable(getattr(self, "quit_application", None)):            
            gr.Warning("Restart hook is unavailable. WAN2GP will now quit. Please start WAN2GP again manually.")
            self.quit_application()
            return

    def _handle_save_action(self, payload_str: str):
        if not payload_str:
            return
        payload = json.loads(payload_str)
        enabled_plugins = self.server_config.get("enabled_plugins", []) if payload["enabled_plugins"] is None else payload["enabled_plugins"]
        if payload["restart"]:
            self._save_and_restart(enabled_plugins)
        else:  # a new order needs no restart: the app bar follows it
            update_config(self.server_config, self.server_config_filename, {"enabled_plugins": enabled_plugins})
            self.app.app_bar.refresh_default()

    def _install_plugin_and_refresh(self, url, progress=WangpProgress()):
        progress(0, desc="Starting installation...")
        result_message = self.app.plugin_manager.install_plugin_from_url(url, progress=progress)
        result_message = self._finish_install_from_url(url, result_message)
        if "[Success]" in result_message:
            if "[Warning]" in result_message or "[CRITICAL" in result_message:
                gr.Warning(result_message)
            else:
                gr.Info(result_message)
        else:
            gr.Warning(result_message)
        if hasattr(self, '_community_plugins_cache'):
            del self._community_plugins_cache
        return self._build_plugins_html(), *self._build_available_plugins_html_outputs(), ""

    def _handle_plugin_action_from_json(self, payload_str: str, progress=WangpProgress()):
        if not payload_str:
            return (gr.update(), *[gr.update() for _ in range(self._available_outputs_count())])
        try:
            payload = json.loads(payload_str)
            action = payload.get("action")
            plugin_id = payload.get("plugin_id")
            
            if action == 'install_from_store':
                url = payload.get("url")
                if not url:
                    raise ValueError("URL is required for install_from_store action.")
                result_message = self.app.plugin_manager.install_plugin_from_url(url, progress=progress)
                result_message = self._finish_install_from_url(url, result_message)
            else:
                if not action or not plugin_id:
                     raise ValueError("Action and plugin_id are required.")
                result_message = ""
                if action == 'uninstall':
                    result_message = self.app.plugin_manager.uninstall_plugin(plugin_id)
                    with config_lock:
                        current_enabled = self.server_config.get("enabled_plugins", [])
                        if plugin_id in current_enabled:
                            current_enabled.remove(plugin_id)
                            self.server_config["enabled_plugins"] = current_enabled
                            self._write_server_config()
                elif action == 'update':
                    result_message = self.app.plugin_manager.update_plugin(plugin_id, progress=progress)
                elif action == 'reinstall':
                    result_message = self.app.plugin_manager.reinstall_plugin(plugin_id, progress=progress)
                elif action == 'enable_local':
                    self._enable_plugin_id(plugin_id)
                    ready = plugin_id in FEATURED_APPS and plugin_id in self.app.plugin_manager.get_all_plugins()
                    result_message = f"[Success] '{plugin_id}' enabled. " + ("Ready to open." if ready else "Please restart WanGP for changes to take effect.")
                elif action == 'install_local':
                    result_message = self.app.plugin_manager.install_local_plugin(plugin_id, progress=progress)
                    if "[Success]" in result_message:
                        self._enable_plugin_id(plugin_id)
                        result_message = result_message.replace("Please enable it in the list and restart WanGP.", "It has been enabled. Please restart WanGP.")
                elif action == 'disable_local':
                    with config_lock:
                        enabled_plugins = self.server_config.get("enabled_plugins", [])
                        if plugin_id in enabled_plugins:
                            enabled_plugins.remove(plugin_id)
                            self.server_config["enabled_plugins"] = enabled_plugins
                            self._write_server_config()
                    result_message = f"[Success] '{plugin_id}' disabled. Please restart WanGP for changes to take effect."
            
            if "[Success]" in result_message:
                if action in ("install_from_store", "uninstall", "update", "reinstall", "enable_local", "install_local", "disable_local") and not (action == "enable_local" and plugin_id in FEATURED_APPS and plugin_id in self.app.plugin_manager.get_all_plugins()):
                    self.restart_required = True
                if "[Warning]" in result_message or "[CRITICAL" in result_message:
                    gr.Warning(result_message)
                else:
                    gr.Info(result_message)
            elif "[Error]" in result_message or "[Warning]" in result_message or "[CRITICAL" in result_message:
                gr.Warning(result_message)
            else:
                gr.Info(result_message)
        except (json.JSONDecodeError, ValueError) as e:
            gr.Warning(f"Could not perform app or plugin action: {e}")
            traceback.print_exc()
        except Exception as e:
            gr.Warning(f"App or plugin action failed: {e}")
            traceback.print_exc()

        if hasattr(self, '_community_plugins_cache'):
            del self._community_plugins_cache

        return self._build_plugins_html(), *self._build_available_plugins_html_outputs()
