# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""App navigation over the native Gradio tabs, retaining their event contracts."""
import json
from html import escape
from pathlib import Path
from threading import Lock

import gradio as gr
from shared.gradio.app_icons import icon
from shared.utils.bundled_apps import FEATURED_APPS, VISITED_KEY, review_featured_app, visit_featured_app
from shared.utils.plugins import plugin_navigation

ASSETS = Path(__file__).parent


class AppBar:
    def __init__(self, app, tabs, state):
        self.app, self.tabs, self.state = app, tabs, state
        self.config = app.plugin_manager.server_config
        self.filename = app.plugin_manager.server_config_filename
        self.entries = []
        self._default_lock = Lock()
        tabs.elem_classes = [*(tabs.elem_classes or []), "wangp-native-app-tabs"]
        self.component = gr.HTML(elem_id="wangp-app-bar")
        self.component.parent.children.remove(self.component)
        self.component.parent = tabs.parent
        tabs.parent.children.insert(tabs.parent.children.index(tabs), self.component)
        with gr.Column(visible=False) as controls:
            self.route = gr.Textbox(elem_id="wangp_app_route")
            self.review = gr.Textbox(elem_id="wangp_app_review")
            self.refresh = gr.Textbox(elem_id="wangp_app_refresh")
            self.visited = gr.Textbox(value="[]")
        controls.parent.children.remove(controls)
        controls.parent = tabs.parent
        tabs.parent.children.append(controls)

    @staticmethod
    def stylesheet():
        return f'<style>{ASSETS.joinpath("app_bar.css").read_text(encoding="utf-8")}</style>'

    def bind(self):
        plugin_tabs = {f"plugin_{tab.id}": (pid, tab) for pid, plugin in self.app.plugin_manager.plugins.items() for tab in plugin.tabs.values()}
        for tab in self.tabs.children:
            if not isinstance(tab, gr.Tab):
                continue
            # An explicit id also makes tabs declared directly by extensions routable.
            if tab.id is None:
                tab.id = f"app_tab_{tab._id}"
            pid, declaration = plugin_tabs.get(tab.id, (str(tab.id), None))
            navigation = plugin_navigation(pid, self.app.plugin_manager.plugins[pid]) if declaration else None
            if navigation and navigation.tab_id != tab.id:
                navigation = None
            self.entries.append(dict(id=str(tab.id), label=tab.label, plugin=pid, icon=declaration.icon if declaration else None, navigation=navigation, panel=tab.elem_id or "", visible=tab.visible))
            tab.select(fn=None, js=f"() => window.WanGPAppBar?.selected({json.dumps(str(tab.id))})")
            if pid in FEATURED_APPS:
                tab.select(fn=self._visit, inputs=[gr.State(pid)], outputs=[self.visited], queue=False, show_progress="hidden")
        self.root = gr.context.get_blocks_context().root_block
        self.component.value = self.render()
        self.root.head += f'<script>{ASSETS.joinpath("app_bar.js").read_text(encoding="utf-8")}</script>'
        self.root.load(fn=None, js="() => {" + ASSETS.joinpath("app_navigation.js").read_text(encoding="utf-8") + "window.WanGPAppBar.refresh(); }")
        self.route.change(fn=None, inputs=[self.route], js="(value) => window.WanGPAppBar.commitRoute(value)")
        self.visited.change(fn=None, inputs=[self.visited], js="(value) => window.WanGPAppBar.visited(value)")
        manager = self.app.plugin_manager.plugins["plugin_manager"]
        self.review.change(fn=self._review, inputs=[self.review], outputs=manager.on_tab_outputs, show_progress="hidden").then(fn=None, inputs=[self.review], js="(value) => window.WanGPAppBar.reviewed(value)")
        self.refresh.change(fn=lambda _: manager.refresh_html(), inputs=[self.refresh], outputs=manager.on_tab_outputs, queue=False, show_progress="hidden").then(fn=None, js="() => window.WanGPAppBar?.refresh()")

    def _review(self, value):
        action = json.loads(value)
        if action["action"] not in ("keep", "remove"):
            raise ValueError("Unknown app review action")
        review_featured_app(self.config, self.filename, action["plugin"], action["action"] == "keep")
        manager = self.app.plugin_manager.plugins["plugin_manager"]
        return manager.refresh_html()

    def _visit(self, plugin_id):
        if visit_featured_app(self.config, self.filename, plugin_id):
            self.refresh_default()
            return json.dumps(self.config[VISITED_KEY])
        return gr.skip()

    def refresh_default(self):
        # Gradio serves new pages from its serialized Blocks config. A session output
        # alone would leave new pages showing the startup order and badges.
        with self._default_lock:
            html = self.render()
            self.component.value = html
            component = next(c for c in self.root.config["components"] if c["id"] == self.component._id)
            component["props"]["value"] = html
            return html

    def render(self):
        order = {pid: index for index, pid in enumerate(self.config.get("enabled_plugins", []))}
        enabled = set(order)
        visited = set(self.config.get(VISITED_KEY, []))
        apps, utilities = [], []
        # Enabled Plugins take the places of their tabs in their saved order, which may have changed since the tabs were created.
        reordered = iter(sorted((entry for entry in self.entries if entry["plugin"] in order), key=lambda entry: order[entry["plugin"]]))
        for entry in [next(reordered) if entry["plugin"] in order else entry for entry in self.entries]:
            pid, target, label = entry["plugin"], entry["id"], entry["label"]
            if pid in ("models_manager", "about") or (pid in FEATURED_APPS and pid not in enabled):
                continue
            utility = pid in ("configuration", "plugin_manager", "guides")
            label = {"plugin_manager": "Apps & PlugIn Manager", "guides": "Infos"}.get(pid, label)
            tooltip = {"configuration": "Configuration — application settings", "plugin_manager": "Apps & PlugIn Manager", "guides": "Infos — guides and About"}.get(pid, label)
            shortcut = {"configuration": "C", "plugin_manager": "P", "guides": "I"}.get(pid)
            hint = f' data-tooltip="{escape(tooltip, quote=True)} [Alt+{shortcut}]" aria-keyshortcuts="Alt+{shortcut}"' if shortcut else f' title="{escape(tooltip, quote=True)}"'
            svg = entry["icon"] or icon("apps" if pid == "plugin_manager" else pid)
            new = pid in FEATURED_APPS and pid not in visited
            intro_pending = new
            badge = f'<span class="app-bar-new{ "" if new else " is-seen"}" aria-hidden="{str(not new).lower()}">New!</span>' if pid in FEATURED_APPS else ''
            navigation = entry['navigation']
            navigation_data = f' data-app-url="{escape(navigation.path, quote=True)}" data-app-fullscreen="{str(navigation.fullscreen).lower()}" data-app-panel="{escape(entry["panel"], quote=True)}"' if navigation else ''
            initial_state = "" if entry["visible"] else " hidden"
            if target == self.tabs.selected:
                initial_state += ' aria-current="page"'
            button = f'<button type="button" class="app-bar-link{" app-bar-utility" if utility else ""}" data-app-route="{escape(target, quote=True)}" data-app-plugin="{escape(pid, quote=True)}" data-app-intro-pending="{str(intro_pending).lower()}"{navigation_data}{initial_state} aria-label="{escape(label, quote=True)}"{hint}>{svg}<span class="app-bar-label" data-label="{escape(label, quote=True)}"><span>{escape(label)}</span></span>{badge}</button>'
            (utilities if utility else apps).append(button)
        manager = self.app.plugin_manager.plugins["plugin_manager"]
        intros = manager.information_cards_html()
        removed = [entry['id'] for entry in self.entries if entry['plugin'] in FEATURED_APPS and entry['plugin'] not in enabled]
        return self.stylesheet() + f'''<nav class="app-bar" aria-label="WanGP Apps" data-removed-apps="{escape(json.dumps(removed), quote=True)}">
            <div class="app-bar-viewport"><wangp-app-links class="app-bar-apps">{''.join(apps)}</wangp-app-links></div>
            <div class="app-bar-overflow" hidden><button type="button" class="app-bar-more" aria-label="More Apps" title="More Apps" aria-expanded="false">{icon('more')}</button><div class="app-bar-menu" hidden>{''.join(apps)}</div></div>
            <div class="app-bar-utilities">{''.join(utilities)}</div></nav>
            <div class="wangp-addon-route-status" role="status" hidden></div>
            <div class="app-welcome-backdrop" hidden><section class="app-welcome" role="dialog" aria-modal="true" aria-label="Plugin Details"><button type="button" class="app-welcome-close" aria-label="Close Introduction" title="Close">&times;</button>{intros}</section></div>'''
