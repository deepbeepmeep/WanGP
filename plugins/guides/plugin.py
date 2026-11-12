import json
import os
from datetime import datetime, timedelta

import gradio as gr
from shared.utils.config_store import update_config
from shared.utils.plugins import WAN2GPPlugin

PROJECT_PAGE = "https://github.com/deepbeepmeep/WanGP"
STAR_REQUEST = "WanGP is free to use, at home or at work. If you like it, please give it a star on GitHub: it helps others find the project."
# wgp_config.json: {"new_user": true} in a config created at first launch, then {"github_opened": true} or
# {"ask_later": "<date>", "later_count": n}. WanGP Manager sets WANGP_GITHUB_OPENED=1 once its own star link was clicked.
WELCOME_KEY = "welcome_popup"
# Three requests at most: at launch, then a week and a month after each Ask Me Again Later.
ASK_AGAIN_AFTER = (timedelta(days=7), timedelta(days=30))
WELCOME_TEXTS = {
    "new": ("Welcome to WanGP!", "The Infos tab has guides to get you started: choosing a model, writing prompts, using LoRAs, Deepy and more."),
    "returning": ("Thank You for Using WanGP!", "We hope you have had great fun using WanGP and found it useful."),
}
# Escape or a click outside counts as the first button. GitHub opens from the click itself, so that no popup blocker stops it.
WELCOME_JS = """async (popup) => {
    if (!popup) return "";
    const {title, message, later} = JSON.parse(popup);
    return await window.wangpConfirm({title, message, buttons: [
        {text: later, value: "later", cancel: true},
        {text: "Go to GitHub", value: "github", primary: true, action: () => { window.open(%s, "_blank", "noopener"); }},
    ]});
}""" % json.dumps(PROJECT_PAGE)

class GuidesPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Guides Tab"
        self.version = "1.0.0"
        self.description = "Guides for using WAN2GP"
        self.goto_model_type = None

    def setup_ui(self):
        self.request_global("app")
        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("model_choice_target")
        self.request_global("goto_model_type")
        self.request_global("server_config")
        self.request_global("server_config_filename")
        self.add_custom_js(self._script_block())
        self.add_tab(
            tab_id="info",
            label="Infos",
            component_constructor=self.create_guides_ui,
        )

    def create_guides_ui(self):
        with open("docs/MODELS.md", "r", encoding="utf-8") as reader:
            models = reader.read()

        with open("docs/LORAS.md", "r", encoding="utf-8") as reader:
            loras = reader.read()

        with open("docs/PROMPTS.md", "r", encoding="utf-8") as reader:
            prompts = reader.read()

        with open("docs/PROCESSING.md", "r", encoding="utf-8") as reader:
            processing = reader.read()

        with open("docs/FINETUNES.md", "r", encoding="utf-8") as reader:
            finetunes = reader.read()

        with open("docs/PLUGINS.md", "r", encoding="utf-8") as reader:
            plugins = reader.read()

        with open("docs/OVERVIEW.md", "r", encoding="utf-8") as reader:
            overview = reader.read()

        with open("docs/DEEPY.md", "r", encoding="utf-8") as reader:
            deepy = reader.read()

        with open("docs/WORKSPACES.md", "r", encoding="utf-8") as reader:
            workspaces = reader.read()

        gr.Markdown(f"⭐ **WanGP is free to use, at home or at work.** If you like it, please give it a [star on GitHub]({PROJECT_PAGE}): it helps others find the project.")
        welcome_popup = gr.Textbox(visible=False)
        gr.context.get_blocks_context().root_block.load(self._welcome_popup, outputs=[welcome_popup], queue=False, show_progress="hidden").then(
            self._record_welcome_choice, inputs=[welcome_popup], js=WELCOME_JS, queue=False, show_progress="hidden")
        with gr.Tabs():
            with gr.Tab("Overview", id="overview"):
                gr.Markdown(overview, elem_id="guides_overview_markdown")
                gr.HTML("<style>.guides-hidden-controls{display:none !important;}</style>")
                with gr.Row(elem_classes="guides-hidden-controls"):
                    model_selection = gr.Text(value="", label="", elem_id="guides_overview_target")
                    apply_btn = gr.Button("Apply Selection", elem_id="guides_overview_trigger")
                apply_btn.click(
                    fn=self._apply_overview_selection,
                    inputs=[self.state, model_selection],
                    outputs=[self.model_choice_target],
                    show_progress="hidden"
                ).then(
                    fn=self.goto_media_tab,
                    inputs=[self.state],
                    outputs=[self.main_tabs]
                )
            with gr.Tab("Deepy", id="deepy"):
                gr.Markdown(deepy)
            with gr.Tab("Workspaces", id="workspaces"):
                gr.Markdown(workspaces)
            with gr.Tab("Prompts", id="prompts"):
                gr.Markdown(prompts)
            with gr.Tab("LoRAs", id="loras"):
                gr.Markdown(loras)
            with gr.Tab("Processing", id="processing"):
                gr.Markdown(processing)
            with gr.Tab("Finetunes", id="finetunes"):
                gr.Markdown(finetunes)
            with gr.Tab("Plugins", id="plugins"):
                gr.Markdown(plugins)
            with gr.Tab("About", id="about"):
                self.app.plugin_manager.plugins["about"].create_about_ui()

    def _apply_overview_selection(self, state, model_type):
        return model_type

    def _welcome_popup(self):
        """The welcome popup for this page load, empty when not due: never after Go to GitHub (here or in WanGP Manager), else three times at most."""
        record = self.server_config.get(WELCOME_KEY, {})
        if not record.get("github_opened") and os.environ.get("WANGP_GITHUB_OPENED") == "1":
            self._record_welcome_choice("github")
            return ""
        later = record.get("later_count", 0)
        if record.get("github_opened") or later > len(ASK_AGAIN_AFTER) or later and datetime.now() < datetime.fromisoformat(record["ask_later"]) + ASK_AGAIN_AFTER[later - 1]:
            return ""
        title, intro = WELCOME_TEXTS["new" if record.get("new_user") else "returning"]
        # The last request does not promise another one.
        return json.dumps({"title": title, "message": f"{intro}\n\n⭐ {STAR_REQUEST}", "later": "No Thanks" if later == len(ASK_AGAIN_AFTER) else "Ask Me Again Later"})

    def _record_welcome_choice(self, choice):
        if choice:
            later = self.server_config.get(WELCOME_KEY, {}).get("later_count", 0) + 1
            update_config(self.server_config, self.server_config_filename, {WELCOME_KEY: {"github_opened": True} if choice == "github" else {"ask_later": datetime.now().isoformat(timespec="seconds"), "later_count": later}})

    def _script_block(self) -> str:
        return """
    (function () {
        const RETRY_DELAY = 500;

        function root() {
            if (window.gradioApp) return window.gradioApp();
            const app = document.querySelector("gradio-app");
            return app ? (app.shadowRoot || app) : document;
        }

        function bindOverviewLinks() {
            const appRoot = root();
            if (!appRoot) return false;
            const markdown = appRoot.querySelector("#guides_overview_markdown");
            const targetInput = appRoot.querySelector("#guides_overview_target textarea, #guides_overview_target input");
            let triggerButton = appRoot.querySelector("#guides_overview_trigger");

            if (!markdown || !targetInput || !triggerButton) return false;

            if (!triggerButton.matches("button")) {
                const innerButton = triggerButton.querySelector("button");
                if (innerButton) triggerButton = innerButton;
            }

            if (markdown.dataset.guidesBound === "1") return true;
            markdown.dataset.guidesBound = "1";

            markdown.addEventListener("click", (event) => {
                const anchor = event.target.closest("a");
                if (!anchor) return;
                const hrefValue = anchor.getAttribute("href") || "";
                if (!hrefValue.startsWith("modeltype:")) return;
                event.preventDefault();
                const modelType = hrefValue.replace("modeltype:", "").trim();
                if (!modelType) return;
                targetInput.value = modelType;
                targetInput.dispatchEvent(new Event("input", { bubbles: true }));
                triggerButton.click();
            });

            return true;
        }

        function ensureBinding() {
            if (!bindOverviewLinks()) setTimeout(ensureBinding, RETRY_DELAY);
        }

        ensureBinding();
    })();
"""
