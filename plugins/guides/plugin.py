import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

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
