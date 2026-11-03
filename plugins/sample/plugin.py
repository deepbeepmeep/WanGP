# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from datetime import datetime
import html
import time
import json
import gradio as gr
from shared.gradio.progress import WangpProgress
from shared.gradio.strip_gallery import gradio_file_url

from shared.utils.plugins import WAN2GPPlugin
from shared.utils.process_locks import acquire_GPU_ressources, any_GPU_process_running, release_GPU_ressources

PlugIn_Name = "Sample App"
PlugIn_Id = "SamplePlugin"
Deepy_Request = "add a red hat and then generate a 5s video of the person singing"
# The button keeps its label on one line; the box beside it holds the video link once Deepy has made it.
Deepy_Style = "<style>#sample-deepy-ask{white-space:nowrap;flex:0 0 auto !important;width:auto !important;min-width:max-content !important}.sample-deepy-link{display:flex;align-items:center;min-height:40px;padding:0 12px;border:1px dashed var(--neutral-400);border-radius:var(--radius-lg)}.sample-deepy-link span{opacity:.6}</style>"
Deepy_Link_Placeholder = Deepy_Style + '<div class="sample-deepy-link"><span>The video Deepy makes will be downloadable here</span></div>'


def acquire_GPU(state):
    GPU_process_running = any_GPU_process_running(state, PlugIn_Id)
    if GPU_process_running:
        gr.Error("Another Plugin is using the GPU")
    acquire_GPU_ressources(state, PlugIn_Id, PlugIn_Name, gr=gr)


def release_GPU(state):
    release_GPU_ressources(state, PlugIn_Id)


class ConfigTabPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_global("get_current_model_settings")
        self.request_global("refresh_model_defs")        
        self.request_component("refresh_form_trigger")
        self.request_component("state")
        self.request_component("resolution")
        self.request_component("main_tabs")
        self.request_component("model_choice_target")
        self.request_global("switch_to_model")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    @staticmethod
    def _demo_settings(_state) -> dict[str, object]:
        return {
            "model_type": "ltx2_22B_distilled",
            "prompt": "A reframed cinematic 10 second video shot: a woman walks slowly through a rainy neon street at night, camera gently orbiting, reflections shimmering on the wet pavement, natural motion, realistic lighting.",
            "resolution": "1280x720",
            "num_inference_steps": 8,
            "video_length": 241,
            "sliding_window_size": 481,
            "sliding_window_overlap": 17,
        }

    def on_tab_select(self, state: dict) -> None:
        settings = self.get_current_model_settings(state)
        prompt = settings["prompt"]
        return prompt

    def on_tab_deselect(self, state: dict) -> None:
        pass

    def on_model_change(self, state: dict, model_type) -> None:
        pass

    def ask_deepy(self, progress=WangpProgress()):
        """Send Deepy the selected picture with a request; return a download link to the video it makes."""
        unavailable = self.deepy.unavailable_reason
        picture = self.deepy.selected()
        if unavailable or picture is None or picture["kind"] != "image":
            gr.Info(unavailable or "Select a picture of a person in the gallery first.")
            return gr.update()
        ratio = 0.0
        # In the current workspace, the hat edit and the video join the user's gallery.
        with self.deepy.run(Deepy_Request, [picture["id"]], workspace="current") as job:
            for event in job.events():
                if event.kind == "queued":
                    progress(0, desc="Waiting for Deepy")
                elif event.kind == "status":
                    progress(ratio, desc=event.data["text"])
                elif event.kind == "tool":
                    progress(ratio, desc=f"{event.data['label']}: {event.data['status_text']}")
                elif event.kind == "progress":
                    ratio = event.data.progress / 100
                    progress(ratio, desc=event.data.status)
            result = job.result()
        if result.status == "error":
            raise gr.Error(result.error)
        video = next((output for output in reversed(result.outputs) if output["kind"] == "video"), None)
        if video is None:
            gr.Info("Deepy was stopped." if result.cancelled else "Deepy finished without making a video.")
            return gr.update()
        name = html.escape(video["name"])
        return Deepy_Style + f'<div class="sample-deepy-link"><a href="{gradio_file_url(video["path"])}" download="{name}">Download {name}</a></div>'

    def create_config_ui(self, api_session):
        active_job = {"job": None}

        def update_prompt(state, text):
            settings = self.get_current_model_settings(state)
            settings["prompt"] = text
            return time.time()

        def big_process(state):
            acquire_GPU(state)
            gr.Info("Doing something important that Requires Full VRAM & GPU available")
            time.sleep(30)
            release_GPU(state)
            return "42"

        def generate_media(state_value, progress=WangpProgress()):
            class DemoCallbacks:
                ratio = 0.0

                def on_status(self, status):
                    status = str(status or "").strip()
                    if status:
                        progress(self.ratio, desc=status)

                def on_progress(self, update):
                    self.ratio = max(0.0, min(1.0, float(getattr(update, "progress", 0)) / 100.0))
                    progress(self.ratio, desc=str(getattr(update, "status", "") or "Generating..."))

            job = api_session.submit_task(self._demo_settings(state_value), callbacks=DemoCallbacks())
            active_job["job"] = job
            try:
                result = job.result()
            finally:
                if active_job.get("job") is job:
                    active_job["job"] = None
            if result.success and result.generated_files:
                return result.generated_files[0]
            if result.cancelled:
                return gr.update()
            errors = list(result.errors or [])
            raise gr.Error(str(errors[0] if errors else "WanGP completed without returning an output file."))

        def cancel_demo():
            job = active_job.get("job")
            if job is not None and not job.done:
                job.cancel()

        def write_finetune():
            the_time= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            finetune_def =  {
                "model": {
                        "name": f"Finetune of {the_time}",
                        "architecture": "ltx2_22B",
                        "description": "enjoy this finetune of the moment",
                "URLs": "ltx2_22B",
                "preload_URLs":"ltx2_22B",        
                "ltx2_pipeline": "distilled"
                },
                "prompt": f"A video that is all about {the_time}",
            }
            with open(f"finetunes/to_be_deleted{the_time}.json", "w", encoding="utf-8") as writer:
                writer.write(json.dumps(finetune_def, indent=4))
            self.refresh_model_defs()
            gr.Info(f"finetune {the_time} had been created")

        with gr.Column():
            state = self.state
            settings = self.get_current_model_settings(state.value)
            prompt = settings["prompt"]
            gr.HTML("<B><B>Sample App that illustrates</B>:<BR>-How to get Settings from Main Form and then Modify them<BR>-How to suspend the Video Gen (and release VRAM) to execute your own GPU intensive process.<BR>-How to switch back automatically to the Main Tab<BR>-How to trigger a Video Gen from a Plugin and track its progress<BR>-How to ask Deepy to work on the selected picture and get its video back<BR>-Add a new finetune on the fly")
            sample_text = gr.Text(label="Prompt Copy", value=prompt, lines=5)
            update_btn = gr.Button("Update Prompt On Main Page")
            gr.Markdown()
            process_btn = gr.Button("Use GPU To Do Something Important")
            process_output = gr.Text(label="Process Output", value="")
            goto_btn = gr.Button("Goto Media Generator Tab")
            gr.Markdown("---")
            start_btn = gr.Button("Generate a LTX 2.3 Video")
            output_video = gr.Video(label="Output")
            demo_progress = WangpProgress.component()
            abort_btn = gr.Button("Abort Generation")
            gr.Markdown("---")
            with gr.Row(equal_height=True):
                deepy_btn = gr.Button("Ask Deepy to add a red hat and to make a singing video", elem_id="sample-deepy-ask", scale=0)
                deepy_link = gr.HTML(Deepy_Link_Placeholder, padding=False)
            deepy_progress = WangpProgress.component()

            write_finetune_btn = gr.Button("Generate a Random LTX 2 Finetune")
            with gr.Row():
                switch_wan21_t2v_btn = gr.Button("Switch to Wan2.1 T2V")
                switch_ltx2_distilled_btn = gr.Button("Switch to LTX-2 Distilled")
                open_media_tab = gr.Checkbox(label="Open media tab", value=False)

        self.on_tab_outputs = [sample_text]

        # This edits the current browser's generation draft, not shared config.
        update_btn.click(update_prompt, inputs=[state, sample_text], outputs=[self.refresh_form_trigger])
        process_btn.click(fn=big_process, inputs=[state], outputs=[process_output])
        goto_btn.click(fn=self.goto_media_tab, inputs=[state], outputs=[self.main_tabs])
        WangpProgress.bind(start_btn.click, generate_media, inputs=[state], outputs=[output_video], component=demo_progress)
        abort_btn.click(fn=cancel_demo, queue=False)
        WangpProgress.bind(deepy_btn.click, self.ask_deepy, inputs=[], outputs=[deepy_link], component=deepy_progress)
        write_finetune_btn.click(fn=write_finetune)
        switch_wan21_t2v_btn.click(fn=lambda open_tab: self.switch_to_model("t2v", open_tab), inputs=[open_media_tab], outputs=[self.model_choice_target, self.main_tabs], show_progress="hidden")
        switch_ltx2_distilled_btn.click(fn=lambda open_tab: self.switch_to_model("ltx2_22B_distilled", open_tab), inputs=[open_media_tab], outputs=[self.model_choice_target, self.main_tabs], show_progress="hidden")
