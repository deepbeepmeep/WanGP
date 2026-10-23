# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import base64
import io
import json
import subprocess
import time
from pathlib import Path

import ffmpeg
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from shared.utils.plugins import WAN2GPPlugin
from shared.utils.subprocess_stdin import child_kwargs


class MotionDesignerPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Motion Designer"
        self.version = "1.0.0"
        self.description = (
            "Cut objects, design their motion paths, preview the animation, and send the mask directly into WanGP."
        )
        self._iframe_html_cache: str | None = None
        self._iframe_cache_signature: tuple[int, int, int] | None = None

    def setup_ui(self):
        self.request_global("update_video_prompt_type")
        self.request_global("get_model_def")
        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("refresh_form_trigger")
        self.request_global("get_current_model_settings")
        self.add_custom_js(self._js_bridge())
        self.add_tab(
            tab_id="motion_designer",
            label="Motion Designer",
            component_constructor=self._build_ui,
        )

    def _build_ui(self):
        iframe_html = self._get_iframe_markup()
        iframe_wrapper_style = """
        <style>
            #motion_designer_iframe_container,
            #motion_designer_iframe_container > div {
                padding: 0 !important;
                margin: 0 !important;
            }
            #motion_designer_iframe_container iframe {
                display: block;
            }
        </style>
        """
        with gr.Column(elem_id="motion_designer_plugin"):
            gr.HTML(
                value=iframe_wrapper_style + iframe_html,
                elem_id="motion_designer_iframe_container",
                min_height=None,
            )
            mask_payload = gr.Textbox(
                label="Mask Payload",
                visible=False,
                elem_id="motion_designer_mask_payload",
            )
            metadata_payload = gr.Textbox(
                label="Mask Metadata",
                visible=False,
                elem_id="motion_designer_meta_payload",
            )
            background_payload = gr.Textbox(
                label="Background Payload",
                visible=False,
                elem_id="motion_designer_background_payload",
            )
            guide_payload = gr.Textbox(
                label="Guide Payload",
                visible=False,
                elem_id="motion_designer_guide_payload",
            )
            guide_metadata_payload = gr.Textbox(
                label="Guide Metadata",
                visible=False,
                elem_id="motion_designer_guide_meta_payload",
            )
            mode_sync = gr.Textbox(
                label="Mode Sync",
                value="cut_drag",
                visible=False,
                elem_id="motion_designer_mode_sync",
            )
            trajectory_payload = gr.Textbox(
                label="Trajectory Payload",
                visible=False,
                elem_id="motion_designer_trajectory_payload",
            )
            trajectory_metadata = gr.Textbox(
                label="Trajectory Metadata",
                visible=False,
                elem_id="motion_designer_trajectory_meta",
            )
            trajectory_background = gr.Textbox(
                label="Trajectory Background",
                visible=False,
                elem_id="motion_designer_trajectory_background",
            )
            trigger = gr.Button(
                "Apply Motion Designer data",
                visible=False,
                elem_id="motion_designer_apply_trigger",
            )
            trajectory_trigger = gr.Button(
                "Apply Trajectory data",
                visible=False,
                elem_id="motion_designer_trajectory_trigger",
            )

        trajectory_trigger.click(
            fn=self._apply_trajectory,
            inputs=[
                self.state,
                trajectory_payload,
                trajectory_metadata,
                trajectory_background,
            ],
            outputs=[self.refresh_form_trigger],
            show_progress="hidden",
        ).then(
            fn=self.goto_media_tab,
            inputs=[self.state],
            outputs=[self.main_tabs],
        )

        trigger.click(
            fn=self._apply_mask,
            inputs=[
                self.state,
                mask_payload,
                metadata_payload,
                background_payload,
                guide_payload,
                guide_metadata_payload,
            ],
            outputs=[self.refresh_form_trigger],
            show_progress="hidden",
        ).then(
            fn=self.goto_media_tab,
            inputs=[self.state],
            outputs=[self.main_tabs],
        )

        mode_sync.change(
            fn=lambda _: None,
            inputs=[mode_sync],
            outputs=[],
            show_progress="hidden",
            queue=False,
            js="""
            (mode) => {
                const raw = (mode || "").toString();
                const normalized = raw.split("|", 1)[0]?.trim().toLowerCase();
                if (!normalized) {
                    return;
                }
                if (window.motionDesignerSetRenderMode) {
                    window.motionDesignerSetRenderMode(normalized);
                } else {
                    console.warn("[MotionDesignerPlugin] motionDesignerSetRenderMode not ready yet.");
                }
            }
            """,
        )
        self.on_tab_outputs = [mode_sync]

    def on_tab_select(self, state: dict) -> str:
        model_def = self.get_model_def(state["model_type"])
        mode = "cut_drag" 
        if model_def.get("i2v_v2v", False):
            mode = "cut_drag"
        elif model_def.get("vace_class", False):
            mode = "classic"
        elif model_def.get("i2v_trajectory", False):
            mode = "trajectory"
        else:
            return gr.update()
        return f"{mode}|{time.time():.6f}"

    def _apply_mask(
        self,
        state,
        encoded_video: str | None,
        metadata_json: str | None,
        background_image_data: str | None,
        guide_video_data: str | None,
        guide_metadata_json: str | None,
    ):
        if not encoded_video:
            raise gr.Error("No mask video received from Motion Designer.")

        encoded_video = encoded_video.strip()
        try:
            video_bytes = base64.b64decode(encoded_video)
        except Exception as exc:
            raise gr.Error("Unable to decode the mask video payload.") from exc

        metadata: dict[str, object] = {}
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                if not isinstance(metadata, dict):
                    metadata = {}
            except json.JSONDecodeError:
                metadata = {}

        guide_metadata: dict[str, object] = {}
        if guide_metadata_json:
            try:
                guide_metadata = json.loads(guide_metadata_json)
                if not isinstance(guide_metadata, dict):
                    guide_metadata = {}
            except json.JSONDecodeError:
                guide_metadata = {}

        background_image = self._decode_background_image(background_image_data)

        output_dir = Path("mask_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = output_dir / f"motion_designer_mask_{timestamp}.webm"
        file_path.write_bytes(video_bytes)

        guide_path: Path | None = None
        if guide_video_data:
            try:
                guide_bytes = base64.b64decode(guide_video_data.strip())
                guide_path = output_dir / f"motion_designer_guide_{timestamp}.webm"
                guide_path.write_bytes(guide_bytes)
            except Exception as exc:
                print(f"[MotionDesignerPlugin] Failed to decode guide video payload: {exc}")
                guide_path = None

        fps_hint = None
        render_mode = ""
        if isinstance(metadata, dict):
            fps_hint = metadata.get("fps")
            render_mode = str(metadata.get("renderMode") or "").lower()
        if fps_hint is None and isinstance(guide_metadata, dict):
            fps_hint = guide_metadata.get("fps")
        if render_mode not in ("classic", "cut_drag") and isinstance(guide_metadata, dict):
            render_mode = str(guide_metadata.get("renderMode") or "").lower()
        if render_mode not in ("classic", "cut_drag"):
            render_mode = "cut_drag"

        sanitized_mask_path = self._transcode_video(file_path, fps_hint)
        sanitized_guide_path = self._transcode_video(guide_path, fps_hint) if guide_path else None
        self._log_frame_check("mask", sanitized_mask_path, metadata)
        if sanitized_guide_path:
            self._log_frame_check("guide", sanitized_guide_path, guide_metadata or metadata)

        # sanitized_mask_path = file_path
        # sanitized_guide_path = guide_path

        ui_settings = self.get_current_model_settings(state)
        if render_mode == "classic":
            ui_settings["video_guide"] = str(sanitized_mask_path)
            ui_settings.pop("video_mask", None)
            ui_settings.pop("video_mask_meta", None)
            if metadata:
                ui_settings["video_guide_meta"] = metadata
            else:
                ui_settings.pop("video_guide_meta", None)
        else:
            ui_settings["video_mask"] = str(sanitized_mask_path)
            if metadata:
                ui_settings["video_mask_meta"] = metadata
            else:
                ui_settings.pop("video_mask_meta", None)

            guide_video_path = sanitized_guide_path or sanitized_mask_path
            ui_settings["video_guide"] = str(guide_video_path)
            if guide_metadata:
                ui_settings["video_guide_meta"] = guide_metadata
            elif metadata:
                ui_settings["video_guide_meta"] = metadata
            else:
                ui_settings.pop("video_guide_meta", None)

        if background_image is not None:
            if render_mode == "classic":
                existing_refs = ui_settings.get("image_refs")
                if isinstance(existing_refs, list) and existing_refs:
                    new_refs = list(existing_refs)
                    new_refs[0] = background_image
                    ui_settings["image_refs"] = new_refs
                else:
                    ui_settings["image_refs"] = [background_image]
            else:
                ui_settings["image_start"] = [background_image]
        if render_mode == "classic":
            self.update_video_prompt_type(state, any_video_guide = True, any_background_image_ref = True, process_type = "")
        else:
            self.update_video_prompt_type(state, any_video_guide = True, any_video_mask = True, default_update="G")

        gr.Info("Motion Designer data transferred to the Media Generator.")
        return time.time()

    def _apply_trajectory(
        self,
        state,
        trajectory_json: str | None,
        metadata_json: str | None,
        background_data_url: str | None,
    ):
        if not trajectory_json:
            raise gr.Error("No trajectory data received from Motion Designer.")

        try:
            trajectories = json.loads(trajectory_json)
            if not isinstance(trajectories, list) or len(trajectories) == 0:
                raise gr.Error("Invalid trajectory data: expected non-empty array.")
        except json.JSONDecodeError as exc:
            raise gr.Error("Unable to parse trajectory data.") from exc

        metadata: dict[str, object] = {}
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                if not isinstance(metadata, dict):
                    metadata = {}
            except json.JSONDecodeError:
                metadata = {}

        # Convert to numpy array with shape [T, N, 2]
        # T = number of frames, N = number of trajectories, 2 = (x, y) coordinates
        trajectory_array = np.array(trajectories, dtype=np.float32)

        # Validate shape
        if len(trajectory_array.shape) != 3 or trajectory_array.shape[2] != 2:
            raise gr.Error(f"Invalid trajectory shape: expected [T, N, 2], got {trajectory_array.shape}")

        # Save to .npy file
        output_dir = Path("mask_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = output_dir / f"motion_designer_trajectory_{timestamp}.npy"
        np.save(file_path, trajectory_array)

        print(f"[MotionDesignerPlugin] Trajectory saved: {file_path} (shape: {trajectory_array.shape})")

        # Update UI settings with custom_guide path
        ui_settings = self.get_current_model_settings(state)
        ui_settings["custom_guide"] = str(file_path.absolute())

        # Decode and set background image as image_start
        background_image = self._decode_background_image(background_data_url)
        if background_image is not None:
            ui_settings["image_start"] = [background_image]

        gr.Info(f"Trajectory data saved ({trajectory_array.shape[0]} frames, {trajectory_array.shape[1]} trajectories).")
        return time.time()

    def _decode_background_image(self, data_url: str | None):
        if not data_url:
            return None
        payload = data_url
        if isinstance(payload, str) and "," in payload:
            _, payload = payload.split(",", 1)
        try:
            image_bytes = base64.b64decode(payload)
            with Image.open(io.BytesIO(image_bytes)) as img:
                return ImageOps.exif_transpose(img.convert("RGB"))
        except Exception as exc:
            print(f"[MotionDesignerPlugin] Failed to decode background image: {exc}")
            return None

    def _transcode_video(self, source_path: Path, fps: int | float | None) -> Path:
        frame_rate = max(int(fps), 1) if isinstance(fps, (int, float)) and fps else 16
        temp_path = source_path.with_suffix(".clean.webm")
        try:
            # Stream copy while stamping a constant frame rate into the container.
            (
                ffmpeg
                .input(str(source_path))
                .output(
                    str(temp_path),
                    c="copy",
                    r=frame_rate,
                    **{
                        "vsync": "cfr",
                        "fps_mode": "cfr",
                        "fflags": "+genpts",
                        "copyts": None,
                    },
                )
                .overwrite_output()
                .run(quiet=True)
            )
            if source_path.exists():
                source_path.unlink()
            temp_path.replace(source_path)
        except ffmpeg.Error as err:
            stderr = getattr(err, "stderr", b"")
            decoded = stderr.decode("utf-8", errors="ignore") if isinstance(stderr, (bytes, bytearray)) else str(stderr)
            print(f"[MotionDesignerPlugin] FFmpeg failed to sanitize mask video: {decoded.strip()}")
        except Exception as exc:
            print(f"[MotionDesignerPlugin] Unexpected error while sanitizing mask video: {exc}")
        return source_path

    def _probe_frames_fps(self, video_path: Path) -> tuple[int | None, float | None]:
        if not video_path or not video_path.exists():
            return (None, None)
        ffprobe_path = Path("ffprobe.exe")
        if not ffprobe_path.exists():
            ffprobe_path = Path("ffprobe")
        cmd = [
            str(ffprobe_path),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,avg_frame_rate,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, **child_kwargs())
            output = (result.stdout or "").strip().splitlines()
            # Expected output: nb_read_frames, nb_frames, avg_frame_rate, r_frame_rate (all may be present)
            frame_count = None
            fps_val = None
            for line in output:
                if line.strip().isdigit():
                    # First integer we see is usually nb_read_frames
                    val = int(line.strip())
                    frame_count = val
                elif "/" in line:
                    num, _, denom = line.partition("/")
                    try:
                        n = float(num)
                        d = float(denom)
                        if d != 0:
                            fps_val = n / d
                    except (ValueError, ZeroDivisionError):
                        continue
            return (frame_count, fps_val)
        except Exception:
            return (None, None)

    def _log_frame_check(self, label: str, video_path: Path, metadata: dict[str, object] | None):
        expected_frames = None
        if isinstance(metadata, dict):
            exp = metadata.get("expectedFrames")
            if isinstance(exp, (int, float)):
                expected_frames = int(exp)
        actual_frames, fps = self._probe_frames_fps(video_path)
        if expected_frames is None or actual_frames is None:
            return
        if expected_frames != actual_frames:
            print(
                f"[MotionDesignerPlugin] Frame count mismatch for {label}: "
                f"expected {expected_frames}, got {actual_frames} (fps probed: {fps or 'n/a'})"
            )

    def _get_iframe_markup(self) -> str:
        assets_dir = Path(__file__).parent / "assets"
        template_path = assets_dir / "motion_designer_iframe_template.html"
        script_path = assets_dir / "app.js"
        style_path = assets_dir / "style.css"

        cache_signature: tuple[int, int, int] | None = None
        try:
            cache_signature = (
                template_path.stat().st_mtime_ns,
                script_path.stat().st_mtime_ns,
                style_path.stat().st_mtime_ns,
            )
        except FileNotFoundError:
            cache_signature = None
        if (
            self._iframe_html_cache
            and cache_signature
            and cache_signature == self._iframe_cache_signature
        ):
            return self._iframe_html_cache

        template_html = template_path.read_text(encoding="utf-8")
        script_js = script_path.read_text(encoding="utf-8")
        style_css = style_path.read_text(encoding="utf-8")

        iframe_html = template_html.replace("<!-- MOTION_DESIGNER_STYLE_INLINE -->", f"<style>{style_css}</style>")
        iframe_html = iframe_html.replace("<!-- MOTION_DESIGNER_SCRIPT_INLINE -->", f"<script>{script_js}</script>")

        encoded = base64.b64encode(iframe_html.encode("utf-8")).decode("ascii")
        self._iframe_html_cache = (
            "<iframe id='motion-designer-iframe' "
            "title='Motion Designer' "
            "sandbox='allow-scripts allow-same-origin allow-pointer-lock allow-downloads' "
            "style='width:100%;border:none;border-radius:12px;display:block;' "
            f"src='data:text/html;base64,{encoded}'></iframe>"
        )
        self._iframe_cache_signature = cache_signature
        return self._iframe_html_cache

    def _js_bridge(self) -> str:
        return r"""
    const MOTION_DESIGNER_EVENT_TYPE = "WAN2GP_MOTION_DESIGNER";
    const MOTION_DESIGNER_CONTROL_MESSAGE_TYPE = "WAN2GP_MOTION_DESIGNER_CONTROL";
    const MOTION_DESIGNER_MODE_INPUT_SELECTOR = "#motion_designer_mode_sync textarea, #motion_designer_mode_sync input";
    const MOTION_DESIGNER_IFRAME_SELECTOR = "#motion-designer-iframe";
    const MOTION_DESIGNER_MODAL_LOCK = "WAN2GP_MOTION_DESIGNER_MODAL_LOCK";
    const MODAL_PLACEHOLDER_ID = "motion-designer-iframe-placeholder";
    let modalLockState = {
        locked: false,
        scrollX: 0,
        scrollY: 0,
        placeholder: null,
        prevStyles: {},
        unlockTimeout: null,
    };
    console.log("[MotionDesignerPlugin] Bridge script injected");

    function motionDesignerRoot() {
        if (window.gradioApp) {
            return window.gradioApp();
        }
        const app = document.querySelector("gradio-app");
        return app ? (app.shadowRoot || app) : document;
    }

    function motionDesignerDispatchInput(element, value) {
        if (!element) {
            return;
        }
        element.value = value;
        element.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function motionDesignerTriggerButton(appRoot) {
        return appRoot.querySelector("#motion_designer_apply_trigger button, #motion_designer_apply_trigger");
    }

    function motionDesignerGetIframe() {
        return document.querySelector(MOTION_DESIGNER_IFRAME_SELECTOR);
    }

    function motionDesignerSendControlMessage(action, value) {
        const iframe = motionDesignerGetIframe();
        if (!iframe || !iframe.contentWindow) {
            console.warn("[MotionDesignerPlugin] Unable to locate Motion Designer iframe for", action);
            return;
        }
        console.debug("[MotionDesignerPlugin] Posting control message", action, value);
        iframe.contentWindow.postMessage(
            { type: MOTION_DESIGNER_CONTROL_MESSAGE_TYPE, action, value },
            "*",
        );
    }

    function motionDesignerExtractMode(value) {
        if (!value) {
            return "";
        }
        return value.split("|", 1)[0]?.trim().toLowerCase() || "";
    }

    window.motionDesignerSetRenderMode = (mode) => {
        const normalized = motionDesignerExtractMode(mode);
        if (!normalized) {
            return;
        }
        let target;
        if (normalized === "classic") {
            target = "classic";
        } else if (normalized === "trajectory") {
            target = "trajectory";
        } else {
            target = "cut_drag";
        }
        console.log("[MotionDesignerPlugin] Mode sync triggered:", target);
        motionDesignerSendControlMessage("setMode", target);
    };

    window.addEventListener("message", (event) => {
        if (event?.data?.type === "WAN2GP_MOTION_DESIGNER_RESIZE") {
            if (typeof event.data.height === "number") {
                const iframe = document.querySelector("#motion-designer-iframe");
                if (iframe) {
                    iframe.style.height = `${Math.max(event.data.height, 400)}px`;
                }
            }
            return;
        }
        if (event?.data?.type === MOTION_DESIGNER_MODAL_LOCK) {
            const iframe = document.querySelector(MOTION_DESIGNER_IFRAME_SELECTOR);
            if (!iframe) {
                return;
            }
            const lock = Boolean(event.data.open);
            const clearUnlockTimeout = () => {
                if (modalLockState.unlockTimeout) {
                    clearTimeout(modalLockState.unlockTimeout);
                    modalLockState.unlockTimeout = null;
                }
            };
            if (lock) {
                clearUnlockTimeout();
                if (modalLockState.locked) {
                    return;
                }
                modalLockState.locked = true;
                modalLockState.scrollX = window.scrollX;
                modalLockState.scrollY = window.scrollY;
                const rect = iframe.getBoundingClientRect();
                const placeholder = document.createElement("div");
                placeholder.id = MODAL_PLACEHOLDER_ID;
                placeholder.style.width = `${rect.width}px`;
                placeholder.style.height = `${iframe.offsetHeight}px`;
                placeholder.style.pointerEvents = "none";
                placeholder.style.flex = iframe.style.flex || "0 0 auto";
                iframe.insertAdjacentElement("afterend", placeholder);
                modalLockState.placeholder = placeholder;
                modalLockState.prevStyles = {
                    position: iframe.style.position,
                    top: iframe.style.top,
                    left: iframe.style.left,
                    right: iframe.style.right,
                    bottom: iframe.style.bottom,
                    width: iframe.style.width,
                    height: iframe.style.height,
                    maxHeight: iframe.style.maxHeight,
                    zIndex: iframe.style.zIndex,
                };
                iframe.style.position = "fixed";
                iframe.style.top = "0";
                iframe.style.left = `${rect.left}px`;
                iframe.style.right = "auto";
                iframe.style.bottom = "auto";
                iframe.style.width = `${rect.width}px`;
                iframe.style.height = "100vh";
                iframe.style.maxHeight = "100vh";
                iframe.style.zIndex = "2147483647";
                document.documentElement.classList.add("modal-open");
                document.body.classList.add("modal-open");
            } else {
                clearUnlockTimeout();
                modalLockState.unlockTimeout = setTimeout(() => {
                    if (!modalLockState.locked) {
                        return;
                    }
                    modalLockState.locked = false;
                    iframe.style.position = modalLockState.prevStyles.position || "";
                    iframe.style.top = modalLockState.prevStyles.top || "";
                    iframe.style.left = modalLockState.prevStyles.left || "";
                    iframe.style.right = modalLockState.prevStyles.right || "";
                    iframe.style.bottom = modalLockState.prevStyles.bottom || "";
                    iframe.style.width = modalLockState.prevStyles.width || "";
                    iframe.style.height = modalLockState.prevStyles.height || "";
                    iframe.style.maxHeight = modalLockState.prevStyles.maxHeight || "";
                    iframe.style.zIndex = modalLockState.prevStyles.zIndex || "";
                    if (modalLockState.placeholder?.parentNode) {
                        modalLockState.placeholder.parentNode.removeChild(modalLockState.placeholder);
                    }
                    modalLockState.placeholder = null;
                    modalLockState.prevStyles = {};
                    document.documentElement.classList.remove("modal-open");
                    document.body.classList.remove("modal-open");
                    window.scrollTo(modalLockState.scrollX, modalLockState.scrollY);
                }, 120);
            }
            return;
        }
        if (!event?.data || event.data.type !== MOTION_DESIGNER_EVENT_TYPE) {
            return;
        }
        console.debug("[MotionDesignerPlugin] Received iframe payload", event.data);
        const appRoot = motionDesignerRoot();

        // Handle trajectory export separately
        if (event.data.isTrajectoryExport) {
            const trajectoryInput = appRoot.querySelector("#motion_designer_trajectory_payload textarea, #motion_designer_trajectory_payload input");
            const trajectoryMetaInput = appRoot.querySelector("#motion_designer_trajectory_meta textarea, #motion_designer_trajectory_meta input");
            const trajectoryBgInput = appRoot.querySelector("#motion_designer_trajectory_background textarea, #motion_designer_trajectory_background input");
            const trajectoryButton = appRoot.querySelector("#motion_designer_trajectory_trigger button, #motion_designer_trajectory_trigger");
            if (!trajectoryInput || !trajectoryMetaInput || !trajectoryBgInput || !trajectoryButton) {
                console.warn("[MotionDesignerPlugin] Trajectory bridge components missing in Gradio DOM.");
                return;
            }
            const trajectoryData = event.data.trajectoryData || [];
            const trajectoryMetadata = event.data.metadata || {};
            const backgroundImage = event.data.backgroundImage || "";
            motionDesignerDispatchInput(trajectoryInput, JSON.stringify(trajectoryData));
            motionDesignerDispatchInput(trajectoryMetaInput, JSON.stringify(trajectoryMetadata));
            motionDesignerDispatchInput(trajectoryBgInput, backgroundImage);
            trajectoryButton.click();
            return;
        }

        const maskInput = appRoot.querySelector("#motion_designer_mask_payload textarea, #motion_designer_mask_payload input");
        const metaInput = appRoot.querySelector("#motion_designer_meta_payload textarea, #motion_designer_meta_payload input");
        const bgInput = appRoot.querySelector("#motion_designer_background_payload textarea, #motion_designer_background_payload input");
        const guideInput = appRoot.querySelector("#motion_designer_guide_payload textarea, #motion_designer_guide_payload input");
        const guideMetaInput = appRoot.querySelector("#motion_designer_guide_meta_payload textarea, #motion_designer_guide_meta_payload input");
        const button = motionDesignerTriggerButton(appRoot);
        if (!maskInput || !metaInput || !bgInput || !guideInput || !guideMetaInput || !button) {
            console.warn("[MotionDesignerPlugin] Bridge components missing in Gradio DOM.");
            return;
        }

        const payload = event.data.payload || "";
        const metadata = event.data.metadata || {};
        const backgroundImage = event.data.backgroundImage || "";
        const guidePayload = event.data.guidePayload || "";
        const guideMetadata = event.data.guideMetadata || {};

        motionDesignerDispatchInput(maskInput, payload);
        motionDesignerDispatchInput(metaInput, JSON.stringify(metadata));
        motionDesignerDispatchInput(bgInput, backgroundImage);
        motionDesignerDispatchInput(guideInput, guidePayload);
        motionDesignerDispatchInput(guideMetaInput, JSON.stringify(guideMetadata));
        button.click();
    });

    function motionDesignerBindModeInput() {
        const appRoot = motionDesignerRoot();
        if (!appRoot) {
            setTimeout(motionDesignerBindModeInput, 300);
            return;
        }
        const input = appRoot.querySelector(MOTION_DESIGNER_MODE_INPUT_SELECTOR);
        if (!input) {
            setTimeout(motionDesignerBindModeInput, 300);
            return;
        }
        if (input.dataset.motionDesignerModeBound === "1") {
            return;
        }
        input.dataset.motionDesignerModeBound = "1";
        let lastValue = "";
        const handleChange = () => {
            const rawValue = input.value || "";
            const extracted = motionDesignerExtractMode(rawValue);
            if (!extracted || extracted === lastValue) {
                return;
            }
            lastValue = extracted;
            window.motionDesignerSetRenderMode(extracted);
        };
        const observer = new MutationObserver(handleChange);
        observer.observe(input, { attributes: true, attributeFilter: ["value"] });
        input.addEventListener("input", handleChange);
        handleChange();
    }

    motionDesignerBindModeInput();
    console.log("[MotionDesignerPlugin] Bridge initialization complete");
"""
