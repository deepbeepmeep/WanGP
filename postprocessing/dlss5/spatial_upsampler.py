from __future__ import annotations

from postprocessing import spatial_upsamplers as api
from .runtime import DEPTH_RESOLUTION_DIVISORS, MOTION_VECTOR_METHODS, NR_MODES, USE_DEPTH_GUIDE, configure_depth_estimator, neural_render, release_flow_model, unavailable_reason


class DLSS5SpatialUpsampler(api.SimpleScaleSuffixMixin):
    METHOD = "dlss5"
    DEPTH_RESOLUTION_CHOICES = [("Full Res", "full"), ("Half Res (default)", "half"), ("Quarter Res", "quarter")]
    MOTION_VECTOR_CHOICES = [("Original (default, faster)", "original"), ("RAFT (better quality)", "raft")]
    batch_image_inputs = True

    def __init__(self, server_config=None, files_locator=None):
        self.server_config = server_config
        if server_config is not None:
            configure_depth_estimator(server_config)

    @classmethod
    def default_config(cls):
        return {"depth_resolution": "half", "motion_vector": "original"}

    @classmethod
    def normalize_config_section(cls, config):
        config = {**cls.default_config(), **dict(config or {})}
        depth_resolution = str(config["depth_resolution"]).strip().lower()
        motion_vector = str(config["motion_vector"]).strip().lower()
        return {
            "depth_resolution": depth_resolution if depth_resolution in DEPTH_RESOLUTION_DIVISORS else "half",
            "motion_vector": motion_vector if motion_vector in MOTION_VECTOR_METHODS else "original",
        }

    def config(self):
        return api.read_config_section(self.server_config, self)

    def query_download_defs(self, enabled_only=True):
        from preprocessing.processors import query_preprocessor_download_def

        definitions = [query_preprocessor_download_def("depth", self.server_config)] if USE_DEPTH_GUIDE else []
        if self.config()["motion_vector"] == "raft":
            definitions.append(query_preprocessor_download_def("flow", self.server_config))
        return definitions

    def download(self, process_files, *, send_cmd=None, status_text=None, gen=None, **_kwargs):
        from shared.utils.download import process_files_def_if_needed

        return process_files_def_if_needed(self.query_download_defs(), send_cmd=send_cmd, status_text=status_text or "Downloading DLSS 5 Model Files", gen=gen, process_files=process_files)

    @property
    def status(self):
        return api.PROCESSOR_STATUS_DISABLED if self.reason_disabled else api.PROCESSOR_STATUS_ENABLED

    @property
    def reason_disabled(self):
        return unavailable_reason(temporal=False)

    def create_config_ui(self, gr, config, *, lock_config=False):
        with gr.Group():
            with gr.Row():
                depth_resolution = gr.Dropdown(choices=self.DEPTH_RESOLUTION_CHOICES, value=config["depth_resolution"], label="DLSS 5 Depth Resolution Precision", info="Run depth extraction at the render resolution, or resize each frame to half/quarter width and height for lower latency and memory use.", interactive=not lock_config)
                motion_vector = gr.Dropdown(choices=self.MOTION_VECTOR_CHOICES, value=config["motion_vector"], label="DLSS 5 Motion Vector", info="Original uses the faster OpenCV DIS estimator. RAFT generally produces higher-quality motion vectors.", interactive=not lock_config)
        return [("depth_resolution", depth_resolution), ("motion_vector", motion_vector)]

    @classmethod
    def query_upsampler_def(cls):
        reason = unavailable_reason(temporal=False)
        label = "DLSS 5 Neural Rendering" + (f" ({reason})" if reason else "")
        return {
            "name": "DLSS 5 Neural Rendering",
            "upsampler_types": (api.UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video", "image"),
            "profile": api.UPSAMPLER_PROFILE_VIDEO,
            "config_key": "dlss5",
            "pos": 10_000,
            "method_pos": {cls.METHOD: 10_000},
            "methods": [(label, cls.METHOD)],
            "multipliers": {cls.METHOD: tuple(NR_MODES)},
            "default_spatial_upsampling": "dlss5*1",
            "postprocessing_category": api.POSTPROCESSING_CATEGORY_UPSAMPLER,
            "description": "DLSS 5 Neural Rendering. At x1 it refines at native resolution; higher modes refine and upscale. Requires Windows and a compatible NVIDIA RTX GPU. WSL can use the Windows worker through Windows interoperability; this is not native Linux Neural Rendering.",
            "method_parameters": {cls.METHOD: [
                {"name": "dlss_strength", "setting": "intensity", "type": "number", "component": "slider", "ui": (api.PARAMETER_UI_POSTPROCESSING, api.PARAMETER_UI_LATE_POSTPROCESSING, api.PARAMETER_UI_MEDIA_FLOW), "required": False, "default": 1.0, "minimum": 0.0, "maximum": 2.0, "step": 0.05, "label": "DLSS 5 NR Intensity", "description": "Controls the native Neural Rendering intensity. 1 is the native default; values up to 2 strengthen the effect."},
            ]},
        }

    def validate_upsampling(self, spatial_upsampling, image_mode):
        split = self.split_value(spatial_upsampling)
        if split is None or split[1] not in NR_MODES:
            return f"Unknown DLSS 5 Neural Rendering mode: {spatial_upsampling}"
        reason = unavailable_reason(temporal=False)
        return f"DLSS 5 Neural Rendering is unavailable: {reason}. See docs/DLSS5.md." if reason else ""

    def upscale(self, sample, spatial_upsampling, *, still_image=False, intensity=1.0, abort_callback=None, progress_callback=None, **kwargs):
        error = self.validate_upsampling(spatial_upsampling, int(still_image))
        if error:
            raise RuntimeError(error)
        config = self.config()
        output = neural_render(sample, self.split_value(spatial_upsampling)[1], still_image=still_image, depth_resolution=config["depth_resolution"], motion_vector=config["motion_vector"], intensity=intensity, abort_callback=abort_callback, progress_callback=progress_callback)
        return output, None

    def release_vram(self):
        release_flow_model()
