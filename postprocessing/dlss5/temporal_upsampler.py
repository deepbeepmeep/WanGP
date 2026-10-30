from __future__ import annotations

from postprocessing import spatial_upsamplers as spatial_api
from postprocessing import temporal_upsamplers as api
from .runtime import DLSSG_LINUX, DLSSG_MULTIPLIERS, DLSSG_NATIVE_METHOD, DLSSG_UPSTREAM, DlssgVariant, dlssg_capabilities, dlssg_variants, format_multiplier_range, frame_generate, is_rtx_50_series, multiplier_range, multipliers_for, nominal_multiplier_range, release_flow_model, unavailable_reason
from .spatial_upsampler import DLSS5SpatialUpsampler


class DLSSGTemporalUpsampler(api.SimpleScaleSuffixMixin):
    METHOD = DLSSG_UPSTREAM.method
    NATIVE_METHOD = DLSSG_NATIVE_METHOD
    MULTIPLIERS = DLSSG_MULTIPLIERS

    def __init__(self, server_config=None, files_locator=None):
        self.server_config = server_config
        if server_config is not None:
            from .runtime import configure_depth_estimator
            configure_depth_estimator(server_config)

    def config(self):
        return DLSS5SpatialUpsampler.normalize_config_section(spatial_api.read_config_section_by_key(self.server_config, "dlss5"))

    @staticmethod
    def _reason(variant: DlssgVariant) -> str:
        return unavailable_reason(temporal=True, variant=variant)

    @staticmethod
    def _variant(method) -> DlssgVariant | None:
        method = str(method or "").strip().lower()
        return next((variant for variant in dlssg_variants() if variant.method == method), None)

    def query_download_defs(self, enabled_only=True):
        from preprocessing.processors import query_preprocessor_download_def

        return [query_preprocessor_download_def("flow", self.server_config)] if self.config()["motion_vector"] == "raft" else []

    def download(self, process_files, *, send_cmd=None, status_text=None, gen=None, **_kwargs):
        from shared.utils.download import process_files_def_if_needed

        return process_files_def_if_needed(self.query_download_defs(), send_cmd=send_cmd, status_text=status_text or "Downloading DLSS Frame Generation Model Files", gen=gen, process_files=process_files)

    @property
    def status(self):
        return api.PROCESSOR_STATUS_DISABLED if all(self._reason(variant) for variant in dlssg_variants()) else api.PROCESSOR_STATUS_ENABLED

    @property
    def reason_disabled(self):
        return "; ".join(f"{variant.name}: {self._reason(variant)}" for variant in dlssg_variants() if self._reason(variant))

    @staticmethod
    def _variant_of(native_linux: bool) -> DlssgVariant | None:
        return next((variant for variant in dlssg_variants() if variant.native_linux == native_linux), None)

    @classmethod
    def _upstream_covers_x2(cls) -> bool:
        """True when the upstream worker is installed, usable and offers x2."""
        upstream = cls._variant_of(False)
        return upstream is not None and not cls._reason(upstream) and 2.0 in multipliers_for(upstream)

    @classmethod
    def _usable(cls, variant: DlssgVariant) -> bool:
        """True when the worker is installed, probed and currently usable."""
        capabilities = dlssg_capabilities(variant)
        return not cls._reason(variant) and bool(capabilities) and bool(capabilities.get("available"))

    @classmethod
    def _redundant_native(cls) -> DlssgVariant | None:
        """The native worker a working upstream x2 entry already covers.

        An x2-only native worker that works is a strict subset of a working
        upstream x2 path, so it is hidden from the menus. A native worker that
        cannot run is never hidden, because its label is what explains why.
        """
        native = cls._variant_of(True)
        upstream = cls._variant_of(False)
        if native is None or upstream is None:
            return None
        if cls._upstream_covers_x2() and cls._usable(native) and multipliers_for(native) == (2.0,):
            return native
        return None

    def method_visible(self, method: str) -> bool:
        """Hide the covered native entry from the menus only, never from the handler.

        The shared contract keeps a hidden method usable and puts it back in a
        menu when a saved selection asks for it, so the redundancy check above
        cannot swallow a stored value, a status line or a discovery listing.
        """
        variant = self._variant(method)
        return not (variant is not None and variant.native_linux and self._redundant_native() is not None)

    @classmethod
    def _display_range(cls, variant: DlssgVariant) -> tuple[float, float, bool]:
        """The range to show, and whether the capability probe reported it."""
        capabilities = dlssg_capabilities(variant)
        verified = bool(capabilities) and int(capabilities.get("worker_version", 0) or 0) > 0 and "multi_frame_count_max" in capabilities
        low, high = multiplier_range(variant) if verified else nominal_multiplier_range(variant)
        if verified and variant.native_linux and high > 2.0 and low == 2.0 and cls._upstream_covers_x2():
            low = 3.0
        return low, high, verified

    @staticmethod
    def _format_range(low: float, high: float) -> str:
        return f"x{low:g}" if high <= low else f"x{low:g}-x{high:g}"

    @classmethod
    def _label(cls, variant: DlssgVariant) -> str:
        """Name the entry with the multipliers it brings, plus the blocker when any.

        "supported" means the capability probe verified the range on this GPU;
        "max" is this GPU's ceiling, shown while the worker cannot be probed.
        x2 is left out of the range when a usable upstream entry already offers
        it. The upstream entry keeps upstream's text, and Windows lists a single
        entry, so the range is shown on native Linux only.
        """
        base = DLSSG_UPSTREAM.name
        reason = unavailable_reason(temporal=True, variant=variant, short=True)
        if not DLSSG_LINUX or not variant.native_linux:
            return f"{base} ({reason})" if reason else base
        low, high, verified = cls._display_range(variant)
        text = f"{cls._format_range(low, high)} {'supported' if verified else 'max'}"
        if reason:
            text = f"{text}; disabled: {reason}"
        return f"{base} ({text})"

    @staticmethod
    def _method_description(variant: DlssgVariant) -> str:
        capabilities = dlssg_capabilities(variant)
        if not variant.native_linux:
            base = "NVIDIA's DLSS Frame Generation worker. Windows supports x2-x4 and capability-gated x5/x6 on RTX 50 series, and requires Windows 11, HAGS, and RTX 40 or newer."
            if DLSSG_LINUX:
                base += f" Native Linux runs the upstream x2 path with the NVIDIA Vulkan driver, pinned to {variant.runtime.name} in {variant.dir.name}/."
            return f"{base} Worker: {variant.label_path}."
        detail = (
            f" (probe: worker v{capabilities.get('worker_version', '?')}, {capabilities.get('multi_frame_count_max', 1)} generated frame(s) per pair, runtime {capabilities.get('runtime_version', 'unknown')})"
            if capabilities
            else " (the capability probe failed; run the worker with --probe for diagnostics)"
        )
        note = (
            " A label reading 'supported' is the range the probe verified on this GPU, while 'max' is this GPU's ceiling shown while the worker cannot be probed."
            " When the upstream x2 entry is usable, this entry is labelled from x3 because that entry already offers x2, and an x2-only native worker is hidden from the menus while the upstream entry works: it stays usable, the menus restore it when a saved selection asks for it, and it reappears when it is the only path."
            " On a multi-GPU host the range follows the GPU WanGP is restricted to (CUDA_VISIBLE_DEVICES), the weakest of them, and the worker is pinned to that GPU; see docs/DLSS5.md."
        )
        return f"Native Linux DLSSG worker (scripts/build_dlssg_worker_linux.sh) supporting up to {format_multiplier_range(variant, nominal=True)} on this GPU{detail}.{note} Requires the NVIDIA Vulkan driver and an RTX 40 or newer GPU; x5/x6 require an RTX 50 series GPU. Worker: {variant.label_path}."

    def query_temporal_upsampler_def(self):
        methods, multipliers, method_pos, method_descriptions = [], {}, {}, {}
        for offset, variant in enumerate(dlssg_variants()):
            methods.append((self._label(variant), variant.method))
            multipliers[variant.method] = multipliers_for(variant)
            method_pos[variant.method] = 10_000 + offset
            method_descriptions[variant.method] = self._method_description(variant)
        return {
            "name": "DLSS Frame Generation",
            "config_key": "dlssg",
            "pos": 10_000,
            "method_pos": method_pos,
            "methods": methods,
            # Compact name for status lines and video metadata; the method
            # labels above are the long ones the pickers show.
            "method_short_labels": {variant.method: DLSSG_UPSTREAM.name for variant in dlssg_variants()},
            "multipliers": multipliers,
            "method_descriptions": method_descriptions,
            "default_temporal_upsampling": "dlssg*2",
            "description": "NVIDIA DLSS Frame Generation using video motion estimation. Every installed worker is listed separately: the upstream worker (Windows x2-x4 plus capability-gated x5/x6 on RTX 50 series, requiring Windows 11, HAGS, and RTX 40 or newer; native Linux x2 with the NVIDIA Vulkan driver) and, on Linux, the native multi-multiplier worker shown with the range it supports here. The native range omits x2 when a usable upstream entry already offers it, and an x2-only native worker is hidden from the menus while that upstream entry works, keeping its multipliers, its validation and any saved selection that uses it. The choice is made per generation with no restart and no environment variable. See docs/DLSS5.md.",
        }

    def validate_upsampling(self, temporal_upsampling, *, source_is_image=False):
        split = self.split_value(temporal_upsampling)
        if split is None or split[1] not in self.MULTIPLIERS:
            return f"Unknown DLSS Frame Generation mode: {temporal_upsampling}"
        if source_is_image:
            return "Temporal Upsampling can not be used with an Image"
        variant = self._variant(split[0])
        if variant is None:
            return "The native Linux DLSS Frame Generation worker is not installed. See docs/DLSS5.md."
        if DLSSG_LINUX and not variant.native_linux and split[1] != 2:
            return "Linux DLSS Frame Generation currently supports x2 only. Select the native Linux worker for higher multipliers."
        reason = self._reason(variant)
        if reason:
            return f"DLSS Frame Generation is unavailable: {reason}. See docs/DLSS5.md."
        capabilities = dlssg_capabilities(variant)
        worker_version = int(capabilities.get("worker_version", 0) or 0)
        maximum = int(capabilities.get("multi_frame_count_max", 1) or 1)
        if split[1] > 4 and not is_rtx_50_series(variant):
            return "DLSS Frame Generation x5/x6 requires a GeForce RTX 50-series GPU."
        if split[1] > 2 and worker_version < 2:
            return "DLSS Frame Generation x3-x6 requires worker v2. See docs/DLSS5.md."
        if split[1] - 1 > maximum:
            return f"DLSS Frame Generation on this GPU supports up to x{maximum + 1}."
        return ""

    def temporal_upsample(self, temporal_upsampling, sample, previous_last_frame, fps, *, abort_callback=None, progress_callback=None, **kwargs):
        error = self.validate_upsampling(temporal_upsampling)
        if error:
            raise RuntimeError(error)
        method, scale = self.split_value(temporal_upsampling)
        return frame_generate(sample, previous_last_frame, fps, int(scale), motion_vector=self.config()["motion_vector"], abort_callback=abort_callback, progress_callback=progress_callback, variant=self._variant(method))

    def release_vram(self):
        release_flow_model()
