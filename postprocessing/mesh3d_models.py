# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared pieces of the 3D post-processors (3D Refiners of the spatial slot, Texturers of the temporal slot) built on WanGP model types."""

from __future__ import annotations

from shared.utils import offload_registry
from shared.utils.default_device import call_with_default_device


class ModelRuntime:
    """A WanGP model type loaded for a 3D post-processor with the post-processors' memory policy (wgp.load_models, extension), its
    offload object registered under name so that WanGP releases it centrally. The loaded main model is used instead when the
    processor can borrow it (loaded_model_context)."""

    def __init__(self, name: str, model_type: str):
        self.name, self.model_type = name, model_type
        self.pipe = self.offloadobj = None

    def model(self, loaded_model_context=None):
        if loaded_model_context is not None:
            return loaded_model_context.model
        if self.pipe is None:
            import wgp

            self.pipe, self.offloadobj = wgp.load_models(self.model_type, output_type="image", track_as_main=False, extension=True)
            offload_registry.register_offloadobj(self.name, self.offloadobj, self.release)
        return self.pipe

    def run(self, fn, *args, **kwargs):
        """Calls a method of the model with the default-device policy of its model type, as for its generate()."""
        import wgp

        return call_with_default_device(wgp.get_model_def(self.model_type), self.model_type, fn, *args, **kwargs)

    def release(self) -> None:
        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj(self.name, self.offloadobj)
            self.offloadobj.release()
        self.pipe = self.offloadobj = None


class MethodValueMixin:
    """Processor values without multiplier: the method key itself."""

    def _method_keys(self):
        definition = self.query_temporal_upsampler_def() if hasattr(self, "query_temporal_upsampler_def") else self.query_upsampler_def()
        return [key for _, key in definition["methods"]]

    def split_value(self, value):
        method = str(value or "").strip().lower()
        return (method, 1.0) if method in self._method_keys() else None

    def build_value(self, method, scale=None):
        method = str(method or "").strip().lower()
        return method if method in self._method_keys() else None

    def is_upsampling(self, value) -> bool:
        return self.split_value(value) is not None
