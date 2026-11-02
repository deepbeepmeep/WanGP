# Temporal Upsampler Plugin API

Temporal upsamplers (RIFE, frame interpolation Plugins, ...) are registered in
`postprocessing/temporal_upsamplers.py`. Each upsampler declares its methods and
supported multipliers through a handler object; the registry owns dropdown state,
value parsing, validation helpers, config nesting, downloads, and dispatch.

Temporal upsamplers are used on decoded video frames during generation and in
late post processing. They are video-only; late post processing validation rejects
temporal upsampling for still images.

## Handler Contract

```python
class MyTemporalUpsampler:
    def query_temporal_upsampler_def(self):
        return {
            "name": "MyTemporalUpsampler",
            "config_key": "mytemporal",                       # wgp_config["temporal_upsamplers"] subkey
            "pos": 100,                                       # default dropdown order for this handler's methods
            "method_pos": {"mytemporal": 100},                # optional per-method order
            "methods": [("MyTemporalUpsampler", "mytemporal")],
            "multipliers": {"mytemporal": (2.0, 4.0)},        # supported multipliers per method
            "default_temporal_upsampling": "mytemporal*2",
            "description": "Interpolate smoother video frames.", # optional discovery fallback
            "method_descriptions": {"mytemporal": "..."},    # optional per-method descriptions
            "method_parameters": {"mytemporal": [...]},      # optional extra parameter descriptors
        }

    def is_upsampling(self, value): ...                       # does this handler own the value?
    def split_value(self, value): ...                         # -> (method, scale) or None
    def build_value(self, method, scale): ...                 # -> value or None
    def validate_upsampling(self, value, *, source_is_image=False): ...  # -> "" or error text
    def temporal_upsample(self, value, sample, previous_last_frame, fps, *, processing_device="cuda", abort_callback=None, progress_callback=None, **kwargs): ...
    def download(self, process_files, send_cmd=None, status_text=None, temporal_upsampling=None): ...
    def query_download_def(self, *, enabled_only=True): ...    # -> one process_files definition
    def query_download_defs(self, *, enabled_only=True): ...   # -> list of process_files definitions
    def load_upsampler(self, value, **kwargs): ...             # optional pre-dispatch load hook
    def supports_loaded_model(self, value, context): ...       # optional core-model borrowing
    def release_private_runtime(self): ...                     # optional before borrowing core model
    def persistent_models(self): ...                          # optional, True keeps model in RAM and unloads VRAM only
    def release_vram(self): ...                               # optional Configuration-tab release hook
    def enabled(self): ...                                    # optional UI gating
    @property
    def status(self): ...                                     # optional: "enabled" or "disabled"
    @property
    def reason_disabled(self): ...                            # optional user-facing reason
    # optional Configuration tab integration:
    def default_config(self): ...                             # -> dict
    def normalize_config_section(self, section): ...          # -> normalized dict
    def create_config_ui(self, gr, section, *, lock_config=False): ...
    def validate_config_section(self, section): ...           # -> "" or message/list
    def config_requires_release(self, old, new, changed_keys): ...
```

Discovery evaluates the historical `enabled()` method first: `True` maps to
`enabled` and `False` to `disabled`. Only handlers without `enabled()` use the
optional `status` property. `unknown` means neither mechanism supplied a valid
status. A non-empty `reason_disabled` is exposed only for disabled handlers.
The internal catalog includes all registered temporal processors with this
metadata. Prime's post-processing toolbox omits disabled processors from
callable discovery; internal/compatibility views may include them for
diagnostics. A processor with disabled status cannot be dispatched.

`SimpleScaleSuffixMixin` provides `is_upsampling` / `split_value` / `build_value`
for the common `<method>*<multiplier>` value encoding, for example `rife*2` or
`mytemporal*4`. Its parser also accepts the former concatenated encoding so
existing settings and queues continue to work, while `build_value()` always
emits the `*` form.
The `*` character is reserved and must not appear inside a method id. Custom
handlers can use the registry's `format_multiplier_value()` and
`parse_multiplier_suffix()` helpers to follow the same mapping rule.

Handler-exposed method ids in `methods`, `multipliers`, and `method_pos` must be
multiplier-free. The multiplier only appears in serialized
`temporal_upsampling` values returned by `build_value()` or stored as
`default_temporal_upsampling`.

Dropdown entries are sorted by method position, then by method label. A handler
can define a default `pos` and override individual methods with `method_pos`.
Position is independent of multiplier; expanded choices such as `mytemporal*2`
and `mytemporal*4` share the `mytemporal` method position.

Discovery consumers infer the required `multiplier` parameter from
`multipliers`. Optional `description`, `method_descriptions`, and
`method_parameters` fields add reusable presentation and parameter metadata.
The Post Processing and Late Postprocessing information popup automatically
lists registered handlers using `description` and any per-method details from
`method_descriptions`.
Each `method_parameters` entry is a list of dictionaries with at least `name`;
it may also define `type`, `description`, `required`, `default`, `enum`, and a
keyword override named `setting`. These fields are optional so older and
third-party handlers remain compatible. Parameters are declared as for spatial
upsamplers: `name` is the id of the value in the slot's
`temporal_postprocessing_parameters` setting, `ui` lists the contexts where the
inputs below the slot's selector show it, an images parameter reads the reference
images shared with the spatial slot, and the processor receives each value under
its `setting` name. See
[Spatial Upsampler Plugin API](SPATIAL_UPSAMPLERS.md).

## Runtime Contract

`temporal_upsample(...)` receives the current decoded sample, an optional
`previous_last_frame` for chunk continuity, and the current `fps`. It must return:

```python
return sample, previous_last_frame, fps
```

When a handler inserts frames, it must update `fps` to match the new temporal
rate. For example, x2 interpolation should return `fps * 2`.

`previous_last_frame` is the continuity frame saved from the prior chunk. If the
handler consumes it, it should return the last frame of the postprocessed chunk so
the next chunk can continue smoothly.

Do not assume a different frame layout than the one passed by WanGP. Convert
dtypes or layouts only when the handler implementation requires it, and keep the
returned sample layout compatible with the surrounding WanGP postprocessing flow.

`postprocessing/temporal_upsamplers.py` wraps dispatch in shared attention state,
unloads the main generation offload object before running the temporal upsampler,
and releases registered extension resources after use. If `persistent_models()`
returns `True`, WanGP unloads the handler VRAM through the offload registry but
keeps persistent CPU/RAM state available for the next call.

A temporal handler may borrow an already loaded compatible generation model by
implementing `supports_loaded_model(value, context)`. On a match, dispatch skips
`load_upsampler()`, calls `release_private_runtime()` when present, and passes
the core-owned context as `loaded_model_context` to `temporal_upsample()`. A
non-match follows the existing private-runtime path. The handler may use the
borrowed model and MMGP offload object for that call, but must not release them
and must restore any temporary model state before returning.

## Registration

Registration is owned by `postprocessing/temporal_upsamplers.py`. Add the handler
class path to `temporal_upsampler_handlers`:

```python
temporal_upsampler_handlers = [
    "postprocessing.my_temporal.temporal_upsampler.MyTemporalUpsampler",
]
```

`wgp.py` only calls
`temporal_upsampler_api.register_temporal_upsamplers(server_config, fl)`; it
should not import or keep one explicit variable per temporal upsampler.

Enabled Plugins can expose temporal upsamplers without editing core code by
adding `temporal_upsampler_handlers` to `plugin_info.json`. Entries that start
with `.` are relative to the Plugin package root; entries without a leading `.`
are absolute import paths:

```json
{
  "name": "My Plugin",
  "temporal_upsampler_handlers": [
    ".temporal.MyTemporalUpsampler",
    "./nested/other_temporal.py:OtherTemporalUpsampler",
    "postprocessing.my_temporal.temporal_upsampler.MySharedTemporalUpsampler"
  ]
}
```

Only enabled/loaded Plugins are considered. The Plugin manager reuses cached
`plugin_info.json` metadata when registering those handlers.

## Configuration

Temporal upsampler settings are stored under
`wgp_config["temporal_upsamplers"][config_key]`.

Handlers with configuration should implement `default_config()` and
`normalize_config_section(...)`. If they expose Configuration-tab controls, return
`[(field_name, gradio_component), ...]` from `create_config_ui(...)`; WanGP uses
that binding to collect, normalize, validate, and save the nested section.

If a config change requires releasing loaded runtime state, implement either
`release_vram()` or `config_requires_release(old, new, changed_keys)`.

## Built-In Example

RIFE v4.26 is implemented by `postprocessing/rife/temporal_upsampler.py`:

- method id: `rife`
- serialized values: `rife*2`, `rife*3`, `rife*4`
- model file: `rife4.26.pkl` from `DeepBeepMeep/Wan2.1`
- no model-version configuration; all multipliers use RIFE v4.26

For a minimal external handler, see the
[Temporal Blend reference Plugin](https://github.com/deepbeepmeep/wan2gp-blender-upsampler).
It exposes `blend*2` and supplies the description displayed by the Temporal
Upsampling information button.

## Extension Offload Object Registry

Every extension that creates its own mmgp offload object (`offload.profile(...)`)
must register it in `shared/utils/offload_registry.py`:

```python
offload_registry.register_offloadobj("MyTemporalUpsampler", offloadobj, release_fn)
...
offload_registry.unregister_offloadobj("MyTemporalUpsampler", offloadobj)  # in release_fn
```

This lets WanGP track every extension offload object and release all extension
resources centrally: the toolbar "Unload Models" tool and Configuration Plugin
release button call `offload_registry.release_all()`.

Alternative methods can be available without appearing in the normal dropdown.
Extracting settings restores the saved method even when the extension currently
hides it. Selecting another method hides that alternative again. Disabled or
unavailable methods still follow the usual availability rules.

MMGP-managed temporal models use the shared **Postprocessor VRAM Usage** control
in Configuration > Extensions: **Minimum** (default), **Manual Choice** (at least
the processor's minimum), or **Dynamic** preload. Native processors keep their
own memory requirements.

---

> Applies to: Developing temporal upsampler Plugins: handlers, registration and shared configuration. Installed frame-interpolation options are available through WanGP's post-processing controls and API.
