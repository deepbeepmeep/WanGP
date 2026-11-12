# Spatial Upsampler / Visual Refiner Plugin API

Spatial upsamplers and visual refiners (Lanczos, FlashVSR, RelayVSR, SeedVR2, PiD, H3 Face Refiner, TaoMate-FVSR Face Refiner,
Chain-of-Zoom, VAE upscalers, ...) are
registered in `postprocessing/spatial_upsamplers.py`. Each upsampler declares itself and its
capabilities through a handler object; the registry owns dropdown state, value
parsing, validation helpers, config nesting, downloads and dispatch.

## Upsampler types

**MiniMax H3 VAE** is available in Spatial Upsampling during H3 video or image
generation. It replaces the default VAE and handles VAE decoding and upsampling
together. Choose **×2** to double the output width and height, **×1.5** to
enlarge them by half, or **×1** to keep the original size. ×1 and ×1.5 are
downscaled from the ×2 decoding. **VAE Upscaling** (Wan, Qwen and Krea image
generation) offers the same three choices. When a video continues a Source
Video, the source frames are resized to the output size.

- `postprocessing`: works on decoded frames. Interchangeable: WanGP can call any of
  them through the same `upscale()` interface, both at generation time and in
  **late post processing** (Post Processing tab on existing media). Image-only
  handlers (`"media": ("image",)`) are offered for image outputs and in late
  post processing.
- `vae`: plugged directly into a model pipeline (e.g. Wan VAE 2x, PiD VAE). They
  expose capabilities through the same API. Handlers declare whether a selected
  VAE upsampler requires a main model reload (Wan VAE) or an external runtime
  session passed to the model pipeline (PiD-style upsamplers).

Decoded-media handlers also declare `postprocessing_category`:

- `upsampler`: increases spatial resolution and declares one or more scale
  multipliers.
- `refiner`: improves decoded media without necessarily resizing it. A refiner
  may omit `multipliers`; WanGP then hides the Scale control and serializes the
  bare method id.

## FlashVSR Versions

FlashVSR v1.1 and FlashVSR+ Turbo have separate selections, including their
two-pass variants. **Configuration > Extensions > FlashVSR > Model** controls
which version is normally shown. Saving a version change updates the generation
and gallery menus immediately. A selected FlashVSR method keeps its scale and
one-pass or two-pass setting. **Extract Settings** restores the saved
version and scale even if the current configuration hides that version;
running those restored settings uses that version without changing the
configuration preference. Existing `flashvsr` and `flashvsr2pass` settings mean
FlashVSR v1.1; `flashvsrplus` and `flashvsrplus2pass` mean FlashVSR+ Turbo.

FlashVSR remembers what it computed for the last frames of the video so far.
With **Configuration > Extensions > Postprocessor VRAM Usage** set to **Minimum**,
that memory stays in RAM and the GPU holds one layer of it at a time, the least
VRAM. With **Manual Choice** or **Dynamic** it stays in VRAM as far as the VRAM
allows, which speeds the upscaling further.

## SeedVR2 Models

SeedVR2 3B and SeedVR2 7B have separate selections, and
**Configuration > Extensions > SeedVR2 > SeedVR2 Model** controls which one is
normally shown (3B by default). Saving a change switches an active SeedVR2
selection to that model at the same scale. The two trade appearance for fidelity: in the
authors' comparisons 3B looks better on AI-generated and real-world video,
while 7B stays closer to the source on degraded footage. 7B downloads as an
INT8 file a little larger than 3B and runs slightly slower. **Extract Settings**
restores the saved model even when the configuration hides it. Existing
`seedvr2` settings mean SeedVR2 3B; `seedvr2_7b` means SeedVR2 7B. Media Flow's
SeedVR2 processes use the configured model.

## RelayVSR

RelayVSR restores detail while upscaling real-world and generated video,
several times faster than FlashVSR+ Turbo. Every 15 frames a one-step
Wan 2.2 5B model generates a detailed keyframe; a small decoder then carries
the detail of the two keyframes around each frame to every frame. Images are
processed as one-frame videos.

It has two models. The x4 model serves the ratios above x2. **RelayVSR X2**,
LightX2V's native x2 version, serves x1 to x2: it does a quarter of the x4
model's work with a quarter of its memory, more than twice as fast. Other ratios
are resized from the model's output. **Configuration > Extensions > RelayVSR >
RelayVSR Model For Upscaling Up To x2** can use the x4 model there instead,
resized down. Each model downloads on its first use.

Its memory grows with the size of the source, not with the length of the
video: the keyframe model remembers its last four keyframes, in INT8, about
4.6 GB for an 832x480 source and 10.5 GB for 1280x720 with the x4 model, a
quarter of that with RelayVSR X2. Where it lives follows
**Configuration > Extensions > Postprocessor VRAM Usage**:
- **Minimum** keeps it in RAM: the GPU holds one layer of it at a time, the
  least VRAM, at some cost in speed (each layer is brought in as it runs).
- **Manual Choice** and **Dynamic** keep it in VRAM as far as the VRAM allows,
  the rest in RAM, brought to the GPU ahead of each layer, which costs little
  speed: a source whose memory exceeds the GPU's still runs.

RelayVSR suits sources up to about 540p at x4, and larger ones at x2. Its
keyframe models download in INT8, whatever the transformer quantization.

## Handler contract

```python
class MyUpsampler:
    def query_upsampler_def(self):
        return {
            "name": "MyUpsampler",
            "upsampler_types": ("postprocessing",),        # and/or "vae"
            "media": ("video", "image"),
            "profile": "video",                            # memory profile: video, image, or audio
            "config_key": "myup",                          # wgp_config["spatial_upsamplers"] subkey
            "pos": 100,                                     # default dropdown order for this handler's methods
            "method_pos": {"myup": 100},                    # optional per-method order; independent of multiplier
            "methods": [("MyUpsampler", "myup")],          # method dropdown entries
            "vae_methods": [],                             # VAE entries (manual integration)
            "multipliers": {"myup": (2.0, 4.0)},           # supported multipliers per method
            "default_spatial_upsampling": "myup*2",
            "postprocessing_category": "upsampler",       # "upsampler" or "refiner"
            "description": "Upscale while restoring detail.", # processor-owned help/discovery text
            "media_descriptions": {"video": "Uses overlapping windows."}, # optional media-specific help
            "method_descriptions": {"myup": "..."},       # optional per-method descriptions
            "method_parameters": {"myup": [{
                "name": "prompt",                         # id in spatial_postprocessing_parameters
                "setting": "prompt",                      # upscale() keyword
                "type": "string",
                "component": "textbox",
                "ui": ("late_postprocessing",),           # where to display the control
                "required": False,
                "default": "",
                "label": "Refiner Prompt",
                "description": "Optional refinement instructions.",
            }]},
            "source_audio_conditioning": False,            # request source audio in late postprocessing
        }

    def is_upsampling(self, value): ...                    # does this handler own the value?
    def split_value(self, value): ...                      # -> (method, scale) or None
    def build_value(self, method, scale): ...              # -> value or None
    def validate_upsampling(self, value, image_mode): ...  # -> "" or error text
    def query_chunking(self, value): ...                   # optional: how a long video is processed by chunks (see Long videos)
    # postprocessing type only:
    def upscale(self, sample, value, *, seed, ..., abort_callback, progress_callback): ...
    # progress_callback(phase, current=None, total=None, unit=None): unit (e.g. "tiles") labels the counter, default "steps"
    def download(self, process_files, send_cmd=None, status_text=None, spatial_upsampling=None): ...
    def query_checkpoint_inventory(self): ...                 # all checkpoint variants, independent of the selected settings; see PLUGINS.md
    def load_upsampler(self, value, **kwargs): ...            # optional pre-dispatch load hook
    def supports_loaded_model(self, value, context, **kwargs): ... # optional core-model borrowing
    def release_private_runtime(self): ...                    # optional before borrowing core model
    def release_vram(self): ...
    def enabled(self): ...                                 # optional UI gating
    def method_available(self, method): ...                # optional per-method availability
    def method_visible(self, method): ...                  # optional default dropdown visibility
    @property
    def status(self): ...                                  # optional: "enabled" or "disabled"
    @property
    def reason_disabled(self): ...                         # optional user-facing reason
    # VAE type only:
    def supports_model_vae_method(self, method, model_type, model_def, image_mode): ...
    def prepare_vae_upsampler(self, value, *, send_cmd, process_files, init_pipe, profile, attention_mode=None): ...
    def model_load_upsampling_value(self, value, model_type, model_def, image_mode): ...
    def loaded_model_vae_upsampling_value(self, model): ...
    def model_load_kwargs_for_vae_upsampling(self, value, model_type, model_def, image_mode): ...
    def post_model_process_vae_upsampling(self, sample, value): ...  # optional: final output size, applied right after decoding
    # optional Configuration tab integration:
    def default_config(self): ...                          # -> dict
    def legacy_config(self, server_config): ...            # -> old top-level values, if any
    def legacy_config_keys(self): ...                      # -> keys deleted after migration
    def normalize_config_section(self, section): ...       # -> normalized dict
    def create_config_ui(self, gr, section, *, lock_config=False): ...
    def validate_config_section(self, section): ...        # -> "" or message/list
    def config_requires_release(self, old, new, changed_keys): ...
```

Discovery evaluates the historical `enabled()` method first: `True` maps to
`enabled` and `False` to `disabled`. Only handlers without `enabled()` use the
optional `status` property. Discovery always emits `enabled`, `disabled`, or
`unknown`; `unknown` means neither mechanism supplied a valid status.
`reason_disabled` is included only when the normalized status is `disabled` and
the handler provides a non-empty reason. This describes the internal catalog;
Prime's post-processing toolbox omits disabled processors from callable
discovery. Internal/compatibility views may include them for diagnostics,
but a process reported as disabled cannot be dispatched.

`SimpleScaleSuffixMixin` provides `is_upsampling`/`split_value`/`build_value` for the
common `<method>*<multiplier>` value encoding (e.g. `lanczos*2`, `coz*4`). Its
parser also accepts the former concatenated encoding so existing settings and
queues continue to work, while `build_value()` always emits the `*` form.
The `*` character is reserved and must not appear inside a method id. Custom
handlers can use the registry's `format_multiplier_value()` and
`parse_multiplier_suffix()` helpers to follow the same mapping rule.
Handler-exposed method ids in `methods`, `vae_methods`, `multipliers`,
`method_pos`, `model_def["vae_upsamplers"]`, and
`model_def["excluded_spatial_upsamplers"]` must be multiplier-free. The multiplier
only appears in serialized `spatial_upsampling` values returned by `build_value()`
or stored as `default_spatial_upsampling`.

Dropdown entries are sorted by method position, then by method label. A handler
can define a default `pos` and override individual methods with `method_pos`.
Position is independent of multiplier; expanded choices such as `myup*2` and
`myup*4` share the `myup` method position.

`method_visible(method)` can hide an available method from normal dropdown
choices. The registry includes a hidden method when restoring it as the current
selection, then hides it again when another method is selected. Visibility does
not affect method lookup, metadata labels, API discovery or execution. Media
compatibility, model exclusions and `method_available(method)` still apply;
restoring a hidden choice never makes an unavailable method usable. Handlers
without this hook show all their available methods.

### Long videos: chunk rules

A long video is upsampled a chunk at a time, so that it never has to fit in
memory: in Late Post Processing, for the Source Video of a continued
generation, and in Media Flow. WanGP does not know any upsampler: each one
declares how its chunks are cut and joined through `query_chunking(value)`,
which returns an object with the fields below. Every field is optional. An
upsampler without `query_chunking` receives its video in one piece.

| Field | Meaning |
|---|---|
| `get_chunk_frames(frames_count)` | Frames of a chunk when the model has a window of its own. Without it WanGP sizes the chunks by memory (Media Flow: the user's Chunk Size). |
| `chunk_frame_step` | Chunks sized by WanGP hold `n * chunk_frame_step + 1` frames. Default 1. |
| `minimum_requested_frames` | Fewest frames of a chunk. Default 1. |
| `overlap_frames`, `get_overlap_frames(chunk_frames)` or `get_overlap_frames_for_target(chunk_frames, value)` | Source frames a chunk shares with the one before it. Default 0. |
| `crossfade_overlap_outputs` | `True`: the shared frames fade from the previous chunk's output to this one's. `False` (default): they are this chunk's. |
| `overlap_output_split(overlap_frames)` | With `crossfade_overlap_outputs`: how many shared frames are kept from the previous chunk, the others being this chunk's. `None` fades. |
| `output_write_range(returned_frames, source_write_start=, source_write_end=, next_overlap_frames=)` | The output frames kept from a chunk, for an upsampler that does not return one frame per source frame. |
| `carries_cache(value)` | `True`: `upscale()` is called with `return_continue_cache=True`, and what it returns beside the frames reaches the next chunk as `continue_cache`. |
| `release_continue_cache()` | Called after the last chunk, or when the video is stopped. |

Each chunk is one `upscale()` call, which also receives `frame_offset`, the
place of its first frame in the video. Choose the join that matches the model:

- **Frames that do not depend on their neighbours** (a resize): return
  `postprocessing.chunking.FRAME_BY_FRAME`. Any chunk gives the frames of the
  whole video.
- **Independent runs that invent detail:** declare a few `overlap_frames` and
  `crossfade_overlap_outputs = True`, so that the texture of one chunk turns
  into the next one's over the shared frames instead of changing in one frame.
- **A model that continues from its own state:** declare `carries_cache` and
  hand that state from chunk to chunk. No fade is needed when the shared frames
  come out the same in both chunks.
- **A model that builds each frame from the ones before it, without a state to
  hand over:** a chunk starts cold. Share enough frames for it to settle and
  fade them (`crossfade_overlap_outputs = True`); DLSS 5 shares 16. One shared
  frame leaves a visible change at every chunk start.

Check a declaration on the real model: process a video by chunks and in one
piece, then compare the change from frame to frame across each chunk start.

A video that fits in memory is still processed in one call. Upsamplers that
declare `source_audio_conditioning` also receive their video in one piece, and
so does a video going through a frame interpolator and an upsampler that both
need neighbouring frames.

```python
class MyChunkRules:
    overlap_frames = 4
    crossfade_overlap_outputs = True

    def get_overlap_frames(self, chunk_frames):
        return min(self.overlap_frames, chunk_frames - 1)

class MyUpsampler:
    def query_chunking(self, value):
        return MyChunkRules()
```

The built-in upsamplers return the object that also describes their Media Flow
process (`postprocessing/<name>/process_handler.py`), so both read the same
rules.

### Borrowing the loaded generation model

A postprocessor that can run through an already loaded generation pipeline may
implement `supports_loaded_model(value, context)`. The context contains the
core-owned pipeline, MMGP offload object, model type/family/definition, profile,
and selected config. If the method returns true, the registry skips
`load_upsampler()`, passes the context to `upscale(...,
loaded_model_context=context)`, and leaves ownership of the model and offload
object with WanGP. `release_private_runtime()` is called first so the handler
does not retain a duplicate private model. If compatibility returns false, the
normal private-runtime path is used; this is also the fallback when no core
model is loaded.

Borrowed runtimes always detach their borrowed references after the call,
even when private processor runtimes are retained for reuse. This detaches references
before the core MMGP owner can be released or replaced.

Borrowing is an explicit capability contract, not an architecture guess. A
pipeline should advertise the protocol its handler expects (for example H3 uses
`refinement_api = "masked_video_sigma_v1"`). A borrowing handler that changes
temporary model state, such as active LoRAs or caches, must restore that state
before returning.

Discovery consumers infer a required `multiplier` parameter only when the method
declares `multipliers`. `description` or `method_descriptions` supplies both
Deepy's process description and the dynamic help next to WanGP's selector.
`media_descriptions` can add guidance specific to `image` or `video`. The UI
compiles help only from methods available for the current media and, during
generation, the current model; late postprocessing follows the selected gallery
item.

Each `method_parameters` entry is a list of dictionaries with at least `name`,
`type`, `description`, and `required`. It may also define `default`, `enum`,
`minimum`, `maximum`, and a runtime keyword override named `setting`. `name` is
the parameter's id: its value travels under that id in the slot's dictionary,
`spatial_postprocessing_parameters` (`temporal_postprocessing_parameters` for the
temporal slot), like a model's `custom_settings`, and the UI shows it below the
slot's selector (at most 2 sliders, 2 dropdowns and 1 text input per slot; a number
is a slider). The two slots never share a value, whatever their ids. Selecting
another processor resets the slot's values to its defaults, except the normalized
ids, which keep one meaning for every processor declaring them: `prompt`, a text
description of the content to process. Give an option its own id when its meaning
differs from a normalized one (the LTX Detail Refiner's style prompt is
`refine_prompt`, not H3's source description `prompt`). An `images` parameter
(`type: "array"`, `media_type: "image"`) reads the reference images input shared by
both slots, `postprocessing_reference_images`: the object image of the 3D refiner
and the texturers is given once. Defaults are owned by the descriptors. `ui` selects one or more UI
contexts: `postprocessing` means the normal generation-time Post Processing
section, `late_postprocessing` means the Post Processing tab for an existing
gallery item, and `media_flow` exposes scalar controls for the currently selected
Media Flow spatial process. A parameter can still be inferred and passed by
WanGP when it is absent from a UI context; H3, for example, receives generation
prompt/reference data without showing redundant controls during generation.

The UI shows the parameters of the two selected processors (the spatial one's,
then the temporal one's) in a fixed set of shared inputs, two per line: sliders,
dropdowns, numbers, a textbox and one images field, their counts set by the
constants of `postprocessing/parameter_ui.py`. Selecting a processor relabels
these inputs and fills them from the dictionary or from the defaults. A processor
that would need more inputs of one kind than the shared set holds is refused when
it is selected, so raise the matching constant together with it.

Use `label` for the short gallery label and optional `label_long` for a generation
label with brief guidance. Declare `description` and optional `method_descriptions`
on the processor so the UI can explain which choice suits the user's input.

Supported UI components are `textbox`, `number`, `slider`, `dropdown`, and
`images`. The images field is rendered by `AdvancedMediaGallery` and accepts an
ordered list (`multiple: True`). Deepy receives only the call-relevant fields (`name`, `type`,
`description`, `required`, `default`, `enum`, `minimum`, `maximum`, and
`media_type`), so UI/runtime fields such as `component`, `ui`, `label`, `step`,
and `setting` do not consume assistant context. Parameters with `media_type:
"image"` are resolved from media ids to paths. Dispatch reads the selected method's
parameters from `spatial_postprocessing_parameters` (an images parameter from the
shared reference images) and maps each to the `upscale()`
keyword named by `setting`.

Registration is owned by `postprocessing/spatial_upsamplers.py`. Add the handler class path
to `spatial_upsampler_handlers`:

```python
spatial_upsampler_handlers = [
    "postprocessing.my_upsampler.wgp_bridge.MyUpsampler",
]
```

`wgp.py` only calls `upsampler_api.register_spatial_upsamplers(server_config, fl)`;
it should not import or keep one explicit variable per spatial upsampler.

For a minimal external handler, see the
[Spatial Pixel Duplicate reference Plugin](https://github.com/deepbeepmeep/wan2gp-pixel-upsampler).
It exposes `pixel*1` through `pixel*4` and supplies the description displayed by
the Spatial Upsampling information button.

Enabled Plugins can expose upsamplers without editing core code by adding
`spatial_upsampler_handlers` to `plugin_info.json`. Entries that start with `.`
are relative to the Plugin package root; entries without a leading `.` are
absolute import paths:

```json
{
  "name": "My Plugin",
  "spatial_upsampler_handlers": [
    ".upsampler.MyUpsampler",
    "./nested/other_upsampler.py:OtherUpsampler",
    "postprocessing.my_upsampler.wgp_bridge.MySharedUpsampler"
  ]
}
```

Only enabled/loaded Plugins are considered. The Plugin manager reuses its cached
`plugin_info.json` metadata when registering those handlers, so the file is not
parsed a second time for upsampler discovery.

Upsampler settings are stored under `wgp_config["spatial_upsamplers"][config_key]`.
Handlers can read old top-level keys during migration with `legacy_config()`, but
those keys are deleted after the nested section is written.

RelayVSR stores `x2_model` (`x2`: RelayVSR X2 for ratios up to x2, or `x4`)
under `spatial_upsamplers.relayvsr`.

SeedVR2 stores `model` (`3b` or `7b`) and `window_size` under
`spatial_upsamplers.seedvr2`. For `window_size`, `0` selects the
GPU-based automatic limit, `-1` disables windowing, and positive values are
finite frame limits. SeedVR2 aligns finite limits down to its required `4n+1`
input shape and crossfades three overlapping output frames.

The LTX video handler exposes LTX 2.3 and LTX 2.5 as decoded-video x2 methods.
It reuses the existing checkpoint declarations and LTX family loader under
private runtime model types and always creates its MMGP offload object with
memory profile 5. Each source window is VAE encoded and appended as a
downscale-2 reference for the matching official Pixel Spatial Upscaler IC-LoRA;
the x2 target starts from noise, follows the official eight-step distilled sigma
schedule, and is VAE decoded. LTX 2.3 uses the Dev checkpoint with Distilled
1.1 at 0.5 and the x2 IC-LoRA at 1.0; LTX 2.5 uses its distilled checkpoint with
the official 2.5 x2 IC-LoRA at 1.0. Inputs longer than the configured window use
stride-aligned windows (81 frames with a 17-frame overlap by default);
overlapping windows address the same deterministic global noise and time
coordinates. Both versions are shown in Post Processing and Late Postprocessing,
and Media Flow keeps explicit processes for both versions. The Configuration
Plugin exposes only the shared LTX window size and overlap controls under
`spatial_upsamplers.ltx2`. Both values follow
the VAE's `8n+1` frame cadence; window size ranges from 9 to 481 frames, with 81
as the default. The values are read at the start of every native or Media Flow
upscale. Audio remains under the existing WGP and Media Flow preservation paths;
the LTX spatial upsampler itself only returns video frames.

**Face refiners** share `postprocessing/face_refiner`: `face.py` (detection, identity tracks, crops, the choice of
a reference, paste-back) and `bridge.py`, a `FaceRefinerBridge` to subclass, with `PrivateRuntime` and
`model_step_callback` for refiners running a model of their own. A new refiner declares its method, parameters and
files, and its `open_engine(job)` returns an object with `prepare(tracked_faces)`, `window_count(frames)`,
`refine_segment(segment)` (restored crops, uint8) and `close()`.
`canvas_tiers` in its configuration restricts the crops to the square sizes a model works at. A processor running
another one lists that one's files with `download_defs_for_value`. H3 Face Refiner (`h3facerefine`) and TaoMate-FVSR
Face Refiner (`taomatefvsr`) are the two examples.

The same handler also exposes **LTX 2.5 Detail Refiner** (`ltx25_refine`, x1,
x1.5 or x2) with the Refine Details IC-LoRA on the LTX 2.5 distilled checkpoint.
The source is Lanczos-resized to the target size, padded to multiples of 32 and
by eight frames, and refined from a canvas that is split into overlapping
1024x576 tiles (576x1024 for portrait) and 97-frame temporal windows. Every
denoising step evaluates each tile with its own crop of a freshly encoded
reference window and fuses the tile predictions with Gaussian weights
(`shared/utils/tiled_fusion.py`), so tiles share one trajectory and no seams form.
Gentle mode (default) starts from the encoded source at sigma 0.909 with three
steps; full mode starts from noise with the eight distilled steps. The
`refiner_window_size` (49 to 121 frames, default 97), `refiner_gentle` and
`refiner_tiles_per_call` (0 = automatic from free VRAM, or 1 to 16 tiles per
transformer call; the output does not depend on it) options live under
`spatial_upsamplers.ltx2`. The canvas, its blend buffer and the window
references move to system RAM when they would need more than 10 % of free VRAM.
In Media Flow each refiner chunk returns its refined overlap frames, re-encoded
as a short clip, through the continue cache; the next chunk starts from them at
half conditioning strength, and the process handler's `overlap_output_split()`
tells Media Flow to keep the previous chunk's frames in the overlap instead of
crossfading. The re-encoded last written overlap frames a resumed run needs are
kept in a `.ltx25_refine_cache.safetensors` sidecar; data whose start frame or
mode does not match the chunk is ignored. The x2
methods and the refiner share one private model; switching between them only
swaps LoRAs. Outputs are limited to a 4320p short side; a larger request fails
before denoising with the largest multiplier that fits. Its progress counts
tiles, text-encoder layers and VAE tiles with their units.

**Postprocessor VRAM Usage**, at the top of Configuration > Extensions, applies
to audio, temporal and spatial processors. **Minimum** is the default and uses
each processor's own minimum. **Manual Choice** shows a preload amount in MB;
the effective amount is the greater of that value and the processor's minimum.
**Dynamic** lets memory profile 0 adapt preload to available VRAM. Whole-model
requirements remain intact. These controls affect model preload, not a hard
limit on total VRAM consumed by processing.

The most recently used spatial processor stays loaded in RAM for reuse. Switching
to another processor releases it, as does changing the shared VRAM policy.
Processors that reuse the generation model retain that model's existing memory
configuration. Native processors without MMGP-managed weights keep their own
memory requirements.

Models declare external VAE upsampler support with method ids under
`model_def["vae_upsamplers"]`, for example:

```python
{
    "vae_upsamplers": {
        "flux_vae_pid": [1, 2],
        "flux2_vae_pid": {"image_modes": [1]},
    },
    "excluded_spatial_upsamplers": ["flux_pid"]
}
```

`excluded_spatial_upsamplers` hides post-processing methods that should not be
offered for that model, for example when a model supports the corresponding VAE
upsampler path and should not show both choices in the generation UI.

Optional attributes honored by `wgp.py`: `batch_image_inputs` (process image batches
in one call instead of per-frame). `uses_image_profile` is still accepted for older
handlers, but new handlers should declare `"profile"`.

## Extension offload object registry

Every extension that creates its own mmgp offload object (`offload.profile(...)`)
must register it in `shared/utils/offload_registry.py`:

```python
offload_registry.register_offloadobj("MyUpsampler", offloadobj, release_fn)
...
offload_registry.unregister_offloadobj("MyUpsampler", offloadobj)  # in release_fn
```

This lets WanGP track every extension offload object and release all extension
resources centrally: the toolbar "Unload Models" tool (and the Configuration Plugin
release button) calls `offload_registry.release_all()`.

---

> Applies to: Developing spatial upsampler/refiner Plugins: handlers, registration and shared configuration. Installed processing options are available through WanGP's post-processing controls and API.
