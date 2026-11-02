# Audio Processor Plugin API

Audio processors (custom soundtrack remuxing, MMAudio, PrismAudio, SeedVC,
background removal, ...) are registered in `postprocessing/audio_processors.py`.
Each processor declares one or more methods and their capabilities through a
handler object; the registry owns method metadata, dropdown state, validation
helpers, config nesting, downloads, and dispatch.

## Processor Types

Audio processors are used in three stages:

- `soundtrack`: replace or generate the soundtrack for a video.
- `voice_replacement`: replace voice tracks in a video remux flow.
- `audio_edit`: process a standalone audio file in late audio postprocessing.

One method can support multiple stages. SeedVC, for example, supports both
`voice_replacement` and `audio_edit`.

## Handler Contract

```python
class MyAudioProcessor:
    def query_audio_processor_def(self):
        return {
            "name": "MyAudioProcessor",
            "processor_types": ("soundtrack",),             # coarse handler capability
            "methods": [("MyAudioProcessor", "myaudio")],
            "method_types": {"myaudio": ("soundtrack",)},   # per-method stage capability
            "needs_prompt": {"myaudio": True},
            "needs_negative_prompt": {"myaudio": False},
            "needs_audio_source": {"myaudio": False},
            "needs_voice_sample": {"myaudio": False},
            "needs_voice_sample2": {"myaudio": False},
            "supports_repeat": {"myaudio": True},
            "speaker_count": {"myaudio": 0},
            "status": {"myaudio": "Generating Audio"},
            "method_context_labels": {
                "late_postprocessing": {"myaudio": "Generate audio (MyAudioProcessor)"},
            },
            "config_key": "myaudio",                        # wgp_config["audio_processors"] subkey
            "pos": 100,                                     # default dropdown order for this handler's methods
            "method_pos": {"myaudio": 100},                 # optional per-method order
            "description": "Generate a soundtrack for video.", # optional discovery fallback
            "method_descriptions": {"myaudio": "..."},     # optional per-method descriptions
            "method_parameters": {"myaudio": [...]},       # optional extra parameter descriptors
        }

    def validate_method(self, method, **kwargs): ...        # -> "" or error text
    def download(self, method, process_files, send_cmd=None, status_text=None, **kwargs): ...
    def query_download_defs(self, *, enabled_only=True): ... # -> list of process_files definitions
    def enabled(self): ...                                  # optional UI gating
    @property
    def status(self): ...                                   # optional: "enabled" or "disabled"
    @property
    def reason_disabled(self): ...                          # optional user-facing reason
    def release_vram(self): ...                             # optional Configuration-tab release hook
    # soundtrack type:
    def generate_soundtrack(self, method, video_path, prompt="", negative_prompt="", seed=-1, duration=0, output_path=None, send_cmd=None, status_callback=None, **kwargs): ...
    # voice_replacement type:
    def replace_voice_tracks(self, method, audio_tracks, voice_sample=None, voice_sample2=None, output_dir="", prefix="", process_files=None, **kwargs): ...
    # audio_edit type:
    def process_audio_file(self, method, audio_source, output_path=None, status_callback=None, **kwargs): ...
    # optional Configuration tab integration:
    def default_config(self): ...                           # -> dict
    def normalize_config_section(self, section): ...        # -> normalized dict
    def create_config_ui(self, gr, section, *, lock_config=False): ...
    def validate_config_section(self, section): ...         # -> "" or message/list
    def config_requires_release(self, old, new, changed_keys): ...
```

The handler instance's `status` availability property is distinct from the
existing `query_audio_processor_def()["status"]` mapping, which contains
per-method progress text. Discovery evaluates the historical `enabled()` method
first: `True` maps to `enabled` and `False` to `disabled`. Only handlers without
`enabled()` use the optional instance `status` property; `unknown` means neither
mechanism supplied a valid status.
`reason_disabled` is returned only for disabled handlers. The internal processor
catalog includes this availability metadata. Prime's post-processing toolbox
omits disabled processors from callable discovery; internal/compatibility views
may include them for diagnostics, but disabled processors cannot be dispatched.

The supported processor type constants are:

```python
AUDIO_PROCESSOR_TYPE_SOUNDTRACK = "soundtrack"
AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT = "voice_replacement"
AUDIO_PROCESSOR_TYPE_AUDIO_EDIT = "audio_edit"
```

`method_types` is the authoritative per-method stage declaration. Use it even
when a handler exposes only one method; it lets a future handler expose mixed
methods without changing registry behavior.

`method_metadata(method)` resolves the UI/validation metadata from
`query_audio_processor_def()`. The common metadata flags are:

- `needs_prompt`: show/pass `postprocess_audio_prompt`.
- `needs_negative_prompt`: show/pass `postprocess_audio_neg_prompt`.
- `needs_audio_source`: show/pass `audio_source`.
- `needs_voice_sample`: show/pass `replace_voice_sample`.
- `needs_voice_sample2`: show/pass `replace_voice_sample2`.
- `supports_repeat`: show/pass `repeat_generation`.
- `speaker_count`: number of speakers expected by a voice replacement method.

`method_context_labels` lets a method show a different label in a specific UI
context. The current late-postprocessing context key is `late_postprocessing`.

Discovery consumers infer prompt, audio-source, and voice-sample parameters
from the existing `needs_*` flags. Optional `description`,
`method_descriptions`, and `method_parameters` fields add reusable presentation
and parameter metadata. Each `method_parameters` entry is a list of dictionaries
with at least `name`; it may also define `type`, `description`, `required`,
`default`, `enum`, and a queue-setting override named `setting`. These fields
are optional so older and third-party handlers remain compatible.

`control` is an internal pseudo-method used by the generation UI to reuse a
control video audio track. Do not register a Plugin method with that id.

## Dispatch Contract

WanGP dispatches audio processors by declared stage:

- `generate_soundtrack(...)` returns a path to an audio file to mux into the video.
- `replace_voice_tracks(...)` receives a list of source audio track paths and
  returns `(new_audio_tracks, temp_audio_tracks)`.
- `process_audio_file(...)` returns the path to the edited standalone audio file.

`validate_method(...)` is called before dispatch with the context available at
that stage. Common keyword arguments include:

- `video_source`
- `audio_source`
- `voice_sample`
- `voice_sample2`
- `frames_count`
- `fps`
- `duration`
- `media_source_exists`
- `has_audio_file_extension`

Return an empty string for success or a user-facing validation error.

## Registration

Registration is owned by `postprocessing/audio_processors.py`. Add the handler
class path to `audio_processor_handlers`:

```python
audio_processor_handlers = [
    "postprocessing.my_audio.audio_processor.MyAudioProcessor",
]
```

`wgp.py` only calls
`audio_processor_api.register_audio_processors(server_config, fl)`; it should not
import or keep one explicit variable per audio processor.

Enabled Plugins can expose audio processors without editing core code by adding
`audio_processors` to `plugin_info.json`. Entries that start with `.` are relative
to the Plugin package root; entries without a leading `.` are absolute import
paths:

```json
{
  "name": "My Plugin",
  "audio_processors": [
    ".audio.MyAudioProcessor",
    "./nested/other_audio.py:OtherAudioProcessor",
    "postprocessing.my_audio.audio_processor.MySharedAudioProcessor"
  ]
}
```

Only enabled/loaded Plugins are considered. The Plugin manager reuses cached
`plugin_info.json` metadata when registering those handlers.

## Configuration

Audio processor settings are stored under
`wgp_config["audio_processors"][config_key]`.

**Postprocessor VRAM Usage**, at the top of Configuration > Extensions, is shared
with temporal and spatial processors. **Minimum** (default) uses each processor's
minimum; **Manual Choice** uses the greater of that minimum and the selected MB
amount; **Dynamic** adapts profile 0 preload to available VRAM. This controls
preloaded weights rather than imposing a limit on total VRAM use.

The most recently used audio processor stays loaded in RAM for reuse and is
released when a different processor runs or the shared VRAM policy changes.
Processors without MMGP-managed weights retain their own memory requirements.

An extension can hide alternative methods from normal menus while keeping them
available. Extracting settings restores the exact saved method, including a
hidden one. Selecting another method hides the restored alternative again.

Handlers with configuration should implement `default_config()` and
`normalize_config_section(...)`. If they expose Configuration-tab controls, return
`[(field_name, gradio_component), ...]` from `create_config_ui(...)`; WanGP uses
that binding to collect, normalize, validate, and save the nested section.

If a config change requires releasing loaded runtime state, implement either
`release_vram()` or `config_requires_release(old, new, changed_keys)`.

## API Settings

The public API and queue settings use the same processor fields as the UI:

- `postprocess_audio`: registered method id.
- `postprocess_audio_prompt`: positive prompt for methods with `needs_prompt`.
- `postprocess_audio_neg_prompt`: negative prompt for methods with
  `needs_negative_prompt`.
- `audio_source`: custom soundtrack/source audio when `needs_audio_source`.
- `replace_voice_sample`: first voice sample when `needs_voice_sample`.
- `replace_voice_sample2`: second voice sample when `needs_voice_sample2`.
- `repeat_generation`: repeat count when `supports_repeat`.

Legacy `seedvc` and `seedvc2` method values are normalized to
`seedvc_one_speaker` and `seedvc_two_speakers`. New callers and new docs should
use the normalized method ids.

## Built-In Methods

| Method | Types | Handler |
| --- | --- | --- |
| `custom` | `soundtrack` | `postprocessing/custom_soundtrack/audio_processor.py` |
| `mmaudio` | `soundtrack` | `postprocessing/mmaudio/audio_processor.py` |
| `prismaudio` | `soundtrack` | `postprocessing/prismaudio/audio_processor.py` |
| `seedvc_one_speaker` | `voice_replacement`, `audio_edit` | `postprocessing/seedvc_processor/audio_processor.py` |
| `seedvc_two_speakers` | `voice_replacement`, `audio_edit` | `postprocessing/seedvc_processor/audio_processor.py` |
| `remove_background` | `audio_edit` | `postprocessing/audio_background_removal/audio_processor.py` |
| `remove_vocals` | `audio_edit` | `postprocessing/audio_background_removal/audio_processor.py` |

## Extension Offload Object Registry

Every extension that creates its own mmgp offload object (`offload.profile(...)`)
must register it in `shared/utils/offload_registry.py`:

```python
offload_registry.register_offloadobj("MyAudioProcessor", offloadobj, release_fn)
...
offload_registry.unregister_offloadobj("MyAudioProcessor", offloadobj)  # in release_fn
```

This lets WanGP track every extension offload object and release all extension
resources centrally: the toolbar "Unload Models" tool and Configuration Plugin
release button call `offload_registry.release_all()`.

---

> Applies to: Developing audio processor Plugins: handler methods, registration and shared configuration. Installed processing options are available through WanGP's post-processing controls and API.
