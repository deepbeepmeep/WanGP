# WanGP Settings

WanGP generation settings are JSON-serializable values consumed by `wgp.py` and by the Python API in `shared/api.py`.

Topics: [model selection](#model-selection), [prompts](#prompt-settings), [output dimensions and duration](#output-shape), [sampling](#core-sampling), [guidance](#guidance), [image and video inputs](#image-and-video-inputs), [audio inputs](#audio-inputs), [acceleration and caching](#acceleration-and-cache), [audio post-processing](#post-processing-audio), [advanced sampling](#advanced-sampling), [sliding windows](#sliding-window), [LoRAs](#loras), and [flag settings](#flag-settings).

The baseline schema lives in `models/_settings.json`. Model defaults in `defaults/*.json` and `finetunes/*.json` override those values, then handler code can update or hide settings according to the selected model definition. In practice, an exported settings file is the safest template for a specific model.

## Model Selection

| Setting | Type | Meaning |
| --- | --- | --- |
| `model_type` | string | Model definition id, for example `ltx2_22B_distilled`. Required for API tasks. |
| `model_mode` | string, number, or null | Optional model-specific mode selected from `model_def["model_modes"]`. Some image edit and TTS models use it for edit method, language, speaker, or scheduler variants. |
| `settings_version` | number | Saved settings schema version. WanGP uses this to migrate older saved settings in `fix_settings(...)`. |
| `client_id` | string | Optional API correlation id. Required only when an API caller wants returned in-memory artifacts for a specific submitted task. |

## Prompt Settings

| Setting | Type | Meaning |
| --- | --- | --- |
| `prompt` | string | Main text prompt, lyrics, speech text, or task instruction. |
| `negative_prompt` | string | Negative prompt when the model supports it. Ignored or removed for models with `no_negative_prompt`. |
| `alt_prompt` | string | Model-specific secondary prompt, such as music caption, emotion instruction, or alternate conditioning prompt. |
| `prompt_enhancer` | string flags | Prompt enhancer mode. Empty disables enhancement unless explicitly enabled elsewhere. See flag details below. |
| `multi_prompts_gen_type` | string flags | How multiline prompts are split into queue tasks or sliding-window prompts. See flag details below. |
| `multi_images_gen_type` | integer | For models that can treat multiple images as text prompts. `0` generally means one queue task per image; other values are model/UI dependent. |
| `custom_settings` | object or null | Model-specific settings keyed by custom setting id. WanGP supports the first six custom setting slots in the UI. |

## Output Shape

Click the small switch icon to the left of the resolution controls to alternate between familiar presets and a megapixel slider with an **Aspect Ratio** selector. One megapixel is 1,000,000 pixels. When all high resolutions are unlocked, the model header shows **Max Recommended Res.** for the selected phase. The **Resolution Budget (MP)** slider label previews the actual dimensions after alignment to the model's block size; changing views preserves the current size, including custom sizes. The current view is remembered with the next resolution save or model switch, independently of generation settings.

In the finetune editor, **Maximum Resolution**, **Maximum With Tiling** (when supported), and **Minimum Resolution** accept either a pixel count or dimensions such as `4096x4096`. These dimensions describe pixel area, so they do not force a square image. The maximum supplies the predefined presets; **Custom Resolutions** adds more choices. The minimum hides smaller presets and sets the ordinary slider minimum. Existing imported sizes remain available for unlocked models, with a warning during submission validation when outside the recommended range.

Enable **Lock Resolutions** to allow only that model's custom list. Locked models show the category picker without a mode switch, and API/Deepy requests must use an allowed size. Switching back to an unlocked model restores your preferred picker mode. An incompatible imported size requires an explicit choice; it is not silently replaced. Older exclusive preset lists retain their restriction when migrated, while old category expressions are expanded into additional presets. Saved dimensions are preserved.

Categories are approximate pixel-area groups. **4.2 MP** combines the former 1440p-area presets with 2048×2048, so square and wide formats share a useful group. **4K UHD** contains the familiar UHD-sized choices; **16.8 MP** covers larger presets including 4096×4096. Exact dimensions remain visible on each preset. Existing saved dimensions are preserved.

When no clear maximum is declared, WanGP uses a 1080p recommendation (1920x1088 to include VAE-aligned presets). Enable **3K/4K+ Resolutions are available for all models** to expose the full resolution catalog and allow sizes outside the recommended range in the UI. A nonblocking message warns about possible quality loss during submission validation, never when changing resolutions. **Lock Resolutions** still restricts a model to its custom list. Python API, Deepy and external agents receive a validation error when submitting sizes above the active maximum or below the declared minimum, independently of this UI option. Input fitting and optional upscaling can change the final media dimensions from the requested canvas.

A start image or reference image can set the canvas shape. With a start image, or when the first reference image is the main subject or landscape (`K` in `video_prompt_type`, shown as "Conditional Image is first Main Subject / Landscape"), WanGP keeps the requested resolution's pixel budget but follows that image's aspect ratio: a 1024x1024 request with a 1280x720 landscape reference produces 1360x768. When reference images are only people or objects (`I` alone), the requested resolution is kept, which suits a portrait poster built from a landscape reference.

| Setting | Type | Meaning |
| --- | --- | --- |
| `image_mode` | integer | `0` for video/audio-generation mode, `1` for image mode, `2` for image inpainting mode. The available tabs depend on `image_outputs`, `v2i_switch_supported`, and `inpaint_support` in the model definition. Deepy omits a fixed normal-image value from image-only model defaults, templates, and profiles; mode `2` remains explicit. |
| `resolution` | string | Requested canvas as `WIDTHxHEIGHT`, for example `1280x720`. Exact sizes can be requested independently of the picker mode; models with **Lock Resolutions** require a listed size. WanGP aligns each dimension down to the model's block size (at least one block); input fitting and postprocessing may change final dimensions. |
| `batch_size` | integer | Number of images for image-output models. For some special image models this may have model-specific meaning. Video and audio paths usually force one sample per repeat. |
| `video_length` | integer or seconds string | Requested output frames. Settings files, queue imports, and API/MCP requests may use a value such as `"10s"`; WanGP converts it using numeric `force_fps` when supplied, otherwise the model FPS, then selects the nearest frame count valid for the model. For audio-only models this is usually `0`; duration may come from `duration_seconds`. |
| `duration_seconds` | number | Requested duration for audio models or models exposing `duration_slider`. |
| `pause_seconds` | number | Pause inserted between multi-speaker or multi-sentence audio segments when supported. |
| `force_fps` | string | Output FPS policy. Empty uses the model default. Accepts `auto`, `control`, `source`, or a numeric string such as `24`. |
| `repeat_generation` | integer | Number of generated videos/audio files per prompt. Image models use `batch_size` instead. |
| `output_filename` | string | Optional output file name override. |

## Core Sampling

| Setting | Type | Meaning |
| --- | --- | --- |
| `seed` | integer | Random seed. `-1` means random. |
| `num_inference_steps` | integer | Denoising or generation step count when the model exposes inference steps. |
| `sample_solver` | string | Optional scheduler/solver selected from `model_def["sample_solvers"]`. |
| `flow_shift` | number | Flow matching shift scale when the model exposes `flow_shift`. |
| `temperature` | number | Sampling temperature for audio/TTS models that expose it. |
| `top_p` | number | Nucleus sampling limit for supported audio/TTS models. |
| `top_k` | integer | Top-k sampling limit for supported audio/TTS models. `0` usually disables top-k. |

## Guidance

| Setting | Type | Meaning |
| --- | --- | --- |
| `guidance_phases` | integer | Number of active CFG/guidance phases when a model supports phased guidance. |
| `model_switch_phase` | integer | Phase where a multi-submodel pipeline switches model, when applicable. |
| `switch_threshold` | integer | Phase 1 to 2 switch step/threshold, or model switch threshold for multi-submodel pipelines. |
| `switch_threshold2` | integer | Phase 2 to 3 switch step/threshold. |
| `guidance_scale` | number | CFG guidance for phase 1. |
| `guidance2_scale` | number | CFG guidance for phase 2. |
| `guidance3_scale` | number | CFG guidance for phase 3. |
| `audio_guidance_scale` | number | Audio guidance scale when `audio_guidance` is exposed by the model. |
| `embedded_guidance_scale` | number | Embedded guidance scale for models with embedded guidance. |
| `alt_guidance_scale` | number | Alternate-condition guidance scale for models exposing `alt_guidance`. |
| `alt_scale` | number | Alternate-condition strength for models exposing `alt_scale`. |
| `audio_scale` | number | Audio prompt/source strength when the model exposes `audio_scale_name`. |
| `control_net_weight` | number | Primary control weight for model families with control modules. |
| `control_net_weight2` | number | Secondary control weight when the model exposes two control weights. |
| `control_net_weight_alt` | number | Alternate control weight, for example Lynx or another named side condition. |

## Image And Video Inputs

| Setting | Type | Meaning |
| --- | --- | --- |
| `image_prompt_type` | string flags | Start/end/source continuation mode. See flag details below. |
| `image_start` | image or list | Start image(s) for image-to-video or image-conditioned generation. MCP accepts Gallery media IDs or authorized paths, including returned output paths. |
| `image_end` | image or list | End image(s) when the model supports end frames. MCP accepts Gallery media IDs or authorized paths; sliding-window anchors are an ordered list, one per window. |
| `image_refs` | image or list | Reference images or frames to inject, according to the model's declared mode; MCP accepts a list of Gallery media IDs or authorized paths. Use the complete model/template `video_prompt_type` choice, such as `KI` for references or `KFI` with `frames_positions` for frame injection. These are separate capabilities: H3 FL2VA supports frame injection, while H3 REF2VA supports general references. |
| `image_refs_relative_size` | integer | Relative internal size for reference images on models exposing `any_image_refs_relative_size`. |
| `remove_background_images_ref` | integer | Background-removal mode for reference images. Usually `0` off, `1` auto/on, with older values migrated by `fix_settings`. |
| `frames_positions` | string | Comma-separated frame positions: `1` = first frame, `L` = window end, `X` = skip a window without consuming an image. Repeat `X` to skip multiple windows. `L,L,X,L,L` uses four images at the ends of windows 1, 2, 4, and 5. |
| `image_guide` | image | Control image used in image mode. WanGP maps this into `video_guide` internally for generation. |
| `image_mask` | image | Image mask used in image inpainting/control modes. |
| `video_source` | video | Source video for continuation (`image_prompt_type` `V`) or post-processing edit tasks. |
| `keep_frames_video_source` | string | Truncation for source video after resampling. Empty keeps all; negative truncates from the end. |
| `input_video_strength` | number | Strength of the source/start input when the selected model exposes this setting. |
| `video_prompt_type` | string flags | Main control/reference/mask mode. See flag details below. |
| `video_guide` | video | Control video for video mode, or control image after internal mapping in image mode. |
| `video_guide2` | video | Second control or reference video. Used when `+` is present in `video_prompt_type`. |
| `video_guide3` | video | Third reference video. Used when `*` is present in `video_prompt_type`. |
| `keep_frames_video_guide` | string | Truncation for control video after resampling. Empty keeps all. |
| `denoising_strength` | number | Control strength for guide-based generation. Visible when `G` is present in `video_prompt_type`. |
| `masking_strength` | number | Masked-control duration/strength. Used when a mask is active or the model always enables mask strength. |
| `video_mask` | video | Video mask for video control/inpainting. |
| `mask_expand` | integer | Expands or shrinks mask area for mask-enabled guide modes. |
| `custom_guide` | file | Model-specific guide file, for example trajectory or camera data. |
| `video_guide_outpainting` | string | Four-part outpainting expansion percentages encoded as `top;bottom;left;right`, or `#` for disabled/default. |
| `video_guide_outpainting_ratio` | string | Preset target ratio for guide outpainting, for example `16:9`; empty means manual expansion. |
| `min_frames_if_references` | integer | Image-mode workaround for v2i-capable models: generate one or more frames to preserve reference identity. Values above `1000` mean always generate that many frames minus `1000`. |

## Audio Inputs

| Setting | Type | Meaning |
| --- | --- | --- |
| `audio_prompt_type` | string flags | Audio conditioning/source mode. See flag details below. |
| `audio_guide` | audio | Primary audio prompt, source, reference voice, or soundtrack depending on model. |
| `audio_guide2` | audio | Secondary audio reference for two-speaker, emotion, or timbre modes. |
| `audio_guide3` | audio | Third audio reference, for example a third voice. Used when `D` is present in `audio_prompt_type`. |
| `audio_source` | audio | External audio used by post-processing mode `custom`. |
| `speakers_locations` | string | Speaker ranges for multi-speaker audio/video models, for example `0:45 55:100`. |
| `replace_voice_sample` | audio | Primary reference voice for a voice-replacement audio processor. |
| `replace_voice_sample2` | audio | Secondary reference voice for two-speaker voice-replacement processors. |

## Acceleration And Cache

| Setting | Type | Meaning |
| --- | --- | --- |
| `skip_steps_cache_type` | string | Step-skipping cache type. Empty disables. Known values are `tea` and `mag` when the model exposes TeaCache or MagCache. |
| `skip_steps_multiplier` | number | Cache speed/skip multiplier. The meaning is model/cache dependent. |
| `skip_steps_start_step_perc` | integer | Percentage of denoising steps before cache skipping starts. |
| `temporal_upsampling` | string | Temporal post-processing method and multiplier encoded as `<method>*<multiplier>`, for example `rife*2` or `dlssg*4`. The former concatenated format remains accepted when loading older settings. |
| `spatial_upsampling` | string | Spatial post-processing method and multiplier encoded as `<method>*<multiplier>`, for example `vae*2`, `lanczos*2`, or `dlss5*1`. Empty disables; refiners without a multiplier retain their bare method id. The former concatenated format remains accepted when loading older settings. |
| `film_grain_intensity` | number | Film grain amount. `0` disables. |
| `film_grain_saturation` | number | Saturation of generated film grain. |
| `RIFLEx_setting` | integer | RIFLEx long-video policy. `0` auto, `1` always on, `2` always off. |
| `overrides` | string | Configuration choices replaced for this task, written as `name=value` pairs separated by commas, for example `attention_mode='sage2', memory_profile=4`. Empty uses the WanGP configuration for all of them. `memory_profile`: memory/offload profile `0`, `1`, `2`, `3`, `3.5`, `4`, `4.5` or `5` instead of the Video, Image or Audio profile of the configuration; it controls RAM/VRAM placement and speed only, is not a saved settings/accelerator profile and does not accept a `setting_id`. `compilation`: `'disabled'`, `'cudagraph'` or `'transformer'`. `attention_mode`: an attention backend such as `'sdpa'` or `'sage2'`, or a model-specific mode. `transformer_quantization`: `'int8'`, `'fp8'` or `'bf16'`; `text_encoder_quantization`: `'bf16'` or `'int8'`; when the model has no weights in that format, the closest available ones are used. A change of profile, compilation or quantization reloads the model. |

## Post-Processing Audio

| Setting | Type | Meaning |
| --- | --- | --- |
| `postprocess_audio` | string | Audio post-processing method. Empty disables. Built-ins include `custom`, `mmaudio`, `control`, `seedvc_one_speaker`, `seedvc_two_speakers`, and for audio edit tasks `remove_background`; Plugins may register more methods. |
| `postprocess_audio_prompt` | string | Positive prompt for audio processors that request a prompt. |
| `postprocess_audio_neg_prompt` | string | Negative prompt for audio processors that request a negative prompt. |
| `replace_voice_method` | string | Voice-replacement processor used after generated or remuxed audio. Empty disables. |

## Advanced Sampling

| Setting | Type | Meaning |
| --- | --- | --- |
| `perturbation_switch` | integer | Enables perturbation/skip-layer guidance mode when supported. Model definitions provide the displayed choices. |
| `perturbation_layers` | list[int] | Transformer layers affected by perturbation. |
| `perturbation_start_perc` | integer | Start percentage for perturbation. |
| `perturbation_end_perc` | integer | End percentage for perturbation. |
| `apg_switch` | integer | Adaptive Projected Guidance. Usually requires guidance greater than 1. |
| `cfg_star_switch` | integer | CFG Star switch for models exposing `cfg_star`. |
| `cfg_zero_step` | integer | CFG Zero layer cutoff. `-1` disables. |
| `NAG_scale` | number | Normalized Attention Guidance scale. |
| `NAG_tau` | number | NAG tau value. |
| `NAG_alpha` | number | NAG alpha value. |
| `self_refiner_setting` | integer | Self-refiner mode. `0` disables; other values select the normalization strategy. |
| `self_refiner_plan` | list or string | Self-refiner step/iteration rules. WanGP normalizes this through `normalize_self_refiner_plan`. |
| `self_refiner_f_uncertainty` | number | Self-refiner uncertainty threshold. |
| `self_refiner_certain_percentage` | number | Self-refiner certainty skip percentage. |
| `motion_amplitude` | number | Motion acceleration/intensity multiplier for models exposing motion amplitude. |

## Sliding Window

| Setting | Type | Meaning |
| --- | --- | --- |
| `sliding_window_size` | integer | Number of frames in each sliding-window generation chunk. |
| `sliding_window_overlap` | integer | Frames overlapped between consecutive windows. |
| `sliding_window_color_correction_strength` | number | Color matching strength between windows. |
| `sliding_window_overlap_noise` | integer | Noise injected in overlap frames to reduce blur. |
| `sliding_window_discard_last_frames` | integer | Discarded tail frames per window. |
| `sliding_window_trim_first_frames` | integer | Leading frames trimmed when a sliding window has no overlap frames. |
| `keep_intermediate_sliding_windows` | integer | Saved in server config, but exported settings may include it for traceability. Controls whether intermediate windows are retained. |

## LoRAs

| Setting | Type | Meaning |
| --- | --- | --- |
| `activated_loras` | list[string] | LoRA file names, URLs, or cached local entries. WanGP resolves them under the model's LoRA directory. |
| `loras_multipliers` | string | Per-LoRA multipliers. Space/newline separated; lines starting with `#` are ignored. Phase multipliers use semicolon syntax such as `0;1`. |

## Flag Settings

WanGP stores several UI selections as compact strings of letters. These are not free-form tags. Each model definition supplies allowed choices through keys such as `image_prompt_types_allowed`, `guide_preprocessing`, `mask_preprocessing`, `guide_custom_choices`, `image_ref_choices`, `audio_prompt_type_sources`, and `prompt_enhancer_def`.

### `image_prompt_type`

`image_prompt_type` describes the source/start/end media for the main generation path.

| Flag | Meaning |
| --- | --- |
| empty | Text-only/new generation. |
| `S` | Use `image_start` as starting image(s). |
| `E` | Use `image_end` as end image(s), if the selected source mode supports end frames. |
| `V` | Continue from `video_source`. The returned video contains the source video followed by the newly generated continuation; it is the merged result, not only the new continuation segment. |
| `L` | Continue from the last generated video. |
| `T` | Model definition sentinel for text/new-video availability. The stored value is usually empty rather than `T`. |

Only flags listed in `model_def["image_prompt_types_allowed"]` survive settings migration.

### `video_prompt_type`

`video_prompt_type` is the main control, mask, reference-image, and guide mode string. Model handlers build the visible UI choices, so not every flag is legal for every model.

| Flag | Meaning |
| --- | --- |
| `V` | A control video or control image is used (`video_guide` in video mode, `image_guide` in image mode). |
| `+` | A second control or reference video is used (`video_guide2`). |
| `*` | A third reference video is used (`video_guide3`), with `+`. |
| `-` | The guide videos are reference videos (appearance or motion references) rather than control videos, for example H3 Ref2VA `V-U` for one reference video, `V+-U` for two and `V+*-U` for three. H3 Ref2VA `V1-U` takes up to three excerpts from one reference video instead. With sliding windows, every window receives the whole reference video; without `-`, a control video is split so that each window receives its own section. |
| `\|` | Cap the video length at the control video's length (**Capped By: Control Length**). |
| `G` | Guide/denoise against the control media. Enables `denoising_strength`. |
| `U` | Keep/use the guide unchanged, identity/raw format, or bypass preprocessing depending on the model choice. |
| `A` | A mask is active. Uses `video_mask` in video mode or image mask data in image mode. |
| `N` | Non-masked-area variant when paired with mask choices such as `NA`, `XNA`, `YNA`, `WNA`, or `ZNA`. |
| `P` | Pose preprocessing. |
| `O` | Aligned pose preprocessing. |
| `D` | Depth preprocessing. |
| `E` | Canny edge preprocessing. |
| `S` | Scribble/shapes preprocessing. |
| `L` | Optical flow preprocessing. |
| `C` | Grayscale/recolor preprocessing. |
| `M` | Inpaint-mask preprocessing. |
| `B` | Face movement/control preprocessing. |
| `H` | Bounding-box preprocessing. |
| `X` | Inpaint outside the selected mask area. |
| `Y` | Depth outside the selected mask area. |
| `W` | Shapes/scribble outside the selected mask area. |
| `Z` | Flow outside the selected mask area. |
| `I` | Reference image(s) are used through `image_refs`. |
| `K` | Landscape, background, main subject, or first reference image role, depending on the model's `image_ref_choices`. |
| `F` | Positioned-frame reference images. Uses `frames_positions`. |
| `J` | Style image role used by Flux USO-style definitions. |
| `T` | Align control media/audio/positioned frames to the first generated sliding-window sample instead of the beginning of the source video. |
| `Q` | Hidden/model-forced special option used by specific handlers. Do not set manually unless the model definition selects it. |
| `&` | LTX-2 HDR-output option in the IC-LoRA control choices. |
| `#` | Shows the model's extra custom option when present in the selected choice, for example Wan Animate's **Apply Relighting** checkbox (value `1`) on its "Replace Person" choices. SCAIL instead writes the number of people between two `#`, for example `V#2#` for two people. |
| `0`–`9` | Model-specific options, for example InfiniteTalk's `0` to start a new shot. |

The same letter can have slightly different labels in different handlers. For example, `V` in `video_prompt_type` means "Control Video" in video mode and "Control Image" in image mode. Do not confuse it with `V` in `image_prompt_type`: there it selects video continuation, and WanGP returns the source video merged with the generated continuation. Always prefer a value from the selected model's exposed choices instead of composing a string by hand.

### `audio_prompt_type`

`audio_prompt_type` combines model-specific source choices with shared audio option flags.

The source part is defined by `model_def["audio_prompt_type_sources"]`. Common patterns are:

| Flag | Common meaning |
| --- | --- |
| empty | Text-only audio/video generation, or no audio source. |
| `A` | Primary audio source, reference voice, soundtrack, or source audio. |
| `B` | Secondary audio reference, second speaker, emotion reference, or timbre reference. |
| `D` | Third audio reference (`audio_guide3`), with `A` and `B`, for example `ABD` for three voices in H3 and in multi-speaker TTS models (Qwen3 Base, OmniVoice, KugelAudio, DramaBox, Scenema `ABD2`, IndexTTS2 `ABD2`). |
| `X` | Auto-separate two speakers from one primary audio source. |
| `C` | Treat two audio sources as consecutive, played in a row. Stable Audio 3 uses it, with `A`, to continue the supplied audio. |
| `E` | Stable Audio 3: edit the supplied audio (audio-to-audio), with `A`. |
| `I` | Stable Audio 3: inpaint part of the supplied audio, with `A`. |
| `P` | Treat two audio sources as parallel speaker/audio tracks. |
| `K` | Use the control video's audio track as the audio prompt. Requires a control video mode. |
| `O` | Output/generated-audio selector for models that otherwise reuse the input audio as the soundtrack. |
| `S` | Keep the input audio (from `A` or `K`) as the output soundtrack: the video follows it and WanGP puts the original audio in the final file, in sync across sliding windows. Needed only where the audio would otherwise be imitated as a reference, for example H3 Ref2VA `AS` and `KS`; `O` still keeps the generated audio. |
| `F` | Use the full audio guide instead of per-window audio slices. |
| `0`, `1`, `2` | Model-specific source modes. For example, LTX-2 uses `1` inside `A1OF` for an ID-LoRA reference voice path and `2` for audio generation from control video plus text; DramaBox uses `0` to remove unexpected words. |
| `T` | Recorded in saved media settings when a post-processing soundtrack used a separate audio source. Not a generation choice. |

Shared option flags can be appended:

| Flag | Meaning |
| --- | --- |
| `V` | Remove or ignore background music/noise in every audio source (`audio_guide`, `audio_guide2`, `audio_guide3`), keeping the voices. |
| `L` | Allow video length to continue beyond the audio end when the model supports it. |
| `N` | Bring two or three audio sources (`A`, `B`, `D`) to the same loudness. With `V`, the voices are isolated first, so their levels are matched. |
| custom | A one-letter model-defined option from `audio_prompt_type_custom_option`, when present. |

### `prompt_enhancer`

`prompt_enhancer` selects what the prompt enhancer receives and which model-specific instruction set it uses.

| Flag | Meaning |
| --- | --- |
| empty | Disabled. |
| `T` | Use the existing text prompt. |
| `I` | Include the start/source image when the selected model supports image-aware enhancement. |
| `K` | Enable "Think" mode for supported prompt enhancer backends. |
| `1` | Use the model's alternate enhancer instruction set, often a relayed/dialogue prompt mode. |

Examples: `T`, `TI`, `T1`, `TI1`, and MOSS `T1` (`T1K` with Think). Older saved values `I` and `IK` are migrated to `TI` and `TIK`.

Enhancer system instructions can insert generation values with double-brace placeholders. For example, `Compose for {{width}}x{{height}} pixels` becomes `Compose for 1536x896 pixels` at that selected resolution. This works with manual and automatic enhancement, local and external enhancers, and each instruction set in a chain. No extra flag is needed: values enter the instructions only where a placeholder requests them. The user's prompt and ordinary JSON braces are unchanged.

| Placeholder | Value |
| --- | --- |
| `{{width}}`, `{{height}}`, `{{resolution}}` | Selected canvas dimensions in pixels, or `WIDTHxHEIGHT`. These precede reference-image fitting and postprocessing. Available for images and videos. |
| `{{aspect_ratio}}` | Selected canvas ratio, such as `12:7`. |
| `{{half_width}}`, `{{half_height}}` | Half the selected dimensions, useful for layouts with a centered pixel origin. |
| `{{duration_seconds}}` | Video duration, or the selected positive audio duration. For video windows, each prompt receives its own window duration. Unavailable for still images or automatic audio duration. |
| `{{fps}}`, `{{frame_count}}` | Video frame rate and number of frames; for video windows, the frame count belongs to that window. |

Templates keep the model's coordinate convention: MOSS uses centered pixels, while models with normalized layouts retain their normalized units. Unknown or unavailable placeholders produce an error rather than guessed values. To include a literal placeholder in system instructions, escape it as `\{{width}}`.

Automatic duration guidance keeps its existing behavior unless the model definition sets `prompt_enhancer_video_duration` to `false`. This is a model declaration, not a generation setting. Explicit `false` disables the generic duration sentence for both whole videos and individual windows; `true` enables it, and omitting the declaration preserves the existing workflow. A template can still request `{{duration_seconds}}` when automatic insertion is disabled. When the selected system instructions already use that placeholder, WanGP omits the generic sentence to avoid duplication. An escaped literal placeholder does not suppress it. Start/end image roles, reference labels, injected-frame positions and timestamps remain available in either case.

### `multi_prompts_gen_type`

| Value | Meaning |
| --- | --- |
| `G` | Each non-empty line creates a new generation queue task. |
| `PG` | Each paragraph separated by an empty line creates a new generation queue task. |
| `W` | Each line becomes a prompt for a sliding window of the same video. |
| `PW` | Each paragraph becomes a prompt for a sliding window of the same video. |
| `FG` | Full prompt. All lines belong to one prompt. |

If the selected model does not support sliding windows, `W` modes are migrated to queue-task modes.

## API Metadata About Models

This appendix describes internal Python/model-definition records, not the shape of every Deepy tool response. Compact model capabilities expose supported `capabilities` and `media_inputs` roles as lists and omit `setting_values`; use the returned contract and the relevant model-definition property for exact choices. The generation setting names documented above are shared across these presentations.

Each in-memory model definition now includes a `metadata` object inferred after handler initialization:

```json
{
  "metadata": {
    "model_type": "ltx2_22B_distilled",
    "family": "ltx2",
    "family_label": "LTX-2",
    "base_model_type": "ltx2_22B",
    "finetune": false,
    "main_output": ["video"],
    "outputs": ["video", "audio"],
    "inputs": ["text", "image", "video", "audio"],
    "media_inputs": {
      "image": {
        "start": true,
        "end": true,
        "reference": true,
        "single_reference": false,
        "multiple_references": true,
        "background": false,
        "injected_frames": false,
        "control": false,
        "mask": false
      },
      "video": {
        "continue": true,
        "last": false,
        "control": true,
        "mask": false
      },
      "audio": {
        "prompt": true,
        "output": true
      }
    },
    "capabilities": {
      "text_to_video": true,
      "image_to_video": true,
      "audio_output": true
    },
    "setting_values": {
      "image_prompt_type": {"allowed": "TVSE"},
      "video_prompt_type": {"image_ref_choices": {"choices": []}}
    }
  }
}
```

Inference rules:

- `finetune` is true when the model definition comes from `finetunes/`.
- `main_output` is `["audio"]` for `audio_only` models.
- `main_output` includes `image` for `image_outputs` models.
- `main_output` includes both `image` and `video` for models with `v2i_switch_supported` or `inpaint_support`; this includes Vace image generation.
- otherwise `main_output` defaults to `["video"]`.
- `outputs` starts from `main_output`, then adds `audio` for models with intrinsic audio output or audio-track generation (`returns_audio`), such as LTX-2, OVI, InfiniteTalk, Fantasy, LongCat Avatar, Hunyuan Avatar/Custom Audio, Magi Human, and Cosmos3. It does not include audio created later by `postprocess_audio`.
- `inputs` always includes `text`.
- `inputs` includes `audio` when `any_audio_prompt` is true.
- `inputs` includes `image` when start images, image references, alternate guide references, or inpainting are supported.
- `inputs` includes `video` when source-video or control-video choices are supported.
- `media_inputs.image` describes optional image attachment roles: start image, end image, one or more reference images, background image (`K` choices), injected frames (`F` choices), control image, and mask image.
- `media_inputs.video` describes optional video roles: continuation/source video, last generated video continuation, control video, and mask video.
- `media_inputs.audio.prompt` describes audio input support. It is not audio output; output audio is indicated by `outputs` containing `audio` or `capabilities.audio_output`.
- `capabilities` provides common text/image/video/audio workflow booleans derived from the model definition.
- `setting_values` exposes normalized allowed values for agent-facing settings, especially letter-flag settings such as `image_prompt_type`, `video_prompt_type`, `audio_prompt_type`, `prompt_enhancer`, and model-specific selectors such as `model_mode`.
- For models whose outputs include video, compact model-schema metadata exposes `frames_maximum` as the suggested maximum for one generation window, including the first window of models without sliding-window support. WanGP first calls the model handler's `update_default_settings(...)` with an empty dictionary and uses its `sliding_window_size`; if absent, it uses `model_def["sliding_window_defaults"]["window_default"]`; if that is also absent, it selects the model-valid frame count closest to 97 using `frames_minimum`, `frames_steps`, and `frames_offset`. It is not the total-frame UI limit. Model definitions may use `frames_selection_maximum` to cap the total `video_length` selector. Non-video model schemas omit `frames_maximum`.

---

> Applies to: Generation setting meanings: image_mode, image_start/image_end, image_refs, video duration, audio inputs, flags, sampling, LoRAs and sliding windows. The API metadata appendix describes model declarations rather than generation parameters.
