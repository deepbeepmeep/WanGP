# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""WanGP family handler for MiniMax H3."""

import os

import gradio as gr
import torch

from shared.utils.download import process_files_def_if_needed
from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url
from shared.utils.frame_scheduler import normalize_overlap
from shared.utils.generation_overrides import get_override, set_override

from .excerpts import H3_AUDIO_EXCERPTS_SETTING, H3_EXCERPT_SETTINGS, H3_VIDEO_EXCERPTS_SETTING, parse_excerpts, reference_video_frame_limit
from .constants import (H3_AUDIO_REFINEMENT_SETTING, H3_CONTROL_LATENT_CONTINUATION, H3_MASK_MODE_DEFAULT, H3_MASK_MODE_GROUPED_ROWS,
                        H3_MASK_MODE_SHARED_TIMESTEP, H3_MASK_MODE_SETTING, H3_PHASE_2_NOISE_LEVEL_START_DEFAULT,
                        h3_grouped_masking_enabled)
from .dialogue import H3_DIALOGUE_GENERATION, H3_DIALOGUE_MAX_TOTAL_SECONDS, H3_DIALOGUE_PROMPT_INFOS, load_dialogue_whisper
from .minimax_h3_main import (AUDIO_VAE_FILE, LATENT_UPSCALER_FILE, LATENT_UPSCALER_FOLDER, TEXT_ENCODER_FOLDER,
                              VIDEO_VAE_FILE, VIDEO_VAE_FP8MIX_FILE, VIDEO_VAE_INT8_FILE, VIDEO_VAE_LIGHT_FILE, VIDEO_VAE_LIGHT_INT8_FILE)
from .pdd import PDD_BLOCK_SIZE, PDD_NUM_STEPS
from .pika import PIKA_SOLVER, PIKA_STEPS, PIKA_VIDEO_SHIFT
from .ptd import PTD_FOLDER, PTD_HEAD_FILE, PTD_INFOS, PTD_DEEPY_INFOS
from .vae_upsampler import X2_VAE_DESCRIPTION, X2_VAE_FILE, X2_VAE_INT8_FILE, X2_VAE_METHOD, X2_VAE_VALUE, query_x2_vae_files
from .viggle import VIGGLE_ARCHITECTURE, VIGGLE_ASSET_FOLDER, VIGGLE_INFOS, VIGGLE_PROMPT_FILE, VIGGLE_REPO_ID
from .prompt_enhancer import (FL2VA_DEEPY_PROMPT_INFOS, FL2VA_IMAGE_SYSTEM_PROMPT, FL2VA_PROMPT_INFOS, FL2VA_TEXT_SYSTEM_PROMPT,
                              H3_AUDIO_DEEPY_PROMPT_INFOS, H3_AUDIO_DIALOGUE_SYSTEM_PROMPT, H3_AUDIO_MONOLOGUE_SYSTEM_PROMPT,
                              H3_STILL_IMAGE_SYSTEM_PROMPT, H3_STILL_TEXT_SYSTEM_PROMPT,
                              REF2VA_DEEPY_PROMPT_INFOS, REF2VA_IMAGE_SYSTEM_PROMPT, REF2VA_PROMPT_INFOS, REF2VA_TEXT_SYSTEM_PROMPT)


REPO_ID = "DeepBeepMeep/MiniMax-H3"
TEXT_ENCODER_BF16 = "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors"
TEXT_ENCODER_INT8 = "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"
TEXT_ENCODER_GGUF_Q2 = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
TEXT_ENCODER_GGUF_Q4 = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
TEXT_ENCODER_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
TURBO_LORA_FILE = "minimax_h3_lightx2v_fl2v_turbo_4step_alpha16_v0.1.safetensors"
TURBO_LORA_KEY = "minimax_h3_lora_turbo"
REF_TURBO_LORA_FILE = "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16.safetensors"
REF_TURBO_LORA_KEY = "minimax_h3_ref_lora_turbo"
TEXT_ENCODER_VARIANTS = {
    "gguf_q2_k": [TEXT_ENCODER_GGUF_Q2],
    "gguf_q4_k_m": [TEXT_ENCODER_GGUF_Q4],
    "nvfp4_awq": [TEXT_ENCODER_NVFP4],
}
FL2VA_ARCHITECTURE = "minimax_h3_fl2va"
FL2VA_PRUNED_ARCHITECTURE = "minimax_h3_fl2va_pruned"
REF2VA_ARCHITECTURE = "minimax_h3_ref2va"
REF2VA_PRUNED_ARCHITECTURE = "minimax_h3_ref2va_pruned"
TTS_REF2VA_PRUNED_ARCHITECTURE = "minimax_h3_tts_ref2va_pruned"
CONTROL_ARCHITECTURE = "minimax_h3_control"
CONTROL_PRUNED_ARCHITECTURE = "minimax_h3_control_pruned"
CONTROL_ARCHITECTURES = (CONTROL_ARCHITECTURE, CONTROL_PRUNED_ARCHITECTURE)
FIRST_BLOCK_CACHE_THRESHOLDS = (0.06, 0.08, 0.10, 0.12, 0.14)
LEGACY_FIRST_BLOCK_CACHE_THRESHOLDS = {1.5: 0.06, 1.75: 0.08, 2.0: 0.10, 2.25: 0.12, 2.5: 0.14}
FIRST_BLOCK_CACHE_STRENGTHS = [
    ("Low (0.06)", 0.06),
    ("Balanced (0.08, upstream default)", 0.08),
    ("High (0.10)", 0.10),
    ("Very High (0.12)", 0.12),
    ("Maximum (0.14)", 0.14),
]

FL2VA_DEEPY_INFOS = """Generate video and stereo sound from `prompt`. `image_start` / `image_end` anchor the opening / ending; together they constrain the transition. `video_source` continues an existing video; sliding windows carry overlapping video and audio forward.

Control Video (`video_guide`) is split across sliding windows, each window receiving its own section. `GV` (Denoise Control Video) edits its frames: lower Denoising Strength preserves more source content; Whole Frame at strength 1 gives full freedom. A mask selects the edited area. `VU` (In-Context Guide) supplies its unchanged frames as `<Video 1>` for LoRAs trained on FL2VA to transform a source video; name it `<Video 1>` in the prompt. Without such a LoRA, FL2VA is not trained to follow `VU`; for motion/appearance references, use Ref2VA Reference Video. Inject Frames uses ordered `image_refs` and explicit positions (`1` = first frame, `L` = last frame of the window, `X` = skip a window without consuming an image). A frame on a window's first or last frame acts as its start or end image; frames inside the overlap are unused. Per window, pictures are numbered: start or continued frame (`<Picture 1>`), end image, then that window's injected frames by time.

`audio_prompt_type`: empty = generate video and audio; `A` = `audio_guide` is the soundtrack; `K` = the control video's audio track is the soundtrack (`GV` or `VU`); `2` = keep control frames and generate their audio (`GV`). The video follows a soundtrack and WanGP keeps it as the output audio when it covers the video; a shorter one permits generated sound afterward. For visible speech in soundtrack mode, identify the character and explicitly request `speaking with natural lip movements synchronized` or equivalent lip-sync wording, without mentioning a supplied audio track or its language. Do not write the spoken words (no `<d>` dialogue): given a written line, H3 voices its own dialogue and the mouth often stops following the soundtrack. Keep the speaker's action to speaking: smiles, gestures and nods compete with the mouth. Exact lip sync is not guaranteed.

Use `capabilities` for window limits. WanGP rounds overlap to compatible values (1, 18, 35, ...); `video_length` sets total duration across windows. A window may jump onto an end image its action did not reach; `sliding_window_discard_last_frames` about 5 hides it: the end image of every window but the last is generated after the discarded frames (steers, not shown); the last window ends on its end image. Read `prompt_infos` for H3's structured prompt syntax.
"""

FL2VA_INFOS = """## FL2VA — First/Last Frame to Video and Audio

FL2VA creates a video with stereo sound from your text prompt. You can optionally provide a start image, an end image, a control video, injected frames, or a soundtrack.

### Start and end images

- **No image:** generate the video and audio from the text prompt.
- **Start image only:** begin the video with that image.
- **End image only:** finish the video with that image.
- **Start and end images:** guide both ends of the video.

Start and end images are placed at those exact points in the video. For general character, object, or style references, use Ref2VA instead.

### Control Video / Frames Injection

- **Generate without using a Control Video:** generate normally from the prompt and any start/end images.
- **Use Control Video as In-Context Guide:** H3 receives the control video's unchanged frames as `<Video 1>` and generates a new video from them. Choose this mode with LoRAs trained on FL2VA to transform a source video: name the video `<Video 1>` in the prompt and say what to change and what to keep. FL2VA is not trained to follow this input on its own; without such a LoRA, use Ref2VA with **Reference Video**.
- **Denoise Control Video (Video-to-Video Edit):** use an uploaded video to guide the result. Lower **Denoising Strength** values keep the result closer to the control video; `1.0` gives the model full freedom. At `1.0` with **Whole Frame** selected, the control video does not affect the result, so WanGP skips that work. Choose **Masked Area** or **Non Masked Area** to limit editing to part of the frame. **Masking Strength** controls how strongly the rest of the frame stays close to the control video. Use a lower masking strength (<0.75) to facilitate continuity with masked areas.
- **Inject Frames:** add images at specific points in the generated video. Add the images under **Reference Images**, then enter one position per image in the same order. Position `1` means the first frame; `L` means the last frame of a sliding-window segment. Each `X` skips a window without consuming an image. An injected frame on a window's first or last frame acts as that window's start or end image; one inside the sliding-window overlap is not used, because the overlap already continues the previous window.

Both control video modes are split across sliding windows: each window receives only its own section of the control video. For motion or appearance transfer from a video reference, use Ref2VA with **Reference Video**.

### Audio Source

- **Generate Video and Audio from Text Prompt:** let H3 create both.
- **Generate Video based on Soundtrack and Text Prompt (Soundtrack Kept):** upload a soundtrack and H3 will create the video around it. If the soundtrack covers the whole video, WanGP uses the original audio in the final file. If it ends early, H3 generates the remaining sound.
- **Generate Video based on Control Video + its Audio Track and Text Prompt (Soundtrack Kept):** select one of the control video modes. H3 uses both its frames and its existing audio track, which is kept as the soundtrack like an uploaded one; with **Denoise Control Video**, the visual changes follow the denoising and mask settings.
- **Generate Audio based on Control Video and Text Prompt:** select **Denoise Control Video**. WanGP keeps the control video's frames unchanged and asks H3 to create a new soundtrack.

For visible speech in soundtrack mode, identify the character and explicitly request synchronized speaking, for example `The woman is speaking with natural lip movements synchronized.` Do not write the spoken words or mention the supplied audio track or its language: given a written line, the model voices its own dialogue and the mouth often stops following the soundtrack. Keep the action to speaking: smiles and gestures compete with the mouth. See the prompt guidance for examples; supplying speech alone does not guarantee visible speaking or exact lip sync.

### Longer videos

Sliding windows can continue a video beyond one generation. Choose any overlap amount and WanGP will round it to the nearest H3-compatible value (1, 18, 35, 52...). It automatically reuses the overlapping video and audio to make the join smoother.

**End images between windows:** each window lands exactly on its end image. When the action has drifted away from that picture, the last few frames jump to it. To hide this jump, set **Discard Last Frames** to about `5`: the end image of every window except the last is then generated just after the discarded frames, so it steers the action without being shown, and the next window continues from the frames before it. The last window still ends exactly on its end image.

H3 is designed for 24 FPS, although WanGP can generate at another frame rate. MiniMax documents an official duration of 4–15 seconds per generation window; longer videos are possible through sliding windows.
"""

REF2VA_DEEPY_INFOS = """Generate video and 32 kHz stereo audio from `prompt` and references. Ordered `image_refs` guide identity, objects or setting; reference flags in `video_prompt_type`: `I` preserves chosen output dimensions, `KI` derives them from the first image. `image_start` / `image_end` are timeline anchors shown before general image references. `FI` injects the first `image_refs` at `frames_positions` (`1` = first frame, `L` = window end, `X` = skip a window); remaining images are references; not with `GV`. Per window, pictures are numbered: start or continued frame (`<Picture 1>`), end image, that window's injected frames by time, then references. `video_source` and sliding windows provide continuation. A window may jump onto an end image its action did not reach; `sliding_window_discard_last_frames` about 5 hides it: the end image of every window but the last is generated after the discarded frames (steers, not shown); the last window ends on its end image.

`video_guide`, `video_guide2` and `video_guide3` supply up to three reference videos for appearance, motion or camera; choose the corresponding video mode. `V1-U` instead takes up to three excerpts from `video_guide` at `custom_settings.h3_video_excerpt_positions`: frame numbers or seconds such as `5.2s`, each optionally followed by `/duration` (default 3s), for example `3 5.2s/4s 12s`. Reference videos preserve the chosen output size and are given whole to every sliding window. Control videos (`VU`, `GV`) use the control video's aspect ratio and are split across sliding windows, each window receiving its own section. Use Reference Video for motion transfer to image-reference characters; for depth, pose or edge control use the MiniMax H3 FL2VA ControlNet-Union model. `VU` supplies the control video's unchanged frames as `<Video 1>`: an in-context guide for LoRAs that transform a source video, such as character swap; name it `<Video 1>` in the prompt and state what to do with it. `GV` (Denoise Control Video) edits through denoising: lower strength preserves more source content; at strength 1 without a mask, the source supplies no visual conditioning. `GV` does not supply a video reference.

`audio_prompt_type`: empty = no audio input. References, whose voice or sound H3 reuses in newly generated audio: `A` = `audio_guide`, `AB` = two audio guides, `ABD` = three audio guides, `K` = reference-video soundtracks (not with excerpts);  `K1` = up to three soundtrack excerpts at `custom_settings.h3_audio_excerpt_positions`, same syntax, independent of the video excerpts; the prompt defines each reference's role. Soundtrack, kept as the output audio and followed by the video across sliding windows: `AS` = `audio_guide`, `KS` = the whole audio track of the reference video or of a control video (`GV`, `VU`). `K` / `K1` need reference videos (`K1` a single one or excerpts from one); `KS` also accepts a control video. A kept soundtrack is not a reference: give it no <Audio N> label.

Limits: 9 reference images (injected frames not counted); 3 videos, each at least 2s, a `VU` control counting as one but split per window instead of trimmed; one reference video keeps its first 15s (362 frames), and videos above 15s combined keep an equal share each (7.3s for two, 4.5s for three); 3 audio references, each at least 2s. Audio above 15s combined is split evenly: 15s for one reference, 7.5s each for two, 5s each for three. Image + video reference count must cover audio reference count. At most 12 uploaded reference files; a video soundtrack shares its video's file. Keep backgrounds when scene context matters; optional background removal isolates subjects. Read `prompt_infos` for Ref2VA's six-section syntax.
"""

REF2VA_INFOS = """## Ref2VA — Reference to Video and Audio

Ref2VA generates a new video with native 32 kHz stereo audio from text plus multimodal references. Images can guide identity, appearance, or scene content; video can guide content, appearance, and motion; audio can guide or reuse sound and voice. A reference video is contextual material, not a guaranteed frame-exact continuation constraint.

### Reference limits

- **Images:** up to 9. Injected frames are not counted.
- **Videos:** up to 3 clips, each at least 2 seconds long. An in-context control video counts as one of them, but it is split across sliding windows instead of being trimmed. A single reference video keeps its first 15 seconds (362 frames). When several videos total more than 15 seconds, each keeps an equal share: about 7.3 seconds for two and 4.5 seconds for three. A reference video's soundtrack is trimmed with its video.
- **Audio:** up to 3 inputs, each at least 2 seconds long. When their combined duration exceeds 15 seconds, WanGP splits the 15 seconds evenly: 15 seconds for one reference, 7.5 seconds each for two, 5 seconds each for three.
- **Audio requires matching visual references:** the combined number of reference images and videos must be at least the number of reference audio clips.
- **Video soundtracks:** selecting reference-video soundtracks uses one audio-reference slot per selected video; soundtrack excerpts use one slot per excerpt. Each video excerpt counts as one reference video. A soundtrack shares its video's uploaded file, so it does not add another file to the mixed-input count.
- **Mixed references:** at most 12 files across images, videos, and audio.

### Audio Source

- **Generate without an Audio Reference:** H3 creates the sound from the prompt.
- **Use One / Two / Three Audio References:** H3 reuses the voice or sound of each uploaded audio in newly generated audio. Name them `<Audio 1>` to `<Audio 3>` in the prompt and state their role.
- **Use Reference Videos Soundtrack(s) as Audio References:** the same, using each reference video's audio track. Not available with excerpts: use the soundtrack excerpt choice below.
- **Use Up to 3 Excerpts from Reference-Video Soundtrack:** takes audio references from the reference video's soundtrack at the positions entered in **Audio Reference Positions**, independently of any video excerpts. Requires a single reference video or excerpts from one.
- **Generate Video based on Soundtrack and Text Prompt (Soundtrack Kept):** upload a speech or music track that must be heard exactly as it is. H3 creates the video around it and WanGP keeps the original audio in the final file, in sync across sliding windows.
- **Generate Video based on Reference / Control Video + its Audio Track and Text Prompt (Soundtrack Kept):** the same, using the whole audio track of the reference video or of the control video. A reference video (or its excerpts) still guides appearance and motion; a control video (in-context or denoised) drives the generation in sync with its own audio. Requires a single reference video, excerpts from one, or a control video.

A kept soundtrack is not a reference: do not mention a supplied audio track or assign it an `<Audio N>` label in the prompt. For visible speech, identify the character and include `speaking with natural lip movements synchronized` or equivalent lip-sync wording. Do not write the spoken words (no `<d>` dialogue) or name the language, and keep the action to speaking; see the prompt guidance for examples.

### Excerpts from one Reference Video

Select **Use Up to 3 Excerpts from same Reference Video** to take up to three moments from one longer video instead of uploading separate clips. Enter them in **Reference Positions from Control Video** as frame numbers or seconds, separated by spaces or commas. Each position is the centre of an excerpt lasting 3 seconds unless a duration follows a slash:

- `3 5s 12s`: three 3-second excerpts around frame 3, 5 seconds, and 12 seconds.
- `3/4s 5.2s/3s 12s/2.5s`: the same moments with individual durations.

Each excerpt lasts at least 2 seconds, the excerpts total at most 15 seconds, and an excerpt near the start or end of the video is moved inward to keep its duration. Each excerpt counts as one reference video; name them `<Video 1>` to `<Video 3>` in order.

### Choosing a video input

- **Reference Video:** reuse subjects, appearance, or motion without changing the selected output resolution. Choose this mode to transfer motion from a video to characters supplied in reference images; describe the subject replacements in the prompt. Reference videos do not use Denoising Strength.
- **Use Control Video as In-Context Guide:** H3 receives the control video's unchanged frames as `<Video 1>` and generates a new video from them. Choose this mode for LoRAs that transform a source video, such as a character swap: name the video `<Video 1>` in the prompt and say what to change and what to keep. Without such a LoRA, it behaves like a Reference Video that follows the control video section by section.
- **Denoise Control Video (Video-to-Video Edit):** edit the source video using the prompt and **Denoising Strength**. Lower strength preserves more source content. At `1.0` with **Whole Frame** selected, the source video supplies no visual conditioning. Choose **Masked Area** or **Non Masked Area** to edit part of the frame. This mode does not supply the video as a motion/appearance reference.

Reference videos adapt to your chosen output size and guide the result creatively, without guaranteeing exact motion reproduction. Each sliding window receives the whole reference video (up to its first 15 seconds).

To follow a video's depth, pose, edges or shapes, use the **MiniMax H3 FL2VA ControlNet-Union** model instead.

Control videos define the output size and are split across sliding windows: each window receives only its own section, so a long control video can drive a long generation. A window can be longer than 15 seconds; MiniMax documents 4–15 seconds per window, so longer windows may lose quality.

### Reference-image size

The reference-image selector can preserve the selected output dimensions or use the first reference image to define them. In **Advanced Mode**, **Rescale Internaly Image Ref (% in relation to Output Video) to change Output Composition** controls each reference image's internal pixel budget while preserving its aspect ratio. **100%** matches the output video's pixel budget. Higher values can preserve more reference detail and improve fidelity, but generation is slower; lower values are faster but can lose fine details. This setting does not change the output resolution after it has been selected or derived.

### Background removal

Background removal is disabled by default. After selecting reference images, use **Automatic Removal of Background behind People or Objects in Reference Images** when you want WanGP to isolate people or objects before sending the images to H3. Keep backgrounds when the surrounding scene is part of the reference you want H3 to follow.

### Start/end images and longer videos

Ref2VA also supports optional start and end images. These are shown to the prompt before the general reference images: with a start image and one reference image, the start image is `<Picture 1>` and the reference image is `<Picture 2>`.

### Injected frames

Select **Inject Frames at chosen Positions** in **Reference Images** to place images at exact moments of the video, and enter one position per image under **Positions of Injected Frames**: `1` is the first frame, `L` the last frame of a sliding-window segment, and `X` skips a window without consuming an image. Add the injected images first; any images beyond the positions are ordinary reference images for identity, objects or style. Injected frames combine with reference videos and in-context control videos, but not with **Denoise Control Video**. The output size follows the first injected image unless a control video defines it.

An injected frame on a window's first or last frame acts as that window's start or end image; one inside the sliding-window overlap is not used, because the overlap already continues the previous window. In each window, pictures are numbered in this order: the start image or the frame continued from the previous window (`<Picture 1>`), the end image, the injected frames of that window in time order, then the reference images. The numbering restarts in every window, so describe each window separately when injected frames or reference images change the count, and declare each injected frame at its time in the window.

For longer videos, choose any sliding-window overlap amount and WanGP will round it to the nearest H3-compatible value (1, 18, 35, 52...). It automatically carries the overlapping video and audio into the next window while keeping the selected references available.

**End images between windows:** each window lands exactly on its end image. When the action has drifted away from that picture, the last few frames jump to it. To hide this jump, set **Discard Last Frames** to about `5`: the end image of every window except the last is then generated just after the discarded frames, so it steers the action without being shown, and the next window continues from the frames before it. The last window still ends exactly on its end image.

H3 is designed for 24 FPS, although WanGP can generate at another frame rate. MiniMax documents an official duration of 4–15 seconds per generation window; longer videos are possible through sliding windows.

See the [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/README.md) for the upstream specifications and prompting guidance.
"""

CONTROL_CONTINUATION_INFOS = (
    "ControlNet overlap choices are 5, 22, 39, and so on; 22 is the default. Older 1- and 18-frame settings become 5 and 22. Automatic continuation reuses generated overlap latents."
    if H3_CONTROL_LATENT_CONTINUATION else
    "ControlNet overlap choices are 1, 18, 35, and so on; 18 is the default. WanGP rounds other overlap amounts to the nearest supported value."
)

CONTROL_INFOS = f"""## FL2VA ControlNet-Union — Control Video to Video and Audio

This model adds the VideoX-Fun ControlNet-Union 2.0 branch to MiniMax H3 FL2VA. A control video drives the structure and motion of the generated video, while the text prompt defines its content, style and sound. The output takes the control video's aspect ratio and H3 still generates synchronized stereo sound.

### Control Video

- **Transfer Human Motion:** WanGP extracts an Open Pose skeleton from the control video. Use it to animate new characters with the same body movements.
- **Transfer Depth:** a depth map keeps the scene layout, camera movement and object placement while the prompt restyles everything.
- **Transfer Canny Edges:** fine outlines preserve shapes and silhouettes closely; this is the strictest structural guide.
- **Transfer Shapes:** softer scribble-like outlines keep the overall composition with more freedom for details.
- **Recolorize:** a grayscale version of the video keeps its lighting and content; describe the colors in the prompt.
- **Use Control Video as is:** normally, upload an already prepared control map (Pose, Depth, Canny, HED, MLSD, Scribble, Layout or Gray). With **Spatial Outpainting** enabled, supply the original video instead: its picture is kept as context while ControlNet generates the new borders. No Video Mask is needed for border expansion alone.
- **Perform Inpainting:** regenerate only the masked area of the video from the prompt, without a control map.

**Control Strength** scales the control branch: `1.0` follows the control video closely, lower values give the prompt more freedom and `0` disables it.

### Masks

With a Video Mask, only the masked area is regenerated and follows the control map, while the rest of the source video is kept as a reference for the model. The kept area is guided rather than copied pixel for pixel, so fine details can still drift. Masks require one of the preprocessing modes above, or **Use Control Video as is** with Spatial Outpainting enabled, because WanGP needs the original video to know what to keep.

### Spatial Outpainting

Select **Use Control Video as is**, supply the original video, enable **Spatial Outpainting**, then set the top, bottom, left and right expansion or choose a target aspect ratio. Describe the complete scene and the surroundings you want to add. The requested resolution is the final expanded canvas; use a larger resolution if you want to retain the source's detail.

The original rectangle is visible context for ControlNet's inpainting branch, and the new borders are automatically masked for generation. This uses ControlNet conditioning; **Denoising Strength** and **Mask Denoising Mode** from ordinary FL2VA are not used. **Control Strength** controls how strongly the visible context guides the result. The original area can still change slightly.

To also edit an area inside the original picture, add a Video Mask or select **Perform Inpainting** with a mask, then enable Spatial Outpainting. Pose, depth, edge, shape and grayscale modes can likewise combine expansion with an explicit mask; their unmasked area supplies the original picture. For outpainting with **Use Control Video as is**, upload the original RGB video rather than a pose/depth/edge map. Sliding windows extend each matching source-video section. {CONTROL_CONTINUATION_INFOS} The original picture remains guidance and can still change.

### Start / End Images, Continuation and Audio

A start image, an end image or a video to continue can be combined with the control video. H3 anchors those frames while the control branch drives the rest; the control model was trained without such anchors, so keep them consistent with the control video (for example the first frame of the source video). Sliding windows split long control videos: each window receives its own section.

- **Generate Video and Audio from Text Prompt:** H3 creates new sound matching the prompt and motion.
- **Generate Video based on Soundtrack and Text Prompt:** an uploaded soundtrack is kept and the video follows it.
- **Generate Video based on Control Video + its Audio Track:** keep the control video's own soundtrack, for example the music of a dance video.

A kept soundtrack stays in the saved output, but visible speech also depends on the prompt and the pose guide. Identify the character and explicitly request synchronized speaking, for example `The woman is speaking with natural lip movements synchronized.` Do not write the spoken words or mention the supplied audio track or its language: given a written line, the model voices its own dialogue and the mouth often stops following the soundtrack. Keep the action to speaking: smiles and gestures compete with the mouth. A pose guide showing a still mouth or a lowered face can compete with speaking motion; keeping the soundtrack does not guarantee exact lip sync.

### Recommended settings

The branch is guidance-distilled: keep **Guidance** at 1. Upstream demonstrations use 40 steps; 20 steps already give usable results. Keep each window within 15 seconds (362 frames), the duration the control branch was trained on. Two-phase generation and text-to-image are not available with this model.

See the [ControlNet-Union 2.0 model card](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0) for upstream details.
"""

CONTROL_DEEPY_INFOS = f"""Generate video and stereo sound from `prompt` guided by a control video (`video_guide`) through the ControlNet-Union branch; the output uses the control video's aspect ratio. `video_prompt_type`: `PV` = Open Pose motion, `DV` = depth, `EV` = Canny edges, `SV` = soft shapes/scribble, `CV` = grayscale recolorize, `V` = video already a control map (HED, MLSD, layout...), `MV` = inpainting only. Add `A` (masked area) or `NA` (non-masked area) with `video_mask` to regenerate only part of the frame; masks need a preprocessing mode or `V` with spatial outpainting enabled. `control_net_weight` (0-2, default 1) sets control strength. Long control videos are split across sliding windows (max 362 frames each). {CONTROL_CONTINUATION_INFOS} `image_start` / `image_end` / `video_source` anchor frames and should match the control video. `audio_prompt_type`: empty = generate audio; `A` = keep `audio_guide` as soundtrack; `K` = keep the control video's soundtrack. For visible speech in soundtrack mode, identify the speaker and explicitly request `speaking with natural lip movements synchronized` or equivalent lip-sync wording, without mentioning a supplied audio track or its language. Do not write the spoken words (no `<d>` dialogue); face pose and gestures compete with speech, and exact lip sync is not guaranteed. Guidance stays 1. Read `prompt_infos` for H3's structured prompt syntax."""

CONTROL_DEEPY_INFOS += """\n\nSpatial outpainting: use `video_prompt_type: V` (Use Control Video as is), the original RGB `video_guide`, and `video_guide_outpainting` as four space-separated percentages (top bottom left right), e.g. `0 0 25 25`. No `video_mask` is needed for borders alone. Alternatively, set `video_guide_outpainting_ratio`, e.g. `16:9`, with `video_guide_outpainting: 0 0 0 0` for balanced expansion; `#` disables expansion. `resolution` is the final canvas. Enabling expansion routes the visible source and automatic border mask through ControlNet inpainting, not FL2VA denoising; no Grouped Rows requirement. With expansion disabled, `V` retains its normal prepared-control-map behavior. For simultaneous interior edits, use `VA` or `MVA` plus `video_mask`; masked pose/depth/edge/shape/grayscale modes also support expansion. Describe the completed scene and new surroundings; the original rectangle is guided, not copied pixel for pixel."""

H3_PHASE_INFOS = """
### How to use one phase, two phases, and tiling

Enable **Advanced Mode**, open **General**, and choose an option under **Phases**:

- **One Phase (default):** generates directly at the selected resolution. Use it for standard resolutions or when whole-frame consistency matters more than high-resolution speed.
- **Two Phases:** use it for faster high-resolution generation and improved fine detail. Most of the work is performed at a lower resolution before H3 enhances the result at the selected output resolution. This mode does not reduce the peak VRAM required by the final enhancement.
- **Two Phases with Tiling (experimental):** use it when regular two-phase generation runs out of VRAM. It processes the final enhancement as four overlapping areas, reducing peak VRAM at the cost of extra processing time and a possible risk of visible seams or local inconsistencies.

**Phase 2 Steps** controls the final enhancement independently of the first phase; the default is 3. Other counts use evenly spaced noise levels from **Phase 2 Noise Level Start** to zero. Start with the default noise level; lower it to keep the result closer to the first phase and favor smoother tile blending, or raise it to encourage stronger new details, with a greater risk of seams or changes between tiles. More steps are not necessarily better with a few-step accelerator.

Leave **Custom Sigmas** empty for the normal schedules. Use commas between values and `;` between phases; an empty phase slot keeps its normal schedule. Explicit values set that phase's steps and starting noise. The **i** beside the field explains the syntax.

For FL2VA, **Open-H3 Regenerate 2K Phase 2** selects the refiner LoRA and its three-step schedule. Phase 2 uses the first phase's audio as frozen conditioning while preserving its output soundtrack. **H3 Phase 2 (Original)** restores the usual Turbo and schedule; **Vanilla Phase 1 (Reset)** returns to one-phase generation. Tiling remains experimental with the refiner.
"""

H3_PHASE_TURBO_INFOS = """
Phase 2 uses the automatic Turbo LoRA unless an accelerator before `|` in **LoRAs Multipliers** has a nonzero phase-2 multiplier. In that case, all selected accelerators keep their phase-2 multipliers and replace the automatic Turbo. For example, `1;0` keeps an accelerator in phase 1 only and uses the automatic Turbo in phase 2; `1;0.6` uses that accelerator at 0.6 in phase 2; `1` applies to both phases. Multiple accelerators and per-step multiplier sequences are supported. LoRAs after `|` retain their own multipliers.
"""

H3_SPEED_INFOS = """
### Speed and memory choices

Enable **Advanced Mode** to access these options:

- **Spectrum:** in **Steps Skipping**, select **Spectrum Feature Forecasting**. Spectrum captures an accelerated local-only trajectory, retains its actual-step anchors in system RAM, then performs a transformer-free smoothing replay with independent video and audio prediction. Keep the default 25% start for five full warmup steps in a 20-step generation; increasing it starts later and skips fewer steps. Short Euler schedules can bootstrap after their first actual step, while RES Multistep preserves a three-step actual tail.
- **First Block Cache:** in **Steps Skipping**, select **First Block Cache**. It runs the first transformer block to decide whether the remaining blocks can reuse their previous result. The balanced strength uses the upstream 0.08 threshold; higher strengths skip more work but can change motion or fine details. The displayed strength is not an exact speed multiplier.
"""

H3_STANDARD_SAMPLER_INFOS = """
- **Ralston 2S:** in **Sampler Solver / Scheduler**, select **Ralston 2S** to use the anchored deterministic second-order Runge-Kutta sampler. It evaluates H3 at the start and two-thirds point of every interval, anchors the second prediction to the interval start, then combines both predictions with Ralston's `1/4, 3/4` weights. This can reduce numerical integration error and may improve fine-detail retention, motion stability, and audio/video coherence. Perceptual improvements are prompt-dependent and are not guaranteed. Its second prediction depends on the first, so they cannot run in parallel: Ralston performs two full transformer predictions per step and sampling is approximately **2x slower** than Euler or RES Multistep at the same step count. Spectrum Feature Forecasting is unsupported with Ralston 2S.
"""

PIKA_INFOS = """

### Pika H3 Flex

Apply **Pika H3 Flex FL2VA 6 Steps** or **Pika H3 Flex Ref2VA 6 Steps** from the accelerator profiles for the matching model. FL2VA generates video and stereo audio from text, a first frame, a last frame, or both. Ref2VA uses image/video references and optional audio references; it requires at least one image or video.

The profile selects **Pika Re-noise (6 Steps)**, six steps, flow shift 12 and one generation phase. Keep **Steps Skipping** off and **Custom Sigmas** empty. The sampler requires the matching Pika adapter; selecting it alone does not accelerate the base model. Use **Vanilla Phase 1 (Reset)** to return to ordinary H3 generation.

Start with 480p or 768p and a 5–15 second clip. These are Pika's released local workflows; 1080p/2K/4K generation and separate audio stems from Pika's hosted API are not included. Use the existing H3 reference controls and their picture/video/audio labels. Reference-video soundtracks are included when selected as audio references. Prompt enhancement is optional and off in the Pika templates.
"""

PIKA_DEEPY_INFOS = (" Pika H3 Flex accelerator profiles select the matching FL2VA/Ref2VA rank-128 adapter at multiplier 1, "
                   "sample_solver=pika_renoise, num_inference_steps=6, flow_shift=12, guidance_phases=1, custom_sigmas='', "
                   "skip_steps_cache_type=''. Use the profile, not the sampler alone. Start at 480p/768p and 124–362 frames at 24 fps. "
                   "Ref2VA requires at least one image/video; up to 9 images, 3 videos and 3 standalone audio references, 12 files total. "
                   "Video soundtracks selected with audio_prompt_type=K stay attached to their videos. H3 reference order is images, videos, "
                   "then standalone audios; use <Picture N>, <Video N>, <Audio N> labels. Pika has no released local high-resolution or audio-stem stage. "
                   "Vanilla Phase 1 (Reset) restores Euler and removes accelerators while retaining additional LoRAs.")

H3_AUDIO_REFINEMENT_INFOS = """
### Audio Refinement Extra Phase

Use **Audio Refinement Extra Phase** to improve the soundtrack after video generation without LoRAs. WanGP preserves the completed video latent exactly, partially re-noises the existing audio latent at 0.5 denoising strength, and runs a few additional audio steps before decoding both streams. Keeping the original full-resolution video and audio latents preserves fine lip motion and the first pass's timing. This extra phase runs after all video phases, so it is the second phase after normal generation and the third phase after two-phase video generation.

When enabled, the refinement uses 6 extra steps at 0.5 denoising strength.

The extra phase uses the text prompt and the locked final video only. Reference images, reference video or audio, Control Video, and selected LoRAs are not injected again. Each extra step still processes the full-resolution packed sequence so that audio can attend to the video; the video output of those steps is discarded. FL2VA hides this option when an input soundtrack controls the result, and PDD variants do not offer it because their denoising schedule is fixed to 8 steps.
"""

H3_COMMON_RUNTIME_INFOS = """
- **Sol-Attn:** in **Advanced Mode > Configs > Override Attention Mode**, select **sol**. The **Start Tau** slider then appears below the attention selector and shows that End Tau is fixed at `0.8`. H3 defaults to `1.3`; this value is used on the first denoising step and decreases linearly to `0.8` on the final step. Use `1.0` for the Sol-Attn paper starting value, increase it to route more attention blocks through the approximate path for greater speed, or lower it for denser attention and higher fidelity. It uses sparse attention only on large visual sequences and requires BF16, Triton 3.6 or newer, and a CUDA NVIDIA GPU using SM86, SM89, SM90, SM100, SM120, or SM121 (such as RTX 30/40/50-series, H100/H200, B100/B200, or DGX Spark); the dropdown reports whether it is available on the current system.
"""

H3_TEXT_ENCODER_INFOS = """Choose how much system RAM to use for interpreting prompts.

- **Default:** use this model's default text encoder and the applicable Text Encoder Quantization setting.
- **Qwen3-VL BF16:** uses the most memory.
- **Qwen3-VL Quanto INT8:** a balanced lower-memory choice.
- **Qwen3-VL NVFP4 AWQ** and **Qwen3-VL GGUF Q4_K_M:** reduce memory further.
- **Qwen3-VL GGUF Q2_K:** the most aggressive memory reduction.

More aggressive quantization can slightly affect prompt interpretation.
"""

H3_VIDEO_VAE_INFOS = """Choose the decoder used to turn generated video into frames.

- **Auto:** follows Transformer Quantization: INT8 selects INT8 ConvRot Decoder, FP8 selects FP8 Mixed Precision, and other choices select BF16.
- **BF16:** use the full H3 VAE in BF16.
- **FP8 Mixed Precision:** use the mixed-precision version of the full H3 VAE.
- **INT8 ConvRot Decoder:** use the quantized full decoder.
- **LightVAE Lite BF16** and **LightVAE Lite INT8 ConvRot:** use a smaller 12-layer decoder. Both keep the original H3 encoder for first/last frames and references. Decoded detail can differ from the full VAE.

Selecting **MiniMax H3 VAE** spatial upsampling replaces the selected decoder, including LightVAE, with the X2 decoder.
"""

H3_DENOISING_PRIORITY_INFOS = """Choose which memory limit matters most during denoising and video decoding.

- **Lower VRAM (default):** use all memory optimizations to greatly reduce GPU memory consumption, including optimizations that need extra system RAM.
- **Lower RAM:** use only the GPU memory optimizations that do not need extra system RAM.
"""

H3_AUDIO_GENERATOR_INFOS = """## MiniMax H3 audio generator

This audio-only preset reuses the pruned Ref2VA checkpoint. H3 still jointly denoises a tiny 32x32 video internally, but WanGP skips the video decode and saves only the generated 32 kHz stereo audio.

- **No reference:** generate speech, ambience, Foley, or music from the prompt alone.
- **One audio reference:** use `<Audio 1>` in the prompt to describe the voice, delivery, music, or sound characteristics to retain.
- **Two or three audio references:** use `<Audio 1>` to `<Audio 3>` independently, for example as cloned speaker voices.
- Each reference must be at least 2 seconds long. Above a combined 15 seconds, the 15 seconds are split evenly: 15 seconds for one reference, 7.5 seconds each for two, 5 seconds each for three.
- **Maximum Total Audio Duration** is the cumulative limit for the assembled monologue or dialogue. Each speaker turn is generated separately; individual segments above H3's official 15-second range remain experimental.
- **Early Stop** finishes the H3 segment currently being generated, then assembles and returns all completed segments.

H3 remains an audiovisual model even though this preset discards the video. A structured H3 prompt that describes the hidden scene and binds each speaker to a stable ID such as `(S1)` or `(S2)` can improve audio coherence. Put exact dialogue inside `<d>[Language] ...</d>`.
"""

PDD_INFOS = """
### PDD 8-step acceleration

At each step, PDD merges four learned denoising-interval outputs into one prediction, covering 32 intervals in only 8 model evaluations.

This model requires exactly **8 inference steps** and the **Euler** sampler. Two-phase generation is disabled. Use the FL2VA PDD weights only with FL2VA and the Ref2VA PDD weights only with Ref2VA. LoRAs trained for the matching base model apply; their output-layer changes reach all PDD heads.
"""

H3_VDN_INFOS = "\n\n### Automatic 8-step acceleration\nThe VDN 8-step acceleration LoRA is automatically loaded and generation defaults to 8 steps."

FASTH3_STEPS = 8
FASTH3_VIDEO_SHIFT = 12.0
FASTH3_INFOS = """## FastH3 Omni — 8-Step Reference to Video and Audio

FastH3 Omni is FastVideo's distilled MiniMax H3 Ref2VA student: it generates a video with stereo sound in **8 model evaluations**, on the PDD schedule learned during distillation.

### What it accepts

- **The Ref2VA inputs.** Ordered image, video and audio references named `<Picture N>`, `<Video N>` and `<Audio N>` in the prompt, start and end images, frame injection, reference or control videos, soundtracks, sliding windows and continuation, as described below for Ref2VA.
- **Distilled with references.** FastVideo trained the student on ordered references (up to nine images, three videos and three audio clips, twelve in total, at least one image or video) and validates only those; the other inputs come from the Ref2VA base it was trained from. Expect the best results with at least one image or video reference.

### How it runs

- **8 inference steps, Euler sampler:** both are locked (a 32-interval PDD grid fused into 8 forwards). The **Flow Shift** (default 12) is the video shift the student was trained with; the audio shift stays 3.
- **VSA sparse attention** is always on: every query tile of the generated video attends the 10% most relevant tiles of the video and 10% of each reference video, plus every text, audio and image token, with a learned coarse branch over the rest. It needs Triton on an NVIDIA RTX 30xx or newer GPU.
- **Quality:** a distilled student favors speed; difficult motion, fine detail and some sounds can fall short of the full H3 models.
- **LoRAs:** LoRAs trained for Ref2VA apply; their output-layer changes reach all PDD heads.
"""
FASTH3_DEEPY_INFOS = ("Generate video and stereo sound in 8 steps (FastVideo FastH3 Omni, a distilled Ref2VA student) with the Ref2VA settings: ordered "
                      "`image_refs` (`I`, or `KI` to take the output size from the first image), `video_guide` / `video_guide2` / `video_guide3` reference "
                      "videos (`V` modes, `V1-U` excerpts) or control videos (`VU`, `GV`), `audio_guide`... references (`A` / `AB` / `ABD`, `K` / `K1` "
                      "soundtracks, `AS` / `KS` kept soundtracks), `image_start` / `image_end`, `FI` frame injection, `video_source` and sliding windows. "
                      "Name references `<Picture N>`, `<Video N>`, `<Audio N>`. The student was distilled with ordered references (up to 9 images, 3 videos, "
                      "3 audio clips, 12 in total): use at least one image or video reference for the best results; the other inputs are inherited from "
                      "the Ref2VA base. `num_inference_steps` is locked to 8 with the Euler sampler; `flow_shift` defaults to 12 (the trained video shift). "
                      "Attention is locked to VSA sparse attention (Triton, NVIDIA SM80+). LoRAs trained for Ref2VA apply. Read `prompt_infos` for H3's structured prompt syntax.")

H3_RUNTIME_INFOS = H3_PHASE_INFOS + H3_PHASE_TURBO_INFOS + H3_AUDIO_REFINEMENT_INFOS + H3_SPEED_INFOS + H3_STANDARD_SAMPLER_INFOS + H3_COMMON_RUNTIME_INFOS
H3_PDD_RUNTIME_INFOS = PDD_INFOS + H3_SPEED_INFOS + H3_COMMON_RUNTIME_INFOS
H3_FASTH3_RUNTIME_INFOS = FASTH3_INFOS + H3_SPEED_INFOS + H3_COMMON_RUNTIME_INFOS

PRUNED_INFOS = """
### Pruned 20B checkpoint

The Pruned checkpoint replaces the full AdaLN timestep projection matrices with precomputed low-rank modulation curves. It accepts the same inputs and settings as its 33B counterpart while reducing checkpoint size and weight-transfer cost.
"""

H3_FINETUNES_INFOS = """### H3 finetune QKV layout

Most H3 checkpoints use the official **Interleaved** QKV layout. Select **Grouped** only when the checkpoint stores all Q rows, then K rows, then V rows. INT8 ConvRot uses its own layout metadata.
"""

H3_FINETUNES_PARAMS = {
    "qkv_layout": {
        "label": "QKV Layout",
        "choices": [("Interleaved (official H3)", "interleaved"), ("Grouped Q / K / V", "grouped")],
        "default": "interleaved",
        "description": "Physical row order of fused QKV tensors in the finetune checkpoint. This does not enable or disable QKV splitting.",
    },
}


def _uint8_guides(video_prompt_type, audio_prompt_type, any_outpainting):
    # with the audio generated from the control video, the control video becomes the output video: float control videos for this mode
    return "2" not in (audio_prompt_type or "")


def _notify_audio_reference_limit(audio_durations):
    total_duration = sum(audio_durations)
    if total_duration <= 15 or not audio_durations:
        return
    limit = 15 / len(audio_durations)
    gr.Info(f"MiniMax H3 reference audio totals {total_duration:.2f}s, above the 15s limit. Each reference will be limited to at most {limit:g}s.")


def _get_audio_generator_model_def(model_def):
    text_encoder_variant = model_def.get("text_encoder_variant")
    text_encoder_files = [TEXT_ENCODER_BF16, TEXT_ENCODER_INT8] if text_encoder_variant is None else TEXT_ENCODER_VARIANTS[text_encoder_variant]
    return {
        "audio_only": True,
        "device_explicit": True,
        "image_outputs": False,
        "profile_type": "video",
        "preserve_empty_prompt_lines": True,
        "sliding_window": False,
        "guidance_max_phases": 1,
        "visible_phases": 0,
        "no_negative_prompt": True,
        "inference_steps": True,
        "temperature": False,
        "flow_shift": True,
        "fps": 24,
        "supports_early_stop": True,
        "profiles_dir": ["minimax_h3_tts"],
        "duration_slider": {
            "label": "Maximum Total Audio Duration (seconds)",
            "name": "Maximum Total Audio Duration",
            "min": 4,
            "max": int(H3_DIALOGUE_MAX_TOTAL_SECONDS),
            "increment": 1,
            "default": 15,
        },
        "any_audio_prompt": True,
        "audio_prompt_choices": True,
        "audio_reference_max_total_duration": 15,
        "reference_audio_enabled": True,
        "audio_guide_label": "Voice / Audio Reference 1",
        "audio_guide2_label": "Voice / Audio Reference 2",
        "audio_guide3_label": "Voice / Audio Reference 3",
        "audio_prompt_type_sources": {
            "selection": ["", "A", "AB", "ABD"],
            "labels": {
                "": "Generate without an Audio Reference",
                "A": "Use One Voice / Audio Reference",
                "AB": "Use Two Voice / Audio References",
                "ABD": "Use Three Voice / Audio References",
            },
            "letters_filter": "ABD",
            "label": "Voice / Audio References",
            "show_label": True,
            "default": "A",
        },
        "enabled_audio_lora": True,
        "lora_multiplier_phases": 1,
        "custom_settings": [],
        "spectrum_cache": True,
        "first_block_cache": True,
        "skip_steps_multiplier_choices": FIRST_BLOCK_CACHE_STRENGTHS,
        "skip_steps_multiplier_label": "First Block Cache Threshold",
        "first_block_cache_thresholds": FIRST_BLOCK_CACHE_THRESHOLDS,
        "custom_attention_modes": {
            "sol": {"label": "Sol sparse attention, requires Triton and RTX 30xx or newer", "supports_sparsity": True},
        },
        "default_attention_modes_supported": True,
        "attention_sparsity": {
            "label": "Start Tau (higher = more sparse/faster; lower = more faithful; End Tau = 0.8)",
            "start": 0.0,
            "end": 4.0,
            "inc": 0.05,
        },
        "sample_solvers": [("Euler", "euler"), ("RES Multistep", "res_multistep"), ("Ralston 2S (~2x slower)", "ralston_2s")],
        "infos": H3_AUDIO_GENERATOR_INFOS + H3_SPEED_INFOS + H3_STANDARD_SAMPLER_INFOS + H3_COMMON_RUNTIME_INFOS + PRUNED_INFOS,
        "prompt_infos": (H3_DIALOGUE_PROMPT_INFOS if H3_DIALOGUE_GENERATION else "") + REF2VA_PROMPT_INFOS,
        "deepy_infos": "Generate 32 kHz stereo audio from `prompt`. `audio_prompt_type`: empty = no sample; `A` = voice/audio from `audio_guide`; `AB` = two audio guides; `ABD` = three audio guides. For Speaker scripts, samples map to speakers 1 to 3. Each sample must be at least 2s; above 15s combined, the 15s are split evenly across samples. `duration_seconds` caps the assembled audio. Speaker turns are generated and joined automatically; Early Stop finishes the current turn and returns completed turns.",
        "deepy_prompt_infos": H3_AUDIO_DEEPY_PROMPT_INFOS if H3_DIALOGUE_GENERATION else REF2VA_DEEPY_PROMPT_INFOS,
        "prompt_enhancer_button_label": "Write",
        "prompt_enhancer_def": {
            "selection": ["T", "T1"],
            "labels": {"T": "A Monologue from Text", "T1": "A Dialogue from Text"},
            "default": "",
        },
        "text_prompt_enhancer_instructions": H3_AUDIO_MONOLOGUE_SYSTEM_PROMPT,
        "text_prompt_enhancer_instructions1": H3_AUDIO_DIALOGUE_SYSTEM_PROMPT,
        "text_prompt_enhancer_max_tokens": 1024,
        "text_prompt_enhancer_max_tokens1": 2048,
        "dtype": "bf16",
        "qkv_splitting": True,
        "qkv_layout": "interleaved",
        "text_encoder_folder": TEXT_ENCODER_FOLDER,
        "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, filename) for filename in text_encoder_files],
        "system_configs": {
            "_name": "Text Encoder",
            "_infos": H3_TEXT_ENCODER_INFOS,
            "bf16": {"name": "Qwen3-VL BF16", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_BF16)]},
            "int8": {"name": "Qwen3-VL Quanto INT8", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_INT8)]},
            "nvfp4_awq": {"name": "Qwen3-VL NVFP4 AWQ", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_NVFP4)]},
            "gguf_q4_k_m": {"name": "Qwen3-VL GGUF Q4_K_M", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_GGUF_Q4)]},
            "gguf_q2_k": {"name": "Qwen3-VL GGUF Q2_K", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_GGUF_Q2)]},
        },
        "system_configs2": {
            "_name": "DiT Denoising Priority",
            "_default_label": "Lower VRAM",
            "_infos": H3_DENOISING_PRIORITY_INFOS,
            "lower_ram": {"name": "Lower RAM", "qkv_splitting": False},
        },
    }


class family_handler:
    @staticmethod
    def query_supported_types():
        return [FL2VA_ARCHITECTURE, FL2VA_PRUNED_ARCHITECTURE, CONTROL_ARCHITECTURE, CONTROL_PRUNED_ARCHITECTURE,
                REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE, TTS_REF2VA_PRUNED_ARCHITECTURE, VIGGLE_ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {
            FL2VA_PRUNED_ARCHITECTURE: FL2VA_ARCHITECTURE,
            CONTROL_ARCHITECTURE: FL2VA_ARCHITECTURE,
            CONTROL_PRUNED_ARCHITECTURE: FL2VA_ARCHITECTURE,
            REF2VA_ARCHITECTURE: FL2VA_ARCHITECTURE,
            REF2VA_PRUNED_ARCHITECTURE: FL2VA_ARCHITECTURE,
            VIGGLE_ARCHITECTURE: FL2VA_ARCHITECTURE,
        }, {}

    @staticmethod
    def query_model_family():
        return "minimax_h3"

    @staticmethod
    def query_family_infos():
        return {"minimax_h3": (70, "MiniMax H3")}

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("minimax_h3")

    @staticmethod
    def get_lora_dir(base_model_type):
        return "minimax_h3"

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        if cache_type == "first_block":
            skip_steps_cache.threshold = float(skip_steps_cache.multiplier)
        elif cache_type != "spectrum":
            raise ValueError(f"MiniMax H3 does not support step-skipping type {cache_type!r}")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        if base_model_type == VIGGLE_ARCHITECTURE:
            result = family_handler.query_model_def(REF2VA_PRUNED_ARCHITECTURE, model_def)
            result.pop("resolution_budget_tiled", None)
            result.update({
                "profiles_dir": [VIGGLE_ARCHITECTURE],
                "device_explicit": True,
                "specialities": [{"name": "character replacement"}, {"name": "motion transfer"}],
                "infos": VIGGLE_INFOS,
                "prompt_infos": "Viggle uses a fixed prompt. Prepare the character replacement in the Edited Reference Frame; generation prompt text is ignored.",
                "deepy_infos": "Viggle combines `video_guide` with one edited frame from that video in `image_refs` (reference mode `I`, Control Video mode `VU`). Edit the character while preserving that frame's pose, props, background, framing and dimensions; any clear source frame works. The video supplies motion/camera, the edited frame supplies appearance. Windows are fixed at 124 frames with 18-frame overlap by default. `audio_prompt_type`: empty = model audio; `A` = `audio_guide`; `K` = control-video soundtrack. Audio conditioning is experimental: use synchronized audio. A full input track is reused; a shorter one allows generated audio afterward.",
                "deepy_prompt_infos": "Express the replacement through the Edited Reference Frame. Viggle uses a fixed built-in prompt; generation text is ignored.",
                "text_encoder_URLs": [], "text_encoder_folder": None, "system_configs": {},
                "prompt_enhancer_def": {"selection": [], "labels": {}, "default": ""},
                "image_outputs": False, "v2i_switch_supported": False, "sliding_window": True, "video_continuation": False,
                "sliding_window_size_locked": True,
                "sliding_window_defaults": {**result["sliding_window_defaults"], "window_max": 124, "window_default": 124, "overlap_default": 18},
                "extract_guide_from_window_start": True, "control_video_trim_disabled": False, "control_video_trim": False,
                "image_prompt_types_allowed": "T", "end_frames_always_enabled": False, "image_end_frame_position": False,
                "guidance_max_phases": 1, "lock_guidance_phases": True, "lora_multiplier_phases": 1,
                "phase_2_spatial_tiling": False, "inference_steps2": False, "custom_settings": [], "sample_solvers": [("Euler", "euler")],
                "spectrum_cache": False, "first_block_cache": False,
                "one_image_ref_needed": True, "no_background_removal": True, "any_image_refs_relative_size": False, "fit_into_canvas_image_refs": 1,
                "image_ref_choices": {"choices": [("Use Edited Reference Frame", "I")], "letters_filter": "I", "default": "I", "label": "Edited Reference Frame"},
                "guide_custom_choices": {"choices": [("Use Control Video", "VU")], "letters_filter": "V-U", "default": "VU", "label": "Control Video"},
                "video_guide_label": "Control Video", "preprocess_video_guide2": False, "reference_video_enabled": False,
                "any_audio_prompt": True, "audio_prompt_choices": True, "output_audio_is_input_audio": True,
                "audio_guide_label": "Custom Audio",
                "audio_prompt_type_sources": {
                    "selection": ["", "A", "K"],
                    "labels": {"": "No Input Audio", "A": "Use Custom Audio", "K": "Reuse Control Video Audio"},
                    "letters_filter": "AK", "label": "Control Audio", "show_label": True, "default": "",
                },
            })
            return result
        if base_model_type in CONTROL_ARCHITECTURES:
            pruned = base_model_type == CONTROL_PRUNED_ARCHITECTURE
            result = family_handler.query_model_def(FL2VA_PRUNED_ARCHITECTURE if pruned else FL2VA_ARCHITECTURE, model_def)
            result.pop("resolution_budget_tiled", None)
            result["sample_solvers"] = [choice for choice in result["sample_solvers"] if choice[1] != PIKA_SOLVER]
            for key in ("guide_custom_choices", "custom_frames_injection", "one_image_ref_only", "switch_threshold"):
                result.pop(key)
            result.update({
                "profiles_dir": result["profiles_dir"] + [CONTROL_ARCHITECTURE],  # main blocks are FL2VA: its LoRA accelerator profiles apply
                "specialities": [{"name": "control video", "aliases": ["pose transfer", "depth control", "canny control", "controlnet"],
                                  "description": "Drive motion and structure with a pose, depth, edge, shape or grayscale control video."},
                                 {"name": "video inpainting", "description": "Regenerate a masked area of a video."},
                                 {"name": "video outpainting", "description": "Extend the video canvas using ControlNet's visible-source and border-mask conditioning."}],
                "infos": CONTROL_INFOS + H3_SPEED_INFOS + H3_STANDARD_SAMPLER_INFOS + H3_COMMON_RUNTIME_INFOS + (PRUNED_INFOS if pruned else ""),
                "deepy_infos": CONTROL_DEEPY_INFOS,
                "prompt_infos": FL2VA_PROMPT_INFOS, "deepy_prompt_infos": FL2VA_DEEPY_PROMPT_INFOS,
                "v2i_switch_supported": False, "image_outputs": False,
                "guidance_max_phases": 1, "lock_guidance_phases": True, "lora_multiplier_phases": 1, "phase_2_spatial_tiling": False, "inference_steps2": False,
                "custom_settings": [setting for setting in result["custom_settings"] if setting["id"] == H3_AUDIO_REFINEMENT_SETTING],
                "sliding_window_defaults": {**result["sliding_window_defaults"], "window_max": 362,
                                            **({"overlap_min": 5, "overlap_max": 107, "overlap_offset": 5, "overlap_default": 22, "discard_step": 17} if H3_CONTROL_LATENT_CONTINUATION else {})},
                "guide_preprocessing": {
                    "selection": ["", "PV", "DV", "EV", "SV", "CV", "V", "MV"],
                    "labels": {"V": "Use Control Video as is"},
                    "label": "Control Video",
                },
                "mask_preprocessing": {"selection": ["", "A", "NA"]},
                "control_net_weight_name": "Control", "control_net_weight_size": 1,
                "video_guide_label": "Control Video",
                "audio_prompt_type_sources": {**result["audio_prompt_type_sources"], "selection": ["", "A", "K"], "letters_filter": "AK"},
            })
            return result
        if base_model_type == TTS_REF2VA_PRUNED_ARCHITECTURE:
            return _get_audio_generator_model_def(model_def)
        reference_mode = base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE)
        pruned = base_model_type in (FL2VA_PRUNED_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE)
        vdn = model_def.get("vdn", False)
        pdd = model_def.get("pdd", False)
        ptd = model_def.get("ptd", False)
        if ptd and not (reference_mode and pdd):
            raise ValueError("PTD RefAsymOPD requires a Ref2VA architecture with PDD heads")
        veda = model_def.get("veda", False)
        fasth3 = model_def.get("fasth3", False)
        if fasth3 and not (reference_mode and pdd):
            raise ValueError("FastH3 Omni is a Ref2VA PDD student: declare it on the Ref2VA architecture with pdd")
        fixed_schedule = pdd or fasth3
        pika_supported = not (fixed_schedule or ptd or vdn or veda)
        text_encoder_variant = model_def.get("text_encoder_variant")
        text_encoder_files = [TEXT_ENCODER_BF16, TEXT_ENCODER_INT8] if text_encoder_variant is None else TEXT_ENCODER_VARIANTS[text_encoder_variant]
        result = {
            "dtype": "bf16",
            "device_explicit": True,
            "cuda_graphs": [] if veda or fasth3 else ["transformer"],
            "uint8_guides": _uint8_guides,
            "tiny_vae_architecture": REF2VA_ARCHITECTURE,
            "size": "lighter" if pruned else "large",
            **({"accelerated": "native"} if pdd or vdn or veda or fasth3 else {}),
            **({"specialities": [{"name": "character consistency", "aliases": ["identity preservation"]}, {"name": "motion transfer", "description": "Transfer motion or camera from reference videos to image-reference characters."}]} if reference_mode else {}),
            "fps": 24,
            "v2i_switch_supported": True,
            "image_batch_size_max": 1,
            "prompt_enhancer_video_duration": True,
            "frames_minimum": 107,
            "frames_steps": 17,
            "frames_offset": 5,
            "block_size": 32,
            "vae_block_size": 32,
            "guidance_max_phases": 1 if fixed_schedule else 2,
            "lock_guidance_phases": fixed_schedule,
            "visible_phases": 0,
            "lora_multiplier_phases": 1 if fixed_schedule else 2,
            "phase_2_spatial_tiling": not fixed_schedule,
            "custom_sigmas": not fixed_schedule and not ptd,
            "resolution_budget": "2048x2048",
            **({"resolution_budget_tiled": "3840x2176"} if not fixed_schedule else {}),
            "custom_settings": [{
                "id": H3_MASK_MODE_SETTING,
                "name": "Mask Denoising Mode",
                "label": "Mask Denoising Mode",
                "type": "dropdown",
                "default": H3_MASK_MODE_DEFAULT,
                "choices": [
                    ("Grouped Rows [conditioning timestep for fixed rows; denoising timestep for editable rows]", H3_MASK_MODE_GROUPED_ROWS),
                    ("Shared Timestep [same denoising timestep for fixed and editable latent rows]", H3_MASK_MODE_SHARED_TIMESTEP),
                ],
                "video_prompt_type": "G",
            }, {
                "id": H3_AUDIO_REFINEMENT_SETTING,
                "name": "Audio Refinement",
                "label": "Audio Refinement Extra Phase",
                "type": "dropdown",
                "default": "none",
                "choices": [
                    ("None", "none"),
                    ("Enabled (6 extra steps, denoising 0.5)", "enabled"),
                ],
                "audio_prompt_type_not": "S" if reference_mode else "AK",
            }],
            "switch_threshold": {
                "label": "Phase 2 Noise Level Start",
                "type": "number",
                "min": 0.7,
                "max": 1.0,
                "step": 0.0001,
            },
            "inference_steps": True,
            "inference_steps2": not fixed_schedule,
            "lock_inference_steps": fixed_schedule,
            "flow_shift": True,
            "spectrum_cache": True,
            "first_block_cache": True,
            "skip_steps_multiplier_choices": FIRST_BLOCK_CACHE_STRENGTHS,
            "skip_steps_multiplier_label": "First Block Cache Threshold",
            "first_block_cache_thresholds": FIRST_BLOCK_CACHE_THRESHOLDS,
            "custom_attention_modes": {
                "vdn": {"label": "VDN hybrid attention", "supports_sparsity": False, "installed": True, "supported": True},
            } if vdn else {
                "vsa": {"label": "VSA Video Sparse Attention (Triton, NVIDIA SM80+)", "supports_sparsity": False},
            } if fasth3 else {
                "veda": {"label": "Veda Learned Sparse Attention (Triton, NVIDIA SM80+)", "supports_sparsity": False},
            } if veda else {
                "sol": {"label": "Sol sparse attention, requires Triton and RTX 30xx or newer", "supports_sparsity": True},
            },
            "default_attention_modes_supported": not (vdn or veda or fasth3),
            "attention": {">=0": "vdn"} if vdn else {">=80": "vsa"} if fasth3 else {">=80": "veda"} if veda else None,
            "attention_sparsity": {
                "label": "Start Tau (higher = more sparse/faster; lower = more faithful; End Tau = 0.8)",
                "start": 0.0,
                "end": 4.0,
                "inc": 0.05,
            },
            "sample_solvers": [("Euler", "euler")] if fixed_schedule else [("Euler", "euler"), ("RES Multistep", "res_multistep"), ("Ralston 2S (~2x slower)", "ralston_2s")] + ([("Pika Re-noise (6 Steps)", PIKA_SOLVER)] if pika_supported else []),
            "no_negative_prompt": True,
            "returns_audio": True,
            "audio_window_alignment": True,
            "audio_overlap_repair_ms": 25,  # sources without an audio tail: one audio latent, plus the measured waveform shortfall
            "audio_tail_ms": 100,  # generate() returns audio running past the last frame (H3_AUDIO_TAIL_LATENTS)
            "end_image_before_discarded_frames": True,  # discarded last frames follow the end image of non-final windows
            "multimedia_generation": True,
            "image_end_frame_position": True,
            "control_video_trim_disabled": True,
            "extract_guide_from_window_start": True,  # sliding windows get pure window-aligned guide frames; continuation comes through input_video
            "prompt_enhancer_picture_labels": True,  # enhancer images are labelled with the <Picture N> the pipeline presents
            "infos": (H3_FASTH3_RUNTIME_INFOS if fasth3 else (REF2VA_INFOS if reference_mode else FL2VA_INFOS) + (H3_PDD_RUNTIME_INFOS if pdd else H3_RUNTIME_INFOS)) + (PRUNED_INFOS if pruned else "") + (H3_VDN_INFOS if vdn else "") + model_def.get("infos", ""),
            "prompt_infos": REF2VA_PROMPT_INFOS if reference_mode else FL2VA_PROMPT_INFOS,
            "prompt_enhancer_button_label": "Write",
            "prompt_enhancer_def": {
                "selection": ["T", "TI", "T1", "TI1"],
                "labels": {
                    "TV": "An H3 Reference Prompt from Text" if reference_mode else "An H3 Prompt from Text",
                    "TIV": "An H3 Reference Prompt from Text + {image_inputs}" if reference_mode else "An H3 Prompt from Text + {image_inputs}",
                    "T1P": "An H3 Image Prompt from Text",
                    "TI1P": "An H3 Image Prompt from Text + {image_inputs}",
                },
                "default": "",
            },
            "text_prompt_enhancer_instructions": REF2VA_TEXT_SYSTEM_PROMPT if reference_mode else FL2VA_TEXT_SYSTEM_PROMPT,
            "video_prompt_enhancer_instructions": REF2VA_IMAGE_SYSTEM_PROMPT if reference_mode else FL2VA_IMAGE_SYSTEM_PROMPT,
            "text_prompt_enhancer_max_tokens": 2048 if reference_mode else 1024,
            "video_prompt_enhancer_max_tokens": 2048 if reference_mode else 1024,
            "text_prompt_enhancer_instructions1": H3_STILL_TEXT_SYSTEM_PROMPT,
            "image_prompt_enhancer_instructions1": H3_STILL_IMAGE_SYSTEM_PROMPT,
            "text_prompt_enhancer_max_tokens1": 1024,
            "image_prompt_enhancer_max_tokens1": 1024,
            "profiles_dir": ["minimax_h3_vdn"] if vdn else ["minimax_h3_fasth3"] if fasth3 else [] if pdd else ["minimax_h3", "minimax_h3_ref2va" if reference_mode else "minimax_h3_fl2va"],
            "preset_profiles_dir": [] if fixed_schedule or vdn else ["minimax_h3_presets"],
            "finetune_custom_urls": ["video_vae_file", "audio_vae_file"],
            "vae_upsamplers": {X2_VAE_METHOD: [0, 1, 2]},
            "finetunes_infos": H3_FINETUNES_INFOS,
            "finetunes_params": H3_FINETUNES_PARAMS,
            TURBO_LORA_KEY: build_hf_url(REPO_ID, "loras", TURBO_LORA_FILE),
            REF_TURBO_LORA_KEY: build_hf_url(REPO_ID, "loras", REF_TURBO_LORA_FILE),
            "qkv_splitting": True,
            "qkv_layout": "interleaved",
            "keep_frames_video_guide_not_supported": True,
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, filename) for filename in text_encoder_files],
            "system_configs": {
                "_name": "Text Encoder",
                "_infos": H3_TEXT_ENCODER_INFOS,
                "bf16": {"name": "Qwen3-VL BF16", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_BF16)]},
                "int8": {"name": "Qwen3-VL Quanto INT8", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_INT8)]},
                "nvfp4_awq": {"name": "Qwen3-VL NVFP4 AWQ", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_NVFP4)]},
                "gguf_q4_k_m": {"name": "Qwen3-VL GGUF Q4_K_M", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_GGUF_Q4)]},
                "gguf_q2_k": {"name": "Qwen3-VL GGUF Q2_K", "text_encoder_URLs": [build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_GGUF_Q2)]},
            },
            "system_configs2": {
                "_name": "Video VAE",
                "_default_label": "Auto",
                "_infos": H3_VIDEO_VAE_INFOS,
                "bf16": {"name": "BF16", "video_vae_file": VIDEO_VAE_FILE, "x2_vae_file": X2_VAE_FILE},
                "fp8mix": {"name": "FP8 Mixed Precision", "video_vae_file": VIDEO_VAE_FP8MIX_FILE, "x2_vae_file": X2_VAE_FILE},
                "int8_convrot": {"name": "INT8 ConvRot Decoder", "video_vae_file": VIDEO_VAE_INT8_FILE, "x2_vae_file": X2_VAE_INT8_FILE},
                "lightvae": {"name": "LightVAE Lite BF16", "video_vae_file": VIDEO_VAE_LIGHT_FILE, "x2_vae_file": X2_VAE_FILE},
                "lightvae_int8": {"name": "LightVAE Lite INT8 ConvRot", "video_vae_file": VIDEO_VAE_LIGHT_INT8_FILE, "x2_vae_file": X2_VAE_INT8_FILE},
            },
            "system_configs3": {
                "_name": "DiT Denoising Priority",
                "_default_label": "Lower VRAM",
                "_infos": H3_DENOISING_PRIORITY_INFOS,
                "lower_ram": {"name": "Lower RAM", "qkv_splitting": False},
            },
        }
        if pdd:
            result["custom_settings"] = result["custom_settings"][:1]
        if reference_mode:
            result.update({
                "sliding_window": True,
                "video_continuation": True,
                "deepy_infos": REF2VA_DEEPY_INFOS,
                "deepy_prompt_infos": REF2VA_DEEPY_PROMPT_INFOS,
                "sliding_window_defaults": {"window_min": 124, "window_max": 481, "window_step": 17, "window_default": 362,
                                            "overlap_min": 1, "overlap_max": 120, "overlap_step": 17, "overlap_offset": 1, "overlap_default": 18,
                                            "discard_step": 1},
                "frames_selection_maximum": 737,
                "image_prompt_types_allowed": "TSEVL",
                "end_frames_always_enabled": True,
                "image_ref_choices": {
                    "choices": [("Generate without Reference Images", ""),
                                ("First Reference Image is the Main Subject / Landscape, defines Output Dimensions, and may be followed by other Reference Images", "KI"),
                                ("Use Reference Images", "I"),
                                ("Inject Frames at chosen Positions, any Images beyond the Positions are Reference Images", "FI")],
                    "letters_filter": "KFI",
                    "default": "",
                    "label": "Reference Images",
                },
                "custom_frames_injection": True,
                "reference_image_enabled": True,
                "return_image_refs_tensor": False,
                "fit_into_canvas_image_refs": 0,
                "any_image_refs_relative_size": True,
                "image_refs_relative_size": {"min": 50, "max": 400, "step": 1},
                "guide_custom_choices": {
                    "choices": [("Generate without a Reference or Control Video", ""), ("Use One Reference Video", "V-U"),
                                ("Use Two Reference Videos", "V+-U"),
                                ("Use Three Reference Videos", "V+*-U"),
                                ("Use Up to 3 Excerpts from same Reference Video", "V1-U"),
                                # ("Transfer Human Pose From Control Video", "PV"),
                                # ("Transfer Depth Map From Control Video", "DV"),  # superseded by the FL2VA ControlNet-Union model
                                # ("Transfer Edges Map From Control Video", "EV"),
                                ("Denoise Control Video (Video-to-Video Edit)", "GV"),
                                ("Use Control Video as In-Context Guide", "VU")],
                    "letters_filter": "GPDEV+*-U1",
                    "default": "",
                    "label": "Reference / Control Video",
                },
                "preprocess_video_guide2": True,
                "skip_video_guide_preprocess": "1",  # excerpts are decoded by the pipeline from the source file
                "mask_preprocessing": {"selection": ["", "A", "NA"]},
                "reference_video_enabled": True,
                "reference_video_max_frames": reference_video_frame_limit(1, 24),  # 362 frames (15 s)
                "reference_video_max_size": (768, 1344),
                "any_audio_prompt": True,
                "audio_prompt_choices": True,
                "audio_reference_max_total_duration": 15,
                "reference_audio_enabled": True,
                "video_guide_label": "Reference / Control Video 1",
                "video_guide2_label": "Reference Video 2",
                "video_guide3_label": "Reference Video 3",
                "audio_guide_label": "Audio Reference 1",
                "audio_guide2_label": "Audio Reference 2",
                "audio_guide3_label": "Audio Reference 3",
                "audio_prompt_type_sources": {
                    "selection": ["", "A", "AB", "ABD", "K", "K1", "AS", "KS"],
                    "labels": {
                        "": "Generate without an Audio Reference",
                        "A": "Use One Audio Reference",
                        "AB": "Use Two Audio References",
                        "ABD": "Use Three Audio References",
                        "K": "Use Reference Videos Soundtrack(s) as Audio References",
                        "K1": "Use Up to 3 Excerpts from Reference-Video Soundtrack",
                        "AS": "Generate Video based on Soundtrack and Text Prompt (Soundtrack Kept)",
                        "KS": "Generate Video based on Reference / Control Video + its Audio Track and Text Prompt (Soundtrack Kept)",
                    },
                    "letters_filter": "ABDKS1",
                    "label": "Audio References",
                    "show_label": True,
                    "default": "",
                },
                "audio_guide_window_slicing": True,
                "video_length_not_limited_by_audio": True,
            })
            result["custom_settings"] = result["custom_settings"] + H3_EXCERPT_SETTINGS
        else:
            result.update({
                "sliding_window": True,
                "video_continuation": True,
                "deepy_infos": FL2VA_DEEPY_INFOS,
                "deepy_prompt_infos": FL2VA_DEEPY_PROMPT_INFOS,
                "sliding_window_defaults": {"window_min": 124, "window_max": 481, "window_step": 17, "window_default": 362,
                                            "overlap_min": 1, "overlap_max": 120, "overlap_step": 17, "overlap_offset": 1, "overlap_default": 18,
                                            "discard_step": 1},
                "image_prompt_types_allowed": "TSEVL",
                "end_frames_always_enabled": True,
                "audio_guide_window_slicing": True,
                "guide_custom_choices": {
                    "choices": [("Generate without using a Control Video", ""),
                                ("Use Control Video as In-Context Guide", "VU"),
                                ("Denoise Control Video (Video-to-Video Edit)", "GV"),
                                ("Inject Frames", "KFI")],
                    "letters_filter": "GVUKFI",
                    "default": "",
                    "label": "Control Video / Frames Injection",
                },
                "video_guide_label": "Control Video",
                "mask_preprocessing": {"selection": ["", "A", "NA"]},
                "video_guide_outpainting": [0],
                "video_guide_outpainting_label": "Enable Spatial Outpainting on the H3 Control Video",
                "outpainting_quantize_margins": 32,
                "custom_frames_injection": True,
                "one_image_ref_only": True,
                "no_background_removal": True,
                "any_audio_prompt": True,
                "audio_prompt_choices": True,
                "audio_guide_label": "Source Audio / Soundtrack",
                "audio_prompt_type_sources": {
                    "selection": ["", "A", "K", "2"],
                    "labels": {
                        "": "Generate Video and Audio from Text Prompt",
                        "A": "Generate Video based on Soundtrack and Text Prompt (Soundtrack Kept)",
                        "K": "Generate Video based on Control Video + its Audio Track and Text Prompt (Soundtrack Kept)",
                        "2": "Generate Audio based on Control Video and Text Prompt",
                    },
                    "letters_filter": "AK2",
                    "label": "Audio Source",
                    "show_label": True,
                    "default": "",
                },
                "video_length_not_limited_by_audio": True,
                "output_audio_is_input_audio": True,
            })
        if fasth3:  # FastH3 Omni: the Ref2VA inputs, the student's own schedule and attention (locked above)
            result["deepy_infos"] = FASTH3_DEEPY_INFOS
        if pdd:
            result["deepy_infos"] += " PDD requires exactly 8 inference steps and the Euler sampler. LoRAs trained for the matching base model apply."
        else:
            result["deepy_infos"] += " Two-phase generation uses `num_inference_steps2` (default 3) for final enhancement and `switch_threshold` for its starting noise. Accelerator LoRAs before `|` with any nonzero phase-2 multiplier replace the automatic Turbo together, at their specified multipliers; if all are zero, the automatic Turbo is used. `1;0` means phase 1 only, `1;0.6` sets phase 2 to 0.6, and `1` applies to both phases. LoRAs after `|` keep their multipliers."
        if vdn:
            result["deepy_infos"] += " VDN loads its acceleration LoRA automatically and defaults to 8 steps."
        if result["custom_sigmas"]:
            result["custom_sigmas_schedulers"] = ["euler", "res_multistep", "ralston_2s"]
            result["deepy_infos"] += " `custom_sigmas` defaults to empty. Commas separate positive descending sigmas below 1; semicolons separate exactly `guidance_phases` slots. Blank slots use normal schedules; trailing zero is ignored. Explicit schedules override that phase's steps and starting sigma without another flow shift. FL2VA and Ref2VA share the H3 presets. Open-H3 Regenerate uses `;0.909375,0.8571428571428571,0.6666666666666666` and `open_h3_regenerate_2k_lora.safetensors` at `0;1` before `|`; it uses phase-1 audio as frozen phase-2 conditioning and retains phase-1 output audio. Tiling is experimental. The presets replace accelerator LoRAs while preserving additional LoRAs and their multipliers. Original Phase 2 and Vanilla Phase 1 reset presets clear custom sigmas."
        if veda:
            result["compile"] = False
            help_text = " Veda variants always use learned sparse attention in the H3 transformer; the attention override is locked to Veda. Other attention operations use the application attention backend. It reuses the standard H3 settings, LoRAs and step counts. The predictor follows Transformer Quantization (BF16 or INT8 ConvRot). Sparsity can change fine detail or motion; compare with a standard H3 variant at the same seed. Speed benefits are greatest for large videos. Requires Triton and an NVIDIA SM80+ GPU. Veda runs without CUDA graphs or transformer compilation."
            result["infos"] += "\n\n### Veda Sparse Attention\n\n" + help_text.strip()
            result["deepy_infos"] += help_text
        if pika_supported:
            result["infos"] += PIKA_INFOS
            result["deepy_infos"] += PIKA_DEEPY_INFOS
        still_infos = "\n\n**Text to Image:** generate one still image without audio. Video duration, extra frames and Audio Refinement do not apply in this mode."
        still_prompt_infos = "\n\nFor Text to Image, describe one still scene: its subject, composition, lighting and details. A plain-language image prompt is sufficient; speech and soundtrack instructions are unnecessary. The optional Write enhancer offers image prompts from text alone or from text plus the selected images. With an image, specify what to change and what to preserve."
        result["infos"] += still_infos
        result["deepy_infos"] += still_infos + " Set `image_mode` to `1`."
        result["infos"] += "\n\n**MiniMax H3 VAE:** " + X2_VAE_DESCRIPTION
        result["deepy_infos"] += " MiniMax H3 VAE replaces the default VAE and handles decoding and upsampling together. Choose `h3_vae*2` to double the output width and height, `h3_vae*1.5` to enlarge them by half, or `h3_vae*1` to keep the original size."
        result["deepy_infos"] += " Video VAE configs `lightvae` and `lightvae_int8` select LightVAE's 12-layer decoder with the original H3 encoder; decoding may change detail. VAE upsampling overrides this choice with the X2 decoder."
        result["prompt_infos"] += still_prompt_infos
        result["deepy_prompt_infos"] += still_prompt_infos
        if ptd:
            result.update({
                "infos": PTD_INFOS + "\n\n" + result["infos"],
                "deepy_infos": PTD_DEEPY_INFOS + "\n\n" + result["deepy_infos"],
                "profiles_dir": ["minimax_h3_pruned_ptd" if pruned else "minimax_h3_ptd"],
                "spectrum_cache": False, "first_block_cache": False, "flow_shift": False,
                "custom_settings": [setting for setting in result["custom_settings"] if setting["id"] != H3_AUDIO_REFINEMENT_SETTING],
            })
        return result

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        pika = inputs["sample_solver"] == PIKA_SOLVER
        if pika:
            if base_model_type not in (FL2VA_ARCHITECTURE, FL2VA_PRUNED_ARCHITECTURE, REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE) or any(model_def.get(flag, False) for flag in ("pdd", "ptd", "vdn", "veda", "fasth3")):
                return "Pika H3 Flex requires a standard FL2VA or Ref2VA model"
            if inputs["num_inference_steps"] != PIKA_STEPS or inputs["flow_shift"] != PIKA_VIDEO_SHIFT or inputs["guidance_phases"] != 1:
                return "Pika H3 Flex requires 6 steps, flow shift 12 and one generation phase; apply its matching accelerator profile"
            if inputs["custom_sigmas"] or inputs["skip_steps_cache_type"]:
                return "Pika H3 Flex requires empty Custom Sigmas and No Skipping"
        if model_def.get("ptd", False):
            inputs.update({"flow_shift": 12.0, "guidance_scale": 1.0, "guidance_phases": 1, "skip_steps_cache_type": ""})
        if model_def.get("veda", False):
            inputs["overrides"] = set_override(inputs["overrides"], "attention_mode", "veda")
        if model_def.get("fasth3", False):
            inputs["overrides"] = set_override(inputs["overrides"], "attention_mode", "vsa")
            if inputs["sample_solver"] != "euler":
                return "MiniMax H3 FastH3 requires the Euler sampler"
            inputs["num_inference_steps"] = FASTH3_STEPS
            image_refs = inputs.get("image_refs") or []
            video_prompt_type, audio_prompt_type = inputs.get("video_prompt_type") or "", inputs.get("audio_prompt_type") or ""
            videos = (inputs.get("video_guide") is not None) + (inputs.get("video_guide2") is not None) + (inputs.get("video_guide3") is not None) if "V" in video_prompt_type else 0
            audios = sum(flag in audio_prompt_type for flag in "ABD")
            if len(image_refs) > 9 or len(image_refs) + videos + audios > 12:  # the student's trained reference budget
                return "FastH3 Omni accepts at most nine image references and twelve references in total"
        if ((inputs.get("spatial_upsampling") or "").startswith(X2_VAE_METHOD) and int(inputs["image_mode"]) == 0
                and base_model_type not in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE, VIGGLE_ARCHITECTURE)
                and "2" in inputs["audio_prompt_type"]):
            return "MiniMax H3 VAE Upsampling requires generated video; Audio from Control Video preserves the input frames and does not decode video latents"
        if base_model_type == VIGGLE_ARCHITECTURE:
            if inputs["video_guide"] is None:
                return "Viggle-Animate requires a Control Video and one edited frame from that video"
            inputs["video_prompt_type"] = inputs["video_prompt_type"].replace("-", "")
            scheduler = inputs.get("frame_scheduler")
            if inputs["sliding_window_size"] > 124 or (scheduler is not None and scheduler["active"] and any(window["frame_num"] > 124 for window in scheduler["windows"])):
                return "Viggle-Animate supports at most 124 frames per sliding window"
        audio_generator = base_model_type == TTS_REF2VA_PRUNED_ARCHITECTURE
        if audio_generator:
            try:
                duration = float(inputs["duration_seconds"])
            except (TypeError, ValueError):
                return f"MiniMax H3 maximum total audio duration must be a number between 4 and {H3_DIALOGUE_MAX_TOTAL_SECONDS:g} seconds"
            if not 4 <= duration <= H3_DIALOGUE_MAX_TOTAL_SECONDS:
                return f"MiniMax H3 maximum total audio duration must be between 4 and {H3_DIALOGUE_MAX_TOTAL_SECONDS:g} seconds"
            audio_prompt_type = inputs["audio_prompt_type"]
            audios = [inputs["audio_guide"]] if "A" in audio_prompt_type else []
            if "B" in audio_prompt_type:
                audios.append(inputs["audio_guide2"])
            if "D" in audio_prompt_type:
                audios.append(inputs["audio_guide3"])
            import librosa

            audio_durations = []
            for index, audio in enumerate(audios, 1):
                try:
                    audio_duration = float(librosa.get_duration(path=os.fspath(audio)))
                except Exception as error:
                    return f"Unable to read Audio Reference {index}: {error}"
                if not audio_duration >= 2:
                    return f"Audio Reference {index} must be at least 2 seconds long (found {audio_duration:.2f}s)"
                audio_durations.append(audio_duration)
            _notify_audio_reference_limit(audio_durations)
            return None
        if model_def.get("pdd", False):
            required_steps = PDD_NUM_STEPS // PDD_BLOCK_SIZE
            if inputs["sample_solver"] != "euler":
                return "MiniMax H3 PDD requires the Euler sampler"
            inputs["num_inference_steps"] = required_steps
        overlap, error = normalize_overlap(int(inputs["sliding_window_overlap"] or 0), 17, 5 if H3_CONTROL_LATENT_CONTINUATION and base_model_type in CONTROL_ARCHITECTURES else 1)
        if error:
            return error
        inputs["sliding_window_overlap"] = min(overlap, 107) if H3_CONTROL_LATENT_CONTINUATION and base_model_type in CONTROL_ARCHITECTURES else overlap
        from shared.utils.utils import get_outpainting_dims

        grouped_masking = base_model_type not in CONTROL_ARCHITECTURES and h3_grouped_masking_enabled(inputs.get("custom_settings"))
        outpainting = get_outpainting_dims(inputs.get("video_guide_outpainting"), inputs.get("video_guide_outpainting_ratio", "")) is not None
        if outpainting and base_model_type not in CONTROL_ARCHITECTURES and not grouped_masking:
            return "MiniMax H3 outpainting requires Mask Denoising Mode to be set to Grouped Rows"
        if grouped_masking and get_override(inputs["overrides"], "attention_mode") == "sol":
            masked_control = inputs.get("video_mask") is not None or outpainting or "A" in (inputs.get("video_prompt_type") or "")
            if masked_control:
                return "MiniMax H3 Grouped Rows mask denoising is not compatible with Sol Attention; select Shared Timestep or another attention mode"
        if "~" in (inputs["video_prompt_type"] or ""):
            from .pipeline import H3_PHASE_2_TILE_COUNT, _spatial_tiles

            width, height = map(int, inputs["resolution"].split("x"))
            rows, columns = _spatial_tiles(height), _spatial_tiles(width)
            gr.Info(f"MiniMax H3 phase 2 tiling: {H3_PHASE_2_TILE_COUNT} tiles of {columns[0][1]}x{rows[0][1]} pixels (2x2 grid) for a {width}x{height} output.")
        if base_model_type not in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE, VIGGLE_ARCHITECTURE):
            video_prompt_type = inputs["video_prompt_type"]
            audio_prompt_type = inputs["audio_prompt_type"]
            if base_model_type in CONTROL_ARCHITECTURES and "V" in video_prompt_type:
                preprocessed = any(letter in video_prompt_type for letter in "PDESCM")
                if outpainting and preprocessed and not ("A" in video_prompt_type and inputs.get("video_mask") is not None):
                    return "MiniMax H3 ControlNet outpainting requires Use Control Video as is with the original video, or a preprocessing mode with a Video Mask to keep the original picture outside the mask"
                if inputs.get("video_mask") is not None and "A" in video_prompt_type and not preprocessed and not outpainting:
                    return "MiniMax H3 ControlNet masks need a preprocessing mode (Pose, Depth, Canny, Shapes, Recolorize or Inpainting) so the source video can be kept outside the mask"
                if "M" in video_prompt_type and (inputs.get("video_mask") is None or "A" not in video_prompt_type):
                    return "MiniMax H3 ControlNet inpainting requires a Video Mask"
            if "F" in inputs["video_prompt_type"]:
                position_count = sum(pos.upper() != "X" for pos in (inputs["frames_positions"] or "").replace(",", " ").split())
                image_count = len(inputs["image_refs"] or [])
                if position_count != image_count:
                    return f"MiniMax H3 frame injection requires one position per Reference Image (found {position_count} positions and {image_count} images)"
            if "2" in audio_prompt_type:
                if "A" in audio_prompt_type or "K" in audio_prompt_type:
                    return "MiniMax H3 audio generation from Control Video cannot also use a source soundtrack"
                if "G" not in video_prompt_type or "V" not in video_prompt_type or inputs["video_guide"] is None:
                    return "MiniMax H3 audio generation from Control Video requires Denoise Control Video and a Control Video file"
            if "K" in audio_prompt_type:
                if "V" not in video_prompt_type or inputs["video_guide"] is None:
                    return "MiniMax H3 Control Video soundtrack mode requires a Control Video mode and a Control Video file"
                from shared.utils.audio_video import extract_audio_tracks

                try:
                    if extract_audio_tracks(inputs["video_guide"], query_only=True) == 0:
                        return "MiniMax H3 Control Video has no audio track"
                except Exception as error:
                    return f"Unable to inspect the MiniMax H3 Control Video soundtrack: {error}"
            return None

        video_prompt_type = inputs["video_prompt_type"]
        audio_prompt_type = inputs["audio_prompt_type"]
        image_count = len(inputs["image_refs"] or [])
        if "F" in video_prompt_type:
            if "G" in video_prompt_type:
                return "MiniMax H3 frame injection cannot be combined with Denoise Control Video"
            position_count = sum(pos.upper() != "X" for pos in (inputs["frames_positions"] or "").replace(",", " ").split())
            if position_count == 0 or position_count > image_count:
                return f"MiniMax H3 frame injection requires at least one position and one Reference Image per position (found {position_count} positions and {image_count} images)"
            image_count -= position_count  # images beyond the injected frames are reference images
        videos = []
        if "V" in video_prompt_type and "G" not in video_prompt_type:
            if inputs["image_mode"] > 0:
                image_count += 1  # image_guide supplies the reference in still-image mode.
            else:
                videos.append(inputs["video_guide"])
                if "+" in video_prompt_type:
                    videos.append(inputs["video_guide2"])
                if "*" in video_prompt_type:
                    videos.append(inputs["video_guide3"])
        soundtrack = "S" in audio_prompt_type or base_model_type == VIGGLE_ARCHITECTURE  # Viggle's A/K inputs are also kept soundtracks, not audio references
        audios = [inputs["audio_guide"]] if "A" in audio_prompt_type and not soundtrack and base_model_type != VIGGLE_ARCHITECTURE else []
        if "B" in audio_prompt_type:
            audios.append(inputs["audio_guide2"])
        if "D" in audio_prompt_type:
            audios.append(inputs["audio_guide3"])

        if image_count > 9:
            return "MiniMax H3 Ref2VA accepts at most 9 reference images"
        if len(videos) > 3:
            return "MiniMax H3 accepts at most 3 reference videos"

        from shared.utils.utils import get_video_info

        video_seconds = []
        for index, video in enumerate(videos, 1):
            try:
                fps, _, _, frames = get_video_info(video)
                duration = frames / fps
            except Exception as error:
                return f"Unable to read Reference Video {index}: {error}"
            if duration < 2:
                return f"Reference Video {index} must be at least 2 seconds long (found {duration:.2f}s)"
            video_seconds.append(duration)
        custom_settings = inputs["custom_settings"] or {}
        fps = model_def["fps"]
        video_durations = [min(duration, reference_video_frame_limit(1, fps) / fps) for duration in video_seconds]
        if "1" in video_prompt_type:
            try:
                video_durations = [duration for _, duration in parse_excerpts(custom_settings.get(H3_VIDEO_EXCERPTS_SETTING), fps, video_seconds[0], "Reference Positions", video_frames=True)]
            except ValueError as error:
                return str(error)
        elif sum(video_durations) > reference_video_frame_limit(1, fps) / fps:
            limit = reference_video_frame_limit(len(video_durations), fps) / fps
            gr.Info(f"MiniMax H3 reference videos total {sum(video_durations):.2f}s, above the 15s limit. Each video will keep its first {limit:.2f}s.")
            video_durations = [min(duration, limit) for duration in video_durations]
        elif "-" in video_prompt_type and video_seconds and video_seconds[0] > video_durations[0]:
            gr.Info(f"MiniMax H3 reference video lasts {video_seconds[0]:.2f}s. Only its first {video_durations[0]:.2f}s will be used.")

        soundtrack_mode = "K" in audio_prompt_type
        if soundtrack_mode:
            control_soundtrack = soundtrack and "V" in video_prompt_type and "-" not in video_prompt_type  # KS can keep a control video's own soundtrack (In-Context or Denoise)
            soundtrack_videos = [inputs["video_guide"]] if control_soundtrack else videos if "-" in video_prompt_type else []
            video_kind = "Control" if control_soundtrack else "Reference"
            if not soundtrack_videos:
                return "Keeping a video soundtrack requires a Reference Video or a Control Video" if soundtrack else "Using video soundtracks as audio references requires a Reference Video"
            audio_excerpts = "1" in audio_prompt_type
            if "1" in video_prompt_type and not (soundtrack or audio_excerpts):
                return "Reference-video excerpts cannot also be used as audio references; select Use Up to 3 Excerpts from Reference-Video Soundtrack instead"
            if len(videos) > 1 and (soundtrack or audio_excerpts):
                return "Keeping a reference-video soundtrack or taking excerpts from it requires a single Reference Video (or excerpts from one)"
            from shared.utils.audio_video import extract_audio_tracks

            for index, video in enumerate(soundtrack_videos[:1] if soundtrack else soundtrack_videos, 1):
                try:
                    if extract_audio_tracks(video, query_only=True) == 0:
                        return f"{video_kind} Video {index} has no audio track"
                except Exception as error:
                    return f"Unable to inspect the soundtrack of {video_kind} Video {index}: {error}"
            if audio_excerpts:
                try:
                    audio_durations = [duration for _, duration in parse_excerpts(custom_settings.get(H3_AUDIO_EXCERPTS_SETTING), fps, video_seconds[0], "Audio Reference Positions")]
                except ValueError as error:
                    return str(error)
            else:
                audio_durations = [] if soundtrack else video_durations
            audio_count = len(audio_durations)
        else:
            import librosa

            audio_durations = []
            for index, audio in enumerate(audios, 1):
                try:
                    duration = float(librosa.get_duration(path=os.fspath(audio)))
                except Exception as error:
                    return f"Unable to read Audio Reference {index}: {error}"
                if not duration >= 2:
                    return f"Audio Reference {index} must be at least 2 seconds long (found {duration:.2f}s)"
                audio_durations.append(duration)
            audio_count = len(audios)

        if audio_count > 3:
            return "MiniMax H3 accepts at most 3 audio references"
        _notify_audio_reference_limit(audio_durations)
        visual_count = image_count + len(video_durations)
        if pika and not visual_count:
            return "Pika H3 Flex Ref2VA requires at least one image or video reference"
        if not pika and audio_count > visual_count:
            return f"MiniMax H3 requires at least as many reference images and videos as audio references (found {visual_count} visual and {audio_count} audio)"
        file_count = image_count + len(videos) + (0 if soundtrack_mode else audio_count)
        if file_count > 12:
            return f"MiniMax H3 accepts at most 12 reference files (found {file_count})"
        return None

    @staticmethod
    def resolve_runtime_model_def(model_def, runtime_context):
        """Resolve Auto once for both asset downloads and model loading."""
        if "video_vae_file" in model_def or model_def.get("system_configs2", {}).get("_name") != "Video VAE":
            return model_def
        filename = {"int8": VIDEO_VAE_INT8_FILE, "fp8": VIDEO_VAE_FP8MIX_FILE}.get(runtime_context["transformer_quantization"], VIDEO_VAE_FILE)
        return {**model_def, "video_vae_file": filename, "x2_vae_file": X2_VAE_INT8_FILE if filename == VIDEO_VAE_INT8_FILE else X2_VAE_FILE}

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        source_folders = []
        file_lists = []
        if model_def.get("ptd", False):
            source_folders.append(PTD_FOLDER)
            file_lists.append([PTD_HEAD_FILE])
        vae_files = []
        video_vae_file = model_def.get("video_vae_file", VIDEO_VAE_FILE)
        if video_vae_file in (VIDEO_VAE_FILE, VIDEO_VAE_FP8MIX_FILE):
            vae_files.append(video_vae_file)
        if video_vae_file in (VIDEO_VAE_INT8_FILE, VIDEO_VAE_LIGHT_FILE, VIDEO_VAE_LIGHT_INT8_FILE):
            source_folders.append("minimax_h3")
            file_lists.append([video_vae_file.rsplit("/", 1)[-1]])
        if "audio_vae_file" not in model_def:
            vae_files.append(AUDIO_VAE_FILE)
        if vae_files:
            source_folders.append("")
            file_lists.append(vae_files)
        if base_model_type != VIGGLE_ARCHITECTURE:
            source_folders.append(TEXT_ENCODER_FOLDER)
            file_lists.append(["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "vocab.json"])
            source_folders.append(LATENT_UPSCALER_FOLDER)
            file_lists.append([LATENT_UPSCALER_FILE])
        downloads = [{
            "repoId": REPO_ID,
            "sourceFolderList": source_folders,
            "fileList": file_lists,
        }]
        if video_vae_file in (X2_VAE_FILE, X2_VAE_INT8_FILE):
            downloads.append(query_x2_vae_files(video_vae_file))
        if base_model_type == VIGGLE_ARCHITECTURE:
            downloads.append({"repoId": VIGGLE_REPO_ID, "sourceFolderList": [VIGGLE_ASSET_FOLDER], "fileList": [[VIGGLE_PROMPT_FILE]]})
        if base_model_type == TTS_REF2VA_PRUNED_ARCHITECTURE and H3_DIALOGUE_GENERATION:
            from shared.deepy.assets import whisper_download_def

            downloads.append(whisper_download_def())
        return downloads

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer=False,
                   text_encoder_quantization=None, dtype=torch.bfloat16,
                   mixed_precision_transformer=False, save_quantized=False, submodel_no_list=None,
                   text_encoder_filename=None, shared_h3_pipeline=None, shared_h3_offloadobj=None,
                   disable_pinning=False, VAE_upsampling=None, **kwargs):
        from .minimax_h3_main import model_factory

        pdd = model_def.get("pdd", False)
        viggle = base_model_type == VIGGLE_ARCHITECTURE
        if model_def.get("ptd", False):
            model_filename = list(model_filename) if isinstance(model_filename, (list, tuple)) else [model_filename]
            model_filename.append(fl.locate_file(os.path.join(PTD_FOLDER, PTD_HEAD_FILE)))
        video_vae_file = model_def.get("video_vae_file", VIDEO_VAE_FILE)
        if VAE_upsampling == X2_VAE_VALUE:  # the X2 decoder of the selected Video VAE replaces it; downloaded once both choices are known
            video_vae_file = model_def.get("x2_vae_file", X2_VAE_FILE)
            process_files_def_if_needed(query_x2_vae_files(video_vae_file), status_text="Downloading MiniMax H3 X2 VAE")
        pipeline = model_factory(model_filename, text_encoder_filename, dtype=dtype,
                                 reference_mode=base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE, TTS_REF2VA_PRUNED_ARCHITECTURE, VIGGLE_ARCHITECTURE),
                                 save_quantized=save_quantized, model_type=model_type,
                                 qkv_splitting=model_def["qkv_splitting"],
                                 qkv_layout=model_def["qkv_layout"],
                                 video_vae_filename=video_vae_file,
                                 audio_vae_filename=model_def.get("audio_vae_file", AUDIO_VAE_FILE), shared_h3_pipeline=shared_h3_pipeline,
                                 pdd=pdd, pdd_num_steps=PDD_NUM_STEPS if pdd else None, pdd_block_size=PDD_BLOCK_SIZE if pdd else None,
                                 vdn=model_def.get("vdn", False), veda=model_def.get("veda", False), vsa=model_def.get("fasth3", False),
                                 audio_only=base_model_type == TTS_REF2VA_PRUNED_ARCHITECTURE,
                                 fixed_prompt_filename=os.path.join(VIGGLE_ASSET_FOLDER, VIGGLE_PROMPT_FILE) if viggle else None)
        pipeline.ptd = model_def.get("ptd", False)
        pipe = {"transformer": pipeline.transformer}
        if base_model_type == TTS_REF2VA_PRUNED_ARCHITECTURE and H3_DIALOGUE_GENERATION:
            pipeline.dialogue_whisper = load_dialogue_whisper()
            pipe["dialogue_whisper"] = pipeline.dialogue_whisper
        if shared_h3_pipeline is None:
            pipe.update({
                "vae": pipeline.video_decoder,
                "video_encoder": pipeline.video_encoder,
                "audio_vae": pipeline.audio_vae,
            })
            if not viggle:
                pipe.update({"text_encoder": pipeline.text_encoder.language_model, "vision_encoder": pipeline.text_encoder.visual,
                             "latent_upscaler": pipeline.latent_upscaler})
        else:
            class BorrowingPipe(dict):
                pass

            borrowed_names = ("text_encoder", "vision_encoder", "vae", "video_encoder", "audio_vae", "latent_upscaler")
            pipe = BorrowingPipe(pipe)
            pipe.update({name: shared_h3_offloadobj.models[name] for name in borrowed_names})
            pipe._mmgp_ignore_models = borrowed_names
        return pipeline, {"pipe": pipe, "pinnedMemory": False} if disable_pinning else pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.pop("input_video_strength", None)
        if model_def.get("veda", False):
            ui_defaults["overrides"] = set_override(ui_defaults.get("overrides", ""), "attention_mode", "veda")
        elif get_override(ui_defaults.get("overrides", ""), "attention_mode") == "veda":
            ui_defaults["overrides"] = set_override(ui_defaults["overrides"], "attention_mode", "")
        if model_def.get("fasth3", False):
            ui_defaults["overrides"] = set_override(ui_defaults.get("overrides", ""), "attention_mode", "vsa")
        elif get_override(ui_defaults.get("overrides", ""), "attention_mode") == "vsa":
            ui_defaults["overrides"] = set_override(ui_defaults["overrides"], "attention_mode", "")
        if base_model_type == TTS_REF2VA_PRUNED_ARCHITECTURE:
            return
        if base_model_type in CONTROL_ARCHITECTURES:
            # The former VU/UV outpainting choice is now V plus the existing spatial-outpainting settings.
            ui_defaults["video_prompt_type"] = ui_defaults.get("video_prompt_type", "").replace("U", "")
        if settings_version < 2.75:
            ui_defaults["switch_threshold"] = H3_PHASE_2_NOISE_LEVEL_START_DEFAULT
        if settings_version < 2.74:
            ui_defaults["attention_sparsity"] = 1.3
        if settings_version < 2.73 and "sliding_window_overlap" in ui_defaults:
            overlap = max(1, int(ui_defaults["sliding_window_overlap"] or 18))
            ui_defaults["sliding_window_overlap"] = normalize_overlap(overlap, 17, 1)[0]
        if settings_version < 2.73 and base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE):
            ui_defaults["sliding_window_size"] = 362
        if settings_version < 2.71:
            ui_defaults["denoising_strength"] = 1.0
        if settings_version < 2.70:
            cache_value = float(ui_defaults.get("skip_steps_multiplier", 0.08))
            ui_defaults["skip_steps_multiplier"] = LEGACY_FIRST_BLOCK_CACHE_THRESHOLDS.get(cache_value, cache_value)
        if settings_version < 2.69:
            encoder, priority, _, finetune = (str(ui_defaults.get("config", "")).split(",") + [""] * 4)[:4]
            ui_defaults["config"] = ",".join((encoder, "", priority, finetune)).rstrip(",")
        if base_model_type == VIGGLE_ARCHITECTURE:
            ui_defaults["sliding_window_size"] = 124
            ui_defaults["sliding_window_overlap"] = ui_defaults.get("sliding_window_overlap", 18) or 18
            ui_defaults["video_prompt_type"] = ui_defaults.get("video_prompt_type", "IVU").replace("-", "")
        if base_model_type in CONTROL_ARCHITECTURES and "sliding_window_overlap" in ui_defaults:
            overlap, error = normalize_overlap(int(ui_defaults["sliding_window_overlap"] or 0), 17, 5 if H3_CONTROL_LATENT_CONTINUATION else 1)
            if error is None:
                ui_defaults["sliding_window_overlap"] = min(overlap, 107) if H3_CONTROL_LATENT_CONTINUATION else overlap
        if base_model_type not in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE):
            return
        if settings_version < 2.67:
            ui_defaults["image_refs_relative_size"] = 100
            ui_defaults["video_prompt_type"] = ui_defaults.get("video_prompt_type", "").replace("G", "")
        if settings_version < 2.68 and "V" in ui_defaults.get("video_prompt_type", "") and "-" not in ui_defaults["video_prompt_type"]:
            ui_defaults["video_prompt_type"] += "-"
        if settings_version < 2.76:
            video_prompt_type = ui_defaults.get("video_prompt_type", "")
            if "V" in video_prompt_type and not any(flag in video_prompt_type for flag in "PDEG+-"):
                video_prompt_type += "-"
            if "V" in video_prompt_type and any(flag in video_prompt_type for flag in "+-") and "U" not in video_prompt_type:
                video_prompt_type += "U"
            ui_defaults["video_prompt_type"] = video_prompt_type

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.pop("input_video_strength", None)
        if base_model_type == VIGGLE_ARCHITECTURE:
            family_handler.update_default_settings(REF2VA_PRUNED_ARCHITECTURE, model_def, ui_defaults)
            ui_defaults.update({"num_inference_steps": 3, "flow_shift": 3.0, "video_prompt_type": "IVU",
                                "sliding_window_size": 124, "sliding_window_overlap": 18, "prompt_enhancer": ""})
            return
        if base_model_type == TTS_REF2VA_PRUNED_ARCHITECTURE:
            ui_defaults.update({
                "video_length": 0,
                "duration_seconds": 15,
                "num_inference_steps": 20,
                "guidance_phases": 1,
                "guidance_scale": 1.0,
                "flow_shift": 12.0,
                "sample_solver": "euler",
                "attention_sparsity": 1.3,
                "skip_steps_start_step_perc": 25,
                "skip_steps_multiplier": 0.08,
                "audio_prompt_type": "A",
                "multi_prompts_gen_type": "FG",
            })
            return
        reference_mode = base_model_type in (REF2VA_ARCHITECTURE, REF2VA_PRUNED_ARCHITECTURE)
        ui_defaults.update({
            "video_length": 124,
            "sliding_window_size": 362,
            "sliding_window_overlap": 18,
            "num_inference_steps": 20,
            "guidance_phases": 1,
            "num_inference_steps2": 3,
            "custom_sigmas": "",
            "switch_threshold": H3_PHASE_2_NOISE_LEVEL_START_DEFAULT,
            "guidance_scale": 1.0,
            "flow_shift": 12.0,
            "sample_solver": "euler",
            "attention_sparsity": 1.3,
            "skip_steps_start_step_perc": 25,
            "skip_steps_multiplier": 0.08,
            "denoising_strength": 1.0,
            "audio_prompt_type": "",
            "video_prompt_type": "",
            "image_mode": 0,
        })
        if reference_mode:
            ui_defaults.update({"image_refs_relative_size": 100, "remove_background_images_ref": 0})
        if model_def.get("ptd", False):
            ui_defaults.update({"num_inference_steps": 8, "video_prompt_type": "I", "prompt_enhancer": "", "skip_steps_cache_type": ""})
        if model_def.get("fasth3", False):
            ui_defaults.update({"num_inference_steps": FASTH3_STEPS, "flow_shift": FASTH3_VIDEO_SHIFT, "guidance_phases": 1, "sample_solver": "euler"})
        if base_model_type in CONTROL_ARCHITECTURES:
            ui_defaults.update({"video_prompt_type": "PV", "control_net_weight": 1.0, "sliding_window_overlap": 22 if H3_CONTROL_LATENT_CONTINUATION else 18})
