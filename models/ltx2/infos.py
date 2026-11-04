from models.help_shared import specialities


def compact_help(*, msr, joy, editanything, ltx25):
    prompt = 'Describe a chronological audiovisual scene: subjects, actions, framing, light, audible events, voice and exact quoted dialogue. Keep speech feasible within the clip. With a start image, describe motion consistent with it. Prompt enhancement is optional.'
    info = "Generate synchronized video/audio from text or start/end images; continuation and sliding windows extend output. Keep the selected full/distilled sampling preset."
    strengths = ["audio-video generation"]
    if msr:
        info = "MSR uses reference images only: background-first mode = 2–5 images (scene + up to 4 subjects); subject-only mode = 1–4 subjects. Prefer white/neutral subject backgrounds or enable background removal. Character sheets help. No control video, pose/depth/edges, masks, outpainting or frame injection. Generate audio or supply a soundtrack. MSR V2 reference length Auto is recommended and does not set output duration."
        prompt += " Assign each reference its role, describe subject interaction and retain consistent identity/clothing."
        strengths += ["identity preservation"]
    elif joy:
        info = "JoyAI-Echo builds connected story windows with reusable visual/voice memories. Start/continue source applies to the first window. Prefer one phase. Memories are not saved automatically: explicitly store named moments; up to 7 can be active. Control Video Memory can supply reference moments from a video with sound, not full-video editing."
        prompt = 'One story window per line/paragraph matching Text Prompt processing. Repeat stable character/voice IDs and appearances. Commands: [/duration=5s], [/new_shot], [/overlap=9], [/store_mem=alice], [/load_mem=alice,bob], [/load_mem=] clears active memory, [/drop_mem=alice] removes it. Without load_mem, active memories carry forward. Example: `[/duration=5s,/store_mem=alice] ID_A, a woman in a red coat, says "Welcome" in a quiet courtyard.` Later: `[/new_shot,/load_mem=alice] ID_A in the same red coat turns toward the gate.` Read full prompt help for timing/memory details.'
        strengths += ["multi-shot video", "identity preservation"]
    else:
        info += " Audio modes: new soundtrack, supplied audio, control video with its audio, or audio generation for raw control video, as offered by the selected entry."
        if editanything:
            info = "EditAnything reference video editing requires a source/control video and one reference image. Describe the reference subject to add or edit and the source details to preserve. Use the selected full/distilled preset and the entry's offered audio modes."
            prompt = "Describe the desired edited scene, the reference image's role, the action and the source details to preserve. State changes explicitly; keep requested audio and quoted dialogue feasible within the clip."
            strengths += ["video editing"]
        else:
            info += " Control-video processing offers pose/depth/canny/raw guidance; selected 22B workflows also offer masks and Ingredients references. Raw-video audio generation must not be combined with other visual controls."
            if not ltx25:
                info += " Reference Voice uses the ID-LoRA mode when offered."
            strengths += ["control video"]
    return {"deepy_infos": "### LTX Audio and Video\n\n" + info, "deepy_prompt_infos": "### Prompt\n\n" + prompt, "specialities": specialities(*strengths)}


LTX2_INFOS = """
# LTX2 Workflows

## What The Model Can Do

- Text to video with soundtrack: write a cinematic prompt and LTX2 generates both the video and an audio track.
- Image or video continuation: provide a Start Image, End Image, or Video to Continue to keep identity, framing, or motion continuity.
- Control Video / Frames Injection: guide the new video with motion, structure, raw frames, HDR conversion, outpainting, or injected reference frames.
- Inpainting: 22B can regenerate masked regions of a Control Video while preserving the unmasked video context.
- Ingredients Reference Sheet: 22B can use one composite reference-sheet image with the Ingredients IC-LoRA to keep characters, props, and location consistent.
- EditAnything variants: provide a source/control video plus one reference image to add or edit a subject in the video.

## VAE Decoder Choices

Choose the decoder from the `VAE` system configuration:

- `Default VAE`: the original decoder and default choice. It offers the best balance of speed, quality, and VRAM use for normal generation.
- `PrunaAI VAE (faster, slightly worse quality)`: an optimized alternative available for both LTX-2.3 and LTX-2.5.
- `NAD Diffusion Decoder (slower, higher VRAM, better motion)`: an alternative with improved motion at the cost of more time and VRAM. It supports tiled decoding and is available for both LTX-2.3 and LTX-2.5.

## Text To Image Mode

Image mode returns one image.

Image quality, reference identity, and control-image adherence can often be improved from the Quality tab with `Generate more frames to preserve Reference Image Identity / Control Image Information or improve`.

Higher values can improve image quality and reference adherence, but take more time and memory. Pose, depth and canny controls may increase this value automatically to meet their minimum requirement; the result is still one image.

## Control Video Processes

- `No Video Process`: the Control Video is not used as a visual guide.
- `Transfer Human Motion`: extracts body pose/motion from the Control Video.
- `Transfer Human Motion With Pose Alignment`: extracts pose and aligns it to your Start Image, Video to Continue, or background reference.
- `Transfer Depth`: follows the depth and scene layout of the Control Video.
- `Transfer Canny Edges`: follows strong edges and outlines from the Control Video.
- `LTX2 Raw Format / Control Video for Ic Lora`: uses the Control Video frames directly. Use this for IC-LoRA style control, to provide the video to be outpainted, and for `Generate Audio based on Control Video`.
- `Inpaint Masked Area`: 22B only. Uses the Control Video plus a Video Mask to regenerate the masked area. This mode requires `Control Video Strength` set to `1` and `Unmasked Area Strength` set to `0`; the unmasked area is preserved by the inpainting workflow.
- `Ingredients Reference Sheet`: 22B only. Duplicates one uploaded reference-sheet image as the IC-LoRA guide video; use a clean composite sheet on a white background, with black separator lines between individual pieces and without text.
- `Convert SDR to HDR (IC-LoRA)`: 22B only. Converts an SDR Control Video into an HDR video (10-bit HDR10 output). The text prompt is ignored: the conversion follows the Control Video.
- `Inject Frames`: places selected Reference Images at exact frame positions. In `Positions of Injected Frames`, `1` means the first frame and `L` means the last frame of a sliding-window segment. Each `X` skips a window without consuming an image.

## Audio Options

- `Generate Video & Soundtrack based on Text Prompt`: no audio file is needed. The prompt drives both the visuals and the generated soundtrack.
- `Generate Video based on Soundtrack and Text Prompt`: upload an Audio Prompt to guide timing, rhythm, speech, or sound events. If you leave it blank, WanGP uses null audio so the model is not driven by a real soundtrack. When the audio covers the generated window, it is normally reused as the final output audio.
- `Generate Video based on Control Video + its Audio Track and Text Prompt`: the audio track is extracted from the Control Video and used like the soundtrack prompt. This requires a Control Video with an audio track. When that track covers the generated window, it is normally reused as the final output audio.
- `Generate Audio based on Control Video and Text Prompt`: generates audio from a raw Control Video plus the text prompt. It requires `LTX2 Raw Format / Control Video for Ic Lora` and is not compatible with pose/depth/canny/HDR/outpainting/mask/injected-frame modes.
- `Generate Video based on Reference Voice (ID-LoRA) and Text Prompt`: available when the ID-LoRA option is listed. Provide a reference voice/audio sample and describe the on-camera speaker and speech in the prompt.
- `Prompt Audio Strength`: controls how strongly the uploaded soundtrack/audio prompt affects the generated result.
- `Ignore Background Music`: removes or reduces background music/noise from the audio prompt before using it for conditioning.
- `Postprocess Remux Audio`: after generation, you can replace or reuse the final soundtrack with a Custom Soundtrack, MMAudio, or the Control Video audio track.

## New Content Generation

If the Control Video or Control Audio has a shorter duration than the number of frames to generate, LTX2 will complete with new video and audio content.

## Control Video & Audio Timing With Video Continuation

When you use an Audio Prompt or the Control Video audio track, WanGP slices the audio per sliding window so each window receives the matching part of the soundtrack.

Sliding Window overlap frames are used for continuity and then removed from the final stitched video. Their matching audio is also used only to help the next window start smoothly.

If you continue from a Video to Continue, the alignment dropdown decides where Control Video, Control Audio, and Positioned Frames start on the timeline:

- `Aligned to the beginning of the Source Video`: frame/time 0 means the first frame of the source video you are continuing. Use this when your Control Video, audio prompt, or positioned frame numbers include the original source video at the beginning.
- `Aligned to the beginning of the First Window of the new Video Sample`: frame/time 0 means the first newly generated part after the source video. Use this when your Control Video, soundtrack, or positioned frame numbers are only meant for the continuation.

Example: if you continue a 4 second source video and your soundtrack starts with the new action, choose `Aligned to the beginning of the First Window of the new Video Sample`. If your soundtrack starts with those same 4 source seconds before the new action, keep `Aligned to the beginning of the Source Video`.

# Advanced

## Changing System LoRA 

LTX2 automatically adds some system LoRAs when a feature needs them. Each system LoRA comes with predefined LoRA Multipliers that may adjust to the context. 
You can change the LoRA used for a system process and the LoRA multipliers that will used for this LoRA.
A selected LoRA will be automatically used instead of a system LoRA if it contains in its name a text sequence signature.

Recognized system LoRA signatures:
- `distilled-lora`: distilled stage LoRA used by dev models for two-phase, Distilled 8 Steps, HQ/res2s, and some ID-LoRA cases.
- `union-control`: IC-LoRA used by Pose, Pose Alignment, Depth, and Canny control.
- `ic-lora-hdr`: HDR IC-LoRA used by LTX-2.3 HDR output.
- `ic-lora-sdr-to-hdr`: SDR-to-HDR IC-LoRA used by LTX-2.5 HDR output.
- `ic-lora-outpaint`: outpainting IC-LoRA used by 22B legacy spatial outpainting.
- `in-outpainting`: inpainting/outpainting IC-LoRA used by 22B mask-based inpainting and new outpainting.
- `ic-lora-ingredients`: Ingredients IC-LoRA used by the 22B Ingredients Reference Sheet process.
- `id-lora-celebvhq`: ID-LoRA used by the reference voice workflow.

To change one of these, manually select a LoRA whose filename contains the same recognized signature, then set its multipliers in the LoRAs tab. Because your selected LoRA has the recognized signature, WanGP skips the automatic default and uses your selected one instead.

Use a single number for one weight, such as `0.7`. Use `phase1;phase2` for two-phase generation, such as `0;1`, `1;0`, or `0.25;0.5`.


Examples:

```text
Select: custom-distilled-lora-v2.safetensors
Multiplier: 0.3;0.7
Filename rule: it contains "distilled-lora"
Result: the dev model uses your custom distilled LoRA instead of the default distilled LoRA and and it will apply your choices of LoRA multipliers.
```

```text
Select: ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
Multiplier: 0.6;0.2
Result: Pose/Depth/Canny control uses your manual phase weights instead of the automatic union-control weight.
```

```text
Select: my-ic-lora-outpaint-experiment.safetensors
Multiplier: 0.8
Filename rule: it contains "ic-lora-outpaint"
Result: 22B outpainting uses your outpaint LoRA instead of the built-in one.
```


```text
Select: my-id-lora-celebvhq-voice.safetensors
Multiplier: 1;0
Filename rule: it contains "id-lora-celebvhq"
Result: the reference voice workflow uses your ID-LoRA file and weight.
```
"""

LTX2_SOL_INFOS = """
## Sol-Attn Sparse Attention

In **Advanced Mode > Misc. > Override Attention Mode**, select **sol** to route the large self-attention blocks of the LTX2 video stream (sequences of 8192 tokens or more with 128-dim heads) through the bundled Triton Sol-Attn kernels. The **Start Tau** slider then appears below the attention selector and shows that End Tau is fixed at `0.8`: the value is used on the first denoising step of each phase and moves to `0.8` on the final step, in five levels, because CUDA graphs keep one graph per value. The default is `1.0`, the starting value the Sol-Attn paper gives: increase it to route more attention blocks through the approximate path for greater speed, or lower it for denser attention and higher fidelity. The first third of the denoising steps of each phase runs with full attention, so the face and the shape are built densely first. The tokens that carry conditioning keep full attention as well: a reference video, a keyframe, a start image and a Multiple Subject Reference are attended to exactly, which keeps the identity of the person, and only the generated frames take the approximate path, for about twice the speed on RTX 30/40/50-series cards. A window with no conditioning at all uses the fastest path. When the conditioning tokens of a window are not one leading run, or fill more than a quarter of it, that window keeps full attention: the sparse path has nothing left to win there. Cross-attention, audio self-attention, masked self-attention, and short sequences always keep the regular attention backend. Sol-Attn applies to the LTX2 video stream only, and requires BF16, Triton 3.6 or newer, and a CUDA NVIDIA GPU using SM86, SM89, SM90, SM100, SM120, or SM121 (such as RTX 30/40/50-series, H100/H200, B100/B200, or DGX Spark); the dropdown reports whether it is available on the current system. Do not combine it with a step-skipping mode without checking the result on the same seed.
"""

LTX2_25_INFOS = LTX2_INFOS + """
Reference Voice (ID-LoRA) is unavailable for LTX-2.5.

## SDR to HDR Conversion (LTX-2.5)

Select `Convert SDR to HDR (IC-LoRA)` and provide the SDR clip as Control Video. LTX-2.5 uses its own SDR-to-HDR IC-LoRA, which recovers highlight range and detail that an SDR grade clipped or compressed. The result is saved as a 10-bit HDR10 video; SDR white is placed at the standard 203-nit reference level, and only the recovered highlights go brighter.

- No prompt is needed: the conversion is driven entirely by the Control Video, and the text prompt is ignored.
- The conversion always runs as one pass of 8 distilled steps at the selected resolution, without guidance. Steps, guidance, sampler and two-phase settings have no effect. The Dev model automatically adds the distilled LoRA for this mode.
- Choose `Control Video` in `Force FPS` to keep the frame rate of the source clip. Frame rates above 30 fps are supported: the model's internal timing is capped at 30 fps, but the output keeps the source frame rate.
- For the best quality, select `NAD Diffusion Decoder` in the `VAE` configuration. Every 24 or 32 frames the model also generates a full-detail keyframe, and only the NAD decoder uses these keyframes when decoding the video. Other decoders still convert the video, without that extra detail.
- Long clips are converted by sliding windows. Each window continues from the previous HDR frames, and Continue Video also works with HDR sources.
- Start / End Images, Video Masks, Inject Frames, pose / depth / canny control and outpainting cannot be combined with HDR conversion.
- Decoding needs more VRAM than denoising, especially with the NAD decoder. If decoding runs out of memory, enable VAE tiling or convert shorter windows.
"""

LTX2_25_DEEPY_INFOS = """Generate video and synchronized sound from `prompt`. `image_start` / `image_end` anchor the opening / ending; `video_source` continues video. Sliding windows carry overlapping video and audio forward.

Control Video (`video_guide`) supplies raw frames or a selected pose/depth/edge guide; higher Control Video Strength follows it more closely. Inpainting uses source + mask: Control Video Strength 1 and Unmasked Area Strength 0 preserve unmasked content. Inject Frames places ordered `image_refs` at explicit positions (`1` = first frame, `L` = last in the window, `X` = skip a window without consuming an image). Ingredients uses one composite reference sheet for characters, objects and setting.

`audio_prompt_type`: empty = generate soundtrack; `A` = condition on `audio_guide`; `K` = control video and its audio; `2` = generate audio for control frames. A complete input soundtrack is normally reused; a shorter one allows audio continuation. Make action and speech agree with soundtrack timing.

For continuation, the alignment selector places controls/injected frames relative to source-video time zero or the new continuation. Use capabilities for window limits and `prompt_infos` for speech and timed prompting.

SDR to HDR: `video_prompt_type` `V&G` with the SDR clip as `video_guide` gives an HDR10 video. The prompt is ignored, and the mode always uses 8 distilled steps without guidance. `force_fps` `control` keeps the source frame rate, and the `NAD Diffusion Decoder` VAE config gives the best quality. Start/End images and masks are not supported.
"""

LTX2_MSR_INFOS = """
# LTX2 Multiple Subject Reference

This model is configured for the LiconStudio Multiple Subject Reference LoRA. It uses reference images only; Control Video, pose, depth, canny, HDR, outpainting, masks, and frame injection are disabled for this workflow.

## Reference Image Order

Use one of the `MSR Reference Images` modes:

- `Background + Up to 4 Subjects`: upload 2 to 5 images. Image 1 is the background or scene reference; images 2 to 5 are the subject or object references to preserve.
- `Subjects / Objects only`: upload 1 to 4 subject or object references on a plain white or neutral background.

Refer to images in their displayed upload order when describing their roles. Keep the scene first in background mode.

## Reference Image Preparation

Subject or object references work best on a plain white background. If your non-background references are not already isolated on white, keep `Remove Background behind MSR Subjects / Objects` enabled so WanGP removes the subject background and places it on white.

Character sheets are recommended for character references: use an image that shows the same character from several points of view, poses, or close-up/detail angles. This gives MSR more identity and clothing information than a single portrait.

Use the text prompt to describe how the referenced subjects should appear together in the referenced environment.
"""

LTX2_25_MSR_INFOS = LTX2_MSR_INFOS + """

## LTX-2.5 MSR V1

This preset combines LTX-2.5 Distilled with LiconStudio MSR V1 for scenes containing several referenced subjects. The reference preparation does not set the output video length.

Start with 8 sampling steps and describe each reference's role, the subjects' actions, and their spatial relationships. The model can generate sound with the video or use a supplied soundtrack. Prompt enhancement is optional and off by default.
"""

LTX2_MSR_V2_INFOS = LTX2_MSR_INFOS + """

## **MSR Reference Video Length** (MSR V2 only)

Leave this on `Auto` for normal use. WanGP automatically gives each uploaded subject enough reference space and keeps the background separate:

- 1 subject: `17`
- 2 subjects: `33`
- 3 subjects: `49`
- 4 subjects: `65`

Choose `17` to `65` manually only when you want to experiment with reference strength or reduce memory use. This setting changes reference influence; it does not change the generated video length.

MSR V2 keeps subject references separate from the background and generally preserves identities and multi-subject scenes better than V1.
"""
