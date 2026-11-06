# Processing

Processing is everything WanGP does around generation to prepare inputs, control what the model produces, or refine media after it exists.

Use this guide when you are deciding how to prepare a Control Video, build masks, use reference images, extend a video, or apply late postprocessing.

## At A Glance

| Stage | When It Happens | What It Does |
| --- | --- | --- |
| Preprocessing | Before generation | Converts source media into model-ready inputs such as pose, depth, masks, control frames, or cleaned references. |
| Generation-time processing | During generation | Uses temporal, spatial, mask, reference, and sliding-window controls to decide what the model should change or preserve. |
| Postprocessing | After generation | Improves or transforms completed images, videos, or audio with upscaling, remuxing, audio tools, or late media processing. |

## Before Generation: Preprocessing

Preprocessing changes what the model receives as conditioning. It does not produce the final media by itself; it prepares the inputs that guide generation.

Common preprocessing jobs:

- Resize and resample inputs to the model frame rate and output resolution.
- Extract pose, depth, canny, raw-frame, mask, or other control information from a Control Video.
- Prepare Video Masks for masked-area or non-masked-area workflows.
- Prepare reference images before inserting people, objects, landscapes, or exact frames.
- Build injected reference frames for temporal inpainting or spatial outpainting.
- Prepare source media for video continuation or long-video sliding windows.

### Control Videos

Control Video processing tells the model what structure, motion, or area to follow.

- **Keep Unchanged:** use the Control Video as the source structure without extracting a new annotation.
- **Transfer Human Motion:** extract pose or Open Pose style motion from a source video.
- **Depth / Canny / Similar Structure Controls:** extract scene structure while letting the prompt change appearance.
- **Inpainting:** mark areas that should be regenerated.
- **Outpainting:** add missing space around existing frames.

If a Video Mask is provided, the selected process can apply only to the masked area or only to the non-masked area. This lets you keep a subject while changing the background, replace one person among several people, or combine different effects inside and outside the mask.

A Control Video keeps its timing. At a rate within 0.1 frames per second of the model's, such as 23.976 for a 24 fps model, WanGP reads every frame without repeating or dropping any, and it records the result and places the sound on the video's real rate, so they stay together however long the video is.

A part of a video named without encoding it (a view, or a path ending in `|start_frame=...,end_frame=...`) can be any length, even one the model cannot render. WanGP fills the render by playing the part's last frames backwards, then trims the output back, so you get exactly its frames. Asking for clearly more than the part holds still extends it by temporal outpainting.

WanGP previews processed Control Video and Video Mask thumbnails before generation. Use `--save-masks` when you need to inspect the full generated control media.

### Masks

Masks decide where a process applies.

- Black areas are usually preserved.
- White areas are usually processed.
- Some model families support colored masks to identify several people or objects independently.

For person or object replacement, a stable mask across frames is usually more important than a perfect mask on one frame.

Matanyone is useful for creating Video Masks:

1. Load the source video.
2. Select the face, person, object, or region in the first frame.
3. Validate the mask.
4. Generate the matting video.
5. Export the source video and mask back into the Control Video and Video Mask inputs.

Mask tips:

- Use negative point prompts to remove unwanted parts from a selection.
- Use sub masks when one selection is too hard to define cleanly.
- Use mask expansion or shrinking when the generated replacement needs more or less room than the original subject.

### Reference Images

Reference images can represent a subject, object, landscape, style source, or exact frame position. Their role should be clear in the prompt.

Use foreground cleanup for:

- People or objects that should be inserted into a generated scene.
- References where the original background should not influence the output.
- Character or object swaps where a clean foreground improves identity and placement.

Avoid foreground cleanup for:

- Landscape, room, or setting references.
- Background images that should define the environment.
- Injected frames that should appear as complete frames at a specific position.

If automatic cleanup is not precise enough, use the Image version of Matanyone to create a cleaner foreground mask.

## During Generation: Temporal And Spatial Processing

Generation-time processing changes what the model adds, keeps, or regenerates while it is producing the output.

### Temporal Processing

Temporal processing controls frames over time.

- **Temporal Outpainting:** adds frames before or after existing media. It is implicit when you continue a Source Video or provide a Control Video shorter than the requested output.
- **Temporal Inpainting:** fills missing frames between existing frames.
- **Injected Reference Frames:** places specific images at chosen frame positions so the model can fill the gaps between them.
- **Frames To Keep:** preserves selected Control Video frames and regenerates the others.

When a model supports positioned frames, frame positions are counted from the first frame of the active timeline. In sliding-window workflows, this usually means the first frame of the first window.

### Spatial Processing

Spatial processing controls image regions or canvas size.

- **Spatial Outpainting:** adds new content to the top, bottom, left, or right of existing frames.
- **Inpainting:** replaces masked image regions.
- **Mask Expansion / Shrinking:** adjusts the editable area when the replacement subject is larger or smaller than the original target.
- **Reference Image Placement:** uses injected frames or reference images to guide what appears in newly generated areas.

Spatial outpainting targets the resolution you request. If the requested resolution is the same as the source media, the original content may lose detail because canvas space is being reallocated. For outpainting, a higher output resolution is often useful.

### Long Videos Workflows

Sliding windows let WanGP generate longer videos by merging multiple generation windows.

The longer the video, the more likely quality is to degrade over time: artifacts can accumulate and character identity may progressively drift. The effect is usually less visible when the generated video mostly preserves unmodified control-video content. If the model supports end frames, a useful way to limit drift is to prepare several end-frame images in advance with an image-generation model and assign them to successive windows. These visual anchors steer later windows back toward the intended characters, composition, and appearance.

A reliable way to create related end frames is to generate one strong master image first, then use an image-editing model several times on that same master image to produce the planned keyframes. Ask each edit for the pose, camera angle, action, environment, or composition required at the end of its matching video window while preserving the original character identity and style. Always branch each edit from the master image rather than editing the previous keyframe in sequence, because chained edits can accumulate visual drift before video generation even begins.

Some uncommon models, including MiniMax H3 and VACE, can also re-inject reference images or frames between windows. Supplying suitable intermediate references gives the model recurring visual anchors and can restore character identity, appearance, or composition before drift becomes too noticeable. The available Reference Images or Injected Frames controls and their placement rules depend on the selected model.

There are two ways to build a multi-window video:

- **Plan all windows in one generation:** provide multiple prompts in advance, with one prompt for each window. This usually gives the best technical quality because WanGP keeps the windows in memory and performs the final merge and video encoding only once. The tradeoffs are higher RAM usage and the need to decide the sequence in advance.
- **Continue an existing video repeatedly:** use each completed video as the source for another continuation. This supports improvisation because the next prompt can be chosen after reviewing the latest result. However, every continuation reads an already encoded video and produces another encoded result, so repeated lossy re-encoding can slightly degrade quality at each stage.

Avoid making a window larger than the model can handle reliably. Larger windows take longer and require more VRAM, and every model has a practical or declared frame limit. On the other hand, a reasonably long window gives the model more room to preserve motion and character identity within that window. Choose the longest window that remains reliable for the selected model and available hardware rather than automatically using either the maximum or a very short window.

When sliding windows are enabled, positional arguments such as injected-frame positions or frames to keep are usually relative to the first frame of the first window. This lets you change the sliding-window size without recalculating all frame positions.

If you continue from a Source Video, the Source Video is treated as the first window. The overlap size controls how many frames from that source or previous window are used to condition the next generated segment.

How sliding windows work:

- Each window uses the matching time segment of the Control Video.
- Example: 0-4s control video -> first window, 4-8s -> second window.
- Overlap management helps smooth transitions between windows.

Generated frame count formula:

```text
Generated Frames = [Nb Windows - 1] x [Window Size - Overlap - Discard] + Window Size
```

Multi-line prompts can be used when the model supports assigning one prompt line per sliding window. If there are more windows than prompt lines, the last prompt line repeats.

Choose how window prompts are separated with **How to Process each Line of the Text Prompt**:

- **Each Line Will be used for a new Sliding Window of the same Video Generation:** every non-empty line starts a new window prompt.
- **Each Paragraph Separated by an Empty line will be used for a new Sliding Window of the same Video Generation:** every paragraph separated by an empty line starts a new window prompt. Use this choice when an individual window prompt contains line breaks of its own.

Each window prompt may begin with optional `[/...]` commands. Use `[/overlap=9]` to select a different overlap for that transition, or `[/overlap]` to restore the model's default overlap. On text-to-video-capable models, `[/overlap=0]` creates a hard cut by preventing the preceding window from conditioning the new one; `[/new_shot]` is its more explicit alias. Different window lengths can be requested with `[/duration=121]`, `[/duration=5s]`, or `[/duration=20%]`. When duration commands are present, they define the window schedule and predicted final length instead of treating the UI's Number of Frames as a strict final cap; windows without an explicit duration consume the remaining requested frames, up to the configured Sliding Window Size. Commands can be combined, for example `[/duration=4s,/new_shot]`.

When generating sound with LTX-2, MiniMax H3 or Ovi, WanGP aligns each window's audio to the video timeline and replaces a short ending of the preceding soundtrack with newly generated overlap. This also applies to Continue Video: the source's ending may be replaced even if it came from another model or was never padded. Keep the model's default overlap; if WanGP reports that it is too short for audio repair, increase it to an allowed value. This prevents accumulated timing drift; Ovi still generates each window's sound independently, so sound or voice continuity is not guaranteed. MiniMax H3 generates a little sound past each window's last frame, so its windows join without replacing anything, and the videos it saves keep about 0.1 s of sound after the last frame: continuing such a video reuses its whole soundtrack. Players may show these videos 0.1 s longer, holding the last frame.

A control or source soundtrack selected for preservation keeps its original audio. Audio supplied only as a generation reference does not imply preservation. A deliberate cut without overlap can retain a short silent gap if the preceding generated soundtrack ends early.

With these models, if the generated soundtrack ends slightly before the final video frame, WanGP fills the remaining ending with silence when saving. This keeps the complete video; that export padding is not reused to condition the next sliding window. H3 native continuation requires the source soundtrack to reach close to the video's end; WanGP reports when the gap is too long to repair.

Sliding-window settings:

- **Window Size:** choose a duration that fits the motion and model quality.
- **Overlap Frames:** use enough overlap for continuity, but not so much that blur propagates.
- **Discard Last Frames:** remove weak final frames from each window when a model tends to blur at the end. With MiniMax H3 any number of frames can be discarded; when a window has an end image and another window follows, that end image is generated just after the discarded frames. It steers the action without being shown, which hides the jump onto an end image the action did not reach (about 5 frames is enough). The last window still ends exactly on its end image.
- **Add Overlapped Noise:** may help some workflows, but can reduce quality in others.

## After Generation: Postprocessing

Postprocessing is applied after media exists. It can run immediately after generation or later from the Media Info / Late Post Processing area.

Use postprocessing when the content is good but the final media needs a technical or finishing pass.

Common postprocessing jobs:

- Spatial upsampling for images or videos.
- Video frame interpolation or temporal upsampling.
- Audio postprocessing, voice replacement, or soundtrack remuxing.
- Cropping, resizing, or importing existing media for a later workflow.
- Applying a system postprocessing model through MediaFlow.

SeedVC voice replacement, also used by Scenema Audio for reference voices, is distributed separately under the GPL-3.0 license. WanGP downloads it automatically from https://github.com/deepbeepmeep/WanGP-SeedVC the first time you use it, so that first use needs internet access.

Late postprocessing is useful when you want to improve existing media without regenerating the original scene.

### Face Refinement

Face refiners find and follow up to five faces, restore each one and paste it back: the rest of the picture and the video's resolution do not change. Set *Faces to Refine* to `0` to let WanGP pick the relevant faces.

- **H3 Face Refiner** restores faces with the H3 model, a text description and the soundtrack, which the model hears as it does when generating. Mouths stay closer to the source's, and it reuses H3 when H3 is already loaded.
- **TaoMate-FVSR Face Refiner** restores speaking and singing faces from the soundtrack: lips and teeth come out sharp and follow the audio, with more skin and eye detail. It takes no prompt and pushes the mouth toward the audio much harder than H3 Face Refiner, so it is the one to try when a mouth looks wrong, and the mouth can differ more from the source.

| | H3 Face Refiner | TaoMate-FVSR Face Refiner |
|---|---|---|
| **Choose it for** | Keeping the source's mouth movements; small or distant faces | Close-ups; a mouth that looks wrong |
| **Detail** | Less skin, eye and teeth detail on close-ups | Sharpest lips and teeth, most skin and eye detail on close-ups |
| **Mouth** | Stays close to the source; does little to correct a wrong mouth | Pushed toward the soundtrack: closed lips may open, and a face following the soundtrack mouths its words even if that person is only listening |
| **Rest of the face** | Expressions and the upper face can change | On close-ups the upper face and identity stay close to the source; on small or distant faces, mouths open wider and the face is less faithful than with H3 |
| **Control** | Accepts a prompt (Late Post Processing) | No prompt; you choose which faces follow the soundtrack |
| **Speed and memory** | Faster with several faces, a little less VRAM; reuses H3 when it is already loaded | Faster on a single close-up; faces large in the frame take several times longer; SeedVR2 downloads with it |

TaoMate-FVSR choices:

- **Faces Following the Soundtrack:** a face following the soundtrack mouths its words even if that person is only listening. *Largest Face* (the default) suits one speaker with others around; use *All Faces* when everyone sings or speaks together, and *No Face* to restore detail from the picture alone. A video without audio is always restored from the picture alone.
- **Face Refinement Strength** blends the result over the source. Below 1 the original mouth can show through the new one, so keep it at 1 unless you see a change you dislike.
- **Optional Face Reference Images** (Late Post Processing) work as for H3 Face Refiner: each picture is matched to the person it shows, and gives the model the clean detail of that face. Without one, the images the video was generated from are used when its file holds them. A person left without a picture uses the best frame of their own track: the largest, sharpest and most frontal. If that frame is damaged too, some of the damage can remain: supply a clean picture of that person.
- **Reference Restorer** (in *Configuration / Extensions*) restores that frame, or a small supplied picture, before the model uses it. SeedVR2 is the default and downloads with the refiner; with *None* faces stay softer.
- **Working Size** (in *Configuration / Extensions*): *Auto* works at 512 and moves to 1024 for faces large in the frame. 1024 takes several times longer and more VRAM.

Limitations: new facial detail is an estimate, not the person's true appearance, and the mouth may differ from the source because it follows the audio. Video frame rates near 25 fps match its training best, and clear speech gives the most reliable lips; music and very fast cuts are less predictable.

### Detail Refinement Up To 8K

**LTX 2.5 Detail Refiner** rebuilds fine texture, edges and natural grain in videos that look soft: compressed web video, clips upscaled with a basic resizer, or generated videos that lost detail in the VAE. Framing, motion, exposure and color stay those of the source.

- **Scale:** x1 only restores detail, x1.5 and x2 also enlarge the video. The best results come from 1080p sources refined to 4K. The refiner accepts at most x2 of the source; to reach 8K, refine to 4K first and then refine the 4K result again.
- **Speed:** the video is processed in overlapping 1024x576 areas that are refined together at every step, so there are no seams, but the work grows with the output size: FullHD needs 9 areas per step, 4K needs 49 and 8K 196. Expect minutes for FullHD and much longer for 4K or 8K.
- **Style prompt:** describe the rendering only, for example sharpness, texture, grain or lighting. Every area of the frame receives the whole prompt, so naming a subject can paint it where it does not belong.
- **Gentle Mode** (enabled by default in *Configuration / Extensions*) starts from the source video and runs three steps: it is faster and stays closer to the source. Disable it to rebuild detail from scratch in eight steps when the source is very soft.
- **Temporal Window** (97 frames by default) is the number of frames refined together. Shorter windows use less VRAM.
- **Tiles per Model Call** (*Auto* by default) is how many areas are refined in one pass of the model. Lower it to cap VRAM usage; the result is identical, only the speed changes. Very large or long outputs are also kept in system RAM automatically instead of VRAM.
- **Media Flow:** the *LTX 2.5 Detail Refiner* process offers the same x1, x1.5 and x2 *Upsampling* choices. Outputs larger than 4320p (8K) are refused with the largest multiplier that fits. Each chunk starts from the refined end of the previous chunk, so content stays continuous across chunk boundaries; on very fine texture such as grass or foliage, a slight change of texture can remain visible at a boundary. A small `.ltx25_refine_cache.safetensors` file is kept next to an unfinished output so a stopped run resumes just as smoothly; it is removed when the output completes. Keep it with the video if you move a partial output you plan to resume.

Limitations: text and logos that are heavily degraded may become shapes that only resemble letters; foam, fireworks or wind-blown foliage can look over-sharpened; archive footage becomes cleaner but can lose some mid-level texture. Save 4K and 8K results with a high-quality video codec setting, otherwise compression hides the added detail.

## Checklists

Before generation:

- Preview processed Control Video and Video Mask thumbnails.
- Confirm masks apply to the intended region.
- Check that reference images match the role described in the prompt.
- Use `--save-masks` when the preview is not enough.

Before a long sliding-window run:

- Test a short window first.
- Confirm frame positions are measured from the intended timeline start.
- Pick enough overlap for continuity.
- Avoid excessive overlap when blur propagates.

After generation:

- Check that outputs are visually usable, not only valid files.
- Use late postprocessing when resolution, audio, or motion needs refinement.
- Keep the original generation output when trying multiple postprocessing passes.

## Troubleshooting

If windows are inconsistent:

- Increase overlap frames.
- Keep prompts consistent across windows.
- Reduce discard frames if too much useful content is lost.
- Add overlapped noise only if it helps the specific workflow.

If results are blurry:

- Reduce excessive overlap.
- Increase discarded end frames for models that blur at window ends.
- Use higher-quality reference images or control videos.
- Verify that preprocessing did not resize the useful region too aggressively.

If masks do not behave as expected:

- Inspect the preview thumbnails.
- Save masks and control videos with `--save-masks`.
- Rebuild the mask with cleaner selections or sub masks.
- Expand or shrink the mask when the replacement subject needs different space.

---

> Applies to: Control media, masks, reference preparation, long-video sliding windows, duration schedules, overlaps and post-processing. Procedures describe interface controls; programmatic access uses the API and MCP interfaces.
