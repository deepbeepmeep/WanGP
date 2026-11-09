# Prompts Guide

This page explains how WanGP interprets the main text prompt: how multiline prompts are split or preserved, how prompt lines can be paired with multiple images, how a local or remote Prompt Enhancer changes the text, and how macros generate prompt variations.

## Prompt Types In Practice

In WanGP, the main prompt is always the text you write in the `prompt` box.

Depending on the model, that text can be used alone or together with:

- `Start Image`, `End Image`, or `Reference Images`
- `Control Video` or `Source Video`
- `Audio Source`, `Audio Prompt (Soundtrack)`, or `Speaker reference voice`

Even when a model also uses image, audio, or video conditioning, the text prompt still tells the model what you want to happen.

The same text box is used across very different kinds of models, so the best prompt style depends on the model you chose.

### Text-only models

Examples:

- `Wan2.1 Text2video 14B`
- `Qwen Image 20B`
- `TTS HeartMuLa OSS 3B`

Practical use:

- For `Wan2.1 Text2video 14B`, the prompt is your whole scene description.
- For `Qwen Image 20B`, the prompt can also describe posters, signs, or images with a lot of visible text, because this model is especially good at rendering long text inside images.
- For `TTS HeartMuLa OSS 3B`, the prompt is usually lyrics, while the extra tags field steers the musical style.

### Text + image models

Examples:

- `Wan2.1 Image2video 480p 14B`
- `Wan2.2 Image2video Enhanced Lightning v2 14B`

Practical use:

- The image gives the model the subject, appearance, and composition.
- The text prompt should focus on motion, action, camera movement, and scene evolution.

This is why image-to-video prompts are often more useful when they say things like:

```text
the woman smiles, turns toward the window, and the camera slowly pushes in
```

instead of re-describing every detail that is already visible in the image.

### Text + audio models

Examples:

- `TTS Chatterbox Multilingual`
- `TTS Index TTS 2`
- `TTS Qwen3 Base (12Hz) 1.7B`
- `TTS KugelAudio 0 Open 7B`

Practical use:

- The text is the script or speech content.
- The audio input usually gives the voice identity, emotion, or speaking style.

For dialogue-capable models such as `TTS Index TTS 2`, `TTS Qwen3 Base (12Hz) 1.7B`, and `TTS KugelAudio 0 Open 7B`, the prompt can contain speaker tags such as `Speaker 1:` and `Speaker 2:`.

### Text + image + audio models

Examples:

- `Hunyuan Video Custom Audio 720p 13B`
- `Hunyuan Video Avatar 720p 13B`

Practical use:

- `Hunyuan Video Custom Audio 720p 13B` is useful when you want a person from a reference image to speak or sing using a recorded voice or song. The text prompt is still useful to describe the scene, mood, framing, or action.
- `Hunyuan Video Avatar 720p 13B` is useful when the audio drives the animation, but you still want the prompt to guide the result, for example by asking for a close-up, a realistic setting, or a particular emotional tone.

### Text + video models

Examples:

- `Hunyuan Video 1.5 Upsampler 720p 8B`
- control-video workflows in Wan and VACE models

Practical use:

- The control or source video gives the timing and structure.
- The text prompt tells the model what style, subject, or scene direction you want on top of that structure.

### Instruction-style editing models

Some models do not want a purely descriptive prompt.

They expect the prompt to be an instruction applied to an existing image or video.

Practical examples in WanGP:

- `Qwen Image Edit 20B`
- `Qwen Image Edit Plus (2509) 20B`
- `Qwen Image Edit Plus (2511) 20B`
- `Flux 1 Dev Kontext 12B`
- `Wan2.1 Chrono Edit 14B`
- `Ditto 14B`

For those models, the input image or video already gives the starting point.

So the prompt should say what to change, not just describe a final scene.

Good action verbs are:

- `add`
- `remove`
- `replace`
- `change`
- `turn`
- `rotate`
- `recolor`
- `relight`

Weak prompt for an edit model:

```text
a woman with a red hat in a rainy street
```

Better:

```text
Add a red wool hat to the woman, keep her face, hairstyle, and the rainy street unchanged.
```

More practical examples:

```text
Remove the people in the background and keep the main subject untouched.
Change the car color to matte black and keep the camera angle the same.
Replace the cloudy sky with a sunset sky, but keep the buildings unchanged.
Rotate the pose of the woman so that she is facing the right.
Render the subjects as classical sculptures carved from single blocks of pristine white marble.
```

This is especially important for `Qwen Image Edit`, `Flux Kontext`, `Chrono Edit`, and `Ditto`.


For video edit models such as `Ditto 14B`, write instructions that apply to the whole clip or the whole frames, for example:

```text
Turn the whole video into a black-and-white film look while keeping the original motion.
Replace the material of all visible statues with polished gold.
```

For `Wan2.1 Chrono Edit 14B`, the optional Prompt Enhancer can help with its stricter prompt format. Enable it when the user requests prompt enhancement; otherwise write the model's required format directly.

## Text Prompt Basics

WanGP reads the prompt line by line before generation.

- A line starting with `#` is a comment.
- A line starting with `!` is a macro line (see Macro section below).
- Other lines are prompt content.

Blank lines separate complete prompts in paragraph modes `PG` and `PW`; single newlines stay inside each paragraph. In line modes `G` and `W`, empty lines are ignored. Full-prompt mode `FG` preserves multiline content, and some speech models also preserve empty lines as speech/dialogue split markers. Full-line `#` comments are removed before prompt splitting.

This is especially practical with:

- `TTS Qwen3 Base (12Hz) 1.7B`
- `TTS KugelAudio 0 Open 7B`
- `TTS Index TTS 2`

Those models are often used for long speeches or dialogue, so keeping a multiline script together is usually more useful than treating each line as a separate generation.

## How To Process Each Line Of The Text Prompt

The dropdown `How to Process each Line of the Text Prompt` changes how WanGP interprets multiline prompts.

The UI shows these choices:

- `Each New Line Will Add a new Video/Image/Audio Request to the Generation Queue`
- `Each new Paragraph separated by an Empty Line Will Add a new Video/Image/Audio Request to the Generation Queue`
- `Each Line Will be used for a new Sliding Window of the same Video Generation`
- `Each Paragraph Separated by an Empty line will be used for a new Sliding Window of the same Video Generation`
- `All the Lines are Part of the Same Prompt`

Which wording you see depends on the current model and whether it outputs video, image, or audio.

These choices correspond to `G`, `PG`, `W`, `PW`, and `FG` in `multi_prompts_gen_type`. Use `PW` for structured multi-line window prompts: exactly one blank line between complete windows, single newlines within each window. A searchable `# WINDOW 01 — ...` title can precede its duration command without a blank line between them.

### 1. Each New Line Adds A New Queue Item

This is the most practical choice when each line is meant to be a separate idea.

Example:

```text
a fox running through snow
a whale breaching at sunset
a drone shot of a neon city in the rain
```

This creates 3 separate jobs.

Use this when:

- you want to batch several unrelated ideas
- you want quick A/B tests of prompt variants
- you want one `Start Image` reused with several motion prompts in an image-to-video model

Good examples:

- `Wan2.1 Text2video 14B`: try several scene ideas at once
- `Qwen Image 20B`: try several poster or ad concepts at once
- `Wan2.1 Image2video 480p 14B`: try several motion ideas from the same image

### 2. Each Line Is Used For A New Sliding Window Of The Same Video

This is useful for long-video workflows.

Instead of treating each line as a separate job, WanGP uses:

- line 1 for the first window
- line 2 for the second window
- line 3 for the third window
- and if the video needs more windows than lines, the last line is reused

Example:

```text
wide establishing shot of the city at dawn
the camera moves into the crowd and follows the singer
close shot of the singer as the chorus starts
```

This is practical when you want one long video with changing story beats.

It is especially useful for models meant for long or multi-window video generation, such as `Infinitetalk`, which is described as supporting very long videos with smooth transitions between shots.

Use this when:

- you want one line per shot or story beat
- you want a long talking or performance video to evolve over time
- you want a rough storyboard without writing one huge paragraph

Important practical limitation:

- only one `Start Image` is supported in this mode

#### Injecting Frames Across Windows

For models with **Inject Frames**, add your images under **Reference Images** and enter comma-separated tokens in **Positions of Injected Frames**:

- A frame number, such as `1`, injects the next image at that position (`1` is the first frame).
- `L` injects the next image at the end of the next window.
- `X` skips that window without consuming an image. Repeat it to delay injection for multiple windows.

For example, `L, L, X, L, L` uses four images at the ends of windows 1, 2, 4, and 5. Window 3 has no new frame injection. `L, X, X, L` injects two images at the ends of windows 1 and 4. Lowercase `l` and `x` also work. Enter these tokens in the positions field, not in the text prompt.

#### Optional `[/...]` Window Commands

Sliding-window prompts can include optional slash commands in brackets. WanGP removes these commands before sending the prompt text to the model. Brackets that do not start with `/` are ignored by this parser and remain available for model-specific prompt syntax such as Prompt Relay.

Generic WanGP window commands:

- `[/duration=121]`: this window should contribute 121 output frames
- `[/duration=5s]`: this window should contribute about 5 seconds of output at the generation FPS
- `[/duration=20%]`: this window should contribute 20% of the requested total frame count
- `[/overlap]`: use the model's default overlap for this window
- `[/overlap=9]`: use 9 overlap frames for this window, rounded to the model's overlap frame step
- `[/overlap=0]`: use no overlap frames, when the model supports text-to-video windows
- `[/new_shot]`: start this window without overlap frames, creating a hard transition, it is an alias for `[/overlap=0]`
- `[/no_end_image]`: skip this window's End Image without consuming it. The next window without this command uses the next unused End Image. Repeat the command on consecutive window prompts to skip several windows; it takes no value and does not affect Inject Frames or overlap.
- `[/loras_mult=1;3]`: override the active LoRA multipliers for this window only. The selected LoRAs stay the same; use the same syntax as the LoRAs Multipliers field, such as `[/loras_mult=1;3 0.5;0.5]` for two active LoRAs. Windows without `[/loras_mult=...]` use the normal LoRA multipliers from the UI/default settings.

Use `[/new_shot]` when a window should behave like a hard cut: a new scene, a new character introduction, or the first generated window after Continue Video when the source video should remain in the final output but should not visually condition the new generated window.

For example, with four End Images and five window prompts, add `[/no_end_image]` to the third prompt to use the images in windows 1, 2, 4, and 5. This command applies only to the marked window; automatically added windows use End Images normally.

Multiple commands can be combined in one bracket, for example `[/duration=5s,/overlap=9]`, `[/duration=4s,/new_shot]`, or `[/duration=5s,/loras_mult=1;3]`.

Overlap frames are generated in addition to the requested window duration. For example, a `[/duration=5s,/overlap=9]` window at 25 fps aims to contribute about 5 seconds to the final video, while the 9 overlap frames are only used to condition the transition and are not counted as newly committed output frames.

If a window has no `[/duration=...]`, WanGP uses the remaining requested frame count, capped by the Sliding Window Size. If explicit durations produce more output frames than the UI frame count, validation does not block the prompt; WanGP reports the predicted total frame count. Validation only rejects a prompt when a later window would receive 0 frames because previous windows already consumed the requested frame count.

Example:

```text
[/duration=25%] A wide dawn shot of a mountain train station. Steam rolls across the platform, red lanterns sway, and the camera slowly pushes toward a violinist waiting beside a blue suitcase.

[/duration=5s,/overlap=17] The violinist steps onto the train as the platform slides backward. Keep the blue suitcase visible, the lantern reflections in the glass, and a gentle handheld camera rhythm.

[/new_shot,/duration=4s] A sharp cut to inside the dining car at night. The violinist now sits across from a silent chess player in a silver coat while rain lashes the window.

[/duration=30%,/overlap=9] The chess player moves a knight, the train lights flicker, and the blue suitcase opens by itself, revealing a tiny glowing city.
```

Models may declare their own additional slash commands. Unknown slash commands are rejected during validation. JoyAI-Echo adds named-memory commands such as `[/store_mem=man1,man2]`, `[/load_mem=man1,woman1]`, `[/load_mem]`, and `[/drop_mem=old_scene]`. If a Joy window has no `[/load_mem...]`, it keeps the active memory set from the previous window plus any memory stored by that previous window. `[/no_mem]` is deprecated and ignored because Joy memories are no longer saved automatically.

### 3. All The Lines Are Part Of The Same Prompt

This is the right choice when line breaks are part of the prompt format itself.

Use this when the prompt is one structured block, such as:

- a second-by-second timeline
- a dialogue script
- song lyrics with sections like `[Verse]` and `[Chorus]`

Example: timeline prompt

```text
(at 0 seconds: the man stands in front of the fridge, camera static)
(at 1 second: he opens the fridge and reaches for a can)
(at 2 seconds: the camera shifts to a side angle as he drinks)
```

This is especially useful for `Wan2.2 Image2video Enhanced Lightning v2 14B`, because its own default prompt uses that kind of timeline format.

Example: dialogue prompt

```text
Speaker 1: We should leave before the rain gets heavier.
Speaker 2: Give me one minute, I still need my jacket.
Speaker 1: One minute, then we run for the bus.
```

This is the practical choice for:

- `TTS Index TTS 2`
- `TTS Qwen3 Base (12Hz) 1.7B`
- `TTS KugelAudio 0 Open 7B`

Those models are used for speech and dialogue, so splitting each line into a separate job would usually destroy the conversation structure.

Example: lyrics prompt

```text
[Verse]
Morning light through the window pane
I hum a tune to chase the rain
[Chorus]
Stay with me through every mile
Hold this fire, hold this smile
```

This is the practical choice for:

- `TTS HeartMuLa OSS 3B`
- `TTS ACE-Step v1.5 Turbo 2B`

Those music models work best when the lyrics stay together as one structured prompt.

## Multiple Images As Text Prompts

The dropdown `Multiple Images as Texts Prompts` matters when you use several images together with several prompt lines.

This is mainly useful for image-to-video models such as:

- `Wan2.1 Image2video 480p 14B`
- `Wan2.2 Image2video Enhanced Lightning v2 14B`

The two choices are:

- `Generate every combination of images and texts`
- `Match images and text prompts`

In both modes, WanGP uses the image prompt field currently driving the job:

- if you provided `Start Image`, those images are used
- otherwise WanGP uses `End Image`
- if both are present, `Start Image` is the one paired with the text prompts

So in normal image-to-video practice, this usually means the `Start Image` field. It is not about `Reference Images`.

### Generate Every Combination Of Images And Texts

This is the exploration mode.

WanGP takes every prompt line and combines it with every image from the active image prompt field in use, usually `Start Image`.

If you have:

- 3 images
- 2 prompt lines

WanGP will create 6 jobs.

Use this when:

- you want to see which motion prompt works best on which image
- you are testing several portraits against several actions
- you are exploring, not pairing

Practical example:

- 3 portraits
- prompt 1: `the subject smiles and leans toward the camera`
- prompt 2: `the subject turns away and walks toward the window`

This is useful if you want to compare all results.

### Match Images And Text Prompts

This is the pairing mode.

Use it when image 1 should go with prompt 1, image 2 with prompt 2, and so on.

This is usually the better choice when:

- each image already has its own intended motion
- you are preparing a batch of separate image-to-video jobs
- you do not want the combinatorial explosion of testing everything against everything

Practical example:

- portrait 1 + `she smiles and waves`
- portrait 2 + `he turns left and starts walking`
- portrait 3 + `the child looks up and starts laughing`

If you used `Generate every combination of images and texts`, you would get many unwanted cross-combinations. `Match images and text prompts` keeps the batch clean.

## Prompt Enhancer

Prompt enhancement is easiest to think of as a writing assistant built into WanGP.

For **AuK**, **Refine** offers **AuK instruction from text** and **AuK instruction from text + source transcription**. The first option rewrites your instruction and refines voice descriptions. The transcription option first transcribes the source recording locally, then gives that text to the enhancer as context. It covers the same opening section that AuK will edit: the requested target duration, capped at the clip's length. It is most useful for replacing, inserting or deleting words. Select the source-audio mode, upload the recording and set the target duration before using it.

Transcription adds processing time and may mishear words. If no source audio is selected, no speech is detected or transcription fails, a notice appears and refinement continues using text only. Cancelling still stops the operation. Quote original and replacement words explicitly when you know them; those words take priority over the transcript. Check the enhanced instruction before generating. Neither option adjusts AuK's target duration. Enhancement is disabled by default. If you use an external enhancer, it receives the transcript as well as your request; choose a local enhancer to keep both on your machine.

WanGP has two levels for it:

- a global Prompt Enhancer setup in `Configuration`
- a per-generation Prompt Enhancer control next to the main prompt box

The Prompt Enhancer can use a local model on the WanGP machine or the same external Codex, Claude Code, or OpenCode engine configured for Deepy Prime. The Prompt Enhancer row appears for models that provide an enhancement workflow; its exact choices depend on the selected model.

### Where It Is Enabled

Open **Configuration > Prompt Enhancer / Deepy**, then select **Prompt Enhancer / Deepy LLM Engine**.

That global configuration decides:

- which local or external engine writes enhanced prompts
- whether it works automatically during generation or on demand
- for local Qwen engines, which quantization and acceleration options are used
- for local engines, sampling behavior such as temperature, top-p, and seed randomization

If you select an external engine, configure its executable, model, and reasoning options in the panel that appears. External engines require **Deepy Prime**, even when you mainly intend to use prompt enhancement, because Prime provides their WanGP integration. Save the configuration when finished.

For a model that supports prompt enhancement, the generation screen shows the Prompt Enhancer row near the main prompt field.

Depending on your configuration, it either:

- runs automatically during generation
- or appears as a button you click manually before generation

### Choose a local or external engine

| Engine | Functional impact | Recommendation |
|---|---|---|
| **Florence 2 + Llama 3.2 3B** | A lightweight local combination for image understanding and text rewriting. | Use on limited hardware when a simple local enhancer is sufficient. |
| **Florence 2 + Llama Joy 8B** | Produces richer local rewriting than the 3B option but uses more memory. | Use when you prefer this writing style and do not need Deepy Prime. |
| **Qwen3.5 VL Abliterated 4B** | Handles text and images in one local model with relatively low memory use. | Recommended local starting point for most users. |
| **Qwen3.5 VL Abliterated 9B** | Better instruction following and richer prompts than 4B, with higher VRAM/RAM use. | Use when quality matters and it fits comfortably beside your generation model. |
| **Qwen3.8 VL Uncensored 27B** | Strongest local understanding and rewriting, but much heavier to load and run. It supports Deepy Prime. | Use for quality-first work and complex visual instructions on high-memory systems; on GPUs with less than 16 GB of VRAM, choose its Bonsai 2 PTQ1_0 quantization. |
| **Qwen3.8 VL Flash-Next 125B** | The largest local model, kept mostly in RAM and on the drive with what the text needs in VRAM. It supports Deepy Prime. | Use on a PC with about 64 GB of RAM and a fast SSD (an 80 GB download). |
| **Codex** | Uses the selected external Codex model and does not occupy WanGP's local GPU memory. | Use when you already have Codex configured and want strong remote instruction following. |
| **Claude Code** | Uses the selected external Claude model and does not occupy WanGP's local GPU memory. | Use when Claude is your preferred writing/reasoning provider. |
| **OpenCode** | Connects through OpenCode to its configured provider, including supported cloud or local OpenAI-compatible services. | Use when you need provider flexibility or already manage models through OpenCode. |

An external engine is useful when you want to preserve local VRAM for image or video generation, avoid downloading a local enhancer model, or use a stronger provider model. Network and provider latency can make each enhancement slower to start.

The selected engine is shared by **Deepy, Prompt Enhancer, and visual inspection**; there is no separate Prompt Enhancer engine selector. With an external engine, prompts, attached images, sampled video frames, and relevant instructions may be sent outside the WanGP machine. Provider privacy and retention rules apply. WanGP does not store provider passwords or access tokens; authentication stays with the external tool. A local OpenCode server using a local provider can keep data on the machine.

For installation, authentication, model selection, and provider-specific behavior, see [Remote LLMs](REMOTE_LLMS.md).


### Automatic Versus On-Demand

**Automatic mode:**

The enhanced prompt will appear in the generation preview on the right. 

- Pros: fastest workflow, no extra click, good when you always want prompt rewriting
- Pros: useful for models where the enhancer is almost part of the workflow, such as `Wan2.1 Chrono Edit 14B`
- Cons: you do not review the rewritten prompt before generation
- Cons: if the enhancer over-interprets your idea, you only notice it after the run starts or in the saved metadata

**On-demand mode:**

The original prompt will be commented and an enhanced prompt will appear below. If you are not happy with the enhanced prompt you can request for another and it will overwrite automatically the one that was generated previously.
Alternatively you can modify the original prompt which is in the comment and the new instructions will be taken into account (no need to remove the # comments prefix).

For instance if you got after the first prompt enhancement request:
```text
#!PROMPT!: Jensen Huang is talking on stage and says "Welcome guys, we are going to have a lot of fun today."
Jensen Huang steps into the spotlight, black leather jacket creaking slightly as he shifts his weight. His glasses reflect the stage lights ...
```

Modify the prompt in the comment:
```text
#!PROMPT!: A woman is talking on stage and says "Welcome guys, we are going to have a lot of fun today."
She steps onto the dimly lit stage, spotlight cutting through haze. "Welcome guys. We're going to have a lot of fun"...
```

- Pros: you can inspect and edit the rewritten prompt before generating
- Pros: best for strict prompt formats, edits, lyrics, or dialogue where you want manual control
- Cons: one more step in the workflow
- Cons: with multiple `Start Image`s, it only works cleanly when `Multiple Images as Texts Prompts` is set to `Match images and text prompts`, and the number of images matches the number of prompt lines

### Images And Window Commands During Enhancement

Local Qwen enhancement and remote LLMs can use all Reference Images together with the current Start Image, End Image, and Control Image where supported; a remote LLM receives the images of each prompt with their names. You can refer to them in your prompt as `Image reference no 1`, `start image`, `end image`, and `Control Image`. When the first reference is selected as the main image, it is named `Main Ref. Image`; subsequent references keep their numbers. Injected references also have a time label, such as `Image reference no 2 (Frame 2s/15s)`; a reference at the window's endpoint is identified as `end image`. Florence-based enhancers use one image caption per prompt.

When continuing a video, its last retained frame (after source trimming) is available to the enhancer as `start image`. `[/new_shot]` omits this visual anchor.

In Sliding Window prompt modes, write each prompt for that window's references and End Image. The previous window's end image supplies the next start image, unless a new shot is requested. `[/no_end_image]` omits the gallery End Image for that window, leaving it available for the next window. For injected references, `X` skips a window's end-frame slot without consuming a reference, and `L` places a reference at the window's last kept frame, respecting duration changes and tail trimming.

Window commands such as `[/duration=10s]` survive enhancement, and the enhanced action takes the window's duration into account. Commands appear before the enhanced text: on the same line in one-prompt-per-line modes, or on their own line in paragraph and whole-text modes. Edit commands in the visible prompt, outside the `#!PROMPT!:` history line; re-enhancement uses those current commands, including changes or removals.

### Qwen Backend Choices

The **Qwen LLM Quantization** choice is available only for local Qwen engines. Lower quantization uses less VRAM and RAM but can reduce prompt quality.

For Qwen3.5:

- **Quanto Int8** generally gives better quality and uses more memory. This is the recommended choice when it fits comfortably.
- **GGUF Q4** uses less memory and can be faster with compatible GGUF CUDA kernels, at some cost to prompt compliance. Use it when Int8 leaves too little memory for generation.

For Qwen3.8:

- **GGUF Q4** gives the highest quality and uses the most memory.
- **GGUF IQ3_S** provides the recommended balance of quality and memory for most local Qwen3.8 users.
- **GGUF Q2** has the lowest memory use and the greatest quality loss. Use it only when Q3 or Q4 cannot fit.

The local Qwen engines always run at the fastest speed your installation allows: WanGP uses CUDA graphs and every accelerated kernel that is installed (Triton, FlashAttention 2, the GGUF kernels), with no setting to choose. Installing Triton and FlashAttention 2 makes them faster. When the language model does not fit in VRAM beside your other needs, WanGP keeps as many of its layers in VRAM as fit and streams the others from RAM: responses are slower, but they still work. **Configuration > Performance > Language Models CPU Compute** makes this case faster: the CPU processes the layers left in RAM while the model writes. **Auto** (default) lets the CPU help when it is faster on your PC, which with the Reserved RAM that these models borrow while they write (Memory Profile 0, which they always use) answers about twice as fast. See the Deepy guide for details. The audio and music models that use a language model work the same way, each optimization used wherever it is supported.

### Speculative Decoding (MTP)

Speculative decoding can make supported local Qwen text generation faster by predicting multiple tokens ahead. It affects Prompt Enhancer and Deepy response speed, not media-generation speed.

- **Auto** is recommended. It normally enables acceleration with at least 12 GB VRAM for Qwen3.5 9B or 24 GB for Qwen3.8 27B.
- **Disabled** saves the additional VRAM and is appropriate when the generation model needs that memory.
- **A method with Number of Tokens on Auto** keeps that method on and lets WanGP choose the draft count for your model and GPU.
- **A method with a fixed Number of Tokens** allows manual tuning. More draft tokens are not always faster; compare repeated prompts if you want to tune it.

Qwen3.5 4B and the Florence/Llama engines do not support this option. Quantization, speculative decoding, local sampling controls, and local VRAM-loading controls are hidden when an external engine is selected because the external provider owns its runtime behavior.

### The Main Prompt Enhancer Choices

The user-facing dropdown usually offers:

- `Disabled`
- `Based on Text Prompt Content`
- `Based on Images Prompts Content (such as Start Image and Reference Images)`
- `Based on both Text Prompt and Images Prompts Content`

But not every model shows exactly that set.

Some models expose model-specific options instead.

Practical examples:

- `TTS Index TTS 2`, `TTS Qwen3 Base (12Hz) 1.7B`, and `TTS KugelAudio 0 Open 7B` can show:
  - `A Speech based on current Prompt`
  - `A Dialogue between two People based on current Prompt`
- `Wan2.1 Chrono Edit 14B` restricts the Prompt Enhancer to the combined text+image mode, because that model expects both the edit instruction and the source image

So the safest rule is:

- use the supported `prompt_enhancer` choices: `T` enables enhancement, and `I` adds image context to form `TI`
- prefer the model-specific labels when WanGP exposes them, because they were defined for that model on purpose

### Based On Text Prompt Content

This is the best choice when your text idea is short, rough, or under-detailed.

Practical use:

- Turn a simple video idea into a more cinematic paragraph
- Turn a rough speech topic into a proper spoken script
- Turn a basic song idea into structured lyrics

Good examples:

- `Wan2.1 Text2video 14B`: useful if you type a short scene idea and want richer cinematic detail
- `Qwen Image 20B`: useful if you want a more polished image prompt
- `TTS Chatterbox Multilingual`: useful if you want the app to draft the speech before generating the voice

### Based On Images Prompts Content

This is most useful when the image matters more than the text.

Practical use:

- Start from a portrait or reference image
- let the enhancer read the image
- then have it write a motion prompt that fits what is actually visible

This is often useful for image-to-video workflows where the face, clothing, pose, or framing should stay faithful to the source image.


### Based On Both Text Prompt And Images Prompts Content

This is the most useful choice when both the instruction and the image are important.

The clearest case is `Qwen Edit 14B`.

- the image tells the model what exists now
- the text tells it what you want changed
- the prompt enhancer rewrites that into the format the model expects

Pros:

- the best choice for edit models and instruction-following workflows
- balances image faithfulness with explicit user intent

### Model-Specific Writing Buttons

Common button labels are `Enhance Prompt`, `Write`, `Write Speech`, and `Compose Lyrics`.

Some models expose more specific prompt-writing workflows.

#### `Write`

Used on speech models such as:

- `TTS Index TTS 2`
- `TTS Qwen3 Base (12Hz) 1.7B`
- `TTS KugelAudio 0 Open 7B`

Practical use:

- write a monologue from a short topic
- write a two-person dialogue from a rough idea

This is especially useful because those models are often used for spoken content, and two of them explicitly support `Speaker 1:` / `Speaker 2:` dialogue workflows.

#### `Write Speech`

Used on `TTS Chatterbox Multilingual`.

Practical use:

- quickly turn a topic into a spoken script before generating voice audio

This is helpful because Chatterbox is meant for multilingual speech generation, so many users start with an idea, not a finished script.

#### `Compose Lyrics`

Used on music models such as:

- `TTS HeartMuLa OSS 3B`
- `TTS ACE-Step`

Practical use:

- turn a short idea such as `a dreamy synth-pop song about missing someone`
- into actual lyrics with sections like `[Verse]` and `[Chorus]`

This is especially useful because those models are conditioned by lyrics, and in HeartMuLa's case also by style tags.

## `@` And `@@` Syntax

These two syntaxes only matter when the Prompt Enhancer is being used.

If the Prompt Enhancer is disabled, WanGP treats them as normal characters.

### `@` adds extra instructions to the enhancer

Format:

```text
your prompt @ extra instructions
```

Example:

```text
a serious speech about AI safety @ keep it under 6 sentences, natural spoken English
```

Practical use:

- keep a script short
- ask for a certain tone
- ask for a certain output shape without replacing the model's tuned enhancer instructions

This is the safer and more practical everyday option.

### `@@` replaces the enhancer instructions completely

Format:

```text
your prompt @@ full replacement instructions
```

Example:

```text
a woman opens a door @@ Output exactly 6 lines. Each line must start with "(at X seconds:" and describe only visible motion.
```

Practical use:

- force a very specific output format
- completely override the built-in enhancer behavior

Use this only when you know exactly what format you want. It is powerful, but easier to misuse than `@`.

## Think Mode

With local Qwen3.5- and Qwen3.8-based prompt enhancers, WanGP can show a `Think` checkbox. External engines use their configured model and reasoning-effort controls instead.

Practical use:

- enable it when the prompt is vague, messy, or structurally difficult
- disable it when you want the fastest rewrite

It is most useful for tasks like:

- turning a rough story idea into a clean second-by-second prompt
- turning a topic into a believable two-person dialogue
- rewriting a difficult edit instruction so it stays aligned with the input image

The thinking text itself is not meant to become part of the final enhanced prompt.

## Macro System

Create multiple prompts from templates using macros. This allows you to generate variations of a sentence by defining lists of values for different variables.

Macros are expanded before WanGP applies the line-processing option from `How to Process each Line of the Text Prompt`. In practice, this means:

- macros first generate the prompt lines
- then WanGP decides whether those lines become separate jobs, sliding-window steps, or one single multiline prompt

### Syntax Rule

Define your variables on a single line starting with `!`. Each complete variable definition, including its name and values, must be separated by a colon (`:`).

### Format

```text
! {Variable1}="valueA","valueB" : {Variable2}="valueC","valueD"
This is a template using {Variable1} and {Variable2}.
```

### How It Works

- Every `{Variable}` in the following prompt lines is replaced with one value from its list.
- WanGP cycles through the values line by line.
- If two variables do not have the same number of values, the shorter one is reused cyclically.

This makes macros practical for:

- batch-testing several scene ideas
- generating multiple poster captions for image models
- creating several prompt variations from one reusable template

### Example

The following macro will generate three distinct prompts by cycling through the values for each variable.

**Macro Definition:**

```text
! {Subject}="cat","woman","man" : {Location}="forest","lake","city" : {Possessive}="its","her","his"
In the video, a {Subject} is presented. The {Subject} is in a {Location} and looks at {Possessive} watch.
```

**Generated Output:**

```text
In the video, a cat is presented. The cat is in a forest and looks at its watch.
In the video, a woman is presented. The woman is in a lake and looks at her watch.
In the video, a man is presented. The man is in a city and looks at his watch.
```

### Practical Macro Example

If you want to test several ad concepts in an image model, you can do this:

```text
! {Product}="watch","perfume","sneakers" : {Mood}="luxury","minimal","energetic"
Studio advertising photo of a {Product}, {Mood} style, clean background, premium lighting, centered composition
```

If your line-processing mode is set to add each new line as a new request, WanGP will queue one image job per generated prompt.

## Troubleshooting

- If dialogue lines are turning into separate jobs or a timeline prompt is split across several jobs, choose `All the Lines are Part of the Same Prompt`.
- If you want a long video with changing story beats, choose `Each Line Will be used for a new Sliding Window of the same Video Generation`.
- If you want several separate prompt ideas queued at once, choose `Each New Line Will Add a new ... Request to the Generation Queue`.
- If `Match images and text prompts` fails, the number of images and prompts must match cleanly.
- If `@` or `@@` seems ignored, the Prompt Enhancer is probably disabled or not used for that run.
- If an external engine cannot be saved, enable **Deepy Prime** and complete that engine's setup in **Configuration > Prompt Enhancer / Deepy**.
- If remote enhancement cannot inspect an image, confirm that the selected provider/model accepts images and that the external tool is authenticated.
- If you do not want prompts or visual inputs to leave the machine, select a local engine instead of Codex, Claude Code, or a cloud-backed OpenCode provider.

---

> Applies to: Prompt writing, comments, line and paragraph splitting, window commands, image pairing, optional enhancement and macros. Prompt style and supported syntax vary by model and are described in its prompt help.
