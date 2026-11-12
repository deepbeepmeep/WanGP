# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Generative Models Accessible to the GPU Poor</b>
</p>

WanGP is a one-stop super app for open source generative models: create videos, images, music, speech and 3D assets, then step inside interactive worlds. Run locally on your own GPU, use the browser interface, or let **Deepy Prime** turn your ideas into finished creations.

## Highlights

| Modality | Supported models |
| --- | --- |
| **Video** | **Wan 2.1/2.2** and derived models, **MiniMax H3**, **LTX-2/2.3/2.5**, **Hunyuan Video 1/1.5**, **LongCat**, **Kandinsky**, **LTXV**, **MagiHuman** |
| **Image** | **Fibo 2.0**, **Flux 3 Klein** and **Flux 1/2**, **MOSS Image / Edit**, **Krea 2**, **Qwen Image**, **Z-Image**, **SenseNova**, **Ideogram 4**, **HiDream** |
| **Audio / TTS** | **Fish Audio Drama 3**, **MOSS-TTS v1.5**, **Qwen3 TTS**, **MiniMax H3 Voice Clone**, **Ace Step 1/2/XL**, **Omnivoice**, **Index TTS2/2.5**, **KugelAudio**, **HeartMula**, **Chatterbox**, **Minimax Music**, **Stable Audio 3** |
| **3D** | **Hunyuan3D 2.1**, **TRELLIS.2**, **TripoSG**, **TripoSplat** (including **Idol**), **PartCrafter**, **WorldMirror 2.0**, **MapAnything**, **Depth Anything 3 Points**, **MoGe-2** |
| **Interactive worlds** | **Hy-Nomas 8B** from text; **EVOKE Turbo 14B** from an image. Walk, look around, introduce events and save your journey. |

### Create a World, Then Explore It

- **Keyboard exploration**: open a generated world in the **Worlds gallery**, press **Explore**, and walk with WASD while the arrow keys turn the camera. Each move generates the next part of the scene.
- **Keep what you discover**: capture a view, record the journey as a video, or **Save World** to resume later. Continue a saved `.wgpworld` with movement instructions or ask Deepy to extend it.
- **Choose your tradeoffs**: adjust memory and, for Hy-Nomas, quality and move length. **8 GB of VRAM is enough to get started**; exploration speed depends on the model, settings and GPU.

### Deepy Prime — From an Idea to a Finished Creation

- **A local creative assistant**: plan and run image, video, speech, music, 3D and world generation, inspect the results, and refine them in chat.
- **More context in the same VRAM**: the **Q4 KV cache** fits roughly x4 the conversation context within the same memory budget.
- **Research and find resources**: optional web search finds facts and visual references; Deepy can also find compatible LoRAs on Hugging Face and Civitai. Web access is off by default; Civitai downloads need your API key.
- **Create with Python and Media Flow**: make exact lettering, charts, subtitles and procedural 3D props; automate batches and restore, upscale or smooth long videos while preserving their audio.
- **Deepy Apps — development preview**: ask Deepy to build games, interactive stories, music toys and explorable 3D scenes using generated assets. Play them in WanGP, keep progress, share `.wgpapp` files and continue improving them in later chats.  

[Meet Deepy](https://wangp.ai/deepy/) · [Try 10 generated apps](https://wangp.ai/deepy/apps/) · [World Models demos and exploration guide](https://wangp.ai/worlds/)

### Run More Models on More Hardware

- **Low VRAM requirements**: run select models with as little as **4 GB of VRAM**.
- **Older Nvidia GPU support**: use GTX 10XX, RTX 20XX, and newer cards.
- **AMD GPU support**: run on RDNA 4, 3, 3.5, and 2 hardware; see the Installation section below.
- **Fast latest-GPU performance**: take advantage of modern GPU acceleration.
- **Full web interface**: generate, manage, and reuse outputs from an easy browser UI.
- **LoRA customization**: adapt each model with LoRAs, reuse LoRAs stored in another App.
- **Many quantized checkpoint formats**: use int8, fp8, gguf, NV FP4, and Nunchaku.
- **Architecture-aware downloads**: automatically fetch the model files suited to your hardware.
- **Finetunes**: add your own finetunes / checkpoints or the ones you found on Hugging Face or CivitAI
- **Generation queue**: line up videos, images, and audio jobs, then come back later.
- **Headless mode**: launch batches from the command line for images, videos, and audio.
- **WanGP API**: add generative capabilities to your own apps.
- **MCP Interface**: external agents can access to all WanGP features and their activity is  visible in the Web interface.

### Built-In Creation Tools

- **Video, image, audio, 3D and Worlds galleries**: browse generations, inspect 3D assets, explore worlds and reuse outputs as new inputs.
- **Reusable settings**: extract settings from any generation, create templates, and share them.
- **Per-model prompt enhancer**: improve prompts with model-specific syntax and expectations.
- **Input preparation tools**: use the mask editor, background remover, pose/depth/flow extractors, speaker diarization, and background noise/song remover.
- **Offline workflows**: keep generation and Deepy local; use `--offline` to require already-downloaded files and prevent automatic downloads.
- **Temporal and spatial upsampling**: improve outputs with DLLS5, RIFE, FlashVSR, RelayVSR, SeedVR and Lanczos.
- **Audio postprocessing**: generate soundtracks with MMAudio, replace voices with SeedVC, or remux a video with any soundtrack.
- **Ready-to-use Plugins**: Gallery Browser, Motion Designer, Models/Checkpoints Manager, CivitAI browser and downloader, and more.

**Discord Server to get Help from the WanGP Community and show your Best Gens:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

**Official WanGP Web Site**: https://wangp.ai/

> [!IMPORTANT]
> **WanGP is free to use locally, at home or at work.** The official project will never ask you to pay a license fee, subscription, or donation to run WanGP on your own computer (see the license for terms). If you like WanGP, a ⭐ [star on GitHub](https://github.com/deepbeepmeep/WanGP) is always appreciated: it helps others find the project.
>
> **Use only the official GitHub repository or wangp.ai / wan2gp.ai websites. WanGP is not affiliated to any other third-party service using the WanGP/Wan2GP names**, unless explicitly stated here.


## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)


## 🔥 Latest Updates : 

## 12th of November 2026: WanGP v19.292 - Bonus Track
- **Qwen Image 2.1 Viggle Image Swap**: a Qwen Image 2.1 variant and **Showrunner's default frame painter for Viggle Remake character swaps**. Replace one or several people in **6 steps**, keeping their poses, expressions and the scene's framing. Add the **source picture first**, then one **character sheet** per person; the picture's aspect ratio is preserved at about **1 megapixel**. Results can be slightly darker.

- **H3 HyperFlow LoRA**: bring **8-step** generation to the standard and Pruned **Ref2VA** models with the new **HyperFlow Ref2VA 8 Steps** accelerator profile.

- **H3 Regenerate 2K IC**: **better identity preservation and fewer hallucinated details** than the previous Regenerate release, which could invent features and change faces. IC uses the draft itself as a reference to add detail while keeping its identity, lip sync and composition, with the original soundtrack. Finish at up to **2560x1440** in **3 extra steps**: **6 + 3** on **HyperFlow Flash Ref2VA**, or **8 + 3** through the **Ref2VA preset**; tiling reduces memory use and works much better here than before.

- **H3 LoRA compatibility**: improved LoRA loading on supported merges of Pruned models, fixing the rejection of adapters such as **Neat Prime** on **DaSiWa GXS v5**.

- **H3 DMAD accelerators**: generate video with sound in **4 or 8 steps** on FL2VA and Ref2VA: choose 4 for speed or 8 for better quality. The profiles apply the matching LoRA and settings automatically. The new **DMAD Re-noise** sampler also supports **H3 Turbo Neat Prime**.

- **Official H3-Image**: MiniMax's own image model generates from text, or edits and combines up to **16 reference images**, in **6 steps**. It works best around **16 megapixels** with detailed descriptions; dedicated prompt enhancers help expand short requests. Choose the full **33B** model or **Pruned 20B**, which produces nearly identical images with a smaller download and faster loading. H3 Image seems to be the **best image swapper** for Viggle.

- **H3-Image: pDMD + LoRAs**: generate in **2 steps** for speed or **4 steps** for crisper detail. Two-step denoising takes less than half the time of the usual six steps; tested 4 MP images finish about **25% sooner** overall. H3-Image now shares the H3 LoRA list, so you can also try video style, realism and character LoRAs on stills, though their effect may be weaker or different.

- **Faster Deepy and Prompt Enhancer on Qwen3.8 27B**: once **GGUF kernels 1.0.30** are installed, Deepy writes its answers, tool calls and image or video inspections faster with the same quality: **up to 19% faster** with the **Q3** and **Q2** checkpoints and about **4%** with **Q4**. The new optional **Faster Context Reading** setting lets Deepy read tool results, documents and restored conversations **up to twice as fast**, for about 140 MB of VRAM while reading.

- **Showrunner: choose who sings**: have an uploaded or generated song sung by a cast member or a supplied voice sample, including **two voices for a duet**. Use **Change the voice** on an existing song or **Sung by** before composing. The music, lyrics and timing stay, so existing shots keep their lip sync. Requires **SeedVC v1.0 Singing** in **Configuration > Extensions**.

- **Low RAM Continue Video & Late Post Processing**: extend or post-process long videos without loading them whole into RAM. Continue Video reads only the ending frames for generation and adds each new window without re-encoding earlier ones; a source needing conversion is processed once, in chunks. Now supports **Lanczos, FlashVSR, SeedVR2, RelayVSR and DLSS 5**, plus **RIFE / DLSS frame interpolation and Film Grain**. HDR continuation, LTX-2 upscalers / Detail Refiner, face refiners, and interpolation combined with an AI upscaler still need the whole video in memory. Use **Media Flow** which it still best suited for very long video as it supports pause/resume, manual chunk sizing and keep original audio not reencoded.


## 8th of November 2026: WanGP v19.25 - Showrunner Makes and Remakes

- **Viggle Remake**: recast one or several people with **Swap**, or change objects, backgrounds, weather and lighting with **Edit**, while keeping the original motion, camera, timing and soundtrack. Supply your video, one to four frames from it edited to show the new look, and a prompt describing the result. An optional mask keeps everything outside the changed area exactly as it was. Works best at **720p**, with faces clearly visible and characters facing the camera; small faces and rapid cuts remain challenging.

- **Showrunner Viggle Remake**: too lazy to do all the preparation work required by Viggle Remake? Ask Showrunner: it splits long videos into shots, extracts and edits the reference frames one by one, checks the results and writes the prompts for you. Review the prepared frames, then let it render each shot and put everything back together with the original timing and soundtrack.

- **Showrunner Draft / Final**: try shots at **480p** to check the look, framing and timing before spending time on full-resolution renders. Switch to **Final** to reuse the chosen take's prompt, seed and reference pictures at the target size, without another planning pass. Both takes stay available to compare; judge lip sync at the final size.

- **Showrunner: more control**: drop your own videos or pictures onto **Add a shot**, and drop sound onto a shot to add a soundtrack. Save prompt rules per model template, including style instructions and LoRA trigger words, for reuse across productions. Export a production as a **ZIP** to back it up or open it in another WanGP, with its shots, media, versions and exports.

- **LoRA speedups**: LoRAs add work at every generation step, so using them comes with a speed penalty. New optimisations reduce that extra denoising time to around 10%. 

- **RelayVSR**: a fast new upscaler for real-world and generated images and videos, with the original **x4 model** and **LightX2V's native x2 version**. X2 is used by default for scales up to x2, taking less memory and running more than twice as fast as using the x4 model for the same scale. Long videos need no more working memory than short ones.

- **FlashVSR**: spend less time upscaling, with around **30% faster denoising** in tested FlashVSR+ Turbo

- **TaoMate-FVSR Face Refiner**: sharpen lips, teeth, skin and eyes on up to five faces, with optional soundtrack guidance for mouths. Best on close-ups; mouth movements can change. Use it after generation or through **Late Post Processing**, without a prompt, while keeping the surrounding video unchanged. See the [refiner comparison](docs/PROCESSING.md#face-refinement).

- **NitroFusion2 Z-Image Turbo**: generate images in **1 to 4 steps**, with **2 steps by default**, for faster prompt and composition experiments. Pick an **Accelerator** profile to apply the matching settings automatically. It uses the same memory as Z-Image Turbo and shares its LoRA folder.

- **H3 HyperFlow Flash**: generate video with sound in **4, 6 or 8 steps**, from text, start/end frames or references. The smaller **14B** models need less RAM; at the default 6 steps, denoising takes about a third less time than 8-step PDD. Matching H3 LoRAs are supported, though their effect can be weaker.

- **H3 Image: pDMD + Regenerate**: generate and edit images in **2 or 4 steps** with pDMD: 2 steps is about **2.5x faster** than the usual 8, while 4 gives more detail and better text. Add **Regenerate 2K Detail** after either a 2-step or 8-step draft for a three-step detail pass at full resolution; faces can change.

- **Singularity V1.3 + LoRAs**: the **Pruned 20B** release joins H3 for those who prefer its look. Matching Ref2VA LoRAs, including Turbo accelerators, now load on V1.3 and the WAS V2 r2 repack that previously rejected them.

- **WanGP Manager 0.1.15**: a new **Update WanGP** shortcut on the Overview takes you straight to updating to the latest release, with a summary of what changed since your version. Manager also notifies you of its own updates, and **Open in App** now supports dragging files into WanGP and moving items within its pages, including Showrunner.

## 4th of November 2026: WanGP v19.14 - The Show must Go On

### Showrunner 0.2.1
  - **Places that stay the same**: a kitchen or a cockpit no longer changes from shot to shot. The showrunner makes a picture of each recurring place and shoots in it, or uses your own photo.
  - **Fix what is wrong, keep the rest**: pause on a problem or pick shots, say what should change or press **+** to insert a shot. Only that changes; compare with the previous version and go back in one click. Bigger changes (a new actor, a new song) stay one sentence away in the chat.
  - **Voices and songs that hold together**: each voice is checked against its cast card and each line against its voice, so a man does not sound like a woman and a voice does not change between lines. Lines that are not really sung are found before any shot depends on them, subtitles follow the singing, and a shot inserted in a music video keeps every later shot in sync with the song.
  - **Deep Review**: one click measures a finished version against its script (words, subtitles, voices, lips, sound and picture), then the showrunner looks at every shot. Each finding says where and what, and goes to the fix window in one click.
  - **Faster and clearer at every step**: a new request on the production you just worked on starts in seconds, the panel says what the showrunner is doing right now, silent shots play their whole video, and comparing versions has a bar to jump to any moment. Turn on **Show the Deepy tab** in **Settings** to follow Deepy's thoughts and every step.
  - **Let the video model speak**: set **Voices** to **From the video model** in the Brief, or just ask the showrunner, and MiniMax H3 says the lines itself with its own sound, with nothing recorded first and no speech model needed. **Even voices** on the Timeline brings every shot to one level in the export. A voice can change from shot to shot and a take can change or add words, so listen to each one: this suits short pieces best.
  - **Long takes and shots by hand**: give the Brief a **Shot length** range, such as 8 to 15 seconds, to leave the usual 5-second shots behind. **Render now** makes a shot from what you wrote, with no language model and no showrunner, so you can build a whole production by hand, even without Deepy Prime.

### 3D

- **Qwen3.5 GLB 3D Coder 4B** turns a description and an optional reference image into a low-poly game asset. Ask for movement to export an animated GLB, or enable **Save Three.js Source** to keep the editable scene.
- **Pixal3D Multi-View Shape** builds a mesh from front, right, back and left views. Upload four images in that order or one horizontal turnaround sheet.
- **Pixal3D Native Texture** paints PBR materials on a new or existing mesh from one image or four views.
- **Speridlabs Idol** brings new weights to TripoSplat. Turn a single photo or render of an object into a 3D Gaussian splat you can rotate in the gallery or export as a PLY. Choose **Photograph** for a wider crop.

### WanGP Manager 0.1.14

- **Updates that reinstall less**: your Python environment is kept when it still fits the new version and your plugins.
- **Free disk space**: remove old rollback versions under **Update WanGP → Saved versions and rollback**.
- **More reliable start**: many plugins no longer make a start fail, and if one does fail, the console says why.

[Download it](https://github.com/deepbeepmeep/WanGP/releases/latest/download/WanGPManager.exe) and choose **Update in place**.

### Other
- **H3 Image**: [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image) by Pruna AI turns MiniMax H3 into an image model: **8-megapixel stills in 8 steps** without guidance, from text or by editing with up to 8 input images. Select **MiniMax H3 → P-H3-Image 8-Step 20B**. 
- **H3 Distilled Text Encoder**: choose **Distilled Qwen3-VL-4B** under **Text Encoder** on the H3 models for **about 5 GB instead of 27 GB** (INT8) and about half the encoding time.
## 30th of October 2026: WanGP v19.02 - More Treats Than You Can Handle

- **Showrunner**: from a brief to a finished production of up to ten minutes: a music video, a short film, an ad, a trailer or an illustrated story. Deepy Prime becomes your showrunner: it plans every shot, casts, shoots, checks its own takes, redoes the failed ones and assembles the cut ([Showrunner guide](docs/SHOWRUNNER.md)).
  - **The same cast from start to finish**: faces, outfits and voices stay consistent across dozens of shots, costume changes included, and long productions get titles and chapter cards.
  - **Music videos cut on the song**: shots follow its parts and bars, with lip-synced singers. No song yet? The showrunner composes one.
  - **You stay the director**: review the plan before any video is generated, approve takes or ask for reshoots, add new pictures or voices, and ask for changes in plain words at any time. Every change, yours or the showrunner's, can be undone.
  - **Instant editing**: the Timeline plays the cut at once, without rendering, and titles are drawn over the picture, so they are always sharp and spelled right.
  - **Every version kept**: compare versions side by side, go back to one or branch it into a new production. Exports come with subtitles and a levelled soundtrack.
  - **Works with your models**: Showrunner uses Deepy's templates, your own included, and can pause to free the GPU for other work. It needs Deepy Prime, with a local model or a remote agent.


- **Nemotron Transcription**: turn speech into text offline, with little VRAM: who said what (up to eight speakers, even talking over each other), when each word was said, subtitles, and translation to or from English, for recordings of up to an hour in 40 languages. **Q4 Experts** (32 GB of RAM) is the recommended version.
  - **Audio or video**: transcribe an audio file or a video's soundtrack.
  - **Jump in the recording**: the transcript shows under the Audio gallery player, as text or as a timeline per speaker; click a line or a turn to play from there.
  - **Subtitled videos**: get the video back with the transcript or translation as subtitles, shown or hidden with the new **CC** button of the Video gallery.
  - **Your names and words**: name the speakers, give the spelling of names and jargon, keep every hesitation or get clean text, with timings per turn or per word; translations can be timed and subtitled too.
  - **Deepy Prime**: ask for speaker-labelled transcripts, translations or subtitled videos once a Nemotron model is chosen in **Configuration > Prompt Enhancer / Deepy > Transcription**.

- **WanGP Manager (Experimental, Windows)**: [download the official app](https://github.com/deepbeepmeep/WanGP/releases/latest/download/WanGPManager.exe) for guided setup that selects the recommended GPU profile and installs Python and dependencies for you.\
  Keep several installations, update their dependencies and recommended kernels, try an older WanGP build or restore both source and environment from a saved version.\
  Launch WanGP in its own window or your browser; add existing Conda, uv or venv installations, update their dependencies and keep a rollback checkpoint.

- **Refreshed UI**:
  - **App bar**: apps sit side by side with an icon and a name, and the icons on the right open **Configuration**, **Apps & PlugIn Manager** and **Infos** (guides and About together).
  - **Full-screen apps**: Media Flow and Showrunner open full screen, also straight from `/media-flow/` and `/showrunner/`; **WanGP** brings you back where you were.
  - **Media Flow**: redesigned as an app, with help on its controls and its [guide](docs/MEDIA_FLOW.md) built in.

## 29th of October 2026: WanGP v18.888 - Are You Game?

- **Deepy Prime Can Now Build Apps**: turn an idea into a game, an interactive story, a music toy or an explorable 3D scene, with images, video, sound and 3D assets made by WanGP's generative models. Play in the chat, keep your creations in the new **Apps gallery**, save progress and share them as `.wgpapp` files. Deepy can keep improving an app in later conversations, including one someone else shared with you. This development preview is off by default: under **Configuration → Prompt Enhancer / Deepy → Python**, enable **Python Sandbox** and set **App Development (Advanced)** to **Web Apps**, then save. See the [Apps guide](docs/APPS.md).

- **3D Models, Organized by What You Want to Do**: start with **Create**, **Texture**, **Refine**, **Split**, **Animate** or **Reconstruct**, then choose a model for that task. Shape generation and texturing are now separate choices: use a preset to combine them in one generation, optionally adding new **Hi3D** refinement before also new **SUPRA-Tex**, **Hunyuan3D 2.1 Paint** or **TRELLIS.2 Texture** paints the result. Existing meshes can go straight to the matching tool with **To Input Mesh** in the 3D gallery. See the [3D models guide](docs/MODELS.md#3d-models).

- **H3 Regenerate for High-Resolution Video**: choose the **H3 Regenerate 2K 2 Phases** preset for a first pass at lower resolution followed by a new three-step refinement at the selected output resolution. It enhances fine detail while preserving the first pass's soundtrack. The previous workflow remains available through **H3 Original 2 Phases** and **H3 Vanilla 1 Phase (Reset)** returns to single-pass generation.

- **H3 Pika**: **six steps instead of vanilla H3's original 50**, with improved aesthetics, anatomy, prompt following and audio quality according to [Pika's evaluations](https://pika.art/blog/pika-h3-flex-research). Select the **Pika H3 Flex FL2VA / Ref2VA 6 Steps** accelerator profile.

- **H3 Singularity**: **clearer fast action, sharper distant faces and less glossy skin** than vanilla H3, according to [its author](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity). Singularity V2 focuses on combat, fantasy effects and stronger prompt/style control. Available in **33B** and **pruned 20B**.

- **More Room When RAM Runs Short**: WanGP automatically reclaims memory held by idle models to make room for the model doing the work. On Windows, it also frees cached checkpoint data left behind after loading, reducing RAM pressure when several generation and assistant models share the same PC.


- **Custom Sigmas**: pick your own *Denoising Sigmas* for H3, LTX2 and Wan

*Stories, rhythm games and 3D adventures made with Deepy. [Play all 10 demos and download their .wgpapp files](https://wangp.ai/deepy/apps/).*

## 27th of October 2026: WanGP v18.84 - The One World Generator

###  New World Generators
Create a world, then walk through it with the keyboard in the new **Worlds gallery**. Save your exploration to continue later, record the journey as a video, or capture a view for another generation. You don't need a 24 GB card to join in: **8 GB is enough to get started** and almost at **NO SPEED COST** thanks to **MMGPv4**. Don't expect miracles, though: near-real-time exploration depends on the model and a fast, recent GPU.

- **Hy-Nomas 8B** starts from a text description and remembers earlier views to help keep places and characters consistent. Walk around, turn the camera, or type an event such as "the fox sits down". Choose how much history it remembers and trade generation quality for quicker moves.

- **EVOKE Turbo 14B** starts from an image and keeps a memory of the scene's depth, helping places retain their layout when you return to them. It is a good fit for exploring streets, landscapes and interiors at **640×384**, though moves take longer to generate.

### Business as usual
- **H3 PTD RefAsymOPD**  another **high-quality PDD 8 steps**, the trick is here to apply a different LoRA at each step (as a consequence you will need 11GB extra RAM) 

- **H3 LoRAs Accelerators**: new **FL2V v1.3 / v1.1** and **Ref2V v1.0 / v1.1** accelerators.

- **Two MOSS Image Editors**: **Edit SFT** changes clothing, backgrounds, lighting or style and combines up to **four ordered references**. **Edit HQ v4** focuses on localized changes to a single image while preserving surrounding details; combining multiple images is more limited.

- **Fast Quantized KV Cache**: **MOSS Image Edit**, **Qwen Image 2.1** and **Wan Animate 2** now support quantized KV caches that make generation from image or video references much faster, even with **memory profiles 4 and 5**.

- **More Speed Boost and VRAM Savers**: update **WGPKernels to 1.4.3** and **GGUF kernels to 1.0.30** using the [installation guide](docs/INSTALLATION.md).

### Deepy Prime — Twice the Context, More Ways to Create

Thanks to the new **Q4 KV cache**, Deepy Prime can fit **roughly twice the context within the same VRAM budget**, or use less VRAM at your current context size.

Deepy Prime, your built-in personal assistant, can now:

- **Search the web** for facts and visual references, read pages and download assets. Web access is optional and off by default; no search API key is needed.

- **Find and install compatible LoRAs** from Civitai and Hugging Face for your chosen style or effect. Civitai downloads need your API key.

- **Create with Python**: add exact lettering, draw charts, assemble contact sheets, write timed subtitles, animate titles and build procedural 3D props in a restricted sandbox.

- **Automate batches** of image inspection, transcription and generation without filling the conversation with intermediate steps.

- **Generate 3D assets and worlds**, or continue a saved world from movement instructions before exploring it yourself.

- **Run Media Flow from chat**: restore, outpaint, upscale or smooth long videos and whole folders, with resumable batches and original audio preserved.


## 21st of October 2026: WanGP v18.21 - No GPU Left Behind

- **Profile 0 — More Speed with Less RAM (Experimental)**: a new, fully automated mode. WanGP gives fast Reserved RAM and spare VRAM to the model currently running, then makes those resources available to the next task. This makes switching between large models much friendlier to low-RAM PCs. Higher-end PCs may still get better results by choosing a profile manually.

- **RTX 20XX Gets a Second Wind**: older RTX cards now benefit from the new **WGPKernels 1.3.0** (please check *docs/INSTALLATION.md*) and faster local language-model processing. LTX-2.5 image and video VAE encoding, decoding, and upscaling are measured **4.8–6.8× faster**, turning jobs that were painfully slow into practical workflows.

- **Optimizations Across the Board**: many image, video, 3D, speech, and music models are now up to **10% faster** and use slightly less VRAM.

- **Faster Deepy, Prompt Enhancer, Speech and Music**: WanGP automatically selects the best acceleration supported by your GPU. When a language model is too large for VRAM, the CPU can work on the part left in RAM while the GPU handles the rest—often making responses about twice as fast. For instance, you can now run **Deepy Prime** with **Qwen 3.8 Bonsai** on an RTX 2080 Ti at **60 tokens/s**.

- **MiniMax H3 FastH3 Omni**: on RTX 30XX or newer, this new H3 variant combines the quality of the **PDD** process with **VSA sparse attention**. Best of all, it is an **omni** model that should work well for both **FL2VA** and **Ref2VA** workflows, which is why you will find it in the Ref2VA section. There is also an **NVFP4** variant that should bring the RTX 5090 close to real-time generation.

- **More Flexible H3 Two-Pass Generation**: the second pass has its own **Phase 2 Steps** setting and can use your selected LoRA accelerators instead of the automatic Turbo LoRA. Put accelerator multipliers before `|` in **LoRAs Multipliers**: for example, `1;0.6 |` uses the selected accelerator at strength 1 in phase 1 and 0.6 in phase 2. Use `1;0 |` to keep it out of phase 2 and retain the automatic Turbo LoRA there.

- **Fibo 2.0**: ranked as the **best open-source image generator**, it can also edit with up to **four references** or replace a masked area. **Edit Turbo** makes edits in only four steps. Prompts can be simple descriptions or structured captions, with extra help for rendering the intended text.

- **FlashVSR+ Turbo**: a new FlashVSR model that restores richer detail and runs faster at high resolutions. It supports images and videos, from **1× refinement to 4× upscaling**. Select it under **Configuration → Extensions → FlashVSR Spatial Upsampling → Model**.

- **Useful Extras and Fixes**: `--offline` prevents all downloads and clearly reports missing files. Linux DLSS Frame Generation x2 is easier to install, and several audio-model compatibility issues are fixed.



See full changelog: **[Changelog](docs/CHANGELOG.md)**


## 🚀 Quick Start

Choose one of these four guided installation methods:

1. **[Bat/SH scripts](#one-click-batsh-script-auto-installer)** — install and maintain WanGP from the included scripts.
2. **[Pinokio](#one-click-installers)** — use the community installer in Pinokio.
3. **[Wan2GP Desktop](#one-click-installers)** — use GKArtist's desktop launcher.
4. **[WanGP Manager](#wangp-manager-experimental-windows)** — the experimental official Windows app for guided installation, launching, updates and rollback.

Manual installation instructions follow these options below.

### One-click Bat/SH Script Auto-installer:

The 1-click automated scripts for both **Windows (`.bat`)** and **Linux/macOS (`.sh`)** make installation, environment management, and updates as seamless as possible. These scripts will not only install WanGP but also best acceleration kernels (Triton, Sage, Flash, GGuf, Lightx2v, Nunchaku) available for your config.

*👉 **Windows Users:** Double-click the `.bat` files. **Linux Users:** Run the `.sh` files in your terminal.*

#### **1️⃣ Installation (`scripts\install.bat` | `scripts/install.sh`)**

**Choose Installation Type**
- **Auto Install**
- **Manual Install**

**Manual Install**

If you selected Manual Install, you will be guided through:

1. **Choose your package manager**
2. **Name your environment**
3. **Select your Install Mode**

#### 2️⃣ Starting the App (`scripts\run.bat` | `scripts/run.sh`)
Once installed, use this script to launch the application. It runs WAN2GP using your active environment.

*   **⚙️ Customizing Launch Arguments (`args.txt`)**
    *   If you want to pass extra command-line flags to the launcher (like enabling advanced UI features or automatically opening your browser), create an `args.txt` file in your `scripts` folder.
    *   **Example `args.txt`:**
        ```text
        --advanced --open-browser
        ```

#### 3️⃣ Updating & Upgrading (`scripts\update.bat` | `scripts/update.sh`)
Use this script to get the latest updates for WAN2GP and upgrade dependencies.
* **1. Update:** Fetches the latest code from GitHub and updates requirements.
* **2. Upgrade:** Allows you to manually individually upgrade heavy backend components (like PyTorch, Triton, Sage Attention).

Triton recommendations follow both your GPU and selected PyTorch: **3.3.x with PyTorch 2.7**, **3.6.x with PyTorch 2.10** on RTX 30XX or newer, **3.2.x on RTX 20XX**, and **3.7.x with PyTorch 2.13** on AMD. Use **Upgrade** to correct an older Triton installation; **Update** alone does not change it.

#### 4️⃣ Managing Environments (`scripts\manage.bat` | `scripts/manage.sh`)
Use this script to manage and switch between your sandboxed environments safely.

* **Example Scenario 1: Migrating an Existing Setup**
    * If you have a folder named `venv` that works perfectly and want to use it with the new one-click scripts, run `manage.bat` and select **Add Existing Environment**.
    * Copy-paste the folder path (e.g., `C:\WAN2GP\venv`), select type `venv`, then use **Set Active Environment** to make it the default. Now `run.bat` and `update.bat` will target your existing setup.

* **Example Scenario 2: Testing New Configurations**
    * Let's say you have an environment named `env_stable` that works perfectly, but you want to try different components. Run `install.bat`, create a *new* environment called `env_testing`, and select **Manual Selection**. **Autoselect** installs the recommended stack for your GPU.
    * If the testing environment breaks, simply open `manage.bat`, select **Set Active Environment**, and switch back to `env_stable`. You are back up and running instantly.

---

### One-click Installers
- Pinokio installer
Get started instantly with [Pinokio App](https://pinokio.computer/)\
It is recommended to use in Pinokio the Community Scripts *wan2gp* or *wan2gp-amd* by **Morpheus** rather than the official Pinokio install.

- Wan2GP Desktop by GKArtist
[Wan2GP Desktop](https://github.com/GKartist75/Wan2GP-Desktop-Tauri) is a desktop launcher for Wan2GP that installs, updates, and runs it from one window — handling Git, Python, CUDA, and PyTorch setup so you don't have to configure them manually.

### WanGP Manager (Experimental, Windows)

**The fourth installation method is an experimental Windows x64 app.** WanGP Manager guides you through choosing an installation folder, a WanGP version and the recommended profile for your detected GPU. It installs Python and dependencies for you, then lets you launch WanGP in its own app window or your browser. You can also update a managed installation, try an older build or roll back to a saved version.

**Download:** [WanGPManager.exe](https://github.com/deepbeepmeep/WanGP/releases/latest/download/WanGPManager.exe). This permanent link downloads the latest Manager release.

Run `WanGPManager.exe`, open Manager and follow **Install / Add**. Already have WanGP? Choose **Use existing installation** at the top, browse to its folder and let Manager detect its saved, `env_uv` or venv environment. You can also select a Conda environment. Registration leaves your packages unchanged. Choose **Update WanGP → Prepare latest update**, then confirm **Update WanGP** after reviewing the result. No new WanGP installation is required. Existing-installation updates require a Git checkout and a Conda or virtual environment.

To update **Manager itself**, download the same installer, close Manager and run it. **Update in place** is selected for upgrades; it preserves your installation list and preferences. You do not need to uninstall first. **Preferences → Check for Manager updates** checks for a newer installer.

The official [WanGP repository](https://github.com/deepbeepmeep/WanGP) is selected by default. Model downloads, generation and Deepy remain in WanGP's web interface. To reuse existing checkpoints, add all of your checkpoint folders in WanGP's configuration.

### Manual installation: (for RTX20xx - RTX50xx)

```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.11.14
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

On Windows, install Triton in the same environment, using the command matching your GPU and PyTorch:

| GPU | PyTorch | Command |
| --- | --- | --- |
| RTX 30XX - RTX 50XX | 2.10 (recommended) | `python -m pip install -U "triton-windows>=3.6,<3.7"` |
| RTX 30XX - RTX 50XX | 2.7 / 2.7.1 | `python -m pip install -U "triton-windows>=3.3,<3.4"` |
| RTX 20XX | 2.7 or 2.10 (legacy Triton compatibility exception) | `python -m pip install -U "triton-windows>=3.2,<3.3"` |

Triton 3.2 on RTX 30XX or newer can crash YuE2's optimized INT8 ConvRot kernels because it lacks `triton.language.gather`. Upgrade Triton as above and restart WanGP. RTX 20XX must stay on 3.2 and uses the existing standard INT8 path; compatibility with newer PyTorch is not guaranteed upstream. RTX 50XX users should use PyTorch 2.10 / Triton 3.6 for WanGP's optimized INT8 and NV FP4 support. Linux normally receives matching Triton through PyTorch. See the **[Triton Installation Guide](docs/INSTALLATION.md#triton-installation)** for details and RTX 20XX limitations.

For optimized attention, install **SageAttention 1.0.6 on RTX 20XX** and **SageAttention 2.2.0 on RTX 30XX or newer**. SageAttention 2 requires an Ampere-or-newer GPU; GTX 10XX should use SDPA. On Linux, prebuilt SageAttention 2 wheels with CUDA Graph support are available from the **[WanGP SageAttention fork](https://github.com/deepbeepmeep/SageAttention/releases/tag/v2.2.0-linux.cg1)**. See the **[Installation Guide](docs/INSTALLATION.md#sage-attention)** for the platform-specific commands.

### Manual installation: (for GTX 10xx)

```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

#### Run the application:

```bash
python wgp.py
```
If you are low on VRAM, there is a trick to increase the amount of VRAM available (between 1GB and 5GB of VRAM to be gained depending on the GPU): *disable GPU Usage in your Web Browser*.

Run *scripts/start-chrome-no-gpu.bat* or *scripts/start-chrome-no-gpu.sh* to launch Chrome without using your GPU. 

First time using WanGP ? Just check the *Guides* tab, and you will find a selection of recommended models to use.

#### Windows: Diagnose and Free VRAM

If you still run out of VRAM on Windows with an NVIDIA GPU, open Command Prompt in the WanGP folder and list the applications using GPU memory:

```cmd
scripts\gpumem.cmd
```

This shows per-process VRAM usage, sorted from highest to lowest, followed by total GPU memory usage. To attempt to reclaim idle VRAM, stop generation, activate your WanGP Python environment, and run:

```cmd
scripts\gputrim.cmd
```

Trimming briefly applies memory pressure to encourage Windows to move idle GPU allocations to system RAM, then releases its allocations and displays updated usage. It can briefly stall GPU applications, and savings are not guaranteed. By default, it refuses if a process other than the Windows desktop compositor uses more than 1 GB; close GPU-heavy applications before retrying.

#### Update the application (stay in the current python / pytorch version):
In WanGP Manager, use **Update WanGP → Prepare latest update**, review the result and confirm **Update WanGP**. For another guided installer, use its update action or the included update script.

For a manual installation, get in the directory where WanGP is installed and:
```bash
git pull
conda activate wan2gp
pip install -r requirements.txt
```

#### Upgrade from Python 3.10, Pytorch 2.7.1, Cuda 12.8 to Python 3.11, Pytorch 2.10, Cuda 13/13.1 (for non GTX10xx users)
I recommend renaming first the old conda environment to avoid bad surprises when installing a different config in this old environment.

```bash
conda rename -n wan2gp  old_wan2gp
```

Get in the directory where WanGP is installed and:
```bash
git pull
conda create -n wan2gp python=3.11.9
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Once you are done you will have to reinstall *Sage Attention*, *Triton*, *Flash Attention*. Check the **[Installation Guide](docs/INSTALLATION.md)** -

if you get some error messages related to git, you may try the following (beware this will overwrite local changes made to the source code of WanGP):
```bash
git fetch origin && git reset --hard origin/main
conda activate wan2gp
pip install -r requirements.txt
```
When you have the confirmation it works well you can then delete the old conda env:
```bash
conda uninstall -n old_wan2gp --all  
```

#### Run headless (batch processing):

Process saved queues without launching the web UI:
```bash
# Process a saved queue
python wgp.py --process my_queue.zip
```
Create your queue in the web UI, save it with "Save Queue", then process it headless. See [CLI Documentation](docs/CLI.md) for details.

## 🐳 Docker:

**For Debian-based systems (Ubuntu, Debian, etc.):**

```bash
./run-docker-cuda-deb.sh
```

This automated script will:

- Detect your GPU model and VRAM automatically
- Select optimal CUDA architecture for your GPU
- Install NVIDIA Docker runtime if needed
- Build a Docker image with all dependencies
- Run WanGP with optimal settings for your hardware

**Docker environment includes:**

- NVIDIA CUDA 12.4.1 with cuDNN support
- PyTorch 2.6.0 with CUDA 12.4 support
- SageAttention compiled for your specific GPU architecture
- Optimized environment variables for performance (TF32, threading, etc.)
- Automatic cache directory mounting for faster subsequent runs
- Current directory mounted in container - all downloaded models, loras, generated videos and files are saved locally

**Supported GPUs:** RTX 40XX, RTX 30XX, RTX 20XX, GTX 16XX, GTX 10XX, Tesla V100, A100, H100, and more.

## 📦 Installation

See the [four guided installation methods](#-quick-start), including **[WanGP Manager (Experimental, Windows)](#wangp-manager-experimental-windows)** and its permanent download link. The guides below cover manual installation and optional components.

### Nvidia
For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions for GTX 10XX, RTX 20XX to RTX 50XX
- **[Optional DLSS 5 Upsamplers](docs/DLSS5.md)** - Native-resolution refinement, spatial upsampling, and Frame Generation runtime setup


### AMD
For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/AMD-INSTALLATION.md)** - Complete setup instructions for RDNA 4, 3, 3.5, and 2

## 🎯 Usage

### Basic Usage
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - First steps and basic usage
- **[Models Overview](docs/MODELS.md)** - Available models and their capabilities
- **[Prompts Guide](docs/PROMPTS.md)** - How WanGP interprets prompts, images as prompts, enhancers, and macros

### Advanced Features
- **[Deepy Assistant](docs/DEEPY.md)** - Launch Deepy in Gradio, CLI or standalone Web mode; configure tools, media references, saved sessions and phone access
- **[Authentication, HTTPS, and Reverse Proxies](docs/AUTHENTICATION.md)** - Protect web access, configure proxy hosting with `--public-url`, and set up MCP OAuth
- **[Remote LLMs](docs/REMOTE_LLMS.md)** - Configure Codex, Claude Code, and OpenCode providers for Deepy and Prompt Enhancer
- **[Loras Guide](docs/LORAS.md)** - Using and managing Loras for customization
- **[Finetunes](docs/FINETUNES.md)** - Add manually new models to WanGP
- **[VACE ControlNet](docs/VACE.md)** - Advanced video control and manipulation
- **[Processing Guide](docs/PROCESSING.md)** - Preprocessing, masks, sliding windows, and postprocessing
- **[Command Line Reference](docs/CLI.md)** - All available command line options

## 📚 Documentation

- **[Changelog](docs/CHANGELOG.md)** - Latest updates and version history
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 📚 Video Guides
- Nice Video that explain how to use Vace:\
https://www.youtube.com/watch?v=FMo9oN2EAvE
- Another Vace guide:\
https://www.youtube.com/watch?v=T5jNiEhf9xk

## 🔗 Related Projects

### Other Models for the GPU Poor
- **[HuanyuanVideoGP](https://github.com/deepbeepmeep/HunyuanVideoGP)** - One of the best open source Text to Video generators
- **[Hunyuan3D-2GP](https://github.com/deepbeepmeep/Hunyuan3D-2GP)** - Image to 3D and text to 3D tool
- **[FluxFillGP](https://github.com/deepbeepmeep/FluxFillGP)** - Inpainting/outpainting tools based on Flux
- **[Cosmos1GP](https://github.com/deepbeepmeep/Cosmos1GP)** - Text to world generator and image/video to world
- **[OminiControlGP](https://github.com/deepbeepmeep/OminiControlGP)** - Flux-derived application for object transfer
- **[YuE GP](https://github.com/deepbeepmeep/YuEGP)** - Song generator with instruments and singer's voice

---

<p align="center">
Made with ❤️ by DeepBeepMeep
</p>
