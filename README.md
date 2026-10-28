# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Generative Models Accessible to the GPU Poor</b>
</p>

WanGP is a one-stop super app for the best open source generative models across video, image, audio, and text-to-speech.

## Highlights

| Modality | Supported models |
| --- | --- |
| **Video** | **Wan 2.1/2.2** and derived models, **MiniMax H3**, **LTX-2/2.3/2.5**, **Hunyuan Video 1/1.5**, **LongCat**, **Kandinsky**, **LTXV**, **MagiHuman** |
| **Image** | **Krea 2**, **Qwen Image**, **Z-Image**, **Flux 1/2** (*Klein*, Chroma), **SenseNova**, **Ideogram 4**, **HiDream** |
| **Audio / TTS** | **Qwen3 TTS**, **MiniMax H3 Voice Clone**, **Ace Step 1/2/XL**, **Omnivoice**, **Index TTS2/2.5**, **KugelAudio**, **HeartMula**, **Chatterbox**, **Minimax Music**, **Stable Audio 3** |

### Run More Models on More Hardware

- **Low VRAM requirements**: run select models with as little as **6 GB of VRAM**.
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

### Built-In Creation Tools

- **Video, image, and audio galleries**: browse generations and reuse them as new inputs.
- **Reusable settings**: extract settings from any generation, create templates, and share them.
- **Per-model prompt enhancer**: improve prompts with model-specific syntax and expectations.
- **Input preparation tools**: use the mask editor, background remover, pose/depth/flow extractors, speaker diarization, and background noise/song remover.
- **Deepy low-VRAM offline agent**: orchestrate generation jobs and tedious tasks such as transcription, video splitting, and color-frame generation while you are away.
- **AI Agent Orchestrator**: route high-level tasks to specialized agents, track their status, and expose explicit task tools for the existing Deepy/plugin system. See [docs/AGENT_ORCHESTRATOR.md](docs/AGENT_ORCHESTRATOR.md).
- **Temporal and spatial upsampling**: improve outputs with RIFE, FlashVSR, and Lanczos.
- **Audio postprocessing**: generate soundtracks with MMAudio, replace voices with SeedVC, or remux a video with any soundtrack.
- **Ready-to-use plug-ins**: Gallery Browser, Motion Designer, Models/Checkpoints Manager, CivitAI browser and downloader, and more.

**Discord Server to get Help from the WanGP Community and show your Best Gens:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

**Official WanGP Web Site**: https://wangp.ai/

> [!IMPORTANT]
> **WanGP is free to use locally.** The official project will never ask you to pay a license fee, subscription, or donation to run WanGP on your own computer (see the license for terms).
>
> **Use only the official GitHub repository or wangp.ai / wan2gp.ai websites. WanGP is not affiliated to any other third-party service using the WanGP/Wan2GP names**, unless explicitly stated here.


## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)


## 🔥 Latest Updates : 

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

- **More Speed Boost and VRAM Savers**: update **WGPKernels to 1.4.2** and **GGUF kernels to 1.0.29** using the [installation guide](docs/INSTALLATION.md).

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


## 18h of October 2026: WanGP v18.19 - More Drama per Gigabyte

- **Flux 3 Klein 5.6B**: text-to-image generation and editing with up to **10 reference images**. Choose the distilled model for **16-step generation**, or Base for **50 steps with guidance**. Both come in BF16, INT8 ConvRot and NVFP4. The new **Prompt Helper** lets you draw layout boxes, place elements from reference images, and mark objects for removal. Boxes guide the composition; they do not act as exact pixel masks.

- **Fish Audio Drama 3**: give your dialogue a cast and a performance. Write speaker lines with emotions, whispers, laughs and pauses, and clone up to **three voices** from recordings. Missing reference transcripts can be filled in automatically. Long scripts can be generated in parts while keeping recent speech as context. The **INT4 Experts** variant roughly halves the experts' memory requirements and speeds up their CPU execution, with a possible small change in delivery. Select **vLLM** for the fastest supported engine.

- **New WanGP Kernels**: **WGPKernels** bring higher-precision INT8 math, with performance close to Comfy Kitchen and faster in some workloads. Results depend on the model, GPU and precision setting. Install the matching package from the [installation guide](docs/INSTALLATION.md#wgpkernels-installation), then select **WGPKernels** under *Config → Performance → INT8 Math Kernels*.

- **H3 LightVAE**: a lighter **12-layer video decoder**, available in **BF16** and **INT8 ConvRot** through the **Video VAE** selector for FL2VA and Ref2VA. It keeps the original H3 encoder for start/end frames and references. Decoded detail can differ from the full VAE; selecting H3's x2 VAE upsampler uses the dedicated x2 decoder instead.

- **New TaoMate H3 FL2VA Profile**: the **15 October 2026** release gets its own dated **3-step** accelerator profile, alongside the original TaoMate profile. This new LoRA is specific to **FL2VA** and downloads automatically when selected.

- **Deepy Prime Got More Efficient, and a Bit Artsy**: give it a one-minute brief (an ad, a short film, a music video, a podcast clip) and it plans the shots, keeps the cast consistent from shot to shot, paces the cuts to the genre and puts music or narration under the whole film. Before handing the result over, it measures and fixes what used to slip through: black frames, unwanted cuts inside a generated shot, silent stretches and typos in on-screen text. Titles, lower thirds, labels and app screens can now be drawn over the footage with exact lettering, while subtitles stay a separate track you can switch off. 

- **MOSS Image 5B**: a new text-to-image family from **OpenMOSS**, with **Base** and **SFT** variants. Describe a scene in plain language, use tags, or arrange text and objects with the **Prompt Helper** layout editor. SFT adds training on curated data; both start at **50 steps**. The optional layout enhancer follows your selected canvas size. Boxes guide the composition; text and placement can still vary.

- **MOSS-TTS v1.5**: **8B** and **Local 4B** multilingual speech models with cloning of up to **three voices**, dialogue, adjustable pauses and optional script writing. Both support **BF16 / INT8 ConvRot** and **PyTorch / CUDA Graphs / vLLM**; Local 4B outputs **48 kHz stereo**.

- **Krea 2 Edit v2**: **General Edit** for changes to colors, lighting, objects and style; **Identity Edit** for changing poses, outfits and scenes while preserving likeness. Both use **Krea 2 Turbo**, support **two reference images** and start at **8 steps**.

*Update v18.17*: added MOSS Image\
*Update v18.19*: lots of bugs fixes, MOSS TTS, Krea Edit v2

## 11th of October 2026: WanGP v18.00 - Distortion of Space and Time

- **LTX 2.x VAE x3 faster**: the LTX 2 VAE was rewritten in WanGP: 20s of 480p (24fps) video now takes about 28s to generate on an RTX 5090 using Profile 4. Realtime, we can see you from here...

- **Wan & HunyuanVideo VAEs x2 to x3.5 faster**: I didn't want these two to feel abandoned. The Wan 2.1 VAE is up to 3.5x faster (Wan 2.2 up to 1.6x) and needs less VRAM on top of it: decoding 1080p goes from 28 s / 10.5 GB to 11 s / 4.7 GB. HunyuanVideo decodes about 2x faster. Bonus for all three: tiling seams are history on Wan (tiled = untiled) and much fainter on LTX 2 and HunyuanVideo.

- **Kandinsky 6.0**: new video generator with synchronized audio (speech with lip-sync, music, sound effects), Pro 30B and Lite 3B, both distilled. Clips are short for now (5 s), but the WanGP version is low VRAM as usual and its VAE is 2x faster than the official one.

- **Wan 2.2 Prism Video**: image to video with audio, derived from the Wan 2.2 architecture, natively up to 2K.

- **Wan 2.1 Camera Control**: *Fun Control-Camera 1.3B* and *Uni3C 14B* take a start image and fly the camera along the path you choose.

- **CUDA Graph Acceleration**: kills the CPU time that keeps your GPU waiting. Mostly visible with lighter GPU work (image models) and on modest CPUs: SenseNova +30%, Qwen Image 2.1 +10%, H3 about 0%.

- **RTX 20xx / RTX 30xx**: these GPUs missed part of the v17 VRAM diet. Fixed, they now get the full menu.

- **Faster Startup on RTX 50xx, Happier SSD**: on Blackwell GPUs, the NVIDIA driver optimizes the GPU code again at every launch and keeps the result in a shared 1 GB cache, too small to hold it all: 10-15 s lost at each first generation, and a cache rewritten on your SSD every time. Tick *Config → Performance → WanGP Compute Cache* to give WanGP its own cache (up to 2 GB): the work is done once and for all.

- **3D Gen Models**: WanGP enters the third dimension, with plenty of synergies with the other models: rotate a 3D asset in the new *3D Viewer*, snap a view at the resolution you want and use it as a start image, or turn your generated images and videos into 3D.
  - *One picture to a 3D asset*: **Hunyuan3D 2.1** (mesh + PBR textures), **TRELLIS.2** (textured PBR mesh), **TripoSG** (detailed mesh), **TripoSplat** (Gaussian splats), **PartCrafter** (one mesh per part)
  - *Photos or video to a 3D scene*: **WorldMirror 2.0**, **MapAnything**, **Depth Anything 3 Points**, and **MoGe-2** from a single photo
  - *Mesh tools*: **GeoSAM2** splits a mesh into its parts, **AnimateAnyMesh** animates a mesh from a text description

- **Deepy Goes 3D**: Deepy can now look at a 3D asset from any angle, zoom or animation frame, draw layouts in SVG, and has a new *Character Sheet* skill: one 3D character in, a turnaround sheet out.

- **H3 Veda Sparse**: MiniMax H3 gets learned sparse attention, with up to **2x faster generation** for higher-resolution or longer videos. Available for both the full 33B and pruned 20B FL2VA/Ref2VA models, using the same settings and LoRAs. 

- **Workspace Folders**: keep each project's generated and imported media in its own folder for easier organization and backups. Choose dedicated folders for new workspaces, or use **Gallery Manager** to convert an existing workspace or choose a different storage folder. To enable this feature: *Config / Outputs* tab

- **Expanded MCP Support for Agents**: external agents gain new tools to validate generation settings, manage their own workspaces, inspect images, videos and 3D assets, and transfer files. Agent activity is now visible in the Gradio interface, so you can follow their work.

- **UI Improvements**: the loaded model's name stays visible in the status area, including between generations. Set a custom target resolution with the **megapixel slider** and **Aspect Ratio** selector, while compact buttons and adaptive controls make the interface fit more comfortably in narrow windows.

## 7th of October 2026: WanGP v17.17 - With Great Speed Comes Even Less VRAM

I went to the future and brought back **MMGP v4** for you. 

**15 seconds of H3 video at 1080p used to require  25 GB of VRAM,  now you just need 11 GB of VRAM**.  Everybody wins, only 5-6GB of VRAM to gen 15 seconds of H3 video at 480p. 

Pair MMGP with WanGP, and your GPU punches above its weight:

- **Profile 4 is up to 25% faster and uses up to 50% less VRAM** when generating large videos. And that's on top of the optimizations that already made WanGP the go-to *Low VRAM App*. Even better: in many cases, Profile 4 now beats the old Profile 1 and matches the new Profile 1.

- **Good old failsafe Profile 5 is up to 50% faster.** Even the safety net got a turbo boost.

These VRAM savings should benefit everyone, especially at higher resolutions and longer durations. 

To get the most out of this update, first make sure you use *Sage2/2+ Attention* as quite a few optimizations depends on it. Then open the new *Config → RAM/VRAM Management* tab, choose an *MMGP Optimized VRAM Allocator* option under *VRAM Allocator*, and turn on *Smart Memory Pinning*. Restart WanGP if you change the allocator.

For an extra **20% VRAM saving**, with a speed penalty of up to 10%, go to *Config → Performance* and set *Attention Head Split* to *Medium (good balance)*. A pretty good trade-off when every gigabyte counts.

**Dynamic VRAM Preload** fills spare VRAM with model weights and frees space as needed. Set *VRAM Preload* to *Dynamic* for less waiting between layers, especially when generating images.

MMGP v4 puts more of your rig to work: PCIe bandwidth, SSD/NVMe speed, idle CPU cores... Actual gains will depend on your hardware. And to make updates easier, **WanGP and MMGP now live together in the same repository**.

There are also two new upsamplers to play with:

- **LTX 2.5 Detail Refiner x2**: high-quality upsampling with tiling. Just don't expect it to win any speed races.
- **H3 VAE Upsampler x2**: double the width and height of H3 videos and images as part of VAE decoding.

By pure coincidence, WanGP has just hit **10,000 GitHub stars**. If you're enjoying it, spread the word! Plenty of fellow GPU Poors could use the extra breathing room. Let's take together WanGP to **100,000 stars**.

### v17.10: LTX-2.5 VFX Tools

- **LTX-2.5 SDR to HDR**: LTX-2.5 now converts SDR videos to HDR10 with its own dedicated HDR LoRA (LTX-2.3 keeps its previous one). HDR videos in the *Video Gallery* also get an automatic SDR preview, so they no longer show up as a black clip in your browser.
- **LTX-2.5 Layout to Render**: turn a rough viewport animation or playblast into a finished shot. Feed it the layout video plus one appearance reference image (a finished render of its first frame): the layout drives the camera and object placement, the reference sets the look.
- **LTX-2.5 Alpha Gen**: extract a soft alpha matte from any video, optionally restricted by a selection mask, and get the original footage back with transparency as PNG frames (ZIP) or ProRes 4444. Pick the format in *Config → Outputs → RGBA Video Output*.
- **Two Phases with Tiling** (LTX-2, 2.3 and 2.5): a new *Guidance Phases* choice. Phase 1 lays out the whole scene, then phase 2 refines it in overlapping tiles blended at every step. It is particularly suited to generate 4K because LTX will see in the worst case 2K tiles (bye bye freak show). Don't be surprised if it is slower since proper tiling requires a 50% overlap. 

### v17.17: New MMPG RAM Allocator
New Allocator should reduce now RAM usage: open the new *Config → RAM/VRAM Management* tab, choose *MMGP Optimized RAM Allocator* in *RAM Allocator* 

## 29th of September 2026: WanGP v13.141 — Community Release

Thanks to **WanGP community** contributing code, testing and feedback!

**Community Contributions**

- **Live Video Previews**: watch motion develop during generation! Under *Configuration / General / Generation Preview*, choose legacy fast RGB frames, clearer Tiny VAE frames, or a looping Tiny VAE video. Click the video to pause/resume it. Thanks *@GOvEy1nw*!

- **YuE2 Score Continuation**: the *Composition source* menu lets you import or extend an ABC score, including scores transcribed from audio. Thanks *@Beanow*!

- **FLAC Audio**: import FLAC, export 16-bit lossless audio with embedded generation settings, and use FLAC soundtracks in MP4/MKV. Thanks *@Beanow*!

- **Balanced x265 Exports**: **CRF 18 (Balanced)** joins **CRF 28 (Small)** and **CRF 8 (High Quality)**. Thanks *@Beanow*!

- **More Patient Media Flow Exports**: saving and continuation merges now allow five minutes instead of 30 seconds. Thanks *@kito408*!

- **The Enhance Button Stays Put**: manual enhancement remains available when automatic enhancement is enabled. Thanks *@GKartist75*!

**Other Additions and Improvements**

- **H3 Welcomes More References**: Ref2VA now accepts up to **three reference videos and three audio references**, alongside its image references. Reference videos share a 15-second budget; audio references share a separate 15-second budget.

- **H3 Reference Excerpts**: pick up to three moments from one reference video, or independently from its soundtrack. Enter positions and optional durations, such as `5s/3s` for a three-second excerpt centred on five seconds.

- **H3 Keeps Your Soundtrack**: Ref2VA can now build a video around uploaded speech/music or a reference video's soundtrack, keeping the original audio synchronized across sliding windows. Select an audio choice marked *Soundtrack Kept*.

- **H3 Learns to Hold Still**: select **Text to Image** for a single still. The prompt enhancer gains dedicated image prompts, including reference-based edits. 

- **H3 Ref2VA Frames Injection**: Ref2VA should be now on par with Fl2VA. But keep in mind, you will get better results with Fl2VA for Start,End,Injected Frames and Text2Video

- **H3 In-Context LoRAs**: **Use Control Video as In-Context Guide** enables compatible face/character-swap LoRAs. The previous editing mode is renamed **Denoise Control Video (Video-to-Video Edit)**.

- **H3 ControlNet-Union 2.0**: **FL2VA 33B and Pruned 20B** gain pose, depth, edge, shape and recoloring controls, plus inpainting and spatial outpainting. 

- **Floating Generate Buttons**: controls float on the right until their normal position is visible; Edit Mode buttons float too. Enabled by default in **Configuration / General**.

- **Three Voices, One Conversation**: three uploaded voice references now work with **H3 Audio, Qwen3 TTS Base, OmniVoice, KugelAudio, IndexTTS2/2.5, DramaBox and Scenema**. Use `Speaker 1:` through `Speaker 3:`; Scenema applies its references through SeedVC.

- **Krea 2 Real and HD VAEs**: Krea 2 and compatible Qwen Image 20B/Edit models gain a *VAE* dropdown: try **Real** for photographic textures or **HD** for detail and contrast.

- **More LoKr Compatibility**: Krea 2, Qwen Image, Ming Image and Qwen Image 2.1 gain support for compatible decomposed LoKr adapters.

- **Deepy Learns Magic Mask**: ask Deepy to mask named objects in an image or video, invert the selection, or create simple rectangular masks. The masks go straight into the Gallery for editing.

- **Deepy Handles RGBA**: convert between RGB and RGBA, extract or recombine color channels, and use a grayscale mask as an image's transparency channel.

- **Deepy Prime Mixes and Muxes**: replace a video's soundtrack, mix audio sources, keep separate selectable audio tracks, or add selectable subtitles. The video is preserved without re-encoding.

- **Faster Deepy and Prompt Enhancement**: **GGUF CUDA kernels 1.0.25** accelerate speculative decoding for Q4_K and Bonsai PTQ1_0 without extra VRAM. See the [installation guide](docs/INSTALLATION.md).


## 24th of September 2026: WanGP v13.1315 — It's Your Lucky Day^?!

- **Qwen Image 2.1**: A new Qwen Image model with out-of-the-box editing capabilities and strong text rendering. 

WanGP supports *VRAM Optimization*, *Pose/Depth/Edge Transfer*, *Inpainting*, *LanPaint*, *Outpainting*, *KV Cache Acceleration*, and *Enhanced Prompts* from day one.

The base model works without an accelerator; optional LoRA profiles below offer faster 4- to 8-step generation.

- **Full JIT Checkpoint Loading**: WanGP now downloads preprocessing and postprocessing checkpoints only when needed, instead of downloading them all on first use. This saves disk space if you use WanGP for a specific task, such as audio generation, or want to build a portable version of WanGP.

- **Comfy Kitchen Kernels Support**: These kernels are installed automatically and can accelerate specific models. They have been integrated into H3 and LTX2.x, which are now 10% faster.

- **Minimax H3 INT8 ConvRot VAE**: Twice as fast. Now downloaded automatically and used by default when INT8 is selected in your transformer configuration. 

- **Deepy Prime for the Masses**: Thanks to the new *Bonsai 2 Abliterated PTQ1_0 checkpoint*, *Deepy Prime* and its advanced *Prompt Enhancer* can now run with **10 GB of VRAM, or possibly less**. You will need *GGUF CUDA kernels 1.0.25*. Please see the installation guide.  

- **UI Optimizations**: The UI should be even faster, especially the Image/Video Gallery.

- **Yue2 Instrumental Model Only**: you can now generate instrumentals track only, please check new option in main dropdown box and also prompt instructions or new prompt enhancer templates. 

- **Yue2 Hum to Song**: Turn a clear hummed melody, your lyrics and a music style into a new stereo song.

- **YuE2 support AR LoRAs**: YuE2 first phase, is an Auto Regressive phase (score writing phase) and it can now accept LoRAs. LoRAs for second Diffusion phase was already added previously.

- **DFlash2 & DSpark Acceleration for Deepy & Prompt Enhancer***: for an extra VRAM cost you will be able to generate up tp 200 tokens / s

- **Various UI Improvements**: add directly the current frame of Video Gallery to the Image Gallery, Extract settings of a Video/Audio/Image in Workspace Gallery 

- **Ming Image 0.1 Design**: a new image generation and editing model for detailed infographics, posters, and carefully arranged layouts. It accepts standard text prompts and a structured JSON format for more precise control over composition. WanGP's visual **Prompt Helper** lets you arrange layers and edit their positions, colors, and descriptions. You can also choose a classic or JSON prompt enhancer to turn a short brief into a detailed prompt with explicit titles, labels, and explanations.

- **Ming Image 0.1 Design-Layer**: turn a finished poster, infographic, or other design into separate transparent image layers. Provide one reference image and a text prompt describing the layers from front to back. WanGP adds every RGBA layer to the image gallery and saves a ZIP of the complete layer set. Design and Design-Layer share their language encoder and vision tower checkpoints to reduce downloads and disk use.

- **Qwen Image 2.1 Viggle Turbo LoRA Accelerators**: generate images faster with **v0.1 (4 steps)**, **v0.2 (5 steps)**, or **v0.2.1 (6 steps)**. Select an acceleration profile to apply the matching LoRA, scheduler, and recommended settings. The 6-step profile is the recommended Viggle starting point and is also available in Deepy for image generation and editing.

- **Qwen Image 2.1 Pruna LoRA Accelerators**: choose **Pruna v0.1 (5 steps)** for speed or the recommended **8-step** profile for better quality. Each profile applies its matching LoRA and Pruna scheduler with the correct sigma sequence, guidance 1, and LoRA strength 1.; Pruna's adapters were trained for edits with up to three reference images.

- **Remove Vocals Audio Postprocessing**: create an instrumental version of an audio track by removing its vocals. Available in **Post Processing**, **Late Post Processing**, and through **Deepy Prime**.


*Update v13.1313*: YuE2 instrumental mode, YuE2 Hum to Song, AR LoRA support, and DFlash2 / DSpark acceleration\
*Update v13.1314*: Ming Image 0.1 Design and Design-Layer, Qwen Image 2.1 Viggle Turbo and Pruna LoRA accelerators, and Remove Vocals audio postprocessing

## 16th of September 2026: WanGP v13.10 — It's Your Lucky Day!

**WanGP Major Release**

- **Upgraded User Interface**: a more responsive interface for switching models and browsing media, a modern look, and five color themes, including Classic Gradio. Use the microphone beside a prompt to dictate it, then review and edit the text before generating.

- **Access Anywhere**: start a generation at home and follow it from another browser or device connected to the same running WanGP server (VPN recommended for remote access). Galleries, selected media, progress, and the generation queue stay synchronized, so you can check results and add new jobs. Each Gradio page keeps its own unsent prompts and draft settings.

- **Clearer Progress and Cancellation**: see what WanGP is doing while it downloads files, loads models, encodes prompts, or generates media. Progress bars now cover more stages, and you can request cancellation during preparation as well as generation.

- **Workspaces**: keep generated and imported media organized by project, with collections and selections remembered after restarting WanGP. Switch workspaces between tasks, or open the workspace manager beside the gallery selector to browse the full collection, reorder items, copy or move them between workspaces, and download a selection as a ZIP. Copying media between workspaces does not duplicate the files on disk. See the [workspace guide](docs/WORKSPACES.md).

- **Deepy Web App**: take Deepy with you in a phone-friendly interface. Upload a photo or recording, describe what you want, and follow the conversation and results from your phone or desktop. Open **Web app →** in Deepy's settings to find it, and add it to your phone's home screen for quick access. See the [Deepy guide](docs/DEEPY.md). You can use this way both *Deepy Zero* & *Deepy Prime*, although you will get best results with *Deepy Prime*.

- **Prompt Enhancer Upgrades**: when using a Qwen3.5/3.8 powered Prompt Enhancer, now all attached images (Start Image/End Image/Control Image/Ref Images) may be used to produce the enhanced prompt. The prompt enhancer is also given the duration of the video / sliding window you want to generate (this works also when prompt commands like [\duration=3s] are used). Even better multiple sliding windows prompts can be enhanced at the same time based on the actual start/end frames they will see.

**New Models**

- **YuE2**: turn lyrics and a musical style into a complete song with vocals and accompaniment. The cover workflow can extract melody and chords from a source song to guide a new performance; supply the lyrics separately and keep them aligned with the original sections.

- **AuK Speech**: generate speech from written instructions, or use a source recording for voice cloning, spoken-word edits, speech cleanup, and speaker separation. Choose **Flash** for a fast four-step result or **Base** for more control. Start with a short clip and describe both what to change and what to preserve.

- **LTX2.5 updates**: added LTX 2.5 MSR (reference to videos), LTX 2.5 Ingredients is now used when Ref. Images are provided, updated Media Flow processes with LTX 2.5 unblur and uncompress LoRAs

*update 13.10*: Prompt Enhancer Upgrades, LTX2.5 updates

## 6th of September 2026: WanGP v12.72 — Power Up, Polish, Pause
- **H3 VDN**: at least 20% Faster and even more on larger / longer videos, requires a bit more VRAM and Triton must be installed

- **DLSS 5 Neural Rendering — More Than an Upscale**: give finished images and videos an AI polish that can enrich lighting and material appearance, improve fine features such as skin, hair, fabric, and foliage, and keep enhancements stable over time. Use x1 for native-resolution refinement or x1.5–x3 to refine and enlarge, with adjustable intensity. WanGP estimates depth and motion for recorded media, so results remain content-dependent. See the **[DLSS 5 overview](https://research.nvidia.com/labs/adlr/DLSS5/)** and **[installation guide](docs/DLSS5.md)**.

- **Fast Temporal Upsampling up to x6**: DLSS Frame Generation uses a native RTX path designed for real-time multi-frame generation, so it should generally run faster than RIFE on supported hardware while turning low-FPS footage into smoother video. It supports x2–x4 on compatible RTX 40/50 GPUs and x5/x6 where supported on RTX 50. **RIFE v4.26 now adds x3 temporal upsampling** alongside x2/x4, with no extra native DLSS runtime.

- **Media Flow / Temporal Upsampling & Neural Engine**: DLSS5 / Rife Temporal Upsampling and DLSS5 Neural Engine can now be used in Media Flow, so you can convert your entire collection of stop motion movies.

- **H3 Voice Audio**: a new TTS preset reuses MiniMax H3 Ref2VA Pruned for voice cloning and general audio generation. It accepts one or two audio references, under the hood it denoises a hidden 32x32 video for speed, skips video decoding, and saves only 32 kHz stereo audio.

- **H3 Outpainting**: discover the truth that lies beyond the borders of the H3 frames, also available in *Media Flow*

- **H3 Audio Refinement Extra Phase:** optionally improve the soundtrack after FL2VA or Ref2VA video generation using 6 extra steps without LoRAs. WanGP preserves the original full-resolution video and audio latents, locks the video exactly, partially re-noises only the audio at 0.5 denoising strength, and then decodes the refined result. The pass deliberately does not re-inject reference media or the original Control Video. It is unavailable when an FL2VA soundtrack controls generation and on fixed 8-step PDD variants.

- **Viggle Animate**: a H3 Based model similar to Wan Animate but 3 steps only. Give Viggle a Control Video and an Edited Frame  (the person or object to animate injected in one frame taken from the control video, use a WanGP Image Editor or ask Deepy). Motion is quite good, but model will need to build 5s long sliding windows.

- **Deepy Goodies**:
  - **Pause and resume** Deepy without losing its progress. Pausing temporarily releases its GPU and VRAM resources for another WanGP task or application; an active tool is allowed to finish safely first.
  - **Optional sessions** continuously preserve the conversation, displayed cards, workspace, and media used or generated by Deepy. Resume past work, switch sessions, or rename, duplicate, export, and delete them from the interface.
  - **Faster responses** through decoding optimizations that can improve speed from 50% to 100%, depending on the model and configuration. (please update to the latest *GGUF 1.0.24 kernels*, see *docs/INSTALLATION.md*)
  - **Qwen3.8 27B IQ3_S** offers a new middle ground between Q2 and Q4: better quality than Q2 with a lower memory footprint than Q4, while remaining compatible with faster speculative decoding. It may make 16 GB VRAM configurations practical with suitable context settings.

  
*update WanGP v12.72**: H3 Outpainting, Viggle Animate


See full changelog: **[Changelog](docs/CHANGELOG.md)**


## 🚀 Quick Start

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
If using Pinokio use Pinokio to update otherwise:
Get in the directory where WanGP is installed and:
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
