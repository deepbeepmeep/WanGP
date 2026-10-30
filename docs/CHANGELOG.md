# Changelog

### Unreleased

- **Save Latents:** the new config option makes Wan, LongCat, LTX2 and MiniMax H3 save the latent right before the final decode, as a `*_latent.pt` companion file next to each video. Sliding-window videos save one latent per window, and LTX2 and MiniMax H3 also save their audio latent, so the file holds the video the way the model saw it. A *Save Latents* choice in the *Misc.* tab of the generate form overrides the option for a single generation: *On* writes the companion file, *Off* skips it, and the default *Configuration* option follows the *Configuration* tab, which is also what Deepy and API jobs use. The choice is stored with the generation's settings, so it keeps its value across form refreshes.

- **Latent Decode:** the new process in *Media Flow* takes a video, finds its `*_latent.pt` companion file, and turns the file back into video without loading the diffusion model. For LTX2 and MiniMax H3 it also rebuilds the soundtrack and puts it back into the video.

- **From WanGP latent file:** a new option in *Continue Video* and *Continue Last Video* starts the job from the saved latent file instead of re-encoding the part you keep. The two pickers form one pair seen from either end: the latent file is the input — you pick it with the new file picker, or WanGP matches it to a selected video — and WanGP checks it before any model load. Picking a file selects its companion video (the video the file was saved from) into the video control when exactly one matches; selecting a video with no file picked mirrors the matched file into the file picker, so the detection shows up in the UI either way. The pickers' labels follow the value's origin (a *Detected* variant names the picker the app itself filled), the file's label carries the file's own recorded info (resolution, fps, frame count), and the console and the job log report both sides of the pair. If a previously detected file is no longer on disk, the lookup falls back to the selected video's own sidecar instead of failing. The companion video is used for preview only: if several match you pick one, and if none is found the branch goes ahead from the frames recorded in the file, so it works on a machine that no longer has the video. The *keep frames* field becomes the branch point, and the part you keep comes out at the quality of the original generation. You can keep branching from the same file. The frame count keeps the classic continuation's meaning in a branch job: it is the new segment generated after the cut, and the kept part is on top of it, so the final video is kept + this many frames; the job log prints the planned result (kept, new, total) as one line before the first window generates. If the length recorded in the file ever disagrees with the video, the file is saved flagged: *Latent Decode* still works, but a branch asks you to re-run the original job first. When the saved file cannot be used for the job, WanGP falls back to the classic continuation and the job log says so.

- **Not covered:** the older LTX (0.9 / 1.0) builds its video in two stages, and its decode may add noise, so a re-decode would only be an approximation of the original. Hunyuan is not covered, because a branch on its image-driven models would not keep the starting image exactly.

## 26th of August 2026: WanGP v12.643, Twice the Phase, Half the Trouble

- **H3 Two-Phase Generation with Latent Upscaling**: H3 now offers an LTX2-style two-phase workflow. Under *Advanced Mode / General*, select *Two Phases* from *Phases*. H3 first generates at half the target width and height, upscales the latent, and then performs a fixed three-step refinement at the target resolution using the LightX2V Turbo LoRA. This makes high-resolution generation much faster, even when the first phase uses many steps (for example, 20), but it does not reduce peak VRAM usage. To lower peak VRAM usage, select *Two Phases with Tiling*, which divides the high-resolution video into four tiles. Tiling may introduce visible seams; adjust the new *Phase 2 Noise Level Start* slider to balance fine detail against seamless tile blending.

- **H3 Face Refiner**: detect, identity-track, and refine up to five faces with H3 Ref2VA without changing the video resolution. Select *H3 Face Refiner* under *Advanced Mode / Post Processing*, use the selected Gallery video's *Post Processing* tab for Late Post Processing, or ask Deepy. Set *Faces to Refine* to `0` for automatic selection of up to five relevant faces.

- **H3 PDD LoRA Acceleration** (new WanGP 12.645): new 8-step acceleration variants are available for FL2VA and Ref2VA. This is not a mere *LoRA Accelerator*, each sequential model evaluation combines four learned denoising-interval outputs, so it why I had to create new model variants to handle this particular LoRA accelerator. Before you ask, yes when using PDD you are stuck to 8 steps and to the Euler Scheduler. 

- **Sense Nova U1.5**: *SenseNova-U1.5 8B MoT* is a unified image generator and editor designed for native-4K images, text-heavy layouts, and infographics. Select it from the Image model list and generate from text or one or more *Reference Images*. Its built-in *Infographic Prompt* enhancer can expand a short brief into a structured layout; native 4K can be still be generated with 8-9GB of RAM and will produce the best infographics. SenseNova comes with a 8 steps *LoRA Accelerator Profile*. Generation can be further accelerated at the cost of some VRAM by enabling the *KV Cache*.

- **New H3 Grouped Row Denoising Mode for Inpainting**: available for both FL2VA & Ref2VA can be used for *Inpainting* in a *Masked Area*

- **Deepy Prime for Everyone / Prompts Enhanced by Claude or ChatGPT**: use an OpenAI Codex or Anthropic Claude account, including supported free accounts, to power Deepy Prime and Prompt Enhancer without pausing a WanGP's generation. Open *Configuration / Prompt Enhancer / Deepy*, select *Codex* or *Claude Code* as the LLM engine, sign in, select *Deepy Prime*, and save. Choose models carefully and monitor the token usage shown in the chat footer; see the [Remote LLM guide](docs/REMOTE_LLMS.md) for details.

- **New Deepy Capabilities**: Deepy can now analyse the content of a video, compare frames and use the new H3 Head refiner. 

- **Even More Deepy Capabilities**:  (WanGP 12.643),  Deepy can create text files, create sub folders, zip multiple files and send notifications. This opens unlimited possibilities: for instance Deepy can store reference images in a dedicated folder so that these images can be used later, create zip of all generation made today, notify you when the work is done or let you know about current work progress, ...

- **Remote Notifications**: (WanGP 12.643) you can now define *Apprise Destinations* in the *Config / Notification* tab and receives Queue or Deepy related notifications in WhatsApp, Discord, Ntfyn .... The easiest way is to use the *ntfy.sh* service. For instance, just open in your Web Browser https://ntfy.sh/my_unique_sequence and then enter in Apprise Destination *ntfys://my_unique_sequence*

*WanGP 12.643*: Even More Deepy Capabilities, Remote Notifications\
*WanGP 12.644*: Masked Denoising Grouped Row\
*WanGP 12.645*: H3 PDD LoRA Acceleration
## 19th of August 2026: WanGP v12.61, Remember to be nice with Deepy

Deepy and the WanGP Prompt Enhancer have learned some impressive new tricks (enable them from the *Configuration / Deepy* menu):
- **Speculative Decoding** lets Deepy (and Prompt Enhancer) think up to twice as fast, so there is less waiting between ideas and actions.
- **INT8 KV Cache Quantization** lets it remember conversations roughly twice as long for the same VRAM budget. Install **GGUF Kernels 1.0.14** for the best performance; see the [installation guide](docs/INSTALLATION.md).
- **Smart context compaction** lets Deepy summarize the journey and carry on when its memory starts getting crowded, instead of suddenly forgetting the plot.
- The new **Qwen3.8 VL 27B** model brings sharper reasoning and richer enhanced prompts. It is available in GGUF Q2, Q3, and Q4 (recommended).
- With your permission, Deepy can browse your files and use them as part of a creative workflow.
- Deepy could already inspect media, trim clips, and process audio; it can now reach much further into WanGP's processing toolbox, including upscaling videos on demand.

But the star of this release is **Deepy Prime**. Open *Configuration / Deepy* and use the **Deepy** menu to invite either **Deepy Prime** or the original **Deepy Zero**:
- **Deepy Zero** is the quick, lightweight companion for straightforward requests and smaller language models.
- **Deepy Prime**, powered by **Qwen3.8 VL 27B**, is the ambitious one: give it a multi-step idea involving images, sound, and video, and it will plan the whole production.
- It understands WanGP's creative toolbox well enough to choose models, combine media, and plan long videos as connected sliding windows.
- When a plan hits a wall, Prime is resourceful: it can reconsider, try another route, and keep working toward the goal. Watching its thoughts unfold can be half the fun.
- It can connect to optional **MCP Servers** when you want outside help or interaction with another app; these connections live under **Deepy Prime Guidance** in the same settings page.
- Set **Deepy Filesystem Access** to read or read/write and you can ask for things like *"make one video for every prompt in this text file"* or the wonderfully vague *"use the files in this folder and make something nice out of them."* Access is scoped to WanGP outputs and folders you select.

There is one catch worthy of a tiny violin: for now, **Deepy Prime requires a GPU with at least 24 GB of VRAM**.

**Breaking News WanGP 12.61**: by selecting the newly added *Qwen3.8 VL 27B GGUF Q2*, a max *32000 KV Cache*, *Speculative Decoding Disabled*, and launching the Web Browser in *no GPU mode*, **you can now use Deepy Prime with 16GB of VRAM** !

Also in the news:
- **Sol Attention 0.6.2 for H3**: faster generations and better-looking results, according to its creators.
- **Ralston 2S Sampler for H3**: considered one of the best-looking H3 samplers, although it likes to take roughly twice as long.
- **IndexTTS 2.5**: a new version of this Text-to-Speech model for voices that delivers realistic emotions.

## 16th of August 2026: WanGP v12.53, Jess is Back !

**LTX 2.5. Jess is back, Dad is still in the garden, and this time the camera crew brought an upgrade.**

- **LTX 2.5 Visual Quality Upgrade:** expect sharper details, better visual consistency, and cleaner results than with LTX 2.3.

- **A new VAE decoder to play with:** the Original (much Faster) VAE remains the default, while the optional NAD Diffusion Decoder trades speed and VRAM for higher quality. NAD also works with LTX 2.3 and uses its Triton accelerator when supported.

- **Bring your LTX2 LoRAs:** LTX 2.5 finds LoRAs in the shared `loras/ltx2` folder and restores the compatible LTX 2.3 LoRA workflows, including pose/depth/canny control, inpainting, outpainting, Ingredients references, SDR-to-HDR conversion, and reference-voice conditioning.

- **LTX-2 Video Upsampler:** use LTX 2.3 or LTX 2.5 as a spatial video upsampler—directly after generation, as a late postprocessor, or as part of a Media Flow process. In WanGP 12.53, LTX 2.6 upsampler has been greatly improved an produces much better outputs.

Dev and Distilled are available in BF16 and INT8 ConvRot, while Distilled also gets an NVFP4 option for compatible NVIDIA hardware. Shared embedders are reused instead of being duplicated inside every checkpoint, saving both disk space and RAM.

**Bonus:**  
- **Wan2.2 Animate 2 Cache**: up to 50% faster thanks to VRAM or RAM cache

- **MinaMax Music 3**: Ace Step has met its match as Minimax Music 3 generates high quality songs. WanGP implementation is optimized with a *vllm engine* for x3 faster generation. It is recommended to have 16 GB of VRAM to use profile 3/3+ for much faster generation. This implementation contains several *Prompt Enhancers* that can either generate for you lyrics and / or the music description.

- **Upgraded GGUF Kernels 1.08**: lower VRAM, accurate native BF16, and CUDA-graph-safe Stream-K for GGUF models; check *docs/INSTALLATION.md* to upgrade.

*update 12.51*: Unlocked Control Video Processes, LTX-2 video upsampler\
*update 12.53*: MiniMax 3, LTX Upsampler 2.5 improved, upgraded GGUF Kernels
 
## 9th of August 2026: WanGP v12.45, Meet The One

**MiniMax H3 had all that potential waiting to be unleashed. We found the keys.**

- **Sliding Windows / Continue Video:** both *FL2VA* and *Ref2VA* can now build longer videos. WanGP carries the previous window's closing motion and matching audio into the next one. Most importantly, overlap is no longer limited to a single frame: using multiple overlap frames gives H3 real motion and sound context across the join, delivering much smoother transitions.

- **Start Image / End Image for Ref2VA:** launch a new shot from a chosen image, aim for a specific ending, or give a continued video the destination it deserves. Ref2VA preserves its selected reference memories across every Sliding Window, so later windows can keep following the same people, places, motion, and sound.

- **Frames Injection in FL2VA:** place several selected images at exact moments in an FL2VA video. Enter frame positions for precise timing or `L` for the end of a sliding-window segment—digital storyboarding without the sticky notes.

- **Audio Source:** FL2VA can create everything from text, follow an uploaded soundtrack, use a Control Video with its original audio, or keep the video unchanged while composing a new soundtrack. Full-length source audio is preserved in the final file; if it runs out early, H3 takes over instead of serving silence.


- **Spectrum v0.2.1 with offline replay:** H3 Spectrum now captures a clean accelerated trajectory and performs a transformer-free smoothing replay. Video and audio are reconstructed independently for better audio quality.

- **Control Video / Denoising Strength:** FL2VA can stay close to a Control Video or wander further from it as the strength increases. At `1.0` with *Whole Frame*, the visual control is unnecessary, so WanGP skips the extra work—your GPU may now take a very short coffee break.

- **Video Mask / Masking Strength:** choose *Whole Frame*, *Masked Area*, or *Non Masked Area* to decide where FL2VA may make changes and how firmly the remaining picture should follow the original.

> **Best practices for longer H3 videos**
> **For a multi-sequence video →** Direct it window by window: give each part its own prompt and duration, connect it smoothly with overlap, or use `[/new_shot]` for a hard cut. WanGP hands you the clapperboard instead of deciding where the story changes. Please check the Prompt Inline Help for the syntax.
> **For one very long continuous shot →** Use one Start Image followed by several End Images. Each End Image becomes the destination of a later Sliding Window, guiding the action from one visual milestone to the next. This works with both FL2VA and Ref2VA.

**Bonus:**  
- **Wan2.2 Animate 2**. *Animate* is back—and it wants to reclaim the crown *Scail 2* snatched away. Give it a character image and a driving video, and it will make that character follow the video's movements, expressions, and camera action: dance routines, performances, gestures, fashion clips, creature animation, and more.

*update 12.45*: spectrum upgraded, animate 2

## 6th of August 2026: WanGP v12.434, Cache Me If You Can

**MiniMax H3 shifts up a gear!**

H3 now has new accelerators and RAM shrinkers. Pick one or stack them—the exact gain depends on your video, hardware, and settings.

- **First Block Cache:** under *Advanced Mode / Steps Skipping*, H3 runs the first block and reuses the remaining blocks' previous result when little has changed—think TeaCache's cool cousin, driven by the first block's output. *Balanced (0.08)* is the upstream default; higher thresholds can skip more work and go faster, with a possible trade in motion or fine detail. The cache is tuned to add very little VRAM overhead. And yes, *Skip Steps starting moment in % of generation* means exactly what it says: it chooses when skipping may begin, not an acceleration factor.

- **Sol-Attn:** under *Advanced Mode / Misc. / Override Attention Mode*, sparse attention speeds up large visual sequences. It requires BF16, Triton 3.6+, and a compatible NVIDIA GPU (RTX 40/50-series, H100/H200, or B100/B200). Expected gains range from 10–20% on RTX 40-series to around 30% on RTX 50-series, with a possible small quality trade-off.

- **Mix and match:** *Spectrum* and *First Block Cache* are alternative step-skipping modes. Sol-Attn can technically run with either, but stacking approximations may reduce quality and should be checked with the same seed before relying on the combination.

- **Lower-RAM Video VAE:** select *FP8 Mixed Precision* under *Advanced Mode / Misc. / Video VAE* to reduce the RAM occupied by H3's Video VAE weights. Thanks to *Kijai* for creating this quantized VAE.

- **New W4A8 INT8 support:** H3 can now load asymmetric W4A8 checkpoints. Their 4-bit weights reduce checkpoint size and system RAM use, while 8-bit activations use optimized INT8 kernels on compatible NVIDIA GPUs (RTX 30-series or newer). Seriously short on RAM? Look for compatible community H3 W4A8/Q4 or NVFP4 checkpoints already available online. See *docs/FINETUNES.md* to add them to WanGP—and don't forget to share the finetune files you create on the Discord server!

- **Ref2VA tune-up:** this one is on me—I followed the original implementation and could end up feeding H3 a 4K reference image for a 480p video. Great for detail, less great for your stopwatch! You can now choose the reference-image pixel budget from 50% to 400%: lower is faster, 100% matches the output, and higher favors fidelity. The immediate payoff: **WanGP H3 Ref2VA is now twice as fast as before.**

- **New control-video choices:** use a *Reference Video* to reuse subjects, appearance, or motion without changing the output size; *Depth Control* to guide the scene's depth and layout; or *Generic Control* to feed the clip directly to H3. Control videos define the output canvas, while reference videos do not.

- **No LoRA Lost in Translation:** Pruned and non-pruned models can now read either LoRA format—the translation happens automatically as they load. Pruned (4 rank) LoRAs can also be used on original WanGP pruned checkpoints (rank 64) if you still use them.

- **LoRAs Accelerators**: kudos to *Lightx2v* and *larryvrh* for delivering the first *LoRAs accelerators* for Minimax H3. You will find them in WanGP as predefined profiles in the *Settings* dropdown box at the top. You may need to increase the number of steps to 8 if not happy with the quality and / or to play with the *LoRA multiplier* (default is 0.5 as 1.0 seems too strong)

*Update v12.431 + Update v12.432*: more LoRAs format supported, fixed NVFP4 Format, on the fly LoRA conversion of Non Pruned Loras\
*Update v12.433 + Update v12.434*: even more LoRAs format and quantization supported, LoRAs accelerators

## 5th of August 2026: WanGP v12.42, No Time for Taglines

**MiniMax H3**

MiniMax H3 is a top-notch open-weight contender to Seedance 2, combining cinematic video generation, convincing motion, strong prompt adherence, and a synchronized native stereo soundtrack in one model.

Given no *Steps Distilled Checkpoints* is available for the moment, 15-20 inference steps is a minimum.

But rejoice WanGP version is as usual Ultra Optimized: **5-6GB of VRAM only for 5s (124 frames) and 8-9GB of VRAM for 15s at 832x480**. 

- **MiniMax H3 FL2VA: create or continue a shot**: choose this version to generate synchronized video and stereo audio from text alone, start from an image or the last frame of a previous video, target an end image, or constrain both ends of the shot. It also supports longer generations with sliding windows.

- **MiniMax H3 Ref2VA: reuse people, scenes, motion, or voices**: choose this version when the new video should follow *Reference Images*, *Reference Videos*, or *Reference Audio*. References guide the newly generated result rather than becoming fixed frames, and remain available across sliding windows.

Both flavours offer the same controls in full 33B and lighter pruned 20B versions.

- **Spectrum step skipping**: Spectrum can make MiniMax H3 generation substantially faster, with a modest potential quality tradeoff. Its default offline replay retains every actual-step anchor in system RAM, reconstructs skipped steps from bracketing and spectral estimates, and keeps audio on local interpolation. Enable it under *Advanced Mode / Steps Skipping* by setting *Skip Steps Cache Type* to *Spectrum Feature Forecasting*.

- **Spatial upsampler improvements**: high-resolution MiniMax H3 generation can be slow, so a practical alternative is to generate at a lower resolution, such as 480p, and upscale the result afterward.
 - **FlashVSR optimizations**: FlashVSR has been further optimized to reduce system RAM usage.
 - **SeedVR2**: this high-quality image and video upsampler previously required too much VRAM for longer videos on many consumer GPUs. The WanGP integration reduces its VRAM requirement to roughly one-third of the original implementation. SeedVR2 is available under *Advanced Mode / Post Processing*, in *Late Post Processing*, and through the *Media Flow* plugin.

- **Memory priority**: MiniMax H3 defaults to *Lower VRAM*. If system RAM is the limiting factor and you have spare VRAM, select *Lower RAM* under *Advanced Mode / Misc. / Priority*.

- **Updated pruned checkpoints**: WanGP now uses ComfyUI-compatible pruned H3 checkpoints so upcoming H3 LoRAs can work with both applications. The previous WanGP-specific checkpoints were slightly less compressed and offered a small, usually imperceptible quality advantage. Existing installations will download the replacement files after upgrading. The old checkpoints are not removed automatically; keep them only if you plan to create finetunes from them, otherwise they can be deleted. The replacements require about 1 GB less disk space and system RAM.

- **Lower-RAM text encoders**: if system RAM is limited, open *Advanced Mode / Misc. / Text Encoder* and select one of the quantized Qwen3-VL variants.

*WanGP v12.41*: Added quantized text-encoder selection.\
*WanGP v12.42*: Added Spectrum, SeedVR2, the memory-priority selector, and updated pruned checkpoints.
## 25th of July 2026: Featured Plugins / Apps

WanGP's growing community has developed more than 20 plugins that expand what you can do. Here is a selection of seven newly available community plugins, all of which can be installed or updated directly from the WanGP Plugin Manager:

- **Finetune Manager** by *GKartist* — Browse community finetunes, load them into WanGP, and create, improve, or share your own.
- **Image Suite** by *saintorphan* — Create and edit images with text-to-image, image-to-image, layered canvases, masks, inpainting, cropping, resizing, and color adjustments.
- **Prompt Library** by *saintorphan* — Save your favorite prompts and generation settings, then reuse them with any supported model.
- **Prompt Manager** by *David Brum* — Search and organize generated images and videos, copy their settings, and manage reusable prompts in one place.
- **Queue Notifier** by *Javier-bat* — Get progress, completion, and failure alerts through services such as Discord, Telegram, WhatsApp, and Google Chat.
- **VRAM / RAM Adjuster** by *g3n3rativ3* — Tune how much graphics and system memory WanGP uses without manually editing configuration files.
- **Wildcards** by *GKartist* — Add reusable variables and random choices to prompts so you can quickly produce controlled variations.

**New Wan2GP Desktop Installer** — [Wan2GP Desktop](https://github.com/GKartist75/Wan2GP-Desktop-Tauri) by GKArtist lets you install, update, and launch WanGP from a single window. It handles Git, Python, CUDA, and PyTorch setup for you, making it the easiest way to get started on Windows.

### 29th of July 2026: WanGP v12.3456, Increasingly Greater

- **Krea 2 Identity Edit**: this Krea2 finetune adds Editing capabilities to Krea 2. You can edit an existing image or combine up to 2 *Reference Images* to produce a new one. WanGP implementation comes out of the box with *Inpainting* and *Outpainting*   

- **PiD 1.5**: The *PiD Spatial Upsampler* has been updated and should deliver better quality (v1 still there if you prefer it) and also now exists in *Qwen VAE* flavor (that is it can be plugged directly to Wan2.1 t2i, Qwen or Krea2 latent output for best quality)

- **LTX2 MSR 2.0**: this new version of this LTX2 finetune with Image Reference support preserves better Identity. WanGP v12.345 adds the setting *MSR Reference Video Length* that will let you control how the *Reference Images* are packed (please check model help for more info)

- **Joy Echo Surgical**: as a reminder the *Joy Echo* LTX2 variant lets you reuse characters identities between shots. This *Surgical* finetune claims to preserve better identity between shots and offers better audio quality.

- **ConvRot LoRA support**: Int8 ConvRot checkpoints can now use LoRAs without producing garbage output 

- **Text Encoder GGUF Support**: you should be able now to use in your Finetunes Text Encoder *GGUF* Checkpoints with LTX2, Krea2, Flux 1/2 and Wan 2.1/2.2

- **Onmnivoice Speed ajustment**: a new option gives you more control on the pace on spoken words (for instance to you fit more words in a shorter timespan)

- **More Krea2 LoRA Support**: more LoRAs formats are supported 

- **Shotplan**: some form *PromptRelay* for Wan 2.1 and Wan 2.2, you can divide a gen into different shots that reuses the same characters or objects

- **PrunaVAED**: this an alternative *VAE Video Decoder** for LTX2 that is 2.7 faster and requires half the VRAM during the *VAE decoding*. You will be able to select PrunaVAED in the *Misc* Tab at the bottom, in the new *Config* dropdown box

- **WanGP Configs**: if you just want to change the VAE or the text encoder you no longer need to create multiple finetune, you can now just create a single finetune with multiple configs in it. Please check *docs/FINETUNES.md*

*update 12.345*: Text Encode GGUF Support, Joy Echo Surgical, MSR2 new setting, Omnivoice Speed adjustment, new Krea2 LoRA formats\
*update 12.346*: Shotplan, PrunaVAED, WanGP Configs

### 1st of July 2026: WanGP v12.3, The VRAM Digger

- **Krea2 Lanpaint**: Krea2 can now do *inpainting* thanks to *Lanpaint*. To get the best results you will need to adjust the prompt and increase the number of Lanpaint steps.

- **Krea2 NAG**: WanGP exclusivity, *NAG* will allow you to define *Negative Prompts* with distilled models such as *Krea2 Turbo*

- **Gradio Optimizations**: thanks to numerous exclusive optimizations, Gradio UI should be faster (especially using the *Image Editor*) 

- **Chrome CPU Only Scripts**: you probably noticed that you Web Browser takes away VRAM just to display the UI. If you disable GPU Usage in Chrome for instance **you could save between 1GB of VRAM and 5GB of VRAM !!!**. The more VRAM capacity your GPU has the greater the gain (as Chrome tends to be greedier). I have added in the *Scripts* folder two scripts to disable GPU when using Chrome. WanGP has been optimized to still offer decent UI speed even if the web browser uses only the CPU. 

### 26th of June 2026: WanGP v12.278, Let's Experiment!

- **KREA-2** : new Image Generator model that claims to be the most aesthetic open-source image model available.

- **LTX-2.3 Multiple Subject Reference**: Here comes another way to add *Reference Images* when using LTX 2.3. This finetune combines Distilled 1.1 and a new LoRA from *LiconStudio*. Just provide 2 to 5 reference images; background first, then subjects and objects. Please note that the embedded lora is quite fond of character sheets with white background.

I added an experimental support for text to image, not sure it works as MSR doesnt seem to be made for that. 

- **LTX 2.3 Inpainting**: you will find this new *Inpainting* capability for LTX2 in the *Process List*. It is based on the set of *LoRAs* just released by the LTX Team. If you see glitches dont hesitate to expand the mask.

- **LTX 2.3 Ingredients**: part of the same new LoRAs collection the *Ingredients* process allows you to inject a character defined in a character sheet, preferably on a white background with black separator lines between individual pieces. Dont expect miracles with slidings windows or start frames.

- **Easy Frames Cap based on Control Video/Audio**: for supported models (*LTX2*, *Vace*) if you provide a control video or source audio you can ask WanGP either to stop when the control video / audio is done or continue until all the requested frames have been produced.

- **Ideograms v4 unlocked**: most hidden settings are now exposed (*mu*, *std*), you can change guidance half way, use a different scheduler. I added  resolutions used by Ideograms. Also please note that Ideograms runs two transformers in parallel *cond* and *uncond*. If you want to apply different *loras multipliers* to each transformer, use the new ":" separator, for instance with *1:1.2*, 1 will be applied to cond and 1.2 to uncond.

- **Ideograms v4 Turbo Time**: distilled version of Ideograms v4, from 4 steps to 8 steps and no guidance. 

- **Experimental Scail 2 Parallel Subwindows**: in order to reduce image degradations with long videos, I am experimenting a new concept: *Parallel Subwindows*, the idea is to work on a much larger Sliding Windows than usual (>200 frames) and to generate multiple sub windows (of 80 frames of so) in parallel. It is experimental, may end up a big fail and removed in next version, let me know...

- **Scail 2 Start Frame Fix**: you should no longer see a few bad frames at the beginning of the video in *Animate* mode. Many thanks to @pauldps that gave me part of the solution.

- **Scail 2 Experimental Multi References**: you can now provide different point of view of your character. This is an official feature but experimental.

- **PrismAudio**: this a *video to audio* processor, an alternative to *MMaudio*, quite good to add sound to an existing video. It requires a prompt. It is not made to generate spoken words.

- **More Plugins Types: Temporal Upsamplers / Audio Processors**: you can now add your own Temporal Upsampler (*Rife* alternative) or *Audio Processor* (*MMAudio* alternative). As a reminder the previous version allowed already to add a custom *Spatial Upsampler*.

- **API+, MCP+.**: I have improved the API capabilties (please check *docs/API.md*), and widened *MCP* support. Feel free to share feedback on Discord

- **Finetune Resolutions**: define custom resolutions directly in finetunes

*Update 12.25*: Ideograms v4 Turbo Time, MSR t2i, Scail2 parallel subwindows\
*Update 12.26*: LTX2 inpainting & ingredients, Easy cap, Scail2 fix\
*Update 12.27*: KREA2, Scail2 multiref


### 14th of June 2026: WanGP v12.22, Go with the Flow
- **Media Flow Plugin**: the *Full Video Process* is now named *Media Flow* because it can process *Images* as well as *Videos*. Even better, the new *Batch* mode can process any number of files: for instance, give *Media Flow* the path to the folder containing your collection of butterfly pictures and *all the corresponding images will be upsampled in one click*!

- **Scail 2**: the sequel to one of the best video *Character Animators*, and a very good alternative to *Wan 2.2 Animate*. You can either *Animate* up to 5 people by providing a *Start Image* and a *Control Video* that contains the movement, or *Replace* one person in an existing Control Video. Animate mode preserves identity well thanks to the new *Reference Image* input and, best of all, it supports *Sliding Windows* for non-stop dancing!

Please note that Scail 2 *Replace* and *Animate* modes require colored masks if more than two people are being replaced or animated. You can build them easily with *WanGP Magic Mask* (remember the magic wand icon). Also, for best results, I recommend using a *Reference Image* or a *Start Image* that is closely aligned to the first frame of the control video; you can use an *Image Model* generator for this.

Version *update 12.21* introduces RAM optimisations when using many *sliding windows* and added support for *Lora accelerator lightx2v 4 steps*

- **Int8 ConvRot Support**: model checkpoints saved in this quantized int8 format used by Comfy can now be loaded in WanGP.

- **LTX2 Image Generator (t2i)**: this one was always within grasp but required a little bit of packaging. Here we go we, just pick the *text to image* tab and use *LTX2* to use your favorite *Ic LoRAs* (outpainting, refiner, ...) on *images*. Best of all, the *LTX2 Image Processes* are available in the *Media Flow* Plugin.

- **Bernini 1.3B**: a much more gentle version (*lower VRAM requirements and faster*) for your GPU. Not as good as the 14GB version, but still produces some nice outputs.

- **Chain of Zoom Upsampler**: new upsampler that can magnify up to x16, quite good with hair and skin. However it expects low quality image so it may reinvent existing details. WanGP optimized: low VRAM and up to x4 times faster

- **Upsampler & Model Plugins**: PlugIn developers can now create plugins that add new *Spatial Upsampler* or new *Models*

As sample plugins, enjoy:
   - **Stable Diffusion 1.4**: the father of all image generators !
   - **Pixel Upsampler**: upsample by duplicating the same pixel for a grandiose Pixel Art effect !

*update 12.21*: Scail 2 RAM optim + lightx2v support, added LTX2 t2v & Bernini 1.3B\
*update 12.22*: Chain of Zoom Upsampler,  Upsampler & Model Plugins

### 7th of June 2026: WanGP v12.13, Prompt Control
- **Ideogram 4 Prompt Helper**: the great thing about Ideogram 4 is that you can position every object or text exactly where you expect it in your output image. Ideogram 4 now has a *Visual Helper* to create and edit its JSON prompt format. Click the *Magic Wand* next to the prompt to draw or resize text/object boxes, tune the main prompt fields, and apply the final JSON back to the prompt. *Magic Prompt* can still create the first draft for you.

- **JoyAI-Echo**: this new LTX-2.3 model is the closest thing to *SeeDance 2* that you may find in the open source world. It is an audio-video model for connected multi-window stories. JoyAI-Echo keeps compact memories between windows so later shots can reuse characters, voices, objects, and places. WanGP implementation of *JoyAI-Echo* goes well beyond the original implementation:
   - With the new *Sliding Window commands* (see below), you can extend existing *Sliding Windows*, *Create New Shots*, or *Continue a Video*.
   - The new memory command system (`[/store_mem]`, `[/load_mem]`, and `[/drop_mem]`) lets you pick which sliding windows can be reused for future memory and which ones should no longer be used. Please check the JoyAI-Echo *Prompt help* for the full syntax.
   - Use a *Control Video* to target audio/video segments in the *Joy Memory Positions* field and seed the first memories with characters and background. 
For instance *Joy Memory Positions* could be `man=4s,woman=12s`, if a man is speaking at around 4s and a woman at around 12s. The two memories can be used in later windows with the command [/load_mem=man] or  [/load_mem=woman] 

- **Sliding Window Commands**: thanks to new inline prompt commands (for instance `[/duration=...]`, `[/overlap=...]`, and `[/new_shot]`), you can now define a different duration, number of frames, or transition style on a per-window basis. You can also change the LoRAs multipliers of the current window with `[/loras_mult=1;0]`. See `docs/PROMPTS.md` for the full syntax and examples.

### 4th of June 2026: WanGP v12.00, The Journey Continues
- **PiD**: a new high quality x4 spatial upsampler for images by Nvidia. It is supposed to work with only Flux/Flux2 compatible models since it needs to plug directly to the VAE Decoder. However thanks to a simple trick it is available everywhere. Some automated Tiling may be triggered if you ask for very high out res. WanGP version is as usual ultra optimized and should require little VRAM even when tiling is not used.

- **Ideograms v4**: this image generator claims to be the best open source image generator. It consumes a special *Json Prompt Format* that WanGP *Prompt Enhancer* can produce for you. There is a snag though: occasionnaly, even a harmless prompt may trigger a *Safety Filter*. No way to get around this as it is hardcoded in the model weights.

- **Stable Audio 3**:  WanGP *Text To Speech* (TTS) collection of models is now completed with a model that can generate sounds, background music or special effects 

- **Bernini 14B**: the video model derived from Wan 2.2 is really incredible. You can ask it to modify the content of an existing video or to generate a new video with any number of *References Images*. and *it just works*. There is a price to pay though: to generate 81 frames, you will need 12 GB of VRAM for *v2v* / 16GB for *v2v + ref frames*. v2v  works quite well with Lora Accelerators such as *lightning 4 steps* . But as soon as you include reference frames, you will have to go for at least 15 steps with guidance and no lora accelerator. You are not allowed to complain, this model is advertised to work on a H100 and thanks to WanGP magic you can run it at home.

- **MCP Server & Agent Skills**: WanGP includes now a *MCP server* to make life much easier to your AI Agents. WanGP exposes also new discovery functions that can be queried by to agent to get the list of all generative models and features that are available.

### 1st of June 2026: WanGP v11.90, Everything will be fine...
**Finetune Creator / Editor**
*Create* a new *Finetune* (use an existing model with your own checkpoints), *Edit* or *Import* an existing Finetune in only one click directly from the *WanGP UI*. You can then share easily a finetune with other users by clicking the *Export* button.

Look for the new **+** in the *WanGP Tool Bar*.

The finetune creator allows you not only to customize an existing models with *Custom URLs* or *Local Paths* for both the main *Transformer files* & *Text encoders* but also to define *User help* and set *Custom System Templates* to be used with the finetune *Prompt Enhancer*.

Please check *docs/FINETUNE.md* doc for info about finetunes.

### 29th of May 2026: WanGP v11.88, Humans Accelerators
- **Create Hierarchies of Loras / Change Order of Loras**

- **WanGP Toolbar** with keyboard shortcuts:
    - **Search**: switch quickly to another model by just entering a few letters of its name
    - **Refresh Model List**: no longer needed to restart the app to add or modify a finetune
    - **Unload All**: free most of the RAM/VRAM used by WanGP

- **MOV/MKV Container Support**: beside *mp4* files you can now store you video gens in *mov* and *mkv* containers

- **ProRes422 & DNxHR HQ Video Codecs**: these professional video codecs have some fans out there

- **LTX-2 Guide**: click the "i" to the right of the model description to get tips / explanations on how to use LTX2 models

- **LTX2 Smearing Fix**: the smearing / ghosting is now mostly gone

- **Omnivoice Fix**: you will enjoy this fix unless you liked the gibberish generator of the previous version


### 21st of May 2026: WanGP v11.77, I can hear Voices
It has never been easier to do voice cloning directly in video models:

- **Voice Cloning with any Video Model**: you generated a great *LTX2/Ovi/Multitalk/...* and are sad the model didnt support natively *Voice Cloning*? Just use the new *SeedVC Audio Postprocessing* to replace up to two voices of your choice, it works magically with any video model ! You will find this feature in the *Audio* advanced tab or as *Late Posprocessing for Audio or Video*.  WanGP exclusive *Two Voices* feature will detect who is talking and will make seamlessly the voices replacements at the right audio locations.

*New WanGP v11.75*: Voice cloning preserves background noise / music & supports singing. You can also enable *SeedVC v2* in the *Config / Extensions* tab for a higher quality voice cloning (alas no singing support with v2).

- **DramaBox**: like *ScenemeAI* that *DramaBox* uses LTX2.3 world knowledge to generate lively audio outputs. DramaBox is even more expressive (but also slower) than ScenemeAI. Of course as usual you get an exclusive Dialogue mode available out of the box.

- **LTX2.3 Id Lora Distilled**: Nice surprise ! it seems *Id Lora* worked from day 1 with *LTX2.3 Distilled*. It is now unlocked, you can now generate your own LTX2 video with voice cloning.

- **LTX2.3 EditAnything Reference**: you can at last inject one reference image in a LTX2 Video. You will need to use the dedicated finetunes *dev* and *distilled* finetunes I have prepared. Please note this feature is experimental.

- **LTX2 OmniNFT Lora Preset for better audio/video sync**: I have added this *LTX2 OmniNFT Lora* in a *Preset* so that it can be applied quickly. According to the authors of this Lora Audio/Video sync should be greatly improved.

- **LTX2 Dev reborn in Dev-Distilled**: WanGP LTX2 Dev implementation was based on LTX2 official implementation. I hadn't noticed that ComfyUI version of Dev was now completely different as it was mixing the *Distilled Lora* with Dev in both phases ( not just in phase 2). This makes dev faster and reduces the color saturation specific to Dev. So I have added a few *Dev Distilled Accelerator Profiles* you can pick from the *Settings List*. And now since Dev & Distilled are closer than ever, I have unlocked all the *Control Video* processes for Dev.

- **LTX2 Prompt Relay**: you can now target specific time range for a part of the prompt, for instance *[25%:50%]the man says "hello". Check the new *Prompt Online Help* marked with "i" for more info. 

- **LongCat 1.5 Avatar**: with this new *Talking Head* model you are going to become at last a fan of *LongCat*. It is fast (8 steps distilled) and delivers high quality potentially unlimited gens using *Sliding Windows*.

- **Settings can now store Audio/Video/Images**: you can ask WanGP to store (in option) all the media you use frequently in a WanGP *Settings file*. This is very convenient for instance if you always use the same *Voice sample* or *Reference Images*. Even better, you can use these settings with *Deepy* of the *Full Video Process* plugin

- **Extensions Enabled by Default**: most extensions (upsampling, mmaudio, prompt enhancer, ...) are now enabled by Default so that they are easier to be found. Don't worry their corresponding checkpoints will be downloaded only if you actually use these extensions

- **FlashVSR Spatial Upsampling for Images**: this excellent spatial upsampler has been optimized for images and is now can be used as a *Post Processing* option or on existing images (thanks to the new *Late Post Processing* added on Images!)

- **FlashVSR Two Pass**:  banding artifacts may appear when FlashVSR is used at very high res. The Two Pass mode which is twice as slow may reduce the banding. 

- **HiDreamO1**: new 2604 finetune that should reduce the annoying blocking effect of this model. I have also regenerated all the quanto int8 files (they are now 20% larger, price to pay for quality) to reduce even further the blocking. Keep in mind that this model likes res >= 1080p

Also various fixes (Omnivoice, IndexTTS, Chatterbox, ...)

*Update 11.75*: Voice cloning with background and voice supports, FlashVSR for Images, Dev Distilled\
*Update 11.77*: LTX2 Prompt Relay, LongCat Avatar

### 12th of May 2026: WanGP v11.66, Can you keep up?

- **HiDreamO1**: New Image Image model with editing capabilities is quite good to preserve identify and write text. WanGP version requires very Low VRAM and supports out of the Box *Control Image* & *Preview*.

- **Omnivoice**: This *Text To Speech model* (TTS) is fast and supports 100 languages with voice cloning. WanGP offers as a bonus an experimental dialogue mode (not the best one since it is hard to predict when Omnivoice has finished generating)

- **ScenemeAI**: A LTX2.3 derived *TTS* that leverages *LTX-2* world knowledge : it can produce lifelike audio generations since you can drive the audio generation by describing what a speaker is doing / saying. I have implemented on top a dialogue mode between any number of speakers (first two speakers support voice cloning) with very smooth transitions between speakers especially when generating English. You will find ScenemeAI among the *TTS models* but be aware it will use by default a *Video Memory Profile* since it uses LTX2 engine behind the scene. Don't hesitate to use WanGP *Prompt Enhancer* to generate lively dialogues.

- **MPS / Apple Early Support**: Mac users are about to discover the world of WanGP albeit for start it wont be fast nor very optimized and not all models will be supported. Many thanks to *huangyebiaoke* (for the port), *cn0ss* & *SquishedSquirrel* (for the testing). Don't hesitate to report in the new *MPS* Discord channel your feedback if you are a mac user.

### 9th of May 2026: WanGP v11.61, The Last Mile

With a slight (half year) delay WanGP supports now officially *FlashVSR* a very high quality *Spatial Upsampler* which can upsample up to 4x you videos. As FlashVSR has been almost entirely rewritten for WanGP, it can be branded as the *Ultimate Upsampler for the GPU Poor*, check these figures:
- x2 Spatial Upsampling will need to work only 6GB of VRAM
- x4 Spatial Upsampling will require only 10GB of VRAM (see the 5k example below)

The VRAM requirements above are independent of the Video Length (still the longer the video the more RAM)

You first need to install *Triton* and optionally *SpargeAttention* for best quality (please check the INSTALLATION.md for download links) and enable *FlashVSR* in the *Configuration > Extensions* Tab.

FlashVSR is available in the following contexts:
- a Postprocessing option in *Advanced Tab > Postprocessing*
- a *Late Postprocessing* that can be applied on already generated videos
- in Model *WanGP System Video Postprocessing* of the *MediaFlow* Plugin you can Upsample a few hours long Video !

Please note as FlashVSR is now natively supported by WanGP and highly optimized, you may no longer need the *FlashVSR Plugin* developed by @h4k4z3. In any case many thanks to @h4k4z3 for developing this plugin which was very useful.

### 2nd of May 2026: WanGP v11.52, a Kind of Magic

- **Vista 4D**: Vista4D allows a *Video Reshooting* of a *Dynamic scene* from novel camera trajectories and viewpoints. In other words this Wan 2.1 model will let you relive from a different (moving) perspective a scene with moving people or objects. The sequences are quite short (usually 49 frames, max around 97 frames) but it is a lot of fun as for once this really works. 

In real life, there is no chance you should have been able to run this model (it requires x3 the amount of VRAM than what is usally required for equivalent output res and the preprocessing needs 24 GB of VRAM to build a 4D map). But once again thanks to WanGP magic VRAM requirements have been reduced to 10 GB of VRAM or less.

It is highly recommended to apply the *Lightx2v 4 steps* lora profile. Also for best efficiency, you must list all the dynamic objects / people in the *Dynamic object keywords* input.

- **Magic Mask**: generating a *Video Mask* or *Image Mask* has never been easier and faster. No need to get into the *Mask Generator* tab, just click the *Magic Wand* next to *Mask field* and enter a few keywords like *blue car* or *lady to the right* and a high quality mask powered by *SAM3* will be generated automatically. You will appreciate the very good *Temporal Consistency* brought by SAM3.

- **Mask Generator with SAM3 support**: if you still need to generate complex masks you can combine the good old point and click masks with the SAM3 / Magic Mask masks. You need to enable this feature in the *Config / Extensions* tab.

- **LTX-2 Video to Audio**: it was more or less already possible but this new Control Video Process will be much faster and the output video will be unaltered

*update 11.51*: various fixes\
*update 11.52*: LTX-Video to Audio, fixed bugs in audio continuation with sliding windows 

### 25th of April 2026: WanGP v11.41, LTX-2 Mega Mix Part 2
More nice goodies for **LTX-2**:
- **HDR Control Video support**: you can now provide an HDR Control Video it will be automatically converted to SDR if model doesnt support HDR

- **LTX 2.3 SDR to HDR**: thanks to a new HDR Ic lora, you can now convert SDR Videos to HDR using LTX 2.3. This feature is available as a new *Control Video process* and also in the *MediaFlow* plugin. Please note that the embedded Gradio Gallery video player converts automatically any HDR content to SDR, so if you want to enjoy the full HDR content you will need an external media player (for instance *MPC-BE*)

- **LTX 2.3 Control Video Injection in Phase 2**: up to now even if you picked 2 phases, the *Control Video* was only injected in Phase 1 (Phase 2 was only used for upsampling). Now if you have chosen for at least one Ic Lora, a non null mutiplier for phase 2, the control video will be injected also for phase 2. This will increase output quality with 2 phases but will require more VRAM for phase 2.

- **MediaFlow Custom Settings**: you can now reuse your own presaved settings in the plugin. As you will  link the plugin to your settings any change to the saved settings  will be immediatly available in the plugin. If you find some great combination of loras / model / settings to be used with this Plugin please share them on the discord server so that I can add them in the official list.

*update 11.41*: added MediaFlow Custom Settings

### 21st of April 2026: WanGP v11.35, LTX-2 Mega Mix
Lots of nice goodies for **LTX-2**:

- **LTX-2.3 Distilled 1.1**: new version of the *Distilled model* released by *LTX team*, it should offer better audio and visuals. You will find also a Dev 1.1 version which uses Distilled 1.1 for Phase 2.

- **VBVR Lora Preset**: This LoRA enhances the base LTX-2 for Enhanced Complex Prompt Understanding, Improved Motion Dynamics & Temporal Consistency. You can select it in the *Settings list* at the top.

- **Phase 1/2 Choice**: you can now either you go for a good old *2 Phases Gen* (1st Phase Low Res, 2nd shorter Phase High res) or go straight to a single High Res Phase (needs more VRAM and slower, but potentially higher quality). Please note that Outpainting mode and Pose/Edge/Depth extractors are always using 1 phase.

- **Improved Sliding Window**: transition between windows should be less noticable, *Sliding Windows overlapped Frames* carry now also the audio of the overlapped frames, so the higher the number of overlapped frames the higher the chance that the sound / voice used in the previous window will be used in the new one.

- **Video Length not Limited by Audio**: if you provide an Audio input, WanGP will no longer stops when the audio is consumed. It will continue the Video/Audio Gen based on the content of your Text prompt, and guess what ? it may reuse the same voice/sound used up to now !  This is an option, you need to check the checkbox *Video Length not Limited by Audio*.

- **Silent Movie Mode**: if for some reason you want video with not only no sound but that takes into account that there is no sound (you dont want people to open their mouth for instance), just now leave the *Control Audio* empty

~~ - LTX2/2.3 Loras Split: as LTX2.0 Loras work badly with LTX2-3 and were getting on the way, now each version of LTX2 has its own lora folder. Loras will be moved automatically at startup using a lora migration script. I invit you to verify that the loras landed in the right folder.~~ 

- **System Loras Multipliers Overrides**: WanGP adds automatically and transparently loras (that is they are loaded although they are not visible) if needed by a feature (distilled lora, id lora, outpaint lora, union control lora). You can now override the default multipliers used by WanGP by selecting the target lora in the *Activated Loras* input and by specifiying the corresponding *Loras Multipliers*.

- **Transfer Human Motion With Pose Alignment**: you are trying to transfer a human motion from a control video, but you use a start image with a person who has a different body shape (larger, taller, ...) and stands in a different location in the frame. This is not going to work well as you start image wil end up distorted. This is a past issue, as now the control video pose can be aligned with the start image if you pick Transfer *Human Motion With Pose Alignment*. This feature is also supported by *Wan Vace*, start image  must be the *Background ref image*.

- **Injected Frames & Sliding Windows**: injected frames were not properly injected starting from window no 2. This is now supported.

- **MediaFlow Plugin**: this *bundled PlugIn* which needs to be enabled first in *the PlugIn tab*, right now supports only *Outpainting*. It relies on *LTX2 Lora outpainting*. It is more or less a *Super Sliding Windows* mode but without the *RAM restrictions* and no risk to explode the *Video Gallery* with huge files. If you are patient enough you can change the Aspect Ratio of a few hours movie (check out below the 1 min sample). Behold how *Sliding Windows transitions* are almost invisible !

- **NEW Processes for Full Video Plugin**: *Refocus* (remove blur), *Ungrade* (remove stylized color grading) and *Uncompress* (remove compression artifacts) have been added. Many thanks to *Oumoumad Mohamed* who created the Ic Loras (including the *Outpainting* lora ) that power these processes. If you have found some Ic Loras that are useful and dont cause glitches with Sliding Windows, let me know and I will add them.

- **WanGP API Video Gen**: *Plugin Developers* can now *Queue a Gen* directly from a plugin. This opens the possibility of plugins that place various gen orders and then combine the results (hint: we could have our very own version of *LTX-Destop* inside WanGP).

- **New One Click Install / Update Scripts**: We have to thank **Tophness / @steve_Jabz** for that one. *Huge Kudos to him!* The scripts will not only install WanGP but also all the *Kernels* (among *Triton, Sage, Flash, GGuf, Lightx2v, Nunchaku*) supported by your GPU. Please have a look at the instructions further down. Dont't hesitate to share feedback or report any issue.

*update 11.31*: fixed phase 1 forced incorrectly in some cases\
*update 11.32*: bugs fixes, MediaFlow now supports Distilled 1.1 & accepts video without audio\
*update 11.33*: Separated LTX2 & LTX2.3 loras in different folders, added easy loras multipliers override\
*update 11.34*: Reverted split as not popular\
*update 11.35*: added Aligned Pose Transfer, Injected Frames & Sliding Windows support, new processes for MediaFlow Plugin

### 11th of April 2026: WanGP v11.26, Now I Can See

- **LTX-2 Ic Lora Rebooted**: *Ic Loras* behave like *Control Nets* and can do *Video to Video* by applying an effect specific to the Ic Lora for instance *Pose Extraction*, *Upsampling*, *Transfer Camera Movement*, ...  More and More Ic Loras are available nowadays. Until now WanGP Ic Lora implementation was based on the official LTX-2 github implementation (which a 2 phases process where the Ic Lora is only applied during the first low res phase). However I have just discovered that all the Ic Loras around expect in fact the ComfyUI implementation which is one phase only process at full res. 

So from then on WanGP Ic Lora will work this way too. The downside is that a single Full Res pass is much more GPU intensive. But all is good in WanGP world, as the LTX2 VRAM optimisations will allow you to use Ic Loras at resolutions impossible anywhere else.

As a bonus I have tuned *Sliding Windows* for Ic Loras, and if you set *Overlap Size* to a single frame, transitions between windows when using Ic Lora will be almost invisible. 

- **Outpaint Ic Lora**: this new impressive Ic Lora will be loaded automatically if you select the *Control Video for Ic Lora* option and enable *Outpainting*. If you use Sliding Windows with Outpainting you will be able to outpaint a full movie (assuming you have enough RAM).

- **New Outpainting Auto Change Aspect Ratio**: As a reminder WanGP let you define manually where an Outpainting should happen. Alternatively you can now ask WanGP to use outpainting to change the *Width/ Height Aspect ratio* of the Control Video. For instance you can turn any 16/9 video into a 4/3 video by generating new details instead of adding black bars. The *Top/Bottom/Left/Right Sliders* in this new mode will be used to define which area should be expanded in priority to meet the requested aspect ratio.. 

*update 11.26*: fixed outpainting ignored with if Manual Expansion was selected

### 8th of April 2026: WanGP v11.22, Self Destructing Model

- **Magi Human**: this is a newly *Talking Head* model that accepts either a *custom soundrack* or can generate the *audio speech* that comes with the video. 
   - *The bad news* :it is VRAM hungry (targets RTX 5090+) and very res picky, that is the ouput res must be either 256p or 1080p (using a 2 stage pipeline with upsampling). There is also a 540p version (using also an upsampler) but it is not included as I found it unpractical (ghosting guaranteed if your output is not exactly the right height/width ratio), 
   - *The good news* : now that it is WanGP optimized, 101 frames at 1080p requires "only" 16 GB of VRAM. If you dont have that much VRAM I recommend to still go for 1080p but set a 45 frames *Sliding Window* (not too low to avoid artifacts) as *Sliding Windows* sometime works well with this model.  

**I have spent a lot of time optimizing Magi Human, but I am not yet sure it is worth keeping it given all the constraints to run this model. So this is where I need YOU. Please share your experience using Magi Human on the Discord server and you shall decide its fate. Should we keep it or send it to the model graveyard ?**

- **Ace 1.5 Turbo XL**: the best open source song generator has now a big brother *XL* that delivers better audio quality and sticks closer to the requested lyrics. 

- **LTX 2 Id Lora**: due to a huge popular demand I have added this one (it is a new *Generate Video* option). You can provide a voice audio sample, a start image and text script and it will turn LTX 2/2.3 into talking heads. Cost is high to get this feature as **Id Lora works only with LTX2/2.3 DEV**. By chance it seems it can produce decent results in only 10 inference steps. To get the best results it is recommended to use prefix tags [VISUAL], [SPEECH] & [SOUND]. Alternatively you can use WanGP *Prompt Enhancer* that has been to tuned to generate a prompt following this syntax. 

- **LTX 2 NAG**: you can now inject a *Negative Prompt* even if you use the Distilled Model thanks to *NAG* support for LTX 2

- **LTX 2 DEV HQ Mode**: this High Quality mode should produce better output at higher res. You can turn it on using the new *HQ (res2s)* Sampler and set 15 steps and guidance rescaler to 0.45. It is compatible with *Id Loras*. Note that a HQ steps is twice as slow as a vanilla Dev step, so it is going to be as slow as Dev if not slower.

- **LTX2 DEV Presets**: Vanilla Dev mode & HQ Mode have lots of tunable settings. To make your life easier I have added selectionable presets in the *Settings Drop Downbox*

- **More Deepy** : 
   - *UI Improvements*: you can *queue* requests by inserting empty lines between two requests, get the last turn by clicking the *Down Arrow*
   - *More Responsive*: Deepy should execute much more quickly consecutive actions
   - *More Reliable*: fast full context compaction (when deepy ran out of tokens), Deepy will remember what you stopped / aborted
   - *More Capabilities*: you can ask Deepy to specifiy a *guidance*, *denoising strength*, ... value (the value defined in the *tool template* will be overridden)

As a reminder beside writting huge essays about how great you are, Deepy can generate Video, Image & Audio, extract / transcribe / trim / resize (when applicable) video or audio clip, inspect the content of an image or a video frame, generate black frames, ... Deepy used Tool templates but you can specify for one task the loras, number of frames, dimensions, ... There is also a CLI version of Deepy quite useful for remote use. Please check the fulldoc *docs/DEEPY.md*. 

- **Multi Multilines Prompts**: check new options in *"How to Process each Line of the Text Prompt"*, you can now have multiple multi lines prompts. They just need to be separated by an empty line.
   
 *update 11.21*: added Ace Step 1.5 Turbo XL\
 *update 11.22*: added LTX2 NAG

### March 30th 2026: WanGP v11.13, The Machine Within The Machine

Meet **Deepy** your friendly *WanGP Agent*.

It works *offline* with as little of *8 GB of VRAM* and won't *divulge your secrets*. It is *100% free* (no need for a ChatGPT/Claude subscription).

You can ask Deepy to perform for you tedious tasks such as: 
```text
generate a black frame, crop a  video, extract a specific frame from a video, trim an audio, ...
```

Deepy can also perform full workflows:
```text
1) Generate an image of a robot disco dancing on top of a horse in a nightclub.
2) Now edit the image so the setting stays the same, but the robot has gotten off the horse and the horse is standing next to the robot.
3) Verify that the edited image matches the description; if it does not, generate another one.
4) Generate a transition between the two images.
```
or

```text
Create a high quality image portrait that you think represents you best in your favorite setting. Then create an audio sample in which you will introduce the users to your capabilities. When done generate a video based on these two files.
```

Deepy can also transcribe the audio content of a video (*new to WanGP 11.11*)
```text
extract the video from the moment it says "Deepy changed my life"
```

*Deepy* reuses the *Qwen3VL Abliterated* checkpoints and it is highly recommended to install the *GGUF kernels* (check docs/INSTALLATION.md) for low VRAM / fast inference. **now available with Linux!**

Please install also *flash attention 2* and *triton* to enable *vllm* and get x2/x3 speed gain and lower VRAM usage.

You can customize Deepy to use the settings of your choice when generating a video, image, ... (please check docs/DEEPY.Md). 

*Go the Config > Prompt Enhancer / Deep tab to enable Deepy (you must first choose a Qwen3.5VL Prompt Enhancer)*

**Important**: in order to save Deepy from learning all the specificities of each model to generate image, videos or audio, Deepy uses *Predefined Settings Templates* for its six main tools (*Generate Video*, *Generate Image*, ...). You can change the templates used in a session or even add your own settings. Just have a look at the doc.

With WanGP 11.11 you can *ask Deepy to generate a Video or an Image in specific dimensions and also a number of frames for a video*. You can also specify an optional *number of inference of steps* or *loras* to use with *multipliers*. If you don't mention any of these to Deepy, Deepy Default settings or the current Templated Settings will be used instead.

WanGP 11 addresses a long standing Gradio issue: *Queues keep being processed even if your Web Browser is in the background*. Beware this feature may drain more battery, so you can disable it in the *Config / General tab*.

You have maybe also noticed the new option *Keep Intermediate Sliding Windows* in the *Config / Outputs* tab that allows you to discard intermediate *Sliding Windows*

### March 17th 2026: WanGP v10.9875, Prompt Enhancer has just Been Abliterated

- **Qwen3.5 VL Abliterated Prompt Enhancer**: new choice of Prompt Enhancer
   * Based on widely acclaimed *Qwen3.5 model* that has just been released
   * *Uncensored* thanks to the *Abliterating* process that nullifies any *LLM will* to decline any of your request
   * 4 choices of models: depending on how much VRAM you have *4B & 9B models*, and *GGUF Q4* or *Int8*
   * *vllm accelerated* x5 faster, if Flash Attention 2 & Triton are installed (please check docs/INSTALLATION.md) 
   * *Think Mode*: for complex prompt queries
   
   Also you can now expand or override a *System Prompt prompt Enhancer* with add @ or @@ (check new doc *PROMPTS.md*)

- **GGUF CUDA Kernels**: 15% speed gain when using GGUF on Diffusion Video Models & x3 speed with GGUF LLM (*Qwen 3.5 VL GGUF* for instance). GGUF Kernels are for the moment only available for Windows (please check docs/INSTALLATION.md).

- **LTX2.3 Improvements**
   * *End Frame without Start Frame*: you know how your story ends but want to see how it started, just give an End Frame (no start Frame) 
   * New GGUF Checkpoints
   * VAE Decoding hopefully should expose less banding
   * *Multiple Frames Injections*: inject at different positions the reference frames of your choice (works for LTX-2.0 too)
   * *Image Strength* can be applied now too *End Frames* & *Injected Frames*
   * New Spatial Upsampler 1.1, hotfix supposed to improved quality with long video
   * *More VRAM optimisations*: Oops I dit it again ! not that is was needed since WanGP is by far the LTX2 implementation that needs the least VRAM. But now we can in theory (output wont look nice due to LTX2 limitations) generate 15s at full 4K with 24GB of VRAM. So it means that with lower config you should be able to generate longer videos at 720p/1080p. As a bonus you get a 8% speedup.
   * *NVFP4 Dev checkpoint*: if you have a RTX 50xx, help yourself 

- **WanGP API**: rejoice developers (or agents) among you ! WanGP offers now an internal API that allows you to use WanGP as a backend for your apps. It is subject to compliance to the terms & conditions of WanGP license and more specifically to inform the users of your app that WanGP is working behind the scene.

- **LTX Desktop WanGP**: as a sample app (made just for fun) that uses WanGP API, you may try LTX Desktop. This app offers Video / Audio nice editing capabilities but will require 32+ VRAM to run. As now it uses WanGP as its core engine, VRAM requirements are much smaller. It will use LTX 2.3 for Video Gen & Z Image turbo fo Image gen. You can reuse (in theory) your current WanGP install with *LTX Destop WanGP*. https://github.com/deepbeepmeep/LTX-Desktop-WanGP

- **New Audio Ouput formats in mp4**: audio stored in video file can now be of higher quality (*AAC192 - AAC320*) or *ALAC* (lossless). Please note that you wont be able listen to ALAC audio track directly in the webapp.

Also note as people preferred mataynone v1 over v2 I have added an option to select matanyone version in the Config / Extension tab

*update 10.9871*: Improved Qwen3.5 GGUF Prompt Enhancer Output Quality & added Think mode\
*update 10.9872*: Added LTX 2.0/2.3 frames injection\
*update 10.9873*: Fixed low fidelity LTX2 injected frames + added Image Strength slider for end & injected frames\
*update 10.9874*: Replaced LTX-2.3 spatial upsampler by hotfix v1.1\
*update 10.9875*: LTX-2 more VRAM optimisations + NVFP4 checkpoint

### March 7th 2026: WanGP v10.981, Expecting an Update ? 

- **LTX-2 2.3**: 0 day delivery of LTX 2 latest version with better *audio*, *image 2 video* and *greater details*. This model is bigger (22B versus 19B), but with WanGP VRAM usage will be still ridiculously low. Try it at 720p or 1080p, this is where it will shine the most !

*Control Video Support* (*Ic lora Union Control*) will let you transfer *Human Motion*, *Edges*, ... in your new video.

For expert users, *Dev* finetune offers extra new configurable settings (*modality guidance*, *audio guidance*, *STG pertubation/skip self attention *, *guidance rescaling*). LTX team suggests: Cfg=3, Audio cfg=7, Modality Cfg=3, Rescale=0.7, STG Perturbation Skip Attention on all steps.

I recommend to stick to the *Distilled* finetune for higher resolutions (see sample video below) as it seems to have been distilled from a higher quality model (pro model?).

- **Kiwi Edit**: a great model that lets you edit video and / or inject objects in a video. It exists in 3 flavours depending on what you want to do

- **SVI PRO2 End Frames**: this should allow in theory to generate very long shots by splitting one shot into sub shots (sliding windows) by inserting key frames (the *End Frames*). This is an alternative to the *Infinitalk* references frames method (see my old release notes). I am waiting for your feedback to know which method is the best one.

- **Upgraded Models Selector** with *already Downloaded indicator*: Next to each model or finetune, you will find a colored square: *Blue* = fully downloaded & available, *Yellow* = partially downloaded & *Black* = not downloaded at all. Please note that the square color will depend on your current choices of requested model quantization.

- **Upgraded Models Manager**: colors squares have also been added so that you can see in glance what has already been downloaded. New filter for a quick model lookout. List of missing files per finetune.

- **Matanyone 2**: everyone favorite Mask extractor has been been updated and is now more precise

*update 10.981*: LTX2.3 Ic Lora Support & expert settings, Matanyone 2, SVI Pro end frames

### February 19th 2026: WanGP v10.951, Breaking the Sound Barrier, Mach 2
Here comes the (last ?) missing bit in WanGP of the *Text To Speech* offering: **emotions**

There isnt many TTS models around that let you express emotions, so I hope you will forgive me for adding an old TTS model (6 months old!) in WanGP: **Index TTS 2**.

But in WanGP, you wont just get the vanilla version of Index TTS:
- *Two speakers Dialogue* (with 2 Cloned Voices) out of the Box
- Each speaker can express *varrying emotions* within the same prompt   
- *Volume Normalization* of the two speaker voices (no one will speak louder)
- Optimized for a *dialogue that can last forever* (new *infinitalk* best friend)
- *GPU Poor version*: VRAM, 6GB will be sufficient
- Highly Optimized with *vllm* & *Cuda Graph acceleration*: *up to 10x faster* than vanilla Index TTS. On high end GPUs, generating a 1 min of dialogue will take only 30s !

Here is how to use it: By default Index TTS, will detect automatically the emotion to apply to a Text Prompt based on the text itself. However, it will apply the same emotion for the whole prompt. If you want a different emotion per sentence, just insert empty lines between each sentence.

You can also set manually which emotion you expect with [] tags, here is one example for one speaker:
```bash
[fear] At the very beginning I was so afraid to speak.
[sadness] Nobody would talk to me. I felt so alone.
[disgust] They would just ignore me and pretend that I didnt exist
[happy] By chance I discovered this wonderful App, and now everything is different.
[anger] I have a new voice and now everybody will have no choice but to listen to my words !!! 
```
You can mix emotions *[sadness,disgust]* or if you want to precise the weight of one or several emotions *[sadness=0.7,disgust]* (in any case total of weights is 1)

Remember two speakers mode requires to insert *"Speaker 1:"* & *"Speaker 2:"* to indicate who is talking.

There is only one snag: Index TTS 2 supports only English & Chinese. But dont' panic ! not all is lost. There is a workaround:
1) Feed *Index TTS 2* with the voice to clone and ask it to generate a sample English spoken text with with the emotion you expected
2) Now ask *Qwen3TTS*, to clone this newly generated voice sample (in English) in the other language you want


### February 16th 2026: WanGP v10.9, Breaking the Sound Barrier
With this new release of WanGP you should have the best TTS (Text To Speech) experience you can find:

- **Qwen3 TTS Powered Up**:  
   - with new *Cuda Graph* optimized mode, speech generation is up to *4x faster*, in some occasion generating 1s of speech may take less than 1s !
   - with int8 quantization, the Qwen3 TTS can work at full speed with only 6 GB of VRAM
   - New **Two Speakers Mode** with **Two Cloned Voices** can generate dialogues with much smoother transitions than with *Kugel Audio* 

- **Heart Mula Powered Up**:
   - with new *Cuda Graph* optimized mode, song generation is up to 6x faster

- **Ace Step 1.5 Powered Up**:
   - WanGP version offers an exclusive support for *vllm* & *int8 quantization* for LM (that is fast LM & low VRAM at the same time)
   - you will only need 10GB of RAM and 6 GB oF VRAM, to run Ace Step with all its features

Also you now have a choice of multiple *Prompt Enhancements* for *Qwen3 TTS* & *Kugel Audio*: *Prompt Enhancer* can now generate for you either a *Monologue* or a *Dialogue between two Speakers*

Please note that to use the new *Cuda Graph*, mode you will need to select either *vllm* or *cuda graph* in *Configuration / Performance / Language Models Decoder Engine*. Profiles 1,3  or 3+ will need to be enabled for the corresponding Model. vllm is a powered up version of cuda graph that may not always work with all GPUs. But don't worry if it is not available for your GPU there will be an automatic fallback to cuda graph.

### February 12th 2026: WanGP v10.84, Easy Metal

- **Ace Step 1.5 Turbo Super Charged**: all the best features of *Ace Step 1.5* are now in *WanGP* and are *Fast* & *Easy* to use:
   * Manual Selection of *Bpm*, *Keyscale*, *Time Signature* & *Language*
   * Use *LM* to auto detect *Bpm*, *Keyscale*, *Time Signature* & *Language* that best suits your *Lyrics*
   * Use *LM* to refine *Music Caption* or auto detect *Song Duration*
   * Choice of *vllm* engine for *LM* for up to *10x faster LM generation!!!*. Also as a WanGP exclusive, *vllm* is offered in *INT8 quantized* format for lower VRAM requirements. Please note you will need to install *Triton* and *Flash Attention 2* (check the *INSTALLATION.Md* for easy install)
   * Use *LM* to refine *Music Caption* (usually the key to get the song theme you expected)
   * UI Makeover to better match vocabulary used in original Ace Step App (but without its complexity...)
   * Refined *System Prompt* used in *Prompt Enhancer* to generate *Lyrics* (I recommend to use the *LLama Joy Prompt Enhancer*)

- **LoKr support**: this "Lora" like format has been tested with *Flux Klein 9B*

- **Optimized Int8 Kernels**: all the *Quantized INT8 checkpoints* (most of the quantized checkpoints) used with WanGP should be now *10% faster !!!*. You will need to install *Triton*. It is experimental, so for the moment it needs to be enabled manually in the *Config / Performance* tab. Please share your feedback on *discord* by mentioning your GPU so that I know if it works properly.

- **Auto Queue Saved if Gen Error**: if for whatever reason you have got an error during a Gen, the queue will now be automatically saved. So you can try again this queue later (with a different config or when the related bug is fixed, if ever ...).

- **UI Updates** (thx *Tophness!*):
Updated the *Self-Refiner UI* to a dynamic, slider-based interface (no more manual text input).
Improved queue reordering: items can now be dragged and dropped directly onto the Top and Bottom buttons while rearranging the queue in order to snap scroll to the top and bottom.

- **Kugel Audio Audio Split**: Kugel Audio is a great model but strangely it tends to accelerate with long speeches. In order to avoid this effect, we need to split audio speeches. You can either do that manually by inserting an *Empty Line* or by specifiying an *Auto Audio Split Duration* (don't worry WanGP will try to split between lines or sentences). 

*update 10.81*: Fixes\
*update 10.82*: UI update\
*update 10.83*: Kugel Audio Split\
*update 10.84*: Ace Step RAM optimizations (fixed memory leak & reduce RAM requirements)

**Note to RTX 50xx owners**: you will need to upgrade to *pytorch 2.10* (see upgrade procedure below) to be able to use *Triton*

### February 4rd 2026: WanGP v10.70, Let's Get Ready To Rumble !
*The competition between Open Source & Close Source has never been that hot !*

- **Ace Step 1.5 Turbo**: this long waited open source project claims to have overthrown *Suno 5*. It lets you generate high multi minutes quality songs. It comes in four flavours: *Vanilla* (No Language Model Preprocessing, **4s Generation Time!!!**) & *3 levels of LM Preprocessing* for a higher Quality (and increasing VRAM requirements)

Please note that when using the *Ace Step LM* variants, this may get very slow with *Memory Profiles 2 or 4* since the LM is an *Autoregressive Model*. It is why I recommed to stick to *Memory Profiles 1/3/3+* unless you have very little VRAM.


- **Kugel Audio 0**: another *TTS* with *Voice Cloning*, this one claims to outperform *ElevenLabs* !!! The nice thing about Kugel Audio is that it can be used to create Dialogues between two cloned voices. Have Fun !

Kugel Audio is entirely an *Autoregressive Model* and quite VRAM Hungry. So either you've got 16GB VRAM and you can run it with *Memory Profile 1/3/3+* or you will have to go the slow way with other Profiles. 

- **LTX-2 Self Refiner**: WanGP exclusive *Self Refiner* has been added to *Distilled/Non Distilled* models, so hopefully this will improve the quality of our Video Gens.


### February 1st 2026: WanGP v10.61, Upgrade Time !

- **LTX-2 Base Tweaks**: new *Quality* features if you found the base model was too fast :
   - New *Modality Guidance* should improve audio / video (lipsync...) according to *LTX-2 team* (beware first *denoising phase* will be 50% slower when used that is if modality guidance> 1)
   - *CFG star*, *Adaptive Project Guidance* should improve quality and better prompt adherence
   - *Skip Layer Guidance*: skipping layer 29 during phase may or may not improve quality
Note that these features are only triggered during first phase of denoising because second phase is distilled denoising no matter what (even on the non distilled model)


- **Flux Klein 4B & 9B Base Models**: *Z Image* has its *base model* in WanGP, so it was fair that *Flux Klein* would have its base model too. Base Models require more steps (up 50) and guidance > 1 but are good starting points for finetunes

The real novelty about this new release is that is has been tested and tuned to work with more recent versions of *Python, Pytorch & Cuda*.
My end goal is to have everbody upgrade to **Python 3.11, Pytorch 2.10, Cuda 13/13.1**.
Once we are all there it will be much easier to provide precompiled kernels for *Nunchaku* *NVPF4*, *Sage Attention*, *Flash Attention*, ...
So please follow the *manual upgrade instructions below* (no Pinokio auto upgrade for the moment) and let me know on Discord if it works with all generations of GPUs (starting from GTX10xx to RTX50xx).
You will find the kernels for this new setup in the **guides/INSTALLATION.md**.

- **Wan Motion Self Refiner**: You will have to thank **Steve Jabz** (*Tophness*) for this one as he has been a big sponsor of the *Self Refiner* and did some extensive study to show me its beauty. The *Self Refiner* should improve the quality of the motion (find it in the *Quality Tab*). It relies on a *Refiner Plan* which indicate which steps should be refined for instance: "2-5:3" (default plan suits well for *lightx2v* 4 steps) means steps 2-3 will be refined 3 times (that is 3 denoising attempts will be made to improve each of them, so if the self refiner is used the gen will be up to 3x slower). For the moment the *Self Refiner* is enabled only on Wan t2v & i2v. If you are happy with it, we will support more models.


**Note that PyTorch 2.10 represents at last a decent upgrade, no memory leak when switching models (pytorch 2.8) and bad perfs / VRAM peaks with VAE decoding (pytorch 2.9).**

*Update*: It seems GTX10xx doesnt support Cuda 13.0. Dont't worry I will keep WanGP compatibility with Pytorch 2.7.1 / Cuda 12.8.\
*Update 10.61*: added Self Refiner

### January 29th 2026: WanGP v10.56, Music for your Hearts

WanGP Special *TTS* (Text To Speech) Release:

- **Heart Mula**: *Suno* quality song with lyrics on your local PC. You can generate up to 4 min of music.

- **Ace Step v1**: while waiting for *Ace Step v1.5* (which should be released very soon), enjoy this oldie (2025!) but goodie song generatpr as an appetizer. Ace Step v1 is a very fast Song generator. It is a *Diffusion* based, so dont hesitate to turn on Profile 4 to go as low as 4B VRAM while remaining fast.

- **Qwen 3 TTS**: you can either do *Voice Cloning*, *Generate a Custom Voice based on a Prompt* or use a *Predefined Voice*

- **TTS Features**:
   - **Early stop** : you can abort a gen, while still keeping what has been generated (will work only for TTS models which are *Autoregressive Models*, no need to ask that for Image/Video gens which are *Diffusion Models*)
   - **Specialized Prompt Enhancers**: if you enter the prompt in Heart Mula *"a song about AI generation"*, *WanGP Prompt Enhancer* will generate the corresponding masterpiece for you. Likewise you can enhance "A speech about AI generation" when using Qwen3 TTS or ChatterBox.
   - **Custom Output folder for Audio Gens**: you can now choose a different folder for the *Audio Outputs*
   - **Default Memory Profile for Audio Models**: TTS models can get very slow if you use profile 4 (being autoregressive models, they will need to load all the layers one per one to generate one single audio token then rinse & repeat). On the other hand, they dont't need as much VRAM, so you can now define a more agressive profile (3+ for instance)

- **Z Image Base**: try it if you are into the *Z Image* hype but it will be probably useless for you unless you are a researcher and / or want to build a finetune out of it. This model requires from 35 to 50 steps (4x to 6x slower than *Z Image turbo*) and cfg > 1 (an additional 2x slower) and there is no *Reinforcement Learning* so Output Images wont be as good. The plus side is a higher diversity and *Native Negative Prompt* (versus Z Image virtual Negative Prompt using *NAG*).

Note that Z Image Base is very sensitive to the *Attention Mode*: it is not compatible with *Sage 1* as it produces black frames. So I have disabled Sage for RTX 30xx. Also there are reports it produces some vertical banding artifacts with *Sage 2*

- **Flux 1/2 NAG** : *Flux 2 Klein* is your new best friend but you miss *Negative Prompts*, *NAG* support for Distilled models will make you best buddies forever as NAG simulates Negative prompts.

- **Various Improvements**:
   - Video /Audio Galleries now support deletions of gens done outside WanGP
   - added *MP3 support* for audio outputs
   - *Check for Updates* button for *Plugins* to see in a glance if any of your plugin can be updated
   - *Prompt Enhancer* generates a different enhanced prompt each timee you click on it. You can define in the config tab its gen parameters (top k, temperature)
   - New *Root Loras* folder can be defined in the config Tab. Useful if you have multiple WanGP instances or want to store easily all your loras in a different hard drive 
   - added new setting *Attention Mode Override* in the *Misc* tab
   - Experimental: allowed changing *Configuration* during a *Generation*

*update 10.51*: new Heart Mula Finetune better at following instructions, Extra settings (cfg, top k) for TTS models, Rife v4\
*update 10.52*: updated plugin list and added version tracking\
*update 10.53*: video/audio galleries now support deletions\
*update 10.54*: added Z Image Base, prompt enhancers improvements, configurable loras root folder\
*update 10.55*: blocked Sage with Z Image on RTX30xx and added override attention mode settings, allowed changing config during generation\
*update 10.56*: added NAG for Flux 1/2 & Ace Step v1

### January 20th 2026: WanGP v10.43, The Cost Saver
*GPUs are expensive, RAM is expensive, SSD are expensive, sadly we live now in a GPU & RAM poor.*

WanGP comes again to the rescue:

- **GGUF support**: as some of you know, I am not a big fan of this format because when used with image / video generative models we don't get any speed boost (matrices multiplications are still done at 16 bits), VRAM savings are small and quality is worse than with int8/fp8. Still gguf has one advantage: it consumes less RAM and harddrive space. So enjoy gguf support. I have added ready to use *Kijai gguf finetunes* for *LTX-2*.

- **Models Manager PlugIn**: use this *Plugin* to identify how much space is taken by each *model* / *finetune* and delete the ones you no longer use. Try to avoid deleting shared files otherwise they will be downloaded again.  

- **LTX-2 Dual Video & Audio Control**: you no longer need to extract the audio track of a *Control Video* if you want to use it as well to drive the video generation. New mode will allow you to use both motion and audio from Video Control.

- **LTX-2 - Custom VAE URL**: some users have asked if they could use the old *Distiller VAE* instead of the new one. To do that, create a *finetune* def based on an existing model definition and save it in the *finetunes/* folder with this entry (check the *docs/FINETUNES.md* doc):
```
		"VAE_URLs": ["https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/ltx-2-19b_vae_old.safetensors"]
```

- **Flux 2 Klein 4B & 9B**: try these distilled models as fast as Z_Image if not faster but with out of the box image edition capabiltities

- **Flux 2 & Qwen Outpainting + Lanpaint**: the inpaint mode of these models support now *outpainting* + more combination possible with *Lanpaint* 

- **RAM Optimizations for multi minutes Videos**: processing, saving, spatial & Temporal upsampling very long videos should require much less RAM. 

- **Text Encoder Cache**: if you are asking a Text prompt already used recently with the current model, it will be taken straight from a cache. The cache is optimized to consume little RAM. It wont work with certain models such as Qwen where the Text Prompt is combined internally with an Image.

*update 10.41*: added Flux 2 klein\
*update 10.42*: added RAM optimizations & Text Encoder Cache\
*update 10.43*: added outpainting for Qwen & Flux 2, Lanpaint for Flux 2

### January 15th 2026: WanGP v10.30, The Need for Speed ...

- **LTX Distilled VAE Upgrade**: *Kijai* has observed that the Distilled VAE produces images that were less sharp that the VAE of the Non Distilled model. I have used this as an opportunity to repackage all the LTX-2 checkpoints and reduce their overal HD footprint since they all share around 5GB. 

**So dont be surprised if the old checkpoints are deleted and new are downloaded !!!**.

- **LTX-2 Multi Passes Loras multipliers**: *LTX-2* supports now loras multiplier that depend on the Pass No. For instance "1;0.5" means 1 will the strength for the first LTX-2 pass and 0.5 will be the strength for the second pass.

- **New Profile 3.5**: here is the lost kid of *Profile 3* & *Profile 5*, you got tons of VRAM, but little RAM ? Profile 3.5 will be your new friend as it will no longer use Reserved RAM to accelerate transfers. Use Profile 3.5 only if you can fit entirely a *Diffusion / Transformer* model in VRAM, otherwise the gen may be much slower.

- **NVFP4 Quantization for LTX-2 & Flux 2**: you will now be able to load *NV FP4* model checkpoints in WanGP. On top of *Wan NV4* which was added recently, we now have *LTX-2 (non distilled)* & *Flux 2* support. NV FP4 uses slightly less VRAM and up to 30% less RAM. 

To enjoy fully the NV FP4 checkpoints (**at least 30% faster gens**), you will need a RTX 50xx and to upgrade to *Pytorch 2.9.1 / Cuda 13* with the latest version of *lightx2v kernels* (check *docs/INSTALLATION.md*). To observe the speed gain, you have to make sure the workload is quite high (high res, long video).


### January 13th 2026: WanGP v10.24, When there is no VRAM left there is still some VRAM left ...

- **LTX-2 - SUPER VRAM OPTIMIZATIONS**  

*With WanGP 10.21 HD 720p Video Gens of 10s just need now 8GB of VRAM!*

LTX Team said this video gen was for 4k. So I had no choice but to squeeze more VRAM with further optimizations.

After much suffering I have managed to reduce by at least 1/3 the VRAM requirements of LTX-2, which means:
  - 10s at 720p can be done with only 8GB of VRAM
  - 10s at 1080p with only 12 GB of VRAM
  - 20s at 1080p with only 16 GB of VRAM
  - 10s at Full 4k (3840 x 2176 !!!) with 24 GB of VRAM.  However the bad news is LTX-2 video is not for 4K, as 4K outputs may give you nightmares ...

3K/4K resolutions will be available only if you enable them in the *Config* / *General* tab.

- **Ic Loras support**: Use a *Control Video* to transfer *Pose*, *Depth*, *Canny Edges*. I have added some extra tweaks: with WanGP you can restrict the transfer to a *masked area*, define a *denoising strength* (how much the control video is going to be followed) and a *masking strength* (how much unmasked area is impacted) 

- **Start Image Strength**: This new slider will appear below a *Start Image* or Source *Video*. If you set it to values lower than 1 you may to reduce the static image effect, you get sometime with LTX-2 i2v
 
- **Custom Gemma Text Encoder for LTX-2**: As a practical case, the *Heretic* text encoder is now supported by WanGP. Check the *finetune* doc, but in short create a *finetune* that has a *text_encoder_URLS* key that contains a list of one or more file paths or URLs.  

- **Experimental Auto Recovery Failed Lora Pin**: Some users (with usually PC with less than 64 GB of RAM) have reported Out Of Memory although a model seemed to load just fine when starting a gen with Loras. This is sometime related to WanGP attempting (and failing due to unsufficient reserved RAM) to pin the Loras to Reserved Memory for faster gen. I have experimented a recovery mode that should release sufficient ressources to continue the Video Gen. This may solve the oom crashes with *LTX-2 Default (non distilled)* 

- **Max Loras Pinned Slider**:  If the Auto Recovery Mode is still not sufficient, I have added a Slider at the bottom of the  *Configuration*  / *Performance* tab that you can use to prevent WanGP from Pinning Loras (to do so set it to 0). As if there is no loading attempt there wont be any crash...

*update 10.21*: added slider Loras Max Pinning slider\
*update 10.22*: added support for custom LTX-2 Text Encoder + Auto Recovery mode if Lora Pinning failed\
*update 10.23*: Fixed text prompt ignore in profile 1 & 2 (this created random output videos)

### January 9st 2026: WanGP v10.11, Spoiled again

- **LTX-2**: here is the long awaited *Ovi Challenger*, LTX-2 generates video and an audio soundtrack. As usual this WanGP version is *low VRAM*. You should be able to run it with as low as 10 GB of VRAM. If you have at least 24 GB of VRAM you will be able to generate 20s at 720p in a single window in only 2 minutes with the distilled model.  WanGP LTX-2 version supports on day one, *Start/End keyframes*, *Sliding-Window* / *Video Continuation* and *Generation Preview*. A *LTX-2 distilled* is part of the package for a very fast generation.

With WanGP v10.11 you can now force your soundtrack, it works like *Multitalk* / *Avatar* except in theory it should work with any kind of sound (not just vocals). Thanks to *Kijai* for showing it was possible.

- **Z Image Twin Folder Turbo**: Z Image even faster as this variant can generate images with as little as 1 step (3 steps recommend) 

- **Qwen LanPaint**: very precise *In Painting*, offers a better integration of the inpainted area in the rest of the image. Beware it is up to 5x slower as it "searches" for the best replacement. 

- **Optimized Pytorch Compiler** : *Patience is the Mother of Virtue*. Finally I may (or may not) have fixed the PyTorch compiler with the Wan models. It should work in much diverse situations and takes much less time. 

- **LongCat Video**: experimental support which includes *LongCat Avatar* a talking head model. For the moment it is mostly for models collectors as it is very slow. It needs 40+ steps and each step contains up 3 passes.

- **MMaudio NSFW**: for alternative audio background

*update v10.11*: LTX-2, use your own soundtrack

### January 1st 2026: WanGP v10.01, Happy New Year !

- **Wan 2.2 i2v Stable Vision Infinity Pro 2**: SVI Pro 2 offers potentially unlimited Videos to Continue for i2v models. It will use either the Start frame as a Reference Image or you may provide an Anchor image to be used across all the windows or multiple Anchor Images one per Window.

- **Wan 2.1 Alpha 2**: This new version of Alpha generates transparent videos with fine-grained alpha detail (hair, glow, smoke). 

- **Qwen Image 2512**: This December release offers Enhanced Human Realism, Finer Natural Details & Improved Text Rendering.

- **Wan NVP4**: *Light2xv nvfp4 support for Wan 2.1 i2v & t2v 1.3B*, you can now load nvfp4 (4 bits quantized file) in WanGP. These will make really a difference with RTX 50xx as they support natively scaled FP4 calculation. Other GPUs will get the pytorch fallback which is slower. This model can be useful for machines with low RAM but don't expect significant VRAM reduction of much faster speed for non RTX 50xx owners. You will need to install the Light2xv kernels.

- **Nunckaku int4 & fp4 support for Qwen 2509 & Z Image**: int4 versions will work with most GPUs, fp4 will accelerate only RTX50xx. You will need to install the nunchaku kernels. See lightx2v nvfp4 above, as the other comments apply here too.

- **Z Image Control Net 2.1**: Control Net upgraded should work better. I have enabled as well inpainting for the control net.

- **New Qwen Loras Accelerators Added**

*Quantization Kernels Wheels for Windows / Python 3.10 / Pytorch 2.70:*
- *Light2xv (WAN-FP4)*
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/WAN_NVP4/lightx2v_kernel-0.0.1-cp39-abi3-win_amd64.whl
   ```

- *Nunchaku*
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Nunchaku/nunchaku-1.1.0+torch2.7-cp310-cp310-win_amd64.whl
   ```
   
### December 23 2025: WanGP v9.92, Early Christmas

- **SCAIL Preview**: enjoy this *Wan Animate*, *Steady Dancer* contender that can support multiple people. Thanks to its 3D positioning, it can take into account which parts of the body are hidden and which are not. 

WanGP version has the following perks: 3D pose Preprocessing entirely rewritten to be fast,  and compatible with any pytorch version, very Low VRAM requirements for multicharacters, experimental long gen mode / sliding windows (SCAIL Preview doesnt  support officialy long gen yet)

- **pi-Flux 2**: you don't use Flux 2 because you find it too slow ? You won't be able to use this excuse anymore: pi-Flux 2 is *4 steps distills* of the best image generator. It supports both image edition and text to image generation.

- **Zandinksy v5** : for the video models collectors among you, you can try the Zandinsky model families, the 2B model quality is especially impressive given its small size

- **Qwen Image Layered**: a new Qwen Image variant that lets you extract RGBA layers of your images so that  each layer can be edited separately

- **Qwen Image Edit Plus 2511**: Qwen Image Edit Plus 2511 improves identity preservation (especially at 1080p) and integrates out of the box popular effects such as religthing and camera changes

- **loras accelerator**: *loras accelerator* for *Wan 2.2 t2v* and *Wan 2.1 i2v* have been added (activable using the *Profile settings* as usual) 

*update 9.91*: added Kandinsky 5 & Qwen Image Layered\
*update 9.92*: added Qwen Image Edit Plus 2511

### December 14 2025: WanGP v9.86, Simple Pleasures...

These two features are going to change the life of many people:
- **Pause Button**: ever had a urge to use your GPU for a very important task that can't wait (a game for instance ?), here comes your new friend the *Pause* button. Not only it will suspend the current gen in progress but it will free most of the VRAM used by WanGP (please note that the RAM by WanGP used wont be released). When you are done just click the *Resume* button to restart exactly from where you stopped.

- **WanGP Headless**:  trouble running remotely WanGP or having some stability issues with Gradio or your Web Browser. This is all past thanks to *WanGP Headless* mode. Here is how it works : first make you shopping list of Video Gen using the classic WanGP gradio interface. When you are done, click the *Save Queue* button and quit WanGP.

Then in your terminal window just write this:
```bash
python wgp.py --process my_queue.zip
```
With WanGP 9.82, you can also process settings file (.json file exported using th *Export Settings* button):
```bash
python wgp.py --process my_settings.json
```
Processing Settings can be useful to do some quick gen / testing if you don't need to provide source image files (otherwise you will need to fill the paths to Start Images, Ref Image, ...)

- **Output Filename Customization**: in the *Misc* tab you can now customize how the file names of new Generation are created, for example:
```
{date(YYYY-MM-DD_HH-mm-ss)}_{seed}_{prompt(50)}, {num_inference_steps}
```

- **Hunyuan Video 1.5 i2v distilled** : for those in need of their daily dose of new models, added *Hunyuan Video 1.5 i2v Distilled* (official release) + Lora Accelerator extracted from it (to be used in future finetunes). Also added *Magcache* support (optimized for 20 steps) for Hunyuan Video 1.5.

- **Wan-Move** : Another model specialized to control motion using a *Start Image* and *Trajectories*. According to the author's paper it is the best one. *Motion Designer* has been upgraded to generate also trajectories for *Wan-Move*.

- **Z-Image Control Net v2** : This is an upgrade of Z-Image Control Net. It offers much better results but requires much more processing an VRAM. But don't panic yet, as it was VRAM optimized. It was not an easy trick as this one is complex. It has also Inpainting support,but I need more info to release this feature.

*update 9.81*: added Hunyuan Video 1.5 i2v distilled + magcache\
*update 9.82*: added Settings headless processing, output file customization, refactored Task edition and queue processing\
*update 9.83*: Qwen Edit+ upgraded: no more any zoom out at 1080p, enabled mask, enabled image refs with inpainting\
*update 9.84*: added Wan-Move support\
*update 9.85*: added Z-Image Control net v2\
*update 9.86*: added NAG support for Z-Image

### December 4 2025: WanGP v9.74, The Alpha & the Omega ... and the Dancer

- **Flux 2**: the best ever open source *Image Generator* has just landed. It does everything very well: generate an *Image* based a *Text Prompt* or combine up to 10 *Images References* 

The only snag is that it is a 60B parameters for the *Transformer* part and 40B parameters for the *Text Encoder* part.

Behold the WanGP Miracle ! Flux 2 wil work with only 8 GB of VRAM if you are happy with 8 bits quantization (no need for lower quality 4bits). With 9GB of VRAM you can run the model at full power. You will need at least 64 GB of RAM. If not maybe Memory Profile 5 will be your friend.

*With WanGP v9.74*, **Flux 2 Control Net** hidden power has also been unleashed from the vanilla model. You can now enjoy Flux 2 *Inpainting* and *Pose transfer*. This can be combined with *Image Refs* to get the best *Identity Preservation* / *Face Swapping* an Image Model can offer: just target the effect to a specific area using a *Mask* and set *Denoising Strength* to 0.9-1.0 and *Masking Strength* to 0.3-0.4 for a perfect blending 

- **Z-Image**: a small model, very fast (8 steps), very low VRAM (optimized even more in WanGP for fun, just in case you want to generate 16 images at a time) that produces outstanding Image quality. Not yet the Flux 2 level, and no Image editing yet but a very good trade-off.

While waiting for Z-Image edit, *WanGP 9.74* offers now support for **Z-Image Fun Control Net**. You can use it for *Pose transfer*, *Canny Edge* transfer. Don't be surprised if it is a bit slower. Please note it will work best at 1080p and will require a minimum of 9 steps.

- **Steady Dancer**: here is *Wan Steady Dancer* a very nice alternative to *Wan Animate*. You can transfer the motion of a Control video in a very smooth way. It will work best with Videos where the action happens center stage (hint: *dancing*). Use the *Lora accelerator* *Fusionix i2v 10 steps* for a fast generation. For higher quality you can set *Condition Guidance* to 2 or if you are very patient keep *Guidance* to a value greater than 1.   

I have added a new Memory Profile *Profile 4+* that is sligthly slower than *Profile 4* but can save you up to 1GB of VRAM with Flux 2.

Also as we have now quite few models and Loras folders. *I have moved all the loras folder in the 'loras' folder*. There are also now unique subfolders for *Wan 5B* and *Wan 1.3B* models. A conversion script should have moved the loras in the right locations, but I advise that you check just in case.

*update 9.71* : added missing source file, have fun !\
*update 9.72* : added Z-Image & Loras reorg\
*update 9.73* : added Steady Dancer\
*update 9.74* : added Z-Image Fun Control Net & Flux 2 Control Net + Masking

### November 24 2025: WanGP v9.62, The Return of the King

So here is *Tencet* who is back in the race: let's welcome **Hunyuan Video 1.5**

Despite only 8B parameters it offers quite a high level of quality. It is not just one model but a family of models:
- Text 2 Video
- Image 2 Video
- Upsamplers (720p & 1080p)

Each model comes on day one with several finetunes specialized for a specific resolution.
The downside right now is that to get the best quality you need to use guidance > 1 and a high number of Steps (20+). 

But dont go away yet ! **LightX2V** (https://huggingface.co/lightx2v/Hy1.5-Distill-Models/) is on deck and has already delivered an *Accelerated 4 steps Finetune* for the *t2v 480p* model. It is part of today's delivery.

I have extracted *LighX2V Magic* into an *8 steps Accelerator Lora* that seems to work for i2v and the other resolutions. This should be good enough while waiting for other the official LighX2V releases (just select this lora in the *Settings* Dropdown Box).

WanGP implementation of Hunyuan 1.5 is quite complete as you will get straight away *Video Gen Preview* (WanGP exclusivity!) and *Sliding Window* support. It is also ready for *Tea Cache* or *Mag Cache* (just waiting for the official parameters) 

*WanGP Hunyuan 1.5 is super VRAM optimized, you will need less than 20 GB of VRAM to generate 12s (289 frames) at 720p.*

Please note Hunyuan v1 Loras are not compatible since the latent space is different. You can add loras for Hunyuan Video 1.5 in the *loras_hunyuan/1.5* folder. 

*Update 9.62* : Added Lora Accelerator\
*Update 9.61* : Added VAE Temporal Tiling

### November 21 2025: WanGP v9.52, And there was motion

In this release WanGP turns you into a Motion Master:
- **Motion Designer**: this new preinstalled home made Graphical Plugin will let you design trajectories for *Vace* and for *Wan 2.2 i2v Time to Move*. 

- **Vace Motion**: this is a less known feature of the almighty *Vace* (this was last Vace feature not yet implemented in WanGP), just put some moving rectangles in your *Control Video* (in Vace raw format) and you will be able to move around people / objects or even the camera. The *Motion Designer* will let you create these trajectories in only a few clicks.

- **Wan 2.2 i2v Time to Move**: a few brillant people (https://github.com/time-to-move/TTM) discovered that you could steer the motion of a model such as *Wan 2.2 i2v* without changing its weights. You just need to apply specific *Control* and *Mask* videos. The *Motion Designer* has an *i2v TTM* mode that will let you generate the videos in the right format. The way it works is that using a *Start Image* you are going to define objects and their corresponding trajectories. For best results, it is recommended to provide as well a *Background Image* which is the Start Image without the objects you are moving (use Qwen for that). TTM works with Loras Accelerators.

*TTM Suggested Settings:  Lightning i2v v1.0 2 Phases (8 Steps), Video to Video, Denoising Strenght 0.9, Masking Strength 0.1*. I will upload Sample Settings later in the *Settings Channel* 

- **PainterI2V**: (https://github.com/princepainter/). You found that the i2v loras accelerators kill the motion ? This is an alternative to 3 phases guidance to restore motion, it is free as it doesnt require any extra processing or changing the weights. It works best in a scene where the background remains the same. In order to control the acceleration in i2v models, you will find a new *Motion Amplitude* slider in the *Quality* tab.

- **Nexus 1.3B**: this is an incredible *Wan 2.1 1.3B* finetune made by @Nexus. It is specialized in *Human Motion* (dance, fights, gym, ...). It is fast as it is already *Causvid* accelerated. Try it with the *Prompt Enhancer* at 720p.

- **Black Start Frames** for Wan 2.1/2.2 i2v: some i2v models can be turned into powerful t2v models by providing a **black frame** as a *Start Frame*. From now on if you dont provide any start frame, WanGP will generate automatically a black start frame of the current output resolution or of the correspondig *End frame resolution* (if any). 

*update 9.51*: Fixed Chrono Edit Output, added Temporal Reasoning Video\
*update 9.52*: Black start frames support for Wan i2v models

### November 12 2025: WanGP v9.44, Free Lunch

**VAE Upsampler** for Wan 2.1/2.2 Text 2 Image and Qwen Image: *spacepxl* has tweaked the VAE Decoder used by *Wan* & *Qwen* so that it can decode and upsample x2 at the same time. The end Result is a Fast High Quality Image Upsampler (much better than Lanczos). Check the *Postprocessing Tab* / *Spatial Upsampling* Dropdown box. Unfortunately this will work only with Image Generation, no support yet for Video Generation. I have also added a VAE Refiner that keeps the existing resolution but slightly improves the details.

**Mocha**: a very requested alternative to *Wan Animate* . Use this model to replace a person in a control video. For best results you will need to provide two reference images for the new the person, the second image should be a face close up. This model seems to be optimized to generate 81 frames. First output frame is often messed up. *Lightx2v t2v 4 steps Lora Accelarator* works well. Please note this model is VRAM hungry, for 81 frames to generate it will process internaly 161 frames.

**Lucy Edit v1.1**: a new version (finetune) has been released. Not sure yet if I like it better than the original one. In theory it should work better with changing the background setting for instance.

**Ovi 1.1**: This new version exists in two flavors 5s & 10s ! Thanks to WanGP VRAM optimisations only 8 GB will be only needed for a 10s generation. Beware, the Prompt syntax has slightly changed since an audio background is now introduced using *"Audio:"* instead of using tags.

**Top Models Selection**: if you are new to WanGP or are simply lost among the numerous models offered by WanGP, just check the updated *Guides* tab. You will find a list of highlighted models and advice about how & when to use them.


*update 9.41*: Added Mocha & Lucy Edit 1.1\
*update 9.42*: Added Ovi 1.1
*update 9.43*: Improved Linux support: no more visual artifacts with fp8 finetunes, auto install ffmpeg, detect audio device, ...
*update 9.44*: Added links to highlighted models in Guide tab

### November 6 2025: WanGP v9.35, How many bananas are too many bananas ?

- **Chrono Edit**: a new original way to edit an Image. This one will generate a Video will that performs the full edition work and return the last Image. It can be hit or a miss but when it works it is quite impressive. Please note you must absolutely use the *Prompt Enhancer* on your *Prompt Instruction* because this model expects a very specific format. The Prompt Enhancer for this model has a specific System Prompt to generate the right Chrono Edit Prompt.

- **LyCoris** support: preliminary basic Lycoris support for this Lora format. At least Qwen Multi Camera should work (https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles). If you have a Lycoris that does not work and it may be interesting please mention it in the Request Channel

- **i2v Enhanced Lightning v2** (update 9.37): added this impressive *Finetune* in the default selection of models, not only it is accelerated (4 steps), but it is very good at following camera and timing instructions.

This finetune loves long prompts. Therefore to increase the prompt readability WanGP supports now multilines prompts (in option).

*update 9.35*: Added a Sample PlugIn App that shows how to collect and modify settings from a PlugIn\
*update 9.37*: Added i2v Enhanced Lightning

### October 29 2025: WanGP v9.21, Why isn't all my VRAM used ?


*WanGP exclusive*:  VRAM requirements have never been that low !

**Wan 2.2 Ovi 10 GB** for all the GPU Poors of the World: *only 6 GB of VRAM to generate 121 frames at 720p*. With 16 GB of VRAM, you may even be able to load all the model in VRAM with *Memory Profile 3*

To get the x10 speed effect just apply the FastWan Lora Accelerator that comes prepackaged with Ovi (acccessible in the  dropdown box Settings at the top)

After thorough testing it appears that *Pytorch 2.8* is causing RAM memory leaks when switching models as it won't release all the RAM. I could not find any workaround. So the default Pytorch version to use with WanGP is back to *Pytorch 2.7*
Unless you want absolutely to use Pytorch compilation which is not stable with Pytorch 2.7 with RTX 50xx , it is recommended to switch back to Pytorch 2.7.1 (tradeoff between 2.8 and 2.7):
```bash
cd Wan2GP
conda activate wan2gp
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/test/cu128
```
You will need to reinstall SageAttention FlashAttnetion, ...

*update v9.21*: Got FastWan to work with Ovi: it is now 10 times faster ! (not including the VAE)\
*update v9.25*: added Chroma Radiance october edition + reverted to pytorch 2.7

### October 24 2025: WanGP v9.10, What else will you ever need after this one ?

With WanGP v9 you will have enough features to go to a desert island with no internet connection and comes back with a full Hollywood movie.

First here are the new models supported:
- **Wan 2.1 Alpha** : a very requested model that can generate videos with *semi transparent background* (as it is very lora picky it supports only the *Self Forcing / lightning* loras accelerators)
- **Chatterbox Multilingual**: the first *Voice Generator* in WanGP. Let's say you have a flu and lost your voice (somehow I can't think of another usecase), the world will still be able to hear you as *Chatterbox* can generate up to 15s clips of your voice using a recorded voice sample. Chatterbox works with numerous languages out the box.
- **Flux DreamOmni2** : another wannabe *Nano Banana* image Editor / image composer. The *Edit Mode* ("Conditional Image is first Main Subject ...") seems to work better than the *Gen Mode* (Conditional Images are People / Objects ..."). If you have at least 16 GB of VRAM it is recommended to force profile 3 for this model (it uses an autoregressive model for the prompt encoding and the start may be slow).
- **Ditto** (new with *WanGP 9.1* !): a powerful Video 2 Video model, can change for instance the style or the material visible in the video. Be aware it is an instruct based model, so the prompt should contain intructions. 

Upgraded Features:
- A new **Audio Gallery** to store your Chatterbox generations and import your audio assets. *Metadata support* (stored gen settings) for *Wav files* generated with WanGP available from day one. 
- **Matanyone** improvements: you can now use it during a video gen, it will *suspend gracefully the Gen in progress*. *Input Video / Images* can be resized for faster processing & lower VRAM. Image version can now generate *Green screens* (not used by WanGP but I did it because someone asked for it and I am nice) and *Alpha masks*.
- **Images Stored in Metadata**: Video Gen *Settings Metadata* that are stored in the Generated Videos can now contain the Start Image, Image Refs used to generate the Video. Many thanks to **Gunther-Schulz** for this contribution
- **Three Levels of Hierarchy** to browse the models / finetunes: you can collect as many finetunes as you want now and they will no longer encumber the UI.
- Added **Loras Accelerators** for *Wan 2.1 1.3B*, *Wan 2.2 i2v*, *Flux* and the latest *Wan 2.2 Lightning*
- Finetunes now support **Custom Text Encoders** : you will need to use the "text_encoder_URLs" key. Please check the finetunes doc. 
- Sometime Less is More: removed the palingenesis finetunes that were controversial

Huge Kudos & Thanks to **Tophness** that has outdone himself with these Great Features:
- **Multicolors Queue** items with **Drag & Drop** to reorder them
- **Edit a Gen Request** that is already in the queue
- Added **Plugin support** to WanGP : found that features are missing in WanGP, you can now add tabs at the top in WanGP. Each tab may contain a full embedded App that can share data with the Video Generator of WanGP. Please check the Plugin guide written by Tophness and don't hesitate to contact him or me on the Discord if you have a plugin you want to share. I have added a new Plugins channels to discuss idea of plugins and help each other developing plugins. *Idea for a PlugIn that may end up popular*: a screen where you view the hard drive space used per model and that will let you remove unused models weights
- Two Plugins ready to use designed & developped by **Tophness**: an **Extended Gallery** and a **Lora multipliers Wizard**

WanGP v9 is now targetting Pytorch 2.8 although it should still work with 2.7, don't forget to upgrade by doing:
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/test/cu128
```
You will need to upgrade Sage Attention or Flash (check the installation guide)

*Update info: you might have some git error message while upgrading to v9 if WanGP is already installed.*
Sorry about that if that's the case, you will need to reinstall WanGP.
There are two different ways to fix this issue while still preserving your data:
1) **Command Line**
If you have access to a terminal window :
```
cd installation_path_of_wangp
git fetch origin && git reset --hard origin/main
pip install -r requirements.txt
```

2) **Generic Method**
a) move outside the installation WanGP folder the folders **ckpts**, **settings**, **outputs** and all the **loras** folders and the file **wgp_config.json**
b) delete the WanGP folder and reinstall
c) move back what you moved in a)


### October 6 2025: WanGP v8.999 - A few last things before the Big Unknown ...

This new version hasn't any new model...

...but temptation to upgrade will be high as it contains a few Loras related features that may change your Life:
- **Ready to use Loras Accelerators Profiles** per type of model that you can apply on your current *Generation Settings*. Next time I will recommend a *Lora Accelerator*, it will be only one click away. And best of all of the required Loras will be downloaded automatically. When you apply an *Accelerator Profile*, input fields like the *Number of Denoising Steps* *Activated Loras*, *Loras Multipliers* (such as "1;0 0;1" ...) will be automatically filled. However your video specific fields will be preserved, so it will be easy to switch between Profiles to experiment. With *WanGP 8.993*, the *Accelerator Loras* are now merged with *Non Accelerator Loras". Things are getting too easy...

- **Embedded Loras URL** : WanGP will now try to remember every Lora URLs it sees. For instance if someone sends you some settings that contain Loras URLs or you extract the Settings of Video generated by a friend with Loras URLs, these URLs will be automatically added to *WanGP URL Cache*. Conversely everything you will share (Videos, Settings, Lset files) will contain the download URLs if they are known. You can also download directly a Lora in WanGP by using the *Download Lora* button a the bottom. The Lora will be immediatly available and added to WanGP lora URL cache. This will work with *Hugging Face* as a repository. Support for CivitAi will come as soon as someone will nice enough to post a GitHub PR ...

- **.lset file** supports embedded Loras URLs. It has never been easier to share a Lora with a friend. As a reminder a .lset file can be created directly from *WanGP Web Interface* and it contains a list of Loras and their multipliers, a Prompt and Instructions how to use these loras (like the Lora's *Trigger*). So with embedded Loras URL, you can send an .lset file by email or share it on discord: it is just a 1 KB tiny text, but with it other people will be able to use Gigabytes Loras as these will be automatically downloaded. 

I have created the new Discord Channel **share-your-settings** where you can post your *Settings* or *Lset files*. I will be pleased to add new Loras Accelerators in the list of WanGP *Accelerators Profiles if you post some good ones there. 

*With the 8.993 update*, I have added support for **Scaled FP8 format**. As a sample case, I have created finetunes for the **Wan 2.2 PalinGenesis** Finetune which is quite popular recently. You will find it in 3 flavors : *t2v*, *i2v* and *Lightning Accelerated for t2v*.

The *Scaled FP8 format* is widely used as it the format used by ... *ComfyUI*. So I except a flood of Finetunes in the *share-your-finetune* channel. If not it means this feature was useless and I will remove it &#x1F608;&#x1F608;&#x1F608;

Not enough Space left on your SSD to download more models ? Would like to reuse Scaled FP8 files in your ComfyUI Folder without duplicating them ? Here comes *WanGP 8.994* **Multiple Checkpoints Folders** : you just need to move the files into different folders / hard drives or reuse existing folders and let know WanGP about it in the *Config Tab* and WanGP will be able to put all the parts together.   

Last but not least the Lora's documentation has been updated.

*update 8.991*: full power of *Vace Lynx* unleashed with new combinations such as Landscape + Face / Clothes + Face  / Injectd Frame (Start/End frames/...) + Face\
*update 8.992*: optimized gen with Lora, should be 10% faster if many loras\
*update 8.993*: Support for *Scaled FP8* format and samples *Paligenesis* finetunes, merged Loras Accelerators and Non Accelerators\
*update 8.994*: Added custom checkpoints folders\
*update 8.999*: fixed a lora + fp8 bug and version sync for the jump to the unknown 

### September 30 2025: WanGP v8.9 - Combinatorics

This new version of WanGP introduces **Wan 2.1 Lynx** the best Control Net so far to transfer *Facial Identity*. You will be amazed to recognize your friends even with a completely different hair style. Congrats to the *Byte Dance team* for this achievement. Lynx works quite with well *Fusionix t2v* 10 steps.

*WanGP 8.9* also illustrate how existing WanGP features can be easily combined with new models. For instance with *Lynx* you will get out of the box *Video to Video* and *Image/Text to Image*.

Another fun combination is *Vace* + *Lynx*, which works much better than *Vace StandIn*. I have added sliders to change the weight of Vace & Lynx to allow you to tune the effects.

### September 28 2025: WanGP v8.76 - ~~Here Are Two Three New Contenders in the Vace Arena !~~ The Never Ending Release 

So in ~~today's~~ this release you will find two Wannabe Vace that covers each only a subset of Vace features but offers some interesting advantages:
- **Wan 2.2 Animate**: this model is specialized in *Body Motion* and *Facial Motion transfers*. It does that very well. You can use this model to either *Replace* a person in an in Video or *Animate* the person of your choice using an existing *Pose Video* (remember *Animate Anyone* ?). By default it will keep the original soundtrack. *Wan 2.2 Animate* seems to be under the hood a derived i2v model and should support the corresponding Loras Accelerators (for instance *FusioniX t2v*). Also as a WanGP exclusivity, you will find support for *Outpainting*.

In order to use Wan 2.2 Animate you will need first to stop by the *Mat Anyone* embedded tool, to extract the *Video Mask* of the person from which you want to extract the motion.

With version WanGP 8.74, there is an extra option that allows you to apply *Relighting* when Replacing a person. Also, you can now Animate a person without providing a Video Mask to target the source of the motion (with the risk it will be less precise) 

For those of you who have a mask halo effect when Animating a character I recommend trying *SDPA attention* and to use the *FusioniX i2v* lora. If this issue persists (this will depend on the control video) you have now a choice of the two *Animate Mask Options* in *WanGP 8.76*. The old masking option which was a WanGP exclusive has been renamed *See Through Mask* because the background behind the animated character was preserved but this creates sometime visual artifacts. The new option which has the shorter name is what you may find elsewhere online. As it uses internally a much larger mask, there is no halo. However the immediate background behind the character is not preserved and may end completely different.

- **Lucy Edit**: this one claims to be a *Nano Banana* for Videos. Give it a video and asks it to change it (it is specialized in clothes changing) and voila ! The nice thing about it is that is it based on the *Wan 2.2 5B* model and therefore is very fast especially if you the *FastWan* finetune that is also part of the package.

Also because I wanted to spoil you:
- **Qwen Edit Plus**: also known as the *Qwen Edit 25th September Update* which is specialized in combining multiple Objects / People. There is also a new support for *Pose transfer* & *Recolorisation*. All of this made easy to use in WanGP. You will find right now only the quantized version since HF crashes when uploading the unquantized version.

- **T2V Video 2 Video Masking**: ever wanted to apply a Lora, a process (for instance Upsampling) or a Text Prompt on only a (moving) part of a Source Video. Look no further, I have added *Masked Video 2 Video* (which works also in image2image) in the *Text 2 Video* models. As usual you just need to use *Matanyone* to creatre the mask.


*Update 8.71*: fixed Fast Lucy Edit that didnt contain the lora
*Update 8.72*: shadow drop of Qwen Edit Plus
*Update 8.73*: Qwen Preview & InfiniteTalk Start image 
*Update 8.74*: Animate Relighting / Nomask mode , t2v Masked Video to Video  
*Update 8.75*: REDACTED  
*Update 8.76*: Alternate Animate masking that fixes the mask halo effect that some users have    

### September 15 2025: WanGP v8.6 - Attack of the Clones

- The long awaited **Vace for Wan 2.2** is at last here or maybe not: it has been released by the *Fun Team* of *Alibaba* and it is not official. You can play with the vanilla version (**Vace Fun**) or with the one accelerated with Loras (**Vace Fan Cocktail**)

- **First Frame / Last Frame for Vace** : Vace models are so powerful that they could do *First frame / Last frame* since day one using the *Injected Frames* feature. However this required to compute by hand the locations of each end frame since this feature expects frames positions. I made it easier to compute these locations by using the "L" alias :

For a video Gen from scratch *"1 L L L"* means the 4 Injected Frames will be injected like this: frame no 1 at the first position, the next frame at the end of the first window, then the following frame at the end of the next window, and so on ....
If you *Continue a Video* , you just need *"L L L"* since the first frame is the last frame of the *Source Video*. In any case remember that numeral frames positions (like "1") are aligned by default to the beginning of the source window, so low values such as 1 will be considered in the past unless you change this behaviour in *Sliding Window Tab/ Control Video, Injected Frames aligment*.

- **Qwen Edit Inpainting** exists now in two versions: the original version of the previous release and a Lora based version. Each version has its pros and cons. For instance the Lora version supports also **Outpainting** ! However it tends to change slightly the original image even outside the outpainted area.

- **Better Lipsync with all the Audio to Video models**: you probably noticed that *Multitalk*, *InfiniteTalk* or *Hunyuan Avatar* had so so lipsync when the audio provided contained some background music. The problem should be solved now thanks to an automated background music removal all done by IA. Don't worry you will still hear the music as it is added back in the generated Video.

### September 11 2025: WanGP v8.5/8.55 - Wanna be a Cropper or a Painter ?

I have done some intensive internal refactoring of the generation pipeline to ease support of existing models or add new models. Nothing really visible but this makes WanGP is little more future proof.

Otherwise in the news:
- **Cropped Input Image Prompts**: as quite often most *Image Prompts* provided (*Start Image, Input Video, Reference Image,  Control Video, ...*) rarely matched your requested *Output Resolution*. In that case I used the resolution you gave either as a *Pixels Budget* or as an *Outer Canvas* for the Generated Video. However in some occasion you really want the requested Output Resolution and nothing else. Besides some models deliver much better Generations if you stick to one of their supported resolutions. In order to address this need I have added a new Output Resolution choice in the *Configuration Tab*:  **Dimensions Correspond to the Ouput Weight & Height as the Prompt Images will be Cropped to fit Exactly these dimensins**. In short if needed the *Input Prompt Images* will be cropped (centered cropped for the moment). You will see this can make quite a difference for some models

- *Qwen Edit* has now a new sub Tab called **Inpainting**, that lets you target with a brush which part of the *Image Prompt* you want to modify. This is quite convenient if you find that Qwen Edit modifies usually too many things. Of course, as there are more constraints for Qwen Edit don't be surprised if sometime it will return the original image unchanged. A piece of advise: describe in your *Text Prompt* where (for instance *left to the man*, *top*, ...) the parts that you want to modify are located.

The mask inpainting is fully compatible with *Matanyone Mask generator*: generate first an *Image Mask* with Matanyone, transfer it to the current Image Generator and modify the mask with the *Paint Brush*. Talking about matanyone I have fixed a bug that caused a mask degradation with long videos (now WanGP Matanyone is as good as the original app and still requires 3 times less VRAM)

- This **Inpainting Mask Editor** has been added also to *Vace Image Mode*. Vace is probably still one of best Image Editor today. Here is a very simple & efficient workflow that do marvels with Vace:
Select *Vace Cocktail > Control Image Process = Perform Inpainting & Area Processed = Masked Area > Upload a Control Image, then draw your mask directly on top of the image & enter a text Prompt that describes the expected change > Generate > Below the Video Gallery click 'To Control Image' > Keep on doing more changes*.

Doing more sophisticated thing Vace Image Editor works very well too: try Image Outpainting, Pose transfer, ...

For the best quality I recommend to set in *Quality Tab* the option: "*Generate a 9 Frames Long video...*" 

**update 8.55**: Flux Festival
- **Inpainting Mode** also added for *Flux Kontext*
- **Flux SRPO** : new finetune with x3 better quality vs Flux Dev according to its authors. I have also created a *Flux SRPO USO* finetune which is certainly the best open source *Style Transfer* tool available
- **Flux UMO**: model specialized in combining multiple reference objects / people together. Works quite well at 768x768

Good luck with finding your way through all the Flux models names !

### September 5 2025: WanGP v8.4 - Take me to Outer Space
You have probably seen these short AI generated movies created using *Nano Banana* and the *First Frame - Last Frame* feature of *Kling 2.0*. The idea is to generate an image, modify a part of it with Nano Banana and give the these two images to Kling that will generate the Video between these two images, use now the previous Last Frame as the new First Frame, rinse and repeat and you get a full movie.

I have made it easier to do just that with *Qwen Edit* and *Wan*:
- **End Frames can now be combined with Continue a Video** (and not just a Start Frame)
- **Multiple End Frames can be inputed**, each End Frame will be used for a different Sliding Window

You can plan in advance all your shots (one shot = one Sliding Window) : I recommend using Wan 2.2 Image to Image with multiple End Frames (one for each shot / Sliding Window), and a different Text Prompt for each shot / Sliding Winow (remember to enable *Sliding Windows/Text Prompts Will be used for a new Sliding Window of the same Video Generation*)

The results can quite be impressive. However, Wan 2.1 & 2.2 Image 2 Image are restricted to a single overlap frame when using Slide Windows, which means only one frame is reeused for the motion. This may be unsufficient if you are trying to connect two shots with fast movement.

This is where *InfinitTalk* comes into play. Beside being one best models to generate animated audio driven avatars, InfiniteTalk uses internally more one than motion frames. It is quite good to maintain the motions between two shots. I have tweaked InfinitTalk so that **its motion engine can be used even if no audio is provided**.
So here is how to use InfiniteTalk: enable *Sliding Windows/Text Prompts Will be used for a new Sliding Window of the same Video Generation*), and if you continue an existing Video  *Misc/Override Frames per Second" should be set to "Source Video*. Each Reference Frame inputed will play the same role as the End Frame except it wont be exactly an End Frame (it will correspond more to a middle frame, the actual End Frame will differ but will be close)


You will find below a 33s movie I have created using these two methods. Quality could be much better as I havent tuned at all the settings (I couldn't bother, I used 10 steps generation without Loras Accelerators for most of the gens).

### September 2 2025: WanGP v8.31 - At last the pain stops

- This single new feature should give you the strength to face all the potential bugs of this new release:
**Images Management (multiple additions or deletions, reordering) for Start Images / End Images / Images References.**  

- Unofficial **Video to Video (Non Sparse this time) for InfinitTalk**. Use the Strength Noise slider to decide how much motion of the original window you want to keep. I have also *greatly reduced the VRAM requirements for Multitalk / Infinitalk* (especially the multispeakers version & when generating at 1080p). 

- **Experimental Sage 3 Attention support**: you will need to deserve this one, first you need a Blackwell GPU (RTX50xx) and request an access to Sage 3 Github repo, then you will have to compile Sage 3, install it and cross your fingers ...


*update 8.31: one shouldnt talk about bugs if one doesn't want to attract bugs*


## 🔥 Latest News
### August 29 2025: WanGP v8.21 -  Here Goes Your Weekend

- **InfiniteTalk Video to Video**: this feature can be used for Video Dubbing. Keep in mind that it is a *Sparse Video to Video*, that is internally only image is used by Sliding Window. However thanks to the new *Smooth Transition* mode, each new clip is connected to the previous and all the camera work is done by InfiniteTalk. If you dont get any transition, increase the number of frames of a Sliding Window (81 frames recommended)

- **StandIn**: very light model specialized in Identity Transfer. I have provided two versions of Standin: a basic one derived from the text 2 video model and another based on Vace. If used with Vace, the last reference frame given to Vace will be also used for StandIn

- **Flux ESO**: a new Flux dervied *Image Editing tool*, but this one is specialized both in *Identity Transfer* and *Style Transfer*. Style has to be understood in its wide meaning: give a reference picture of a person and another one of Sushis and you will turn this person into Sushis

### August 24 2025: WanGP v8.1 -  the RAM Liberator

- **Reserved RAM entirely freed when switching models**, you should get much less out of memory related to RAM. I have also added a button in *Configuration / Performance* that will release most of the RAM used by WanGP if you want to use another application without quitting WanGP 
- **InfiniteTalk** support: improved version of Multitalk that supposedly supports very long video generations based on an audio track. Exists in two flavors (*Single Speaker* and *Multi Speakers*) but doesnt seem to be compatible with Vace. One key new feature compared to Multitalk is that you can have different visual shots associated to the same audio: each Reference frame you provide you will be associated to a new Sliding Window. If only Reference frame is provided, it will be used for all windows. When Continuing a video, you can either continue the current shot (no Reference Frame) or add new shots (one or more Reference Frames).\
If you are not into audio, you can use still this model to generate infinite long image2video, just select "no speaker". Last but not least, Infinitetalk works works with all the Loras accelerators.
- **Flux Chroma 1 HD** support: uncensored flux based model and lighter than Flux (8.9B versus 12B) and can fit entirely in VRAM with only 16 GB of VRAM. Unfortunalely it is not distilled and you will need CFG at minimum 20 steps

### August 21 2025: WanGP v8.01 - the killer of seven

- **Qwen Image Edit** : Flux Kontext challenger (prompt driven image edition). Best results (including Identity preservation) will be obtained at 720p. Beyond you may get image outpainting and / or lose identity preservation. Below 720p prompt adherence will be worse. Qwen Image Edit works with Qwen Lora Lightning 4 steps. I have also unlocked all the resolutions for Qwen models. Bonus Zone: support for multiple image compositions but identity preservation won't be as good.
- **On demand Prompt Enhancer** (needs to be enabled in Configuration Tab) that you can use to Enhance a Text Prompt before starting a Generation. You can refine the Enhanced Prompt or change the original Prompt.
- Choice of a **Non censored Prompt Enhancer**. Beware this is one is VRAM hungry and will require 12 GB of VRAM to work
- **Memory Profile customizable per model** : useful to set for instance Profile 3 (preload the model entirely in VRAM) with only Image Generation models, if you have 24 GB of VRAM. In that case Generation will be much faster because with Image generators (contrary to Video generators) as a lot of time is wasted in offloading 
- **Expert Guidance Mode**: change the Guidance during the generation up to 2 times. Very useful with Wan 2.2 Ligthning to reduce the slow motion effect. The idea is to insert a CFG phase before the 2 accelerated phases that follow and have no Guidance. I have added the finetune *Wan2.2 Vace Lightning 3 Phases 14B* with a prebuilt configuration. Please note that it is a 8 steps process although the lora lightning is 4 steps. This expert guidance mode is also available with Wan 2.1.

*WanGP 8.01 update, improved Qwen Image Edit Identity Preservation*
### August 12 2025: WanGP v7.7777 - Lucky Day(s)

This is your lucky day ! thanks to new configuration options that will let you store generated Videos and Images in lossless compressed formats, you will find they in fact they look two times better without doing anything !

Just kidding, they will be only marginally better, but at least this opens the way to professionnal editing.

Support:
- Video: x264, x264 lossless, x265
- Images: jpeg, png, webp, wbp lossless
Generation Settings are stored in each of the above regardless of the format (that was the hard part).

Also you can now choose different output directories for images and videos.

unexpected luck: fixed lightning 8 steps for Qwen, and lightning 4 steps for Wan 2.2, now you just need 1x multiplier no weird numbers. 
*update 7.777 : oops got a crash a with FastWan ? Luck comes and goes, try a new update, maybe you will have a better chance this time*
*update 7.7777 : Sometime good luck seems to last forever. For instance what if Qwen Lightning 4 steps could also work with WanGP ?*
- https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors (Qwen Lightning 4 steps)
- https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors (new improved version of Qwen Lightning 8 steps)


### August 10 2025: WanGP v7.76 - Faster than the VAE ...
We have a funny one here today: FastWan 2.2 5B, the Fastest Video Generator, only 20s to generate 121 frames at 720p. The snag is that VAE is twice as slow... 
Thanks to Kijai for extracting the Lora that is used to build the corresponding finetune.

*WanGP 7.76: fixed the messed up I did to i2v models (loras path was wrong for Wan2.2 and Clip broken)*

### August 9 2025: WanGP v7.74 - Qwen Rebirth part 2
Added support for Qwen Lightning lora for a 8 steps generation (https://huggingface.co/lightx2v/Qwen-Image-Lightning/blob/main/Qwen-Image-Lightning-8steps-V1.0.safetensors). Lora is not normalized and you can use a multiplier around 0.1.

Mag Cache support for all the Wan2.2 models Don't forget to set guidance to 1 and 8 denoising steps , your gen will be 7x faster !

### August 8 2025: WanGP v7.73 - Qwen Rebirth
Ever wondered what impact not using Guidance has on a model that expects it ? Just look at Qween Image in WanGP 7.71 whose outputs were erratic. Somehow I had convinced myself that Qwen was a distilled model. In fact Qwen was dying for a negative prompt. And in WanGP 7.72 there is at last one for him.

As Qwen is not so picky after all I have added also quantized text encoder which reduces the RAM requirements of Qwen by 10 GB (the text encoder quantized version produced garbage before) 

Unfortunately still the Sage bug for older GPU architectures. Added Sdpa fallback for these architectures.

*7.73 update: still Sage / Sage2 bug for GPUs before RTX40xx. I have added a detection mechanism that forces Sdpa attention if that's the case*


### August 6 2025: WanGP v7.71 - Picky, picky

This release comes with two new models :
- Qwen Image: a Commercial grade Image generator capable to inject full sentences in the generated Image while still offering incredible visuals
- Wan 2.2 TextImage to Video 5B: the last Wan 2.2 needed if you want to complete your Wan 2.2 collection (loras for this folder can be stored in "\loras\5B"     )

There is catch though, they are very picky if you want to get good generations: first they both need lots of steps (50 ?) to show what they have to offer. Then for Qwen Image I had to hardcode the supported resolutions, because if you try anything else, you will get garbage. Likewise Wan 2.2 5B will remind you of Wan 1.0 if you don't ask for at least  720p. 

*7.71 update: Added VAE Tiling for both Qwen Image and Wan 2.2 TextImage to Video 5B, for low VRAM during a whole gen.*


### August 4 2025: WanGP v7.6 - Remuxed

With this new version you won't have any excuse if there is no sound in your video.

*Continue Video* now works with any video that has already some sound (hint: Multitalk ).

Also, on top of MMaudio and the various sound driven models I have added the ability to use your own soundtrack.

As a result you can apply a different sound source on each new video segment when doing a *Continue Video*. 

For instance:
- first video part: use Multitalk with two people speaking
- second video part: you apply your own soundtrack which will gently follow the multitalk conversation
- third video part: you use Vace effect and its corresponding control audio will be concatenated to the rest of the audio

To multiply the combinations I have also implemented *Continue Video* with the various image2video models.

Also:
- End Frame support added for LTX Video models
- Loras can now be targetted specifically at the High noise or Low noise models with Wan 2.2, check the Loras and Finetune guides
- Flux Krea Dev support

### July 30 2025: WanGP v7.5:  Just another release ... Wan 2.2 part 2
Here is now Wan 2.2 image2video a very good model if you want to set Start and End frames. Two Wan 2.2 models delivered, only one to go ...

Please note that although it is an image2video model it is structurally very close to Wan 2.2 text2video (same layers with only a different initial projection). Given that Wan 2.1 image2video loras don't work too well (half of their tensors are not supported), I have decided that this model will look for its loras in the text2video loras folder instead of the image2video folder.

I have also optimized RAM management with Wan 2.2 so that loras and modules will be loaded only once in RAM and Reserved RAM, this saves up to 5 GB of RAM which can make a difference...

And this time I really removed Vace Cocktail Light which gave a blurry vision.

### July 29 2025: WanGP v7.4:  Just another release ... Wan 2.2 Preview
Wan 2.2 is here.  The good news is that WanGP wont require a single byte of extra VRAM to run it and it will be as fast as Wan 2.1. The bad news is that you will need much more RAM if you want to leverage entirely this new model since it has twice has many parameters.

So here is a preview version of Wan 2.2 that is without the 5B model and Wan 2.2 image to video for the moment.

However as I felt bad to deliver only half of the wares, I gave you instead .....** Wan 2.2 Vace Experimental Cocktail** !

Very good surprise indeed, the loras and Vace partially work with Wan 2.2. We will need to wait for the official Vace 2.2 release since some Vace features are broken like identity preservation

Bonus zone: Flux multi images conditions has been added, or maybe not if I broke everything as I have been distracted by Wan...

7.4 update: I forgot to update the version number. I also removed Vace Cocktail light which didnt work well.

### July 27 2025: WanGP v7.3 : Interlude
While waiting for Wan 2.2, you will appreciate the model selection hierarchy which is very useful to collect even more models. You will also appreciate that WanGP remembers which model you used last in each model family.

### July 26 2025: WanGP v7.2 : Ode to Vace
I am really convinced that Vace can do everything the other models can do and in a better way especially as Vace can be combined with Multitalk.

Here are some new Vace improvements:
- I have provided a default finetune named *Vace Cocktail*  which is a model created on the fly using the Wan text 2 video model and the Loras used to build FusioniX. The weight of the *Detail Enhancer* Lora has been reduced to improve identity preservation. Copy the model definition in *defaults/vace_14B_cocktail.json* in the *finetunes/* folder to change the Cocktail composition. Cocktail contains already some Loras acccelerators so no need to add on top a Lora Accvid, Causvid or Fusionix, ... . The whole point of Cocktail is to be able  to build you own FusioniX (which originally is a combination of 4 loras) but without the inconvenient of FusioniX.
- Talking about identity preservation, it tends to go away when one generates a single Frame instead of a Video which is shame for our Vace photoshop. But there is a solution : I have added an Advanced Quality option, that tells WanGP to generate a little more than a frame (it will still keep only the first frame). It will be a little slower but you will be amazed how Vace Cocktail combined with this option will preserve identities (bye bye *Phantom*). 
- As in practise I have observed one switches frequently between *Vace text2video* and *Vace text2image* I have put them in the same place they are now just one tab away, no need to reload the model. Likewise *Wan text2video* and *Wan tex2image* have been merged.
- Color fixing when using Sliding Windows. A new postprocessing *Color Correction* applied automatically by default (you can disable it in the *Advanced tab Sliding Window*) will try to match the colors of the new window with that of the previous window. It doesnt fix all the unwanted artifacts of the new window but at least this makes the transition smoother. Thanks to the multitalk team for the original code.

Also you will enjoy our new real time statistics (CPU / GPU usage, RAM / VRAM used, ... ). Many thanks to **Redtash1** for providing the framework for this new feature ! You need to go in the Config tab to enable real time stats.


### July 21 2025: WanGP v7.12
- Flux Family Reunion : *Flux Dev* and *Flux Schnell* have been invited aboard WanGP. To celebrate that, Loras support for the Flux *diffusers* format has also been added.

- LTX Video upgraded to version 0.9.8: you can now generate 1800 frames (1 min of video !) in one go without a sliding window. With the distilled model it will take only 5 minutes with a RTX 4090 (you will need 22 GB of VRAM though). I have added options to select higher humber frames if you want to experiment (go to Configuration Tab / General / Increase the Max Number of Frames, change the value and restart the App)

- LTX Video ControlNet : it is a Control Net that allows you for instance to transfer a Human motion or Depth from a control video. It is not as powerful as Vace but can produce interesting things especially as now you can generate quickly a 1 min video. Under the scene IC-Loras (see below) for Pose, Depth and Canny are automatically loaded for you, no need to add them. 

- LTX IC-Lora support: these are special Loras that consumes a conditional image or video
Beside the pose, depth and canny IC-Loras transparently loaded there is the *detailer* (https://huggingface.co/Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8) which is basically an upsampler. Add the *detailer* as a Lora and use LTX Raw Format as control net choice to use it.

- Matanyone is now also for the GPU Poor as its VRAM requirements have been divided by 2! (7.12 shadow update)

- Easier way to select video resolution 

### July 15 2025: WanGP v7.0 is an AI Powered Photoshop
This release turns the Wan models into Image Generators. This goes way more than allowing to generate a video made of single frame :
- Multiple Images generated at the same time so that you can choose the one you like best.It is Highly VRAM optimized so that you can generate for instance 4 720p Images at the same time with less than 10 GB
- With the *image2image* the original text2video WanGP becomes an image upsampler / restorer
- *Vace image2image* comes out of the box with image outpainting, person / object replacement, ...
- You can use in one click a newly Image generated as Start Image or Reference Image for a Video generation

And to complete the full suite of AI Image Generators, Ladies and Gentlemen please welcome for the first time in WanGP : **Flux Kontext**.\
As a reminder Flux Kontext is an image editor : give it an image and a prompt and it will do the change for you.\
This highly optimized version of Flux Kontext will make you feel that you have been cheated all this time as WanGP Flux Kontext requires only 8 GB of VRAM to generate 4 images at the same time with no need for quantization.

WanGP v7 comes with *Image2image* vanilla and *Vace FusinoniX*. However you can build your own finetune where you will combine a text2video or Vace model with any combination of Loras.

Also in the news:
- You can now enter the *Bbox* for each speaker in *Multitalk* to precisely locate who is speaking. And to save some headaches the *Image Mask generator* will give you the *Bbox* coordinates of an area you have selected.
- *Film Grain* post processing to add a vintage look at your video
- *First Last Frame to Video* model should work much better now as I have discovered rencently its implementation was not complete
- More power for the finetuners, you can now embed Loras directly in the finetune definition. You can also override the default models (titles, visibility, ...) with your own finetunes. Check the doc that has been updated.


### July 10 2025: WanGP v6.7, is NAG a game changer ? you tell me
Maybe you knew that already but most *Loras accelerators* we use today (Causvid, FusioniX) don't use *Guidance* at all (that it is *CFG* is set to 1). This helps to get much faster generations but the downside is that *Negative Prompts* are completely ignored (including the default ones set by the models). **NAG** (https://github.com/ChenDarYen/Normalized-Attention-Guidance) aims to solve that by injecting the *Negative Prompt* during the *attention* processing phase.

So WanGP 6.7 gives you NAG, but not any NAG, a *Low VRAM* implementation, the default one ends being VRAM greedy. You will find NAG in the *General* advanced tab for most Wan models. 

Use NAG especially when Guidance is set to 1. To turn it on set the **NAG scale** to something around 10. There are other NAG parameters **NAG tau** and **NAG alpha** which I recommend to change only if you don't get good results by just playing with the NAG scale. Don't hesitate to share on this discord server the best combinations for these 3 parameters.

The authors of NAG claim that NAG can also be used when using a Guidance (CFG > 1) and to improve the prompt adherence.

### July 8 2025: WanGP v6.6, WanGP offers you **Vace Multitalk Dual Voices Fusionix Infinite** :
**Vace** our beloved super Control Net has been combined with **Multitalk** the new king in town that can animate up to two people speaking (**Dual Voices**). It is accelerated by the **Fusionix** model and thanks to *Sliding Windows* support and *Adaptive Projected Guidance* (much slower but should reduce the reddish effect with long videos) your two people will be able to talk for very a long time (which is an **Infinite** amount of time in the field of video generation).

Of course you will get as well *Multitalk* vanilla and also *Multitalk 720p* as a bonus.

And since I am mister nice guy I have enclosed as an exclusivity an *Audio Separator* that will save you time to isolate each voice when using Multitalk with two people.

As I feel like resting a bit I haven't produced yet a nice sample Video to illustrate all these new capabilities. But here is the thing, I ams sure you will publish in the *Share Your Best Video* channel your *Master Pieces*. The best ones will be added to the *Announcements Channel* and will bring eternal fame to its authors.

But wait, there is more:
- Sliding Windows support has been added anywhere with Wan models, so imagine with text2video recently upgraded in 6.5 into a video2video, you can now upsample very long videos regardless of your VRAM. The good old image2video model can now reuse the last image to produce new videos (as requested by many of you)
- I have added also the capability to transfer the audio of the original control video (Misc. advanced tab) and an option to preserve the fps into the generated video, so from now on you will be to upsample / restore your old families video and keep the audio at their original pace. Be aware that the duration will be limited to 1000 frames as I still need to add streaming support for unlimited video sizes.

Also, of interest too:
- Extract video info from Videos that have not been generated by WanGP, even better you can also apply post processing (Upsampling / MMAudio) on non WanGP videos
- Force the generated video fps to your liking, works wery well with Vace when using a Control Video
- Ability to chain URLs of Finetune models (for instance put the URLs of a model in your main finetune and reference this finetune in other finetune models to save time)

### July 2 2025: WanGP v6.5.1, WanGP takes care of you: lots of quality of life features:
- View directly inside WanGP the properties (seed, resolutions, length, most settings...) of the past generations
- In one click use the newly generated video as a Control Video or Source Video to be continued 
- Manage multiple settings for the same model and switch between them using a dropdown box 
- WanGP will keep the last generated videos in the Gallery and will remember the last model you used if you restart the app but kept the Web page open
- Custom resolutions : add a file in the WanGP folder with the list of resolutions you want to see in WanGP (look at the instruction readme in this folder)

Taking care of your life is not enough, you want new stuff to play with ?
- MMAudio directly inside WanGP : add an audio soundtrack that matches the content of your video. By the way it is a low VRAM MMAudio and 6 GB of VRAM should be sufficient. You will need to go in the *Extensions* tab of the WanGP *Configuration* to enable MMAudio
- Forgot to upsample your video during the generation ? want to try another MMAudio variation ? Fear not you can also apply upsampling or add an MMAudio track once the video generation is done. Even better you can ask WangGP for multiple variations of MMAudio to pick the one you like best
- MagCache support: a new step skipping approach, supposed to be better than TeaCache. Makes a difference if you usually generate with a high number of steps
- SageAttention2++ support : not just the compatibility but also a slightly reduced VRAM usage
- Video2Video in Wan Text2Video : this is the paradox, a text2video can become a video2video if you start the denoising process later on an existing video
- FusioniX upsampler: this is an illustration of Video2Video in Text2Video. Use the FusioniX text2video model with an output resolution of 1080p and a denoising strength of 0.25 and you will get one of the best upsamplers (in only 2/3 steps, you will need lots of VRAM though). Increase the denoising strength and you will get one of the best Video Restorer
- Choice of Wan Samplers / Schedulers
- More Lora formats support

**If you had upgraded to v6.5 please upgrade again to 6.5.1 as this will fix a bug that ignored Loras beyond the first one**

### June 23 2025: WanGP v6.3, Vace Unleashed. Thought we couldnt squeeze Vace even more ?
- Multithreaded preprocessing when possible for faster generations
- Multithreaded frames Lanczos Upsampling as a bonus
- A new Vace preprocessor : *Flow* to extract fluid motion
- Multi Vace Controlnets: you can now transfer several properties at the same time. This opens new possibilities to explore, for instance if you transfer *Human Movement* and *Shapes* at the same time for some reasons the lighting of your character will take into account much more the environment of your character.
- Injected Frames Outpainting, in case you missed it in WanGP 6.21

Don't know how to use all of the Vace features ? Check the Vace Guide embedded in WanGP as it has also been updated.


### June 19 2025: WanGP v6.2, Vace even more Powercharged
👋 Have I told you that I am a big fan of Vace ? Here are more goodies to unleash its power: 
- If you ever wanted to watch Star Wars in 4:3, just use the new *Outpainting* feature and it will add the missing bits of image at the top and the bottom of the screen. The best thing is *Outpainting* can be combined with all the other Vace modifications, for instance you can change the main character of your favorite movie at the same time  
- More processing can combined at the same time  (for instance the depth process can be applied outside the mask)
- Upgraded the depth extractor to Depth Anything 2 which is much more detailed

As a bonus, I have added two finetunes based on the Safe-Forcing technology (which requires only 4 steps to generate a video): Wan 2.1 text2video Self-Forcing and Vace Self-Forcing. I know there is Lora around but the quality of the Lora is worse (at least with Vace) compared to the full model. Don't hesitate to share your opinion about this on the discord server. 
### June 17 2025: WanGP v6.1, Vace Powercharged
👋 Lots of improvements for Vace the Mother of all Models:
- masks can now be combined with on the fly processing of a control video, for instance you can extract the motion of a specific person defined by a mask
- on the fly modification of masks : reversed masks (with the same mask you can modify the background instead of the people covered by the masks), enlarged masks (you can cover more area if for instance the person you are trying to inject is larger than the one in the mask), ...
- view these modified masks directly inside WanGP during the video generation to check they are really as expected
- multiple frames injections: multiples frames can be injected at any location of the video
- expand past videos in on click: just select one generated video to expand it

Of course all these new stuff work on all Vace finetunes (including Vace Fusionix).

Thanks also to Reevoy24 for adding a Notfication sound at the end of a generation and for fixing the background color of the current generation summary.

### June 12 2025: WanGP v6.0
👋 *Finetune models*: You find the 20 models supported by WanGP not sufficient ? Too impatient to wait for the next release to get the support for a newly released model ? Your prayers have been answered: if a new model is compatible with a model architecture supported by WanGP, you can add by yourself the support for this model in WanGP by just creating a finetune model definition. You can then store this model in the cloud (for instance in Huggingface) and the very light finetune definition file can be easily shared with other users. WanGP will download automatically the finetuned model for them.

To celebrate the new finetunes support, here are a few finetune gifts (directly accessible from the model selection menu):
- *Fast Hunyuan Video* : generate model t2v in only 6 steps
- *Hunyuan Vido AccVideo* : generate model t2v in only 5 steps
- *Wan FusioniX*: it is a combo of AccVideo / CausVid ans other models and can generate high quality Wan videos in only 8 steps

One more thing...

The new finetune system can be used to combine complementaty models : what happens when you combine  Fusionix Text2Video and Vace Control Net ?

You get **Vace FusioniX**: the Ultimate Vace Model, Fast (10 steps, no need for guidance) and with a much better quality Video than the original slower model (despite being the best Control Net out there). Here goes one more finetune...

Check the *Finetune Guide* to create finetune models definitions and share them on the WanGP discord server.

### June 11 2025: WanGP v5.5
👋 *Hunyuan Video Custom Audio*: it is similar to Hunyuan Video Avatar excpet there isn't any lower limit on the number of frames and you can use your reference images in a different context than the image itself\
*Hunyuan Video Custom Edit*: Hunyuan Video Controlnet, use it to do inpainting and replace a person in a video while still keeping his poses. Similar to Vace but less restricted than the Wan models in terms of content...

### June 6 2025: WanGP v5.41
👋 Bonus release: Support for **AccVideo** Lora to speed up x2 Video generations in Wan models. Check the Loras documentation to get the usage instructions of AccVideo.

### June 6 2025: WanGP v5.4
👋 World Exclusive : Hunyuan Video Avatar Support ! You won't need 80 GB of VRAM nor 32 GB oF VRAM, just 10 GB of VRAM will be sufficient to generate up to 15s of high quality speech / song driven Video at a high speed with no quality degradation. Support for TeaCache included.

### May 26, 2025: WanGP v5.3
👋 Happy with a Video generation and want to do more generations using the same settings but you can't remember what you did or you find it too hard to copy/paste one per one each setting from the file metadata? Rejoice! There are now multiple ways to turn this tedious process into a one click task:
- Select one Video recently generated in the Video Gallery and click *Use Selected Video Settings*
- Click *Drop File Here* and select a Video you saved somewhere, if the settings metadata have been saved with the Video you will be able to extract them automatically
- Click *Export Settings to File* to save on your harddrive the current settings. You will be able to use them later again by clicking *Drop File Here* and select this time a Settings json file

### May 23, 2025: WanGP v5.21
👋 Improvements for Vace: better transitions between Sliding Windows, Support for Image masks in Matanyone, new Extend Video for Vace, different types of automated background removal

### May 20, 2025: WanGP v5.2
👋 Added support for Wan CausVid which is a distilled Wan model that can generate nice looking videos in only 4 to 12 steps. The great thing is that Kijai (Kudos to him!) has created a CausVid Lora that can be combined with any existing Wan t2v model 14B like Wan Vace 14B. See [LORAS.md](LORAS.md) for instructions on how to use CausVid.

Also as an experiment I have added support for the MoviiGen, the first model that claims to be capable of generating 1080p videos (if you have enough VRAM (20GB...) and be ready to wait for a long time...). Don't hesitate to share your impressions on the Discord server.

### May 18, 2025: WanGP v5.1
👋 Bonus Day, added LTX Video 13B Distilled: generate in less than one minute, very high quality Videos!

### May 17, 2025: WanGP v5.0
👋 One App to Rule Them All! Added support for the other great open source architectures:
- **Hunyuan Video**: text 2 video (one of the best, if not the best t2v), image 2 video and the recently released Hunyuan Custom (very good identity preservation when injecting a person into a video)
- **LTX Video 13B** (released last week): very long video support and fast 720p generation. Wan GP version has been greatly optimized and reduced LTX Video VRAM requirements by 4!

Also:
- Added support for the best Control Video Model, released 2 days ago: Vace 14B
- New Integrated prompt enhancer to increase the quality of the generated videos

*You will need one more `pip install -r requirements.txt`*

### May 5, 2025: WanGP v4.5
👋 FantasySpeaking model, you can animate a talking head using a voice track. This works not only on people but also on objects. Also better seamless transitions between Vace sliding windows for very long videos. New high quality processing features (mixed 16/32 bits calculation and 32 bits VAE)

### April 27, 2025: WanGP v4.4
👋 Phantom model support, very good model to transfer people or objects into video, works quite well at 720p and with the number of steps > 30

### April 25, 2025: WanGP v4.3
👋 Added preview mode and support for Sky Reels v2 Diffusion Forcing for high quality "infinite length videos". Note that Skyreel uses causal attention that is only supported by Sdpa attention so even if you choose another type of attention, some of the processes will use Sdpa attention.

### April 18, 2025: WanGP v4.2
👋 FLF2V model support, official support from Wan for image2video start and end frames specialized for 720p.

### April 17, 2025: WanGP v4.1
👋 Recam Master model support, view a video from a different angle. The video to process must be at least 81 frames long and you should set at least 15 steps denoising to get good results.

### April 13, 2025: WanGP v4.0
👋 Lots of goodies for you!
- A new UI, tabs were replaced by a Dropdown box to easily switch models
- A new queuing system that lets you stack in a queue as many text2video, image2video tasks, ... as you want. Each task can rely on complete different generation parameters (different number of frames, steps, loras, ...). Many thanks to **Tophness** for being a big contributor on this new feature
- Temporal upsampling (Rife) and spatial upsampling (Lanczos) for a smoother video (32 fps or 64 fps) and to enlarge your video by x2 or x4. Check these new advanced options.
- Wan Vace Control Net support: with Vace you can inject in the scene people or objects, animate a person, perform inpainting or outpainting, continue a video, ... See [VACE.md](VACE.md) for introduction guide.
- Integrated *Matanyone* tool directly inside WanGP so that you can create easily inpainting masks used in Vace
- Sliding Window generation for Vace, create windows that can last dozens of seconds
- New optimizations for old generation GPUs: Generate 5s (81 frames, 15 steps) of Vace 1.3B with only 5GB and in only 6 minutes on a RTX 2080Ti and 5s of t2v 14B in less than 10 minutes.

### March 27, 2025
👋 Added support for the new Wan Fun InP models (image2video). The 14B Fun InP has probably better end image support but unfortunately existing loras do not work so well with it. The great novelty is the Fun InP image2 1.3B model: Image 2 Video is now accessible to even lower hardware configuration. It is not as good as the 14B models but very impressive for its size. Many thanks to the VideoX-Fun team (https://github.com/aigc-apps/VideoX-Fun)

### March 26, 2025
👋 Good news! Official support for RTX 50xx please check the [installation instructions](INSTALLATION.md).

### March 24, 2025: Wan2.1GP v3.2
👋 
- Added Classifier-Free Guidance Zero Star. The video should match better the text prompt (especially with text2video) at no performance cost: many thanks to the **CFG Zero * Team**. Don't hesitate to give them a star if you appreciate the results: https://github.com/WeichenFan/CFG-Zero-star
- Added back support for PyTorch compilation with Loras. It seems it had been broken for some time
- Added possibility to keep a number of pregenerated videos in the Video Gallery (useful to compare outputs of different settings)

*You will need one more `pip install -r requirements.txt`*

### March 19, 2025: Wan2.1GP v3.1
👋 Faster launch and RAM optimizations (should require less RAM to run)

*You will need one more `pip install -r requirements.txt`*

### March 18, 2025: Wan2.1GP v3.0
👋 
- New Tab based interface, you can switch from i2v to t2v conversely without restarting the app
- Experimental Dual Frames mode for i2v, you can also specify an End frame. It doesn't always work, so you will need a few attempts.
- You can save default settings in the files *i2v_settings.json* and *t2v_settings.json* that will be used when launching the app (you can also specify the path to different settings files)
- Slight acceleration with loras

*You will need one more `pip install -r requirements.txt`*

Many thanks to *Tophness* who created the framework (and did a big part of the work) of the multitabs and saved settings features

### March 18, 2025: Wan2.1GP v2.11
👋 Added more command line parameters to prefill the generation settings + customizable output directory and choice of type of metadata for generated videos. Many thanks to *Tophness* for his contributions.

*You will need one more `pip install -r requirements.txt` to reflect new dependencies*

### March 18, 2025: Wan2.1GP v2.1
👋 More Loras!: added support for 'Safetensors' and 'Replicate' Lora formats.

*You will need to refresh the requirements with a `pip install -r requirements.txt`*

### March 17, 2025: Wan2.1GP v2.0
👋 The Lora festival continues:
- Clearer user interface
- Download 30 Loras in one click to try them all (expand the info section)
- Very easy to use Loras as now Lora presets can input the subject (or other needed terms) of the Lora so that you don't have to modify manually a prompt
- Added basic macro prompt language to prefill prompts with different values. With one prompt template, you can generate multiple prompts.
- New Multiple images prompts: you can now combine any number of images with any number of text prompts (need to launch the app with --multiple-images)
- New command lines options to launch directly the 1.3B t2v model or the 14B t2v model

### March 14, 2025: Wan2.1GP v1.7
👋 
- Lora Fest special edition: very fast loading/unload of loras for those Loras collectors around. You can also now add/remove loras in the Lora folder without restarting the app.
- Added experimental Skip Layer Guidance (advanced settings), that should improve the image quality at no extra cost. Many thanks to the *AmericanPresidentJimmyCarter* for the original implementation

*You will need to refresh the requirements `pip install -r requirements.txt`*

### March 13, 2025: Wan2.1GP v1.6
👋 Better Loras support, accelerated loading Loras.

*You will need to refresh the requirements `pip install -r requirements.txt`*

### March 10, 2025: Wan2.1GP v1.5
👋 Official Teacache support + Smart Teacache (find automatically best parameters for a requested speed multiplier), 10% speed boost with no quality loss, improved lora presets (they can now include prompts and comments to guide the user)

### March 7, 2025: Wan2.1GP v1.4
👋 Fix PyTorch compilation, now it is really 20% faster when activated

### March 4, 2025: Wan2.1GP v1.3
👋 Support for Image to Video with multiples images for different images/prompts combinations (requires *--multiple-images* switch), and added command line *--preload x* to preload in VRAM x MB of the main diffusion model if you find there is too much unused VRAM and you want to (slightly) accelerate the generation process.

*If you upgrade you will need to do a `pip install -r requirements.txt` again.*

### March 4, 2025: Wan2.1GP v1.2
👋 Implemented tiling on VAE encoding and decoding. No more VRAM peaks at the beginning and at the end

### March 3, 2025: Wan2.1GP v1.1
👋 Added Tea Cache support for faster generations: optimization of kijai's implementation (https://github.com/kijai/ComfyUI-WanVideoWrapper/) of teacache (https://github.com/ali-vilab/TeaCache)

### March 2, 2025: Wan2.1GP by DeepBeepMeep v1
👋 Brings:
- Support for all Wan including the Image to Video model
- Reduced memory consumption by 2, with possibility to generate more than 10s of video at 720p with a RTX 4090 and 10s of video at 480p with less than 12GB of VRAM. Many thanks to REFLEx (https://github.com/thu-ml/RIFLEx) for their algorithm that allows generating nice looking video longer than 5s.
- The usual perks: web interface, multiple generations, loras support, sage attention, auto download of models, ...

## Original Wan Releases

### February 25, 2025
👋 We've released the inference code and weights of Wan2.1.

### February 27, 2025
👋 Wan2.1 has been integrated into [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/wan/). Enjoy!

---

> Applies to: Historical release notes. Dates and versions describe behavior at release time; current product guides and interface contracts describe supported behavior in the installed version.
