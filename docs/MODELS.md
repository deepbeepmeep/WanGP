# Models

WanGP brings video, image, audio and 3D creation into one workspace. This guide covers the main families and how to choose between their workflows. The model selector and each model's **Help** describe the variants, controls and settings available in your installation.

**Browse models: [Video](#video-models) / [Image](#image-models) / [Audio](#audio-models) / [3D](#3d-models)**

Model links below select a model when this guide is opened inside WanGP. When reading on GitHub, select the same name in WanGP's model selector or toolbar search.

- [Choose a starting point](#recommended-starting-points)
- [Video models](#video-models)
- [Image models](#image-models)
- [Speech, music and sound](#audio-models)
- [3D generation, reconstruction and mesh tools](#3d-models)
- [Upscaling and finishing](#upscaling-and-finishing)
- [Choose a variant for your hardware](#choosing-variants)

## Recommended starting points

| Goal | Models to try | How to choose |
| --- | --- | --- |
| Video with a synchronized soundtrack | [MiniMax H3 FL2VA Pruned](modeltype:minimax_h3_fl2va_pruned), [LTX 2.5 Distilled](modeltype:ltx2_25_22B_distilled) | H3 offers text and first/last-frame direction with stereo audio. LTX 2.5 Distilled is an eight-step starting point with keyframes, controls and continuation. |
| Short audio-video clips with a smaller model | [Kandinsky 6 Lite Distilled](modeltype:k6_lite_distill) | A 3B generator for text or a start image, with speech, music and effects. Start with five seconds and the supplied ten-step settings. |
| A planned camera move from an image | [Fun Control-Camera 1.3B](modeltype:fun_camera_1.3B), [Uni3C 14B](modeltype:uni3c) | Pick a camera preset or supply a custom path. Fun is smaller; Uni3C uses the start image's estimated geometry to guide the shot. |
| Video following reference subjects, scenes or voices | [MiniMax H3 Ref2VA Pruned](modeltype:minimax_h3_ref2va_pruned), [LTX 2.5 MSR](modeltype:ltx2_25_22B_msr) | Ref2VA accepts image, video and audio references. MSR combines subject/object references, with an optional background reference. |
| Character animation and replacement | [SCAIL-2](modeltype:scail2_14B), [Wan 2.2 Animate 2](modeltype:animate2), [Viggle Animate](modeltype:viggle_animate) | SCAIL-2 and Animate 2 are Wan-derived workflows. Viggle is H3-derived and needs an edited frame from the original driving video. |
| General text-to-video or image-to-video | [Wan 2.2 T2V](modeltype:t2v_2_2), [Wan 2.2 I2V](modeltype:i2v_2_2) | General-purpose generation with a broad choice of LoRAs and specialist variants. |
| Connected stories with recurring characters | [JoyAI-Echo Surgical](modeltype:joyai_echo_surgical) | This LTX-2 variant reuses character, voice, object and location memories between shots. |
| Video editing and outpainting | [H3 ControlNet-Union Pruned](modeltype:minimax_h3_control_pruned), [Wan VACE](modeltype:vace_14B), [Bernini-R](modeltype:bernini) | Pick H3 for controlled audio-video generation, VACE for mask/control editing, or Bernini for video edits and multiple references. |
| Talking characters and longer dialogue | [LongCat Avatar 1.5](modeltype:longcat_avatar_v1_5), [InfiniteTalk](modeltype:infinitetalk), [Magi Human Distill](modeltype:magi_human_distill) | LongCat Avatar and InfiniteTalk follow speech audio. Magi Human can generate speech with motion or follow supplied audio. |
| General image generation | [Krea 2 Turbo](modeltype:krea2_turbo), [Z-Image Turbo](modeltype:z_image), [Flux 2 Klein 4B](modeltype:flux2_klein_4b) | Distilled choices for iteration. Compare their appearance, reference support and memory use on your own prompts. |
| Image editing and references | [Qwen Image 2.1](modeltype:qwen_image_21_7B), [Qwen Image Edit Plus 2511](modeltype:qwen_image_edit_plus2_20B), [Krea 2 Identity Edit](modeltype:krea2_raw_edit) | Qwen 2.1 combines generation and editing with up to ten references. Edit Plus emphasizes image edits and identity. Krea Identity Edit accepts up to two references. |
| Posters, typography and infographics | [Ming Image Design](modeltype:ming_image_0_1_design), [Ideogram 4](modeltype:ideogram4), [SenseNova U1.5](modeltype:sensenova_u1_5_8b_mot) | Ming and Ideogram offer structured layout helpers; SenseNova supports detailed native-4K layouts. Proofread generated text. |
| Editable transparent layers | [Ming Image Design-Layer](modeltype:ming_image_0_1_design_layer), [Qwen Image Layered](modeltype:qwen_image_layered_20B) | Decompose a flattened image into separate RGBA layers for further editing. |
| Speech generation, editing and cleanup | [AuK Flash](modeltype:auk_flash), [AuK Base](modeltype:auk) | Describe the speech or the change to a source recording. Flash uses four steps; Base offers more sampling control. |
| Cloned voices and dialogue | [Qwen3 TTS Base](modeltype:qwen3_tts_base), [IndexTTS 2.5](modeltype:index_tts25), [OmniVoice](modeltype:omnivoice) | Use short, clean voice references. Compatible dialogue modes accept up to three speakers; IndexTTS adds emotion controls. |
| Songs with lyrics | [YuE2](modeltype:yue2), [MiniMax Music 3](modeltype:minimax_music3), [ACE-Step 1.5 XL](modeltype:ace_step_v1_5_xl) | YuE2 adds composition, cover and score workflows. MiniMax Music 3 generates songs from lyrics and a description. ACE-Step offers fast music generation with optional planning models. |
| A song from a hummed melody | [YuE2 Hum-to-Song](modeltype:yue2_hum) | Supply a clear hum, your lyrics and a musical style. It follows the melody, not the humming voice's identity. |
| Instrumentals, ambience and sound effects | [Stable Audio 3](modeltype:stable_audio3_small), [YuE2](modeltype:yue2) in instrumental mode | Stable Audio covers descriptive sound and music prompts; YuE2 supports structured instrumental composition. |
| Transcripts that say who spoke, subtitles or a translation | [Nemotron 3.5 Transcribe](modeltype:nemotron_transcribe), [Q4 Experts](modeltype:nemotron_transcribe_q4) | Turns a recording or a video's soundtrack into text with speaker labels, word timings and subtitles, in 40 languages. Q4 Experts needs about half the memory. |
| A textured 3D asset from one image | [TRELLIS.2 Shape](modeltype:trellis2_shape), [Pixal3D](modeltype:pixal3d) or [Hunyuan3D 2.1 Shape](modeltype:hunyuan3d_21_shape), with a Texturer preset | The shape model makes the mesh; a **preset** adds a Texturer in the same generation. SUPRA-Tex keeps the picture's colours and print exactly on the visible side. |
| PBR textures for a mesh you already have | [SUPRA-Tex Texture](modeltype:supratex_texture), [Hunyuan3D 2.1 Paint](modeltype:hunyuan3d_21_paint), [TRELLIS.2 Texture](modeltype:trellis2_texture) | Give the mesh and a picture of the same object; the same texturers run on gallery meshes from the late **Post Processing** tab. They are listed from the most to the least faithful to the picture. |
| An untextured mesh or separate object parts | [TripoSG](modeltype:triposg), [PartCrafter](modeltype:partcrafter) | TripoSG produces one detailed shape; PartCrafter generates separate part meshes from the same image. |
| A 3D scene from photos or video | [WorldMirror 2.0](modeltype:worldmirror_2), [MapAnything](modeltype:mapanything), [Depth Anything 3 Points](modeltype:da3_points) | Use overlapping views of a static scene. Choose Gaussian splats for viewing or point clouds with camera information for further work. |
| Add fine detail to a coarse mesh | [Hi3D](modeltype:hi3d) | Refines a Hunyuan3D, TripoSG or TRELLIS.2 shape into high-resolution geometry from the same picture; texture it afterwards with Hunyuan3D or SUPRA-Tex, or chain all three in post-processing. |
| Split or animate an existing mesh | [GeoSAM2](modeltype:geosam2), [AnimateAnyMesh](modeltype:animate_any_mesh) | GeoSAM2 separates parts; AnimateAnyMesh creates a short animation from a motion description without a rig. |

<a id="video-models"></a>

## Video models

### MiniMax H3

H3 creates video with native stereo audio. The main variants are available as full **33B** and smaller **Pruned 20B** models:

- **[FL2VA](modeltype:minimax_h3_fl2va_pruned)**: generate from text, a start image, an end image or both. Use injected frames for visual milestones, or sliding windows to continue a longer shot. It can follow an uploaded soundtrack or a control video's audio.
- **[Ref2VA](modeltype:minimax_h3_ref2va_pruned)**: guide appearance, subjects, motion or voices with reference images, videos and audio. It supports up to three reference videos and three audio references. Video references share a 15-second budget; audio references have a separate 15-second budget. Reference excerpts let you select useful moments from a longer clip. Audio choices marked **Soundtrack Kept** preserve the supplied soundtrack.
- **[FL2VA ControlNet-Union](modeltype:minimax_h3_control_pruned)**: follow pose, depth, edges, shapes or recoloring controls, or work with inpainting and spatial outpainting. Select this variant when those explicit controls matter to the shot.

Both main workflows support continuation, reference or frame guidance as appropriate, and still-image generation. For straightforward text, start/end or injected-frame direction, start with FL2VA; choose Ref2VA when reusing reference content is central.

H3's **Two Phases** option generates at a lower resolution before refining at the target size. This can accelerate high-resolution work, but does not by itself lower peak VRAM. **Two Phases with Tiling** lowers the high-resolution phase's memory requirement, with a possible trade-off in tile seams.

Acceleration choices have different trade-offs. **PDD** variants use a fixed eight-step Euler workflow. **VDN** variants accelerate longer video and work best with the required Triton support. Spectrum, First Block Cache and supported sparse attention settings offer other speed/quality choices; start with the selected variant's defaults before combining them.

**Veda Sparse** adds learned sparse attention to **[FL2VA 33B](modeltype:minimax_h3_fl2va_veda) / [Pruned 20B](modeltype:minimax_h3_fl2va_pruned_veda)** and **[Ref2VA 33B](modeltype:minimax_h3_ref2va_veda) / [Pruned 20B](modeltype:minimax_h3_ref2va_pruned_veda)**. It reuses the corresponding standard model's settings, LoRAs and step counts, with the largest speed benefits on longer or higher-resolution videos. It requires Triton and an NVIDIA GPU with compute capability 8.0 or newer. Sparse attention can change fine detail or motion, so compare results with the standard variant. Veda uses its own attention mode and runs without CUDA Graphs or transformer compilation.

**[Viggle Animate](modeltype:viggle_animate) is an H3-derived character-replacement model.** Supply the original control video and one frame from it in which an image editor has replaced the character. Preserve that frame's pose, background, framing and dimensions. Viggle follows the original motion in a fixed three-step workflow, using sliding windows for longer footage. Its text generation prompt is ignored: express the change in the edited frame. Large pose changes, interacting characters and cuts are more challenging.

The family also provides an audio-only **[H3 Voice Clone](modeltype:minimax_h3_tts_ref2va_pruned)** preset and **H3 Face Refiner** for finishing video; these are related H3 workflows, not separate model families.

<a id="ltx-2-and-ltx-23"></a>

### LTX-2, LTX 2.3 and LTX 2.5

LTX-2 generates video and sound together. **[LTX 2.5 Distilled](modeltype:ltx2_25_22B_distilled)** is the current fast eight-step starting point; **[LTX 2.5 Dev](modeltype:ltx2_25_22B)** gives more control over generation. LTX 2.0 and 2.3 variants remain available for existing workflows and compatible LoRAs.

Depending on the selected variant and process, the family supports:

- start/end images, injected frames, control video and soundtrack guidance;
- continuation and sliding windows for longer sequences;
- pose, depth, edge, inpainting, outpainting and HDR workflows;
- reference-voice conditioning and talking-character workflows;
- multi-phase generation and video upscaling.

LTX 2.5 uses **Ingredients** when reference images are supplied in its compatible workflow. Choose **[LTX 2.5 MSR](modeltype:ltx2_25_22B_msr)** for one to four subject/object references, or two to five references with the background first. **[LTX 2.3 MSR V2](modeltype:ltx2_22B_msr_v2)** remains another reference-based option. Describe what each reference contributes.

**[JoyAI-Echo](modeltype:joyai_echo)** and **[JoyAI-Echo Surgical](modeltype:joyai_echo_surgical)** are LTX-2 variants for connected shots with reusable memories. They are useful when the same people, voices or locations must recur.

LTX 2.5 also offers finishing and visual-effects workflows:

- **[Layout to Render](modeltype:ltx2_25_22B_layout_to_render)** turns a viewport animation or playblast into a finished shot. Supply the layout video and an appearance reference made from its first frame; the layout guides placement and camera movement, while the image guides the look.
- **[Alpha Gen](modeltype:ltx2_25_22B_alpha_gen)** extracts a soft alpha matte from a source video, optionally restricted by a mask. Save transparent footage as PNG frames in a ZIP or ProRes 4444 using **Config → Outputs → RGBA Video Output**.
- **Convert SDR to HDR** uses the compatible LTX 2.5 workflow to produce HDR10 video from a control video. The conversion follows the footage rather than the text prompt.

Decoder choices affect both memory and appearance. The optional **NAD** decoder for LTX 2.3/2.5 trades additional time and VRAM for a different quality profile. Use the original decoder as the initial baseline, and choose only decoder/LoRA combinations offered by the selected model. LTX-based upscaling and detail refinement are also available in postprocessing and Media Flow.

### Wan 2.1 and Wan 2.2

The Wan family covers general generation and many specialist animation, editing and control workflows:

- **[Wan 2.2 T2V](modeltype:t2v_2_2)** and **[Wan 2.2 I2V](modeltype:i2v_2_2)**: general text-to-video and image-to-video.
- **[Wan 2.2 TI2V 5B](modeltype:ti2v_2_2)**: a smaller unified text/image-to-video option.
- **[Wan 2.2 Animate 2](modeltype:animate2)** and its **[Distilled variant](modeltype:animate2_distilled)**: transfer body motion, expressions and camera movement from a driving video to a reference character. The original Animate remains available.
- **[SCAIL-2](modeltype:scail2_14B)**: a Wan 2.1-derived character animator with reference, pose and mask guidance. Animate mode can direct multiple characters; replacement mode changes a person in an existing video. Use aligned references, and the model's mask guidance for multi-character scenes. Sliding windows support longer performances.
- **[VACE](modeltype:vace_14B)**: masked edits, outpainting, object replacement and pose/depth/motion control.
- **[Bernini-R](modeltype:bernini)**: video-to-video generation and multiple reference images; a smaller **[1.3B variant](modeltype:bernini_1.3B)** is also available.
- **[Lynx](modeltype:lynx)** and **[VACE Lynx](modeltype:vace_lynx_14B)**: identity-guided face replacement.
- **[MultiTalk](modeltype:multitalk)** and **[InfiniteTalk](modeltype:infinitetalk)**: audio-driven talking characters and dialogue.
- **[Vista4D](modeltype:vista4d)**: reshoot a dynamic scene along a new camera trajectory.
- **[Fun Control-Camera 1.3B](modeltype:fun_camera_1.3B)** and **[Uni3C 14B](modeltype:uni3c)**: animate a start image along a camera preset or custom path. Fun also accepts text alone; Uni3C requires a start image and uses its estimated geometry. Begin with a modest move: hidden areas must be invented, and neither workflow supports sliding windows.
- **[Prism Video / Prism Alpha](modeltype:prism)**: Wan 2.2-based image-to-video with jointly generated audio. Supply a start image and describe the motion and sound; an optional separate audio prompt can direct the soundtrack. Use the supplied defaults first. This preview generates each clip in one pass, without end images, supplied audio or continuation.

Names such as FusioniX, FastWan, Lightning and Self-Forcing identify particular variants or acceleration workflows. NVFP4 names identify a weight format. Check the underlying model before choosing compatible LoRAs or assuming that two variants expose the same controls.

### Kandinsky 6

**[Kandinsky 6 Pro Distilled 30B](modeltype:k6_pro_distill)** and **[Lite Distilled 3B](modeltype:k6_lite_distill)** generate video and a synchronized mono soundtrack from text or a start image. Pro offers greater capacity for detailed prompts; Lite is the smaller option. Both use ten distilled steps, without guidance or a negative prompt.

Start with **five seconds at 832×480 and 24 fps**. Longer clips, up to ten seconds, are possible but can lose consistency. Describe the scene, action, camera and sound; the optional Prompt Enhancer can help with Kandinsky's speech and audio tags. There is no end-image, supplied-audio or continuation workflow. Kandinsky 5 remains available separately.

### Other video families

These are independent families, rather than variants of each other:

| Family | Main uses and choices |
| --- | --- |
| **HunyuanVideo** | [HunyuanVideo 1.5](modeltype:hunyuan_1_5_t2v) offers text-to-video and image-to-video, with distilled and upsampler variants. Original HunyuanVideo, Custom and Avatar workflows remain available. |
| **LongCat** | [LongCat Video](modeltype:longcat_video) handles general generation; [LongCat Avatar 1.5](modeltype:longcat_avatar_v1_5) is an audio-driven avatar variant. |
| **Kandinsky 5** | [Pro](modeltype:k5_pro_t2v) offers text/image-to-video with camera direction; [Lite](modeltype:k5_lite_t2v) and distilled variants provide smaller or faster choices. |
| **Magi Human** | [Base](modeltype:magi_human) and [Distill](modeltype:magi_human_distill) animate a start image with generated speech or supplied audio. Use the offered staged upsampler variants for higher-resolution results. |
| **Ovi** | [Ovi 1.1](modeltype:ovi_1_1) generates video with synchronized audio, including speaking characters. FastWan and longer-duration variants are available. |
| **LTX-Video** | [LTX-Video 0.9.8](modeltype:ltxv_13B) and its distilled variant retain the earlier LTX video workflow; they are distinct from the LTX-2 audio-video generation described above. |

Other specialist entries remain in the selector. Use their own Help for required inputs and limitations rather than transferring settings between unrelated models.

<a id="image-models"></a>

## Image models

<a id="qwen-image"></a>

### Qwen Image and Qwen Image 2.1

**[Qwen Image 2.1 7B](modeltype:qwen_image_21_7B)** combines image generation and editing. It accepts up to ten references and supports masked editing, LanPaint, outpainting, pose/depth/edge guidance and resolutions up to 4K. Enable **RGBA** and request a transparent background to save cutouts as PNG with an alpha channel.

Its optional **KV Cache** trades extra GPU memory for faster denoising; its **INT8** choice gives the same speedup in about half that memory. **Viggle Turbo** and **Pruna** acceleration profiles apply the matching adapter, scheduler and step settings; these image accelerators are unrelated to the H3-based Viggle Animate video workflow. Prefer short text in Qwen 2.1 images and proofread it; use a design-oriented model for dense infographics.

The older 20B family remains useful:

- **[Qwen Image 2512](modeltype:qwen_image_2512_20B)** for generation and text-heavy images;
- **[Edit Plus 2511](modeltype:qwen_image_edit_plus2_20B)** for reference-based editing and identity preservation, alongside original Edit and Edit Plus 2509;
- **[Qwen Image Layered](modeltype:qwen_image_layered_20B)** for decomposing an image into editable RGBA layers.

Selected 20B variants have Nunchaku checkpoints for supported hardware. Qwen 2.1 uses its own model and adapter formats: do not assume older Qwen Image LoRAs or VAEs are interchangeable.

### Ming Image

**[Ming Image 0.1 Design](modeltype:ming_image_0_1_design)** generates posters, infographics and arranged layouts, or edits one reference image. Use a normal prompt or the structured JSON format for more precise layout direction. The visual **Prompt Helper** lets you position layers and edit their descriptions, colors and text. Optional RGBA output supports transparent designs.

**[Ming Image Design-Layer](modeltype:ming_image_0_1_design_layer)** separates one flattened design into transparent layers. Supply the image and describe the layers from front to back. WanGP adds the RGBA layers to the gallery and saves the full set as a ZIP. Use this when you need individual elements for later compositing.

### SenseNova U1.5

**[SenseNova U1.5 8B MoT](modeltype:sensenova_u1_5_8b_mot)** generates and edits images, with native-4K output, detailed composition and text-heavy layouts. Its **Infographic Prompt** enhancer helps turn a short brief into a structured design. An eight-step acceleration profile and optional KV Cache offer speed/memory trade-offs.

### Krea 2

**[Krea 2 RAW](modeltype:krea2_raw)** is the undistilled option with guidance control; **[Krea 2 Turbo](modeltype:krea2_turbo)** is distilled for faster generation. **Identity Edit** variants add instruction-based edits and up to two reference images. WanGP also supports LanPaint inpainting and negative prompting through NAG on supported distilled workflows.

Krea 2 and compatible Qwen Image 20B/Edit models offer a **VAE** dropdown: the default keeps the established appearance, **Krea 2 Real** offers an alternative for photographic texture, and **Krea 2 HD** offers another for detail and contrast. Compare color and detail on your own images; alternatives can use more memory. Selecting the Spacepxl VAE upsampler overrides that choice. These alternatives do not apply to Qwen 2.1, Qwen Layered or Ming Image.

<a id="flux-z-image-and-hidream"></a>

### Flux 1 and Flux 2

Flux is a separate family from Krea 2, Z-Image and HiDream:

- **[Flux 1 Schnell](modeltype:flux_schnell)** and **[Dev](modeltype:flux)** cover general generation; Kontext, DreamOmni2, Krea, Chroma, SRPO and other variants specialize the workflow or appearance. **Flux 1 Krea is a Flux variant, not Krea 2.**
- **[Flux 2 Dev](modeltype:flux2_dev)** and **[Klein 4B](modeltype:flux2_klein_4b) / [9B](modeltype:flux2_klein_9b)** provide generation and reference-based workflows. Base and distilled Klein variants expose different speed/control choices; supported NVFP4 variants target compatible hardware.

Choose LoRAs and reference workflows for the exact Flux architecture and variant.

### Z-Image

**[Z-Image Turbo](modeltype:z_image)** is a distilled 6B generator. **Base**, **Fun ControlNet**, **TwinFlow** and supported **Nunchaku** variants offer different control, sampling and memory choices. Use a ControlNet variant when you need its particular pose/depth/edge or editing workflow, rather than expecting those controls on every Z-Image entry.

### Ideogram 4

**[Ideogram 4](modeltype:ideogram4)** focuses on typography, layout and structured graphic design. Plain prompts work; its Magic Prompt and visual helper assist with the model's JSON layout format. **TurboTime** and **NF4** variants offer different speed or memory trade-offs.

### HiDream

**[HiDream O1 Dev 2604](modeltype:hidream_o1_dev_2604)** supports text-to-image and reference-guided creation, with control-image and preview support. Full and other Dev checkpoints are also available. HiDream is independent of Ideogram and Flux; use its own model help and compatible settings.

<a id="audio-models"></a>

## Speech, music, and sound

### Speech and dialogue

| Family or derived workflow | When to use it |
| --- | --- |
| **[AuK](modeltype:auk) / [AuK Flash](modeltype:auk_flash)** | Generate speech from written instructions, clone a voice, edit spoken content, clean up a recording or separate speakers. Supply source audio for tasks that transform an existing recording, and describe what must stay unchanged. Flash is the fixed four-step option; Base allows more sampling control. |
| **[Qwen3 TTS](modeltype:qwen3_tts_base)** | Base clones reference voices; Custom Voice uses its offered voices; Voice Design creates a voice from a description. Base supports one to three voice references for dialogue. |
| **[IndexTTS 2](modeltype:index_tts2) / [2.5](modeltype:index_tts25)** | Voice cloning and expressive dialogue with text- or audio-guided emotion. Version 2.5 adds speech-speed control and pronunciation annotations, with support for Chinese, English, Japanese, Spanish and Arabic. |
| **[OmniVoice](modeltype:omnivoice)** | Multilingual speech, voice design, cloning and dialogue, with control over speaking speed. |
| **[KugelAudio](modeltype:kugelaudio_0_open)** | An alternative for generated speech and reference-voice dialogue. |
| **[Chatterbox](modeltype:chatterbox)** | Another reference-based speech synthesis workflow. |
| **[MiniMax H3 Voice Clone](modeltype:minimax_h3_tts_ref2va_pruned)** | H3's audio-only preset for cloned speech and general audio; it reuses H3 model files and saves stereo audio rather than video. |
| **[DramaBox](modeltype:dramabox_audio) / [Scenema Audio](modeltype:scenema_audio)** | LTX-2-derived expressive speech and dialogue workflows. Scenema applies voice references through SeedVC. |

For supported three-voice modes in **H3 Audio, Qwen3 TTS Base, OmniVoice, KugelAudio, IndexTTS 2/2.5, DramaBox and Scenema**, upload the references in speaker order and write `Speaker 1:`, `Speaker 2:` and `Speaker 3:` lines. Select the corresponding reference mode first. Do not assume that every speech model or every voice mode supports the same number of speakers.

### Music and sound

**[YuE2](modeltype:yue2)** creates songs with vocals and accompaniment from lyrics and a musical description. Its workflows include:

- composing a new song or choosing instrumental mode;
- transcribing melody and chords from source audio to guide a new cover, with lyrics supplied separately;
- importing or extending a compatible ABC score, including a score transcribed from audio;
- exporting the composition as ABC and MIDI when enabled;
- using compatible composition-stage and audio-generation LoRAs.

**[YuE2 Hum-to-Song](modeltype:yue2_hum)** turns a clear hummed melody, lyrics and a style into a new song. Choose whether to use the transcribed melody alone or continue it. Neither covers nor hum-to-song preserve the source recording or clone its singer. The duration setting is an upper limit; a result can end earlier or be cut off if the limit is too short.

**[MiniMax Music 3](modeltype:minimax_music3)** generates complete songs from lyrics and a music description, with stereo output and tracks up to five minutes. Its prompt enhancers can help write lyrics or the musical brief. Actual duration and quality depend on the prompt and settings.

**[ACE-Step 1.5 Turbo](modeltype:ace_step_v1_5)** and **[ACE-Step 1.5 XL Turbo](modeltype:ace_step_v1_5_xl)** provide fast music workflows; ACE-Step 1.0 remains available. The optional 0.6B, 1.7B or 4B planning-model variants add a stage that can improve structure and lyric matching at an extra time and memory cost.

**[HeartMuLa](modeltype:heartmula_oss_3b)** is another lyrics-driven song family, with base and updated checkpoints.

**[Stable Audio 3](modeltype:stable_audio3_small)** generates instrumentals, loops, ambience and sound effects. Small, Medium and a dedicated Small SFX variant serve different speed, size and sound-design needs.

FLAC import and 16-bit lossless export let you keep audio quality and embedded generation settings. WAV/MP3 and other offered output choices remain available.

### Transcription

**[Nemotron 3.5 Transcribe](modeltype:nemotron_transcribe)** turns an audio file or a video's soundtrack into text, offline, in 40 languages and for recordings of up to an hour. It labels who said what (up to eight speakers, which **Speaker Names** can name), times every word, writes subtitles, and translates to or from English. Add the names and jargon that the recording contains as **Context Terms** so that they are spelled correctly, and choose **Verbatim** to keep hesitations and fillers or **Readable** for cleaner text. The transcript appears under the player of the Audio gallery, as text or as a **Timeline** with one lane per speaker: clicking a line or a turn plays the recording from there, and the text can be copied or downloaded. It is also saved in a `.txt` file beside the audio, one line per turn with its time span and speaker, together with `.srt` subtitles. A video's soundtrack is read from the **Video to Transcribe** input that appears when **Video File (Its Soundtrack)** is chosen; its **Output** can then be **Video with Subtitles**: the video itself, unchanged, with the transcript or the translation as subtitles, shown in the Video gallery where the **CC** button switches them on and off. With **Timestamps and Subtitles**, which can also time every word (**Per Word**), a translation is timed too: each sentence is translated from its own stretch of audio, which takes longer than a single translation of the whole recording. Speaker numbers distinguish the voices of one recording; they do not identify people. The **[Q4 Experts](modeltype:nemotron_transcribe_q4)** variant needs about half the memory and suits computers with 32 GB of RAM, at a possible small loss of accuracy.

<a id="3d-models"></a>

## 3D models

WanGP supports creating objects from images, reconstructing scenes from photos or video, and working on existing meshes. Choose the output for your next step: a **mesh** has surfaces you can edit or texture; **Gaussian splats** render a scene from coloured 3D points and need a compatible viewer; a **point cloud** records sampled geometry and colour.

The model selector starts with what you want to do: **Create** a 3D asset from an image, **Texture**, **Refine**, **Split** or **Animate** a mesh, or **Reconstruct** a scene from photos or video. The second list then offers the models for that goal.

### Create an object from one image

| Model | Output and when to use it |
| --- | --- |
| **[TRELLIS.2 Shape](modeltype:trellis2_shape)** | An untextured GLB mesh that keeps thin and intricate structures. Start with **1024 Cascade** for a balance of shape and detail; 512 is faster, while 1536 Cascade needs more time and memory. Its presets add a Texturer in the same generation: *SUPRA-Tex PBR*, *Hunyuan3D 2.1 Paint PBR* or *TRELLIS.2 Texture PBR* (the former textured TRELLIS.2 model: the fastest, with approximate colours). |
| **[Hunyuan3D 2.1 Shape](modeltype:hunyuan3d_21_shape)** | An untextured GLB mesh. Its preset **Hunyuan3D 2.1 Paint PBR** adds Hunyuan3D's PBR textures in the same generation (the former textured Hunyuan3D 2.1 model); [Hunyuan3D 2.1 Paint](modeltype:hunyuan3d_21_paint) paints a mesh you supply. Texturing replaces the mesh's existing UVs and materials. |
| **[TripoSG](modeltype:triposg)** | A detailed, untextured GLB mesh. Useful when you want to edit the shape or texture it separately. Higher mesh resolution preserves finer detail but increases extraction time and face count. |
| **[PartCrafter](modeltype:partcrafter)** | A GLB containing separate, untextured part meshes generated together. Choose the number of parts to suit the object. Parts can be moved or edited individually, but their order and colours are arbitrary and small parts may overlap or be missing. |
| **[Pixal3D](modeltype:pixal3d)** + SUPRA-Tex | [Pixal3D](modeltype:pixal3d) builds the mesh in the picture's own camera, so it fits the image exactly; its preset **SUPRA-Tex PBR** adds PBR textures that keep the picture's colours, print and patterns on the side the picture shows, the hidden side completed to match (the former SUPRA-Tex model). [SUPRA-Tex Texture](modeltype:supratex_texture) textures a mesh you supply. SUPRA-Tex reuses Qwen Image 2.1 to read the picture's materials, which is for **non-commercial use only** (Qwen Research License). |
| **[TripoSplat](modeltype:triposplat)** | A Gaussian-splat PLY for viewing objects with fine, fuzzy or thin details. It produces splats rather than an editable mesh, so choose another model for mesh texturing or rigging. |

Use one whole, uncropped object on a plain or transparent background, preferably in a three-quarter view with even lighting. The background of a picture without transparency is removed automatically, cast shadows included (the 3D models use BiRefNet, a one-time download of about 1 GB). Hidden sides are inferred and may differ from the real object. For these models, **Asset Name / Description** names the output; the reference image determines its appearance. To start from text, generate an image first, then use it as the 3D reference.

### Reconstruct a scene from photos or video

| Model | Inputs and result |
| --- | --- |
| **[WorldMirror 2.0](modeltype:worldmirror_2)** | Photos, a reference video, or both, producing Gaussian splats or a point cloud with camera information. Use overlapping views around a static scene. Areas no view shows remain empty; a single image gives only a depth-based view of the visible surfaces. |
| **[MapAnything](modeltype:mapanything)** | Photos and/or video frames, producing a merged point cloud with estimated camera poses and metric scale. Useful for reconstructing a scene from multiple viewpoints. |
| **[Depth Anything 3 Points](modeltype:da3_points)** | Photos and/or video frames, producing a point cloud and cameras at relative scale. A lighter depth-based reconstruction choice when metric scale is unnecessary. |
| **[MoGe-2](modeltype:moge2)** | One photo, producing a metric point cloud or a textured GLB of its visible surface. It creates a depth relief, not a complete object with reconstructed hidden sides. |

Sharp, overlapping views with a moving camera and a static scene work best. Moving subjects, reflections and textureless surfaces are harder to reconstruct. More views and higher working resolution increase memory use and processing time. The text field names the scene; it does not alter the reconstruction.

### Refine, split and animate existing meshes

- **[Hi3D](modeltype:hi3d)** refines a coarse mesh into high-resolution geometry from a picture of the same object: seams, buttons, scales and engravings that image-to-3D models only suggest. Use the **To Input Mesh** button of a Hunyuan3D Shape, TripoSG or TRELLIS.2 result together with its picture. **1024** gives a strong improvement in a few minutes on a fast GPU; **512** is quicker and **1536** adds the finest relief for about twice the time of 1024. The result keeps the input's position and scale but has no textures: texture it afterwards with [SUPRA-Tex Texture](modeltype:supratex_texture), [Hunyuan3D 2.1 Paint](modeltype:hunyuan3d_21_paint) or [TRELLIS.2 Texture](modeltype:trellis2_texture), or select a Texturer in Hi3D's own Post Processing tab, keeping enough faces that the detail survives. Hi3D keeps the overall shape, so fix missing or merged parts in the coarse mesh first. Its detail goes to solid parts: thin sheets such as leaves, fabric or paper are finer than its grid, so where it would perforate or lose them it keeps the coarse mesh's surface there, without added relief.

#### 3D post-processing: refine and texture in one go

The **Post Processing** tab of untextured mesh models (Hunyuan3D Shape, TripoSG, TRELLIS.2 Shape, Pixal3D, Hi3D) offers two 3D steps, run on each generated mesh with the generation's object image. Their **presets** choose ready-made combinations, for example *SUPRA-Tex PBR* or *Hi3D + SUPRA-Tex PBR*:

- **3D Refiner** (Hi3D): adds fine geometric detail first.
- **Texturer**: then paints PBR textures. From the most to the least faithful to the picture:
  1. **SUPRA-Tex** keeps the picture's colours, print and patterns on the visible side and completes the hidden side to match. The picture must line up with the mesh: the front or a three-quarter view, at most moderately from above (a steep top-down photo misplaces the colours). Its material stage is for non-commercial use only.
  2. **Hunyuan3D 2.1 Paint** paints all around from generated views: colours follow the picture, fine print comes out blurred. The safe choice when the picture and the mesh differ.
  3. **TRELLIS.2 Texture** is several times faster than the others: plausible materials whose colours only approximate the picture and change with the seed, without its print or patterns.

Combining them gives a shape, its fine relief and its textures in one generation. A model that already textures its meshes cannot take a Texturer, and a textured mesh cannot be refined (refining replaces the geometry and drops its textures). In late post-processing, select a mesh in the gallery and add its **Object Image**: one field, used by both steps. With **Embed Source Images** enabled in Configuration, new 3D assets keep their object image inside the GLB (about the size of the picture): late post-processing uses it when you add none, and loading the asset's settings restores it. SUPRA-Tex places the mesh from the picture's silhouette, so the picture must show the same object; regions the picture does not show are completed and can come out plainer than the visible side.

- **[GeoSAM2](modeltype:geosam2)** splits an input mesh into separate parts within one GLB. Use automatic segmentation or optional point prompts to guide it. Existing textures are retained; small or hidden parts can be merged.
- **[AnimateAnyMesh](modeltype:animate_any_mesh)** animates an input mesh from an English motion description, such as `The dragon flaps its wings and flies`. It needs no skeleton or rig and retains the mesh's textures. Motion is limited to 16 generated frames, with an optional forward-and-backward loop. It deforms the existing shape, so it cannot split a closed mesh into new pieces. Start with an upright mesh and try a few seeds.

### Reuse 3D assets

Open results in the **3D gallery** to rotate, zoom and inspect them. Save a view at your chosen resolution, then use that image as a reference or the start of a video. Compatible mesh results can also become inputs for texturing, refinement, segmentation or animation: select one in the 3D gallery and press **To Input Mesh**, shown for models that take a mesh (those under Texture, Refine, Split and Animate). [Deepy](DEEPY.md) can inspect 3D assets from different angles and help turn a character into a turnaround sheet.

## Upscaling and finishing

Generation models and finishing tools are separate choices. Select a processor in **Post Processing**, use **Late Post Processing** on a gallery result, or use **Media Flow** for supported batch workflows.

- **Spatial upscaling and detail**: SeedVR2, FlashVSR and LTX-based upscaling offer different appearance and memory trade-offs. LTX 2.5 Detail Refiner can restore texture in soft or compressed video. Lanczos provides conventional resizing; latent/VAE upsamplers apply only to compatible generation workflows.
- **Smoother motion**: RIFE supports x2/x3/x4 interpolation. Optional DLSS Frame Generation offers hardware-dependent multipliers on compatible RTX GPUs.
- **Neural refinement**: optional DLSS 5 can refine lighting, material appearance and detail at native or enlarged resolution, on supported hardware. See the [DLSS 5 guide](DLSS5.md) for setup and limits.
- **Faces**: H3 Face Refiner can refine up to five detected, tracked faces in a video.
- **Audio**: remove vocals, replace voices with SeedVC, add a soundtrack with MMAudio or PrismAudio, or remux existing tracks. Deepy Prime can also mix audio and package selectable audio/subtitle tracks.

See [Processing](PROCESSING.md) for input preparation, masks, control media and finishing workflows.

## Choosing variants

Model size alone does not determine whether a job will fit. Resolution, duration, references, batch size, quantization, attention backend, memory profile and tiling all affect RAM and VRAM. Select models can run with about **6 GB of VRAM**, but that is not a promise for every model or every setting.

For a new workflow:

1. Select the variant that matches your inputs and intended result. A reference editor, an animator and a text-to-video model are different workflows even within one family.
2. Start with the supplied defaults or an offered acceleration profile. A fixed-step distilled or PDD model should keep its required scheduler and step count.
3. Choose a memory profile suited to your GPU and available system RAM. WanGP's memory optimizations reduce the cost, but larger jobs still need more resources.
4. Test a short clip or moderate image resolution before increasing duration, batch size or references.
5. Use sliding windows for supported long-video workflows, and tiling where the model exposes it. Look for seams or continuity changes when tuning these options.
6. Add compatible LoRAs, controls and upsampling one feature at a time so their effects are easy to compare.

INT8, FP8, GGUF, NVFP4, NF4 and Nunchaku are checkpoint/quantization choices, not separate creative model families. Availability, memory savings, speed and output can vary with the model, GPU and installed kernels. Follow the [installation guide](INSTALLATION.md) or [AMD guide](AMD-INSTALLATION.md) for hardware-specific setup.

## Model switching and custom models

WanGP downloads models on demand. Optional preprocessing and postprocessing files also download only when a workflow needs them. Switching models does not require a restart; the previous model is unloaded as needed to recover memory. A model normally leaves the VRAM after each generation. A few models that run as a single piece for their whole generation (Nemotron 3.5 Transcribe, HiDream O1, SenseNova U1, KugelAudio, the 3D reconstruction models and GeoSAM2) stay loaded instead when the next sample or the next queued task uses the same model, so a batch of short jobs does not reload them each time. With the *MMGP Optimized VRAM Allocator* (the default), other models and programs still get that VRAM back as soon as they need it; with PyTorch's allocator, these models are unloaded after each generation as usual.

Use the model selector or toolbar search, save reusable settings, and organize media in [Workspaces](WORKSPACES.md). The refreshed browser interface lets you check progress, inspect galleries and add generation jobs from another device connected to the same running WanGP server. The queue and galleries stay synchronized; each Gradio page keeps its own unsent prompts and draft settings. See [Authentication and network access](AUTHENTICATION.md) for remote setup.

[Deepy](DEEPY.md) can help select models and connect generation/editing steps. Its language-model engine is a separate choice from the image, video, audio or 3D model doing the generation; the phone-friendly Deepy Web App controls that same local workspace.

User-provided checkpoints belong in `finetunes/`, while model Plugins can add other families. Refresh the model list after adding definitions. See [Finetunes](FINETUNES.md), [LoRAs](LORAS.md) and [Plugins](PLUGINS.md) for compatibility and setup.

---

> Applies to: Choosing WanGP video, image, speech, music, sound, transcription and 3D models; model-family and variant relationships; compatible references and workflows; image-to-3D generation, scene reconstruction, mesh segmentation and animation; upscaling and finishing choices; hardware-dependent memory and acceleration trade-offs. The model selector and model Help provide the current available variants and settings.
