import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class AboutPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "About Tab"
        self.version = "1.0.0"
        self.description = "Credits for the creator and all co-creators of WAN2GP"

    def setup_ui(self):
        self.add_tab(
            tab_id="about_tab",
            label="About",
            component_constructor=self.create_about_ui,
        )

    def create_about_ui(self):
        gr.Markdown("<H2>WanGP - AI Generative Models for the GPU Poor by <B>DeepBeepMeep</B> (<A HREF='https://github.com/deepbeepmeep/WanGP'>GitHub</A>)</H2>")
        gr.Markdown("""### Installation

1. **Bat/SH scripts** — use WanGP's included install, run and update scripts.
2. **[Pinokio](https://pinokio.computer/)** — use the community *wan2gp* or *wan2gp-amd* scripts by Morpheus.
3. **[Wan2GP Desktop](https://github.com/GKartist75/Wan2GP-Desktop-Tauri)** — GKArtist's desktop installer and launcher.
4. **WanGP Manager (Experimental, Windows x64)** — the official app for guided installation, launching, updates and rollback. **Download:** [WanGPManager.exe](https://github.com/deepbeepmeep/WanGP/releases/latest/download/WanGPManager.exe). This permanent link downloads the latest Manager release.

See the [installation instructions](https://github.com/deepbeepmeep/WanGP#-quick-start) for these four methods and manual setup.
""")
        gr.Markdown("Many thanks to:")
        gr.Markdown("- <B>Bria AI</B> for Fibo 2.0 Dev\\* image generation and editing, and Fibo 2.0 Edit Turbo\\* four-step editing (https://huggingface.co/briaai/Fibo-2.0-dev, https://huggingface.co/briaai/Fibo-2.0-edit-turbo)")
        gr.Markdown("- <B>Alibaba Wan Team</B> for the best open source video generators (https://github.com/Wan-Video/Wan2.1, https://github.com/Wan-Video/Wan2.2)")
        gr.Markdown("- <B>Alibaba Vace, Multitalk and Fun Teams</B> for their incredible control net models (https://github.com/ali-vilab/VACE), (https://github.com/MeiGen-AI/MultiTalk) and  (https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP) ")
        gr.Markdown("- <B>Tencent</B> for the impressive Hunyuan Video\\* models (https://github.com/Tencent-Hunyuan/HunyuanVideo, https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)")
        gr.Markdown("- <B>Black Forest Labs</B> for the innovative Flux\\* image generators, including Flux 3 Image Klein\\* and Klein Base\\* (https://github.com/black-forest-labs/flux, https://github.com/black-forest-labs/flux-3-image-klein)")
        gr.Markdown("- <B>Alibaba Qwen Team</B> for their state of the art Qwen Image\\* generators (https://github.com/QwenLM/Qwen-Image)")
        gr.Markdown("- <B>inclusionAI</B> for Ming Image 0.1 Design and Design-Layer (https://github.com/inclusionAI/Ming-Image, https://huggingface.co/inclusionAI/Ming-Image-0.1-Design, https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer)")
        gr.Markdown("- <B>OpenMOSS Team</B> for MOSS-Image Base and SFT, with text rendering and layout-guided image generation (https://github.com/OpenMOSS/MOSS-Image, https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base, https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)")
        gr.Markdown("- <B>Krea AI</B> for the Krea 2\\* RAW and Turbo image generators (https://github.com/krea-ai/krea-2, https://huggingface.co/krea/Krea-2-Raw, https://huggingface.co/krea/Krea-2-Turbo, https://www.krea.ai/blog/krea-2-technical-report)")
        gr.Markdown("- <B>Lightricks</B> for their super fast LTX Video\\* models and LTX-2\\* audio-video model (https://github.com/Lightricks/LTX-Video, https://github.com/Lightricks/LTX-2)")
        gr.Markdown("- <B>MiniMax AI</B> for the MiniMax H3\\* native audio-video generation models (https://huggingface.co/MiniMaxAI/MiniMax-H3)")
        gr.Markdown("- <B>NVIDIA Research Sol-Attn team</B> for Sol-Attn and its reference kernels, and <B>Saganaki22 (drbaph)</B> for the consumer-GPU Triton kernel improvements and MiniMax H3 integration (https://nvlabs.github.io/Sana/Sol-Attn/, https://github.com/Saganaki22/ComfyUI-sol-attn/tree/e2fc225)")
        gr.Markdown("- <B>Echo Team @ Joy Future Academy, JD</B> for JoyAI-Echo\\* (https://github.com/jd-opensource/JoyAI-Echo, https://huggingface.co/jdopensource/JoyAI-Echo)")
        gr.Markdown("- <B>Eyeline Labs</B>, <B>Netflix</B> and collaborators for Vista4D (https://github.com/Eyeline-Labs/Vista4D)")
        gr.Markdown("- <B>Alibaba PAI</B> for Wan2.1-Fun Control-Camera (https://github.com/aigc-apps/VideoX-Fun) and <B>Alibaba DAMO Academy</B> for Uni3C camera control (https://github.com/alibaba-damo-academy/Uni3C)")
        gr.Markdown("- <B>Tencent Hunyuan3D Team</B> for the Hunyuan3D 2.1\\* image-to-3D shape and texture models (https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1); license and notice in models/hunyuan3d/LICENSE and Notice.txt")
        gr.Markdown("- <B>Tencent HY-World Team</B> for WorldMirror 2.0\\* 3D reconstruction from photos or videos (https://github.com/Tencent-Hunyuan/HY-World-2.0); license and notice in models/worldmirror/LICENSE and Notice.txt")
        gr.Markdown("- <B>Tencent Hunyuan</B> for the Hy-Nomas explorable world model (https://github.com/Tencent-Hunyuan/Hy-Nomas), with <B>NVIDIA</B>'s PixelUMM\\* video decoder (https://github.com/nv-tlabs/PixelUMM, non-commercial weights); licenses and notice in models/hy_nomas/LICENSE, LICENSE_PixelUMM and NOTICE")
        gr.Markdown("- <B>Alaya Lab / The Evoke Team</B> for EVOKE-Turbo explorable worlds with a geometric memory (https://github.com/AlayaLab/Evoke), with <B>aigc3d</B>'s ViGeo1.1\\* video depth model (https://github.com/aigc3d/ViGeo, non-commercial weights); license and notice in models/evoke/LICENSE and NOTICE")
        gr.Markdown("- <B>Microsoft</B> for MoGe-2 (https://github.com/microsoft/MoGe), <B>Meta</B> for MapAnything with <B>CMU AirLab</B>'s UniCeption (https://github.com/facebookresearch/map-anything): 3D reconstruction from photos and videos")
        gr.Markdown("- <B>Microsoft</B> for TRELLIS.2 and the TRELLIS sparse-structure decoder (https://github.com/microsoft/TRELLIS.2), with Meta's DINOv3\\* image encoder")
        gr.Markdown("- <B>Hi3D</B> (OCEANO HK HOLDING LIMITED) for Hi3D v1.0\\* high-resolution mesh refinement built on Sparc3D (https://github.com/hi3d-official/Hi3Dv1.0): Powered by Hi3D 1.0 (http://hi3d.ai/); license, notice and Acceptable Use Policy in models/hi3d/LICENSE and NOTICE")
        gr.Markdown("- <B>SUPRA-Tex Authors</B> for SUPRA-Tex observation-guided PBR texture completion (https://github.com/Supra-3D/SUPRA-Tex), with its Material Adapter for Qwen Image 2.1\\* (Qwen Research License, non-commercial), <B>Tencent ARC</B> for Pixal3D pixel-aligned image-to-3D geometry (https://huggingface.co/TencentARC/Pixal3D) and <B>valeo.ai</B> for the NAF feature upsampler (https://github.com/valeoai/NAF); licenses in models/supratex")
        gr.Markdown("- <B>VAST / Tripo</B> for TripoSplat image-to-Gaussian-splats (https://github.com/VAST-AI-Research/TripoSplat), with Meta's DINOv3\\* image encoder")
        gr.Markdown("- <B>VAST-AI Research</B> and <B>Yuchen Lin</B> for TripoSG\\*, PartCrafter\\* and GeoSAM2 image-to-mesh, part generation and 3D part segmentation (https://github.com/VAST-AI-Research/TripoSG, https://github.com/wgsxm/PartCrafter); TripoSG code includes Tencent Hunyuan-derived parts, see models/triposg/Notice.txt")
        gr.Markdown("- <B>Zijie Wu, Chaohui Yu, Fan Wang and Xiang Bai</B> (HUST / Alibaba DAMO Academy) for AnimateAnyMesh text-driven mesh animation (https://github.com/JarrentWu1031/AnimateAnyMesh)")
        gr.Markdown("- <B>Resemble.AI</B> for the incredible ChatterBox (https://github.com/resemble-ai/chatterbox)")
        gr.Markdown("- <B>HeartMuLa Team</B> for the open music generation models (https://github.com/HeartMuLa/heartlib)")
        gr.Markdown("- <B>Multimodal Art Projection (MAP) Team and collaborators</B> for YuE2\\* song generation, SheetSage2\\* music transcription and the MERT2\\* music encoder (https://github.com/multimodal-art-projection/YuE, https://huggingface.co/m-a-p/SheetSage2, https://huggingface.co/m-a-p/MERT-v2-FullSong)")
        gr.Markdown("- **Mothersuperior** for the [YuE2 Hum-to-Song adapter](https://huggingface.co/Mothersuperior/YuE2-hum-to-song)\\*, hum-conditioning workflow and [real-audio acoustic LoRA](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)\\* used by this contribution.")
        gr.Markdown("- <B>ACE-Step Team</B> for the ACE-Step music generation model (https://github.com/ace-step/ACE-Step) & ACE-Step 1.5 (https://github.com/ace-step/ACE-Step-1.5)")
        gr.Markdown("- <B>Stability AI</B> for Stable Audio 3\\* (https://github.com/Stability-AI/stable-audio-3)")
        gr.Markdown("- <B>Alibaba Qwen Team</B> for Qwen 3 TTS (https://github.com/QwenLM/Qwen3-TTS)")
        gr.Markdown("- <B>Fish Audio</B> for Drama 3\\* expressive speech, dialogue and voice cloning, including the INT4 Experts variant (https://github.com/fishaudio/fish-speech, https://huggingface.co/fishaudio/drama-3)")
        gr.Markdown("- <B>Tencent Hunyuan Team</B> for the AuK audio generation and editing models (https://github.com/Tencent-Hunyuan/AuK)")
        gr.Markdown("- <B>KugelAudio</B> for Kugel Audio (https://huggingface.co/kugelaudio/kugelaudio-0-open)")
        gr.Markdown("- <B>Remade_AI</B> : for their awesome Loras collection (https://huggingface.co/Remade-AI)")
        gr.Markdown("- <B>ByteDance Bernini Team</B> for Bernini (https://github.com/bytedance/Bernini)")
        gr.Markdown("- <B>ByteDance</B> : for their great Wan and Flux extensions Lynx (https://github.com/bytedance/lynx), UMO (https://github.com/bytedance/UMO), USO (https://github.com/bytedance/USO) ")
        gr.Markdown("- <B>Hugging Face</B> for providing hosting for the models and developing must have open source libraries such as Tranformers, Diffusers, Accelerate and Gradio (https://huggingface.co/)")

        gr.Markdown("<BR>Huge acknowledgments to these great open source projects used in WanGP:")
        gr.Markdown("- <B>Rife</B>: temporal upsampler (https://github.com/hzwer/ECCV2022-RIFE)")
        gr.Markdown("- <B>DwPose</B>: Open Pose extractor (https://github.com/IDEA-Research/DWPose)")
        gr.Markdown("- <B>DepthAnything</B>\\* & <B>Midas</B>: Depth extractors (https://github.com/DepthAnything/Depth-Anything-V2) and (https://github.com/isl-org/MiDaS")
        gr.Markdown("- <B>Matanyone</B>\\*: Mask Generation (https://github.com/pq-yang/MatAnyone)")
        gr.Markdown("- <B>SAM2</B> and <B>SAM3</B>\\*: Mask Generation (https://github.com/facebookresearch/sam2, https://github.com/facebookresearch/sam3)")
        gr.Markdown("- <B>Pyannote</B>: speaker diarization (https://github.com/pyannote/pyannote-audio)")
        gr.Markdown("- <B>Sherpa-ONNX</B>: speaker diarization runtime (https://github.com/k2-fsa/sherpa-onnx)")
        gr.Markdown("- <B>Kokoro</B>: text-to-speech (https://github.com/hexgrad/kokoro)")
        gr.Markdown("- <B>OpenPhonemizer</B> (NeuralVox) and <B>DeepPhonemizer</B> (Axel Springer Ideas Engineering): English phonemizer for Kokoro's out-of-dictionary words (https://github.com/NeuralVox/OpenPhonemizer, https://github.com/as-ideas/DeepPhonemizer)")
        gr.Markdown("- <B>SeedVC</B>\\*: voice conversion (https://github.com/Plachtaa/seed-vc), downloaded on first use from its separate GPL-3.0 repository (https://github.com/deepbeepmeep/WanGP-SeedVC)")
        gr.Markdown("- <B>MMAudio</B>\\*: sound generator (https://github.com/hkchengrex/MMAudio)")
        gr.Markdown("- <B>model-viewer</B>, <B>three.js</B> and <B>Spark</B>: 3D asset and Gaussian splat viewers (https://github.com/google/model-viewer, https://github.com/mrdoob/three.js, https://github.com/sparkjsdev/spark)")
        gr.Markdown("- <B>resvg</B> (Yevhenii Reizner, linebender) and <B>resvg-py</B>: SVG rendering for Deepy's drawings (https://github.com/linebender/resvg, https://github.com/baseplate-admin/resvg-py)")
        gr.Markdown("- <B>ddgs</B> (deedy5) and <B>primp</B>: metasearch over public engines and HTTP client for Deepy's web search (https://github.com/deedy5/ddgs, https://github.com/deedy5/primp)")
        gr.Markdown("- <B>pypdf</B>: PDF text for Deepy's web reading and Python sandbox (https://github.com/py-pdf/pypdf)")
        gr.Markdown("- <B>Playwright</B>: web page screenshots for Deepy, through the installed Edge or Chrome (https://github.com/microsoft/playwright-python)")
        gr.Markdown("- <B>ByteDance Seed Team and the SeedVR2 authors</B>: one-step image and video restoration (https://github.com/ByteDance-Seed/SeedVR, https://arxiv.org/abs/2506.05301)")
        gr.Markdown("- <B>FlashVSR</B>: high quality video super-resolution (https://github.com/OpenImagingLab/FlashVSR)")
        gr.Markdown("- <B>NVIDIA PiD</B>\\*: diffusion-based image super-resolution (https://github.com/nv-tlabs/PiD)")
        gr.Markdown("- <B>Carasibana</B>: the MIT-licensed ComfyUI H3 FaceRefine workflow adapted by WanGP (https://github.com/Carasibana/ComfyUI-H3-FaceRefine)")
        gr.Markdown("- <B>Ultralytics YOLO\\* and Bingsu's ADetailer</B>: face and person detection used by H3 Face Refiner (https://github.com/ultralytics/ultralytics, https://github.com/Bing-su/adetailer)")
        gr.Markdown("- <B>InsightFace</B>\\*: face detection, landmarks, and identity embeddings used by H3 Face Refiner (https://github.com/deepinsight/insightface)")
        gr.Markdown("- <B>Merserk</B> and <B>DLSS5-Feeder</B>: foundations for the DLSS 5 video-processing integration (https://github.com/Merserk/dlss5-visual-enhancer, https://github.com/jlrouzies-fr/DLSS5-Feeder)")

        gr.Markdown("<BR>Special thanks to the following people for their Contributions & Support:")
        gr.Markdown("- <B>Tophness</B> : Designed & developped the Queuing Framework, Edit Mode, Windows Installation scripts and WanGP PlugIns System")
        gr.Markdown("- <B>GKArtist</B> : For the great Wan2GP Desktop installer (https://github.com/GKartist75/Wan2GP-Desktop-Tauri)")
        gr.Markdown("- <B>Gunther-Schulz</B> : for adding image Start Image / Image Refs storing in Video metadata")
        gr.Markdown("- <B>Cocktail Peanuts</B> : QA and simple installation via Pinokio.computer")
        gr.Markdown("- <B>huangyebiaoke</B> : for his support in porting WanGP to MPS")
        gr.Markdown("- <B>AmericanPresidentJimmyCarter</B> : added original support for Skip Layer Guidance")
        gr.Markdown("- <B>Reevoy24</B> : for his repackaging / completion of the documentation")

        gr.Markdown("<BR>\\* Restrictive license (non-commercial, territorial, usage-based or copyleft terms): refer to and comply with the corresponding license terms.")
