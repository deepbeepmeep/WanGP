# WanGP

-----
<p align="center">
<b>WanGP by DeepBeepMeep : The best Open Source Video Generative Models Accessible to the GPU Poor</b>
</p>

WanGP supports the Wan (and derived models) but also Hunyuan Video, Flux, Qwen, Z-Image, LongCat, Kandinsky, LTXV, LTX-2, Qwen3 TTS, Chatterbox, HearMula, ... with:
- Low VRAM requirements (as low as 6 GB of VRAM is sufficient for certain models)
- Support for old Nvidia GPUs (RTX 10XX, 20xx, ...)
- Support for AMD GPUs (RDNA 4, 3, 3.5, and 2), instructions in the Installation Section Below.
- Very Fast on the latest GPUs
- Easy to use Full Web based interface
- Support for many checkpoint Quantized formats: int8, fp8, gguf, NV FP4, Nunchaku
- Auto download of the required model adapted to your specific architecture
- Tools integrated to facilitate Video Generation : Mask Editor, Prompt Enhancer, Temporal and Spatial Generation, MMAudio, Video Browser, Pose / Depth / Flow extractor, Motion Designer
- Plenty of ready to use Plug Ins: Gallery Browser, Upscaler, Models/Checkpoints Manager, CivitAI browser and downloader, ...
- Loras Support to customize each model
- Queuing system : make your shopping list of videos to generate and come back later
- Headless mode: launch the generation of multiple image / videos / audio files using a command line

**Discord Server to get Help from the WanGP Community and show your Best Gens:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)


## 🔥 Latest Updates : 
### 2nd of May 2026: WanGP v11.52, a Kind of Magic

- **Vista 4D**: Vista4D allows a *Video Reshooting* of a *Dynamic scene* from novel camera trajectories and viewpoints. In other words this Wan 2.1 model will let you relive from a different (moving) perspective a scene with moving people or objects. The sequences are quite short (usually 49 frames, max around 97 frames) but it is a lot of fun as for once this really works. 

In real life, there is no chance you should have been able to run this model (it requires x3 the amount of VRAM than what is usally required for equivalent output res and the preprocessing needs 24 GB of VRAM to build a 4D map). But once again thanks to WanGP magic VRAM requirements have been reduced to 10 GB of VRAM or less.

It is highly recommended to apply the *Lightx2v 4 steps* lora profile. Also for best efficiency, you must list all the dynamic objects / people in the *Dynamic object keywords* input.

- **Magic Mask**: generating a *Video Mask* or *Image Mask* has never been easier and faster. No need to get into the *Video Mask Generator* tab, just click the *Magic Wand* next to *Mask field* and enter a few keywords like *blue car* or *lady to the right* and a high quality mask powered by *SAM3* will be generated automatically. You will appreciate the very good *Temporal Consistency* brought by SAM3.

- **Video Mask Generator with SAM3 support**: if you still need to generate complex masks you can combine the good old point and click masks with the SAM3 / Magic Mask masks. You need to enable this feature in the *Config / Extensions* tab.

- **LTX-2 Video to Audio**: it was more or less already possible but this new Control Video Process will be much faster and the output video will be unaltered

*update 11.51*: various fixes\
*update 11.52*: LTX-Video to Audio, fixed bugs in audio continuation with sliding windows 

### 25th of April 2026: WanGP v11.41, LTX-2 Mega Mix Part 2
More nice goodies for **LTX-2**:
- **HDR Control Video support**: you can now provide an HDR Control Video it will be automatically converted to SDR if model doesnt support HDR

- **LTX 2.3 SDR to HDR**: thanks to a new HDR Ic lora, you can now convert SDR Videos to HDR using LTX 2.3. This feature is available as a new *Control Video process* and also in the *Process Full Video* plugin. Please note that the embedded Gradio Gallery video player converts automatically any HDR content to SDR, so if you want to enjoy the full HDR content you will need an external media player (for instance *MPC-BE*)

- **LTX 2.3 Control Video Injection in Phase 2**: up to now even if you picked 2 phases, the *Control Video* was only injected in Phase 1 (Phase 2 was only used for upsampling). Now if you have chosen for at least one Ic Lora, a non null mutiplier for phase 2, the control video will be injected also for phase 2. This will increase output quality with 2 phases but will require more VRAM for phase 2.

- **Process Full Video Custom Settings**: you can now reuse your own presaved settings in the plugin. As you will  link the plugin to your settings any change to the saved settings  will be immediatly available in the plugin. If you find some great combination of loras / model / settings to be used with this Plugin please share them on the discord server so that I can add them in the official list.

*update 11.41*: added Process Full Video Custom Settings 

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

- **Process Process Full Video Plugin**: this *bundled PlugIn* which needs to be enabled first in *the PlugIn tab*, right now supports only *Outpainting*. It relies on *LTX2 Lora outpainting*. It is more or less a *Super Sliding Windows* mode but without the *RAM restrictions* and no risk to explode the *Video Gallery* with huge files. If you are patient enough you can change the Aspect Ratio of a few hours movie (check out below the 1 min sample). Behold how *Sliding Windows transitions* are almost invisible !

- **NEW Processes for Full Video Plugin**: *Refocus* (remove blur), *Ungrade* (remove stylized color grading) and *Uncompress* (remove compression artifacts) have been added. Many thanks to *Oumoumad Mohamed* who created the Ic Loras (including the *Outpainting* lora ) that power these processes. If you have found some Ic Loras that are useful and dont cause glitches with Sliding Windows, let me know and I will add them.

- **WanGP API Video Gen**: *Plugin Developers* can now *Queue a Gen* directly from a plugin. This opens the possibility of plugins that place various gen orders and then combine the results (hint: we could have our very own version of *LTX-Destop* inside WanGP).

- **New One Click Install / Update Scripts**: We have to thank **Tophness / @steve_Jabz** for that one. *Huge Kudos to him!* The scripts will not only install WanGP but also all the *Kernels* (among *Triton, Sage, Flash, GGuf, Lightx2v, Nunchaku*) supported by your GPU. Please have a look at the instructions further down. Dont't hesitate to share feedback or report any issue.

*update 11.31*: fixed phase 1 forced incorrectly in some cases\
*update 11.32*: bugs fixes, Process Full Video now supports Distilled 1.1 & accepts video without audio\
*update 11.33*: Separated LTX2 & LTX2.3 loras in different folders, added easy loras multipliers override\
*update 11.34*: Reverted split as not popular\
*update 11.35*: added Aligned Pose Transfer, Injected Frames & Sliding Windows support, new processes for Process Full Video Plugin 

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

#### 4️⃣ Managing Environments (`scripts\manage.bat` | `/manage.sh`)
Use this script to manage and switch between your sandboxed environments safely.

* **Example Scenario 1: Migrating an Existing Setup**
    * If you have a folder named `venv` that works perfectly and want to use it with the new one-click scripts, run `manage.bat` and select **Add Existing Environment**.
    * Copy-paste the folder path (e.g., `C:\WAN2GP\venv`), select type `venv`, then use **Set Active Environment** to make it the default. Now `run.bat` and `update.bat` will target your existing setup.

* **Example Scenario 2: Testing New Configurations**
    * Let's say you have an environment named `env_stable` that works perfectly, but you want to try the new "Use Latest" combo. Instead of risking your working setup, run `install.bat`, create a *new* environment called `env_testing`, and select **Use Latest**.
    * If the testing environment breaks, simply open `manage.bat`, select **Set Active Environment**, and switch back to `env_stable`. You are back up and running instantly.

---

### One-click (Pinokio) installer:

Get started instantly with [Pinokio App](https://pinokio.computer/)\
It is recommended to use in Pinokio the Community Scripts *wan2gp* or *wan2gp-amd* by **Morpheus** rather than the official Pinokio install.

---


### Manual installation: (for RTX20xx - RTX50xx)

```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.11.14
conda activate wan2gp
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### Manual installation: (for GTX 10xx)

```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

#### Run the application:
```bash
python wgp.py
```

First time using WanGP ? Just check the *Guides* tab, and you will find a selection of recommended models to use.

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
conda create -n wa2gp python=3.11.9
conda activate wan2gp
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
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
- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions for RTX 10XX to RTX 50XX

### AMD
For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/AMD-INSTALLATION.md)** - Complete setup instructions for RDNA 4, 3, 3.5, and 2

## 🎯 Usage

### Basic Usage
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - First steps and basic usage
- **[Models Overview](docs/MODELS.md)** - Available models and their capabilities
- **[Prompts Guide](docs/PROMPTS.md)** - How WanGP interprets prompts, images as prompts, enhancers, and macros

### Advanced Features
- **[Deepy Assistant](docs/DEEPY.md)** - Enable Deepy, configure its tool presets, use selected media and frames, and run Deepy from the CLI
- **[Loras Guide](docs/LORAS.md)** - Using and managing Loras for customization
- **[Finetunes](docs/FINETUNES.md)** - Add manually new models to WanGP
- **[VACE ControlNet](docs/VACE.md)** - Advanced video control and manipulation
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
