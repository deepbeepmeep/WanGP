# Command Line Reference

This document covers current command line options for WanGP. Deprecated Wan model-selection shortcuts remain accepted for existing launch scripts, but are omitted from command help.

## Basic Usage

```bash
# Default launch
python wgp.py

```

## Deepy: Gradio, CLI or Web

Choose one access mode per launch, using your saved Deepy configuration:

```bash
# Gradio at / and the synchronized mobile Web app at /deepy/
python wgp.py --listen

# Interactive terminal chat, without Gradio
python wgp.py --ask-deepy

# Standalone browser app, accessible on the local network
python wgp.py --deepy-server --listen --server-port 7860
```

Gradio includes the Web app at `http://<PC-address>:7860/deepy/`; standalone Web serves it at `http://<PC-address>:7860/`. In a Gradio launch, both interfaces share chat, galleries, selections, settings and active work. Chat and generations continue with every browser closed. Authentication is disabled by default; `--auth` enables a shared password login for Gradio and Deepy. On smartphones the app uses native keyboard dictation. Separate launches do not attach to another process; use saved sessions when moving between processes.

Common Deepy launch options:

| Option | Purpose |
|---|---|
| `--config FOLDER` | Configuration folder containing `wgp_config.json` |
| `--deepy-sessions-dir FOLDER` | Persistent-session location; see [session storage](#deepy-session-location) |
| `--output-dir FOLDER` | Override image, video and audio output folders in CLI or Web mode |
| `--deepy-voice-language CODE` | Overrides the configured microphone transcription language in Gradio or Web; `fr`, `en`, etc., or `auto` for detection |
| `--debug-deepy FOLDER` | Deepy debug logs |
| `--llm-io FOLDER` | LLM input/output transcripts; see [LLM I/O Transcript](#llm-io-transcript) |

Shared network protection options:

| Option | Purpose |
|---|---|
| `--auth` | Enable password-only login for Gradio and Deepy; generate a password when none is supplied |
| `--no-auth` | Explicitly disable web authentication; this is the default |
| `--auth-password PASSPHRASE` | Fixed web passphrase; requires `--auth`. Environment alternative: `WANGP_AUTH_PASSWORD` |
| `--public-url ORIGIN` | Optional exact browser-origin restriction, e.g. `https://wangp.example.com`. Also supports proxies that rewrite the backend Host. A normal HTTPS-to-HTTP proxy with the public Host preserved needs no option. Applies to Gradio and Deepy, with or without `--auth` |
| `--mcp-auth` | Enable separate OAuth authorization for network MCP |
| `--mcp-auth-password PASSPHRASE` | Fixed MCP approval passphrase; requires `--mcp-auth`. Environment alternative: `WANGP_MCP_AUTH_PASSWORD` |
| `--mcp-auth-url ORIGIN` | Public MCP server origin, such as `https://wangp.example.com:7866`; required with `--mcp-auth` |
| `--ssl-certfile FILE`, `--ssl-keyfile FILE` | Certificate and private key for any web/HTTP MCP launch; environment alternatives: `WANGP_SSL_CERT`, `WANGP_SSL_KEY` |
| `--https-port PORT` | Serve HTTPS on this port and redirect the main HTTP port; requires the certificate and key |

The authentication and certificate flags no longer use Deepy-specific names. Authentication remains off unless enabled. See [Authentication, HTTPS, and Reverse Proxies](AUTHENTICATION.md) for web login, `--public-url` examples, certificates, and MCP OAuth setup.

See the [Deepy guide](DEEPY.md) for initial configuration and [interactive CLI commands](DEEPY.md#cli-mode).

## CLI Queue Processing (Headless Mode)

Process saved queues without launching the web UI. Useful for batch processing or automated workflows.

### Quick Start
```bash
# Process a saved queue (ZIP with attachments)
python wgp.py --process my_queue.zip

# Process a settings file (JSON)
python wgp.py --process my_settings.json

# Validate without generating (dry-run)
python wgp.py --process my_queue.zip --dry-run

# Process with custom output directory
python wgp.py --process my_queue.zip --output-dir ./batch_outputs
```

### Supported File Formats
| Format | Description |
|--------|-------------|
| `.zip` | Full queue with embedded attachments (images, videos, audio). Created via "Save Queue" button. |
| `.json` | Settings file only. Media paths are used as-is (absolute or relative to WanGP folder). Created via "Export Settings" button. |

### Workflow
1. **Create your queue** in the web UI using the normal interface
2. **Save the queue** using the "Save Queue" button (creates a .zip file)
3. **Close the web UI** if desired
4. **Process the queue** via command line:
   ```bash
   python wgp.py --process saved_queue.zip --output-dir ./my_outputs
   ```

### CLI Queue Options
```bash
--process PATH          # Path to queue (.zip) or settings (.json) file (enables headless mode)
--dry-run               # Validate file without generating (use with --process)
--output-dir PATH       # Override output directory (use with --process)
--verbose LEVEL         # Verbosity level 0-2 for detailed logging
```

### Console Output
The CLI mode provides real-time feedback:
```
WanGP CLI Mode - Processing queue: my_queue.zip
Output directory: ./batch_outputs
Loaded 3 task(s)

[Task 1/3] A beautiful sunset over the ocean...
  [12/30] Prompt 1/3 - Denoising | Phase 2/2 Low Noise
  Video saved
  Task 1 completed

[Task 2/3] A cat playing with yarn...
  [30/30] Prompt 2/3 - VAE Decoding
  Video saved
  Task 2 completed

==================================================
Queue completed: 3/3 tasks in 5m 23s
```

### Exit Codes
| Code | Meaning |
|------|---------|
| 0 | Success (all tasks completed) |
| 1 | Error (file not found, invalid queue, or task failures) |
| 130 | Interrupted by user (Ctrl+C) |

## MCP Server

```bash
--mcp                              # Start WanGP as an MCP server without the web UI
--mcp-webui                        # Override Agents config: enable MCP with the web UI
--no-mcp-webui                     # Override Agents config: disable MCP with the web UI
--mcp-transport TRANSPORT          # stdio (default with --mcp), sse, or streamable-http (default with the web UI)
--mcp-host HOST                    # Host for HTTP transports
--mcp-port PORT                    # Override Configuration > Agents port for HTTP transports (default: 8000)
--mcp-api-version {1,2,2.1}        # WanGP API contract; latest (2.1) by default
--mcp-async                        # Permit wait=false in v2/v2.1; disabled by default
--mcp-console-output               # Mirror WanGP output while serving MCP
--mcp-allow-read-file-system       # Allow agents to submit arbitrary server file paths (disabled by default)
```

MCP is off by default with the web UI, including after upgrading. Control autostart with **Enable MCP with Web UI** under **Configuration > Agents**. When enabled, agents' generations join the queue shown in the browser, run after the tasks already waiting and appear in every open page's progress and preview; their outputs go to the agent's workspace without switching the gallery you are viewing. In this mode asynchronous calls are always available. Set **MCP Server Port** under **Configuration > Agents** and restart, or override it for one launch with `--mcp-port PORT`. `--mcp-webui` and `--no-mcp-webui` override the saved autostart choice for one launch without changing it. See [MCP server inside the web UI](API.md#mcp-server-inside-the-web-ui).

Media IDs returned by the Gallery remain usable when unrestricted filesystem reads are disabled. V2.1 also provides output-folder reads and a writable session workspace. Streamable HTTP and SSE servers expose short-lived file and Gallery upload/download URLs; stdio downloads use binary MCP resources with byte-range support.

The default MCP interface is v2.1, with `remux_media`, native visual inspection, file delivery and searchable documentation. Existing clients can pin an earlier contract with `--mcp-api-version 1` or `--mcp-api-version 2`. Both `--mcp-api-version` and `--mcp-async` also work with `python -m shared.mcp`. The number selects the WanGP tool API, not the MCP protocol version; unsupported versions are rejected. API v1 retains its original wait behavior. In v2/v2.1, generation and post-processing wait by default; `--mcp-async` enables optional asynchronous calls without changing that default. See [v2.1 additions](API.md#mcp-api-v21-additions) and [API migration](API.md#mcp-api-v2-and-migration).

### Examples
```bash
# Overnight batch processing
python wgp.py --process overnight_jobs.zip --output-dir ./renders

# Quick validation before long run
python wgp.py --process big_queue.zip --dry-run

# Verbose mode for debugging
python wgp.py --process my_queue.zip --verbose 2

# Combined with other options
python wgp.py --process queue.zip --output-dir ./out --attention sage2
```

## LLM I/O Transcript

```bash
--llm-io FOLDER                     # Record local and remote LLM traffic as a readable plain-text transcript
```

WanGP creates a timestamped `.log` file in the supplied folder. Each record is labelled `[OUT → LLM]` or `[IN ← LLM]` and identifies the engine and stream. Text is preserved verbatim; known special tokens include their names and numeric IDs, while other token streams use numeric IDs. Binary media is described by its source, type, dimensions, and size instead of being written as encoded data.

The transcript can contain prompts, conversation history, tool arguments/results, and model output. Enable it only while diagnosing a problem and treat the resulting file as private data.

## Deepy Session Location

```bash
--deepy-sessions-dir FOLDER          # Override the persistent Deepy sessions folder
```

The default is `deepy_sessions` in the WanGP installation root. The folder is created just in time, only after multi-session mode is enabled and Deepy receives its first request (or when a session archive is explicitly imported).

Each materialized session keeps its canonical decoder context in `context.json` and an append-only `cards.jsonl` journal of consolidated Web-client commands. Streaming token fragments are not written: a thought, response section, or tool card is journaled once it reaches a safe completed boundary, then those commands are replayed when the session is resumed. During replay, the Web client keeps the transcript hidden and suppresses per-command layout, disclosure, animation, and scroll work before one final refresh.

## Model and Performance Options

### Model Configuration
```bash
--quantize-transformer BOOL   # Enable/disable transformer quantization (default: True)
--compile                     # Enable PyTorch compilation (requires Triton)
--attention MODE              # Force attention mode: sdpa, flash, sage, sage2
--profile NUMBER              # Performance profile 0-5 (default: 4)
--preload NUMBER              # Preload N MB of diffusion model in VRAM (Manual VRAM Preload for every kind of output)
--fp16                        # Force fp16 instead of bf16 models
--save-quantized              # Save an INT8 Quanto checkpoint during model loading
--convrot                     # With --save-quantized, save INT8 ConvRot instead of Quanto INT8
--gpu DEVICE                  # Run on specific GPU device (e.g., "cuda:1")
```

### Performance Profiles
- **Profile 0**: like Profile 4, models are sent to the GPU part by part, but no model keeps Reserved RAM for good: the parts of the model running are kept in Reserved RAM while it runs, then the RAM goes to the next model. Close to the speed of the profiles that keep models in Reserved RAM, with much less RAM. By default it also keeps in VRAM as much of each model as the generation leaves free, like Profile 3 (*VRAM Preload* below; choose *Manual* to set a fixed amount instead). The language models of Deepy and the prompt enhancer always run this way, with *Dynamic* preload, whatever the memory profile of your model, and get the CPU's help when they write (see *Language Models CPU Compute*)
- **Profile 1**: each model loaded whole in VRAM, all models kept in Reserved RAM. The fastest generations and model switches, needs the most RAM and VRAM
- **Profile 2**: all models kept in Reserved RAM, sent to the GPU part by part. Runs models larger than your VRAM, leaves VRAM for long videos or large images and switches models fast, needs a lot of RAM
- **Profile 3**: each model loaded whole in VRAM, only the main models kept in Reserved RAM. Fast generations with less RAM, needs enough VRAM for the whole model
- **Profile 3+** (3.5, recommended for audio models): Profile 3 without any Reserved RAM. Audio models are usually small enough to fit whole in VRAM, where the language models many of them include run fastest (they use CUDA Graphs with every profile, also when sent to the GPU part by part); models load more slowly
- **Profile 4** (recommended): only the main models kept in Reserved RAM, sent to the GPU part by part. The most versatile: runs models larger than your VRAM and leaves VRAM for long videos or large images
- **Profile 4+** (4.5): Profile 4 sending one part at a time. Saves up to about 1 GB of VRAM, slightly slower
- **Profile 5** (fail safe): almost no Reserved RAM, all models sent to the GPU part by part. For PCs short of RAM and VRAM, slower with short steps such as images

### Preloading Part of the Model in VRAM
With profiles 0, 2, 4 and 5, models are transferred to the GPU block by block at every denoising step. Image models such as Flux have short steps, so with profile 4 these transfers, rather than the GPU computation, can set the speed. In *Configuration / RAM/VRAM Management*, a *VRAM Preload* choice next to the default memory profile of each kind of output (video, image and audio) decides how much of each model stays in VRAM, so that less is transferred at each step:
- *Default*: the memory profile's own choice (*Dynamic* for profile 0, none for profiles 2, 4 and 5).
- *Dynamic*: as much of each model as the generation leaves free, adapted as it goes (below).
- *Manual*: the amount set with the *VRAM Preload (MB)* slider shown below it, spread across the blocks of each model. Models that work side by side and stay in VRAM together (a world model and its video decoder, the two stages of some audio models) share that amount instead of getting it each, the main model first: for example 12000 MB with a world model of 8.4 GB and a decoder of 7 GB keeps the world model whole and 3.6 GB of the decoder, 12 GB in all. Some models keep more in VRAM on their own (for instance the language models of audio models, which stay whole): the preload never gives a model less than that. Models made of many experts (Nemotron 3.5 Transcribe, Fish Audio) give what the preload leaves beyond such a model to their experts, which then run on the GPU instead of the processor. If the generation needs more VRAM than the preload leaves free, just enough of the preloaded parts leave the VRAM (also when the preload holds the whole model) until the model is loaded again, instead of the generation spilling into RAM and slowing down several times.

The choice is hidden for profiles 1, 3 and 3+, which always keep as much of each model in VRAM as fits. A memory profile chosen in the settings of a generation uses the VRAM Preload of its kind of output, and `--preload` sets a manual value for all three.

For Flux dev at 1024x1024, a manual 6000 MB shortened the profile 4 steps by about 22%, to within about 20% of profile 1, with about half of its peak VRAM. Preloading uses VRAM during denoising only: at this resolution it did not raise the peak VRAM, which the image decoding sets.

#### Dynamic VRAM Preload
*Dynamic* helps most where transfers set the speed: images and low resolutions, with short steps. Video steps are usually long enough to hide the transfers, so *Default* is about as fast for videos. It needs the *MMGP Optimized VRAM Allocator* ([VRAM Allocator](#vram-allocator), the default); with PyTorch's allocator, or with Profile 4+, which sends one part at a time, the profile default applies.

- The first step of a new resolution, frame count or denoising stage runs with the profile default (as with a preload of 0) and measures the VRAM that the generation itself needs. During the next step, the parts of the model that fit in the rest of the VRAM stay there as they are transferred, without extra transfers, and the following steps no longer transfer them. This shortens the steps that transfers slow down, such as images and low resolutions; when the whole model fits, the speed then matches Profile 1.
- When the *Reserved RAM for Pinning* cannot hold the whole model, the parts that are not pinned, the slowest to transfer, are the first kept in VRAM.
- Each denoising stage has its own preload: a low resolution first stage (LTX-2, two-pass H3) keeps more of the model in VRAM than the full resolution stage that follows, and each model of a two-model video generator (Wan 2.2) gets its own.
- What is measured is kept for each model until WanGP closes or its definitions are refreshed (for instance after a finetune is edited), and is measured again with another quantization. Later generations with the same settings skip the measuring step: from their second step, the model fills the free VRAM.
- If VRAM runs short anyway (another program took VRAM, or a later step needs more than the first), parts of the model that are quick to transfer leave the VRAM instead of an out of memory error, and the following steps of that resolution keep that much less in VRAM so that it does not happen again. The generated images and videos are the same as with a manual preload.
- The VRAM stays filled during denoising: other programs that need VRAM at the same time get less. Not used with compilation (*Compile Transformer Model*) or when `--preload` is set.

### Memory Management
```bash
--perc-reserved-mem-max FLOAT # Share of RAM that pinning may lock, as a fraction (0.4 = 40%)
```
Pinned ("reserved") RAM makes the transfers to the GPU fast, but nothing else can use it. *Configuration / RAM/VRAM Management / Reserved RAM for Pinning* sets the share of RAM that WanGP may pin, in percent (0 = automatic: 40% on Windows, 60% on Linux). `--perc-reserved-mem-max` takes precedence over this setting, which takes precedence over the `perc_reserved_mem_max` environment variable.

When the models do not all fit in it, WanGP reserves first the parts used at every step; with [Smart Memory Pinning](#smart-memory-pinning), the rest still reaches the GPU almost as fast, so a smaller share mostly costs speed on short steps.

### Trying a Smaller GPU
```bash
--max-vram 8   # WanGP behaves as on a GPU with 8 GB of VRAM
```
`--max-vram GB` makes WanGP behave as it would on a GPU with that much VRAM: memory profiles, preloads, VAE tiling and Deepy's choices follow that card, and WanGP uses at most GB - 1.2 GB of VRAM (a real card leaves about that much to the driver and the desktop). Use it to check that settings or a workflow fit a smaller GPU before sharing them. The VRAM other programs use on your GPU does not count, unless they leave less than the emulated card would: then WanGP has only what is really free. The speed stays that of your GPU.

### RAM Budget
```bash
--max-ram 24   # WanGP plans its RAM within 24 GB
```
`--max-ram GB` gives WanGP a RAM budget, to leave room for the other programs you use beside it: the RAM it reserves for fast transfers, the RAM it keeps free and the moment it gives back the RAM of models that are not running all follow that budget, counting the RAM WanGP already uses. The RAM other programs use does not count, unless they leave WanGP less than its budget. The RAM WanGP may reserve stays a share of your PC's real RAM (see `--perc-reserved-mem-max`), a safeguard that keeps your system responsive, and never exceeds the budget. It guides WanGP's memory planning but is not a hard limit: a generation that needs more RAM than the budget still gets it from your PC.

### VRAM Allocator
```bash
--vram-allocator vmm_spill  # MMGP Optimized VRAM Allocator with RAM spilling (default)
--vram-allocator vmm        # MMGP Optimized VRAM Allocator, an out of memory error when no VRAM is left
--vram-allocator default    # PyTorch's allocator
```
The *MMGP Optimized VRAM Allocator* (*Configuration / RAM/VRAM Management / VRAM Allocator*, on by default) recycles more efficiently the VRAM that is no longer used, which lowers the peak VRAM of long videos, large images and Deepy, with the same outputs and speed:

| Workload | VRAM saved |
|---|---:|
| H3 1920x1088, 241 frames | about 5.4 GB |
| H3 1280x720, 241 frames | about 1.5 GB |
| Flux dev 1024x1024 (image decoding) | about 1.3 GB |
| Deepy, Bonsai 2 27B, 20K-token prompt | about 0.2 GB |

The gain grows with the resolution and duration. The setting applies when WanGP starts; `--vram-allocator` takes precedence over it. It works on Windows and Linux with PyTorch 2.3 or newer and NVIDIA GPUs; elsewhere, WanGP says so at startup and uses PyTorch's allocator. The VRAM used by other programs on the same GPU (another WanGP, ComfyUI, a game, a browser) is left to them; only when WanGP would otherwise run out of VRAM does it take the VRAM they are not using, which Windows then moves to system RAM. When the error happens anyway, its message shows the VRAM free on the whole GPU and for WanGP, how much more was needed, and why it was refused.

*With RAM Spilling* (the default), when VRAM runs out, what no longer fits goes to system RAM instead of stopping the generation: the steps that use it are much slower, but a generation slightly too large for your VRAM can finish. When that would leave less than a tenth of the RAM (at least 4 GB) available, the NVIDIA driver takes over on Windows and places what no longer fits in shared GPU memory, as it does with PyTorch's allocator: a generation that finishes with PyTorch's allocator also finishes with this one. On a GPU that also drives your display, the screen may flash or go black for a moment meanwhile, without affecting the generation. Without spilling (`vmm`), running out of VRAM stops the generation with an out of memory error, instead of slowly spilling into shared GPU memory as the NVIDIA driver otherwise does on Windows.

### RAM Allocator
```bash
--ram-allocator mmgp      # MMGP RAM Allocator (default)
--ram-allocator default   # PyTorch's CPU allocator
```
Same as *Configuration > RAM/VRAM Management > RAM Allocator*. PyTorch keeps the RAM of the CPU tensors it frees (decoded frames, converted weights, frames waiting to be saved) to reuse it later: after a generation or after a model is released, several GB can stay in use, and on Windows the *committed* memory shown by the Task Manager can grow up to twice what these tensors need. With the *MMGP RAM Allocator*, the RAM of the CPU tensors of 1 MB and more goes back to the system when the queue is done, when a model is released, and whenever the RAM runs short (before the VRAM allocator would have to spill into RAM). While generations of the same model follow one another, up to 2 GB of it (5% of the RAM on smaller PCs) is kept for the next one, which reuses it at full speed; nothing is released in the middle of a generation unless the RAM runs short. It also commits only what each tensor needs, where PyTorch's allocator on Windows can commit up to twice as much.

The setting applies when WanGP starts; `--ram-allocator` takes precedence over it. It works on Windows and Linux with PyTorch 2.6 to 2.15; elsewhere, WanGP says so at startup and uses PyTorch's allocator.

### Windows Power Throttling
```bash
--no-prevent-power-throttling # Let Windows slow down WanGP in the background to save power
```
On Windows, WanGP asks by default to keep its full CPU speed when its window is minimized or in the background. Windows may otherwise slow down a background application several times, which slows down generation on the steps where the GPU waits for the CPU. On a laptop running on battery, `--no-prevent-power-throttling` saves power at the cost of slower generations while WanGP is in the background.

You can also change **Configuration / Performance / Prevent Windows Background Power Throttling**, then save and restart the server. It applies to the server process and all browser sessions, without raising process priority. Explicit `--prevent-power-throttling` or `--no-prevent-power-throttling` launch options override the saved choice. This is separate from **Continue Browser Actions in Background**, which lets pending interface actions and updates continue while the tab is in the background. Already-started generation queues run on the server and continue even when all browser tabs are closed, regardless of this browser option.

### Compute Cache
*Configuration / Performance / WanGP Compute Cache* (off by default, shown with RTX 50xx and other NVIDIA Blackwell GPUs). With these GPUs, PyTorch and the other GPU libraries ship their code in a form that the NVIDIA driver optimizes for the exact GPU each time WanGP starts, which costs about 10-15 seconds at the first generation. The driver keeps the result in a cache shared by every program and limited to 1 GB, where models push out one another's code and other programs trim it, so the work is done again after restarts and rewritten on the disk at each launch. Older GPUs run the shipped code as is and do not need this option.

The option keeps this cache in a folder of its own: `ComputeCache` in the WanGP folder, or the *Compute Cache Folder* you choose (relative to the WanGP folder, or a full path). The code of each model is completed over its first few launches (the driver optimizes a limited amount per launch). After that, the first generation after a restart starts sooner, and nothing more is written to the disk at launch, which spares the SSD. It may take up to 2 GB of disk space. The setting applies when WanGP starts.

### Smart Memory Pinning
*Configuration / RAM/VRAM Management / Smart Memory Pinning* (On by default) speeds up the models that the memory profile does not keep in Reserved RAM: the text encoders with Profile 4, every model with Profile 5, and the part of a model that does not fit when *Reserved RAM for Pinning* is low. With it, they reach the GPU almost as fast as the models kept in Reserved RAM, for about 1-2 GB of Reserved RAM.

- It helps most with Profile 5, with image models and other generations made of short steps (low resolutions, the first pass of distilled models), and on PCs with little RAM to reserve. With long steps (high resolutions, long videos), the difference is small.
- Off: these models reach the GPU more slowly, and the 1-2 GB of Reserved RAM stay free for other programs.
- Profile 0 always uses it, with a larger share: the parts of the model running that do not fit in VRAM use the *Reserved RAM for Pinning* left by the other models, released when you switch models or when your PC runs short of RAM.
- Changing this option reloads the model.

## Lora Configuration

```bash
--loras PATH                 # Root folder for default LoRA subfolders (default: loras)
--lora-config FILE           # Optional JSON mapping LoRA subfolder names to paths
--lora-preset PRESET         # Load lora preset file (.lset) on startup
--check-loras                # Filter incompatible loras (slower startup)
```

Use `python wgp.py --lora-config lora_paths.json` to override individual collections. JSON keys match the exact subfolder names in the default `loras/` folder, such as `wan`, `wan_5B`, or `flux2_klein_4b`. Values are complete directory paths; relative values resolve beside the JSON file.

JSON entries take precedence over `--loras`. Unlisted keys use the root from `--loras`, then `wgp_config.json`'s `loras_root`, then `loras/`. The former model-specific `--lora-dir*` flags have been removed. See the [LoRA guide](LORAS.md#custom-lora-directories) for a sample JSON and setup instructions.

## Generation Settings

### Basic Generation
```bash
--seed NUMBER                # Set default seed value
--frames NUMBER              # Set default number of frames to generate
--steps NUMBER               # Set default number of denoising steps
--advanced                   # Launch with advanced mode enabled
```

### Advanced Generation
```bash
--teacache MULTIPLIER        # TeaCache speed multiplier: 0, 1.5, 1.75, 2.0, 2.25, 2.5
```

## Interface and Server Options

### Server Configuration
```bash
--server-port PORT           # Gradio server port (default: 7860)
--server-name NAME           # Gradio server name (default: localhost)
--listen                     # Make server accessible on network
--share                      # Create shareable HuggingFace URL for remote access
--open-browser               # Open browser automatically when launching
```

### Interface Options
```bash
--lock-config                # Prevent modifying video engine configuration from interface
--theme THEME_NAME           # UI theme: "default" or "gradio"
```

## File and Directory Options

```bash
--settings PATH              # Path to folder containing default settings for all models
--config PATH                # Config folder for wgp_config.json and queue.zip
--workspaces-dir FOLDER       # Workspace manifests/media; overrides Config / Outputs (default: ./workspaces)
--offline                    # Never download: use only the files already on this computer
--verbose LEVEL              # Information level 0-2 (default: 1)
```

With `--offline`, WanGP and the libraries it uses never download anything: models, text encoders, LoRAs from links and processing tools must already be in your checkpoints folders. A missing file stops the job with an error naming it, instead of starting a download. Useful on a computer without internet access, on a metered connection, or to make sure a test or a batch runs only with the files you already have.

## Examples

### Basic Usage Examples
```bash
# Launch with specific model and loras
python wgp.py ----lora-preset mystyle.lset

# High-performance setup with compilation
python wgp.py --compile --attention sage2 --profile 3

# Low VRAM setup
python wgp.py --profile 4 --attention sdpa
```

### Server Configuration Examples
```bash
# Network accessible server
python wgp.py --listen --server-port 8080

# Shareable server with custom theme
python wgp.py --share --theme gradio --open-browser

# Locked configuration for public use
python wgp.py --lock-config --share
```

### Advanced Performance Examples
```bash
# Maximum performance (requires high-end GPU)
python wgp.py --compile --attention sage2 --profile 3 --preload 2000

# Optimized for RTX 2080Ti
python wgp.py --profile 4 --attention sdpa --teacache 2.0

# Memory-efficient setup
python wgp.py --fp16 --profile 4 --perc-reserved-mem-max 0.3
```

### TeaCache Configuration
```bash
# Different speed multipliers
python wgp.py --teacache 1.5   # 1.5x speed, minimal quality loss
python wgp.py --teacache 2.0   # 2x speed, some quality loss
python wgp.py --teacache 2.5   # 2.5x speed, noticeable quality loss
python wgp.py --teacache 0     # Disable TeaCache
```

## Attention Modes

### SDPA (Default)
```bash
python wgp.py --attention sdpa
```
- Available by default with PyTorch
- Good compatibility with all GPUs
- Moderate performance

### Sage Attention
```bash
python wgp.py --attention sage
```
- Requires Triton installation
- On RTX 20XX, install SageAttention 1.0.6
- 30% faster than SDPA
- Small quality cost

### Sage2 Attention
```bash
python wgp.py --attention sage2
```
- Requires Triton and SageAttention 2.x
- Requires RTX 30XX or newer (Ampere or newer)
- 40% faster than SDPA
- Best performance option

### Flash Attention
```bash
python wgp.py --attention flash
```
- May require CUDA kernel compilation
- Good performance
- Can be complex to install on Windows

## Troubleshooting Command Lines

### Fallback to Basic Setup
```bash
# If advanced features don't work
python wgp.py --attention sdpa --profile 4 --fp16
```

### Debug Mode
```bash
# Maximum verbosity for troubleshooting
python wgp.py --verbose 2 --check-loras
```

### Memory Issue Debugging
```bash
# Minimal memory usage
python wgp.py --profile 4 --attention sdpa --perc-reserved-mem-max 0.2
```



## Configuration Files

### Settings Files
Load custom settings:
```bash
python wgp.py --settings /path/to/settings/folder
```

### Config Folder
Use a separate folder for the UI config and autosaved queue:
```bash
python wgp.py --config /path/to/config
```
If missing, `wgp_config.json` or `queue.zip` are loaded once from the WanGP root and then written to the config folder.

### Lora Presets
Create and share lora configurations:
```bash
# Load specific preset
python wgp.py --lora-preset anime_style.lset

# With custom lora root
python wgp.py --loras /shared/loras --lora-preset mystyle.lset
```

## Console Log Stamps (`--log-timestamp`)

The console lines WanGP prints can carry the time of day at their start:
`2026-10-12 15:06:41 | Model loaded: ...`. The feature is off by default: a run without the flag
prints exactly what it printed before, and `--log-timestamp=on` -- or the flag on its own -- stamps
the console of a single run; `WAN2GP_LOG_TIMESTAMP=1` stamps every run. The MCP server over stdio
refuses the stamps in any case: its stdout carries JSON-RPC frames.
The stamps ride on stdout, so `python wgp.py --process my_queue.zip --log-timestamp=on 2>&1 | tee
job.log` stores them in a file. Progress bars are never disturbed: a bar is stamped once, where
its line starts, and every redraw reaches the output byte for byte, colour codes included.

```bash
# No flags: the console prints exactly as it always has - the stamps are off
python wgp.py --process my_queue.zip

# Stamp the console of this run: the flag on its own, =on, and a separate 'on' all mean the same
python wgp.py --process my_queue.zip --log-timestamp
python wgp.py --process my_queue.zip --log-timestamp=on

# Keep the log of a run: the stamps are on stdout, so tee them
python wgp.py --process my_queue.zip --log-timestamp=on 2>&1 | tee job.log

# Stamp the Deepy chat, the MCP server or any other mode
python wgp.py --ask-deepy --log-timestamp=on
python wgp.py --process my_queue.zip --log-timestamp=off
```

| Flag | Default | Effect |
|------|---------|--------|
| `--log-timestamp` (`on` / `off`) | off in every mode | Stamp the console lines WanGP prints. The flag on its own means on, and `off` leaves the console exactly as it prints today |
| `--log-timestamp-format` | `full` | `full` (`YYYY-MM-DD HH:MM:SS`), `clock` (`HH:MM:SS`), `date`, `ms`, `uptime` (`+1h23m45s`), `clock+uptime`, any strftime pattern, or a token mix such as `{time}.{ms} +{uptime}` |
| `--log-timestamp-sep` | `pipe` (`\|`) | Text between the stamp and the message: `pipe`, `space`, `dash`, `none`, or a literal (`">"`). It is padded to one trailing space, so the next stamp never glues to the text before it |

The stamp is the plain text `2026-10-12 15:06:41 | `, with no brackets: WanGP already uses
`[50/50]`, `[ERROR]` and tqdm's `[00:41<00:00, 1.21it/s]`, and syslog reads a bracketed
number as a process id. Brackets are available by writing them into the pattern:
`--log-timestamp-format "[%H:%M:%S]"`.

Progress rules, since they are what decides how a console reads:

- A *complete* console line gets exactly one stamp, at its start, using the time of day.
- A self-overwriting line (the CLI queue status line, `tqdm`, the download bars) is
  echoed untouched, so a bar is stamped once where its line starts and its redraws -- colour
  codes included -- reach the terminal exactly as they are written.
- A line with nothing on it is echoed unstamped -- a bare `print()`, a padded status line closing
  on a carriage return, a line of spaces, a bar's closing write of cursor codes -- so an empty console
  line stays empty, with no stamp sitting on it. Control codes are nothing to draw: the cursor moves,
  the colour codes, `ESC [ ? 25 h`, the `ESC ] 8 ; ;` link strings and the `Ctrl+C` byte an interrupt
  leaves behind are consumed by the terminal and print as nothing.
- The console always ends on a line of its own. A bar that closes on a carriage return leaves its line
  open, and the stamp written on it used to be the last thing the app printed, with the shell prompt
  arriving glued to it (`2026-10-22 18:51:17 | user@host:~$`). The stamped streams write one newline at
  exit -- on a terminal only, and only under a stamp they actually wrote.
- stdout is never stamped in MCP stdio mode: that stream carries JSON-RPC.
- A `Traceback (most recent call last)` line is the last line the console is ever stamped on:
  the crash dump under it, and the prompt that returns after the crash, are echoed exactly as
  they are -- a stamped backtrace is noise to read through.
- A line that opens with a rule of stars is a header, not a message, and is echoed unstamped: the
  `WanGP v…` banner, the `Memory Management for the GPU Poor` banner and the `Applying MMGP Memory
  Profile` banner, each centred in a field of stars. The line keeps the width it was built with, so
  a stamped run shows the same rule as a plain run. A single star is a bullet and two stars open a
  bold line, so `* Running on local URL:` keeps its stamp.

Shutting the server down: in a terminal, Ctrl+C writes one newline of its own, so the shutdown
lines read `2026-10-13 15:50:10 | Keyboard interruption in main thread... closing server.` on a
line that is theirs, clear of the `^C` the terminal echoes. It is the interrupt as it reaches the
app that adds the newline, so a run whose output is a pipe is untouched, and a component that
installs an interrupt handler of its own after the stamps keeps its own shape.

### Reading a Stamped Console Log

```bash
python wgp.py --process q.zip --log-timestamp=on 2>&1 | tee job.log
grep -E 'Finished|finished|ERROR' job.log           # what happened, and when
```

## Environment Variables

While not command line options, these environment variables can affect behavior:
- `CUDA_VISIBLE_DEVICES` - Limit visible GPUs
- `PYTORCH_CUDA_ALLOC_CONF` - CUDA memory allocation settings
- `TRITON_CACHE_DIR` - Triton cache directory (for Sage attention) 
- `WAN2GP_DEEPY_TELEMETRY=1` - Enable detailed Deepy decode, MTP, CUDA-memory, and GPU telemetry when verbose level 2 is active (disabled by default)
- `WAN2GP_FFMPEG_TIMEOUT` - Timeout in seconds when finalizing ffmpeg containers or merging continuation segments in Media Flow (default: `300.0`)
- `WAN2GP_LOG_TIMESTAMP=1` - Turn the console log stamps on in every mode, the Deepy chat and the MCP server included (the default is off everywhere; the per-run `--log-timestamp=on|off` decides one run)
- `WAN2GP_LOG_TIMESTAMP_FORMAT` - Default stamp pattern when `--log-timestamp-format` is left out (`full`, `clock`, `uptime`, or a pattern)
- `WAN2GP_LOG_TIMESTAMP_SEP` - Default separator when `--log-timestamp-sep` is left out (`pipe`, `space`, `dash`, `none`, or a literal)

---

> Applies to: WanGP startup options and saved-queue processing from a shell. Command-line flags configure the application process; generation settings and API tool arguments are specified separately.
