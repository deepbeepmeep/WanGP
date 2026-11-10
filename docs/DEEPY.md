# Deepy

Deepy is WanGP's conversational media assistant. Tell it what you want to create or change, and it can generate, inspect, edit, extract, transcribe, merge, and transform images, video, and audio while keeping the context of your project.

Use Deepy when you want to work toward an outcome instead of manually operating every generation and post-processing control. For example, you can ask it to turn a portrait and a voice recording into a talking video, compare several results, or build a sequence of related shots.

**Deepy can make mistakes, so verify important results.**

## Choose Deepy Zero or Deepy Prime

- **Deepy Zero** is fast and lightweight. Use it for focused tasks such as generating one asset, editing selected media, extracting a clip, resizing a file, or producing a transcript. It works well with the smaller supported Qwen models.
- **Deepy Prime** is for projects that need planning or several connected actions. Use it when Deepy must compare compatible models, combine multiple media assets, inspect intermediate results, manage project files, or work with external MCP services. Prime requires a local Qwen3.8 VL model or a configured remote LLM.

Both versions share generation features, galleries, media references, and saved sessions. Both offer a dedicated video-with-references template. Choose the assistant in **Configuration > Prompt Enhancer / Deepy**.

## Enable Deepy

1. Start WanGP with `python wgp.py`.
2. Open **Configuration > Prompt Enhancer / Deepy**.
3. Choose **Deepy Zero** or **Deepy Prime**.
4. Choose a supported local model or, for Prime, a configured remote LLM.
5. Save the configuration.

Supported local choices are:

- `Qwen3.5VL Abliterated 4B`
- `Qwen3.5VL Abliterated 9B`
- `Qwen3.8VL Uncensored 27B` (supports local Deepy Prime)
- `Qwen3.8VL Flash-Next 125B` (supports local Deepy Prime)

For a remote Prime engine, see [Remote LLMs](REMOTE_LLMS.md). Once Deepy is enabled, **Ask Deepy** appears in the Gradio left dock. The CLI and standalone Web app use the same saved configuration.

## Choose an interface

| Interface | Launch | Best for |
|---|---|---|
| **Gradio + Web** | `python wgp.py --listen --server-port 7860` | Full WanGP controls on the PC plus a synchronized, phone-friendly Deepy app at `/deepy/`. |
| **CLI** | `python wgp.py --ask-deepy` | Terminal-based work without a browser or server. |
| **Standalone Web** | `python wgp.py --deepy-server --listen --server-port 7860` | A focused browser app for chat, galleries, and basic Deepy settings without Gradio. |

All interfaces use the same engine configuration, tool templates, and saved-session format. Use the same `--config FOLDER` and `--deepy-sessions-dir FOLDER` when you want separate launches to use the same configuration and sessions. `--config` names the folder containing `wgp_config.json`, not the JSON file itself.

### Use Gradio and Web together

While Gradio is running, open:

- `http://<PC-address>:7860/` for Gradio
- `http://<PC-address>:7860/deepy/` for the phone-friendly app

You can also follow **Web app ->** in Deepy's settings. Both views share the current conversation, galleries, selected media, settings, progress, and generation queue. Start a request on one device and follow, pause, stop, or inspect its result on the other.

The Deepy Web app and all open Gradio pages are synchronized while connected to the same WanGP process. Gallery workspace changes and completed generations appear everywhere, while each Gradio page keeps its own unsent generation form and draft values.

Closing every browser does not stop accepted work while WanGP remains running. Reopen the same address to recover the current view. Live sharing applies only to Gradio and Web clients connected to the same WanGP process.

### Use standalone Web

For the same computer:

```powershell
python wgp.py --deepy-server
```

For a phone or another computer on the local network:

```powershell
python wgp.py --deepy-server --listen --server-port 7860
```

Open `http://<PC-address>:7860`, using the PC's LAN address, such as `192.168.1.50`. Do not enter `0.0.0.0`; that is only the listening address. If the page cannot connect, allow the selected port through the PC firewall.

Standalone Web and CLI run as separate processes and do not attach to another running instance. To continue work in a different process, enable saved sessions, finish current work, stop the old process, and resume the session using the same configuration and session folder.

## What Deepy can do

You can ask Deepy to:

- generate or edit images
- generate video from a prompt, image, or compatible media inputs
- create a talking video from a still image and speech audio
- create speech from a description or voice sample, generate songs, and generate sound from a description
- inspect images, video frames, and 3D assets, or compare several visuals
- report useful media details such as dimensions, duration, FPS, frame count, and audio tracks, and for 3D assets their size and animation frames, and list the files a model saved beside a media item (its transcript and subtitles, the transparent frames of an RGBA video) and read the text ones, without needing file access
- extract images, clips, or audio; transcribe audio or video; mute or replace a video soundtrack; resize or crop media; compose assets side by side; and merge videos
- find available LoRAs, explain the active tool defaults, and answer WanGP usage questions

Prime can also:

- generate explorable worlds from a description, walked through with exploration instructions, or continue a world of the Worlds gallery; describe its worlds (scene, style, duration and each segment). Exploring a world stays yours, with the keyboard
- make 3D assets from a picture of an object, or from words by first generating that picture
- render views of a 3D asset at one consistent scale (front, sides, back, any angle, any animation frame, close-ups, and depth or normal images to guide other models)
- draw images from a precise layout, with shapes, text, and gallery images placed where needed
- build a character sheet: the same character from several angles with a title and labels, which then helps keep the character consistent in later images and videos
- measure audio levels and silences, separate vocals, convert or combine image channels, and mask named objects
- run Python in a sandbox to place exact text and layouts over generated images, draw charts and infographics, assemble grids and picture-book pages, extract palettes, write subtitle files from measured timings, build animated clips frame by frame, and model procedural 3D props that join the 3D gallery
- look things up on the web when you enable it: facts, dates, current events, references, what a real place, object or style looks like before writing a prompt about it, and LoRAs from Civitai and Hugging Face for a style or an effect, installed for the model it is about to use
- run the workflows that enabled Plugins offer, such as Media Flow's restoration, outpainting, upscaling and interpolation of long videos and whole folders (see [Plugin workflows](#plugin-workflows-prime))

Zero keeps a shorter tool list so that it stays fast and leaves most of its context for the conversation.

Prime is especially useful for requests with dependencies. It can create an intermediate asset, inspect it, revise it if necessary, and use the accepted result in the next stage.
Prime can use `remux_media` to mix stems into one audio stream, package separate selectable audio tracks, or add selectable subtitle tracks to a video. Subtitle files may be SRT, VTT, ASS, or SSA in an authorized workspace path; use `subtitle_tracks` with optional language, title, and default status. Existing video subtitle tracks are retained unless `include_video_subtitles=false`. With a video and subtitles but no audio inputs, compatible existing audio is copied; incompatible audio is converted only when the chosen container requires it. MP4/MOV subtitles use `mov_text`; MKV retains compatible subtitles or converts incompatible ones. Audio-only mixes follow the configured standalone WAV/MP3 format unless you request a supported output extension; separate audio-only tracks use M4A.

## Python sandbox (Prime)

Deepy Prime can write and run small Python programs when a result needs exact text, measured layout, numbers turned into pictures, or many similar elements: a poster title placed precisely over generated key art, a chart, a contact sheet with captions, a colour palette, subtitles timed from a transcription, a title card that animates frame by frame. Models paint; Python measures, places, labels and animates. Finished images, audio and clips join the Gallery like any other result, with their generation details.

Those programs can also call WanGP's own media tools, one by one, as Deepy would: scan a folder of photos and ask the visual inspector about each one, transcribe every clip, make a variation of each image, then report only the result. This keeps long loops out of the conversation, where each step would otherwise cost time and context, and each call still runs on the same loaded models with the same access rules.

The programs run in a separate interpreter on the CPU, inside a sandbox folder of the session (`documents/sandbox`). They can read Gallery media and the session's documents, write only inside that folder, and have no network, no child processes and no access to the GPU or to WanGP's models. Memory and running time are limited, and Deepy can restart the interpreter when an experiment breaks its state. On Windows the operating system itself refuses writes outside the sandbox folder; reads of other files are refused by the interpreter. Turn the sandbox off with **Configuration > Prompt Enhancer / Deepy > Python > Enable Python Sandbox** if you prefer Deepy to work with its media tools only.

Notes, tables, JSON and SVG files that a program saves as deliverables open as previews in the chat. With **App Development (Advanced)** set to **Web Apps** in the same tab, Deepy can also build interactive animations, games and three.js scenes that you play in WanGP and keep in the Apps gallery: see the Apps documentation.

## Advanced transcription (Prime)

Deepy uses Whisper by default. Choose a Nemotron model in **Configuration > Prompt Enhancer / Deepy > Transcription > Transcription Model** (**Nemotron Q4 Experts** is recommended) to let Deepy Prime transcribe recordings with NVIDIA Nemotron 3.5 Transcribe: who said what (up to eight speakers), when each word was spoken, subtitles with the speaker named at each change, a translation to or from English, and names or jargon spelled the way you give them. Ask for example "transcribe this interview with speaker names as subtitles" or "translate what they say into English". Deepy runs it as a normal WanGP generation, so it waits in the queue like other jobs, the model downloads on first use (about 19 GB for **Q4 Experts**, which suits 32 GB of RAM, about 33 GB for **INT8 Experts**, best with 64 GB), and the transcribed audio joins the Audio gallery with the transcript beside it: a `.txt` file with one line per turn, its time span and its speaker, plus `.srt` subtitles. Once a Nemotron model is chosen, **Names and Jargon** takes the names, places, products and terms that come up in your recordings, separated by commas or lines: Deepy passes them to every transcription, besides the terms it picks for each recording. Speaker numbers tell the voices of one recording apart; Deepy cannot tell who a speaker is unless the recording or you say so: once you tell it who is who, it can name them in the transcript. Ask for word-level timing when you need it (one timed line per word), or for a copy of a video with the transcript or translation as subtitles that players switch on and off; Deepy reads long transcripts part by part without needing file access. For a quick transcript without speakers, Deepy keeps using its fast Whisper transcription.

## Web search (Prime)

Off by default. With **Configuration > Prompt Enhancer / Deepy > Enable Web Search**, Deepy Prime can search the web for facts, dates, names, current events, references and documentation it does not know, look at news, image and video results, and read the text of a public page before quoting it. This helps with prompts about real subjects (a landmark, a breed, a product, a period style) and with questions whose answer changes over time. Deepy names the pages it used so you can check them.

Searches go to free public search engines (DuckDuckGo, Bing, Brave, Google, Mojeek, Startpage, Wikipedia and others, whichever answer) directly from this machine, without an account or API key, so your queries leave the computer like a search typed in a browser. Deepy treats everything it reads on the web as information to weigh, never as instructions, and tells you when a page contains directions aimed at an assistant. It can only open public web addresses, never local or private network ones. Public engines occasionally rate-limit or change; Deepy then waits or falls back to another engine.

Deepy can also download a file from the web (a reference picture, a sound, a clip, a PDF) into the `documents/web` folder of the session, where it can inspect it, use it as a generation input or add it to the Gallery when you ask. Downloads are limited in size, are never run or unpacked, and Deepy says where each file came from: the rights belong to the file's owners, so use them as you would any file you save from a browser.

Deepy can also find and install LoRAs. Ask for a style, a subject or an effect ("make it look like a watercolor", "add a dolly zoom") and it searches Civitai and Hugging Face for LoRAs made for the model it is about to use, tells you which ones fit, with their trigger words, size, popularity and page, and installs the one you choose into that model's LoRA folder, ready to use in the next generation. Downloads and LoRA installs show their progress in the chat and the status bar. Searching needs no account. Downloading from Civitai requires a Civitai API key, which you create in your Civitai account settings (Account settings, then API Keys) and paste into **Configuration > Prompt Enhancer / Deepy > Civitai API Key**; it is sent to civitai.com only and kept in your operating system's credential manager, like the notification destinations, unless you choose to store it in plaintext. Hugging Face needs no token for public LoRAs, which is nearly all of them; the optional **Hugging Face Token** field, stored the same way, is only for LoRAs in gated or private repositories (a wrong token blocks public downloads too until it is corrected). LoRAs carry their authors' licenses, so check the page Deepy names before using one commercially; adult-rated LoRAs are left out unless you ask for them.

Sites that set cookies (consent choices, preferences) get them back on later visits during the same session, and the jar is saved with the session. For a site where you are signed in, export your browser's cookies as a `cookies.txt` file (browser extensions do this), put it in the session documents and ask Deepy to import it: Deepy then reads and downloads from that site as you, without ever seeing your password. Cookie values are never shown to Deepy or in its answers, and you can ask it to list or clear the jar at any time. Pages read with your cookies may contain your private information, so import cookies only for the sites a task needs.

## Plugin workflows (Prime)

Enabled Plugins can offer Deepy Prime their own workflows. With the bundled Media Flow Plugin, Deepy can process existing videos and images with Media Flow's ready-made templates: remove blur, compression artifacts or a stylized color grade, convert SDR to HDR, add detail, outpaint to a new aspect ratio, upscale, or add frames for smoother motion. It handles what single post-processing treatments do not: videos of any length, processed in chunks with the original audio kept, and whole folders, lists or wildcard patterns as one batch.

Describe the job, for example "remove the blur from @documents/interview.mp4", "upscale every video in @test/clips by 2 as one resumable batch" or "outpaint this clip to 16:9", or name the Plugin ("with Media Flow"). Deepy picks a Plugin when the request fits it, lists the Plugin's templates for the task, chooses the recommended one unless you name another, and can tell you which settings a template accepts.

- **Progress:** each Plugin action shows a card named after the Plugin, such as "Media Flow: Run", with live progress (files done, chunk of the current file) in the card and on the status bar.
- **GPU:** while a Plugin action runs, Deepy unloads its own model so that the action's generations have the GPU, and reloads it to answer.
- **Stopping and resuming:** the chat **Stop** button stops a Media Flow run after its current chunk. Ask Deepy to run the same job again and it continues where it stopped, also after a restart.
- **Outputs:** results go next to each source when Deepy may write in that folder, otherwise into the outputs folder (`media_flow`), or where you ask. Deepy links them in its answer. Its usual file access rules apply: it can only process media it may read and write into folders it may write.
- **One run at a time:** Media Flow runs one process at a time, shared with its tab. If the tab is already busy, Deepy says so, and can report that run's progress or stop it when you ask.
- **Python:** Python scripts can call Plugin workflows too, for example to process a selection of files found by a script.

Only Plugins that offer workflows to Deepy appear, and only while they are enabled; changes take effect after a restart. Deepy Zero does not use Plugin workflows. External MCP agents connected to WanGP can use the same workflows.

Plugin developers: see [Plugin toolboxes for agents](PLUGINS.md#plugin-toolboxes-for-agents) in the Plugin guide.

## Everyday workflow

1. Select a suitable template for each generation tool.
2. Import or select any source media.
3. Describe the result you want and any details that must be preserved.
4. Review the generated media and ask for refinements in the same conversation.

Deepy uses the configured template for each generation task. Zero uses the selected template directly. Prime also starts with the selected template and looks for another compatible model when you request model selection, name a model or family, request a declared speciality such as infographics, or the current template cannot perform the task.

You can override supported values in a request, including width, height, frame count, audio duration, FPS, inference steps, seed, and LoRAs. Put model-specific choices that Deepy cannot override into a linked template.

## Work with media

### Import and attach media

In Gradio, either use the chat **+** button or open **Media Info / Late Post Processing / Import Media > Import Media to Galleries**. In the Web app, use the chat **+** for media needed by the next request, or import files from a gallery. The chat **+** accepts images, videos and audio.

Files waiting for the next request appear as thumbnails above the message box: click one to view it, or its **×** to remove it. Sending the request consumes those attachments, and your message keeps them above its text, so you can see what each request used without searching the gallery. Removing an attachment from the composer does not delete its workspace file.

Importing a file that is already in the gallery moves it to the end of the gallery, beside your latest creations, instead of jumping back to its old place.

### Select media and refer to it naturally

Select a gallery item, then use phrases such as:

- `edit this image`
- `inspect the selected frame`
- `transcribe the last audio`
- `use the previous video`
- `compare these images`

Deepy normally prefers the selected item for words such as `selected`, `current`, `this image`, `this video`, or `this audio`. It can also resolve recent results and descriptions such as `the robot dancing image`. Internal IDs such as `image_2` and `video_3` remain available when you need precision.

In Gradio, scrub a selected video before referring to `this frame` or `this time`. In the Web app, viewing or playing a video does not select a reference moment, so include the time or frame number in your request. Deepy asks for clarification when a reference is ambiguous.

When generating video from an image, the image can have different roles. A **start image** anchors the opening scene and composition. A **reference image** guides a subject's identity or appearance without fixing the opening frame, according to the model's supported mode. For example, “animate this photo” and “put this character in a new scene” use the source differently. A model may support one role, both, or other roles such as inserting frames at specified positions.

### Browse galleries

The image/video and audio galleries let you select, inspect, play, and download media. Selecting a tile marks it for your next request; it does not submit anything. The information pane shows prompts, model settings, dimensions, and creation details when available.

When the latest item is selected, new output is selected automatically. If you are reviewing an older item, Deepy preserves that selection and your position.

### Open media from the chat

Click a result thumbnail, or a link to an image, video or 3D asset in one of Deepy's replies, to look at it closely:

- In Gradio, images and videos of the image/video gallery open in that gallery's full-screen viewer, with its download, zoom and video frame buttons, and become the selected gallery item. 3D assets open full screen in the 3D gallery's viewer, with its animation, snapshot and download buttons. Leaving full screen brings you back to the conversation where you left it, with your unsent request. Images and videos that are not in the current gallery, and audio, open in a new browser tab.
- In the Web app, images, videos, audio and 3D assets open in the app's media viewer.

Links to other files, such as archives or documents, download them. To open images and videos in a new browser tab instead of the viewer, set the `wgp_config.json` key `deepy_gallery_media_viewer` to `false` while WanGP is stopped; 3D assets still open in the 3D viewer.

## Control work in progress

The chat and generation progress bars show what Deepy and WanGP are doing.

- **Pause** suspends Deepy's current turn without losing it. Running media generation and postprocessing pause at their next processing checkpoint; other tools finish before Deepy pauses. The Generate panel reflects the same paused state. Postprocessing retains its GPU state while paused.
- **Resume** continues the same turn and its paused media operation. You can resume from either chat or the Generate panel.
- **Stop** ends Deepy's current turn. Depending on the **Auto-abort** setting, it may also cancel or remove generation work started by Deepy.
- **Abort** cancels the active WanGP generation.

You can send a new instruction while Deepy is working. Use this to steer the current task, or queue the instruction for afterward when the interface offers that choice.

Deepy's generations run after the current task and ahead of the tasks you added to the queue yourself, so a request to Deepy does not wait behind a long queue. Several of them run in the order Deepy requested them.

**Compacted View of Thoughts and Actions** keeps long turns readable. A collapsed turn shows its latest useful statement, generated media, and final answer; expand it to review individual thoughts and actions. The preference is shared between Gradio and Web.

## Tool templates and defaults

Open **Ask Deepy > Settings** in Gradio for the complete settings panel. The standalone Web app provides the commonly used generation defaults and existing template choices.

### Prime model preferences

At the top of **Templates Settings used by Tools**, Prime offers **Speed** and **Model Size**, also available in the Web app. They apply when Prime chooses beyond your configured templates; an explicit model or template request takes priority.

- **Speed:** Fast (default) favors models with native acceleration or accelerator profiles. Standard favors ordinary generation without automatically adding an accelerator. No preference ignores speed.
- **Model Size:** Smaller (default) favors declared lighter variants; Larger favors full variants. No preference ignores size. These labels compare variants, not absolute GB, VRAM use or quality.

Prime first checks required inputs and the requested speciality, then prefers candidates matching both preferences over those matching only one. For example, an H3 request with a subject reference can select a compatible lighter Ref2VA variant and its recommended accelerator. An infographic request can find a model declaring that strength. If only some speciality terms match, Prime receives the missing terms and checks essential requirements before generating.

Fast uses a model's recommended accelerator settings when available. Natively accelerated models already use accelerated defaults. Your configured templates keep their settings; these preferences do not replace them on ordinary requests. Changes reach Prime as a hidden runtime update on the next turn, including with a remote LLM; no conversation restart is needed. Deepy Zero has no model-preference controls or injection.

### Generation defaults

Choose whether each tool uses dimensions, durations, and seed from its template or replaces them with Deepy's defaults:

- width and height
- video frame count
- audio duration
- seed (`-1` means random)

Choose **Use template defaults first** to prefer the selected template's values, or **Use the values below** to prefer your Deepy defaults. For Deepy Prime, if a generation request omits dimensions, duration, or seed, WanGP fills the missing values from your Deepy defaults before using the model's factory settings. Values already supplied in the request are preserved.

Most changes apply immediately. Click **Save Deepy Settings** to reuse them after restarting WanGP.

### Template choices

Deepy has templates for:

- Image Generator
- Image Editor
- Video Generator
- Video With Speech
- Video Generator with Ref.
- Song Generator
- Speech From Description
- Speech From Sample

Deepy uses **Video Generator with Ref.** when images or videos supply subject identity, appearance or motion. A start image instead fixes the opening scene and uses the regular video template. Available reference templates are MiniMax H3 Ref2VA Pruned with its eight-step accelerator (default), LTX-2 2.3 MSR V2 Distilled 1.1, and Vace Fusionix. H3 accepts image and video references; the other templates accept image references. Deepy checks model support when combining references with other inputs.

WanGP includes built-in templates. To reuse your own model setup:

1. Configure normal WanGP generation settings.
2. Save them as a preset: in the preset list at the top of the generator (beside **Apply** and **Save**), type a name, press **Save** and
   confirm. Keep that preset selected.
3. Open Deepy settings.
4. Click **+** beside the Deepy tool that should use it.
5. Confirm the link and save Deepy settings.

The link follows later changes to that user settings file. The trash action removes the live link and returns to the previous or built-in template. If a linked file is deleted, Deepy returns to the tool's default template.

## Sessions and workspaces

Use saved sessions when you want to continue a conversation after restarting WanGP. Workspaces organize the media shown in the galleries.

### Choose a session mode

In **Ask Deepy > Settings > Sessions**, choose:

- **Disabled**: one temporary conversation. Reset clears it. Use this for quick, disposable work.
- **Multisessions with selectable Workspace**: save conversations while choosing which general gallery workspace each one uses. Use this when several projects share media collections.
- **Multisessions with dedicated Workspace**: give every saved conversation its own gallery workspace. Use this for self-contained projects and the clearest separation between jobs.

Save the setting and restart WanGP if Deepy has already started. In a dedicated workspace, the first request names the session and workspace. You may import media before that request; it is kept under **New Deepy session** until the first request supplies a name.

Sessions save automatically. There is no separate Save Session action. A saved session includes its conversation, completed actions, displayed results, workspace association, and media references. When resuming a long session, the conversation may appear before Deepy finishes preparing it.

Saved sessions use readable folders based on their names, with a numbered suffix only when needed. Duplicates and imported sessions follow the same rule. On upgrade, old identifier-named folders are renamed once, keeping the session identity and workspace links intact. Locked folders, folders that cannot be renamed, and folders used as an explicit dedicated workspace location stay in place and remain usable; later startups do not retry them. Renaming a session afterwards changes its displayed name and keeps its folder unchanged.

A one-time corrective upgrade handles folders previously skipped because an old session lock file remained. Locks belonging to processes that have exited no longer block renaming. Active sessions are still skipped, and the startup warning explains the reason.

Unavailable media do not prevent resuming a conversation or opening its gallery. When a saved session includes a copy of an unavailable source file, Deepy uses that copy.

In Gradio you can resume, rename, duplicate, export, import, or delete sessions. In Web, use the session selector in the top bar. In CLI, use `/sessions` and `/resume <ref>`. Wait for active or paused work to finish before switching sessions. If an automatic or manually entered session name is already used, Deepy adds an available number, such as **My video (2)**. Names differing only in capitalization count as duplicates.

### Keep links or copy media

The **Gallery Media** session setting controls portability:

- **Keep links to Gallery files** saves disk space, but the original files must remain at their recorded locations.
- **Copy Gallery files into each session** uses more disk space but makes the session independent of the original gallery files.

### Organize workspaces

A gallery workspace remembers gallery contents, media order, selected items, and the active gallery tab. Shared storage keeps references to existing files. Dedicated storage keeps independent media in a folder derived from the workspace name. Choose the default for new workspaces and the Workspaces Root in **Configuration > Outputs**; convert existing workspaces with the manager's **Shared Storage / Folder Storage** button. Conversion to a dedicated folder copies media and known companion files, retaining originals.

Ordinary workspaces use a folder icon; session-owned and MCP agent workspaces use a robot icon. A **Folder** suffix indicates dedicated storage. Renaming keeps the folder stable, and deleting a workspace retains its media. Session ownership and media storage are independent options. Deepy's output filesystem mounts follow its assigned workspace; explicit selected-folder and Read Everywhere grants still apply. External MCP actions can visibly switch the main gallery without changing Deepy's association. **Configuration > Agents** controls MCP assignment; workspace visibility and queue cancellation still follow the current shared-user policy.

In Gradio, the magnifier beside the workspace selector opens the full workspace viewer. Use it to:

- inspect every item, including older media hidden by the gallery display limit
- select multiple items with Ctrl/Cmd-click, Shift-click, or rectangle selection
- reorder media or move a selection to the start or end
- eject items from a workspace without deleting their files
- copy items to another workspace
- download selected items as a ZIP
- import more media
- permanently delete selected files after confirmation

The lock protects a workspace from automatic archiving. The broom sets an inactivity period for archiving old, unlocked workspaces at WanGP startup. Archiving hides the workspace but leaves its media files in place. There is currently no restore button, so protect any workspace you expect to revisit.

See [Gallery Workspaces](WORKSPACES.md) for the relationship between workspaces and galleries, the complete workspace-manager workflow, synchronization rules, dedicated Deepy workspaces, safe deletion, archiving, restoration, and backups.

### When a Plugin asks Deepy

Plugins can send Deepy requests of their own, with media, for example to edit the selected picture and turn it into a video. The bundled **Sample App** shows one once enabled in **Apps & PlugIn Manager**: its **Ask Deepy to add a red hat and to make a singing video** button sends the selected picture with that request and puts a download link to the video beside the button.

- **Waiting:** a Plugin's request starts when Deepy and the generation queue are free.
- **While it runs:** Deepy shows the Plugin's own conversation, its requests marked with the Plugin's name. The gallery shows the workspace the Plugin chose: a new workspace that stays in the list so you can browse the results later, a temporary one removed when the Plugin is done, a workspace the Plugin keeps for its project, or the one you had open, in which case the results join your gallery.
- **Control:** you can follow, pause and stop the Plugin's work as any Deepy turn, and a message you send goes to the Plugin's conversation. Changing conversation or workspace is refused until the Plugin's request ends.
- **Afterwards:** your conversation and workspace come back, also when saved sessions are disabled; Deepy reloads your conversation as when resuming a session. Plugin conversations are kept by their Plugin and do not appear in the session list.

Plugin developers: see [Running Deepy Jobs](PLUGINS.md#running-deepy-jobs) in the Plugin guide.

## Use Deepy on a phone

The Web app has tabs for **Chat**, **Image/video**, **Audio**, and **Settings**. Replies, results, selections, and progress stay synchronized with Gradio when both connect to the same process.

For an app-like view on iPhone, open the Deepy address in Safari and choose **Share > Add to Home Screen**. On Android, use **Install app** or **Add to home screen**. The connection overlay tells you when the phone loses contact with WanGP and disappears after reconnection.

### Protect network access

See [Authentication, HTTPS, and Reverse Proxies](AUTHENTICATION.md) for the shared Gradio/Deepy password login, session behavior, and network protection.

### Set up HTTPS

See [HTTPS certificates](AUTHENTICATION.md#https-certificates) and [reverse proxy setup](AUTHENTICATION.md#hosting-behind-a-reverse-proxy), including `--public-url` for RunPod, Nginx, Traefik, and Cloudflare.

## Voice input and transcription

In Gradio and desktop Web, press the microphone beside **Send**, speak, then press it again to transcribe. The text remains editable before you submit it. On smartphones, the standalone Web app uses the phone keyboard's native dictation instead.

In **Configuration > General**, choose the microphone transcription mode:

- **Auto** uses the GPU when it is available and the CPU while the GPU is busy.
- **CUDA** waits for the GPU and is usually faster once it starts.
- **CPU** avoids competing for GPU memory.
- **Disabled** hides the microphone controls.

Choose **Voice transcription language** to improve short recordings, or leave it on **Auto**. You can override the language for one launch with `--deepy-voice-language es`, `en`, or `auto`. Whisper files download on first use.

Dictating a request is different from asking Deepy to transcribe an existing audio or video file. For media transcription, select or attach the file and ask for segment timestamps, word timestamps, or a particular audio track.

## CLI mode

Launch the terminal interface with:

```powershell
python wgp.py --ask-deepy
```

Add `--config`, `--deepy-sessions-dir`, or `--output-dir` when you want to use custom locations. Do not combine `--ask-deepy` with `--deepy-server`.

Prompt entry:

- `Enter`: send
- `Alt+Enter` or `Ctrl+J`: insert a newline
- `Ctrl+S`: stop the current turn

Useful commands:

| Command | Purpose |
|---|---|
| `/add <path>` | Add and select an image, video, or audio file. |
| `/image <path>`, `/video <path>`, `/audio <path>` | Add and select a specific media type. |
| `/list [all|image|video|audio]` | List known media. |
| `/select <ref>` | Select media by ID, list number, or part of its name. |
| `/selected` | Show the current selection. |
| `/time <seconds>` | Choose a time in the selected video. |
| `/frame <index>` | Choose a zero-based frame in the selected video. |
| `/size <WxH>`, `/frames <count>`, `/duration <seconds>`, `/seed <value>` | Override generation defaults. |
| `/templates [tool]` | List available templates. |
| `/template <tool> <variant>` | Select a template. |
| `/sessions`, `/resume <ref>`, `/new` | List, resume, or start saved sessions. |
| `/reset` | Clear the temporary conversation or start a new saved session, according to the session mode. |
| `/help` | Show the complete command summary. |
| `/quit` | Exit. |

For example:

```text
/video E:\media\my_clip.mp4
/frame 120
inspect the selected frame and tell me whether the subject is centered
```

## Important configuration choices

The settings under **Configuration > Prompt Enhancer / Deepy** determine Deepy's capabilities, quality, speed, memory use, and access to your files. The recommendations below are starting points; change them when your hardware or workflow has a different priority.

### Prompt Enhancer / Deepy LLM Engine

This selects the language and vision model that understands requests, plans work, and writes responses.

- **Qwen3.5VL Abliterated 4B** starts quickly and uses the least memory, but is less reliable with long instructions and multi-step decisions. **Recommended for:** Deepy Zero on limited hardware and simple, direct requests.
- **Qwen3.5VL Abliterated 9B** understands more complex instructions and media better than 4B, with higher VRAM and RAM use. **Recommended for:** the best general Deepy Zero experience when it fits comfortably.
- **Qwen3.8VL Uncensored 27B** offers the strongest local planning, knowledge and image understanding for Deepy Prime, but needs considerably more memory and takes longer to load. On GPUs with less than 16 GB of VRAM, choose its **Bonsai 2 PTQ1_0** quantization (see below), whose weights take less than 6 GB. **Recommended for:** local Prime and complex multimedia projects.
- **Qwen3.8VL Flash-Next 125B** is much larger than the VRAM of a PC: it keeps in VRAM what the current conversation uses and leaves the rest in RAM and on the drive, so it stays usable on 12 GB GPUs and gets faster as a conversation goes on. It needs an 80 GB download on a fast SSD and about 64 GB of RAM: with less, it spends its time reading the drive. Long prompts take longer to read than with the 27B. **Recommended for:** local Prime on a PC with plenty of RAM, when you want the largest local model.
- **A remote LLM** avoids loading the language model on the WanGP GPU and may provide stronger reasoning, but adds network latency and sends conversation content to the configured provider. Remote engines require Prime. **Recommended for:** Prime when local memory is insufficient or a supported remote engine is preferred. Review [Remote LLMs](REMOTE_LLMS.md) before using one with private media or instructions.

Changing the engine can require new model downloads and a runtime reload.

### Qwen LLM Quantization

Quantization trades model quality for lower VRAM/RAM use. It is shown only for local Qwen engines.

For Qwen3.5:

- **Quanto Int8** uses more memory but generally preserves better quality. **Recommended for:** most systems that can run it comfortably.
- **GGUF Q4** uses less memory and can be faster when compatible kernels are installed, with some quality loss. **Recommended for:** systems where Int8 does not leave enough memory for generation models.

For Qwen3.8 27B:

- **GGUF Q4** has the highest quality and memory use. **Recommended for:** quality-first work when it fits without forcing excessive unloading. Usually this will require a 24 GB VRAM GPU.
- **GGUF IQ3_S** is the middle ground for quality and memory. **Recommended for:** most local Prime installations. You may run this version with a 16 GB VRAM GPU.
- **GGUF Q2** has lower memory use and a larger quality loss. **Recommended for:** this version can be also used with a 16 GB VRAM GPU but the advantage over Q3 is that you will be able to expand the context window.
- **Bonsai 2 PTQ1_0** has the lowest memory use. **Recommended for:** GPUs with less than 16 GB of VRAM. It needs the GGUF kernels 1.0.22 or newer; without them, a first launch below 16 GB stays on Qwen3.5 4B and WanGP suggests Q2 if you choose Qwen3.8.

On first launch, WanGP sets up Deepy Prime on the Qwen3.8 version that suits your NVIDIA GPU, with nothing to configure, so that Showrunner works right away: Q4 from 24 GB of VRAM, IQ3_S from 16 GB, Bonsai below that, with a 64K context from 10 GB of VRAM and 32K below. Settings you have already saved are never changed. You can pick another version, but if it is less compressed than the one suited to your GPU, or if a Speculative Decoding you chose pushes the model out of VRAM, WanGP warns you that Deepy may perform poorly: in **Configuration > Prompt Enhancer / Deepy**, at the first request of a Deepy chat, and in Showrunner. The same places tell you when Speculative Decoding is Disabled although your version of Qwen3.8 fits in VRAM with MTP on your GPU, as Deepy could then run about twice as fast.

Deepy keeps as much of its local model in VRAM as fits beside its context cache. When the whole model does not fit, it streams the remaining layers from RAM at each step: it keeps working, but it answers more slowly the more layers are streamed, and it frees VRAM by itself when an image is inspected or another component needs memory. Deepy always uses the fastest kernels that are installed; there is no decoder setting to choose.

**Configuration > Performance > Language Models CPU Compute** speeds up this case: while Deepy writes, the CPU processes the layers that stay in RAM instead of sending them to the GPU at each word.
- **Auto** (default) chooses for your PC: it measures whether your CPU or the GPU transfers are faster for the model's format, and the CPU processes part of the layers left in RAM while the GPU receives the others. Deepy then answers about twice as fast.
- **On** makes the CPU help even when it measured slower, to compare both ways on your PC.
- Deepy's language model always uses Memory Profile 0, whatever the profile of your other models: while it writes, the layers it streams are kept in your Reserved RAM, which goes back to the other models when you switch. This setting only decides whether the CPU helps.
- Models that are expensive to process on a CPU (2-bit and ternary weights such as IQ2 and Bonsai) keep streaming their layers to the GPU.
- The setting has no effect when the model fits in VRAM. Reading prompts and images still uses the GPU.
- While Deepy writes, the CPU is fully busy. Choose **Off** if other work needs the CPU.

If Deepy frequently unloads other models, runs out of memory, answers slowly, or leaves too little VRAM for media generation, select a smaller model or lower quantization before reducing the context window drastically.

### Speculative Decoding

Speculative decoding can generate Deepy's text faster by predicting several tokens ahead. More draft tokens can improve speed for some requests but use more VRAM, and the fastest setting varies by model and workload.

Choose a method and **Number of Tokens** on the row above **KV Cache Quantization**. Only compatible methods appear; switching to an incompatible model or quantization resets the method to **Auto**.

**Number of Tokens** starts on **Auto** when you pick a method: you choose how Deepy predicts and WanGP chooses how far. Pick a number instead to use exactly that many draft tokens on every prediction.

- **Auto** enables MTP on supported models when it speeds Deepy up: when the model fits in VRAM with it, or when part of the model has to stream from RAM anyway (MTP then predicts several tokens for each pass over the streamed weights). It stays off when MTP would push a model that otherwise fits in VRAM partly out of it, and on GPUs with 8 GB or less. From 13 to 16 GB, it enables MTP only when the whole model fits in VRAM with it. On Qwen3.8 27B, it uses 4 draft tokens from 24 GB of VRAM, or from 12 GB with Bonsai, and 2 below. With 4 draft tokens, it predicts up to 7 while Deepy writes text that is easy to predict (tool calls, summaries, lists), which makes those parts faster, and keeps 4 on the rest. How readily it does so follows what the longer prediction costs on your GPU, measured while Deepy runs. For GGUF models, WanGP reads the checkpoint, context size and KV Cache Quantization to decide; for other quantizations Auto needs at least 12 GB for Qwen3.5 9B. The token count is automatic, and the console reports the choice Auto made.
- **Disabled** adds no draft-model VRAM and disables the token control. Use it when memory is tight or prediction is slower for your workload.
- **MTP** keeps MTP on whatever the VRAM, on Qwen3.5 9B and Qwen3.8/Bonsai 27B. With **Number of Tokens** on **Auto** it uses the draft count Auto would choose for your model and GPU, including the longer prediction on easy text, and the console reports that count. It also supports a fixed 1–8 draft tokens; more are not always faster.
- **DSpark** supports 1–7 draft tokens on Qwen3.8/Bonsai 27B when Triton and FlashAttention 2 are installed. **Auto** uses its recommended count, as it does for DFlash2 and DSpine.
- **DFlash2** supports 1–7 draft tokens on Qwen3.8 Q2, Q3 and Q4, and 1–5 on Bonsai PTQ1, when Triton and FlashAttention 2 are installed. WanGP selects the Qwen drafter for Q2/Q3/Q4 and the Bonsai-adapted drafter for PTQ1 automatically.

Each method label includes its estimated additional VRAM: approximately 0.5–1 GiB for MTP and 4–5 GiB for DSpark or DFlash2 at roughly 32K context. Auto adds between zero and the MTP cost. Longer context, draft count, and model choice affect actual usage. Compare representative prompts before choosing a prediction method: Disabled can be fastest, especially with Bonsai.

This setting changes response speed, not the quality or speed of image, video, or audio generation.

### Prompt Enhancer usage and sampling

These controls affect the shared local language model's output style. They are normally best left at their defaults.

- **Prompt Enhancer Usage** chooses whether normal WanGP generations enhance prompts automatically or only when you click the enhancer button. The manual button is available in both modes when the enhancer is enabled. Automatic enhancement can add detail but can also reinterpret carefully written prompts. **Recommendation:** use **Manual Button Only** when prompt fidelity matters; use **Manual Button + Automatic on Generation** when you routinely start from short ideas.
- **Sampling Temperature** controls creativity. Lower values are more consistent and literal; higher values are more varied but more likely to wander. **Recommendation:** keep the default `0.6`; try `0.3-0.5` for precise instructions or `0.7-0.9` for ideation.
- **Sampling Top-p** controls how broad the model's word choices can be. Lower values narrow responses; higher values add variety. **Recommendation:** keep the default `0.9` and adjust temperature first.
- **Randomize Prompt Enhancer Seed** allows different wording and ideas on repeated requests. Disabling it improves repeatability when the prompt and settings are unchanged. **Recommendation:** keep it enabled for creative work; disable it when comparing configuration changes.

### Deepy

This chooses the assistant level:

- **Disabled** removes the conversational assistant while leaving ordinary WanGP generation available.
- **Deepy Zero** prioritizes speed and direct execution with the selected templates.
- **Deepy Prime** adds planning, model discovery, durable workspace files, and optional external services for multi-step work.

**Recommendation:** use Zero for isolated operations and Prime when the request requires multiple dependent steps, model comparison, project files, or external tools. See [Choose Deepy Zero or Deepy Prime](#choose-deepy-zero-or-deepy-prime) for engine requirements.

### Deepy VRAM Loading Mode

This controls how long a local Deepy model stays in GPU memory. It is not shown for a remote engine.

- **Unload from VRAM as soon as possible** frees memory after Deepy becomes idle, but the next request must reload the model and starts more slowly. **Recommended for:** GPUs that need nearly all available VRAM for image or video generation; this is the safest default.
- **Unload if VRAM is requested by another WanGP component** keeps Deepy responsive between requests but gives its memory back when another operation needs it. **Recommended for:** most systems with enough VRAM to hold Deepy during normal browsing and light generation.
- **Always loaded in VRAM** gives the fastest Deepy response times. A generation still gets the memory it needs: it takes the parts of Deepy it requires, and Deepy's conversation memory only when nothing else is left (it is then saved to RAM, not lost). Deepy's next answer brings them back without starting the conversation over. **Recommended for:** systems where you alternate between Deepy and generations, or where Deepy asks for progress while a generation runs.

In every mode, quick tools that need little memory (image and video inspection, transcription, masks, vocal removal) keep Deepy loaded while they run and take from it only the memory they need, so Deepy continues right after them. With **Always loaded in VRAM**, a prompt enhancement runs beside Deepy's conversation instead of interrupting it.

These partial releases need the *MMGP Optimized VRAM Allocator* (the default, see [CLI.md](CLI.md#vram-allocator)); with PyTorch's allocator, Deepy is released whole as described above. If generations fail to fit after changing this setting, move one level toward earlier unloading.

### Context Window Tokens

The context window is how much recent conversation, tool output, and project state the local model can consider at once. A larger value helps Deepy follow long projects and retain more detail before summarization, but its KV cache consumes more VRAM and long context can take longer to prepare.

- **20K** is the minimum, suitable for short Zero conversations, and uses the least memory.
- **32K** is the practical minimum for Prime with **Summarize**.
- **48K or more** supports **Summarize with Thinking** and longer projects, with increasing memory cost.
- **Very large windows** are useful only when the extra history materially improves the workflow and the displayed KV-cache estimate fits comfortably alongside your generation models.

**Recommendation:** start around 20K for Zero, 32K for Prime, or 48K for Prime when using compaction thinking. Increase it only when Deepy summarizes too often or loses relevant recent detail.

### KV Cache Quantization

The KV cache holds the active conversation context in GPU memory. Quantizing it reduces that memory cost without changing the model checkpoint.

- **Auto** chooses for your installation: Q4 when the GGUF and Triton kernels are installed and the GPU is an RTX 30XX or newer, fast INT8 with the GGUF kernels alone, and BF16 otherwise. **Recommended for:** most users.
- **Disabled (BF16)** uses more VRAM and avoids cache quantization. **Recommended for:** quality-sensitive troubleshooting, or systems where INT8 cache performance is worse.
- **INT8** uses about half the KV-cache VRAM and makes larger context windows practical. **Recommended for:** long Prime sessions or tight VRAM, provided the installed kernels support it efficiently.
- **Below INT8** (INT8 keys with Q4 values, Q6, Q6 keys with Q4 values, Q4.1, Q4) shrinks the cache further, by a quarter to almost a half of the INT8 size. The first page of the conversation and the most recent tokens always stay in INT8; only the older middle of the context uses the smaller format. Q6 and INT8 keys with Q4 values are practically indistinguishable from INT8 on long texts; Q4.1 and Q4 cost a little precision on the oldest part of the context. Decoding speed is unchanged; reading a long prompt is 5 to 12 percent slower. **Recommended for:** very long Prime sessions on cards where the cache, not the model, is what no longer fits.

**Exact Sink and Recent Tokens** appears with a cache below INT8 and chooses which tokens keep the INT8 precision: the first page plus the last 1024 tokens (recommended), the first page plus the last 256 tokens, the last 1024 tokens alone, or none for a uniformly quantized cache.

The context-window label displays an estimated cache size. Treat that estimate as VRAM reserved before image or video model requirements.

### KV Caches Kept in RAM

When you go back to a conversation you left (your chat after a Showrunner request, or a production you worked on last), Deepy continues from its saved cache instead of rebuilding its context, so it answers in seconds rather than up to a minute. Each conversation keeps one cache, restored when it next needs Deepy; the oldest makes way once the limit is reached, and that conversation simply rebuilds its context on its next request.

- **2** (default) keeps your chat and your last production ready.
- **More** keeps more productions ready, at the cost of RAM: 64K tokens of context take about 1.2 GB in Q4, 2.1 GB in INT8 and 4 GB in BF16.
- **Disabled** frees that RAM, and every conversation you come back to rebuilds its context. **Recommended for:** low RAM.

### Compaction Type When Context is Full

Compaction decides what happens when the live conversation no longer fits in the context window.

- **Discard Oldest Entries** simply removes the oldest context. It is fast but may lose goals, decisions, references, and unfinished plans. **Recommended for:** short, disposable Zero chats only.
- **Summarize** condenses older work while preserving important goals, decisions, completed actions, file references, and next steps. It requires at least 32K context. **Recommended for:** most long sessions and all local Prime use.
- **Summarize with Thinking** lets Deepy reason before creating the summary, improving preservation in complicated projects at the cost of more time and a minimum 48K context. **Recommended for:** long Prime workflows with many dependencies, not routine one-step requests.

Summaries help but cannot guarantee perfect recall. Keep critical specifications in your request (one longer than about a fifth of the context window is summarized too) or in a workspace document when mistakes would be costly.

### Faster Context Reading

Available with the local Qwen models when **KV Cache Quantization** is INT8 or below. Before it can answer, Deepy reads everything new in the conversation: your message, tool results, documents, or a whole conversation you come back to. When enabled, that reading is up to twice as fast, and the gain grows with the length of the conversation.

- **Disabled** reads with the most reproducible arithmetic and no extra memory. **Recommended for:** cards where graphics memory is tight during long conversations.
- **Enabled** favors speed. It takes up to about 140 MB of graphics memory while reading, whatever the length of the conversation, released afterwards. The wording of answers can differ from what the disabled setting gives; quality measured the same in our tests. **Recommended for:** long conversations and work with many tool results.

### Skip Low-Ranked Experts

Available with Qwen3.8 Flash-Next. When enabled, the model leaves out the least important part of its work when that part is not already in graphics memory. Answers come faster, about 15-30% on a 32 GB card in our tests, and slightly less precise.

- **Disabled** keeps full quality. **Recommended for:** careful work, where a precise answer matters more than speed.
- **Enabled** favors speed. **Recommended for:** long answers and exploration, where you would rather have the text sooner.

### Repetition Penalty

When enabled, this reduces rambling and repeated phrases across Deepy's responses and planning. It costs roughly 10% of local text-generation speed.

- **Enabled** produces cleaner long responses and plans. **Recommended for:** normal use, especially Prime.
- **Disabled** is slightly faster but can become repetitive. **Recommended for:** maximum-speed experiments or troubleshooting only.

### Filesystem access

Under **Configuration > Prompt Enhancer / Deepy**, Prime groups its options into **Filesystem**, **Guidance**, **MCP Servers**, **Python**, **Internet Access**, and **Transcription** tabs. Deepy Zero shows only **Filesystem**. The **Internet Access** tab contains Web Search, the Civitai API key, the Hugging Face token, and their OS Credential Manager options. The **Transcription** tab chooses the transcription model (Whisper only, or a Nemotron model for Advanced Transcription) and, for Nemotron, your names and jargon.

Filesystem access determines which local files Deepy can inspect or create. Use the smallest scope that supports the task.

For Zero:

- **Disabled** prevents filesystem tools. **Recommended for:** generation-only use.
- **Read Outputs + Selected Folders** lets Deepy inspect existing files without changing them. **Recommended for:** analysis, lookup, and media-selection workflows.
- **Read / Write Outputs + Selected Folders** allows file creation and changes in the configured locations. **Recommended for:** tasks that explicitly require file editing or organization.

For Prime, its session workspace always remains available for drafts and project files. The outside-workspace choices are:

- **Outputs only (read)** keeps external access narrow. **Recommended for:** most users and projects that can keep working files in the session workspace.
- **Read outputs + selected folders** adds read-only access to chosen source folders. **Recommended for:** projects that consume an existing media or document library.
- **Read + create in outputs; read/write selected folders** allows new output files and full work in selected folders. Existing files under WanGP output folders remain protected from overwrite, rename, move, and deletion. **Recommended for:** only projects that need Deepy to maintain files outside its workspace.

### Additional Filesystem Folders

Add one folder per line to make specific project locations available. An optional alias gives Deepy a short, unambiguous name for the folder:

```text
"D:\My Media" media
E:\Projects\Current project
```

**Impact:** broader folders expose more filenames and content and, with write access, allow more changes. **Recommendation:** add the narrowest project folder possible instead of a drive root, give it a clear alias, and remove it when the project is finished.

### Read Everywhere

This allows Deepy to read absolute paths anywhere the WanGP process can access. It does not expand write access.

**Impact:** convenient references to arbitrary local files become possible, but Deepy can also read unrelated or sensitive files if asked. **Recommendation:** leave it disabled and use selected folders. Enable it temporarily only when read-only work genuinely spans many locations.

### Deepy Prime Guidance

The **Guidance** tab's **Deepy Prime User Guidance** field provides standing instructions applied to future interactions. Use it for durable project preferences such as quality priorities, approval points, naming conventions, or how to choose models.

**Impact:** broad or conflicting instructions affect every request and can make otherwise simple tasks less predictable. **Recommendation:** keep guidance short, specific, and reusable. Put one-off requirements in the request itself. Do not paste secrets into this field, especially with a remote engine.

### External MCP Servers

External MCP servers give Prime access to additional applications, data, or specialized tools.

**Impact:** they can make cross-application workflows possible, but may send prompts or files to another process or service and may perform actions under that service's permissions. A broken server can also delay or prevent its tools from being used. **Recommendation:** configure only trusted services needed for a real workflow, review their permissions and privacy terms, and remove unused entries.

### Allow Searching for Changed MCP Executable Paths

Some locally installed MCP programs place each version in a different folder. When this option is enabled and the saved executable disappears, Deepy may look in sibling version folders for the newest executable with the same filename.

**Impact:** local integrations survive routine upgrades more easily, but WanGP may run a newer version than the one originally configured. **Recommendation:** leave it disabled for tightly controlled environments; enable it for trusted local tools that update frequently and whose version-folder layout is stable.

Use Prime's session workspace for plans, drafts, and other files that belong to a long project. It offers the safest default place for durable working material and remains useful after older conversation content is summarized.

## Example requests

Focused requests for Zero or Prime:

```text
Generate a cinematic image of a robot violinist on a rainy Paris rooftop at night.
```

```text
Edit the selected image so the background becomes a neon alley while keeping the character identity. Use 8 inference steps.
```

```text
Transcribe audio track 2 from the selected video with word timestamps.
```

```text
Use the selected portrait and the last audio clip to make a talking video.
```

Multi-step requests suited to Prime:

```text
Generate a master image of a robot dancing beside a horse in a nightclub. Create a second image with the same subjects and setting but a different pose. Check identity and composition, revise if needed, then generate a transition between the two images.
```

```text
Create a portrait for an introduction video, generate a short speech explaining WanGP's capabilities, then combine the portrait and speech into a talking video.
```

## Practical tips

- State the final goal, the source media to reuse, and any details that must not change.
- Give an exact resolution such as **1920×1080** when size matters. Prime filters models by pixel budget and can suggest an available tiled two-phase mode. Missing or ambiguous limits use a shared 1080p budget (1920×1088 to include aligned presets). Deepy rejects generation requests above the active maximum or below the minimum; the global UI 3K/4K+ option does not bypass this check. Input fitting or block alignment may still affect final dimensions. The Presets / MP switch does not change an agent's requested size.
- Use Zero for a direct task and Prime for planning, model comparison, validation, or a chain of dependent actions.
- Ask Deepy which template, defaults, or LoRAs are active when you want to confirm the setup.
- Put recurring model-specific choices in a linked template instead of repeating them in every prompt.
- Specify a time, frame, or audio track when the source contains several possible references.
- Ask for word timestamps when segment timestamps are not precise enough.
- Use **Pause** to temporarily suspend work without abandoning the turn.
- Use saved sessions and copied gallery media when a project must remain portable after source files move.
- You can ask Deepy WanGP-specific questions instead of searching the manuals yourself.

---

> Applies to: Choosing, enabling, launching, configuring, and using Deepy Zero or Deepy Prime in Gradio, CLI, and the standalone Web app, including media, templates, Plugin workflows, sessions, workspaces, phone access, voice input, and network protection.
