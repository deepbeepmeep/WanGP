# Wan2GP Plugin System

This system allows you to extend and customize the Wan2GP user interface and functionality without modifying the core application code. This document will guide you through the process of creating and installing your own Plugins.

For configuration forms shared across browser windows, use the [shared form synchronization API](FORM_SYNC.md), including draft preservation and conflict handling. Generation drafts, such as the Sample App's Prompt Copy, stay local to the invoking browser.

## Discover Apps Included with WanGP

The app bar is the main way to move between **Media Generator**, **Mask Generator** and your installed apps. Each app has an icon and a name. **More Apps** lists only the apps that do not fit when the window is narrow. Selecting an app keeps the other labels and icons in place. The icon buttons on the right open **Configuration** (**Alt+C**), **Apps & PlugIn Manager** (**Alt+P**) and **Infos** (**Alt+I**). Infos brings the guides together with **About** as its final page.

**Media Flow** and **Showrunner** come enabled with WanGP. Media Flow restores, upscales and processes long videos or folders; Showrunner plans and assembles complete productions with Deepy Prime. Each app keeps its yellow **New!** badge until its own tab has been opened, even if it was already enabled. Viewing or dismissing an introduction from the app bar does not clear the badge; visits are remembered across restarts. Select one to see an illustrated introduction, then choose **Keep It** to open it or **Remove It** to hide it from your app bar. Removal hides the app immediately and stops it loading at the next startup. Your choice is saved across restarts. The introduction also appears when you first open the app through its folder URL. After a visit, it remains available from the app's **info icon** beside its ID in **Apps & PlugIn Manager**.

Media Flow and Showrunner appear in the normal **Enabled Apps & Plugins** list, where you can reorder or disable them. Disabled apps appear under **Available Locally** on the right; these two are listed first with a subtle highlight. Their info icons open the illustrated information cards; **Featured** appears at the top right of these apps. Restoring an app already loaded in the current session works immediately; otherwise restart WanGP after enabling it. Showrunner requires Deepy Prime, configured in **Configuration → Prompt Enhancer / Deepy**.

Both apps have a **Full Screen** control. `/media-flow/` and `/showrunner/` on your WanGP address open the corresponding app directly in full screen. The address keeps that folder URL, which you can bookmark or share. **WanGP** returns to the same tab, preserving its form or editor. Expanding and returning reuse the existing page.

**Models Manager** is always installed. Open **Configuration → General → Manage Models & Disk Space**, beside **Selectable Generative Models**, or press **Alt+M**. Once disk usage is ready, **Models** lists generative models and finetunes, and **System** lists local Deepy and prompt enhancer variants, preprocessors, postprocessors, preview decoders and checkpoints reported by loaded plugins. Expand a family and select an entry to see its files. A file used by multiple models or variants stays **Shared** at every level and counts once per total. **Total Used** and the drive totals cover both tabs. **Delete Exclusive** removes only files that no entry outside the selected branch uses. The compact **Back to ...** link (or **Alt+M** again) returns to the screen you came from.

Apps and plugins use the same Plugin system described below. Developer APIs, Plugin identifiers and installation folders keep their existing names.

## Table of Contents
1.  [Plugin Structure](#plugin-structure)
    *   [Reference Plugins and Specialized APIs](#reference-plugins-and-specialized-apis)
2.  [Getting Started: Creating a Plugin](#getting-started-creating-a-plugin)
3.  [Plugin Distribution and Installation](#plugin-distribution-and-installation)
4.  [Plugin API Reference](#plugin-api-reference)
    *   [The `WAN2GPPlugin` Class](#the-wan2gpplugin-class)
    *   [Core Methods](#core-methods)
    *   [Deepy Tool Registration](#deepy-tool-registration)
    *   [Plugin toolboxes for agents](#plugin-toolboxes-for-agents)
    *   [Plugin skills](#plugin-skills)
    *   [Running Deepy Jobs](#running-deepy-jobs)
    *   [Plugin Web Apps](#plugin-web-apps)
5.  [Examples](#examples)
    *   [Example 1: Creating a New Tab](#example-1-creating-a-new-tab)
    *   [Example 2: Injecting UI Elements](#example-2-injecting-ui-elements)
    *   [Example 3: Advanced UI Injection and Interaction](#example-3-advanced-ui-injection-and-interaction)
    *   [Example 4: Accessing Global Functions and Variables](#example-4-accessing-global-functions-and-variables)
    *   [Example 5: Using Helper Modules (Relative Imports)](#example-5-using-helper-modules-relative-imports)
    *   [Example 6: Extending Deepy Prime and Deepy Zero](#example-6-extending-deepy-prime-and-deepy-zero)
6.  [Finding Component IDs](#finding-component-ids)

## Plugin Structure

Plugins are standard Python packages (folders) located within the main `plugins/` directory. This structure allows for multiple files, dependencies, and proper packaging.

The Sample App in `plugins/sample/` illustrates how to:

- Read and modify settings from the main form.
- Suspend generation and release VRAM for your own GPU work.
- Switch back to the main tab automatically.
- Start generation from a Plugin and track its progress.

A valid Plugin folder must contain at a minimum:
*   `__init__.py`: An empty file that tells Python to treat the directory as a package.
*   `plugin.py`: The main file containing your class that inherits from `WAN2GPPlugin`.

Community Plugin folder names should use the `wan2gp-` prefix, for example `wan2gp-stable-diffusion-1-4`.

Plugins may also include `plugin_info.json` for metadata shown in **Apps & PlugIn Manager**. Its optional `type` property can be a string or a list of strings. Missing `type` values default to `"app"`. Supported values are:
*   `"app"`: an application Plugin, usually with its own tab.
*   `"extension"`: a feature Plugin that does not add its own tab.
*   `"processor"`: a processing Plugin, such as a spatial upsampler today or future preprocessors/audio postprocessors.
*   `"model"`: a Plugin that provides model integrations.

### Optional Discovery Metadata

Any Plugin type can provide an `info_card` in `plugin_info.json`. This adds an info icon immediately after its ID in the first line of its enabled, available-local or downloadable card. The card does not enable the Plugin or clear its **New!** badge. Omit `info_card` to keep a simple row.

```json
{
  "name": "My App",
  "type": "app",
  "important": false,
  "info_card": {
    "tagline": "A Short Benefit",
    "description": "Explain what users can do with this Plugin.",
    "details": "Optional requirements or practical limits.",
    "image": "assets/preview.webp",
    "image_alt": "A useful description of the preview"
  }
}
```

All card fields are optional plain text. `description` defaults to the Plugin description. Images can use HTTPS, a file inside the Plugin folder, or a shared `icons/` path. Without an image, the card uses the app icon. Downloadable catalog cards can use HTTPS images; packaged local images become available after installation. The optional Boolean `important` flag adds a **Featured** badge at the top right of its card, places it first and highlights it in the right-hand available lists. It does not change the user's enabled-app order. Media Flow and Showrunner set it to `true`; Sample App and Motion Designer provide cards without that priority.

Model Plugins declare their model integration in `plugin_info.json`. They may omit `plugin.py` when they only provide handlers and metadata. Use:
*   `model_handlers`: a string or list of model handler module paths. Paths beginning with `.` or `./` are relative to the Plugin package root; absolute import paths are also supported. Each module must expose a `family_handler` object, matching WanGP's built-in model handlers.
*   `defaults`: the root folder containing model definition JSON files, equivalent to WanGP's built-in `defaults/`. Paths beginning with `.` are relative to the Plugin root.
*   `profiles`: the root folder containing built-in profile JSON folders, equivalent to WanGP's built-in `profiles/`. Paths beginning with `.` are relative to the Plugin root.

Example:
```json
{
  "name": "My Model Pack",
  "type": "model",
  "model_handlers": [".models.my_family_handler"],
  "defaults": "./defaults",
  "profiles": "./profiles"
}
```

### Reference Plugins and Specialized APIs

**Reporting checkpoints to Models Manager.** Implement `query_checkpoint_inventory()` on the plugin or its registered processor. It lists **all** supported downloads, even when the extension is turned off or another variant is selected. Return or yield one entry per distinct set of checkpoints. Iterate the choices your extension offers and reuse the same per-variant definitions as the downloader. The manager must be able to call this offline without mutating settings, loading weights, or downloading anything.

```python
def query_checkpoint_inventory(self):
    for label, model in self.MODEL_CHOICES:
        for precision in self.PRECISIONS:
            yield {"name": label, "variant": precision, "download_defs": [self.model_download_def(model, precision)]}
```

A fixed-checkpoint extension can keep its existing `query_download_def()` or `query_download_defs()`. The manager calls it only if the inventory hook is absent or returns `None`; an empty inventory explicitly declares no checkpoints. Legacy queries receive `enabled_only=False` if supported. That flag does **not** enumerate other configuration values: an extension with variant-dependent downloads must implement the inventory hook. Processors registered by a plugin are discovered automatically; the plugin need not repeat their declarations.

`name` defaults to the owner's name, and `family` defaults to the processor category or **Plugin Extensions**. Set `variant` to group entries as **family → model → variant**. Entries without a variant appear directly under their family. Use `files` for exact checkpoint-relative or absolute file paths, including the individual files of a folder snapshot without a file list. Use `model_types` for WanGP models the processor reuses and `model_lora_keys` to map a model ID to model-definition keys for required LoRAs. Report only model assets that the user may remove through Models Manager.

Reference Plugins:
*   [Stable Diffusion 1.4 Model Plugin](https://github.com/deepbeepmeep/wan2gp-stable-diffusion-1-4): compact model Plugin template that adds Stable Diffusion 1.4 using MMGP-managed UNet, text encoder, and VAE components.
*   [Spatial Pixel Duplicate](https://github.com/deepbeepmeep/wan2gp-pixel-upsampler): compact reference spatial upsampler that performs exact nearest-neighbor pixel duplication and documents the handler contract.
*   [Temporal Blend](https://github.com/deepbeepmeep/wan2gp-blender-upsampler): compact reference temporal upsampler that averages neighboring frames for model-free x2 interpolation and documents the handler contract.

Specialized Plugin APIs:
*   Spatial upsamplers are documented in [Spatial Upsampler Plugin API](SPATIAL_UPSAMPLERS.md). Use this guide for post-processing upsamplers, VAE upsampler capability declarations, Plugin-discovered `spatial_upsampler_handlers`, and extension offload object registration.
*   Temporal upsamplers are documented in [Temporal Upsampler Plugin API](TEMPORAL_UPSAMPLERS.md). Use this guide for frame interpolation handlers, Plugin-discovered `temporal_upsampler_handlers`, and temporal upsampler config sections.
*   Audio processors are documented in [Audio Processor Plugin API](AUDIO_PROCESSORS.md). Use this guide for soundtrack, voice replacement, standalone audio edit handlers, Plugin-discovered `audio_processors`, and audio processor config sections.

A complete Plugin folder typically looks like this:

```
plugins/
└── my-awesome-plugin/
    ├── __init__.py         # (Required, can be empty) Makes this a Python package.
    ├── plugin.py           # (Required) Main Plugin logic and class definition.
    ├── requirements.txt    # (Optional) Lists pip dependencies for your Plugin.
    └── ...                 # Other helper .py files, assets, etc.
```

## Getting Started: Creating a Plugin

1.  **Create a Plugin Folder**: Inside the main `plugins/` directory, create a new folder for your Plugin (e.g., `my-awesome-plugin`).

2.  **Create Core Files**:
    *   Inside `my-awesome-plugin/`, create an empty file named `__init__.py`.
    *   Create another file named `plugin.py`. This will be the entry point for your Plugin.

3.  **Define a Plugin Class**: In `plugin.py`, create a class that inherits from `WAN2GPPlugin` and set its metadata attributes.

    ```python
    from shared.utils.plugins import WAN2GPPlugin

    class MyPlugin(WAN2GPPlugin):
        def __init__(self):
            super().__init__()
            self.name = "My Awesome Plugin"
            self.version = "1.0.0"
            self.description = "This Plugin adds awesome new features."
    ```

4.  **Add Dependencies (Optional)**: If your Plugin requires external Python libraries (e.g., `numpy`), list them in a `requirements.txt` file inside your Plugin folder. These will be installed automatically when a user installs your Plugin via the UI.

5.  **Enable and Test**:
    *   Start Wan2GP.
    *   Go to the **Apps & PlugIn Manager** screen.
    *   You should see your new Plugin (`my-awesome-plugin`) in the list.
    *   Under **Available Locally**, click **Install** (for a community Plugin) or **Enable** (for a bundled app). Changes are saved automatically.
    *   **Restart the Wan2GP application.** Your Plugin will now be active.

## Plugin Distribution and Installation

#### Packaging for Distribution
To share your Plugin, simply upload your entire Plugin folder (e.g., `my-awesome-plugin/`) to a public GitHub repository.

#### Installing from the UI
Users can install your Plugin directly from the Wan2GP interface:
1.  Go to the **Apps & PlugIn Manager** screen.
2.  Under **Install from URL**, paste the full URL of your Plugin's GitHub repository.
3.  Click **Download and Install from URL**.
4.  The system will clone the repository and install any dependencies from `requirements.txt`.
5.  After a successful installation, it appears under **Enabled Apps & Plugins**. Restart WanGP to activate it.

The Plugin manager also supports updating Plugins (if installed from git) and uninstalling them.

## Plugin API Reference

### The `WAN2GPPlugin` Class
Every Plugin must define its main class in `plugin.py` inheriting from `WAN2GPPlugin`.

```python
# in plugins/my-awesome-plugin/plugin.py
from shared.utils.plugins import WAN2GPPlugin
import gradio as gr

class MyAwesomePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        # Metadata for the Apps & PlugIn Manager UI
        self.name = "My Awesome Plugin"
        self.version = "1.0.0"
        self.description = "A short description of what my Plugin does."
        
    def setup_ui(self):
        # UI setup calls go here
        pass
        
    def post_ui_setup(self, components: dict):
        # Event wiring and UI injection calls go here
        pass
```

### Core Methods
These are the methods you can override or call to build your Plugin.

#### `setup_ui(self)`
This method is called when your Plugin is first loaded. It's the place to declare new tabs or request access to components and globals before the main UI is built.

*   **`self.add_tab(tab_id, label, component_constructor, position=-1, *, icon=None)`**: Adds an app to the main bar. Optionally pass SVG markup as `icon`; otherwise WanGP supplies an app icon. The native Gradio tab remains available behind the app bar, so existing selection, deselection, component insertion and `main_tabs` navigation continue to work.
*   **`self.request_component(component_id)`**: Requests access to an existing Gradio component by its `elem_id`. The component will be available as an attribute (e.g., `self.loras_multipliers`) in `post_ui_setup`.
*   **`self.request_global(global_name)`**: Requests access to a global variable or function from the main `wgp.py` application. The global will be available as an attribute (e.g., `self.server_config`).

#### `post_ui_setup(self, components)`
This method runs after the entire main UI has been built. Use it to wire up events for your custom UI and to inject new components into the existing layout.

*   `components` (dict): A dictionary of the components you requested via `request_component`.
*   **`self.insert_after(target_component_id, new_component_constructor)`**: A powerful method to dynamically inject new UI elements into the page.

#### `on_tab_select(self, state)` and `on_tab_deselect(self, state)`
If you used `add_tab`, these methods will be called automatically when your tab is selected or deselected, respectively. This is useful for loading data or managing resources.

#### `on_model_change(self, state, model_type)`
This optional callback runs when the main model selection changes in the Gradio UI. `state["model_type"]` has already been updated, so Plugins can use it to refresh per-model caches, reset Plugin state, or synchronize custom UI logic.

#### Switching the Main Model
Plugins can reuse WanGP's normal model-switch flow by requesting the hidden target component and the public helper:

```python
def setup_ui(self):
    self.request_component("model_choice_target")
    self.request_component("main_tabs")
    self.request_global("switch_to_model")
```

`switch_to_model(model_type, open_media_tab=False)` returns two values: the model switch target update and a main-tab update. Wire them to `[self.model_choice_target, self.main_tabs]`. Keep `open_media_tab=False` to stay in the Plugin, or pass `True` to open the Media Generator tab.

#### `set_global(self, variable_name, new_value)`
Allows your Plugin to safely modify a global variable in the main `wgp.py` application.

#### `register_data_hook(self, hook_name, callback)`
Allows you to intercept and modify data at key points. For example, the `before_metadata_save` hook lets you add custom data to the metadata before it's saved to a file.

### Deepy Tool Registration

An enabled Plugin can expose ordinary Python callables to either Deepy runtime. Register tools in `setup_ui`; a Plugin does not need to create a tab. The Plugin manager publishes the declarations after `setup_ui` completes, and only tools belonging to enabled Plugins are visible. Plugin and tool changes require a WanGP restart, and already-open Deepy sessions do not refresh their tool catalog.

#### Deepy Prime MCP tools

```python
self.register_deepy_prime_tool(
    function,
    *,
    name=None,
    display_name=None,
    description=None,
    pause_runtime=True,
    pause_reason="tool",
    requires_file_system=False,
)
```

This registers `function` on Deepy Prime's in-process MCP server. FastMCP builds the input schema from the callable's type annotations, defaults, and docstring. Both synchronous and asynchronous functions are supported.

Arguments:

*   `function`: A callable using named parameters. Positional-only parameters, `*args`, and `**kwargs` are rejected.
*   `name`: Optional MCP tool name. It defaults to `function.__name__` and must match `[A-Za-z_][A-Za-z0-9_]*`.
*   `display_name`: Optional short label shown in the Deepy chat UI. It defaults to the title-cased tool name.
*   `description`: Optional model-facing description. It defaults to the function docstring.
*   `pause_runtime`: When `True`, WanGP pauses/releases Deepy's local model runtime before calling the tool so the Plugin can use its resources. Set it to `False` for lightweight CPU-only work.
*   `pause_reason`: Internal pause category reported to Deepy's runtime. Plugins normally leave this as `"tool"`.
*   `requires_file_system`: When `True`, the tool is omitted unless Deepy's filesystem-reading option is enabled. This controls discovery only; the Plugin remains responsible for validating paths and access inside the callable.

The helper returns the original callable. A Prime tool name cannot replace a built-in MCP tool or one registered by another enabled Plugin.

#### Deepy Zero native tools

```python
self.register_deepy_zero_tool(
    function,
    *,
    name=None,
    display_name=None,
    description=None,
    parameters=None,
    pause_runtime=True,
    pause_reason="tool",
    requires_file_system=False,
)
```

This registers a synchronous callable in Deepy Zero's native tool list. `name`, `display_name`, `description`, `pause_runtime`, `pause_reason`, and `requires_file_system` have the same meanings as for Prime. Zero functions must be synchronous.

`parameters` is an optional dictionary keyed by exact Python parameter name. Each value may contain:

*   `description`: Model-facing explanation of the argument.
*   `type`: Optional JSON-schema type override. Without it, Zero infers `string`, `integer`, `number`, `boolean`, `array`, or `object` from the Python annotation.
*   `required`: Optional Boolean override. Without it, parameters without a Python default are required and parameters with a default are optional.
*   Any additional JSON-schema properties that should be advertised to the model, such as `enum`, `minimum`, `maximum`, `items`, or `maxItems`.

Unknown parameter names are rejected during Plugin setup. A Zero tool name cannot replace a built-in Zero tool or one registered by another enabled Plugin.

#### Shared callable contract

The same bound method may be registered once for Prime and once for Zero, even with the same name; the two namespaces are independent. At invocation time WanGP passes tool arguments as keyword arguments. Prime's MCP layer validates its generated schema. Zero checks required values, while other schema constraints guide the model; a Zero callable must still validate semantic limits and untrusted input. Return a JSON-serializable value such as a dictionary, list, string, number, Boolean, or `None`. Unhandled exceptions are reported to Deepy as tool failures.

Registration happens before requested WanGP globals and Gradio components are injected, but the bound method runs later and may use attributes requested with `request_global`. Tool functions run in Deepy's backend worker rather than a Gradio event callback, so they should not mutate Gradio components directly.

#### Plugin toolboxes for agents

A Plugin with several related functions can publish them as one toolbox instead of separate tools. Every agent reaches it through the `wangp_plugins` tool: MCP clients, Deepy Prime and Deepy Python scripts (`wangp.plugins()`, `wangp.plugin(plugin, action, ...)`). Agents discover it step by step, so an enabled toolbox costs only one line in their context until it is used:

1.  The `wangp_plugins` description lists one summary line per toolbox.
2.  `wangp_plugins(plugin="<id>")` returns the toolbox guide and its actions with their call signatures.
3.  `wangp_plugins(plugin, action, arguments=null)` returns the action's contract: description, argument schema, example and routing.
4.  `wangp_plugins(plugin, action, arguments={...})` runs it.

```python
def setup_ui(self):
    self.register_deepy_toolbox("One line: what the Plugin does and for which media.", guide="How to choose and combine the actions.")
    self.register_deepy_action(self.list_presets, uses_gpu=False, summary="List the presets.")
    self.register_deepy_action(self.process, summary="Process a file or a folder.", paths={"source": "read", "output": "write"},
                               routing="Processing a whole folder of clips", instead_of="wangp_postprocess")

def process(self, ctx, source: str | list[str], output: str | None = None, strength: float = 1.0) -> dict:
    """Process the media at source and return the result."""
    ...
```

`register_deepy_toolbox(summary, *, guide="", toolbox_id=None, display_name=None)` declares the toolbox once. Its identifier defaults to the Plugin folder name and its display name to the Plugin name.

`register_deepy_action(function, *, name=None, summary=None, description=None, parameters=None, example=None, uses_gpu=True, paths=None, routing="", instead_of="", scripts=True)` adds an action:

*   `function(ctx, **arguments)` is synchronous and takes the call context as its first parameter. The argument schema comes from the type hints, the description from the docstring.
*   `parameters`: optional per-argument JSON-schema entries (`description`, `enum`, `minimum`, ...) merged into the derived schema. An entry with its own `type`, `anyOf` or `enum` replaces the derived one.
*   `uses_gpu`: Deepy releases the GPU before every call of the action, so that the generations the action submits can run. Without that release, a generation submitted by the Plugin would wait for Deepy, which is waiting for the Plugin. Leave it `True` unless the action never runs GPU work. Discovery and contract reads never release Deepy.
*   `paths`: argument roles, `"read"` or `"write"`. Before the call, WanGP turns Gallery media ids and `@alias` paths into absolute paths and checks them against the caller's file access. Folders and wildcard patterns (`@documents/clips/*.mp4`) are accepted; the folder part is authorized and the pattern is kept for the Plugin to expand.
*   `routing`: when to choose this action. WanGP adds it to the agents' instructions, so a request is routed to the Plugin without hand-written prompt changes. `instead_of` names a WanGP tool the action competes with, such as `wangp_postprocess`, whose discovery then mentions the action.
*   `scripts=False` keeps an action out of Deepy Python scripts (for example one that stops a job).

The context `ctx` gives:

*   `ctx.state` and `ctx.session`: the caller's WanGP state and API session. Tasks submitted with `ctx.session.submit_task(...)` join the caller's queue; WanGP starts Deepy's browser queue for them.
*   `ctx.progress(done=None, total=None, text="")`: progress on the caller's tool card (Deepy) or job (MCP), at most three updates per second.
*   `ctx.cancelled()`: true once the caller stops: the Deepy chat Stop button, or `cancel_job` on a background job.
*   `ctx.call(tool, arguments)`: another WanGP tool, run on the action's thread. Its arguments and result name files as the caller does (`@outputs/...`): `ctx.resolve(path)` gives a file's real path. A GPU tool releases Deepy first. In Deepy, a generation run this way shows its steps on the status bar and moves the action's progress bar within its current item (report `done`/`total` items with `ctx.progress` before each one).
*   `ctx.can_write(path)`, `ctx.outputs_dir`, `ctx.virtualize(path)` and `ctx.resolve(path)`: the caller's write access, WanGP's outputs folder, the caller's name for a path and the real path of a name.

An action returns a dictionary. Its `status` defaults to `done`; use `stopped` or `error` for other outcomes. A `gr.Error` raised by shared validation code reaches the agent as a tool error. MCP clients may pass `wait=false` to run an action as a background job followed with `wangp_session` `get_job`, `wait_jobs` and `cancel_job`. The bundled Media Flow Plugin is a complete example: it exposes its templates by task and runs them on single files, folders or wildcard batches.

#### Plugin skills

A skill is a Markdown method that Deepy Prime and MCP agents read when a task calls for it, like WanGP's own `wangp://skills/...` (Deepy Zero does not read skills). A Plugin ships its skills in its own folder, one folder per skill, and WanGP serves them while the Plugin is enabled; no registration call is needed:

```
plugins/my_plugin/
    plugin.py
    skills/
        comic-pages/SKILL.md   # general skill: wangp://skills/comic-pages
        inking/SKILL.md        # toolbox skill: wangp://skills/my_plugin/inking
```

Each `SKILL.md` starts with front matter giving the `name` (the folder's name, lowercase and hyphenated) and a `description` of what the skill covers, followed by the Markdown body. Whether it has a `routing` line decides who reads it:

```markdown
---
name: comic-pages
description: Lay out comic pages and strips from generated panels.
routing: Drawing a comic page or strip
---
```

*   **General skill (with `routing`):** it extends Deepy for every conversation. It is served as `wangp://skills/<name>` and listed with WanGP's skills. Its routing line joins the agents' instructions ("Drawing a comic page or strip: read wangp://skills/comic-pages"), at the cost of that one line in every session. A name already used by WanGP or another enabled Plugin is refused.
*   **Toolbox skill (without `routing`):** it is the method for the Plugin's own toolbox and jobs, for example how to plan, which templates to choose and how to review results. It is served as `wangp://skills/<plugin folder>/<name>`, outside the skills listing. The toolbox discovery (`wangp_plugins(plugin="<id>")`) names it next to the guide, and a job's `instructions` can name it too, so agents read it only when they use the Plugin.

Prefer toolbox skills. Write a general skill only when the method helps outside the Plugin's own toolbox.

### Running Deepy Jobs

A Plugin can also send its own request to Deepy, with media, and follow its progress. `self.deepy` gives the Plugin's Deepy client in the WanGP web UI (it raises an error before the UI has started or in command-line runs).

```python
clips = self.deepy.gallery(kind="video")
with self.deepy.run("Cut a 10 s trailer from these clips", media=[clips[0]["id"], "D:/shots/intro.mp4"], session_dir="D:/my_plugin/sessions/trailer-12", title="Trailer 12") as job:
    for event in job.events():
        if event.kind == "progress":
            print(event.data.status, event.data.progress)
    result = job.result()
print(result.status, result.text, [output["path"] for output in result.outputs])

# Later, even after a restart: the conversation continues where it stopped.
with self.deepy.run("Make it 6 s", session_dir="D:/my_plugin/sessions/trailer-12") as job:
    result = job.result()
```

**How a job runs.** Deepy has one conversation at a time, so a job waits until Deepy and the generation queue are idle (and until earlier Plugin jobs have finished). It then sets the user's conversation and workspace aside, shows its own conversation and workspace in Deepy and in the gallery while it runs, and brings the user's back when it ends. The user's conversation is restored whether or not saved sessions are enabled; Deepy reloads its context then, as when resuming a session. Meanwhile the user sees the request (badged with the Plugin's name) and Deepy's work, and can stop it, but cannot switch conversation or workspace. Do not wait for a job inside a Deepy tool: the job would wait for that tool's request to finish.

**Listing media.** `self.deepy.gallery(workspace=None, kind=None)` lists the media of a workspace, the one shown by default: dictionaries with `id`, `path`, `kind` (`image`, `video`, `audio` or `3d`), `name`, `selected` and `workspace`. `self.deepy.selected()` is the media selected in the shown gallery (or `None`) and `self.deepy.workspaces()` lists the workspaces with their `id`, `name`, `storage_mode` and whether the UI `shown` them. The `id` is a Gallery media id, the form Deepy tools and MCP agents use (for example `visual:3f2a91c04b7e`). `run()` accepts these ids from any workspace, and also the ids Deepy reports for media of the shown gallery.

**`run(prompt, media=(), *, media_mode="copy", session_dir=None, workspace=None, title=None, instructions=None)`** returns a running `DeepyJob` at once. It raises an error when Deepy cannot run requests: check `self.deepy.unavailable_reason` (an empty string when Deepy is ready) beforehand to tell the user why, for example with `gr.Info`.

*   `prompt`: the request, sent as one message.
*   `media`: Gallery media ids (from any workspace), items returned by `gallery()`, or paths to image, video, audio or 3D files. They are attached to this request only, never to a message the user is writing. A path does not need to be readable by Deepy's own file access: the Plugin imports the file into the job's gallery, as a browser upload does, and Deepy then uses it by its media id.
*   `media_mode`: `"copy"` copies each file into the job's workspace, so the job keeps working if the Plugin deletes its file; media already in the job's gallery (for example the user's selection with `workspace="current"`) are used as they are. `"link"` uses the file where it is, without a copy; the Plugin must keep the file available. Linking needs a workspace with shared storage: it is refused for a workspace that keeps its media in its own folder (temporary workspaces, workspaces of a `session_dir`, and dedicated storage).
*   `workspace`: where Deepy's work goes.
    *   The id of an existing workspace, for example one from `self.deepy.ensure_workspace(...)`: the job works there, and the workspace stays after the job.
    *   `"isolated"` (the default with `session_dir`): a new workspace for this session. It stays in the workspace list, so the user can browse the results later, and the session uses it again when resumed. With `session_dir` its media are stored in `session_dir/gallery`; otherwise it follows the "Default Storage for New Workspaces" setting.
    *   `"temporary"` (the default without `session_dir`): a workspace in a private folder, removed with its files when the job is closed. Outputs stay readable until then: copy the ones to keep.
    *   `"current"`: the workspace the user has open; Deepy's outputs join the user's gallery.
*   `session_dir`: a folder holding the conversation (`session.json`, `context.json`, the chat journal) and, with an isolated workspace, its media. An empty or missing folder starts a new session; a folder holding a session resumes it, right after the previous job or days later. The folder must be outside Deepy's sessions folder, and the session belongs to the Deepy version (Zero or Prime) that created it. Plugin sessions are not listed in the Deepy session picker. Without `session_dir`, the job's conversation is discarded when the job is closed.
*   `title`: the session (and isolated workspace) name; by default it is derived from the first request.
*   `instructions`: standing rules for this job, added to Deepy Prime's system prompt while the job runs and removed when it ends; they never reach the user's conversation. Deepy Zero ignores them. Keep them short and the same from one job to the next (a changed text makes Deepy process the session's context again) and put what is specific to the request in `prompt`; a longer method belongs in a Plugin skill they name.

**Following a job.** `job.events(timeout=None)` yields `SessionEvent(kind, data)` objects until the job ends:

| Kind | Data |
| --- | --- |
| `queued` | `None`: the job waits for Deepy to be idle. |
| `started` | `{"session_dir", "workspace_id"}`: Deepy now shows the job. |
| `status` | `{"kind", "text"}`: Deepy's status line (starting, thinking, paused...). |
| `text` | `{"text", "delta", "last"}`: all Deepy said so far, what was added, and its last message alone. |
| `thought` | `{"id", "text", "delta", "streaming"}` while Deepy thinks, when its model does: one `id` per thought, the thought so far, what was added, and whether it is still being written. Up to three updates per second. |
| `stats` | The line Deepy's chat shows, as `{"text"}`, with the numbers behind it. With a local model: `avg_prefill_tokens_per_second`, `avg_generated_tokens_per_second`, `consumed_tokens`, `max_tokens`, and `prefilling_tokens` while it reads that many tokens before writing. Up to three updates per second. |
| `tool` | `{"id", "name", "label", "status", "status_text"}` when a Deepy action starts or changes state (`running`, `done`, `error`, `interrupted`); a failed action adds `"error"`, its reason. |
| `progress` | A `ProgressUpdate` (`phase`, `status`, `progress` in percent, `current_step`, `total_steps`, `unit`) for generations run by the job. |
| `output` | `{"id", "path", "kind", "name", "label"}` for each media file Deepy writes. |
| `chat` | When you set `job.chat_events = True` right after `run`: each update of Deepy's chat, for Deepy's window in your page (see below). |
| `done`, `cancelled`, `error` | The `DeepyResult`; always the last event. |

`job.result(timeout=None)` waits for the `DeepyResult`: `status` (`"done"`, `"cancelled"` or `"error"`), `text` (all Deepy said during the job), `reply` (its last message alone), `outputs` (as in `output` events), `session_dir` and `error`. The job ends once the user's conversation has been brought back. `job.cancel()` stops a job that is waiting, or interrupts Deepy's work on it; a stop from the Deepy chat ends the job as `cancelled` too. `job.close()` (or leaving a `with` block) cancels a job still running and deletes its private files, temporary workspace included.

**Steering a running job.** `job.steer(text)` sends Deepy a message while it works on the job; Deepy applies it at its next step, as a message typed in the Deepy chat while it works. The reply and media that follow belong to the job's result. It raises an error once the job has ended: send the message with a new `run()` on the same `session_dir`.

**Pausing and resuming.** A job kept in a `session_dir` can be stopped with `job.cancel()` and continued later, even after WanGP restarts: a new `run()` on the same `session_dir` finds the conversation again, so the request only needs to say what to finish. To leave the GPU free meanwhile, `self.deepy.release_vram(timeout=120)` takes Deepy's model out of VRAM once Deepy is idle (it returns `False` if Deepy is still busy after `timeout` seconds); the model comes back at Deepy's next request.

**Workspaces of your own.** `self.deepy.ensure_workspace(name, folder)` returns the id of the workspace that keeps its media in `folder`, created with `name` the first time and never archived for inactivity: a Plugin can keep a project's media in its own folder and pass the id as `workspace` to `run()`. `self.deepy.deepy_type` is the Deepy version jobs run with (`"zero"` or `"prime"`).

**Generating without Deepy.** `self.deepy.web_session(workspace=None)` returns a `WanGPSession` whose `submit_task(settings, callbacks=None)` adds tasks to the web UI's queue from any thread, for example from a Plugin web app. With a workspace id, its generations record in that workspace without changing the gallery the user is viewing.

### Plugin Web Apps

Declare navigation in `plugin_info.json`; the Plugin manager provides the controls, URL routing and return behavior for ordinary Gradio tabs and embedded web apps alike:

```json
{
  "name": "My App",
  "app_url": "/my-app",
  "fullscreen": true,
  "app_tab": "my_tab"
}
```

- `app_url` is an optional short local path, with one lowercase segment. It opens the app's tab, in full screen when supported. WanGP routes and URLs claimed by another plugin are rejected. The existing `url` property remains the repository URL.
- `fullscreen` adds the shared **Full Screen** and **WanGP** return controls. Full screen expands the existing tab in place, preserving form values, iframe state and running event subscriptions. Refreshing its address reopens that tab in full screen.
- `app_tab` selects the tab declared with `add_tab(tab_id=...)`; omit it to use the first declared tab. A Plugin must have a matching tab. Other tabs retain normal navigation.

Both navigation properties are optional and independent. A fullscreen app without `app_url` uses `/plugins/<folder_id>/`. Only enabled, loaded plugins register their routes. Folder URLs keep their address and open the requested app without showing another tab first, behind WanGP's existing authentication. Returning with **WanGP** shows the same app in the main interface at WanGP's root address. Ordinary tab switching keeps this root address; only fullscreen or opening a folder link uses the app's folder URL. Use the shared controls rather than creating a separate return link to the WanGP root.

A Plugin can also serve its own page when its interface does not fit Gradio components. In `setup_ui`:

```python
self.register_web_app("/my_plugin", self.create_web_app)

def create_web_app(self):
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    def index():
        return {"hello": "world"}

    return app
```

`register_web_app(path, factory)` serves `factory()`, an ASGI application such as a FastAPI app, at `path` from the WanGP server, behind its authentication. The factory runs once when the server starts, after WanGP's globals have been injected. If `app_url` uses that same path, embed the web page with `<iframe src="./my_plugin/?embedded=1">`; the embedded request reaches the web page while the public URL opens its WanGP tab. API and asset subpaths continue to reach the web app. Use relative URLs in the page so that it works wherever it is mounted. Wrap a full-height iframe in `<div class="wangp-addon-web-view">` to fit it to the shared fullscreen view. The bundled Showrunner Plugin is an example; Media Flow uses ordinary Gradio components and needs only the JSON declarations. Web apps without these declarations retain their existing independent URLs.

A web page in an iframe can carry the Full Screen control in its own header, which saves the bar above the page. Each time the page draws its header, it calls `window.parent.WanGPAddonNavigation?.headerButton(window.frameElement, header, className)`: the header gets a button that opens Full Screen from the tab and goes back to WanGP from full screen, with its tooltip, and the bar goes away. `className` is optional and styles the button as the page's own. Pages that do not call it keep the bar.

#### Deepy's Window in Your Page

Your web page can show Deepy's chat window, with Deepy's own look, streaming and scrolling (it follows the latest words unless the reader scrolled up), in an iframe:

```html
<iframe src="../deepy/window?contour=0&input=0&avatars=0&wide=1"></iframe>
```

| Option | Effect |
| --- | --- |
| `contour=0` | No border around the window. |
| `input=0` | No text input: the window only shows the conversation. |
| `stats=0` | No line of speeds under the conversation. |
| `avatars=0` | No name beside the messages. |
| `stop=0` | No Pause and Stop on the status line. |
| `actions=0` | No buttons on the messages. |
| `compact=0` | Every step shown, instead of grouped under each answer. |
| `wide=1` | Messages use the whole width, for a narrow panel. |
| `font=0.84` | Text size. |
| `theme=dark` | Dark colors, also with `DeepyWindow.theme("dark")`. |

Once the iframe has loaded, your page drives it through `frame.contentWindow.DeepyWindow`:

- `replay(events)` shows a conversation at once, from `shared.deepy.session_store.read_ui_updates(session_dir, after=0)`, which returns `events`, a `next` position to pass as `after` next time, and `restart` when the conversation was rewound.
- `update(data)` applies one `chat` event of a running job.
- `reset()` empties the window, `rejoin()` comes before `update` again after your page skipped some, and `shown()` puts the reader back in place when the iframe returns from being hidden.
- `onGap` is called when an update could not be placed: read the saved conversation again. `onSubmit(text)` receives what the user typed in its input.

## Examples

### Example 1: Creating a New Tab

**File Structure:**
```
plugins/
└── greeter_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/greeter_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class GreeterPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Greeter Plugin"
        self.version = "1.0.0"
        self.description = "Adds a simple 'Greeter' tab."

    def setup_ui(self):
        self.add_tab(
            tab_id="greeter_tab",
            label="Greeter",
            component_constructor=self.create_greeter_ui,
            position=2 # Place it as the 3rd tab (0-indexed)
        )
        
    def create_greeter_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("## A Simple Greeter")
            with gr.Row():
                name_input = gr.Textbox(label="Enter your name")
                output_text = gr.Textbox(label="Output")
            greet_btn = gr.Button("Greet")
            
            greet_btn.click(
                fn=lambda name: f"Hello, {name}!",
                inputs=[name_input],
                outputs=output_text
            )
        return demo
```

### Example 2: Injecting UI Elements

This example adds a simple HTML element right after the "Loras Multipliers" textbox.

**File Structure:**
```
plugins/
└── injector_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/injector_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class InjectorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "UI Injector"
        self.version = "1.0.0"
        self.description = "Injects a message into the main UI."

    def post_ui_setup(self, components: dict):
        def create_inserted_component():
            return gr.HTML(value="<div style='padding: 10px; color: gray; text-align: center;'>--- Injected by a Plugin! ---</div>")

        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_inserted_component
        )
```

### Example 3: Advanced UI Injection and Interaction

This Plugin injects a button that interacts with other components on the page.

**File Structure:**
```
plugins/
└── advanced_ui_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/advanced_ui_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class AdvancedUIPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "LoRA Helper"
        self.description = "Adds a button to copy selected LoRAs."
        
    def setup_ui(self):
        # Request access to the components we want to read from and write to.
        self.request_component("loras_multipliers")
        self.request_component("loras_choices")

    def post_ui_setup(self, components: dict):
        # This function will create our new UI and wire its events.
        def create_and_wire_advanced_ui():
            with gr.Accordion("LoRA Helper Panel (Plugin)", open=False):
                copy_btn = gr.Button("Copy selected LoRA names to Multipliers")

            # Define the function for the button's click event.
            def copy_lora_names(selected_loras):
                return " ".join(selected_loras)

            # Wire the event. We can access the components as attributes of `self`.
            copy_btn.click(
                fn=copy_lora_names,
                inputs=[self.loras_choices],
                outputs=[self.loras_multipliers]
            )
            return panel # Return the top-level component to be inserted.

        # Tell the manager to insert our UI after the 'loras_multipliers' textbox.
        self.insert_after(
            target_component_id="loras_multipliers",
            new_component_constructor=create_and_wire_advanced_ui
        )
```

### Example 4: Accessing Global Functions and Variables

**File Structure:**
```
plugins/
└── global_access_plugin/
    ├── __init__.py
    └── plugin.py
```

**Code:**
```python
# in plugins/global_access_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

class GlobalAccessPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Global Access Plugin"
        self.description = "Demonstrates reading and writing global state."

    def setup_ui(self):
        # Request read access to globals
        self.request_global("server_config")
        self.request_global("get_video_info")
        
        # Add a tab to host our UI
        self.add_tab("global_access_tab", "Global Access", self.create_ui)
        
    def create_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("### Read Globals")
            video_input = gr.Video(label="Upload a video to analyze")
            info_output = gr.JSON(label="Video Info")
            
            def analyze_video(video_path):
                if not video_path: return "Upload a video."
                # Access globals as attributes of `self`
                save_path = self.server_config.get("save_path", "outputs")
                fps, w, h, frames = self.get_video_info(video_path)
                return {"save_path": save_path, "fps": fps, "dimensions": f"{w}x{h}"}

            analyze_btn = gr.Button("Analyze Video")
            analyze_btn.click(fn=analyze_video, inputs=[video_input], outputs=[info_output])

            gr.Markdown("--- \n ### Write Globals")
            theme_changer = gr.Dropdown(choices=["default", "gradio"], label="Change UI Theme (Requires Restart)")
            save_theme_btn = gr.Button("Save Theme Change")

            def save_theme(theme_choice):
                # Use the safe `set_global` method
                self.set_global("UI_theme", theme_choice)
                gr.Info(f"Theme set to '{theme_choice}'. Restart required.")

            save_theme_btn.click(fn=save_theme, inputs=[theme_changer])

        return demo
```

### Example 5: Using Helper Modules (Relative Imports)
This example shows how to organize your code into multiple files within your Plugin package.

**File Structure:**
```
plugins/
└── helper_plugin/
    ├── __init__.py
    ├── plugin.py
    └── helpers.py
```

**Code:**
```python
# in plugins/helper_plugin/helpers.py
def format_greeting(name: str) -> str:
    """A helper function in a separate file."""
    if not name:
        return "Hello, mystery person!"
    return f"A very special hello to {name.upper()}!"

# in plugins/helper_plugin/plugin.py
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from .helpers import format_greeting # <-- Relative import works!

class HelperPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Helper Module Example"
        self.description = "Shows how to use relative imports."

    def setup_ui(self):
        self.add_tab("helper_tab", "Helper Example", self.create_ui)

    def create_ui(self):
        with gr.Blocks() as demo:
            name_input = gr.Textbox(label="Name")
            output = gr.Textbox(label="Formatted Greeting")
            btn = gr.Button("Greet with Helper")
            
            btn.click(fn=format_greeting, inputs=[name_input], outputs=[output])
        return demo
```

### Example 6: Extending Deepy Prime and Deepy Zero

This extension exposes one implementation to both Deepy runtimes. Prime receives a normal MCP function; Zero receives its classic native equivalent.

```python
# in plugins/preset_lookup/plugin.py
from shared.utils.plugins import WAN2GPPlugin


class PresetLookupPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Preset Lookup"
        self.description = "Lets Deepy search this Plugin's presets."

    def setup_ui(self):
        self.register_deepy_prime_tool(
            self.search_presets,
            name="preset_lookup",
            display_name="Search Plugin Presets",
            pause_runtime=False,
        )
        self.register_deepy_zero_tool(
            self.search_presets,
            name="preset_lookup",
            display_name="Search Plugin Presets",
            parameters={
                "query": {"description": "Words to match in preset names and descriptions."},
                "limit": {"description": "Maximum results to return.", "minimum": 1, "maximum": 20},
            },
            pause_runtime=False,
        )

    def search_presets(self, query: str, limit: int = 5) -> dict:
        """Search the Plugin preset catalog and return the best matches."""
        matches = self._search_catalog(query)[:limit]
        return {"status": "done", "matches": matches}

    def _search_catalog(self, query: str) -> list[dict]:
        # Replace with the Plugin's real lookup implementation.
        return [{"name": "Example", "description": query}]
```

## Finding Component IDs

To interact with an existing component using `request_component` or `insert_after`, you need its `elem_id`. You can find these IDs by:

1.  **Inspecting the Source Code**: Look through `wgp.py` for Gradio components defined with an `elem_id`.
2.  **Browser Developer Tools**: Run Wan2GP, open your browser's developer tools (F12), and use the "Inspect Element" tool to find the `id` of the HTML element you want to target.

Some common `elem_id`s include:
*   `loras_multipliers`
*   `loras_choices`
*   `main_tabs`
*   `gallery`
*   `family_list`, `model_base_types_list`, `model_list`

---

> Applies to: Installing and developing WanGP Plugins. Registration and handler examples are Python code; exposed tools depend on the Plugins enabled in the installation.
