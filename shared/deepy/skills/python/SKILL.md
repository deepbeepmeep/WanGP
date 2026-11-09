---
name: python
description: "Use before the first wangp_python run of a session: how the sandbox works, how a script calls WanGP's tools (generation, inspection, the toolbox, files), and which python-* skill to read for images and text, clips, 3D, batches, or data and charts."
---

# Python in WanGP

`wangp_python` runs Python in a persistent sandbox on the CPU (numpy, PIL, scipy, OpenCV, matplotlib, pandas, sympy, soundfile, pypdf, an SVG rasterizer). Models paint, sing and animate; Python measures, places, counts, decides and loops. Scripts can call WanGP's own tools, so a loop over files or generations is one run, not one tool call per item.

## Working loop

1. One step per run: a `# title` comment first (it becomes the description of published media), under about 80 lines. Variables persist between runs; a timeout or `restart` loses variables, never files.
2. Read inputs: Gallery media and authorized files with `wangp.path(ref)` (`Image.open(wangp.path("visual:IMAGE"))`, `open("@documents/notes.txt")`); folders with `wangp.listdir("@test/photos", recursive=True)` (dicts with path, file, name, type, kind, bytes, date).
3. Write only with relative paths inside the sandbox (`tmp/`, `scratch/`, `out/`). Finished deliverables go under `out/`: images, audio and meshes there are published to the Gallery after the run and come back in `outputs` with a `media_id` (at most 8 per run). Frames and scratch never go under `out/`.
4. Look at what you made before building on it: `inspect_media` with a question (or `wangp.inspect(media, question)` from a script), several times along a clip.
5. Print only what the next decision needs (counts, sizes, the matches): long output is cut.

## WanGP's tools from a script

- `wangp.generate(settings)`: generates with a template or model and waits; returns the job result with the new media ids. `wangp.call("templates", task="gen_video")` / `wangp.call("models", action="lookup", arguments={"query": ...})` find templates and model types.
- `wangp.video_frames(ref, start=0, stop=None, step=1)`: the frames of a video or a view (`@views/...`) as RGB numpy arrays, one by one and with nothing encoded; `wangp.path` on a view cuts a real clip the first time.
- `wangp.inspect(media, question)`: asks the visual model about an image, a video frame (`media_inputs=[{"media_id": ref, "time_s": t}]`) or a 3D asset.
- `wangp.toolbox(action, **arguments)`: media utilities (resize_crop, extract_video, merge_videos, remux_media, render_3d_views, create_mask, transcribe_media...).
- `wangp.io(action, {...})`: list, info, read_text, mkdir, copy, move, zip in authorized folders (writing there goes through it: no delete).
- `wangp.download(url)` / `wangp.web(action, ...)`: the web when web search is enabled (search, images, open, download, screenshot). Pass an address you extracted (an article's `og:image`, a list of file links) straight from its variable: retyping a long url changes it.
- `wangp.plugins()` / `wangp.plugin(plugin, action, **arguments)`: the toolboxes of enabled Plugins (`wangp_plugins`), such as Media Flow's processing of long videos and folders. `wangp.plugins("media_flow")` lists a Plugin's actions with their calls; `wangp.plugin` runs one and waits for its result.
- `wangp.call(tool, action, arguments)`: any other callable tool. Not callable from scripts: the web, sessions, Python itself.
- `wangp.progress(i, total, name)` as each item of a loop starts shows the user a progress bar from the first item, with the item running; `wangp.progress(total, total)` after the loop completes it. `print` lines appear live.

Each call costs what the tool costs (a generation takes the GPU, an inspection runs the visual model); time inside tools does not count toward `timeout_s`.

## Which skill next

Read the one the next run needs when you are about to write that run, not all of them while planning: early reads are compacted away before use.

- `wangp://skills/python-images`: text over pictures, posters, grids and contact sheets, palettes, masks, procedural SVG.
- `wangp://skills/python-clips`: clips built frame by frame, animated charts and titles, pan and zoom, subtitles (SRT) from timestamps.
- `wangp://skills/python-3d`: 3D models built in Python, read it before any of them: diagrams, 3D charts, props, terrains, 3D text and labels (`wangp.text_mesh`), logos, flags, icons and silhouettes extruded from a picture (`wangp.extrude`, `wangp.extrude_parts` per colour); rendered with `render_3d_views`. An exact shape (a logo, a diagram, a chart) is built here; the image-to-3D models suit organic objects, not flat designs.
- `wangp://skills/python-batch`: loops over many files or generations with WanGP's tools (classify, transcribe, vary, find a person), progress and stopping.
- `wangp://skills/python-data`: tables, CSV, charts and reports from data, files or web results.
- `wangp://skills/web-apps`: interactive animations, games, three.js scenes and toys the user plays in WanGP (`app_new`, `app_check`, `app_publish`), when `wangp_python` lists those actions.

Markdown, CSV, JSON, SVG (and HTML pages with web apps) saved under `out/` open as previews in the chat.

No GPU, raw network, processes or ffmpeg inside the sandbox: generate with the models, cut and mix with the toolbox, reach the web with `wangp.download` / `wangp.web` when web search is enabled.
