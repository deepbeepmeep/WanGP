# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Interactive exploration of a world of the Worlds gallery, server side. The gallery's full screen explorer sends JSON requests through a
hidden textbox and receives JSON responses through another one:
- {"action": "start", "path", "seconds", "steps", "memory"}: remembers the viewer's Move, Quality and Memory choices in wgp_config.json
  ("world_explorer", per world model architecture) when they changed, and queues an exploration task for the world (its model and resolution
  from the world's manifest, the task settings from the model's menus),
  or, while a generation runs, explores at once by taking the GPU from it (the generation pauses until the viewer leaves), and answers
  once the world is ready ({"action": "ready"}), which takes the model loading when it is not loaded;
- {"action": "move", "path", "keys", "text", "seconds", "steps", "memory"}: one move (held keys as "w+left", key names of the world
  model's explorer controls, such as "jump" for game worlds; optional text event,
  denoising steps, history frames seen), answered with the URL of its clip and the move's GPU statistics ({"action": "clip", "url",
  "stats"});
- {"action": "leave", "path"}: ends the exploration (and its task); the world restarts from its stored state next time;
- {"action": "render", "path", "record"}: the whole video explored so far (the world's own video, then the moves), recorded as a new
  video of the Video gallery (record true) or as a clip to replay ({"action": "rendered", "url", "recorded": its file name});
- {"action": "save", "path"}: the world as explored so far (its own segments, then the moves) saved as a new world of the Worlds gallery,
  to explore or continue later ({"action": "saved", "recorded": its file name}); the gallery shows it once the viewer leaves (refreshing
  it now would replace the viewer);
- {"action": "stats", "path"}: the whole GPU's VRAM use and load and the RAM ({"action": "stats", "stats"}), requested when statistics open
  and when the world becomes ready. Move responses also include updated statistics.
An exploration ends with {"action": "ended"} when its task stops (aborted from the queue); a request that fails answers
{"action": "error", "message"}.
The explorer is the same for every world model; what differs comes from the world's model handler, whose query_world_explorer(base model
type, manifest, memory) every world model (model def "world_outputs") implements. For the memory profile its worlds use (memory: {"profile",
"preload": whether that profile or the VRAM Preload setting keeps blocks in VRAM}) it gives {"controls": the names of the shared explorer
controls its model understands (shared.utils.world_instructions.CONTROLS: the keys the explorer captures, its help, whether it offers text
events), "seconds": [(value, label, frames)], "steps": [(value, label)] (one choice: no Quality menu), "memory": [(value, video seconds,
needs)] (value in the frames unit of "seconds"), "memory_title", "defaults": {"seconds", "steps", "memory"}, "task": settings of the
exploration task}."""
import json
import os
import shutil
import threading
import traceback
import zipfile
from pathlib import Path

import imageio.v3 as iio

from shared.gradio.strip_gallery import gradio_file_url
from shared.utils.gallery_sources import WORLD

CLIP_FOLDER = ".world_exploration"  # in the outputs folder: the clips of the current explorations, removed when they end
MENUS = ("seconds", "steps", "memory")  # the viewer's Move, Quality and Memory menus
_config = {"config": None, "save": None, "model_def_of": None, "handler_of": None, "memory_profile_of": None}  # wgp_config.json, how to save a
# change of it, the model defs and handlers, the memory profile of a model's worlds (bind)
_START_TIMEOUT, _MOVE_TIMEOUT = 3600, 1800  # seconds: model loading or a busy queue can delay the start
_jobs, _clips, _saved = {}, {}, set()  # world path -> queued task, list of its clip paths; explored worlds whose saved world the gallery shows at leave


def world_model_available(manifest):
    """Whether WanGP knows the model of a world: a world made by a model since removed or disabled can be watched, not explored."""
    return _config["model_def_of"](manifest["model_type"]) is not None


def explorer_menus(manifest):
    """The controls and menus of a world's explorer, from the world's model handler (query_world_explorer) for the memory profile its
    worlds use, with that profile ("memory_profile")."""
    model_type = manifest["model_type"]
    memory, architecture = _config["memory_profile_of"](model_type), _config["model_def_of"](model_type)["architecture"]
    return {**_config["handler_of"](model_type).query_world_explorer(architecture, manifest, memory), "memory_profile": memory}


def _saved_choices():
    """wgp_config.json "world_explorer": the last choices per world model architecture."""
    saved = (_config["config"] or {}).get("world_explorer") or {}
    return {key: value for key, value in saved.items() if isinstance(value, dict)}


def explorer_choices(manifest, menus=None):
    """The Move, Quality and Memory choices the viewer opens with for a world: those of the last exploration of a world of the same model
    architecture, where they still are menu choices, else the model's defaults."""
    menus = menus or explorer_menus(manifest)
    saved = _saved_choices().get(_config["model_def_of"](manifest["model_type"])["architecture"], {})
    return {key: saved[key] if saved.get(key) in [choice[0] for choice in menus[key]] else menus["defaults"][key] for key in MENUS}


def _remember_choices(request, manifest):
    menus = explorer_menus(manifest)
    choices = {key: request.get(key) for key in MENUS}  # the browser's values: kept only when they are menu choices
    choices = {key: next((choice[0] for choice in menus[key] if choice[0] == float(value or 0)), None) for key, value in choices.items()}
    if _config["save"] is not None and None not in choices.values() and choices != explorer_choices(manifest, menus):
        _config["save"]({**_saved_choices(), _config["model_def_of"](manifest["model_type"])["architecture"]: choices})


def _manifest(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def _submit(state, settings):
    from shared.api import WanGPSession
    from shared.deepy.hybrid import service_for
    service = service_for(state)
    session = WanGPSession(webui_state=service._state, console_output=False, console_isatty=False, webui_append=True)
    session._gradio_webui_context = {"load_queue": service.load_inline_queue}
    return session.submit_task(settings)


def _wait(exploration, kinds, timeout, job=None):
    """The next event of kinds; with the exploration's queued task (job), an error event when the task ends without one."""
    waited = 0.0
    while waited < timeout and not exploration.left.is_set():
        finished = job is not None and job.done  # SessionJob.done: a property
        event = exploration.next_event(0.5)
        waited += 0.5
        if event is not None and event["type"] in kinds:
            return event
        if finished:
            return {"type": "error", "message": "The exploration task ended before the world was ready (see the console)"}
    return None


def _remove_clips(path):
    for clip in _clips.pop(path, []):
        try:
            os.remove(clip)
        except OSError:
            pass


def _explore_now(explore_now, state, model_type, exploration):
    try:
        explore_now(state, model_type, exploration.world_path)
    finally:
        exploration.publish("ended")  # also when the model could not start (its own end message comes first otherwise)


def handle_request(state, request, outputs_folder, explore_now=None):
    from shared.utils.world_exploration import EXPLORE_PROMPT, exploration_for, open_exploration
    request = json.loads(request)
    action, path = request["action"], request["path"]
    if action == "stats":
        from shared.utils.stats import live_stats
        return json.dumps({"action": "stats", "path": path, "stats": live_stats()})
    if action == "start":
        manifest = _manifest(path)
        if not world_model_available(manifest):  # a page opened before the model was removed
            return json.dumps({"action": "error", "path": path, "message": f"the model of this world ({manifest['model_type']}) is not available"})
        _remember_choices(request, manifest)
        exploration = open_exploration(path)
        settings = {"model_type": manifest["model_type"], "prompt": EXPLORE_PROMPT, "custom_guide": path, "resolution": f"{manifest['width']}x{manifest['height']}",
                    "guidance_scale": 1.0, "seed": 0, "multi_prompts_gen_type": "FG", **explorer_menus(manifest)["task"]}
        state["gen"]["last_gallery_source"] = WORLD.id  # the task records no output: the Worlds gallery stays shown when it ends
        from shared.utils.process_locks import main_generation_running
        if explore_now is not None and main_generation_running(state):  # not behind the running generation: it pauses meanwhile
            threading.Thread(target=_explore_now, args=(explore_now, state, manifest["model_type"], exploration), daemon=True).start()
        else:
            _jobs[path] = _submit(state, settings)
        event = _wait(exploration, ("ready", "ended"), _START_TIMEOUT, _jobs.get(path))
        if event is not None and event["type"] == "error":
            exploration.leave()
            return json.dumps({"action": "error", "path": path, "message": event["message"]})
        return json.dumps({"action": event["type"] if event else "ended", "path": path})
    exploration = exploration_for(path)
    if action == "leave":
        if path in _saved:  # the worlds saved while exploring: the gallery shows them now
            _saved.discard(path)
            from shared.deepy.hybrid import service_for
            service_for(state).host_changed()
        if exploration is not None:
            exploration.leave()
        _jobs.pop(path, None)
        _remove_clips(path)
        return json.dumps({"action": "left", "path": path})
    if exploration is None:
        return json.dumps({"action": "ended", "path": path})
    if action == "render":
        exploration.render(bool(request.get("record")))
        event = _wait(exploration, ("rendered", "ended"), _MOVE_TIMEOUT)
        if event is None or event["type"] == "ended":
            return json.dumps({"action": "ended", "path": path})
        if event["record"]:
            from shared.deepy.hybrid import service_for
            from shared.gradio.gallery_snapshots import add_exploration_video
            output = add_exploration_video(service_for(state), path, event["frames"], event["fps"], event["moves"])
            return json.dumps({"action": "rendered", "path": path, "url": gradio_file_url(str(output)), "recorded": output.name})
        folder = Path(outputs_folder) / CLIP_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        clip = str(folder / f"{exploration.token}_replay_{event['moves']:04d}.mp4")
        iio.imwrite(clip, event["frames"], fps=event["fps"], codec="libx264", quality=8)
        _clips.setdefault(path, []).append(clip)
        return json.dumps({"action": "rendered", "path": path, "url": gradio_file_url(clip)})
    if action == "save":
        exploration.save()
        event = _wait(exploration, ("saved", "ended"), _MOVE_TIMEOUT)
        if event is None or event["type"] == "ended":
            return json.dumps({"action": "ended", "path": path})
        from shared.deepy.hybrid import service_for
        from shared.gradio.gallery_snapshots import add_exploration_world
        output = add_exploration_world(service_for(state), path, event["data"], event["moves"])
        _saved.add(path)
        return json.dumps({"action": "saved", "path": path, "recorded": output.name})
    keys = request.get("keys") or ""
    exploration.move({"keys": keys} if keys else None, request.get("text") or "", float(request["seconds"]), int(request["steps"]), int(request["memory"]))
    event = _wait(exploration, ("clip", "ended"), _MOVE_TIMEOUT)
    if event is None or event["type"] == "ended":
        return json.dumps({"action": "ended", "path": path})
    folder = Path(outputs_folder) / CLIP_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    clips = _clips.setdefault(path, [])
    clip = str(folder / f"{exploration.token}_{len(clips):04d}.mp4")
    iio.imwrite(clip, event["frames"], fps=event["fps"], codec="libx264", quality=8)
    clips.append(clip)
    return json.dumps({"action": "clip", "path": path, "url": gradio_file_url(clip), "stats": event.get("stats")})


def clear_clip_folder(outputs_folder):
    """Removes the clips left by explorations of a previous run."""
    shutil.rmtree(Path(outputs_folder) / CLIP_FOLDER, ignore_errors=True)


def bind(state, outputs_folder_of, model_def_of, handler_of, explore_now=None, config=None, save_choices=None, memory_profile_of=None):
    """The request / response textboxes of the Worlds gallery explorer (outputs_folder_of(state): the folder of its clips; model_def_of(model_type):
    a model's definition; handler_of(model_type): its handler, whose query_world_explorer gives the controls and menus of its worlds;
    explore_now(state, model_type, world_path): serves an exploration
    outside the queue, taking the GPU from a running generation; config: wgp_config.json, save_choices(choices): saves the viewer's choices
    in it; memory_profile_of(model_type): the memory profile of a model's worlds, {"profile", "preload"})."""
    _config.update(config=config, save=save_choices, model_def_of=model_def_of, handler_of=handler_of, memory_profile_of=memory_profile_of)
    import gradio as gr
    request = gr.Textbox(visible=False, elem_id="wangp_world_request")
    response = gr.Textbox(visible=False, elem_id="wangp_world_response")
    # each request runs on its own (a leave must not wait for a move being generated)
    def answer(state_value, value):
        try:
            return handle_request(state_value, value, outputs_folder_of(state_value), explore_now)
        except Exception as exc:  # the viewer shows the error instead of waiting for an answer that never comes
            traceback.print_exc()
            return json.dumps({"action": "error", "path": json.loads(value)["path"], "message": str(exc) or exc.__class__.__name__})

    request.input(answer, inputs=[state, request], outputs=[response],
                  queue=True, concurrency_limit=None, trigger_mode="multiple", show_progress="hidden", api_name=False).then(
        fn=None, inputs=[response], outputs=None, js="(value) => { if (window.WangpWorld) window.WangpWorld.onResponse(value); }")
