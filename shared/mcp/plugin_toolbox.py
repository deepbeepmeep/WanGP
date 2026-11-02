# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""wangp_plugins: the toolboxes that enabled plugins declare (register_deepy_toolbox / register_deepy_action) for every agent, MCP clients,
Deepy Prime and Deepy Python scripts alike. Discovery is progressive: one summary line per plugin in the tool description, the actions of one
plugin with their call signatures, an action's contract, then execution. Routing sentences declared on actions join the agents' instructions
and the discovery of the WanGP tool an action competes with; a plugin's toolbox skills are named by its discovery."""
from __future__ import annotations

import copy
import difflib
import functools
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

TOOL_NAME = "wangp_plugins"
_WILDCARDS = re.compile(r"[*?\[]")


def active_toolboxes() -> dict:
    from shared.utils.plugins import get_deepy_plugin_toolboxes

    return get_deepy_plugin_toolboxes()


def plugin_skills(plugin_id: str | None = None) -> tuple:
    from shared.utils.plugins import get_deepy_plugin_skills

    return tuple(skill for skill in get_deepy_plugin_skills() if plugin_id is None or skill.plugin_id == plugin_id)


def instructions(toolboxes=None, skills=None) -> str:
    """The routing of the enabled plugins' actions and general skills, for the agents' instructions; empty when none declares one."""
    toolboxes = active_toolboxes() if toolboxes is None else toolboxes
    skills = plugin_skills() if skills is None else skills
    lines = [f"- {action.routing}: {TOOL_NAME} plugin=\"{box.toolbox_id}\" action=\"{action.name}\"" + (f" rather than {action.instead_of}" if action.instead_of else "") + "."
             for box in toolboxes.values() for action in box.actions.values() if action.routing]
    reads = [f"- {skill.routing}: read {skill.uri}." for skill in skills if skill.routing]
    parts = ["\n".join([f"Plugin toolboxes ({TOOL_NAME}) add workflows from enabled plugins. Route these requests to them:", *lines]) if lines else "",
             "\n".join(["Skills from enabled plugins add methods. Read them before these tasks:", *reads]) if reads else ""]
    return "\n\n".join(part for part in parts if part)


def tool_description(toolboxes) -> str:
    plugins = "; ".join(f"{box.toolbox_id}: {box.summary}" for box in toolboxes.values())
    return ("Workflows added by enabled WanGP plugins, one toolbox per plugin. Omit plugin to list them; plugin alone lists its actions with their call "
            f"signatures; action with arguments=null reads its contract; action with arguments={{...}} runs it. Plugins: {plugins}")


def action_definition(action) -> dict[str, Any]:
    from .v2 import function_def

    definition = function_def(action.function, action.description, exclude=("ctx",))
    properties = definition["parameters"]["properties"]
    for name, override in action.parameters.items():
        replaced = {"type", "anyOf", "oneOf", "enum", "const"} & override.keys()
        properties[name] = {**({key: value for key, value in properties[name].items() if key == "default"} if replaced else properties[name]), **copy.deepcopy(override)}
    definition["_summary"] = action.summary
    if action.routing:
        definition["use_when"] = action.routing
    if action.example is not None:
        definition["example"] = copy.deepcopy(action.example)
    return definition


def call_policy(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Deepy's runtime policy for a wangp_plugins call: discovery and contracts never pause it; executions follow the action's uses_gpu."""
    arguments = dict(arguments or {})
    box = active_toolboxes().get(str(arguments.get("plugin") or ""))
    action = box.actions.get(str(arguments.get("action") or "")) if box is not None else None
    executes = action is not None and isinstance(arguments.get("arguments"), dict)
    return {"pause_runtime": bool(executes and action.uses_gpu), "pause_reason": "tool"}


def script_problem(arguments: dict[str, Any] | None) -> str:
    """Why a script may not run this call, or an empty string."""
    arguments = dict(arguments or {})
    box = active_toolboxes().get(str(arguments.get("plugin") or ""))
    action = box.actions.get(str(arguments.get("action") or "")) if box is not None else None
    if action is not None and not action.scripts and isinstance(arguments.get("arguments"), dict):
        return f"{TOOL_NAME} {box.toolbox_id}.{action.name} cannot be called from a script: call it directly."
    return ""


def transcript_label(arguments: dict[str, Any] | None, humanize: Callable[[Any], str]) -> str:
    arguments = dict(arguments or {})
    box = active_toolboxes().get(str(arguments.get("plugin") or ""))
    if box is None:
        return "Explore Plugins"
    action = str(arguments.get("action") or "").strip()
    if not action:
        return f"Explore {box.display_name}"
    label = f"{box.display_name}: {humanize(action)}"
    return f"Get {label} Schema" if arguments.get("arguments") is None else label


class PluginHost:
    """Hooks of the server's wangp_plugins tool. Deepy Prime replaces them with its own (its Stop button, its tool card, its re-entrant tool calls and
    its deferred browser queue); an MCP server keeps these defaults, where a background job carries the progress and the cancellation."""

    def __init__(self, mcp) -> None:
        self.interrupted: Callable[[], bool] = lambda: False
        self.progress: Callable[[str, dict[str, Any]], None] | None = None
        self.call_tool: Callable[[str, dict[str, Any]], dict[str, Any]] = lambda name, arguments: _call_inline(mcp, name, arguments)
        self.trigger_queue: Callable[[Any], None] | None = None


def _call_inline(mcp, name, arguments):
    from .server import call_tool_inline

    return call_tool_inline(mcp, name, arguments)


class _QueueSession:
    """The caller's WanGP session for a plugin: a session whose browser queue is started by its host (Deepy Prime) gets that start after each
    submission, which the plugin's own waiting would otherwise never trigger."""

    def __init__(self, session, trigger) -> None:
        self._session, self._trigger = session, trigger

    def __getattr__(self, name):
        return getattr(self._session, name)

    def _watched(self, job):
        def start():
            if job.wait_for_webui_submission_or_completion() and not job.done and not job.cancel_requested:
                self._trigger(job)

        threading.Thread(target=start, name="wangp-plugin-queue", daemon=True).start()
        return job

    def submit_task(self, settings, callbacks=None):
        return self._watched(self._session.submit_task(settings, callbacks=callbacks))

    def submit_manifest(self, settings_list, callbacks=None):
        return self._watched(self._session.submit_manifest(settings_list, callbacks=callbacks))


class PluginContext:
    """What an action receives as ctx: the caller's WanGP state and session, file access checks, progress, cancellation and re-entrant tool calls."""

    def __init__(self, host: PluginHost, session, policy, *, cancel: threading.Event | None = None, job_progress: dict | None = None) -> None:
        self._host, self._policy, self._cancel, self._job_progress, self._caller = host, policy, cancel, job_progress, session
        deferred = (getattr(session, "_gradio_webui_context", None) or {}).get("defer_load_queue_trigger", False)
        self.session = _QueueSession(session, host.trigger_queue) if deferred and host.trigger_queue is not None else session
        self.state = session._state
        self.started = time.monotonic()
        self._reported = 0.0

    @property
    def outputs_dir(self) -> Path:
        """WanGP's outputs folder, where a plugin writes results nobody placed elsewhere."""
        return Path(self._policy.output_roots[0])

    def can_write(self, path) -> bool:
        return self._policy.can_write(path)

    def virtualize(self, path) -> str:
        return self._policy.virtualize_path(path)

    def resolve(self, path) -> Path:
        """The real path of a file the caller names: an @alias path (as in ctx.call results) or a Gallery media id, checked against the caller's
        read access, as for the arguments declared in paths; a real path stays as it is."""
        text = str(path)
        return Path(text) if Path(text).is_absolute() else Path(_authorize(self._policy, self._caller, text, "read", "path"))

    def cancelled(self) -> bool:
        return bool(self._cancel is not None and self._cancel.is_set() or self._host.interrupted())

    def progress(self, done: int | None = None, total: int | None = None, text: str = "", *, force: bool = False) -> None:
        """Report progress (at most three updates per second unless force): done of total items, and a short status text."""
        if not force and time.monotonic() - self._reported < 0.34:
            return
        self._reported = time.monotonic()
        elapsed = time.monotonic() - self.started
        summary: dict[str, Any] = {"status": "running", "live": True, "elapsed_s": round(elapsed, 1)}
        parts = []
        if done is not None:
            summary["progress"] = {"done": int(done), "total": None if total is None else int(total), "text": str(text)[:120]}
            if total:
                parts.append(f"{int(done)}/{int(total)}")
                if 0 < done < total:
                    summary["progress"]["eta_s"] = round(elapsed / done * (total - done))
        if text:
            parts.append(str(text)[:120])
        status_text = " · ".join(parts) or "Running"
        if self._job_progress is not None:
            self._job_progress.update(status=status_text, **({"progress": summary["progress"]} if "progress" in summary else {}))
        if self._host.progress is not None:
            self._host.progress(status_text, summary)

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run another WanGP tool from inside this action, on this thread (the request loop is busy with this action); a GPU tool releases Deepy first."""
        return self._host.call_tool(tool, dict(arguments))


def _parts(text: str) -> list[str]:
    return [part for part in re.split(r"[\\/]", text)]


def _authorize(policy, session, value, role: str, label: str):
    from . import server as core

    if isinstance(value, list):
        return [_authorize(policy, session, item, role, f"{label}[{index}]") for index, item in enumerate(value)]
    if not isinstance(value, str) or not value.strip():
        return value
    text = value.strip()
    if role == "read" and core._is_media_id(text):
        return core._resolve_mcp_media_reference(session, text, label, policy.read_enabled, policy)
    check = policy.require_read if role == "read" else policy.require_write
    if _WILDCARDS.search(text):  # folder part authorized, pattern kept: the plugin expands it
        parts = _parts(text)
        index = next(index for index, part in enumerate(parts) if _WILDCARDS.search(part))
        base = "/".join(parts[:index]) or "."
        root = check(base, directory=True) if role == "read" else check(base)
        return str(Path(root).joinpath(*parts[index:]))
    path = str(check(text))
    return path + "/" if role == "write" and text.endswith(("/", "\\")) else path


def register(mcp, session, jobs, policy, *, allow_async: bool, virtualized: bool) -> PluginHost | None:
    """Add wangp_plugins to this MCP server when enabled plugins declare toolboxes; mcp._wangp_plugins holds its hooks. virtualized names result
    files by the client's @alias roots (external agents; Deepy Prime does it for its own results)."""
    from .v2 import invocation

    toolboxes = active_toolboxes()
    mcp._wangp_plugins = None
    if not toolboxes:
        return None
    host = mcp._wangp_plugins = PluginHost(mcp)
    definitions = {box_id: {name: action_definition(action) for name, action in box.actions.items()} for box_id, box in toolboxes.items()}
    try:
        from gradio import Error as GradioError
    except ImportError:  # pragma: no cover - WanGP always ships gradio
        GradioError = ()

    def run(box, action, arguments, cancel=None, job_progress=None):
        resolved = {name: _authorize(policy, session, value, action.paths[name], name) if name in action.paths else value for name, value in arguments.items()}
        ctx = PluginContext(host, session, policy, cancel=cancel, job_progress=job_progress)
        try:
            result = action.function(ctx, **resolved)
        except GradioError as error:  # plugins share the UI's validation, which raises gr.Error
            raise ValueError(str(getattr(error, "message", error) or error)) from error
        result = result if isinstance(result, dict) else {"status": "done", "content": result}
        result.setdefault("status", "done")
        return policy.virtualize_result(result) if virtualized else result

    def wangp_plugins(plugin: str | None = None, action: str | None = None, arguments: dict[str, Any] | None = None, wait: bool = True) -> dict[str, Any]:
        if not plugin and isinstance(arguments, dict) and isinstance(arguments.get("plugin"), str):  # the call wrapped one level too deep:
            arguments = dict(arguments)                                                                   # {"arguments": {"plugin", "action"?, "arguments"}}
            plugin, action = arguments.pop("plugin"), action or arguments.pop("action", None)
            arguments = arguments.pop("arguments", arguments)
        if not plugin:
            if action or arguments:
                raise ValueError(f"Name the plugin beside action: {{\"plugin\": \"<plugin>\", \"action\": \"{action or '<action>'}\", \"arguments\": {{...}}}}. Plugins: {', '.join(toolboxes)}. Nothing was executed.")
            return {"status": "discovery", "plugins": {box_id: box.summary for box_id, box in toolboxes.items()}, "usage": "List a plugin's actions with plugin=<plugin>."}
        box = toolboxes.get(plugin)
        if box is None:
            close = difflib.get_close_matches(plugin, toolboxes, n=2, cutoff=0.5)
            raise ValueError(f"Unknown plugin: {plugin}. " + (f"Did you mean {' or '.join(close)}? " if close else "") + f"Plugins: {', '.join(toolboxes)}.")
        response = invocation(definitions[plugin], action, arguments, signatures=True)
        if response is not None:
            response["plugin"] = plugin
            if response["status"] == "discovery":
                response["guide"] = box.guide
                skills = {skill.uri: skill.description for skill in plugin_skills(box.plugin_id)}
                if skills:
                    response["skills"] = skills
                response["usage"] = "Execute: plugin + action + arguments={...} following the call (* = required). Read the contract (arguments=null) for rules, defaults and examples." + (" Read the skills (resource URIs) for the plugin's method." if skills else "")
            return response
        selected = box.actions[action]
        if not wait:
            if not allow_async:
                raise ValueError("Asynchronous execution is disabled on this server: omit wait.")
            cancel, job_progress = threading.Event(), {}
            return jobs.start_tool(f"{plugin}.{action}", functools.partial(run, box, selected, arguments, cancel, job_progress), cancel=cancel, progress=job_progress).snapshot()
        return run(box, selected, arguments)

    mcp.tool(name=TOOL_NAME, description=tool_description(toolboxes))(wangp_plugins)
    return host


def finalize(mcp, *, allow_async: bool) -> None:
    """After every tool is registered (and its parameters rebuilt): the job option of wangp_plugins, and the discovery of each WanGP tool that a
    plugin action competes with (instead_of) naming that action."""
    tool = mcp._tool_manager.get_tool(TOOL_NAME)
    if tool is None:
        return
    if allow_async:
        tool.description += " wait=false runs an action in the background and returns a job_id for wangp_session get_job / wait_jobs / cancel_job."
    else:  # without asynchronous jobs (Prime's server) wait=false could only fail
        tool.parameters["properties"].pop("wait")
    hints: dict[str, list[str]] = {}
    for box in active_toolboxes().values():
        for action in box.actions.values():
            if action.instead_of:
                hints.setdefault(action.instead_of, []).append(f"{action.routing or action.summary}: {TOOL_NAME} plugin=\"{box.toolbox_id}\" action=\"{action.name}\".")
    for name, lines in hints.items():
        tool = mcp._tool_manager.get_tool(name)
        if tool is None:
            continue
        original = tool.fn

        @functools.wraps(original)
        def routed(*args, _original=original, _lines=tuple(lines), **kwargs):
            result = _original(*args, **kwargs)
            if isinstance(result, dict) and result.get("status") == "discovery":
                result["see_also"] = list(_lines)
            return result

        tool.fn = routed
