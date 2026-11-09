# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deepy as the production's director: one Deepy job at a time per production, in the production's own session and workspace.

Deepy keeps its loop (generate, inspect with its own tools, decide the next step) inside the job and records its work in the
production document through the showrunner toolbox; the user's edits and verdicts reach it through the same document."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
import re
from typing import Any

from shared.deepy import session_store

from . import media, production as model

TASKS = ("plan", "shoot", "continue", "message", "reshoot", "assemble", "review", "resume")
KIND_SKILLS = {"music_video": "wangp://skills/music-video", "short_film": "wangp://skills/edited-film", "trailer": "wangp://skills/edited-film",
               "story": "wangp://skills/production", "ad": "wangp://skills/production", "remake": "wangp://skills/showrunner/remaking", "free": "wangp://skills/production"}  # long-story-writing is for long texts, and only listed with long-text tools
KIND_NAMES = {"music_video": "music video", "short_film": "short film", "ad": "ad", "trailer": "trailer", "story": "illustrated story", "remake": "remake of an existing video", "free": "production"}
KIND_RULES = {"story": "An illustrated story is narrated picture-book pages: each page a still picture (template_advice `pages`, or `pages_with_cast` for the pages "
                       "showing the cast) held as long as its narration, a narrator (a cast member with role narrator) reading every page, crossfades between "
                       "pages; the edit moves slowly over each still. A page moves (a video template) only where the brief asks for motion."}


def _name(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _brief_text(doc: dict[str, Any]) -> str:
    """The brief in words. Media are named only: the production action gives their paths as Deepy uses them."""
    brief = doc["brief"]
    lines = [f"Kind: {KIND_NAMES[brief['kind']]}." + (f" {KIND_RULES[brief['kind']]}" if brief["kind"] in KIND_RULES else "")]
    if brief.get("text"):
        lines.append(f"Concept: {brief['text']}")
    if brief.get("script"):
        lines.append(f"{'Lyrics of the song' if brief['kind'] == 'music_video' else 'Script'} (keep its words exactly):\n{brief['script']}")
    if brief.get("source"):
        lines.append(f"Source video to remake (its motion, camera, timing and soundtrack stay): {_name(brief['source'])}")
    if brief.get("song"):
        lines.append(f"Song (drives timing and lip sync): {_name(brief['song'])}")
    if brief.get("references"):
        lines.append("Reference media: " + ", ".join(_name(path) for path in brief["references"]))
    format_parts = [f"aspect {brief['aspect']}", f"{brief['resolution']}" if not brief.get("draft_resolution") else f"{brief['draft_resolution']} draft for {brief['resolution']}"]
    if brief["kind"] != "remake" and (brief.get("duration_s") or not brief.get("song")):  # a remake lasts as long as its source
        format_parts.append(f"about {brief.get('duration_s') or model.DEFAULT_LENGTH_S:g} s in total")
    lines.append("Format: " + ", ".join(format_parts) + ".")
    if brief.get("draft_resolution"):
        lines.append(f"Make every picture (cast, looks, places, cards) at {model.resolution(brief)}, the target: a draft and its final use the same pictures. render shoots video at the draft size by itself; the user shoots shots final when happy.")
    style = ", ".join([*brief.get("style_tags", []), brief.get("style", "")]).strip(", ")
    if style:
        lines.append(f"Style: {style}.")
    pacing = int(brief.get("pacing") or 0)
    if pacing:
        lines.append("Pacing: " + ("faster cuts than usual." if pacing > 0 else "longer, calmer shots than usual."))
    if model.shot_range_words(doc):
        lines.append(f"Shot length, the user's choice: video shots of {model.shot_range_words(doc)}.")
    if brief.get("voices_from_model"):
        lines.append("Voices, the user's choice: the video model says the lines, in every shot that follows no song or audio of its own: template_advice says how, and this replaces `speak` for those shots.")
    if brief.get("budget_minutes"):
        lines.append(f"GPU budget: about {brief['budget_minutes']:g} minutes of generation in total.")
    if brief.get("review_before_edit"):
        lines.append("Review before the edit: the user approves every take before the edit is assembled. Never assemble on your own: when the shots are shot "
                     "and judged, stop and ask the user to review them.")
    if doc["cast"]:
        lines.append("Cast: " + "; ".join(f"{member['id']} {member['name'] or '(unnamed)'}" + (f", {member['role']}" if member["role"] else "")
                                          + (f", {len(member['images'])} picture(s)" if member["images"] else "")
                                          + (", a voice sample" if member.get("voice") else "") + (" [locked]" if member["locked"] else "") for member in doc["cast"]) + ".")
    if brief.get("song") or brief.get("references") or doc["cast"]:
        lines.append("The production action lists these media with the paths to use in settings.")
    if brief.get("read_other_productions"):
        lines.append("The user lets you look at their other productions when a request needs them: `list_productions` lists them, `production` reads one, "
                     "import_media copies its files into this one; never change them.")
    return "\n".join(lines)


# Deepy's standing rules while it directs a production, in its system prompt for the job only. The same for every production, so that the
# rendered prompt stays reusable between jobs; the method itself is the showrunner skill.
INSTRUCTIONS = """## Showrunner
You are the showrunner of a production for the user: you plan, cast, shoot, edit and deliver it while the user follows and edits it on a storyboard and a timeline. In your messages you are the showrunner. Read wangp://skills/showrunner/directing before your first decision.
- The production document is your plan and memory: read it with the showrunner toolbox (wangp_plugins with plugin="showrunner", action "production") and record your work with its actions. Do not write another plan file.
- Every toolbox call has this shape, plugin and action beside arguments, never inside them: wangp_plugins({"plugin": "showrunner", "action": "<action>", "arguments": {"production": "<id>", ...}}).
- Render shots with `render`: it adds the cast's pictures and the driving audio (song slice or spoken line) that the shot's template needs, which a direct wangp_generate call does not. `assemble` reports the delivered file's streams, loudness and true peak: do not parse the export yourself.
- Locked shots and cast members are the user's, and the user's edits and verdicts override yours. Read the document again after each long step.
- Shots showing a cast member take that member's pictures as reference images, never as the start image. Follow template_advice and fix the warnings returned.
- A shot's generation names its template and gives only what the shot changes (prompt, media): the template brings the model, LoRAs and steps, the brief the resolution and its duration_s the length. `plan` and `add_shots` validate the shots and return what they will render at (`renders`): quote those numbers when you tell the user a size or a length.
- Words meant to be read (titles, taglines, names, captions) are edit overlays or drawn cards, never part of a video prompt: video models misspell them.
- Spoken and sung words get subtitles unless the brief says otherwise: write them in their shots' dialogue.
- Each speaker keeps one voice for the whole production: give every speaker (the narrator too, as a cast member with role narrator) a voice_description and make their lines with `speak`. A voice described again for each line comes out different every time. If the user asks for the video model's own voices instead of recordings, call edit(voices_from_model=true).
- Choose a text design that suits the production (style.text_design) and vary cards with it; a chapter card over a dimmed still of the scene it opens beats a plain page.
- Inspect every take before you judge it, and log one short line for each decision the user should know about.
- Your replies and log lines are for the user, a filmmaker: short and in plain words. Name a shot by its id (s7: the page shows its number) or its title, never "shot 7", people by their names and an export by its version number (assemble returns it: "version 3"); leave out file paths, revisions, settings keys, model, template and tool names (shot ids are fine: the page shows them as shot numbers)."""


def _task_words(task: str, message: str = "", shots: list[str] | tuple[str, ...] = ()) -> str:
    return {"plan": "planning the production", "shoot": f"shooting {', '.join(shots)}" if shots else "shooting the production", "reshoot": f"re-shooting {', '.join(shots)}", "assemble": "assembling the edit", "review": "reviewing the last version in depth",
            "continue": "continuing the production"}.get(task) or f"working on the user's note: {message}"


def repeating_failure(failures: dict[str, int], tool: dict[str, Any]) -> bool:
    """Whether this failed call is the third of its kind in the task (the tool and its error, without the numbers and file names that change from one try to the next). A loop of guesses around one
    tool failure cost a production half an hour."""
    words = re.sub(r"[@\w\-. ]*[\\/][^\s'\"]*|\d+", "#", str(tool.get("error") or ""))[:80]
    key = f"{tool.get('label') or tool.get('name')}|{words}"
    failures[key] = failures.get(key, 0) + 1
    return failures[key] == 3


def method_key(kind: str = "") -> str:
    """What the director's method is: its standing instructions and its skills. A production's conversation that began under another
    method still holds the old skills: the next task asks the director to read them again."""
    skills = (Path(__file__).parent / "skills" / name / "SKILL.md" for name in ("directing", *(("remaking",) if kind == "remake" else ())))
    return hashlib.sha1("".join((INSTRUCTIONS, *(skill.read_text(encoding="utf-8") for skill in skills))).encode("utf-8")).hexdigest()[:12]


def _shoot_body(doc: dict[str, Any]) -> str:
    """How the whole production is shot: the shoot task's request, and the plan's when the user does not review the plan."""
    review = doc["brief"].get("review_before_edit")
    body = ("Shoot the production: render the included shots that still need a good take, in order, inspect each take and record a verdict, then "
            + ("stop and ask the user to review the takes: they approve them before the edit is assembled." if review else "assemble the edit, check it and report."))
    length = doc["brief"].get("duration_s") or (doc["audio"].get("analysis") or {}).get("duration_s") or 0
    if len(doc["shots"]) > 24 or length > 90:
        body += " Work sequence by sequence" + ("." if review or length <= 300 else " and assemble after each one, so the user can follow the edit as it grows.")
    return body


def build_prompt(doc: dict[str, Any], task: str, message: str = "", shots: list[str] | tuple[str, ...] = (), new_version: bool = True) -> str:
    """new_version False: the user fixes shot after shot before watching, and makes the next version themselves."""
    kind = doc["brief"]["kind"]
    header = (f"[Showrunner] Production \"{doc['title']}\" (production id: {doc['id']}). Method for its kind: {KIND_SKILLS[kind]}. "
              f"Read it with wangp_plugins(plugin=\"showrunner\", action=\"production\", arguments={{\"production\": \"{doc['id']}\"}}).")
    review_plan = doc["brief"]["plan_first"]
    then = (" Do not render shots: the user reviews and edits the plan first, then asks you to shoot." if review_plan else
            " The user chose not to review the plan: once it is ready, go on to shoot it in this same task. " + _shoot_body(doc))
    if task == "plan":
        body = ("Plan the production from its brief: cast (with a master image of each recurring person, object and place that has no picture yet, and one in the "
                "production's style for a member whose pictures have another look), style, and every shot with its template, complete generation settings and "
                "duration." + then)
        length = doc["brief"].get("duration_s") or (doc["audio"].get("analysis") or {}).get("duration_s") or 0
        if length > 90:
            body += f" It runs about {length / 60:.1f} minutes: outline its sequences first, then plan each sequence's shots (the skill's long productions section)."
        if kind == "remake":
            body = ("Plan the remake from its brief, following the method above. Change only what the brief asks for; if it does not say what should change, ask the user and stop. Read the source with ingest_source, look at its shots, cast whoever changes, prepare each shot "
                    "as its template needs (painted frames, references, a mask) and write each shot's prompt." + then)
    elif task == "shoot" and shots:
        body = (f"Shoot {', '.join(shots)} and leave every other shot and its takes as they are: the user asked for these alone and nothing is wrong with them, so there is no fault to look for. "
                "Render them, inspect each take and record a verdict, then stop and say how it came out. Do not assemble: the user decides what comes next.")
    elif task == "shoot":
        body = _shoot_body(doc)
    elif task == "reshoot":
        note = f' The user says: "{message}".' if message else " The user wrote no note: if no note or verdict in the document says what is wrong, shoot them again without looking for a fault."
        body = (f"Fix shots {', '.join(shots)} and leave every other shot and its takes as they are.{note} Read the user's notes and verdicts in the document first. What the user asks for is theirs to decide, a new length or other content included: do it. When they say what is wrong rather "
                "what to do, find the cause before you shoot: when a new take would not change it (the song, a subtitle, a trim, a card), fix that instead, and when that touches other shots, say which and what it costs before you do it. "
                "Render new takes where they are needed, inspect them and record verdicts, then " + ("stop and ask the user to review them." if doc["brief"].get("review_before_edit") else
                "assemble a new version, check that the fault is gone and say what changed." if new_version else "stop and say how they came out. Do not assemble: the user makes the next version when they are ready."))
    elif task == "assemble":
        body = "Assemble the edit with the showrunner assemble action, check the result (picture, sound, length) and report what you find."
    elif task == "review":
        body = ("Review the last version in depth, and change nothing. Run the showrunner deep_review action: it measures words, subtitles, voices, lips, sound and picture, and says when each shot starts (shots_at). Then look at a frame of "
                "every shot of the version against the pictures of its cast and place (inspect_media): another face, an extra person, another set, a costume that changed. Record what differs with deep_review(add=[{shot, at_s, text}]), one short "
                "sentence each. Then tell the user the faults that matter most, in a few lines, and what you would do about them. Do not fix anything: the user decides.")
    elif task == "continue":
        last = doc["director"].get("last") or {}
        ended = {"interrupted": "was interrupted (WanGP stopped)", "cancelled": "was stopped", "error": "failed"}.get(str(last.get("status")), "ended")
        body = (f"Continue the production where it stopped: read the document to see what is left. The previous task {ended}: nothing it started "
                "(a render, a generation) is still running, and a call it was waiting for will not return; render again what has no take. Takes that have your verdict are judged: do not inspect them again.")
    elif task == "resume":
        paused = doc["director"].get("paused") or {}
        body = (f"The user paused you while you were {_task_words(paused.get('task') or 'continue', paused.get('message', ''), paused.get('shots') or ())}, and now resumes: "
                "finish that task. Read the document first: the user may have changed shots, verdicts or the edit meanwhile. Nothing you started is still "
                "running and a call you were waiting for will not return: render again what has no take. Takes that have your verdict are judged: do not inspect them again.")
        if paused.get("message") and (paused.get("task") or "continue") in ("plan", "shoot", "reshoot", "assemble", "review", "continue"):  # the task's own note: a conversation cut before its turn was saved lost it
            body += f"\n\nWhen that task began, the user wrote: {paused['message']} (it may be done already: check the document before you act on it)."
        if paused.get("new_version") is False:
            body += "\n\nThat task makes no new version: do not assemble, the user makes the next version when they are ready."
        if message:
            body += f"\n\nThe user adds: {message}"
    else:
        body = (f"Shots concerned: {', '.join(shots)}.\n" if shots else "") + f"The user says: {message}" + ("" if new_version else "\n\nDo not assemble a new version: the user makes it when they are ready.")
    brief = _brief_text(doc) if task in ("plan", "shoot", "continue", "resume") else ""
    return "\n\n".join(part for part in (header, f"Brief:\n{brief}" if brief else "", body) if part)


class Director:
    def __init__(self, store, host, settle=None):
        self.store, self.host, self.settle = store, host, settle or (lambda production_id: [])
        self._jobs: dict[str, Any] = {}
        self._pausing: dict[str, bool] = {}  # productions whose director stops after its current step
        self._tools: dict[str, set] = {}  # the tool calls each director job is running
        self._failures: dict[str, dict[str, int]] = {}  # a task's failed calls, by kind
        self._noted: dict[str, bool] = {}  # a note written during a batch
        self._live: dict[str, dict[str, Any]] = {}  # what each production's last job showed (its text and steps), for a page that missed the events
        self._lock = threading.Lock()

    def conversation(self, production_id: str, after: int = 0) -> dict[str, Any]:
        """What the Deepy tab shows when it opens or catches up: the conversation Deepy saved, since `after`, its media links working after a restart."""
        from shared.utils.downloads import restore_file_downloads
        folder = self.store.get(production_id).folder
        updates = session_store.read_ui_updates(folder / "session", after)
        restore_file_downloads(json.dumps(updates["events"]), [folder])  # the production's files: its media, the session's and the cache's
        return updates

    def status(self, production_id: str) -> dict[str, Any]:
        production = self.store.get(production_id)
        doc = production.snapshot()
        with self._lock:
            job = self._jobs.get(production_id)
        state = doc["director"].get("state", "idle")
        if state not in ("idle", "paused") and job is None:  # a job of a process that stopped (WanGP restarted): nothing works on it now
            fields = self._paused_fields(doc) if state == "pausing" else {"state": "idle", "last": {"status": "interrupted", "task": doc["director"].get("task")}}
            production.apply([{"op": "director", "fields": fields}], by="system", record=False)
            self.settle(production_id)
            doc = production.snapshot()
        info = self.host.director_info()
        live = self._live.get(production_id)
        return {"state": doc["director"].get("state", "idle"), "task": doc["director"].get("task"), "running": job is not None and not job.done(),
                "available": info["available"], "reason": info["reason"], "info": info, "last": doc["director"].get("last"),
                **({"live": {**live, "tools": list(live["tools"].values())[-20:]}} if live else {})}

    def start(self, production_id: str, task: str, message: str = "", shots: list[str] | None = None, new_version: bool = True) -> dict[str, Any]:
        if task not in TASKS:
            raise ValueError(f"task must be one of {', '.join(TASKS)}.")
        if task == "message" and not str(message or "").strip():
            raise ValueError("Write a message for the showrunner.")
        current = self.store.get(production_id).snapshot()
        brief = current["brief"]
        if task == "plan" and not any(str(brief.get(key) or "").strip() for key in ("text", "script", "song")):
            raise ValueError("Write your idea in the brief first (or drop a song or a script): the showrunner plans from it.")
        if task == "resume" and current["director"].get("state") != "paused":
            raise ValueError("The showrunner is not paused.")
        reason = self.host.director_unavailable()
        if reason:
            raise RuntimeError(reason)
        production = self.store.get(production_id)
        with self._lock:
            job = self._jobs.get(production_id)
            if job is not None and not job.done():
                raise RuntimeError("The showrunner is already working on this production: send your message as a steering note, or stop it first.")
            self.settle(production_id)  # no job runs here now: a shot still shown rendering was interrupted, and the showrunner must render it again
            doc = production.snapshot()
            workspace = self.host.production_workspace(doc["title"], production.folder / "media")
            prompt = build_prompt(doc, task, message, shots or (), new_version)
            method = method_key(doc["brief"]["kind"])
            if doc["director"].get("method", "") != method and (production.folder / "session" / "context.json").is_file():
                prompt += "\n\nThe Showrunner method changed since your last task in this production: read wangp://skills/showrunner/directing" + (" and wangp://skills/showrunner/remaking" if doc["brief"]["kind"] == "remake" else "") + " again before you act."
            job = self.host.run_director(prompt, session_dir=production.folder / "session", workspace=workspace, title=f"Showrunner - {doc['title']}", instructions=INSTRUCTIONS)
            job.chat_events = True
            self._jobs[production_id] = job
            self._pausing.pop(production_id, None)
            self._tools[production_id] = set()
            self._live[production_id] = {"text": "", "tools": {}, "status": None, "thought": None, "stats": None}
            self._failures[production_id] = {}
            self._noted.pop(production_id, None)
        # a resumed task keeps the paused one's task, note and shots: pausing again resumes the same work
        paused = (doc["director"].get("paused") or {}) if task == "resume" else {}
        work = {"task": paused.get("task") or "continue", "message": paused.get("message", ""), "shots": paused.get("shots") or [], "new_version": paused.get("new_version", True)} if task == "resume" else \
            {"task": task, "message": message, "shots": list(shots or []), "new_version": bool(new_version)}
        stage = {"plan": "planning", "shoot": "shooting", "reshoot": "shooting", "assemble": "editing"}.get(work["task"])
        production.apply([{"op": "director", "fields": {"state": "queued", **work, "paused": None, "started": time.time(), "workspace": workspace, "method": method}},
                          *([{"op": "stage", "stage": stage}] if stage else []), {"op": "log", "text": self._log_line(task, message, shots, new_version)}], by="user", record=False)
        threading.Thread(target=self._follow, args=(production_id, job), name=f"Showrunner director {production_id}", daemon=True).start()
        return self.status(production_id)

    @staticmethod
    def _log_line(task, message, shots, new_version=True):
        if task == "resume":
            return f"Asked the showrunner to resume: {message}" if message else "Asked the showrunner to resume."
        what = {"plan": "Asked the showrunner to plan the production.", "shoot": f"Asked the showrunner to shoot {', '.join(shots)}." if shots else "Asked the showrunner to shoot the production.", "continue": "Asked the showrunner to continue.",
                "assemble": "Asked the showrunner to assemble the edit.", "review": "Asked the showrunner for a deep review of the last version.", "reshoot": f"Asked the showrunner to fix {', '.join(shots or [])}{': ' + message if message else '.'}"}.get(task)
        return (what or f"Told the showrunner: {message}") + ("" if new_version else " No new version for now.")

    def steer(self, production_id: str, text: str) -> None:
        with self._lock:
            job = self._jobs.get(production_id)
        if job is None or job.done():  # a note to a paused director resumes its task with the note
            self.start(production_id, "resume" if self.status(production_id)["state"] == "paused" else "message", text)
            return
        job.steer(text)
        self._noted[production_id] = True
        self.store.get(production_id).apply([{"op": "log", "text": f"Told the showrunner: {text}"}], by="user", record=False)

    def note_waiting(self, production_id: str) -> bool:
        """The user wrote while a long batch runs: said once, for the toolbox to stop between its shots, as a note otherwise waits until the whole batch has ended."""
        return self._noted.pop(production_id, False)

    def stop(self, production_id: str) -> None:
        with self._lock:
            job = self._jobs.get(production_id)
            self._pausing.pop(production_id, None)
        if job is not None and not job.done():
            job.cancel()
        elif self.store.get(production_id).snapshot()["director"].get("state") == "paused":  # a paused task the user gives up
            self.store.get(production_id).apply([{"op": "director", "fields": {"state": "idle", "paused": None, "last": {"status": "cancelled", "time": time.time()}}}],
                                                by="user", record=False)

    def pause(self, production_id: str, now: bool = False) -> dict[str, Any]:
        """Pause the director after its current step (a render or a generation finishes and is recorded first), or at once (now: what it was
        generating is stopped, and rendered again on resume). The production keeps the task and the showrunner's conversation in its folder:
        resume() continues them, also after WanGP restarts. Once the director has stopped, the GPU is freed."""
        production = self.store.get(production_id)
        with self._lock:
            job = self._jobs.get(production_id)
            if job is None or job.done():
                raise RuntimeError("The showrunner is not working on this production.")
            self._pausing[production_id] = True
            busy = bool(self._tools.get(production_id))
        production.apply([{"op": "director", "fields": {"state": "pausing"}}, {"op": "log", "text": "Asked the showrunner to pause" + (" now." if now or not busy else " after its current step.")}],
                         by="user", record=False)
        if now or not busy:
            job.cancel()
        return self.status(production_id)

    def pausing(self, production_id: str) -> bool:
        """Whether the director should stop starting new work (the toolbox checks it between the shots of a render)."""
        return self._pausing.get(production_id, False)

    def resume(self, production_id: str, message: str = "") -> dict[str, Any]:
        return self.start(production_id, "resume", message)

    @staticmethod
    def _paused_fields(doc: dict[str, Any]) -> dict[str, Any]:
        director = doc["director"]
        return {"state": "paused", "paused": {"task": director.get("task"), "message": director.get("message", ""), "shots": director.get("shots") or [], "new_version": director.get("new_version", True), "time": time.time()},
                "last": {"status": "paused", "task": director.get("task"), "time": time.time()}}

    def _tell_user(self, production, error: str | None) -> None:
        """Settings > Tell me when a task ends: the sound and the user's notification destinations, for when the page is out of sight. A task's end stays
        a success when its notification fails."""
        try:
            if self.host.settings()["notify_turns"]:
                doc = production.snapshot()
                self.host.notify(f"Showrunner: {doc['title']}", f"It stopped with an error: {str(error)[:300]}" if error else model.next_step(doc))
        except Exception as failure:
            print(f"[Showrunner] The notification failed: {failure}", flush=True)

    def _follow(self, production_id: str, job) -> None:
        production = self.store.get(production_id)
        publish = lambda kind, data=None: self.store.publish({"type": "director", "production": production_id, "event": kind, "data": data})
        live = self._live[production_id]
        try:
            for event in job.events():
                if event.kind == "queued":
                    publish("queued")
                elif event.kind == "started":
                    production.apply([{"op": "director", "fields": {"state": "running"}}], by="system", record=False)
                    publish("started")
                elif event.kind == "text":
                    live["text"] = (event.data.get("last") or event.data["text"])[-4000:]
                    publish("text", {"text": live["text"]})
                elif event.kind == "status":
                    live["status"] = event.data
                    publish("status", event.data)
                elif event.kind == "thought":
                    live["thought"] = {"text": event.data["text"][-4000:], "streaming": event.data["streaming"]}
                    publish("thought", live["thought"])
                elif event.kind == "chat":
                    if self.host.settings()["show_deepy_tab"]:  # the Deepy tab streams it
                        publish("chat", event.data)
                elif event.kind == "stats":
                    live["stats"] = event.data
                    publish("stats", event.data)
                elif event.kind == "tool":
                    if event.data.get("status") == "error" and repeating_failure(self._failures.setdefault(production_id, {}), event.data):
                        job.steer("(Automatic note, not from the user.) This call has failed the same way three times. Stop trying other ways around it: go on with the rest of the task if you can, and say in your report what fails and the exact error.")
                    running = self._tools.setdefault(production_id, set())
                    (running.add if event.data.get("status") == "running" else running.discard)(event.data.get("id"))
                    live["tools"][event.data.get("id")] = event.data
                    publish("tool", event.data)
                    if self._pausing.get(production_id) and not running:  # its step is done: pause now
                        job.cancel()
                elif event.kind == "progress":
                    data = event.data
                    publish("progress", {"phase": data.phase, "status": data.status, "progress": data.progress, "step": data.current_step, "steps": data.total_steps})
                elif event.kind == "output":
                    self._bin_output(production, event.data)
                    publish("output", event.data)
            result = job.result()
            if self._pausing.pop(production_id, False) and result.status == "cancelled":
                production.apply([{"op": "director", "fields": self._paused_fields(production.snapshot())},
                                  {"op": "log", "text": "The showrunner paused. Resume continues its task where it stopped, in the same conversation."}], by="system", record=False)
                self.settle(production_id)
                publish("paused", {"status": "paused"})
                self.host.release_gpu()
                return
            last = {"status": result.status, "text": (getattr(result, "reply", "") or result.text or "")[:4000], "error": result.error, "time": time.time()}
            ops = [{"op": "director", "fields": {"state": "idle", "last": last}}]
            if result.status == "done" and production.snapshot()["director"].get("task") == "plan":
                ops.append({"op": "stage", "stage": "review"})
            if result.status == "error":
                ops.append({"op": "log", "text": f"The showrunner stopped with an error: {result.error}"})
            elif result.status == "cancelled":
                ops.append({"op": "log", "text": "The showrunner was stopped."})
            production.apply(ops, by="system", record=False)
            if last["text"]:  # its last words stay in the conversation
                production.apply([{"op": "log", "text": last["text"]}], by="deepy", record=False)
            publish(result.status, last)
            if result.status in ("done", "error"):
                self._tell_user(production, result.error if result.status == "error" else None)
        except Exception as error:  # the job thread reports to the page instead of dying silently
            production.apply([{"op": "director", "fields": {"state": "idle", "last": {"status": "error", "error": str(error), "time": time.time()}}}], by="system", record=False)
            publish("error", {"error": str(error)})
        finally:
            self._pausing.pop(production_id, None)
            self._tools.pop(production_id, None)
            job.close()

    def _bin_output(self, production, output: dict[str, Any]) -> None:
        """Media Deepy made that no take records yet (masters, tests, songs) go to the production's bin, where the user can use them; a take
        recorded later leaves it. What it made only to measure or to cut (a vocal stem, a song's slices, frames of a take) stays out."""
        path = output.get("path")
        if path and _WORKING_FILE.search(Path(path).name):
            return
        doc = production.snapshot()
        recorded = {str(production.resolve(take["path"])) for shot in doc["shots"] for take in shot["takes"]} | {str(production.resolve(item["path"])) for item in doc.get("bin", [])}
        if path and str(path) not in recorded:
            name = _WITHOUT_STAMP.sub("", str(output.get("name") or Path(path).stem))
            same = sum(1 for item in doc["bin"] if str(item.get("name") or "").split(" (")[0] == name)
            item = {"path": production.rel(path), "media": output.get("id"), "kind": output.get("kind"), "name": f"{name} ({same + 1})" if same else name, "by": "deepy"}
            if item["kind"] in ("video", "image") and Path(path).is_file():  # the bin shows what it is, as for the user's imports
                item["poster"] = production.rel(media.poster(Path(path), production.folder / "cache" / f"{media.cache_key(Path(path))}.jpg"))
            production.apply([{"op": "bin_add", "item": item}], by="system", record=False)


_WORKING_FILE = re.compile(r"(_post|_audio(\(\d+\))?|_frame\d+\S*)\.\w+$")
_WITHOUT_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}h\d{2}m\d{2}s_seed\d+_")


def media_entries(result: Any) -> list[dict[str, Any]]:
    """Generated media found in a WanGP tool result: dictionaries with a path and a media id, in order, without duplicates."""
    found, seen = [], set()

    def walk(value):
        if isinstance(value, dict):
            path, media_id = value.get("path"), value.get("media_id") or value.get("id")
            if isinstance(path, str) and isinstance(media_id, str) and ":" in media_id and path not in seen:
                seen.add(path)
                found.append(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(result)
    return found


__all__ = ["Director", "INSTRUCTIONS", "build_prompt", "media_entries", "model"]
