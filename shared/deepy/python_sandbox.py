# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deepy's Python sandbox: a contained, restartable interpreter for the assistant's own code (toolbox wangp_python).

The code runs in a worker process (python_worker.py) started with the interpreter in isolated mode inside the session's sandbox
folder. Containment layers: the separate process (a crash or a hang never reaches WanGP; the host kills it on timeout, Stop,
restart, idle and close); OS limits (Windows: a job object with no child process, a memory limit and kill-on-close, and a low
integrity token so that the OS refuses writes outside the sandbox folder, which is labeled low; Linux: rlimits for memory, processes
and file size, and death with the parent); and the worker's own guards on imports, file access and network. Files written under
out/ after a run are published to WanGP's outputs and Gallery through the toolbox."""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from shared.deepy import python_worker

DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 600
STARTUP_TIMEOUT_S = 60
WORKER_STDERR_BYTES = 4096
MEMORY_MB = 4096
IDLE_S = 600
OUTPUT_CHARS = 5000  # printed output returned to the assistant: summaries, not whole listings
RESULT_CHARS = 2000
MAX_FILES = 200
PUBLISH_CAP = 8  # more media files than this under out/ in one run is a frame sequence, not deliverables
MAX_TOOL_CALLS = 500  # WanGP tool calls one run may make through wangp.call
STDOUT_TAIL_LINES = 20  # printed lines shown live in the transcript entry
STOP_GRACE_S = 10.0  # after Stop, how long a script may take to reach a tool call, print or progress report (where it stops with its variables kept) before it is killed
_COUNT_LINE = re.compile(r"\b(\d+)\s*(?:/|of)\s*(\d+)\b")
MAX_LISTING = 10000
TOOLS_FROM_SCRIPTS = {"wangp_generate", "wangp_postprocess", "wangp_toolbox", "wangp_list_gallery", "wangp_deepy_templates", "wangp_templates", "wangp_search_models", "wangp_models", "wangp_model", "wangp_validate", "wangp_io", "wangp_web_search", "wangp_plugins"}  # no sessions, external servers or Python itself
IO_FROM_SCRIPTS = {"list", "info", "read_text", "search_text", "mkdir", "copy", "move", "zip"}  # file actions a script may loop over: nothing that deletes or overwrites
# The web only through wangp_web_search (when the user enabled it): what Deepy could already do itself, with the same guards (no local network, size limits),
# so a script downloads the address it just extracted instead of the model retyping it. Requests share the session's cookie jar, whose values no script or agent
# can read; clearing it and LoRA installs stay direct calls. The sandbox has no raw network.
WEB_FROM_SCRIPTS = {"search", "news", "images", "videos", "open", "download", "screenshot", "cookies", "lora_search"}
_WORKER = Path(python_worker.__file__).resolve()
_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
_MESH = {".glb", ".gltf", ".obj", ".ply", ".stl"}

HELPERS = {
    "wangp.path(ref)": "local path of a Gallery media id or an authorized @alias/... file, readable afterwards (nothing is copied)",
    "wangp.font(size, family='sans', bold=False, italic=False)": "the way to get a TrueType font here (families sans, serif, mono, condensed, display): font file names and paths vary per machine, so never guess them or fall back to ImageFont.load_default",
    "wangp.svg(markup, width=None, height=None, background=None)": "RGBA PIL image of self-contained SVG markup (resvg)",
    "wangp.text_mesh(text, height=None, width=None, depth=0.1, family='sans', bold=False)": "extruded 3D text as (vertices, faces, normals) arrays, '\\n' for several lines, centred and facing +Z, fitted inside height and/or width: labels and titles of 3D models (never triangulate glyph outlines by hand)",
    "wangp.extrude(shape, height=None, width=None, depth=0.1, threshold=128)": "any flat shape as a 3D mesh, same arrays: a PIL image, an image path or Gallery id, or a boolean mask; bright (or opaque) pixels become solid. Logos, icons, silhouettes, stencils (a dark logo on white: invert it first)",
    "wangp.extrude_parts(image, colors=2, height=None, width=None, depth=0.1)": "a coloured flat design (logo, flag, icon) as one extruded part per colour: [(vertices, faces, normals, rgb), ...] aligned to fit together; gradients fall to their nearest colour, the background (transparent, or the corners' colour) is left out",
    "wangp.publish_video(frames, fps=24, name='clip', audio=None)": "the way to make a clip: encodes PIL images / arrays of one size, PNG paths or a glob such as 'tmp/frames/*.png' into a Gallery H.264 MP4 (audio=<video or audio id> gives it that soundtrack, e.g. the source video's sound when its frames were redrawn; otherwise silent: add sound with remux_media). Frames never go under out/ (each file there would be published as a deliverable): for clips longer than about 10 s save frames as PNG under tmp/ while rendering and pass the glob. cv2.VideoWriter is not a substitute (no H.264 here)",
    "wangp.out / wangp.root": "'out' and '.': the sandbox folder is the working directory, so address its content with relative paths (never with @documents paths, which belong to wangp_io)",
    "wangp.call(tool, **arguments)": "runs one of WanGP's tools as Deepy would and returns its result dict: wangp_toolbox (action=, arguments=), wangp_generate (settings=), wangp_postprocess, the templates, model search and Gallery listing tools. Shortcuts: wangp.inspect(media, question), wangp.toolbox(action, **arguments), wangp.generate(settings). Relative sandbox paths in arguments are accepted; results are not truncated; time spent inside these tools does not count toward timeout_s, so one run can loop over hundreds of files",
    "wangp.listdir(ref, pattern='*', recursive=False)": "every file and subfolder of an authorized folder in one call (no paging), as dicts: path (@alias path: open() and tool arguments accept it), file (local path for any library, cv2 included), name, type ('file' or 'dir'), kind (image, video, audio, 3d, other, folder), bytes, modified (epoch seconds), date ('YYYY-MM-DD HH:MM')",
    "wangp.download(url, **arguments) / wangp.web(action, **arguments)": "the web from a script, when web search is enabled: download(url) saves a page, image, video or file under @documents/web and returns its path (copy the address from a variable, never retype it); web('search' | 'news' | 'images' | 'videos' | 'open' | 'screenshot' | 'cookies' | 'lora_search', ...) returns that action's result (requests share the session's cookies; screenshot(url=...) returns a PNG path). Raw network modules (urllib, requests, sockets) are not available",
    "wangp.io(action, {arguments} or **arguments)": "wangp_io file actions from a script: list, info, read_text, search_text, mkdir, copy, move, zip (no delete); a loop of moves or copies is one run instead of one tool call per file",
    "wangp.plugins(plugin=None) / wangp.plugin(plugin, action, **arguments)": "the toolboxes of enabled plugins (wangp_plugins): plugins() lists them, plugins('media_flow') lists one plugin's actions with their calls, plugin('media_flow', 'templates', media='video') runs an action and waits for its result (a long one keeps the run going; its time does not count toward timeout_s)",
    "wangp.progress(done, total=None, text='')": "shows the user a progress bar with the time left while a loop runs: call wangp.progress(i, len(items), name) as each item starts (the bar then shows from the first item, with the item running) and wangp.progress(len(items), len(items)) after the loop. print() lines also appear live, so a short line per item ('12/80 beach.mp4: yes') keeps the user informed; no progress files are needed",
}
RULES = [
    "Save finished deliverables under out/ (PNG/JPG/WEBP images, WAV/FLAC/MP3 audio, OBJ/PLY/STL/GLTF/GLB meshes; clips through wangp.publish_video): after the run each one is published to the Gallery and returned in outputs with a media_id (at most 8 per run; more means frames, which are not published). A mesh is converted to GLB, given a rendered preview and placed in the 3D gallery, where render_3d_views and inspect_media work on it; write OBJ from numpy (v x y z and f a b c lines, 1-based) for procedural props, terrains and extrusions. Other files under out/ (SRT, JSON, CSV, SVG, TXT) are returned in files with their @documents path for other tools. Everything else under the sandbox (tmp/, work files, frame sequences, comparison or debug images) is scratch and stays private: published files cannot be deleted afterwards.",
    "Write files only with relative paths inside the sandbox ('scratch/face.png', 'out/poster.png', 'tmp/frames'): other folders (/tmp, C:\\...) are refused, and C libraries such as OpenCV fail silently there (cv2.imwrite returns False), so check their result. Other tools (inspect_media, the toolbox) reach a sandbox file as @documents/sandbox/<relative path>; wangp.inspect and wangp.toolbox accept the relative path directly.",
    "The sandbox folder is the working directory: use relative paths ('out/poster.png', 'tmp/frames'). Read Gallery media or @documents files through wangp.path(ref) (images: PIL.Image.open(wangp.path('visual:IMAGE'))). Other paths are refused, and nothing outside the sandbox can be written.",
    "No GPU, raw network, processes or ffmpeg: compute, draw and compose here; generate media with WanGP's models; cut, mix, merge and inspect with the toolbox; reach the web through wangp.download / wangp.web when web search is enabled.",
    "Keep one run to one step (a trailing expression is returned as result, print() as stdout); state persists between runs, so build up gradually; restart after a broken state.",
    f"Loops over many files belong in Python with wangp.call: list a folder with wangp.listdir, call wangp.inspect / wangp.toolbox / wangp.generate per item, report each item with wangp.progress(i, len(items), name) as it starts, collect the results in a list and print only the summary or the matches you need (each call is a real tool call with its full cost, at most {MAX_TOOL_CALLS} per run; time inside the tools does not count toward timeout_s, so no checkpoint files or split runs are needed for long loops; the run result reports the calls made; if the user presses Stop the script stops at its next tool call, print or wangp.progress and keeps its variables, so keep results in a list or dict as you go and read them in the next run). Tools that write files, browse the web, manage sessions or run Python are not callable from scripts.",
]
ACTIONS = {
    "run": {
        "description": "Run Python in the persistent sandbox interpreter: a trailing expression's value comes back as result, print() output as stdout, files saved under out/ are published to the Gallery.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "minLength": 1, "description": "Python source. Start with a `# title` comment naming what the code makes: it becomes the description of published media."},
            "timeout_s": {"type": "number", "minimum": 1, "default": DEFAULT_TIMEOUT_S, "description": f"Seconds of the script's own computing before it is killed (larger values are capped at {MAX_TIMEOUT_S}; time inside WanGP's tools does not count). Does not extend the separate {STARTUP_TIMEOUT_S}-second worker startup limit."},
        }, "required": ["code"], "additionalProperties": False},
        "example": {"action": "run", "arguments": {"code": "# Poster title\nfrom PIL import Image, ImageDraw\nimg = Image.open(wangp.path('visual:IMAGE')).convert('RGB')\nd = ImageDraw.Draw(img)\nd.text((60, 40), 'BLUE HOUR', font=wangp.font(96, 'display'), fill='white')\nimg.save('out/poster.png')\nimg.size"}},
    },
    "restart": {"description": "Restart the interpreter: in-memory variables are discarded, files in the sandbox are kept.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    "files": {
        "description": "List files in the sandbox folder (newest first) with sizes.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "default": "**/*", "description": "Glob relative to the sandbox folder, e.g. out/*.png."}, "limit": {"type": "integer", "minimum": 1, "maximum": MAX_FILES, "default": 50}}, "required": [], "additionalProperties": False},
    },
}


def _media_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return "image" if suffix in _IMAGE else "video" if suffix in _VIDEO else "audio" if suffix in _AUDIO else "3d" if suffix in _MESH else ""


def _font_config() -> dict[str, Any]:
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs = [os.path.join(windir, "Fonts"), os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")]
        fonts = {
            "sans": {"regular": ["segoeui.ttf", "arial.ttf"], "bold": ["segoeuib.ttf", "arialbd.ttf"], "italic": ["segoeuii.ttf", "ariali.ttf"], "bold_italic": ["segoeuiz.ttf", "arialbi.ttf"]},
            "serif": {"regular": ["georgia.ttf", "times.ttf"], "bold": ["georgiab.ttf", "timesbd.ttf"], "italic": ["georgiai.ttf", "timesi.ttf"], "bold_italic": ["georgiaz.ttf", "timesbi.ttf"]},
            "mono": {"regular": ["consola.ttf", "cour.ttf"], "bold": ["consolab.ttf", "courbd.ttf"], "italic": ["consolai.ttf", "couri.ttf"], "bold_italic": ["consolaz.ttf", "courbi.ttf"]},
            "condensed": {"regular": ["ARIALN.TTF", "bahnschrift.ttf"], "bold": ["ARIALNB.TTF", "bahnschrift.ttf"], "italic": ["ARIALNI.TTF"], "bold_italic": ["ARIALNBI.TTF"]},
            "display": {"regular": ["impact.ttf", "bahnschrift.ttf", "arialbd.ttf"], "bold": ["impact.ttf", "arialbd.ttf"], "italic": ["impact.ttf"], "bold_italic": ["impact.ttf"]},
        }
    else:
        dirs = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts")]
        fonts = {
            "sans": {"regular": ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"], "bold": ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"], "italic": ["DejaVuSans-Oblique.ttf", "LiberationSans-Italic.ttf"], "bold_italic": ["DejaVuSans-BoldOblique.ttf", "LiberationSans-BoldItalic.ttf"]},
            "serif": {"regular": ["DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"], "bold": ["DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf"], "italic": ["DejaVuSerif-Italic.ttf", "LiberationSerif-Italic.ttf"], "bold_italic": ["DejaVuSerif-BoldItalic.ttf", "LiberationSerif-BoldItalic.ttf"]},
            "mono": {"regular": ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"], "bold": ["DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf"], "italic": ["DejaVuSansMono-Oblique.ttf"], "bold_italic": ["DejaVuSansMono-BoldOblique.ttf"]},
            "condensed": {"regular": ["DejaVuSansCondensed.ttf", "LiberationSansNarrow-Regular.ttf"], "bold": ["DejaVuSansCondensed-Bold.ttf", "LiberationSansNarrow-Bold.ttf"], "italic": ["DejaVuSansCondensed-Oblique.ttf"], "bold_italic": ["DejaVuSansCondensed-BoldOblique.ttf"]},
            "display": {"regular": ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"], "bold": ["DejaVuSans-Bold.ttf"], "italic": ["DejaVuSans-BoldOblique.ttf"], "bold_italic": ["DejaVuSans-BoldOblique.ttf"]},
        }
    return {"font_dirs": [folder for folder in dirs if folder and os.path.isdir(folder)], "fonts": fonts}


# ------------------------------------------------------------------------------------------------------ launchers
class _Worker:
    """A running worker process with its pipes; killed as a whole on close."""

    def __init__(self, root: Path, memory_bytes: int, low_integrity: bool, alias: str = ""):
        config = {**_font_config(), "max_output": OUTPUT_CHARS, "max_result": RESULT_CHARS, "alias": alias}
        command = [sys.executable, "-I", "-B", "-u", "-X", "utf8", str(_WORKER), str(root), json.dumps(config)]
        tmp = root / "tmp"
        tmp.mkdir(exist_ok=True)
        (tmp / "mpl").mkdir(exist_ok=True)
        env = {"TEMP": str(tmp), "TMP": str(tmp), "TMPDIR": str(tmp), "HOME": str(tmp), "USERPROFILE": str(tmp), "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp / "mpl"), "PATH": os.path.dirname(sys.executable)}
        for name in ("SYSTEMROOT", "SystemDrive", "WINDIR", "LANG", "LC_ALL"):
            if name in os.environ:
                env[name] = os.environ[name]
        self._job = None
        if sys.platform == "win32":
            self._launch_windows(command, root, env, memory_bytes, low_integrity)
        else:
            self._launch_posix(command, root, env, memory_bytes)
        self.lines: queue.Queue = queue.Queue()
        self.stderr_tail = b""
        self._reader = threading.Thread(target=self._read, name="deepy-python-sandbox", daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, name="deepy-python-stderr", daemon=True)
        self._reader.start()
        self._stderr_reader.start()

    def _launch_windows(self, command, root, env, memory_bytes, low_integrity):
        import msvcrt
        import win32api
        import win32con
        import win32job
        import win32pipe
        import win32process
        import win32security

        def pipe():
            attributes = win32security.SECURITY_ATTRIBUTES()
            attributes.bInheritHandle = True
            return win32pipe.CreatePipe(attributes, 0)

        if os.path.normcase(sys.executable) != os.path.normcase(sys._base_executable):
            # a venv's python.exe is a redirector that starts the base interpreter as its child, which the one-process job forbids:
            # start that interpreter directly, made the venv's as the redirector does
            command = [sys._base_executable, *command[1:]]
            env["__PYVENV_LAUNCHER__"] = sys.executable
        stdin_read, stdin_write = pipe()
        stdout_read, stdout_write = pipe()
        stderr_read, stderr_write = pipe()
        for parent_end in (stdin_write, stdout_read, stderr_read):
            win32api.SetHandleInformation(parent_end, win32con.HANDLE_FLAG_INHERIT, 0)
        startup = win32process.STARTUPINFO()
        startup.dwFlags = win32con.STARTF_USESTDHANDLES
        startup.hStdInput, startup.hStdOutput, startup.hStdError = stdin_read, stdout_write, stderr_write
        job = win32job.CreateJobObject(None, "")
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] = (win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
                                                         | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY | win32job.JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
        limits["BasicLimitInformation"]["ActiveProcessLimit"] = 1
        limits["ProcessMemoryLimit"] = limits["JobMemoryLimit"] = memory_bytes
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
        creation = win32con.CREATE_SUSPENDED | win32con.CREATE_NO_WINDOW
        if low_integrity:
            process, thread, self.pid, _tid = win32process.CreateProcessAsUser(_low_integrity_token(), None, subprocess.list2cmdline(command), None, None, True, creation, env, str(root), startup)
        else:
            process, thread, self.pid, _tid = win32process.CreateProcess(None, subprocess.list2cmdline(command), None, None, True, creation, env, str(root), startup)
        win32job.AssignProcessToJobObject(job, process)
        win32process.ResumeThread(thread)
        thread.Close()
        stdin_read.Close()
        stdout_write.Close()
        stderr_write.Close()
        self._job, self._process = job, process
        self.stdout = os.fdopen(msvcrt.open_osfhandle(stdout_read.Detach(), os.O_RDONLY), "rb", buffering=0)
        self.stdin = os.fdopen(msvcrt.open_osfhandle(stdin_write.Detach(), os.O_WRONLY), "wb", buffering=0)
        self.stderr = os.fdopen(msvcrt.open_osfhandle(stderr_read.Detach(), os.O_RDONLY), "rb", buffering=0)

    def _launch_posix(self, command, root, env, memory_bytes):
        import resource

        # RLIMIT_NPROC also forbids threads on Linux: numerical libraries must stay single-threaded, including during import.
        for name in ("OPENBLAS_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"

        def limits():
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            resource.setrlimit(resource.RLIMIT_FSIZE, (2 << 30, 2 << 30))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            try:
                import ctypes
                ctypes.CDLL(None).prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL: die with the host
            except Exception:
                pass

        self._popen = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(root), env=env, preexec_fn=limits, start_new_session=True, bufsize=0)
        self.pid, self.stdin, self.stdout, self.stderr = self._popen.pid, self._popen.stdin, self._popen.stdout, self._popen.stderr

    def _read(self):
        try:
            while line := self.stdout.readline():
                self.lines.put(line)
        except (OSError, ValueError):
            pass
        self.lines.put(None)

    def _read_stderr(self):
        try:
            while chunk := self.stderr.read(WORKER_STDERR_BYTES):
                self.stderr_tail = (self.stderr_tail + chunk)[-WORKER_STDERR_BYTES:]
        except (OSError, ValueError):
            pass

    def alive(self) -> bool:
        if sys.platform == "win32":
            import win32event
            return win32event.WaitForSingleObject(self._process, 0) == win32event.WAIT_TIMEOUT
        return self._popen.poll() is None

    def kill(self):
        try:
            if sys.platform == "win32":
                import win32event
                import win32job
                win32job.TerminateJobObject(self._job, 1)
                win32event.WaitForSingleObject(self._process, 1000)
            else:
                self._popen.kill()
                self._popen.wait(timeout=1)
        except Exception:
            pass
        for reader in (self._reader, self._stderr_reader):
            reader.join(timeout=1)
        for stream in (self.stdin, self.stdout, self.stderr):
            try:
                stream.close()
            except Exception:
                pass


def _low_integrity_token():
    import win32api
    import win32con
    import win32security

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_ALL_ACCESS)
    low = win32security.DuplicateTokenEx(token, win32security.SecurityImpersonation, win32con.TOKEN_ALL_ACCESS, win32security.TokenPrimary)
    win32security.SetTokenInformation(low, win32security.TokenIntegrityLevel, (win32security.ConvertStringSidToSid("S-1-16-4096"), 0x20))  # SE_GROUP_INTEGRITY
    return low


def _label_low_integrity(path: Path) -> bool:
    """Mandatory label 'low, no write up' on the sandbox folder, inherited by its content: the low integrity worker may write there and nowhere else.

    Setting a label needs Write-Owner on the folder; drives whose inherited ACL only grants Modify refuse it, so the owner (this process
    created the folder) first grants itself full control. False, with a console warning, when the label cannot be set: the worker then
    runs with a normal token and only the interpreter's own guards bound its writes."""
    import ntsecuritycon
    import pywintypes
    import win32api
    import win32con
    import win32security
    from tqdm import tqdm

    sacl = win32security.ACL()
    sacl.AddMandatoryAce(win32security.ACL_REVISION_DS, win32con.OBJECT_INHERIT_ACE | win32con.CONTAINER_INHERIT_ACE, 0x1, win32security.ConvertStringSidToSid("S-1-16-4096"))  # SYSTEM_MANDATORY_LABEL_NO_WRITE_UP
    for attempt in range(2):
        try:
            win32security.SetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT, win32security.LABEL_SECURITY_INFORMATION, None, None, None, sacl)
            return True
        except pywintypes.error as error:
            if attempt:
                tqdm.write(f"[Deepy] Python sandbox: the folder {path} could not be labeled low integrity ({error.strerror}); the sandbox runs without the OS write barrier.")
                return False
            try:  # grant the owner full control (Write-Owner included), keeping the inherited entries
                user = win32security.GetTokenInformation(win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY), win32security.TokenUser)[0]
                dacl = win32security.GetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION).GetSecurityDescriptorDacl() or win32security.ACL()
                dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, win32con.OBJECT_INHERIT_ACE | win32con.CONTAINER_INHERIT_ACE, ntsecuritycon.FILE_ALL_ACCESS, user)
                win32security.SetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION, None, None, dacl, None)
            except pywintypes.error as dacl_error:
                tqdm.write(f"[Deepy] Python sandbox: the folder {path} could not be labeled low integrity ({error.strerror}; granting full control failed: {dacl_error.strerror}); the sandbox runs without the OS write barrier.")
                return False
    return False


# ------------------------------------------------------------------------------------------------------ sandbox
class PythonSandbox:
    """One sandbox per session: the folder, the worker (started on demand) and the publication of its outputs.

    resolve_path(ref) returns the local path of a Gallery media id or authorized file (raises for anything else); publish_media(path,
    description) and publish_frames(frame_paths, fps, name, description) return compact Gallery records; interrupted() is polled while
    code runs (the chat Stop button)."""

    def __init__(self, root: Path, *, resolve_path: Callable[[str], str], publish_media: Callable[[str, str], dict | None], publish_frames: Callable[[list[str], float, str, str], dict | None], virtualize: Callable[[str], str] = str, memory_mb: int = MEMORY_MB,
                 call_tool: Callable[[str, dict], dict] | None = None, list_dir: Callable[[str], Path] | None = None, apps=None, resolve_part: Callable[[str], dict] | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "out").mkdir(exist_ok=True)
        (self.root / "scratch").mkdir(exist_ok=True)  # the work folder the rules name in their examples
        self._low_integrity = sys.platform == "win32" and _label_low_integrity(self.root)
        self._resolve_path, self._publish_media, self._publish_frames, self._virtualize = resolve_path, publish_media, publish_frames, virtualize
        self._resolve_part = resolve_part or (lambda reference: {"path": resolve_path(reference), "start_frame": 0, "end_frame": None})  # what wangp.video_frames reads, a view being a part of its source
        self.call_tool, self._list_dir = call_tool, list_dir  # call_tool(name, arguments) runs a WanGP tool for a script (Prime installs its own); list_dir(ref) resolves an authorized folder
        self.apps = apps  # shared.apps.dev.AppWorkshop when App Development is on: the app_* actions and wangp.app helpers
        self.progress: Callable[[str, dict], None] = lambda status_text, summary: None  # live status of a running script: a short status line and a summary dict for the transcript entry
        self._tool_calls: dict[str, int] = {}
        self._tool_errors = 0
        self._live: dict = {}
        self._reported = 0.0
        self.interrupted: Callable[[], bool] = lambda: False
        self._memory_bytes = int(memory_mb) << 20
        self._worker: _Worker | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._facts: dict = {}
        self._lost_state = ""  # why the next run starts a fresh interpreter: "timeout", "crash", "idle", "restart"
        self._idle_timer: threading.Timer | None = None
        self._ever_started = False

    # ---- lifecycle
    def _ensure_worker(self) -> bool:
        """Start the worker when needed; True when a previous interpreter's state was lost since the last run."""
        if self._worker is not None and self._worker.alive():
            return False
        if self._worker is not None:
            self._worker.kill()
            self._lost_state = self._lost_state or "crash"
        lost = self._ever_started and bool(self._lost_state)
        worker = self._worker = _Worker(self.root, self._memory_bytes, self._low_integrity, self._virtualize(str(self.root)).replace("\\", "/"))
        self._ever_started = True
        response = self._exchange({"op": "ping"}, timeout_s=STARTUP_TIMEOUT_S, interruptible=False)  # Stop interrupts runs, not the start-up
        if response.get("status") != "done":
            self._kill("startup")
            error = response.get("error", "no answer")
            if response.get("status") == "timeout":
                error = f"Worker initialization exceeded {STARTUP_TIMEOUT_S:g} s and was stopped before your code ran. timeout_s only controls script execution; increasing it does not extend startup. Check the Python environment and worker diagnostics."
            if diagnostics := worker.stderr_tail.decode("utf-8", errors="replace").strip():
                error += f"\nWorker stderr (last {WORKER_STDERR_BYTES} bytes):\n{diagnostics}"
            raise RuntimeError(f"The Python sandbox did not start: {error}")
        self._facts = {"python": response.get("python"), "libraries": response.get("libraries", {})}
        return lost

    def _stop_flag(self) -> Path:
        return self.root / "tmp" / ".stop"

    def _request_stop(self) -> None:
        self._stop_flag().parent.mkdir(exist_ok=True)
        self._stop_flag().write_text("stop", encoding="utf-8")

    def _kill(self, reason: str) -> None:
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
            self._lost_state = reason
        self._cancel_idle()

    def _cancel_idle(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _arm_idle(self):
        self._cancel_idle()
        self._idle_timer = threading.Timer(IDLE_S, self._idle_kill)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_kill(self):
        with self._lock:
            if self._worker is not None:
                self._kill("idle")

    def close(self) -> None:
        with self._lock:
            self._kill("closed")

    # ---- protocol
    def _exchange(self, payload: dict, timeout_s: float, interruptible: bool = True) -> dict:
        worker = self._worker
        self._next_id += 1
        payload = {**payload, "id": self._next_id}
        try:
            worker.stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
            worker.stdin.flush()
        except (OSError, ValueError):
            self._kill("crash")
            return {"status": "error", "error": "The Python sandbox process is not running; it restarts on the next run."}
        deadline = time.monotonic() + timeout_s
        stop_at = None
        while True:
            if interruptible and stop_at is None and self.interrupted():  # checked on every message: a script that prints or calls tools is never silent
                stop_at = time.monotonic()
                self._request_stop()  # the script stops at its next tool call, print or progress report, keeping its variables
            if stop_at is not None and time.monotonic() - stop_at >= STOP_GRACE_S:
                self._kill("interrupted")
                return {"status": "interrupted", "error": f"The user pressed Stop and the script reached no tool call, print or progress report within {STOP_GRACE_S:g} s, so it was killed (this is not an error of the script or of the tools). Its variables are lost; files it wrote are kept. Loops that print a line or call wangp.progress per item keep their results when stopped."}
            try:
                line = worker.lines.get(timeout=0.25)
            except queue.Empty:
                if payload.get("op") == "run" and time.monotonic() - self._reported >= 1.0:
                    self._report()  # keeps the elapsed time moving while the script is silent
                if time.monotonic() >= deadline:
                    self._kill("timeout")
                    return {"status": "timeout", "error": f"The run exceeded {timeout_s:g} s and was killed; the interpreter state is lost and restarts on the next run. Pass a larger timeout_s (at most {MAX_TIMEOUT_S}) or do less per run."}
                continue
            if line is None:
                self._kill("crash")
                return {"status": "error", "error": f"The Python sandbox process ended unexpectedly (memory limit {self._memory_bytes >> 20} MB, or a crash in a library); it restarts on the next run."}
            try:
                message = json.loads(line)
            except ValueError:
                continue  # a stray write to the real stdout
            if "notify" in message:
                self._on_notify(message["notify"])
                continue
            if "rpc" in message:
                started = time.monotonic()
                reply = {"stop": True} if stop_at is not None else self._serve_rpc(message["rpc"])  # after Stop no new tool runs; a call already running delivers its result
                deadline += time.monotonic() - started  # time spent in WanGP (a generation, an inspection) is not the script's time
                worker.stdin.write(json.dumps(reply, ensure_ascii=False, default=str).encode("utf-8") + b"\n")
                worker.stdin.flush()
                continue
            if message.get("id") == payload["id"]:
                return message

    def _serve_rpc(self, request: dict) -> dict:
        try:
            if request.get("op") == "realpath":  # an @alias path used directly in open(), os or pathlib
                reference = str(request.get("ref", ""))
                try:
                    return {"value": {"path": self._resolve_path(reference), "dir": False}}
                except FileNotFoundError:
                    if self._list_dir is None:
                        raise
                    try:
                        return {"value": {"path": str(self._list_dir(reference)), "dir": True}}
                    except Exception:
                        raise FileNotFoundError(f"No such file or folder: {reference}") from None
            if request.get("op") == "path":
                reference = str(request.get("ref", ""))
                try:
                    return {"value": self._resolve_path(reference)}
                except FileNotFoundError:
                    if self._list_dir is not None:
                        try:
                            self._list_dir(reference)
                        except Exception:
                            raise FileNotFoundError(f"No such file: {reference}") from None
                        raise IsADirectoryError(f"{reference} is a folder: wangp.path is for files. List it with wangp.listdir({reference!r}); the sandbox's own files are relative paths here (scratch/a.png) and @documents/sandbox/scratch/a.png for other tools.") from None
                    raise
            if request.get("op") == "part":
                return {"value": self._resolve_part(str(request.get("ref", "")))}
            if request.get("op") == "video":
                frames = [str(Path(frame).resolve()) for frame in request.get("frames", [])]
                outside = [frame for frame in frames if not Path(frame).is_relative_to(self.root)]
                if outside:
                    raise PermissionError(f"publish_video frames must be inside the sandbox folder: {outside[0]}")
                missing = [frame for frame in frames if not Path(frame).is_file()]
                if missing:
                    raise FileNotFoundError(f"publish_video frame not found: {missing[0]}")
                item = self._publish_frames(frames, float(request.get("fps", 24)), str(request.get("name", "clip")), self._description)
                if item is None:
                    raise RuntimeError("WanGP could not publish the clip.")
                self._published.append(item)
                return {"value": item}
            if request.get("op") == "tool":
                return {"value": self._tool_call(str(request.get("tool", "")), request.get("arguments") or {})}
            if request.get("op") == "list":
                return {"value": self._listing(str(request.get("ref", "")), str(request.get("pattern") or "*"), bool(request.get("recursive")))}
            if request.get("op") == "app":
                if self.apps is None:
                    raise PermissionError("Web apps are disabled: the user can enable them in Configuration > Deepy > Python > App Development (Advanced).")
                return {"value": self.apps.script_call(str(request.get("action", "")), dict(request.get("arguments") or {}))}
            return {"error": f"Unknown sandbox request {request.get('op')!r}."}
        except Exception as error:
            return {"error": str(error), "error_type": type(error).__name__}

    # ---- live progress of a running script
    def _on_notify(self, notice: dict) -> None:
        if notice.get("op") == "progress":
            self._live["progress"] = {"done": int(notice.get("done") or 0), "total": notice.get("total"), "text": str(notice.get("text") or "")}
            self._live["explicit_progress"] = True
            self._report(force=notice.get("total") is not None and int(notice.get("done") or 0) >= int(notice["total"]))
        elif notice.get("op") == "stdout":
            lines = [str(line) for line in notice.get("lines", [])][-STDOUT_TAIL_LINES:]
            self._live["stdout_tail"] = lines
            if not self._live.get("explicit_progress"):  # a script that prints "120 / 669" or "120 of 669" gets a bar too
                for line in reversed(lines):
                    match = _COUNT_LINE.search(line)
                    if match and 0 < int(match.group(2)) and int(match.group(1)) <= int(match.group(2)):
                        self._live["progress"] = {"done": int(match.group(1)), "total": int(match.group(2)), "text": ""}
                        break
            self._report()

    def inner_status(self, status_text: str) -> None:
        """Progress reported by a tool a script called (Inspecting, Running, Stopping generation): shown as the current step of the script."""
        if status_text and status_text not in ("Done", "Error"):
            self._live["step"] = str(status_text)
            self._report()

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        return f"{seconds} s" if seconds < 60 else f"{round(seconds / 60)} min" if seconds < 3600 else f"{seconds // 3600} h {seconds % 3600 // 60:02d}"

    def live_summary(self) -> tuple[str, dict]:
        """(status line, summary) of the running script for its transcript entry."""
        elapsed = time.monotonic() - self._live.get("started", time.monotonic())
        summary: dict = {"status": "running", "live": True, "elapsed_s": round(elapsed, 1)}
        parts = []
        progress = self._live.get("progress")
        if progress:
            done, total = progress["done"], progress["total"]
            summary["progress"] = dict(progress)
            if total:
                parts.append(f"{done}/{total}")
                if 0 < done < total:
                    summary["progress"]["eta_s"] = round(elapsed / done * (total - done))
                    parts.append(f"{self._duration(summary['progress']['eta_s'])} left")
            else:
                parts.append(f"{done} done")
        count = sum(self._tool_calls.values())
        if count:
            summary["tool_calls"] = {"count": count, "by_tool": dict(self._tool_calls), "errors": self._tool_errors, "seconds_each": round(self._live.get("tool_seconds", 0.0) / count, 1)}
            if not progress:
                parts.append(f"{count} tool call{'s' if count > 1 else ''}")
        if self._tool_errors:
            parts.append(f"{self._tool_errors} error{'s' if self._tool_errors > 1 else ''}")
        for key in ("current", "step", "last", "stdout_tail"):
            if self._live.get(key):
                summary[key] = self._live[key]
        if progress and progress.get("text"):
            parts.append(progress["text"])
        elif self._live.get("current"):
            parts.append(self._live["current"] + (f" ({self._live['step']})" if self._live.get("step") else ""))
        return " · ".join(parts) or f"Running {self._duration(elapsed)}", summary

    def _report(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._reported < 0.34:  # at most three updates per second
            return
        self._reported = time.monotonic()
        self.progress(*self.live_summary())

    @staticmethod
    def _item_of(tool: str, arguments: dict) -> str:
        values = arguments.get("arguments") if isinstance(arguments.get("arguments"), dict) else arguments
        for key in ("media_id", "media_ids", "path", "video_id", "audio_id"):
            value = values.get(key) if isinstance(values, dict) else None
            if isinstance(value, list) and value:
                value = f"{Path(str(value[0])).name} (+{len(value) - 1})" if len(value) > 1 else value[0]
            if isinstance(value, str) and value:
                return Path(value).name[:80]
        settings = arguments.get("settings")
        if isinstance(settings, dict) and settings.get("prompt"):
            return str(settings["prompt"]).splitlines()[0][:60]
        return ""

    @staticmethod
    def _outcome(result: dict) -> str:
        for key in ("answer", "error", "message"):
            if result.get(key):
                return " ".join(str(result[key]).split())[:160]
        for key in ("output_file", "path", "media_id"):
            if result.get(key):
                return Path(str(result[key])).name
        return str(result.get("status", ""))

    def _tool_call(self, tool: str, arguments: dict) -> dict:
        if self.call_tool is None:
            raise RuntimeError("WanGP's tools are not reachable from this sandbox.")
        if tool not in TOOLS_FROM_SCRIPTS:
            raise PermissionError(f"{tool or 'this tool'} cannot be called from a script. Callable: {', '.join(sorted(TOOLS_FROM_SCRIPTS))}; sessions, external servers and Python itself are not.")
        if tool == "wangp_io" and arguments.get("action") not in IO_FROM_SCRIPTS:
            writing = arguments.get("action") in ("write_text", "append_text", "edit")
            raise PermissionError(f"wangp_io {arguments.get('action')!r} cannot be called from a script: scripts may {', '.join(sorted(IO_FROM_SCRIPTS))}. "
                                  + ("A script writes the sandbox's own files (apps/..., out/...) with open(path, 'w'); other files take a direct wangp_io call." if writing else "Deleting or overwriting files is a direct wangp_io call."))
        if tool == "wangp_web_search" and (arguments.get("action") not in WEB_FROM_SCRIPTS or arguments.get("action") == "cookies" and (arguments.get("arguments") or {}).get("clear")):
            raise PermissionError(f"wangp_web_search {arguments.get('action')!r} cannot be called from a script: scripts may {', '.join(sorted(WEB_FROM_SCRIPTS))} (cookies without clear). Clearing cookies and installing LoRAs are direct wangp_web_search calls.")
        if tool == "wangp_plugins":
            from shared.mcp.plugin_toolbox import script_problem

            if problem := script_problem(arguments):
                raise PermissionError(problem)
        if sum(self._tool_calls.values()) >= MAX_TOOL_CALLS:
            raise RuntimeError(f"This run already made {MAX_TOOL_CALLS} tool calls, the limit per run: continue in another run.")
        arguments = self._localize(dict(arguments))
        self._tool_calls[tool] = self._tool_calls.get(tool, 0) + 1
        label = arguments.get("action") if tool == "wangp_toolbox" and isinstance(arguments.get("action"), str) else f"{arguments.get('plugin')}.{arguments.get('action')}" if tool == "wangp_plugins" and arguments.get("action") else tool.removeprefix("wangp_")
        item = self._item_of(tool, arguments)
        self._live.update(current=f"{label}: {item}" if item else str(label), step="")
        self._report(force=sum(self._tool_calls.values()) == 1)
        started = time.monotonic()
        try:
            result = self.call_tool(tool, arguments)
        finally:
            self._live["tool_seconds"] = self._live.get("tool_seconds", 0.0) + time.monotonic() - started
        if not isinstance(result, dict):
            result = {"status": "done", "content": result}
        if result.get("status") == "error":
            self._tool_errors += 1
        self._live.update(last={"tool": label, "item": item, "status": result.get("status", "done"), "outcome": self._outcome(result)}, step="")
        self._report()
        return result

    def _localize(self, value):
        """Relative sandbox file paths inside tool arguments become the tools' own form of them (@documents/sandbox/...): the tools know the sandbox
        folder as an authorized folder, not as the working directory, and refuse absolute paths."""
        if isinstance(value, dict):
            return {key: self._localize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._localize(item) for item in value]
        if isinstance(value, str) and value and not value.startswith("@") and "://" not in value and not os.path.isabs(value) and len(value) < 260:
            candidate = self.root / value
            if candidate.is_file():
                return self._virtualize(str(candidate.resolve()))
        return value

    def _listing(self, ref: str, pattern: str, recursive: bool) -> list[dict]:
        if self._list_dir is None:
            raise RuntimeError("Folder listings are not available in this sandbox.")
        folder = Path(self._list_dir(ref))
        entries = []
        for path in sorted(folder.rglob(pattern) if recursive else folder.glob(pattern)):
            if path.name.startswith(".") or not (path.is_file() or path.is_dir()):
                continue
            stat = path.stat()
            is_dir = path.is_dir()
            entries.append({"path": self._virtualize(str(path)), "file": str(path), "name": path.name, "type": "dir" if is_dir else "file", "bytes": 0 if is_dir else stat.st_size,
                            "modified": round(stat.st_mtime), "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)), "kind": "folder" if is_dir else _media_kind(str(path)) or "other"})
            if len(entries) >= MAX_LISTING:
                break
        return entries

    # ---- files
    def _snapshot(self) -> dict[str, tuple[int, int]]:
        files = {}
        for folder, dirs, names in os.walk(self.root):
            dirs[:] = [name for name in dirs if name not in ("tmp", "__pycache__")]
            for name in names:
                path = Path(folder) / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                files[str(path)] = (stat.st_size, stat.st_mtime_ns)
        return files

    def _changed(self, before: dict, after: dict) -> list[str]:
        return sorted(path for path, signature in after.items() if before.get(path) != signature)

    def files(self, pattern: str = "**/*", limit: int = 50) -> dict:
        entries = []
        for path in self.root.glob(pattern):
            if path.is_file() and "tmp" not in path.relative_to(self.root).parts[:1]:
                stat = path.stat()
                entries.append((stat.st_mtime, {"path": self._virtualize(str(path)), "bytes": stat.st_size}))
        entries.sort(key=lambda item: -item[0])
        return {"status": "done", "folder": self._virtualize(str(self.root)), "count": len(entries), "files": [entry for _mtime, entry in entries[:limit]], "truncated": len(entries) > limit}

    # ---- actions
    @property
    def actions(self) -> dict:
        if self.apps is None:
            return ACTIONS
        from shared.apps.dev import ACTIONS as APP_ACTIONS
        return {**ACTIONS, **APP_ACTIONS}

    def alias_of(self, path: str) -> str | None:
        """The @alias form of a sandbox-relative path (apps/x/main.js) naming a sandbox file, or a new file in one of its subfolders; else None."""
        if not path or path.startswith(("@", "/", "\\")) or Path(path).is_absolute():
            return None
        local = (self.root / path).resolve()
        if local.is_relative_to(self.root) and local.parent != self.root.parent and (local.exists() and local != self.root or local.parent.is_dir() and local.parent != self.root):
            return self._virtualize(str(local))
        return None

    def describe(self) -> dict:
        if not self._facts:
            try:
                from importlib import metadata
                self._facts = {"python": sys.version.split()[0], "libraries": {name: metadata.version(package) for name, package in (("numpy", "numpy"), ("PIL", "pillow"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("pandas", "pandas"), ("cv2", "opencv-python"), ("skimage", "scikit-image"), ("sympy", "sympy"), ("soundfile", "soundfile"), ("pypdf", "pypdf")) if _installed(package)}}
            except Exception:
                self._facts = {"python": sys.version.split()[0], "libraries": {}}
        isolation = ["separate process killed on timeout, Stop, restart and idle", f"memory limit {self._memory_bytes >> 20} MB, no child processes", "imports limited to " + python_worker.LIBRARY_SUMMARY, "files: read/write inside the sandbox folder, read-only Gallery media and @documents through wangp.path, no network"]
        if self._low_integrity:
            isolation.insert(2, "low integrity process: the OS refuses writes outside the sandbox folder")
        facts = {**self._facts, "folders": {"root": self._virtualize(str(self.root)), "out": self._virtualize(str(self.root / "out"))}, "limits": {"timeout_s_default": DEFAULT_TIMEOUT_S, "timeout_s_max": MAX_TIMEOUT_S, "memory_mb": self._memory_bytes >> 20, "output_chars": OUTPUT_CHARS, "tool_calls_per_run": MAX_TOOL_CALLS}, "isolation": isolation, "helpers": HELPERS, "rules": RULES}
        if self.call_tool is not None:
            facts["tools_from_scripts"] = sorted(TOOLS_FROM_SCRIPTS)
        if self.apps is not None:
            apps = self.apps.describe()
            facts["helpers"] = {**HELPERS, **apps.pop("helpers")}
            facts["web_apps"] = apps
        return facts

    def run_action(self, action: str, arguments: dict) -> dict:
        if action == "run":
            requested = float(arguments.get("timeout_s") or DEFAULT_TIMEOUT_S)
            result = self.run(str(arguments["code"]), requested)
            if requested > MAX_TIMEOUT_S:
                result["info"] = f"timeout_s was capped at {MAX_TIMEOUT_S} s (time inside WanGP's tools does not count toward it)."
            return result
        if action == "restart":
            return self.restart()
        if action == "files":
            return self.files(str(arguments.get("pattern") or "**/*"), int(arguments.get("limit") or 50))
        if self.apps is not None and action in self.actions and action not in ACTIONS:  # the app actions
            return self.apps.run_action(action, arguments)
        raise ValueError(f"Unknown sandbox action: {action}")

    def restart(self) -> dict:
        with self._lock:
            self._kill("restart")
            self._lost_state = ""  # a requested restart is not a lost state worth a warning
            return {"status": "done", "message": "Python sandbox restarted: variables and imports are gone, files in the sandbox are kept."}

    def run(self, code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
        timeout_s = max(1.0, min(float(timeout_s), MAX_TIMEOUT_S))
        with self._lock:
            self._cancel_idle()
            lost_reason = self._lost_state
            restarted = self._ensure_worker()
            self._lost_state = ""
            self._description = _description(code)
            self._published: list[dict] = []
            self._tool_calls, self._tool_errors = {}, 0
            self._live, self._reported = {"started": time.monotonic()}, 0.0
            self._stop_flag().unlink(missing_ok=True)
            before = self._snapshot()
            response = self._exchange({"op": "run", "code": code}, timeout_s)
            result = {key: value for key, value in response.items() if key != "id"}
            if self._tool_calls:
                result["tool_calls"] = {"count": sum(self._tool_calls.values()), "by_tool": dict(self._tool_calls), "errors": self._tool_errors}
            if self._live.get("progress"):
                result["progress"] = {key: value for key, value in self._live["progress"].items() if key != "eta_s"}
            self._stop_flag().unlink(missing_ok=True)
            if result.get("status") in ("interrupted", "timeout") and self._live.get("stdout_tail"):
                result["last_output"] = self._live["stdout_tail"]  # the killed worker's captured output is lost: what the user saw last
            changed = self._changed(before, self._snapshot())
            outputs = list(self._published)
            files = []
            deliverables = [path for path in changed if Path(path).relative_to(self.root).parts[:1] == ("out",) and _media_kind(path)]
            if len(deliverables) > PUBLISH_CAP:
                result["hint"] = f"{len(deliverables)} media files were written under out/ in one run, which looks like a frame sequence: none was published. out/ is for finished deliverables; frames belong under tmp/ and become one clip through wangp.publish_video('tmp/frames/*.png', fps)."
                deliverables = []
            unpublished = []
            previewable = {".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".svg"} | ({".html", ".htm"} if self.apps is not None else set())  # opened in the chat; pages run scripts, so only with web apps on
            previews = [self._virtualize(path) for path in changed if Path(path).relative_to(self.root).parts[:1] == ("out",) and Path(path).suffix.lower() in previewable]
            if previews:
                result["previews"] = previews[:PUBLISH_CAP]
            if self.apps is not None and (apps := self.apps.changed(changed)):
                result["apps"] = apps  # the chat shows each with a Play button; their files are not listed one by one
                changed = [path for path in changed if not Path(path).is_relative_to(self.apps.root)]
            for path in changed:
                if path in deliverables:
                    try:
                        item = self._publish_media(path, self._description)
                    except Exception as error:  # a broken deliverable (an empty mesh, a truncated image) fails alone, with its name
                        item = None
                        unpublished.append(f"{Path(path).relative_to(self.root).as_posix()}: {type(error).__name__}: {error}")
                    if item is not None:
                        outputs.append({**item, "source": Path(path).relative_to(self.root).as_posix()})  # the published copy may carry a numbered name; the sandbox file keeps its own
                        continue
                files.append(self._virtualize(path))
            if outputs:
                result["outputs"] = outputs
                result["output_files"] = [item["path"] for item in outputs if item.get("path")]
                if any(item.get("media_type") == "3d" for item in outputs):
                    result["next"] = "The mesh is a 3D gallery asset: wangp_toolbox render_3d_views on its media_id gives shaded views, depth and normal maps (no need to rasterize it yourself), inspect_media describes it."
            if files:
                result["files"] = files[:MAX_FILES]
            if unpublished:
                result["unpublished"] = unpublished[:8]
                result["hint"] = "These files under out/ could not be published (the file is probably empty or broken, e.g. a mesh without faces): fix the script and save them again."
            if restarted:
                result["restarted"] = f"The interpreter restarted ({lost_reason}): earlier variables and imports are gone, files were kept."
            if self._worker is not None:
                self._arm_idle()
            return result


def _description(code: str) -> str:
    first = next((line.strip() for line in code.splitlines() if line.strip()), "")
    title = first.lstrip("# ").strip() if first.startswith("#") else ""
    if title.lower().startswith("title:"):
        title = title[6:].strip()
    return f"Made with Python in Deepy's sandbox: {title}" if title else "Made with Python in Deepy's sandbox."


def _installed(package: str) -> bool:
    from importlib import metadata
    try:
        metadata.version(package)
        return True
    except metadata.PackageNotFoundError:
        return False
