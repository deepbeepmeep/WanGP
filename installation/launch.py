"""Launch managed WanGP under the installation lease used by maintenance."""

import hashlib
import os
import shlex
from contextlib import contextmanager
from pathlib import Path

from .environments import interpreter_path
from .execute import ProcessRunner
from .resolution import load_resolution
from .runtime import ensure_pip
from .state import InstallError, InstallationLock, inside
from .transactions import ready_record, resolution_location


def application_command(python, entry, arguments):
    probe = Path("shared/gradio/gradio_startup_patch.py")
    installed = entry.parent / probe
    # Only backport to the unchanged probe shipped in 18.888 through 19.0.
    # Updated releases and forks with a different probe use their own code.
    compatibility = []
    if installed.is_file() and installed.stat().st_size < 4096:
        digest = hashlib.sha256(installed.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest == "cc4ee6f75134213cc886e5b82af8650233f2e9ad3a3d1c925fc5721e87d91753":
            compatibility.append(str(Path(__file__).resolve().parents[1] / probe))
    return [python, "-u", *compatibility, str(entry), *arguments]


def saved_arguments(app):
    path = inside(app, "scripts/args.txt")
    if not path.exists():
        return []
    arguments = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        if os.name == "nt":
            lexer.escape = ""
        arguments.extend(lexer)
    return arguments


def normalized_options(options):
    defaults = {"listen": False, "verbose": False, "authentication": False, "password_mode": "generated", "debug_deepy": False, "debug_directory": "", "telemetry": False, "telemetry_file": ""}
    options = {} if options is None else options
    if not isinstance(options, dict) or any(key not in defaults or type(value) is not type(defaults[key]) for key, value in options.items()):
        raise InstallError("invalid_launch_options", "Invalid Manager launch options")
    result = {**defaults, **options}
    if result["password_mode"] not in ("custom", "generated") or any("\0" in value for value in result.values() if isinstance(value, str)):
        raise InstallError("invalid_launch_options", "Invalid authentication mode or output path")
    for enabled, key in (("debug_deepy", "debug_directory"), ("telemetry", "telemetry_file")):
        if result[enabled] and not Path(result[key]).is_absolute():
            raise InstallError("invalid_launch_options", "Choose an absolute debug output path")
    if result["telemetry"] and Path(result["telemetry_file"]).suffix not in (".txt", ".log"):
        raise InstallError("invalid_launch_options", "Choose a .log or .txt telemetry file")
    return result


def manager_arguments(options):
    options = normalized_options(options)
    arguments = []
    if options["listen"]:
        arguments.append("--listen")
    if options["verbose"] or options["debug_deepy"] or options["telemetry"]:
        arguments.extend(["--verbose", "2"])
    if options["authentication"]:
        arguments.append("--auth")
    if options["debug_deepy"]:
        arguments.extend(["--debug-deepy", options["debug_directory"]])
    return arguments


@contextmanager
def launch_context(runtime, python, options=None, auth_password=None, events=None, github_opened=False):
    options = normalized_options(options)
    env = runtime.environment(python)
    # The project page was opened from Manager: WanGP records it and no longer asks for a star.
    if github_opened:
        env["WANGP_GITHUB_OPENED"] = "1"
    # Gradio checks its own HTTP endpoint through httpx, which honors host proxies.
    # Keep external downloads proxied, but always reach the local server directly.
    bypass = next((value for key, value in env.items() if key.lower() == "no_proxy"), "")
    bypass = ",".join(dict.fromkeys([*filter(None, bypass.split(",")), "localhost", "127.0.0.1", "::1"]))
    for key in list(env):
        if key.lower() == "no_proxy":
            del env[key]
    env.update(NO_PROXY=bypass, no_proxy=bypass)
    # Use the same IPv4 loopback address as Manager's readiness probe and browser.
    env["SERVER_NAME"] = "127.0.0.1"
    if options["authentication"]:
        if options["password_mode"] == "custom":
            if not isinstance(auth_password, str) or not auth_password.strip() or len(auth_password.encode("utf-8")) > 1024 or "\0" in auth_password:
                raise InstallError("password_required", "Set your custom password in Preferences")
            env["WANGP_AUTH_PASSWORD"] = auth_password
        else:
            env.pop("WANGP_AUTH_PASSWORD", None)
    emit = events or (lambda event: None)
    output = None
    try:
        if options["telemetry"]:
            from .state import plain_path
            path = plain_path(options["telemetry_file"])
            path.parent.mkdir(parents=True, exist_ok=True)
            output = path.open("a", encoding="utf-8", buffering=1)
            env["WAN2GP_DEEPY_TELEMETRY"] = "1"
            output.write("\n--- WanGP Manager telemetry session ---\n")
            emit({"type": "log", "text": f"Deepy telemetry enabled. Console output is appended to {path}\n"})
        def forward(event):
            if output is not None and event.get("type") == "log":
                output.write(event["text"])
                output.flush()
            emit(event)
        yield env, forward
    finally:
        if output is not None:
            output.close()


def combined_arguments(app, arguments):
    """Explicit launch choices supersede matching args.txt options."""
    saved = iter(saved_arguments(app))
    flags = {arg.split("=", 1)[0] for arg in arguments}
    replaced = {"--server-port", "--verbose", "--debug-deepy"} & flags
    switches = {"--listen"} & flags
    if "--auth" in flags:
        switches.update({"--auth", "--no-auth"})
        replaced.add("--auth-password")
    result = []
    for argument in saved:
        key, separator, _ = argument.partition("=")
        if key in replaced:
            if not separator:
                next(saved, None)
        elif key not in switches:
            result.append(argument)
    return [*result, *arguments]


def launch_managed(root, runtime, arguments=(), events=None, cancelled=None, on_start=None, launch_options=None, auth_password=None, github_opened=False):
    with InstallationLock(root):
        record = ready_record(root)
        app = inside(root, "app")
        entry = inside(app, "wgp.py")
        if not entry.is_file():
            raise InstallError("missing_entrypoint", "Managed source does not contain wgp.py")
        python = interpreter_path(inside(root, record["active"]["environment"]))
        resolution = load_resolution(resolution_location(root, record["active"]["resolution"]), check_inputs=False)
        with launch_context(runtime, python, launch_options, auth_password, events, github_opened) as (env, output_events):
            arguments = combined_arguments(app, arguments)
            device = resolution["selected_device"]
            if device is not None:
                if any(argument == "--gpu" or argument.startswith("--gpu=") for argument in arguments):
                    raise InstallError("device_conflict", "This installation has a validated device selection; do not override it with --gpu")
                if device["vendor"] == "NVIDIA":
                    env["CUDA_VISIBLE_DEVICES"] = device["id"]
                elif device["vendor"] == "AMD" and "index" in device:
                    env["HIP_VISIBLE_DEVICES"] = str(device["index"])
            runner = ProcessRunner(inside(root, ".manager/server.log"), events=output_events, cancelled=cancelled)
            ensure_pip(python, runtime, runner)
            command = application_command(python, entry, arguments)
            while True:
                result = runner.run(command, cwd=app, env=env, timeout=None, check=False, on_start=on_start)
                if result.returncode != 42:
                    return {"status": "stopped" if result.returncode == 0 else "crashed", "returncode": result.returncode}
