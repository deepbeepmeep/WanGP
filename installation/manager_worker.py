"""Versioned stdio adapter used only by the native Manager control window."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from .catalog import load_config
from .checkpoints import remove_version
from .conda import discover_conda_environments
from . import conda_transactions
from .environments import inspect_interpreter, installation_environments, interpreter_path, managed_record
from .execute import Cancelled, ProcessRunner
from .hardware import detect_hardware, select_device
from .launch import application_command, launch_managed, manager_arguments, combined_arguments, launch_context
from .manager_demo import create_demo
from .plan import build_plan
from .releases import create_release, load_release, source_path
from .resolution import resolve_dependencies
from .runtime import EnvironmentService, ensure_pip
from .server_probe import require_free_port, wait_for_server
from .sources import OFFICIAL_REPOSITORY, VERSION_FILE, cached_download_release, list_versions, remote_catalog, version_metadata
from .state import InstallError, InstallationLock, atomic_json, inside, plain_path, read_json
from .transactions import TransactionEngine, active_operation, identifier, operation_status, prepare_transaction, read_installation, version_history
from .uv import UvRuntime
from .version import local_version


PROTOCOL = 1
ACTIONS = {"hardware", "conda_environments", "detect_installation", "catalog", "sources", "register", "inspect", "preview", "apply", "recover", "versions", "remove_version", "launch", "demo", "manager_update", "latest_versions"}


def application_root(root):
    root = plain_path(root)
    if root.name == "app" and (root.parent / ".manager/installation.json").is_file():
        return root.parent
    if not (root / "wgp.py").is_file() and not (root / ".manager/installation.json").is_file() and (root / "app/wgp.py").is_file():
        return root / "app"
    return root


def installation_info(root, python=None, data=None):
    root = application_root(root)
    conda = conda_transactions.read_record(root)
    pending_conda = conda_transactions.active_operation(root)
    if conda or pending_conda:
        if conda is None:
            journal = read_json(inside(root, f".manager/conda-operations/{pending_conda}/journal.json"))
            conda = journal["plan"]["previous"]
        active = conda["active"]
        kind = conda["ownership"]
        python = interpreter_path(active["environment"], kind)
        return {"root": str(root), "name": root.name, "ownership": kind, "maintenance": kind, "python": python, "release": active["release_id"], "environment": active["environment"], "inspection": inspect_interpreter(python), "pending_operation": pending_conda}
    if (root / ".manager/installation.json").exists() or (root / ".manager/active_operation.json").exists():
        current = read_installation(root)
        pending = active_operation(root)
        active = current["active"] if current else None
        info = {"root": str(root), "name": root.name, "ownership": "managed", "python": str(interpreter_path(inside(root, active["environment"]))) if active else None, "release": active["release_id"] if active else None, "environment": active["environment"] if active else None, "generation": current["generation"] if current else 0, "pending_operation": pending}
        if data and root.parent.parent == Path(data) / "test-workspaces":
            info.update(name="Test installation", test_workspace=True, demo_releases=[{"version": version, "path": str(root.parent / ("release-" + version))} for version in ("1.0", "2.0")])
        return info
    if not (root / "wgp.py").is_file():
        raise InstallError("missing_application", "Select the WanGP folder containing wgp.py, or the root of a managed installation")
    if managed_record(root):
        raise InstallError("select_parent", "Select the parent installation folder containing .manager and app")
    if python is None:
        python = installation_environments(root)["python"]
        if not python:
            raise InstallError("python_required", "Choose the Python environment used by this WanGP installation. Use Detect environment to find env_uv, venv or the saved selection")
    inspection = inspect_interpreter(plain_path(python))
    if inspection["status"] != "inspected":
        raise InstallError("python_unavailable", inspection["error"])
    demo = (root / "setup_config.json").is_file() and set(load_config(root / "setup_config.json")["gpu_profiles"]) == {"Demo"}
    if not inspection["packages"]["torch"] and not demo:
        raise InstallError("torch_missing", "This Python has no PyTorch. Choose the WanGP environment (usually env_uv or env), not the base Miniconda Python. Use Detect environment to find it")
    version = local_version(root)
    kind = "conda" if inspection["conda_environment"] else "venv" if inspection["prefix"] != inspection["base_prefix"] else None
    return {"root": str(root), "name": root.name, "ownership": "external", "maintenance": kind if (root / ".git").exists() else None, "python": inspection["interpreter"], "release": version or "Existing checkout", "inspection": inspection, "pending_operation": None}


def source_catalog(source):
    source = plain_path(source)
    root = source / "source" if (source / "release.json").exists() else source
    config = load_config(root / "setup_config.json")
    version = local_version(root)
    return {**catalog_summary(config), "bundled": root != source, "release": read_json(source / "release.json")["release_id"] if root != source else version}


def catalog_summary(config):
    return {"profiles": {key: {"python": config["components"]["python"][value["python"]]["ver"], "torch": config["components"]["torch"][value["torch"]]["label"], "label": key.replace("_", " "), "platforms": value.get("platforms", ["win", "linux", "macos"])} for key, value in config["gpu_profiles"].items()}}


def snapshot_checkout(source, destination, release_id, runner, emit):
    source = plain_path(source)
    if not (source / ".git").exists():
        raise InstallError("source_inventory_required", "Choose a Git checkout or a prepared release bundle with release.json")
    result = runner.run(["git", "-C", str(source), "ls-files", "-z", "--cached"], capture=True, timeout=30)
    filenames, excluded = [], 0
    for filename in result.stdout.split("\0"):
        if not filename or filename.startswith("manager/"):
            continue
        try:
            source_path(filename)
        except InstallError:
            excluded += 1
            continue
        if inside(source, filename).is_file():
            filenames.append(filename)
    # The shared installer can be present before its first source commit.
    if (source / "installation").is_dir():
        filenames.extend(path.relative_to(source).as_posix() for path in (source / "installation").glob("*.py"))
    if (source / VERSION_FILE).is_file():
        filenames.append(VERSION_FILE)
    runner.check_cancelled()
    emit({"type": "progress", "title": "Preparing Local Source", "message": f"Snapshotting {len(set(filenames))} source files; {excluded} protected files excluded"})
    create_release(source, sorted(set(filenames)), destination, release_id)
    runner.check_cancelled()
    return destination


def run_action(action, args, data, runtime, cancelled, emit, request_id):
    runner = ProcessRunner(data / "logs" / (request_id + ".log"), cancelled=cancelled.is_set, events=emit)
    engine = TransactionEngine(runtime, events=emit, cancelled=cancelled.is_set)
    conda_engine = conda_transactions.CondaTransactionEngine(runtime, events=emit, cancelled=cancelled.is_set)
    if action == "hardware":
        return detect_hardware()
    if action == "conda_environments":
        return discover_conda_environments(check_cancelled=runner.check_cancelled)
    if action == "detect_installation":
        root = application_root(args["root"])
        if managed_record(root) or conda_transactions.read_record(root):
            info = installation_info(root)
            return {"root": str(root), "python": info["python"], "environments": [{"name": "Saved Manager environment", "interpreter": info["python"], **inspect_interpreter(info["python"])}]}
        if not (root / "wgp.py").is_file():
            raise InstallError("missing_application", "Choose your existing WanGP folder containing wgp.py")
        return {"root": str(root), **installation_environments(root)}
    if action == "manager_update":
        from urllib.request import Request, urlopen
        request = Request("https://api.github.com/repos/deepbeepmeep/WanGP/releases/latest", headers={"Accept": "application/vnd.github+json", "User-Agent": "WanGP-Manager"})
        with urlopen(request, timeout=20) as response:
            release = json.load(response)
        tag = release["tag_name"]
        if not tag.startswith("manager-v"):
            raise InstallError("manager_release_unavailable", "The latest release is not a Manager release. Visit the download page")
        asset = "WanGPManager-x86_64.AppImage" if sys.platform.startswith("linux") else "WanGPManager.exe"
        return {"version": tag.removeprefix("manager-v"), "url": f"https://github.com/deepbeepmeep/WanGP/releases/latest/download/{asset}"}
    if action == "catalog":
        if args.get("repository"):
            config, version = remote_catalog(args["repository"], args["commit"], runner)
            return {**catalog_summary(config), "bundled": True, "release": version, "commit": args["commit"], "repository": args["repository"]}
        return source_catalog(args["source"])
    if action == "latest_versions":  # the newest official builds, for the update notice of the Overview: no progress shown
        return list_versions(OFFICIAL_REPOSITORY, data, runner, lambda event: None)
    if action == "sources":
        return list_versions(args.get("repository", OFFICIAL_REPOSITORY), data, runner, emit, refresh=bool(args.get("refresh", False)), page=args.get("page", 1), catalog_id=args.get("catalog_id"))
    if action in ("register", "inspect"):
        info = installation_info(args["root"], args.get("python") or None, data)
        if action == "inspect" and info["python"]:
            info["inspection"] = inspect_interpreter(info["python"])
            if info["ownership"] == "managed":
                info["operation"] = operation_status(info["root"])
        return info
    if action == "versions":
        if conda_transactions.read_record(args["root"]) or args.get("maintenance") in ("conda", "venv"):
            return conda_transactions.version_history(args["root"])
        return version_history(args["root"])
    if action == "remove_version":
        preferences = data / "preferences.json"
        registrations = read_json(preferences)["installations"] if preferences.exists() else []
        emit({"type": "progress", "title": "Removing Saved Version", "message": "Checking which versions and installations use its environment"})
        return remove_version(args["root"], args["checkpoint"], registrations)
    if action == "demo":
        target = data / "test-workspaces" / request_id[:12]
        emit({"type": "progress", "title": "Preparing Test Installation"})
        return create_demo(target)
    if action == "preview":
        operation = args["operation"]
        if operation not in ("install", "update", "repair", "switch-version", "rollback"):
            raise InstallError("invalid_operation", "Unknown installation action")
        preview = data / "previews" / request_id
        preview.mkdir(parents=True, exist_ok=False)
        existing = installation_info(args["root"], args.get("python"), data) if operation != "install" else None
        is_conda = existing and existing.get("maintenance") in ("conda", "venv")
        if is_conda:
            with InstallationLock(args["root"]):
                if existing["pending_operation"]:
                    raise InstallError("recovery_required", "Recover the interrupted Conda update first")
                conda_transactions.git_state(plain_path(args["root"]), runner)
        if operation == "rollback":
            plan = conda_engine.prepare(args["root"], operation="rollback", checkpoint=args["checkpoint"]) if is_conda else prepare_transaction(args["root"], operation="rollback", checkpoint=args["checkpoint"], emit=emit, check_cancelled=runner.check_cancelled)
        else:
            if args.get("repository"):
                release = cached_download_release(args["repository"], args["commit"], inside(args["root"], ".manager/cache/sources"), runner, emit)
            else:
                source = plain_path(args["source"])
                release = source if (source / "release.json").exists() else snapshot_checkout(source, preview / "release", args.get("release_id") or source.name, runner, emit)
                if release == source:
                    load_release(release, emit=emit, check_cancelled=runner.check_cancelled)
            emit({"type": "progress", "title": "Reading Installation Requirements", "message": "Checking the selected source and hardware profile"})
            hardware = detect_hardware()
            device = select_device(hardware, args["device"]) if args.get("device") else None
            config = load_config(release / "source/setup_config.json")
            if args["profile"] != "Demo" and (not device or device["profile"] != args["profile"]):
                raise InstallError("gpu_profile_mismatch", "Choose a detected local GPU and its recommended installation profile")
            targets = [device["architecture"]] if device and device.get("architecture", "").startswith("gfx") else []
            catalog = build_plan(config, release / "source", plain_path(args["root"]) / "environments" / request_id, args["profile"], recommended_extras=bool(args.get("extras", False)), plugins=args.get("plugins", []), device=device, amd_targets=targets)
            emit({"type": "progress", "title": "Resolving Dependencies", "message": "Checking compatible versions and package hashes. uv may download package metadata or wheels; this can take several minutes."})
            plugins = []
            if existing:
                app_root = Path(existing["root"]) / "app" if existing["ownership"] == "managed" else Path(existing["root"])
                for requirements in sorted(app_root.glob("plugins/*/requirements.txt")):
                    # Built-in plugins must use the target release's requirements.
                    target = release / "source" / requirements.relative_to(app_root)
                    plugins.append(target if target.is_file() else requirements)
            policy = args.get("environment", "auto")
            service = EnvironmentService(runtime, runner, kind="conda" if existing and existing.get("maintenance") == "conda" else "uv")
            installed = None
            if existing and operation in ("update", "switch-version") and policy != "new":
                info = existing.get("inspection") or inspect_interpreter(existing["python"])
                if info.get("python") == catalog["python"]:
                    installed = service.inventory(info["prefix"])["packages"]
            resolve_dependencies(catalog, preview / "resolution", runtime, runner=runner, extra_requirements=plugins, installed=installed)
            runner.check_cancelled()
            emit({"type": "progress", "title": "Preparing Installation Plan", "message": "Comparing files and recording the changes for your review"})
            if is_conda:
                repository = args.get("repository") or args.get("source")
                commit = args.get("commit") or conda_transactions.git(plain_path(repository), runner, "rev-parse", "HEAD")
                plan = conda_engine.prepare(args["root"], python=existing["python"], release=release, resolution=preview / "resolution", repository=repository, commit=commit, operation=operation, environment_policy=policy)
            else:
                plan = prepare_transaction(args["root"], release, preview / "resolution", operation=operation, environment_policy=policy, emit=emit, check_cancelled=runner.check_cancelled, environment_service=service)
        atomic_json(preview / "transaction.json", plan)
        return {"preview_id": request_id, "plan": plan}
    if action == "apply":
        path = inside(data, f"previews/{identifier(args['preview_id'])}/transaction.json")
        plan = read_json(path)
        outcome = conda_engine.apply(plan) if plan.get("backend") == "conda" else engine.apply(plan)
        if outcome["status"] == "succeeded":
            outcome["installation"] = installation_info(plan["installation_root"], data=data)
        return outcome
    if action == "recover":
        is_conda = conda_transactions.read_record(args["root"]) or conda_transactions.active_operation(args["root"])
        outcome = conda_engine.recover(args["root"]) if is_conda else engine.recover(args["root"])
        if outcome.get("error") and outcome["error"]["code"] == "interrupted":
            outcome = {"status": "recovered", "interrupted_operation": outcome}
        if read_installation(args["root"]) or is_conda:
            outcome["installation"] = installation_info(args["root"], data=data)
        return outcome
    if action == "launch":
        info = installation_info(args["root"], args.get("python") or None)
        if info["pending_operation"]:
            raise InstallError("recovery_required", "Recover the interrupted installation operation before launching")
        port = args.get("port", 7860)
        require_free_port(port)
        monitors = []
        def started(pid):
            for monitor, stopped in monitors:
                stopped.set()
                monitor.join(timeout=2)
            monitors.clear()
            emit({"type": "server_starting", "pid": pid, "port": port})
            stopped = threading.Event()
            monitor = threading.Thread(target=wait_for_server, args=(pid, port, stopped, emit), daemon=True)
            monitors.append((monitor, stopped))
            monitor.start()
        arguments = [*manager_arguments(args.get("launch_options")), "--server-port", str(port)]
        try:
            if info["ownership"] == "managed":
                return launch_managed(info["root"], runtime, arguments, emit, cancelled.is_set, started, args.get("launch_options"), args.get("auth_password"))
            # The lease is in Manager data, never inside an externally owned checkout.
            import hashlib
            lease = Path(info["root"]) if info.get("maintenance") in ("conda", "venv") else data / "external-leases" / hashlib.sha256(os.path.normcase(info["root"]).encode()).hexdigest()
            with InstallationLock(lease):
                root = Path(info["root"])
                with launch_context(runtime, info["python"], args.get("launch_options"), args.get("auth_password"), emit) as (env, output_events):
                    runner.events = output_events
                    if info.get("maintenance") == "venv":
                        ensure_pip(info["python"], runtime, runner)
                    while True:
                        command = application_command(info["python"], root / "wgp.py", combined_arguments(root, arguments))
                        result = runner.run(command, cwd=root, env=env, timeout=None, check=False, on_start=started)
                        if result.returncode != 42:
                            break
                    return {"status": "stopped" if result.returncode == 0 else "crashed", "returncode": result.returncode}
        except Cancelled:
            return {"status": "stopped"}
        finally:
            for monitor, stopped in monitors:
                stopped.set()
                monitor.join(timeout=2)
    raise InstallError("unknown_action", "Unsupported Manager command")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--uv", required=True)
    args = parser.parse_args(argv)
    data = plain_path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    sequence = 0
    request_id = None
    def emit(value):
        nonlocal sequence
        with lock:
            sequence += 1
            print(json.dumps({**value, "protocol": PROTOCOL, "request_id": request_id, "sequence": sequence}), flush=True)
    emit({"type": "hello", "worker_version": "0.1.0", "actions": sorted(ACTIONS)})
    cancelled = threading.Event()
    try:
        request = json.loads(sys.stdin.readline(1024 * 1024))
        if request.get("protocol") != PROTOCOL or request.get("action") not in ACTIONS:
            raise InstallError("protocol_mismatch", "Manager and installation worker do not support the same protocol")
        request_id = identifier(request["id"])
        def controls():
            for line in sys.stdin:
                try:
                    control = json.loads(line)
                    if control == {"cancel": True}:
                        cancelled.set()
                except ValueError:
                    pass
            cancelled.set()
        threading.Thread(target=controls, daemon=True).start()
        # The Linux AppImage mounts the bundled runtime at a new path on each start: environments built on it would break,
        # so they use uv's managed Python in the data folder, as other Python versions already do.
        runtime = UvRuntime(args.uv, data / "runtime", base_python=sys.executable if os.name == "nt" else None)
        result = run_action(request["action"], request.get("args", {}), data, runtime, cancelled, emit, request_id)
        emit({"type": "finished", "result": result})
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        emit({"type": "finished", "result": {"status": "cancelled" if isinstance(exc, Cancelled) else "failed", "error": {"code": getattr(exc, "code", "worker_error"), "message": str(exc)}}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
