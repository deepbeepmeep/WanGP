"""Standalone planning, dependency resolution and managed installation commands."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .catalog import load_config
from .environments import inspect_interpreter, inspect_registry
from .hardware import detect_hardware, select_device
from .plan import build_plan
from .uv import UvRuntime
from .execute import ProcessRunner
from .launch import launch_managed
from .releases import create_release
from .resolution import resolve_dependencies
from .state import atomic_json, read_json
from .transactions import TransactionEngine, cancel_operation, operation_status, prepare_transaction, version_history


def parser():
    result = argparse.ArgumentParser(description="WanGP shared installation services")
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Inspect an installation plan without changing files or packages")
    plan.add_argument("--root", default=".")
    plan.add_argument("--destination", required=True)
    plan.add_argument("--profile")
    plan.add_argument("--device", help="Exact hardware inventory device ID")
    plan.add_argument("--platform", choices=("win", "linux", "macos"))
    plan.add_argument("--amd-target", action="append", default=[])
    plan.add_argument("--recommended-extras", action="store_true", help="Explicitly include the profile's recommended optional components")
    plan.add_argument("--plugin", action="append", default=[])
    plan.add_argument("--output", help="Save the catalog plan as JSON")
    commands.add_parser("hardware", help="Read GPU identities and capability evidence")
    inspect = commands.add_parser("inspect", help="Read environment metadata without importing Torch")
    target = inspect.add_mutually_exclusive_group()
    target.add_argument("--python", help="Explicit interpreter path")
    target.add_argument("--root", default=".")
    create = commands.add_parser("create-env", help="Create a fresh uv environment; never replaces or activates an existing one")
    create.add_argument("--destination", required=True)
    create.add_argument("--python", required=True, help="Exact Python version or absolute interpreter path")
    create.add_argument("--uv", default="uv", help="Existing uv executable; binary bootstrap is a later milestone")
    create.add_argument("--data-directory")
    create.add_argument("--offline", action="store_true")
    create.add_argument("--no-seed", action="store_true")
    bundle = commands.add_parser("bundle", help="Snapshot an explicitly selected local source release")
    bundle.add_argument("--source", required=True)
    bundle.add_argument("--files", required=True, help="JSON array of owned source paths, relative to --source")
    bundle.add_argument("--destination", required=True)
    bundle.add_argument("--release-id", required=True)
    resolve = commands.add_parser("resolve", help="Resolve core and selected plugin dependencies together")
    resolve.add_argument("--plan", required=True)
    resolve.add_argument("--destination", required=True)
    resolve.add_argument("--extra-requirements", action="append", default=[])
    prepare = commands.add_parser("prepare", help="Preview a managed install, update or repair transaction")
    prepare.add_argument("--installation", required=True)
    prepare.add_argument("--release", required=True)
    prepare.add_argument("--resolution", required=True)
    prepare.add_argument("--operation", choices=("install", "update", "repair"), default="install")
    prepare.add_argument("--output", required=True)
    switch = commands.add_parser("switch-version", help="Prepare any selected release, including an older version")
    switch.add_argument("--installation", required=True)
    switch.add_argument("--release", required=True)
    switch.add_argument("--resolution", required=True)
    switch.add_argument("--environment", choices=("auto", "current", "new"), default="auto", help="Reuse compatible dependencies or build a separate environment")
    switch.add_argument("--output", required=True)
    versions = commands.add_parser("versions", help="List the current version and local rollback checkpoints")
    versions.add_argument("--installation", required=True)
    rollback = commands.add_parser("rollback", help="Prepare a rollback plan; apply it with the apply command")
    rollback.add_argument("--installation", required=True)
    rollback.add_argument("--checkpoint", required=True)
    rollback.add_argument("--output", required=True)
    apply = commands.add_parser("apply", help="Apply an unchanged prepared transaction")
    apply.add_argument("--plan", required=True)
    recover = commands.add_parser("recover", help="Recover an interrupted transaction")
    recover.add_argument("--installation", required=True)
    status = commands.add_parser("operation-status", help="Read operation or installation state")
    status.add_argument("--installation", required=True)
    status.add_argument("--operation-id")
    cancel = commands.add_parser("cancel", help="Request cancellation of an active operation")
    cancel.add_argument("--installation", required=True)
    cancel.add_argument("--operation-id", required=True)
    launch = commands.add_parser("launch", help="Launch managed WanGP while holding its maintenance lease")
    launch.add_argument("--installation", required=True)
    launch.add_argument("arguments", nargs=argparse.REMAINDER)
    for command in (resolve, apply, recover, launch):
        command.add_argument("--uv", default="uv")
        command.add_argument("--data-directory")
        command.add_argument("--offline", action="store_true")
    return result


def run(args):
    event = lambda value: print(json.dumps(value), file=sys.stderr, flush=True)
    if args.command == "bundle":
        return create_release(args.source, read_json(args.files), args.destination, args.release_id)
    if args.command == "resolve":
        runtime = UvRuntime(args.uv, args.data_directory, args.offline)
        return resolve_dependencies(read_json(args.plan), args.destination, runtime, extra_requirements=args.extra_requirements, runner=ProcessRunner(Path(args.destination) / "resolve.log", events=event))
    if args.command == "prepare":
        result = prepare_transaction(args.installation, args.release, args.resolution, args.operation)
        atomic_json(args.output, result)
        return result
    if args.command == "switch-version":
        result = prepare_transaction(args.installation, args.release, args.resolution, "switch-version", environment_policy=args.environment)
        atomic_json(args.output, result)
        return result
    if args.command == "versions":
        return version_history(args.installation)
    if args.command == "rollback":
        result = prepare_transaction(args.installation, operation="rollback", checkpoint=args.checkpoint)
        atomic_json(args.output, result)
        return result
    if args.command in ("apply", "recover"):
        engine = TransactionEngine(UvRuntime(args.uv, args.data_directory, args.offline), events=event)
        return engine.apply(read_json(args.plan)) if args.command == "apply" else engine.recover(args.installation)
    if args.command == "operation-status":
        return operation_status(args.installation, args.operation_id)
    if args.command == "cancel":
        return cancel_operation(args.installation, args.operation_id)
    if args.command == "launch":
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        return launch_managed(args.installation, UvRuntime(args.uv, args.data_directory, args.offline), arguments, event)
    if args.command == "hardware":
        return detect_hardware()
    if args.command == "inspect":
        return inspect_interpreter(args.python) if args.python else inspect_registry(args.root)
    if args.command == "create-env":
        runtime = UvRuntime(args.uv, args.data_directory, args.offline)
        return runtime.create_environment(args.destination, args.python, seed=not args.no_seed)
    root = Path(args.root).resolve()
    config = load_config(root / "setup_config.json")
    profile, targets, device = args.profile, args.amd_target, None
    if profile is None or args.device is not None:
        inventory = detect_hardware()
        device = select_device(inventory, args.device)
        profile = profile or device["profile"]
        if profile is None:
            raise ValueError(f"No verified automatic profile for {device['name']}; select --profile explicitly")
        if device.get("architecture", "").startswith("gfx") and not targets:
            targets = [device["architecture"]]
    result = build_plan(config, root, args.destination, profile, platform=args.platform, amd_targets=targets, recommended_extras=args.recommended_extras, plugins=args.plugin, device=device)
    if args.output:
        atomic_json(args.output, result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "error", "code": getattr(exc, "code", "cancelled" if isinstance(exc, KeyboardInterrupt) else "error"), "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2))
    return 1 if result.get("status") in ("unavailable", "failed", "cancelled", "recovery_required", "crashed") else 0
