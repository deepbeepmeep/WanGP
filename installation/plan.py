"""Read-only installation plans derived from the existing setup catalog."""

import hashlib
import json
from pathlib import Path

from .catalog import catalog_digest, component_commands, kernel_variant, os_key, select_profile, validate_catalog
from .environments import interpreter_path, managed_record


def build_plan(config, root, destination, profile, *, platform=None, amd_targets=(), recommended_extras=False, plugins=(), device=None):
    validate_catalog(config)
    root, destination = Path(root).resolve(), Path(destination).resolve()
    platform = os_key(platform)
    selection = select_profile(config, profile)
    inputs, steps, unavailable = {}, [], []

    def requirement(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Requirements must be inside the selected source root: {path}")
        try:
            inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"Cannot read requirements: {path}: {exc}") from exc
        return str(path)

    python_version = config["components"]["python"][selection["python"]]["ver"]
    interpreter = interpreter_path(str(destination), platform=platform)
    steps.append({"id": "create_environment", "title": "Creating Python Environment", "kind": "create_environment", "python": python_version, "destination": str(destination), "seed_pip": True})

    def component(group, key):
        commands = component_commands(config, group, key, platform, selection["python"], amd_targets)
        if not commands:
            unavailable.append({"component": f"{group}.{key}", "reason": f"No catalog recipe for {platform}"})
        for index, command in enumerate(commands):
            steps.append({"id": f"{group}.{key}.{index}", "title": f"Installing {config['components'][group][key]['label']}", "kind": "install_packages", **command})

    component("torch", selection["torch"])
    steps.append({"id": "requirements", "title": "Installing WanGP Requirements", "kind": "install_packages", "args": ["-r", requirement(root / "requirements.txt")], "environment": {}})
    if recommended_extras:
        for group in ("triton", "sage", "sparge", "flash"):
            key = selection.get(group)
            if key:
                component(group, key)
        for key in selection["kernels"]:
            selected = kernel_variant(key, selection["torch"], selection["python"], amd_targets)
            if selected is None:
                unavailable.append({"component": f"kernels.{key}", "reason": "No compatible kernel build for this selected stack/device"})
            else:
                component("kernels", selected)
    for plugin in dict.fromkeys(plugins):
        path = Path(plugin)
        if path.name != plugin or plugin in (".", "..") or "\\" in plugin:
            raise ValueError(f"Plugin must be one directory name: {plugin!r}")
        steps.append({"id": f"plugin.{plugin}", "title": f"Installing Plugin Requirements - {plugin}", "kind": "install_packages", "args": ["-r", requirement(root / "plugins" / plugin / "requirements.txt")], "environment": {}})
    blockers = []
    if platform not in selection.get("platforms", ("win", "linux", "macos")):
        blockers.append(f"Hardware profile {profile} is not supported on {platform}.")
    if platform != os_key():
        blockers.append("Cross-platform catalog preview; paths and runtime have not been validated on the target OS.")
    if destination.exists():
        blockers.append("Destination already exists; environment replacement requires a new final path.")
    if destination == root or root.is_relative_to(destination):
        blockers.append("The environment destination must not contain the application source root.")
    record = managed_record(root)
    if record:
        blockers.append(f"Managed installation detected at {record}; resolve a separate release snapshot, then use prepare/apply for transactional changes.")
    if unavailable:
        blockers.append("Some selected components are unavailable; explicitly reconcile the selection before applying.")
    plan = {
        "schema_version": 1,
        "operation": "prepare_environment",
        "root": str(root),
        "destination": str(destination),
        "platform": platform,
        "profile": profile,
        "python": python_version,
        "interpreter": interpreter,
        "installer": "uv",
        "catalog_sha256": catalog_digest(config),
        "requirements_sha256": inputs,
        "selected_device": device,
        "hardware_validation": "inventory_only" if device is not None else "pending",
        "amd_targets": list(amd_targets),
        "recommended_extras": recommended_extras,
        "steps": steps,
        "unavailable_components": unavailable,
        "blockers": blockers,
        "dependency_resolution": "pending",
        "apply_supported": False,
        "package_index_strategy": "first-index",
    }
    # The plan is a catalog translation, not a resolved or signed release lock.
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan
