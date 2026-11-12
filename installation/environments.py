"""Read-only environment inspection and interpreter paths."""

import json
import os
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

from .catalog import os_key


def interpreter_path(directory, kind="uv", platform=None):
    platform = os_key(platform)
    path = PureWindowsPath(directory) if platform == "win" else PurePosixPath(directory)
    if kind not in ("uv", "venv", "conda"):
        raise ValueError(f"Unsupported environment type: {kind}")
    if platform == "win":
        return str(path / "python.exe" if kind == "conda" else path / "Scripts" / "python.exe")
    return str(path / "bin" / "python")


def clean_environment(base=None):
    """Keep host/network configuration, removing ambient Python/package targeting."""
    base = dict(os.environ if base is None else base)
    roots = [Path(value).resolve() for key, value in base.items() if key.upper() in ("VIRTUAL_ENV", "CONDA_PREFIX") and value]
    env = {key: value for key, value in base.items() if not key.upper().startswith(("UV_", "PIP_", "PYTHON", "CONDA_", "_CE_")) and key.upper() not in ("VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT")}
    path_key = next((key for key in env if key.upper() == "PATH"), "PATH")
    entries = []
    for entry in env.get(path_key, "").split(os.pathsep):
        if entry and not any(Path(entry).resolve().is_relative_to(root) for root in roots):
            entries.append(entry)
    env[path_key] = os.pathsep.join(entries)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def read_registry(root):
    root = Path(root).resolve()
    filename = root / "envs.json"
    if not filename.exists():
        return {"active": None, "envs": {}}
    data = json.loads(filename.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("envs"), dict):
        raise ValueError(f"Invalid environment registry: {filename}")
    for name, entry in data["envs"].items():
        if not isinstance(entry, dict) or entry.get("type") not in ("uv", "venv", "conda", "none") or not isinstance(entry.get("path"), str):
            raise ValueError(f"Invalid environment entry: {name}")
    if data.get("active") is not None and (not isinstance(data["active"], str) or data["active"] not in data["envs"]):
        raise ValueError(f"Unknown active environment in {filename}: {data['active']}")
    return data


INSPECT_SCRIPT = """
import ast, importlib.metadata, json, sys
from pathlib import Path
packages = {}
for name in ('torch', 'torchvision', 'torchaudio', 'triton', 'triton-windows', 'sageattention', 'spas-sage-attn', 'flash-attn', 'wgpkernels'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
cuda = None
if packages['torch']:
    version_file = importlib.metadata.distribution('torch').locate_file('torch/version.py')
    for node in ast.parse(version_file.read_text(encoding='utf-8')).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == 'cuda' for target in targets):
                cuda = ast.literal_eval(node.value)
prefix = Path(sys.prefix)
conda = ('base' if (prefix / 'envs').is_dir() else prefix.name) if (prefix / 'conda-meta').is_dir() else None
print(json.dumps({'python': sys.version.split()[0], 'executable': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'packages': packages, 'cuda': cuda, 'conda_environment': conda}))
"""


def inspect_interpreter(interpreter, timeout=15):
    # Outside Windows a venv's bin/python links to its base interpreter: run through the link, or the venv is lost.
    interpreter = str(Path(interpreter).resolve()) if os.name == "nt" else os.path.abspath(interpreter)
    try:
        result = subprocess.run([interpreter, "-I", "-B", "-c", INSPECT_SCRIPT], stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=clean_environment(), check=True, **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        data = json.loads(result.stdout)
        return {"status": "inspected", "interpreter": interpreter, **data, "validation": "metadata_only"}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"status": "unavailable", "interpreter": interpreter, "error": str(exc)}


def inspect_registry(root, timeout=15):
    root = Path(root).resolve()
    registry = read_registry(root)
    records = {}
    for name, entry in registry["envs"].items():
        if entry["type"] == "none":
            records[name] = {**entry, "status": "unresolved", "error": "System environments require an explicitly selected interpreter"}
            continue
        directory = (root / entry["path"]).resolve()
        inspection = inspect_interpreter(interpreter_path(directory, entry["type"]), timeout)
        if inspection["status"] == "inspected" and Path(inspection["prefix"]).resolve() != directory:
            inspection.update(status="unavailable", error="Interpreter prefix differs from the registered environment directory")
        records[name] = {**entry, **inspection}
    return {"active": registry["active"], "envs": records}


def installation_environments(root):
    """Prefer the saved selection and local launcher/Pinokio environments."""
    root = Path(root).resolve()
    registry = inspect_registry(root)
    candidates = {}
    for name, entry in registry["envs"].items():
        if entry.get("interpreter"):
            candidates[entry["interpreter"]] = {**entry, "name": name, "saved": name == registry["active"]}
    for name in ("env_uv", "venv", ".venv", "env", "environment"):
        directory = root / name
        kind = "conda" if (directory / "conda-meta").is_dir() else "venv"
        python = interpreter_path(directory, kind)
        if Path(python).is_file() and python not in candidates:
            candidates[python] = {**inspect_interpreter(python), "name": name, "saved": False}
    usable = [item for item in candidates.values() if item["status"] == "inspected" and item["packages"]["torch"]]
    preferred = next((item for item in usable if item["saved"]), None)
    if preferred is None and len(usable) == 1:
        preferred = usable[0]
    return {"environments": list(candidates.values()), "python": preferred["interpreter"] if preferred else None}


def managed_record(root):
    root = Path(root).resolve()
    for parent in (root, root.parent):
        for name in ("installation.json", "active_operation.json"):
            candidate = parent / ".manager" / name
            if candidate.exists():
                return candidate
    return None
