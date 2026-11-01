"""Build a local portable Windows app from explicit, verified runtime inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess


def icon(source, destination):
    png = source.read_bytes()
    width, height = struct.unpack(">II", png[16:24])
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or not 1 <= width <= 256 or not 1 <= height <= 256:
        raise ValueError("The application icon must be a PNG at most 256 pixels wide and high")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", width % 256, height % 256, 0, 0, 1, 32, len(png), 22) + png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, help="Standalone CPython directory containing python.exe")
    parser.add_argument("--uv", required=True, help="Existing verified uv.exe")
    parser.add_argument("--destination")
    parser.add_argument("--cargo-target")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--installer", action="store_true", help="Also build a self-contained per-user Windows setup executable")
    args = parser.parse_args()
    manager = Path(__file__).resolve().parent
    resources = manager / ".bundle"
    resources.mkdir(exist_ok=True)
    python = Path(args.python).resolve()
    if not (python / "python.exe").is_file():
        raise ValueError("Select a standalone Python runtime directory")
    shutil.copytree(python, resources / "python", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(args.uv, resources / "uv.exe")
    worker = resources / "worker"
    (worker / "installation").mkdir(parents=True, exist_ok=True)
    for source in (manager.parent / "installation").glob("*.py"):
        shutil.copyfile(source, worker / "installation" / source.name)
    startup_probe = Path("shared/gradio/gradio_startup_patch.py")
    (worker / startup_probe).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manager.parent / startup_probe, worker / startup_probe)
    shutil.copytree(manager.parent / "installation/catalog_seed", worker / "installation/catalog_seed", dirs_exist_ok=True)
    (worker / "entry.py").write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nfrom installation.manager_worker import main\nraise SystemExit(main())\n", encoding="utf-8")
    inventory = {path.relative_to(resources).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(resources.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}
    manifest = {"schema_version": 1, "protocol": 1, "files": inventory}
    (manager / "src-tauri" / "runtime-manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    icon(manager / "assets" / "manager.png", manager / "src-tauri" / "icons" / "icon.ico")
    shutil.copyfile(manager / "assets" / "manager.png", manager / "ui" / "manager.png")
    if args.prepare_only:
        print(f"Prepared {len(inventory)} runtime files")
        return
    if not args.destination or not args.cargo_target:
        parser.error("--destination and --cargo-target are required for a portable build")
    subprocess.run(["npm.cmd", "run", "build:installer" if args.installer else "build"], cwd=manager, check=True)
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.cargo_target) / "release" / "wangp-manager.exe", destination / "WanGP Manager.exe")
    shutil.copytree(resources, destination / "runtime", dirs_exist_ok=True)
    print(destination / "WanGP Manager.exe")
    if args.installer:
        version = json.loads((manager / "package.json").read_text(encoding="utf-8"))["version"]
        installer = Path(args.cargo_target) / "release" / "bundle" / "nsis" / f"WanGP Manager_{version}_x64-setup.exe"
        download = destination / "WanGPManager.exe"
        shutil.copyfile(installer, download)
        print(download)


if __name__ == "__main__":
    main()
