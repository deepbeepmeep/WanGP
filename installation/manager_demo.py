"""Small, explicitly labelled local releases for exercising Manager without ML downloads."""

from pathlib import Path
import sys
import zipfile

from .releases import create_release
from .state import atomic_json


def create_demo(directory):
    directory = Path(directory)
    wheels = directory / "wheels"
    wheels.mkdir(parents=True, exist_ok=False)
    releases = []
    for version in ("1.0", "2.0"):
        name = "wangp_manager_demo"
        info = f"{name}-{version}.dist-info"
        filename = wheels / f"{name}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(filename, "w") as archive:
            archive.writestr(f"{name}.py", f"VERSION = {version!r}\n")
            archive.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: wangp-manager-demo\nVersion: {version}\n")
            archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            archive.writestr(f"{info}/RECORD", "")
        source = directory / ("source-" + version)
        source.mkdir()
        config = {"components": {"python": {"demo": {"label": "Bundled Python", "ver": sys.version.split()[0]}}, "torch": {"demo": {"label": "Sample Package (No GPU)", "cmd": f"wangp-manager-demo=={version}"}}}, "gpu_profiles": {"Demo": {"python": "demo", "torch": "demo", "kernels": [], "platforms": ["win", "linux", "macos"]}}}
        atomic_json(source / "setup_config.json", config)
        (source / "requirements.txt").write_text(f"--no-index\n--find-links {wheels.as_uri()}\nwangp-manager-demo=={version}\n")
        (source / "wgp.py").write_text('''import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from wangp_manager_demo import VERSION
parser = argparse.ArgumentParser()
parser.add_argument("--server-port", type=int, default=7865)
args = parser.parse_args()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        content = ("<!doctype html><html><head><meta charset='utf-8'><title>Manager Test Installation</title></head><body style='font:18px system-ui;background:#f6f7fb;padding:70px;color:#222b40'><h1>Test installation " + VERSION + "</h1><p>WanGP Manager started this isolated environment.</p><p>This is a tiny test server. It does not generate media or load models.</p><p>Return to Manager to stop it, switch versions or roll back.</p></body></html>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
HTTPServer(("127.0.0.1", args.server_port), Handler).serve_forever()
''', encoding="utf-8")
        target = directory / ("release-" + version)
        create_release(source, ["setup_config.json", "requirements.txt", "wgp.py"], target, "Test " + version)
        releases.append({"version": version, "path": str(target)})
    return {"releases": releases, "root": str(directory / "installation"), "profile": "Demo", "demo": True}
