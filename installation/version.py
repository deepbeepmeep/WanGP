"""Read WanGP's version without importing application code."""

import argparse
import ast
import json
from pathlib import Path
import re

from .state import InstallError, atomic_json

VERSION_FILE = "shared/version.json"
HEADER_BYTES = 65536


def version_metadata(value):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("version"), str) or not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,79}", value["version"]):
        raise InstallError("invalid_version", "shared/version.json must contain schema_version 1 and a version string")
    return value["version"]


def version_from_header(text):
    for line in text.splitlines():
        if not re.match(r"^WanGP_version\s*(?:=|:)", line):
            continue
        try:
            statement = ast.parse(line).body[0]
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = ast.literal_eval(statement.value)
        except (SyntaxError, ValueError, TypeError):
            continue
        return version_metadata({"schema_version": 1, "version": value})
    return None


def local_version(root):
    root = Path(root)
    application = root / "wgp.py"
    if application.is_file():
        with application.open("rb") as stream:
            version = version_from_header(stream.read(HEADER_BYTES).decode("utf-8-sig", errors="replace"))
        if version:
            return version
    metadata = root / VERSION_FILE
    return version_metadata(json.loads(metadata.read_text(encoding="utf-8-sig"))) if metadata.is_file() else None


def main():
    parser = argparse.ArgumentParser(description="Generate release version metadata from WanGP_version in wgp.py")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="Check existing metadata without changing files")
    args = parser.parse_args()
    with (args.root / "wgp.py").open("rb") as stream:
        version = version_from_header(stream.read(HEADER_BYTES).decode("utf-8-sig"))
    if version is None:
        parser.error("wgp.py must declare a literal WanGP_version string near its beginning")
    value = {"schema_version": 1, "version": version}
    path = args.root / VERSION_FILE
    if args.check:
        if not path.exists() or json.loads(path.read_text(encoding="utf-8-sig")) != value:
            parser.error("Version metadata is stale; run python -m installation.version")
    else:
        atomic_json(path, value)
    print(version)


if __name__ == "__main__":
    main()
