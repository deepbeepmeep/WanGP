"""Synchronize shared/version.json from wgp.py; pass --check to only validate."""

from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    return subprocess.call(
        [sys.executable, "-m", "installation.version", "--root", str(root), *sys.argv[1:]],
        cwd=root,
    )


if __name__ == "__main__":
    sys.exit(main())
