"""Paged, content-addressed build catalogs and incremental Git publication."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from .state import InstallError, atomic_json, digest, read_json
from .version import HEADER_BYTES, version_from_header, version_metadata


PAGE_SIZE = 20
MINIMUM_VERSION = "18.888"
FIRST_COMMIT = "d951836f9fd8429dc703183623a1d25d77d7bce7"
CATALOG_BRANCH = "manager-catalog"


def revision(value, length=40):
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise InstallError("invalid_catalog", "Invalid catalog revision")
    return value


def supported_version(value):
    version_metadata({"schema_version": 1, "version": value})
    match = re.match(r"^(\d+)(?:\.(\d+))?", value)
    if not match or (int(match[1]), int(match[2] or 0)) < (18, 888):
        raise InstallError("unsupported_version", "Manager supports WanGP 18.888 and later")
    return value


def validate_index(value):
    if value.get("schema_version") != 1 or value.get("page_size") != PAGE_SIZE or value.get("minimum_version") != MINIMUM_VERSION:
        raise InstallError("invalid_catalog", "Unsupported version catalog")
    revision(value["head"])
    revision(value["first_commit"])
    pages = value.get("pages")
    if not isinstance(pages, list) or not pages or len(pages) > 10000:
        raise InstallError("invalid_catalog", "Invalid catalog page list")
    for number, page in enumerate(pages):
        revision(page["sha256"], 64)
        if type(page.get("count")) is not int or not 1 <= page["count"] <= PAGE_SIZE or number < len(pages) - 1 and page["count"] != PAGE_SIZE:
            raise InstallError("invalid_catalog", "Catalog pages must contain 20 builds except the last page")
    return value


def validate_page(value, descriptor):
    if digest(value) != descriptor["sha256"] or value.get("schema_version") != 1:
        raise InstallError("invalid_catalog", "Catalog page checksum mismatch")
    builds = value.get("builds")
    if not isinstance(builds, list) or len(builds) != descriptor["count"]:
        raise InstallError("invalid_catalog", "Catalog page count mismatch")
    seen = set()
    for build in builds:
        revision(build["commit"])
        supported_version(build["version"])
        if build["commit"] in seen or not isinstance(build.get("message"), str) or len(build["message"]) > 65536 or not isinstance(build.get("date"), str):
            raise InstallError("invalid_catalog", "Invalid build metadata")
        seen.add(build["commit"])
    return builds


def write_catalog(builds, destination, first_commit):
    destination = Path(destination)
    pages = []
    for offset in range(0, len(builds), PAGE_SIZE):
        page = {"schema_version": 1, "builds": builds[offset:offset + PAGE_SIZE]}
        descriptor = {"sha256": digest(page), "count": len(page["builds"])}
        validate_page(page, descriptor)
        atomic_json(destination / "pages" / (descriptor["sha256"] + ".json"), page)
        pages.append(descriptor)
    index = validate_index({"schema_version": 1, "page_size": PAGE_SIZE, "minimum_version": MINIMUM_VERSION, "first_commit": first_commit, "head": builds[0]["commit"], "pages": pages})
    atomic_json(destination / "index.json", index)
    return index


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], encoding="utf-8", errors="strict").strip()


def update_catalog(root, destination, *, ref="HEAD", first_commit=FIRST_COMMIT):
    """Read version headers only for newly published first-parent commits."""
    destination = Path(destination)
    previous = []
    if (destination / "index.json").exists():
        old = validate_index(read_json(destination / "index.json"))
        if old["first_commit"] != first_commit:
            raise InstallError("catalog_floor_changed", "The catalog starting commit cannot change implicitly")
        for page in old["pages"]:
            previous.extend(validate_page(read_json(destination / "pages" / (page["sha256"] + ".json")), page))
    stop = previous[0]["commit"] if previous else first_commit
    revision(stop)
    head = revision(git(root, "rev-parse", ref))
    chain = git(root, "rev-list", "--first-parent", head).splitlines()
    if stop not in chain:
        raise InstallError("catalog_history_missing", "Fetch history back to the previous catalog head; rewritten history requires explicit catalog repair")
    commits = chain[:chain.index(stop) + (0 if previous else 1)]
    builds = []
    for commit in commits:
        # No checkout and no application imports. Git reads the committed blob.
        process = subprocess.Popen(["git", "-C", str(root), "show", commit + ":wgp.py"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            header = process.stdout.read(HEADER_BYTES)
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()
        version = supported_version(version_from_header(header.decode("utf-8-sig")))
        date, message = git(root, "show", "-s", "--format=%cI%n%B", commit).split("\n", 1)
        builds.append({"version": version, "commit": commit, "date": date, "message": message})
    if not builds and previous:
        return {"head": head, "added": 0, "builds": len(previous)}
    write_catalog(builds + previous, destination, first_commit)
    return {"head": head, "added": len(builds), "builds": len(builds) + len(previous)}


def main():
    parser = argparse.ArgumentParser(description="Update WanGP Manager's paginated build catalog")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--first-commit", default=FIRST_COMMIT)
    args = parser.parse_args()
    print(json.dumps(update_catalog(args.root, args.output, ref=args.ref, first_commit=args.first_commit)))


if __name__ == "__main__":
    main()
