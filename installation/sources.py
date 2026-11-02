"""Public GitHub sources, pinned revisions and bounded source downloads."""

import hashlib
import json
import re
import stat
import time
import uuid
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .catalog import validate_catalog
from .releases import load_release, source_manifest, source_path
from .state import InstallError, InstallationLock, atomic_json, digest, inside, plain_path, read_json, relative_path
from .version import VERSION_FILE, local_version, version_metadata


OFFICIAL_REPOSITORY = "https://github.com/deepbeepmeep/WanGP"


def repository_name(url):
    parsed = urlsplit(url.strip())
    path = parsed.path.rstrip("/").removesuffix(".git")
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com" or parsed.query or parsed.fragment or not re.fullmatch(r"/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", path) or path.split("/")[-1] in (".", ".."):
        raise InstallError("invalid_repository", "Use a public https://github.com/owner/repository URL, or choose a local source folder")
    return path.lstrip("/")


def commit_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise InstallError("invalid_revision", "Select a version from the repository's version list")
    return value


def response(url):
    request = Request(url, headers={"User-Agent": "WanGP-Manager/0.1", "Accept": "application/vnd.github+json"})
    return urlopen(request, timeout=30)


def get_json(url, runner, *, optional=False):
    runner.check_cancelled()
    try:
        with response(url) as stream:
            data = stream.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise InstallError("source_metadata_limit", "Repository metadata exceeded the 2 MiB limit")
        runner.check_cancelled()
        return json.loads(data.decode("utf-8-sig"))
    except HTTPError as exc:
        if optional and exc.code == 404:
            return None
        if exc.code in (403, 429):
            raise InstallError("github_rate_limit", "GitHub refused the request or its public request limit was reached. Wait before refreshing versions, or use a local source folder") from exc
        raise InstallError("source_unavailable", f"GitHub returned HTTP {exc.code}. Check the public repository URL and selected version") from exc
    except (URLError, TimeoutError) as exc:
        raise InstallError("source_unavailable", f"Could not reach GitHub: {exc}") from exc


def remote_file(repo, commit, path, runner, *, optional=False):
    return get_json(f"https://raw.githubusercontent.com/{repo}/{commit_id(commit)}/{path}", runner, optional=optional)


def list_versions(url, data, runner, emit, *, refresh=False, page=1, catalog_id=None):
    from .version_catalog import CATALOG_BRANCH, validate_index, validate_page, revision
    repo = repository_name(url)
    if type(page) is not int or not 1 <= page <= 10000:
        raise InstallError("invalid_page", "Invalid version history page")
    cache = Path(data) / "source-catalogs" / hashlib.sha256(repo.lower().encode()).hexdigest()
    current = cache / "current.json"
    seed = Path(__file__).parent / "catalog_seed"
    base = f"https://raw.githubusercontent.com/{repo}/{CATALOG_BRANCH}"
    emit({"type": "progress", "title": "Finding WanGP Versions", "message": f"Loading build page {page}"})
    stale = False
    if page > 1:
        revision(catalog_id, 64)
        snapshot = cache / "snapshots" / (catalog_id + ".json")
        if not snapshot.is_file():
            raise InstallError("catalog_expired", "Reload versions before requesting an older page")
        record = read_json(snapshot)
    elif current.exists() and not refresh and time.time() - current.stat().st_mtime < 300:
        record = read_json(current)
    else:
        try:
            index = get_json(base + "/index.json", runner, optional=True)
            if index is None:
                if current.exists():
                    raise InstallError("source_unavailable", "The published catalog is temporarily unavailable")
                if repo.lower() != repository_name(OFFICIAL_REPOSITORY).lower():
                    raise InstallError("catalog_unpublished", "This fork has no Manager catalog. Enable the Manager catalog workflow, or select a local checkout")
                index = read_json(seed / "index.json")
                origin = "bundled"
            else:
                origin = "published"
            validate_index(index)
            record = {"index": index, "catalog_id": digest(index), "origin": origin}
            atomic_json(cache / "snapshots" / (record["catalog_id"] + ".json"), record)
            atomic_json(current, record)
        except InstallError as exc:
            if not current.exists() or exc.code not in ("source_unavailable", "github_rate_limit"):
                raise
            record, stale = read_json(current), True
    index = validate_index(record["index"])
    if digest(index) != record["catalog_id"] or page > len(index["pages"]):
        raise InstallError("invalid_catalog", "Invalid catalog snapshot or page number")
    descriptor = index["pages"][page - 1]
    filename = descriptor["sha256"] + ".json"
    cached_page = cache / "pages" / filename
    if cached_page.exists():
        contents = read_json(cached_page)
    elif record["origin"] == "bundled":
        contents = read_json(seed / "pages" / filename)
    else:
        contents = get_json(base + "/pages/" + filename, runner)
    builds = validate_page(contents, descriptor)
    if page == 1 and builds[0]["commit"] != index["head"] or page == len(index["pages"]) and builds[-1]["commit"] != index["first_commit"]:
        raise InstallError("invalid_catalog", "Catalog endpoint revisions do not match its pages")
    if not cached_page.exists():
        atomic_json(cached_page, contents)
    versions = []
    for build in builds:
        latest = build["commit"] == index["head"]
        subject = build["message"].splitlines()[0] if build["message"] else ""
        label = ("Latest — " if latest else "") + f"{build['version']} ({build['commit'][:8]})" + (" — " + subject if subject else "")
        versions.append({**build, "label": label, "kind": "latest" if latest else "version"})
    return {"repository": "https://github.com/" + repo, "versions": versions, "page": page,
            "has_more": page < len(index["pages"]), "catalog_id": record["catalog_id"],
            "minimum_version": index["minimum_version"], "catalog_origin": record["origin"], "stale": stale}


def remote_catalog(url, commit, runner):
    repo, commit = repository_name(url), commit_id(commit)
    config = validate_catalog(remote_file(repo, commit, "setup_config.json", runner))
    return config, None


def unpack_source(archive, destination, runner, emit=None, *, inventory=None):
    destination = plain_path(destination)
    filenames, names, total, prefix, last_event = [], set(), 0, None, 0
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.infolist()) > 100000:
            raise InstallError("source_archive_limit", "Source archive contains too many entries")
        entries = []
        # Validate the complete inventory before writing any archive member.
        for entry in bundle.infolist():
            runner.check_cancelled()
            parts = entry.filename.rstrip("/").split("/")
            relative_path(entry.filename.rstrip("/"))
            if prefix is None:
                prefix = parts[0]
            if parts[0] != prefix or stat.S_ISLNK(entry.external_attr >> 16):
                raise InstallError("unsafe_source_archive", "Source archive contains multiple roots or symbolic links")
            if len(parts) == 1 or entry.is_dir():
                continue
            filename = "/".join(parts[1:])
            if filename.casefold() in names:
                raise InstallError("unsafe_source_archive", f"Source archive contains duplicate paths: {filename}")
            names.add(filename.casefold())
            total += entry.file_size
            if total > 3 * 1024**3:
                raise InstallError("source_archive_limit", "Source archive exceeds the 3 GiB expanded limit")
            try:
                source_path(filename)
            except InstallError as exc:
                if exc.code == "protected_file":
                    continue
                raise
            if filename.startswith("manager/"):
                continue
            entries.append((filename, entry))
        files = {filename.casefold() for filename, _ in entries}
        if any(parent.as_posix().casefold() in files for filename, _ in entries for parent in Path(filename).parents if str(parent) != "."):
            raise InstallError("unsafe_source_archive", "Source archive contains a file/directory collision")
        if not {"wgp.py", "setup_config.json"}.issubset(filename for filename, _ in entries):
            raise InstallError("invalid_source", "This revision does not contain wgp.py and setup_config.json")
        def progress(completed):
            if emit:
                emit({"type": "progress", "title": "Preparing WanGP Source", "message": f"{completed:,} / {len(entries):,} files unpacked and checked", "completed": completed, "total": len(entries)})
        progress(0)
        for filename, entry in entries:
            runner.check_cancelled()
            target = inside(destination, filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            hasher = hashlib.sha256() if inventory is not None else None
            with bundle.open(entry) as incoming, target.open("xb") as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    runner.check_cancelled()
                    outgoing.write(chunk)
                    if hasher is not None:
                        hasher.update(chunk)
            if inventory is not None:
                inventory[filename] = hasher.hexdigest()
            filenames.append(filename)
            if emit and time.monotonic() - last_event >= 1 / 3:
                progress(len(filenames))
                last_event = time.monotonic()
        progress(len(filenames))
    return filenames


def cached_download_release(url, commit, cache, runner, emit):
    repo, commit = repository_name(url), commit_id(commit)
    cache = inside(cache, digest({"repository": repo.lower(), "commit": commit})[:24])
    release = inside(cache, "release")
    with InstallationLock(cache):
        runner.check_cancelled()
        if release.exists():
            emit({"type": "progress", "title": "Reusing Downloaded WanGP", "message": "Verifying cached source files; no source download is needed."})
            manifest = load_release(release, emit=emit, check_cancelled=runner.check_cancelled)
            if manifest.get("origin", {}).get("commit") != commit or manifest.get("origin", {}).get("repository", "").lower() != ("https://github.com/" + repo).lower():
                raise InstallError("changed_release", "Cached source does not match the selected repository and commit")
            runner.check_cancelled()
            return release
        # Failed/cancelled downloads remain separate from the published cache entry.
        downloaded = download_release(url, commit, inside(cache, uuid.uuid4().hex[:12]), runner, emit)
        runner.check_cancelled()
        downloaded.replace(release)
        return release


def download_release(url, commit, directory, runner, emit):
    repo, commit = repository_name(url), commit_id(commit)
    directory = plain_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "source.zip"
    count, last_event = 0, 0
    emit({"type": "progress", "title": "Downloading WanGP", "message": f"{repo} · {commit[:8]} - Connecting to GitHub"})
    try:
        with response(f"https://codeload.github.com/{repo}/zip/{commit}") as incoming, archive.open("xb") as outgoing:
            total = int(incoming.headers.get("Content-Length", 0))
            while chunk := incoming.read(1024 * 1024):
                runner.check_cancelled()
                count += len(chunk)
                if count > 1024**3:
                    raise InstallError("source_archive_limit", "Source download exceeds the 1 GiB limit")
                outgoing.write(chunk)
                if time.monotonic() - last_event >= 1 / 3:
                    emit({"type": "progress", "title": "Downloading WanGP", "message": f"{count / 1024**2:.1f} MB downloaded", "completed": count, "total": total or None})
                    last_event = time.monotonic()
        emit({"type": "progress", "title": "Downloading WanGP", "message": f"{count / 1024**2:.1f} MB downloaded", "completed": count, "total": count})
        emit({"type": "progress", "title": "Preparing WanGP Source", "message": "Reading the source archive"})
        release = directory / "release"
        source = release / "source"
        inventory = {}
        unpack_source(archive, source, runner, emit, inventory=inventory)
        runner.check_cancelled()
        version = local_version(source)
        # This is a disposable staging cache, not an active installation. Hash
        # the bytes during extraction instead of reopening/copying/fsyncing
        # every file again. Transaction preparation validates the staged files;
        # activation still uses durable, checked copies and rollback journals.
        manifest = source_manifest(inventory, (version + "-" if version else "") + commit[:12])
        manifest.update(trust="github_commit_https", origin={"repository": "https://github.com/" + repo, "commit": commit, "version": version})
        manifest["manifest_sha256"] = digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        atomic_json(release / "release.json", manifest)
        runner.check_cancelled()
        return release
    except (HTTPError, URLError, TimeoutError, zipfile.BadZipFile) as exc:
        raise InstallError("source_download_failed", f"Could not download this WanGP revision: {exc}") from exc
    finally:
        # Only the temporary archive created by this operation is disposable.
        archive.unlink(missing_ok=True)
