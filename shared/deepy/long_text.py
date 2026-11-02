# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import difflib
import hashlib
import os
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import threading
import urllib.request
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from shared.deepy.config import DEEPY_LONG_TEXT_TOOLS_EXPERIMENT


_RUNTIME_ROOT = Path(__file__).resolve().parent / "_runtime"
_WORKSPACES_ROOT = _RUNTIME_ROOT / "workspaces"
_RIPGREP_ROOT = _RUNTIME_ROOT / "ripgrep"
_RIPGREP_VERSION = "15.2.0"
_MAX_RG_ARGUMENT_CHARS = 8192
_MAX_RG_OUTPUT_CHARS = 64000
_MAX_TEXT_EDIT_CHARS = 1024 * 1024
_MAX_PROMPT_FILE_BYTES = 4 * 1024 * 1024
_PROMPT_FILE_RE = re.compile(r'^\s*@file\(\s*"([^"\r\n]+)"\s*\)\s*$')
_PROMPT_FIELDS = {"prompt", "negative_prompt", "alt_prompt", "audio_prompt", "lyrics", "music_caption", "voice_description"}
_RUNTIME_LOCK = threading.RLock()
_WORKSPACES_PURGED = False

_RIPGREP_ASSETS = {
    ("windows", "x86_64"): (
        "ripgrep-15.2.0-x86_64-pc-windows-msvc.zip",
        "71b2fef860abe467217a538ff31de02f5258807c0129f771846f87bd029aafc5",
    ),
    ("windows", "aarch64"): (
        "ripgrep-15.2.0-aarch64-pc-windows-msvc.zip",
        "e4abca10c3a64ebea742667dd7009449d49403db5460dd6873e389fa2945360f",
    ),
    ("linux", "x86_64"): (
        "ripgrep-15.2.0-x86_64-unknown-linux-musl.tar.gz",
        "33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c",
    ),
    ("linux", "aarch64"): (
        "ripgrep-15.2.0-aarch64-unknown-linux-musl.tar.gz",
        "800b1e7206afe799dfb5a6901f23147cfaabe0e52210538100f61e86e1740915",
    ),
}

_RG_FLAG_OPTIONS = {
    "-n", "--line-number", "-i", "--ignore-case", "-s", "--case-sensitive", "-S", "--smart-case",
    "-F", "--fixed-strings", "-w", "--word-regexp", "-x", "--line-regexp", "-U", "--multiline",
    "--multiline-dotall", "-l", "--files-with-matches", "-L", "--files-without-match", "-c", "--count",
    "--count-matches", "--files", "--hidden", "--no-ignore", "--no-ignore-vcs", "-o", "--only-matching",
    "--trim", "-P", "--pcre2", "--crlf", "--stats", "--no-messages",
}
_RG_VALUE_OPTIONS = {
    "-g", "--glob", "--iglob", "-t", "--type", "-T", "--type-not", "-A", "--after-context", "-B", "--before-context",
    "-C", "--context", "-m", "--max-count", "--max-depth", "--max-filesize", "-j", "--threads", "-e", "--regexp",
    "-r", "--replace", "--engine", "--sort", "--sortr", "--encoding",
}
_RG_ATTACHED_VALUE_OPTIONS = ("-g", "-t", "-T", "-A", "-B", "-C", "-m", "-j", "-e", "-r")


def long_text_tools_active(policy) -> bool:
    return bool(policy is not None and policy.write_enabled and (DEEPY_LONG_TEXT_TOOLS_EXPERIMENT or getattr(policy, "prompt_read_only", False)))


def _session_name(session_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(session_id or "").strip()).strip("_-")
    if not normalized:
        raise ValueError("Deepy chat session id is empty.")
    return normalized[:128]


def _purge_previous_workspaces() -> None:
    global _WORKSPACES_PURGED
    with _RUNTIME_LOCK:
        if _WORKSPACES_PURGED:
            return
        if _WORKSPACES_ROOT.exists():
            shutil.rmtree(_WORKSPACES_ROOT)
        _WORKSPACES_ROOT.mkdir(parents=True, exist_ok=True)
        _WORKSPACES_PURGED = True


def add_session_workspace(policy, session_id: str, workspace_path: str | os.PathLike[str] | None = None):
    if not long_text_tools_active(policy):
        return policy
    if workspace_path is None:
        _purge_previous_workspaces()
        workspace = (_WORKSPACES_ROOT / _session_name(session_id)).resolve()
    else:
        workspace = Path(workspace_path).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    used = {alias.casefold() for alias in policy.aliases}
    alias = "documents"
    suffix = 2
    while alias.casefold() in used:
        alias = f"documents{suffix}"
        suffix += 1
    return policy.with_root(workspace, alias)


def workspace_mount(policy) -> tuple[str, Path] | None:
    for alias, path in policy.mounts:
        if re.fullmatch(r"documents\d*", alias, flags=re.IGNORECASE):
            return alias, path
    return None


def long_text_system_instructions(policy) -> str:
    mount = workspace_mount(policy)
    if not long_text_tools_active(policy) or mount is None:
        return ""
    workspace = f"@{mount[0]}"
    return (
        f"Experimental long-document tools are enabled. Use {workspace} as session working storage; it is saved and restored with the active Deepy session. "
        "Use rg(arguments) for bounded, read-only text search. Its syntax is rg options plus one pattern, then `--` and authorized @alias paths; omit paths to search the temporary workspace. "
        "Use append_text(file_path, text) to create a missing UTF-8 file or append exact literal text; include every intended newline and do not prefix lines with patch markers. "
        "Use edit(file_path, old_string, new_string, replace_all=false) for exact replacement. Search for enough local context that old_string occurs once, preserve its whitespace and line endings exactly, then search or read back the changed area. "
        f"A generation prompt may be the exact reference `@file(\"{workspace}/prompt.txt\")`; WanGP snapshots the UTF-8 file into the queued task and preserves blank-line sliding-window boundaries."
    )


def _platform_key() -> tuple[str, str]:
    operating_system = platform.system().strip().casefold()
    machine = platform.machine().strip().casefold()
    operating_system = "windows" if operating_system == "windows" else "linux" if operating_system == "linux" else operating_system
    machine = "x86_64" if machine in {"amd64", "x86_64"} else "aarch64" if machine in {"arm64", "aarch64"} else machine
    return operating_system, machine


def ensure_ripgrep() -> Path:
    key = _platform_key()
    try:
        asset_name, expected_sha256 = _RIPGREP_ASSETS[key]
    except KeyError as exc:
        raise RuntimeError(f"The Deepy ripgrep experiment does not support {key[0]} {key[1]}.") from exc
    executable_name = "rg.exe" if key[0] == "windows" else "rg"
    target_dir = _RIPGREP_ROOT / _RIPGREP_VERSION / f"{key[0]}-{key[1]}"
    target = target_dir / executable_name
    with _RUNTIME_LOCK:
        if target.is_file():
            return target
        url = f"https://github.com/BurntSushi/ripgrep/releases/download/{_RIPGREP_VERSION}/{asset_name}"
        request = urllib.request.Request(url, headers={"User-Agent": "WanGP-Deepy"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read(32 * 1024 * 1024 + 1)
        if len(archive) > 32 * 1024 * 1024:
            raise RuntimeError("The ripgrep download exceeded the 32 MiB safety limit.")
        actual_sha256 = hashlib.sha256(archive).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"The ripgrep archive checksum is invalid: expected {expected_sha256}, got {actual_sha256}.")
        if asset_name.endswith(".zip"):
            with zipfile.ZipFile(BytesIO(archive)) as bundle:
                members = [name for name in bundle.namelist() if Path(name).name.casefold() == executable_name.casefold()]
                if len(members) != 1:
                    raise RuntimeError("The ripgrep archive does not contain exactly one executable.")
                executable = bundle.read(members[0])
        else:
            with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as bundle:
                members = [member for member in bundle.getmembers() if member.isfile() and Path(member.name).name == executable_name]
                if len(members) != 1:
                    raise RuntimeError("The ripgrep archive does not contain exactly one executable.")
                extracted = bundle.extractfile(members[0])
                if extracted is None:
                    raise RuntimeError("The ripgrep executable could not be extracted.")
                executable = extracted.read()
        target_dir.mkdir(parents=True, exist_ok=True)
        temporary = target_dir / f".{executable_name}.{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(executable)
        if key[0] == "linux":
            temporary.chmod(0o755)
        os.replace(temporary, target)
        return target


def _parse_rg_arguments(arguments: str, *, experimental: bool = True) -> tuple[list[str], list[str]]:
    source = str(arguments or "").strip()
    if not source:
        raise ValueError("rg arguments are empty.")
    if len(source) > _MAX_RG_ARGUMENT_CHARS:
        raise ValueError(f"rg arguments exceed {_MAX_RG_ARGUMENT_CHARS} characters.")
    try:
        tokens = shlex.split(source, posix=True)
    except ValueError as exc:
        raise ValueError(f"Unable to parse rg arguments: {exc}") from exc
    if tokens and tokens[0].casefold() in {"rg", "rg.exe"}:
        raise ValueError("Pass only rg arguments, without the rg executable name.")
    separator = tokens.index("--") if "--" in tokens else len(tokens)
    search_tokens, path_tokens = tokens[:separator], tokens[separator + 1:] if separator < len(tokens) else []
    command, positionals, index, regexp_count = [], [], 0, 0
    while index < len(search_tokens):
        token = search_tokens[index]
        if token in _RG_FLAG_OPTIONS:
            command.append(token)
            index += 1
            continue
        option, has_equals, attached = token.partition("=")
        if option in _RG_VALUE_OPTIONS and has_equals:
            if not attached:
                raise ValueError(f"rg option {option} has an empty value.")
            command.append(token)
            regexp_count += option in {"-e", "--regexp"}
            index += 1
            continue
        if token in _RG_VALUE_OPTIONS:
            if index + 1 >= len(search_tokens):
                raise ValueError(f"rg option {token} requires a value.")
            command.extend((token, search_tokens[index + 1]))
            regexp_count += token in {"-e", "--regexp"}
            index += 2
            continue
        attached_option = next((option for option in _RG_ATTACHED_VALUE_OPTIONS if token.startswith(option) and token != option), None)
        if attached_option is not None:
            command.append(token)
            regexp_count += attached_option == "-e"
            index += 1
            continue
        if token.startswith("-"):
            context = "this experiment" if experimental else "this tool"
            raise ValueError(f"rg option is not enabled for {context}: {token}")
        positionals.append(token)
        command.append(token)
        index += 1
    files_mode = "--files" in command
    expected_positionals = 0 if files_mode or regexp_count else 1
    if len(positionals) != expected_positionals:
        if not experimental and expected_positionals == 1 and not positionals:
            raise ValueError("Text search requires a pattern before -- (or patterns with -e). To list filenames without searching content, use --files, e.g. --files -g '*.txt' -- @outputs. -l lists files containing matching text and still requires a pattern.")
        expectation = "no positional pattern" if expected_positionals == 0 else "exactly one positional pattern"
        raise ValueError(f"This rg wrapper requires {expectation} before `--`; put every search path after `--`.")
    return command, path_tokens


def run_rg(policy, arguments: str) -> dict[str, Any]:
    if not long_text_tools_active(policy):
        raise PermissionError("The experimental rg tool requires Deepy read/write filesystem access.")
    command, path_tokens = _parse_rg_arguments(arguments)
    if not path_tokens:
        mount = workspace_mount(policy)
        if mount is None:
            raise RuntimeError("The Deepy temporary workspace is unavailable.")
        path_tokens = [f"@{mount[0]}"]
    resolved_paths = []
    for path in path_tokens:
        resolved = policy.require_read(path)
        if not policy.can_write(resolved):
            raise PermissionError(f"rg may search only configured read/write roots: {policy.virtualize_path(resolved)}")
        resolved_paths.append(resolved)
    invocation = [str(ensure_ripgrep()), "--no-config", "--color=never", *command, "--", *map(str, resolved_paths)]
    environment = dict(os.environ)
    environment.pop("RIPGREP_CONFIG_PATH", None)
    process = subprocess.Popen(invocation, cwd=str(resolved_paths[0] if resolved_paths[0].is_dir() else resolved_paths[0].parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, shell=False)
    chunks, output_bytes, truncated = [], 0, threading.Event()

    def read_output() -> None:
        nonlocal output_bytes
        assert process.stdout is not None
        while chunk := process.stdout.read(8192):
            remaining = _MAX_RG_OUTPUT_CHARS - output_bytes
            if remaining > 0:
                chunks.append(chunk[:remaining])
                output_bytes += min(len(chunk), remaining)
            if len(chunk) > remaining:
                truncated.set()
                if process.poll() is None:
                    process.terminate()
                break

    reader = threading.Thread(target=read_output, daemon=True, name="deepy-rg-output")
    reader.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        return_code = process.wait(timeout=5)
    reader.join(timeout=5)
    output = b"".join(chunks).decode("utf-8", errors="replace")
    if truncated.is_set():
        output += "\n[rg output truncated]"
    elif timed_out:
        output += "\n[rg search timed out after 30 seconds]"
    status = "truncated" if truncated.is_set() else "error" if timed_out else "done" if return_code == 0 else "no_matches" if return_code == 1 else "error"
    result = {
        "status": status,
        "exit_code": return_code,
        "arguments": str(arguments or "").strip(),
        "output": output,
        "truncated": truncated.is_set(),
        "output_characters": len(output),
    }
    return policy.virtualize_result(result)


def _decode_text(data: bytes, label: str) -> tuple[str, str]:
    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        return data.decode(encoding), encoding
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 text.") from exc


def _closest_passage(original: str, old_string: str) -> str:
    """Where an old_string quoted from memory (another indentation, a value changed since) most resembles the file, as the text to quote instead."""
    lines, wanted = original.splitlines(), old_string.replace("\r\n", "\n")
    size, best, where = wanted.count("\n") + 1, 0.0, -1
    for index in range(max(1, len(lines) - size + 1)):
        matcher = difflib.SequenceMatcher(None, wanted, "\n".join(lines[index:index + size]), autojunk=False)
        if matcher.real_quick_ratio() > best and matcher.quick_ratio() > best and matcher.ratio() > best:
            best, where = matcher.ratio(), index
    if best < 0.6:
        return ""
    return f" The closest passage, at line {where + 1}, reads: {chr(10).join(lines[where:where + size])[:600]!r}. Copy it exactly into old_string."


def _text_argument(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value and not allow_empty:
        raise ValueError(f"{name} is empty.")
    if len(value) > _MAX_TEXT_EDIT_CHARS:
        raise ValueError(f"{name} exceeds the {_MAX_TEXT_EDIT_CHARS}-character limit.")
    return value


def _atomic_write(target: Path, data: bytes, *, create_only: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(data)
        if create_only:
            os.link(temporary, target)
        else:
            os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def edit_text(policy, file_path: str, old_string: str = "", new_string: str = "", replace_all: bool = False, edits: list[dict[str, str]] | None = None) -> dict[str, Any]:
    if not long_text_tools_active(policy):
        raise PermissionError("The experimental edit tool requires Deepy read/write filesystem access.")
    where = (lambda index: f"edits[{index}]: ") if edits else (lambda index: "")
    pairs = [(item.get("old_string"), item.get("new_string")) for item in edits] if edits else [(old_string, new_string)]
    pairs = [(_text_argument(old, "old_string"), _text_argument(new, "new_string", allow_empty=True)) for old, new in pairs]
    for index, (old, new) in enumerate(pairs):
        if old == new:
            raise ValueError(f"{where(index)}old_string and new_string are identical.")
    with _RUNTIME_LOCK:
        target = policy.require_read(file_path, file=True)
        policy.require_write(target)
        original, encoding = _decode_text(target.read_bytes(), policy.virtualize_path(target))
        crlf = "\r\n" in original and "\n" not in original.replace("\r\n", "")  # read_text shows a CRLF file with plain newlines
        updated, replacements = original, 0
        for index, (old, new) in enumerate(pairs):  # in order, on the text the previous ones left; written once, all or none
            if crlf:
                old, new = (value.replace("\r\n", "\n").replace("\n", "\r\n") for value in (old, new))
            occurrences = updated.count(old)
            if occurrences == 0:
                raise ValueError(f"{where(index)}old_string was not found exactly in the target file. Use rg and preserve whitespace and line endings." + _closest_passage(updated, old)
                                 + (" Nothing was changed." if edits else ""))
            if occurrences > 1 and not replace_all:
                raise ValueError(f"{where(index)}old_string occurs {occurrences} times; include more surrounding context or set replace_all=true.")
            replacements += occurrences if replace_all else 1
            updated = updated.replace(old, new, -1 if replace_all else 1)
        data = updated.encode(encoding)
        _atomic_write(target, data, create_only=policy.create_only(target))
    return {"status": "done", "action": "edit", "path": policy.virtualize_path(target), "replacements": replacements, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def append_text(policy, file_path: str, text: str, *, include_structure: bool = False) -> dict[str, Any]:
    if not long_text_tools_active(policy):
        raise PermissionError("The experimental append_text tool requires Deepy read/write filesystem access.")
    text = _text_argument(text, "text")
    with _RUNTIME_LOCK:
        target = policy.require_write(file_path)
        created = not target.exists()
        if created:
            original, encoding = "", "utf-8"
            target.parent.mkdir(parents=True, exist_ok=True)  # inside the authorized write area, like the file itself
        else:
            target = policy.require_read(target, file=True)
            original, encoding = _decode_text(target.read_bytes(), policy.virtualize_path(target))
        data = (original + text).encode(encoding)
        _atomic_write(target, data, create_only=policy.create_only(target))
    result = {"status": "done", "action": "create" if created else "append", "path": policy.virtualize_path(target), "created": created, "appended_characters": len(text), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    heading_pattern = r"^#{1,6}[ \t]+\S[^\r\n]*"
    if include_structure and re.search(heading_pattern, text, re.MULTILINE):
        headings = list(re.finditer(heading_pattern, original + text, re.MULTILINE))
        last_heading = headings[-1].group() if headings else None
        result["structure"] = {"headings_added": sum(match.start() >= len(original) for match in headings), "heading_count": len(headings), "last_heading": last_heading, "last_heading_occurrences": sum(match.group() == last_heading for match in headings)}
    return result


def resolve_prompt_file(value: Any, policy, *, read_only: bool = False) -> Any:
    if not isinstance(value, str) or not (read_only or long_text_tools_active(policy)):
        return value
    match = _PROMPT_FILE_RE.fullmatch(value)
    if match is None:
        return value
    target = policy.require_read(match.group(1), file=True)
    if not (read_only or getattr(policy, "prompt_read_only", False)) and not policy.can_write(target):
        raise PermissionError(f"Prompt files must be inside a configured read/write root: {policy.virtualize_path(target)}")
    size = target.stat().st_size
    if size > _MAX_PROMPT_FILE_BYTES:
        raise ValueError(f"Prompt file exceeds the {_MAX_PROMPT_FILE_BYTES}-byte limit: {policy.virtualize_path(target)}")
    content, _encoding = _decode_text(target.read_bytes(), policy.virtualize_path(target))
    if not content.strip():
        raise ValueError(f"Prompt file is empty: {policy.virtualize_path(target)}")
    return content


def resolve_prompt_references(value: Any, policy, key: str = "", *, read_only: bool = False) -> Any:
    if isinstance(value, list):
        return [resolve_prompt_references(item, policy, key, read_only=read_only) for item in value]
    if isinstance(value, dict):
        return {child_key: resolve_prompt_references(child_value, policy, str(child_key).casefold(), read_only=read_only) for child_key, child_value in value.items()}
    return resolve_prompt_file(value, policy, read_only=read_only) if key in _PROMPT_FIELDS else value


__all__ = [
    "add_session_workspace", "append_text", "edit_text", "ensure_ripgrep",
    "long_text_system_instructions", "long_text_tools_active", "resolve_prompt_file", "resolve_prompt_references", "run_rg", "workspace_mount",
]
