"""Resolve one combined, hash-locked dependency set without modifying an install."""

import hashlib
import json
import platform
import re
import shlex
from urllib.parse import unquote, urlsplit

from .catalog import catalog_digest, load_config, os_key
from .execute import ProcessRunner
from .state import InstallError, atomic_json, digest, file_hash, inside, plain_path, read_json


def package_name(requirement):
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if not match:
        raise InstallError("invalid_requirement", f"Cannot identify requirement: {requirement}")
    return re.sub(r"[-_.]+", "-", match[1]).lower()


def source_url(value):
    parts = urlsplit(value)
    if parts.scheme not in ("https", "file") or parts.username is not None or parts.password is not None:
        raise InstallError("unsupported_source", "Package sources must be HTTPS or explicit local files, without embedded credentials")
    return value


class DependencyInputs:
    def __init__(self, root):
        self.root = plain_path(root)
        self.requirements = []
        self.constraints = []
        self.files = {}
        self.indexes = []
        self.default_index = None
        self.find_links = []
        self.no_index = False
        self.torch_index = None
        self.torch_packages = []

    def add_index(self, option, url):
        url = source_url(url)
        if option in ("--index-url", "--default-index"):
            if self.default_index is not None and self.default_index != url:
                raise InstallError("conflicting_indexes", "Selected requirements disagree on the default package index")
            self.default_index = url
        else:
            self.indexes.append(url)

    def add_requirement(self, value, constraint=False):
        value = value.strip()
        if value.startswith(("https://", "file://")):
            source_url(value)
            filename = unquote(urlsplit(value).path.rsplit("/", 1)[-1])
            if not filename.endswith(".whl"):
                raise InstallError("unsupported_source", "Direct catalog sources must be prepared wheels")
            value = f"{package_name(filename.split('-')[0])} @ {value}"
        if "git+" in value or value.startswith((".", "/", "\\")):
            raise InstallError("source_build_required", f"Prepare a versioned wheel for this source-build requirement before locking: {value}")
        if " @ " in value:
            source_url(value.split(" @ ", 1)[1].split(" ;", 1)[0].strip())
        package_name(value)
        (self.constraints if constraint else self.requirements).append(value)

    def read_requirements(self, path, constraint=False, stack=()):
        path = plain_path(path)
        if path in stack:
            raise InstallError("requirements_cycle", f"Recursive requirement include: {path}")
        raw_content = path.read_bytes()
        content = raw_content.decode("utf-8-sig")
        self.files[str(path)] = hashlib.sha256(raw_content).hexdigest()
        content = content.replace("\\\r\n", "").replace("\\\n", "")
        for raw in content.splitlines():
            line = re.split(r"\s+#", raw.strip(), maxsplit=1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                tokens = shlex.split(line)
                option, separator, embedded = tokens[0].partition("=")
                value = embedded if separator else tokens[1] if len(tokens) == 2 else None
                if option in ("-r", "--requirement", "-c", "--constraint") and value:
                    child = plain_path(path.parent / value)
                    if not child.is_relative_to(self.root):
                        raise InstallError("unsafe_path", f"Requirement include leaves the source root: {child}")
                    self.read_requirements(child, constraint or option in ("-c", "--constraint"), (*stack, path))
                elif option in ("--index-url", "--extra-index-url", "--index", "--default-index") and value:
                    self.add_index(option, value)
                elif option in ("--find-links", "-f") and value:
                    self.find_links.append(source_url(value) if "://" in value else str(plain_path(path.parent / value)))
                elif option == "--no-index" and len(tokens) == 1:
                    self.no_index = True
                else:
                    raise InstallError("unsupported_requirement_option", f"Unsupported option in {path}: {line}")
            else:
                self.add_requirement(line, constraint)

    def add_step(self, step):
        arguments = iter(step["args"])
        is_torch = step["id"].startswith("torch.")
        for argument in arguments:
            if argument in ("--index-url", "--extra-index-url", "--index", "--default-index"):
                url = source_url(next(arguments))
                if is_torch:
                    if self.torch_index and self.torch_index != url:
                        raise InstallError("conflicting_indexes", "The selected Torch recipe declares multiple package indexes")
                    self.torch_index = url
                else:
                    self.add_index(argument, url)
            elif argument in ("-r", "--requirement"):
                self.read_requirements(next(arguments))
            elif argument in ("-U", "--upgrade", "--no-deps"):
                # The combined graph still validates dependencies of selected wheels.
                continue
            elif argument == "--no-build-isolation" or "git+" in argument or step["environment"]:
                raise InstallError("source_build_required", f"{step['id']} needs a reviewed prebuilt wheel before transactional installation")
            elif argument.startswith("-"):
                raise InstallError("unsupported_requirement_option", f"Unsupported catalog option: {argument}")
            else:
                self.add_requirement(argument)
                if is_torch:
                    self.torch_packages.append(package_name(self.requirements[-1]))

    def index_configuration(self):
        quote = lambda value: json.dumps(value, ensure_ascii=True)
        lines = []
        if self.constraints:
            lines.append("constraint-dependencies = [" + ", ".join(map(quote, self.constraints)) + "]")
        if self.torch_index:
            lines.extend(["[[index]]", 'name = "wangp-torch"', f"url = {quote(self.torch_index)}", "explicit = true"])
        for index, url in enumerate(dict.fromkeys(self.indexes)):
            lines.extend(["[[index]]", f'name = "wangp-extra-{index}"', f"url = {quote(url)}"])
        if self.default_index:
            lines.extend(["[[index]]", 'name = "wangp-default"', f"url = {quote(self.default_index)}", "default = true"])
        return "\n".join(lines) + "\n"

    def project(self, python):
        quote = lambda value: json.dumps(value, ensure_ascii=True)
        lines = ["[project]", 'name = "wangp-managed-environment"', 'version = "0"', f"requires-python = {quote('==' + python)}", "dependencies = ["]
        lines.extend("  " + quote(value) + "," for value in dict.fromkeys(self.requirements))
        lines.append("]")
        if self.constraints:
            lines.extend(["[tool.uv]", "constraint-dependencies = [" + ", ".join(map(quote, self.constraints)) + "]"])
        if self.torch_index:
            lines.append("[tool.uv.sources]")
            lines.extend(f'{quote(name)} = {{ index = "wangp-torch" }}' for name in dict.fromkeys(self.torch_packages))
            lines.extend(["[[tool.uv.index]]", 'name = "wangp-torch"', f"url = {quote(self.torch_index)}", "explicit = true"])
        for index, url in enumerate(dict.fromkeys(self.indexes)):
            lines.extend(["[[tool.uv.index]]", f'name = "wangp-extra-{index}"', f"url = {quote(url)}"])
        if self.default_index:
            lines.extend(["[[tool.uv.index]]", 'name = "wangp-default"', f"url = {quote(self.default_index)}", "default = true"])
        return "\n".join(lines) + "\n"


def collect_inputs(plan, extra_requirements=()):
    root = plain_path(plan["root"])
    if plan["plan_sha256"] != digest({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise InstallError("changed_plan", "The catalog plan was modified")
    if plan["blockers"]:
        raise InstallError("blocked_plan", "; ".join(plan["blockers"]))
    config = root / "setup_config.json"
    if catalog_digest(load_config(config)) != plan["catalog_sha256"]:
        raise InstallError("stale_plan", "The setup catalog changed after planning")
    for relative, expected in plan["requirements_sha256"].items():
        if file_hash(inside(root, relative.replace("\\", "/"))) != expected:
            raise InstallError("stale_plan", f"Requirements changed after planning: {relative}")
    inputs = DependencyInputs(root)
    inputs.files[str(config)] = file_hash(config)
    for step in plan["steps"]:
        if step["kind"] == "install_packages":
            inputs.add_step(step)
    for path in extra_requirements:
        # Explicit external plugin inputs may include siblings inside their own folder.
        original_root = inputs.root
        inputs.root = plain_path(path).parent
        try:
            inputs.read_requirements(path)
        finally:
            inputs.root = original_root
    return inputs


def resolve_dependencies(plan, destination, runtime, *, extra_requirements=(), runner=None):
    if plan["platform"] != os_key():
        raise InstallError("platform_mismatch", "Resolve on the target operating system")
    inputs = collect_inputs(plan, extra_requirements)
    destination = plain_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    project = destination / "pyproject.toml"
    project.write_text(inputs.project(plan["python"]), encoding="utf-8")
    configuration = destination / "uv.toml"
    configuration.write_text(inputs.index_configuration(), encoding="utf-8")
    output = destination / "pylock.toml"
    runner = runner or ProcessRunner(destination / "resolve.log")
    arguments = runtime.command("pip", "compile", str(project), "--python-version", plan["python"], "--format", "pylock.toml", "--generate-hashes", "--no-header", "--index-strategy", "first-index", "--output-file", str(output))
    # An explicit config file excludes user/ancestor config while retaining our named indexes.
    arguments[arguments.index("--no-config"):arguments.index("--no-config") + 1] = ["--config-file", str(configuration)]
    for location in dict.fromkeys(inputs.find_links):
        arguments.extend(("--find-links", location))
    if inputs.no_index:
        arguments.append("--no-index")
    runner.run(arguments, env=runtime.environment(), cwd=destination)
    for path, expected in inputs.files.items():
        if file_hash(path) != expected:
            raise InstallError("changed_input", f"Dependency input changed during resolution: {path}")
    lock_hash = file_hash(output)
    result = {
        "schema_version": 1, "python": plan["python"], "platform": plan["platform"], "machine": platform.machine(),
        "profile": plan["profile"], "amd_targets": plan["amd_targets"], "selected_device": plan["selected_device"],
        "plan_sha256": plan["plan_sha256"], "catalog_sha256": plan["catalog_sha256"],
        "input_files": inputs.files, "source_root": plan["root"],
        "files": {"pylock.toml": lock_hash, "pyproject.toml": file_hash(project), "uv.toml": file_hash(configuration)},
        "dependency_fingerprint": digest({"lock": lock_hash, "python": plan["python"], "platform": plan["platform"], "machine": platform.machine()}),
        "extra_requirements": list(map(str, extra_requirements)), "trust": "explicit_local_inputs",
        "no_index": inputs.no_index,
        "requires_gpu": any(package_name(value) == "torch" for value in inputs.requirements),
    }
    result["resolution_sha256"] = digest(result)
    atomic_json(destination / "resolution.json", result)
    return result


def load_resolution(directory, check_inputs=True):
    directory = plain_path(directory)
    record = read_json(directory / "resolution.json")
    if record["schema_version"] != 1 or record["resolution_sha256"] != digest({key: value for key, value in record.items() if key != "resolution_sha256"}):
        raise InstallError("invalid_resolution", "Invalid dependency resolution record")
    if record["platform"] != os_key() or record["machine"] != platform.machine():
        raise InstallError("platform_mismatch", "Dependency resolution belongs to another host platform")
    for relative, expected in record["files"].items():
        if file_hash(inside(directory, relative)) != expected:
            raise InstallError("changed_input", f"Resolved dependency artifact changed: {relative}")
    if check_inputs:
        for filename, expected in record["input_files"].items():
            if file_hash(plain_path(filename)) != expected:
                raise InstallError("stale_plan", f"Dependency input changed: {filename}")
    return record
