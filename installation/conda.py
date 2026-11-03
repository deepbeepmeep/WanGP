"""Discover existing Conda Python environments without modifying Conda state."""

import os
from pathlib import Path
import shutil

from .environments import clean_environment, interpreter_path
from .runtime import EnvironmentService
from .state import InstallError, digest, file_hash, plain_path


def discover_conda_environments(home=None, environ=None, check_cancelled=None):
    home = Path.home() if home is None else Path(home)
    environ = os.environ if environ is None else environ
    check_cancelled = check_cancelled or (lambda: None)
    prefixes, roots, env_dirs, warnings = set(), set(), {home / ".conda/envs"}, []

    def add_prefix(value):
        if not value or "\0" in str(value):
            return
        path = Path(value).expanduser()
        if not path.is_absolute():
            return
        prefixes.add(path)
        if path.parent.name == "envs":
            roots.add(path.parent.parent)
        if (path / "condabin").is_dir() or (path / "envs").is_dir():
            roots.add(path)

    # Conda's env-list command can clean this registry. Read it directly instead.
    registry = home / ".conda/environments.txt"
    try:
        for line in registry.read_text(encoding="utf-8-sig").splitlines():
            check_cancelled()
            add_prefix(line.strip())
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError) as error:
        warnings.append(f"Could not read Conda's environment registry: {error}")

    for key in ("CONDA_PREFIX", "CONDA_ROOT"):
        add_prefix(environ.get(key))
    executable = environ.get("CONDA_EXE") or shutil.which("conda", path=environ.get("PATH", environ.get("Path", "")))
    if executable:
        parent = Path(executable).parent
        roots.add(parent.parent if parent.name.lower() in ("scripts", "bin", "condabin") else parent)
    for value in environ.get("CONDA_ENVS_PATH", "").split(os.pathsep):
        if value and Path(value).is_absolute():
            env_dirs.add(Path(value))

    locations = [home, home / "opt"]
    if os.name == "nt":
        locations += [Path(environ[key]) for key in ("LOCALAPPDATA", "ProgramData", "PROGRAMDATA") if environ.get(key)]
        if environ.get("LOCALAPPDATA"):
            locations.append(Path(environ["LOCALAPPDATA"]) / "Continuum")
    else:
        locations += [Path("/opt"), Path("/usr/local")]
    for location in locations:
        for name in ("anaconda3", "miniconda3", "miniforge3", "mambaforge"):
            roots.add(location / name)
    for root in roots:
        check_cancelled()
        if (root / "conda-meta").is_dir():
            prefixes.add(root)
            env_dirs.add(root / "envs")
    for directory in sorted(env_dirs):
        check_cancelled()
        try:
            prefixes.update(directory.iterdir())
        except FileNotFoundError:
            pass
        except OSError as error:
            warnings.append(f"Could not read Conda environments in {directory}: {error}")

    environments = {}
    for prefix in sorted(prefixes):
        check_cancelled()
        if not (prefix / "conda-meta").is_dir():
            continue
        python = Path(interpreter_path(prefix, "conda"))
        if not python.is_file():
            continue
        prefix = prefix.resolve()
        key = os.path.normcase(str(prefix))
        name = "base" if (prefix / "condabin").is_dir() or (prefix / "envs").is_dir() else prefix.name
        environments[key] = {"name": name, "prefix": str(prefix), "python": str(python.resolve())}
    return {"environments": sorted(environments.values(), key=lambda item: (item["name"].casefold(), item["prefix"].casefold())), "warnings": warnings}


def conda_command(prefix):
    """Find the owning Conda installation, including environments at custom prefixes."""
    prefix = plain_path(prefix)
    roots = [prefix.parent.parent] if prefix.parent.name == "envs" else []
    roots += [Path(item["prefix"]) for item in discover_conda_environments()["environments"] if item["name"] == "base"]
    for root in dict.fromkeys(roots):
        executable = root / ("Scripts/conda.exe" if os.name == "nt" else "bin/conda")
        if executable.is_file():
            return [str(executable)]
    raise InstallError("conda_unavailable", "Conda could not be found. Install Conda or make its executable available on PATH before updating this installation")


class CondaEnvironmentService(EnvironmentService):
    def __init__(self, runtime, runner=None, cache_directory=None):
        super().__init__(runtime, runner, cache_directory, kind="conda")

    def snapshot(self, environment):
        environment = plain_path(environment)
        inventory = self.inventory(environment)
        metadata = {path.name: file_hash(path) for path in sorted((environment / "conda-meta").iterdir()) if path.is_file()}
        return {"inventory": inventory, "conda_metadata_sha256": digest(metadata)}

    def clone(self, source, destination):
        source, destination = plain_path(source), plain_path(destination)
        if destination.exists() or source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
            raise InstallError("invalid_environment", "The Conda clone needs a new, separate directory")
        if (source / "condabin").is_dir():
            raise InstallError("conda_base", "Choose a dedicated Conda environment; the base environment cannot be used for Manager updates")
        before = self.snapshot(source)
        command = conda_command(source)
        env = clean_environment()
        env.update(CONDA_REGISTER_ENVS="false", CONDA_ALWAYS_COPY="true", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.runner.events({"type": "progress", "title": "Cloning Conda Environment", "message": "Copying Python and installed packages. The current environment remains unchanged."})
        self.runner.run([*command, "create", "--yes", "--copy", "--clone", str(source), "--prefix", str(destination)], env=env, timeout=None)
        if self.snapshot(source) != before:
            raise InstallError("environment_changed", "The original Conda environment changed while it was being cloned")
        cloned = self.inventory(destination)
        if cloned["python"] != before["inventory"]["python"] or cloned["packages"] != before["inventory"]["packages"]:
            raise InstallError("clone_mismatch", "Conda did not preserve the installed package versions. The current installation was left unchanged")
        return command

    def update_clone(self, source, destination, resolution_directory, resolution):
        command = self.clone(source, destination)
        python = interpreter_path(destination, "conda")
        if self.inventory(destination)["python"] != resolution["python"]:
            env = clean_environment()
            env.update(CONDA_REGISTER_ENVS="false", CONDA_ALWAYS_COPY="true", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
            self.runner.events({"type": "progress", "title": "Updating Cloned Python", "message": "Installing Python " + resolution["python"]})
            self.runner.run([*command, "install", "--yes", "--copy", "--prefix", str(destination), "python=" + resolution["python"]], env=env)
        lock = plain_path(resolution_directory) / "pylock.toml"
        if file_hash(lock) != resolution["files"]["pylock.toml"]:
            raise InstallError("changed_input", "Dependency lock changed before installation")
        self.runner.events({"type": "progress", "title": "Updating Cloned Dependencies", "message": "Installing the selected version's requirements and optional kernels with uv"})
        arguments = self.runtime.command("pip", "install", "--python", python, "--require-hashes", "--link-mode", "copy", "-r", str(lock))
        if resolution.get("no_index", False):
            arguments.append("--no-index")
        env = self.runtime.environment(python)
        if self.cache_directory:
            env["UV_CACHE_DIR"] = str(self.cache_directory)
        self.runner.run(arguments, env=env)
        self.runner.events({"type": "progress", "title": "Validating Conda Environment", "message": "Checking package compatibility and GPU execution"})
        return self.validate(destination, resolution)
