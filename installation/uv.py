"""Explicit, isolated uv commands. No generation packages are imported."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .environments import clean_environment, inspect_interpreter, interpreter_path


def default_data_directory():
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "WanGP" / "Manager"


class UvRuntime:
    def __init__(self, executable="uv", data_directory=None, offline=False, base_python=None):
        self.executable = str(executable)
        self.data_directory = Path(data_directory or default_data_directory()).resolve()
        self.offline = offline
        self.base_python = str(base_python) if base_python else None

    def environment(self, target=None, base=None):
        env = clean_environment(base)
        env.update(UV_CACHE_DIR=str(self.data_directory / "cache" / "uv"), UV_PYTHON_INSTALL_DIR=str(self.data_directory / "python"), UV_PYTHON_BIN_DIR=str(self.data_directory / "bin"), UV_PYTHON_INSTALL_REGISTRY="0", UV_PYTHON_INSTALL_BIN="0")
        if target is not None:
            path_key = next((key for key in env if key.upper() == "PATH"), "PATH")
            prefix = Path(target).parent
            paths = [str(prefix)]
            if os.name == "nt" and (prefix / "conda-meta").is_dir():
                paths += [str(prefix / "Scripts"), str(prefix / "Library/bin")]
            env[path_key] = os.pathsep.join(paths) + os.pathsep + env.get(path_key, "")
        return env

    def command(self, *arguments):
        return [self.executable, "--no-config", "--no-progress", *(["--offline"] if self.offline else []), *map(str, arguments)]

    def create_command(self, directory, python, seed=True):
        if self.base_python and re.fullmatch(r"\d+\.\d+\.\d+", str(python)):
            base = inspect_interpreter(self.base_python)
            if base["status"] != "inspected":
                raise ValueError("The bundled management Python could not be inspected")
            if base["python"] == str(python):
                python = self.base_python
        args = ["venv", "--allow-existing", "--prompt", "WanGP", "--python", str(python)]
        if re.fullmatch(r"\d+\.\d+\.\d+", str(python)):
            args.append("--managed-python")
        elif not Path(python).is_absolute():
            raise ValueError("Python must be an exact version or an absolute interpreter path")
        if seed:
            args.append("--seed")
        return self.command(*args, str(Path(directory).resolve()))

    def install_command(self, interpreter, arguments, index_strategy="first-index"):
        if index_strategy not in ("first-index", "unsafe-best-match"):
            raise ValueError(f"Unsupported index strategy: {index_strategy}")
        interpreter = Path(interpreter)
        if not interpreter.is_absolute():
            raise ValueError("An absolute target interpreter is required")
        forbidden = {"--python", "-p", "--target", "--prefix", "--system", "--break-system-packages", "--config-file", "--directory", "--project", "--index-strategy"}
        for argument in arguments:
            option = argument.split("=", 1)[0]
            if option in forbidden or argument.startswith("-p") and not argument.startswith("--"):
                raise ValueError(f"Package arguments cannot redirect the installer: {argument}")
        return self.command("pip", "install", "--python", str(interpreter), "--index-strategy", index_strategy, *arguments)

    def _executable(self, environment):
        executable = shutil.which(self.executable, path=environment.get("PATH", environment.get("Path", "")))
        if executable is None:
            raise FileNotFoundError(f"uv executable not found: {self.executable}. Supply --uv with an installed executable.")
        return executable

    def _run(self, command, environment, cwd=None):
        return subprocess.run([self._executable(environment), *command[1:]], env=environment, cwd=cwd, check=True)

    def create_environment(self, directory, python, seed=True):
        directory = Path(directory).resolve()
        command = self.create_command(directory, python, seed)
        environment = self.environment()
        self._executable(environment)
        # Reserve a fresh final path; never ask uv to clear an existing directory.
        directory.mkdir(parents=True, exist_ok=False)
        self._run(command, environment)
        interpreter = interpreter_path(directory)
        result = inspect_interpreter(interpreter)
        if result["status"] != "inspected" or Path(result["prefix"]).resolve() != directory:
            raise RuntimeError(f"Created environment failed interpreter validation: {result}")
        return result

    def install_packages(self, interpreter, arguments, *, index_strategy="first-index", environment=None, cwd=None):
        target = Path(interpreter).resolve()
        result = inspect_interpreter(target)
        if result["status"] != "inspected":
            raise ValueError(f"Cannot inspect target interpreter: {result['error']}")
        if result["prefix"] == result["base_prefix"]:
            raise ValueError("uv package installation requires an explicit virtual environment")
        env = self.environment(target)
        for key, value in (environment or {}).items():
            if key != "FLASH_ATTENTION_TRITON_AMD_ENABLE":
                raise ValueError(f"Unsupported component environment variable: {key}")
            env[key] = value
        return self._run(self.install_command(target, arguments, index_strategy), env, cwd)
