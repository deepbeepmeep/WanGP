import os
import sys
import importlib
import inspect
import re
import datetime
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, replace
import gradio as gr
import traceback
import subprocess
import git
import shutil
import stat
import json
from shared.utils.config_store import config_lock, write_config
import requests
from shared.utils.wgp_config_migration import INSTALLED_REMOTE_PLUGINS_KEY, migrate_bundled_plugin_ids
media_gen_label = "Media Generator"

COMMUNITY_PLUGINS_URL = "https://github.com/deepbeepmeep/Wan2GP/raw/refs/heads/main/plugins.json"
PLUGIN_CATALOG_FILENAME = "plugins.json"
PLUGIN_LOCAL_CATALOG_FILENAME = "plugins_local.json"
PLUGIN_METADATA_FILENAME = "plugin_info.json"
PLUGIN_TYPE_CHOICES = ("app", "extension", "processor", "model")
PLUGIN_SPATIAL_UPSAMPLER_HANDLERS_KEY = "spatial_upsampler_handlers"
PLUGIN_TEMPORAL_UPSAMPLER_HANDLERS_KEY = "temporal_upsampler_handlers"
PLUGIN_AUDIO_PROCESSORS_KEY = "audio_processors"
PLUGIN_MODEL_HANDLERS_KEY = "model_handlers"
PLUGIN_MODEL_DEFAULTS_KEY = "defaults"
PLUGIN_MODEL_PROFILES_KEY = "profiles"
PENDING_DELETIONS_KEY = "pending_plugin_deletions"
_DEEPY_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEEPY_PRIME_RESERVED_TOOL_NAMES = {"mcp_resource"}


@dataclass
class PluginDeepyTool:
    name: str
    function: callable
    display_name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    pause_runtime: bool
    pause_reason: str
    requires_file_system: bool
    plugin_id: str = ""

    def assistant_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "parameters": self.parameters,
            "pause_runtime": self.pause_runtime,
            "pause_reason": self.pause_reason,
            "requires_file_system": self.requires_file_system,
        }


_ACTIVE_DEEPY_PRIME_TOOLS: Dict[str, PluginDeepyTool] = {}
_ACTIVE_DEEPY_ZERO_TOOLS: Dict[str, PluginDeepyTool] = {}


def get_deepy_prime_plugin_tools() -> tuple[PluginDeepyTool, ...]:
    return tuple(_ACTIVE_DEEPY_PRIME_TOOLS.values())


def get_deepy_zero_plugin_tools() -> tuple[PluginDeepyTool, ...]:
    return tuple(_ACTIVE_DEEPY_ZERO_TOOLS.values())


def _reset_deepy_plugin_tools() -> None:
    _ACTIVE_DEEPY_PRIME_TOOLS.clear()
    _ACTIVE_DEEPY_ZERO_TOOLS.clear()

def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True

def _split_github_repo(url: str) -> Optional[tuple]:
    if not isinstance(url, str):
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]
    if cleaned.startswith("git@github.com:"):
        cleaned = "https://github.com/" + cleaned[len("git@github.com:"):]
    cleaned = cleaned.rstrip("/")
    marker = "github.com/"
    idx = cleaned.lower().find(marker)
    if idx < 0:
        return None
    tail = cleaned[idx + len(marker):]
    parts = [part for part in tail.split("/") if part]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return owner, repo

def normalize_plugin_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    cleaned = url.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]
    if cleaned.startswith("git@github.com:"):
        cleaned = "https://github.com/" + cleaned[len("git@github.com:"):]
    cleaned = cleaned.rstrip("/")
    repo_info = _split_github_repo(cleaned)
    if repo_info:
        owner, repo = repo_info
        return f"https://github.com/{owner}/{repo}"
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    return cleaned.rstrip("/")

def _parse_version_parts(version: str) -> List[Any]:
    if not isinstance(version, str):
        return []
    version = version.strip()
    if not version:
        return []
    parts = re.split(r"[^0-9A-Za-z]+", version)
    tokens = []
    for part in parts:
        if not part:
            continue
        for token in re.findall(r"\d+|[A-Za-z]+", part):
            if token.isdigit():
                tokens.append((0, int(token)))
            else:
                tokens.append((1, token.lower()))
    return tokens

def compare_versions(left: str, right: str) -> int:
    left_text = left if isinstance(left, str) else ""
    right_text = right if isinstance(right, str) else ""
    left_has_digits = bool(re.search(r"\d", left_text))
    right_has_digits = bool(re.search(r"\d", right_text))
    if left_has_digits != right_has_digits:
        return 1 if left_has_digits else -1
    left_parts = _parse_version_parts(left_text)
    right_parts = _parse_version_parts(right_text)
    max_len = max(len(left_parts), len(right_parts))
    if max_len == 0:
        return 0
    filler = (0, 0)
    for idx in range(max_len):
        left_part = left_parts[idx] if idx < len(left_parts) else filler
        right_part = right_parts[idx] if idx < len(right_parts) else filler
        if left_part == right_part:
            continue
        return 1 if left_part > right_part else -1
    return 0

def _parse_date(value: Any) -> Optional[datetime.datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None

def compare_release_metadata(left: Dict[str, Any], right: Dict[str, Any]) -> int:
    left_date = _parse_date(left.get("date"))
    right_date = _parse_date(right.get("date"))
    if left_date or right_date:
        if left_date and right_date:
            if left_date != right_date:
                return 1 if left_date > right_date else -1
        elif left_date:
            return 1
        else:
            return -1
    return compare_versions(left.get("version", ""), right.get("version", ""))

def is_wangp_compatible(required_version: str, current_version: str) -> bool:
    if not _has_value(required_version):
        return True
    return compare_versions(current_version or "", required_version) >= 0

def plugin_id_from_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    repo_info = _split_github_repo(url)
    if repo_info:
        return repo_info[1]
    clean = normalize_plugin_url(url)
    if not clean:
        return ""
    return clean.split("/")[-1]

def normalize_plugin_types(value: Any) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    elif value is None:
        values = []
    else:
        values = [str(value)]
    cleaned = []
    seen = set()
    for item in values:
        if not isinstance(item, str):
            item = str(item)
        item = item.strip().lower()
        if not item or item not in PLUGIN_TYPE_CHOICES or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned or ["app"]

def merge_plugin_types(primary: Any, secondary: Any) -> List[str]:
    primary_types = normalize_plugin_types(primary)
    secondary_types = normalize_plugin_types(secondary)
    if primary_types == ["app"] and secondary_types != ["app"]:
        return secondary_types
    if secondary_types == ["app"]:
        return primary_types
    merged = []
    for item in primary_types + secondary_types:
        if item not in merged:
            merged.append(item)
    return merged or ["app"]

def auto_install_and_enable_default_plugins(manager: 'PluginManager', wgp_globals: dict):
    server_config = wgp_globals.get("server_config")
    server_config_filename = wgp_globals.get("server_config_filename")

    if not server_config or not server_config_filename:
        print("[Plugins] WARNING: Cannot auto-install/enable default plugins. Server config not found.")
        return

    default_plugins = {}
    
    config_modified = False
    enabled_plugins = server_config.get("enabled_plugins", [])
    newly_installed = []

    for repo_name, url in default_plugins.items():
        target_dir = os.path.join(manager.plugins_dir, repo_name)
        if not os.path.isdir(target_dir):
            print(f"[Plugins] Auto-installing default plugin: {repo_name}...")
            result = manager.install_plugin_from_url(url)
            print(f"[Plugins] Install result for {repo_name}: {result}")
            
            if "[Success]" in result:
                newly_installed.append(repo_name)
    
    for repo_name in newly_installed:
        if repo_name in enabled_plugins:
            enabled_plugins.remove(repo_name)
            config_modified = True

    if config_modified:
        print("[Plugins] Disabling newly installed default plugins...")
        server_config["enabled_plugins"] = enabled_plugins
        try:
            write_config(server_config, server_config_filename)
        except Exception as e:
            print(f"[Plugins] ERROR: Failed to update config file '{server_config_filename}': {e}")


SYSTEM_PLUGINS = [
    "video_mask_creator",
    "guides",
    "configuration",
    "plugin_manager",
    "about",
]
BUNDLED_PLUGINS = {
    "agent_orchestrator",
    "downloads",
    "media_flow",
    "models_manager",
    "motion_designer",
    "sample",
}
USER_PLUGIN_INSERT_POSITION = 1

@dataclass
class InsertAfterRequest:
    target_component_id: str
    new_component_constructor: callable

@dataclass
class PluginTab:
    id: str
    label: str
    component_constructor: callable
    position: int = -1

@dataclass
class PluginModelExtension:
    plugin_id: str
    plugin_path: str
    model_handlers: List[str]
    defaults_root: str
    profiles_root: str

class WAN2GPPlugin:
    def __init__(self):
        self.tabs: Dict[str, PluginTab] = {}
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = "No description provided."
        self.type = ["app"]
        self.uninstallable = True
        self._component_requests: List[str] = []
        self._global_requests: List[str] = []
        self._insert_after_requests: List[InsertAfterRequest] = []
        self._setup_complete = False
        self._data_hooks: Dict[str, List[callable]] = {}
        self.tab_ids: List[str] = []
        self._set_wgp_global_func = None
        self._custom_js_snippets: List[str] = []
        self._deepy_prime_tools: List[PluginDeepyTool] = []
        self._deepy_zero_tools: List[PluginDeepyTool] = []
        
    def setup_ui(self) -> None:
        pass
        
    def add_tab(self, tab_id: str, label: str, component_constructor: callable, position: int = -1):
        self.tabs[tab_id] = PluginTab(id=tab_id, label=label, component_constructor=component_constructor, position=position)

    def post_ui_setup(self, components: Dict[str, gr.components.Component]) -> Dict[gr.components.Component, Union[gr.update, Any]]:
        return {}

    def on_tab_select(self, state: Dict[str, Any]) -> None:
        pass

    def on_tab_deselect(self, state: Dict[str, Any]) -> None:
        pass

    def on_model_change(self, state: Dict[str, Any], model_type: str) -> None:
        pass

    def request_component(self, component_id: str) -> None:
        if component_id not in self._component_requests:
            self._component_requests.append(component_id)
            
    def request_global(self, global_name: str) -> None:
        if global_name not in self._global_requests:
            self._global_requests.append(global_name)

    def sync_form(self, name, components, **kwargs):
        """Register a plugin form; request the shared 'state' component in setup_ui."""
        from shared.gradio.form_sync import GradioForm
        return GradioForm(self.state.value.service.forms, f'plugin-{self.name}-{name}', components, **kwargs)

    def set_global(self, variable_name: str, new_value: Any):
        if self._set_wgp_global_func:
            return self._set_wgp_global_func(variable_name, new_value)

    @property
    def component_requests(self) -> List[str]:
        return self._component_requests.copy()

    @property
    def global_requests(self) -> List[str]:
        return self._global_requests.copy()
        
    def register_data_hook(self, hook_name: str, callback: callable):
        if hook_name not in self._data_hooks:
            self._data_hooks[hook_name] = []
        self._data_hooks[hook_name].append(callback)

    def add_custom_js(self, js_code: str) -> None:
        if isinstance(js_code, str) and js_code.strip():
            self._custom_js_snippets.append(js_code)

    @staticmethod
    def _deepy_tool_signature(function: callable) -> inspect.Signature:
        if not callable(function):
            raise TypeError("Deepy tool function must be callable.")
        signature = inspect.signature(function)
        unsupported = [parameter.name for parameter in signature.parameters.values() if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}]
        if unsupported:
            raise TypeError(f"Deepy tool functions only support named parameters: {', '.join(unsupported)}")
        return signature

    @staticmethod
    def _deepy_tool_identity(function: callable, name: str | None, display_name: str | None, description: str | None) -> tuple[str, str, str]:
        tool_name = str(name or getattr(function, "__name__", "")).strip()
        if not _DEEPY_TOOL_NAME_RE.fullmatch(tool_name):
            raise ValueError(f"Invalid Deepy tool name '{tool_name}'. Use letters, digits, and underscores, starting with a letter or underscore.")
        tool_display_name = str(display_name or tool_name.replace("_", " ").title()).strip()
        tool_description = str((inspect.getdoc(function) or "") if description is None else description or "").strip()
        return tool_name, tool_display_name, tool_description

    @staticmethod
    def _add_deepy_tool(definitions: List[PluginDeepyTool], definition: PluginDeepyTool) -> callable:
        if any(existing.name == definition.name for existing in definitions):
            raise ValueError(f"Deepy tool '{definition.name}' is already registered by this plugin.")
        definitions.append(definition)
        return definition.function

    def register_deepy_prime_tool(self, function: callable, *, name: str | None = None, display_name: str | None = None, description: str | None = None, pause_runtime: bool = True, pause_reason: str = "tool", requires_file_system: bool = False) -> callable:
        """Expose a typed Python callable to Deepy Prime as an in-process MCP tool."""
        self._deepy_tool_signature(function)
        tool_name, tool_display_name, tool_description = self._deepy_tool_identity(function, name, display_name, description)
        if tool_name in _DEEPY_PRIME_RESERVED_TOOL_NAMES:
            raise ValueError(f"Deepy Prime tool name '{tool_name}' is reserved.")
        definition = PluginDeepyTool(tool_name, function, tool_display_name, tool_description, {}, bool(pause_runtime), str(pause_reason or "tool").strip() or "tool", bool(requires_file_system))
        return self._add_deepy_tool(self._deepy_prime_tools, definition)

    def register_deepy_zero_tool(self, function: callable, *, name: str | None = None, display_name: str | None = None, description: str | None = None, parameters: Dict[str, Dict[str, Any]] | None = None, pause_runtime: bool = True, pause_reason: str = "tool", requires_file_system: bool = False) -> callable:
        """Expose a typed synchronous Python callable to Deepy Zero as a native tool."""
        if inspect.iscoroutinefunction(function):
            raise TypeError("Deepy Zero tool functions must be synchronous.")
        signature = self._deepy_tool_signature(function)
        tool_name, tool_display_name, tool_description = self._deepy_tool_identity(function, name, display_name, description)
        parameter_overrides = dict(parameters or {})
        unknown_parameters = set(parameter_overrides) - set(signature.parameters)
        if unknown_parameters:
            raise ValueError(f"Unknown parameters for Deepy Zero tool '{tool_name}': {', '.join(sorted(unknown_parameters))}")
        tool_parameters = {}
        for parameter in signature.parameters.values():
            metadata = dict(parameter_overrides.get(parameter.name, {}))
            metadata.setdefault("required", parameter.default is inspect.Parameter.empty)
            tool_parameters[parameter.name] = metadata
        definition = PluginDeepyTool(tool_name, function, tool_display_name, tool_description, tool_parameters, bool(pause_runtime), str(pause_reason or "tool").strip() or "tool", bool(requires_file_system))
        return self._add_deepy_tool(self._deepy_zero_tools, definition)

    @property
    def custom_js_snippets(self) -> List[str]:
        return self._custom_js_snippets.copy()

    def insert_after(self, target_component_id: str, new_component_constructor: callable) -> None:
        if not hasattr(self, '_insert_after_requests'):
            self._insert_after_requests = []
        self._insert_after_requests.append(
            InsertAfterRequest(
                target_component_id=target_component_id,
                new_component_constructor=new_component_constructor
            )
        )

class PluginManager:
    def __init__(self, plugins_dir="plugins"):
        self.plugins: Dict[str, WAN2GPPlugin] = {}
        self.plugins_dir = plugins_dir
        os.makedirs(self.plugins_dir, exist_ok=True)
        if self.plugins_dir not in sys.path:
            sys.path.insert(0, self.plugins_dir)
        self.data_hooks: Dict[str, List[callable]] = {}
        self.restricted_globals: Set[str] = set()
        self.custom_js_snippets: List[str] = []
        self.repo_root = os.path.abspath(os.getcwd())
        self.catalog_path = os.path.join(self.repo_root, PLUGIN_CATALOG_FILENAME)
        self.local_catalog_path = os.path.join(self.repo_root, PLUGIN_LOCAL_CATALOG_FILENAME)
        self.server_config: Optional[Dict[str, Any]] = None
        self.server_config_filename: str = ""
        self._plugin_metadata_cache: Dict[str, tuple[Optional[int], Optional[Dict[str, Any]]]] = {}

    def set_server_config(self, server_config: Optional[Dict[str, Any]], server_config_filename: str = "") -> None:
        self.server_config = server_config if isinstance(server_config, dict) else None
        self.server_config_filename = server_config_filename or ""

    def _save_server_config(self) -> None:
        if not self.server_config or not self.server_config_filename:
            return
        try:
            write_config(self.server_config, self.server_config_filename)
        except Exception as e:
            print(f"[PluginManager] Failed to write config file '{self.server_config_filename}': {e}")

    def record_remote_plugin_install(self, plugin_id: str) -> None:
        if not self.server_config:
            return
        plugin_id = str(plugin_id or "").strip()
        if not plugin_id or plugin_id in SYSTEM_PLUGINS or plugin_id in BUNDLED_PLUGINS:
            return
        with config_lock:
            installed = self.server_config.get(INSTALLED_REMOTE_PLUGINS_KEY, [])
            if not isinstance(installed, list):
                installed = []
            if plugin_id not in installed:
                installed.append(plugin_id)
                self.server_config[INSTALLED_REMOTE_PLUGINS_KEY] = installed
                self._save_server_config()

    def clear_remote_plugin_install(self, plugin_id: str) -> None:
        if not self.server_config:
            return
        with config_lock:
            plugin_id = str(plugin_id or "").strip()
            installed = self.server_config.get(INSTALLED_REMOTE_PLUGINS_KEY, [])
            if not plugin_id or not isinstance(installed, list) or plugin_id not in installed:
                return
            self.server_config[INSTALLED_REMOTE_PLUGINS_KEY] = [item for item in installed if item != plugin_id]
            self._save_server_config()

    def _get_pending_deletions(self) -> List[str]:
        if not self.server_config:
            return []
        pending = self.server_config.get(PENDING_DELETIONS_KEY, [])
        if not isinstance(pending, list):
            return []
        cleaned = []
        for item in pending:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
        return cleaned

    def _set_pending_deletions(self, pending: List[str]) -> None:
        if not self.server_config:
            return
        unique = []
        seen = set()
        for item in pending:
            if not isinstance(item, str):
                continue
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(key)
        with config_lock:
            self.server_config[PENDING_DELETIONS_KEY] = unique
            self._save_server_config()

    def _add_pending_deletion(self, plugin_id: str) -> None:
        if not plugin_id:
            return
        with config_lock:
            pending = self._get_pending_deletions()
            if plugin_id not in pending:
                pending.append(plugin_id)
                self._set_pending_deletions(pending)

    def _clear_pending_deletion(self, plugin_id: str) -> None:
        if not plugin_id:
            return
        with config_lock:
            pending = self._get_pending_deletions()
            if plugin_id in pending:
                pending = [item for item in pending if item != plugin_id]
                self._set_pending_deletions(pending)

    def _is_cleanup_candidate(self, path: str) -> bool:
        try:
            for entry in os.scandir(path):
                name = entry.name
                if name.startswith("."):
                    continue
                if entry.is_file(follow_symlinks=False):
                    if name.endswith(".pyc"):
                        continue
                    return False
                if entry.is_dir(follow_symlinks=False):
                    if name == "__pycache__":
                        continue
                    if not self._is_cleanup_candidate(entry.path):
                        return False
            return True
        except Exception:
            return False

    def cleanup_pending_deletions(self) -> None:
        pending = self._get_pending_deletions()
        if not pending:
            return
        remaining = []
        for plugin_id in pending:
            plugin_dir = os.path.join(self.plugins_dir, plugin_id)
            if not os.path.isdir(plugin_dir):
                continue
            if self._is_cleanup_candidate(plugin_dir):
                try:
                    shutil.rmtree(plugin_dir, onerror=self._remove_readonly)
                    continue
                except Exception:
                    remaining.append(plugin_id)
                    continue
            remaining.append(plugin_id)
        self._set_pending_deletions(remaining)

    def _coerce_bool(self, value: Any, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
        return default

    def _coerce_string_list(self, value: Any) -> List[str]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            return []
        cleaned = []
        for item in values:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                cleaned.append(text)
        return cleaned

    def _load_json_file(self, path: str) -> Optional[Any]:
        if not path or not os.path.isfile(path):
            return None
        last_error = None
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                with open(path, "r", encoding=encoding) as reader:
                    return json.load(reader)
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except json.JSONDecodeError as e:
                last_error = e
                if encoding == "utf-8" and "UTF-8 BOM" in str(e):
                    continue
                break
            except Exception as e:
                last_error = e
                break
        if last_error is not None:
            print(f"[PluginManager] Failed to read JSON from {path}: {last_error}")
        return None

    def _write_json_file(self, path: str, payload: Any) -> None:
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as writer:
                json.dump(payload, writer, indent=2)
        except Exception as e:
            print(f"[PluginManager] Failed to write JSON to {path}: {e}")

    def _load_plugin_metadata(self, plugin_path: str) -> Optional[Dict[str, Any]]:
        metadata_path = os.path.join(plugin_path, PLUGIN_METADATA_FILENAME)
        try:
            mtime_ns = os.stat(metadata_path).st_mtime_ns
        except FileNotFoundError:
            mtime_ns = None
        except Exception as e:
            print(f"[PluginManager] Failed to stat {metadata_path}: {e}")
            mtime_ns = None
        cached = self._plugin_metadata_cache.get(metadata_path)
        if cached is not None and cached[0] == mtime_ns:
            return dict(cached[1]) if cached[1] is not None else None
        payload = self._load_json_file(metadata_path)
        if not isinstance(payload, dict):
            self._plugin_metadata_cache[metadata_path] = (mtime_ns, None)
            return None
        metadata = self._normalize_plugin_metadata(payload)
        self._plugin_metadata_cache[metadata_path] = (mtime_ns, dict(metadata))
        return dict(metadata)

    def _invalidate_plugin_metadata_cache(self, plugin_path: str) -> None:
        self._plugin_metadata_cache.pop(os.path.join(plugin_path, PLUGIN_METADATA_FILENAME), None)

    def _normalize_plugin_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(payload)
        for key in ("name", "version", "description", "author", "date", "wan2gp_version", "url"):
            value = metadata.get(key)
            if isinstance(value, str):
                metadata[key] = value.strip()
            elif value is None:
                metadata[key] = ""
            else:
                metadata[key] = str(value)
        legacy_version = metadata.get("wangp_version")
        if not _has_value(metadata.get("wan2gp_version")) and _has_value(legacy_version):
            metadata["wan2gp_version"] = str(legacy_version).strip()
        metadata.pop("wangp_version", None)
        metadata["url"] = normalize_plugin_url(metadata.get("url", ""))
        metadata["type"] = normalize_plugin_types(metadata.get("type"))
        metadata["uninstallable"] = self._coerce_bool(metadata.get("uninstallable"), default=True)
        metadata[PLUGIN_SPATIAL_UPSAMPLER_HANDLERS_KEY] = self._coerce_string_list(metadata.get(PLUGIN_SPATIAL_UPSAMPLER_HANDLERS_KEY))
        metadata[PLUGIN_TEMPORAL_UPSAMPLER_HANDLERS_KEY] = self._coerce_string_list(metadata.get(PLUGIN_TEMPORAL_UPSAMPLER_HANDLERS_KEY))
        metadata[PLUGIN_AUDIO_PROCESSORS_KEY] = self._coerce_string_list(metadata.get(PLUGIN_AUDIO_PROCESSORS_KEY))
        metadata[PLUGIN_MODEL_HANDLERS_KEY] = self._coerce_string_list(metadata.get(PLUGIN_MODEL_HANDLERS_KEY))
        for key in (PLUGIN_MODEL_DEFAULTS_KEY, PLUGIN_MODEL_PROFILES_KEY):
            value = metadata.get(key)
            if isinstance(value, str):
                metadata[key] = value.strip()
            elif value is None:
                metadata[key] = ""
            else:
                metadata[key] = str(value)
        return metadata

    def _apply_metadata_to_plugin(self, plugin: WAN2GPPlugin, metadata: Optional[Dict[str, Any]], is_system: bool) -> None:
        if not metadata:
            if is_system:
                plugin.uninstallable = False
            return
        for key in ("name", "version", "description", "author", "date", "wan2gp_version"):
            if key in metadata:
                setattr(plugin, key, metadata.get(key))
        if "type" in metadata:
            plugin.type = normalize_plugin_types(metadata.get("type"))
        if not _has_value(getattr(plugin, "wan2gp_version", "")) and _has_value(metadata.get("wangp_version")):
            setattr(plugin, "wan2gp_version", str(metadata.get("wangp_version")).strip())
        if "uninstallable" in metadata:
            plugin.uninstallable = self._coerce_bool(metadata.get("uninstallable"), default=True)
        if is_system:
            plugin.uninstallable = False

    def _resolve_plugin_class_path(self, plugin_id: str, handler_path: str, label: str = "handler") -> Optional[str]:
        text = str(handler_path or "").strip()
        if not text:
            return None
        is_relative = text.startswith(".")
        if ":" in text:
            module_path, class_name = text.rsplit(":", 1)
        else:
            if "." not in text:
                print(f"[PluginManager] Invalid {label} path for plugin {plugin_id}: {handler_path}")
                return None
            module_path, class_name = text.rsplit(".", 1)
        module_path = module_path.strip().replace("\\", "/")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        module_path = module_path.replace("/", ".")
        if is_relative:
            module_path = module_path.lstrip(".")
        module_path = module_path.strip(".")
        class_name = class_name.strip()
        if not module_path or not class_name:
            print(f"[PluginManager] Invalid {label} path for plugin {plugin_id}: {handler_path}")
            return None
        full_module_path = f"{plugin_id}.{module_path}" if is_relative else module_path
        return f"{full_module_path}.{class_name}"

    def _resolve_plugin_module_path(self, plugin_id: str, module_path: str, label: str = "module") -> Optional[str]:
        text = str(module_path or "").strip()
        if not text:
            return None
        is_relative = text.startswith(".")
        text = text.replace("\\", "/")
        if text.endswith(".py"):
            text = text[:-3]
        if is_relative:
            text = text.lstrip(".").lstrip("/")
        text = text.strip("/").replace("/", ".").strip(".")
        if not text:
            print(f"[PluginManager] Invalid {label} path for plugin {plugin_id}: {module_path}")
            return None
        return f"{plugin_id}.{text}" if is_relative else text

    def _resolve_plugin_folder_path(self, plugin_path: str, path_value: str) -> str:
        path_text = str(path_value or "").strip()
        if not path_text:
            return ""
        path_text = path_text.replace("\\", os.sep).replace("/", os.sep)
        if os.path.isabs(path_text):
            return os.path.abspath(path_text)
        if path_text.startswith("."):
            return os.path.abspath(os.path.join(plugin_path, path_text))
        return os.path.abspath(path_text)

    def discover_plugin_model_extensions(self, enabled_plugins: List[str]) -> List[PluginModelExtension]:
        enabled = set(enabled_plugins or []) | set(SYSTEM_PLUGINS)
        extensions = []
        for plugin_id in self.discover_plugins():
            if plugin_id not in enabled:
                continue
            plugin_path = os.path.join(self.plugins_dir, plugin_id)
            metadata = self._load_plugin_metadata(plugin_path)
            if not metadata:
                continue
            handlers = metadata.get(PLUGIN_MODEL_HANDLERS_KEY, [])
            if not handlers:
                continue
            defaults_root = self._resolve_plugin_folder_path(plugin_path, metadata.get(PLUGIN_MODEL_DEFAULTS_KEY, ""))
            profiles_root = self._resolve_plugin_folder_path(plugin_path, metadata.get(PLUGIN_MODEL_PROFILES_KEY, ""))
            if not defaults_root or not profiles_root:
                print(f"[PluginManager] Plugin {plugin_id} declares model handlers but is missing '{PLUGIN_MODEL_DEFAULTS_KEY}' or '{PLUGIN_MODEL_PROFILES_KEY}' in {PLUGIN_METADATA_FILENAME}.")
                continue
            handler_paths = []
            for handler_path in handlers:
                resolved_path = self._resolve_plugin_module_path(plugin_id, handler_path, label="model handler")
                if resolved_path is not None:
                    handler_paths.append(resolved_path)
            if not handler_paths:
                continue
            extensions.append(PluginModelExtension(plugin_id, plugin_path, handler_paths, defaults_root, profiles_root))
            print(f"[PluginManager] Loaded model plugin: {metadata.get('name') or plugin_id} (from {plugin_id})")
        return extensions

    def _register_plugin_spatial_upsamplers(self, plugin_id: str, metadata: Optional[Dict[str, Any]], files_locator=None) -> None:
        if not metadata:
            return
        handlers = metadata.get(PLUGIN_SPATIAL_UPSAMPLER_HANDLERS_KEY, [])
        if not handlers:
            return
        handler_paths = []
        for handler_path in handlers:
            resolved_path = self._resolve_plugin_class_path(plugin_id, handler_path, label="spatial upsampler handler")
            if resolved_path is not None:
                handler_paths.append(resolved_path)
        if not handler_paths:
            return
        try:
            from postprocessing import spatial_upsamplers as upsampler_api
            upsampler_api.register_spatial_upsamplers(self.server_config, files_locator, handler_paths)
        except Exception as e:
            print(f"[PluginManager] Error registering spatial upsamplers for plugin {plugin_id}: {e}")
            traceback.print_exc()

    def _register_plugin_temporal_upsamplers(self, plugin_id: str, metadata: Optional[Dict[str, Any]], files_locator=None) -> None:
        if not metadata:
            return
        handlers = metadata.get(PLUGIN_TEMPORAL_UPSAMPLER_HANDLERS_KEY, [])
        if not handlers:
            return
        handler_paths = []
        for handler_path in handlers:
            resolved_path = self._resolve_plugin_class_path(plugin_id, handler_path, label="temporal upsampler handler")
            if resolved_path is not None:
                handler_paths.append(resolved_path)
        if not handler_paths:
            return
        try:
            from postprocessing import temporal_upsamplers as temporal_upsampler_api
            temporal_upsampler_api.register_temporal_upsamplers(self.server_config, files_locator, handler_paths)
        except Exception as e:
            print(f"[PluginManager] Error registering temporal upsamplers for plugin {plugin_id}: {e}")
            traceback.print_exc()

    def _register_plugin_audio_processors(self, plugin_id: str, metadata: Optional[Dict[str, Any]], files_locator=None) -> None:
        if not metadata:
            return
        handlers = metadata.get(PLUGIN_AUDIO_PROCESSORS_KEY, [])
        if not handlers:
            return
        handler_paths = []
        for handler_path in handlers:
            resolved_path = self._resolve_plugin_class_path(plugin_id, handler_path, label="audio processor")
            if resolved_path is not None:
                handler_paths.append(resolved_path)
        if not handler_paths:
            return
        try:
            from postprocessing import audio_processors as audio_processor_api
            audio_processor_api.register_audio_processors(self.server_config, files_locator, handler_paths)
        except Exception as e:
            print(f"[PluginManager] Error registering audio processors for plugin {plugin_id}: {e}")
            traceback.print_exc()

    def _normalize_catalog_entry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        entry = dict(payload)
        for key in ("name", "author", "version", "description", "url", "date", "wan2gp_version", "last_check"):
            value = entry.get(key)
            if key == "url":
                if isinstance(value, str):
                    entry[key] = normalize_plugin_url(value)
                elif value is None:
                    entry[key] = ""
                else:
                    entry[key] = normalize_plugin_url(str(value))
                continue
            if isinstance(value, str):
                entry[key] = value.strip()
            elif value is None:
                entry[key] = ""
            else:
                entry[key] = str(value)
        legacy_version = entry.get("wangp_version")
        if not _has_value(entry.get("wan2gp_version")) and _has_value(legacy_version):
            entry["wan2gp_version"] = str(legacy_version).strip()
        entry.pop("wangp_version", None)
        entry["type"] = normalize_plugin_types(entry.get("type"))
        return entry

    def _merge_entry_fields(self, primary: Dict[str, Any], secondary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        result = dict(primary) if primary else {}
        if not secondary:
            return result
        for key, value in secondary.items():
            if key == "type":
                result[key] = merge_plugin_types(result.get(key), value)
                continue
            if _has_value(result.get(key)):
                continue
            if _has_value(value):
                result[key] = value
        return result

    def _merge_catalog_entries(
        self,
        base_entries: List[Dict[str, Any]],
        local_entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        base_map: Dict[str, Dict[str, Any]] = {}
        local_map: Dict[str, Dict[str, Any]] = {}
        for entry in base_entries:
            plugin_id = entry.get("id") or plugin_id_from_url(entry.get("url", ""))
            if not plugin_id:
                continue
            base_map[plugin_id] = self._normalize_catalog_entry(entry)
        for entry in local_entries:
            plugin_id = entry.get("id") or plugin_id_from_url(entry.get("url", ""))
            if not plugin_id:
                continue
            local_map[plugin_id] = self._normalize_catalog_entry(entry)

        merged: Dict[str, Dict[str, Any]] = {}
        all_ids = set(base_map.keys()) | set(local_map.keys())
        for plugin_id in all_ids:
            base_entry = base_map.get(plugin_id)
            local_entry = local_map.get(plugin_id)
            if base_entry and local_entry:
                comparison = compare_release_metadata(local_entry, base_entry)
                if comparison > 0:
                    merged_entry = dict(local_entry)
                    merged_entry["type"] = merge_plugin_types(local_entry.get("type"), base_entry.get("type"))
                else:
                    merged_entry = dict(base_entry)
                    merged_entry["type"] = merge_plugin_types(base_entry.get("type"), local_entry.get("type"))
                    if _has_value(local_entry.get("last_check")):
                        merged_entry["last_check"] = local_entry.get("last_check")
                merged[plugin_id] = merged_entry
            else:
                merged[plugin_id] = base_entry or local_entry
        return merged

    def _fetch_remote_catalog_entries(self) -> List[Dict[str, Any]]:
        try:
            response = requests.get(COMMUNITY_PLUGINS_URL, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                return [self._normalize_catalog_entry(entry) for entry in payload if isinstance(entry, dict)]
        except Exception as e:
            print(f"[PluginManager] Could not fetch community plugins info: {e}")
        return []

    def load_catalog_entries(self, use_remote: bool = True) -> List[Dict[str, Any]]:
        entries = None # self._fetch_remote_catalog_entries() if use_remote else []
        if not entries:
            payload = self._load_json_file(self.catalog_path)
            if isinstance(payload, list):
                entries = [self._normalize_catalog_entry(entry) for entry in payload if isinstance(entry, dict)]
        return entries or []

    def load_local_catalog_entries(self) -> List[Dict[str, Any]]:
        payload = self._load_json_file(self.local_catalog_path)
        if isinstance(payload, list):
            return [self._normalize_catalog_entry(entry) for entry in payload if isinstance(entry, dict)]
        return []

    def get_merged_catalog_entries(self, use_remote: bool = True) -> Dict[str, Dict[str, Any]]:
        base_entries = self.load_catalog_entries(use_remote=use_remote)
        local_entries = self.load_local_catalog_entries()
        return self._merge_catalog_entries(base_entries, local_entries)

    def _build_plugin_json_url(self, url: str) -> str:
        repo_info = _split_github_repo(url)
        if not repo_info:
            return ""
        owner, repo = repo_info
        return f"https://github.com/{owner}/{repo}/raw/HEAD/{PLUGIN_METADATA_FILENAME}"

    def _fetch_plugin_json(self, url: str, quiet: bool = False) -> Optional[Dict[str, Any]]:
        plugin_json_url = self._build_plugin_json_url(url)
        if not plugin_json_url:
            return None
        try:
            response = requests.get(plugin_json_url, timeout=10)
            if response.status_code != 200:
                return None
            payload = response.json()
            if not isinstance(payload, dict):
                return None
            return self._normalize_plugin_metadata(payload)
        except Exception as e:
            if not quiet:
                print(f"[PluginManager] Could not fetch {PLUGIN_METADATA_FILENAME} for {url}: {e}")
            return None

    def _metadata_to_catalog_entry(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        entry = {}
        for key in ("name", "author", "version", "description", "type", "date", "wan2gp_version", "url"):
            entry[key] = metadata.get(key, "")
        return self._normalize_catalog_entry(entry)

    def _extract_catalog_metadata(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = {}
        for key in ("name", "author", "version", "description", "type", "url", "date", "wan2gp_version"):
            payload[key] = entry.get(key, "")
        return self._normalize_catalog_entry(payload)

    def _find_catalog_entry(self, plugin_id: str, url: str = "", use_remote: bool = False) -> Optional[Dict[str, Any]]:
        entries = self.load_catalog_entries(use_remote=use_remote)
        target_id = plugin_id.strip().lower() if isinstance(plugin_id, str) else ""
        target_url = normalize_plugin_url(url).lower() if isinstance(url, str) else ""
        for entry in entries:
            entry_url = entry.get("url", "")
            entry_id = entry.get("id") or plugin_id_from_url(entry_url)
            entry_id = entry_id.strip().lower() if isinstance(entry_id, str) else ""
            entry_url_norm = normalize_plugin_url(entry_url).lower() if isinstance(entry_url, str) else ""
            if target_id and entry_id and entry_id == target_id:
                return self._normalize_catalog_entry(entry)
            if target_url and entry_url_norm and entry_url_norm == target_url:
                return self._normalize_catalog_entry(entry)
        return None

    def _get_git_remote_url(self, plugin_path: str) -> str:
        if not plugin_path:
            return ""
        git_dir = os.path.join(plugin_path, ".git")
        if not os.path.isdir(git_dir):
            return ""
        try:
            repo = git.Repo(plugin_path)
            if repo.remotes:
                return repo.remotes.origin.url
        except Exception:
            return ""
        return ""

    def refresh_catalog(self, installed_only: bool = True, use_remote: bool = True) -> Dict[str, int]:
        base_entries = self.load_catalog_entries(use_remote=False)
        local_entries = self.load_local_catalog_entries()
        merged_catalog = self._merge_catalog_entries(base_entries, local_entries)
        local_map: Dict[str, Dict[str, Any]] = {}
        for entry in local_entries:
            plugin_id = entry.get("id") or plugin_id_from_url(entry.get("url", ""))
            if plugin_id:
                local_map[plugin_id] = self._normalize_catalog_entry(entry)

        def _info_to_entry(info: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "name": info.get("name", ""),
                "author": info.get("author", ""),
                "version": info.get("version", ""),
                "description": info.get("description", ""),
                "type": info.get("type", ["app"]),
                "url": info.get("url", ""),
                "date": info.get("date", ""),
                "wan2gp_version": info.get("wan2gp_version", ""),
            }

        targets: List[Dict[str, Any]] = []
        if installed_only:
            installed_plugins = [info for info in self.get_plugins_info() if not info.get("system")]
            for info in installed_plugins:
                plugin_id = info.get("id", "")
                if not plugin_id:
                    continue
                base_entry = merged_catalog.get(plugin_id, {})
                git_url = self._get_git_remote_url(info.get("path", ""))
                url = normalize_plugin_url(base_entry.get("url") or info.get("url") or git_url or "")
                if git_url and plugin_id_from_url(git_url) == plugin_id:
                    url = normalize_plugin_url(git_url)
                if not _has_value(url):
                    continue
                targets.append(
                    {
                        "id": plugin_id,
                        "url": url,
                        "base_entry": base_entry,
                        "info_entry": _info_to_entry(info),
                    }
                )
        else:
            for plugin_id, entry in merged_catalog.items():
                url = normalize_plugin_url(entry.get("url") or "")
                if not _has_value(url):
                    continue
                targets.append({"id": plugin_id, "url": url, "base_entry": entry, "info_entry": {}})

        checked = 0
        updated = 0
        updates_available = 0
        for target in targets:
            plugin_id = target["id"]
            url = target["url"]
            base_entry = target.get("base_entry", {})
            info_entry = target.get("info_entry", {})
            checked += 1
            metadata = self._fetch_plugin_json(url, quiet=True)
            if not metadata:
                continue
            catalog_entry = self._metadata_to_catalog_entry(metadata)
            if _has_value(url):
                catalog_entry_url = catalog_entry.get("url", "")
                if not _has_value(catalog_entry_url) or plugin_id_from_url(catalog_entry_url) != plugin_id:
                    catalog_entry["url"] = url
            catalog_entry = self._merge_entry_fields(catalog_entry, base_entry)
            if info_entry:
                catalog_entry = self._merge_entry_fields(catalog_entry, info_entry)
            catalog_entry["last_check"] = datetime.datetime.now().isoformat(timespec="seconds")
            catalog_entry = self._normalize_catalog_entry(catalog_entry)
            existing_entry = local_map.get(plugin_id, {})
            if info_entry:
                if compare_release_metadata(catalog_entry, info_entry) > 0:
                    updates_available += 1
            elif existing_entry:
                if compare_release_metadata(catalog_entry, existing_entry) > 0:
                    updates_available += 1
            local_map[plugin_id] = catalog_entry
            updated += 1

        if updated > 0:
            self._write_json_file(self.local_catalog_path, list(local_map.values()))
        return {"checked": checked, "updated": updated, "updates_available": updates_available}

    def record_plugin_metadata(self, plugin_id: str, url: str = "") -> bool:
        if not plugin_id:
            return False
        url = normalize_plugin_url(url)
        plugin_path = os.path.join(self.plugins_dir, plugin_id)
        metadata = self._load_plugin_metadata(plugin_path) if os.path.isdir(plugin_path) else None
        plugin_json_found = metadata is not None
        if not metadata and _has_value(url):
            metadata = self._fetch_plugin_json(url)
            plugin_json_found = metadata is not None
        if not metadata:
            plugin_info = next((info for info in self.get_plugins_info() if info.get("id") == plugin_id), None)
            if plugin_info:
                metadata = {
                    "name": plugin_info.get("name", ""),
                    "version": plugin_info.get("version", ""),
                    "description": plugin_info.get("description", ""),
                    "author": plugin_info.get("author", ""),
                    "type": plugin_info.get("type", ["app"]),
                    "url": plugin_info.get("url", ""),
                    "date": plugin_info.get("date", ""),
                    "wan2gp_version": plugin_info.get("wan2gp_version", ""),
                }
        if not metadata:
            return False

        catalog_entry = self._metadata_to_catalog_entry(metadata)
        if _has_value(url):
            catalog_entry_url = catalog_entry.get("url", "")
            if not _has_value(catalog_entry_url) or plugin_id_from_url(catalog_entry_url) != plugin_id:
                catalog_entry["url"] = url

        merged_catalog = self.get_merged_catalog_entries(use_remote=False)
        catalog_entry = self._merge_entry_fields(catalog_entry, merged_catalog.get(plugin_id, {}))
        base_entry = self._find_catalog_entry(plugin_id, url=url, use_remote=False)
        if base_entry:
            catalog_entry = self._merge_entry_fields(catalog_entry, base_entry)
        if plugin_json_found:
            catalog_entry["last_check"] = datetime.datetime.now().isoformat(timespec="seconds")
        catalog_entry = self._normalize_catalog_entry(catalog_entry)

        local_entries = self.load_local_catalog_entries()
        local_map: Dict[str, Dict[str, Any]] = {}
        for entry in local_entries:
            existing_id = entry.get("id") or plugin_id_from_url(entry.get("url", ""))
            if existing_id:
                local_map[existing_id] = self._normalize_catalog_entry(entry)
        local_map[plugin_id] = catalog_entry
        self._write_json_file(self.local_catalog_path, list(local_map.values()))
        return True

    def merge_local_catalog(self) -> str:
        local_entries = self.load_local_catalog_entries()
        if not local_entries:
            return "[Info] No local catalog entries to merge."
        base_entries = self.load_catalog_entries(use_remote=False)
        merged = self._merge_catalog_entries(base_entries, local_entries)
        merged_list = []
        for entry in merged.values():
            entry.pop("last_check", None)
            merged_list.append(entry)
        merged_list.sort(key=lambda item: item.get("name", ""))
        self._write_json_file(self.catalog_path, merged_list)
        try:
            os.remove(self.local_catalog_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            return f"[Warning] Catalog merged, but failed to remove {self.local_catalog_path}: {e}"
        return "[Success] Catalog merged and local overrides removed."

    def get_plugins_info(self) -> List[Dict[str, Any]]:
        plugins_info = []
        for dir_name in self.discover_plugins():
            plugin_path = os.path.join(self.plugins_dir, dir_name)
            is_system = dir_name in SYSTEM_PLUGINS
            info = {
                'id': dir_name,
                'name': dir_name,
                'version': 'N/A',
                'description': 'No description provided.',
                'author': '',
                'type': ['app'],
                'url': '',
                'date': '',
                'wan2gp_version': '',
                'path': plugin_path,
                'system': is_system,
                'uninstallable': True,
            }
            metadata = self._load_plugin_metadata(plugin_path)
            if metadata:
                info['name'] = metadata.get('name', info['name'])
                info['version'] = metadata.get('version', info['version'])
                info['description'] = metadata.get('description', info['description'])
                info['author'] = metadata.get('author', info['author'])
                info['type'] = normalize_plugin_types(metadata.get('type', info['type']))
                info['url'] = metadata.get('url', info['url'])
                info['date'] = metadata.get('date', info['date'])
                info['wan2gp_version'] = metadata.get('wan2gp_version', info['wan2gp_version'])
                info['uninstallable'] = bool(metadata.get('uninstallable', info['uninstallable']))
            else:
                try:
                    module = importlib.import_module(f"{dir_name}.plugin")
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                            instance = obj()
                            info['name'] = instance.name
                            info['version'] = instance.version
                            info['description'] = instance.description
                            info['type'] = normalize_plugin_types(getattr(instance, 'type', info['type']))
                            info['uninstallable'] = bool(getattr(instance, 'uninstallable', True))
                            break
                except Exception as e:
                    print(f"Could not load metadata for plugin {dir_name}: {e}")
            if is_system:
                info['uninstallable'] = False
            if info['id'] in BUNDLED_PLUGINS:
                info['uninstallable'] = False
            if not is_system:
                merged_catalog = self.get_merged_catalog_entries(use_remote=False)
                catalog_entry = merged_catalog.get(info.get("id", ""))
                if catalog_entry:
                    info = self._merge_entry_fields(info, self._extract_catalog_metadata(catalog_entry))
            plugins_info.append(info)
        
        plugins_info.sort(key=lambda p: (not p['system'], p['name']))
        return plugins_info

    def _remove_readonly(self, func, path, exc_info):
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        else:
            raise

    def _is_plugin_uninstallable(self, plugin_id: str) -> bool:
        if plugin_id in SYSTEM_PLUGINS:
            return False
        if plugin_id in BUNDLED_PLUGINS:
            return False
        try:
            for info in self.get_plugins_info():
                if info.get('id') == plugin_id:
                    return bool(info.get('uninstallable', True))
        except Exception:
            pass
        return True

    def uninstall_plugin(self, plugin_id: str):
        if not plugin_id:
            return "[Error] No plugin selected for uninstallation."
        
        if plugin_id in SYSTEM_PLUGINS:
            return f"[Error] Cannot uninstall system plugin '{plugin_id}'."
        if not self._is_plugin_uninstallable(plugin_id):
            return f"[Error] Cannot uninstall protected plugin '{plugin_id}'."

        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(target_dir):
            return f"[Error] Plugin '{plugin_id}' directory not found."
        self._add_pending_deletion(plugin_id)

        try:
            shutil.rmtree(target_dir, onerror=self._remove_readonly)
            self.clear_remote_plugin_install(plugin_id)
            return f"[Success] Plugin '{plugin_id}' uninstalled. Please restart WanGP."
        except Exception as e:
            return f"[Error] Failed to remove plugin '{plugin_id}': {e}"

    def update_plugin(self, plugin_id: str, progress=None):
        if not plugin_id:
            return "[Error] No plugin selected for update."
            
        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(os.path.join(target_dir, '.git')):
            return f"[Error] '{plugin_id}' is not a git repository and cannot be updated automatically."

        try:
            if progress is not None: progress(0, desc=f"Updating '{plugin_id}'...")
            repo = git.Repo(target_dir)
            if not repo.remotes:
                return f"[Error] Update failed: no git remote configured for '{plugin_id}'."
            origin = repo.remotes.origin
            
            if progress is not None: progress(0.2, desc=f"Fetching updates for '{plugin_id}'...")
            origin.fetch()

            local_commit = repo.head.commit
            try:
                branch_name = repo.active_branch.name
                remote_ref = origin.refs[branch_name]
                remote_commit = remote_ref.commit
            except Exception:
                return f"[Error] Update failed: could not resolve remote branch for '{plugin_id}'."

            if local_commit == remote_commit:
                 self.record_plugin_metadata(plugin_id)
                 return f"[Info] Plugin '{plugin_id}' is already up to date."

            if progress is not None: progress(0.6, desc=f"Pulling updates for '{plugin_id}'...")
            origin.pull()
            
            requirements_path = os.path.join(target_dir, 'requirements.txt')
            if os.path.exists(requirements_path):
                if progress is not None: progress(0.8, desc=f"Re-installing dependencies for '{plugin_id}'...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])

            if progress is not None: progress(1.0, desc="Update complete.")
            self.record_plugin_metadata(plugin_id)
            return f"[Success] Plugin '{plugin_id}' updated. Please restart WanGP for changes to take effect."
        except git.exc.GitCommandError as e:
            traceback.print_exc()
            stderr = (e.stderr or str(e)).strip()
            lowered = stderr.lower()
            if any(token in lowered for token in ("not found", "repository", "could not read from remote")):
                return f"[Error] Update failed: remote repository not found or unreachable for '{plugin_id}'."
            if any(token in lowered for token in ("authentication", "access denied", "permission denied")):
                return f"[Error] Update failed: access denied to remote for '{plugin_id}'."
            return f"[Error] Git update failed for '{plugin_id}': {stderr}"
        except Exception as e:
            traceback.print_exc()
            return f"[Error] An unexpected error occurred during update of '{plugin_id}': {str(e)}"

    def reinstall_plugin(self, plugin_id: str, progress=None):
        if not plugin_id:
            return "[Error] No plugin selected for reinstallation."

        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(target_dir):
            return f"[Error] Plugin '{plugin_id}' not found."

        if progress is not None: progress(0, desc=f"Reinstalling '{plugin_id}' from local folder...")
        install_msg = self.install_local_plugin(plugin_id, progress=progress)
        if "[Success]" in install_msg:
            return f"[Success] Plugin '{plugin_id}' reinstalled from local folder. Please restart WanGP."
        return f"[Error] Reinstallation failed from local folder: {install_msg}"

    def install_local_plugin(self, plugin_id: str, progress=None):
        plugin_id = str(plugin_id or "").strip()
        if not plugin_id:
            return "[Error] No plugin selected for installation."
        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.isdir(target_dir):
            return f"[Error] Plugin '{plugin_id}' directory not found."
        try:
            return self._finalize_local_plugin_install(plugin_id, target_dir, progress=progress)
        except Exception as e:
            traceback.print_exc()
            return f"[Error] An unexpected error occurred during installation of '{plugin_id}': {str(e)}"

    def _finalize_local_plugin_install(self, plugin_id: str, target_dir: str, progress=None, remove_invalid: bool = False, source_url: str = ""):
        plugin_entry = os.path.join(target_dir, "plugin.py")
        metadata = self._load_plugin_metadata(target_dir)
        metadata_handlers = []
        if metadata:
            metadata_handlers = metadata.get(PLUGIN_MODEL_HANDLERS_KEY, []) + metadata.get(PLUGIN_SPATIAL_UPSAMPLER_HANDLERS_KEY, []) + metadata.get(PLUGIN_TEMPORAL_UPSAMPLER_HANDLERS_KEY, []) + metadata.get(PLUGIN_AUDIO_PROCESSORS_KEY, [])
        if not os.path.isfile(plugin_entry) and not metadata_handlers:
            if remove_invalid:
                shutil.rmtree(target_dir, onerror=self._remove_readonly)
            return "[Error] Invalid Plugin."

        requirements_path = os.path.join(target_dir, 'requirements.txt')
        if os.path.exists(requirements_path):
            if progress is not None: progress(0.5, desc=f"Installing dependencies for '{plugin_id}'...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
            except subprocess.CalledProcessError as e:
                traceback.print_exc()
                return f"[Error] Failed to install dependencies for {plugin_id}. Check console for details. Error: {e}"

        setup_path = os.path.join(target_dir, 'setup.py')
        if os.path.exists(setup_path):
            if progress is not None: progress(0.8, desc=f"Running setup for '{plugin_id}'...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', target_dir])
            except subprocess.CalledProcessError as e:
                traceback.print_exc()
                return f"[Error] Failed to run setup.py for {plugin_id}. Check console for details. Error: {e}"

        init_path = os.path.join(target_dir, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                pass

        self._strip_uninstallable_flag(target_dir)
        if progress is not None: progress(1.0, desc="Installation complete.")
        self._clear_pending_deletion(plugin_id)
        self.record_remote_plugin_install(plugin_id)
        self.record_plugin_metadata(plugin_id, url=source_url)
        return f"[Success] Plugin '{plugin_id}' installed. Please enable it in the list and restart WanGP."

    def install_plugin_from_url(self, git_url: str, progress=None):
        cleaned_url = normalize_plugin_url(git_url)
        if not cleaned_url or not cleaned_url.startswith("https://github.com/"):
            return "[Error] Invalid URL."

        try:
            repo_name = plugin_id_from_url(cleaned_url)
            if not repo_name:
                return "[Error] Invalid URL."
            target_dir = os.path.join(self.plugins_dir, repo_name)

            if os.path.exists(target_dir):
                return f"[Warning] Plugin '{repo_name}' already exists. Please remove it manually to reinstall."

            if progress is not None: progress(0.1, desc=f"Cloning '{repo_name}'...")
            git.Repo.clone_from(cleaned_url, target_dir)

            return self._finalize_local_plugin_install(repo_name, target_dir, progress=progress, remove_invalid=True, source_url=cleaned_url)

        except git.exc.GitCommandError as e:
            traceback.print_exc()
            stderr = (e.stderr or str(e)).strip()
            lowered = stderr.lower()
            if any(token in lowered for token in ("not found", "repository", "fatal", "could not read from remote")):
                return "[Error] Invalid URL."
            return f"[Error] Git clone failed: {stderr}"
        except Exception as e:
            traceback.print_exc()
            return f"[Error] An unexpected error occurred: {str(e)}"

    def _strip_uninstallable_flag(self, plugin_dir: str) -> None:
        if not plugin_dir or not os.path.isdir(plugin_dir):
            return
        metadata_path = os.path.join(plugin_dir, PLUGIN_METADATA_FILENAME)
        payload = self._load_json_file(metadata_path)
        if not isinstance(payload, dict):
            return
        if "uninstallable" not in payload:
            return
        if not self._coerce_bool(payload.get("uninstallable"), default=True):
            payload.pop("uninstallable", None)
            try:
                with open(metadata_path, "w", encoding="utf-8") as writer:
                    json.dump(payload, writer, indent=2, ensure_ascii=True)
                self._invalidate_plugin_metadata_cache(plugin_dir)
            except Exception as e:
                print(f"[PluginManager] Failed to update {metadata_path}: {e}")

    @staticmethod
    def _activate_plugin_deepy_tools(plugin_id: str, plugin: WAN2GPPlugin) -> None:
        prime_tools = [replace(definition, plugin_id=plugin_id) for definition in plugin._deepy_prime_tools]
        zero_tools = [replace(definition, plugin_id=plugin_id) for definition in plugin._deepy_zero_tools]
        registrations = [(definition, _ACTIVE_DEEPY_PRIME_TOOLS, "Deepy Prime") for definition in prime_tools]
        registrations += [(definition, _ACTIVE_DEEPY_ZERO_TOOLS, "Deepy Zero") for definition in zero_tools]
        for definition, registry, runtime_name in registrations:
            existing = registry.get(definition.name)
            if existing is not None:
                raise ValueError(f"{runtime_name} tool '{definition.name}' is already registered by plugin '{existing.plugin_id}'.")
        _ACTIVE_DEEPY_PRIME_TOOLS.update({definition.name: definition for definition in prime_tools})
        _ACTIVE_DEEPY_ZERO_TOOLS.update({definition.name: definition for definition in zero_tools})

    def discover_plugins(self) -> List[str]:
        discovered = []
        for item in os.listdir(self.plugins_dir):
            if item.endswith(".bak"):
                continue
            path = os.path.join(self.plugins_dir, item)
            if os.path.isdir(path) and (os.path.exists(os.path.join(path, '__init__.py')) or os.path.exists(os.path.join(path, 'plugin.py'))):
                discovered.append(item)
        return sorted(discovered)

    def load_plugins_from_directory(self, enabled_user_plugins: List[str], safe_mode: bool = False, files_locator=None) -> None:
        self.custom_js_snippets = []
        _reset_deepy_plugin_tools()
        if safe_mode:
            print("[Safe Mode] User plugins are disabled. Only system plugins will be loaded.")
            plugins_to_load = SYSTEM_PLUGINS
        else:
            plugins_to_load = SYSTEM_PLUGINS + [p for p in enabled_user_plugins if p not in SYSTEM_PLUGINS]

        for plugin_dir_name in self.discover_plugins():
            if plugin_dir_name not in plugins_to_load:
                continue
            try:
                plugin_path = os.path.join(self.plugins_dir, plugin_dir_name)
                metadata = self._load_plugin_metadata(plugin_path)
                is_bundled = plugin_dir_name in BUNDLED_PLUGINS
                if not os.path.isfile(os.path.join(plugin_path, "plugin.py")):
                    self._register_plugin_spatial_upsamplers(plugin_dir_name, metadata, files_locator=files_locator)
                    self._register_plugin_temporal_upsamplers(plugin_dir_name, metadata, files_locator=files_locator)
                    self._register_plugin_audio_processors(plugin_dir_name, metadata, files_locator=files_locator)
                    continue

                module = importlib.import_module(f"{plugin_dir_name}.plugin")

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, WAN2GPPlugin) and obj != WAN2GPPlugin:
                        plugin = obj()
                        self._apply_metadata_to_plugin(plugin, metadata, plugin_dir_name in SYSTEM_PLUGINS)
                        if is_bundled:
                            plugin.uninstallable = False
                        plugin.setup_ui()
                        self._activate_plugin_deepy_tools(plugin_dir_name, plugin)
                        self.plugins[plugin_dir_name] = plugin
                        if plugin.custom_js_snippets:
                            self.custom_js_snippets.extend(plugin.custom_js_snippets)
                        for hook_name, callbacks in plugin._data_hooks.items():
                            if hook_name not in self.data_hooks:
                                self.data_hooks[hook_name] = []
                            self.data_hooks[hook_name].extend(callbacks)
                        self._register_plugin_spatial_upsamplers(plugin_dir_name, metadata, files_locator=files_locator)
                        self._register_plugin_temporal_upsamplers(plugin_dir_name, metadata, files_locator=files_locator)
                        self._register_plugin_audio_processors(plugin_dir_name, metadata, files_locator=files_locator)
                        if plugin_dir_name not in SYSTEM_PLUGINS:
                            print(f"Loaded plugin: {plugin.name} (from {plugin_dir_name})")
                        break
            except Exception as e:
                print(f"Error loading plugin from directory {plugin_dir_name}: {e}")
                traceback.print_exc()

    def get_all_plugins(self) -> Dict[str, WAN2GPPlugin]:
        return self.plugins.copy()

    def _plugin_overrides_method(self, plugin: WAN2GPPlugin, method_name: str) -> bool:
        plugin_method = getattr(type(plugin), method_name, None)
        base_method = getattr(WAN2GPPlugin, method_name, None)
        return callable(plugin_method) and plugin_method is not base_method

    def _dispatch_plugin_event(self, method_name: str, *args, **kwargs) -> None:
        for plugin_id, plugin in self.plugins.items():
            if not self._plugin_overrides_method(plugin, method_name):
                continue
            try:
                getattr(plugin, method_name)(*args, **kwargs)
            except Exception as e:
                print(f"[PluginManager] Error in {method_name} for plugin {plugin_id}: {e}")
                traceback.print_exc()

    def notify_model_change(self, state: Dict[str, Any], model_type: str) -> None:
        if not model_type:
            return
        self._dispatch_plugin_event("on_model_change", state, model_type)

    def get_custom_js(self) -> str:
        if not self.custom_js_snippets:
            return ""
        return "\n".join(self.custom_js_snippets)

    def inject_globals(self, global_references: Dict[str, Any]) -> None:
        for plugin_id, plugin in self.plugins.items():
            try:
                if 'set_wgp_global' in global_references:
                    plugin._set_wgp_global_func = global_references['set_wgp_global']
                for global_name in plugin.global_requests:
                    if global_name in self.restricted_globals:
                        setattr(plugin, global_name, None)
                    elif global_name in global_references:
                        setattr(plugin, global_name, global_references[global_name])
            except Exception as e:
                print(f"  [!] ERROR injecting globals for {plugin_id}: {str(e)}")

    def update_global_reference(self, global_name: str, new_value: Any) -> None:
        safe_value = None if global_name in self.restricted_globals else new_value
        for plugin_id, plugin in self.plugins.items():
            try:
                if hasattr(plugin, '_global_requests') and global_name in plugin._global_requests:
                    setattr(plugin, global_name, safe_value)
            except Exception as e:
                print(f"  [!] ERROR updating global '{global_name}' for plugin {plugin_id}: {str(e)}")

    def setup_ui(self) -> Dict[str, Dict[str, Any]]:
        tabs = {}
        for plugin_id, plugin in self.plugins.items():
            try:
                for tab_id, tab in plugin.tabs.items():
                    tabs[tab_id] = {
                        'label': tab.label,
                        'component_constructor': tab.component_constructor,
                        'position': tab.position
                    }
            except Exception as e:
                print(f"Error in setup_ui for plugin {plugin_id}: {str(e)}")
        return {'tabs': tabs}
        
    def run_data_hooks(self, hook_name: str, *args, **kwargs):
        if hook_name not in self.data_hooks:
            return kwargs.get('configs')

        callbacks = self.data_hooks[hook_name]
        data = kwargs.get('configs')

        if 'configs' in kwargs:
            kwargs.pop('configs')

        for callback in callbacks:
            try:
                data = callback(data, **kwargs)
            except Exception as e:
                print(f"[PluginManager] Error running hook '{hook_name}' from {callback.__module__}: {e}")
                traceback.print_exc()
        return data
        
    def run_component_insertion_and_setup(self, all_components: Dict[str, Any]):
        all_insert_requests: List[InsertAfterRequest] = []

        for plugin_id, plugin in self.plugins.items():
            try:
                for comp_id in plugin.component_requests:
                    if comp_id in all_components and (not hasattr(plugin, comp_id) or getattr(plugin, comp_id) is None):
                        setattr(plugin, comp_id, all_components[comp_id])

                requested_components = {
                    comp_id: all_components[comp_id]
                    for comp_id in plugin.component_requests
                    if comp_id in all_components
                }
                
                plugin.post_ui_setup(requested_components)
                
                insert_requests = getattr(plugin, '_insert_after_requests', [])
                if insert_requests:
                    all_insert_requests.extend(insert_requests)
                    plugin._insert_after_requests.clear()
                
            except Exception as e:
                print(f"[PluginManager] ERROR in post_ui_setup for {plugin_id}: {str(e)}")
                traceback.print_exc()

        if all_insert_requests:
            for request in all_insert_requests:
                try:
                    target = all_components.get(request.target_component_id)
                    parent = getattr(target, 'parent', None)
                    if not target or not parent or not hasattr(parent, 'children'):
                        print(f"[PluginManager] ERROR: Target '{request.target_component_id}' for insertion not found or invalid.")
                        continue
                        
                    target_index = parent.children.index(target)
                    with parent:
                        new_component = request.new_component_constructor()
                    
                    newly_added = parent.children.pop(-1)
                    parent.children.insert(target_index + 1, newly_added)

                except Exception as e:
                    print(f"[PluginManager] ERROR processing insert_after for {request.target_component_id}: {str(e)}")
                    traceback.print_exc()

class WAN2GPApplication:
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.tab_to_plugin_map: Dict[str, WAN2GPPlugin] = {}
        self.all_rendered_tabs: List[gr.Tab] = []
        self.enabled_plugins: List[str] = []

    def initialize_plugins(self, wgp_globals: dict):
        if not hasattr(self, 'plugin_manager'):
            return

        safe_mode = wgp_globals.get("SAFE_MODE", False)

        if not safe_mode:
            auto_install_and_enable_default_plugins(self.plugin_manager, wgp_globals)
        
        server_config = wgp_globals.get("server_config")
        server_config_filename = wgp_globals.get("server_config_filename", "")
        if not server_config:
            print("[PluginManager] ERROR: server_config not found in globals.")
            return
        self.plugin_manager.set_server_config(server_config, server_config_filename)
        if not safe_mode:
            migrate_bundled_plugin_ids(server_config, server_config_filename)
        if not safe_mode and not server_config.get("motion_designer_bundled_migrated", 0):
            server_config["enabled_plugins"] = server_config.get("enabled_plugins", []) + ([] if "motion_designer" in server_config.get("enabled_plugins", []) else ["motion_designer"]); server_config["motion_designer_bundled_migrated"] = 1
            self.plugin_manager._save_server_config()
        if not safe_mode and not server_config.get("agent_orchestrator_bundled_migrated", 0):
            server_config["enabled_plugins"] = server_config.get("enabled_plugins", []) + ([] if "agent_orchestrator" in server_config.get("enabled_plugins", []) else ["agent_orchestrator"]); server_config["agent_orchestrator_bundled_migrated"] = 1
            self.plugin_manager._save_server_config()
        self.plugin_manager.cleanup_pending_deletions()

        self.enabled_plugins = server_config.get("enabled_plugins", [])

        self.plugin_manager.load_plugins_from_directory(self.enabled_plugins, safe_mode=safe_mode, files_locator=wgp_globals.get("fl"))
        self.plugin_manager.inject_globals(wgp_globals)

    def setup_ui_tabs(self, main_tabs_component: gr.Tabs, state_component: gr.State, set_save_form_event):
        self._create_plugin_tabs(main_tabs_component, state_component)
        self._setup_tab_events(main_tabs_component, state_component, set_save_form_event)
    
    def _create_plugin_tabs(self, main_tabs, state):
        if not hasattr(self, 'plugin_manager'):
            return
        
        loaded_plugins = self.plugin_manager.get_all_plugins()
        system_tabs, user_tabs = [], []
        system_order = {pid: idx for idx, pid in enumerate(SYSTEM_PLUGINS)}

        for plugin_id, plugin in loaded_plugins.items():
            for tab_id, tab in plugin.tabs.items():
                self.tab_to_plugin_map[tab.label] = plugin
                tab_info = {
                    'id': tab_id,
                    'label': tab.label,
                    'component_constructor': tab.component_constructor,
                    'position': system_order.get(plugin_id, tab.position),
                    'plugin_id': plugin_id,
                }
                if plugin_id in SYSTEM_PLUGINS:
                    system_tabs.append(tab_info)
                else:
                    user_tabs.append((plugin_id, tab_info))

        system_tabs_sorted = sorted(
            system_tabs,
            key=lambda t: (system_order.get(t['plugin_id'], 1_000_000), t['label']),
        )
        pre_user_tabs = system_tabs_sorted[:USER_PLUGIN_INSERT_POSITION]
        post_user_tabs = system_tabs_sorted[USER_PLUGIN_INSERT_POSITION:]

        sorted_user_tabs = [tab_info for plugin_id in self.enabled_plugins for pid, tab_info in user_tabs if pid == plugin_id]

        all_tabs_to_render = pre_user_tabs + sorted_user_tabs + post_user_tabs

        def goto_media_tab(state):
            self._handle_tab_selection(state, None)
            return  gr.Tabs(selected="media_gen")
        

        for tab_info in all_tabs_to_render:
            with gr.Tab(tab_info['label'], id=f"plugin_{tab_info['id']}") as new_tab:
                self.all_rendered_tabs.append(new_tab)
                plugin = self.tab_to_plugin_map[new_tab.label]
                plugin.goto_media_tab = goto_media_tab
                constructor = tab_info['component_constructor']
                if self._component_constructor_accepts_api(constructor):
                    if not hasattr(plugin, "_wangp_session") or getattr(plugin, "_wangp_session", None) is None:
                        from shared.api import create_gradio_webui_session

                        plugin._wangp_session = create_gradio_webui_session(plugin)
                    with plugin._wangp_session.plugin_ui_context():
                        constructor(plugin._wangp_session)
                else:
                    constructor()

    @staticmethod
    def _component_constructor_accepts_api(component_constructor: callable) -> bool:
        try:
            parameters = list(inspect.signature(component_constructor).parameters.values())
        except (TypeError, ValueError):
            return False
        if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
            return True
        positional_parameters = [
            parameter
            for parameter in parameters
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        return len(positional_parameters) >= 1


    def _setup_tab_events(self, main_tabs_component: gr.Tabs, state_component: gr.State, set_save_form_event):
        if main_tabs_component and state_component:
            main_tabs_component.select(
                fn=self._handle_tab_selection,
                inputs=[state_component],
                outputs=None,
                show_progress="hidden",
            )


            for tab in self.all_rendered_tabs:
                # def test_tab(state_component, evt: gr.SelectData):
                #     last_save_form = state_component.get("last_save_form", media_gen_label)
                #     if last_save_form != media_gen_label :
                #         state_component["ignore_save_form"] = True
                #     else:
                #         state_component["last_save_form"] = evt.value


                plugin = self.tab_to_plugin_map[tab.label]
                # event = tab.select(fn=test_tab, inputs=[state_component])
                # event = set_save_form_event(event.then)
                event = set_save_form_event(tab.select)
                event.then(
                        fn=self._handle_one_tab_selection,
                        inputs=[state_component, gr.State(tab.label)],
                        outputs=plugin.on_tab_outputs if hasattr(plugin, "on_tab_outputs") else None,
                        show_progress="hidden",
                        trigger_mode="multiple",
                    )

            
    def _handle_tab_selection(self, state: dict, evt: gr.SelectData):
        if not hasattr(self, 'previous_tab_id'):
            self.previous_tab_id = media_gen_label
        
        new_tab_id = media_gen_label if evt is None else evt.value
        
        if self.previous_tab_id == new_tab_id:
            return

        if self.previous_tab_id and self.previous_tab_id in self.tab_to_plugin_map:
            plugin_to_deselect = self.tab_to_plugin_map[self.previous_tab_id]
            try:
                plugin_to_deselect.on_tab_deselect(state)
            except Exception as e:
                print(f"[PluginManager] Error in on_tab_deselect for plugin {plugin_to_deselect.name}: {e}")
                traceback.print_exc()

        # if new_tab_id and new_tab_id in self.tab_to_plugin_map:
            # plugin_to_select = self.tab_to_plugin_map[new_tab_id]
            # if not hasattr(plugin_to_select, "on_tab_outputs"):
            #     try:
            #         plugin_to_select.on_tab_select(state)
            #     except Exception as e:
            #         print(f"[PluginManager] Error in on_tab_select for plugin {plugin_to_select.name}: {e}")
            #         traceback.print_exc()

        self.previous_tab_id = new_tab_id

    def _handle_one_tab_selection(self, state: dict, new_tab_id): #, evt: gr.SelectData
        plugin_to_select = self.tab_to_plugin_map.get(new_tab_id, None)
        try:
            ret = plugin_to_select.on_tab_select(state)
        except Exception as e:
            print(f"[PluginManager] Error in on_tab_select for plugin {plugin_to_select.name}: {e}")
            traceback.print_exc()
            ret = None
        return ret
    
    def run_component_insertion(self, components_dict: Dict[str, Any]):
        if hasattr(self, 'plugin_manager'):
            self.plugin_manager.run_component_insertion_and_setup(components_dict)
