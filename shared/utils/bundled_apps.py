# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""One-time discovery of the included creative apps; enabled_plugins owns activation."""
from shared.utils.config_store import config_lock, write_config

FEATURED_APPS = ("media_flow", "showrunner")
REVIEW_KEY = "bundled_apps_reviewed"
VISITED_KEY = "bundled_apps_visited"


def migrate_featured_apps(config, filename):
    with config_lock:
        reviewed = config.setdefault(REVIEW_KEY, {})
        enabled = config.get("enabled_plugins", []).copy()
        installed = set(config.get("installed_remote_plugins", []))
        changed = False
        for plugin_id in FEATURED_APPS:
            if plugin_id in reviewed:
                continue
            reviewed[plugin_id] = plugin_id in enabled or plugin_id in installed
            if plugin_id not in enabled:
                enabled.append(plugin_id)
            changed = True
        if "models_manager" in enabled:
            enabled.remove("models_manager")
            changed = True
        if changed:
            config["enabled_plugins"] = enabled
            write_config(config, filename)


def review_featured_app(config, filename, plugin_id, keep):
    if plugin_id not in FEATURED_APPS:
        raise ValueError("Unknown included app")
    with config_lock:
        config.setdefault(REVIEW_KEY, {})[plugin_id] = True
        enabled = config.get("enabled_plugins", []).copy()
        if keep and plugin_id not in enabled:
            enabled.append(plugin_id)
        elif not keep and plugin_id in enabled:
            enabled.remove(plugin_id)
        config["enabled_plugins"] = enabled
        write_config(config, filename)


def visit_featured_app(config, filename, plugin_id):
    if plugin_id not in FEATURED_APPS:
        raise ValueError("Unknown included app")
    with config_lock:
        visited = config.setdefault(VISITED_KEY, [])
        if plugin_id in visited:
            return False
        visited.append(plugin_id)
        write_config(config, filename)
        return True
