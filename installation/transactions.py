"""Journaled managed installs, updates and rollback of source/environment pairs."""

import base64
import json
from pathlib import Path
import re
import shutil
import uuid

from .execute import Cancelled, ProcessRunner
from .releases import cache_release, load_release
from .resolution import load_resolution
from .runtime import EnvironmentService
from .state import InstallError, InstallationLock, atomic_bytes, atomic_copy, atomic_json, digest, file_hash, inside, plain_path, read_json, sync_directory


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{32}", value):
        raise InstallError("invalid_id", "Invalid operation/checkpoint ID")
    return value


def hash_id(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise InstallError("invalid_id", "Invalid release/resolution digest")
    return value


def record_path(root):
    return inside(root, ".manager/installation.json")


def active_operation(root):
    path = inside(root, ".manager/active_operation.json")
    return identifier(read_json(path)["operation_id"]) if path.exists() else None


def validate_record(root, data):
    if data["schema_version"] != 1 or plain_path(data["root"]) != root or data["ownership"] != "managed":
        raise InstallError("invalid_installation", "Installation ownership record does not match this root")
    identifier(data["installation_id"])
    active = data["active"]
    hash_id(active["release"])
    hash_id(active["resolution"])
    if not re.fullmatch("environments/[0-9a-f]{32}", active["environment"]):
        raise InstallError("invalid_installation", "Invalid managed environment location")
    for checkpoint in data["history"]:
        identifier(checkpoint)
    return data


def read_installation(root):
    root = plain_path(root)
    path = record_path(root)
    return validate_record(root, read_json(path)) if path.exists() else None


def release_location(root, release_hash):
    return inside(root, f".manager/releases/{hash_id(release_hash)}")


def resolution_location(root, resolution_hash):
    return inside(root, f".manager/resolutions/{hash_id(resolution_hash)}")


def file_observation(root, filename):
    path = inside(root, filename)
    if path.exists():
        if not path.is_file():
            raise InstallError("source_conflict", f"Expected a file at {path}")
        return file_hash(path)
    return None


def clear_bytecode(app, filename):
    source = inside(app, filename)
    if source.suffix == ".py":
        cache = plain_path(source.parent / "__pycache__")
        if cache.exists():
            for entry in cache.glob(source.stem + ".*.pyc"):
                plain_path(entry).unlink()
            sync_directory(cache)


def assert_idle(root):
    operation = active_operation(root)
    if operation:
        raise InstallError("recovery_required", f"Operation {operation} must finish or be recovered before another action")


def ready_record(root):
    assert_idle(root)
    record = read_installation(root)
    if record is None:
        raise InstallError("not_managed", "No managed installation exists at this root")
    registry = read_json(inside(root, "app/envs.json"))
    active_name = registry.get("active")
    allowed_names = {"WanGP", "manager-" + Path(record["active"]["environment"]).name}
    expected_path = str(inside(root, record["active"]["environment"]))
    if active_name not in allowed_names or registry.get("envs", {}).get(active_name) != {"type": "uv", "path": expected_path}:
        raise InstallError("inconsistent_activation", "Installation record and CLI environment projection disagree")
    return record


def _prepare(root, release, resolution, operation, operation_id, checkpoint=None, environment_policy="auto", *, emit=None, check_cancelled=None, environment_service=None):
    root = plain_path(root)
    assert_idle(root)
    if any((parent / ".manager/installation.json").exists() or (parent / ".manager/active_operation.json").exists() for parent in root.parents):
        raise InstallError("nested_installation", "A managed installation cannot be nested inside another managed installation")
    current = read_installation(root)
    if current and current.get("removing_checkpoint"):
        raise InstallError("cleanup_pending", "Finish removing the saved version before updating")
    if environment_policy not in ("auto", "current", "new"):
        raise InstallError("invalid_environment_policy", "Choose auto, current or new for the target environment")
    if operation in ("update", "switch-version") and current:
        ready_record(root)
    if operation == "install" and current is not None:
        raise InstallError("already_managed", "Use an update plan for this managed installation")
    if operation != "install" and current is None:
        raise InstallError("not_managed", "Existing external installations cannot be updated or adopted implicitly")
    target_active = None
    if operation == "rollback":
        checkpoint = identifier(checkpoint)
        if checkpoint not in current["history"]:
            raise InstallError("unknown_checkpoint", "Checkpoint is not in this installation's rollback history")
        target = validate_record(root, read_json(inside(root, f".manager/checkpoints/{checkpoint}.json")))
        if target["installation_id"] != current["installation_id"] or target["root"] != current["root"]:
            raise InstallError("invalid_checkpoint", "Rollback checkpoint belongs to another installation")
        target_active = target["active"]
        release = release_location(root, target_active["release"])
        resolution = resolution_location(root, target_active["resolution"])
    release, resolution = plain_path(release), plain_path(resolution)
    manifest = load_release(release, emit=emit, check_cancelled=check_cancelled)
    locked = load_resolution(resolution, check_inputs=operation != "rollback")
    if locked["requires_gpu"] and locked["profile"] != "MPS" and not locked["selected_device"]:
        raise InstallError("device_required", "Select a hardware device in the catalog plan before preparing a GPU installation")
    source_root = plain_path(locked["source_root"])
    # Resolution inputs inside its source root must be present in the release with matching bytes.
    for filename, expected in locked["input_files"].items():
        path = plain_path(filename)
        if path.is_relative_to(source_root):
            relative = path.relative_to(source_root).as_posix()
            if manifest["files"].get(relative) != expected:
                raise InstallError("resolution_mismatch", f"Release does not contain the resolved input: {relative}")
    config_schema = current["config_schema"] if current else manifest["config_schema"]
    if config_schema not in manifest["compatible_config_schemas"]:
        raise InstallError("config_migration_required", "This release cannot read the existing configuration schema; a reviewed migration is required")
    old_files = load_release(release_location(root, current["active"]["release"]), emit=emit, check_cancelled=check_cancelled)["files"] if current else {}
    app = inside(root, "app")
    if current is None and app.exists() and any(app.iterdir()):
        raise InstallError("external_installation", "New managed installation requires an empty app directory")
    observations, changes = {}, []
    if emit:
        emit({"type": "progress", "title": "Comparing Installation Files", "message": "Checking the installation for changed or conflicting files"})
    for filename in sorted(set(old_files) | set(manifest["files"])):
        if check_cancelled:
            check_cancelled()
        # Validate future cache and recovery paths before changing the installation.
        inside(root, f".manager/releases/{manifest['manifest_sha256']}/source/{filename}")
        inside(root, f".manager/operations/{operation_id}/backup/{filename}")
        observed = file_observation(app, filename)
        observations[filename] = observed
        old, new = old_files.get(filename), manifest["files"].get(filename)
        if old is None and observed is not None:
            raise InstallError("source_conflict", f"Incoming source collides with an unowned file: {filename}")
        if old is not None and observed != old:
            if not (operation == "repair" and observed is None and new is not None):
                raise InstallError("source_conflict", f"Locally modified or missing owned source: {filename}")
        if observed != new:
            changes.append({"path": filename, "before": observed, "after": new})
    environment = f"environments/{operation_id}"
    reuse = False
    if target_active:
        environment, reuse = target_active["environment"], True
    elif current and operation != "repair" and environment_policy != "new":
        if environment_service is None:
            reuse = current["active"]["dependency_fingerprint"] == locked["dependency_fingerprint"]
        else:
            prefix = inside(root, current["active"]["environment"])
            previous_dir = resolution_location(root, current["active"]["resolution"])
            reuse = environment_service.satisfies(prefix, resolution, locked, environment_service.inventory(prefix), previous_dir)
        if reuse:
            environment = current["active"]["environment"]
    if environment_policy == "current" and (not current or not reuse or environment != current["active"]["environment"]):
        raise InstallError("incompatible_environment", "This version needs different dependencies. Choose auto or new to preserve the current environment and build a separate one")
    registry = inside(root, "app/envs.json")
    config = inside(root, "app/wgp_config.json")
    result = {
        "schema_version": 1, "operation": operation, "operation_id": identifier(operation_id), "installation_root": str(root),
        "release_directory": str(release), "resolution_directory": str(resolution), "release_sha256": manifest["manifest_sha256"],
        "resolution_sha256": locked["resolution_sha256"], "previous_record_sha256": digest(current),
        "registry_sha256": file_hash(registry) if registry.exists() else None,
        "config_sha256": file_hash(config) if config.exists() else None, "config_schema": config_schema,
        "source_observations": observations, "changes": changes, "environment": environment, "reuse_environment": reuse, "environment_policy": environment_policy,
        "previous_release_id": current["active"]["release_id"] if current else None, "target_release_id": manifest["release_id"],
        "rollback_checkpoint": checkpoint, "target_active": target_active,
        "trust": "explicit_local_release", "apply_supported": True,
    }
    if environment_service is not None and operation in ("update", "switch-version"):
        result["check_installed_packages"] = True
    result["plan_sha256"] = digest(result)
    return result


def prepare_transaction(root, release=None, resolution=None, operation="install", checkpoint=None, environment_policy="auto", *, emit=None, check_cancelled=None, environment_service=None):
    if operation not in ("install", "update", "repair", "rollback", "switch-version"):
        raise InstallError("invalid_operation", f"Unsupported transaction: {operation}")
    with InstallationLock(root):
        return _prepare(root, release, resolution, operation, uuid.uuid4().hex, checkpoint, environment_policy, emit=emit, check_cancelled=check_cancelled, environment_service=environment_service)


def _cache_resolution(source, destination):
    record = load_resolution(source)
    if destination.exists():
        if (destination / "resolution.json").exists() and load_resolution(destination, check_inputs=False)["resolution_sha256"] != record["resolution_sha256"]:
            raise InstallError("changed_input", "Cached dependency resolution differs")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in record["files"].items():
        target = inside(destination, relative)
        if target.exists():
            if file_hash(target) != expected:
                raise InstallError("changed_input", f"Partial dependency artifact differs: {relative}")
        else:
            atomic_copy(inside(source, relative), target, expected)
    atomic_json(destination / "resolution.json", record)


class TransactionEngine:
    def __init__(self, runtime, events=None, checkpoint_hook=None, environment_service=None, cancelled=None):
        self.runtime = runtime
        self.events = events or (lambda event: None)
        self.checkpoint_hook = checkpoint_hook or (lambda name: None)
        self.environment_service = environment_service
        self.cancelled = cancelled or (lambda: False)

    def _save(self, operation, journal, name):
        journal["sequence"] += 1
        journal["journal_sha256"] = digest({key: value for key, value in journal.items() if key != "journal_sha256"})
        atomic_json(operation / "journal.json", journal)
        self.checkpoint_hook(name)

    def _phase(self, operation, journal, phase):
        journal["phase"] = phase
        self._save(operation, journal, phase)
        self.events({"type": "progress", "operation_id": journal["operation_id"], "sequence": journal["sequence"], "phase": phase, "title": phase.replace("_", " ").title()})

    def _finish(self, root, operation, journal, status, error=None):
        journal.update(status=status, error=error)
        journal["result"] = {"operation_id": journal["operation_id"], "status": status, "error": error}
        self._save(operation, journal, "terminal_" + status)
        pointer = inside(root, ".manager/active_operation.json")
        if pointer.exists():
            if read_json(pointer)["operation_id"] != journal["operation_id"]:
                raise InstallError("recovery_conflict", "Active operation pointer changed")
            pointer.unlink()
            sync_directory(pointer.parent)
        self.events({"type": "result", **journal["result"], "sequence": journal["sequence"]})
        return journal["result"]

    def apply(self, plan):
        if plan.get("schema_version") != 1 or plan.get("plan_sha256") != digest({key: value for key, value in plan.items() if key != "plan_sha256"}):
            raise InstallError("changed_plan", "Transaction plan was modified")
        root = plain_path(plan["installation_root"])
        with InstallationLock(root):
            checker = None
            if plan.get("check_installed_packages"):
                checker = self.environment_service or EnvironmentService(self.runtime, ProcessRunner(events=self.events, cancelled=self.cancelled))
            expected = _prepare(root, plan["release_directory"], plan["resolution_directory"], plan["operation"], plan["operation_id"], plan["rollback_checkpoint"], plan["environment_policy"], environment_service=checker)
            if expected != plan:
                raise InstallError("stale_plan", "Installation state changed after preview; prepare a new plan")
            previous = read_installation(root)
            operation = inside(root, f".manager/operations/{identifier(plan['operation_id'])}")
            operation.mkdir(parents=True, exist_ok=False)
            registry = inside(root, "app/envs.json")
            registry_before = registry.read_bytes() if registry.exists() else None
            journal = {"schema_version": 1, "operation_id": plan["operation_id"], "plan": plan, "status": "running", "phase": "preparing", "sequence": 0, "previous_record": previous, "previous_registry": base64.b64encode(registry_before).decode() if registry_before is not None else None}
            self._save(operation, journal, "journal_created")
            atomic_json(inside(root, ".manager/active_operation.json"), {"operation_id": plan["operation_id"]})
            runner = ProcessRunner(operation / "operation.log", cancelled=lambda: self.cancelled() or (operation / "cancel").exists(), events=self.events)
            service = self.environment_service or EnvironmentService(self.runtime, runner, cache_directory=inside(root, ".manager/cache/uv"))
            try:
                runner.check_cancelled()
                release = release_location(root, plan["release_sha256"])
                self._phase(operation, journal, "staging_release")
                manifest = cache_release(plan["release_directory"], release)
                resolved_dir = resolution_location(root, plan["resolution_sha256"])
                if plan["operation"] != "rollback":
                    _cache_resolution(Path(plan["resolution_directory"]), resolved_dir)
                resolution = load_resolution(resolved_dir, check_inputs=False)
                environment = inside(root, plan["environment"])
                checkpoint_record = previous
                if previous and plan["operation"] not in ("rollback", "repair"):
                    old_resolution = load_resolution(resolution_location(root, previous["active"]["resolution"]), check_inputs=False)
                    # Plugins may add packages after installation. Save the actual,
                    # healthy environment for rollback rather than its old inventory.
                    current_validation = service.validate(inside(root, previous["active"]["environment"]), old_resolution)
                    checkpoint_record = {**previous, "active": {**previous["active"], "validation": current_validation}}
                self._phase(operation, journal, "validating_environment" if plan["reuse_environment"] else "building_environment")
                if plan["reuse_environment"]:
                    expected_inventory = plan["target_active"]["validation"]["inventory"] if plan["target_active"] else checkpoint_record["active"]["validation"]["inventory"]
                    validation = service.validate(environment, resolution, expected_inventory)
                else:
                    validation = service.build(environment, resolved_dir, resolution)
                runner.check_cancelled()
                # Recheck mutable source, registry and config immediately before backup/activation.
                for filename, observed in plan["source_observations"].items():
                    if file_observation(inside(root, "app"), filename) != observed:
                        raise InstallError("stale_plan", f"Source changed during preparation: {filename}")
                for relative, expected_hash in (("app/envs.json", plan["registry_sha256"]), ("app/wgp_config.json", plan["config_sha256"])):
                    if file_observation(root, relative) != expected_hash:
                        raise InstallError("stale_plan", f"Configuration changed during preparation: {relative}")
                self._phase(operation, journal, "backing_up")
                needed = sum(inside(root, "app/" + item["path"]).stat().st_size for item in plan["changes"] if item["before"])
                needed += sum(inside(release / "source", item["path"]).stat().st_size for item in plan["changes"] if item["after"])
                if shutil.disk_usage(root).free < needed + 1024 * 1024:
                    raise InstallError("disk_space", "Insufficient space for activation and its recovery files")
                for item in plan["changes"]:
                    if item["before"]:
                        atomic_copy(inside(root, "app/" + item["path"]), inside(operation / "backup", item["path"]), item["before"])
                if previous:
                    atomic_json(inside(root, f".manager/checkpoints/{plan['operation_id']}.json"), checkpoint_record)
                history = [*previous["history"], plan["operation_id"]] if previous else []
                next_record = {
                    "schema_version": 1, "ownership": "managed", "root": str(root),
                    "installation_id": previous["installation_id"] if previous else uuid.uuid4().hex,
                    "generation": previous["generation"] + 1 if previous else 1, "config_schema": plan["config_schema"], "history": history,
                    "active": {"release": manifest["manifest_sha256"], "release_id": manifest["release_id"], "resolution": resolution["resolution_sha256"], "dependency_fingerprint": resolution["dependency_fingerprint"], "environment": plan["environment"], "validation": validation},
                }
                projected = json.loads(registry_before) if registry_before else {"active": None, "envs": {}}
                name = "WanGP"
                projected["envs"][name] = {"type": "uv", "path": str(environment)}
                projected["active"] = name
                journal.update(next_record=next_record, next_registry=projected)
                self._phase(operation, journal, "activating")
                for index, item in enumerate(plan["changes"]):
                    runner.check_cancelled()
                    target = inside(root, "app/" + item["path"])
                    if file_observation(inside(root, "app"), item["path"]) != item["before"]:
                        raise InstallError("source_conflict", f"Source changed before activation: {item['path']}")
                    if item["after"]:
                        atomic_copy(inside(release / "source", item["path"]), target, item["after"])
                    else:
                        target.unlink()
                        sync_directory(target.parent)
                    clear_bytecode(inside(root, "app"), item["path"])
                    self.checkpoint_hook(f"source_{index}")
                atomic_json(record_path(root), next_record)
                self.checkpoint_hook("record_activated")
                atomic_json(registry, projected)
                self.checkpoint_hook("registry_activated")
                self._phase(operation, journal, "checking_activation")
                service.validate(environment, resolution, validation["inventory"])
                self._verify_activation(root, journal)
                runner.check_cancelled()
                self._phase(operation, journal, "committed")
                return self._finish(root, operation, journal, "succeeded")
            except (Exception, KeyboardInterrupt) as exc:
                error = {"code": "cancelled" if isinstance(exc, (Cancelled, KeyboardInterrupt)) else getattr(exc, "code", "operation_failed"), "message": str(exc)}
                if journal["phase"] == "committed":
                    journal.update(status="recovery_required", error=error)
                    self._save(operation, journal, "finalization_required")
                    raise InstallError("recovery_required", "Activation committed; recover the operation to finish recording its result") from exc
                try:
                    if journal["phase"] in ("activating", "checking_activation"):
                        self._restore(root, operation, journal)
                    return self._finish(root, operation, journal, "cancelled" if error["code"] == "cancelled" else "failed", error)
                except Exception as recovery_error:
                    journal.update(status="recovery_required", error=error, recovery_error=str(recovery_error))
                    self._save(operation, journal, "recovery_required")
                    raise InstallError("recovery_required", f"Recovery requires attention: {recovery_error}") from exc

    def _verify_activation(self, root, journal):
        if read_installation(root) != journal["next_record"] or read_json(inside(root, "app/envs.json")) != journal["next_registry"]:
            raise InstallError("inconsistent_activation", "Activation metadata does not match its journal")
        manifest = load_release(release_location(root, journal["next_record"]["active"]["release"]))
        for filename, expected in manifest["files"].items():
            if file_observation(inside(root, "app"), filename) != expected:
                raise InstallError("source_conflict", f"Activated source differs: {filename}")
        for item in journal["plan"]["changes"]:
            if item["after"] is None and file_observation(inside(root, "app"), item["path"]) is not None:
                raise InstallError("source_conflict", f"Removed source reappeared: {item['path']}")

    def _restore(self, root, operation, journal):
        for item in reversed(journal["plan"]["changes"]):
            target = inside(root, "app/" + item["path"])
            current = file_observation(inside(root, "app"), item["path"])
            if current not in (item["before"], item["after"]):
                raise InstallError("recovery_conflict", f"Recovery would overwrite a later edit: {item['path']}")
            if current == item["before"]:
                clear_bytecode(inside(root, "app"), item["path"])
                continue
            if item["before"]:
                atomic_copy(inside(operation / "backup", item["path"]), target, item["before"])
            elif target.exists():
                target.unlink()
                sync_directory(target.parent)
            clear_bytecode(inside(root, "app"), item["path"])
        current = read_installation(root)
        if current not in (journal["previous_record"], journal["next_record"]):
            raise InstallError("recovery_conflict", "Installation record changed outside this operation")
        if journal["previous_record"] is None:
            if record_path(root).exists():
                record_path(root).unlink()
                sync_directory(record_path(root).parent)
        else:
            atomic_json(record_path(root), journal["previous_record"])
        registry = inside(root, "app/envs.json")
        old = base64.b64decode(journal["previous_registry"]) if journal["previous_registry"] is not None else None
        present = registry.read_bytes() if registry.exists() else None
        if present != old and (present is None or json.loads(present) != journal["next_registry"]):
            raise InstallError("recovery_conflict", "CLI environment record changed outside this operation")
        if old is None:
            if registry.exists():
                registry.unlink()
                sync_directory(registry.parent)
        else:
            atomic_bytes(registry, old)
        self.checkpoint_hook("restored")

    def recover(self, root):
        root = plain_path(root)
        with InstallationLock(root):
            operation_id = active_operation(root)
            if operation_id is None:
                return {"status": "idle"}
            operation = inside(root, f".manager/operations/{operation_id}")
            journal = read_json(operation / "journal.json")
            if journal.get("journal_sha256") != digest({key: value for key, value in journal.items() if key != "journal_sha256"}):
                raise InstallError("invalid_journal", "Operation journal integrity check failed; no recovery files were changed")
            if journal["operation_id"] != operation_id or plain_path(journal["plan"]["installation_root"]) != root:
                raise InstallError("invalid_journal", "Operation journal belongs to another installation")
            if journal["plan"]["plan_sha256"] != digest({key: value for key, value in journal["plan"].items() if key != "plan_sha256"}):
                raise InstallError("invalid_journal", "Journal plan integrity check failed")
            if journal["status"] in ("succeeded", "failed", "cancelled"):
                return self._finish(root, operation, journal, journal["status"], journal.get("error"))
            if journal["phase"] == "committed":
                self._verify_activation(root, journal)
                return self._finish(root, operation, journal, "succeeded")
            try:
                if journal["phase"] in ("activating", "checking_activation"):
                    self._restore(root, operation, journal)
                return self._finish(root, operation, journal, "failed", {"code": "interrupted", "message": "Interrupted operation recovered; the previous installation was preserved or restored"})
            except Exception as exc:
                journal.update(status="recovery_required", recovery_error=str(exc))
                self._save(operation, journal, "recovery_required")
                raise


def operation_status(root, operation_id=None):
    root = plain_path(root)
    operation_id = identifier(operation_id) if operation_id else active_operation(root)
    if operation_id is None:
        return {"status": "idle", "installation": read_installation(root)}
    return read_json(inside(root, f".manager/operations/{operation_id}/journal.json"))


def version_history(root):
    """List local recovery points; version labels are not assumed to be sortable."""
    with InstallationLock(root):
        current = ready_record(root)
        records = [(None, current)]
        for checkpoint in reversed(current["history"]):
            saved = validate_record(plain_path(root), read_json(inside(root, f".manager/checkpoints/{identifier(checkpoint)}.json")))
            if saved["installation_id"] != current["installation_id"] or saved["root"] != current["root"]:
                raise InstallError("invalid_checkpoint", "Version history contains a foreign checkpoint")
            records.append((checkpoint, saved))
        versions = [{"checkpoint": checkpoint, "current": checkpoint is None, "release_id": record["active"]["release_id"], "release_sha256": record["active"]["release"], "environment": record["active"]["environment"], "config_schema": record["config_schema"]} for checkpoint, record in records]
        if current.get("removing_checkpoint"):
            versions.append({**current["removing_checkpoint"], "current": False, "removing": True})
        return {"versions": versions}


def cancel_operation(root, operation_id):
    root, operation_id = plain_path(root), identifier(operation_id)
    status = operation_status(root, operation_id)
    if status["status"] not in ("running", "recovery_required"):
        return {"status": status["status"], "operation_id": operation_id}
    atomic_bytes(inside(root, f".manager/operations/{operation_id}/cancel"), b"cancel\n")
    return {"status": "cancellation_requested", "operation_id": operation_id}
