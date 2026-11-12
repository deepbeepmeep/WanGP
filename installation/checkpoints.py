"""Remove a saved version and reclaim only environments created by this installation."""

from contextlib import ExitStack
from pathlib import Path
import re
import shutil

from . import conda_transactions as imported, transactions as managed
from .environments import read_registry
from .execute import ProcessRunner
from .state import InstallError, InstallationLock, atomic_json, digest, inside, plain_path, read_json, sync_directory


def saved_version(root, current, key):
    folder = "checkpoints" if current["ownership"] == "managed" else "conda-checkpoints"
    path = inside(root, f".manager/{folder}/{managed.identifier(key)}.json")
    record = read_json(path)
    if (record["root"], record["installation_id"], record["ownership"]) != (current["root"], current["installation_id"], current["ownership"]):
        raise InstallError("invalid_checkpoint", "Version history contains a foreign checkpoint")
    return record["active"]


def owned_environment(root, current, active):
    is_managed = current["ownership"] == "managed"
    environment = plain_path(root / active["environment"])
    directory = inside(root, "environments" if is_managed else ".manager/conda-envs")
    pattern = r"[0-9a-f]{32}" if is_managed else r"[0-9a-f]{12}"
    if environment.parent != directory or not re.fullmatch(pattern, environment.name):
        return None
    # Legacy installations already have a durable creation journal. A path or
    # folder name alone is insufficient evidence that Manager owns its contents.
    operations = inside(root, ".manager/operations" if is_managed else ".manager/conda-operations")
    for operation in operations.glob(environment.name + "*"):
        path = inside(operation, "journal.json")
        if not path.is_file():
            continue
        journal = read_json(path)
        plan = journal["plan"]
        if (journal.get("next_record") or {}).get("installation_id") != current["installation_id"]:
            continue
        if journal["journal_sha256"] != digest({k: v for k, v in journal.items() if k != "journal_sha256"}):
            raise InstallError("invalid_journal", "Could not verify the environment's creation record")
        if (journal["status"] == "succeeded" and plan["operation_id"] == operation.name and not plan["reuse_environment"]
                and plain_path(plan["installation_root"]) == root and plain_path(root / plan["environment"]) == environment):
            return environment.resolve()
    return None


def references_environment(root, current, candidate):
    app = root / "app" if current and current["ownership"] == "managed" else root
    if current:
        if (root / current["active"]["environment"]).resolve().is_relative_to(candidate):
            return True
        if any((root / saved_version(root, current, key)["environment"]).resolve().is_relative_to(candidate) for key in current["history"]):
            return True
    # Named CLI environments can be selected independently of rollback history.
    return any((app / entry["path"]).resolve().is_relative_to(candidate)
               for entry in read_registry(app)["envs"].values() if entry["type"] != "none")


def shared_elsewhere(root, candidate, registrations, locks):
    for item in registrations:
        other = plain_path(item["root"])
        if other == root:
            continue
        if item.get("python") and Path(item["python"]).parent.resolve().is_relative_to(candidate):  # a venv's python may link outside it
            return True
        # An unavailable/busy installation may acquire or retain a reference.
        # Leave cleanup retryable until we can hold its lease through deletion.
        if not other.is_dir():
            raise InstallError("installation_unavailable", f"Cannot check a registered installation at {other}. Restore access and retry cleanup.")
        if (other / ".manager").is_dir():
            locks.enter_context(InstallationLock(other))
            if managed.active_operation(other) or imported.active_operation(other):
                raise InstallError("recovery_required", f"Finish recovery at {other} before retrying cleanup")
        current = managed.read_installation(other) or imported.read_record(other)
        if references_environment(other, current, candidate):
            return True
    return False


def remove_version(root, checkpoint, registrations=()):
    root, checkpoint = plain_path(root), managed.identifier(checkpoint)
    with InstallationLock(root), ExitStack() as locks:
        if managed.active_operation(root) or imported.active_operation(root):
            raise InstallError("recovery_required", "Recover the interrupted update before removing a saved version")
        current = managed.read_installation(root) or imported.read_record(root)
        if not current:
            raise InstallError("unknown_checkpoint", "This installation has no saved versions")
        is_managed = current["ownership"] == "managed"
        record_path = managed.record_path(root) if is_managed else imported.record_path(root)
        pending = current.get("removing_checkpoint")
        if pending and pending["checkpoint"] != checkpoint:
            raise InstallError("cleanup_pending", "Finish removing the previous saved version first")
        if not pending:
            if checkpoint not in current["history"]:
                raise InstallError("unknown_checkpoint", "Select a saved rollback version")
            active = saved_version(root, current, checkpoint)
            current["history"].remove(checkpoint)
            current["generation"] += 1
            current["removing_checkpoint"] = {"checkpoint": checkpoint, "release_id": active["release_id"], "environment": active["environment"]}
        environment = owned_environment(root, current, current["removing_checkpoint"])
        reclaim = environment is not None and not references_environment(root, current, environment) and not shared_elsewhere(root, environment, registrations, locks)
        if not pending:
            # Retire the reference before deleting anything. A crash or locked
            # file leaves a retryable cleanup, never a selectable broken rollback.
            atomic_json(record_path, current)
        if reclaim and environment.exists():
            shutil.rmtree(environment)
        if not is_managed:
            imported.git(root, ProcessRunner(), "update-ref", "-d", f"refs/wangp-manager/{current['installation_id']}/{checkpoint}")
        folder = "checkpoints" if is_managed else "conda-checkpoints"
        path = inside(root, f".manager/{folder}/{checkpoint}.json")
        path.unlink(missing_ok=True)
        sync_directory(path.parent)
        del current["removing_checkpoint"]
        atomic_json(record_path, current)
        return {"status": "succeeded", "environment_removed": reclaim}
