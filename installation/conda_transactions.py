"""Update existing Git checkouts with retained Conda/venv environments and revisions.

The original Conda record paths remain stable for installations already adopted.
"""

import base64
import json
from pathlib import Path
import re
import uuid

from .conda import CondaEnvironmentService, conda_command
from .environments import clean_environment, inspect_interpreter, interpreter_path
from .execute import Cancelled, ProcessRunner
from .releases import load_release, source_path
from .resolution import load_resolution
from .runtime import EnvironmentService
from .state import InstallError, InstallationLock, atomic_bytes, atomic_json, digest, file_hash, inside, plain_path, read_json
from .transactions import _cache_resolution, identifier
from .version import local_version


def record_path(root):
    return inside(root, ".manager/conda.json")


def read_record(root):
    path = record_path(root)
    if not path.exists():
        return None
    record = read_json(path)
    if record["schema_version"] != 1 or record["ownership"] not in ("conda", "venv") or plain_path(record["root"]) != plain_path(root):
        raise InstallError("invalid_installation", "Installation record belongs to another folder")
    identifier(record["installation_id"])
    revision(record["active"]["commit"])
    return record


def active_operation(root):
    path = inside(root, ".manager/conda-active.json")
    return identifier(read_json(path)["operation_id"]) if path.exists() else None


def revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise InstallError("invalid_revision", "Choose an exact Git revision")
    return value


def git(root, runner, *arguments, check=True):
    env = clean_environment()
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    # Checkout hooks must not install packages or change user data during activation.
    return runner.run(["git", "-c", "core.hooksPath=" + str(inside(root, ".manager/no-hooks")), "-C", str(root), *arguments], env=env, capture=True, check=check, timeout=300, emit_output=False).stdout.strip()


def git_state(root, runner):
    if plain_path(git(root, runner, "rev-parse", "--show-toplevel")) != plain_path(root):
        raise InstallError("git_required", "Choose the root of a WanGP Git checkout for updates")
    dirty = git(root, runner, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise InstallError("source_conflict", "Save or stash tracked source changes before updating or rolling back. Manager will not overwrite them.\n" + dirty[:1600])
    return {"commit": revision(git(root, runner, "rev-parse", "HEAD")), "branch": git(root, runner, "symbolic-ref", "--quiet", "--short", "HEAD", check=False) or None}


def observation(root, relative):
    path = inside(root, relative)
    return file_hash(path) if path.exists() else None


def target_changes(root, runner, before, target):
    tree = git(root, runner, "ls-tree", "-r", "-z", target)
    if any(entry.startswith(("120000 ", "160000 ")) for entry in tree.split("\0")):
        raise InstallError("unsupported_source", "Updates require a checkout without symlinks or submodules")
    changes = []
    tokens = git(root, runner, "diff", "--no-renames", "--name-status", "-z", before, target).split("\0")
    for status, filename in zip(tokens[::2], tokens[1::2]):
        source_path(filename)
        path = inside(root, filename)
        if status == "A" and path.exists():
            raise InstallError("source_conflict", "Incoming source would overwrite an untracked file: " + filename)
        changes.append({"path": filename, "before": None if status == "A" else "git", "after": None if status == "D" else "git"})
    return changes


class CondaTransactionEngine:
    def __init__(self, runtime, events=None, cancelled=None, environment_service=None, checkpoint_hook=None):
        self.runtime = runtime
        self.events = events or (lambda event: None)
        self.cancelled = cancelled or (lambda: False)
        self.service = environment_service
        self.checkpoint_hook = checkpoint_hook or (lambda phase: None)

    def environment_service(self, root, runner, kind="conda"):
        cls = CondaEnvironmentService if kind == "conda" else EnvironmentService
        return self.service or cls(self.runtime, runner, inside(root, ".manager/cache/uv"))

    def prepare(self, root, python=None, release=None, resolution=None, repository=None, commit=None, operation="update", checkpoint=None, environment_policy="auto"):
        root = plain_path(root)
        with InstallationLock(root):
            if active_operation(root):
                raise InstallError("recovery_required", "Recover the interrupted update first")
            if operation not in ("update", "switch-version", "rollback"):
                raise InstallError("invalid_operation", "Existing installations support updates, version switching and rollback")
            if environment_policy not in ("auto", "current", "new"):
                raise InstallError("invalid_environment_policy", "Choose auto, current or new for the target environment")
            runner = ProcessRunner(events=self.events, cancelled=self.cancelled)
            state = git_state(root, runner)
            current = read_record(root)
            if current and current.get("removing_checkpoint"):
                raise InstallError("cleanup_pending", "Finish removing the saved version before updating")
            if current:
                if current["active"]["commit"] != state["commit"]:
                    raise InstallError("source_conflict", "The Git revision changed outside Manager. Restore the recorded revision before maintaining this installation")
                environment = plain_path(current["active"]["environment"])
                kind = current["ownership"]
            else:
                info = inspect_interpreter(python)
                if info["status"] != "inspected" or (not info["conda_environment"] and info["prefix"] == info["base_prefix"]):
                    raise InstallError("environment_required", "Select this installation's Conda or virtual Python environment")
                environment = plain_path(info["prefix"])
                kind = "conda" if info["conda_environment"] else "venv"
            service = self.environment_service(root, runner, kind)
            before = service.snapshot(environment)
            # Plugins may legitimately add packages between updates. Snapshot the
            # actual current environment for rollback, and detect changes after preview.
            operation_id = uuid.uuid4().hex
            previous = {**current, "active": {**current["active"], "snapshot": before}} if current else {"schema_version": 1, "ownership": kind, "root": str(root), "installation_id": operation_id, "generation": 0, "history": [], "active": {**state, "release_id": (local_version(root) or "Existing checkout") + "-" + state["commit"][:12], "environment": str(environment), "snapshot": before, "resolution": None}}
            target = None
            reuse_environment = operation == "rollback"
            if operation == "rollback":
                if current is None or checkpoint not in current["history"]:
                    raise InstallError("unknown_checkpoint", "Select a saved version")
                saved = read_json(inside(root, f".manager/conda-checkpoints/{identifier(checkpoint)}.json"))
                if saved["installation_id"] != current["installation_id"] or saved["root"] != str(root):
                    raise InstallError("invalid_checkpoint", "This checkpoint belongs to another installation")
                target = saved["active"]
                commit = revision(target["commit"])
                if service.snapshot(target["environment"]) != target["snapshot"]:
                    raise InstallError("environment_changed", "The saved environment has changed and cannot be restored as a verified checkpoint")
                target_version = target["release_id"]
                resolution = target["resolution"]
            else:
                commit = revision(commit)
                self.events({"type": "progress", "title": "Preparing Git Revision", "message": "Fetching the selected source revision without changing the checkout"})
                git(root, runner, "fetch", "--no-tags", "--depth=1", "--", str(repository), commit)
                if revision(git(root, runner, "rev-parse", "FETCH_HEAD")) != commit:
                    raise InstallError("changed_source", "Fetched source does not match the selected revision")
                manifest = load_release(release)
                locked = load_resolution(resolution)
                reuse_environment = environment_policy != "new" and service.satisfies(environment, resolution, locked, before["inventory"], previous["active"]["resolution"])
                if environment_policy == "current" and not reuse_environment:
                    raise InstallError("incompatible_environment", "The current environment needs package changes. Choose Automatic or Create a new environment.")
                if kind == "conda" and not reuse_environment:
                    conda_command(environment)
                # The resolved requirements must be the ones in the selected Git commit.
                for filename, expected in locked["input_files"].items():
                    if filename in locked.get("extra_requirements", []) or not plain_path(filename).is_relative_to(plain_path(locked["source_root"])):
                        continue  # Explicit installed-plugin inputs were hash-checked by load_resolution.
                    relative = plain_path(filename).relative_to(plain_path(locked["source_root"])).as_posix()
                    if manifest["files"].get(relative) != expected:
                        raise InstallError("resolution_mismatch", "Dependency inputs do not match the selected release")
                    if git(root, runner, "show", commit + ":" + relative).replace("\r\n", "\n").strip() != Path(filename).read_text(encoding="utf-8-sig").strip():
                        raise InstallError("resolution_mismatch", "Dependencies differ from the selected Git revision: " + relative)
                target_version = manifest["release_id"]
                resolution = str(plain_path(resolution))
            changes = target_changes(root, runner, state["commit"], commit)
            plan = {"schema_version": 1, "backend": "conda", "environment_kind": kind, "operation": operation, "operation_id": operation_id, "installation_root": str(root), "previous_record": current, "previous": previous, "git_before": state, "environment_before": before, "target_commit": commit, "target_active": target, "resolution_directory": resolution, "resolution_sha256": load_resolution(resolution, check_inputs=False)["resolution_sha256"] if resolution else None, "registry_sha256": observation(root, "envs.json"), "config_sha256": observation(root, "wgp_config.json"), "previous_release_id": previous["active"]["release_id"], "target_release_id": target_version, "changes": changes, "reuse_environment": reuse_environment, "environment": target["environment"] if target else str(environment) if reuse_environment else str(inside(root, ".manager/conda-envs/" + operation_id[:12])), "apply_supported": True}
            plan["plan_sha256"] = digest(plan)
            return plan

    def _phase(self, root, journal, phase):
        journal["phase"] = phase
        journal["journal_sha256"] = digest({key: value for key, value in journal.items() if key != "journal_sha256"})
        atomic_json(inside(root, ".manager/conda-operations/" + journal["operation_id"] + "/journal.json"), journal)
        self.checkpoint_hook(phase)
        self.events({"type": "progress", "title": phase.replace("_", " ").title(), "phase": phase})

    def _finish(self, root, journal, status, error=None):
        journal.update(status=status, error=error)
        self._phase(root, journal, journal["phase"])
        inside(root, ".manager/conda-active.json").unlink(missing_ok=True)
        return {"status": status, "operation_id": journal["operation_id"], "error": error}

    def apply(self, plan):
        if plan.get("plan_sha256") != digest({key: value for key, value in plan.items() if key != "plan_sha256"}):
            raise InstallError("changed_plan", "Prepared update was modified")
        root = plain_path(plan["installation_root"])
        with InstallationLock(root):
            if active_operation(root):
                raise InstallError("recovery_required", "Recover the interrupted update first")
            operation_id = identifier(plan["operation_id"])
            operation = inside(root, ".manager/conda-operations/" + operation_id)
            runner = ProcessRunner(events=self.events, cancelled=self.cancelled)
            service = self.environment_service(root, runner, plan.get("environment_kind", "conda"))
            self._check_before(root, plan, runner, service)
            registry = inside(root, "envs.json")
            journal = {"schema_version": 1, "operation_id": operation_id, "plan": plan, "status": "running", "phase": "preparing", "registry_before": base64.b64encode(registry.read_bytes()).decode() if registry.exists() else None, "next_record": None, "next_registry": None}
            operation.mkdir(parents=True, exist_ok=False)
            runner.log_path = operation / "operation.log"
            self._phase(root, journal, "preparing")
            atomic_json(inside(root, ".manager/conda-active.json"), {"operation_id": operation_id})
            try:
                previous = plan["previous"]
                # Keep old commits reachable even when source branches move or Git runs GC.
                git(root, runner, "update-ref", "refs/wangp-manager/" + previous["installation_id"] + "/" + operation_id, plan["git_before"]["commit"])
                self._phase(root, journal, "preparing_environment")
                if plan["target_active"]:
                    active = plan["target_active"]
                    if service.snapshot(active["environment"]) != active["snapshot"]:
                        raise InstallError("environment_changed", "The rollback environment changed after preview")
                else:
                    resolution = load_resolution(plan["resolution_directory"])
                    if resolution["resolution_sha256"] != plan["resolution_sha256"]:
                        raise InstallError("changed_input", "Dependency resolution changed after preview")
                    cached = inside(root, ".manager/conda-resolutions/" + resolution["resolution_sha256"])
                    _cache_resolution(Path(plan["resolution_directory"]), cached)
                    if plan["reuse_environment"]:
                        if not service.satisfies(plan["environment"], cached, resolution, plan["environment_before"]["inventory"], previous["active"]["resolution"]):
                            raise InstallError("environment_changed", "The current environment no longer satisfies the selected dependencies")
                        validation = service.validate(plan["environment"], resolution, plan["environment_before"]["inventory"])
                    else:
                        validation = service.update_clone(previous["active"]["environment"], plan["environment"], cached, resolution)
                    active = {"commit": plan["target_commit"], "branch": None, "release_id": plan["target_release_id"], "environment": plan["environment"], "snapshot": service.snapshot(plan["environment"]), "resolution": str(cached), "validation": validation}
                runner.check_cancelled()
                self._check_before(root, plan, runner, service)
                next_record = {**previous, "generation": previous["generation"] + 1, "history": [*previous["history"], operation_id], "active": active}
                projected = json.loads(base64.b64decode(journal["registry_before"])) if journal["registry_before"] else {"envs": {}}
                projected["envs"]["WanGP Manager"] = {"type": plan.get("environment_kind", "conda"), "path": active["environment"]}
                projected["active"] = "WanGP Manager"
                journal.update(next_record=next_record, next_registry=projected)
                atomic_json(inside(root, f".manager/conda-checkpoints/{operation_id}.json"), previous)
                self._phase(root, journal, "activating")
                git(root, runner, "checkout", "--detach", "--no-overwrite-ignore", plan["target_commit"])
                if active["branch"] and git(root, runner, "rev-parse", "refs/heads/" + active["branch"], check=False) == plan["target_commit"]:
                    git(root, runner, "checkout", "--no-overwrite-ignore", active["branch"])
                self.checkpoint_hook("source_activated")
                atomic_json(record_path(root), next_record)
                self.checkpoint_hook("record_activated")
                atomic_json(registry, projected)
                self.checkpoint_hook("registry_activated")
                self._verify(root, journal, runner)
                runner.check_cancelled()
                self._phase(root, journal, "committed")
                return self._finish(root, journal, "succeeded")
            except (Exception, KeyboardInterrupt) as error:
                detail = {"code": "cancelled" if isinstance(error, (Cancelled, KeyboardInterrupt)) else getattr(error, "code", "operation_failed"), "message": str(error)}
                if journal["phase"] == "committed":
                    raise InstallError("recovery_required", "The update committed; use Recover to finish recording it") from error
                try:
                    self._restore(root, journal)
                    return self._finish(root, journal, "cancelled" if detail["code"] == "cancelled" else "failed", detail)
                except Exception as restore_error:
                    journal["error"] = detail
                    self._phase(root, journal, "recovery_required")
                    raise InstallError("recovery_required", "The update needs recovery: " + str(restore_error)) from error

    def _check_before(self, root, plan, runner, service):
        if read_record(root) != plan["previous_record"] or git_state(root, runner) != plan["git_before"]:
            raise InstallError("stale_plan", "The installation changed after preview; prepare the update again")
        if service.snapshot(plan["previous"]["active"]["environment"]) != plan["environment_before"]:
            raise InstallError("environment_changed", "The current environment changed after preview")
        if observation(root, "envs.json") != plan["registry_sha256"] or observation(root, "wgp_config.json") != plan["config_sha256"]:
            raise InstallError("stale_plan", "Application settings changed after preview; prepare the update again")
        target_changes(root, runner, plan["git_before"]["commit"], plan["target_commit"])

    def _verify(self, root, journal, runner):
        if git_state(root, runner)["commit"] != journal["plan"]["target_commit"] or read_record(root) != journal["next_record"] or read_json(inside(root, "envs.json")) != journal["next_registry"]:
            raise InstallError("inconsistent_activation", "Source and environment activation did not finish together")

    def _restore(self, root, journal):
        if journal["next_record"] is None:
            return
        runner = ProcessRunner(events=self.events)
        plan = journal["plan"]
        state = git_state(root, runner)
        if state["commit"] not in (plan["git_before"]["commit"], plan["target_commit"]):
            raise InstallError("recovery_conflict", "Source changed after the interrupted update")
        if read_record(root) not in (plan["previous_record"], journal["next_record"]):
            raise InstallError("recovery_conflict", "Installation metadata changed after the interrupted update")
        registry = inside(root, "envs.json")
        old = base64.b64decode(journal["registry_before"]) if journal["registry_before"] else None
        present = registry.read_bytes() if registry.exists() else None
        if present != old and (present is None or json.loads(present) != journal["next_registry"]):
            raise InstallError("recovery_conflict", "Environment selection changed after the interrupted update")
        self.events({"type": "progress", "title": "Restoring Previous Installation"})
        git(root, runner, "checkout", "--detach", "--no-overwrite-ignore", plan["git_before"]["commit"])
        branch = plan["git_before"]["branch"]
        if branch and git(root, runner, "rev-parse", "refs/heads/" + branch) == plan["git_before"]["commit"]:
            git(root, runner, "checkout", "--no-overwrite-ignore", branch)
        if plan["previous_record"]:
            atomic_json(record_path(root), plan["previous_record"])
        else:
            record_path(root).unlink(missing_ok=True)
        if old is None:
            registry.unlink(missing_ok=True)
        else:
            atomic_bytes(registry, old)

    def recover(self, root):
        root = plain_path(root)
        with InstallationLock(root):
            operation_id = active_operation(root)
            if operation_id is None:
                return {"status": "idle"}
            journal = read_json(inside(root, f".manager/conda-operations/{operation_id}/journal.json"))
            if journal["journal_sha256"] != digest({key: value for key, value in journal.items() if key != "journal_sha256"}) or journal["operation_id"] != operation_id or plain_path(journal["plan"]["installation_root"]) != root:
                raise InstallError("invalid_journal", "Recovery journal failed validation")
            if journal["phase"] == "committed":
                self._verify(root, journal, ProcessRunner(events=self.events))
                return self._finish(root, journal, "succeeded")
            self._restore(root, journal)
            return self._finish(root, journal, "recovered", {"code": "interrupted", "message": "The previous source and environment were restored"})


def version_history(root):
    with InstallationLock(root):
        current = read_record(root)
        if current is None:
            return {"versions": []}
        records = [(None, current)] + [(key, read_json(inside(root, f".manager/conda-checkpoints/{identifier(key)}.json"))) for key in reversed(current["history"])]
        versions = [{"checkpoint": key, "current": key is None, "release_id": record["active"]["release_id"], "environment": record["active"]["environment"]} for key, record in records]
        if current.get("removing_checkpoint"):
            versions.append({**current["removing_checkpoint"], "current": False, "removing": True})
        return {"versions": versions}
