# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prime filesystem access: free workspace, create-only outputs, configured folders."""

from dataclasses import dataclass, replace
from pathlib import Path

from shared.deepy.filesystem import FileAccessPolicy, _inside, _resolved_path, _unique_paths
from shared.utils.virtual_media import VIEW_FOLDER


@dataclass(frozen=True)
class PrimeFileAccessPolicy(FileAccessPolicy):
    workspace: Path | None = None
    prompt_read_only: bool = True

    @property
    def read_enabled(self):
        return True

    @property
    def write_enabled(self):
        return True

    @property
    def base_roots(self):
        return _unique_paths([*self.output_roots, *([self.workspace] if self.workspace is not None else [])])

    @property
    def read_roots(self):
        return _unique_paths([*self.base_roots, *(self.selected_roots if self.mode in {"read", "read_write"} else ())])

    @property
    def write_roots(self):
        return _unique_paths([*([self.workspace] if self.workspace is not None else []), *([*self.output_roots, *self.selected_roots] if self.mode == "read_write" else [])])

    @property
    def mounts(self):
        all_roots = _unique_paths([*self.output_roots, *self.selected_roots])
        return (*((alias, root) for alias, root in zip(self.root_aliases, all_roots) if root in self.read_roots), ("views", self.views))

    @property
    def views(self):
        """Where extract_video and Showrunner keep parts of videos as views: @views, inside the first output folder."""
        return _resolved_path(self.output_roots[0]) / VIEW_FOLDER

    def roots(self):
        return [*super().roots(), {"path": "@views", "exists": True, "writable": False}]

    def require_read(self, path, *, file=False, directory=False):
        if directory and self.resolve_path(path) == self.views:  # listed before the first view: an empty folder, made only then, so that an output folder holds none for nothing
            self.views.mkdir(parents=True, exist_ok=True)
        return super().require_read(path, file=file, directory=directory)

    def with_root(self, path, alias):
        root = Path(path).expanduser().resolve()
        if alias in self.root_aliases:
            raise ValueError(f"Filesystem root alias already exists: {alias}")
        return replace(self, selected_roots=(*self.selected_roots, root), root_aliases=(*self.root_aliases, alias), workspace=root if alias.startswith("documents") else self.workspace)

    def create_only(self, path):
        target = self.resolve_path(path)
        return not (self.workspace is not None and _inside(target, self.workspace)) and any(_inside(target, root) for root in self.output_roots)

    def require_write(self, path):
        target = super().require_write(path)
        if self.create_only(target) and target.exists() and not target.is_dir():
            raise PermissionError(f"Output files are create-only; existing files cannot be modified or overwritten: {self.virtualize_path(target)}")
        return target

    def require_mutation(self, path):
        target = super().require_write(path)
        if self.create_only(target) or any(_inside(root, target) for root in self.output_roots):
            raise PermissionError("Output folders and their contents cannot be deleted, renamed or moved.")
        if self.workspace is not None and _inside(target, self.workspace):
            return target
        if self.workspace is not None and _inside(self.workspace, target):
            raise PermissionError("The parent of the session workspace cannot be moved or deleted.")
        return super().require_mutation(target)


def build_prime_policy(base, workspace):
    policy = PrimeFileAccessPolicy(mode=base.mode, output_roots=base.output_roots, selected_roots=base.selected_roots, read_everywhere=base.read_everywhere, root_aliases=base.aliases)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    for alias, root in zip(base.aliases, base.read_roots):
        if root == workspace:
            return replace(policy, workspace=workspace)
    alias, index = "documents", 2
    while alias in base.aliases:
        alias, index = f"documents{index}", index + 1
    return policy.with_root(workspace, alias)
