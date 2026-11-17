# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deepy's Skills tab: the user's skills listed, edited, imported and deleted in place. Their folder is the only record, so the tab and a
folder copied by hand are the same thing."""
import html
import zipfile
from pathlib import Path

import yaml

from shared.mcp.skills import MAX_SKILL_BYTES, USER_SKILLS_DIR, SkillStore, skill_index, skill_metadata

INTRO = """A skill is a method you give Deepy Prime and connected MCP agents for a kind of task: a folder holding a `SKILL.md` (a `name`, a `description` saying what the skill does and when to use it, then the instructions), in the Agent Skills format that other AI agents use too. Prime sees every skill's description and reads the whole skill when a task matches it.

Importing a skill's `SKILL.md`, or a `.zip` of its folder, is the same as copying the folder into `{folder}`. Import, Save and Delete act at once, without Apply Changes. A skill is instructions Prime will follow: read a skill from elsewhere before importing it.

**Start a new Deepy session after creating, editing, importing or deleting a skill, so Prime uses the updated skills.**"""
SESSION_NOTICE = "Start a new Deepy session to use the updated skills."
LISTED_FILES = 40
NEW_SKILL_JS = r"""async (previous) => {
    const result = await window.wangpConfirm({
        title: "New Skill", closeOnBackdrop: false, message: "Name the skill and describe when Deepy should use it. Nothing is saved until you choose Save Skill.",
        fields: [
            {name: "filename", label: "File Name", value: previous?.filename, required: true, maxLength: 67,
             pattern: "[a-z0-9]+(?:-[a-z0-9]+)*(?:\\.md)?", help: "For example, brand-ads or brand-ads.md. Saved as brand-ads/SKILL.md."},
            {name: "description", label: "Description", value: previous?.description, required: true, multiline: true, maxLength: 1024}
        ], confirmText: "Start Editing"
    });
    return [result ? result.fields : null];
}"""
DELETE_SKILL_JS = """async (name) => {
    if (!name) return [null];
    const confirmed = await window.wangpConfirm({title: "Delete Skill", danger: true, confirmText: "Delete Skill",
        message: `Delete ${name}? Its folder and all supporting files will move into user_skills/.archive. Deepy will no longer discover or read it as a skill.`});
    return [confirmed ? name : null];
}"""


def _cell(text):
    return html.escape(" ".join(str(text).split())).replace("|", "&#124;")


def _listing(rows):
    if not rows:
        return "No skills yet."
    lines = ["| Skill | Description (What Prime Sees) |", "|---|---|"]
    lines += [f"| **{_cell(row['name'])}** | " + (f"**Not used:** {_cell(row['problem'])}" if "problem" in row else _cell(row["description"])) + " |" for row in rows]
    index = skill_index(rows)
    return "\n".join(lines) + (f"\n\nTogether they add about {len(index) // 4:,} tokens to every Deepy Prime session." if index else "")


def _details(store, rows, name):
    """The notes (problem, other files) and the SKILL.md text of a listed skill folder."""
    row = next((row for row in rows if row["name"] == name), None)
    if row is None:
        return "", ""
    folder = store.root / name
    others = sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file() and path != folder / "SKILL.md")
    notes = [f"**Not used:** {html.escape(row['problem'])}"] if "problem" in row else []
    if others:
        notes.append("Other files: " + ", ".join(f"`{path}`" for path in others[:LISTED_FILES]) + (f" and {len(others) - LISTED_FILES} more." if len(others) > LISTED_FILES else "."))
    if any(path.startswith("scripts/") for path in others):
        notes.append("Deepy does not run a skill's scripts: it reads the skill's text files only.")
    try:
        text = row["content"] if "content" in row else (folder / "SKILL.md").read_text(encoding="utf-8-sig", errors="replace")[:MAX_SKILL_BYTES]
    except OSError:
        text = ""
    return "\n\n".join(notes), text


def skills_tab(gr, *, visible: bool, lock_config: bool):
    """Build and bind the Skills tab inside Deepy's settings tabs; returns the tab for the Prime-only visibility toggle."""
    store = SkillStore(USER_SKILLS_DIR)
    assets = Path(__file__).parent
    root = gr.context.get_blocks_context().root_block
    root.head += f'<style>{assets.joinpath("skills_editor.css").read_text(encoding="utf-8")}</style><script>{assets.joinpath("skills_editor.js").read_text(encoding="utf-8")}</script>'

    def state(name=None):
        rows = store.scan()
        names = [row["name"] for row in rows]
        name = name if name in names else (names[0] if names else None)
        return rows, names, name

    rows, names, name = state()
    notes, text = _details(store, rows, name)
    with gr.Tab("Skills", id="skills", visible=visible, elem_id="deepy_skills_tab") as tab:
        # Keep the edit snapshot in this page across server restarts, just like its draft.
        editing = gr.JSON(value=None, visible=False)
        new_fields = gr.JSON(value=None, visible=False)
        gr.Markdown(INTRO.format(folder=store.root))
        listing = gr.Markdown(_listing(rows))
        with gr.Row(elem_classes=["wangp-bottom-aligned-row"]):
            selected = gr.Dropdown(choices=names, value=name, label="Skill", show_label=False, container=False, scale=4)
            new = gr.Button("New Skill", scale=0, min_width=110, interactive=not lock_config)
            delete = gr.Button("Delete Skill", scale=0, min_width=110, interactive=bool(name) and not lock_config)
        details = gr.Markdown(notes)
        # Syntax colouring follows explicit text and scroll events around the native textarea.
        content = gr.Textbox(value=text, interactive=False, label="SKILL.md", lines=16, max_lines=16, autoscroll=False, show_copy_button=True, elem_id="deepy_skill_content")
        with gr.Row():
            edit = gr.Button("Edit Skill", scale=0, interactive=bool(name) and not lock_config)
            save = gr.Button("Save Skill", variant="primary", scale=0, visible=False, interactive=not lock_config)
            discard = gr.Button("Cancel Editing", scale=0, visible=False)
        edit_help = gr.Markdown("Edit the description and instructions, keeping `name` unchanged. Save writes only `SKILL.md`; Cancel Editing reloads the saved version. Your draft stays here when you leave the tab.", visible=False)
        with gr.Row(elem_classes=["wangp-bottom-aligned-row"]):
            upload = gr.UploadButton("Import Skill (SKILL.md or .zip)", file_types=[".md", ".zip"], type="filepath", scale=0, min_width=260, interactive=not lock_config)
            replace = gr.Checkbox(value=False, label="Replace an Installed Skill of the Same Name", interactive=not lock_config)
        status = gr.Markdown()

    def view(name=None, message=""):
        rows, names, name = state(name)
        notes, text = _details(store, rows, name)
        return {listing: _listing(rows), selected: gr.update(choices=names, value=name, interactive=True), details: notes,
                content: gr.update(value=text, interactive=False), delete: gr.update(visible=True, interactive=bool(name) and not lock_config),
                new: gr.update(interactive=not lock_config), new_fields: None, status: message, editing: None,
                edit: gr.update(visible=True, interactive=bool(name) and not lock_config), save: gr.update(visible=False),
                discard: gr.update(visible=False), edit_help: gr.update(visible=False), upload: gr.update(interactive=not lock_config), replace: gr.update(interactive=not lock_config)}

    def refresh(name, snapshot):
        # Re-entering the tab refreshes the index without replacing a local draft.
        return {listing: _listing(store.scan())} if snapshot else view(name)

    def start_edit(name):
        if lock_config or not name:
            return {status: gr.skip()}
        try:
            document = store.read_document(name)
        except FileNotFoundError:
            document = {"content": "", "revision": None}
        except (OSError, ValueError) as exc:
            return {status: f"**Could not edit the skill:** {html.escape(str(exc))}"}
        return edit_view(name, document)

    def edit_view(name, document):
        return {editing: {"name": name, "revision": document["revision"]}, content: gr.update(value=document["content"], interactive=True),
                selected: gr.update(choices=[name], value=name, interactive=False), delete: gr.update(interactive=False), new: gr.update(interactive=False), new_fields: None,
                edit: gr.update(visible=False), save: gr.update(visible=True), discard: gr.update(visible=True), edit_help: gr.update(visible=True),
                upload: gr.update(interactive=False), replace: gr.update(interactive=False), status: ""}

    def create_draft(fields):
        if lock_config or fields is None:
            return {status: gr.skip()}
        name = fields["filename"].strip().removesuffix(".md")
        try:
            if store.path(name).parent.exists():
                raise ValueError(f"A skill named {name} already exists. Choose another file name or edit that skill.")
            metadata = yaml.safe_dump({"name": name, "description": fields["description"].strip()}, allow_unicode=True, sort_keys=False)
            text = f"---\n{metadata}---\n\n# {name.replace('-', ' ').title()}\n\n"
            skill_metadata(text, name)
        except (OSError, ValueError) as exc:
            return {status: f"**Could not create the skill:** {html.escape(str(exc))}", new_fields: fields}
        return {**edit_view(name, {"content": text, "revision": None}), details: "", status: f"New draft: **{html.escape(name)}**. Add the instructions, then choose Save Skill."}

    def save_edit(snapshot, text):
        if lock_config or not snapshot:
            return {status: "**Could not save:** Reopen the skill with Edit Skill before saving."}
        try:
            store.save(snapshot["name"], text, expected_revision=snapshot["revision"])
        except (OSError, ValueError) as exc:
            return {status: f"**Could not save the skill:** {html.escape(str(exc))}\n\nYour draft is still in the editor. If the saved skill changed elsewhere, copy your draft before Cancel Editing and reopen it with Edit Skill."}
        return view(snapshot["name"], f"Saved **{html.escape(snapshot['name'])}**. {SESSION_NOTICE}")

    def remove(name):
        if lock_config or not name:
            return {status: gr.skip()}
        try:
            archived = store.archive(name)
        except (OSError, ValueError) as exc:
            return view(name, f"**Could not delete {html.escape(name)}:** {html.escape(str(exc))}")
        return view(None, f"Deleted **{html.escape(name)}** from active skills. Its files are preserved in `{html.escape(archived)}`. {SESSION_NOTICE}")

    def install(path, replace, current):
        if lock_config:
            return {status: gr.skip()}
        try:
            record = store.install(path, replace=replace)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            return view(current, f"**Could not import the skill:** {html.escape(str(exc))}")
        return view(record["name"], f"Imported **{html.escape(record['name'])}**. {SESSION_NOTICE}")

    outputs = [listing, selected, details, content, delete, new, new_fields, status, editing, edit, save, discard, edit_help, upload, replace]
    highlight = "(text) => window.WanGPSkillEditor.update(text)"
    root.load(fn=None, inputs=[content], js=highlight)
    content.change(fn=None, inputs=[content], js=highlight)
    tab.select(refresh, inputs=[selected, editing], outputs=outputs, show_progress="hidden")
    selected.input(view, inputs=[selected], outputs=outputs, show_progress="hidden", queue=False)
    edit.click(start_edit, inputs=[selected], outputs=outputs, show_progress="hidden")
    save.click(save_edit, inputs=[editing, content], outputs=outputs, show_progress="hidden")
    discard.click(view, inputs=[selected], outputs=outputs, show_progress="hidden")
    new.click(create_draft, inputs=[new_fields], outputs=outputs, js=NEW_SKILL_JS, show_progress="hidden")
    delete.click(remove, inputs=[selected], outputs=outputs, js=DELETE_SKILL_JS, show_progress="hidden")
    upload.upload(install, inputs=[upload, replace, selected], outputs=outputs, show_progress="hidden")
    return tab
