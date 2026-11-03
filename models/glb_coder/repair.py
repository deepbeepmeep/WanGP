# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Conservative syntax corrections and one source-edit request for a failed generated scene."""
import os
import re
import subprocess
from pathlib import Path

from .export import SceneExportError
from .prompts import SYSTEM_PROMPT

SCRIPTS = re.compile(r"(<script\b([^>]*)>)(.*?)(</script\s*>)", re.I | re.S)
CLOSERS = ")]}"
# Skip quoted source, template text and comments. Only swap a literal Line material with a literal geometry built from points.
LINE_ARGUMENTS = re.compile(
    r"""//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|"""
    r"(?P<ctor>\bnew\s+THREE\.Line(?:Segments|Loop)?\s*\(\s*)"
    r"(?P<material>new\s+THREE\.Line(?:Basic|Dashed)Material\s*\(\s*\{[\w\s:#.,-]*\}\s*\))"
    r"(?P<comma>\s*,\s*)(?P<geometry>new\s+THREE\.BufferGeometry\s*\(\s*\)\s*\.setFromPoints\s*\(\s*(?:[\w$]+|\[[\w$,\s]*\])\s*\))(?=\s*\))")


def _syntax_check(source, module):
    import playwright

    # Playwright ships Node on each platform. --check on stdin only parses the script; generated code never runs on the host.
    node = Path(playwright.__file__).parent / "driver" / ("node.exe" if os.name == "nt" else "node")
    try:
        run = subprocess.run([str(node), "--input-type=" + ("module" if module else "commonjs"), "--check", "-"], input=source,
                             encoding="utf-8", errors="replace", capture_output=True, timeout=5,
                             env={k: v for k, v in os.environ.items() if k.upper() not in {"NODE_OPTIONS", "NODE_V8_COVERAGE"}},
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return None
    return run.stderr if run.returncode else ""


def _fix_script(source, module, check_abort):
    diagnostic = _syntax_check(source, module)
    if not diagnostic:
        return source
    lines = source.splitlines(keepends=True)
    location = re.search(r"^\[stdin\]:(\d+)\n[^\n]*\n([^\n]*)\n", diagnostic)
    if not location or not lines:
        return source
    if "SyntaxError: Unexpected end of input" in diagnostic:
        # Try the shortest closing suffix first, without extending a trailing line comment.
        candidates = [source + "\n" + end for end in (*CLOSERS, "})", "])", ")]", "]}")]
    else:
        line_no = int(location[1]) - 1
        marker = location[2]
        if "^" not in marker or line_no >= len(lines):
            return source
        # V8 caret columns are UTF-16 code units.
        column = len(lines[line_no].encode("utf-16-le")[:2 * marker.index("^")].decode("utf-16-le", errors="ignore"))
        pos = sum(map(len, lines[:line_no])) + column
        candidates = []
        if pos < len(source) and source[pos] in CLOSERS:
            candidates.extend(source[:pos] + replacement + source[pos + 1:] for replacement in ("", *CLOSERS) if replacement != source[pos])
        candidates.extend(source[:pos] + closer + source[pos:] for closer in CLOSERS)
        if "missing ) after argument list" in diagnostic:
            # V8 can underline the last argument rather than the semicolon following it.
            end = pos + len(marker.strip())
            candidates.append(source[:end] + ")" + source[end:])

    valid = None
    for candidate in dict.fromkeys(candidates):
        check_abort()
        if _syntax_check(candidate, module) == "":
            if valid is not None:
                return source  # More than one possible correction. Let the model use the scene context.
            valid = candidate
    return valid if valid is not None else source


def fix_obvious_syntax(text, check_abort):
    start = re.search(r"<!doctype\s+html\b|<html\b", text, re.I)
    if not start:
        return text
    html = text[start.start():]
    if "</html>" not in html.lower():
        html = re.sub(r"\n[ \t]*`{3,}[ \t]*\Z", "", html.rstrip())
        if html.lower().rfind("<script") > html.lower().rfind("</script"):
            html += "\n</script>"
        html += "\n</html>"
    else:
        html = html[:html.lower().rfind("</html>") + 7]

    def fix(match):
        attrs, source = match[2], match[3]
        if not source.strip() or re.search(r"\bsrc\s*=", attrs, re.I):
            return match[0]
        kind = re.search(r"""\btype\s*=\s*["']?([^\s"'/>]+)""", attrs, re.I)
        kind = kind[1].lower() if kind else "text/javascript"
        if kind not in {"module", "text/javascript", "application/javascript"}:
            return match[0]
        check_abort()
        source = _fix_script(source, kind == "module", check_abort)
        source = LINE_ARGUMENTS.sub(lambda m: m["ctor"] + m["geometry"] + m["comma"] + m["material"] if m["ctor"] else m[0], source)
        return match[1] + source + match[4]

    fixed = SCRIPTS.sub(fix, html)
    return fixed if fixed != html or "</html>" not in text.lower() else text


def repair_input(tokenizer, prompt, page, error, context_size):
    """Return a repair conversation and the character span to replace, or None for a complete page."""
    instruction = ("The page above failed in Chromium during GLB export. The actual error follows.\n" + error +
                   "\nCorrect the cause and check other instances of that mistake. Preserve the asset, its materials, reference details and requested motion. "
                   "Check the export button as well. For a static request omit the animations option. AnimationClip expects KeyframeTrack objects directly.")
    if "Line.updateMorphTargets" in error or "at new Line " in error:
        instruction += " Check each THREE.Line constructor: pass the BufferGeometry first and the material second."

    def encode(request, previous, task, system=SYSTEM_PROMPT):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": request},
                    {"role": "assistant", "content": previous}, {"role": "user", "content": instruction + "\n" + task}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(chat, add_special_tokens=False)

    section_system = "You fix a section of an existing Three.js scene. Return only replacement JavaScript for the supplied section."
    if error.startswith("GLTFExporter failed:"):
        # A rejected exporter call already has a functioning scene. Preserve everything outside the statement.
        call = re.search(r"(?ms)^(?P<indent>[ \t]*)(?:const|let)\s+\w+\s*=\s*await\s+[\w.]+\.parseAsync\([^,\n]+,\s*\{.*?^(?P=indent)\}\s*\)\s*;", page)
        if call:
            task = ("Replace only this failing parseAsync statement. Keep its result variable and asset argument. "
                    "Return a single complete JavaScript statement. For a static asset the options can be just {binary: true}.")
            ids = encode(prompt, call[0], task, section_system)
            if len(ids) + 256 < context_size:
                return ids, call.span()
    task = "Return the complete corrected HTML page, including the working Export GLB button. Output HTML only."
    ids = encode(prompt, page, task)
    if len(ids) + len(tokenizer.encode(page, add_special_tokens=False)) + 128 <= context_size:
        return ids, None

    # Rewrite a single contiguous section when reproducing the complete page would exhaust the selected context.
    request_ids = tokenizer.encode(prompt, add_special_tokens=False)
    request = tokenizer.decode(request_ids[:context_size // 8])
    lines = page.splitlines(keepends=True)
    site = re.search(r"/scene:(\d+)", error)
    name = re.search(r"(\b[A-Za-z_$][\w$]*) is not defined", error)
    focus = min(len(lines) - 1, int(site[1]) - 1) if site else len(lines) - 1
    if name:
        focus = next((i for i, line in enumerate(lines) if name[1] in line), focus)
    task = ("Only the supplied section is editable. Return its complete corrected replacement as plain source code, "
            "including unchanged lines in that section. The rest of the page is retained verbatim.")
    width = min(96, len(lines))
    while width:
        first, last = max(0, focus - width), min(len(lines), focus + width + 1)
        section = "".join(lines[first:last])
        ids = encode(request, section, task, section_system)
        if len(ids) + len(tokenizer.encode(section, add_special_tokens=False)) + 64 <= context_size:
            start = sum(map(len, lines[:first]))
            return ids, (start, start + len(section))
        width //= 2
    # A minified page can be one long line. The model can still attempt a compact complete page from the original request and diagnostic.
    task = "The previous source exceeded the context window. Recreate a concise working HTML page for this asset and correct the reported error."
    ids = encode(request, "", task)
    if len(ids) >= context_size:
        raise SceneExportError("The diagnostic does not fit in the repair context. Increase KV Cache Size.")
    return ids, None


def apply_repair(page, answer, span):
    if span is None:
        return answer  # The ordinary page reader extracts and verifies the complete HTML.
    answer = answer.strip("\r\n")
    fence = re.fullmatch(r"```[\w+-]*\s*\n(.*?)\n```", answer, re.S)
    replacement = fence[1] if fence else answer
    if not replacement.strip() or re.search(r"<!doctype\s+html\b|<html\b", replacement, re.I):
        raise SceneExportError("The model did not return a replacement for the failing section.")
    start, end = span
    ending = "\r\n" if page[:end].endswith("\r\n") else "\n" if page[:end].endswith("\n") else ""
    return page[:start] + replacement.rstrip("\r\n") + ending + page[end:]
