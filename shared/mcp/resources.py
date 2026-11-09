# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared, paginated documentation browsing for MCP and Deepy Prime."""

import fnmatch
import hashlib
import json
import re
from pathlib import Path
from typing import Any

WORKFLOWS_URI = "wangp://guides/workflows"
GUIDE_PATH = Path(__file__).parents[1] / "deepy" / "prime_guide.md"
INDEXED_DOCUMENTS = {WORKFLOWS_URI: GUIDE_PATH}  # index resources whose searches and section reads cover the whole document
RESOURCE_ALIASES = {"wangp://guides/v2": "wangp://guides/workflows"}
RESOURCE_DESCRIPTION = "Find and read WanGP documentation, skills and workflows. Without uri, query filters the resource index (e.g. 'skills' or a task). With uri, query searches passages ('*' lists headings), section reads matching headings; omit both for full text. Follow next_call for more."
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


GUIDE_SECTIONS = {  # heading -> when to read it; the workflows index lists these, each section is its own resource
    "Templates and generation": "choosing a template or model and building settings: defaults, accelerator profiles, presets, LoRAs, batches",
    "Media selection and inputs": "giving a generation images, videos or audio: start/end frames, references, control videos, masks, selected Gallery media",
    "3D assets and worlds": "3D assets from text or a picture, refiners and texturers, retexturing a mesh, checking 3D views and animations; explorable worlds: walks, continuation, a picture from another viewpoint",
    "Post-processing and media utilities": "upscaling, frame interpolation, sound treatments, and the toolbox utilities (extract, merge, remux, resize, 3D views, SVG drawings)",
    "Speech and chained media": "speech, songs and media made from other generated media, such as talking videos driven by a speech clip",
    "Extraction by spoken words": "cutting audio or video where given words are spoken",
    "Python sandbox": "computing and composing with Python: exact typography over images, charts, grids, palettes, procedural SVG, subtitles from timings, clips frame by frame",
    "Web research": "looking things up on the web when the toolbox offers wangp_web_search: facts, dates, current events, references, the look of real subjects, reading a page before quoting it, downloading reference files, cookies for signed-in sites, LoRAs from Civitai and Hugging Face",
    "Visual verification": "checking produced images or videos: content, text spelling, anatomy, artifacts",
    "Long video and sliding windows": "videos longer than one window: window planning, key frames and end-frame anchors, injected frames, continuation",
    "Prompt files and long documents": "skills, long prompts kept in files, and writing long documents",
    "File search and pagination": "finding files and paging through listings and searches",
    "Audio quality checks and preflight": "validating settings before a job, and measuring audio levels, loudness and silences",
}


def guide_sections() -> dict[str, tuple[str, str]]:
    """slug -> (heading, Markdown) of each section of the workflows guide, with its introduction under "introduction"."""
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    parts = re.split(r"^## (.+)$", guide, flags=re.M)
    sections = {"introduction": ("Introduction", parts[0].strip())}
    for heading, body in zip(parts[1::2], parts[2::2]):
        sections[re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")] = (heading, f"## {heading}\n{body.rstrip()}")
    return sections


def guide_index() -> str:
    sections = guide_sections()
    lines = [sections.pop("introduction")[1], "", "## Sections", "", "Read only the sections the task needs; each URI returns one section:", ""]
    lines += [f"- {heading} ({len(text) / 1000:.1f} KB): {GUIDE_SECTIONS[heading]}. `{WORKFLOWS_URI}/{slug}`" for slug, (heading, text) in sections.items()]
    return "\n".join(lines) + "\n"


def indexed_section_record(uri: str) -> dict[str, Any] | None:
    """The record of a workflows guide section URI: served by a resource template, it is absent from resource listings."""
    slug = uri.removeprefix(WORKFLOWS_URI + "/")
    if slug == uri or slug not in (sections := guide_sections()):
        return None
    return {"server": "wangp", "uri": uri, "name": slug, "title": sections[slug][0], "description": "A section of the WanGP workflows guide.", "mime_type": "text/markdown"}


def plugin_skill_record(uri: str) -> dict[str, Any] | None:
    """The record of a plugin toolbox skill URI: served by a resource template, it is absent from resource listings."""
    from shared.utils.plugins import get_deepy_plugin_skill

    skill = get_deepy_plugin_skill(uri)
    if skill is None or skill.routing:
        return None
    return {"server": "wangp", "uri": uri, "name": skill.name, "title": skill.name.replace("-", " ").title(), "description": skill.description, "mime_type": "text/markdown"}


def resource_record(item, server="wangp"):
    return {"server": server, "uri": str(item.uri), "name": item.name, "title": item.title or item.name, "description": item.description or "", "mime_type": item.mimeType or ""}


async def query_resources(pages, resources, read_resource, arguments, *, servers=("wangp",), unavailable_servers=None, read_page_chars=16000):
    server = str(arguments.get("server", "") or "").strip()
    uri = str(arguments.get("uri", "") or "").strip()
    if not server or server == "wangp":
        uri = RESOURCE_ALIASES.get(uri, uri)
    query = str(arguments.get("query", "") or "").strip()
    section = str(arguments.get("section", "") or "").strip()
    if server and server not in servers:
        raise ValueError(f"Unknown MCP server: {server}")
    if not uri:
        if section:
            raise ValueError("uri is required when section is provided")
        records = [resource for resource in resources if not server or resource["server"] == server]
        if query:
            from shared.deepy.engine import _score_doc_section, _tokenize_doc_query

            tokens = _tokenize_doc_query(query)
            ranked = [(_score_doc_section(query, tokens, resource["title"] or resource["name"], {"section": resource["uri"], "body": resource["description"]}), resource) for resource in records]
            records = [resource for score, resource in sorted(ranked, key=lambda item: -item[0]) if score > 0]
        records = ({key: resource[key] for key in ("server", "uri", "description")} for resource in records)
        page = pages.page(["resources", server, query], records, key="resources", limit=arguments.get("limit", 20), cursor=arguments.get("cursor"), metadata={"unavailable_servers": dict(unavailable_servers or {})})
    else:
        matches = [resource for resource in resources if resource["uri"] == uri and (not server or resource["server"] == server)]
        if not matches and (not server or server == "wangp") and (section_record := indexed_section_record(uri) or plugin_skill_record(uri)) is not None:
            matches = [section_record]
        if not matches:
            prefix = uri.rsplit("/", 1)[0] + "/" if "/" in uri.removeprefix("wangp://") else ""
            siblings = sorted(resource["uri"].removeprefix(prefix) for resource in resources if prefix and resource["uri"].startswith(prefix))
            hint = f" Available under {prefix}: {', '.join(siblings)}." if siblings else " Search with query instead of uri."
            raise ValueError(f"Unknown MCP resource: {uri}.{hint}")
        matching_servers = sorted({resource["server"] for resource in matches})
        if not server:
            if len(matching_servers) != 1:
                raise ValueError(f"Resource URI is exposed by multiple MCP servers; specify one of: {', '.join(matching_servers)}")
            server = matching_servers[0]
        page_query = ["resource", server, uri, query, section]
        page_options = {"key": "sections" if query == "*" else "matches" if query else "contents", "limit": arguments.get("limit", 100 if query == "*" else 5 if query else 20), "max_chars": 6000 if query else read_page_chars}
        if arguments.get("cursor"):
            page = pages.page(page_query, cursor=arguments["cursor"], **page_options)
        else:
            indexed = INDEXED_DOCUMENTS.get(uri) if server == "wangp" and (query or section) else None
            contents = [{"uri": uri, "mime_type": "text/markdown", "text": indexed.read_text(encoding="utf-8")}] if indexed else await read_resource(server, uri)
            metadata = {"server": server, "uri": uri}
            if server == "wangp" and uri.startswith("wangp://skills/user/"):
                metadata["revision"] = hashlib.sha256(contents[0]["text"].encode("utf-8")).hexdigest()
            if server == "wangp" and uri.startswith("wangp://docs/"):
                metadata["applicability"] = matches[0]["description"]
            if query:
                records = []
                for content in contents:
                    if "text" not in content:
                        raise ValueError("Markdown resource search is only available for text resources.")
                    text = _select_markdown_sections(content["text"], section)[0] if section else content["text"]  # a section narrows the search
                    if query == "*":
                        records.extend(item["section"] for item in _extract_markdown_sections(text))
                    else:
                        records.extend(_search_markdown_sections(text, query, title=matches[0]["title"] or matches[0]["name"]))
                if query != "*":
                    records.sort(key=lambda item: (-int(item["score"]), len(str(item["section"]))))
                metadata.update(query=query, **({"section": section} if section else {}))
            else:
                selected, matched_sections = [], []
                for content in contents:
                    item = dict(content)
                    if section:
                        if "text" not in item:
                            raise ValueError("Markdown section filtering is only available for text resources.")
                        item["text"], names = _select_markdown_sections(item["text"], section)
                        matched_sections.extend(names)
                    selected.append(item)
                records = _resource_read_blocks(selected, read_page_chars)
                metadata.update(section=section, matched_sections=matched_sections)
            page = pages.page(page_query, records, metadata=metadata, **page_options)
    if page["has_more"]:
        page["next_call"] = {**arguments, "cursor": page["next_cursor"]}
    return page


def _extract_markdown_sections(markdown: str) -> list[dict[str, Any]]:
    content = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    content = re.sub(r"\n\n---\n\n> Applies to: [^\n]*\Z", "", content)
    lines = content.split("\n") if content else []
    headings = []
    in_code_block = False
    for line_no, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = _MARKDOWN_HEADING_RE.match(line)
        if match is not None:
            headings.append((line_no, len(match.group(1)), match.group(2).strip()))
    include_top_level = not any(level > 1 for _line_no, level, _heading in headings)
    stack = []
    sections = []
    for heading_no, (start_line, level, heading) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        if not include_top_level and level == 1:
            continue
        end_line = len(lines)
        for next_line, next_level, _next_heading in headings[heading_no + 1:]:
            if next_level <= level:
                end_line = next_line
                break
        section_path = " > ".join(title for heading_level, title in stack if include_top_level or heading_level > 1)
        sections.append({"section": section_path or heading, "heading": heading, "heading_level": level, "content": "\n".join(lines[start_line:end_line]).strip(), "body": "\n".join(lines[start_line + 1:end_line]).strip()})
    if not sections and content:
        sections.append({"section": "Document", "heading": "Document", "heading_level": 1, "content": content, "body": content})
    return sections


def _select_markdown_sections(markdown: str, section_filter: str) -> tuple[str, list[str]]:
    sections = _extract_markdown_sections(markdown)
    pattern = str(section_filter or "").strip().casefold()
    wildcard = "*" in pattern or "?" in pattern
    if wildcard:
        matches = [section for section in sections if any(fnmatch.fnmatchcase(str(section[key]).casefold(), pattern) for key in ("section", "heading"))]
    else:
        matches = [section for section in sections if any(pattern == str(section[key]).casefold() for key in ("section", "heading"))]
        if not matches:
            matches = [section for section in sections if any(pattern in str(section[key]).casefold() for key in ("section", "heading"))]
    if not matches:
        raise ValueError(f"Markdown section not found: {section_filter}. Its sections: " + "; ".join(str(section["section"]) for section in sections))
    return "\n\n".join(str(section["content"]) for section in matches), [str(section["section"]) for section in matches]


def _search_markdown_sections(markdown: str, query: str, title: str = "") -> list[dict[str, Any]]:
    from shared.deepy.engine import _build_doc_excerpt, _score_doc_section, _tokenize_doc_query

    query = str(query or "").strip()
    if not query:
        raise ValueError("query is empty.")
    query_tokens = _tokenize_doc_query(query)
    matches = []
    sections = _extract_markdown_sections(markdown)
    exact_rows = []
    for section in sections:
        rows = [line for line in section["body"].splitlines() if line.startswith("|") and line.split("|", 2)[1].strip().strip("`").casefold() == query.casefold()]
        if rows:
            excerpt = "\n".join(rows)
            exact_rows.append({"section": section["section"], "heading": section["heading"], "heading_level": section["heading_level"], "excerpt": excerpt if len(excerpt) <= 1000 else excerpt[:1000] + "…", "score": 100})
    if exact_rows:
        return exact_rows
    for section in sections:
        score = _score_doc_section(query, query_tokens, str(title or ""), section)
        if score <= 0:
            continue
        matches.append({"section": section["section"], "heading": section["heading"], "heading_level": section["heading_level"], "excerpt": _build_doc_excerpt(section, query, query_tokens), "score": int(score)})
    matches.sort(key=lambda item: (-int(item["score"]), len(str(item["section"]))))
    return matches


def _resource_read_blocks(contents, max_chars):
    """Split oversized text records without omitting or duplicating source text."""
    for item in contents:
        if "text" not in item or len(json.dumps(item, ensure_ascii=False, separators=(",", ":"))) + 1 <= max_chars:
            yield item
            continue
        text, start = item["text"], 0
        text_budget = max_chars - len(json.dumps({**item, "text": ""}, ensure_ascii=False, separators=(",", ":"))) - 1
        if text_budget < 1:
            raise ValueError("Resource metadata exceeds the read page budget.")
        while start < len(text):
            end = min(len(text), start + text_budget)
            while (encoded_chars := len(json.dumps(text[start:end], ensure_ascii=False)) - 2) > text_budget:
                end = start + (end - start) * text_budget // encoded_chars
                if end == start:
                    raise ValueError("Resource metadata exceeds the read page budget.")
            if end < len(text) and (newline := text.rfind("\n", start, end)) >= start:
                end = newline + 1
            yield {**item, "text": text[start:end]}
            start = end


