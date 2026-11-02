# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared, user-facing Markdown sections for model help declarations."""
from copy import deepcopy


VIDEO_PROMPT = """### Describe the Shot

Describe the subject, visible action, setting, framing, lighting and camera movement in natural language. Keep actions in chronological order and achievable within the selected duration. With a start image, focus on what moves and what stays still.

**Example:** A ceramic bird on a windowsill slowly turns its head. Rain runs down the glass behind it. Soft morning light, fixed close-up camera.

Put unwanted content in the Negative Prompt when the selected model supports it. Prompt enhancement is optional."""
VIDEO_PROMPT_SHORT = "Describe subject, action, setting, framing, light and camera movement chronologically. With an image, describe motion consistent with it. Keep the action feasible within the duration."
IMAGE_PROMPT = """### Describe the Finished Image

Name the subject, composition, setting, lighting, colors and medium. Put the most important requirements first. For visible writing, quote the exact words and say where they belong; proofread the result.

**Example:** A cream poster with a blue ceramic bird centered above the title \"SPRING MARKET\", dark blue lettering, generous margins, flat screen-print illustration."""
IMAGE_PROMPT_SHORT = 'Describe subject, composition, lighting, colors and medium. Quote exact visible text and its placement; proofread output.'
EDIT_PROMPT = """### Describe the Edit

State the requested change and identify the details to preserve. When using a mask, describe the desired result in that area.

**Example:** Replace the blue jacket with a red wool coat. Keep the person's face, pose, lighting and background unchanged."""
EDIT_PROMPT_SHORT = "State the change and what to preserve. Example: replace the blue jacket with a red wool coat, retaining face, pose, light and background."
REFERENCE_PROMPT = "With several references, number images in selection order and assign each a role. Example: use the person from image 1 and the coat from image 2, keeping the person's face and pose."
SPEECH_PROMPT = """### Write the Words to Speak

Enter the actual speech, with natural spelling and punctuation, rather than an instruction to write speech. Keep stage directions out of the spoken text unless this model explicitly supports them.

**Example:** Welcome back. Take your time; we can begin when you are ready."""
SPEECH_PROMPT_SHORT = "Write exact speech with natural punctuation. Do not put unsupported stage directions in the spoken text."
LYRICS_PROMPT = """### Lyrics and Musical Style

Put the words to sing in **Lyrics**, with section tags on separate lines and blank lines between sections. Put genre, instruments, mood and vocal character in the separate style field.

```text
[Verse]
The morning train rolls into town
The quiet streets are waking now

[Chorus]
Carry me home through the light
```

Repeat the intended words for repeated choruses. Section tags guide the arrangement; they do not guarantee exact timings."""
LYRICS_PROMPT_SHORT = "Use actual lyrics with [Verse], [Chorus], [Bridge] and [Outro] sections; keep musical style in alt_prompt. Repeat chorus words explicitly. Tags do not guarantee exact timings."


# These names describe a user's task; aliases keep discovery consistent across families.
SPECIALITIES = {
    "identity preservation": ("character consistency", "identity-preserving edits", "subject consistency"),
    "motion transfer": ("pose transfer", "dance animation", "human animation"),
    "character replacement": ("replace a person", "character swap"),
    "video editing": ("edit video", "instruction video editing"),
    "image editing": ("edit image", "instruction image editing"),
    "video viewpoint change": ("video reshooting", "recamera", "change video perspective", "novel view video"),
    "image viewpoint change": ("change image perspective", "novel view image"),
    "camera control": ("camera path", "camera trajectory", "camera movement", "orbit", "dolly", "3D camera"),
    "trajectory control": ("point tracking", "object motion control", "motion path"),
    "lip sync": ("talking avatar", "talking head", "audio-driven video"),
    "audio-video generation": ("video with sound", "synchronized audio and video"),
    "multi-shot video": ("shot planning", "hard cuts"),
    "style transfer": ("restyling", "stylization"),
    "image inpainting": ("masked image editing",),
    "video inpainting": ("masked video editing",),
    "video outpainting": ("extend video canvas",),
    "image outpainting": ("extend image canvas",),
    "control image": ("image structure control", "image ControlNet"),
    "control video": ("video structure control", "video ControlNet"),
    "video upscaling": ("video super-resolution",),
    "voice cloning": ("reference voice", "speaker imitation"),
    "voice design": ("voice description",),
    "speech dialogue": ("two-speaker dialogue", "multi-speaker dialogue"),
    "music generation": ("song generation", "lyrics to song"),
    "audio editing": ("edit audio", "sound editing"),
    "audio inpainting": ("replace audio interval",),
    "audio continuation": ("extend audio",),
    "sound effects": ("SFX", "foley", "ambience", "sound design", "audio effects"),
    "text rendering": ("typography", "lettering", "text writing"),
}


def specialities(*names):
    return [{"name": name, "aliases": list(SPECIALITIES[name])} for name in names]


def normalize_specialities(entries):
    canonical = {alias: name for name, aliases in SPECIALITIES.items() for alias in (name, *aliases)}
    result = {}
    for entry in entries:
        entry = {"name": entry} if isinstance(entry, str) else deepcopy(entry)
        original = entry["name"]
        name = canonical.get(original, original)
        aliases = [*SPECIALITIES.get(name, ()), *entry.get("aliases", ())]
        if name != original:
            aliases.append(original)
        current = result.setdefault(name, {**entry, "name": name, "aliases": []})
        current["aliases"] = list(dict.fromkeys(alias for alias in [*current["aliases"], *aliases] if alias != name))
        description = entry.get("description", "")
        if description and description != current.get("description", ""):
            current["description"] = " ".join(filter(None, [current.get("description"), description]))
    return list(result.values())


def help_text(title, infos, prompt, compact_infos, compact_prompt, sources=()):
    links = " · ".join(f"[Upstream {i + 1}]({url})" for i, url in enumerate(sources))
    return {
        "infos": f"## {title}\n\n{infos.strip()}" + (f"\n\n### Further Reading\n\n{links}" if links else ""),
        "prompt_infos": prompt.strip(),
        "deepy_infos": f"### {title}\n\n{compact_infos.strip()}",
        "deepy_prompt_infos": f"### Prompt\n\n{compact_prompt.strip()}",
    }


def merge_help(declarations, help_def):
    """Keep explicit handler help while inheriting shared family declarations."""
    result = {**help_def, **declarations}
    if "specialities" in help_def:
        entries = deepcopy(declarations.get("specialities", []))
        names = {entry["name"] for entry in entries}
        entries.extend(deepcopy(entry) for entry in help_def["specialities"] if entry["name"] not in names)
        result["specialities"] = entries
    return result
