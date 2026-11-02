# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Designed text over the edit: titles, lower thirds, captions and chapter cards, drawn as SVG at the production's format and rasterized
by WanGP's Draw SVG renderer (resvg). Video models never write text: the edit lays these pictures over the shots.

A design is the production's typography (its faces, case, alignment and ornaments), kept for every piece of text so that a film's
titles, lower thirds and cards belong together, and different from one production to the next. A card can sit over a dimmed picture
(a still of the scene it opens) instead of a plain page."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

from PIL import ImageColor

STYLES = {"title": "Title", "lower_third": "Lower third", "caption": "Caption", "chapter": "Chapter card", "end_card": "End card"}
DESIGNS = {"classic": "Classic", "bold": "Bold", "elegant": "Elegant", "minimal": "Minimal", "typewriter": "Typewriter", "storybook": "Storybook"}
SERIF = "Georgia, 'Times New Roman', serif"
SANS = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
HEAVY = "Impact, 'Arial Black', 'DejaVu Sans Condensed', sans-serif"
BOOK = "'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif"
LIGHT = "'Segoe UI Light', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
MONO = "'Courier New', Consolas, 'DejaVu Sans Mono', monospace"
# Each design's display face (family, weight, italic, capitals, letter spacing and average glyph width in em) and its body face.
FACES = {
    "classic": {"family": SERIF, "weight": 400, "glyph": 0.46, "body": SANS},
    "bold": {"family": HEAVY, "weight": 900, "upper": True, "spacing": 0.02, "glyph": 0.5, "body": SANS},
    "elegant": {"family": BOOK, "weight": 400, "italic": True, "glyph": 0.42, "body": BOOK},
    "minimal": {"family": LIGHT, "weight": 300, "upper": True, "spacing": 0.3, "glyph": 0.62, "body": LIGHT},
    "typewriter": {"family": MONO, "weight": 700, "glyph": 0.6, "body": MONO},
    "storybook": {"family": SERIF, "weight": 400, "glyph": 0.46, "body": SERIF},
}


def _lines(text: str, size: float, width: float, glyph: float = 0.46) -> list[str]:
    """Words wrapped to the width (an average glyph is `glyph` times the font size), keeping the user's line breaks."""
    per_line = max(6, int(width / (size * glyph)))
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            if line and len(line) + 1 + len(word) > per_line:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
    return [line for line in lines if line] or [""]


def _text(lines: list[str], x: float, y: float, size: float, *, family: str, anchor: str = "middle", weight: int = 400, fill: str = "#ffffff",
          spacing: float = 0.0, leading: float = 1.18, italic: bool = False, extra: str = "") -> str:
    spans = "".join(f'<tspan x="{x:.1f}" dy="{0 if index == 0 else size * leading:.1f}">{escape(line)}</tspan>' for index, line in enumerate(lines))
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{escape(family)}" font-size="{size:.1f}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" letter-spacing="{spacing:.1f}"{style} {extra}>{spans}</text>')


def _shade(color: str, factor: float) -> str:
    red, green, blue = ImageColor.getrgb(color)[:3]
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(channel * factor))) for channel in (red, green, blue)))


def _display(c, lines: list[str], x: float, y: float, size: float, anchor: str = "middle", fill: str | None = None, shadow: bool = True) -> str:
    """Lines in the design's display face."""
    face = c.face
    return _text(lines, x, y, size, family=face["family"], anchor=anchor, weight=face["weight"], fill=fill or c.ink, spacing=size * face.get("spacing", 0.0),
                 italic=face.get("italic", False), extra='filter="url(#s)"' if shadow and c.shadowed else "")


def _wrap(c, text: str, size: float, width: float) -> list[str]:
    face = c.face
    text = text.upper() if face.get("upper") else text
    return _lines(text, size, width, face["glyph"] + face.get("spacing", 0.0))


def _defs(c) -> str:
    return ('<defs><filter id="s" x="-20%" y="-50%" width="140%" height="200%"><feDropShadow dx="0" dy="{0:.1f}" stdDeviation="{1:.1f}" '
            'flood-color="#000" flood-opacity="0.75"/></filter><filter id="b" x="-5%" y="-5%" width="110%" height="110%"><feGaussianBlur stdDeviation="{2:.1f}"/></filter>'
            '<radialGradient id="v" cx="50%" cy="50%" r="75%"><stop offset="0.5" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.5"/></radialGradient>'
            '<radialGradient id="g" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#000" stop-opacity="0.35"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>'
            '<linearGradient id="p" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{3}"/><stop offset="1" stop-color="{4}"/></linearGradient>'
            '<radialGradient id="r" cx="50%" cy="46%" r="70%"><stop offset="0.35" stop-color="{3}"/><stop offset="1" stop-color="{5}"/></radialGradient>'
            '<linearGradient id="l" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#0b0d12" stop-opacity="0.82"/><stop offset="1" stop-color="#0b0d12" stop-opacity="0"/></linearGradient>'
            '</defs>').format(c.unit * 0.25, c.unit * 0.6, c.unit * 0.35, c.paper, _shade(c.paper, 0.86 if c.light else 0.4), _shade(c.paper, 0.8 if c.light else 0.45))


def _backdrop(c) -> str:
    """The card's page: the paper (flat, shaded or vignetted by design), or the picture blurred a little and dimmed with the paper's colour."""
    w, h = c.width, c.height
    if c.picture:
        return (f'<rect width="{w}" height="{h}" fill="{c.paper}"/><image href="{c.picture}" x="0" y="0" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" filter="url(#b)"/>'
                f'<rect width="{w}" height="{h}" fill="{c.paper}" fill-opacity="{0.7 if c.light else 0.58}"/><rect width="{w}" height="{h}" fill="url(#v)"/>')
    fill = {"classic": "url(#p)", "elegant": "url(#p)", "storybook": "url(#r)"}.get(c.design, c.paper)
    return f'<rect width="{w}" height="{h}" fill="{fill}"/>'


def _card(c, text: str, subtitle: str) -> str:
    w, h, u, accent = c.width, c.height, c.unit, c.accent
    body = _backdrop(c)
    face = c.face
    if c.design in ("bold", "typewriter"):  # left-aligned blocks
        x0 = w * (0.09 if c.design == "bold" else 0.11)
        size = u * (11.0 if c.design == "bold" else 6.5)
        lines = _wrap(c, text, size, w * 0.78)
        small = u * 2.2
        sub = _lines(subtitle.upper(), small, w * 0.78, 0.62) if subtitle else []
        block = len(lines) * size * 1.08 + (len(sub) * small * 1.35 + u * 2.4 if sub else 0)
        top = h / 2 - block / 2
        if c.design == "bold":
            body += f'<rect x="{x0 - u * 2.6:.1f}" y="{top:.1f}" width="{u * 1.1:.1f}" height="{block:.1f}" fill="{accent}"/>'
            body += _text(lines, x0, top + size * 0.86, size, family=face["family"], anchor="start", weight=face["weight"], fill=c.ink, spacing=size * 0.02, leading=1.08)
            if sub:
                body += _text(sub, x0, top + len(lines) * size * 1.08 + u * 2.4 + small * 0.4, small, family=SANS, anchor="start", weight=700, fill=accent, spacing=u * 0.4, leading=1.35)
            return body
        label_y = top + small * 0.8
        if sub:  # the label first, as a typed heading: CHAPTER TWO / The night shift
            body += _text(sub, x0, label_y, small, family=MONO, anchor="start", weight=700, fill=accent, spacing=u * 0.35, leading=1.35)
        first = top + (len(sub) * small * 1.35 + u * 2.4 if sub else 0) + size * 0.8
        body += _text(lines, x0, first, size, family=MONO, anchor="start", weight=700, fill=c.ink, leading=1.08)
        last = lines[-1]
        cursor_x = x0 + len(last) * size * 0.6 + u * 0.6
        body += f'<rect x="{cursor_x:.1f}" y="{first + (len(lines) - 1) * size * 1.08 - size * 0.72:.1f}" width="{size * 0.5:.1f}" height="{size * 0.86:.1f}" fill="{accent}"/>'
        return body
    size = u * {"classic": 9.0, "elegant": 8.5, "minimal": 4.6, "storybook": 8.0}[c.design]
    lines = _wrap(c, text, size, w * 0.78)
    small = u * (1.8 if c.design == "minimal" else 2.0)
    sub_face = SANS if c.design in ("classic", "elegant") else face["body"]
    sub = _lines(subtitle.upper() if c.design in ("classic", "elegant") else subtitle, small, w * 0.78, 0.6) if subtitle else []
    ornament = {"elegant": size * 0.42, "storybook": size * 0.45}.get(c.design)  # below the last line's descenders
    gap = ornament + u * (1.3 if c.design == "elegant" else 2.0) if ornament else u * (2.5 if c.design == "classic" else 2.2)  # last baseline to the subtitle's top
    block = len(lines) * size * 1.18 + (len(sub) * small * 1.3 + gap if sub else ornament or 0)
    top = h / 2 - block / 2 + size * 0.8
    cx = w / 2
    last = top + (len(lines) - 1) * size * 1.18
    if c.design == "classic":
        body += f'<rect x="{cx - u * 4:.1f}" y="{top - size * 1.25:.1f}" width="{u * 8:.1f}" height="{u * 0.35:.1f}" fill="{accent}"/>'
    elif c.design == "elegant":  # a double frame, and a rule with a diamond between the title and its subtitle
        for inset, stroke, opacity, line in ((u * 3.0, accent, 0.75, 0.14), (u * 3.9, c.ink, 0.28, 0.08)):
            body += (f'<rect x="{inset:.1f}" y="{inset:.1f}" width="{w - 2 * inset:.1f}" height="{h - 2 * inset:.1f}" fill="none" stroke="{stroke}" '
                     f'stroke-opacity="{opacity}" stroke-width="{u * line:.2f}"/>')
        y = last + ornament
        diamond = u * 0.6
        body += (f'<path d="M{cx - u * 10:.1f} {y:.1f}H{cx - diamond * 2:.1f}M{cx + diamond * 2:.1f} {y:.1f}H{cx + u * 10:.1f}" stroke="{accent}" stroke-width="{u * 0.12:.2f}"/>'
                 f'<path d="M{cx:.1f} {y - diamond:.1f}L{cx + diamond:.1f} {y:.1f}L{cx:.1f} {y + diamond:.1f}L{cx - diamond:.1f} {y:.1f}Z" fill="{accent}"/>')
    elif c.design == "storybook":  # a swash under the title and flourished corners
        y = last + ornament
        span = u * 12
        body += (f'<path d="M{cx - span:.1f} {y:.1f}C{cx - span / 2:.1f} {y - u * 1.6:.1f} {cx - span / 6:.1f} {y + u * 1.6:.1f} {cx:.1f} {y:.1f}'
                 f'S{cx + span / 2:.1f} {y - u * 1.6:.1f} {cx + span:.1f} {y:.1f}" fill="none" stroke="{accent}" stroke-width="{u * 0.24:.2f}" stroke-linecap="round"/>'
                 f'<circle cx="{cx - span:.1f}" cy="{y:.1f}" r="{u * 0.45:.1f}" fill="{accent}"/><circle cx="{cx + span:.1f}" cy="{y:.1f}" r="{u * 0.45:.1f}" fill="{accent}"/>')
        inset, arm = u * 3.2, u * 5.0
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            x, yy = (inset if sx > 0 else w - inset), (inset if sy > 0 else h - inset)
            body += (f'<path d="M{x:.1f} {yy + sy * arm:.1f}V{yy:.1f}H{x + sx * arm:.1f}" fill="none" stroke="{accent}" stroke-width="{u * 0.2:.2f}" stroke-opacity="0.8"/>'
                     f'<circle cx="{x + sx * u * 1.1:.1f}" cy="{yy + sy * u * 1.1:.1f}" r="{u * 0.32:.1f}" fill="{accent}" fill-opacity="0.8"/>')
    body += _display(c, lines, cx, top, size)
    if sub:
        sub_fill = c.ink if c.design == "minimal" else accent
        opacity = ' fill-opacity="0.65"' if c.design == "minimal" else ""
        body += _text(sub, cx, last + gap + small * 0.9, small, family=sub_face, weight=600 if sub_face == SANS else 400, fill=sub_fill,
                      spacing=u * (0.45 if c.design in ("classic", "elegant") else 0.12), leading=1.3, italic=c.design == "storybook",
                      extra=opacity + (' filter="url(#s)"' if c.shadowed else ""))
    return body


def _title(c, text: str, subtitle: str) -> str:
    """A title over the opening shot: the design's face over a soft shade (lower left for bold and typewriter designs)."""
    w, h, u, face = c.width, c.height, c.unit, c.face
    if c.design in ("bold", "typewriter"):
        x0, size = w * 0.07, u * (8.5 if c.design == "bold" else 5.0)
        lines = _wrap(c, text, size, w * 0.6)
        small = u * 2.0
        sub = _lines(subtitle.upper(), small, w * 0.6, 0.62) if subtitle else []
        block = len(lines) * size * 1.08 + (len(sub) * small * 1.35 + u * 2.0 if sub else 0)
        top = h - u * 8 - block
        body = ""
        if c.design == "bold":
            body += f'<rect x="{x0 - u * 2.2:.1f}" y="{top:.1f}" width="{u * 0.9:.1f}" height="{block:.1f}" fill="{c.accent}"/>'
        else:
            body += (f'<rect x="{x0 - u * 1.6:.1f}" y="{top - u * 1.4:.1f}" width="{max(len(line) for line in lines + sub) * size * 0.6 + u * 3.2:.1f}" '
                     f'height="{block + u * 2.8:.1f}" fill="#000" fill-opacity="0.58"/>')
        body += _text(lines, x0, top + size * 0.86, size, family=face["family"], anchor="start", weight=face["weight"], spacing=size * face.get("spacing", 0.0), leading=1.08,
                      extra='filter="url(#s)"')
        if sub:
            body += _text(sub, x0, top + len(lines) * size * 1.08 + u * 2.0 + small * 0.4, small, family=MONO if c.design == "typewriter" else SANS, anchor="start", weight=700,
                          fill=c.accent, spacing=u * 0.35, leading=1.35, extra='filter="url(#s)"')
        return body
    size = u * (5.0 if c.design == "minimal" else 7.5)
    lines = _wrap(c, text, size, w * 0.8)
    small = u * 2.0
    sub = _lines(subtitle.upper(), small, w * 0.8, 0.6) if subtitle else []
    block = len(lines) * size * 1.18 + (len(sub) * u * 2.6 + u * 2.5 if sub else 0)
    top = h / 2 - block / 2 + size * 0.8
    body = f'<ellipse cx="{w / 2:.1f}" cy="{h / 2:.1f}" rx="{w * 0.48:.1f}" ry="{block * 1.1 + u * 6:.1f}" fill="url(#g)"/>'
    body += _display(c, lines, w / 2, top, size, fill="#ffffff")
    if sub:
        body += _text(sub, w / 2, top + (len(lines) - 1) * size * 1.18 + u * 4.6, small, family=SANS if c.design != "storybook" else SERIF, weight=600, fill=c.accent,
                      spacing=u * 0.45, leading=1.3, extra='filter="url(#s)"')
    return body


def _lower_third(c, text: str, subtitle: str) -> str:
    """A name and a role at the lower left, styled by the design."""
    w, h, u, face, accent = c.width, c.height, c.unit, c.face, c.accent
    left, bottom = u * 6, h - u * 7
    if c.design == "bold":  # name in a block of the accent colour, the role under it on black
        size, small = u * 3.6, u * 2.0
        name = _lines(text.upper(), size, w * 0.5, 0.5)
        role = _lines(subtitle, small, w * 0.5, 0.52) if subtitle else []
        name_h, role_h = len(name) * size * 1.1 + u * 1.6, (len(role) * small * 1.3 + u * 1.4 if role else 0)
        top = bottom - name_h - role_h
        body = (f'<rect x="{left:.1f}" y="{top:.1f}" width="{max(len(line) for line in name) * size * 0.5 + u * 3:.1f}" height="{name_h:.1f}" fill="{accent}"/>'
                + _text(name, left + u * 1.5, top + u * 0.8 + size * 0.86, size, family=HEAVY, anchor="start", weight=900, fill="#111111", spacing=size * 0.02, leading=1.1))
        if role:
            body += (f'<rect x="{left:.1f}" y="{top + name_h:.1f}" width="{max(len(line) for line in role) * small * 0.56 + u * 3:.1f}" height="{role_h:.1f}" fill="#111111" fill-opacity="0.85"/>'
                     + _text(role, left + u * 1.5, top + name_h + u * 0.7 + small * 0.85, small, family=SANS, anchor="start", weight=600, leading=1.3))
        return body
    if c.design in ("minimal", "elegant", "storybook"):  # no box: the name, a thin rule, the role
        size, small = u * (3.2 if c.design == "minimal" else 3.8), u * 1.9
        name = _wrap(c, text, size, w * 0.5)
        role = _lines(subtitle.upper() if c.design != "storybook" else subtitle, small, w * 0.5, 0.62) if subtitle else []
        top = bottom - len(name) * size * 1.15 - (len(role) * small * 1.35 + u * 1.8 if role else 0)
        body = _display(c, name, left, top + size * 0.86, size, anchor="start", fill="#ffffff")
        rule_y = top + len(name) * size * 1.15 + u * 0.2
        width = max(len(line) for line in name) * size * (face["glyph"] + face.get("spacing", 0.0))
        body += f'<rect x="{left:.1f}" y="{rule_y:.1f}" width="{min(width, u * 14):.1f}" height="{u * 0.16:.2f}" fill="{accent}" filter="url(#s)"/>'
        if role:
            body += _text(role, left, rule_y + u * 1.4 + small * 0.85, small, family=face["body"] if c.design != "elegant" else SANS, anchor="start", weight=400, fill="#e6e8ee",
                          spacing=u * (0.3 if c.design != "storybook" else 0.0), leading=1.35, italic=c.design == "storybook", extra='filter="url(#s)"')
        return body
    family = MONO if c.design == "typewriter" else SANS
    glyph = 0.6 if c.design == "typewriter" else 0.52
    size, small = u * (3.0 if c.design == "typewriter" else 3.4), u * 2.1
    lines = _lines(text, size, w * 0.5, glyph)
    sub = _lines(subtitle, small, w * 0.5, glyph) if subtitle else []
    box_h = len(lines) * size * 1.2 + len(sub) * small * 1.3 + u * 3.2
    box_w = max(len(line) for line in lines + sub) * size * (glyph + 0.04) + u * 6
    if c.design == "typewriter":
        body = f'<rect x="{left:.1f}" y="{bottom - box_h:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" fill="#000" fill-opacity="0.68"/>'
    else:
        body = (f'<rect x="{left:.1f}" y="{bottom - box_h:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" fill="url(#l)"/>'
                f'<rect x="{left:.1f}" y="{bottom - box_h:.1f}" width="{u * 0.5:.1f}" height="{box_h:.1f}" fill="{accent}"/>')
    body += _text(lines, left + u * 2.2, bottom - box_h + u * 1.4 + size * 0.85, size, family=family, anchor="start", weight=700, leading=1.2)
    if sub:
        body += _text(sub, left + u * 2.2, bottom - box_h + u * 1.6 + len(lines) * size * 1.2 + small * 0.6, small, family=family, anchor="start",
                      fill=accent if c.design == "typewriter" else "#d9dde6", leading=1.3)
    return body


def _caption(c, text: str) -> str:
    size = c.unit * 3.1
    family = {"typewriter": MONO, "elegant": BOOK, "storybook": SERIF}.get(c.design, SANS)
    lines = _lines(text, size, c.width * 0.82, 0.6 if family == MONO else 0.52)
    y = c.height - c.unit * 6 - (len(lines) - 1) * size * 1.25
    return _text(lines, c.width / 2, y, size, family=family, weight=600 if family == SANS else 700, leading=1.25,
                 extra=f'stroke="#000" stroke-width="{c.unit * 0.18:.2f}" paint-order="stroke" filter="url(#s)"')


def svg(text: str, subtitle: str, style: str, width: int, height: int, accent: str = "#f2c14e", background: str | None = None, design: str = "classic",
        picture: str | None = None) -> str:
    """The drawing of one piece of text at the canvas size in a design, transparent around it (opaque for chapter and end cards: on the
    production's `background` colour, a pale page for a storybook or a night blue for a thriller, the words in the contrasting ink; or over
    `picture`, an image data URI, dimmed with that colour)."""
    if style not in STYLES:
        raise ValueError(f"Text style must be one of {', '.join(STYLES)}.")
    design = design or "classic"
    if design not in DESIGNS:
        raise ValueError(f"Text design must be one of {', '.join(DESIGNS)}.")
    paper = background or "#14161c"
    red, green, blue = ImageColor.getrgb(paper)[:3]
    light = 0.2126 * red + 0.7152 * green + 0.0722 * blue > 150  # dark words on a light page
    card = style in ("chapter", "end_card")
    c = SimpleNamespace(width=width, height=height, unit=min(width, height) / 56, paper=paper, light=light, ink="#2a2522" if light and card else "#ffffff", accent=accent,
                        design=design, face=FACES[design], picture=picture if card else None, shadowed=not (card and light))
    # the unit is the frame's short side / 56: 12.9 px at 1280x720, as large in a vertical frame (a width-based scale made portrait text tiny)
    if card:
        body = _card(c, text, subtitle)
    elif style == "title":
        body = _title(c, text, subtitle)
    elif style == "lower_third":
        body = _lower_third(c, text, subtitle)
    else:
        body = _caption(c, text)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{_defs(c)}{body}</svg>'


def render(text: str, subtitle: str, style: str, size: tuple[int, int], out: Path, accent: str | None = None, background: str | None = None, design: str = "classic",
           picture: str | Path | None = None) -> Path:
    """Draw the text to `out` (PNG). picture: an image file shown dimmed behind a card."""
    from shared.deepy import svg_drawing
    width, height = size
    card = style in ("chapter", "end_card")
    image = svg_drawing.rasterize(svg(text, subtitle, style, width, height, accent or "#f2c14e", background if card else None, design,
                                      svg_drawing.data_uri(str(picture)) if picture and card else None), width, height)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out
