"""Composition-source compatibility and validation for YuE2."""
from pathlib import Path
import re

SOURCE_LABELS = {
    "0": "New Composition",
    "1": "Use ABC Score",
    "12": "Continue ABC Score",
    "A": "Use Score from Audio",
    "A2": "Continue Score from Audio",
}


def resolve_source(source, custom_guide=None, custom_guide_used=False, custom_settings=None):
    source = str(source) if source is not None else ""
    source = source.replace("N", "").replace("V", "")
    if source and set(source) <= set("01A2"):
        source = "".join(flag for flag in "01A2" if flag in source)
    if not source:
        source = "1" if custom_guide or custom_guide_used else "0"
    if source not in SOURCE_LABELS:
        raise ValueError("Choose one of the five YuE2 composition sources.")
    if custom_settings is not None and custom_settings.get("extend_score") == 1:
        source = {"1": "12", "A": "A2"}.get(source, source)
    return source


def migrate_source(settings, settings_version=None):
    flags = settings.get("audio_prompt_type") or ""
    # Older releases could retain the letter choices beside the new numeric flags.
    if settings_version is not None and settings_version < 2.81:
        flags = flags.translate(str.maketrans("SQE", "012"))
    source = resolve_source(flags, settings.get("custom_guide"), settings.get("custom_guide_used", False), settings.get("custom_settings"))
    settings["audio_prompt_type"] = source + "".join(flag for flag in "NV" if flag in flags)
    if settings.get("custom_settings") is not None:
        settings["custom_settings"] = {key: value for key, value in settings["custom_settings"].items() if key != "extend_score"}
    return source


def validate_source(source, model_mode, custom_guide, audio_guide):
    if model_mode not in (0, 1, 2, 3):
        raise ValueError("Choose melody and chords, melody only, direct generation, or instrumental.")
    if model_mode == 2 and source != "0":
        raise ValueError("Direct generation cannot use or continue a score. Select New Composition or enable planning.")
    if source in ("1", "12") and not custom_guide:
        raise ValueError("Select an ABC file for this composition source.")
    if source in ("A", "A2") and not audio_guide:
        raise ValueError("Select source audio for this composition source.")


def validate_abc(abc, melody_only=False):
    if not abc.strip():
        raise ValueError("The ABC score is empty.")
    if not re.search(r"(?m)^K\s*:", abc):
        raise ValueError("The ABC score needs a key header (K:).")
    if melody_only and re.search(r'"[A-G][#b]?(?:m|maj|min|dim|aug|sus|add|[0-9]|/|\")', abc):
        raise ValueError("Melody-only planning requires a score without chord symbols.")
    from symusic import Score
    try:
        score = Score.from_abc(abc)
    except Exception as exc:
        raise ValueError("The ABC score could not be parsed as musical notation.") from exc
    music = "\n".join(line for line in abc.splitlines() if not re.match(r"\s*(?:[A-Za-z]:|%)", line))
    if score.note_num() == 0 and not re.search(r"[zZxX](?:\d|/|\s|\||$)", music):
        raise ValueError("The ABC score contains no notes or rests.")
    return abc


def read_abc(path, melody_only=False):
    path = Path(path)
    if path.suffix.lower() != ".abc":
        raise ValueError("Select a UTF-8 score with the .abc extension.")
    try:
        abc = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError("The ABC file cannot be read as UTF-8 text.") from exc
    return validate_abc(abc, melody_only)
