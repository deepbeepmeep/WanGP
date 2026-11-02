# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A transcription as a WanGP generation: the transcribed recording is the output audio, the transcript its Info and side files."""
import math
import os
import re
import tempfile

import soundfile as sf
import torch

from shared.utils.speech_dialogue import parse_speaker_names, speaker_label

from .nemotron_transcribe_handler import CUSTOM_SETTINGS, TASKS

DEFAULTS = {setting["id"]: setting["default"] for setting in CUSTOM_SETTINGS}
INFO_PREVIEW_CHARS = 4000  # the Info panel shows 4096 characters of a value
PER_WORD = 2  # the word_timestamps setting of a .txt with a line per word
CUE_MAX_SECONDS, CUE_MAX_CHARS, CUE_LINE_CHARS, CUE_MAX_GAP = 6.0, 84, 42, 1.0
PAUSE_SECONDS = 1.0  # a pause that starts a new line of a transcript without speakers
SENTENCE_ENDS = ".?!。？！"
_NO_SPACE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")  # kana and CJK ideographs are written without spaces


class TranscriptionModel:
    """What WanGP drives: generate() takes WanGP's generation arguments and runs NemotronTranscribePipeline.generate()."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.transformer = pipeline.transformer
        self.perception = pipeline.perception
        self._interrupt = False

    @property
    def _interrupt(self):  # WanGP's abort flag is the pipeline's own
        return self.pipeline._interrupt

    @_interrupt.setter
    def _interrupt(self, value):
        self.pipeline._interrupt = value

    def generate(self, input_prompt, audio_guide, video_guide, model_mode, custom_settings, seed, callback, set_progress_status, **kwargs):
        settings = {**DEFAULTS, **custom_settings}
        task = TASKS[model_mode]
        target_language = (settings["target_language"] or DEFAULTS["target_language"]) if task == "translate" else ""
        set_progress_status("Extracting Audio")
        handle, wav_path = tempfile.mkstemp(suffix=".wav", prefix="wangp_transcribe_")
        os.close(handle)
        try:
            from shared.utils.audio_video import extract_audio_track_to_wav
            extract_audio_track_to_wav(audio_guide, wav_path)  # one PCM file whatever the container (a video's soundtrack: extracted by WanGP)
            if self._interrupt:
                return None
            result = self.pipeline.generate(audio_path=wav_path, task=task, speakers=bool(settings["speakers"]), word_timestamps=bool(settings["word_timestamps"]),
                                            verbatim=settings["style"] == "verbatim", context_terms=input_prompt.strip(), target_language=target_language, seed=seed,
                                            callback=callback, abort_fn=None, set_progress_status=set_progress_status)
            if result is None or self._interrupt:
                return None
            audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        finally:
            os.remove(wav_path)
        details = {"task": task, "target_language": target_language, "speakers": bool(settings["speakers"]), "word_timestamps": int(settings["word_timestamps"]),
                   "speaker_names": parse_speaker_names(settings["speaker_names"])}
        side_files, info = transcript_side_files(result, details), transcript_info(result, details)
        output = {"x": torch.from_numpy(audio).T, "audio_sampling_rate": sample_rate, "side_files": side_files, "overridden_inputs": {"extra_info": info}}
        if video_guide is not None and settings["output"] == "video":  # the source video (its video was the recording) with the subtitles as its subtitle track
            if ".srt" in side_files:
                set_progress_status("Adding Subtitles")
                output["video_file"] = subtitled_video(video_guide, side_files[".srt"], f"{target_language} Translation" if task == "translate" else "Transcript")
            else:
                info["Transcription Notes"] = (info.get("Transcription Notes", "") + " No subtitles could be timed: the audio is saved instead of the video.").strip()
        return output


def subtitled_video(source_video, subtitles, title):
    """A temporary copy of the source video (streams copied, not re-encoded) with the subtitles (SRT bytes) as its subtitle track."""
    from shared.utils.audio_video import add_subtitle_track, subtitled_video_extension
    handle, srt_path = tempfile.mkstemp(suffix=".srt", prefix="wangp_subtitles_")
    with os.fdopen(handle, "wb") as stream:
        stream.write(subtitles)
    handle, video_path = tempfile.mkstemp(suffix=subtitled_video_extension(source_video), prefix="wangp_subtitled_")
    os.close(handle)
    try:
        add_subtitle_track(source_video, srt_path, video_path, title)
    except Exception:
        os.remove(video_path)
        raise
    finally:
        os.remove(srt_path)
    return video_path


def _clock(seconds, decimals=1):
    units = round(float(seconds) * 10 ** decimals)
    whole, fraction = divmod(units, 10 ** decimals)
    return f"{whole // 3600:02}:{whole // 60 % 60:02}:{whole % 60:02}.{fraction:0{decimals}}"


def _stamp(seconds):
    milliseconds = round(float(seconds) * 1000)
    return f"{milliseconds // 3600000:02}:{milliseconds // 60000 % 60:02}:{milliseconds // 1000 % 60:02},{milliseconds % 1000:03}"


def _join_words(words):
    text = ""
    for word in (str(item["word"]).strip() for item in words):
        if not word:
            continue
        glued = text and (_NO_SPACE.match(word[0]) and _NO_SPACE.match(text[-1]) or word[0] in ",.;:!?%)]}»、。，．！？")
        text += word if not text or glued else " " + word
    return text


def _wrap(text):
    """Two subtitle lines when the cue is longer than one: split at the space nearest the middle."""
    if len(text) <= CUE_LINE_CHARS or " " not in text:
        return text
    middle = len(text) // 2
    split = min((index for index, char in enumerate(text) if char == " "), key=lambda index: abs(index - middle))
    return text[:split] + "\n" + text[split + 1:]


def _split_turn(start, end, speaker, text):
    """The cues of a timed turn without word times (a translated sentence): its words in chunks within the cue limits, each timed in
    proportion to its length."""
    count = max(1, math.ceil((end - start) / CUE_MAX_SECONDS), math.ceil(len(text) / CUE_MAX_CHARS))
    groups, current = [], []
    for word in text.split():
        if current and len(groups) < count - 1 and len(" ".join(current + [word])) > len(text) / count:
            groups.append(current)
            current = []
        current.append(word)
    pieces = [" ".join(group) for group in groups + [current]]
    duration, total, cues = end - start, sum(map(len, pieces)), []
    for piece in pieces:
        span = duration * len(piece) / total
        cues.append((start, start + span, speaker, piece))
        start += span
    return cues


def subtitle_cues(result):
    """(start, end, speaker, text) cues: from the words (a cue ends at a change of speaker, a pause, a sentence end or the cue limits), else one per timed turn."""
    words = [word for word in result["words"] or [] if str(word["word"]).strip()]
    if not words:
        return [cue for turn in result["turns"] if turn["start"] is not None and turn["end"] is not None and turn["text"].strip()
                for cue in _split_turn(turn["start"], turn["end"], turn["speaker"], turn["text"].strip())]
    cues, current = [], []
    for word in words:
        if current:
            text = _join_words(current)
            previous = current[-1]
            if (word["speaker"] != previous["speaker"] or word["start"] - previous["end"] > CUE_MAX_GAP or word["end"] - current[0]["start"] > CUE_MAX_SECONDS
                    or len(text) + 1 + len(str(word["word"]).strip()) > CUE_MAX_CHARS or text[-1] in ".?!。？！" and len(text) >= CUE_LINE_CHARS):
                cues.append((current[0]["start"], previous["end"], current[0]["speaker"], text))
                current = []
        current.append(word)
    if current:
        cues.append((current[0]["start"], current[-1]["end"], current[0]["speaker"], _join_words(current)))
    return cues


def _sentences(words):
    """(start, end, None, text) of the timed words by sentence; a pause also starts a new one."""
    spans, current = [], []
    for word in words:
        if current and word["start"] - current[-1]["end"] > PAUSE_SECONDS:
            spans.append(current)
            current = []
        current.append(word)
        if str(word["word"]).strip()[-1:] in SENTENCE_ENDS:
            spans.append(current)
            current = []
    spans += [current] if current else []
    return [(span[0]["start"], span[-1]["end"], None, _join_words(span)) for span in spans]


def transcript_text(result, speakers, names, per_word=False):
    """The transcript: a line per speaker turn, "[start - end] Speaker N: text" (times when the words are timed: a transcription with
    Timestamps; the speaker's name in place of Speaker N when `names` gives one); without speakers, a timed line per sentence, else the plain
    text. `per_word`: a line per timed word instead, its times to the hundredth of a second."""
    words = [word for word in result["words"] or [] if str(word["word"]).strip()]
    if per_word and words:
        return "\n".join(f"[{_clock(word['start'], 2)} - {_clock(word['end'], 2)}] " + (f"{speaker_label(word['speaker'], names)}: " if speakers and word["speaker"] is not None else "")
                         + str(word["word"]).strip() for word in words)
    spans = _sentences(words) if words and not speakers else [(turn["start"], turn["end"], turn["speaker"], turn["text"].strip()) for turn in result["turns"] if turn["text"].strip()]
    return "\n".join(("" if start is None or end is None else f"[{_clock(start)} - {_clock(end)}] ") + (f"{speaker_label(speaker, names)}: " if speaker is not None else "") + text
                     for start, end, speaker, text in spans)


def transcript_srt(result, speakers, names):
    lines, previous = [], None
    for index, (start, end, speaker, text) in enumerate(subtitle_cues(result), 1):
        label = speaker_label(speaker, names) if speakers and speaker is not None and speaker != previous else ""
        previous = speaker
        lines.append(f"{index}\n{_stamp(start)} --> {_stamp(max(end, start))}\n{_wrap(f'{label}: {text}' if label else text)}\n")
    return "\n".join(lines)


def transcript_side_files(result, details):
    """Companions of the output audio: the .txt transcript (speakers and times), and .srt subtitles when anything is timed."""
    files = {".txt": (transcript_text(result, details["speakers"], details["speaker_names"], details["word_timestamps"] == PER_WORD) + "\n").encode("utf-8")}
    subtitles = transcript_srt(result, details["speakers"], details["speaker_names"])
    if subtitles:
        files[".srt"] = subtitles.encode("utf-8")
    return files


def speaking_times(result):
    """Seconds of speech per speaker, from the diarization segments or else the timed turns."""
    spans = result["diarization"] or [turn for turn in result["turns"] if turn["speaker"] is not None and turn["start"] is not None and turn["end"] is not None]
    totals = {}
    for span in spans:
        totals[span["speaker"]] = totals.get(span["speaker"], 0.0) + max(0.0, span["end"] - span["start"])
    return dict(sorted(totals.items()))


def _duration(seconds):
    seconds = round(seconds)
    return f"{seconds // 60} min {seconds % 60:02} s" if seconds >= 60 else f"{seconds} s"


def transcript_info(result, details):
    """The generation Info entries: the transcript (its beginning when long), the speakers and what went wrong."""
    text = transcript_text(result, details["speakers"], details["speaker_names"])
    if len(text) > INFO_PREVIEW_CHARS:
        cut = text.rfind("\n", 0, INFO_PREVIEW_CHARS)
        text = text[:cut if cut > INFO_PREVIEW_CHARS // 2 else INFO_PREVIEW_CHARS].rstrip() + "\n[...] The full text is in the .txt file beside the audio."
    info = {"Translation" if details["task"] == "translate" else "Transcript": text or "(No speech detected.)"}
    if details["task"] == "translate":
        info["Translated To"] = details["target_language"]
    if details["speakers"] and (times := speaking_times(result)):
        info["Speakers"] = f"{len(times)} speaker{'s' if len(times) > 1 else ''}: " + ", ".join(f"{speaker_label(speaker, details['speaker_names'])} ({_duration(seconds)})" for speaker, seconds in times.items())
    notes = []
    if result["truncated"]:
        notes.append("The output limit was reached: the end of the recording is missing. Transcribe it in shorter parts.")
    if details["word_timestamps"] == PER_WORD and details["task"] == "translate":
        notes.append("Per-word times are for transcriptions: the translation is timed per sentence.")
    if details["word_timestamps"] and result["alignment_error"]:
        notes.append(f"Word timestamps are unavailable: {result['alignment_error']}")
    if notes:
        info["Transcription Notes"] = " ".join(notes)
    return info
