# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Streaming CPU measurements and bounded previews of an audio stream."""

import math
import re
import subprocess
import tempfile
from pathlib import Path

import ffmpeg
import numpy as np

from shared.deepy.video_tools import _get_ffmpeg_path
from shared.utils.subprocess_stdin import child_kwargs


MAX_ENVELOPE_WINDOWS = 600
_FRAME_LOG = re.compile(r"\bt:\s*([\d.]+)\s+TARGET:.*?\bM:\s*(-?[\d.]+|-?inf|nan)")


def inspect_audio(path, *, audio_track_no=1, start_time=0.0, duration=None, silence_db=-50.0, min_silence=0.2, preview_path=None, preview_seconds=10.0, window_s=None, window_peaks=False):
    numbers = [start_time, silence_db, min_silence, preview_seconds, *([] if duration is None else [duration]), *([] if window_s is None else [window_s])]
    if not all(math.isfinite(value) for value in numbers) or start_time < 0 or duration is not None and duration <= 0:
        raise ValueError("Times must be finite; start_time must be non-negative and duration positive.")
    if not -120 <= silence_db <= 0 or min_silence <= 0 or not 0 < preview_seconds <= 30 or window_s is not None and not 0.1 <= window_s <= 600:
        raise ValueError("silence_db must be -120..0, min_silence positive, preview_seconds in (0, 30] and window_s in [0.1, 600].")
    binary = _get_ffmpeg_path()
    probe = ffmpeg.probe(str(path))
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if not 1 <= audio_track_no <= len(streams):
        raise ValueError(f"audio_track_no must select an available 1-based audio stream (found {len(streams)}).")
    stream = streams[audio_track_no - 1]
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    container = probe.get("format", {})
    audio_start = max(0.0, float(stream.get("start_time", 0)) - float(container.get("start_time", 0)))
    audio_duration = float(stream["duration"]) if stream.get("duration") not in (None, "N/A") else None
    end_time = start_time + duration if duration is not None else None
    if audio_duration is not None:
        end_time = min(end_time, audio_start + audio_duration) if end_time is not None else audio_start + audio_duration
    start_time = max(start_time, audio_start)
    duration = end_time - start_time if end_time is not None else None
    if duration is not None and duration <= 0:
        raise ValueError("The selected audio range contains no samples.")
    command = [binary, "-nostdin", "-hide_banner", "-nostats", "-threads", "1", "-ss", str(start_time), "-i", str(path), "-map", f"0:a:{audio_track_no - 1}", "-vn"]
    if duration is not None:
        command += ["-t", str(duration)]
    window = max(1, round(rate * 0.02))
    if window_s is not None and duration is not None and duration / window_s > MAX_ENVELOPE_WINDOWS:
        raise ValueError(f"window_s gives more than {MAX_ENVELOPE_WINDOWS} windows for this range; use a larger window_s or a shorter duration.")
    peak, squares, clipped = np.zeros(channels), np.zeros(channels), np.zeros(channels, dtype=np.int64)
    envelope = {}  # window index -> [sum of squares over all channels, samples, peak]
    frames, silence_start, silent_seconds, interval_count, intervals = 0, None, 0.0, 0, []

    def end_silence(end):
        nonlocal silence_start, silent_seconds, interval_count
        if silence_start is not None and end - silence_start >= min_silence:
            silent_seconds += end - silence_start
            interval_count += 1
            if len(intervals) < 200:
                intervals.append({"start": start_time + silence_start, "end": start_time + end})
        silence_start = None

    # ebur128 passes the original samples through; measurements below are not loudness-normalized.
    # Trim before metering: an output -t alone can let the filter inspect later frames.
    filters = ([f"atrim=end_sample={round(duration * rate)}"] if duration is not None else []) + [f"ebur128=peak=true:framelog={'info' if window_s is not None else 'verbose'}"]  # info logs the 100 ms frames
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command + ["-af", ",".join(filters), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], stdout=subprocess.PIPE, stderr=log, **child_kwargs())
        try:
            while data := process.stdout.read(window * channels * 4):
                samples = np.frombuffer(data, dtype="<f4").reshape(-1, channels).astype(np.float64)
                if not np.isfinite(samples).all():
                    raise ValueError("Decoded audio contains non-finite samples.")
                absolute = np.abs(samples)
                peak = np.maximum(peak, absolute.max(axis=0))
                squares += np.square(samples).sum(axis=0)
                clipped += (absolute >= 1.0).sum(axis=0)
                quiet = np.sqrt(np.square(samples).mean()) <= 10 ** (silence_db / 20)
                if window_s is not None:
                    bucket = envelope.setdefault(int(frames / rate / window_s), [0.0, 0, 0.0])
                    bucket[0] += float(np.square(samples).sum()); bucket[1] += samples.size; bucket[2] = max(bucket[2], float(absolute.max()))
                if quiet and silence_start is None:
                    silence_start = frames / rate
                elif not quiet:
                    end_silence(frames / rate)
                frames += len(samples)
            code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        log.seek(0)
        diagnostics = log.read().decode("utf-8", errors="replace")
        if code:
            raise ValueError(f"Audio decoding failed: {diagnostics[-1500:]}")
    if not frames:
        raise ValueError("The selected audio range contains no samples.")
    end_silence(frames / rate)

    def db(value):
        return round(20 * math.log10(value), 2) if value > 0 else None

    summary = diagnostics.rsplit("Summary:", 1)[-1]
    integrated = re.search(r"\bI:\s*([-+\w.]+)\s*LUFS", summary)
    true_peak = re.search(r"Peak:\s*([-+\w.]+)\s*dBFS", summary)

    def measured(match):
        value = float(match[1]) if match else None
        return value if value is not None and math.isfinite(value) else None

    result = {"status": "done", "audio_track_no": audio_track_no, "sample_rate": rate, "channels": channels,
              "channel_layout": stream.get("channel_layout"), "source_duration_seconds": float(container["duration"]) if container.get("duration") not in (None, "N/A") else None,
              "audio_start_time_seconds": audio_start, "audio_duration_seconds": audio_duration,
              "coverage": {"start_time": start_time, "end_time": start_time + frames / rate, "duration_seconds": frames / rate, "samples_per_channel": frames},
              "per_channel": [{"channel": index + 1, "peak_dbfs": db(peak[index]), "rms_dbfs": db(math.sqrt(squares[index] / frames)), "full_scale_samples": int(clipped[index])} for index in range(channels)],
              "loudness": {"method": "FFmpeg EBU R128", "integrated_lufs": measured(integrated), "true_peak_dbfs": measured(true_peak)},
              "silence": {"threshold_dbfs": silence_db, "window_seconds": window / rate, "minimum_seconds": min_silence,
                          "duration_seconds": silent_seconds, "interval_count": interval_count, "intervals": intervals, "truncated": interval_count > len(intervals)},
              "note": "Times are relative to the source file. Silence: all-channel RMS below the threshold in 20 ms windows. Full-scale samples suggest possible clipping. Null dBFS: zero amplitude."}
    if (measured(true_peak) or -100) > -1 or any(clipped):
        result["fix"] = "Peaks are above -1 dBTP: remux_media with mode \"mix\" and this one file (optionally normalize_lufs) masters it below the ceiling with its true-peak limiter, without regenerating."
    if preview_path:
        seconds = min(preview_seconds, frames / rate)
        # A compact stereo preview is deliberately bounded; measurements above retain all source channels.
        preview = command.copy()
        if duration is not None:
            preview = preview[:-2]
        completed = subprocess.run(preview + ["-t", str(seconds), "-ac", "2", "-ar", "24000", "-c:a", "pcm_s16le", "-y", str(preview_path)], capture_output=True, **child_kwargs())
        if completed.returncode:
            Path(preview_path).unlink(missing_ok=True)
            raise ValueError(completed.stderr.decode(errors="replace")[-1500:])
        result.update(output_file=str(preview_path), preview={"start_time": start_time, "duration_seconds": seconds, "sample_rate": 24000, "channels": 2})
    if window_s is not None:
        if len(envelope) > MAX_ENVELOPE_WINDOWS:
            raise ValueError(f"window_s gives more than {MAX_ENVELOPE_WINDOWS} windows for this range; use a larger window_s or a shorter duration.")
        # Momentary (400 ms) EBU R128 loudness, logged every 100 ms; each window keeps its loudest value.
        momentary = {}
        for time_text, value in _FRAME_LOG.findall(diagnostics):
            if math.isfinite(loudness := float(value)) and loudness > -70:
                index = min(int(max(float(time_text) - 0.05, 0) / window_s), max(envelope))
                momentary[index] = max(momentary.get(index, -math.inf), loudness)
        windows = [envelope[index] for index in range(max(envelope) + 1)]  # contiguous: every window of the range holds samples
        # whole dB: levels per window locate sections, gaps and balance (the overall figures above keep their decimals); peaks per window on request
        result["envelope"] = {"window_s": window_s, "start_s": start_time, "rms_dbfs": [None if (value := db(math.sqrt(total / count))) is None else round(value) for total, count, _ in windows],
                              "momentary_max_lufs": [None if (value := momentary.get(index)) is None else round(value) for index in range(len(windows))]}
        if window_peaks:
            result["envelope"]["peak_dbfs"] = [None if (value := db(top)) is None else round(value, 1) for _, _, top in windows]
    return result


BEAT_RATE, BEAT_HOP, MAX_LISTED_BEATS, MAX_BEAT_SECONDS = 22050, 512, 240, 900  # 23 ms onset frames


def music_beats(path, *, audio_track_no=1, start_time=0.0, duration=None):
    """Tempo, beat times and bar starts of music, in seconds of the source file. The tempo is the onset envelope's most prominent
    periodicity, preferring 70-140 BPM over its double or half; beats come from librosa's dynamic-programming tracker at that tempo,
    moved half a beat when the kick drum falls between them (a tracker locked on the off-beats or on the snare of a half-time feel).
    Bar starts assume 4 beats a bar and are given only when chord changes, bass changes and kick-versus-snare accents clearly agree
    on one beat of the four, and the chords do not follow a 3-beat bar instead."""
    import librosa
    from scipy.signal import find_peaks

    limit = MAX_BEAT_SECONDS + 1 if duration is None else min(duration, MAX_BEAT_SECONDS + 1)
    command = [_get_ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", str(start_time), "-i", str(path), "-t", str(limit),
               "-map", f"0:a:{audio_track_no - 1}", "-vn", "-ac", "1", "-ar", str(BEAT_RATE), "-f", "f32le", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, **child_kwargs())
    if completed.returncode:
        raise ValueError(f"Audio decoding failed: {completed.stderr.decode(errors='replace')[-1500:]}")
    samples, fps = np.frombuffer(completed.stdout, dtype="<f4"), BEAT_RATE / BEAT_HOP
    if len(samples) > MAX_BEAT_SECONDS * BEAT_RATE:
        raise ValueError(f"beats are measured over {MAX_BEAT_SECONDS // 60} minutes at most: choose a range with start_s and duration_s.")
    none = {"tempo_bpm": None, "note": "No steady beat found."}
    if len(samples) < 4 * BEAT_RATE:
        return none
    spectrum = np.abs(librosa.stft(samples, n_fft=2048, hop_length=BEAT_HOP))
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(librosa.feature.melspectrogram(S=spectrum ** 2, sr=BEAT_RATE)), sr=BEAT_RATE, center=False)
    centered = onset - onset.mean()
    if not centered.any():
        return none
    correlation = librosa.autocorrelate(centered, max_size=int(2 * fps) + 2)
    correlation = correlation / correlation[0] / (1 - np.arange(len(correlation)) / len(centered))  # unbiased: long lags overlap fewer frames
    shortest = int(fps * 60 / 240)  # lags of 240 to 40 BPM
    peaks, properties = find_peaks(correlation[shortest:int(np.ceil(fps * 60 / 40)) + 1], prominence=0.02)
    if not len(peaks):
        return none
    lags = peaks + shortest
    lag = lags[np.argmax(properties["prominences"] * np.exp(-0.5 * (np.log2(60 * fps / lags / 100) / 0.5) ** 2))]
    before, peak, after = correlation[lag - 1: lag + 2]
    curvature = before - 2 * peak + after
    tempo = 60 * fps / (lag + (0.5 * (before - after) / curvature if curvature else 0.0))
    beats = librosa.beat.beat_track(onset_envelope=onset, sr=BEAT_RATE, hop_length=BEAT_HOP, bpm=tempo, trim=True)[1]
    if len(beats) < 4:
        return none
    freqs = librosa.fft_frequencies(sr=BEAT_RATE, n_fft=2048)

    def flux(low, high):  # rise of the log energy of a band, per frame
        band = np.log1p(100 * spectrum[(freqs >= low) & (freqs < high)]).sum(0)
        return np.maximum(0, np.diff(band, prepend=band[:1]))

    def near(values, frames):  # strongest value within 2 frames of each frame
        return np.array([values[max(0, frame - 2): frame + 3].max() for frame in frames])

    kick = flux(0, 120)
    if len(beats) > 8 and near(kick, (beats[:-1] + beats[1:]) // 2).mean() > 1.5 * near(kick, beats[:-1]).mean():
        beats = (beats[:-1] + beats[1:]) // 2
    times = start_time + (beats + 1) / fps  # the energy rise of an attack peaks about one frame before it
    tempo = float(60 * (len(times) - 1) / (times[-1] - times[0]))
    result = {"tempo_bpm": round(tempo, 1), "beat_s": round(60 / tempo, 3), "times": [round(float(value), 2) for value in times[:MAX_LISTED_BEATS]]}
    spans = len(beats) - 1  # beat i spans frames beats[i]..beats[i + 1]
    if spans >= 12:
        def bar_change(features, beats_per_bar=4):  # cosine distance between the bar before each beat and the bar from it
            values = librosa.util.sync(features, beats, aggregate=np.mean)[:, 1:spans + 1]
            change = np.full(spans, np.nan)
            for index in range(beats_per_bar, spans - beats_per_bar + 1):
                before, after = values[:, index - beats_per_bar:index].mean(1), values[:, index:index + beats_per_bar].mean(1)
                change[index] = 1 - before @ after / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-9)
            return change

        def by_phase(values, beats_per_bar=4):
            return np.array([np.nanmean(values[phase::beats_per_bar]) for phase in range(beats_per_bar)])

        def contrast(values):  # how much more on one beat of the bar than on average
            return (values.max() - values.mean()) / values.mean()

        def standard(values):
            return (values - values.mean()) / (values.std() + 1e-9)

        chroma = librosa.feature.chroma_stft(S=spectrum ** 2, sr=BEAT_RATE)
        chords = bar_change(chroma)
        accent = standard(near(kick, beats[:spans])) - standard(near(flux(150, 400), beats[:spans]) + near(flux(1500, 6000), beats[:spans]))  # kick, not snare
        phases = np.stack([by_phase(chords), by_phase(bar_change(librosa.feature.chroma_stft(S=(spectrum * (freqs < 250)[:, None]) ** 2, sr=BEAT_RATE))), by_phase(accent)], 1)
        scores = ((phases - phases.mean(0)) / (phases.std(0) + 1e-9)).sum(1)
        best, runner_up = np.argsort(-scores)[:2]
        # chords changing more on a 3-beat grid than on a 4-beat one: 4-beat bars would drift through the bar
        if scores[best] - scores[runner_up] >= 1.5 and contrast(phases[:, 0]) >= contrast(by_phase(bar_change(chroma, 3), 3)):
            result["bar_starts"] = result["times"][best::4]
    notes = [] if "bar_starts" in result else ["Bar starts are unclear: count bars of 4 beats from a beat where a sung line or a section starts."]
    if len(times) > MAX_LISTED_BEATS:
        result["truncated"] = True
        notes.append(f"These are the first {MAX_LISTED_BEATS} of {len(times)} beats: inspect a later range (start_s) for the others.")
    if notes:
        result["note"] = " ".join(notes)
    return result


SPEECH_RATE, SPEECH_FRAME = 16000, 320  # 20 ms windows


def speech_intervals(path, *, start_time=0.0, duration=None, isolated=None, margin_db=9.0, min_speech=0.06, min_gap=0.15):
    """Spans of speech as [(start, end)] seconds from start_time, from speech-band (200-4000 Hz) energy in 20 ms windows.
    In a mixture the energy must rise margin_db above the local background, the 10th percentile over the surrounding 10 s, so a
    steady music bed or noise raises the threshold instead of counting as speech. An isolated voice, whose speech may not pause
    for seconds, needs only to stay within 40 dB of its peak; by default a recording with real silence (5% of its windows below
    -60 dBFS) is treated as isolated. Energy below -55 dBFS never counts."""
    command = [_get_ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", str(start_time), "-i", str(path), *(["-t", str(duration)] if duration else []),
               "-vn", "-af", "highpass=f=200,lowpass=f=4000", "-ac", "1", "-ar", str(SPEECH_RATE), "-f", "f32le", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, **child_kwargs())
    if completed.returncode:
        raise ValueError(f"Audio decoding failed: {completed.stderr.decode(errors='replace')[-1500:]}")
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    frames = samples[: len(samples) // SPEECH_FRAME * SPEECH_FRAME].reshape(-1, SPEECH_FRAME).astype(np.float64)
    if not len(frames):
        return []
    energy = 10 * np.log10(np.square(frames).mean(axis=1) + 1e-12)
    if isolated is None:
        isolated = np.percentile(energy, 5) < -60
    if isolated:
        threshold = np.full(len(energy), max(-55.0, energy.max() - 40.0))
    else:
        block = 25  # the background is estimated per 0.5 s block, over the 10 s around it
        blocks = [np.percentile(energy[max(0, index - 250): index + 250], 10) for index in range(0, len(energy), block)]
        threshold = np.maximum(np.repeat(blocks, block)[: len(energy)] + margin_db, -55.0)
    active = np.concatenate([[False], energy > threshold, [False]])
    edges = np.flatnonzero(active[1:] != active[:-1]) * SPEECH_FRAME / SPEECH_RATE
    spans = []
    for start, end in zip(edges[::2], edges[1::2]):
        if spans and start - spans[-1][1] < min_gap:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return [(round(float(start), 3), round(float(end), 3)) for start, end in spans if end - start >= min_speech]


def voice_pitch(path, *, start_time=0.0, duration=None):
    """Median fundamental frequency (Hz) of an isolated voice (a vocal stem or speech), from the autocorrelation of its clearly
    voiced 64 ms frames between 70 and 1000 Hz, or None without enough voiced frames: female voices mostly sit above about 165 Hz,
    male ones below. Meaningless on a full mix, whose instruments dominate the autocorrelation."""
    command = [_get_ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", str(start_time), "-i", str(path), *(["-t", str(duration)] if duration else []),
               "-vn", "-af", "lowpass=f=1200", "-ac", "1", "-ar", str(SPEECH_RATE), "-f", "f32le", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, **child_kwargs())
    if completed.returncode:
        raise ValueError(f"Audio decoding failed: {completed.stderr.decode(errors='replace')[-1500:]}")
    size = 1024
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    frames = samples[: len(samples) // size * size].reshape(-1, size).astype(np.float64)
    loud = frames[10 * np.log10(np.square(frames).mean(axis=1) + 1e-12) > -40]
    if len(loud) < 10:
        return None
    loud = loud - loud.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(loud, 2 * size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), 2 * size)[:, :size]
    low, high = SPEECH_RATE // 1000, SPEECH_RATE // 70
    lags = low + np.argmax(correlation[:, low:high], axis=1)
    clarity = correlation[np.arange(len(lags)), lags] / (correlation[:, 0] + 1e-12)
    voiced = lags[clarity > 0.6]
    return None if len(voiced) < 10 else round(float(np.median(SPEECH_RATE / voiced)), 1)
