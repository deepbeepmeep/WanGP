# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""File-backed assembly regression tests.

Run: python -m unittest discover -s plugins/showrunner/tests -v
"""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

from plugins.showrunner import assemble, media


class FileBackedGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="showrunner graph ")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def command(self, graph="sine=frequency=440:duration=0.1[out]"):
        return [media.ffmpeg(), "-y", "-v", "error", "-filter_complex", graph,
                "-map", "[out]", "-c:a", "pcm_s16le", str(self.folder / "sound ü.wav")]

    def assert_sound(self):
        with wave.open(str(self.folder / "sound ü.wav"), "rb") as sound:
            self.assertEqual(sound.getnframes(), 4410)
            self.assertNotEqual(set(sound.readframes(4410)), {0})

    def test_graph_larger_than_windows_command_line_runs_from_file(self):
        # Passing the graph inline would exceed CreateProcess's 32K limit.
        graph = "sine=frequency=440:duration=0.1" + " " * 40000 + "[out]"
        assemble._run_graph(self.command(graph), timeout=30)
        self.assert_sound()

    def test_old_ffmpeg_retry_reuses_graph_file(self):
        legacy = shutil.which("ffmpeg")
        if not legacy:
            self.skipTest("No system FFmpeg for legacy execution")
        run = media._run
        probe = self.folder / "legacy graph.txt"
        probe.write_text("sine=frequency=440:duration=0.1[out]", encoding="utf-8")
        try:
            run([legacy, "-v", "error", "-filter_complex_script", str(probe),
                 "-map", "[out]", "-f", "null", "-"], timeout=30)
        except RuntimeError as error:
            if "Unrecognized option 'filter_complex_script'" in str(error):
                self.skipTest("System FFmpeg no longer supports the legacy graph option")
            raise
        scripts = []

        def old_option_parser(command, **kwargs):
            # Only the old option parser is simulated; the fallback executes FFmpeg.
            option = "-/filter_complex" if "-/filter_complex" in command else "-filter_complex_script"
            script = Path(command[command.index(option) + 1])
            scripts.append((script, script.read_text(encoding="utf-8")))
            if option == "-/filter_complex":
                raise RuntimeError("ffmpeg failed: Unrecognized option '/filter_complex'.")
            return run([legacy, *command[1:]], **kwargs)

        with patch.object(media, "_run", side_effect=old_option_parser):
            assemble._run_graph(self.command(), timeout=30)
        self.assert_sound()
        self.assertEqual(len(scripts), 2)
        self.assertEqual(scripts[0], scripts[1])

    def test_bad_filter_is_reported_without_retry(self):
        run = media._run
        errors = []

        def record_errors(command, **kwargs):
            try:
                return run(command, **kwargs)
            except RuntimeError as error:
                errors.append(str(error))
                raise

        with patch.object(media, "_run", side_effect=record_errors):
            with self.assertRaisesRegex(RuntimeError, "No such filter"):
                assemble._run_graph(self.command("no_such_showrunner_filter[out]"), timeout=30)
        # A genuinely old binary may reject the modern flag before parsing the graph.
        if len(errors) == 2:
            self.assertIn("Unrecognized option '/filter_complex'", errors[0])
        else:
            self.assertEqual(len(errors), 1)
        self.assertIn("No such filter", errors[-1])


if __name__ == "__main__":
    unittest.main()
