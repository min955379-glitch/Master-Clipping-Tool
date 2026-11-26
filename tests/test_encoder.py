"""Fast, hardware-independent tests for the FFmpeg encoder probe."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from core.encoder import (
    EncoderProbeError,
    _cached_detect_encoder,
    detect_encoder,
    parse_encoder_listing,
    select_encoder,
)

SAMPLE_LISTING = """
Encoders:
 V..... h264_amf AMD AMF H.264 Encoder
 V....D h264_nvenc NVIDIA NVENC H.264 encoder
 V....D libx264 libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10
 A..... aac AAC (Advanced Audio Coding)
 ------
"""


class ParseEncoderListingTests(unittest.TestCase):
    def test_parses_video_encoder_names_and_ignores_other_rows(self) -> None:
        names = parse_encoder_listing(SAMPLE_LISTING)
        self.assertEqual(names, {"h264_amf", "h264_nvenc", "libx264", "aac"})

    def test_ignores_malformed_flags_and_headers(self) -> None:
        names = parse_encoder_listing("""Encoders:\n V..... h264_amf AMF\n V.... h264_bad malformed\n""")
        self.assertEqual(names, {"h264_amf"})


class EncoderSelectionTests(unittest.TestCase):
    def test_prefers_amf_when_all_candidates_are_available(self) -> None:
        selection = select_encoder({"h264_amf", "h264_nvenc", "libx264"})
        self.assertEqual(selection.name, "h264_amf")
        self.assertTrue(selection.hardware_accelerated)

    def test_uses_nvenc_when_amf_is_not_available(self) -> None:
        selection = select_encoder({"h264_nvenc", "libx264"})
        self.assertEqual(selection.name, "h264_nvenc")
        self.assertTrue(selection.hardware_accelerated)

    def test_falls_back_to_libx264(self) -> None:
        selection = select_encoder({"libx264"})
        self.assertEqual(selection.name, "libx264")
        self.assertFalse(selection.hardware_accelerated)

    def test_fails_clearly_when_no_supported_h264_encoder_exists(self) -> None:
        with self.assertRaisesRegex(EncoderProbeError, "none of the supported H.264 encoders"):
            select_encoder({"mpeg4", "aac"})


class EncoderProbeTests(unittest.TestCase):
    @patch("core.encoder.subprocess.run")
    def test_default_runtime_probe_is_cached_per_executable(self, run) -> None:
        _cached_detect_encoder.cache_clear()
        self.addCleanup(_cached_detect_encoder.cache_clear)
        run.side_effect = [
            subprocess.CompletedProcess(args=["ffmpeg", "-encoders"], returncode=0, stdout=SAMPLE_LISTING, stderr=""),
            subprocess.CompletedProcess(args=["ffmpeg", "-f", "lavfi"], returncode=0, stdout="", stderr=""),
        ]
        first = detect_encoder("cached-ffmpeg")
        second = detect_encoder("cached-ffmpeg")
        self.assertEqual(first, second)
        self.assertEqual(run.call_count, 2)

    @patch("core.encoder.subprocess.run")
    def test_detect_encoder_verifies_amf_with_one_runtime_frame(self, run) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(args=["ffmpeg", "-encoders"], returncode=0, stdout=SAMPLE_LISTING, stderr=""),
            subprocess.CompletedProcess(args=["ffmpeg", "-f", "lavfi"], returncode=0, stdout="", stderr=""),
        ]

        selection = detect_encoder("ffmpeg.exe", run=run)

        self.assertEqual(selection.name, "h264_amf")
        self.assertIn("one-frame runtime probe passed", selection.reason)
        self.assertEqual(run.call_count, 2)
        run.assert_any_call(
            ["ffmpeg.exe", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10.0,
        )
        runtime_command = run.call_args_list[1].args[0]
        self.assertIn("-frames:v", runtime_command)
        self.assertIn("h264_amf", runtime_command)
        self.assertEqual(runtime_command[-2:], ["null", "-"])

    @patch("core.encoder.subprocess.run")
    def test_falls_back_to_libx264_when_hardware_probe_fails(self, run) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(args=["ffmpeg", "-encoders"], returncode=0, stdout=SAMPLE_LISTING, stderr=""),
            subprocess.CompletedProcess(args=["ffmpeg", "-amf"], returncode=1, stdout="", stderr="AMF unavailable"),
            subprocess.CompletedProcess(args=["ffmpeg", "-nvenc"], returncode=1, stdout="", stderr="NVENC unavailable"),
        ]
        selection = detect_encoder(run=run)
        self.assertEqual(selection.name, "libx264")
        self.assertFalse(selection.hardware_accelerated)
        self.assertIn("AMF unavailable", selection.reason)
        self.assertIn("NVENC unavailable", selection.reason)
        self.assertEqual(run.call_count, 3)

    @patch("core.encoder.subprocess.run")
    def test_tries_nvenc_after_amf_runtime_probe_fails(self, run) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(args=["ffmpeg", "-encoders"], returncode=0, stdout=SAMPLE_LISTING, stderr=""),
            subprocess.CompletedProcess(args=["ffmpeg", "-amf"], returncode=1, stdout="", stderr="AMF unavailable"),
            subprocess.CompletedProcess(args=["ffmpeg", "-nvenc"], returncode=0, stdout="", stderr=""),
        ]
        selection = detect_encoder(run=run)
        self.assertEqual(selection.name, "h264_nvenc")
        self.assertTrue(selection.hardware_accelerated)
        self.assertEqual(run.call_count, 3)

    @patch("core.encoder.subprocess.run", side_effect=FileNotFoundError)
    def test_reports_missing_ffmpeg_without_traceback(self, _run) -> None:
        with self.assertRaisesRegex(EncoderProbeError, "FFmpeg executable not found"):
            detect_encoder("ffmpeg.exe", run=_run)

    @patch(
        "core.encoder.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=1, stdout="", stderr="probe error"
        ),
    )
    def test_reports_nonzero_ffmpeg_exit(self, _run) -> None:
        with self.assertRaisesRegex(EncoderProbeError, "exit 1.*probe error"):
            detect_encoder(run=_run)


if __name__ == "__main__":
    unittest.main()
