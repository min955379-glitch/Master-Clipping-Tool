from __future__ import annotations

import array
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import patch

from core.brain import (
    BrainError,
    build_clip_prompt,
    call_anthropic,
    call_gemini,
    call_ollama,
    call_openai_compatible,
    parse_ranked_clips,
    rank_clips,
    route_provider,
)
from core.captions import CaptionError, generate_ass, write_ass
from core.encoder import EncoderSelection
from core.ingest import (
    IngestError,
    IngestServices,
    analyze_audio_peaks,
    classify_source,
    ingest_source,
    parse_json3_words,
    peaks_from_loudness,
    probe_media,
)
from core.models import (
    AudioPeak,
    CaptionPreset,
    ClipCandidate,
    ExportPreset,
    IngestResult,
    JobRequest,
    MediaMetadata,
    ProviderName,
    ReframeMode,
    SourceDescriptor,
    Word,
)
from core.reframe import ReframeServices, plan_reframe
from core.render import RenderServices, prepare_clip_source, render_clip


def sample_words() -> list[Word]:
    return [
        Word(start=10.0, end=10.35, text="This"),
        Word(start=10.36, end=10.72, text="idea"),
        Word(start=10.74, end=11.0, text="works."),
    ]


def sample_candidate(start: float = 10.0, end: float = 40.0, score: float = 91) -> ClipCandidate:
    return ClipCandidate(
        start=start,
        end=end,
        virality_score=score,
        hook_title="A surprising idea",
        reason="The speaker explains a useful idea and lands a strong payoff.",
        suggested_caption="Try this surprising idea.",
    )


def sample_ingest() -> IngestResult:
    return IngestResult(
        metadata=MediaMetadata(
            source_id="sample",
            title="Sample talk",
            duration=120,
            width=1920,
            height=1080,
            has_audio=True,
        ),
        source=SourceDescriptor(kind="local", source_id="sample", local_path=Path("sample.mp4")),
        words=sample_words(),
        audio_peaks=[AudioPeak(time=11, score=0.9, loudness_dbfs=-8, baseline_dbfs=-30)],
        transcript_source="faster_whisper",
    )


class FakeResponse:
    def __init__(self, data: dict[str, Any] | bytes) -> None:
        self.raw = data if isinstance(data, bytes) else json.dumps(data).encode()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.raw if size < 0 else self.raw[:size]


class FakePCMProcess:
    def __init__(self, samples: bytes) -> None:
        self.stdout = io.BytesIO(samples)
        self.returncode = 0

    def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


class IngestTests(unittest.TestCase):
    def test_classifies_local_and_supported_remote_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "clip.MP4"
            video.write_bytes(b"video")
            self.assertEqual(classify_source(str(video))[0], "local")
            self.assertEqual(classify_source("https://youtu.be/abcdefghijk")[0], "youtube")
            self.assertEqual(classify_source("https://www.twitch.tv/videos/12345")[0], "twitch")
            with self.assertRaises(IngestError):
                classify_source("https://example.org/video.mp4")
            with self.assertRaises(IngestError):
                classify_source("https://user:secret@youtube.com/watch?v=x")

    def test_probe_media_uses_ffprobe_json_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "talk.mp4"
            path.write_bytes(b"stub")
            calls: list[list[str]] = []

            def run(command: list[str], **_: Any) -> Any:
                calls.append(command)
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "format": {"duration": "45.25"},
                            "streams": [
                                {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30000/1001"},
                                {"codec_type": "audio"},
                            ],
                        }
                    ),
                    stderr="",
                )

            metadata = probe_media(path, run=run)
            self.assertEqual(metadata.duration, 45.25)
            self.assertEqual(metadata.fps, 30000 / 1001)
            self.assertTrue(metadata.has_audio)
            self.assertEqual(calls[0][0], "ffprobe")
            self.assertNotIn("shell", calls[0])

    def test_json3_word_timing_and_untimed_cue_distribution(self) -> None:
        timed = parse_json3_words(
            {
                "events": [
                    {
                        "tStartMs": 1000,
                        "dDurationMs": 2000,
                        "segs": [
                            {"utf8": "Hello", "tOffsetMs": 0},
                            {"utf8": "wonderful world", "tOffsetMs": 500},
                        ],
                    }
                ]
            },
            duration=5,
        )
        self.assertEqual([word.text for word in timed], ["Hello", "wonderful", "world"])
        self.assertEqual([word.start for word in timed], [1.0, 1.5, 2.25])
        self.assertTrue(all(word.end > word.start for word in timed))

        untimed = parse_json3_words(
            {"events": [{"tStartMs": 5000, "dDurationMs": 3000, "segs": [{"utf8": "one two three"}]}]}
        )
        self.assertEqual([round(word.start, 3) for word in untimed], [5.0, 6.0, 7.0])
        self.assertEqual([round(word.end, 3) for word in untimed], [5.8, 6.8, 7.8])

    def test_loudness_peaks_are_salient_and_separated(self) -> None:
        values = [-30.0] * 5 + [-10.0] + [-30.0] * 4 + [-5.0] + [-30.0] * 5
        peaks = peaks_from_loudness(values, window_seconds=1.0, min_gap_seconds=2.0)
        self.assertEqual([peak.time for peak in peaks], [5.0, 10.0])
        self.assertTrue(all(0 <= peak.score <= 1 for peak in peaks))

    def test_audio_analysis_streams_pcm_through_mock_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "audio.mp4"
            source.write_bytes(b"stub")
            pcm = array.array("h", [4000] * 8000).tobytes()
            process = FakePCMProcess(pcm)
            commands: list[list[str]] = []

            def popen(command: list[str], **_: Any) -> FakePCMProcess:
                commands.append(command)
                return process

            peaks = analyze_audio_peaks(source, popen=popen)
            self.assertEqual(peaks, [])
            self.assertIn("s16le", commands[0])
            self.assertIn("pipe:1", commands[0])

    def test_local_ingest_uses_probe_and_skips_disabled_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "local.mp4"
            video.write_bytes(b"stub")
            request = JobRequest(source=str(video), work_dir=Path(temporary) / "job", transcribe_missing=False)

            def run(_command: list[str], **_: Any) -> Any:
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "format": {"duration": 20},
                            "streams": [{"codec_type": "video", "width": 1280, "height": 720}],
                        }
                    ),
                    stderr="",
                )

            result = ingest_source(request, services=IngestServices(run=run))
            self.assertEqual(result.transcript_source, "none")
            self.assertEqual(result.metadata.duration, 20)
            self.assertEqual(result.words, [])
            self.assertIn("Source has no audio track", result.warnings[0])

    def test_youtube_captions_are_fast_path_and_asr_is_not_called(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            options_seen: list[dict[str, Any]] = []

            class FakeYDL:
                def __init__(self, options: dict[str, Any]) -> None:
                    self.options = options
                    options_seen.append(options)

                def __enter__(self) -> Self:
                    return self

                def __exit__(self, *_: object) -> None:
                    return None

                def extract_info(self, _url: str, download: bool = False) -> dict[str, Any]:
                    return {
                        "id": "yt123",
                        "title": "Captioned video",
                        "duration": 30,
                        "width": 1920,
                        "height": 1080,
                        "language": "en",
                        "automatic_captions": {
                            "en-orig": [{"ext": "json3", "url": "https://youtube.com/timedtext"}]
                        },
                    }

                def download(self, _urls: list[str]) -> None:
                    target = Path(self.options["outtmpl"].replace("%(ext)s", "m4a"))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"audio")

            caption_bytes = json.dumps(
                {"events": [{"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "Hello world", "tOffsetMs": 0}]}]}
            ).encode()
            audio_process = FakePCMProcess(array.array("h", [4000] * 8000).tobytes())
            transcription_calls: list[str] = []

            def whisper(**_: Any) -> Any:
                transcription_calls.append("called")
                raise AssertionError("ASR should be skipped when captions are usable")

            result = ingest_source(
                JobRequest(
                    source="https://www.youtube.com/watch?v=yt123",
                    work_dir=Path(temporary) / "job",
                    transcribe_missing=True,
                ),
                services=IngestServices(
                    ydl_factory=FakeYDL,
                    urlopen=lambda *_args, **_kwargs: FakeResponse(caption_bytes),
                    popen=lambda *_args, **_kwargs: audio_process,
                    whisper_factory=whisper,
                ),
            )
            self.assertEqual(result.transcript_source, "youtube_captions")
            self.assertEqual([word.text for word in result.words], ["Hello", "world"])
            self.assertFalse(transcription_calls)
            self.assertEqual(result.metadata.source_id, "yt123")
            self.assertEqual(len(options_seen), 2)


class BrainTests(unittest.TestCase):
    def test_prompt_combines_direction_transcript_and_audio_peaks(self) -> None:
        request = JobRequest(source="sample.mp4", custom_direction="Focus on the surprising moment")
        prompt = build_clip_prompt(sample_ingest(), request)
        self.assertIn("Focus on the surprising moment", prompt)
        self.assertIn("This", prompt)
        self.assertIn("relative peak 0.90", prompt)
        self.assertIn("absolute seconds", prompt)

    def test_ranked_clip_schema_sorting_bounds_and_overlap_deduplication(self) -> None:
        request = JobRequest(source="sample.mp4", clip_count=3, min_clip_seconds=15, max_clip_seconds=60)
        response = json.dumps(
            {
                "clips": [
                    {"start": 20, "end": 50, "virality_score": 70, "hook_title": "Good", "reason": "Useful", "suggested_caption": "Good clip"},
                    {"start": 21, "end": 49, "virality_score": 93, "hook_title": "Best", "reason": "Strong hook", "suggested_caption": "Watch this"},
                    {"start": 70, "end": 200, "virality_score": 99, "hook_title": "Out of range", "reason": "Invalid", "suggested_caption": ""},
                    {"start": 65, "end": 90, "score": 81, "title": "Second", "reason": "Different beat", "caption": "Next"},
                ]
            }
        )
        clips = parse_ranked_clips(response, duration=120, request=request)
        self.assertEqual([clip.hook_title for clip in clips], ["Best", "Second"])
        self.assertTrue(all(isinstance(clip, ClipCandidate) for clip in clips))
        self.assertTrue(all(0 <= clip.start < clip.end <= 120 for clip in clips))

    def test_rejects_invalid_or_out_of_range_model_output(self) -> None:
        request = JobRequest(source="sample.mp4")
        with self.assertRaises(BrainError):
            parse_ranked_clips("not json", duration=120, request=request)
        with self.assertRaises(BrainError):
            parse_ranked_clips(
                json.dumps({"clips": [{"start": 0, "end": 10, "virality_score": 150, "hook_title": "Short", "reason": "Too short"}]}),
                duration=120,
                request=request,
            )

    def test_ollama_provider_uses_native_local_chat_route_and_schema(self) -> None:
        requests: list[Any] = []
        response = FakeResponse({"message": {"content": "{\"clips\": []}"}})

        def opener(request: Any, timeout: float) -> FakeResponse:
            requests.append((request, timeout))
            return response

        content = call_ollama("select clips", base_url="http://localhost:11434/v1", model="qwen2.5:7b", opener=opener)
        self.assertEqual(content, "{\"clips\": []}")
        request, timeout = requests[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(timeout, 120)
        body = json.loads(request.data)
        self.assertEqual(body["model"], "qwen2.5:7b")
        self.assertIn("properties", body["format"])

    def test_gemini_provider_uses_opt_in_key_and_extracts_text(self) -> None:
        requests: list[Any] = []

        def opener(request: Any, timeout: float) -> FakeResponse:
            requests.append(request)
            return FakeResponse({"candidates": [{"content": {"parts": [{"text": "{\"clips\": []}"}]}}]})

        self.assertEqual(call_gemini("select clips", api_key="private-test-key", opener=opener), "{\"clips\": []}")
        self.assertEqual(requests[0].full_url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent")
        self.assertEqual(requests[0].get_header("X-goog-api-key"), "private-test-key")
        self.assertNotIn("private-test-key", requests[0].full_url)
        with self.assertRaises(BrainError):
            call_gemini("select clips", api_key="", opener=opener)

    def test_openai_compatible_provider_uses_saved_style_base_url_and_auth_header(self) -> None:
        requests: list[Any] = []

        def opener(request: Any, timeout: float) -> FakeResponse:
            requests.append(request)
            return FakeResponse({"choices": [{"message": {"content": "{\"clips\": []}"}}]})

        content = call_openai_compatible(
            "select clips",
            provider=ProviderName.OPENROUTER,
            api_key="private-openrouter-key",
            model="openai/gpt-test",
            base_url="https://proxy.example/v1/",
            opener=opener,
        )
        self.assertEqual(content, "{\"clips\": []}")
        request = requests[0]
        self.assertEqual(request.full_url, "https://proxy.example/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer private-openrouter-key")
        self.assertEqual(json.loads(request.data)["model"], "openai/gpt-test")
        self.assertNotIn("private-openrouter-key", request.full_url)

    def test_provider_http_errors_redact_api_keys_echoed_in_response_bodies(self) -> None:
        api_key = "private-openai-key"

        def opener(request: Any, timeout: float) -> FakeResponse:
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                {},
                io.BytesIO(f"invalid key {api_key}".encode()),
            )

        with self.assertRaises(BrainError) as error:
            call_openai_compatible(
                "select clips",
                provider=ProviderName.OPENAI,
                api_key=api_key,
                base_url="https://api.openai.com/v1",
                opener=opener,
            )
        self.assertNotIn(api_key, str(error.exception))
        self.assertIn("[redacted]", str(error.exception))

    def test_anthropic_messages_api_uses_custom_base_and_key_header(self) -> None:
        requests: list[Any] = []

        def opener(request: Any, timeout: float) -> FakeResponse:
            requests.append(request)
            return FakeResponse({"content": [{"type": "text", "text": "{\"clips\": []}"}]})

        content = call_anthropic(
            "select clips",
            api_key="private-anthropic-key",
            model="claude-test",
            base_url="https://proxy.example/api",
            opener=opener,
        )
        self.assertEqual(content, "{\"clips\": []}")
        request = requests[0]
        self.assertEqual(request.full_url, "https://proxy.example/api/v1/messages")
        self.assertEqual(request.get_header("X-api-key"), "private-anthropic-key")
        self.assertEqual(json.loads(request.data)["model"], "claude-test")

    def test_gemini_connection_test_sends_key_in_header_and_uses_configured_base(self) -> None:
        from core.brain import test_provider_connection

        requests: list[Any] = []

        def opener(request: Any, timeout: float) -> FakeResponse:
            requests.append(request)
            return FakeResponse({"candidates": [{"content": {"parts": [{"text": "OK"}]}}]})

        message = test_provider_connection(
            ProviderName.GEMINI,
            model="gemini-test",
            base_url="https://proxy.example/v1beta",
            api_key="private-gemini-key",
            opener=opener,
        )
        self.assertIn("Connected to Gemini", message)
        self.assertEqual(requests[0].full_url, "https://proxy.example/v1beta/models/gemini-test:generateContent")
        self.assertEqual(requests[0].get_header("X-goog-api-key"), "private-gemini-key")
        self.assertNotIn("private-gemini-key", requests[0].full_url)

    def test_route_provider_uses_persisted_settings_for_each_provider(self) -> None:
        saved = {
            "model": "persisted-model",
            "base_url": "https://proxy.example/v1",
            "api_key": "persisted-secret",
        }
        adapters = {
            ProviderName.OLLAMA: "call_ollama",
            ProviderName.GEMINI: "call_gemini",
            ProviderName.OPENAI: "call_openai_compatible",
            ProviderName.ANTHROPIC: "call_anthropic",
            ProviderName.GROQ: "call_openai_compatible",
            ProviderName.OPENROUTER: "call_openai_compatible",
        }
        for provider, adapter in adapters.items():
            with self.subTest(provider=provider.value):
                request = JobRequest(source="sample.mp4", provider=provider)
                with (
                    patch("core.brain.provider_config", return_value=saved) as load_settings,
                    patch(f"core.brain.{adapter}", return_value="provider response") as call,
                ):
                    result = route_provider("pick clips", request)
                self.assertEqual(result, "provider response")
                load_settings.assert_called_once_with(provider)
                self.assertEqual(call.call_args.kwargs["model"], saved["model"])
                if provider != ProviderName.OLLAMA:
                    self.assertEqual(call.call_args.kwargs["api_key"], saved["api_key"])
                    self.assertEqual(call.call_args.kwargs["base_url"], saved["base_url"])
                if provider in {ProviderName.OPENAI, ProviderName.GROQ, ProviderName.OPENROUTER}:
                    self.assertEqual(call.call_args.kwargs["provider"], provider)

    def test_provider_routing_is_injectable_without_network(self) -> None:
        request = JobRequest(source="sample.mp4", provider=ProviderName.OLLAMA)
        calls: list[tuple[str, str]] = []

        def provider(prompt: str, job: JobRequest) -> str:
            calls.append((prompt, job.provider.value))
            return json.dumps(
                {"clips": [{"start": 15, "end": 45, "virality_score": 88, "hook_title": "Hook", "reason": "Payoff", "suggested_caption": "Nice"}]}
            )

        clips = rank_clips(sample_ingest(), request, provider_call=provider)
        self.assertEqual(len(clips), 1)
        self.assertEqual(calls[0][1], "ollama")
        self.assertIn("AUDIO LOUDNESS", calls[0][0])


class FakeFrame:
    def __init__(self, width: int, height: int) -> None:
        self.shape = (height, width, 3)
        self.flags = SimpleNamespace(writeable=True)


class FakeCapture:
    def __init__(self, width: int, height: int, opened: bool = True) -> None:
        self.frame = FakeFrame(width, height)
        self.opened = opened
        self.positions: list[float] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def set(self, _prop: int, value: float) -> None:
        self.positions.append(value)

    def read(self) -> tuple[bool, FakeFrame]:
        return True, self.frame

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_PROP_POS_MSEC = 0
    COLOR_BGR2RGB = 1

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _path: str) -> FakeCapture:
        return self.capture

    def cvtColor(self, frame: FakeFrame, _code: int) -> FakeFrame:
        return frame


def fake_detection(x: float = 0.45, y: float = 0.2, width: float = 0.1, height: float = 0.2) -> Any:
    return SimpleNamespace(
        location_data=SimpleNamespace(
            relative_bounding_box=SimpleNamespace(xmin=x, ymin=y, width=width, height=height)
        ),
        score=[0.95],
    )


class ReframeTests(unittest.TestCase):
    def test_face_track_builds_smoothed_dynamic_crop_keyframes(self) -> None:
        capture = FakeCapture(1280, 720)
        cv2 = FakeCV2(capture)

        class Detector:
            def process(self, _image: Any) -> Any:
                return SimpleNamespace(detections=[fake_detection()])

            def close(self) -> None:
                pass

        plan = plan_reframe(
            "sample.mp4",
            source_width=1280,
            source_height=720,
            clip_start=10,
            duration=2,
            mode=ReframeMode.FACE_TRACK,
            services=ReframeServices(cv2_module=cv2, face_detector_factory=lambda **_: Detector()),
        )
        self.assertEqual(plan.output_width, 1080)
        self.assertEqual(plan.output_height, 1920)
        self.assertTrue(plan.keyframes)
        self.assertIn("crop=", plan.filter_complex)
        self.assertIn("trunc((", plan.filter_complex)
        self.assertTrue(capture.released)

    def test_no_face_falls_back_and_other_layouts_build_valid_graphs(self) -> None:
        capture = FakeCapture(1920, 1080)

        class Detector:
            def process(self, _image: Any) -> Any:
                return SimpleNamespace(detections=[])

            def close(self) -> None:
                pass

        services = ReframeServices(cv2_module=FakeCV2(capture), face_detector_factory=lambda **_: Detector())
        face = plan_reframe("sample.mp4", source_width=1920, source_height=1080, duration=5, services=services)
        self.assertIn("No face was detected", face.warnings[0])
        self.assertIn("scale=1080:1920", face.filter_complex)

        podcast = plan_reframe(
            "sample.mp4", source_width=1920, source_height=1080, duration=5, mode=ReframeMode.PODCAST_SPLIT
        )
        self.assertIn("vstack=inputs=2", podcast.filter_complex)
        gaming = plan_reframe(
            "sample.mp4", source_width=1920, source_height=1080, duration=5, mode=ReframeMode.GAMING, services=services
        )
        self.assertIn("overlay", gaming.filter_complex) if gaming.webcam_box else self.assertIn("No webcam", gaming.warnings[-1])
        fit = plan_reframe(
            "sample.mp4", source_width=1920, source_height=1080, duration=5, mode=ReframeMode.FIT_BLUR
        )
        self.assertIn("boxblur", fit.filter_complex)
        self.assertEqual(fit.output_width, 1080)

    def test_gaming_layout_places_detected_facecam_as_overlay(self) -> None:
        capture = FakeCapture(1920, 1080)

        class Detector:
            def process(self, _image: Any) -> Any:
                return SimpleNamespace(detections=[fake_detection(0.1, 0.1, 0.08, 0.15)])

            def close(self) -> None:
                pass

        plan = plan_reframe(
            "sample.mp4",
            source_width=1920,
            source_height=1080,
            duration=3,
            mode=ReframeMode.GAMING,
            export_preset=ExportPreset.SHORT_VERTICAL,
            services=ReframeServices(cv2_module=FakeCV2(capture), face_detector_factory=lambda **_: Detector()),
        )
        self.assertIsNotNone(plan.webcam_box)
        self.assertIn("overlay=", plan.filter_complex)
        self.assertIn("drawbox", plan.filter_complex)


class CaptionTests(unittest.TestCase):
    def test_ass_active_word_highlighting_is_clip_relative_and_style_specific(self) -> None:
        content = generate_ass(
            [Word(start=5, end=5.3, text="Hello"), Word(start=5.4, end=5.8, text="{world}")],
            clip_start=5,
            clip_end=8,
            preset=CaptionPreset.HORMOZI_BOLD,
        )
        self.assertIn("Style: Default,Arial Black", content)
        self.assertIn("{\\c&H0055ADD6&}Hello", content)
        self.assertIn("{\\c&H0055ADD6&}〔world〕", content)
        self.assertIn("Dialogue: 0,0:00:00.00", content)
        self.assertNotIn("{world}", content)

    def test_presets_and_fallback_caption_and_file_write(self) -> None:
        for preset in CaptionPreset:
            with self.subTest(preset=preset):
                content = generate_ass([], clip_start=0, clip_end=4, preset=preset, fallback_text="Fallback hook")
                self.assertIn("Fallback", content)
                self.assertIn("Style: Default", content)
        with tempfile.TemporaryDirectory() as temporary:
            path = write_ass(Path(temporary) / "captions" / "clip.ass", sample_words(), clip_start=10, clip_end=40)
            self.assertTrue(path.is_file())
            self.assertIn("Dialogue:", path.read_text(encoding="utf-8"))
        with self.assertRaises(CaptionError):
            generate_ass([], clip_start=4, clip_end=4)


class FakeYDL:
    def __init__(self, options: dict[str, Any], calls: list[dict[str, Any]], fail_partial: bool = False) -> None:
        self.options = options
        self.calls = calls
        self.fail_partial = fail_partial
        calls.append(options)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def download(self, _urls: list[str]) -> None:
        if "download_ranges" in self.options and self.fail_partial:
            raise RuntimeError("range not supported by extractor")
        target = Path(self.options["outtmpl"].replace("%(ext)s", "mkv"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"remote-video")


class RenderTests(unittest.TestCase):
    def test_remote_video_tries_partial_range_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calls: list[dict[str, Any]] = []
            prepared = prepare_clip_source(
                SourceDescriptor(kind="youtube", source_id="id", url="https://youtube.com/watch?v=id"),
                sample_candidate(15, 45),
                temporary,
                services=RenderServices(
                    ydl_factory=lambda options: FakeYDL(options, calls),
                    range_factory=lambda start, end: {"start": start, "end": end},
                ),
            )
            self.assertTrue(prepared.partial)
            self.assertEqual(prepared.source_start, 15)
            self.assertEqual(prepared.trim_start, 0)
            self.assertTrue(prepared.path.is_file())
            self.assertIn("download_ranges", calls[0])

    def test_remote_download_falls_back_to_full_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calls: list[dict[str, Any]] = []
            prepared = prepare_clip_source(
                SourceDescriptor(kind="twitch", source_id="id", url="https://twitch.tv/videos/1"),
                sample_candidate(20, 50),
                temporary,
                services=RenderServices(
                    ydl_factory=lambda options: FakeYDL(options, calls, fail_partial=True),
                    range_factory=lambda _start, _end: object(),
                ),
            )
            self.assertFalse(prepared.partial)
            self.assertEqual(prepared.trim_start, 20)
            self.assertTrue(prepared.warnings)
            self.assertEqual(len(calls), 2)

    def test_render_burns_ass_normalizes_audio_and_retries_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "local.mp4"
            video.write_bytes(b"source")
            output = root / "exports" / "clip.mp4"
            calls: list[list[str]] = []

            def run(command: list[str], **_: Any) -> Any:
                calls.append(command)
                if "h264_amf" in command:
                    return SimpleNamespace(returncode=1, stderr="AMF device unavailable", stdout="")
                Path(command[-1]).write_bytes(b"rendered")
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            selection = EncoderSelection(
                name="h264_amf",
                hardware_accelerated=True,
                reason="test AMF probe",
                available=("h264_amf", "libx264"),
            )
            from core.models import ReframePlan

            plan = ReframePlan(
                requested_mode=ReframeMode.FIT_BLUR,
                mode=ReframeMode.FIT_BLUR,
                source_width=1920,
                source_height=1080,
                output_width=1080,
                output_height=1920,
                filter_complex="[0:v]scale=1080:1920[vout]",
            )
            result = render_clip(
                video,
                sample_candidate(5, 35),
                plan,
                sample_words(),
                output,
                work_dir=root / "job",
                suggested_caption="A helpful caption",
                services=RenderServices(run=run, encoder_probe=lambda _ffmpeg: selection),
            )
            self.assertEqual(result.encoder, "libx264")
            self.assertEqual(result.duration, 30)
            self.assertTrue(result.path.is_file())
            self.assertEqual(len(calls), 2)
            self.assertIn("-ss", calls[0])
            self.assertIn("loudnorm=I=-14:TP=-1.5:LRA=11:print_format=summary", calls[0])
            self.assertIn("subtitles=filename=", calls[0][calls[0].index("-filter_complex") + 1])
            self.assertEqual(calls[0][calls[0].index("-c:v") + 1], "h264_amf")
            self.assertEqual(calls[1][calls[1].index("-c:v") + 1], "libx264")
            self.assertIn("CPU libx264", result.warnings[0])

    def test_production_render_streams_progress_and_diagnostics_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "local.mp4"
            video.write_bytes(b"source")
            from core.models import ReframePlan

            plan = ReframePlan(
                requested_mode=ReframeMode.FIT_BLUR,
                mode=ReframeMode.FIT_BLUR,
                source_width=1920,
                source_height=1080,
                output_width=1080,
                output_height=1920,
                filter_complex="[0:v]scale=1080:1920[vout]",
            )
            output = root / "clip.mp4"
            callback_events: list[tuple[str, float, str]] = []

            class FakeProcess:
                def __init__(self) -> None:
                    self.stdout = iter(["out_time_us=10000000\n", "out_time_ms=15000000\n", "progress=end\n"])
                    self.stderr = iter(["frame=1\n", "encoded\n"])
                    self.returncode = 0
                    self.killed = False

                def poll(self) -> int | None:
                    return self.returncode

                def wait(self, timeout: float | None = None) -> int:
                    return self.returncode

                def kill(self) -> None:
                    self.killed = True
                    self.returncode = -9

            process = FakeProcess()
            popen_calls: list[dict[str, Any]] = []

            def popen(command: list[str], **kwargs: Any) -> FakeProcess:
                popen_calls.append(kwargs)
                Path(command[-1]).write_bytes(b"rendered")
                return process

            selection = EncoderSelection("libx264", False, "CPU", ("libx264",))
            result = render_clip(
                video,
                sample_candidate(5, 35),
                plan,
                sample_words(),
                output,
                work_dir=root / "job",
                services=RenderServices(popen=popen, encoder_probe=lambda _ffmpeg: selection),
                progress=lambda stage, fraction, message: callback_events.append((stage, fraction, message)),
            )
            self.assertEqual(result.encoder, "libx264")
            self.assertEqual(callback_events[-1][1], 1.0)
            progress_fractions = [event[1] for event in callback_events if "Encoding with FFmpeg" in event[2]]
            self.assertAlmostEqual(progress_fractions[0], 1 / 3)
            self.assertEqual(progress_fractions[1], 0.5)
            self.assertEqual(len(progress_fractions), 2)
            self.assertTrue(popen_calls)
            self.assertTrue(popen_calls[0]["stdout"])
            self.assertTrue(popen_calls[0]["stderr"])

    def test_render_without_audio_does_not_add_loudness_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "local.mp4"
            video.write_bytes(b"source")
            from core.models import ReframePlan

            plan = ReframePlan(
                requested_mode=ReframeMode.FIT_BLUR,
                mode=ReframeMode.FIT_BLUR,
                source_width=1920,
                source_height=1080,
                output_width=1080,
                output_height=1920,
                filter_complex="[0:v]scale=1080:1920[vout]",
            )
            commands: list[list[str]] = []

            def run(command: list[str], **_: Any) -> Any:
                commands.append(command)
                Path(command[-1]).write_bytes(b"rendered")
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            selection = EncoderSelection("libx264", False, "CPU", ("libx264",))
            render_clip(
                video,
                sample_candidate(5, 35),
                plan,
                sample_words(),
                root / "clip.mp4",
                work_dir=root / "job",
                has_audio=False,
                services=RenderServices(run=run, encoder_probe=lambda _ffmpeg: selection),
            )
            self.assertNotIn("loudnorm=I=-14:TP=-1.5:LRA=11:print_format=summary", commands[0])
            self.assertNotIn("0:a:0?", commands[0])


if __name__ == "__main__":
    unittest.main()
