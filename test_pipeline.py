"""End-to-end smoke harness: deterministic mocks by default, opt-in real ``--live`` mode."""

from __future__ import annotations

import argparse
import array
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.encoder import EncoderSelection, detect_encoder
from core.env import load_local_env
from core.ingest import IngestServices, probe_media
from core.models import (
    CaptionPreset,
    ClipCandidate,
    ExportPreset,
    JobRequest,
    PipelineProgress,
    ProviderName,
    ReframeMode,
    Word,
)
from core.pipeline import PipelineServices, run_pipeline
from core.reframe import plan_reframe
from core.render import RenderServices, render_clip


class _FakePCMProcess:
    def __init__(self, data: bytes) -> None:
        self.stdout = io.BytesIO(data)
        self.returncode = 0

    def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _mocked_check() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="frameforge-test-") as temporary:
        root = Path(temporary)
        source = root / "mock-source.mp4"
        source.write_bytes(b"mock video input")
        work_dir = root / "job"
        probe_calls: list[list[str]] = []
        ffmpeg_calls: list[list[str]] = []
        transcription_settings: list[tuple[str, str]] = []
        prompts: list[str] = []

        def ffprobe(command: list[str], **_: Any) -> Any:
            probe_calls.append(command)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "format": {"duration": 120},
                        "streams": [
                            {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1"},
                            {"codec_type": "audio"},
                        ],
                    }
                ),
                stderr="",
            )

        amplitudes = [4000] * 4 + [20000] + [4000] * 5
        pcm = b"".join(array.array("h", [amplitude] * 8000).tobytes() for amplitude in amplitudes)

        def pcm_ffmpeg(command: list[str], **_: Any) -> _FakePCMProcess:
            if "pipe:1" not in command:
                raise AssertionError("audio analysis should request streamed PCM")
            return _FakePCMProcess(pcm)

        class MockWhisper:
            def __init__(self, model: str, **options: Any) -> None:
                transcription_settings.append((model, str(options["compute_type"])))

            def transcribe(self, _path: str, **options: Any) -> tuple[list[Any], None]:
                if options.get("word_timestamps") is not True:
                    raise AssertionError("faster-whisper must return word timestamps")
                words = [
                    SimpleNamespace(start=10.1, end=10.4, word="Here"),
                    SimpleNamespace(start=10.4, end=10.6, word="is"),
                    SimpleNamespace(start=10.6, end=11.0, word="the"),
                    SimpleNamespace(start=11.0, end=11.5, word="moment."),
                    SimpleNamespace(start=12.0, end=12.5, word="Stay"),
                    SimpleNamespace(start=12.5, end=13.0, word="with"),
                    SimpleNamespace(start=13.0, end=13.4, word="me."),
                ]
                return [SimpleNamespace(words=words)], None

        def local_provider(prompt: str, _request: JobRequest) -> str:
            prompts.append(prompt)
            return json.dumps(
                {
                    "clips": [
                        {
                            "start": 10,
                            "end": 40,
                            "virality_score": 92,
                            "hook_title": "The moment everything changes",
                            "reason": "The speaker sets up a moment and delivers a clear payoff.",
                            "suggested_caption": "Stay with this moment.",
                        }
                    ]
                }
            )

        selection = EncoderSelection(
            name="libx264",
            hardware_accelerated=False,
            reason="Mocked CPU fallback",
            available=("libx264",),
        )

        def ffmpeg_render(command: list[str], **_: Any) -> Any:
            ffmpeg_calls.append(command)
            if "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=summary" not in command:
                raise AssertionError("render must normalize audio to -14 LUFS")
            if not any("subtitles=filename=" in item for item in command):
                raise AssertionError("render must burn in an ASS captions file")
            Path(command[-1]).write_bytes(b"mock mp4 output")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        request = JobRequest(
            source=str(source),
            job_id=uuid.uuid4().hex,
            work_dir=work_dir,
            output_dir=work_dir / "exports",
            provider=ProviderName.OLLAMA,
            custom_direction="Keep the surprising payoff and leave out the intro.",
            layout=ReframeMode.FIT_BLUR,
            caption_style=CaptionPreset.HORMOZI_BOLD,
            export_preset=ExportPreset.SHORT_VERTICAL,
            clip_count=1,
            min_clip_seconds=15,
            max_clip_seconds=60,
        )
        progress_events: list[PipelineProgress] = []
        services = PipelineServices(
            ingest=IngestServices(run=ffprobe, popen=pcm_ffmpeg, whisper_factory=MockWhisper),
            provider_call=local_provider,
            render=RenderServices(
                run=ffmpeg_render,
                encoder_probe=lambda _ffmpeg: selection,
            ),
        )
        result = run_pipeline(request, services=services, progress=progress_events.append)

        assert result.status == "completed", result.error or result.warnings
        assert len(result.clips) == 1
        assert result.clips[0].render.path.is_file()
        assert result.clips[0].render.path.suffix == ".mp4"
        assert list((work_dir / "captions").glob("*.ass")), "ASS captions were not generated"
        assert probe_calls and "ffprobe" in probe_calls[0][0]
        assert ffmpeg_calls and "libx264" in ffmpeg_calls[0]
        assert transcription_settings == [("small", "int8")]
        assert "Keep the surprising payoff" in prompts[0]
        assert "moment." in prompts[0]
        assert "AUDIO LOUDNESS" in prompts[0]
        assert any(event.stage == "COMPLETED" and event.progress == 1 for event in progress_events)

        return {
            "mode": "mocked",
            "status": result.status,
            "job_id": result.job_id,
            "clips": len(result.clips),
            "encoder": result.clips[0].render.encoder,
            "transcription": "faster-whisper CPU/int8 mocked",
            "ffmpeg_calls": len(ffmpeg_calls),
            "provider_calls": len(prompts),
            "progress_events": len(progress_events),
        }


def _ollama_status() -> dict[str, Any]:
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(base)
        path = parsed.path.rstrip("/").removesuffix("/v1")
        endpoint = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path + "/api/tags", "", ""))
        request = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read(1_000_000).decode("utf-8"))
        return {"reachable": True, "models_listed": len(payload.get("models") or []), "endpoint": endpoint}
    except (OSError, ValueError, urllib.error.URLError) as exc:
        # This is a liveness probe only; it never issues a model-pull request.
        return {"reachable": False, "error": str(getattr(exc, "reason", exc))}


def _live_runtime_smoke(work_dir: Path) -> dict[str, Any]:
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    ffprobe = os.environ.get("FFPROBE_PATH", "ffprobe")
    selection = detect_encoder(ffmpeg)
    source_encoder = "libx264" if "libx264" in selection.available else selection.name
    smoke_dir = work_dir / "synthetic-render-check"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    source_path = smoke_dir / "synthetic-15s.mp4"
    source_args = ["-preset", "ultrafast", "-crf", "28"]
    if source_encoder == "h264_amf":
        source_args = ["-quality", "speed", "-b:v", "4M"]
    elif source_encoder == "h264_nvenc":
        source_args = ["-preset", "p1", "-cq", "30", "-b:v", "0"]
    command = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-t",
        "15",
        "-c:v",
        source_encoder,
        *source_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(source_path),
    ]
    try:
        generated = subprocess.run(command, capture_output=True, text=True, check=False, timeout=240)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"Could not generate a synthetic FFmpeg source: {exc}") from exc
    if generated.returncode != 0 or not source_path.is_file():
        detail = (generated.stderr or generated.stdout or "no FFmpeg diagnostic output").strip()
        raise RuntimeError(f"Synthetic source generation failed: {detail[-900:]}")
    source_metadata = probe_media(source_path, ffprobe=ffprobe)
    if abs(source_metadata.duration - 15) > 0.75 or not source_metadata.has_audio:
        raise RuntimeError("Synthetic FFmpeg source did not probe as a 15-second audio/video file.")

    plan = plan_reframe(
        source_path,
        source_width=source_metadata.width or 640,
        source_height=source_metadata.height or 360,
        duration=15,
        mode=ReframeMode.FIT_BLUR,
        export_preset=ExportPreset.SHORT_VERTICAL,
    )
    candidate = ClipCandidate(
        start=0,
        end=15,
        virality_score=100,
        hook_title="Synthetic render check",
        reason="A generated test pattern validates the local export path.",
        suggested_caption="Synthetic render check",
    )
    words = [
        Word(start=0.4, end=1.0, text="Synthetic"),
        Word(start=1.0, end=1.7, text="render"),
        Word(start=1.7, end=2.3, text="check"),
    ]
    output_path = smoke_dir / "rendered-15s-vertical.mp4"
    rendered = render_clip(
        source_path,
        candidate,
        plan,
        words,
        output_path,
        work_dir=smoke_dir / "job",
        caption_preset=CaptionPreset.CLEAN_MINIMAL.value,
        suggested_caption="Synthetic render check",
        has_audio=True,
        ffmpeg=ffmpeg,
        services=RenderServices(encoder_probe=lambda _ffmpeg: selection),
    )
    output_metadata = probe_media(rendered.path, ffprobe=ffprobe)
    if (
        abs(output_metadata.duration - 15) > 1.0
        or output_metadata.width != plan.output_width
        or output_metadata.height != plan.output_height
        or not output_metadata.has_audio
    ):
        raise RuntimeError("Rendered sample failed duration, geometry or audio probing.")
    return {
        "encoder_probe": {
            "selected": selection.name,
            "hardware_accelerated": selection.hardware_accelerated,
            "supported": list(selection.available),
        },
        "ollama": _ollama_status(),
        "synthetic_source": {
            "path": str(source_path),
            "duration": source_metadata.duration,
            "encoder": source_encoder,
        },
        "render": {
            "path": str(rendered.path),
            "encoder": rendered.encoder,
            "duration": output_metadata.duration,
            "width": output_metadata.width,
            "height": output_metadata.height,
            "has_audio": output_metadata.has_audio,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FrameForge pipeline end-to-end smoke checks")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="Probe FFmpeg/Ollama and render a 15-second synthetic sample; optionally run a real source",
    )
    mode.add_argument("--mocked", action="store_true", help="Run deterministic no-network/no-FFmpeg mocks (default)")
    parser.add_argument("--source", help="Optional local video path or YouTube/Twitch URL for a full live pipeline run")
    parser.add_argument("--provider", choices=[item.value for item in ProviderName], default="ollama")
    parser.add_argument("--model", help="Optional provider model override")
    parser.add_argument("--custom-direction", default="", help="Creative direction used in --live mode")
    parser.add_argument("--layout", choices=[item.value for item in ReframeMode], default="face_track")
    parser.add_argument("--caption-style", choices=[item.value for item in CaptionPreset], default="Hormozi Bold")
    parser.add_argument("--export-preset", choices=[item.value for item in ExportPreset], default="short_vertical")
    parser.add_argument("--clip-count", type=int, default=4)
    parser.add_argument("--min-clip-seconds", type=float, default=15)
    parser.add_argument("--max-clip-seconds", type=float, default=60)
    parser.add_argument("--work-dir", type=Path, help="Override local job storage for --live")
    parser.add_argument("--json", action="store_true", help="Print a JSON result summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = _parser().parse_args(argv)
    if not args.live:
        summary = _mocked_check()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    try:
        from core.storage import default_data_root

        smoke_root = (args.work_dir or (default_data_root() / "live-checks")).expanduser().resolve()
        runtime = _live_runtime_smoke(smoke_root)
        payload: dict[str, Any] = {"mode": "live", "status": "runtime-smoke-passed", "runtime": runtime}
        if args.source:
            request = JobRequest(
                source=args.source,
                job_id=uuid.uuid4().hex,
                work_dir=(args.work_dir / "pipeline") if args.work_dir else None,
                provider=ProviderName(args.provider),
                model=args.model,
                custom_direction=args.custom_direction,
                layout=ReframeMode(args.layout),
                caption_style=CaptionPreset(args.caption_style),
                export_preset=ExportPreset(args.export_preset),
                clip_count=args.clip_count,
                min_clip_seconds=args.min_clip_seconds,
                max_clip_seconds=args.max_clip_seconds,
            )
            progress = None if args.json else lambda event: print(
                f"{event.progress:6.1%} {event.stage}: {event.message}", file=sys.stderr, flush=True
            )
            result = run_pipeline(request, progress=progress)
            payload["pipeline"] = result.model_dump(mode="json")
            payload["status"] = result.status
            if result.status == "failed":
                print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
                return 1
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - a smoke script should report external tool failures cleanly.
        print(json.dumps({"mode": "live", "status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
