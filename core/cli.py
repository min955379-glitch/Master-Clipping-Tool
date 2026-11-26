"""Command-line adapter for the same pipeline used by the web API and MCP server."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from core.env import load_local_env
from core.models import (
    CaptionPreset,
    ExportPreset,
    JobRequest,
    PipelineProgress,
    ProviderName,
    ReframeMode,
)
from core.pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frameforge",
        description="Find and render short-form video clips locally.",
    )
    parser.add_argument("--source", required=True, help="Local video path or YouTube/Twitch URL")
    parser.add_argument("--provider", choices=[item.value for item in ProviderName], default=ProviderName.OLLAMA.value)
    parser.add_argument("--model", help="Optional provider model override")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Override the Ollama base URL")
    parser.add_argument("--direction", default="", help="Optional creative direction for clip selection")
    parser.add_argument("--language", help="Optional transcription language code")
    parser.add_argument("--layout", choices=[item.value for item in ReframeMode], default=ReframeMode.FACE_TRACK.value)
    parser.add_argument("--caption-style", choices=[item.value for item in CaptionPreset], default=CaptionPreset.HORMOZI_BOLD.value)
    parser.add_argument("--export-preset", choices=[item.value for item in ExportPreset], default=ExportPreset.SHORT_VERTICAL.value)
    parser.add_argument("--clip-count", type=int, default=4, help="Number of clips to request (1-12)")
    parser.add_argument("--min-clip-seconds", type=float, default=15)
    parser.add_argument("--max-clip-seconds", type=float, default=60)
    parser.add_argument("--asr-model", default="small", help="faster-whisper model name (CPU/int8)")
    parser.add_argument("--work-dir", type=Path, help="Override the per-user local job folder")
    parser.add_argument("--output-dir", type=Path, help="Override the export folder")
    parser.add_argument("--no-transcribe", action="store_true", help="Skip faster-whisper when captions are absent")
    parser.add_argument("--json", action="store_true", help="Print the typed result as JSON")
    return parser


def _progress(event: PipelineProgress) -> None:
    percent = round(event.progress * 100)
    clip = f" [{event.current_clip}/{event.total_clips}]" if event.current_clip and event.total_clips else ""
    print(f"{percent:3d}% {event.stage}{clip}: {event.message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = _parser().parse_args(argv)
    try:
        request = JobRequest(
            source=args.source,
            job_id=uuid.uuid4().hex,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            provider=ProviderName(args.provider),
            model=args.model,
            base_url=args.base_url,
            custom_direction=args.direction,
            language=args.language,
            layout=ReframeMode(args.layout),
            caption_style=CaptionPreset(args.caption_style),
            export_preset=ExportPreset(args.export_preset),
            clip_count=args.clip_count,
            min_clip_seconds=args.min_clip_seconds,
            max_clip_seconds=args.max_clip_seconds,
            transcribe_missing=not args.no_transcribe,
            asr_model=args.asr_model,
        )
    except (TypeError, ValueError) as exc:
        _parser().error(str(exc))
        return 2

    result = run_pipeline(request, progress=None if args.json else _progress)
    payload = result.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for clip in result.clips:
            print(
                f"{clip.render.path}\t{clip.candidate.virality_score:.1f}\t"
                f"{clip.candidate.hook_title}"
            )
        if result.error:
            print(f"Error: {result.error}", file=sys.stderr)
        for warning in result.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        print(f"Job {result.job_id}: {result.status}", file=sys.stderr)
    return 0 if result.status in {"completed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
