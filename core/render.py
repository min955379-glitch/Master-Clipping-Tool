"""Partial source preparation and normalized FFmpeg rendering with encoder fallback."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.captions import write_ass
from core.encoder import EncoderProbeError, EncoderSelection, detect_encoder
from core.ingest import IngestError, classify_source
from core.models import (
    ClipCandidate,
    IngestResult,
    JobRequest,
    PreparedClipSource,
    ReframePlan,
    RenderResult,
    SourceDescriptor,
    Word,
)

logger = logging.getLogger(__name__)


class RenderError(RuntimeError):
    """Actionable error raised by source preparation or FFmpeg rendering."""


@dataclass
class RenderServices:
    """Injection points for FFmpeg, encoder probing and yt-dlp in tests/hosts."""

    run: Callable[..., Any] | None = None
    popen: Callable[..., Any] | None = None
    encoder_probe: Callable[..., EncoderSelection] | None = None
    ydl_factory: Callable[[dict[str, Any]], AbstractContextManager[Any]] | None = None
    range_factory: Callable[[float, float], Any] | None = None


def _default_ydl_factory(options: dict[str, Any]) -> AbstractContextManager[Any]:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - exercised on the target setup
        raise RenderError("yt-dlp is not installed. Run setup.bat to install the runtime.") from exc
    return yt_dlp.YoutubeDL(options)


def _download_range(start: float, end: float) -> Any:
    try:
        from yt_dlp.utils import download_range_func
    except ImportError as exc:  # pragma: no cover - exercised on the target setup
        raise RenderError("This yt-dlp build does not support partial range downloads.") from exc
    return download_range_func(None, [(start, end)])


def _source_descriptor(source: SourceDescriptor | str | Path) -> SourceDescriptor:
    if isinstance(source, SourceDescriptor):
        return source
    raw = str(source)
    try:
        kind, value = classify_source(raw)
    except IngestError as exc:
        raise RenderError(str(exc)) from exc
    if kind == "local":
        path = Path(value).expanduser().resolve()
        return SourceDescriptor(kind="local", source_id=path.stem, local_path=path)
    return SourceDescriptor(kind=kind, source_id="remote", url=str(value))


def _downloaded_files(directory: Path, prefix: str) -> list[Path]:
    return [
        path
        for path in directory.glob(prefix + "*")
        if path.is_file() and not path.name.endswith((".part", ".ytdl", ".temp")) and path.stat().st_size > 0
    ]


def _remove_download_artifacts(directory: Path, prefix: str) -> None:
    for path in directory.glob(prefix + "*"):
        if not path.is_file():
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Could not remove incomplete download %s: %s", path, exc)


def _download_remote(
    url: str,
    destination: Path,
    *,
    clip_start: float,
    clip_end: float,
    ydl_factory: Callable[[dict[str, Any]], AbstractContextManager[Any]],
    range_factory: Callable[[float, float], Any],
) -> PreparedClipSource:
    destination.mkdir(parents=True, exist_ok=True)
    duration = clip_end - clip_start
    prefix = f"source-{clip_start:.3f}-{clip_end:.3f}".replace(".", "p")
    _remove_download_artifacts(destination, prefix)
    template = str(destination / (prefix + ".%(ext)s"))
    warnings: list[str] = []
    base_options: dict[str, Any] = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "format": "bv*+ba/best",
        "outtmpl": template,
        "merge_output_format": "mkv",
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
    }

    try:
        range_callback = range_factory(clip_start, clip_end)
        partial_options = {
            **base_options,
            "download_ranges": range_callback,
            "force_keyframes_at_cuts": True,
        }
        with ydl_factory(partial_options) as ydl:
            ydl.download([url])
        files = _downloaded_files(destination, prefix)
        if not files:
            raise RenderError("yt-dlp returned without creating a ranged video file.")
        path = max(files, key=lambda item: item.stat().st_size)
        return PreparedClipSource(
            path=path,
            trim_start=0.0,
            duration=duration,
            source_start=clip_start,
            partial=True,
        )
    except Exception as exc:  # noqa: BLE001 - yt-dlp extractor errors vary across providers/builds.
        warnings.append(f"Partial-range download unavailable; downloading the full source instead. {exc}")
        _remove_download_artifacts(destination, prefix)

    full_options = {**base_options}
    full_options.pop("download_ranges", None)
    full_options.pop("force_keyframes_at_cuts", None)
    full_prefix = prefix + "-full"
    _remove_download_artifacts(destination, full_prefix)
    full_options["outtmpl"] = str(destination / (full_prefix + ".%(ext)s"))
    try:
        with ydl_factory(full_options) as ydl:
            ydl.download([url])
    except Exception as exc:
        _remove_download_artifacts(destination, full_prefix)
        raise RenderError(f"Could not download the remote video: {exc}") from exc
    files = _downloaded_files(destination, full_prefix)
    if not files:
        raise RenderError("yt-dlp completed without creating a full video file.")
    path = max(files, key=lambda item: item.stat().st_size)
    return PreparedClipSource(
        path=path,
        trim_start=clip_start,
        duration=duration,
        source_start=0.0,
        partial=False,
        warnings=warnings,
    )


def prepare_clip_source(
    source: SourceDescriptor | str | Path,
    candidate: ClipCandidate,
    work_dir: str | Path,
    *,
    services: RenderServices | None = None,
) -> PreparedClipSource:
    """For URLs try yt-dlp range download, then fall back to a full download."""
    dependencies = services or RenderServices()
    descriptor = _source_descriptor(source)
    destination = Path(work_dir).expanduser().resolve() / "sources"
    if descriptor.kind == "local":
        path = descriptor.local_path
        if path is None or not path.is_file():
            raise RenderError(f"Local source file does not exist: {path}")
        return PreparedClipSource(
            path=path,
            trim_start=candidate.start,
            duration=candidate.end - candidate.start,
            source_start=0.0,
            partial=False,
        )
    if not descriptor.url:
        raise RenderError("Remote source descriptor has no URL.")
    factory = dependencies.ydl_factory or _default_ydl_factory
    range_factory = dependencies.range_factory or _download_range
    return _download_remote(
        descriptor.url,
        destination,
        clip_start=candidate.start,
        clip_end=candidate.end,
        ydl_factory=factory,
        range_factory=range_factory,
    )


def _ffmpeg_filter_path(path: Path) -> str:
    """Escape an absolute subtitle filename for FFmpeg's filtergraph parser."""
    return (
        path.resolve().as_posix()
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(";", "\\;")
    )


def _encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_amf":
        return ["-quality", "balanced", "-rc", "vbr_peak", "-b:v", "8M", "-maxrate", "12M"]
    if encoder == "h264_nvenc":
        return ["-preset", "p5", "-cq", "23", "-b:v", "0"]
    if encoder == "libx264":
        return ["-preset", "medium", "-crf", "20"]
    return []


def _ffmpeg_command(
    *,
    ffmpeg: str,
    prepared: PreparedClipSource,
    output_path: Path,
    plan: ReframePlan,
    ass_path: Path,
    encoder: str,
    has_audio: bool,
    include_progress: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    escaped_ass = _ffmpeg_filter_path(ass_path)
    graph = f"{plan.filter_complex};[vout]subtitles=filename='{escaped_ass}'[captioned]"
    command = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if prepared.trim_start > 0:
        command.extend(["-ss", f"{prepared.trim_start:.3f}"])
    command.extend(
        [
            "-i",
            str(prepared.path),
            "-t",
            f"{prepared.duration:.3f}",
            "-filter_complex",
            graph,
            "-map",
            "[captioned]",
        ]
    )
    if has_audio:
        command.extend(
            [
                "-map",
                "0:a:0?",
                "-af",
                "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=summary",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )
    command.extend(
        [
            "-c:v",
            encoder,
            *_encoder_args(encoder),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-map_metadata",
            "-1",
        ]
    )
    if include_progress:
        command.extend(["-progress", "pipe:1", "-nostats"])
    if extra_args:
        command.extend(extra_args)
    command.append(str(output_path))
    return command


def _run_ffmpeg(command: list[str], runner: Callable[..., Any], timeout: float) -> Any:
    try:
        return runner(command, capture_output=True, text=True, check=False, timeout=timeout)
    except FileNotFoundError as exc:
        raise RenderError("FFmpeg was not found. Install FFmpeg and add its bin folder to PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"FFmpeg render exceeded the {timeout:g}-second timeout.") from exc
    except OSError as exc:
        raise RenderError(f"Could not start FFmpeg: {exc}") from exc


def _run_ffmpeg_streaming(
    command: list[str],
    popen: Callable[..., Any],
    *,
    timeout: float,
    duration: float,
    progress: Callable[[str, float, str], None],
) -> Any:
    """Read FFmpeg's `-progress pipe:1` records without blocking on stderr."""
    try:
        process = popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RenderError("FFmpeg was not found. Install FFmpeg and add its bin folder to PATH.") from exc
    except OSError as exc:
        raise RenderError(f"Could not start FFmpeg: {exc}") from exc
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RenderError("FFmpeg did not expose progress and diagnostic output streams.")

    messages: queue.Queue[str | None] = queue.Queue()
    errors: deque[str] = deque(maxlen=200)

    def read_progress() -> None:
        try:
            for line in process.stdout:
                messages.put(line.rstrip("\r\n"))
        finally:
            messages.put(None)

    def read_errors() -> None:
        for line in process.stderr:
            errors.append(line)

    stdout_thread = threading.Thread(target=read_progress, name="ffmpeg-progress", daemon=True)
    stderr_thread = threading.Thread(target=read_errors, name="ffmpeg-diagnostics", daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    started = time.monotonic()
    stdout_done = False
    timed_out = False
    while not stdout_done or process.poll() is None:
        if time.monotonic() - started >= timeout:
            timed_out = True
            try:
                process.kill()
            except OSError as exc:
                logger.debug("Could not stop timed-out FFmpeg process: %s", exc)
            break
        try:
            line = messages.get(timeout=0.2)
        except queue.Empty:
            continue
        if line is None:
            stdout_done = True
            continue
        key, separator, raw_value = line.partition("=")
        if not separator or key not in {"out_time_us", "out_time_ms"}:
            continue
        try:
            elapsed = max(0.0, float(raw_value) / 1_000_000)
        except ValueError:
            continue
        fraction = min(0.97, elapsed / max(0.001, duration))
        try:
            progress("RENDER", fraction, f"Encoding with FFmpeg · {round(fraction * 100)}%")
        except Exception as exc:  # noqa: BLE001 - telemetry must not interrupt an active render.
            logger.warning("Render progress callback failed: %s", exc)

    try:
        return_code = process.wait(timeout=5 if timed_out else max(1, timeout - (time.monotonic() - started)))
    except subprocess.TimeoutExpired as exc:
        try:
            process.kill()
        except OSError as cleanup_error:
            logger.debug("Could not stop a stuck FFmpeg process: %s", cleanup_error)
        raise RenderError(f"FFmpeg render exceeded the {timeout:g}-second timeout.") from exc
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    if timed_out:
        raise RenderError(f"FFmpeg render exceeded the {timeout:g}-second timeout.")
    return subprocess.CompletedProcess(command, return_code, stdout="", stderr="".join(errors)[-2000:])


def render_clip(
    source: SourceDescriptor | str | Path,
    candidate: ClipCandidate,
    plan: ReframePlan,
    words: list[Word],
    output_path: str | Path,
    *,
    work_dir: str | Path,
    caption_preset: str = "Hormozi Bold",
    suggested_caption: str = "",
    has_audio: bool = True,
    ffmpeg: str = "ffmpeg",
    timeout_seconds: float = 1800,
    services: RenderServices | None = None,
    progress: Callable[[str, float, str], None] | None = None,
    prepared_source: PreparedClipSource | None = None,
) -> RenderResult:
    """Render one selected clip, burn captions, normalize audio and retry on CPU."""
    dependencies = services or RenderServices()
    runner = dependencies.run or subprocess.run
    popen = dependencies.popen or subprocess.Popen
    probe = dependencies.encoder_probe or detect_encoder
    stream_progress = dependencies.run is None and progress is not None
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    job_directory = Path(work_dir).expanduser().resolve()
    job_directory.mkdir(parents=True, exist_ok=True)
    if candidate.start < 0 or candidate.end <= candidate.start:
        raise RenderError("Clip time range must be positive and ordered.")
    if progress:
        progress("PREPARE", 0.0, "Preparing only the selected source time range")
    prepared = prepared_source or prepare_clip_source(source, candidate, job_directory, services=dependencies)
    if abs(prepared.duration - (candidate.end - candidate.start)) > 0.05:
        raise RenderError("Prepared source duration does not match the selected clip range.")
    ass_path = job_directory / "captions" / f"{destination.stem}.ass"
    write_ass(
        ass_path,
        words,
        clip_start=candidate.start,
        clip_end=candidate.end,
        preset=caption_preset,
        output_width=plan.output_width,
        output_height=plan.output_height,
        fallback_text=suggested_caption,
    )
    try:
        selection = probe(ffmpeg)
    except (EncoderProbeError, OSError) as exc:
        raise RenderError(f"Could not select an H.264 encoder: {exc}") from exc
    if progress:
        progress("RENDER", 0.1, f"Encoding with {selection.name}")

    command = _ffmpeg_command(
        ffmpeg=ffmpeg,
        prepared=prepared,
        output_path=destination,
        plan=plan,
        ass_path=ass_path,
        encoder=selection.name,
        has_audio=has_audio,
        include_progress=stream_progress,
    )
    if stream_progress:
        result = _run_ffmpeg_streaming(
            command,
            popen,
            timeout=timeout_seconds,
            duration=prepared.duration,
            progress=progress,
        )
    else:
        result = _run_ffmpeg(command, runner, timeout_seconds)
    chosen_encoder = selection.name
    warnings = list(prepared.warnings)
    if result.returncode != 0 and selection.hardware_accelerated:
        error_detail = (result.stderr or result.stdout or "unknown hardware encoder error").strip()
        if "libx264" not in selection.available:
            raise RenderError(
                f"{selection.name} failed and this FFmpeg build did not report libx264 for CPU fallback: "
                f"{error_detail[-700:]}"
            )
        warnings.append(f"{selection.name} encoding failed; retried with CPU libx264.")
        if progress:
            progress("RENDER", 0.2, "Hardware encoder failed; retrying with CPU libx264")
        command = _ffmpeg_command(
            ffmpeg=ffmpeg,
            prepared=prepared,
            output_path=destination,
            plan=plan,
            ass_path=ass_path,
            encoder="libx264",
            has_audio=has_audio,
            include_progress=stream_progress,
        )
        if stream_progress:
            result = _run_ffmpeg_streaming(
                command,
                popen,
                timeout=timeout_seconds,
                duration=prepared.duration,
                progress=progress,
            )
        else:
            result = _run_ffmpeg(command, runner, timeout_seconds)
        chosen_encoder = "libx264"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "FFmpeg returned no diagnostic output").strip()
        raise RenderError(f"FFmpeg render failed (exit {result.returncode}): {detail[-1000:]}")
    if not destination.is_file():
        raise RenderError("FFmpeg reported success but did not create the output file.")
    if progress:
        progress("RENDER", 1.0, f"Rendered {destination.name}")
    return RenderResult(
        path=destination,
        encoder=chosen_encoder,
        duration=candidate.end - candidate.start,
        warnings=warnings,
    )


def render_job_clip(
    ingest: IngestResult,
    candidate: ClipCandidate,
    plan: ReframePlan,
    output_path: str | Path,
    *,
    work_dir: str | Path,
    request: JobRequest,
    services: RenderServices | None = None,
    progress: Callable[[str, float, str], None] | None = None,
) -> RenderResult:
    """Convenience adapter from pipeline models to :func:`render_clip`."""
    return render_clip(
        ingest.source,
        candidate,
        plan,
        ingest.words,
        output_path,
        work_dir=work_dir,
        caption_preset=request.caption_style.value,
        suggested_caption=candidate.suggested_caption,
        has_audio=ingest.metadata.has_audio,
        services=services,
        progress=progress,
    )
