"""Media intake, YouTube caption fast-path, streaming audio peaks and local ASR fallback."""

from __future__ import annotations

import array
import json
import logging
import math
import re
import statistics
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.models import AudioPeak, IngestResult, JobRequest, MediaMetadata, SourceDescriptor, Word
from core.storage import job_directory

ALLOWED_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".ts"}
)
YOUTUBE_HOSTS = ("youtube.com", "youtu.be")
TWITCH_HOSTS = ("twitch.tv",)
AUDIO_SAMPLE_RATE = 8000
AUDIO_WINDOW_SECONDS = 1.0
AUDIO_BASELINE_SECONDS = 120
MAX_RETURNED_PEAKS = 500
MAX_JSON3_BYTES = 40_000_000

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]
YdlFactory = Callable[[dict[str, Any]], AbstractContextManager[Any]]


class IngestError(RuntimeError):
    """Actionable error raised for invalid media or a failed ingest stage."""


@dataclass
class IngestServices:
    """Dependency seam for tests and host-specific executable configuration."""

    ydl_factory: YdlFactory | None = None
    urlopen: Callable[..., Any] | None = None
    run: Callable[..., Any] | None = None
    popen: Callable[..., Any] | None = None
    whisper_factory: Callable[..., Any] | None = None


def _report(progress: ProgressCallback | None, stage: str, fraction: float, message: str) -> None:
    if progress is not None:
        progress(stage, max(0.0, min(1.0, fraction)), message)


def _default_ydl_factory(options: dict[str, Any]) -> AbstractContextManager[Any]:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - exercised by setup on Windows
        raise IngestError("yt-dlp is not installed. Run setup.bat to install the runtime.") from exc
    return yt_dlp.YoutubeDL(options)


def classify_source(source: str) -> tuple[str, str | Path]:
    """Accept a supported local video or a YouTube/Twitch URL; reject other URLs."""
    raw = (source or "").strip()
    if not raw:
        raise IngestError("Choose a local video file or a YouTube/Twitch URL.")

    path = Path(raw).expanduser()
    if path.is_file():
        if path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
            raise IngestError(f"Unsupported video extension: {path.suffix or '(none)'}.")
        return "local", path.resolve()

    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        if path.exists():
            raise IngestError("The selected local path is not a regular file.")
        raise IngestError("Source must be an existing video file or an HTTPS YouTube/Twitch URL.")
    if parsed.username or parsed.password:
        raise IngestError("URLs containing embedded credentials are not accepted.")
    hostname = parsed.hostname.lower().rstrip(".")
    if any(hostname == domain or hostname.endswith("." + domain) for domain in YOUTUBE_HOSTS):
        return "youtube", raw
    if any(hostname == domain or hostname.endswith("." + domain) for domain in TWITCH_HOSTS):
        return "twitch", raw
    raise IngestError("Only YouTube and Twitch URLs are supported.")


def _parse_ratio(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            result = float(numerator) / float(denominator)
            return result if result > 0 else None
        except (ValueError, ZeroDivisionError):
            return None
    try:
        result = float(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None


def probe_media(
    path: str | Path,
    *,
    ffprobe: str = "ffprobe",
    run: Callable[..., Any] | None = None,
    timeout_seconds: float = 30,
) -> MediaMetadata:
    """Read local media metadata with ffprobe using a shell-free argument list."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise IngestError(f"Video file does not exist: {source}")
    runner = run or subprocess.run
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = runner(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise IngestError("ffprobe was not found. Install FFmpeg and add its bin folder to PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise IngestError("ffprobe timed out while reading the video.") from exc
    except OSError as exc:
        raise IngestError(f"Could not start ffprobe: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "ffprobe returned no diagnostic output").strip()
        raise IngestError(f"ffprobe could not read {source.name}: {detail[-600:]}")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise IngestError("ffprobe returned malformed JSON metadata.") from exc

    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio = any(stream.get("codec_type") == "audio" for stream in streams)
    try:
        duration = float((payload.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise IngestError("The video has no readable, positive duration.")

    return MediaMetadata(
        source_id=source.stem,
        title=source.stem or source.name,
        duration=duration,
        width=video.get("width"),
        height=video.get("height"),
        fps=_parse_ratio(video.get("r_frame_rate")),
        has_audio=audio,
    )


def choose_json3_caption_track(info: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Choose original-language word-timed captions, preferring YouTube JSON3."""
    automatic = info.get("automatic_captions") or {}
    manual = info.get("subtitles") or {}
    language = str(info.get("language") or "").strip()

    choices: list[tuple[dict[str, Any], str]] = []
    if language:
        choices.extend(((automatic.get(f"{language}-orig") or {}, f"{language}-orig"),))
        choices.extend(((automatic.get(language) or {}, language),))
        choices.extend(((manual.get(language) or {}, language),))
    original_codes = [code for code in automatic if code.endswith("-orig")]
    for code in ([f"{language}-orig"] if language else []) + original_codes:
        choices.append((automatic.get(code) or {}, code))
    if "en-orig" in automatic:
        choices.append((automatic.get("en-orig") or {}, "en-orig"))
    for code, tracks in automatic.items():
        choices.append((tracks or {}, code))
    for code, tracks in manual.items():
        choices.append((tracks or {}, code))

    seen: set[tuple[str, int]] = set()
    for tracks, code in choices:
        marker = (code, id(tracks))
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(tracks, dict):
            tracks = [tracks]
        if not isinstance(tracks, list):
            continue
        for track in tracks:
            if isinstance(track, dict) and track.get("url") and track.get("ext") == "json3":
                return code, track
    return None


def _read_json3_url(url: str, *, urlopen: Callable[..., Any] | None = None) -> dict[str, Any]:
    opener = urlopen or urllib.request.urlopen
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with opener(request, timeout=30) as response:
            raw = response.read(MAX_JSON3_BYTES + 1)
    except Exception as exc:
        raise IngestError(f"Could not fetch YouTube caption data: {exc}") from exc
    if len(raw) > MAX_JSON3_BYTES:
        raise IngestError("Caption response exceeded the 40 MB safety limit.")
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IngestError("YouTube returned invalid JSON3 caption data.") from exc
    if not isinstance(value, dict):
        raise IngestError("YouTube JSON3 caption response was not an object.")
    return value


def parse_json3_words(data: dict[str, Any], duration: float | None = None) -> list[Word]:
    """Parse JSON3 caption events into absolute word timestamps.

    YouTube original-language tracks often contain per-word offsets. When a cue
    has only one timed caption segment, its words are distributed across the cue
    duration; the caller still gets an immediate caption fast-path without ASR.
    """
    raw_words: list[tuple[float, str, float | None]] = []
    for event in data.get("events") or []:
        if not isinstance(event, dict):
            continue
        try:
            event_start = float(event.get("tStartMs") or 0) / 1000.0
            event_duration = max(0.0, float(event.get("dDurationMs") or 0) / 1000.0)
        except (TypeError, ValueError):
            continue
        segments = [
            segment
            for segment in (event.get("segs") or [])
            if isinstance(segment, dict) and isinstance(segment.get("utf8"), str)
        ]
        if not segments:
            continue
        timed = any("tOffsetMs" in segment for segment in segments)
        flattened: list[tuple[float, str, float | None]] = []
        if not timed:
            tokens = [token for segment in segments for token in segment["utf8"].split()]
            for token_index, token in enumerate(tokens):
                word_start = event_start + event_duration * token_index / max(1, len(tokens))
                estimated_end = event_start + event_duration * (token_index + 1) / max(1, len(tokens))
                flattened.append((word_start, token, estimated_end))
        else:
            for segment_index, segment in enumerate(segments):
                tokens = segment["utf8"].split()
                if not tokens:
                    continue
                try:
                    offset = float(segment.get("tOffsetMs", 0)) / 1000.0
                except (TypeError, ValueError):
                    offset = 0.0
                next_offset: float | None = None
                for next_segment in segments[segment_index + 1 :]:
                    if "tOffsetMs" in next_segment:
                        try:
                            next_offset = float(next_segment["tOffsetMs"]) / 1000.0
                        except (TypeError, ValueError):
                            pass
                        break
                segment_span = max(0.0, (next_offset if next_offset is not None else event_duration) - offset)
                for token_index, token in enumerate(tokens):
                    word_start = event_start + offset + segment_span * token_index / max(1, len(tokens))
                    estimated_end = event_start + offset + segment_span * (token_index + 1) / max(1, len(tokens))
                    flattened.append((word_start, token, estimated_end))
        raw_words.extend(flattened)

    raw_words.sort(key=lambda item: item[0])
    output: list[Word] = []
    skipping_annotation = False
    for index, (start, text, estimated_end) in enumerate(raw_words):
        if duration is not None and start >= duration:
            continue
        if text[:1] in "[(" and not re.fullmatch(r"[\[(].*[\])\]]", text):
            skipping_annotation = True
            continue
        if skipping_annotation:
            if text.endswith(("]", ")")):
                skipping_annotation = False
            continue
        if re.fullmatch(r"[\[(].*[\])\]]", text) or text in {"[", "]", "__", ">>"}:
            continue
        if not any(character.isalnum() for character in text):
            continue

        next_start = raw_words[index + 1][0] if index + 1 < len(raw_words) else None
        candidate_end = estimated_end if estimated_end and estimated_end > start else None
        if next_start is not None and next_start > start:
            candidate_end = min(candidate_end, next_start) if candidate_end else next_start
        if not candidate_end or candidate_end <= start:
            candidate_end = start + 0.4
        end = min(candidate_end, start + 0.8)
        if duration is not None:
            end = min(end, duration)
        if end <= start:
            continue
        try:
            output.append(Word(start=round(start, 3), end=round(end, 3), text=text))
        except ValueError:
            continue
    return output


def _calculate_rms_dbfs(raw: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return -120.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    rms = math.sqrt(mean_square) / 32768.0
    return max(-120.0, min(0.0, 20.0 * math.log10(max(rms, 1e-6))))


def peaks_from_loudness(
    loudness_windows: Iterable[float],
    *,
    window_seconds: float = AUDIO_WINDOW_SECONDS,
    baseline_seconds: int = AUDIO_BASELINE_SECONDS,
    minimum_jump_db: float = 4.0,
    min_gap_seconds: float = 2.0,
    max_peaks: int = MAX_RETURNED_PEAKS,
) -> list[AudioPeak]:
    """Turn per-window dBFS values into salient jumps over a rolling median baseline."""
    values = list(loudness_windows)
    if not values:
        return []
    history: deque[float] = deque(maxlen=max(1, int(baseline_seconds / window_seconds)))
    scored: list[tuple[int, float, float, float]] = []
    for index, loudness in enumerate(values):
        current = max(-120.0, min(0.0, float(loudness)))
        baseline = statistics.median(history) if history else current
        jump = current - baseline
        score = max(0.0, min(1.0, (jump - minimum_jump_db) / 15.0))
        if current > -48.0:
            scored.append((index, score, current, baseline))
        history.append(current)

    selected: list[AudioPeak] = []
    last_time = -float("inf")
    for position, (index, score, loudness, baseline) in enumerate(scored):
        if score <= 0:
            continue
        previous_score = scored[position - 1][1] if position else -1.0
        next_score = scored[position + 1][1] if position + 1 < len(scored) else -1.0
        if score < previous_score or score < next_score:
            continue
        timestamp = index * window_seconds
        if timestamp - last_time < min_gap_seconds:
            if selected and score > selected[-1].score:
                selected[-1] = AudioPeak(
                    time=round(timestamp, 3),
                    score=round(score, 4),
                    loudness_dbfs=round(loudness, 2),
                    baseline_dbfs=round(baseline, 2),
                )
                last_time = timestamp
            continue
        selected.append(
            AudioPeak(
                time=round(timestamp, 3),
                score=round(score, 4),
                loudness_dbfs=round(loudness, 2),
                baseline_dbfs=round(baseline, 2),
            )
        )
        last_time = timestamp
    if len(selected) > max_peaks:
        selected = sorted(selected, key=lambda peak: peak.score, reverse=True)[:max_peaks]
        selected.sort(key=lambda peak: peak.time)
    return selected


def analyze_audio_peaks(
    audio_path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    popen: Callable[..., Any] | None = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    window_seconds: float = AUDIO_WINDOW_SECONDS,
) -> list[AudioPeak]:
    """Stream mono PCM through FFmpeg and compute bounded-memory loudness peaks."""
    source = Path(audio_path).expanduser().resolve()
    if not source.is_file():
        raise IngestError(f"Audio analysis source does not exist: {source}")
    process_factory = popen or subprocess.Popen
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        process = process_factory(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise IngestError("FFmpeg was not found. Install FFmpeg and add its bin folder to PATH.") from exc
    except OSError as exc:
        raise IngestError(f"Could not start FFmpeg audio analysis: {exc}") from exc

    if process.stdout is None:
        raise IngestError("FFmpeg did not provide a PCM output stream.")
    window_bytes = max(2, int(sample_rate * window_seconds) * 2)
    pending = bytearray()
    loudness: list[float] = []
    try:
        while True:
            block = process.stdout.read(64 * 1024)
            if not block:
                break
            pending.extend(block)
            while len(pending) >= window_bytes:
                raw = bytes(pending[:window_bytes])
                del pending[:window_bytes]
                loudness.append(_calculate_rms_dbfs(raw))
        if pending:
            loudness.append(_calculate_rms_dbfs(bytes(pending)))
        return_code = process.wait()
    except Exception as exc:
        try:
            process.kill()
        except OSError as cleanup_error:
            logger.debug("Could not stop FFmpeg after an audio-read failure: %s", cleanup_error)
        raise IngestError(f"FFmpeg audio analysis failed: {exc}") from exc
    finally:
        try:
            process.stdout.close()
        except (OSError, ValueError) as cleanup_error:
            logger.debug("Could not close FFmpeg PCM pipe: %s", cleanup_error)
    if return_code != 0:
        raise IngestError(f"FFmpeg audio analysis exited with code {return_code}.")
    return peaks_from_loudness(loudness, window_seconds=window_seconds)


def _extract_info(url: str, factory: YdlFactory) -> dict[str, Any]:
    options = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 20,
        "retries": 2,
        "extract_flat": False,
    }
    try:
        with factory(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise IngestError(f"yt-dlp could not inspect the remote video: {exc}") from exc
    if not isinstance(info, dict) or info.get("_type") == "playlist":
        raise IngestError("The URL did not resolve to one video. Playlist URLs are not supported.")
    return info


def _remote_metadata(info: dict[str, Any], source_id: str) -> MediaMetadata:
    try:
        duration = float(info.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise IngestError("The remote source did not provide a valid video duration.")
    try:
        width = int(info["width"]) if info.get("width") else None
        height = int(info["height"]) if info.get("height") else None
    except (TypeError, ValueError):
        width, height = None, None
    return MediaMetadata(
        source_id=str(info.get("id") or source_id),
        title=str(info.get("title") or "Untitled video")[:300],
        duration=duration,
        width=width,
        height=height,
        fps=_parse_ratio(info.get("fps")),
        has_audio=True,
        language=info.get("language"),
        thumbnail=info.get("thumbnail"),
    )


def download_audio_only(
    url: str,
    work_dir: str | Path,
    *,
    ydl_factory: YdlFactory | None = None,
) -> Path:
    """Download only the best available audio track for analysis/transcription."""
    factory = ydl_factory or _default_ydl_factory
    directory = Path(work_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / "source-audio.%(ext)s"
    options = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(prefix),
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
    }
    try:
        with factory(options) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise IngestError(f"Could not download the remote audio track: {exc}") from exc
    matches = [
        path
        for path in directory.glob("source-audio.*")
        if path.is_file() and not path.name.endswith((".part", ".ytdl"))
    ]
    if not matches:
        raise IngestError("yt-dlp completed without producing an audio file.")
    return max(matches, key=lambda path: path.stat().st_size)


def transcribe_local_audio(
    audio_path: str | Path,
    *,
    model_name: str = "small",
    language: str | None = None,
    model_factory: Callable[..., Any] | None = None,
    download_root: str | Path | None = None,
) -> list[Word]:
    """Run faster-whisper on CPU with int8 weights and word timestamps."""
    factory = model_factory
    if factory is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised by setup on Windows
            raise IngestError("faster-whisper is not installed. Run setup.bat to install the runtime.") from exc
        factory = WhisperModel
    try:
        model = factory(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(download_root) if download_root else None,
        )
        segments, _info = model.transcribe(
            str(Path(audio_path).expanduser().resolve()),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        words: list[Word] = []
        for segment in segments:
            segment_words = getattr(segment, "words", None) or []
            for item in segment_words:
                text = str(getattr(item, "word", "") or "").strip()
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)
                if not text or start is None or end is None:
                    continue
                try:
                    words.append(Word(start=float(start), end=float(end), text=text))
                except (TypeError, ValueError):
                    continue
            if not segment_words:
                text = str(getattr(segment, "text", "") or "").strip()
                start = float(getattr(segment, "start", 0.0) or 0.0)
                end = float(getattr(segment, "end", start) or start)
                tokens = text.split()
                span = max(0.0, end - start)
                for index, token in enumerate(tokens):
                    try:
                        words.append(
                            Word(
                                start=start + span * index / max(1, len(tokens)),
                                end=start + span * (index + 1) / max(1, len(tokens)),
                                text=token,
                            )
                        )
                    except ValueError:
                        continue
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"Local faster-whisper transcription failed: {exc}") from exc
    words.sort(key=lambda word: word.start)
    return words


def default_job_workdir(request: JobRequest) -> Path:
    if request.work_dir is not None:
        return request.work_dir.expanduser().resolve()
    return job_directory(request.job_id)


def ingest_source(
    request: JobRequest,
    *,
    progress: ProgressCallback | None = None,
    services: IngestServices | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> IngestResult:
    """Ingest a URL or local video, preferring word-timed YouTube captions over ASR."""
    dependencies = services or IngestServices()
    work_dir = default_job_workdir(request)
    work_dir.mkdir(parents=True, exist_ok=True)
    kind, value = classify_source(request.source)
    _report(progress, "INGEST", 0.02, "Checking source and reading metadata")

    if kind == "local":
        local_path = Path(value)
        metadata = probe_media(local_path, ffprobe=ffprobe, run=dependencies.run)
        source = SourceDescriptor(kind="local", source_id=metadata.source_id, local_path=local_path)
        audio_path = local_path
        caption_words: list[Word] = []
    else:
        url = str(value)
        factory = dependencies.ydl_factory or _default_ydl_factory
        info = _extract_info(url, factory)
        metadata = _remote_metadata(info, str(info.get("id") or "remote-video"))
        source = SourceDescriptor(kind=kind, source_id=metadata.source_id, url=url)
        caption_words = []
        chosen_track = choose_json3_caption_track(info)
        if chosen_track is not None:
            _report(progress, "CAPTIONS", 0.08, f"Reading {chosen_track[0]} word-timed YouTube captions")
            try:
                caption_data = _read_json3_url(chosen_track[1]["url"], urlopen=dependencies.urlopen)
                caption_words = parse_json3_words(caption_data, metadata.duration)
            except IngestError:
                caption_words = []
        audio_path = None
        try:
            _report(progress, "AUDIO", 0.12, "Downloading the audio track only")
            audio_path = download_audio_only(url, work_dir, ydl_factory=factory)
            source.audio_path = audio_path
        except IngestError as exc:
            if not caption_words:
                raise
            # Captions still permit transcript-based selection when the source
            # disallows audio-only delivery; continue with empty audio features.
            warnings = [str(exc)]
        else:
            warnings = []

    if kind == "local":
        warnings = []

    peaks: list[AudioPeak] = []
    if audio_path is not None and metadata.has_audio:
        _report(progress, "AUDIO", 0.2, "Measuring loudness changes in bounded-memory windows")
        peaks = analyze_audio_peaks(
            audio_path,
            ffmpeg=ffmpeg,
            popen=dependencies.popen,
        )

    transcript_source: Literal["youtube_captions", "faster_whisper", "none"] = "none"
    if caption_words:
        transcript_source = "youtube_captions"
        words = caption_words
    elif request.transcribe_missing and metadata.has_audio and audio_path is not None:
        _report(progress, "TRANSCRIBE", 0.35, "Transcribing locally with faster-whisper CPU/int8")
        words = transcribe_local_audio(
            audio_path,
            model_name=request.asr_model,
            language=request.language or metadata.language,
            model_factory=dependencies.whisper_factory,
            download_root=work_dir / "models",
        )
        transcript_source = "faster_whisper" if words else "none"
    else:
        words = []
        if not metadata.has_audio:
            warnings.append("Source has no audio track; transcription and audio peaks were skipped.")
        elif not request.transcribe_missing:
            warnings.append("No usable word-timed captions; local transcription was disabled.")

    if not words and request.transcribe_missing and metadata.has_audio and audio_path is None:
        warnings.append("No audio file was available for local transcription.")
    _report(progress, "INGEST", 0.55, f"Ingested {len(words)} words and {len(peaks)} audio peaks")
    return IngestResult(
        metadata=metadata,
        source=source,
        words=words,
        audio_peaks=peaks,
        transcript_source=transcript_source,
        warnings=warnings,
    )
