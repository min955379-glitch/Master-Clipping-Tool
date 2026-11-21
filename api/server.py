"""FastAPI adapter for the shared local-first clipping pipeline."""

from __future__ import annotations

import os
import re
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, BinaryIO, Literal
from urllib.parse import quote

import anyio
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from core.encoder import EncoderProbeError, detect_encoder
from core.ingest import ALLOWED_VIDEO_SUFFIXES
from core.job_store import JobStore, StoredJob
from core.models import (
    CaptionPreset,
    ExportPreset,
    JobRequest,
    PipelineProgress,
    PipelineResult,
    ProviderName,
    ReframeMode,
)
from core.pipeline import run_pipeline

APP_VERSION = "0.3.0"
MAX_UPLOAD_BYTES = int(os.environ.get("FRAMEFORGE_MAX_UPLOAD_BYTES", str(16 * 1024**3)))
store = JobStore()
_job_semaphore = threading.Semaphore(1)


class ApiClip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float
    end: float
    virality_score: float
    hook_title: str
    reason: str
    suggested_caption: str
    encoder: str
    file_name: str
    download_url: str


class ApiJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    status: Literal["queued", "running", "completed", "partial", "failed"]
    progress: float = Field(ge=0, le=1)
    stage: str
    message: str
    source_label: str
    created_at: datetime
    updated_at: datetime
    clips: list[ApiClip] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


app = FastAPI(
    title="All-in-One AI Video Clipping Studio",
    version=APP_VERSION,
    description="Local-first video clipping studio API.",
)


def _api_job(record: StoredJob) -> ApiJob:
    clip_items: list[ApiClip] = []
    warnings: list[str] = []
    if record.result is not None:
        warnings.extend(record.result.warnings)
        for clip in record.result.clips:
            file_name = clip.render.path.name
            clip_items.append(
                ApiClip(
                    start=clip.candidate.start,
                    end=clip.candidate.end,
                    virality_score=clip.candidate.virality_score,
                    hook_title=clip.candidate.hook_title,
                    reason=clip.candidate.reason,
                    suggested_caption=clip.candidate.suggested_caption,
                    encoder=clip.render.encoder,
                    file_name=file_name,
                    download_url=f"/api/jobs/{record.job_id}/files/{quote(file_name)}",
                )
            )
    return ApiJob(
        job_id=record.job_id,
        status=record.status,
        progress=record.progress,
        stage=record.stage,
        message=record.message,
        source_label=record.source_label,
        created_at=record.created_at,
        updated_at=record.updated_at,
        clips=clip_items,
        warnings=warnings,
        error=record.error,
    )


def _source_label(source: str) -> str:
    raw = source.strip()
    if raw.startswith(("http://", "https://")):
        return "Remote video source"
    return Path(raw).name[:180] or "Local video"


class UploadTooLargeError(ValueError):
    """Raised when a local upload exceeds the configured per-job limit."""


def _copy_upload(source: BinaryIO, destination: Path) -> int:
    source.seek(0)
    total_bytes = 0
    with destination.open("wb") as target:
        while chunk := source.read(8 * 1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise UploadTooLargeError(
                    f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024**3)} GB local upload limit."
                )
            target.write(chunk)
    return total_bytes


def _progress_for_job(job_id: str, event: PipelineProgress) -> None:
    try:
        store.update(
            job_id,
            status="running",
            progress=event.progress,
            stage=event.stage,
            message=event.message[:500],
        )
    except (KeyError, OSError, ValueError):
        # A progress snapshot is best-effort; failure to update it must not kill FFmpeg.
        return


def _run_job(job_id: str, request: JobRequest) -> None:
    with _job_semaphore:
        try:
            store.update(job_id, status="running", stage="STARTING", message="Starting local pipeline")
            result: PipelineResult = run_pipeline(
                request,
                progress=lambda event: _progress_for_job(job_id, event),
            )
            store.finish(job_id, result)
        except Exception as exc:  # noqa: BLE001 - persist any unexpected pipeline failure for the UI.
            store.update(
                job_id,
                status="failed",
                progress=1.0,
                stage="FAILED",
                message=str(exc)[:500],
                error=str(exc)[:2000],
            )


def _queue_job(background_tasks: BackgroundTasks, request: JobRequest, source_label: str) -> ApiJob:
    job_id = uuid.uuid4().hex
    work_dir = store.work_dir(job_id)
    effective = request.model_copy(
        update={
            "job_id": job_id,
            "work_dir": work_dir,
            "output_dir": work_dir / "exports",
        }
    )
    record = store.create(job_id, source_label=source_label)
    background_tasks.add_task(_run_job, job_id, effective)
    return _api_job(record)


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness endpoint that does not launch FFmpeg or an AI model."""
    return {"status": "ok", "version": APP_VERSION, "pipeline": "ready"}


@app.get("/api/runtime", tags=["system"])
def runtime_status() -> dict[str, object]:
    """Report local optional-provider and encoder availability without failing health."""
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        encoder = detect_encoder(os.environ.get("FFMPEG_PATH", "ffmpeg"))
        encoder_info: dict[str, object] = {
            "available": True,
            "selected": encoder.name,
            "hardware_accelerated": encoder.hardware_accelerated,
            "supported": list(encoder.available),
        }
    except EncoderProbeError as exc:
        encoder_info = {"available": False, "error": str(exc)}
    return {
        "python": sys.version.split()[0],
        "ollama_url": ollama_url,
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "encoder": encoder_info,
    }


@app.get("/api/jobs", response_model=list[ApiJob], tags=["jobs"])
def list_jobs(limit: int = Query(default=50, ge=1, le=100)) -> list[ApiJob]:
    """Return the newest local job snapshots."""
    return [_api_job(record) for record in store.list(limit)]


@app.post("/api/jobs", response_model=ApiJob, status_code=202, tags=["jobs"])
def submit_job(request: JobRequest, background_tasks: BackgroundTasks) -> ApiJob:
    """Queue a remote URL or an existing local path for processing."""
    if not request.source.strip():
        raise HTTPException(status_code=422, detail="A source URL or local video path is required.")
    return _queue_job(background_tasks, request, _source_label(request.source))


@app.post("/api/jobs/upload", response_model=ApiJob, status_code=202, tags=["jobs"])
async def upload_job(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    provider: Annotated[ProviderName, Form()] = ProviderName.OLLAMA,
    model: Annotated[str | None, Form()] = None,
    base_url: Annotated[str, Form()] = "http://localhost:11434",
    custom_direction: Annotated[str, Form()] = "",
    language: Annotated[str | None, Form()] = None,
    layout: Annotated[ReframeMode, Form()] = ReframeMode.FACE_TRACK,
    caption_style: Annotated[CaptionPreset, Form()] = CaptionPreset.HORMOZI_BOLD,
    export_preset: Annotated[ExportPreset, Form()] = ExportPreset.SHORT_VERTICAL,
    clip_count: Annotated[int, Form(ge=1, le=12)] = 4,
    min_clip_seconds: Annotated[float, Form(ge=5, le=175)] = 15,
    max_clip_seconds: Annotated[float, Form(ge=10, le=180)] = 60,
    transcribe_missing: Annotated[bool, Form()] = True,
    asr_model: Annotated[str, Form()] = "small",
) -> ApiJob:
    """Stream an uploaded local video into the job folder, then use the same pipeline."""
    original_name = Path(file.filename or "video.mp4").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        await file.close()
        raise HTTPException(status_code=415, detail=f"Unsupported video extension: {suffix or '(none)'}")
    job_id = uuid.uuid4().hex
    work_dir = store.work_dir(job_id)
    request = JobRequest(
        source=str(work_dir / "uploads" / f"source{suffix}"),
        job_id=job_id,
        work_dir=work_dir,
        output_dir=work_dir / "exports",
        provider=provider,
        model=model,
        base_url=base_url,
        custom_direction=custom_direction,
        language=language,
        layout=layout,
        caption_style=caption_style,
        export_preset=export_preset,
        clip_count=clip_count,
        min_clip_seconds=min_clip_seconds,
        max_clip_seconds=max_clip_seconds,
        transcribe_missing=transcribe_missing,
        asr_model=asr_model,
    )
    record = store.create(job_id, source_label=original_name[:180])
    upload_path = Path(request.source)
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        total_bytes = await anyio.to_thread.run_sync(_copy_upload, file.file, upload_path)
    except UploadTooLargeError as exc:
        upload_path.unlink(missing_ok=True)
        store.update(job_id, status="failed", stage="FAILED", progress=1, message=str(exc), error=str(exc))
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except OSError as exc:
        upload_path.unlink(missing_ok=True)
        store.update(job_id, status="failed", stage="FAILED", progress=1, message="Could not save upload", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Could not save uploaded video: {exc}") from exc
    finally:
        await file.close()
    if total_bytes == 0:
        store.update(job_id, status="failed", stage="FAILED", progress=1, message="Upload was empty", error="Upload was empty")
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")
    background_tasks.add_task(_run_job, job_id, request)
    return _api_job(record)


@app.get("/api/jobs/{job_id}", response_model=ApiJob, tags=["jobs"])
def get_job(job_id: str) -> ApiJob:
    """Read current progress and completed clip metadata for one job."""
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _api_job(record)


@app.get("/api/jobs/{job_id}/files/{file_name}", tags=["jobs"])
def download_clip(job_id: str, file_name: str) -> FileResponse:
    """Serve only an MP4 recorded as an output of the requested local job."""
    record = store.get(job_id)
    if record is None or record.result is None:
        raise HTTPException(status_code=404, detail="Export not found.")
    safe_name = Path(file_name).name
    if not re.fullmatch(r"clip-[A-Za-z0-9_.-]+\.mp4", safe_name, flags=re.IGNORECASE):
        raise HTTPException(status_code=404, detail="Export not found.")
    allowed = {
        clip.render.path.resolve()
        for clip in record.result.clips
        if clip.render.path.name == safe_name
    }
    if not allowed:
        raise HTTPException(status_code=404, detail="Export not found.")
    path = next(iter(allowed))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Export file is missing.")
    return FileResponse(path, media_type="video/mp4", filename=safe_name)
