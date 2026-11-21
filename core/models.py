"""Shared, validated contracts for ingestion, ranking, reframing and rendering."""

from __future__ import annotations

import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProviderName(str, Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"


class ReframeMode(str, Enum):
    FACE_TRACK = "face_track"
    PODCAST_SPLIT = "podcast_split"
    GAMING = "gaming"
    FIT_BLUR = "fit_blur"


class CaptionPreset(str, Enum):
    HORMOZI_BOLD = "Hormozi Bold"
    CLEAN_MINIMAL = "Clean Minimal"
    GAMING_NEON = "Gaming Neon"
    PODCAST_CLASSIC = "Podcast Classic"


class ExportPreset(str, Enum):
    SHORT_VERTICAL = "short_vertical"
    SQUARE = "square"
    LANDSCAPE = "landscape"


class StudioModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Word(StudioModel):
    start: float = Field(ge=0, description="Absolute source start time in seconds")
    end: float = Field(gt=0, description="Absolute source end time in seconds")
    text: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def check_order(self) -> Word:
        if self.end <= self.start:
            raise ValueError("word end must be after word start")
        if not self.text:
            raise ValueError("word text cannot be blank")
        return self


class AudioPeak(StudioModel):
    time: float = Field(ge=0, description="Absolute source time in seconds")
    score: float = Field(ge=0, le=1)
    loudness_dbfs: float = Field(ge=-120, le=0)
    baseline_dbfs: float = Field(ge=-120, le=0)


class MediaMetadata(StudioModel):
    source_id: str
    title: str = "Untitled video"
    duration: float = Field(gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    has_audio: bool = True
    language: str | None = None
    thumbnail: str | None = None


class SourceDescriptor(StudioModel):
    kind: Literal["local", "youtube", "twitch"]
    source_id: str
    url: str | None = None
    local_path: Path | None = None
    audio_path: Path | None = None

    @model_validator(mode="after")
    def check_source(self) -> SourceDescriptor:
        if self.kind == "local" and self.local_path is None:
            raise ValueError("local sources require local_path")
        if self.kind != "local" and not self.url:
            raise ValueError("remote sources require a URL")
        return self


class IngestResult(StudioModel):
    metadata: MediaMetadata
    source: SourceDescriptor
    words: list[Word] = Field(default_factory=list)
    audio_peaks: list[AudioPeak] = Field(default_factory=list)
    transcript_source: Literal["youtube_captions", "faster_whisper", "none"] = "none"
    warnings: list[str] = Field(default_factory=list)


class ClipCandidate(StudioModel):
    start: float = Field(ge=0, description="Absolute source start time in seconds")
    end: float = Field(gt=0, description="Absolute source end time in seconds")
    virality_score: float = Field(ge=0, le=100)
    hook_title: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=600)
    suggested_caption: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def check_range(self) -> ClipCandidate:
        if self.end <= self.start:
            raise ValueError("clip end must be after clip start")
        return self


class JobRequest(StudioModel):
    source: str = Field(min_length=1, max_length=4096)
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=8, max_length=80)
    work_dir: Path | None = None
    output_dir: Path | None = None
    provider: ProviderName = ProviderName.OLLAMA
    model: str | None = None
    base_url: str = "http://localhost:11434"
    api_key: str | None = Field(default=None, repr=False, exclude=True)
    custom_direction: str = Field(default="", max_length=2000)
    language: str | None = Field(default=None, max_length=24)
    layout: ReframeMode = ReframeMode.FACE_TRACK
    caption_style: CaptionPreset = CaptionPreset.HORMOZI_BOLD
    export_preset: ExportPreset = ExportPreset.SHORT_VERTICAL
    clip_count: int = Field(default=4, ge=1, le=12)
    min_clip_seconds: float = Field(default=15, ge=5, le=175)
    max_clip_seconds: float = Field(default=60, ge=10, le=180)
    transcribe_missing: bool = True
    asr_model: str = Field(default="small", min_length=1, max_length=80)
    request_timeout_seconds: float = Field(default=120, ge=5, le=600)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", value):
            raise ValueError("job_id may contain only letters, numbers, underscores, and hyphens")
        return value

    @model_validator(mode="after")
    def check_clip_limits(self) -> JobRequest:
        if self.max_clip_seconds <= self.min_clip_seconds:
            raise ValueError("max_clip_seconds must be greater than min_clip_seconds")
        return self


class FaceBox(StudioModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    confidence: float = Field(default=1.0, ge=0, le=1)


class CropKeyframe(StudioModel):
    time: float = Field(ge=0, description="Time relative to the output clip")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ReframePlan(StudioModel):
    requested_mode: ReframeMode
    mode: ReframeMode
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    output_width: int = Field(gt=0)
    output_height: int = Field(gt=0)
    filter_complex: str
    keyframes: list[CropKeyframe] = Field(default_factory=list)
    webcam_box: FaceBox | None = None
    warnings: list[str] = Field(default_factory=list)


class PreparedClipSource(StudioModel):
    path: Path
    trim_start: float = Field(ge=0)
    duration: float = Field(gt=0)
    source_start: float = Field(ge=0, description="Source time represented at file time zero")
    partial: bool = False
    warnings: list[str] = Field(default_factory=list)


class RenderResult(StudioModel):
    path: Path
    encoder: str
    duration: float = Field(gt=0)
    warnings: list[str] = Field(default_factory=list)


class PipelineProgress(StudioModel):
    job_id: str
    stage: str
    progress: float = Field(ge=0, le=1)
    message: str = ""
    current_clip: int | None = Field(default=None, ge=1)
    total_clips: int | None = Field(default=None, ge=1)


class ClipOutput(StudioModel):
    candidate: ClipCandidate
    render: RenderResult
    reframe: ReframePlan


class PipelineResult(StudioModel):
    job_id: str
    status: Literal["completed", "partial", "failed"]
    metadata: MediaMetadata | None = None
    clips: list[ClipOutput] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
