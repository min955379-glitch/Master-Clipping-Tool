"""One shared ingest-to-export pipeline used by API, CLI, MCP and test harnesses."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from core.brain import BrainServices, rank_clips
from core.ingest import IngestServices, default_job_workdir, ingest_source, probe_media
from core.models import (
    ClipCandidate,
    ClipOutput,
    IngestResult,
    JobRequest,
    PipelineProgress,
    PipelineResult,
    PreparedClipSource,
    RenderResult,
)
from core.reframe import ReframeServices, plan_reframe
from core.render import RenderServices, prepare_clip_source, render_clip

logger = logging.getLogger(__name__)
ProgressHandler = Callable[[PipelineProgress], None]
ProviderCall = Callable[[str, JobRequest], str]


@dataclass
class PipelineServices:
    """Inject lower-level external services for deterministic tests and hosts."""

    ingest: IngestServices = field(default_factory=IngestServices)
    brain: BrainServices = field(default_factory=BrainServices)
    reframe: ReframeServices = field(default_factory=ReframeServices)
    render: RenderServices = field(default_factory=RenderServices)
    provider_call: ProviderCall | None = None


def _safe_filename(candidate: ClipCandidate, index: int) -> str:
    title = re.sub(r"[^A-Za-z0-9_-]+", "-", candidate.hook_title).strip("-_")[:48]
    if not title:
        title = "clip"
    return f"clip-{index:02d}-{candidate.start:07.2f}-{title}.mp4"


def _push_progress(
    callback: ProgressHandler | None,
    *,
    job_id: str,
    stage: str,
    progress: float,
    message: str,
    current_clip: int | None = None,
    total_clips: int | None = None,
) -> None:
    if callback is None:
        return
    event = PipelineProgress(
        job_id=job_id,
        stage=stage,
        progress=max(0.0, min(1.0, progress)),
        message=message,
        current_clip=current_clip,
        total_clips=total_clips,
    )
    try:
        callback(event)
    except Exception as exc:  # noqa: BLE001 - progress subscribers must not abort media work.
        logger.warning("Progress callback failed for job %s: %s", job_id, exc)


def _clip_dimensions(
    prepared: PreparedClipSource,
    ingest: IngestResult,
    services: PipelineServices,
) -> tuple[int, int]:
    width, height = ingest.metadata.width, ingest.metadata.height
    if width and height:
        return width, height
    probed = probe_media(prepared.path, run=services.ingest.run)
    if not probed.width or not probed.height:
        raise RuntimeError(f"Could not determine source dimensions for {prepared.path.name}.")
    return probed.width, probed.height


def run_pipeline(
    request: JobRequest,
    *,
    services: PipelineServices | None = None,
    progress: ProgressHandler | None = None,
) -> PipelineResult:
    """Run the complete local-first workflow and return a typed, serializable result.

    This is the sole high-level pipeline entry point. Frontends may schedule it
    asynchronously, but ingest, AI ranking, reframing, captions and rendering
    always run through this shared implementation.
    """
    dependencies = services or PipelineServices()
    work_dir = request.work_dir.expanduser().resolve() if request.work_dir else default_job_workdir(request)
    output_dir = request.output_dir.expanduser().resolve() if request.output_dir else work_dir / "exports"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_request = request.model_copy(update={"work_dir": work_dir, "output_dir": output_dir})
    job_id = effective_request.job_id
    _push_progress(
        progress,
        job_id=job_id,
        stage="QUEUED",
        progress=0.0,
        message="Job is ready to process",
    )

    try:
        _push_progress(progress, job_id=job_id, stage="INGEST", progress=0.02, message="Starting source ingest")
        ingest = ingest_source(
            effective_request,
            services=dependencies.ingest,
            progress=lambda stage, fraction, message: _push_progress(
                progress,
                job_id=job_id,
                stage=stage,
                progress=0.02 + fraction * 0.34,
                message=message,
            ),
        )
        _push_progress(
            progress,
            job_id=job_id,
            stage="RANK",
            progress=0.39,
            message="Ranking transcript and audio moments with the selected AI provider",
        )
        candidates = rank_clips(
            ingest,
            effective_request,
            services=dependencies.brain,
            provider_call=dependencies.provider_call,
        )
    except Exception as exc:
        logger.exception("Pipeline failed before rendering (job %s)", job_id)
        _push_progress(progress, job_id=job_id, stage="FAILED", progress=1.0, message=str(exc))
        return PipelineResult(job_id=job_id, status="failed", error=str(exc))

    total = len(candidates)
    _push_progress(
        progress,
        job_id=job_id,
        stage="RANKED",
        progress=0.47,
        message=f"Selected {total} ranked clips",
        total_clips=total,
    )
    clip_outputs: list[ClipOutput] = []
    warnings = list(ingest.warnings)
    failures: list[str] = []
    interval = 0.48 / max(1, total)

    for index, candidate in enumerate(candidates, start=1):
        base = 0.49 + (index - 1) * interval
        target_path = output_dir / _safe_filename(candidate, index)
        try:
            _push_progress(
                progress,
                job_id=job_id,
                stage="PREPARE",
                progress=base,
                message=f"Preparing clip {index} of {total}",
                current_clip=index,
                total_clips=total,
            )
            prepared = prepare_clip_source(
                ingest.source,
                candidate,
                work_dir,
                services=dependencies.render,
            )
            source_width, source_height = _clip_dimensions(prepared, ingest, dependencies)
            reframe = plan_reframe(
                prepared.path,
                source_width=source_width,
                source_height=source_height,
                clip_start=prepared.trim_start,
                duration=candidate.end - candidate.start,
                mode=effective_request.layout,
                export_preset=effective_request.export_preset,
                services=dependencies.reframe,
            )

            def render_progress(
                stage: str,
                fraction: float,
                message: str,
                clip_base: float = base,
                clip_index: int = index,
            ) -> None:
                _push_progress(
                    progress,
                    job_id=job_id,
                    stage=stage,
                    progress=clip_base + interval * (0.12 + 0.86 * fraction),
                    message=message,
                    current_clip=clip_index,
                    total_clips=total,
                )

            rendered: RenderResult = render_clip(
                ingest.source,
                candidate,
                reframe,
                ingest.words,
                target_path,
                work_dir=work_dir,
                caption_preset=effective_request.caption_style.value,
                suggested_caption=candidate.suggested_caption,
                has_audio=ingest.metadata.has_audio,
                services=dependencies.render,
                progress=render_progress,
                prepared_source=prepared,
            )
            warnings.extend(reframe.warnings)
            warnings.extend(rendered.warnings)
            clip_outputs.append(ClipOutput(candidate=candidate, render=rendered, reframe=reframe))
        except Exception as exc:
            failure = f"Clip {index} ({candidate.hook_title}) could not be rendered: {exc}"
            logger.exception("Clip rendering failed (job %s, clip %s)", job_id, index)
            failures.append(failure)
            warnings.append(failure)

    if not clip_outputs:
        detail = "; ".join(failures) or "No clips were rendered."
        _push_progress(progress, job_id=job_id, stage="FAILED", progress=1.0, message=detail)
        return PipelineResult(
            job_id=job_id,
            status="failed",
            metadata=ingest.metadata,
            warnings=warnings,
            error=detail,
        )

    partial = len(clip_outputs) < total or total < effective_request.clip_count
    result = PipelineResult(
        job_id=job_id,
        status="partial" if partial else "completed",
        metadata=ingest.metadata,
        clips=clip_outputs,
        warnings=warnings,
    )
    _push_progress(
        progress,
        job_id=job_id,
        stage="COMPLETED" if not partial else "PARTIAL",
        progress=1.0,
        message=f"Exported {len(clip_outputs)} of {total} selected clips",
        current_clip=len(clip_outputs) or None,
        total_clips=total,
    )
    return result
