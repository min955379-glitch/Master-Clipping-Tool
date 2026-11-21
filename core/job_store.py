"""Small atomic JSON job registry persisted under the user's local app-data folder."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import Field

from core.models import PipelineResult, StudioModel
from core.storage import jobs_root

logger = logging.getLogger(__name__)


class StoredJob(StudioModel):
    job_id: str
    status: Literal["queued", "running", "completed", "partial", "failed"] = "queued"
    progress: float = Field(default=0.0, ge=0, le=1)
    stage: str = "QUEUED"
    message: str = "Waiting to start"
    source_label: str = ""
    created_at: datetime
    updated_at: datetime
    result: PipelineResult | None = None
    error: str | None = None


def _now() -> datetime:
    return datetime.now(UTC)


def _safe_job_id(job_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{8,80}", job_id or ""))


class JobStore:
    """Thread-safe in-process job index with atomic local JSON snapshots."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else jobs_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._jobs: dict[str, StoredJob] = {}
        self._restore()

    def work_dir(self, job_id: str) -> Path:
        if not _safe_job_id(job_id):
            raise ValueError("Unsafe job ID")
        return self.root / job_id

    def create(self, job_id: str, source_label: str = "") -> StoredJob:
        if not _safe_job_id(job_id):
            raise ValueError("Unsafe job ID")
        now = _now()
        record = StoredJob(
            job_id=job_id,
            status="queued",
            progress=0,
            stage="QUEUED",
            message="Waiting to start",
            source_label=source_label[:200],
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            if job_id in self._jobs:
                raise ValueError(f"Job {job_id} already exists")
            self.work_dir(job_id).mkdir(parents=True, exist_ok=False)
            self._jobs[job_id] = record
            self._persist(record)
            return record.model_copy(deep=True)

    def update(self, job_id: str, **changes: object) -> StoredJob:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise KeyError(job_id)
            updated = current.model_copy(update={**changes, "updated_at": _now()})
            self._jobs[job_id] = updated
            self._persist(updated)
            return updated.model_copy(deep=True)

    def finish(self, job_id: str, result: PipelineResult) -> StoredJob:
        status = result.status
        message = {
            "completed": "All selected clips were exported",
            "partial": "The job finished with some skipped clips or warnings",
            "failed": result.error or "The job failed",
        }[status]
        return self.update(
            job_id,
            status=status,
            progress=1.0,
            stage=status.upper(),
            message=message,
            result=result,
            error=result.error,
        )

    def get(self, job_id: str) -> StoredJob | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record.model_copy(deep=True) if record else None

    def list(self, limit: int = 100) -> list[StoredJob]:
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [record.model_copy(deep=True) for record in records[: max(1, limit)]]

    def _persist(self, record: StoredJob) -> None:
        directory = self.work_dir(record.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "job.json"
        temporary = directory / "job.json.tmp"
        temporary.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(target)

    def _restore(self) -> None:
        for record_path in self.root.glob("*/job.json"):
            try:
                record = StoredJob.model_validate_json(record_path.read_text(encoding="utf-8"))
                if record_path.parent.name != record.job_id or not _safe_job_id(record.job_id):
                    logger.warning("Skipping job snapshot with a mismatched or unsafe ID: %s", record_path)
                    continue
                if record.status in {"queued", "running"}:
                    record = record.model_copy(
                        update={
                            "status": "failed",
                            "stage": "FAILED",
                            "message": "Job interrupted when the local service stopped",
                            "error": "The previous API process stopped before this job completed.",
                            "updated_at": _now(),
                        }
                    )
                    self._persist(record)
                self._jobs[record.job_id] = record
            except Exception as exc:  # noqa: BLE001 - corrupt snapshots must not block other jobs.
                logger.warning("Could not restore job snapshot %s: %s", record_path, exc)
