"""Local-first job/media storage locations shared by API, CLI and MCP."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIRECTORY_NAME = "FrameForge"


def default_data_root() -> Path:
    """Return a per-user writable data directory, honoring an explicit override."""
    configured = os.environ.get("FRAMEFORGE_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_DIRECTORY_NAME


def jobs_root() -> Path:
    """Return the directory that contains one isolated folder per job."""
    return default_data_root() / "jobs"


def job_directory(job_id: str) -> Path:
    """Resolve a validated job ID to its local work directory."""
    if not job_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in job_id):
        raise ValueError("job_id contains unsafe path characters")
    return jobs_root() / job_id
