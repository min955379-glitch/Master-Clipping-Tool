"""Shared loading of a project-local ``.env`` file for every entry point."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_local_env(dotenv_path: str | Path | None = None) -> bool:
    """Load the repository ``.env`` (or an explicit path) without overriding the shell."""
    path = Path(dotenv_path) if dotenv_path is not None else Path(__file__).resolve().parents[1] / ".env"
    return load_dotenv(dotenv_path=path, override=False)
