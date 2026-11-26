"""Compatibility entry point for ``python cli.py``."""

from __future__ import annotations

from core.env import load_local_env


def main(argv: list[str] | None = None) -> int:
    """Load local configuration before delegating to the shared CLI adapter."""
    load_local_env()
    from core.cli import main as core_main

    return core_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
