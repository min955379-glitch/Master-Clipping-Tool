"""Compatibility entry point for ``python cli.py``."""

from core.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
