"""Compatibility entry point for ``python mcp_server.py``."""

from __future__ import annotations

from core.env import load_local_env


def main() -> None:
    """Load local configuration before delegating to the shared MCP server."""
    load_local_env()
    from core.mcp_server import main as core_main

    core_main()


if __name__ == "__main__":
    main()
