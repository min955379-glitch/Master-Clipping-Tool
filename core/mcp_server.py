"""Optional stdio MCP server exposing the shared clip pipeline as local tools."""

from __future__ import annotations

import os
from typing import Any

from core.encoder import EncoderProbeError, detect_encoder
from core.env import load_local_env
from core.models import CaptionPreset, ExportPreset, JobRequest, ProviderName, ReframeMode
from core.pipeline import run_pipeline
from core.settings_store import provider_config, public_settings


def create_server() -> Any:
    """Construct an MCP server lazily so importing core never requires MCP."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError(
            "The MCP adapter dependency is missing. Run setup.bat or install with: pip install -r requirements.txt"
        ) from exc

    server = MCPServer("FrameForge Local Video Studio")

    @server.tool()
    def process_video(
        source: str,
        provider: str = "ollama",
        model: str | None = None,
        custom_direction: str = "",
        layout: str = "face_track",
        caption_style: str = "Hormozi Bold",
        export_preset: str = "short_vertical",
        clip_count: int = 4,
        min_clip_seconds: float = 15,
        max_clip_seconds: float = 60,
        language: str | None = None,
        transcribe_missing: bool = True,
    ) -> dict[str, Any]:
        """Run the shared local pipeline and return validated clip metadata and file paths."""
        request = JobRequest(
            source=source,
            provider=ProviderName(provider),
            model=model,
            custom_direction=custom_direction,
            layout=ReframeMode(layout),
            caption_style=CaptionPreset(caption_style),
            export_preset=ExportPreset(export_preset),
            clip_count=clip_count,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
            language=language,
            transcribe_missing=transcribe_missing,
        )
        return run_pipeline(request).model_dump(mode="json")

    @server.tool()
    def get_local_runtime() -> dict[str, Any]:
        """Report encoder selection and provider configuration without revealing secrets."""
        try:
            selection = detect_encoder(os.environ.get("FFMPEG_PATH", "ffmpeg"))
            encoder = {
                "available": True,
                "selected": selection.name,
                "hardware_accelerated": selection.hardware_accelerated,
                "supported": list(selection.available),
            }
        except EncoderProbeError as exc:
            encoder = {"available": False, "error": str(exc)}
        settings = public_settings()
        return {
            "encoder": encoder,
            "providers_configured": {
                name: bool(config.get("configured"))
                for name, config in settings["providers"].items()
            },
            "ollama_url": provider_config(ProviderName.OLLAMA)["base_url"],
        }

    return server


def main() -> None:
    """Run the stdio MCP transport used by local MCP hosts."""
    load_local_env()
    create_server().run()


if __name__ == "__main__":
    main()
