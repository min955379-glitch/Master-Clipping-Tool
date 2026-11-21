from __future__ import annotations

import asyncio
import unittest

from core.mcp_server import create_server


class McpAdapterTests(unittest.TestCase):
    def test_registers_shared_pipeline_and_runtime_tools_when_mcp_is_installed(self) -> None:
        try:
            server = create_server()
        except RuntimeError as exc:
            if "MCP adapter dependency is missing" in str(exc):
                self.skipTest(str(exc))
            raise
        tools = asyncio.run(server.list_tools())
        self.assertEqual({tool.name for tool in tools}, {"process_video", "get_local_runtime"})
        process_video = next(tool for tool in tools if tool.name == "process_video")
        self.assertIn("source", process_video.input_schema["properties"])


if __name__ == "__main__":
    unittest.main()
