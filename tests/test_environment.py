from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.env import load_local_env


class EnvironmentLoadingTests(unittest.TestCase):
    def test_load_local_env_reads_dotenv_and_keeps_shell_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dotenv = Path(temporary) / ".env"
            dotenv.write_text(
                "FRAMEFORGE_DOTENV_TEST=from-file\nFRAMEFORGE_SHELL_VALUE=from-file\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"FRAMEFORGE_SHELL_VALUE": "from-shell"}, clear=False):
                os.environ.pop("FRAMEFORGE_DOTENV_TEST", None)
                self.assertTrue(load_local_env(dotenv))
                self.assertEqual(os.environ["FRAMEFORGE_DOTENV_TEST"], "from-file")
                self.assertEqual(os.environ["FRAMEFORGE_SHELL_VALUE"], "from-shell")
                os.environ.pop("FRAMEFORGE_DOTENV_TEST", None)

    def test_entrypoint_mains_invoke_dotenv_loader(self) -> None:
        import test_pipeline
        from core import cli, mcp_server

        with patch("core.cli.load_local_env") as cli_loader, patch("core.cli.run_pipeline") as pipeline:
            result = type("Result", (), {"status": "completed", "model_dump": lambda _self, **_: {}})()
            pipeline.return_value = result
            self.assertEqual(cli.main(["--source", "source.mp4", "--json"]), 0)
            cli_loader.assert_called_once_with()

        with patch("core.mcp_server.load_local_env") as mcp_loader, patch("core.mcp_server.create_server") as server_factory:
            mcp_server.main()
            mcp_loader.assert_called_once_with()
            server_factory.return_value.run.assert_called_once_with()

        with patch("test_pipeline.load_local_env") as smoke_loader, patch("test_pipeline._mocked_check", return_value={"ok": True}):
            self.assertEqual(test_pipeline.main(["--mocked"]), 0)
            smoke_loader.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
