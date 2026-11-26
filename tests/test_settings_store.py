from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.brain import DEFAULT_MODELS
from core.models import ProviderName
from core.settings_store import (
    DEFAULT_PROVIDER_SETTINGS,
    InvalidProviderSettingsError,
    provider_config,
    public_settings,
    save_provider_settings,
)


class SettingsStoreTests(unittest.TestCase):
    def test_gemini_default_model_is_the_requested_version(self) -> None:
        self.assertEqual(DEFAULT_PROVIDER_SETTINGS[ProviderName.GEMINI.value]["model"], "gemini-3.8-flash")
        self.assertEqual(DEFAULT_MODELS[ProviderName.GEMINI], "gemini-3.8-flash")

    def test_provider_secrets_persist_locally_but_never_enter_public_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"FRAMEFORGE_DATA_DIR": temporary, "OPENAI_API_KEY": ""}
        ):
            key = "sk-local-test-secret"
            public = save_provider_settings(
                "openai",
                model="gpt-test-model",
                base_url="https://api.openai.com/v1",
                api_key=key,
            )
            config = provider_config("openai")
            path = Path(temporary) / "settings.json"
            content = path.read_text(encoding="utf-8")

            self.assertEqual(config["api_key"], key)
            self.assertTrue(public["providers"]["openai"]["configured"])
            self.assertTrue(public["providers"]["openai"]["api_key_saved"])
            self.assertNotIn(key, repr(public))
            self.assertIn(key, content)
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

            save_provider_settings("openai", model="gpt-next", base_url="", api_key="")
            self.assertEqual(provider_config("openai")["api_key"], key)
            save_provider_settings("openai", model="gpt-next", clear_api_key=True)
            self.assertEqual(provider_config("openai")["api_key"], "")
            self.assertFalse(public_settings()["providers"]["openai"]["configured"])

    def test_base_urls_cannot_persist_embedded_credentials_or_query_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"FRAMEFORGE_DATA_DIR": temporary}
        ):
            for unsafe_url in (
                "https://user:secret@example.com/v1",
                "https://example.com/v1?token=secret",
            ):
                with self.subTest(url=unsafe_url), self.assertRaises(InvalidProviderSettingsError):
                    save_provider_settings("openai", model="gpt-test", base_url=unsafe_url)
            self.assertFalse((Path(temporary) / "settings.json").exists())


if __name__ == "__main__":
    unittest.main()
