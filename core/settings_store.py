"""Private, per-user provider settings for the local studio."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlsplit

from core.models import ProviderName
from core.storage import default_data_root

DEFAULT_PROVIDER_SETTINGS: dict[str, dict[str, str]] = {
    "ollama": {"model": "qwen2.5:7b", "base_url": "http://localhost:11434"},
    "gemini": {
        "model": "gemini-3.8-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "openai": {"model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
    "anthropic": {"model": "claude-3-5-haiku-latest", "base_url": "https://api.anthropic.com"},
    "groq": {"model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
    "openrouter": {"model": "openai/gpt-4.1-mini", "base_url": "https://openrouter.ai/api/v1"},
}

API_KEY_ENV: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
BASE_URL_ENV: dict[str, str] = {"ollama": "OLLAMA_BASE_URL"}
_LOCK = RLock()


class SettingsStoreError(RuntimeError):
    """An invalid or unreadable local provider settings file."""


class InvalidProviderSettingsError(SettingsStoreError):
    """A submitted model, endpoint, or key violates a provider setting constraint."""


def settings_path() -> Path:
    """Return the private settings file under the user's FrameForge data folder."""
    return default_data_root() / "settings.json"


def _empty_document() -> dict[str, Any]:
    return {"active_provider": ProviderName.OLLAMA.value, "providers": {}}


def _read_document() -> dict[str, Any]:
    path = settings_path()
    if not path.exists():
        return _empty_document()
    try:
        if path.stat().st_size > 1_000_000:
            raise SettingsStoreError("Local provider settings exceed the 1 MB safety limit.")
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SettingsStoreError(f"Could not read local provider settings: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("providers", {}), dict):
        raise SettingsStoreError("Local provider settings have an invalid format.")
    return document


def provider_config(provider: ProviderName | str) -> dict[str, str]:
    """Resolve one provider's saved values, falling back to environment/defaults."""
    name = provider.value if isinstance(provider, ProviderName) else str(provider)
    if name not in DEFAULT_PROVIDER_SETTINGS:
        raise SettingsStoreError(f"Unsupported AI provider: {name}")
    with _LOCK:
        document = _read_document()
    saved = document.get("providers", {}).get(name, {})
    if not isinstance(saved, dict):
        saved = {}
    defaults = DEFAULT_PROVIDER_SETTINGS[name]
    model = str(saved.get("model") or defaults["model"]).strip()
    env_base = os.environ.get(BASE_URL_ENV.get(name, ""), "").strip() if name in BASE_URL_ENV else ""
    base_url = str(saved.get("base_url") or env_base or defaults["base_url"]).strip()
    env_key_name = API_KEY_ENV.get(name, "")
    api_key = str(saved.get("api_key") or (os.environ.get(env_key_name, "") if env_key_name else "")).strip()
    return {"model": model, "base_url": base_url, "api_key": api_key}


def active_provider() -> str:
    """Get the saved default provider, validating corrupt/stale values safely."""
    with _LOCK:
        value = _read_document().get("active_provider")
    return value if value in DEFAULT_PROVIDER_SETTINGS else ProviderName.OLLAMA.value


def public_settings() -> dict[str, Any]:
    """Return UI-safe settings metadata; secret values are never included."""
    document = _read_document()
    active = document.get("active_provider")
    if active not in DEFAULT_PROVIDER_SETTINGS:
        active = ProviderName.OLLAMA.value
    providers: dict[str, dict[str, Any]] = {}
    raw_providers = document.get("providers", {})
    for name, defaults in DEFAULT_PROVIDER_SETTINGS.items():
        raw = raw_providers.get(name, {})
        if not isinstance(raw, dict):
            raw = {}
        env_key_name = API_KEY_ENV.get(name, "")
        has_key = bool(raw.get("api_key") or (env_key_name and os.environ.get(env_key_name)))
        env_base = os.environ.get(BASE_URL_ENV.get(name, ""), "").strip() if name in BASE_URL_ENV else ""
        providers[name] = {
            "model": str(raw.get("model") or defaults["model"]),
            "base_url": str(raw.get("base_url") or env_base or defaults["base_url"]),
            "configured": has_key if env_key_name else True,
            "api_key_saved": has_key,
        }
    return {"active_provider": active, "providers": providers}


def save_provider_settings(
    provider: ProviderName | str,
    *,
    model: str,
    base_url: str = "",
    api_key: str | None = None,
    clear_api_key: bool = False,
) -> dict[str, Any]:
    """Persist one provider configuration; blank key fields preserve existing secrets."""
    name = provider.value if isinstance(provider, ProviderName) else str(provider)
    if name not in DEFAULT_PROVIDER_SETTINGS:
        raise SettingsStoreError(f"Unsupported AI provider: {name}")
    clean_model = (model or DEFAULT_PROVIDER_SETTINGS[name]["model"]).strip()
    clean_base = (base_url or "").strip()
    clean_key = (api_key or "").strip()
    if not clean_model or len(clean_model) > 200:
        raise InvalidProviderSettingsError("Model name must contain between 1 and 200 characters.")
    if len(clean_base) > 2000:
        raise InvalidProviderSettingsError("Provider base URL is too long.")
    if clean_base:
        try:
            parsed_base = urlsplit(clean_base)
            _ = parsed_base.port
        except ValueError as exc:
            raise InvalidProviderSettingsError("Provider base URL is invalid.") from exc
        if (
            parsed_base.scheme not in {"http", "https"}
            or not parsed_base.netloc
            or parsed_base.username
            or parsed_base.password
            or parsed_base.query
            or parsed_base.fragment
        ):
            raise InvalidProviderSettingsError(
                "Provider base URLs must use http:// or https:// and cannot include credentials, query parameters, or fragments."
            )
    if len(clean_key) > 4096:
        raise InvalidProviderSettingsError("API key exceeds the 4096-character safety limit.")

    with _LOCK:
        document = _read_document()
        providers = document.setdefault("providers", {})
        previous = providers.get(name, {})
        if not isinstance(previous, dict):
            previous = {}
        updated: dict[str, str] = {"model": clean_model, "base_url": clean_base}
        old_key = str(previous.get("api_key") or "")
        if clean_key:
            updated["api_key"] = clean_key
        elif not clear_api_key and old_key:
            updated["api_key"] = old_key
        providers[name] = updated
        document["active_provider"] = name
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if os.name != "nt":
                temporary.chmod(0o600)
            temporary.replace(path)
            if os.name != "nt":
                path.chmod(0o600)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise SettingsStoreError(f"Could not save local provider settings: {exc}") from exc
    return public_settings()


def configured_provider_keys() -> dict[str, bool]:
    """Return provider key-availability flags without exposing any secret material."""
    providers = public_settings()["providers"]
    return {name: bool(config.get("api_key_saved")) for name, config in providers.items()}
