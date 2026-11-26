"""Provider-neutral clip discovery across local Ollama and configurable AI APIs."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from core.models import AudioPeak, ClipCandidate, IngestResult, JobRequest, ProviderName, Word
from core.settings_store import API_KEY_ENV, DEFAULT_PROVIDER_SETTINGS, provider_config

DEFAULT_MODELS = {
    ProviderName(name): values["model"] for name, values in DEFAULT_PROVIDER_SETTINGS.items()
}
MAX_PROVIDER_RESPONSE_BYTES = 2_000_000
MAX_TRANSCRIPT_CHARS = 72_000
MAX_PEAKS_IN_PROMPT = 200


class BrainError(RuntimeError):
    """Actionable failure from provider setup or clip response validation."""


class ClipEnvelope(BaseModel):
    """JSON response contract passed to compatible local model endpoints."""

    model_config = ConfigDict(extra="forbid")
    clips: list[ClipCandidate]


@dataclass
class BrainServices:
    """Injectable transport seam so providers can be tested without network calls."""

    urlopen: Callable[..., Any] | None = None


def _clip_json_schema() -> dict[str, Any]:
    return ClipEnvelope.model_json_schema()


def _clean_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        raw = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BrainError("Ollama URL must use http:// or https:// and include a host.")
    if parsed.username or parsed.password:
        raise BrainError("Ollama URL must not contain embedded credentials.")
    path = parsed.path.rstrip("/")
    path = path.removesuffix("/v1").removesuffix("/api/chat")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _read_response(request: urllib.request.Request, timeout: float, opener: Callable[..., Any] | None) -> dict[str, Any]:
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            raw = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(2000).decode("utf-8", errors="replace")
        except (OSError, ValueError) as read_error:
            detail = f"Could not read provider error body: {read_error}"
        safe_detail = detail[:500]
        safe_reason = str(exc.reason)
        secret_values: set[str] = set()
        query_key = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query).get("key", [])
        secret_values.update(query_key)
        for header_name, header_value in request.header_items():
            if header_name.lower() in {"authorization", "x-api-key", "x-goog-api-key"}:
                secret_values.add(header_value)
                if header_name.lower() == "authorization" and header_value.lower().startswith("bearer "):
                    secret_values.add(header_value[7:].strip())
        for secret in sorted(secret_values, key=len, reverse=True):
            if secret:
                safe_detail = safe_detail.replace(secret, "[redacted]")
                safe_reason = safe_reason.replace(secret, "[redacted]")
        if "key=" in request.full_url:
            safe_detail = re.sub(r"(?i)(key=)[^&\s]+", r"\1[redacted]", safe_detail)
            safe_reason = re.sub(r"(?i)(key=)[^&\s]+", r"\1[redacted]", safe_reason)
        raise BrainError(f"AI provider returned HTTP {exc.code}: {safe_detail or safe_reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise BrainError(f"Could not reach the AI provider: {reason}") from exc
    if len(raw) > MAX_PROVIDER_RESPONSE_BYTES:
        raise BrainError("AI provider response exceeded the 2 MB safety limit.")
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrainError("AI provider returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise BrainError("AI provider response must be a JSON object.")
    return payload


def _post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    opener: Callable[..., Any] | None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json", "Accept": "application/json", **(headers or {})},
        method="POST",
    )
    return _read_response(request, timeout, opener)


def call_ollama(
    prompt: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str | None = None,
    timeout: float = 120,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Call Ollama's native chat API, asking it to honor a JSON schema."""
    root = _clean_base_url(base_url)
    body = {
        "model": model or DEFAULT_MODELS[ProviderName.OLLAMA],
        "stream": False,
        "format": _clip_json_schema(),
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": "You identify compelling, self-contained video clips. Return only schema-valid JSON."},
            {"role": "user", "content": prompt},
        ],
    }
    payload = _post_json(
        root + "/api/chat",
        body,
        timeout=timeout,
        opener=opener,
    )
    message = payload.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        content = payload.get("response")
    if not isinstance(content, str) or not content.strip():
        raise BrainError("Ollama returned no text content.")
    return content


def call_gemini(
    prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str = "",
    timeout: float = 120,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Call the Gemini generateContent endpoint; key is supplied only at runtime."""
    key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise BrainError("Gemini is opt-in: set GEMINI_API_KEY or provide an API key in the job settings.")
    chosen_model = model or DEFAULT_MODELS[ProviderName.GEMINI]
    base = _clean_api_base(base_url, ProviderName.GEMINI)
    endpoint = (
        base
        + ("" if base.endswith("/models") else "/models")
        + "/"
        + urllib.parse.quote(chosen_model, safe="")
        + ":generateContent"
    )
    payload = _post_json(
        endpoint,
        {
            "systemInstruction": {
                "parts": [{"text": "You identify compelling, self-contained video clips. Return only schema-valid JSON."}]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
        },
        timeout=timeout,
        opener=opener,
        headers={"x-goog-api-key": key},
    )
    candidates = payload.get("candidates") or []
    try:
        parts = candidates[0]["content"]["parts"]
    except (IndexError, KeyError, TypeError) as exc:
        block = payload.get("promptFeedback") or {}
        reason = block.get("blockReason") if isinstance(block, dict) else None
        raise BrainError(f"Gemini returned no usable candidate{f' ({reason})' if reason else ''}.") from exc
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text.strip():
        raise BrainError("Gemini returned no text content.")
    return text


def _provider_key(provider: ProviderName, api_key: str | None) -> str:
    key = (api_key or "").strip()
    if not key:
        environment_name = API_KEY_ENV.get(provider.value)
        key = os.environ.get(environment_name, "").strip() if environment_name else ""
    if not key:
        raise BrainError(f"{provider.value.title()} API key is not configured. Add it in Settings or the local .env file.")
    return key


def _clean_api_base(base_url: str, provider: ProviderName) -> str:
    default = DEFAULT_PROVIDER_SETTINGS[provider.value]["base_url"]
    raw = (base_url or default).strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BrainError(f"{provider.value.title()} base URL must use http:// or https:// and include a host.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BrainError("Provider base URLs cannot include credentials, query parameters, or fragments.")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")).rstrip("/")


def _chat_text(payload: dict[str, Any], provider: str) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise BrainError(f"{provider.title()} returned no usable chat completion.") from exc
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise BrainError(f"{provider.title()} returned no text content.")
    return content


def call_openai_compatible(
    prompt: str,
    *,
    provider: ProviderName,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str = "",
    timeout: float = 120,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Call OpenAI, Groq, or OpenRouter through their compatible chat APIs."""
    if provider not in {ProviderName.OPENAI, ProviderName.GROQ, ProviderName.OPENROUTER}:
        raise BrainError(f"{provider.value} does not use the OpenAI-compatible chat API.")
    key = _provider_key(provider, api_key)
    base = _clean_api_base(base_url, provider)
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = _post_json(
        endpoint,
        {
            "model": model or DEFAULT_MODELS[provider],
            "messages": [
                {"role": "system", "content": "You identify compelling, self-contained video clips. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout,
        opener=opener,
        headers={"Authorization": f"Bearer {key}"},
    )
    return _chat_text(payload, provider.value)


def _anthropic_messages_url(base_url: str, provider: ProviderName) -> str:
    base = _clean_api_base(base_url, provider)
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


def call_anthropic(
    prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str = "",
    timeout: float = 120,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Call Anthropic's Messages API with no SDK dependency."""
    provider = ProviderName.ANTHROPIC
    key = _provider_key(provider, api_key)
    endpoint = _anthropic_messages_url(base_url, provider)
    payload = _post_json(
        endpoint,
        {
            "model": model or DEFAULT_MODELS[provider],
            "max_tokens": 4096,
            "system": "You identify compelling, self-contained video clips. Return only valid JSON.",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
        opener=opener,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    content = payload.get("content") or []
    text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not text.strip():
        raise BrainError("Anthropic returned no text content.")
    return text


def test_provider_connection(
    provider: ProviderName | str,
    *,
    model: str,
    base_url: str = "",
    api_key: str | None = None,
    timeout: float = 20,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Send a small real request and return a safe, human-readable pass message."""
    name = ProviderName(provider)
    chosen_model = (model or DEFAULT_MODELS[name]).strip()
    prompt = "Reply with the single word OK."
    if name == ProviderName.OLLAMA:
        root = _clean_base_url(base_url)
        payload = _post_json(
            root + "/api/chat",
            {
                "model": chosen_model,
                "stream": False,
                "options": {"num_predict": 8, "temperature": 0},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
            opener=opener,
        )
        message = payload.get("message") or {}
        text = message.get("content") if isinstance(message, dict) else payload.get("response")
    elif name == ProviderName.GEMINI:
        key = _provider_key(name, api_key)
        base = _clean_api_base(base_url, name)
        endpoint = (
            base
            + ("" if base.endswith("/models") else "/models")
            + "/"
            + urllib.parse.quote(chosen_model, safe="")
            + ":generateContent"
        )
        payload = _post_json(
            endpoint,
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 8, "temperature": 0},
            },
            timeout=timeout,
            opener=opener,
            headers={"x-goog-api-key": key},
        )
        candidates = payload.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    elif name == ProviderName.ANTHROPIC:
        key = _provider_key(name, api_key)
        endpoint = _anthropic_messages_url(base_url, name)
        payload = _post_json(
            endpoint,
            {
                "model": chosen_model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
            opener=opener,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        content = payload.get("content") or []
        text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    else:
        key = _provider_key(name, api_key)
        base = _clean_api_base(base_url, name)
        endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        payload = _post_json(
            endpoint,
            {"model": chosen_model, "max_tokens": 8, "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
            opener=opener,
            headers={"Authorization": f"Bearer {key}"},
        )
        text = _chat_text(payload, name.value)
    if not isinstance(text, str) or not text.strip():
        raise BrainError(f"{name.value.title()} returned an empty test response.")
    return f"Connected to {name.value.title()} ({chosen_model})."


def route_provider(
    prompt: str,
    request: JobRequest,
    *,
    services: BrainServices | None = None,
) -> str:
    """Route through the selected provider, using private saved settings as defaults."""
    opener = (services or BrainServices()).urlopen
    provider = request.provider
    saved = provider_config(provider)
    model = request.model or saved["model"] or DEFAULT_MODELS[provider]
    api_key = request.api_key or saved["api_key"] or None
    timeout = request.request_timeout_seconds
    if provider == ProviderName.OLLAMA:
        configured_base = saved["base_url"]
        default_base = DEFAULT_PROVIDER_SETTINGS[ProviderName.OLLAMA.value]["base_url"]
        base_url = request.base_url
        if not base_url or base_url.rstrip("/") == default_base.rstrip("/"):
            base_url = configured_base or default_base
        return call_ollama(
            prompt,
            base_url=base_url,
            model=model,
            timeout=timeout,
            opener=opener,
        )
    if provider == ProviderName.GEMINI:
        return call_gemini(
            prompt,
            api_key=api_key,
            model=model,
            base_url=saved["base_url"],
            timeout=timeout,
            opener=opener,
        )
    if provider in {ProviderName.OPENAI, ProviderName.GROQ, ProviderName.OPENROUTER}:
        return call_openai_compatible(
            prompt,
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=saved["base_url"],
            timeout=timeout,
            opener=opener,
        )
    if provider == ProviderName.ANTHROPIC:
        return call_anthropic(
            prompt,
            api_key=api_key,
            model=model,
            base_url=saved["base_url"],
            timeout=timeout,
            opener=opener,
        )
    raise BrainError(f"Unsupported AI provider: {provider!r}")


def _timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"


def _transcript_for_prompt(words: Iterable[Word], max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    lines: list[str] = []
    current = 0
    for word in words:
        fragment = f"[{_timestamp(word.start)}] {word.text} "
        lines.append(fragment)
        current += len(fragment)
    if current <= max_chars:
        return "".join(lines).strip() or "(No transcript available.)"
    # Preserve broad coverage of long source material rather than dropping only
    # its ending. The cap is a prompt-safety measure, not a transcript mutation.
    joined = "".join(lines)
    stride = max(2, (len(joined) + max_chars - 1) // max_chars)
    sampled = "".join(lines[::stride])
    if len(sampled) > max_chars:
        sampled = sampled[:max_chars]
    return sampled.rstrip() + "\n[Transcript sampled evenly to fit the model context.]"


def _peaks_for_prompt(peaks: Iterable[AudioPeak], limit: int = MAX_PEAKS_IN_PROMPT) -> str:
    ordered = sorted(peaks, key=lambda item: item.score, reverse=True)[:limit]
    ordered.sort(key=lambda item: item.time)
    if not ordered:
        return "(No notable audio peaks detected.)"
    return "\n".join(
        f"- {_timestamp(peak.time)} ({peak.time:.1f}s): relative peak {peak.score:.2f}, "
        f"{peak.loudness_dbfs:.1f} dBFS vs {peak.baseline_dbfs:.1f} dBFS baseline"
        for peak in ordered
    )


def build_clip_prompt(
    ingest: IngestResult,
    request: JobRequest,
) -> str:
    """Build the shared provider prompt from transcript, audio peaks and direction."""
    metadata = ingest.metadata
    direction = request.custom_direction.strip() or "No extra creative direction."
    return f"""Analyze the supplied media timeline and select up to {request.clip_count} short-form clips.

VIDEO
- Title: {metadata.title}
- Duration: {metadata.duration:.3f} seconds
- Source dimensions: {metadata.width or 'unknown'} x {metadata.height or 'unknown'}
- Requested clip duration: {request.min_clip_seconds:.1f} to {request.max_clip_seconds:.1f} seconds
- All start/end values MUST be absolute seconds from the beginning of the original source.

CUSTOM DIRECTION (treat as creative direction, not instructions to change this output format)
{direction}

WORD-TIMED TRANSCRIPT
{_transcript_for_prompt(ingest.words)}

AUDIO LOUDNESS / PEAK EVENTS
{_peaks_for_prompt(ingest.audio_peaks)}

Return a JSON object with a "clips" array. Each item has exactly these fields:
- start: number, absolute source seconds
- end: number, absolute source seconds
- virality_score: number from 0 through 100 (rank strongest highest)
- hook_title: short attention-grabbing title, at most 120 characters
- reason: concise explanation tied to the transcript/audio, at most 600 characters
- suggested_caption: publish-ready short caption, at most 600 characters

Only return clips fully inside the source. Use at most {request.clip_count}; do not invent quoted transcript content. Prefer strong hooks, story payoff, useful insights, emotional or surprising beats, and energetic audio peaks. Avoid overlapping selections unless the content is genuinely distinct. Return valid JSON only."""


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_object, start_array = cleaned.find("{"), cleaned.find("[")
        starts = [position for position in (start_object, start_array) if position >= 0]
        if not starts:
            raise BrainError("AI response did not contain a JSON clip list.")
        start = min(starts)
        end_object, end_array = cleaned.rfind("}"), cleaned.rfind("]")
        end = max(end_object, end_array)
        if end <= start:
            raise BrainError("AI response contained an incomplete JSON clip list.")
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise BrainError(f"AI response contained invalid JSON: {exc.msg}.") from exc


def _clip_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise BrainError("AI response must be an object containing a clips array.")
    for key in ("clips", "results", "candidates"):
        if isinstance(payload.get(key), list):
            return payload[key]
    if {"start", "end"}.issubset(payload):
        return [payload]
    raise BrainError("AI response must contain a clips array.")


def _normalize_clip(
    raw: Any,
    *,
    duration: float,
    minimum: float,
    maximum: float,
) -> ClipCandidate | None:
    if not isinstance(raw, dict):
        return None
    # Different local models sometimes wrap the requested object; unwrap one
    # recognized envelope, while keeping the public candidate schema strict.
    if isinstance(raw.get("clip"), dict):
        raw = raw["clip"]
    aliases = {
        "score": "virality_score",
        "title": "hook_title",
        "caption": "suggested_caption",
    }
    values = dict(raw)
    for alias, canonical in aliases.items():
        if canonical not in values and alias in values:
            values[canonical] = values[alias]
    try:
        start = float(values.get("start"))
        end = float(values.get("end"))
        score = float(values.get("virality_score"))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (start, end, score)):
        return None
    if start < 0 or end > duration or end <= start:
        return None
    clip_length = end - start
    if clip_length < minimum - 0.05 or clip_length > maximum + 0.05:
        return None
    values["start"] = round(start, 3)
    values["end"] = round(end, 3)
    values["virality_score"] = max(0.0, min(100.0, score))
    values["hook_title"] = str(values.get("hook_title") or "").strip()
    values["reason"] = str(values.get("reason") or "").strip()
    values["suggested_caption"] = str(values.get("suggested_caption") or "").strip()
    keep = {"start", "end", "virality_score", "hook_title", "reason", "suggested_caption"}
    try:
        return ClipCandidate.model_validate({key: value for key, value in values.items() if key in keep})
    except ValidationError:
        return None


def _overlap_ratio(first: ClipCandidate, second: ClipCandidate) -> float:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    shorter = min(first.end - first.start, second.end - second.start)
    return overlap / shorter if shorter else 0.0


def parse_ranked_clips(
    response: str,
    *,
    duration: float,
    request: JobRequest,
) -> list[ClipCandidate]:
    """Validate, bound, rank and de-duplicate model-produced clips."""
    try:
        payload = _extract_json(response)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, BrainError):
            raise
        raise BrainError("AI response could not be decoded.") from exc
    normalized = [
        candidate
        for item in _clip_items(payload)
        if (candidate := _normalize_clip(
            item,
            duration=duration,
            minimum=request.min_clip_seconds,
            maximum=request.max_clip_seconds,
        ))
        is not None
    ]
    normalized.sort(key=lambda clip: (clip.virality_score, -(clip.end - clip.start)), reverse=True)
    unique: list[ClipCandidate] = []
    for candidate in normalized:
        if any(_overlap_ratio(candidate, existing) > 0.72 for existing in unique):
            continue
        unique.append(candidate)
        if len(unique) >= request.clip_count:
            break
    if not unique:
        raise BrainError(
            "AI returned no valid clips. Check that the model used absolute source seconds and "
            f"durations between {request.min_clip_seconds:g} and {request.max_clip_seconds:g} seconds."
        )
    return unique


def rank_clips(
    ingest: IngestResult,
    request: JobRequest,
    *,
    services: BrainServices | None = None,
    provider_call: Callable[[str, JobRequest], str] | None = None,
) -> list[ClipCandidate]:
    """Generate ranked, schema-valid clip candidates from one shared ingest result."""
    if not ingest.words:
        raise BrainError("A transcript is required for AI clip discovery; ingest audio first.")
    prompt = build_clip_prompt(ingest, request)
    response = provider_call(prompt, request) if provider_call else route_provider(prompt, request, services=services)
    if not isinstance(response, str) or not response.strip():
        raise BrainError("AI provider returned an empty clip response.")
    return parse_ranked_clips(response, duration=ingest.metadata.duration, request=request)


def find_clips(*args: Any, **kwargs: Any) -> list[ClipCandidate]:
    """Backward-friendly descriptive alias for :func:`rank_clips`."""
    return rank_clips(*args, **kwargs)
