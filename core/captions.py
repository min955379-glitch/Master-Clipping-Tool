"""ASS caption generation with synchronized active-word coloring and local presets."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from core.models import CaptionPreset, Word


@dataclass(frozen=True)
class CaptionStyle:
    name: CaptionPreset
    font_name: str
    font_size: int
    primary_color: str
    highlight_color: str
    outline_color: str = "&H00000000"
    back_color: str = "&H90000000"
    bold: int = -1
    outline: int = 4
    shadow: int = 1
    margin_fraction: float = 0.12
    max_words: int = 5
    max_chars: int = 40


CAPTION_STYLES: dict[CaptionPreset, CaptionStyle] = {
    CaptionPreset.HORMOZI_BOLD: CaptionStyle(
        name=CaptionPreset.HORMOZI_BOLD,
        font_name="Arial Black",
        font_size=68,
        primary_color="&H00FFFFFF",
        highlight_color="&H0055ADD6",  # warm gold #D6AD55 in ASS BGR order
        outline_color="&H00000000",
        back_color="&H88000000",
        bold=-1,
        outline=5,
        shadow=2,
        margin_fraction=0.14,
        max_words=4,
        max_chars=34,
    ),
    CaptionPreset.CLEAN_MINIMAL: CaptionStyle(
        name=CaptionPreset.CLEAN_MINIMAL,
        font_name="Arial",
        font_size=52,
        primary_color="&H00F7F3E9",  # warm ivory
        highlight_color="&H00F8BD38",  # sky blue #38BDF8
        outline_color="&H80071A35",
        back_color="&H00000000",
        bold=0,
        outline=2,
        shadow=0,
        margin_fraction=0.1,
        max_words=6,
        max_chars=48,
    ),
    CaptionPreset.GAMING_NEON: CaptionStyle(
        name=CaptionPreset.GAMING_NEON,
        font_name="Arial Black",
        font_size=62,
        primary_color="&H00FFFFFF",
        highlight_color="&H00F8BD38",  # sky blue #38BDF8
        outline_color="&H00631606",  # deep indigo outline
        back_color="&H90071A35",
        bold=-1,
        outline=4,
        shadow=2,
        margin_fraction=0.12,
        max_words=5,
        max_chars=42,
    ),
    CaptionPreset.PODCAST_CLASSIC: CaptionStyle(
        name=CaptionPreset.PODCAST_CLASSIC,
        font_name="Arial",
        font_size=54,
        primary_color="&H00F7F3E9",
        highlight_color="&H0055ADD6",
        outline_color="&H00071A35",
        back_color="&H78071A35",
        bold=0,
        outline=3,
        shadow=1,
        margin_fraction=0.1,
        max_words=7,
        max_chars=54,
    ),
}


class CaptionError(ValueError):
    """Raised when caption settings or output range are invalid."""


def _style(preset: CaptionPreset | str) -> CaptionStyle:
    try:
        key = preset if isinstance(preset, CaptionPreset) else CaptionPreset(preset)
    except ValueError as exc:
        raise CaptionError(f"Unknown caption preset: {preset!r}") from exc
    return CAPTION_STYLES[key]


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _safe_word(text: str) -> str:
    # ASS uses braces for override tags. Strip control characters and neutralize
    # braces so transcript text cannot inject formatting or extra events.
    cleaned = "".join(character for character in text if character >= " " and character not in "\r\n")
    cleaned = cleaned.replace("{", "〔").replace("}", "〕").replace("\\", "／")
    return re.sub(r"\s+", " ", cleaned).strip()


def _clip_words(words: Iterable[Word], clip_start: float, clip_end: float) -> list[Word]:
    selected: list[Word] = []
    for word in words:
        start = max(clip_start, word.start)
        end = min(clip_end, word.end)
        if end <= start:
            continue
        text = _safe_word(word.text)
        if text:
            selected.append(Word(start=start, end=end, text=text))
    return sorted(selected, key=lambda word: (word.start, word.end))


def _groups(words: list[Word], style: CaptionStyle) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    text_length = 0
    for word in words:
        next_length = text_length + len(word.text) + (1 if current else 0)
        pause = current and word.start - current[-1].end > 0.85
        duration = word.end - current[0].start if current else 0.0
        should_break = bool(
            current
            and (
                len(current) >= style.max_words
                or next_length > style.max_chars
                or pause
                or duration > 3.8
                or re.search(r"[.!?…][\"'’”)]*$", current[-1].text)
            )
        )
        if should_break:
            groups.append(current)
            current = []
            text_length = 0
        current.append(word)
        text_length += len(word.text) + (1 if len(current) > 1 else 0)
    if current:
        groups.append(current)
    return groups


def _style_line(style: CaptionStyle, output_width: int, output_height: int) -> str:
    scaled_font_size = round(style.font_size * output_height / 1920)
    scaled_outline = max(1, round(style.outline * output_height / 1920))
    scaled_shadow = max(0, round(style.shadow * output_height / 1920))
    margin = round(output_height * style.margin_fraction)
    return ",".join(
        (
            "Default",
            style.font_name,
            str(max(20, scaled_font_size)),
            style.primary_color,
            style.primary_color,
            style.outline_color,
            style.back_color,
            str(style.bold),
            "0",
            "0",
            "0",
            "100",
            "100",
            "0",
            "0",
            "1",
            str(scaled_outline),
            str(scaled_shadow),
            "2",
            str(round(output_width * 0.05)),
            str(round(output_width * 0.05)),
            str(margin),
            "1",
        )
    )


def _format_group(group: list[Word], active_index: int, style: CaptionStyle) -> str:
    parts: list[str] = []
    for index, word in enumerate(group):
        color = style.highlight_color if index == active_index else style.primary_color
        token = word.text
        if index < len(group) - 1 and re.search(r"[,;:]$", token):
            token += "\\N"
            separator = ""
        else:
            separator = " " if index < len(group) - 1 else ""
        parts.append(f"{{\\c{color}&}}{token}{{\\c{style.primary_color}&}}{separator}")
    return "".join(parts).rstrip()


def generate_ass(
    words: Iterable[Word],
    *,
    clip_start: float,
    clip_end: float,
    preset: CaptionPreset | str = CaptionPreset.HORMOZI_BOLD,
    output_width: int = 1080,
    output_height: int = 1920,
    fallback_text: str = "",
) -> str:
    """Return ASS subtitles with a separate synchronized event per active word."""
    if clip_start < 0 or clip_end <= clip_start:
        raise CaptionError("Caption clip range must be positive and ordered.")
    if output_width <= 0 or output_height <= 0:
        raise CaptionError("Caption output dimensions must be positive.")
    style = _style(preset)
    clipped = _clip_words(words, clip_start, clip_end)
    if not clipped and fallback_text.strip():
        fallback = [token for token in re.split(r"\s+", fallback_text.strip()) if token]
        duration = min(clip_end - clip_start, max(2.0, min(5.0, len(fallback) * 0.32)))
        span = duration / max(1, len(fallback))
        clipped = [
            Word(
                start=clip_start + index * span,
                end=min(clip_end, clip_start + (index + 1) * span),
                text=_safe_word(token),
            )
            for index, token in enumerate(fallback)
            if _safe_word(token)
        ]
    groups = _groups(clipped, style)
    lines = [
        "[Script Info]",
        "Title: FrameForge captions",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {output_width}",
        f"PlayResY: {output_height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: {_style_line(style, output_width, output_height)}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for group in groups:
        for index, word in enumerate(group):
            relative_start = max(0.0, word.start - clip_start)
            relative_end = max(relative_start + 0.04, word.end - clip_start)
            if index + 1 < len(group):
                next_start = max(0.0, group[index + 1].start - clip_start)
                if next_start > relative_start and next_start - relative_start <= 1.25:
                    relative_end = max(relative_end, next_start)
            relative_end = min(clip_end - clip_start, relative_end)
            if relative_end <= relative_start:
                continue
            text = _format_group(group, index, style)
            lines.append(
                "Dialogue: 0,"
                f"{_ass_time(relative_start)},{_ass_time(relative_end)},Default,,0,0,0,,{text}"
            )
    return "\n".join(lines) + "\n"


def write_ass(
    output_path: str | Path,
    words: Iterable[Word],
    *,
    clip_start: float,
    clip_end: float,
    preset: CaptionPreset | str = CaptionPreset.HORMOZI_BOLD,
    output_width: int = 1080,
    output_height: int = 1920,
    fallback_text: str = "",
) -> Path:
    """Create the output directory and write an UTF-8 ASS caption file."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_ass(
        words,
        clip_start=clip_start,
        clip_end=clip_end,
        preset=preset,
        output_width=output_width,
        output_height=output_height,
        fallback_text=fallback_text,
    )
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def create_ass(*args: object, **kwargs: object) -> str:
    """Descriptive alias for callers that only need in-memory ASS content."""
    return generate_ass(*args, **kwargs)  # type: ignore[arg-type]
