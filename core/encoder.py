"""Small FFmpeg H.264 encoder capability probe.

This module has no third-party Python dependencies and can be used by setup,
API runtime reporting, and the rendering pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from os import PathLike, fspath
from typing import Any

ENCODER_PRIORITY = ("h264_amf", "h264_nvenc", "libx264")
_HARDWARE_ENCODERS = frozenset(("h264_amf", "h264_nvenc"))


class EncoderProbeError(RuntimeError):
    """Raised when FFmpeg cannot be queried or no H.264 encoder is available."""


@dataclass(frozen=True)
class EncoderSelection:
    """Selected H.264 encoder and the reason it was chosen."""

    name: str
    hardware_accelerated: bool
    reason: str
    available: tuple[str, ...]


def parse_encoder_listing(output: str) -> frozenset[str]:
    """Return encoder names from the output of ``ffmpeg -encoders``.

    FFmpeg prints a six-character media/capability flag column before each
    encoder name. Only rows with that shape are considered, so heading and
    legend text cannot accidentally be treated as encoders.
    """
    found: set[str] = set()
    for line in output.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) < 2:
            continue
        flags, name = fields[0], fields[1]
        if len(flags) != 6 or flags[0] not in "VAS.":
            continue
        if any(not (char.isalnum() or char == ".") for char in flags):
            continue
        if name and all(char.isalnum() or char == "_" for char in name):
            found.add(name)
    return frozenset(found)


def select_encoder(available_encoders: Iterable[str]) -> EncoderSelection:
    """Select AMF first, then NVENC, with libx264 as the guaranteed CPU fallback."""
    available_set = {str(name).strip() for name in available_encoders}
    available = tuple(name for name in ENCODER_PRIORITY if name in available_set)

    for name in ENCODER_PRIORITY:
        if name not in available_set:
            continue
        if name == "h264_amf":
            reason = "FFmpeg exposes AMD AMF; selecting the preferred hardware encoder."
        elif name == "h264_nvenc":
            reason = "AMF is unavailable; selecting detected NVIDIA NVENC."
        else:
            reason = "Hardware H.264 encoders are unavailable; selecting CPU libx264."
        return EncoderSelection(
            name=name,
            hardware_accelerated=name in _HARDWARE_ENCODERS,
            reason=reason,
            available=available,
        )

    raise EncoderProbeError(
        "FFmpeg is installed, but none of the supported H.264 encoders were listed "
        "(h264_amf, h264_nvenc, libx264). Install an FFmpeg build with libx264 enabled."
    )


def probe_encoders(
    ffmpeg_executable: str | PathLike[str] = "ffmpeg",
    *,
    timeout_seconds: float = 10.0,
    run: Callable[..., Any] | None = None,
) -> frozenset[str]:
    """Run FFmpeg's encoder listing and return the available encoder names."""
    executable = fspath(ffmpeg_executable)
    runner = run or subprocess.run
    try:
        result = runner(
            [executable, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise EncoderProbeError(
            f"FFmpeg executable not found: {executable!r}. Install FFmpeg and add its bin folder to PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EncoderProbeError(
            f"FFmpeg encoder probe timed out after {timeout_seconds:g} seconds."
        ) from exc
    except OSError as exc:
        raise EncoderProbeError(f"Could not start FFmpeg at {executable!r}: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise EncoderProbeError(f"FFmpeg encoder probe failed (exit {result.returncode}): {detail[-600:]}")

    names = parse_encoder_listing(f"{result.stdout}\n{result.stderr}")
    if not names:
        raise EncoderProbeError("FFmpeg returned no parseable encoder entries.")
    return names


def _probe_one_hardware_frame(
    executable: str,
    encoder: str,
    *,
    timeout_seconds: float,
    run: Callable[..., Any],
) -> tuple[bool, str]:
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=0.1",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    ]
    try:
        result = run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"{encoder} one-frame runtime probe timed out after {timeout_seconds:g} seconds"
    except OSError as exc:
        return False, f"{encoder} one-frame runtime probe could not start: {exc}"
    if result.returncode == 0:
        return True, ""
    detail = (result.stderr or result.stdout or "no FFmpeg diagnostic output").strip()
    return False, f"{encoder} one-frame runtime probe failed: {detail[-500:]}"


def _detect_encoder_with_runner(
    executable: str,
    timeout_seconds: float,
    runner: Callable[..., Any],
) -> EncoderSelection:
    names = probe_encoders(executable, timeout_seconds=timeout_seconds, run=runner)
    supported = tuple(name for name in ENCODER_PRIORITY if name in names)
    failures: list[str] = []
    for name in ("h264_amf", "h264_nvenc"):
        if name not in names:
            continue
        passed, detail = _probe_one_hardware_frame(
            executable,
            name,
            timeout_seconds=timeout_seconds,
            run=runner,
        )
        if passed:
            provider = "AMD AMF" if name == "h264_amf" else "NVIDIA NVENC"
            return EncoderSelection(
                name=name,
                hardware_accelerated=True,
                reason=f"The {provider} one-frame runtime probe passed.",
                available=supported,
            )
        failures.append(detail)

    if "libx264" in names:
        reason = (
            "Hardware H.264 runtime probe failed (" + "; ".join(failures) + "); selecting CPU libx264."
            if failures
            else "Hardware H.264 encoders are unavailable; selecting CPU libx264."
        )
        return EncoderSelection(
            name="libx264",
            hardware_accelerated=False,
            reason=reason,
            available=supported,
        )
    if failures:
        raise EncoderProbeError(
            "No listed hardware encoder passed the one-frame runtime probe and libx264 is unavailable: "
            + "; ".join(failures)
        )
    return select_encoder(names)


@lru_cache(maxsize=8)
def _cached_detect_encoder(executable: str, timeout_seconds: float) -> EncoderSelection:
    return _detect_encoder_with_runner(executable, timeout_seconds, subprocess.run)


def detect_encoder(
    ffmpeg_executable: str | PathLike[str] = "ffmpeg",
    *,
    timeout_seconds: float = 10.0,
    run: Callable[..., Any] | None = None,
) -> EncoderSelection:
    """Probe hardware with one frame and cache the default runtime result per process."""
    executable = fspath(ffmpeg_executable)
    if run is None:
        return _cached_detect_encoder(executable, timeout_seconds)
    return _detect_encoder_with_runner(executable, timeout_seconds, run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe FFmpeg H.264 encoder availability.")
    parser.add_argument("--check", action="store_true", help="Run the encoder probe (the default action).")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path or command name for ffmpeg.exe.")
    args = parser.parse_args(argv)

    try:
        selection = detect_encoder(args.ffmpeg)
    except EncoderProbeError as exc:
        print(f"FFmpeg encoder check failed: {exc}", file=sys.stderr)
        return 2

    print(f"Selected encoder: {selection.name}")
    print(f"Hardware accelerated: {'yes' if selection.hardware_accelerated else 'no'}")
    print(f"Available supported H.264 encoders: {', '.join(selection.available)}")
    print(selection.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
