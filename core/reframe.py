"""MediaPipe-assisted shot analysis and FFmpeg graph construction for vertical clips."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.models import CropKeyframe, ExportPreset, FaceBox, ReframeMode, ReframePlan

OUTPUT_SIZES = {
    ExportPreset.SHORT_VERTICAL: (1080, 1920),
    ExportPreset.SQUARE: (1080, 1080),
    ExportPreset.LANDSCAPE: (1920, 1080),
}


class ReframeError(RuntimeError):
    """Raised when a video frame cannot be sampled for reframing."""


@dataclass
class ReframeServices:
    """Optional dependency injection for frame sampling and face detection."""

    cv2_module: Any | None = None
    face_detector_factory: Callable[..., Any] | None = None


def _even(value: float, minimum: int = 2) -> int:
    rounded = max(minimum, int(value))
    return rounded - (rounded % 2)


def _crop_dimensions(source_width: int, source_height: int, aspect: float) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0 or aspect <= 0:
        raise ValueError("source dimensions and aspect ratio must be positive")
    if source_width / source_height >= aspect:
        height = _even(source_height)
        width = min(_even(source_width), _even(height * aspect))
    else:
        width = _even(source_width)
        height = min(_even(source_height), _even(width / aspect))
    return min(width, _even(source_width)), min(height, _even(source_height))


def _filter_number(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".") or "0"


def _interpolate_expression(keyframes: list[CropKeyframe], axis: str) -> str:
    if not keyframes:
        return "0"
    ordered = sorted(keyframes, key=lambda item: item.time)
    expression = str(getattr(ordered[-1], axis))
    for left, right in reversed(list(itertools.pairwise(ordered))):
        start = _filter_number(left.time)
        end = _filter_number(right.time)
        start_value = getattr(left, axis)
        end_value = getattr(right, axis)
        span = max(1e-6, right.time - left.time)
        interpolated = (
            f"{start_value}+({end_value}-{start_value})*(t-{start})/{_filter_number(span)}"
        )
        expression = f"if(lt(t,{end}),{interpolated},{expression})"
    first = ordered[0]
    expression = f"if(lt(t,{_filter_number(first.time)}),{getattr(first, axis)},{expression})"
    return f"trunc(({expression})/2)*2"


def _default_detector_factory(**kwargs: Any) -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - exercised on the target setup
        raise ReframeError("MediaPipe is not installed. Run setup.bat to install the runtime.") from exc
    try:
        return mp.solutions.face_detection.FaceDetection(**kwargs)
    except AttributeError as exc:  # pragma: no cover - depends on future MediaPipe packaging
        raise ReframeError("This MediaPipe build does not expose the face-detection solution API.") from exc


def _cv2(services: ReframeServices | None) -> Any:
    if services is not None and services.cv2_module is not None:
        return services.cv2_module
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised on the target setup
        raise ReframeError("OpenCV is not installed. Run setup.bat to install the runtime.") from exc
    return cv2


def _relative_detection_boxes(results: Any, frame_width: int, frame_height: int) -> list[FaceBox]:
    boxes: list[FaceBox] = []
    for detection in getattr(results, "detections", None) or []:
        try:
            relative = detection.location_data.relative_bounding_box
            x = max(0.0, min(frame_width - 1.0, float(relative.xmin) * frame_width))
            y = max(0.0, min(frame_height - 1.0, float(relative.ymin) * frame_height))
            width = max(1.0, min(frame_width - x, float(relative.width) * frame_width))
            height = max(1.0, min(frame_height - y, float(relative.height) * frame_height))
            score_values = getattr(detection, "score", None) or [1.0]
            confidence = max(0.0, min(1.0, float(score_values[0])))
            boxes.append(FaceBox(x=x, y=y, width=width, height=height, confidence=confidence))
        except (AttributeError, TypeError, ValueError, IndexError):
            continue
    return boxes


def _select_face(boxes: list[FaceBox], previous: FaceBox | None = None) -> FaceBox | None:
    if not boxes:
        return None
    if previous is None:
        return max(boxes, key=lambda box: (box.width * box.height) * (0.5 + box.confidence))
    previous_center = (previous.x + previous.width / 2, previous.y + previous.height / 2)
    def score(box: FaceBox) -> float:
        center = (box.x + box.width / 2, box.y + box.height / 2)
        distance = math.dist(center, previous_center)
        area = box.width * box.height
        return area * (0.4 + box.confidence) / (1.0 + distance / 250.0)
    return max(boxes, key=score)


def sample_face_track(
    video_path: str | Path,
    *,
    source_width: int,
    source_height: int,
    clip_start: float,
    duration: float,
    services: ReframeServices | None = None,
    sample_interval: float = 0.5,
    max_samples: int = 120,
) -> list[tuple[float, FaceBox]]:
    """Sample the requested clip and return face centers with clip-relative times."""
    cv2_module = _cv2(services)
    detector_factory = (services.face_detector_factory if services else None) or _default_detector_factory
    path = Path(video_path).expanduser().resolve()
    capture = cv2_module.VideoCapture(str(path))
    if not capture.isOpened():
        raise ReframeError(f"OpenCV could not open the source video: {path}")
    detector = None
    selected: list[tuple[float, FaceBox | None]] = []
    previous: FaceBox | None = None
    try:
        try:
            detector = detector_factory(model_selection=1 if source_width > 1280 else 0, min_detection_confidence=0.5)
        except TypeError:
            detector = detector_factory()
        count = max(1, min(max_samples, math.ceil(duration / max(0.1, sample_interval)) + 1))
        sample_times = sorted({min(duration, index * duration / max(1, count - 1)) for index in range(count)})
        for relative_time in sample_times:
            capture.set(cv2_module.CAP_PROP_POS_MSEC, (clip_start + relative_time) * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                selected.append((relative_time, None))
                continue
            frame_height, frame_width = frame.shape[:2]
            rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = detector.process(rgb)
            face = _select_face(_relative_detection_boxes(results, frame_width, frame_height), previous)
            if face is not None:
                selected.append((relative_time, face))
                previous = face
            else:
                selected.append((relative_time, None))
    finally:
        try:
            if detector is not None:
                detector.close()
        finally:
            capture.release()

    if not any(face is not None for _time, face in selected):
        return []
    # Fill undetected frames from the nearest real observation to avoid sudden
    # jumps when the detector briefly misses a face.
    populated = [(time, face) for time, face in selected if face is not None]
    complete: list[tuple[float, FaceBox]] = []
    for time, face in selected:
        if face is None:
            nearest = min(populated, key=lambda item: abs(item[0] - time))[1]
            complete.append((time, nearest))
        else:
            complete.append((time, face))

    # Smooth box centers while preserving a stable, constant crop size.
    alpha = 0.38
    smoothed: list[tuple[float, FaceBox]] = []
    center_x = center_y = 0.0
    for index, (time, face) in enumerate(complete):
        current_x = face.x + face.width / 2
        current_y = face.y + face.height / 2
        if index == 0:
            center_x, center_y = current_x, current_y
        else:
            center_x += alpha * (current_x - center_x)
            center_y += alpha * (current_y - center_y)
        smoothed.append(
            (
                time,
                FaceBox(
                    x=center_x - face.width / 2,
                    y=center_y - face.height / 2,
                    width=face.width,
                    height=face.height,
                    confidence=face.confidence,
                ),
            )
        )
    return smoothed


def _centered_crop_filter(source_width: int, source_height: int, out_width: int, out_height: int) -> str:
    crop_width, crop_height = _crop_dimensions(source_width, source_height, out_width / out_height)
    return (
        f"crop={crop_width}:{crop_height}:(iw-{crop_width})/2:(ih-{crop_height})/2,"
        f"scale={out_width}:{out_height}:flags=lanczos,setsar=1"
    )


def _face_track_plan(
    video_path: str | Path,
    *,
    source_width: int,
    source_height: int,
    clip_start: float,
    duration: float,
    services: ReframeServices | None,
    output_width: int,
    output_height: int,
) -> tuple[str, list[CropKeyframe], list[str]]:
    crop_width, crop_height = _crop_dimensions(source_width, source_height, 9 / 16)
    warnings: list[str] = []
    faces: list[tuple[float, FaceBox]] = []
    try:
        faces = sample_face_track(
            video_path,
            source_width=source_width,
            source_height=source_height,
            clip_start=clip_start,
            duration=duration,
            services=services,
        )
    except ReframeError as exc:
        warnings.append(f"Face tracking unavailable; using a centered crop. {exc}")
    if not faces:
        warnings.append("No face was detected; using a centered 9:16 crop.")
        crop_filter = _centered_crop_filter(source_width, source_height, output_width, output_height)
        return f"[0:v]{crop_filter}[vout]", [], warnings

    keyframes: list[CropKeyframe] = []
    max_x = max(0, source_width - crop_width)
    max_y = max(0, source_height - crop_height)
    for time, face in faces:
        center_x = face.x + face.width / 2
        center_y = face.y + face.height / 2
        left = max(0, min(max_x, round(center_x - crop_width / 2)))
        # Bias the portrait framing upward so eyes and captions remain visible.
        top = max(0, min(max_y, round(center_y - crop_height * 0.39)))
        keyframes.append(
            CropKeyframe(
                time=round(time, 3),
                x=_even(left, minimum=0),
                y=_even(top, minimum=0),
                width=crop_width,
                height=crop_height,
            )
        )
    x_expression = _interpolate_expression(keyframes, "x")
    y_expression = _interpolate_expression(keyframes, "y")
    filters = (
        f"crop={crop_width}:{crop_height}:x='{x_expression}':y='{y_expression}',"
        f"scale={output_width}:{output_height}:flags=lanczos,setsar=1"
    )
    return f"[0:v]{filters}[vout]", keyframes, warnings


def _podcast_split_filter(source_width: int, source_height: int, out_width: int, out_height: int) -> str:
    half_width = _even(source_width // 2)
    crop_height = _even(source_height)
    right_x = _even(source_width - half_width, minimum=0)
    panel_height = _even(out_height // 2)
    panel = (
        f"crop={half_width}:{crop_height}:x=0:y=0,"
        f"scale={out_width}:{panel_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={out_width}:{panel_height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )
    other = (
        f"crop={half_width}:{crop_height}:x={right_x}:y=0,"
        f"scale={out_width}:{panel_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={out_width}:{panel_height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )
    return (
        f"[0:v]split=2[left_in][right_in];"
        f"[left_in]{panel}[left];[right_in]{other}[right];"
        f"[left][right]vstack=inputs=2,scale={out_width}:{out_height},setsar=1[vout]"
    )


def _fit_blur_filter(out_width: int, out_height: int) -> str:
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={out_width}:{out_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={out_width}:{out_height},boxblur=24:2[blurred];"
        f"[fg]scale={out_width}:{out_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"setsar=1[fit];[blurred][fit]overlay=(W-w)/2:(H-h)/2,setsar=1[vout]"
    )


def _webcam_region(faces: list[tuple[float, FaceBox]], source_width: int, source_height: int) -> FaceBox | None:
    if not faces:
        return None
    face = max(faces, key=lambda item: item[1].confidence * item[1].width * item[1].height)[1]
    desired_height = max(face.height * 3.2, source_height * 0.12)
    desired_width = max(face.width * 4.0, desired_height * (16 / 9))
    aspect = 16 / 9
    desired_width = min(source_width, desired_height * aspect)
    desired_height = desired_width / aspect
    desired_height = min(source_height, desired_height)
    center_x = face.x + face.width / 2
    center_y = face.y + face.height / 2
    x = max(0.0, min(source_width - desired_width, center_x - desired_width / 2))
    y = max(0.0, min(source_height - desired_height, center_y - desired_height / 2))
    return FaceBox(x=x, y=y, width=desired_width, height=desired_height, confidence=face.confidence)


def _gaming_filter(
    source_width: int,
    source_height: int,
    out_width: int,
    out_height: int,
    webcam: FaceBox | None,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    base = _centered_crop_filter(source_width, source_height, out_width, out_height)
    if webcam is None:
        warnings.append("No webcam face was detected; exporting the gameplay with a centered vertical crop.")
        return f"[0:v]{base}[vout]", warnings
    overlay_width = _even(out_width * 0.34)
    overlay_height = _even(overlay_width / (16 / 9))
    overlay_x = _even(out_width - overlay_width - out_width * 0.035)
    overlay_y = _even(out_height * 0.58)
    x, y = _even(webcam.x, minimum=0), _even(webcam.y, minimum=0)
    width, height = _even(webcam.width), _even(webcam.height)
    graph = (
        f"[0:v]split=2[game][cam];[game]{base}[background];"
        f"[cam]crop={width}:{height}:x={x}:y={y},"
        f"scale={overlay_width}:{overlay_height}:flags=lanczos,setsar=1[webcam];"
        f"[background][webcam]overlay=x={overlay_x}:y={overlay_y}:shortest=1,"
        f"drawbox=x={overlay_x}:y={overlay_y}:w={overlay_width}:h={overlay_height}:"
        f"color=white@0.9:t=4,setsar=1[vout]"
    )
    return graph, warnings


def plan_reframe(
    video_path: str | Path,
    *,
    source_width: int,
    source_height: int,
    clip_start: float = 0.0,
    duration: float,
    mode: ReframeMode | str = ReframeMode.FACE_TRACK,
    export_preset: ExportPreset | str = ExportPreset.SHORT_VERTICAL,
    services: ReframeServices | None = None,
) -> ReframePlan:
    """Build a dynamic crop/composite graph for the requested output layout."""
    try:
        selected_mode = ReframeMode(mode)
    except ValueError as exc:
        raise ReframeError(f"Unsupported reframe mode: {mode!r}") from exc
    try:
        preset = ExportPreset(export_preset)
    except ValueError as exc:
        raise ReframeError(f"Unsupported export preset: {export_preset!r}") from exc
    if source_width <= 0 or source_height <= 0:
        raise ReframeError("Source width and height must be positive.")
    if clip_start < 0 or duration <= 0:
        raise ReframeError("Clip start must be non-negative and duration must be positive.")
    output_width, output_height = OUTPUT_SIZES[preset]
    warnings: list[str] = []
    keyframes: list[CropKeyframe] = []
    webcam: FaceBox | None = None

    if selected_mode == ReframeMode.FACE_TRACK:
        # Face-track is explicitly a 9:16 short-form mode regardless of the
        # user's alternative still-image export presets.
        output_width, output_height = OUTPUT_SIZES[ExportPreset.SHORT_VERTICAL]
        graph, keyframes, warnings = _face_track_plan(
            video_path,
            source_width=source_width,
            source_height=source_height,
            clip_start=clip_start,
            duration=duration,
            services=services,
            output_width=output_width,
            output_height=output_height,
        )
        if preset != ExportPreset.SHORT_VERTICAL:
            warnings.append("Auto Face-Track always exports 9:16; the selected alternate preset was ignored.")
    elif selected_mode == ReframeMode.PODCAST_SPLIT:
        graph = _podcast_split_filter(source_width, source_height, output_width, output_height)
    elif selected_mode == ReframeMode.GAMING:
        faces: list[tuple[float, FaceBox]] = []
        try:
            faces = sample_face_track(
                video_path,
                source_width=source_width,
                source_height=source_height,
                clip_start=clip_start,
                duration=duration,
                services=services,
                sample_interval=1.0,
                max_samples=30,
            )
        except ReframeError as exc:
            warnings.append(f"Webcam detection unavailable: {exc}")
        webcam = _webcam_region(faces, source_width, source_height)
        graph, game_warnings = _gaming_filter(source_width, source_height, output_width, output_height, webcam)
        warnings.extend(game_warnings)
    else:
        graph = _fit_blur_filter(output_width, output_height)

    return ReframePlan(
        requested_mode=selected_mode,
        mode=selected_mode,
        source_width=source_width,
        source_height=source_height,
        output_width=output_width,
        output_height=output_height,
        filter_complex=graph,
        keyframes=keyframes,
        webcam_box=webcam,
        warnings=warnings,
    )


# Readable aliases used by CLI/API integrations.
build_reframe_plan = plan_reframe
create_reframe_plan = plan_reframe
