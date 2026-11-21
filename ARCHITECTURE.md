# All-in-One AI Video Clipping Studio — Architecture

**Status:** Phase 1 architecture audit complete. This document is the implementation plan; no application code has been written yet.

## 1. Design contract

- **Windows-first, local-first:** Python 3.11+, FFmpeg/ffprobe, and a local React/Vite studio. Media, transcripts, job state, and exports stay on disk locally; network access is limited to retrieving a chosen remote source/captions and the explicitly selected LLM provider (Ollama defaults to localhost; Gemini is opt-in).
- **One engine, three surfaces:** the web API, `python cli.py`, and `mcp_server.py` are thin adapters over the same typed Python pipeline. They must not implement separate clipping logic.
- **Small, explicit dependency set:** use the requested FastAPI/Uvicorn/Pydantic, yt-dlp, FFmpeg, and optional vision/transcription packages. Load heavier MediaPipe/OpenCV/faster-whisper components only when their features are needed, pin versions with Windows wheels, and do not introduce a distributed queue, cloud database, or CUDA requirement.
- **Clean-room implementation:** the references below are shallow, MIT-licensed study clones. Their files and assets remain ignored and unshipped. Implementations will be original; no reference code, model weights, or fonts will be copied. Audit bundled third-party assets and dependency licenses separately before including any.

## 2. MIT reference audit

All four repositories reported SPDX `MIT` through GitHub's license endpoint and have an MIT `LICENSE` in the shallow checkout. Audited snapshots:

| Reference | Snapshot | Exact patterns studied |
|---|---|---|
| [`neonclipping`](https://github.com/iamajmeena/neonclipping) | `3a3da401c1bcc065e650dcb97a3e009a69a6a379` | `highlights.py`: per-second RMS/loudness jumps relative to a rolling local baseline, high-frequency voice/shout energy, optional motion/cut features, then non-overlapping peak candidates. `captions_yt.py`: parse YouTube JSON3 word timing into absolute source times. `app.py`: audio-only analysis for long VODs, `yt-dlp` range downloads for selected candidates, keyframe padding and audio matching to correct range offsets. |
| [`openshorts`](https://github.com/mutonby/openshorts) | `5da5b839d3231e3b2152250fa83aa17d40c91e64` | `main.py`: downscaled MediaPipe face detection, safe-zone crop tracking, confirmation of large target jumps and scene-cut resets. `split_layout.py` / `active_speaker.py`: detect stable co-present speakers and stack their crops rather than blindly splitting every two-person video. `camera_inset.py`: sample and stabilize webcam-inset bounds before composing a screen/webcam layout. `reframe_v2.py`: analyze in Python, then render with native FFmpeg filter graphs. `app.py`, `mcp_server.py`, `mcp_stdio.py`: REST and MCP/stdio entrypoints with job/progress concepts. |
| [`autoclip`](https://github.com/zhouxiaoka/autoclip) | `cd8e6bed18d87a81273d48bcd165751708ac715c` | `services/local_runner.py` / `quick_output_runner.py`: typed requests, progress callbacks, and a local pipeline reused by CLI and MCP. `services/platform_strategy.py`: one declarative source of truth for platform output geometry, fit/crop behavior, duration policy, caption/title styling and aliases. We will use this separation without importing its broader desktop/cloud/database stack. |
| [`yt-short-clipper`](https://github.com/jipraks/yt-short-clipper) | `f6d6d98f3bcee1b056cd49990a7f7176c55ac10b` | `highlight_finder.py` / `prompts.py`: per-video direction as bounded, sanitized user data, re-anchored after long transcripts and validated against the returned clips. `subtitle_downloader.py` / `session.py`: fetch YouTube subtitle data before video processing and separately parse the original word-timed track for word captions, enabling the fast path without local ASR. This reference requires a subtitle for its highlight session and skips word-caption burning when original word timing is unavailable; our app will add the requested faster-whisper fallback. |

The `_references/` directory is intentionally git-ignored. These snapshots are for architectural comparison only, not runtime dependencies.

## 3. Unified runtime shape

```text
React/Vite studio ──> FastAPI routes ─┐
python cli.py ────────────────────────┼──> core.pipeline.run(JobRequest, progress_sink)
MCP tools / stdio ────────────────────┘            │
                                  ingest → brain → reframe → captions → render
```

`core.pipeline.run` will be the sole orchestration entrypoint. It will accept a validated request (source, custom AI direction, provider/model, layout, caption preset, export preset) and emit structured stage/progress events. The local API will enqueue that same call in a small in-process job manager and expose status/events/results; the CLI and MCP adapters will call it directly. Job state and artifacts will use a per-job workspace under the user's Windows data directory, not the repository. No Celery or Linux-only process manager is needed.

Planned modules:

- `core/models.py`: Pydantic job, word, peak, clip, progress and export contracts; timestamps are absolute source seconds until a clip is rendered.
- `core/ingest.py`: validate local files and supported YouTube/Twitch URLs; probe media; prefer original YouTube word-timed captions; analyze audio in bounded-memory windows; use audio-only source acquisition for long remote videos; run faster-whisper CPU/int8 only when captions are absent or for local media. Preserve a graceful no-caption result when the user opts out of transcription.
- `core/brain.py`: provider interface for Ollama/OpenAI-compatible local endpoints and Gemini (using the leanest compatible HTTP client). Offer `qwen2.5:7b-instruct-q4_K_M` as a suggested Ollama model while allowing any configured local model. Combine transcript, audio peaks and a bounded per-video direction; require schema-valid JSON, finite/in-range times, sensible duration and de-overlapping candidates. Keep a deterministic activity-based fallback for provider errors.
- `core/reframe.py`: sampled-frame MediaPipe/OpenCV analysis and clamped, smoothed source-space crop trajectories; explicit Face-Track, Podcast Split, Gaming Webcam + Gameplay, and Fit-Blur modes. Low-confidence detection falls back safely rather than failing a job.
- `core/captions.py`: `.ass` generation from word times with a per-word active state and the four named styles: Hormozi Bold, Clean Minimal, Gaming Neon and Podcast Classic. Prefer available/local fonts and check FFmpeg's ASS/libass support.
- `core/render.py`: safely constructed FFmpeg argument lists; range-only yt-dlp acquisition where supported, precise trim after keyframe-aligned downloads, ASS burn-in, `-14 LUFS` audio normalization, encoder probing and fallback, and checked output metadata.
- `api/server.py`, `web/`, `cli.py`, `mcp_server.py`: transport/presentation only. The UI gets progress over SSE (or a simple job-status endpoint if the selected Vite setup makes SSE unsuitable); CLI output is scriptable JSON; MCP tools map to the same job request/status contracts.

## 4. Key synthesis decisions

1. **Audio-first discovery, not full-VOD download.** For a long YouTube source, obtain metadata and word captions first, fetch audio only for activity analysis, rank moments, then request only selected video sections with a small safety pad. Because section downloads can begin on an earlier keyframe, reconcile the actual clip offset before burning subtitles. If a source cannot serve byte/range sections, report that and use a documented full-source fallback. Use overlapping windows and bounded buffers for long media.
2. **Evidence-guided reframing.** Sample frames at scene-aware intervals, downscale only for detection, map detections back to source pixels, reject isolated jumps, smooth between observations, and reset at scene cuts. Split-screen requires two distinct faces co-visible for enough samples. Gaming mode requires a stable webcam inset; Fit-Blur remains the universal fallback. Audio peaks are a ranking signal, not a substitute for semantic context.
3. **Directed, validated AI output.** Treat a custom direction as untrusted data: cap and sanitize it, keep it distinct from the transcript/system rules, validate every returned clip with Pydantic and clamp/reject invalid ranges. Return the requested `start`, `end`, `virality_score`, `hook_title`, `reason`, and `suggested_caption` fields. The same router supports local Ollama (default/local preference) and opt-in Gemini; keys never enter source control or logs.
4. **Presets are data, not branches.** Keep selection/ranking separate from packaging. Define export presets for vertical Shorts/Reels/TikTok, landscape and square output, with dimensions, fit/crop/blur policy, safe areas, duration guidance and caption defaults. Do not add social-posting services in this project scope.
5. **Hardware selection is capability-based.** Probe FFmpeg encoders at runtime; prefer `h264_amf`, then usable `h264_nvenc`, then guaranteed `libx264`. If a detected hardware encoder fails on the actual render, retry with the next available encoder and include the final choice/failure in job diagnostics. Never require CUDA.
6. **Local operations and safety.** Use `pathlib`, `subprocess` argument arrays (never shell-interpolated user input), timeouts, explicit temp cleanup, typed errors and cancellation checks. Validate URL schemes/hosts and source ranges. Do not silently upload media; make the selected LLM/data flow visible in the UI.

## 5. UI, setup and completion checks

Use a lightweight React/Vite dashboard with the specified navy/ivory/gold/indigo/sky palette and **no green anywhere in the UI**; status indicators use sky blue or gold. The studio will expose URL/file intake, provider, layout, caption and export presets, custom direction, progress, clip review and downloads.

`setup.bat` will check Python 3.11+, FFmpeg/ffprobe and optional Ollama availability before installing the project environment; `run.bat` will start the local API/UI. `test_pipeline.py` will report encoder detection, probe Ollama connectivity without downloading a model, and render/probe a synthetic 15-second sample through FFmpeg. The final validation will also cover the pipeline's mocked provider/error paths and Windows-safe path/command construction.

**Build order after this plan is reviewed:** establish the Python/web structure and Windows checks; implement and test ingest; add the provider router and clip schema; implement/test the four reframe modes; add ASS captions; build encoder/range-aware rendering; connect FastAPI and the local studio; then expose the same pipeline via CLI and MCP and run the smoke suite.
