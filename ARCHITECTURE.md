# FrameForge — Architecture and Reference Audit

**Status (2026-11-21):** Phase 1 architecture audit and Phase 2 Windows/web scaffolding are complete. Phase 3's shared pipeline, five media modules, FastAPI/React UI, CLI/MCP adapters and mocked/live smoke harness are implemented. Automated checks pass in the Linux sandbox. FFmpeg is not installed here, so no real render, Windows setup run, or AMF hardware check has been performed; the user will run `setup.bat` and verify `h264_amf` on Windows.

## 1. Design contract

- **Windows-first, local-first:** CPython 3.11 x64, FFmpeg/ffprobe and a local React/Vite studio. Job snapshots, uploads, captions and exports use per-user local storage. The only intended network use is a chosen remote source/caption, the selected AI provider, or package/setup downloads.
- **One engine, three surfaces:** FastAPI, `python -m core.cli` / `python cli.py`, and MCP/stdio all call `core.pipeline.run_pipeline`. They do not implement separate clipping logic.
- **Small, Windows-compatible runtime:** pinned FastAPI/Uvicorn/Pydantic, yt-dlp, OpenCV headless, MediaPipe and faster-whisper. Transcription is CPU/int8; PyTorch and Docker are not hard requirements. No CUDA is required.
- **No silent media upload or social posting:** Gemini is an explicit cloud provider choice and receives the selection prompt/context, not a video upload. There are no social publishing integrations.
- **Clean-room implementation:** the MIT references below were studied for patterns only. Their source, model weights, fonts and other assets are not shipped or imported into the implementation.

## 2. MIT reference audit

All four repositories reported SPDX `MIT` through GitHub's license endpoint and have an MIT `LICENSE` in the shallow checkout. Audited snapshots:

| Reference | Snapshot | Patterns studied |
|---|---|---|
| [`neonclipping`](https://github.com/iamajmeena/neonclipping) | `3a3da401c1bcc065e650dcb97a3e009a69a6a379` | RMS/loudness changes relative to a rolling local baseline; YouTube JSON3 word timing; audio-only analysis with selected-range video acquisition. |
| [`openshorts`](https://github.com/mutonby/openshorts) | `5da5b839d3231e3b2152250fa83aa17d40c91e64` | Sampled/downscaled MediaPipe detection, safe crop tracking, Python analysis followed by FFmpeg filter graphs, and REST/MCP progress concepts. |
| [`autoclip`](https://github.com/zhouxiaoka/autoclip) | `cd8e6bed18d87a81273d48bcd165751708ac715c` | Typed requests, reusable local pipeline, progress callbacks and declarative layout/preset separation. |
| [`yt-short-clipper`](https://github.com/jipraks/yt-short-clipper) | `f6d6d98f3bcee1b056cd49990a7f7176c55ac10b` | Bounded per-video direction, schema/range validation, and fetching original word-timed captions before local ASR. |

`_references/` remains git-ignored and is not a runtime dependency.

## 3. Shared runtime shape

```text
React/Vite ──> FastAPI job API ─┐
python cli.py ──────────────────┼──> core.pipeline.run_pipeline(JobRequest, progress_sink)
MCP tools / stdio ──────────────┘           │
                               ingest → brain → reframe → captions → render
```

`core/models.py` defines the Pydantic contracts. `core/pipeline.py` owns orchestration, per-clip error handling and progress events. `core/job_store.py` atomically persists API job snapshots as JSON under the user's local data root; work directories are isolated by validated job ID. The UI polls `/api/jobs/{job_id}` and serves exports through guarded file routes. `web/vite.config.ts` proxies relative `/api` requests to the local API, so browser code does not call a browser-side localhost service.

### Implemented modules

- **`core/ingest.py`** — validates supported local video extensions and YouTube/Twitch hosts; probes local metadata with shell-free ffprobe arguments; prefers word-timed YouTube JSON3 captions; downloads audio-only for remote analysis; streams FFmpeg PCM in windows for RMS/loudness peak detection; and uses faster-whisper with `device="cpu"`, `compute_type="int8"`, and word timestamps when no caption fast-path is available.
- **`core/brain.py`** — routes to Ollama's native `/api/chat` API at `http://localhost:11434` or Google's Gemini `generateContent` API. It combines timed transcript, detected peaks and bounded custom direction, asks for a JSON clip envelope, validates every candidate, rejects out-of-source/improper-duration ranges, ranks by score and drops near-duplicate intervals. Gemini is opt-in through `GEMINI_API_KEY` or an explicit in-memory request value; the secret is excluded from serialized job requests.
- **`core/reframe.py`** — lazily imports MediaPipe/OpenCV and samples frames for a smoothed face trajectory. It generates filter graphs for Auto Face-Track 9:16, Podcast Split-Screen, Gaming Webcam + Gameplay and Fit-Blur. Missing face tracking falls back to a safe center crop. Podcast Split-Screen currently stacks the source's left and right halves; automatic active-speaker identification is not implemented. Gaming uses face detections to estimate a webcam crop; unusual inset geometry may need manual source/layout review.
- **`core/captions.py`** — writes clip-relative ASS events using word times. The entire phrase remains visible while the active word changes color. Four presets are provided: Hormozi Bold, Clean Minimal, Gaming Neon and Podcast Classic; a suggested-caption fallback is available when there are no timed words.
- **`core/render.py`** — tries yt-dlp's selected time-range download for each remote clip and falls back to a full-source download if the extractor cannot provide a range. Local cuts use FFmpeg seek/duration options. ASS is burned in; audio uses one-pass `loudnorm=I=-14:TP=-1.5:LRA=11`; encoder discovery prefers AMF, then detected NVENC, then libx264, and an actual hardware render failure retries with libx264 when listed. All FFmpeg calls use argument arrays and timeouts.
- **`api/server.py` + `web/`** — multipart file upload or supported URL entry; Ollama/Gemini selection; layout/caption/export controls; custom direction; polling progress; local status/runtime display; and preview/download cards. The palette uses deep navy, warm ivory, muted gold, indigo and sky blue only; there is no green UI state color.
- **`core/cli.py`, `core/mcp_server.py`** — terminal and MCP adapters for the same pipeline. MCP is exposed over stdio and is launched with `python -m core.mcp_server` (or `frameforge-mcp`).
- **`test_pipeline.py`** — deterministic mocked mode by default and a `--live` mode. Mocked checks exercise the full typed pipeline without real FFmpeg, an LLM, network access or ASR weights. Live mode probes encoders, checks Ollama `/api/tags` without a model pull, creates/renders/probes a 15-second synthetic A/V sample, and can additionally process `--source` end-to-end.

## 4. Operating decisions and boundaries

1. **Audio-first remote analysis.** Ingest fetches remote metadata/captions and an audio-only track. The renderer asks yt-dlp for only each selected video interval when supported. If the extractor cannot do that, the fallback is a full download per selected clip; the user is warned in the job result. The sandbox tests verify the range-attempt/fallback branches, but the actual offset/keyframe behavior depends on the source extractor and must be checked with real media.
2. **Evidence-guided but conservative reframing.** Face tracking samples video and clamps/smooths crop coordinates. Face-detection failure degrades to center crop. Podcast and Gaming are practical layout heuristics rather than semantic speaker/scene understanding; Fit-Blur is the universal no-detection option.
3. **Validated AI output.** Custom direction is capped and kept separate from system/output rules. Returned time ranges must be absolute source seconds, inside source duration, and within configured clip length. The selected provider receives the transcript and audio-peak context; it does not receive the source video.
4. **Encoder selection is capability-based, not a hardware claim.** FFmpeg's listing is probed at runtime. `h264_amf` is preferred; `h264_nvenc` can be selected if AMF is absent; `libx264` is the CPU fallback. A sandbox unit test of selection logic does not prove Windows FFmpeg installation, AMD driver availability or successful AMF encoding.
5. **Local operations.** No user input is interpolated into a shell command. URL schemes/hosts, job IDs, upload extensions, output filenames, time ranges and provider response shapes are validated. API uploads default to a 16 GB limit and job status is persisted locally.

## 5. Setup and verification

`setup.bat` requires CPython 3.11 x64, FFmpeg/ffprobe, Node/npm and installs the pinned runtime (including MCP 2.3.0) in `.venv`, then installs the web lockfile and probes the local encoder listing. The Windows setup aligns and replaces MediaPipe's GUI OpenCV wheel with the headless build. `run.bat` starts the local API/UI.

Sandbox checks run without a Windows or FFmpeg claim:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check core api cli.py mcp_server.py test_pipeline.py tests
.venv\Scripts\python.exe test_pipeline.py --mocked
```

On Windows, run `setup.bat` and verify the actual encoder output. To do the real local smoke check and optional full pipeline run:

```bat
.venv\Scripts\python.exe test_pipeline.py --live
.venv\Scripts\python.exe test_pipeline.py --live --source "C:\Videos\talk.mp4" --provider ollama
```

No CUDA/PyTorch/Docker, external font download, cloud database or social-posting integration is needed.
