# FrameForge · All-in-One AI Video Clipping Studio

A Windows-first, local-first studio for finding, reframing, captioning and exporting short clips from local videos or supported YouTube/Twitch URLs. The web UI, CLI and MCP adapter all call the same typed Python pipeline. No social posting, CUDA, PyTorch or Docker is required.

## Status

The shared local pipeline, multi-provider routing, persistent local provider settings, per-clip deletion, four reframing layouts, ASS captions, FFmpeg rendering, API/UI, CLI, MCP and mocked/live smoke harness are implemented. `setup.bat` builds the production web UI, and the local API serves it on port 8000 with SPA route fallback. The Linux sandbox has no FFmpeg and is not a Windows host, so real rendering and Windows AMF/NVENC acceleration still need verification on a machine with the relevant hardware and drivers.

## Windows prerequisites

- **Windows 10/11 x64**
- **CPython 3.11 x64** (the project intentionally pins `>=3.11,<3.12` for the selected MediaPipe and faster-whisper wheels)
- **Node.js 22.12+** or 20.19+ with npm, for the Vite studio
- **FFmpeg and ffprobe** on `PATH`; include `h264_amf` when AMD AMF is available and `libx264` for CPU fallback. Detected NVENC may be selected when AMF is not listed.
- **Ollama** is optional for local AI. Install it separately, start its service at `http://localhost:11434`, and pull a model such as `qwen2.5:7b` before selecting it.
- **Cloud AI providers** are optional. Configure Gemini, OpenAI, Anthropic, Groq or OpenRouter in the Studio's Settings panel; Ollama also supports a custom local base URL. Provider keys are saved to a private per-user `settings.json`, are never returned by the API, and can instead be supplied through a local `.env` file.

## Windows 10/11 quick start

Install **CPython 3.11 x64** and **Node.js 22.12+ (or 20.19+)** first. Open PowerShell in the repository root (the folder containing `setup.bat`) and install FFmpeg with WinGet:

```powershell
winget install -e --id Gyan.FFmpeg
```

Close and reopen PowerShell so the updated `PATH` is loaded. Then, from the repository root, run setup, verify the encoder probe, run the real local smoke test, and launch the studio:

```powershell
.\setup.bat
.\.venv\Scripts\python.exe -m core.encoder --check
.\.venv\Scripts\python.exe test_pipeline.py --live
.\run.bat
```

For AMF verification, check that the encoder probe prints `Selected encoder: h264_amf` and lists `h264_amf` among the supported encoders. If it selects `h264_nvenc` or `libx264` instead, FFmpeg or the installed graphics driver is not exposing AMF; `libx264` is the CPU fallback. The live smoke test creates and renders a synthetic 15-second clip, then probes the exported file. Its Ollama connectivity result is informational; a local model is only needed to process your own source. `setup.bat` installs the pinned Python and web dependencies and reports the encoder selection; it may pause for a keypress when it completes.

`setup.bat` builds the production UI in `web/dist`. `run.bat` launches the loopback-only FastAPI service, which serves both the API and the studio on port 8000, then opens the browser. Choose a local video or paste a YouTube/Twitch URL, select the provider/layout/caption preset and add optional custom direction. The UI polls job progress, shows in-browser previews, and lets you download or delete each generated clip.

Uploaded source media, job snapshots, captions and exports are stored under `%LOCALAPPDATA%\FrameForge\jobs\<job-id>` on Windows; provider settings and keys are stored separately in `%LOCALAPPDATA%\FrameForge\settings.json`. Set `FRAMEFORGE_DATA_DIR` to choose another local storage root. The API accepts local uploads up to 16 GB by default; adjust `FRAMEFORGE_MAX_UPLOAD_BYTES` before launch if needed.

On Linux, `run.sh` activates the existing `.venv` and serves the built UI at `http://127.0.0.1:8000`. Prepare the Python environment and run `npm ci && npm run build` from `web/` first; then make sure `run.sh` is executable and launch it from the repository root.

## AI providers

- **Ollama** is the default local option. The app calls its native chat API at `http://localhost:11434/api/chat`; the default model suggestion is `qwen2.5:7b`. Change its base URL and model in Settings.
- **Gemini** defaults to `gemini-3.8-flash`. **OpenAI, Anthropic, Groq and OpenRouter** are also supported; each provider can be configured with its model, base URL and API key in Settings.
- The Settings panel persists provider selection locally. Save & Test sends a small text-only request and reports a clear pass/fail result. Keys are masked in the UI and are never included in API responses or job snapshots; a blank key preserves an existing saved key. A project-local `.env` can provide `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY` and `OPENROUTER_API_KEY` instead.
- Cloud providers receive transcript/audio-event context for clip selection, not video files. Transcription uses faster-whisper on CPU with `compute_type="int8"`; YouTube word-timed JSON3 captions are used as a fast path where available. PyTorch is not a hard dependency.

## CLI and MCP

The command-line adapter uses the exact same `core.pipeline.run_pipeline` entry point:

```bat
.venv\Scripts\python.exe -m core.cli --source "C:\Videos\talk.mp4" --provider ollama --layout face_track --caption-style "Hormozi Bold" --clip-count 4
```

Use `--json` for machine-readable output, `--no-transcribe` to opt out of local ASR when timed captions are missing, or `--help` for all supported options. `python cli.py` is an equivalent repository-root entry point; the installed `frameforge` command is also available.

MCP 2.3.0 is pinned and installed as part of the setup. Its stdio tools call the same pipeline:

```bat
.venv\Scripts\python.exe -m core.mcp_server
```

The `process_video` tool accepts a source, provider/model, custom direction, layout, caption style and clip-count/duration settings. `get_local_runtime` reports the H.264 selection and configured-provider availability without exposing keys. The server does not publish clips to social platforms.

## Verification

Run the deterministic no-network/no-FFmpeg end-to-end smoke check:

```bat
.venv\Scripts\python.exe test_pipeline.py --mocked
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check core api cli.py mcp_server.py test_pipeline.py tests
```

The `--live` mode probes FFmpeg encoders, checks Ollama's local `/api/tags` endpoint without pulling a model, creates a synthetic 15-second audio/video source, renders a vertical sample with ASS captions and `-14 LUFS` normalization, then probes the export. To additionally run the complete pipeline on your own media:

```bat
.venv\Scripts\python.exe test_pipeline.py --live --source "C:\Videos\talk.mp4" --provider ollama
```

`--live` writes its synthetic smoke artifacts below the per-user data folder (or the path supplied with `--work-dir`). It performs real local FFmpeg/AI work and may take several minutes. On Windows, also verify the output of the encoder probe in `setup.bat`; the sandbox's mocked tests are not evidence of AMF hardware support.

## Implementation notes

- `core/ingest.py` validates local files and supported YouTube/Twitch URLs, probes metadata, streams audio to FFmpeg for loudness peaks, prefers YouTube JSON3 word timing and falls back to faster-whisper CPU/int8.
- `core/brain.py` routes to Ollama, Gemini, OpenAI-compatible APIs or Anthropic; `core/settings_store.py` persists per-provider model/base URL/key settings with key-excluded API snapshots.
- `core/reframe.py` builds FFmpeg graphs for Auto Face-Track 9:16, Podcast Split-Screen, Gaming Webcam + Gameplay and Fit-Blur. Face detection failures fall back safely. Podcast Split-Screen currently divides the source into left/right panels; it does not infer who is speaking.
- `core/captions.py` writes clip-relative ASS with active-word highlighting and the Hormozi Bold, Clean Minimal, Gaming Neon and Podcast Classic presets.
- `core/encoder.py` performs a real one-frame runtime encode with each listed AMF/NVENC encoder before choosing it; failed hardware probes fall back to `libx264`. `core/render.py` also retries a failed hardware render with `libx264` when available.
- `core/job_store.py` persists job progress/results as atomic JSON snapshots in local app-data. Interrupted in-flight API jobs are marked failed when the service next starts.

## Dependency policy

Runtime versions are pinned in `pyproject.toml` and Windows compatibility constraints in `constraints-windows-py311.txt`. MediaPipe 0.10.21's upstream metadata pulls an OpenCV GUI/contrib wheel and CPU JAX packages; the constraints align OpenCV versions, and `setup.bat` reinstalls the matching headless wheel last. The faster-whisper conversion extra that introduces PyTorch is not installed. No CUDA, Docker, external fonts or social-posting integrations are required.
