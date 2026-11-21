# FrameForge · All-in-One AI Video Clipping Studio

A Windows-first, local-first studio for finding, reframing, captioning and exporting short clips from local videos or supported YouTube/Twitch URLs. The web UI, CLI and MCP adapter all call the same typed Python pipeline. No social posting, CUDA, PyTorch or Docker is required.

## Status

Phase 3 processing and integration are implemented, including local ingest, Ollama/Gemini routing, four reframing layouts, four ASS caption presets, FFmpeg rendering, API/UI, CLI, MCP and the mocked/live pipeline smoke script. Automated sandbox checks pass. This Linux sandbox has no FFmpeg and is not a Windows host, so neither real FFmpeg rendering nor Windows AMF/hardware acceleration has been verified. Run `setup.bat` and check the encoder result on your Windows PC before relying on hardware encoding.

## Windows prerequisites

- **Windows 10/11 x64**
- **CPython 3.11 x64** (the project intentionally pins `>=3.11,<3.12` for the selected MediaPipe and faster-whisper wheels)
- **Node.js 22.12+** or 20.19+ with npm, for the Vite studio
- **FFmpeg and ffprobe** on `PATH`; include `h264_amf` when AMD AMF is available and `libx264` for CPU fallback. Detected NVENC may be selected when AMF is not listed.
- **Ollama** is optional for local AI. Install it separately, start its service at `http://localhost:11434`, and pull a model such as `qwen2.5:7b` before selecting it.
- **Gemini** is opt-in. Set `GEMINI_API_KEY` in the environment for the process that launches the API/CLI/MCP server. The key is not saved in job records or returned by the API.

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

`run.bat` starts the loopback-only FastAPI service on port 8000, starts Vite on port 5173 and opens the studio in your browser. Choose a local video or paste a YouTube/Twitch URL, select the provider/layout/caption preset and add optional custom direction. The UI polls job progress and shows in-browser previews with MP4 download links.

Uploaded source media, job snapshots, captions and exports are stored per user under `%LOCALAPPDATA%\FrameForge\jobs\<job-id>` on Windows. Set `FRAMEFORGE_DATA_DIR` to choose another local storage root. The API accepts local uploads up to 16 GB by default; adjust `FRAMEFORGE_MAX_UPLOAD_BYTES` before launch if needed.

## AI providers

- **Ollama** is the default and local-first provider. The app calls Ollama's native chat API at `http://localhost:11434/api/chat`. Enter a model already installed in Ollama; the default suggestion is `qwen2.5:7b`.
- **Google Gemini** is an explicit cloud choice. Set `GEMINI_API_KEY`; the app sends the transcript, audio-peak timings and custom direction to Gemini's `generateContent` API. Media files are not uploaded to Gemini by the provider adapter.
- Transcription uses faster-whisper on CPU with `compute_type="int8"`. YouTube word-timed JSON3 captions are used as a fast path where available. PyTorch is not installed as a hard dependency.

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

The `process_video` tool accepts a source, provider/model, custom direction, layout, caption style and clip-count/duration settings. `get_local_runtime` reports the H.264 selection and whether Gemini is configured. The server does not publish clips to social platforms.

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
- `core/brain.py` routes to Ollama or opt-in Gemini and validates, ranks and de-duplicates clip candidates against source duration and requested clip lengths.
- `core/reframe.py` builds FFmpeg graphs for Auto Face-Track 9:16, Podcast Split-Screen, Gaming Webcam + Gameplay and Fit-Blur. Face detection failures fall back safely. Podcast Split-Screen currently divides the source into left/right panels; it does not infer who is speaking.
- `core/captions.py` writes clip-relative ASS with active-word highlighting and the Hormozi Bold, Clean Minimal, Gaming Neon and Podcast Classic presets.
- `core/render.py` tries a yt-dlp partial range for each selected remote clip and falls back to a full-source download if unsupported. It burns ASS captions, normalizes audio, and retries with `libx264` if an available hardware encoder fails.
- `core/job_store.py` persists job progress/results as atomic JSON snapshots in local app-data. Interrupted in-flight API jobs are marked failed when the service next starts.

## Dependency policy

Runtime versions are pinned in `pyproject.toml` and Windows compatibility constraints in `constraints-windows-py311.txt`. MediaPipe 0.10.21's upstream metadata pulls an OpenCV GUI/contrib wheel and CPU JAX packages; the constraints align OpenCV versions, and `setup.bat` reinstalls the matching headless wheel last. The faster-whisper conversion extra that introduces PyTorch is not installed. No CUDA, Docker, external fonts or social-posting integrations are required.
