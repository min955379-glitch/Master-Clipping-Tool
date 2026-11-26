#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/.venv/bin/activate" ]]; then
  echo "[ERROR] Python environment is missing. Create .venv and install requirements.txt first." >&2
  exit 1
fi
if [[ ! -f "$ROOT/web/dist/index.html" ]]; then
  echo "[ERROR] Built studio UI is missing. Run npm ci && npm run build from web/ first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
printf 'FrameForge is starting at http://127.0.0.1:8000\nPress Ctrl+C to stop.\n'
exec uvicorn api.server:app --host 127.0.0.1 --port 8000
