#!/usr/bin/env bash
# Create the plugin virtualenv and install imprint-embed-sidecar (embed + relate).
# Ownership: venv -> ~/.imprint/plugins/embed/venv, weights -> ~/.cache/imprint/models
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${IMPRINT_EMBED_VENV:-$HOME/.imprint/plugins/embed/venv}"
CACHE="${IMPRINT_EMBED_CACHE:-$HOME/.cache/imprint/models}"
PYTHON_BIN="${PYTHON:-python3}"

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  echo "setup: $PYTHON_BIN is older than Python 3.11; set PYTHON=/path/to/python3.11+" >&2
  exit 1
fi

mkdir -p "$VENV" "$CACHE"
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
if [[ -f "$HERE/pyproject.toml" ]]; then
  "$VENV/bin/pip" install "$HERE"
else
  "$VENV/bin/pip" install "fastembed>=0.4.2" "onnxruntime>=1.18.0" "sentence-transformers>=3.0.0" "PyYAML>=6.0"
fi

echo "setup: installed into $VENV"
echo "setup: plugin dir (for imprint.yaml package:):"
IMPRINT_EMBED_CACHE="$CACHE" "$VENV/bin/imprint-embed-sidecar" --print-plugin-dir
