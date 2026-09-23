#!/usr/bin/env bash
# Create the plugin virtualenv and install the embedding backend.
# Ownership: venv -> ~/.imprint/plugins/embed/venv, weights -> ~/.cache/imprint/models
set -euo pipefail

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
"$VENV/bin/pip" install "fastembed>=0.4.2" "onnxruntime>=1.18.0"

echo "setup: installed into $VENV"
echo "setup: verifying model load (downloads weights on first run)"
IMPRINT_EMBED_CACHE="$CACHE" "$VENV/bin/python" - <<'PY'
from fastembed import TextEmbedding
import os

m = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir=os.environ["IMPRINT_EMBED_CACHE"])
vec = next(iter(m.embed(["测试"])))
print(f"setup: dim={len(vec)}")
PY
