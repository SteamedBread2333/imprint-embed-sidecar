#!/usr/bin/env bash
# Launch the embed sidecar with the plugin's own virtualenv.
#
# The venv is global (~/.imprint/plugins/embed/venv) per the ownership decision
# in the handover docs: model weights and interpreter stay out of the project,
# only derived vectors live in the vault. Exit non-zero on a missing venv so the
# plugin manager's health check fails cleanly and the Go side degrades to the
# lexical path instead of hanging.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${IMPRINT_EMBED_VENV:-$HOME/.imprint/plugins/embed/venv}"

if [[ -x "$VENV/bin/python" ]]; then
  PY="$VENV/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  # Fallback: a system interpreter new enough for fastembed.
  PY="$(command -v python3)"
else
  echo "imprint-embed: no python >= 3.11 found; run $HERE/scripts/setup.sh" >&2
  exit 1
fi

# Load embed + relate before listening so the first add does not pay cold-start
# latency or hit Go timeout_seconds. Set IMPRINT_EMBED_NO_WARM=1 to skip.
WARM=(--warm)
if [[ "${IMPRINT_EMBED_NO_WARM:-}" == "1" ]]; then
  WARM=()
fi

if [[ "${1:-}" == "prepare" ]]; then
  shift
  exec "$PY" "$HERE/server.py" --prepare "$@"
fi

exec "$PY" "$HERE/server.py" "${WARM[@]}" "$@"
