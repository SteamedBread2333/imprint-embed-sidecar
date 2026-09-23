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

exec "$PY" "$HERE/server.py" "$@"
