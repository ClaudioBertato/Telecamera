#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
exec python3 "$HERE/examples/live_capture.py" "$@"
