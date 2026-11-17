#!/usr/bin/env bash
# Live demo: reruns three persona mixes (fp32, ~30-37 s on an 8-core CPU).
cd "$(dirname "$0")"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi  # optional: VENV=/path/to/venv
python demo.py 2>/dev/null
