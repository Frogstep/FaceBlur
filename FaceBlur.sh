#!/bin/sh
# Starts the FaceBlur app: ./FaceBlur.sh [VIDEO | PROJECT.faceblur.json]
# Other commands: ./FaceBlur.sh pick VIDEO, ./FaceBlur.sh render ..., ./FaceBlur.sh serve (see README).
DIR="$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-}" in
    pick|render|check|gui|serve) exec "$DIR/.venv/bin/python" -m faceblur "$@" ;;
    *) exec "$DIR/.venv/bin/python" -m faceblur gui "$@" ;;
esac
