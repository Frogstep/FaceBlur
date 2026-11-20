#!/bin/sh
# Updates FaceBlur to the newest release from GitHub, keeping models, projects and settings.
#   ./update.sh                  install the newest version (if newer)
#   ./update.sh --check          only check for a newer version
#   ./update.sh FaceBlur-X.zip   install from a zip file
#   ./update.sh --rollback       go back to the previous version
# exec: the update replaces this file, so the shell must not read any more of it.
cd "$(dirname "$(readlink -f "$0")")" && exec .venv/bin/python install/update.py "$@"
