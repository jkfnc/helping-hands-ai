#!/bin/sh
# The demo's local Whisper helper, with all Whisper files kept inside whisper-local/.
cd "$(dirname "$0")" || exit 1
export HF_HOME="$PWD/whisper-local/hf-cache"
exec whisper-local/venv/bin/python -u ears.py
