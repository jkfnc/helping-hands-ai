#!/bin/sh
# viam-server starts this; it passes the module socket path as $1.
cd "$(dirname "$0")" && exec ../.venv/bin/python main.py "$@"
