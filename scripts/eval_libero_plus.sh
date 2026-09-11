#!/usr/bin/env bash
set -euo pipefail
BENCH=libero-plus exec bash "$(dirname "$0")/eval_libero.sh" "$@"
