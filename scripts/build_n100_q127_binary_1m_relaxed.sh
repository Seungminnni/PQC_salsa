#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yu_mcc/PQC_salsa"

# Heuristic 1M-train build for n=100, q=127, h=3.
# Relaxed RA settings are chosen to avoid the timeout pattern seen at n=80.
export RA_TIMEOUT=180
export MAX_TIMEOUT_COUNT=50
export BKZ_BLOCK_SIZE=8
export BKZ_BLOCK_SIZE2=10
export THRESHOLD=3.0
export THRESHOLD2=4.0

# About 1M train rows if train-per-matrix stays near ~2N.
bash "$ROOT/scripts/build_q127_binary_1m.sh" 100 5500 127 3
