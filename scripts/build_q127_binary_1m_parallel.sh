#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yu_mcc/PQC_salsa"

# Calibrated to target about 1M train.prefix rows.
# n=5:  1500 matrices ->  20000 train rows => ~75000 matrices for 1M
# n=10: 1500 matrices ->  20000 train rows => ~75000 matrices for 1M
# n=30: 7000 matrices -> 410000 train rows => ~17074 matrices for 1M
# n=50: 7000 matrices -> 657340 train rows => ~10650 matrices for 1M
# n=80: estimated near ~2N reduced rows per matrix => use 7000 matrices as a first 1M-scale target.

bash "$ROOT/scripts/build_q127_binary_1m.sh" 30 17074 &
PID30=$!

bash "$ROOT/scripts/build_q127_binary_1m.sh" 50 10650 &
PID50=$!

bash "$ROOT/scripts/build_q127_binary_1m.sh" 80 7000 &
PID80=$!

echo "Started:"
echo "  n30 pid=$PID30"
echo "  n50 pid=$PID50"
echo "  n80 pid=$PID80"

wait "$PID30" "$PID50" "$PID80"
