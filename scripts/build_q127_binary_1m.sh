#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <N> <RA_EPOCH> [Q=127] [H=3]" >&2
  exit 1
fi

ROOT="/home/yu_mcc/PQC_salsa"
N="$1"
RA_EPOCH="$2"
Q="${3:-127}"
H="${4:-3}"
RUN_ROOT="$ROOT/runs/n${N}_q${Q}_h${H}_1m"
NUM_ORIG=$((4 * N))
AB_EPOCH="$RA_EPOCH"
RELOAD_SIZE=1000000
M_OVERRIDE="${M_OVERRIDE:-$N}"
RA_TIMEOUT="${RA_TIMEOUT:-30}"
MAX_TIMEOUT_COUNT="${MAX_TIMEOUT_COUNT:-10}"
BKZ_BLOCK_SIZE="${BKZ_BLOCK_SIZE:-10}"
BKZ_BLOCK_SIZE2="${BKZ_BLOCK_SIZE2:-15}"
THRESHOLD="${THRESHOLD:-3.0}"
THRESHOLD2="${THRESHOLD2:-4.0}"

source /home/yu_mcc/miniconda3/etc/profile.d/conda.sh
conda activate lattice_env

mkdir -p "$RUN_ROOT"

if [[ ! -f "$RUN_ROOT/origA_4n/orig_A.npy" ]]; then
  python generate.py \
    --cpu true \
    --step origA \
    --lwe true \
    --N "$N" \
    --Q "$Q" \
    --num_orig_samples "$NUM_ORIG" \
    --dump_path "$ROOT/runs" \
    --exp_name "n${N}_q${Q}_h${H}_1m" \
    --exp_id origA_4n
fi

if [[ ! -s "$RUN_ROOT/ra_main/data.prefix" ]]; then
  python generate.py \
    --cpu true \
    --step RA_tiny2 \
    --reload_data "$RUN_ROOT/origA_4n/orig_A.npy" \
    --N "$N" \
    --Q "$Q" \
    --m "$M_OVERRIDE" \
    --num_workers 1 \
    --epoch_size "$RA_EPOCH" \
    --timeout "$RA_TIMEOUT" \
    --max_timeout_count "$MAX_TIMEOUT_COUNT" \
    --float_type double \
    --lll_penalty 10 \
    --algo BKZ2.0 \
    --lll_delta 0.96 \
    --bkz_block_size "$BKZ_BLOCK_SIZE" \
    --algo2 BKZ2.0 \
    --lll_delta2 0.99 \
    --bkz_block_size2 "$BKZ_BLOCK_SIZE2" \
    --threshold "$THRESHOLD" \
    --threshold2 "$THRESHOLD2" \
    --use_polish false \
    --dump_path "$ROOT/runs" \
    --exp_name "n${N}_q${Q}_h${H}_1m" \
    --exp_id ra_main
fi

if [[ ! -s "$RUN_ROOT/ab_binary/train.prefix" ]]; then
  python generate.py \
    --cpu true \
    --step Ab \
    --reload_data "$RUN_ROOT/ra_main" \
    --N "$N" \
    --Q "$Q" \
    --m "$M_OVERRIDE" \
    --min_hamming "$H" \
    --max_hamming "$H" \
    --num_secret_seeds 1 \
    --secret_type binary \
    --sigma 3 \
    --epoch_size "$AB_EPOCH" \
    --reload_size "$RELOAD_SIZE" \
    --num_workers 1 \
    --dump_path "$ROOT/runs" \
    --exp_name "n${N}_q${Q}_h${H}_1m" \
    --exp_id ab_binary
fi
