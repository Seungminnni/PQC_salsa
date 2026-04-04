#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yu_mcc/PQC_salsa"
RUN_ROOT="$ROOT/runs/n50_q127_h3_paperstyle_4m"
N=50
Q=127
H=3
NUM_ORIG=200

# Calibrated from the completed n50_q127_h3_paperstyle_200k_fresh run:
# 7000 reduced matrices -> 657340 train samples, so target ~4,000,000
# train samples requires about 42591 matrices.
RA_EPOCH=42591
AB_EPOCH=42591
RELOAD_SIZE=1000000

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
    --exp_name "$(basename "$RUN_ROOT")" \
    --exp_id origA_4n
fi

if [[ ! -f "$RUN_ROOT/ra_main/data.prefix" ]]; then
  python generate.py \
    --cpu true \
    --step RA_tiny2 \
    --reload_data "$RUN_ROOT/origA_4n/orig_A.npy" \
    --N "$N" \
    --Q "$Q" \
    --m "$N" \
    --num_workers 1 \
    --epoch_size "$RA_EPOCH" \
    --timeout 30 \
    --float_type double \
    --lll_penalty 10 \
    --algo BKZ2.0 \
    --lll_delta 0.96 \
    --bkz_block_size 10 \
    --algo2 BKZ2.0 \
    --lll_delta2 0.99 \
    --bkz_block_size2 15 \
    --threshold 3.0 \
    --threshold2 4.0 \
    --use_polish false \
    --dump_path "$ROOT/runs" \
    --exp_name "$(basename "$RUN_ROOT")" \
    --exp_id ra_main
fi

if [[ ! -f "$RUN_ROOT/ab_binary/train.prefix" ]]; then
  python generate.py \
    --cpu true \
    --step Ab \
    --reload_data "$RUN_ROOT/ra_main" \
    --N "$N" \
    --Q "$Q" \
    --m "$N" \
    --min_hamming "$H" \
    --max_hamming "$H" \
    --num_secret_seeds 1 \
    --secret_type binary \
    --sigma 3 \
    --epoch_size "$AB_EPOCH" \
    --reload_size "$RELOAD_SIZE" \
    --num_workers 1 \
    --dump_path "$ROOT/runs" \
    --exp_name "$(basename "$RUN_ROOT")" \
    --exp_id ab_binary
fi
