#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yu_mcc/PQC_salsa"
RUNS_ROOT="$ROOT/runs"
Q=127
H=3
SIGMA=3
RELOAD_SIZE=1000000
NUM_SECRET_SEEDS=10
N70_TARGET_RA=8000
TS="$(date +%Y%m%d_%H%M%S)"

source /home/yu_mcc/miniconda3/etc/profile.d/conda.sh
conda activate lattice_env

backup_dir() {
  local dir="$1"
  if [[ -e "$dir" ]]; then
    local backup="${dir}_backup_${TS}"
    echo "[backup] $dir -> $backup"
    mv "$dir" "$backup"
  fi
}

generate_ab() {
  local n="$1"
  local secret_type="$2"
  local ab_epoch="$3"
  local run_root="$RUNS_ROOT/n${n}_q${Q}_h${H}_1m"
  local exp_id="ab_${secret_type}"

  echo "[generate] n=${n} secret_type=${secret_type} ab_epoch=${ab_epoch}"
  python generate.py \
    --cpu true \
    --step Ab \
    --reload_data "$run_root/ra_main" \
    --N "$n" \
    --Q "$Q" \
    --m "$n" \
    --min_hamming "$H" \
    --max_hamming "$H" \
    --num_secret_seeds "$NUM_SECRET_SEEDS" \
    --secret_type "$secret_type" \
    --sigma "$SIGMA" \
    --epoch_size "$ab_epoch" \
    --reload_size "$RELOAD_SIZE" \
    --num_workers 1 \
    --dump_path "$RUNS_ROOT" \
    --exp_name "n${n}_q${Q}_h${H}_1m" \
    --exp_id "$exp_id"
}

continue_n70_ra_if_needed() {
  local n=70
  local run_root="$RUNS_ROOT/n${n}_q${Q}_h${H}_1m"
  local data_prefix="$run_root/ra_main/data.prefix"
  local current_lines=0
  local current_matrices=0
  local remaining="$N70_TARGET_RA"

  if [[ -s "$data_prefix" ]]; then
    current_lines="$(wc -l < "$data_prefix")"
    current_matrices="$(( current_lines / n ))"
    remaining="$(( N70_TARGET_RA - current_matrices ))"
  fi

  if (( remaining <= 0 )); then
    echo "[skip] n=70 ra_main already has at least ${N70_TARGET_RA} matrices"
    return
  fi

  echo "[generate] n=70 ra_main current=${current_matrices} remaining=${remaining}"
  python generate.py \
    --cpu true \
    --step RA_tiny2 \
    --reload_data "$run_root/origA_4n/orig_A.npy" \
    --N "$n" \
    --Q "$Q" \
    --m "$n" \
    --num_workers 1 \
    --epoch_size "$remaining" \
    --timeout 30 \
    --max_timeout_count 10 \
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
    --dump_path "$RUNS_ROOT" \
    --exp_name "n${n}_q${Q}_h${H}_1m" \
    --exp_id ra_main
}

echo "[phase] n30 ab refresh"
backup_dir "$RUNS_ROOT/n30_q127_h3_1m/ab_binary"
backup_dir "$RUNS_ROOT/n30_q127_h3_1m/ab_ternary"
generate_ab 30 binary 17074
generate_ab 30 ternary 17074

echo "[phase] n50 ab refresh"
backup_dir "$RUNS_ROOT/n50_q127_h3_1m/ab_binary"
backup_dir "$RUNS_ROOT/n50_q127_h3_1m/ab_ternary"
generate_ab 50 binary 10650
generate_ab 50 ternary 10650

echo "[phase] n70 ra completion and ab refresh"
continue_n70_ra_if_needed
backup_dir "$RUNS_ROOT/n70_q127_h3_1m/ab_binary"
backup_dir "$RUNS_ROOT/n70_q127_h3_1m/ab_ternary"
generate_ab 70 binary 8000
generate_ab 70 ternary 8000

echo "[done] rebuild completed"
