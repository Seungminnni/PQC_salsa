#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yu_mcc/PQC_salsa"
RUN_ROOT="$ROOT/runs/n70_q127_h3_1m"
AB_DIR="$RUN_ROOT/ab_ternary"
SECRET_SEED="${1:-0}"
INIT_SEED="${2:-}"

source /home/yu_mcc/miniconda3/etc/profile.d/conda.sh
conda activate lattice_env

if [[ ! -f "$AB_DIR/train.prefix" ]]; then
  echo "Missing dataset: $AB_DIR/train.prefix" >&2
  exit 1
fi

AVAILABLE_SECRET_SEEDS=$(python - <<'PY'
import numpy as np
s = np.load('/home/yu_mcc/PQC_salsa/runs/n70_q127_h3_1m/ab_ternary/secret.npy')
print(s.shape[1])
PY
)

if [[ "$SECRET_SEED" -ge "$AVAILABLE_SECRET_SEEDS" ]]; then
  echo "Requested secret_seed=$SECRET_SEED, but dataset only has $AVAILABLE_SECRET_SEEDS secret seed(s); falling back to secret_seed=0."
  SECRET_SEED=0
fi

run_one() {
  local init_seed="$1"
  python train.py \
    --cpu false \
    --reload_data "$AB_DIR" \
    --secret_seed "$SECRET_SEED" \
    --env_base_seed "$((1000 + init_seed))" \
    --hamming 3 \
    --input_int_base 16 \
    --share_token 1 \
    --dump_path "$ROOT/runs" \
    --exp_name n70_q127_h3_1m \
    --exp_id "train_gpu_ternary_secret${SECRET_SEED}_init${init_seed}_paperlike" \
    --batch_size 128 \
    --epoch_size 20000 \
    --max_epoch 10000 \
    --eval_size 10000 \
    --distinguisher_size 128 \
    --enc_emb_dim 1024 \
    --dec_emb_dim 512 \
    --n_enc_layers 1 \
    --n_dec_layers 2 \
    --n_enc_heads 4 \
    --n_dec_heads 4 \
    --n_cross_heads 4 \
    --enc_loops 1 \
    --dec_loops 8 \
    --dropout 0 \
    --attention_dropout 0 \
    --optimizer 'adam_warmup,lr=0.00001,warmup_updates=8000,warmup_init_lr=0.00000001,weight_decay=0.99' \
    --num_workers 1 \
    --beam_size 1 \
    --max_output_len 10
}

if [[ -n "$INIT_SEED" ]]; then
  run_one "$INIT_SEED"
else
  for init_seed in 0 1 2 3 4; do
    echo "=== Running secret_seed=${SECRET_SEED}, init_seed=${init_seed} ==="
    run_one "$init_seed"
  done
fi
