#!/usr/bin/env bash
set -uo pipefail

# Phase 23: FBDB-focused TCMS follow-up.
# 1) Repeat FBDB r=0.5 TCMS keep to check whether the previous high run is reproducible.
# 2) Try a small set of FBDB r=0.2 configurations aimed at reducing low-supervision degradation.

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
PYTHON=${PYTHON:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
GPU_IDS=${GPU_IDS:-4,5,6,7}
RUN_TAG=${RUN_TAG:-phase23_fbdb_tcms_repro_r02_$(date +%m%d_%H%M%S)}
BATCH_SIZE=${BATCH_SIZE:-1024}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/$RUN_TAG"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 1
fi

COMMON=(
  main.py
  --gpu 0
  --data_path "$DATA_ROOT"
  --data_split norm
  --data_choice FBDB15K
  --model_name SGMEA
  --epoch 250
  --eval_epoch 5
  --lr 5e-4
  --hidden_units 300,300,300
  --batch_size "$BATCH_SIZE"
  --csls
  --csls_k 3
  --workers 0
  --dist 0
  --accumulation_steps 1
  --scheduler cos
  --attr_dim 300
  --img_dim 300
  --name_dim 300
  --char_dim 300
  --hidden_size 300
  --tau 0.1
  --structure_encoder gat
  --num_attention_heads 1
  --num_hidden_layers 1
  --use_surface 0
  --use_intermediate 0
  --disable_sgmea_guidance
  --no_tensorboard
  --use_tcms
  --tcms_prefusion
  --tcms_zero_init_residual
  --tcms_noise_same_type_ratio 0.7
  --tcms_disable_gate_match_features
  --tcms_beta_warmup_start 70
  --tcms_beta_warmup_end 150
  --tcms_selfsup_start 100
  --tcms_selfsup_warmup_end 170
  --external_anchor_type_jsonl "$DATA_ROOT/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl"
)

# Columns:
# rate exp_id seed save_model missing_mode beta sparse_weight gate_init pretrain_weight missing_blend
# missing_blend=- means no --tcms_missing_blend flag.
JOBS=(
  # r=0.5 keep reproducibility. Same nominal setting as the previous high run, plus seed repeats.
  "0.5 fbdb_r05_keep_repro_seed42 42 1 keep 0.25 0.03 -3.0 0.002 -"
  "0.5 fbdb_r05_keep_repro_seed43 43 1 keep 0.25 0.03 -3.0 0.002 -"
  "0.5 fbdb_r05_keep_repro_seed44 44 1 keep 0.25 0.03 -3.0 0.002 -"

  # r=0.2 low-supervision attempts. Conservative residuals and softer sparse pressure are tested first.
  "0.2 fbdb_r02_keep_beta010_seed42 42 0 keep 0.10 0.03 -3.0 0.002 -"
  "0.2 fbdb_r02_keep_beta015_gate25_sparse01_seed42 42 0 keep 0.15 0.01 -2.5 0.002 -"
  "0.2 fbdb_r02_keep_beta020_gate25_sparse01_seed42 42 0 keep 0.20 0.01 -2.5 0.002 -"
  "0.2 fbdb_r02_blend010_beta015_gate25_seed42 42 0 anchor_proto_blend 0.15 0.01 -2.5 0.002 0.10"
  "0.2 fbdb_r02_anchor_beta015_gate25_seed42 42 0 anchor 0.15 0.01 -2.5 0.002 -"
)

PIDS=()
RUNNING=0
GPU_IDX=0
FAILED=0

launch_job() {
  local rate=$1
  local exp_id=$2
  local seed=$3
  local save_model=$4
  local missing_mode=$5
  local beta=$6
  local sparse_weight=$7
  local gate_init=$8
  local pretrain_weight=$9
  local missing_blend=${10}
  local gpu=${GPUS[$GPU_IDX]}
  GPU_IDX=$(((GPU_IDX + 1) % ${#GPUS[@]}))

  local log="$LOG_DIR/${exp_id}.log"
  echo "[launch] gpu=$gpu rate=$rate exp_id=$exp_id seed=$seed mode=$missing_mode beta=$beta sparse=$sparse_weight gate=$gate_init pretrain=$pretrain_weight blend=$missing_blend"
  (
    echo "[start] $(date '+%F %T') gpu=$gpu rate=$rate exp_id=$exp_id seed=$seed mode=$missing_mode beta=$beta sparse=$sparse_weight gate=$gate_init pretrain=$pretrain_weight blend=$missing_blend"
    extra=()
    if [[ "$missing_blend" != "-" ]]; then
      extra+=(--tcms_missing_blend "$missing_blend")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "${COMMON[@]}" \
      --data_rate "$rate" \
      --random_seed "$seed" \
      --save_model "$save_model" \
      --tcms_missing_mode "$missing_mode" \
      --tcms_beta "$beta" \
      --tcms_sparse_weight "$sparse_weight" \
      --tcms_gate_init "$gate_init" \
      --tcms_pretrain_loss_weight "$pretrain_weight" \
      --exp_name "$RUN_TAG" \
      --exp_id "$exp_id" \
      "${extra[@]}"
    status=$?
    echo "[exit] $(date '+%F %T') status=$status exp_id=$exp_id"
    exit "$status"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  RUNNING=$((RUNNING + 1))
}

echo "[info] run_tag=$RUN_TAG"
echo "[info] log_dir=$LOG_DIR"
echo "[info] gpus=$GPU_IDS"
echo "[info] batch_size=$BATCH_SIZE"

for job in "${JOBS[@]}"; do
  read -r rate exp_id seed save_model missing_mode beta sparse_weight gate_init pretrain_weight missing_blend <<< "$job"
  launch_job "$rate" "$exp_id" "$seed" "$save_model" "$missing_mode" "$beta" "$sparse_weight" "$gate_init" "$pretrain_weight" "$missing_blend"
  if (( RUNNING >= ${#GPUS[@]} )); then
    if ! wait -n; then
      FAILED=1
    fi
    RUNNING=$((RUNNING - 1))
  fi
done

while (( RUNNING > 0 )); do
  if ! wait -n; then
    FAILED=1
  fi
  RUNNING=$((RUNNING - 1))
done

echo "[done] $(date '+%F %T') failed=$FAILED logs=$LOG_DIR"
exit "$FAILED"
