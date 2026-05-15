#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
TYPE_JSONL_FIXED=${TYPE_JSONL_FIXED:-$DATA_ROOT/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl}
LABSE_PATH=${LABSE_PATH:-/gly/tongqiang/dongyufeng/models/LaBSE}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RUN_TAG=${RUN_TAG:-typebias_$(date +%m%d_%H%M%S)}
BASE_CKPT=${BASE_CKPT:-SGMEA_FBDB15K_0.5_baseline_warm_softw_grid_0510_121128_}
EPOCH=${EPOCH:-40}
EVAL_EPOCH=${EVAL_EPOCH:-1}
SCHEDULER=${SCHEDULER:-fixed}
BATCH_SIZE=${BATCH_SIZE:-2048}
WORKERS=${WORKERS:-8}

export LABSE_PATH
cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/type_modality_bias_runs/$RUN_TAG"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"

run_one() {
  local gpu="$1"
  local name="$2"
  local lr="$3"
  local l2="$4"
  local scale="$5"
  local extra_flags="${6:-}"
  local exp_id="${name}_${RUN_TAG}"
  local log_file="$LOG_DIR/${name}_gpu${gpu}.log"
  echo "[launch] gpu=$gpu name=$name lr=$lr l2=$l2 scale=$scale scheduler=$SCHEDULER log=$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" /root/anaconda3/envs/sgmea/bin/python main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_choice FBDB15K \
    --data_split norm \
    --data_rate 0.5 \
    --model_name SGMEA \
    --model_name_save "$BASE_CKPT" \
    --exp_id "$exp_id" \
    --epoch "$EPOCH" \
    --eval_epoch "$EVAL_EPOCH" \
    --lr "$lr" \
    --hidden_units "300,300,300" \
    --batch_size "$BATCH_SIZE" \
    --csls \
    --csls_k 3 \
    --random_seed 42 \
    --workers "$WORKERS" \
    --scheduler "$SCHEDULER" \
    --attr_dim 300 \
    --img_dim 300 \
    --name_dim 300 \
    --char_dim 300 \
    --hidden_size 300 \
    --tau 0.1 \
    --structure_encoder gat \
    --num_attention_heads 1 \
    --num_hidden_layers 1 \
    --use_surface 0 \
    --use_intermediate 0 \
    --enable_sota \
    --disable_sgmea_guidance \
    --save_model 1 \
    --use_type_modality_bias \
    --type_modality_bias_only_train \
    --freeze_type_modality_bias_entity \
    --type_modality_bias_scale "$scale" \
    --type_modality_bias_l2 "$l2" \
    --external_anchor_type_jsonl "$TYPE_JSONL_FIXED" \
    $extra_flags \
    > "$log_file" 2>&1 &
}

CONFIGS=(
  "tamr_lr1e2_s10 1e-2 0 1.0"
  "tamr_lr5e3_s10 5e-3 0 1.0"
  "tamr_lr3e3_s10 3e-3 0 1.0"
  "tamr_lr1e3_s10 1e-3 0 1.0"
  "tamr_lr1e2_s20 1e-2 0 2.0"
  "tamr_lr5e3_s20 5e-3 0 2.0"
  "tamr_lr3e3_s20 3e-3 0 2.0"
  "tamr_lr1e3_s20 1e-3 0 2.0"
  "tamr_lr5e3_s30 5e-3 0 3.0"
  "tamr_lr1e3_s30 1e-3 0 3.0"
  "shared_lr1e2_s10 1e-2 0 1.0 --type_modality_bias_shared"
  "shared_lr5e3_s10 5e-3 0 1.0 --type_modality_bias_shared"
  "shared_lr1e3_s10 1e-3 0 1.0 --type_modality_bias_shared"
  "shared_lr5e3_s20 5e-3 0 2.0 --type_modality_bias_shared"
)

idx=0
pids=()
failed=0
for cfg in "${CONFIGS[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  # shellcheck disable=SC2086
  run_one "$gpu" $cfg
  pids+=("$!")
  idx=$((idx + 1))
  if (( idx % ${#GPUS[@]} == 0 )); then
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        echo "[warn] child pid=$pid failed; see logs under $LOG_DIR"
        failed=1
      fi
    done
    pids=()
  fi
done

if (( ${#pids[@]} > 0 )); then
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[warn] child pid=$pid failed; see logs under $LOG_DIR"
      failed=1
    fi
  done
fi
if (( failed != 0 )); then
  echo "[done-with-warnings] logs: $LOG_DIR"
else
  echo "[done] logs: $LOG_DIR"
fi
