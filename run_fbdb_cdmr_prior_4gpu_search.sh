#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
TYPE_JSONL_FIXED=${TYPE_JSONL_FIXED:-$DATA_ROOT/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl}
PRIOR_JSON=${PRIOR_JSON:-$DATA_ROOT/SGMEA/train_relation_hnm/type_modality_oracle_4seg_refine_0510_193727.json}
PRIOR_KEY=${PRIOR_KEY:-combined_greedy.best.scales}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RUN_TAG=${RUN_TAG:-cdmr_prior_$(date +%m%d_%H%M%S)}
BASE_CKPT=${BASE_CKPT:-SGMEA_FBDB15K_0.5_baseline_warm_softw_grid_0510_121128_}
EPOCH=${EPOCH:-40}
EVAL_EPOCH=${EVAL_EPOCH:-1}
SCHEDULER=${SCHEDULER:-fixed}
BATCH_SIZE=${BATCH_SIZE:-2048}
WORKERS=${WORKERS:-8}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/cdmr_prior_runs/$RUN_TAG"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"

run_one() {
  local gpu="$1"
  local name="$2"
  local lr="$3"
  local beta_init="$4"
  local beta_max="$5"
  local eval_tau="$6"
  local hidden_dim="$7"
  local proj_dim="$8"
  local clamp="$9"
  local type_l2="${10}"
  local exp_id="${name}_${RUN_TAG}"
  local log_file="$LOG_DIR/${name}_gpu${gpu}.log"
  echo "[launch] gpu=$gpu name=$name lr=$lr beta_init=$beta_init beta_max=$beta_max eval_tau=$eval_tau hidden=$hidden_dim proj=$proj_dim clamp=$clamp type_l2=$type_l2 log=$log_file"
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
    --type_modality_bias_scale 1.0 \
    --type_modality_bias_l2 "$type_l2" \
    --type_modality_bias_init_json "$PRIOR_JSON" \
    --type_modality_bias_init_key "$PRIOR_KEY" \
    --use_cdmr_router \
    --cdmr_proj_dim "$proj_dim" \
    --cdmr_type_emb_dim 16 \
    --cdmr_hidden_dim "$hidden_dim" \
    --cdmr_beta_max "$beta_max" \
    --cdmr_beta_init "$beta_init" \
    --cdmr_tau_route 1.0 \
    --cdmr_eval_tau_route "$eval_tau" \
    --cdmr_residual_clamp "$clamp" \
    --external_anchor_type_jsonl "$TYPE_JSONL_FIXED" \
    > "$log_file" 2>&1 &
}

CONFIGS=(
  "priorinit_lr1e3_b010_m05_t10_h64_p32_c05_l20 1e-3 0.10 0.5 1.0 64 32 0.5 0"
  "priorinit_lr5e4_b010_m05_t10_h64_p32_c05_l20 5e-4 0.10 0.5 1.0 64 32 0.5 0"
  "priorinit_lr3e4_b010_m05_t10_h64_p32_c05_l20 3e-4 0.10 0.5 1.0 64 32 0.5 0"
  "priorinit_lr2e3_b010_m05_t10_h64_p32_c05_l20 2e-3 0.10 0.5 1.0 64 32 0.5 0"

  "priordist_lr1e3_b010_m05_t10_h64_p32_c05_l2e4 1e-3 0.10 0.5 1.0 64 32 0.5 1e-4"
  "priordist_lr5e4_b010_m05_t10_h64_p32_c05_l2e4 5e-4 0.10 0.5 1.0 64 32 0.5 1e-4"
  "priordist_lr1e3_b010_m05_t10_h64_p32_c05_l1e3 1e-3 0.10 0.5 1.0 64 32 0.5 1e-3"
  "priordist_lr5e4_b010_m05_t10_h64_p32_c05_l1e3 5e-4 0.10 0.5 1.0 64 32 0.5 1e-3"

  "priorinit_lr1e3_b005_m05_t10_h64_p32_c05_l20 1e-3 0.05 0.5 1.0 64 32 0.5 0"
  "priorinit_lr1e3_b010_m03_t10_h64_p32_c03_l20 1e-3 0.10 0.3 1.0 64 32 0.3 0"
  "priorinit_lr1e3_b010_m07_t10_h64_p32_c07_l20 1e-3 0.10 0.7 1.0 64 32 0.7 0"
  "priorinit_lr1e3_b010_m05_t10_h128_p32_c05_l20 1e-3 0.10 0.5 1.0 128 32 0.5 0"
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
