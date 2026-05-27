#!/usr/bin/env bash
set -euo pipefail

# Unified conservative missing-image completion for TCMS.
# Runs all FBDB/FBYG rates with the same completion strategy:
#   --tcms_missing_mode anchor_proto_blend --tcms_missing_blend 0.25
# By default, completed logs are skipped. Use FORCE=1 to rerun.

PY=${PY:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
DATA_PATH=${DATA_PATH:-/gly/tongqiang/dongyufeng/data/mmkg}
LOGDIR=${LOGDIR:-log/run_outputs/phase21_tcms_unified_blend025_4gpu}
FORCE=${FORCE:-0}
mkdir -p "$LOGDIR"

COMMON=(
  main.py
  --gpu 0
  --data_path "$DATA_PATH"
  --data_split norm
  --model_name SGMEA
  --epoch 250
  --eval_epoch 5
  --lr 5e-4
  --hidden_units 300,300,300
  --save_model 0
  --batch_size 1024
  --csls
  --csls_k 3
  --random_seed 42
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
  --tcms_beta 0.25
  --tcms_beta_warmup_start 70
  --tcms_beta_warmup_end 150
  --tcms_pretrain_loss_weight 0.002
  --tcms_selfsup_start 100
  --tcms_selfsup_warmup_end 170
  --tcms_sparse_weight 0.03
  --tcms_gate_init -3.0
  --tcms_missing_mode anchor_proto_blend
  --tcms_missing_blend 0.25
)

JOBS=(
  "FBDB15K 0.2 fbdb_r02_unified_blend025"
  "FBDB15K 0.5 fbdb_r05_unified_blend025"
  "FBDB15K 0.8 fbdb_r08_unified_blend025"
  "FBYG15K 0.2 fbyg_r02_unified_blend025"
  "FBYG15K 0.5 fbyg_r05_unified_blend025"
  "FBYG15K 0.8 fbyg_r08_unified_blend025"
)

GPUS=(4 5 6 7)
running=0
gpu_idx=0

launch_job() {
  local dataset=$1
  local rate=$2
  local name=$3
  local gpu=$4
  local anchor="$DATA_PATH/anchors_nameless/$dataset/norm_anchor_type_fixed.jsonl"
  local log="$LOGDIR/${name}.log"

  if [[ "$FORCE" != "1" && -f "$log" ]] && rg -q "Test result" "$log"; then
    echo "[skip] $name already completed: $log"
    return 0
  fi

  echo "[launch] gpu=$gpu dataset=$dataset rate=$rate name=$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "${COMMON[@]}" \
    --data_choice "$dataset" \
    --data_rate "$rate" \
    --external_anchor_type_jsonl "$anchor" \
    --exp_name "phase21_${name}" \
    --exp_id "phase21_${name}" \
    > "$log" 2>&1 &
  running=$((running + 1))
}

for job in "${JOBS[@]}"; do
  read -r dataset rate name <<< "$job"
  launch_job "$dataset" "$rate" "$name" "${GPUS[$gpu_idx]}"
  gpu_idx=$(((gpu_idx + 1) % ${#GPUS[@]}))

  if (( running >= ${#GPUS[@]} )); then
    wait -n
    running=$((running - 1))
  fi
done

while (( running > 0 )); do
  wait -n
  running=$((running - 1))
done

echo "[done] logs written to $LOGDIR"
