#!/usr/bin/env bash
set -uo pipefail

# Phase 22: save TCMS-keep checkpoints, then test TCMS-keep + learned-DEHR stacking.
# Default GPUs are the back four cards. Override with GPU_IDS=4,5,6,7 if needed.

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
PYTHON=${PYTHON:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
GPU_IDS=${GPU_IDS:-4,5,6,7}
RUN_TAG=${RUN_TAG:-phase22_tcms_keep_dehr_combo_$(date +%m%d_%H%M%S)}
FORCE=${FORCE:-0}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/$RUN_TAG"
SAVE_DIR="$DATA_ROOT/SGMEA/save"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} < 4 )); then
  echo "[error] GPU_IDS must contain at least four GPU ids, got: $GPU_IDS" >&2
  exit 1
fi

common_tcms_args=(
  main.py
  --gpu 0
  --data_path "$DATA_ROOT"
  --data_split norm
  --model_name SGMEA
  --epoch 250
  --eval_epoch 5
  --lr 5e-4
  --hidden_units 300,300,300
  --save_model 1
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
  --tcms_missing_mode keep
)

common_dehr_args=(
  main.py
  --gpu 0
  --data_path "$DATA_ROOT"
  --data_split norm
  --model_name SGMEA
  --hidden_units 300,300,300
  --batch_size 2048
  --csls
  --csls_k 1
  --random_seed 42
  --workers 0
  --dist 0
  --accumulation_steps 1
  --scheduler fixed
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
  --enable_sota
  --disable_sgmea_guidance
  --freeze_type_modality_bias_entity
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
  --tcms_missing_mode keep
  --epoch 0
  --eval_epoch 1
  --save_model 0
  --type_modality_bias_only_train
  --use_dehr_router
  --dehr_direction_mode learned_type
  --dehr_direction_values=0,0,0,0
  --dehr_global_direction_weight 0.0
  --dehr_type_direction_residual_scale 0.0
  --dehr_type_direction_residual_max 0.0
  --dehr_rho_init_values 0.7,0.7,0.7,0.7,0.7,0.7
  --dehr_rho_max 1.0
  --dehr_bias_clamp 0.35
  --dehr_learned_bias_scale 0.35
  --dehr_contextual_mix 0.15
  --dehr_contextual_source residual
  --dehr_learned_bias_l2_weight 1e-4
  --dehr_stat_feature_mask none
  --dehr_calib_pretrain_epochs 25
  --dehr_calib_pretrain_lr 5e-3
  --dehr_calib_pretrain_tau 0.05
  --dehr_calib_pretrain_batch_size 256
  --dehr_calib_pretrain_val_ratio 0.2
  --dehr_calib_pretrain_patience 8
  --dehr_calib_bias_l2 1e-4
  --dehr_calib_pair_source train_pseudo
  --dehr_calib_selection final
  --dehr_calib_pseudo_margin 0.04
  --dehr_calib_pseudo_csls_k 1
  --dehr_calib_pseudo_source dropout_consensus
  --dehr_calib_consensus_min_votes 2
  --lr 3e-4
)

stage1_jobs=(
  "FBDB15K 0.2 fbdb_r02_tcms_keep_ckpt"
  "FBDB15K 0.5 fbdb_r05_tcms_keep_ckpt"
  "FBDB15K 0.8 fbdb_r08_tcms_keep_ckpt"
  "FBYG15K 0.2 fbyg_r02_tcms_keep_ckpt"
  "FBYG15K 0.5 fbyg_r05_tcms_keep_ckpt"
  "FBYG15K 0.8 fbyg_r08_tcms_keep_ckpt"
)

stage2_jobs=(
  "FBDB15K 0.5 fbdb_r05_tcms_keep_ckpt fbdb_r05_tcms_keep_plus_dehr"
  "FBYG15K 0.5 fbyg_r05_tcms_keep_ckpt fbyg_r05_tcms_keep_plus_dehr"
)

checkpoint_name() {
  local dataset=$1
  local rate=$2
  local exp_id=$3
  echo "SGMEA_${dataset}_${rate}_${exp_id}_"
}

launch_stage1() {
  local dataset=$1
  local rate=$2
  local exp_id=$3
  local gpu=$4
  local anchor="$DATA_ROOT/anchors_nameless/$dataset/norm_anchor_type_fixed.jsonl"
  local ckpt
  ckpt=$(checkpoint_name "$dataset" "$rate" "$exp_id")
  local log="$LOG_DIR/${exp_id}.log"
  local ckpt_path="$SAVE_DIR/${ckpt}.pkl"

  if [[ "$FORCE" != "1" && -f "$ckpt_path" && -f "$log" ]] && rg -q "Test result" "$log"; then
    echo "[stage1-skip] $exp_id checkpoint exists: $ckpt_path"
    return 0
  fi

  echo "[stage1-launch] gpu=$gpu dataset=$dataset rate=$rate exp_id=$exp_id ckpt=$ckpt"
  (
    echo "[start] $(date '+%F %T') stage=tcms_keep gpu=$gpu dataset=$dataset rate=$rate exp_id=$exp_id"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "${common_tcms_args[@]}" \
      --data_choice "$dataset" \
      --data_rate "$rate" \
      --external_anchor_type_jsonl "$anchor" \
      --exp_name "$RUN_TAG" \
      --exp_id "$exp_id"
    status=$?
    echo "[exit] $(date '+%F %T') status=$status stage=tcms_keep exp_id=$exp_id"
    exit "$status"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
}

launch_stage2() {
  local dataset=$1
  local rate=$2
  local warm_exp_id=$3
  local exp_id=$4
  local gpu=$5
  local anchor="$DATA_ROOT/anchors_nameless/$dataset/norm_anchor_type_fixed.jsonl"
  local warm_ckpt
  warm_ckpt=$(checkpoint_name "$dataset" "$rate" "$warm_exp_id")
  local warm_path="$SAVE_DIR/${warm_ckpt}.pkl"
  local log="$LOG_DIR/${exp_id}.log"

  if [[ ! -f "$warm_path" ]]; then
    echo "[error] missing warm checkpoint for $exp_id: $warm_path" >&2
    exit 1
  fi
  if [[ "$FORCE" != "1" && -f "$log" ]] && rg -q "Test result" "$log"; then
    echo "[stage2-skip] $exp_id already completed: $log"
    return 0
  fi

  echo "[stage2-launch] gpu=$gpu dataset=$dataset rate=$rate exp_id=$exp_id warm=$warm_ckpt"
  (
    echo "[start] $(date '+%F %T') stage=tcms_keep_plus_dehr gpu=$gpu dataset=$dataset rate=$rate exp_id=$exp_id warm=$warm_ckpt"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "${common_dehr_args[@]}" \
      --data_choice "$dataset" \
      --data_rate "$rate" \
      --external_anchor_type_jsonl "$anchor" \
      --model_name_save "$warm_ckpt" \
      --exp_name "$RUN_TAG" \
      --exp_id "$exp_id"
    status=$?
    echo "[exit] $(date '+%F %T') status=$status stage=tcms_keep_plus_dehr exp_id=$exp_id"
    exit "$status"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
}

wait_for_batch() {
  local failed=0
  local pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  return "$failed"
}

echo "[info] run_tag=$RUN_TAG"
echo "[info] log_dir=$LOG_DIR"
echo "[info] save_dir=$SAVE_DIR"
echo "[info] gpus=$GPU_IDS"

PIDS=()
gpu_idx=0
for job in "${stage1_jobs[@]}"; do
  read -r dataset rate exp_id <<< "$job"
  launch_stage1 "$dataset" "$rate" "$exp_id" "${GPUS[$gpu_idx]}"
  gpu_idx=$(((gpu_idx + 1) % ${#GPUS[@]}))
  if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then
    if ! wait_for_batch; then
      echo "[error] at least one stage1 job failed; inspect $LOG_DIR" >&2
      exit 1
    fi
  fi
done

if (( ${#PIDS[@]} > 0 )); then
  if ! wait_for_batch; then
    echo "[error] at least one stage1 job failed; inspect $LOG_DIR" >&2
    exit 1
  fi
fi

echo "[stage1-done] verifying checkpoints"
for job in "${stage1_jobs[@]}"; do
  read -r dataset rate exp_id <<< "$job"
  ckpt=$(checkpoint_name "$dataset" "$rate" "$exp_id")
  if [[ ! -f "$SAVE_DIR/${ckpt}.pkl" ]]; then
    echo "[error] expected checkpoint not found: $SAVE_DIR/${ckpt}.pkl" >&2
    exit 1
  fi
  echo "[checkpoint-ok] $ckpt"
done

PIDS=()
gpu_idx=0
for job in "${stage2_jobs[@]}"; do
  read -r dataset rate warm_exp_id exp_id <<< "$job"
  launch_stage2 "$dataset" "$rate" "$warm_exp_id" "$exp_id" "${GPUS[$gpu_idx]}"
  gpu_idx=$(((gpu_idx + 1) % ${#GPUS[@]}))
done

if (( ${#PIDS[@]} > 0 )); then
  if ! wait_for_batch; then
    echo "[error] at least one stage2 job failed; inspect $LOG_DIR" >&2
    exit 1
  fi
fi

echo "[stage2-done] logs written to $LOG_DIR"

"$PYTHON" - "$LOG_DIR" <<'PY'
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
metric_re = re.compile(r"Ep\s+(?:\d+|Test)\s+\|\s+(l2r|r2l): acc of top .*?= \[\s*([0-9.]+).*?\], mr =\s*([0-9.]+), mrr =\s*([0-9.]+)")
for path in sorted(log_dir.glob("*.log")):
    vals = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = metric_re.search(line)
        if m:
            vals[m.group(1)] = (float(m.group(2)), float(m.group(4)))
    if vals:
        l = vals.get("l2r")
        r = vals.get("r2l")
        if l and r:
            print(f"[metric] {path.name}\tH1_l2r={l[0]:.4f}\tH1_r2l={r[0]:.4f}\tH1_avg={(l[0]+r[0])/2:.4f}\tMRR_l2r={l[1]:.4f}\tMRR_r2l={r[1]:.4f}\tMRR_avg={(l[1]+r[1])/2:.4f}")
PY
