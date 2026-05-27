#!/usr/bin/env bash
set -euo pipefail

# DBP15K r=0.3 experiment sweep for the two modules.
# Stage 1 trains and saves checkpoints for:
#   baseline, TCMS-keep, TCMS-completion(anchor_proto_blend)
# Stage 2 runs DEHR-only calibration on top of the saved baseline checkpoints.

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
PYTHON=${PYTHON:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
RUN_TAG=${RUN_TAG:-dbp03_tcms_dehr_$(date +%m%d_%H%M%S)}

BASE_EPOCHS=${BASE_EPOCHS:-500}
TCMS_EPOCHS=${TCMS_EPOCHS:-500}
EVAL_EPOCH=${EVAL_EPOCH:-10}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${BASE_BATCH_SIZE:-3500}}
BASE_BATCH_SIZE=${BASE_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
TCMS_BATCH_SIZE=${TCMS_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
DEHR_BATCH_SIZE=${DEHR_BATCH_SIZE:-2048}
WORKERS=${WORKERS:-0}
FORCE=${FORCE:-0}
START_STAGGER_SECONDS=${START_STAGGER_SECONDS:-45}
STRICT_SGMEA_PARAMS=${STRICT_SGMEA_PARAMS:-0}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/$RUN_TAG"
SAVE_DIR="$DATA_ROOT/SGMEA/save"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 1
fi

split_seed() {
  if [[ "$STRICT_SGMEA_PARAMS" == "1" ]]; then
    echo 42
    return 0
  fi
  case "$1" in
    zh_en) echo 42 ;;
    ja_en) echo 42 ;;
    fr_en) echo 3407 ;;
    *) echo "[error] unknown split: $1" >&2; return 1 ;;
  esac
}

split_lr() {
  if [[ "$STRICT_SGMEA_PARAMS" == "1" ]]; then
    echo 0.0005
    return 0
  fi
  case "$1" in
    zh_en) echo 0.0005 ;;
    ja_en) echo 0.0005 ;;
    fr_en) echo 0.00045 ;;
    *) echo "[error] unknown split: $1" >&2; return 1 ;;
  esac
}

split_tau() {
  if [[ "$STRICT_SGMEA_PARAMS" == "1" ]]; then
    echo 0.1
    return 0
  fi
  case "$1" in
    zh_en) echo 0.08 ;;
    ja_en) echo 0.08 ;;
    fr_en) echo 0.07 ;;
    *) echo "[error] unknown split: $1" >&2; return 1 ;;
  esac
}

type_jsonl_for() {
  echo "$CODE_ROOT/entity_type/dbp_${1}_type_refined.jsonl"
}

exp_id_for() {
  local split=$1 variant=$2
  echo "dbp15k_${split}_r03_${variant}_${RUN_TAG}"
}

ckpt_for() {
  local split=$1 variant=$2
  echo "SGMEA_DBP15K_${split}_$(exp_id_for "$split" "$variant")_"
}

common_train_args() {
  local split=$1 epochs=$2 batch_size=$3 exp_id=$4 use_external_types=${5:-1}
  local seed lr tau type_jsonl
  seed=$(split_seed "$split")
  lr=$(split_lr "$split")
  tau=$(split_tau "$split")
  type_jsonl=$(type_jsonl_for "$split")
  printf '%s ' \
    main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_choice DBP15K \
    --data_split "$split" \
    --data_rate 0.3 \
    --model_name SGMEA \
    --epoch "$epochs" \
    --eval_epoch "$EVAL_EPOCH" \
    --lr "$lr" \
    --tau "$tau" \
    --hidden_units 300,300,300 \
    --attr_dim 300 \
    --img_dim 300 \
    --name_dim 300 \
    --char_dim 300 \
    --hidden_size 300 \
    --batch_size "$batch_size" \
    --workers "$WORKERS" \
    --accumulation_steps 1 \
    --scheduler cos \
    --num_attention_heads 1 \
    --num_hidden_layers 1 \
    --structure_encoder gat \
    --use_surface 0 \
    --use_intermediate 1 \
    --csls \
    --csls_k 3 \
    --clip 10.0 \
    --save_model 1 \
    --enable_sota \
    --dist 0 \
    --random_seed "$seed" \
    --exp_name "$RUN_TAG" \
    --exp_id "$exp_id" \
    --no_tensorboard
  if [[ "$use_external_types" == "1" ]]; then
    printf '%s ' --external_anchor_type_jsonl "$type_jsonl"
  fi
}

tcms_args() {
  local missing_mode=$1
  printf '%s ' \
    --use_tcms \
    --tcms_prefusion \
    --tcms_zero_init_residual \
    --tcms_noise_same_type_ratio 0.7 \
    --tcms_disable_gate_match_features \
    --tcms_beta 0.25 \
    --tcms_beta_warmup_start 70 \
    --tcms_beta_warmup_end 150 \
    --tcms_pretrain_loss_weight 0.002 \
    --tcms_selfsup_start 100 \
    --tcms_selfsup_warmup_end 170 \
    --tcms_sparse_weight 0.03 \
    --tcms_gate_init -3.0 \
    --tcms_missing_mode "$missing_mode" \
    --tcms_missing_blend 0.25
}

common_dehr_args() {
  local split=$1 exp_id=$2 ckpt=$3
  local seed lr tau type_jsonl
  seed=$(split_seed "$split")
  lr=$(split_lr "$split")
  tau=$(split_tau "$split")
  type_jsonl=$(type_jsonl_for "$split")
  printf '%s ' \
    main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_choice DBP15K \
    --data_split "$split" \
    --data_rate 0.3 \
    --model_name SGMEA \
    --model_name_save "$ckpt" \
    --epoch 0 \
    --eval_epoch 1 \
    --lr 3e-4 \
    --tau "$tau" \
    --hidden_units 300,300,300 \
    --attr_dim 300 \
    --img_dim 300 \
    --name_dim 300 \
    --char_dim 300 \
    --hidden_size 300 \
    --batch_size "$DEHR_BATCH_SIZE" \
    --workers "$WORKERS" \
    --accumulation_steps 1 \
    --scheduler fixed \
    --num_attention_heads 1 \
    --num_hidden_layers 1 \
    --structure_encoder gat \
    --use_surface 0 \
    --use_intermediate 1 \
    --csls \
    --csls_k 3 \
    --clip 10.0 \
    --save_model 0 \
    --enable_sota \
    --dist 0 \
    --random_seed "$seed" \
    --external_anchor_type_jsonl "$type_jsonl" \
    --exp_name "$RUN_TAG" \
    --exp_id "$exp_id" \
    --no_tensorboard \
    --type_modality_bias_only_train \
    --use_dehr_router \
    --dehr_direction_mode learned_type \
    --dehr_direction_values=0,0,0,0 \
    --dehr_global_direction_weight 0.0 \
    --dehr_type_direction_residual_scale 0.0 \
    --dehr_type_direction_residual_max 0.0 \
    --dehr_rho_init_values 0.7,0.7,0.7,0.7,0.7,0.7 \
    --dehr_rho_max 1.0 \
    --dehr_bias_clamp 0.35 \
    --dehr_learned_bias_scale 0.35 \
    --dehr_contextual_mix 0.15 \
    --dehr_contextual_source residual \
    --dehr_learned_bias_l2_weight 1e-4 \
    --dehr_stat_feature_mask none \
    --dehr_calib_pretrain_epochs 25 \
    --dehr_calib_pretrain_lr 5e-3 \
    --dehr_calib_pretrain_tau 0.05 \
    --dehr_calib_pretrain_batch_size 256 \
    --dehr_calib_pretrain_val_ratio 0.2 \
    --dehr_calib_pretrain_patience 8 \
    --dehr_calib_bias_l2 1e-4 \
    --dehr_calib_pair_source train_pseudo \
    --dehr_calib_selection final \
    --dehr_calib_pseudo_margin 0.04 \
    --dehr_calib_pseudo_csls_k 1 \
    --dehr_calib_pseudo_source dropout_consensus \
    --dehr_calib_consensus_min_votes 2
}

launch() {
  local gpu=$1 log=$2
  shift 2
  echo "[launch] gpu=$gpu log=$log"
  (
    set +e
    echo "[start] $(date '+%F %T') gpu=$gpu"
    echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $*"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u $*
    status=$?
    echo "[exit] $(date '+%F %T') status=$status"
    exit "$status"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  if (( START_STAGGER_SECONDS > 0 )); then
    sleep "$START_STAGGER_SECONDS"
  fi
}

wait_all() {
  local failed=0 pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if (( failed != 0 )); then
    echo "[error] one or more jobs failed; inspect $LOG_DIR" >&2
    exit 1
  fi
}

maybe_wait_batch() {
  if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then
    wait_all
  fi
}

echo "[info] run_tag=$RUN_TAG"
echo "[info] log_dir=$LOG_DIR"
echo "[info] save_dir=$SAVE_DIR"
echo "[info] gpu_ids=$GPU_IDS"
echo "[info] base_epochs=$BASE_EPOCHS tcms_epochs=$TCMS_EPOCHS eval_epoch=$EVAL_EPOCH"
echo "[info] strict_sgmea_params=$STRICT_SGMEA_PARAMS base_batch_size=$BASE_BATCH_SIZE tcms_batch_size=$TCMS_BATCH_SIZE dehr_batch_size=$DEHR_BATCH_SIZE"

for split in zh_en ja_en fr_en; do
  type_jsonl=$(type_jsonl_for "$split")
  [[ -f "$type_jsonl" ]] || { echo "[error] missing type jsonl: $type_jsonl" >&2; exit 1; }
done

PIDS=()
gpu_idx=0

# Stage 1: train checkpoint-producing variants.
for split in zh_en ja_en fr_en; do
  exp_id=$(exp_id_for "$split" baseline)
  ckpt="$SAVE_DIR/$(ckpt_for "$split" baseline).pkl"
  log="$LOG_DIR/${split}_baseline.log"
  if [[ "$FORCE" != "1" && -f "$ckpt" ]]; then
    echo "[skip] existing baseline checkpoint: $ckpt"
  else
    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    launch "$gpu" "$log" $(common_train_args "$split" "$BASE_EPOCHS" "$BASE_BATCH_SIZE" "$exp_id" 0)
    maybe_wait_batch
  fi
done

for split in zh_en ja_en fr_en; do
  exp_id=$(exp_id_for "$split" tcms_keep)
  ckpt="$SAVE_DIR/$(ckpt_for "$split" tcms_keep).pkl"
  log="$LOG_DIR/${split}_tcms_keep.log"
  if [[ "$FORCE" != "1" && -f "$ckpt" ]]; then
    echo "[skip] existing TCMS-keep checkpoint: $ckpt"
  else
    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    launch "$gpu" "$log" $(common_train_args "$split" "$TCMS_EPOCHS" "$TCMS_BATCH_SIZE" "$exp_id" 1) $(tcms_args keep)
    maybe_wait_batch
  fi
done

for split in zh_en ja_en fr_en; do
  exp_id=$(exp_id_for "$split" tcms_completion)
  ckpt="$SAVE_DIR/$(ckpt_for "$split" tcms_completion).pkl"
  log="$LOG_DIR/${split}_tcms_completion.log"
  if [[ "$FORCE" != "1" && -f "$ckpt" ]]; then
    echo "[skip] existing TCMS-completion checkpoint: $ckpt"
  else
    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    launch "$gpu" "$log" $(common_train_args "$split" "$TCMS_EPOCHS" "$TCMS_BATCH_SIZE" "$exp_id" 1) $(tcms_args anchor_proto_blend)
    maybe_wait_batch
  fi
done

if (( ${#PIDS[@]} > 0 )); then
  wait_all
fi

# Stage 2: DEHR-only calibration over baseline checkpoints.
for split in zh_en ja_en fr_en; do
  base_ckpt=$(ckpt_for "$split" baseline)
  base_path="$SAVE_DIR/${base_ckpt}.pkl"
  [[ -f "$base_path" ]] || { echo "[error] missing baseline checkpoint for DEHR: $base_path" >&2; exit 1; }
  exp_id=$(exp_id_for "$split" dehr)
  log="$LOG_DIR/${split}_dehr.log"
  gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
  launch "$gpu" "$log" $(common_dehr_args "$split" "$exp_id" "$base_ckpt")
  maybe_wait_batch
done

if (( ${#PIDS[@]} > 0 )); then
  wait_all
fi

"$PYTHON" - <<'PY' "$LOG_DIR"
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
pat = re.compile(r"Ep\s+(?:Test|\d+)\s+\|\s+(l2r|r2l): acc of top \[1, 10, 50\] = \[([^\]]+)\].*?mrr = ([0-9.]+)")
summary = {}
for log in sorted(log_dir.glob("*.log")):
    rows = []
    for line in log.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            hits = [float(x) for x in m.group(2).replace(',', ' ').split()]
            rows.append({"dir": m.group(1), "hits1": hits[0], "hits10": hits[1], "hits50": hits[2], "mrr": float(m.group(3))})
    if rows:
        last = rows[-2:] if len(rows) >= 2 else rows
        summary[log.stem] = last
(log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[done] $(date '+%F %T') run_tag=$RUN_TAG"
