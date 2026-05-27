#!/usr/bin/env bash
set -euo pipefail

# Iterative-learning experiments for FB15K-DB15K / FB15K-YG15K.
# Runs 12 experiment logs: 2 datasets x 3 seed ratios x {baseline, TCMS+DEHR}.
# For each TCMS+DEHR experiment, TCMS is first trained with IL and saved, then DEHR
# is calibrated/evaluated on the saved TCMS checkpoint. Early stopping is explicitly
# disabled during IL training so each run reaches the configured epoch.

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
PYTHON=${PYTHON:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
RUN_TAG=${RUN_TAG:-phase27_il_baseline_tcms_dehr_$(date +%m%d_%H%M%S)}

EPOCHS=${EPOCHS:-1000}
IL_START=${IL_START:-500}
SEMI_LEARN_STEP=${SEMI_LEARN_STEP:-5}
EVAL_EPOCH=${EVAL_EPOCH:-2}
BATCH_SIZE=${BATCH_SIZE:-2048}
LR=${LR:-5e-4}
SAVE_MODEL=${SAVE_MODEL:-1}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/$RUN_TAG"
SAVE_DIR="$DATA_ROOT/SGMEA/save"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 1
fi

anchor_for() {
  local dataset=$1
  echo "$DATA_ROOT/anchors_nameless/$dataset/norm_anchor_type_fixed.jsonl"
}

tag_for() {
  local dataset=$1 rate=$2
  if [[ "$dataset" == "FBDB15K" ]]; then
    echo "fbdb_r${rate/./}"
  else
    echo "fbyg_r${rate/./}"
  fi
}

common_train_args() {
  local dataset=$1 rate=$2 exp_id=$3 anchor=$4
  printf '%s ' \
    main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_split norm \
    --model_name SGMEA \
    --data_choice "$dataset" \
    --data_rate "$rate" \
    --epoch "$EPOCHS" \
    --il_start "$IL_START" \
    --semi_learn_step "$SEMI_LEARN_STEP" \
    --eval_epoch "$EVAL_EPOCH" \
    --only_test 0 \
    --save_model "$SAVE_MODEL" \
    --lr "$LR" \
    --hidden_units 300,300,300 \
    --batch_size "$BATCH_SIZE" \
    --csls \
    --csls_k 3 \
    --random_seed 42 \
    --workers 0 \
    --dist 0 \
    --accumulation_steps 1 \
    --scheduler cos \
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
    --disable_early_stop \
    --il \
    --external_anchor_type_jsonl "$anchor" \
    --no_tensorboard \
    --exp_name "$RUN_TAG" \
    --exp_id "$exp_id"
}

common_eval_args() {
  local dataset=$1 rate=$2 exp_id=$3 anchor=$4 ckpt=$5
  printf '%s ' \
    main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_split norm \
    --model_name SGMEA \
    --data_choice "$dataset" \
    --data_rate "$rate" \
    --model_name_save "$ckpt" \
    --epoch 0 \
    --eval_epoch 1 \
    --only_test 0 \
    --save_model 0 \
    --lr 3e-4 \
    --hidden_units 300,300,300 \
    --batch_size "$BATCH_SIZE" \
    --csls \
    --csls_k 3 \
    --random_seed 42 \
    --workers 0 \
    --dist 0 \
    --accumulation_steps 1 \
    --scheduler fixed \
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
    --disable_sgmea_guidance \
    --external_anchor_type_jsonl "$anchor" \
    --no_tensorboard \
    --exp_name "$RUN_TAG" \
    --exp_id "$exp_id"
}

saved_ckpt_name_for() {
  local dataset=$1 rate=$2 exp_id=$3
  local il_span=$((EPOCHS - IL_START))
  echo "SGMEA_${dataset}_${rate}_${exp_id}_il${il_span}_b${IL_START}_"
}

tcms_args() {
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
    --tcms_missing_mode keep
}

dehr_args() {
  printf '%s ' \
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
    --dehr_calib_pretrain_patience 25 \
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
  local -a args=("$@")
  echo "[launch] gpu=$gpu log=$log"
  (
    echo "[start] $(date '+%F %T') gpu=$gpu"
    printf '[cmd] CUDA_VISIBLE_DEVICES=%q %q -u' "$gpu" "$PYTHON"
    printf ' %q' "${args[@]}"
    printf '\n'
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "${args[@]}"
    status=$?
    echo "[exit] $(date '+%F %T') status=$status"
    exit "$status"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
}

launch_ours() {
  local gpu=$1 log=$2 dataset=$3 rate=$4 exp_id=$5 anchor=$6
  local ckpt
  ckpt=$(saved_ckpt_name_for "$dataset" "$rate" "$exp_id")
  local train_common eval_common
  train_common=$(common_train_args "$dataset" "$rate" "$exp_id" "$anchor")
  eval_common=$(common_eval_args "$dataset" "$rate" "${exp_id}_dehr_eval" "$anchor" "$ckpt")
  local -a train_args eval_args tcms_extra dehr_extra
  read -r -a train_args <<< "$train_common"
  read -r -a eval_args <<< "$eval_common"
  read -r -a tcms_extra <<< "$(tcms_args)"
  read -r -a dehr_extra <<< "$(dehr_args)"
  echo "[launch] gpu=$gpu log=$log"
  (
    echo "[start] $(date '+%F %T') gpu=$gpu stage=tcms_il_train"
    printf '[cmd-train] CUDA_VISIBLE_DEVICES=%q %q -u' "$gpu" "$PYTHON"
    printf ' %q' "${train_args[@]}" "${tcms_extra[@]}"
    printf '\n'
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "${train_args[@]}" "${tcms_extra[@]}"
    train_status=$?
    echo "[exit-train] $(date '+%F %T') status=$train_status ckpt=$ckpt"
    if (( train_status != 0 )); then
      exit "$train_status"
    fi
    if [[ ! -f "$SAVE_DIR/${ckpt}.pkl" ]]; then
      echo "[error] missing saved checkpoint $SAVE_DIR/${ckpt}.pkl" >&2
      exit 2
    fi
    echo "[start] $(date '+%F %T') gpu=$gpu stage=dehr_eval ckpt=$ckpt"
    printf '[cmd-dehr] CUDA_VISIBLE_DEVICES=%q %q -u' "$gpu" "$PYTHON"
    printf ' %q' "${eval_args[@]}" "${tcms_extra[@]}" "${dehr_extra[@]}"
    printf '\n'
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "${eval_args[@]}" "${tcms_extra[@]}" "${dehr_extra[@]}"
    eval_status=$?
    echo "[exit-dehr] $(date '+%F %T') status=$eval_status"
    exit "$eval_status"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
}

wait_batch() {
  local failed=0 pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  PIDS=()
  if (( failed != 0 )); then
    echo "[error] one or more jobs failed; inspect $LOG_DIR" >&2
    exit 1
  fi
}

echo "[info] run_tag=$RUN_TAG"
echo "[info] log_dir=$LOG_DIR"
echo "[info] gpu_ids=$GPU_IDS"
echo "[info] epochs=$EPOCHS il_start=$IL_START semi_learn_step=$SEMI_LEARN_STEP eval_epoch=$EVAL_EPOCH batch_size=$BATCH_SIZE lr=$LR save_model=$SAVE_MODEL"

for dataset in FBDB15K FBYG15K; do
  anchor=$(anchor_for "$dataset")
  [[ -f "$anchor" ]] || { echo "[error] missing anchor $anchor" >&2; exit 1; }
done

PIDS=()
gpu_idx=0

for dataset in FBDB15K FBYG15K; do
  for rate in 0.2 0.5 0.8; do
    tag=$(tag_for "$dataset" "$rate")
    anchor=$(anchor_for "$dataset")

    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    base_exp="${tag}_il_baseline_seed42"
    base_common=$(common_train_args "$dataset" "$rate" "$base_exp" "$anchor")
    launch "$gpu" "$LOG_DIR/${base_exp}.log" $base_common
    if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then wait_batch; fi

    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    ours_exp="${tag}_il_tcms_dehr_seed42"
    launch_ours "$gpu" "$LOG_DIR/${ours_exp}.log" "$dataset" "$rate" "$ours_exp" "$anchor"
    if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then wait_batch; fi
  done
done

if (( ${#PIDS[@]} > 0 )); then wait_batch; fi

"$PYTHON" - <<'PY' "$LOG_DIR"
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
metric_re = re.compile(r"Ep (?:Test|\d+) \| (l2r|r2l): acc of top \[1, 10, 50\] = \[([^\]]+)\].*?mrr = ([0-9.]+)")
rows = []
for log in sorted(log_dir.glob("*.log")):
    text = log.read_text(errors="ignore")
    last = {}
    for direction, hstr, mrr in metric_re.findall(text):
        hits = [float(x) for x in re.findall(r"[0-9.]+", hstr)]
        last[direction] = {"h1": hits[0], "h10": hits[1], "mrr": float(mrr)}
    name = log.stem
    parts = name.split("_")
    dataset = parts[0] if parts else ""
    rate = parts[1][1:] if len(parts) > 1 and parts[1].startswith("r") else ""
    variant = "tcms_dehr" if "tcms_dehr" in name else "baseline"
    row = {"log": log.name, "dataset": dataset, "rate": rate, "variant": variant}
    if "l2r" in last and "r2l" in last:
        row.update({
            "h1_l2r": last["l2r"]["h1"],
            "h1_r2l": last["r2l"]["h1"],
            "h1_avg": round((last["l2r"]["h1"] + last["r2l"]["h1"]) / 2, 6),
            "h10_avg": round((last["l2r"]["h10"] + last["r2l"]["h10"]) / 2, 6),
            "mrr_avg": round((last["l2r"]["mrr"] + last["r2l"]["mrr"]) / 2, 6),
            "status": "ok",
        })
    else:
        row["status"] = "missing_metrics"
    rows.append(row)

(log_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[summary] {log_dir / 'summary.json'}")
print("dataset\trate\tvariant\th1\th10\tmrr\tstatus")
for r in rows:
    if r["status"] == "ok":
        print(f"{r['dataset']}\t{r['rate']}\t{r['variant']}\t{r['h1_avg']:.5f}\t{r['h10_avg']:.5f}\t{r['mrr_avg']:.5f}\tok")
    else:
        print(f"{r['dataset']}\t{r['rate']}\t{r['variant']}\t-\t-\t-\t{r['status']}")
PY

echo "[done] $LOG_DIR"
