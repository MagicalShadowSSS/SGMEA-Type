#!/usr/bin/env bash
set -uo pipefail

# Phase 24: compare visual sanitization (TCMS) and type-aware modality routing (DEHR).
# Coverage: FBDB15K / FBYG15K at rates 0.2, 0.5, 0.8.
# Variants:
#   1) TCMS only: train/save TCMS keep checkpoint if missing, then evaluate it.
#   2) DEHR only: learned-DEHR V2-P over baseline checkpoint.
#   3) TCMS + DEHR: learned-DEHR V2-P over the TCMS checkpoint.
# Baseline metrics are not rerun here; use existing baseline checkpoints/logs for comparison.

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
PYTHON=${PYTHON:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
RUN_TAG=${RUN_TAG:-phase24_tcms_dehr_all_effect_$(date +%m%d_%H%M%S)}
FORCE=${FORCE:-0}
TRAIN_TCMS=${TRAIN_TCMS:-1}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/$RUN_TAG"
SAVE_DIR="$DATA_ROOT/SGMEA/save"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 1
fi

JOBS=(
  "FBDB15K 0.2 fbdb_r02"
  "FBDB15K 0.5 fbdb_r05"
  "FBDB15K 0.8 fbdb_r08"
  "FBYG15K 0.2 fbyg_r02"
  "FBYG15K 0.5 fbyg_r05"
  "FBYG15K 0.8 fbyg_r08"
)

baseline_ckpt_for() {
  local dataset=$1
  local rate=$2
  if [[ "$dataset" == "FBDB15K" ]]; then
    echo "SGMEA_FBDB15K_${rate}_baseline_warm_fbdb_rates_cdmr_0511_104108_r${rate//./}_"
  elif [[ "$dataset" == "FBYG15K" ]]; then
    echo "SGMEA_FBYG15K_${rate}_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r${rate//./}_"
  else
    echo "[error] unsupported dataset: $dataset" >&2
    return 1
  fi
}

tcms_exp_id_for() {
  local tag=$1
  echo "${tag}_tcms_keep_ckpt"
}

tcms_ckpt_for() {
  local dataset=$1
  local rate=$2
  local tag=$3
  local exp_id
  exp_id=$(tcms_exp_id_for "$tag")
  echo "SGMEA_${dataset}_${rate}_${exp_id}_"
}

anchor_for() {
  local dataset=$1
  echo "$DATA_ROOT/anchors_nameless/$dataset/norm_anchor_type_fixed.jsonl"
}

common_base_args=(
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
)

tcms_train_args=(
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

tcms_eval_extra=(
  --epoch 0
  --eval_epoch 1
  --save_model 0
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

dehr_extra=(
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

launch_bg() {
  local gpu=$1
  local log=$2
  shift 2
  echo "[launch] gpu=$gpu log=$log"
  (
    echo "[start] $(date '+%F %T') gpu=$gpu"
    echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $*"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$@"
    status=$?
    echo "[exit] $(date '+%F %T') status=$status"
    exit "$status"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
}

wait_batch_if_needed() {
  if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then
    wait_all
  fi
}

wait_all() {
  local failed=0
  local pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if (( failed != 0 )); then
    echo "[error] at least one job failed; inspect $LOG_DIR" >&2
    exit 1
  fi
}

echo "[info] run_tag=$RUN_TAG"
echo "[info] log_dir=$LOG_DIR"
echo "[info] save_dir=$SAVE_DIR"
echo "[info] gpus=$GPU_IDS"

PIDS=()
gpu_idx=0

if [[ "$TRAIN_TCMS" == "1" ]]; then
  for job in "${JOBS[@]}"; do
    read -r dataset rate tag <<< "$job"
    anchor=$(anchor_for "$dataset")
    exp_id=$(tcms_exp_id_for "$tag")
    ckpt=$(tcms_ckpt_for "$dataset" "$rate" "$tag")
    log="$LOG_DIR/${tag}_tcms_train.log"
    if [[ ! -f "$anchor" ]]; then
      echo "[error] missing anchor file: $anchor" >&2
      exit 1
    fi
    if [[ "$FORCE" != "1" && -f "$SAVE_DIR/${ckpt}.pkl" ]]; then
      echo "[tcms-train-skip] $ckpt exists"
      continue
    fi
    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}
    gpu_idx=$((gpu_idx + 1))
    launch_bg "$gpu" "$log" "${tcms_train_args[@]}" \
      --data_choice "$dataset" --data_rate "$rate" \
      --external_anchor_type_jsonl "$anchor" \
      --exp_name "$RUN_TAG" --exp_id "$exp_id"
    wait_batch_if_needed
  done
  if (( ${#PIDS[@]} > 0 )); then
    wait_all
  fi
fi

echo "[tcms-check] verifying TCMS checkpoints"
for job in "${JOBS[@]}"; do
  read -r dataset rate tag <<< "$job"
  ckpt=$(tcms_ckpt_for "$dataset" "$rate" "$tag")
  if [[ ! -f "$SAVE_DIR/${ckpt}.pkl" ]]; then
    echo "[error] missing TCMS checkpoint: $SAVE_DIR/${ckpt}.pkl" >&2
    exit 1
  fi
  echo "[tcms-ok] $ckpt"
done

PIDS=()
gpu_idx=0

# TCMS-only eval logs for this run tag.
for job in "${JOBS[@]}"; do
  read -r dataset rate tag <<< "$job"
  anchor=$(anchor_for "$dataset")
  ckpt=$(tcms_ckpt_for "$dataset" "$rate" "$tag")
  log="$LOG_DIR/${tag}_tcms_only_eval.log"
  if [[ "$FORCE" != "1" && -f "$log" ]] && rg -q "Test result" "$log"; then
    echo "[tcms-eval-skip] $log"
    continue
  fi
  gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}
  gpu_idx=$((gpu_idx + 1))
  launch_bg "$gpu" "$log" "${common_base_args[@]}" "${tcms_eval_extra[@]}" \
    --data_choice "$dataset" --data_rate "$rate" \
    --external_anchor_type_jsonl "$anchor" \
    --model_name_save "$ckpt" \
    --exp_name "$RUN_TAG" --exp_id "${tag}_tcms_only_eval"
  wait_batch_if_needed
done

# DEHR-only over baseline and TCMS+DEHR over TCMS checkpoint.
for variant in dehr_only tcms_plus_dehr; do
  for job in "${JOBS[@]}"; do
    read -r dataset rate tag <<< "$job"
    anchor=$(anchor_for "$dataset")
    if [[ "$variant" == "dehr_only" ]]; then
      warm_ckpt=$(baseline_ckpt_for "$dataset" "$rate") || exit 1
      extra_tcms=()
    else
      warm_ckpt=$(tcms_ckpt_for "$dataset" "$rate" "$tag")
      extra_tcms=("${tcms_eval_extra[@]}")
    fi
    if [[ ! -f "$SAVE_DIR/${warm_ckpt}.pkl" ]]; then
      echo "[error] missing warm checkpoint: $SAVE_DIR/${warm_ckpt}.pkl" >&2
      exit 1
    fi
    log="$LOG_DIR/${tag}_${variant}.log"
    if [[ "$FORCE" != "1" && -f "$log" ]] && rg -q "Test result" "$log"; then
      echo "[${variant}-skip] $log"
      continue
    fi
    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}
    gpu_idx=$((gpu_idx + 1))
    launch_bg "$gpu" "$log" "${common_base_args[@]}" "${extra_tcms[@]}" "${dehr_extra[@]}" \
      --data_choice "$dataset" --data_rate "$rate" \
      --external_anchor_type_jsonl "$anchor" \
      --model_name_save "$warm_ckpt" \
      --exp_name "$RUN_TAG" --exp_id "${tag}_${variant}"
    wait_batch_if_needed
  done
done

if (( ${#PIDS[@]} > 0 )); then
  wait_all
fi

"$PYTHON" - "$LOG_DIR" <<'PY'
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
metric_re = re.compile(r"Ep\s+(?:\d+|Test)\s+\|\s+(l2r|r2l): acc of top .*?= \[\s*([0-9.]+).*?\], mr =\s*([0-9.]+), mrr =\s*([0-9.]+)")
probe_re = re.compile(r"TCMSProbe \| (.*)")
dehr_re = re.compile(r"DEHRRouterTable .*?type_modality_weight_delta_abs_mean=([0-9.]+)|DEHRCalibPretrain .*?type_modality_weight_delta_abs_mean=([0-9.]+)")
rows = []
for path in sorted(log_dir.glob("*.log")):
    vals = {}
    tcms_probe = ""
    dehr_delta = ""
    status = "unknown"
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        if "[exit]" in line:
            status = line.rsplit("status=", 1)[-1].strip() if "status=" in line else status
        m = metric_re.search(line)
        if m:
            vals[m.group(1)] = (float(m.group(2)), float(m.group(4)))
        p = probe_re.search(line)
        if p:
            tcms_probe = p.group(1)
        d = dehr_re.search(line)
        if d:
            dehr_delta = d.group(1) or d.group(2) or dehr_delta
    name = path.stem
    if "_tcms_only_eval" in name:
        variant = "TCMS"
        key = name.replace("_tcms_only_eval", "")
    elif "_tcms_plus_dehr" in name:
        variant = "TCMS+DEHR"
        key = name.replace("_tcms_plus_dehr", "")
    elif "_dehr_only" in name:
        variant = "DEHR"
        key = name.replace("_dehr_only", "")
    elif "_tcms_train" in name:
        variant = "TCMS-train"
        key = name.replace("_tcms_train", "")
    else:
        variant = "other"
        key = name
    row = {"key": key, "variant": variant, "status": status, "log": str(path)}
    if {"l2r", "r2l"} <= vals.keys():
        row.update({
            "l2r_hits1": vals["l2r"][0],
            "r2l_hits1": vals["r2l"][0],
            "avg_hits1": (vals["l2r"][0] + vals["r2l"][0]) / 2.0,
            "l2r_mrr": vals["l2r"][1],
            "r2l_mrr": vals["r2l"][1],
            "avg_mrr": (vals["l2r"][1] + vals["r2l"][1]) / 2.0,
        })
    if tcms_probe:
        row["tcms_probe"] = tcms_probe
    if dehr_delta:
        row["dehr_weight_delta_abs_mean"] = dehr_delta
    rows.append(row)

rows.sort(key=lambda r: (r["key"], r["variant"]))
(log_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"[summary] {log_dir / 'summary.json'}")
print("key\tvariant\tavg_hits1\tavg_mrr\tl2r_h1\tr2l_h1\tstatus")
for r in rows:
    if "avg_hits1" in r:
        print(f"{r['key']}\t{r['variant']}\t{r['avg_hits1']:.5f}\t{r['avg_mrr']:.5f}\t{r['l2r_hits1']:.5f}\t{r['r2l_hits1']:.5f}\t{r['status']}")
    else:
        print(f"{r['key']}\t{r['variant']}\tNA\tNA\tNA\tNA\t{r['status']}")
PY

echo "[done] logs written to $LOG_DIR"
