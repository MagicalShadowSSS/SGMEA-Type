#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
GPU_IDS=${GPU_IDS:-0,1,2,3}
DATASETS=${DATASETS:-FBDB15K,FBYG15K}
RATES=${RATES:-0.2,0.5,0.8}
BASELINE_LRS=${BASELINE_LRS:-5e-4}
RUN_TAG=${RUN_TAG:-dehr_typescale_$(date +%m%d_%H%M%S)}

DEHR_EPOCH=${DEHR_EPOCH:-40}
DEHR_LR=${DEHR_LR:-1e-4}
DEHR_BATCH_SIZE=${DEHR_BATCH_SIZE:-2048}
DEHR_SCHEDULER=${DEHR_SCHEDULER:-fixed}
DEHR_CSLS_K=${DEHR_CSLS_K:-1}
DEHR_KL_WEIGHT=${DEHR_KL_WEIGHT:-1e-3}
DEHR_RHO_WEIGHT=${DEHR_RHO_WEIGHT:-1e-3}
DEHR_BIAS_CLAMP=${DEHR_BIAS_CLAMP:-0.5}
DEHR_DIRECTION_MODE=${DEHR_DIRECTION_MODE:-free}
DEHR_DIRECTION_VALUES=${DEHR_DIRECTION_VALUES:--0.35,0.15,-0.10,0.30}
DEHR_EVIDENCE_MIX=${DEHR_EVIDENCE_MIX:-0}
DEHR_RESIDUAL_GAMMA=${DEHR_RESIDUAL_GAMMA:-0}
DEHR_ANCHOR_SCALE=${DEHR_ANCHOR_SCALE:-1.0}
DEHR_RESIDUAL_FORM=${DEHR_RESIDUAL_FORM:-additive}
DEHR_RESIDUAL_CONFIDENCE=${DEHR_RESIDUAL_CONFIDENCE:-1}
DEHR_TOKEN_FUSION=${DEHR_TOKEN_FUSION:-concat}
DEHR_ZERO_INIT_HEAD=${DEHR_ZERO_INIT_HEAD:-0}
DEHR_RESIDUAL_L2_WEIGHT=${DEHR_RESIDUAL_L2_WEIGHT:-0}
EVAL_EPOCH=${EVAL_EPOCH:-1}
WORKERS=${WORKERS:-0}
SEED=${SEED:-42}

# Four interpretable type-level concentration initializations.
# The code maps these values to the first four non-generic type rows in the dataset order.
RHO_GRID=${RHO_GRID:-"0.6,0.8,1.0,1.2;0.8,1.0,1.2,1.4;1.0,1.2,1.4,1.6;1.2,1.4,1.6,1.8"}
RHO_MAX_GRID=${RHO_MAX_GRID:-"2.0 3.0"}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/dehr_typescale_sweep/$RUN_TAG"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
IFS=',' read -r -a RATE_LIST <<< "$RATES"
IFS=',' read -r -a LR_LIST <<< "$BASELINE_LRS"
IFS=';' read -r -a RHO_LIST <<< "$RHO_GRID"
read -r -a RHO_MAX_LIST <<< "$RHO_MAX_GRID"

type_jsonl_for() {
  local dataset="$1"
  echo "$DATA_ROOT/anchors_nameless/${dataset}/norm_anchor_type_fixed.jsonl"
}

base_ckpt_for() {
  local dataset="$1"
  local rate="$2"
  local base_lr="$3"
  if [[ "$dataset" == "FBDB15K" ]]; then
    echo "SGMEA_FBDB15K_${rate}_baseline_warm_fbdb_rates_cdmr_0511_104108_r${rate//./}_"
  elif [[ "$dataset" == "FBYG15K" ]]; then
    local lr_tag
    if [[ "$base_lr" == "5e-4" ]]; then
      lr_tag="lr5e4"
    elif [[ "$base_lr" == "5e-5" ]]; then
      lr_tag="lr5e5"
    else
      echo "[error] unsupported FBYG baseline lr: $base_lr" >&2
      return 1
    fi
    echo "SGMEA_FBYG15K_${rate}_baseline_warm_fbyg_specific_prior_0512_111257_${lr_tag}_r${rate//./}_"
  else
    echo "[error] unsupported dataset: $dataset" >&2
    return 1
  fi
}

run_dehr() {
  local gpu="$1"
  local dataset="$2"
  local rate="$3"
  local base_lr="$4"
  local rho_values="$5"
  local rho_max="$6"

  local type_jsonl
  type_jsonl="$(type_jsonl_for "$dataset")"
  if [[ ! -f "$type_jsonl" ]]; then
    echo "[error] missing type jsonl: $type_jsonl" >&2
    return 1
  fi

  local base_ckpt
  base_ckpt="$(base_ckpt_for "$dataset" "$rate" "$base_lr")"
  local ckpt_path="$DATA_ROOT/SGMEA/save/${base_ckpt}.pkl"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[error] missing baseline checkpoint: $ckpt_path" >&2
    return 1
  fi

  local rho_tag="${rho_values//,/-}"
  local exp_suffix="dehr_${RUN_TAG}_${dataset}_r${rate//./}_blr${base_lr//-}_rho${rho_tag}_m${rho_max}"
  local log_file="$LOG_DIR/${exp_suffix}_gpu${gpu}.log"
  echo "[dehr] gpu=$gpu dataset=$dataset rate=$rate base_lr=$base_lr rho=$rho_values rho_max=$rho_max log=$log_file"
  {
    echo "[start] $(date '+%F %T') gpu=$gpu dataset=$dataset rate=$rate base_lr=$base_lr rho=$rho_values rho_max=$rho_max"
    echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu python main.py --data_choice $dataset --data_rate $rate --model_name_save $base_ckpt --lr $DEHR_LR --workers $WORKERS --dehr_bias_clamp $DEHR_BIAS_CLAMP --dehr_direction_mode $DEHR_DIRECTION_MODE --dehr_direction_values $DEHR_DIRECTION_VALUES --dehr_kl_weight $DEHR_KL_WEIGHT --dehr_rho_weight $DEHR_RHO_WEIGHT"
  } >> "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu" /root/anaconda3/envs/sgmea/bin/python main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_choice "$dataset" \
    --data_split norm \
    --data_rate "$rate" \
    --model_name SGMEA \
    --model_name_save "$base_ckpt" \
    --exp_id "$exp_suffix" \
    --epoch "$DEHR_EPOCH" \
    --eval_epoch "$EVAL_EPOCH" \
    --lr "$DEHR_LR" \
    --hidden_units "300,300,300" \
    --save_model 1 \
    --batch_size "$DEHR_BATCH_SIZE" \
    --csls \
    --csls_k "$DEHR_CSLS_K" \
    --random_seed "$SEED" \
    --workers "$WORKERS" \
    --dist 0 \
    --accumulation_steps 1 \
    --scheduler "$DEHR_SCHEDULER" \
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
    --type_modality_bias_only_train \
    --freeze_type_modality_bias_entity \
    --use_dehr_router \
    --dehr_proj_dim 32 \
    --dehr_hidden_dim 64 \
    --dehr_heads 4 \
    --dehr_layers 1 \
    --dehr_dropout 0.1 \
    --dehr_alpha0 1.0 \
    --dehr_direction_mode "$DEHR_DIRECTION_MODE" \
    --dehr_direction_values="$DEHR_DIRECTION_VALUES" \
    --dehr_evidence_mix "$DEHR_EVIDENCE_MIX" \
    --dehr_residual_gamma "$DEHR_RESIDUAL_GAMMA" \
    --dehr_anchor_scale "$DEHR_ANCHOR_SCALE" \
    --dehr_residual_form "$DEHR_RESIDUAL_FORM" \
    --dehr_residual_confidence "$DEHR_RESIDUAL_CONFIDENCE" \
    --dehr_token_fusion "$DEHR_TOKEN_FUSION" \
    $(if [[ "$DEHR_ZERO_INIT_HEAD" == "1" ]]; then echo "--dehr_zero_init_head"; fi) \
    --dehr_residual_l2_weight "$DEHR_RESIDUAL_L2_WEIGHT" \
    --dehr_rho_init_values "$rho_values" \
    --dehr_rho_max "$rho_max" \
    --dehr_kl_weight "$DEHR_KL_WEIGHT" \
    --dehr_rho_weight "$DEHR_RHO_WEIGHT" \
    --dehr_bias_clamp "$DEHR_BIAS_CLAMP" \
    --external_anchor_type_jsonl "$type_jsonl" \
    >> "$log_file" 2>&1
  local status=$?
  echo "[exit] $(date '+%F %T') status=$status" >> "$log_file"
  return "$status"
}

jobs=()
idx=0
failed=0
for dataset in "${DATASET_LIST[@]}"; do
  for base_lr in "${LR_LIST[@]}"; do
    for rate in "${RATE_LIST[@]}"; do
      for rho_values in "${RHO_LIST[@]}"; do
        for rho_max in "${RHO_MAX_LIST[@]}"; do
          gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
          run_dehr "$gpu" "$dataset" "$rate" "$base_lr" "$rho_values" "$rho_max" &
          jobs+=("$!")
          idx=$((idx + 1))
          if (( ${#jobs[@]} >= ${#GPUS[@]} )); then
            next_jobs=()
            for pid in "${jobs[@]}"; do
              if ! wait "$pid"; then
                failed=1
              fi
            done
            jobs=("${next_jobs[@]}")
          fi
        done
      done
    done
  done
done

for pid in "${jobs[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

/root/anaconda3/envs/sgmea/bin/python - "$LOG_DIR" <<'PY'
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
line_re = re.compile(r"Ep\s+(\d+|Test)\s+\|\s+(l2r|r2l): acc of top .*?= \[\s*([0-9.]+)")
rho_re = re.compile(r"DEHRRouterTable .*? rho=([0-9.,-]+)")
rows = []
for path in sorted(log_dir.glob("*.log")):
    vals = {}
    last_rho = ""
    for line in path.read_text(errors="ignore").splitlines():
        m = line_re.search(line)
        if m:
            vals[m.group(2)] = float(m.group(3))
        r = rho_re.search(line)
        if r:
            last_rho = r.group(1)
    row = {"name": path.stem, "log": str(path), "rho": last_rho}
    if {"l2r", "r2l"} <= vals.keys():
        row.update({
            "l2r": vals["l2r"],
            "r2l": vals["r2l"],
            "avg_hits1": (vals["l2r"] + vals["r2l"]) / 2.0,
        })
    else:
        row["status"] = "missing_metric"
    rows.append(row)
rows.sort(key=lambda x: x.get("avg_hits1", -1), reverse=True)
(log_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"[summary] {log_dir / 'summary.json'}")
for row in rows[:50]:
    if "avg_hits1" in row:
        print(f"{row['avg_hits1']:.6f}\tl2r={row['l2r']:.4f}\tr2l={row['r2l']:.4f}\trho={row.get('rho','')}\t{row['name']}")
    else:
        print(f"NA\t{row['status']}\t{row['name']}")
PY

exit "$failed"
