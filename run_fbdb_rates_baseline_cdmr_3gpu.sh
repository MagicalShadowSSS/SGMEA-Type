#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
DATA_CHOICE=${DATA_CHOICE:-FBDB15K}
DATA_SPLIT=${DATA_SPLIT:-norm}
GPU_IDS=${GPU_IDS:-1,2,3}
RATES=${RATES:-0.2,0.5,0.8}
RUN_TAG=${RUN_TAG:-fbdb_rates_cdmr_$(date +%m%d_%H%M%S)}
TYPE_JSONL_FIXED=${TYPE_JSONL_FIXED:-$DATA_ROOT/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl}
PRIOR_JSON=${PRIOR_JSON:-$DATA_ROOT/SGMEA/train_relation_hnm/type_modality_oracle_4seg_refine_0510_193727.json}
PRIOR_KEY=${PRIOR_KEY:-combined_greedy.best.scales}

BASE_EPOCH=${BASE_EPOCH:-500}
BASE_LR=${BASE_LR:-5e-4}
BASE_BATCH_SIZE=${BASE_BATCH_SIZE:-2048}
BASE_SCHEDULER=${BASE_SCHEDULER:-cos}
BASE_CSLS_K=${BASE_CSLS_K:-3}

CDMR_EPOCH=${CDMR_EPOCH:-40}
CDMR_LR=${CDMR_LR:-3e-4}
CDMR_BATCH_SIZE=${CDMR_BATCH_SIZE:-2048}
CDMR_SCHEDULER=${CDMR_SCHEDULER:-fixed}

EVAL_EPOCH=${EVAL_EPOCH:-1}
WORKERS=${WORKERS:-8}
CDMR_CSLS_K=${CDMR_CSLS_K:-1}
SEED=${SEED:-42}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/fbdb_rates_baseline_cdmr/$RUN_TAG"
mkdir -p "$LOG_DIR"

if [[ ! -f "$TYPE_JSONL_FIXED" ]]; then
  echo "[error] missing type jsonl: $TYPE_JSONL_FIXED" >&2
  exit 1
fi
if [[ ! -f "$PRIOR_JSON" ]]; then
  echo "[error] missing prior json: $PRIOR_JSON" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
IFS=',' read -r -a RATE_LIST <<< "$RATES"

if (( ${#GPUS[@]} < ${#RATE_LIST[@]} )); then
  echo "[warn] fewer GPUs than rates; queues will wrap by index"
fi

run_baseline() {
  local gpu="$1"
  local rate="$2"
  local exp_suffix="baseline_warm_${RUN_TAG}_r${rate//./}"
  local exp_id="$exp_suffix"
  local log_file="$LOG_DIR/baseline_r${rate//./}_gpu${gpu}.log"
  echo "[baseline] gpu=$gpu rate=$rate log=$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" /root/anaconda3/envs/sgmea/bin/python main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_choice "$DATA_CHOICE" \
    --data_split "$DATA_SPLIT" \
    --data_rate "$rate" \
    --model_name SGMEA \
    --exp_id "$exp_id" \
    --epoch "$BASE_EPOCH" \
    --eval_epoch "$EVAL_EPOCH" \
    --lr "$BASE_LR" \
    --hidden_units "300,300,300" \
    --save_model 1 \
    --batch_size "$BASE_BATCH_SIZE" \
    --csls \
    --csls_k "$BASE_CSLS_K" \
    --random_seed "$SEED" \
    --workers "$WORKERS" \
    --dist 0 \
    --accumulation_steps 1 \
    --scheduler "$BASE_SCHEDULER" \
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
    --external_anchor_type_jsonl "$TYPE_JSONL_FIXED" \
    > "$log_file" 2>&1
}

run_cdmr() {
  local gpu="$1"
  local rate="$2"
  local base_ckpt="SGMEA_${DATA_CHOICE}_${rate}_baseline_warm_${RUN_TAG}_r${rate//./}_"
  local exp_suffix="cdmr_prior_${RUN_TAG}_r${rate//./}"
  local log_file="$LOG_DIR/cdmr_r${rate//./}_gpu${gpu}.log"
  local ckpt_path="$DATA_ROOT/SGMEA/save/${base_ckpt}.pkl"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[error] missing baseline checkpoint for rate=$rate: $ckpt_path" | tee "$log_file" >&2
    return 1
  fi
  echo "[cdmr] gpu=$gpu rate=$rate base=$base_ckpt log=$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" /root/anaconda3/envs/sgmea/bin/python main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_choice "$DATA_CHOICE" \
    --data_split "$DATA_SPLIT" \
    --data_rate "$rate" \
    --model_name SGMEA \
    --model_name_save "$base_ckpt" \
    --exp_id "$exp_suffix" \
    --epoch "$CDMR_EPOCH" \
    --eval_epoch "$EVAL_EPOCH" \
    --lr "$CDMR_LR" \
    --hidden_units "300,300,300" \
    --save_model 1 \
    --batch_size "$CDMR_BATCH_SIZE" \
    --csls \
    --csls_k "$CDMR_CSLS_K" \
    --random_seed "$SEED" \
    --workers "$WORKERS" \
    --dist 0 \
    --accumulation_steps 1 \
    --scheduler "$CDMR_SCHEDULER" \
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
    --use_type_modality_bias \
    --type_modality_bias_only_train \
    --freeze_type_modality_bias_entity \
    --type_modality_bias_scale 1.0 \
    --type_modality_bias_l2 0 \
    --type_modality_bias_init_json "$PRIOR_JSON" \
    --type_modality_bias_init_key "$PRIOR_KEY" \
    --use_cdmr_router \
    --cdmr_proj_dim 32 \
    --cdmr_type_emb_dim 16 \
    --cdmr_hidden_dim 64 \
    --cdmr_beta_max 0.5 \
    --cdmr_beta_init 0.10 \
    --cdmr_tau_route 1.0 \
    --cdmr_eval_tau_route 1.0 \
    --cdmr_residual_clamp 0.5 \
    --external_anchor_type_jsonl "$TYPE_JSONL_FIXED" \
    > "$log_file" 2>&1
}

run_phase() {
  local phase="$1"
  local pids=()
  local failed=0
  local idx=0
  for rate in "${RATE_LIST[@]}"; do
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    if [[ "$phase" == "baseline" ]]; then
      run_baseline "$gpu" "$rate" &
    else
      run_cdmr "$gpu" "$rate" &
    fi
    pids+=("$!")
    idx=$((idx + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

echo "[run] tag=$RUN_TAG rates=$RATES gpus=$GPU_IDS logs=$LOG_DIR base_csls_k=$BASE_CSLS_K cdmr_csls_k=$CDMR_CSLS_K"
echo "[phase] baseline"
run_phase baseline
echo "[phase] cdmr"
run_phase cdmr

/root/anaconda3/envs/sgmea/bin/python - "$LOG_DIR" <<'PY'
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
line_re = re.compile(r"Ep\s+(\d+|Test)\s+\|\s+(l2r|r2l): acc of top .*?= \[\s*([0-9.]+)")
rows = []
for path in sorted(log_dir.glob("*.log")):
    vals = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = line_re.search(line)
        if m:
            vals[m.group(2)] = float(m.group(3))
    if {"l2r", "r2l"} <= vals.keys():
        rows.append({
            "name": path.stem,
            "log": str(path),
            "l2r": vals["l2r"],
            "r2l": vals["r2l"],
            "avg_hits1": (vals["l2r"] + vals["r2l"]) / 2.0,
        })
rows.sort(key=lambda x: x["name"])
(log_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"[summary] {log_dir / 'summary.json'}")
for row in rows:
    print(f"{row['avg_hits1']:.6f}\tl2r={row['l2r']:.4f}\tr2l={row['r2l']:.4f}\t{row['name']}")
PY

echo "[done] logs: $LOG_DIR"
