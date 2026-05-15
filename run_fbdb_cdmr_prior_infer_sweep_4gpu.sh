#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
TYPE_JSONL_FIXED=${TYPE_JSONL_FIXED:-$DATA_ROOT/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RUN_TAG=${RUN_TAG:-cdmr_prior_infer_$(date +%m%d_%H%M%S)}
CKPT=${CKPT:-SGMEA_FBDB15K_0.5_priorinit_lr3e4_b010_m05_t10_h64_p32_c05_l20_cdmr_prior_0511_010310_}
BATCH_SIZE=${BATCH_SIZE:-2048}
WORKERS=${WORKERS:-8}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/cdmr_prior_infer_sweeps/$RUN_TAG"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"

run_one() {
  local gpu="$1"
  local name="$2"
  local csls_mode="$3"
  local csls_k="$4"
  local eval_tau="$5"
  local bias_scale="$6"
  local distance="$7"
  local log_file="$LOG_DIR/${name}_gpu${gpu}.log"
  local -a csls_args=()
  if [[ "$csls_mode" == "csls" ]]; then
    csls_args=(--csls --csls_k "$csls_k")
  fi
  echo "[launch] gpu=$gpu name=$name csls=$csls_mode k=$csls_k eval_tau=$eval_tau bias_scale=$bias_scale distance=$distance log=$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" /root/anaconda3/envs/sgmea/bin/python main.py \
    --gpu 0 \
    --only_test 1 \
    --data_path "$DATA_ROOT" \
    --data_choice FBDB15K \
    --data_split norm \
    --data_rate 0.5 \
    --model_name SGMEA \
    --model_name_save "$CKPT" \
    --exp_id "$name" \
    --hidden_units "300,300,300" \
    --batch_size "$BATCH_SIZE" \
    "${csls_args[@]}" \
    --random_seed 42 \
    --workers "$WORKERS" \
    --scheduler fixed \
    --distance "$distance" \
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
    --freeze_type_modality_bias_entity \
    --type_modality_bias_scale "$bias_scale" \
    --use_cdmr_router \
    --cdmr_proj_dim 32 \
    --cdmr_type_emb_dim 16 \
    --cdmr_hidden_dim 64 \
    --cdmr_beta_max 0.5 \
    --cdmr_beta_init 0.10 \
    --cdmr_tau_route 1.0 \
    --cdmr_eval_tau_route "$eval_tau" \
    --cdmr_residual_clamp 0.5 \
    --external_anchor_type_jsonl "$TYPE_JSONL_FIXED" \
    > "$log_file" 2>&1 &
}

CONFIGS=()

# First isolate CSLS. This is the cleanest ablation and the most likely free gain.
CONFIGS+=("base_nocsls none 0 1.00 1.00 2")
for k in 1 2 3 5 10 20 50; do
  CONFIGS+=("base_csls${k} csls ${k} 1.00 1.00 2")
done

# Local inference-only sharpening/softening around the current checkpoint.
for k in 3 5 10; do
  for tau in 0.90 0.95 1.00 1.05 1.10; do
    for scale in 0.90 1.00 1.10; do
      CONFIGS+=("local_csls${k}_tau${tau}_s${scale} csls ${k} ${tau} ${scale} 2")
    done
  done
done

# Check whether L1 distance unexpectedly helps with the routed embedding.
CONFIGS+=("dist1_nocsls none 0 1.00 1.00 1")
for k in 3 5 10; do
  CONFIGS+=("dist1_csls${k} csls ${k} 1.00 1.00 1")
done

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

/root/anaconda3/envs/sgmea/bin/python - "$LOG_DIR" <<'PY'
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
rows = []
line_re = re.compile(r"Ep\s+(\d+|Test)\s+\|\s+(l2r|r2l): acc of top .*?= \[\s*([0-9.]+)")
for path in sorted(log_dir.glob("*.log")):
    vals = {}
    for line in path.read_text(errors="ignore").splitlines():
        match = line_re.search(line)
        if match:
            vals[match.group(2)] = float(match.group(3))
    if {"l2r", "r2l"} <= vals.keys():
        avg = (vals["l2r"] + vals["r2l"]) / 2.0
        rows.append({"name": path.stem.rsplit("_gpu", 1)[0], "log": str(path), "l2r": vals["l2r"], "r2l": vals["r2l"], "avg_hits1": avg})

rows.sort(key=lambda x: x["avg_hits1"], reverse=True)
out = log_dir / "summary.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"[summary] {out}")
for row in rows[:20]:
    print(f"{row['avg_hits1']:.6f}\tl2r={row['l2r']:.4f}\tr2l={row['r2l']:.4f}\t{row['name']}")
PY

if (( failed != 0 )); then
  echo "[done-with-warnings] logs: $LOG_DIR"
  exit 1
fi
echo "[done] logs: $LOG_DIR"
