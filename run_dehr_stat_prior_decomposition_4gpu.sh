#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
PYTHON=${PYTHON:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RUN_TAG=${RUN_TAG:-dehr_stat_prior_decomp_$(date +%m%d_%H%M%S)}

DATASET=${DATASET:-FBDB15K}
RATE=${RATE:-0.5}
BASE_CKPT=${BASE_CKPT:-SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05_}
TYPE_JSONL=${TYPE_JSONL:-$DATA_ROOT/anchors_nameless/$DATASET/norm_anchor_type_fixed.jsonl}
PRIOR_JSON=${PRIOR_JSON:-$CODE_ROOT/log/audits/statistical_modality_prior/selected_priors/fbdb15k_statistical_prior_best.json}

DEHR_EPOCH=${DEHR_EPOCH:-20}
DEHR_LR=${DEHR_LR:-1e-4}
DEHR_BATCH_SIZE=${DEHR_BATCH_SIZE:-2048}
EVAL_EPOCH=${EVAL_EPOCH:-1}
WORKERS=${WORKERS:-0}
SEED=${SEED:-42}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/dehr_stat_prior_decomposition/$RUN_TAG"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"

common_args=(
  --gpu 0
  --data_path "$DATA_ROOT"
  --data_choice "$DATASET"
  --data_split norm
  --data_rate "$RATE"
  --model_name SGMEA
  --model_name_save "$BASE_CKPT"
  --hidden_units 300,300,300
  --batch_size "$DEHR_BATCH_SIZE"
  --csls
  --csls_k 1
  --random_seed "$SEED"
  --workers "$WORKERS"
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
  --external_anchor_type_jsonl "$TYPE_JSONL"
  --no_tensorboard
)

dehr_common=(
  --epoch "$DEHR_EPOCH"
  --eval_epoch "$EVAL_EPOCH"
  --lr "$DEHR_LR"
  --save_model 0
  --type_modality_bias_only_train
  --use_dehr_router
  --dehr_proj_dim 32
  --dehr_hidden_dim 64
  --dehr_heads 4
  --dehr_layers 1
  --dehr_dropout 0.1
  --dehr_alpha0 1.0
  --dehr_direction_mode constrained
  --dehr_direction_source stat_prior_json
  --dehr_stat_prior_json "$PRIOR_JSON"
  --dehr_stat_prior_key combined_greedy.best.scales
  --dehr_evidence_mix 0
  --dehr_rho_init_values 0.5,0.7,0.9,1.1
  --dehr_rho_max 2.0
  --dehr_kl_weight 0
  --dehr_rho_weight 0
  --dehr_bias_clamp 0.5
)

run_one() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log_file="$LOG_DIR/${name}_gpu${gpu}.log"
  echo "[launch] gpu=$gpu name=$name log=$log_file"
  {
    echo "[start] $(date '+%F %T')"
    echo "[variant] $name"
  } > "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" main.py \
    "${common_args[@]}" \
    --exp_id "${RUN_TAG}_${name}" \
    "$@" >> "$log_file" 2>&1
  echo "[exit] $(date '+%F %T') status=$?" >> "$log_file"
}

pids=()
failed=0

run_one "${GPUS[0]}" baseline_onlytest --only_test 1 &
pids+=("$!")

run_one "${GPUS[1]}" prior_global_full \
  "${dehr_common[@]}" \
  --dehr_stat_prior_global mean \
  --dehr_global_direction_weight 1.0 \
  --dehr_type_direction_residual_scale 0.0 \
  --dehr_type_direction_residual_max 0.0 &
pids+=("$!")

run_one "${GPUS[2]}" prior_type_only \
  "${dehr_common[@]}" \
  --dehr_stat_prior_global zero \
  --dehr_global_direction_weight 0.0 \
  --dehr_stat_prior_init_type_residual \
  --dehr_type_direction_residual_scale 0.7 \
  --dehr_type_direction_residual_max 0.7 \
  --dehr_type_direction_residual_l2_weight 1e-4 &
pids+=("$!")

run_one "${GPUS[3]}" prior_half_global_type \
  "${dehr_common[@]}" \
  --dehr_stat_prior_global mean \
  --dehr_global_direction_weight 0.5 \
  --dehr_stat_prior_init_type_residual \
  --dehr_type_direction_residual_scale 0.7 \
  --dehr_type_direction_residual_max 0.7 \
  --dehr_type_direction_residual_l2_weight 1e-4 &
pids+=("$!")

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

pids=()

run_one "${GPUS[0]}" prior_type_anchor_residual \
  "${dehr_common[@]}" \
  --dehr_direction_mode anchor_residual \
  --dehr_stat_prior_global zero \
  --dehr_global_direction_weight 0.0 \
  --dehr_stat_prior_init_type_residual \
  --dehr_type_direction_residual_scale 0.7 \
  --dehr_type_direction_residual_max 0.7 \
  --dehr_type_direction_residual_l2_weight 1e-4 \
  --dehr_anchor_scale 0.85 \
  --dehr_residual_gamma 0.15 \
  --dehr_residual_form additive \
  --dehr_residual_confidence 1 \
  --dehr_residual_l2_weight 1e-4 &
pids+=("$!")

run_one "${GPUS[1]}" prior_quarter_global_type \
  "${dehr_common[@]}" \
  --dehr_stat_prior_global mean \
  --dehr_global_direction_weight 0.25 \
  --dehr_stat_prior_init_type_residual \
  --dehr_type_direction_residual_scale 0.7 \
  --dehr_type_direction_residual_max 0.7 \
  --dehr_type_direction_residual_l2_weight 1e-4 &
pids+=("$!")

run_one "${GPUS[2]}" manual_global_reference \
  "${dehr_common[@]}" \
  --dehr_direction_source manual \
  --dehr_direction_values=-0.35,0.15,-0.10,0.30 \
  --dehr_stat_prior_global mean \
  --dehr_global_direction_weight 1.0 \
  --dehr_type_direction_residual_scale 0.0 \
  --dehr_type_direction_residual_max 0.0 &
pids+=("$!")

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

"$PYTHON" - "$LOG_DIR" <<'PY'
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
metric_re = re.compile(r"Ep\s+(\d+|Test)\s+\|\s+(l2r|r2l): acc of top .*?= \[\s*([0-9.]+)")
direction_re = re.compile(r"DEHRStatPriorDirection .*?global=([0-9.,+-]+).*?residual_abs_mean=([0-9.]+)")
kv_re = re.compile(r"([A-Za-z_]+)=([0-9.+-]+)")

rows = []
for path in sorted(log_dir.glob("*.log")):
    vals = {}
    router = {}
    direction = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = metric_re.search(line)
        if m:
            vals[m.group(2)] = float(m.group(3))
        if "DEHRRouterTable" in line:
            kv = dict(kv_re.findall(line))
            router = {
                "bias_abs_mean": float(kv.get("bias_abs_mean", 0.0)),
                "global_bias_abs_mean": float(kv.get("global_bias_abs_mean", 0.0)),
                "type_bias_abs_mean": float(kv.get("type_bias_abs_mean", 0.0)),
                "type_bias_fraction": float(kv.get("type_bias_fraction", 0.0)),
                "evidence_bias_abs_mean": float(kv.get("evidence_bias_abs_mean", 0.0)),
                "evidence_bias_fraction": float(kv.get("evidence_bias_fraction", 0.0)),
            }
        d = direction_re.search(line)
        if d:
            direction = {"global_direction": d.group(1), "residual_abs_mean": float(d.group(2))}
    row = {"name": path.stem.rsplit("_gpu", 1)[0], "log": str(path)}
    if {"l2r", "r2l"} <= vals.keys():
        row.update({"l2r": vals["l2r"], "r2l": vals["r2l"], "avg_hits1": (vals["l2r"] + vals["r2l"]) / 2.0})
    else:
        row["status"] = "missing_metric"
    row.update(router)
    row.update(direction)
    rows.append(row)

rows.sort(key=lambda x: x.get("avg_hits1", -1), reverse=True)
baseline = next((r for r in rows if r["name"] == "baseline_onlytest" and "avg_hits1" in r), None)
if baseline is not None:
    base = baseline["avg_hits1"]
    for row in rows:
        if "avg_hits1" in row:
            row["gain_vs_baseline"] = row["avg_hits1"] - base
global_only = next((r for r in rows if r["name"] == "prior_global_full" and "avg_hits1" in r), None)
if baseline is not None and global_only is not None:
    global_gain = global_only["avg_hits1"] - baseline["avg_hits1"]
    for row in rows:
        if "gain_vs_baseline" in row and row["gain_vs_baseline"] > 0:
            if row.get("global_bias_abs_mean", 0.0) <= 1e-8:
                row["global_calibration_gain_fraction"] = 0.0
            else:
                row["global_calibration_gain_fraction"] = global_gain / row["gain_vs_baseline"]

out = log_dir / "summary.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"[summary] {out}")
for row in rows:
    if "avg_hits1" not in row:
        print(f"NA\t{row.get('status')}\t{row['name']}")
        continue
    bits = [
        f"{row['avg_hits1']:.6f}",
        f"gain={row.get('gain_vs_baseline', 0.0):+.6f}",
        f"l2r={row['l2r']:.4f}",
        f"r2l={row['r2l']:.4f}",
    ]
    if "global_calibration_gain_fraction" in row:
        bits.append(f"global_calib_gain_frac={row['global_calibration_gain_fraction']:.3f}")
    if "global_bias_abs_mean" in row:
        bits.append(f"gb={row['global_bias_abs_mean']:.4f}")
        bits.append(f"tb={row['type_bias_abs_mean']:.4f}")
        bits.append(f"eb={row.get('evidence_bias_abs_mean', 0.0):.4f}")
        bits.append(f"type_frac={row.get('type_bias_fraction', 0.0):.3f}")
    bits.append(row["name"])
    print("\t".join(bits))
PY

exit "$failed"
