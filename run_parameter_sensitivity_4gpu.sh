#!/usr/bin/env bash
set -euo pipefail

# Parameter sensitivity under the fixed-checkpoint protocol.
# Datasets: FB15K-DB15K and FB15K-YG15K, 50% seed alignment.
# Curves:
#   1) TCMS beta on the fixed TCMS checkpoint, inference only.
#   2) DEHR learned bias scale on the fixed TCMS checkpoint, router calibration only.

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
PYTHON=${PYTHON:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
GPU_IDS=${GPU_IDS:-0,1,4,5,6,7}
RUN_TAG=${RUN_TAG:-phase26_parameter_sensitivity_$(date +%m%d_%H%M%S)}

cd "$CODE_ROOT"

LOG_DIR="$CODE_ROOT/log/run_outputs/$RUN_TAG"
SAVE_DIR="$DATA_ROOT/SGMEA/save"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 1
fi

anchor_for() {
  local dataset=$1
  echo "$DATA_ROOT/anchors_nameless/$dataset/norm_anchor_type_fixed.jsonl"
}

tcms_ckpt_for() {
  local dataset=$1
  case "$dataset" in
    FBDB15K) echo "SGMEA_FBDB15K_0.5_fbdb_r05_keep_repro_seed42_" ;;
    FBYG15K) echo "SGMEA_FBYG15K_0.5_fbyg_r05_tcms_keep_ckpt_" ;;
    *) echo "[error] unknown TCMS checkpoint for $dataset" >&2; return 1 ;;
  esac
}

short_dataset() {
  local dataset=$1
  if [[ "$dataset" == "FBDB15K" ]]; then echo "fbdb"; else echo "fbyg"; fi
}

common_eval_args() {
  local dataset=$1 ckpt=$2 exp_id=$3 anchor=$4
  printf '%s ' \
    main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_split norm \
    --data_choice "$dataset" \
    --data_rate 0.5 \
    --model_name SGMEA \
    --model_name_save "$ckpt" \
    --hidden_units 300,300,300 \
    --batch_size 2048 \
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
    --no_tensorboard \
    --external_anchor_type_jsonl "$anchor" \
    --exp_name "$RUN_TAG" \
    --exp_id "$exp_id"
}

tcms_args() {
  local beta=$1
  printf '%s ' \
    --use_tcms \
    --tcms_prefusion \
    --tcms_zero_init_residual \
    --tcms_noise_same_type_ratio 0.7 \
    --tcms_disable_gate_match_features \
    --tcms_beta "$beta" \
    --tcms_beta_warmup_start -1 \
    --tcms_beta_warmup_end 150 \
    --tcms_pretrain_loss_weight 0.002 \
    --tcms_selfsup_start 100 \
    --tcms_selfsup_warmup_end 170 \
    --tcms_sparse_weight 0.03 \
    --tcms_gate_init -3.0 \
    --tcms_missing_mode keep
}

dehr_args() {
  local scale=$1
  printf '%s ' \
    --epoch 0 \
    --eval_epoch 1 \
    --save_model 0 \
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
    --dehr_learned_bias_scale "$scale" \
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
    --dehr_calib_consensus_min_votes 2 \
    --lr 3e-4
}

launch() {
  local gpu=$1 log=$2
  shift 2
  echo "[launch] gpu=$gpu log=$log"
  (
    echo "[start] $(date '+%F %T') gpu=$gpu"
    echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $*"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u $*
    status=$?
    echo "[exit] $(date '+%F %T') status=$status"
    exit "$status"
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

for dataset in FBDB15K FBYG15K; do
  anchor=$(anchor_for "$dataset")
  ckpt=$(tcms_ckpt_for "$dataset")
  [[ -f "$anchor" ]] || { echo "[error] missing anchor $anchor" >&2; exit 1; }
  [[ -f "$SAVE_DIR/${ckpt}.pkl" ]] || { echo "[error] missing checkpoint $SAVE_DIR/${ckpt}.pkl" >&2; exit 1; }
done

echo "[info] run_tag=$RUN_TAG"
echo "[info] log_dir=$LOG_DIR"
echo "[info] gpu_ids=$GPU_IDS"

PIDS=()
gpu_idx=0

for dataset in FBDB15K FBYG15K; do
  short=$(short_dataset "$dataset")
  anchor=$(anchor_for "$dataset")
  ckpt=$(tcms_ckpt_for "$dataset")
  for beta in 0.00 0.10 0.25 0.40; do
    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    common=$(common_eval_args "$dataset" "$ckpt" "sens_${short}_beta_${beta}" "$anchor")
    extra=$(tcms_args "$beta")
    launch "$gpu" "$LOG_DIR/${short}_beta_${beta}.log" $common --only_test 1 --save_model 0 $extra
    if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then wait_batch; fi
  done
done

if (( ${#PIDS[@]} > 0 )); then wait_batch; fi

for dataset in FBDB15K FBYG15K; do
  short=$(short_dataset "$dataset")
  anchor=$(anchor_for "$dataset")
  ckpt=$(tcms_ckpt_for "$dataset")
  for scale in 0.00 0.20 0.35 0.50; do
    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    common=$(common_eval_args "$dataset" "$ckpt" "sens_${short}_dehr_scale_${scale}" "$anchor")
    tcms_extra=$(tcms_args 0.25)
    dehr_extra=$(dehr_args "$scale")
    launch "$gpu" "$LOG_DIR/${short}_dehr_scale_${scale}.log" $common $tcms_extra $dehr_extra
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
probe_re = re.compile(r"RouterProbe \| ([^\n]+)")
rows = []
for log in sorted(log_dir.glob("*.log")):
    text = log.read_text(errors="ignore")
    last = {}
    for direction, hstr, mrr in metric_re.findall(text):
        hits = [float(x) for x in re.findall(r"[0-9.]+", hstr)]
        last[direction] = {"h1": hits[0], "h10": hits[1], "mrr": float(mrr)}
    stem = log.stem
    m_beta = re.match(r"(fbdb|fbyg)_beta_([0-9.]+)$", stem)
    m_scale = re.match(r"(fbdb|fbyg)_dehr_scale_([0-9.]+)$", stem)
    if m_beta:
        dataset, value = m_beta.groups()
        kind = "TCMS beta"
    elif m_scale:
        dataset, value = m_scale.groups()
        kind = "DEHR scale"
    else:
        dataset, value, kind = stem, "", "unknown"
    if "l2r" not in last or "r2l" not in last:
        rows.append({"log": log.name, "dataset": dataset, "kind": kind, "value": value, "status": "missing_metrics"})
        continue
    probes = probe_re.findall(text)
    rows.append({
        "log": log.name,
        "dataset": dataset,
        "kind": kind,
        "value": float(value),
        "h1_avg": round((last["l2r"]["h1"] + last["r2l"]["h1"]) / 2, 6),
        "h10_avg": round((last["l2r"]["h10"] + last["r2l"]["h10"]) / 2, 6),
        "mrr_avg": round((last["l2r"]["mrr"] + last["r2l"]["mrr"]) / 2, 6),
        "probe": probes[-1] if probes else "",
        "status": "ok",
    })

summary = {"rows": rows, "avg_by_param": []}
for kind in sorted({r["kind"] for r in rows}):
    values = sorted({r["value"] for r in rows if r.get("status") == "ok" and r["kind"] == kind})
    for value in values:
        subset = [r for r in rows if r.get("status") == "ok" and r["kind"] == kind and r["value"] == value]
        if not subset:
            continue
        summary["avg_by_param"].append({
            "kind": kind,
            "value": value,
            "h1_avg": round(sum(r["h1_avg"] for r in subset) / len(subset), 6),
            "h10_avg": round(sum(r["h10_avg"] for r in subset) / len(subset), 6),
            "mrr_avg": round(sum(r["mrr_avg"] for r in subset) / len(subset), 6),
        })

(log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[summary] {log_dir / 'summary.json'}")
print("kind\tvalue\tdataset\th1\th10\tmrr")
for r in rows:
    if r.get("status") == "ok":
        print(f"{r['kind']}\t{r['value']:.2f}\t{r['dataset']}\t{r['h1_avg']:.5f}\t{r['h10_avg']:.5f}\t{r['mrr_avg']:.5f}")
    else:
        print(f"{r['kind']}\t{r['value']}\t{r['dataset']}\t{r['status']}")
print("avg")
for r in summary["avg_by_param"]:
    print(f"{r['kind']}\t{r['value']:.2f}\tavg\t{r['h1_avg']:.5f}\t{r['h10_avg']:.5f}\t{r['mrr_avg']:.5f}")
PY

echo "[done] $LOG_DIR"
