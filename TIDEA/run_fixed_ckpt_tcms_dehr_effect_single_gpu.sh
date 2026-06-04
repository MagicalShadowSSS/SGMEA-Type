#!/usr/bin/env bash
set -euo pipefail

# Single-GPU fixed-checkpoint protocol for TCMS and DEHR ablations.
# Variants per dataset/rate:
#   baseline  : fixed baseline checkpoint
#   tcms      : fixed TCMS checkpoint
#   dehr      : fixed baseline checkpoint + DEHR calibration
#   tcms_dehr : fixed TCMS checkpoint + DEHR calibration

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${CODE_ROOT:-$SCRIPT_DIR}
DATA_PATH=${DATA_PATH:-mmkg}
PYTHON=${PYTHON:-python}
GPU_ID=${GPU_ID:-0}
RUN_TAG=${RUN_TAG:-fixed_ckpt_tcms_dehr_$(date +%m%d_%H%M%S)}

cd "$CODE_ROOT"

if [[ "$DATA_PATH" = /* ]]; then
  DATA_DIR="$DATA_PATH"
else
  DATA_DIR="$CODE_ROOT/../../data/$DATA_PATH"
fi

LOG_DIR="$CODE_ROOT/log/run_outputs/$RUN_TAG"
SAVE_DIR="$DATA_DIR/TIDEA/save"
mkdir -p "$LOG_DIR"

anchor_for() {
  local dataset=$1
  echo "$DATA_DIR/anchors_nameless/$dataset/norm_anchor_type_fixed.jsonl"
}

baseline_ckpt_for() {
  local dataset=$1 rate=$2
  case "$dataset:$rate" in
    FBDB15K:0.2) echo "TIDEA_FBDB15K_0.2_baseline_warm_fbdb_rates_0511_104108_r02_" ;;
    FBDB15K:0.5) echo "TIDEA_FBDB15K_0.5_baseline_warm_fbdb_rates_0511_104108_r05_" ;;
    FBDB15K:0.8) echo "TIDEA_FBDB15K_0.8_baseline_warm_fbdb_rates_0511_104108_r08_" ;;
    FBYG15K:0.2) echo "TIDEA_FBYG15K_0.2_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r02_" ;;
    FBYG15K:0.5) echo "TIDEA_FBYG15K_0.5_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r05_" ;;
    FBYG15K:0.8) echo "TIDEA_FBYG15K_0.8_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r08_" ;;
    *) echo "[error] unknown baseline checkpoint for $dataset rate=$rate" >&2; return 1 ;;
  esac
}

tcms_ckpt_for() {
  local dataset=$1 rate=$2
  case "$dataset:$rate" in
    FBDB15K:0.2) echo "TIDEA_FBDB15K_0.2_fbdb_r02_tcms_keep_ckpt_" ;;
    FBDB15K:0.5) echo "TIDEA_FBDB15K_0.5_fbdb_r05_keep_repro_seed42_" ;;
    FBDB15K:0.8) echo "TIDEA_FBDB15K_0.8_fbdb_r08_tcms_keep_ckpt_" ;;
    FBYG15K:0.2) echo "TIDEA_FBYG15K_0.2_fbyg_r02_tcms_keep_ckpt_" ;;
    FBYG15K:0.5) echo "TIDEA_FBYG15K_0.5_fbyg_r05_tcms_keep_ckpt_" ;;
    FBYG15K:0.8) echo "TIDEA_FBYG15K_0.8_fbyg_r08_tcms_keep_ckpt_" ;;
    *) echo "[error] unknown TCMS checkpoint for $dataset rate=$rate" >&2; return 1 ;;
  esac
}

tag_for() {
  local dataset=$1 rate=$2
  local d
  if [[ "$dataset" == "FBDB15K" ]]; then d="fbdb"; else d="fbyg"; fi
  echo "${d}_r${rate/./}"
}

common_eval_args() {
  local dataset=$1 rate=$2 ckpt=$3 exp_id=$4
  printf '%s ' \
    main.py \
    --gpu 0 \
    --data_path "$DATA_PATH" \
    --data_split norm \
    --model_name TIDEA \
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
    --disable_tidea_guidance \
    --no_tensorboard \
    --data_choice "$dataset" \
    --data_rate "$rate" \
    --external_anchor_type_jsonl "$(anchor_for "$dataset")" \
    --model_name_save "$ckpt" \
    --exp_name "$RUN_TAG" \
    --exp_id "$exp_id"
}

tcms_args() {
  printf '%s ' \
    --use_tcms \
    --tcms_prefusion \
    --tcms_zero_init_residual \
    --tcms_noise_same_type_ratio 0.7 \
    --tcms_disable_gate_match_features \
    --tcms_beta 0.25 \
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
    --dehr_calib_consensus_min_votes 2 \
    --lr 3e-4
}

run_one() {
  local dataset=$1 rate=$2 variant=$3 ckpt=$4
  local tag log args extra
  tag=$(tag_for "$dataset" "$rate")
  log="$LOG_DIR/${tag}_${variant}.log"
  args=$(common_eval_args "$dataset" "$rate" "$ckpt" "${tag}_${variant}")
  extra=""

  case "$variant" in
    baseline) extra="--only_test 1 --save_model 0" ;;
    tcms) extra="--only_test 1 --save_model 0 $(tcms_args)" ;;
    dehr) extra="$(dehr_args)" ;;
    tcms_dehr) extra="$(tcms_args) $(dehr_args)" ;;
    *) echo "[error] unknown variant $variant" >&2; exit 1 ;;
  esac

  echo "[run] ${tag}_${variant} gpu=$GPU_ID log=$log"
  {
    echo "[start] $(date '+%F %T') gpu=$GPU_ID"
    echo "[cmd] CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u $args $extra"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u $args $extra
    echo "[exit] $(date '+%F %T') status=0"
  } > "$log" 2>&1
}

echo "[info] run_tag=$RUN_TAG"
echo "[info] log_dir=$LOG_DIR"
echo "[info] data_path=$DATA_PATH"
echo "[info] gpu_id=$GPU_ID"

for dataset in FBDB15K FBYG15K; do
  for rate in 0.2 0.5 0.8; do
    anchor=$(anchor_for "$dataset")
    baseline_ckpt=$(baseline_ckpt_for "$dataset" "$rate")
    tcms_ckpt=$(tcms_ckpt_for "$dataset" "$rate")
    [[ -f "$anchor" ]] || { echo "[error] missing anchor $anchor" >&2; exit 1; }
    [[ -f "$SAVE_DIR/${baseline_ckpt}.pkl" ]] || { echo "[error] missing checkpoint $SAVE_DIR/${baseline_ckpt}.pkl" >&2; exit 1; }
    [[ -f "$SAVE_DIR/${tcms_ckpt}.pkl" ]] || { echo "[error] missing checkpoint $SAVE_DIR/${tcms_ckpt}.pkl" >&2; exit 1; }
  done
done

for dataset in FBDB15K FBYG15K; do
  for rate in 0.2 0.5 0.8; do
    baseline_ckpt=$(baseline_ckpt_for "$dataset" "$rate")
    tcms_ckpt=$(tcms_ckpt_for "$dataset" "$rate")
    run_one "$dataset" "$rate" baseline "$baseline_ckpt"
    run_one "$dataset" "$rate" tcms "$tcms_ckpt"
    run_one "$dataset" "$rate" dehr "$baseline_ckpt"
    run_one "$dataset" "$rate" tcms_dehr "$tcms_ckpt"
  done
done

"$PYTHON" - <<'PY' "$LOG_DIR"
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
pat = re.compile(r"Ep (?:Test|\d+) \| (l2r|r2l): acc of top \[1, 10, 50\] = \[([^\]]+)\].*?mrr = ([0-9.]+)")
probe_pat = re.compile(r"RouterProbe \| ([^\n]+)")
rows = []
for log in sorted(log_dir.glob("*.log")):
    text = log.read_text(errors="ignore")
    ms = pat.findall(text)
    if len(ms) < 2:
        rows.append({"log": log.name, "status": "missing_metrics"})
        continue
    vals = []
    for direction, hstr, mrr in ms[-2:]:
        vals.append({"direction": direction, "h1": float(hstr.split()[0]), "mrr": float(mrr)})
    name = log.stem
    variant = None
    for suffix in ["tcms_dehr", "baseline", "tcms", "dehr"]:
        if name.endswith("_" + suffix):
            variant = suffix
            tag = name[: -(len(suffix) + 1)]
            break
    if variant is None:
        variant = "unknown"
        tag = name
    probes = probe_pat.findall(text)
    rows.append({
        "tag": tag,
        "variant": variant,
        "h1_l2r": vals[-2]["h1"],
        "h1_r2l": vals[-1]["h1"],
        "h1_avg": round((vals[-2]["h1"] + vals[-1]["h1"]) / 2, 6),
        "mrr_l2r": vals[-2]["mrr"],
        "mrr_r2l": vals[-1]["mrr"],
        "mrr_avg": round((vals[-2]["mrr"] + vals[-1]["mrr"]) / 2, 6),
        "probe": probes[-1] if probes else "",
        "log": log.name,
        "status": "ok",
    })

base = {(r.get("tag"), r.get("variant")): r for r in rows if r.get("status") == "ok"}
for row in rows:
    if row.get("status") != "ok":
        continue
    baseline = base.get((row["tag"], "baseline"))
    if baseline:
        row["h1_gain_pp"] = round((row["h1_avg"] - baseline["h1_avg"]) * 100, 4)
        row["mrr_gain_pp"] = round((row["mrr_avg"] - baseline["mrr_avg"]) * 100, 4)

(log_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[summary] {log_dir / 'summary.json'}")
print("tag\tvariant\th1_avg\th1_gain_pp\tmrr_avg\tmrr_gain_pp")
for row in rows:
    if row.get("status") == "ok":
        print(f"{row['tag']}\t{row['variant']}\t{row['h1_avg']:.5f}\t{row.get('h1_gain_pp', 0):+.3f}\t{row['mrr_avg']:.5f}\t{row.get('mrr_gain_pp', 0):+.3f}")
    else:
        print(f"{row['log']}\t{row['status']}")
PY

echo "[done] $LOG_DIR"
