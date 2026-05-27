#!/usr/bin/env bash
set -euo pipefail

# Fixed-checkpoint / fixed-seed protocol for measuring two modules:
#   1) TCMS visual sanitization
#   2) learned-DEHR type-aware modality routing
#
# Variants per dataset/rate:
#   baseline       : strict only_test on fixed baseline checkpoint
#   tcms           : strict only_test on fixed TCMS checkpoint, beta forced active
#   dehr           : fixed baseline checkpoint + DEHR calibration only
#   tcms_dehr      : fixed TCMS checkpoint + DEHR calibration only, beta forced active
#
# Important reproducibility guard:
#   TCMS was trained with beta warmup. In pure only_test current_epoch defaults to 0,
#   so beta would be zero if warmup_start=70. We set warmup_start=-1 in all
#   TCMS evaluation / TCMS+DEHR runs to evaluate the saved final sanitizer.

CODE_ROOT=${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}
DATA_ROOT=${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}
PYTHON=${PYTHON:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
RUN_TAG=${RUN_TAG:-phase25_fixed_ckpt_tcms_dehr_$(date +%m%d_%H%M%S)}

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

baseline_ckpt_for() {
  local dataset=$1 rate=$2
  case "$dataset:$rate" in
    FBDB15K:0.2) echo "SGMEA_FBDB15K_0.2_baseline_warm_fbdb_rates_cdmr_0511_104108_r02_" ;;
    FBDB15K:0.5) echo "SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05_" ;;
    FBDB15K:0.8) echo "SGMEA_FBDB15K_0.8_baseline_warm_fbdb_rates_cdmr_0511_104108_r08_" ;;
    FBYG15K:0.2) echo "SGMEA_FBYG15K_0.2_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r02_" ;;
    FBYG15K:0.5) echo "SGMEA_FBYG15K_0.5_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r05_" ;;
    FBYG15K:0.8) echo "SGMEA_FBYG15K_0.8_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r08_" ;;
    *) echo "[error] unknown baseline ckpt for $dataset r=$rate" >&2; return 1 ;;
  esac
}

tcms_ckpt_for() {
  local dataset=$1 rate=$2
  case "$dataset:$rate" in
    FBDB15K:0.2) echo "SGMEA_FBDB15K_0.2_fbdb_r02_tcms_keep_ckpt_" ;;
    # Use the verified high-score fixed checkpoint for FBDB r=0.5.
    FBDB15K:0.5) echo "SGMEA_FBDB15K_0.5_fbdb_r05_keep_repro_seed42_" ;;
    FBDB15K:0.8) echo "SGMEA_FBDB15K_0.8_fbdb_r08_tcms_keep_ckpt_" ;;
    FBYG15K:0.2) echo "SGMEA_FBYG15K_0.2_fbyg_r02_tcms_keep_ckpt_" ;;
    FBYG15K:0.5) echo "SGMEA_FBYG15K_0.5_fbyg_r05_tcms_keep_ckpt_" ;;
    FBYG15K:0.8) echo "SGMEA_FBYG15K_0.8_fbyg_r08_tcms_keep_ckpt_" ;;
    *) echo "[error] unknown TCMS ckpt for $dataset r=$rate" >&2; return 1 ;;
  esac
}

seed_for() {
  # Current fixed protocol: all selected checkpoints are evaluated with seed 42.
  echo 42
}

tag_for() {
  local dataset=$1 rate=$2
  local d
  if [[ "$dataset" == "FBDB15K" ]]; then d="fbdb"; else d="fbyg"; fi
  echo "${d}_r${rate/./}"
}

common_eval_args() {
  local dataset=$1 rate=$2 seed=$3 anchor=$4 ckpt=$5 exp_id=$6
  printf '%s ' \
    main.py \
    --gpu 0 \
    --data_path "$DATA_ROOT" \
    --data_split norm \
    --model_name SGMEA \
    --hidden_units 300,300,300 \
    --batch_size 2048 \
    --csls \
    --csls_k 3 \
    --random_seed "$seed" \
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
    --data_choice "$dataset" \
    --data_rate "$rate" \
    --external_anchor_type_jsonl "$anchor" \
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

echo "[info] run_tag=$RUN_TAG"
echo "[info] log_dir=$LOG_DIR"
echo "[info] gpu_ids=$GPU_IDS"

for dataset in FBDB15K FBYG15K; do
  for rate in 0.2 0.5 0.8; do
    anchor=$(anchor_for "$dataset")
    [[ -f "$anchor" ]] || { echo "[error] missing anchor $anchor" >&2; exit 1; }
    for ckpt in "$(baseline_ckpt_for "$dataset" "$rate")" "$(tcms_ckpt_for "$dataset" "$rate")"; do
      [[ -f "$SAVE_DIR/${ckpt}.pkl" ]] || { echo "[error] missing checkpoint $SAVE_DIR/${ckpt}.pkl" >&2; exit 1; }
    done
  done
done

PIDS=()
gpu_idx=0

for dataset in FBDB15K FBYG15K; do
  for rate in 0.2 0.5 0.8; do
    tag=$(tag_for "$dataset" "$rate")
    seed=$(seed_for "$dataset" "$rate")
    anchor=$(anchor_for "$dataset")
    baseline_ckpt=$(baseline_ckpt_for "$dataset" "$rate")
    tcms_ckpt=$(tcms_ckpt_for "$dataset" "$rate")

    tcms_extra=$(tcms_args)
    dehr_extra=$(dehr_args)

    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    base_common=$(common_eval_args "$dataset" "$rate" "$seed" "$anchor" "$baseline_ckpt" "${tag}_baseline")
    launch "$gpu" "$LOG_DIR/${tag}_baseline.log" $base_common --only_test 1 --save_model 0
    if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then wait_batch; fi

    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    tcms_common=$(common_eval_args "$dataset" "$rate" "$seed" "$anchor" "$tcms_ckpt" "${tag}_tcms")
    launch "$gpu" "$LOG_DIR/${tag}_tcms.log" $tcms_common --only_test 1 --save_model 0 $tcms_extra
    if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then wait_batch; fi

    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    base_common=$(common_eval_args "$dataset" "$rate" "$seed" "$anchor" "$baseline_ckpt" "${tag}_dehr")
    launch "$gpu" "$LOG_DIR/${tag}_dehr.log" $base_common $dehr_extra
    if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then wait_batch; fi

    gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}; gpu_idx=$((gpu_idx + 1))
    tcms_common=$(common_eval_args "$dataset" "$rate" "$seed" "$anchor" "$tcms_ckpt" "${tag}_tcms_dehr")
    launch "$gpu" "$LOG_DIR/${tag}_tcms_dehr.log" $tcms_common $tcms_extra $dehr_extra
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
for r in rows:
    if r.get("status") != "ok":
        continue
    b = base.get((r["tag"], "baseline"))
    if b:
        r["h1_gain_pp"] = round((r["h1_avg"] - b["h1_avg"]) * 100, 4)
        r["mrr_gain_pp"] = round((r["mrr_avg"] - b["mrr_avg"]) * 100, 4)

(log_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[summary] {log_dir / 'summary.json'}")
print("tag\tvariant\th1_avg\th1_gain_pp\tmrr_avg\tmrr_gain_pp")
for r in rows:
    if r.get("status") == "ok":
        print(f"{r['tag']}\t{r['variant']}\t{r['h1_avg']:.5f}\t{r.get('h1_gain_pp', 0):+.3f}\t{r['mrr_avg']:.5f}\t{r.get('mrr_gain_pp', 0):+.3f}")
    else:
        print(f"{r['log']}\t{r['status']}")
PY

echo "[done] $LOG_DIR"
