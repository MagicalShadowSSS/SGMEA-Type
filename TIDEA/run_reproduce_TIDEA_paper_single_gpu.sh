#!/usr/bin/env bash
set -euo pipefail

# One-command paper reproduction for TIDEA on a single GPU.
# Each run loads the fixed TCMS checkpoint, applies DEHR calibration, and
# evaluates the final TCMS+DEHR model reported as "Ours" in the paper.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${CODE_ROOT:-$SCRIPT_DIR}
DATA_PATH=${DATA_PATH:-mmkg}
PYTHON=${PYTHON:-python}
GPU_ID=${GPU_ID:-0}
RUN_TAG=${RUN_TAG:-paper_reproduce_tidea_$(date +%m%d_%H%M%S)}

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

common_args() {
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

tcms_dehr_args() {
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
    --tcms_missing_mode keep \
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
  local dataset=$1 rate=$2
  local tag ckpt log args extra
  tag=$(tag_for "$dataset" "$rate")
  ckpt=$(tcms_ckpt_for "$dataset" "$rate")
  log="$LOG_DIR/${tag}_paper.log"
  args=$(common_args "$dataset" "$rate" "$ckpt" "${tag}_paper")
  extra=$(tcms_dehr_args)

  echo "[run] $tag gpu=$GPU_ID log=$log"
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
echo "[info] chain=TCMS checkpoint -> DEHR calibration -> final evaluation"

for dataset in FBDB15K FBYG15K; do
  for rate in 0.2 0.5 0.8; do
    anchor=$(anchor_for "$dataset")
    ckpt=$(tcms_ckpt_for "$dataset" "$rate")
    [[ -f "$anchor" ]] || { echo "[error] missing anchor $anchor" >&2; exit 1; }
    [[ -f "$SAVE_DIR/${ckpt}.pkl" ]] || { echo "[error] missing checkpoint $SAVE_DIR/${ckpt}.pkl" >&2; exit 1; }
  done
done

for dataset in FBDB15K FBYG15K; do
  for rate in 0.2 0.5 0.8; do
    run_one "$dataset" "$rate"
  done
done

"$PYTHON" - <<'PY' "$LOG_DIR"
import json
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
paper_h1 = {
    "fbdb_r02": 0.592,
    "fbdb_r05": 0.756,
    "fbdb_r08": 0.834,
    "fbyg_r02": 0.693,
    "fbyg_r05": 0.830,
    "fbyg_r08": 0.884,
}
pat = re.compile(r"Ep (?:Test|\d+) \| (l2r|r2l): acc of top \[1, 10, 50\] = \[([^\]]+)\].*?mrr = ([0-9.]+)")
rows = []
for log in sorted(log_dir.glob("*_paper.log")):
    text = log.read_text(errors="ignore")
    ms = pat.findall(text)
    tag = log.stem.removesuffix("_paper")
    if len(ms) < 2:
        rows.append({"tag": tag, "status": "missing_metrics", "log": log.name})
        continue
    vals = []
    for direction, hstr, mrr in ms[-2:]:
        vals.append({"direction": direction, "h1": float(hstr.split()[0]), "mrr": float(mrr)})
    h1_avg = round((vals[-2]["h1"] + vals[-1]["h1"]) / 2, 6)
    mrr_avg = round((vals[-2]["mrr"] + vals[-1]["mrr"]) / 2, 6)
    target = paper_h1.get(tag)
    rows.append({
        "tag": tag,
        "h1_l2r": vals[-2]["h1"],
        "h1_r2l": vals[-1]["h1"],
        "h1_avg": h1_avg,
        "mrr_l2r": vals[-2]["mrr"],
        "mrr_r2l": vals[-1]["mrr"],
        "mrr_avg": mrr_avg,
        "paper_h1": target,
        "h1_diff_pp": None if target is None else round((h1_avg - target) * 100, 4),
        "log": log.name,
        "status": "ok",
    })

(log_dir / "paper_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[summary] {log_dir / 'paper_summary.json'}")
print("tag\tHits@1(avg)\tPaper Hits@1\tDiff(pp)\tMRR(avg)")
for row in rows:
    if row["status"] != "ok":
        print(f"{row['tag']}\t{row['status']}\t\t\t")
        continue
    target = "" if row["paper_h1"] is None else f"{row['paper_h1']:.3f}"
    diff = "" if row["h1_diff_pp"] is None else f"{row['h1_diff_pp']:+.3f}"
    print(f"{row['tag']}\t{row['h1_avg']:.5f}\t{target}\t{diff}\t{row['mrr_avg']:.5f}")
PY

echo "[done] $LOG_DIR"
