#!/usr/bin/env bash
set -euo pipefail

PY=/gly/tongqiang/anaconda3/envs/sgmea/bin/python
DATA=/gly/tongqiang/dongyufeng/data/mmkg
TYPE=$DATA/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl
CKPT=SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05_
OUT=log/run_outputs/dehr_learned_router/codex_learned_type_0514
mkdir -p "$OUT"

COMMON=(
  --data_path "$DATA"
  --data_choice FBDB15K
  --data_split norm
  --data_rate 0.5
  --model_name SGMEA
  --model_name_save "$CKPT"
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
  --external_anchor_type_jsonl "$TYPE"
  --no_tensorboard
  --epoch 20
  --eval_epoch 1
  --save_model 0
  --type_modality_bias_only_train
  --use_dehr_router
  --dehr_direction_mode learned_type
  --dehr_proj_dim 32
  --dehr_hidden_dim 64
  --dehr_heads 4
  --dehr_layers 1
  --dehr_dropout 0.1
  --dehr_alpha0 1.0
  --dehr_direction_values=0,0,0,0
  --dehr_global_direction_weight 0.0
  --dehr_type_direction_residual_scale 0.0
  --dehr_type_direction_residual_max 0.0
  --dehr_rho_init_values 0.5,0.7,0.9,1.1
  --dehr_rho_max 2.0
  --dehr_bias_clamp 0.5
  --dehr_learned_bias_l2_weight 1e-4
  --dehr_aux_tau 0.05
  --dehr_aux_candidate_scope all_right
)

run_one() {
  local gpu=$1
  local name=$2
  shift 2
  echo "[launch] $name gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" main.py --gpu 0 "${COMMON[@]}" --exp_id "$name" "$@" \
    > "$OUT/${name}.log" 2>&1
  echo "[exit] $name $?" >> "$OUT/${name}.log"
}

run_one 0 learned_lr3e4_s05_aux002 --lr 3e-4 --dehr_learned_bias_scale 0.5 --dehr_aux_weight 0.02 &
run_one 1 learned_lr3e4_s07_aux005 --lr 3e-4 --dehr_learned_bias_scale 0.7 --dehr_aux_weight 0.05 &
run_one 2 learned_lr5e4_s07_aux010 --lr 5e-4 --dehr_learned_bias_scale 0.7 --dehr_aux_weight 0.10 &
run_one 3 learned_lr1e3_s05_aux005 --lr 1e-3 --dehr_learned_bias_scale 0.5 --dehr_aux_weight 0.05 &
wait

"$PY" - <<'PY'
import json
import pathlib
import re

out = pathlib.Path("log/run_outputs/dehr_learned_router/codex_learned_type_0514")
metric = re.compile(r"Ep\s+(\d+|Test)\s+\|\s+(l2r|r2l): acc of top .*?= \[\s*([0-9.]+)")
kv = re.compile(r"([A-Za-z_]+)=([0-9.+-]+)")
rows = []
for path in sorted(out.glob("learned_*.log")):
    vals = {}
    router = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = metric.search(line)
        if m:
            vals[m.group(2)] = float(m.group(3))
        if "DEHRRouterTable" in line:
            data = dict(kv.findall(line))
            router = {
                key: float(data.get(key, 0.0))
                for key in [
                    "bias_abs_mean",
                    "global_bias_abs_mean",
                    "type_bias_abs_mean",
                    "type_bias_fraction",
                    "evidence_bias_abs_mean",
                    "evidence_bias_fraction",
                    "res_abs_mean",
                    "res_abs_max",
                ]
            }
    row = {"name": path.stem, "log": str(path)}
    row.update(router)
    if {"l2r", "r2l"} <= vals.keys():
        row.update(l2r=vals["l2r"], r2l=vals["r2l"], avg_hits1=(vals["l2r"] + vals["r2l"]) / 2.0)
    rows.append(row)
rows.sort(key=lambda item: item.get("avg_hits1", -1), reverse=True)
summary_path = out / "summary_learned_type_fbdb.json"
summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY
