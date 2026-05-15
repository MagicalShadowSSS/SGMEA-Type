#!/usr/bin/env bash
set -euo pipefail
PY=/gly/tongqiang/anaconda3/envs/sgmea/bin/python
DATA=/gly/tongqiang/dongyufeng/data/mmkg
TYPE=$DATA/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl
CKPT=SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05_
COMMON=(--gpu 0 --data_path "$DATA" --data_choice FBDB15K --data_split norm --data_rate 0.5 --model_name SGMEA --model_name_save "$CKPT" --hidden_units 300,300,300 --batch_size 2048 --csls --csls_k 1 --random_seed 42 --workers 0 --dist 0 --accumulation_steps 1 --scheduler fixed --attr_dim 300 --img_dim 300 --name_dim 300 --char_dim 300 --hidden_size 300 --tau 0.1 --structure_encoder gat --num_attention_heads 1 --num_hidden_layers 1 --use_surface 0 --use_intermediate 0 --enable_sota --disable_sgmea_guidance --freeze_type_modality_bias_entity --external_anchor_type_jsonl "$TYPE" --no_tensorboard --epoch 20 --eval_epoch 1 --save_model 0 --type_modality_bias_only_train --use_dehr_router --dehr_proj_dim 32 --dehr_hidden_dim 64 --dehr_heads 4 --dehr_layers 1 --dehr_dropout 0.1 --dehr_alpha0 1.0 --dehr_direction_values=0,0,0,0 --dehr_global_direction_weight 0.0 --dehr_type_direction_residual_scale 0.0 --dehr_type_direction_residual_max 0.0 --dehr_rho_init_values 0.5,0.7,0.9,1.1 --dehr_rho_max 2.0 --dehr_bias_clamp 0.5)
run(){ local gpu=$1 name=$2; shift 2; echo "[launch] $name gpu=$gpu"; CUDA_VISIBLE_DEVICES=$gpu "$PY" main.py "${COMMON[@]}" --exp_id "learned_probe_$name" "$@" > "$(dirname "$0")/${name}.log" 2>&1; echo "[exit] $name $?" >> "$(dirname "$0")/${name}.log"; }
run 0 free_zero_lr1e4 --lr 1e-4 --dehr_direction_mode free --dehr_zero_init_head --dehr_kl_weight 0 --dehr_rho_weight 0 --dehr_residual_l2_weight 1e-4 &
run 1 free_zero_lr3e4 --lr 3e-4 --dehr_direction_mode free --dehr_zero_init_head --dehr_kl_weight 0 --dehr_rho_weight 0 --dehr_residual_l2_weight 1e-4 &
run 2 anchor_zero_g05_lr3e4 --lr 3e-4 --dehr_direction_mode anchor_residual --dehr_zero_init_head --dehr_residual_gamma 0.5 --dehr_residual_form additive --dehr_residual_confidence 1 --dehr_residual_l2_weight 1e-4 --dehr_kl_weight 0 --dehr_rho_weight 0 &
run 3 anchor_zero_g10_lr3e4 --lr 3e-4 --dehr_direction_mode anchor_residual --dehr_zero_init_head --dehr_residual_gamma 1.0 --dehr_residual_form additive --dehr_residual_confidence 1 --dehr_residual_l2_weight 1e-4 --dehr_kl_weight 0 --dehr_rho_weight 0 &
wait
python - <<'PY'
import pathlib,re,json
log_dir=pathlib.Path('log/run_outputs/dehr_learned_router/codex_learned_probe_0514')
metric=re.compile(r"Ep\s+(\d+|Test)\s+\|\s+(l2r|r2l): acc of top .*?= \[\s*([0-9.]+)")
kv=re.compile(r"([A-Za-z_]+)=([0-9.+-]+)")
rows=[]
for p in sorted(log_dir.glob('*.log')):
 vals={}; router={}
 for line in p.read_text(errors='ignore').splitlines():
  m=metric.search(line)
  if m: vals[m.group(2)]=float(m.group(3))
  if 'DEHRRouterTable' in line:
   d=dict(kv.findall(line)); router={k:float(d.get(k,0)) for k in ['bias_abs_mean','global_bias_abs_mean','type_bias_abs_mean','evidence_bias_abs_mean','type_bias_fraction','evidence_bias_fraction','res_abs_mean','res_abs_max']}
 row={'name':p.stem,'log':str(p)}; row.update(router)
 if {'l2r','r2l'}<=vals.keys(): row.update(l2r=vals['l2r'],r2l=vals['r2l'],avg=(vals['l2r']+vals['r2l'])/2)
 rows.append(row)
rows.sort(key=lambda r:r.get('avg',-1), reverse=True)
print(json.dumps(rows,indent=2))
PY
