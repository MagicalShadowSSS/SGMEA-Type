#!/usr/bin/env bash
set -euo pipefail
GPU="$1"
TOPK="$2"
OUT="$3"
cd /gly/tongqiang/dongyufeng/Mode-Test/SGMEA
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda activate sgmea
CUDA_VISIBLE_DEVICES="$GPU" LABSE_PATH=/gly/tongqiang/dongyufeng/models/LaBSE python audit_virtual_edge_offline.py \
  --cache_jsonl /gly/tongqiang/dongyufeng/data/mmkg/SGMEA/train_relation_hnm/relation_aware_hnm_cache_top5.jsonl \
  --output_summary_json "$OUT" \
  --focus_coarse_types "Person,Place,Creative Work,Organization" \
  --min_confidence 0.9 \
  --quality_filter all \
  --max_edges_per_positive "$TOPK" \
  --max_total_edges 0 \
  --sample_limit 0 \
  --gpu 0 \
  --data_choice FBDB15K \
  --data_split norm \
  --data_rate 0.5 \
  --model_name SGMEA \
  --model_name_save SGMEA_FBDB15K_0.5_rebuild_gpu0_ \
  --only_test 1 \
  --hidden_units "300,300,300" \
  --batch_size 2048 \
  --csls \
  --csls_k 3 \
  --random_seed 42 \
  --workers 12 \
  --scheduler cos \
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
  --external_anchor_type_jsonl /gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl
