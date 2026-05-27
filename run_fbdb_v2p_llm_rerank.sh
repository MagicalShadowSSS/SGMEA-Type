#!/usr/bin/env bash
set -euo pipefail

# Minimal direct-LLM semantic rerank pipeline for the current V2-P router line.
# Scope: extract test top-k pairs from the current SGMEA/V2-P setting, optionally
# label them with a local LLM, then run conservative semantic reranking.
# This intentionally does not run the old teacher-student or HNM branches.

ROOT_DIR="${ROOT_DIR:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
DATA_ROOT="${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/SGMEA/llm_rerank_v2p}"
MODEL_PATH="${MODEL_PATH:-/root/code/models/Meta-Llama-3-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-llama3_8b_instruct}"
PYTHON_BIN="${PYTHON_BIN:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EXTRACT_GPU="${EXTRACT_GPU:-0}"

DATA_CHOICE="${DATA_CHOICE:-FBDB15K}"
DATA_RATE="${DATA_RATE:-0.5}"
COARSE_TYPES="${COARSE_TYPES:-Place,Creative Work,Organization}"
TYPE_FILE="${TYPE_FILE:-${DATA_ROOT}/anchors_nameless/${DATA_CHOICE}/norm_anchor_type_fixed.jsonl}"
BASE_MODEL_NAME_SAVE="${BASE_MODEL_NAME_SAVE:-SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05_}"

TOPK="${TOPK:-10}"
SAME_TYPE_ONLY="${SAME_TYPE_ONLY:-1}"
RUN_LABELS="${RUN_LABELS:-0}"
RUN_RERANK="${RUN_RERANK:-1}"
RESUME="${RESUME:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TOTAL_QUERY_COUNT_BY_TYPE="${TOTAL_QUERY_COUNT_BY_TYPE:-Place:1343,Creative Work:1131,Organization:1005}"

mkdir -p "${OUT_DIR}"

TYPE_SLUG="$(echo "${COARSE_TYPES}" | tr ', ' '__' | tr -cd '[:alnum:]_')"
PREFIX="${OUT_DIR}/${DATA_CHOICE}_r${DATA_RATE}_v2p_top${TOPK}_${TYPE_SLUG}"
PAIRS_JSONL="${PAIRS_JSONL:-${PREFIX}_pairs.jsonl}"
PAIRS_SUMMARY_JSON="${PAIRS_SUMMARY_JSON:-${PREFIX}_pairs_summary.json}"
LLM_JSONL="${LLM_JSONL:-${PREFIX}_llm_labels.jsonl}"
LLM_SUMMARY_JSON="${LLM_SUMMARY_JSON:-${PREFIX}_llm_labels_summary.json}"
RERANK_JSONL="${RERANK_JSONL:-${PREFIX}_reranked.jsonl}"
RERANK_SUMMARY_JSON="${RERANK_SUMMARY_JSON:-${PREFIX}_rerank_summary.json}"

COMMON_MODEL_ARGS=(
  --gpu "${EXTRACT_GPU}"
  --data_choice "${DATA_CHOICE}"
  --model_name_save "${BASE_MODEL_NAME_SAVE}"
  --external_anchor_type_jsonl "${TYPE_FILE}"
  --data_path "${DATA_ROOT}"
  --data_split norm
  --data_rate "${DATA_RATE}"
  --model_name SGMEA
  --hidden_units "300,300,300"
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
  --no_tensorboard
  --epoch 0
  --eval_epoch 1
  --save_model 0
  --type_modality_bias_only_train
)

V2P_ROUTER_ARGS=(
  --use_dehr_router
  --dehr_direction_mode learned_type
  --dehr_direction_values=0,0,0,0
  --dehr_global_direction_weight 0.0
  --dehr_type_direction_residual_scale 0.0
  --dehr_type_direction_residual_max 0.0
  --dehr_rho_init_values "0.7,0.7,0.7,0.7,0.7,0.7"
  --dehr_rho_max 1.0
  --dehr_bias_clamp 0.35
  --dehr_learned_bias_scale 0.35
  --dehr_contextual_mix 0.15
  --dehr_contextual_source residual
  --dehr_learned_bias_l2_weight 1e-4
  --dehr_stat_feature_mask none
  --dehr_calib_pretrain_epochs 25
  --dehr_calib_pretrain_lr 5e-3
  --dehr_calib_pretrain_tau 0.05
  --dehr_calib_pretrain_batch_size 256
  --dehr_calib_pretrain_val_ratio 0.2
  --dehr_calib_pretrain_patience 8
  --dehr_calib_bias_l2 1e-4
  --dehr_calib_pair_source train_pseudo
  --dehr_calib_selection final
  --dehr_calib_pseudo_margin 0.04
  --dehr_calib_pseudo_csls_k 1
  --dehr_calib_pseudo_source dropout_consensus
  --dehr_calib_consensus_min_votes 2
  --lr 3e-4
)

echo "[1/3] Extracting V2-P top-${TOPK} pairs -> ${PAIRS_JSONL}"
"${PYTHON_BIN}" "${ROOT_DIR}/extract_test_topk_pairs.py" \
  --output_jsonl "${PAIRS_JSONL}" \
  --output_summary_json "${PAIRS_SUMMARY_JSON}" \
  --topk "${TOPK}" \
  --coarse_types "${COARSE_TYPES}" \
  --same_type_only "${SAME_TYPE_ONLY}" \
  --run_router_calibration 1 \
  "${COMMON_MODEL_ARGS[@]}" \
  "${V2P_ROUTER_ARGS[@]}"

"${PYTHON_BIN}" "${ROOT_DIR}/count_rerank_queries.py" --input_jsonl "${PAIRS_JSONL}"

if [[ "${RUN_LABELS}" == "1" ]]; then
  echo "[2/3] Running local LLM labels -> ${LLM_JSONL}"
  INPUT_JSONL="${PAIRS_JSONL}" \
  OUTPUT_JSONL="${LLM_JSONL}" \
  SUMMARY_JSON="${LLM_SUMMARY_JSON}" \
  ROOT_DIR="${ROOT_DIR}" \
  OUT_DIR="${OUT_DIR}" \
  MODEL_PATH="${MODEL_PATH}" \
  MODEL_NAME="${MODEL_NAME}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  COARSE_TYPES="${COARSE_TYPES}" \
  GPU_IDS="${GPU_IDS}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  RESUME="${RESUME}" \
  bash "${ROOT_DIR}/run_llm_rerank_labels_4gpu.sh"
else
  echo "[2/3] Skipping LLM labeling because RUN_LABELS=${RUN_LABELS}."
fi

if [[ "${RUN_RERANK}" == "1" && -s "${LLM_JSONL}" ]]; then
  echo "[3/3] Running rerank sweep -> ${RERANK_SUMMARY_JSON}"
  "${PYTHON_BIN}" "${ROOT_DIR}/rerank_fbdb_topk.py" \
    --pairs_jsonl "${PAIRS_JSONL}" \
    --llm_jsonl "${LLM_JSONL}" \
    --output_reranked_jsonl "${RERANK_JSONL}" \
    --output_summary_json "${RERANK_SUMMARY_JSON}" \
    --topk_eval 1,10 \
    --unknown_penalty 0.0 \
    --total_query_count_by_type "${TOTAL_QUERY_COUNT_BY_TYPE}" \
    --sweep \
    --sweep_same_entity_bonus 0.02,0.05,0.10 \
    --sweep_safe_negative_penalty -0.03,-0.05,-0.08,-0.10 \
    --sweep_closely_related_penalty 0.0,-0.01 \
    --sweep_lambda_weight 0.5,1.0,1.5
elif [[ "${RUN_RERANK}" == "1" ]]; then
  echo "[3/3] Skipping rerank because ${LLM_JSONL} does not exist or is empty."
else
  echo "[3/3] Skipping rerank because RUN_RERANK=${RUN_RERANK}."
fi

echo "[DONE] Direct LLM rerank branch files are under ${OUT_DIR}"
