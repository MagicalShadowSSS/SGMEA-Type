#!/usr/bin/env bash
set -euo pipefail

# One-click full pipeline for Phase 5 relation-aware HNM on FBDB15K.
# Policy:
# - extraction / cache build / pretrain audit: single GPU or single process
# - labeling: auto switch between single GPU and multi-GPU sharded execution
#
# Full scope:
# 1) extract training-side hardest HNM cases
# 2) run LLM labeling
# 3) audit training-side label quality
# 4) build unified relation-aware HNM cache
# 5) run pretrain-side cache adequacy audit

CODE_ROOT="${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
DATA_ROOT="${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}"
DATA_CHOICE="${DATA_CHOICE:-FBDB15K}"
DATA_SPLIT="${DATA_SPLIT:-norm}"
DATA_RATE="${DATA_RATE:-0.5}"
LLM_PATH="${LLM_PATH:-/gly/tongqiang/lxwlxwlxw/HomeBenchReproduction/models/llama3-8b-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SINGLE_GPU="${SINGLE_GPU:-0}"
RESUME="${RESUME:-0}"

OUT_DIR="${OUT_DIR:-${DATA_ROOT}/SGMEA/train_relation_hnm}"
TYPE_JSONL_FIXED="${TYPE_JSONL_FIXED:-${DATA_ROOT}/anchors_nameless/${DATA_CHOICE}/${DATA_SPLIT}_anchor_type_fixed.jsonl}"
CKPT_NAME="${CKPT_NAME:-SGMEA_FBDB15K_0.5_rebuild_gpu0_}"

RUN_DATA_PIPELINE="${RUN_DATA_PIPELINE:-1}"
RUN_CACHE_BUILD="${RUN_CACHE_BUILD:-1}"
RUN_PRETRAIN_AUDIT="${RUN_PRETRAIN_AUDIT:-1}"

LLM_HNM_TOPN="${LLM_HNM_TOPN:-5}"
HNM_BUDGET_UNIT="${HNM_BUDGET_UNIT:-anchors}"
HNM_MIN_CASES="${HNM_MIN_CASES:-200}"
HNM_MAX_CASES="${HNM_MAX_CASES:-1000}"
HNM_THRESHOLD="${HNM_THRESHOLD:-0.55}"
HNM_FALLBACK_THRESHOLD="${HNM_FALLBACK_THRESHOLD:-0.50}"
HNM_WRITE_QUALITY="${HNM_WRITE_QUALITY:-all}"
LABEL_DISTRIBUTED="${LABEL_DISTRIBUTED:-auto}"
LABEL_MULTI_GPU_MIN_CASES="${LABEL_MULTI_GPU_MIN_CASES:-600}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

KEEP_STATUSES="${KEEP_STATUSES:-closely_related,safe_negative}"
KEEP_COARSE_TYPES="${KEEP_COARSE_TYPES:-Person,Place,Creative Work,Organization}"
CACHE_QUALITY_FILTER="${CACHE_QUALITY_FILTER:-all}"
CACHE_MIN_CONFIDENCE="${CACHE_MIN_CONFIDENCE:-0.0}"
CACHE_DEDUPE_BY="${CACHE_DEDUPE_BY:-anchor_candidate_direction}"

PRETRAIN_MARGIN_BASE="${PRETRAIN_MARGIN_BASE:-0.5}"
PRETRAIN_MARGIN_SWEEP="${PRETRAIN_MARGIN_SWEEP:-0.2,0.5,0.6,0.7,0.8,1.0}"
PRETRAIN_DEGREE_BUCKET_QUANTILES="${PRETRAIN_DEGREE_BUCKET_QUANTILES:-0.25,0.75}"

PERSON_LABELS_JSONL="${PERSON_LABELS_JSONL:-${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_person_train_hnm_labels_top${LLM_HNM_TOPN}.jsonl}"
PLACE_LABELS_JSONL="${PLACE_LABELS_JSONL:-${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_place_train_hnm_labels_top${LLM_HNM_TOPN}.jsonl}"
WORK_LABELS_JSONL="${WORK_LABELS_JSONL:-${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_creative_work_train_hnm_labels_top${LLM_HNM_TOPN}.jsonl}"
ORG_LABELS_JSONL="${ORG_LABELS_JSONL:-${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_organization_train_hnm_labels_top${LLM_HNM_TOPN}.jsonl}"

CACHE_JSONL="${CACHE_JSONL:-${OUT_DIR}/relation_aware_hnm_cache_top${LLM_HNM_TOPN}.jsonl}"
CACHE_SUMMARY_JSON="${CACHE_SUMMARY_JSON:-${OUT_DIR}/relation_aware_hnm_cache_top${LLM_HNM_TOPN}_summary.json}"
PRETRAIN_AUDIT_SUMMARY_JSON="${PRETRAIN_AUDIT_SUMMARY_JSON:-${OUT_DIR}/relation_aware_hnm_pretrain_audit_top${LLM_HNM_TOPN}_summary.json}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

file_ready() {
  local path="$1"
  [[ -f "${path}" && -s "${path}" ]]
}

skip_if_resumed() {
  local label="$1"
  shift
  if [[ "${RESUME}" != "1" ]]; then
    return 1
  fi
  local missing=0
  local path
  for path in "$@"; do
    if ! file_ready "${path}"; then
      missing=1
      break
    fi
  done
  if [[ "${missing}" == "0" ]]; then
    echo "[RESUME] Skip ${label}: outputs already exist."
    local shown
    for shown in "$@"; do
      echo "         ${shown}"
    done
    return 0
  fi
  return 1
}

echo "============================================================"
echo "[PIPELINE] FBDB Relation-Aware HNM Full"
echo "[CODE_ROOT] ${CODE_ROOT}"
echo "[DATA_ROOT] ${DATA_ROOT}"
echo "[DATA_CHOICE] ${DATA_CHOICE}"
echo "[DATA_SPLIT] ${DATA_SPLIT}"
echo "[DATA_RATE] ${DATA_RATE}"
echo "[CKPT_NAME] ${CKPT_NAME}"
echo "[TYPE_JSONL_FIXED] ${TYPE_JSONL_FIXED}"
echo "[OUT_DIR] ${OUT_DIR}"
echo "[LLM_PATH] ${LLM_PATH}"
echo "[GPU_IDS] ${GPU_IDS}"
echo "[SINGLE_GPU] ${SINGLE_GPU}"
echo "[LLM_HNM_TOPN] ${LLM_HNM_TOPN}"
echo "[HNM_BUDGET_UNIT] ${HNM_BUDGET_UNIT}"
echo "[HNM_MIN_CASES] ${HNM_MIN_CASES}"
echo "[HNM_MAX_CASES] ${HNM_MAX_CASES}"
echo "[HNM_THRESHOLD] ${HNM_THRESHOLD}"
echo "[HNM_FALLBACK_THRESHOLD] ${HNM_FALLBACK_THRESHOLD}"
echo "[HNM_WRITE_QUALITY] ${HNM_WRITE_QUALITY}"
echo "[LABEL_DISTRIBUTED] ${LABEL_DISTRIBUTED}"
echo "[LABEL_MULTI_GPU_MIN_CASES] ${LABEL_MULTI_GPU_MIN_CASES}"
echo "[KEEP_STATUSES] ${KEEP_STATUSES}"
echo "[KEEP_COARSE_TYPES] ${KEEP_COARSE_TYPES}"
echo "[PRETRAIN_MARGIN_BASE] ${PRETRAIN_MARGIN_BASE}"
echo "[PRETRAIN_MARGIN_SWEEP] ${PRETRAIN_MARGIN_SWEEP}"
echo "============================================================"

if [[ "${RUN_DATA_PIPELINE}" == "1" ]]; then
  echo "[STAGE 1] Run relation-aware HNM data pipeline"
  CODE_ROOT="${CODE_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  DATA_CHOICE="${DATA_CHOICE}" \
  DATA_SPLIT="${DATA_SPLIT}" \
  DATA_RATE="${DATA_RATE}" \
  LLM_PATH="${LLM_PATH}" \
  GPU_IDS="${GPU_IDS}" \
  SINGLE_GPU="${SINGLE_GPU}" \
  RESUME="${RESUME}" \
  TYPE_JSONL_FIXED="${TYPE_JSONL_FIXED}" \
  OUT_DIR="${OUT_DIR}" \
  CKPT_NAME="${CKPT_NAME}" \
  LLM_HNM_TOPN="${LLM_HNM_TOPN}" \
  HNM_BUDGET_UNIT="${HNM_BUDGET_UNIT}" \
  HNM_MIN_CASES="${HNM_MIN_CASES}" \
  HNM_MAX_CASES="${HNM_MAX_CASES}" \
  HNM_THRESHOLD="${HNM_THRESHOLD}" \
  HNM_FALLBACK_THRESHOLD="${HNM_FALLBACK_THRESHOLD}" \
  HNM_WRITE_QUALITY="${HNM_WRITE_QUALITY}" \
  LABEL_DISTRIBUTED="${LABEL_DISTRIBUTED}" \
  LABEL_MULTI_GPU_MIN_CASES="${LABEL_MULTI_GPU_MIN_CASES}" \
  LABEL_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  bash "${CODE_ROOT}/run_fbdb_relation_hnm_data_pipeline.sh"
fi

if [[ "${RUN_CACHE_BUILD}" == "1" ]]; then
  if ! skip_if_resumed "[STAGE 2] relation-aware cache build" "${CACHE_JSONL}" "${CACHE_SUMMARY_JSON}"; then
    echo "[STAGE 2] Build unified relation-aware HNM cache"
    python "${CODE_ROOT}/build_relation_aware_hnm_cache.py" \
      --labels_jsonls "${PERSON_LABELS_JSONL},${PLACE_LABELS_JSONL},${WORK_LABELS_JSONL},${ORG_LABELS_JSONL}" \
      --output_jsonl "${CACHE_JSONL}" \
      --output_summary_json "${CACHE_SUMMARY_JSON}" \
      --keep_statuses "${KEEP_STATUSES}" \
      --keep_coarse_types "${KEEP_COARSE_TYPES}" \
      --quality_filter "${CACHE_QUALITY_FILTER}" \
      --min_confidence "${CACHE_MIN_CONFIDENCE}" \
      --dedupe_by "${CACHE_DEDUPE_BY}"
  fi
fi

if [[ "${RUN_PRETRAIN_AUDIT}" == "1" ]]; then
  if ! skip_if_resumed "[STAGE 3] pretrain audit" "${PRETRAIN_AUDIT_SUMMARY_JSON}"; then
    echo "[STAGE 3] Run pretrain-side cache adequacy audit"
    CUDA_VISIBLE_DEVICES="${SINGLE_GPU}" python "${CODE_ROOT}/audit_relation_aware_hnm_pretrain.py" \
      --cache_jsonl "${CACHE_JSONL}" \
      --output_summary_json "${PRETRAIN_AUDIT_SUMMARY_JSON}" \
      --degree_bucket_quantiles "${PRETRAIN_DEGREE_BUCKET_QUANTILES}" \
      --focus_coarse_types "${KEEP_COARSE_TYPES}" \
      --margin_base "${PRETRAIN_MARGIN_BASE}" \
      --margin_base_sweep "${PRETRAIN_MARGIN_SWEEP}" \
      --gpu 0 \
      --data_choice "${DATA_CHOICE}" \
      --data_split "${DATA_SPLIT}" \
      --data_rate "${DATA_RATE}" \
      --model_name SGMEA \
      --model_name_save "${CKPT_NAME}" \
      --only_test 1 \
      --lr 5e-4 \
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
      --external_anchor_type_jsonl "${TYPE_JSONL_FIXED}"
  fi
fi

echo "[DONE] Relation-aware HNM full pipeline finished."
