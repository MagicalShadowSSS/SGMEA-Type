#!/usr/bin/env bash
set -euo pipefail

# One-command rebuild for Phase 5 top5 relation-aware HNM assets.
# Policy:
# - extraction, label audit, cache build, and pretrain audit use one GPU/process
# - LLM labeling is sharded into 4 independent chunks on GPUs 0,1,2,3

CODE_ROOT="${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
DATA_ROOT="${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}"
LLM_PATH="${LLM_PATH:-/gly/tongqiang/lxwlxwlxw/HomeBenchReproduction/models/llama3-8b-Instruct}"
LABSE_PATH="${LABSE_PATH:-/gly/tongqiang/dongyufeng/models/LaBSE}"

export CODE_ROOT
export DATA_ROOT
export LLM_PATH
export LABSE_PATH

export DATA_CHOICE="${DATA_CHOICE:-FBDB15K}"
export DATA_SPLIT="${DATA_SPLIT:-norm}"
export DATA_RATE="${DATA_RATE:-0.5}"
export CKPT_NAME="${CKPT_NAME:-SGMEA_FBDB15K_0.5_rebuild_gpu0_}"
export TYPE_JSONL_FIXED="${TYPE_JSONL_FIXED:-${DATA_ROOT}/anchors_nameless/${DATA_CHOICE}/${DATA_SPLIT}_anchor_type_fixed.jsonl}"
export OUT_DIR="${OUT_DIR:-${DATA_ROOT}/SGMEA/train_relation_hnm}"

export SINGLE_GPU="${SINGLE_GPU:-0}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export LABEL_DISTRIBUTED="${LABEL_DISTRIBUTED:-1}"
export LABEL_MULTI_GPU_MIN_CASES="${LABEL_MULTI_GPU_MIN_CASES:-1}"

export RESUME="${RESUME:-0}"
export RUN_DATA_PIPELINE="${RUN_DATA_PIPELINE:-1}"
export RUN_CACHE_BUILD="${RUN_CACHE_BUILD:-1}"
export RUN_PRETRAIN_AUDIT="${RUN_PRETRAIN_AUDIT:-1}"

export LLM_HNM_TOPN="${LLM_HNM_TOPN:-5}"
export HNM_BUDGET_UNIT="${HNM_BUDGET_UNIT:-anchors}"
export HNM_MIN_CASES="${HNM_MIN_CASES:-200}"
export HNM_MAX_CASES="${HNM_MAX_CASES:-500}"
export HNM_THRESHOLD="${HNM_THRESHOLD:-0.40}"
export HNM_FALLBACK_THRESHOLD="${HNM_FALLBACK_THRESHOLD:-0.35}"
export HNM_WRITE_QUALITY="${HNM_WRITE_QUALITY:-all}"

export LABEL_QUALITY_FILTER="${LABEL_QUALITY_FILTER:-all}"
export LABEL_MAX_CASES="${LABEL_MAX_CASES:-10000000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

export KEEP_STATUSES="${KEEP_STATUSES:-closely_related,safe_negative}"
export KEEP_COARSE_TYPES="${KEEP_COARSE_TYPES:-Person,Place,Creative Work,Organization}"
export CACHE_QUALITY_FILTER="${CACHE_QUALITY_FILTER:-all}"
export CACHE_MIN_CONFIDENCE="${CACHE_MIN_CONFIDENCE:-0.0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUT_DIR}/logs"

STAMP="$(date +%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/logs/rebuild_top5_4gpu_${STAMP}.log}"

echo "============================================================"
echo "[RUN] Phase 5 top5 rebuild, 4-GPU labeling"
echo "[CODE_ROOT] ${CODE_ROOT}"
echo "[DATA_ROOT] ${DATA_ROOT}"
echo "[OUT_DIR] ${OUT_DIR}"
echo "[LLM_PATH] ${LLM_PATH}"
echo "[LABSE_PATH] ${LABSE_PATH}"
echo "[SINGLE_GPU] ${SINGLE_GPU}"
echo "[GPU_IDS] ${GPU_IDS}"
echo "[HNM_THRESHOLD] ${HNM_THRESHOLD}"
echo "[HNM_FALLBACK_THRESHOLD] ${HNM_FALLBACK_THRESHOLD}"
echo "[HNM_WRITE_QUALITY] ${HNM_WRITE_QUALITY}"
echo "[LOG_FILE] ${LOG_FILE}"
echo "============================================================"

exec bash "${CODE_ROOT}/run_fbdb_relation_hnm_full_pipeline.sh" 2>&1 | tee "${LOG_FILE}"
