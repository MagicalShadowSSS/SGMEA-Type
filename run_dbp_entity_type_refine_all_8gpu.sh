#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-1}"
MAX_CASES_PER_SHARD="${MAX_CASES_PER_SHARD:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.70}"

cd "${ROOT_DIR}"

for lang_pair in zh_en ja_en fr_en; do
  echo "[$(date '+%F %T')] START ${lang_pair}"
  LANG_PAIR="${lang_pair}" \
  GPU_IDS="${GPU_IDS}" \
  RESUME="${RESUME}" \
  MAX_CASES_PER_SHARD="${MAX_CASES_PER_SHARD}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  MIN_CONFIDENCE="${MIN_CONFIDENCE}" \
  bash "${ROOT_DIR}/run_dbp_entity_type_refine_8gpu.sh"
  echo "[$(date '+%F %T')] DONE ${lang_pair}"
done
