#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
PYTHON_BIN="${PYTHON_BIN:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

MODEL_PATH="${MODEL_PATH:-/gly/tongqiang/lxwlxwlxw/HomeBenchReproduction/models/llama3-8b-Instruct}"
MODEL_NAME="${MODEL_NAME:-llama3_8b_instruct}"
LANG_PAIR="${LANG_PAIR:-zh_en}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_CASES_PER_SHARD="${MAX_CASES_PER_SHARD:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.70}"
RESUME="${RESUME:-1}"
INCLUDE_DEFAULT_TARGET="${INCLUDE_DEFAULT_TARGET:-1}"
SHARD_START_STAGGER_SECONDS="${SHARD_START_STAGGER_SECONDS:-2}"

INPUT_JSONL="${INPUT_JSONL:-${ROOT_DIR}/entity_type/dbp_${LANG_PAIR}_type_fixed.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/entity_type/llm_refine_${LANG_PAIR}}"
FINAL_LABELS="${FINAL_LABELS:-${ROOT_DIR}/entity_type/dbp_${LANG_PAIR}_type_refined.labels.jsonl}"
FINAL_JSONL="${FINAL_JSONL:-${ROOT_DIR}/entity_type/dbp_${LANG_PAIR}_type_refined.jsonl}"
FINAL_SUMMARY="${FINAL_SUMMARY:-${ROOT_DIR}/entity_type/dbp_${LANG_PAIR}_type_refined_summary.json}"

mkdir -p "${OUT_DIR}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"

echo "[INFO] lang_pair=${LANG_PAIR}"
echo "[INFO] input=${INPUT_JSONL}"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] gpus=${GPU_IDS} shards=${NUM_SHARDS}"
echo "[INFO] out_dir=${OUT_DIR}"

pids=()
for shard_id in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$shard_id]}"
  shard_labels="${OUT_DIR}/shard_${shard_id}.labels.jsonl"
  shard_jsonl="${OUT_DIR}/shard_${shard_id}.jsonl"
  shard_summary="${OUT_DIR}/shard_${shard_id}.summary.json"
  log_file="${OUT_DIR}/shard_${shard_id}.gpu${gpu}.log"
  resume_flag=()
  if [[ "${RESUME}" == "1" ]]; then
    resume_flag=(--resume)
  fi
  target_flag=()
  if [[ "${INCLUDE_DEFAULT_TARGET}" != "1" ]]; then
    target_flag=(--exclude_default_target)
  fi
  echo "[START] shard=${shard_id}/${NUM_SHARDS} gpu=${gpu} labels=${shard_labels}"
  cmd=(
    "${PYTHON_BIN}" tools/refine_dbp_entity_types_with_llm.py
    --lang_pair "${LANG_PAIR}"
    --input_jsonl "${INPUT_JSONL}"
    --output_jsonl "${shard_jsonl}"
    --output_labels_jsonl "${shard_labels}"
    --output_summary_json "${shard_summary}"
    --model_path "${MODEL_PATH}"
    --model_name "${MODEL_NAME}"
    --max_cases "${MAX_CASES_PER_SHARD}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --min_confidence "${MIN_CONFIDENCE}"
    --num_shards "${NUM_SHARDS}"
    --shard_id "${shard_id}"
  )
  if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
  fi
  if [[ "${INCLUDE_DEFAULT_TARGET}" != "1" ]]; then
    cmd+=(--exclude_default_target)
  fi
  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  ) >"${log_file}" 2>&1 &
  pid="$!"
  pids+=("${pid}")
  echo "[PID] shard=${shard_id} gpu=${gpu} pid=${pid} log=${log_file}"
  sleep "${SHARD_START_STAGGER_SECONDS}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "[ERROR] shard process failed: pid=${pid}" >&2
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "[ERROR] at least one shard failed; see ${OUT_DIR}/shard_*.gpu*.log" >&2
  exit 1
fi

echo "[INFO] merging shard labels -> ${FINAL_LABELS}"
tmp_labels="${FINAL_LABELS}.tmp"
cat "${OUT_DIR}"/shard_*.labels.jsonl > "${tmp_labels}"
mv "${tmp_labels}" "${FINAL_LABELS}"

echo "[INFO] applying labels -> ${FINAL_JSONL}"
(
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" tools/refine_dbp_entity_types_with_llm.py \
    --lang_pair "${LANG_PAIR}" \
    --input_jsonl "${INPUT_JSONL}" \
    --output_jsonl "${FINAL_JSONL}" \
    --output_labels_jsonl "${FINAL_LABELS}" \
    --output_summary_json "${FINAL_SUMMARY}" \
    --model_name "${MODEL_NAME}" \
    --min_confidence "${MIN_CONFIDENCE}" \
    --apply_only
)

echo "[DONE] final=${FINAL_JSONL} summary=${FINAL_SUMMARY}"
