#!/usr/bin/env bash
set -euo pipefail

# One-click data-construction pipeline for Phase 5 relation-aware HNM.
# Policy:
# - extraction / audit: single GPU
# - labeling: auto switch between single GPU and multi-GPU sharded execution
#
# Scope:
# 1) extract training-side hardest HNM cases for Person / Place / Creative Work / Organization
# 2) run LLM fine-type / relation labeling on those cases
# 3) audit label quality and same_entity confidence
#
# Resume policy:
# - extraction: skip if output jsonl already exists and is non-empty when RESUME=1
# - labeling: skip if output jsonl + summary already exist and are non-empty when RESUME=1
# - audit: skip if summary already exists and is non-empty when RESUME=1

CODE_ROOT="${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
DATA_ROOT="${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}"
DATA_CHOICE="${DATA_CHOICE:-FBDB15K}"
DATA_SPLIT="${DATA_SPLIT:-norm}"
DATA_RATE="${DATA_RATE:-0.5}"
LLM_PATH="${LLM_PATH:-/gly/tongqiang/lxwlxwlxw/HomeBenchReproduction/models/llama3-8b-Instruct}"
SINGLE_GPU="${SINGLE_GPU:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CKPT_NAME="${CKPT_NAME:-SGMEA_FBDB15K_0.5_rebuild_gpu0_}"
RESUME="${RESUME:-0}"

TYPE_JSONL_FIXED="${TYPE_JSONL_FIXED:-${DATA_ROOT}/anchors_nameless/${DATA_CHOICE}/${DATA_SPLIT}_anchor_type_fixed.jsonl}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/SGMEA/train_relation_hnm}"

RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_LABEL="${RUN_LABEL:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"

HNM_THRESHOLD="${HNM_THRESHOLD:-0.55}"
HNM_FALLBACK_THRESHOLD="${HNM_FALLBACK_THRESHOLD:-0.50}"
HNM_MIN_CASES="${HNM_MIN_CASES:-200}"
HNM_MAX_CASES="${HNM_MAX_CASES:-1000}"
HNM_BUDGET_UNIT="${HNM_BUDGET_UNIT:-anchors}"
HNM_ATTR_LIMIT="${HNM_ATTR_LIMIT:-40}"
HNM_MIN_ATTR_ITEMS="${HNM_MIN_ATTR_ITEMS:-3}"
HNM_MIN_COMBINED_ATTR_ITEMS="${HNM_MIN_COMBINED_ATTR_ITEMS:-6}"
HNM_WRITE_QUALITY="${HNM_WRITE_QUALITY:-all}"
HNM_DIRECTION="${HNM_DIRECTION:-both}"
HNM_SORT_BY="${HNM_SORT_BY:-top1_sim_desc}"
HNM_SEED="${HNM_SEED:-42}"
HNM_TYPES="${HNM_TYPES:-Person,Place,Creative Work,Organization}"

LABEL_MODEL_NAME="${LABEL_MODEL_NAME:-llama3_8b_instruct}"
LABEL_QUALITY_FILTER="${LABEL_QUALITY_FILTER:-all}"
LABEL_MAX_CASES="${LABEL_MAX_CASES:-10000000}"
LABEL_MAX_NEW_TOKENS="${LABEL_MAX_NEW_TOKENS:-1024}"
LABEL_DISTRIBUTED="${LABEL_DISTRIBUTED:-auto}"
LABEL_MULTI_GPU_MIN_CASES="${LABEL_MULTI_GPU_MIN_CASES:-600}"
AUDIT_HIGH_CONF="${AUDIT_HIGH_CONF:-0.8}"
LLM_HNM_TOPN="${LLM_HNM_TOPN:-5}"

COMMON_MODEL_ARGS=(
  --gpu 0
  --data_choice "${DATA_CHOICE}"
  --data_split "${DATA_SPLIT}"
  --data_rate "${DATA_RATE}"
  --model_name SGMEA
  --model_name_save "${CKPT_NAME}"
  --only_test 1
  --lr 5e-4
  --hidden_units "300,300,300"
  --batch_size 2048
  --csls
  --csls_k 3
  --random_seed 42
  --workers 12
  --scheduler cos
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
  --external_anchor_type_jsonl "${TYPE_JSONL_FIXED}"
)

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

type_slug_for() {
  case "$1" in
    Person) echo "person" ;;
    Place) echo "place" ;;
    "Creative Work") echo "creative_work" ;;
    Organization) echo "organization" ;;
    *)
      echo "[ERROR] Unsupported HNM type: $1" >&2
      return 1
      ;;
  esac
}

selected_hnm_types=()
IFS=',' read -ra _raw_hnm_types <<< "${HNM_TYPES}"
for raw_type in "${_raw_hnm_types[@]}"; do
  trimmed="$(echo "${raw_type}" | xargs)"
  if [[ -n "${trimmed}" ]]; then
    selected_hnm_types+=("${trimmed}")
  fi
done

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
  local path
  for path in "$@"; do
    if ! file_ready "${path}"; then
      return 1
    fi
  done
  echo "[RESUME] Skip ${label}: outputs already exist."
  for path in "$@"; do
    echo "         ${path}"
  done
  return 0
}

count_jsonl_rows() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return 0
  fi
  wc -l < "${path}" | tr -d '[:space:]'
}

gpu_count() {
  local old_ifs="${IFS}"
  IFS=','
  read -ra gpu_array <<< "${GPU_IDS}"
  IFS="${old_ifs}"
  echo "${#gpu_array[@]}"
}

should_use_multi_gpu_labeling() {
  local input_jsonl="$1"
  local num_gpus
  local row_count
  num_gpus="$(gpu_count)"
  row_count="$(count_jsonl_rows "${input_jsonl}")"

  if [[ "${LABEL_DISTRIBUTED}" == "1" ]]; then
    [[ "${num_gpus}" -gt 1 ]]
    return $?
  fi
  if [[ "${LABEL_DISTRIBUTED}" == "0" ]]; then
    return 1
  fi

  if [[ "${num_gpus}" -le 1 ]]; then
    return 1
  fi
  [[ "${row_count}" -ge "${LABEL_MULTI_GPU_MIN_CASES}" ]]
}

effective_hnm_min_cases() {
  if [[ "${HNM_BUDGET_UNIT}" == "anchors" ]]; then
    echo $(( HNM_MIN_CASES * LLM_HNM_TOPN ))
    return 0
  fi
  echo "${HNM_MIN_CASES}"
}

effective_hnm_max_cases() {
  if [[ "${HNM_BUDGET_UNIT}" == "anchors" ]]; then
    echo $(( HNM_MAX_CASES * LLM_HNM_TOPN ))
    return 0
  fi
  echo "${HNM_MAX_CASES}"
}

extract_cases() {
  local type_name="$1"
  local type_slug="$2"
  local output_jsonl="${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_${type_slug}_train_hnm_cases_top${LLM_HNM_TOPN}.jsonl"
  local effective_min_cases
  local effective_max_cases
  effective_min_cases="$(effective_hnm_min_cases)"
  effective_max_cases="$(effective_hnm_max_cases)"

  if skip_if_resumed "[EXTRACT] ${type_name}" "${output_jsonl}"; then
    return 0
  fi

  echo "[EXTRACT] ${type_name} -> budget_unit=${HNM_BUDGET_UNIT} min_rows=${effective_min_cases} max_rows=${effective_max_cases}"
  CUDA_VISIBLE_DEVICES="${SINGLE_GPU}" python "${CODE_ROOT}/extract_llm_hnm_cases.py" \
    "${COMMON_MODEL_ARGS[@]}" \
    --llm_hnm_type "${type_name}" \
    --llm_hnm_output "${output_jsonl}" \
    --llm_hnm_threshold "${HNM_THRESHOLD}" \
    --llm_hnm_fallback_threshold "${HNM_FALLBACK_THRESHOLD}" \
    --llm_hnm_min_cases "${effective_min_cases}" \
    --llm_hnm_max_cases "${effective_max_cases}" \
    --llm_hnm_attr_limit "${HNM_ATTR_LIMIT}" \
    --llm_hnm_min_attr_items "${HNM_MIN_ATTR_ITEMS}" \
    --llm_hnm_min_combined_attr_items "${HNM_MIN_COMBINED_ATTR_ITEMS}" \
    --llm_hnm_write_quality "${HNM_WRITE_QUALITY}" \
    --llm_hnm_direction "${HNM_DIRECTION}" \
    --llm_hnm_sort_by "${HNM_SORT_BY}" \
    --llm_hnm_seed "${HNM_SEED}" \
    --llm_hnm_topn "${LLM_HNM_TOPN}"
}

label_cases() {
  local type_slug="$1"
  local input_jsonl="${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_${type_slug}_train_hnm_cases_top${LLM_HNM_TOPN}.jsonl"
  local output_jsonl="${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_${type_slug}_train_hnm_labels_top${LLM_HNM_TOPN}.jsonl"
  local output_summary="${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_${type_slug}_train_hnm_labels_top${LLM_HNM_TOPN}_summary.json"
  local resume_flag=()

  if skip_if_resumed "[LABEL] ${type_slug}" "${output_jsonl}" "${output_summary}"; then
    return 0
  fi

  if [[ "${RESUME}" == "1" ]]; then
    resume_flag+=(--resume)
  fi

  echo "[LABEL] ${type_slug}"
  if should_use_multi_gpu_labeling "${input_jsonl}"; then
    local case_count
    case_count="$(count_jsonl_rows "${input_jsonl}")"
    echo "[LABEL] ${type_slug} -> multi-gpu (${GPU_IDS}) because case_count=${case_count}"
    INPUT_JSONL="${input_jsonl}" \
    OUTPUT_JSONL="${output_jsonl}" \
    SUMMARY_JSON="${output_summary}" \
    ROOT_DIR="${CODE_ROOT}" \
    OUT_DIR="${OUT_DIR}" \
    MODEL_PATH="${LLM_PATH}" \
    MODEL_NAME="${LABEL_MODEL_NAME}" \
    GPU_IDS="${GPU_IDS}" \
    QUALITY_FILTER="${LABEL_QUALITY_FILTER}" \
    MAX_CASES="${LABEL_MAX_CASES}" \
    MAX_NEW_TOKENS="${LABEL_MAX_NEW_TOKENS}" \
    RESUME="${RESUME}" \
    bash "${CODE_ROOT}/run_relation_aware_hnm_labels_4gpu.sh"
  else
    local case_count
    case_count="$(count_jsonl_rows "${input_jsonl}")"
    echo "[LABEL] ${type_slug} -> single-gpu (${SINGLE_GPU}) because case_count=${case_count}"
    CUDA_VISIBLE_DEVICES="${SINGLE_GPU}" python "${CODE_ROOT}/build_llm_fine_types.py" \
      --input_jsonl "${input_jsonl}" \
      --output_jsonl "${output_jsonl}" \
      --output_summary_json "${output_summary}" \
      --model_path "${LLM_PATH}" \
      --model_name "${LABEL_MODEL_NAME}" \
      --quality_filter "${LABEL_QUALITY_FILTER}" \
      --max_cases "${LABEL_MAX_CASES}" \
      --max_new_tokens "${LABEL_MAX_NEW_TOKENS}" \
      "${resume_flag[@]}"
  fi
}

audit_labels() {
  local type_slug="$1"
  local cases_jsonl="${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_${type_slug}_train_hnm_cases_top${LLM_HNM_TOPN}.jsonl"
  local labels_jsonl="${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_${type_slug}_train_hnm_labels_top${LLM_HNM_TOPN}.jsonl"
  local summary_json="${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_${type_slug}_train_hnm_audit_top${LLM_HNM_TOPN}_summary.json"
  local same_jsonl="${OUT_DIR}/${DATA_CHOICE}_${DATA_RATE}_${type_slug}_train_hnm_same_entity_top${LLM_HNM_TOPN}.jsonl"

  if skip_if_resumed "[AUDIT] ${type_slug}" "${summary_json}" "${same_jsonl}"; then
    return 0
  fi

  echo "[AUDIT] ${type_slug}"
  CUDA_VISIBLE_DEVICES="${SINGLE_GPU}" python "${CODE_ROOT}/audit_relation_aware_hnm_labels.py" \
    --cases_jsonl "${cases_jsonl}" \
    --labels_jsonl "${labels_jsonl}" \
    --output_summary_json "${summary_json}" \
    --output_same_entity_jsonl "${same_jsonl}" \
    --high_conf_threshold "${AUDIT_HIGH_CONF}"
}

echo "============================================================"
echo "[PIPELINE] Phase 5 Relation-Aware HNM Data"
echo "[CODE_ROOT] ${CODE_ROOT}"
echo "[DATA_ROOT] ${DATA_ROOT}"
echo "[DATA_CHOICE] ${DATA_CHOICE}"
echo "[DATA_SPLIT] ${DATA_SPLIT}"
echo "[DATA_RATE] ${DATA_RATE}"
echo "[CKPT_NAME] ${CKPT_NAME}"
echo "[TYPE_JSONL_FIXED] ${TYPE_JSONL_FIXED}"
echo "[OUT_DIR] ${OUT_DIR}"
echo "[LLM_PATH] ${LLM_PATH}"
echo "[SINGLE_GPU] ${SINGLE_GPU}"
echo "[GPU_IDS] ${GPU_IDS}"
echo "[RESUME] ${RESUME}"
echo "[LLM_HNM_TOPN] ${LLM_HNM_TOPN}"
echo "[HNM_BUDGET_UNIT] ${HNM_BUDGET_UNIT}"
echo "[HNM_MIN_CASES] ${HNM_MIN_CASES}"
echo "[HNM_MAX_CASES] ${HNM_MAX_CASES}"
echo "[EFFECTIVE_HNM_MIN_ROWS] $(effective_hnm_min_cases)"
echo "[EFFECTIVE_HNM_MAX_ROWS] $(effective_hnm_max_cases)"
echo "[HNM_TYPES] ${HNM_TYPES}"
echo "[LABEL_DISTRIBUTED] ${LABEL_DISTRIBUTED}"
echo "[LABEL_MULTI_GPU_MIN_CASES] ${LABEL_MULTI_GPU_MIN_CASES}"
echo "============================================================"

if [[ "${RUN_EXTRACT}" == "1" ]]; then
  for type_name in "${selected_hnm_types[@]}"; do
    type_slug="$(type_slug_for "${type_name}")"
    extract_cases "${type_name}" "${type_slug}"
  done
fi

if [[ "${RUN_LABEL}" == "1" ]]; then
  for type_name in "${selected_hnm_types[@]}"; do
    type_slug="$(type_slug_for "${type_name}")"
    label_cases "${type_slug}"
  done
fi

if [[ "${RUN_AUDIT}" == "1" ]]; then
  for type_name in "${selected_hnm_types[@]}"; do
    type_slug="$(type_slug_for "${type_name}")"
    audit_labels "${type_slug}"
  done
fi

echo "[DONE] Phase 5 relation-aware HNM data pipeline finished."
