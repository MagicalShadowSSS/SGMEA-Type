#!/usr/bin/env bash
set -euo pipefail

# One-click 4-GPU search launcher for single-GPU relation-aware HNM SGMEA training.
# Policy:
# - each experiment uses exactly one GPU
# - 4 GPUs are used only for parallel experiment search
# - final training graph remains single-GPU

CODE_ROOT="${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
DATA_ROOT="${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}"
DATA_CHOICE="${DATA_CHOICE:-FBDB15K}"
DATA_SPLIT="${DATA_SPLIT:-norm}"
DATA_RATE="${DATA_RATE:-0.5}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CKPT_NAME="${CKPT_NAME:-SGMEA_FBDB15K_0.5_rebuild_gpu0_}"
TYPE_JSONL_FIXED="${TYPE_JSONL_FIXED:-${DATA_ROOT}/anchors_nameless/${DATA_CHOICE}/${DATA_SPLIT}_anchor_type_fixed.jsonl}"
CACHE_JSONL="${CACHE_JSONL:-${DATA_ROOT}/SGMEA/train_relation_hnm/relation_aware_hnm_cache_top5.jsonl}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/SGMEA/relation_aware_hnm_runs/logs}"
RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/SGMEA/relation_aware_hnm_runs}"
DUMP_PATH="${DUMP_PATH:-${RUN_ROOT}/dump}"
RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
BASE_LR="${BASE_LR:-5e-4}"
LR_SCALE="${LR_SCALE:-0.1}"
SAVE_MODEL="${SAVE_MODEL:-1}"
SEED="${SEED:-42}"
STOP_ON_FIRST_FAIL="${STOP_ON_FIRST_FAIL:-0}"

EFFECTIVE_LR="$(python - "${BASE_LR}" "${LR_SCALE}" <<'PY'
import sys
base_lr = float(sys.argv[1])
scale = float(sys.argv[2])
print(f"{base_lr * scale:.12g}")
PY
)"

TRAIN_COMMON_ARGS=(
  --only_test 0
  --model_name SGMEA
  --model_name_save "${CKPT_NAME}"
  --data_choice "${DATA_CHOICE}"
  --data_split "${DATA_SPLIT}"
  --data_rate "${DATA_RATE}"
  --epoch "${EPOCHS}"
  --lr "${EFFECTIVE_LR}"
  --hidden_units "300,300,300"
  --save_model "${SAVE_MODEL}"
  --batch_size "${BATCH_SIZE}"
  --csls
  --csls_k 3
  --random_seed "${SEED}"
  --workers 12
  --dist 0
  --accumulation_steps 1
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
  --clip 1.0
  --eval_epoch 1
  --dump_path "${DUMP_PATH}"
)

REL_COMMON_ARGS=(
  --use_relation_aware_hnm
  --relation_hnm_cache_jsonl "${CACHE_JSONL}"
  --relation_hnm_grad_clip 1.0
)

RUN_LOG_DIR="${LOG_DIR}/${RUN_TAG}"
mkdir -p "${RUN_LOG_DIR}"
STATUS_FILE="${RUN_LOG_DIR}/run_status_${RUN_TAG}.txt"
MANIFEST_FILE="${RUN_LOG_DIR}/run_manifest_${RUN_TAG}.txt"
CKPT_FILE="${RUN_LOG_DIR}/checkpoint_paths_${RUN_TAG}.txt"
: > "${STATUS_FILE}"
: > "${MANIFEST_FILE}"
: > "${CKPT_FILE}"

record_manifest() {
  local gpu_id="$1"
  local exp_id="$2"
  shift 2
  printf 'GPU=%s EXP=%s ARGS=%s\n' "${gpu_id}" "${exp_id}" "$*" >> "${MANIFEST_FILE}"
}

record_checkpoint_path() {
  local exp_id="$1"
  local resolved_exp_id="SGMEA_${DATA_CHOICE}_${DATA_RATE}_${exp_id}"
  local ckpt_path="${DATA_ROOT}/SGMEA/save/${resolved_exp_id}.pkl"
  printf '%s %s\n' "${resolved_exp_id}" "${ckpt_path}" >> "${CKPT_FILE}"
}

run_base_exp() {
  local gpu_id="$1"
  local exp_id="$2"
  local log_file="$3"
  shift 3
  local extra_args=("$@")

  echo "[START] exp_id=${exp_id} gpu=${gpu_id}" | tee "${log_file}"
  record_manifest "${gpu_id}" "${exp_id}" "${extra_args[*]}"
  record_checkpoint_path "${exp_id}"
  if ! (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    python "${CODE_ROOT}/main.py" \
      --gpu 0 \
      --exp_name "RelationAwareHNM_${DATA_CHOICE}_${DATA_RATE}" \
      --exp_id "${exp_id}" \
      "${TRAIN_COMMON_ARGS[@]}" \
      "${extra_args[@]}"
  ) >> "${log_file}" 2>&1; then
    echo "[FAIL] exp_id=${exp_id} gpu=${gpu_id}" | tee -a "${log_file}"
    printf 'FAILED %s %s\n' "${exp_id}" "${log_file}" >> "${STATUS_FILE}"
    return 1
  fi
  echo "[DONE] exp_id=${exp_id} gpu=${gpu_id}" | tee -a "${log_file}"
  printf 'SUCCESS %s %s\n' "${exp_id}" "${log_file}" >> "${STATUS_FILE}"
}

run_ra_exp() {
  local gpu_id="$1"
  local exp_id="$2"
  local log_file="$3"
  shift 3
  local extra_args=("$@")

  echo "[START] exp_id=${exp_id} gpu=${gpu_id}" | tee "${log_file}"
  record_manifest "${gpu_id}" "${exp_id}" "${extra_args[*]}"
  record_checkpoint_path "${exp_id}"
  if ! (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    python "${CODE_ROOT}/main.py" \
      --gpu 0 \
      --exp_name "RelationAwareHNM_${DATA_CHOICE}_${DATA_RATE}" \
      --exp_id "${exp_id}" \
      "${TRAIN_COMMON_ARGS[@]}" \
      "${REL_COMMON_ARGS[@]}" \
      "${extra_args[@]}"
  ) >> "${log_file}" 2>&1; then
    echo "[FAIL] exp_id=${exp_id} gpu=${gpu_id}" | tee -a "${log_file}"
    printf 'FAILED %s %s\n' "${exp_id}" "${log_file}" >> "${STATUS_FILE}"
    return 1
  fi
  echo "[DONE] exp_id=${exp_id} gpu=${gpu_id}" | tee -a "${log_file}"
  printf 'SUCCESS %s %s\n' "${exp_id}" "${log_file}" >> "${STATUS_FILE}"
}

maybe_stop_queue() {
  local failed="$1"
  if [[ "${failed}" == "1" && "${STOP_ON_FIRST_FAIL}" == "1" ]]; then
    return 0
  fi
  return 1
}

run_queue_gpu0() {
  local gpu_id="$1"
  local failed=0
  run_base_exp "${gpu_id}" "baseline_warm_${RUN_TAG}" "${RUN_LOG_DIR}/baseline_warm_${RUN_TAG}.log" || failed=1
  if maybe_stop_queue "${failed}"; then return 1; fi
  run_ra_exp "${gpu_id}" "mask_only_${RUN_TAG}" "${RUN_LOG_DIR}/mask_only_${RUN_TAG}.log" \
    --relation_hnm_mask_close || failed=1
  return "${failed}"
}

run_queue_gpu1() {
  local gpu_id="$1"
  local failed=0
  run_ra_exp "${gpu_id}" "softw_s2_c02_${RUN_TAG}" "${RUN_LOG_DIR}/softw_s2_c02_${RUN_TAG}.log" \
    --relation_hnm_mask_close \
    --relation_hnm_use_soft_weight \
    --relation_hnm_safe_weight 2.0 \
    --relation_hnm_close_weight 0.2 || failed=1
  if maybe_stop_queue "${failed}"; then return 1; fi
  run_ra_exp "${gpu_id}" "softw_s3_c02_${RUN_TAG}" "${RUN_LOG_DIR}/softw_s3_c02_${RUN_TAG}.log" \
    --relation_hnm_mask_close \
    --relation_hnm_use_soft_weight \
    --relation_hnm_safe_weight 3.0 \
    --relation_hnm_close_weight 0.2 || failed=1
  return "${failed}"
}

run_queue_gpu2() {
  local gpu_id="$1"
  local failed=0
  run_ra_exp "${gpu_id}" "softw_s2_c01_${RUN_TAG}" "${RUN_LOG_DIR}/softw_s2_c01_${RUN_TAG}.log" \
    --relation_hnm_mask_close \
    --relation_hnm_use_soft_weight \
    --relation_hnm_safe_weight 2.0 \
    --relation_hnm_close_weight 0.1 || failed=1
  if maybe_stop_queue "${failed}"; then return 1; fi
  run_ra_exp "${gpu_id}" "softw_s3_c01_${RUN_TAG}" "${RUN_LOG_DIR}/softw_s3_c01_${RUN_TAG}.log" \
    --relation_hnm_mask_close \
    --relation_hnm_use_soft_weight \
    --relation_hnm_safe_weight 3.0 \
    --relation_hnm_close_weight 0.1 || failed=1
  return "${failed}"
}

run_queue_gpu3() {
  local gpu_id="$1"
  local failed=0
  run_ra_exp "${gpu_id}" "softw_nomask_s2_c02_${RUN_TAG}" "${RUN_LOG_DIR}/softw_nomask_s2_c02_${RUN_TAG}.log" \
    --relation_hnm_use_soft_weight \
    --relation_hnm_safe_weight 2.0 \
    --relation_hnm_close_weight 0.2 || failed=1
  if maybe_stop_queue "${failed}"; then return 1; fi
  run_ra_exp "${gpu_id}" "softw_s2_c05_${RUN_TAG}" "${RUN_LOG_DIR}/softw_s2_c05_${RUN_TAG}.log" \
    --relation_hnm_mask_close \
    --relation_hnm_use_soft_weight \
    --relation_hnm_safe_weight 2.0 \
    --relation_hnm_close_weight 0.5 || failed=1
  return "${failed}"
}

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -lt 4 ]]; then
  echo "[ERROR] GPU_IDS must provide 4 GPUs for this launcher." >&2
  exit 1
fi

run_queue_gpu0 "${GPU_ARRAY[0]}" &
PID_0=$!

run_queue_gpu1 "${GPU_ARRAY[1]}" &
PID_1=$!

run_queue_gpu2 "${GPU_ARRAY[2]}" &
PID_2=$!

run_queue_gpu3 "${GPU_ARRAY[3]}" &
PID_3=$!

PIDS=("${PID_0}" "${PID_1}" "${PID_2}" "${PID_3}")
EXP_NAMES=(
  "gpu0_queue_${RUN_TAG}"
  "gpu1_queue_${RUN_TAG}"
  "gpu2_queue_${RUN_TAG}"
  "gpu3_queue_${RUN_TAG}"
)
EXP_LOGS=(
  "${RUN_LOG_DIR}"
  "${RUN_LOG_DIR}"
  "${RUN_LOG_DIR}"
  "${RUN_LOG_DIR}"
)

FAILED_COUNT=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    echo "QUEUE_SUCCESS ${EXP_NAMES[$idx]} ${EXP_LOGS[$idx]}" | tee -a "${STATUS_FILE}"
  else
    echo "QUEUE_FAILED ${EXP_NAMES[$idx]} ${EXP_LOGS[$idx]}" | tee -a "${STATUS_FILE}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
  fi
done

echo "[INFO] RUN_TAG=${RUN_TAG}" | tee -a "${STATUS_FILE}"
echo "[INFO] Status summary: ${STATUS_FILE}" | tee -a "${STATUS_FILE}"
echo "[INFO] Manifest: ${MANIFEST_FILE}" | tee -a "${STATUS_FILE}"
echo "[INFO] Checkpoints: ${CKPT_FILE}" | tee -a "${STATUS_FILE}"
echo "[INFO] Run logs: ${RUN_LOG_DIR}" | tee -a "${STATUS_FILE}"
echo "[INFO] Effective LR shared by baseline and relation-aware runs: ${EFFECTIVE_LR}" | tee -a "${STATUS_FILE}"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
  echo "[ERROR] Relation-aware HNM 4-GPU search finished with ${FAILED_COUNT} failed experiment(s)." >&2
  exit 1
fi

echo "[DONE] Relation-aware HNM 4-GPU single-run search finished."

