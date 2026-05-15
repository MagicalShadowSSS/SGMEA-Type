#!/usr/bin/env bash
set -euo pipefail

# Relation-aware HNM grid launcher.
# Design:
# - one experiment uses one GPU
# - 4 GPUs are used for parallel single-GPU runs
# - the main grid is Soft-Weighted InfoNCE without close masking
# - mask-only and a few mask+soft runs are kept only as controls

CODE_ROOT="${CODE_ROOT:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
DATA_ROOT="${DATA_ROOT:-/gly/tongqiang/dongyufeng/data/mmkg}"
DATA_CHOICE="${DATA_CHOICE:-FBDB15K}"
DATA_SPLIT="${DATA_SPLIT:-norm}"
DATA_RATE="${DATA_RATE:-0.5}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
CKPT_NAME="${CKPT_NAME:-SGMEA_FBDB15K_0.5_rebuild_gpu0_}"
TYPE_JSONL_FIXED="${TYPE_JSONL_FIXED:-${DATA_ROOT}/anchors_nameless/${DATA_CHOICE}/${DATA_SPLIT}_anchor_type_fixed.jsonl}"
CACHE_JSONL="${CACHE_JSONL:-${DATA_ROOT}/SGMEA/train_relation_hnm/relation_aware_hnm_cache_top5.jsonl}"
LOG_DIR="${LOG_DIR:-${CODE_ROOT}/log/run_outputs/relation_aware_hnm_train_runs}"
RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/SGMEA/relation_aware_hnm_runs}"
DUMP_PATH="${DUMP_PATH:-${RUN_ROOT}/dump}"
RUN_TAG="${RUN_TAG:-softw_grid_$(date +%m%d_%H%M%S)}"
GRID_PRESET="${GRID_PRESET:-core}"

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

EXPERIMENTS=()

add_exp() {
  local exp_id="$1"
  local mode="$2"
  local safe_weight="${3:-1.0}"
  local close_weight="${4:-1.0}"
  EXPERIMENTS+=("${exp_id}|${mode}|${safe_weight}|${close_weight}")
}

build_grid() {
  add_exp "baseline_warm_${RUN_TAG}" "base"
  add_exp "mask_only_${RUN_TAG}" "mask"

  local safe_weights
  local close_weights
  if [[ "${GRID_PRESET}" == "wide" ]]; then
    safe_weights=(1.25 1.5 1.75 2.0 2.5 3.0)
    close_weights=(0.05 0.1 0.2 0.35 0.5)
  else
    safe_weights=(1.5 2.0 2.5 3.0)
    close_weights=(0.1 0.2 0.35)
  fi

  local s
  local c
  for s in "${safe_weights[@]}"; do
    for c in "${close_weights[@]}"; do
      add_exp "softw_s${s//./}_c${c//./}_${RUN_TAG}" "soft" "${s}" "${c}"
    done
  done

  # Controls: these combine close masking with denominator weighting. They are
  # not the main hypothesis because the close weight is mostly irrelevant once
  # close pairs are masked out of the denominator.
  add_exp "masksoft_s15_c02_${RUN_TAG}" "masksoft" 1.5 0.2
  add_exp "masksoft_s20_c02_${RUN_TAG}" "masksoft" 2.0 0.2
  add_exp "masksoft_s25_c01_${RUN_TAG}" "masksoft" 2.5 0.1
  add_exp "masksoft_s30_c02_${RUN_TAG}" "masksoft" 3.0 0.2
}

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

run_exp() {
  local gpu_id="$1"
  local exp_id="$2"
  local mode="$3"
  local safe_weight="$4"
  local close_weight="$5"
  local log_file="${RUN_LOG_DIR}/${exp_id}.log"
  local extra_args=()

  case "${mode}" in
    base)
      extra_args=()
      ;;
    mask)
      extra_args=("${REL_COMMON_ARGS[@]}" --relation_hnm_mask_close)
      ;;
    soft)
      extra_args=(
        "${REL_COMMON_ARGS[@]}"
        --relation_hnm_use_soft_weight
        --relation_hnm_safe_weight "${safe_weight}"
        --relation_hnm_close_weight "${close_weight}"
      )
      ;;
    masksoft)
      extra_args=(
        "${REL_COMMON_ARGS[@]}"
        --relation_hnm_mask_close
        --relation_hnm_use_soft_weight
        --relation_hnm_safe_weight "${safe_weight}"
        --relation_hnm_close_weight "${close_weight}"
      )
      ;;
    *)
      echo "[ERROR] unknown mode: ${mode}" >&2
      return 1
      ;;
  esac

  echo "[START] exp_id=${exp_id} mode=${mode} gpu=${gpu_id} safe=${safe_weight} close=${close_weight}" | tee "${log_file}"
  record_manifest "${gpu_id}" "${exp_id}" "mode=${mode}" "safe=${safe_weight}" "close=${close_weight}" "${extra_args[*]}"
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
    echo "[FAIL] exp_id=${exp_id} mode=${mode} gpu=${gpu_id}" | tee -a "${log_file}"
    printf 'FAILED %s %s\n' "${exp_id}" "${log_file}" >> "${STATUS_FILE}"
    return 1
  fi

  echo "[DONE] exp_id=${exp_id} mode=${mode} gpu=${gpu_id}" | tee -a "${log_file}"
  printf 'SUCCESS %s %s\n' "${exp_id}" "${log_file}" >> "${STATUS_FILE}"
}

run_queue() {
  local queue_idx="$1"
  local gpu_id="$2"
  local num_gpus="$3"
  local failed=0
  local idx
  for idx in "${!EXPERIMENTS[@]}"; do
    if (( idx % num_gpus != queue_idx )); then
      continue
    fi
    IFS='|' read -r exp_id mode safe_weight close_weight <<< "${EXPERIMENTS[$idx]}"
    if ! run_exp "${gpu_id}" "${exp_id}" "${mode}" "${safe_weight}" "${close_weight}"; then
      failed=1
      if [[ "${STOP_ON_FIRST_FAIL}" == "1" ]]; then
        return 1
      fi
    fi
  done
  return "${failed}"
}

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -lt 4 ]]; then
  echo "[ERROR] GPU_IDS must provide at least 4 GPUs for this launcher." >&2
  exit 1
fi

build_grid

{
  echo "[INFO] RUN_TAG=${RUN_TAG}"
  echo "[INFO] GRID_PRESET=${GRID_PRESET}"
  echo "[INFO] Experiment count=${#EXPERIMENTS[@]}"
  echo "[INFO] Effective LR=${EFFECTIVE_LR}"
  echo "[INFO] Cache=${CACHE_JSONL}"
  echo "[INFO] Logs=${RUN_LOG_DIR}"
  printf '[INFO] EXPERIMENTS:\n'
  printf '  %s\n' "${EXPERIMENTS[@]}"
} | tee -a "${STATUS_FILE}"

run_queue 0 "${GPU_ARRAY[0]}" 4 &
PID_0=$!
run_queue 1 "${GPU_ARRAY[1]}" 4 &
PID_1=$!
run_queue 2 "${GPU_ARRAY[2]}" 4 &
PID_2=$!
run_queue 3 "${GPU_ARRAY[3]}" 4 &
PID_3=$!

FAILED_COUNT=0
for pid in "${PID_0}" "${PID_1}" "${PID_2}" "${PID_3}"; do
  if ! wait "${pid}"; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
  fi
done

echo "[INFO] Status summary: ${STATUS_FILE}" | tee -a "${STATUS_FILE}"
echo "[INFO] Manifest: ${MANIFEST_FILE}" | tee -a "${STATUS_FILE}"
echo "[INFO] Checkpoints: ${CKPT_FILE}" | tee -a "${STATUS_FILE}"
echo "[INFO] Run logs: ${RUN_LOG_DIR}" | tee -a "${STATUS_FILE}"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
  echo "[ERROR] Soft-weight grid finished with ${FAILED_COUNT} failed queue(s)." >&2
  exit 1
fi

echo "[DONE] Soft-weight relation-aware HNM grid finished." | tee -a "${STATUS_FILE}"
