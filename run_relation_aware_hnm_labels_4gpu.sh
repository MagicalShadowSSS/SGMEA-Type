#!/usr/bin/env bash
set -euo pipefail

# Generic multi-GPU local-Llama labeling runner for training-side relation-aware HNM cases.
#
# Required env vars:
#   INPUT_JSONL=/path/to/cases.jsonl
#   OUTPUT_JSONL=/path/to/merged_labels.jsonl
#   SUMMARY_JSON=/path/to/merged_summary.json
#
# Optional env vars:
#   ROOT_DIR=/gly/tongqiang/dongyufeng/Mode-Test/SGMEA
#   OUT_DIR=/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/train_relation_hnm
#   MODEL_PATH=/gly/tongqiang/lxwlxwlxw/HomeBenchReproduction/models/llama3-8b-Instruct
#   MODEL_NAME=llama3_8b_instruct
#   GPU_IDS=0,1,2,3,4,5,6,7
#   QUALITY_FILTER=all
#   MAX_CASES=10000000
#   MAX_NEW_TOKENS=1024
#   RESUME=1

ROOT_DIR="${ROOT_DIR:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
OUT_DIR="${OUT_DIR:-/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/train_relation_hnm}"
MODEL_PATH="${MODEL_PATH:-/gly/tongqiang/lxwlxwlxw/HomeBenchReproduction/models/llama3-8b-Instruct}"
MODEL_NAME="${MODEL_NAME:-llama3_8b_instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
QUALITY_FILTER="${QUALITY_FILTER:-all}"
MAX_CASES="${MAX_CASES:-10000000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RESUME="${RESUME:-0}"

if [[ -z "${INPUT_JSONL:-}" ]]; then
  echo "[ERROR] INPUT_JSONL is required." >&2
  exit 1
fi
if [[ -z "${OUTPUT_JSONL:-}" ]]; then
  echo "[ERROR] OUTPUT_JSONL is required." >&2
  exit 1
fi
if [[ -z "${SUMMARY_JSON:-}" ]]; then
  echo "[ERROR] SUMMARY_JSON is required." >&2
  exit 1
fi

RUN_SLUG="$(basename "${OUTPUT_JSONL}" .jsonl | tr -cd '[:alnum:]_')"
TMP_DIR="${TMP_DIR:-${OUT_DIR}/relation_hnm_label_chunks_${RUN_SLUG}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
mkdir -p "${TMP_DIR}" "${LOG_DIR}"

CHUNK_PREFIX="${TMP_DIR}/chunk"
export INPUT_JSONL OUTPUT_JSONL SUMMARY_JSON CHUNK_PREFIX GPU_IDS MODEL_NAME QUALITY_FILTER MAX_CASES

echo "[INFO] Input: ${INPUT_JSONL}"
echo "[INFO] Output: ${OUTPUT_JSONL}"
echo "[INFO] Summary: ${SUMMARY_JSON}"
echo "[INFO] GPUs: ${GPU_IDS}"
echo "[INFO] Quality filter: ${QUALITY_FILTER}"
echo "[INFO] Max cases: ${MAX_CASES}"
echo "[INFO] Resume: ${RESUME}"
echo "[INFO] Splitting input into anchor-group-balanced chunks..."

python - <<'PY'
import json
import os
from collections import defaultdict

input_jsonl = os.environ["INPUT_JSONL"]
chunk_prefix = os.environ["CHUNK_PREFIX"]
gpu_ids = [x.strip() for x in os.environ["GPU_IDS"].split(",") if x.strip()]
num_chunks = len(gpu_ids)
quality_filter = os.environ.get("QUALITY_FILTER", "all")
max_cases = int(os.environ.get("MAX_CASES", "10000000"))

grouped = defaultdict(list)
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if quality_filter != "all" and row.get("evidence_quality", "low") != quality_filter:
            continue
        key = (
            row.get("dataset"),
            row.get("cache_row"),
            row.get("direction"),
        )
        grouped[key].append(row)

groups = sorted(
    grouped.values(),
    key=lambda rows: (
        -(len(rows)),
        str(rows[0].get("cache_row", "")),
        str(rows[0].get("direction", "")),
    ),
)

selected_groups = []
selected_rows = 0
for rows in groups:
    if max_cases > 0 and selected_rows >= max_cases:
        break
    selected_groups.append(rows)
    selected_rows += len(rows)

buckets = [[] for _ in range(num_chunks)]
bucket_sizes = [0 for _ in range(num_chunks)]
for rows in selected_groups:
    idx = min(range(num_chunks), key=lambda i: bucket_sizes[i])
    buckets[idx].extend(rows)
    bucket_sizes[idx] += len(rows)

for i, rows in enumerate(buckets):
    path = f"{chunk_prefix}_{i}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    anchor_groups = {
        (r.get("dataset"), r.get("cache_row"), r.get("direction"))
        for r in rows
    }
    print(
        f"[INFO] chunk {i}: anchor_groups={len(anchor_groups)} cases={len(rows)} -> {path}",
        flush=True,
    )
PY

run_chunk() {
  local gpu_id="$1"
  local chunk_id="$2"
  local input_chunk="${CHUNK_PREFIX}_${chunk_id}.jsonl"
  local output_chunk="${CHUNK_PREFIX}_${chunk_id}_labels.jsonl"
  local summary_chunk="${CHUNK_PREFIX}_${chunk_id}_summary.json"
  local log_file="${LOG_DIR}/relation_hnm_${RUN_SLUG}_chunk_${chunk_id}.log"
  local extra_args=()

  if [[ "${RESUME}" == "1" ]]; then
    extra_args+=(--resume)
  fi

  echo "[START] chunk=${chunk_id} gpu=${gpu_id}" | tee "${log_file}"
  if ! (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    python "${ROOT_DIR}/build_llm_fine_types.py" \
      --input_jsonl "${input_chunk}" \
      --output_jsonl "${output_chunk}" \
      --output_summary_json "${summary_chunk}" \
      --model_path "${MODEL_PATH}" \
      --model_name "${MODEL_NAME}" \
      --quality_filter "${QUALITY_FILTER}" \
      --max_cases "${MAX_CASES}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --sleep_seconds 0.0 \
      "${extra_args[@]}"
  ) >> "${log_file}" 2>&1; then
    echo "[ERROR] chunk=${chunk_id} gpu=${gpu_id} failed. See ${log_file}" | tee -a "${log_file}"
    return 1
  fi
  if [[ -s "${input_chunk}" && ! -s "${output_chunk}" ]]; then
    echo "[ERROR] chunk=${chunk_id} gpu=${gpu_id} finished without a non-empty output file: ${output_chunk}" | tee -a "${log_file}"
    return 1
  fi
  echo "[DONE] chunk=${chunk_id} gpu=${gpu_id}" | tee -a "${log_file}"
}

progress_monitor() {
  local total_cases="$1"
  shift
  local done=0
  while true; do
    done=0
    local chunk_id
    for chunk_id in "${!GPU_ARRAY[@]}"; do
      local output_chunk="${CHUNK_PREFIX}_${chunk_id}_labels.jsonl"
      if [[ -f "${output_chunk}" ]]; then
        local count
        count="$(wc -l < "${output_chunk}" | tr -d '[:space:]')"
        done=$((done + count))
      fi
    done
    echo "[PROGRESS] labeled_rows=${done}/${total_cases} active_pids=$*" >&2
    local all_finished=1
    local pid
    for pid in "$@"; do
      if kill -0 "${pid}" 2>/dev/null; then
        all_finished=0
        break
      fi
    done
    if [[ "${all_finished}" == "1" ]]; then
      break
    fi
    sleep 60
  done
}

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
PIDS=()
for chunk_id in "${!GPU_ARRAY[@]}"; do
  run_chunk "${GPU_ARRAY[$chunk_id]}" "${chunk_id}" &
  PIDS+=("$!")
done

TOTAL_CASES=0
for chunk_id in "${!GPU_ARRAY[@]}"; do
  chunk_path="${CHUNK_PREFIX}_${chunk_id}.jsonl"
  if [[ -f "${chunk_path}" ]]; then
    count="$(wc -l < "${chunk_path}" | tr -d '[:space:]')"
    TOTAL_CASES=$((TOTAL_CASES + count))
  fi
done
progress_monitor "${TOTAL_CASES}" "${PIDS[@]}" &
MONITOR_PID="$!"
wait "${PIDS[@]}"
wait "${MONITOR_PID}" || true

echo "[INFO] Merging chunk outputs..."
python - <<'PY'
import json
import os
from collections import Counter, defaultdict

chunk_prefix = os.environ["CHUNK_PREFIX"]
gpu_ids = [x.strip() for x in os.environ["GPU_IDS"].split(",") if x.strip()]
output_jsonl = os.environ["OUTPUT_JSONL"]
summary_json = os.environ["SUMMARY_JSON"]

rows = []
missing_required_outputs = []
for i in range(len(gpu_ids)):
    input_path = f"{chunk_prefix}_{i}.jsonl"
    path = f"{chunk_prefix}_{i}_labels.jsonl"
    input_count = 0
    if os.path.exists(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            input_count = sum(1 for line in f if line.strip())
    if not os.path.exists(path):
        if input_count > 0:
            missing_required_outputs.append((i, path, input_count))
        else:
            print(f"[WARN] missing empty-chunk output: {path}", flush=True)
        continue
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

if missing_required_outputs:
    details = "; ".join(
        f"chunk={chunk_id} input_rows={input_count} missing={path}"
        for chunk_id, path, input_count in missing_required_outputs
    )
    raise RuntimeError(f"Missing required chunk label outputs: {details}")

os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
with open(output_jsonl, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

status_counter = Counter(r.get("llm_relationship_status", "unknown") for r in rows)
kind_counter = Counter(r.get("llm_relationship_kind", "unknown") for r in rows)
anchor_type_counter = Counter(r.get("llm_anchor_fine_type", "unknown") for r in rows)
candidate_type_counter = Counter(r.get("llm_candidate_fine_type", "unknown") for r in rows)
coarse_counter = defaultdict(Counter)
for r in rows:
    coarse_counter[r.get("coarse_type", "unknown")][r.get("llm_relationship_status", "unknown")] += 1

summary = {
    "teacher_model": os.environ.get("MODEL_NAME", "unknown"),
    "case_count": len(rows),
    "relationship_status_counts": dict(status_counter),
    "relationship_kind_counts": dict(kind_counter),
    "anchor_fine_type_counts": dict(anchor_type_counter),
    "candidate_fine_type_counts": dict(candidate_type_counter),
    "coarse_type_breakdown": {k: dict(v) for k, v in coarse_counter.items()},
}
os.makedirs(os.path.dirname(summary_json), exist_ok=True)
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"[INFO] merged rows: {len(rows)}", flush=True)
print(f"[INFO] merged output: {output_jsonl}", flush=True)
print(f"[INFO] merged summary: {summary_json}", flush=True)
PY

echo "[INFO] Multi-GPU relation-aware HNM labeling finished."

