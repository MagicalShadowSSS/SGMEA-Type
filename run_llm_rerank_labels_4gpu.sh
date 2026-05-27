#!/usr/bin/env bash
set -euo pipefail

# Generic 4-GPU local-Llama labeling runner for test-time rerank top-k pairs.
#
# Required / commonly overridden env vars:
#   INPUT_JSONL=/path/to/topk_pairs.jsonl
#   OUTPUT_JSONL=/path/to/merged_labels.jsonl
#   SUMMARY_JSON=/path/to/merged_summary.json
#   COARSE_TYPES="Place" or "Place,Creative Work"
#
# Optional env vars:
#   ROOT_DIR=/gly/tongqiang/dongyufeng/Mode-Test/SGMEA
#   OUT_DIR=/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/llm_hnm_cases
#   MODEL_PATH=/root/code/models/Meta-Llama-3-8B-Instruct
#   MODEL_NAME=llama3_8b_instruct
#   PYTHON_BIN=/gly/tongqiang/anaconda3/envs/sgmea/bin/python
#   GPU_IDS=0,1,2,3
#   MAX_NEW_TOKENS=1024
#   RESUME=1

ROOT_DIR="${ROOT_DIR:-/gly/tongqiang/dongyufeng/Mode-Test/SGMEA}"
OUT_DIR="${OUT_DIR:-/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/llm_hnm_cases}"
MODEL_PATH="${MODEL_PATH:-/root/code/models/Meta-Llama-3-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-llama3_8b_instruct}"
PYTHON_BIN="${PYTHON_BIN:-/gly/tongqiang/anaconda3/envs/sgmea/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi
COARSE_TYPES="${COARSE_TYPES:-Place}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
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

TYPE_SLUG="$(echo "${COARSE_TYPES}" | tr ', ' '__' | tr -cd '[:alnum:]_')"
RUN_SLUG="$(basename "${OUTPUT_JSONL}" .jsonl | tr -cd '[:alnum:]_')"
TMP_DIR="${TMP_DIR:-${OUT_DIR}/rerank_label_chunks_${RUN_SLUG}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
mkdir -p "${TMP_DIR}" "${LOG_DIR}"

CHUNK_PREFIX="${TMP_DIR}/chunk"
export INPUT_JSONL OUTPUT_JSONL SUMMARY_JSON CHUNK_PREFIX GPU_IDS MODEL_NAME COARSE_TYPES RESUME

echo "[INFO] Input: ${INPUT_JSONL}"
echo "[INFO] Output: ${OUTPUT_JSONL}"
echo "[INFO] Summary: ${SUMMARY_JSON}"
echo "[INFO] Coarse types: ${COARSE_TYPES}"
echo "[INFO] GPUs: ${GPU_IDS}"
echo "[INFO] Resume: ${RESUME}"
echo "[INFO] Splitting input into query-balanced chunks..."

"${PYTHON_BIN}" - <<'PY'
import json
import os
from collections import defaultdict

input_jsonl = os.environ["INPUT_JSONL"]
chunk_prefix = os.environ["CHUNK_PREFIX"]
gpu_ids = [x.strip() for x in os.environ["GPU_IDS"].split(",") if x.strip()]
num_chunks = len(gpu_ids)

by_query = defaultdict(list)
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        by_query[row["query_id"]].append(row)

query_ids = sorted(by_query.keys())
buckets = [[] for _ in range(num_chunks)]
for i, qid in enumerate(query_ids):
    buckets[i % num_chunks].extend(by_query[qid])

for i, rows in enumerate(buckets):
    path = f"{chunk_prefix}_{i}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[INFO] chunk {i}: queries~{len(set(r['query_id'] for r in rows))} pairs={len(rows)} -> {path}", flush=True)
PY

run_chunk() {
  local gpu_id="$1"
  local chunk_id="$2"
  local input_chunk="${CHUNK_PREFIX}_${chunk_id}.jsonl"
  local output_chunk="${CHUNK_PREFIX}_${chunk_id}_labels.jsonl"
  local summary_chunk="${CHUNK_PREFIX}_${chunk_id}_summary.json"
  local log_file="${LOG_DIR}/rerank_${RUN_SLUG}_chunk_${chunk_id}.log"
  local extra_args=()

  if [[ "${RESUME}" == "1" ]]; then
    extra_args+=(--resume)
  fi

  echo "[START] coarse_types=${COARSE_TYPES} chunk=${chunk_id} gpu=${gpu_id}" | tee "${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    "${PYTHON_BIN}" "${ROOT_DIR}/build_llm_rerank_labels.py" \
      --input_jsonl "${input_chunk}" \
      --output_jsonl "${output_chunk}" \
      --output_summary_json "${summary_chunk}" \
      --model_path "${MODEL_PATH}" \
      --model_name "${MODEL_NAME}" \
      --coarse_types "${COARSE_TYPES}" \
      --max_cases 10000000 \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --sleep_seconds 0.0 \
      "${extra_args[@]}"
  ) >> "${log_file}" 2>&1
  echo "[DONE] coarse_types=${COARSE_TYPES} chunk=${chunk_id} gpu=${gpu_id}" | tee -a "${log_file}"
}

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
PIDS=()
for chunk_id in "${!GPU_ARRAY[@]}"; do
  run_chunk "${GPU_ARRAY[$chunk_id]}" "${chunk_id}" &
  PIDS+=("$!")
done

wait "${PIDS[@]}"

echo "[INFO] Merging chunk outputs..."
"${PYTHON_BIN}" - <<'PY'
import json
import os
from collections import Counter, defaultdict

chunk_prefix = os.environ["CHUNK_PREFIX"]
gpu_ids = [x.strip() for x in os.environ["GPU_IDS"].split(",") if x.strip()]
output_jsonl = os.environ["OUTPUT_JSONL"]
summary_json = os.environ["SUMMARY_JSON"]

rows = []
for i in range(len(gpu_ids)):
    path = f"{chunk_prefix}_{i}_labels.jsonl"
    if not os.path.exists(path):
        print(f"[WARN] missing chunk output: {path}", flush=True)
        continue
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
with open(output_jsonl, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

status_counter = Counter(r.get("llm_relationship_status", "unknown") for r in rows)
kind_counter = Counter(r.get("llm_relationship_kind", "unknown") for r in rows)
query_type_counter = Counter(r.get("llm_query_fine_type", "unknown") for r in rows)
candidate_type_counter = Counter(r.get("llm_candidate_fine_type", "unknown") for r in rows)
coarse_counter = defaultdict(Counter)
for r in rows:
    coarse_counter[r.get("query_coarse_type", "unknown")][r.get("llm_relationship_status", "unknown")] += 1

summary = {
    "teacher_model": os.environ.get("MODEL_NAME", "unknown"),
    "case_count": len(rows),
    "coarse_types": os.environ.get("COARSE_TYPES", ""),
    "relationship_status_counts": dict(status_counter),
    "relationship_kind_counts": dict(kind_counter),
    "query_fine_type_counts": dict(query_type_counter),
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

echo "[INFO] 4GPU local Llama rerank labeling finished."
