#!/usr/bin/env bash
set -euo pipefail

# Single-GPU TIDEA type generation chain:
#   entity name + attributes + relation context -> coarse type anchors
#   optional aligned-pair type repair -> norm_anchor_type_fixed.jsonl

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${CODE_ROOT:-$SCRIPT_DIR}
DATA_ROOT=${DATA_ROOT:-$CODE_ROOT/../../data}
DATA_PATH=${DATA_PATH:-mmkg}
DATA_CHOICE=${DATA_CHOICE:-FBDB15K}
DATA_SPLIT=${DATA_SPLIT:-norm}
PYTHON=${PYTHON:-python}
GPU_ID=${GPU_ID:-0}
LLM_PATH=${LLM_PATH:-}
OUTPUT_JSONL=${OUTPUT_JSONL:-}
RAW_OUTPUT_JSONL=${RAW_OUTPUT_JSONL:-}
TRACE_PATH=${TRACE_PATH:-}
SUMMARY_JSON=${SUMMARY_JSON:-}
CONSISTENCY_JSON=${CONSISTENCY_JSON:-}
MAX_ENTITIES=${MAX_ENTITIES:--1}
RESUME=${RESUME:-0}
RULE_BASED=${RULE_BASED:-0}
SKIP_TYPE_FIX=${SKIP_TYPE_FIX:-0}

cd "$CODE_ROOT"

args=(
  tools/generate_anchor_types.py
  --data_root "$DATA_ROOT"
  --data_path "$DATA_PATH"
  --data_choice "$DATA_CHOICE"
  --data_split "$DATA_SPLIT"
  --max_entities "$MAX_ENTITIES"
)

if [[ -n "$LLM_PATH" ]]; then
  args+=(--llm_path "$LLM_PATH")
fi
if [[ -n "$OUTPUT_JSONL" ]]; then
  args+=(--output_jsonl "$OUTPUT_JSONL")
fi
if [[ -n "$RAW_OUTPUT_JSONL" ]]; then
  args+=(--raw_output_jsonl "$RAW_OUTPUT_JSONL")
fi
if [[ -n "$TRACE_PATH" ]]; then
  args+=(--trace_path "$TRACE_PATH")
fi
if [[ -n "$SUMMARY_JSON" ]]; then
  args+=(--summary_json "$SUMMARY_JSON")
fi
if [[ -n "$CONSISTENCY_JSON" ]]; then
  args+=(--consistency_json "$CONSISTENCY_JSON")
fi
if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
fi
if [[ "$RULE_BASED" == "1" ]]; then
  args+=(--rule_based)
fi
if [[ "$SKIP_TYPE_FIX" == "1" ]]; then
  args+=(--skip_type_fix)
fi

echo "[info] code_root=$CODE_ROOT"
echo "[info] data_root=$DATA_ROOT"
echo "[info] data_path=$DATA_PATH"
echo "[info] data_choice=$DATA_CHOICE"
echo "[info] data_split=$DATA_SPLIT"
echo "[info] gpu_id=$GPU_ID"
echo "[info] rule_based=$RULE_BASED"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u "${args[@]}"
