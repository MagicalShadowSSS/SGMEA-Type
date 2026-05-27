# PHASE11 Direct LLM Rerank Bridge 0517

## Scope

This phase ports only the early direct LLM semantic rerank branch from
`/gly/tongqiang/dongyufeng/Mode-Test/Test-1/SGMEA` into the current V2-P
router workspace.

Included:

- top-k pair extraction from a current SGMEA / V2-P checkpoint configuration
- local LLM semantic labeling for query-candidate pairs
- conservative test-time reranking from LLM labels
- query-balanced sampling and coverage counting helpers

Excluded intentionally:

- teacher-student reranker training
- Relation-Aware HNM training
- old LLM-HNM diagnosis scripts
- any change to the V2-P router itself

## Ported Files

- `extract_test_topk_pairs.py`
- `build_llm_rerank_labels.py`
- `rerank_fbdb_topk.py`
- `sample_rerank_queries.py`
- `count_rerank_queries.py`
- `estimate_overall_rerank_gain.py`
- `run_llm_rerank_labels_4gpu.sh`
- `run_fbdb_v2p_llm_rerank.sh`

## Current-Workspace Adaptations

1. `run_llm_rerank_labels_4gpu.sh` now defaults to:
   - `ROOT_DIR=/gly/tongqiang/dongyufeng/Mode-Test/SGMEA`
   - `OUT_DIR=/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/llm_hnm_cases`
   - `PYTHON_BIN=/gly/tongqiang/anaconda3/envs/sgmea/bin/python`

2. `extract_test_topk_pairs.py` now supports current router-only V2-P extraction:
   - router warm-start uses `strict=False` when DEHR/TMHG/type-bias modules are enabled
   - new flag `--run_router_calibration 1` reruns DEHR/TMHG calibration before exporting top-k pairs
   - this is needed because V2-P results are produced by loading a baseline SGMEA checkpoint and calibrating the DEHR router at evaluation time

3. `run_fbdb_v2p_llm_rerank.sh` is the current one-command bridge:
   - extracts V2-P top-k pairs
   - optionally runs 4-GPU local LLM labeling
   - optionally runs conservative rerank sweep

## Direct Rerank Label Taxonomy

The LLM assigns one relation label per query-candidate pair:

- `same_entity`: promote
- `closely_related`: protect from harsh demotion
- `safe_negative`: demote
- `unknown`: leave neutral

The conservative rerank score changes candidate distance as:

```text
final_distance = base_distance - lambda_weight * confidence * delta(label)
```

## Smoke Test

Command:

```bash
RUN_LABELS=0 RUN_RERANK=0 COARSE_TYPES=Place TOPK=3 \
OUT_DIR=/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/llm_rerank_v2p_smoke \
bash run_fbdb_v2p_llm_rerank.sh
```

Result:

- V2-P DEHR calibration completed with the accepted residual-mix setting
- extracted `3432` top-3 candidate pairs
- covered `1270` unique `Place` queries
- output:
  - `/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/llm_rerank_v2p_smoke/FBDB15K_r0.5_v2p_top3_Place_pairs.jsonl`
  - `/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/llm_rerank_v2p_smoke/FBDB15K_r0.5_v2p_top3_Place_pairs_summary.json`

## Full Current V2-P Rerank Command

To extract top-10 pairs for the three historically validated FBDB types without LLM labeling:

```bash
RUN_LABELS=0 RUN_RERANK=0 \
COARSE_TYPES="Place,Creative Work,Organization" TOPK=10 \
bash run_fbdb_v2p_llm_rerank.sh
```

To run the full direct LLM rerank branch:

```bash
RUN_LABELS=1 RUN_RERANK=1 \
COARSE_TYPES="Place,Creative Work,Organization" TOPK=10 \
MODEL_PATH=/path/to/local/instruct-llm \
GPU_IDS=0,1,2,3 \
bash run_fbdb_v2p_llm_rerank.sh
```

Default rerank denominator setting inherited from the early FBDB line:

```text
Place:1343, Creative Work:1131, Organization:1005
```

For paper-grade reporting, refresh these denominators if the active split/profile changes.

## Interpretation Boundary

This branch should be described as a test-time semantic verification / rerank add-on over the current V2-P retrieval output. It is not part of the lightweight V2-P router core and should not be mixed into the router ablation unless explicitly studied as a separate inference-time enhancement.
