# Phase 6: Training Internalization After Relation-Aware HNM

Date: 2026-05-10  
Code root: `/gly/tongqiang/dongyufeng/Mode-Test/SGMEA`  
Data root: `/gly/tongqiang/dongyufeng/data/mmkg`

## Goal

Decide whether the LLM semantic signal can be internalized into SGMEA training, rather than used as test-time reranking.

## Key Results So Far

### 1. Test Top10 Oracle

File:

`/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/llm_hnm_cases/FBDB15K_test_top10_pco_relation_oracle_upper_bound.json`

PCO top10 baseline:

- Avg Hits@1: `0.7606`
- Top10 GT oracle: `0.8459`
- Potential PCO gain: `+8.54%`

Important decomposition:

- `gt_in_top10 = 2943`
- GT labeled as `same_entity = 2936`
- `same_entity_gt_fraction = 99.76%`
- `same_entity_only_oracle` almost matches full top10 oracle.

Conclusion:

The large rerank gain mainly comes from identifying the exact same entity inside topK, not from close/safe negative protection.

### 2. Soft-Weighted InfoNCE

Implemented internal denominator weighting, not old auxiliary loss.

Observed results:

- baseline warm: about `0.7051`
- mask only: about `0.7045`
- best soft-weight variant observed: about `0.7054`

Conclusion:

Soft-weighted InfoNCE is too weak and does not reliably improve over baseline.

### 3. Virtual Edge Offline Audit

Virtual edges can pull close pairs closer, but do not improve final ranking.

Filtered strong edge audit:

- edges: about `1052`
- final Hits@1 delta: `0.0`

Conclusion:

Virtual edges are not a high-upside training route for the current goal.

### 4. Stage A Projection Probe

Script:

`stage_a_projection_probe.py`

Design:

- Freeze SGMEA.
- Train a residual projection head on LLM-labeled micro-clusters.
- Evaluate by original NN/CSLS, not rerank.

Observed pattern:

- Epoch 1 may show a transient small positive bump.
- Validation micro-cluster loss continues improving.
- Test NN Hits@1 degrades as training continues.

Completed active-only grid:

- `c005_s025`: selected delta `-0.81%`
- `c005_s040`: selected delta `-2.67%`
- `c010_s025`: selected delta `-0.65%`

Conclusion:

Local micro-cluster margin optimization conflicts with global nearest-neighbor retrieval.

### 5. Stage A2 Decoupled Identity Probe

Script:

`stage_a2_decoupled_identity_probe.py`

Design:

- Freeze topology embedding.
- Add identity branch.
- Final embedding: `[E_topology || gamma * E_identity]`.
- `gamma` is bounded.

Smoke result:

- baseline avg Hits@1: `0.7097`
- A2 best avg Hits@1: `0.7096`
- delta: `-0.00008`

Conclusion:

Decoupling makes the method safer, but does not produce useful gain.

### 6. LLM-Risk Modality Routing Audit

Script:

`audit_llm_risk_modality_routing.py`

Actual final embedding layout:

```text
[img, attr, rel, graph, gat_img, gat_attr]
```

Each segment is 300 dimensions; final embedding is 1800 dimensions.

Risk score:

```text
risk = close_count / (close_count + safe_count)
```

Findings:

- High risk anchors do not have higher graph weight.
- `risk_vs_graph_weight = -0.103`
- `risk_vs_attr_weight = +0.009`
- `risk_vs_rel_weight = +0.073`
- Train-risk anchors do not overlap test anchors.
- Train alignment is almost saturated, so train-risk does not explain test errors.

Conclusion:

LLM close-risk routing is not supported by the current audit.

### 7. Type-Modality Oracle Sweep

Script:

`audit_type_modality_oracle_sweep.py`

Fast sweep result:

`/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/train_relation_hnm/type_modality_oracle_sweep_baseline_warm_fast.json`

Baseline avg Hits@1:

- overall: `0.7097`
- Person: `0.8778`
- Place: `0.6797`
- Organization: `0.5532`
- Creative Work: `0.7850`

Type-local oracle improvements:

- Person: `+0.79%`
- Place: `+2.11%` best type-only, `+1.84%` in best-global setting
- Organization: `+2.40%`
- Creative Work: `+1.73%`

Combined type-specific reweighting:

- overall: `0.7097 -> 0.7260`
- delta: `+1.63%`

Combined by type:

- Person: `0.8778 -> 0.8873`
- Place: `0.6797 -> 0.7024`
- Organization: `0.5532 -> 0.5850`
- Creative Work: `0.7850 -> 0.8032`

Conclusion:

Type-aware post-fusion modality reweighting has the strongest evidence so far among training-internalization routes.

## Current Decision

Stop expanding close/safe HNM, virtual edges, projection-head fine-tuning, and LLM-risk routing.

Prioritize Type-Aware Post-Fusion Reweighting:

```text
w_base = original SGMEA fusion weight
w_final = softmax(log(w_base + eps) + type_bias[type])
E_final = concat_i w_final_i * modal_i
```

The next experiment should first train only `type_bias` while freezing all existing SGMEA parameters.


### 8. Type-Bias Protocol Fix

First freeze-only type-bias run used the intended command-line `--epoch 80`, but `--enable_sota` overwrote FBDB15K non-IL training to `epoch=500` in `config.py`. This also made cosine warmup too long (`warmup_steps=300`, `total_steps=2000`), so the first 75 epochs were still increasing LR. Hits@1 peaked early and then collapsed as the learned bias kept growing.

Observed first-run best:

```text
baseline avg Hits@1 = 0.7097
best avg Hits@1 = 0.7137
best gain = +0.40%
```

Failure mode:

- best point appeared around Ep17 for `lr=1e-2` and Ep35 for `lr=3e-3`.
- later epochs increased `type_modality_bias_abs_mean` and `type_modality_weight_delta_abs_mean` while Hits@1 dropped.
- the result should be treated as a protocol bug / over-training signal, not a clean negative result.

Code fixes:

- `config.py`: `type_modality_bias_only_train` now bypasses SOTA epoch override, so `--epoch` is respected.
- `main.py`: type-bias-only training now saves best checkpoint by avg Hits@1 instead of l2r MRR.
- `run_fbdb_type_modality_bias_4gpu_search.sh`: short-run grid defaults to `EPOCH=40`, `EVAL_EPOCH=1`, `SCHEDULER=fixed`, and runs configs in 4-GPU batches.

New short-grid protocol:

```text
lr: 1e-3, 3e-4, 1e-4
scale: 0.25, 0.5, 1.0
l2: 1e-2, 1e-1
epoch: 40
eval_epoch: 1
scheduler: fixed
best metric: avg Hits@1
```

### 9. Guidance-Branch Correction

A later review found an implementation mismatch: `--disable_sgmea_guidance` existed in config, but `MultiModalEncoder.forward` still computed and passed `gat_img` and `gat_attr` into fusion. Therefore the previous 6-segment audit/oracle used this unintended order:

```text
[img, attr, rel, graph, gat_img, gat_attr]
```

This is not the intended name-blind SGMEA mainline. The intended disabled-guidance embedding has 4 segments:

```text
[img, attr, rel, graph]
```

Fixes:

- `model/SGMEA_tools.py`: when `disable_sgmea_guidance=True`, `gat_img_emb` and `gat_att_emb` are set to `None` and are not counted in `fusion_modal_num`.
- `audit_llm_risk_modality_routing.py`: modal-name inference now respects `disable_sgmea_guidance`.

Verification with the baseline warm checkpoint:

```text
disable_sgmea_guidance=True
fusion.modal_num=4
modal_names=['img', 'attr', 'rel', 'graph']
joint_emb_shape=(27793, 1200)
weight_shape=(27793, 4)
checkpoint strict load: OK
```

Implication:

The previous 6-segment type-modality oracle result (`0.7097 -> 0.7260`) should be treated as a stale guidance-enabled diagnostic, not as the current mainline oracle. Before rerunning type-bias training, rerun the type-modality oracle sweep under the corrected 4-segment setting.
