# Type-Aware Post-Fusion Reweighting Plan

Date: 2026-05-10

## Motivation

Close/safe relation-aware HNM did not produce stable gains. The strongest current evidence is from type-aware modality reweighting:

```text
offline combined oracle: 0.7097 -> 0.7260
delta: +1.63%
```

The goal is to internalize this effect into SGMEA training without LLM at inference and without test-time reranking.

## Step 1: Verify Type Availability

Use the existing fixed type file:

`/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl`

Required coarse types:

```text
Person
Place
Organization
Creative Work
Event
Entity / Unknown
```

Action:

- Verify all train/test entities have an `entity_type_id`.
- Missing or generic entities should map to `Entity`.
- The `Entity` type bias should initialize to zero and can optionally be frozen or strongly regularized.

## Step 2: Implement Post-Fusion Type Bias

Modify `MformerFusion` after Transformer attention and before final concat:

Current:

```python
weight_norm = softmax(attention_score)
joint_emb = concat_i(weight_norm_i * normalize(modal_i))
```

Target:

```python
base_logit = log(weight_norm + eps)
type_bias = type_bias_table[type_id]          # [modal_num]
weight_final = softmax(base_logit + scale * type_bias)
joint_emb = concat_i(weight_final_i * normalize(modal_i))
```

Important:

- This must happen after Transformer fusion so the effect is not washed out.
- Inference remains a single embedding nearest-neighbor search.
- No LLM, no rerank, no test-time sweep.

## Step 3: Freeze & Learn Extreme Test

This is the first actual training experiment.

Load:

```text
SGMEA_FBDB15K_0.5_baseline_warm_softw_grid_0510_121128_
```

Freeze:

```text
all SGMEA parameters
```

Train only:

```text
type_bias_table
```

Parameter count:

```text
type_count x modal_count = about 6 x 6 = 36 parameters
```

Use the original SGMEA loss:

```text
L = L_original_SGMEA + lambda_bias_l2 * ||type_bias||^2
```

Recommended settings:

```text
lr: 1e-2, 3e-3
epoch: 30 to 80
eval_epoch: 1 or 2
lambda_bias_l2: 1e-4 to 1e-3
type_bias_scale: 1.0
initialize type_bias = 0
freeze generic Entity bias or apply stronger regularization
```

Expected direction if it works:

```text
Person:
  attr bias positive
  rel bias negative

Place / Organization / Creative Work:
  graph bias positive
```

Success criterion:

```text
Avg Hits@1 improves by >= +1.0% over baseline_warm.
No major drop on Person.
```

If it only gives `+0.2%` or less, do not move to larger fine-tuning.

## Step 4: Optional Joint Fine-Tuning

Only if Step 3 is positive.

Unfreeze:

```text
type_bias_table
MformerFusion only
```

Keep frozen:

```text
GNN
attribute encoder
relation encoder
image encoder
entity embedding
```

Use a smaller learning rate:

```text
type_bias lr: 3e-3
fusion lr: 5e-5 to 1e-4
```

Add regularization:

```text
KL(weight_final || weight_base)
or
L2(type_bias)
```

Success criterion:

```text
approach offline oracle direction without sacrificing full-test Hits@1.
```

## Stop Conditions

Stop this route if:

- freeze-only type bias does not improve by at least `+1%`;
- learned type bias goes opposite the oracle direction;
- gains appear only under test-time/manual sweep but not training;
- generic Entity bias dominates, suggesting type leakage or unstable routing.

