# Relation-Aware HNM Single-GPU Training Plan

## Goal

Turn the audited `relation_aware_hnm_cache_top5.jsonl` asset into a **single-GPU trainable** SGMEA upgrade that produces a **clear Hits@1 gain**, without relying on multi-GPU memory scaling during training.

This document replaces the earlier "minimal only" mindset with a stronger but still controlled implementation target:

- single GPU for final training
- 4 GPUs only for parallel experiment search
- no online LLM
- no pseudo-positive stage
- no `same_entity` in the first training line

## Current Data Status

The current top5 cache is now considered usable for training integration.

### Cache Summary

- `input_row_count = 5080`
- `kept_row_count = 5069`
- dropped only:
  - `same_entity = 11`
- kept labels:
  - `closely_related = 1850`
  - `safe_negative = 3219`
- `unique_anchor_count = 1016`
- `covered_anchor_rate = 0.079103`
- `per_anchor_relation_row_stats.mean = 4.989173`
- `high_quality = 2858`
- confidence:
  - `mean = 0.949693`
  - `median = 0.95`

### Audit Interpretation

- Coverage is no longer collapsed as in the failed "top5 with row-budget" run.
- Each covered anchor now has an almost complete top5 relation-aware local hard pool.
- `Place` is the strongest signal source.
- `Creative Work` is usable.
- `Organization` is usable, but with fewer `closely_related` cases.
- `same_entity` remains excluded from the first training line.

## Why Top5 Is Needed

The earlier top1 cache was too thin:

- one relation-aware row per covered anchor
- weak local hard-negative supervision
- too sensitive to single-candidate noise

The top5 cache solves this by giving each covered anchor a small local confusion neighborhood. Training can now distinguish:

- `safe_negative`: strong push-away
- `closely_related`: weak push-away

This is the main reason the top5 asset is worth integrating.

## Core Training Principle

Do **not** inject the relation-aware negatives into SGMEA's existing `neg_l / neg_r` path.

Reason:

- current `neg_l / neg_r` behavior is a batch-shared broadcast negative pool
- relation-aware cache entries are anchor-specific
- broadcasting a local `closely_related` negative to unrelated anchors would corrupt geometry

Therefore, the relation-aware signal must be implemented as a **separate matched loss**, computed only on the anchor it belongs to.

## Final Planned Design

The final single-GPU training line has 4 components.

### 1. In-Batch Shield Mask

Apply a zero-memory protection mask inside the main contrastive logits:

- detect when a batch accidentally contains a `closely_related` entity for a covered anchor
- mask that pair in the relevant contrastive matrix
- use non-inplace masking only

Required implementation rule:

- use `masked_fill`
- never use inplace edits like `logits[mask] = ...`

Masking must cover the relevant cross-view matrices, not only self-view matrices.

### 2. Step-Level Top5 Sampling

Do not fix the anchor to rank-1 only.

For each cache-hit anchor in a batch:

- sample at most one `safe_negative`
- sample at most one `closely_related`
- sample at **step-level**, not epoch-level

Recommended policy:

- stratified random sampling from the anchor's top5 pool
- optional preference for lower candidate rank
- optional confidence-aware weighting later

This converts top5 from a static table into local dynamic augmentation.

### 3. Matched Relation-Aware Margin Loss

Use an anchor-matched auxiliary loss on the same `joint_emb` graph as the original SGMEA loss.

For each sampled matched triplet:

- anchor: `A`
- positive: `P`
- negative: `N`

define:

\[
\Delta = \gamma_r + d(A, P) - d(A, N)
\]

and:

\[
L_r = \max(0, \Delta)
\]

where:

- `r = safe_negative` or `closely_related`
- `gamma_r` depends on relation type

Recommended first strong settings:

- `gamma_safe = 1.0`
- `gamma_close = 0.2`

Strong ablation:

- `gamma_close = 0.0`

### 4. Nonlinear Focal Violation Weighting

The matched loss should not weight all violations equally.

If a sample already satisfies the boundary, its effect should be weak.
If a sample violates badly, its penalty should rise sharply.

Recommended form:

\[
w(\Delta) = (1 + \max(0, \Delta))^\beta
\]

with:

- `beta = 2` as first strong trial

Alternative:

\[
w(\Delta) = \exp(\max(0, \Delta))
\]

Then:

\[
L^{focal}_r = w(\Delta) \cdot \max(0, \Delta)
\]

This keeps compute cheap while forcing the optimizer to prioritize truly pathological cases.

## How It Differs From Original SGMEA

Original SGMEA:

- in-batch negatives are treated uniformly
- replay negatives are also treated uniformly
- no notion of semantic negative roles
- hardest same-type negatives are pushed away with the same force

Relation-aware SGMEA:

- preserves the original SGMEA objective
- adds semantic shielding for `closely_related`
- adds matched anchor-specific weak/strong negative control
- uses the audited offline top5 cache

In short:

- original SGMEA: all hard negatives are equally repulsive
- new SGMEA: hard negatives are heterogeneous, so repulsion must be role-aware

## Computation and Backward Strategy

Use:

- one forward pass
- one total loss
- one unified backward pass

Recommended form:

\[
L_{total} = L_{original} + \lambda_{rel}(L_{safe} + L_{close})
\]

Do **not** split original loss and relation-aware loss into two separate backward passes unless memory forces a redesign.

Reason:

- two-step backward usually requires `retain_graph=True`
- this often increases memory pressure instead of reducing it

If memory becomes tight, reduce batch and use gradient accumulation across micro-batches instead.

## Mandatory Engineering Defenses

### Empty-Tensor Defense

Some batches will have no cache hits.

Required rule:

- if no valid matched samples exist, set:
  - `loss_rel_hnm = joint_emb.new_tensor(0.0)`

Do not run reductions over empty tensors.

### Loss Scale Monitoring

The original SGMEA loss and relation-aware loss may live on very different scales.

Must log raw values separately:

- `loss_original_raw`
- `loss_rel_safe_raw`
- `loss_rel_close_raw`
- `loss_rel_scaled`

If relation-aware loss is too small, increase `lambda_rel`.

### Non-Inplace Masking

Never modify logits inplace.

Use:

- `masked_fill`

Never use:

- direct assignment into tensors on the active computation graph

### Positive Mask Exclusion

The shield mask must never suppress the true aligned positive target.

Required rule:

- build an explicit positive mask for the valid positive coordinates
- apply:
  - `shield_mask = shield_mask & (~positive_mask)`

This must hold for every masked contrastive matrix used in the main loss.

### ID Consistency Check

In the current SGMEA pipeline, the cache IDs and `joint_emb` row indices are already in the same global entity-id space.

The real risk is not remapping, but **dataset/version mismatch**.

Must assert:

- same `data_choice`
- same `data_split`
- same `data_rate`
- same checkpoint family
- cache ids are within `joint_emb.shape[0]`

### Focal Weight Safety Cap

Nonlinear focal weighting is useful, but must not be allowed to spike without bound in early warm-start epochs.

Required rule:

- clamp the focal weight
- optionally clip global gradients

Recommended first implementation:

\[
w(\Delta) = \min \left( (1 + \max(0, \Delta))^\beta,\ 5.0 \right)
\]

and:

- `clip_grad_norm_(model.parameters(), 1.0)`

The exact cap can be tuned, but the first production version should not leave focal weights unbounded.

### Warm-Start Optimizer Reset

Warm-start must load only model weights.

Do **not** restore the optimizer state from the base SGMEA run.

Reason:

- the old optimizer momentum encodes the geometry of the old "uniform negative" objective
- relation-aware weak/strong losses change the local descent direction
- stale Adam moments can directly fight the new training signal

Required rule:

- load checkpoint weights
- re-initialize a fresh optimizer
- use a lower warm-start learning rate than the original base run

Recommended starting point:

- new optimizer state only
- learning rate around one tenth of the original base LR for the first strong relation-aware run

### Relation-Loss Scale Window

The relation-aware auxiliary loss must be strong enough to matter, but not so large that it dominates global geometry.

Required monitoring:

- `loss_original_raw`
- `loss_rel_safe_raw`
- `loss_rel_close_raw`
- `loss_rel_scaled`
- number of matched `safe` and `close` samples per batch

Recommended first control window:

- keep `lambda_rel * loss_rel_hnm` roughly within `5% ~ 15%` of `loss_original` over typical batches

This is not a hard law, but it is a strong first operating region.

If the relation-aware term is much smaller, increase `lambda_rel`.
If it becomes comparable to or larger than the main loss too often, reduce it.

### Small-Sample Variance Awareness

The matched relation-aware loss is computed on a fluctuating subset of the batch.

Therefore:

- batch-to-batch variance will be higher than the original SGMEA loss
- raw matched-loss curves may look noisy even when the signal is valid

Required rule:

- log matched sample counts together with loss values
- interpret relation-aware loss only together with hit statistics

If needed later, use lightweight smoothing in logging or EMA-style diagnostics, but do not hide the raw values during the first debugging pass.

## Single-GPU Strong v1

This is the recommended first production-grade training variant.

Components:

1. original SGMEA loss
2. in-batch shield mask on `closely_related`
3. step-level top5 stratified sampling
4. matched weak/strong relation-aware margin loss
5. nonlinear focal weighting

This is the main candidate expected to produce a clear gain.

## 4-GPU Parallel Experiment Layout

Use 4 GPUs for parallel runs, not for scaling a single training job.

Recommended matrix:

### GPU 0

- mask enabled
- top5 step sampling enabled
- `safe_negative` loss only
- no `closely_related` auxiliary loss

Purpose:

- isolate pure strong-negative gain

### GPU 1

- mask enabled
- top5 step sampling enabled
- `safe_negative` + `closely_related`
- `gamma_close = 0.2`

Purpose:

- main full relation-aware baseline

### GPU 2

- same as GPU 1
- focal violation weighting enabled
- `beta = 2`

Purpose:

- main high-impact candidate

### GPU 3

- same as GPU 2
- search one key sensitivity:
  - `gamma_close`
  - or `lambda_rel`

Recommended first search:

- `gamma_close = 0.0`
  or
- `lambda_rel` stronger than baseline

## Immediate Implementation Order

1. Add cache loader and in-memory per-anchor top5 index
2. Add matched-sample builder for current training batch
3. Add non-inplace shield mask into main contrastive path
4. Add matched weak/strong relation-aware margin loss
5. Add nonlinear focal weighting
6. Add logging for:
   - cache-hit counts
   - safe/close matched counts
   - raw loss scales
7. Add shell launchers for 4 parallel single-GPU runs

## Success Criterion

This phase is successful if:

- the new single-GPU line is stable
- relation-aware losses are non-trivial in scale
- no NaN / inplace / empty-tensor failures occur
- the strong v1 variant beats base SGMEA clearly enough to justify further refinement

## Explicit Non-Goals For First Round

Do not add in first round:

- `same_entity` training signal
- pseudo-positive augmentation
- multi-round cache refresh
- online LLM interaction
- large architectural refactors unrelated to relation-aware control
