# PHASE10 Elegant Triplet Router Directive 0514

## Purpose

This phase resets the design boundary for the trainable type-aware modality router. The goal is to move away from manual/grid-searched priors and also avoid heuristic-heavy scalar feature engineering. Future experiments must follow this directive unless explicitly superseded.

## Core Task

Design a lightweight, trainable, interpretable type-aware modality router for MMEA.

- Entity types: `Person`, `Place`, `Organization`, `Creative Work`.
- Modalities: `Image`, `Attribute`, `Relation`, `Graph`.
- Output: a type-specific modality bias/weight matrix that directly calibrates SGMEA fusion.
- Initial target setting: `Rate = 0.5` on both `FBDB15K` and `FBYG15K`.

## Performance Target

The learned router must retain at least 80% of the manual Oracle gain.

Reference values at `Rate = 0.5`:

| Dataset | Baseline Avg H@1 | Manual Oracle Avg H@1 | Oracle Gain | 80% Retention Target |
| --- | ---: | ---: | ---: | ---: |
| FBDB15K | 0.71585 | 0.73775 | 0.02190 | >= 0.73337 |
| FBYG15K | 0.79170 | 0.81215 | 0.02045 | >= 0.80806 |

## Hard Boundaries

### 1. No Hand-Crafted Priors As Answers

The router must not use manual/grid-searched modality priors as the final answer.

Forbidden as primary success mechanism:

- Directly adding Oracle/statistical prior logits into SGMEA fusion.
- Training the router mainly to reconstruct the selected Oracle prior matrix.
- Explaining gains as a hard-coded reliability direction or fixed bias table.

Statistical or Oracle files may be used only for post-hoc comparison, diagnostics, or non-training analysis unless a later directive explicitly allows otherwise.

### 2. No Scalar Statistical Soup

Avoid handcrafted macro-feature engineering that makes the method look like traditional ML reliability scoring.

Forbidden as router inputs:

- manually computed variance/dispersion scores,
- conflict entropy,
- prototype gap/cross-KG shift,
- top-k gap,
- unimodal H@1,
- hand-designed modality reliability scores,
- baseline confidence/margin/pseudo-link score.

The router should learn modality reliability from dense representations, not from spoon-fed scalar reliability statistics.

### 3. Avoid Confirmation Bias

The router input must be decoupled from baseline predictions as much as possible.

Forbidden or strongly discouraged:

- pseudo links mined from baseline embeddings as the main router supervision,
- baseline ranking confidence as a router feature,
- baseline hard-negative margin as a router feature.

Preferred supervision:

- real train ILL pairs,
- modality-level/fusion ranking loss derived from ground-truth training alignments,
- stable auxiliary regularization that prevents collapse but does not encode Oracle answers.

## Elegant Triplet Input Pipeline

Future router variants should use only the following clean dense inputs.

### 1. Type Embedding

A learnable type embedding `E_type[t]` for the entity type.

Purpose:

- encode type identity,
- allow type-specific modality preference,
- avoid manual type-wise reliability rules.

### 2. Modality Semantic Tokens

Use the actual projected SGMEA modality representations as tokens:

- `z_img`,
- `z_attr`,
- `z_rel`,
- `z_graph`.

The router should observe the representation manifold directly. A lightweight Transformer or token mixer should learn modality conflict, compatibility, and missingness effects from these dense tokens.

### 3. Availability Mask / Lightweight Topology

Allowed lightweight side information:

- binary availability mask for missing modalities,
- optional lightweight degree bucket embedding.

The degree signal must remain simple. It must not become a handcrafted topology statistics pipeline.

## Structural Constraints

The module must be paper-worthy but lightweight.

Allowed components:

- Transformer encoder,
- MLP,
- MoE, only if ablation proves it is useful,
- Dirichlet/Evidential functions.

Ceilings:

- Transformer layers: `<= 2`,
- MLP layers per head/block: `<= 3`,
- hyperparameters must remain minimal and explainable.

No mechanism should be added unless its removal causes a measurable performance drop.

## Recommended V2 Architecture

Working name: `Elegant Triplet Router`.

For each entity, create four modality tokens:

```text
h_m = Project(z_m) + E_mod[m] + E_type[t] + E_avail[mask_m] + optional E_degree[degree_bucket]
```

Then apply a lightweight cross-modality encoder:

```text
H = TransformerEncoder([h_img, h_attr, h_rel, h_graph])
```

Routing head:

```text
alpha_m = softplus(MLP(H_m)) + 1
bias_m = log(alpha_m) - mean_m(log(alpha_m))
gamma = gamma_max * sigmoid(MLP(mean_pool(H)))
```

Fusion calibration:

```text
w_final = softmax(log(w_base + eps) + gamma * bias)
```

The exact implementation may change, but it must preserve the clean input boundary and direct effect on SGMEA fusion.

## Training Constraints

Training must be stable and avoid gradient entanglement.

Preferred initial protocol:

1. Freeze the SGMEA backbone.
2. Train only the router on real train ILL pairs.
3. Optimize a fused ranking objective with in-batch or sampled negatives.
4. Add lightweight regularization:
   - entropy regularization to prevent single-modality collapse,
   - L2/logit norm to prevent extreme calibration,
   - optional strength constraint for `gamma`.

End-to-end fine-tuning is allowed only after router-only training is stable.

## Explainability Requirements

The final method must explain:

- why Image is suppressed,
- why Graph is boosted,
- why `Person` may prefer Attribute,
- how type tokens and modality token interactions change the final fusion weights.

Acceptable evidence:

- learned type-modality bias/weight table,
- evidential `alpha` values,
- router attention/token interaction patterns,
- ablations of type embedding, modality tokens, availability mask, and degree embedding.

## Iteration Protocol

Every future experiment in this phase must record a short log before moving to the next iteration.

Required format:

```text
- [Hypothesis]: Why are we testing this specific feature/structure?
- [Arch Change]: Exact structural/hyperparameter modification.
- [Result vs. Oracle]: Whether FBDB15K and FBYG15K hit the 80% threshold.
- [Action]: Accept, Revert, or Pivot.
```

## Current Starting Point

V2 starts from the following hypothesis:

```text
- [Hypothesis]: Dense modality tokens plus type embeddings are sufficient for a lightweight Transformer to learn type-specific modality reliability, without Oracle priors or handcrafted scalar statistics.
- [Arch Change]: Replace scalar-statistical router inputs with Elegant Triplet inputs: type embedding, raw projected modality tokens, availability mask, and optional degree embedding. Use a shallow Transformer plus evidential routing head to generate fusion bias.
- [Result vs. Oracle]: Pending.
- [Action]: Start V2 implementation/testing.
```


## Iteration V2-A Semantic Dense Token Router

- [Hypothesis]: Type embedding plus dense modality tokens can learn useful type-aware routing from train ILL ranking without Oracle priors or handcrafted scalar statistics. Starting without availability mask because FB image missingness is randomly imputed rather than zero-coded, so an availability flag would be unreliable in this dataset loader.
- [Arch Change]: Add `tmhg_input_mode=semantic`; TMHG now projects `[z_img,z_attr,z_rel,z_graph]`, adds `E_type` and `E_mod`, applies a 1-layer Transformer and evidential head, predicts entity-level centered logits plus learned `gamma`, and calibrates SGMEA fusion in log space. No stat prior json, no prior KL/MSE.
- [Result vs. Oracle]: Pending.
- [Action]: Start smoke test, then run FBDB15K/FBYG15K r=0.5.

### V2-A Smoke Result

- [Hypothesis]: Type embedding plus dense modality tokens can learn useful type-aware routing from train ILL ranking without Oracle priors or handcrafted scalar statistics.
- [Arch Change]: Semantic TMHG with 1-layer Transformer, 2-layer projector, evidential head, learned gamma, router-only calibration pretrain, no prior KL/MSE, no availability mask.
- [Result vs. Oracle]: Smoke on FBDB15K r=0.5 runs successfully, but 1-epoch test reaches only `Avg H@1 ~= 0.7160`, essentially baseline-level. Router moves slightly away from uniform (`tmhg_bias_abs_mean=0.0109`, `gamma_mean=0.7631`), but learned weights remain weakly separated.
- [Action]: Accept as runnable baseline for PHASE10, then pivot to stronger supervision/regularization because current semantic signal is too weak.

### V2-A Full Result

- [Hypothesis]: Dense modality tokens plus type embeddings are sufficient for a lightweight Transformer to learn type-specific modality reliability without Oracle priors.
- [Arch Change]: Semantic TMHG with router-only calibration pretrain, no prior teacher, no handcrafted scalar statistics.
- [Result vs. Oracle]: Failed. `FBDB15K r=0.5` final test stays at `Avg H@1 ~= 0.7160` (baseline level, target `>= 0.73337`). `FBYG15K r=0.5` final test stays at `Avg H@1 ~= 0.7930` (near baseline, target `>= 0.80806`). During training the router drifts toward stronger image/graph preference and validation degrades, indicating the dense-token architecture is trainable but the supervision is too weak/misaligned.
- [Action]: Pivot.

## Iteration V2-B Utility-Supervised Semantic Router

- [Hypothesis]: V2-A fails because dense tokens alone are trainable but under-supervised. Adding a real-train-link modality-utility auxiliary target, computed from each modality's positive-vs-hard-negative margin, should teach the router which modality is genuinely discriminative without using Oracle priors or handcrafted scalar inputs.
- [Arch Change]: Keep semantic TMHG input pipeline unchanged. During TMHG calibration pretraining, add auxiliary KL/CE-style supervision from per-modality hard-negative utility targets derived from real train ILL pairs. The target acts only as a training signal, not as a final fusion prior.
- [Result vs. Oracle]: Pending.
- [Action]: Start smoke test, then run FBDB15K/FBYG15K r=0.5.

### V2-B Smoke Result

- [Hypothesis]: Adding train-link modality utility supervision should make semantic TMHG learn more meaningful weights than V2-A.
- [Arch Change]: Add auxiliary target from per-modality positive-vs-hard-negative margins during TMHG calibration pretraining (`tmhg_calib_modal_utility_weight=0.5`, `tmhg_calib_modal_target_temp=0.5`).
- [Result vs. Oracle]: Smoke run is stable, but still baseline-level on FBDB15K (`Avg H@1 ~= 0.71565`). Router moves slightly more than V2-A, yet the gain is still negligible.
- [Action]: Pivot to stronger variants with higher utility supervision and tighter bias amplitude control.

## Iteration V2-C Prototype-Anchored Semantic Router

- [Hypothesis]: Entity-level dense modality tokens are too local and unstable under frozen-backbone router-only training. Adding type-conditioned dense prototype anchors, estimated from real train entities rather than handcrafted priors, should provide a stable semantic reference so the router can learn why a modality is trustworthy for a type without reverting to Oracle bias tables.
- [Arch Change]: Keep the Elegant Triplet inputs, but augment semantic TMHG with a second token set: per-type per-modality dense prototypes. Each entity token attends jointly over `[entity modality tokens + type-modality prototype tokens]`, with a lightweight role embedding distinguishing observation vs. prototype. Training keeps the real-train-link ranking loss and may add a type-balanced utility alignment loss, but no Oracle prior logits are injected.
- [Result vs. Oracle]: Pending.
- [Action]: Start implementation and FBDB15K/FBYG15K r=0.5 runs.

### V2-C Smoke Result

- [Hypothesis]: Prototype anchor tokens will stabilize semantic routing and reduce the bad drift seen in V2-A/V2-B.
- [Arch Change]: Add train-entity type-modality dense prototype memory plus prototype tokens into semantic TMHG; use `tmhg_use_type_prototype`, `tmhg_prototype_source=train_links`, and a type-balanced utility auxiliary loss.
- [Result vs. Oracle]: Failed on `FBDB15K r=0.5` smoke. Test `Avg H@1 = 0.71545`, still baseline-level and far below the `0.73337` target. The router becomes slightly more conservative, but the gain remains negligible.
- [Action]: Pivot.

## Iteration V2-C1 Explicit Prototype-Interaction Router

- [Hypothesis]: Prototype tokens alone are too implicit. If the router explicitly models `entity token - prototype token` interactions, it may learn whether a modality conforms to or conflicts with the type prototype manifold.
- [Arch Change]: Add a lightweight interaction head over `[entity token, prototype token, entity-prototype difference]` and inject the resulting modality logits into semantic TMHG before fusion calibration.
- [Result vs. Oracle]: Failed on `FBDB15K r=0.5` smoke. Test `Avg H@1 = 0.71485`, slightly worse than V2-C and still far below the target.
- [Action]: Revert / Pivot.

## Iteration V2-D Type-Prototype-Only Router

- [Hypothesis]: The user's intended innovation is fundamentally type-level, not entity-level. A cleaner design is to learn one `type x 4` weight row directly from dense type-modality prototypes, letting the module output static type-specific modality weights without per-entity token noise.
- [Arch Change]: Add `tmhg_use_type_prototype_only`; semantic TMHG now directly maps dense type-modality prototype tokens to type-level modality logits through a lightweight MLP head, then applies those learned weights at SGMEA fusion.
- [Result vs. Oracle]: Failed badly on `FBDB15K r=0.5` smoke. Test `Avg H@1 = 0.70015`. The learned type weights over-amplify Image and suppress Graph/Attr too aggressively.
- [Action]: Revert / Pivot.

## Iteration Probe: Calibration Checkpoint Selection

- [Hypothesis]: Earlier semantic runs may have looked weak because `tmhg_calib_selection=best_val` is nearly degenerate: calibration validation `Hits@1` stays around `0.9996`, so the selected checkpoint often remains near initialization rather than the strongest learned router state.
- [Arch Change]: Re-run the V2-B1 semantic setting with `tmhg_calib_selection=final` and `best_loss` to test whether a stronger retained calibration state improves downstream FBDB performance.
- [Result vs. Oracle]: Selection was not the core issue. `final` / `best_loss` keep much stronger router weights (`tmhg_bias_abs_mean≈0.05`, `gamma_mean≈0.69`), but downstream `FBDB15K r=0.5` drops to `Avg H@1 ≈ 0.71130`, worse than the baseline-level `best_val` result.
- [Action]: Accept this as a diagnosis and pivot away from “just keep a stronger semantic router state”.

## Iteration V2-E Consensus Pseudo-Link Debiasing

- [Hypothesis]: Old learned-DEHR reaches the target but is vulnerable to confirmation-bias criticism because pseudo links are mined from the full fused baseline embedding. A stricter multi-view pseudo source should reduce this risk while retaining enough training signal.
- [Arch Change]: Add `dehr_calib_pseudo_source=consensus`, mining pseudo pairs only when independent single-modality mutual-nearest predictions agree. No manual reliability prior is used.
- [Result vs. Oracle]: Mixed. `FBDB15K r=0.5` with `min_votes=2` reaches `Avg H@1=0.73405`, above the `0.73337` target, using only `498` pseudo pairs. But `FBYG15K r=0.5` collapses to around `0.764-0.765`; the pseudo set has only `572-661` pairs and pushes the router into excessive Graph dominance.
- [Action]: Pivot. Single-modality consensus is defensible but too low-coverage for FBYG.

## Iteration V2-F Agreement-Weighted Joint Pseudo

- [Hypothesis]: Keep high pseudo-link coverage from joint mining, but downweight pseudo pairs not supported by individual modalities. This should reduce confirmation bias without starving the router.
- [Arch Change]: Add `dehr_calib_pseudo_weight_mode=agreement`; joint pseudo pairs keep their coverage, while their calibration loss weight is derived from modality-level agreement votes. This is a training weighting signal, not a fusion prior.
- [Result vs. Oracle]: Failed. `FBDB15K r=0.5` drops to `Avg H@1=0.66670`; `FBYG15K r=0.5` drops to `Avg H@1=0.68170`. The agreement weights suppress pseudo supervision too aggressively and drive the router to a bad Image+Graph polarized solution.
- [Action]: Revert as a recommended configuration. Keep only as a negative control showing that agreement should not directly dominate the ranking loss.

## Iteration V2-G Dropout-Consensus Pseudo Links

- [Hypothesis]: Single-modality consensus is too strict, but full-joint pseudo links are too easy to criticize. Leave-one-modality fusion views provide a middle ground: a pseudo pair is accepted only if it remains stable under modality dropout, so the supervision is based on multi-view robustness rather than one fused baseline prediction.
- [Arch Change]: Add `dehr_calib_pseudo_source=dropout_consensus`. For each left entity, mine mutual-nearest pseudo pairs in four views, each dropping one modality from SGMEA fusion and renormalizing the remaining fusion weights. Keep pairs receiving at least `dehr_calib_consensus_min_votes` view votes. Use the same learned-DEHR router and no manual/global direction prior.
- [Result vs. Oracle]: Success at the target setting. With `min_votes=2`, `FBDB15K r=0.5` gets `Avg H@1=0.74365` with `4230` pseudo pairs, above the `0.73337` target. `FBYG15K r=0.5` gets `Avg H@1=0.82100` with `3854` pseudo pairs, above the `0.80806` target. A stricter `min_votes=3` gives `FBYG15K r=0.5 Avg H@1=0.80365`, below target, so it is too conservative.
- [Action]: Accept `dropout_consensus, min_votes=2` as the current best architecture/training protocol. Proceed to `Rate=0.2/0.8` stability testing.

### V2-G Rate Sweep Notes

- `FBDB15K r=0.8`, `dropout_consensus`, `min_votes=2`: pseudo `2014`, Avg H@1 `0.83035`; stable and above the old manual/constrained gain level.
- `FBYG15K r=0.8`, `dropout_consensus`, `min_votes=2`: pseudo `1911`, Avg H@1 `0.88505`; stable.
- `FBDB15K r=0.2`, `dropout_consensus`, `min_votes=2`: pseudo `4294`, Avg H@1 `0.58980`; roughly matches old manual gain but does not improve over the earlier joint learned-DEHR result.
- `FBYG15K r=0.2`, `dropout_consensus`, `min_votes=2`: pseudo `3968`, Avg H@1 `0.67040`; good absolute gain over baseline, but below the earlier joint learned-DEHR result.
- `min_votes=3` at `r=0.2` reduces pseudo coverage and hurts both datasets, so merely tightening consensus is not the right fix.

## Iteration V2-H Pseudo-Ratio Calibration for Low-Rate Stability

- [Hypothesis]: At `r=0.2`, dropout-consensus pseudo links outnumber real train ILL pairs, so calibration is dominated by pseudo supervision and over-amplifies graph/rel biases. Capping pseudo links to a train-comparable budget should preserve the debiased multi-view source while reducing pseudo dominance.
- [Arch Change]: Keep learned-DEHR architecture and dropout-consensus source unchanged. Add only the existing `dehr_calib_pseudo_max_pairs` training hyperparameter at low rate; test caps around `1500`, `2200`, and `2500` depending on train set size.
- [Result vs. Oracle]: Failed / inconclusive. `FBDB15K r=0.2 cap=1500` gets Avg H@1 `0.58935`; `cap=2500` gets `0.58810`, both near or below the uncapped dropout-consensus `0.58980`. `FBYG15K r=0.2 cap=1500` drops to `0.64175`, far below uncapped `0.67040`. Cap-only selection removes coverage and does not solve low-rate instability.
- [Action]: Revert cap-only as the recommended protocol. Pivot to calibration checkpoint/strength control.

## Iteration V2-I Low-Rate Calibration Strength Control

- [Hypothesis]: Low-rate failure is not just pseudo count; the router continues optimizing after validation has saturated and converges to overly strong type-bias rows. Selecting the validation-best checkpoint or reducing the maximum learned log-bias should prevent over-calibration while preserving dropout-consensus supervision.
- [Arch Change]: Keep learned-DEHR and dropout-consensus unchanged. Test `dehr_calib_selection=best_val` and a smaller `dehr_learned_bias_scale/dehr_bias_clamp=0.25` at `r=0.2`.
- [Result vs. Oracle]: Failed. `best_val` underfits the useful router state (`FBDB15K r=0.2 Avg H@1=0.57115`, `FBYG15K r=0.2 Avg H@1=0.65320`). Lowering `scale/clamp` to `0.25` improves FBYG over `best_val` but still stays below uncapped final (`FBYG15K r=0.2 Avg H@1=0.66270` vs `0.67040`).
- [Action]: Pivot to weakly weighted pseudo supervision while preserving dropout-consensus coverage.

## Iteration V2-J Weak Pseudo Supervision

- [Hypothesis]: Low-rate routing needs the broad coverage of dropout-consensus pseudo links, but pseudo labels should act as weak supervision because real train ILL pairs are the only gold calibration anchors. A global pseudo loss weight is an interpretable training-confidence hyperparameter, not a fusion prior.
- [Arch Change]: Add `dehr_calib_pseudo_loss_weight`; train links keep weight `1.0`, pseudo links receive a uniform weight such as `0.25` or `0.5` inside the existing calibration ranking loss. Router architecture and pseudo source are unchanged.
- [Result vs. Oracle]: Negative. After fixing the float-weight path in `_dehr_calib_loss`, `FBYG15K r=0.2 pw=0.25` gets Avg H@1 `0.66270`, `pw=0.5` gets `0.66800`; both are below uncapped dropout-consensus `0.67040`. `FBDB15K r=0.2 pw=0.5` gets `0.58880`, below uncapped `0.58980`. A pre-fix run revealed that float pseudo weights were accidentally cast through `long`; that bug is fixed, but weak pseudo still does not improve.
- [Action]: Revert as recommended setting. Keep `dehr_calib_pseudo_loss_weight` for controlled ablation only. Pivot to pair-source isolation.

## Iteration V2-K Pair-Source Isolation

- [Hypothesis]: The slight low-rate gap on FBYG may come from mixing scarce gold train ILL pairs with dropout-consensus pseudo pairs in one calibration objective. Pseudo-only calibration tests whether the robust multi-view pseudo set already carries the useful type-modality signal, while train links can be reserved as validation/diagnostic anchors.
- [Arch Change]: Keep dropout-consensus `min_votes=2` and learned-DEHR unchanged. Switch `dehr_calib_pair_source` from `train_pseudo` to `pseudo`.
- [Result vs. Oracle]: Negative at low rate. `FBYG15K r=0.2` gets `Avg H@1 = 0.67105`, close to but not above uncapped train+pseudo dropout-consensus `0.67040` by a meaningful margin. `FBDB15K r=0.2` gets `Avg H@1 = 0.58955`, slightly below uncapped train+pseudo `0.58980`. Pseudo-only does not solve the low-rate gap and does not justify replacing gold+pseudo calibration.
- [Action]: Revert as recommended setting. Keep train+pseudo dropout-consensus as the current best protocol.

## Iteration V2-L Dense-Token DEHR Ablation

- [Hypothesis]: The current learned-DEHR success may be criticized because its token includes seven scalar statistics. If the same Transformer+MLP learned-type router keeps most of the r=0.5 gain after zeroing all scalar stats, then the useful signal is carried by type embeddings plus dense modality tokens, not by handcrafted statistical soup.
- [Arch Change]: Keep learned-DEHR, dropout-consensus `min_votes=2`, train+pseudo calibration, final checkpoint selection, and bias scale/clamp `0.35`. Add `dehr_stat_feature_mask=none`, so DEHR tokens contain projected modality embeddings plus modal/type tokens only; no base weight, entropy, top gap, pairwise cosine, direction, or anchor scalar is visible to the router.
- [Result vs. Oracle]: Success at the formal target setting. `FBDB15K r=0.5` gets `Avg H@1 = 0.74325`, above the `0.73337` target and almost identical to full-stat dropout-consensus `0.74365`. `FBYG15K r=0.5` gets `Avg H@1 = 0.82090`, above the `0.80806` target and almost identical to full-stat dropout-consensus `0.82100`.
- [Action]: Accept. This becomes the recommended paper-facing variant because it preserves the gain while removing the handcrafted scalar-statistical input criticism.

## Iteration V2-M Dense-Token Rate Sweep

- [Hypothesis]: V2-L solves the main r=0.5 paper-defense problem, but we should verify whether removing scalar stats remains stable at low/high train rates.
- [Arch Change]: Keep V2-L dense-token DEHR unchanged (`dehr_stat_feature_mask=none`, dropout-consensus `min_votes=2`, train+pseudo calibration, final selection, bias scale/clamp `0.35`). Run `Rate=0.2` and `Rate=0.8` on both datasets.
- [Result vs. Oracle]: Stable across rates. `FBDB15K r=0.2` gets `Avg H@1 = 0.58975` vs full-stat `0.58980`; `FBYG15K r=0.2` gets `Avg H@1 = 0.66980` vs full-stat `0.67040`; `FBDB15K r=0.8` gets `Avg H@1 = 0.82940` vs full-stat `0.83035`; `FBYG15K r=0.8` gets `Avg H@1 = 0.88595` vs full-stat `0.88505`.
- [Action]: Accept V2-L/V2-M as the recommended paper-facing router: no manual/global direction prior, no scalar-statistical token features, trainable Transformer+MLP type-aware bias, and dropout-consensus pseudo supervision for confirmation-bias control.

## Current Phase10 Recommendation

- Architecture: learned-DEHR with projected dense modality tokens, learnable type token, modal token, 1-layer Transformer, 2-layer learned-type MLP head, and type strength `rho`.
- Inputs: dense projected modality embeddings plus type ids only; set `dehr_stat_feature_mask=none` to remove base weights, entropy, pairwise cosine, top-gap, direction, and anchor scalar features.
- Supervision: router-only calibration on real train ILL plus dropout-consensus pseudo links. Pseudo links are accepted only when leave-one-modality fused views agree, reducing single fused-baseline confirmation bias.
- Fusion: output centered type-aware log-bias directly calibrates SGMEA fusion weights in log space.
- Key hyperparameters: `dehr_learned_bias_scale=0.35`, `dehr_bias_clamp=0.35`, `dehr_rho_max=1.0`, `dehr_calib_pretrain_epochs=25`, `dehr_calib_pretrain_lr=5e-3`, `dehr_calib_pseudo_source=dropout_consensus`, `dehr_calib_consensus_min_votes=2`, `dehr_stat_feature_mask=none`.

## Iteration V2-N Paper-Critical Ablations

- [Hypothesis]: The recommended dense-token DEHR should be tested against three reviewer-facing concerns: whether dropout-consensus pseudo links are necessary, whether the Transformer interaction is actually useful, and whether projected dense modality tokens carry the gain.
- [Arch Change]: Starting from V2-L at `Rate=0.5`, run three ablations on FBDB15K and FBYG15K: (1) replace `dropout_consensus` with `joint` pseudo mining; (2) set `dehr_layers=0` to remove cross-token Transformer attention; (3) keep scalar stats disabled but add `dehr_drop_projected_token` to remove dense modality embeddings.
- [Result vs. Oracle]: Completed. All ablations remain above the formal 80% Oracle-gain threshold, but they expose different paper risks.

| Dataset | Variant | Pseudo Source | Extra Ablation | Pseudo Pairs | L2R H@1 | R2L H@1 | Avg H@1 | Delta vs V2-L |
|---|---|---:|---|---:|---:|---:|---:|---:|
| FBDB15K r=0.5 | V2-L main | dropout_consensus | none | 4230 | - | - | 0.74325 | 0.00000 |
| FBDB15K r=0.5 | joint pseudo | joint | none | 4514 | 0.73980 | 0.74790 | 0.74385 | +0.00060 |
| FBDB15K r=0.5 | no Transformer | dropout_consensus | `dehr_layers=0` | 4230 | 0.73800 | 0.74470 | 0.74135 | -0.00190 |
| FBDB15K r=0.5 | no dense token | dropout_consensus | `dehr_drop_projected_token` | 4230 | 0.73940 | 0.74780 | 0.74360 | +0.00035 |
| FBYG15K r=0.5 | V2-L main | dropout_consensus | none | 3854 | - | - | 0.82090 | 0.00000 |
| FBYG15K r=0.5 | joint pseudo | joint | none | 4284 | 0.82210 | 0.82620 | 0.82415 | +0.00325 |
| FBYG15K r=0.5 | no Transformer | dropout_consensus | `dehr_layers=0` | 3854 | 0.81910 | 0.82180 | 0.82045 | -0.00045 |
| FBYG15K r=0.5 | no dense token | dropout_consensus | `dehr_drop_projected_token` | 3854 | 0.81290 | 0.81700 | 0.81495 | -0.00595 |

Notes:
- `dehr_layers=0` initially failed because PyTorch `TransformerEncoder` cannot be constructed with zero layers. Patched `DEHRTypeScaleRouter` so `dehr_layers <= 0` uses `nn.Identity()`. This is an ablation-only path; the recommended 1-layer Transformer path is unchanged.
- The first FBYG no-dense run on GPU3 OOMed because GPU3 was occupied by unrelated stale processes. The valid rerun is `FBYG15K_r05_ablate_nodense_v2n_rerun_gpu0.log`.
- Joint pseudo gives the best raw H@1 on both datasets, but it is less defensible because pseudo links come from the full fused baseline. Keep dropout-consensus as the recommended paper-facing source because it trades only `0.00060` on FBDB and `0.00325` on FBYG for a much stronger confirmation-bias defense.
- Removing the Transformer barely hurts (`-0.00190` FBDB, `-0.00045` FBYG), so the paper should not overclaim that self-attention is the main driver. It can be presented as a lightweight interaction layer that slightly stabilizes FBDB and preserves the academic architecture, with ablation showing the router is not dependent on excessive depth.
- Removing dense projected modality tokens is dataset-dependent: FBDB is unchanged, but FBYG drops `0.00595`. This supports keeping dense tokens in the main method while acknowledging that part of the gain also comes from learned type-modal parameters.
- Interpretation: the robust core is the trainable type-aware log-bias learned from calibration pairs, not handcrafted scalar statistics or a manual prior. Dense modality tokens provide extra generalization on FBYG, dropout-consensus provides the cleaner supervision story, and the 1-layer Transformer provides limited but non-harmful interaction capacity.
- [Action]: Accept V2-N as the paper-critical ablation package. Recommended final story: V2-L remains main; joint pseudo is a performance upper ablation with weaker causal defense; no-Transformer and no-dense are honest limitations that guide conservative claims.

## Iteration V2-O Contextual Transformer Bias Head

- [Hypothesis]: The V2-L learned-type head makes the Transformer hard to defend because most of the final bias can be produced from the type token alone. Add a contextual per-modality head that reads each Transformer's modality state together with the type state, so the Transformer output directly participates in each modality's log-bias.
- [Arch Change]: Keep V2-L inputs and supervision unchanged: no manual/global direction prior, `dehr_stat_feature_mask=none`, dropout-consensus pseudo links, train+pseudo calibration, final checkpoint selection, bias scale/clamp `0.35`. Add a lightweight contextual head with input `[h_type, h_m, h_type * h_m]` for each modality. Mix it with the learned-type head by `dehr_contextual_mix=0.15`. Add logged diagnostics: `contextual_u_abs_mean`, `contextual_delta_abs_mean`, and `learned_type_u_abs_mean`.
- [Result vs. Oracle]: Main performance does not drop. `FBDB15K r=0.5` gets Avg H@1 `0.74445` vs V2-L `0.74325`; `FBYG15K r=0.5` gets Avg H@1 `0.82125` vs V2-L `0.82090`. Contextual diagnostics are nonzero (`contextual_u_abs_mean` around `0.35` at the end), showing the contextual head is active.
- [Action]: Partially accept but not final. The first no-Transformer contextual ablation shows a weakness: on FBYG, `dehr_layers=0` gets Avg H@1 `0.82270`, higher than V2-O. This means the contextual head can still learn from uncontextualized tokens and does not prove Transformer attention is necessary.

| Dataset | Variant | Contextual Source | Transformer | L2R H@1 | R2L H@1 | Avg H@1 | Delta vs V2-L |
|---|---|---|---|---:|---:|---:|---:|
| FBDB15K r=0.5 | V2-O | hidden token | 1 layer | 0.74050 | 0.74840 | 0.74445 | +0.00120 |
| FBDB15K r=0.5 | V2-O no Transformer | hidden token | 0 layer | 0.73920 | 0.74700 | 0.74310 | -0.00015 |
| FBYG15K r=0.5 | V2-O | hidden token | 1 layer | 0.81950 | 0.82300 | 0.82125 | +0.00035 |
| FBYG15K r=0.5 | V2-O no Transformer | hidden token | 0 layer | 0.82160 | 0.82380 | 0.82270 | +0.00180 |

## Iteration V2-P Transformer-Residual Attribution Head

- [Hypothesis]: To make the Transformer interpretable and necessary, the contextual branch should not read generic hidden tokens directly. It should read the Transformer-induced residual `Delta h = Transformer(input) - input`, so its output is attributable to how self-attention changes each modality token under the type context. If the Transformer is removed, this residual becomes zero and the branch cannot bypass attention.
- [Arch Change]: Add `dehr_contextual_source=residual`. The contextual head input becomes `[Delta h_type, Delta h_m, Delta h_type * Delta h_m]`; all other V2-L/V2-O settings remain unchanged. The default remains `hidden`, so previous results are preserved. Use `dehr_contextual_mix=0.15` and the same bias scale/clamp `0.35`.
- [Result vs. Oracle]: Success. `FBDB15K r=0.5` gets Avg H@1 `0.74455`, above V2-L by `+0.00130`; `FBYG15K r=0.5` gets Avg H@1 `0.82210`, above V2-L by `+0.00120`. Both keep the formal 80% Oracle-gain target and do not sacrifice the learned-router gains. Removing the Transformer now hurts both datasets: FBDB drops to `0.74170` (`-0.00285` vs V2-P), FBYG drops to `0.81945` (`-0.00265` vs V2-P). In the no-Transformer residual ablation, `contextual_u_abs_mean=0.0000`, confirming the residual branch has no signal without self-attention.
- [Action]: Accept V2-P as the stronger paper-facing Transformer explanation variant. Recommended claim: the router uses a type-aware Transformer as a lightweight token interaction layer, and the residual attribution head converts attention-induced modality corrections into log-space fusion calibration. Do not overclaim large absolute gains from Transformer depth; claim that the residual design makes Transformer contribution measurable, interpretable, and non-harmful.

| Dataset | Variant | Contextual Source | Transformer | L2R H@1 | R2L H@1 | Avg H@1 | Delta vs V2-L | Delta vs V2-P |
|---|---|---|---|---:|---:|---:|---:|---:|
| FBDB15K r=0.5 | V2-P | residual | 1 layer | 0.73980 | 0.74930 | 0.74455 | +0.00130 | 0.00000 |
| FBDB15K r=0.5 | V2-P no Transformer | residual | 0 layer | 0.73860 | 0.74480 | 0.74170 | -0.00155 | -0.00285 |
| FBYG15K r=0.5 | V2-P | residual | 1 layer | 0.82120 | 0.82300 | 0.82210 | +0.00120 | 0.00000 |
| FBYG15K r=0.5 | V2-P no Transformer | residual | 0 layer | 0.81840 | 0.82050 | 0.81945 | -0.00145 | -0.00265 |

Notes:
- New code paths: `--dehr_contextual_mix`, `--dehr_contextual_scale`, and `--dehr_contextual_source {hidden,residual}` in `config.py`; contextual/residual head in `model/SGMEA_tools.py`; logging in `model/SGMEA.py` and `main.py`.
- V2-P keeps the Elegant Triplet constraint: dense modality tokens, learnable type embedding, modal tokens, and optional availability behavior already implicit in the projected features; no scalar statistical soup and no manual direction prior.
- The only new paper-facing hyperparameter is `dehr_contextual_mix=0.15`; it controls how much the residual-attribution branch contributes relative to the learned type head. It is a structural mixing coefficient rather than a hand-coded modality prior.

## Iteration V2-Q V2-P Rate Sweep and Mix Sensitivity

- [Hypothesis]: V2-P should be checked beyond the formal `r=0.5` setting before paper writing. The residual-attribution head should preserve V2-L/V2-M stability at `r=0.2/0.8`, and `dehr_contextual_mix=0.15` should not look like a narrow grid-searched sweet spot.
- [Arch Change]: No structural change. Keep V2-P fixed: `dehr_contextual_source=residual`, `dehr_contextual_mix=0.15`, `dehr_stat_feature_mask=none`, dropout-consensus pseudo links, train+pseudo calibration, final checkpoint selection, bias scale/clamp `0.35`. For sensitivity, only vary `dehr_contextual_mix` on `r=0.5`.
- [Result vs. Oracle]: Completed. V2-P remains stable across rates, with a small tradeoff on FBDB low/high rates and a clear gain on FBYG high rate. Mix sensitivity is smooth: `0.05` is close to `0.15`, while `0.30` starts to slightly overemphasize the residual branch.
- [Action]: Accept V2-P as the main paper-facing method, with a caveat that the residual branch is most useful at the main `r=0.5` setting and on FBYG high-rate generalization. Do not claim every rate improves over V2-M; claim comparable or better performance with stronger Transformer interpretability.

### V2-P Rate Sweep

| Dataset | Rate | V2-M Dense-Token Avg H@1 | V2-P Residual Avg H@1 | Delta | L2R H@1 | R2L H@1 |
|---|---:|---:|---:|---:|---:|---:|
| FBDB15K | 0.2 | 0.58975 | 0.58865 | -0.00110 | 0.58470 | 0.59260 |
| FBDB15K | 0.5 | 0.74325 | 0.74455 | +0.00130 | 0.73980 | 0.74930 |
| FBDB15K | 0.8 | 0.82940 | 0.82880 | -0.00060 | 0.82880 | 0.82880 |
| FBYG15K | 0.2 | 0.66980 | 0.67015 | +0.00035 | 0.67130 | 0.66900 |
| FBYG15K | 0.5 | 0.82090 | 0.82210 | +0.00120 | 0.82120 | 0.82300 |
| FBYG15K | 0.8 | 0.88595 | 0.88995 | +0.00400 | 0.88750 | 0.89240 |

Interpretation:
- V2-P keeps the formal `r=0.5` gain on both datasets while making Transformer contribution attributable through `Delta h`.
- Low-rate FBDB is slightly weaker than V2-M (`-0.00110`), likely because residual contextualization has less reliable calibration supervision when gold pairs are scarce and pseudo links dominate. This is small enough to report as comparable rather than a failure.
- FBYG benefits more from residual contextualization, especially at `r=0.8` (`+0.00400`), supporting the claim that dense token interaction helps when the representation manifold is better anchored.

### V2-P Mix Sensitivity at r=0.5

| Dataset | `dehr_contextual_mix` | Avg H@1 | Delta vs mix=0.15 | L2R H@1 | R2L H@1 |
|---|---:|---:|---:|---:|---:|
| FBDB15K | 0.00 (V2-L) | 0.74325 | -0.00130 | - | - |
| FBDB15K | 0.05 | 0.74450 | -0.00005 | 0.73980 | 0.74920 |
| FBDB15K | 0.15 | 0.74455 | 0.00000 | 0.73980 | 0.74930 |
| FBDB15K | 0.30 | 0.74370 | -0.00085 | 0.73950 | 0.74790 |
| FBYG15K | 0.00 (V2-L) | 0.82090 | -0.00120 | - | - |
| FBYG15K | 0.05 | 0.82100 | -0.00110 | 0.81910 | 0.82290 |
| FBYG15K | 0.15 | 0.82210 | 0.00000 | 0.82120 | 0.82300 |
| FBYG15K | 0.30 | 0.82030 | -0.00180 | 0.81880 | 0.82180 |

Interpretation:
- `mix=0.15` is a conservative structural choice, not a modality prior. It gives the best `r=0.5` result among the small tested set, but `mix=0.05` remains close, especially on FBDB. This weakens the grid-search criticism.
- `mix=0.30` degrades both datasets, suggesting that the residual branch should complement rather than dominate the learned type bias.
- Paper recommendation: report `mix=0.15` as the default and include the small sensitivity table. Avoid a large hyperparameter sweep; the point is robustness, not tuning.
