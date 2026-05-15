# DEHR-TypeScale Experiment Plan

Date: 2026-05-12

Working repo: `/gly/tongqiang/dongyufeng/Mode-Test/SGMEA`

## 1. Goal

The next implementation target is a data-conditioned modality router that can approach the strong type-specific prior without exposing the method as a 16-value grid search. The core constraint is:

- FBDB15K and FBYG15K must use the same model, same formula, and same training protocol.
- The module should have enough method depth to be written as a real contribution.
- The final prior-like bias should be close to the current diagnostic prior, but it should be generated through a learnable evidence module plus a small number of interpretable calibration parameters.

## 2. Method Name

Type-Calibrated Dirichlet Evidence Hyper-Router, abbreviated as **DEHR-TypeScale**.

## 3. High-Level Idea

The router is decomposed into two parts:

1. **Evidence direction**: a support-conditioned module estimates which modalities are reliable for each entity type.
2. **Type-level strength**: each entity type owns one scalar calibration parameter that controls how strongly the learned evidence direction should affect routing.

This changes the parameter story from 16 manually selected type-modality values to:

- a shared evidence generator that predicts the 4-modal direction from train support evidence;
- only 4 type-level strength parameters, one for each coarse type.

The 4 scalar parameters are easier to explain: they represent the concentration or confidence level of each type's modality preference.

## 4. Inputs

For each coarse type `t`, collect support pairs from Train ILL:

```text
S_t = {(e_i^L, e_i^R, y_i=t)}
```

For each aligned support pair and modality `m in {img, attr, rel, graph}`, build an evidence token:

```text
z_{i,m} = [
  P_m h_{i,m}^L,
  P_m h_{i,m}^R,
  |P_m h_{i,m}^L - P_m h_{i,m}^R|,
  P_m h_{i,m}^L * P_m h_{i,m}^R,
  w_base_{i,m}
]
```

Implementation notes:

- `P_m` should be modality-specific and trainable.
- SGMEA backbone features are detached when training the router.
- Missing or zero-valued modality features must be handled with eps-safe normalization.
- `w_base` should be the original SGMEA attention/fusion weight before router intervention.

## 5. Evidence Generator

Use a lightweight support-conditioned Transformer/hypernetwork:

```text
H_t = Transformer([TypeToken_t, ModToken_img, ModToken_attr, ModToken_rel, ModToken_graph, EvidenceTokens_t])
o_t = Linear(H_t[TypeToken]) in R^4
```

`o_t` is not directly used as a routing bias. It is interpreted as modality evidence logits.

Recommended first implementation:

- 1 Transformer layer
- hidden size 64 or 128
- 4 attention heads if hidden size is 64/128
- dropout 0.1
- max support tokens per type: configurable, default 256 or 512 pairs sampled from Train ILL

## 6. Dirichlet Evidence Mapping

Convert module output to non-negative evidence:

```text
e_{t,m} = softplus(o_{t,m})
alpha_{t,m} = alpha_0 + e_{t,m}
```

Default:

```text
alpha_0 = 1.0
```

Compute modality direction with the Dirichlet expected log probability:

```text
u_{t,m} = digamma(alpha_{t,m}) - mean_k digamma(alpha_{t,k})
```

`u_t` is centered, so it controls relative modality preference rather than global logit shift.

## 7. Type-Level Strength Calibration

Introduce one scalar strength per coarse type:

```text
rho_t = rho_min + (rho_max - rho_min) * sigmoid(eta_t)
b_{t,m} = rho_t * u_{t,m}
```

Recommended defaults:

```text
rho_min = 0.0
rho_max = 3.0
eta_t initialized so rho_t is around 1.0
```

This gives exactly 4 interpretable calibration parameters:

```text
rho_Person, rho_Place, rho_Organization, rho_CreativeWork
```

Interpretation:

- `u_t` decides the direction of type-specific modality preference.
- `rho_t` decides how decisive that type's routing should be.

## 8. Routing Formula

The final routed fusion weight is:

```text
w_final = softmax(log(w_base + eps) + b_type[type])
```

where:

```text
b_type[type] = rho_type * u_type
```

Unknown or unsupported types should fall back to:

```text
b = [0, 0, 0, 0]
```

## 9. Training Protocol

Start from a trained SGMEA baseline checkpoint.

Freeze:

- SGMEA backbone
- modality encoders
- base fusion module unless explicitly needed for extracting `w_base`

Train:

- modality projections `P_m`
- evidence Transformer
- output evidence head
- 4 type strength parameters `eta_t`

Primary loss:

```text
L_main = InfoNCE or hard-negative ranking loss on Train/Dev ILL
```

Regularizers:

```text
L_kl = KL(Dir(alpha_t) || Dir(1))
L_rho = sum_t (log(rho_t + eps))^2
L_total = L_main + lambda_kl * L_kl + lambda_rho * L_rho
```

Recommended initial values:

```text
lambda_kl = 1e-4 or 1e-3
lambda_rho = 1e-4
router_lr = 5e-4, 1e-3
batch_size = 2048
```

## 10. Two-Stage Variant

If one-stage training is unstable, use this variant:

Stage A: Evidence Direction Learning

- Fix `rho_t = 1.0`
- Train only evidence generator
- Goal: learn stable direction `u_t`

Stage B: Type Strength Calibration

- Freeze evidence generator or use a lower LR
- Train only `eta_t`, optionally with the final evidence head
- Goal: move bias magnitude closer to the strong diagnostic prior without changing direction arbitrarily

This is the preferred variant if the generated bias drifts away from the known strong prior shape.

## 11. Ablations

Required ablations:

1. SGMEA baseline.
2. ZeroInit TMR: 16 type-modality parameters from zero, no evidence generator.
3. DEHR-Base: evidence generator without `rho_t`.
4. DEHR-TypeScale: full method with 4 type strength parameters.
5. DEHR-TypeScale two-stage: freeze direction, calibrate strength.
6. Oracle/diagnostic prior: upper-bound probe only, not the main method.

Useful negative ablation:

- Entity-level MLP residual, if it underperforms, to justify that instance-level residual routing is noisy in this setting.

## 12. Reporting Metrics

Main alignment metrics:

- Hits@1
- Hits@10
- MRR

Router analysis:

- learned `rho_t`
- generated `u_t`
- final `scale = exp(b_t)`
- MAE/correlation to diagnostic prior as a probe, not as training objective
- per-type modality weights before/after routing
- weight entropy before/after routing

## 13. Expected Behavior

Expected qualitative pattern:

- Image evidence should generally be suppressed on FBDB/FBYG because visual alignment quality is weak.
- Graph and/or attribute should be amplified depending on type and dataset.
- `rho_t` should differ across types, reflecting type-specific confidence in modality preference.

Expected quantitative behavior:

- DEHR-TypeScale should outperform SGMEA baseline.
- It may not exactly match the diagnostic prior, but should be closer than plain zero-init TMR or DEHR-Base.
- The two-stage variant should be the closest to the diagnostic prior if Stage A learns the right direction.

## 14. Risk Points

1. If evidence tokens are too many, training will be slow. Use type-level support sampling.
2. If `rho_t` grows too large, routing may collapse. Keep bounded sigmoid parameterization.
3. If evidence direction is weak, `rho_t` can only amplify noise. Check `u_t` before trusting final metrics.
4. If FBYG type quality is lower than FBDB, generated priors may be noisier. Unknown/inconsistent types should be masked to zero bias.

## 15. Implementation Order

1. Inspect current SGMEA CDMR/type-bias hooks.
2. Add a reusable DEHR router module.
3. Add support evidence extraction from Train ILL and type jsonl.
4. Add routing mode flags:

```text
--router_mode none|tmr|dehr|dehr_typescale
--dehr_two_stage
--dehr_support_per_type
--dehr_rho_max
--dehr_kl_weight
--dehr_rho_weight
```

5. Add checkpoint logging of:

```text
b_type
rho_t
u_t
alpha_t
scale_t = exp(b_type)
```

6. Run FBDB and FBYG under the same protocol.

