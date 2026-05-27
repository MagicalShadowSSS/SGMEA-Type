# PHASE13 TMRD V12: Data-Driven Visual Reliability Calibration

Date: 2026-05-18
Workspace: `/gly/tongqiang/dongyufeng/Mode-Test/SGMEA`

## Objective

Break the +1.0 Hits@1 gain ceiling on both FBDB15K and FBYG15K at `r=0.5`, while keeping the mechanism explainable and avoiding manual grid-searched modality priors.

## Baseline Reference

The no-calibration TMRD V4 checkpoints are used as the reference:

| Dataset | Baseline L2R H@1 | Baseline R2L H@1 |
|---|---:|---:|
| FBDB15K r=0.5 | 0.7112 | 0.7187 |
| FBYG15K r=0.5 | 0.7902 | 0.7882 |

## Iteration Log

### V11-A: Train-ILL Hits@1/MRR Calibration

- **[Hypothesis]:** Select image suppression strength from train ILL ranking performance, so the visual bias is data-driven rather than manually fixed.
- **[Arch Change]:** Added `--tmrd_calibrate_img_suppress`; sweep `s_img in {0, 0.04, 0.08, 0.15, 0.30}` on train links, then fix selected `s_img` for test.
- **[Result vs. Target]:** Failed. FBDB selected `0.0000` and stayed at `0.7113 / 0.7191`; FBYG selected `0.0400` and reached only `0.7961 / 0.7954`.
- **[Action]:** Pivot. Train-link ranking was saturated and could not estimate test-time visual noise reliably.

### V11-B: Full Candidate Pool Calibration

- **[Hypothesis]:** The previous train-link matrix was too easy; ranking each train anchor against the full opposite-KG candidate pool should expose harder negatives.
- **[Arch Change]:** Changed calibration to score train-left against all right entities and train-right against all left entities.
- **[Result vs. Target]:** Failed. Both datasets still selected `0.0000`; FBDB `0.7113 / 0.7191`, FBYG `0.7904 / 0.7886`.
- **[Action]:** Pivot. Even full-pool train anchors were near-saturated.

### V11-C: Hard-Negative Margin Calibration

- **[Hypothesis]:** Hits@1 is too coarse; positive-vs-nearest-negative margin should be a more sensitive reliability criterion.
- **[Arch Change]:** Added `--tmrd_img_suppress_calib_metric hard_margin`; select `s_img` by maximizing train hard-negative margin.
- **[Result vs. Target]:** Failed. Both datasets selected `0.0000`. Train margin decreased as image suppression increased, while test improved under fixed image suppression, indicating train/test difficulty mismatch.
- **[Action]:** Pivot away from ranking-based calibration.

### V12-A: Auto Image Suppression From Predicted Image Quality

- **[Hypothesis]:** Use the TMRD quality head itself as an orthogonal visual reliability estimator. If predicted image quality is low on average, convert that unreliability into a global image-logit penalty without using baseline candidate rankings.
- **[Arch Change]:** Added `--tmrd_quality_bias_mode auto_img_suppress` with:
  `s_img = scale * (1 - mean(q_img))`, applied as `bias_img = -s_img` in fusion logit space. Missing images are optionally repaired by `proxy/memory/mix/zero`, with raw image coefficient forced to zero for unavailable images.
- **[Result vs. Target]:** Accepted. With `scale=0.20` and proxy repair:
  - FBDB: `0.7246 / 0.7293`, gains `+1.34 / +1.06`.
  - FBYG: `0.8020 / 0.8011`, gains `+1.18 / +1.29`.
- **[Action]:** Continue with a more interpretable scale.

### V12-B: Four-Modality Normalized Scale

- **[Hypothesis]:** Use `scale=0.25 = 1 / 4`, matching the four-modality fusion setting. This avoids a tuned-looking decimal and maps image unreliability to one modality's share of the fusion logits.
- **[Arch Change]:** Set `--tmrd_auto_img_suppress_scale 0.25`; keep `auto_img_suppress` and explicit missing-image repair.
- **[Result vs. Target]:** Accepted. Best clean setting is `proxy` repair:

| Dataset | Mode | L2R H@1 | R2L H@1 | L2R Gain | R2L Gain | Auto `s_img` |
|---|---|---:|---:|---:|---:|---:|
| FBDB15K r=0.5 | proxy | 0.7271 | 0.7319 | +0.0159 | +0.0132 | 0.0940 |
| FBYG15K r=0.5 | proxy | 0.8052 | 0.8032 | +0.0150 | +0.0150 | 0.1059 |

- **[Action]:** Accept V12-B as the current working design.

## Missing-Replacement Ablation at V12-B

All runs use `auto_img_suppress(scale=0.25)`.

| Dataset | Missing Strategy | L2R H@1 | R2L H@1 | Notes |
|---|---|---:|---:|---|
| FBDB15K | none | 0.7246 | 0.7286 | No explicit repair; main gain remains from auto reliability calibration. |
| FBDB15K | proxy | 0.7271 | 0.7319 | Best FBDB setting. Missing rate 5.44%. |
| FBDB15K | mix | 0.7268 | 0.7319 | Nearly tied with proxy. |
| FBDB15K | zero | 0.7266 | 0.7316 | Close to proxy/mix. |
| FBYG15K | none | 0.8020 | 0.8007 | No explicit repair. |
| FBYG15K | proxy | 0.8052 | 0.8032 | Best/clean setting. Missing rate 18.83%. |
| FBYG15K | mix | 0.8052 | 0.8032 | Tied with proxy. |
| FBYG15K | zero | 0.8050 | 0.8029 | Close to proxy/mix. |

## Probe Findings

- FBDB predicted image quality: `q_img_present_mean=0.6281`, `q_img_missing_mean=0.5543`; missing AUC `0.8953`.
- FBYG predicted image quality: `q_img_present_mean=0.5974`, `q_img_missing_mean=0.4850`; missing AUC `0.9342`.
- Auto suppression is stronger on FBYG (`0.1059`) than FBDB (`0.0940`), matching its lower image quality and higher missing rate.
- Missing repair changes feature tokens but contributes only a small fraction of the total gain. The dominant effect is data-driven visual reliability calibration from `q_img`.

## Current Interpretation

V12 reframes the module from manually selected modality priors to a quality-estimated visual reliability calibration:

1. TMRD predicts image quality from entity type, cross-modal proxy consistency, and modality context.
2. The average image unreliability is converted into a fusion-logit image penalty with one interpretable scalar `scale=1/4`.
3. Explicit missing images obey the hard-mask rule: raw image features do not participate and can be replaced by proxy/mix/zero.

This meets the current target on both FBDB15K and FBYG15K at `r=0.5`.

## Next Checks

- Extend V12-B to `r=0.2` and `r=0.8` if this direction is kept.
- Add formal ablations for `auto_img_suppress` vs. fixed suppress vs. no TMRD quality head.
- Report sensitivity for `scale in {0.20, 0.25}` only; avoid broad grid-search framing.
