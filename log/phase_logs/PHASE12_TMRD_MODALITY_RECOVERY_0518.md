# PHASE12: TMRD Type-aware Modality Recovery and Denoising

## Directive

- Target: move beyond pure type-aware modality weighting by introducing type-aware modality recovery, denoising, and quality-aware fusion.
- Motivation: use existing LLM-derived entity types, frozen modality features, and same-type entity structure to address missing/noisy modalities, especially image noise, without relying on manual oracle weights.
- Training protocol: decoupled self-supervised pretraining first, then conservative ranking-time calibration or light fine-tuning. Avoid multi-loss gradient conflict.
- Probes required in each iteration:
  - ranking metrics: Hits@1/10/50, MR, MRR for l2r/r2l;
  - quality distribution: q mean/min/max/std;
  - fusion impact: quality bias magnitude and final modality weight delta;
  - corruption probe: clean-vs-corrupt pair accuracy and quality gap;
  - missing-image probe: present-vs-missing image quality gap, missing detection accuracy, and AUC;
  - memory probe: clustered memory diversity.

## V1: Memory-anchored TMRD Self-supervised Quality Head

- **[Hypothesis]:** A type-conditioned clustered memory plus cross-modal proxy can learn whether a modality token is consistent with same-type semantics; this should support denoising/calibration without manual modality priors.
- **[Arch Change]:** Freeze the alignment backbone and train only TMRD. Use type embedding, other-modality context, semantic proxy, clustered memory reference, and a lightweight quality head. Use same-type modality corruption, pairwise clean/corrupt ranking, clean anchor, and weak InfoNCE. Keep `clean_mix_scale=0` and `memory_mix=0` to avoid destructive feature replacement.
- **[Probe Result]:** Saved Phase 1 checkpoints show useful but imperfect quality discrimination. FBDB pair accuracy is about 0.74 with quality gap about 0.0115; FBYG pair accuracy is about 0.64 with quality gap about 0.0047. Memory diversity is high on both datasets, about 0.85-0.86, so the clustered memory is not collapsing.
- **[Ranking Result]:** Weak `gamma * log(q)` calibration is almost neutral. FBDB r=0.5 stays around l2r/r2l Hits@1 = 0.7112/0.7188. FBYG r=0.5 reaches about 0.7907/0.7886 at `gamma=0.03, clip=0.08`, a tiny positive movement over the baseline/checkpoint range.
- **[Action]:** Pivot. The quality head learns a signal, but the current log-quality injection is too weak to affect fusion.

## V1 Probe Check: Ranking-time Instrumentation

- **[Hypothesis]:** The current ranking logs are insufficient; we need to know whether failures come from bad quality prediction or from weak fusion injection.
- **[Arch Change]:** Add test-time `RouterProbe` and `TMRDSelfsupProbe` logging. This does not change forward behavior or ranking metrics.
- **[Probe Result]:** With `gamma=0.03, clip=0.08`, FBDB has `tmrd_corrupt_pair_acc_probe=0.7539`, `quality_gap=0.0144`, but `type_modality_weight_delta_abs_mean=0.0003`. FBYG has `pair_acc=0.6650`, `quality_gap=0.0077`, and the same tiny weight delta about 0.0003.
- **[Ranking Result]:** Same as V1 probe check: FBDB l2r/r2l Hits@1 = 0.7112/0.7188; FBYG = 0.7907/0.7886.
- **[Action]:** Accept the probe. Next iteration should replace `gamma * log(q)` with a relative reliability calibration that uses within-entity modality contrast.

## V2: Entity-local Relative Reliability Calibration

- **[Hypothesis]:** Absolute TMRD quality scores are dataset-shifted, but within-entity relative modality quality is more reliable. A centered log-odds calibration should amplify clean-vs-noisy modality contrast without becoming a global manual prior.
- **[Arch Change]:** Add `tmrd_quality_bias_mode=centered_logit`. It computes `bias = gamma * (logit(q_m) - mean_m logit(q_m))`, then clips symmetrically. This directly calibrates fusion logits using each entity's own modality quality contrast.
- **[Probe Result]:** Fusion impact increased. The old `logq` mode had `type_modality_weight_delta_abs_mean≈0.0003`. Centered logit gives about 0.0010 at `gamma=0.05`, 0.0017-0.0019 at `gamma=0.10`, and 0.0035-0.0038 at `gamma=0.20`.
- **[Ranking Result]:** FBYG improves slightly and consistently: at `gamma=0.10`, l2r/r2l Hits@1 = 0.7914/0.7896; at `gamma=0.20`, 0.7916/0.7895. FBDB is mixed: r2l improves slightly, l2r drops slightly, with average roughly unchanged.
- **[Action]:** Keep as a better inference calibration than `logq`, but do not claim a major gain. It proves the quality signal can move fusion, while exposing that quality learning is still not sufficiently ranking-aligned.

## V3: Phase-2 EA-supervised TMRD-only Calibration

- **[Hypothesis]:** If self-supervised corruption detection is not ranking-aligned enough, a short supervised calibration stage using train ILL may make the TMRD quality scores more useful while the backbone remains frozen.
- **[Arch Change]:** Add `tmrd_only_train_use_ea`. In `tmrd_only_train`, only TMRD parameters are trainable; with this flag the EA alignment loss is retained, plus a small self-supervised term (`tmrd_pretrain_loss_weight=0.05`) as a denoising regularizer.
- **[Probe Result]:** Weight delta grows over training, e.g. FBDB reaches `type_modality_weight_delta_abs_mean≈0.0039`; FBYG reaches about 0.0022. However, FBYG clean/corrupt gap weakens to about 0.005-0.006, showing supervised pressure is not aligned with the denoising probe.
- **[Ranking Result]:** Negative or neutral. FBDB final l2r/r2l Hits@1 = 0.7106/0.7176, below or around the inference-only centered calibration. FBYG final = 0.7896/0.7871, worse than inference-only V2.
- **[Action]:** Reject for now. Do not extend training epochs. The issue is not insufficient training time; the supervised objective pushes quality scores in a direction that does not improve ranking robustness.

## V4 Draft: Real Missing-image Availability Calibration Probe

- **[Hypothesis]:** The earlier same-type corruption task gives TMRD a generic consistency signal, but it does not explicitly teach the model to identify the dataset's real missing-image artifact. Since missing images are filled by random Gaussian vectors in the loader, a weak availability target may teach the quality head to separate real image tokens from synthetic placeholders without hard-coding a fusion rule.
- **[Arch Change]:** Add `tmrd_missing_loss_weight` and a missing-image probe path. During TMRD-only self-supervised pretraining, the image quality score `q_img` receives a weak BCE target from real `image_available`. Inference still uses the learned quality head and centered-logit calibration; the availability flag is not directly injected into the final fusion bias.
- **[Probe Result]:** Pending. Required probes: `tmrd_missing_acc_probe`, `tmrd_missing_auc_probe`, `tmrd_missing_q_gap_probe`, plus the existing corruption and memory probes.
- **[Ranking Result]:** Pending on FBDB15K/FBYG15K r=0.5.
- **[Action]:** Start with conservative weights (`tmrd_missing_loss_weight=0.05/0.10`) and reject if missing probes improve only by collapsing all image quality or if ranking drops.

## V4 Result: Missing Probe Works, Direct Ranking Injection Fails

- **[Hypothesis]:** If real missing-image supervision improves `q_img`, centered-logit quality calibration or feature repair should improve final ranking.
- **[Arch Change]:** Train TMRD-only checkpoints with `tmrd_missing_loss_weight=0.05/0.10`, then evaluate either centered-logit fusion calibration or small feature-level cleaning (`clean_mix_scale=0.05`).
- **[Probe Result]:** The missing-image probe clearly improves. FBYG with `missing_loss_weight=0.10` reaches `tmrd_img_q_present_missing_gap=0.1021` during unbiased test and `tmrd_missing_auc_probe≈0.93`. FBDB improves more mildly, e.g. `gap≈0.0404` for weight `0.10`. Corruption pair accuracy remains usable around `0.69-0.77` depending on dataset.
- **[Ranking Result]:** Direct use hurts. Centered-logit calibration drops FBDB to about `0.6984/0.7054` Hits@1 at weight `0.10`, and FBYG to about `0.7841/0.7825`. Feature-level cleaning with `clean_mix_scale=0.05` is much worse (`FBDB≈0.63`, `FBYG≈0.66` Hits@1), showing that the current proxy is not geometrically safe as a replacement representation.
- **[Action]:** Reject direct quality-to-weight and all-modal feature replacement for V4. Pivot to constrained repair: only image modality, only low-quality tokens, no global fusion boost. Treat proxy/memory as a diagnostic/recovery reference, not a universal substitute.

## V5 Result: Conservative Low-quality Image Cleaning

- **[Hypothesis]:** Since all-modal feature replacement damages the representation geometry, restrict cleaning to the noisiest surface: image tokens with low predicted quality, and use a very small replacement gate.
- **[Arch Change]:** Keep the V4 missing-supervised TMRD checkpoint. Set `tmrd_clean_modal_idx=0`, use `clean_mix_scale=0.01/0.02`, and trigger cleaning only when `q_img` is below `0.52/0.55`. Disable fusion quality bias so the test isolates feature-level repair.
- **[Probe Result]:** Missing detection remains strong on FBYG (`tmrd_missing_auc_probe` around 0.93, present-missing quality gap around 0.10). Feature deltas are now tiny and localized instead of globally replacing all modalities.
- **[Ranking Result]:** The method no longer catastrophically hurts ranking, but the gain is too small. On FBYG r=0.5, image-only cleaning gives roughly l2r/r2l Hits@1 `0.7900/0.7882` to `0.7905/0.7889`, close to the unbiased checkpoint and below the older centered-logit V2 result (`0.7914/0.7896`).
- **[Action]:** Reject as the main injection mechanism. It is a safe diagnostic repair, not a strong ranking module.

## V6 Draft: One-way Low-quality Image Suppression

- **[Hypothesis]:** The failure of centered-logit comes from treating quality as a symmetric reward/suppress signal. In this dataset, image availability only proves that the image is present, not that it is alignment-reliable. A safer rule is to never reward image quality and only suppress image fusion when the learned TMRD quality is low.
- **[Arch Change]:** Add `tmrd_quality_bias_mode=image_suppress`: for image modality only, compute `bias_img = -gamma * sigmoid((threshold - q_img) / temp) * relu(threshold - q_img)`. Other modality logits and all features are untouched. New probes record suppress bias, active rate, gate mean, and quality shortfall.
- **[Probe Result]:** Pending. Required checks: missing AUC/gap must remain high, image suppress active rate must be nonzero but not global, and `type_modality_weight_delta_abs_mean` should be small enough to avoid becoming another manual prior.
- **[Ranking Result]:** Pending on FBDB15K/FBYG15K r=0.5 with `gamma in {0.05, 0.10, 0.20}` and thresholds `0.52/0.55`.
- **[Action]:** Run inference-only tests first. Accept only if ranking is at least neutral and the probe shows targeted suppression of low-quality images.

## V6 Result: Shortfall Suppression Is Too Weak

- **[Hypothesis]:** Only suppress image logits when `q_img` is below a threshold, without rewarding high-quality images.
- **[Arch Change]:** Use `image_suppress`, where `bias_img = -gamma * sigmoid((threshold - q_img) / temp) * relu(threshold - q_img)`.
- **[Probe Result]:** The missing-image quality signal is strong on FBYG (`present-missing gap=0.1124`), and suppression is targeted (`active_rate=0.1447` at threshold `0.52`, `0.2030` at threshold `0.55`). But the actual fusion movement is too small: `type_modality_weight_delta_abs_mean` stays around `0.0001-0.0010`.
- **[Ranking Result]:** FBYG r=0.5 remains nearly unchanged: best tested setting reaches only about l2r/r2l Hits@1 `0.7907/0.7884`.
- **[Action]:** Pivot. The trigger is interpretable, but multiplying by the tiny quality shortfall makes the calibration too weak.

## V7 Result: Gate Suppression Works but Becomes a Global Image Prior

- **[Hypothesis]:** Remove the shortfall multiplier so that low-quality evidence can exert enough force on image fusion.
- **[Arch Change]:** Add `image_suppress_gate`, where `bias_img = -gamma * sigmoid((threshold - q_img) / temp)`, clipped to a small negative range.
- **[Probe Result]:** Fusion movement becomes meaningful (`type_modality_weight_delta_abs_mean=0.0030-0.0078`), and FBYG ranking improves. However, because every image receives a negative gate after clipping, `tmrd_quality_suppress_active_rate=1.0000`. This makes the method look like a global image penalty rather than learned low-quality suppression.
- **[Ranking Result]:** FBYG r=0.5 improves up to l2r/r2l Hits@1 `0.8020/0.8005` at clip `0.08`, clearly better than previous TMRD injections.
- **[Action]:** Reject as a main paper mechanism despite strong performance. It is useful as an upper-bound diagnostic, but its probe exposes a reviewer-facing weakness: the gain can be attributed to uniformly down-weighting image.

## V8 Result: Targeted Low-quality Image Suppression

- **[Hypothesis]:** Keep the stronger gate, but make it fire only for images below the learned quality threshold. This should preserve the V7 benefit while restoring white-box interpretability.
- **[Arch Change]:** Add `image_suppress_targeted`: `bias_img = -gamma * sigmoid((threshold - q_img) / temp) * I(q_img < threshold)`. Features are untouched; only the image fusion logit is reduced for low-quality tails.
- **[Probe Result]:** The mechanism is now targeted. On FBYG, threshold `0.58` triggers `active_rate=0.3410`, with full missing-image AUC `0.9342`; on FBDB, threshold `0.60` triggers `active_rate=0.2703`, with full missing-image AUC `0.8953`. Weight deltas are moderate (`0.0022-0.0027`) and not a global prior.
- **[Ranking Result]:** Compared with the same TMRD checkpoint and no quality bias, FBDB improves from `0.7112/0.7187` to `0.7146/0.7222` Hits@1. FBYG improves from `0.7902/0.7882` to `0.7920/0.7891` Hits@1. Gains are smaller than V7 but defensible.
- **[Action]:** Accept as the current interpretable candidate. It establishes a data-driven quality signal, targeted intervention, and positive ranking effect without hard-coded modality direction.

## V9 Result: Quantile-targeted Low-quality Tail Suppression

- **[Hypothesis]:** Fixed quality thresholds (`0.58/0.60`) may look dataset-tuned. A quantile rule can suppress the learned low-quality tail with one shared hyperparameter and clearer interpretation.
- **[Arch Change]:** Add `tmrd_quality_suppress_quantile`; when enabled, the threshold is set to the quality quantile of `q_img`, then `image_suppress_targeted` suppresses only that bottom tail. Tested bottom `20%` and `30%`.
- **[Probe Result]:** The rule gives exact active rates (`0.20/0.30`) and preserves the strong missing-image probes: FBDB full AUC `0.8953`, FBYG full AUC `0.9342`. The bottom-30% setting yields moderate weight deltas (`0.0025` on FBDB, `0.0023` on FBYG).
- **[Ranking Result]:** Bottom `30%` is the best unified setting so far: FBDB r=0.5 reaches l2r/r2l Hits@1 `0.7151/0.7222`; FBYG reaches `0.7916/0.7893`. It is close to the best fixed-threshold V8 while avoiding dataset-specific thresholds.
- **[Action]:** Prefer V9 for paper-facing experiments. Keep V8 fixed-threshold as an analysis variant and V7 gate as an upper-bound showing why targeted probes are necessary.
