# PHASE9 TMHG Redesign 0514

## Goal

Redesign DEHR/TMHG-style type-modality routing so the gain is not explainable as a hand-crafted or grid-searched prior directly injected into final fusion. The learned module should:

1. train the type x modality weights from data,
2. keep moderate architectural complexity,
3. preserve most of the prior-based gain,
4. remain explainable.

## What Was Tried

### 1. `distill` only, no direct prior injection

- Change:
  - `--tmhg_prior_mode distill`
  - prior not added into final logits
  - learned logits come from `strength * (dirichlet_score + residual) + global_bias + type_bias`
- Result:
  - failed on both FBDB/FBYG
  - router stayed near uniform weights
  - best checkpoint remained very close to epoch 0
- Diagnosis:
  - training signal too weak
  - main objective alone cannot push a frozen-backbone router far from baseline

### 2. `distill + prior as feature`

- Change:
  - `--tmhg_use_prior_logit_feature`
  - prior logits used as token feature, but still not directly injected
- Result:
  - still near-uniform
  - no meaningful gain over baseline
- Diagnosis:
  - using prior as input evidence is not enough if there is no strong supervisory signal on router outputs

### 3. `logit-MSE` supervision on TMHG outputs

- Added args:
  - `--tmhg_prior_logit_mse_weight`
- Added loss in `TypeModalityHyperGate.regularization_loss()`:
  - MSE between predicted type-level centered logits and stored statistical-prior logits
- First attempt:
  - bug: loss used detached `last_q_all`, so MSE term had no gradient effect
  - this run should be treated as invalid
- Fix:
  - switched to live tensor `_last_q_all_live`

### 4. `logit-MSE + logit bias`

- Added arg:
  - `--tmhg_bias_mode {logprob,logit}`
- Motivation:
  - old TMHG converted learned logits into `log q - log uniform`, which compressed amplitude
  - this weakened the effect of a successfully learned type-level score map
- New `logit` mode:
  - final fusion bias uses centered raw logits directly

## Best Current Results

### FBDB15K, rate=0.5

Run:
- `log/run_outputs/tmhg_router/logitbias_0514/FBDB15K_r05_logitbias_v1.log`

Settings:
- `tmhg_prior_mode=distill`
- `tmhg_use_prior_logit_feature`
- `tmhg_prior_logit_mse_weight=10.0`
- `tmhg_bias_mode=logit`
- `tmhg_disable_type_bias`
- `tmhg_residual_scale=0.6`

Final best:
- Avg H@1 = `0.72980`
- l2r H@1 = `0.7283`
- r2l H@1 = `0.7313`

Comparison:
- baseline avg H@1 about `0.71585`
- gain over baseline: about `+0.01395`
- old hand prior best: `0.73775`
- retained ratio vs hand prior gain:
  - hand gain: `0.73775 - 0.71585 = 0.02190`
  - current gain: `0.72980 - 0.71585 = 0.01395`
  - ratio: about `63.7%`

Observation:
- `prior_cos` rose to about `0.95`
- type weights clearly moved away from uniform
- graph weight increased while img/attr/rel were rebalanced by type
- this is the first variant where “learned from target logits” and “actual H@1 gain” appear together

### FBYG15K, rate=0.5

Run:
- `log/run_outputs/tmhg_router/logitbias_0514/FBYG15K_r05_logitbias_v1.log`

Final best:
- Avg H@1 = `0.79355`

Comparison:
- baseline avg H@1 about `0.79170`
- gain over baseline: about `+0.00185`
- old hand prior best around `0.81215`
- retained ratio vs hand prior gain is still very low

Observation:
- `prior_cos` improved to roughly `0.79`
- learned distribution moved away from uniform, but downstream gain remained limited
- FBYG is still the bottleneck

## Interpretation

The redesign is now materially different from the earlier “direct prior injection” story:

1. Statistical prior is no longer directly added to fusion logits.
2. It is used as:
   - input feature,
   - soft KL teacher,
   - explicit logit reconstruction target.
3. TMHG must map train-stat features + prior descriptors through Transformer/MLP to produce usable type-modal logits.
4. Using raw logits as fusion bias preserves the learned structure much better than log-prob compression.

## Remaining Problems

1. FBDB improved clearly, but still below the 80% gain-retention target.
2. FBYG remains much harder; current learned router only recovers a small fraction of prior-based gain.
3. `type_modality_bias_only_train` means the frozen backbone gives weak task pressure; router supervision is still the main driver.
4. Current teacher target is still the statistical prior, so the story is better than direct injection, but not yet fully independent.

## Immediate Next Direction

Most promising next step:

1. keep current `logit-MSE + logit bias` formulation,
2. replace or augment prior target with a train-derived target from harder calibration:
   - type-level hard-negative margin target,
   - type-level unimodal ranking target,
   - or calibration pretrain with more difficult candidate pools,
3. then re-run FBDB/FBYG 0.5,
4. if stable, expand to 0.2/0.8.

## Files Changed In This Phase

- `config.py`
- `model/SGMEA_tools.py`

## New Key Args

- `--tmhg_prior_logit_mse_weight`
- `--tmhg_bias_mode`

