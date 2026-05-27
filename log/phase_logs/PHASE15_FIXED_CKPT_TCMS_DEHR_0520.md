# PHASE15 Fixed-Checkpoint TCMS + DEHR Reproducibility Record

Date: 2026-05-20

This file freezes the current reliable conclusion and configuration for the two-module evaluation:

- TCMS: pre-fusion visual sanitization / residual calibration module.
- DEHR: learned type-aware four-modality routing / calibration module.
- Protocol: fixed seed, fixed checkpoints, fixed evaluation flags, l2r/r2l averaged metrics.

The canonical run is:

- Script: `run_fixed_ckpt_tcms_dehr_effect_8gpu.sh`
- Run tag: `phase25_fixed_ckpt_tcms_dehr_0520_seed42_ckptfixed_v2`
- Log dir: `log/run_outputs/phase25_fixed_ckpt_tcms_dehr_0520_seed42_ckptfixed_v2`
- JSON snapshot: `log/phase_logs/phase15_fixed_ckpt_tcms_dehr_0520.json`
- Status: all 24 runs completed successfully; no Traceback / CUDA OOM / Killed errors.

Important note: the first script attempt failed because shell quoting escaped `300,300,300` as `300\\,300\\,300`. The fixed v2 run uses normal comma-separated hidden units and is the canonical result.

## Fixed Reproducibility Protocol

Seed:

- `--random_seed 42`
- `config.py` also defaults to `42`; the explicit argument is kept to lock the experiment protocol.

Common evaluation settings:

- `--data_split norm`
- `--hidden_units 300,300,300`
- `--batch_size 2048`
- `--csls --csls_k 3`
- `--workers 0`
- `--scheduler fixed`
- `--attr_dim 300 --img_dim 300 --name_dim 300 --char_dim 300 --hidden_size 300`
- `--tau 0.1`
- `--structure_encoder gat`
- `--num_attention_heads 1 --num_hidden_layers 1`
- `--use_surface 0 --use_intermediate 0`
- `--disable_sgmea_guidance`
- `--external_anchor_type_jsonl /gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/<DATASET>/norm_anchor_type_fixed.jsonl`

Variants:

- `baseline`: fixed baseline checkpoint, `--only_test 1`.
- `tcms`: fixed TCMS checkpoint, `--only_test 1`, TCMS enabled and beta forced active.
- `dehr`: fixed baseline checkpoint + DEHR calibration only.
- `tcms_dehr`: fixed TCMS checkpoint + DEHR calibration only, TCMS beta forced active.

## Fixed Checkpoints

Baseline checkpoints:

| Dataset / Rate | Checkpoint |
|:---|:---|
| FBDB r=0.2 | `SGMEA_FBDB15K_0.2_baseline_warm_fbdb_rates_cdmr_0511_104108_r02_` |
| FBDB r=0.5 | `SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05_` |
| FBDB r=0.8 | `SGMEA_FBDB15K_0.8_baseline_warm_fbdb_rates_cdmr_0511_104108_r08_` |
| FBYG r=0.2 | `SGMEA_FBYG15K_0.2_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r02_` |
| FBYG r=0.5 | `SGMEA_FBYG15K_0.5_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r05_` |
| FBYG r=0.8 | `SGMEA_FBYG15K_0.8_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r08_` |

TCMS checkpoints:

| Dataset / Rate | Checkpoint |
|:---|:---|
| FBDB r=0.2 | `SGMEA_FBDB15K_0.2_fbdb_r02_tcms_keep_ckpt_` |
| FBDB r=0.5 | `SGMEA_FBDB15K_0.5_fbdb_r05_keep_repro_seed42_` |
| FBDB r=0.8 | `SGMEA_FBDB15K_0.8_fbdb_r08_tcms_keep_ckpt_` |
| FBYG r=0.2 | `SGMEA_FBYG15K_0.2_fbyg_r02_tcms_keep_ckpt_` |
| FBYG r=0.5 | `SGMEA_FBYG15K_0.5_fbyg_r05_tcms_keep_ckpt_` |
| FBYG r=0.8 | `SGMEA_FBYG15K_0.8_fbyg_r08_tcms_keep_ckpt_` |

FBDB r=0.5 must use `SGMEA_FBDB15K_0.5_fbdb_r05_keep_repro_seed42_`; the older `fbdb_r05_tcms_keep_ckpt` is not the fixed high-score checkpoint.

## Fixed TCMS Configuration

TCMS arguments:

```bash
--use_tcms
--tcms_prefusion
--tcms_zero_init_residual
--tcms_noise_same_type_ratio 0.7
--tcms_disable_gate_match_features
--tcms_beta 0.25
--tcms_beta_warmup_start -1
--tcms_beta_warmup_end 150
--tcms_pretrain_loss_weight 0.002
--tcms_selfsup_start 100
--tcms_selfsup_warmup_end 170
--tcms_sparse_weight 0.03
--tcms_gate_init -3.0
--tcms_missing_mode keep
```

Critical reproducibility guard:

- TCMS checkpoints were trained with beta warmup.
- In pure `only_test`, `current_epoch=0`; if `--tcms_beta_warmup_start 70` is kept, beta becomes 0 and the sanitizer is effectively disabled.
- Therefore, canonical TCMS evaluation uses `--tcms_beta_warmup_start -1` and `--tcms_beta 0.25`.

## Fixed DEHR Configuration

DEHR arguments:

```bash
--epoch 0
--eval_epoch 1
--save_model 0
--type_modality_bias_only_train
--use_dehr_router
--dehr_direction_mode learned_type
--dehr_direction_values=0,0,0,0
--dehr_global_direction_weight 0.0
--dehr_type_direction_residual_scale 0.0
--dehr_type_direction_residual_max 0.0
--dehr_rho_init_values 0.7,0.7,0.7,0.7,0.7,0.7
--dehr_rho_max 1.0
--dehr_bias_clamp 0.35
--dehr_learned_bias_scale 0.35
--dehr_contextual_mix 0.15
--dehr_contextual_source residual
--dehr_learned_bias_l2_weight 1e-4
--dehr_stat_feature_mask none
--dehr_calib_pretrain_epochs 25
--dehr_calib_pretrain_lr 5e-3
--dehr_calib_pretrain_tau 0.05
--dehr_calib_pretrain_batch_size 256
--dehr_calib_pretrain_val_ratio 0.2
--dehr_calib_pretrain_patience 8
--dehr_calib_bias_l2 1e-4
--dehr_calib_pair_source train_pseudo
--dehr_calib_selection final
--dehr_calib_pseudo_margin 0.04
--dehr_calib_pseudo_csls_k 1
--dehr_calib_pseudo_source dropout_consensus
--dehr_calib_consensus_min_votes 2
--lr 3e-4
```

## Canonical Results

All values are l2r/r2l averages. Gains are percentage points over the fixed baseline checkpoint.

| Dataset / Rate | Baseline Hits@1 | Baseline MRR | TCMS Hits@1 | TCMS Gain | TCMS MRR | TCMS MRR Gain | DEHR Hits@1 | DEHR Gain | DEHR MRR | DEHR MRR Gain | TCMS+DEHR Hits@1 | TCMS+DEHR Gain | TCMS+DEHR MRR | TCMS+DEHR MRR Gain |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FBDB r=0.2 | 0.52895 | 0.6140 | 0.52140 | -0.755 | 0.6095 | -0.45 | 0.59505 | +6.610 | 0.6715 | +5.75 | 0.59180 | +6.285 | 0.6700 | +5.60 |
| FBDB r=0.5 | 0.71495 | 0.7740 | 0.73315 | +1.820 | 0.7910 | +1.70 | 0.74460 | +2.965 | 0.7965 | +2.25 | 0.75615 | +4.120 | 0.8080 | +3.40 |
| FBDB r=0.8 | 0.80720 | 0.8505 | 0.83015 | +2.295 | 0.8700 | +1.95 | 0.82525 | +1.805 | 0.8635 | +1.30 | 0.83385 | +2.665 | 0.8720 | +2.15 |
| FBYG r=0.2 | 0.60205 | 0.6860 | 0.63235 | +3.030 | 0.7115 | +2.55 | 0.66715 | +6.510 | 0.7385 | +5.25 | 0.69325 | +9.120 | 0.7600 | +7.40 |
| FBYG r=0.5 | 0.78920 | 0.8400 | 0.81150 | +2.230 | 0.8560 | +1.60 | 0.81945 | +3.025 | 0.8635 | +2.35 | 0.83030 | +4.110 | 0.8710 | +3.10 |
| FBYG r=0.8 | 0.87770 | 0.9100 | 0.88010 | +0.240 | 0.9115 | +0.15 | 0.88505 | +0.735 | 0.9160 | +0.60 | 0.88415 | +0.645 | 0.9160 | +0.60 |
| Average | - | - | - | +1.477 | - | +1.25 | - | +3.608 | - | +2.92 | - | +4.491 | - | +3.71 |

## Fixed Conclusions

1. TCMS alone is reproducibly useful overall, with average Hits@1 gain +1.477 pp and average MRR gain +1.25 pp, but it is not uniformly positive: FBDB r=0.2 drops by -0.755 Hits@1 pp.

2. DEHR alone is stronger and more stable than TCMS in this fixed protocol, with average Hits@1 gain +3.608 pp and average MRR gain +2.92 pp.

3. TCMS and DEHR are complementary in the main middle/low-supervision settings. TCMS+DEHR reaches average Hits@1 gain +4.491 pp and average MRR gain +3.71 pp.

4. The most important positive cases for the combined module are:
   - FBYG r=0.2: +9.120 Hits@1 pp, +7.40 MRR pp.
   - FBDB r=0.5: +4.120 Hits@1 pp, +3.40 MRR pp.
   - FBYG r=0.5: +4.110 Hits@1 pp, +3.10 MRR pp.

5. High-supervision FBYG r=0.8 is close to saturation. Both TCMS and TCMS+DEHR only provide small gains there.

6. For future discussions and paper tables, this file is the fixed checkpoint/seed conclusion unless a later phase explicitly supersedes it with another fixed protocol.

