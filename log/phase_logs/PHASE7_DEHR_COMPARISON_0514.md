# Phase 7: DEHR 对比汇总（2026-05-14）

## 1. 目的

整理当前项目里三类结果的统一对比：

1. `baseline`：SGMEA 原始 checkpoint 的测试结果
2. `old DEHR`：之前依赖手工 / constrained bias 的 DEHR 版本
3. `current learned-DEHR`：当前去掉手工全局方向、仅保留训练得到类型条件校准的版本

本文档重点回答两个问题：

1. 相比 baseline，旧版 DEHR 和当前 learned-DEHR 分别提升了多少？
2. 当前版本的收益，是否仍主要来自手工先验？

---

## 2. 口径说明

### 2.1 主比较口径

- `FBDB15K` 的 baseline 采用：
  - `log/run_outputs/fbdb_rates_baseline_cdmr/fbdb_rates_cdmr_0511_104108/*.log`
- `FBYG15K` 的 baseline 采用：
  - `log/run_outputs/fbyg_specific_prior_full/fbyg_specific_prior_0512_111257/sgmea_baseline_lr5e4_*.log`
- `old constrained DEHR` 采用：
  - `log/run_outputs/dehr_typescale_sweep/dehr_constrained_0512_210405/*.log`
- `current learned-DEHR` 采用：
  - `log/run_outputs/dehr_learned_router/rates_0514_172031/*.log`

### 2.2 关于 r=0.5 的 baseline 微小差异

`r=0.5` 还存在一套 `baseline_onlytest` 复测日志：

- FBDB:
  - `log/run_outputs/dehr_stat_prior_decomposition/codex_dehr_stat_prior_decomp_0514b/baseline_onlytest_gpu0.log`
- FBYG:
  - `log/run_outputs/dehr_stat_prior_decomposition/codex_dehr_stat_prior_decomp_fbyg_0514/baseline_onlytest_gpu0.log`

它们与 sweep baseline 只有千分位差异：

- FBDB `0.71495` vs `0.71585`
- FBYG `0.78920` vs `0.79170`

原因是复测环境、only-test 调用路径和训练末次导出存在轻微数值漂移。

**本记录的主表仍以 learned-DEHR 实际加载的 baseline checkpoint 对应 sweep 分数为主**，这样和当前 learned 版最一致。

---

## 3. 主结果表：Avg H@1 / Avg MRR

### 3.1 FBDB15K

| Rate | Baseline H@1 | Baseline MRR | Old DEHR H@1 | Old DEHR MRR | Gain vs Base | Learned-DEHR H@1 | Learned-DEHR MRR | Gain vs Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.52895 | 0.6140 | 0.58995 | 0.6675 | +0.06100 | 0.58560 | 0.6640 | +0.05665 |
| 0.5 | 0.71495 | 0.7740 | 0.74220 | 0.7955 | +0.02725 | 0.74410 | 0.7970 | +0.02915 |
| 0.8 | 0.80720 | 0.8505 | 0.82470 | 0.8630 | +0.01750 | 0.83015 | 0.8675 | +0.02295 |

### 3.2 FBYG15K

| Rate | Baseline H@1 | Baseline MRR | Old DEHR H@1 | Old DEHR MRR | Gain vs Base | Learned-DEHR H@1 | Learned-DEHR MRR | Gain vs Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.60205 | 0.6860 | 0.65665 | 0.7300 | +0.05460 | 0.67370 | 0.7445 | +0.07165 |
| 0.5 | 0.78920 | 0.8400 | 0.81330 | 0.8585 | +0.02410 | 0.82395 | 0.8670 | +0.03475 |
| 0.8 | 0.87770 | 0.9100 | 0.89395 | 0.9220 | +0.01625 | 0.88930 | 0.9190 | +0.01160 |

---

## 4. 如果按“旧手工 manual DEHR”看 r=0.5

这组结果来自最直接的 prior decomposition 日志。

### FBDB15K r=0.5

- baseline_onlytest: `0.71585`
- old manual global DEHR: `0.74395`
- 提升：`+0.02810`
- current learned-DEHR: `0.74410`
- 相对 baseline sweep (`0.71495`) 提升：`+0.02915`
- 相对 old manual 的绝对差：`+0.00015`

### FBYG15K r=0.5

- baseline_onlytest: `0.79170`
- old manual global DEHR: `0.81580`
- 提升：`+0.02410`
- current learned-DEHR: `0.82395`
- 相对 baseline sweep (`0.78920`) 提升：`+0.03475`
- 相对 old manual 的绝对差：`+0.00815`

**结论：**

- `FBDB r=0.5`：当前 learned 版和旧手工版基本持平，略高
- `FBYG r=0.5`：当前 learned 版明显高于旧手工版

---

## 5. 当前 learned-DEHR 的关键日志证据

### 5.1 当前版本不再依赖手工 global direction

在 `log/run_outputs/dehr_learned_router/rates_0514_172031/*.log` 中，运行命令已经固定：

- `--dehr_direction_mode learned_type`
- `--dehr_direction_values=0,0,0,0`
- `--dehr_global_direction_weight 0.0`

并且训练日志里持续出现：

- `direction=+0.0000,+0.0000,+0.0000,+0.0000`
- `global_bias_abs_mean=0.0000`
- `type_bias_fraction=1.0000`

这说明当前收益不是由手工设定的 4 维 global bias 直接提供，而是来自：

1. 伪链接校准预训练
2. 类型条件的 learned bias

### 5.2 learned-DEHR 的伪链接规模

| Dataset | Rate | Train ILL | Pseudo Links | Pseudo Margin Mean |
|---|---:|---:|---:|---:|
| FBDB15K | 0.2 | 2569 | 5123 | 0.1259 |
| FBDB15K | 0.5 | 6423 | 4510 | 0.1788 |
| FBDB15K | 0.8 | 10276 | 2106 | 0.2196 |
| FBYG15K | 0.2 | 2239 | 5062 | 0.1264 |
| FBYG15K | 0.5 | 5599 | 4324 | 0.1685 |
| FBYG15K | 0.8 | 8959 | 2091 | 0.2052 |

观察：

- 低监督率 (`0.2`) 下，伪链接数量大于监督链接数，校准收益更明显
- 高监督率 (`0.8`) 下，伪链接更少，但 margin 更高，说明伪链接更“保守”

---

## 6. 当前实验结论

### 6.1 相比 baseline，current learned-DEHR 仍然稳定有效

- FBDB:
  - `r=0.2`: `+0.05665`
  - `r=0.5`: `+0.02915`
  - `r=0.8`: `+0.02295`
- FBYG:
  - `r=0.2`: `+0.07165`
  - `r=0.5`: `+0.03475`
  - `r=0.8`: `+0.01160`

### 6.2 相比旧 constrained DEHR

- `FBDB`
  - `r=0.2`：当前略低（`-0.00435`）
  - `r=0.5`：当前略高（`+0.00190`）
  - `r=0.8`：当前更高（`+0.00545`）
- `FBYG`
  - `r=0.2`：当前更高（`+0.01705`）
  - `r=0.5`：当前更高（`+0.01065`）
  - `r=0.8`：当前略低（`-0.00465`）

### 6.3 总体判断

当前 learned-DEHR 已经实现了一个比较重要的性质：

1. **去掉了手工 global bias 的直接贡献**
2. **大多数设置下仍保留甚至超过旧 DEHR 的收益**
3. **最关键的 r=0.5 档位上，FBDB 基本不掉，FBYG 明显更强**

因此当前版本的主论点可以写成：

> 模态可靠性校准的收益并不必须来自人工网格搜索得到的全局偏置；  
> 通过伪链接监督下的类型条件学习，同样可以学到稳定有效的模态重加权，并在多数设置下保持甚至超过旧 DEHR 的性能。

---

## 7. 关键日志入口

### 7.1 Baseline

- FBDB:
  - `log/run_outputs/fbdb_rates_baseline_cdmr/fbdb_rates_cdmr_0511_104108/baseline_r02_gpu1.log`
  - `log/run_outputs/fbdb_rates_baseline_cdmr/fbdb_rates_cdmr_0511_104108/baseline_r05_gpu2.log`
  - `log/run_outputs/fbdb_rates_baseline_cdmr/fbdb_rates_cdmr_0511_104108/baseline_r08_gpu3.log`
- FBYG:
  - `log/run_outputs/fbyg_specific_prior_full/fbyg_specific_prior_0512_111257/sgmea_baseline_lr5e4_r0.2_gpu3.log`
  - `log/run_outputs/fbyg_specific_prior_full/fbyg_specific_prior_0512_111257/sgmea_baseline_lr5e4_r0.5_gpu0.log`
  - `log/run_outputs/fbyg_specific_prior_full/fbyg_specific_prior_0512_111257/sgmea_baseline_lr5e4_r0.8_gpu1.log`

### 7.2 Old constrained DEHR

- `log/run_outputs/dehr_typescale_sweep/dehr_constrained_0512_210405/`

### 7.3 Old manual DEHR (r=0.5 decomposition)

- FBDB:
  - `log/run_outputs/dehr_stat_prior_decomposition/codex_dehr_stat_prior_decomp_0514b/manual_global_reference_gpu2.log`
- FBYG:
  - `log/run_outputs/dehr_stat_prior_decomposition/codex_dehr_stat_prior_decomp_fbyg_0514/manual_global_reference_gpu2.log`

### 7.4 Current learned-DEHR

- `log/run_outputs/dehr_learned_router/rates_0514_172031/FBDB15K_r02.log`
- `log/run_outputs/dehr_learned_router/rates_0514_172031/FBDB15K_r05.log`
- `log/run_outputs/dehr_learned_router/rates_0514_172031/FBDB15K_r08.log`
- `log/run_outputs/dehr_learned_router/rates_0514_172031/FBYG15K_r02.log`
- `log/run_outputs/dehr_learned_router/rates_0514_172031/FBYG15K_r05.log`
- `log/run_outputs/dehr_learned_router/rates_0514_172031/FBYG15K_r08.log`

