# Phase 8: DEHR 消融实验汇总（2026-05-14）

## 1. 目的

本轮消融直接对应两个潜在审稿质疑：

1. `Confirmation Bias`：当前 learned-DEHR 的收益是否主要来自 pseudo links？
2. `Feature Attribution`：DEHR 的 7 类统计特征和 projected token 到底谁在起作用？

所有实验都固定为当前 `learned-DEHR` 官方设置，只改动单一消融变量。

---

## 2. 实验设置

主对比基线：

- `FBDB15K r=0.5`
  - `log/run_outputs/dehr_learned_router/rates_0514_172031/FBDB15K_r05.log`
- `FBYG15K r=0.5`
  - `log/run_outputs/dehr_learned_router/rates_0514_172031/FBYG15K_r05.log`

消融日志目录：

- `log/run_outputs/dehr_learned_router/ablation_0514_175247/`

统一口径：

- `Avg H@1 = (l2r H@1 + r2l H@1) / 2`
- `Avg MRR = (l2r MRR + r2l MRR) / 2`

注意：

- `loaded_final val_avg` 只在**同一 pair source** 内部有可比性
- `train_only` 的 `val_avg≈1.0` 不代表泛化更强，反而常常意味着对该校准源严重过拟合

---

## 3. 消融一：Pseudo Link 来源

### 3.1 FBDB15K r=0.5

| Setting | Pair Source | Pseudo Links | Val Avg | Avg H@1 | Avg MRR | Delta vs Full |
|---|---|---:|---:|---:|---:|---:|
| Full learned-DEHR | `train_pseudo` | 4510 | 0.9053 | 0.7441 | 0.7970 | 0.0000 |
| Train only | `train` | 0 | 0.9996 | 0.6666 | 0.7365 | -0.0776 |
| Pseudo only | `pseudo` | 4510 | 0.7483 | 0.7441 | 0.7970 | -0.0001 |

### 3.2 FBYG15K r=0.5

| Setting | Pair Source | Pseudo Links | Val Avg | Avg H@1 | Avg MRR | Delta vs Full |
|---|---|---:|---:|---:|---:|---:|
| Full learned-DEHR | `train_pseudo` | 4324 | 0.9202 | 0.8240 | 0.8670 | 0.0000 |
| Train only | `train` | 0 | 1.0000 | 0.6801 | 0.7530 | -0.1439 |
| Pseudo only | `pseudo` | 4324 | 0.8058 | 0.8257 | 0.8675 | +0.0017 |

### 3.3 结论

1. `train_only` 在两套数据上都明显崩掉
2. `pseudo_only` 与 `train_pseudo` 几乎等价，在 `FBYG` 上还略强
3. 因此，当前 learned-DEHR 的主要收益**不是**来自 gold train ILL 校准，而是来自 pseudo-link 校准

这条结论是双刃剑：

- 好处：说明当前版本已经不是“手工 global bias 直接提分”
- 风险：也说明 `confirmation bias` 质疑**不能靠这轮消融直接化解**

更准确的表述应该是：

> 当前 learned-DEHR 的增益主要由 pseudo-link 驱动，而不是由少量监督 ILL 或手工 bias 驱动。

---

## 4. 消融二：特征归因

三组设置：

- `pairwise_only`
  - `--dehr_stat_feature_mask pairwise`
- `stats_no_pairwise`
  - `--dehr_stat_feature_mask weight,entropy,max_weight,top1_gap,direction,anchor`
- `drop_projected_token`
  - `--dehr_drop_projected_token`

### 4.1 FBDB15K r=0.5

| Setting | Val Avg | Avg H@1 | Avg MRR | Delta vs Full |
|---|---:|---:|---:|---:|
| Full learned-DEHR | 0.9053 | 0.7441 | 0.7970 | 0.0000 |
| Pairwise only | 0.9015 | 0.7438 | 0.7965 | -0.0003 |
| Stats no pairwise | 0.9053 | 0.7441 | 0.7970 | 0.0000 |
| Drop projected token | 0.8992 | 0.7434 | 0.7965 | -0.0007 |

### 4.2 FBYG15K r=0.5

| Setting | Val Avg | Avg H@1 | Avg MRR | Delta vs Full |
|---|---:|---:|---:|---:|
| Full learned-DEHR | 0.9202 | 0.8240 | 0.8670 | 0.0000 |
| Pairwise only | 0.9193 | 0.8239 | 0.8665 | -0.0001 |
| Stats no pairwise | 0.9163 | 0.8217 | 0.8655 | -0.0023 |
| Drop projected token | 0.9174 | 0.8213 | 0.8650 | -0.0027 |

### 4.3 结论

1. `projected token` 不是主要增益来源
   - 去掉后两套数据都只小幅下降
2. `pairwise` 不是唯一决定因素
   - `FBDB` 上去掉 pairwise 后基本不掉
   - `FBYG` 上去掉 pairwise 有小幅下降，但仍保留绝大部分收益
3. 统计特征之间存在很强冗余
   - `pairwise_only` 接近 full
   - `stats_no_pairwise` 也接近 full

这说明：

- 当前 DEHR 并不是单纯靠 projected embedding token 的黑盒路由
- 但它确实主要依赖**统计特征 token**，而且这些统计特征内部存在较强可替代性

---

## 5. 对审稿质疑的当前回答边界

### 5.1 质疑一：Confirmation Bias

这轮消融**不能**证明当前 pseudo-link 完全无偏。

它只能证明：

1. learned-DEHR 的收益不来自手工 bias
2. 也不来自少量 gold train ILL 校准
3. 当前收益主要来自 pseudo-link 校准

因此，如果要正面化解这个质疑，后续需要继续补：

- 更严格的 pseudo-link 去偏策略
- 或 pseudo-link 来源替换 / 过滤实验
- 或 multi-view agreement / type-consistent pseudo-link 实验

### 5.2 质疑二：Feature Attribution

这轮消融已经能支撑一个比较清楚的回答：

1. 不是 projected token 在主导
2. 不是某一个单特征完全垄断
3. 统计 token 的不同子集都能支撑大部分收益，说明学到的是**鲁棒统计模式**，而不是单一数值 hack

但也要诚实承认：

- 当前 7 特征输入存在冗余
- 模块还可以进一步瘦身，做更干净的 minimal variant

---

## 6. 当前最重要的结论

如果只看本轮数据，最稳妥的总结是：

1. `learned-DEHR` 的收益确实不是手工 global bias 直接带来的
2. 但当前版本的核心收益主要由 pseudo-link calibration 驱动
3. projected token 贡献有限，统计 token 才是主信号
4. `pairwise` 较强，但不是唯一解释；不同统计子集之间存在明显冗余

---

## 7. 关键日志入口

### 7.1 Pair-source ablation

- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBDB15K_r05_train_pseudo.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBDB15K_r05_train_only.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBDB15K_r05_pseudo_only.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBYG15K_r05_train_pseudo.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBYG15K_r05_train_only.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBYG15K_r05_pseudo_only.log`

### 7.2 Feature attribution ablation

- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBDB15K_r05_pairwise_only.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBDB15K_r05_stats_no_pairwise.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBDB15K_r05_drop_projected_token.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBYG15K_r05_pairwise_only.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBYG15K_r05_stats_no_pairwise.log`
- `log/run_outputs/dehr_learned_router/ablation_0514_175247/FBYG15K_r05_drop_projected_token.log`
