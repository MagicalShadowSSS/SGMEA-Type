# Phase14 TCMS-Former Visual Sanitization Log (2026-05-18)

## Goal

围绕“模态质量提升/视觉去噪/缺失补全”重新设计一个不与四模态权重路由重叠的模块。当前阶段只验证 `FBDB15K r=0.5` 与 `FBYG15K r=0.5`，目标是两套数据集 Hits@1 至少提升 1 个点。

Baseline checkpoint 指标：

| Dataset | L2R H@1 | R2L H@1 |
|---|---:|---:|
| FBDB15K r=0.5 | 0.7112 | 0.7187 |
| FBYG15K r=0.5 | 0.7902 | 0.7882 |

## TCMS-Former V1 Design

TCMS-Former 不输出模态权重，也不改变四模态融合权重。它只在融合前对视觉 token 做保守残差净化：

```text
tokens = [type token, image token + availability, attr token, rel token, graph token]
h = Lightweight Transformer(tokens)
anchor = LN(mean(non-image tokens) + type token)
proto = EMA type visual memory(anchor, type)
z_img_clean = normalize(z_img + beta * gate * residual)
```

关键约束：

- `tcms_zero_init_residual`: 残差头零初始化，模块初始等价于 identity，避免随机扰动收益。
- `tcms_only_train`: 冻结主干，只训练 TCMS 参数，避免主干重训造成归因混乱。
- `type_modality_weight_delta_abs_mean=0`: 确认没有走四模态权重调整路线。
- 语义锚点来自非视觉模态，避免 InfoNCE 追逐噪声图像本身。
- hard corruption 使用同类型替换为主，但当前噪声 gate 探针没有学出有效 clean/corrupt 间隔，不能作为核心贡献过度宣称。

## Iteration Records

### I1: Random TCMS Sanity

- **[Hypothesis]:** 如果随机初始化就大幅提升，说明收益可能来自随机视觉扰动，不适合写论文。
- **[Arch Change]:** 开启 TCMS，但不使用 zero-init；only-test。
- **[Result vs. Goal]:** FBDB `0.7405/0.7464`，FBYG `0.8125/0.8230`，表面很强。
- **[Action]:** Revert as unsafe. 随机模块带来收益不可解释。

### I2: Zero-Init Identity Check

- **[Hypothesis]:** zero-init 后 only-test 应回到 baseline，证明收益不是随机扰动。
- **[Arch Change]:** `--tcms_zero_init_residual --only_test`。
- **[Result vs. Goal]:** FBDB `0.7112/0.7187`，FBYG `0.7902/0.7882`，回到 baseline。
- **[Action]:** Accept. 这是后续实验的安全起点。

### I3: Zero-Init + EA Fine-Tuning (Main)

- **[Hypothesis]:** 冻结主干，只让 EA loss 通过 TCMS 的视觉净化残差回传，可以学习有用的视觉修正而不是调权重。
- **[Arch Change]:** `--tcms_zero_init_residual --tcms_only_train --tcms_only_train_use_ea --tcms_pretrain_loss_weight 0.05 --tcms_missing_mode keep --tcms_disable_gate_match_features`。
- **[Result vs. Goal]:** 达到双数据集 +1 点目标。

| Dataset | L2R H@1 | R2L H@1 | Gain L2R | Gain R2L |
|---|---:|---:|---:|---:|
| FBDB15K | 0.7338 | 0.7434 | +0.0226 | +0.0247 |
| FBYG15K | 0.8112 | 0.8171 | +0.0210 | +0.0289 |

- **[Action]:** Accept as current main line. 探针显示 `type_modality_weight_delta_abs_mean=0.0000`。

### I4: Pure Self-Supervised TCMS

- **[Hypothesis]:** 如果不用 EA，仅靠语义锚点对齐也有提升，说明模块不是完全依赖监督对齐损失。
- **[Arch Change]:** `--tcms_zero_init_residual --tcms_only_train --tcms_pretrain_loss_weight 1.0`，不使用 EA。
- **[Result vs. Goal]:** 仍然达到或接近目标，尤其 H@1 提升稳定。

| Dataset | L2R H@1 | R2L H@1 | Gain L2R | Gain R2L |
|---|---:|---:|---:|---:|
| FBDB15K | 0.7317 | 0.7451 | +0.0205 | +0.0264 |
| FBYG15K | 0.8111 | 0.8171 | +0.0209 | +0.0289 |

- **[Action]:** Accept as support evidence. 但 Top10/Top50 有时不如 EA 版，论文主结果仍应使用 EA fine-tuning。

### I5: Fixed Semantic Anchor for Corruption Probe

- **[Hypothesis]:** hard corruption 时保持原始非视觉 anchor，不让 corrupt image 影响 anchor，可让 gate 更容易学视觉-语义冲突。
- **[Arch Change]:** corruption branch 使用原始 `h_type/anchor/proto` 计算 gate；加入 `tcms_gate_gap_probe`。
- **[Result vs. Goal]:** 收益保持，但 gate gap 仍约等于 0。

| Dataset | Setting | L2R H@1 | R2L H@1 |
|---|---|---:|---:|
| FBDB15K | fixed-anchor EA | 0.7336 | 0.7434 |
| FBYG15K | fixed-anchor EA | 0.8112 | 0.8171 |

- **[Action]:** Partial accept. 固定 anchor 合理，但不能声称 gate 已学到强噪声二分类。

### I6: Dense Gate Match Features Ablation

- **[Hypothesis]:** gate 输入加入 `h_img * anchor` 与 `|h_img-anchor|`，显式暴露视觉-语义匹配/冲突维度，可能提高噪声判别可解释性。
- **[Arch Change]:** gate/residual 输入从 `[h_img,h_type,anchor,proto]` 扩展到 `[h_img,h_type,anchor,proto,h_img*anchor,|h_img-anchor|]`；可用 `--tcms_disable_gate_match_features` 消融。
- **[Result vs. Goal]:** 收益仍达标，但 gate gap 仍接近 0；FBYG 比 I3 略低。

| Dataset | L2R H@1 | R2L H@1 | Gain L2R | Gain R2L |
|---|---:|---:|---:|---:|
| FBDB15K | 0.7355 | 0.7430 | +0.0243 | +0.0243 |
| FBYG15K | 0.8096 | 0.8129 | +0.0194 | +0.0247 |

- **[Action]:** Revert for main setting. 可作为辅助结构，但噪声 gate 仍不能作为主要卖点；主配置显式关闭该增强以减少结构复杂度。

### I7: No Transformer Ablation

- **[Hypothesis]:** 如果 `tcms_layers=0` 不掉点，Transformer 就只是装饰；如果掉点，则 Transformer 有必要。
- **[Arch Change]:** `--tcms_layers 0`，其他保持 EA fine-tuning。
- **[Result vs. Goal]:** 仍比 baseline 高，但低于 1-layer TCMS。

| Dataset | L2R H@1 | R2L H@1 | Drop vs I3 L2R | Drop vs I3 R2L |
|---|---:|---:|---:|---:|
| FBDB15K | 0.7279 | 0.7358 | -0.0057 | -0.0076 |
| FBYG15K | 0.8075 | 0.8136 | -0.0037 | -0.0035 |

- **[Action]:** Accept Transformer as useful but not sole source. 论文可以说轻量跨模态交互提升语义锚点质量，不能夸成唯一贡献。

### I8: Explicit Missing Replacement

- **[Hypothesis]:** 对缺失图像用 semantic anchor + type prototype 替换可能改善 FBYG，因为 FBYG missing rate 约 18.8%。
- **[Arch Change]:** `--tcms_missing_mode anchor_proto`。
- **[Result vs. Goal]:** FBDB 可达标，但 FBYG 低于 keep；说明强替换会伤害现有 checkpoint 的视觉/融合分布。

| Dataset | L2R H@1 | R2L H@1 | Compare to keep |
|---|---:|---:|---|
| FBDB15K | 0.7341 | 0.7445 | 接近 keep |
| FBYG15K | 0.8082 | 0.8134 | 低于 keep |

- **[Action]:** Revert to `keep` as main setting. 缺失补全当前只能作为可选策略，不能作为主贡献。

## Current Conclusion

1. TCMS-Former 的主要有效来源是：基于非视觉语义锚点的视觉特征残差净化 + EA fine-tuning，而不是四模态权重调整。
2. `zero-init` identity check 很关键：证明模块不是靠随机扰动获得收益。
3. `tcms_layers=0` 明显低于 1-layer，说明 Transformer 有实际价值，但不是全部收益来源。
4. 当前 hard corruption gate 没学出稳定 clean/corrupt 间隔，`tcms_noise_acc_probe≈0.5`、`tcms_gate_gap_probe≈0`。论文中不应把“噪声二分类器”作为核心结论，应把它降级为辅助约束/探针。
5. `missing_mode=anchor_proto` 没有超过 `keep`，说明显式缺失补全还不稳；主线应强调“保守视觉语义净化”，而不是强替换缺失视觉。

## Recommended Main Setting

当前主结果建议使用：

```bash
--use_tcms \
--tcms_zero_init_residual \
--tcms_only_train \
--tcms_only_train_use_ea \
--tcms_pretrain_loss_weight 0.05 \
--tcms_beta 0.25 \
--tcms_missing_mode keep \
--tcms_sparse_weight 0.01 \
--tcms_noise_same_type_ratio 0.7 \
--tcms_disable_gate_match_features
```

当前代码支持 dense gate match features，但主结果建议关闭它，因为该增强没有改善噪声 gate 探针，且会增加结构解释负担。
