# Type-Modality Audit Summary

Date: 2026-05-10

## Files

Script:

`audit_type_modality_oracle_sweep.py`

Output:

`/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/train_relation_hnm/type_modality_oracle_sweep_baseline_warm_fast.json`

## Actual SGMEA Final Embedding Layout

For FBDB15K with `use_surface=0`, the final joint embedding is:

```text
[img, attr, rel, graph, gat_img, gat_attr]
```

Shape:

```text
weight_norm: [27793, 6]
joint_emb:   [27793, 1800]
segment dim: 300
```

Global mean fusion weights:

```text
img      0.1744
attr     0.1658
rel      0.1423
graph    0.1753
gat_img  0.1743
gat_attr 0.1678
```

## Baseline Test Metrics

Overall:

```text
Avg Hits@1 = 0.7097
L2R Hits@1 = 0.7051
R2L Hits@1 = 0.7143
```

By type:

```text
Person        0.8778
Place         0.6797
Organization  0.5532
Creative Work 0.7850
```

## Oracle Sweep Result

Fast diagnostic sweep tested per-type modality scaling using the grid:

```text
0.7, 1.0, 1.3
```

Type-local gains:

```text
Person        +0.79%
Place         +2.11%
Organization  +2.40%
Creative Work +1.73%
```

Best combined configuration:

```text
Person:
  attr x 1.3
  rel  x 0.7

Place:
  graph x 1.3

Organization:
  graph x 1.3

Creative Work:
  graph x 1.3
```

Combined result:

```text
overall avg Hits@1: 0.7097 -> 0.7260
delta: +1.63%
```

Combined by type:

```text
Person        0.8778 -> 0.8873
Place         0.6797 -> 0.7024
Organization  0.5532 -> 0.5850
Creative Work 0.7850 -> 0.8032
```

## Interpretation

This is the first route after Phase 5 that shows a plausible `+1%` to `+2%` training-internalization upside.

Unlike close/safe HNM, this signal is not sparse. Every entity has a coarse type, so a type-aware router can affect the whole train/test distribution.

## Caveat

The sweep is an offline oracle-style reweighting of modality distances. It is not yet a trained model. The next step is to learn the same effect with a small trainable `type_bias` inside SGMEA, while preserving standard NN/CSLS inference.

