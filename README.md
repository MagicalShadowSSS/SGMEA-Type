# TIDEA

This repository contains the anonymous reproduction code for TIDEA.

## Data Layout
SGMEA File shared via Baidu Netdisk: mmkg.zip Link: https://pan.baidu.com/s/11pEZHDM7SAVN0wDk3vwm1w Extraction code: 4o9t

Download and unpack `mmkg.zip`, then organize the files as follows:

```text
ROOT
├── data
│   └── mmkg
└── code
    └── TIDEA
```

From `code/TIDEA`, all scripts use `../../data/mmkg` by default. The data root
can also be overridden by setting `DATA_ROOT` and `DATA_PATH`.

## Main Results

The following table reports the results reproduced by the fixed-checkpoint
TCMS + DEHR evaluation chain. `Delta Imp.` is the absolute percentage-point
improvement over the best baseline.

| Seed | Model | FB15K-DB15K Hits@1 | FB15K-DB15K Hits@10 | FB15K-DB15K MRR | FB15K-YG15K Hits@1 | FB15K-YG15K Hits@10 | FB15K-YG15K MRR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20% | MCLEA | 0.295 | 0.582 | 0.393 | 0.254 | 0.484 | 0.332 |
| 20% | MEAformer | 0.417 | 0.715 | 0.518 | 0.327 | 0.595 | 0.417 |
| 20% | PCMEA | 0.436 | 0.707 | 0.528 | 0.397 | 0.654 | 0.486 |
| 20% | LoginMEA | 0.536 | 0.784 | 0.667 | 0.572 | 0.758 | 0.652 |
| 20% | SGMEA | 0.543 | 0.777 | 0.625 | 0.587 | 0.826 | 0.670 |
| 20% | CDMEA | 0.549 | 0.741 | 0.611 | 0.434 | 0.682 | 0.523 |
| 20% | RICEA | 0.471 | 0.720 | 0.557 | 0.411 | 0.658 | 0.497 |
| 20% | TIDEA | **0.592** | **0.813** | **0.670** | **0.693** | **0.881** | **0.760** |
| 20% | Delta Imp. | +4.3% | +2.9% | +0.3% | +10.6% | +5.5% | +9.0% |
| 50% | MCLEA | 0.555 | 0.784 | 0.637 | 0.501 | 0.705 | 0.574 |
| 50% | MEAformer | 0.619 | 0.843 | 0.698 | 0.560 | 0.778 | 0.639 |
| 50% | PCMEA | 0.633 | 0.825 | 0.701 | 0.606 | 0.785 | 0.652 |
| 50% | LoginMEA | 0.712 | 0.857 | 0.755 | 0.648 | 0.805 | 0.699 |
| 50% | SGMEA | 0.716 | 0.882 | 0.775 | 0.780 | 0.924 | 0.832 |
| 50% | CDMEA | 0.675 | 0.861 | 0.737 | 0.628 | 0.824 | 0.692 |
| 50% | RICEA | 0.648 | 0.852 | 0.721 | 0.617 | 0.811 | 0.687 |
| 50% | TIDEA | **0.756** | **0.901** | **0.808** | **0.830** | **0.938** | **0.871** |
| 50% | Delta Imp. | +4.0% | +1.9% | +3.3% | +5.0% | +1.4% | +3.9% |
| 80% | MCLEA | 0.735 | 0.890 | 0.790 | 0.667 | 0.824 | 0.722 |
| 80% | MEAformer | 0.765 | 0.916 | 0.820 | 0.703 | 0.873 | 0.766 |
| 80% | PCMEA | 0.744 | 0.904 | 0.818 | 0.715 | 0.869 | 0.784 |
| 80% | LoginMEA | 0.773 | 0.908 | 0.822 | 0.724 | 0.904 | 0.803 |
| 80% | SGMEA | 0.815 | 0.931 | 0.828 | 0.857 | 0.951 | 0.894 |
| 80% | CDMEA | 0.804 | 0.931 | 0.859 | 0.743 | 0.891 | 0.806 |
| 80% | RICEA | 0.776 | 0.916 | 0.829 | 0.734 | 0.892 | 0.792 |
| 80% | TIDEA | **0.834** | **0.938** | **0.872** | **0.884** | **0.969** | **0.916** |
| 80% | Delta Imp. | +1.9% | +0.7% | +1.3% | +2.7% | +1.8% | +2.2% |

## Reproduce The Main Results

Run the following command to reproduce all TIDEA rows in the main table on one
GPU. The script evaluates six settings: two datasets and three seed ratios.

```bash
cd code/TIDEA
GPU_ID=0 ./run_reproduce_TIDEA_paper_single_gpu.sh
```

The command expects the following files to exist under `../../data/mmkg`:

```text
anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl
anchors_nameless/FBYG15K/norm_anchor_type_fixed.jsonl
TIDEA/save/*.pkl
```

After completion, the aggregated metrics are written to:

```text
log/run_outputs/<RUN_TAG>/paper_summary.json
```

To use a custom data location or Python executable:

```bash
cd code/TIDEA
DATA_ROOT=<DATA_ROOT> DATA_PATH=mmkg PYTHON=<PYTHON_BIN> GPU_ID=0 \
  ./run_reproduce_TIDEA_paper_single_gpu.sh
```

## Generate Type Anchors

If the type-anchor files are missing, generate them with the single-GPU type
construction chain:

```bash
cd code/TIDEA
DATA_CHOICE=FBDB15K LLM_PATH=<LOCAL_LLM_PATH> GPU_ID=0 ./run_generate_types_single_gpu.sh
DATA_CHOICE=FBYG15K LLM_PATH=<LOCAL_LLM_PATH> GPU_ID=0 ./run_generate_types_single_gpu.sh
```

This generates:

```text
../../data/mmkg/anchors_nameless/<DATASET>/norm_anchor_type_fixed.jsonl
```

For a parser-only smoke test without loading an LLM:

```bash
cd code/TIDEA
DATA_CHOICE=FBDB15K RULE_BASED=1 MAX_ENTITIES=5 SKIP_TYPE_FIX=1 ./run_generate_types_single_gpu.sh
```

## TCMS And DEHR Ablations

The following command evaluates fixed-checkpoint variants for TCMS, DEHR, and
TCMS + DEHR:

```bash
cd code/TIDEA
GPU_ID=0 ./run_fixed_ckpt_tcms_dehr_effect_single_gpu.sh
```

The outputs are saved under:

```text
log/run_outputs/<RUN_TAG>/
```
