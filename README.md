TIDEA
File shared via Baidu Netdisk: mmkg.zip Link: https://pan.baidu.com/s/11pEZHDM7SAVN0wDk3vwm1w Extraction code: 4o9t

ROOT
├── data
│   └── mmkg
└── code
    └── TIDEA

## Generate Type Anchors

Run the single-GPU type construction chain before training if
`anchors_nameless/<DATASET>/norm_anchor_type_fixed.jsonl` is missing:

```bash
cd code/TIDEA
DATA_CHOICE=FBDB15K LLM_PATH=<LOCAL_LLM_PATH> ./run_generate_types_single_gpu.sh
DATA_CHOICE=FBYG15K LLM_PATH=<LOCAL_LLM_PATH> ./run_generate_types_single_gpu.sh
```

No absolute path is required inside the code. Override paths with environment
variables when needed: `DATA_ROOT`, `DATA_PATH`, `LLM_PATH`, `GPU_ID`, and
`PYTHON`. For a quick parser-only smoke test, set `RULE_BASED=1`.
