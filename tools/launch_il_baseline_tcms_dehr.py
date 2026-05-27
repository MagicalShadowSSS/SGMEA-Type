#!/usr/bin/env python3
"""Launch IL baseline and TCMS+DEHR experiments on multiple GPUs."""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time


CODE_ROOT = pathlib.Path(os.environ.get("CODE_ROOT", "/gly/tongqiang/dongyufeng/Mode-Test/SGMEA"))
DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", "/gly/tongqiang/dongyufeng/data/mmkg"))
PYTHON = os.environ.get("PYTHON", "/gly/tongqiang/anaconda3/envs/sgmea/bin/python")
GPU_IDS = [x.strip() for x in os.environ.get("GPU_IDS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
RUN_TAG = os.environ.get("RUN_TAG", f"phase27_il_baseline_tcms_dehr_{time.strftime('%m%d_%H%M%S')}")

EPOCHS = int(os.environ.get("EPOCHS", "1000"))
IL_START = int(os.environ.get("IL_START", "500"))
SEMI_LEARN_STEP = int(os.environ.get("SEMI_LEARN_STEP", "5"))
EVAL_EPOCH = int(os.environ.get("EVAL_EPOCH", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2048"))
LR = os.environ.get("LR", "5e-4")
SAVE_MODEL = int(os.environ.get("SAVE_MODEL", "1"))

LOG_DIR = CODE_ROOT / "log" / "run_outputs" / RUN_TAG
SAVE_DIR = DATA_ROOT / "SGMEA" / "save"


def anchor_for(dataset: str) -> pathlib.Path:
    return DATA_ROOT / "anchors_nameless" / dataset / "norm_anchor_type_fixed.jsonl"


def tag_for(dataset: str, rate: str) -> str:
    prefix = "fbdb" if dataset == "FBDB15K" else "fbyg"
    return f"{prefix}_r{rate.replace('.', '')}"


def saved_ckpt_name(dataset: str, rate: str, exp_id: str) -> str:
    return f"SGMEA_{dataset}_{rate}_{exp_id}_il{EPOCHS - IL_START}_b{IL_START}_"


def common_train_args(dataset: str, rate: str, exp_id: str, anchor: pathlib.Path) -> list[str]:
    return [
        "main.py",
        "--gpu", "0",
        "--data_path", str(DATA_ROOT),
        "--data_split", "norm",
        "--model_name", "SGMEA",
        "--data_choice", dataset,
        "--data_rate", rate,
        "--epoch", str(EPOCHS),
        "--il_start", str(IL_START),
        "--semi_learn_step", str(SEMI_LEARN_STEP),
        "--eval_epoch", str(EVAL_EPOCH),
        "--only_test", "0",
        "--save_model", str(SAVE_MODEL),
        "--lr", LR,
        "--hidden_units", "300,300,300",
        "--batch_size", str(BATCH_SIZE),
        "--csls",
        "--csls_k", "3",
        "--random_seed", "42",
        "--workers", "0",
        "--dist", "0",
        "--accumulation_steps", "1",
        "--scheduler", "cos",
        "--attr_dim", "300",
        "--img_dim", "300",
        "--name_dim", "300",
        "--char_dim", "300",
        "--hidden_size", "300",
        "--tau", "0.1",
        "--structure_encoder", "gat",
        "--num_attention_heads", "1",
        "--num_hidden_layers", "1",
        "--use_surface", "0",
        "--use_intermediate", "0",
        "--enable_sota",
        "--disable_sgmea_guidance",
        "--disable_early_stop",
        "--il",
        "--external_anchor_type_jsonl", str(anchor),
        "--no_tensorboard",
        "--exp_name", RUN_TAG,
        "--exp_id", exp_id,
    ]


def common_eval_args(dataset: str, rate: str, exp_id: str, anchor: pathlib.Path, ckpt: str) -> list[str]:
    return [
        "main.py",
        "--gpu", "0",
        "--data_path", str(DATA_ROOT),
        "--data_split", "norm",
        "--model_name", "SGMEA",
        "--data_choice", dataset,
        "--data_rate", rate,
        "--model_name_save", ckpt,
        "--epoch", "0",
        "--eval_epoch", "1",
        "--only_test", "0",
        "--save_model", "0",
        "--lr", "3e-4",
        "--hidden_units", "300,300,300",
        "--batch_size", str(BATCH_SIZE),
        "--csls",
        "--csls_k", "3",
        "--random_seed", "42",
        "--workers", "0",
        "--dist", "0",
        "--accumulation_steps", "1",
        "--scheduler", "fixed",
        "--attr_dim", "300",
        "--img_dim", "300",
        "--name_dim", "300",
        "--char_dim", "300",
        "--hidden_size", "300",
        "--tau", "0.1",
        "--structure_encoder", "gat",
        "--num_attention_heads", "1",
        "--num_hidden_layers", "1",
        "--use_surface", "0",
        "--use_intermediate", "0",
        "--disable_sgmea_guidance",
        "--external_anchor_type_jsonl", str(anchor),
        "--no_tensorboard",
        "--exp_name", RUN_TAG,
        "--exp_id", exp_id,
    ]


TCMS_ARGS = [
    "--use_tcms",
    "--tcms_prefusion",
    "--tcms_zero_init_residual",
    "--tcms_noise_same_type_ratio", "0.7",
    "--tcms_disable_gate_match_features",
    "--tcms_beta", "0.25",
    "--tcms_beta_warmup_start", "70",
    "--tcms_beta_warmup_end", "150",
    "--tcms_pretrain_loss_weight", "0.002",
    "--tcms_selfsup_start", "100",
    "--tcms_selfsup_warmup_end", "170",
    "--tcms_sparse_weight", "0.03",
    "--tcms_gate_init", "-3.0",
    "--tcms_missing_mode", "keep",
]

DEHR_ARGS = [
    "--type_modality_bias_only_train",
    "--use_dehr_router",
    "--dehr_direction_mode", "learned_type",
    "--dehr_direction_values=0,0,0,0",
    "--dehr_global_direction_weight", "0.0",
    "--dehr_type_direction_residual_scale", "0.0",
    "--dehr_type_direction_residual_max", "0.0",
    "--dehr_rho_init_values", "0.7,0.7,0.7,0.7,0.7,0.7",
    "--dehr_rho_max", "1.0",
    "--dehr_bias_clamp", "0.35",
    "--dehr_learned_bias_scale", "0.35",
    "--dehr_contextual_mix", "0.15",
    "--dehr_contextual_source", "residual",
    "--dehr_learned_bias_l2_weight", "1e-4",
    "--dehr_stat_feature_mask", "none",
    "--dehr_calib_pretrain_epochs", "25",
    "--dehr_calib_pretrain_lr", "5e-3",
    "--dehr_calib_pretrain_tau", "0.05",
    "--dehr_calib_pretrain_batch_size", "256",
    "--dehr_calib_pretrain_val_ratio", "0.2",
    "--dehr_calib_pretrain_patience", "25",
    "--dehr_calib_bias_l2", "1e-4",
    "--dehr_calib_pair_source", "train_pseudo",
    "--dehr_calib_selection", "final",
    "--dehr_calib_pseudo_margin", "0.04",
    "--dehr_calib_pseudo_csls_k", "1",
    "--dehr_calib_pseudo_source", "dropout_consensus",
    "--dehr_calib_consensus_min_votes", "2",
]


def run_command(cmd: list[str], gpu: str, log, stage: str) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log.write(f"[start] {time.strftime('%F %T')} gpu={gpu} stage={stage}\n")
    log.write("[cmd] CUDA_VISIBLE_DEVICES={} {}\n".format(gpu, " ".join(repr(x) for x in cmd)))
    log.flush()
    proc = subprocess.Popen(cmd, cwd=str(CODE_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    status = proc.wait()
    log.write(f"[exit] {time.strftime('%F %T')} gpu={gpu} stage={stage} status={status}\n")
    log.flush()
    return status


def run_job(job: dict) -> int:
    log_path = pathlib.Path(job["log"])
    gpu = job["gpu"]
    with log_path.open("w", encoding="utf-8", errors="ignore") as log:
        if job["variant"] == "baseline":
            return run_command([PYTHON, "-u"] + job["train_args"], gpu, log, "baseline_il_train")
        if job["variant"] == "dehr_eval_only":
            return run_command([PYTHON, "-u"] + job["eval_args"] + TCMS_ARGS + DEHR_ARGS, gpu, log, "tcms_dehr_eval")

        status = run_command([PYTHON, "-u"] + job["train_args"] + TCMS_ARGS, gpu, log, "tcms_il_train")
        if status != 0:
            return status
        ckpt = job["ckpt"]
        ckpt_path = SAVE_DIR / f"{ckpt}.pkl"
        log.write(f"[checkpoint] expected={ckpt_path}\n")
        log.flush()
        if not ckpt_path.exists():
            log.write(f"[error] missing saved checkpoint {ckpt_path}\n")
            return 2
        return run_command([PYTHON, "-u"] + job["eval_args"] + TCMS_ARGS + DEHR_ARGS, gpu, log, "tcms_dehr_eval")


def parse_summary() -> list[dict]:
    metric_re = re.compile(r"Ep (?:Test|\d+) \| (l2r|r2l): acc of top \[1, 10, 50\] = \[([^\]]+)\].*?mrr = ([0-9.]+)")
    rows = []
    for log_path in sorted(LOG_DIR.glob("*.log")):
        text = log_path.read_text(errors="ignore")
        last = {}
        for direction, hstr, mrr in metric_re.findall(text):
            hits = [float(x) for x in re.findall(r"[0-9.]+", hstr)]
            last[direction] = {"h1": hits[0], "h10": hits[1], "mrr": float(mrr)}
        name = log_path.stem
        parts = name.split("_")
        row = {
            "log": log_path.name,
            "dataset": parts[0] if parts else "",
            "rate": parts[1][1:] if len(parts) > 1 and parts[1].startswith("r") else "",
            "variant": "tcms_dehr" if "tcms_dehr" in name else "baseline",
        }
        if "l2r" in last and "r2l" in last:
            row.update({
                "h1_l2r": last["l2r"]["h1"],
                "h1_r2l": last["r2l"]["h1"],
                "h1_avg": round((last["l2r"]["h1"] + last["r2l"]["h1"]) / 2, 6),
                "h10_avg": round((last["l2r"]["h10"] + last["r2l"]["h10"]) / 2, 6),
                "mrr_avg": round((last["l2r"]["mrr"] + last["r2l"]["mrr"]) / 2, 6),
                "status": "ok",
            })
        else:
            row["status"] = "missing_metrics"
        rows.append(row)
    return rows


def main() -> int:
    if not GPU_IDS:
        print("[error] GPU_IDS is empty", file=sys.stderr)
        return 1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(CODE_ROOT)

    print(f"[info] run_tag={RUN_TAG}")
    print(f"[info] log_dir={LOG_DIR}")
    print(f"[info] gpu_ids={','.join(GPU_IDS)}")
    print(f"[info] epochs={EPOCHS} il_start={IL_START} semi_learn_step={SEMI_LEARN_STEP} eval_epoch={EVAL_EPOCH} batch_size={BATCH_SIZE} lr={LR} save_model={SAVE_MODEL}")

    jobs = []
    gpu_idx = 0
    for dataset in ["FBDB15K", "FBYG15K"]:
        anchor = anchor_for(dataset)
        if not anchor.exists():
            print(f"[error] missing anchor {anchor}", file=sys.stderr)
            return 1
        for rate in ["0.2", "0.5", "0.8"]:
            tag = tag_for(dataset, rate)
            base_exp = f"{tag}_il_baseline_seed42"
            gpu = GPU_IDS[gpu_idx % len(GPU_IDS)]
            gpu_idx += 1
            jobs.append({
                "variant": "baseline",
                "gpu": gpu,
                "log": str(LOG_DIR / f"{base_exp}.log"),
                "train_args": common_train_args(dataset, rate, base_exp, anchor),
            })

            ours_exp = f"{tag}_il_tcms_dehr_seed42"
            ckpt = saved_ckpt_name(dataset, rate, ours_exp)
            gpu = GPU_IDS[gpu_idx % len(GPU_IDS)]
            gpu_idx += 1
            jobs.append({
                "variant": "tcms_dehr",
                "gpu": gpu,
                "log": str(LOG_DIR / f"{ours_exp}.log"),
                "ckpt": ckpt,
                "train_args": common_train_args(dataset, rate, ours_exp, anchor),
                "eval_args": common_eval_args(dataset, rate, f"{ours_exp}_dehr_eval", anchor, ckpt),
            })

    (LOG_DIR / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    active: list[tuple[dict, subprocess.Popen, str]] = []
    free_gpus = list(GPU_IDS)
    next_job = 0
    failed = False
    while next_job < len(jobs) or active:
        while next_job < len(jobs) and free_gpus:
            job = jobs[next_job]
            next_job += 1
            job["gpu"] = free_gpus.pop(0)
            print(f"[launch] gpu={job['gpu']} variant={job['variant']} log={job['log']}", flush=True)
            cmd = [PYTHON, str(CODE_ROOT / "tools" / "launch_il_baseline_tcms_dehr.py"), "--worker", json.dumps(job)]
            proc = subprocess.Popen(cmd, cwd=str(CODE_ROOT), env=os.environ.copy())
            active.append((job, proc, job["gpu"]))
        time.sleep(5)
        still_active = []
        for job, proc, gpu in active:
            status = proc.poll()
            if status is None:
                still_active.append((job, proc, gpu))
                continue
            print(f"[exit] status={status} log={job['log']}", flush=True)
            free_gpus.append(gpu)
            if status != 0:
                failed = True
        active = still_active
        if failed:
            for _, proc, _ in active:
                proc.terminate()
            return 1

    rows = parse_summary()
    (LOG_DIR / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summary] {LOG_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(run_job(json.loads(sys.argv[2])))
    sys.exit(main())
