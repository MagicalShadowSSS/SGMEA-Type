import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg as Config
from main import Runner
from torchlight import set_seed


class SimpleLogger:
    def info(self, msg):
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit TCMS and DEHR mechanism statistics.")
    parser.add_argument("--data_root", default=os.path.abspath(os.path.join(ROOT, "..", "..", "data")))
    parser.add_argument("--data_path", default="mmkg")
    parser.add_argument("--datasets", default="FBDB15K,FBYG15K")
    parser.add_argument("--data_rate", type=float, default=0.5)
    parser.add_argument("--data_rates", default="", help="Optional comma-separated seed alignment rates, e.g. 0.2,0.5,0.8")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default="log/audits/mechanism_analysis/fbdb_fbyg_r05_mechanism.json")
    parser.add_argument("--skip_dehr", action="store_true")
    parser.add_argument("--skip_tcms", action="store_true")
    return parser.parse_args()


def ckpt_for(dataset, rate, variant):
    key = f"{dataset}:{rate}:{variant}"
    table = {
        "FBDB15K:0.2:baseline": "TIDEA_FBDB15K_0.2_baseline_warm_fbdb_rates_0511_104108_r02_",
        "FBDB15K:0.2:tcms": "TIDEA_FBDB15K_0.2_fbdb_r02_tcms_keep_ckpt_",
        "FBDB15K:0.5:baseline": "TIDEA_FBDB15K_0.5_baseline_warm_fbdb_rates_0511_104108_r05_",
        "FBDB15K:0.5:tcms": "TIDEA_FBDB15K_0.5_fbdb_r05_keep_repro_seed42_",
        "FBDB15K:0.8:baseline": "TIDEA_FBDB15K_0.8_baseline_warm_fbdb_rates_0511_104108_r08_",
        "FBDB15K:0.8:tcms": "TIDEA_FBDB15K_0.8_fbdb_r08_tcms_keep_ckpt_",
        "FBYG15K:0.2:baseline": "TIDEA_FBYG15K_0.2_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r02_",
        "FBYG15K:0.2:tcms": "TIDEA_FBYG15K_0.2_fbyg_r02_tcms_keep_ckpt_",
        "FBYG15K:0.5:baseline": "TIDEA_FBYG15K_0.5_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r05_",
        "FBYG15K:0.5:tcms": "TIDEA_FBYG15K_0.5_fbyg_r05_tcms_keep_ckpt_",
        "FBYG15K:0.8:baseline": "TIDEA_FBYG15K_0.8_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r08_",
        "FBYG15K:0.8:tcms": "TIDEA_FBYG15K_0.8_fbyg_r08_tcms_keep_ckpt_",
    }
    if key not in table:
        raise ValueError(f"No checkpoint mapping for {key}")
    return table[key]


def anchor_for(data_root, data_path, dataset):
    return os.path.join(data_root, data_path, "anchors_nameless", dataset, "norm_anchor_type_fixed.jsonl")


def build_args(base, dataset, data_rate, checkpoint, use_tcms=False, use_dehr=False):
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    try:
        config = Config()
        config.get_args()
    finally:
        sys.argv = old_argv
    args = config.cfg
    args.data_root = base.data_root
    args.data_path = base.data_path
    args.data_choice = dataset
    args.data_split = "norm"
    args.data_rate = data_rate
    args.model_name = "TIDEA"
    args.model_name_save = checkpoint
    args.gpu = base.gpu
    args.device = torch.device("cuda")
    args.only_test = 1
    args.save_model = 0
    args.no_tensorboard = True
    args.exp_name = "mechanism_audit"
    args.exp_id = f"mechanism_{dataset.lower()}_{'tcms' if use_tcms else 'dehr' if use_dehr else 'base'}"
    args.batch_size = 2048
    args.workers = 0
    args.dist = 0
    args.rank = 0
    args.accumulation_steps = 1
    args.scheduler = "fixed"
    args.attr_dim = 300
    args.img_dim = 300
    args.name_dim = 300
    args.char_dim = 300
    args.hidden_size = 300
    args.hidden_units = "300,300,300"
    args.tau = 0.1
    args.structure_encoder = "gat"
    args.num_attention_heads = 1
    args.num_hidden_layers = 1
    args.use_surface = 0
    args.use_intermediate = 0
    args.disable_tidea_guidance = True
    args.csls = True
    args.csls_k = 3
    args.random_seed = base.seed
    args.external_anchor_type_jsonl = anchor_for(base.data_root, base.data_path, dataset)

    if use_tcms:
        args.use_tcms = True
        args.tcms_prefusion = True
        args.tcms_zero_init_residual = True
        args.tcms_noise_same_type_ratio = 0.7
        args.tcms_disable_gate_match_features = True
        args.tcms_beta = 0.25
        args.tcms_beta_warmup_start = -1
        args.tcms_beta_warmup_end = 150
        args.tcms_pretrain_loss_weight = 0.002
        args.tcms_selfsup_start = 100
        args.tcms_selfsup_warmup_end = 170
        args.tcms_sparse_weight = 0.03
        args.tcms_gate_init = -3.0
        args.tcms_missing_mode = "keep"

    if use_dehr:
        args.use_dehr_router = True
        args.type_modality_bias_only_train = True
        args.dehr_direction_mode = "learned_type"
        args.dehr_direction_values = "0,0,0,0"
        args.dehr_global_direction_weight = 0.0
        args.dehr_type_direction_residual_scale = 0.0
        args.dehr_type_direction_residual_max = 0.0
        args.dehr_rho_init_values = "0.7,0.7,0.7,0.7,0.7,0.7"
        args.dehr_rho_max = 1.0
        args.dehr_bias_clamp = 0.35
        args.dehr_learned_bias_scale = 0.35
        args.dehr_contextual_mix = 0.15
        args.dehr_contextual_source = "residual"
        args.dehr_learned_bias_l2_weight = 1e-4
        args.dehr_stat_feature_mask = "none"
        args.dehr_calib_pretrain_epochs = 25
        args.dehr_calib_pretrain_lr = 5e-3
        args.dehr_calib_pretrain_tau = 0.05
        args.dehr_calib_pretrain_batch_size = 256
        args.dehr_calib_pretrain_val_ratio = 0.2
        args.dehr_calib_pretrain_patience = 8
        args.dehr_calib_bias_l2 = 1e-4
        args.dehr_calib_pair_source = "train_pseudo"
        args.dehr_calib_selection = "final"
        args.dehr_calib_pseudo_margin = 0.04
        args.dehr_calib_pseudo_csls_k = 1
        args.dehr_calib_pseudo_source = "dropout_consensus"
        args.dehr_calib_consensus_min_votes = 2
        args.lr = 3e-4

    updated = config.update_train_configs()
    updated.device = torch.device("cuda")
    return updated


def make_runner(args):
    set_seed(args.random_seed)
    runner = Runner(args, writer=None, logger=SimpleLogger(), rank=0)
    runner.epoch = 0
    return runner


def cosine_mean(a, b):
    return float(F.cosine_similarity(F.normalize(a, dim=-1), F.normalize(b, dim=-1), dim=-1).mean().item())


@torch.no_grad()
def visual_calibration_stats(runner):
    model = runner.model
    model.eval()
    model.joint_emb_generat()
    fusion = model.multimodal_encoder.fusion
    tcms = getattr(fusion, "tcms", None)
    if tcms is None:
        raise RuntimeError("TCMS is not enabled.")
    raw_embs = getattr(fusion, "last_tcms_raw_embs", None)
    clean_embs = getattr(fusion, "last_modal_embs", None)
    if raw_embs is None or clean_embs is None:
        raise RuntimeError("Missing cached TCMS modal embeddings.")

    raw = torch.stack(raw_embs[:tcms.modal_num], dim=1)
    clean = torch.stack(clean_embs[:tcms.modal_num], dim=1)
    type_ids = model.entity_type_ids.to(raw.device)
    available = model.image_available.to(raw.device).bool() if model.image_available is not None else None
    _, _, _, anchor = tcms._encode(raw, type_ids, available)
    raw_img = raw[:, 0, :]
    clean_img = clean[:, 0, :]

    pair_dataset = runner.test_set if getattr(runner, "test_set", None) is not None else runner.eval_set
    pairs = torch.as_tensor(np.asarray(pair_dataset.data, dtype=np.int64), device=raw.device)
    left = pairs[:, 0].long()
    right = pairs[:, 1].long()
    neg_right = right.roll(1)
    valid_neg = neg_right != right
    if not valid_neg.all():
        neg_right = right.roll(2)

    def pair_sim(emb, a, b):
        return F.cosine_similarity(F.normalize(emb[a], dim=-1), F.normalize(emb[b], dim=-1), dim=-1)

    raw_pos = pair_sim(raw_img, left, right)
    raw_neg = pair_sim(raw_img, left, neg_right)
    clean_pos = pair_sim(clean_img, left, right)
    clean_neg = pair_sim(clean_img, left, neg_right)

    out = {
        "vsc_before": cosine_mean(raw_img, anchor),
        "vsc_after": cosine_mean(clean_img, anchor),
        "pos_sim_before": float(raw_pos.mean().item()),
        "pos_sim_after": float(clean_pos.mean().item()),
        "gap_before": float((raw_pos - raw_neg).mean().item()),
        "gap_after": float((clean_pos - clean_neg).mean().item()),
        "delta_norm": float((clean_img - raw_img).norm(dim=-1).mean().item()),
        "gate_mean": float(tcms.last_gate.mean().item()) if getattr(tcms, "last_gate", None) is not None else None,
        "missing_rate": float((~available).float().mean().item()) if available is not None else None,
        "test_pair_count": int(pairs.shape[0]),
    }
    return out


@torch.no_grad()
def type_routing_stats(runner, top_k=6):
    model = runner.model
    model.eval()
    model.joint_emb_generat()
    fusion = model.multimodal_encoder.fusion
    w0 = fusion.last_weight_norm.detach()
    w1 = fusion.last_weight_final.detach()
    type_ids = model.entity_type_ids.detach().to(w0.device)
    type_names = list(model.top_type_names)
    modal_names = ["Image", "Attribute", "Relation", "Structure"][: w0.shape[1]]
    delta = (w1 - w0).abs().sum(dim=1)
    entropy0 = -(w0.clamp_min(1e-12) * torch.log(w0.clamp_min(1e-12))).sum(dim=1)
    entropy1 = -(w1.clamp_min(1e-12) * torch.log(w1.clamp_min(1e-12))).sum(dim=1)

    rows = []
    for type_id, name in enumerate(type_names):
        mask = type_ids == type_id
        count = int(mask.sum().item())
        if count <= 0:
            continue
        before = w0[mask].mean(dim=0)
        after = w1[mask].mean(dim=0)
        rows.append({
            "type": name,
            "count": count,
            "before": {m: float(before[i].item()) for i, m in enumerate(modal_names)},
            "after": {m: float(after[i].item()) for i, m in enumerate(modal_names)},
            "delta_l1": float(delta[mask].mean().item()),
            "entropy_before": float(entropy0[mask].mean().item()),
            "entropy_after": float(entropy1[mask].mean().item()),
        })
    rows_sorted = sorted(rows, key=lambda x: x["count"], reverse=True)
    out = {
        "modal_names": modal_names,
        "avg_delta_l1": float(delta.mean().item()),
        "entropy_before": float(entropy0.mean().item()),
        "entropy_after": float(entropy1.mean().item()),
        "type_rows": rows_sorted[:top_k],
        "all_type_rows": rows_sorted,
    }
    return out


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    rates = [float(x.strip()) for x in args.data_rates.split(",") if x.strip()] if args.data_rates else [args.data_rate]
    results = {}
    for rate in rates:
        rate_key = f"{rate:g}"
        for dataset in datasets:
            print(f"[dataset] {dataset} [rate] {rate_key}", flush=True)
            dataset_out = {"dataset": dataset, "data_rate": rate}
            if not args.skip_tcms:
                tcms_args = build_args(args, dataset, rate, ckpt_for(dataset, rate, "tcms"), use_tcms=True)
                runner = make_runner(tcms_args)
                dataset_out["visual_calibration"] = visual_calibration_stats(runner)
                del runner
                torch.cuda.empty_cache()
            if not args.skip_dehr:
                dehr_args = build_args(args, dataset, rate, ckpt_for(dataset, rate, "baseline"), use_dehr=True)
                runner = make_runner(dehr_args)
                runner.dehr_calibration_pretrain()
                dataset_out["type_routing"] = type_routing_stats(runner)
                del runner
                torch.cuda.empty_cache()
            results.setdefault(dataset, {})[rate_key] = dataset_out

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[wrote] {args.output_json}", flush=True)

    print("\n[Visual Evidence Calibration]")
    print("Dataset\tVSC Before\tVSC After\tPos Before\tPos After\tGap Before\tGap After")
    for dataset, by_rate in results.items():
        for rate_key, data in by_rate.items():
            v = data.get("visual_calibration")
            if not v:
                continue
            print(
                f"{dataset}\t{rate_key}\t{v['vsc_before']:.4f}\t{v['vsc_after']:.4f}\t"
                f"{v['pos_sim_before']:.4f}\t{v['pos_sim_after']:.4f}\t"
                f"{v['gap_before']:.4f}\t{v['gap_after']:.4f}"
            )
    print("\n[Type-conditioned Evidence Routing]")
    for dataset, by_rate in results.items():
        for rate_key, data in by_rate.items():
            r = data.get("type_routing")
            if not r:
                continue
            print(
                f"{dataset}\t{rate_key}\tDeltaL1={r['avg_delta_l1']:.4f}\t"
                f"EntropyBefore={r['entropy_before']:.4f}\tEntropyAfter={r['entropy_after']:.4f}"
            )
            for row in r["type_rows"]:
                after = ", ".join(f"{m}:{row['after'][m]:.3f}" for m in r["modal_names"])
                print(f"  {row['type']}({row['count']}): {after} | delta={row['delta_l1']:.4f}")


if __name__ == "__main__":
    main()
