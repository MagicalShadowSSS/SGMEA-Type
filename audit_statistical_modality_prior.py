import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model import SGMEA
from src.data import load_data
from stage_a_projection_probe import SimpleLogger, build_model_args, load_checkpoint


MODAL_ORDER = ["img", "attr", "rel", "graph"]
FOCUS_TYPES = ["Person", "Place", "Organization", "Creative Work"]


DEFAULT_DATASETS = {
    "FBDB15K": {
        "checkpoint": "SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05_",
        "type_jsonl": "/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl",
        "reference_json": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/train_relation_hnm/type_modality_oracle_4seg_refine_0510_193727.json",
        "reference_key": "combined_greedy.best.scales",
        "visual_low_ratio": 0.8646,
        "visual_very_low_ratio": 0.4350,
    },
    "FBYG15K": {
        "checkpoint": "SGMEA_FBYG15K_0.5_baseline_warm_fbyg_specific_prior_0512_111257_lr5e4_r05_",
        "type_jsonl": "/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBYG15K/norm_anchor_type_fixed.jsonl",
        "reference_json": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/train_relation_hnm/fbyg_type_modality_oracle_4seg_refine_4gpu_0512_015217/fbyg_specific_prior_best_global.json",
        "reference_key": "combined_greedy.best.scales",
        "visual_low_ratio": 0.8417,
        "visual_very_low_ratio": 0.4182,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive type-modality priors from training-set statistics and compare them with diagnostic priors."
    )
    parser.add_argument("--data_root", default="/gly/tongqiang/dongyufeng/data")
    parser.add_argument("--data_path", default="mmkg")
    parser.add_argument("--data_split", default="norm")
    parser.add_argument("--data_rate", type=float, default=0.5)
    parser.add_argument("--datasets", default="FBDB15K,FBYG15K")
    parser.add_argument("--output_json", default="log/audits/statistical_modality_prior/stat_prior_0512.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--scale_min", type=float, default=0.4)
    parser.add_argument("--scale_max", type=float, default=1.7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no_csls", action="store_true", default=False)
    parser.add_argument("--csls_k", type=int, default=3)
    return parser.parse_args()


def get_by_path(payload, key_path):
    node = payload
    for key in key_path.split("."):
        if key:
            node = node[key]
    return node


def load_reference(path, key_path):
    with open(path, "r", encoding="utf-8") as fp:
        return get_by_path(json.load(fp), key_path)


def build_type_maps(kgs):
    type_ids = kgs.get("entity_type_ids")
    type_names = kgs.get("top_type_names", [])
    if type_ids is None:
        raise ValueError("entity_type_ids missing")
    ids = type_ids.detach().cpu().numpy() if torch.is_tensor(type_ids) else np.asarray(type_ids)
    ent_to_type = {}
    for ent, tid in enumerate(ids):
        ent_to_type[ent] = type_names[int(tid)] if int(tid) < len(type_names) else str(int(tid))
    return ent_to_type


@torch.no_grad()
def get_modal_embeddings(model):
    encoder = model.multimodal_encoder
    gph_emb = encoder.cross_graph_model(encoder.entity_emb(model.input_idx), model.adj) if model.args.w_gcn else None
    img_emb = encoder.img_fc(model.img_features) if model.args.w_img else None
    rel_emb = encoder.rel_fc(model.rel_features) if model.args.w_rel else None
    attr_emb = encoder.att_fc(model.att_features) if model.args.w_attr else None
    embs = {
        "img": img_emb,
        "attr": attr_emb,
        "rel": rel_emb,
        "graph": gph_emb,
    }
    return {name: F.normalize(emb, dim=1) for name, emb in embs.items() if emb is not None}


def load_degree(data_path, dataset, split):
    degree = Counter()
    base = os.path.join(data_path, dataset, split)
    for name in ("triples_1", "triples_2"):
        path = os.path.join(base, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h, _, t = [int(x) for x in parts[:3]]
                degree[h] += 1
                degree[t] += 1
    return degree


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


@torch.no_grad()
def train_pair_modal_stats(embs, train_ill, ent_to_type, batch_size):
    left_ids = torch.as_tensor(train_ill[:, 0], dtype=torch.long, device="cuda")
    right_ids = torch.as_tensor(train_ill[:, 1], dtype=torch.long, device="cuda")
    left_types = [ent_to_type.get(int(x), "Entity") for x in train_ill[:, 0]]

    stats = {
        typ: {
            modal: {"pos_sim": [], "margin": [], "hit1": [], "mrr": []}
            for modal in MODAL_ORDER
        }
        for typ in FOCUS_TYPES
    }
    stats["ALL"] = {
        modal: {"pos_sim": [], "margin": [], "hit1": [], "mrr": []}
        for modal in MODAL_ORDER
    }

    n = int(train_ill.shape[0])
    diag_all = torch.arange(n, device="cuda")
    for modal, emb in embs.items():
        right_emb = emb[right_ids]
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            rows = emb[left_ids[start:end]]
            sim = rows @ right_emb.T
            local = torch.arange(end - start, device="cuda")
            pos_cols = diag_all[start:end]
            pos = sim[local, pos_cols]
            sim_without_pos = sim.clone()
            sim_without_pos[local, pos_cols] = -1e9
            top_neg = sim_without_pos.max(dim=1).values
            margin = pos - top_neg
            rank = (sim > pos[:, None]).sum(dim=1) + 1
            hit1 = (rank == 1).float()
            mrr = 1.0 / rank.float()

            pos_np = pos.detach().cpu().numpy()
            margin_np = margin.detach().cpu().numpy()
            hit1_np = hit1.detach().cpu().numpy()
            mrr_np = mrr.detach().cpu().numpy()
            for offset, typ in enumerate(left_types[start:end]):
                if typ not in stats:
                    continue
                for key, arr in (
                    ("pos_sim", pos_np),
                    ("margin", margin_np),
                    ("hit1", hit1_np),
                    ("mrr", mrr_np),
                ):
                    val = float(arr[offset])
                    stats[typ][modal][key].append(val)
                    stats["ALL"][modal][key].append(val)

    return {
        typ: {
            modal: {key: summarize(vals) for key, vals in metric_map.items()}
            for modal, metric_map in modal_map.items()
        }
        for typ, modal_map in stats.items()
    }


def degree_stats(train_ill, ent_to_type, degree):
    out = {typ: [] for typ in FOCUS_TYPES}
    out["ALL"] = []
    for left, _ in train_ill:
        left = int(left)
        typ = ent_to_type.get(left, "Entity")
        val = math.log1p(float(degree[left]))
        out["ALL"].append(val)
        if typ in out:
            out[typ].append(val)
    return {typ: summarize(vals) for typ, vals in out.items()}


def minmax_by_modal(raw_scores):
    values = {modal: [] for modal in MODAL_ORDER}
    for typ in FOCUS_TYPES:
        for modal in MODAL_ORDER:
            values[modal].append(float(raw_scores[typ][modal]))
    out = {typ: {} for typ in FOCUS_TYPES}
    for modal in MODAL_ORDER:
        arr = np.asarray(values[modal], dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        den = hi - lo if hi > lo else 1.0
        for typ in FOCUS_TYPES:
            out[typ][modal] = (float(raw_scores[typ][modal]) - lo) / den
    return out


def center_to_scale(scores, alpha, scale_min, scale_max):
    prior = {}
    for typ in FOCUS_TYPES:
        vals = np.asarray([scores[typ][m] for m in MODAL_ORDER], dtype=np.float64)
        mu = float(vals.mean())
        prior[typ] = {}
        for modal in MODAL_ORDER:
            scale = 1.0 + alpha * (float(scores[typ][modal]) - mu)
            prior[typ][modal] = float(max(scale_min, min(scale_max, scale)))
    return prior


def compare(candidate, reference):
    diffs = []
    rows = {}
    for typ in FOCUS_TYPES:
        rows[typ] = {}
        for modal in MODAL_ORDER:
            cand = float(candidate[typ][modal])
            ref = float(reference[typ][modal])
            rows[typ][modal] = {
                "candidate": cand,
                "reference": ref,
                "abs_diff": abs(cand - ref),
            }
            diffs.append(abs(cand - ref))
    cand_flat = np.asarray([candidate[t][m] for t in FOCUS_TYPES for m in MODAL_ORDER], dtype=np.float64)
    ref_flat = np.asarray([reference[t][m] for t in FOCUS_TYPES for m in MODAL_ORDER], dtype=np.float64)
    corr = 0.0
    if np.std(cand_flat) > 0 and np.std(ref_flat) > 0:
        corr = float(np.corrcoef(cand_flat, ref_flat)[0, 1])
    return {
        "mae": float(np.mean(diffs)),
        "max_abs_diff": float(np.max(diffs)),
        "corr": corr,
        "by_type": rows,
    }


def extract_feature_scores(stats, degree_summary, visual_low_ratio, visual_very_low_ratio):
    features = {}
    for metric in ("pos_sim", "margin", "hit1", "mrr"):
        features[metric] = {
            typ: {modal: float(stats[typ][modal][metric]["mean"]) for modal in MODAL_ORDER}
            for typ in FOCUS_TYPES
        }

    # Type-level structural availability is only meaningful for graph, so keep it as an additive feature.
    deg_vals = np.asarray([degree_summary[t]["mean"] for t in FOCUS_TYPES], dtype=np.float64)
    deg_min, deg_max = float(deg_vals.min()), float(deg_vals.max())
    deg_den = deg_max - deg_min if deg_max > deg_min else 1.0
    degree_norm = {
        typ: (float(degree_summary[typ]["mean"]) - deg_min) / deg_den
        for typ in FOCUS_TYPES
    }

    quality = {
        "img": max(0.0, 1.0 - visual_low_ratio),
        "attr": 1.0,
        "rel": 1.0,
        "graph": 1.0,
    }
    severe_quality = {
        "img": max(0.0, 1.0 - 0.5 * visual_low_ratio - 0.5 * visual_very_low_ratio),
        "attr": 1.0,
        "rel": 1.0,
        "graph": 1.0,
    }
    return features, degree_norm, quality, severe_quality


def make_candidates(features, degree_norm, quality, severe_quality, args):
    candidates = []

    def add(name, scores, alpha_grid):
        for alpha in alpha_grid:
            candidates.append({
                "name": f"{name}|alpha={alpha:g}",
                "scores": scores,
                "alpha": alpha,
                "scales": center_to_scale(scores, alpha, args.scale_min, args.scale_max),
            })

    alpha_grid = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    for metric, raw in features.items():
        add(metric, raw, alpha_grid)
        add(f"{metric}_type_minmax", minmax_by_modal(raw), alpha_grid)
        add(
            f"{metric}_quality",
            {
                typ: {m: float(raw[typ][m]) * quality[m] for m in MODAL_ORDER}
                for typ in FOCUS_TYPES
            },
            alpha_grid,
        )
        add(
            f"{metric}_severe_quality",
            {
                typ: {m: float(raw[typ][m]) * severe_quality[m] for m in MODAL_ORDER}
                for typ in FOCUS_TYPES
            },
            alpha_grid,
        )

    # Reliability score: same-pair agreement + hard-negative separability + retrieval success.
    for w_margin in (0.5, 1.0, 1.5):
        for w_hit in (0.25, 0.5, 1.0):
            for w_deg in (0.0, 0.15, 0.3):
                for q_name, q in (("quality", quality), ("severe_quality", severe_quality)):
                    scores = {typ: {} for typ in FOCUS_TYPES}
                    for typ in FOCUS_TYPES:
                        for modal in MODAL_ORDER:
                            val = (
                                features["pos_sim"][typ][modal]
                                + w_margin * features["margin"][typ][modal]
                                + w_hit * features["hit1"][typ][modal]
                            )
                            if modal == "graph":
                                val += w_deg * degree_norm[typ]
                            val *= q[modal]
                            scores[typ][modal] = float(val)
                    add(f"reliability_{q_name}_wm{w_margin:g}_wh{w_hit:g}_wd{w_deg:g}", scores, alpha_grid)

    return candidates


def run_dataset(dataset, cfg, args):
    local_args = argparse.Namespace(**vars(args))
    local_args.data_choice = dataset
    local_args.checkpoint_name = cfg["checkpoint"]
    local_args.external_anchor_type_jsonl = cfg["type_jsonl"]

    model_args = build_model_args(local_args)
    logger = SimpleLogger()
    kgs, non_train, train_set, eval_set, test_set, test_ill = load_data(logger, model_args)
    model = SGMEA(kgs, model_args).cuda()
    checkpoint_path = load_checkpoint(model, local_args, model_args)
    model.eval()

    ent_to_type = build_type_maps(kgs)
    train_ill = np.asarray(train_set.data, dtype=np.int64)
    embs = get_modal_embeddings(model)
    stats = train_pair_modal_stats(embs, train_ill, ent_to_type, args.batch_size)
    degree = load_degree(model_args.data_path, dataset, args.data_split)
    deg_stats = degree_stats(train_ill, ent_to_type, degree)

    features, degree_norm, quality, severe_quality = extract_feature_scores(
        stats,
        deg_stats,
        cfg["visual_low_ratio"],
        cfg["visual_very_low_ratio"],
    )
    reference = load_reference(cfg["reference_json"], cfg["reference_key"])
    candidates = make_candidates(features, degree_norm, quality, severe_quality, args)
    for row in candidates:
        row["reference_compare"] = compare(row["scales"], reference)
    candidates.sort(key=lambda x: (x["reference_compare"]["mae"], -x["reference_compare"]["corr"]))

    return {
        "dataset": dataset,
        "checkpoint_name": cfg["checkpoint"],
        "checkpoint_path": checkpoint_path,
        "type_jsonl": cfg["type_jsonl"],
        "train_pair_count": int(train_ill.shape[0]),
        "visual_quality": {
            "low_ratio": cfg["visual_low_ratio"],
            "very_low_ratio": cfg["visual_very_low_ratio"],
            "quality": quality,
            "severe_quality": severe_quality,
        },
        "degree_norm": degree_norm,
        "degree_stats": deg_stats,
        "feature_stats": stats,
        "reference_prior": reference,
        "best_candidates": candidates[:30],
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    output = {
        "input": vars(args),
        "modal_order": MODAL_ORDER,
        "focus_types": FOCUS_TYPES,
        "method_note": (
            "All candidates are computed from train-set modality agreement/separability, "
            "node degree, and dataset-level visual quality statistics. Diagnostic priors "
            "are used only for offline comparison in this audit."
        ),
        "datasets": {},
    }
    for dataset in datasets:
        if dataset not in DEFAULT_DATASETS:
            raise ValueError(f"unknown dataset config: {dataset}")
        print(f"[run] dataset={dataset}", flush=True)
        output["datasets"][dataset] = run_dataset(dataset, DEFAULT_DATASETS[dataset], args)
        best = output["datasets"][dataset]["best_candidates"][0]
        print(f"[best] {dataset} {best['name']} mae={best['reference_compare']['mae']:.4f} corr={best['reference_compare']['corr']:.4f}", flush=True)
        print(json.dumps(best["scales"], indent=2, ensure_ascii=False), flush=True)

    with open(args.output_json, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)
    print(f"[done] output={args.output_json}", flush=True)


if __name__ == "__main__":
    main()
