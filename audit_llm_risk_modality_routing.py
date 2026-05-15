import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model import SGMEA
from src.data import load_data
from src.utils import pairwise_distances, csls_sim
from stage_a_projection_probe import SimpleLogger, build_model_args, load_checkpoint


MODAL_NAMES_FULL = [
    "img",
    "attr",
    "rel",
    "graph",
    "name",
    "char",
    "gat_img",
    "gat_attr",
    "gat_rel",
    "gat_name",
    "gat_char",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/gly/tongqiang/dongyufeng/data")
    parser.add_argument("--data_path", default="mmkg")
    parser.add_argument("--data_choice", default="FBDB15K")
    parser.add_argument("--data_split", default="norm")
    parser.add_argument("--data_rate", type=float, default=0.5)
    parser.add_argument("--checkpoint_name", default="SGMEA_FBDB15K_0.5_baseline_warm_softw_grid_0510_121128_")
    parser.add_argument("--labels_jsonls", required=True)
    parser.add_argument("--external_anchor_type_jsonl", default="/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csls", action="store_true", default=True)
    parser.add_argument("--no_csls", dest="csls", action="store_false")
    parser.add_argument("--csls_k", type=int, default=3)
    parser.add_argument("--risk_high", type=float, default=0.65)
    parser.add_argument("--risk_low", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def infer_active_modal_names(model):
    names = []
    with torch.no_grad():
        gph_emb = model.multimodal_encoder.cross_graph_model(model.multimodal_encoder.entity_emb(model.input_idx), model.adj) if model.args.w_gcn else None
        img_emb = model.multimodal_encoder.img_fc(model.img_features) if model.args.w_img else None
        gat_img_emb = None
        if img_emb is not None and not getattr(model.args, "disable_sgmea_guidance", False):
            gat_img_emb = model.multimodal_encoder.gat_img(img_emb, model.adj)
        rel_emb = model.multimodal_encoder.rel_fc(model.rel_features) if model.args.w_rel else None
        gat_rel_emb = None
        att_emb = model.multimodal_encoder.att_fc(model.att_features) if model.args.w_attr else None
        gat_att_emb = None
        if att_emb is not None and not getattr(model.args, "disable_sgmea_guidance", False):
            gat_att_emb = model.multimodal_encoder.gat_att(att_emb, model.adj)
        name_emb = None
        char_emb = None
        gat_name_emb = None
        gat_char_emb = None
        embs = [
            ("img", img_emb),
            ("attr", att_emb),
            ("rel", rel_emb),
            ("graph", gph_emb),
            ("name", name_emb),
            ("char", char_emb),
            ("gat_img", gat_img_emb),
            ("gat_attr", gat_att_emb),
            ("gat_rel", gat_rel_emb),
            ("gat_name", gat_name_emb),
            ("gat_char", gat_char_emb),
        ]
        names = [name for name, emb in embs if emb is not None]
    return names


def load_risk(paths):
    stats = defaultdict(lambda: {
        "close": 0,
        "safe": 0,
        "unknown": 0,
        "same_entity": 0,
        "sim_sum": 0.0,
        "rank_sum": 0.0,
        "row_count": 0,
        "coarse_type": Counter(),
        "directions": Counter(),
    })
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                anchor = int(row["anchor_id"])
                status = row.get("llm_relationship_status", "unknown")
                if status == "closely_related":
                    stats[anchor]["close"] += 1
                elif status == "safe_negative":
                    stats[anchor]["safe"] += 1
                elif status == "same_entity":
                    stats[anchor]["same_entity"] += 1
                else:
                    stats[anchor]["unknown"] += 1
                stats[anchor]["sim_sum"] += float(row.get("candidate_sim", row.get("candidate_distance", 0.0)) or 0.0)
                stats[anchor]["rank_sum"] += float(row.get("candidate_rank", 0.0) or 0.0)
                stats[anchor]["row_count"] += 1
                stats[anchor]["coarse_type"][row.get("coarse_type", "unknown")] += 1
                stats[anchor]["directions"][row.get("direction", "unknown")] += 1

    out = {}
    for anchor, item in stats.items():
        total = item["close"] + item["safe"]
        risk = item["close"] / total if total else 0.0
        dominant_type = item["coarse_type"].most_common(1)[0][0] if item["coarse_type"] else "unknown"
        out[anchor] = {
            "close": item["close"],
            "safe": item["safe"],
            "unknown": item["unknown"],
            "same_entity": item["same_entity"],
            "risk_score": risk,
            "row_count": item["row_count"],
            "mean_candidate_sim": item["sim_sum"] / max(item["row_count"], 1),
            "mean_candidate_rank": item["rank_sum"] / max(item["row_count"], 1),
            "coarse_type": dominant_type,
            "directions": dict(item["directions"]),
        }
    return out


def bucket_for(score, low, high):
    if score >= high:
        return "high"
    if score <= low:
        return "low"
    return "mid"


@torch.no_grad()
def compute_test_ranks(emb, test_pairs_np, use_csls=True, csls_k=3):
    left = torch.as_tensor(test_pairs_np[:, 0], dtype=torch.long, device=emb.device)
    right = torch.as_tensor(test_pairs_np[:, 1], dtype=torch.long, device=emb.device)
    distance = pairwise_distances(emb[left], emb[right])
    if use_csls:
        distance = 1 - csls_sim(1 - distance, csls_k)
    l2r = {}
    r2l = {}
    for idx in range(distance.shape[0]):
        indices = torch.argsort(distance[idx], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item() + 1
        l2r[int(left[idx].item())] = {
            "rank": rank,
            "is_hit1": rank == 1,
            "gt": int(right[idx].item()),
            "top1": int(right[indices[0]].item()),
        }
    for idx in range(distance.shape[1]):
        indices = torch.argsort(distance[:, idx], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item() + 1
        r2l[int(right[idx].item())] = {
            "rank": rank,
            "is_hit1": rank == 1,
            "gt": int(left[idx].item()),
            "top1": int(left[indices[0]].item()),
        }
    return l2r, r2l


def mean(values):
    return float(np.mean(values)) if values else 0.0


def summarize_entities(entity_ids, weights, modal_names, risk_info, rank_info=None):
    if not entity_ids:
        return {"count": 0}
    idx = torch.as_tensor(entity_ids, dtype=torch.long, device=weights.device)
    w = weights[idx].detach().cpu().numpy()
    out = {
        "count": len(entity_ids),
        "modal_weight_mean": {name: float(w[:, i].mean()) for i, name in enumerate(modal_names)},
        "modal_weight_std": {name: float(w[:, i].std()) for i, name in enumerate(modal_names)},
        "risk_score_mean": mean([risk_info[e]["risk_score"] for e in entity_ids if e in risk_info]),
        "close_mean": mean([risk_info[e]["close"] for e in entity_ids if e in risk_info]),
        "safe_mean": mean([risk_info[e]["safe"] for e in entity_ids if e in risk_info]),
    }
    if rank_info is not None:
        covered = [e for e in entity_ids if e in rank_info]
        ranks = [rank_info[e]["rank"] for e in covered]
        out.update({
            "test_covered_count": len(covered),
            "hits@1": mean([1.0 if rank_info[e]["is_hit1"] else 0.0 for e in covered]),
            "mrr": mean([1.0 / rank_info[e]["rank"] for e in covered]),
            "mean_rank": mean(ranks),
        })
    return out


def pearson(x, y):
    if len(x) < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = parse_args()
    args.no_csls = not args.csls
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    model_args = build_model_args(args)
    model_args.device = torch.device("cuda")
    logger = SimpleLogger()
    print("loading data/model...", flush=True)
    kgs, _, train_set, test_set, _, _ = load_data(logger, model_args)
    model = SGMEA(kgs, model_args).cuda().eval()
    checkpoint_path = load_checkpoint(model, args, model_args)
    for p in model.parameters():
        p.requires_grad = False

    modal_names = infer_active_modal_names(model)
    with torch.no_grad():
        joint_emb, weight_norm = model.joint_emb_generat()
        joint_emb = F.normalize(joint_emb, dim=1)
    if len(modal_names) != int(weight_norm.shape[1]):
        raise RuntimeError(f"modal_names={modal_names} but weight_norm shape={tuple(weight_norm.shape)}")
    print(f"modal_names={modal_names}", flush=True)
    print(f"weight_norm shape={tuple(weight_norm.shape)} joint_emb shape={tuple(joint_emb.shape)}", flush=True)

    label_paths = [p for p in args.labels_jsonls.split(",") if p]
    risk_info = load_risk(label_paths)
    test_np = np.asarray(test_set.data, dtype=np.int64)
    train_np = np.asarray(train_set.data, dtype=np.int64)
    train_l2r_rank, train_r2l_rank = compute_test_ranks(joint_emb, train_np, use_csls=False, csls_k=args.csls_k)
    l2r_rank, r2l_rank = compute_test_ranks(joint_emb, test_np, use_csls=args.csls, csls_k=args.csls_k)

    all_risk_entities = sorted(risk_info)
    by_bucket = defaultdict(list)
    by_type_bucket = defaultdict(lambda: defaultdict(list))
    for ent in all_risk_entities:
        bucket = bucket_for(risk_info[ent]["risk_score"], args.risk_low, args.risk_high)
        by_bucket[bucket].append(ent)
        by_type_bucket[risk_info[ent]["coarse_type"]][bucket].append(ent)

    summary = {
        "input": {
            "checkpoint_path": checkpoint_path,
            "labels_jsonls": label_paths,
            "risk_entity_count": len(all_risk_entities),
            "test_count": int(test_np.shape[0]),
            "risk_low": args.risk_low,
            "risk_high": args.risk_high,
        },
        "modal": {
            "modal_names": modal_names,
            "weight_norm_shape": list(weight_norm.shape),
            "joint_emb_shape": list(joint_emb.shape),
            "joint_segment_dim": int(joint_emb.shape[1] // weight_norm.shape[1]),
            "segment_order": modal_names,
            "global_weight_mean": summarize_entities(list(range(weight_norm.shape[0])), weight_norm, modal_names, risk_info)["modal_weight_mean"],
        },
        "risk_buckets_all_train_anchors": {
            bucket: summarize_entities(ids, weight_norm, modal_names, risk_info)
            for bucket, ids in sorted(by_bucket.items())
        },
        "risk_buckets_l2r_test_overlap": {
            bucket: summarize_entities(ids, weight_norm, modal_names, risk_info, l2r_rank)
            for bucket, ids in sorted(by_bucket.items())
        },
        "risk_buckets_r2l_test_overlap": {
            bucket: summarize_entities(ids, weight_norm, modal_names, risk_info, r2l_rank)
            for bucket, ids in sorted(by_bucket.items())
        },
        "risk_buckets_by_type": {
            typ: {
                bucket: summarize_entities(ids, weight_norm, modal_names, risk_info)
                for bucket, ids in sorted(bucket_map.items())
            }
            for typ, bucket_map in sorted(by_type_bucket.items())
        },
        "risk_buckets_l2r_train_overlap": {
            bucket: summarize_entities(ids, weight_norm, modal_names, risk_info, train_l2r_rank)
            for bucket, ids in sorted(by_bucket.items())
        },
        "risk_buckets_r2l_train_overlap": {
            bucket: summarize_entities(ids, weight_norm, modal_names, risk_info, train_r2l_rank)
            for bucket, ids in sorted(by_bucket.items())
        },
    }

    # Correlations over anchors with risk labels.
    weights_np = weight_norm.detach().cpu().numpy()
    risk_scores = [risk_info[e]["risk_score"] for e in all_risk_entities]
    correlations = {}
    for i, name in enumerate(modal_names):
        correlations[f"risk_vs_{name}_weight"] = pearson(risk_scores, [weights_np[e, i] for e in all_risk_entities])
    l2r_overlap = [e for e in all_risk_entities if e in l2r_rank]
    r2l_overlap = [e for e in all_risk_entities if e in r2l_rank]
    train_l2r_overlap = [e for e in all_risk_entities if e in train_l2r_rank]
    train_r2l_overlap = [e for e in all_risk_entities if e in train_r2l_rank]
    correlations["risk_vs_l2r_error"] = pearson(
        [risk_info[e]["risk_score"] for e in l2r_overlap],
        [0.0 if l2r_rank[e]["is_hit1"] else 1.0 for e in l2r_overlap],
    )
    correlations["risk_vs_r2l_error"] = pearson(
        [risk_info[e]["risk_score"] for e in r2l_overlap],
        [0.0 if r2l_rank[e]["is_hit1"] else 1.0 for e in r2l_overlap],
    )
    correlations["risk_vs_train_l2r_error"] = pearson(
        [risk_info[e]["risk_score"] for e in train_l2r_overlap],
        [0.0 if train_l2r_rank[e]["is_hit1"] else 1.0 for e in train_l2r_overlap],
    )
    correlations["risk_vs_train_r2l_error"] = pearson(
        [risk_info[e]["risk_score"] for e in train_r2l_overlap],
        [0.0 if train_r2l_rank[e]["is_hit1"] else 1.0 for e in train_r2l_overlap],
    )
    graph_candidates = [name for name in modal_names if "graph" in name or name in {"gat_attr", "gat_img"}]
    for name in graph_candidates:
        i = modal_names.index(name)
        correlations[f"{name}_weight_vs_l2r_error"] = pearson(
            [weights_np[e, i] for e in l2r_overlap],
            [0.0 if l2r_rank[e]["is_hit1"] else 1.0 for e in l2r_overlap],
        )
        correlations[f"{name}_weight_vs_r2l_error"] = pearson(
            [weights_np[e, i] for e in r2l_overlap],
            [0.0 if r2l_rank[e]["is_hit1"] else 1.0 for e in r2l_overlap],
        )
        correlations[f"{name}_weight_vs_train_l2r_error"] = pearson(
            [weights_np[e, i] for e in train_l2r_overlap],
            [0.0 if train_l2r_rank[e]["is_hit1"] else 1.0 for e in train_l2r_overlap],
        )
        correlations[f"{name}_weight_vs_train_r2l_error"] = pearson(
            [weights_np[e, i] for e in train_r2l_overlap],
            [0.0 if train_r2l_rank[e]["is_hit1"] else 1.0 for e in train_r2l_overlap],
        )
    summary["correlations"] = correlations

    # Compare wrong vs correct among risk-covered test anchors.
    def correct_wrong(rank_map):
        correct = [e for e in all_risk_entities if e in rank_map and rank_map[e]["is_hit1"]]
        wrong = [e for e in all_risk_entities if e in rank_map and not rank_map[e]["is_hit1"]]
        return {
            "correct": summarize_entities(correct, weight_norm, modal_names, risk_info, rank_map),
            "wrong": summarize_entities(wrong, weight_norm, modal_names, risk_info, rank_map),
        }

    summary["correct_vs_wrong_l2r_overlap"] = correct_wrong(l2r_rank)
    summary["correct_vs_wrong_r2l_overlap"] = correct_wrong(r2l_rank)
    summary["correct_vs_wrong_train_l2r_overlap"] = correct_wrong(train_l2r_rank)
    summary["correct_vs_wrong_train_r2l_overlap"] = correct_wrong(train_r2l_rank)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "modal": summary["modal"],
        "risk_buckets_l2r_test_overlap": summary["risk_buckets_l2r_test_overlap"],
        "risk_buckets_r2l_test_overlap": summary["risk_buckets_r2l_test_overlap"],
        "risk_buckets_l2r_train_overlap": summary["risk_buckets_l2r_train_overlap"],
        "risk_buckets_r2l_train_overlap": summary["risk_buckets_r2l_train_overlap"],
        "correlations": summary["correlations"],
        "correct_vs_wrong_l2r_overlap": summary["correct_vs_wrong_l2r_overlap"],
        "correct_vs_wrong_train_l2r_overlap": summary["correct_vs_wrong_train_l2r_overlap"],
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
