import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from model import SGMEA
from src.data import load_data, read_raw_data
from src.utils import pairwise_distances, csls_sim
from torchlight import set_seed


class SimpleLogger:
    def info(self, msg):
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--output_summary_json", required=True)
    parser.add_argument("--degree_bucket_quantiles", default="0.25,0.75")
    parser.add_argument("--focus_coarse_types", default="Place,Creative Work,Organization")
    parser.add_argument("--margin_base", type=float, default=1.0)
    parser.add_argument("--margin_base_sweep", default="")
    audit_args, remaining = parser.parse_known_args()

    original_argv = os.sys.argv
    try:
        os.sys.argv = [os.sys.argv[0]] + remaining
        train_cfg = cfg()
        train_cfg.get_args()
        args = train_cfg.update_train_configs()
    finally:
        os.sys.argv = original_argv

    return audit_args, args


def load_cache_rows(path, keep_coarse_types):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if keep_coarse_types and row.get("coarse_type") not in keep_coarse_types:
                continue
            rows.append(row)
    return rows


def build_degree_map(file_dir):
    _, _, triples, _, _, _ = read_raw_data(file_dir, [1, 2])
    degree = Counter()
    for head, _, tail in triples:
        degree[int(head)] += 1
        degree[int(tail)] += 1
    return degree


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    w = idx - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def summarize_degree_distribution(covered_anchor_ids, train_anchor_ids, degree_map, quantile_points):
    train_anchor_degrees = sorted([int(degree_map.get(anchor_id, 0)) for anchor_id in train_anchor_ids])
    covered_degrees = [int(degree_map.get(anchor_id, 0)) for anchor_id in covered_anchor_ids]

    q_low = quantile(train_anchor_degrees, quantile_points[0]) if train_anchor_degrees else 0.0
    q_high = quantile(train_anchor_degrees, quantile_points[1]) if train_anchor_degrees else 0.0

    bucket_counts = {"low": 0, "medium": 0, "high": 0}
    for deg in covered_degrees:
        if deg <= q_low:
            bucket_counts["low"] += 1
        elif deg >= q_high:
            bucket_counts["high"] += 1
        else:
            bucket_counts["medium"] += 1

    if covered_degrees:
        degree_stats = {
            "min": int(min(covered_degrees)),
            "max": int(max(covered_degrees)),
            "mean": round(float(sum(covered_degrees) / len(covered_degrees)), 6),
            "median": round(float(np.median(covered_degrees)), 6),
        }
    else:
        degree_stats = {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

    return {
        "train_anchor_degree_quantiles": {
            "q_low": round(q_low, 6),
            "q_high": round(q_high, 6),
        },
        "covered_anchor_degree_stats": degree_stats,
        "covered_anchor_degree_buckets": bucket_counts,
    }


def build_distance_matrix(args, model, anchor_ids, opposite_ids):
    with torch.no_grad():
        final_emb, _ = model.joint_emb_generat()
        final_emb = F.normalize(final_emb)
        distance = pairwise_distances(final_emb[anchor_ids], final_emb[opposite_ids])
        if args.csls and min(distance.shape[0], distance.shape[1]) >= args.csls_k:
            distance = 1 - csls_sim(1 - distance, args.csls_k)
    return distance


def describe_values(values):
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "min": round(float(min(vals)), 6),
        "max": round(float(max(vals)), 6),
        "mean": round(float(sum(vals) / len(vals)), 6),
        "median": round(float(np.median(vals)), 6),
    }


def summarize_cache_subset(rows):
    if not rows:
        return {
            "row_count": 0,
            "status_counts": {},
            "direction_counts": {},
            "coarse_type_breakdown": {},
            "unique_anchor_count": 0,
        }
    status_counter = Counter(row["llm_relationship_status"] for row in rows)
    direction_counter = Counter(row["direction"] for row in rows)
    coarse_counter = defaultdict(Counter)
    unique_anchor_ids = set()
    for row in rows:
        coarse_counter[row["coarse_type"]][row["llm_relationship_status"]] += 1
        unique_anchor_ids.add((int(row["anchor_id"]), row["direction"]))
    return {
        "row_count": len(rows),
        "status_counts": dict(status_counter),
        "direction_counts": dict(direction_counter),
        "coarse_type_breakdown": {k: dict(v) for k, v in coarse_counter.items()},
        "unique_anchor_count": len(unique_anchor_ids),
    }


def build_filter_simulation(cache_rows):
    scenarios = {
        "all": cache_rows,
        "high_quality": [row for row in cache_rows if row.get("evidence_quality") == "high"],
        "conf_ge_0.8": [row for row in cache_rows if float(row.get("llm_confidence", 0.0)) >= 0.8],
        "conf_ge_0.9": [row for row in cache_rows if float(row.get("llm_confidence", 0.0)) >= 0.9],
    }
    return {name: summarize_cache_subset(rows) for name, rows in scenarios.items()}


def compute_gt_rank_profile(cache_rows, train_links, left_entity_ids, right_entity_ids, args, model, margin_base):
    left_set = set(int(x) for x in left_entity_ids.detach().cpu().tolist())
    right_set = set(int(x) for x in right_entity_ids.detach().cpu().tolist())
    with torch.no_grad():
        final_emb, _ = model.joint_emb_generat()
        final_emb = F.normalize(final_emb)

    evaluated = []
    for row_meta in cache_rows:
        anchor_id = int(row_meta["anchor_id"])
        positive_id = int(row_meta["positive_id"])
        direction = str(row_meta["direction"])
        if direction == "left_to_right":
            if anchor_id not in left_set or positive_id not in right_set:
                continue
            anchor_tensor = torch.tensor([anchor_id], dtype=torch.long).cuda()
            opposite_tensor = right_entity_ids
            gt_local = (right_entity_ids == positive_id).nonzero(as_tuple=False)
        else:
            if anchor_id not in right_set or positive_id not in left_set:
                continue
            anchor_tensor = torch.tensor([anchor_id], dtype=torch.long).cuda()
            opposite_tensor = left_entity_ids
            gt_local = (left_entity_ids == positive_id).nonzero(as_tuple=False)

        if gt_local.numel() == 0:
            continue
        gt_local_idx = int(gt_local.flatten()[0].item())
        distance = pairwise_distances(final_emb[anchor_tensor], final_emb[opposite_tensor])
        if args.csls and min(distance.shape[0], distance.shape[1]) >= args.csls_k:
            distance = 1 - csls_sim(1 - distance, args.csls_k)
        _, indices = torch.sort(distance[0], descending=False)
        rank = int((indices == gt_local_idx).nonzero(as_tuple=False).flatten()[0].item()) + 1
        candidate_id = int(row_meta["candidate_id"])
        if direction == "left_to_right":
            candidate_local = (right_entity_ids == candidate_id).nonzero(as_tuple=False)
        else:
            candidate_local = (left_entity_ids == candidate_id).nonzero(as_tuple=False)
        if candidate_local.numel() == 0:
            continue
        candidate_local_idx = int(candidate_local.flatten()[0].item())
        positive_distance = float(distance[0, gt_local_idx].item())
        candidate_distance = float(distance[0, candidate_local_idx].item())
        margin_delta = candidate_distance - positive_distance
        margin_violated = int(margin_delta < float(margin_base))
        evaluated.append(
            {
                "direction": direction,
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "gt_rank": rank,
                "top1_correct": int(rank == 1),
                "candidate_id": candidate_id,
                "positive_distance": positive_distance,
                "candidate_distance": candidate_distance,
                "margin_delta": margin_delta,
                "margin_violated": margin_violated,
                "candidate_rank": int(row_meta.get("candidate_rank", 1)),
                "coarse_type": row_meta["coarse_type"],
                "status": row_meta["llm_relationship_status"],
            }
        )

    if not evaluated:
        return {
            "count": 0,
            "top1_error_count": 0,
            "top1_error_rate": 0.0,
            "mean_gt_rank": 0.0,
            "median_gt_rank": 0.0,
            "rank_buckets": {},
            "direction_breakdown": {},
        }

    gt_ranks = [row["gt_rank"] for row in evaluated]
    top1_errors = sum(1 for row in evaluated if row["top1_correct"] == 0)
    margin_deltas = [row["margin_delta"] for row in evaluated]
    margin_violation_count = sum(1 for row in evaluated if row["margin_violated"] == 1)
    rank_buckets = {
        "rank_1": sum(1 for rank in gt_ranks if rank == 1),
        "rank_2_5": sum(1 for rank in gt_ranks if 2 <= rank <= 5),
        "rank_6_10": sum(1 for rank in gt_ranks if 6 <= rank <= 10),
        "rank_gt_10": sum(1 for rank in gt_ranks if rank > 10),
    }
    direction_breakdown = defaultdict(lambda: {"count": 0, "top1_error_count": 0, "mean_gt_rank": 0.0})
    coarse_breakdown = defaultdict(lambda: {"count": 0, "top1_error_count": 0, "margin_violation_count": 0, "mean_gt_rank": 0.0, "mean_margin_delta": 0.0})
    status_breakdown = defaultdict(lambda: {"count": 0, "top1_error_count": 0, "margin_violation_count": 0, "mean_gt_rank": 0.0, "mean_margin_delta": 0.0})
    for row in evaluated:
        bucket = direction_breakdown[row["direction"]]
        bucket["count"] += 1
        bucket["top1_error_count"] += int(row["top1_correct"] == 0)
        bucket["mean_gt_rank"] += row["gt_rank"]
        type_bucket = coarse_breakdown[row["coarse_type"]]
        type_bucket["count"] += 1
        type_bucket["top1_error_count"] += int(row["top1_correct"] == 0)
        type_bucket["margin_violation_count"] += int(row["margin_violated"])
        type_bucket["mean_gt_rank"] += row["gt_rank"]
        type_bucket["mean_margin_delta"] += row["margin_delta"]
        status_bucket = status_breakdown[row["status"]]
        status_bucket["count"] += 1
        status_bucket["top1_error_count"] += int(row["top1_correct"] == 0)
        status_bucket["margin_violation_count"] += int(row["margin_violated"])
        status_bucket["mean_gt_rank"] += row["gt_rank"]
        status_bucket["mean_margin_delta"] += row["margin_delta"]
    for direction, item in direction_breakdown.items():
        item["top1_error_rate"] = round(item["top1_error_count"] / item["count"], 6) if item["count"] > 0 else 0.0
        item["mean_gt_rank"] = round(item["mean_gt_rank"] / item["count"], 6) if item["count"] > 0 else 0.0
    for bucket_dict in [coarse_breakdown, status_breakdown]:
        for _, item in bucket_dict.items():
            item["top1_error_rate"] = round(item["top1_error_count"] / item["count"], 6) if item["count"] > 0 else 0.0
            item["margin_violation_rate"] = round(item["margin_violation_count"] / item["count"], 6) if item["count"] > 0 else 0.0
            item["mean_gt_rank"] = round(item["mean_gt_rank"] / item["count"], 6) if item["count"] > 0 else 0.0
            item["mean_margin_delta"] = round(item["mean_margin_delta"] / item["count"], 6) if item["count"] > 0 else 0.0

    recoverable_anchor_ids = {
        (row["anchor_id"], row["direction"])
        for row in evaluated
        if row["gt_rank"] > 1 or row["margin_violated"] == 1
    }
    rank1_margin_violated = sum(1 for row in evaluated if row["gt_rank"] == 1 and row["margin_violated"] == 1)

    return {
        "count": len(evaluated),
        "top1_error_count": top1_errors,
        "top1_error_rate": round(top1_errors / len(evaluated), 6),
        "mean_gt_rank": round(float(sum(gt_ranks) / len(gt_ranks)), 6),
        "median_gt_rank": round(float(np.median(gt_ranks)), 6),
        "positive_distance_stats": describe_values([row["positive_distance"] for row in evaluated]),
        "candidate_distance_stats": describe_values([row["candidate_distance"] for row in evaluated]),
        "margin_delta_stats": describe_values(margin_deltas),
        "margin_base": float(margin_base),
        "margin_violation_count": margin_violation_count,
        "margin_violation_rate": round(margin_violation_count / len(evaluated), 6),
        "rank1_but_margin_violated_count": rank1_margin_violated,
        "recoverable_anchor_count": len(recoverable_anchor_ids),
        "recoverable_anchor_rate": round(len(recoverable_anchor_ids) / len(evaluated), 6),
        "rank_buckets": rank_buckets,
        "direction_breakdown": {k: dict(v) for k, v in direction_breakdown.items()},
        "coarse_type_breakdown": {k: dict(v) for k, v in coarse_breakdown.items()},
        "status_breakdown": {k: dict(v) for k, v in status_breakdown.items()},
    }


def build_margin_sweep_profiles(cache_rows, train_links, left_entity_ids, right_entity_ids, args, model, margin_values):
    if not margin_values:
        return {}
    results = {}
    for margin_base in margin_values:
        key = f"{float(margin_base):.4f}".rstrip("0").rstrip(".")
        results[key] = compute_gt_rank_profile(
            cache_rows=cache_rows,
            train_links=train_links,
            left_entity_ids=left_entity_ids,
            right_entity_ids=right_entity_ids,
            args=args,
            model=model,
            margin_base=float(margin_base),
        )
    return results


def main():
    audit_args, args = parse_args()
    logger = SimpleLogger()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because SGMEA builds CUDA tensors internally.")

    set_seed(args.random_seed)
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)
    focus_types = {x.strip() for x in audit_args.focus_coarse_types.split(",") if x.strip()}
    margin_values = []
    if audit_args.margin_base_sweep.strip():
        margin_values = [float(x.strip()) for x in audit_args.margin_base_sweep.split(",") if x.strip()]

    cache_rows = load_cache_rows(audit_args.cache_jsonl, focus_types)
    cache_status_counter = Counter(row["llm_relationship_status"] for row in cache_rows)
    direction_counter = Counter(row["direction"] for row in cache_rows)
    coarse_counter = defaultdict(Counter)
    anchor_to_rows = defaultdict(list)
    for row in cache_rows:
        coarse_counter[row["coarse_type"]][row["llm_relationship_status"]] += 1
        anchor_to_rows[(int(row["anchor_id"]), row["direction"])].append(row)

    kgs, non_train, train_set, test_set, eval_set, test_ill_ = load_data(logger, args)
    model = SGMEA(kgs, args).cuda()
    save_path = os.path.join(args.data_path, args.model_name, "save", f"{args.model_name_save}.pkl")
    state = torch.load(save_path, map_location=args.device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()

    train_links = train_set.data if hasattr(train_set, "data") else train_set
    train_left_ids = [int(row[0]) for row in train_links]
    train_right_ids = [int(row[1]) for row in train_links]
    train_anchor_ids = set(train_left_ids + train_right_ids)
    covered_anchor_ids = {anchor_id for anchor_id, _ in anchor_to_rows.keys()}
    entity_type_ids = model.entity_type_ids.detach().cpu().tolist()
    type_names = list(model.top_type_names)
    train_anchor_type_counter = Counter()
    for anchor_id in train_anchor_ids:
        type_id = int(entity_type_ids[anchor_id])
        type_name = type_names[type_id] if 0 <= type_id < len(type_names) else str(type_id)
        train_anchor_type_counter[type_name] += 1

    left_entity_ids = kgs["left_entity_ids"].cuda()
    right_entity_ids = kgs["right_entity_ids"].cuda()

    file_dir = os.path.join(args.data_path, args.data_choice, args.data_split)
    degree_map = build_degree_map(file_dir)
    quantiles = [float(x.strip()) for x in audit_args.degree_bucket_quantiles.split(",") if x.strip()]
    if len(quantiles) != 2:
        raise ValueError("--degree_bucket_quantiles must provide exactly two values")

    degree_summary = summarize_degree_distribution(covered_anchor_ids, train_anchor_ids, degree_map, quantiles)

    per_anchor_counts = [len(rows) for rows in anchor_to_rows.values()]
    per_anchor_close = [sum(1 for row in rows if row["llm_relationship_status"] == "closely_related") for rows in anchor_to_rows.values()]
    per_anchor_safe = [sum(1 for row in rows if row["llm_relationship_status"] == "safe_negative") for rows in anchor_to_rows.values()]

    coverage = {
        "train_anchor_count": len(train_anchor_ids),
        "covered_anchor_count": len(covered_anchor_ids),
        "covered_anchor_rate": round(len(covered_anchor_ids) / len(train_anchor_ids), 6) if train_anchor_ids else 0.0,
        "cache_row_count": len(cache_rows),
        "cache_status_counts": dict(cache_status_counter),
        "cache_direction_counts": dict(direction_counter),
        "cache_coarse_type_breakdown": {k: dict(v) for k, v in coarse_counter.items()},
        "per_anchor_relation_row_stats": {
            "min": int(min(per_anchor_counts)) if per_anchor_counts else 0,
            "max": int(max(per_anchor_counts)) if per_anchor_counts else 0,
            "mean": round(float(sum(per_anchor_counts) / len(per_anchor_counts)), 6) if per_anchor_counts else 0.0,
            "median": round(float(np.median(per_anchor_counts)), 6) if per_anchor_counts else 0.0,
        },
        "per_anchor_close_stats": {
            "min": int(min(per_anchor_close)) if per_anchor_close else 0,
            "max": int(max(per_anchor_close)) if per_anchor_close else 0,
            "mean": round(float(sum(per_anchor_close) / len(per_anchor_close)), 6) if per_anchor_close else 0.0,
            "median": round(float(np.median(per_anchor_close)), 6) if per_anchor_close else 0.0,
        },
        "per_anchor_safe_stats": {
            "min": int(min(per_anchor_safe)) if per_anchor_safe else 0,
            "max": int(max(per_anchor_safe)) if per_anchor_safe else 0,
            "mean": round(float(sum(per_anchor_safe) / len(per_anchor_safe)), 6) if per_anchor_safe else 0.0,
            "median": round(float(np.median(per_anchor_safe)), 6) if per_anchor_safe else 0.0,
        },
    }

    directional_coverage = defaultdict(set)
    coarse_anchor_coverage = defaultdict(set)
    for row in cache_rows:
        directional_coverage[row["direction"]].add(int(row["anchor_id"]))
        coarse_anchor_coverage[row["coarse_type"]].add(int(row["anchor_id"]))

    coverage["directional_anchor_coverage"] = {
        direction: {
            "covered_anchor_count": len(anchor_ids),
        }
        for direction, anchor_ids in directional_coverage.items()
    }
    coverage["coarse_type_anchor_coverage"] = {
        coarse_type: {
            "train_anchor_count": int(train_anchor_type_counter.get(coarse_type, 0)),
            "covered_anchor_count": len(anchor_ids),
            "covered_anchor_rate": round(len(anchor_ids) / train_anchor_type_counter.get(coarse_type, 1), 6) if train_anchor_type_counter.get(coarse_type, 0) > 0 else 0.0,
        }
        for coarse_type, anchor_ids in coarse_anchor_coverage.items()
    }
    coverage["train_anchor_count_by_type"] = dict(train_anchor_type_counter)

    num_batches = int(math.ceil(len(train_links) / float(args.batch_size))) if len(train_links) > 0 else 0
    covered_pair_rows = 0
    covered_pair_rows_ltr = 0
    covered_pair_rows_rtl = 0
    directional_anchor_only = {
        direction: {int(anchor_id) for anchor_id in anchor_ids}
        for direction, anchor_ids in directional_coverage.items()
    }
    for direction in ["left_to_right", "right_to_left"]:
        directional_anchor_only.setdefault(direction, set())
    for left_id, right_id in train_links:
        left_id = int(left_id)
        right_id = int(right_id)
        hit_ltr = left_id in directional_anchor_only["left_to_right"]
        hit_rtl = right_id in directional_anchor_only["right_to_left"]
        if hit_ltr or hit_rtl:
            covered_pair_rows += 1
        if hit_ltr:
            covered_pair_rows_ltr += 1
        if hit_rtl:
            covered_pair_rows_rtl += 1
    coverage["exposure_estimate"] = {
        "train_pair_row_count": int(len(train_links)),
        "estimated_batches_per_epoch": num_batches,
        "covered_pair_rows_per_epoch": int(covered_pair_rows),
        "covered_pair_row_rate": round(covered_pair_rows / len(train_links), 6) if len(train_links) > 0 else 0.0,
        "covered_pair_rows_ltr_per_epoch": int(covered_pair_rows_ltr),
        "covered_pair_rows_rtl_per_epoch": int(covered_pair_rows_rtl),
        "estimated_relation_rows_per_batch": round(len(cache_rows) / num_batches, 6) if num_batches > 0 else 0.0,
        "estimated_hit_pair_rows_per_batch": round(covered_pair_rows / num_batches, 6) if num_batches > 0 else 0.0,
    }

    filter_simulation = build_filter_simulation(cache_rows)

    gt_rank_profile = compute_gt_rank_profile(
        cache_rows=cache_rows,
        train_links=train_links,
        left_entity_ids=left_entity_ids,
        right_entity_ids=right_entity_ids,
        args=args,
        model=model,
        margin_base=audit_args.margin_base,
    )
    margin_sweep = build_margin_sweep_profiles(
        cache_rows=cache_rows,
        train_links=train_links,
        left_entity_ids=left_entity_ids,
        right_entity_ids=right_entity_ids,
        args=args,
        model=model,
        margin_values=margin_values,
    )

    summary = {
        "cache_jsonl": audit_args.cache_jsonl,
        "focus_coarse_types": sorted(focus_types),
        "coverage": coverage,
        "degree_summary": degree_summary,
        "filter_simulation": filter_simulation,
        "gt_rank_profile": gt_rank_profile,
        "margin_base_sweep": margin_sweep,
        "interpretation_note": "This audit estimates whether the relation-aware cache is large enough, balanced enough, and hard enough to justify a first warm-start relation-aware fine-tuning run.",
    }

    os.makedirs(os.path.dirname(audit_args.output_summary_json), exist_ok=True)
    with open(audit_args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===== Relation-Aware HNM Pretrain Audit =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
