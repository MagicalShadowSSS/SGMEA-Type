import argparse
import json
import math
from collections import Counter, defaultdict


def load_groups(path):
    groups = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            groups[row["query_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: (float(r.get("candidate_distance", math.inf)), int(r.get("candidate_rank", 10**9))))
    return groups


def reciprocal_rank(rows):
    for idx, row in enumerate(rows, 1):
        if row.get("is_gt"):
            return 1.0 / idx
    return 0.0


def top_hit(rows, k=1):
    return any(row.get("is_gt") for row in rows[:k])


def row_score(row, penalties):
    status = row.get("llm_relationship_status", "unknown")
    return float(row.get("candidate_distance", math.inf)) + penalties.get(status, 0.0)


def rerank_rows(rows, penalties):
    return sorted(
        rows,
        key=lambda r: (
            row_score(r, penalties),
            int(r.get("candidate_rank", 10**9)),
            int(r.get("candidate_id", 10**9)),
        ),
    )


def promote_gt_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            0 if r.get("is_gt") else 1,
            float(r.get("candidate_distance", math.inf)),
            int(r.get("candidate_rank", 10**9)),
        ),
    )


def metric_bundle(group_rows, total_denominator=None):
    n = len(group_rows)
    denom = total_denominator or n
    hits1 = sum(1 for rows in group_rows.values() if top_hit(rows, 1))
    hits10 = sum(1 for rows in group_rows.values() if top_hit(rows, 10))
    mrr = sum(reciprocal_rank(rows) for rows in group_rows.values())
    return {
        "labeled_group_count": n,
        "denominator": denom,
        "hits@1": hits1 / denom if denom else 0.0,
        "hits@10": hits10 / denom if denom else 0.0,
        "mrr": mrr / denom if denom else 0.0,
        "hits@1_count": hits1,
        "hits@10_count": hits10,
    }


def evaluate_variant(groups, make_rows, pco_total, full_total):
    ranked = {qid: make_rows(rows) for qid, rows in groups.items()}
    by_type_groups = defaultdict(dict)
    for qid, rows in ranked.items():
        by_type_groups[rows[0].get("query_coarse_type", "unknown")][qid] = rows

    out = {
        "labeled": metric_bundle(ranked),
        "pco_denominator": metric_bundle(ranked, pco_total),
        "full_test_denominator": metric_bundle(ranked, full_total),
        "by_type_labeled": {typ: metric_bundle(sub) for typ, sub in sorted(by_type_groups.items())},
    }
    return out, ranked


def compare_rankings(baseline, variant):
    rescue = kill = stable_correct = stable_wrong = 0
    rescue_by_type = Counter()
    kill_by_type = Counter()
    top1_status_when_wrong = Counter()
    for qid, base_rows in baseline.items():
        var_rows = variant[qid]
        typ = base_rows[0].get("query_coarse_type", "unknown")
        b = top_hit(base_rows, 1)
        v = top_hit(var_rows, 1)
        if (not b) and v:
            rescue += 1
            rescue_by_type[typ] += 1
        elif b and (not v):
            kill += 1
            kill_by_type[typ] += 1
        elif b and v:
            stable_correct += 1
        else:
            stable_wrong += 1
            top1_status_when_wrong[var_rows[0].get("llm_relationship_status", "unknown")] += 1
    return {
        "rescue": rescue,
        "kill": kill,
        "net": rescue - kill,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "rescue_by_type": dict(rescue_by_type),
        "kill_by_type": dict(kill_by_type),
        "variant_top1_status_when_still_wrong": dict(top1_status_when_wrong),
    }


def sweep_negative_only(groups, safe_grid, close_grid):
    best = None
    for safe in safe_grid:
        for close in close_grid:
            penalties = {
                "safe_negative": safe,
                "closely_related": close,
                "same_entity": 0.0,
                "unknown": 0.0,
            }
            ranked = {qid: rerank_rows(rows, penalties) for qid, rows in groups.items()}
            m = metric_bundle(ranked)
            item = {
                "safe_penalty": safe,
                "close_penalty": close,
                "hits@1": m["hits@1"],
                "mrr": m["mrr"],
                "hits@1_count": m["hits@1_count"],
            }
            if best is None or (item["hits@1"], item["mrr"]) > (best["hits@1"], best["mrr"]):
                best = item
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--pco_total", type=int, default=3479)
    parser.add_argument("--full_total", type=int, default=6423)
    parser.add_argument("--max_penalty", type=float, default=2.0)
    args = parser.parse_args()

    groups = load_groups(args.labels_jsonl)
    label_counts = Counter()
    gt_label_counts = Counter()
    by_type_count = Counter()
    row_count = 0
    gt_in_topk = 0
    for rows in groups.values():
        by_type_count[rows[0].get("query_coarse_type", "unknown")] += 1
        has_gt = False
        for row in rows:
            row_count += 1
            status = row.get("llm_relationship_status", "unknown")
            label_counts[status] += 1
            if row.get("is_gt"):
                has_gt = True
                gt_label_counts[status] += 1
        if has_gt:
            gt_in_topk += 1

    variants = {}
    ranked_by_variant = {}

    variant_specs = {
        "baseline_distance": lambda rows: list(rows),
        "top10_gt_oracle": promote_gt_rows,
        "same_entity_only_oracle": lambda rows: rerank_rows(
            rows,
            {"same_entity": -args.max_penalty, "closely_related": 0.0, "safe_negative": 0.0, "unknown": 0.0},
        ),
        "safe_penalty_only_no_same": lambda rows: rerank_rows(
            rows,
            {"same_entity": 0.0, "closely_related": 0.0, "safe_negative": args.max_penalty, "unknown": 0.0},
        ),
        "close_penalty_only_no_same": lambda rows: rerank_rows(
            rows,
            {"same_entity": 0.0, "closely_related": args.max_penalty, "safe_negative": 0.0, "unknown": 0.0},
        ),
        "close_protect_safe_penalty_no_same": lambda rows: rerank_rows(
            rows,
            {"same_entity": 0.0, "closely_related": 0.0, "safe_negative": args.max_penalty, "unknown": 0.0},
        ),
        "status_full_oracle": lambda rows: rerank_rows(
            rows,
            {"same_entity": -args.max_penalty, "closely_related": 0.0, "safe_negative": args.max_penalty, "unknown": 0.0},
        ),
    }

    for name, fn in variant_specs.items():
        metrics, ranked = evaluate_variant(groups, fn, args.pco_total, args.full_total)
        variants[name] = metrics
        ranked_by_variant[name] = ranked

    baseline = ranked_by_variant["baseline_distance"]
    comparisons = {
        name: compare_rankings(baseline, ranked)
        for name, ranked in ranked_by_variant.items()
        if name != "baseline_distance"
    }

    sweep_grid = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0]
    best_negative_only = sweep_negative_only(groups, safe_grid=sweep_grid, close_grid=sweep_grid)
    best_negative_penalties = {
        "same_entity": 0.0,
        "safe_negative": best_negative_only["safe_penalty"],
        "closely_related": best_negative_only["close_penalty"],
        "unknown": 0.0,
    }
    best_negative_metrics, best_negative_ranked = evaluate_variant(
        groups,
        lambda rows: rerank_rows(rows, best_negative_penalties),
        args.pco_total,
        args.full_total,
    )
    variants["best_negative_label_only_sweep_no_same"] = best_negative_metrics
    comparisons["best_negative_label_only_sweep_no_same"] = compare_rankings(baseline, best_negative_ranked)
    best_negative_only.update({"penalties": best_negative_penalties})

    top1_status = Counter()
    wrong_top1_status = Counter()
    for rows in baseline.values():
        status = rows[0].get("llm_relationship_status", "unknown")
        top1_status[status] += 1
        if not rows[0].get("is_gt"):
            wrong_top1_status[status] += 1

    out = {
        "input": {
            "labels_jsonl": args.labels_jsonl,
            "row_count": row_count,
            "labeled_group_count": len(groups),
            "gt_in_topk_count": gt_in_topk,
            "pco_total_denominator": args.pco_total,
            "full_test_denominator": args.full_total,
            "by_type_group_count": dict(by_type_count),
            "label_counts": dict(label_counts),
            "gt_label_counts": dict(gt_label_counts),
            "baseline_top1_status_counts": dict(top1_status),
            "baseline_wrong_top1_status_counts": dict(wrong_top1_status),
        },
        "variants": variants,
        "comparisons_vs_baseline": comparisons,
        "best_negative_label_only_sweep_no_same": best_negative_only,
        "interpretation_flags": {
            "same_entity_gt_fraction": gt_label_counts.get("same_entity", 0) / gt_in_topk if gt_in_topk else 0.0,
            "top10_oracle_remaining_gap_pco_hits1": 1.0 - variants["top10_gt_oracle"]["pco_denominator"]["hits@1"],
            "negative_only_pco_hits1_gain": (
                variants["best_negative_label_only_sweep_no_same"]["pco_denominator"]["hits@1"]
                - variants["baseline_distance"]["pco_denominator"]["hits@1"]
            ),
            "same_entity_only_pco_hits1_gain": (
                variants["same_entity_only_oracle"]["pco_denominator"]["hits@1"]
                - variants["baseline_distance"]["pco_denominator"]["hits@1"]
            ),
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["interpretation_flags"], ensure_ascii=False, indent=2))
    print(json.dumps({k: v["pco_denominator"] for k, v in variants.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
