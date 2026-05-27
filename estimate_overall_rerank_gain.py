import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate overall FBDB rerank gain from one or more per-type rerank summaries."
    )
    parser.add_argument(
        "--rerank_summary_jsons",
        type=str,
        required=True,
        help="Comma-separated rerank summary JSON paths from rerank_fbdb_topk.py",
    )
    parser.add_argument("--overall_total_query_count", type=int, required=True)
    parser.add_argument("--overall_baseline_hits1", type=float, required=True)
    parser.add_argument("--overall_baseline_mrr", type=float, required=True)
    parser.add_argument("--overall_baseline_hits10", type=float, default=None)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    summary_paths = [x.strip() for x in args.rerank_summary_jsons.split(",") if x.strip()]
    if not summary_paths:
        raise ValueError("No rerank summary JSON paths were provided.")
    if args.overall_total_query_count <= 0:
        raise ValueError("--overall_total_query_count must be positive.")

    baseline_metrics = {
        "hits@1": args.overall_baseline_hits1,
        "mrr": args.overall_baseline_mrr,
    }
    if args.overall_baseline_hits10 is not None:
        baseline_metrics["hits@10"] = args.overall_baseline_hits10

    per_type = {}
    for path in summary_paths:
        payload = load_summary(path)
        baseline_by_type = payload.get("baseline", {}).get("by_type", {})
        reranked_by_type = payload.get("reranked", {}).get("by_type", {})
        common_types = sorted(set(baseline_by_type.keys()) & set(reranked_by_type.keys()))
        if not common_types:
            raise ValueError(f"No shared by-type entries found in summary: {path}")

        for type_name in common_types:
            if type_name in per_type:
                raise ValueError(f"Duplicate type summary detected for: {type_name}")
            base_item = baseline_by_type[type_name]
            rerank_item = reranked_by_type[type_name]
            count = int(rerank_item.get("count", base_item.get("count", 0)))
            if count <= 0:
                raise ValueError(f"Invalid count for type {type_name} in {path}")

            metric_payload = {}
            for metric_name in baseline_metrics.keys():
                if metric_name not in base_item or metric_name not in rerank_item:
                    raise ValueError(f"Metric {metric_name} missing for type {type_name} in {path}")
                base_value = float(base_item[metric_name])
                rerank_value = float(rerank_item[metric_name])
                delta = rerank_value - base_value
                contribution = delta * count / args.overall_total_query_count
                metric_payload[metric_name] = {
                    "baseline": round(base_value, 6),
                    "reranked": round(rerank_value, 6),
                    "delta": round(delta, 6),
                    "overall_contribution": round(contribution, 6),
                }

            per_type[type_name] = {
                "count": count,
                "summary_json": os.path.abspath(path),
                "metrics": metric_payload,
            }

    covered_queries = sum(item["count"] for item in per_type.values())
    estimated_overall = {}
    metric_gain_breakdown = {}
    for metric_name, baseline_value in baseline_metrics.items():
        total_contribution = sum(
            item["metrics"][metric_name]["overall_contribution"] for item in per_type.values()
        )
        estimated_overall[metric_name] = {
            "baseline": round(baseline_value, 6),
            "estimated_reranked": round(baseline_value + total_contribution, 6),
            "estimated_gain": round(total_contribution, 6),
        }
        metric_gain_breakdown[metric_name] = {
            type_name: item["metrics"][metric_name]["overall_contribution"]
            for type_name, item in sorted(per_type.items())
        }

    result = {
        "overall_total_query_count": args.overall_total_query_count,
        "covered_type_query_count": covered_queries,
        "covered_query_ratio": round(covered_queries / args.overall_total_query_count, 6),
        "estimated_overall_metrics": estimated_overall,
        "per_type": per_type,
        "metric_gain_breakdown": metric_gain_breakdown,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("===== Estimated Overall Rerank Gain =====", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
