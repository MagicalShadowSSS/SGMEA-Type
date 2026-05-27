import argparse
import json
import os
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank FBDB top-k candidates using LLM semantic labels.")
    parser.add_argument("--pairs_jsonl", required=True, help="Top-k candidate pairs from extract_test_topk_pairs.py")
    parser.add_argument("--llm_jsonl", required=True, help="LLM labels on test top-k pairs")
    parser.add_argument("--output_reranked_jsonl", required=True, help="Per-query reranked candidates output")
    parser.add_argument("--output_summary_json", required=True, help="Summary metrics output")
    parser.add_argument("--topk_eval", type=str, default="1,10", help="Comma-separated metrics, e.g. 1,10")
    parser.add_argument("--same_entity_bonus", type=float, default=0.05)
    parser.add_argument("--closely_related_penalty", type=float, default=0.0)
    parser.add_argument("--safe_negative_penalty", type=float, default=-0.10)
    parser.add_argument("--unknown_penalty", type=float, default=0.0)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--sweep", action="store_true", default=False)
    parser.add_argument("--sweep_same_entity_bonus", type=str, default="0.02,0.05,0.10")
    parser.add_argument("--sweep_safe_negative_penalty", type=str, default="-0.03,-0.05,-0.08,-0.10")
    parser.add_argument("--sweep_closely_related_penalty", type=str, default="0.0,-0.01")
    parser.add_argument("--sweep_lambda_weight", type=str, default="0.5,1.0,1.5")
    parser.add_argument(
        "--total_query_count_by_type",
        type=str,
        default="",
        help='Optional full denominator, e.g. "Place:1343,Creative Work:1131". Missing queries are counted as failures.',
    )
    return parser.parse_args()


def load_pairs(path):
    queries = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["dataset"], row["query_id"])
            queries[key].append(row)
    return queries


def load_llm(path):
    llm = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row.get("dataset"), row.get("query_id", row.get("anchor_id")), row.get("candidate_id", row.get("negative_id")))
            llm[key] = {
                "status": row.get("llm_relationship_status", "unknown"),
                "kind": row.get("llm_relationship_kind", "unknown"),
                "confidence": float(row.get("llm_confidence", 0.0)),
            }
    return llm


def delta_for_status(status, params):
    if status == "same_entity":
        return params["same_entity_bonus"]
    if status == "closely_related":
        return params["closely_related_penalty"]
    if status == "safe_negative":
        return params["safe_negative_penalty"]
    return params["unknown_penalty"]


def compute_metrics(queries, topk_eval, total_query_count_by_type=None):
    results = {
        "count": 0,
        "hits": {k: 0 for k in topk_eval},
        "mrr": 0.0,
    }
    per_type = defaultdict(lambda: {"count": 0, "hits": {k: 0 for k in topk_eval}, "mrr": 0.0})

    for _, candidates in queries.items():
        if not candidates:
            continue
        ordered = candidates
        rank = None
        for idx, item in enumerate(ordered, start=1):
            if item.get("is_gt", False):
                rank = idx
                break
        if rank is None:
            continue

        query_type = ordered[0].get("query_coarse_type", "unknown")
        results["count"] += 1
        results["mrr"] += 1.0 / rank
        per_type[query_type]["count"] += 1
        per_type[query_type]["mrr"] += 1.0 / rank
        for k in topk_eval:
            if rank <= k:
                results["hits"][k] += 1
                per_type[query_type]["hits"][k] += 1

    if total_query_count_by_type:
        adjusted_overall_count = 0
        for query_type, total_count in total_query_count_by_type.items():
            if query_type not in per_type:
                per_type[query_type]
            per_type[query_type]["reported_count"] = per_type[query_type]["count"]
            per_type[query_type]["count"] = int(total_count)
            adjusted_overall_count += int(total_count)
        results["reported_count"] = results["count"]
        results["count"] = adjusted_overall_count

    summary = {
        "overall": {
            "count": results["count"],
            "reported_count": results.get("reported_count", results["count"]),
            **{f"hits@{k}": round(results["hits"][k] / results["count"], 4) if results["count"] else 0.0 for k in topk_eval},
            "mrr": round(results["mrr"] / results["count"], 4) if results["count"] else 0.0,
        },
        "by_type": {},
    }
    for t, stats in per_type.items():
        summary["by_type"][t] = {
            "count": stats["count"],
            "reported_count": stats.get("reported_count", stats["count"]),
            **{f"hits@{k}": round(stats["hits"][k] / stats["count"], 4) if stats["count"] else 0.0 for k in topk_eval},
            "mrr": round(stats["mrr"] / stats["count"], 4) if stats["count"] else 0.0,
        }
    return summary


def rerank_once(queries, llm, topk_eval, params, total_query_count_by_type=None):
    original_queries = {}
    reranked_queries = {}
    label_counter = Counter()

    for key, candidates in queries.items():
        original_sorted = sorted(candidates, key=lambda item: item["candidate_distance"])
        original_queries[key] = original_sorted

        reranked = []
        for item in original_sorted:
            llm_key = (item["dataset"], item["query_id"], item["candidate_id"])
            tag = llm.get(llm_key, {"status": "unknown", "kind": "missing", "confidence": 0.0})
            delta = delta_for_status(tag["status"], params)
            # smaller distance is better; positive delta should reduce distance
            final_distance = item["candidate_distance"] - params["lambda_weight"] * tag["confidence"] * delta
            row = dict(item)
            row.update(
                {
                    "llm_relationship_status": tag["status"],
                    "llm_relationship_kind": tag["kind"],
                    "llm_confidence": tag["confidence"],
                    "rerank_delta": params["lambda_weight"] * tag["confidence"] * delta,
                    "final_distance": final_distance,
                }
            )
            reranked.append(row)
            label_counter[tag["status"]] += 1

        reranked.sort(key=lambda item: item["final_distance"])
        reranked_queries[key] = reranked

    baseline_summary = compute_metrics(original_queries, topk_eval, total_query_count_by_type)
    rerank_summary = compute_metrics(reranked_queries, topk_eval, total_query_count_by_type)
    return original_queries, reranked_queries, label_counter, baseline_summary, rerank_summary


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_count_by_type(text):
    if not text.strip():
        return {}
    result = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --total_query_count_by_type part: {part}")
        key, value = part.split(":", 1)
        result[key.strip()] = int(value.strip())
    return result


def avg_hits1(summary):
    return summary["overall"].get("hits@1", 0.0)


def main():
    args = parse_args()
    topk_eval = [int(x.strip()) for x in args.topk_eval.split(",") if x.strip()]
    queries = load_pairs(args.pairs_jsonl)
    llm = load_llm(args.llm_jsonl)
    total_query_count_by_type = parse_count_by_type(args.total_query_count_by_type)

    param_list = []
    if args.sweep:
        for same_bonus in parse_float_list(args.sweep_same_entity_bonus):
            for safe_penalty in parse_float_list(args.sweep_safe_negative_penalty):
                for close_penalty in parse_float_list(args.sweep_closely_related_penalty):
                    for lambda_weight in parse_float_list(args.sweep_lambda_weight):
                        param_list.append({
                            "same_entity_bonus": same_bonus,
                            "closely_related_penalty": close_penalty,
                            "safe_negative_penalty": safe_penalty,
                            "unknown_penalty": args.unknown_penalty,
                            "lambda_weight": lambda_weight,
                        })
    else:
        param_list.append({
            "same_entity_bonus": args.same_entity_bonus,
            "closely_related_penalty": args.closely_related_penalty,
            "safe_negative_penalty": args.safe_negative_penalty,
            "unknown_penalty": args.unknown_penalty,
            "lambda_weight": args.lambda_weight,
        })

    sweep_results = []
    best = None
    best_payload = None
    for params in param_list:
        original_queries, reranked_queries, label_counter, baseline_summary, rerank_summary = rerank_once(
            queries=queries,
            llm=llm,
            topk_eval=topk_eval,
            params=params,
            total_query_count_by_type=total_query_count_by_type,
        )
        result = {
            **params,
            "baseline": baseline_summary,
            "reranked": rerank_summary,
            "llm_label_counts": dict(label_counter),
            "hits1_gain": round(
                rerank_summary["overall"].get("hits@1", 0.0) - baseline_summary["overall"].get("hits@1", 0.0),
                4,
            ),
            "mrr_gain": round(
                rerank_summary["overall"].get("mrr", 0.0) - baseline_summary["overall"].get("mrr", 0.0),
                4,
            ),
        }
        sweep_results.append(result)
        if best is None or (result["hits1_gain"], result["mrr_gain"]) > (best["hits1_gain"], best["mrr_gain"]):
            best = result
            best_payload = (original_queries, reranked_queries, label_counter, baseline_summary, rerank_summary, params)

    original_queries, reranked_queries, label_counter, baseline_summary, rerank_summary, best_params = best_payload
    os.makedirs(os.path.dirname(args.output_reranked_jsonl), exist_ok=True)
    with open(args.output_reranked_jsonl, "w", encoding="utf-8") as f:
        for key, candidates in reranked_queries.items():
            for row in candidates:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "topk_eval": topk_eval,
        **best_params,
        "llm_label_counts": dict(label_counter),
        "baseline": baseline_summary,
        "reranked": rerank_summary,
        "best_sweep_result": best,
        "sweep_results": sweep_results if args.sweep else [],
    }
    os.makedirs(os.path.dirname(args.output_summary_json), exist_ok=True)
    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Baseline =====", flush=True)
    print(json.dumps(baseline_summary, ensure_ascii=False, indent=2), flush=True)
    print("\n===== Reranked =====", flush=True)
    print(json.dumps(rerank_summary, ensure_ascii=False, indent=2), flush=True)
    if args.sweep:
        print("\n===== Best Sweep Result =====", flush=True)
        print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)
    print(f"\nSummary written to {args.output_summary_json}", flush=True)


if __name__ == "__main__":
    main()
