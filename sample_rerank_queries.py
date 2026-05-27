import argparse
import json
import os
import random
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Sample rerank pilot queries while keeping all top-k pairs per query.")
    parser.add_argument("--input_jsonl", required=True, help="Full top-k pair JSONL from extract_test_topk_pairs.py")
    parser.add_argument("--output_jsonl", required=True, help="Pilot subset JSONL")
    parser.add_argument("--output_summary_json", required=True, help="Pilot subset summary JSON")
    parser.add_argument(
        "--query_budget_by_type",
        type=str,
        default="Place:200,Creative Work:200",
        help='Comma-separated per-type budgets, e.g. "Place:200,Creative Work:200"',
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_budget(spec):
    budgets = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid budget part: {part}")
        type_name, count = part.split(":", 1)
        budgets[type_name.strip()] = int(count.strip())
    return budgets


def load_pairs(path):
    by_query = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["query_coarse_type"], row["query_id"])
            by_query[key].append(row)
    return by_query


def main():
    args = parse_args()
    budgets = parse_budget(args.query_budget_by_type)
    rng = random.Random(args.seed)

    by_query = load_pairs(args.input_jsonl)
    query_ids_by_type = defaultdict(list)
    for (query_type, query_id) in by_query.keys():
        query_ids_by_type[query_type].append(query_id)

    selected_keys = []
    sampled_query_counter = Counter()
    available_query_counter = {t: len(ids) for t, ids in query_ids_by_type.items()}

    for query_type, budget in budgets.items():
        ids = query_ids_by_type.get(query_type, [])
        ids = sorted(set(ids))
        rng.shuffle(ids)
        chosen = ids[:budget]
        sampled_query_counter[query_type] = len(chosen)
        for query_id in chosen:
            selected_keys.append((query_type, query_id))

    rows_written = 0
    pair_counter = Counter()
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for key in selected_keys:
            rows = by_query.get(key, [])
            rows = sorted(rows, key=lambda item: item.get("candidate_rank", 10**9))
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
                pair_counter[row["query_coarse_type"]] += 1

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "query_budget_by_type": budgets,
        "available_query_count_by_type": available_query_counter,
        "sampled_query_count_by_type": dict(sampled_query_counter),
        "pair_count_by_type": dict(pair_counter),
        "rows_written": rows_written,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.output_summary_json), exist_ok=True)
    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===== Rerank Pilot Sampling =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
