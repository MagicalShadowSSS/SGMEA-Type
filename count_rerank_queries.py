import argparse
import json
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser(description="Count unique query ids in a rerank top-k pair file.")
    parser.add_argument("--input_jsonl", required=True)
    args = parser.parse_args()

    query_ids = set()
    query_type_counter = Counter()
    pairs_per_query = defaultdict(int)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("query_id")
            qtype = row.get("query_coarse_type", "unknown")
            query_ids.add(qid)
            query_type_counter[qtype] += 0  # ensure key exists
            pairs_per_query[(qtype, qid)] += 1

    for qtype, qid in {(qt, q) for qt, q in pairs_per_query.keys()}:
        query_type_counter[qtype] += 1

    total_pairs = sum(pairs_per_query.values())
    total_queries = len(query_ids)
    avg_pairs = total_pairs / total_queries if total_queries > 0 else 0.0

    print("===== Rerank Pair Coverage =====", flush=True)
    print(f"total_pairs: {total_pairs}", flush=True)
    print(f"unique_query_ids: {total_queries}", flush=True)
    print(f"avg_pairs_per_query: {avg_pairs:.2f}", flush=True)
    print("query_count_by_type:", flush=True)
    for qtype, count in sorted(query_type_counter.items()):
        print(f"  {qtype}: {count}", flush=True)


if __name__ == "__main__":
    main()
