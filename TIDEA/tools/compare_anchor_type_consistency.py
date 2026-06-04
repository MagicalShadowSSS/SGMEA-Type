#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def load_ids(path):
    ids = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                ids.append(int(line.split("\t", 1)[0]))
    return ids


def load_pairs(path):
    pairs = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                left, right = line.split("\t")[:2]
                pairs.append((int(left), int(right)))
    return pairs


def load_types(path):
    id2type = {}
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                row = json.loads(line)
                id2type[int(row["id"])] = row.get("coarse_type", "Entity")
    return id2type


def main():
    parser = argparse.ArgumentParser(description="Check coarse-type consistency on aligned entity pairs.")
    parser.add_argument("--anchor_jsonl", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    id2type = load_types(args.anchor_jsonl)
    left_ids = load_ids(split_dir / "ent_ids_1")
    right_ids = load_ids(split_dir / "ent_ids_2")
    pairs = load_pairs(split_dir / "ill_ent_ids")

    left_counter = Counter(id2type.get(ent_id, "UNKNOWN") for ent_id in left_ids)
    right_counter = Counter(id2type.get(ent_id, "UNKNOWN") for ent_id in right_ids)
    mismatch = Counter()
    pair_match = 0
    pair_unknown = 0
    for left_id, right_id in pairs:
        lt = id2type.get(left_id, "UNKNOWN")
        rt = id2type.get(right_id, "UNKNOWN")
        if lt == "UNKNOWN" or rt == "UNKNOWN":
            pair_unknown += 1
        if lt == rt:
            pair_match += 1
        else:
            mismatch[(lt, rt)] += 1

    type_rows = []
    for typ in sorted(set(left_counter) | set(right_counter)):
        left_count = left_counter.get(typ, 0)
        right_count = right_counter.get(typ, 0)
        left_ratio = left_count / len(left_ids) if left_ids else 0.0
        right_ratio = right_count / len(right_ids) if right_ids else 0.0
        type_rows.append(
            {
                "coarse_type": typ,
                "left_count": left_count,
                "left_ratio": left_ratio,
                "right_count": right_count,
                "right_ratio": right_ratio,
                "ratio_gap_abs": abs(left_ratio - right_ratio),
            }
        )

    summary = {
        "anchor_jsonl": args.anchor_jsonl,
        "split_dir": args.split_dir,
        "left_total": len(left_ids),
        "right_total": len(right_ids),
        "distribution_rows": type_rows,
        "pair_total": len(pairs),
        "pair_match": pair_match,
        "pair_match_rate": pair_match / len(pairs) if pairs else 0.0,
        "pair_unknown": pair_unknown,
        "pair_unknown_rate": pair_unknown / len(pairs) if pairs else 0.0,
        "top_mismatches": [
            {"left_type": lt, "right_type": rt, "count": count}
            for (lt, rt), count in mismatch.most_common(20)
        ],
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 96)
    print(args.anchor_jsonl)
    print(f"Pair match rate: {summary['pair_match_rate']:.2%}")
    print("Top mismatches:")
    for item in summary["top_mismatches"][:10]:
        print(f"  {item['left_type']} -> {item['right_type']}: {item['count']}")


if __name__ == "__main__":
    main()
