import argparse
import json
import os
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Build unified relation-aware HNM cache from training-side LLM labels.")
    parser.add_argument(
        "--labels_jsonls",
        required=True,
        help="Comma-separated training-side label jsonl paths.",
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary_json", required=True)
    parser.add_argument(
        "--keep_statuses",
        default="closely_related,safe_negative",
        help="Comma-separated statuses to keep.",
    )
    parser.add_argument(
        "--keep_coarse_types",
        default="Place,Creative Work,Organization",
        help="Comma-separated coarse types to keep.",
    )
    parser.add_argument(
        "--quality_filter",
        default="all",
        choices=["all", "high", "low"],
        help="Optional evidence_quality filter.",
    )
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument(
        "--dedupe_by",
        default="anchor_candidate_direction",
        choices=["anchor_candidate_direction", "anchor_positive_candidate_direction"],
    )
    return parser.parse_args()


def split_csv(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value


def make_dedupe_key(row, mode):
    if mode == "anchor_positive_candidate_direction":
        return (
            safe_text(row.get("direction")),
            int(row.get("anchor_id")),
            int(row.get("positive_id")),
            int(row.get("negative_id")),
            int(row.get("candidate_rank", 1)),
        )
    return (
        safe_text(row.get("direction")),
        int(row.get("anchor_id")),
        int(row.get("negative_id")),
        int(row.get("candidate_rank", 1)),
    )


def project_row(row):
    return {
        "dataset": row.get("dataset"),
        "direction": row.get("direction"),
        "coarse_type": row.get("coarse_type"),
        "cache_row": row.get("cache_row"),
        "anchor_id": row.get("anchor_id"),
        "positive_id": row.get("positive_id"),
        "candidate_id": row.get("negative_id"),
        "candidate_rank": row.get("candidate_rank", 1),
        "candidate_distance": row.get("candidate_sim", row.get("top1_sim")),
        "llm_relationship_status": row.get("llm_relationship_status"),
        "llm_relationship_kind": row.get("llm_relationship_kind"),
        "llm_confidence": row.get("llm_confidence"),
        "evidence_quality": row.get("evidence_quality"),
        "known_full_ill_hit": row.get("known_full_ill_hit", False),
    }


def summarize_rows(rows):
    status_counter = Counter()
    coarse_counter = defaultdict(Counter)
    direction_counter = Counter()
    quality_counter = Counter()
    conf_values = []
    unique_anchor_ids = set()
    unique_candidate_ids = set()
    unique_pairs = set()

    for row in rows:
        status = safe_text(row.get("llm_relationship_status"))
        coarse_type = safe_text(row.get("coarse_type"))
        direction = safe_text(row.get("direction"))
        quality = safe_text(row.get("evidence_quality"))
        conf = safe_float(row.get("llm_confidence"), 0.0)

        status_counter[status] += 1
        coarse_counter[coarse_type][status] += 1
        direction_counter[direction] += 1
        quality_counter[quality] += 1
        conf_values.append(conf)
        unique_anchor_ids.add(int(row.get("anchor_id")))
        unique_candidate_ids.add(int(row.get("candidate_id")))
        unique_pairs.add((direction, int(row.get("anchor_id")), int(row.get("candidate_id"))))

    conf_values = sorted(conf_values)
    if conf_values:
        conf_summary = {
            "min": round(conf_values[0], 6),
            "max": round(conf_values[-1], 6),
            "mean": round(sum(conf_values) / len(conf_values), 6),
            "median": round(conf_values[len(conf_values) // 2], 6),
        }
    else:
        conf_summary = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

    return {
        "row_count": len(rows),
        "status_counts": dict(status_counter),
        "coarse_type_breakdown": {k: dict(v) for k, v in coarse_counter.items()},
        "direction_counts": dict(direction_counter),
        "evidence_quality_counts": dict(quality_counter),
        "confidence_summary": conf_summary,
        "unique_anchor_count": len(unique_anchor_ids),
        "unique_candidate_count": len(unique_candidate_ids),
        "unique_anchor_candidate_direction_count": len(unique_pairs),
    }


def main():
    args = parse_args()
    label_paths = split_csv(args.labels_jsonls)
    keep_statuses = set(split_csv(args.keep_statuses))
    keep_coarse_types = set(split_csv(args.keep_coarse_types))

    input_rows = []
    for path in label_paths:
        input_rows.extend(read_jsonl(path))

    kept_rows = []
    seen = set()
    dropped_reasons = Counter()

    for row in input_rows:
        status = safe_text(row.get("llm_relationship_status"))
        coarse_type = safe_text(row.get("coarse_type"))
        quality = safe_text(row.get("evidence_quality"))
        conf = safe_float(row.get("llm_confidence"), 0.0)

        if status not in keep_statuses:
            dropped_reasons[f"status:{status or 'empty'}"] += 1
            continue
        if coarse_type not in keep_coarse_types:
            dropped_reasons[f"coarse_type:{coarse_type or 'empty'}"] += 1
            continue
        if args.quality_filter != "all" and quality != args.quality_filter:
            dropped_reasons[f"quality:{quality or 'empty'}"] += 1
            continue
        if conf < args.min_confidence:
            dropped_reasons["low_confidence"] += 1
            continue

        proj = project_row(row)
        key = make_dedupe_key(row, args.dedupe_by)
        if key in seen:
            dropped_reasons["dedupe"] += 1
            continue
        seen.add(key)
        kept_rows.append(proj)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "labels_jsonls": label_paths,
        "keep_statuses": sorted(keep_statuses),
        "keep_coarse_types": sorted(keep_coarse_types),
        "quality_filter": args.quality_filter,
        "min_confidence": args.min_confidence,
        "dedupe_by": args.dedupe_by,
        "input_row_count": len(input_rows),
        "dropped_reason_counts": dict(dropped_reasons),
        "kept_summary": summarize_rows(kept_rows),
    }

    os.makedirs(os.path.dirname(args.output_summary_json), exist_ok=True)
    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===== Relation-Aware HNM Cache Built =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
