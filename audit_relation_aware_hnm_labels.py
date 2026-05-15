import argparse
import json
import math
import os
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Audit training-side relation-aware HNM labels and confidence distribution.")
    parser.add_argument("--cases_jsonl", required=True, help="Training-side hardest-case jsonl from extract_llm_hnm_cases.py")
    parser.add_argument("--labels_jsonl", required=True, help="LLM fine-type label jsonl from build_llm_fine_types.py")
    parser.add_argument("--output_summary_json", required=True)
    parser.add_argument("--output_same_entity_jsonl", default="", help="Optional path to export same_entity rows")
    parser.add_argument("--high_conf_threshold", type=float, default=0.8)
    return parser.parse_args()


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        value = float(value)
        if math.isnan(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def join_cases_and_labels(cases_rows, label_rows):
    case_map = {}
    for row in cases_rows:
        key = (
            safe_text(row.get("dataset")),
            safe_text(row.get("cache_row")),
            safe_text(row.get("direction")),
            safe_text(row.get("candidate_rank", 1)),
            safe_text(row.get("negative_id")),
        )
        case_map[key] = row

    merged = []
    missing = 0
    for row in label_rows:
        key = (
            safe_text(row.get("dataset")),
            safe_text(row.get("cache_row")),
            safe_text(row.get("direction")),
            safe_text(row.get("candidate_rank", 1)),
            safe_text(row.get("negative_id")),
        )
        base = case_map.get(key)
        if base is None:
            missing += 1
            continue
        obj = dict(base)
        obj.update(row)
        merged.append(obj)
    return merged, missing


def describe_conf(values):
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    vals = sorted(float(v) for v in values)
    n = len(vals)

    def percentile(p):
        if n == 1:
            return vals[0]
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        w = idx - lo
        return vals[lo] * (1.0 - w) + vals[hi] * w

    mean = sum(vals) / n
    median = percentile(0.5)
    return {
        "count": n,
        "min": round(vals[0], 6),
        "max": round(vals[-1], 6),
        "mean": round(mean, 6),
        "median": round(median, 6),
        "p25": round(percentile(0.25), 6),
        "p75": round(percentile(0.75), 6),
    }


def bucket_conf(conf):
    if conf >= 0.9:
        return "0.9-1.0"
    if conf >= 0.8:
        return "0.8-0.9"
    if conf >= 0.7:
        return "0.7-0.8"
    if conf >= 0.6:
        return "0.6-0.7"
    if conf >= 0.5:
        return "0.5-0.6"
    return "<0.5"


def summarize_rows(rows, high_conf_threshold):
    status_counter = Counter()
    quality_counter = Counter()
    route_counter = defaultdict(Counter)
    conf_by_status = defaultdict(list)
    conf_by_type = defaultdict(list)
    conf_by_quality = defaultdict(list)
    conf_bucket_by_status = defaultdict(Counter)
    same_entity_known_hit_counter = Counter()
    same_entity_by_type = defaultdict(Counter)
    known_hit_by_status = defaultdict(Counter)
    high_conf_status_counter = Counter()
    same_entity_rows = []

    for row in rows:
        status = safe_text(row.get("llm_relationship_status", "unknown")) or "unknown"
        coarse_type = safe_text(row.get("coarse_type", "unknown")) or "unknown"
        evidence_quality = safe_text(row.get("evidence_quality", "unknown")) or "unknown"
        conf = safe_float(row.get("llm_confidence", 0.0), default=0.0)
        known_hit = int(bool(row.get("known_full_ill_hit", False)))

        status_counter[status] += 1
        quality_counter[evidence_quality] += 1
        route_counter[coarse_type][status] += 1
        conf_by_status[status].append(conf)
        conf_by_type[coarse_type].append(conf)
        conf_by_quality[evidence_quality].append(conf)
        conf_bucket_by_status[status][bucket_conf(conf)] += 1
        known_hit_by_status[status][str(known_hit)] += 1

        if conf >= high_conf_threshold:
            high_conf_status_counter[status] += 1

        if status == "same_entity":
            same_entity_known_hit_counter[str(known_hit)] += 1
            same_entity_by_type[coarse_type][str(known_hit)] += 1
            same_entity_rows.append(row)

    same_total = status_counter.get("same_entity", 0)
    same_known_hit_rate = round(same_entity_known_hit_counter.get("1", 0) / same_total, 6) if same_total > 0 else 0.0

    by_type_same_entity = {}
    for coarse_type, counter in same_entity_by_type.items():
        total = sum(counter.values())
        by_type_same_entity[coarse_type] = {
            "count": total,
            "known_full_ill_hit_0": counter.get("0", 0),
            "known_full_ill_hit_1": counter.get("1", 0),
            "known_full_ill_hit_rate": round(counter.get("1", 0) / total, 6) if total > 0 else 0.0,
        }

    summary = {
        "row_count": len(rows),
        "relationship_status_counts": dict(status_counter),
        "evidence_quality_counts": dict(quality_counter),
        "high_conf_threshold": high_conf_threshold,
        "high_conf_status_counts": dict(high_conf_status_counter),
        "coarse_type_breakdown": {k: dict(v) for k, v in route_counter.items()},
        "confidence_by_status": {k: describe_conf(v) for k, v in conf_by_status.items()},
        "confidence_by_coarse_type": {k: describe_conf(v) for k, v in conf_by_type.items()},
        "confidence_by_evidence_quality": {k: describe_conf(v) for k, v in conf_by_quality.items()},
        "confidence_bucket_by_status": {k: dict(v) for k, v in conf_bucket_by_status.items()},
        "known_full_ill_hit_by_status": {k: dict(v) for k, v in known_hit_by_status.items()},
        "same_entity_summary": {
            "count": same_total,
            "known_full_ill_hit_0": same_entity_known_hit_counter.get("0", 0),
            "known_full_ill_hit_1": same_entity_known_hit_counter.get("1", 0),
            "known_full_ill_hit_rate": same_known_hit_rate,
            "by_coarse_type": by_type_same_entity,
        },
    }
    return summary, same_entity_rows


def main():
    args = parse_args()
    case_rows = read_jsonl(args.cases_jsonl)
    label_rows = read_jsonl(args.labels_jsonl)
    merged_rows, missing_label_joins = join_cases_and_labels(case_rows, label_rows)
    summary, same_entity_rows = summarize_rows(merged_rows, args.high_conf_threshold)
    summary["cases_jsonl"] = args.cases_jsonl
    summary["labels_jsonl"] = args.labels_jsonl
    summary["input_case_count"] = len(case_rows)
    summary["input_label_count"] = len(label_rows)
    summary["missing_label_joins"] = missing_label_joins

    os.makedirs(os.path.dirname(args.output_summary_json), exist_ok=True)
    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.output_same_entity_jsonl:
        os.makedirs(os.path.dirname(args.output_same_entity_jsonl), exist_ok=True)
        with open(args.output_same_entity_jsonl, "w", encoding="utf-8") as f:
            for row in same_entity_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("===== Relation-Aware HNM Label Audit =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.output_same_entity_jsonl:
        print(f"saved same_entity rows to {args.output_same_entity_jsonl}", flush=True)


if __name__ == "__main__":
    main()
