import argparse
import csv
import glob
import json
import math
import os
import os.path as osp
import pickle
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np


DEFAULT_MODAL_NAMES = {
    4: ["img", "attr", "rel", "graph"],
    6: ["img", "attr", "rel", "graph", "name", "char"],
}


DEFAULT_PAIRS = [
    {
        "name": "FBDB15K_r02",
        "dataset": "FBDB15K",
        "rate": "0.2",
        "baseline": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBDB15K_0.2_baseline_warm_fbdb_rates_cdmr_0511_104108_r02__test_ep500_pred",
        "cdmr": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBDB15K_0.2_cdmr_prior_fbdb_rates_cdmr_0511_104108_r02__test_ep40_pred",
        "type_jsonl": "/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl",
    },
    {
        "name": "FBDB15K_r05",
        "dataset": "FBDB15K",
        "rate": "0.5",
        "baseline": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05__test_ep500_pred",
        "cdmr": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBDB15K_0.5_cdmr_prior_fbdb_rates_cdmr_0511_104108_r05__test_ep40_pred",
        "type_jsonl": "/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl",
    },
    {
        "name": "FBDB15K_r08",
        "dataset": "FBDB15K",
        "rate": "0.8",
        "baseline": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBDB15K_0.8_baseline_warm_fbdb_rates_cdmr_0511_104108_r08__test_ep500_pred",
        "cdmr": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBDB15K_0.8_cdmr_prior_fbdb_rates_cdmr_0511_104108_r08__test_ep40_pred",
        "type_jsonl": "/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl",
    },
    {
        "name": "FBYG15K_r02_lr5e4",
        "dataset": "FBYG15K",
        "rate": "0.2",
        "baseline": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBYG15K_0.2_baseline_warm_fbyg_ordered_lr5e4_0511_223456_r02__test_ep250_pred",
        "cdmr": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBYG15K_0.2_cdmr_prior_rerun_lr5e4_fbyg_ordered_lr5e4_0511_223456_r02__test_ep40_pred",
        "type_jsonl": "/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBYG15K/norm_anchor_type_fixed.jsonl",
    },
    {
        "name": "FBYG15K_r05_lr5e4",
        "dataset": "FBYG15K",
        "rate": "0.5",
        "baseline": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBYG15K_0.5_baseline_warm_fbyg_ordered_lr5e4_0511_230140_r05__test_ep250_pred",
        "cdmr": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBYG15K_0.5_cdmr_prior_rerun_lr5e4_fbyg_ordered_lr5e4_0511_230140_r05__test_ep40_pred",
        "type_jsonl": "/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBYG15K/norm_anchor_type_fixed.jsonl",
    },
    {
        "name": "FBYG15K_r08_lr5e4",
        "dataset": "FBYG15K",
        "rate": "0.8",
        "baseline": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBYG15K_0.8_baseline_warm_fbyg_ordered_lr5e4_0511_230140_r08__test_ep250_pred",
        "cdmr": "/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/SGMEA_FBYG15K_0.8_cdmr_prior_rerun_lr5e4_fbyg_ordered_lr5e4_0511_230140_r08__test_ep40_pred",
        "type_jsonl": "/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBYG15K/norm_anchor_type_fixed.jsonl",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline modality-weight interpretability audit for saved SGMEA baseline/CDMR predictions."
    )
    parser.add_argument("--output_dir", default="", help="directory for summary json/csv files")
    parser.add_argument("--data_root", default="/gly/tongqiang/dongyufeng/data/mmkg")
    parser.add_argument("--correct_rule", choices=["ret1", "rank"], default="ret1",
                        help="use ret1==gt_id by default; older baseline pred files can have stale rank fields")
    parser.add_argument("--pair", action="append", default=[],
                        help="custom pair as name|dataset|rate|baseline_pred_dir|cdmr_pred_dir|type_jsonl")
    parser.add_argument("--skip_missing", action="store_true", default=True)
    parser.add_argument("--no_skip_missing", dest="skip_missing", action="store_false")
    parser.add_argument("--write_entity_csv", action="store_true", default=True)
    parser.add_argument("--no_entity_csv", dest="write_entity_csv", action="store_false")
    return parser.parse_args()


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def find_one(directory, suffix):
    matches = sorted(glob.glob(osp.join(directory, f"*{suffix}")))
    if not matches:
        raise FileNotFoundError(f"cannot find *{suffix} in {directory}")
    return matches[0]


def load_pred_dir(directory):
    pred_path = find_one(directory, "_pred.txt")
    weight_path = find_one(directory, "_wight_dic.pkl")
    with open(pred_path, "r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    with open(weight_path, "rb") as fp:
        weights = pickle.load(fp)
    weights = {key: as_numpy(value).astype(np.float64) for key, value in weights.items()}
    return {
        "pred_path": pred_path,
        "weight_path": weight_path,
        "rows": rows,
        "weights": weights,
        "type_bias": load_type_bias(directory),
    }


def load_type_bias(directory):
    path = osp.join(directory, "type_modality_bias.json")
    if not osp.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    bias = np.asarray(data.get("bias", []), dtype=np.float64)
    type_names = data.get("type_names", [])
    return {
        "path": path,
        "type_names": type_names,
        "bias": bias.tolist(),
        "exp_bias": np.exp(bias).tolist() if bias.size else [],
        "scale": data.get("scale", 1.0),
    }


def load_types(path):
    ent_to_type = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ent_to_type[int(row["id"])] = row.get("coarse_type", "Entity")
    return ent_to_type


def load_degree(data_root, dataset, split="norm"):
    degree = Counter()
    base = osp.join(data_root, dataset, split)
    for name in ("triples_1", "triples_2"):
        path = osp.join(base, name)
        if not osp.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h, _, t = [int(x) for x in parts[:3]]
                degree[h] += 1
                degree[t] += 1
    return degree


def parse_pairs(args):
    if not args.pair:
        return DEFAULT_PAIRS
    pairs = []
    for raw in args.pair:
        parts = raw.split("|")
        if len(parts) != 6:
            raise ValueError("--pair must be name|dataset|rate|baseline_pred_dir|cdmr_pred_dir|type_jsonl")
        name, dataset, rate, baseline, cdmr, type_jsonl = parts
        pairs.append({
            "name": name,
            "dataset": dataset,
            "rate": rate,
            "baseline": baseline,
            "cdmr": cdmr,
            "type_jsonl": type_jsonl,
        })
    return pairs


def modal_names_for(weights):
    dim = int(weights.shape[1])
    if dim in DEFAULT_MODAL_NAMES:
        return DEFAULT_MODAL_NAMES[dim]
    return [f"modal_{idx}" for idx in range(dim)]


def entropy(weights):
    arr = np.clip(weights, 1e-12, 1.0)
    return -np.sum(arr * np.log(arr), axis=1)


def stats_vector(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def stats_matrix(weights, modal_names):
    out = {"count": int(weights.shape[0])}
    if weights.shape[0] == 0:
        for name in modal_names:
            out[name] = {"count": 0}
        out["entropy"] = {"count": 0}
        return out
    for idx, name in enumerate(modal_names):
        out[name] = stats_vector(weights[:, idx])
    out["entropy"] = stats_vector(entropy(weights))
    out["argmax_modal"] = {
        modal: int(np.sum(np.argmax(weights, axis=1) == idx))
        for idx, modal in enumerate(modal_names)
    }
    return out


def flatten_stats(prefix, stats, modal_names):
    row = {f"{prefix}_n": stats.get("count", 0)}
    for modal in modal_names:
        modal_stats = stats.get(modal, {})
        for key in ("mean", "std", "min", "median", "max"):
            row[f"{prefix}_{modal}_{key}"] = modal_stats.get(key, "")
    ent = stats.get("entropy", {})
    for key in ("mean", "std", "median"):
        row[f"{prefix}_entropy_{key}"] = ent.get(key, "")
    return row


def correct_flags(rows, rule):
    flags = []
    for row in rows:
        if rule == "rank":
            flags.append(int(row["rank"]) == 0)
        else:
            flags.append(str(row["ret1"]) == str(row["gt_id"]))
    return np.asarray(flags, dtype=bool)


def rank_ret1_mismatch(rows):
    mismatch = 0
    for row in rows:
        mismatch += (int(row["rank"]) == 0) != (str(row["ret1"]) == str(row["gt_id"]))
    return mismatch


def align_rows(base_rows, cdmr_rows):
    if len(base_rows) != len(cdmr_rows):
        raise ValueError(f"row count mismatch: baseline={len(base_rows)} cdmr={len(cdmr_rows)}")
    for idx, (a, b) in enumerate(zip(base_rows, cdmr_rows)):
        key_a = (a.get("idx"), a.get("query_id"), a.get("gt_id"))
        key_b = (b.get("idx"), b.get("query_id"), b.get("gt_id"))
        if key_a != key_b:
            raise ValueError(f"pred row mismatch at {idx}: baseline={key_a} cdmr={key_b}")


def degree_bins_for(query_ids, degree):
    deg = np.asarray([degree.get(int(ent), 0) for ent in query_ids], dtype=np.float64)
    if deg.size == 0:
        return deg, np.asarray([], dtype=object)
    q33, q67 = np.percentile(deg, [33.333, 66.667])
    bins = np.full(deg.shape, "mid_degree", dtype=object)
    bins[deg <= q33] = "low_degree"
    bins[deg >= q67] = "high_degree"
    return deg, bins


def summarize_subset(mask, base_w, cdmr_w, base_names, cdmr_names):
    base_subset = base_w[mask]
    cdmr_subset = cdmr_w[mask]
    out = {
        "baseline": stats_matrix(base_subset, base_names),
        "cdmr": stats_matrix(cdmr_subset, cdmr_names),
    }
    common = [name for name in base_names if name in cdmr_names]
    if common:
        base_idx = [base_names.index(name) for name in common]
        cdmr_idx = [cdmr_names.index(name) for name in common]
        out["delta_cdmr_minus_baseline"] = stats_matrix(cdmr_subset[:, cdmr_idx] - base_subset[:, base_idx], common)
    return out


def summarize_pair(pair, args):
    base = load_pred_dir(pair["baseline"])
    cdmr = load_pred_dir(pair["cdmr"])
    align_rows(base["rows"], cdmr["rows"])

    ent_to_type = load_types(pair["type_jsonl"])
    query_ids = np.asarray([int(row["query_id"]) for row in base["rows"]], dtype=np.int64)
    gt_ids = np.asarray([int(row["gt_id"]) for row in base["rows"]], dtype=np.int64)
    query_types = np.asarray([ent_to_type.get(int(ent), "Entity") for ent in query_ids], dtype=object)
    gt_types = np.asarray([ent_to_type.get(int(ent), "Entity") for ent in gt_ids], dtype=object)
    type_mismatch = query_types != gt_types

    base_w_left = np.asarray(base["weights"]["left"], dtype=np.float64)
    cdmr_w_left = np.asarray(cdmr["weights"]["left"], dtype=np.float64)
    if base_w_left.shape[0] != len(base["rows"]) or cdmr_w_left.shape[0] != len(base["rows"]):
        raise ValueError("left weight row count does not match pred rows")

    base_names = modal_names_for(base_w_left)
    cdmr_names = modal_names_for(cdmr_w_left)

    base_correct = correct_flags(base["rows"], args.correct_rule)
    cdmr_correct = correct_flags(cdmr["rows"], args.correct_rule)
    groups = {
        "all": np.ones(len(base["rows"]), dtype=bool),
        "baseline_correct": base_correct,
        "cdmr_correct": cdmr_correct,
        "both_correct": base_correct & cdmr_correct,
        "both_wrong": ~base_correct & ~cdmr_correct,
        "rescued": ~base_correct & cdmr_correct,
        "harmed": base_correct & ~cdmr_correct,
        "type_consistent": ~type_mismatch,
        "type_mismatch": type_mismatch,
    }

    degree, degree_bin = degree_bins_for(query_ids, load_degree(args.data_root, pair["dataset"]))
    for bin_name in ("low_degree", "mid_degree", "high_degree"):
        groups[bin_name] = degree_bin == bin_name

    summary = {
        "name": pair["name"],
        "dataset": pair["dataset"],
        "rate": pair["rate"],
        "baseline_dir": pair["baseline"],
        "cdmr_dir": pair["cdmr"],
        "type_jsonl": pair["type_jsonl"],
        "pred_paths": {
            "baseline": base["pred_path"],
            "cdmr": cdmr["pred_path"],
        },
        "weight_paths": {
            "baseline": base["weight_path"],
            "cdmr": cdmr["weight_path"],
        },
        "modal_names": {
            "baseline": base_names,
            "cdmr": cdmr_names,
        },
        "correct_rule": args.correct_rule,
        "rank_ret1_mismatch": {
            "baseline": rank_ret1_mismatch(base["rows"]),
            "cdmr": rank_ret1_mismatch(cdmr["rows"]),
        },
        "metrics": {
            "n": int(len(base["rows"])),
            "baseline_hits1": float(np.mean(base_correct)),
            "cdmr_hits1": float(np.mean(cdmr_correct)),
            "delta_hits1": float(np.mean(cdmr_correct) - np.mean(base_correct)),
            "rescued_count": int(np.sum(groups["rescued"])),
            "harmed_count": int(np.sum(groups["harmed"])),
            "both_correct_count": int(np.sum(groups["both_correct"])),
            "both_wrong_count": int(np.sum(groups["both_wrong"])),
            "type_mismatch_count": int(np.sum(type_mismatch)),
        },
        "type_distribution": dict(Counter(query_types.tolist())),
        "degree": {
            "overall": stats_vector(degree),
            "bin_counts": dict(Counter(degree_bin.tolist())),
        },
        "type_bias": cdmr["type_bias"],
        "groups": {},
        "by_type": {},
        "degree_bins": {},
    }

    for group_name, mask in groups.items():
        summary["groups"][group_name] = summarize_subset(mask, base_w_left, cdmr_w_left, base_names, cdmr_names)

    for type_name in sorted(set(query_types.tolist())):
        mask = query_types == type_name
        type_summary = summarize_subset(mask, base_w_left, cdmr_w_left, base_names, cdmr_names)
        type_summary["metrics"] = {
            "n": int(np.sum(mask)),
            "baseline_hits1": float(np.mean(base_correct[mask])) if np.sum(mask) else 0.0,
            "cdmr_hits1": float(np.mean(cdmr_correct[mask])) if np.sum(mask) else 0.0,
            "delta_hits1": float(np.mean(cdmr_correct[mask]) - np.mean(base_correct[mask])) if np.sum(mask) else 0.0,
            "rescued_count": int(np.sum(groups["rescued"] & mask)),
            "harmed_count": int(np.sum(groups["harmed"] & mask)),
            "degree_mean": float(np.mean(degree[mask])) if np.sum(mask) else 0.0,
        }
        summary["by_type"][type_name] = type_summary

    for bin_name in ("low_degree", "mid_degree", "high_degree"):
        mask = groups[bin_name]
        bin_summary = summarize_subset(mask, base_w_left, cdmr_w_left, base_names, cdmr_names)
        bin_summary["metrics"] = {
            "n": int(np.sum(mask)),
            "baseline_hits1": float(np.mean(base_correct[mask])) if np.sum(mask) else 0.0,
            "cdmr_hits1": float(np.mean(cdmr_correct[mask])) if np.sum(mask) else 0.0,
            "delta_hits1": float(np.mean(cdmr_correct[mask]) - np.mean(base_correct[mask])) if np.sum(mask) else 0.0,
            "degree_mean": float(np.mean(degree[mask])) if np.sum(mask) else 0.0,
        }
        summary["degree_bins"][bin_name] = bin_summary

    entity_rows = []
    if args.write_entity_csv:
        common = [name for name in base_names if name in cdmr_names]
        for idx, row in enumerate(base["rows"]):
            out = {
                "idx": row["idx"],
                "query_id": int(row["query_id"]),
                "gt_id": int(row["gt_id"]),
                "type": query_types[idx],
                "gt_type": gt_types[idx],
                "type_mismatch": bool(type_mismatch[idx]),
                "degree": float(degree[idx]),
                "degree_bin": degree_bin[idx],
                "baseline_correct": bool(base_correct[idx]),
                "cdmr_correct": bool(cdmr_correct[idx]),
                "group": "rescued" if groups["rescued"][idx] else "harmed" if groups["harmed"][idx] else "both_correct" if groups["both_correct"][idx] else "both_wrong",
                "baseline_rank": int(base["rows"][idx]["rank"]),
                "cdmr_rank": int(cdmr["rows"][idx]["rank"]),
                "baseline_ret1": int(base["rows"][idx]["ret1"]),
                "cdmr_ret1": int(cdmr["rows"][idx]["ret1"]),
            }
            for modal_i, modal in enumerate(base_names):
                out[f"baseline_{modal}"] = float(base_w_left[idx, modal_i])
            for modal_i, modal in enumerate(cdmr_names):
                out[f"cdmr_{modal}"] = float(cdmr_w_left[idx, modal_i])
            for modal in common:
                out[f"delta_{modal}"] = float(cdmr_w_left[idx, cdmr_names.index(modal)] - base_w_left[idx, base_names.index(modal)])
            entity_rows.append(out)

    return summary, entity_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.output_dir:
        tag = datetime.now().strftime("%m%d_%H%M%S")
        args.output_dir = osp.join("log", "audits", "modality_weight_stats", f"sgmea_modal_stats_{tag}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": osp.abspath(args.output_dir),
        "correct_rule": args.correct_rule,
        "pairs": {},
    }
    per_type_rows = []
    group_rows = []
    degree_rows = []

    for pair in parse_pairs(args):
        missing = [key for key in ("baseline", "cdmr", "type_jsonl") if not osp.exists(pair[key])]
        if missing:
            msg = f"missing inputs for {pair['name']}: {missing}"
            if args.skip_missing:
                print(f"[skip] {msg}")
                continue
            raise FileNotFoundError(msg)

        print(f"[audit] {pair['name']}")
        summary, entity_rows = summarize_pair(pair, args)
        all_summary["pairs"][pair["name"]] = summary

        pair_dir = osp.join(args.output_dir, pair["name"])
        os.makedirs(pair_dir, exist_ok=True)
        with open(osp.join(pair_dir, "summary.json"), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        if entity_rows:
            write_csv(osp.join(pair_dir, "entity_rows.csv"), entity_rows)

        base_names = summary["modal_names"]["baseline"]
        cdmr_names = summary["modal_names"]["cdmr"]
        common_names = [name for name in base_names if name in cdmr_names]
        for type_name, type_summary in summary["by_type"].items():
            row = {
                "pair": pair["name"],
                "dataset": pair["dataset"],
                "rate": pair["rate"],
                "type": type_name,
                **type_summary["metrics"],
            }
            row.update(flatten_stats("baseline", type_summary["baseline"], base_names))
            row.update(flatten_stats("cdmr", type_summary["cdmr"], cdmr_names))
            if "delta_cdmr_minus_baseline" in type_summary:
                row.update(flatten_stats("delta", type_summary["delta_cdmr_minus_baseline"], common_names))
            per_type_rows.append(row)

        for group_name, group_summary in summary["groups"].items():
            row = {
                "pair": pair["name"],
                "dataset": pair["dataset"],
                "rate": pair["rate"],
                "group": group_name,
            }
            row.update(flatten_stats("baseline", group_summary["baseline"], base_names))
            row.update(flatten_stats("cdmr", group_summary["cdmr"], cdmr_names))
            if "delta_cdmr_minus_baseline" in group_summary:
                row.update(flatten_stats("delta", group_summary["delta_cdmr_minus_baseline"], common_names))
            group_rows.append(row)

        for bin_name, bin_summary in summary["degree_bins"].items():
            row = {
                "pair": pair["name"],
                "dataset": pair["dataset"],
                "rate": pair["rate"],
                "degree_bin": bin_name,
                **bin_summary["metrics"],
            }
            row.update(flatten_stats("baseline", bin_summary["baseline"], base_names))
            row.update(flatten_stats("cdmr", bin_summary["cdmr"], cdmr_names))
            if "delta_cdmr_minus_baseline" in bin_summary:
                row.update(flatten_stats("delta", bin_summary["delta_cdmr_minus_baseline"], common_names))
            degree_rows.append(row)

        metrics = summary["metrics"]
        print(
            f"  n={metrics['n']} base={metrics['baseline_hits1']:.4f} "
            f"cdmr={metrics['cdmr_hits1']:.4f} delta={metrics['delta_hits1']:+.4f} "
            f"rescued={metrics['rescued_count']} harmed={metrics['harmed_count']}"
        )

    with open(osp.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(all_summary, fp, indent=2, ensure_ascii=False)
    write_csv(osp.join(args.output_dir, "per_type_stats.csv"), per_type_rows)
    write_csv(osp.join(args.output_dir, "group_stats.csv"), group_rows)
    write_csv(osp.join(args.output_dir, "degree_bin_stats.csv"), degree_rows)
    print(f"[done] {osp.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
