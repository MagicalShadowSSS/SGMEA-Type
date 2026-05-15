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
from src.utils import get_adjr, pairwise_distances, csls_sim
from torchlight import set_seed


class SimpleLogger:
    def info(self, msg):
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache_jsonl", required=True)
    parser.add_argument("--output_summary_json", required=True)
    parser.add_argument("--focus_coarse_types", default="Person,Place,Creative Work,Organization")
    parser.add_argument("--min_confidence", type=float, default=0.9)
    parser.add_argument("--quality_filter", default="all", choices=["all", "high"])
    parser.add_argument(
        "--edge_filter_preset",
        default="all_close",
        choices=["all_close", "strong_plus_wide095_rank1"],
        help="optional relationship-kind filter before selecting virtual edges",
    )
    parser.add_argument("--wide_min_confidence", type=float, default=0.95)
    parser.add_argument("--wide_max_candidate_rank", type=int, default=1)
    parser.add_argument("--max_edges_per_positive", type=int, default=1)
    parser.add_argument("--max_total_edges", type=int, default=0)
    parser.add_argument("--sample_limit", type=int, default=0)
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


BROAD_RELATION_KINDS = {
    "same_profession",
    "same_profession_same_era",
    "same_profession_different_era",
    "same_region",
    "same_industry",
    "same_coarse_type",
    "same_birth_year",
    "same_founding_year",
    "same_country",
    "same_type",
    "same_category",
    "same_domain",
    "same_genre",
    "same_location",
    "same_type_same_domain",
    "same_type_different_location",
    "same_type_different_offset",
    "same_political_system",
    "broad_related",
    "broad_unrelated",
}


STRONG_RELATION_KEYWORDS = (
    "parent",
    "child",
    "capital",
    "subsidiary",
    "merger",
    "acquisition",
    "acquired",
    "franchise",
    "cofounder",
    "co_founder",
    "founder",
    "collaborator",
    "collaboration",
    "partner",
    "duo",
    "same_series",
    "series",
    "same_franchise",
    "same_system",
    "local_sibling",
    "sibling_place",
    "local_sibling_place",
    "county",
    "neighbor_states",
    "adjacent",
    "geographic_proximity",
    "same_university",
    "same_group",
    "administration",
    "president_vice_president",
    "band_member",
    "family",
)


def relation_kind(row):
    return str(row.get("llm_relationship_kind", "")).strip()


def is_strong_relation_kind(kind):
    if kind in BROAD_RELATION_KINDS:
        return False
    low = kind.lower()
    return any(token in low for token in STRONG_RELATION_KEYWORDS)


def keep_by_edge_filter_preset(row, preset, wide_min_confidence, wide_max_candidate_rank):
    if preset == "all_close":
        return True
    if preset != "strong_plus_wide095_rank1":
        raise ValueError(f"unknown edge_filter_preset: {preset}")

    kind = relation_kind(row)
    if is_strong_relation_kind(kind):
        return True
    if kind in BROAD_RELATION_KINDS:
        return False
    confidence = float(row.get("llm_confidence", 0.0) or 0.0)
    candidate_rank = int(row.get("candidate_rank", 999999) or 999999)
    return confidence >= float(wide_min_confidence) and candidate_rank <= int(wide_max_candidate_rank)


def load_cache_rows(
    path,
    focus_types,
    min_confidence,
    quality_filter,
    edge_filter_preset="all_close",
    wide_min_confidence=0.95,
    wide_max_candidate_rank=1,
):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("llm_relationship_status") != "closely_related":
                continue
            if focus_types and row.get("coarse_type") not in focus_types:
                continue
            if float(row.get("llm_confidence", 0.0)) < min_confidence:
                continue
            if quality_filter == "high" and row.get("evidence_quality") != "high":
                continue
            if not keep_by_edge_filter_preset(
                row,
                preset=edge_filter_preset,
                wide_min_confidence=wide_min_confidence,
                wide_max_candidate_rank=wide_max_candidate_rank,
            ):
                continue
            rows.append(row)
    return rows


def describe(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": round(float(np.mean(vals)), 6),
        "median": round(float(np.median(vals)), 6),
        "min": round(float(np.min(vals)), 6),
        "max": round(float(np.max(vals)), 6),
    }


def pair_cos(emb, pairs):
    out = []
    for a, b in pairs:
        out.append(float((emb[int(a)] * emb[int(b)]).sum().item()))
    return out


def build_virtual_edges(rows, left_ids, right_ids, max_edges_per_positive, max_total_edges):
    left_set = set(int(x) for x in left_ids.detach().cpu().tolist())
    right_set = set(int(x) for x in right_ids.detach().cpu().tolist())
    by_positive = defaultdict(list)
    rejected = Counter()

    for row in rows:
        direction = row.get("direction")
        positive_id = int(row["positive_id"])
        candidate_id = int(row["candidate_id"])
        if positive_id == candidate_id:
            rejected["self_loop"] += 1
            continue
        if direction == "left_to_right":
            if positive_id not in right_set or candidate_id not in right_set:
                rejected["not_same_right_kg"] += 1
                continue
        elif direction == "right_to_left":
            if positive_id not in left_set or candidate_id not in left_set:
                rejected["not_same_left_kg"] += 1
                continue
        else:
            rejected["bad_direction"] += 1
            continue
        key = (direction, positive_id)
        by_positive[key].append(row)

    selected = []
    for key in sorted(by_positive.keys(), key=lambda x: (x[0], x[1])):
        candidates = sorted(
            by_positive[key],
            key=lambda r: (-float(r.get("llm_confidence", 0.0)), int(r.get("candidate_rank", 999999))),
        )
        selected.extend(candidates[: max(1, max_edges_per_positive)])
        if max_total_edges > 0 and len(selected) >= max_total_edges:
            selected = selected[:max_total_edges]
            break

    edge_pairs = []
    seen = set()
    edge_rows = []
    for row in selected:
        pair = tuple(sorted((int(row["positive_id"]), int(row["candidate_id"]))))
        if pair in seen:
            continue
        seen.add(pair)
        edge_pairs.append(pair)
        edge_rows.append(row)

    return edge_pairs, edge_rows, dict(rejected)


def graph_stats(edge_pairs, left_ids, right_ids, original_triples):
    left_set = set(int(x) for x in left_ids.detach().cpu().tolist())
    right_set = set(int(x) for x in right_ids.detach().cpu().tolist())
    original_pairs = set(tuple(sorted((int(h), int(t)))) for h, _, t in original_triples if int(h) != int(t))
    degree_add = Counter()
    cross_kg = 0
    already_existing = 0
    for a, b in edge_pairs:
        if (a in left_set and b in right_set) or (a in right_set and b in left_set):
            cross_kg += 1
        if tuple(sorted((a, b))) in original_pairs:
            already_existing += 1
        degree_add[a] += 1
        degree_add[b] += 1
    return {
        "virtual_edge_count": len(edge_pairs),
        "cross_kg_edge_count": cross_kg,
        "already_existing_undirected_count": already_existing,
        "added_degree": describe(degree_add.values()),
        "max_added_degree_entities": [{"entity_id": int(k), "added_degree": int(v)} for k, v in degree_add.most_common(10)],
    }


def build_augmented_adj(ent_num, original_triples, edge_pairs):
    virtual_relation_id = -999999
    augmented = list(original_triples)
    augmented.extend((int(a), virtual_relation_id, int(b)) for a, b in edge_pairs)
    return get_adjr(ent_num, augmented, norm=True)


def load_model(args, kgs, adj=None):
    if adj is not None:
        kgs = dict(kgs)
        kgs["adj"] = adj
    model = SGMEA(kgs, args)
    if args.model_name_save:
        save_path = os.path.join(args.data_path, args.model_name, "save", f"{args.model_name_save}.pkl")
        if not os.path.exists(save_path):
            raise FileNotFoundError(save_path)
        state = torch.load(save_path, map_location=args.device)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.cuda()
    model.eval()
    return model


def get_embeddings(model):
    with torch.no_grad():
        joint, _ = model.joint_emb_generat()
        joint = F.normalize(joint)
        gph, img, rel, att, name, char, gat_img, gat_att, gat_rel, gat_name, gat_char, full_joint, _ = model.joint_emb_generat(only_joint=False)
        views = {
            "joint": joint.detach(),
            "gph": F.normalize(gph.detach()) if gph is not None else None,
            "gat_att": F.normalize(gat_att.detach()) if gat_att is not None else None,
            "gat_img": F.normalize(gat_img.detach()) if gat_img is not None else None,
            "att": F.normalize(att.detach()) if att is not None else None,
            "rel": F.normalize(rel.detach()) if rel is not None else None,
            "img": F.normalize(img.detach()) if img is not None else None,
        }
    return views


def rank_profile(emb, rows, left_ids, right_ids, args, sample_limit):
    left_set = set(int(x) for x in left_ids.detach().cpu().tolist())
    right_set = set(int(x) for x in right_ids.detach().cpu().tolist())
    eval_rows = rows[:sample_limit] if sample_limit and sample_limit > 0 else rows
    ranks = []
    cand_ranks = []
    positive_dists = []
    candidate_dists = []
    improved_candidate_margin = []
    with torch.no_grad():
        for row in eval_rows:
            anchor_id = int(row["anchor_id"])
            positive_id = int(row["positive_id"])
            candidate_id = int(row["candidate_id"])
            direction = row["direction"]
            if direction == "left_to_right":
                if anchor_id not in left_set or positive_id not in right_set or candidate_id not in right_set:
                    continue
                opposite = right_ids
            else:
                if anchor_id not in right_set or positive_id not in left_set or candidate_id not in left_set:
                    continue
                opposite = left_ids
            gt_local = (opposite == positive_id).nonzero(as_tuple=False)
            cand_local = (opposite == candidate_id).nonzero(as_tuple=False)
            if gt_local.numel() == 0 or cand_local.numel() == 0:
                continue
            gt_idx = int(gt_local.flatten()[0].item())
            cand_idx = int(cand_local.flatten()[0].item())
            anchor = torch.tensor([anchor_id], dtype=torch.long, device=emb.device)
            distance = pairwise_distances(emb[anchor], emb[opposite])
            if args.csls and min(distance.shape[0], distance.shape[1]) >= args.csls_k:
                distance = 1 - csls_sim(1 - distance, args.csls_k)
            _, indices = torch.sort(distance[0], descending=False)
            rank = int((indices == gt_idx).nonzero(as_tuple=False).flatten()[0].item()) + 1
            cand_rank = int((indices == cand_idx).nonzero(as_tuple=False).flatten()[0].item()) + 1
            pos_d = float(distance[0, gt_idx].item())
            cand_d = float(distance[0, cand_idx].item())
            ranks.append(rank)
            cand_ranks.append(cand_rank)
            positive_dists.append(pos_d)
            candidate_dists.append(cand_d)
            improved_candidate_margin.append(cand_d - pos_d)
    return {
        "row_count": len(ranks),
        "gt_rank": describe(ranks),
        "candidate_rank": describe(cand_ranks),
        "positive_distance": describe(positive_dists),
        "candidate_distance": describe(candidate_dists),
        "candidate_minus_positive_distance": describe(improved_candidate_margin),
        "hits1": round(float(sum(1 for r in ranks if r <= 1) / len(ranks)), 6) if ranks else 0.0,
        "hits10": round(float(sum(1 for r in ranks if r <= 10) / len(ranks)), 6) if ranks else 0.0,
    }


def compare_rank_profiles(base_profile, aug_profile):
    return {
        "gt_rank_mean_delta": round(aug_profile["gt_rank"]["mean"] - base_profile["gt_rank"]["mean"], 6),
        "candidate_rank_mean_delta": round(aug_profile["candidate_rank"]["mean"] - base_profile["candidate_rank"]["mean"], 6),
        "candidate_minus_positive_distance_delta": round(
            aug_profile["candidate_minus_positive_distance"]["mean"] - base_profile["candidate_minus_positive_distance"]["mean"], 6
        ),
        "hits1_delta": round(aug_profile["hits1"] - base_profile["hits1"], 6),
        "hits10_delta": round(aug_profile["hits10"] - base_profile["hits10"], 6),
    }


def main():
    audit_args, args = parse_args()
    os.environ.setdefault("LABSE_PATH", "/gly/tongqiang/dongyufeng/models/LaBSE")
    set_seed(args.random_seed)
    torch.cuda.set_device(args.gpu)
    logger = SimpleLogger()
    kgs, _, train_set, _, test_set, _ = load_data(logger, args)
    left_ids = kgs["left_entity_ids"].cuda()
    right_ids = kgs["right_entity_ids"].cuda()
    file_dir = os.path.join(args.data_path, args.data_choice, args.data_split)
    _, _, original_triples, _, _, _ = read_raw_data(file_dir, [1, 2])

    focus_types = set(x.strip() for x in audit_args.focus_coarse_types.split(",") if x.strip())
    rows = load_cache_rows(
        audit_args.cache_jsonl,
        focus_types=focus_types,
        min_confidence=audit_args.min_confidence,
        quality_filter=audit_args.quality_filter,
        edge_filter_preset=audit_args.edge_filter_preset,
        wide_min_confidence=audit_args.wide_min_confidence,
        wide_max_candidate_rank=audit_args.wide_max_candidate_rank,
    )
    edge_pairs, edge_rows, rejected = build_virtual_edges(
        rows=rows,
        left_ids=left_ids,
        right_ids=right_ids,
        max_edges_per_positive=audit_args.max_edges_per_positive,
        max_total_edges=audit_args.max_total_edges,
    )

    base_model = load_model(args, kgs)
    aug_adj = build_augmented_adj(kgs["ent_num"], original_triples, edge_pairs)
    aug_model = load_model(args, kgs, adj=aug_adj)

    base_views = get_embeddings(base_model)
    aug_views = get_embeddings(aug_model)

    positive_close_pairs = [(int(r["positive_id"]), int(r["candidate_id"])) for r in edge_rows]
    train_positive_pairs = [(int(a), int(b)) for a, b in train_set.data]
    safe_rows = []
    with open(audit_args.cache_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("llm_relationship_status") == "safe_negative" and (not focus_types or row.get("coarse_type") in focus_types):
                safe_rows.append(row)
    safe_pairs = [(int(r["positive_id"]), int(r["candidate_id"])) for r in safe_rows if r.get("direction") in ("left_to_right", "right_to_left")]

    view_summary = {}
    for view_name, base_emb in base_views.items():
        if base_emb is None or aug_views.get(view_name) is None:
            continue
        aug_emb = aug_views[view_name]
        close_base = pair_cos(base_emb, positive_close_pairs)
        close_aug = pair_cos(aug_emb, positive_close_pairs)
        safe_base = pair_cos(base_emb, safe_pairs[: len(positive_close_pairs)])
        safe_aug = pair_cos(aug_emb, safe_pairs[: len(positive_close_pairs)])
        pos_base = pair_cos(base_emb, train_positive_pairs)
        pos_aug = pair_cos(aug_emb, train_positive_pairs)
        view_summary[view_name] = {
            "close_cos_base": describe(close_base),
            "close_cos_aug": describe(close_aug),
            "close_cos_delta": describe([a - b for a, b in zip(close_aug, close_base)]),
            "safe_cos_base": describe(safe_base),
            "safe_cos_aug": describe(safe_aug),
            "safe_cos_delta": describe([a - b for a, b in zip(safe_aug, safe_base)]),
            "train_positive_cos_base": describe(pos_base),
            "train_positive_cos_aug": describe(pos_aug),
            "train_positive_cos_delta": describe([a - b for a, b in zip(pos_aug, pos_base)]),
        }

    base_rank = rank_profile(base_views["joint"], edge_rows, left_ids, right_ids, args, audit_args.sample_limit)
    aug_rank = rank_profile(aug_views["joint"], edge_rows, left_ids, right_ids, args, audit_args.sample_limit)

    summary = {
        "config": {
            "cache_jsonl": audit_args.cache_jsonl,
            "focus_coarse_types": sorted(focus_types),
            "min_confidence": audit_args.min_confidence,
            "quality_filter": audit_args.quality_filter,
            "edge_filter_preset": audit_args.edge_filter_preset,
            "wide_min_confidence": audit_args.wide_min_confidence,
            "wide_max_candidate_rank": audit_args.wide_max_candidate_rank,
            "max_edges_per_positive": audit_args.max_edges_per_positive,
            "max_total_edges": audit_args.max_total_edges,
            "sample_limit": audit_args.sample_limit,
            "model_name_save": args.model_name_save,
            "structure_encoder": args.structure_encoder,
        },
        "cache_rows_after_filter": len(rows),
        "selected_edge_rows": len(edge_rows),
        "rejected_rows": rejected,
        "graph_stats": graph_stats(edge_pairs, left_ids, right_ids, original_triples),
        "edge_row_breakdown": {
            "coarse_type": dict(Counter(r.get("coarse_type") for r in edge_rows)),
            "direction": dict(Counter(r.get("direction") for r in edge_rows)),
            "relationship_kind": dict(Counter(r.get("llm_relationship_kind") for r in edge_rows).most_common(30)),
        },
        "view_similarity": view_summary,
        "rank_profile_base": base_rank,
        "rank_profile_aug": aug_rank,
        "rank_profile_delta": compare_rank_profiles(base_rank, aug_rank),
    }

    os.makedirs(os.path.dirname(audit_args.output_summary_json), exist_ok=True)
    with open(audit_args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary["graph_stats"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary["rank_profile_delta"], ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {audit_args.output_summary_json}", flush=True)


if __name__ == "__main__":
    main()
