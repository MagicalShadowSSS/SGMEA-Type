import json
import os
from collections import defaultdict

import torch


VALID_RELATION_STATUSES = {"safe_negative", "closely_related"}


def load_relation_aware_hnm_cache(cache_jsonl, ent_num):
    if not cache_jsonl:
        raise ValueError("relation-aware HNM cache path is empty")
    if not os.path.exists(cache_jsonl):
        raise FileNotFoundError(f"relation-aware HNM cache not found: {cache_jsonl}")

    index = {
        "left_to_right": defaultdict(lambda: {"safe_negative": [], "closely_related": []}),
        "right_to_left": defaultdict(lambda: {"safe_negative": [], "closely_related": []}),
    }
    stats = {
        "row_count": 0,
        "status_counts": defaultdict(int),
        "direction_counts": defaultdict(int),
        "anchor_count_by_direction": defaultdict(set),
    }

    with open(cache_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            direction = str(row.get("direction", "")).strip()
            status = str(row.get("llm_relationship_status", "")).strip()
            if direction not in index or status not in VALID_RELATION_STATUSES:
                continue

            anchor_id = int(row["anchor_id"])
            positive_id = int(row["positive_id"])
            candidate_id = int(row["candidate_id"])
            if not (0 <= anchor_id < ent_num and 0 <= positive_id < ent_num and 0 <= candidate_id < ent_num):
                raise ValueError(
                    f"relation-aware HNM id out of range: anchor={anchor_id} "
                    f"positive={positive_id} candidate={candidate_id} ent_num={ent_num}"
                )

            item = {
                "candidate_id": candidate_id,
                "positive_id": positive_id,
                "candidate_rank": int(row.get("candidate_rank", 1)),
                "llm_confidence": float(row.get("llm_confidence", 0.0)),
                "coarse_type": row.get("coarse_type", ""),
            }
            index[direction][anchor_id][status].append(item)
            stats["row_count"] += 1
            stats["status_counts"][status] += 1
            stats["direction_counts"][direction] += 1
            stats["anchor_count_by_direction"][direction].add(anchor_id)

    for direction_map in index.values():
        for pools in direction_map.values():
            for items in pools.values():
                items.sort(
                    key=lambda item: (
                        int(item.get("candidate_rank", 1)),
                        -float(item.get("llm_confidence", 0.0)),
                        int(item.get("candidate_id", -1)),
                    )
                )

    stats["anchor_count_by_direction"] = {
        direction: len(anchor_ids) for direction, anchor_ids in stats["anchor_count_by_direction"].items()
    }
    stats["status_counts"] = dict(stats["status_counts"])
    stats["direction_counts"] = dict(stats["direction_counts"])
    return index, stats


def build_relation_aware_batch(
    batch_tensor,
    cache_index,
    mask_close,
    use_soft_weight,
    safe_weight,
    close_weight,
    device,
):
    batch_cpu = batch_tensor.detach().cpu()
    batch_left = batch_cpu[:, 0].tolist()
    batch_right = batch_cpu[:, 1].tolist()
    batch_size = len(batch_left)

    if batch_size == 0:
        return {
            "mask_ab": None,
            "mask_ba": None,
            "weight_ab": None,
            "weight_ba": None,
            "stats": _empty_stats(),
        }

    left_positions = defaultdict(list)
    right_positions = defaultdict(list)
    for idx, ent_id in enumerate(batch_left):
        left_positions[int(ent_id)].append(idx)
    for idx, ent_id in enumerate(batch_right):
        right_positions[int(ent_id)].append(idx)

    mask_ab = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device) if mask_close else None
    mask_ba = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device) if mask_close else None
    weight_ab = torch.ones((batch_size, batch_size), dtype=torch.float32, device=device) if use_soft_weight else None
    weight_ba = torch.ones((batch_size, batch_size), dtype=torch.float32, device=device) if use_soft_weight else None
    positive_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

    hit_anchor_keys = set()
    shield_hits = 0
    safe_weight_hits = 0
    close_weight_hits = 0

    for row_idx, (left_id, right_id) in enumerate(zip(batch_left, batch_right)):
        ltr_pools = cache_index["left_to_right"].get(int(left_id))
        if ltr_pools is not None:
            if ltr_pools["safe_negative"] or ltr_pools["closely_related"]:
                hit_anchor_keys.add(("left_to_right", int(left_id)))
            shield_hits += _apply_relation_entries(
                row_idx=row_idx,
                items=ltr_pools["closely_related"],
                positions=right_positions,
                mask=mask_ab,
                weight=weight_ab,
                relation_weight=float(close_weight),
            )[0]
            if use_soft_weight:
                safe_weight_hits += _apply_relation_entries(
                    row_idx=row_idx,
                    items=ltr_pools["safe_negative"],
                    positions=right_positions,
                    mask=None,
                    weight=weight_ab,
                    relation_weight=float(safe_weight),
                )[1]
                close_weight_hits += _count_batch_hits(ltr_pools["closely_related"], right_positions)

        rtl_pools = cache_index["right_to_left"].get(int(right_id))
        if rtl_pools is not None:
            if rtl_pools["safe_negative"] or rtl_pools["closely_related"]:
                hit_anchor_keys.add(("right_to_left", int(right_id)))
            shield_hits += _apply_relation_entries(
                row_idx=row_idx,
                items=rtl_pools["closely_related"],
                positions=left_positions,
                mask=mask_ba,
                weight=weight_ba,
                relation_weight=float(close_weight),
            )[0]
            if use_soft_weight:
                safe_weight_hits += _apply_relation_entries(
                    row_idx=row_idx,
                    items=rtl_pools["safe_negative"],
                    positions=left_positions,
                    mask=None,
                    weight=weight_ba,
                    relation_weight=float(safe_weight),
                )[1]
                close_weight_hits += _count_batch_hits(rtl_pools["closely_related"], left_positions)

    if mask_ab is not None:
        mask_ab = mask_ab & (~positive_mask)
    if mask_ba is not None:
        mask_ba = mask_ba & (~positive_mask)
    if weight_ab is not None:
        weight_ab = weight_ab.masked_fill(positive_mask, 1.0)
    if weight_ba is not None:
        weight_ba = weight_ba.masked_fill(positive_mask, 1.0)
    probe_stats = _build_probe_stats(mask_ab, mask_ba, weight_ab, weight_ba, positive_mask)
    if probe_stats["positive_mask_violation_count"] > 0 or probe_stats["positive_weight_violation_max_abs"] > 1e-6:
        raise RuntimeError(
            "Relation-aware HNM probe failed: positive diagonal was modified "
            f"(mask_violations={probe_stats['positive_mask_violation_count']}, "
            f"weight_max_abs={probe_stats['positive_weight_violation_max_abs']})"
        )

    return {
        "mask_ab": mask_ab,
        "mask_ba": mask_ba,
        "weight_ab": weight_ab,
        "weight_ba": weight_ba,
        "stats": {
            "batch_hit_anchor_count": float(len(hit_anchor_keys)),
            "shield_mask_hit_count": float(shield_hits),
            "safe_weight_hit_count": float(safe_weight_hits),
            "close_weight_hit_count": float(close_weight_hits),
            "safe_denominator_weight": float(safe_weight),
            "close_denominator_weight": float(close_weight),
            **probe_stats,
        },
    }


def _empty_stats():
    return {
        "batch_hit_anchor_count": 0.0,
        "shield_mask_hit_count": 0.0,
        "safe_weight_hit_count": 0.0,
        "close_weight_hit_count": 0.0,
        "safe_denominator_weight": 1.0,
        "close_denominator_weight": 1.0,
        "positive_mask_violation_count": 0.0,
        "positive_weight_violation_max_abs": 0.0,
        "mask_weight_overlap_count": 0.0,
        "cross_weight_nonunit_count": 0.0,
        "cross_weight_min": 1.0,
        "cross_weight_max": 1.0,
        "cross_weight_mean": 1.0,
    }


def _build_probe_stats(mask_ab, mask_ba, weight_ab, weight_ba, positive_mask):
    positive_mask_violation_count = 0
    positive_weight_violation_max_abs = 0.0
    mask_weight_overlap_count = 0
    nonunit_weights = []

    for mask in (mask_ab, mask_ba):
        if mask is not None:
            positive_mask_violation_count += int((mask & positive_mask).sum().item())

    non_positive_mask = ~positive_mask
    for weight in (weight_ab, weight_ba):
        if weight is None:
            continue
        diag_delta = (weight[positive_mask] - 1.0).abs()
        if diag_delta.numel() > 0:
            positive_weight_violation_max_abs = max(
                positive_weight_violation_max_abs,
                float(diag_delta.max().item()),
            )
        changed = non_positive_mask & ((weight - 1.0).abs() > 1e-6)
        if changed.any():
            nonunit_weights.append(weight[changed].detach())

    for mask, weight in ((mask_ab, weight_ab), (mask_ba, weight_ba)):
        if mask is not None and weight is not None:
            overlap = mask & non_positive_mask & ((weight - 1.0).abs() > 1e-6)
            mask_weight_overlap_count += int(overlap.sum().item())

    if nonunit_weights:
        values = torch.cat(nonunit_weights)
        cross_weight_nonunit_count = int(values.numel())
        cross_weight_min = float(values.min().item())
        cross_weight_max = float(values.max().item())
        cross_weight_mean = float(values.mean().item())
    else:
        cross_weight_nonunit_count = 0
        cross_weight_min = 1.0
        cross_weight_max = 1.0
        cross_weight_mean = 1.0

    return {
        "positive_mask_violation_count": float(positive_mask_violation_count),
        "positive_weight_violation_max_abs": float(positive_weight_violation_max_abs),
        "mask_weight_overlap_count": float(mask_weight_overlap_count),
        "cross_weight_nonunit_count": float(cross_weight_nonunit_count),
        "cross_weight_min": float(cross_weight_min),
        "cross_weight_max": float(cross_weight_max),
        "cross_weight_mean": float(cross_weight_mean),
    }


def _count_batch_hits(items, positions):
    count = 0
    for item in items:
        count += len(positions.get(int(item["candidate_id"]), []))
    return count


def _apply_relation_entries(row_idx, items, positions, mask, weight, relation_weight):
    mask_hits = 0
    weight_hits = 0
    for item in items:
        for col_idx in positions.get(int(item["candidate_id"]), []):
            if mask is not None:
                mask[row_idx, col_idx] = True
                mask_hits += 1
            if weight is not None:
                weight[row_idx, col_idx] = float(relation_weight)
                weight_hits += 1
    return mask_hits, weight_hits
