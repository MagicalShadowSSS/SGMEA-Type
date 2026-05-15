import torch
import torch.nn.functional as F


SUPPORTED_TAG_HNM_TYPES = ("Person", "Organization", "Place", "Creative Work")


def _as_long_tensor(values, device):
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.long)
    return torch.as_tensor(values, device=device, dtype=torch.long)


def _type_ids_from_names(type_names, supported_type_names):
    supported = set(supported_type_names)
    return {name: idx for idx, name in enumerate(type_names) if name in supported}


def _safe_float(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 0:
            return float("nan")
        return float(value.float().mean().item())
    return float(value)


def _summarize_type_cache(cache, sim_cache, rows, type_name):
    if rows.numel() == 0:
        return {
            "type": type_name,
            "rows": 0,
            "avg_valid_negatives": 0.0,
            "empty_ratio": 1.0,
            "avg_similarity": float("nan"),
            "avg_top1_similarity": float("nan"),
            "avg_topk_similarity": float("nan"),
            "p50_top1_similarity": float("nan"),
            "p90_top1_similarity": float("nan"),
            "max_top1_similarity": float("nan"),
        }

    type_cache = cache[rows]
    type_sim = sim_cache[rows]
    valid = type_cache >= 0
    valid_counts = valid.sum(dim=1).float()
    non_empty = valid_counts > 0
    valid_sim = type_sim[valid]
    top1_sim = type_sim[:, 0][valid[:, 0]] if type_sim.size(1) > 0 else type_sim.new_empty(0)

    if valid.any():
        row_sum = torch.where(valid, type_sim.nan_to_num(nan=0.0), torch.zeros_like(type_sim)).sum(dim=1)
        row_topk = row_sum[non_empty] / valid_counts[non_empty].clamp_min(1.0)
    else:
        row_topk = type_sim.new_empty(0)

    if top1_sim.numel() > 0:
        p50 = torch.quantile(top1_sim.float(), 0.5)
        p90 = torch.quantile(top1_sim.float(), 0.9)
        max_top1 = top1_sim.max()
    else:
        p50 = p90 = max_top1 = type_sim.new_tensor(float("nan"))

    return {
        "type": type_name,
        "rows": int(rows.numel()),
        "avg_valid_negatives": _safe_float(valid_counts),
        "empty_ratio": float((~non_empty).float().mean().item()),
        "avg_similarity": _safe_float(valid_sim),
        "avg_top1_similarity": _safe_float(top1_sim),
        "avg_topk_similarity": _safe_float(row_topk),
        "p50_top1_similarity": _safe_float(p50),
        "p90_top1_similarity": _safe_float(p90),
        "max_top1_similarity": _safe_float(max_top1),
    }


def format_cache_stat(prefix, stat):
    def fmt(value):
        if isinstance(value, int):
            return str(value)
        if value != value:
            return "nan"
        return f"{value:.4f}"

    return (
        f"{prefix}[{stat['type']}] "
        f"rows={stat['rows']} "
        f"avg_valid={fmt(stat['avg_valid_negatives'])} "
        f"avg_sim={fmt(stat['avg_similarity'])} "
        f"avg_top1={fmt(stat['avg_top1_similarity'])} "
        f"avg_topk={fmt(stat['avg_topk_similarity'])} "
        f"p50_top1={fmt(stat['p50_top1_similarity'])} "
        f"p90_top1={fmt(stat['p90_top1_similarity'])} "
        f"max_top1={fmt(stat['max_top1_similarity'])} "
        f"empty_ratio={fmt(stat['empty_ratio'])}"
    )


def build_one_direction_hard_negative_cache(
    anchor_ids,
    positive_ids,
    candidate_ids,
    emb,
    entity_type_ids,
    type_names,
    k=32,
    supported_type_names=SUPPORTED_TAG_HNM_TYPES,
    chunk_size=2048,
):
    device = emb.device
    anchor_ids = _as_long_tensor(anchor_ids, device)
    positive_ids = _as_long_tensor(positive_ids, device)
    candidate_ids = _as_long_tensor(candidate_ids, device)
    entity_type_ids = _as_long_tensor(entity_type_ids, device)
    type_id_by_name = _type_ids_from_names(type_names, supported_type_names)

    num_rows = anchor_ids.numel()
    cache = torch.full((num_rows, k), -1, dtype=torch.long, device=device)
    sim_cache = torch.full((num_rows, k), float("nan"), dtype=emb.dtype, device=device)
    emb = F.normalize(emb.detach(), dim=1)

    per_type_stats = {}
    with torch.no_grad():
        anchor_type_ids = entity_type_ids[anchor_ids]
        candidate_type_ids = entity_type_ids[candidate_ids]

        for type_name, type_id in type_id_by_name.items():
            rows = torch.nonzero(anchor_type_ids == type_id, as_tuple=False).flatten()
            typed_candidates = candidate_ids[candidate_type_ids == type_id]

            if rows.numel() > 0 and typed_candidates.numel() > 0:
                k_eff = min(k, int(typed_candidates.numel()))
                cand_emb = emb[typed_candidates]
                for start in range(0, int(rows.numel()), chunk_size):
                    row_chunk = rows[start:start + chunk_size]
                    anchor_chunk = anchor_ids[row_chunk]
                    positive_chunk = positive_ids[row_chunk]
                    sim = emb[anchor_chunk].matmul(cand_emb.t())
                    positive_mask = typed_candidates.unsqueeze(0).eq(positive_chunk.unsqueeze(1))
                    sim = sim.masked_fill(positive_mask, float("-inf"))
                    top_sim, top_pos = torch.topk(sim, k=k_eff, dim=1)
                    top_ids = typed_candidates[top_pos]
                    valid = torch.isfinite(top_sim)
                    cache[row_chunk, :k_eff] = torch.where(valid, top_ids, torch.full_like(top_ids, -1))
                    sim_cache[row_chunk, :k_eff] = torch.where(
                        valid,
                        top_sim,
                        torch.full_like(top_sim, float("nan")),
                    )
                    del sim, top_sim, top_pos, top_ids, valid, positive_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del cand_emb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            per_type_stats[type_name] = _summarize_type_cache(cache, sim_cache, rows, type_name)

    supported_mask = torch.zeros(num_rows, dtype=torch.bool, device=device)
    for type_id in type_id_by_name.values():
        supported_mask |= entity_type_ids[anchor_ids] == type_id
    supported_rows = torch.nonzero(supported_mask, as_tuple=False).flatten()
    overall = _summarize_type_cache(cache, sim_cache, supported_rows, "overall")
    overall["supported_rows"] = int(supported_rows.numel())
    overall["total_rows"] = int(num_rows)

    return cache, sim_cache, {"per_type": per_type_stats, "overall": overall}


def build_bidirectional_hard_negative_cache(
    train_links,
    left_entity_ids,
    right_entity_ids,
    emb,
    entity_type_ids,
    type_names,
    k=32,
    supported_type_names=SUPPORTED_TAG_HNM_TYPES,
    chunk_size=2048,
):
    device = emb.device
    train_links = _as_long_tensor(train_links, device)
    left_entity_ids = _as_long_tensor(left_entity_ids, device)
    right_entity_ids = _as_long_tensor(right_entity_ids, device)

    left_ids = train_links[:, 0]
    right_ids = train_links[:, 1]
    right_cache, right_sim, right_stats = build_one_direction_hard_negative_cache(
        anchor_ids=left_ids,
        positive_ids=right_ids,
        candidate_ids=right_entity_ids,
        emb=emb,
        entity_type_ids=entity_type_ids,
        type_names=type_names,
        k=k,
        supported_type_names=supported_type_names,
        chunk_size=chunk_size,
    )
    left_cache, left_sim, left_stats = build_one_direction_hard_negative_cache(
        anchor_ids=right_ids,
        positive_ids=left_ids,
        candidate_ids=left_entity_ids,
        emb=emb,
        entity_type_ids=entity_type_ids,
        type_names=type_names,
        k=k,
        supported_type_names=supported_type_names,
        chunk_size=chunk_size,
    )

    return {
        "right_cache": right_cache,
        "left_cache": left_cache,
        "right_sim": right_sim,
        "left_sim": left_sim,
        "stats": {
            "right": right_stats,
            "left": left_stats,
        },
    }
