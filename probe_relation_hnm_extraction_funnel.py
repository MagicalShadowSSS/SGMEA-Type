import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from extract_llm_hnm_cases import (
    SimpleLogger,
    enrich_case,
    load_attribute_kv,
    load_checkpoint,
    load_full_ill,
    load_id_name_maps,
    set_seed,
)
from model import SGMEA
from src.data import load_data


SUPPORTED_TYPES = ("Place", "Creative Work", "Organization")


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tag_hnm_k", type=int, default=32)
    parser.add_argument("--tag_hnm_chunk_size", type=int, default=2048)
    parser.add_argument("--probe_types", type=str, default="Place,Creative Work,Organization")
    parser.add_argument("--llm_hnm_threshold", type=float, default=0.55)
    parser.add_argument("--llm_hnm_fallback_threshold", type=float, default=0.50)
    parser.add_argument("--llm_hnm_min_cases", type=int, default=1000)
    parser.add_argument("--llm_hnm_topn", type=int, default=5)
    parser.add_argument("--llm_hnm_max_cases", type=int, default=2500)
    parser.add_argument("--llm_hnm_attr_limit", type=int, default=40)
    parser.add_argument("--llm_hnm_min_attr_items", type=int, default=3)
    parser.add_argument("--llm_hnm_min_combined_attr_items", type=int, default=6)
    parser.add_argument(
        "--llm_hnm_direction",
        default="both",
        choices=["both", "left_to_right", "right_to_left"],
    )
    parser.add_argument(
        "--llm_hnm_sort_by",
        default="top1_sim_desc",
        choices=["top1_sim_desc", "random"],
    )
    parser.add_argument("--llm_hnm_seed", type=int, default=42)
    parser.add_argument("--output_summary_json", type=str, default="")
    probe_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        train_cfg = cfg()
        train_cfg.get_args()
        args = train_cfg.update_train_configs()
    finally:
        sys.argv = original_argv

    return probe_args, args


def as_long_tensor(values, device):
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.long)
    return torch.as_tensor(values, device=device, dtype=torch.long)


def quantiles(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def build_one_direction_cache(
    anchor_ids,
    positive_ids,
    candidate_ids,
    emb,
    entity_type_ids,
    type_names,
    k,
    chunk_size,
):
    device = emb.device
    anchor_ids = as_long_tensor(anchor_ids, device)
    positive_ids = as_long_tensor(positive_ids, device)
    candidate_ids = as_long_tensor(candidate_ids, device)
    entity_type_ids = as_long_tensor(entity_type_ids, device)

    num_rows = anchor_ids.numel()
    cache = torch.full((num_rows, k), -1, dtype=torch.long, device=device)
    sim_cache = torch.full((num_rows, k), float("nan"), dtype=emb.dtype, device=device)
    emb = F.normalize(emb.detach(), dim=1)

    type_id_by_name = {name: idx for idx, name in enumerate(type_names)}
    with torch.no_grad():
        anchor_type_ids = entity_type_ids[anchor_ids]
        candidate_type_ids = entity_type_ids[candidate_ids]
        for type_name in SUPPORTED_TYPES:
            type_id = type_id_by_name.get(type_name)
            if type_id is None:
                continue
            rows = torch.nonzero(anchor_type_ids == type_id, as_tuple=False).flatten()
            typed_candidates = candidate_ids[candidate_type_ids == type_id]
            if rows.numel() == 0 or typed_candidates.numel() == 0:
                continue
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
            del cand_emb
    return cache, sim_cache


def direction_specs(probe_args):
    specs = []
    if probe_args.llm_hnm_direction in ["both", "left_to_right"]:
        specs.append(("left_to_right", "right", 0, 1))
    if probe_args.llm_hnm_direction in ["both", "right_to_left"]:
        specs.append(("right_to_left", "left", 1, 0))
    return specs


def init_type_stats(type_names):
    return {
        name: {
            "train_anchor_count": 0,
            "cache_non_empty_anchor_count": 0,
            "pass_threshold_anchor_count": 0,
            "pass_fallback_anchor_count": 0,
            "candidate_rows_at_threshold": 0,
            "candidate_rows_at_fallback": 0,
            "after_budget_rows": 0,
            "after_budget_unique_anchor_count": 0,
            "evidence_high_rows": 0,
            "evidence_low_rows": 0,
            "write_all_rows": 0,
            "write_high_rows": 0,
            "write_auto_rows": 0,
            "top1_sim_quantiles": None,
        }
        for name in type_names
    }


def add_case_rows(cases_by_type, row, direction, direction_label, train_links_cpu, neg_cache, sim_cache, type_name, topn):
    left_id = int(train_links_cpu[row, 0].item())
    right_id = int(train_links_cpu[row, 1].item())
    if direction == "right":
        anchor_id = left_id
        positive_id = right_id
    else:
        anchor_id = right_id
        positive_id = left_id

    top1_sim = float(sim_cache[row, 0].item())
    for idx in range(topn):
        negative_id = int(neg_cache[row, idx].item())
        if negative_id < 0:
            continue
        cases_by_type[type_name].append(
            {
                "row": int(row),
                "candidate_rank": idx + 1,
                "candidate_sim": float(sim_cache[row, idx].item()),
                "topn_limit": topn,
                "direction": direction,
                "direction_label": direction_label,
                "coarse_type": type_name,
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "top1_sim": top1_sim,
            }
        )


def sort_cases(cases, probe_args):
    if probe_args.llm_hnm_sort_by == "top1_sim_desc":
        return sorted(cases, key=lambda item: (-item["top1_sim"], item["candidate_rank"]))
    rng = random.Random(probe_args.llm_hnm_seed)
    cases = list(cases)
    rng.shuffle(cases)
    return cases


def summarize_funnel(cache_by_direction, train_links, entity_type_ids, type_names, probe_args):
    requested_types = [name.strip() for name in probe_args.probe_types.split(",") if name.strip()]
    stats = init_type_stats(requested_types)
    by_direction = defaultdict(lambda: init_type_stats(requested_types))
    top1_values = {name: [] for name in requested_types}
    cases_at_threshold = defaultdict(list)
    cases_at_fallback = defaultdict(list)
    type_ids_cpu = entity_type_ids.detach().cpu()
    train_links_cpu = train_links.detach().cpu()
    type_id_by_name = {name: idx for idx, name in enumerate(type_names)}

    for direction_label, direction, anchor_col, _positive_col in direction_specs(probe_args):
        neg_cache, sim_cache = cache_by_direction[direction]
        neg_cache = neg_cache.detach().cpu()
        sim_cache = sim_cache.detach().cpu()
        topn = min(int(probe_args.llm_hnm_topn), int(neg_cache.shape[1]), int(sim_cache.shape[1]))
        anchor_ids = train_links_cpu[:, anchor_col]
        valid_first = neg_cache[:, 0] >= 0

        for type_name in requested_types:
            type_id = type_id_by_name.get(type_name)
            if type_id is None:
                continue
            type_rows = torch.nonzero(type_ids_cpu[anchor_ids] == type_id, as_tuple=False).flatten()
            non_empty_rows = type_rows[valid_first[type_rows]]
            threshold_rows = non_empty_rows[sim_cache[non_empty_rows, 0] >= probe_args.llm_hnm_threshold]
            fallback_rows = non_empty_rows[sim_cache[non_empty_rows, 0] >= probe_args.llm_hnm_fallback_threshold]

            direction_stats = by_direction[direction_label][type_name]
            direction_stats["train_anchor_count"] += int(type_rows.numel())
            direction_stats["cache_non_empty_anchor_count"] += int(non_empty_rows.numel())
            direction_stats["pass_threshold_anchor_count"] += int(threshold_rows.numel())
            direction_stats["pass_fallback_anchor_count"] += int(fallback_rows.numel())

            stats[type_name]["train_anchor_count"] += int(type_rows.numel())
            stats[type_name]["cache_non_empty_anchor_count"] += int(non_empty_rows.numel())
            stats[type_name]["pass_threshold_anchor_count"] += int(threshold_rows.numel())
            stats[type_name]["pass_fallback_anchor_count"] += int(fallback_rows.numel())

            top1 = sim_cache[non_empty_rows, 0].tolist()
            top1_values[type_name].extend(float(v) for v in top1)

            for row in threshold_rows.tolist():
                add_case_rows(cases_at_threshold, row, direction, direction_label, train_links_cpu, neg_cache, sim_cache, type_name, topn)
            for row in fallback_rows.tolist():
                add_case_rows(cases_at_fallback, row, direction, direction_label, train_links_cpu, neg_cache, sim_cache, type_name, topn)

    for type_name in requested_types:
        threshold_cases = sort_cases(cases_at_threshold[type_name], probe_args)
        fallback_cases = sort_cases(cases_at_fallback[type_name], probe_args)
        stats[type_name]["candidate_rows_at_threshold"] = len(threshold_cases)
        stats[type_name]["candidate_rows_at_fallback"] = len(fallback_cases)
        if (
            len(threshold_cases) >= probe_args.llm_hnm_min_cases
            or probe_args.llm_hnm_fallback_threshold >= probe_args.llm_hnm_threshold
        ):
            chosen_cases = threshold_cases
            used_threshold = probe_args.llm_hnm_threshold
        else:
            chosen_cases = fallback_cases
            used_threshold = probe_args.llm_hnm_fallback_threshold
        if probe_args.llm_hnm_max_cases > 0:
            chosen_cases = chosen_cases[:probe_args.llm_hnm_max_cases]
        stats[type_name]["used_threshold_if_extract_default"] = used_threshold
        stats[type_name]["after_budget_rows"] = len(chosen_cases)
        stats[type_name]["after_budget_unique_anchor_count"] = len({case["anchor_id"] for case in chosen_cases})
        stats[type_name]["top1_sim_quantiles"] = quantiles(top1_values[type_name])
        stats[type_name]["_chosen_cases"] = chosen_cases

    return stats, by_direction


def enrich_and_count(stats, args, probe_args):
    file_dir = os.path.join(args.data_path, args.data_choice, args.data_split)
    side_maps = load_id_name_maps(file_dir)
    id_to_attrs = load_attribute_kv(
        file_dir,
        left_name_to_id=side_maps["left"]["name_to_id"],
        right_name_to_id=side_maps["right"]["name_to_id"],
        attr_limit=probe_args.llm_hnm_attr_limit,
    )
    full_ill = load_full_ill(file_dir)

    for type_name, item in stats.items():
        enriched = [
            enrich_case(
                case=case,
                dataset=args.data_choice,
                left_maps=side_maps["left"],
                right_maps=side_maps["right"],
                id_to_attrs=id_to_attrs,
                full_ill=full_ill,
                llm_args=probe_args,
            )
            for case in item.pop("_chosen_cases")
        ]
        high_rows = sum(1 for row in enriched if row["evidence_quality"] == "high")
        low_rows = sum(1 for row in enriched if row["evidence_quality"] == "low")
        item["evidence_high_rows"] = int(high_rows)
        item["evidence_low_rows"] = int(low_rows)
        item["write_all_rows"] = len(enriched)
        item["write_high_rows"] = int(high_rows)
        item["write_auto_rows"] = int(high_rows if high_rows > 0 else len(enriched))
        item["covered_anchor_rate_after_budget"] = (
            item["after_budget_unique_anchor_count"] / item["train_anchor_count"]
            if item["train_anchor_count"] > 0
            else 0.0
        )
        item["cache_non_empty_anchor_rate"] = (
            item["cache_non_empty_anchor_count"] / item["train_anchor_count"]
            if item["train_anchor_count"] > 0
            else 0.0
        )
        item["threshold_anchor_rate"] = (
            item["pass_threshold_anchor_count"] / item["train_anchor_count"]
            if item["train_anchor_count"] > 0
            else 0.0
        )


def patch_offline_text_models():
    try:
        import transformers
        import sentence_transformers
    except Exception:
        return

    class _UnusedTokenizer:
        pass

    original_sentence_transformer = sentence_transformers.SentenceTransformer

    def tokenizer_from_pretrained(*_args, **_kwargs):
        return _UnusedTokenizer()

    def sentence_transformer_from_pretrained(model_name_or_path, *args, **kwargs):
        if model_name_or_path == "sentence-transformers/LaBSE":
            model_name_or_path = os.environ.get(
                "LABSE_PATH",
                "/gly/tongqiang/dongyufeng/models/LaBSE",
            )
        kwargs.setdefault("local_files_only", True)
        return original_sentence_transformer(model_name_or_path, *args, **kwargs)

    transformers.BertTokenizer.from_pretrained = tokenizer_from_pretrained
    sentence_transformers.SentenceTransformer = sentence_transformer_from_pretrained


def main():
    probe_args, args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because SGMEA currently constructs CUDA tensors internally.")

    set_seed(probe_args.llm_hnm_seed)
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)
    logger = SimpleLogger()
    patch_offline_text_models()

    kgs, _non_train, train_set, _eval_set, _test_set, _test_ill = load_data(logger, args)
    model = SGMEA(kgs, args).cuda()
    ckpt_path = load_checkpoint(model, args)
    model.eval()
    logger.info(f"loaded checkpoint: {ckpt_path}")

    with torch.no_grad():
        joint_emb, _ = model.joint_emb_generat()
        train_links = torch.as_tensor(train_set.data, dtype=torch.long, device=joint_emb.device)
        right_cache, right_sim = build_one_direction_cache(
            anchor_ids=train_links[:, 0],
            positive_ids=train_links[:, 1],
            candidate_ids=model.right_entity_ids,
            emb=joint_emb,
            entity_type_ids=model.entity_type_ids,
            type_names=model.top_type_names,
            k=probe_args.tag_hnm_k,
            chunk_size=probe_args.tag_hnm_chunk_size,
        )
        left_cache, left_sim = build_one_direction_cache(
            anchor_ids=train_links[:, 1],
            positive_ids=train_links[:, 0],
            candidate_ids=model.left_entity_ids,
            emb=joint_emb,
            entity_type_ids=model.entity_type_ids,
            type_names=model.top_type_names,
            k=probe_args.tag_hnm_k,
            chunk_size=probe_args.tag_hnm_chunk_size,
        )

    stats, by_direction = summarize_funnel(
        cache_by_direction={
            "right": (right_cache, right_sim),
            "left": (left_cache, left_sim),
        },
        train_links=train_links,
        entity_type_ids=model.entity_type_ids,
        type_names=model.top_type_names,
        probe_args=probe_args,
    )
    enrich_and_count(stats, args, probe_args)

    summary = {
        "data_choice": args.data_choice,
        "data_rate": args.data_rate,
        "checkpoint": ckpt_path,
        "threshold": probe_args.llm_hnm_threshold,
        "fallback_threshold": probe_args.llm_hnm_fallback_threshold,
        "min_cases_rows": probe_args.llm_hnm_min_cases,
        "topn": probe_args.llm_hnm_topn,
        "max_cases_rows": probe_args.llm_hnm_max_cases,
        "direction": probe_args.llm_hnm_direction,
        "by_type": stats,
        "by_direction": by_direction,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if probe_args.output_summary_json:
        os.makedirs(os.path.dirname(probe_args.output_summary_json), exist_ok=True)
        with open(probe_args.output_summary_json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        logger.info(f"wrote probe summary to {probe_args.output_summary_json}")
    print(text)


if __name__ == "__main__":
    main()
