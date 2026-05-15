import argparse
import itertools
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model import SGMEA
from src.data import load_data
from src.utils import pairwise_distances, csls_sim
from stage_a_projection_probe import SimpleLogger, build_model_args, eval_alignment, load_checkpoint
from audit_llm_risk_modality_routing import infer_active_modal_names, compute_test_ranks, summarize_entities, pearson


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/gly/tongqiang/dongyufeng/data")
    parser.add_argument("--data_path", default="mmkg")
    parser.add_argument("--data_choice", default="FBDB15K")
    parser.add_argument("--data_split", default="norm")
    parser.add_argument("--data_rate", type=float, default=0.5)
    parser.add_argument("--checkpoint_name", default="SGMEA_FBDB15K_0.5_baseline_warm_softw_grid_0510_121128_")
    parser.add_argument("--external_anchor_type_jsonl", default="/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csls", action="store_true", default=True)
    parser.add_argument("--no_csls", dest="csls", action="store_false")
    parser.add_argument("--csls_k", type=int, default=3)
    parser.add_argument("--scale_grid", default="0.7,1.0,1.3")
    parser.add_argument("--focus_types", default="Person,Place,Organization,Creative Work")
    parser.add_argument("--sweep_types", default="",
                        help="optional comma-separated subset of focus_types to sweep; other focus types keep fixed/identity scales")
    parser.add_argument("--fixed_scales_json", default="",
                        help="optional json containing fixed type->modal scales")
    parser.add_argument("--fixed_scales_key", default="combined_greedy.best.scales",
                        help="dot path to type->modal->scale map inside --fixed_scales_json")
    parser.add_argument("--sweep_on_fixed_base", action="store_true", default=False,
                        help="when sweeping a subset, evaluate each candidate on top of fixed scales for other focus types")
    parser.add_argument("--max_configs_per_type", type=int, default=0,
                        help="0 means exhaustive over scale grid for img/attr/rel/graph/gat_img/gat_attr")
    parser.add_argument("--config_mode", default="single_plus_pairs", choices=["single_plus_pairs", "exhaustive"],
                        help="single_plus_pairs is a fast diagnostic sweep; exhaustive tries all modal scale combinations")
    parser.add_argument("--fast_hits1_sweep", action="store_true", default=False,
                        help="use argmin/top1-only evaluation for dense oracle sweeps")
    parser.add_argument("--greedy_rounds", type=int, default=2,
                        help="rounds for greedy combined type-specific search")
    parser.add_argument("--greedy_top_configs", type=int, default=256,
                        help="candidate configs kept per type for greedy combined search")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def load_scales_from_json(path, key_path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        node = json.load(fp)
    for key in key_path.split("."):
        if key:
            node = node[key]
    return {
        str(typ): {str(modal): float(scale) for modal, scale in scale_map.items()}
        for typ, scale_map in node.items()
    }


def normalize_type_scales(type_scales, focus_types, modal_names):
    out = {}
    for typ in focus_types:
        scale_map = type_scales.get(typ, {})
        out[typ] = {name: float(scale_map.get(name, 1.0)) for name in modal_names}
    return out


def build_type_maps(kgs):
    type_ids = kgs.get("entity_type_ids")
    type_names = kgs.get("top_type_names", [])
    if type_ids is None:
        raise ValueError("entity_type_ids missing")
    ids = type_ids.detach().cpu().numpy() if torch.is_tensor(type_ids) else np.asarray(type_ids)
    ent_to_type = {}
    type_to_ents = defaultdict(list)
    for ent, tid in enumerate(ids):
        name = type_names[int(tid)] if int(tid) < len(type_names) else str(int(tid))
        ent_to_type[ent] = name
        type_to_ents[name].append(ent)
    return ent_to_type, type_to_ents


def split_segments(emb, modal_names):
    seg_dim = emb.shape[1] // len(modal_names)
    return {name: emb[:, i * seg_dim:(i + 1) * seg_dim] for i, name in enumerate(modal_names)}, seg_dim


def apply_type_scales(base_emb, modal_names, ent_to_type, type_to_scale):
    segs, seg_dim = split_segments(base_emb, modal_names)
    out = base_emb.clone()
    for modal_idx, modal in enumerate(modal_names):
        start = modal_idx * seg_dim
        end = start + seg_dim
        scales = torch.ones((base_emb.shape[0], 1), dtype=base_emb.dtype, device=base_emb.device)
        for typ, scale_map in type_to_scale.items():
            scale = float(scale_map.get(modal, 1.0))
            if abs(scale - 1.0) < 1e-12:
                continue
            ents = [ent for ent, t in ent_to_type.items() if t == typ]
            if ents:
                scales[torch.tensor(ents, dtype=torch.long, device=base_emb.device)] = scale
        out[:, start:end] = out[:, start:end] * scales
    return F.normalize(out, dim=1)


def evaluate_by_type(emb, test_np, ent_to_type, focus_types, use_csls, csls_k):
    all_metrics = eval_alignment(emb, test_np, use_csls=use_csls, csls_k=csls_k)
    l2r_rank, r2l_rank = compute_test_ranks(emb, test_np, use_csls=use_csls, csls_k=csls_k)
    by_type = {}
    for typ in focus_types:
        left_ids = [int(l) for l, _ in test_np if ent_to_type.get(int(l)) == typ]
        right_ids = [int(r) for _, r in test_np if ent_to_type.get(int(r)) == typ]
        l_hits = [1.0 if l2r_rank[e]["is_hit1"] else 0.0 for e in left_ids if e in l2r_rank]
        r_hits = [1.0 if r2l_rank[e]["is_hit1"] else 0.0 for e in right_ids if e in r2l_rank]
        l_mrr = [1.0 / l2r_rank[e]["rank"] for e in left_ids if e in l2r_rank]
        r_mrr = [1.0 / r2l_rank[e]["rank"] for e in right_ids if e in r2l_rank]
        by_type[typ] = {
            "l2r_count": len(l_hits),
            "r2l_count": len(r_hits),
            "l2r_hits@1": float(np.mean(l_hits)) if l_hits else 0.0,
            "r2l_hits@1": float(np.mean(r_hits)) if r_hits else 0.0,
            "avg_hits@1": 0.5 * ((float(np.mean(l_hits)) if l_hits else 0.0) + (float(np.mean(r_hits)) if r_hits else 0.0)),
            "l2r_mrr": float(np.mean(l_mrr)) if l_mrr else 0.0,
            "r2l_mrr": float(np.mean(r_mrr)) if r_mrr else 0.0,
            "avg_mrr": 0.5 * ((float(np.mean(l_mrr)) if l_mrr else 0.0) + (float(np.mean(r_mrr)) if r_mrr else 0.0)),
        }
    return all_metrics, by_type, l2r_rank, r2l_rank


@torch.no_grad()
def precompute_modal_distance(base_emb, modal_names, test_np):
    segs, _ = split_segments(base_emb, modal_names)
    left = torch.as_tensor(test_np[:, 0], dtype=torch.long, device=base_emb.device)
    right = torch.as_tensor(test_np[:, 1], dtype=torch.long, device=base_emb.device)
    dists = {}
    for name in modal_names:
        lseg = F.normalize(segs[name][left], dim=1)
        rseg = F.normalize(segs[name][right], dim=1)
        dists[name] = pairwise_distances(lseg, rseg)
    return dists


@torch.no_grad()
def evaluate_distance_by_type(distance, test_np, ent_to_type, focus_types, use_csls, csls_k):
    if use_csls:
        distance = 1 - csls_sim(1 - distance, csls_k)
    l2r_rank = {}
    r2l_rank = {}
    left_ids = [int(x) for x in test_np[:, 0]]
    right_ids = [int(x) for x in test_np[:, 1]]
    for idx in range(distance.shape[0]):
        indices = torch.argsort(distance[idx], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item() + 1
        l2r_rank[left_ids[idx]] = {"rank": rank, "is_hit1": rank == 1}
    for idx in range(distance.shape[1]):
        indices = torch.argsort(distance[:, idx], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item() + 1
        r2l_rank[right_ids[idx]] = {"rank": rank, "is_hit1": rank == 1}

    def metrics(ranks):
        arr = np.asarray(ranks, dtype=np.float64)
        return {
            "hits@1": float(np.mean(arr <= 1)),
            "hits@10": float(np.mean(arr <= 10)),
            "hits@50": float(np.mean(arr <= 50)),
            "mr": float(np.mean(arr)),
            "mrr": float(np.mean(1.0 / arr)),
        }

    all_l = [l2r_rank[e]["rank"] for e in left_ids]
    all_r = [r2l_rank[e]["rank"] for e in right_ids]
    l2r = metrics(all_l)
    r2l = metrics(all_r)
    all_metrics = {
        "count": int(distance.shape[0]),
        "l2r": l2r,
        "r2l": r2l,
        "avg_hits@1": 0.5 * (l2r["hits@1"] + r2l["hits@1"]),
        "avg_mrr": 0.5 * (l2r["mrr"] + r2l["mrr"]),
    }
    by_type = {}
    for typ in focus_types:
        l_ranks = [l2r_rank[e]["rank"] for e in left_ids if ent_to_type.get(e) == typ]
        r_ranks = [r2l_rank[e]["rank"] for e in right_ids if ent_to_type.get(e) == typ]
        lm = metrics(l_ranks) if l_ranks else {"hits@1": 0.0, "mrr": 0.0}
        rm = metrics(r_ranks) if r_ranks else {"hits@1": 0.0, "mrr": 0.0}
        by_type[typ] = {
            "l2r_count": len(l_ranks),
            "r2l_count": len(r_ranks),
            "l2r_hits@1": lm["hits@1"],
            "r2l_hits@1": rm["hits@1"],
            "avg_hits@1": 0.5 * (lm["hits@1"] + rm["hits@1"]),
            "l2r_mrr": lm["mrr"],
            "r2l_mrr": rm["mrr"],
            "avg_mrr": 0.5 * (lm["mrr"] + rm["mrr"]),
        }
    return all_metrics, by_type, l2r_rank, r2l_rank


@torch.no_grad()
def evaluate_distance_hits1_fast(distance, test_np, ent_to_type, focus_types, use_csls, csls_k):
    if use_csls:
        sim = 1 - distance
        row_avg = torch.topk(sim, k=csls_k, dim=1).values.mean(dim=1, keepdim=True)
        col_avg = torch.topk(sim, k=csls_k, dim=0).values.mean(dim=0, keepdim=True)
        distance = 1 - (2 * sim - row_avg - col_avg)
    l2r_pred = torch.argmin(distance, dim=1).detach().cpu().numpy()
    r2l_pred = torch.argmin(distance, dim=0).detach().cpu().numpy()
    n = int(distance.shape[0])
    diag = np.arange(n)
    l_hits = l2r_pred == diag
    r_hits = r2l_pred == diag
    all_metrics = {
        "count": n,
        "l2r": {"hits@1": float(np.mean(l_hits))},
        "r2l": {"hits@1": float(np.mean(r_hits))},
        "avg_hits@1": 0.5 * (float(np.mean(l_hits)) + float(np.mean(r_hits))),
    }
    left_ids = [int(x) for x in test_np[:, 0]]
    right_ids = [int(x) for x in test_np[:, 1]]
    by_type = {}
    for typ in focus_types:
        l_mask = np.asarray([ent_to_type.get(e) == typ for e in left_ids], dtype=bool)
        r_mask = np.asarray([ent_to_type.get(e) == typ for e in right_ids], dtype=bool)
        l_val = float(np.mean(l_hits[l_mask])) if np.any(l_mask) else 0.0
        r_val = float(np.mean(r_hits[r_mask])) if np.any(r_mask) else 0.0
        by_type[typ] = {
            "l2r_count": int(np.sum(l_mask)),
            "r2l_count": int(np.sum(r_mask)),
            "l2r_hits@1": l_val,
            "r2l_hits@1": r_val,
            "avg_hits@1": 0.5 * (l_val + r_val),
        }
    return all_metrics, by_type


def weighted_distance(modal_dists, modal_names, typ, cfg, ent_to_type, test_np):
    dist = None
    left_scale = {}
    right_scale = {}
    left_types = [ent_to_type.get(int(e), "unknown") for e in test_np[:, 0]]
    right_types = [ent_to_type.get(int(e), "unknown") for e in test_np[:, 1]]
    for modal in modal_names:
        base = modal_dists[modal]
        ls = torch.ones((base.shape[0], 1), dtype=base.dtype, device=base.device)
        rs = torch.ones((1, base.shape[1]), dtype=base.dtype, device=base.device)
        scale = float(cfg.get(modal, 1.0))
        if abs(scale - 1.0) > 1e-12:
            lmask = torch.tensor([t == typ for t in left_types], dtype=torch.bool, device=base.device).unsqueeze(1)
            rmask = torch.tensor([t == typ for t in right_types], dtype=torch.bool, device=base.device).unsqueeze(0)
            ls = torch.where(lmask, torch.full_like(ls, scale), ls)
            rs = torch.where(rmask, torch.full_like(rs, scale), rs)
        term = base * (ls * rs)
        dist = term if dist is None else dist + term
    return dist


def combined_weighted_distance(modal_dists, modal_names, type_to_scale, ent_to_type, test_np):
    dist = None
    left_types = [ent_to_type.get(int(e), "unknown") for e in test_np[:, 0]]
    right_types = [ent_to_type.get(int(e), "unknown") for e in test_np[:, 1]]
    for modal in modal_names:
        base = modal_dists[modal]
        ls = torch.ones((base.shape[0], 1), dtype=base.dtype, device=base.device)
        rs = torch.ones((1, base.shape[1]), dtype=base.dtype, device=base.device)
        for typ, cfg in type_to_scale.items():
            scale = float(cfg.get(modal, 1.0))
            if abs(scale - 1.0) < 1e-12:
                continue
            lmask = torch.tensor([t == typ for t in left_types], dtype=torch.bool, device=base.device).unsqueeze(1)
            rmask = torch.tensor([t == typ for t in right_types], dtype=torch.bool, device=base.device).unsqueeze(0)
            ls = torch.where(lmask, torch.full_like(ls, scale), ls)
            rs = torch.where(rmask, torch.full_like(rs, scale), rs)
        term = base * (ls * rs)
        dist = term if dist is None else dist + term
    return dist


def evaluate_oracle_distance(distance, test_np, ent_to_type, focus_types, use_csls, csls_k, fast_hits1=False):
    if fast_hits1:
        metrics, by_type = evaluate_distance_hits1_fast(
            distance, test_np, ent_to_type, focus_types, use_csls=use_csls, csls_k=csls_k
        )
        return metrics, by_type, None, None
    return evaluate_distance_by_type(
        distance, test_np, ent_to_type, focus_types, use_csls=use_csls, csls_k=csls_k
    )


def greedy_combine_search(
    modal_dists,
    modal_names,
    candidate_by_type,
    base_metrics,
    ent_to_type,
    test_np,
    focus_types,
    use_csls,
    csls_k,
    fast_hits1,
    rounds,
    initial_scales=None,
):
    if initial_scales is None:
        current = {typ: {name: 1.0 for name in modal_names} for typ in focus_types}
    else:
        current = {
            typ: dict(initial_scales.get(typ, {name: 1.0 for name in modal_names}))
            for typ in focus_types
        }
    current_dist = combined_weighted_distance(modal_dists, modal_names, current, ent_to_type, test_np)
    current_metrics, current_by_type, _, _ = evaluate_oracle_distance(
        current_dist, test_np, ent_to_type, focus_types, use_csls, csls_k, fast_hits1
    )
    history = []
    for round_idx in range(rounds):
        changed = False
        for typ in focus_types:
            best = {
                "type": typ,
                "scale": current[typ],
                "overall": current_metrics,
                "by_type": current_by_type,
                "overall_delta": current_metrics["avg_hits@1"] - base_metrics["avg_hits@1"],
            }
            for cand in candidate_by_type.get(typ, []):
                trial = dict(current)
                trial[typ] = cand["scale"]
                dist = combined_weighted_distance(modal_dists, modal_names, trial, ent_to_type, test_np)
                metrics, by_type, _, _ = evaluate_oracle_distance(
                    dist, test_np, ent_to_type, focus_types, use_csls, csls_k, fast_hits1
                )
                delta = metrics["avg_hits@1"] - base_metrics["avg_hits@1"]
                if delta > best["overall_delta"]:
                    best = {
                        "type": typ,
                        "scale": cand["scale"],
                        "overall": metrics,
                        "by_type": by_type,
                        "overall_delta": delta,
                    }
            if best["overall_delta"] > current_metrics["avg_hits@1"] - base_metrics["avg_hits@1"]:
                current[typ] = best["scale"]
                current_metrics = best["overall"]
                current_by_type = best["by_type"]
                changed = True
            history.append({
                "round": round_idx,
                "type": typ,
                "chosen_scale": current[typ],
                "overall_avg_hits@1": current_metrics["avg_hits@1"],
                "overall_delta": current_metrics["avg_hits@1"] - base_metrics["avg_hits@1"],
            })
        if not changed:
            break
    return {
        "scales": current,
        "overall": current_metrics,
        "overall_delta": current_metrics["avg_hits@1"] - base_metrics["avg_hits@1"],
        "by_type": current_by_type,
        "history": history,
    }


def run_multi_start_greedy(
    modal_dists,
    modal_names,
    candidate_by_type,
    base_metrics,
    ent_to_type,
    test_np,
    focus_types,
    use_csls,
    csls_k,
    fast_hits1,
    rounds,
    initializations,
):
    all_results = {}
    best_name = None
    best_result = None
    for name, init in initializations.items():
        result = greedy_combine_search(
            modal_dists,
            modal_names,
            candidate_by_type,
            base_metrics,
            ent_to_type,
            test_np,
            focus_types,
            use_csls=use_csls,
            csls_k=csls_k,
            fast_hits1=fast_hits1,
            rounds=rounds,
            initial_scales=init,
        )
        all_results[name] = result
        if best_result is None or result["overall_delta"] > best_result["overall_delta"]:
            best_name = name
            best_result = result
    return {"best_start": best_name, "best": best_result, "all": all_results}


def summarize_type_weights(weights, modal_names, type_to_ents, focus_types, rank_map_l=None, rank_map_r=None):
    out = {}
    dummy_risk = {ent: {"risk_score": 0, "close": 0, "safe": 0} for ents in type_to_ents.values() for ent in ents}
    for typ in focus_types:
        ents = type_to_ents.get(typ, [])
        out[typ] = summarize_entities(ents, weights, modal_names, dummy_risk)
        if rank_map_l is not None:
            correct_l = [e for e in ents if e in rank_map_l and rank_map_l[e]["is_hit1"]]
            wrong_l = [e for e in ents if e in rank_map_l and not rank_map_l[e]["is_hit1"]]
            correct_r = [e for e in ents if e in rank_map_r and rank_map_r[e]["is_hit1"]]
            wrong_r = [e for e in ents if e in rank_map_r and not rank_map_r[e]["is_hit1"]]
            out[typ]["l2r_correct"] = summarize_entities(correct_l, weights, modal_names, dummy_risk, rank_map_l)
            out[typ]["l2r_wrong"] = summarize_entities(wrong_l, weights, modal_names, dummy_risk, rank_map_l)
            out[typ]["r2l_correct"] = summarize_entities(correct_r, weights, modal_names, dummy_risk, rank_map_r)
            out[typ]["r2l_wrong"] = summarize_entities(wrong_r, weights, modal_names, dummy_risk, rank_map_r)
    return out


def config_iter(modal_names, grid, max_configs=0, seed=42, mode="single_plus_pairs"):
    if mode == "single_plus_pairs":
        configs = []
        for modal in modal_names:
            for scale in grid:
                if abs(scale - 1.0) < 1e-12:
                    continue
                cfg = {name: 1.0 for name in modal_names}
                cfg[modal] = scale
                configs.append(cfg)
        pair_templates = [
            {"graph": 0.7, "gat_img": 0.7, "gat_attr": 0.7, "attr": 1.3, "rel": 1.3},
            {"graph": 0.7, "gat_img": 0.7, "gat_attr": 0.7},
            {"attr": 1.3, "rel": 1.3},
            {"img": 1.3, "gat_img": 1.3},
            {"graph": 1.3, "gat_img": 1.3, "gat_attr": 1.3},
            {"attr": 0.7, "rel": 1.3},
            {"attr": 1.3, "rel": 0.7},
        ]
        for tmpl in pair_templates:
            cfg = {name: 1.0 for name in modal_names}
            for key, val in tmpl.items():
                if key in cfg:
                    cfg[key] = val
            if any(abs(v - 1.0) > 1e-12 for v in cfg.values()):
                configs.append(cfg)
        # stable unique order
        seen = set()
        unique = []
        for cfg in configs:
            key = tuple((name, cfg[name]) for name in modal_names)
            if key not in seen:
                seen.add(key)
                unique.append(cfg)
        return unique
    all_configs = []
    for values in itertools.product(grid, repeat=len(modal_names)):
        cfg = dict(zip(modal_names, values))
        if all(abs(v - 1.0) < 1e-12 for v in values):
            continue
        all_configs.append(cfg)
    if max_configs and len(all_configs) > max_configs:
        rng = random.Random(seed)
        rng.shuffle(all_configs)
        all_configs = all_configs[:max_configs]
    return all_configs


def main():
    args = parse_args()
    args.no_csls = not args.csls
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    model_args = build_model_args(args)
    model_args.device = torch.device("cuda")
    logger = SimpleLogger()
    print("loading data/model...", flush=True)
    kgs, _, train_set, test_set, _, _ = load_data(logger, model_args)
    model = SGMEA(kgs, model_args).cuda().eval()
    checkpoint_path = load_checkpoint(model, args, model_args)
    for p in model.parameters():
        p.requires_grad = False

    modal_names = infer_active_modal_names(model)
    with torch.no_grad():
        base_emb, weight_norm = model.joint_emb_generat()
        base_emb = F.normalize(base_emb, dim=1)
    ent_to_type, type_to_ents = build_type_maps(kgs)
    focus_types = [x.strip() for x in args.focus_types.split(",") if x.strip()]
    sweep_types = [x.strip() for x in args.sweep_types.split(",") if x.strip()] if args.sweep_types else list(focus_types)
    fixed_scales = load_scales_from_json(args.fixed_scales_json, args.fixed_scales_key)
    grid = [float(x) for x in args.scale_grid.split(",") if x.strip()]
    test_np = np.asarray(test_set.data, dtype=np.int64)
    base_metrics, base_by_type, l2r_rank, r2l_rank = evaluate_by_type(
        base_emb, test_np, ent_to_type, focus_types, use_csls=args.csls, csls_k=args.csls_k
    )
    modal_dists = precompute_modal_distance(base_emb, modal_names, test_np)
    weight_summary = summarize_type_weights(weight_norm, modal_names, type_to_ents, focus_types, l2r_rank, r2l_rank)
    print(f"modal_names={modal_names} weight_shape={tuple(weight_norm.shape)} emb_shape={tuple(base_emb.shape)}", flush=True)
    print(f"baseline avg_hits@1={base_metrics['avg_hits@1']:.4f} l2r={base_metrics['l2r']['hits@1']:.4f} r2l={base_metrics['r2l']['hits@1']:.4f}", flush=True)
    fixed_scales_full = normalize_type_scales(fixed_scales, focus_types, modal_names)
    fixed_base_metrics = None
    fixed_base_by_type = None
    if args.sweep_on_fixed_base and fixed_scales:
        fixed_base_dist = combined_weighted_distance(modal_dists, modal_names, fixed_scales_full, ent_to_type, test_np)
        fixed_base_metrics, fixed_base_by_type, _, _ = evaluate_oracle_distance(
            fixed_base_dist,
            test_np,
            ent_to_type,
            focus_types,
            use_csls=args.csls,
            csls_k=args.csls_k,
            fast_hits1=args.fast_hits1_sweep,
        )
        print(
            f"fixed_base avg_hits@1={fixed_base_metrics['avg_hits@1']:.4f} "
            f"delta={fixed_base_metrics['avg_hits@1'] - base_metrics['avg_hits@1']:+.4f}",
            flush=True,
        )

    configs = config_iter(modal_names, grid, args.max_configs_per_type, args.seed, args.config_mode)
    sweep = {}
    for typ in focus_types:
        if typ not in sweep_types:
            fixed = fixed_scales.get(typ, {name: 1.0 for name in modal_names})
            fixed = {name: float(fixed.get(name, 1.0)) for name in modal_names}
            dist = weighted_distance(modal_dists, modal_names, typ, fixed, ent_to_type, test_np)
            metrics, by_type, _, _ = evaluate_oracle_distance(
                dist,
                test_np,
                ent_to_type,
                focus_types,
                use_csls=args.csls,
                csls_k=args.csls_k,
                fast_hits1=args.fast_hits1_sweep,
            )
            item = {
                "type": typ,
                "scale": fixed,
                "global_avg_hits@1": metrics["avg_hits@1"],
                "global_delta": metrics["avg_hits@1"] - base_metrics["avg_hits@1"],
                "type_avg_hits@1": by_type[typ]["avg_hits@1"],
                "type_delta": by_type[typ]["avg_hits@1"] - base_by_type[typ]["avg_hits@1"],
                "global_avg_mrr": metrics.get("avg_mrr"),
                "type_avg_mrr": by_type[typ].get("avg_mrr"),
                "fixed": True,
            }
            sweep[typ] = {
                "best_global": item,
                "best_type": item,
                "top10_by_global": [item],
                "top10_by_type": [item],
                "top_by_global_for_greedy": [item],
                "top_by_type_for_greedy": [item],
            }
            print(
                f"type={typ} fixed_global_delta={item['global_delta']:+.4f} "
                f"fixed_type_delta={item['type_delta']:+.4f}",
                flush=True,
            )
            continue
        best_global = None
        best_type = None
        rows = []
        print(f"sweeping type={typ} configs={len(configs)}", flush=True)
        for cfg in configs:
            if args.sweep_on_fixed_base and fixed_scales:
                trial_scales = {name: dict(scale_map) for name, scale_map in fixed_scales_full.items()}
                trial_scales[typ] = dict(cfg)
                dist = combined_weighted_distance(modal_dists, modal_names, trial_scales, ent_to_type, test_np)
            else:
                dist = weighted_distance(modal_dists, modal_names, typ, cfg, ent_to_type, test_np)
            metrics, by_type, _, _ = evaluate_oracle_distance(
                dist,
                test_np,
                ent_to_type,
                focus_types,
                use_csls=args.csls,
                csls_k=args.csls_k,
                fast_hits1=args.fast_hits1_sweep,
            )
            item = {
                "type": typ,
                "scale": cfg,
                "global_avg_hits@1": metrics["avg_hits@1"],
                "global_delta": metrics["avg_hits@1"] - base_metrics["avg_hits@1"],
                "type_avg_hits@1": by_type[typ]["avg_hits@1"],
                "type_delta": by_type[typ]["avg_hits@1"] - base_by_type[typ]["avg_hits@1"],
                "global_avg_mrr": metrics.get("avg_mrr"),
                "type_avg_mrr": by_type[typ].get("avg_mrr"),
            }
            rows.append(item)
            if best_global is None or item["global_delta"] > best_global["global_delta"]:
                best_global = item
            if best_type is None or item["type_delta"] > best_type["type_delta"]:
                best_type = item
        rows_sorted = sorted(rows, key=lambda x: (x["global_delta"], x["type_delta"]), reverse=True)
        sweep[typ] = {
            "best_global": best_global,
            "best_type": best_type,
            "top10_by_global": rows_sorted[:10],
            "top10_by_type": sorted(rows, key=lambda x: (x["type_delta"], x["global_delta"]), reverse=True)[:10],
            "top_by_global_for_greedy": rows_sorted[:args.greedy_top_configs],
            "top_by_type_for_greedy": sorted(rows, key=lambda x: (x["type_delta"], x["global_delta"]), reverse=True)[:args.greedy_top_configs],
        }
        print(
            f"type={typ} best_global_delta={best_global['global_delta']:+.4f} "
            f"best_type_delta={best_type['type_delta']:+.4f}",
            flush=True,
        )

    # Combine best global per type, then run greedy coordinate search over top candidates.
    combined_scales = {typ: sweep[typ]["best_global"]["scale"] for typ in focus_types}
    combined_type_scales = {typ: sweep[typ]["best_type"]["scale"] for typ in focus_types}
    combined_dist = combined_weighted_distance(modal_dists, modal_names, combined_scales, ent_to_type, test_np)
    combined_metrics, combined_by_type, _, _ = evaluate_oracle_distance(
        combined_dist,
        test_np,
        ent_to_type,
        focus_types,
        use_csls=args.csls,
        csls_k=args.csls_k,
        fast_hits1=args.fast_hits1_sweep,
    )
    candidate_by_type = {}
    for typ in focus_types:
        seen = set()
        merged = []
        all_rows = (
            sweep[typ]["top_by_global_for_greedy"]
            + sweep[typ]["top_by_type_for_greedy"]
            + [sweep[typ]["best_global"], sweep[typ]["best_type"]]
        )
        for row in all_rows:
            key = tuple((name, row["scale"][name]) for name in modal_names)
            if key not in seen:
                seen.add(key)
                merged.append(row)
        candidate_by_type[typ] = merged[:args.greedy_top_configs]
    identity_scales = {typ: {name: 1.0 for name in modal_names} for typ in focus_types}
    greedy = run_multi_start_greedy(
        modal_dists,
        modal_names,
        candidate_by_type,
        base_metrics,
        ent_to_type,
        test_np,
        focus_types,
        use_csls=args.csls,
        csls_k=args.csls_k,
        fast_hits1=args.fast_hits1_sweep,
        rounds=args.greedy_rounds,
        initializations={
            "identity": identity_scales,
            "best_global_per_type": combined_scales,
            "best_type_per_type": combined_type_scales,
        },
    )

    out = {
        "input": {
            "checkpoint_path": checkpoint_path,
            "scale_grid": grid,
            "focus_types": focus_types,
            "csls": args.csls,
            "csls_k": args.csls_k,
            "fast_hits1_sweep": args.fast_hits1_sweep,
            "config_mode": args.config_mode,
            "config_count": len(configs),
            "sweep_types": sweep_types,
            "fixed_scales_json": args.fixed_scales_json,
            "fixed_scales_key": args.fixed_scales_key,
            "sweep_on_fixed_base": args.sweep_on_fixed_base,
            "greedy_rounds": args.greedy_rounds,
            "greedy_top_configs": args.greedy_top_configs,
        },
        "modal": {
            "modal_names": modal_names,
            "weight_norm_shape": list(weight_norm.shape),
            "joint_emb_shape": list(base_emb.shape),
            "joint_segment_dim": int(base_emb.shape[1] // len(modal_names)),
            "segment_order": modal_names,
        },
        "baseline": {
            "overall": base_metrics,
            "by_type": base_by_type,
            "weight_by_type": weight_summary,
        },
        "fixed_base": {
            "scales": fixed_scales_full,
            "overall": fixed_base_metrics,
            "overall_delta": fixed_base_metrics["avg_hits@1"] - base_metrics["avg_hits@1"] if fixed_base_metrics else None,
            "by_type": fixed_base_by_type,
        } if fixed_base_metrics else None,
        "sweep": sweep,
        "combined_best_global_per_type": {
            "scales": combined_scales,
            "overall": combined_metrics,
            "overall_delta": combined_metrics["avg_hits@1"] - base_metrics["avg_hits@1"],
            "by_type": combined_by_type,
        },
        "combined_best_type_per_type_scales": combined_type_scales,
        "combined_greedy": greedy,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "baseline": out["baseline"]["overall"],
        "baseline_by_type": out["baseline"]["by_type"],
        "best": {typ: sweep[typ]["best_global"] for typ in focus_types},
        "combined": out["combined_best_global_per_type"],
        "greedy": out["combined_greedy"]["best"],
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
