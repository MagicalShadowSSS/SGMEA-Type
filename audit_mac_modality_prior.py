import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model import SGMEA
from src.data import load_data
from stage_a_projection_probe import SimpleLogger, build_model_args, load_checkpoint


MODAL_ORDER = ["img", "attr", "rel", "graph"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute a closed-form type-modality prior from train-pair modality agreement."
    )
    parser.add_argument("--data_root", default="/gly/tongqiang/dongyufeng/data")
    parser.add_argument("--data_path", default="mmkg")
    parser.add_argument("--data_choice", default="FBDB15K")
    parser.add_argument("--data_split", default="norm")
    parser.add_argument("--data_rate", type=float, default=0.5)
    parser.add_argument("--checkpoint_name", default="SGMEA_FBDB15K_0.5_baseline_warm_fbdb_rates_cdmr_0511_104108_r05_")
    parser.add_argument("--external_anchor_type_jsonl", default="/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_grid", default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--scale_min", type=float, default=0.4)
    parser.add_argument("--scale_max", type=float, default=1.7)
    parser.add_argument("--focus_types", default="Person,Place,Organization,Creative Work")
    parser.add_argument("--reference_prior_json", default="/gly/tongqiang/dongyufeng/data/mmkg/SGMEA/train_relation_hnm/type_modality_oracle_4seg_refine_0510_193727.json")
    parser.add_argument("--reference_prior_key", default="combined_greedy.best.scales")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no_csls", action="store_true", default=False)
    parser.add_argument("--csls_k", type=int, default=3)
    return parser.parse_args()


def get_by_path(payload, key_path):
    node = payload
    for key in key_path.split("."):
        if key:
            node = node[key]
    return node


def load_reference(path, key_path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        return get_by_path(json.load(fp), key_path)


def build_type_maps(kgs):
    type_ids = kgs.get("entity_type_ids")
    type_names = kgs.get("top_type_names", [])
    if type_ids is None:
        raise ValueError("entity_type_ids missing")
    ids = type_ids.detach().cpu().numpy() if torch.is_tensor(type_ids) else np.asarray(type_ids)
    ent_to_type = {}
    for ent, tid in enumerate(ids):
        ent_to_type[ent] = type_names[int(tid)] if int(tid) < len(type_names) else str(int(tid))
    return ent_to_type


@torch.no_grad()
def get_modal_embeddings(model):
    encoder = model.multimodal_encoder
    gph_emb = encoder.cross_graph_model(encoder.entity_emb(model.input_idx), model.adj) if model.args.w_gcn else None
    img_emb = encoder.img_fc(model.img_features) if model.args.w_img else None
    rel_emb = encoder.rel_fc(model.rel_features) if model.args.w_rel else None
    attr_emb = encoder.att_fc(model.att_features) if model.args.w_attr else None
    embs = {
        "img": img_emb,
        "attr": attr_emb,
        "rel": rel_emb,
        "graph": gph_emb,
    }
    return {name: F.normalize(emb, dim=1) for name, emb in embs.items() if emb is not None}


def summarize_values(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "lt_0.5": float(np.mean(arr < 0.5)),
        "lt_0.3": float(np.mean(arr < 0.3)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_agreement(embs, train_ill, ent_to_type, focus_types):
    raw = {
        typ: {modal: [] for modal in MODAL_ORDER if modal in embs}
        for typ in focus_types
    }
    raw["ALL"] = {modal: [] for modal in MODAL_ORDER if modal in embs}
    for left, right in train_ill:
        left = int(left)
        right = int(right)
        typ = ent_to_type.get(left, "Entity")
        for modal, emb in embs.items():
            sim = float(torch.sum(emb[left] * emb[right]).detach().cpu())
            raw["ALL"][modal].append(sim)
            if typ in raw:
                raw[typ][modal].append(sim)
    return {
        typ: {modal: summarize_values(vals) for modal, vals in modal_map.items()}
        for typ, modal_map in raw.items()
    }


def scale_from_scores(scores, alpha, scale_min, scale_max):
    vals = np.asarray([scores[m] for m in MODAL_ORDER], dtype=np.float64)
    mu = float(vals.mean())
    out = {}
    for modal in MODAL_ORDER:
        scale = 1.0 + float(alpha) * (float(scores[modal]) - mu)
        scale = max(float(scale_min), min(float(scale_max), scale))
        out[modal] = float(scale)
    return out, mu


def make_prior(agreement, focus_types, alpha, scale_min, scale_max):
    prior = {}
    details = {}
    for typ in focus_types:
        scores = {modal: float(agreement[typ][modal]["mean"]) for modal in MODAL_ORDER}
        scale, mu = scale_from_scores(scores, alpha, scale_min, scale_max)
        prior[typ] = scale
        details[typ] = {"scores": scores, "mean_center": mu, "scale": scale}
    return prior, details


def compare_priors(candidate, reference, focus_types):
    rows = {}
    diffs = []
    for typ in focus_types:
        rows[typ] = {}
        ref_map = reference.get(typ, {})
        for modal in MODAL_ORDER:
            cand = float(candidate.get(typ, {}).get(modal, 1.0))
            ref = float(ref_map.get(modal, 1.0))
            rows[typ][modal] = {"candidate": cand, "reference": ref, "abs_diff": abs(cand - ref)}
            diffs.append(abs(cand - ref))
    return {
        "by_type": rows,
        "mae": float(np.mean(diffs)) if diffs else 0.0,
        "max_abs_diff": float(np.max(diffs)) if diffs else 0.0,
    }


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    focus_types = [x.strip() for x in args.focus_types.split(",") if x.strip()]
    model_args = build_model_args(args)
    logger = SimpleLogger()
    kgs, non_train, train_set, eval_set, test_set, test_ill = load_data(logger, model_args)
    model = SGMEA(kgs, model_args).cuda()
    checkpoint_path = load_checkpoint(model, args, model_args)
    model.eval()

    ent_to_type = build_type_maps(kgs)
    train_ill = np.asarray(train_set.data, dtype=np.int64)
    embs = get_modal_embeddings(model)
    agreement = compute_agreement(embs, train_ill, ent_to_type, focus_types)
    reference = load_reference(args.reference_prior_json, args.reference_prior_key)

    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    alpha_results = {}
    for alpha in alpha_grid:
        prior, details = make_prior(agreement, focus_types, alpha, args.scale_min, args.scale_max)
        alpha_results[str(alpha)] = {
            "prior": prior,
            "details": details,
            "reference_compare": compare_priors(prior, reference, focus_types),
        }
    selected_prior, selected_details = make_prior(agreement, focus_types, args.alpha, args.scale_min, args.scale_max)

    output = {
        "data_choice": args.data_choice,
        "data_split": args.data_split,
        "data_rate": args.data_rate,
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_path": checkpoint_path,
        "external_anchor_type_jsonl": args.external_anchor_type_jsonl,
        "focus_types": focus_types,
        "modal_names": list(embs.keys()),
        "train_pair_count": int(train_ill.shape[0]),
        "agreement": agreement,
        "mapping": {
            "formula": "scale[t,m] = clip(1 + alpha * (mean_sim[t,m] - mean_m mean_sim[t,m]))",
            "alpha": args.alpha,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
        },
        "selected_prior": selected_prior,
        "selected_details": selected_details,
        "reference_prior": reference,
        "reference_compare": compare_priors(selected_prior, reference, focus_types),
        "alpha_grid": alpha_results,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)

    print(f"checkpoint={checkpoint_path}")
    print(f"output={args.output_json}")
    print(f"modal_names={list(embs.keys())} train_pairs={train_ill.shape[0]}")
    print("selected_prior:")
    print(json.dumps(selected_prior, indent=2))
    print("reference_compare:")
    print(json.dumps(output["reference_compare"], indent=2))
    print("alpha_grid_mae:")
    for alpha, row in alpha_results.items():
        print(alpha, row["reference_compare"]["mae"])


if __name__ == "__main__":
    main()
