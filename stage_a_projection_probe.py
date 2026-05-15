import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import cfg as Config
from model import SGMEA
from src.data import load_data
from src.utils import pairwise_distances, csls_sim


class SimpleLogger:
    def info(self, msg):
        print(msg, flush=True)


class ProjectionHead(nn.Module):
    def __init__(self, dim, hidden_dim=0, dropout=0.0, residual=True):
        super().__init__()
        self.residual = residual
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )
        else:
            self.net = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        last = self.net[-1] if isinstance(self.net, nn.Sequential) else self.net
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, emb):
        delta = self.net(emb)
        if self.residual:
            delta = emb + delta
        return F.normalize(delta, dim=1)


class ClusterDataset(Dataset):
    def __init__(self, clusters):
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        return self.clusters[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/gly/tongqiang/dongyufeng/data")
    parser.add_argument("--data_path", default="mmkg")
    parser.add_argument("--data_choice", default="FBDB15K")
    parser.add_argument("--data_split", default="norm")
    parser.add_argument("--data_rate", type=float, default=0.5)
    parser.add_argument("--checkpoint_name", default="SGMEA_FBDB15K_0.5_baseline_warm_softw_grid_0510_121128_")
    parser.add_argument("--labels_jsonls", required=True)
    parser.add_argument("--external_anchor_type_jsonl", default="/gly/tongqiang/dongyufeng/data/mmkg/anchors_nameless/FBDB15K/norm_anchor_type_fixed.jsonl")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--margin_close", type=float, default=0.05)
    parser.add_argument("--margin_safe", type=float, default=0.25)
    parser.add_argument("--lambda_close", type=float, default=0.5)
    parser.add_argument("--lambda_safe", type=float, default=1.0)
    parser.add_argument("--max_close_per_anchor", type=int, default=3)
    parser.add_argument("--max_safe_per_anchor", type=int, default=3)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--active_only", action="store_true", default=False,
                        help="train only pairs that violate the configured margin under the frozen baseline embedding")
    parser.add_argument("--min_active_violation", type=float, default=0.0)
    parser.add_argument("--no_csls", action="store_true", default=False)
    parser.add_argument("--csls_k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def build_model_args(args):
    config = Config()
    old_argv = os.sys.argv
    os.sys.argv = [old_argv[0]]
    try:
        config.get_args()
    finally:
        os.sys.argv = old_argv
    cfg = config.cfg
    cfg.data_root = args.data_root
    cfg.data_path = args.data_path
    cfg.data_choice = args.data_choice
    cfg.data_split = args.data_split
    cfg.data_rate = args.data_rate
    cfg.model_name = "SGMEA"
    cfg.model_name_save = ""
    cfg.gpu = args.gpu
    cfg.device = "cuda"
    cfg.only_test = 1
    cfg.batch_size = 2048
    cfg.workers = args.workers
    cfg.scheduler = "cos"
    cfg.attr_dim = 300
    cfg.img_dim = 300
    cfg.name_dim = 300
    cfg.char_dim = 300
    cfg.hidden_size = 300
    cfg.hidden_units = "300,300,300"
    cfg.tau = 0.1
    cfg.structure_encoder = "gat"
    cfg.num_attention_heads = 1
    cfg.num_hidden_layers = 1
    cfg.use_surface = 0
    cfg.use_intermediate = 0
    cfg.enable_sota = True
    cfg.disable_sgmea_guidance = True
    cfg.external_anchor_type_jsonl = args.external_anchor_type_jsonl
    cfg.csls = not args.no_csls
    cfg.csls_k = args.csls_k
    cfg.rank = 0
    cfg.dist = 0
    cfg.random_seed = args.seed
    cfg.save_model = 0
    cfg.no_tensorboard = True
    return config.update_train_configs()


def load_checkpoint(model, args, model_args):
    path = os.path.join(model_args.data_path, model_args.model_name, "save", f"{args.checkpoint_name}.pkl")
    if not os.path.exists(path):
        alt = os.path.join(model_args.data_path, "SGMEA", "save", f"{args.checkpoint_name}.pkl")
        if os.path.exists(alt):
            path = alt
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location="cuda")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return path


def load_clusters(paths, min_confidence, max_close, max_safe):
    grouped = defaultdict(lambda: {"close": [], "safe": [], "meta": None})
    raw_status = Counter()
    kept_status = Counter()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                status = row.get("llm_relationship_status", "unknown")
                raw_status[status] += 1
                if float(row.get("llm_confidence", 0.0) or 0.0) < min_confidence:
                    continue
                if status not in {"closely_related", "safe_negative"}:
                    continue
                key = (row["direction"], int(row["anchor_id"]), int(row["positive_id"]))
                item = (
                    int(row["negative_id"]),
                    int(row.get("candidate_rank", 10**9)),
                    float(row.get("llm_confidence", 0.0) or 0.0),
                    row.get("coarse_type", "unknown"),
                )
                bucket = "close" if status == "closely_related" else "safe"
                grouped[key][bucket].append(item)
                grouped[key]["meta"] = {
                    "direction": row["direction"],
                    "anchor_id": int(row["anchor_id"]),
                    "positive_id": int(row["positive_id"]),
                    "coarse_type": row.get("coarse_type", "unknown"),
                }
                kept_status[status] += 1

    clusters = []
    by_type = Counter()
    for (_, anchor, positive), data in grouped.items():
        close = sorted(data["close"], key=lambda x: (x[1], -x[2]))[:max_close]
        safe = sorted(data["safe"], key=lambda x: (x[1], -x[2]))[:max_safe]
        if not close and not safe:
            continue
        meta = data["meta"]
        clusters.append(
            {
                "anchor": anchor,
                "positive": positive,
                "close": [x[0] for x in close],
                "safe": [x[0] for x in safe],
                "coarse_type": meta["coarse_type"],
                "direction": meta["direction"],
            }
        )
        by_type[meta["coarse_type"]] += 1
    return clusters, {"raw_status": dict(raw_status), "kept_status": dict(kept_status), "cluster_by_type": dict(by_type)}


def split_clusters(clusters, val_ratio, seed):
    rng = random.Random(seed)
    shuffled = list(clusters)
    rng.shuffle(shuffled)
    val_n = max(1, int(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
    return shuffled[val_n:], shuffled[:val_n]


@torch.no_grad()
def filter_active_clusters(clusters, emb, args):
    filtered = []
    removed_pairs = 0
    kept_pairs = 0
    for item in clusters:
        anchor = item["anchor"]
        positive = item["positive"]
        a = emb[anchor]
        p = emb[positive]
        sim_pos = torch.dot(a, p)
        close_keep = []
        safe_keep = []
        for neg in item["close"]:
            sim_neg = torch.dot(a, emb[neg])
            violation = args.margin_close - sim_pos + sim_neg
            if float(violation.detach().cpu()) > args.min_active_violation:
                close_keep.append(neg)
                kept_pairs += 1
            else:
                removed_pairs += 1
        for neg in item["safe"]:
            sim_neg = torch.dot(a, emb[neg])
            violation = args.margin_safe - sim_pos + sim_neg
            if float(violation.detach().cpu()) > args.min_active_violation:
                safe_keep.append(neg)
                kept_pairs += 1
            else:
                removed_pairs += 1
        if close_keep or safe_keep:
            next_item = dict(item)
            next_item["close"] = close_keep
            next_item["safe"] = safe_keep
            filtered.append(next_item)
    return filtered, {"cluster_count": len(filtered), "kept_pairs": kept_pairs, "removed_pairs": removed_pairs}


def collate_clusters(batch):
    anchors, positives, negs, margins, weights, status_ids = [], [], [], [], [], []
    for item in batch:
        anchor = item["anchor"]
        positive = item["positive"]
        for neg in item["close"]:
            anchors.append(anchor)
            positives.append(positive)
            negs.append(neg)
            margins.append(0.0)
            weights.append(0.0)
            status_ids.append(0)
        for neg in item["safe"]:
            anchors.append(anchor)
            positives.append(positive)
            negs.append(neg)
            margins.append(0.0)
            weights.append(0.0)
            status_ids.append(1)
    return {
        "anchor": torch.tensor(anchors, dtype=torch.long),
        "positive": torch.tensor(positives, dtype=torch.long),
        "negative": torch.tensor(negs, dtype=torch.long),
        "status_id": torch.tensor(status_ids, dtype=torch.long),
    }


def compute_loss(projected, batch, args):
    a = projected[batch["anchor"].cuda(non_blocking=True)]
    p = projected[batch["positive"].cuda(non_blocking=True)]
    n = projected[batch["negative"].cuda(non_blocking=True)]
    status = batch["status_id"].cuda(non_blocking=True)
    sim_pos = (a * p).sum(dim=1)
    sim_neg = (a * n).sum(dim=1)
    margin = torch.where(
        status == 0,
        torch.full_like(sim_pos, args.margin_close),
        torch.full_like(sim_pos, args.margin_safe),
    )
    weight = torch.where(
        status == 0,
        torch.full_like(sim_pos, args.lambda_close),
        torch.full_like(sim_pos, args.lambda_safe),
    )
    raw = F.relu(margin - sim_pos + sim_neg)
    denom = weight.sum().clamp_min(1.0)
    loss = (raw * weight).sum() / denom
    active = (raw > 0).float()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "active_rate": float(active.mean().detach().cpu()) if active.numel() else 0.0,
        "pos_sim": float(sim_pos.mean().detach().cpu()) if sim_pos.numel() else 0.0,
        "neg_sim": float(sim_neg.mean().detach().cpu()) if sim_neg.numel() else 0.0,
        "pair_count": int(sim_pos.numel()),
    }


@torch.no_grad()
def eval_clusters(projected, clusters, args):
    if not clusters:
        return {}
    pairs = []
    for item in clusters:
        for neg in item["close"]:
            pairs.append((item["anchor"], item["positive"], neg, 0))
        for neg in item["safe"]:
            pairs.append((item["anchor"], item["positive"], neg, 1))
    if not pairs:
        return {}
    a = torch.tensor([x[0] for x in pairs], dtype=torch.long, device=projected.device)
    p = torch.tensor([x[1] for x in pairs], dtype=torch.long, device=projected.device)
    n = torch.tensor([x[2] for x in pairs], dtype=torch.long, device=projected.device)
    status = torch.tensor([x[3] for x in pairs], dtype=torch.long, device=projected.device)
    sim_pos = (projected[a] * projected[p]).sum(dim=1)
    sim_neg = (projected[a] * projected[n]).sum(dim=1)
    close_mask = status == 0
    safe_mask = status == 1
    margin = torch.where(
        close_mask,
        torch.full_like(sim_pos, args.margin_close),
        torch.full_like(sim_pos, args.margin_safe),
    )
    weight = torch.where(
        close_mask,
        torch.full_like(sim_pos, args.lambda_close),
        torch.full_like(sim_pos, args.lambda_safe),
    )
    raw = F.relu(margin - sim_pos + sim_neg)
    out = {
        "pair_count": len(pairs),
        "pos_gt_neg_rate": float((sim_pos > sim_neg).float().mean().cpu()),
        "mean_margin": float((sim_pos - sim_neg).mean().cpu()),
        "target_loss": float(((raw * weight).sum() / weight.sum().clamp_min(1.0)).cpu()),
        "target_active_rate": float((raw > 0).float().mean().cpu()),
    }
    if close_mask.any():
        out["close_pos_gt_neg_rate"] = float((sim_pos[close_mask] > sim_neg[close_mask]).float().mean().cpu())
        out["close_mean_margin"] = float((sim_pos[close_mask] - sim_neg[close_mask]).mean().cpu())
    if safe_mask.any():
        out["safe_pos_gt_neg_rate"] = float((sim_pos[safe_mask] > sim_neg[safe_mask]).float().mean().cpu())
        out["safe_mean_margin"] = float((sim_pos[safe_mask] - sim_neg[safe_mask]).mean().cpu())
    return out


@torch.no_grad()
def eval_alignment(emb, pairs_np, use_csls=False, csls_k=3):
    left = torch.as_tensor(pairs_np[:, 0], dtype=torch.long, device=emb.device)
    right = torch.as_tensor(pairs_np[:, 1], dtype=torch.long, device=emb.device)
    distance = pairwise_distances(emb[left], emb[right])
    if use_csls:
        distance = 1 - csls_sim(1 - distance, csls_k)
    ranks_l2r = []
    ranks_r2l = []
    for idx in range(distance.shape[0]):
        indices = torch.argsort(distance[idx], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item() + 1
        ranks_l2r.append(rank)
    for idx in range(distance.shape[1]):
        indices = torch.argsort(distance[:, idx], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item() + 1
        ranks_r2l.append(rank)

    def metrics(ranks):
        arr = np.asarray(ranks, dtype=np.float64)
        return {
            "hits@1": float(np.mean(arr <= 1)),
            "hits@10": float(np.mean(arr <= 10)),
            "hits@50": float(np.mean(arr <= 50)),
            "mr": float(np.mean(arr)),
            "mrr": float(np.mean(1.0 / arr)),
        }

    l2r = metrics(ranks_l2r)
    r2l = metrics(ranks_r2l)
    return {
        "count": int(distance.shape[0]),
        "l2r": l2r,
        "r2l": r2l,
        "avg_hits@1": 0.5 * (l2r["hits@1"] + r2l["hits@1"]),
        "avg_mrr": 0.5 * (l2r["mrr"] + r2l["mrr"]),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    model_args = build_model_args(args)
    model_args.device = torch.device("cuda")
    logger = SimpleLogger()
    print("loading data...", flush=True)
    kgs, _, train_set, test_set, _, _ = load_data(logger, model_args)
    model = SGMEA(kgs, model_args).cuda()
    checkpoint_path = load_checkpoint(model, args, model_args)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"loaded checkpoint: {checkpoint_path}", flush=True)
    with torch.no_grad():
        base_emb, _ = model.joint_emb_generat()
        base_emb = F.normalize(base_emb.detach(), dim=1)
    dim = int(base_emb.shape[1])
    print(f"base embedding shape: {tuple(base_emb.shape)}", flush=True)

    label_paths = [p for p in args.labels_jsonls.split(",") if p]
    clusters, cluster_stats = load_clusters(label_paths, args.min_confidence, args.max_close_per_anchor, args.max_safe_per_anchor)
    train_clusters, val_clusters = split_clusters(clusters, args.val_ratio, args.seed)
    print(f"clusters total={len(clusters)} train={len(train_clusters)} val={len(val_clusters)} stats={cluster_stats}", flush=True)

    train_loader = DataLoader(
        ClusterDataset(train_clusters),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_clusters,
        pin_memory=True,
    )

    train_np = np.asarray(train_set.data, dtype=np.int64)
    test_np = np.asarray(test_set.data, dtype=np.int64)
    base_train_eval = eval_alignment(base_emb, train_np, use_csls=False, csls_k=args.csls_k)
    base_test_eval = eval_alignment(base_emb, test_np, use_csls=not args.no_csls, csls_k=args.csls_k)
    base_val_cluster = eval_clusters(base_emb, val_clusters, args)
    print(f"baseline test avg_hits@1={base_test_eval['avg_hits@1']:.4f} l2r={base_test_eval['l2r']['hits@1']:.4f} r2l={base_test_eval['r2l']['hits@1']:.4f}", flush=True)
    print(f"baseline val_cluster={base_val_cluster}", flush=True)

    active_filter_stats = None
    if args.active_only:
        train_clusters, active_filter_stats = filter_active_clusters(train_clusters, base_emb, args)
        print(f"active-only train clusters={len(train_clusters)} stats={active_filter_stats}", flush=True)

    head = ProjectionHead(dim=dim, hidden_dim=args.hidden_dim, dropout=args.dropout, residual=True).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_score = -1e9
    best_epoch = -1
    best_test_observed = None
    bad = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum = 0.0
        pair_sum = 0
        active_sum = 0.0
        for batch in tqdm(train_loader, desc=f"stageA ep{epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            projected = head(base_emb)
            loss, stats = compute_loss(projected, batch, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            loss_sum += stats["loss"] * stats["pair_count"]
            active_sum += stats["active_rate"] * stats["pair_count"]
            pair_sum += stats["pair_count"]

        head.eval()
        with torch.no_grad():
            projected = head(base_emb)
            val_cluster = eval_clusters(projected, val_clusters, args)
            train_eval = eval_alignment(projected, train_np, use_csls=False, csls_k=args.csls_k)
            test_eval = eval_alignment(projected, test_np, use_csls=not args.no_csls, csls_k=args.csls_k)
        score = -val_cluster.get("target_loss", 0.0)
        val_pair_acc = val_cluster.get("pos_gt_neg_rate", 0.0)
        item = {
            "epoch": epoch,
            "train_loss": loss_sum / max(pair_sum, 1),
            "train_active_rate": active_sum / max(pair_sum, 1),
            "val_cluster": val_cluster,
            "train_alignment": train_eval,
            "test_alignment": test_eval,
            "test_avg_hits1_delta": test_eval["avg_hits@1"] - base_test_eval["avg_hits@1"],
        }
        history.append(item)
        print(
            f"ep={epoch} loss={item['train_loss']:.4f} active={item['train_active_rate']:.4f} "
            f"val_pair_acc={val_pair_acc:.4f} val_target_loss={val_cluster.get('target_loss', 0.0):.6f} "
            f"val_active={val_cluster.get('target_active_rate', 0.0):.4f} test_avg_h1={test_eval['avg_hits@1']:.4f} "
            f"delta={item['test_avg_hits1_delta']:+.4f}",
            flush=True,
        )
        if best_test_observed is None or item["test_avg_hits1_delta"] > best_test_observed["test_avg_hits1_delta"]:
            best_test_observed = {
                "epoch": epoch,
                "test_avg_hits1_delta": item["test_avg_hits1_delta"],
                "test_alignment": test_eval,
                "val_cluster": val_cluster,
            }
        # Early stopping on validation target loss, with test only reported.
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch={epoch}, best_epoch={best_epoch}, best_val_score={best_score:.6f}", flush=True)
                break

    if best_state is not None:
        head.load_state_dict({k: v.cuda() for k, v in best_state.items()})
    head.eval()
    with torch.no_grad():
        best_emb = head(base_emb)
        best_train_eval = eval_alignment(best_emb, train_np, use_csls=False, csls_k=args.csls_k)
        best_test_eval = eval_alignment(best_emb, test_np, use_csls=not args.no_csls, csls_k=args.csls_k)
        best_val_cluster = eval_clusters(best_emb, val_clusters, args)

    output = {
        "args": vars(args),
        "checkpoint_path": checkpoint_path,
        "embedding_dim": dim,
        "cluster_stats": cluster_stats,
        "cluster_count": len(clusters),
        "train_cluster_count": len(train_clusters),
        "val_cluster_count": len(val_clusters),
        "active_filter_stats": active_filter_stats,
        "baseline": {
            "train_alignment_no_csls": base_train_eval,
            "test_alignment": base_test_eval,
            "val_cluster": base_val_cluster,
        },
        "best": {
            "epoch": best_epoch,
            "val_score": best_score,
            "train_alignment_no_csls": best_train_eval,
            "test_alignment": best_test_eval,
            "val_cluster": best_val_cluster,
            "test_avg_hits1_delta": best_test_eval["avg_hits@1"] - base_test_eval["avg_hits@1"],
            "test_l2r_hits1_delta": best_test_eval["l2r"]["hits@1"] - base_test_eval["l2r"]["hits@1"],
            "test_r2l_hits1_delta": best_test_eval["r2l"]["hits@1"] - base_test_eval["r2l"]["hits@1"],
        },
        "best_test_observed_diagnostic_only": best_test_observed,
        "history": history,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output["best"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
