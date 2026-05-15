import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SGMEA
from src.data import load_data

from stage_a_projection_probe import (
    ClusterDataset,
    SimpleLogger,
    build_model_args,
    collate_clusters,
    eval_alignment,
    eval_clusters,
    filter_active_clusters,
    load_checkpoint,
    load_clusters,
    split_clusters,
)


class DecoupledIdentityHead(nn.Module):
    def __init__(
        self,
        input_dim,
        id_dim=256,
        hidden_dim=512,
        dropout=0.1,
        gamma_init=0.15,
        gamma_max=0.3,
    ):
        super().__init__()
        self.gamma_max = float(gamma_max)
        hidden_dim = int(hidden_dim)
        id_dim = int(id_dim)
        self.id_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, id_dim),
        )
        init_ratio = min(max(gamma_init / max(gamma_max, 1e-8), 1e-4), 1.0 - 1e-4)
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))

    def gamma(self):
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    def forward(self, top_emb):
        top_emb = F.normalize(top_emb, dim=1)
        id_emb = F.normalize(self.id_net(top_emb), dim=1)
        gamma = self.gamma()
        final_emb = F.normalize(torch.cat([top_emb, gamma * id_emb], dim=1), dim=1)
        return top_emb, id_emb, final_emb, gamma


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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--id_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma_init", type=float, default=0.15)
    parser.add_argument("--gamma_max", type=float, default=0.3)
    parser.add_argument("--margin_close", type=float, default=0.05)
    parser.add_argument("--margin_safe", type=float, default=0.25)
    parser.add_argument("--lambda_close", type=float, default=0.5)
    parser.add_argument("--lambda_safe", type=float, default=1.0)
    parser.add_argument("--lambda_final", type=float, default=0.2)
    parser.add_argument("--lambda_gamma", type=float, default=0.01)
    parser.add_argument("--lambda_preserve", type=float, default=0.05)
    parser.add_argument("--preserve_pairs", type=int, default=512)
    parser.add_argument("--max_close_per_anchor", type=int, default=3)
    parser.add_argument("--max_safe_per_anchor", type=int, default=3)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--active_only", action="store_true", default=False)
    parser.add_argument("--min_active_violation", type=float, default=0.0)
    parser.add_argument("--no_csls", action="store_true", default=False)
    parser.add_argument("--csls_k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def compute_margin_loss(emb, batch, args):
    a = emb[batch["anchor"].cuda(non_blocking=True)]
    p = emb[batch["positive"].cuda(non_blocking=True)]
    n = emb[batch["negative"].cuda(non_blocking=True)]
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
    loss = (raw * weight).sum() / weight.sum().clamp_min(1.0)
    return loss, {
        "active_rate": float((raw > 0).float().mean().detach().cpu()) if raw.numel() else 0.0,
        "pair_count": int(raw.numel()),
        "pos_sim": float(sim_pos.mean().detach().cpu()) if raw.numel() else 0.0,
        "neg_sim": float(sim_neg.mean().detach().cpu()) if raw.numel() else 0.0,
    }


def compute_preserve_loss(top_emb, final_emb, batch, max_pairs):
    ids = torch.cat([batch["anchor"], batch["positive"], batch["negative"]]).unique().cuda(non_blocking=True)
    if ids.numel() < 2:
        return final_emb.new_tensor(0.0)
    if ids.numel() > max_pairs:
        perm = torch.randperm(ids.numel(), device=ids.device)[:max_pairs]
        ids = ids[perm]
    shuffled = ids[torch.randperm(ids.numel(), device=ids.device)]
    top_sim = (top_emb[ids] * top_emb[shuffled]).sum(dim=1)
    final_sim = (final_emb[ids] * final_emb[shuffled]).sum(dim=1)
    return F.mse_loss(final_sim, top_sim)


@torch.no_grad()
def eval_decoupled_clusters(top_emb, id_emb, final_emb, clusters, args):
    return {
        "id": eval_clusters(id_emb, clusters, args),
        "final": eval_clusters(final_emb, clusters, args),
        "top": eval_clusters(top_emb, clusters, args),
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
    print(f"frozen topology embedding shape: {tuple(base_emb.shape)}", flush=True)

    label_paths = [p for p in args.labels_jsonls.split(",") if p]
    clusters, cluster_stats = load_clusters(
        label_paths,
        args.min_confidence,
        args.max_close_per_anchor,
        args.max_safe_per_anchor,
    )
    train_clusters, val_clusters = split_clusters(clusters, args.val_ratio, args.seed)
    active_filter_stats = None
    if args.active_only:
        train_clusters, active_filter_stats = filter_active_clusters(train_clusters, base_emb, args)

    print(
        f"clusters total={len(clusters)} train={len(train_clusters)} val={len(val_clusters)} "
        f"active_filter={active_filter_stats} stats={cluster_stats}",
        flush=True,
    )

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
    print(
        f"baseline test avg_hits@1={base_test_eval['avg_hits@1']:.4f} "
        f"l2r={base_test_eval['l2r']['hits@1']:.4f} r2l={base_test_eval['r2l']['hits@1']:.4f}",
        flush=True,
    )
    print(f"baseline val_cluster={base_val_cluster}", flush=True)

    head = DecoupledIdentityHead(
        input_dim=dim,
        id_dim=args.id_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gamma_init=args.gamma_init,
        gamma_max=args.gamma_max,
    ).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_score = -1e9
    best_epoch = -1
    bad = 0
    history = []
    best_test_observed = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum = id_loss_sum = final_loss_sum = preserve_sum = active_sum = 0.0
        pair_sum = 0
        for batch in tqdm(train_loader, desc=f"stageA2 ep{epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            top_emb, id_emb, final_emb, gamma = head(base_emb)
            id_loss, id_stats = compute_margin_loss(id_emb, batch, args)
            final_loss, final_stats = compute_margin_loss(final_emb, batch, args)
            preserve_loss = compute_preserve_loss(top_emb, final_emb, batch, args.preserve_pairs)
            gamma_loss = gamma.pow(2)
            loss = (
                id_loss
                + args.lambda_final * final_loss
                + args.lambda_preserve * preserve_loss
                + args.lambda_gamma * gamma_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            pairs = id_stats["pair_count"]
            loss_sum += float(loss.detach().cpu()) * pairs
            id_loss_sum += float(id_loss.detach().cpu()) * pairs
            final_loss_sum += float(final_loss.detach().cpu()) * pairs
            preserve_sum += float(preserve_loss.detach().cpu()) * pairs
            active_sum += id_stats["active_rate"] * pairs
            pair_sum += pairs

        head.eval()
        with torch.no_grad():
            top_emb, id_emb, final_emb, gamma = head(base_emb)
            val_cluster = eval_decoupled_clusters(top_emb, id_emb, final_emb, val_clusters, args)
            train_eval = eval_alignment(final_emb, train_np, use_csls=False, csls_k=args.csls_k)
            test_eval = eval_alignment(final_emb, test_np, use_csls=not args.no_csls, csls_k=args.csls_k)

        val_final_loss = val_cluster["final"].get("target_loss", 0.0)
        score = -val_final_loss
        item = {
            "epoch": epoch,
            "loss": loss_sum / max(pair_sum, 1),
            "id_loss": id_loss_sum / max(pair_sum, 1),
            "final_loss": final_loss_sum / max(pair_sum, 1),
            "preserve_loss": preserve_sum / max(pair_sum, 1),
            "train_active_rate_id": active_sum / max(pair_sum, 1),
            "gamma": float(gamma.detach().cpu()),
            "val_cluster": val_cluster,
            "train_alignment_no_csls": train_eval,
            "test_alignment": test_eval,
            "test_avg_hits1_delta": test_eval["avg_hits@1"] - base_test_eval["avg_hits@1"],
        }
        history.append(item)
        print(
            f"ep={epoch} loss={item['loss']:.5f} id={item['id_loss']:.5f} final={item['final_loss']:.5f} "
            f"pres={item['preserve_loss']:.6f} gamma={item['gamma']:.4f} "
            f"val_final_loss={val_final_loss:.6f} val_final_active={val_cluster['final'].get('target_active_rate', 0.0):.4f} "
            f"test_avg_h1={test_eval['avg_hits@1']:.4f} delta={item['test_avg_hits1_delta']:+.4f}",
            flush=True,
        )

        if best_test_observed is None or item["test_avg_hits1_delta"] > best_test_observed["test_avg_hits1_delta"]:
            best_test_observed = {
                "epoch": epoch,
                "test_avg_hits1_delta": item["test_avg_hits1_delta"],
                "test_alignment": test_eval,
                "val_cluster": val_cluster,
                "gamma": item["gamma"],
            }
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
        top_emb, id_emb, final_emb, gamma = head(base_emb)
        best_train_eval = eval_alignment(final_emb, train_np, use_csls=False, csls_k=args.csls_k)
        best_test_eval = eval_alignment(final_emb, test_np, use_csls=not args.no_csls, csls_k=args.csls_k)
        best_val_cluster = eval_decoupled_clusters(top_emb, id_emb, final_emb, val_clusters, args)

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
            "gamma": float(gamma.detach().cpu()),
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
