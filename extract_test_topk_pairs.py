import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from config import cfg
from model import SGMEA
from main import Runner
from src.data import load_data
from src.utils import pairwise_distances, csls_sim
from torchlight import set_seed


class SimpleLogger:
    def info(self, msg):
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_summary_json", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--coarse_types",
        type=str,
        default="Place,Creative Work",
        help="Comma-separated query coarse types to keep",
    )
    parser.add_argument(
        "--same_type_only",
        type=int,
        default=1,
        help="When 1, only export candidates with the same coarse type as the query",
    )
    parser.add_argument(
        "--run_router_calibration",
        type=int,
        default=0,
        help="When 1, run DEHR/TMHG calibration before exporting top-k pairs; needed for V2-P style router-only runs.",
    )
    pair_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        train_cfg = cfg()
        train_cfg.get_args()
        args = train_cfg.update_train_configs()
    finally:
        sys.argv = original_argv

    return pair_args, args


def load_checkpoint(model, args, logger=None):
    if not args.model_name_save:
        raise ValueError("--model_name_save is required")
    save_path = os.path.join(args.data_path, args.model_name, "save", f"{args.model_name_save}.pkl")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"checkpoint not found: {save_path}")
    state = torch.load(save_path, map_location=args.device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    if (
        getattr(args, "use_type_modality_bias", False)
        or getattr(args, "use_dehr_router", False)
        or getattr(args, "use_tmhg_router", False)
    ):
        incompatible = model.load_state_dict(state, strict=False)
        if logger is not None:
            logger.info(
                "router warmstart load_state_dict strict=False | "
                f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}"
            )
    else:
        model.load_state_dict(state)
    return save_path


def run_router_calibration_if_requested(pair_args, args, logger):
    if not pair_args.run_router_calibration:
        return None
    prev_only_test = getattr(args, "only_test", 0)
    args.only_test = 0
    runner = Runner(args, writer=None, logger=logger, rank=0)
    runner.epoch = 0
    runner.dehr_calibration_pretrain()
    runner.tmhg_calibration_pretrain()
    runner.model.eval()
    args.only_test = prev_only_test
    logger.info("router calibration finished before top-k extraction")
    return runner


def build_distance_matrix(args, model, test_left, test_right):
    with torch.no_grad():
        final_emb, _ = model.joint_emb_generat()
        final_emb = F.normalize(final_emb)
        if args.distance != 2:
            raise NotImplementedError("Only distance=2 is supported in this extraction script.")
        distance = pairwise_distances(final_emb[test_left], final_emb[test_right])
        if args.csls:
            distance = 1 - csls_sim(1 - distance, args.csls_k)
    return distance


def normalize_raw_name(raw):
    return (raw or "").strip()


def parse_attr_line(line):
    line = line.strip()
    if not line:
        return None
    if "\t" in line:
        parts = line.split("\t", 2)
    else:
        parts = line.split(" ", 2)
    if len(parts) < 3:
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def detect_entity_style(name):
    stripped = normalize_raw_name(name).strip().strip("<>")
    if "dbpedia.org/resource/" in stripped:
        return "dbpedia"
    if stripped.startswith("/m/"):
        return "freebase"
    return "other"


def infer_attr_file_side(path):
    style_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_attr_line(line)
            if parsed is None:
                continue
            ent_name, _, _ = parsed
            style_counter[detect_entity_style(ent_name)] += 1
            if sum(style_counter.values()) >= 200:
                break
    return style_counter


def load_id_name_maps(file_dir):
    side_maps = {}
    for side, filename in [("left", "ent_ids_1"), ("right", "ent_ids_2")]:
        path = os.path.join(file_dir, filename)
        id_to_display = {}
        id_to_raw = {}
        name_to_id = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                ent_id = int(parts[0])
                raw_name = normalize_raw_name(parts[1])
                id_to_raw[ent_id] = raw_name
                id_to_display[ent_id] = raw_name.strip("<>").split("/")[-1].replace("_", " ")
                name_to_id[raw_name] = ent_id
                name_to_id[raw_name.strip("<>")] = ent_id
        side_maps[side] = {
            "id_to_display": id_to_display,
            "id_to_raw": id_to_raw,
            "name_to_id": name_to_id,
        }
    return side_maps


def load_attribute_kv(file_dir, left_name_to_id, right_name_to_id, attr_limit):
    id_to_attrs = defaultdict(list)
    seen = defaultdict(set)
    file_specs = []
    for filename in ["att_triples1", "att_triples2"]:
        path = os.path.join(file_dir, filename)
        if not os.path.exists(path):
            continue
        style_counter = infer_attr_file_side(path)
        dominant_style = style_counter.most_common(1)[0][0] if style_counter else "other"
        if dominant_style == "dbpedia":
            name_to_id = right_name_to_id
        elif dominant_style == "freebase":
            name_to_id = left_name_to_id
        else:
            continue
        file_specs.append((filename, name_to_id))

    for filename, name_to_id in file_specs:
        path = os.path.join(file_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_attr_line(line)
                if parsed is None:
                    continue
                ent_name, attr_name, value = parsed
                ent_id = name_to_id.get(ent_name)
                if ent_id is None:
                    ent_id = name_to_id.get(ent_name.strip("<>"))
                if ent_id is None:
                    continue
                item = f"{attr_name.strip()}: {value.strip()}"
                if item in seen[ent_id]:
                    continue
                seen[ent_id].add(item)
                if len(id_to_attrs[ent_id]) < attr_limit:
                    id_to_attrs[ent_id].append(item)
    return id_to_attrs


def main():
    pair_args, args = parse_args()
    logger = SimpleLogger()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because SGMEA builds CUDA tensors internally.")

    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)
    set_seed(args.random_seed)
    query_type_filter = {x.strip() for x in pair_args.coarse_types.split(",") if x.strip()}

    runner = run_router_calibration_if_requested(pair_args, args, logger)
    if runner is not None:
        model = runner.model
        train_set = runner.train_set
        test_set = runner.test_set
        eval_set = runner.eval_set
    else:
        # load_data returns: KGs, non_train, train_ill, test_ill, eval_ill, test_ill_
        kgs, non_train, train_set, test_set, eval_set, test_ill_ = load_data(logger, args)
        model = SGMEA(kgs, args).cuda()
        ckpt_path = load_checkpoint(model, args, logger=logger)
        model.eval()
        logger.info(f"loaded checkpoint: {ckpt_path}")

    active_set = test_set if test_set is not None else eval_set
    if active_set is None:
        raise ValueError("both test_set and eval_set are None")

    test_np = active_set.data if hasattr(active_set, "data") else active_set
    test_left = torch.LongTensor(test_np[:, 0].squeeze()).cuda()
    test_right = torch.LongTensor(test_np[:, 1].squeeze()).cuda()
    distance = build_distance_matrix(args, model, test_left, test_right)

    file_dir = os.path.join(args.data_path, args.data_choice, args.data_split)
    side_maps = load_id_name_maps(file_dir)
    left_maps = side_maps["left"]
    right_maps = side_maps["right"]
    id_to_attrs = load_attribute_kv(
        file_dir,
        left_name_to_id=left_maps["name_to_id"],
        right_name_to_id=right_maps["name_to_id"],
        attr_limit=20,
    )

    type_ids_cpu = model.entity_type_ids.detach().cpu()
    test_left_cpu = test_left.detach().cpu()
    test_right_cpu = test_right.detach().cpu()

    os.makedirs(os.path.dirname(pair_args.output_jsonl), exist_ok=True)
    rows_written = 0
    query_counter = Counter()
    pair_counter = Counter()

    with open(pair_args.output_jsonl, "w", encoding="utf-8") as f:
        for idx in range(distance.shape[0]):
            query_id = int(test_left_cpu[idx].item())
            gt_id = int(test_right_cpu[idx].item())
            query_type_id = int(type_ids_cpu[query_id].item())
            query_type = model.top_type_names[query_type_id] if 0 <= query_type_id < len(model.top_type_names) else str(query_type_id)
            if query_type not in query_type_filter:
                continue

            values, indices = torch.sort(distance[idx, :], descending=False)
            values = values[: pair_args.topk].detach().cpu()
            indices = indices[: pair_args.topk].detach().cpu()
            query_counter[query_type] += 1

            for rank, (cand_local_idx, cand_distance) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                cand_id = int(test_right_cpu[cand_local_idx].item())
                cand_type_id = int(type_ids_cpu[cand_id].item())
                cand_type = model.top_type_names[cand_type_id] if 0 <= cand_type_id < len(model.top_type_names) else str(cand_type_id)
                same_coarse_type = (cand_type == query_type)
                if pair_args.same_type_only and not same_coarse_type:
                    continue

                row = {
                    "dataset": args.data_choice,
                    "query_coarse_type": query_type,
                    "candidate_coarse_type": cand_type,
                    "same_coarse_type": same_coarse_type,
                    "query_id": query_id,
                    "query_name_display": left_maps["id_to_display"].get(query_id, str(query_id)),
                    "query_name_raw": left_maps["id_to_raw"].get(query_id, str(query_id)),
                    "query_attr_kv": id_to_attrs.get(query_id, []),
                    "gt_id": gt_id,
                    "gt_name_display": right_maps["id_to_display"].get(gt_id, str(gt_id)),
                    "gt_name_raw": right_maps["id_to_raw"].get(gt_id, str(gt_id)),
                    "gt_attr_kv": id_to_attrs.get(gt_id, []),
                    "candidate_id": cand_id,
                    "candidate_rank": rank,
                    "candidate_distance": float(cand_distance),
                    "candidate_name_display": right_maps["id_to_display"].get(cand_id, str(cand_id)),
                    "candidate_name_raw": right_maps["id_to_raw"].get(cand_id, str(cand_id)),
                    "candidate_attr_kv": id_to_attrs.get(cand_id, []),
                    "is_gt": cand_id == gt_id,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
                pair_counter[query_type] += 1

    summary = {
        "dataset": args.data_choice,
        "topk": pair_args.topk,
        "same_type_only": bool(pair_args.same_type_only),
        "query_type_filter": sorted(query_type_filter),
        "query_count_by_type": dict(query_counter),
        "pair_count_by_type": dict(pair_counter),
        "rows_written": rows_written,
    }
    os.makedirs(os.path.dirname(pair_args.output_summary_json), exist_ok=True)
    with open(pair_args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"wrote {rows_written} top-k candidate pairs to {pair_args.output_jsonl}", flush=True)
    print(f"saved summary to {pair_args.output_summary_json}", flush=True)


if __name__ == "__main__":
    main()
