import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import unquote

import numpy as np
import torch

from config import cfg
from model import SGMEA
from src.data import load_data


class SimpleLogger:
    def info(self, msg):
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tag_hnm_k", type=int, default=32)
    parser.add_argument("--tag_hnm_chunk_size", type=int, default=2048)
    parser.add_argument("--llm_hnm_output", type=str, default="")
    parser.add_argument("--llm_hnm_type", type=str, default="Person")
    parser.add_argument("--llm_hnm_threshold", type=float, default=0.55)
    parser.add_argument("--llm_hnm_fallback_threshold", type=float, default=0.50)
    parser.add_argument("--llm_hnm_min_cases", type=int, default=200)
    parser.add_argument("--llm_hnm_max_cases", type=int, default=500)
    parser.add_argument("--llm_hnm_topn", type=int, default=5)
    parser.add_argument("--llm_hnm_attr_limit", type=int, default=40)
    parser.add_argument("--llm_hnm_min_attr_items", type=int, default=3)
    parser.add_argument("--llm_hnm_min_combined_attr_items", type=int, default=6)
    parser.add_argument(
        "--llm_hnm_write_quality",
        type=str,
        default="auto",
        choices=["auto", "high", "low", "all"],
        help="Which evidence-quality rows to write. auto writes high if available, otherwise all.",
    )
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
    llm_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        train_cfg = cfg()
        train_cfg.get_args()
        args = train_cfg.update_train_configs()
    finally:
        sys.argv = original_argv

    return llm_args, args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_camel_case(text):
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", text)
    return " ".join(words) if words else text


def normalize_raw_name(raw):
    token = (raw or "").strip()
    token = unquote(token)
    return token


def is_opaque_entity_id(token):
    token = token.strip().strip("<>")
    if "/" in token or " " in token:
        return False
    return re.fullmatch(r"[0-9a-z._-]{4,32}", token) is not None


def readable_token(raw):
    token = (raw or "").strip()
    token = token.split("^^")[0].strip()
    token = token.strip("<>").strip('"')
    token = unquote(token)
    if "/" in token:
        token = token.rsplit("/", 1)[-1]
    token = token.replace("_", " ")
    token = split_camel_case(token)
    token = re.sub(r"\s+", " ", token).strip()
    return token


def readable_entity_name(raw):
    token = normalize_raw_name(raw)
    stripped = token.strip("<>")
    if "/" in stripped:
        return readable_token(stripped)
    if is_opaque_entity_id(stripped):
        return stripped
    stripped = stripped.replace("_", " ")
    stripped = split_camel_case(stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def truncate_text(text, max_len=160):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def load_id_name_maps(file_dir):
    side_maps = {}
    for side, filename in [
        ("left", "ent_ids_1"),
        ("right", "ent_ids_2"),
    ]:
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
                raw_name = parts[1]
                id_to_raw[ent_id] = normalize_raw_name(raw_name)
                id_to_display[ent_id] = readable_entity_name(raw_name)
                name_to_id[raw_name] = ent_id
                name_to_id[raw_name.strip("<>")] = ent_id
        side_maps[side] = {
            "id_to_display": id_to_display,
            "id_to_raw": id_to_raw,
            "name_to_id": name_to_id,
        }
    return side_maps


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
    name = normalize_raw_name(name).strip()
    stripped = name.strip("<>")
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
            style = detect_entity_style(ent_name)
            style_counter[style] += 1
            if sum(style_counter.values()) >= 200:
                break
    return style_counter


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
        file_specs.append((filename, name_to_id, dominant_style))

    for filename, name_to_id, _dominant_style in file_specs:
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
                key = readable_token(attr_name)
                val = readable_token(value)
                if not key or not val:
                    continue
                item = f"{truncate_text(key, 80)}: {truncate_text(val, 160)}"
                if item in seen[ent_id]:
                    continue
                seen[ent_id].add(item)
                if len(id_to_attrs[ent_id]) < attr_limit:
                    id_to_attrs[ent_id].append(item)
    return id_to_attrs


def load_full_ill(file_dir):
    pairs = set()
    path = os.path.join(file_dir, "ill_ent_ids")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, right = line.split("\t")[:2]
            pairs.add((int(left), int(right)))
    return pairs


def load_checkpoint(model, args):
    if not args.model_name_save:
        raise ValueError("--model_name_save is required")
    save_path = os.path.join(args.data_path, args.model_name, "save", f"{args.model_name_save}.pkl")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"checkpoint not found: {save_path}")
    state = torch.load(save_path, map_location=args.device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return save_path


def direction_keys(direction):
    if direction == "right":
        return "left_to_right", "right_cache", "right_sim"
    return "right_to_left", "left_cache", "left_sim"


def collect_cases(cache, train_links, entity_type_ids, type_names, llm_args, threshold):
    directions = []
    if llm_args.llm_hnm_direction in ["both", "left_to_right"]:
        directions.append("right")
    if llm_args.llm_hnm_direction in ["both", "right_to_left"]:
        directions.append("left")

    train_links_cpu = train_links.detach().cpu()
    type_ids_cpu = entity_type_ids.detach().cpu()
    type_filter = (llm_args.llm_hnm_type or "").strip()
    cases = []

    for direction in directions:
        direction_label, cache_key, sim_key = direction_keys(direction)
        neg_cache = cache[cache_key].detach().cpu()
        sim_cache = cache[sim_key].detach().cpu()
        valid_rows = torch.nonzero(neg_cache[:, 0] >= 0, as_tuple=False).flatten().tolist()

        for row in valid_rows:
            left_id = int(train_links_cpu[row, 0].item())
            right_id = int(train_links_cpu[row, 1].item())
            if direction == "right":
                anchor_id = left_id
                positive_id = right_id
            else:
                anchor_id = right_id
                positive_id = left_id

            type_id = int(type_ids_cpu[anchor_id].item())
            type_name = type_names[type_id] if 0 <= type_id < len(type_names) else str(type_id)
            top1_sim = float(sim_cache[row, 0].item())
            if type_filter and type_name != type_filter:
                continue
            if top1_sim < threshold:
                continue
            topn = min(int(llm_args.llm_hnm_topn), int(neg_cache.shape[1]), int(sim_cache.shape[1]))
            for idx in range(topn):
                negative_id = int(neg_cache[row, idx].item())
                if negative_id < 0:
                    continue
                candidate_sim = float(sim_cache[row, idx].item())
                cases.append(
                    {
                        "row": row,
                        "candidate_rank": idx + 1,
                        "candidate_sim": candidate_sim,
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

    if llm_args.llm_hnm_sort_by == "top1_sim_desc":
        cases.sort(key=lambda item: (-item["top1_sim"], item["candidate_rank"]))
    else:
        rng = random.Random(llm_args.llm_hnm_seed)
        rng.shuffle(cases)
    return cases


def get_side_fields(case, left_maps, right_maps):
    if case["direction"] == "right":
        anchor_display = left_maps["id_to_display"].get(case["anchor_id"], str(case["anchor_id"]))
        anchor_raw = left_maps["id_to_raw"].get(case["anchor_id"], str(case["anchor_id"]))
        positive_display = right_maps["id_to_display"].get(case["positive_id"], str(case["positive_id"]))
        positive_raw = right_maps["id_to_raw"].get(case["positive_id"], str(case["positive_id"]))
        negative_display = right_maps["id_to_display"].get(case["negative_id"], str(case["negative_id"]))
        negative_raw = right_maps["id_to_raw"].get(case["negative_id"], str(case["negative_id"]))
    else:
        anchor_display = right_maps["id_to_display"].get(case["anchor_id"], str(case["anchor_id"]))
        anchor_raw = right_maps["id_to_raw"].get(case["anchor_id"], str(case["anchor_id"]))
        positive_display = left_maps["id_to_display"].get(case["positive_id"], str(case["positive_id"]))
        positive_raw = left_maps["id_to_raw"].get(case["positive_id"], str(case["positive_id"]))
        negative_display = left_maps["id_to_display"].get(case["negative_id"], str(case["negative_id"]))
        negative_raw = left_maps["id_to_raw"].get(case["negative_id"], str(case["negative_id"]))
    return (
        anchor_display,
        anchor_raw,
        positive_display,
        positive_raw,
        negative_display,
        negative_raw,
    )


def evidence_quality(anchor_attr_kv, positive_attr_kv, negative_attr_kv, llm_args, coarse_type):
    min_attr_items = llm_args.llm_hnm_min_attr_items
    min_combined_attr_items = llm_args.llm_hnm_min_combined_attr_items
    if coarse_type == "Organization":
        min_attr_items = min(min_attr_items, 2)
        min_combined_attr_items = min(min_combined_attr_items, 4)

    positive_ok = len(positive_attr_kv) >= min_attr_items
    negative_ok = len(negative_attr_kv) >= min_attr_items
    combined_ok = (len(positive_attr_kv) + len(negative_attr_kv)) >= min_combined_attr_items
    passed = positive_ok and negative_ok and combined_ok
    notes = []
    if not positive_ok:
        notes.append("low_positive_attr")
    if not negative_ok:
        notes.append("low_negative_attr")
    if not combined_ok:
        notes.append("low_combined_attr")
    return ("high" if passed else "low"), notes


def has_readable_identity(display_name, raw_name):
    display_name = (display_name or "").strip()
    raw_name = (raw_name or "").strip()
    if "dbpedia.org/resource/" in raw_name:
        return True
    if not display_name:
        return False
    if display_name.startswith("/m/"):
        return False
    compact = display_name.replace(" ", "")
    if len(compact) <= 6 and any(ch.isdigit() for ch in compact):
        return False
    return True


def enrich_case(case, dataset, left_maps, right_maps, id_to_attrs, full_ill, llm_args):
    (
        anchor_display,
        anchor_raw,
        positive_display,
        positive_raw,
        negative_display,
        negative_raw,
    ) = get_side_fields(case, left_maps, right_maps)
    if case["direction"] == "right":
        known_full_ill_hit = (case["anchor_id"], case["negative_id"]) in full_ill
    else:
        known_full_ill_hit = (case["negative_id"], case["anchor_id"]) in full_ill

    anchor_attr_kv = id_to_attrs.get(case["anchor_id"], [])
    positive_attr_kv = id_to_attrs.get(case["positive_id"], [])
    negative_attr_kv = id_to_attrs.get(case["negative_id"], [])
    quality, notes = evidence_quality(
        anchor_attr_kv,
        positive_attr_kv,
        negative_attr_kv,
        llm_args,
        case["coarse_type"],
    )
    positive_name_usable = has_readable_identity(positive_display, positive_raw)
    negative_name_usable = has_readable_identity(negative_display, negative_raw)
    anchor_name_usable = has_readable_identity(anchor_display, anchor_raw)

    return {
        "dataset": dataset,
        "direction": case["direction_label"],
        "coarse_type": case["coarse_type"],
        "cache_row": case["row"],
        "candidate_rank": case.get("candidate_rank", 1),
        "candidate_sim": round(float(case.get("candidate_sim", case["top1_sim"])), 6),
        "topn_limit": case.get("topn_limit", 1),
        "top1_sim": round(case["top1_sim"], 6),
        "anchor_id": case["anchor_id"],
        "anchor_name": anchor_display,
        "anchor_name_display": anchor_display,
        "anchor_name_raw": anchor_raw,
        "anchor_attr_kv": anchor_attr_kv,
        "positive_id": case["positive_id"],
        "positive_name": positive_display,
        "positive_name_display": positive_display,
        "positive_name_raw": positive_raw,
        "positive_attr_kv": positive_attr_kv,
        "negative_id": case["negative_id"],
        "negative_name": negative_display,
        "negative_name_display": negative_display,
        "negative_name_raw": negative_raw,
        "negative_attr_kv": negative_attr_kv,
        "known_full_ill_hit": known_full_ill_hit,
        "evidence_quality": quality,
        "evidence_notes": notes,
        "anchor_name_usable": anchor_name_usable,
        "positive_name_usable": positive_name_usable,
        "negative_name_usable": negative_name_usable,
        "pair_name_usable": positive_name_usable and negative_name_usable,
        "llm_instruction": (
            "The anchor may have an unreadable Freebase ID. Use the positive entity as the "
            "known aligned identity reference for the anchor. The final relationship label must "
            "describe the candidate negative relative to the anchor/positive identity only. "
            "Do not output same_entity merely because the anchor and positive are the same entity; "
            "output same_entity only when the candidate negative is also the exact same real-world entity."
        ),
    }


def default_output_path(args, llm_args, threshold):
    type_part = (llm_args.llm_hnm_type or "all").replace(" ", "_")
    filename = f"{args.data_choice}_{args.data_rate}_{type_part}_llm_hnm_cases_t{threshold:.2f}.jsonl"
    return os.path.join(args.data_path, args.model_name, "llm_hnm_cases", filename)


def main():
    llm_args, args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because SGMEA currently constructs CUDA tensors internally.")

    set_seed(llm_args.llm_hnm_seed)
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)
    logger = SimpleLogger()

    kgs, non_train, train_set, eval_set, test_set, test_ill_ = load_data(logger, args)
    model = SGMEA(kgs, args).cuda()
    ckpt_path = load_checkpoint(model, args)
    model.eval()
    logger.info(f"loaded checkpoint: {ckpt_path}")

    with torch.no_grad():
        joint_emb, _ = model.joint_emb_generat()
        cache = model.update_hard_negative_cache(
            joint_emb=joint_emb,
            train_links=train_set.data,
            k=llm_args.tag_hnm_k,
            chunk_size=llm_args.tag_hnm_chunk_size,
            logger=logger,
        )

    train_links = torch.as_tensor(train_set.data, dtype=torch.long)
    threshold = llm_args.llm_hnm_threshold
    cases = collect_cases(cache, train_links, model.entity_type_ids, model.top_type_names, llm_args, threshold)
    if len(cases) < llm_args.llm_hnm_min_cases and llm_args.llm_hnm_fallback_threshold < threshold:
        logger.info(
            f"Only {len(cases)} cases found at threshold={threshold:.2f}; "
            f"falling back to threshold={llm_args.llm_hnm_fallback_threshold:.2f}"
        )
        threshold = llm_args.llm_hnm_fallback_threshold
        cases = collect_cases(cache, train_links, model.entity_type_ids, model.top_type_names, llm_args, threshold)

    if llm_args.llm_hnm_max_cases > 0:
        cases = cases[:llm_args.llm_hnm_max_cases]

    file_dir = os.path.join(args.data_path, args.data_choice, args.data_split)
    side_maps = load_id_name_maps(file_dir)
    left_maps = side_maps["left"]
    right_maps = side_maps["right"]
    id_to_attrs = load_attribute_kv(
        file_dir,
        left_name_to_id=left_maps["name_to_id"],
        right_name_to_id=right_maps["name_to_id"],
        attr_limit=llm_args.llm_hnm_attr_limit,
    )
    full_ill = load_full_ill(file_dir)

    output_path = llm_args.llm_hnm_output or default_output_path(args, llm_args, threshold)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enriched_cases = []
    quality_counts = defaultdict(int)
    note_counts = defaultdict(int)
    for case in cases:
        obj = enrich_case(
            case=case,
            dataset=args.data_choice,
            left_maps=left_maps,
            right_maps=right_maps,
            id_to_attrs=id_to_attrs,
            full_ill=full_ill,
            llm_args=llm_args,
        )
        enriched_cases.append(obj)
        quality_counts[obj["evidence_quality"]] += 1
        for note in obj["evidence_notes"]:
            note_counts[note] += 1

    if llm_args.llm_hnm_write_quality == "all":
        selected_cases = enriched_cases
    elif llm_args.llm_hnm_write_quality == "high":
        selected_cases = [obj for obj in enriched_cases if obj["evidence_quality"] == "high"]
    elif llm_args.llm_hnm_write_quality == "low":
        selected_cases = [obj for obj in enriched_cases if obj["evidence_quality"] == "low"]
    else:
        selected_cases = [obj for obj in enriched_cases if obj["evidence_quality"] == "high"]

    if llm_args.llm_hnm_write_quality == "auto" and len(selected_cases) == 0:
        logger.info(
            "No high-evidence cases passed the current attribute filter; "
            "falling back to writing all cases with evidence_quality tags."
        )
        selected_cases = enriched_cases

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in selected_cases:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        "LLM HNM evidence summary | "
        + " ".join(f"{k}={v}" for k, v in sorted(quality_counts.items()))
    )
    if note_counts:
        logger.info(
            "LLM HNM evidence notes | "
            + " ".join(f"{k}={v}" for k, v in sorted(note_counts.items()))
        )
    logger.info(
        f"wrote {written} LLM HNM cases to {output_path} "
        f"(type={llm_args.llm_hnm_type or 'all'}, threshold={threshold:.2f})"
    )


if __name__ == "__main__":
    main()
