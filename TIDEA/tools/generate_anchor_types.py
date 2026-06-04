#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from tqdm import tqdm


TYPE_SET = ["Person", "Place", "Organization", "Creative Work", "Event", "Entity"]
TYPE_RE = re.compile(r"Person|Place|Organization|Creative Work|Event|Entity", re.IGNORECASE)


def repo_data_root():
    return Path(__file__).resolve().parents[3] / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TIDEA coarse entity type anchors from names, attributes, and relation context."
    )
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", str(repo_data_root())))
    parser.add_argument("--data_path", default=os.environ.get("DATA_PATH", "mmkg"))
    parser.add_argument("--data_choice", required=True, choices=["FBDB15K", "FBYG15K", "DBP15K"])
    parser.add_argument("--data_split", default="norm")
    parser.add_argument("--llm_path", default=os.environ.get("LLM_PATH", ""))
    parser.add_argument("--output_jsonl", default="")
    parser.add_argument("--raw_output_jsonl", default="")
    parser.add_argument("--trace_path", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--consistency_json", default="")
    parser.add_argument("--max_entities", type=int, default=-1)
    parser.add_argument("--max_attr", type=int, default=20)
    parser.add_argument("--max_rel", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--type_fix_max_new_tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_type_fix", action="store_true")
    parser.add_argument(
        "--rule_based",
        action="store_true",
        help="Use deterministic heuristics instead of loading an LLM; useful for smoke tests.",
    )
    return parser.parse_args()


def resolve_data_dir(args):
    data_path = Path(args.data_path)
    if data_path.is_absolute():
        return data_path
    return Path(args.data_root) / data_path


def split_dir(args):
    data_dir = resolve_data_dir(args)
    return data_dir / args.data_choice / args.data_split


def default_anchor_dir(args):
    return resolve_data_dir(args) / "anchors_nameless" / args.data_choice


def default_paths(args):
    out_dir = default_anchor_dir(args)
    prefix = args.data_split
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else out_dir / f"{prefix}_anchor_type_fixed.jsonl"
    raw_output_jsonl = (
        Path(args.raw_output_jsonl)
        if args.raw_output_jsonl
        else out_dir / f"{prefix}_anchor_type_raw.jsonl"
    )
    trace_path = Path(args.trace_path) if args.trace_path else out_dir / f"{prefix}_anchor_type_trace.log"
    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else output_jsonl.with_name(output_jsonl.stem + "_summary.json")
    )
    consistency_json = (
        Path(args.consistency_json)
        if args.consistency_json
        else output_jsonl.with_name(output_jsonl.stem + "_consistency.json")
    )
    return output_jsonl, raw_output_jsonl, trace_path, summary_json, consistency_json


def strip_uri(value):
    value = str(value or "").strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return value


def uri_label(value):
    value = unquote(strip_uri(value)).rstrip("/")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip()


def compact_predicate(value):
    label = uri_label(value)
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", label)
    return " ".join(words) if words else label


def compact_value(value):
    value = str(value or "").strip().rstrip(" .")
    if "^^" in value:
        value = value.split("^^", 1)[0]
    if value.startswith('"'):
        end = value.rfind('"')
        if end > 0:
            value = value[1:end]
    if value.startswith("<") and value.endswith(">"):
        value = uri_label(value)
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip()[:200]


def read_ent_ids(path):
    rows = {}
    uri_to_id = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            ent_id, uri = line.split("\t", 1)
            ent_id = int(ent_id)
            rows[ent_id] = {
                "id": ent_id,
                "raw_uri": uri,
                "name": uri_label(uri),
                "attributes": [],
                "relations": [],
            }
            for key in {uri, strip_uri(uri), f"<{strip_uri(uri)}>"}:
                uri_to_id[key] = ent_id
    return rows, uri_to_id


def parse_attr_line(line):
    line = line.rstrip("\n")
    if "\t" in line:
        parts = line.split("\t", 2)
    else:
        parts = line.split(" ", 2)
    if len(parts) < 3:
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def load_entities(args):
    root = split_dir(args)
    if not root.exists():
        raise FileNotFoundError(f"dataset split directory not found: {root}")

    entities = {}
    uri_to_id = {}
    for filename in ["ent_ids_1", "ent_ids_2"]:
        rows, mapping = read_ent_ids(root / filename)
        entities.update(rows)
        uri_to_id.update(mapping)

    for filename in ["att_triples1", "att_triples2", "att_triples_1", "att_triples_2"]:
        path = root / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                parsed = parse_attr_line(line)
                if parsed is None:
                    continue
                subj, pred, value = parsed
                ent_id = uri_to_id.get(subj) or uri_to_id.get(strip_uri(subj))
                if ent_id is None:
                    continue
                bucket = entities[ent_id]["attributes"]
                if len(bucket) < args.max_attr:
                    bucket.append((compact_predicate(pred), compact_value(value)))

    for filename in ["triples_1", "triples_2"]:
        path = root / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = int(parts[0]), parts[1], int(parts[2])
                if h in entities and len(entities[h]["relations"]) < args.max_rel:
                    entities[h]["relations"].append(f"r_{r} -> {entities.get(t, {}).get('name', t)}")
                if t in entities and len(entities[t]["relations"]) < args.max_rel:
                    entities[t]["relations"].append(f"{entities.get(h, {}).get('name', h)} <- r_{r}")

    return root, entities


class LocalLLM:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if os.environ.get("TIDEA_HF_ENDPOINT"):
            os.environ.setdefault("HF_ENDPOINT", os.environ["TIDEA_HF_ENDPOINT"])
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"[LLM] loading {model_path} on {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def input_ids(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        return self.tokenizer(system_prompt + "\n\n" + user_prompt, return_tensors="pt").input_ids

    def generate(self, system_prompt, user_prompt, max_new_tokens):
        input_ids = self.input_ids(system_prompt, user_prompt).to(self.device)
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            out_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def format_list(items):
    if not items:
        return "- None"
    return "\n".join(f"- {item}" if isinstance(item, str) else f"- {item[0]}: {item[1]}" for item in items)


TYPE_SYSTEM_PROMPT = """You are a conservative Knowledge Graph entity type classifier.

Classify the entity into exactly one coarse type:
- Person: humans, artists, athletes, politicians, writers, musicians, scientists.
- Place: countries, cities, regions, rivers, mountains, venues, geographic or administrative locations.
- Organization: companies, universities, teams, parties, agencies, clubs, publishers, military units.
- Creative Work: films, books, albums, songs, TV series, video games, artworks, publications.
- Event: wars, tournaments, ceremonies, festivals, elections, conferences, historical events.
- Entity: abstract concepts, species, languages, chemicals, awards, professions, ambiguous residual cases.

Use the name, attributes, and relation context as evidence. Return exactly:
Type: <Person|Place|Organization|Creative Work|Event|Entity>
Reason: <one short sentence>
"""


ANCHOR_SYSTEM_PROMPT = """You are a Knowledge Graph summarization expert.

Write one concise English semantic anchor for the entity. Use only the provided name, attributes, and relation context.
Keep exact useful numbers and years. Do not invent facts. Stay under 80 words.
Return exactly:
Anchor: <one sentence>
"""


def build_entity_prompt(entity, args, hide_name=False):
    name = "[HIDDEN]" if hide_name else entity["name"]
    return (
        f"Entity ID: {entity['id']}\n"
        f"Name: {name}\n"
        f"Raw URI: {entity['raw_uri']}\n"
        f"Attributes:\n{format_list(entity['attributes'][:args.max_attr])}\n"
        f"Relation context:\n{format_list(entity['relations'][:args.max_rel])}"
    )


def parse_type(text):
    match = re.search(r"Type\s*:\s*(Person|Place|Organization|Creative Work|Event|Entity)", text, re.IGNORECASE)
    if match:
        value = match.group(1)
        return "Creative Work" if value.lower() == "creative work" else value.title()
    match = TYPE_RE.search(text or "")
    if not match:
        return "Entity"
    value = match.group(0)
    return "Creative Work" if value.lower() == "creative work" else value.title()


def parse_anchor(text, entity_type):
    match = re.search(r"Anchor\s*:\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    anchor = match.group(1).strip() if match else (text or "").strip()
    anchor = re.sub(r"\s+", " ", anchor).strip()
    if not anchor:
        subject = {
            "Person": "The person",
            "Place": "The place",
            "Organization": "The organization",
            "Creative Work": "The creative work",
            "Event": "The event",
        }.get(entity_type, "The entity")
        anchor = f"{subject} is described by sparse knowledge graph evidence."
    return anchor[:600]


def heuristic_type(entity):
    text = " ".join(
        [entity.get("name", ""), entity.get("raw_uri", "")]
        + [f"{k} {v}" for k, v in entity.get("attributes", [])]
        + entity.get("relations", [])
    ).lower()
    if re.search(r"birth|death|spouse|occupation|person|actor|writer|politician|athlete", text):
        return "Person"
    if re.search(r"population|country|city|located|coordinate|latitude|longitude|place|river|mountain", text):
        return "Place"
    if re.search(r"company|university|school|team|club|party|agency|organization|publisher", text):
        return "Organization"
    if re.search(r"film|book|album|song|series|game|genre|runtime|author|director|creative", text):
        return "Creative Work"
    if re.search(r"war|battle|tournament|festival|election|event|season|conference", text):
        return "Event"
    return "Entity"


def heuristic_anchor(entity, entity_type):
    bits = []
    if entity["attributes"]:
        bits.extend(f"{k}: {v}" for k, v in entity["attributes"][:3])
    if entity["relations"]:
        bits.extend(entity["relations"][:2])
    subject = {
        "Person": "The person",
        "Place": "The place",
        "Organization": "The organization",
        "Creative Work": "The creative work",
        "Event": "The event",
    }.get(entity_type, "The entity")
    if not bits:
        return f"{subject} is represented by sparse relation context in the knowledge graph."
    return f"{subject} is described by " + "; ".join(bits) + "."


def classify_entity(entity, args, llm):
    if args.rule_based:
        entity_type = heuristic_type(entity)
        return entity_type, heuristic_anchor(entity, entity_type), "RULE_BASED", "RULE_BASED"

    prompt = build_entity_prompt(entity, args, hide_name=False)
    type_raw = llm.generate(TYPE_SYSTEM_PROMPT, prompt, args.max_new_tokens)
    entity_type = parse_type(type_raw)
    anchor_raw = llm.generate(ANCHOR_SYSTEM_PROMPT, prompt, args.max_new_tokens)
    anchor = parse_anchor(anchor_raw, entity_type)
    return entity_type, anchor, type_raw, anchor_raw


def existing_rows(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for ent_id in sorted(rows):
            fp.write(json.dumps(rows[ent_id], ensure_ascii=False) + "\n")


def load_ill_pairs(root):
    path = root / "ill_ent_ids"
    pairs = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            left, right = line.split("\t")[:2]
            pairs.append((int(left), int(right)))
    return pairs


def load_side_ids(root):
    side_ids = []
    for filename in ["ent_ids_1", "ent_ids_2"]:
        ids = set()
        with (root / filename).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    ids.add(int(line.split("\t", 1)[0]))
        side_ids.append(ids)
    return side_ids[0], side_ids[1]


def repair_type_by_aligned_name(rows, root, args, llm):
    if args.skip_type_fix:
        return rows, 0
    pairs = load_ill_pairs(root)
    fixed = 0
    for left_id, right_id in tqdm(pairs, desc="Repairing aligned type mismatches"):
        left = rows.get(left_id)
        right = rows.get(right_id)
        if not left or not right or left.get("coarse_type") == right.get("coarse_type"):
            continue
        readable_name = uri_label(right.get("raw_uri", ""))
        if not readable_name:
            continue
        if args.rule_based:
            new_type = heuristic_type({"name": readable_name, "raw_uri": right.get("raw_uri", ""), "attributes": [], "relations": []})
            raw = "RULE_BASED_TYPE_FIX"
        else:
            prompt = (
                f"Name: {readable_name}\n"
                f"Raw URI: {right.get('raw_uri', '')}\n"
                f"Previous type: {right.get('coarse_type', 'Entity')}\n"
                "Return the corrected coarse type."
            )
            raw = llm.generate(TYPE_SYSTEM_PROMPT, prompt, args.type_fix_max_new_tokens)
            new_type = parse_type(raw)
        right["coarse_type"] = new_type
        right["type_reason"] = "Repaired from aligned-pair type mismatch using readable right-side name."
        right["type_raw_response"] = raw
        right["type_source"] = "aligned_name_type_fix"
        fixed += 1
    return rows, fixed


def summarize(rows, root, output_jsonl, summary_json, consistency_json, fixed_count):
    counter = Counter(row.get("coarse_type", "Entity") for row in rows.values())
    source_counter = Counter(row.get("type_source", "") for row in rows.values())
    left_ids, right_ids = load_side_ids(root)
    pairs = load_ill_pairs(root)
    pair_match = 0
    pair_unknown = 0
    mismatch_counter = Counter()
    for left_id, right_id in pairs:
        lt = rows.get(left_id, {}).get("coarse_type", "UNKNOWN")
        rt = rows.get(right_id, {}).get("coarse_type", "UNKNOWN")
        if lt == "UNKNOWN" or rt == "UNKNOWN":
            pair_unknown += 1
        if lt == rt:
            pair_match += 1
        else:
            mismatch_counter[(lt, rt)] += 1

    summary = {
        "output_jsonl": str(output_jsonl),
        "total_rows": len(rows),
        "fixed_aligned_type_mismatches": fixed_count,
        "coarse_type_distribution": dict(counter),
        "type_source_distribution": dict(source_counter),
    }
    consistency = {
        "anchor_jsonl": str(output_jsonl),
        "split_dir": str(root),
        "left_total": len(left_ids),
        "right_total": len(right_ids),
        "pair_total": len(pairs),
        "pair_match": pair_match,
        "pair_match_rate": pair_match / len(pairs) if pairs else 0.0,
        "pair_unknown": pair_unknown,
        "pair_unknown_rate": pair_unknown / len(pairs) if pairs else 0.0,
        "top_mismatches": [
            {"left_type": left, "right_type": right, "count": count}
            for (left, right), count in mismatch_counter.most_common(20)
        ],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    consistency_json.write_text(json.dumps(consistency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary, consistency


def main():
    args = parse_args()
    output_jsonl, raw_output_jsonl, trace_path, summary_json, consistency_json = default_paths(args)
    root, entities = load_entities(args)
    print(f"[DATA] split_dir={root}")
    print(f"[DATA] entities={len(entities)}")
    print(f"[OUT] raw={raw_output_jsonl}")
    print(f"[OUT] fixed={output_jsonl}")

    if not args.rule_based and not args.llm_path:
        raise ValueError("--llm_path or LLM_PATH is required unless --rule_based is set")
    llm = None if args.rule_based else LocalLLM(args.llm_path)

    rows = existing_rows(raw_output_jsonl) if args.resume else {}
    entity_ids = sorted(entities)
    if args.max_entities > 0:
        entity_ids = entity_ids[: args.max_entities]

    raw_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_output_jsonl.open("a" if args.resume else "w", encoding="utf-8") as out_fp, trace_path.open(
        "a" if args.resume else "w", encoding="utf-8"
    ) as trace_fp:
        for ent_id in tqdm(entity_ids, desc="Generating TIDEA type anchors"):
            if ent_id in rows:
                continue
            entity = entities[ent_id]
            coarse_type, anchor, type_raw, anchor_raw = classify_entity(entity, args, llm)
            row = {
                "id": ent_id,
                "raw_uri": entity["raw_uri"],
                "name": entity["name"],
                "coarse_type": coarse_type,
                "type_reason": "Generated from entity name, attributes, and relation context.",
                "type_raw_response": type_raw,
                "type_source": "rule_based" if args.rule_based else "llm_name_attr_context",
                "attr_count": len(entity["attributes"]),
                "rel_count": len(entity["relations"]),
                "anchor": anchor,
                "raw_response": anchor_raw,
                "generated_at": dt.datetime.now().isoformat(),
            }
            rows[ent_id] = row
            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fp.flush()
            trace_fp.write(
                f"[{row['generated_at']}] id={ent_id} type={coarse_type}\n"
                f"{build_entity_prompt(entity, args, hide_name=False)}\n"
                f"TYPE_RAW:\n{type_raw}\nANCHOR_RAW:\n{anchor_raw}\n{'=' * 80}\n"
            )
            trace_fp.flush()

    rows, fixed_count = repair_type_by_aligned_name(rows, root, args, llm)
    write_jsonl(output_jsonl, rows)
    summary, consistency = summarize(rows, root, output_jsonl, summary_json, consistency_json, fixed_count)
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[CONSISTENCY]")
    print(json.dumps(consistency, ensure_ascii=False, indent=2))
    print(f"[DONE] {output_jsonl}")


if __name__ == "__main__":
    main()
