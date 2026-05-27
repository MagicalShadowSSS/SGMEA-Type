#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

TYPE_SET = {"Person", "Place", "Organization", "Creative Work", "Event", "Entity"}
REFINABLE_TYPES = TYPE_SET - {"Entity"}

SYSTEM_PROMPT = """You are a conservative Knowledge Graph entity type classifier.

Classify one DBpedia entity into exactly one coarse type:
- Person: human beings, named people, artists, athletes, politicians, writers, fictional characters only when clearly character-like.
- Place: geographic locations, countries, cities, regions, buildings, venues, airports, roads, stations, architectural structures.
- Organization: companies, universities, schools, political parties, sports clubs, teams, leagues, government agencies, bands, military units.
- Creative Work: films, books, albums, songs, software, TV series, languages, named written/audio/visual works.
- Event: sports seasons, tournaments, awards, ceremonies, elections, wars, battles, historical events, festivals.
- Entity: concepts, devices, animals, taxons, currencies, professions, abstract topics, ambiguous or insufficient cases.

Be conservative. If the evidence is weak or the entity does not clearly fit one of the five specific types, output Entity.
Return ONLY a JSON object with this schema:
{
  "coarse_type": "Person|Place|Organization|Creative Work|Event|Entity",
  "confidence": 0.0,
  "reason": "one short sentence"
}
"""


class LocalLlamaClassifier:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"[LLM] loading {model_path} on {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _input_ids(self, user_content):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        return self.tokenizer(SYSTEM_PROMPT + "\n\n" + user_content, return_tensors="pt").input_ids

    def classify(self, user_content, max_new_tokens):
        input_ids = self._input_ids(user_content).to(self.device)
        with self.torch.inference_mode():
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return normalize_llm_output(text), text


def parse_json_object(text):
    text = (text or "").replace("```json", "").replace("```JSON", "").replace("```", "")
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end])
    except Exception:
        return {}


def normalize_llm_output(text):
    parsed = parse_json_object(text)
    coarse_type = str(parsed.get("coarse_type", "Entity")).strip()
    if coarse_type not in TYPE_SET:
        coarse_type = "Entity"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(parsed.get("reason", "")).strip()
    return {"coarse_type": coarse_type, "confidence": confidence, "reason": reason}


def strip_uri(value):
    value = str(value or "").strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return value


def uri_label(uri):
    label = str(uri or "").rstrip("/").rsplit("/", 1)[-1]
    return label.replace("_", " ")


def compact_predicate(uri):
    return uri_label(strip_uri(uri)).replace("property/", "")


def compact_value(value):
    value = str(value or "").strip().rstrip(" .")
    if "^^" in value:
        value = value.split("^^", 1)[0]
    if value.startswith('"'):
        end = value.rfind('"')
        if end > 0:
            value = value[1:end]
    if value.startswith("<") and value.endswith(">"):
        value = uri_label(value[1:-1])
    return value[:160]


def load_entity_names(dbp_root, lang_pair):
    data_dir = Path(dbp_root) / lang_pair
    id_to_uri = {}
    uri_to_id = {}
    for filename in ["ent_ids_1", "ent_ids_2"]:
        with (data_dir / filename).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                ent_id, uri = line.split("\t", 1)
                ent_id = int(ent_id)
                id_to_uri[ent_id] = uri
                uri_to_id[uri] = ent_id
    return id_to_uri, uri_to_id


def load_attribute_context(dbp_root, lang_pair, uri_to_id, attr_limit, keep_ids=None):
    data_dir = Path(dbp_root) / lang_pair
    attrs = {}
    for filename in ["att_triples1", "att_triples2"]:
        path = data_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.rstrip("\n").split(" ", 2)
                if len(parts) < 3:
                    parts = line.rstrip("\n").split("\t", 2)
                if len(parts) < 3:
                    continue
                subj = strip_uri(parts[0])
                ent_id = uri_to_id.get(subj)
                if ent_id is None:
                    continue
                if keep_ids is not None and ent_id not in keep_ids:
                    continue
                bucket = attrs.setdefault(ent_id, [])
                if len(bucket) >= attr_limit:
                    continue
                pred = compact_predicate(parts[1])
                val = compact_value(parts[2])
                if pred and val:
                    bucket.append(f"{pred}: {val}")
    return attrs


def load_relation_context(dbp_root, lang_pair, id_to_uri, rel_limit, keep_ids=None):
    data_dir = Path(dbp_root) / lang_pair
    rels = {}
    for filename in ["triples_1", "triples_2"]:
        path = data_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = int(parts[0]), parts[1], int(parts[2])
                if keep_ids is None or h in keep_ids:
                    h_bucket = rels.setdefault(h, [])
                    if len(h_bucket) < rel_limit:
                        h_bucket.append(f"r_{r} -> {uri_label(id_to_uri.get(t, str(t)))}")
                if keep_ids is None or t in keep_ids:
                    t_bucket = rels.setdefault(t, [])
                    if len(t_bucket) < rel_limit:
                        t_bucket.append(f"{uri_label(id_to_uri.get(h, str(h)))} <- r_{r}")
    return rels


def format_list(items):
    if not items:
        return "[EMPTY]"
    return "\n".join(f"- {item}" for item in items)


def build_prompt(row, attrs=None, rels=None):
    attrs = attrs or {}
    rels = rels or {}
    ent_id = int(row.get("id"))
    return (
        f"Entity ID: {row.get('id')}\n"
        f"URI: {row.get('raw_uri', '')}\n"
        f"Name: {uri_label(row.get('raw_uri', ''))}\n"
        f"Existing DBpedia type path: {row.get('dbpedia_type_path', '')}\n"
        f"Existing type source: {row.get('type_source', '')}\n"
        f"Attributes:\n{format_list(attrs.get(ent_id, []))}\n"
        f"Relation context:\n{format_list(rels.get(ent_id, []))}\n"
        "Classify this entity conservatively into one coarse type."
    )


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done(path):
    done = {}
    if not path or not Path(path).exists():
        return done
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[int(row["id"])] = row
    return done


def should_refine(row, include_default_target):
    if row.get("coarse_type") != "Entity":
        return False
    if not include_default_target and row.get("type_source") == "default_target":
        return False
    return True


def apply_refinements(rows, refinements, min_confidence):
    now = datetime.now().isoformat()
    changed = 0
    for row in rows:
        result = refinements.get(int(row["id"]))
        if not result:
            continue
        new_type = result.get("llm_coarse_type", "Entity")
        conf = float(result.get("llm_confidence", 0.0))
        if new_type in REFINABLE_TYPES and conf >= min_confidence:
            row["coarse_type"] = new_type
            row["type_reason"] = result.get("llm_type_reason", "")
            row["type_raw_response"] = result.get("llm_raw_response", "")
            row["type_source"] = "llm_entity_refine"
            row["llm_refined_from"] = "Entity"
            row["llm_refine_confidence"] = conf
            row["generated_at"] = now
            changed += 1
    return changed


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as fp:
        for row in sorted(rows, key=lambda item: int(item["id"])):
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows, changed, attempted, output_jsonl, model_name):
    return {
        "model_name": model_name,
        "output_jsonl": str(output_jsonl),
        "total_rows": len(rows),
        "attempted_entity_rows": attempted,
        "changed_rows": changed,
        "coarse_type_distribution": dict(Counter(row.get("coarse_type", "Entity") for row in rows)),
        "type_source_distribution": dict(Counter(row.get("type_source", "") for row in rows)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Refine DBP Entity coarse types with a local LLM.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_labels_jsonl", required=True)
    parser.add_argument("--output_summary_json", required=True)
    parser.add_argument("--dbp_root", default="/gly/tongqiang/dongyufeng/data/mmkg/DBP15K")
    parser.add_argument("--lang_pair", required=True, choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--model_path", default="")
    parser.add_argument("--model_name", default="local_llama")
    parser.add_argument("--max_cases", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_confidence", type=float, default=0.70)
    parser.add_argument("--attr_limit", type=int, default=24)
    parser.add_argument("--rel_limit", type=int, default=16)
    parser.add_argument("--include_default_target", action="store_true", default=True)
    parser.add_argument("--exclude_default_target", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--apply_only", action="store_true", help="Only apply existing labels to input_jsonl.")
    parser.add_argument("--dry_run", action="store_true", help="Write prompts as labels without loading the LLM.")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.exclude_default_target:
        args.include_default_target = False
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    rows = load_rows(args.input_jsonl)
    done = load_done(args.output_labels_jsonl) if (args.resume or args.apply_only) else {}
    if args.apply_only:
        changed = apply_refinements(rows, done, args.min_confidence)
        write_jsonl(args.output_jsonl, rows)
        summary = summarize(rows, changed, 0, args.output_jsonl, args.model_name)
        summary["apply_only"] = True
        with Path(args.output_summary_json).open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    all_candidates = [row for row in rows if should_refine(row, args.include_default_target)]
    candidates = [
        row for idx, row in enumerate(all_candidates)
        if idx % args.num_shards == args.shard_id and int(row["id"]) not in done
    ]
    if args.max_cases > 0:
        candidates = candidates[: args.max_cases]
    candidate_ids = {int(row["id"]) for row in candidates}
    id_to_uri, uri_to_id = load_entity_names(args.dbp_root, args.lang_pair)
    print(
        f"[DATA] lang={args.lang_pair} shard={args.shard_id}/{args.num_shards} "
        f"candidates={len(candidates)} all_entity_candidates={len(all_candidates)}",
        flush=True,
    )
    attrs = load_attribute_context(args.dbp_root, args.lang_pair, uri_to_id, args.attr_limit, keep_ids=candidate_ids)
    rels = load_relation_context(args.dbp_root, args.lang_pair, id_to_uri, args.rel_limit, keep_ids=candidate_ids)
    print(f"[DATA] loaded attrs={len(attrs)} rels={len(rels)}", flush=True)

    classifier = None
    if not args.dry_run:
        if not args.model_path:
            raise ValueError("--model_path is required unless --dry_run is set")
        classifier = LocalLlamaClassifier(args.model_path)

    Path(args.output_labels_jsonl).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and Path(args.output_labels_jsonl).exists() else "w"
    records = list(done.values())
    with Path(args.output_labels_jsonl).open(mode, encoding="utf-8") as out_fp:
        for idx, row in enumerate(candidates, 1):
            prompt = build_prompt(row, attrs=attrs, rels=rels)
            if args.dry_run:
                normalized = {"coarse_type": "Entity", "confidence": 0.0, "reason": "DRY_RUN"}
                raw = prompt
            else:
                normalized, raw = classifier.classify(prompt, args.max_new_tokens)
            record = {
                "id": int(row["id"]),
                "raw_uri": row.get("raw_uri", ""),
                "old_coarse_type": row.get("coarse_type", "Entity"),
                "dbpedia_type_path": row.get("dbpedia_type_path", ""),
                "type_source": row.get("type_source", ""),
                "llm_coarse_type": normalized["coarse_type"],
                "llm_confidence": normalized["confidence"],
                "llm_type_reason": normalized["reason"],
                "llm_raw_response": raw,
            }
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fp.flush()
            records.append(record)
            print(f"[{idx}/{len(candidates)}] id={record['id']} -> {record['llm_coarse_type']} conf={record['llm_confidence']:.2f}", flush=True)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    refinements = {int(row["id"]): row for row in records}
    changed = apply_refinements(rows, refinements, args.min_confidence)
    write_jsonl(args.output_jsonl, rows)
    summary = summarize(rows, changed, len(candidates), args.output_jsonl, args.model_name)
    with Path(args.output_summary_json).open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
