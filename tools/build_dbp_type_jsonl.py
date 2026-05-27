#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT_MAP = {
    "Person": "Person",
    "Place": "Place",
    "Organisation": "Organization",
    "Organization": "Organization",
    "Work": "Creative Work",
    "work": "Creative Work",
    "Language": "Creative Work",
    "ArchitecturalStructure": "Place",
    "Event": "Event",
    "SportsSeason": "Event",
}


def read_ent_ids(path):
    mapping = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            ent_id, uri = line.split("\t", 1)
            mapping[int(ent_id)] = uri
    return mapping


def read_ill(path):
    pairs = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            left, right = line.split("\t")[:2]
            pairs.append((int(left), int(right)))
    return pairs


def coarse_type(type_path):
    if not type_path:
        return "Entity"
    root = type_path.split("➔", 1)[0].strip()
    root = root.split("->", 1)[0].strip()
    return ROOT_MAP.get(root, "Entity")


def build_one(lang_pair, dbp_root, type_dir, output_dir):
    data_dir = dbp_root / lang_pair
    type_path = type_dir / f"{lang_pair}_source_ultimate_types_filled.json"
    out_path = output_dir / f"dbp_{lang_pair}_type_fixed.jsonl"
    summary_path = output_dir / f"dbp_{lang_pair}_type_fixed_summary.json"

    source_entities = read_ent_ids(data_dir / "ent_ids_1")
    target_entities = read_ent_ids(data_dir / "ent_ids_2")
    ill_pairs = read_ill(data_dir / "ill_ent_ids")

    with type_path.open("r", encoding="utf-8") as fp:
        source_row_types = json.load(fp)

    entity_rows = {}
    now = datetime.now().isoformat()

    for row_key, dbpedia_type in source_row_types.items():
        ent_id = int(row_key)
        if ent_id not in source_entities:
            continue
        mapped = coarse_type(dbpedia_type)
        entity_rows[ent_id] = {
            "id": ent_id,
            "raw_uri": source_entities.get(ent_id, ""),
            "coarse_type": mapped,
            "type_reason": f"Mapped from DBpedia ontology path: {dbpedia_type}",
            "type_raw_response": f"Type: {mapped}",
            "dbpedia_type_path": dbpedia_type,
            "type_source": "dbpedia_source",
            "generated_at": now,
        }

    propagated = 0
    skipped_source_conflicts = 0
    for source_id, target_id in ill_pairs:
        source_row = entity_rows.get(source_id)
        if source_row is None:
            continue
        if target_id in entity_rows:
            skipped_source_conflicts += 1
            continue
        entity_rows[target_id] = {
            "id": target_id,
            "raw_uri": target_entities.get(target_id, ""),
            "coarse_type": source_row["coarse_type"],
            "type_reason": f"Propagated from aligned source entity {source_id}.",
            "type_raw_response": source_row["type_raw_response"],
            "dbpedia_type_path": source_row.get("dbpedia_type_path", ""),
            "type_source": "alignment_propagated",
            "source_entity_id": source_id,
            "generated_at": now,
        }
        propagated += 1

    for ent_id, uri in target_entities.items():
        if ent_id in entity_rows:
            continue
        entity_rows[ent_id] = {
            "id": ent_id,
            "raw_uri": uri,
            "coarse_type": "Entity",
            "type_reason": "No aligned source type available; defaulted to Entity.",
            "type_raw_response": "Type: Entity",
            "dbpedia_type_path": "Entity",
            "type_source": "default_target",
            "generated_at": now,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for ent_id in sorted(entity_rows):
            fp.write(json.dumps(entity_rows[ent_id], ensure_ascii=False) + "\n")

    counter = Counter(row["coarse_type"] for row in entity_rows.values())
    source_counter = Counter(row["type_source"] for row in entity_rows.values())
    summary = {
        "lang_pair": lang_pair,
        "output": str(out_path),
        "total_rows": len(entity_rows),
        "source_entities": len(source_entities),
        "target_entities": len(target_entities),
        "ill_pairs": len(ill_pairs),
        "propagated_target_types": propagated,
        "skipped_source_conflicts": skipped_source_conflicts,
        "coarse_type_distribution": dict(counter),
        "type_source_distribution": dict(source_counter),
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbp_root", default="/gly/tongqiang/dongyufeng/data/mmkg/DBP15K")
    parser.add_argument("--type_dir", default="entity_type")
    parser.add_argument("--output_dir", default="entity_type")
    parser.add_argument("--lang_pairs", default="zh_en,ja_en,fr_en")
    args = parser.parse_args()

    dbp_root = Path(args.dbp_root)
    type_dir = Path(args.type_dir)
    output_dir = Path(args.output_dir)
    summaries = []
    for lang_pair in [item.strip() for item in args.lang_pairs.split(",") if item.strip()]:
        summaries.append(build_one(lang_pair, dbp_root, type_dir, output_dir))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
