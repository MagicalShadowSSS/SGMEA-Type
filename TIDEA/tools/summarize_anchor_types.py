#!/usr/bin/env python3
import argparse
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Summarize TIDEA coarse type anchor JSONL files.")
    parser.add_argument("--input", nargs="+", required=True)
    args = parser.parse_args()

    for path in args.input:
        total = 0
        coarse = Counter()
        source = Counter()
        attrs = Counter()
        rels = Counter()
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                typ = row.get("coarse_type", "Entity")
                coarse[typ] += 1
                source[row.get("type_source", "")] += 1
                attrs[typ] += int(row.get("attr_count", 0))
                rels[typ] += int(row.get("rel_count", 0))

        print("=" * 96)
        print(path)
        print(f"Total entities: {total}")
        print(f"{'Type':<18} | {'Count':>8} | {'Ratio':>8} | {'AvgAttr':>8} | {'AvgRel':>8}")
        print("-" * 96)
        for typ, count in coarse.most_common():
            ratio = count / total if total else 0.0
            avg_attr = attrs[typ] / count if count else 0.0
            avg_rel = rels[typ] / count if count else 0.0
            print(f"{typ:<18} | {count:>8} | {ratio:>7.2%} | {avg_attr:>8.2f} | {avg_rel:>8.2f}")
        print("Type sources:")
        for name, count in source.most_common():
            print(f"  {name or '[empty]'}: {count}")


if __name__ == "__main__":
    main()
