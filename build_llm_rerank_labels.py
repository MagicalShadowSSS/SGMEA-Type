import argparse
import json
import os
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are an expert Knowledge Graph Ontologist. Your task is to analyze whether a test-time candidate should be promoted or demoted during entity alignment reranking.

You are given:
- a Query entity from the source KG
- the Ground-Truth aligned target entity for that Query
- a Candidate target entity currently appearing in the model's Top-k list

Your job is NOT to do global entity alignment from scratch.
Your job is to judge the semantic relationship between the Candidate and the Query/Ground-Truth identity so we can safely rerank the candidate list.

Use the Ground-Truth target entity as the definitive identity anchor for who the Query truly is.

Return ONLY a JSON object with the following schema:
{
  "query_fine_type": "string",
  "candidate_fine_type": "string",
  "relationship_status": "string (MUST BE EXACTLY ONE OF: same_entity, closely_related, safe_negative, unknown)",
  "relationship_kind": "string",
  "confidence": "float (0.0 to 1.0)",
  "reasoning": "string (1 brief sentence explaining the relationship)"
}

Definitions:
- same_entity: Candidate is likely the exact same real-world entity as the Query/Ground-Truth identity. This candidate should be promoted.
- closely_related: Candidate is not the same entity, but is strongly related in the real world, e.g. collaborator, sibling, same franchise, region-city, capital-country. This candidate should not be harshly demoted.
- safe_negative: Candidate is a distinct entity with no strong topology-sensitive relation to the Query/Ground-Truth identity. This candidate can be safely demoted.
- unknown: Evidence is insufficient.

### EXAMPLES ###

Example 1 (safe_negative / Person):
Query: {name: "027dtv3", attributes: "born in London, English actor"}
Ground Truth: {name: "Jack Huston", attributes: "English actor, known for Boardwalk Empire"}
Candidate: {name: "Charlie Cox", attributes: "English actor, known for Daredevil"}
Output:
{
  "query_fine_type": "actor",
  "candidate_fine_type": "actor",
  "relationship_status": "safe_negative",
  "relationship_kind": "same_profession",
  "confidence": 0.95,
  "reasoning": "Charlie Cox and Jack Huston are different actors without a strong identity-level relation."
}

Example 2 (closely_related / Creative Work):
Query: {name: "03t95n", attributes: "fantasy action film, released in 2002"}
Ground Truth: {name: "The Scorpion King", attributes: "fantasy action adventure film starring Dwayne Johnson"}
Candidate: {name: "The Mummy Returns", attributes: "fantasy adventure film, related franchise context"}
Output:
{
  "query_fine_type": "film",
  "candidate_fine_type": "film",
  "relationship_status": "closely_related",
  "relationship_kind": "same_franchise",
  "confidence": 0.95,
  "reasoning": "The candidate is not the same film but is closely tied through franchise context."
}

Example 3 (closely_related / Place):
Query: {name: "01zv", attributes: "autonomous community in Spain"}
Ground Truth: {name: "Catalonia", attributes: "autonomous community of Spain"}
Candidate: {name: "Barcelona", attributes: "capital city of Catalonia"}
Output:
{
  "query_fine_type": "region",
  "candidate_fine_type": "city",
  "relationship_status": "closely_related",
  "relationship_kind": "region_city",
  "confidence": 0.98,
  "reasoning": "Barcelona is not the same entity as Catalonia, but it is tightly linked as its capital city."
}

Example 4 (safe_negative / Organization):
Query: {name: "0bwfn", attributes: "educational institution, private university"}
Ground Truth: {name: "New York University", attributes: "private research university in New York City"}
Candidate: {name: "Yale University", attributes: "private Ivy League university in New Haven"}
Output:
{
  "query_fine_type": "university",
  "candidate_fine_type": "university",
  "relationship_status": "safe_negative",
  "relationship_kind": "same_institution_type",
  "confidence": 0.95,
  "reasoning": "The candidate is another university rather than the same institution."
}

### CRITICAL INSTRUCTIONS ###
1. First briefly reason about the Query/Ground-Truth identity and the Candidate.
2. Then output EXACTLY ONE JSON object.
3. Do NOT wrap the JSON in markdown code blocks.
4. The final JSON must start with { and end with }.
5. If evidence is mixed, prefer closely_related over safe_negative when a strong relation is obvious.
6. If evidence is insufficient, output unknown rather than guessing.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM rerank label diagnosis on extracted test top-k pairs.")
    parser.add_argument("--input_jsonl", required=True, help="Top-k test candidate pairs JSONL")
    parser.add_argument("--output_jsonl", required=True, help="Per-pair LLM label results JSONL")
    parser.add_argument("--output_summary_json", required=True, help="Aggregated summary JSON")
    parser.add_argument("--model_path", required=True, help="Local Llama model path")
    parser.add_argument("--model_name", default="local_llama", help="Label recorded in summary")
    parser.add_argument("--max_cases", type=int, default=100)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--coarse_types", type=str, default="", help="Optional comma-separated query coarse types to keep")
    return parser.parse_args()


class LocalLlamaJudge:
    def __init__(self, model_path):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"[LLM] Initializing local model from: {model_path}", flush=True)
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
        print(f"[LLM] Ready on {self.device}", flush=True)

    def apply_chat_template(self, system_prompt, user_content):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        return self.tokenizer(f"{system_prompt}\n\n{user_content}", return_tensors="pt").input_ids

    def generate(self, system_prompt, user_content, max_new_tokens):
        input_ids = self.apply_chat_template(system_prompt, user_content).to(self.device)
        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return response


def load_cases(path, coarse_types):
    keep = {x.strip() for x in coarse_types.split(",") if x.strip()}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if keep and row.get("query_coarse_type") not in keep:
                continue
            rows.append(row)
    return rows


def load_done_keys(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row.get("dataset"), row.get("query_id"), row.get("candidate_id")))
    return done


def attrs_to_text(kv_list):
    return "\n".join(kv_list) if kv_list else "[EMPTY]"


def build_user_prompt(case):
    return (
        f"Dataset: {case.get('dataset', '')}\n"
        f"Query Coarse Type: {case.get('query_coarse_type', '')}\n"
        f"Candidate Coarse Type: {case.get('candidate_coarse_type', '')}\n"
        f"Candidate Rank: {case.get('candidate_rank', '')}\n"
        f"Candidate Distance: {case.get('candidate_distance', '')}\n\n"
        f"Query:\n"
        f"- name_display: {case.get('query_name_display', '')}\n"
        f"- name_raw: {case.get('query_name_raw', '')}\n"
        f"- attributes:\n{attrs_to_text(case.get('query_attr_kv', []))}\n\n"
        f"Ground Truth (true aligned identity of Query):\n"
        f"- name_display: {case.get('gt_name_display', '')}\n"
        f"- name_raw: {case.get('gt_name_raw', '')}\n"
        f"- attributes:\n{attrs_to_text(case.get('gt_attr_kv', []))}\n\n"
        f"Candidate:\n"
        f"- name_display: {case.get('candidate_name_display', '')}\n"
        f"- name_raw: {case.get('candidate_name_raw', '')}\n"
        f"- attributes:\n{attrs_to_text(case.get('candidate_attr_kv', []))}\n\n"
        "Let's analyze this step-by-step. First explain the relationship briefly, then output exactly one raw JSON object."
    )


def extract_json(text):
    if not text:
        return {}
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "")
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as exc:
        print(f"[LLM JSON Decode Error] {exc}", flush=True)
    return {}


def normalize_output(parsed):
    status = str(parsed.get("relationship_status", "unknown")).strip()
    if status not in {"same_entity", "closely_related", "safe_negative", "unknown"}:
        status = "unknown"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "query_fine_type": str(parsed.get("query_fine_type", "unknown")).strip() or "unknown",
        "candidate_fine_type": str(parsed.get("candidate_fine_type", "unknown")).strip() or "unknown",
        "relationship_status": status,
        "relationship_kind": str(parsed.get("relationship_kind", "unknown")).strip() or "unknown",
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def call_llm_case(judge, args, case):
    content = judge.generate(SYSTEM_PROMPT, build_user_prompt(case), args.max_new_tokens)
    parsed = extract_json(content)
    normalized = normalize_output(parsed)
    return normalized, content


def build_summary(records, model_name):
    status_counter = Counter(r["llm_relationship_status"] for r in records)
    relation_counter = Counter(r["llm_relationship_kind"] for r in records)
    query_type_counter = Counter(r["llm_query_fine_type"] for r in records)
    cand_type_counter = Counter(r["llm_candidate_fine_type"] for r in records)
    coarse_counter = defaultdict(Counter)
    for r in records:
        coarse_counter[r["query_coarse_type"]][r["llm_relationship_status"]] += 1
    return {
        "teacher_model": model_name,
        "case_count": len(records),
        "relationship_status_counts": dict(status_counter),
        "relationship_kind_counts": dict(relation_counter),
        "query_fine_type_counts": dict(query_type_counter),
        "candidate_fine_type_counts": dict(cand_type_counter),
        "coarse_type_breakdown": {k: dict(v) for k, v in coarse_counter.items()},
    }


def main():
    args = parse_args()
    judge = LocalLlamaJudge(args.model_path)
    cases = load_cases(args.input_jsonl, args.coarse_types)
    if args.resume:
        done = load_done_keys(args.output_jsonl)
        cases = [c for c in cases if (c.get("dataset"), c.get("query_id"), c.get("candidate_id")) not in done]
    cases = cases[: args.max_cases]

    print(f"Running LLM rerank-label diagnosis on {len(cases)} cases...", flush=True)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    records = []
    if args.resume and os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    run_start = time.time()
    for idx, case in enumerate(cases, start=1):
        step_start = time.time()
        try:
            result, raw = call_llm_case(judge, args, case)
            row = dict(case)
            row.update(
                {
                    "llm_query_fine_type": result["query_fine_type"],
                    "llm_candidate_fine_type": result["candidate_fine_type"],
                    "llm_relationship_status": result["relationship_status"],
                    "llm_relationship_kind": result["relationship_kind"],
                    "llm_confidence": result["confidence"],
                    "llm_reasoning": result["reasoning"],
                    "llm_raw_response": raw,
                    "llm_error": "",
                }
            )
        except Exception as exc:
            print(f"[LLM Error] query_id={case.get('query_id')} cand_id={case.get('candidate_id')} error={type(exc).__name__}: {exc}", flush=True)
            continue

        records.append(row)
        with open(args.output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        step_elapsed = time.time() - step_start
        elapsed = time.time() - run_start
        avg_step = elapsed / idx if idx > 0 else 0.0
        eta = (len(cases) - idx) * avg_step
        print(
            f"[{idx}/{len(cases)}] query={case.get('query_id')} cand={case.get('candidate_id')} "
            f"status={row['llm_relationship_status']} kind={row['llm_relationship_kind']} "
            f"conf={row['llm_confidence']:.2f} step={step_elapsed:.1f}s elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary = build_summary(records, args.model_name)
    os.makedirs(os.path.dirname(args.output_summary_json), exist_ok=True)
    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary written to {args.output_summary_json}", flush=True)


if __name__ == "__main__":
    main()
