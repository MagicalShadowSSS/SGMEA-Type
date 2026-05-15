import argparse
import json
import os
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are an expert Knowledge Graph Ontologist and Contrastive Learning Architect. Your task is to analyze the relationship between an Anchor/Positive identity and a Candidate Negative entity to determine whether the candidate should be forcefully pushed apart or softly protected in contrastive learning.

The Anchor entity often uses an unreadable Freebase ID. Therefore, you MUST use the provided "Positive" entity, which is the known correctly aligned identity of the Anchor, as the definitive reference for who the Anchor truly is.

CRITICAL IDENTITY RULE:
- The Positive entity is only an identity reference for the Anchor.
- The final label `relationship_status` MUST describe the relationship between:
  1. the Anchor/Positive identity
  2. the Candidate Negative
- You must NOT output `same_entity` merely because Anchor and Positive are the same entity.
- Output `same_entity` ONLY when the Candidate Negative is also the exact same real-world entity as the Anchor/Positive identity (for example alias, alternate title, birth name, canonical rename).
- For Creative Works, sequels, prequels, remakes, reboots, franchise installments, numbered follow-ups, different seasons, different episodes, and different editions/versions are NOT `same_entity`.
- Such Creative Work variants should usually be labeled `closely_related`, not `same_entity`.

Attributes may be sparse or partially missing. When attributes are sparse, rely primarily on:
- the Positive identity
- the Candidate identity
- the coarse type
- the semantic plausibility of their relationship
- your general world knowledge about famous named entities

For this experiment, false negatives are more harmful than false positives:
- A false negative means a truly close hard negative is forcefully pushed apart and may shatter local representation topology.
- A false positive only soft-protects one extra hard negative, which is acceptable for this relation-aware training signal.
- Therefore, for same-coarse-type hard negatives, prefer high recall for `closely_related`.

Return ONLY a JSON object with the following schema:
{
  "anchor_fine_type": "string",
  "candidate_fine_type": "string",
  "reasoning": "string (1 brief sentence explaining the relationship first)",
  "relationship_status": "string (MUST BE EXACTLY ONE OF: same_entity, closely_related, safe_negative, unknown)",
  "relationship_kind": "string",
  "confidence": "float (0.0 to 1.0)"
}

Prefer normalized fine_type labels from a compact vocabulary when possible, such as:
- actor
- athlete
- politician
- musician
- writer
- fictional_character
- university
- company
- record_label
- sports_team
- country
- city
- region
- film
- tv_series
- song
- book

DEFINITIONS for relationship_status:
- same_entity: They refer to the exact same real-world entity, e.g. aliases or alternative names. DO NOT PUSH APART.
- closely_related: The candidate is NOT the same entity, but it is a semantically or contextually close hard negative. They may share a narrow context, same profession within the same era or circle, strong collaboration, sibling relationship in a local cluster, same small group/franchise, direct relation, or high-risk semantic confusion. If the model strongly pushes them far away, it may damage useful local semantic topology.
- safe_negative: Distinct entities with no meaningful narrow shared context. Use this only when the pair is clearly broad, generic, cross-domain, or unrelated and it is safe to push apart.
- unknown: Attributes and names are insufficient to make a judgment.

Graph Topology Rule:
- Parent-child relationships are closely_related because their representation spaces should overlap.
- Local sibling relationships can be closely_related when they share a narrow parent/context and are likely to be high-risk semantic confusions, e.g. nearby counties in the same state, cities in the same metropolitan region, companies in the same group, teams in the same league context, or works in the same franchise.
- Broad sibling relationships are safe_negative only when they are clearly generic and not part of a narrow shared context.
- If both entities are recognizable and share the same coarse type, default to closely_related unless they are obviously cross-domain, pure name collisions, or clearly unrelated in time, location, and context.
- Sharing the same state, country, city, coarse type, industry, profession, franchise, or category can be enough for closely_related if the pair sits in the same local semantic neighborhood.

Category-agnostic close rules:
- Treat same-coarse-type hard negatives as `closely_related` by default when both names are recognizable and there is any plausible shared context.
- For Person, Place, Organization, and Creative Work, `closely_related` includes local siblings, same narrow domain, same era/circle, same franchise/system/group, direct competitors, collaborators, parent-child pairs, and other high-risk semantic neighbors.
- `safe_negative` is reserved for cases that are clearly unrelated or only share a very broad label with no plausible local neighborhood.

Hard close rules for Person and Organization:
- Person pairs that are siblings, co-founders, long-term creative/business partners, same notable group/team/troupe, same administration, same political circle, or repeated collaborators MUST be `closely_related`, even if they are distinct people with different roles or dates.
- Organization pairs that are in the same corporate group, label family, university/system, political system, sports league, niche market, direct rivalry, merger/acquisition history, or parent/subsidiary chain MUST be `closely_related`, even if they are distinct organizations with different founding years.
- For Person, different birth dates, death dates, roles, titles, or centuries are NOT sufficient reasons to choose `safe_negative` if the pair is still in the same broad professional, political, creative, or historical circle.
- For Organization, different founding years, label names, corporate titles, or active years are NOT sufficient reasons to choose `safe_negative` if the pair still belongs to the same group, network, rivalry, or business ecosystem.

IMPORTANT NEGATIVE RULES:
- Sharing a broad profession, e.g. "both are American actors", can still be closely_related if they are from the same era, creative circle, administration, team, or collaboration network.
- Sharing a broad location, e.g. "both are cities in Europe", can still be closely_related if they are adjacent, in the same immediate local hierarchy, or in the same regional cluster.
- Sharing a broad industry, e.g. "both are software companies", can still be closely_related if they are in the same group, same market niche, or direct partnership network.
- Sharing only a common name across unrelated fine types is safe_negative and should be pushed apart for disambiguation.
- Rule of thumb: for same-coarse-type hard negatives, prefer closely_related whenever there is any plausible shared context, same profession or era, local sibling topology, frequent collaboration, or high-risk semantic confusion. Only use safe_negative when the pair is clearly broad, generic, cross-domain, or unrelated.

### EXAMPLES ###

Example 1 (Same coarse type, closely_related):
Input:
Anchor: {name: "027dtv3", attributes: "born in London, English actor"}
Positive: {name: "Jack Huston", attributes: "English actor, known for Boardwalk Empire"}
Candidate: {name: "Charlie Cox", attributes: "English actor, known for Daredevil"}
Output:
{
  "anchor_fine_type": "actor",
  "candidate_fine_type": "actor",
  "reasoning": "They are distinct actors but share the same profession, era, country, and entertainment-domain neighborhood, making this a high-risk same-type hard negative.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_profession_same_era",
  "confidence": 0.95
}

Example 2 (closely_related):
Input:
Anchor: {name: "04pzy", attributes: "fictional character, Daily Planet reporter"}
Positive: {name: "Lois Lane", attributes: "fictional character, romantic interest of Superman"}
Candidate: {name: "Superman", attributes: "fictional superhero, DC comics"}
Output:
{
  "anchor_fine_type": "fictional_character",
  "candidate_fine_type": "fictional_character",
  "reasoning": "Lois Lane and Superman are distinct characters but are fundamentally linked in the same fictional universe.",
  "relationship_status": "closely_related",
  "relationship_kind": "character_pair",
  "confidence": 0.98
}

Example 3 (closely_related):
Input:
Anchor: {name: "09pl3s", attributes: "writer, producer"}
Positive: {name: "Roberto Orci", attributes: "Mexican-American film and television screenwriter"}
Candidate: {name: "Alex Kurtzman", attributes: "American film director, producer, frequent collaborator of Roberto Orci"}
Output:
{
  "anchor_fine_type": "screenwriter",
  "candidate_fine_type": "producer",
  "reasoning": "They are famous long-term collaborators and their semantic neighborhoods strongly overlap.",
  "relationship_status": "closely_related",
  "relationship_kind": "frequent_collaborator",
  "confidence": 0.99
}

Example 4 (same_entity):
Input:
Anchor: {name: "02cvp8", attributes: "American boxer, born Cassius Clay"}
Positive: {name: "Muhammad Ali", attributes: "American professional boxer, activist"}
Candidate: {name: "Cassius Clay", attributes: "American boxer, olympic gold medalist"}
Output:
{
  "anchor_fine_type": "athlete",
  "candidate_fine_type": "athlete",
  "reasoning": "Cassius Clay is the birth name of Muhammad Ali, so they refer to the same person.",
  "relationship_status": "same_entity",
  "relationship_kind": "alias",
  "confidence": 0.95
}

Example 5 (unknown):
Input:
Anchor: {name: "03abcd", attributes: ""}
Positive: {name: "Unknown Person", attributes: "limited information"}
Candidate: {name: "Tom Hanks", attributes: "American actor"}
Output:
{
  "anchor_fine_type": "unknown",
  "candidate_fine_type": "actor",
  "reasoning": "The anchor identity is too weakly specified to determine whether the candidate is safely negative or closely related.",
  "relationship_status": "unknown",
  "relationship_kind": "insufficient_information",
  "confidence": 0.35
}

Example 5b (Place, closely_related):
Input:
Anchor: {name: "nm n", attributes: "county in Maine"}
Positive: {name: "York County Maine", attributes: "county in Maine"}
Candidate: {name: "Hancock County Maine", attributes: "county in Maine"}
Output:
{
  "anchor_fine_type": "county",
  "candidate_fine_type": "county",
  "reasoning": "They are distinct counties but sibling geographic nodes in the same state, so they share a local place topology that should be softly protected.",
  "relationship_status": "closely_related",
  "relationship_kind": "local_sibling_county",
  "confidence": 0.95
}

Example 6 (Creative Work, closely_related):
Input:
Anchor: {name: "03t95n", attributes: "fantasy action film, released in 2002"}
Positive: {name: "The Scorpion King", attributes: "fantasy action adventure film starring Dwayne Johnson"}
Candidate: {name: "The Mummy Returns", attributes: "fantasy adventure film, related franchise context"}
Output:
{
  "anchor_fine_type": "film",
  "candidate_fine_type": "film",
  "reasoning": "These are distinct films but they are tightly connected through franchise and narrative context.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_franchise",
  "confidence": 0.95
}

Example 7 (Place, closely_related):
Input:
Anchor: {name: "01zv", attributes: "autonomous community in Spain"}
Positive: {name: "Catalonia", attributes: "autonomous community of Spain"}
Candidate: {name: "Barcelona", attributes: "capital city of Catalonia"}
Output:
{
  "anchor_fine_type": "region",
  "candidate_fine_type": "city",
  "reasoning": "Catalonia and Barcelona are distinct places but are strongly linked by region-capital structure.",
  "relationship_status": "closely_related",
  "relationship_kind": "region_city",
  "confidence": 0.98
}

Example 8 (Creative Work, NOT same_entity):
Input:
Anchor: {name: "06ztvyx", attributes: "animated film released in 2011"}
Positive: {name: "Kung Fu Panda 2", attributes: "animated sequel film"}
Candidate: {name: "Kung Fu Panda", attributes: "original animated film released in 2008"}
Output:
{
  "anchor_fine_type": "film",
  "candidate_fine_type": "film",
  "reasoning": "Kung Fu Panda 2 and Kung Fu Panda belong to the same franchise, but one is the sequel and the other is the original film, so they are distinct works.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_franchise",
  "confidence": 0.95
}

Example 9 (Creative Work, NOT same_entity):
Input:
Anchor: {name: "03d16q3", attributes: "TV season released in 2002"}
Positive: {name: "The Wire season 1", attributes: "first season of The Wire"}
Candidate: {name: "The Wire season 3", attributes: "third season of The Wire"}
Output:
{
  "anchor_fine_type": "tv_series_season",
  "candidate_fine_type": "tv_series_season",
  "reasoning": "They are seasons of the same TV series, but different seasons are distinct works and therefore not the same entity.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_series",
  "confidence": 0.95
}

Example 10 (Disambiguation, safe_negative):
Input:
Anchor: {name: "John Smith", attributes: "American politician, mayor"}
Positive: {name: "John Smith (politician)", attributes: "Mayor of Anytown"}
Candidate: {name: "John Smith", attributes: "British Olympic swimmer"}
Output:
{
  "anchor_fine_type": "politician",
  "candidate_fine_type": "athlete",
  "reasoning": "They share the same name, but there is no direct structural dependency between an American politician and a British swimmer, so pushing them apart is necessary to resolve the name collision.",
  "relationship_status": "safe_negative",
  "relationship_kind": "name_collision",
  "confidence": 0.99
}

Example 11 (Person, closely_related):
Input:
Anchor: {name: "02k54", attributes: "44th President of the United States"}
Positive: {name: "Barack Obama", attributes: "American politician, 44th US President"}
Candidate: {name: "Joe Biden", attributes: "47th Vice President under Obama, 46th US President"}
Output:
{
  "anchor_fine_type": "politician",
  "candidate_fine_type": "politician",
  "reasoning": "They are distinct people but served as president and vice president in the same administration, so they share a narrow political context that should be softly protected.",
  "relationship_status": "closely_related",
  "relationship_kind": "president_vice_president",
  "confidence": 0.99
}

Example 12 (Person, closely_related):
Input:
Anchor: {name: "William Hanna", attributes: "American animator, director, producer"}
Positive: {name: "William Hanna", attributes: "Co-founder of Hanna-Barbera"}
Candidate: {name: "Joseph Barbera", attributes: "American animator, co-founder of Hanna-Barbera"}
Output:
{
  "anchor_fine_type": "animator",
  "candidate_fine_type": "animator",
  "reasoning": "They are long-term creative and business partners and co-founders of the same studio, making them a high-risk hard negative that should not be forcefully separated.",
  "relationship_status": "closely_related",
  "relationship_kind": "long_term_partners",
  "confidence": 0.99
}

Example 13 (Place, closely_related):
Input:
Anchor: {name: "Kent County Delaware", attributes: "county in Delaware"}
Positive: {name: "Kent County Delaware", attributes: "county in Delaware"}
Candidate: {name: "Sussex County Delaware", attributes: "county in Delaware"}
Output:
{
  "anchor_fine_type": "county",
  "candidate_fine_type": "county",
  "reasoning": "They are distinct counties but sibling local geographic nodes in the same small state, so forcefully pushing them apart may damage local place topology.",
  "relationship_status": "closely_related",
  "relationship_kind": "local_sibling_county",
  "confidence": 0.9
}

Example 14 (Person, closely_related):
Input:
Anchor: {name: "Thomas Jefferson", attributes: "American president and founding-era politician"}
Positive: {name: "Thomas Jefferson", attributes: "3rd President of the United States"}
Candidate: {name: "James Madison", attributes: "4th President of the United States"}
Output:
{
  "anchor_fine_type": "politician",
  "candidate_fine_type": "politician",
  "reasoning": "They are distinct politicians but belong to the same founding-era political circle and are high-risk semantic neighbors, so they should not be forcefully separated.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_political_circle",
  "confidence": 0.95
}

Example 15 (Person, closely_related):
Input:
Anchor: {name: "Lawrence Bender", attributes: "film producer"}
Positive: {name: "Lawrence Bender", attributes: "film producer"}
Candidate: {name: "Quentin Tarantino", attributes: "film director and screenwriter"}
Output:
{
  "anchor_fine_type": "film_producer",
  "candidate_fine_type": "film_director",
  "reasoning": "They are distinct people but are strongly associated through repeated producer-director collaboration in the same film circle.",
  "relationship_status": "closely_related",
  "relationship_kind": "producer_director_collaboration",
  "confidence": 0.95
}

Example 16 (Person, closely_related):
Input:
Anchor: {name: "Curly Howard", attributes: "American comedian and actor, member of The Three Stooges"}
Positive: {name: "Curly Howard", attributes: "American comedian and actor, member of The Three Stooges"}
Candidate: {name: "Shemp Howard", attributes: "American comedian and actor, brother of Curly Howard, member of The Three Stooges"}
Output:
{
  "anchor_fine_type": "actor",
  "candidate_fine_type": "actor",
  "reasoning": "They are distinct people but are brothers and members of the same famous comedy troupe, so this is a high-risk local semantic neighbor.",
  "relationship_status": "closely_related",
  "relationship_kind": "family_same_group",
  "confidence": 0.99
}

Example 17 (Organization, closely_related):
Input:
Anchor: {name: "Warner Bros Records", attributes: "American record label"}
Positive: {name: "Warner Bros Records", attributes: "American record label"}
Candidate: {name: "Reprise Records", attributes: "American record label associated with Warner Music"}
Output:
{
  "anchor_fine_type": "record_label",
  "candidate_fine_type": "record_label",
  "reasoning": "They are distinct record labels but belong to the same narrow music-label ecosystem, so forcefully separating them would damage the local organization neighborhood.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_label_family",
  "confidence": 0.95
}

Example 18 (Organization, closely_related):
Input:
Anchor: {name: "Blizzard Entertainment", attributes: "video game company"}
Positive: {name: "Blizzard Entertainment", attributes: "video game company"}
Candidate: {name: "Vivendi Games", attributes: "video game publisher formerly associated with Blizzard"}
Output:
{
  "anchor_fine_type": "company",
  "candidate_fine_type": "company",
  "reasoning": "They are distinct organizations but have a direct corporate history in the same gaming business network.",
  "relationship_status": "closely_related",
  "relationship_kind": "corporate_history",
  "confidence": 0.95
}

Example 19 (Organization, closely_related):
Input:
Anchor: {name: "Federalist Party", attributes: "early American political party"}
Positive: {name: "Federalist Party", attributes: "early American political party"}
Candidate: {name: "Democratic-Republican Party", attributes: "early American political party"}
Output:
{
  "anchor_fine_type": "political_party",
  "candidate_fine_type": "political_party",
  "reasoning": "They are distinct parties but are tightly coupled in the same early American political system and rivalry context.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_political_system",
  "confidence": 0.95
}

Example 16 (Person, closely_related):
Input:
Anchor: {name: "Thomas Jefferson", attributes: "American president and founding-era politician"}
Positive: {name: "Thomas Jefferson", attributes: "3rd President of the United States"}
Candidate: {name: "George Washington", attributes: "1st President of the United States"}
Output:
{
  "anchor_fine_type": "politician",
  "candidate_fine_type": "politician",
  "reasoning": "They are distinct founding-era politicians but belong to the same high-risk political circle, so they should be softly protected rather than forcefully separated.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_political_circle",
  "confidence": 0.95
}

Example 17 (Person, closely_related):
Input:
Anchor: {name: "John Locke", attributes: "English philosopher"}
Positive: {name: "John Locke", attributes: "English philosopher"}
Candidate: {name: "David Hume", attributes: "Scottish philosopher and economist"}
Output:
{
  "anchor_fine_type": "philosopher",
  "candidate_fine_type": "philosopher",
  "reasoning": "They are distinct philosophers in the same intellectual tradition and era, making them a high-risk semantic neighbor.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_profession_same_era",
  "confidence": 0.95
}

Example 18 (Person, closely_related):
Input:
Anchor: {name: "Barack Obama", attributes: "American politician, 44th US President"}
Positive: {name: "Barack Obama", attributes: "American politician, 44th US President"}
Candidate: {name: "Joe Biden", attributes: "American politician, 46th US President and former vice president"}
Output:
{
  "anchor_fine_type": "politician",
  "candidate_fine_type": "politician",
  "reasoning": "They are distinct politicians but share a narrow administration and political circle, so they should be softly protected.",
  "relationship_status": "closely_related",
  "relationship_kind": "president_vice_president",
  "confidence": 0.99
}

Example 19 (Organization, closely_related):
Input:
Anchor: {name: "Warner Bros Records", attributes: "American record label"}
Positive: {name: "Warner Bros Records", attributes: "American record label"}
Candidate: {name: "Reprise Records", attributes: "American record label associated with Warner Music"}
Output:
{
  "anchor_fine_type": "record_label",
  "candidate_fine_type": "record_label",
  "reasoning": "They are distinct record labels but belong to the same narrow label family and music business network.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_label_family",
  "confidence": 0.95
}

Example 20 (Organization, closely_related):
Input:
Anchor: {name: "Blizzard Entertainment", attributes: "video game company"}
Positive: {name: "Blizzard Entertainment", attributes: "video game company"}
Candidate: {name: "Vivendi Games", attributes: "video game publisher formerly associated with Blizzard"}
Output:
{
  "anchor_fine_type": "company",
  "candidate_fine_type": "company",
  "reasoning": "They are distinct organizations but have direct corporate history inside the same gaming ecosystem.",
  "relationship_status": "closely_related",
  "relationship_kind": "corporate_history",
  "confidence": 0.95
}

Example 21 (Organization, closely_related):
Input:
Anchor: {name: "Warner Bros Records", attributes: "American record label"}
Positive: {name: "Warner Bros Records", attributes: "American record label"}
Candidate: {name: "Elektra Records", attributes: "American record label in the same music business ecosystem"}
Output:
{
  "anchor_fine_type": "record_label",
  "candidate_fine_type": "record_label",
  "reasoning": "They are distinct labels in the same record-label ecosystem, so they are high-risk organization neighbors rather than safe negatives.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_label_ecosystem",
  "confidence": 0.95
}

Example 22 (Organization, closely_related):
Input:
Anchor: {name: "Reprise Records", attributes: "American record label"}
Positive: {name: "Reprise Records", attributes: "American record label"}
Candidate: {name: "Geffen Records", attributes: "American record label in the same broader music network"}
Output:
{
  "anchor_fine_type": "record_label",
  "candidate_fine_type": "record_label",
  "reasoning": "They are distinct record labels in the same music business network, so pushing them apart too hard would damage the local organization topology.",
  "relationship_status": "closely_related",
  "relationship_kind": "same_network",
  "confidence": 0.95
}

### CRITICAL INSTRUCTIONS ###
1. Step-by-step analysis: First identify shared attributes. Second, assess whether they share a narrow context, local sibling cluster, frequent collaboration, or strong direct tie. Third, conclude whether forcefully pushing them apart would damage useful local semantic topology.
2. Final decision target: `relationship_status` must describe Candidate Negative vs Anchor/Positive identity, not Anchor vs Positive.
3. In the JSON, fill `reasoning` first, then decide `relationship_status`.
4. JSON output: After your analysis, output EXACTLY ONE JSON object matching the required schema.
5. Do NOT wrap the JSON in markdown code blocks like ```json ... ```.
6. The final JSON must start with { and end with }.
7. Priority Rule: For hard negatives from the same coarse type, if there is any plausible shared context, same profession or era, local sibling topology, frequent collaboration, or plausible high-risk semantic confusion, PREFER `closely_related`. Use `safe_negative` only when the pair is clearly broad, generic, or unrelated.
8. Aggressive recall rule: If both entities have recognizable names and the same coarse type, label `closely_related` by default unless they are obviously cross-domain, pure name collisions, or clearly unrelated across time, location, and context.
9. If evidence is insufficient, output "unknown" rather than guessing.
}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM fine-type and safe-negative diagnosis on extracted HNM cases.")
    parser.add_argument("--input_jsonl", required=True, help="Path to LLM-ready HNM cases JSONL")
    parser.add_argument("--output_jsonl", required=True, help="Path to save per-case LLM outputs JSONL")
    parser.add_argument("--output_summary_json", required=True, help="Path to save aggregated summary")
    parser.add_argument("--model_path", required=True, help="Local Llama model path")
    parser.add_argument("--model_name", default="local_llama", help="Label recorded in summary")
    parser.add_argument("--max_cases", type=int, default=100)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--quality_filter", type=str, default="all", choices=["all", "high", "low"])
    parser.add_argument("--resume", action="store_true", default=False, help="Skip cases already present in output_jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


class LocalLlamaJudge:
    def __init__(self, model_path):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_path = model_path
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
        return self.tokenizer(
            f"{system_prompt}\n\n{user_content}",
            return_tensors="pt",
        ).input_ids

    def generate(self, system_prompt, user_content, max_new_tokens):
        input_ids = self.apply_chat_template(system_prompt, user_content).to(self.device)
        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(
            out_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return response


def load_cases(path, quality_filter):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if quality_filter != "all" and row.get("evidence_quality", "low") != quality_filter:
                continue
            cases.append(row)
    return cases


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
            key = (
                row.get("dataset"),
                row.get("cache_row"),
                row.get("direction"),
                row.get("candidate_rank", 1),
                row.get("negative_id"),
            )
            done.add(key)
    return done


def attrs_to_text(kv_list):
    if not kv_list:
        return "[EMPTY]"
    return "\n".join(kv_list)


def build_user_prompt(case):
    return (
        f"Dataset: {case.get('dataset', '')}\n"
        f"Direction: {case.get('direction', '')}\n"
        f"Coarse Type: {case.get('coarse_type', '')}\n"
        f"Top-1 Similarity: {case.get('top1_sim', '')}\n\n"
        f"Candidate Rank in same-type hard pool: {case.get('candidate_rank', '')}\n"
        f"Candidate Similarity: {case.get('candidate_sim', case.get('top1_sim', ''))}\n\n"
        f"Anchor:\n"
        f"- name_display: {case.get('anchor_name_display', case.get('anchor_name', ''))}\n"
        f"- name_raw: {case.get('anchor_name_raw', '')}\n"
        f"- attributes:\n{attrs_to_text(case.get('anchor_attr_kv', []))}\n\n"
        f"Positive (true aligned identity of Anchor):\n"
        f"- name_display: {case.get('positive_name_display', case.get('positive_name', ''))}\n"
        f"- name_raw: {case.get('positive_name_raw', '')}\n"
        f"- attributes:\n{attrs_to_text(case.get('positive_attr_kv', []))}\n\n"
        f"Candidate Negative:\n"
        f"- name_display: {case.get('negative_name_display', case.get('negative_name', ''))}\n"
        f"- name_raw: {case.get('negative_name_raw', '')}\n"
        f"- attributes:\n{attrs_to_text(case.get('negative_attr_kv', []))}\n\n"
        "### CRITICAL INSTRUCTION FOR THIS CASE ###\n"
        f"{case.get('llm_instruction', '')}\n\n"
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
    return {}


def normalize_output(parsed):
    status = str(parsed.get("relationship_status", "unknown")).strip()
    if status not in {"same_entity", "closely_related", "safe_negative", "unknown"}:
        status = "unknown"
    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "anchor_fine_type": str(parsed.get("anchor_fine_type", "unknown")).strip() or "unknown",
        "candidate_fine_type": str(parsed.get("candidate_fine_type", "unknown")).strip() or "unknown",
        "relationship_status": status,
        "relationship_kind": str(parsed.get("relationship_kind", "unknown")).strip() or "unknown",
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def call_llm_case(judge, args, case):
    content = judge.generate(
        system_prompt=SYSTEM_PROMPT,
        user_content=build_user_prompt(case),
        max_new_tokens=args.max_new_tokens,
    )
    parsed = extract_json(content)
    normalized = normalize_output(parsed)
    reasoning_trace = content
    return normalized, content, reasoning_trace


def build_summary(records, model_name):
    status_counter = Counter(r["llm_relationship_status"] for r in records)
    relation_counter = Counter(r["llm_relationship_kind"] for r in records)
    anchor_type_counter = Counter(r["llm_anchor_fine_type"] for r in records)
    candidate_type_counter = Counter(r["llm_candidate_fine_type"] for r in records)
    route_counter = defaultdict(Counter)
    for r in records:
        route_counter[r["coarse_type"]][r["llm_relationship_status"]] += 1

    return {
        "teacher_model": model_name,
        "case_count": len(records),
        "relationship_status_counts": dict(status_counter),
        "relationship_kind_counts": dict(relation_counter),
        "anchor_fine_type_counts": dict(anchor_type_counter),
        "candidate_fine_type_counts": dict(candidate_type_counter),
        "coarse_type_breakdown": {k: dict(v) for k, v in route_counter.items()},
    }


def main():
    args = parse_args()
    judge = LocalLlamaJudge(args.model_path)
    cases = load_cases(args.input_jsonl, args.quality_filter)
    if args.resume:
        done = load_done_keys(args.output_jsonl)
        cases = [
            c for c in cases
            if (
                c.get("dataset"),
                c.get("cache_row"),
                c.get("direction"),
                c.get("candidate_rank", 1),
                c.get("negative_id"),
            ) not in done
        ]
    cases = cases[: args.max_cases]

    print(f"Running LLM fine-type diagnosis on {len(cases)} cases...", flush=True)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    records = []
    if args.resume:
        open(args.output_jsonl, "a", encoding="utf-8").close()
    else:
        with open(args.output_jsonl, "w", encoding="utf-8"):
            pass
    run_start = time.time()
    if args.resume and os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    initial_record_count = len(records)
    error_count = 0

    for idx, case in enumerate(cases, start=1):
        step_start = time.time()
        try:
            result, raw_response, reasoning_trace = call_llm_case(judge, args, case)
            row = dict(case)
            row.update(
                {
                    "llm_anchor_fine_type": result["anchor_fine_type"],
                    "llm_candidate_fine_type": result["candidate_fine_type"],
                    "llm_relationship_status": result["relationship_status"],
                    "llm_relationship_kind": result["relationship_kind"],
                    "llm_confidence": result["confidence"],
                    "llm_reasoning": result["reasoning"],
                    "llm_raw_response": raw_response,
                    "llm_reasoning_trace": reasoning_trace,
                    "llm_error": "",
                }
            )
        except Exception as exc:
            error_count += 1
            print(f"[LLM Error] cache_row={case.get('cache_row')} error={type(exc).__name__}: {exc}", flush=True)
            continue

        records.append(row)
        with open(args.output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        step_elapsed = time.time() - step_start
        elapsed = time.time() - run_start
        avg_step = elapsed / idx if idx > 0 else 0.0
        remaining = len(cases) - idx
        eta = remaining * avg_step
        print(
            f"[{idx}/{len(cases)}] cache_row={case.get('cache_row')} "
            f"status={row['llm_relationship_status']} "
            f"kind={row['llm_relationship_kind']} "
            f"conf={row['llm_confidence']:.2f} "
            f"step={step_elapsed:.1f}s "
            f"elapsed={elapsed:.1f}s "
            f"eta={eta:.1f}s",
            flush=True,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary = build_summary(records, args.model_name)
    summary["input_case_count"] = len(cases)
    summary["new_record_count"] = len(records) - initial_record_count
    summary["llm_error_count"] = error_count
    os.makedirs(os.path.dirname(args.output_summary_json), exist_ok=True)
    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    total_elapsed = time.time() - run_start
    if len(cases) > 0 and summary["new_record_count"] == 0:
        raise RuntimeError(
            f"No new label rows were written for non-empty input: input_case_count={len(cases)} error_count={error_count}"
        )
    print(f"Total elapsed: {total_elapsed:.1f}s", flush=True)
    print(f"Summary written to {args.output_summary_json}", flush=True)


if __name__ == "__main__":
    main()
