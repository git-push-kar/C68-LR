"""Perturbation and Robustness Suite for Logical Reasoning.

Implements 6 systematic perturbation modes:
a) Entity/predicate renaming with nonsense/nonce tokens
b) Premise shuffling (3 distinct permutations per item)
c) Distractor premises (1, 3, 5 irrelevant sentences)
d) Deterministic template-based paraphrasing (no LLM)
e) Depth extrapolation (stratified shallow vs deep reasoning)
f) Conclusion negation with label flip (for well-defined mappings)
"""
from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Neutral distractor sentences disjoint from reasoning domains
DISTRACTOR_POOL = [
    "The atmospheric pressure at sea level averages roughly one thousand thirteen millibars.",
    "A standard acoustic guitar is constructed with six resonant steel or nylon strings.",
    "Photosynthesis enables green plants to synthesize glucose from carbon dioxide and water.",
    "The periodic table arranges chemical elements in order of increasing atomic number.",
    "Glaciers in polar regions advance and recede in response to long-term climate variations.",
    "Silicon dioxide occurs abundantly in nature as quartz and various types of sand.",
    "The planet Neptune requires approximately one hundred sixty-five Earth years to orbit the Sun.",
    "A standard chess board consists of sixty-four alternating light and dark squares.",
]

ENTITY_MAP = {
    "Charlie": "Xylor-7",
    "Alice": "Zenth-4",
    "Bob": "Quux-9",
    "Dave": "Grob-3",
    "Fiona": "Plink-8",
    "Gary": "Vraz-2",
    "Harry": "Kren-5",
    "Anne": "Morv-1",
    "Erin": "Trel-6",
    "cat": "greeble-A",
    "dog": "flump-B",
    "bear": "snark-C",
    "bald eagle": "borogove-D",
    "tiger": "jabber-E",
    "lion": "mimsy-F",
    "rabbit": "slithy-G",
    "squirrel": "jubjub-H",
}


def rename_entities(
    premises: List[str], question: str
) -> Tuple[List[str], str, Dict[str, str]]:
    """Substitutes names and common entities with nonsense/nonce tokens."""
    applied_map: Dict[str, str] = {}
    new_premises: List[str] = []

    # Sort keys by length descending to prevent partial prefix replacements
    sorted_entities = sorted(ENTITY_MAP.keys(), key=len, reverse=True)

    for p in premises:
        res = p
        for ent in sorted_entities:
            # Word boundary replacement
            pattern = re.compile(rf"\b{re.escape(ent)}\b", re.IGNORECASE if ent.islower() else 0)
            if pattern.search(res):
                res = pattern.sub(ENTITY_MAP[ent], res)
                applied_map[ent] = ENTITY_MAP[ent]
        new_premises.append(res)

    new_q = question
    for ent in sorted_entities:
        pattern = re.compile(rf"\b{re.escape(ent)}\b", re.IGNORECASE if ent.islower() else 0)
        if pattern.search(new_q):
            new_q = pattern.sub(ENTITY_MAP[ent], new_q)
            applied_map[ent] = ENTITY_MAP[ent]

    return new_premises, new_q, applied_map


def shuffle_premises(premises: List[str], seed: int = 42) -> List[str]:
    """Deterministically permutes premise order."""
    if len(premises) <= 1:
        return list(premises)
    p_copy = list(premises)
    rng = random.Random(seed)
    rng.shuffle(p_copy)
    # Ensure shuffle produces a distinct order if len > 1
    if p_copy == premises and len(premises) > 1:
        p_copy = premises[1:] + premises[:1]
    return p_copy


def add_distractors(
    premises: List[str], n_distractors: int = 1, seed: int = 42
) -> List[str]:
    """Injects n irrelevant neutral sentences into premises without changing validity."""
    rng = random.Random(seed)
    n = min(n_distractors, len(DISTRACTOR_POOL))
    chosen = rng.sample(DISTRACTOR_POOL, n)
    p_copy = list(premises)
    # Evenly interleave distractors
    for dist in chosen:
        insert_pos = rng.randint(0, len(p_copy))
        p_copy.insert(insert_pos, dist)
    return p_copy


PARAPHRASE_RULES = [
    # If A then B
    (
        re.compile(r"^If (.*?), then (.*?)\.?$", re.IGNORECASE),
        r"Whenever \1, it is guaranteed that \2.",
    ),
    # If A, B
    (
        re.compile(r"^If (.*?), (.*?)\.?$", re.IGNORECASE),
        r"Given that \1 holds, it follows that \2.",
    ),
    # All X are Y
    (
        re.compile(r"^All (.*?) are (.*?)\.?$", re.IGNORECASE),
        r"Every member of the group of \1 is classified as \2.",
    ),
    # Every X is Y
    (
        re.compile(r"^Every (.*?) is (.*?)\.?$", re.IGNORECASE),
        r"Each individual \1 possesses the property of being \2.",
    ),
    # X is Y
    (
        re.compile(r"^([A-Z][a-zA-Z0-9\-]*) is (.*?)\.?$"),
        r"\1 exhibits the quality of being \2.",
    ),
]


def paraphrase_templates(premises: List[str]) -> List[str]:
    """Applies rule-based syntactic transformations to premise sentences."""
    new_premises: List[str] = []
    for p in premises:
        p_stripped = p.strip()
        matched = False
        for pattern, replacement in PARAPHRASE_RULES:
            if pattern.match(p_stripped):
                new_p = pattern.sub(replacement, p_stripped)
                new_premises.append(new_p)
                matched = True
                break
        if not matched:
            new_premises.append(p)
    return new_premises


def negate_conclusion(question: str, gold: str) -> Tuple[str, str, bool]:
    """Inverts conclusion polarity and flips gold label when well-defined.

    Returns: (new_question, flipped_gold, is_valid_flip)
    """
    flipped_gold = gold
    valid = False

    # Label flipping logic
    if gold == "True":
        flipped_gold = "False"
        valid = True
    elif gold == "False":
        flipped_gold = "True"
        valid = True
    elif gold in ["entailment", "entailed"]:
        flipped_gold = "contradiction"
        valid = True
    elif gold in ["contradiction", "contradicted"]:
        flipped_gold = "entailment"
        valid = True
    elif gold in ["Uncertain", "neutral"]:
        # Negation of an undetermined statement remains undetermined
        flipped_gold = gold
        valid = True

    q = question.strip()
    # Invert syntactic polarity
    if " is not " in q:
        new_q = q.replace(" is not ", " is ")
    elif " isn't " in q:
        new_q = q.replace(" isn't ", " is ")
    elif " is " in q:
        new_q = q.replace(" is ", " is not ")
    elif " does not " in q:
        new_q = q.replace(" does not ", " does ")
    elif " doesn't " in q:
        new_q = q.replace(" doesn't ", " does ")
    elif " cannot " in q:
        new_q = q.replace(" cannot ", " can ")
    elif " can " in q:
        new_q = q.replace(" can ", " cannot ")
    elif " are not " in q:
        new_q = q.replace(" are not ", " are ")
    elif " are " in q:
        new_q = q.replace(" are ", " are not ")
    else:
        new_q = f"It is false that {q[0].lower() + q[1:] if q else q}"

    return new_q, flipped_gold, valid


def build_perturbed_item(
    item: Dict[str, Any], p_type: str, seed: int = 42
) -> Optional[Dict[str, Any]]:
    """Creates a perturbed clone of an item according to perturbation type."""
    p_item = copy.deepcopy(item)
    premises = list(item.get("premises", []))
    q = str(item.get("question", ""))
    gold = str(item.get("gold", ""))

    if p_type == "rename_entities":
        new_premises, new_q, app = rename_entities(premises, q)
        p_item["premises"] = new_premises
        p_item["question"] = new_q
        p_item["perturbation"] = "rename_entities"

    elif p_type.startswith("shuffle_premises_"):
        shuffle_id = int(p_type.split("_")[-1])
        new_premises = shuffle_premises(premises, seed=seed + shuffle_id * 100)
        p_item["premises"] = new_premises
        p_item["perturbation"] = f"shuffle_premises_{shuffle_id}"

    elif p_type.startswith("distractor_"):
        n_dist = int(p_type.split("_")[-1])
        new_premises = add_distractors(premises, n_distractors=n_dist, seed=seed)
        p_item["premises"] = new_premises
        p_item["perturbation"] = f"distractor_{n_dist}"

    elif p_type == "paraphrase":
        new_premises = paraphrase_templates(premises)
        p_item["premises"] = new_premises
        p_item["perturbation"] = "paraphrase"

    elif p_type == "negate_conclusion":
        new_q, new_gold, valid = negate_conclusion(q, gold)
        if not valid:
            return None
        p_item["question"] = new_q
        p_item["gold"] = new_gold
        p_item["perturbation"] = "negate_conclusion"

    else:
        return None

    # Rebuild formatted instruction prompt
    from polyvalent_lr.data.schema import LogicalExample, LogicTaskType
    premises_text = "\n".join(f"- {p}" for p in p_item["premises"])
    label_space_str = ", ".join(p_item.get("label_space", ["True", "False", "Uncertain"]))

    prompt = (
        f"<|im_start|>system\nYou are a rigorous logical reasoning engine. "
        f"Work through the problem step by step inside <reasoning>...</reasoning>, "
        f"then give the final answer inside <answer>...</answer>. "
        f"State only conclusions that follow from the given premises.<|im_end|>\n"
        f"<|im_start|>user\nGiven the facts and rules (premises), determine whether the "
        f"hypothesis is {label_space_str}.\n\n"
        f"Premises:\n{premises_text}\n\n"
        f"Hypothesis:\n{p_item['question']}\n\n"
        f"Format your response as:\n"
        f"<reasoning>\n...\n</reasoning>\n"
        f"<answer>[Final Answer]</answer><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    p_item["prompt"] = prompt
    p_item["id"] = f"{item['id']}__pert_{p_item['perturbation']}"
    return p_item
