import re
from typing import Dict, Any, List, Tuple
from polyvalent_lr.data.schema import LogicalExample
from polyvalent_lr.config import LogicTaskType


# Symbolic "soup" markers produced by ProofWriter/AbductionRules raw representations,
# e.g. "(tripleM)", "rule7", "((triple2) -> rule4)". These teach the model to emit
# meaningless symbol tokens instead of natural-language reasoning.
_SYMBOLIC_PATTERNS = [
    re.compile(r"\btriple(?:M|\d+)\b", re.IGNORECASE),
    re.compile(r"\brule[_]?\d+\b", re.IGNORECASE),
    re.compile(r"\bintermediate\d+\b", re.IGNORECASE),
    re.compile(r"\bproof_?\d+\b", re.IGNORECASE),
]


def is_symbolic_text(text: Any) -> bool:
    """Returns True if the text looks like raw symbolic proof soup rather than natural language."""
    if text is None:
        return False
    t = str(text).strip()
    if not t:
        return False
    if any(p.search(t) for p in _SYMBOLIC_PATTERNS):
        return True
    # Parenthesis/operator soup combined with symbolic keywords or excessive nesting.
    if len(re.findall(r"[()]", t)) >= 6 and re.search(r"->|rule|triple", t, re.IGNORECASE):
        return True
    if t.count("(") >= 3 and t.count(")") >= 3 and re.search(r"->", t):
        return True
    return False


def clean_premises(premises: List[str]) -> List[str]:
    """Drops premises that are raw symbolic soup, preserving order of the rest."""
    cleaned = []
    for p in premises or []:
        p_str = str(p).strip()
        if p_str and not is_symbolic_text(p_str):
            cleaned.append(p_str)
    return cleaned


def _fingerprint(ex: LogicalExample) -> Tuple[str, str]:
    prompt = re.sub(r"\s+", " ", ex.to_instruction_prompt()).strip().lower()
    target = re.sub(r"\s+", " ", ex.to_target_text()).strip().lower()
    return (prompt, target)


def sanitize_examples(
    examples: List[LogicalExample],
    drop_symbolic_targets: bool = True,
    dedup: bool = True
) -> Tuple[List[LogicalExample], Dict[str, int]]:
    """Removes symbolic-soup targets/premises and duplicate (prompt, target) pairs.

    Returns the cleaned examples and a corruption report.
    """
    clean: List[LogicalExample] = []
    seen = set()
    report = {
        "input": len(examples),
        "dropped_symbolic_target": 0,
        "dropped_symbolic_premises_only": 0,
        "dropped_empty": 0,
        "dropped_duplicate": 0,
        "kept": 0,
    }

    for ex in examples:
        target_text = (ex.proof_trace_text or "").strip()
        if drop_symbolic_targets and is_symbolic_text(target_text):
            report["dropped_symbolic_target"] += 1
            continue

        cleaned_premises = clean_premises(ex.premises)
        if len(cleaned_premises) < len(ex.premises):
            report["dropped_symbolic_premises_only"] += 1

        if ex.task_type != LogicTaskType.GENERAL and not cleaned_premises and not ex.hypothesis_or_query.strip():
            report["dropped_empty"] += 1
            continue

        ex.premises = cleaned_premises

        if dedup:
            fp = _fingerprint(ex)
            if fp in seen:
                report["dropped_duplicate"] += 1
                continue
            seen.add(fp)

        clean.append(ex)

    report["kept"] = len(clean)
    return clean, report


def sanitize_report_string(name: str, report: Dict[str, int]) -> str:
    return (
        f"[Sanitize] {name}: input={report['input']:,} kept={report['kept']:,} "
        f"| dropped_symbolic_target={report['dropped_symbolic_target']:,} "
        f"| symbolic_premises_trimmed={report['dropped_symbolic_premises_only']:,} "
        f"| dropped_empty={report['dropped_empty']:,} "
        f"| dropped_duplicate={report['dropped_duplicate']:,}"
    )
