"""Stage-4 inference boosters: extraction + self-consistency (majority vote)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+?)\|\s*$")


def extract_final_answer(text: str) -> str:
    """Return the content of the last <answer> block, else the whole text."""
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else (text or "").strip()


def _cells(line: str) -> Optional[Tuple[str, ...]]:
    m = _TABLE_ROW_RE.match(line)
    if not m:
        return None
    cells = tuple(c.strip() for c in m.group(1).split("|"))
    if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
        return None  # separator row
    return cells


def table_signature(text: str) -> Optional[frozenset]:
    """Canonical signature of a 4-column layout table (desk,name,domain,os)."""
    rows = []
    for line in (text or "").splitlines():
        cells = _cells(line)
        if cells and len(cells) == 4:
            desk = re.sub(r"[^0-9]", "", cells[0])
            if not desk:
                continue
            rows.append((int(desk), cells[1].strip().lower(), cells[2].strip().lower(), cells[3].strip().lower()))
    if not rows or len({r[0] for r in rows}) < 2:
        return None
    return frozenset(sorted(rows))


def majority_vote(
    samples: Sequence[str],
    key_fn: Callable[[str], Optional[frozenset]] = table_signature,
) -> str:
    """Pick the sample whose final-table signature is most common (ties -> first)."""
    if not samples:
        return ""
    sigs = [key_fn(s) for s in samples]
    counts = Counter(s for s in sigs if s is not None)
    if not counts:
        return samples[0]
    best_sig, _ = counts.most_common(1)[0]
    for s, sig in zip(samples, sigs):
        if sig == best_sig:
            return s
    return samples[0]


def build_generation_kwargs(
    tokenizer,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 768,
    stop_strings: Optional[List[str]] = None,
) -> dict:
    """Shared kwargs for ``model.generate`` in the UIs (Stage 4 boosters)."""
    kwargs = {
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else None,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if stop_strings:
        kwargs["stop_strings"] = list(stop_strings)
        kwargs["tokenizer"] = tokenizer
    return {k: v for k, v in kwargs.items() if v is not None}
