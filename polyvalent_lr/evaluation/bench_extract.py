"""One deterministic answer extractor for every benchmark.

Priority:
  1. the last non-empty ``<answer>...</answer>`` block
  2. an ``answer:`` / ``final answer:`` line
  3. the last non-empty line of the generation

Within the extracted span, the *last valid label token* wins (the benchmark
label vocabulary is supplied per item).  Repeated-tag spam is stripped before
matching.  Semantic cross-space mapping (True->entailment, etc.) is opt-in and
reported separately; the default is strict in-space matching.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_LINE_RE = re.compile(r"(?:final\s+)?answer\s*[:：]\s*(.+)", re.IGNORECASE)
_TAG_RE = re.compile(r"</?(?:answer|reasoning|think)\b[^>]*>", re.IGNORECASE)
_SPECIAL_RE = re.compile(r"<\|im_(?:start|end)\|>")
_NONWORD_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}

NLI_SEMANTIC_ALIASES = {
    "true": "entailment",
    "entails": "entailment",
    "entailed": "entailment",
    "false": "contradiction",
    "contradicts": "contradiction",
    "contradicted": "contradiction",
    "uncertain": "neutral",
    "unknown": "neutral",
    "undetermined": "neutral",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = _NONWORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _clean_span(text: str) -> str:
    text = _SPECIAL_RE.sub(" ", text or "")
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.strip(" \t\r\n\"'`.,:;!?-")


def extract_answer_span(text: str) -> Tuple[str, str]:
    """Return ``(span, method)`` from the raw generation."""
    raw = str(text or "")
    matches = [m.group(1) for m in _ANSWER_RE.finditer(raw)]
    for candidate in reversed(matches):
        cleaned = _clean_span(candidate)
        if cleaned:
            return cleaned, "answer_tag"

    for m in reversed(list(_ANSWER_LINE_RE.finditer(raw))):
        cleaned = _clean_span(m.group(1))
        if cleaned:
            return cleaned, "answer_line"

    cleaned = _clean_span(raw)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if lines:
        return lines[-1], "last_line"
    return "", "empty"


def _lookup_table(
    label_space: Sequence[str],
    aliases: Optional[Dict[str, str]] = None,
    choices: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    table: Dict[str, str] = {}
    for label in label_space:
        table[normalize(label)] = label
        table[normalize(str(label).replace("_", " "))] = label
    for alias, label in (aliases or {}).items():
        if label in label_space:
            table[normalize(alias)] = label
            table[normalize(str(alias))] = label
    if choices:
        for key, value in choices.items():
            if key not in label_space:
                continue
            table[normalize(key)] = key
            table[normalize(str(value))] = key
    table.update({k: v for k, v in _NUMBER_WORDS.items() if v in label_space})
    return {k: v for k, v in table.items() if k}


def _token_candidates(span: str, max_len: int = 4) -> List[str]:
    toks = normalize(span).split()
    out: List[str] = []
    for n in range(1, min(max_len, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            out.append(" ".join(toks[i:i + n]))
    return out


def canonicalize(
    span: str,
    label_space: Sequence[str],
    aliases: Optional[Dict[str, str]] = None,
    choices: Optional[Dict[str, str]] = None,
    allow_semantic_nli: bool = False,
) -> Tuple[Optional[str], str]:
    """Return ``(canonical_label_or_None, reason)``.

    ``reason`` is one of ``exact``, ``token``, ``option_text``, ``semantic``,
    ``empty``, ``no_label``.
    """
    if not span:
        return None, "empty"
    norm = normalize(span)
    table = _lookup_table(label_space, aliases, choices)

    if norm in table:
        return table[norm], "exact"

    # Longest option text first, so a full sentence wins over a key fragment.
    if choices:
        options = sorted(
            ((normalize(v), k) for k, v in choices.items() if k in label_space),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        for opt_norm, key in options:
            if opt_norm and opt_norm in norm:
                return key, "option_text"

    # Rightmost valid label token (the "last valid label token" rule).
    cands = _token_candidates(span)
    for cand in reversed(cands):
        if cand in table:
            return table[cand], "token"

    if allow_semantic_nli and set(label_space) == {
        "entailment", "contradiction", "neutral", "self_contradiction"
    }:
        for cand in reversed(cands):
            if cand in NLI_SEMANTIC_ALIASES:
                return NLI_SEMANTIC_ALIASES[cand], "semantic"

    return None, "no_label"


def parse_prediction(
    text: str,
    item: Dict[str, Any],
    allow_semantic_nli: bool = False,
) -> Dict[str, Any]:
    """Full extraction record for one generation/item pair."""
    span, method = extract_answer_span(text)
    label, reason = canonicalize(
        span,
        item.get("label_space") or [],
        item.get("aliases"),
        item.get("choices"),
        allow_semantic_nli=allow_semantic_nli,
    )
    return {
        "answer_span": span,
        "answer_span_method": method,
        "normalized_span": normalize(span),
        "label": label,
        "match_reason": reason,
        "malformed": label is None,
    }
