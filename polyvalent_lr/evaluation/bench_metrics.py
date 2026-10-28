"""Metrics and bootstrap helpers for the benchmark report."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")
_DESK_RE = re.compile(r"\d+")


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 20261028,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means: List[np.ndarray] = []
    done = 0
    chunk = 1000
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, arr.size, size=(m, arr.size))
        means.append(arr[idx].mean(axis=1))
        done += m
    all_means = np.concatenate(means)
    lo, hi = np.percentile(all_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 20261028,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.size == 0 or a.size != b.size:
        return (float("nan"), float("nan"))
    d = a - b
    return bootstrap_mean_ci(d, n_boot=n_boot, seed=seed, alpha=alpha)


def accuracy(correct: Sequence[int]) -> float:
    return float(sum(correct) / len(correct)) if correct else float("nan")


def token_f1(pred: str, gold: str) -> float:
    p = _WORD_RE.findall((pred or "").lower())
    g = _WORD_RE.findall((gold or "").lower())
    if not p or not g:
        return 0.0
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def parse_markdown_table(text: str) -> Dict[int, Tuple[str, str, str]]:
    """Parse rows of the form ``| desk | name | domain | os |``."""
    rows: Dict[int, Tuple[str, str, str]] = {}
    for line in (text or "").splitlines():
        s = line.strip()
        if not s.startswith("|") or s.count("|") < 4:
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 4:
            continue
        if cells[0].lower().startswith("desk"):
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        m = _DESK_RE.search(cells[0])
        if not m:
            continue
        rows[int(m.group())] = (cells[1], cells[2], cells[3])
    return rows


def per_cell_accuracy(
    pred_rows: Dict[int, Tuple[str, str, str]],
    gold: Dict[int, Dict[str, str]],
) -> Dict[str, Tuple[int, int]]:
    """Return ``{person: (correct,total), domain: ..., os: ...}``."""
    out = {"person": [0, 0], "domain": [0, 0], "os": [0, 0]}
    idx = {"person": 0, "domain": 1, "os": 2}
    for desk, cells in gold.items():
        pred = pred_rows.get(int(desk))
        for cat, j in idx.items():
            out[cat][1] += 1
            if pred is not None and pred[j].strip().lower() == cells[cat].strip().lower():
                out[cat][0] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def confusion_matrix(
    pairs: Iterable[Tuple[str, str]],
    labels: Sequence[str],
    malformed: str = "<malformed>",
) -> Dict[str, Dict[str, int]]:
    cols = list(labels) + [malformed]
    mat = {g: {p: 0 for p in cols} for g in labels}
    for gold, pred in pairs:
        g = gold if gold in mat else malformed
        p = pred if pred in cols else malformed
        mat.setdefault(g, {x: 0 for x in cols})
        mat[g][p] += 1
    return mat


def fmt_ci(acc: float, ci: Tuple[float, float]) -> str:
    if math.isnan(acc):
        return "n/a"
    return f"{acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]"


def fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"
