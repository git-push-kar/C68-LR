"""Deterministic stratified sampling for benchmark runs.

Rule used by run_bench.py:
  * n <= 1000 -> evaluate everything
  * n > 1000  -> stratified sample of at least 500 (default 600)

Allocation is proportional to stratum size with at least one item per stratum
when possible.  The selected ids are saved by the runner.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


def _stable_seed(seed: int, key: str) -> int:
    h = hashlib.sha1(f"{seed}\x00{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def stratified_sample(
    items: Sequence[Dict[str, Any]],
    n: int,
    key_fn: Callable[[Dict[str, Any]], str],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample ``n`` items proportional to strata defined by ``key_fn``."""
    items = list(items)
    if n >= len(items):
        return sorted(items, key=lambda x: str(x.get("id", "")))

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(key_fn(item))].append(item)
    strata = sorted(groups)
    m = len(strata)
    total = len(items)
    if n < m:
        # Not enough budget for one per stratum; take the largest strata.
        chosen = sorted(strata, key=lambda s: (-len(groups[s]), s))[:n]
        return sorted(
            [random.Random(_stable_seed(seed, s)).choice(groups[s]) for s in chosen],
            key=lambda x: str(x.get("id", "")),
        )

    target = {s: n * len(groups[s]) / total for s in strata}
    alloc = {s: min(len(groups[s]), max(1, int(target[s]))) for s in strata}

    # Largest-remainder adjustment to hit n exactly.
    while sum(alloc.values()) < n:
        eligible = [s for s in strata if alloc[s] < len(groups[s])]
        if not eligible:
            break
        s = max(eligible, key=lambda x: (target[x] - alloc[x], len(groups[x]), x))
        alloc[s] += 1
    while sum(alloc.values()) > n:
        eligible = [s for s in strata if alloc[s] > 1]
        if not eligible:
            break
        s = min(eligible, key=lambda x: (target[x] - alloc[x], -len(groups[x]), x))
        alloc[s] -= 1

    out: List[Dict[str, Any]] = []
    for s in strata:
        pool = sorted(groups[s], key=lambda x: str(x.get("id", "")))
        rng = random.Random(_stable_seed(seed, s))
        rng.shuffle(pool)
        out.extend(pool[: alloc[s]])
    return sorted(out, key=lambda x: str(x.get("id", "")))


def sample_plan(
    groups: Dict[str, List[Dict[str, Any]]],
    n_over_1000: int = 600,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Apply the benchmark sampling rule to every group."""
    plan: Dict[str, List[Dict[str, Any]]] = {}
    for name, items in groups.items():
        if name == "LogicBench-BQA":
            key = lambda x: f"{x.get('pattern')}|{x.get('gold')}"
        elif name == "Multi-LogiEval":
            key = lambda x: (
                f"{x.get('meta', {}).get('folder')}|{x.get('domain')}|{x.get('pattern')}"
            )
        elif name == "LogicNLI":
            key = lambda x: str(x.get("gold"))
        elif name == "v3-Holdout":
            key = lambda x: str(x.get("split"))
        else:
            key = lambda x: str(x.get("pattern"))
        n = len(items) if len(items) <= 1000 else max(500, n_over_1000)
        plan[name] = stratified_sample(items, n, key, seed=seed)
    return plan
