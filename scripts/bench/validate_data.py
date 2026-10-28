"""Validate the canonical benchmark items before any GPU run.

Hard gates (per benchmark/test split):
  * at least two distinct gold labels
  * no single gold label exceeds 70% of the split
  * 100% unique item ids
  * >=95% unique full prompts

The gates are deliberately strict.  A split that fails is *not* silently
repaired or re-sampled; it is reported and must be excluded from the headline
accuracy table (see report.py).  Multi-LogiEval is expected to fail the 70%
label-balance gate because its official release is 71.5% yes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENCH_ROOT = ROOT / "data" / "benchmarks_real"
HEADLINE = ["LogicBench-BQA", "LogicBench-MCQA", "Multi-LogiEval", "LogicNLI"]
MAX_LABEL_SHARE = 0.70
MIN_UNIQUE_PROMPTS = 0.95


def load_items() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in sorted(BENCH_ROOT.glob("*/items.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def _summary(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    gold = Counter(str(r.get("gold", "")) for r in rows if str(r.get("gold", "")).strip())
    ids = Counter(str(r.get("id", "")) for r in rows)
    prompts = Counter(str(r.get("prompt", "")) for r in rows)
    max_label = max(gold.values()) / n if n and gold else 0.0
    majority = max(gold, key=gold.get) if gold else None
    # Chance is reported per task type: 1/2 for binary, 1/4 for NLI, mean
    # 1/n_choices for MCQA, and undefined for the mixed holdout.
    task_types = Counter(r.get("task_type") for r in rows)
    item_chances = [
        float(r["chance"]) for r in rows if r.get("chance") is not None
    ]
    chance = sum(item_chances) / len(item_chances) if item_chances else None
    if chance is None:
        if task_types == Counter({"binary": n}):
            chance = 0.5
        elif task_types == Counter({"nli4": n}):
            chance = 0.25
        elif task_types == Counter({"mcqa": n}):
            chance = sum(1.0 / max(1, len(r.get("label_space") or [])) for r in rows) / n
    return {
        "benchmark": name,
        "n": n,
        "labels": dict(gold.most_common()),
        "n_labels": len(gold),
        "max_label_share": round(max_label, 4),
        "majority_label": majority,
        "majority_baseline": round(max_label, 4),
        "chance": round(chance, 4) if chance is not None else None,
        "unique_ids": len(ids),
        "unique_ids_ratio": round(len(ids) / n, 4) if n else 0.0,
        "unique_prompts": len(prompts),
        "unique_prompts_ratio": round(len(prompts) / n, 4) if n else 0.0,
        "max_prompt_chars": max((len(str(r.get("prompt", ""))) for r in rows), default=0),
    }


def _check(s: Dict[str, Any], enforce_labels: bool = True) -> List[str]:
    failures: List[str] = []
    if enforce_labels:
        if s["n_labels"] < 2:
            failures.append(f"only {s['n_labels']} gold label(s)")
        if s["max_label_share"] > MAX_LABEL_SHARE:
            failures.append(
                f"majority label {s['majority_label']!r} is "
                f"{s['max_label_share']:.1%} (> {MAX_LABEL_SHARE:.0%})"
            )
    if s["unique_ids_ratio"] < 1.0:
        failures.append(f"unique ids {s['unique_ids_ratio']:.1%} (< 100%)")
    if s["unique_prompts_ratio"] < MIN_UNIQUE_PROMPTS:
        failures.append(
            f"unique prompts {s['unique_prompts_ratio']:.1%} (< {MIN_UNIQUE_PROMPTS:.0%})"
        )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate official benchmark items.")
    ap.add_argument("--out", default=str(BENCH_ROOT / "validation.json"))
    args = ap.parse_args()

    items = load_items()
    if not items:
        raise SystemExit(f"no items found under {BENCH_ROOT}/*/items.jsonl")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in items:
        groups[str(r.get("benchmark", "?"))].append(r)

    summaries = {name: _summary(name, rows) for name, rows in groups.items()}

    # The holdout is reported per category; D has no gold labels.
    holdout = groups.get("v3-Holdout", [])
    holdout_split = {
        cat: _summary(f"v3-Holdout/{cat}", [r for r in holdout if r.get("split") == cat])
        for cat in ("A", "D", "E")
    }

    failures: Dict[str, List[str]] = {}
    for name in HEADLINE:
        if name not in summaries:
            failures[name] = ["missing items.jsonl"]
            continue
        f = _check(summaries[name])
        if f:
            failures[name] = f

    warnings: Dict[str, List[str]] = {}
    for cat, s in holdout_split.items():
        # The holdout is an in-distribution sanity check, not a public headline
        # benchmark. Label balance is reported but only E is a label task; A is
        # intentionally 90% unique + 10% insufficient by construction.
        f = _check(s, enforce_labels=(cat == "E"))
        if f:
            failures[f"v3-Holdout/{cat}"] = f
        if cat == "A" and s["max_label_share"] > MAX_LABEL_SHARE:
            warnings[f"v3-Holdout/{cat}"] = [
                f"majority label {s['majority_label']!r} is {s['max_label_share']:.1%}; "
                "diagnostic only (10% insufficient by design)"
            ]

    print("=" * 78)
    print("BENCHMARK VALIDATION")
    print("=" * 78)
    for name in HEADLINE + ["v3-Holdout/A", "v3-Holdout/D", "v3-Holdout/E"]:
        s = holdout_split[name.split("/")[-1]] if name.startswith("v3-Holdout/") else summaries.get(name)
        if s is None:
            print(f"\n{name}: MISSING")
            continue
        status = "FAIL" if name in failures else "PASS"
        print(f"\n[{status}] {name}  n={s['n']:,}")
        print(f"  labels: {s['labels']}")
        print(
            f"  majority={s['majority_label']!r} baseline={s['majority_baseline']:.2%} "
            f"chance={s['chance'] if s['chance'] is not None else 'n/a'}"
        )
        print(
            f"  unique ids={s['unique_ids']:,}/{s['n']:,} "
            f"({s['unique_ids_ratio']:.2%}); unique prompts={s['unique_prompts']:,} "
            f"({s['unique_prompts_ratio']:.2%}); max prompt chars={s['max_prompt_chars']:,}"
        )
        for msg in failures.get(name, []):
            print(f"  !! {msg}")
        for msg in warnings.get(name, []):
            print(f"  (warning) {msg}")

    payload = {
        "benchmarks": summaries,
        "holdout": holdout_split,
        "failures": failures,
        "warnings": warnings,
        "headline_eligible": [n for n in HEADLINE if n not in failures],
        "headline_excluded": [n for n in HEADLINE if n in failures],
        "gates": {
            "max_label_share": MAX_LABEL_SHARE,
            "min_unique_prompts": MIN_UNIQUE_PROMPTS,
            "unique_ids_required": 1.0,
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[validate] wrote {out}")

    if failures:
        print("\n[validate] FAILED gates: " + ", ".join(sorted(failures)))
        print("[validate] Failing splits are not eligible for headline accuracy.")
        raise SystemExit(1)
    print("\n[validate] all gates passed")


if __name__ == "__main__":
    main()
