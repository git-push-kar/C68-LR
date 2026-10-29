"""Build REPORT.md and summary.json for the v3 benchmark run."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from polyvalent_lr.evaluation.bench_metrics import (  # noqa: E402
    bootstrap_mean_ci,
    confusion_matrix,
    fmt_ci,
    fmt_pct,
    paired_bootstrap_ci,
    parse_markdown_table,
    per_cell_accuracy,
    token_f1,
)
from polyvalent_lr.evaluation.metrics import ReasoningMetrics  # noqa: E402
from polyvalent_lr.solvers import puzzle as pz  # noqa: E402

BENCH_ROOT = ROOT / "data" / "benchmarks_real"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_items() -> Dict[str, Dict[str, Any]]:
    items: Dict[str, Dict[str, Any]] = {}
    for path in sorted(BENCH_ROOT.glob("*/items.jsonl")):
        for row in read_jsonl(path):
            items[row["id"]] = row
    return items


def load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    preds: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        preds[row["id"]] = row
    return preds


def _pred(rec: Dict[str, Any], key: str, semantic: bool = False) -> Optional[str]:
    d = rec.get(key) or {}
    if semantic:
        val = d.get("label_semantic")
        if val is not None:
            return val
    return d.get("label")


def _correct(recs: Sequence[Dict[str, Any]], key: str = "greedy", semantic: bool = False) -> List[int]:
    return [int(_pred(r, key, semantic) == r.get("gold")) for r in recs]


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


_TAG_RE = re.compile(r"</?(?:answer|reasoning|think)\b[^>]*>", re.IGNORECASE)


def _tag_spam(text: str) -> bool:
    """True when a generation repeats the contract tags more than the 4 expected."""
    return len(_TAG_RE.findall(text or "")) > 4


def _ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    return bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)


def _malformed(recs: Sequence[Dict[str, Any]], key: str = "greedy") -> int:
    return sum(1 for r in recs if (r.get(key) or {}).get("malformed"))


def _conf(recs: Sequence[Dict[str, Any]], labels: Sequence[str], key: str = "greedy") -> Dict[str, Dict[str, int]]:
    return confusion_matrix(
        ((r.get("gold", ""), _pred(r, key)) for r in recs),
        labels,
    )


def _per_label(recs: Sequence[Dict[str, Any]], labels: Sequence[str], n_boot: int, seed: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label in labels:
        sub = [r for r in recs if r.get("gold") == label]
        if not sub:
            continue
        vals = [int(_pred(r, "greedy") == label) for r in sub]
        out[label] = {
            "n": len(sub),
            "accuracy": _mean(vals),
            "ci": _ci(vals, n_boot, seed),
            "malformed": _malformed(sub),
        }
    return out


def _pattern_stats(
    recs: Sequence[Dict[str, Any]], n_boot: int, seed: int, key_fn
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in recs:
        groups[str(key_fn(rec))].append(rec)
    out: Dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        vals = _correct(rows)
        out[group] = {
            "n": len(rows),
            "accuracy": _mean(vals),
            "ci": _ci(vals, n_boot, seed),
            "majority": max(Counter(r.get("gold") for r in rows).values()) / len(rows),
        }
    return out


def _sc_stats(recs: Sequence[Dict[str, Any]], semantic: bool, n_boot: int, seed: int) -> Dict[str, Any]:
    g = _correct(recs, "greedy", semantic)
    s = _correct(recs, "sc5", semantic)
    sc_tokens = [
        float(sum((r.get("sc5") or {}).get("n_output_tokens") or []))
        / max(1, len((r.get("sc5") or {}).get("n_output_tokens") or []))
        for r in recs
    ]
    return {
        "greedy_accuracy": _mean(g),
        "greedy_ci": _ci(g, n_boot, seed),
        "sc5_accuracy": _mean(s),
        "sc5_ci": _ci(s, n_boot, seed),
        "sc5_minus_greedy": _mean(s) - _mean(g),
        "diff_ci": paired_bootstrap_ci(s, g, n_boot=n_boot, seed=seed),
        "sc5_malformed": _malformed(recs, "sc5"),
        "sc5_mean_tokens": _mean(sc_tokens),
    }


def summarize_benchmark(
    recs: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    n = len(recs)
    g = _correct(recs)
    s = _correct(recs, "sc5")
    gold_counts = Counter(r.get("gold") for r in recs)
    majority = max(gold_counts.values()) / n if n else float("nan")
    if n:
        chances = [
            float(r["chance"]) for r in recs
            if r.get("chance") is not None
        ]
        chance = _mean(chances) if chances else _mean(
            [1.0 / max(1, len(r.get("label_space") or [])) for r in recs]
        )
    else:
        chance = float("nan")
    malformed = _malformed(recs)
    correct = sum(g)
    return {
        "n": n,
        "accuracy": _mean(g),
        "ci": _ci(g, n_boot, seed),
        "majority_baseline_sample": majority,
        "chance_mean": chance,
        "malformed": malformed,
        "malformed_rate": malformed / n if n else float("nan"),
        "wrong_parsed": n - correct - malformed,
        "wrong_parsed_rate": (n - correct - malformed) / n if n else float("nan"),
        "accuracy_parsed": correct / max(1, n - malformed),
        "mean_output_tokens": _mean([float((r.get("greedy") or {}).get("n_output_tokens", 0)) for r in recs]),
        "mean_chars": _mean([len((r.get("greedy") or {}).get("raw", "")) for r in recs]),
        "per_label": _per_label(recs, labels, n_boot, seed),
        "confusion": _conf(recs, labels),
        "sc5": _sc_stats(recs, False, n_boot, seed),
    }


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _conf_md(mat: Dict[str, Dict[str, int]], labels: Sequence[str]) -> str:
    cols = list(labels) + ["<malformed>"]
    rows = []
    for gold in labels:
        rows.append([gold] + [mat.get(gold, {}).get(c, 0) for c in cols])
    return _md_table(["gold \\ pred"] + cols, rows)


def _puzzle(meta: Dict[str, Any]):
    return pz.generate_puzzle(
        int(meta["seed"]),
        int(meta["n"]),
        insufficient=(str(meta.get("answer_kind")) == "insufficient"),
    )


def holdout_a(
    recs: Sequence[Dict[str, Any]],
    items: Dict[str, Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    cache: Dict[Tuple[int, int, bool], Any] = {}
    g_exact: List[int] = []
    s_exact: List[int] = []
    unique: List[Dict[str, Any]] = []
    insufficient: List[Dict[str, Any]] = []
    g_cells = {"person": [0, 0], "domain": [0, 0], "os": [0, 0]}
    s_cells = {"person": [0, 0], "domain": [0, 0], "os": [0, 0]}

    for rec in recs:
        item = items.get(rec["id"], {})
        meta = item.get("meta") or {}
        key = (int(meta.get("seed", 0)), int(meta.get("n", 0)), str(meta.get("answer_kind")) == "insufficient")
        if key not in cache:
            cache[key] = _puzzle(meta) if meta else None
        p = cache[key]
        if p is None:
            continue
        raw_g = (rec.get("greedy") or {}).get("raw", "")
        sc = rec.get("sc5") or {}
        idx = int(sc.get("selected_index") or 0)
        raw_s = (sc.get("raw") or [""])[idx] if (sc.get("raw") or []) else ""
        ge = int(bool(pz.validate_answer(p, raw_g)))
        se = int(bool(pz.validate_answer(p, raw_s)))
        g_exact.append(ge)
        s_exact.append(se)
        bucket = insufficient if p.answer_kind == "insufficient" else unique
        bucket.append({"g": ge, "s": se})
        if p.answer_kind == "unique":
            g_table = parse_markdown_table(raw_g)
            s_table = parse_markdown_table(raw_s)
            for cat, (c, t) in per_cell_accuracy(g_table, p.solution).items():
                g_cells[cat][0] += c
                g_cells[cat][1] += t
            for cat, (c, t) in per_cell_accuracy(s_table, p.solution).items():
                s_cells[cat][0] += c
                s_cells[cat][1] += t

    def _cell_summary(cells):
        return {
            k: {
                "correct": v[0],
                "total": v[1],
                "accuracy": (v[0] / v[1]) if v[1] else float("nan"),
            }
            for k, v in cells.items()
        }

    return {
        "n": len(g_exact),
        "majority_baseline": len(unique) / len(g_exact) if g_exact else float("nan"),
        "greedy_exact": _mean(g_exact),
        "greedy_exact_ci": _ci(g_exact, n_boot, seed),
        "sc5_exact": _mean(s_exact),
        "sc5_exact_ci": _ci(s_exact, n_boot, seed),
        "unique": {
            "n": len(unique),
            "greedy_exact": _mean([x["g"] for x in unique]),
            "greedy_ci": _ci([x["g"] for x in unique], n_boot, seed),
            "sc5_exact": _mean([x["s"] for x in unique]),
            "sc5_ci": _ci([x["s"] for x in unique], n_boot, seed),
        },
        "insufficient": {
            "n": len(insufficient),
            "greedy_exact": _mean([x["g"] for x in insufficient]),
            "greedy_ci": _ci([x["g"] for x in insufficient], n_boot, seed),
            "sc5_exact": _mean([x["s"] for x in insufficient]),
            "sc5_ci": _ci([x["s"] for x in insufficient], n_boot, seed),
        },
        "greedy_cells": _cell_summary(g_cells),
        "sc5_cells": _cell_summary(s_cells),
        "sc5_minus_greedy": _mean(s_exact) - _mean(g_exact),
        "diff_ci": paired_bootstrap_ci(s_exact, g_exact, n_boot=n_boot, seed=seed),
    }


def holdout_d(
    recs: Sequence[Dict[str, Any]],
    items: Dict[str, Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    g_f1: List[float] = []
    s_f1: List[float] = []
    g_deg = g_trunc = g_tag = 0
    s_deg = s_trunc = s_tag = 0
    s_samples = 0
    g_tokens: List[int] = []
    g_chars: List[int] = []
    g_rep: List[float] = []
    g_maxrep: List[float] = []
    s_rep: List[float] = []
    s_maxrep: List[float] = []
    for rec in recs:
        item = items.get(rec["id"], {})
        target = item.get("target", "")
        raw_g = (rec.get("greedy") or {}).get("raw", "")
        g_f1.append(token_f1(raw_g, target))
        deg_g = ReasoningMetrics.detect_degeneracy(raw_g)
        g_deg += int(bool(deg_g["is_degenerate"]))
        g_tag += int(_tag_spam(raw_g))
        g_rep.append(float(deg_g["repetition_ratio"]))
        g_maxrep.append(float(deg_g["max_ngram_repeat"]))
        g_trunc += int(bool((rec.get("greedy") or {}).get("truncated")))
        g_tokens.append(int((rec.get("greedy") or {}).get("n_output_tokens", 0)))
        g_chars.append(len(raw_g))
        sc = rec.get("sc5") or {}
        raws = sc.get("raw") or []
        if raws:
            f1s = [token_f1(x, target) for x in raws]
            s_f1.append(float(sum(f1s) / len(f1s)))
            for x in raws:
                s_samples += 1
                deg_s = ReasoningMetrics.detect_degeneracy(x)
                s_deg += int(bool(deg_s["is_degenerate"]))
                s_tag += int(_tag_spam(x))
                s_rep.append(float(deg_s["repetition_ratio"]))
                s_maxrep.append(float(deg_s["max_ngram_repeat"]))
        for t in sc.get("truncated") or []:
            s_trunc += int(bool(t))
    n = len(recs)
    return {
        "n": n,
        "greedy_token_f1": _mean(g_f1),
        "greedy_token_f1_ci": _ci(g_f1, n_boot, seed),
        "sc5_token_f1": _mean(s_f1),
        "sc5_token_f1_ci": _ci(s_f1, n_boot, seed),
        "greedy_degenerate_rate": g_deg / n if n else float("nan"),
        "sc5_degenerate_rate": s_deg / s_samples if s_samples else float("nan"),
        "greedy_truncated_rate": g_trunc / n if n else float("nan"),
        "sc5_truncated_rate": s_trunc / s_samples if s_samples else float("nan"),
        "greedy_tag_spam_rate": g_tag / n if n else float("nan"),
        "sc5_tag_spam_rate": s_tag / s_samples if s_samples else float("nan"),
        "greedy_mean_repetition_ratio": _mean(g_rep),
        "sc5_mean_repetition_ratio": _mean(s_rep),
        "greedy_mean_max_ngram_repeat": _mean(g_maxrep),
        "sc5_mean_max_ngram_repeat": _mean(s_maxrep),
        "mean_output_tokens": _mean([float(x) for x in g_tokens]),
        "mean_chars": _mean([float(x) for x in g_chars]),
    }


def holdout_e(
    recs: Sequence[Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    labels = ["True", "False", "Uncertain"]
    return summarize_benchmark(recs, labels, n_boot, seed)


def contamination_stats(recs: Sequence[Dict[str, Any]], flags: Dict[str, bool]) -> Dict[str, Any]:
    flagged = [r for r in recs if flags.get(r["id"], False)]
    if not flagged:
        return {"n_flagged": 0, "accuracy_with": _mean(_correct(recs)), "accuracy_without": _mean(_correct(recs))}
    keep = [r for r in recs if not flags.get(r["id"], False)]
    return {
        "n_flagged": len(flagged),
        "accuracy_with": _mean(_correct(recs)),
        "accuracy_without": _mean(_correct(keep)),
    }


def build_report(
    results_dir: Path,
    out_path: Path,
    n_boot: int,
    seed: int,
) -> None:
    items = load_items()
    preds = load_predictions(results_dir / "predictions.jsonl")
    recs = list(preds.values())
    if not recs:
        raise SystemExit(f"no predictions in {results_dir / 'predictions.jsonl'}")

    validation_path = BENCH_ROOT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    contamination_path = BENCH_ROOT / "contamination.json"
    contamination = json.loads(contamination_path.read_text(encoding="utf-8")) if contamination_path.exists() else {}
    flags = {
        r["item_id"]: bool(r.get("contaminated"))
        for r in read_jsonl(BENCH_ROOT / "contamination_flags.jsonl")
    }
    run_config_path = results_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    timing_path = results_dir / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}

    by_bench: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_holdout: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in recs:
        if rec.get("benchmark") == "v3-Holdout":
            by_holdout[str(rec.get("split"))].append(rec)
        else:
            by_bench[str(rec.get("benchmark"))].append(rec)

    summary: Dict[str, Any] = {
        "meta": {
            "created_utc": run_config.get("created_utc"),
            "model": run_config.get("model"),
            "adapter": run_config.get("adapter"),
            "adapter_sha256": run_config.get("adapter_sha256"),
            "git_commit": run_config.get("git_commit"),
            "gpu": run_config.get("gpu"),
            "torch": run_config.get("torch"),
            "transformers": run_config.get("transformers"),
            "decoding": run_config.get("decoding"),
            "batch_size": run_config.get("batch_size"),
            "seed": run_config.get("seed"),
            "n_predictions": len(recs),
        },
        "validation": validation,
        "contamination": contamination,
        "timing": timing,
        "benchmarks": {},
        "holdout": {},
    }

    for name, rows in sorted(by_bench.items()):
        labels: List[str] = []
        for row in rows:
            for label in items.get(row["id"], {}).get("label_space") or []:
                if label not in labels:
                    labels.append(label)
        if not labels:
            labels = sorted({r.get("gold") for r in rows if r.get("gold")})
        stats = summarize_benchmark(rows, labels, n_boot, seed)
        stats["templated_rate"] = _mean([
            1.0 if "combining the premises" in str((r.get("greedy") or {}).get("raw", "")).lower() else 0.0
            for r in rows
        ])
        stats["per_pattern"] = _pattern_stats(rows, max(2000, n_boot // 4), seed, lambda r: r.get("pattern"))
        stats["contamination"] = contamination_stats(rows, flags)
        stats["headline"] = name in set(validation.get("headline_eligible", []))
        stats["labels"] = labels
        if name == "LogicNLI":
            sem_vals = _correct(rows, "greedy", True)
            stats["semantic"] = {
                "accuracy": _mean(sem_vals),
                "ci": _ci(sem_vals, n_boot, seed),
                "sc5": _sc_stats(rows, True, n_boot, seed),
            }
        if name == "Multi-LogiEval":
            stats["per_depth"] = _pattern_stats(
                rows, max(2000, n_boot // 4), seed,
                lambda r: (items.get(r["id"], {}).get("meta") or {}).get("depth", "?"),
            )
            # Keep the 111 shipped file/rule groups distinct instead of
            # collapsing the same rule across depths.
            stats["per_pattern"] = _pattern_stats(
                rows, max(2000, n_boot // 4), seed,
                lambda r: "{}:{}".format(
                    (items.get(r["id"], {}).get("meta") or {}).get("folder", "?"),
                    r.get("pattern"),
                ),
            )
        summary["benchmarks"][name] = stats

    if "A" in by_holdout:
        summary["holdout"]["A"] = holdout_a(by_holdout["A"], items, n_boot, seed)
    if "D" in by_holdout:
        summary["holdout"]["D"] = holdout_d(by_holdout["D"], items, n_boot, seed)
    if "E" in by_holdout:
        summary["holdout"]["E"] = holdout_e(by_holdout["E"], n_boot, seed)

    general_path = results_dir / "general_predictions.jsonl"
    general_rows = read_jsonl(general_path)
    if general_rows:
        summary["general"] = {
            "n": len(general_rows),
            "failed": sum(bool(r.get("failed")) for r in general_rows),
            "empty": sum(bool(r.get("empty")) for r in general_rows),
            "degenerate": sum(bool(r.get("degenerate")) for r in general_rows),
            "loose_ngram_flags": sum(bool(r.get("loose_ngram_flag")) for r in general_rows),
            "truncated": sum(bool(r.get("truncated")) for r in general_rows),
            "short_lt20_chars": sum(1 for r in general_rows if int(r.get("chars", 0)) < 20),
            "mean_chars": _mean([float(r.get("chars", 0)) for r in general_rows]),
            "mean_output_tokens": _mean([float(r.get("n_output_tokens", 0)) for r in general_rows]),
        }
    else:
        summary["general"] = {"status": "not run"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    out_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote {out_path} and {results_dir / 'summary.json'}")


def render_markdown(s: Dict[str, Any]) -> str:
    meta = s["meta"]
    val = s["validation"]
    cont = s["contamination"]
    timing = s["timing"]
    lines: List[str] = []
    add = lines.append

    add("# v3 Benchmark Report")
    add("")
    add("Model: `{}`".format(meta.get("model")))
    add("")
    add("Adapter: `{}`".format(meta.get("adapter")))
    add("")
    add("Adapter SHA-256: `{}`".format(meta.get("adapter_sha256")))
    add("")
    add("Git commit: `{}`".format(meta.get("git_commit")))
    add("")
    add("GPU: {} | torch {} | transformers {}".format(meta.get("gpu"), meta.get("torch"), meta.get("transformers")))
    add("")
    add("Predictions: {:,} | batch size {} | sampling seed {}".format(
        int(meta.get("n_predictions", 0)), meta.get("batch_size"), meta.get("seed")))
    add("")
    dec = meta.get("decoding") or {}
    add("Decoding: greedy (max_new_tokens={}) and SC5 (n=5, T={}, top_p={}, repetition_penalty={}).".format(
        (dec.get("greedy") or {}).get("max_new_tokens"),
        (dec.get("sc5") or {}).get("temperature"),
        (dec.get("sc5") or {}).get("top_p"),
        (dec.get("sc5") or {}).get("repetition_penalty"),
    ))
    add("")
    if timing:
        add("Timing gate: {:,} items in {:.0f} s ({:.2f} s/item); projected full run {:.2f} h.".format(
            int(timing.get("n_timing", 0)), float(timing.get("seconds", 0)),
            float(timing.get("seconds_per_item", 0)), float(timing.get("eta_full_hours", 0))))
        add("")

    add("## What this is")
    add("")
    add("This is a zero-shot, inference-only evaluation of the released v3 adapter on three official public logic benchmarks plus the v3 in-distribution holdout. No benchmark data was added to training, no per-benchmark prompt tuning was done, and the base model was not evaluated by default. Accuracy is reported against the dataset's majority-class baseline and chance, not against an unstated reference model.")
    add("")
    add("A benchmark is **headline-eligible** only if it passes the label-balance and prompt-uniqueness gates in `validate_data.py`. Multi-LogiEval fails the 70% label-balance gate (71.46% yes) and is therefore reported as a secondary diagnostic, not as a headline result.")
    add("")

    add("## Provenance and gates")
    add("")
    add(_md_table(
        ["Benchmark", "Items", "Evaluated", "Max label share", "Majority", "Chance", "Gate"],
        [
            [
                name,
                (val.get("benchmarks", {}).get(name) or {}).get("n", "?"),
                stats["n"],
                "{:.1%}".format((val.get("benchmarks", {}).get(name) or {}).get("max_label_share", 0.0)),
                fmt_pct((val.get("benchmarks", {}).get(name) or {}).get("majority_baseline")),
                "n/a" if (val.get("benchmarks", {}).get(name) or {}).get("chance") is None else "{:.1%}".format((val.get("benchmarks", {}).get(name) or {}).get("chance")),
                "PASS" if name in set(val.get("headline_eligible", [])) else "EXCLUDED",
            ]
            for name, stats in sorted(s["benchmarks"].items())
        ],
    ))
    add("")
    add("Gate failures: " + (json.dumps(val.get("failures", {})) if val.get("failures") else "none") + ".")
    add("")
    add("Warnings: " + (json.dumps(val.get("warnings", {})) if val.get("warnings") else "none") + ".")
    add("")

    add("## Contamination audit")
    add("")
    verdict = cont.get("verdict", "not run")
    add(verdict)
    add("")
    if cont.get("benchmarks"):
        add(_md_table(
            ["Benchmark", "Items scanned", "Contaminated", "Exact matches", "Any shared 8-gram", "Max Jaccard"],
            [[
                k, v.get("n"), v.get("contaminated"), v.get("exact_matches"),
                v.get("items_with_any_shared_8gram", "n/a"), v.get("max_jaccard", "n/a"),
            ] for k, v in sorted(cont["benchmarks"].items())],
        ))
        add("")
    add("This is an exact and 8-gram/near-duplicate audit against all four v3 training corpora (`A`, `D`, `E`, and the assembled `mix`). It is not a proof that no semantic overlap exists, and it does not audit the base model's pretraining corpus.")
    add("")

    add("## All evaluated benchmarks")
    add("")
    add(_md_table(
        ["Benchmark", "Role", "n", "Accuracy [95% CI]", "Majority", "Chance", "Malformed", "Wrong (parsed)"],
        [
            [
                n,
                "headline" if n in set(val.get("headline_eligible", [])) else "diagnostic",
                st["n"],
                fmt_ci(st["accuracy"], st["ci"]),
                fmt_pct(st["majority_baseline_sample"]),
                fmt_pct(st["chance_mean"]),
                fmt_pct(st["malformed_rate"]),
                fmt_pct(st["wrong_parsed_rate"]),
            ]
            for n, st in sorted(s["benchmarks"].items())
        ],
    ))
    add("")
    add("`Malformed` means no valid label could be extracted; `Wrong (parsed)` means a valid label was extracted but did not match the gold label. They are deliberately separate.")
    add("")

    add("## Headline results")
    add("")
    headline = [n for n in val.get("headline_eligible", []) if n in s["benchmarks"]]
    add(_md_table(
        ["Benchmark", "n", "Accuracy [95% CI]", "Majority", "Chance", "Malformed", "Mean out tokens"],
        [
            [
                n,
                s["benchmarks"][n]["n"],
                fmt_ci(s["benchmarks"][n]["accuracy"], s["benchmarks"][n]["ci"]),
                fmt_pct(s["benchmarks"][n]["majority_baseline_sample"]),
                fmt_pct(s["benchmarks"][n]["chance_mean"]),
                fmt_pct(s["benchmarks"][n]["malformed_rate"]),
                "{:.1f}".format(s["benchmarks"][n]["mean_output_tokens"]),
            ]
            for n in headline
        ],
    ))
    add("")
    add("Majority/chance are reference points, not thresholds for success. A model at or below majority has not demonstrated benchmark-level reasoning on that set.")
    add("")
    add("For the binary benchmarks, `Chance` is 50% because the gold task has two classes. The extractor also accepts the model's training label `Uncertain`, so a uniform three-way guesser would score 33.3%; the 50% column is the stricter binary-task reference.")
    add("")

    add("Contamination sensitivity (all four v3 training corpora):")
    add("")
    add(_md_table(
        ["Benchmark", "Flagged", "Accuracy with", "Accuracy without"],
        [
            [
                n,
                st["contamination"]["n_flagged"],
                fmt_pct(st["contamination"]["accuracy_with"]),
                fmt_pct(st["contamination"]["accuracy_without"]),
            ]
            for n, st in sorted(s["benchmarks"].items())
        ],
    ))
    add("")
    add("When no item is flagged these columns are identical by construction; the flag file is still written per item.")
    add("")

    for n in headline:
        st = s["benchmarks"][n]
        add(f"### {n}")
        add("")
        add("Per-label accuracy:")
        add("")
        add(_md_table(
            ["Gold label", "n", "Accuracy [95% CI]", "Malformed"],
            [
                [k, v["n"], fmt_ci(v["accuracy"], v["ci"]), v["malformed"]]
                for k, v in st["per_label"].items()
            ],
        ))
        add("")
        add("Confusion matrix (greedy):")
        add("")
        add(_conf_md(st["confusion"], st["labels"]))
        add("")
        pred_counts = {
            lab: sum(st["confusion"][g].get(lab, 0) for g in st["labels"])
            for lab in st["labels"]
        }
        top_lab, top_n = max(pred_counts.items(), key=lambda kv: kv[1])
        add("Most frequent prediction: `{}` on {:.1%} of items{}.".format(
            top_lab,
            top_n / max(1, st["n"]),
            " (gold share {:.1%})".format(st["per_label"][top_lab]["n"] / max(1, st["n"]))
            if top_lab in st["per_label"] else "",
        ))
        add("")
        add("Greedy vs SC5:")
        add("")
        add(_md_table(
            ["Decoding", "Accuracy [95% CI]", "Malformed", "Mean out tokens"],
            [
                ["greedy", fmt_ci(st["sc5"]["greedy_accuracy"], st["sc5"]["greedy_ci"]), st["malformed"], "{:.1f}".format(st["mean_output_tokens"])],
                ["SC5", fmt_ci(st["sc5"]["sc5_accuracy"], st["sc5"]["sc5_ci"]), st["sc5"]["sc5_malformed"], "{:.1f}".format(st["sc5"]["sc5_mean_tokens"])],
            ],
        ))
        add("")
        add("Paired SC5 - greedy: {:+.1%} [95% CI {:+.1%}, {:+.1%}].".format(
            st["sc5"]["sc5_minus_greedy"], st["sc5"]["diff_ci"][0], st["sc5"]["diff_ci"][1]))
        add("")
        if n == "LogicNLI" and "semantic" in st:
            sem = st["semantic"]
            add("Optional semantic mapping (`True -> entailment`, `False -> contradiction`, `Uncertain -> neutral`): greedy {:.1%} [95% CI {:.1%}, {:.1%}], SC5 {:.1%} [95% CI {:.1%}, {:.1%}].".format(
                sem["accuracy"], sem["ci"][0], sem["ci"][1],
                sem["sc5"]["sc5_accuracy"], sem["sc5"]["sc5_ci"][0], sem["sc5"]["sc5_ci"][1]))
            add("")
            add("Strict-label accuracy above is primary. The mapped number is secondary and cannot recover `self_contradiction` because T/F/U has no corresponding token.")
            add("")

    add("## Secondary diagnostic: Multi-LogiEval")
    add("")
    if "Multi-LogiEval" in s["benchmarks"]:
        st = s["benchmarks"]["Multi-LogiEval"]
        add("Multi-LogiEval is **not** in the headline table because 71.46% of its labels are `True`, failing the pre-registered 70% label-balance gate. It is shown only to make the imbalance and the model's behavior visible.")
        add("")
        add(_md_table(
            ["n", "Accuracy [95% CI]", "Majority", "Chance", "Malformed", "Mean out tokens"],
            [[st["n"], fmt_ci(st["accuracy"], st["ci"]), fmt_pct(st["majority_baseline_sample"]), fmt_pct(st["chance_mean"]), fmt_pct(st["malformed_rate"]), "{:.1f}".format(st["mean_output_tokens"])]],
        ))
        add("")
        add("Per-depth accuracy:")
        add("")
        add(_md_table(
            ["Depth", "n", "Accuracy [95% CI]", "Majority"],
            [[k, v["n"], fmt_ci(v["accuracy"], v["ci"]), fmt_pct(v["majority"])] for k, v in st.get("per_depth", {}).items()],
        ))
        add("")
        add("The full per-file rule breakdown (111 file/rule groups) is in `summary.json` under `benchmarks.Multi-LogiEval.per_pattern`.")
        add("")

    add("## Per-pattern accuracy (LogicBench)")
    add("")
    for n in ("LogicBench-BQA", "LogicBench-MCQA"):
        if n not in s["benchmarks"]:
            continue
        st = s["benchmarks"][n]
        add(f"### {n}")
        add("")
        add(_md_table(
            ["Inference rule", "n", "Accuracy [95% CI]", "Majority"],
            [[k, v["n"], fmt_ci(v["accuracy"], v["ci"]), fmt_pct(v["majority"])] for k, v in st["per_pattern"].items()],
        ))
        add("")

    add("## v3 holdout (in-distribution sanity check)")
    add("")
    add("These are not public-benchmark results. They are diagnostic checks on the v3 training distribution.")
    add("")
    if "A" in s["holdout"]:
        a = s["holdout"]["A"]
        add("### A — logic puzzles")
        add("")
        add("Exact-table accuracy (solver-validated). The stored targets for 30/300 `insufficient` puzzles are reference tables; scoring uses the regenerated puzzle, not the stored target.")
        add("")
        add(_md_table(
            ["Subset", "n", "Greedy exact [95% CI]", "SC5 exact [95% CI]"],
            [
                ["all", a["n"], fmt_ci(a["greedy_exact"], a["greedy_exact_ci"]), fmt_ci(a["sc5_exact"], a["sc5_exact_ci"])],
                ["unique", a["unique"]["n"], fmt_ci(a["unique"]["greedy_exact"], a["unique"]["greedy_ci"]), fmt_ci(a["unique"]["sc5_exact"], a["unique"]["sc5_ci"])],
                ["insufficient", a["insufficient"]["n"], fmt_ci(a["insufficient"]["greedy_exact"], a["insufficient"]["greedy_ci"]), fmt_ci(a["insufficient"]["sc5_exact"], a["insufficient"]["sc5_ci"])],
            ],
        ))
        add("")
        add("Majority baseline: always output the `unique` table = {:.1%} exact (it is wrong on every insufficient item).".format(a["majority_baseline"]))
        add("")
        add("Per-cell accuracy on unique puzzles:")
        add("")
        add(_md_table(
            ["Cell", "Greedy", "SC5"],
            [[k, fmt_pct(a["greedy_cells"][k]["accuracy"]), fmt_pct(a["sc5_cells"][k]["accuracy"])] for k in ("person", "domain", "os")],
        ))
        add("")
        add("Paired SC5 - greedy: {:+.1%} [95% CI {:+.1%}, {:+.1%}].".format(a["sc5_minus_greedy"], a["diff_ci"][0], a["diff_ci"][1]))
        add("")
    if "D" in s["holdout"]:
        d = s["holdout"]["D"]
        add("### D — open-ended explanations")
        add("")
        add(_md_table(
            ["Metric", "Greedy", "SC5"],
            [
                ["token-F1 [95% CI]", fmt_ci(d["greedy_token_f1"], d["greedy_token_f1_ci"]), fmt_ci(d["sc5_token_f1"], d["sc5_token_f1_ci"])],
                ["degenerate rate", fmt_pct(d["greedy_degenerate_rate"]), fmt_pct(d["sc5_degenerate_rate"])],
                ["truncated rate", fmt_pct(d["greedy_truncated_rate"]), fmt_pct(d["sc5_truncated_rate"])],
                ["tag-spam rate", fmt_pct(d["greedy_tag_spam_rate"]), fmt_pct(d["sc5_tag_spam_rate"])],
                ["mean repetition ratio", "{:.3f}".format(d["greedy_mean_repetition_ratio"]), "{:.3f}".format(d["sc5_mean_repetition_ratio"])],
                ["mean max repeated n-gram", "{:.2f}".format(d["greedy_mean_max_ngram_repeat"]), "{:.2f}".format(d["sc5_mean_max_ngram_repeat"])],
            ],
        ))
        add("")
        add("Mean greedy output: {:.1f} tokens, {:.0f} chars. Token-F1 is lexical overlap with the reference, not a correctness or quality score. The repetition ratio is `1 - unique_tokens/total_tokens`; the n-gram repeat count is the most frequent 2..8-gram count.".format(
            d["mean_output_tokens"], d["mean_chars"]))
        add("")
    if "E" in s["holdout"]:
        e = s["holdout"]["E"]
        add("### E — label sanity")
        add("")
        add(_md_table(
            ["n", "Accuracy [95% CI]", "Majority", "Chance", "Malformed"],
            [[e["n"], fmt_ci(e["accuracy"], e["ci"]), fmt_pct(e["majority_baseline_sample"]), fmt_pct(e["chance_mean"]), fmt_pct(e["malformed_rate"])]],
        ))
        add("")
        add("Paired SC5 - greedy: {:+.1%} [95% CI {:+.1%}, {:+.1%}].".format(e["sc5"]["sc5_minus_greedy"], e["sc5"]["diff_ci"][0], e["sc5"]["diff_ci"][1]))
        add("")

    add("## General-chat degeneration probes")
    add("")
    g = s.get("general") or {}
    if g.get("status") == "not run":
        add("Not run. Run `scripts/bench/general_probes.py` after the benchmark to populate this section.")
    else:
        add("{} ordinary prompts, scored for emptiness, looping, truncation, and very short output. These are not a capability score.".format(g.get("n", 0)))
        add("")
        add(_md_table(
            ["Failed", "Degenerate (loop)", "Loose n-gram flag", "Empty", "Truncated", "<20 chars", "Mean chars", "Mean out tokens"],
            [[g.get("failed"), g.get("degenerate"), g.get("loose_ngram_flags", 0), g.get("empty"), g.get("truncated"), g.get("short_lt20_chars"), "{:.0f}".format(g.get("mean_chars", 0)), "{:.1f}".format(g.get("mean_output_tokens", 0))]],
        ))
        add("")
        add("`Degenerate (loop)` requires more than 60% of tokens to be repeats, so enumerated lists and recipes do not count. `Loose n-gram flag` shows how many outputs the generic 5x n-gram rule would flag; those were inspected and are structured answers, not loops.")
        add("")
        add("Failure here means the adapter is not safe to present as a general assistant; it does not affect the logic-benchmark accuracy, which is reported separately.")
    add("")

    add("## Plain-language conclusion")
    add("")
    for n in headline:
        st = s["benchmarks"][n]
        rel = "above" if st["accuracy"] > st["majority_baseline_sample"] else ("at" if abs(st["accuracy"] - st["majority_baseline_sample"]) < 0.005 else "below")
        add("- **{}**: {:.1%} accuracy (95% CI {:.1%}–{:.1%}) vs {:.1%} majority and {:.1%} chance; the point estimate is {} the majority baseline.".format(
            n, st["accuracy"], st["ci"][0], st["ci"][1], st["majority_baseline_sample"], st["chance_mean"], rel))
        pred_counts = {lab: sum(st["confusion"][g].get(lab, 0) for g in st["labels"]) for lab in st["labels"]}
        top_lab, top_n = max(pred_counts.items(), key=lambda kv: kv[1])
        if top_n / max(1, st["n"]) >= 0.6:
            add("- **{} label concentration**: `{}` accounts for {:.1%} of predictions, so the aggregate accuracy is dominated by one output label rather than balanced performance.".format(
                n, top_lab, top_n / max(1, st["n"])))
    add("- **Multi-LogiEval** is excluded from the headline because its 71.46% yes rate fails the label-balance gate; see the diagnostic table.")
    add("- **SC5** is a paired decoding comparison to greedy on the same items; its CI is the uncertainty of that difference, not an independent effect claim.")
    add("- Only one model (the v3 adapter on the fixed base) was evaluated. No v2, base, or second system was run, so these are absolute accuracies; no comparative claim is made.")
    add("")
    add("## Limitations")
    add("")
    add("- **Single model, one checkpoint.** Only v3 (base + `internvl3_2b_reasoning_v3/sft`) is reported. `--include_base` is off by default, and no v2 result is comparable.")
    add("- **Sample sizes.** LogicBench-BQA uses a 600/1,520 stratified sample, Multi-LogiEval 600/1,696, LogicNLI 600/2,000, LogicBench-MCQA all 500, and each holdout split all 300. Bootstrap intervals reflect item-sampling variability only.")
    add("- **One seed.** Sampling uses seed 42 and bootstrap CIs use seed 20261028. Self-consistency is stochastic; these intervals do not include prompt, seed, or hyperparameter uncertainty.")
    add("- **Contamination scope.** The audit is exact normalized match plus 8-gram Jaccard/containment and MinHash confirmation over A, D, E, and mix. It is not a semantic-entailment audit, and it cannot rule out the base model having seen these public benchmarks during pretraining.")
    add("- **Label space.** The binary benchmarks keep their official True/False gold, but the model is allowed to emit `Uncertain` because that is its training contract; `Uncertain` is parsed and counted wrong, not malformed.")
    add("- Multi-LogiEval is label-imbalanced (71.5% True) and is not a headline result.")
    add("- LogicNLI uses the natural-language `test_language` split only; `test_logic` is symbolic and lacks a shipped verbaliser. Its four labels are a zero-shot output space not seen in v3 SFT.")
    add("- LogicBench-MCQA gold is the `choice_N` key; a model that writes the option text is mapped to a key by the extractor, but a model that writes only T/F/U is malformed. MCQA option keys are also a zero-shot output space.")
    add("- The v3 holdout is in-distribution and is not evidence of generalization. A has 30/300 `insufficient` items whose stored targets are reference tables; the scorer regenerates the puzzle and validates the answer kind.")
    add("- D has no gold labels, so token-F1 and degeneration are diagnostics, not accuracy. Token-F1 is lexical overlap, not correctness.")
    add("- **No human evaluation was performed.**")
    add("- **Outputs are short and templated.** Greedy answers average {:.0f}–{:.0f} tokens, and {:.0%}–{:.0%} of them reduce to `Combining the premises, the answer is: X.`; these numbers measure final-label accuracy, not the quality of the reasoning trace.".format(
        min(s["benchmarks"][n]["mean_output_tokens"] for n in headline) if headline else 0,
        max(s["benchmarks"][n]["mean_output_tokens"] for n in headline) if headline else 0,
        min(s["benchmarks"][n]["templated_rate"] for n in headline) if headline else 0,
        max(s["benchmarks"][n]["templated_rate"] for n in headline) if headline else 0,
    ))
    add("")
    add("## Artifacts")
    add("")
    add("- `predictions.jsonl` — per-item greedy and SC5 outputs, labels, malformed flags, latencies.")
    add("- `summary.json` — machine-readable metrics, per-pattern breakdown, CIs.")
    add("- `timing.json` — 25-item gate and ETA.")
    add("- `run_config.json` — model/adapter hashes, versions, decoding, git state.")
    add("- `general_predictions.jsonl` / `general_summary.json` — general-chat probes.")
    add("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/bench_20261028")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    results_dir = ROOT / args.results_dir
    out = Path(args.out) if args.out else results_dir / "REPORT.md"
    build_report(results_dir, out, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
