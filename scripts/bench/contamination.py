"""Contamination audit: benchmark items vs the v3 SFT corpus.

Four passes, one per v3 training corpus:
  1. ``data/v3/A/train.jsonl`` (20,000 puzzle examples).
  2. ``data/v3/D/train.jsonl`` (13,913 rehearsal examples).
  3. ``data/v3/E/train.jsonl`` (514,769 label examples).
  4. ``data/v3/mix/train.jsonl`` (72,153 examples) — the actual SFT mixture.

Signals (on normalized premise+hypothesis content):
  * exact normalized content match
  * word 8-gram Jaccard >= 0.50
  * word 8-gram containment (shared / benchmark grams) >= 0.80
  * bottom-k MinHash estimate for the strongest candidates

The official benchmark repositories are *not* downloaded by the scripts that
built Set E (ProofWriter + FOLIO), but this is verified empirically rather than
assumed.  Flagged items are written out so report.py can show metrics with and
without them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENCH_ROOT = ROOT / "data" / "benchmarks_real"
MIX_TRAIN = ROOT / "data" / "v3" / "mix" / "train.jsonl"
SET_A_TRAIN = ROOT / "data" / "v3" / "A" / "train.jsonl"
SET_D_TRAIN = ROOT / "data" / "v3" / "D" / "train.jsonl"
SET_E_TRAIN = ROOT / "data" / "v3" / "E" / "train.jsonl"
OUT = BENCH_ROOT / "contamination.json"
FLAGS_OUT = BENCH_ROOT / "contamination_flags.jsonl"

NGRAM = 8
JACCARD_THRESHOLD = 0.50
CONTAINMENT_THRESHOLD = 0.80
MINHASH_K = 64

_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^a-z0-9]+")
_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
_PREMISES_RE = re.compile(
    r"Premises:\s*(.*?)(?:\n\s*(?:Hypothesis|Question|Statement):\s*(.*))?$",
    re.DOTALL | re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _NONWORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def train_content(prompt: str) -> str:
    m = _USER_RE.search(prompt or "")
    user = m.group(1) if m else (prompt or "")
    pm = _PREMISES_RE.search(user)
    if pm:
        premises = pm.group(1) or ""
        query = pm.group(2) or ""
        return premises + "\n" + query
    return user


def benchmark_content(item: Dict[str, Any]) -> str:
    return "\n".join(str(p) for p in item.get("premises", [])) + "\n" + str(item.get("question", ""))


def grams(text: str) -> frozenset[Tuple[str, ...]]:
    toks = text.split()
    if len(toks) < NGRAM:
        return frozenset()
    return frozenset(tuple(toks[i:i + NGRAM]) for i in range(len(toks) - NGRAM + 1))


def minhash_signature(text: str, k: int = MINHASH_K) -> Tuple[int, ...]:
    return minhash_signature_from_grams(grams(text), k)


def _gram_hash(gram: Tuple[str, ...], seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(
            " ".join(gram).encode("utf-8"),
            digest_size=8,
            key=seed.to_bytes(8, "little"),
        ).digest(),
        "big",
    )


def minhash_signature_from_grams(gs: Iterable[Tuple[str, ...]], k: int = MINHASH_K) -> Tuple[int, ...]:
    """True MinHash: minimum hash under ``k`` independent hash functions."""
    grams = list(gs)
    if not grams:
        return ()
    return tuple(min(_gram_hash(g, seed) for g in grams) for seed in range(k))


def minhash_jaccard(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def load_benchmarks() -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(BENCH_ROOT.glob("*/items.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("benchmark") == "v3-Holdout":
                    continue  # in-distribution by construction; not a contamination test
                groups[str(item["benchmark"])].append(item)
    return dict(groups)


def iter_train(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def scan(
    corpus: Iterable[Dict[str, Any]],
    bench: Dict[str, Dict[str, Any]],
    bench_norm: Dict[str, str],
    bench_grams: Dict[str, frozenset],
    gram_index: Dict[Tuple[str, ...], Set[str]],
    gram_set: Set[Tuple[str, ...]],
    corpus_name: str,
) -> Dict[str, Any]:
    """Return best match per benchmark item plus counts."""
    best: Dict[str, Dict[str, Any]] = {}
    exact_hits: Dict[str, List[str]] = defaultdict(list)
    n = 0
    t0 = time.time()
    for idx, row in enumerate(corpus):
        n += 1
        content = train_content(row.get("prompt", ""))
        norm = normalize(content)
        if not norm:
            continue
        if norm in bench_norm:
            for iid in bench_norm[norm]:
                exact_hits[iid].append(f"{corpus_name}:{idx}")
        gs = grams(norm)
        if not gs:
            continue
        shared = gs & gram_set
        if not shared:
            continue
        candidates: Set[str] = set()
        for g in shared:
            candidates.update(gram_index[g])
        for iid in candidates:
            bg = bench_grams[iid]
            inter = len(gs & bg)
            if not inter:
                continue
            union = len(gs | bg)
            jac = inter / union if union else 0.0
            contain = inter / len(bg) if bg else 0.0
            prev = best.get(iid)
            if prev is None or (jac, contain) > (prev["jaccard"], prev["containment"]):
                best[iid] = {
                    "item_id": iid,
                    "benchmark": bench[iid]["benchmark"],
                    "corpus": corpus_name,
                    "train_index": idx,
                    "jaccard": round(jac, 4),
                    "containment": round(contain, 4),
                    "shared_8grams": inter,
                    "benchmark_8grams": len(bg),
                    "train_chars": len(content),
                    "train_grams": gs,
                    "train_snippet": content[:300].replace("\n", " "),
                }
        if (idx + 1) % 100000 == 0:
            print(f"  [{corpus_name}] {idx + 1:,} examples, {time.time() - t0:.1f}s", flush=True)
    print(f"  [{corpus_name}] scanned {n:,} examples in {time.time() - t0:.1f}s")
    return {"best": best, "exact_hits": dict(exact_hits), "n": n}


def is_contaminated(rec: Dict[str, Any], exact: bool) -> bool:
    return bool(
        exact
        or rec["jaccard"] >= JACCARD_THRESHOLD
        or rec["containment"] >= CONTAINMENT_THRESHOLD
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit benchmark contamination in the v3 corpus.")
    ap.add_argument("--skip_set_e", action="store_true", help="Skip the 514k full Set E pass.")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--flags_out", default=str(FLAGS_OUT))
    args = ap.parse_args()

    groups = load_benchmarks()
    bench: Dict[str, Dict[str, Any]] = {}
    bench_norm: Dict[str, List[str]] = defaultdict(list)
    bench_grams: Dict[str, frozenset] = {}
    gram_index: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
    for name, items in groups.items():
        for item in items:
            iid = item["id"]
            bench[iid] = item
            content = benchmark_content(item)
            norm = normalize(content)
            bench_norm[norm].append(iid)
            gs = grams(norm)
            bench_grams[iid] = gs
            for g in gs:
                gram_index[g].add(iid)
    print(
        f"[contamination] {len(bench):,} benchmark items, "
        f"{len(gram_index):,} unique 8-grams, {sum(len(v) for v in groups.values()):,} items"
    )

    passes: Dict[str, Any] = {}
    corpora = [
        ("set_a_train", SET_A_TRAIN),
        ("set_d_train", SET_D_TRAIN),
    ]
    if not args.skip_set_e:
        corpora.append(("set_e_train", SET_E_TRAIN))
    corpora.append(("mix_train", MIX_TRAIN))
    for corpus_name, corpus_path in corpora:
        if not corpus_path.exists():
            raise SystemExit(f"missing {corpus_path}")
        passes[corpus_name] = scan(
            iter_train(corpus_path), bench, bench_norm, bench_grams,
            gram_index, set(gram_index), corpus_name,
        )

    # Merge passes: a benchmark item is contaminated if either corpus flags it.
    merged: Dict[str, Dict[str, Any]] = {}
    exact_sources: Dict[str, List[str]] = defaultdict(list)
    for pname, p in passes.items():
        for iid, rec in p["best"].items():
            prev = merged.get(iid)
            if prev is None or (rec["jaccard"], rec["containment"]) > (prev["jaccard"], prev["containment"]):
                merged[iid] = rec
        for iid, refs in p["exact_hits"].items():
            exact_sources[iid].extend(refs)
    for iid, rec in merged.items():
        rec["exact_normalized_match"] = iid in exact_sources
        rec["minhash_jaccard"] = round(
            minhash_jaccard(
                minhash_signature_from_grams(rec["train_grams"]),
                minhash_signature_from_grams(bench_grams[iid]),
            ),
            4,
        )

    # Build per-benchmark summaries and flags.
    flags: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        flagged = []
        for item in items:
            iid = item["id"]
            rec = merged.get(iid)
            exact = iid in exact_sources
            if rec is None:
                if exact:
                    rec = {
                        "jaccard": 0.0, "containment": 0.0, "shared_8grams": 0,
                        "benchmark_8grams": len(bench_grams[iid]), "corpus": "exact",
                        "train_index": -1, "train_snippet": "",
                        "minhash_jaccard": 0.0,
                    }
                else:
                    flags.append({"item_id": iid, "benchmark": name, "contaminated": False})
                    continue
            contam = is_contaminated(rec, exact)
            out = {
                "item_id": iid,
                "benchmark": name,
                "contaminated": contam,
                "exact_normalized_match": exact,
                "jaccard": rec["jaccard"],
                "containment": rec["containment"],
                "shared_8grams": rec["shared_8grams"],
                "benchmark_8grams": rec["benchmark_8grams"],
                "minhash_jaccard": rec["minhash_jaccard"],
                "corpus": rec["corpus"],
                "train_index": rec["train_index"],
                "train_snippet": rec["train_snippet"],
            }
            flags.append(out)
            if contam:
                flagged.append(out)
        name_flags = [f for f in flags if f["benchmark"] == name]
        candidates = [
            f for f in name_flags
            if (f.get("shared_8grams") or 0) > 0 or f.get("exact_normalized_match")
        ]
        top = sorted(
            candidates,
            key=lambda r: (r.get("jaccard") or 0.0, r.get("containment") or 0.0),
            reverse=True,
        )[:10]
        summary[name] = {
            "n": len(items),
            "contaminated": len(flagged),
            "contaminated_rate": round(len(flagged) / len(items), 4) if items else 0.0,
            "exact_matches": sum(1 for r in flagged if r["exact_normalized_match"]),
            "items_with_any_shared_8gram": len(candidates),
            "max_jaccard": round(max((f.get("jaccard") or 0.0 for f in name_flags), default=0.0), 4),
            "max_containment": round(max((f.get("containment") or 0.0 for f in name_flags), default=0.0), 4),
            "top_matches": top,
        }

    provenance = {
        "set_e_builder": "scripts/build_sft_mix.py:prepare_e()",
        "set_e_sources": [
            "data/processed/ProofWriter_train.jsonl",
            "data/processed/FOLIO_train.jsonl",
        ],
        "corpora_scanned": [
            "data/v3/A/train.jsonl",
            "data/v3/D/train.jsonl",
            "data/v3/E/train.jsonl",
            "data/v3/mix/train.jsonl",
        ],
        "set_a_examples_scanned": passes.get("set_a_train", {}).get("n", 0),
        "set_d_examples_scanned": passes.get("set_d_train", {}).get("n", 0),
        "proofwriter_origin": "AI2 official ProofWriter archive (download_real_datasets.py)",
        "folio_origin": "Yale-LILY/FOLIO GitHub v0.0 (download_real_datasets.py)",
        "benchmark_downloads": (
            "download_real_datasets.py fetches FOLIO, ProofWriter and AbductionRules only; "
            "it does not fetch LogicBench, Multi-LogiEval or LogicNLI. prepare_data.py "
            "writes small starter/placeholder benchmark files, not the real releases. "
            "The three real benchmark repositories were cloned separately under "
            "data/benchmarks_real/*/raw for this audit."
        ),
        "full_set_e_scan": not args.skip_set_e,
        "set_e_examples_scanned": passes.get("set_e_train", {}).get("n", 0),
    }

    payload = {
        "thresholds": {
            "ngram": NGRAM,
            "jaccard": JACCARD_THRESHOLD,
            "containment": CONTAINMENT_THRESHOLD,
            "minhash_k": MINHASH_K,
        },
        "passes": {k: {"n": v["n"]} for k, v in passes.items()},
        "benchmarks": summary,
        "provenance": provenance,
        "verdict": (
            "No benchmark item was derived from the v3 SFT corpora (A, D, E, or "
            "mix) by exact or near-duplicate content match."
            if all(v["contaminated"] == 0 for v in summary.values())
            else "At least one benchmark item overlaps the v3 SFT corpora; see flags."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(args.flags_out, "w", encoding="utf-8") as fh:
        for rec in flags:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 78)
    print("CONTAMINATION SUMMARY")
    print("=" * 78)
    for name, s in summary.items():
        print(
            f"{name:20s} n={s['n']:5,d} contaminated={s['contaminated']:4d} "
            f"({s['contaminated_rate']:.2%}) exact={s['exact_matches']}"
        )
    print(f"\n[contamination] wrote {args.out}")
    print(f"[contamination] wrote {args.flags_out}")
    print(f"[contamination] verdict: {payload['verdict']}")


if __name__ == "__main__":
    main()
