"""Assemble the v3 SFT mixture (A 45 / B 20 / C 10 / D 10 / E 15).

Also materialises Sets D (Dolly rehearsal) and E (ProofWriter/FOLIO labels) from
the existing processed data into the v3 prompt/target schema if missing.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr import config_v3
from polyvalent_lr.reasoning_v3.data import (
    InstructionExample, build_mix, dedup_by_prompt, read_jsonl, write_jsonl,
)
from polyvalent_lr.reasoning_v3 import prompts

PROCESSED = Path("data/processed")
DEFAULT_TRAINING_SYSTEM = prompts.SYSTEM_LOGIC

_JUNK_MARKERS = ("[@", "[CWA", "CWA.", "deepest failure", "(FAIL)")


def _clean_trace(raw: str, label: str) -> str:
    """Drop dataset-internal debug strings; otherwise keep the gold trace."""
    text = (raw or "").strip()
    if not text or len(text) > 800 or any(m in text for m in _JUNK_MARKERS):
        return f"Combining the premises, the answer is: {label}."
    return text


def _to_chat_rehearsal(rec: Dict) -> InstructionExample:
    user = (rec.get("hypothesis_or_query") or "").strip()
    answer = (rec.get("proof_trace_text") or rec.get("label") or "").strip()
    return InstructionExample(
        prompt=prompts.build_chat_prompt(prompts.SYSTEM_CHAT, user),
        target=prompts.build_chat_target(answer),
        category="D", task="chat", meta={"source": rec.get("source", "Rehearsal")},
    )


def _to_label_example(rec: Dict) -> InstructionExample:
    premises = "\n".join(f"- {p}" for p in rec.get("premises", []))
    query = (rec.get("hypothesis_or_query") or "").strip()
    label = str(rec.get("label", "")).strip()
    reasoning = _clean_trace(rec.get("proof_trace_text") or "", label)
    user = (
        "Given the facts and rules (premises), decide whether the hypothesis is "
        "True, False, or Uncertain.\n\n"
        f"Premises:\n{premises}\n\nHypothesis:\n{query}"
    )
    return InstructionExample(
        prompt=prompts.build_chat_prompt(DEFAULT_TRAINING_SYSTEM, user),
        target=prompts.build_reasoning_target(reasoning, label),
        category="E", task="labels", meta={"source": rec.get("source"), "gold": label},
    )


def prepare_d(seed: int = 42) -> None:
    d_dir = config_v3.DATA_D_DIR
    if (d_dir / "train.jsonl").exists():
        return
    rows = [json.loads(l) for l in (PROCESSED / "Rehearsal_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples = [_to_chat_rehearsal(r) for r in rows]
    write_jsonl(d_dir / "train.jsonl", examples[: -300])
    write_jsonl(d_dir / "holdout.jsonl", examples[-300:])
    print(f"[prepare D] wrote {len(examples) - 300} train + 300 holdout")


def prepare_e(seed: int = 42) -> None:
    e_dir = config_v3.DATA_E_DIR
    if (e_dir / "train.jsonl").exists():
        return
    rows: List[Dict] = []
    for name in ("ProofWriter_train.jsonl", "FOLIO_train.jsonl"):
        rows += [json.loads(l) for l in (PROCESSED / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples = [_to_label_example(r) for r in rows]
    write_jsonl(e_dir / "train.jsonl", examples[:-300])
    write_jsonl(e_dir / "holdout.jsonl", examples[-300:])
    print(f"[prepare E] wrote {len(examples) - 300} train + 300 holdout")


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the v3 SFT mixture.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_tokens", type=int, default=config_v3.SFTV3().target_tokens_per_epoch)
    ap.add_argument("--skip_prepare", action="store_true")
    args = ap.parse_args()

    if not args.skip_prepare:
        prepare_d(args.seed)
        prepare_e(args.seed)

    set_dirs = {
        "A": config_v3.DATA_A_DIR,
        "B": config_v3.DATA_B_DIR,
        "C": config_v3.DATA_C_DIR,
        "D": config_v3.DATA_D_DIR,
        "E": config_v3.DATA_E_DIR,
    }
    weights = config_v3.SFTV3().mix_weights
    config_v3.validate_mix_weights(weights)
    print(f"[mix] weights = {weights}")

    train, holdout, counts = build_mix(set_dirs, weights, args.target_tokens, seed=args.seed)
    n_tr = write_jsonl(config_v3.MIX_DIR / "train.jsonl", train)
    n_ho = write_jsonl(config_v3.MIX_DIR / "holdout.jsonl", holdout)
    approx = sum(max(1, (len(e.prompt) + len(e.target)) // 4) for e in train)
    print(f"[mix] per-set counts: {counts}")
    print(f"[mix] wrote {n_tr} train + {n_ho} holdout (~{approx:,} tokens/epoch) -> {config_v3.MIX_DIR}")


if __name__ == "__main__":
    main()
