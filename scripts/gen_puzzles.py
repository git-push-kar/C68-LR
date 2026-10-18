"""Set A: generate grid/ordering puzzles (solver traces) — CPU only, no GPU.

Each example stores its generation seed+size so Stage-3 rejection sampling can
regenerate the exact puzzle and verify answers without shipping the full object.

Example:
    python scripts/gen_puzzles.py --n_total 20000 --holdout 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr import config_v3
from polyvalent_lr.reasoning_v3.data import InstructionExample, write_jsonl
from polyvalent_lr.reasoning_v3 import prompts
from polyvalent_lr.solvers import puzzle as pz


def _example(seed: int, n: int, insufficient: bool) -> InstructionExample | None:
    p = pz.generate_puzzle(seed, n=n, insufficient=insufficient)
    if p is None:
        return None
    problem = p.render_problem()
    target = prompts.build_puzzle_target(p.render_trace(), p.render_table())
    return InstructionExample(
        prompt=prompts.build_puzzle_prompt(problem),
        target=target,
        category="A",
        task="puzzle",
        meta={"seed": seed, "n": p.n, "answer_kind": p.answer_kind, "query": p.query},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Set A grid/ordering puzzles.")
    ap.add_argument("--out_dir", type=str, default=str(config_v3.DATA_A_DIR))
    ap.add_argument("--n_total", type=int, default=20000)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--min_entities", type=int, default=3)
    ap.add_argument("--max_entities", type=int, default=6)
    ap.add_argument("--unsolvable_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--max_attempts_per_item", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.n_total + args.holdout
    train: list[InstructionExample] = []
    holdout: list[InstructionExample] = []
    made = 0
    seed = args.seed
    insufficient_every = max(1, int(round(1.0 / max(1e-6, args.unsolvable_ratio))))
    while made < total:
        n = args.min_entities + (seed % (args.max_entities - args.min_entities + 1))
        insufficient = (made % insufficient_every == 0)
        ex = None
        for _ in range(args.max_attempts_per_item):
            ex = _example(seed, n, insufficient)
            seed += 1
            if ex is not None:
                break
        if ex is None:
            continue
        (holdout if made < args.holdout else train).append(ex)
        made += 1
        if made % 2000 == 0:
            print(f"[Set A] generated {made}/{total}")

    n_tr = write_jsonl(out_dir / "train.jsonl", train)
    n_ho = write_jsonl(out_dir / "holdout.jsonl", holdout)
    print(f"[Set A] wrote {n_tr} train + {n_ho} holdout -> {out_dir}")


if __name__ == "__main__":
    main()
