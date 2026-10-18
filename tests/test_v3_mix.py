"""Tests for the v3 mixture builder and decoding self-consistency helpers."""
from pathlib import Path

from polyvalent_lr.reasoning_v3.data import (
    InstructionExample, build_mix, dedup_by_prompt, write_jsonl,
)
from polyvalent_lr.reasoning_v3 import decoding


def _make_set(root: Path, cat: str, n: int):
    d = root / cat
    d.mkdir(parents=True, exist_ok=True)
    train = [InstructionExample(f"p{cat}{i}" * 5, f"t{cat}{i}" * 5, cat) for i in range(n)]
    hold = [InstructionExample(f"h{cat}{i}", f"x{cat}{i}", cat) for i in range(10)]
    write_jsonl(d / "train.jsonl", train)
    write_jsonl(d / "holdout.jsonl", hold)


def test_build_mix_counts_and_holdout(tmp_path):
    for cat in "ABCDE":
        _make_set(tmp_path, cat, 200)
    dirs = {c: tmp_path / c for c in "ABCDE"}
    weights = {"A": 0.45, "B": 0.20, "C": 0.10, "D": 0.10, "E": 0.15}
    train, hold, counts = build_mix(dirs, weights, target_tokens=2000, seed=1)
    assert train, "no training examples produced"
    assert len(hold) == 50  # 10 per set x 5
    assert all(c >= 0 for c in counts.values())
    # dedup invariant
    assert len(dedup_by_prompt(train)) == len(train)


def test_majority_vote_picks_consistent_table():
    good = "answer\n| Desk | Name | Domain | OS |\n| 1 | Alice | AI | Linux |\n| 2 | Bob | Cloud | macOS |"
    other = "answer\n| Desk | Name | Domain | OS |\n| 1 | Bob | AI | Linux |\n| 2 | Alice | Cloud | macOS |"
    chosen = decoding.majority_vote([good, good, other])
    assert chosen == good
    assert decoding.table_signature(good) is not None
