"""v3 SFT dataset: loading, assistant-only masking (incl. EOS), mixing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # torch is optional at import time so the mixing helpers stay testable
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


# ------------------------------------------------------------------------------
# Example container
# ------------------------------------------------------------------------------
@dataclass
class InstructionExample:
    prompt: str
    target: str
    category: str                 # A|B|C|D|E
    task: str = "reasoning"       # puzzle|deduction|commonsense|chat|labels
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        h = hashlib.sha1((self.category + "\x00" + self.prompt).encode("utf-8"))
        return h.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "target": self.target,
            "category": self.category,
            "task": self.task,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InstructionExample":
        return InstructionExample(
            prompt=d["prompt"],
            target=d["target"],
            category=d.get("category", "?"),
            task=d.get("task", "reasoning"),
            meta=d.get("meta", {}) or {},
        )


# ------------------------------------------------------------------------------
# JSONL helpers
# ------------------------------------------------------------------------------
def read_jsonl(path: Path) -> List[InstructionExample]:
    out: List[InstructionExample] = []
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(InstructionExample.from_dict(json.loads(line)))
    return out


def write_jsonl(path: Path, examples: Iterable[InstructionExample]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def dedup_by_prompt(examples: Sequence[InstructionExample]) -> List[InstructionExample]:
    seen = set()
    out: List[InstructionExample] = []
    for ex in examples:
        if ex.uid in seen:
            continue
        seen.add(ex.uid)
        out.append(ex)
    return out


# ------------------------------------------------------------------------------
# Tokenization with assistant-only masking (prompt -> -100, target + EOS kept)
# ------------------------------------------------------------------------------
def encode_supervised(
    tokenizer,
    prompt: str,
    target: str,
    max_length: int,
) -> Dict[str, List[int]]:
    """Return input_ids/attention_mask/labels with the prompt masked to -100.

    The assistant target (including its EOS: ``<|im_end|>``) is always supervised.
    If the pair exceeds ``max_length`` the *prompt* is truncated from the left,
    preserving the target and its EOS; only a target that alone exceeds the
    budget is trimmed.
    """
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
        target_ids = target_ids + [eos_id]

    if len(target_ids) >= max_length:
        target_ids = target_ids[:max_length]
        prompt_ids = []
    else:
        budget = max_length - len(target_ids)
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SupervisedDataset(Dataset):  # type: ignore[misc]
    def __init__(self, examples: Sequence[InstructionExample], tokenizer, max_length: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        ex = self.examples[idx]
        return encode_supervised(self.tokenizer, ex.prompt, ex.target, self.max_length)


@dataclass
class AssistantMaskingCollator:
    """Pads a batch (right padding); label padding is -100."""
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for collation")
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0

        input_ids, attn, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ------------------------------------------------------------------------------
# Mixture assembly
# ------------------------------------------------------------------------------
def _approx_tokens(ex: InstructionExample) -> int:
    return max(1, (len(ex.prompt) + len(ex.target)) // 4)


def build_mix(
    set_dirs: Dict[str, Path],
    weights: Dict[str, float],
    target_tokens: int,
    seed: int = 42,
    holdout_per_set: int = 300,
) -> Tuple[List[InstructionExample], List[InstructionExample], Dict[str, int]]:
    """Sample each set to approximate ``target_tokens`` weighted by ``weights``.

    Returns (train, holdout, per-set sample counts). Sets that cannot fill their
    share have their surplus redistributed to the others.
    """
    import random

    rng = random.Random(seed)
    loaded: Dict[str, List[InstructionExample]] = {}
    holdouts: Dict[str, List[InstructionExample]] = {}
    for cat, d in set_dirs.items():
        train = read_jsonl(Path(d) / "train.jsonl")
        hold = read_jsonl(Path(d) / "holdout.jsonl")
        if not hold and train:
            train, hold = split_off(train, holdout_per_set, rng)
        loaded[cat] = train
        holdouts[cat] = hold

    mean_tok = {c: (sum(_approx_tokens(e) for e in v) / len(v)) if v else 1.0
                for c, v in loaded.items()}

    counts = {c: 0 for c in set_dirs}
    remaining = dict(weights)
    pool = [c for c in set_dirs if loaded[c]]
    while pool and target_tokens > 0:
        active = [c for c in pool if remaining[c] > 1e-9]
        if not active:
            active = pool  # redistribute uniformly among the remaining
            share = 1.0 / len(active)
            for c in active:
                remaining[c] = share
        wsum = sum(remaining[c] for c in active)
        progressed = False
        for c in list(active):
            want_tokens = target_tokens * (remaining[c] / wsum)
            want = int(want_tokens / mean_tok[c])
            avail = len(loaded[c]) - counts[c]
            take = min(want, avail)
            if take > 0:
                counts[c] += take
                progressed = True
            if counts[c] >= len(loaded[c]):
                pool.remove(c)
                remaining[c] = 0.0
        if not progressed:
            break

    train: List[InstructionExample] = []
    for c, v in loaded.items():
        rng.shuffle(v)
        train.extend(v[: counts[c]])
    rng.shuffle(train)
    train = dedup_by_prompt(train)

    combined_hold = dedup_by_prompt([e for v in holdouts.values() for e in v])
    return train, combined_hold, counts


def split_off(examples: List[InstructionExample], k: int, rng) -> Tuple[List[InstructionExample], List[InstructionExample]]:
    examples = list(examples)
    rng.shuffle(examples)
    hold = examples[:k]
    train = examples[k:]
    return train, hold
