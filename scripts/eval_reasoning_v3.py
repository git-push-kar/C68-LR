"""Evaluate base vs base+adapter on the v3 holdouts (and optional benchmarks).

Reports per-set accuracy for the assembled holdout mixture:
  A puzzles  -> exact match via the solver validator
  B/C/E      -> expected-answer match against the stored gold
  D chat     -> exact/contains match (sanity, not a capability metric)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr import config_v3
from polyvalent_lr.reasoning_v3.data import read_jsonl
from polyvalent_lr.reasoning_v3 import prompts, decoding
from polyvalent_lr.solvers import puzzle as pz


def _load(model_path: str, adapter: str | None):
    import torch
    from polyvalent_lr.models.loader import load_internvl_model_and_tokenizer
    model, tokenizer = load_internvl_model_and_tokenizer(model_path, is_eval=True,
                                                         torch_dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.eval()
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _expected(ex):
    if ex.category == "A":
        meta = ex.meta
        p = pz.generate_puzzle(int(meta["seed"]), n=int(meta["n"]),
                               insufficient=(meta.get("answer_kind") == "insufficient"))
        return p
    return str(ex.meta.get("gold", ""))


def _is_correct(ex, puzzle_or_gold, text) -> bool:
    if ex.category == "A":
        return puzzle_or_gold is not None and pz.validate_answer(puzzle_or_gold, text)
    gold = re.sub(r"\s+", " ", str(puzzle_or_gold)).strip().lower()
    pred = decoding.extract_final_answer(text).lower()
    if not gold:
        return False
    if ex.category == "D":
        return gold[:60] in pred or pred[:60] in gold
    return gold in pred or pred.split(" ")[:3] == gold.split(" ")[:3]


def main() -> None:
    dec = config_v3.DecodingV3()
    ap = argparse.ArgumentParser(description="Evaluate v3 holdouts.")
    ap.add_argument("--model", type=str, default=config_v3.BASE_MODEL_PATH)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--holdout", type=str, default=str(config_v3.MIX_DIR / "holdout.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=dec.temperature)
    ap.add_argument("--self_consistency", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=dec.max_new_tokens)
    ap.add_argument("--out", type=str, default="evaluation_results_v3.json")
    args = ap.parse_args()

    examples = read_jsonl(Path(args.holdout))
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"no holdout examples in {args.holdout}")

    import torch
    model, tokenizer = _load(args.model, args.adapter)

    correct = defaultdict(int)
    total = defaultdict(int)
    for i, ex in enumerate(examples):
        enc = tokenizer(ex.prompt, return_tensors="pt").to(model.device)
        kwargs = decoding.build_generation_kwargs(
            tokenizer, temperature=args.temperature, top_p=dec.top_p,
            repetition_penalty=dec.repetition_penalty, max_new_tokens=args.max_new_tokens,
            stop_strings=dec.stop_strings,
        )
        samples = []
        for _ in range(max(1, args.self_consistency)):
            with torch.no_grad():
                out = model.generate(**enc, **kwargs)
            samples.append(tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        text = decoding.majority_vote(samples) if len(samples) > 1 else samples[0]

        gold = _expected(ex)
        total[ex.category] += 1
        correct[ex.category] += int(_is_correct(ex, gold, text))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(examples)}")

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "self_consistency": args.self_consistency,
        "per_category": {
            c: {"correct": correct[c], "total": total[c],
                "accuracy": (correct[c] / total[c]) if total[c] else 0.0}
            for c in sorted(total)
        },
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
