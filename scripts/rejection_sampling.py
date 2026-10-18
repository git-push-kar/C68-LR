"""Stage 3 — rejection-sampling / expert iteration (runs under WSL2 + vLLM).

Samples several completions per *fresh* puzzle at T>0, keeps only those the
dependency-free solver confirms correct, and writes them as new SFT targets.
Then fine-tune those with ``scripts/train_reasoning_v3.py --init_adapter ...``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr import config_v3
from polyvalent_lr.reasoning_v3.data import InstructionExample, read_jsonl, write_jsonl
from polyvalent_lr.reasoning_v3 import prompts
from polyvalent_lr.solvers import puzzle as pz


def _collect_puzzles(source_file: Path, num: int, seed: int, oversample_hard: bool):
    rows = read_jsonl(source_file)
    metas = [e.meta for e in rows]
    rng = random.Random(seed)
    rng.shuffle(metas)
    puzzles = []
    for m in metas:
        p = pz.generate_puzzle(int(m["seed"]), n=int(m["n"]),
                               insufficient=(m.get("answer_kind") == "insufficient"))
        if p is not None:
            puzzles.append(p)
        if len(puzzles) >= num:
            break
    if oversample_hard:
        hard = [p for p in puzzles if p.n >= 5]
        puzzles = puzzles + hard  # duplicate hard puzzles to weight them up
    return puzzles


def main() -> None:
    cfg = config_v3.RejectionV3()
    ap = argparse.ArgumentParser(description="Stage 3 rejection sampling.")
    ap.add_argument("--model", type=str, default=config_v3.BASE_MODEL_PATH)
    ap.add_argument("--stage2_adapter", type=str, default=str(config_v3.STAGE2_ADAPTER_DIR))
    ap.add_argument("--source_file", type=str, default=str(config_v3.DATA_A_DIR / "train.jsonl"))
    ap.add_argument("--out", type=str, default=str(config_v3.REJECTION_DIR / "accepted.jsonl"))
    ap.add_argument("--num_puzzles", type=int, default=cfg.num_puzzles)
    ap.add_argument("--samples", type=int, default=cfg.samples_per_puzzle)
    ap.add_argument("--temperature", type=float, default=cfg.temperature)
    ap.add_argument("--max_new_tokens", type=int, default=cfg.max_new_tokens)
    ap.add_argument("--gpu_memory_utilization", type=float, default=cfg.gpu_memory_utilization)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    puzzles = _collect_puzzles(Path(args.source_file), args.num_puzzles, args.seed, cfg.oversample_hard)
    chat_prompts = [prompts.build_puzzle_prompt(p.render_problem()) for p in puzzles]
    print(f"[RS] {len(puzzles)} puzzles x {args.samples} samples @ T={args.temperature}")

    if args.dry_run:
        out = Path(args.out).with_name("_rs_prompt_preview.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n\n" + "=" * 80 + "\n\n".join(chat_prompts[:5]), encoding="utf-8")
        print(f"[dry_run] wrote preview -> {out}")
        return

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"vLLM required (run under WSL2): {exc}")

    llm = LLM(model=args.model, enable_lora=True, max_model_len=cfg.reward_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    lora = LoRARequest("stage2", 1, args.stage2_adapter)
    sp = SamplingParams(n=args.samples, temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)
    results = llm.generate(chat_prompts, sp, lora_request=lora)

    accepted: list[InstructionExample] = []
    total = correct = 0
    for p, res in zip(puzzles, results):
        for out in res.outputs:
            total += 1
            if pz.validate_answer(p, out.text):
                correct += 1
                accepted.append(InstructionExample(
                    prompt=prompts.build_puzzle_prompt(p.render_problem()),
                    target=out.text.strip(),
                    category="A", task="puzzle",
                    meta={"seed": p.seed, "n": p.n, "source": "rejection"},
                ))
    write_jsonl(Path(args.out), accepted)
    rate = (correct / total) if total else 0.0
    print(f"[RS] kept {len(accepted)} / {total} ({rate:.1%}) -> {args.out}")
    print("[RS] next: python scripts/train_reasoning_v3.py "
          f"--init_adapter {args.stage2_adapter} --train_file {args.out} "
          f"--learning_rate 5e-5 --output_dir {config_v3.STAGE3_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
