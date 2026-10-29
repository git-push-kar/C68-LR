"""Run >=30 harmless general prompts and score degeneration.

These probes are not a capability benchmark.  They check that the v3 adapter
still produces coherent, non-looping, non-truncated answers on ordinary chat
prompts after the logic/reasoning SFT stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from polyvalent_lr.evaluation.metrics import ReasoningMetrics  # noqa: E402
from polyvalent_lr.reasoning_v3 import prompts  # noqa: E402
from scripts.bench.run_bench import _adapter_dir, _gen_batch, _load_model  # noqa: E402


def load_probes(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out_dir", default="results/bench_20261028")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import polyvalent_lr.config_v3 as cfg

    model_path = args.model or cfg.BASE_MODEL_PATH
    adapter = _adapter_dir(args.adapter or cfg.STAGE2_ADAPTER_DIR)
    prompts_path = ROOT / "scripts" / "bench" / "general_probes.jsonl"
    probes = load_probes(prompts_path)
    if args.limit:
        probes = probes[: args.limit]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "general_predictions.jsonl"
    summary_path = out_dir / "general_summary.json"

    model, tok = _load_model(model_path, adapter, args.device)
    gen_args = SimpleNamespace(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.1,
        temperature=0.6,
        top_p=0.95,
    )

    records: List[Dict[str, Any]] = []
    t_start = time.time()
    for start in range(0, len(probes), args.batch_size):
        batch = probes[start:start + args.batch_size]
        batch_prompts = [
            prompts.build_chat_prompt(prompts.SYSTEM_CHAT, p["prompt"]) for p in batch
        ]
        t0 = time.time()
        texts, lens, _ = _gen_batch(
            tok, model, batch_prompts, 1, False, gen_args, args.device
        )
        latency = time.time() - t0
        for p, group, glens in zip(batch, texts, lens):
            raw = group[0]
            n_tokens = int(glens[0])
            deg = ReasoningMetrics.detect_degeneracy(raw)
            # The generic detector flags any 5+ repeat of a 2..8-gram, which
            # fires on legitimate bulleted lists and step-by-step recipes.
            # For these probes a loop is only called when most tokens repeat.
            degenerate = float(deg["repetition_ratio"]) > 0.6
            stripped = raw.strip()
            rec = {
                "id": p["id"],
                "prompt": p["prompt"],
                "raw": raw,
                "n_output_tokens": n_tokens,
                "chars": len(stripped),
                "words": len(stripped.split()),
                "truncated": n_tokens >= args.max_new_tokens,
                "degenerate": degenerate,
                "loose_ngram_flag": bool(deg["is_degenerate"]),
                "max_ngram_repeat": deg["max_ngram_repeat"],
                "repetition_ratio": deg["repetition_ratio"],
                "empty": not stripped,
                "failed": (
                    not stripped
                    or len(stripped) < 20
                    or degenerate
                    or n_tokens >= args.max_new_tokens
                ),
                "latency_s": round(latency / max(1, len(batch)), 3),
            }
            records.append(rec)
        print(f"[general] {len(records)}/{len(probes)}", flush=True)

    with pred_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(records)
    summary = {
        "n": n,
        "failed": sum(r["failed"] for r in records),
        "empty": sum(r["empty"] for r in records),
        "degenerate": sum(r["degenerate"] for r in records),
        "loose_ngram_flags": sum(r["loose_ngram_flag"] for r in records),
        "truncated": sum(r["truncated"] for r in records),
        "short_lt20_chars": sum(1 for r in records if r["chars"] < 20),
        "mean_chars": round(sum(r["chars"] for r in records) / n, 1) if n else 0,
        "mean_words": round(sum(r["words"] for r in records) / n, 1) if n else 0,
        "mean_output_tokens": round(sum(r["n_output_tokens"] for r in records) / n, 1) if n else 0,
        "seconds": round(time.time() - t_start, 1),
        "model": model_path,
        "adapter": str(adapter) if adapter else None,
        "prompts_path": str(prompts_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[general] wrote {pred_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
