"""Run the v3 model (base + adapter) on the official benchmarks.

Default: adapter only, greedy + 5-sample self-consistency, bf16, 384 new tokens.
The base model is NOT evaluated unless ``--include_base`` is passed.

Timing gate:
    python scripts/bench/run_bench.py --timing
runs 5 items from every benchmark (25 total), prints raw generations and an ETA,
and exits.  Use ``--full`` for the real run.

Predictions are written incrementally to ``results/bench_<date>/predictions.jsonl``
and the run is resumable with ``--resume``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Keep the Windows allocator workaround that the project loader applies.
from polyvalent_lr import config_v3  # noqa: E402
from polyvalent_lr.evaluation.bench_extract import parse_prediction  # noqa: E402
from polyvalent_lr.evaluation.bench_sampling import sample_plan  # noqa: E402
from polyvalent_lr.reasoning_v3 import decoding  # noqa: E402

BENCH_ROOT = ROOT / "data" / "benchmarks_real"
DEFAULT_BENCHMARKS = [
    "LogicBench-BQA", "LogicBench-MCQA", "Multi-LogiEval", "LogicNLI", "v3-Holdout",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_batch_seed(seed: int, batch: Sequence[Dict[str, Any]]) -> int:
    """Stable per-batch sampling seed, independent of resume boundaries."""
    key = "|".join(str(it["id"]) for it in batch)
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return (int(seed) + int(h[:8], 16)) % (2**31 - 1)


def _validation_gate(benchmarks: Sequence[str], groups: Dict[str, List[Dict[str, Any]]], allow: bool) -> Dict[str, Any]:
    """Refuse to run splits that failed the data validator unless explicitly allowed."""
    path = BENCH_ROOT / "validation.json"
    if not path.exists():
        raise SystemExit(
            "validation.json is missing; run scripts/bench/validate_data.py first."
        )
    validation = json.loads(path.read_text(encoding="utf-8"))
    recorded = validation.get("benchmarks", {})
    for name in benchmarks:
        n_recorded = (recorded.get(name) or {}).get("n")
        if n_recorded is not None and int(n_recorded) != len(groups.get(name, [])):
            raise SystemExit(
                f"validation.json is stale for {name}: recorded {n_recorded}, "
                f"items.jsonl has {len(groups.get(name, []))}. Re-run validate_data.py."
            )
    failures = validation.get("failures", {})
    blocked = [name for name in benchmarks if name in failures]
    if blocked:
        detail = "; ".join(f"{name}: {'; '.join(failures[name])}" for name in blocked)
        if not allow:
            raise SystemExit(
                "refusing to run splits that failed validate_data.py: " + detail +
                ". Pass --allow_failed_gates only if they are reported as diagnostics."
            )
        print("[run] WARNING: running diagnostic-only splits that failed data gates: " + detail)
    return validation


def _adapter_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return p
    if (p / "adapter" / "adapter_config.json").exists():
        return p / "adapter"
    raise SystemExit(f"no adapter_config.json under {p}")


def _load_groups(benchmarks: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(BENCH_ROOT.glob("*/items.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("benchmark") in benchmarks:
                    groups[item["benchmark"]].append(item)
    missing = [b for b in benchmarks if not groups.get(b)]
    if missing:
        raise SystemExit(f"missing benchmark items: {missing}")
    return dict(groups)


def _timing_items(plan: Dict[str, List[Dict[str, Any]]], k: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in sorted(plan):
        rows = plan[name]
        step = max(1, len(rows) // k)
        out.extend(rows[::step][:k])
    return out


def _read_done(path: Path, need_sc: bool) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("greedy") and (rec.get("sc5") or not need_sc):
                done[rec["id"]] = rec
    return done


def _compact(path: Path) -> Dict[str, Dict[str, Any]]:
    """Rewrite a predictions file keeping the last record per id."""
    if not path.exists():
        return {}
    keep: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            keep[rec["id"]] = rec
    with open(path, "w", encoding="utf-8") as fh:
        for iid in sorted(keep):
            fh.write(json.dumps(keep[iid], ensure_ascii=False) + "\n")
    return keep


def _load_model(base: str, adapter: Optional[Path], device: str):
    import torch
    from polyvalent_lr.models.loader import load_internvl_model_and_tokenizer

    model, tok = load_internvl_model_and_tokenizer(
        base,
        is_eval=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if device.startswith("cuda") else "cpu",
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.eval()
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def _generation_kwargs(tok, do_sample: bool, args) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
        "use_cache": True,
        "num_beams": 1,
        "stop_strings": ["</answer>", "<|im_end|>", "<|im_start|>"],
        "tokenizer": tok,
    }
    if do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    return kwargs


def _gen_batch(tok, model, prompts: List[str], n_samples: int, do_sample: bool, args, device: str):
    import torch

    repeated = [p for p in prompts for _ in range(n_samples)] if n_samples > 1 else list(prompts)
    enc = tok(repeated, return_tensors="pt", padding=True, truncation=False).to(device)
    kwargs = _generation_kwargs(tok, do_sample, args)
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    input_len = enc["input_ids"].shape[1]
    gen = out[:, input_len:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    lengths = gen.ne(tok.pad_token_id).sum(dim=1).tolist()
    if n_samples > 1:
        grouped = [texts[i * n_samples:(i + 1) * n_samples] for i in range(len(prompts))]
        glens = [lengths[i * n_samples:(i + 1) * n_samples] for i in range(len(prompts))]
        return grouped, glens, enc["input_ids"].shape[1]
    return [[t] for t in texts], [[n] for n in lengths], enc["input_ids"].shape[1]


def _sc_pick(item: Dict[str, Any], samples: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Return (selected_index, sc_label).  A uses table signatures; D is not scored."""
    task = item.get("task_type")
    if task == "puzzle":
        sigs = [decoding.table_signature(s) for s in samples]
        counts = Counter(s for s in sigs if s is not None)
        if counts:
            best, _ = counts.most_common(1)[0]
            for i, sig in enumerate(sigs):
                if sig == best:
                    return i, None
        return 0, None
    if task == "chat":
        return 0, None
    labels = [
        parse_prediction(s, item, allow_semantic_nli=False)["label"]
        for s in samples
    ]
    counts = Counter(l for l in labels if l is not None)
    if not counts:
        return 0, None
    best, _ = counts.most_common(1)[0]
    for i, label in enumerate(labels):
        if label == best:
            return i, best
    return 0, best


def _make_record(
    item: Dict[str, Any],
    greedy_raw: str,
    greedy_len: int,
    greedy_latency: float,
    sc_raw: List[str],
    sc_lens: List[int],
    sc_latency: float,
    input_tokens: int,
    args,
) -> Dict[str, Any]:
    g_strict = parse_prediction(greedy_raw, item, allow_semantic_nli=False)
    g_sem = parse_prediction(greedy_raw, item, allow_semantic_nli=True)
    sc_strict = [parse_prediction(s, item, allow_semantic_nli=False) for s in sc_raw]
    sc_sem = [parse_prediction(s, item, allow_semantic_nli=True) for s in sc_raw]
    selected, sc_label = _sc_pick(item, sc_raw)
    sc_label_sem = sc_sem[selected]["label"] if selected < len(sc_sem) else None
    return {
        "id": item["id"],
        "benchmark": item["benchmark"],
        "split": item.get("split"),
        "pattern": item.get("pattern"),
        "domain": item.get("domain"),
        "task_type": item.get("task_type"),
        "gold": item.get("gold", ""),
        "label_space": item.get("label_space", []),
        "chance": item.get("chance"),
        "prompt_chars": len(item.get("prompt", "")),
        "input_tokens": input_tokens,
        "greedy": {
            "raw": greedy_raw,
            "label": g_strict["label"],
            "label_semantic": g_sem["label"],
            "malformed": g_strict["malformed"],
            "malformed_semantic": g_sem["malformed"],
            "match_reason": g_strict["match_reason"],
            "latency_s": round(greedy_latency, 4),
            "n_output_tokens": greedy_len,
            "truncated": greedy_len >= args.max_new_tokens,
        },
        "sc5": {
            "n": len(sc_raw),
            "raw": sc_raw,
            "labels": [r["label"] for r in sc_strict],
            "labels_semantic": [r["label"] for r in sc_sem],
            "label": sc_label,
            "label_semantic": sc_label_sem,
            "selected_index": selected,
            "malformed": sc_label is None,
            "malformed_semantic": sc_label_sem is None,
            "latency_s": round(sc_latency, 4),
            "n_output_tokens": sc_lens,
            "truncated": [n >= args.max_new_tokens for n in sc_lens],
        },
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }


def _run_items(
    items: List[Dict[str, Any]],
    model,
    tok,
    args,
    device: str,
    out_path: Path,
    resume: bool,
) -> Dict[str, Any]:
    done = _compact(out_path) if resume else {}
    need_sc = not args.greedy_only
    done_ids = {
        iid for iid, rec in done.items()
        if rec.get("greedy") and (rec.get("sc5") or not need_sc)
    }
    todo = [it for it in items if it["id"] not in done_ids]
    print(f"[run] {len(items):,} planned, {len(done_ids):,} already done, {len(todo):,} to run")

    mode = "a" if resume else "w"
    with open(out_path, mode, encoding="utf-8") as fh:
        t_start = time.time()
        for bi in range(0, len(todo), args.batch_size):
            batch = todo[bi:bi + args.batch_size]
            prompts = [it["prompt"] for it in batch]
            input_tokens = [
                len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts
            ]
            n_prompts = len(prompts)

            g_texts: List[List[str]] = [[""] for _ in range(n_prompts)]
            g_lens: List[List[int]] = [[0] for _ in range(n_prompts)]
            g_time = 0.0
            if not args.sc_only:
                t0 = time.time()
                g_texts, g_lens, _ = _gen_batch(tok, model, prompts, 1, False, args, device)
                g_time = time.time() - t0

            sc_texts: List[List[str]] = [[None] * args.sc for _ in range(n_prompts)]
            sc_lens: List[List[int]] = [[0] * args.sc for _ in range(n_prompts)]
            sc_time = 0.0
            if not args.greedy_only:
                # Sample with a stable per-batch seed so --resume reproduces the
                # same SC samples for the same batch composition.
                import torch

                batch_seed = _stable_batch_seed(args.seed, batch)
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(batch_seed)
                t0 = time.time()
                sc_texts, sc_lens, _ = _gen_batch(tok, model, prompts, args.sc, True, args, device)
                sc_time = time.time() - t0

            for j, item in enumerate(batch):
                g_raw = g_texts[j][0] if g_texts[j] and g_texts[j][0] is not None else ""
                g_len = g_lens[j][0] if g_lens[j] else 0
                if need_sc:
                    s_raw = sc_texts[j]
                    s_len = sc_lens[j]
                else:
                    s_raw, s_len = [], []
                rec = _make_record(
                    item,
                    g_raw,
                    g_len,
                    g_time / max(1, n_prompts),
                    s_raw,
                    s_len,
                    sc_time / max(1, n_prompts),
                    input_tokens[j],
                    args,
                )
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done_n = min(bi + len(batch), len(todo))
            elapsed = time.time() - t_start
            rate = done_n / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - done_n) / rate if rate > 0 else 0.0
            print(
                f"[run] {done_n:,}/{len(todo):,} new items | "
                f"greedy {g_time:.1f}s sc5 {sc_time:.1f}s | "
                f"{rate:.2f} items/s | ETA {eta / 60:.1f} min",
                flush=True,
            )
    return {"done": len(todo), "seconds": time.time() - t_start}


def _environment(adapter: Optional[Path], args) -> Dict[str, Any]:
    import torch
    import transformers
    import peft
    import accelerate
    import tokenizers

    env: Dict[str, Any] = {
        "created_utc": _now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "model": args.model,
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": (
            _sha256(adapter / "adapter_model.safetensors")
            if adapter and (adapter / "adapter_model.safetensors").exists()
            else None
        ),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status": _git(["status", "--porcelain"]),
        "scripts_sha256": {
            p.name: _sha256(p)
            for p in sorted((ROOT / "scripts" / "bench").glob("*.py"))
        },
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "tokenizers": tokenizers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "decoding": {
            "greedy": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
            },
            "sc5": {
                "do_sample": True,
                "n": args.sc,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
            },
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
    }
    return env


def _sanity(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    from polyvalent_lr.evaluation.metrics import ReasoningMetrics

    per: Dict[str, Any] = {}
    for rec in records:
        name = rec["benchmark"]
        if name == "v3-Holdout":
            name = f"v3-Holdout/{rec.get('split')}"
        d = per.setdefault(
            name,
            {
                "n": 0, "empty": 0, "malformed": 0, "degenerate": 0,
                "labels": Counter(), "raws": set(),
            },
        )
        d["n"] += 1
        raw = rec["greedy"].get("raw", "")
        d["raws"].add(raw)
        if not raw.strip():
            d["empty"] += 1
        if rec["greedy"].get("malformed"):
            d["malformed"] += 1
        if ReasoningMetrics.detect_degeneracy(raw).get("is_degenerate"):
            d["degenerate"] += 1
        if rec["greedy"].get("label"):
            d["labels"][rec["greedy"]["label"]] += 1
    return {
        k: {
            "n": v["n"],
            "empty": v["empty"],
            "malformed": v["malformed"],
            "degenerate": v["degenerate"],
            "labels": dict(v["labels"]),
            "distinct_labels": len(v["labels"]),
            "unique_generations": len(v["raws"]),
        }
        for k, v in per.items()
    }


LABEL_BENCHMARKS = {
    "LogicBench-BQA", "LogicBench-MCQA", "Multi-LogiEval",
    "LogicNLI", "v3-Holdout/E",
}


def _sanity_pass(sanity: Dict[str, Any]) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    for name, s in sanity.items():
        n = max(1, s["n"])
        if s["empty"] / n > 0.20:
            problems.append(f"{name}: {s['empty']}/{n} empty generations")
        if s["degenerate"] / n > 0.20:
            problems.append(f"{name}: {s['degenerate']}/{n} degenerate generations")
        if name in LABEL_BENCHMARKS:
            if not s["labels"]:
                problems.append(f"{name}: no greedy label extracted")
            if s["malformed"] / n > 0.20:
                problems.append(
                    f"{name}: {s['malformed']}/{n} malformed label outputs (>20%)"
                )
    return (not problems), problems


def _sanity_warnings(sanity: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    for name, s in sanity.items():
        if name not in LABEL_BENCHMARKS:
            continue
        if s["n"] >= 5 and s["distinct_labels"] == 1:
            only = next(iter(s["labels"]), None)
            warnings.append(
                f"{name}: all {s['n']} greedy predictions are {only!r} "
                f"(unique generations={s['unique_generations']})"
            )
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v3 on official benchmarks.")
    ap.add_argument("--model", default=config_v3.BASE_MODEL_PATH)
    ap.add_argument("--adapter", default=str(config_v3.STAGE2_ADAPTER_DIR))
    ap.add_argument("--no_adapter", action="store_true")
    ap.add_argument("--include_base", action="store_true",
                    help="Also run the base model (off by default; never a headline claim).")
    ap.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--sample", type=int, default=600,
                    help="Items per benchmark when n > 1000 (minimum 500).")
    ap.add_argument("--sample_all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=config_v3.DecodingV3().temperature)
    ap.add_argument("--top_p", type=float, default=config_v3.DecodingV3().top_p)
    ap.add_argument("--repetition_penalty", type=float, default=config_v3.DecodingV3().repetition_penalty)
    ap.add_argument("--sc", type=int, default=config_v3.DecodingV3().self_consistency)
    ap.add_argument("--greedy_only", action="store_true")
    ap.add_argument("--sc_only", action="store_true")
    ap.add_argument("--timing", action="store_true",
                    help="Run 5 items/benchmark, print raw outputs + ETA, then exit.")
    ap.add_argument("--full", action="store_true", help="Run the full plan.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--allow_failed_gates", action="store_true",
        help="Run splits that failed validate_data.py; they must be reported as diagnostics.",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.timing and not args.full:
        raise SystemExit("Pass --timing for the 25-item gate or --full for the real run.")

    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").lower().find("expandable_segments:true") >= 0:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    groups = _load_groups(benchmarks)
    _validation_gate(benchmarks, groups, args.allow_failed_gates)
    plan = groups if args.sample_all else sample_plan(groups, n_over_1000=args.sample, seed=args.seed)
    if args.limit:
        plan = {k: v[:args.limit] for k, v in plan.items()}

    date = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"bench_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = None if args.no_adapter else _adapter_dir(args.adapter)
    env = _environment(adapter, args)
    (out_dir / "run_config.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "sample_ids.json").write_text(
        json.dumps({k: [it["id"] for it in v] for k, v in plan.items()}, indent=2),
        encoding="utf-8",
    )

    print(f"[run] out_dir = {out_dir}")
    print(f"[run] adapter = {adapter}")
    print(f"[run] plan = " + ", ".join(f"{k}:{len(v)}" for k, v in plan.items()))

    model, tok = _load_model(args.model, adapter, args.device)
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.timing:
        items = _timing_items(plan, 5)
        path = out_dir / "timing_predictions.jsonl"
        if path.exists():
            path.unlink()
        t0 = time.time()
        _run_items(items, model, tok, args, args.device, path, resume=False)
        elapsed = time.time() - t0
        records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        total_plan = sum(len(v) for v in plan.values())
        eta = elapsed / len(records) * total_plan if records else 0.0
        sanity = _sanity(records)
        ok, problems = _sanity_pass(sanity)
        warnings = _sanity_warnings(sanity)
        payload = {
            "n_timing": len(records),
            "n_plan": total_plan,
            "seconds": round(elapsed, 2),
            "seconds_per_item": round(elapsed / max(1, len(records)), 3),
            "eta_full_seconds": round(eta, 1),
            "eta_full_hours": round(eta / 3600, 2),
            "batch_size": args.batch_size,
            "sanity": sanity,
            "sanity_pass": ok,
            "sanity_problems": problems,
            "sanity_warnings": warnings,
            "created_utc": _now(),
        }
        (out_dir / "timing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("\n" + "=" * 78)
        print("TIMING / RAW SANITY (greedy, first 5 per benchmark)")
        print("=" * 78)
        by_bench: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            by_bench[rec["benchmark"]].append(rec)
        for name in benchmarks:
            print(f"\n### {name}")
            for rec in by_bench.get(name, [])[:5]:
                raw = rec["greedy"]["raw"].replace("\n", " ")[:280]
                print(
                    f"  {rec['id']}: gold={rec.get('gold')!r} "
                    f"pred={rec['greedy'].get('label')!r} "
                    f"sem={rec['greedy'].get('label_semantic')!r} "
                    f"tokens={rec['greedy'].get('n_output_tokens')} "
                    f"raw={raw!r}"
                )
        print("\n" + "=" * 78)
        print(json.dumps(payload, indent=2))
        print(f"[timing] wrote {out_dir / 'timing.json'}")
        if warnings:
            print("[timing] SANITY WARNINGS (model behavior, not harness failure):")
            for w in warnings:
                print(f"  ~~ {w}")
        if not ok:
            print("[timing] SANITY FAILED:")
            for p in problems:
                print(f"  !! {p}")
            print("[timing] Fix the pipeline before a full run (exit 2).")
            raise SystemExit(2)
        print("[timing] SANITY PASSED.")
        print("[timing] STOPPING before the full run. Re-run with --full after review.")
        return

    predictions = out_dir / "predictions.jsonl"
    if args.resume and predictions.exists():
        _compact(predictions)
    items = [it for name in benchmarks for it in plan.get(name, [])]
    _run_items(items, model, tok, args, args.device, predictions, resume=args.resume)

    if args.include_base:
        print("[run] --include_base requested; running base model too")
        del model
        import torch
        torch.cuda.empty_cache()
        base_model, base_tok = _load_model(args.model, None, args.device)
        base_path = out_dir / "predictions_base.jsonl"
        _run_items(items, base_model, base_tok, args, args.device, base_path, resume=args.resume)
        del base_model
        torch.cuda.empty_cache()

    (out_dir / "DONE").write_text(_now(), encoding="utf-8")
    print(f"[run] wrote {predictions}")
    print(f"[run] done at {_now()}")


if __name__ == "__main__":
    main()
