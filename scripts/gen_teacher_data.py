"""Stage 1 — Sets B & C via the Qwen teacher (runs under WSL2 + vLLM).

This script is intentionally NOT runnable in the Windows training venv: vLLM has
no native Windows support. Run it inside WSL2 Ubuntu (see scripts/setup_wsl_vllm.sh
and docs/v3/RUNBOOK.md).

Set B: existing logic datasets (ProofWriter/FOLIO/AbductionRules) -> the teacher
       writes a step-by-step rationale; keep only samples whose predicted final
       label matches ground truth (reasoning kept, final label forced to gold).
Set C: the teacher invents commonsense / implicit-goal questions, then a second
       pass keeps only those whose chosen answer states the hidden goal.

Use ``--dry_run`` to preview the constructed prompts and write them to disk
WITHOUT loading any model (safe to run anywhere, including this Windows host).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr import config_v3
from polyvalent_lr.reasoning_v3.data import InstructionExample, read_jsonl, write_jsonl
from polyvalent_lr.reasoning_v3 import prompts

REASON_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_SOURCES_B = {
    "ProofWriter": config_v3.V3_ROOT.parent / "processed" / "ProofWriter_train.jsonl",
    "FOLIO": config_v3.V3_ROOT.parent / "processed" / "FOLIO_train.jsonl",
    "AbductionRules": config_v3.V3_ROOT.parent / "processed" / "AbductionRules_train.jsonl",
}


# ------------------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------------------
def _problem_text(rec: Dict) -> str:
    premises = "\n".join(f"- {p}" for p in rec.get("premises", []))
    query = rec.get("hypothesis_or_query", "")
    if rec.get("source") == "AbductionRules":
        instruction = (
            "Given the observations and background rules, infer the most plausible "
            "missing fact that explains the observation."
        )
    else:
        instruction = (
            "Given the facts and rules (premises), decide whether the hypothesis is "
            "True, False, or Uncertain."
        )
    return f"{instruction}\n\nPremises:\n{premises}\n\nHypothesis:\n{query}"


def _gold_answer_text(rec: Dict) -> str:
    label = str(rec.get("label", "")).strip()
    if rec.get("source") == "AbductionRules":
        return label
    return label  # True / False / Uncertain


def build_b_records(limit: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    records: List[Dict] = []
    per_source = max(1, limit // len(_SOURCES_B))
    for source, path in _SOURCES_B.items():
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        rng.shuffle(rows)
        for rec in rows[: per_source * 3]:
            if not rec.get("hypothesis_or_query"):
                continue
            records.append(rec)
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break
    return records[:limit]


_C_TOPICS = [
    "a car wash that is 200 m away (do I walk or drive?)",
    "a car wash that is 100 m away (do I walk or drive?)",
    "buying a single item at a shop 300 m away",
    "fetching a package from a locker 500 m away",
    "getting a coffee from a kiosk 50 m away",
    "returning a library book 400 m away",
    "reaching a bus stop 150 m away to catch a bus in 2 minutes",
    "printing a page at a shop 250 m away",
    "a task that can be done online instead of travelling 200 m",
    "a chore that needs a tool stored at home 300 m away",
]


def _c_invent_prompt(rng: random.Random, n_prompts: int) -> List[Tuple[str, str]]:
    out = []
    for i in range(n_prompts):
        topic = _C_TOPICS[i % len(_C_TOPICS)]
        user = (
            f"Invent one short everyday decision question inspired by this scenario: {topic}. "
            "The question must hide an implicit goal that a careless reader would miss. "
            "Then give exactly two candidate answers (A and B) that both sound plausible, "
            "where only one respects the hidden goal. Respond in this exact format:\n"
            "QUESTION: <the question>\nANSWER_A: <first answer>\nANSWER_B: <second answer>"
        )
        out.append((topic, user))
    return out


def _c_verify_prompt(question: str, a: str, b: str) -> str:
    return (
        "Here is a question and two candidate answers. Decide which answer is correct, "
        "and state explicitly whether the correct one acknowledges the hidden goal. "
        "Give your reply as <reasoning>...</reasoning><answer>...</answer>.\n\n"
        f"QUESTION: {question}\nANSWER_A: {a}\nANSWER_B: {b}"
    )


# ------------------------------------------------------------------------------
# Label parsing / matching
# ------------------------------------------------------------------------------
def parse_reasoning(text: str) -> str:
    m = REASON_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def parse_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def label_matches(predicted: str, gold: str, is_abduction: bool) -> bool:
    p, g = _norm(predicted), _norm(gold)
    if not p or not g:
        return False
    if is_abduction:
        # gold is the missing fact; accept if its content words appear in the prediction
        g_tokens = [t for t in g.split() if len(t) > 2]
        return bool(g_tokens) and all(t in p for t in g_tokens)
    return p.split()[0] == g.split()[0] or g in p


# ------------------------------------------------------------------------------
# vLLM driver
# ------------------------------------------------------------------------------
def _load_vllm(cfg: config_v3.TeacherV3):
    try:
        from vllm import LLM  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "vLLM is not available in this environment.\n"
            "Run this script inside WSL2 Ubuntu after scripts/setup_wsl_vllm.sh.\n"
            f"(import error: {exc})"
        )
    from vllm import LLM
    return LLM(
        model=cfg.model,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        tensor_parallel_size=cfg.tensor_parallel_size,
        trust_remote_code=True,
    )


def _generate(llm, prompts_list: List[str], temperature: float, n: int, max_tokens: int):
    from vllm import SamplingParams
    sp = SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    return llm.generate(prompts_list, sp)


def gen_set_b(cfg: config_v3.TeacherV3, out_dir: Path, limit: int, holdout: int, seed: int) -> None:
    records = build_b_records(limit + holdout, seed)
    chat_prompts = [
        prompts.build_chat_prompt(prompts.SYSTEM_REASONING, _problem_text(r)) for r in records
    ]
    llm = _load_vllm(cfg)
    results = _generate(llm, chat_prompts, cfg.temperature, cfg.set_b_samples, cfg.max_model_len // 2)

    train: List[InstructionExample] = []
    hold: List[InstructionExample] = []
    for rec, res in zip(records, results):
        gold = _gold_answer_text(rec)
        is_abd = rec.get("source") == "AbductionRules"
        for cand in res.outputs:
            pred = parse_answer(cand.text)
            if not label_matches(pred, gold, is_abd):
                continue
            reasoning = parse_reasoning(cand.text)
            target = prompts.build_reasoning_target(reasoning, gold)
            cat = "B"
            ex = InstructionExample(
                prompt=prompts.build_chat_prompt(prompts.SYSTEM_REASONING, _problem_text(rec)),
                target=target, category=cat, task=("abduction" if is_abd else "deduction"),
                meta={"source": rec.get("source"), "gold": gold},
            )
            (hold if len(hold) < holdout else train).append(ex)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "holdout.jsonl", hold)
    print(f"[Set B] kept {len(train)} train + {len(hold)} holdout from {len(records)} prompts")


def gen_set_c(cfg: config_v3.TeacherV3, out_dir: Path, n_prompts: int, holdout: int, seed: int) -> None:
    rng = random.Random(seed)
    invented = _c_invent_prompt(rng, n_prompts + holdout)
    llm = _load_vllm(cfg)

    q_prompts = [prompts.build_chat_prompt(prompts.SYSTEM_CHAT, u) for _, u in invented]
    q_results = _generate(llm, q_prompts, 0.8, 1, 256)

    verify_prompts: List[str] = []
    parsed: List[Optional[Tuple[str, str, str]]] = []
    for res in q_results:
        text = res.outputs[0].text
        q = _field(text, "QUESTION")
        a = _field(text, "ANSWER_A")
        b = _field(text, "ANSWER_B")
        if not (q and a and b):
            parsed.append(None)
            verify_prompts.append("")
            continue
        parsed.append((q, a, b))
        verify_prompts.append(
            prompts.build_chat_prompt(prompts.SYSTEM_REASONING, _c_verify_prompt(q, a, b))
        )

    live = [p for p, pr in zip(verify_prompts, parsed) if pr is not None]
    v_results = _generate(llm, live, cfg.temperature, 1, cfg.max_model_len // 2)
    v_iter = iter(v_results)
    train: List[InstructionExample] = []
    hold: List[InstructionExample] = []
    for triple in parsed:
        if triple is None:
            continue
        res = next(v_iter)
        answer = parse_answer(res.outputs[0].text)
        reasoning = parse_reasoning(res.outputs[0].text)
        q = triple[0]
        if not answer or not reasoning:
            continue
        ex = InstructionExample(
            prompt=prompts.build_chat_prompt(prompts.SYSTEM_REASONING, q),
            target=prompts.build_reasoning_target(reasoning, answer),
            category="C", task="commonsense",
            meta={"gold": answer},
        )
        (hold if len(hold) < holdout else train).append(ex)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "holdout.jsonl", hold)
    print(f"[Set C] kept {len(train)} train + {len(hold)} holdout")


def _field(text: str, name: str) -> str:
    m = re.search(rf"{name}\s*:\s*(.+)", text or "")
    return m.group(1).strip() if m else ""


# ------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 teacher data (B/C) via vLLM.")
    ap.add_argument("--set", choices=["B", "C", "both"], default="both")
    ap.add_argument("--model", type=str, default=config_v3.TeacherV3().model)
    ap.add_argument("--limit", type=int, default=config_v3.TeacherV3().set_b_prompts)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--dry_run", action="store_true", help="Write prompt previews only (no model).")
    args = ap.parse_args()

    cfg = config_v3.TeacherV3(model=args.model)
    b_dir, c_dir = config_v3.DATA_B_DIR, config_v3.DATA_C_DIR

    if args.dry_run:
        recs = build_b_records(20, args.seed)
        preview = [prompts.build_chat_prompt(prompts.SYSTEM_REASONING, _problem_text(r)) for r in recs]
        preview += [prompts.build_chat_prompt(prompts.SYSTEM_CHAT, u) for _, u in _c_invent_prompt(random.Random(args.seed), 5)]
        b_dir.mkdir(parents=True, exist_ok=True)
        (b_dir / "_prompt_preview.txt").write_text("\n\n" + ("=" * 80 + "\n\n").join(preview), encoding="utf-8")
        print(f"[dry_run] wrote {len(preview)} prompt previews to {b_dir / '_prompt_preview.txt'}")
        return

    if args.set in ("B", "both"):
        gen_set_b(cfg, b_dir, args.limit, args.holdout, args.seed)
    if args.set in ("C", "both"):
        gen_set_c(cfg, c_dir, config_v3.TeacherV3().set_c_prompts, args.holdout, args.seed)


if __name__ == "__main__":
    main()
