"""Build canonical benchmark items from the official released repositories.

Sources (cloned fresh under ``data/benchmarks_real/<name>/raw``):
  * LogicBench      https://github.com/Mihir3009/LogicBench
  * Multi-LogiEval  https://github.com/Mihir3009/Multi-LogiEval
  * LogicNLI        https://github.com/omnilabNLP/LogicNLI

Only the official evaluation/test splits are used.  Nothing is generated,
sampled, re-labelled, or repaired here; this script is a deterministic
transcription with an explicit source/commit in each README.

Outputs (one JSONL per dataset):
  data/benchmarks_real/LogicBench/items.jsonl
  data/benchmarks_real/Multi-LogiEval/items.jsonl
  data/benchmarks_real/LogicNLI/items.jsonl
  data/benchmarks_real/v3-holdout/items.jsonl
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from polyvalent_lr.reasoning_v3 import prompts  # noqa: E402

BENCH_ROOT = ROOT / "data" / "benchmarks_real"
RETRIEVED = date.today().isoformat()

SOURCES = {
    "LogicBench": {
        "url": "https://github.com/Mihir3009/LogicBench",
        "license": "MIT",
        "paper": "https://arxiv.org/abs/2404.15522",
    },
    "Multi-LogiEval": {
        "url": "https://github.com/Mihir3009/Multi-LogiEval",
        "license": "MIT",
        "paper": "https://arxiv.org/abs/2406.17169",
    },
    "LogicNLI": {
        "url": "https://github.com/omnilabNLP/LogicNLI",
        "license": "no explicit LICENSE file in the repository (cite the EMNLP 2021 paper)",
        "paper": "https://aclanthology.org/2021.emnlp-main.303/",
    },
    "v3-holdout": {
        "url": "local: data/v3/mix/holdout.jsonl",
        "license": "project-internal",
        "paper": "",
    },
}


def _load_json(path: Path) -> Any:
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


def _commit(name: str) -> str:
    p = BENCH_ROOT / name / "commit.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else "unknown"


def _write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _binary_label(answer: Any) -> Optional[str]:
    s = str(answer).strip().lower()
    if s in {"yes", "true", "entailed", "entailment", "valid", "proved"}:
        return "True"
    if s in {"no", "false", "contradicted", "contradiction", "invalid", "disproved"}:
        return "False"
    return None


BINARY_ALIASES = {
    "true": "True", "yes": "True", "entailed": "True", "entailment": "True",
    "valid": "True", "proved": "True",
    "false": "False", "no": "False", "contradicted": "False",
    "contradiction": "False", "invalid": "False", "disproved": "False",
    "uncertain": "Uncertain", "unknown": "Uncertain", "undetermined": "Uncertain",
}

NLI_ALIASES = {
    "entailment": "entailment", "entailed": "entailment", "entails": "entailment",
    "contradiction": "contradiction", "contradicted": "contradiction",
    "contradicts": "contradiction",
    "neutral": "neutral", "uncertain": "neutral", "unknown": "neutral",
    "self_contradiction": "self_contradiction",
    "self-contradiction": "self_contradiction",
    "self contradiction": "self_contradiction",
    "self_contradictory": "self_contradiction",
    "self-contradictory": "self_contradiction",
    "selfcontradiction": "self_contradiction",
}


def _binary_prompt(premises: List[str], question: str) -> str:
    """Exact user template used by all 514,769 Set E training examples.

    The v3 adapter was trained on the three-way wording (True/False/Uncertain).
    The old benchmark prompt said only "True or False" and appended an
    instruction the model never saw, which drove the all-True collapse in the
    first timing run.  Do not add or remove words here without rebuilding the
    training corpus too.
    """
    ptext = "\n".join(f"- {p}" for p in premises)
    user = (
        "Given the facts and rules (premises), decide whether the hypothesis is "
        "True, False, or Uncertain.\n\n"
        f"Premises:\n{ptext}\n\nHypothesis:\n{question}"
    )
    return prompts.build_chat_prompt(prompts.SYSTEM_LOGIC, user)


def _mcqa_prompt(premises: List[str], question: str, choices: Dict[str, str], keys: List[str]) -> str:
    ptext = "\n".join(f"- {p}" for p in premises)
    ctext = "\n".join(f"{k}: {choices[k]}" for k in keys)
    user = (
        "Given the facts and rules (premises), choose the conclusion that follows.\n\n"
        f"Premises:\n{ptext}\n\nQuestion:\n{question}\n\nOptions:\n{ctext}\n\n"
        f"Answer with exactly one option key: {', '.join(keys)}."
    )
    return prompts.build_chat_prompt(prompts.SYSTEM_LOGIC, user)


def _nli_prompt(premises: List[str], statement: str) -> str:
    ptext = "\n".join(f"- {p}" for p in premises)
    user = (
        "Given the facts and rules (premises), classify the relationship with the "
        "statement.\n\n"
        f"Premises:\n{ptext}\n\nStatement:\n{statement}\n\n"
        "Answer with exactly one of: entailment, contradiction, neutral, "
        "self_contradiction."
    )
    return prompts.build_chat_prompt(prompts.SYSTEM_LOGIC, user)


def build_logicbench() -> Dict[str, Any]:
    raw = BENCH_ROOT / "LogicBench" / "raw"
    eval_root = raw / "data" / "LogicBench(Eval)"
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for task in ("BQA", "MCQA"):
        for path in sorted((eval_root / task).glob("**/*.json")):
            rel = path.relative_to(eval_root / task).parts
            if len(rel) < 2:
                skipped.append(str(path))
                continue
            logic = rel[0]
            axiom = path.parent.name
            obj = _load_json(path)
            for sample in obj.get("samples", []):
                sid = str(sample.get("id"))
                context = str(sample.get("context", "")).strip()
                if not context:
                    skipped.append(f"{path}:{sid}:empty-context")
                    continue
                if task == "BQA":
                    for qi, qa in enumerate(sample.get("qa_pairs", [])):
                        gold = _binary_label(qa.get("answer"))
                        question = str(qa.get("question", "")).strip()
                        if gold is None or not question:
                            skipped.append(f"{path}:{sid}:{qi}:bad-qa")
                            continue
                        items.append({
                            "id": f"lb-bqa:{logic}:{axiom}:{sid}:{qi}",
                            "benchmark": "LogicBench-BQA",
                            "split": "BQA",
                            "pattern": axiom,
                            "domain": logic,
                            "task_type": "binary",
                            "premises": [context],
                            "question": question,
                            "choices": None,
                            "gold": gold,
                            "label_space": ["True", "False", "Uncertain"],
                            "aliases": BINARY_ALIASES,
                            "chance": 0.5,
                            "prompt": _binary_prompt([context], question),
                            "meta": {"sample_id": sid, "qa_index": qi},
                        })
                else:
                    choices = sample.get("choices") or {}
                    if not isinstance(choices, dict) or not choices:
                        skipped.append(f"{path}:{sid}:no-choices")
                        continue
                    keys = sorted(
                        choices.keys(),
                        key=lambda k: int(re.sub(r"\D", "", k) or 0),
                    )
                    gold = str(sample.get("answer", "")).strip()
                    if gold not in choices:
                        skipped.append(f"{path}:{sid}:gold-not-in-choices")
                        continue
                    question = str(sample.get("question", "")).strip()
                    aliases = {}
                    for i, key in enumerate(keys, start=1):
                        aliases[key.lower()] = key
                        aliases[str(i)] = key
                        aliases[chr(96 + i)] = key
                        aliases[f"option {i}"] = key
                        aliases[f"choice {i}"] = key
                    items.append({
                        "id": f"lb-mcqa:{logic}:{axiom}:{sid}",
                        "benchmark": "LogicBench-MCQA",
                        "split": "MCQA",
                        "pattern": axiom,
                        "domain": logic,
                        "task_type": "mcqa",
                        "premises": [context],
                        "question": question,
                        "choices": choices,
                        "gold": gold,
                        "label_space": keys,
                        "aliases": aliases,
                        "chance": round(1.0 / len(keys), 6),
                        "prompt": _mcqa_prompt([context], question, choices, keys),
                        "meta": {"sample_id": sid, "n_choices": len(keys)},
                    })

    out = BENCH_ROOT / "LogicBench" / "items.jsonl"
    _write_jsonl(out, items)
    _write_readme(
        "LogicBench",
        out,
        items,
        used="LogicBench(Eval)/BQA and LogicBench(Eval)/MCQA (all 25 inference rules).",
        excluded=(
            "LogicBench(Aug) was excluded: it is the synthetic training augmentation, "
            "not the human-verified evaluation set."
        ),
        extra=(
            "BQA answers are mapped yes->True and no->False and use the exact v3 "
            "training prompt wording (True/False/Uncertain). MCQA gold is the official "
            "``choice_N`` key; prompts list the original options and ask for that key. "
            "Some MCQA items have five options, so chance is 1/n_choices per item."
        ),
        skipped=skipped,
    )
    return {"items": len(items), "path": str(out), "skipped": len(skipped)}


def build_multi_logieval() -> Dict[str, Any]:
    raw = BENCH_ROOT / "Multi-LogiEval" / "raw" / "data"
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for path in sorted(raw.glob("**/*.json")):
        rel = path.relative_to(raw).parts
        folder = rel[0]
        obj = _load_json(path)
        logic = str(obj.get("logic") or (rel[1] if len(rel) > 1 else "unknown"))
        rule = str(obj.get("rule") or path.stem)
        depth = str(obj.get("depth") or ("multivariable" if "multivariable" in folder else folder))
        if "multivariable" in folder:
            logic = "multivar_fol"
        for sample in obj.get("samples", []):
            sid = str(sample.get("id"))
            context = str(sample.get("context", "")).strip()
            question = str(sample.get("question", "")).strip()
            gold = _binary_label(sample.get("answer"))
            if not context or not question or gold is None:
                skipped.append(f"{path}:{sid}:bad-sample")
                continue
            items.append({
                "id": f"mle:{folder}:{logic}:{path.stem}:{sid}",
                "benchmark": "Multi-LogiEval",
                "split": f"{folder}/{logic}",
                "pattern": rule,
                "domain": logic,
                "task_type": "binary",
                "premises": [context],
                "question": question,
                "choices": None,
                "gold": gold,
                "label_space": ["True", "False", "Uncertain"],
                "aliases": BINARY_ALIASES,
                "chance": 0.5,
                "prompt": _binary_prompt([context], question),
                "meta": {
                    "sample_id": sid, "depth": depth, "logic": logic, "rule": rule,
                    "folder": folder,
                },
            })

    out = BENCH_ROOT / "Multi-LogiEval" / "items.jsonl"
    _write_jsonl(out, items)
    _write_readme(
        "Multi-LogiEval",
        out,
        items,
        used=(
            "All shipped JSON files under d1_Data..d5_Data and "
            "multivariable_fol_Data (fol, pl, nm, multivar_fol)."
        ),
        excluded="Nothing excluded.",
        extra=(
            "Official answers are yes/no; mapped yes->True and no->False. The "
            "release is label-imbalanced (1212 yes / 484 no = 71.5% yes), which "
            "fails the 70% label-balance gate used by validate_data.py. Predictions "
            "use the v3 training label space (True/False/Uncertain); Uncertain is "
            "a parsed but wrong answer for this binary gold, never 'malformed'."
        ),
        skipped=skipped,
    )
    return {"items": len(items), "path": str(out), "skipped": len(skipped)}


def build_logicnli() -> Dict[str, Any]:
    zpath = BENCH_ROOT / "LogicNLI" / "raw" / "dataset" / "LogicNLI_sim.zip"
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []
    with zipfile.ZipFile(zpath) as zf:
        obj = json.loads(zf.read("LogicNLI_sim/test_language.json").decode("utf-8"))
        for ctx_key, ctx in obj.items():
            facts = ctx.get("facts")
            rules = ctx.get("rules")
            statements = ctx.get("statements")
            labels = ctx.get("labels")
            if not all(isinstance(x, list) for x in (facts, rules, statements, labels)):
                skipped.append(f"context {ctx_key}:unexpected-shape")
                continue
            if len(statements) != len(labels):
                skipped.append(f"context {ctx_key}:length-mismatch")
                continue
            premises = [str(x).strip() for x in (facts + rules) if str(x).strip()]
            for si, (statement, label) in enumerate(zip(statements, labels)):
                statement = str(statement).strip()
                label = str(label).strip().lower()
                if not statement or label not in NLI_ALIASES.values():
                    skipped.append(f"context {ctx_key} item {si}:bad-item")
                    continue
                items.append({
                    "id": f"nli:language:{ctx_key}:{si}",
                    "benchmark": "LogicNLI",
                    "split": "test_language",
                    "pattern": "nli",
                    "domain": "language",
                    "task_type": "nli4",
                    "premises": premises,
                    "question": statement,
                    "choices": None,
                    "gold": label,
                    "label_space": ["entailment", "contradiction", "neutral", "self_contradiction"],
                    "aliases": NLI_ALIASES,
                    "chance": 0.25,
                    "prompt": _nli_prompt(premises, statement),
                    "meta": {"context_index": ctx_key, "statement_index": si},
                })

    out = BENCH_ROOT / "LogicNLI" / "items.jsonl"
    _write_jsonl(out, items)
    _write_readme(
        "LogicNLI",
        out,
        items,
        used="LogicNLI_sim.zip -> test_language.json (the natural-language test split).",
        excluded=(
            "test_logic.json is included in the release but is symbolic "
            "(structured facts/rules), and the release ships no verbaliser. It was "
            "excluded rather than inventing a rendering; test_language is the "
            "official natural-language test split."
        ),
        extra=(
            "The real label set is four-way: entailment, contradiction, neutral, "
            "self_contradiction (500 each). It is not the three-way "
            "entailment/contradiction/neutral set that the old normalizer assumed."
        ),
        skipped=skipped,
    )
    return {"items": len(items), "path": str(out), "skipped": len(skipped)}


def build_holdout() -> Dict[str, Any]:
    src = ROOT / "data" / "v3" / "mix" / "holdout.jsonl"
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            category = str(row.get("category", "?"))
            prompt = str(row.get("prompt", ""))
            target = str(row.get("target", ""))
            meta = row.get("meta") or {}
            base = f"holdout:{category}:{_sha1(prompt)[:12]}"
            iid = base
            n = 1
            while iid in seen:
                n += 1
                iid = f"{base}-{n}"
            seen.add(iid)

            chance = None
            if category == "E":
                gold = str(meta.get("gold", "")).strip()
                task_type, label_space, aliases = "label", ["True", "False", "Uncertain"], BINARY_ALIASES
                chance = 1.0 / 3.0
            elif category == "A":
                gold = str(meta.get("answer_kind", "")).strip()
                task_type, label_space, aliases = "puzzle", ["unique", "insufficient"], {}
                chance = 0.5
            else:
                gold, task_type, label_space, aliases = "", "chat", [], {}

            items.append({
                "id": iid,
                "benchmark": "v3-Holdout",
                "split": category,
                "pattern": category,
                "domain": category,
                "task_type": task_type,
                "premises": [],
                "question": "",
                "choices": None,
                "gold": gold,
                "label_space": label_space,
                "aliases": aliases,
                "chance": chance,
                "prompt": prompt,
                "target": target,
                "meta": meta,
            })

    out = BENCH_ROOT / "v3-holdout" / "items.jsonl"
    _write_jsonl(out, items)
    _write_readme(
        "v3-holdout",
        out,
        items,
        used="data/v3/mix/holdout.jsonl (A/D/E, 300 each; in-distribution sanity check).",
        excluded="Not a public benchmark; it is the v3 training holdout.",
        extra=(
            "This split is expected to be in-distribution. Category A is scored with "
            "the dependency-free puzzle validator (exact table), E with label match, "
            "and D with token-F1 + degeneration diagnostics (not an accuracy claim)."
        ),
        skipped=[],
    )
    return {"items": len(items), "path": str(out), "skipped": 0}


def _write_readme(
    name: str,
    items_path: Path,
    items: List[Dict[str, Any]],
    used: str,
    excluded: str,
    extra: str,
    skipped: List[str],
) -> None:
    src = SOURCES[name]
    labels = Counter(i["gold"] for i in items if i.get("gold"))
    n = len(items)
    lines = [
        f"# {name} — official benchmark data",
        "",
        f"- Source: {src['url']}",
        f"- Commit: `{_commit(name) if name != 'v3-holdout' else 'local v3 holdout'}`",
        f"- Retrieved: {RETRIEVED}",
        f"- License: {src['license']}",
        f"- Paper: {src['paper']}",
        f"- Used: {used}",
        f"- Excluded: {excluded}",
        f"- Items: **{n}**",
        "",
        "## Label distribution",
        "",
        "| gold | count | share |",
        "|------|------:|------:|",
    ]
    for label, count in labels.most_common():
        lines.append(f"| {label} | {count} | {count / n:.1%} |" if n else f"| {label} | {count} | 0% |")
    lines += [
        "",
        "## Notes",
        "",
        extra,
        "",
        f"Rebuild with: `python scripts/bench/build_real_benchmarks.py` "
        f"(writes `{items_path.relative_to(ROOT)}`).",
        "",
    ]
    if skipped:
        lines += [
            "## Skipped records",
            "",
            f"{len(skipped)} record(s) were skipped because they did not match the "
            "documented release schema (see build log). First 20:",
            "",
        ]
        lines += [f"- `{s}`" for s in skipped[:20]]
        lines.append("")
    (items_path.parent / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"[bench] official data root: {BENCH_ROOT}")
    results = {
        "LogicBench": build_logicbench(),
        "Multi-LogiEval": build_multi_logieval(),
        "LogicNLI": build_logicnli(),
        "v3-holdout": build_holdout(),
    }
    print(json.dumps(results, indent=2))
    top = [
        "# Official benchmark data",
        "",
        f"Retrieved {RETRIEVED}. Raw repositories are under `<name>/raw`; canonical "
        "items are `<name>/items.jsonl`.",
        "",
        "| benchmark | items | commit | source |",
        "|-----------|------:|--------|--------|",
    ]
    for name, r in results.items():
        commit = _commit(name) if name != "v3-holdout" else "local"
        top.append(f"| {name} | {r['items']:,} | `{commit}` | {SOURCES.get(name, {}).get('url', 'local')} |")
    top += [
        "",
        "No synthetic or placeholder records are written by this pipeline. If a "
        "source file is missing, the build fails rather than substituting data.",
        "",
    ]
    (BENCH_ROOT / "README.md").write_text("\n".join(top), encoding="utf-8")
    print(f"[bench] wrote {BENCH_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
