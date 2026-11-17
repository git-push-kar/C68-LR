"""Generates Markdown and LaTeX tables directly from results/*.json for paper/draft.md."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load_json(name: str) -> Dict[str, Any]:
    path = RESULTS_DIR / name
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def make_main_results_table(p1: Dict[str, Any], p2: Dict[str, Any]) -> str:
    """Builds the main benchmark comparison table across in-dist and unseen sets."""
    headers = [
        "Model Configuration",
        "ProofWriter (In)",
        "FOLIO (In)",
        "Holdout-E (In)",
        "LB-BQA (Unseen)",
        "LB-MCQA (Unseen)",
        "LogicNLI (Unseen)",
        "Multi-LogiEval (Diag)",
        "In-Dist Mean",
        "Unseen Mean",
        "Gen. Gap",
    ]

    rows = []

    def get_acc(m_dict: Dict[str, Any], bench: str) -> str:
        if bench in m_dict and "accuracy" in m_dict[bench]:
            acc = m_dict[bench]["accuracy"] * 100.0
            return f"{acc:.1f}%"
        return "TBD"

    models_p1 = p1.get("models", {})
    gaps_p1 = p1.get("generalization_gaps", {})
    base_modes = p2.get("baselines", {}).get("base", {})
    tuned_modes = p2.get("baselines", {}).get("v3_sft", {})
    sc_data = p2.get("self_consistency", {})

    # Row 1: Base Direct
    if "base" in models_p1:
        bm = models_p1["base"]
        gap = gaps_p1.get("base", {})
        rows.append([
            "Base (InternVL3-2B) Direct",
            get_acc(bm, "ProofWriter_val"),
            get_acc(bm, "FOLIO_val"),
            get_acc(bm, "v3-holdout"),
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            f"{gap.get('in_dist_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('unseen_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('gap', 0.0)*100:+.1f}%" if gap else "TBD",
        ])

    # Row 2: Base CoT
    if "cot" in base_modes:
        bm = base_modes["cot"]
        rows.append([
            "Base (InternVL3-2B) CoT",
            "-", "-", "-",
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            "-", "-", "-",
        ])

    # Row 3: Base 3-Shot Direct
    if "few_shot" in base_modes:
        bm = base_modes["few_shot"]
        rows.append([
            "Base (InternVL3-2B) 3-Shot Direct",
            "-", "-", "-",
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            "-", "-", "-",
        ])

    # Row 4: Stage 1 (ProofWriter)
    if "stage1" in models_p1:
        bm = models_p1["stage1"]
        gap = gaps_p1.get("stage1", {})
        rows.append([
            "Stage 1 (Structured Deduction)",
            get_acc(bm, "ProofWriter_val"),
            get_acc(bm, "FOLIO_val"),
            get_acc(bm, "v3-holdout"),
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            f"{gap.get('in_dist_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('unseen_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('gap', 0.0)*100:+.1f}%" if gap else "TBD",
        ])

    # Row 5: Stage 2 (FOL / P-FOLIO)
    if "stage2" in models_p1:
        bm = models_p1["stage2"]
        gap = gaps_p1.get("stage2", {})
        rows.append([
            "Stage 2 (Proof & NL-FOL)",
            get_acc(bm, "ProofWriter_val"),
            get_acc(bm, "FOLIO_val"),
            get_acc(bm, "v3-holdout"),
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            f"{gap.get('in_dist_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('unseen_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('gap', 0.0)*100:+.1f}%" if gap else "TBD",
        ])

    # Row 6: Stage 3 (Abductive Reasoning)
    if "stage3" in models_p1:
        bm = models_p1["stage3"]
        gap = gaps_p1.get("stage3", {})
        rows.append([
            "Stage 3 (Abductive Optional)",
            get_acc(bm, "ProofWriter_val"),
            get_acc(bm, "FOLIO_val"),
            get_acc(bm, "v3-holdout"),
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            f"{gap.get('in_dist_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('unseen_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('gap', 0.0)*100:+.1f}%" if gap else "TBD",
        ])

    # Row 7: Final v2 Adapter
    if "final_lr_lora" in models_p1:
        bm = models_p1["final_lr_lora"]
        gap = gaps_p1.get("final_lr_lora", {})
        rows.append([
            "Final LR LoRA (v2 Curriculum)",
            get_acc(bm, "ProofWriter_val"),
            get_acc(bm, "FOLIO_val"),
            get_acc(bm, "v3-holdout"),
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            f"{gap.get('in_dist_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('unseen_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('gap', 0.0)*100:+.1f}%" if gap else "TBD",
        ])

    # Row 8: Latest Trained Model (v3 SFT Adapter) Direct
    if "v3_sft" in models_p1:
        bm = models_p1["v3_sft"]
        gap = gaps_p1.get("v3_sft", {})
        rows.append([
            "Latest Trained (v3 SFT) Direct",
            get_acc(bm, "ProofWriter_val"),
            get_acc(bm, "FOLIO_val"),
            get_acc(bm, "v3-holdout"),
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            f"{gap.get('in_dist_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('unseen_mean_acc', 0.0)*100:.1f}%" if gap else "TBD",
            f"{gap.get('gap', 0.0)*100:+.1f}%" if gap else "TBD",
        ])

    # Row 9: Latest Trained Model (v3 SFT) CoT
    if "cot" in tuned_modes:
        bm = tuned_modes["cot"]
        rows.append([
            "Latest Trained (v3 SFT) CoT",
            "-", "-", "-",
            get_acc(bm, "LogicBench-BQA"),
            get_acc(bm, "LogicBench-MCQA"),
            get_acc(bm, "LogicNLI"),
            get_acc(bm, "Multi-LogiEval"),
            "-", "-", "-",
        ])

    # Row 10: Latest + Self-Consistency (SC5)
    if "v3_sft" in sc_data:
        sc_m = sc_data["v3_sft"]
        sc_acc = sc_m.get("majority_vote_accuracy", 0.0) * 100.0
        rows.append([
            "Latest + Self-Consistency (SC5)",
            "-", "-", "-",
            f"{sc_acc:.1f}% (2k Subset)",
            "-", "-", "-", "-", "-", "-",
        ])

    # Row 11: Multimodal Regression Probes
    rows.append([
        "Multimodal Regression (MMBench / MMStar)",
        "TBD", "TBD", "TBD", "TBD", "TBD", "TBD", "TBD", "TBD", "TBD", "TBD",
    ])

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def make_robustness_table(p3: Dict[str, Any]) -> str:
    headers = [
        "Perturbation Mode",
        "Evaluated (n)",
        "Base Direct Acc",
        "Base CoT Acc",
        "Tuned Direct Acc",
        "Tuned CoT Acc",
        "Tuned Acc Drop",
        "Tuned Consistency",
    ]
    types_data = p3.get("types", {})
    rows = []
    for p_type, d in types_data.items():
        n = str(d.get("n", 0))
        b_dir = f"{d.get('base_direct_acc', 0.0)*100:.1f}%"
        b_cot = f"{d.get('base_cot_acc', 0.0)*100:.1f}%"
        t_dir = f"{d.get('tuned_direct_acc', 0.0)*100:.1f}%"
        t_cot = f"{d.get('tuned_cot_acc', 0.0)*100:.1f}%"
        t_drop = f"{d.get('tuned_drop', 0.0)*100:.1f}%"
        t_cons = f"{d.get('tuned_consistency', 0.0)*100:.1f}%"
        rows.append([p_type, n, b_dir, b_cot, t_dir, t_cot, t_drop, t_cons])

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def make_depth_table(p1: Dict[str, Any]) -> str:
    headers = [
        "Benchmark / Depth Slice",
        "Items (n)",
        "Base Acc [95% CI]",
        "Stage 1 Acc",
        "Stage 2 Acc",
        "Stage 3 Acc",
        "Tuned (v3 SFT) Acc [95% CI]",
    ]
    models = p1.get("models", {})
    b_res = models.get("base", {}).get("Multi-LogiEval", {}).get("depth_breakdown", {})
    t_res = models.get("v3_sft", {}).get("Multi-LogiEval", {}).get("depth_breakdown", {})
    s1_res = models.get("stage1", {}).get("Multi-LogiEval", {}).get("depth_breakdown", {})
    s2_res = models.get("stage2", {}).get("Multi-LogiEval", {}).get("depth_breakdown", {})
    s3_res = models.get("stage3", {}).get("Multi-LogiEval", {}).get("depth_breakdown", {})

    rows = []
    all_depths = sorted(set(b_res.keys()) | set(t_res.keys()))
    for d in all_depths:
        bd = b_res.get(d, {})
        td = t_res.get(d, {})
        n = str(bd.get("n", td.get("n", 0)))
        b_acc = f"{bd.get('accuracy', 0.0)*100:.1f}%"
        s1_acc = f"{s1_res.get(d, {}).get('accuracy', 0.0)*100:.1f}%" if d in s1_res else "-"
        s2_acc = f"{s2_res.get(d, {}).get('accuracy', 0.0)*100:.1f}%" if d in s2_res else "-"
        s3_acc = f"{s3_res.get(d, {}).get('accuracy', 0.0)*100:.1f}%" if d in s3_res else "-"
        t_acc = f"{td.get('accuracy', 0.0)*100:.1f}%"
        rows.append([f"Multi-LogiEval {d}", n, b_acc, s1_acc, s2_acc, s3_acc, t_acc])

    # ProofWriter depths
    pw_b = models.get("base", {}).get("ProofWriter_val", {}).get("depth_breakdown", {})
    pw_t = models.get("v3_sft", {}).get("ProofWriter_val", {}).get("depth_breakdown", {})
    pw_s1 = models.get("stage1", {}).get("ProofWriter_val", {}).get("depth_breakdown", {})
    pw_s2 = models.get("stage2", {}).get("ProofWriter_val", {}).get("depth_breakdown", {})
    pw_s3 = models.get("stage3", {}).get("ProofWriter_val", {}).get("depth_breakdown", {})

    for d in sorted(set(pw_b.keys()) | set(pw_t.keys())):
        bd = pw_b.get(d, {})
        td = pw_t.get(d, {})
        n = str(bd.get("n", td.get("n", 0)))
        b_acc = f"{bd.get('accuracy', 0.0)*100:.1f}%"
        s1_acc = f"{pw_s1.get(d, {}).get('accuracy', 0.0)*100:.1f}%" if d in pw_s1 else "-"
        s2_acc = f"{pw_s2.get(d, {}).get('accuracy', 0.0)*100:.1f}%" if d in pw_s2 else "-"
        s3_acc = f"{pw_s3.get(d, {}).get('accuracy', 0.0)*100:.1f}%" if d in pw_s3 else "-"
        t_acc = f"{td.get('accuracy', 0.0)*100:.1f}%"
        rows.append([f"ProofWriter depth {d}", n, b_acc, s1_acc, s2_acc, s3_acc, t_acc])

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def make_label_distribution_table(p1: Dict[str, Any]) -> str:
    headers = [
        "Model",
        "Benchmark",
        "Parse Fail %",
        "Pred Distribution",
        "Gold Distribution",
        "Max Pred Share",
        "Label Prior Flag",
    ]
    models = p1.get("models", {})
    rows = []
    for m_key, m_benches in models.items():
        for b_name, b_m in m_benches.items():
            pf = f"{b_m.get('parse_failure_rate', 0.0)*100:.1f}%"
            p_dist = str(b_m.get("pred_distribution", {}))
            g_dist = str(b_m.get("gold_distribution", {}))
            mp = f"{b_m.get('max_pred_share', 0.0)*100:.1f}%"
            flag = "YES (PRIOR COLLAPSE)" if b_m.get("label_prior_flag") else "NO"
            rows.append([m_key, b_name, pf, p_dist, g_dist, mp, flag])

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def update_paper_draft(p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any]) -> None:
    """Injects generated markdown tables directly into paper/draft.md."""
    draft_path = ROOT / "paper" / "draft.md"
    if not draft_path.exists():
        print(f"[Warning] {draft_path} does not exist.")
        return

    with open(draft_path, "r", encoding="utf-8") as f:
        content = f.read()

    main_tbl = make_main_results_table(p1, p2)
    depth_tbl = make_depth_table(p1)
    label_tbl = make_label_distribution_table(p1)
    robust_tbl = make_robustness_table(p3)

    import re
    content = re.sub(
        r"```markdown\s*<!-- TABLE_MAIN_RESULTS_PLACEHOLDER -->\s*```",
        main_tbl,
        content
    )
    content = re.sub(
        r"```markdown\s*<!-- TABLE_DEPTH_BREAKDOWN_PLACEHOLDER -->\s*```",
        depth_tbl,
        content
    )
    content = re.sub(
        r"```markdown\s*<!-- TABLE_LABEL_DISTRIBUTION_PLACEHOLDER -->\s*```",
        label_tbl,
        content
    )
    content = re.sub(
        r"```markdown\s*<!-- TABLE_ROBUSTNESS_PLACEHOLDER -->\s*```",
        robust_tbl,
        content
    )

    with open(draft_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[make_tables] Successfully updated {draft_path} with computed tables.")


def main():
    p1 = load_json("summary_phase1.json")
    p2 = load_json("summary_phase2.json")
    p3 = load_json("summary_perturb.json")

    print("=== MAIN BENCHMARK RESULTS TABLE ===")
    print(make_main_results_table(p1, p2))
    print("\n=== ROBUSTNESS TABLE ===")
    print(make_robustness_table(p3))
    print("\n=== DEPTH REASONING TABLE ===")
    print(make_depth_table(p1))
    print("\n=== PARSE FAILURE & LABEL DISTRIBUTION TABLE ===")
    print(make_label_distribution_table(p1))

    update_paper_draft(p1, p2, p3)


if __name__ == "__main__":
    main()
