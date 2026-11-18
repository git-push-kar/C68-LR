"""
Export script: Copies all relevant research papers, reports, benchmarks, scripts,
and model weights to E:\Polyvalent_LR_Handover_2026-11-18.
"""

import os
import shutil
import time
from pathlib import Path

DEST_ROOT = Path("E:/Polyvalent_LR_Handover_2026-11-18")
SRC_ROOT = Path("C:/Users/Admin/Desktop/C68-LR")


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"[FILE] {dst.relative_to(DEST_ROOT)} ({size_mb:.2f} MB)")


def copy_dir(src: Path, dst: Path, ignore=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)
    count = sum(1 for _ in dst.rglob("*") if _.is_file())
    size_mb = sum(_.stat().st_size for _ in dst.rglob("*") if _.is_file()) / (1024 * 1024)
    print(f"[DIR]  {dst.relative_to(DEST_ROOT)}: {count} files ({size_mb:.2f} MB)")


def main():
    start_time = time.time()
    print("=" * 70)
    print(f"Exporting Polyvalent Logical Reasoning assets to: {DEST_ROOT}")
    print("=" * 70)

    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Paper & Submission Documents
    print("\n--- 1. Paper & Submission Documents ---")
    copy_file(SRC_ROOT / "paper/draft.md", DEST_ROOT / "paper/draft.md")
    copy_file(SRC_ROOT / "paper/open_items.md", DEST_ROOT / "paper/open_items.md")
    copy_file(SRC_ROOT / "paper/FILE_INVENTORY.md", DEST_ROOT / "paper/FILE_INVENTORY.md")

    # 2. Integration Guides & Architectural Documentation
    print("\n--- 2. Integration & Architectural Documentation ---")
    copy_file(SRC_ROOT / "docs/ADAPTER_INTEGRATION_GUIDE.md", DEST_ROOT / "docs/ADAPTER_INTEGRATION_GUIDE.md")
    copy_file(SRC_ROOT / "Polyvalent_LR_Integration_Note.md", DEST_ROOT / "docs/Polyvalent_LR_Integration_Note.md")
    copy_file(SRC_ROOT / "DECISIONS.md", DEST_ROOT / "reports/DECISIONS.md")
    copy_file(SRC_ROOT / "PROGRESS.md", DEST_ROOT / "reports/PROGRESS.md")

    # 3. Reports & Audits
    print("\n--- 3. Experimental Reports & Forensic Audits ---")
    copy_file(SRC_ROOT / "results/gate_decision.md", DEST_ROOT / "reports/gate_decision.md")
    copy_file(SRC_ROOT / "results/perturb_examples.md", DEST_ROOT / "reports/perturb_examples.md")
    copy_file(SRC_ROOT / "results/contamination_report.json", DEST_ROOT / "reports/contamination_report.json")
    copy_file(SRC_ROOT / "results/historical_reproduction_audit.json", DEST_ROOT / "reports/historical_reproduction_audit.json")

    # 4. Aggregated Metric Summaries
    print("\n--- 4. Machine-Readable Evaluation Summaries ---")
    copy_file(SRC_ROOT / "results/summary_phase1.json", DEST_ROOT / "results/summary_phase1.json")
    copy_file(SRC_ROOT / "results/summary_phase2.json", DEST_ROOT / "results/summary_phase2.json")
    copy_file(SRC_ROOT / "results/summary_perturb.json", DEST_ROOT / "results/summary_perturb.json")

    # 5. Raw Inference Data (95 JSONL files)
    print("\n--- 5. Raw Evaluation Generations ---")
    copy_dir(SRC_ROOT / "results/raw", DEST_ROOT / "results/raw")

    # 6. Evaluation, Perturbation & Table Scripts
    print("\n--- 6. Pipeline Scripts & Tools ---")
    copy_file(SRC_ROOT / "scripts/run_all.py", DEST_ROOT / "scripts/run_all.py")
    copy_file(SRC_ROOT / "scripts/perturb.py", DEST_ROOT / "scripts/perturb.py")
    copy_file(SRC_ROOT / "scripts/make_tables.py", DEST_ROOT / "scripts/make_tables.py")
    copy_file(SRC_ROOT / "logs/run_all.log", DEST_ROOT / "logs/run_all.log")

    # 7. Core Package & Unit Tests
    print("\n--- 7. Python Library & Test Suite ---")
    copy_dir(SRC_ROOT / "polyvalent_lr", DEST_ROOT / "polyvalent_lr", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    copy_dir(SRC_ROOT / "tests", DEST_ROOT / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 8. Model Weights & Adapters
    print("\n--- 8. Model Weights & Adapters ---")
    # A. Active v3 SFT Adapter (Standalone PEFT Package)
    copy_dir(SRC_ROOT / "models_v3/adapter/adapter", DEST_ROOT / "model_weights/v3_sft_adapter")

    # B. Active v3 SFT Final Checkpoint (root files of sft/ without intermediate optimizer states)
    sft_dest = DEST_ROOT / "model_weights/v3_sft_checkpoint"
    sft_dest.mkdir(parents=True, exist_ok=True)
    sft_src = SRC_ROOT / "checkpoints/internvl3_2b_reasoning_v3/sft"
    for item in sft_src.iterdir():
        if item.is_file():
            copy_file(item, sft_dest / item.name)

    # C. Curriculum LoRA Adapters (Stages 1-3 & Final)
    copy_dir(
        SRC_ROOT / "checkpoints/internvl3_2b_lr_v2/stage_1_structured_deduction/adapter",
        DEST_ROOT / "model_weights/curriculum_stages/stage_1_structured_deduction",
        ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt")
    )
    copy_dir(
        SRC_ROOT / "checkpoints/internvl3_2b_lr_v2/stage_2_proof_and_nl_fol/adapter",
        DEST_ROOT / "model_weights/curriculum_stages/stage_2_proof_and_nl_fol",
        ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt")
    )
    copy_dir(
        SRC_ROOT / "checkpoints/internvl3_2b_lr_v2/stage_3_abductive_reasoning_optional/adapter",
        DEST_ROOT / "model_weights/curriculum_stages/stage_3_abductive_reasoning",
        ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt")
    )
    copy_dir(
        SRC_ROOT / "checkpoints/internvl3_2b_lr_v2/final_lr_lora",
        DEST_ROOT / "model_weights/curriculum_stages/final_lr_lora",
        ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt")
    )

    # D. Base Foundation Model Weights (InternVL3-2B)
    print("\n--- 9. Base Foundation Model (InternVL3-2B) ---")
    copy_dir(
        SRC_ROOT / "models/lr_adapter",
        DEST_ROOT / "model_weights/base_model_internvl3_2b",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )

    # 9. Top-Level Release README
    print("\n--- 10. Generating Release Overview Document ---")
    readme_content = f"""# Polyvalent: Logical Reasoning (LR) Master Release & Handover

**Export Date:** 2026-11-18  
**Destination:** `E:\\Polyvalent_LR_Handover_2026-11-18`  
**Base Architecture:** InternVL3-2B (`Qwen2-1.5B` causal LM + `InternViT-300M` vision transformer)  
**Active Reasoning Adapter:** `v3_sft` (Rank 64, Alpha 128, 73,859,072 parameters)  
**Git Origin:** `https://github.com/git-push-kar/C68-LR.git` (branch `master`, commit `9982f31`)  

---

## 1. Directory Structure

```
E:\\Polyvalent_LR_Handover_2026-11-18\\
├── README.md                              (This release overview)
├── paper\\
│   ├── draft.md                           (Full journal paper manuscript)
│   ├── open_items.md                      (Pre-submission open items)
│   └── FILE_INVENTORY.md                  (Comprehensive file & artifact manifest)
├── docs\\
│   ├── ADAPTER_INTEGRATION_GUIDE.md       (Production integration guide with Python engine)
│   └── Polyvalent_LR_Integration_Note.md  (Historical architectural integration note)
├── reports\\
│   ├── gate_decision.md                   (GRPO gate assessment report: FIX-FORMAT-FIRST)
│   ├── perturb_examples.md                (20 spot-checked examples per perturbation axis)
│   ├── contamination_report.json          (Exact & 8-gram train/test contamination audit)
│   ├── historical_reproduction_audit.json (Forensic audit of legacy 100% scores)
│   ├── DECISIONS.md                       (Engineering decisions 1-8)
│   └── PROGRESS.md                        (Step-by-step verified execution checklist)
├── results\\
│   ├── summary_phase1.json                (Phase 1 diagnostics: 6 models x 7 benchmarks)
│   ├── summary_phase2.json                (Phase 2 baselines & SC5 self-consistency)
│   ├── summary_perturb.json               (Phase 3 robustness under 9 perturbation types)
│   └── raw\\                               (95 raw generation JSONL logs, ~139 MB)
├── scripts\\
│   ├── run_all.py                         (Master evaluation runner with dynamic batching)
│   ├── perturb.py                         (6-axis perturbation generator)
│   └── make_tables.py                     (Automated table compiler)
├── logs\\
│   └── run_all.log                        (Complete 15,379s execution log on RTX A5000)
├── tests\\                                 (45 passing unit tests)
├── polyvalent_lr\\                         (Core Python library package)
└── model_weights\\
    ├── v3_sft_adapter\\                    (Active rank-64 adapter safetensors & config)
    ├── v3_sft_checkpoint\\                 (Final SFT checkpoint with tokenizer files)
    ├── curriculum_stages\\                 (Stages 1-3 & final_lr_lora rank-16 adapters)
    └── base_model_internvl3_2b\\           (Complete InternVL3-2B base model weights, 4.27 GB)
```

---

## 2. Key Scientific Findings & Numbers

1. **Self-Consistency (SC5, 2,000 Stratified Items):**
   - **Tuned (`v3_sft`): 47.50%** majority vote [95% CI: 45.35%, 49.65%] with **0.0%** parse failure.
   - **Base (InternVL3-2B): 28.95%** majority vote [95% CI: 26.95%, 30.95%] with **9.05%** parse failure (+18.55% gain for tuned).
2. **Noise / Distractor Resilience (3 Neutral Distractor Premises):**
   - **Tuned CoT:** **71.60%** accuracy.
   - **Base CoT:** Collapses to **30.80%** (a **+40.80%** resilience margin for tuned).
3. **GRPO Gate Assessment (1,500 Training Items x 8 Rollouts = 12,000 Completions):**
   - Parse failure: **0.29%** (PASS)
   - Mixed-result rate: **41.73%** (PASS)
   - Pass@1: **67.38%**, Pass@8: **87.33%**
   - Mean completion length: **13.9 tokens**; Reasoning presence (>=30 tokens): **1.69%** (FAIL)
   - **Verdict:** **`FIX-FORMAT-FIRST`** (mandates a brief rationale SFT step prior to RL).
4. **Adapter Parameter Count:** Exactly **73,859,072 parameters** (392 tensors across 28 layers, 7 modules, rank 64).
5. **Training Mixture:** Exactly **72,153 items** (Set A Puzzles: 20,000 / 27.7%, Set D Dolly: 13,759 / 19.1%, Set E FOL: 38,394 / 53.2%).

---

## 3. Quick-Start Loading Example

```python
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

base_path = "E:/Polyvalent_LR_Handover_2026-11-18/model_weights/base_model_internvl3_2b"
adapter_path = "E:/Polyvalent_LR_Handover_2026-11-18/model_weights/v3_sft_adapter"

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
base_model = AutoModel.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Polyvalent LR (v3_sft) successfully loaded from E: drive!")
```
"""
    (DEST_ROOT / "README.md").write_text(readme_content, encoding="utf-8")
    Path("E:/START_HERE_LR_RELEASE.md").write_text(readme_content, encoding="utf-8")
    print(f"[FILE] README.md written to {DEST_ROOT / 'README.md'}")
    print(f"[FILE] Summary copy written to E:/START_HERE_LR_RELEASE.md")

    # Compute total size of export
    total_files = sum(1 for _ in DEST_ROOT.rglob("*") if _.is_file())
    total_bytes = sum(_.stat().st_size for _ in DEST_ROOT.rglob("*") if _.is_file())
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"EXPORT COMPLETED SUCCESSFULLY!")
    print(f"Destination:  {DEST_ROOT}")
    print(f"Total Files:  {total_files:,}")
    print(f"Total Size:   {total_bytes / (1024**3):.2f} GB ({total_bytes / (1024**2):.2f} MB)")
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
