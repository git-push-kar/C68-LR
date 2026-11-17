# Task Execution Progress

## Overall Status: COMPLETED

### Step 1: Environment, Inspection & Batching Infrastructure Setup
- [x] Read repository files, architecture notes, and existing evaluation scripts.
- [x] Inspect existing checkpoints (`checkpoints/internvl3_2b_lr_v2/` stages 1-3 + final, `checkpoints/internvl3_2b_reasoning_v3/sft`, `models/`, `models_v3/`).
- [x] Audit historical evaluation files (`evaluation_results_full.json`, `results_baseline.json`) and explain why they don't reproduce on canonical datasets.
- [x] Profile GPU memory and batched generation (Python 3.12, PyTorch 2.6.0+cu124, RTX A5000 24GB).
- [x] Verify text-only generation speed: batch size 64 direct eval (~40.8 items/sec, 4.05 GB VRAM), batch size 64 CoT (~8.7 items/sec, 4.27 GB VRAM).
- [x] Create `DECISIONS.md` documenting all architectural, batching, and methodological choices.
- [x] Create `scripts/perturb.py`, `scripts/make_tables.py`, and unit tests (`tests/test_perturbation.py`, `tests/test_parser_advanced.py`) — all 45 tests pass.
- [x] Run contamination check and save `results/contamination_report.json` and `results/historical_reproduction_audit.json`.

### Step 2: Phase 1 - Diagnostics
- [x] Completed `scripts/run_all.py --phase 1 --batch_size 64`:
  - Evaluated 6 models: `base`, `stage1`, `stage2`, `stage3`, `final_lr_lora`, `v3_sft`.
  - Evaluated 7 benchmarks: `LogicBench-BQA`, `LogicBench-MCQA`, `Multi-LogiEval`, `LogicNLI`, `ProofWriter_val`, `FOLIO_val`, `v3-holdout`.
  - Computed accuracy, macro-F1, per-class P/R, confusion matrices, parse-failure rates, predicted vs gold distributions, depth breakdown, in-distribution vs unseen gaps, 1000-resample bootstrap 95% CIs, and McNemar tests against base.
  - Verified and saved all 42 raw files to `results/raw/*.jsonl` and metrics to `results/summary_phase1.json`.

### Step 3: Phase 2 - Baselines & Self-Consistency
- [x] Base Direct (cached: BQA 70.33%, MCQA 76.0%, Multi-LogiEval 51.83%, LogicNLI 28.6%), CoT (cached: BQA 32.3%, MCQA 38.4%, Multi-LogiEval 24.35%, LogicNLI 0.65%), Few-Shot 3-shot Direct (cached: BQA 64.54%, MCQA 73.6%, Multi-LogiEval 41.27%, LogicNLI 28.3%).
- [x] Tuned Direct (cached: BQA 53.62%, MCQA 75.4%, Multi-LogiEval 60.02%, LogicNLI 25.15%), Tuned CoT (completed: BQA 40.72%, MCQA 72.60%, Multi-LogiEval 51.65%, LogicNLI 30.10%).
- [x] Tuned SC5 Self-Consistency (completed: **47.50%** majority vote [CI: 45.35%, 49.65%], **45.40%** single sample [CI: 43.10%, 47.45%], parse failure: 0.0%).
- [x] Base SC5 Self-Consistency (completed: **28.95%** majority vote [CI: 26.95%, 30.95%], **30.20%** single sample [CI: 28.20%, 32.25%], parse failure: 9.05%).
- [x] Output finalized: [summary_phase2.json](file:///c:/Users/Admin/Desktop/C68-LR/results/summary_phase2.json).

### Step 4: Phase 3 - Perturbation and Robustness Suite
- [x] Generated 20 spot-checked examples per perturbation type: saved to [perturb_examples.md](file:///c:/Users/Admin/Desktop/C68-LR/results/perturb_examples.md).
- [x] Evaluated 9 perturbation types across 250 shared-correct targets for Base and Tuned.
- [x] Output finalized: [summary_perturb.json](file:///c:/Users/Admin/Desktop/C68-LR/results/summary_perturb.json).

### Step 5: Research Paper Draft & Automated Table Generator
- [x] Ran `scripts/make_tables.py`: all markdown tables generated from computed numbers and automatically injected into [paper/draft.md](file:///c:/Users/Admin/Desktop/C68-LR/paper/draft.md).
- [x] Documented open submission items in [paper/open_items.md](file:///c:/Users/Admin/Desktop/C68-LR/paper/open_items.md).

### Step 6: GRPO Gate Analysis & Verdict
- [x] Evaluated 1,500 training problems x 8 CoT rollouts @ $T=0.7$ on `v3_sft` (ProofWriter: 1000, FOLIO: 500) with early stopping.
- [x] Computed metrics: Parse failure 0.29%, Mixed-result fraction 41.73%, Pass@1 67.38%, Pass@8 87.33%, Reasoning presence 1.69%.
- [x] Gate verdict: **FIX-FORMAT-FIRST** (saved to [gate_decision.md](file:///c:/Users/Admin/Desktop/C68-LR/results/gate_decision.md)).

### Step 7: Final Synthesis
- [x] Populated research paper Section 7 with exact gate metrics.
- [x] Delivered final comprehensive research report and summary message with verdict, 5 key numbers, and suspicious anomalies.
