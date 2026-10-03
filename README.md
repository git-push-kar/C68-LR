# Polyvalent: Logical Reasoning (LR) Module

This module implements the complete Logical Reasoning pipeline for **InternVL3-2B** based on [`Polyvalent_LR_Integration_Note.md`](Polyvalent_LR_Integration_Note.md).

---

## 🏗️ Architecture & Pipeline Overview

```text
1. Data Ingestion & Splitting (ProofWriter, P-FOLIO, FOLIO, AbductionRules)
   └── Zero-leakage premise-group partitioning
   └── Strict isolation of unseen benchmarks (LogicBench, Multi-LogiEval, LogicNLI)

2. Staged Curriculum LoRA Training
   ├── Stage 1: Structured deduction (ProofWriter)
   ├── Stage 2: Proof supervision & First-Order Logic (P-FOLIO, FOLIO)
   └── Stage 3 (Optional): Abductive reasoning (AbductionRules)

3. Benchmark Evaluation
   └── Matched prompt & greedy decoding comparison against quarantined benchmarks

4. Foundation Weight Fusion & Equivalence Verification
   ├── Additive LoRA merge: W_LR = W_base + (alpha / r) * B @ A
   ├── Numerical logit equivalence check (tolerance < 1e-3)
   └── Export frozen foundation: checkpoints/internvl3_2b_lr_fused
```

---

## 🚀 Quickstart & Execution Steps

### 1. All-in-One Continuous Dataset Setup (Cross-Platform)
Run this single command on **any device** (Linux, macOS, Windows) to download, extract, normalize, and partition all datasets:
```bash
python scripts/setup_and_prepare_data.py
```
*(Or via chained CLI commands: `python scripts/download_real_datasets.py && python scripts/prepare_data.py`)*

### 2. Run Staged Curriculum LoRA Training
Trains the LR LoRA adapter across the staged curriculum on `InternVL3-2B`:
```bash
python scripts/train_lr_lora.py --model_name OpenGVLab/InternVL3-2B --output_dir checkpoints/internvl3_2b_lr --rank 32 --alpha 64
```

### 3. Evaluate Unseen Benchmarks
Evaluates the base model vs. base + LR LoRA on quarantined benchmarks:
```bash
python scripts/evaluate_benchmarks.py --model_path OpenGVLab/InternVL3-2B --adapter_dir checkpoints/internvl3_2b_lr/final_lr_lora --benchmark_dir data/benchmarks
```

### 4. Merge Weights into Standalone Foundation Checkpoint
Merges the LoRA adapter into a copy of `InternVL3-2B` and verifies numerical equivalence:
```bash
python scripts/merge_weights.py --base_model OpenGVLab/InternVL3-2B --adapter_dir checkpoints/internvl3_2b_lr/final_lr_lora --merged_output_dir checkpoints/internvl3_2b_lr_fused
```

---

## 🧪 Running Tests
Run the comprehensive unit test suite:
```bash
python -m pytest tests/ -v
```
