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

The normalizer sanitizes symbolic proof soup (`triple*`/`rule*` token strings) and de-duplicates
`(prompt, target)` pairs before writing `data/processed/`.

### 1b. Prepare Rehearsal Data (Dolly-15k)
Downloads and normalizes `databricks/databricks-dolly-15k` into general-instruction rehearsal
examples used to prevent catastrophic forgetting of general chat:
```bash
python scripts/prepare_rehearsal.py
```
Writes `data/processed/Rehearsal_train.jsonl` / `Rehearsal_val.jsonl`. The curriculum interleaves
these at `rehearsal_ratio` per stage (defaults: stage 1 = 10%, stage 2 = 15%, stage 3 = 10%).

### 2. Run Staged Curriculum LoRA Training
Trains the LR LoRA adapter across the staged curriculum on the **local** InternVL3-2B base
(`C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B`, set via
`DEFAULT_BASE_MODEL_PATH` in `polyvalent_lr/config.py`), rank 16 / alpha 32, capped
per-stage sampling, early stopping with best-model selection:
```bash
python scripts/train_lr_lora.py --output_dir checkpoints/internvl3_2b_lr_v2 --rank 16 --alpha 32
```

Per-stage caps (`max_train_samples` = 50k / 50k / 40k) now apply by default; pass
`--max_train_samples N` to override all stages, or `--full_dataset` to disable caps.
Preview the resolved schedule without loading the model:
```bash
python scripts/train_lr_lora.py --dry_run --batch_size 4 --grad_accum 8
```

`--batch_size 4 --grad_accum 8` (effective batch 32) keeps the training footprint well
under 24 GB so a remote session (e.g. AnyDesk) can stay open without VRAM paging, which
otherwise slows the run by ~5x.

Do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on Windows — it is
unstable and can crash the CUDA caching allocator with `!handles_.at(i) INTERNAL ASSERT
FAILED` during `model.to(cuda)`. The training/eval/chat entry points now force it off
automatically.

Redo/continue a single stage from a previously saved adapter (e.g. rerun only stage 3
starting from the stage-2 adapter):
```bash
python scripts/train_lr_lora.py --output_dir checkpoints/internvl3_2b_lr_v2 --rank 16 --alpha 32 \
  --batch_size 4 --grad_accum 8 --max_length 384 \
  --only_stage 3 --init_adapter checkpoints/internvl3_2b_lr_v2/stage_2_proof_and_nl_fol/adapter
```

### 3. Evaluate Unseen Benchmarks
Evaluates the base model vs. base + LR LoRA on quarantined benchmarks:
```bash
python scripts/evaluate_benchmarks.py --adapter_dir checkpoints/internvl3_2b_lr_v2/final_lr_lora --benchmark_dir data/benchmarks
```

### 4. Merge Weights into Standalone Foundation Checkpoint
Merges the LoRA adapter into a copy of the local base and verifies numerical equivalence:
```bash
python scripts/merge_weights.py --adapter_dir checkpoints/internvl3_2b_lr_v2/final_lr_lora --merged_output_dir checkpoints/internvl3_2b_lr_v2_fused
```

### 5. Launch the Chat UI
Starts a zero-dependency web chat interface (stdlib `http.server` + SSE streaming) against the fused LR checkpoint:
```bash
python scripts/chat_ui.py
```
Then open `http://127.0.0.1:8005`. Useful flags: `--model_path`, `--adapter_dir`, `--port`, `--host`, `--device`, `--dtype`.

To chat with the base model + LoRA adapter instead of the fused checkpoint:
```bash
python scripts/chat_ui.py --adapter_dir checkpoints/internvl3_2b_lr_v2/final_lr_lora
```

---

## 🧪 Running Tests
Run the comprehensive unit test suite:
```bash
python -m pytest tests/ -v
```
