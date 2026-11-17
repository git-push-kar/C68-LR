# Architectural and Methodological Decisions

This document records all engineering and scientific decisions made during the autonomous execution of the Polyvalent Logical Reasoning (LR) diagnostic, baseline, perturbation, gate assessment, and paper generation tasks on InternVL3-2B.

---

### Decision 1: Model Variants and Base Model Path
- **Base Model:** `DEFAULT_BASE_MODEL_PATH` points to the local weights at `C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B`. Inside the workspace, a byte-identical copy is preserved under `models/lr_adapter/` (and `models_v3/adapter/`), which allows isolated, sandboxed execution without requiring filesystem breakout.
- **Evaluated Checkpoints:**
  1. `base`: Pure base InternVL3-2B (frozen language model backbone).
  2. `stage_1`: `checkpoints/internvl3_2b_lr_v2/stage_1_structured_deduction/adapter` (r=16, alpha=32).
  3. `stage_2`: `checkpoints/internvl3_2b_lr_v2/stage_2_proof_and_nl_fol/adapter` (r=16, alpha=32).
  4. `stage_3`: `checkpoints/internvl3_2b_lr_v2/stage_3_abductive_reasoning_optional/adapter` (r=16, alpha=32).
  5. `final_lr_lora`: `checkpoints/internvl3_2b_lr_v2/final_lr_lora` (r=16, alpha=32, culmination of v2 staged curriculum).
  6. `v3_sft` (Latest Trained Model): `checkpoints/internvl3_2b_reasoning_v3/sft` (packaged under `models_v3/adapter/adapter`, r=64, alpha=128), as specified in the instruction: *"The model to be used is the latest trained model which is not the fused model"*.
- **Note on Merged/Fused Checkpoints:** Per instructions, neither `checkpoints/internvl3_2b_lr_fused` nor `models/lr_merged` nor `models_v3/merged` are used for training or evaluation; all inference runs strictly on the base model dynamically patched with the specified LoRA adapter via `peft.PeftModel`.

---

### Decision 2: Python Environment and CUDA Configuration
- **Runtime Environment:** Python 3.12 at `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe` is selected as the primary execution engine for all GPU inference and evaluation scripts. It features PyTorch 2.6.0+cu124, `peft` 0.17.1, `transformers` 4.55.4, and `timm` 1.0.30.
- **Unit Testing:** Python 3.11 (`python -m pytest tests/ -v`) is used for running test suites where `pytest` is cleanly provisioned.
- **CUDA Cache & Allocator Safety:**
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is strictly enforced to avoid known WDDM/Windows CUDA allocator crashes (`!handles_.at(i) INTERNAL ASSERT FAILED`).
  - `HF_HOME`, `HF_MODULES_CACHE`, and `TRANSFORMERS_CACHE` are redirected to `<workspace>/.cache/huggingface` to guarantee reliable in-workspace caching without permission errors.

---

### Decision 3: High-Throughput Text-Only Batching and Memory Sizing
- **Vision Tower Bypass:** As logic reasoning benchmarks are purely textual, inference directly routes inputs to `model.language_model` (bypassing the InternViT vision encoder and projection modules).
- **Padding & Tokenization:** Prompts are tokenized with left-padding (`tokenizer.padding_side = "left"`) and sorted by prompt sequence length to minimize extraneous padding tokens.
- **Batch Sizing & OOM Auto-Recovery:**
  - Direct Evaluation (`max_new_tokens=16`): Initial batch size = 64. Profiled on RTX A5000: 4.05 GB peak VRAM, throughput ~40.8 items/sec.
  - CoT Evaluation (`max_new_tokens=256`): Initial batch size = 64. Profiled on RTX A5000: 4.27 GB peak VRAM, throughput ~8.7 items/sec.
  - OOM handler catches `torch.cuda.OutOfMemoryError` or CUDA driver errors, flushes cache with `gc.collect()` and `torch.cuda.empty_cache()`, halves the batch size (64 -> 32 -> 16 -> 8), retries, and logs the adjusted batch size.

---

### Decision 4: Benchmark Dataset Strategy and In-Distribution Validation
- **Quarantined Unseen Benchmarks:** Canonical data from `data/benchmarks_real/`:
  - `LogicBench-BQA` (1,520 items, binary True/False)
  - `LogicBench-MCQA` (500 items, 4-choice)
  - `LogicNLI` (2,000 items, 4-class NLI)
  - `Multi-LogiEval` (1,696 items, diagnostic split due to 71.5% True label imbalance)
- **Legacy Files (`data/benchmarks/`):** Evaluated strictly for the reproduction audit. The audit documents why `evaluation_results_full.json` (100% on LogicBench/Multi-LogiEval, 0% on LogicNLI) and `results_baseline.json` (0% across all sets) resulted from legacy single-label file containers and absence of `<answer>` format markers in base zero-shot generation.
- **In-Distribution Validation Sets:**
  - `ProofWriter_val` (`data/processed/ProofWriter_val.jsonl`, stratified/subsampled to 1,000 examples with depths 0 to 5)
  - `FOLIO_val` (`data/processed/FOLIO_val.jsonl`, all 120 validation examples)
  - `AbductionRules_val` (`data/processed/AbductionRules_val.jsonl`, 1,000 examples)
  - `v3-holdout` (`data/benchmarks_real/v3-holdout/items.jsonl`, 900 examples: Set A, Set D, Set E)

---

### Decision 5: Decoding, Self-Consistency, and Prompt Templates
- **Direct Prompt:** System prompt instructs immediate deductive verdict output formatted as `<answer>[Label]</answer>`. `max_new_tokens = 16`.
- **CoT Prompt:** Instructs step-by-step reasoning inside `<reasoning>...</reasoning>` or `<think>...</think>`, followed by `<answer>[Label]</answer>`. `max_new_tokens = 256`. Early stopping halts generation as soon as `</answer>` is decoded.
- **Few-Shot Direct (3-Shot):** Fixed 3 exemplar reasoning problems prepended to the prompt, demonstrating problem statement and immediate `<answer>[Label]</answer>`.
- **Self-Consistency (SC5):** 5 stochastic samples per item at `temperature = 0.7, top_p = 0.95, seed = 42`. Evaluated across a fixed stratified sample of 2,000 items (spanning LogicBench-BQA, LogicBench-MCQA, LogicNLI, Multi-LogiEval, and in-distribution sets). Final label determined by majority vote; ties broken deterministically.
- **Answer Extraction:** Utilizes `polyvalent_lr/evaluation/bench_extract.py`, standardizing extraction across `<answer>` tags, explicit `Answer:` lines, and fallback to the last valid token matching the benchmark's label space.

---

### Decision 6: Contamination and Perturbation Protocols
- **Contamination Check:** Exact string match and normalized 8-gram Jaccard / containment computed between training corpora (`ProofWriter_train`, `FOLIO_train`, `AbductionRules_train`, and `data/v3/mix/train.jsonl`) and every test split.
- **Perturbations (Phase 3):**
  - Synthesizes 6 perturbation modes across items solved correctly by both base and tuned models:
    a) Entity/predicate renaming with nonce alphanumeric tokens.
    b) Premise shuffling (3 deterministic permutations per item).
    c) Distractor premise insertion (1, 3, 5 irrelevant but syntactically coherent premises).
    d) Deterministic template-based paraphrasing.
    e) Depth extrapolation (stratified across reasoning depths).
    f) Conclusion negation with label flip (e.g. True <-> False, Entailment <-> Contradiction).
  - Validation spot-check: 20 examples per type exported to `results/perturb_examples.md`.

---

### Decision 7: Windows WDDM Memory Footprint and Dynamic Sequence Batch Sizing
- **Host GPU Footprint:** Under Windows WDDM, ~11.3 GB of the 24,564 MiB RTX A5000 VRAM is reserved by the OS display driver and desktop window manager. The net headroom available for PyTorch without risking driver memory paging is ~13.2 GB.
- **Paging Bottleneck Discovery:** When prompts with sequence lengths >500 tokens (e.g. LogicNLI with 24 premises) were run at a static batch size of 64, total allocation hit 24.0 GB, causing Windows WDDM to page KV tensors to host RAM across PCIe Gen4, degrading throughput from ~45 items/sec to ~0.5 items/sec.
- **Dynamic Sequence Batching:**
  - Direct / Few-Shot (`max_new_tokens=16`): Prompts >1200 chars are dynamically capped at batch size 16; prompts >700 chars at batch size 32; short prompts (<700 chars) use batch size 64.
  - CoT / Sampling (`max_new_tokens=256`): Prompts >1200 chars are capped at batch size 16; prompts >800 chars at batch size 24; prompts >400 chars at batch size 48; short prompts (<400 chars) at batch size 64.
  - This keeps peak PyTorch allocation below 8.5 GB (total VRAM < 19.8 GB), completely eliminating WDDM paging while sustaining high throughput across both direct and multi-sample self-consistency evaluations.

---

### Decision 8: Early Stopping on Answer Markers and Sequential GPU Model Scheduling
- **Early Stopping via Custom `StopOnSequenceCriteria`:**
  - In zero-shot and CoT decoding, the model emits its rationale inside `<reasoning>` and outputs its verdict inside `<answer>...</answer>`. Because the tokenizer vocabulary lacks a single compound token for `</answer>`, standard decoding would continue emitting trailing tokens up to `max_new_tokens = 256`, wasting 85-90% of GPU compute.
  - Implemented `StopOnSequenceCriteria(StoppingCriteria)` targeting token sequences `[522, 9217, 29]` (`</answer>`), `[522, 9217, 397]` (`</answer>\n`), and `[151645]` (`<|im_end|>`).
  - Employs an $O(1)$ per-sequence completion mask across the active batch. As soon as all sequences in the batch have produced an answer tag or EOS, batch generation immediately terminates. This reduces mean generated tokens from 256 to ~35 tokens, accelerating CoT and multi-sample self-consistency by ~7x.
- **Sequential Model Scheduling in Phase 3:**
  - Evaluates all perturbation splits on `base` first, calls `engine_base.unload()`, and only then loads `engine_tuned` (`v3_sft`), strictly observing the single-model GPU rule and preventing concurrent VRAM allocation spikes.


