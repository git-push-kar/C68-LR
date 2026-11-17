# Open Items for Formal Submission

This document tracks items required for formal conference submission (e.g., NeurIPS / ICLR / ACL track) following the completion of diagnostic evaluation, baseline comparison, and robustness analysis.

---

### 1. Reinforcement Learning via GRPO
- **Status:** Evaluated at Gate. Execution pending gate verdict and computational schedule.
- **Requirements:**
  - Execute Group Relative Policy Optimization (GRPO) training using the verified group size ($G=4$) and KL penalty ($\beta=0.04$) on the language backbone.
  - Evaluate checkpoint across quarantined benchmarks (`LogicBench-BQA`, `LogicBench-MCQA`, `LogicNLI`) to verify whether rule-based reinforcement learning resolves remaining deductive and NLI failure modes without catastrophic forgetting.
  - Perform ablation studies comparing reward functions: rule-based outcome reward vs format reward vs curriculum by depth.

---

### 2. Multimodal and Vision-Language Regression Benchmarks
- **Status:** Pre-registered as `TBD` in the main results table.
- **Requirements:**
  - Standard multimodal benchmark evaluations:
    - **MMBench** (Standard cross-domain multimodal perception and reasoning).
    - **MMStar** (High-quality vision-language reasoning benchmark).
  - Verification that the frozen vision tower (`InternViT-300M`) and cross-modal MLP projection layer retain zero performance degradation after pure-text LoRA adaptation.

---

### 3. Multi-Seed Replication
- **Status:** Single primary seed (`seed = 42`) evaluated in diagnostic and self-consistency suites; 1,000 bootstrap resamples computed for confidence intervals.
- **Requirements:**
  - Run 3 additional independent random seeds (`seed = 123, 456, 789`) across the entire diagnostic and perturbation suite.
  - Report mean $\pm$ standard error across distinct seeds to quantify optimization variance.

---

### 4. Qualitative and Human Evaluation
- **Status:** Automated metrics and solver verification utilized across all benchmarks.
- **Requirements:**
  - Blind expert human evaluation of 100 randomly sampled chain-of-thought derivations (50 base, 50 tuned) scored on:
    1. Premise fidelity (no hallucinated premises or entities).
    2. Deductive validity (each intermediate step follows necessarily from previous steps).
    3. Conciseness and absence of tag-spam or looping degeneracy.
