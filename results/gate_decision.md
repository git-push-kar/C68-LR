# GRPO Gate Decision & Assessment Report

## **Verdict: `FIX-FORMAT-FIRST`**

**Action Recommendation:** Execute a targeted SFT on verified natural-language rationales to restore reasoning tags and eliminate malformed formats before GRPO.

### 1. Quantitative Diagnostic Numbers

- **Evaluated Training Problems:** 1500 (ProofWriter-train: 1000, FOLIO-train: 500)
- **Total CoT Completions:** 12000 (8 per problem, Temperature = 0.7)
- **Parse Failure Rate:** 0.29%
- **Mixed-Result Problems (1–7 / 8 correct):** 41.73% (626/1500)
- **Zero-Correct Problems (0 / 8 correct):** 12.67% (190/1500)
- **Consistently Solved Problems (8 / 8 correct):** 45.60% (684/1500)
- **Pass@1:** 67.38%
- **Pass@8:** 87.33%
- **Mean Output Length:** 13.9 tokens
- **Reasoning Text Presence (>=30 non-answer tokens):** 1.69% (Threshold: 50%)

### 2. Recommended GRPO Hardware & Hyperparameter Configuration (RTX A5000 24GB)

- **Target Architecture:** Pure language model backbone (`Qwen2ForCausalLM`); vision tower frozen and excluded from compute graph.
- **LoRA Configuration:** `r = 16`, `lora_alpha = 32`, targeting attention & MLP projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
- **Group Size (Rollouts per Prompt):** `G = 4` to keep per-step rollout memory within 18 GB VRAM.
- **Max Completion Length:** 256 tokens.
- **KL Divergence Coefficient:** `beta = 0.04` (prevents reward collapse and format degeneration).
- **Reward Design:** `R = 1.0` for correct label exact match; format bonus `+0.1` for valid `<reasoning>` and `<answer>` XML tags; penalty `-0.5` for unparseable output.
- **Curriculum by Depth:** Phase GRPO training from shallow deductive problems (depth 0–1) into multi-step FOL problems (depth 2–5).

### 3. Key Operational Risks

1. **Reward Hacking & Label Prior Bias:** Without KL penalty, the model may optimize for majority class guessing rather than deductive chains.
2. **Format Degeneration:** Output collapsing into single-token emits without reasoning trace.
3. **WDDM VRAM Allocator Overhead:** Enforce `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` to ensure sustained 24GB stability.
