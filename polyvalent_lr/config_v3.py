"""v3 ("reasoning repair") configuration.

This is a NEW, self-contained configuration namespace for the post-v2 approach.
The v2 artifacts and defaults (``polyvalent_lr/config.py``) are quarantined and
must not be reused or overwritten. v3 writes only to its own directories.

See ``docs/v3/RUNBOOK.md`` for the stage-by-stage commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from polyvalent_lr.config import DEFAULT_BASE_MODEL_PATH, LoRAConfig


# ==============================================================================
# Paths (all v3-specific; v2 outputs are never touched)
# ==============================================================================
BASE_MODEL_PATH = DEFAULT_BASE_MODEL_PATH  # the previously used local InternVL3-2B (unmodified)

V3_ROOT = Path("data/v3")
DATA_A_DIR = V3_ROOT / "A"          # grid/ordering puzzles (solver-generated, CPU)
DATA_B_DIR = V3_ROOT / "B"          # logic + NL rationale (teacher)
DATA_C_DIR = V3_ROOT / "C"          # commonsense / implicit-goal (teacher)
DATA_D_DIR = V3_ROOT / "D"          # Dolly rehearsal (existing)
DATA_E_DIR = V3_ROOT / "E"          # ProofWriter/FOLIO labels (existing)
MIX_DIR = V3_ROOT / "mix"           # assembled SFT mixture
REJECTION_DIR = V3_ROOT / "rejection"

CHECKPOINT_DIR = Path("checkpoints/internvl3_2b_reasoning_v3")
STAGE2_ADAPTER_DIR = CHECKPOINT_DIR / "sft"
STAGE3_ADAPTER_DIR = CHECKPOINT_DIR / "rs"
BEST_ADAPTER_DIR = CHECKPOINT_DIR / "best"

MODELS_DIR = Path("models_v3")
MERGED_MODEL_DIR = MODELS_DIR / "merged"
ADAPTER_MODEL_DIR = MODELS_DIR / "adapter"

HOLDOUT_DIR = V3_ROOT / "holdout"


# ==============================================================================
# LoRA (v3 recipe: rank 64 / alpha 128 / dropout 0.05, LM projections only)
# ==============================================================================
@dataclass
class LoRAV3:
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    # Applied to the InternVL language-model projections. The vision tower has
    # none of these module names, so it stays untouched even though we wrap the
    # whole model (decision Q5: option 1 -> keeps save/merge/chat tooling intact).
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_lora_config(self) -> LoRAConfig:
        return LoRAConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.target_modules),
            bias=self.bias,
        )


# ==============================================================================
# SFT (Stage 2)
# ==============================================================================
@dataclass
class SFTV3:
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    num_epochs: int = 2
    per_device_batch_size: int = 4
    grad_accum: int = 8                       # effective batch 32
    max_length: int = 1024
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True
    group_by_length: bool = True
    dataloader_num_workers: int = 0
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 4
    logging_steps: int = 10
    seed: int = 42
    # Mix weights (must sum to 1.0): A 45 / B 20 / C 10 / D 10 / E 15
    mix_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "A": 0.45, "B": 0.20, "C": 0.10, "D": 0.10, "E": 0.15,
        }
    )
    # Approximate total supervised tokens per epoch (for the ~22M target).
    target_tokens_per_epoch: int = 22_000_000


# ==============================================================================
# Stage 3 (rejection sampling / expert iteration)
# ==============================================================================
@dataclass
class RejectionV3:
    num_puzzles: int = 8000
    samples_per_puzzle: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 768
    learning_rate: float = 5e-5
    num_epochs: int = 1
    reward_model_len: int = 2048
    gpu_memory_utilization: float = 0.92
    oversample_hard: bool = True    # bias toward 5-entity puzzles


# ==============================================================================
# Decoding (Stage 4 inference boosters)
# ==============================================================================
@dataclass
class DecodingV3:
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    max_new_tokens: int = 768
    self_consistency: int = 5        # sample N then majority-vote the final table
    stop_strings: List[str] = field(default_factory=lambda: ["<|im_end|>", "<|im_start|>"])


# ==============================================================================
# Teacher (Stage 1; runs under WSL2 + vLLM, never on this Windows venv)
# ==============================================================================
@dataclass
class TeacherV3:
    model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.92
    tensor_parallel_size: int = 1
    set_b_prompts: int = 6000
    set_b_samples: int = 2
    set_c_prompts: int = 4000
    temperature: float = 0.6


@dataclass
class V3Config:
    base_model_path: str = BASE_MODEL_PATH
    lora: LoRAV3 = field(default_factory=LoRAV3)
    sft: SFTV3 = field(default_factory=SFTV3)
    rejection: RejectionV3 = field(default_factory=RejectionV3)
    decoding: DecodingV3 = field(default_factory=DecodingV3)
    teacher: TeacherV3 = field(default_factory=TeacherV3)


def validate_mix_weights(weights: Dict[str, float], tol: float = 1e-6) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Mix weights must sum to 1.0 (got {total:.4f}): {weights}")
