from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import Trainer, TrainingArguments

from polyvalent_lr.config import LRModuleConfig, CurriculumStageConfig
from polyvalent_lr.data.curriculum import CurriculumManager, LogicalReasoningDataset
from polyvalent_lr.training.collator import LogicalDataCollator


class StagedCurriculumTrainer:
    """Orchestrates high-throughput staged curriculum LoRA training for the Polyvalent LR module."""

    def __init__(self, config: LRModuleConfig, model: Any, tokenizer: Any):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.curriculum_manager = CurriculumManager(config.data.processed_data_dir)

        # Enable Ampere Tensor Core hardware acceleration (TF32)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def train_stage(self, stage_config: CurriculumStageConfig) -> Dict[str, Any]:
        """Executes fine-tuning for a single curriculum stage."""
        print(f"\n{'='*50}\nStarting Curriculum Stage {stage_config.stage_id}: {stage_config.name}\n{'='*50}")
        print(f"Description: {stage_config.description}")
        print(f"Datasets: {[d.value for d in stage_config.datasets]}")

        # Check model device placement
        try:
            primary_device = next(self.model.parameters()).device
            print(f"[Device Placement] Model parameter tensor device: {primary_device}")
        except Exception:
            primary_device = "unknown"

        train_examples, val_examples = self.curriculum_manager.load_stage_data(stage_config)
        print(f"Loaded {len(train_examples)} train examples, {len(val_examples)} validation examples.")

        if not train_examples:
            print(f"[Warning] No training data found for stage {stage_config.name}. Skipping.")
            return {"status": "skipped", "reason": "empty_dataset"}

        train_dataset = LogicalReasoningDataset(
            train_examples,
            tokenizer=self.tokenizer,
            max_length=stage_config.max_length,
            pre_tokenize=True
        )
        val_dataset = LogicalReasoningDataset(
            val_examples,
            tokenizer=self.tokenizer,
            max_length=stage_config.max_length,
            pre_tokenize=True
        ) if val_examples else None

        stage_output_dir = Path(self.config.output_dir) / f"stage_{stage_config.stage_id}_{stage_config.name}"
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        use_cuda = torch.cuda.is_available()
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
        use_fp16 = use_cuda and not use_bf16

        training_args_dict = {
            "output_dir": str(stage_output_dir),
            "num_train_epochs": stage_config.epochs,
            "per_device_train_batch_size": stage_config.batch_size,
            "gradient_accumulation_steps": stage_config.gradient_accumulation_steps,
            "learning_rate": stage_config.learning_rate,
            "warmup_ratio": stage_config.warmup_ratio,
            "logging_steps": 25,
            "save_strategy": "epoch",
            "eval_strategy" if hasattr(TrainingArguments, "eval_strategy") else "evaluation_strategy": "epoch" if val_dataset else "no",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "tf32": use_cuda,
            "group_by_length": True,
            "dataloader_pin_memory": use_cuda,
            "dataloader_num_workers": 0,  # Zero-IPC overhead on Windows with in-memory pre-tokenized tensors
            "optim": "adamw_torch_fused" if use_cuda else "adamw_torch",
            "save_total_limit": 2,
            "remove_unused_columns": False,
            "report_to": "none"
        }

        training_args = TrainingArguments(**training_args_dict)

        collator = LogicalDataCollator(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator
        )

        train_result = trainer.train()
        print(f"Stage {stage_config.stage_id} completed. Final loss: {train_result.training_loss:.4f}")

        # Save stage adapter
        stage_adapter_dir = stage_output_dir / "adapter"
        self.model.save_pretrained(stage_adapter_dir)
        self.tokenizer.save_pretrained(stage_adapter_dir)
        print(f"Saved stage adapter to {stage_adapter_dir}")

        return {
            "stage_id": stage_config.stage_id,
            "stage_name": stage_config.name,
            "training_loss": train_result.training_loss,
            "adapter_dir": str(stage_adapter_dir)
        }

    def run_full_curriculum(self) -> Dict[str, Any]:
        """Runs all curriculum stages sequentially, accumulating adapter updates."""
        all_stage_results = []
        for stage_cfg in self.config.curriculum_stages:
            res = self.train_stage(stage_cfg)
            all_stage_results.append(res)

        final_adapter_dir = Path(self.config.output_dir) / "final_lr_lora"
        self.model.save_pretrained(final_adapter_dir)
        self.tokenizer.save_pretrained(final_adapter_dir)
        print(f"\n[Completed] Full LR curriculum finished. Final adapter saved to {final_adapter_dir}")

        return {
            "status": "completed",
            "stages": all_stage_results,
            "final_adapter_dir": str(final_adapter_dir)
        }
