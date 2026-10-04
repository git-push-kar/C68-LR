import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from torch.utils.data import Dataset
from tqdm import tqdm
from polyvalent_lr.config import CurriculumStageConfig, DatasetSource
from polyvalent_lr.data.schema import LogicalExample


class LogicalReasoningDataset(Dataset):
    """PyTorch Dataset wrapping structured LogicalExample items with fast cached tokenization."""

    def __init__(self, examples: List[LogicalExample], tokenizer=None, max_length: int = 512, pre_tokenize: bool = True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cached_features = None

        # Pre-tokenize in memory if dataset is under 150k samples for instant loading;
        # For larger sets, tokenize on-the-fly with multi-worker pre-fetching.
        if self.tokenizer is not None and pre_tokenize and 0 < len(examples) <= 150000:
            self._pre_tokenize_all()

    def _pre_tokenize_all(self):
        """Pre-tokenizes all examples in memory to eliminate DataLoader tokenization bottlenecks."""
        print(f"[Dataset] Pre-tokenizing {len(self.examples):,} examples (max_length={self.max_length})...")
        self.cached_features = []
        for ex in tqdm(self.examples, desc="Pre-tokenizing"):
            prompt = ex.to_instruction_prompt()
            target = ex.to_target_text()
            full_text = prompt + target

            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            full_tokens = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]

            labels = list(input_ids)
            prompt_len = len(prompt_tokens["input_ids"])
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            self.cached_features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "example_id": ex.id
            })
        print(f"[Dataset] Pre-tokenization complete for {len(self.cached_features):,} items.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cached_features is not None:
            return self.cached_features[idx]

        example = self.examples[idx]
        prompt = example.to_instruction_prompt()
        target = example.to_target_text()
        full_text = prompt + target

        if self.tokenizer is None:
            return {
                "id": example.id,
                "prompt": prompt,
                "target": target,
                "full_text": full_text,
                "group_id": example.group_id,
                "source": example.source.value,
                "task_type": example.task_type.value
            }

        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        labels = list(input_ids)
        prompt_len = len(prompt_tokens["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "example_id": example.id
        }


class CurriculumManager:
    """Manages progression through the defined training stages with balanced representative sampling."""

    def __init__(self, processed_data_dir: str):
        self.data_dir = Path(processed_data_dir)

    def load_stage_data(self, stage_config: CurriculumStageConfig) -> Tuple[List[LogicalExample], List[LogicalExample]]:
        """Loads train and dev examples with optional balanced subsampling for fast training."""
        train_examples = []
        val_examples = []

        for ds in stage_config.datasets:
            train_file = self.data_dir / f"{ds.value}_train.jsonl"
            val_file = self.data_dir / f"{ds.value}_val.jsonl"

            if train_file.exists():
                with open(train_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            train_examples.append(LogicalExample(**json.loads(line)))

            if val_file.exists():
                with open(val_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            val_examples.append(LogicalExample(**json.loads(line)))

        # Subsample training data if max_train_samples is set and > 0
        if stage_config.max_train_samples and stage_config.max_train_samples > 0 and len(train_examples) > stage_config.max_train_samples:
            stride = len(train_examples) / stage_config.max_train_samples
            sampled_indices = [int(i * stride) for i in range(stage_config.max_train_samples)]
            print(f"[CurriculumManager] Sampling {stage_config.max_train_samples:,} representative examples from {len(train_examples):,} total.")
            train_examples = [train_examples[i] for i in sampled_indices]
        else:
            print(f"[CurriculumManager] Training on 100% full dataset: {len(train_examples):,} examples.")

        # Subsample validation data for fast evaluation
        if stage_config.max_val_samples and stage_config.max_val_samples > 0 and len(val_examples) > stage_config.max_val_samples:
            stride = len(val_examples) / stage_config.max_val_samples
            sampled_val_indices = [int(i * stride) for i in range(stage_config.max_val_samples)]
            print(f"[CurriculumManager] Sampling {stage_config.max_val_samples:,} validation examples from {len(val_examples):,} total.")
            val_examples = [val_examples[i] for i in sampled_val_indices]
        else:
            print(f"[CurriculumManager] Full validation dataset: {len(val_examples):,} examples.")

        return train_examples, val_examples
