import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from torch.utils.data import Dataset
from polyvalent_lr.config import CurriculumStageConfig, DatasetSource
from polyvalent_lr.data.schema import LogicalExample


class LogicalReasoningDataset(Dataset):
    """PyTorch Dataset wrapping structured LogicalExample items for instruction fine-tuning."""

    def __init__(self, examples: List[LogicalExample], tokenizer=None, max_length: int = 2048):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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

        # Tokenize prompt and target for causal LM loss with prompt masking
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # Mask prompt tokens with -100 so loss is computed ONLY on proof + answer target
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
    """Manages progression through the defined training stages (ProofWriter -> P-FOLIO/FOLIO -> Abduction)."""

    def __init__(self, processed_data_dir: str):
        self.data_dir = Path(processed_data_dir)

    def load_stage_data(self, stage_config: CurriculumStageConfig) -> Tuple[List[LogicalExample], List[LogicalExample]]:
        """Loads train and dev examples corresponding to the datasets in the specified stage."""
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

        return train_examples, val_examples
