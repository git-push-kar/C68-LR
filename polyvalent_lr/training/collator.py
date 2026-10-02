import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import DataCollatorForSeq2Seq


@dataclass
class LogicalDataCollator:
    """Collates batch items, padding sequences and applying label masks."""
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # Pad sequences to max length in batch
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id or 0
        )
        padded_attention = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": padded_inputs,
            "attention_mask": padded_attention,
            "labels": padded_labels
        }
