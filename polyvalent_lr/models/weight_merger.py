import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from polyvalent_lr.models.loader import load_internvl_model_and_tokenizer


class WeightMerger:
    """Merges trained LR LoRA weights into InternVL3-2B base model and verifies numerical equivalence."""

    @staticmethod
    def merge_lora_and_save(
        base_model_path: str,
        adapter_path: str,
        output_dir: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None
    ) -> str:
        """Fuses delta weights W_LR = W_base + (alpha/r) * B * A into a new standalone model checkpoint."""
        print(f"[Merge] Loading base model from {base_model_path}...")
        base_model, tokenizer = load_internvl_model_and_tokenizer(
            model_name_or_path=base_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            is_eval=True
        )

        print(f"[Merge] Loading LR LoRA adapter from {adapter_path}...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)

        print("[Merge] Merging LoRA layers into base model weights...")
        # peft merge_and_unload computes W_effective = W + (alpha/r)*B*A
        merged_model = peft_model.merge_and_unload()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"[Merge] Saving merged InternVL3-2B-LR foundation to {output_path}...")
        merged_model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)

        print("[Merge] Standalone foundation checkpoint saved successfully.")
        return str(output_path)

    @staticmethod
    def verify_numerical_equivalence(
        base_model_path: str,
        adapter_path: str,
        merged_model_path: str,
        sample_prompt: str = "Premises:\n- Fact A is true.\nHypothesis:\nIs A true?\n",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tolerance: float = 1e-3
    ) -> Dict[str, Any]:
        """Tests that dynamic LoRA inference and merged checkpoint inference produce equivalent logits."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            if device == "cuda" and torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32

        print(f"[Equivalence Check] Loading dynamic base + LoRA model on {device} ({torch_dtype})...")
        base, tokenizer = load_internvl_model_and_tokenizer(
            model_name_or_path=base_model_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            is_eval=True
        )
        lora_model = PeftModel.from_pretrained(base, adapter_path)
        lora_model.eval()

        print(f"[Equivalence Check] Loading merged standalone model on {device} ({torch_dtype})...")
        merged_model, _ = load_internvl_model_and_tokenizer(
            model_name_or_path=merged_model_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            is_eval=True
        )
        merged_model.eval()

        inputs = tokenizer(sample_prompt, return_tensors="pt")
        target_device = getattr(base, "device", torch.device(device))
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits_lora = lora_model(**inputs).logits
            logits_merged = merged_model(**inputs).logits

        diff = torch.abs(logits_lora - logits_merged)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        is_equivalent = max_diff < tolerance

        result = {
            "max_absolute_difference": max_diff,
            "mean_absolute_difference": mean_diff,
            "tolerance": tolerance,
            "is_equivalent": is_equivalent,
            "device": str(target_device),
            "torch_dtype": str(torch_dtype)
        }
        print(f"[Equivalence Check] Max diff: {max_diff:.6f} | Mean diff: {mean_diff:.6f} | Passed: {is_equivalent}")
        return result
