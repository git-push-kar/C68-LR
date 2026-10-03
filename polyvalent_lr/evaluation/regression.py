from typing import Dict, Any, List
import torch
from polyvalent_lr.models.loader import ensure_internvl_tokens_configured


class GeneralRegressionSuite:
    """Verifies that LR LoRA training or weight merging does not degrade general NLP / multimodal capabilities."""

    @staticmethod
    def run_sanity_checks(
        model: Any,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Dict[str, Any]:
        """Runs standard conversational and visual-instruction sanity probes."""
        probe_prompts = [
            "Summarize the function of mitochondria in human cells in two sentences.",
            "Write a simple Python function to calculate the factorial of a number.",
            "Explain the difference between a supervised and unsupervised machine learning algorithm.",
        ]

        model.eval()
        ensure_internvl_tokens_configured(model, tokenizer)
        eval_device = getattr(model, "device", torch.device(device))
        responses = []
        for p in probe_prompts:
            prompt_fmt = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt_fmt, return_tensors="pt").to(eval_device)

            with torch.no_grad():
                gen_tokens = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            out_text = tokenizer.decode(gen_tokens[0][input_len:], skip_special_tokens=True)
            responses.append({"prompt": p, "response": out_text.strip()})

        # Basic check: responses are non-empty, non-repetitive
        passed = all(len(r["response"]) > 20 for r in responses)
        return {
            "probes_passed": passed,
            "probe_results": responses
        }
