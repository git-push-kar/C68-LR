import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

from polyvalent_lr.config import UnseenBenchmark, EvaluationConfig
from polyvalent_lr.data.schema import LogicalExample
from polyvalent_lr.evaluation.metrics import ReasoningMetrics
from polyvalent_lr.models.loader import ensure_internvl_tokens_configured


class UnseenBenchmarkEvaluator:
    """Evaluates base InternVL3-2B vs. InternVL3-2B + LR LoRA on quarantined logic benchmarks."""

    def __init__(self, benchmark_dir: str = "data/benchmarks", config: Optional[EvaluationConfig] = None):
        self.benchmark_dir = Path(benchmark_dir)
        self.config = config or EvaluationConfig()

    def load_benchmark(self, benchmark_name: UnseenBenchmark) -> List[LogicalExample]:
        """Loads benchmark records from the quarantined directory."""
        file_path = self.benchmark_dir / f"{benchmark_name.value}.jsonl"
        if not file_path.exists():
            print(f"[Warning] Benchmark file {file_path} not found.")
            return []

        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(LogicalExample(**json.loads(line)))

        if self.config.max_eval_samples_per_bench:
            examples = examples[:self.config.max_eval_samples_per_bench]

        return examples

    def evaluate_model(
        self,
        model: Any,
        tokenizer: Any,
        benchmark_name: UnseenBenchmark,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Dict[str, Any]:
        """Runs matched prompt & decoding inference across the designated benchmark."""
        examples = self.load_benchmark(benchmark_name)
        if not examples:
            return {"error": f"No examples loaded for {benchmark_name.value}"}

        model.eval()
        ensure_internvl_tokens_configured(model, tokenizer)
        results = []

        eval_device = getattr(model, "device", torch.device(device))
        print(f"[Benchmark Eval] Evaluating {len(examples)} examples on {benchmark_name.value} (Device: {eval_device})...")
        for ex in tqdm(examples, desc=f"Eval {benchmark_name.value}"):
            prompt = ex.to_instruction_prompt()
            inputs = tokenizer(prompt, return_tensors="pt").to(eval_device)

            with torch.no_grad():
                gen_tokens = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=(self.config.temperature > 0.0),
                    pad_token_id=tokenizer.eos_token_id
                )

            # Strip input prompt from generated output
            input_len = inputs["input_ids"].shape[1]
            generated_text = tokenizer.decode(gen_tokens[0][input_len:], skip_special_tokens=True)

            eval_res = ReasoningMetrics.evaluate_prediction(
                predicted_text=generated_text,
                ground_truth_label=ex.label
            )
            eval_res["id"] = ex.id
            eval_res["task_type"] = ex.task_type.value
            results.append(eval_res)

        aggregate = ReasoningMetrics.compute_aggregate_metrics(results)
        return {
            "benchmark": benchmark_name.value,
            "metrics": aggregate,
            "detailed_results": results
        }
