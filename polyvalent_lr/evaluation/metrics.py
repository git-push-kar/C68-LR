import re
from typing import Dict, Any, List, Optional, Tuple
from polyvalent_lr.data.normalizer import DatasetNormalizer


class ReasoningMetrics:
    """Calculates accuracy, proof step validity, and reasoning depth from model outputs."""

    @staticmethod
    def extract_think_and_answer(response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts the <think> reasoning trace and <answer> label from generated text."""
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL | re.IGNORECASE)

        think_trace = think_match.group(1).strip() if think_match else None
        answer_label = answer_match.group(1).strip() if answer_match else None

        # Fallback if XML tags were partially omitted
        if not answer_label:
            # Look for lines like "Answer: True", "Conclusion: False"
            alt_match = re.search(r"(?:answer|conclusion|verdict):\s*(true|false|uncertain|entailed|contradiction|neutral)", response_text, re.IGNORECASE)
            if alt_match:
                answer_label = alt_match.group(1).strip()
            else:
                # Last line heuristic
                last_line = response_text.strip().split("\n")[-1]
                answer_label = last_line

        return think_trace, answer_label

    @classmethod
    def evaluate_prediction(
        cls,
        predicted_text: str,
        ground_truth_label: str,
        expected_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluates a single model prediction against ground truth."""
        think_trace, pred_label_raw = cls.extract_think_and_answer(predicted_text)
        pred_label_norm = DatasetNormalizer.normalize_label(pred_label_raw or "")
        gt_label_norm = DatasetNormalizer.normalize_label(ground_truth_label)

        exact_match = (pred_label_norm.lower() == gt_label_norm.lower())

        # Measure reasoning depth (number of reasoning sentences or steps)
        reasoning_depth = 0
        has_reasoning_trace = False
        if think_trace:
            has_reasoning_trace = True
            steps = [s for s in think_trace.split("\n") if s.strip()]
            reasoning_depth = len(steps)

        return {
            "exact_match": exact_match,
            "pred_label": pred_label_norm,
            "gt_label": gt_label_norm,
            "has_reasoning_trace": has_reasoning_trace,
            "reasoning_depth": reasoning_depth,
            "raw_prediction": predicted_text
        }

    @classmethod
    def compute_aggregate_metrics(cls, eval_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Computes summary statistics over all evaluated examples."""
        if not eval_results:
            return {"accuracy": 0.0, "reasoning_trace_rate": 0.0, "avg_depth": 0.0}

        total = len(eval_results)
        correct = sum(1 for r in eval_results if r["exact_match"])
        with_trace = sum(1 for r in eval_results if r["has_reasoning_trace"])
        avg_depth = sum(r["reasoning_depth"] for r in eval_results) / total

        return {
            "accuracy": (correct / total) * 100.0,
            "reasoning_trace_rate": (with_trace / total) * 100.0,
            "avg_reasoning_depth": avg_depth,
            "total_examples": total
        }
