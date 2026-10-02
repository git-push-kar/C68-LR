import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from polyvalent_lr.config import UnseenBenchmark, EvaluationConfig
from polyvalent_lr.evaluation.benchmarks import UnseenBenchmarkEvaluator
from polyvalent_lr.evaluation.regression import GeneralRegressionSuite


def main():
    parser = argparse.ArgumentParser(description="Evaluate on quarantined unseen benchmarks (LogicBench, Multi-LogiEval, LogicNLI)")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-2B", help="Path or HF ID of model to evaluate")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--benchmark_dir", type=str, default="data/benchmarks", help="Quarantined benchmark dir")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output summary report file")
    parser.add_argument("--run_regression_probes", action="store_true", default=True, help="Run general NLP regression probes")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Evaluation] Loading model from {args.model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True
    )

    if args.adapter_dir:
        print(f"[Evaluation] Attaching LoRA adapter from {args.adapter_dir}...")
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    evaluator = UnseenBenchmarkEvaluator(benchmark_dir=args.benchmark_dir)
    report = {"model": args.model_path, "adapter": args.adapter_dir, "benchmarks": {}}

    for bench in [UnseenBenchmark.LOGICBENCH, UnseenBenchmark.MULTI_LOGIEVAL, UnseenBenchmark.LOGICNLI]:
        bench_result = evaluator.evaluate_model(model, tokenizer, bench, device=device)
        report["benchmarks"][bench.value] = bench_result
        if "metrics" in bench_result:
            print(f"[{bench.value}] Accuracy: {bench_result['metrics']['accuracy']:.2f}% | "
                  f"Trace Rate: {bench_result['metrics']['reasoning_trace_rate']:.2f}% | "
                  f"Avg Depth: {bench_result['metrics']['avg_reasoning_depth']:.2f}")

    if args.run_regression_probes:
        print("\n[Evaluation] Running General NLP Regression Probes...")
        reg_res = GeneralRegressionSuite.run_sanity_checks(model, tokenizer, device=device)
        report["regression_probes"] = reg_res
        print(f"[Evaluation] Regression probes passed: {reg_res['probes_passed']}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Evaluation Completed] Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
