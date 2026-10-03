import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from peft import PeftModel

from polyvalent_lr.config import UnseenBenchmark, EvaluationConfig
from polyvalent_lr.models.loader import load_internvl_model_and_tokenizer
from polyvalent_lr.evaluation.benchmarks import UnseenBenchmarkEvaluator
from polyvalent_lr.evaluation.regression import GeneralRegressionSuite


def main():
    parser = argparse.ArgumentParser(description="Evaluate on quarantined unseen benchmarks (LogicBench, Multi-LogiEval, LogicNLI)")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-2B", help="Path or HF ID of model to evaluate")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--benchmark_dir", type=str, default="data/benchmarks", help="Quarantined benchmark dir")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output summary report file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target compute device (cuda/cpu)")
    parser.add_argument("--device_map", type=str, default="auto" if torch.cuda.is_available() else "cpu", help="device_map for transformers (auto/cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "fp16", "float32"], help="Model torch_dtype")
    parser.add_argument("--run_regression_probes", action="store_true", default=True, help="Run general NLP regression probes")
    args = parser.parse_args()

    # Determine torch dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:  # auto
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

    # Load model and tokenizer with GPU acceleration and version compatibility
    model, tokenizer = load_internvl_model_and_tokenizer(
        model_name_or_path=args.model_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        is_eval=True
    )

    if args.adapter_dir:
        print(f"[Evaluation] Attaching LoRA adapter from {args.adapter_dir}...")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model.eval()

    evaluator = UnseenBenchmarkEvaluator(benchmark_dir=args.benchmark_dir)
    report = {"model": args.model_path, "adapter": args.adapter_dir, "benchmarks": {}}

    for bench in [UnseenBenchmark.LOGICBENCH, UnseenBenchmark.MULTI_LOGIEVAL, UnseenBenchmark.LOGICNLI]:
        bench_result = evaluator.evaluate_model(model, tokenizer, bench, device=args.device)
        report["benchmarks"][bench.value] = bench_result
        if "metrics" in bench_result:
            print(f"[{bench.value}] Accuracy: {bench_result['metrics']['accuracy']:.2f}% | "
                  f"Trace Rate: {bench_result['metrics']['reasoning_trace_rate']:.2f}% | "
                  f"Avg Depth: {bench_result['metrics']['avg_reasoning_depth']:.2f}")

    if args.run_regression_probes:
        print("\n[Evaluation] Running General NLP Regression Probes...")
        reg_res = GeneralRegressionSuite.run_sanity_checks(model, tokenizer, device=args.device)
        report["regression_probes"] = reg_res
        print(f"[Evaluation] Regression probes passed: {reg_res['probes_passed']}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Evaluation Completed] Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
