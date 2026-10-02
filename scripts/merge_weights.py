import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polyvalent_lr.models.weight_merger import WeightMerger


def main():
    parser = argparse.ArgumentParser(description="Merge LR LoRA weights into InternVL3-2B standalone foundation")
    parser.add_argument("--base_model", type=str, default="OpenGVLab/InternVL3-2B", help="Path/name of base model")
    parser.add_argument("--adapter_dir", type=str, default="checkpoints/internvl3_2b_lr/final_lr_lora", help="Trained adapter dir")
    parser.add_argument("--merged_output_dir", type=str, default="checkpoints/internvl3_2b_lr_fused", help="Fused model output dir")
    parser.add_argument("--verify_equivalence", action="store_true", default=True, help="Run numerical equivalence test")
    args = parser.parse_args()

    print(f"[Merge Pipeline] Base Model: {args.base_model}")
    print(f"[Merge Pipeline] Adapter Path: {args.adapter_dir}")
    print(f"[Merge Pipeline] Fused Output Path: {args.merged_output_dir}")

    # 1. Merge and save checkpoint
    saved_path = WeightMerger.merge_lora_and_save(
        base_model_path=args.base_model,
        adapter_path=args.adapter_dir,
        output_dir=args.merged_output_dir
    )

    # 2. Verify numerical equivalence
    if args.verify_equivalence:
        eq_res = WeightMerger.verify_numerical_equivalence(
            base_model_path=args.base_model,
            adapter_path=args.adapter_dir,
            merged_model_path=args.merged_output_dir
        )
        if eq_res["is_equivalent"]:
            print("[Merge Pipeline] Numerical Equivalence Check PASSED! Fused checkpoint is ready for domain adapters.")
        else:
            print(f"[Merge Pipeline] WARNING: Equivalence check exceeded tolerance: Max diff = {eq_res['max_absolute_difference']}")


if __name__ == "__main__":
    main()
