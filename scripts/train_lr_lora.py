import sys
import argparse
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from polyvalent_lr.config import LRModuleConfig, LoRAConfig
from polyvalent_lr.models.lora_setup import setup_internvl_lr_lora
from polyvalent_lr.training.trainer import StagedCurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Staged LR LoRA on InternVL3-2B on GPU (RTX A5000 / Ampere)")
    parser.add_argument("--config_file", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-2B", help="Base model path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/internvl3_2b_lr", help="Adapter output dir")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size (default: 16)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs per stage (default: 2)")
    parser.add_argument("--max_train_samples", type=int, default=50000, help="Max representative training samples per stage (default: 50000)")
    parser.add_argument("--max_val_samples", type=int, default=1000, help="Max validation samples per stage (default: 1000)")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length (default: 512)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device (cuda/cpu)")
    parser.add_argument("--device_map", type=str, default="auto" if torch.cuda.is_available() else "cpu", help="device_map for transformers (auto/cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "fp16", "float32"], help="Model torch_dtype")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    args = parser.parse_args()

    # Hardware & compute diagnostics
    cuda_avail = torch.cuda.is_available()
    print("=" * 65)
    print("  POLYVALENT LOGICAL REASONING (LR) - GPU TRAINING PIPELINE")
    print("=" * 65)
    print(f"  PyTorch Version:       {torch.__version__}")
    print(f"  CUDA Available:        {cuda_avail}")
    if cuda_avail:
        print(f"  CUDA Device:           {torch.cuda.get_device_name(0)} (Device 0)")
        print(f"  CUDA Device Count:     {torch.cuda.device_count()}")
        print(f"  Total GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"  bfloat16 Supported:    {torch.cuda.is_bf16_supported()}")
    else:
        print("  WARNING: CUDA is NOT available! Model will run on CPU.")
        print("  To use GPU acceleration, install PyTorch with CUDA:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 65)

    # Determine torch dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:  # auto
        if cuda_avail:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

    config = LRModuleConfig(
        base_model_name_or_path=args.model_name,
        output_dir=args.output_dir,
        lora=LoRAConfig(r=args.rank, lora_alpha=args.alpha)
    )

    # Apply command-line overrides to curriculum stages
    for stage in config.curriculum_stages:
        stage.batch_size = args.batch_size
        stage.gradient_accumulation_steps = args.grad_accum
        stage.epochs = args.epochs
        stage.max_train_samples = args.max_train_samples
        stage.max_val_samples = args.max_val_samples
        stage.max_length = args.max_length

    effective_batch = args.batch_size * args.grad_accum
    print(f"[Training Pipeline] Base Model:        {config.base_model_name_or_path}")
    print(f"[Training Pipeline] LoRA Rank:          {config.lora.r} | Alpha: {config.lora.lora_alpha}")
    print(f"[Training Pipeline] Output Dir:         {config.output_dir}")
    print(f"[Training Pipeline] Batch Size / Device:{args.batch_size} (Grad Accum: {args.grad_accum} -> Eff Batch: {effective_batch})")
    print(f"[Training Pipeline] Sample Limit/Stage: {args.max_train_samples:,} train / {args.max_val_samples:,} val")
    print(f"[Training Pipeline] Max Sequence Len:   {args.max_length}")
    print(f"[Training Pipeline] Target Device:      {args.device} (device_map='{args.device_map}')")
    print(f"[Training Pipeline] Precision:          {torch_dtype}")

    # Initialize Model and LoRA Adapter on GPU
    model, tokenizer = setup_internvl_lr_lora(
        model_name_or_path=config.base_model_name_or_path,
        lora_config=config.lora,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        gradient_checkpointing=args.gradient_checkpointing
    )

    # Initialize and execute Staged Curriculum Trainer
    trainer = StagedCurriculumTrainer(config=config, model=model, tokenizer=tokenizer)
    results = trainer.run_full_curriculum()

    print("\n" + "=" * 65)
    print("LR LoRA Training Finished Successfully on GPU!")
    print(f"Final adapter saved to: {results['final_adapter_dir']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
