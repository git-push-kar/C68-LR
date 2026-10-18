import os
import sys
import argparse
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# expandable_segments is unstable on Windows: it can trip
# "!handles_.at(i) INTERNAL ASSERT FAILED at c10/cuda/CUDACachingAllocator.cpp"
# during model.to(cuda). Force it off before any CUDA allocation happens.
_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments:true" in _alloc_conf.replace(" ", "").lower():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    print("[CUDA] Disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments (unstable on Windows).")

import torch

from polyvalent_lr.config import LRModuleConfig, LoRAConfig, DEFAULT_BASE_MODEL_PATH
from polyvalent_lr.models.lora_setup import setup_internvl_lr_lora
from polyvalent_lr.training.trainer import StagedCurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Staged LR LoRA on InternVL3-2B on GPU (RTX A5000 / Ampere)")
    parser.add_argument("--config_file", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model_name", type=str, default=DEFAULT_BASE_MODEL_PATH, help="Base model path (local dir or HF repo id)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/internvl3_2b_lr_v2", help="Adapter output dir")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size (default: 32 for 24GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs per stage (default: 1)")
    parser.add_argument("--full_dataset", action="store_true", default=False, help="Train on 100%% full dataset (all 1.76M+ examples)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Override representative training samples per stage (default: None -> use each stage's configured cap; ignored if --full_dataset)")
    parser.add_argument("--max_val_samples", type=int, default=1000, help="Max validation samples per stage (default: 1000)")
    parser.add_argument("--max_length", type=int, default=384, help="Max sequence length (default: 384)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device (cuda/cpu)")
    parser.add_argument("--device_map", type=str, default=None, help="device_map for transformers (default: None for direct high-throughput GPU training)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "fp16", "float32"], help="Model torch_dtype")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing (disable by default on 24GB VRAM for 2x speedup)")
    parser.add_argument("--only_stage", type=int, default=None, help="Run only this curriculum stage_id (e.g. 3 to redo the final stage)")
    parser.add_argument("--init_adapter", type=str, default=None, help="Path to an existing LoRA adapter dir to initialize weights from (resume/continue training)")
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint every N steps instead of every epoch (also sets eval cadence)")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Print the resolved curriculum stages and estimated step counts, then exit without loading the model")
    args = parser.parse_args()

    # Hardware & compute diagnostics
    cuda_avail = torch.cuda.is_available()
    print("=" * 65)
    print("  POLYVALENT LOGICAL REASONING (LR) - HIGH-THROUGHPUT GPU TRAINING")
    print("=" * 65)
    print(f"  PyTorch Version:       {torch.__version__}")
    print(f"  CUDA Available:        {cuda_avail}")
    if cuda_avail:
        print(f"  CUDA Device:           {torch.cuda.get_device_name(0)} (Device 0)")
        print(f"  CUDA Device Count:     {torch.cuda.device_count()}")
        print(f"  Total GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"  bfloat16 Supported:    {torch.cuda.is_bf16_supported()}")
        print(f"  Tensor Core TF32:      Enabled")
        print(f"  Length Grouping:       Enabled (group_by_length=True)")
        print(f"  Gradient Checkpointing:{args.gradient_checkpointing} (Disabled for maximum GPU speed)")
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
        if args.full_dataset:
            stage.max_train_samples = None
        elif args.max_train_samples is not None:
            stage.max_train_samples = args.max_train_samples
        stage.max_val_samples = args.max_val_samples
        stage.max_length = args.max_length

    # Restrict to a single stage if requested (resume/redo one stage only)
    if args.only_stage is not None:
        config.curriculum_stages = [s for s in config.curriculum_stages if s.stage_id == args.only_stage]
        if not config.curriculum_stages:
            parser.error(f"--only_stage {args.only_stage} does not match any curriculum stage.")

    effective_batch = args.batch_size * args.grad_accum
    if args.full_dataset:
        dataset_mode = "100% Full Dataset (All Examples)"
    elif args.max_train_samples is not None:
        dataset_mode = f"Balanced Sampling ({args.max_train_samples:,} per stage)"
    else:
        dataset_mode = "Per-stage configured caps (50k/50k/40k)"
    print(f"[Training Pipeline] Base Model:        {config.base_model_name_or_path}")
    print(f"[Training Pipeline] LoRA Rank:          {config.lora.r} | Alpha: {config.lora.lora_alpha}")
    print(f"[Training Pipeline] Output Dir:         {config.output_dir}")
    print(f"[Training Pipeline] Dataset Mode:       {dataset_mode}")
    print(f"[Training Pipeline] Batch Size / Device:{args.batch_size} (Grad Accum: {args.grad_accum} -> Eff Batch: {effective_batch})")
    print(f"[Training Pipeline] Max Sequence Len:   {args.max_length}")
    print(f"[Training Pipeline] Target Device:      {args.device} (device_map={repr(args.device_map)})")
    print(f"[Training Pipeline] Precision:          {torch_dtype}")

    # Dry run: show resolved stages and estimated step counts, then exit (no model load / no training)
    if args.dry_run:
        from math import ceil
        data_dir = Path(config.data.processed_data_dir)

        def _count_lines(path: Path) -> int:
            if not path.exists():
                return 0
            with open(path, "r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())

        print("\n" + "=" * 65)
        print("  DRY RUN - resolved curriculum (no model loaded, no training)")
        print("=" * 65)
        for stage in config.curriculum_stages:
            available = 0
            for ds in stage.datasets:
                available += _count_lines(data_dir / f"{ds.value}_train.jsonl")
            cap = stage.max_train_samples if (stage.max_train_samples and stage.max_train_samples > 0) else available
            n_base = min(cap, available) if available else 0
            ratio = stage.rehearsal_ratio or 0.0
            n_reh = int(n_base * ratio / max(1e-6, 1.0 - ratio))
            total = n_base + n_reh
            eff = max(1, stage.batch_size * stage.gradient_accumulation_steps)
            steps_per_epoch = ceil(total / eff) if total else 0
            print(f"\n  Stage {stage.stage_id}: {stage.name}")
            print(f"    datasets            : {[d.value for d in stage.datasets]}")
            print(f"    available / cap     : {available:,} / {stage.max_train_samples}")
            print(f"    base + rehearsal    : {n_base:,} + {n_reh:,} = {total:,} (ratio {ratio:.0%})")
            print(f"    batch / accum / eff : {stage.batch_size} / {stage.gradient_accumulation_steps} / {eff}")
            print(f"    max_length / lr     : {stage.max_length} / {stage.learning_rate}")
            print(f"    epochs / steps      : {stage.epochs} / {steps_per_epoch:,} per epoch ({steps_per_epoch * stage.epochs:,} total)")
        if args.init_adapter:
            print(f"\n  init_adapter          : {args.init_adapter}")
        if args.save_steps:
            print(f"  checkpoint cadence    : every {args.save_steps} steps")
        print("=" * 65)
        return

    # Initialize Model and LoRA Adapter on GPU
    model, tokenizer = setup_internvl_lr_lora(
        model_name_or_path=config.base_model_name_or_path,
        lora_config=config.lora,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        gradient_checkpointing=args.gradient_checkpointing
    )

    # Optionally initialize adapter weights from an existing checkpoint (resume/continue)
    if args.init_adapter:
        from safetensors.torch import load_file
        adapter_file = Path(args.init_adapter) / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(f"--init_adapter has no adapter_model.safetensors: {adapter_file}")
        adapter_state = load_file(str(adapter_file))
        try:
            from peft import set_peft_model_state_dict
        except ImportError:
            from peft.utils.save_and_load import set_peft_model_state_dict
        load_result = set_peft_model_state_dict(model, adapter_state)
        print(
            f"[Init Adapter] Loaded {len(adapter_state)} tensors from {adapter_file} "
            f"(unexpected_keys={len(getattr(load_result, 'unexpected_keys', []) or [])})"
        )

    # Initialize and execute Staged Curriculum Trainer
    trainer = StagedCurriculumTrainer(
        config=config, model=model, tokenizer=tokenizer, save_steps=args.save_steps
    )
    results = trainer.run_full_curriculum(stage_ids=[s.stage_id for s in config.curriculum_stages])

    print("\n" + "=" * 65)
    print("LR LoRA Training Finished Successfully on GPU!")
    print(f"Final adapter saved to: {results['final_adapter_dir']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
