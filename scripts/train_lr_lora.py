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
    parser = argparse.ArgumentParser(description="Train Staged LR LoRA on InternVL3-2B")
    parser.add_argument("--config_file", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-2B", help="Base model path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/internvl3_2b_lr", help="Adapter output dir")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    args = parser.parse_args()

    config = LRModuleConfig(
        base_model_name_or_path=args.model_name,
        output_dir=args.output_dir,
        lora=LoRAConfig(r=args.rank, lora_alpha=args.alpha)
    )

    print(f"[Training Pipeline] Base Model: {config.base_model_name_or_path}")
    print(f"[Training Pipeline] LoRA Rank: {config.lora.r} | Alpha: {config.lora.lora_alpha}")
    print(f"[Training Pipeline] Output Dir: {config.output_dir}")

    # Initialize Model and LoRA Adapter
    model, tokenizer = setup_internvl_lr_lora(
        model_name_or_path=config.base_model_name_or_path,
        lora_config=config.lora,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )

    # Initialize and execute Staged Curriculum Trainer
    trainer = StagedCurriculumTrainer(config=config, model=model, tokenizer=tokenizer)
    results = trainer.run_full_curriculum()

    print("\n" + "="*50)
    print("LR LoRA Training Finished Successfully!")
    print(f"Final adapter saved to: {results['final_adapter_dir']}")
    print("="*50)


if __name__ == "__main__":
    main()
