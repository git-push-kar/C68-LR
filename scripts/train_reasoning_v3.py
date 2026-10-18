"""Stage 2/3 — SFT (and rejection-sampled SFT) for the v3 reasoning repair.

Trains a rank-64 LoRA on the InternVL3-2B language model only (vision untouched),
supervising assistant tokens including EOS, evaluating every ``eval_steps`` and
keeping the best adapter.

Examples
    python scripts/train_reasoning_v3.py --dry_run
    python scripts/train_reasoning_v3.py                       # Stage 2
    python scripts/train_reasoning_v3.py --init_adapter checkpoints/internvl3_2b_reasoning_v3/sft \
        --train_file data/v3/rejection/accepted.jsonl --learning_rate 5e-5 \
        --output_dir checkpoints/internvl3_2b_reasoning_v3/rs   # Stage 3
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments:true" in _alloc.replace(" ", "").lower():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import torch  # noqa: E402

from polyvalent_lr import config_v3  # noqa: E402
from polyvalent_lr.models.lora_setup import setup_internvl_lr_lora  # noqa: E402
from polyvalent_lr.reasoning_v3.data import (  # noqa: E402
    AssistantMaskingCollator, SupervisedDataset, read_jsonl,
)


def main() -> None:
    sft = config_v3.SFTV3()
    ap = argparse.ArgumentParser(description="v3 reasoning SFT.")
    ap.add_argument("--model_name", type=str, default=config_v3.BASE_MODEL_PATH)
    ap.add_argument("--output_dir", type=str, default=str(config_v3.STAGE2_ADAPTER_DIR))
    ap.add_argument("--train_file", type=str, default=str(config_v3.MIX_DIR / "train.jsonl"))
    ap.add_argument("--eval_file", type=str, default=str(config_v3.MIX_DIR / "holdout.jsonl"))
    ap.add_argument("--rank", type=int, default=config_v3.LoRAV3().r)
    ap.add_argument("--alpha", type=int, default=config_v3.LoRAV3().lora_alpha)
    ap.add_argument("--learning_rate", type=float, default=sft.learning_rate)
    ap.add_argument("--epochs", type=int, default=sft.num_epochs)
    ap.add_argument("--batch_size", type=int, default=sft.per_device_batch_size)
    ap.add_argument("--grad_accum", type=int, default=sft.grad_accum)
    ap.add_argument("--max_length", type=int, default=sft.max_length)
    ap.add_argument("--save_steps", type=int, default=sft.save_steps)
    ap.add_argument("--eval_steps", type=int, default=sft.eval_steps)
    ap.add_argument("--init_adapter", type=str, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    train_examples = read_jsonl(Path(args.train_file))
    eval_examples = read_jsonl(Path(args.eval_file))
    if args.max_train_samples:
        train_examples = train_examples[: args.max_train_samples]
    if not train_examples:
        raise SystemExit(f"no training examples in {args.train_file}")

    eff = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_examples) // eff)
    print("=" * 70)
    print("  v3 REASONING SFT")
    print("=" * 70)
    print(f"  base model      : {args.model_name}")
    print(f"  lora            : r={args.rank} alpha={args.alpha} dropout={config_v3.LoRAV3().lora_dropout}")
    print(f"  train / eval    : {len(train_examples):,} / {len(eval_examples):,}")
    print(f"  batch/accum/eff : {args.batch_size}/{args.grad_accum}/{eff}")
    print(f"  epochs / steps  : {args.epochs} / ~{steps_per_epoch * args.epochs}")
    print(f"  lr / warmup     : {args.learning_rate} / {sft.warmup_ratio}")
    print(f"  eval/save every : {args.eval_steps} steps -> {args.output_dir}")
    print(f"  init adapter    : {args.init_adapter}")
    print("=" * 70)
    if args.dry_run:
        return

    model, tokenizer = setup_internvl_lr_lora(
        model_name_or_path=args.model_name,
        lora_config=config_v3.LoRAV3(r=args.rank, lora_alpha=args.alpha).to_lora_config(),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        gradient_checkpointing=sft.gradient_checkpointing,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.init_adapter:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        state = load_file(str(Path(args.init_adapter) / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, state)
        print(f"[init_adapter] loaded {len(state)} tensors from {args.init_adapter}")

    from transformers import Trainer, TrainingArguments

    args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=sft.warmup_ratio,
        lr_scheduler_type=sft.lr_scheduler,
        max_grad_norm=sft.max_grad_norm,
        bf16=sft.bf16,
        tf32=sft.tf32,
        gradient_checkpointing=sft.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=sft.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=sft.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=sft.group_by_length,
        dataloader_num_workers=sft.dataloader_num_workers,
        seed=sft.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    training_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SupervisedDataset(train_examples, tokenizer, args.max_length),
        eval_dataset=SupervisedDataset(eval_examples, tokenizer, args.max_length) if eval_examples else None,
        data_collator=AssistantMaskingCollator(tokenizer=tokenizer),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[v3] adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
