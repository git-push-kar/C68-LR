import os
import sys
import shutil
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# expandable_segments is unstable on Windows (crashes the CUDA caching allocator); force off
# before torch reads PYTORCH_CUDA_ALLOC_CONF at import time.
_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments:true" in _conf.replace(" ", "").lower():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import torch
from polyvalent_lr.config import DEFAULT_BASE_MODEL_PATH

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _signature(path: Path):
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


def _copy_base(base: Path, dest: Path) -> None:
    """Copies every file/dir of the base model into dest (base itself is never written to)."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def build_adapter_variant(base: Path, adapter: Path, dest: Path) -> Path:
    """Full copy of the base model plus the LoRA adapter in its own `adapter/` subfolder.

    The adapter lives in a subfolder so the base copy stays clean (a top-level
    adapter_config.json makes transformers auto-attach the adapter on load).
    """
    print(f"[Variants] (1/2) Base + adapter copy -> {dest}")
    _copy_base(base, dest)

    # Remove any stale top-level adapter files from an earlier layout.
    for fname in ADAPTER_FILES:
        stale = dest / fname
        if stale.exists():
            stale.unlink()

    adapter_sub = dest / "adapter"
    adapter_sub.mkdir(parents=True, exist_ok=True)
    for fname in ADAPTER_FILES:
        src = adapter / fname
        if not src.exists():
            raise FileNotFoundError(f"adapter is missing {fname}: {src}")
        shutil.copy2(src, adapter_sub / fname)
    print(f"[Variants]      copied base + adapter/{', adapter/'.join(ADAPTER_FILES)}")
    return dest


def build_merged_variant(base: Path, adapter: Path, dest: Path, device_map: str) -> Path:
    """Full copy of the base model with the LoRA adapter merged into its weights."""
    print(f"[Variants] (2/2) Merged copy -> {dest}")
    _copy_base(base, dest)

    from peft import PeftModel
    from polyvalent_lr.models.loader import load_internvl_model_and_tokenizer

    print("[Variants]      loading the base copy and attaching adapter for merge...")
    model, tokenizer = load_internvl_model_and_tokenizer(
        model_name_or_path=str(dest),
        device_map=device_map,
        trust_remote_code=True,
        is_eval=True,
    )
    peft_model = PeftModel.from_pretrained(model, str(adapter))
    print("[Variants]      merging LoRA layers (W = W_base + (alpha/r) * B @ A)...")
    merged_model = peft_model.merge_and_unload()

    print("[Variants]      saving merged weights over the copy...")
    merged_model.save_pretrained(str(dest), safe_serialization=True)
    tokenizer.save_pretrained(str(dest))

    del merged_model, peft_model, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Variants]      merged model saved (remote code / tokenizer kept from the copy)")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Create two self-contained variants of the LR model from one untouched base: "
                    "(a) base copy + LoRA adapter, (b) base copy with the adapter merged into the weights."
    )
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL_PATH, help="Source base model (never modified)")
    parser.add_argument("--adapter_dir", type=str, default="checkpoints/internvl3_2b_lr_v2/final_lr_lora", help="Trained LR adapter")
    parser.add_argument("--adapter_output", type=str, default="models/lr_adapter", help="Output dir for base + adapter variant")
    parser.add_argument("--merged_output", type=str, default="models/lr_merged", help="Output dir for merged variant")
    parser.add_argument("--device_map", type=str, default="auto" if torch.cuda.is_available() else "cpu", help="device_map used while merging")
    parser.add_argument("--skip_adapter", action="store_true", help="Skip building the base + adapter copy")
    parser.add_argument("--skip_merged", action="store_true", help="Skip building the merged copy")
    parser.add_argument("--no_verify", action="store_true", help="Skip the numerical equivalence check")
    parser.add_argument("--verify_tolerance", type=float, default=None,
                        help="Max logit diff for the equivalence check (default: 3.0 in bf16, 1e-3 in fp32)")
    parser.add_argument("--verify_fp32", action="store_true",
                        help="Run the equivalence check in float32 for a tight comparison")
    args = parser.parse_args()

    base = Path(args.base_model)
    adapter = Path(args.adapter_dir)
    if not base.exists():
        parser.error(f"base model not found: {base}")
    if not (adapter / "adapter_model.safetensors").exists():
        parser.error(f"adapter not found: {adapter}")

    base_weights = base / "model.safetensors"
    before = _signature(base_weights) if base_weights.exists() else None

    print("=" * 68)
    print("  BUILD MODEL VARIANTS (base stays untouched)")
    print("=" * 68)
    print(f"  Base model  (read-only): {base}")
    print(f"  LoRA adapter           : {adapter}")
    print("=" * 68)

    if not args.skip_adapter:
        build_adapter_variant(base, adapter, Path(args.adapter_output))
    if not args.skip_merged:
        build_merged_variant(base, adapter, Path(args.merged_output), args.device_map)

    # Guard-rail: confirm the source base was not modified.
    after = _signature(base_weights) if base_weights.exists() else None
    if before is not None and before != after:
        raise RuntimeError("Base model weights were modified! (unexpected)")
    print(f"[Variants] Base model untouched: model.safetensors signature {'unchanged' if before == after else 'MISSING'}")

    if not args.no_verify and not args.skip_merged:
        from polyvalent_lr.models.weight_merger import WeightMerger
        # bf16 merges round; a 1e-3 tolerance is only meaningful in fp32.
        tolerance = 1e-3 if args.verify_tolerance is None and args.verify_fp32 else (
            args.verify_tolerance if args.verify_tolerance is not None else 3.0
        )
        print(f"[Variants] Running numerical equivalence check (tolerance={tolerance})...")
        # Compares logits of (original base + adapter) against the merged copy.
        res = WeightMerger.verify_numerical_equivalence(
            base_model_path=str(base),
            adapter_path=str(adapter),
            merged_model_path=args.merged_output,
            torch_dtype=torch.float32 if args.verify_fp32 else None,
            tolerance=tolerance,
        )
        if res["is_equivalent"]:
            print("[Variants] Equivalence check PASSED (base+adapter == merged within tolerance).")
        else:
            print(f"[Variants] WARNING: max logit diff {res['max_absolute_difference']:.6f} exceeds tolerance.")

    print("=" * 68)
    print("[Variants] Done.")
    print(f"  Base + adapter : {args.adapter_output}")
    print(f"  Merged         : {args.merged_output}")
    print("=" * 68)


if __name__ == "__main__":
    main()
