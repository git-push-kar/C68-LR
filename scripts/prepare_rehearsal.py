import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr.data.normalizer import DatasetNormalizer
from polyvalent_lr.data.dedup import GroupAwareDataSplitter
from polyvalent_lr.data.sanitize import sanitize_examples, sanitize_report_string


def load_dolly(dataset_name: str, max_samples=None):
    from datasets import load_dataset
    print(f"[Rehearsal] Downloading/loading '{dataset_name}' from Hugging Face Hub...")
    ds = load_dataset(dataset_name, split="train")
    items = list(ds)
    if max_samples:
        items = items[:max_samples]
    return items


def main():
    parser = argparse.ArgumentParser(description="Prepare Dolly-15k rehearsal data for LR training")
    parser.add_argument("--dataset", type=str, default="databricks/databricks-dolly-15k",
                        help="HF dataset id for rehearsal/instruction data")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    raw_items = load_dolly(args.dataset, args.max_samples)

    examples = []
    for idx, item in enumerate(raw_items):
        ex = DatasetNormalizer.normalize_dolly(item, index=idx)
        if ex is not None:
            examples.append(ex)
    print(f"[Rehearsal] Normalized {len(examples):,} general-instruction examples from {len(raw_items):,} raw records.")

    clean, report = sanitize_examples(examples)
    print(sanitize_report_string("Dolly15k", report))

    train_ex, val_ex = GroupAwareDataSplitter.split_by_group(clean, val_ratio=args.val_ratio)
    print(f"[Rehearsal] Partitioned {len(clean):,} -> Train: {len(train_ex):,}, Val: {len(val_ex):,}")

    proc_path = Path(args.processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    train_file = proc_path / "Rehearsal_train.jsonl"
    val_file = proc_path / "Rehearsal_val.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_ex:
            f.write(ex.model_dump_json() + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        for ex in val_ex:
            f.write(ex.model_dump_json() + "\n")

    print(f"[Rehearsal Completed] Wrote {train_file} and {val_file}")


if __name__ == "__main__":
    main()
