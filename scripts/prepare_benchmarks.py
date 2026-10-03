import sys
import os
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr.data.normalizer import DatasetNormalizer
from polyvalent_lr.data.schema import LogicalExample


def parse_logicbench_directory(logicbench_root: Path, benchmark_dest: Path) -> int:
    """Scans and parses LogicBench(Eval) BQA & MCQA subdirectories into data/benchmarks/LogicBench.jsonl.
    
    Strict Isolation: Ignores LogicBench(Aug) completely.
    """
    eval_dir = logicbench_root / "LogicBench(Eval)" if (logicbench_root / "LogicBench(Eval)").exists() else logicbench_root
    if not eval_dir.exists():
        return 0

    dest_file = benchmark_dest / "LogicBench.jsonl"
    examples: List[LogicalExample] = []
    
    print(f"[LogicBench] Scanning {eval_dir}...")
    for json_path in eval_dir.glob("**/*.json"):
        # Ignore Augmentation training folders
        if "LogicBench(Aug)" in str(json_path):
            continue
        
        rel_parts = json_path.relative_to(eval_dir).parts
        eval_type = rel_parts[0] if len(rel_parts) > 1 else "Eval" # BQA / MCQA
        logic_type = rel_parts[1] if len(rel_parts) > 2 else "generic" # prop / fol / nm
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                items = content if isinstance(content, list) else [content]
                for it in items:
                    ex = DatasetNormalizer.normalize_logicbench(it, metadata={"eval_type": eval_type, "logic_type": logic_type})
                    examples.append(ex)
        except Exception as e:
            print(f"[Warning] Error parsing {json_path}: {e}")

    if examples:
        with open(dest_file, "w", encoding="utf-8") as out:
            for ex in examples:
                out.write(ex.model_dump_json() + "\n")
        print(f"[LogicBench Success] Formatted {len(examples)} evaluation examples into {dest_file}")
    return len(examples)


def parse_multi_logieval_directory(multi_logieval_root: Path, benchmark_dest: Path) -> int:
    """Scans and parses Multi-LogiEval (d1_Data to d5_Data & multivariable_fol_Data) into data/benchmarks/Multi-LogiEval.jsonl."""
    if not multi_logieval_root.exists():
        return 0

    dest_file = benchmark_dest / "Multi-LogiEval.jsonl"
    examples: List[LogicalExample] = []

    print(f"[Multi-LogiEval] Scanning {multi_logieval_root}...")
    # Scan d1_Data through d5_Data and multivariable_fol_Data
    for json_path in multi_logieval_root.glob("**/*.json"):
        rel_parts = json_path.relative_to(multi_logieval_root).parts
        depth_folder = rel_parts[0] if len(rel_parts) > 0 else "unknown" # e.g. d1_Data, d2_Data
        logic_type = rel_parts[1] if len(rel_parts) > 1 else "generic"
        
        depth_val = 1
        if depth_folder.startswith("d") and "_" in depth_folder:
            try:
                depth_val = int(depth_folder.split("_")[0][1:])
            except ValueError:
                depth_val = 1
        elif "multivariable" in depth_folder:
            depth_val = 6

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                items = content if isinstance(content, list) else [content]
                for it in items:
                    ex = DatasetNormalizer.normalize_multi_logieval(it, metadata={"depth_folder": depth_folder, "depth": depth_val, "logic_type": logic_type})
                    examples.append(ex)
        except Exception as e:
            print(f"[Warning] Error parsing {json_path}: {e}")

    if examples:
        with open(dest_file, "w", encoding="utf-8") as out:
            for ex in examples:
                out.write(ex.model_dump_json() + "\n")
        print(f"[Multi-LogiEval Success] Formatted {len(examples)} multi-step evaluation examples into {dest_file}")
    return len(examples)


def parse_logicnli_directory(logicnli_root: Path, benchmark_dest: Path) -> int:
    """Scans and parses LogicNLI (test_language.json & test_logic.json) into data/benchmarks/LogicNLI.jsonl.
    
    Strict Isolation: Ignores train_*.json and dev_*.json.
    """
    if not logicnli_root.exists():
        return 0

    dest_file = benchmark_dest / "LogicNLI.jsonl"
    examples: List[LogicalExample] = []

    print(f"[LogicNLI] Scanning {logicnli_root} for test files...")
    # Target only test files
    for test_path in list(logicnli_root.glob("**/test_*.json")) + list(logicnli_root.glob("**/test.json*")):
        variant = "language" if "language" in test_path.name else ("logic" if "logic" in test_path.name else "test")
        try:
            with open(test_path, "r", encoding="utf-8") as f:
                if test_path.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            ex = DatasetNormalizer.normalize_logicnli(json.loads(line), metadata={"variant": variant})
                            examples.append(ex)
                else:
                    content = json.load(f)
                    items = content if isinstance(content, list) else [content]
                    for it in items:
                        ex = DatasetNormalizer.normalize_logicnli(it, metadata={"variant": variant})
                        examples.append(ex)
        except Exception as e:
            print(f"[Warning] Error parsing {test_path}: {e}")

    if examples:
        with open(dest_file, "w", encoding="utf-8") as out:
            for ex in examples:
                out.write(ex.model_dump_json() + "\n")
        print(f"[LogicNLI Success] Formatted {len(examples)} test examples into {dest_file}")
    return len(examples)


def main():
    raw_dir = Path("data/raw")
    benchmark_dir = Path("data/benchmarks")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Polyvalent LR: Quarantined Benchmark Auto-Discovery & Ingestion")
    print("=" * 70)

    # 1. LogicBench
    lb_root = raw_dir / "LogicBench" if (raw_dir / "LogicBench").exists() else raw_dir
    parse_logicbench_directory(lb_root, benchmark_dir)

    # 2. Multi-LogiEval
    mle_root = raw_dir / "Multi-LogiEval" if (raw_dir / "Multi-LogiEval").exists() else raw_dir
    parse_multi_logieval_directory(mle_root, benchmark_dir)

    # 3. LogicNLI
    lnli_root = raw_dir / "LogicNLI" if (raw_dir / "LogicNLI").exists() else raw_dir
    parse_logicnli_directory(lnli_root, benchmark_dir)

    print("\n" + "=" * 70)
    print("Quarantined Benchmark Suite Summary (data/benchmarks/):")
    for b_file in benchmark_dir.glob("*.jsonl"):
        print(f" • {b_file.name} -> {b_file.stat().st_size:,} bytes")
    print("=" * 70)


if __name__ == "__main__":
    main()
