import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr.config import DatasetSource, LogicTaskType, UnseenBenchmark
from polyvalent_lr.data.schema import LogicalExample, ProofStep
from polyvalent_lr.data.normalizer import DatasetNormalizer
from polyvalent_lr.data.dedup import GroupAwareDataSplitter


def generate_starter_datasets(raw_dir: Path, benchmark_dir: Path):
    """Generates starter seed/mock data to make the entire LR pipeline runnable immediately."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # 1. ProofWriter Starter Data
    pw_data = [
        {
            "id": "pw_001_1",
            "theory_id": "theory_animal_1",
            "theory": "If an animal has fur, then it is a mammal. If an animal is a mammal, then it is warm-blooded. Charlie has fur.",
            "question": "Charlie is warm-blooded.",
            "answer": "True",
            "proof": "Charlie has fur -> Charlie is a mammal; Charlie is a mammal -> Charlie is warm-blooded",
            "depth": 2
        },
        {
            "id": "pw_001_2",
            "theory_id": "theory_animal_1",
            "theory": "If an animal has fur, then it is a mammal. If an animal is a mammal, then it is warm-blooded. Charlie has fur.",
            "question": "Charlie is cold-blooded.",
            "answer": "False",
            "proof": "Charlie is warm-blooded -> not cold-blooded",
            "depth": 2
        },
        {
            "id": "pw_002_1",
            "theory_id": "theory_plant_1",
            "theory": "All flowering plants need sunlight. Roses are flowering plants. Cacti store water.",
            "question": "Roses need sunlight.",
            "answer": "True",
            "proof": "Roses are flowering plants; Flowering plants need sunlight -> Roses need sunlight",
            "depth": 1
        },
        {
            "id": "pw_002_2",
            "theory_id": "theory_plant_1",
            "theory": "All flowering plants need sunlight. Roses are flowering plants. Cacti store water.",
            "question": "Cacti need sunlight.",
            "answer": "Uncertain",
            "proof": "No direct rule linking cacti water storage to sunlight requirement.",
            "depth": 0
        }
    ]
    with open(raw_dir / "proofwriter_raw.json", "w", encoding="utf-8") as f:
        json.dump(pw_data, f, indent=2)

    # 2. FOLIO Starter Data
    folio_data = [
        {
            "story_id": "story_fol_101",
            "premises": ["All students who study pass the exam.", "Alex is a student.", "Alex studies."],
            "conclusion": "Alex passes the exam.",
            "label": "True",
            "premises_FOL": ["∀x (Student(x) ∧ Studies(x) → Passes(x))", "Student(Alex)", "Studies(Alex)"],
            "explanation": "From Alex being a student who studies, by universal instantiation Alex passes."
        },
        {
            "story_id": "story_fol_102",
            "premises": ["No mammal can breathe underwater.", "Whales are mammals."],
            "conclusion": "Whales can breathe underwater.",
            "label": "False",
            "premises_FOL": ["∀x (Mammal(x) → ¬BreathesUnderwater(x))", "Mammal(Whale)"],
            "explanation": "Universal negation implies whales cannot breathe underwater."
        }
    ]
    with open(raw_dir / "folio_raw.json", "w", encoding="utf-8") as f:
        json.dump(folio_data, f, indent=2)

    # 3. P-FOLIO Starter Data
    pfolio_data = [
        {
            "id": "pfolio_201",
            "story_id": "pstory_201",
            "premises": ["Every prime number greater than 2 is odd.", "3 is a prime number greater than 2."],
            "conclusion": "3 is odd.",
            "label": "True",
            "proof_trace": "Rule 1: ∀x (Prime(x) ∧ x > 2 → Odd(x)). Fact 2: Prime(3) ∧ 3 > 2. Modus Ponens yields Odd(3)."
        },
        {
            "id": "pfolio_202",
            "story_id": "pstory_202",
            "premises": ["If it rains, the grass is wet.", "The grass is wet."],
            "conclusion": "It rained.",
            "label": "Uncertain",
            "proof_trace": "Affirming the consequent is a logical fallacy. The grass could be wet due to sprinklers."
        }
    ]
    with open(raw_dir / "pfolio_raw.json", "w", encoding="utf-8") as f:
        json.dump(pfolio_data, f, indent=2)

    # 4. AbductionRules Starter Data
    abduction_data = [
        {
            "id": "abduct_301",
            "rule_context_id": "ab_ctx_1",
            "context_rules": ["Faulty alternator causes battery drain.", "Leaving headlights on causes battery drain."],
            "observations": ["Car battery is completely drained.", "Headlight switch was verified in OFF position."],
            "candidate_hypothesis": "The car has a faulty alternator.",
            "plausibility": "True",
            "abductive_chain": "Given headlights are off, the most plausible remaining cause for battery drain is a faulty alternator."
        }
    ]
    with open(raw_dir / "abduction_raw.json", "w", encoding="utf-8") as f:
        json.dump(abduction_data, f, indent=2)

    # 5. Unseen Benchmarks (Quarantined)
    bench_data = {
        UnseenBenchmark.LOGICBENCH: [
            LogicalExample(
                id="lb_001",
                group_id="lb_grp_1",
                source=DatasetSource.PROOFWRITER,
                task_type=LogicTaskType.DEDUCTION,
                premises=["All metals conduct electricity.", "Copper is a metal."],
                hypothesis_or_query="Copper conducts electricity.",
                label="True",
                proof_trace_text="Universal instantiation on metals implies copper conducts electricity."
            ),
            LogicalExample(
                id="lb_002",
                group_id="lb_grp_2",
                source=DatasetSource.PROOFWRITER,
                task_type=LogicTaskType.DEDUCTION,
                premises=["No reptile has feathers.", "Snakes are reptiles."],
                hypothesis_or_query="Snakes have feathers.",
                label="False",
                proof_trace_text="Snakes are reptiles, and no reptiles have feathers, so snakes do not have feathers."
            )
        ],
        UnseenBenchmark.MULTI_LOGIEVAL: [
            LogicalExample(
                id="mle_001",
                group_id="mle_grp_1",
                source=DatasetSource.FOLIO,
                task_type=LogicTaskType.DEDUCTION,
                premises=["If A then B.", "If B then C.", "If C then D.", "A is true."],
                hypothesis_or_query="D is true.",
                label="True",
                proof_trace_text="Multi-step chain: A -> B -> C -> D."
            )
        ],
        UnseenBenchmark.LOGICNLI: [
            LogicalExample(
                id="lnli_001",
                group_id="lnli_grp_1",
                source=DatasetSource.FOLIO,
                task_type=LogicTaskType.DEDUCTION,
                premises=["Either the server is active or backup is initiated.", "The server is active."],
                hypothesis_or_query="Backup is initiated.",
                label="Uncertain",
                proof_trace_text="Inclusive OR allows both to be true, so backup initiation is undetermined."
            )
        ]
    }

    for bench_name, examples in bench_data.items():
        bench_file = benchmark_dir / f"{bench_name.value}.jsonl"
        with open(bench_file, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(ex.model_dump_json() + "\n")
    print("[Data Prep] Starter datasets and benchmark suites created.")


def download_online_datasets(raw_dir: Path):
    """Attempts to auto-download raw datasets from Hugging Face Hub and official repositories."""
    print("[Data Download] Fetching official datasets from online repositories...")
    try:
        from datasets import load_dataset
        
        # 1. Download FOLIO from Hugging Face
        print("[Data Download] Loading FOLIO from Hugging Face (yale-nlp/FOLIO)...")
        try:
            folio_hf = load_dataset("yale-nlp/FOLIO", split="train")
            folio_items = []
            for item in folio_hf:
                folio_items.append({
                    "story_id": item.get("story_id", item.get("example_id")),
                    "premises": item.get("premises", []),
                    "conclusion": item.get("conclusion", ""),
                    "label": item.get("label", ""),
                    "premises_FOL": item.get("premises_FOL", []),
                    "explanation": item.get("explanation", "")
                })
            with open(raw_dir / "folio_raw.json", "w", encoding="utf-8") as f:
                json.dump(folio_items, f, indent=2)
            print(f"[Data Download] Saved {len(folio_items)} FOLIO examples.")
        except Exception as e:
            print(f"[Data Download] Note: Could not auto-download FOLIO via HF ({e}).")

    except ImportError:
        print("[Data Download] 'datasets' package not installed or network unavailable.")


def main():
    parser = argparse.ArgumentParser(description="Prepare and preprocess Polyvalent LR datasets")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Processed output directory")
    parser.add_argument("--benchmark_dir", type=str, default="data/benchmarks", help="Unseen benchmark directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--download_online", action="store_true", help="Auto-download datasets from Hugging Face / online repos")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    proc_path = Path(args.processed_dir)
    bench_path = Path(args.benchmark_dir)

    proc_path.mkdir(parents=True, exist_ok=True)
    raw_path.mkdir(parents=True, exist_ok=True)

    if args.download_online:
        download_online_datasets(raw_path)

    if not any(raw_path.iterdir()):
        print("[Data Prep] Raw data directory empty. Generating initial dataset scaffolds...")
        generate_starter_datasets(raw_path, bench_path)

    # 1. Normalize ProofWriter (supports single json or directory of meta-*.jsonl)
    pw_examples = []
    pw_sources = list(raw_path.glob("**/proofwriter*.json*")) + list(raw_path.glob("**/meta-*.jsonl"))
    for pw_f in set(pw_sources):
        if "meta-abduct" in pw_f.name:
            continue
        try:
            with open(pw_f, "r", encoding="utf-8") as f:
                if pw_f.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            res = DatasetNormalizer.normalize_proofwriter(json.loads(line))
                            if isinstance(res, list):
                                pw_examples.extend(res)
                            else:
                                pw_examples.append(res)
                else:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        res = DatasetNormalizer.normalize_proofwriter(item)
                        if isinstance(res, list):
                            pw_examples.extend(res)
                        else:
                            pw_examples.append(res)
        except Exception as e:
            print(f"[Warning] Error reading {pw_f}: {e}")

    # 2. Normalize FOLIO
    folio_examples = []
    folio_sources = list(raw_path.glob("**/folio*.json*"))
    for folio_f in set(folio_sources):
        try:
            with open(folio_f, "r", encoding="utf-8") as f:
                if folio_f.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            folio_examples.append(DatasetNormalizer.normalize_folio(json.loads(line)))
                else:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        folio_examples.append(DatasetNormalizer.normalize_folio(item))
        except Exception as e:
            print(f"[Warning] Error reading {folio_f}: {e}")

    # 3. Normalize P-FOLIO
    pfolio_examples = []
    pfolio_sources = list(raw_path.glob("**/pfolio*.json*"))
    for pfolio_f in set(pfolio_sources):
        try:
            with open(pfolio_f, "r", encoding="utf-8") as f:
                if pfolio_f.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            pfolio_examples.append(DatasetNormalizer.normalize_p_folio(json.loads(line)))
                else:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        pfolio_examples.append(DatasetNormalizer.normalize_p_folio(item))
        except Exception as e:
            print(f"[Warning] Error reading {pfolio_f}: {e}")

    # 4. Normalize Abduction (AbductionRules + ProofWriter meta-abduct-*.jsonl)
    abduct_examples = []
    abduct_dir = raw_path / "abduction_rules_dataset"
    abduct_sources = (
        list(raw_path.glob("**/abduction*.json*"))
        + list(raw_path.glob("**/meta-abduct*.jsonl"))
        + (list(abduct_dir.glob("**/*.jsonl")) if abduct_dir.exists() else [])
    )
    for abduct_f in set(abduct_sources):
        try:
            with open(abduct_f, "r", encoding="utf-8") as f:
                if "meta-abduct" in abduct_f.name:
                    for line in f:
                        if line.strip():
                            abduct_examples.extend(DatasetNormalizer.normalize_proofwriter_abduct(json.loads(line)))
                elif abduct_f.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            res = DatasetNormalizer.normalize_abduction_rules(json.loads(line))
                            if isinstance(res, list):
                                abduct_examples.extend(res)
                            else:
                                abduct_examples.append(res)
                else:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        res = DatasetNormalizer.normalize_abduction_rules(item)
                        if isinstance(res, list):
                            abduct_examples.extend(res)
                        else:
                            abduct_examples.append(res)
        except Exception as e:
            print(f"[Warning] Error reading {abduct_f}: {e}")

    # Check FOLIO vs P-FOLIO overlap
    overlap_report = GroupAwareDataSplitter.check_folio_pfolio_overlap(folio_examples, pfolio_examples)
    print(f"[Data Check] FOLIO / P-FOLIO Overlap: {overlap_report['overlap_count']} overlapping premises found.")

    # Split and save datasets
    dataset_map = {
        DatasetSource.PROOFWRITER: pw_examples,
        DatasetSource.FOLIO: folio_examples,
        DatasetSource.P_FOLIO: pfolio_examples,
        DatasetSource.ABDUCTION_RULES: abduct_examples
    }

    for source, examples in dataset_map.items():
        if not examples:
            continue
        train_ex, val_ex = GroupAwareDataSplitter.split_by_group(examples, val_ratio=args.val_ratio)
        print(f"[{source.value}] Partitioned {len(examples)} examples -> Train: {len(train_ex)}, Val: {len(val_ex)}")

        with open(proc_path / f"{source.value}_train.jsonl", "w", encoding="utf-8") as f:
            for ex in train_ex:
                f.write(ex.model_dump_json() + "\n")

        with open(proc_path / f"{source.value}_val.jsonl", "w", encoding="utf-8") as f:
            for ex in val_ex:
                f.write(ex.model_dump_json() + "\n")

    print(f"[Data Prep Completed] Processed files written to {proc_path}")


if __name__ == "__main__":
    main()
