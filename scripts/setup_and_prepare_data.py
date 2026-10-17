import sys
import subprocess
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_real_datasets import main as download_main
from scripts.prepare_data import main as prepare_main
from scripts.prepare_rehearsal import main as rehearsal_main


def main():
    print("\n" + "=" * 70)
    print("Polyvalent LR: Automated Dataset Download & Preparation Pipeline")
    print("=" * 70)
    
    # 1. Download official datasets
    print("\n[Step 1/3] Fetching and extracting official datasets...")
    download_main()
    
    # 2. Normalize, deduplicate, and partition datasets
    print("\n[Step 2/3] Normalizing, deduplicating, and creating group-aware splits...")
    prepare_main()

    # 3. Prepare Dolly-15k rehearsal data (optional; needs network/HF Hub)
    print("\n[Step 3/3] Preparing Dolly-15k rehearsal data...")
    try:
        rehearsal_main()
    except Exception as e:
        print(f"[Warning] Rehearsal preparation skipped ({e}). "
              f"Run 'python scripts/prepare_rehearsal.py' manually if needed.")

    print("\n" + "=" * 70)
    print("[Complete] All datasets downloaded, normalized, and split successfully!")
    print("Ready for Stage 1-3 LoRA curriculum training on InternVL3-2B.")
    print("=" * 70)


if __name__ == "__main__":
    main()
