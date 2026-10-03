import sys
import os
import json
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_url(url: str, dest_path: Path) -> bool:
    """Downloads a file directly from a URL."""
    print(f"[Fetching] {url} ...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[Saved] {dest_path.name} ({os.path.getsize(dest_path):,} bytes)")
        return True
    except Exception as e:
        print(f"[Error] Failed to fetch {url}: {e}")
        return False


def fetch_folio(raw_dir: Path):
    """Downloads official FOLIO dataset directly from Yale-LILY GitHub repository (data/v0.0/)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-train.jsonl"
    val_url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl"

    dest_file = raw_dir / "folio_raw.json"
    items = []
    
    for url in [train_url, val_url]:
        temp_file = raw_dir / "temp_folio.jsonl"
        if download_url(url, temp_file):
            with open(temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            if temp_file.exists():
                temp_file.unlink()

    if items:
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
import zipfile


def fetch_proofwriter(raw_dir: Path, max_depth_download: str = "depth-3"):
    """Downloads and extracts the official AllenAI ProofWriter dataset release."""
    pw_dir = raw_dir / "proofwriter"
    pw_dir.mkdir(parents=True, exist_ok=True)
    
    zip_url = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
    zip_path = raw_dir / "proofwriter_temp.zip"

    print(f"[ProofWriter] Downloading official AI2 archive from {zip_url}...")
    if download_url(zip_url, zip_path):
        try:
            print(f"[ProofWriter] Extracting files into {pw_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(pw_dir)
            print(f"[ProofWriter Success] Archive extracted to {pw_dir}")
        except Exception as e:
            print(f"[ProofWriter Error] Failed to extract zip: {e}")
        finally:
            if zip_path.exists():
                zip_path.unlink()


def fetch_abduction_rules(raw_dir: Path):
    """Downloads and extracts the official AbductionRules dataset used by LogiTorch."""
    abduct_dir = raw_dir / "abduction_rules_dataset"
    if abduct_dir.exists() and any(abduct_dir.iterdir()):
        print(f"[AbductionRules] Found existing directory at {abduct_dir}")
        return

    abduct_dir.mkdir(parents=True, exist_ok=True)
    zip_url = "https://www.dropbox.com/s/zvm2v3noak0wt5f/abduction_rules_dataset.zip?dl=1"
    zip_path = raw_dir / "abduction_rules_temp.zip"

    print(f"[AbductionRules] Downloading dataset from {zip_url}...")
    if download_url(zip_url, zip_path):
        try:
            print(f"[AbductionRules] Extracting archive into {abduct_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(abduct_dir)
            print(f"[AbductionRules Success] Extracted to {abduct_dir}")
        except Exception as e:
            print(f"[AbductionRules Error] Failed to extract zip: {e}")
        finally:
            if zip_path.exists():
                zip_path.unlink()


def main():
    raw_dir = Path("data/raw")
    benchmark_dir = Path("data/benchmarks")

    print("=" * 60)
    print("Downloading Official Datasets & Benchmarks")
    print("=" * 60)
    fetch_folio(raw_dir)
    fetch_abduction_rules(raw_dir)


if __name__ == "__main__":
    main()


