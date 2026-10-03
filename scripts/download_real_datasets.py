import sys
import os
import json
import zipfile
import urllib.request
import ssl
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_file_with_progress(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Downloads a file from a URL with live progress reporting and SSL handling."""
    print(f"\n[Fetching] {url} -> {dest_path.name}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create unverified SSL context for cross-platform compatibility where certs might be missing
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    try:
        with urllib.request.urlopen(req, context=ssl_context) as response, open(dest_path, 'wb') as out_file:
            total_size = response.getheader('Content-Length')
            total_size = int(total_size) if total_size else None
            downloaded = 0

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"\rDownloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)

        print(f"\n[Saved] {dest_path.name} ({os.path.getsize(dest_path):,} bytes)")
        return True
    except Exception as e:
        print(f"\n[Error] Download failed for {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def fetch_folio(raw_dir: Path):
    """Downloads official FOLIO dataset directly from Yale-LILY GitHub repository (data/v0.0/)."""
    dest_file = raw_dir / "folio_raw.json"
    if dest_file.exists() and os.path.getsize(dest_file) > 1000:
        print(f"[FOLIO] Found existing dataset at {dest_file}")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    train_url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-train.jsonl"
    val_url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl"

    items = []
    for url in [train_url, val_url]:
        temp_file = raw_dir / "temp_folio.jsonl"
        if download_file_with_progress(url, temp_file):
            with open(temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            if temp_file.exists():
                temp_file.unlink()

    if items:
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        print(f"[FOLIO Success] Saved {len(items)} official FOLIO examples to {dest_file}")


def fetch_proofwriter(raw_dir: Path):
    """Downloads and extracts the official AllenAI ProofWriter dataset release (214 MB)."""
    pw_dir = raw_dir / "proofwriter"
    # Check if already downloaded and extracted
    if pw_dir.exists() and any(pw_dir.glob("**/*.jsonl")):
        print(f"[ProofWriter] Found existing extracted files in {pw_dir}")
        return

    pw_dir.mkdir(parents=True, exist_ok=True)
    zip_url = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
    zip_path = raw_dir / "proofwriter_temp.zip"

    print(f"\n[ProofWriter] Fetching official 214 MB archive from AI2 S3: {zip_url}")
    if download_file_with_progress(zip_url, zip_path):
        try:
            print(f"[ProofWriter] Extracting zip archive into {pw_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(pw_dir)
            print(f"[ProofWriter Success] Archive extracted to {pw_dir}")
        except Exception as e:
            print(f"[ProofWriter Error] Extraction failed: {e}")
        finally:
            if zip_path.exists():
                zip_path.unlink()


def fetch_abduction_rules(raw_dir: Path):
    """Downloads and extracts the official AbductionRules dataset used by LogiTorch."""
    abduct_dir = raw_dir / "abduction_rules_dataset"
    if abduct_dir.exists() and any(abduct_dir.glob("**/*.jsonl")):
        print(f"[AbductionRules] Found existing extracted files in {abduct_dir}")
        return

    abduct_dir.mkdir(parents=True, exist_ok=True)
    zip_url = "https://www.dropbox.com/s/zvm2v3noak0wt5f/abduction_rules_dataset.zip?dl=1"
    zip_path = raw_dir / "abduction_rules_temp.zip"

    print(f"\n[AbductionRules] Fetching dataset from {zip_url}...")
    if download_file_with_progress(zip_url, zip_path):
        try:
            print(f"[AbductionRules] Extracting archive into {abduct_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(abduct_dir)
            print(f"[AbductionRules Success] Extracted to {abduct_dir}")
        except Exception as e:
            print(f"[AbductionRules Error] Extraction failed: {e}")
        finally:
            if zip_path.exists():
                zip_path.unlink()


def main():
    raw_dir = Path("data/raw")
    benchmark_dir = Path("data/benchmarks")
    raw_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Downloading Official Logical Reasoning Datasets")
    print("=" * 70)
    
    # 1. FOLIO (Stage 2)
    fetch_folio(raw_dir)
    
    # 2. ProofWriter (Stage 1) - AllenAI Official S3
    fetch_proofwriter(raw_dir)
    
    # 3. AbductionRules (Stage 3) - LogiTorch Release
    fetch_abduction_rules(raw_dir)

    print("\n" + "=" * 70)
    print("[Done] Automated dataset download routine complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
