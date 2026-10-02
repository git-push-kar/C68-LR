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
        print(f"[FOLIO Success] Saved {len(items)} official FOLIO examples to {dest_file}")


def main():
    raw_dir = Path("data/raw")
    benchmark_dir = Path("data/benchmarks")

    print("=" * 60)
    print("Downloading Official Datasets & Benchmarks")
    print("=" * 60)
    fetch_folio(raw_dir)


if __name__ == "__main__":
    main()
