"""Download and verify the Bitcoin Kaggle data without committing it to Git."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = "mczielinski/bitcoin-historical-data"
EXPECTED_COLUMNS = {"Timestamp", "Open", "High", "Low", "Close", "Volume"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(csv_path: Path) -> None:
    header = set(csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].split(","))
    missing = EXPECTED_COLUMNS - header
    if missing:
        raise ValueError(f"Unexpected CSV schema; missing columns: {sorted(missing)}")
    print(f"Verified schema: {csv_path.name}")
    print(f"SHA-256: {sha256(csv_path)}")


def download() -> Path:
    archive = ROOT / "archive.zip"
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(ROOT), "--force"], check=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(ROOT)
    csv_path = ROOT / "btcusd_1-min_data.csv"
    verify(csv_path)
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or verify the Bitcoin Historical dataset.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify(ROOT / "btcusd_1-min_data.csv") if args.verify_only else download()
