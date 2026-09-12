"""
Download the Customer Support on Twitter dataset from Kaggle.
Usage:
    python data/download.py                    # Download via Kaggle API
    python data/download.py --manual-path X    # Use a manually downloaded CSV
"""
import sys
import os
import zipfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, KAGGLE_DATASET, CSV_FILENAME


def download_kaggle():
    """Download dataset using Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {KAGGLE_DATASET} to {RAW_DATA_DIR}...")
        api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DATA_DIR), unzip=True)
        csv_path = RAW_DATA_DIR / CSV_FILENAME
        if csv_path.exists():
            print(f"  Dataset downloaded: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
            return csv_path
        # Check for alternative filenames
        csvs = list(RAW_DATA_DIR.glob("*.csv"))
        if csvs:
            print(f"  Found CSV: {csvs[0].name}")
            return csvs[0]
        raise FileNotFoundError("No CSV found after download")
    except Exception as e:
        print(f"✗ Kaggle download failed: {e}")
        print("\nManual download instructions:")
        print("  1. Go to https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        print("  2. Download the dataset (twcs.csv)")
        print(f"  3. Place it in: {RAW_DATA_DIR}")
        print(f"  4. Re-run: python data/download.py --manual-path {RAW_DATA_DIR / CSV_FILENAME}")
        return None


def use_manual(path: str):
    """Copy a manually downloaded CSV into the data directory."""
    src = Path(path)
    if not src.exists():
        print(f"✗ File not found: {src}")
        return None
    dst = RAW_DATA_DIR / CSV_FILENAME
    if src != dst:
        shutil.copy2(src, dst)
    print(f"  Using manual CSV: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download customer support dataset")
    parser.add_argument("--manual-path", type=str, help="Path to manually downloaded CSV")
    args = parser.parse_args()

    if args.manual_path:
        use_manual(args.manual_path)
    else:
        download_kaggle()
