"""Download the assignment's Titanic dataset from Kaggle using KaggleHub."""
from pathlib import Path
import shutil

import kagglehub

ROOT = Path(__file__).resolve().parent
TARGET_DIR = ROOT / "dataset"
TARGET_DIR.mkdir(exist_ok=True)

download_dir = Path(kagglehub.dataset_download("yasserh/titanic-dataset"))
csv_files = list(download_dir.rglob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV found in KaggleHub download: {download_dir}")

source = next((p for p in csv_files if p.name.lower() == "titanic-dataset.csv"), csv_files[0])
target = TARGET_DIR / "Titanic-Dataset.csv"
shutil.copy2(source, target)
print(f"Dataset saved to: {target}")
