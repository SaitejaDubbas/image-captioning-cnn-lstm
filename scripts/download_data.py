"""Download Flickr8k dataset from Kaggle.

Usage:
    python scripts/download_data.py

Prerequisites:
    1. pip install kaggle
    2. Place kaggle.json at ~/.kaggle/kaggle.json  (from kaggle.com -> Account -> API)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading Flickr8k to {RAW_DIR} ...")
result = subprocess.run(
    [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", "adityajn105/flickr8k",
        "-p", str(RAW_DIR),
        "--unzip",
    ],
    check=True,
)
print("Download complete.")
print(f"Contents of {RAW_DIR}:")
for f in sorted(RAW_DIR.iterdir()):
    size = f.stat().st_size // 1024
    print(f"  {f.name}  ({size} KB)")
