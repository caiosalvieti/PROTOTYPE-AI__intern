# tools/fetch_kaggle_faces.py
import os, shutil
from pathlib import Path
import kagglehub

# 1) download dataset locally (Kaggle may ask you to sign in/accept terms)
ds_path = kagglehub.dataset_download("ashwingupta3012/human-faces")
print("Dataset at:", ds_path)

# 2) copy only images into data/raw/
target = Path("data/raw")
target.mkdir(parents=True, exist_ok=True)

allowed = {".jpg", ".jpeg", ".png"}
files = [p for p in Path(ds_path).rglob("*") if p.suffix.lower() in allowed]

# (optional) limit if you don't want everything yet
MAX_FILES = None   # e.g., 300 to cap
if isinstance(MAX_FILES, int):
    files = files[:MAX_FILES]

for i, p in enumerate(files, 1):
    dst = target / f"kaggle_{i:05d}{p.suffix.lower()}"
print(f"Copied {len(files)} file(s) into {target}")
