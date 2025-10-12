import os
import sys
import glob
import shutil
import time

# ensure target folder exists
os.makedirs("data/raw", exist_ok=True)

# source directory (e.g., ~/Downloads)
src = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else "."
n = 0

for p in sorted(glob.glob(os.path.join(src, "*.*"))):
    ext = os.path.splitext(p)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".heic"]:
        continue
    if ext == ".heic":
        print(f"Skip HEIC (convert to JPG/PNG first): {p}")
        continue
    n += 1
    dst = os.path.join("data/raw", f"img_{int(time.time())}_{n:04d}{ext}")
    shutil.copy2(p, dst)
    print(f"copied -> {dst}")

print("done.")
