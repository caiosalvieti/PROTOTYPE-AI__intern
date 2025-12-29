# scripts/clean_products_kb.py
import sys
import pandas as pd

def clean(path_in: str, path_out: str):
    df = pd.read_csv(path_in, dtype=str).fillna("")
    # drop rows that are duplicated headers
    df = df[df["id"].str.lower() != "id"]
    # drop duplicate IDs (keep first)
    df = df.drop_duplicates(subset=["id"], keep="first")

    # normalize booleans
    for c in ["fragrance_free", "include_device"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "": False}
            ).fillna(False)

    df.to_csv(path_out, index=False)
    print(f"Saved cleaned KB -> {path_out} ({len(df)} rows)")

if __name__ == "__main__":
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".csv", "_clean.csv")
    clean(inp, out)
