from pathlib import Path
import pandas as pd
import glob

RAW_DIR = Path("data/raw/hcmc")

def show_file(fp, n=5):
    try:
        df = pd.read_csv(fp, nrows=0)
    except Exception as e:
        print(f"[{fp}] read header error:", e)
        return
    print(f"\n=== {fp} ===")
    print("Columns:", list(df.columns))
    try:
        df2 = pd.read_csv(fp, nrows=n)
        print(df2.head(n))
    except Exception as e:
        print("Sample read error:", e)

def main():
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not files:
        print("No CSV in data/raw/hcmc"); return
    for f in files:
        show_file(f)

if __name__ == "__main__":
    main()
