import pandas as pd, glob
files = glob.glob("data/processed_ds/Minneapolis-StPaul/I-94 Corridor/part-i94-d4b26e3f5d9b439a9df55f877943be45.parquet", recursive=True)
# df = pd.read_parquet("data/processed_ds/Minneapolis-StPaul/I-94 Corridor/part-i94-d4b26e3f5d9b439a9df55f877943be45.parquet")
for f in files:
    df = pd.read_parquet(f)
    if "I-94-WB" in df["RouteId"].astype(str).unique():
        print(f)
        g = df[df["RouteId"]=="I-94-WB"]
        print("min:", g["DateTime"].min(), "max:", g["DateTime"].max(), "rows:", len(g))
