from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/seattle/fremont_bridge_bikes.csv")
OUT_DIR = Path("data/processed_ds/Seattle/FremontBridge")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Đọc CSV từ Kaggle
    df = pd.read_csv(RAW_PATH)

    # Parse datetime
    # Kaggle dùng cột "Date" kiểu "2012-10-03 00:00:00"
    df["DateTime"] = pd.to_datetime(df["Date"], errors="coerce")

    # Bỏ những dòng lỗi datetime
    df = df.dropna(subset=["DateTime"])

    # Đảm bảo sort theo thời gian
    df = df.sort_values("DateTime")

    # (tuỳ bạn) fill NaN = 0 cho các cột đếm xe
    # nếu muốn giữ NaN để model_utils tự drop cũng được
    # df[[
    #     "Fremont Bridge Total",
    #     "Fremont Bridge East Sidewalk",
    #     "Fremont Bridge West Sidewalk",
    # ]] = df[[
    #     "Fremont Bridge Total",
    #     "Fremont Bridge East Sidewalk",
    #     "Fremont Bridge West Sidewalk",
    # ]].fillna(0)

    routes_config = [
        (
            "Fremont Bridge Total",
            "Fremont-Total",
            "Fremont_Total.parquet",
        ),
        (
            "Fremont Bridge East Sidewalk",
            "Fremont-East",
            "Fremont_East.parquet",
        ),
        (
            "Fremont Bridge West Sidewalk",
            "Fremont-West",
            "Fremont_West.parquet",
        ),
    ]

    for col, route_id, fname in routes_config:
        if col not in df.columns:
            print(f"Column {col!r} không tồn tại trong CSV, bỏ qua.")
            continue

        tmp = df[["DateTime", col]].copy()
        tmp = tmp.dropna(subset=[col])

        # Đổi tên cột về format chuẩn
        tmp = tmp.rename(columns={col: "Vehicles"})
        tmp["RouteId"] = route_id

        # Bỏ timezone, giữ datetime naive (phù hợp data_loader hiện tại)
        tmp["DateTime"] = pd.to_datetime(tmp["DateTime"]).dt.tz_localize(None)

        # Đảm bảo đúng kiểu
        tmp["Vehicles"] = pd.to_numeric(tmp["Vehicles"], errors="coerce")
        tmp = tmp.dropna(subset=["Vehicles"])

        tmp.to_parquet(OUT_DIR / fname, index=False)
        print(f"Saved {fname} với {len(tmp)} dòng (RouteId={route_id})")


if __name__ == "__main__":
    main()
