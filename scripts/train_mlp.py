# scripts/train_mlp.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

DATA = Path("data/processed/i94_hourly.parquet")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["City", "ZoneName", "RouteId", "Day"]
NUM_COLS = ["Year", "Month", "Date", "Hour", "DayOfWeek", "HourOfDay"]

def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ép về datetime UTC cho chắc chắn (kể cả khi đã là tz-aware)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
    # (tuỳ chọn) bỏ dòng lỗi datetime
    df = df.dropna(subset=["DateTime"])

    # Nếu muốn bỏ timezone (làm "naive") thì mở dòng dưới:
    # df["DateTime"] = df["DateTime"].dt.tz_localize(None)

    df["Year"]      = df["DateTime"].dt.year
    df["Month"]     = df["DateTime"].dt.month
    df["Date"]      = df["DateTime"].dt.day
    df["Hour"]      = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["HourOfDay"] = df["DateTime"].dt.hour
    df["Day"]       = df["DateTime"].dt.day_name()
    return df


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"Không tìm thấy {DATA}. Hãy chạy scripts/prep_data.py trước.")
    df = pd.read_parquet(DATA)
    df = df.dropna(subset=["Vehicles"])
    return build_time_features(df)

def build_model(input_dim: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

def main():
    df = load_data()
    y = df["Vehicles"].astype(float)
    X = df[CAT_COLS + NUM_COLS].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # pre = ColumnTransformer([
    #     ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), CAT_COLS),
    #     ("num", StandardScaler(), NUM_COLS),
    # ])
    # Preprocessor (tương thích nhiều phiên bản sklearn)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # sklearn < 1.2

    pre = ColumnTransformer([
        ("cat", ohe, CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])

    pre.fit(X_tr)

    Xtr = pre.transform(X_tr)
    Xte = pre.transform(X_te)

    model = build_model(Xtr.shape[1])
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    hist = model.fit(
        Xtr, y_tr.values,
        validation_data=(Xte, y_te.values),
        epochs=200, batch_size=1024, callbacks=[es], verbose=1
    )

    mse, mae = model.evaluate(Xte, y_te.values, verbose=0)
    print(f"Test MSE: {mse:.4f}  MAE: {mae:.4f}")

    model.save(MODEL_DIR / "traffic_mlp.h5")
    joblib.dump(pre, MODEL_DIR / "encoder.pkl")

    with open(MODEL_DIR / "required_cols.json", "w") as f:
        json.dump({"cat_cols": CAT_COLS, "num_cols": NUM_COLS}, f, ensure_ascii=False, indent=2)

    print("Saved:", (MODEL_DIR / "traffic_mlp.h5").resolve())
    print("Saved:", (MODEL_DIR / "encoder.pkl").resolve())

if __name__ == "__main__":
    main()
