from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
FEATURES = ["return_1", "return_2", "return_6", "rolling_mean_6", "rolling_volatility_24", "volume_change"]


@dataclass
class BitcoinRun:
    model: Pipeline
    test: pd.DataFrame
    predictions: object
    metrics: dict[str, float]


def _data_path() -> Path:
    files = sorted(DATA_DIR.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            "No Bitcoin CSV was found in 'Bitcoin Historical'. Add the Kaggle CSV there, then rerun this study."
        )
    return files[0]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(_data_path())
    columns = {str(column).lower(): column for column in df.columns}
    timestamp = next((columns[key] for key in columns if "timestamp" in key or key == "date"), None)
    close = next((columns[key] for key in columns if key == "close" or "weighted_price" in key), None)
    volume = next((columns[key] for key in columns if "volume" in key and "currency" not in key), None)
    if timestamp is None or close is None:
        raise ValueError("The Bitcoin CSV must contain timestamp/date and close/weighted-price columns.")
    data = pd.DataFrame({"time": pd.to_datetime(df[timestamp], unit="s", errors="coerce"), "close": pd.to_numeric(df[close], errors="coerce")})
    data["volume"] = pd.to_numeric(df[volume], errors="coerce") if volume else 0.0
    data = data.dropna(subset=["time", "close"]).sort_values("time").set_index("time")
    # Hourly aggregation keeps the example responsive while preserving chronological order.
    return data.resample("1h").agg({"close": "last", "volume": "sum"}).dropna().reset_index()


def prepare_data() -> pd.DataFrame:
    df = load_data().copy()
    df["return_1"] = df["close"].pct_change(1)
    df["return_2"] = df["close"].pct_change(2)
    df["return_6"] = df["close"].pct_change(6)
    df["rolling_mean_6"] = df["close"].pct_change().rolling(6).mean()
    df["rolling_volatility_24"] = df["close"].pct_change().rolling(24).std()
    df["volume_change"] = df["volume"].pct_change().replace([float("inf"), float("-inf")], None)
    df["up_next_hour"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna(subset=FEATURES).iloc[:-1].copy()


def train() -> BitcoinRun:
    df = prepare_data()
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced", random_state=42))])
    model.fit(train_df[FEATURES], train_df["up_next_hour"])
    predictions = model.predict(test_df[FEATURES])
    return BitcoinRun(model, test_df.reset_index(drop=True), predictions, {"accuracy": accuracy_score(test_df["up_next_hour"], predictions), "f1_weighted": f1_score(test_df["up_next_hour"], predictions, average="weighted")})
