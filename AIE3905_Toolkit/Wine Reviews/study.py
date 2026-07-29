from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "winemag-data-130k-v2.csv"


@dataclass
class WineRun:
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    test: pd.DataFrame
    predictions: np.ndarray
    metrics: dict[str, float]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["description", "points"]).copy()
    df["high_score"] = (df["points"] >= 90).astype(int)
    metadata = df[["country", "province", "variety"]].fillna("unknown").astype(str).agg(" ".join, axis=1)
    df["model_text"] = df["description"].astype(str) + " " + metadata
    return df


def train() -> WineRun:
    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["high_score"], random_state=42)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["model_text"])
    model = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
    model.fit(X_train, train_df["high_score"])
    predictions = model.predict(vectorizer.transform(test_df["model_text"]))
    return WineRun(vectorizer, model, test_df.reset_index(drop=True), predictions, {"accuracy": accuracy_score(test_df["high_score"], predictions), "f1_weighted": f1_score(test_df["high_score"], predictions, average="weighted")})


def local_token_contributions(run: WineRun, index: int, top_n: int = 10) -> pd.DataFrame:
    vector = run.vectorizer.transform([run.test.iloc[index]["model_text"]])
    contributions = vector.toarray()[0] * run.model.coef_[0]
    terms = np.asarray(run.vectorizer.get_feature_names_out())
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return pd.DataFrame({"term": terms[order], "contribution_to_high_score": contributions[order]})


def global_token_importance(run: WineRun, top_n: int = 20) -> pd.DataFrame:
    coefficients = run.model.coef_[0]
    terms = np.asarray(run.vectorizer.get_feature_names_out())
    order = np.argsort(np.abs(coefficients))[::-1][:top_n]
    return pd.DataFrame({"term": terms[order], "coefficient": coefficients[order]})


def country_audit(run: WineRun, minimum_n: int = 30) -> pd.DataFrame:
    audit = run.test.assign(prediction=run.predictions).groupby("country", dropna=False).agg(n=("high_score", "size"), actual_high_score_rate=("high_score", "mean"), predicted_high_score_rate=("prediction", "mean"))
    return audit[audit["n"] >= minimum_n].sort_values("n", ascending=False).reset_index()
