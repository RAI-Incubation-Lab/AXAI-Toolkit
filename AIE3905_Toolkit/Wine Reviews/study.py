from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

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
    df = df.drop_duplicates(subset=["description"], keep="first").copy()
    df["winery_group"] = df["winery"].fillna(df["title"]).astype(str)
    df["high_score"] = (df["points"] >= 90).astype(int)
    metadata = df[["country", "province", "variety"]].fillna("unknown").astype(str).agg(" ".join, axis=1)
    df["model_text"] = df["description"].astype(str) + " " + metadata
    return df


def train() -> WineRun:
    df = load_data()
    train_index, test_index = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, df["high_score"], groups=df["winery_group"]))
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["model_text"])
    model = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
    model.fit(X_train, train_df["high_score"])
    X_test = vectorizer.transform(test_df["model_text"]); predictions = model.predict(X_test); scores = model.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, train_df["high_score"])
    baseline_accuracy = accuracy_score(test_df["high_score"], baseline.predict(X_test))
    metrics = {"accuracy": accuracy_score(test_df["high_score"], predictions), "f1_weighted": f1_score(test_df["high_score"], predictions, average="weighted"), "roc_auc": roc_auc_score(test_df["high_score"], scores), "pr_auc": average_precision_score(test_df["high_score"], scores), "baseline_accuracy": baseline_accuracy, "validity_gate_passed": float(accuracy_score(test_df["high_score"], predictions) >= baseline_accuracy + 0.02)}
    return WineRun(vectorizer, model, test_df.reset_index(drop=True), predictions, metrics)


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
