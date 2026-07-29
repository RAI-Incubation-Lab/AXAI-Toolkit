from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "netflix_titles.csv"


@dataclass
class NetflixRun:
    data: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: object


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).drop_duplicates(subset=["show_id"]).copy()
    fields = df[["type", "listed_in", "description", "country", "director", "cast"]].fillna("").astype(str)
    df["content_text"] = fields.agg(" ".join, axis=1)
    return df.reset_index(drop=True)


def train() -> NetflixRun:
    data = load_data()
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=25000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(data["content_text"])
    return NetflixRun(data, vectorizer, matrix)


def recommend(run: NetflixRun, title: str, n: int = 8) -> pd.DataFrame:
    index = int(run.data.index[run.data["title"] == title][0])
    query = run.matrix[index]
    scores = (run.matrix @ query.T).toarray().ravel()
    candidates = np.argsort(scores)[::-1]
    rows: list[dict[str, object]] = []
    terms = np.asarray(run.vectorizer.get_feature_names_out())
    for candidate in candidates:
        if candidate == index:
            continue
        shared = query.multiply(run.matrix[candidate]).toarray().ravel()
        term_ids = np.argsort(shared)[::-1]
        shared_terms = [terms[i] for i in term_ids if shared[i] > 0][:4]
        rows.append({"title": run.data.iloc[candidate]["title"], "type": run.data.iloc[candidate]["type"], "genres": run.data.iloc[candidate]["listed_in"], "similarity": float(scores[candidate]), "shared_explanation_terms": ", ".join(shared_terms)})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def coverage_summary(run: NetflixRun) -> pd.DataFrame:
    return run.data.groupby("type").agg(titles=("show_id", "size"), median_release_year=("release_year", "median")).reset_index()
