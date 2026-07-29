from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


Task = Literal["classification", "regression"]


@dataclass
class TabularRun:
    """All artifacts needed to reproduce a tabular XAI analysis."""

    task: Task
    target_name: str
    model: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: np.ndarray
    scores: np.ndarray | None
    metrics: dict[str, float]


def _encoder() -> OneHotEncoder:
    """Support both current and older scikit-learn releases."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_tabular(
    frame: pd.DataFrame,
    *,
    target: str,
    task: Task,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
    test_size: float = 0.2,
) -> TabularRun:
    """Fit a leakage-aware preprocessing + random forest pipeline.

    Missing values are imputed inside the pipeline, so fit-time statistics come
    from the training partition only. Categories are one-hot encoded with an
    unknown-category guard, allowing students to test new input values safely.
    """
    features = numeric_features + categorical_features
    data = frame[features + [target]].dropna(subset=[target]).copy()
    X = data[features]
    y = data[target]
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    estimator = (
        RandomForestClassifier(
            n_estimators=300, min_samples_leaf=4, class_weight="balanced", random_state=random_state
        )
        if task == "classification"
        else RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=random_state)
    )
    model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1] if task == "classification" and len(model.classes_) == 2 else None

    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_weighted": f1_score(y_test, predictions, average="weighted"),
        }
    else:
        metrics = {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": mean_squared_error(y_test, predictions) ** 0.5,
            "r2": r2_score(y_test, predictions),
        }
    return TabularRun(task, target, model, X_train, X_test, y_train, y_test, predictions, scores, metrics)


def encoded_frame(run: TabularRun, X: pd.DataFrame) -> pd.DataFrame:
    names = run.model.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [name.replace("numeric__", "").replace("categorical__", "") for name in names]
    values = run.model.named_steps["preprocessor"].transform(X)
    return pd.DataFrame(values, columns=clean_names, index=X.index)


def feature_importance(run: TabularRun, top_n: int = 15) -> pd.DataFrame:
    estimator = run.model.named_steps["model"]
    return (
        pd.DataFrame({"feature": encoded_frame(run, run.X_test).columns, "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def group_audit(run: TabularRun, groups: pd.Series) -> pd.DataFrame:
    """Report group outcomes; classification uses selection rate, regression uses MAE."""
    group_values = groups.reindex(run.X_test.index).fillna("Unknown").astype(str)
    rows: list[dict[str, float | int | str]] = []
    for group in sorted(group_values.unique()):
        mask = group_values == group
        actual = run.y_test[mask]
        predicted = run.predictions[mask.to_numpy()]
        row: dict[str, float | int | str] = {"group": group, "n": int(mask.sum())}
        if run.task == "classification":
            row["selection_rate"] = float(np.mean(predicted == 1))
            row["accuracy"] = float(np.mean(predicted == actual.to_numpy()))
        else:
            row["mae"] = float(mean_absolute_error(actual, predicted))
            row["mean_prediction"] = float(np.mean(predicted))
            row["mean_actual"] = float(np.mean(actual))
        rows.append(row)
    return pd.DataFrame(rows)


def shap_summary(run: TabularRun, output_path: str | Path) -> Path:
    """Write a Tree SHAP global summary plot. Dependencies are imported on demand."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise RuntimeError("Install shap and matplotlib to create SHAP plots.") from exc
    encoded = encoded_frame(run, run.X_test)
    explainer = shap.TreeExplainer(run.model.named_steps["model"])
    values = explainer.shap_values(encoded)
    if isinstance(values, list):
        values = values[1]
    elif getattr(values, "ndim", 0) == 3:
        values = values[:, :, 1]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, encoded, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path


def lime_local_explanation(run: TabularRun, sample_index: int, output_path: str | Path) -> tuple[Path, list[tuple[str, float]]]:
    """Fit a local LIME surrogate in the same transformed feature space as the forest."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise RuntimeError("Install lime to create LIME explanations.") from exc
    train_encoded = encoded_frame(run, run.X_train)
    test_encoded = encoded_frame(run, run.X_test)
    estimator = run.model.named_steps["model"]
    mode = "classification" if run.task == "classification" else "regression"
    explainer = LimeTabularExplainer(
        train_encoded.to_numpy(),
        feature_names=train_encoded.columns.tolist(),
        class_names=["negative", "positive"] if run.task == "classification" else None,
        mode=mode,
        random_state=42,
    )
    predict_fn = estimator.predict_proba if run.task == "classification" else estimator.predict
    explanation = explainer.explain_instance(test_encoded.iloc[sample_index].to_numpy(), predict_fn, num_features=10)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explanation.save_to_file(str(output_path))
    return output_path, explanation.as_list()




import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

TITLE = "Google Play High-Rating XAI"
SUBTITLE = "Binary classification: identify apps rated 4.2 or higher"
LEARNING_GOAL = "Learn robust cleaning of string-valued numeric variables before interpreting a high-rating classifier."
CONTEXT = "The target `high_rating` is defined as `Rating >= 4.2`. The application cleans installs, price, size, review counts, and dates before fitting the model. Ratings can be biased by selection effects and should not be treated as objective app quality."
LOCAL_INTERPRETATION = "A local explanation describes why the model labels one catalogue record as high- or lower-rated. It does not explain user satisfaction or future rating changes."
GLOBAL_INTERPRETATION = "Reviews and install counts can reflect popularity and survivorship bias. Interpretation should distinguish correlation from product-quality causes."
AUDIT_TITLE = "Free versus paid audit"
AUDIT_EXPLANATION = "Selection rate and classification accuracy are compared by app type. This is a product-category diagnostic, not a protected-group fairness metric."
AUDIT_CAUTION = "Paid apps are relatively rare in this dataset, so their estimates may be noisy."

TARGET = "high_rating"
NUMERIC = ["reviews_numeric", "installs_numeric", "price_numeric", "size_mb", "updated_year"]
CATEGORICAL = ["Category", "Type", "Content Rating", "Genres"]


def _number(value: object) -> float:
    return pd.to_numeric(re.sub(r"[^0-9.]", "", str(value)), errors="coerce")


def _size_mb(value: object) -> float:
    text = str(value).strip()
    if text == "Varies with device":
        return float("nan")
    number = _number(text)
    return number / 1024 if text.endswith("k") else number


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "googleplaystore.csv")
    df = df.drop_duplicates(subset=["App"], keep="first").copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])
    df["high_rating"] = (df["Rating"] >= 4.2).astype(int)
    df["reviews_numeric"] = pd.to_numeric(df["Reviews"], errors="coerce")
    df["installs_numeric"] = df["Installs"].apply(_number)
    df["price_numeric"] = df["Price"].apply(_number)
    df["size_mb"] = df["Size"].apply(_size_mb)
    df["updated_year"] = pd.to_datetime(df["Last Updated"], errors="coerce").dt.year
    return df


def train() -> TabularRun:
    return train_tabular(load_data(), target=TARGET, task="classification", numeric_features=NUMERIC, categorical_features=CATEGORICAL)


def audit_groups(run: TabularRun) -> pd.Series:
    return run.X_test["Type"]
