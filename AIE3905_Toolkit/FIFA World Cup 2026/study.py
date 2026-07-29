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




import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

TITLE = "FIFA 2026 Player Performance XAI"
SUBTITLE = "Regression: explain a match-level player performance score"
LEARNING_GOAL = "Learn how sporting-event statistics, player role, and match context influence a modelled performance score."
CONTEXT = "The dataset contains match-level player records. This case predicts `performance_score` using event statistics and context, then compares model behavior across playing positions. The score is an observed dataset variable, not an objective measure of football quality."
LOCAL_INTERPRETATION = "Use the selected record to discuss which match actions may explain the model output. This is a model explanation, not a causal claim that changing one statistic would change a real match result."
GLOBAL_INTERPRETATION = "Compare offensive, defensive, physical, and contextual variables. A high importance can reflect the dataset's score construction as well as genuine predictive signal."
AUDIT_TITLE = "Position-group audit"
AUDIT_EXPLANATION = "Regression MAE is reported by position. Large differences may indicate that the same model fits some roles better than others."
AUDIT_CAUTION = "Positions have very different responsibilities and feature distributions; this table is a performance diagnostic, not a fairness verdict."

TARGET = "performance_score"
NUMERIC = ["age", "height_cm", "weight_kg", "market_value_eur", "minutes_played", "goals", "assists", "shots_on_target", "expected_goals_xg", "expected_assists_xa", "pass_accuracy", "successful_passes", "tackles", "interceptions", "saves", "distance_covered_km", "top_speed_kmh", "stamina_score"]
CATEGORICAL = ["position", "preferred_foot", "tournament_stage", "match_result"]


def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "fifa_world_cup_2026_player_performance.csv")


def train() -> TabularRun:
    return train_tabular(load_data(), target=TARGET, task="regression", numeric_features=NUMERIC, categorical_features=CATEGORICAL)


def audit_groups(run: TabularRun) -> pd.Series:
    return run.X_test["position"]
