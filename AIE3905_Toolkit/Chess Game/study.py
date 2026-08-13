from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

Task = Literal["classification", "regression"]
RANDOM_STATE = 42


@dataclass
class TabularRun:
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
    numeric_features: list[str]
    categorical_features: list[str]
    split_description: str
    validity_passed: bool


def _encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _split(data: pd.DataFrame, features: list[str], target: str, task: Task, group_column: str | None, time_column: str | None, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if time_column and time_column in data.columns:
        ordered = data.sort_values(time_column, kind="stable")
        boundary = max(1, int(len(ordered) * (1 - test_size)))
        return ordered.iloc[:boundary], ordered.iloc[boundary:], f"chronological holdout by {time_column}"
    if group_column and group_column in data.columns:
        groups = data[group_column].fillna("Unknown").astype(str)
        train_index, test_index = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE).split(data, data[target], groups))
        return data.iloc[train_index], data.iloc[test_index], f"group holdout by {group_column}"
    stratify = data[target] if task == "classification" and data[target].value_counts().min() >= 2 else None
    train, test = train_test_split(data, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify)
    return train, test, "stratified random holdout"


def train_tabular(frame: pd.DataFrame, *, target: str, task: Task, numeric_features: list[str], categorical_features: list[str], group_column: str | None = None, time_column: str | None = None, test_size: float = 0.2) -> TabularRun:
    features = numeric_features + categorical_features
    support = [column for column in (group_column, time_column) if column and column not in features]
    data = frame[features + support + [target]].dropna(subset=[target]).copy()
    train, test, split_description = _split(data, features, target, task, group_column, time_column, test_size)
    X_train, X_test = train[features], test[features]
    y_train, y_test = train[target], test[target]
    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _encoder())]), categorical_features))
    model = Pipeline([("preprocessor", ColumnTransformer(transformers)), ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE) if task == "classification" else RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE))])
    baseline = DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor(strategy="mean")
    model.fit(X_train, y_train); baseline.fit(X_train, y_train)
    predictions, baseline_predictions = model.predict(X_test), baseline.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1] if task == "classification" and len(model.classes_) == 2 else None
    if task == "classification":
        metrics = {"accuracy": accuracy_score(y_test, predictions), "f1_weighted": f1_score(y_test, predictions, average="weighted"), "baseline_accuracy": accuracy_score(y_test, baseline_predictions)}
        if scores is not None:
            metrics.update({"roc_auc": roc_auc_score(y_test, scores), "pr_auc": average_precision_score(y_test, scores)})
        validity_passed = metrics["accuracy"] >= metrics["baseline_accuracy"] + 0.02
    else:
        metrics = {"mae": mean_absolute_error(y_test, predictions), "rmse": mean_squared_error(y_test, predictions) ** 0.5, "r2": float(1 - np.sum((y_test - predictions) ** 2) / max(np.sum((y_test - y_test.mean()) ** 2), 1e-12)), "baseline_mae": mean_absolute_error(y_test, baseline_predictions)}
        validity_passed = metrics["mae"] <= metrics["baseline_mae"] * 0.98 and metrics["r2"] > 0
    metrics["validity_gate_passed"] = float(validity_passed)
    return TabularRun(task, target, model, X_train, X_test, y_train, y_test, predictions, scores, metrics, numeric_features, categorical_features, split_description, validity_passed)


def encoded_frame(run: TabularRun, X: pd.DataFrame) -> pd.DataFrame:
    names = [name.replace("numeric__", "").replace("categorical__", "") for name in run.model.named_steps["preprocessor"].get_feature_names_out()]
    return pd.DataFrame(run.model.named_steps["preprocessor"].transform(X), columns=names, index=X.index)


def feature_importance(run: TabularRun, top_n: int = 15) -> pd.DataFrame:
    return pd.DataFrame({"feature": encoded_frame(run, run.X_test).columns, "importance": run.model.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def _ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 30: return float("nan"), float("nan")
    rng = np.random.default_rng(RANDOM_STATE)
    draws = [rng.choice(values, len(values), replace=True).mean() for _ in range(300)]
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def group_audit(run: TabularRun, groups: pd.Series) -> pd.DataFrame:
    values = groups.reindex(run.X_test.index).fillna("Unknown").astype(str); rows = []
    for group in sorted(values.unique()):
        mask = values == group; actual = run.y_test[mask].to_numpy(); predicted = run.predictions[mask.to_numpy()]
        row: dict[str, float | int | str | bool] = {"group": group, "n": int(mask.sum()), "small_sample_warning": bool(mask.sum() < 30)}
        if run.task == "classification":
            positive, truth = predicted == 1, actual == 1; negative = ~truth; low, high = _ci(positive.astype(float))
            row.update({"selection_rate": float(positive.mean()), "selection_rate_ci_low": low, "selection_rate_ci_high": high, "accuracy": accuracy_score(actual, predicted), "true_positive_rate": float((positive & truth).sum() / max(truth.sum(), 1)), "false_positive_rate": float((positive & negative).sum() / max(negative.sum(), 1)), "precision": precision_score(actual, predicted, zero_division=0), "recall": recall_score(actual, predicted, zero_division=0)})
        else:
            row.update({"mae": mean_absolute_error(actual, predicted), "mean_prediction": float(np.mean(predicted)), "mean_actual": float(np.mean(actual))})
        rows.append(row)
    return pd.DataFrame(rows)


def shap_summary(run: TabularRun, output_path: str | Path, max_samples: int = 200) -> Path:
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): return output
    try:
        import matplotlib.pyplot as plt; import shap
    except ImportError as exc: raise RuntimeError("Install shap and matplotlib to create SHAP plots.") from exc
    sample = run.X_test.sample(n=min(max_samples, len(run.X_test)), random_state=RANDOM_STATE); encoded = encoded_frame(run, sample)
    values = shap.TreeExplainer(run.model.named_steps["model"]).shap_values(encoded)
    if isinstance(values, list): values = values[1]
    elif getattr(values, "ndim", 0) == 3: values = values[:, :, 1]
    shap.summary_plot(values, encoded, show=False, max_display=15); plt.tight_layout(); plt.savefig(output, dpi=160, bbox_inches="tight"); plt.close(); return output


def lime_local_explanation(run: TabularRun, sample_index: int, output_path: str | Path) -> tuple[Path, list[tuple[str, float]], float]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc: raise RuntimeError("Install lime to create LIME explanations.") from exc
    features = run.numeric_features + run.categorical_features
    categories = {name: sorted(run.X_train[name].fillna("Unknown").astype(str).unique().tolist()) for name in run.categorical_features}
    def matrix(frame: pd.DataFrame) -> np.ndarray:
        result = frame[features].copy()
        for name in run.numeric_features: result[name] = pd.to_numeric(result[name], errors="coerce").fillna(0.0)
        for name in run.categorical_features: result[name] = pd.Categorical(result[name].fillna("Unknown").astype(str), categories=categories[name]).codes
        return result.to_numpy(float)
    def predict_fn(values: np.ndarray) -> np.ndarray:
        result = pd.DataFrame(values, columns=features)
        for name in run.numeric_features: result[name] = pd.to_numeric(result[name], errors="coerce").clip(lower=0)
        for name in run.categorical_features:
            codes = np.rint(result[name]).astype(int).clip(0, len(categories[name]) - 1); result[name] = [categories[name][code] for code in codes]
        return run.model.predict_proba(result) if run.task == "classification" else run.model.predict(result)
    explainer = LimeTabularExplainer(matrix(run.X_train), feature_names=features, mode="classification" if run.task == "classification" else "regression", class_names=["negative", "positive"] if run.task == "classification" else None, categorical_features=[features.index(name) for name in run.categorical_features], categorical_names={features.index(name): categories[name] for name in run.categorical_features}, random_state=RANDOM_STATE)
    explanation = explainer.explain_instance(matrix(run.X_test)[sample_index], predict_fn, num_features=min(10, len(features)))
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True); explanation.save_to_file(str(output)); return output, explanation.as_list(), float(explanation.score)


def write_evidence(run: TabularRun, output_dir: str | Path, sample_index: int = 0) -> tuple[Path, Path]:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True); sample_index = min(sample_index, len(run.X_test) - 1)
    row = run.X_test.iloc[sample_index].to_dict(); prediction = run.predictions[sample_index]
    payload = {"sample_id": str(run.X_test.index[sample_index]), "raw_input": row, "expected_output": run.y_test.iloc[sample_index].item(), "predicted_output": prediction.item() if hasattr(prediction, "item") else prediction, "probability": float(run.scores[sample_index]) if run.scores is not None else None, "correct": bool(prediction == run.y_test.iloc[sample_index]), "explanation_method": "random_forest_feature_importance", "explanation": feature_importance(run, 10).to_dict(orient="records"), "model_hash": hashlib.sha256(repr(run.model).encode()).hexdigest()[:12], "data_hash": hashlib.sha256(pd.util.hash_pandas_object(run.X_train, index=True).values.tobytes()).hexdigest()[:12], "split": run.split_description, "validity_gate_passed": run.validity_passed}
    evidence = directory / "representative_evidence.json"; metrics = directory / "metrics.json"; evidence.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8"); metrics.write_text(json.dumps({"metrics": run.metrics, "split": run.split_description, "validity_gate_passed": run.validity_passed}, indent=2), encoding="utf-8"); return evidence, metrics


import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

TITLE = "Lichess Chess Outcome XAI"
SUBTITLE = "Binary classification: predict whether White wins a decisive game"
LEARNING_GOAL = "Study how rating difference, time control, opening choice, and game length relate to a model's prediction of a White win."
CONTEXT = "Draws are removed so the target is `white_win` for decisive games only. The model uses game-level metadata, not the move sequence. This keeps explanations readable but cannot describe tactical chess quality."
LOCAL_INTERPRETATION = "A local explanation can reveal whether the model relied on rating difference, opening, or time-control patterns for one recorded game."
GLOBAL_INTERPRETATION = "Opening labels may be predictive in this sample but should not be read as universal opening-strength rankings."
AUDIT_TITLE = "Rated versus casual game audit"
AUDIT_EXPLANATION = "Selection rate and accuracy are compared between rated and casual games to check whether the classifier behaves differently across game contexts."
AUDIT_CAUTION = "This is a subgroup performance check. It does not measure player fairness, and causal conclusions require controlled chess analysis."

TARGET = "white_win"
NUMERIC = ["white_rating", "black_rating", "rating_difference", "turns", "opening_ply"]
CATEGORICAL = ["rated", "victory_status", "increment_code", "opening_eco", "opening_name"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "games.csv")
    df = df[df["winner"].isin(["white", "black"])].copy()
    df["white_win"] = (df["winner"] == "white").astype(int)
    df["rating_difference"] = df["white_rating"] - df["black_rating"]
    df["rated"] = df["rated"].map({True: "rated", False: "casual"}).fillna("unknown")
    return df


def train() -> TabularRun:
    return train_tabular(load_data(), target=TARGET, task="classification", numeric_features=NUMERIC, categorical_features=CATEGORICAL, time_column="created_at")


def audit_groups(run: TabularRun) -> pd.Series:
    return run.X_test["rated"]


if __name__ == "__main__":
    run = train()
    output = ROOT / "outputs"
    write_evidence(run, output)
    print(json.dumps({"metrics": run.metrics, "split": run.split_description, "validity_gate_passed": run.validity_passed}, indent=2))
