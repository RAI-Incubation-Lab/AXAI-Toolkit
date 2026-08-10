"""Generate a small, self-contained XAI teaching project from a CSV dataset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


STUDY_TEMPLATE = '''from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_FILE = "{data_file}"
TARGET = "{target}"
TASK = "{task}"
RANDOM_STATE = 42


def encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / DATA_FILE)


def train() -> dict[str, object]:
    data = load_data().dropna(subset=[TARGET]).copy()
    X, y = data.drop(columns=[TARGET]), data[TARGET]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    stratify = y if TASK == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder())]), categorical),
    ])
    estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, random_state=RANDOM_STATE) if TASK == "classification" else RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=RANDOM_STATE)
    baseline = DummyClassifier(strategy="most_frequent") if TASK == "classification" else DummyRegressor(strategy="mean")
    model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    predicted, baseline_predicted = model.predict(X_test), baseline.predict(X_test)
    if TASK == "classification":
        metrics = {{"accuracy": accuracy_score(y_test, predicted), "f1_weighted": f1_score(y_test, predicted, average="weighted"), "baseline_accuracy": accuracy_score(y_test, baseline_predicted)}}
    else:
        metrics = {{"mae": mean_absolute_error(y_test, predicted), "rmse": mean_squared_error(y_test, predicted) ** 0.5, "r2": r2_score(y_test, predicted), "baseline_mae": mean_absolute_error(y_test, baseline_predicted)}}
    return {{"model": model, "X_test": X_test, "y_test": y_test, "predicted": predicted, "metrics": metrics}}


if __name__ == "__main__":
    run = train()
    print(pd.Series(run["metrics"]).to_string())
'''

README_TEMPLATE = '''# {name}

## Learning objective

Build a reproducible {task} model from `{data_file}` and evaluate whether its explanations are trustworthy.

## Quick start

```powershell
python -m pip install -r requirements.txt
python study.py
jupyter notebook xai_workflow.ipynb
```

## Required student work

1. Verify the target and feature availability at prediction time.
2. Replace the default random split with a group or temporal split when the data contain repeated entities or time.
3. Compare the fitted model against the saved dummy baseline before interpreting an explanation.
4. Add one global and one local explanation, record the sampled input, ground truth, prediction, confidence, method, and fidelity.
5. State limitations, subgroup sample sizes, and the difference between a diagnostic and a fairness conclusion.

## Review

From the toolkit root, run:

```powershell
python axai_harness.py audit "{name}" --execute
```
'''

MODEL_CARD_TEMPLATE = '''# Model Card: {name}

## Intended use
Teaching example only. This project must not support consequential decisions without domain validation.

## Data and prediction task
- Dataset: `{data_file}`
- Target: `{target}`
- Task: {task}

## Evaluation and validity gate
Record the train/test strategy, model metrics, dummy baseline, and the pass/fail interpretation gate here after running the notebook.

## Explainability evidence
Record the exact input, expected output, prediction, confidence, explanation method, fidelity, model hash, and data hash for representative cases.

## Limitations and fairness
Document missing data, potential leakage, distribution shift, subgroup sizes, uncertainty, and why any fairness metric is appropriate.
'''

REQUIREMENTS = "pandas>=2.0,<3\nnumpy>=1.24,<3\nscikit-learn>=1.3,<2\nshap>=0.44,<1\nlime>=0.2.0.1,<1\nmatplotlib>=3.7,<4\njupyter>=1,<2\n"


def create_example(destination: str | Path, data: str | Path, target: str, task: str, copy_data: bool = True) -> Path:
    destination = Path(destination).resolve()
    source = Path(data).resolve()
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise FileNotFoundError("data must be an existing CSV file")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    data_file = source.name
    name = destination.name
    if copy_data:
        shutil.copy2(source, destination / data_file)
    (destination / "study.py").write_text(STUDY_TEMPLATE.format(data_file=data_file, target=target, task=task), encoding="utf-8")
    (destination / "README.md").write_text(README_TEMPLATE.format(name=name, data_file=data_file, task=task), encoding="utf-8")
    (destination / "MODEL_CARD.md").write_text(MODEL_CARD_TEMPLATE.format(name=name, data_file=data_file, target=target, task=task), encoding="utf-8")
    (destination / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a standalone XAI teaching example from a CSV file.")
    parser.add_argument("destination")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--no-copy-data", action="store_true")
    args = parser.parse_args()
    created = create_example(args.destination, args.data, args.target, args.task, not args.no_copy_data)
    print(f"Created {created}")


if __name__ == "__main__":
    main()
