from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEMO_DIR = Path(__file__).resolve().parent
DATA_PATH = DEMO_DIR / "Titanic-Dataset.csv"
RANDOM_STATE = 42

RAW_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TitanicRun:
    model: Pipeline
    raw: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, float]


def load_titanic(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the Kaggle Titanic CSV used by this worked example."""
    return pd.read_csv(path)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep explanatory passenger fields and add simple family features."""
    X = df[RAW_COLUMNS].copy()
    X["Embarked"] = X["Embarked"].fillna("Unknown")
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    X["Pclass"] = X["Pclass"].astype(str)
    return X[MODEL_FEATURES]


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model(random_state: int = RANDOM_STATE) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_titanic_model(path: str | Path = DATA_PATH) -> TitanicRun:
    df = load_titanic(path)
    X = clean_features(df)
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return TitanicRun(model, df, X_train, X_test, y_train, y_test, y_pred, y_proba, metrics)


def encoded_feature_names(model: Pipeline) -> list[str]:
    names = model.named_steps["preprocessor"].get_feature_names_out()
    return [name.replace("num__", "").replace("cat__", "") for name in names]


def encoded_frame(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    encoded = model.named_steps["preprocessor"].transform(X)
    return pd.DataFrame(encoded, columns=encoded_feature_names(model), index=X.index)


def model_feature_importance(run: TitanicRun, top_n: int = 10) -> pd.DataFrame:
    rf = run.model.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": encoded_feature_names(run.model),
            "importance": rf.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def shap_analysis(run: TitanicRun, output_dir: str | Path = DEMO_DIR / "outputs", sample_index: int = 0) -> dict[str, Any]:
    """Create SHAP summary and waterfall images if shap/matplotlib are installed."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise RuntimeError("Install shap and matplotlib to run SHAP analysis.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_encoded = encoded_frame(run.model, run.X_test)
    rf = run.model.named_steps["model"]
    explainer = shap.TreeExplainer(rf)
    raw_values = explainer.shap_values(X_encoded)

    if isinstance(raw_values, list):
        shap_values = raw_values[1]
        base_value = explainer.expected_value[1]
    elif getattr(raw_values, "ndim", 0) == 3:
        shap_values = raw_values[:, :, 1]
        base_value = explainer.expected_value[1]
    else:
        shap_values = raw_values
        base_value = explainer.expected_value

    summary_path = output_dir / "shap_summary.png"
    waterfall_path = output_dir / "shap_waterfall_passenger.png"

    shap.summary_plot(shap_values, X_encoded, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=base_value,
        data=X_encoded.iloc[sample_index].values,
        feature_names=X_encoded.columns.tolist(),
    )
    shap.waterfall_plot(explanation, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()

    importance = (
        pd.DataFrame(
            {
                "feature": X_encoded.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "summary_path": summary_path,
        "waterfall_path": waterfall_path,
        "importance": importance,
        "sample_index": sample_index,
    }


def lime_analysis(run: TitanicRun, output_dir: str | Path = DEMO_DIR / "outputs", sample_index: int = 0) -> dict[str, Any]:
    """Create a LIME local explanation HTML file using the encoded feature space."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise RuntimeError("Install lime to run LIME analysis.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train_encoded = encoded_frame(run.model, run.X_train)
    X_test_encoded = encoded_frame(run.model, run.X_test)
    rf = run.model.named_steps["model"]
    explainer = LimeTabularExplainer(
        training_data=X_train_encoded.values,
        feature_names=X_train_encoded.columns.tolist(),
        class_names=["Did not survive", "Survived"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    explanation = explainer.explain_instance(
        X_test_encoded.iloc[sample_index].values,
        rf.predict_proba,
        num_features=8,
    )
    html_path = output_dir / "lime_passenger.html"
    explanation.save_to_file(str(html_path))
    return {
        "html_path": html_path,
        "weights": explanation.as_list(),
        "sample_index": sample_index,
    }


def _group_rates(y_true: pd.Series, y_pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in sorted(groups.astype(str).unique()):
        mask = groups.astype(str) == group
        positives = y_pred[mask] == 1
        actual_positive = y_true[mask] == 1
        actual_negative = y_true[mask] == 0
        rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "selection_rate": float(positives.mean()) if mask.sum() else np.nan,
                "true_positive_rate": float((positives & actual_positive).sum() / max(actual_positive.sum(), 1)),
                "false_positive_rate": float((positives & actual_negative).sum() / max(actual_negative.sum(), 1)),
            }
        )
    return pd.DataFrame(rows)


def fairness_audit(run: TitanicRun) -> dict[str, pd.DataFrame | float]:
    age = run.X_test["Age"].fillna(run.X_train["Age"].median())
    age_group = pd.cut(
        age,
        bins=[0, 18, 60, 120],
        labels=["child", "adult", "senior"],
        include_lowest=True,
    ).astype(str)

    gender_rates = _group_rates(run.y_test, run.y_pred, run.X_test["Sex"])
    age_rates = _group_rates(run.y_test, run.y_pred, age_group)

    def parity_gap(rates: pd.DataFrame, column: str) -> float:
        values = rates[column].dropna()
        return float(values.max() - values.min()) if len(values) else float("nan")

    return {
        "gender_rates": gender_rates,
        "age_rates": age_rates,
        "gender_demographic_parity_gap": parity_gap(gender_rates, "selection_rate"),
        "gender_equalized_odds_gap": max(
            parity_gap(gender_rates, "true_positive_rate"),
            parity_gap(gender_rates, "false_positive_rate"),
        ),
        "age_demographic_parity_gap": parity_gap(age_rates, "selection_rate"),
        "age_equalized_odds_gap": max(
            parity_gap(age_rates, "true_positive_rate"),
            parity_gap(age_rates, "false_positive_rate"),
        ),
    }


def passenger_template(
    pclass: str = "3",
    sex: str = "male",
    age: float = 30.0,
    sibsp: int = 0,
    parch: int = 0,
    fare: float = 12.0,
    embarked: str = "S",
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked,
            }
        ]
    )
    return clean_features(row)


def predict_passenger(run: TitanicRun, passenger: pd.DataFrame) -> dict[str, Any]:
    probability = float(run.model.predict_proba(passenger)[:, 1][0])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "probability_survived": probability,
        "label": "Survived" if prediction else "Did not survive",
    }


def counterfactual_search(run: TitanicRun, passenger: pd.DataFrame, target: int = 1) -> pd.DataFrame:
    base = passenger.iloc[0].copy()
    candidates: list[dict[str, Any]] = []
    pclass_options = ["1", "2", "3"]
    sex_options = ["female", "male"]
    embarked_options = ["C", "Q", "S"]
    fare_options = [7.25, 15.0, 30.0, 60.0, 100.0]
    age_options = [8.0, 18.0, 30.0, 45.0, 65.0]

    for pclass in pclass_options:
        for sex in sex_options:
            for embarked in embarked_options:
                for fare in fare_options:
                    for age in age_options:
                        candidate = base.copy()
                        candidate["Pclass"] = pclass
                        candidate["Sex"] = sex
                        candidate["Embarked"] = embarked
                        candidate["Fare"] = fare
                        candidate["Age"] = age
                        candidate["FamilySize"] = candidate["SibSp"] + candidate["Parch"] + 1
                        candidate["IsAlone"] = int(candidate["FamilySize"] == 1)
                        candidate_df = pd.DataFrame([candidate])[MODEL_FEATURES]
                        result = predict_passenger(run, candidate_df)
                        changed = [
                            col
                            for col in ["Pclass", "Sex", "Age", "Fare", "Embarked"]
                            if str(candidate[col]) != str(base[col])
                        ]
                        candidates.append(
                            {
                                "prediction": result["prediction"],
                                "probability_survived": result["probability_survived"],
                                "changes": ", ".join(changed) if changed else "none",
                                "Pclass": pclass,
                                "Sex": sex,
                                "Age": age,
                                "Fare": fare,
                                "Embarked": embarked,
                            }
                        )

    results = pd.DataFrame(candidates)
    desired = results[results["prediction"] == target].copy()
    if desired.empty:
        desired = results.copy()
    desired["num_changes"] = desired["changes"].apply(lambda text: 0 if text == "none" else len(text.split(", ")))
    return desired.sort_values(["num_changes", "probability_survived"], ascending=[True, False]).head(10)


def demo_passenger_for_counterfactual(run: TitanicRun) -> pd.DataFrame:
    predictions = pd.Series(run.y_pred, index=run.X_test.index)
    non_survivors = predictions[predictions == 0]
    if non_survivors.empty:
        return run.X_test.iloc[[0]].copy()
    return run.X_test.loc[[non_survivors.index[0]]].copy()


if __name__ == "__main__":
    run = train_titanic_model()
    print("Titanic RandomForest metrics")
    for key, value in run.metrics.items():
        print(f"{key}: {value:.3f}")
    print("\nTop model feature importances")
    print(model_feature_importance(run).to_string(index=False))
    print("\nFairness audit")
    audit = fairness_audit(run)
    print(audit["gender_rates"].to_string(index=False))
    print(audit["age_rates"].to_string(index=False))


from titanic_quality_extension import apply_titanic_quality_fixes
apply_titanic_quality_fixes(globals())
