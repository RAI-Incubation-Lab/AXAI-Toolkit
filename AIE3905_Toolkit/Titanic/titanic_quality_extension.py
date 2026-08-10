"""Quality upgrades applied to the original Titanic teaching implementation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def apply_titanic_quality_fixes(namespace: dict[str, Any]) -> None:
    """Install safer evaluation and explanation functions into titanic_demo."""
    original_train = namespace["train_titanic_model"]
    original_fairness = namespace["fairness_audit"]
    model_features = namespace["MODEL_FEATURES"]
    numeric_features = namespace["NUMERIC_FEATURES"]
    categorical_features = namespace["CATEGORICAL_FEATURES"]
    random_state = namespace["RANDOM_STATE"]
    demo_dir = namespace["DEMO_DIR"]

    def train_titanic_model(*args: Any, **kwargs: Any) -> Any:
        run = original_train(*args, **kwargs)
        baseline = DummyClassifier(strategy="most_frequent").fit(run.X_train, run.y_train)
        baseline_pred = baseline.predict(run.X_test)
        run.metrics.update(
            {
                "roc_auc": roc_auc_score(run.y_test, run.y_proba),
                "baseline_accuracy": accuracy_score(run.y_test, baseline_pred),
                "accuracy_improvement": accuracy_score(run.y_test, run.y_pred) - accuracy_score(run.y_test, baseline_pred),
            }
        )
        return run

    def shap_analysis(run: Any, output_dir: str | Any = demo_dir / "outputs", sample_index: int = 0, max_samples: int = 200) -> dict[str, Any]:
        try:
            import matplotlib.pyplot as plt
            import shap
        except ImportError as exc:
            raise RuntimeError("Install shap and matplotlib to run SHAP analysis.") from exc
        output = namespace["Path"](output_dir)
        output.mkdir(parents=True, exist_ok=True)
        sample = run.X_test.sample(n=min(max_samples, len(run.X_test)), random_state=random_state)
        encoded = namespace["encoded_frame"](run.model, sample)
        explainer = shap.TreeExplainer(run.model.named_steps["model"])
        raw_values = explainer.shap_values(encoded)
        if isinstance(raw_values, list):
            values, base_value = raw_values[1], explainer.expected_value[1]
        elif getattr(raw_values, "ndim", 0) == 3:
            values, base_value = raw_values[:, :, 1], explainer.expected_value[1]
        else:
            values, base_value = raw_values, explainer.expected_value
        summary_path, waterfall_path = output / "shap_summary.png", output / "shap_waterfall_passenger.png"
        shap.summary_plot(values, encoded, show=False, max_display=12)
        plt.tight_layout(); plt.savefig(summary_path, dpi=160, bbox_inches="tight"); plt.close()
        selected = min(sample_index, len(encoded) - 1)
        explanation = shap.Explanation(values=values[selected], base_values=base_value, data=encoded.iloc[selected].values, feature_names=encoded.columns.tolist())
        shap.waterfall_plot(explanation, show=False, max_display=12)
        plt.tight_layout(); plt.savefig(waterfall_path, dpi=160, bbox_inches="tight"); plt.close()
        importance = pd.DataFrame({"feature": encoded.columns, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return {"summary_path": summary_path, "waterfall_path": waterfall_path, "importance": importance, "sample_index": selected, "sample_size": len(encoded), "random_state": random_state}

    def lime_analysis(run: Any, output_dir: str | Any = demo_dir / "outputs", sample_index: int = 0) -> dict[str, Any]:
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as exc:
            raise RuntimeError("Install lime to run LIME analysis.") from exc
        categories = {feature: sorted(run.X_train[feature].fillna("Unknown").astype(str).unique().tolist()) for feature in categorical_features}

        def matrix(frame: pd.DataFrame) -> np.ndarray:
            converted = frame[model_features].copy()
            for feature in numeric_features:
                converted[feature] = pd.to_numeric(converted[feature], errors="coerce").fillna(0.0)
            for feature in categorical_features:
                converted[feature] = pd.Categorical(converted[feature].fillna("Unknown").astype(str), categories=categories[feature]).codes
            return converted.to_numpy(dtype=float)

        def predict_fn(values: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(values, columns=model_features)
            for feature in numeric_features:
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce").clip(lower=0)
            for feature in categorical_features:
                codes = np.rint(frame[feature]).astype(int).clip(0, len(categories[feature]) - 1)
                frame[feature] = [categories[feature][code] for code in codes]
            return run.model.predict_proba(frame[model_features])

        explainer = LimeTabularExplainer(
            matrix(run.X_train), feature_names=model_features, class_names=["Did not survive", "Survived"], mode="classification",
            categorical_features=[model_features.index(feature) for feature in categorical_features],
            categorical_names={model_features.index(feature): categories[feature] for feature in categorical_features}, random_state=random_state,
        )
        explanation = explainer.explain_instance(matrix(run.X_test)[sample_index], predict_fn, num_features=8)
        output = namespace["Path"](output_dir); output.mkdir(parents=True, exist_ok=True)
        html_path = output / "lime_passenger.html"; explanation.save_to_file(str(html_path))
        return {"html_path": html_path, "weights": explanation.as_list(), "sample_index": sample_index, "fidelity": float(explanation.score)}

    def fairness_audit(run: Any) -> dict[str, Any]:
        audit = original_fairness(run)
        age = run.X_test["Age"].fillna(run.X_train["Age"].median())
        group_series = {"gender_rates": run.X_test["Sex"].astype(str), "age_rates": pd.cut(age, bins=[0, 18, 60, 120], labels=["child", "adult", "senior"], include_lowest=True).astype(str)}
        for key, groups in group_series.items():
            rows = []
            for group in sorted(groups.unique()):
                mask = groups == group; actual = run.y_test[mask]; predicted = run.y_pred[mask.to_numpy()]
                positive = predicted == 1; true_positive = actual.to_numpy() == 1; true_negative = ~true_positive
                rows.append({"group": group, "n": int(mask.sum()), "selection_rate": float(positive.mean()), "true_positive_rate": float((positive & true_positive).sum() / max(true_positive.sum(), 1)), "false_positive_rate": float((positive & true_negative).sum() / max(true_negative.sum(), 1)), "precision": precision_score(actual, predicted, zero_division=0), "recall": recall_score(actual, predicted, zero_division=0), "small_sample_warning": bool(mask.sum() < 30)})
            audit[key] = pd.DataFrame(rows)
        audit["warnings"] = ["This is a subgroup diagnostic, not a formal fairness conclusion. Groups with fewer than 30 test records must not be interpreted without uncertainty analysis."]
        return audit

    def counterfactual_analysis(run: Any, passenger: pd.DataFrame, target: int = 1) -> dict[str, pd.DataFrame]:
        base = passenger.iloc[0].copy(); feasible, sensitivity = [], []
        for pclass in ["1", "2", "3"]:
            for embarked in ["C", "Q", "S"]:
                for fare in [7.25, 15.0, 30.0, 60.0, 100.0]:
                    valid = not ((pclass == "1" and fare < 30) or (pclass == "2" and fare < 15))
                    if not valid: continue
                    candidate = base.copy(); candidate["Pclass"], candidate["Embarked"], candidate["Fare"] = pclass, embarked, fare
                    result = namespace["predict_passenger"](run, pd.DataFrame([candidate])[model_features])
                    changed = [item for item in ["Pclass", "Fare", "Embarked"] if str(candidate[item]) != str(base[item])]
                    feasible.append({"prediction": result["prediction"], "probability_survived": result["probability_survived"], "target_achieved": bool(result["prediction"] == target), "constraint_valid": valid, "change_cost": len(changed), "changes": ", ".join(changed) or "none", "Pclass": pclass, "Fare": fare, "Embarked": embarked})
        for sex in ["female", "male"]:
            for age in [8.0, 18.0, 30.0, 45.0, 65.0]:
                candidate = base.copy(); candidate["Sex"], candidate["Age"] = sex, age
                result = namespace["predict_passenger"](run, pd.DataFrame([candidate])[model_features])
                sensitivity.append({"Sex": sex, "Age": age, "prediction": result["prediction"], "probability_survived": result["probability_survived"], "non_actionable": True})
        table = pd.DataFrame(feasible)
        return {"actionable_counterfactuals": table[table.target_achieved].sort_values(["change_cost", "probability_survived"], ascending=[True, False]).head(10), "protected_attribute_sensitivity": pd.DataFrame(sensitivity), "infeasible_candidates": table[~table.target_achieved].sort_values("probability_survived", ascending=False).head(10)}

    namespace.update({"train_titanic_model": train_titanic_model, "shap_analysis": shap_analysis, "lime_analysis": lime_analysis, "fairness_audit": fairness_audit, "counterfactual_analysis": counterfactual_analysis, "counterfactual_search": lambda run, passenger, target=1: counterfactual_analysis(run, passenger, target)["actionable_counterfactuals"]})
