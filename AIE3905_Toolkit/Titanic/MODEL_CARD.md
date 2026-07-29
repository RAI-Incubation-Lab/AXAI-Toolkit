# Model Card - Titanic Survival XAI Demo

**Team:** RAI Incubation Lab TA Demo  
**AI Solution:** RandomForest classifier predicting Titanic passenger survival  
**Tech Stack:** Python, pandas, scikit-learn, SHAP, LIME, Streamlit  
**Dataset:** Kaggle Titanic dataset by Yasser H.  
**Date:** 2026-06-26  

---

## 1. Solution Overview

| Property | Value |
|---|---|
| Solution Type | Binary classification |
| Model | RandomForestClassifier |
| Dataset Size | 891 passengers |
| Target Variable | `Survived` |
| Input Features | `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `FamilySize`, `IsAlone` |
| Test Split | Stratified 80/20 split, `random_state=42` |

The model predicts whether a passenger survived the Titanic disaster using passenger ticket, demographic, family, and fare attributes. It is designed as a compact worked example for explainability and fairness education.

## 2. Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.788 |
| Precision | 0.725 |
| Recall | 0.725 |
| F1 Score | 0.725 |

The model has reasonable predictive performance for a small tabular teaching dataset. It should not be interpreted as a production decision system.

## 3. Explainability Analysis

### Tech Choice Justification

RandomForest was chosen because it performs well on small tabular datasets, supports class probabilities, and works naturally with Tree SHAP. A logistic regression baseline would be more transparent but less flexible. A neural network would add complexity without improving the teaching value of this example.

### Top Feature Drivers

| Rank | Feature | Importance | Why it matters |
|---|---|---:|---|
| 1 | `Sex_male` | 0.215 | The model heavily distinguishes male passengers from other passengers. |
| 2 | `Sex_female` | 0.205 | Female passengers had much higher observed survival rates in the data. |
| 3 | `Fare` | 0.180 | Fare is a proxy for ticket class, cabin location, and socioeconomic status. |
| 4 | `Age` | 0.130 | Children and older passengers show different survival patterns. |
| 5 | `Pclass_3` | 0.065 | Third-class passengers had lower survival rates. |

### SHAP

SHAP, short for SHapley Additive exPlanations, is a post-hoc explainability method based on Shapley values from cooperative game theory. It treats a model prediction as the outcome of a "game" where each input feature is a player, then assigns each feature a contribution score.

For a model prediction \(f(x)\), SHAP explains the output with an additive feature attribution model:

```text
g(z') = phi_0 + sum(phi_i * z'_i)
```

where:

- \(g(z')\) is the explanation model for one prediction.
- \(z'_i\) indicates whether feature \(i\) is present in the simplified input.
- \(\phi_0\) is the baseline prediction, usually the expected model output.
- \(\phi_i\) is the SHAP value for feature \(i\), meaning the estimated contribution of that feature to this prediction.

The Shapley value for feature \(i\) is:

```text
phi_i = sum over S subset of F \ {i} [
    |S|! * (|F| - |S| - 1)! / |F|!
    * (f_{S union {i}}(x_{S union {i}}) - f_S(x_S))
]
```

where:

- \(F\) is the full set of input features.
- \(S\) is a subset of features that does not include feature \(i\).
- \(f_S(x_S)\) is the model output when only feature subset \(S\) is known.
- The difference \(f_{S union {i}}(x_{S union {i}}) - f_S(x_S)\) is the marginal contribution of feature \(i\).
- The factorial term averages that marginal contribution over all possible feature orderings.

In this demo, Tree SHAP is used because the predictive model is a `RandomForestClassifier`. Tree SHAP computes exact or efficient Shapley-style attributions for tree-based models, making it faster and more stable than model-agnostic sampling methods on this tabular task.

The notebook generates:

- a SHAP summary plot for global feature importance;
- a SHAP waterfall plot for one passenger-level prediction.

Interpretation:

- A positive SHAP value pushes the prediction toward survival.
- A negative SHAP value pushes the prediction away from survival.
- Larger absolute SHAP values indicate stronger influence for that prediction.

Limitations:

- SHAP explains model behavior, not real-world causality.
- Correlated features can share or redistribute attribution in non-obvious ways.
- In this dataset, features such as `Sex`, `Pclass`, and `Fare` reflect historical social patterns, so high SHAP importance should be interpreted carefully.

### LIME

LIME, short for Local Interpretable Model-agnostic Explanations, is a post-hoc local explanation method. Instead of explaining the full RandomForest globally, LIME explains one prediction by fitting a simple interpretable surrogate model around the selected passenger.

For an original model \(f\) and one sample \(x\), LIME solves:

```text
explanation(x) = argmin over g in G [
    L(f, g, pi_x) + Omega(g)
]
```

where:

- \(f\) is the original black-box model.
- \(g\) is a simple interpretable model, often a sparse linear model.
- \(G\) is the family of interpretable explanation models.
- \(pi_x(z)\) is a proximity kernel that gives more weight to perturbed samples \(z\) that are close to the original sample \(x\).
- \(L(f, g, pi_x)\) measures how poorly the simple model \(g\) approximates \(f\) near \(x\).
- \(\Omega(g)\) penalizes explanation complexity, encouraging a short explanation with only a few important features.

For tabular data, LIME follows this process:

1. Select one passenger to explain.
2. Generate many perturbed versions of that passenger.
3. Ask the trained RandomForest to predict each perturbed sample.
4. Weight perturbed samples by how close they are to the original passenger.
5. Fit a simple local surrogate model.
6. Report the feature rules with the strongest local weights.

In this demo, LIME is applied to the one-hot encoded and imputed feature space used by the RandomForest. The output is a local list of feature rules and weights showing which conditions push the selected passenger's prediction toward or away from survival.

Interpretation:

- A positive LIME weight supports the predicted class shown for the passenger.
- A negative LIME weight pushes against that class.
- The explanation is local and should only be read as an approximation near the selected passenger.

Limitations:

- LIME can change if the random perturbations, discretization, or kernel width change.
- The local surrogate may not perfectly match the RandomForest decision boundary.
- Because this demo uses encoded features, some explanations appear as transformed feature rules rather than original human-friendly variables.

## 4. Counterfactual Analysis

The counterfactual search changes passenger attributes such as ticket class, fare, embarked port, age, and sex to find profiles that would change the model prediction to survival.

Important limitation: changing sex is included only as a diagnostic sensitive-attribute test. It is not an actionable intervention and should be discussed carefully in the fairness section.

## 5. Fairness Audit

### Gender

| Group | n | Selection Rate | True Positive Rate | False Positive Rate |
|---|---:|---:|---:|---:|
| female | 61 | 0.836 | 0.933 | 0.563 |
| male | 118 | 0.153 | 0.333 | 0.106 |

| Metric | Value |
|---|---:|
| Demographic parity gap | 0.684 |
| Equalized odds gap | 0.600 |

### Age Group

| Group | n | Selection Rate | True Positive Rate | False Positive Rate |
|---|---:|---:|---:|---:|
| adult | 142 | 0.373 | 0.692 | 0.189 |
| child | 31 | 0.484 | 0.867 | 0.125 |
| senior | 6 | 0.167 | 0.500 | 0.000 |

| Metric | Value |
|---|---:|
| Demographic parity gap | 0.317 |
| Equalized odds gap | 0.367 |

The gender gap is large. Some of this reflects historical evacuation patterns, but students should still document the sensitive-attribute dependence and discuss whether using `Sex` is appropriate for their task.

## 6. Limitations

- The dataset is small and historical.
- Cabin is mostly missing and not used.
- Ticket text and passenger names are excluded to keep the demo simple.
- Fairness metrics are unstable for small groups, especially senior passengers.
- Counterfactuals are simple grid searches, not causal explanations.

## 7. Stakeholder Notes

**Who should use this solution:** Students and instructors learning XAI workflows on tabular classification.  
**Who should not use this solution:** Anyone making real safety, rescue, insurance, or demographic decisions.  
**Regulatory considerations:** Sensitive attributes such as sex and age require careful justification in real systems.  
**Human oversight required:** Yes. Explanations and fairness metrics must be interpreted by humans; they do not prove the model is fair or causally valid.

## Academic Citation

If you use this toolkit in your research, please reference the software repository. Citation configuration will be updated upon formal publication approval.
