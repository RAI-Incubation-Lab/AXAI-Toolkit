# Model Card - Medical Cost Personal Datasets XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/mirichoi0218/insurance |
| Learning task | Regression of recorded insurance charges. |
| Target / objective | `charges` |
| Explanation method | Tree SHAP and LIME with sensitive-group error audit |
| Audit | Sex-group MAE |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Load age, sex, BMI, children, smoking status, region, and charges.
2. Use median imputation for numeric inputs and one-hot encoding for categorical inputs inside the train-only pipeline.
3. Fit RandomForestRegressor and evaluate MAE, RMSE, and R-squared on a held-out test set.
4. Use Tree SHAP and LIME, then compare group-level error by sex with a clear responsible-use warning.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Tree SHAP and LIME with sensitive-group error audit. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

This healthcare-cost example must not be used for underwriting, care decisions, pricing, or automated profiling. Explanations do not justify historical inequities.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
