# Model Card - FIFA World Cup 2026 Player Performance XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset |
| Learning task | Regression of match-level `performance_score`. |
| Target / objective | `performance_score` |
| Explanation method | Tree SHAP and LIME for tabular regression |
| Audit | Position-group MAE |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Load 54,600 player-match records and retain modelled match statistics plus contextual fields.
2. Use median imputation for numeric fields and one-hot encoding for position, preferred foot, tournament stage, and match result.
3. Hold out 20% of records for testing, fit a RandomForestRegressor, and report MAE, RMSE, and R-squared.
4. Use Tree SHAP globally and LIME locally; compare regression error by player position.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Tree SHAP and LIME for tabular regression. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

The labelled score may itself be built from match statistics, so attribution can partly reveal the dataset's scoring construction rather than independent player quality.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
