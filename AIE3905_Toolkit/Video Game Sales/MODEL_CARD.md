# Model Card - Video Game Sales XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/gregorut/videogamesales |
| Learning task | Regression of global game sales in millions. |
| Target / objective | `Global_Sales` |
| Explanation method | Tree SHAP and LIME with explicit leakage control |
| Audit | Genre-level MAE |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Keep records with global sales and clean the release year.
2. Use year, platform, genre, and publisher as predictors.
3. Deliberately exclude regional sales and rank because regional sales sum into the global-sales target and would leak the answer.
4. Fit RandomForestRegressor, explain with Tree SHAP and LIME, and compare MAE by genre.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Tree SHAP and LIME with explicit leakage control. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

Historical sales reflect market structure and reporting practices. They cannot forecast a future title's commercial performance without richer, time-appropriate data.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
