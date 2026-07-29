# Model Card - Students Performance in Exams XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/spscientist/students-performance-in-exams |
| Learning task | Regression of mathematics score from pre-test background variables. |
| Target / objective | `math score` |
| Explanation method | Tree SHAP and LIME with subgroup-error audit |
| Audit | Gender-group MAE |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Load the 1,000 student records and retain gender, race/ethnicity, parental education, lunch, and preparation status.
2. Exclude reading and writing scores because they are co-measured exam outcomes and would turn the exercise into outcome profiling.
3. One-hot encode categorical fields in the training pipeline and fit a RandomForestRegressor on an 80/20 split.
4. Use Tree SHAP and LIME, then compare error by gender while discussing whether sensitive attributes should be included at all.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Tree SHAP and LIME with subgroup-error audit. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

This small, synthetic-style dataset must not support placement, ability labelling, discipline, or automated educational decisions.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
