# Model Card - Google Play Store Apps XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/lava18/google-play-store-apps |
| Learning task | Binary classification of apps rated 4.2 or higher. |
| Target / objective | `high_rating` |
| Explanation method | Tree SHAP and LIME after robust type cleaning |
| Audit | Free versus paid app context |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Deduplicate apps, remove rows without ratings, and derive `high_rating` from the 4.2 threshold.
2. Convert Reviews, Installs, Price, and Size from strings into numeric values and derive last-updated year.
3. Fit a class-balanced RandomForestClassifier with imputation and one-hot encoding inside the training pipeline.
4. Use Tree SHAP and LIME, then compare predicted positive rate and accuracy between free and paid applications.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Tree SHAP and LIME after robust type cleaning. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

Ratings are affected by survivorship, popularity, review manipulation, and missing data. The classifier does not measure intrinsic app quality.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
