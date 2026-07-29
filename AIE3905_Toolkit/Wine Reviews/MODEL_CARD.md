# Model Card - Wine Reviews XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/zynicide/wine-reviews |
| Learning task | Text classification of whether a review has at least 90 points. |
| Target / objective | `high_score = 1` when `points >= 90` |
| Explanation method | Inherently interpretable linear token attribution |
| Audit | Country-level sampling diagnostic |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Remove rows without review text or score, then create a high-score label from the editorial points field.
2. Concatenate description with country, province, and variety metadata.
3. Transform text with TF-IDF unigrams and bigrams, then fit class-balanced logistic regression on a stratified training split.
4. Explain a prediction directly through token contribution: TF-IDF value times the learned linear coefficient; compare country-level predicted and observed rates.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Inherently interpretable linear token attribution. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

Review language encodes editorial conventions and geographic coverage. Token coefficients are correlations in this corpus, not universal indicators of wine quality.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
