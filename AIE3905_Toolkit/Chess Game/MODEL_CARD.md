# Model Card - Chess Game Dataset (Lichess) XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/datasnaek/chess |
| Learning task | Binary classification of a White win in decisive games. |
| Target / objective | `white_win` |
| Explanation method | Tree SHAP and LIME for tabular classification |
| Audit | Rated versus casual context |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Filter to games won by White or Black and derive `white_win`; draws are intentionally outside this task.
2. Create the rating-difference feature and keep game metadata such as time control, opening, game length, and victory status.
3. Impute and one-hot encode inside the training pipeline, use a stratified test split, and fit a RandomForestClassifier.
4. Use Tree SHAP and LIME to inspect individual games; compare accuracy and predicted White-win rate for rated and casual games.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Tree SHAP and LIME for tabular classification. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

Move sequences and tactical position evaluation are excluded, so the model should not be used as a chess-strength or opening-quality evaluator.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
