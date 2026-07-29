# Model Card - Bitcoin Historical Data XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data |
| Learning task | Chronological binary classification of next-hour price direction. |
| Target / objective | `up_next_hour` |
| Explanation method | Lag-feature importance with chronological evaluation |
| Audit | Temporal hold-out evaluation |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Locate the Kaggle CSV in the Bitcoin Historical folder, parse timestamp/close/volume fields, sort chronologically, and aggregate to hourly observations.
2. Create lagged returns, rolling return mean, rolling volatility, and volume-change features using only current and past data.
3. Create the next-hour direction target and split chronologically: earliest 80% for training and latest 20% for testing.
4. Fit RandomForestClassifier and inspect lag-feature importance. Do not shuffle time series because that would leak future market information.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Lag-feature importance with chronological evaluation. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

The local Bitcoin data file is currently missing. After adding it, the application will run; results are educational only and are not financial advice or a trading system.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
