# Model Card - Netflix Movies and TV Shows XAI Case

## Model Details

| Property | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/shivamb/netflix-shows |
| Learning task | Explainable content-based recommendation. |
| Target / objective | No supervised target; rank titles by content similarity. |
| Explanation method | Transparent TF-IDF cosine-similarity decomposition |
| Audit | Catalogue coverage summary |

## Intended Use

This is a teaching artifact for learning reproducible explainable AI workflows. Results must be interpreted with the data context and should not be used as automated real-world decisions.

## Training and Evaluation Process

1. Deduplicate shows and combine type, genres, description, country, director, and cast into a content field.
2. Fit a TF-IDF vocabulary over the catalogue.
3. For a selected title, calculate cosine similarity against every other title and return the highest-ranked recommendations.
4. Explain every recommendation with the strongest shared TF-IDF terms, then inspect catalogue coverage by movie versus TV-show type.

Metrics are generated when `study.py` runs and are displayed in `app.py`. The test partition is held out from fitting; the Bitcoin case uses chronological rather than random splitting.

## Explainability

The dashboard presents Transparent TF-IDF cosine-similarity decomposition. Explanations describe the behavior of the fitted model or similarity mechanism on the supplied dataset. They are not causal proof and can change when the data, preprocessing, model, or explanation settings change.

## Limitations and Risk

Similarity reflects available metadata, not user preference, quality, safety, or cultural appropriateness. Missing cast/director/country data can change rankings.

## Human Oversight

Required. A human must assess data quality, task appropriateness, subgroup impact, and the limits of any explanation before using results in discussion or decision-making.
