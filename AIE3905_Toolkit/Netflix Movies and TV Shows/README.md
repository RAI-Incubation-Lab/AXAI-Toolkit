# Netflix Movies and TV Shows XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/shivamb/netflix-shows

## Learning Task

Explainable content-based recommendation.

**Target or ranking objective:** No supervised target; rank titles by content similarity.

## Why This XAI Method Fits

This case uses **Transparent TF-IDF cosine-similarity decomposition**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Deduplicate shows and combine type, genres, description, country, director, and cast into a content field.
2. Fit a TF-IDF vocabulary over the catalogue.
3. For a selected title, calculate cosine similarity against every other title and return the highest-ranked recommendations.
4. Explain every recommendation with the strongest shared TF-IDF terms, then inspect catalogue coverage by movie versus TV-show type.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Catalogue coverage summary

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

Similarity reflects available metadata, not user preference, quality, safety, or cultural appropriateness. Missing cast/director/country data can change rankings.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
