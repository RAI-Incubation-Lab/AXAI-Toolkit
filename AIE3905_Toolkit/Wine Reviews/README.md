# Wine Reviews XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/zynicide/wine-reviews

## Learning Task

Text classification of whether a review has at least 90 points.

**Target or ranking objective:** `high_score = 1` when `points >= 90`

## Why This XAI Method Fits

This case uses **Inherently interpretable linear token attribution**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Remove rows without review text or score, then create a high-score label from the editorial points field.
2. Concatenate description with country, province, and variety metadata.
3. Transform text with TF-IDF unigrams and bigrams, then fit class-balanced logistic regression on a stratified training split.
4. Explain a prediction directly through token contribution: TF-IDF value times the learned linear coefficient; compare country-level predicted and observed rates.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Country-level sampling diagnostic

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

Review language encodes editorial conventions and geographic coverage. Token coefficients are correlations in this corpus, not universal indicators of wine quality.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
