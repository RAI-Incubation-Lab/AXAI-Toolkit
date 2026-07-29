# Google Play Store Apps XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/lava18/google-play-store-apps

## Learning Task

Binary classification of apps rated 4.2 or higher.

**Target or ranking objective:** `high_rating`

## Why This XAI Method Fits

This case uses **Tree SHAP and LIME after robust type cleaning**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Deduplicate apps, remove rows without ratings, and derive `high_rating` from the 4.2 threshold.
2. Convert Reviews, Installs, Price, and Size from strings into numeric values and derive last-updated year.
3. Fit a class-balanced RandomForestClassifier with imputation and one-hot encoding inside the training pipeline.
4. Use Tree SHAP and LIME, then compare predicted positive rate and accuracy between free and paid applications.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Free versus paid app context

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

Ratings are affected by survivorship, popularity, review manipulation, and missing data. The classifier does not measure intrinsic app quality.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
