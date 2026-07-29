# Medical Cost Personal Datasets XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/mirichoi0218/insurance

## Learning Task

Regression of recorded insurance charges.

**Target or ranking objective:** `charges`

## Why This XAI Method Fits

This case uses **Tree SHAP and LIME with sensitive-group error audit**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Load age, sex, BMI, children, smoking status, region, and charges.
2. Use median imputation for numeric inputs and one-hot encoding for categorical inputs inside the train-only pipeline.
3. Fit RandomForestRegressor and evaluate MAE, RMSE, and R-squared on a held-out test set.
4. Use Tree SHAP and LIME, then compare group-level error by sex with a clear responsible-use warning.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Sex-group MAE

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

This healthcare-cost example must not be used for underwriting, care decisions, pricing, or automated profiling. Explanations do not justify historical inequities.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
