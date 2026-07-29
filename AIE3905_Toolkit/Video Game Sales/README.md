# Video Game Sales XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/gregorut/videogamesales

## Learning Task

Regression of global game sales in millions.

**Target or ranking objective:** `Global_Sales`

## Why This XAI Method Fits

This case uses **Tree SHAP and LIME with explicit leakage control**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Keep records with global sales and clean the release year.
2. Use year, platform, genre, and publisher as predictors.
3. Deliberately exclude regional sales and rank because regional sales sum into the global-sales target and would leak the answer.
4. Fit RandomForestRegressor, explain with Tree SHAP and LIME, and compare MAE by genre.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Genre-level MAE

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

Historical sales reflect market structure and reporting practices. They cannot forecast a future title's commercial performance without richer, time-appropriate data.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
