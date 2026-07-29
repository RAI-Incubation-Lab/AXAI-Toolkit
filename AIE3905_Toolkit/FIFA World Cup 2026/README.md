# FIFA World Cup 2026 Player Performance XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset

## Learning Task

Regression of match-level `performance_score`.

**Target or ranking objective:** `performance_score`

## Why This XAI Method Fits

This case uses **Tree SHAP and LIME for tabular regression**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Load 54,600 player-match records and retain modelled match statistics plus contextual fields.
2. Use median imputation for numeric fields and one-hot encoding for position, preferred foot, tournament stage, and match result.
3. Hold out 20% of records for testing, fit a RandomForestRegressor, and report MAE, RMSE, and R-squared.
4. Use Tree SHAP globally and LIME locally; compare regression error by player position.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Position-group MAE

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

The labelled score may itself be built from match statistics, so attribution can partly reveal the dataset's scoring construction rather than independent player quality.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
