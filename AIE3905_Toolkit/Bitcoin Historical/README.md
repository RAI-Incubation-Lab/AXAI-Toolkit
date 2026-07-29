# Bitcoin Historical Data XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data

## Learning Task

Chronological binary classification of next-hour price direction.

**Target or ranking objective:** `up_next_hour`

## Why This XAI Method Fits

This case uses **Lag-feature importance with chronological evaluation**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Locate the Kaggle CSV in the Bitcoin Historical folder, parse timestamp/close/volume fields, sort chronologically, and aggregate to hourly observations.
2. Create lagged returns, rolling return mean, rolling volatility, and volume-change features using only current and past data.
3. Create the next-hour direction target and split chronologically: earliest 80% for training and latest 20% for testing.
4. Fit RandomForestClassifier and inspect lag-feature importance. Do not shuffle time series because that would leak future market information.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Temporal hold-out evaluation

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

The local Bitcoin data file is currently missing. After adding it, the application will run; results are educational only and are not financial advice or a trading system.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.


## Dataset Download Guide

The raw minute-level Bitcoin file is intentionally excluded from this repository because it is too large for GitHub. Download it before running the notebook or dashboard:

1. Open the [Bitcoin Historical Data Kaggle dataset](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data).
2. Sign in to Kaggle and download the dataset archive.
3. Extract `btcusd_1-min_data.csv` directly into this `Bitcoin Historical` folder, beside `study.py`.
4. Run `python -m pip install -r requirements.txt`, then open `xai_workflow.ipynb`.
5. Keep the downloaded CSV and ZIP local. They are ignored by `.gitignore` and should not be committed to GitHub.
