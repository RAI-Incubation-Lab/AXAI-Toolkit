# Chess Game Dataset (Lichess) XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/datasnaek/chess

## Learning Task

Binary classification of a White win in decisive games.

**Target or ranking objective:** `white_win`

## Why This XAI Method Fits

This case uses **Tree SHAP and LIME for tabular classification**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Filter to games won by White or Black and derive `white_win`; draws are intentionally outside this task.
2. Create the rating-difference feature and keep game metadata such as time control, opening, game length, and victory status.
3. Impute and one-hot encode inside the training pipeline, use a stratified test split, and fit a RandomForestClassifier.
4. Use Tree SHAP and LIME to inspect individual games; compare accuracy and predicted White-win rate for rated and casual games.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Rated versus casual context

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

Move sequences and tactical position evaluation are excluded, so the model should not be used as a chess-strength or opening-quality evaluator.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
