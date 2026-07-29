# Students Performance in Exams XAI Case Study

**Kaggle source:** https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

## Learning Task

Regression of mathematics score from pre-test background variables.

**Target or ranking objective:** `math score`

## Why This XAI Method Fits

This case uses **Tree SHAP and LIME with subgroup-error audit**. The method is selected for the data and task rather than applying the same explanation tool indiscriminately. Read the case `study.py` alongside this guide: it contains the executable processing logic, while this document explains the methodological choices.

## Processing Workflow

1. Load the 1,000 student records and retain gender, race/ethnicity, parental education, lunch, and preparation status.
2. Exclude reading and writing scores because they are co-measured exam outcomes and would turn the exercise into outcome profiling.
3. One-hot encode categorical fields in the training pipeline and fit a RandomForestRegressor on an 80/20 split.
4. Use Tree SHAP and LIME, then compare error by gender while discussing whether sensitive attributes should be included at all.

## Run the Application

From this dataset folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For the tabular cases, the dashboard can generate Tree SHAP and LIME output after `shap`, `lime`, and `matplotlib` are installed. The Wine and Netflix cases use transparent text/similarity explanations instead.

## Audit and Interpretation

**Audit used:** Gender-group MAE

The audit is a diagnostic, not proof that the application is fair, unbiased, or suitable for deployment. Inspect sample sizes, error distributions, historical context, and whether each feature is appropriate for the proposed use.

## Responsible-Use Note

This small, synthetic-style dataset must not support placement, ability labelling, discipline, or automated educational decisions.

## Suggested Student Exercises

1. Run the baseline and record the held-out metrics.
2. Select two contrasting records and compare their local explanations.
3. Identify one possible source of leakage, sampling bias, or proxy discrimination.
4. Modify one defensible preprocessing choice and document how explanations change.


## Notebook-First Learning Workflow

Open `xai_workflow.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
