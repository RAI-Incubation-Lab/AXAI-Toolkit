# Titanic XAI Worked Example

This folder is Deliverable D from the TA Playbook: a complete worked example showing how a finished XAI submission can look.

## What This Demo Does

- Trains a `RandomForestClassifier` on the Kaggle Titanic dataset.
- Cleans missing values and engineers simple family-size features.
- Reports accuracy, precision, recall, and F1.
- Generates global SHAP and local SHAP waterfall explanations.
- Generates a LIME local explanation for one passenger.
- Audits fairness across gender and age groups.
- Searches for counterfactual passenger profiles that would change a prediction.
- Provides a Streamlit app where a user can enter passenger details and inspect prediction, explanation, fairness, and counterfactual output.

## Files

| File | Purpose |
|---|---|
| `demo-titanic.ipynb` | Completed notebook for the worked example |
| `titanic_demo.py` | Shared data, model, explanation, fairness, and counterfactual utilities |
| `app.py` | Streamlit dashboard |
| `MODEL_CARD.md` | Completed model card with real metrics |
| `requirements.txt` | Local Python dependencies |

## Reproduce Locally

Run these commands from this folder:

```bash
python -m pip install -r requirements.txt
python titanic_demo.py
streamlit run app.py
```

The dataset is expected at the repository root as:

```text
Titanic-Dataset.csv
```

## Current Metrics

The demo uses a stratified 80/20 split with `random_state=42`.

| Metric | Value |
|---|---:|
| Accuracy | 0.788 |
| Precision | 0.725 |
| Recall | 0.725 |
| F1 | 0.725 |

## Fairness Summary

The model predicts survival much more often for female passengers than male passengers. This reflects a real signal in the historical labels but is still a sensitive-attribute concern for students to discuss.

| Sensitive feature | Demographic parity gap | Equalized odds gap |
|---|---:|---:|
| Gender | 0.684 | 0.600 |
| Age group | 0.317 | 0.367 |

## Notes for Students

This model is a teaching artifact, not a deployable survival decision system. Counterfactuals involving sex are included only to expose model dependence on a sensitive attribute; they should not be presented as actionable recommendations.


## Notebook-First Learning Workflow

Open `demo-titanic.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab after placing the dataset folder in the runtime. The notebook contains the full step-by-step data processing, model training, evaluation, and explanation workflow. `app.py` remains an optional Streamlit dashboard entry point.
