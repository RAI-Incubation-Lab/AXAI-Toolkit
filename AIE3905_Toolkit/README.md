# AIE3905 Explainable AI Toolkit

Each dataset folder is a standalone teaching project; Bitcoin includes a small synthetic fixture and an optional verified Kaggle download for the full dataset. You can send any one dataset folder to students without `shared`, `tools`, or `Case_Studies`.

Every project folder contains `study.py`, `app.py`, `README.md`, `MODEL_CARD.md`, and `requirements.txt`. Run it from inside that folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Projects: FIFA World Cup 2026, Wine Reviews, Chess Game, Netflix Movies and TV Shows, Video Game Sales, Students Performance in Exams, Google Play Store Apps, Bitcoin Historical, Medical Cost Personal, and Titanic.

The source code and learning materials are in English so the folders are ready for an English-taught course.

## AXAI project review and authoring

This toolkit now includes a teacher-facing review harness. It evaluates evidence of explainability across code, documentation, input data, prediction and explanation artifacts, validation, model validity, subgroup diagnostics, and counterfactual claims.

```powershell
python -m pip install -r requirements-runtime.txt
python axai_harness.py audit "Titanic"
python axai_harness.py audit-all
```

Read [HARNESS_GUIDE.md](HARNESS_GUIDE.md) before using `--execute`: static audit is the default and execution must be restricted to trusted student code. Use `create_example.py` or `axai_harness.py create-example` to make a new standalone CSV-based teaching example.

`requirements-runtime.txt` contains supported version ranges for applications and reviewer use. `requirements-notebook.txt` adds Jupyter execution dependencies. Supported Python versions are 3.10-3.12. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) before redistributing any dataset.