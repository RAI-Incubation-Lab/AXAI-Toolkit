# AIE3905 AXAI Toolkit: Complete Usage Guide

## 1. Choose your role

- **Student:** open one dataset folder, follow its notebook, then use its dashboard.
- **Instructor or TA:** run the harness to review evidence and create feedback reports.
- **Maintainer:** regenerate execution evidence and run CI checks after changing code or data.

## 2. Install the verified environment

Use Python 3.10-3.12. From the toolkit root, install the pinned environment:

```powershell
# Run this command from the toolkit root directory.
# E.g., ./path/to/your/AIE3905_Toolkit
python -m pip install -r requirements-lock.txt
```

Use `requirements-runtime.txt` for application/reviewer-only use, or `requirements-notebook.txt` when notebook execution is also needed.

## 3. Learn one self-contained case

Each dataset directory can be shared independently. Open its `README.md` first, then use its notebook and dashboard.

```powershell
# In ./path/to/your/AIE3905_Toolkit
cd "Titanic"
jupyter notebook demo-titanic.ipynb
streamlit run app.py
```

For other projects, the notebook is normally named `xai_workflow.ipynb`:

```powershell
# In ./path/to/your/AIE3905_Toolkit
cd "Wine Reviews"
jupyter notebook xai_workflow.ipynb
streamlit run app.py
```

Read `MODEL_CARD.md` for intended use and limitations. `RUN_EVIDENCE.md` records metrics, validation, a representative input/output item, and data/model identifiers from the latest trusted execution.

## 4. Understand saved evidence

Every case stores reproducible execution artifacts in `outputs/`:

- `metrics.json`: model metrics, validation split, and validity-gate result.
- `representative_evidence.json`: raw input, expected output, prediction, explanation method, and model/data hashes.
- SHAP/LIME output when that method is applicable.

Do not treat an explanation as decision-ready when the model-validity gate does not pass.

## 5. Review a student project

Run a static review first. It does not import or execute student code.

```powershell
# Run this command from the toolkit root directory.
python axai_harness.py audit "Titanic"
```

The report is written to the target project under `review_output/axai_review.md` and `review_output/axai_review.json`.

For a trusted local submission, use isolated execution. The harness creates a temporary environment, installs the submission requirements, and runs `study.py` with a timeout.

```powershell
python axai_harness.py audit "C:\path\to\trusted_student_project" --execute --timeout 180
```

Do not use `--execute` for untrusted code outside a controlled environment.

Review all bundled cases with:

```powershell
python axai_harness.py audit-all
```

## 6. Create a new teaching example

Create an independent CSV-based project:

```powershell
python axai_harness.py create-example "My New Example" `
  --data "C:\path\to\data.csv" `
  --target outcome `
  --task classification
```

Use `regression` for a continuous target. The generator creates data, code, documentation, Model Card, and requirements. Add a notebook, representative evidence, limitations, and appropriate validation before using it in class.

## 7. Bitcoin data

The repository includes a small synthetic fixture so the Bitcoin example runs from a fresh clone. To use the full Kaggle data, configure the Kaggle CLI and run:

```powershell
cd "Bitcoin Historical"
python download_dataset.py
```

Use `python download_dataset.py --verify-only` after manually placing the CSV. Do not commit the downloaded minute-level data.

## 8. Maintain the toolkit

After changing trusted example code or data, regenerate evidence and the derived Markdown documents:

```powershell
# Run this command from the toolkit root directory.
python build_execution_evidence.py
python render_run_docs.py
python axai_harness.py audit-all
```

For the six synchronized tabular examples, update the template and apply it with:

```powershell
python sync_tabular_examples.py --apply
```

Then regenerate evidence again. GitHub Actions performs dependency installation from the lockfile, tests, evidence generation, documentation-drift checks, and toolkit audits.

## 9. Codex use

The local `axai-project-review` Skill uses the same harness. Ask Codex to audit a specific project and generate evidence-backed recommendations; it should begin with static review and only execute trusted submissions explicitly.
