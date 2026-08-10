# AXAI Review Harness

The harness evaluates whether a student project demonstrates explainability across its code, documentation, inputs, outputs, validation design, model validity, fairness diagnostics, and counterfactual claims. It does not assume that producing a SHAP or LIME chart alone proves explainability.

## Review a student project

Run the static review from the toolkit root. This mode does not import or execute student code.

```powershell
python axai_harness.py audit "Titanic"
```

The command writes `review_output/axai_review.md` and `review_output/axai_review.json` inside the target project. The JSON report is suitable for a marking or feedback workflow.

For a trusted classroom submission, explicitly run the deterministic `study.py` entry point with a timeout:

```powershell
python axai_harness.py audit "Titanic" --execute --timeout 180
```

Never use `--execute` for untrusted code. Run untrusted submissions in an isolated environment first.

## Review all bundled examples

```powershell
python axai_harness.py audit-all
```

## Create a new standalone example

```powershell
python axai_harness.py create-example "My New Example" --data "C:\data\example.csv" --target outcome --task classification
```

The generator copies the CSV by default and creates `study.py`, `README.md`, `MODEL_CARD.md`, and `requirements.txt`. Students then add a notebook and explanation evidence. Use `--no-copy-data` only when redistribution is prohibited; document the dataset download path prominently.

## What the score means

The score is a teaching triage signal, not a certification. It penalizes missing evidence in seven areas: documentation, validation/leakage, performance, explainability, fairness, counterfactual reasoning, and evidence artifacts. Review the written recommendations before assigning marks.

## Codex or Workbuddy use

Ask an agent: "Use the AXAI project-review Skill to audit `<project path>`, explain only evidence-backed findings, and write the report." The local Skill delegates to this harness, so the commands and report format remain consistent.
