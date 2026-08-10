"""Conservative static checks for common validation and leakage risks."""
from __future__ import annotations

import re

from .schema import Finding


def check_leakage(code: str) -> list[Finding]:
    findings: list[Finding] = []
    if "train_test_split" in code and not re.search(r"Group(?:KFold|ShuffleSplit)|group[_=]|TimeSeriesSplit|time[_=]|date[_=]", code, re.I):
        findings.append(Finding("validation", "medium", "Only a random split was detected", "A random split can leak repeated entities, players, products, users, or time into the test set.", evidence="train_test_split detected without a group or temporal split.", recommendation="Justify the split and add GroupKFold/group holdout or temporal validation when entities or time are present."))
    if re.search(r"(?:fit_transform|\.fit\()\s*\([^\n]*\)\s*#?\s*(?:before|prior to)?", code) and "train_test_split" in code:
        findings.append(Finding("validation", "low", "Review preprocessing order", "Static analysis cannot prove that preprocessing is fit only on training data.", recommendation="Keep preprocessing inside a sklearn Pipeline and fit the pipeline only on training data."))
    if re.search(r"(?:target|label|outcome|survived|rating)\s*\]", code, re.I) and re.search(r"features\s*=\s*df", code, re.I):
        findings.append(Finding("validation", "low", "Review target leakage manually", "Feature selection appears to be defined near the target; verify that post-outcome or target-derived columns are excluded.", recommendation="Document feature availability at prediction time and add a leakage test."))
    return findings
