"""Optional, opt-in runner for trusted teaching projects."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .schema import Finding


def run_study(project_root: Path, timeout: int = 180) -> list[Finding]:
    study = project_root / "study.py"
    if not study.exists():
        return [Finding("execution", "medium", "No study.py entry point", "No trusted batch entry point was found.", recommendation="Provide a deterministic study.py or document the executable entry point.")]
    try:
        completed = subprocess.run([sys.executable, str(study)], cwd=project_root, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return [Finding("execution", "high", "Study execution timed out", f"study.py exceeded {timeout} seconds.", recommendation="Bound expensive explanation jobs and cache sampled results.")]
    if completed.returncode:
        return [Finding("execution", "high", "Study execution failed", completed.stderr[-1200:], recommendation="Fix the exception and add a smoke test before distributing the project.")]
    return []
