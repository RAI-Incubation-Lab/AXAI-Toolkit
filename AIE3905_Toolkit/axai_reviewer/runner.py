"""Opt-in isolated runner for trusted classroom projects."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from .schema import Finding


def run_study(project_root: Path, timeout: int = 180) -> list[Finding]:
    study = project_root / "study.py"
    requirements = project_root / "requirements.txt"
    if not study.exists():
        return [Finding("execution", "medium", "No study.py entry point", "No deterministic batch entry point was found.", recommendation="Provide study.py.")]
    with tempfile.TemporaryDirectory(prefix="axai-review-") as directory:
        environment = Path(directory) / "venv"
        try:
            subprocess.run([sys.executable, "-m", "venv", str(environment)], capture_output=True, text=True, timeout=60, check=True)
            python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            if requirements.exists():
                install = subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], capture_output=True, text=True, timeout=timeout, check=False)
                if install.returncode: return [Finding("execution", "high", "Isolated dependency installation failed", install.stderr[-1200:], recommendation="Pin and validate requirements.")]
            completed = subprocess.run([str(python), str(study)], cwd=project_root, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return [Finding("execution", "high", "Isolated study execution timed out", f"The project exceeded {timeout} seconds.", recommendation="Cache or bound expensive work.")]
        except subprocess.CalledProcessError as exc:
            return [Finding("execution", "high", "Isolated environment creation failed", exc.stderr[-1200:] if exc.stderr else str(exc), recommendation="Check supported Python versions.")]
    if completed.returncode: return [Finding("execution", "high", "Study execution failed", completed.stderr[-1200:], recommendation="Fix the exception and add a smoke test.")]
    return []