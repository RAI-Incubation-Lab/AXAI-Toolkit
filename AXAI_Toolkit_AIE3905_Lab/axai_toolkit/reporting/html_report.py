# -*- coding: utf-8 -*-
"""单文件 HTML 交互式诊断报告生成器。"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

import numpy as np

RADAR_TEMPLATE = """<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const scores = __SCORES_JSON__;
const ctx = document.getElementById('radarChart').getContext('2d');
new Chart(ctx, {
  type: 'radar',
  data: {
    labels: Object.keys(scores),
    datasets: [{
      label: 'AXAI Scores',
      data: Object.values(scores),
      backgroundColor: 'rgba(54, 162, 235, 0.2)',
      borderColor: 'rgba(54, 162, 235, 1)',
      pointBackgroundColor: 'rgba(54, 162, 235, 1)'
    }]
  },
  options: {
    scales: {
      r: {
        beginAtZero: true,
        max: 100
      }
    }
  }
});
</script>
"""


def generate_html_report(
    title: str,
    summary: str,
    findings: list[dict],
    scores: Optional[dict[str, float]] = None,
) -> str:
    """生成单文件 HTML 诊断报告。"""
    if scores is None:
        scores = {}
    scores_json = json.dumps(scores, ensure_ascii=False)

    finding_items = "".join(
        f"<li><b>{html.escape(str(item.get('type', 'finding')))}</b>: "
        f"{html.escape(str(item.get('detail', item)))}</li>"
        for item in findings
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f7f9fc; color: #222; }}
.card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
h1 {{ color: #1f4e79; }}
.chart-container {{ max-width: 500px; margin: 0 auto; }}
</style>
</head>
<body>
<div class="card">
<h1>{html.escape(title)}</h1>
<p>{html.escape(summary)}</p>
</div>
<div class="card">
<h2>Findings</h2>
<ul>{finding_items or '<li>No findings.</li>'}</ul>
</div>
<div class="card">
<h2>AXAI Radar</h2>
<div class="chart-container"><canvas id="radarChart" width="400" height="400"></canvas></div>
</div>
{RADAR_TEMPLATE.replace('__SCORES_JSON__', scores_json)}
</body>
</html>
"""


def export_html_report(
    report_html: str,
    output_path: str,
) -> Path:
    """将 HTML 报告写入文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_html, encoding="utf-8")
    return path
