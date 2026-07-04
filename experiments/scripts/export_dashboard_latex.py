#!/usr/bin/env python3
"""
Export dashboard-style summary to a LaTeX table.

Defaults target the Phase1 results directory so you can run:
  python experiments/scripts/export_dashboard_latex.py \
    --placeholders-dir experiments/placeholders_phase1 \
    --output Latex/phase1_dashboard_table.tex

Columns:
  Model | Dataset | Samples | Success Rate | Clean ASR (avg) | Created
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

# Local import
import sys
from pathlib import Path as _Path

# Ensure project root on sys.path so `experiments` is importable
_project_root = _Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.core.placeholder_dashboard import PlaceholderDashboard


def latex_escape(s: str) -> str:
    """Escape LaTeX-sensitive characters in text fields."""
    if s is None:
        return ""
    # Minimal escaping for table content
    return (
        str(s)
        .replace("\\", r"\textbackslash{}").replace("_", r"\_")
        .replace("%", r"\%").replace("&", r"\&")
        .replace("#", r"\#").replace("$", r"\$")
        .replace("{", r"\{").replace("}", r"\}")
    )


def format_percent(x: float) -> str:
    return f"{x*100:.1f}\%"


def collect_rows(dashboard: PlaceholderDashboard,
                 status: str | None,
                 sort_by: str,
                 limit: int | None) -> List[Dict[str, Any]]:
    dashboard.load_placeholders(status)

    # Sort similar to dashboard display
    def sort_key(ph: Dict[str, Any]):
        config = ph.get("config", {})
        total = ph.get("total_samples") or 0
        success = ph.get("successful_samples") or 0
        success_rate = success / max(total, 1)

        # compute avg clean ASR
        judger_asrs, avg_asr = dashboard._calculate_individual_judger_asrs(ph)

        if sort_by == "asr" or sort_by == "clean_asr":
            return -avg_asr
        elif sort_by == "samples":
            return -total
        elif sort_by == "success_rate":
            return -success_rate
        elif sort_by == "model":
            return (config.get("model", "zzz"))
        elif sort_by == "dataset":
            return (config.get("dataset", "zzz"))
        elif sort_by == "attack":
            return (config.get("attack", "zzz"))
        elif sort_by == "defense":
            return (config.get("defense", "zzz"))
        elif sort_by == "status":
            return ph.get("status", "zzz")
        else:  # time / created_time default
            return -(ph.get("created_time") or 0)

    placeholders = sorted(dashboard.placeholders, key=sort_key)
    if limit and len(placeholders) > limit:
        placeholders = placeholders[:limit]

    rows = []
    for ph in placeholders:
        config = ph.get("config", {})
        total = ph.get("total_samples") or 0
        success = ph.get("successful_samples") or 0
        success_rate = success / max(total, 1)
        judger_asrs, avg_asr = dashboard._calculate_individual_judger_asrs(ph)
        created = ph.get("created_time") or 0
        created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M") if created else "N/A"

        rows.append({
            "model": config.get("model", "N/A"),
            "dataset": config.get("dataset", "N/A"),
            "attack": config.get("attack", "N/A"),
            "defense": config.get("defense", "N/A"),
            "samples": total,
            "success_rate": success_rate,
            "clean_asr_avg": avg_asr,
            "created": created_str,
        })

    return rows


def to_latex_table(rows: List[Dict[str, Any]], caption: str, label: str) -> str:
    # Use longtable for potentially many rows
    header = (
        "\\begin{longtable}{l l r r r l}\n"
        "\\caption{" + latex_escape(caption) + "}\\label{" + latex_escape(label) + "}\\\n"\
        "\\toprule\\\n"
        "Model & Dataset & Samples & Success Rate & Clean ASR (avg) & Created \\\\ \n"
        "\\midrule\\\n"
        "\\endfirsthead\n"
        "\\toprule\\\n"
        "Model & Dataset & Samples & Success Rate & Clean ASR (avg) & Created \\\\ \n"
        "\\midrule\\\n"
        "\\endhead\n"
    )

    body_lines = []
    for r in rows:
        body_lines.append(
            f"{latex_escape(r['model'])} & "
            f"{latex_escape(r['dataset'])} & "
            f"{r['samples']} & "
            f"{format_percent(r['success_rate'])} & "
            f"{format_percent(r['clean_asr_avg'])} & "
            f"{latex_escape(r['created'])} \\\\"  # end row
        )

    footer = (
        "\\bottomrule\\\n"
        "\\end{longtable}\n"
    )

    return "\n".join([header] + body_lines + [footer])


def main():
    parser = argparse.ArgumentParser(description="Export dashboard summary to LaTeX table")
    parser.add_argument("--placeholders-dir", default="experiments/placeholders", help="Placeholders directory")
    parser.add_argument("--status", choices=["pending", "running", "in_progress", "success", "failed"], default="success",
                        help="Filter by placeholder status (default: success)")
    parser.add_argument("--sort-by", choices=["asr", "clean_asr", "samples", "time", "model", "dataset", "attack", "defense", "created_time", "status", "success_rate"],
                        default="model", help="Sorting key (default: model)")
    parser.add_argument("--limit", type=int, help="Limit number of rows")
    parser.add_argument("--output", default="Latex/phase1_dashboard_table.tex", help="Output .tex file path")
    parser.add_argument("--caption", default="Phase1 dashboard summary", help="LaTeX table caption")
    parser.add_argument("--label", default="tab:phase1-dashboard", help="LaTeX table label")

    args = parser.parse_args()

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dashboard = PlaceholderDashboard(placeholders_dir=args.placeholders_dir)
    rows = collect_rows(dashboard, status=args.status, sort_by=args.sort_by, limit=args.limit)

    tex = to_latex_table(rows, caption=args.caption, label=args.label)
    out_path.write_text(tex, encoding="utf-8")

    print(f"✅ LaTeX table written to: {out_path}")


if __name__ == "__main__":
    main()
