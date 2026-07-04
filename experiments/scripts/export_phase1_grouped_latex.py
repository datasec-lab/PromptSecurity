#!/usr/bin/env python3
"""
Export Phase1 results grouped by model into a LaTeX table.

Structure:
  - First column: Model (multirow spanning dataset rows)
  - Second column: Dataset
  - Next 7 columns: Judger ASRs in fixed Phase1 order
      [HB, GPT-C, GPT-H, GPT-HS, GPT-O, GPT-T, REJ]
  - Last column: Average ASR (mean of the seven judgers for that row)

Rows are grouped per model for up to three datasets (harmbench, jbb, airbench),
and models are sorted by the average of per-dataset averages (descending).

Usage example:
  python experiments/scripts/export_phase1_grouped_latex.py \
    --placeholders-dir experiments/placeholders_phase1 \
    --output Latex/phase1_dashboard_grouped.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple
from datetime import datetime
import sys

# Ensure project root on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.core.placeholder_dashboard import PlaceholderDashboard


PHASE1_JUDGERS = [
    "harmbench_judger",                 # HB
    "gpt_judger_contextual_harmbench",  # GPT-C
    "gpt_judger_harmful_binary",        # GPT-H
    "gpt_judger_harmbench_style",       # GPT-HS
    "gpt_judger_openai_policy",        # GPT-O
    "gpt_judger_tap_style",            # GPT-T
    "rejection_prefix_judger",          # REJ
]

JUDGER_SHORT = ["HB", "GPT-C", "GPT-H", "GPT-HS", "GPT-O", "GPT-T", "REJ"]


def latex_escape(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}").replace("_", r"\_")
        .replace("%", r"\%").replace("&", r"\&")
        .replace("#", r"\#").replace("$", r"\$")
        .replace("{", r"\{").replace("}", r"\}")
    )


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}\%"


def load_phase1_placeholders(placeholders_dir: str, status: str = "success") -> Tuple[PlaceholderDashboard, List[Dict[str, Any]]]:
    dash = PlaceholderDashboard(placeholders_dir=placeholders_dir)
    dash.load_placeholders(status)
    return dash, dash.placeholders


def group_by_model(placeholders: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return mapping: model -> dataset -> placeholder dict"""
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for ph in placeholders:
        cfg = ph.get("config", {})
        model = cfg.get("model", "N/A")
        dataset = cfg.get("dataset", "N/A")
        grouped.setdefault(model, {})[dataset] = ph
    return grouped


def extract_judger_asrs(dash: PlaceholderDashboard, ph: Dict[str, Any]) -> Tuple[List[float], float]:
    # Uses PlaceholderDashboard private method to keep order consistent with Phase1
    judger_asrs, avg_asr = dash._calculate_individual_judger_asrs(ph)  # type: ignore[attr-defined]
    # Ensure length 7
    if len(judger_asrs) < 7:
        judger_asrs = (judger_asrs + [0.0]*7)[:7]
    return judger_asrs, avg_asr


def build_rows(dash: PlaceholderDashboard, grouped: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    dataset_order = ["harmbench", "jbb", "airbench"]
    rows: List[Dict[str, Any]] = []

    # Prepare model-level average for sorting
    for model, per_ds in grouped.items():
        ds_rows = []
        for ds in dataset_order:
            ph = per_ds.get(ds)
            if not ph:
                continue
            judger_asrs, avg_asr = extract_judger_asrs(dash, ph)
            ds_rows.append({
                "model": model,
                "dataset": ds,
                "judgers": judger_asrs,
                "avg": avg_asr,
            })
        if not ds_rows:
            continue
        model_avg = mean(r["avg"] for r in ds_rows)
        for i, r in enumerate(ds_rows):
            r["model_avg"] = model_avg
            r["is_first"] = (i == 0)
            r["row_span"] = len(ds_rows)
            rows.append(r)

    # Sort by model_avg desc, then model name, then dataset order
    order_index = {"harmbench": 0, "jbb": 1, "airbench": 2}
    rows.sort(key=lambda r: (-r["model_avg"], r["model"], order_index.get(r["dataset"], 99)))
    return rows


def to_latex(rows: List[Dict[str, Any]], caption: str, label: str, fmt: str = "longtable") -> str:
    # Column spec: Model | Dataset | 7 judgers | Avg
    colspec = "l l r r r r r r r r"
    header_cols = "Model & Dataset & " + " & ".join(JUDGER_SHORT) + " & Avg \\ \n"

    if fmt == "tabular":
        header = (
            f"\\begin{{tabular}}{{{colspec}}}\n"
            "\\toprule\\\n"
            + header_cols +
            "\\midrule\\\n"
        )
        footer = (
            "\\bottomrule\\\n"
            "\\end{tabular}\n"
        )
    else:
        header = (
            f"\\begin{{longtable}}{{{colspec}}}\n"
            f"\\caption{{{latex_escape(caption)}}}\\label{{{latex_escape(label)}}}\\\\\n"
            "\\toprule\\\n"
            + header_cols +
            "\\midrule\\\n"
            "\\endfirsthead\n"
            "\\toprule\\\n"
            + header_cols +
            "\\midrule\\\n"
            "\\endhead\n"
        )
        footer = (
            "\\bottomrule\\\n"
            "\\end{longtable}\n"
        )

    lines: List[str] = []
    for r in rows:
        model_cell = ""
        if r["is_first"]:
            model_cell = f"\\multirow{{{r['row_span']}}}{{*}}{{{latex_escape(r['model'])}}} & "
        else:
            model_cell = " & "  # empty first column continuation

        judger_cells = " & ".join(fmt_pct(v) for v in r["judgers"])
        lines.append(
            model_cell + f"{latex_escape(r['dataset'])} & {judger_cells} & {fmt_pct(r['avg'])} \\\\"  # noqa: E501
        )

    return "\n".join([header] + lines + [footer])


def main():
    ap = argparse.ArgumentParser(description="Export Phase1 grouped-by-model LaTeX table")
    ap.add_argument("--placeholders-dir", default="experiments/placeholders_phase1")
    ap.add_argument("--status", choices=["pending", "running", "in_progress", "success", "failed"], default="success")
    ap.add_argument("--output", default="Latex/phase1_dashboard_grouped.tex")
    ap.add_argument("--caption", default="Phase1 Results by Model (judger breakdown)")
    ap.add_argument("--label", default="tab:phase1-grouped")
    ap.add_argument("--format", choices=["longtable", "tabular"], default="tabular")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dash, placeholders = load_phase1_placeholders(args.placeholders_dir, status=args.status)
    grouped = group_by_model(placeholders)
    rows = build_rows(dash, grouped)
    tex = to_latex(rows, caption=args.caption, label=args.label, fmt=args.format)
    out_path.write_text(tex, encoding="utf-8")
    print(f"✅ LaTeX table written to: {out_path}")


if __name__ == "__main__":
    main()
