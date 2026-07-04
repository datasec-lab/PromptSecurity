#!/usr/bin/env python3
"""
Export dashboard-only-ASR placeholder experiments to an Excel (.xlsx) file.

This script mirrors `python -m experiments --dashboard --dashboard-only-asr`:
- load placeholders from a directory
- keep only placeholders with `status == "success"` and having judger data
- compute individual judger ASRs and avg ASR using the same dashboard logic

Output columns (as requested):
  模型, 攻击, 防御, 成功率, HB ASR, GPT-H ASR, REJ ASR, 平均ASR

By default writes to:
  experiments/exports/dashboard_only_asr.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from xml.sax.saxutils import escape as _xml_escape
import zipfile
import importlib.util


_project_root = Path(__file__).resolve().parents[2]


def _load_placeholder_dashboard_class():
    """
    Load PlaceholderDashboard without importing the `experiments` package.

    `experiments/__init__.py` imports model/defense loaders which may pull optional
    deps (e.g. google.generativeai). For export we only need the standalone
    dashboard module, so load it directly by file path.
    """
    mod_path = _project_root / "experiments" / "core" / "placeholder_dashboard.py"
    spec = importlib.util.spec_from_file_location("_placeholder_dashboard", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlaceholderDashboard


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.1f}%"


def _col_to_a1(col_1_based: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA ..."""
    if col_1_based <= 0:
        raise ValueError(f"invalid col: {col_1_based}")
    s = ""
    n = col_1_based
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(ord("A") + r) + s
    return s


def _cell_ref(row_1_based: int, col_1_based: int) -> str:
    return f"{_col_to_a1(col_1_based)}{row_1_based}"


def _needs_preserve_space(s: str) -> bool:
    return s != s.strip() or "  " in s


def _build_shared_strings(strings_in_order: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    # Keep insertion order of first occurrence.
    table: List[str] = []
    idx: Dict[str, int] = {}
    for s in strings_in_order:
        if s not in idx:
            idx[s] = len(table)
            table.append(s)
    return table, idx


def _xlsx_xml_shared_strings(shared: Sequence[str]) -> str:
    # https://learn.microsoft.com/en-us/office/open-xml/working-with-the-shared-string-table
    sis = []
    for s in shared:
        s_esc = _xml_escape(s)
        if _needs_preserve_space(s):
            sis.append(f'<si><t xml:space="preserve">{s_esc}</t></si>')
        else:
            sis.append(f"<si><t>{s_esc}</t></si>")
    inner = "".join(sis)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared)}" uniqueCount="{len(shared)}">'
        f"{inner}</sst>"
    )


def _xlsx_xml_styles() -> str:
    # Minimal styles so Excel doesn't complain. No custom number formats.
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1">
    <font>
      <sz val="11"/>
      <color theme="1"/>
      <name val="Calibri"/>
      <family val="2"/>
      <scheme val="minor"/>
    </font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>
"""


def _xlsx_xml_worksheet(sheet_name: str, grid: Sequence[Sequence[Any]], shared_index: Dict[str, int]) -> str:
    # Only strings and numbers are expected; everything else is converted to string.
    rows_xml: List[str] = []
    max_row = len(grid)
    max_col = max((len(r) for r in grid), default=0)

    for r_i, row in enumerate(grid, start=1):
        cells: List[str] = []
        for c_i, v in enumerate(row, start=1):
            ref = _cell_ref(r_i, c_i)
            if v is None:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cells.append(f'<c r="{ref}"><v>{v}</v></c>')
                continue

            s = str(v)
            s_idx = shared_index[s]
            cells.append(f'<c r="{ref}" t="s"><v>{s_idx}</v></c>')
        rows_xml.append(f'<row r="{r_i}">{"".join(cells)}</row>')

    dim_ref = f"A1:{_cell_ref(max_row if max_row else 1, max_col if max_col else 1)}"
    sheet_data = "".join(rows_xml)
    # sheet_name isn't embedded here; it's in workbook.xml. Keep this minimal.
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dim_ref}"/>'
        f"<sheetData>{sheet_data}</sheetData>"
        "</worksheet>"
    )


def _xlsx_xml_workbook(sheet_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{_xml_escape(sheet_name)}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""


def _xlsx_xml_root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
                Target="xl/workbook.xml"/>
</Relationships>
"""


def _xlsx_xml_workbook_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
                Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
                Target="styles.xml"/>
  <Relationship Id="rId3"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings"
                Target="sharedStrings.xml"/>
</Relationships>
"""


def _xlsx_xml_content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>
"""


def write_simple_xlsx(path: Path, sheet_name: str, grid: Sequence[Sequence[Any]]) -> None:
    # Collect all strings for sharedStrings (including headers).
    strings: List[str] = []
    for row in grid:
        for v in row:
            if v is None:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                continue
            strings.append(str(v))

    shared, shared_index = _build_shared_strings(strings)
    worksheet_xml = _xlsx_xml_worksheet(sheet_name, grid, shared_index)

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _xlsx_xml_content_types())
        z.writestr("_rels/.rels", _xlsx_xml_root_rels())
        z.writestr("xl/workbook.xml", _xlsx_xml_workbook(sheet_name))
        z.writestr("xl/_rels/workbook.xml.rels", _xlsx_xml_workbook_rels())
        z.writestr("xl/styles.xml", _xlsx_xml_styles())
        z.writestr("xl/sharedStrings.xml", _xlsx_xml_shared_strings(shared))
        z.writestr("xl/worksheets/sheet1.xml", worksheet_xml)


def _filter_only_asr(dashboard: PlaceholderDashboard) -> List[Dict[str, Any]]:
    # Mirrors PlaceholderDashboard.display_table(..., only_asr=True)
    return [
        p for p in dashboard.placeholders
        if p.get("status") == "success" and dashboard._has_judger_data(p)
    ]


def export_rows(placeholders_dir: str,
                model: Optional[str],
                attack: Optional[str],
                defense: Optional[str],
                dataset: Optional[str],
                sort_by: str) -> List[List[Any]]:
    PlaceholderDashboard = _load_placeholder_dashboard_class()
    dashboard = PlaceholderDashboard(placeholders_dir=placeholders_dir)
    dashboard.load_placeholders()

    placeholders = _filter_only_asr(dashboard)

    def match(p: Dict[str, Any]) -> bool:
        cfg = p.get("config", {})
        if model and cfg.get("model") != model:
            return False
        if attack and cfg.get("attack") != attack:
            return False
        if defense and cfg.get("defense") != defense:
            return False
        if dataset and cfg.get("dataset") != dataset:
            return False
        return True

    placeholders = [p for p in placeholders if match(p)]

    def sort_key(p: Dict[str, Any]):
        cfg = p.get("config", {})
        total = p.get("total_samples") or 0
        success = p.get("successful_samples") or 0
        success_rate = success / max(total, 1)
        _, avg_asr = dashboard._calculate_individual_judger_asrs(p)

        if sort_by in ("asr", "clean_asr"):
            return -avg_asr
        if sort_by == "success_rate":
            return -success_rate
        if sort_by == "samples":
            return -total
        if sort_by == "model":
            return cfg.get("model", "")
        if sort_by == "attack":
            return cfg.get("attack", "")
        if sort_by == "defense":
            return cfg.get("defense", "")
        if sort_by == "dataset":
            return cfg.get("dataset", "")
        if sort_by in ("time", "created_time"):
            return -(p.get("created_time") or 0)
        return cfg.get("model", "")

    placeholders.sort(key=sort_key)

    header = ["模型", "攻击", "防御", "成功率", "HB ASR", "GPT-H ASR", "REJ ASR", "平均ASR"]
    rows: List[List[Any]] = [header]

    for p in placeholders:
        cfg = p.get("config", {})
        total = p.get("total_samples") or 0
        success = p.get("successful_samples") or 0
        success_rate = success / max(total, 1)

        judger_asrs, avg_asr = dashboard._calculate_individual_judger_asrs(p)
        # Phase1 fixed order: [HB, GPT-C, GPT-H, GPT-HS, GPT-O, GPT-T, REJ]
        hb_asr = judger_asrs[0] if len(judger_asrs) > 0 else None
        gpth_asr = judger_asrs[2] if len(judger_asrs) > 2 else None
        rej_asr = judger_asrs[6] if len(judger_asrs) > 6 else None

        rows.append([
            cfg.get("model", "N/A"),
            cfg.get("attack", "N/A"),
            cfg.get("defense", "N/A"),
            _pct(success_rate),
            _pct(hb_asr),
            _pct(gpth_asr),
            _pct(rej_asr),
            _pct(avg_asr),
        ])

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Export dashboard-only-ASR placeholder experiments to .xlsx")
    ap.add_argument("--placeholders-dir", default="experiments/placeholders",
                    help="Placeholders directory (default: experiments/placeholders)")
    ap.add_argument("--output", default="experiments/exports/dashboard_only_asr.xlsx",
                    help="Output .xlsx path (default: experiments/exports/dashboard_only_asr.xlsx)")

    ap.add_argument("--model", help="Filter by exact model name")
    ap.add_argument("--attack", help="Filter by exact attack name")
    ap.add_argument("--defense", help="Filter by exact defense name")
    ap.add_argument("--dataset", help="Filter by exact dataset name")
    ap.add_argument("--sort-by",
                    choices=["asr", "clean_asr", "samples", "time", "created_time", "model", "dataset", "attack", "defense", "success_rate"],
                    default="model",
                    help="Sort key (default: model)")
    ap.add_argument("--sheet-name", default="ASR", help="Excel sheet name (default: ASR)")

    args = ap.parse_args()

    grid = export_rows(
        placeholders_dir=args.placeholders_dir,
        model=args.model,
        attack=args.attack,
        defense=args.defense,
        dataset=args.dataset,
        sort_by=args.sort_by,
    )

    out_path = Path(args.output)
    write_simple_xlsx(out_path, sheet_name=args.sheet_name, grid=grid)
    print(f"✅ Wrote {max(len(grid) - 1, 0)} rows to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
