#!/usr/bin/env python3
"""
Minimal smoke test for newly added models.

No external API calls are made:
- models are validated by checking config files only
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_smoke() -> int:
    models_to_check = [
        "gemini-3.0-pro",
        "gemini-3.0-flash",
        "gpt-5",
        "gpt-5.2",
    ]

    results = {"models": {}}

    for model in models_to_check:
        cfg_path = Path("models/usage_examples/configs/api") / f"{model}.json"
        try:
            if not cfg_path.exists():
                raise FileNotFoundError(str(cfg_path))
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            required = ("model_type", "model_name", "parameters")
            missing = [k for k in required if k not in cfg]
            if missing:
                raise ValueError(f"missing keys: {missing}")
            results["models"][model] = {
                "ok": True,
                "detail": f"config_ok ({cfg['model_type']} -> {cfg['model_name']})",
            }
        except Exception as exc:
            results["models"][model] = {
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }

    print("=== Smoke Test Report ===")
    for section in ("models",):
        print(f"[{section}]")
        for name, info in results[section].items():
            tag = "PASS" if info["ok"] else "FAIL"
            print(f"- {name}: {tag} | {info['detail']}")

    all_ok = all(item["ok"] for sec in results.values() for item in sec.values())
    print(f"ALL_OK={all_ok}")
    return 0 if all_ok else 1


def main() -> int:
    return run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
