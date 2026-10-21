#!/usr/bin/env python3
"""Turn the workflow trigger into a job matrix (GITHUB_OUTPUT lines).
Cases come from the workflow_dispatch input (comma-separated) or, on a push, from requests/run.json:
  {"cases": ["m6_euler", ...], "reference": true, "note": "..."}
Unknown case names stop the workflow here."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    arg = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if arg:
        cases = [c.strip() for c in arg.split(",") if c.strip()]
        ref = "true" if (len(sys.argv) > 2 and sys.argv[2].strip().lower() == "true") else "false"
    else:
        req = json.loads((ROOT / "requests" / "run.json").read_text())
        cases = list(req["cases"])
        ref = "true" if req.get("reference") is True else "false"
    bad = [c for c in cases if not (ROOT / "cases" / c / "case.json").exists()]
    if bad:
        print(f"unknown cases: {bad}", file=sys.stderr)
        return 1
    print(f"cases={json.dumps(cases)}")
    print(f"reference={ref}")
    print(f"any_cases={'true' if cases else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
