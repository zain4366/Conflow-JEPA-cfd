#!/usr/bin/env python3
"""Merge downloaded artifacts into a checkout of the `results` branch.
usage: collect.py ARTIFACT_DIR RESULTS_DIR RUN_ID
  ARTIFACT_DIR/out-<case>/...   -> RESULTS_DIR/runs/<run_id>/<case>/ and RESULTS_DIR/latest/<case>/
  ARTIFACT_DIR/reference/...    -> RESULTS_DIR/reference/
A case whose out-artifact is missing (e.g. a job that crashed before writing) is listed as missing in
runs/<run_id>/index.json, so nothing disappears silently."""
import json
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    art, res, run_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    cases = json.loads(sys.argv[4]) if len(sys.argv) > 4 else []
    idx = dict(run_id=run_id, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cases={})
    for c in cases:
        src = art / f"out-{c}"
        if not (src / "status.json").exists():
            idx["cases"][c] = dict(status="missing", note="no out artifact; see the Actions log of this run")
            continue
        st = json.loads((src / "status.json").read_text())
        idx["cases"][c] = dict(status=st["status"], stop_reason=st.get("stop_reason"), n_iter=st.get("n_iter"))
        for dst in (res / "runs" / run_id / c, res / "latest" / c):
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    if (art / "reference" / "manifest.json").exists():
        dst = res / "reference"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(art / "reference", dst)
        idx["reference"] = "updated"
    (res / "runs" / run_id).mkdir(parents=True, exist_ok=True)
    (res / "runs" / run_id / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
    print(json.dumps(idx, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
