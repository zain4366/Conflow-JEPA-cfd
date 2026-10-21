#!/usr/bin/env python3
"""Decide whether a segment job has work to do (system python only).
usage: state_check.py WORK_DIR SEGMENT
Segment 1 always runs. A later segment runs only if the previous segment saved a work directory
(it does so only when it paused at the job time budget) whose state is not final."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "state.json"
seg = int(sys.argv[2])
status = json.loads(p.read_text()).get("status") if p.exists() else None
if seg == 1:
    active = True
else:
    active = status == "paused"
print(f"active={'true' if active else 'false'}")
print(f"prev_status={status}")
