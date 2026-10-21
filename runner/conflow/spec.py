"""Load configs/spec.yaml (machine-readable copy of PLAN section 4)."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "spec.yaml"


def load_spec(path=SPEC_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
