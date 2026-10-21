#!/usr/bin/env python3
"""Download the experimental reference data from the original NASA NPARC archive pages and record
URL, SHA-256, size and retrieval time in manifest.json. No value is typed in anywhere.

  ONERA M6, NPARC test 2308 (M 0.8395, AoA 3.06 deg): Cp at 7 span stations, cp{k}{u,l}.ex
    https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing.html
  RAE 2822 case 6 (AGARD AR-138), experimental Cp (GENPLOT file) of NPARC study 1
    https://www.grc.nasa.gov/WWW/wind/valid/raetaf/raetaf01/raetaf01.html
usage: fetch_reference.py OUT_DIR
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))
from conflow import cfd  # noqa: E402
from conflow.records import utc_now  # noqa: E402

M6 = "https://www.grc.nasa.gov/WWW/wind/valid/m6wing/"
RAE = "https://www.grc.nasa.gov/WWW/wind/valid/raetaf/"
FILES = {f"cp{k}{s}.ex": M6 + f"cp{k}{s}.ex" for k in range(1, 8) for s in "ul"}
FILES.update({
    "m6wing.html": M6 + "m6wing.html",
    "ONERAb114.tec": M6 + "ONERAb114.tec",
    "cp.exp.gen": RAE + "raetaf01/cp.exp.gen",
    "raetaf01.html": RAE + "raetaf01/raetaf01.html",
    "raetaf.html": RAE + "raetaf.html",
    "rae2822_geom.txt": RAE + "geom.txt",
})


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    man = dict(created_utc=utc_now(), files={}, failures={})
    for name, url in FILES.items():
        try:
            man["files"][name] = cfd.download(url, out / name)
        except Exception as e:  # recorded, never hidden
            man["failures"][name] = dict(url=url, error=str(e))
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(dict(ok=len(man["files"]), failed=list(man["failures"]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
