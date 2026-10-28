#!/usr/bin/env python3
"""Run (or continue) one SU2 case inside a GitHub Actions job.

usage: run_case.py --case NAME --work DIR --out DIR --job-start EPOCH [--last-job]

work/ holds the mesh, restart file and per-segment histories and is carried to the next chained
job as an artifact; out/ holds the small result files that are committed to the `results` branch.
Every outcome is written to out/status.json, including failures (with the error) and runs that hit
the iteration cap or the chained-job limit (status 'unconverged').
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))

from conflow import cfd  # noqa: E402
from conflow.records import sha256_file, utc_now  # noqa: E402
from conflow.spec import load_spec  # noqa: E402

FINAL = ("converged", "unconverged", "diverged", "failed", "meshed")


def machine() -> dict:
    cpu = ""
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                cpu = ln.split(":", 1)[1].strip()
                break
        mem = [ln for ln in Path("/proc/meminfo").read_text().splitlines() if ln.startswith("MemTotal")][0]
    except OSError:
        mem = ""
    return dict(nproc=os.cpu_count(), cpu_model=cpu, mem_total=mem.split(":", 1)[-1].strip(),
                platform=platform.platform(), python=platform.python_version(),
                runner_os=os.environ.get("RUNNER_OS"), runner_name=os.environ.get("RUNNER_NAME"))


def su2_version() -> str:
    try:
        p = subprocess.run(["SU2_CFD", "--help"], capture_output=True, text=True, timeout=60)
        return (p.stdout.splitlines() or ["?"])[0].strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"


def tail(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def write_out(case, work: Path, out: Path, st: dict, prep: dict, extra: dict):
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(case=case["name"], status=st.get("status"), stop_reason=st.get("stop_reason"),
               convergence=st.get("convergence"), n_iter=st.get("n_iter"), iters_total=st.get("iters_total"),
               wall_s_total=st.get("wall_s_total"), segments=st.get("segments", []),
               timing=cfd.timing_summary(work) if (work / "state.json").exists() else None,
               prepared=prep, su2_version=su2_version(), machine=machine(), case_definition=case,
               runner_commit=os.environ.get("GITHUB_SHA"), run_id=os.environ.get("GITHUB_RUN_ID"),
               jobs=st.get("jobs", []), memory_solve_jobs=st.get("memory_solve_jobs", []), created_utc=utc_now())
    rec.update(extra)
    if any(work.glob("history_seg*.csv")):
        cfd.merged_history_csv(work, out / "history.csv")
    for name in ("surface_flow.csv", "surface_origingeom.npy", "cp_adflow_cells.npy", "surface_builder.npy"):
        if (work / name).exists():
            shutil.copy2(work / name, out / name)
    if prep and (work / prep["mesh_file"]).exists() and not (out / "wall_faces.npy").exists():
        import numpy as np
        mk = cfd.read_su2_markers(work / prep["mesh_file"], [case["markers"]["wall"]])
        d = mk[case["markers"]["wall"]]
        faces = d[cfd.VTK_LINE] if cfd.VTK_LINE in d else cfd.wall_triangles(mk, case["markers"]["wall"])
        np.save(out / "wall_faces.npy", faces.astype(np.int64))
    segs = sorted(work.glob("seg*.cfg"))
    if segs:
        shutil.copy2(segs[-1], out / "config_last_segment.cfg")
        (out / "log_first_segment_head.txt").write_text("\n".join(
            (work / segs[0].with_suffix(".log").name).read_text(errors="replace").splitlines()[:400]))
        (out / "log_last_segment_tail.txt").write_text(tail(work / segs[-1].with_suffix(".log").name, 300))
    (out / "status.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--job-start", type=float, required=True)
    ap.add_argument("--last-job", action="store_true")
    a = ap.parse_args()
    spec = load_spec()
    rs = spec["cfd"]["runner"]
    case = json.loads((ROOT / "cases" / a.case / "case.json").read_text())
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    deadline = a.job_start + 60.0 * rs["job_budget_minutes"]
    sp = work / "state.json"
    st = json.loads(sp.read_text()) if sp.exists() else dict(segments=[], status="pending", jobs=[])
    st.setdefault("jobs", []).append(dict(job_start=a.job_start, run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"),
                                          job=os.environ.get("GITHUB_JOB")))
    sp.write_text(json.dumps(st, indent=1))
    pp = work / "prepared.json"
    prep, extra = None, {}
    try:
        mesh_only = cfd.is_mesh_only(case)
        rule = cfd.case_convergence_rule(case)          # RT2 / A7 rule if the case has one, else PLAN section 4
        if pp.exists():
            prep = json.loads(pp.read_text())
        else:
            with cfd.MemorySampler() as mem:
                prep = cfd.prepare_case(case, work, log=print)
            prep["memory_prepare"] = mem.result()
            pp.write_text(json.dumps(prep, indent=1, default=str))
        if mesh_only:
            st.update(status="meshed", stop_reason="mesh_only case (not solved)")
        else:
            cfg = cfd.case_config(case, prep)
            kind = "euler" if case["solver"] == "euler" else "rans"
            n = os.cpu_count() or 1
            cmd = ["mpirun", "--oversubscribe", "-np", str(n), "SU2_CFD"] if n > 1 else ["SU2_CFD"]
            with cfd.MemorySampler() as mem:
                st = cfd.run_segments(work, cfg, cmd, rs["segment_iters"][kind], rs["max_iters"][kind], deadline,
                                      log=print, rule=rule)
            st.setdefault("memory_solve_jobs", []).append(mem.result())
        if st["status"] == "paused" and a.last_job:
            st["status"] = "unconverged"
            st["stop_reason"] = f"chained-job limit ({rs['max_segments_jobs']} jobs)"
        if not mesh_only and any(work.glob("history_seg*.csv")):
            hh = cfd.concat_histories([cfd.read_history(p) for p in sorted(work.glob("history_seg*.csv"))])
            extra["final_aoa"] = cfd.final_aoa(hh)
        if case["kind"] == "superwing" and (work / "surface_flow.csv").exists():
            import numpy as np
            s = cfd.read_surface_csv(work / "surface_flow.csv")
            sw = prep["superwing"]["index_row"]
            c = cfd.compare_superwing(s, np.load(work / "surface_origingeom.npy"), sw["aoa"], sw["s_ref_half"],
                                      cp_ref_cells=np.load(work / "cp_adflow_cells.npy"))
            h = cfd.concat_histories([cfd.read_history(p) for p in sorted(work.glob("history_seg*.csv"))])
            mk = cfd.read_su2_markers(work / prep["mesh_file"], [case["markers"]["wall"]])
            tri = cfd.wall_triangles(mk, case["markers"]["wall"])
            lut = np.full(int(s["PointID"].max()) + 1, -1)
            lut[s["PointID"]] = np.arange(len(s["PointID"]))
            pts_sw = cfd.su2_to_sw(np.stack([s["x"], s["y"], s["z"]], 1))
            wi = cfd.integrate_wall(pts_sw, lut[tri], s["Pressure_Coefficient"], sw["aoa"], sw["s_ref_half"],
                                    nodes_sw=np.load(work / "surface_origingeom.npy"))
            extra["superwing_comparison"] = dict(
                su2_pressure_on_su2_wall=wi,
                su2_pressure_mapped_to_data_grid=dict(cl=c["cl"], cd=c["cd"], cm=c["cm"]),
                su2_history_last=dict(cl=float(h["CL"][-1]), cd=float(h["CD"][-1])),
                adflow=dict(cl_solver=sw["cl_solver"], cd_solver=sw["cd_solver"], cm_solver=sw["cm_solver"],
                            cl_surface=sw["cl_surface"], cd_surface=sw["cd_surface"], cm_surface=sw["cm_surface"]),
                cp_rmse_vs_adflow=c["cp_rmse_vs_ref"],
                note=f"SU2 {case['solver']} vs ADflow RANS-SA; differences are reported, not corrected")
    except Exception as e:  # every failure is recorded in status.json and reported
        st["status"] = "failed"
        st["stop_reason"] = f"{type(e).__name__}: {e}"
        extra["traceback"] = traceback.format_exc()
        if prep is None and (work / "selection.json").exists():
            extra["selection_before_failure"] = json.loads((work / "selection.json").read_text())
        print(extra["traceback"])
    sp.write_text(json.dumps(st, indent=1, default=str))
    write_out(case, work, out, st, prep, extra)
    gho = os.environ.get("GITHUB_OUTPUT")
    if gho:
        with open(gho, "a") as f:
            f.write(f"paused={'true' if st['status'] == 'paused' else 'false'}\n")
            f.write(f"status={st['status']}\n")
    print(json.dumps(dict(case=a.case, status=st["status"], reason=st.get("stop_reason"), n_iter=st.get("n_iter"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
