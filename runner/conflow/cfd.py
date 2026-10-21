"""SU2 interface for Task 2 (CFD gates) and Task 4 (design verification).

Contents
  config writer and Euler / RANS-SA templates (numerics follow the SU2 v8 TestCases
    euler/oneram6/inv_ONERAM6.cfg, rans/oneram6/turb_ONERAM6.cfg, rans/rae2822/turb_SA_RAE2822.cfg);
  SU2 history / surface CSV / ASCII mesh readers and an ASCII mesh writer;
  the PLAN section 4 convergence rule;
  a segmented SU2 runner with restart chaining (used by the GitHub Actions runner);
  sonic Cp, shock location and wing station cuts (ONERA M6 check);
  a 2D O-grid generator for airfoils and a gmsh/OpenCASCADE 3D Euler mesh for SuperWing surfaces;
  SuperWing frame rotation, solver-surface-to-data-grid mapping and coefficient comparison;
  parsers for the NPARC experimental files; Gate G1 logic and the held-out wing selection rule.

Frames. SuperWing (SW): x streamwise, y vertical, z span. SU2 3D needs the lift axis along z, so the
mesh is written in the SU2 frame (x, y, z)_SU2 = (x, -z, y)_SW, a proper rotation (det +1); the span
runs along -y. SU2 angle of attack is in the x-z plane, i.e. the SW x-y plane, so AoA is unchanged.
Coefficients are compared in the SW frame with conflow.data.integrate_coefficients (Task 1 check).

Nothing here alters a solver result: no offsets, no clipping; unconverged runs keep that status.
"""
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .spec import load_spec

_SPEC = load_spec()
_CFD = _SPEC["cfd"]
GAMMA = 1.4

VTK_LINE, VTK_TRI, VTK_QUAD, VTK_TET, VTK_HEXA, VTK_PRISM, VTK_PYRA = 3, 5, 9, 10, 12, 13, 14
NODES_PER = {VTK_LINE: 2, VTK_TRI: 3, VTK_QUAD: 4, VTK_TET: 4, VTK_HEXA: 8, VTK_PRISM: 6, VTK_PYRA: 5}
VOLUME_TYPES_2D = (VTK_TRI, VTK_QUAD)
VOLUME_TYPES_3D = (VTK_TET, VTK_HEXA, VTK_PRISM, VTK_PYRA)


# ============================================================================= config writer

_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")


def format_value(v) -> str:
    if isinstance(v, bool):
        return "YES" if v else "NO"
    if isinstance(v, (tuple, list)):
        return "( " + ", ".join(format_value(x) for x in v) + " )"
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            raise ValueError(f"non-finite config value {v}")
        return repr(float(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def write_config(path, cfg: dict) -> Path:
    lines = ["% written by conflow.cfd.write_config"]
    for k, v in cfg.items():
        if not _KEY.match(k):
            raise ValueError(f"bad SU2 option name {k!r}")
        lines.append(f"{k}= {format_value(v)}")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def read_config(path) -> dict:
    out = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.split("%", 1)[0].strip()
        if "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _common(mesh, mach, aoa, ref_area, ref_length, moment_ref, wall, far, sym):
    c = {
        "MATH_PROBLEM": "DIRECT",
        "MACH_NUMBER": float(mach), "AOA": float(aoa), "SIDESLIP_ANGLE": 0.0,
        "REF_ORIGIN_MOMENT_X": float(moment_ref[0]), "REF_ORIGIN_MOMENT_Y": float(moment_ref[1]),
        "REF_ORIGIN_MOMENT_Z": float(moment_ref[2]),
        "REF_LENGTH": float(ref_length), "REF_AREA": float(ref_area),
        "MARKER_FAR": (far,),
        "MARKER_PLOTTING": (wall,), "MARKER_MONITORING": (wall,),
        "MESH_FILENAME": str(mesh), "MESH_FORMAT": "SU2",
        "RESTART_SOL": False,
        "SOLUTION_FILENAME": "restart_flow.dat", "RESTART_FILENAME": "restart_flow.dat",
        "TABULAR_FORMAT": "CSV", "CONV_FILENAME": "history", "SURFACE_FILENAME": "surface_flow",
        "VOLUME_FILENAME": "flow",
        "OUTPUT_FILES": ("RESTART", "SURFACE_CSV"),
        "VOLUME_OUTPUT": ("COORDINATES", "SOLUTION", "PRIMITIVE"),
        "WRT_RESTART_COMPACT": False,   # SU2 v8: otherwise the surface CSV holds only restart fields
        "HISTORY_OUTPUT": ("ITER", "RMS_RES", "AERO_COEFF", "WALL_TIME"),   # Time(sec): this iteration; Time(sec)#2: running mean
        "SCREEN_OUTPUT": ("INNER_ITER", "WALL_TIME", "RMS_DENSITY", "LIFT", "DRAG"),
        "SCREEN_WRT_FREQ_INNER": 10,
        # SU2's own stop criteria are disabled; the runner applies the PLAN rule between segments
        "CONV_FIELD": "RMS_DENSITY", "CONV_RESIDUAL_MINVAL": -16, "CONV_STARTITER": 10,
        "ITER": 500,
    }
    if sym:
        c["MARKER_SYM"] = (sym,)
    return c


def euler_config(mesh, mach, aoa, wall, far, sym, ref_area, ref_length, moment_ref, iters=500) -> dict:
    """Inviscid template (numerics of SU2 TestCases euler/oneram6/inv_ONERAM6.cfg)."""
    c = _common(mesh, mach, aoa, ref_area, ref_length, moment_ref, wall, far, sym)
    c.update({
        "SOLVER": "EULER", "MARKER_EULER": (wall,),
        "FREESTREAM_PRESSURE": 101325.0, "FREESTREAM_TEMPERATURE": 288.15,
        "REF_DIMENSIONALIZATION": "FREESTREAM_VEL_EQ_ONE",
        "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
        "CFL_NUMBER": 5.0, "CFL_ADAPT": False,
        "LINEAR_SOLVER": "FGMRES", "LINEAR_SOLVER_PREC": "LU_SGS",
        "LINEAR_SOLVER_ERROR": 1e-6, "LINEAR_SOLVER_ITER": 2,
        "MGLEVEL": 3, "MGCYCLE": "W_CYCLE",
        "MG_PRE_SMOOTH": (1, 2, 3, 3), "MG_POST_SMOOTH": (0, 0, 0, 0),
        "MG_CORRECTION_SMOOTH": (0, 0, 0, 0),
        "MG_DAMP_RESTRICTION": 0.9, "MG_DAMP_PROLONGATION": 0.9,
        "CONV_NUM_METHOD_FLOW": "JST", "JST_SENSOR_COEFF": (0.5, 0.02),
        "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
        "ITER": int(iters),
    })
    return c


def rans_sa_config(mesh, mach, aoa, reynolds, re_length, temperature, wall, far, sym, ref_area,
                   ref_length, moment_ref, iters=1000, scheme="ROE", multigrid=0, cfl=4.0) -> dict:
    """RANS Spalart-Allmaras template (numerics of SU2 TestCases rans/oneram6/turb_ONERAM6.cfg with
    scheme='ROE', or rans/rae2822/turb_SA_RAE2822.cfg with scheme='JST', multigrid=3, cfl=10)."""
    c = _common(mesh, mach, aoa, ref_area, ref_length, moment_ref, wall, far, sym)
    c.update({
        "SOLVER": "RANS", "KIND_TURB_MODEL": "SA",
        "REYNOLDS_NUMBER": float(reynolds), "REYNOLDS_LENGTH": float(re_length),
        "FREESTREAM_TEMPERATURE": float(temperature),
        "FLUID_MODEL": "STANDARD_AIR", "VISCOSITY_MODEL": "SUTHERLAND",
        "MARKER_HEATFLUX": (wall, 0.0),
        "CFL_NUMBER": float(cfl), "CFL_ADAPT": False,
        "LINEAR_SOLVER": "FGMRES" if scheme == "ROE" else "BCGSTAB",
        "LINEAR_SOLVER_PREC": "LU_SGS" if scheme == "ROE" else "ILU",
        "LINEAR_SOLVER_ERROR": 1e-4 if scheme == "ROE" else 0.1,
        "LINEAR_SOLVER_ITER": 5 if scheme == "ROE" else 3,
        "CONV_NUM_METHOD_TURB": "SCALAR_UPWIND", "MUSCL_TURB": False,
        "TIME_DISCRE_FLOW": "EULER_IMPLICIT", "TIME_DISCRE_TURB": "EULER_IMPLICIT",
        "ITER": int(iters),
        "SCREEN_OUTPUT": ("INNER_ITER", "WALL_TIME", "RMS_DENSITY", "RMS_NU_TILDE", "LIFT", "DRAG"),
    })
    if scheme == "ROE":
        c.update({"NUM_METHOD_GRAD": "GREEN_GAUSS", "CONV_NUM_METHOD_FLOW": "ROE", "MUSCL_FLOW": True,
                  "SLOPE_LIMITER_FLOW": "VENKATAKRISHNAN", "VENKAT_LIMITER_COEFF": 1.0})
    elif scheme == "JST":
        c.update({"NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES", "CONV_NUM_METHOD_FLOW": "JST",
                  "JST_SENSOR_COEFF": (0.5, 0.02)})
    else:
        raise ValueError(f"unknown scheme {scheme}")
    if multigrid:
        c.update({"MGLEVEL": int(multigrid), "MGCYCLE": "W_CYCLE", "MG_PRE_SMOOTH": (4, 4, 4, 4),
                  "MG_POST_SMOOTH": (4, 4, 4, 4), "MG_CORRECTION_SMOOTH": (1, 1, 1, 1),
                  "MG_DAMP_RESTRICTION": 0.5, "MG_DAMP_PROLONGATION": 0.5})
    else:
        c["MGLEVEL"] = 0
    return c


# ============================================================================= history / surface

def _read_csv_columns(path) -> dict:
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f"empty file {path}")
    head, seen = [], {}
    for h in (h.strip().strip('"').strip() for h in lines[0].split(",")):
        seen[h] = seen[h] + 1 if h in seen else 1
        head.append(h if seen[h] == 1 else f"{h}#{seen[h]}")   # SU2 WALL_TIME gives two "Time(sec)"
    rows = [ln for ln in lines[1:] if ln.strip()]
    if not rows:
        return {h: np.zeros(0) for h in head}
    a = np.array([[float(v) for v in ln.split(",")] for ln in rows], dtype=np.float64)
    if a.shape[1] != len(head):
        raise ValueError(f"{path}: {a.shape[1]} values per row, {len(head)} headers")
    out = {}
    for j, h in enumerate(head):
        col = a[:, j]
        out[h] = col.astype(np.int64) if h in ("Time_Iter", "Outer_Iter", "Inner_Iter", "PointID") else col
    return out


def read_history(path) -> dict:
    return _read_csv_columns(path)


def concat_histories(hs) -> dict:
    """Join histories of consecutive restarted segments; adds 'iter_global' (0, 1, 2, ...)."""
    hs = [h for h in hs if len(next(iter(h.values()))) > 0]
    if not hs:
        return {"iter_global": np.zeros(0, np.int64)}
    keys = [k for k in hs[0] if all(k in h for h in hs)]
    out = {k: np.concatenate([h[k] for h in hs]) for k in keys}
    out["iter_global"] = np.arange(len(out[keys[0]]), dtype=np.int64)
    return out


def read_surface_csv(path) -> dict:
    return _read_csv_columns(path)


# ============================================================================= convergence rule

def convergence_settings() -> dict:
    c = _CFD["convergence"]
    return dict(residual_drop_orders=c["residual_drop_orders"], cl_change=float(c["cl_change"]),
                window_iters=int(c["window_iters"]), residual_field=c["residual_field"])


def convergence_status(hist: dict, orders=None, cl_tol=None, window=None, field=None, cl_field="CL") -> dict:
    """PLAN section 4: converged if the log10 residual dropped by >= `orders` (first logged value
    minus last value) or max-min of CL over the last `window` iterations is < `cl_tol`.
    'diverged' if the last residual or CL is not finite. Otherwise 'unconverged'."""
    s = convergence_settings()
    orders = s["residual_drop_orders"] if orders is None else orders
    cl_tol = s["cl_change"] if cl_tol is None else cl_tol
    window = s["window_iters"] if window is None else window
    field = s["residual_field"] if field is None else field
    r = np.asarray(hist.get(field, np.zeros(0)), float)
    cl = np.asarray(hist.get(cl_field, np.zeros(0)), float)
    n = int(len(r))
    out = dict(status="no_data", criterion=None, n_iter=n, residual_drop_orders=None, cl_change_window=None,
               rule=dict(residual_drop_orders=orders, cl_change=cl_tol, window_iters=window, field=field))
    if n == 0:
        return out
    if not (np.isfinite(r[-1]) and (len(cl) == 0 or np.isfinite(cl[-1]))):
        out["status"] = "diverged"
        return out
    drop = float(r[0] - r[-1])
    out["residual_drop_orders"] = drop
    if len(cl) >= window:
        out["cl_change_window"] = float(np.max(cl[-window:]) - np.min(cl[-window:]))
    if drop >= orders:
        out.update(status="converged", criterion="residual_drop")
    elif out["cl_change_window"] is not None and out["cl_change_window"] < cl_tol:
        out.update(status="converged", criterion="cl_change")
    else:
        out["status"] = "unconverged"
    return out


# ============================================================================= SU2 ASCII mesh io

@dataclass
class Su2Mesh:
    dim: int
    points: np.ndarray
    elements: dict                  # vtk type -> (n, k) int array
    markers: dict = field(default_factory=dict)   # name -> {vtk type -> (n, k)}

    @property
    def n_elements(self) -> int:
        return int(sum(len(v) for v in self.elements.values()))


def _group(conn_lines):
    by = {}
    for t, nodes in conn_lines:
        by.setdefault(t, []).append(nodes)
    return {t: np.array(v, dtype=np.int64) for t, v in by.items()}


def read_su2_mesh(path) -> Su2Mesh:
    with open(path) as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    dim, pts, elems, markers = None, None, {}, {}

    def val(s):
        return s.split("=", 1)[1].split()

    while i < n:
        s = lines[i].split("%", 1)[0].strip()
        i += 1
        if not s:
            continue
        if s.startswith("NDIME"):
            dim = int(val(s)[0])
        elif s.startswith("NELEM"):
            ne = int(val(s)[0])
            cl = []
            for ln in lines[i:i + ne]:
                v = ln.split()
                t = int(v[0])
                cl.append((t, [int(x) for x in v[1:1 + NODES_PER[t]]]))
            i += ne
            elems = _group(cl)
        elif s.startswith("NPOIN"):
            npn = int(val(s)[0])
            a = np.array([ln.split()[:dim] for ln in lines[i:i + npn]], dtype=np.float64)
            i += npn
            pts = a
        elif s.startswith("MARKER_TAG"):
            name = s.split("=", 1)[1].strip()
            while not lines[i].strip().startswith("MARKER_ELEMS"):
                i += 1
            me = int(val(lines[i])[0])
            i += 1
            cl = []
            for ln in lines[i:i + me]:
                v = ln.split()
                t = int(v[0])
                cl.append((t, [int(x) for x in v[1:1 + NODES_PER[t]]]))
            i += me
            markers[name] = _group(cl)
    if dim is None or pts is None:
        raise ValueError(f"{path}: not an SU2 ASCII mesh")
    return Su2Mesh(dim, pts, elems, markers)


def write_su2_mesh(path, points, elements: dict, markers: dict) -> Path:
    points = np.asarray(points, float)
    dim = points.shape[1]
    out = [f"NDIME= {dim}"]
    ne = sum(len(v) for v in elements.values())
    out.append(f"NELEM= {ne}")
    k = 0
    for t, conn in elements.items():
        for row in np.asarray(conn):
            out.append(f"{t} " + " ".join(str(int(x)) for x in row) + f" {k}")
            k += 1
    out.append(f"NPOIN= {len(points)}")
    for j, p in enumerate(points):
        out.append(" ".join(f"{x:.15e}" for x in p) + f" {j}")
    out.append(f"NMARK= {len(markers)}")
    for name, d in markers.items():
        out.append(f"MARKER_TAG= {name}")
        out.append(f"MARKER_ELEMS= {sum(len(v) for v in d.values())}")
        for t, conn in d.items():
            for row in np.asarray(conn):
                out.append(f"{t} " + " ".join(str(int(x)) for x in row))
    path = Path(path)
    path.write_text("\n".join(out) + "\n")
    return path


def read_su2_markers(path, names=None) -> dict:
    """Marker elements only (skips the volume elements and points; fast for large meshes)."""
    out = {}
    with open(path) as f:
        it = iter(f)
        for ln in it:
            s = ln.split("%", 1)[0].strip()
            if s.startswith(("NELEM", "NPOIN")):
                n = int(s.split("=", 1)[1].split()[0])
                for _ in range(n):
                    next(it)
            elif s.startswith("MARKER_TAG"):
                name = s.split("=", 1)[1].strip()
                ln2 = next(it)
                while not ln2.strip().startswith("MARKER_ELEMS"):
                    ln2 = next(it)
                n = int(ln2.split("=", 1)[1].split()[0])
                rows = [next(it).split() for _ in range(n)]
                if names is None or name in names:
                    out[name] = _group([(int(r[0]), [int(x) for x in r[1:1 + NODES_PER[int(r[0])]]]) for r in rows])
    return out


def tet_volumes(mesh: Su2Mesh) -> np.ndarray:
    t = mesh.elements.get(VTK_TET)
    if t is None:
        return np.zeros(0)
    p = mesh.points
    a, b, c, d = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]], p[t[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def ordered_marker_loop(mesh: Su2Mesh, marker: str) -> np.ndarray:
    """Node indices of a closed 2D boundary marker in chain order."""
    e = mesh.markers[marker][VTK_LINE]
    nxt = {}
    for a, b in e:
        nxt.setdefault(int(a), []).append(int(b))
        nxt.setdefault(int(b), []).append(int(a))
    start = int(e[0, 0])
    order, prev, cur = [start], None, start
    while True:
        cand = [q for q in nxt[cur] if q != prev]
        if not cand:
            break
        q = cand[0]
        if q == start:
            break
        order.append(q)
        prev, cur = cur, q
        if len(order) > len(nxt):
            raise ValueError(f"marker {marker} is not a simple loop")
    return np.array(order, dtype=np.int64)


# ============================================================================= segmented runner

def _hist_files(work: Path):
    return sorted(work.glob("history_seg*.csv"))


def run_segments(work, cfg: dict, su2_cmd, segment_iters: int, max_iters: int, deadline: float,
                 log=print) -> dict:
    """Run SU2 in segments of `segment_iters` iterations, restarting from its own solution, until
    the convergence rule holds, the iteration cap is reached, SU2 fails, or the wall-clock
    `deadline` (epoch seconds) would be passed by the next segment ('paused': resume later).
    State lives in work/state.json so a later job continues where this one stopped."""
    work = Path(work)
    sp = work / "state.json"
    st = json.loads(sp.read_text()) if sp.exists() else dict(segments=[], status="pending")
    while True:
        h = concat_histories([read_history(p) for p in _hist_files(work)])
        conv = convergence_status(h)
        n_done = int(conv["n_iter"])
        st.update(convergence=conv, n_iter=n_done)
        if conv["status"] in ("converged", "diverged"):
            st["status"] = conv["status"]
            break
        if n_done >= max_iters:
            st["status"] = "unconverged"
            st["stop_reason"] = f"iteration cap {max_iters}"
            break
        walls = [s["wall_s"] for s in st["segments"] if s.get("returncode") == 0]
        est = float(np.median(walls)) if walls else 0.0
        if walls and time.time() + 1.25 * est > deadline:
            st["status"] = "paused"
            st["stop_reason"] = "job time budget"
            break
        k = len(st["segments"])
        n_it = int(min(segment_iters, max_iters - n_done))
        c = dict(cfg, ITER=n_it, RESTART_SOL=n_done > 0, CONV_FILENAME=f"history_seg{k:03d}")
        write_config(work / f"seg{k:03d}.cfg", c)
        t0 = time.time()
        with open(work / f"seg{k:03d}.log", "w") as lf:
            p = subprocess.run(list(su2_cmd) + [f"seg{k:03d}.cfg"], cwd=work, stdout=lf, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        hk = work / f"history_seg{k:03d}.csv"
        n_k = len(read_history(hk).get("Inner_Iter", [])) if hk.exists() else 0
        st["segments"].append(dict(k=k, iters_requested=n_it, iters_logged=int(n_k), wall_s=wall,
                                   returncode=p.returncode, restart=n_done > 0))
        sp.write_text(json.dumps(st, indent=1))
        log(f"segment {k}: rc={p.returncode} iters={n_k} wall={wall:.1f}s")
        if p.returncode != 0 or n_k == 0:
            st["status"] = "failed"
            st["stop_reason"] = f"SU2 returned {p.returncode} in segment {k}"
            break
    st["wall_s_total"] = float(sum(s["wall_s"] for s in st["segments"]))
    st["iters_total"] = int(sum(s["iters_logged"] for s in st["segments"]))
    sp.write_text(json.dumps(st, indent=1))
    return st


def merged_history_csv(work, out_path) -> Path:
    """Write the concatenated history of all segments (with iter_global) as one CSV."""
    h = concat_histories([read_history(p) for p in _hist_files(Path(work))])
    keys = list(h)
    with open(out_path, "w") as f:
        f.write(",".join(f'"{k}"' for k in keys) + "\n")
        for i in range(len(h["iter_global"])):
            f.write(",".join(repr(float(h[k][i])) if h[k].dtype.kind == "f" else str(int(h[k][i])) for k in keys) + "\n")
    return Path(out_path)


# ============================================================================= transonic checks

def cp_star(mach: float, gamma: float = GAMMA) -> float:
    m2 = mach * mach
    return 2.0 / (gamma * m2) * (((2 + (gamma - 1) * m2) / (gamma + 1)) ** (gamma / (gamma - 1)) - 1.0)


def shock_location(x, cp, mach, x_range=(0.02, 0.98)) -> dict:
    """Upper-surface shock position. Primary: last downstream crossing of Cp* from below (supersonic)
    to above, linear interpolation. Secondary: midpoint of the segment with the largest dCp/dx.
    Both None when no point is supersonic."""
    x = np.asarray(x, float)
    cp = np.asarray(cp, float)
    ok = np.isfinite(x) & np.isfinite(cp)
    o = np.argsort(x[ok])
    x, cp = x[ok][o], cp[ok][o]
    m = (x >= x_range[0]) & (x <= x_range[1])
    x, cp = x[m], cp[m]
    cs = cp_star(mach)
    d = cp - cs
    out = dict(cp_star=cs, x_cpstar=None, x_maxgrad=None, n_points=int(len(x)))
    if len(x) < 2 or not np.any(d < 0):
        return out
    idx = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
    if len(idx):
        i = idx[-1]
        out["x_cpstar"] = float(x[i] + (x[i + 1] - x[i]) * (-d[i]) / (d[i + 1] - d[i]))
    g = np.diff(cp) / np.maximum(np.diff(x), 1e-12)
    j = int(np.argmax(g))
    out["x_maxgrad"] = float(0.5 * (x[j] + x[j + 1]))
    return out


def _linfit(s, v):
    A = np.stack([s, np.ones_like(s)], 1)
    return np.linalg.lstsq(A, v, rcond=None)[0]


def wing_station_cut(pts, cp, eta, span_axis=1, up_axis=2, stream_axis=0, n=161, band=0.02, n_slabs=10) -> dict:
    """Chordwise Cp at fractional span `eta` (0 at the root plane, 1 at the tip) from a surface point
    cloud. LE/TE lines are fitted locally (linear in span) from the per-slab min/max streamwise
    coordinate; points are split into upper/lower by the side of the local LE-TE chord line;
    Cp is interpolated linearly in (x/c, span). Returns x/c and Cp per surface."""
    from scipy.interpolate import LinearNDInterpolator
    pts = np.asarray(pts, float)
    s, x, z = pts[:, span_axis], pts[:, stream_axis], pts[:, up_axis]
    s0, s1 = s.min(), s.max()
    b = s1 - s0
    sc = s0 + eta * b
    w = band * b
    sel = np.abs(s - sc) <= w
    if sel.sum() < 50:
        raise ValueError(f"too few surface points near eta={eta}")
    ss, xs, zs, cps = s[sel], x[sel], z[sel], np.asarray(cp, float)[sel]
    edges = np.linspace(sc - w, sc + w, n_slabs + 1)
    le_s, le_x, le_z, te_s, te_x, te_z = [], [], [], [], [], []
    for a, c in zip(edges[:-1], edges[1:]):
        m = (ss >= a) & (ss <= c)
        if m.sum() < 4:
            continue
        i, j = np.argmin(np.where(m, xs, np.inf)), np.argmax(np.where(m, xs, -np.inf))
        le_s.append(ss[i]); le_x.append(xs[i]); le_z.append(zs[i])
        te_s.append(ss[j]); te_x.append(xs[j]); te_z.append(zs[j])
    pl, pt = _linfit(np.array(le_s), np.array(le_x)), _linfit(np.array(te_s), np.array(te_x))
    zl, zt = _linfit(np.array(le_s), np.array(le_z)), _linfit(np.array(te_s), np.array(te_z))
    xle = pl[0] * ss + pl[1]
    chord = (pt[0] * ss + pt[1]) - xle
    xi = (xs - xle) / chord
    zc = (zl[0] * ss + zl[1]) + xi * ((zt[0] * ss + zt[1]) - (zl[0] * ss + zl[1]))
    upper = zs >= zc
    xg = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    out = dict(eta=float(eta), span=float(sc), chord=float((pt[0] * sc + pt[1]) - (pl[0] * sc + pl[1])),
               x_le=float(pl[0] * sc + pl[1]), n_points=int(sel.sum()))
    for name, m in (("upper", upper), ("lower", ~upper)):
        f = LinearNDInterpolator(np.stack([xi[m], (ss[m] - sc) / w], 1), cps[m])
        out[name] = dict(x_c=xg, cp=f(np.stack([xg, np.zeros_like(xg)], 1)))
    return out


def wall_triangles(markers: dict, marker: str) -> np.ndarray:
    """Triangles (global point ids) of a 3D wall marker; quads are split in two. `markers` is
    Su2Mesh.markers or the result of read_su2_markers."""
    d = markers[marker]
    tris = [d[VTK_TRI]] if VTK_TRI in d else []
    if VTK_QUAD in d:
        q = d[VTK_QUAD]
        tris += [q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
    return np.concatenate(tris)


def slice_wall(points, tris, values, s_cut, span_axis=1, up_axis=2, stream_axis=0) -> dict:
    """Exact planar cut (span coordinate = s_cut) of a triangulated wall with linear interpolation of
    `values` along the cut edges. points (N, 3) indexed by `tris`; values (N,).
    Returns x/c and value per surface (sorted), chord and LE position of the cut."""
    P = np.asarray(points, float)
    v = np.asarray(values, float)
    T = np.asarray(tris)
    sv = P[T, span_axis] - s_cut
    crosses = (sv.min(axis=1) < 0) & (sv.max(axis=1) >= 0)
    T, sv = T[crosses], sv[crosses]
    pts, vals = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        sa, sb = sv[:, a], sv[:, b]
        m = (sa < 0) != (sb < 0)
        t = sa[m] / (sa[m] - sb[m])
        ia, ib = T[m, a], T[m, b]
        pts.append(P[ia] + t[:, None] * (P[ib] - P[ia]))
        vals.append(v[ia] + t * (v[ib] - v[ia]))
    if not pts or sum(len(q) for q in pts) < 6:
        raise ValueError(f"cut at {s_cut} crosses too few wall triangles")
    X, V = np.concatenate(pts), np.concatenate(vals)
    x, z = X[:, stream_axis], X[:, up_axis]
    i_le, i_te = int(np.argmin(x)), int(np.argmax(x))
    chord = float(x[i_te] - x[i_le])
    xi = (x - x[i_le]) / chord
    zc = z[i_le] + xi * (z[i_te] - z[i_le])
    upper = z >= zc
    out = dict(chord=chord, x_le=float(x[i_le]), span=float(s_cut), n_points=int(len(x)))
    for name, m in (("upper", upper), ("lower", ~upper)):
        o = np.argsort(xi[m])
        xs, vs = xi[m][o], V[m][o]
        keep = np.r_[True, np.diff(xs) > 1e-9]
        out[name] = dict(x_c=xs[keep], cp=vs[keep])
    return out


# ============================================================================= 2D O-grid

def _point_polyline_distance(p, poly):
    p = np.asarray(p, float)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.einsum("pij,ij->pi", p[:, None, :] - a[None], ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    t = np.minimum(np.maximum(t, 0.0), 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.min(np.linalg.norm(p[:, None, :] - q, axis=2), axis=1)


def _resample(poly, n, cluster=True):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    u = np.linspace(0, 1, n)
    if cluster:
        u = 0.5 * (1 - np.cos(np.pi * u))
    t = u * d[-1]
    return np.stack([np.interp(t, d, poly[:, 0]), np.interp(t, d, poly[:, 1])], 1)


def airfoil_wall_nodes(xy, n_around, n_base=None):
    """Wall nodes of a closed O-grid, clockwise (lower TE -> LE -> upper TE -> base). Input loop runs
    lower TE -> LE -> upper TE; a blunt TE (first != last point) gets a base segment."""
    xy = np.asarray(xy, float)
    ile = int(np.argmin(xy[:, 0]))
    gap = float(np.linalg.norm(xy[-1] - xy[0]))
    blunt = gap > 1e-9
    nb = (n_base if n_base is not None else (4 if blunt else 0)) if blunt else 0
    n_side = (n_around - nb) // 2
    n_low = n_around - nb - n_side
    low = _resample(xy[:ile + 1], n_low + 1)[:-1]
    up = _resample(xy[ile:], n_side + 1)
    up = up[:-1] if not blunt else up
    nodes = [low, up]
    if blunt:
        up = up[:-1] if nb == 0 else up
        base = _resample(np.stack([xy[-1], xy[0]]), nb + 1, cluster=False)[1:-1]
        nodes = [low, up, base]
    w = np.concatenate(nodes)
    if len(w) != n_around:
        w = _resample(np.vstack([w, w[:1]]), n_around + 1, cluster=False)[:-1]
    return w


def _normals_cw(w, smooth=0):
    t = np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)          # left of tangent; for a clockwise loop: outward
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    for _ in range(smooth):
        nrm = 0.5 * nrm + 0.25 * (np.roll(nrm, 1, 0) + np.roll(nrm, -1, 0))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return nrm


def _growth_ratio(h1, n, total):
    lo, hi = 1.0 + 1e-9, 3.0
    f = lambda g: h1 * (g ** n - 1) / (g - 1) - total
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def ogrid_2d(xy, n_around, n_normal, h1, radius, near_dist=0.04, normal_smooth=20):
    """Structured O-grid (n_normal + 1, n_around): wall layer by normal marching with geometric
    spacing from h1 up to `near_dist` (chords), then straight-line blending to a circle of radius
    `radius` centred mid-way between LE and TE. Node j = 1 is exactly h1 from the wall."""
    w = airfoil_wall_nodes(xy, n_around)
    ile = int(np.argmin(w[:, 0]))
    ite = int(np.argmax(w[:, 0]))
    return ogrid_from_loop(w, n_normal, h1, radius, 0.5 * (w[ile] + w[ite]), near_dist, normal_smooth)


def ogrid_from_loop(w, n_normal, h1, radius, centre, near_dist=0.04, normal_smooth=20):
    """O-grid around an exact closed loop of nodes w (n, 2), clockwise; see ogrid_2d."""
    w = np.asarray(w, float)
    a = 0.5 * np.sum(w[:, 0] * np.roll(w[:, 1], -1) - np.roll(w[:, 0], -1) * w[:, 1])
    if a > 0:
        raise ValueError("wall loop must be clockwise")
    n_around = len(w)
    g = _growth_ratio(h1, n_normal, radius)
    d = np.r_[0.0, h1 * (g ** np.arange(1, n_normal + 1) - 1) / (g - 1)]
    jn = int(np.searchsorted(d, near_dist))
    if jn < 2 or jn > n_normal - 2:
        raise ValueError(f"near-wall layer index {jn} outside [2, {n_normal - 2}]: change h1, n_normal or near_dist")
    nrm0 = _normals_cw(w, 0)
    nrm_s = _normals_cw(w, normal_smooth)
    X = np.zeros((n_normal + 1, n_around))
    Y = np.zeros_like(X)
    for j in range(jn + 1):
        bl = min(1.0, d[j] / d[jn])
        nv = (1 - bl) * nrm0 + bl * nrm_s
        nv /= np.linalg.norm(nv, axis=1, keepdims=True)
        X[j], Y[j] = w[:, 0] + d[j] * nv[:, 0], w[:, 1] + d[j] * nv[:, 1]
    c = np.asarray(centre, float)
    L = np.stack([X[jn], Y[jn]], 1)
    th = np.unwrap(np.arctan2(L[:, 1] - c[1], L[:, 0] - c[0]))
    if not np.all(np.diff(th) < 0):
        s_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))]
        th = th[0] - 2 * np.pi * s_ / (s_[-1] + np.linalg.norm(L[-1] - L[0]))
    F = np.stack([c[0] + radius * np.cos(th), c[1] + radius * np.sin(th)], 1)
    for j in range(jn + 1, n_normal + 1):
        phi = (d[j] - d[jn]) / (d[-1] - d[jn])
        X[j], Y[j] = L[:, 0] + phi * (F[:, 0] - L[:, 0]), L[:, 1] + phi * (F[:, 1] - L[:, 1])
    return X, Y


def quad_areas(X, Y):
    """Signed areas of the cells (j, i) -> nodes (j,i), (j,i+1), (j+1,i+1), (j+1,i), periodic in i."""
    x0, y0 = X[:-1], Y[:-1]
    x1, y1 = np.roll(X[:-1], -1, 1), np.roll(Y[:-1], -1, 1)
    x2, y2 = np.roll(X[1:], -1, 1), np.roll(Y[1:], -1, 1)
    x3, y3 = X[1:], Y[1:]
    return 0.5 * ((x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3))


def ogrid_to_su2(X, Y, path, wall="airfoil", far="farfield") -> Path:
    nj, ni = X.shape
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    idx = np.arange(nj * ni).reshape(nj, ni)
    nx = np.roll(idx, -1, axis=1)
    quads = np.stack([idx[:-1].ravel(), nx[:-1].ravel(), nx[1:].ravel(), idx[1:].ravel()], 1)
    wl = np.stack([idx[0], nx[0]], 1)
    fl = np.stack([idx[-1], nx[-1]], 1)
    return write_su2_mesh(path, pts, {VTK_QUAD: quads}, {wall: {VTK_LINE: wl}, far: {VTK_LINE: fl}})


def first_cell_height(reynolds, yplus=1.0, chord=1.0) -> float:
    """Flat-plate estimate (Schlichting skin friction, Cf = (2 log10 Re - 0.65)^-2.3)."""
    cf = (2 * math.log10(reynolds) - 0.65) ** -2.3
    return float(yplus * chord / (reynolds * math.sqrt(cf / 2)))


def airfoil_from_su2(mesh: Su2Mesh, marker: str) -> np.ndarray:
    """Closed airfoil loop from a 2D SU2 mesh marker, ordered lower TE -> LE -> upper TE."""
    loop = ordered_marker_loop(mesh, marker)
    p = mesh.points[loop, :2]
    a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if a > 0:
        p = p[::-1]
    ite = int(np.argmax(p[:, 0]))
    p = np.roll(p, -ite, axis=0)
    ile = int(np.argmin(p[:, 0]))
    # clockwise from the TE: the first half is the lower surface if its mean y is below the upper half
    if np.mean(p[1:ile, 1]) > np.mean(p[ile + 1:, 1]):
        p = np.vstack([p[:1], p[1:][::-1]])
    return np.vstack([p, p[:1]])


# ============================================================================= frames, mapping

SW_TO_SU2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def sw_to_su2(p):
    return np.asarray(p, float) @ SW_TO_SU2.T


def su2_to_sw(p):
    return np.asarray(p, float) @ SW_TO_SU2


def map_surface_to_grid(pts_sw, values, centres, k=4):
    """Inverse-distance (k nearest) interpolation of solver surface values onto data cell centres
    (3, I, J). Exact hits take the coincident value."""
    from scipy.spatial import cKDTree
    q = np.asarray(centres, float).reshape(3, -1).T
    d, i = cKDTree(np.asarray(pts_sw, float)).query(q, k=k)
    w = 1.0 / np.maximum(d, 1e-12) ** 2
    v = np.sum(w * np.asarray(values, float)[i], axis=1) / np.sum(w, axis=1)
    return v.reshape(np.asarray(centres).shape[1:])


def compare_superwing(surface: dict, nodes_sw, aoa, s_ref, cp_ref_cells=None, ref_point=(0.25, 0.0, 0.0)) -> dict:
    """CL, CD (pressure only) and CMz of an SU2 wall solution, integrated on the data grid in the SW
    frame exactly as Task 1 checked against cl_surface; optional Cp RMS against reference cells."""
    from . import data as _data
    pts = su2_to_sw(np.stack([surface["x"], surface["y"], surface["z"]], 1))
    cc = _data.cell_centres(np.asarray(nodes_sw, float))
    cp = map_surface_to_grid(pts, surface["Pressure_Coefficient"], cc)
    cl, cd, cm = _data.integrate_coefficients(nodes_sw, cp, aoa, s_ref, ref_point=ref_point)
    out = dict(cl=cl, cd=cd, cm=cm, cp_cells=cp)
    if cp_ref_cells is not None:
        out["cp_rmse_vs_ref"] = float(np.sqrt(np.mean((cp - np.asarray(cp_ref_cells, float)) ** 2)))
    return out


def integrate_wall(pts_sw, tris, cp, aoa, s_ref, ref_point=(0.25, 0.0, 0.0), c_ref=1.0, nodes_sw=None) -> dict:
    """Pressure-only CL, CD, CMz on the solver's own wall triangulation (SW frame, nodal Cp averaged to
    each triangle). Triangle normals are oriented outward with the nearest data-grid cell normal
    (nodes_sw, the data surface) when given; spanwise-facing tip-cap triangles do not affect lift,
    drag or pitching moment."""
    from scipy.spatial import cKDTree
    from . import data as _data
    P = np.asarray(pts_sw, float)
    T = np.asarray(tris)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    A = 0.5 * np.cross(b - a, c - a)
    cen = (a + b + c) / 3.0
    n_flip = 0
    if nodes_sw is not None:
        Ad = _data.outward_vector_areas(np.asarray(nodes_sw, float)).reshape(3, -1).T
        cd_ = _data.cell_centres(np.asarray(nodes_sw, float)).reshape(3, -1).T
        _, j = cKDTree(cd_).query(cen)
        flip = np.einsum("ij,ij->i", A, Ad[j]) < 0
        A[flip] *= -1
        n_flip = int(flip.sum())
    cpt = np.asarray(cp, float)[T].mean(axis=1)
    dF = -cpt[:, None] * A
    F = dF.sum(axis=0)
    al = np.radians(aoa)
    r = cen - np.asarray(ref_point, float)
    mz = np.sum(r[:, 0] * dF[:, 1] - r[:, 1] * dF[:, 0])
    return dict(cl=float((F[1] * np.cos(al) - F[0] * np.sin(al)) / s_ref),
                cd=float((F[0] * np.cos(al) + F[1] * np.sin(al)) / s_ref),
                cm=float(mz / (s_ref * c_ref)), n_triangles=int(len(T)), n_flipped=n_flip)


# ============================================================================= 3D Euler mesh (gmsh)

def loft_rows(s, stride=4):
    """Span rows used as loft sections: every `stride`-th row plus root, kink and tip. The kink row
    is where the chord law changes slope (largest |second difference| of the chord)."""
    s = np.asarray(s, float)
    ns, nc = s.shape[1], s.shape[2]
    x_te = 0.5 * (s[0, :, 0] + s[0, :, -1])
    chord = x_te - s[0, :, nc // 2]
    d2 = np.abs(np.diff(chord, 2))
    kink = int(np.argmax(d2)) + 1
    rows = sorted(set(range(0, ns, stride)) | {0, kink, ns - 1})
    return np.array(rows, dtype=np.int64), kink


def _point_tri_distance(p, a, b, c):
    """Exact distances from points p (n,3) to triangles (a, b, c) (n,3 each), Ericson's method."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp_ = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp_), np.einsum("ij,ij->i", ac, cp_)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    q = np.empty_like(p)
    den = np.where(np.abs(va + vb + vc) > 0, va + vb + vc, 1.0)
    v, w = vb / den, vc / den
    q[:] = a + ab * v[:, None] + ac * w[:, None]
    m = (d1 <= 0) & (d2 <= 0); q[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3); q[m] = b[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(d1 - d3 != 0, d1 - d3, 1.0); q[m] = (a + ab * t[:, None])[m]
    m = (d6 >= 0) & (d5 <= d6); q[m] = c[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(d2 - d6 != 0, d2 - d6, 1.0); q[m] = (a + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) != 0, (d4 - d3) + (d5 - d6), 1.0)
    q[m] = (b + (c - b) * t[:, None])[m]
    return np.linalg.norm(p - q, axis=1)


def distance_to_surface(points, v, f, k=12):
    from scipy.spatial import cKDTree
    cen = v[f].mean(axis=1)
    _, idx = cKDTree(cen).query(points, k=k)
    best = np.full(len(points), np.inf)
    for j in range(k):
        tri = f[idx[:, j]]
        best = np.minimum(best, _point_tri_distance(points, v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]))
    return best


def wing_mesh_3d(s_sw, out_path, size_factor=1.0, radius=None, stride=None, n_threads=None, log=None) -> dict:
    """Tetrahedral Euler mesh around a SuperWing-format surface (3, 129, 257), SW frame in, SU2 frame
    out. Wing = two OpenCASCADE lofts (root-kink, kink-tip) through B-spline sections that interpolate
    the data nodes; the root section lies on the symmetry plane; the far field is a half box of
    half-width `radius`. Markers: wing, symmetry, farfield. Sizes (root-chord units, spec
    cfd.mesh3d_euler) are multiplied by `size_factor` (grid levels for the grid study)."""
    import gmsh
    from . import geometry as _geom
    ms = _CFD["mesh3d_euler"]
    radius = float(_CFD["superwing_holdout"]["farfield_radius_root_chords"] if radius is None else radius)
    stride = int(ms["loft_span_stride"] if stride is None else stride)
    s_sw = np.asarray(s_sw, float)
    ns, nc = s_sw.shape[1], s_sw.shape[2]
    P = sw_to_su2(s_sw.reshape(3, -1).T).reshape(ns, nc, 3)
    rows, kink = loft_rows(s_sw, stride)
    y_root = float(P[0, 0, 1])
    if np.ptp(P[0, :, 1]) > 1e-9:
        raise ValueError("root section is not planar")
    t0 = time.time()
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 1 if log else 0)
        if n_threads:
            gmsh.option.setNumber("General.NumThreads", int(n_threads))
        gmsh.model.add("superwing")
        occ = gmsh.model.occ

        le = nc // 2

        def wire(i):
            # lower (TE -> LE) and upper (LE -> TE) splines through the data nodes, joined at the LE
            # node; a sharp TE (origingeom.npy: first node == last node) shares one TE point, a blunt
            # TE (builder, 0.004c) gets a straight base segment
            sharp = np.linalg.norm(P[i, 0] - P[i, -1]) < 1e-9
            tags = [occ.addPoint(*P[i, j]) for j in range(nc - 1 if sharp else nc)]
            if sharp:
                tags.append(tags[0])
            lo = occ.addSpline(tags[:le + 1])
            up = occ.addSpline(tags[le:])
            parts = [lo, up] if sharp else [lo, up, occ.addLine(tags[-1], tags[0])]
            return occ.addWire(parts)

        inner = occ.addThruSections([wire(i) for i in rows if i <= kink], makeSolid=True, makeRuled=False)
        outer = occ.addThruSections([wire(i) for i in rows if i >= kink], makeSolid=True, makeRuled=False)
        wing, _ = occ.fuse([d for d in inner if d[0] == 3], [d for d in outer if d[0] == 3])
        occ.synchronize()
        loft_vol = float(sum(occ.getMass(3, t) for d, t in wing if d == 3))
        xc = float(0.5 * (P[..., 0].min() + P[..., 0].max()))
        box = occ.addBox(xc - radius, y_root - radius, -radius, 2 * radius, radius, 2 * radius)
        fluid, _ = occ.cut([(3, box)], wing)
        occ.synchronize()
        vols = [t for d, t in fluid if d == 3]
        if len(vols) != 1:
            raise RuntimeError(f"boolean cut gave {len(vols)} volumes")
        wall, sym, far = [], [], []
        tol = 1e-6 * radius
        for d, t in gmsh.model.getBoundary([(3, vols[0])], oriented=False):
            bb = gmsh.model.getBoundingBox(2, t)
            if abs(bb[1] - y_root) < tol and abs(bb[4] - y_root) < tol:
                sym.append(t)
            elif (bb[3] - bb[0]) > radius or (bb[5] - bb[2]) > radius or abs(bb[1] - (y_root - radius)) < tol:
                far.append(t)
            else:
                wall.append(t)
        if not (wall and len(sym) == 1 and far):
            raise RuntimeError(f"surface classification failed: wall={wall} sym={sym} far={far}")
        gmsh.model.addPhysicalGroup(2, wall, name="wing")
        gmsh.model.addPhysicalGroup(2, sym, name="symmetry")
        gmsh.model.addPhysicalGroup(2, far, name="farfield")
        gmsh.model.addPhysicalGroup(3, vols, name="fluid")
        f = float(size_factor)
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", wall)
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", ms["size_wing_max"] * f)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", ms["size_far"] * f)
        gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.02)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", 0.4 * radius)
        gmsh.model.mesh.field.setAsBackgroundMesh(ft)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", ms["curvature_points"])
        gmsh.option.setNumber("Mesh.MeshSizeMin", ms["size_min"] * f)
        gmsh.option.setNumber("Mesh.MeshSizeMax", ms["size_far"] * f)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        types, tags3, _ = gmsh.model.mesh.getElements(3)
        n_cells = int(sum(len(t) for t in tags3))
        wn = set()
        for t in wall:
            nt, _, _ = gmsh.model.mesh.getNodes(2, t, includeBoundary=True)
            wn.update(int(x) for x in nt)
        allt, allc, _ = gmsh.model.mesh.getNodes()
        lut = {int(t): i for i, t in enumerate(allt)}
        allc = allc.reshape(-1, 3)
        wall_pts = allc[[lut[t] for t in sorted(wn)]]
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        out_path = Path(out_path)
        gmsh.write(str(out_path))
        gmsh_version = gmsh.__version__
    finally:
        gmsh.finalize()
    v, fc = _geom.close_surface(s_sw)
    dist = distance_to_surface(su2_to_sw(wall_pts), v, fc)
    closed_vol = _geom.check_watertight(v, fc)["signed_volume"]
    return dict(n_cells=n_cells, n_wall_nodes=int(len(wall_pts)), loft_rows=rows.tolist(), kink_row=int(kink),
                loft_volume=loft_vol, closed_surface_volume=float(closed_vol),
                wall_to_data_max=float(dist.max()),
                wall_to_data_rms=float(np.sqrt(np.mean(dist ** 2))), y_symmetry=y_root,
                size_factor=float(size_factor), radius=radius, mesh_seconds=time.time() - t0,
                gmsh_version=gmsh_version, options={k: ms[k] for k in ms})


# ============================================================================= experimental files

_NUM = re.compile(r"^[\s,]*[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?([\s,]+[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?)+[\s,]*$")


def _numeric_row(ln):
    if not _NUM.match(ln):
        return None
    return [float(t.replace("D", "e").replace("d", "e")) for t in re.split(r"[\s,]+", ln.strip()) if t]


def parse_numeric_blocks(text) -> list:
    """Blocks of consecutive numeric rows (>= 2 columns); any other line ends a block."""
    blocks, cur = [], []
    for ln in text.splitlines():
        r = _numeric_row(ln)
        if r is None:
            if cur:
                blocks.append(cur)
            cur = []
            continue
        if cur and len(r) != len(cur[0]):
            blocks.append(cur)
            cur = []
        cur.append(r)
    if cur:
        blocks.append(cur)
    return [np.array(b, dtype=np.float64) for b in blocks]


def parse_numeric_table(text):
    """First two columns of all numeric rows (x/c, Cp)."""
    rows = [r for r in (_numeric_row(ln) for ln in text.splitlines()) if r is not None]
    if not rows:
        raise ValueError("no numeric rows")
    a = np.array([r[:2] for r in rows], dtype=np.float64)
    return a[:, 0], a[:, 1]


# ============================================================================= gates and selection

def project_rans_hours(seconds_per_iter, cells_measured, iters_to_converge, cells_target) -> float:
    """Linear projection of RANS wall time in cells (iterations held at the measured count)."""
    return float(seconds_per_iter * (cells_target / cells_measured) * iters_to_converge / 3600.0)


def gate_g1(rans_hours, rans_accuracy, euler_hours) -> dict:
    """PLAN Gate G1. rans_accuracy: dict(d_cl, d_cd_counts, status) on the held-out wing or None."""
    g = _CFD["gates"]["G1"]
    out = dict(rans_hours=rans_hours, euler_hours=euler_hours, rans_accuracy=rans_accuracy,
               thresholds=dict(g), decision=None, n_designs=None, reason=None)
    rans_time_ok = rans_hours is not None and rans_hours <= g["rans_max_hours"]
    if rans_time_ok and rans_accuracy is None:
        out.update(decision="UNDECIDED", reason="RANS is fast enough but its accuracy on the held-out wing is not measured")
        return out
    acc_ok = (rans_accuracy is not None and rans_accuracy.get("status") == "converged"
              and abs(rans_accuracy["d_cl"]) <= g["cl_tol"] and abs(rans_accuracy["d_cd_counts"]) <= g["cd_tol_counts"])
    if rans_time_ok and acc_ok:
        out.update(decision="RANS", n_designs=[8, 10], reason="RANS within time and accuracy limits")
        return out
    why = "RANS too slow" if not rans_time_ok else "RANS outside the accuracy limits or unconverged"
    if rans_hours is None:
        why = "RANS time not measured"
    if euler_hours is None:
        out.update(decision="UNDECIDED", reason=f"{why}; Euler time not measured")
    elif euler_hours <= g["euler_max_hours"]:
        out.update(decision="EULER", n_designs=[30, 30], reason=f"{why}; Euler within {g['euler_max_hours']} h")
    else:
        out.update(decision="FRAMING_D", reason=f"{why}; Euler slower than {g['euler_max_hours']} h")
    return out


def select_holdout_case(case_row, case_split, mach, aoa, row_range, seed, mach_target) -> dict:
    """Pre-declared rule (spec cfd.superwing_holdout): wings of the official test split with
    row_range[0] <= row < row_range[1]; one row drawn by numpy default_rng(seed).choice from the
    sorted eligible rows; its case with Mach closest to mach_target (ties: lower AoA)."""
    from .data import SPLIT_TEST
    case_row = np.asarray(case_row).astype(np.int64)
    case_split = np.asarray(case_split)
    ok = (case_row >= row_range[0]) & (case_row < row_range[1]) & (case_split == SPLIT_TEST)
    rows = np.unique(case_row[ok])
    if not len(rows):
        raise ValueError("no eligible test wings in the row range")
    r = int(np.random.default_rng(seed).choice(rows))
    cases = np.where(case_row == r)[0]
    key = np.lexsort((np.asarray(aoa)[cases], np.abs(np.asarray(mach)[cases] - mach_target)))
    return dict(wing_row=r, case_index=int(cases[key[0]]), n_eligible_wings=int(len(rows)))


# ============================================================================= runner: cases

def download(url, dest, timeout=600, retries=4) -> dict:
    """HTTP download with SHA-256 and size; the record goes into the provenance of every result."""
    import hashlib
    import requests
    from .records import utc_now
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for k in range(retries):
        try:
            h = hashlib.sha256()
            n = 0
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dest.with_suffix(dest.suffix + ".part"), "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
            dest.with_suffix(dest.suffix + ".part").replace(dest)
            return dict(url=url, sha256=h.hexdigest(), bytes=n, retrieved_utc=utc_now())
        except Exception as e:  # network errors are retried, then raised
            last = e
            time.sleep(5 * 2 ** k)
    raise RuntimeError(f"download failed: {url}: {last}")


def github_raw(repo, commit, path) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def hf_url(repo, revision, name) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"


def _require(d, *keys):
    miss = [k for k in keys if k not in d]
    if miss:
        raise KeyError(f"case.json lacks {miss}")
    return [d[k] for k in keys]


def prepare_case(case: dict, work, log=print) -> dict:
    """Mesh and inputs of one case in `work`. Kinds:
      su2_testcase  mesh file from github.com/su2code/TestCases at a pinned commit
      rae_ogrid     airfoil taken from a TestCases mesh marker, new O-grid (spec cfd.mesh2d_ogrid)
      superwing     held-out SuperWing wing (spec cfd.superwing_holdout) from Hugging Face, gmsh mesh
    Returns the 'prepared' record (provenance, mesh statistics, solver inputs)."""
    from . import data as _data
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    kind = case["kind"]
    prep = dict(kind=kind, downloads={})
    t0 = time.time()
    if kind in ("su2_testcase", "rae_ogrid"):
        repo, commit, path = _require(case["mesh"], "repo", "commit", "path")
        src = work / Path(path).name
        prep["downloads"][Path(path).name] = download(github_raw(repo, commit, path), src)
        if kind == "su2_testcase":
            prep["mesh_file"] = src.name
        else:
            og = _CFD["mesh2d_ogrid"]
            m = read_su2_mesh(src)
            xy = airfoil_from_su2(m, case["mesh"]["airfoil_marker"])
            h1 = first_cell_height(case["flow"]["reynolds"], og["yplus"])
            X, Y = ogrid_2d(xy, og["n_around"], og["n_normal"], h1, og["farfield_radius_chords"])
            a = quad_areas(X, Y)
            if np.any(a <= 0):
                raise RuntimeError(f"O-grid has {int(np.sum(a <= 0))} non-positive cells")
            ogrid_to_su2(X, Y, work / "ogrid.su2", wall="airfoil", far="farfield")
            prep["mesh_file"] = "ogrid.su2"
            prep["ogrid"] = dict(og, h1=h1, n_wall_input=int(len(xy) - 1), min_cell_area=float(a.min()))
    elif kind == "superwing":
        hs = _CFD["superwing_holdout"]
        repo, rev = _require(case["hf"], "repo", "revision")
        hf_dir = work / "hf"
        for n in ("configs.dat", "index.npy", "train.parquet", "test.parquet"):
            if not (hf_dir / n).exists():
                prep["downloads"][n] = download(hf_url(repo, rev, n), hf_dir / n)
        cfgw = _data.read_configs(hf_dir / "configs.dat")
        idx = _data.read_index(hf_dir / "index.npy", cfgw)
        tab = _data.read_split_tables(hf_dir / "train.parquet", hf_dir / "test.parquet")
        test_rows = cfgw.rows_of(np.unique(tab["wing_shape_idx"][tab["is_test"]].astype(np.int64)))
        split = np.where(np.isin(idx.wing_row, test_rows), _data.SPLIT_TEST, _data.SPLIT_FIT)
        sel = select_holdout_case(idx.wing_row, split, idx.mach, idx.aoa, tuple(hs["row_range"]), hs["seed"], hs["mach_target"])
        r, ci = sel["wing_row"], sel["case_index"]
        (work / "selection.json").write_text(json.dumps(dict(sel, wing_id=int(cfgw.wing_id[r]), index_row={
            c: float(idx.array[ci, j]) for j, c in enumerate(_data.INDEX_COLUMNS)}, downloads=prep["downloads"]), indent=1))
        surf = _data.NpySource(hf_url(repo, rev, "origingeom.npy")).read(r, r + 1)[0]
        cp_ref = _data.NpySource(hf_url(repo, rev, "data.npy")).read(ci, ci + 1, channel=0)[0]
        np.save(work / "surface_origingeom.npy", surf)
        np.save(work / "cp_adflow_cells.npy", cp_ref)
        row = {c: float(idx.array[ci, j]) for j, c in enumerate(_data.INDEX_COLUMNS)}
        prep["superwing"] = dict(selection=sel, wing_id=int(cfgw.wing_id[r]), index_row=row, hf_repo=repo,
                                 hf_revision=rev, license="CC BY-SA 4.0 (Yang et al., SuperWing)")
        if case["mesh"]["type"] == "euler_tetra":
            st = wing_mesh_3d(surf, work / "mesh.su2", size_factor=float(case["mesh"]["size_factor"]), log=log)
        elif case["mesh"]["type"] == "rans_hybrid":
            mr = _CFD["mesh3d_rans"]
            st = wing_mesh_3d_rans(surf, work / "mesh.su2", reynolds=float(case["flow"]["reynolds"]),
                                   yplus=mr["yplus"], growth=mr["growth"], last_layer_chords=mr["last_layer_chords"],
                                   span_stride=mr["span_stride"], chord_stride=mr["chord_stride"],
                                   n_radial=mr["n_radial"], n_rings=mr["n_rings"], log=log)
        else:
            raise ValueError(f"unknown SuperWing mesh type {case['mesh']['type']}")
        prep["mesh3d"] = st
        prep["mesh_file"] = "mesh.su2"
        prep["flow"] = dict(mach=row["mach"], aoa=row["aoa"], ref_area=row["s_ref_half"])
    else:
        raise ValueError(f"unknown case kind {kind}")
    from .records import sha256_file
    m = work / prep["mesh_file"]
    prep["mesh_sha256"] = sha256_file(m)
    prep["mesh_bytes"] = m.stat().st_size
    prep["mesh_cells"] = _count_cells(m)
    prep["prepare_s"] = time.time() - t0
    return prep


def _count_cells(path) -> int:
    with open(path) as f:
        for ln in f:
            if ln.startswith("NELEM"):
                return int(ln.split("=", 1)[1].split()[0])
    raise ValueError(f"no NELEM in {path}")


def case_config(case: dict, prep: dict) -> dict:
    fl = dict(case["flow"])
    fl.update(prep.get("flow", {}))
    mk = case["markers"]
    common = dict(wall=mk["wall"], far=mk["far"], sym=mk.get("sym"), ref_area=fl["ref_area"],
                  ref_length=fl["ref_length"], moment_ref=tuple(fl["moment_ref"]))
    if case["solver"] == "euler":
        c = euler_config(prep["mesh_file"], fl["mach"], fl["aoa"], **common)
    elif case["solver"] == "rans_sa":
        nm = case["numerics"]
        c = rans_sa_config(prep["mesh_file"], fl["mach"], fl["aoa"], fl["reynolds"], fl["reynolds_length"],
                           fl["temperature"], scheme=nm["scheme"], multigrid=nm["multigrid"], cfl=nm["cfl"], **common)
    else:
        raise ValueError(case["solver"])
    c.update(case.get("config_overrides", {}))
    return c


def timing_summary(work) -> dict:
    """Per-iteration wall time from SU2's own WALL_TIME history column (median over all iterations
    except the first 5 of each segment) and start-up time (segment wall time minus its iterations)."""
    work = Path(work)
    st = json.loads((work / "state.json").read_text())
    per, start = [], []
    for k, s in enumerate(st["segments"]):
        p = work / f"history_seg{k:03d}.csv"
        if not p.exists():
            continue
        h = read_history(p)
        t = h.get("Time(sec)")
        if t is None or len(t) == 0:
            continue
        per.append(t[5:] if len(t) > 10 else t)
        start.append(s["wall_s"] - float(np.sum(t)))
    if not per:
        return dict(seconds_per_iter=None, startup_s_median=None)
    allt = np.concatenate(per)
    return dict(seconds_per_iter=float(np.median(allt)), seconds_per_iter_mean=float(np.mean(allt)),
                startup_s_median=float(np.median(start)), n_segments=len(per))


# ============================================================================= 3D RANS mesh (prism layer + gmsh tetra)

def _vertex_normals(P, T):
    A = 0.5 * np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, T[:, k], A)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def _smooth_normals(N, T, n_iter, fixed_span_nodes, span_axis):
    nb_sum = np.zeros_like(N)
    cnt = np.zeros(len(N))
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    for _ in range(n_iter):
        nb_sum[:] = 0.0
        cnt[:] = 0.0
        np.add.at(nb_sum, E[:, 0], N[E[:, 1]])
        np.add.at(nb_sum, E[:, 1], N[E[:, 0]])
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        N = 0.5 * N + 0.5 * nb_sum / cnt[:, None]
        N[fixed_span_nodes, span_axis] = 0.0
        N /= np.linalg.norm(N, axis=1, keepdims=True)
    return N


def _face_normals(P, T):
    A = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)


def _min_dot(N, F, T):
    """Per node: minimum dot product of its normal with the unit normals of its adjacent faces."""
    out = np.full(len(N), np.inf)
    for k in range(3):
        np.minimum.at(out, T[:, k], np.einsum("ij,ij->i", N[T[:, k]], F))
    return out


def _visible_normals(N, F, T, tau=0.25, n_iter=200):
    """Push node normals toward adjacent faces whose normal makes a dot product below `tau`, so that
    every adjacent face sees the node normal from its outer side (sharp edges: TE, tip cap)."""
    N = N.copy()
    for _ in range(n_iter):
        md = _min_dot(N, F, T)
        act = md < tau
        if not np.any(act):
            break
        corr = np.zeros_like(N)
        for k in range(3):
            d = np.einsum("ij,ij->i", N[T[:, k]], F)
            w = np.where(act[T[:, k]] & (d < tau), tau - d, 0.0)
            np.add.at(corr, T[:, k], 0.5 * w[:, None] * F)
        N = N + corr
        N /= np.linalg.norm(N, axis=1, keepdims=True)
    return N


def prism_corner_jacobians(bottom, top):
    """Six corner Jacobians of prisms (bottom a, b, c counter-clockwise seen from the top d, e, f):
    each corner's two triangle edges and its vertical edge. All six > 0 for a valid prism."""
    a, b, c = bottom
    d, e, f = top
    det = lambda u, v, w: np.einsum("ij,ij->i", np.cross(u, v), w)
    return (det(b - a, c - a, d - a), det(c - b, a - b, e - b), det(a - c, b - c, f - c),
            det(e - d, f - d, d - a), det(f - e, d - e, e - b), det(d - f, e - f, f - c))


def prism_volumes(bottom, top):
    """Volumes of prisms (bottom triangle, top triangle), each split into 3 tetrahedra."""
    def tv(a, b, c, d):
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    a, b, c = bottom
    d, e, f = top
    return tv(a, b, c, d), tv(b, c, d, e), tv(c, d, e, f)


def wing_rans_surface(s_sw, span_stride=2, chord_stride=2):
    """Wall triangulation (SU2 frame) for the RANS mesh: every `span_stride`-th span row plus the kink
    and tip rows, every `chord_stride`-th chordwise node (LE kept); a sharp TE (first node == last)
    is merged into one node; a flat tip cap closes the tip. Returns points (N, 3), triangles (outward),
    ring size, rows used, the index of the root ring and the cap triangles."""
    s_sw = np.asarray(s_sw, float)
    ns, nc = s_sw.shape[1], s_sw.shape[2]
    rows, kink = loft_rows(s_sw, span_stride)
    if np.linalg.norm(s_sw[:, 0, 0] - s_sw[:, 0, -1]) > 1e-9:
        raise ValueError("RANS mesher expects a sharp trailing edge (origingeom.npy convention)")
    cols = np.arange(0, nc - 1, chord_stride)
    if (nc // 2) not in cols:
        raise ValueError("chord stride must keep the LE node")
    m = len(cols)
    P = np.stack([sw_to_su2(s_sw[:, r, cols].T) for r in rows])        # (R, m, 3)
    R = len(rows)
    idx = np.arange(R * m).reshape(R, m)
    nx = np.roll(idx, -1, axis=1)
    T = np.concatenate([np.stack([idx[:-1].ravel(), nx[:-1].ravel(), nx[1:].ravel()], 1),
                        np.stack([idx[:-1].ravel(), nx[1:].ravel(), idx[1:].ravel()], 1)])
    # tip cap: strips between the lower and upper nodes of the tip ring (ring index 0 = TE)
    tip = idx[-1]
    le = m // 2
    k = np.arange(1, le)
    lo, up = tip[k], tip[m - k]
    cap = [np.array([[tip[0], tip[1], tip[m - 1]]])]
    cap.append(np.stack([lo[:-1], lo[1:], up[1:]], 1))
    cap.append(np.stack([lo[:-1], up[1:], up[:-1]], 1))
    cap.append(np.array([[tip[le - 1], tip[le], tip[le + 1]]]))
    cap = np.concatenate(cap)
    pts = P.reshape(-1, 3)
    # orientation: the open root ring is planar, so with the origin on the root plane the missing
    # root cap adds no volume and the signed volume of wall + tip cap decides (outward = positive)
    vol = _signed_volume_tris(pts - pts[idx[0, 0]], np.concatenate([T, cap]))
    if vol < 0:
        T, cap = T[:, ::-1].copy(), cap[:, ::-1].copy()
    return dict(points=pts, tris=np.concatenate([T, cap]), n_wall_tris=int(len(T)), ring=m, rows=rows.tolist(),
                kink_row=int(kink), root_ring=idx[0].copy(), cap_tris=int(len(cap)))


def _signed_volume_tris(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def extrude_prism_layers(points, tris, root_nodes, heights, local_scale, span_axis=1, smooth=30, min_fraction=0.6):
    """March the wall outward in len(heights) layers along smoothed vertex normals (recomputed every
    layer); node i moves min(heights[k] * local_scale[i], half its shortest adjacent edge) in layer k. Root-ring nodes keep their span
    coordinate (they stay on the symmetry plane). Every prism must have six positive corner
    Jacobians; nodes of a failing prism get their step halved (up to 8 times). If a layer still
    fails, marching stops there (the returned array has fewer layers) unless fewer than
    `min_fraction` of the planned layers were built, which raises."""
    X = [np.asarray(points, float)]
    root = np.asarray(root_nodes)
    T = np.asarray(tris)
    for k, h in enumerate(heights):
        F = _face_normals(X[-1], T)
        Nv = _visible_normals(_vertex_normals(X[-1], T), F, T)
        Nv[root, span_axis] = 0.0
        Nv /= np.linalg.norm(Nv, axis=1, keepdims=True)
        N = _smooth_normals(Nv, T, smooth, root, span_axis)
        md_v, md_s = _min_dot(Nv, F, T), _min_dot(N, F, T)
        keep = md_s < 0.5 * np.minimum(md_v, 0.25)
        N[keep] = Nv[keep]
        # aspect-ratio limiter: a layer is never taller than half the shortest adjacent surface edge
        E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        el = np.linalg.norm(X[-1][E[:, 0]] - X[-1][E[:, 1]], axis=1)
        emin = np.full(len(N), np.inf)
        np.minimum.at(emin, E[:, 0], el)
        np.minimum.at(emin, E[:, 1], el)
        step = np.minimum(h * np.asarray(local_scale), 0.5 * emin)
        for _ in range(8):          # local repair: halve the step of nodes of an inverted prism
            Xn = X[-1] + step[:, None] * N
            Xn[root, span_axis] = X[0][root, span_axis]
            J = prism_corner_jacobians((X[-1][T[:, 0]], X[-1][T[:, 1]], X[-1][T[:, 2]]),
                                       (Xn[T[:, 0]], Xn[T[:, 1]], Xn[T[:, 2]]))
            badt = np.min(np.stack(J), axis=0) <= 0
            if not np.any(badt):
                break
            step[np.unique(T[badt])] *= 0.5
        if np.any(badt):
            if k < min_fraction * len(heights):
                raise RuntimeError(f"layer {k + 1}: {int(badt.sum())} non-positive prisms after local step reduction")
            break               # keep the k valid layers; the tetra region starts there
        X.append(Xn)
    return np.stack(X)          # (L + 1, N, 3), L <= len(heights)


def annulus_and_dome(inner_loop_xyz, plane_y, centre_xz, radius, n_radial, h_first, n_rings, near_dist=0.05):
    """Boundary triangles closing the tetra region: (1) the symmetry-plane annulus between the outer
    root loop of the prism layer and a circle of `radius` (O-grid in the plane, ogrid_from_loop), and
    (2) a farfield half-sphere on the wing side (y < plane_y) sharing the circle's nodes. Triangles
    are oriented out of the tetra region (+y on the annulus, radially outward on the dome).
    Returns points, annulus triangles, dome triangles; the first len(loop) points are the loop."""
    L = np.asarray(inner_loop_xyz, float)
    m = len(L)
    xz = L[:, [0, 2]]
    a = 0.5 * np.sum(xz[:, 0] * np.roll(xz[:, 1], -1) - np.roll(xz[:, 0], -1) * xz[:, 1])
    order = np.arange(m) if a < 0 else np.r_[0, np.arange(m - 1, 0, -1)]
    X, Z = ogrid_from_loop(xz[order], n_radial, h_first, radius, centre_xz, near_dist)
    inv = np.empty(m, np.int64)
    inv[order] = np.arange(m)
    X, Z = X[:, inv], Z[:, inv]                       # back to the loop's own node order
    pts = [np.stack([X.ravel(), np.full(X.size, plane_y), Z.ravel()], 1)]
    idx = np.arange((n_radial + 1) * m).reshape(n_radial + 1, m)
    nx = np.roll(idx, -1, axis=1)
    ann = np.concatenate([np.stack([idx[:-1].ravel(), nx[:-1].ravel(), nx[1:].ravel()], 1),
                          np.stack([idx[:-1].ravel(), nx[1:].ravel(), idx[1:].ravel()], 1)])
    c = np.array([centre_xz[0], plane_y, centre_xz[1]])
    th = np.arctan2(Z[-1] - c[2], X[-1] - c[0])
    base = idx[-1]
    nxt0 = (n_radial + 1) * m
    dome, prev = [], base
    for ph in np.linspace(0, np.pi / 2, n_rings + 1)[1:-1]:
        ring = np.stack([c[0] + radius * np.cos(ph) * np.cos(th), np.full(m, plane_y - radius * np.sin(ph)),
                         c[2] + radius * np.cos(ph) * np.sin(th)], 1)
        ids = nxt0 + np.arange(m)
        nxt0 += m
        pts.append(ring)
        dome += [np.stack([prev, np.roll(prev, -1), np.roll(ids, -1)], 1), np.stack([prev, np.roll(ids, -1), ids], 1)]
        prev = ids
    pole = nxt0
    pts.append(np.array([[c[0], plane_y - radius, c[2]]]))
    dome.append(np.stack([prev, np.roll(prev, -1), np.full(m, pole)], 1))
    P = np.concatenate(pts)
    dome = np.concatenate(dome)

    def orient(tri, outward):
        n = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
        flip = np.einsum("ij,ij->i", n, outward(tri)) < 0
        tri = tri.copy()
        tri[flip] = tri[flip][:, ::-1]
        return tri
    ann = orient(ann, lambda t: np.tile([0.0, 1.0, 0.0], (len(t), 1)))
    dome = orient(dome, lambda t: P[t].mean(axis=1) - c)
    return P, ann, dome


def _merge_gmsh_nodes(bnd_pts, used, gtags, gcoords, tol=1e-10):
    """Point table and gmsh-tag -> index lookup after tetra meshing. Boundary nodes are matched by
    coordinates (gmsh may renumber), so they keep their own indices; nodes gmsh created are appended.
    Raises if a boundary node handed to gmsh is not found again."""
    from scipy.spatial import cKDTree
    d, j = cKDTree(bnd_pts[used]).query(gcoords)
    is_b = d <= tol * max(1.0, float(np.abs(bnd_pts).max()))
    if is_b.sum() != len(used):
        raise RuntimeError(f"gmsh returned {int(is_b.sum())} of {len(used)} boundary nodes")
    new = np.where(~is_b)[0]
    allp = np.concatenate([bnd_pts, gcoords[new]])
    lut = np.full(int(gtags.max()) + 1, -1, np.int64)
    lut[gtags[is_b]] = used[j[is_b]]
    lut[gtags[new]] = len(bnd_pts) + np.arange(len(new))
    return allp, lut


def wing_mesh_3d_rans(s_sw, out_path, reynolds, yplus=1.0, growth=1.2, last_layer_chords=0.01,
                      span_stride=2, chord_stride=2, radius=None, n_radial=40, n_rings=16, log=None) -> dict:
    """Hybrid RANS mesh: prism layer extruded from the wall (first cell from the flat-plate y+
    estimate at the local chord Reynolds number, geometric growth until the layer height reaches
    `last_layer_chords` x local chord), then gmsh tetrahedra between the prism layer, the
    symmetry-plane annulus and a farfield half-sphere. Markers: wing, symmetry, farfield."""
    import gmsh
    radius = float(_CFD["superwing_holdout"]["farfield_radius_root_chords"] if radius is None else radius)
    t0 = time.time()
    w = wing_rans_surface(s_sw, span_stride, chord_stride)
    P, T, m = w["points"], w["tris"], w["ring"]
    R = len(w["rows"])
    # local chord per node (ring-wise): streamwise extent of its span row
    xr = P[:, 0].reshape(R, m)
    chord = np.repeat(xr.max(axis=1) - xr.min(axis=1), m)
    h1 = np.array([first_cell_height(reynolds * c, yplus) * c for c in chord])
    n_layers = int(np.ceil(np.log(last_layer_chords / (h1 / chord).max()) / np.log(growth))) + 1
    heights = growth ** np.arange(n_layers)            # times h1 (per node)
    X = extrude_prism_layers(P, T, w["root_ring"], heights, h1, span_axis=1)
    L = len(X) - 1
    nwall = len(P)
    plane_y = float(P[w["root_ring"][0], 1])
    outer = X[-1]
    loop = outer[w["root_ring"]]
    xr0 = P[w["root_ring"], 0]
    centre = (0.5 * (xr0.min() + xr0.max()), float(P[w["root_ring"], 2].mean()))
    last_h = float(np.median(np.linalg.norm(X[-1] - X[-2], axis=1)))
    ap, ann, dome = annulus_and_dome(loop, plane_y, centre, radius, n_radial, last_h, n_rings)
    # tetra-region boundary: outer prism surface + annulus + dome (shared root loop nodes)
    n_prism_nodes = (L + 1) * nwall
    outer_ids = L * nwall + np.arange(nwall)
    ap_ids = np.concatenate([outer_ids[w["root_ring"]], n_prism_nodes + np.arange(len(ap) - m)])
    bnd_pts = np.concatenate([X.reshape(-1, 3), ap[m:]])
    f_outer = outer_ids[T][:, ::-1]                      # tetra region lies outside: flip
    f_ann, f_dome = ap_ids[ann], ap_ids[dome]
    closed = np.concatenate([f_outer, f_ann, f_dome])
    from . import geometry as _geom
    wt = _geom.check_watertight(bnd_pts, closed)
    if wt["n_open_edges"] or wt["n_nonmanifold_edges"] or not wt["orientation_consistent"]:
        raise RuntimeError(f"tetra-region boundary not closed or not consistently oriented: {wt}")
    used = np.unique(closed)
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 1 if log else 0)
        gmsh.model.add("rans")
        gmsh.model.addDiscreteEntity(2, 1)
        gmsh.model.mesh.addNodes(2, 1, (used + 1).tolist(), bnd_pts[used].ravel().tolist())
        gmsh.model.mesh.addElementsByType(1, 2, [], (closed + 1).ravel().tolist())
        sl = gmsh.model.geo.addSurfaceLoop([1])
        gmsh.model.geo.addVolume([sl], 1)
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", radius / 8)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Renumber", 0)
        gmsh.model.mesh.generate(3)
        etypes, etags, enodes = gmsh.model.mesh.getElements(3, 1)
        if list(etypes) != [4]:
            raise RuntimeError(f"unexpected volume element types {etypes}")
        tets = np.asarray(enodes[0], np.int64).reshape(-1, 4) - 1
        ntag, ncoord, _ = gmsh.model.mesh.getNodes()
        gversion = gmsh.__version__
    finally:
        gmsh.finalize()
    ncoord = np.asarray(ncoord).reshape(-1, 3)
    allp, lut = _merge_gmsh_nodes(bnd_pts, used, np.asarray(ntag, np.int64), ncoord)
    n_all = len(allp)
    tets = lut[tets + 1]                              # gmsh tags -> point indices
    if np.any(tets < 0):
        raise RuntimeError("tetrahedra reference nodes gmsh did not return")
    tv = np.einsum("ij,ij->i", np.cross(allp[tets[:, 1]] - allp[tets[:, 0]], allp[tets[:, 2]] - allp[tets[:, 0]]),
                   allp[tets[:, 3]] - allp[tets[:, 0]])
    tets[tv < 0] = tets[tv < 0][:, [0, 2, 1, 3]]
    # prisms: layer k bottom triangle -> layer k+1 top triangle
    pr = np.concatenate([np.concatenate([k * nwall + T, (k + 1) * nwall + T], 1) for k in range(L)])
    # symmetry quads: prism side faces on root-ring edges (ring is closed)
    rr = w["root_ring"]
    e0, e1 = rr, np.roll(rr, -1)
    sq = np.concatenate([np.stack([k * nwall + e0, k * nwall + e1, (k + 1) * nwall + e1, (k + 1) * nwall + e0], 1)
                         for k in range(L)])
    mk = {"wing": {VTK_TRI: T}, "symmetry": {VTK_TRI: f_ann, VTK_QUAD: sq}, "farfield": {VTK_TRI: f_dome}}
    write_su2_mesh(out_path, allp, {VTK_PRISM: pr, VTK_TET: tets}, mk)
    return dict(n_cells=int(len(pr) + len(tets)), n_prisms=int(len(pr)), n_tets=int(len(tets)), n_points=int(n_all),
                n_layers=L, n_layers_planned=n_layers, growth=growth, h1_root=float(h1[w["root_ring"]].mean()), h1_min=float(h1.min()),
                prism_layer_thickness_root=float(np.linalg.norm(X[-1][w["root_ring"]] - X[0][w["root_ring"]], axis=1).mean()),
                n_wall_tris=int(len(T)), ring_nodes=m, span_rows=w["rows"], reynolds=reynolds, yplus_target=yplus,
                radius=radius, mesh_seconds=time.time() - t0, gmsh_version=gversion,
                tet_min_volume=float(np.abs(tv).min() / 6.0))
