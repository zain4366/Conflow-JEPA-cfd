"""SuperWing geometry builder: 38 shape parameters -> 129 x 257 reference surface (span x chord).

The construction follows the reference-mesh definition published by the SuperWing authors in
`cfdpost` (https://github.com/YangYunjia/cfdpost, MIT License, Copyright (c) 2020 Runze Li):
`cfdpost.wing.multi_section_wing.MultiSecWing._reconstruct_surface_grids` and
`cfdpost.modeling.cst_foil / cst_curve / dist_clustcos`. This module is a vectorised
re-implementation of those equations. Two points differ from the public code, which raises an
error for the 38-parameter SuperWing format (checked 21 Oct 2026):
  1. the twist spline has 5 control points but SuperWing gives 4 twists; a root twist of 0 deg is
     prepended (twist is zero at the root, SuperWing paper Sec. 2);
  2. the public `rtcs` branch stacks 128 lower + 128 upper points (256) against 257 x-values;
     here the lower surface without its LE point is stacked with the full upper surface (257),
     as in the public code's other branch.
Whether this reproduces `origingeom.npy` is Gate G2 (tests/test_t1_geometry.py).

Coordinates: channel 0 = x (streamwise), 1 = y (vertical), 2 = z (spanwise), lengths in units of
the trapezoidal root chord (chord = 1 at the symmetry plane of the trapezoidal planform).
Chordwise index 0 = lower trailing edge, 128 = leading edge, 256 = upper trailing edge.
Wing-box and STL topology helpers: the edge-count / Euler check and the binary STL writer are
adapted from the old task10_3d_wing_lofting_and_cad_export.py (genuine part, PLAN section 1).
"""
import struct
from math import comb
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from . import data as _data
from .spec import load_spec

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # NumPy 2 name, NumPy 1 fallback

_SPEC = load_spec()
_B = _SPEC["geometry"]["builder"]
_BOX = _SPEC["geometry"]["wing_box"]

N_CHORD_SIDE = int(_B["chord_points_per_surface"])  # 129 -> 257 points around the section
N_SPAN = int(_SPEC["geometry"]["surface_nodes"][0])   # 129
TAIL = float(_B["te_thickness"])
CABIN = float(_B["fuselage_fraction"])
ROOT_TWIST = float(_B["root_twist_deg"])
N1, N2 = float(_B["cst_class_n1"]), float(_B["cst_class_n2"])
A0, A1 = float(_B["clustcos_a0"]), float(_B["clustcos_a1"])


# ----------------------------------------------------------------------------- sections

def dist_clustcos(nn: int, a0: float = A0, a1: float = A1) -> np.ndarray:
    aa = (1 - np.cos(a0 * np.pi)) / 2.0
    dd = (1 - np.cos(a1 * np.pi)) / 2.0 - aa
    yt = np.linspace(0.0, 1.0, nn)
    a = np.pi * (a0 * (1 - yt) + a1 * yt)
    return ((1 - np.cos(a)) / 2.0 - aa) / dd


def cst_curve(x: np.ndarray, coef) -> np.ndarray:
    coef = np.asarray(coef, dtype=float)
    n = len(coef) - 1
    s = sum(coef[i] * comb(n, i) * x ** i * (1 - x) ** (n - i) for i in range(n + 1))
    y = x ** N1 * (1 - x) ** N2 * s
    y[0] = 0.0
    y[-1] = 0.0
    return y


def cst_foil(nn: int, cst_u, cst_l, t=None, tail: float = 0.0):
    """Upper/lower CST curves on the clustcos distribution; optional max-thickness scaling and
    linear trailing-edge thickness `tail` (as cfdpost.modeling.cst_foil)."""
    x = dist_clustcos(nn)
    yu, yl = cst_curve(x, cst_u), cst_curve(x, cst_l)
    thick = yu - yl
    it = int(np.argmax(thick))
    t0 = thick[it]
    if t is not None:
        r = (t - tail * x[it]) / t0
        t0 = t
        yu, yl = yu * r, yl * r
    yu = yu + 0.5 * tail * x
    yl = yl - 0.5 * tail * x
    if t is None:
        t0 = (yu - yl).max()
    return x, yu, yl, t0


# ----------------------------------------------------------------------------- parameters

def as_dict(p) -> dict:
    p = np.asarray(p, dtype=float)
    if p.shape != (38,):
        raise ValueError(f"expected 38 parameters, got shape {p.shape}")
    n = _data.GEOM_NAMES
    g = {k: float(p[n.index(k)]) for k in ("SA", "DA", "DA_kink", "AR", "TR", "kink", "rootadj", "tcroot")}
    g["rtcs"] = p[n.index("rtcs_0"):n.index("rtcs_2") + 1]
    g["cambers"] = p[n.index("cambers_0"):n.index("cambers_2") + 1]
    g["twists"] = p[n.index("twists_0"):n.index("twists_3") + 1]
    g["cst_u"] = p[n.index("cst_u_0"):n.index("cst_u_9") + 1]
    g["cst_l"] = p[n.index("cst_l_0"):n.index("cst_l_9") + 1]
    return g


def half_span(g: dict) -> float:
    return 0.25 * g["AR"] * (1 + g["TR"])


def _stations(g: dict):
    hs = half_span(g)
    kink = hs * g["kink"]
    return dict(hs=hs, cabin=hs * CABIN, kink=kink, mid_inner=0.5 * kink, mid_outer=0.5 * (kink + hs), tip=hs + 1e-9)


def spanwise_laws(g: dict, y: np.ndarray, i_kink_grid: np.ndarray = None) -> dict:
    """Leading-edge x and vertical position, chord, thickness ratio, camber ratio and twist (deg)
    at spanwise stations y (absolute, from the symmetry plane)."""
    st = _stations(g)
    hs, cab, kink, tip = st["hs"], st["cabin"], st["kink"], st["tip"]
    tsa = np.tan(np.radians(g["SA"]))
    xle = tsa * y
    xle_kink = tsa * kink

    # dihedral: linear inboard of the kink grid index, cubic spline outboard (cfdpost logic)
    sl = np.tan(np.radians(g["DA_kink"]))
    z_kink = sl * kink
    z_tip = np.tan(np.radians(g["DA"])) * hs
    cs = CubicSpline([kink, tip], [z_kink, z_tip], bc_type=((1, sl), "not-a-knot"), extrapolate=False)
    inner = y <= kink if i_kink_grid is None else i_kink_grid
    zle = np.where(inner, sl * y, cs(np.where(inner, kink, y)))

    tr = g["TR"]
    c_tip = tr
    c_kink = tr * g["kink"] + (1 - g["kink"])
    root_adj = g["rootadj"] * (xle_kink + c_kink - 1)
    c_cab = tr * CABIN + (1 - CABIN) + root_adj * (g["kink"] - CABIN) / g["kink"]
    chord = np.where(y < kink, c_cab + (c_kink - c_cab) * (y - cab) / (kink - cab),
                     c_kink + (c_tip - c_kink) * (y - kink) / (hs - kink))

    rt = np.cumprod(g["rtcs"])
    trs = CubicSpline([0, kink, st["mid_outer"], tip], [1.0, *rt], extrapolate=False)(y)
    c = g["cambers"]
    cmb = CubicSpline([0, st["mid_inner"], kink, st["mid_outer"], tip],
                      [0.0, c[0] * c[1], c[1], 1.0, c[2]], extrapolate=False)(y)
    tw = np.cumsum(np.concatenate([[ROOT_TWIST], g["twists"]]))
    tws = CubicSpline([0, st["mid_inner"], kink, st["mid_outer"], tip], tw, extrapolate=True)(y)
    out = dict(xle=xle, zle=zle, chord=chord, trs=trs, cmb=cmb, twist=tws)
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite spanwise law '{k}'")
    return out


def _root_foil(g: dict):
    x, yu, yl, _ = cst_foil(N_CHORD_SIDE, g["cst_u"], g["cst_l"], t=g["tcroot"], tail=TAIL)
    return x, 0.5 * (yu + yl), yu - yl


def build_surface(params, n_span: int = N_SPAN) -> np.ndarray:
    """Reference surface nodes, shape (3, n_span, 257): channels x, y (vertical), z (span)."""
    g = as_dict(params)
    st = _stations(g)
    y = np.linspace(st["cabin"], st["hs"], n_span)
    i_kink = int(np.floor((g["kink"] - CABIN) / (1 - CABIN) * n_span))
    laws = spanwise_laws(g, y, i_kink_grid=np.arange(n_span) <= i_kink)
    x1, cmb1, tc1 = _root_foil(g)
    yu = np.outer(cmb1, laws["cmb"]) + 0.5 * np.outer(tc1, laws["trs"])
    yl = np.outer(cmb1, laws["cmb"]) - 0.5 * np.outer(tc1, laws["trs"])
    zz = np.concatenate([yl[1:][::-1], yu], axis=0)          # (257, n_span)
    xx = np.concatenate([x1[1:][::-1], x1])                   # (257,)
    tw = np.radians(laws["twist"])
    xr = np.cos(tw)[None, :] * xx[:, None] + np.sin(tw)[None, :] * zz
    zr = -np.sin(tw)[None, :] * xx[:, None] + np.cos(tw)[None, :] * zz
    X = xr * laws["chord"][None, :] + laws["xle"][None, :]
    Y = zr * laws["chord"][None, :] + laws["zle"][None, :]
    Z = np.repeat(y[None, :], len(xx), axis=0)
    return np.stack([X, Y, Z], axis=0).transpose(0, 2, 1)


def control_sections(g: dict) -> dict:
    """Untwisted local sections (chord units) at root (exposed), kink, mid-outer and tip."""
    st = _stations(g)
    ys = {"root": st["cabin"], "kink": st["kink"], "mid_outer": st["mid_outer"], "tip": st["hs"]}
    yarr = np.array(list(ys.values()))
    laws = spanwise_laws(g, yarr)
    x1, cmb1, tc1 = _root_foil(g)
    out = {}
    for i, k in enumerate(ys):
        yu = cmb1 * laws["cmb"][i] + 0.5 * tc1 * laws["trs"][i]
        yl = cmb1 * laws["cmb"][i] - 0.5 * tc1 * laws["trs"][i]
        out[k] = dict(y=float(yarr[i]), chord=float(laws["chord"][i]), x=x1, yu=yu, yl=yl,
                      t_over_c=float((yu - yl).max()), twist_deg=float(laws["twist"][i]))
    return out


# ----------------------------------------------------------------------------- wing box

def box_ixx(x, yu, yl, chord, x_front=_BOX["front_spar_xc"], x_rear=_BOX["rear_spar_xc"], n=2001) -> float:
    """Second moment of area (about the horizontal centroidal axis) of the solid section between
    the spars, in root-chord^4 units: I = int (yu^3 - yl^3)/3 dx - A * ybar^2, scaled by chord^4."""
    xs = np.linspace(x_front, x_rear, n)
    u, l = np.interp(xs, x, yu), np.interp(xs, x, yl)
    a = _trapezoid(u - l, xs)
    ybar = _trapezoid(0.5 * (u ** 2 - l ** 2), xs) / a
    i0 = _trapezoid((u ** 3 - l ** 3) / 3.0, xs)
    return float((i0 - a * ybar ** 2) * chord ** 4)


def wing_box_ixx(g: dict) -> dict:
    secs = control_sections(g)
    return {k: box_ixx(s["x"], s["yu"], s["yl"], s["chord"]) for k, s in secs.items()}


# ----------------------------------------------------------------------------- closed mesh / STL

def close_surface(s: np.ndarray):
    """Triangulate the open reference surface and close the trailing-edge gap, root and tip.
    Returns vertices (V, 3) and faces (F, 3) with outward orientation."""
    _, ns, nc = s.shape
    if nc % 2 == 0:
        raise ValueError("chordwise node count must be odd (shared leading-edge node)")
    v = s.reshape(3, -1).T.copy()
    idx = np.arange(ns * nc).reshape(ns, nc)
    faces = []
    j = np.arange(nc)
    jn = (j + 1) % nc  # j = nc-1 -> 0 closes the trailing-edge gap
    for i in range(ns - 1):
        a, b, c, d = idx[i, j], idx[i, jn], idx[i + 1, jn], idx[i + 1, j]
        faces.append(np.stack([a, b, c], 1))
        faces.append(np.stack([a, c, d], 1))
    le = nc // 2
    k = np.arange(le - 1)
    for i, flip in ((0, True), (ns - 1, False)):
        lo, up = idx[i, k], idx[i, nc - 1 - k]
        lo1, up1 = idx[i, k + 1], idx[i, nc - 2 - k]
        cap = [np.stack([lo, lo1, up1], 1), np.stack([lo, up1, up], 1),
               np.array([[idx[i, le - 1], idx[i, le], idx[i, le + 1]]])]
        cap = np.concatenate(cap)
        faces.append(cap[:, ::-1] if flip else cap)
    f = np.concatenate(faces).astype(np.int64)
    if _signed_volume(v, f) < 0:
        f = f[:, ::-1].copy()
    return v, f


def _signed_volume(v, f) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def check_watertight(v: np.ndarray, f: np.ndarray) -> dict:
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    und = np.sort(e, axis=1)
    uniq, cnt = np.unique(und, axis=0, return_counts=True)
    dir_uniq, dir_cnt = np.unique(e, axis=0, return_counts=True)
    rev = np.unique(e[:, ::-1], axis=0)
    paired = len(np.intersect1d(dir_uniq.view([("a", e.dtype), ("b", e.dtype)]),
                                rev.view([("a", e.dtype), ("b", e.dtype)])))
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n_v = len(np.unique(f))
    return dict(
        n_vertices=int(n_v), n_edges=int(len(uniq)), n_faces=int(len(f)),
        n_open_edges=int(np.sum(cnt == 1)), n_nonmanifold_edges=int(np.sum(cnt > 2)),
        euler_characteristic=int(n_v - len(uniq) + len(f)),
        orientation_consistent=bool(np.all(dir_cnt == 1) and paired == len(dir_uniq)),
        n_degenerate_faces=int(np.sum(area < 1e-14)),
        signed_volume=_signed_volume(v, f),
    )


def write_binary_stl(path, v: np.ndarray, f: np.ndarray, header: bytes = b"ConFlow-JEPA SuperWing builder") -> int:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    rec = np.zeros(len(f), dtype=[("n", "<f4", (3,)), ("v0", "<f4", (3,)), ("v1", "<f4", (3,)),
                                  ("v2", "<f4", (3,)), ("attr", "<u2")])
    rec["n"], rec["v0"], rec["v1"], rec["v2"] = n, a, b, c
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        fh.write(header[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())
    tmp.replace(path)
    return path.stat().st_size


def read_binary_stl(path):
    with open(path, "rb") as fh:
        fh.read(80)
        n = struct.unpack("<I", fh.read(4))[0]
        rec = np.frombuffer(fh.read(), dtype=[("n", "<f4", (3,)), ("v0", "<f4", (3,)), ("v1", "<f4", (3,)),
                                               ("v2", "<f4", (3,)), ("attr", "<u2")], count=n)
    pts = np.concatenate([rec["v0"], rec["v1"], rec["v2"]]).astype(np.float64)
    uniq, inv = np.unique(pts, axis=0, return_inverse=True)
    f = inv.reshape(3, n).T
    return uniq, f


# ----------------------------------------------------------------------------- Gate G2

def surface_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMS of the point-to-point Euclidean distance (root-chord units)."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=0))))


def g2_check(cfg, origingeom_src, n_wings: int, pool: int, seed: int) -> dict:
    """Builder vs origingeom.npy on n_wings random rows among the first `pool` rows.
    Row r of origingeom.npy is assumed to be row r of configs.dat; diagnostics test that too."""
    pool = min(pool, origingeom_src.shape[0])
    rows = np.sort(np.random.default_rng(seed).choice(pool, size=n_wings, replace=False))
    per = []
    for r in rows:
        ref = origingeom_src.read(int(r), int(r) + 1)[0].astype(np.float64)
        b = build_surface(cfg.params[r])
        if ref.shape != b.shape:
            raise ValueError(f"origingeom row shape {ref.shape} != builder shape {b.shape}")
        d = b - ref
        per.append(dict(row=int(r), wing_id=int(cfg.wing_id[r]), rmse=surface_rmse(b, ref),
                        max_abs=float(np.abs(d).max()),
                        rmse_xyz=[float(np.sqrt(np.mean(d[c] ** 2))) for c in range(3)],
                        rmse_le_region=float(np.sqrt(np.mean(np.sum(d[:, :, 96:161] ** 2, axis=0)))),
                        rmse_te_region=float(np.sqrt(np.mean(np.sum(d[:, :, list(range(0, 32)) + list(range(225, 257))] ** 2, axis=0)))),
                        rmse_root_half=float(np.sqrt(np.mean(np.sum(d[:, :64] ** 2, axis=0)))),
                        rmse_tip_half=float(np.sqrt(np.mean(np.sum(d[:, 64:] ** 2, axis=0))))))
    rm = np.array([p["rmse"] for p in per])
    out = dict(rows=rows.tolist(), n_wings=int(n_wings), pool=int(pool), seed=int(seed),
               rmse_mean=float(rm.mean()), rmse_median=float(np.median(rm)), rmse_max=float(rm.max()),
               max_abs_max=float(max(p["max_abs"] for p in per)), per_wing=per)
    worst = per[int(np.argmax(rm))]
    ref = origingeom_src.read(worst["row"], worst["row"] + 1)[0].astype(np.float64)
    b = build_surface(cfg.params[worst["row"]])
    out["diag_worst_channel_swap_yz_rmse"] = surface_rmse(b[[0, 2, 1]], ref)
    out["diag_worst_neighbour_rows_rmse"] = {
        str(r2): surface_rmse(build_surface(cfg.params[r2]), ref)
        for r2 in range(max(0, worst["row"] - 2), min(len(cfg.params), worst["row"] + 3))}
    return out


# ----------------------------------------------------------------------------- G2 diagnostics (report only)

def _section_frame(sec):
    """sec: (3, 257) one spanwise station. LE = index 128, TE = mid-gap of indices 0 and 256."""
    te = 0.5 * (sec[:2, 0] + sec[:2, -1])
    le = sec[:2, int(np.argmax(np.hypot(sec[0] - te[0], sec[1] - te[1])))]  # farthest point from TE
    d = te - le
    chord = float(np.hypot(*d))
    twist = float(np.degrees(np.arctan2(-d[1], d[0])))
    return le, te, chord, twist


def _point_to_polyline(p, q, n_dense=20):
    """RMS distance from points p (2, n) to the polyline through q (2, m), densified n_dense x."""
    from scipy.spatial import cKDTree
    t = np.linspace(0, 1, n_dense, endpoint=False)
    dense = (q[:, :-1, None] * (1 - t) + q[:, 1:, None] * t).reshape(2, -1)
    dense = np.concatenate([dense, q[:, -1:]], axis=1)
    dist, _ = cKDTree(dense.T).query(p.T)
    return dist


def g2_diagnostics(cfg, origingeom_src, rows) -> dict:
    """Separates point-distribution effects from shape effects for the G2 wings. Report only:
    the gate itself is decided by g2_check (point-to-point RMSE)."""
    per = []
    for r in rows:
        ref = origingeom_src.read(int(r), int(r) + 1)[0].astype(np.float64)
        b = build_surface(cfg.params[r])
        shape_d, dle, dchord, dtw, dxc, dtc = [], [], [], [], [], []
        for i in range(b.shape[1]):
            sb, sr = b[:, i], ref[:, i]
            shape_d.append(_point_to_polyline(sb[:2], sr[:2]))
            leb, teb, cb, twb = _section_frame(sb)
            ler, ter, cr, twr = _section_frame(sr)
            dle.append(leb - ler)
            dchord.append(cb - cr)
            dtw.append(twb - twr)
            xcb = (sb[0, N_CHORD_SIDE - 1:] - leb[0]) / (teb[0] - leb[0])
            xcr = (sr[0, N_CHORD_SIDE - 1:] - ler[0]) / (ter[0] - ler[0])
            dxc.append(np.abs(xcb - xcr).max())
            tb = (sb[1, N_CHORD_SIDE - 1:] - sb[1, N_CHORD_SIDE - 1::-1]).max() / cb
            tr = (sr[1, N_CHORD_SIDE - 1:] - sr[1, N_CHORD_SIDE - 1::-1]).max() / cr
            dtc.append(tb - tr)
        shape_d = np.concatenate(shape_d)
        dle, dchord, dtw = np.array(dle), np.array(dchord), np.array(dtw)
        half = b.shape[1] // 2
        per.append(dict(
            row=int(r), point_rmse=surface_rmse(b, ref),
            shape_rmse_xy=float(np.sqrt(np.mean(shape_d ** 2))), shape_max_xy=float(shape_d.max()),
            le_dx_max=float(np.abs(dle[:, 0]).max()), le_dy_max=float(np.abs(dle[:, 1]).max()),
            chord_diff_max_root_half=float(np.abs(dchord[:half]).max()),
            chord_diff_max_tip_half=float(np.abs(dchord[half:]).max()),
            chord_diff_at_root=float(dchord[0]), chord_ref_at_root=float(_section_frame(ref[:, 0])[2]),
            twist_diff_max_deg=float(np.abs(dtw).max()),
            xc_distribution_diff_max=float(max(dxc)), tc_diff_max=float(np.abs(dtc).max()),
            rootadj=float(as_dict(cfg.params[r])["rootadj"])))
    keys = [k for k in per[0] if k not in ("row",)]
    summ = {k: dict(median=float(np.median([p[k] for p in per])), max=float(np.max([p[k] for p in per])))
            for k in keys}
    ra = np.array([p["rootadj"] for p in per])
    cdr = np.array([p["chord_diff_at_root"] for p in per])
    summ["corr_rootadj_vs_root_chord_diff"] = (float(np.corrcoef(ra, cdr)[0, 1])
                                               if ra.std() > 0 and cdr.std() > 0 else None)
    return dict(n_wings=len(per), summary=summ, per_wing=per)
