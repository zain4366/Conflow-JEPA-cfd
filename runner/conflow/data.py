"""SuperWing readers, wing-level splits and the memory-mapped Cp / geometry reduction.

File layout (Hugging Face yunplus/SuperWing, checked 21 Oct 2026):
  configs.dat    header line "VARIABLES = ..." then one line per wing: wing id, 38 shape
                 parameters, 8 Mach numbers (M30..M37), 8 angles of attack (AA0..AA7)
  index.npy      (28856, 12) float64, one row per case (columns in INDEX_COLUMNS);
                 column 0 is the configs.dat ROW number (0..4238), checked on Colab 22 Oct 2026
  train/test.parquet  same columns, but column 0 (wing_shape_idx) is the configs.dat wing ID
                 (column 0 of configs.dat, 0..5871); rows of both match after this mapping
  data.npy       (28856, 3, 128, 256) float64; channel 0 = Cp (scale 1)
  geom0.npy      (4239, 3, 128, 256) float64 cell centres
  origingeom.npy (4239, 3, 129, 257) float64 nodes; row r <-> configs.dat row r
  train/test.parquet  the index.npy columns for the official wing-level split
"""
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GEOM_NAMES = (["SA", "DA", "DA_kink", "AR", "TR", "kink", "rootadj"]
              + [f"rtcs_{i}" for i in range(3)] + [f"cambers_{i}" for i in range(3)]
              + [f"twists_{i}" for i in range(4)] + [f"cst_u_{i}" for i in range(10)]
              + [f"cst_l_{i}" for i in range(10)] + ["tcroot"])

INDEX_COLUMNS = ["wing_row", "cond_idx", "aoa", "mach", "s_ref_half", "half_span",
                 "cl_solver", "cd_solver", "cm_solver", "cl_surface", "cd_surface", "cm_surface"]
PARQUET_COLUMNS = ["wing_shape_idx", "condition_idx", "aoa", "mach", "ref_area", "half_span",
                   "cl_solver", "cd_solver", "cm_solver", "cl_surface", "cd_surface", "cm_surface"]

SPLIT_FIT, SPLIT_CALIB, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_FIT: "fit", SPLIT_CALIB: "calib", SPLIT_TEST: "test"}


# ----------------------------------------------------------------------------- configs.dat

@dataclass
class Configs:
    wing_id: np.ndarray   # (W,) int64
    params: np.ndarray    # (W, 38)
    mach: np.ndarray      # (W, 8)
    aoa: np.ndarray       # (W, 8)

    def rows_of(self, ids) -> np.ndarray:
        ids = np.asarray(ids).astype(np.int64)
        rows = np.searchsorted(self.wing_id, ids)
        ok = (rows < len(self.wing_id)) & (self.wing_id[np.minimum(rows, len(self.wing_id) - 1)] == ids)
        if not np.all(ok):
            raise KeyError(f"wing ids not in configs.dat: {np.unique(ids[~ok])[:10]}")
        return rows


def read_configs(path) -> Configs:
    with open(path) as f:
        header = f.readline()
    if not header.startswith("VARIABLES ="):
        raise ValueError(f"unexpected configs.dat header: {header[:40]!r}")
    names = header.split("=", 1)[1].split()
    if len(names) != 54 or names[37] != "tcroot" or names[:7] != GEOM_NAMES[:7]:
        raise ValueError(f"unexpected configs.dat columns ({len(names)}): {names[:8]} ... {names[36:40]}")
    a = np.loadtxt(path, skiprows=1, dtype=np.float64, ndmin=2)
    if a.shape[1] != 55:
        raise ValueError(f"configs.dat rows have {a.shape[1]} values, expected 55")
    wid = a[:, 0].astype(np.int64)
    if not np.array_equal(wid, a[:, 0]) or not np.all(np.diff(wid) > 0):
        raise ValueError("configs.dat wing ids are not strictly increasing integers")
    return Configs(wing_id=wid, params=a[:, 1:39], mach=a[:, 39:47], aoa=a[:, 47:55])


# ----------------------------------------------------------------------------- index.npy and splits

@dataclass
class Index:
    """Case table in index.npy column order. wing_row (configs.dat row = origingeom/geom0 row) and
    wing_id (configs.dat column 0) are set by read_index / from_split_tables; a bare Index(array)
    (e.g. rebuilt from the reduced npz) only offers the flow and coefficient columns."""
    array: np.ndarray  # (N, 12)
    wing_row: np.ndarray = None
    wing_id: np.ndarray = None

    def col(self, name):
        return self.array[:, INDEX_COLUMNS.index(name)]

    @property
    def cond_idx(self):
        return self.col("cond_idx").astype(np.int64)

    @property
    def mach(self):
        return self.col("mach")

    @property
    def aoa(self):
        return self.col("aoa")


def read_index(path, cfg: Configs) -> Index:
    a = np.asarray(np.load(path), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 12:
        raise ValueError(f"index.npy shape {a.shape}, expected (N, 12)")
    row = a[:, 0].astype(np.int64)
    if not np.array_equal(row, a[:, 0]) or row.min() < 0 or row.max() >= len(cfg.wing_id):
        raise ValueError("index.npy column 0 is not a configs.dat row number")
    return Index(a, wing_row=row, wing_id=cfg.wing_id[row])


def index_from_split_tables(tab: dict, cfg: Configs) -> Index:
    """Index built from train/test.parquet: column 0 there is the wing ID; it is converted to the
    configs row so that `array` follows the index.npy convention."""
    wid = tab["wing_shape_idx"].astype(np.int64)
    row = cfg.rows_of(wid)
    a = tab["array"].copy()
    a[:, 0] = row
    return Index(a, wing_row=row, wing_id=wid)


def read_parquet_columns(path, force_builtin: bool = False) -> dict:
    if not force_builtin:
        try:
            import pandas as pd
            import pyarrow  # noqa: F401
            df = pd.read_parquet(path)
            return {c: df[c].to_numpy() for c in df.columns}
        except ImportError:
            pass
    from . import miniparquet
    return miniparquet.read(str(path))


def read_split_tables(train_path, test_path) -> dict:
    tr, te = read_parquet_columns(train_path), read_parquet_columns(test_path)
    for t, nm in ((tr, train_path), (te, test_path)):
        if list(t) != PARQUET_COLUMNS:
            raise ValueError(f"{nm}: columns {list(t)} != {PARQUET_COLUMNS}")
    out = {c: np.concatenate([tr[c], te[c]]) for c in PARQUET_COLUMNS}
    out["is_test"] = np.concatenate([np.zeros(len(tr["mach"]), bool), np.ones(len(te["mach"]), bool)])
    out["array"] = np.stack([out[c] for c in PARQUET_COLUMNS], axis=1).astype(np.float64)
    return out


def case_positions(idx: Index, wing_ids, cond_idx) -> np.ndarray:
    """Row positions in index.npy of (wing ID, condition index) pairs (idx from read_index)."""
    key = idx.wing_id * 16 + idx.cond_idx
    if len(np.unique(key)) != len(key):
        raise ValueError("duplicate (wing, condition) keys in index.npy")
    order = np.argsort(key)
    q = np.asarray(wing_ids).astype(np.int64) * 16 + np.asarray(cond_idx).astype(np.int64)
    p = np.searchsorted(key[order], q)
    p = np.minimum(p, len(key) - 1)
    if not np.all(key[order][p] == q):
        raise KeyError("some (wing, condition) pairs are not in index.npy")
    return order[p]


@dataclass
class Splits:
    fit_wings: np.ndarray
    calib_wings: np.ndarray
    test_wings: np.ndarray
    case_code: np.ndarray  # per case: SPLIT_FIT / SPLIT_CALIB / SPLIT_TEST


def make_splits(case_wing_ids, case_is_test, frac: float, seed: int) -> Splits:
    """Official wing-level train/test split; calibration = round(frac * n_train_wings) training
    wings drawn without replacement by numpy default_rng(seed) from the sorted training wing ids."""
    w = np.asarray(case_wing_ids).astype(np.int64)
    is_test = np.asarray(case_is_test, bool)
    tr = np.unique(w[~is_test])
    te = np.unique(w[is_test])
    if len(np.intersect1d(tr, te)):
        raise ValueError("official split shares wings between train and test")
    n_cal = int(round(frac * len(tr)))
    cal = np.sort(np.random.default_rng(seed).choice(tr, size=n_cal, replace=False))
    fit = np.setdiff1d(tr, cal)
    code = np.full(len(w), SPLIT_FIT, np.int8)
    code[np.isin(w, cal)] = SPLIT_CALIB
    code[is_test] = SPLIT_TEST
    return Splits(fit, cal, te, code)


# ----------------------------------------------------------------------------- npy sources

class NpySource:
    """Row-sliceable view of a C-ordered .npy file: local path (memory map) or HTTP(S) URL
    (byte-range requests, so only the requested rows are transferred)."""

    def __init__(self, loc, retries: int = 5, timeout: float = 120.0):
        self.loc = str(loc)
        self.remote = self.loc.startswith(("http://", "https://"))
        self.retries, self.timeout = retries, timeout
        if self.remote:
            head = self._get(0, 65535)
            self.shape, self.dtype, self.offset = _parse_npy_header(head)
        else:
            self._mm = np.load(self.loc, mmap_mode="r")
            self.shape, self.dtype = self._mm.shape, self._mm.dtype
        self.row_items = int(np.prod(self.shape[1:]))

    def _get(self, a: int, b: int) -> bytes:
        import requests
        last = None
        for k in range(self.retries):
            try:
                r = requests.get(self.loc, headers={"Range": f"bytes={a}-{b}"}, timeout=self.timeout)
                if r.status_code == 206 or (r.status_code == 200 and a == 0 and len(r.content) <= b + 1):
                    return r.content
                if r.status_code == 200:
                    return r.content[a:b + 1]
                last = RuntimeError(f"HTTP {r.status_code} for {self.loc}")
            except Exception as e:  # network errors are retried, then raised
                last = e
            time.sleep(2 ** k)
        raise RuntimeError(f"range request failed for {self.loc} [{a}, {b}]: {last}")

    def read(self, i0: int, i1: int, channel=None) -> np.ndarray:
        if not self.remote:
            a = self._mm[i0:i1] if channel is None else self._mm[i0:i1, channel]
            return np.array(a)
        isz = self.dtype.itemsize
        if channel is None:
            a = self.offset + i0 * self.row_items * isz
            b = self.offset + i1 * self.row_items * isz - 1
            buf = self._get(a, b)
            return np.frombuffer(buf, self.dtype).reshape((i1 - i0,) + tuple(self.shape[1:])).copy()
        ch_items = int(np.prod(self.shape[2:]))
        from concurrent.futures import ThreadPoolExecutor

        def one(i):
            a = self.offset + (i * self.row_items + channel * ch_items) * isz
            return np.frombuffer(self._get(a, a + ch_items * isz - 1), self.dtype)
        with ThreadPoolExecutor(max_workers=8) as ex:
            out = list(ex.map(one, range(i0, i1)))
        return np.stack(out).reshape((i1 - i0,) + tuple(self.shape[2:]))


def _parse_npy_header(buf: bytes):
    import ast
    if buf[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy file")
    major = buf[6]
    if major == 1:
        hl = struct.unpack("<H", buf[8:10])[0]
        start = 10
    else:
        hl = struct.unpack("<I", buf[8:12])[0]
        start = 12
    d = ast.literal_eval(buf[start:start + hl].decode("latin1"))
    if d["fortran_order"]:
        raise ValueError("Fortran-ordered .npy not supported")
    return tuple(d["shape"]), np.dtype(d["descr"]), start + hl


def open_npy(loc) -> NpySource:
    return NpySource(loc)


# ----------------------------------------------------------------------------- surface geometry ops
# nodes: (..., 3, I, J) with channels x, y (vertical), z (span); cells: (..., I-1, J-1)

def cell_centres(nodes):
    return 0.25 * (nodes[..., :-1, :-1] + nodes[..., :-1, 1:] + nodes[..., 1:, 1:] + nodes[..., 1:, :-1])


def cell_vector_areas(nodes):
    """0.5 * (p2 - p0) x (p3 - p1) per quad, shape (..., 3, I-1, J-1); orientation not fixed."""
    d1 = nodes[..., 1:, 1:] - nodes[..., :-1, :-1]
    d2 = nodes[..., 1:, :-1] - nodes[..., :-1, 1:]
    return 0.5 * np.cross(d1, d2, axis=-3)


def cell_area_magnitudes(nodes):
    return np.linalg.norm(cell_vector_areas(nodes), axis=-3)


def outward_vector_areas(nodes):
    """Vector areas oriented away from the wing: the sign is fixed so that sum(y_c * A_y) > 0
    (upper surface normals point up, lower surface normals down)."""
    va = cell_vector_areas(nodes)
    yc = cell_centres(nodes)[..., 1, :, :]
    s = np.sign(np.sum(yc * va[..., 1, :, :], axis=(-2, -1)))
    if np.any(s == 0):
        raise ValueError("cannot determine surface orientation")
    return va * s[..., None, None, None]


def reduce_weights(w):
    return w[..., 0::2, 0::2] + w[..., 1::2, 0::2] + w[..., 0::2, 1::2] + w[..., 1::2, 1::2]


def reduce_cells(f, w):
    """2 x 2 area-weighted agglomeration: (..., I, J) -> (..., I/2, J/2)."""
    if f.shape[-2] % 2 or f.shape[-1] % 2:
        raise ValueError(f"cell grid {f.shape[-2:]} not divisible by 2")
    fw = f * w
    num = fw[..., 0::2, 0::2] + fw[..., 1::2, 0::2] + fw[..., 0::2, 1::2] + fw[..., 1::2, 1::2]
    return num / reduce_weights(w)


def coarse_nodes(nodes):
    return nodes[..., ::2, ::2]


def integrate_coefficients(nodes, cp, aoa_deg, s_ref, ref_point=(0.25, 0.0, 0.0), c_ref=1.0):
    """Pressure-only CL, CD, CMz of one surface. Force = -sum(Cp * A_out) (units of q_inf);
    CL = Fy cos(a) - Fx sin(a), CD = Fx cos(a) + Fy sin(a), CMz about ref_point."""
    A = outward_vector_areas(np.asarray(nodes, np.float64))
    dF = -np.asarray(cp, np.float64)[None] * A
    F = dF.sum(axis=(-2, -1))
    a = np.radians(aoa_deg)
    cl = (F[1] * np.cos(a) - F[0] * np.sin(a)) / s_ref
    cd = (F[0] * np.cos(a) + F[1] * np.sin(a)) / s_ref
    r = cell_centres(np.asarray(nodes, np.float64)) - np.asarray(ref_point, float)[:, None, None]
    mz = np.sum(r[0] * dF[1] - r[1] * dF[0])
    return float(cl), float(cd), float(mz / (s_ref * c_ref))


# ----------------------------------------------------------------------------- reduction (resumable)

def _atomic_save(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def reduce_superwing(out_dir, origingeom, geom0, fields, case_wing_row, chunk_cases: int = 1024,
                     chunk_wings: int = 256, max_chunks=None, cp_channel: int = 0, log=print) -> dict:
    """Reduce Cp (fields channel cp_channel) and geometry from 128 x 256 cells to 64 x 128 cells.

    Wing chunks hold geom_n (coarse nodes 65 x 129, float32) and geom_c (area-weighted geom0
    centres 64 x 128, float32); case chunks hold Cp (64 x 128, float16), area-weighted with the
    fine cell areas of the case's wing. Each chunk is written atomically to out_dir and skipped on
    restart, so an interrupted run resumes where it stopped. Returns a status dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    og, g0, fd = open_npy(origingeom), open_npy(geom0), open_npy(fields)
    wrow = np.asarray(case_wing_row, np.int64)
    n_w, n_c = og.shape[0], fd.shape[0]
    if g0.shape[0] != n_w or len(wrow) != n_c or wrow.max() >= n_w:
        raise ValueError("inconsistent sizes between origingeom, geom0, fields and case_wing_row")
    manifest = dict(n_wings=n_w, n_cases=n_c, chunk_cases=chunk_cases, chunk_wings=chunk_wings,
                    cp_channel=cp_channel, node_shape=list(og.shape[1:]), cell_shape=list(fd.shape[2:]),
                    case_wing_row_sum=int(wrow.sum()))
    mpath = out / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if old != manifest:
            raise ValueError(f"existing chunks in {out} were made with different settings: {old}")
    else:
        mpath.write_text(json.dumps(manifest))

    done = reused = 0
    stop = False

    def budget():
        return max_chunks is not None and done >= max_chunks

    for w0 in range(0, n_w, chunk_wings):
        w1 = min(n_w, w0 + chunk_wings)
        p = out / f"wings_{w0:05d}_{w1:05d}.npz"
        if p.exists():
            reused += 1
            continue
        if budget():
            stop = True
            break
        nodes = og.read(w0, w1).astype(np.float64)
        cen = g0.read(w0, w1).astype(np.float64)
        w = cell_area_magnitudes(nodes)
        geom_c = reduce_cells(cen, w[:, None])
        _atomic_save(p, geom_n=coarse_nodes(nodes).astype(np.float32), geom_c=geom_c.astype(np.float32))
        done += 1
        log(f"wings {w0}-{w1} done")

    if not stop:
        for c0 in range(0, n_c, chunk_cases):
            c1 = min(n_c, c0 + chunk_cases)
            p = out / f"cases_{c0:06d}_{c1:06d}.npz"
            if p.exists():
                reused += 1
                continue
            if budget():
                stop = True
                break
            cp = fd.read(c0, c1, channel=cp_channel).astype(np.float64)
            rows = wrow[c0:c1]
            areas = {int(r): cell_area_magnitudes(og.read(int(r), int(r) + 1)[0].astype(np.float64))
                     for r in np.unique(rows)}
            w = np.stack([areas[int(r)] for r in rows])
            _atomic_save(p, cp=reduce_cells(cp, w).astype(np.float16))
            done += 1
            log(f"cases {c0}-{c1} done")
    complete = not stop
    return dict(complete=complete, n_chunks_done=done, n_chunks_reused=reused, out_dir=str(out))


def slice_first_wings(raw_dir, out_dir, n_wings: int, cfg: Configs, idx: Index) -> dict:
    """Write origingeom/geom0/data/index slices holding the first n_wings configs rows and all
    their cases (in index.npy order) for a quick end-to-end run. Returns the slice paths."""
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = np.where(idx.wing_row < n_wings)[0]
    paths = {}
    for name in ("origingeom", "geom0"):
        src = open_npy(raw / f"{name}.npy")
        p = out / f"{name}_first{n_wings}.npy"
        np.save(p, src.read(0, n_wings))
        paths[name] = p
    fd = open_npy(raw / "data.npy")
    p = out / f"data_first{n_wings}.npy"
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=fd.dtype, shape=(len(cases),) + tuple(fd.shape[1:]))
    for k in range(0, len(cases), 256):
        sel = cases[k:k + 256]
        if np.all(np.diff(sel) == 1):
            mm[k:k + len(sel)] = fd.read(int(sel[0]), int(sel[-1]) + 1)
        else:
            mm[k:k + len(sel)] = np.stack([fd.read(int(i), int(i) + 1)[0] for i in sel])
    mm.flush()
    del mm
    paths["data"] = p
    p = out / f"index_first{n_wings}.npy"
    np.save(p, idx.array[cases])
    paths["index"] = p
    paths["cases"] = cases
    return paths


def subset_configs(cfg: Configs, n_rows: int) -> Configs:
    return Configs(cfg.wing_id[:n_rows], cfg.params[:n_rows], cfg.mach[:n_rows], cfg.aoa[:n_rows])


def load_reduced_chunks(out_dir) -> dict:
    out = Path(out_dir)
    m = json.loads((out / "manifest.json").read_text())
    wf = sorted(out.glob("wings_*.npz"))
    cf = sorted(out.glob("cases_*.npz"))
    exp_w = len(range(0, m["n_wings"], m["chunk_wings"]))
    exp_c = len(range(0, m["n_cases"], m["chunk_cases"]))
    if len(wf) != exp_w or len(cf) != exp_c:
        raise RuntimeError(f"reduction incomplete: {len(wf)}/{exp_w} wing chunks, {len(cf)}/{exp_c} case chunks")
    ws = [np.load(p) for p in wf]
    return dict(geom_n=np.concatenate([w["geom_n"] for w in ws]),
                geom_c=np.concatenate([w["geom_c"] for w in ws]),
                cp=np.concatenate([np.load(p)["cp"] for p in cf]))


def assemble_reduced(out_dir, npz_path, index: Index, cfg: Configs, case_wing_row, split_code, meta: dict):
    r = load_reduced_chunks(out_dir)
    if r["cp"].shape[0] != index.array.shape[0] or r["geom_n"].shape[0] != len(cfg.wing_id):
        raise ValueError("reduced arrays do not match index/configs sizes")
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez(tmp, cp=r["cp"], geom_c=r["geom_c"], geom_n=r["geom_n"], index=index.array,
             case_wing_row=np.asarray(case_wing_row, np.int32), split_code=np.asarray(split_code, np.int8),
             wing_id=cfg.wing_id, params=cfg.params, cfg_mach=cfg.mach, cfg_aoa=cfg.aoa,
             param_names=np.array(GEOM_NAMES), index_columns=np.array(INDEX_COLUMNS),
             meta_json=np.array(json.dumps(meta)))
    os.replace(tmp, npz_path)
    return npz_path


def load_reduced(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- integral checks

def cl_check(red: dict, n_cases: int, seed: int, min_abs_cl: float) -> dict:
    """Pressure-only CL from the reduced Cp on the reduced nodes vs index.npy column 9
    (cl_surface, reference mesh, includes friction)."""
    idx = Index(red["index"])
    n = idx.array.shape[0]
    pick = np.sort(np.random.default_rng(seed).choice(n, size=n_cases, replace=False))
    pred = np.empty(n_cases)
    for k, i in enumerate(pick):
        nodes = red["geom_n"][red["case_wing_row"][i]].astype(np.float64)
        pred[k] = integrate_coefficients(nodes, red["cp"][i].astype(np.float64), idx.aoa[i],
                                         idx.col("s_ref_half")[i])[0]
    ref = idx.col("cl_surface")[pick]
    abs_err = pred - ref
    big = np.abs(ref) >= min_abs_cl
    rel = np.abs(abs_err[big] / ref[big])
    out = dict(n_cases=int(n_cases), seed=int(seed), cases=pick, cl_pred=pred, cl_ref=ref,
               n_large_cl=int(big.sum()), max_rel_err_large_cl=float(rel.max()),
               median_rel_err_large_cl=float(np.median(rel)), mean_abs_err=float(np.mean(np.abs(abs_err))),
               max_abs_err=float(np.max(np.abs(abs_err))), mean_signed_err=float(np.mean(abs_err)),
               ratio_median=float(np.median(pred[big] / ref[big])))
    out["summary"] = {k: v for k, v in out.items() if not isinstance(v, np.ndarray)}
    return out


def cm_reference_check(red: dict, n_cases: int, seed: int) -> dict:
    """Pressure-only CMz about candidate reference points vs index.npy column 11 (cm_surface)."""
    idx = Index(red["index"])
    pick = np.sort(np.random.default_rng(seed).choice(idx.array.shape[0], size=n_cases, replace=False))
    cands = {"origin": (0.0, 0.0, 0.0), "x0.25": (0.25, 0.0, 0.0)}
    res = {}
    ref = idx.col("cm_surface")[pick]
    for name, pt in cands.items():
        cm = np.array([integrate_coefficients(red["geom_n"][red["case_wing_row"][i]].astype(np.float64),
                                              red["cp"][i].astype(np.float64), idx.aoa[i],
                                              idx.col("s_ref_half")[i], ref_point=pt)[2] for i in pick])
        for sgn in (1, -1):
            res[f"{name}_sign{'+' if sgn > 0 else '-'}"] = float(np.mean(np.abs(sgn * cm - ref)))
    best = min(res, key=res.get)
    return dict(n_cases=int(n_cases), seed=int(seed), mae_by_candidate=res, best=best,
                best_mae=res[best], cm_ref_abs_mean=float(np.mean(np.abs(ref))))


# ----------------------------------------------------------------------------- Task 3 wing splits

def task3_splits(red: dict, spec: dict) -> dict:
    """Wing rows (configs.dat rows) of the Task 3 splits, from the npz split codes:
    train / val (val = seeded draw from the fit wings), calib (calib_a fits gamma, calib_b sets the
    conformal quantiles) and test. Every array is sorted."""
    t = spec["task3"]["splits"]
    rows, code = np.asarray(red["case_wing_row"]), np.asarray(red["split_code"])
    per = {}
    for c in (SPLIT_FIT, SPLIT_CALIB, SPLIT_TEST):
        per[c] = np.unique(rows[code == c])
    if len(np.intersect1d(per[SPLIT_FIT], per[SPLIT_CALIB])) or len(np.intersect1d(per[SPLIT_FIT], per[SPLIT_TEST])) \
            or len(np.intersect1d(per[SPLIT_CALIB], per[SPLIT_TEST])):
        raise ValueError("a wing has cases in more than one split")
    fit = per[SPLIT_FIT]
    n_val = int(round(t["val_fraction_of_fit_wings"] * len(fit)))
    val = np.sort(np.random.default_rng(t["val_seed"]).choice(fit, size=n_val, replace=False))
    cal = per[SPLIT_CALIB]
    n_a = int(round(t["calib_gamma_fraction"] * len(cal)))
    cal_a = np.sort(np.random.default_rng(t["calib_gamma_seed"]).choice(cal, size=n_a, replace=False))
    return dict(train=np.setdiff1d(fit, val), val=val, calib=cal, calib_a=cal_a,
                calib_b=np.setdiff1d(cal, cal_a), test=per[SPLIT_TEST])


def ood_split(red: dict, spec: dict, s: dict) -> dict:
    """Outer-tail OOD split (spec task3.ood): threshold from fit + calibration wings, tail wings removed
    from train, val and both calibration halves; id_test = test wings below the threshold."""
    o = spec["task3"]["ood"]
    if o["tail"] != "upper" or o["threshold_population"] != "fit+calib":
        raise ValueError(f"unsupported OOD setting {o}")
    p = np.asarray(red["params"])[:, list(GEOM_NAMES).index(o["parameter"])]
    pop = np.concatenate([s["train"], s["val"], s["calib"]])
    thr = float(np.quantile(p[pop], o["quantile"]))
    tail = np.where(p >= thr)[0]
    keep = lambda a: np.setdiff1d(a, tail)
    return dict(threshold=thr, parameter=o["parameter"], tail=tail, train=keep(s["train"]), val=keep(s["val"]),
                calib_a=keep(s["calib_a"]), calib_b=keep(s["calib_b"]), id_test=keep(s["test"]),
                tail_test=np.intersect1d(tail, s["test"]))


# ----------------------------------------------------------------------------- Task 4 CFD-budget subsets

def budget_subsets(train_wings, fractions, seed: int) -> dict:
    """Nested CFD-budget subsets (spec task4.budget_study): one seeded permutation of the sorted training wings;
    the subset for fraction f is its first round(f * n) wings (sorted). Prefixes of the same permutation, so
    smaller budgets are contained in larger ones whatever fraction list is asked for."""
    w = np.sort(np.asarray(train_wings, np.int64))
    perm = np.random.default_rng(seed).permutation(w)
    out = {}
    for f in fractions:
        f = float(f)
        if not 0.0 < f <= 1.0:
            raise ValueError(f"budget fraction must be in (0, 1], got {f}")
        n = int(round(f * len(w)))
        if n < 1:
            raise ValueError(f"budget fraction {f} leaves no training wing (n = {len(w)})")
        out[f] = np.sort(perm[:n])
    return out


def budget_split(s: dict, fraction: float, spec: dict) -> dict:
    """Task 3 splits with the training wings replaced by the budget subset; val, calib and test unchanged."""
    sub = budget_subsets(s["train"], [fraction], spec["task4"]["budget_study"]["subset_seed"])[float(fraction)]
    return dict(s, train=sub)
