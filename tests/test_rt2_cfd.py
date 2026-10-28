"""RT2 CFD-side tests (copied to the runner repo tests workflow), written before the code changes they test.

Covers: surface rounding before meshing and surface/mesh hashes (A6), the explicit size_min mesh option and the
achieved edge spacing record (A5), the A7 convergence rule (all criteria together), SU2 fixed-CL settings, mesh-only
cases and the memory sampler. Torch-free; gmsh tests run only with CONFLOW_SLOW=1 (runner tests workflow)."""
import json
import os

import numpy as np
import pytest

from conflow import cfd, geometry
from conflow.spec import load_spec
from test_t4_design_case import case, design_params


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, surf, out_path, **kw):
        self.calls.append((np.array(surf, copy=True), kw))
        with open(out_path, "w") as f:
            f.write("NDIME= 3\nNELEM= 1\n")
        return dict(n_cells=1, size_factor=kw.get("size_factor"))


@pytest.fixture
def spy(monkeypatch):
    s = _Spy()
    monkeypatch.setattr(cfd, "wing_mesh_3d", s)
    return s


# ---- A6 determinism: rounding before meshing ------------------------------------------------------------------------

def test_round_surface_values_and_no_negative_zero():
    s = np.array([[1.0 + 4e-16, -1e-15, 0.1234567890123456, -0.0]])
    r = cfd.round_surface(s, 12)
    assert r.dtype == np.float64
    np.testing.assert_array_equal(r, np.round(s, 12) + 0.0)
    assert not np.any(np.signbit(r) & (r == 0))           # -0.0 would hash differently from 0.0
    np.testing.assert_array_equal(cfd.round_surface(r, 12), r)   # idempotent


def test_round_surface_absorbs_one_ulp_perturbations():
    """The A6 cross-CPU differences are 1 ulp. Rounding to 1e-12 maps a 1-ulp neighbour to the same value unless the
    value sits within 1 ulp of a rounding boundary; the fraction of such values must be tiny (it is reported, not hidden)."""
    p = design_params() + np.random.default_rng(3).normal(0.0, 1e-3, 38)   # generic (not round-number) parameters
    s = geometry.build_surface(p)
    up = s.copy()
    up[1] = np.nextafter(s[1], np.inf)          # A6: the cross-CPU differences are in the vertical channel
    diff = cfd.round_surface(s, 12) != cfd.round_surface(up, 12)
    assert diff.mean() < 2e-3


def test_rounded_design_mesh_option(tmp_path, spy):
    c = case(1.0)
    c["mesh"]["round_decimals"] = 12
    prep = cfd.prepare_case(c, tmp_path)
    surf, kw = spy.calls[0]
    raw = geometry.build_surface(design_params())
    np.testing.assert_array_equal(surf, cfd.round_surface(raw, 12))
    np.testing.assert_array_equal(np.load(tmp_path / "surface_builder.npy"), raw)      # raw builder output kept
    np.testing.assert_array_equal(np.load(tmp_path / "surface_meshed.npy"), surf)
    assert prep["mesh_options"] == dict(round_decimals=12)
    h = prep["surface_sha256"]
    assert h["builder"] == cfd.array_sha256(raw) and h["meshed"] == cfd.array_sha256(surf)
    assert set(kw) == {"size_factor", "log"}


def test_default_design_case_has_no_rounding(tmp_path, spy):
    prep = cfd.prepare_case(case(1.0), tmp_path)
    surf, _ = spy.calls[0]
    np.testing.assert_array_equal(surf, geometry.build_surface(design_params()))
    assert "mesh_options" not in prep
    assert prep["surface_sha256"]["builder"] == prep["surface_sha256"]["meshed"]


@pytest.mark.parametrize("bad", [-1, 20, 12.5, "12", True])
def test_bad_round_decimals_rejected(tmp_path, spy, bad):
    c = case(1.0)
    c["mesh"]["round_decimals"] = bad
    with pytest.raises(ValueError):
        cfd.prepare_case(c, tmp_path)
    assert not spy.calls


def test_array_sha256_is_dtype_and_shape_aware():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert cfd.array_sha256(a) == cfd.array_sha256(a.copy())
    assert cfd.array_sha256(a) != cfd.array_sha256(a.reshape(3, 2))
    assert cfd.array_sha256(a) != cfd.array_sha256(a.astype(np.float32))


# ---- A5 mesh family: explicit minimum size ----------------------------------------------------------------------------

def test_size_min_option_passed(tmp_path, spy):
    c = case(1.0)
    c["mesh"].update(edge_refine=dict(h0=0.001, growth=0.15), size_min=0.001)
    prep = cfd.prepare_case(c, tmp_path)
    _, kw = spy.calls[0]
    assert kw["size_min"] == 0.001 and kw["edge_refine"] == dict(h0=0.001, growth=0.15)
    assert prep["mesh_options"]["size_min"] == 0.001


@pytest.mark.parametrize("bad", [0.0, -1e-3, float("nan"), "x"])
def test_bad_size_min_rejected(tmp_path, spy, bad):
    c = case(1.0)
    c["mesh"]["size_min"] = bad
    with pytest.raises(ValueError):
        cfd.prepare_case(c, tmp_path)


def test_rebuilt_a2_case_accepts_rounding():
    opt = cfd.euler_mesh_options(dict(type="euler_tetra", size_factor=1.0, round_decimals=12, size_min=0.0015))
    assert opt == dict(round_decimals=12, size_min=0.0015)


# ---- A7 convergence rule -------------------------------------------------------------------------------------------------

def _hist(n=1000, drop=6.0, cd_drift=0.0, cl_end=0.5):
    it = np.arange(n, dtype=float)
    r = -1.0 - drop * it / (n - 1)
    cd = 0.02 + cd_drift * it / (n - 1)
    cl = np.full(n, float(cl_end))
    return {"rms[Rho]": r, "CD": cd, "CL": cl}


A7 = dict(mode="all", residual_drop_orders=5, cd_window_iters=200, cd_change_max=5e-6, cl_target_tol=1e-4)


def test_a7_rule_all_pass():
    st = cfd.convergence_status(_hist(), rule=dict(A7, cl_target=0.5))
    assert st["status"] == "converged" and st["criterion"] == "all"
    assert st["checks"] == dict(residual_drop=True, cd_cauchy=True, cl_target=True)


@pytest.mark.parametrize("kw,failed", [(dict(drop=4.0), "residual_drop"), (dict(cd_drift=1e-4), "cd_cauchy"),
                                       (dict(cl_end=0.5003), "cl_target")])
def test_a7_rule_each_criterion_needed(kw, failed):
    st = cfd.convergence_status(_hist(**kw), rule=dict(A7, cl_target=0.5))
    assert st["status"] == "unconverged" and st["checks"][failed] is False


def test_a7_rule_without_cl_target():
    st = cfd.convergence_status(_hist(cl_end=0.7), rule=A7)
    assert st["status"] == "converged" and "cl_target" not in st["checks"]


def test_a7_rule_short_history_not_converged():
    h = _hist(n=150)
    st = cfd.convergence_status(h, rule=A7)
    assert st["status"] == "unconverged" and st["checks"]["cd_cauchy"] is False


def test_legacy_rule_unchanged():
    h = _hist(drop=2.0)
    h["CL"] = np.full(1000, 0.5)
    assert cfd.convergence_status(h)["status"] == "converged"        # PLAN section 4 rule: CL window alone suffices
    assert cfd.convergence_status(h, rule=None)["criterion"] == "cl_change"


def test_bad_rule_rejected():
    with pytest.raises(ValueError):
        cfd.convergence_status(_hist(), rule=dict(mode="any"))
    with pytest.raises(ValueError):
        cfd.convergence_status(_hist(), rule=dict(A7, extra=1))


def test_case_rule_with_fixed_cl_target():
    c = case(1.0)
    c["flow"]["fixed_cl"] = dict(target_cl=0.45, dcl_dalpha=0.1, update_aoa_iter_limit=100)
    c["convergence"] = dict(A7)
    rule = cfd.case_convergence_rule(c)
    assert rule["cl_target"] == 0.45 and rule["mode"] == "all"
    assert cfd.case_convergence_rule(case(1.0)) is None


# ---- fixed-CL configuration -----------------------------------------------------------------------------------------------

def _prep():
    return dict(mesh_file="mesh.su2", flow=dict(mach=0.82, aoa=3.0, ref_area=2.0))


def test_fixed_cl_config():
    c = case(1.0)
    c["flow"]["fixed_cl"] = dict(target_cl=0.45, dcl_dalpha=0.1, update_aoa_iter_limit=100)
    cfg = cfd.case_config(c, _prep())
    assert cfg["FIXED_CL_MODE"] is True and cfg["TARGET_CL"] == 0.45 and cfg["DCL_DALPHA"] == 0.1
    assert cfg["UPDATE_AOA_ITER_LIMIT"] == 100 and cfg["ITER_DCL_DALPHA"] == 0 and cfg["EVAL_DOF_DCX"] is False
    assert cfg["AOA"] == 3.0                                     # start from the case AoA
    assert "AOA" in cfg["HISTORY_OUTPUT"] and "FIXED_CL" in cfg["HISTORY_OUTPUT"]
    for k in ("ITER", "RMS_RES", "AERO_COEFF", "WALL_TIME"):
        assert k in cfg["HISTORY_OUTPUT"]


def test_fixed_aoa_config_unchanged():
    cfg = cfd.case_config(case(1.0), _prep())
    assert "FIXED_CL_MODE" not in cfg and cfg["HISTORY_OUTPUT"] == ("ITER", "RMS_RES", "AERO_COEFF", "WALL_TIME")


@pytest.mark.parametrize("bad", [dict(), dict(target_cl=0.45), dict(target_cl=0.45, dcl_dalpha=0.0, update_aoa_iter_limit=100),
                                 dict(target_cl=0.45, dcl_dalpha=0.1, update_aoa_iter_limit=0),
                                 dict(target_cl=0.45, dcl_dalpha=0.1, update_aoa_iter_limit=100, x=1)])
def test_bad_fixed_cl_rejected(bad):
    c = case(1.0)
    c["flow"]["fixed_cl"] = bad
    with pytest.raises(ValueError):
        cfd.case_config(c, _prep())


def test_aoa_history_read(tmp_path):
    p = tmp_path / "history.csv"
    p.write_text('"Inner_Iter","rms[Rho]","CL","CD","AoA"\n0,-1,0.4,0.02,3.0\n1,-2,0.41,0.02,3.1\n')
    h = cfd.read_history(p)
    assert list(h["AoA"]) == [3.0, 3.1]
    assert cfd.final_aoa(h) == 3.1


# ---- mesh-only cases and the memory sampler --------------------------------------------------------------------------------

def test_mesh_only_flag():
    c = case(1.0)
    assert cfd.is_mesh_only(c) is False
    c["mesh_only"] = True
    assert cfd.is_mesh_only(c) is True
    c["mesh_only"] = "yes"
    with pytest.raises(ValueError):
        cfd.is_mesh_only(c)


def test_memory_sampler_records_peak():
    with cfd.MemorySampler(interval_s=0.05) as m:
        x = np.ones(30_000_000)            # 240 MB
        x[::1000] += 1.0
        import time
        time.sleep(0.3)
        del x
    r = m.result()
    assert r["n_samples"] >= 2 and r["mem_total_kb"] > 0
    assert r["peak_used_kb"] >= r["start_used_kb"]
    assert set(r) >= {"peak_used_kb", "start_used_kb", "mem_total_kb", "n_samples", "interval_s", "source"}


# ---- gmsh: achieved edge spacing and the size_min floor (runner tests workflow) -----------------------------------------

_SPACING_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from conflow import cfd, geometry
from test_t4_design_case import design_params
out = sys.argv[3]
s = cfd.round_surface(geometry.build_surface(design_params()), 12)
a = cfd.wing_mesh_3d(s, out + "/a.su2", size_factor=1.0, edge_refine=dict(h0=0.004, growth=0.3))
b = cfd.wing_mesh_3d(s, out + "/b.su2", size_factor=1.0, edge_refine=dict(h0=0.001, growth=0.3), size_min=0.001)
c = cfd.wing_mesh_3d(s, out + "/c.su2", size_factor=1.0, edge_refine=dict(h0=0.001, growth=0.3))
print("RESULT " + json.dumps(dict(a=a["edge_refine"]["achieved"], b=b["edge_refine"]["achieved"], c=c["edge_refine"]["achieved"],
                                  b_size_min=b["size_min"], c_size_min=c["size_min"], 
                                  cells=[a["n_cells"], b["n_cells"], c["n_cells"]])), flush=True)
"""


def test_achieved_edge_spacing(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"gmsh not usable in this environment: {e}")
    if not os.environ.get("CONFLOW_SLOW"):
        pytest.skip("3D meshing test runs only with CONFLOW_SLOW=1")
    here = Path(__file__).resolve().parent
    src = here.parent / "src" if (here.parent / "src").exists() else here.parent / "runner"
    r = subprocess.run([sys.executable, "-u", "-c", _SPACING_SCRIPT, str(src), str(here), str(tmp_path)],
                       capture_output=True, text=True, timeout=2400)
    log = Path("out/pytest/rt2_spacing.log")
    if log.parent.is_dir():
        log.write_text(r.stdout + r.stderr)
    res = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert res, (r.stdout + r.stderr)[-3000:]
    d = json.loads(res[-1][7:])
    for k in ("a", "b", "c"):
        for e in ("le", "te"):
            assert d[k][e]["n_segments"] > 10 and d[k][e]["median"] > 0
    # h0 above the default minimum size: the TE edge spacing follows h0
    assert 0.6 * 0.004 < d["a"]["te"]["median"] < 1.5 * 0.004
    # explicit size_min = h0: the TE spacing follows the finer h0
    assert d["b_size_min"] == 0.001 and d["b"]["te"]["median"] < 1.5 * 0.001
    assert d["c_size_min"] == load_spec()["cfd"]["mesh3d_euler"]["size_min"]


