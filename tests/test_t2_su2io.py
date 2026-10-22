"""Task 2 verification: SU2 config writer, history / surface / mesh parsers and the PLAN section 4
convergence rule (residual drop >= 5 orders, or CL change < 1e-4 over the last 200 iterations).
Written before any CFD run; all inputs are small synthetic files with known answers."""
import numpy as np
import pytest

from conflow import cfd


# ----------------------------------------------------------------------------- config writer

def test_config_roundtrip(tmp_path):
    cfg = {"SOLVER": "EULER", "MACH_NUMBER": 0.8395, "AOA": 3.06, "MARKER_EULER": ("wing",),
           "MARKER_FAR": ("farfield",), "MG_PRE_SMOOTH": (1, 2, 3, 3), "RESTART_SOL": False,
           "REF_ORIGIN_MOMENT_X": 0.25, "ITER": 500, "CONV_RESIDUAL_MINVAL": -14}
    p = cfd.write_config(tmp_path / "a.cfg", cfg)
    back = cfd.read_config(p)
    assert back["SOLVER"] == "EULER"
    assert float(back["MACH_NUMBER"]) == pytest.approx(0.8395, abs=0)
    assert back["MARKER_EULER"] == "( wing )"
    assert back["MG_PRE_SMOOTH"] == "( 1, 2, 3, 3 )"
    assert back["RESTART_SOL"] == "NO"
    assert int(back["ITER"]) == 500


def test_config_rejects_bad_keys(tmp_path):
    with pytest.raises(ValueError):
        cfd.write_config(tmp_path / "b.cfg", {"mach number": 0.8})
    with pytest.raises(ValueError):
        cfd.write_config(tmp_path / "c.cfg", {"MACH_NUMBER": float("nan")})


def test_templates_have_required_keys():
    e = cfd.euler_config("m.su2", mach=0.84, aoa=3.0, wall="wing", far="farfield", sym="symmetry",
                         ref_area=0.75, ref_length=0.65, moment_ref=(0.25, 0.0, 0.0))
    for k in ("SOLVER", "MACH_NUMBER", "AOA", "MARKER_EULER", "MARKER_FAR", "MARKER_SYM", "REF_AREA",
              "MESH_FILENAME", "CONV_FILENAME", "OUTPUT_FILES", "HISTORY_OUTPUT", "ITER"):
        assert k in e, k
    r = cfd.rans_sa_config("m.su2", mach=0.729, aoa=2.31, reynolds=6.5e6, re_length=1.0,
                           temperature=288.15, wall="airfoil", far="farfield", sym=None,
                           ref_area=1.0, ref_length=1.0, moment_ref=(0.25, 0.0, 0.0))
    assert r["SOLVER"] == "RANS" and r["KIND_TURB_MODEL"] == "SA"
    assert "MARKER_SYM" not in r
    assert r["MARKER_HEATFLUX"] == ("airfoil", 0.0)


# ----------------------------------------------------------------------------- history parser

def _write_history(path, it, rho, cl, cd=None, extra_space=True):
    cd = np.full_like(cl, 0.01) if cd is None else cd
    sep = ",  " if extra_space else ","
    head = sep.join(['"Time_Iter"', '"Outer_Iter"', '"Inner_Iter"', '"rms[Rho]"', '"rms[RhoE]"', '"CL"', '"CD"'])
    rows = [sep.join(f"{v:.10e}" if isinstance(v, float) else str(v) for v in (0, 0, int(i), float(r), float(r) - 1.0, float(a), float(b)))
            for i, r, a, b in zip(it, rho, cl, cd)]
    path.write_text(head + "\n" + "\n".join(rows) + "\n")
    return path


def test_history_parser_quoted_headers(tmp_path):
    it = np.arange(5)
    p = _write_history(tmp_path / "history.csv", it, -np.arange(5.0), 0.3 + 0.0 * it)
    h = cfd.read_history(p)
    assert set(("Inner_Iter", "rms[Rho]", "CL", "CD")) <= set(h)
    assert np.array_equal(h["Inner_Iter"], it)
    assert h["rms[Rho]"][-1] == -4.0


def test_history_concat_offsets_restarts(tmp_path):
    a = cfd.read_history(_write_history(tmp_path / "h0.csv", np.arange(100), np.linspace(-1, -3, 100), np.full(100, 0.3)))
    b = cfd.read_history(_write_history(tmp_path / "h1.csv", np.arange(50), np.linspace(-3, -4, 50), np.full(50, 0.31)))
    h = cfd.concat_histories([a, b])
    assert len(h["iter_global"]) == 150
    assert np.all(np.diff(h["iter_global"]) == 1)
    assert h["rms[Rho]"][0] == -1.0 and h["rms[Rho]"][-1] == -4.0


# ----------------------------------------------------------------------------- convergence rule

def _hist(n, drop, cl_tail_amp, seed=0):
    rng = np.random.default_rng(seed)
    rho = np.linspace(-1.0, -1.0 - drop, n)
    cl = 0.4 + cl_tail_amp * rng.uniform(-0.5, 0.5, n)
    return {"iter_global": np.arange(n), "rms[Rho]": rho, "CL": cl}


def test_convergence_by_residual_drop():
    s = cfd.convergence_status(_hist(1000, 5.2, 1e-2))
    assert s["status"] == "converged" and s["criterion"] == "residual_drop"
    assert s["residual_drop_orders"] == pytest.approx(5.2, abs=1e-9)


def test_convergence_by_cl_window():
    s = cfd.convergence_status(_hist(1000, 3.0, 5e-5))
    assert s["status"] == "converged" and s["criterion"] == "cl_change"
    assert s["cl_change_window"] < 1e-4


def test_unconverged_both_fail():
    s = cfd.convergence_status(_hist(1000, 3.0, 5e-3))
    assert s["status"] == "unconverged" and s["criterion"] is None


def test_cl_window_needs_200_iterations():
    s = cfd.convergence_status(_hist(150, 2.0, 0.0))
    assert s["status"] == "unconverged"


def test_residual_drop_measured_from_first_logged_value():
    h = _hist(1000, 4.0, 1e-2)
    h["rms[Rho]"][0] = -0.5          # transient rise after the first iteration counts in the drop
    h["rms[Rho]"][1:20] = 0.5
    s = cfd.convergence_status(h)
    assert s["residual_drop_orders"] == pytest.approx(-0.5 - h["rms[Rho]"][-1], abs=1e-12)


def test_diverged_on_nan():
    h = _hist(400, 6.0, 0.0)
    h["rms[Rho]"][-1] = np.nan
    s = cfd.convergence_status(h)
    assert s["status"] == "diverged"


def test_spec_values_used():
    sp = cfd.convergence_settings()
    assert sp["residual_drop_orders"] == 5 and sp["cl_change"] == 1e-4 and sp["window_iters"] == 200


# ----------------------------------------------------------------------------- surface parser

def test_surface_csv_parser(tmp_path):
    p = tmp_path / "surface_flow.csv"
    p.write_text('"PointID","x","y","z","Pressure","Pressure_Coefficient"\n'
                 '3,0.1,0.2,0.3,1.0e5,-0.5\n7,0.4,0.5,0.6,1.1e5,0.25\n')
    s = cfd.read_surface_csv(p)
    assert np.array_equal(s["PointID"], [3, 7])
    assert s["Pressure_Coefficient"][1] == 0.25


# ----------------------------------------------------------------------------- su2 mesh io

def test_su2_mesh_roundtrip_2d(tmp_path):
    X, Y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    idx = np.arange(12).reshape(3, 4)
    quads = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    lines_b = np.stack([idx[0, :-1], idx[0, 1:]], 1)
    p = cfd.write_su2_mesh(tmp_path / "m.su2", pts, {cfd.VTK_QUAD: quads}, {"bottom": {cfd.VTK_LINE: lines_b}})
    m = cfd.read_su2_mesh(p)
    assert m.dim == 2 and m.points.shape == (12, 2)
    assert m.n_elements == 6
    assert np.array_equal(m.markers["bottom"][cfd.VTK_LINE], lines_b)
    assert np.array_equal(m.elements[cfd.VTK_QUAD], quads)
