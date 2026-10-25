"""Task 4 session 2: a design case (wing built from 38 parameters) is meshed for SU2 Euler with the Task 2 mesher and
its config carries the design's Mach, AoA and reference area. Needs gmsh (runs in the runner repo's tests workflow,
conda-forge environment); skipped elsewhere with the reason."""
import os

import numpy as np
import pytest

from conflow import cfd, data, geometry

NAMES = list(data.GEOM_NAMES)


def design_params():
    """Hand-made SuperWing-family wing for tests (not data)."""
    p = np.zeros(38)
    for k, v in dict(SA=32.0, DA=3.0, DA_kink=1.0, AR=9.5, TR=0.25, kink=0.39, rootadj=0.5, tcroot=0.155).items():
        p[NAMES.index(k)] = v
    p[NAMES.index("rtcs_0"):NAMES.index("rtcs_2") + 1] = [0.8, 0.9, 0.95]
    p[NAMES.index("cambers_0"):NAMES.index("cambers_2") + 1] = [1.0, 1.0, 1.0]
    p[NAMES.index("twists_0"):NAMES.index("twists_3") + 1] = [-2.0, -2.0, -2.0, -2.0]
    u = np.linspace(0.17, 0.20, 10)
    p[NAMES.index("cst_u_0"):NAMES.index("cst_u_9") + 1] = u
    p[NAMES.index("cst_l_0"):NAMES.index("cst_l_9") + 1] = -0.8 * u
    return p


def case(size_factor):
    return dict(name="t4_test_design", kind="design", solver="euler",
                mesh=dict(type="euler_tetra", size_factor=size_factor),
                flow=dict(ref_length=1.0, moment_ref=[0.25, 0.0, 0.0]),
                markers=dict(wall="wing", far="farfield", sym="symmetry"),
                design=dict(params=design_params().tolist(), mach=0.82, aoa_deg=2.5, ref_area=1.234))


def _gmsh():
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"gmsh not usable in this environment: {e}")


def test_design_case_mesh_and_config(tmp_path):
    _gmsh()
    if not os.environ.get("CONFLOW_SLOW"):
        pytest.skip("3D meshing test runs only with CONFLOW_SLOW=1 (runner tests workflow)")
    prep = cfd.prepare_case(case(2.0), tmp_path)
    assert prep["kind"] == "design" and prep["mesh_cells"] > 10000
    surf = np.load(tmp_path / "surface_builder.npy")
    np.testing.assert_array_equal(surf, geometry.build_surface(design_params()))
    m = prep["mesh3d"]
    assert m["wall_to_data_max"] < 1e-2, m["wall_to_data_max"]       # wall follows the built surface (held-out: 3.3e-3)
    cfg = cfd.case_config(case(2.0), prep)
    assert float(cfg["MACH_NUMBER"]) == 0.82 and float(cfg["AOA"]) == 2.5 and float(cfg["REF_AREA"]) == 1.234


def test_design_case_rejects_bad_params(tmp_path):
    c = case(2.0)
    c["design"]["params"] = c["design"]["params"][:37]
    with pytest.raises(ValueError):
        cfd.prepare_case(c, tmp_path)
