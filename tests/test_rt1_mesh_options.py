"""RT1 / A1 fix tests: optional Euler mesh settings for design cases (case["mesh"] keys sharp_te, wall_size_factor,
farfield_radius). Without them a design case must call the mesher exactly as in Task 4 (same mesh, same hash).
Written before the RT1 fix-test runs."""
import os

import numpy as np
import pytest

from conflow import cfd, geometry
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


def test_default_design_mesh_call_unchanged(tmp_path, spy):
    cfd.prepare_case(case(1.0), tmp_path)
    surf, kw = spy.calls[0]
    np.testing.assert_array_equal(surf, geometry.build_surface(design_params()))
    assert set(kw) == {"size_factor", "log"} and kw["size_factor"] == 1.0


def test_mesh_options_are_passed(tmp_path, spy):
    c = case(1.0)
    c["mesh"].update(sharp_te=True, wall_size_factor=0.75, farfield_radius=100.0)
    prep = cfd.prepare_case(c, tmp_path)
    surf, kw = spy.calls[0]
    np.testing.assert_array_equal(surf, geometry.build_surface(design_params(), te_thickness=0.0))
    np.testing.assert_array_equal(surf[:, :, 0], surf[:, :, -1])                  # closed TE (mesher's sharp branch)
    np.testing.assert_array_equal(np.load(tmp_path / "surface_meshed.npy"), surf)
    assert kw["wall_size_factor"] == 0.75 and kw["radius"] == 100.0
    np.testing.assert_array_equal(np.load(tmp_path / "surface_builder.npy"), geometry.build_surface(design_params()))
    assert prep["mesh_options"] == dict(sharp_te=True, wall_size_factor=0.75, farfield_radius=100.0)


def test_builder_default_unchanged_and_sharp_section():
    p = design_params()
    a = geometry.build_surface(p)
    np.testing.assert_array_equal(a, geometry.build_surface(p, te_thickness=geometry.TAIL))
    b = geometry.build_surface(p, te_thickness=0.0)
    assert np.abs(a - b).max() < 0.5 * geometry.TAIL * 1.6          # TE term (<= 0.002 c) scaled by the local chord
    root_t = lambda s: (s[1, 0, 128:] - s[1, 0, 128::-1]).max()     # max thickness of the root section
    assert abs(root_t(a) - root_t(b)) < 1e-12
    np.testing.assert_array_equal(b[:, :, 0], b[:, :, -1])


def test_sharp_te_false_is_default(tmp_path, spy):
    c = case(1.0)
    c["mesh"]["sharp_te"] = False
    cfd.prepare_case(c, tmp_path)
    surf, kw = spy.calls[0]
    np.testing.assert_array_equal(surf, geometry.build_surface(design_params()))
    assert set(kw) == {"size_factor", "log"}


@pytest.mark.parametrize("bad", [dict(wall_size=0.5), dict(wall_size_factor=0.0), dict(farfield_radius=-1.0),
                                 dict(sharp_te="yes")])
def test_bad_mesh_options_rejected(tmp_path, spy, bad):
    c = case(1.0)
    c["mesh"].update(bad)
    with pytest.raises(ValueError):
        cfd.prepare_case(c, tmp_path)
    assert not spy.calls


def test_sharp_and_wall_refined_mesh_builds(tmp_path):
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"gmsh not usable in this environment: {e}")
    if not os.environ.get("CONFLOW_SLOW"):
        pytest.skip("3D meshing test runs only with CONFLOW_SLOW=1")
    s = geometry.build_surface(design_params(), te_thickness=0.0)
    a = cfd.wing_mesh_3d(s, tmp_path / "a.su2", size_factor=3.0)
    b = cfd.wing_mesh_3d(s, tmp_path / "b.su2", size_factor=3.0, wall_size_factor=0.75, radius=60.0)
    assert a["wall_to_data_max"] < 1e-2 and b["wall_to_data_max"] < 1e-2
    assert b["n_cells"] > a["n_cells"] and b["radius"] == 60.0 and b["wall_size_factor"] == 0.75
    assert "wall_size_factor" not in a
