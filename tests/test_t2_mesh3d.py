"""Task 2 verification: gmsh 3D Euler mesh of a SuperWing-format surface (needs gmsh; skipped with a
message where it is not installed, e.g. the Claude workspace; runs on GitHub Actions and Colab).
Checks: markers present, wall nodes on the data surface, symmetry plane at the root, positive
tetrahedra, loft volume equal to the closed data surface volume."""
import numpy as np
import pytest

from conflow import cfd, geometry

try:
    import gmsh  # noqa: F401
except (ImportError, OSError) as e:   # the pip wheel also needs system GL libraries (libGLU)
    pytest.skip(f"gmsh not usable in this environment: {e}", allow_module_level=True)

from test_t2_cfd_geometry import _synthetic_wing  # noqa: E402


@pytest.fixture(scope="module")
def coarse_mesh(tmp_path_factory):
    s = _synthetic_wing()
    out = tmp_path_factory.mktemp("m3d") / "wing.su2"
    stats = cfd.wing_mesh_3d(s, out, size_factor=4.0, radius=20.0)
    return s, out, stats


def test_markers_and_cells(coarse_mesh):
    s, out, st = coarse_mesh
    m = cfd.read_su2_mesh(out)
    assert m.dim == 3
    assert set(m.markers) == {"wing", "symmetry", "farfield"}
    assert m.n_elements == st["n_cells"] > 10000


def test_positive_tets(coarse_mesh):
    _, out, _ = coarse_mesh
    m = cfd.read_su2_mesh(out)
    v = cfd.tet_volumes(m)
    # orientation is either convention (SU2 re-orients); no tetrahedron may be degenerate
    assert np.all(v > 0) or np.all(v < 0), f"{np.sum(v > 0)} / {np.sum(v < 0)} positive / negative"
    assert np.min(np.abs(v)) > 0


def test_wall_on_data_surface_and_root_on_symmetry(coarse_mesh):
    s, out, st = coarse_mesh
    assert st["wall_to_data_max"] < 5e-3      # root-chord units; loft between stride-4 sections
    assert st["wall_to_data_rms"] < 1e-3
    m = cfd.read_su2_mesh(out)
    sym = np.unique(np.concatenate([c.ravel() for c in m.markers["symmetry"].values()]))
    y_sym = m.points[sym, 1]
    assert np.ptp(y_sym) < 1e-9                    # planar
    assert y_sym[0] == pytest.approx(-s[2, 0, 0], abs=1e-9)   # SU2 y = -(SuperWing span z)


def test_loft_volume_matches_closed_surface(coarse_mesh):
    s, _, st = coarse_mesh
    v, f = geometry.close_surface(s)
    vol = geometry.check_watertight(v, f)["signed_volume"]
    assert st["loft_volume"] == pytest.approx(vol, rel=5e-3)


@pytest.mark.skipif(not __import__("os").environ.get("CONFLOW_SLOW"),
                    reason="gmsh tetra fill of the RANS mesh is slow (5.4 h at full size); set CONFLOW_SLOW=1")
def test_rans_hybrid_mesh(tmp_path):
    from test_t2_cfd_geometry import _sharp_te
    s = _sharp_te(_synthetic_wing())
    st = cfd.wing_mesh_3d_rans(s, tmp_path / "r.su2", reynolds=2e7, span_stride=8, chord_stride=4,
                               radius=20.0, n_radial=30, n_rings=8, last_layer_chords=0.01)
    m = cfd.read_su2_mesh(tmp_path / "r.su2")
    assert set(m.markers) == {"wing", "symmetry", "farfield"}
    assert m.n_elements == st["n_cells"] and st["n_prisms"] > 0 and st["n_tets"] > 0
    assert np.all(cfd.tet_volumes(m) > 0)


@pytest.mark.skipif(not __import__("os").environ.get("CONFLOW_SLOW"), reason="RANS mesh test; set CONFLOW_SLOW=1")
def test_rans_mesh_task4_options(tmp_path):
    """Task 4 RANS mesh fixes (spec task4.rans_spot_check.mesh): HXT tetra fill, graded prism-to-tetra transition,
    builder surface with the blunt TE collapsed. Coarse settings for speed."""
    from test_t4_design_case import design_params
    s = cfd.sharpen_te(geometry.build_surface(design_params()))
    assert np.allclose(s[:, :, 0], s[:, :, -1])
    plain = cfd.wing_mesh_3d_rans(s, tmp_path / "a.su2", reynolds=2e7, span_stride=8, chord_stride=4, radius=20.0,
                                  n_radial=30, n_rings=8, last_layer_chords=0.01, algorithm3d=10, n_threads=2)
    graded = cfd.wing_mesh_3d_rans(s, tmp_path / "b.su2", reynolds=2e7, span_stride=8, chord_stride=4, radius=20.0,
                                   n_radial=30, n_rings=8, last_layer_chords=0.01, algorithm3d=10, n_threads=2,
                                   transition_slope=0.2)
    for f, st in (("a.su2", plain), ("b.su2", graded)):
        m = cfd.read_su2_mesh(tmp_path / f)
        assert set(m.markers) == {"wing", "symmetry", "farfield"} and m.n_elements == st["n_cells"]
        assert np.all(cfd.tet_volumes(m) > 0)
    assert graded["n_tets"] > 2 * plain["n_tets"]                 # the graded field refines next to the prisms
    assert graded["transition_s0"] == pytest.approx(1.2 * graded["last_layer_median"])
