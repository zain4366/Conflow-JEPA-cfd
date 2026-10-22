"""Task 2 verification: sonic Cp, shock location, wing station cuts, 2D O-grid, frame rotation,
surface-to-grid mapping, experimental-file parsers. Analytic or synthetic answers only."""
import numpy as np
import pytest

from conflow import cfd, data, geometry


# ----------------------------------------------------------------------------- sonic Cp and shock

def _cp_isentropic(m_inf, m_loc, g=1.4):
    p_p0 = lambda m: (1 + 0.5 * (g - 1) * m * m) ** (-g / (g - 1))
    return (p_p0(m_loc) / p_p0(m_inf) - 1) / (0.5 * g * m_inf ** 2)


@pytest.mark.parametrize("m", [0.729, 0.8395, 0.85])
def test_cp_star_matches_isentropic(m):
    assert cfd.cp_star(m) == pytest.approx(_cp_isentropic(m, 1.0), rel=1e-12)


def test_shock_location_on_synthetic_profile():
    m = 0.8395
    x = np.linspace(0, 1, 801)
    cps = cfd.cp_star(m)
    # supersonic plateau below Cp*, sharp recompression at x = 0.62, subsonic afterwards
    cp = np.where(x < 0.62, cps - 0.35, cps + 0.25) + 0.0 * x
    cp = cps - 0.35 + 0.6 * 0.5 * (1 + np.tanh((x - 0.62) / 0.004))
    s = cfd.shock_location(x, cp, m)
    assert s["x_cpstar"] == pytest.approx(0.62, abs=3e-3)
    assert s["x_maxgrad"] == pytest.approx(0.62, abs=3e-3)


def test_shock_location_sparse_taps_and_no_shock():
    m = 0.8395
    cps = cfd.cp_star(m)
    x = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.58, 0.66, 0.75, 0.85, 0.95, 1.0])
    cp = np.where(x < 0.6, cps - 0.3, cps + 0.3)
    s = cfd.shock_location(x, cp, m)
    assert 0.58 <= s["x_cpstar"] <= 0.66           # crossing between the two taps
    sub = cfd.shock_location(x, np.full_like(x, cps + 0.2), m)
    assert sub["x_cpstar"] is None                   # never supersonic: no shock reported


# ----------------------------------------------------------------------------- station cuts

def _swept_wing_cloud(n=40000, seed=0):
    """Synthetic swept, tapered, untwisted wing (SU2 frame: x stream, y span, z up) with an
    analytic Cp(xi, eta) that differs between the surfaces."""
    rng = np.random.default_rng(seed)
    b, cr, lam, sweep = 1.2, 0.8, 0.56, np.radians(30)
    eta = rng.uniform(0, 1, n)
    xi = rng.uniform(0, 1, n)
    up = rng.uniform(0, 1, n) < 0.5
    c = cr * (1 - (1 - lam) * eta)
    t = 0.10 * c * 2 * np.sqrt(xi) * (1 - xi)
    pts = np.stack([eta * b * np.tan(sweep) + xi * c, eta * b, np.where(up, t, -t)], 1)
    cp = np.where(up, -0.8 * np.sin(np.pi * xi) * (1 - 0.3 * eta), 0.4 * np.cos(np.pi * xi))
    return pts, cp, (lambda xi_, eta_, upper: np.where(upper, -0.8 * np.sin(np.pi * xi_) * (1 - 0.3 * eta_),
                                                        0.4 * np.cos(np.pi * xi_)))


def test_station_cut_recovers_analytic_cp():
    pts, cp, f = _swept_wing_cloud()
    cut = cfd.wing_station_cut(pts, cp, eta=0.44, span_axis=1, up_axis=2, n=101)
    for side, upper in (("upper", True), ("lower", False)):
        xi = cut[side]["x_c"]
        ok = (xi > 0.05) & (xi < 0.95)
        err = np.abs(cut[side]["cp"][ok] - f(xi[ok], 0.44, upper))
        assert np.nanmax(err) < 0.02, (side, np.nanmax(err))
    assert cut["chord"] == pytest.approx(0.8 * (1 - 0.44 * 0.44), rel=0.02)


# ----------------------------------------------------------------------------- 2D O-grid

def _naca(t=0.12, n=161):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    # loop: lower TE -> LE -> upper TE (closed TE: last coefficient -0.1036 leaves a 0.0025 gap)
    xl, yl = x[::-1], -yt[::-1]
    return np.concatenate([np.stack([xl, yl], 1), np.stack([x[1:], yt[1:]], 1)])


@pytest.mark.parametrize("h1", [4e-6, 1e-3])
def test_ogrid_positive_and_first_layer(h1):
    xy = _naca()
    X, Y = cfd.ogrid_2d(xy, n_around=200, n_normal=96, h1=h1, radius=100.0)
    assert X.shape == (97, 200)
    a = cfd.quad_areas(X, Y)
    assert np.all(a > 0), f"{np.sum(a <= 0)} non-positive cells"
    d1 = np.hypot(X[1] - X[0], Y[1] - Y[0])
    assert np.median(d1) == pytest.approx(h1, rel=0.05)
    r = np.hypot(X[-1] - 0.5, Y[-1])
    assert np.all(np.abs(r - 100.0) < 1.0)


def test_ogrid_wall_matches_airfoil():
    xy = _naca()
    X, Y = cfd.ogrid_2d(xy, n_around=240, n_normal=64, h1=1e-4, radius=50.0)
    # every wall node lies on the input polyline (resampled by arclength)
    from conflow.cfd import _point_polyline_distance
    d = _point_polyline_distance(np.stack([X[0], Y[0]], 1), np.vstack([xy, xy[:1]]))
    assert d.max() < 2e-4


def test_ogrid_su2_export(tmp_path):
    X, Y = cfd.ogrid_2d(_naca(), n_around=120, n_normal=40, h1=1e-3, radius=30.0)
    p = cfd.ogrid_to_su2(X, Y, tmp_path / "o.su2", wall="airfoil", far="farfield")
    m = cfd.read_su2_mesh(p)
    assert m.n_elements == 120 * 40
    assert len(m.markers["airfoil"][cfd.VTK_LINE]) == 120
    assert len(m.markers["farfield"][cfd.VTK_LINE]) == 120
    loop = cfd.ordered_marker_loop(m, "airfoil")
    assert len(loop) == 120 and len(set(loop.tolist())) == 120


def test_first_cell_height_order():
    h = cfd.first_cell_height(6.5e6, yplus=1.0)
    assert 2e-6 < h < 8e-6


# ----------------------------------------------------------------------------- frames and mapping

def test_frame_rotation_is_proper_and_invertible():
    R = cfd.SW_TO_SU2
    assert np.linalg.det(R) == pytest.approx(1.0)
    p = np.random.default_rng(1).normal(size=(10, 3))
    assert np.allclose(cfd.su2_to_sw(cfd.sw_to_su2(p)), p)
    up = cfd.sw_to_su2(np.array([[0.0, 1.0, 0.0]]))[0]   # SuperWing y (vertical) -> SU2 z (lift axis)
    assert np.allclose(up, [0, 0, 1])


def _synthetic_wing():
    p = np.zeros(38)
    names = data.GEOM_NAMES
    vals = dict(SA=30.0, DA=5.0, DA_kink=3.0, AR=9.5, TR=0.25, kink=0.39, rootadj=0.5, tcroot=0.15)
    for k, v in vals.items():
        p[names.index(k)] = v
    for i in range(3):
        p[names.index(f"rtcs_{i}")] = 0.85
        p[names.index(f"cambers_{i}")] = 1.0
    for i in range(4):
        p[names.index(f"twists_{i}")] = -1.5
    for i in range(10):
        p[names.index(f"cst_u_{i}")] = 0.17
        p[names.index(f"cst_l_{i}")] = -0.13
    return geometry.build_surface(p)


def test_map_surface_to_grid_reproduces_coefficients():
    s = _synthetic_wing()                       # (3, 129, 257) nodes, SuperWing frame
    cc = data.cell_centres(s)                   # (3, 128, 256)
    # Cp linear in the vertical coordinate gives a lift equal to a volume integral (non-zero)
    f = lambda q: -0.6 * np.sin(np.pi * (q[0] - q[0].min()) / np.ptp(q[0])) - 20.0 * q[1] + 0.1 * q[2]
    cp_true = f(cc)
    # a "solver" point cloud: the nodes themselves (different locations from the cell centres)
    pts = s.reshape(3, -1).T
    cp_pts = f(pts.T)
    cp_map = cfd.map_surface_to_grid(pts, cp_pts, cc)
    s_ref = 1.0
    a = data.integrate_coefficients(s, cp_true, 3.0, s_ref)
    b = data.integrate_coefficients(s, cp_map, 3.0, s_ref)
    assert b[0] == pytest.approx(a[0], rel=5e-3)
    assert np.sqrt(np.mean((cp_map - cp_true) ** 2)) < 5e-3 * np.ptp(cp_true)


def test_section_wires_cover_kink_and_tip():
    s = _synthetic_wing()
    rows, kink = cfd.loft_rows(s, stride=4)
    assert rows[0] == 0 and rows[-1] == s.shape[1] - 1 and kink in rows
    assert np.all(np.diff(rows) > 0)


# ----------------------------------------------------------------------------- experimental files

def test_numeric_table_parser_skips_text():
    txt = "TITLE = cp1u\nVARIABLES = x/c, Cp\nZONE T=\"upper\"\n 0.0000  -0.2500\n  0.0100 -0.9000\n1.0  0.10\n\n"
    x, cp = cfd.parse_numeric_table(txt)
    assert np.allclose(x, [0.0, 0.01, 1.0]) and np.allclose(cp, [-0.25, -0.9, 0.1])


def test_numeric_table_parser_blocks():
    txt = "# upper\n0.0 -0.2\n0.5 -1.0\n1.0 0.1\n\n# lower\n0.0 -0.2\n0.5 0.3\n1.0 0.1\n"
    blocks = cfd.parse_numeric_blocks(txt)
    assert len(blocks) == 2 and blocks[1].shape == (3, 2)


def test_slice_wall_exact_on_triangulated_wing():
    """Planar cut of a structured, triangulated swept wing: Cp linear in x is reproduced exactly at
    the cut, chord and LE position equal the planform values."""
    ni, nj = 41, 80
    eta = np.linspace(0, 1, ni)
    th = np.linspace(0, 2 * np.pi, nj, endpoint=False)
    xi = 0.5 * (1 + np.cos(th))                      # 1 -> 0 -> 1 around the section
    b, cr, lam, sw = 1.2, 0.8, 0.56, np.radians(30)
    E, XI = np.meshgrid(eta, xi, indexing="ij")
    c = cr * (1 - (1 - lam) * E)
    t = 0.06 * c * np.sin(th)[None, :] * np.ones_like(E)
    pts = np.stack([E * b * np.tan(sw) + XI * c, E * b, t], -1).reshape(-1, 3)
    idx = np.arange(ni * nj).reshape(ni, nj)
    nx = np.roll(idx, -1, axis=1)
    tris = np.concatenate([np.stack([idx[:-1].ravel(), nx[:-1].ravel(), nx[1:].ravel()], 1),
                           np.stack([idx[:-1].ravel(), nx[1:].ravel(), idx[1:].ravel()], 1)])
    cp = 0.3 * pts[:, 0] - 0.2 + np.where(pts[:, 2] > 0, -0.5, 0.0)
    e = 0.437                                         # between grid stations
    cut = cfd.slice_wall(pts, tris, cp, e * b)
    c_e = cr * (1 - (1 - lam) * e)
    assert cut["chord"] == pytest.approx(c_e, rel=1e-9)
    assert cut["x_le"] == pytest.approx(e * b * np.tan(sw), rel=1e-9)
    xl = cut["lower"]["x_c"] * c_e + cut["x_le"]
    ok = (cut["lower"]["x_c"] > 0.02) & (cut["lower"]["x_c"] < 0.98)
    assert np.allclose(cut["lower"]["cp"][ok], 0.3 * xl[ok] - 0.2, atol=1e-9)


def test_integrate_wall_matches_grid_integration():
    """Same surface, same Cp: integration on a (randomly oriented) triangulation of the data grid
    equals conflow.data.integrate_coefficients on the grid cells."""
    s = _synthetic_wing()
    v, f = geometry.close_surface(s)
    nr = s.shape[1] * s.shape[2]
    side = f.max(axis=1) < nr                                # drop the cap triangles: grid cells only
    f = f[side]
    f = f[np.all(f // s.shape[2] < s.shape[1] - 0, axis=1)]
    rng = np.random.default_rng(3)
    flip = rng.uniform(size=len(f)) < 0.5
    f[flip] = f[flip][:, ::-1]
    q = lambda p: -0.5 * np.sin(np.pi * (p[:, 0] - p[:, 0].min()) / np.ptp(p[:, 0])) - 10 * p[:, 1]
    cp_nodes = q(v)
    w = cfd.integrate_wall(v, f, cp_nodes, 2.5, 1.0, nodes_sw=s)
    cc = data.cell_centres(s).reshape(3, -1).T
    ref = data.integrate_coefficients(s, q(cc).reshape(s.shape[1] - 1, s.shape[2] - 1), 2.5, 1.0)
    assert w["cl"] == pytest.approx(ref[0], rel=2e-3)
    assert w["cm"] == pytest.approx(ref[2], rel=5e-3)
    assert w["n_flipped"] > 0


def _sharp_te(s):
    s = s.copy()
    s[:, :, -1] = s[:, :, 0]
    return s


def test_rans_wall_closed_and_outward():
    w = cfd.wing_rans_surface(_sharp_te(_synthetic_wing()))
    P, T, rr, m = w["points"], w["tris"], w["root_ring"], w["ring"]
    fan = np.stack([np.full(m - 2, rr[0]), rr[1:-1], rr[2:]], 1)
    wt = geometry.check_watertight(P, np.concatenate([T, fan]))
    assert wt["n_open_edges"] == 0 and wt["orientation_consistent"] and wt["signed_volume"] > 0


def test_prism_layers_valid_and_root_on_plane():
    w = cfd.wing_rans_surface(_sharp_te(_synthetic_wing()), span_stride=8, chord_stride=4)
    P, T, rr = w["points"], w["tris"], w["root_ring"]
    X = cfd.extrude_prism_layers(P, T, rr, 1.2 ** np.arange(12), np.full(len(P), 1e-5), smooth=10)
    assert len(X) == 13
    assert np.allclose(X[:, rr, 1], P[rr, 1])
    for k in range(12):
        J = cfd.prism_corner_jacobians((X[k][T[:, 0]], X[k][T[:, 1]], X[k][T[:, 2]]),
                                       (X[k + 1][T[:, 0]], X[k + 1][T[:, 1]], X[k + 1][T[:, 2]]))
        assert np.min(np.stack(J)) > 0


def test_tetra_region_boundary_closed():
    w = cfd.wing_rans_surface(_sharp_te(_synthetic_wing()), span_stride=8, chord_stride=4)
    P, T, rr, m = w["points"], w["tris"], w["root_ring"], w["ring"]
    X = cfd.extrude_prism_layers(P, T, rr, 1.2 ** np.arange(10), np.full(len(P), 1e-4), smooth=10)
    L, nw = len(X) - 1, len(P)
    xr0 = P[rr, 0]
    ap, ann, dome = cfd.annulus_and_dome(X[-1][rr], P[rr[0], 1], (0.5 * (xr0.min() + xr0.max()), P[rr, 2].mean()),
                                         20.0, 30, 1e-3, 8)
    outer = L * nw + np.arange(nw)
    ids = np.concatenate([outer[rr], (L + 1) * nw + np.arange(len(ap) - m)])
    wt = geometry.check_watertight(np.concatenate([X.reshape(-1, 3), ap[m:]]),
                                   np.concatenate([outer[T][:, ::-1], ids[ann], ids[dome]]))
    assert wt["n_open_edges"] == 0 and wt["orientation_consistent"] and wt["euler_characteristic"] == 2
    assert wt["signed_volume"] > 0


def test_merge_gmsh_nodes_survives_renumbering():
    rng = np.random.default_rng(5)
    bnd = rng.normal(size=(50, 3))
    used = np.arange(10, 40)
    newp = rng.normal(size=(7, 3)) + 10
    coords = np.concatenate([bnd[used], newp])
    perm = rng.permutation(len(coords))
    tags = np.arange(1, len(coords) + 1)[perm]       # gmsh renumbered everything
    allp, lut = cfd._merge_gmsh_nodes(bnd, used, tags, coords)
    assert np.allclose(allp[lut[tags]], coords)
    assert np.all(lut[tags[:len(used)]] == used)
