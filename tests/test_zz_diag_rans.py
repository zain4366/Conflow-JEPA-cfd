"""Temporary diagnostic (Task 4 RANS mesh): tetra algorithm x graded transition on a coarse built wing. Reports
through the skip message (pytest -rs); never fails. Removed after the choice is made."""
import os
import time

import numpy as np
import pytest

from conflow import cfd, geometry


@pytest.mark.parametrize("alg,slope", [(1, None), (1, 0.2), (4, 0.2), (10, None), (10, 0.2)])
def test_diag(tmp_path, alg, slope):
    if not os.environ.get("CONFLOW_SLOW"):
        pytest.skip("diag needs CONFLOW_SLOW")
    from test_t4_design_case import design_params
    s = cfd.sharpen_te(geometry.build_surface(design_params()))
    t0 = time.time()
    try:
        st = cfd.wing_mesh_3d_rans(s, tmp_path / "m.su2", reynolds=2e7, span_stride=8, chord_stride=4, radius=20.0,
                                   n_radial=30, n_rings=8, last_layer_chords=0.01, algorithm3d=alg, n_threads=2,
                                   transition_slope=slope)
        m = cfd.read_su2_mesh(tmp_path / "m.su2")
        v = cfd.tet_volumes(m)
        msg = (f"DIAG alg={alg} slope={slope} t={time.time() - t0:.0f}s tets={st['n_tets']} prisms={st['n_prisms']} "
               f"nonpos={int((v <= 0).sum())} zero={int((v == 0).sum())} vmin={v.min():.3e} vabsmin={np.abs(v).min():.3e} "
               f"layers={st['n_layers']}/{st['n_layers_planned']}")
    except Exception as e:
        msg = f"DIAG alg={alg} slope={slope} t={time.time() - t0:.0f}s ERROR {type(e).__name__}: {str(e)[:200]}"
    pytest.skip(msg)
