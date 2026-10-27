"""RT1 / A2 case kinds (a2_dataset, a2_rebuilt): same dataset wing and condition for both, the dataset surface or the
rebuilt surface meshed, reference area = the dataset case's s_ref_half. Written before the A2 runs. The network
(Hugging Face) and the mesher are replaced by stand-ins here; the runner runs the real thing."""
import numpy as np
import pytest

from conflow import cfd, data, geometry
from test_t4_design_case import design_params
from test_rt1_mesh_options import _Spy


def test_match_dataset_wing():
    P = np.stack([design_params(), design_params() + 0.1, design_params() - 0.1])
    assert cfd.match_dataset_wing(P, design_params() + 0.1) == 1
    with pytest.raises(ValueError):
        cfd.match_dataset_wing(P, design_params() + 0.05)               # no row
    with pytest.raises(ValueError):
        cfd.match_dataset_wing(np.vstack([P, P[:1]]), design_params())  # two rows


def test_dataset_condition_rule():
    rows = np.array([0, 1, 1, 1, 1])
    mach = np.array([0.80, 0.78, 0.85, 0.78, 0.779])
    aoa = np.array([2.0, 3.0, 4.0, 5.0, 9.0])
    assert cfd.dataset_condition(rows, mach, aoa, 1, 0.78, 4.6) == 3        # Mach first, then AoA
    assert cfd.dataset_condition(rows, mach, aoa, 1, 0.84, 0.0) == 2
    with pytest.raises(ValueError):
        cfd.dataset_condition(rows, mach, aoa, 7, 0.8, 3.0)


class _FakeNpy:
    surf = None

    def __init__(self, url):
        self.url = url

    def read(self, i0, i1, channel=None):
        if "origingeom" in self.url:
            return self.surf[None]
        return np.zeros((1, 64, 128))


@pytest.fixture
def fake_dataset(monkeypatch):
    p = design_params()
    s = geometry.build_surface(p, te_thickness=0.0) + 1e-3          # stands in for origingeom (different, sharp TE)
    _FakeNpy.surf = s
    ir = {c: 0.0 for c in data.INDEX_COLUMNS}
    ir.update(mach=0.7812, aoa=3.21, s_ref_half=1.987)
    ds = dict(wing_row=5, wing_id=17, case_index=40, index_row=ir, hf_repo="r", hf_revision="v", params=p,
              mach_target=0.78, aoa_target=3.0)
    monkeypatch.setattr(cfd, "_a2_dataset", lambda case, work, prep: dict(ds))
    monkeypatch.setattr(data, "NpySource", _FakeNpy)
    spy = _Spy()
    monkeypatch.setattr(cfd, "wing_mesh_3d", spy)
    return spy, s, p


def _case(kind, **mesh):
    return dict(name="a2_test", kind=kind, solver="euler", mesh=dict(type="euler_tetra", size_factor=1.0, **mesh),
                flow=dict(ref_length=1.0, moment_ref=[0.25, 0.0, 0.0]), markers=dict(wall="wing", far="farfield", sym="symmetry"),
                hf=dict(repo="r", revision="v"), a2=dict(params=design_params().tolist(), mach_target=0.78, aoa_target=3.0))


def test_a2_pair_same_condition_different_surface(tmp_path, fake_dataset):
    spy, s_data, p = fake_dataset
    er = dict(h0=0.003, growth=0.15)
    pd = cfd.prepare_case(_case("a2_dataset", edge_refine=er), tmp_path / "d")
    pr = cfd.prepare_case(_case("a2_rebuilt", edge_refine=er), tmp_path / "r")
    assert pd["flow"] == pr["flow"] == dict(mach=0.7812, aoa=3.21, ref_area=1.987)
    np.testing.assert_array_equal(spy.calls[0][0], s_data)
    np.testing.assert_array_equal(spy.calls[1][0], geometry.build_surface(p))
    assert spy.calls[0][1]["edge_refine"] == er and spy.calls[1][1]["edge_refine"] == er
    assert pd["a2"]["wing_row"] == pr["a2"]["wing_row"] == 5
    cfg = cfd.case_config(_case("a2_rebuilt", edge_refine=er), pr)
    assert float(cfg["MACH_NUMBER"]) == 0.7812 and float(cfg["AOA"]) == 3.21 and float(cfg["REF_AREA"]) == 1.987


def test_a2_dataset_rejects_sharp_te(tmp_path, fake_dataset):
    spy, _, _ = fake_dataset
    with pytest.raises(ValueError):
        cfd.prepare_case(_case("a2_dataset", sharp_te=True), tmp_path)
    assert not spy.calls
