"""Profile sampler determinism, block expansion, route XML (no SUMO needed)."""

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sumo_env.demand_profiles import DemandProfile, ProfileFamily, load_profile_set  # noqa: E402
from sumo_env.network_builder import _write_routes  # noqa: E402

FAMILY = ROOT / "configs" / "demand.yaml"


def test_sampler_is_deterministic_by_set_and_index():
    fam = ProfileFamily.load(FAMILY)
    a = fam.sample_by_key("val", 3)
    b = ProfileFamily.load(FAMILY).sample_by_key("val", 3)
    assert np.array_equal(a.mainline_blocks, b.mainline_blocks)
    assert np.array_equal(a.ramp_blocks, b.ramp_blocks)
    c = fam.sample_by_key("val", 4)
    assert not (np.array_equal(a.mainline_blocks, c.mainline_blocks) and np.array_equal(a.ramp_blocks, c.ramp_blocks))
    d = fam.sample_by_key("test", 3)
    assert not np.array_equal(a.mainline_blocks, d.mainline_blocks)


def test_family_bounds_and_drain_tail():
    fam = ProfileFamily.load(FAMILY)
    for i in range(300):
        p = fam.sample_by_key("train", i)
        assert p.K == 120 and p.mainline_vph.shape == (120,) and p.ramp_vph.shape == (120,)
        assert p.mainline_blocks.max() <= 2300.0 + 1e-3
        assert p.ramp_blocks.max() <= 800.0 + 1e-3
        assert p.mainline_blocks.min() >= 1200.0 - 1e-3
        # D11 drain tail: the raised-cosine peak is over by minute 45 (last 3 blocks flat);
        # a step plateau ends by minute 45 and its 10-min ramp-down by 55 (last block at base)
        fam_m = p.params["mainline"]["family"]
        if fam_m == "peak":
            assert np.ptp(p.mainline_blocks[9:]) < 5.0, p.params
        elif fam_m == "step":
            assert p.params["mainline"]["t2"] <= 45.0 + 1e-6
            assert abs(p.mainline_blocks[11] - p.params["mainline"]["d_base"]) < 5.0, p.params


def test_block_expansion_and_flow_blocks():
    p = DemandProfile(np.arange(12, dtype=np.float32) * 100 + 1200, np.full(12, 400.0, np.float32))
    assert p.steps_per_block == 10
    assert p.mainline_vph[0] == 1200 and p.mainline_vph[10] == 1300 and p.mainline_vph[119] == 2300
    blocks = p.mainline_flow_blocks()
    assert blocks[0] == (0.0, 300.0, 1200.0) and blocks[-1] == (3300.0, 3600.0, 2300.0)


def test_routes_xml_one_flow_per_block(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/scenario.yaml").read_text(encoding="utf-8"))
    p = DemandProfile(np.linspace(1300, 2200, 12).astype(np.float32), np.full(12, 500.0, np.float32))
    out = tmp_path / "routes.rou.xml"
    _write_routes(out, cfg, p.mainline_flow_blocks())
    txt = out.read_text()
    assert txt.count("<flow ") == 12
    assert 'begin="300" end="600"' in txt
    assert 'departSpeed="desired"' in txt
    _write_routes(out, cfg)
    assert out.read_text().count("<flow ") == 1


def test_frozen_sets_load_and_match_sampler():
    val = load_profile_set(ROOT / "configs/profiles/val.json")
    test = load_profile_set(ROOT / "configs/profiles/test.json")
    ood = load_profile_set(ROOT / "configs/profiles/ood.json")
    assert len(val) == 18 and len(test) == 30 and len(ood) == 12
    assert all(len(p.sumo_seeds) == 1 for p in val) and all(len(p.sumo_seeds) == 3 for p in test)
    fam = ProfileFamily.load(FAMILY)
    regenerated = fam.sample_by_key("test", 7)
    assert np.allclose(regenerated.mainline_blocks, test[7].mainline_blocks)
    assert {p.params["ood"] for p in ood} == {"double", "plateau", "early_surge"}


def test_roundtrip_dict():
    fam = ProfileFamily.load(FAMILY)
    p = fam.sample_by_key("val", 1)
    q = DemandProfile.from_dict(p.to_dict())
    assert np.array_equal(p.mainline_vph, q.mainline_vph) and q.set_name == "val" and q.index == 1


def test_m14_demand_constraints():
    """M14 demand: everything over by minute 45, ramp surge never above 800 veh/h."""
    fam = ProfileFamily.load(FAMILY)
    for i in range(500):
        p = fam.sample_by_key("train", i)
        mp, rp = p.params["mainline"], p.params["ramp"]
        assert p.ramp_blocks.max() <= 800.0 + 1e-3
        if mp["family"] == "peak":
            assert mp["t_center"] + mp["half_width"] <= 45.0 + 1e-6
        elif mp["family"] == "step":
            assert mp["t2"] + 10.0 <= 45.0 + 1e-6 and mp["t2"] >= mp["t1"] + 10.0 - 1e-6
        if rp["family"] == "surge":
            assert rp["t_start"] + rp["length"] <= 45.0 + 1e-6 and rp["t_start"] >= 0.0
        # last three blocks flat at base for both streams
        assert np.ptp(p.mainline_blocks[9:]) < 5.0 and np.ptp(p.ramp_blocks[9:]) < 5.0
    for i in range(30):
        p = fam.sample_ood(fam.rng_for("ood", i), ["double", "plateau", "early_surge"][i % 3], "ood", i)
        assert p.ramp_blocks.max() <= 800.0 + 1e-3
