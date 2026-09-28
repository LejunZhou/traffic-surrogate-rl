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
    assert txt.count("<flow ") == 13
    assert 'begin="480" end="780"' in txt
    assert 'departSpeed="desired"' in txt
    _write_routes(out, cfg)
    assert out.read_text().count("<flow ") == 2


def test_frozen_sets_load_and_match_sampler():
    val = load_profile_set(ROOT / "configs/profiles/val.json")
    test = load_profile_set(ROOT / "configs/profiles/test.json")
    ood = load_profile_set(ROOT / "configs/profiles/ood.json")
    assert len(val) == 18 and len(test) == 30 and len(ood) == 12
    assert all(len(p.sumo_seeds) == 1 for p in val) and all(len(p.sumo_seeds) == 3 for p in test)
    fam = ProfileFamily.load(ROOT / "configs/scenario.yaml")
    regenerated = fam.sample_by_key("test", 7)
    assert np.allclose(regenerated.mainline_blocks, test[7].mainline_blocks)
    assert all(p.params["evaluation_kind"] == "fixed_schedule" for p in ood)
    all_seeds = [seed for group in (val, test, ood) for p in group for seed in p.sumo_seeds]
    assert len(all_seeds) == len(set(all_seeds))


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


def test_entire_study_reads_exact_scenario_schedule(tmp_path):
    from sumo_env.demand_profiles import build_fixed_sets
    scenario_path = ROOT / 'configs/scenario.yaml'
    family = ProfileFamily.load(scenario_path)
    assert family.is_fixed
    p = family.sample_by_key('train', 0)
    assert p.block_min == .5 and p.K == 120
    expected_main = np.r_[np.full(20,1250), np.full(20,1600), np.arange(1600,1800,20),
                          np.full(40,1800), np.arange(1800,1100,-70), np.full(20,1100)]
    expected_ramp = np.r_[np.full(20,250), np.full(20,400), np.arange(400,900,50),
                          np.full(30,900), np.arange(900,300,-60), np.full(30,300)]
    for set_name in ('train', 'e0', 'tune', 'agg', 'val', 'test', 'ood'):
        for i in (0, 11):
            profile = family.sample_by_key(set_name, i)
            np.testing.assert_array_equal(profile.mainline_vph, expected_main)
            np.testing.assert_array_equal(profile.ramp_vph, expected_ramp)
    assert p.params['storage_window_min'] == [20, 45]
    for cfg_name in ('dataset', 'ppo'):
        cfg = yaml.safe_load((ROOT / f'configs/{cfg_name}.yaml').read_text())
        assert cfg['env']['profiles']['family'] == 'configs/scenario.yaml'
    generated = build_fixed_sets(scenario_path, tmp_path/'sets')
    for name in ('val', 'test', 'ood'):
        actual = load_profile_set(ROOT / f'configs/profiles/{name}.json')
        assert len(generated[name]) == len(actual)
        for a, b in zip(generated[name], actual):
            np.testing.assert_array_equal(a.mainline_vph, expected_main)
            np.testing.assert_array_equal(a.ramp_vph, expected_ramp)
            assert a.sumo_seeds == b.sumo_seeds


def test_scenario_edits_propagate_to_evaluation_without_regeneration(tmp_path):
    import json
    cfg = yaml.safe_load((ROOT / 'configs/scenario.yaml').read_text())
    cfg['demand']['profile_segments'][0]['mainline_vph'] = 1400
    (tmp_path/'scenario.yaml').write_text(yaml.safe_dump(cfg))
    path = tmp_path/'test.json'
    path.write_text(json.dumps(dict(scenario='scenario.yaml', set='test', n=1, seeds_per_profile=3, seed_base=20000)))
    p = load_profile_set(path)[0]
    np.testing.assert_array_equal(p.mainline_vph[:20], 1400)
    assert p.sumo_seeds == [20000, 20001, 20002]
