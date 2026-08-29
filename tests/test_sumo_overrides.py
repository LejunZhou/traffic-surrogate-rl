"""CLI scenario-override parsing for the SUMO eval scripts (no SUMO needed)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eval_sumo_baselines import apply_sumo_overrides  # noqa: E402


def test_overrides_are_typed_and_nested():
    cfg = apply_sumo_overrides({}, [
        "vehicle.depart_speed=desired",
        "simulation.max_depart_delay_s=5",
        "vehicle.speed_dev=0.03",
        "simulation.sumo_extra_args=--extrapolate-departpos --no-warnings",
    ])
    assert cfg["sumo_overrides"]["vehicle"] == {"depart_speed": "desired", "speed_dev": 0.03}
    assert cfg["sumo_overrides"]["simulation"] == {
        "max_depart_delay_s": 5,
        "sumo_extra_args": ["--extrapolate-departpos", "--no-warnings"],
    }


def test_overrides_merge_into_existing_block():
    cfg = {"sumo_overrides": {"vehicle": {"speed_dev": 0.1}}}
    apply_sumo_overrides(cfg, ["vehicle.depart_speed=max"])
    assert cfg["sumo_overrides"]["vehicle"] == {"speed_dev": 0.1, "depart_speed": "max"}


def test_none_is_noop():
    assert apply_sumo_overrides({"a": 1}, None) == {"a": 1}


@pytest.mark.parametrize("bad", ["depart_speed=desired", "vehicle.depart_speed"])
def test_bad_spec_raises(bad):
    with pytest.raises(ValueError):
        apply_sumo_overrides({}, [bad])
