"""Scientific evaluation outputs can be reused only for the exact same request."""

from pathlib import Path
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest

from rl import profile_eval
from sumo_env.demand_profiles import DemandProfile
from sumo_env.rollout import save_rollout_npz


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    scenario = tmp_path / "configs/scenario.yaml"
    scenario.parent.mkdir()
    scenario.write_text("demand:\n  ramp_discharge_vph: 1200\n")
    state = SimpleNamespace(calls=[], logs=[], root=tmp_path, scenario=scenario)

    class FakeLedger:
        def __init__(self, study, root):
            pass

        def log(self, *args, **kwargs):
            state.logs.append((args, kwargs))

    def fake_jobs(jobs, env_cfg, **kwargs):
        state.calls.append(copy.deepcopy(jobs))
        results = []
        for job in jobs:
            metrics = {
                "return": -float(job["sumo_seed"]), "breakdown": False, "recovered": False,
                "tts_veh_h": 12.0, "served_veh": 50.0, "final_queue": 1.0,
                "mean_action": 0.5, "wall_s": 0.01,
            }
            meta = {k: v for k, v in job.items() if k != "out_dir"}
            meta["controller"] = {"type": "policy", "policy": job["policy"]}
            meta["metrics"] = metrics
            path = None
            if job["out_dir"]:
                path = save_rollout_npz(
                    Path(job["out_dir"]) / f"{job['name']}.npz",
                    {"metrics": metrics, "arrays": {"density": np.zeros((1, 1), np.float32)}},
                    meta,
                )
            results.append({"ok": True, "name": job["name"], "metrics": metrics,
                            "path": str(path) if path else None, "meta": meta, "arrays": None})
        return results

    monkeypatch.setattr(profile_eval, "run_jobs", fake_jobs)
    monkeypatch.setattr(profile_eval, "Ledger", FakeLedger)
    state.request = {
        "policies": ["u=0.3", "u=0.7"],
        "profiles": [DemandProfile.constant(1500, 400, set_name="val", index=0, sumo_seeds=[7, 8])],
        "env_cfg": {"project_root": str(tmp_path), "sumo_config": "configs/scenario.yaml",
                    "density_mean": 20.0, "density_std": 12.0, "reward": {"form": "tts", "tts_scale": 1.0}},
        "out_jsonl": tmp_path / "eval/episodes.jsonl", "workers": 1,
        "project_root": tmp_path, "purpose": "eval_val", "study": "unit_eval", "quiet": True,
    }
    return state


def snapshot(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_exact_evaluation_repeat_reuses_rows_and_ledger(evaluation):
    e = evaluation
    first = profile_eval.evaluate_on_profiles(**e.request)
    before = snapshot(e.root)
    second = profile_eval.evaluate_on_profiles(**{**e.request, "workers": 3, "quiet": False})
    assert len(e.calls) == 1 and len(e.logs) == 4
    assert second["rows"] == first["rows"]
    assert second["summary"]["n_episodes"] == 4
    assert len(e.request["out_jsonl"].read_text().splitlines()) == 4
    assert snapshot(e.root) == before


@pytest.mark.parametrize("change", ["policy", "seeds", "profile", "reward", "scenario_bytes"])
def test_changed_evaluation_request_is_refused(evaluation, change):
    e = evaluation
    profile_eval.evaluate_on_profiles(**e.request)
    changed = copy.deepcopy(e.request)
    if change == "policy":
        changed["policies"][0] = "u=0.2"
    elif change == "seeds":
        changed["seeds"] = [17, 18]
    elif change == "profile":
        # Same profile identifier; its actual demand is nevertheless different.
        changed["profiles"][0].mainline_blocks[0] += 10
    elif change == "reward":
        changed["env_cfg"]["reward"]["tts_scale"] = 2.0
    elif change == "scenario_bytes":
        e.scenario.write_text("demand:\n  ramp_discharge_vph: 900\n")
    before = snapshot(e.root)
    with pytest.raises(RuntimeError):
        profile_eval.evaluate_on_profiles(**changed)
    assert len(e.calls) == 1 and len(e.logs) == 4
    assert snapshot(e.root) == before


def test_changed_checkpoint_bytes_at_same_path_are_refused(evaluation):
    e = evaluation
    checkpoint = e.root / "policy.zip"
    checkpoint.write_bytes(b"first policy contents")
    e.request["policies"] = [str(checkpoint)]
    profile_eval.evaluate_on_profiles(**e.request)
    checkpoint.write_bytes(b"different policy contents")
    with pytest.raises(RuntimeError):
        profile_eval.evaluate_on_profiles(**e.request)
    assert len(e.calls) == 1 and len(e.logs) == 2


@pytest.mark.parametrize("damage", ["missing_request", "running_request", "truncated", "duplicate", "malformed"])
def test_incomplete_or_unverifiable_outputs_are_refused(evaluation, damage):
    e = evaluation
    profile_eval.evaluate_on_profiles(**e.request)
    output = e.request["out_jsonl"]
    request_file = output.with_suffix(".request.json")
    lines = output.read_text().splitlines()
    if damage == "missing_request":
        request_file.unlink()
    elif damage == "running_request":
        request = json.loads(request_file.read_text())
        request["status"] = "running"
        request_file.write_text(json.dumps(request))
    elif damage == "truncated":
        output.write_text("\n".join(lines[:-1]) + "\n")
    elif damage == "duplicate":
        output.write_text("\n".join([*lines[:-1], lines[0]]) + "\n")
    elif damage == "malformed":
        output.write_text("not JSON\n")
    before = snapshot(e.root)
    with pytest.raises(RuntimeError):
        profile_eval.evaluate_on_profiles(**e.request)
    assert len(e.calls) == 1 and len(e.logs) == 4
    assert snapshot(e.root) == before


def test_saved_rollout_reuse_restores_aggregation_metadata(evaluation):
    e = evaluation
    e.request["save_rollouts_dir"] = e.root / "rollouts"
    first = profile_eval.evaluate_on_profiles(**e.request)
    second = profile_eval.evaluate_on_profiles(**e.request)
    assert len(e.calls) == 1 and len(e.logs) == 4
    assert len(second["results"]) == 4
    for previous, cached in zip(first["results"], second["results"]):
        assert cached["path"] == previous["path"]
        assert cached["meta"]["profile"] == previous["meta"]["profile"]
        assert cached["meta"]["round"] == previous["meta"]["round"]
        assert cached["metrics"] == previous["metrics"]
        assert cached["arrays"] is None
    Path(second["results"][0]["path"]).unlink()
    with pytest.raises(RuntimeError):
        profile_eval.evaluate_on_profiles(**e.request)
    assert len(e.calls) == 1
