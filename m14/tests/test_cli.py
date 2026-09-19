"""CLI isolation and argument propagation, without running training or SUMO."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def files_under(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_pipeline_dry_run_propagates_seed_and_writes_nothing(tmp_path):
    """The seed reaching training must also select that seed's final arms."""
    project = tmp_path / "standalone"
    project.mkdir()
    shutil.copyfile(ROOT / "run.py", project / "run.py")
    outside = tmp_path / "caller"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("do not change")
    before = files_under(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-I", "-B", str(project / "run.py"), "pipeline", "--seed", "1", "--dry-run"],
        cwd=outside, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    commands = [shlex.split(line[2:]) for line in proc.stdout.splitlines() if line.startswith("$ ")]
    manifest = next(cmd for cmd in commands if "scripts/build_arms_manifest.py" in cmd)
    assert manifest[manifest.index("--seeds") + 1:manifest.index("--direct-ee")] == ["1"]
    aggregation = next(cmd for cmd in commands if "scripts/run_aggregation_loop.py" in cmd)
    assert aggregation[aggregation.index("--seed") + 1] == "1"
    assert aggregation[aggregation.index("--study") + 1] == "m14_s1"
    direct = [cmd for cmd in commands if "rl.train_ppo" in cmd]
    assert direct and all(cmd[cmd.index("--seed") + 1] == "1" for cmd in direct)
    assert files_under(tmp_path) == before


def test_simulate_ignores_inherited_scenario_overlay_in_process(tmp_path, monkeypatch):
    """`os.environ.update` alone cannot remove an inherited external overlay."""
    from rl import sumo_env_wrapper
    from sumo_env import rollout

    project = tmp_path / "standalone"
    shutil.copytree(ROOT / "configs", project / "configs")
    overlay = tmp_path / "other_experiment.yaml"
    overlay.write_text("env:\n  ramp_discharge_vph: 9999\n")
    spec = importlib.util.spec_from_file_location("m14_test_cli", ROOT / "run.py")
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "ROOT", project)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("SCENARIO_OVERLAY", str(overlay))
    monkeypatch.setenv("PROFILE_SETS_DIR", str(tmp_path / "missing_parent_profiles"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py", "simulate", "--seed", "37"])
    captured = {}

    class FakeEnv:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def close(self):
            captured["closed"] = True

    def fake_rollout(env, controller, reset_options):
        captured["seed"] = reset_options["sumo_seed"]
        return {"metrics": {"return": -1.0}, "arrays": {}}

    def fake_save(path, result, meta, env):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test rollout")
        captured["path"] = path

    monkeypatch.setattr(sumo_env_wrapper, "SumoEnv", FakeEnv)
    monkeypatch.setattr(rollout, "rollout_episode", fake_rollout)
    monkeypatch.setattr(rollout, "save_rollout_npz", fake_save)
    cli.main()
    assert captured["cfg"]["ramp_discharge_vph"] == 1200.0
    assert Path(captured["cfg"]["project_root"]) == project
    assert captured["seed"] == 37 and captured["closed"]
    assert captured["path"] == project / "runs/simulation/rollout.npz"
    assert json.loads(captured["path"].with_suffix(".json").read_text()) == {"return": -1.0}
    assert overlay.read_text() == "env:\n  ramp_discharge_vph: 9999\n"
