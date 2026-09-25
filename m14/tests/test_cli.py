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


def test_seeds_dry_run_trains_new_seeds_and_evaluates_all(tmp_path):
    """`seeds` trains only the new seeds but evaluates and tabulates seed 0 with them."""
    project = tmp_path / "standalone"
    project.mkdir()
    shutil.copyfile(ROOT / "run.py", project / "run.py")
    proc = subprocess.run(
        [sys.executable, "-I", "-B", str(project / "run.py"), "seeds", "--new-seeds", "1", "2", "--torch-threads", "4", "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    commands = [shlex.split(line[2:]) for line in proc.stdout.splitlines() if line.startswith("$ ")]
    loops = [cmd for cmd in commands if "scripts/run_aggregation_loop.py" in cmd]
    assert [cmd[cmd.index("--study") + 1] for cmd in loops] == ["m14_s1", "m14_s2"]
    direct = [cmd for cmd in commands if "rl.train_ppo" in cmd]
    assert sorted(cmd[cmd.index("--seed") + 1] for cmd in direct) == ["1", "2"]
    manifest = next(cmd for cmd in commands if "scripts/build_arms_manifest.py" in cmd)
    assert manifest[manifest.index("--seeds") + 1:manifest.index("--direct-ee")] == ["0", "1", "2"]
    assert "m14_s0" in manifest[manifest.index("--mpc-spec") + 1]        # MPC stays on seed 0's ensemble (reused)
    tables = next(cmd for cmd in commands if "scripts/build_paper_tables.py" in cmd)
    assert tables[tables.index("--seeds") + 1:tables.index("--sets")] == ["0", "1", "2"]


def test_branches_give_each_branch_its_own_seed(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("m14_test_cli_branches", ROOT / "run.py")
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    args = cli.parser().parse_args(["seeds", "--workers", "3"])
    cli._run_branches({"surrogate_s1": [("surrogate-ppo", 1)], "direct_s2": [("sumo-ppo", 2)]}, args)
    by_stage = {cmd[2]: cmd for cmd in calls}
    assert by_stage["surrogate-ppo"][by_stage["surrogate-ppo"].index("--seed") + 1] == "1"
    assert by_stage["sumo-ppo"][by_stage["sumo-ppo"].index("--seed") + 1] == "2"
    assert all(cmd[cmd.index("--workers") + 1] == "3" for cmd in calls)
    assert (tmp_path / "runs/logs/pipeline_surrogate_s1.log").exists()


def test_extend_direct_dry_run_continues_every_seed_and_tabulates_both_budgets(tmp_path):
    project = tmp_path / "standalone"
    project.mkdir()
    shutil.copyfile(ROOT / "run.py", project / "run.py")
    proc = subprocess.run(
        [sys.executable, "-I", "-B", str(project / "run.py"), "extend-direct", "--to-budget", "1200", "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    commands = [shlex.split(line[2:]) for line in proc.stdout.splitlines() if line.startswith("$ ")]
    direct = [cmd for cmd in commands if "rl.train_ppo" in cmd]
    assert sorted(cmd[cmd.index("--seed") + 1] for cmd in direct) == ["0", "1", "2"]
    assert all(cmd[cmd.index("--total-timesteps") + 1] == "144000" for cmd in direct)
    assert all("output.run_dir=runs/study/m14/direct_ppo_1200ee_s" in " ".join(cmd) for cmd in direct)
    manifest = next(cmd for cmd in commands if "scripts/build_arms_manifest.py" in cmd)
    assert manifest[manifest.index("--direct-ee") + 1:manifest.index("--direct-ee") + 3] == ["1000", "1200"]
    tables = next(cmd for cmd in commands if "scripts/build_paper_tables.py" in cmd)
    assert tables[tables.index("--seeds") + 1:tables.index("--sets")] == ["0", "1", "2"]


def test_continue_direct_run_copies_the_finished_run(tmp_path, monkeypatch):
    import zipfile

    spec = importlib.util.spec_from_file_location("m14_test_cli_extend", ROOT / "run.py")
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "runs/study/m14/direct_ppo_1000ee_s1"
    (source / "checkpoints").mkdir(parents=True)
    (source / "eval").mkdir()
    (source / "checkpoints/ppo_sumo_115200_steps.zip").write_bytes(b"ckpt")
    for name in ("best_model_selected.zip", "best_model.zip", "eval/selection.json", "eval/evaluations.npz"):
        (source / name).write_bytes(b"x")
    with zipfile.ZipFile(source / "final_model.zip", "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": 120000}))
    (tmp_path / "runs/ledger").mkdir(parents=True)
    rows = [{"study": "m14_direct_1000_s1", "purpose": "direct_ppo", "wall_s": 16.0}] * 3
    (tmp_path / "runs/ledger/m14_direct_1000_s1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    cli._continue_direct_run(1, 1000, 1200, dry=False)
    target = tmp_path / "runs/study/m14/direct_ppo_1200ee_s1"
    assert (target / "checkpoints/ppo_sumo_120000_steps.zip").read_bytes() == (source / "final_model.zip").read_bytes()
    assert (target / "checkpoints/ppo_sumo_115200_steps.zip").exists() and (target / "eval/evaluations.npz").exists()
    assert not any((target / name).exists() for name in ("final_model.zip", "best_model_selected.zip", "eval/selection.json"))
    ledger = [json.loads(line) for line in (tmp_path / "runs/ledger/m14_direct_1200_s1.jsonl").read_text().splitlines()]
    assert len(ledger) == 3 and all(r["study"] == "m14_direct_1200_s1" for r in ledger)
    assert json.loads((target / "continued_from.json").read_text())["steps"] == 120000
    (target / "marker").write_text("continued")
    cli._continue_direct_run(1, 1000, 1200, dry=False)                  # a started continuation is left as it is
    assert (target / "marker").exists()
