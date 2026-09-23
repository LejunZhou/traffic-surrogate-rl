"""Direct-PPO resume bookkeeping, paper-table maths and the E0 capacity comparison (no SUMO needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    if str(PROJECT_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / sub))

from rl.train_ppo import checkpoint_step, latest_checkpoint, prepare_resume, restore_eval_history  # noqa: E402


def _interrupted_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "direct"
    (run / "checkpoints").mkdir(parents=True)
    (run / "eval").mkdir()
    for step in (120, 240):
        (run / "checkpoints" / f"ppo_sumo_{step}_steps.zip").write_bytes(b"")
    np.savez(run / "eval" / "evaluations.npz", timesteps=np.array([120, 240, 360]),
             results=np.array([[-50.0], [-40.0], [-45.0]]), ep_lengths=np.array([[120], [120], [120]]))
    (run / "progress.csv").write_text("a\n")
    (run / "monitor.csv").write_text("b\n")
    ledger = tmp_path / "ledger.jsonl"
    rows = ([{"purpose": "direct_ppo", "i": i} for i in range(5)] + [{"purpose": "eval_val", "i": i} for i in range(3)]
            + [{"purpose": "tuning", "i": 0}])
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, ledger


def test_latest_checkpoint_by_step_not_name(tmp_path):
    run, _ = _interrupted_run(tmp_path)
    (run / "checkpoints" / "ppo_sumo_1200_steps.zip").write_bytes(b"")     # sorts before 240 as a string
    assert checkpoint_step(latest_checkpoint(run, "sumo")) == 1200
    assert latest_checkpoint(tmp_path / "empty", "sumo") is None


def test_prepare_resume_cuts_back_to_the_checkpoint(tmp_path):
    run, ledger = _interrupted_run(tmp_path)
    info = prepare_resume(run, steps=240, episode_len=120, n_eval_episodes=1, ledger_path=ledger)
    assert info["history"]["timesteps"].tolist() == [120, 240]            # the evaluation at 360 is after the checkpoint
    assert info["n_train_episodes"] == 2 and info["ledger_rows_lost"] == 4
    kept = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["purpose"] for r in kept].count("direct_ppo") == 2
    assert [r["purpose"] for r in kept].count("eval_val") == 2
    assert [r["purpose"] for r in kept].count("tuning") == 1              # other purposes untouched
    lost = [json.loads(line) for line in (run / "ledger_lost.jsonl").read_text().splitlines()]
    assert len(lost) == 4 and all(r["lost_in_part"] == 1 for r in lost)
    assert (run / "progress_part1.csv").exists() and not (run / "progress.csv").exists()
    assert (run / "monitor_part1.csv").exists() and (run / "eval" / "evaluations_part1.npz").exists()
    # a second interruption numbers its archives part 2
    (run / "progress.csv").write_text("c\n")
    assert prepare_resume(run, steps=240, episode_len=120, n_eval_episodes=1, ledger_path=ledger)["part"] == 2


def test_restore_eval_history_sets_best():
    cb = SimpleNamespace(evaluations_timesteps=[], evaluations_results=[], evaluations_length=[], best_mean_reward=-np.inf)
    history = {"timesteps": np.array([120, 240]), "results": np.array([[-50.0, -52.0], [-40.0, -42.0]]),
               "ep_lengths": np.array([[120, 120], [120, 120]])}
    restore_eval_history(cb, history)
    assert cb.evaluations_timesteps == [120, 240] and cb.best_mean_reward == pytest.approx(-41.0)
    restore_eval_history(cb, None)                                        # no-op
    assert cb.evaluations_timesteps == [120, 240]


def test_paired_reduction_matches_episodes():
    from build_paper_tables import paired_reduction

    a = [{"profile_set": "test", "profile_index": i, "sumo_seed": 100, "tts_veh_h": 9.0} for i in range(4)]
    b = [{"profile_set": "test", "profile_index": i, "sumo_seed": 100, "tts_veh_h": 10.0} for i in range(3, -1, -1)]
    b.append({"profile_set": "test", "profile_index": 9, "sumo_seed": 100, "tts_veh_h": 99.0})   # unmatched, ignored
    res = paired_reduction(a, b, n_boot=200)
    assert res["n_pairs"] == 4 and res["reduction_pct"] == pytest.approx(10.0)
    assert res["tts_saved_veh_h"] == pytest.approx(1.0)


def test_compare_e0_capacity_verdicts():
    from compare_e0_capacity import compare

    def report(first):
        return {"meter_discharge_vph": 1200.0,
                "gate": {str(i): {"peak_total": 2500.0 + i, "breakdown_u_first": u, "best_constant_u": "0.4"}
                         for i, u in enumerate(first)}}

    old = report([0.5, 0.6, 0.5, None])
    assert compare(old, report([0.5, 0.6, 0.5, None]))["mean_shift_vph"] == 0
    half = compare(old, report([0.4, 0.5, 0.5, 0.3]))                        # two of three one step down: -80 veh/h
    assert half["n_down"] == 2 and not half["rescale_recommended"] and "half to one" in half["verdict"]
    assert compare(old, report([0.4, 0.5, 0.4, None]))["rescale_recommended"]  # every profile one step down
    with pytest.raises(ValueError):
        compare(old, {"meter_discharge_vph": 1200.0, "gate": {"0": {"peak_total": 2000.0, "breakdown_u_first": 0.5}}})


def test_interrupted_evaluation_request_is_moved_aside(tmp_path):
    from rl.profile_eval import _move_interrupted_request

    out = tmp_path / "alinea_test.jsonl"
    sidecar, summary = out.with_suffix(".request.json"), out.with_suffix(".summary.json")
    rollouts = tmp_path / "rollouts" / "alinea_test"
    rollouts.mkdir(parents=True)
    (rollouts / "ep0.npz").write_bytes(b"")
    sidecar.write_text(json.dumps({"status": "running"}))
    _move_interrupted_request(out, sidecar, summary, rollouts)
    moved = list(tmp_path.glob("alinea_test.interrupted_*"))
    assert not sidecar.exists() and not rollouts.exists() and len(moved) == 1
    assert (moved[0] / "alinea_test.request.json").exists() and (moved[0] / "rollouts" / "ep0.npz").exists()
    sidecar.write_text(json.dumps({"status": "complete"}))                 # completed requests are never touched
    _move_interrupted_request(out, sidecar, summary, None)
    assert sidecar.exists()


def test_interrupted_aggregation_round_is_moved_aside(tmp_path, monkeypatch):
    import run_aggregation_loop as agg
    from sumo_env.rollout_store import RolloutStore

    monkeypatch.setattr(agg, "PROJECT_ROOT", tmp_path)
    study_dir = tmp_path / "runs/aggregation/m14_s0"
    (study_dir / "ppo_r2").mkdir(parents=True)
    (study_dir / "ppo_r1").mkdir()
    (study_dir / "agg_r2.request.json").write_text("{}")
    store = RolloutStore(study_dir / "store")
    store.entries = [{"file": "a.npz", "round": 0, "split": "train"}, {"file": "b.npz", "round": 2, "split": "train"}]
    store.save_index()
    ledger = tmp_path / "runs/ledger/m14_s0.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("".join(json.dumps({"round": r}) + "\n" for r in (1, 1, 2)))
    monkeypatch.setattr(RolloutStore, "density_stats", lambda self, files: {"mean_density": 0.0, "std_density": 1.0})
    aside = agg.recover_interrupted_round(study_dir, "m14_s0", completed=1)
    assert (aside / "ppo_r2").exists() and (aside / "agg_r2.request.json").exists() and (study_dir / "ppo_r1").exists()
    assert [e["file"] for e in RolloutStore(study_dir / "store").entries] == ["a.npz"]
    assert [json.loads(line)["round"] for line in ledger.read_text().splitlines()] == [1, 1]
    assert len((aside / "ledger_lost.jsonl").read_text().splitlines()) == 1


def test_alinea_edge_check():
    from tune_alinea_profiles import edge_check

    grid = {"dets": [12, 13, 14, 15], "rhos": [20, 23, 26, 30, 34], "kis": [10, 20, 35], "kps": [2, 4, 8]}
    assert edge_check("pialinea:kp=4,ki=20,rho=26,det=13", grid) == {}
    edges = edge_check("pialinea:kp=8,ki=20,rho=20,det=14", grid)
    assert set(edges) == {"kp", "rho"} and "lowest" in edges["rho"] and "highest" in edges["kp"]
    assert edge_check("alinea:ki=20,rho=26,det=13", {"dets": [13], "rhos": [26], "kis": [20]}) == {}   # single values
