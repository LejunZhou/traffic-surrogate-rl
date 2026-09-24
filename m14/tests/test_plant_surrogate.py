"""M14 GRU DeepONet: causality, shape contracts, ensemble loading,
dataset views, SurrogateVecEnv contract (torch + numpy only, no SUMO)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from surrogate.deeponet import DeepONetEnsemble, PlantNormalisation, build_plant_model, load_plant_checkpoint  # noqa: E402

K, NX = 120, 19
X_GRID = (np.arange(NX) + 1) * 100.0


def test_gru_causality():
    torch.manual_seed(0)
    m = build_plant_model({"branch": {"type": "gru", "channels": 32, "hidden": 32, "latent_dim": 64}, "trunk": {"hidden_dim": 64, "latent_dim": 64}})
    x = torch.randn(2, 2, K)
    x2 = x.clone(); x2[:, :, 50:] += 3.0
    with torch.no_grad():
        b1, b2 = m.branch(x), m.branch(x2)
    assert float((b1[:, :50] - b2[:, :50]).abs().max()) < 1e-5
    assert float((b1[:, 50:] - b2[:, 50:]).abs().max()) > 1e-3


def test_forward_shapes():
    m = build_plant_model({"branch": {"type": "gru", "hidden": 16, "latent_dim": 32}, "trunk": {"hidden_dim": 32, "latent_dim": 32}})
    rho, q = m(torch.randn(4, 2, K), torch.randint(0, K, (4, 7)), torch.rand(4, 7, 2))
    assert rho.shape == (4, 7) and q.shape == (4, 7)


def test_prefix_predictions_equal_full_history_predictions():
    """Available history length must not change an already observed field."""
    torch.manual_seed(4)
    model = build_plant_model({
        "branch": {"type": "gru", "hidden": 16, "layers": 2, "latent_dim": 32},
        "trunk": {"hidden_dim": 32, "layers": 2, "latent_dim": 32},
    }).eval()
    history = torch.randn(2, 2, K)
    query_k = torch.tensor([[0, 6, 18], [2, 9, 18]])
    query_xt = torch.rand(2, 3, 2)
    with torch.no_grad():
        full = model(history, query_k, query_xt)
        prefix = model(history[:, :, :19], query_k, query_xt)
    for a, b in zip(full, prefix):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def _make_synthetic_store(root: Path, n: int = 6, seed: int = 0) -> Path:
    """Synthetic rollouts with a simple, causal density field (for pipeline tests)."""
    from sumo_env.rollout import save_rollout_npz
    from sumo_env.rollout_store import RolloutStore

    rng = np.random.default_rng(seed)
    store = RolloutStore(root)

    class _Env:
        x_grid = X_GRID.astype(np.float32); t_grid = np.arange(K, dtype=np.float32) * 30.0; dt_ctrl = 30
        sumo_config = {"network": {"highway_length_m": 2000.0}}

    for i in range(n):
        d = np.repeat(rng.uniform(1300, 2200, 12), 10).astype(np.float32)
        qr = np.repeat(rng.uniform(200, 900, 12), 10).astype(np.float32)
        inflow = (d + qr)
        dens = np.zeros((NX, K), np.float32)
        for k in range(K):
            dens[:, k] = np.clip(inflow[max(k - 2, 0)] / 110.0 + 0.002 * inflow[:k + 1].sum() / (k + 1) - 20, 5, 140)
        arrays = {"density": dens, "speed": np.full((NX, K), 100.0, np.float32), "flow": np.tile(inflow, (NX, 1)).astype(np.float32),
                  "outflow_vph": np.roll(inflow, 2).astype(np.float32), "mainline_demand": d, "ramp_arrival": qr,
                  "ramp_inflow_vph": qr, "ramp_queue": np.zeros(K, np.float32), "pending_mainline": np.zeros(K, np.float32),
                  "action": np.full(K, 0.5, np.float32), "reward": -np.ones(K, np.float32), "q_ref": np.full(K, 2476.0, np.float32),
                  "outflow_penalty": np.zeros(K, np.float32), "queue_penalty": np.zeros(K, np.float32), "std_penalty": np.zeros(K, np.float32)}
        result = {"arrays": arrays, "metrics": {"return": -120.0, "breakdown": False}}
        meta = {"round": 0, "controller": {"type": "constant" if i % 2 else "alinea"}, "profile": {"set": "train", "index": i, "peak_total_vph": float(inflow.max())},
                "sumo_seed": i}
        path = save_rollout_npz(root / f"r0_syn_{i:03d}.npz", result, meta, env=_Env())
        store.add(path, {**meta, "metrics": result["metrics"]})
    store.save_index()
    store.make_splits(0.5, 0.25, seed=0)
    return root



def _tiny_cfg(store: Path) -> dict:
    return {"project_root": str(ROOT),
            "data": {"store_dir": str(store), "sumo_config": "configs/scenario.yaml", "n_query_points": 64, "mode": "causal",
                     "band": {"rho_min": 40, "x_max_m": 1400, "weight": 2.0}, "outflow_weight": 1.0},
            "model": {"branch": {"type": "gru", "hidden": 16, "layers": 1, "latent_dim": 32}, "trunk": {"hidden_dim": 32, "layers": 2, "latent_dim": 32}},
            "ensemble": {"members": 2, "bootstrap": True},
            "training": {"n_epochs": 2, "lr": 1e-3, "warmup_epochs": 0, "batch_size": 4, "eval_every": 1, "seed": 0, "num_threads": 1},
            "finetune": {"epochs": 1, "lr": 3e-4}, "output": {}}


@pytest.fixture(scope="module")
def tiny_ensemble(tmp_path_factory):
    from surrogate.train_plant import train_member
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_ensemble import write_manifest

    root = tmp_path_factory.mktemp("plant")
    store = _make_synthetic_store(root / "store")
    cfg = _tiny_cfg(store)
    out = root / "ens"
    for m in range(2):
        train_member(cfg, m, m, out / f"member_{m}", project_root=ROOT)
    write_manifest(out, cfg, "tiny", str(store))
    return out, store


def test_dataset_views_and_bootstrap(tiny_ensemble):
    from surrogate.datasets import PlantRolloutDataset

    _, store = tiny_ensemble
    split = json.loads((store / "split_index.json").read_text())
    norm = PlantNormalisation(split["metadata"]["mean_density"], split["metadata"]["std_density"])
    files = split["train"] + split["train"][:1]        # duplicate = bootstrap repeat
    ds = PlantRolloutDataset(store, files, norm, n_query_points=32)
    assert len(ds) == len(files) and len(ds.unique_files) == len(split["train"])
    item = ds[0]
    assert item["branch"].shape == (2, K) and item["query_k"].shape == (32,) and item["query_xt"].shape == (32, 2)
    assert item["target_q"].shape == (K,) and float(item["mask_q"].sum()) == K


def test_ensemble_load_and_predict(tiny_ensemble):
    out, _ = tiny_ensemble
    ens = DeepONetEnsemble.load(out)
    assert ens.M == 2 and ens.K == K and ens.Nx == NX
    hist = np.zeros((3, 2, K), np.float32); hist[:, 0] = 0.7; hist[:, 1, :10] = 0.3
    rho, q = ens.predict_step(hist, np.array([0, 5, 9]), np.array([0, 1, 0]))
    assert rho.shape == (3, NX) and q.shape == (3,) and np.all(np.isfinite(rho))
    rho_all, q_all = ens.predict_all_members_step(hist, np.array([0, 5, 9]))
    assert rho_all.shape == (2, 3, NX)
    assert np.allclose(rho_all[0, 0], rho[0]) and np.allclose(rho_all[1, 1], rho[1])
    rf, qf = ens.predict_full(hist[:1])
    assert rf.shape == (2, 1, NX, K) and qf.shape == (2, 1, K)
    # full-field prediction agrees with the per-step read-out
    assert np.allclose(rf[0, 0, :, 5], ens.predict_step(hist[:1], np.array([5]), np.array([0]))[0][0], atol=1e-4)
    m, ckpt, norm = load_plant_checkpoint(out / "member_0" / "best.pt")
    assert ckpt["member"] == 0 and len(ckpt["bootstrap_files"]) > 0


def test_finetune_keeps_bootstrap_and_adds_new_rounds(tiny_ensemble, tmp_path):
    from surrogate.train_plant import train_member
    from sumo_env.rollout import load_rollout_npz, save_rollout_npz
    from sumo_env.rollout_store import RolloutStore

    out, store = tiny_ensemble
    st = RolloutStore(store)
    arrays, meta = load_rollout_npz(store / st.entries[0]["file"])
    meta = {**meta, "round": 1, "controller": {"type": "policy"}}

    class _Env:
        x_grid = X_GRID.astype(np.float32); t_grid = np.arange(K, dtype=np.float32) * 30.0; dt_ctrl = 30
        sumo_config = {"network": {"highway_length_m": 2000.0}}

    p = save_rollout_npz(store / "r1_policy_000.npz", {"arrays": arrays, "metrics": meta["metrics"]}, meta, env=_Env())
    st.append_round([(p, meta)])
    cfg = _tiny_cfg(store)
    best = train_member(cfg, 0, None, tmp_path / "m0", resume=out / "member_0" / "best.pt", finetune=True, new_rounds=[1], project_root=ROOT)
    files = json.loads((tmp_path / "m0" / "train_files.json").read_text())["train"]
    assert "r1_policy_000.npz" in files
    ck = torch.load(out / "member_0" / "best.pt", weights_only=False)
    assert files[: len(ck["bootstrap_files"])] == ck["bootstrap_files"]


def test_vec_env_contract(tiny_ensemble):
    from rl.surrogate_vec_env import SurrogateVecEnv

    out, _ = tiny_ensemble
    cfg = {"project_root": str(ROOT), "ensemble_dir": str(out), "n_envs": 4, "ensemble_mode": "sample", "seed": 1,
           "profiles": {"family": "configs/demand.yaml"}, "sumo_config": "configs/scenario.yaml",
           "reward": {"delta": 3.5, "beta": 1.0, "gamma": 0.06, "q_ref": 2476, "q_ref_mode": "offered", "queue_norm": 400, "sigma_ref": 6, "warmup_s": 90},
           "symmetric_action": True, "observation": {"lookahead_steps": 0}}
    env = SurrogateVecEnv(cfg)
    obs = env.reset()
    assert obs.shape == (4, NX + 4) and env.action_space.low[0] == -1.0
    total = np.zeros(4)
    for k in range(K):
        obs, rew, done, infos = env.step(np.zeros((4, 1), np.float32))   # a = 0 -> u = 0.5
        total += rew
        assert obs.shape == (4, NX + 4) and np.all(np.isfinite(obs))
        assert all(abs(i["u"] - 0.5) < 1e-6 for i in infos)
        for key in ("density", "outflow_vph", "q_ref", "queue_after", "ramp_inflow_vph", "mainline_demand_vph", "ramp_arrival_vph", "ensemble_member", "backend"):
            assert key in infos[0]
        if k < K - 1:
            assert not done.any()
    assert done.all() and all("terminal_observation" in i for i in infos)
    assert np.all(total <= 0.0)
    # warm-up mask: first 3 steps (90 s) carry zero reward
    env.reset()
    _, rew, _, infos = env.step(np.zeros((4, 1), np.float32))
    assert np.all(rew == 0.0) and infos[0]["reward_warmup_active"] == 1.0
    # mean / pessimistic modes run and pessimistic <= mean-mode reward on average
    cfg_mean = {**cfg, "ensemble_mode": "mean"}; cfg_pess = {**cfg, "ensemble_mode": "pessimistic", "pessimistic_kappa": 1.0}
    em, ep = SurrogateVecEnv(cfg_mean), SurrogateVecEnv(cfg_pess)
    em.reset(); ep.reset()
    rm = rp = 0.0
    for _ in range(20):
        rm += em.step(np.zeros((4, 1), np.float32))[1].sum(); rp += ep.step(np.zeros((4, 1), np.float32))[1].sum()
    assert rp <= rm + 1e-6
    # fixed-set evaluation helper
    from sumo_env.demand_profiles import load_profile_set

    val = load_profile_set(ROOT / "configs/profiles/val.json")[:4]
    res = em.rollout_policy(lambda o: np.zeros((4, 1), np.float32), val)
    assert len(res) == 4 and all("tts_veh_h" in r for r in res)


def test_queue_recursion_matches_sumoenv_formula():
    from sumo_env.ramp_queue import MeteredRampQueue, analytic_queue_step

    q_after, released, arrivals = analytic_queue_step(10.0, 0.5, 800.0, 1600.0, 30.0)
    assert arrivals == pytest.approx(800 * 30 / 3600) and released == pytest.approx(min(10 + arrivals, 0.5 * 1600 * 30 / 3600))
    assert q_after == pytest.approx(10 + arrivals - released)
    q = MeteredRampQueue(400.0, 1600.0, 1.0)
    rel = 0
    for s in range(600):
        n = q.step(0.0, arrival_vph=400.0 if s < 300 else 1000.0)
        q.on_released(n); rel += n
    assert rel == 0 and 110 <= q.queue <= 118     # 300 s at 400 vph (33) + 300 s at 1000 vph (83)


def test_surrogate_uses_scenario_meter_and_unbounded_queue(tiny_ensemble):
    from rl.surrogate_vec_env import SurrogateVecEnv

    out, _ = tiny_ensemble
    env = SurrogateVecEnv({
        "project_root": str(ROOT), "sumo_config": "configs/scenario.yaml",
        "ensemble_dir": str(out), "n_envs": 1, "ensemble_mode": "mean",
        "profiles": {"family": "configs/demand.yaml"},
    })
    assert env.ramp_discharge_vph == 1200.0
    assert env.ramp_queue_max_veh is None
    env.close()


def test_sumo_and_surrogate_env_parity(tiny_ensemble, tmp_path):
    """Observation layout, info keys and queue / reward bookkeeping agree between
    SumoEnv (profile mode) and SurrogateVecEnv (needs the sumo binary)."""
    import shutil

    pytest.importorskip("traci")
    if shutil.which("sumo") is None:
        pytest.skip("sumo binary not on PATH")
    from rl.surrogate_vec_env import SurrogateVecEnv
    from rl.sumo_env_wrapper import SumoEnv
    from sumo_env.demand_profiles import load_profile_set

    out, _ = tiny_ensemble
    reward = {"form": "tts", "q_ref": 2476.0, "q_ref_mode": "offered", "queue_norm": 400.0, "sigma_ref": 6.0, "warmup_s": 90}
    common = {"project_root": str(ROOT), "sumo_config": "configs/scenario.yaml", "ramp_discharge_vph": 1200.0, "density_mean": 20.0, "density_std": 16.0, "queue_norm_scale": 100.0,
              "observation": {"lookahead_steps": 2}, "reward": reward, "profiles": {"set": "configs/profiles/val.json"}}
    sumo = SumoEnv({**common, "sumo_config": "configs/scenario.yaml", "network_dir": str(tmp_path / "network")})
    vec = SurrogateVecEnv({**common, "ensemble_dir": str(out), "n_envs": 1, "ensemble_mode": "mean"})
    p = load_profile_set(ROOT / "configs/profiles/val.json")[3]
    obs_s, _ = sumo.reset(options={"profile": p, "sumo_seed": 1})
    vec.set_profiles([p]); obs_v = vec.reset()[0]
    assert obs_s.shape == obs_v.shape == (NX + 4 + 4,)
    # identical demand / look-ahead / time / queue features at reset (densities are 0 in both)
    assert np.allclose(obs_s[NX:], obs_v[NX:])
    try:
        for k in range(4):
            u = np.array([0.4], np.float32)
            obs_s, r_s, _, _, info_s = sumo.step(u)
            obs_v, r_v, _, infos = vec.step(u.reshape(1, 1)); info_v = infos[0]; obs_v = obs_v[0]
            for key in ("density", "outflow_vph", "q_ref", "queue_after", "ramp_released", "ramp_inflow_vph", "mainline_demand_vph",
                        "ramp_arrival_vph", "on_road_veh", "backlog_veh", "tts_step_veh_h", "reward_warmup_active", "u", "k"):
                assert key in info_s and key in info_v, key
            assert info_s["mainline_demand_vph"] == info_v["mainline_demand_vph"]
            assert info_s["q_ref"] == pytest.approx(info_v["q_ref"], rel=0.05)     # both offered-demand references
            # demand look-ahead / time features identical; queue follows the same recursion (SUMO's is integer-rounded)
            assert np.allclose(obs_s[NX:NX + 6], obs_v[NX:NX + 6])
            assert abs(info_s["queue_after"] - info_v["queue_after"]) <= 1.5
            if k < 3:
                assert r_s == 0.0 and r_v[0] == 0.0    # warm-up mask
    finally:
        sumo.close()


def test_cached_step_matches_full_history_step():
    """The incremental GRU gives predict_step's outputs, also after a member switch and a row reset."""
    torch.manual_seed(5)
    cfg = {"branch": {"type": "gru", "hidden": 16, "layers": 2, "latent_dim": 32}, "trunk": {"hidden_dim": 32, "layers": 2, "latent_dim": 32}}
    members = [build_plant_model(cfg).eval() for _ in range(3)]
    ens = DeepONetEnsemble(members, PlantNormalisation(20.0, 15.0), {"model": cfg}, X_GRID, 2000.0, 3600.0, 30.0)
    rng = np.random.default_rng(0)
    n = 5
    hist = np.zeros((n, 2, K), np.float32); hist[:, 0] = rng.random((n, K))
    member = rng.integers(0, 3, n)
    cache = ens.new_branch_cache(n)
    for k in range(K):
        if k == 40:
            member[1] = (member[1] + 1) % 3                     # member switch mid-history: rebuilt from the prefix
        if k == 70:                                             # row 2 starts a new history at step 0
            hist[2] = 0.0; hist[2, 0] = rng.random(K); cache.reset_rows(2)
        hist[:, 1, k] = rng.random(n)
        kv = np.full(n, k); kv[2] = k - 70 if k >= 70 else k
        r_full, q_full = ens.predict_step(hist, kv, member)
        r_inc, q_inc = ens.predict_step_cached(hist, kv, member, cache)
        np.testing.assert_allclose(r_inc, r_full, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(q_inc, q_full, rtol=1e-5, atol=1e-2)
    r_all, q_all = ens.predict_all_members_step(hist, kv)
    r_all_inc, q_all_inc = ens.predict_all_members_step_cached(hist, kv, ens.new_branch_cache(n))
    np.testing.assert_allclose(r_all_inc, r_all, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(q_all_inc, q_all, rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("mode", ["sample", "mean", "pessimistic"])
def test_vec_env_incremental_branch_matches_full_recompute(tiny_ensemble, mode):
    """Environment trajectories with the GRU cache equal those that rerun the branch over the history."""
    from rl.surrogate_vec_env import SurrogateVecEnv

    out, _ = tiny_ensemble
    cfg = {"project_root": str(ROOT), "ensemble_dir": str(out), "n_envs": 3, "ensemble_mode": mode, "seed": 2,
           "profiles": {"family": "configs/demand.yaml"}, "sumo_config": "configs/scenario.yaml",
           "reward": {"delta": 3.5, "beta": 1.0, "gamma": 0.06, "q_ref": 2476, "q_ref_mode": "offered", "queue_norm": 400, "sigma_ref": 6, "warmup_s": 90},
           "symmetric_action": True, "observation": {"lookahead_steps": 0}}
    fast, slow = SurrogateVecEnv(cfg), SurrogateVecEnv({**cfg, "incremental_branch": False})
    assert fast._branch_cache is not None and slow._branch_cache is None
    np.testing.assert_allclose(fast.reset(), slow.reset(), rtol=1e-5, atol=1e-5)
    rng = np.random.default_rng(3)
    for _ in range(K + 30):                                     # crosses an episode boundary (slot resets)
        a = rng.uniform(-1, 1, (3, 1)).astype(np.float32)
        of, rf, df, _ = fast.step(a)
        os_, rs, ds, _ = slow.step(a)
        np.testing.assert_allclose(of, os_, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(rf, rs, rtol=1e-4, atol=1e-4)
        assert (df == ds).all()
