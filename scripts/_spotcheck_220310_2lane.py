"""
Test the 220310 checkpoint on 2-lane 2000-vph rollouts.

Hypothesis: 220310 was trained on the older 2-lane variant of phase1_1.yaml.
If we regenerate test rollouts on a 2-lane network at 2000 vph and the
resulting dataset stats match 220310's normalization (mean≈65, std≈68), we've
identified the original dataset recipe. The model should then predict well.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from utils.config import load_config, merge_configs
from sumo_env.network_builder import build_network
from sumo_env.detectors import build_detector_file
from sumo_env.run_simulation import run_simulation
from sumo_env.dataset_generation import (
    _sample_piecewise_constant, _sample_smooth, _sample_ramp_step,
)
from surrogate.deeponet import BranchNet, TrunkNet, DeepONet
from utils.plotting import plot_density_heatmap


OUT_DIR = _ROOT / "data" / "raw" / "spotcheck_2lane_2000vph"
RUN_DIR = _ROOT / "runs" / "surrogate" / "deeponet_constant_inflow_20260511_220310"
PLOT_DIR = RUN_DIR / "eval_spotcheck_2lane_2000vph"
CKPT_PATH = RUN_DIR / "best.pt"

# 2-lane network params, recovered from git history (commit 7416008, before
# f2bb92f flipped to 1-lane). Demand bumped to 2000 to match 220310's
# constant_mainline_demand_vph override.
SCENARIO_OVERRIDE = {
    "network": {
        "highway_length_m": 2000.0,
        "ramp_position_m": 500.0,
        "ramp_length_m": 200.0,
        "speed_limit_mps": 33.33,
        "ramp_speed_limit_mps": 16.67,
        "num_lanes": 2,
        # no acceleration_lane_length_m -> defaults to 0 (none)
    },
    "demand": {
        "mainline_demand_vph": 2000,
        "ramp_demand_vph": 600,
        "demand_profile": "constant",
    },
    "simulation": {
        "duration_s": 3600,
        "step_length_s": 1.0,
        "dt_ctrl_s": 30,
        "sumo_binary": "sumo",
    },
    "vehicle": {"idm_tau_s": 1.0},
    "detectors": {
        "n_detectors": 19,
        "spacing_m": 100.0,
        "start_position_m": 100.0,
        "vehicle_length_m": 5.0,
    },
    "output": {
        "raw_dir": str(OUT_DIR.relative_to(_ROOT)),
        "network_dir": str((_ROOT / "data" / "raw" / "network_2lane").relative_to(_ROOT)),
    },
}


def sample_fourier(T_ctrl, rng):
    n_modes = 5
    a = np.array([0.18, 0.12, 0.08, 0.05, 0.04], dtype=np.float32)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_modes).astype(np.float32)
    freqs = np.arange(1, n_modes + 1, dtype=np.float32)
    t_norm = np.arange(T_ctrl, dtype=np.float32) / T_ctrl
    u = 0.5 * np.ones(T_ctrl, dtype=np.float32)
    for ai, k, p in zip(a, freqs, phases):
        u = u + ai * np.sin(2 * np.pi * k * t_norm + p)
    return np.clip(u, 0.0, 1.0).astype(np.float32)


def build_scenario(seed):
    base = {"network": {}, "demand": {}, "simulation": {}, "vehicle": {},
            "detectors": {}, "output": {}}
    cfg = merge_configs(base, SCENARIO_OVERRIDE)
    cfg["simulation"]["seed"] = seed
    network_dir = _ROOT / cfg["output"]["network_dir"]
    files = build_network(str(network_dir), cfg)
    det_file = str(network_dir / "detectors.add.xml")
    build_detector_file(det_file, cfg)
    return cfg, files["net"], files["route"], det_file


def run_one(family, control, seed, save_name):
    cfg, net, route, det = build_scenario(seed)
    result = run_simulation(net_file=net, route_file=route, detector_file=det,
                            ramp_control=control.astype(np.float32), config=cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{save_name}.npz"
    np.savez(str(out_path),
             density=result["density"], speed=result["speed"], flow=result["flow"],
             x_grid=result["x_grid"], t_grid=result["t_grid"],
             mainline_demand=result["mainline_demand"], ramp_control=result["ramp_control"],
             seed=np.array(result["metadata"]["seed"]),
             mainline_demand_vph=np.array(result["metadata"]["mainline_demand_vph"]),
             ramp_demand_vph=np.array(result["metadata"]["ramp_demand_vph"]))
    d = result["density"]
    print(f"[{family:<19}] seed={seed:>3}  rho_mean={d.mean():6.2f}  rho_max={d.max():7.2f}  u_mean={control.mean():.3f}  → {out_path.name}")
    return {"family": family, "control": control,
            "density_true": d.astype(np.float32),
            "x_grid": result["x_grid"], "t_grid": result["t_grid"]}


def predict(model, mean, std, u, x_grid, t_grid):
    Nx, Nt = x_grid.size, t_grid.size
    x_norm = x_grid.astype(np.float32) / 2000.0
    t_norm = t_grid.astype(np.float32) / 3600.0
    xg, tg = np.meshgrid(x_norm, t_norm, indexing="ij")
    trunk = np.stack([xg.ravel(), tg.ravel()], axis=-1).astype(np.float32)
    with torch.no_grad():
        branch = torch.from_numpy(u.astype(np.float32)).unsqueeze(0)
        trunk_t = torch.from_numpy(trunk).unsqueeze(0)
        pred_norm = model(branch, trunk_t).cpu().numpy().reshape(Nx, Nt)
    pred_unclip = pred_norm * std + mean
    return np.maximum(pred_unclip, 0.0), pred_unclip


def metric(pred, true):
    diff = pred - true
    rel = float(np.linalg.norm(diff) / max(np.linalg.norm(true), 1e-8))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return rel, rmse


def main():
    rng = np.random.default_rng(42)
    rollouts = []
    rollouts.append(run_one("constant", np.full(120, 0.5, dtype=np.float32), 501, "sim_2lane_const_0000"))
    rollouts.append(run_one("piecewise_constant", _sample_piecewise_constant(120, rng), 502, "sim_2lane_pwc_0000"))
    rollouts.append(run_one("smooth", _sample_smooth(120, rng), 503, "sim_2lane_smooth_0000"))
    rollouts.append(run_one("ramp_step", _sample_ramp_step(120, rng), 504, "sim_2lane_rampstep_0000"))
    rollouts.append(run_one("fourier", sample_fourier(120, np.random.default_rng(2026)), 505, "sim_2lane_fourier_0000"))

    # Dataset stats — compare to 220310's normalization (mean=65.32, std=67.58)
    all_rho = np.concatenate([r["density_true"].ravel() for r in rollouts])
    print(f"\nGenerated dataset stats (5 rollouts):")
    print(f"  rho_mean: {all_rho.mean():.3f}, rho_std: {all_rho.std():.3f}")
    print(f"  220310's training norm: mean=65.315, std=67.583")
    print()

    # Inference
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    mean = float(ckpt["normalization"]["mean_density"])
    std = float(ckpt["normalization"]["std_density"])
    model = DeepONet(
        BranchNet(int(mc.get("branch_input_dim", 120)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
        TrunkNet(int(mc.get("trunk_input_dim", 2)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
    ).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'family':<20} {'rho_mean':>9} {'rho_max':>8} {'u_mean':>7} {'rel_L2':>8} {'RMSE':>8} {'min_unclip':>11}")
    rel_l2s, rmses = [], []
    for r in rollouts:
        pred, pred_unclip = predict(model, mean, std, r["control"], r["x_grid"], r["t_grid"])
        rel, rmse = metric(pred, r["density_true"])
        rel_l2s.append(rel); rmses.append(rmse)
        print(f"{r['family']:<20} {r['density_true'].mean():>9.2f} {r['density_true'].max():>8.2f} {r['control'].mean():>7.3f} {rel:>8.4f} {rmse:>8.3f} {pred_unclip.min():>11.2f}")

        plot_density_heatmap(predicted=pred, true=r["density_true"],
                              x_grid=r["x_grid"], t_grid=r["t_grid"],
                              output_path=PLOT_DIR / f"{r['family']}_heatmap.png")
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(r["t_grid"], r["control"], lw=1.5)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("u(t)")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
        ax.set_title(f"{r['family']} (2-lane, 2000 vph) — rel_L2 = {rel:.3f}, RMSE = {rmse:.2f}")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{r['family']}_control.png", dpi=120)
        plt.close(fig)

    print(f"\nMean rel-L2 across 5 families: {np.mean(rel_l2s):.4f}")
    print(f"Mean RMSE  across 5 families: {np.mean(rmses):.3f} veh/km")
    print("Acceptance gate: 0.15")
    print(f"\nFor reference on the SAME model 220310:")
    print(f"  on 1-lane 2000-vph (current): rel-L2 1.02, RMSE 49.4  ← bad (dist mismatch)")
    print(f"  on its own val (norm):        val_mse 0.097 → RMSE ≈ {np.sqrt(0.097 * 67.583**2):.2f}")


if __name__ == "__main__":
    main()
