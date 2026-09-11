"""
5-family spot-check of the *failed* 2000-vph surrogate (5/14 run).

Generates one fresh rollout per control family at 2000 vph (constant,
piecewise_constant, smooth, ramp_step, fourier-series OOD), runs the
new best.pt on each, saves heatmaps + per-rollout metrics.

The expectation is that this model fails — the point is to *see* how it
fails per family, not to pass acceptance.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from utils.config import load_config, merge_configs
from sumo_env.network_builder import build_network
from sumo_env.detectors import build_detector_file
from sumo_env.run_simulation import run_simulation
from sumo_env.dataset_generation import (
    _sample_piecewise_constant,
    _sample_smooth,
    _sample_ramp_step,
)
from surrogate.deeponet import BranchNet, TrunkNet, DeepONet
from utils.plotting import plot_density_heatmap


OUT_DIR = _ROOT / "data" / "raw" / "spotcheck_2000vph"
RUN_DIR = _ROOT / "runs" / "surrogate" / "deeponet_constant_inflow_20260514_120640"
PLOT_DIR = RUN_DIR / "eval_spotcheck_2000vph"
CKPT_PATH = RUN_DIR / "best.pt"
SUMO_CFG = _ROOT / "configs" / "sumo" / "phase1_1.yaml"
MAINLINE_DEMAND = 2000


def sample_fourier(T_ctrl: int, rng: np.random.Generator) -> np.ndarray:
    n_modes = 5
    amplitudes = np.array([0.18, 0.12, 0.08, 0.05, 0.04], dtype=np.float32)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_modes).astype(np.float32)
    freqs = np.arange(1, n_modes + 1, dtype=np.float32)
    t_norm = np.arange(T_ctrl, dtype=np.float32) / T_ctrl
    u = 0.5 * np.ones(T_ctrl, dtype=np.float32)
    for a, k, p in zip(amplitudes, freqs, phases):
        u = u + a * np.sin(2 * np.pi * k * t_norm + p)
    return np.clip(u, 0.0, 1.0).astype(np.float32)


def build_scenario(seed: int):
    cfg = load_config(str(SUMO_CFG))
    cfg = merge_configs(cfg, {
        "demand": {"mainline_demand_vph": MAINLINE_DEMAND},
        "simulation": {"seed": seed},
        "output": {"raw_dir": str(OUT_DIR.relative_to(_ROOT))},
    })
    network_dir = _ROOT / cfg["output"]["network_dir"]
    files = build_network(str(network_dir), cfg)
    det_file = str(network_dir / "detectors.add.xml")
    build_detector_file(det_file, cfg)
    return cfg, files["net"], files["route"], det_file


def run_one(family: str, control: np.ndarray, seed: int, save_name: str) -> dict:
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
    pred_unclipped = pred_norm * std + mean
    return np.maximum(pred_unclipped, 0.0), pred_unclipped


def metric(pred, true):
    diff = pred - true
    rel = float(np.linalg.norm(diff) / max(np.linalg.norm(true), 1e-8))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return rel, rmse


def main():
    rng = np.random.default_rng(42)
    rollouts = []

    # constant: u = 0.5
    rollouts.append(run_one("constant", np.full(120, 0.5, dtype=np.float32),
                             seed=401, save_name="sim_2000_const_0000"))

    # piecewise_constant
    rollouts.append(run_one("piecewise_constant", _sample_piecewise_constant(120, rng),
                             seed=402, save_name="sim_2000_pwc_0000"))

    # smooth
    rollouts.append(run_one("smooth", _sample_smooth(120, rng),
                             seed=403, save_name="sim_2000_smooth_0000"))

    # ramp_step
    rollouts.append(run_one("ramp_step", _sample_ramp_step(120, rng),
                             seed=404, save_name="sim_2000_rampstep_0000"))

    # fourier (OOD)
    rng_f = np.random.default_rng(2026)
    rollouts.append(run_one("fourier", sample_fourier(120, rng_f),
                             seed=405, save_name="sim_2000_fourier_0000"))

    # ── Inference ──────────────────────────────────────────────────────────────
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    mean = ckpt["normalization"]["mean_density"]
    std = ckpt["normalization"]["std_density"]
    print(f"\nLoaded 5/14 ckpt: hidden={mc['hidden_dim']}, mean={mean:.3f}, std={std:.3f}, epoch={ckpt['epoch']}\n")

    model = DeepONet(
        BranchNet(int(mc.get("branch_input_dim", 120)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
        TrunkNet(int(mc.get("trunk_input_dim", 2)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
    ).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"{'family':<20} {'rho_mean':>9} {'rho_max':>8} {'u_mean':>7} {'rel_L2':>8} {'RMSE':>8} {'min_unclip':>11}")
    for r in rollouts:
        pred, pred_unclipped = predict(model, mean, std, r["control"], r["x_grid"], r["t_grid"])
        rel, rmse = metric(pred, r["density_true"])
        r["density_pred"] = pred
        r["density_pred_unclipped"] = pred_unclipped
        r["rel_l2"] = rel
        r["rmse"] = rmse
        print(f"{r['family']:<20} {r['density_true'].mean():>9.2f} {r['density_true'].max():>8.2f} {r['control'].mean():>7.3f} {rel:>8.4f} {rmse:>8.3f} {pred_unclipped.min():>11.2f}")

    mean_rel = float(np.mean([r["rel_l2"] for r in rollouts]))
    print(f"\nMean rel-L2 across 5 families: {mean_rel:.4f}")
    print("Acceptance gate: 0.15  |  M3 1500-vph mean rel-L2 across same 5 families: ~0.08")

    # ── Heatmaps + control plots ───────────────────────────────────────────────
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    for r in rollouts:
        plot_density_heatmap(
            predicted=r["density_pred"], true=r["density_true"],
            x_grid=r["x_grid"], t_grid=r["t_grid"],
            output_path=PLOT_DIR / f"{r['family']}_heatmap.png",
        )
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(r["t_grid"], r["control"], lw=1.5)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("u(t)")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
        ax.set_title(f"{r['family']} (2000 vph) — rel_L2 = {r['rel_l2']:.3f}, RMSE = {r['rmse']:.2f}")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{r['family']}_control.png", dpi=120)
        plt.close(fig)
        print(f"  saved: {(PLOT_DIR / (r['family'] + '_heatmap.png')).relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
