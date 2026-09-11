"""
M3 OOD spot-check at 1500 vph with a Fourier-series ramp control.

The control family is *not* in M3's training distribution (which only includes
constant, piecewise_constant, smooth, and ramp_step). This tests whether the
surrogate generalizes to a structurally different control class.
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
from surrogate.deeponet import BranchNet, TrunkNet, DeepONet
from utils.plotting import plot_density_heatmap


OUT_DIR = _ROOT / "data" / "raw" / "m3_check_1500vph"
PLOT_DIR = _ROOT / "runs" / "surrogate" / "deeponet_constant_inflow_20260511_234849" / "eval_m3_spotcheck_1500vph"
CKPT_PATH = _ROOT / "runs" / "surrogate" / "deeponet_constant_inflow_20260511_234849" / "best.pt"
SUMO_CFG = _ROOT / "configs" / "sumo" / "phase1_1.yaml"


def sample_fourier(T_ctrl: int, rng: np.random.Generator) -> np.ndarray:
    """u(t) = 0.5 + sum_k a_k sin(2pi*k*t/T + phi_k), clipped to [0, 1]."""
    n_modes = 5
    amplitudes = np.array([0.18, 0.12, 0.08, 0.05, 0.04], dtype=np.float32)  # ~Σ=0.47
    phases = rng.uniform(0.0, 2 * np.pi, size=n_modes).astype(np.float32)
    freqs = np.arange(1, n_modes + 1, dtype=np.float32)  # 1, 2, 3, 4, 5 cycles/horizon
    t_norm = np.arange(T_ctrl, dtype=np.float32) / T_ctrl  # [0, 1)
    u = 0.5 * np.ones(T_ctrl, dtype=np.float32)
    for a, k, p in zip(amplitudes, freqs, phases):
        u = u + a * np.sin(2 * np.pi * k * t_norm + p)
    return np.clip(u, 0.0, 1.0).astype(np.float32)


def build_sumo_scenario_1500(seed: int):
    cfg = load_config(str(SUMO_CFG))
    cfg = merge_configs(cfg, {
        "demand": {"mainline_demand_vph": 1500},
        "simulation": {"seed": seed},
        "output": {"raw_dir": str(OUT_DIR.relative_to(_ROOT))},
    })
    network_dir = _ROOT / cfg["output"]["network_dir"]
    files = build_network(str(network_dir), cfg)
    det_file = str(network_dir / "detectors.add.xml")
    build_detector_file(det_file, cfg)
    return cfg, files["net"], files["route"], det_file


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
    return np.maximum(pred_norm * std + mean, 0.0)


def main():
    rng = np.random.default_rng(2026)
    u = sample_fourier(120, rng)
    print(f"Fourier control: min={u.min():.3f}, max={u.max():.3f}, mean={u.mean():.3f}")
    print(f"u[:10]  = {u[:10]}")
    print(f"u[60:70]= {u[60:70]}")

    cfg, net, route, det = build_sumo_scenario_1500(seed=301)
    result = run_simulation(net_file=net, route_file=route, detector_file=det,
                            ramp_control=u, config=cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "sim_m3chk_fourier_0000.npz"
    np.savez(str(out_path),
             density=result["density"], speed=result["speed"], flow=result["flow"],
             x_grid=result["x_grid"], t_grid=result["t_grid"],
             mainline_demand=result["mainline_demand"], ramp_control=result["ramp_control"],
             seed=np.array(result["metadata"]["seed"]),
             mainline_demand_vph=np.array(result["metadata"]["mainline_demand_vph"]),
             ramp_demand_vph=np.array(result["metadata"]["ramp_demand_vph"]))
    rho_true = result["density"].astype(np.float32)
    print(f"\nSUMO done: rho_mean={rho_true.mean():.2f}  rho_max={rho_true.max():.2f}  saved → {out_path.name}")

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    mean = ckpt["normalization"]["mean_density"]
    std = ckpt["normalization"]["std_density"]
    model = DeepONet(
        BranchNet(int(mc.get("branch_input_dim", 120)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
        TrunkNet(int(mc.get("trunk_input_dim", 2)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
    ).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    pred = predict(model, mean, std, u, result["x_grid"], result["t_grid"])
    diff = pred - rho_true
    rel = float(np.linalg.norm(diff) / max(np.linalg.norm(rho_true), 1e-8))
    rmse = float(np.sqrt(np.mean(diff**2)))
    print(f"\nFourier OOD test — rel_L2 = {rel:.4f},  RMSE = {rmse:.3f} veh/km")
    print("M3 in-distribution mean rel_L2: 0.0738 / 4-family spot-check: 0.0843")
    print("Acceptance gate: 0.15")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOT_DIR / "fourier_heatmap.png"
    plot_density_heatmap(
        predicted=pred, true=rho_true,
        x_grid=result["x_grid"], t_grid=result["t_grid"],
        output_path=out_png,
    )
    print(f"saved heatmap: {out_png.relative_to(_ROOT)}")

    # Also save a control plot so the user can see the input
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(result["t_grid"], u, lw=1.5)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("u(t)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Fourier-series ramp control (5 modes, k=1..5)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        ctrl_png = PLOT_DIR / "fourier_control.png"
        fig.savefig(ctrl_png, dpi=120)
        plt.close(fig)
        print(f"saved control: {ctrl_png.relative_to(_ROOT)}")
    except Exception as e:
        print(f"[warn] control plot skipped: {e}")


if __name__ == "__main__":
    main()
