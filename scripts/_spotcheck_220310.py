"""
Run the 5-family + Fourier spot-check using the older `_220310` checkpoint
(hidden=256, mean=65.3, std=67.6 — trained on a different 2000-vph dataset)
against the 5 fresh 2000-vph rollouts we already generated.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from surrogate.deeponet import BranchNet, TrunkNet, DeepONet
from utils.plotting import plot_density_heatmap

DATA_DIR = _ROOT / "data" / "raw" / "spotcheck_2000vph"
RUN_DIR = _ROOT / "runs" / "surrogate" / "deeponet_constant_inflow_20260511_220310"
PLOT_DIR = RUN_DIR / "eval_spotcheck_2000vph"
CKPT_PATH = RUN_DIR / "best.pt"

ROLLOUTS = [
    ("constant",           "sim_2000_const_0000.npz"),
    ("piecewise_constant", "sim_2000_pwc_0000.npz"),
    ("smooth",             "sim_2000_smooth_0000.npz"),
    ("ramp_step",          "sim_2000_rampstep_0000.npz"),
    ("fourier",            "sim_2000_fourier_0000.npz"),
]


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
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    mean = float(ckpt["normalization"]["mean_density"])
    std = float(ckpt["normalization"]["std_density"])
    print(f"Loaded 220310 ckpt: epoch={ckpt['epoch']}, hidden={mc['hidden_dim']}, mean={mean:.3f}, std={std:.3f}")
    print(f"best_val_mse(norm)={ckpt['best_val_mse']:.5f}\n")

    model = DeepONet(
        BranchNet(int(mc.get("branch_input_dim", 120)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
        TrunkNet(int(mc.get("trunk_input_dim", 2)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
    ).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'family':<20} {'rho_mean':>9} {'rho_max':>8} {'u_mean':>7} {'rel_L2':>8} {'RMSE':>8} {'min_unclip':>11}")
    rel_l2s, rmses = [], []
    for family, fname in ROLLOUTS:
        d = np.load(str(DATA_DIR / fname), allow_pickle=True)
        rho_true = d["density"].astype(np.float32)
        u = d["ramp_control"].astype(np.float32)
        x = d["x_grid"].astype(np.float32)
        t = d["t_grid"].astype(np.float32)
        pred, pred_unclip = predict(model, mean, std, u, x, t)
        rel, rmse = metric(pred, rho_true)
        rel_l2s.append(rel); rmses.append(rmse)
        print(f"{family:<20} {rho_true.mean():>9.2f} {rho_true.max():>8.2f} {u.mean():>7.3f} {rel:>8.4f} {rmse:>8.3f} {pred_unclip.min():>11.2f}")

        # heatmap
        plot_density_heatmap(predicted=pred, true=rho_true, x_grid=x, t_grid=t,
                              output_path=PLOT_DIR / f"{family}_heatmap.png")

    print(f"\nMean rel-L2 across 5 families: {np.mean(rel_l2s):.4f}")
    print(f"Mean RMSE  across 5 families: {np.mean(rmses):.3f} veh/km")
    print("Acceptance gate: 0.15")
    print("\nFor reference (on the SAME 5 rollouts):")
    print("  5/14 run (h=512, current data norm):  rel-L2 0.4026, RMSE 26.67")
    print("  5/13 run (h=512, current data norm):  not re-tested but similar")


if __name__ == "__main__":
    main()
