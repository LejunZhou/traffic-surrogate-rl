"""
M3 spot-check across all 4 ramp-control families at 1500 vph.

For each family (constant reuses an existing rollout; the other 3 are freshly
generated with the same samplers M3 trained on), runs M3 inference and saves
side-by-side density heatmaps (SUMO ground truth vs DeepONet prediction).
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


OUT_DIR = _ROOT / "data" / "raw" / "m3_check_1500vph"
PLOT_DIR = _ROOT / "runs" / "surrogate" / "deeponet_constant_inflow_20260511_234849" / "eval_m3_spotcheck_1500vph"
CKPT_PATH = _ROOT / "runs" / "surrogate" / "deeponet_constant_inflow_20260511_234849" / "best.pt"
SUMO_CFG = _ROOT / "configs" / "sumo" / "phase1_1.yaml"


def build_sumo_scenario_1500(seed: int):
    """Build the SUMO network once at 1500 vph mainline demand."""
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


def run_one(family: str, control: np.ndarray, seed: int, save_name: str) -> dict:
    """Run one SUMO rollout with the given control and save to OUT_DIR."""
    cfg, net, route, det = build_sumo_scenario_1500(seed)
    result = run_simulation(
        net_file=net, route_file=route, detector_file=det,
        ramp_control=control.astype(np.float32), config=cfg,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{save_name}.npz"
    np.savez(
        str(out_path),
        density=result["density"], speed=result["speed"], flow=result["flow"],
        x_grid=result["x_grid"], t_grid=result["t_grid"],
        mainline_demand=result["mainline_demand"], ramp_control=result["ramp_control"],
        seed=np.array(result["metadata"]["seed"]),
        mainline_demand_vph=np.array(result["metadata"]["mainline_demand_vph"]),
        ramp_demand_vph=np.array(result["metadata"]["ramp_demand_vph"]),
    )
    d = result["density"]
    print(f"[{family:<19}] seed={seed:>3}  rho_mean={d.mean():5.2f}  rho_max={d.max():6.2f}  u_mean={control.mean():.3f}  → {out_path.name}")
    return {"family": family, "path": str(out_path), "control": control,
            "density_true": d.astype(np.float32),
            "x_grid": result["x_grid"], "t_grid": result["t_grid"]}


def load_m3():
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    mean = ckpt["normalization"]["mean_density"]
    std = ckpt["normalization"]["std_density"]
    model = DeepONet(
        BranchNet(int(mc.get("branch_input_dim", 120)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
        TrunkNet(int(mc.get("trunk_input_dim", 2)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
    ).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, mean, std


def predict(model: DeepONet, mean: float, std: float, u: np.ndarray,
            x_grid: np.ndarray, t_grid: np.ndarray):
    Nx, Nt = x_grid.size, t_grid.size
    x_norm = x_grid.astype(np.float32) / 2000.0   # matches TrafficDataset
    t_norm = t_grid.astype(np.float32) / 3600.0
    xg, tg = np.meshgrid(x_norm, t_norm, indexing="ij")
    trunk = np.stack([xg.ravel(), tg.ravel()], axis=-1).astype(np.float32)
    with torch.no_grad():
        branch = torch.from_numpy(u.astype(np.float32)).unsqueeze(0)
        trunk_t = torch.from_numpy(trunk).unsqueeze(0)
        pred_norm = model(branch, trunk_t).cpu().numpy().reshape(Nx, Nt)
    return np.maximum(pred_norm * std + mean, 0.0)


def metric(pred, true):
    diff = pred - true
    l2 = float(np.linalg.norm(diff))
    rel = float(l2 / max(np.linalg.norm(true), 1e-8))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return rel, rmse


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Family 1 — constant: reuse an existing rollout (u=0.5, seed 101)
    existing = OUT_DIR / "sim_m3chk_500_0000.npz"
    d_const = np.load(str(existing), allow_pickle=True)
    rollouts = [{
        "family": "constant",
        "path": str(existing),
        "control": d_const["ramp_control"].astype(np.float32),
        "density_true": d_const["density"].astype(np.float32),
        "x_grid": d_const["x_grid"].astype(np.float32),
        "t_grid": d_const["t_grid"].astype(np.float32),
    }]
    print(f"[{'constant':<19}] (reused)        rho_mean={rollouts[0]['density_true'].mean():5.2f}  rho_max={rollouts[0]['density_true'].max():6.2f}  u_mean={rollouts[0]['control'].mean():.3f}  → {existing.name}")

    T_ctrl = 120
    rng = np.random.default_rng(42)

    # Family 2 — piecewise_constant
    u_pwc = _sample_piecewise_constant(T_ctrl, rng)
    rollouts.append(run_one("piecewise_constant", u_pwc, seed=201, save_name="sim_m3chk_pwc_0000"))

    # Family 3 — smooth
    u_smooth = _sample_smooth(T_ctrl, rng)
    rollouts.append(run_one("smooth", u_smooth, seed=202, save_name="sim_m3chk_smooth_0000"))

    # Family 4 — ramp_step
    u_step = _sample_ramp_step(T_ctrl, rng)
    rollouts.append(run_one("ramp_step", u_step, seed=203, save_name="sim_m3chk_rampstep_0000"))

    # ── Inference ──────────────────────────────────────────────────────────────
    model, mean, std = load_m3()
    print(f"\nLoaded M3: mean={mean:.3f}, std={std:.3f}\n")

    print(f"{'family':<20} {'rho_mean':>9} {'rho_max':>8} {'u_mean':>7} {'rel_L2':>8} {'RMSE':>8}")
    for r in rollouts:
        pred = predict(model, mean, std, r["control"], r["x_grid"], r["t_grid"])
        rel, rmse = metric(pred, r["density_true"])
        r["density_pred"] = pred
        r["rel_l2"] = rel
        r["rmse"] = rmse
        print(f"{r['family']:<20} {r['density_true'].mean():>9.2f} {r['density_true'].max():>8.2f} {r['control'].mean():>7.3f} {rel:>8.4f} {rmse:>8.3f}")

    mean_rel = float(np.mean([r["rel_l2"] for r in rollouts]))
    print(f"\nMean rel-L2 across 4 families: {mean_rel:.4f}")
    print("M3 documented test mean rel-L2: 0.0738")
    print("Acceptance gate: 0.15\n")

    # ── Heatmaps ───────────────────────────────────────────────────────────────
    for r in rollouts:
        out_png = PLOT_DIR / f"{r['family']}_heatmap.png"
        plot_density_heatmap(
            predicted=r["density_pred"], true=r["density_true"],
            x_grid=r["x_grid"], t_grid=r["t_grid"], output_path=out_png,
        )
        print(f"  saved heatmap: {out_png.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
