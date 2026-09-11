"""Quick spot-check: load M3 checkpoint, predict on fresh 1500-vph rollouts, report rel-L2."""
import glob, sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from surrogate.deeponet import BranchNet, TrunkNet, DeepONet


def main() -> None:
    ckpt_path = _ROOT / "runs/surrogate/deeponet_constant_inflow_20260511_234849/best.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    mean = ckpt["normalization"]["mean_density"]
    std = ckpt["normalization"]["std_density"]
    print(f"M3 ckpt: hidden={mc['hidden_dim']}, latent={mc['latent_dim']}, mean={mean:.3f}, std={std:.3f}")

    model = DeepONet(
        BranchNet(int(mc.get("branch_input_dim", 120)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
        TrunkNet(int(mc.get("trunk_input_dim", 2)), int(mc["hidden_dim"]), int(mc["latent_dim"])),
    ).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    files = sorted(glob.glob(str(_ROOT / "data/raw/m3_check_1500vph/sim_*.npz")))
    print(f"\nRollouts found: {len(files)}\n")
    print(f"{'file':<42} {'rho_mean':>9} {'rho_max':>8} {'rel_L2':>8} {'phys_RMSE':>10}")
    rel_l2s = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rho_true = d["density"].astype(np.float32)
        u = d["ramp_control"].astype(np.float32)
        x = d["x_grid"].astype(np.float32)
        t = d["t_grid"].astype(np.float32)
        Nx, Nt = rho_true.shape
        # Match TrafficDataset normalization: divide by highway_length / duration (not min-max).
        x_norm = x / 2000.0
        t_norm = t / 3600.0
        xg, tg = np.meshgrid(x_norm, t_norm, indexing="ij")
        trunk = np.stack([xg.ravel(), tg.ravel()], axis=-1)
        with torch.no_grad():
            branch = torch.from_numpy(u).unsqueeze(0)
            trunk_t = torch.from_numpy(trunk).unsqueeze(0)
            pred_norm = model(branch, trunk_t).cpu().numpy().reshape(Nx, Nt)
        pred = np.maximum(pred_norm * std + mean, 0.0)
        diff = pred - rho_true
        l2 = np.linalg.norm(diff)
        rel = float(l2 / max(np.linalg.norm(rho_true), 1e-8))
        rmse = float(np.sqrt(np.mean(diff**2)))
        rel_l2s.append(rel)
        name = Path(f).name
        print(f"{name:<42} {rho_true.mean():>9.2f} {rho_true.max():>8.2f} {rel:>8.4f} {rmse:>10.3f}")
    print(f"\nMean rel-L2 across {len(files)} rollouts: {np.mean(rel_l2s):.4f}")
    print("M3 documented test mean rel-L2: 0.0738 (milestone_3_progress.md)")
    print("Acceptance gate: 0.15")


if __name__ == "__main__":
    main()
