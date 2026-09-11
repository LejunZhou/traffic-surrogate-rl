"""Shockwave-band + per-family evaluation of the retrained surrogate.

Beyond the global rel-L2 from surrogate.eval, this breaks error down by:
  - region: free-flow (x>=1400) vs pre-merge SHOCKWAVE band (x<1300, congested cells)
  - control family (bang_bang, constant_grid, fourier, ...)
so we can see whether the model captures the backward-propagating front or just
smooths it. Saves pred-vs-true heatmaps for the most congested test rollouts.

    PYTHONPATH=src python scripts/_eval_shockwave_band.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from surrogate.deeponet import BranchNet, TrunkNet, DeepONet  # noqa: E402
from utils.plotting import plot_density_heatmap  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           _ROOT / "runs/surrogate/deeponet_shockwave_2000vph_20260626_164127")
SPLIT = _ROOT / "data/splits/shockwave_2000vph/split_index.json"
RAW = _ROOT / "data/raw/shockwave_2000vph"
HIGHWAY_M, DURATION_S = 2000.0, 3600.0
MERGE_M = 1300.0
CONGESTED = 40.0  # veh/km threshold defining a "shockwave" cell


def family_of(name: str) -> str:
    head, _, tail = name.rpartition("_")
    return head if tail.replace(".npz", "").isdigit() else name


def load_model(ckpt):
    mc = ckpt["config"]["model"]
    m = DeepONet(
        BranchNet(int(mc["branch_input_dim"]), int(mc["hidden_dim"]), int(mc["latent_dim"])),
        TrunkNet(int(mc["trunk_input_dim"]), int(mc["hidden_dim"]), int(mc["latent_dim"])),
    ).eval()
    m.load_state_dict(ckpt["model_state_dict"])
    return m


def predict(model, mean, std, u, x, t):
    Nx, Nt = x.size, t.size
    xg, tg = np.meshgrid(x / HIGHWAY_M, t / DURATION_S, indexing="ij")
    trunk = np.stack([xg.ravel(), tg.ravel()], -1).astype(np.float32)
    with torch.no_grad():
        b = torch.from_numpy(u.astype(np.float32)).unsqueeze(0)
        tr = torch.from_numpy(trunk).unsqueeze(0)
        pn = model(b, tr).cpu().numpy().reshape(Nx, Nt)
    return np.maximum(pn * std + mean, 0.0)


def rel_l2(pred, true, mask=None):
    if mask is not None:
        pred, true = pred[mask], true[mask]
    if true.size == 0:
        return float("nan")
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-8))


def main():
    ckpt = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    model = load_model(ckpt)
    mean = ckpt["normalization"]["mean_density"]
    std = ckpt["normalization"]["std_density"]
    test = json.load(open(SPLIT))["test"]
    print(f"Model epoch {ckpt['epoch']}, best_val_mse {ckpt['best_val_mse']:.4f}; "
          f"{len(test)} test rollouts\n")

    glob, band, free = [], [], []
    per_fam = defaultdict(lambda: {"glob": [], "band": []})
    congestion = []  # (congested_frac, fname, pred, true, x, t, u)
    for fname in test:
        z = np.load(RAW / fname)
        true, x, t, u = z["density"], z["x_grid"], z["t_grid"], z["ramp_control"]
        pred = predict(model, mean, std, u, x, t)
        pre = (x[:, None] < MERGE_M) & (true > CONGESTED)   # shockwave cells
        post = np.broadcast_to((x[:, None] >= 1400.0), true.shape)
        g = rel_l2(pred, true)
        b = rel_l2(pred, true, pre)
        glob.append(g); free.append(rel_l2(pred, true, post))
        if not np.isnan(b):
            band.append(b)
        fam = family_of(fname)
        per_fam[fam]["glob"].append(g)
        if not np.isnan(b):
            per_fam[fam]["band"].append(b)
        congestion.append((float(pre.mean()), fname, pred, true, x, t))

    print(f"{'region':<28}{'mean rel-L2':>12}")
    print(f"{'GLOBAL (all cells)':<28}{np.mean(glob):>12.3f}")
    print(f"{'SHOCKWAVE band (pre, >40)':<28}{np.mean(band):>12.3f}")
    print(f"{'free-flow (post-merge)':<28}{np.nanmean(free):>12.3f}")
    print()
    print(f"{'family':<20}{'n':>4}{'glob relL2':>12}{'band relL2':>12}")
    for fam in sorted(per_fam):
        d = per_fam[fam]
        bv = np.mean(d["band"]) if d["band"] else float("nan")
        print(f"{fam:<20}{len(d['glob']):>4}{np.mean(d['glob']):>12.3f}{bv:>12.3f}")

    # heatmaps for the 4 most congested test rollouts
    out = RUN / "eval_shockwave"
    out.mkdir(exist_ok=True)
    congestion.sort(reverse=True)
    print(f"\nSaving pred-vs-true heatmaps for 4 most-congested test rollouts → {out.name}/")
    for frac, fname, pred, true, x, t in congestion[:4]:
        plot_density_heatmap(predicted=pred, true=true, x_grid=x, t_grid=t,
                             output_path=out / f"{Path(fname).stem}_predvtrue.png")
        print(f"  {fname:<28} congested_frac={frac:.2f} "
              f"glob={rel_l2(pred,true):.3f} band={rel_l2(pred,true,(x[:,None]<MERGE_M)&(true>CONGESTED)):.3f}")

    summary = {
        "epoch": int(ckpt["epoch"]),
        "global_rel_l2": float(np.mean(glob)),
        "shockwave_band_rel_l2": float(np.mean(band)),
        "freeflow_rel_l2": float(np.nanmean(free)),
        "per_family": {f: {"global": float(np.mean(d["glob"])),
                           "band": float(np.mean(d["band"])) if d["band"] else None}
                       for f, d in per_fam.items()},
    }
    json.dump(summary, open(out / "shockwave_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
