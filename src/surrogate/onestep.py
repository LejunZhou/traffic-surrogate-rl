"""
One-step autoregressive dynamics model (claim C4 baseline, draft §9).

    s_{k+1} = s_k + f_theta(s_k, d_k, q_r,k, k / K),   s = (rho (Nx), q_out)

trained with one-step MSE on the same rollouts as the DeepONet (bootstrap
ensemble, same normalisation), rolled autoregressively in an otherwise
identical SurrogateVecEnv (plant_type: onestep). Because the env asks for the
prediction at step k given inputs up to k, the plant keeps the state s_{k-1}
per env slot and steps it with (d_k, q_r,k); s_{-1} = 0 (empty road).

  PYTHONPATH=src python -m surrogate.onestep --config configs/surrogate/onestep_v1.yaml \\
      --member 0 --bootstrap-seed 0 --out-dir runs/surrogate/onestep_round0/member_0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from surrogate.deeponet import PlantNormalisation


def clip_state(s: torch.Tensor, norm: PlantNormalisation, rho_max: float = 143.0, q_max: float = 3000.0) -> torch.Tensor:
    """Keep the autoregressive state inside the physical box (0 <= rho <= jam
    density, 0 <= q <= q_max): unclipped rollouts of a one-step model can
    diverge to 1e9 within the hour (compounding error), which would make the
    C4 comparison meaningless rather than unfavourable."""
    z_lo, z_hi = norm.density_to_z(0.0), norm.density_to_z(rho_max)
    rho = s[..., :-1].clamp(z_lo, z_hi)
    q = s[..., -1:].clamp(0.0, q_max / norm.flow_scale)
    return torch.cat([rho, q], dim=-1)


class OneStepModel(nn.Module):
    def __init__(self, n_x: int = 19, hidden: int = 256, layers: int = 3) -> None:
        super().__init__()
        self.n_x = int(n_x)
        d_in = self.n_x + 1 + 3
        mods = []
        d = d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        mods.append(nn.Linear(d, self.n_x + 1))
        self.net = nn.Sequential(*mods)

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """s: (B, Nx+1) normalised state; x: (B, 3) [d/2500, q_r/1600, k/K] -> s_next (B, Nx+1)"""
        return s + self.net(torch.cat([s, x], dim=-1))


def _pairs_from_store(store_dir: Path, files: list[str], norm: PlantNormalisation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sumo_env.rollout import load_rollout_npz

    S, X, Y = [], [], []
    for fn in files:
        arrays, _ = load_rollout_npz(store_dir / fn)
        dens = norm.density_to_z(arrays["density"].astype(np.float32)).T            # (K, Nx)
        q = (arrays["outflow_vph"].astype(np.float32) / norm.flow_scale)[:, None]     # (K, 1)
        state = np.concatenate([dens, q], axis=1)                                    # s_k, k = 0..K-1
        K = state.shape[0]
        prev = np.concatenate([np.zeros((1, state.shape[1]), np.float32), state[:-1]], axis=0)   # s_{k-1}, s_{-1} = 0
        inputs = np.stack([arrays["mainline_demand"] / norm.demand_scale, arrays["ramp_inflow_vph"] / norm.inflow_scale,
                           np.arange(K) / K], axis=1).astype(np.float32)
        S.append(prev); X.append(inputs); Y.append(state)
    return np.concatenate(S), np.concatenate(X), np.concatenate(Y)


def train_member(config: dict, member: int, bootstrap_seed: int | None, out_dir: Path, project_root: Path,
                 epochs_override: int | None = None) -> dict:
    from surrogate.train_plant import geometry_from_sumo_config, load_store_index
    from utils.config import load_config

    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]
    torch.set_num_threads(int(train_cfg.get("num_threads", 2)))
    seed = int(train_cfg.get("seed", 0)) + 1000 * member
    torch.manual_seed(seed); np.random.seed(seed)
    store_dir = project_root / data_cfg["store_dir"]
    split, index = load_store_index(store_dir)
    md = split["metadata"]
    norm = PlantNormalisation(md["mean_density"], md["std_density"])
    sumo_cfg = load_config(str(project_root / data_cfg["sumo_config"]))
    geometry = geometry_from_sumo_config(sumo_cfg)
    round_of = {e["file"]: int(e["round"]) for e in index}
    train_files = [f for f in split["train"] if round_of.get(f, 0) == 0]
    if config.get("ensemble", {}).get("bootstrap", True) and bootstrap_seed is not None:
        rng = np.random.default_rng(int(bootstrap_seed))
        train_files = [train_files[i] for i in rng.integers(0, len(train_files), len(train_files))]
    S, X, Y = _pairs_from_store(store_dir, train_files, norm)
    Sv, Xv, Yv = _pairs_from_store(store_dir, split["val"], norm)
    S, X, Y, Sv, Xv, Yv = (torch.from_numpy(a) for a in (S, X, Y, Sv, Xv, Yv))
    model = OneStepModel(len(geometry["x_grid_m"]), int(model_cfg.get("hidden", 256)), int(model_cfg.get("layers", 3)))
    n_epochs = int(epochs_override or train_cfg.get("n_epochs", 60))
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-6)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    bs = int(train_cfg.get("batch_size", 512))
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"metric": float("inf"), "epoch": 0}
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(len(S))
        losses = []
        for i in range(0, len(S), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = ((model(S[idx], X[idx]) - Y[idx]) ** 2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        if epoch % int(train_cfg.get("eval_every", 5)) == 0 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                val_1step = float(((model(Sv, Xv) - Yv) ** 2).mean())
            roll = _rollout_rel_l2(model, store_dir, split["val"][:20], norm)
            metric = roll["rel_l2_density"] + roll["rel_l2_flow"]
            if metric < best["metric"]:
                best = {"metric": metric, "epoch": epoch, "val_1step_mse": val_1step, **roll}
                torch.save({"model_state_dict": model.state_dict(), "config": {**config, "data": {**data_cfg, "geometry": geometry, "K": int(round(geometry["duration_s"] / geometry["dt_ctrl_s"]))}},
                            "normalization": norm.to_dict(), "epoch": epoch, "bootstrap_files": train_files, "member": member, "best": best},
                           str(out_dir / "best.pt"))
            print(f"[onestep m{member}] ep {epoch:3d} train {np.mean(losses):.5f} val1step {val_1step:.5f} "
                  f"rollout relL2 rho {roll['rel_l2_density']:.4f} q {roll['rel_l2_flow']:.4f} best {best['metric']:.4f}@{best['epoch']} [{time.time() - t0:.0f}s]", flush=True)
    (out_dir / "best.json").write_text(json.dumps(best, indent=1))
    return best


@torch.no_grad()
def _rollout_rel_l2(model: OneStepModel, store_dir: Path, files: list[str], norm: PlantNormalisation) -> dict:
    from sumo_env.rollout import load_rollout_npz

    model.eval()
    rel_r, rel_q = [], []
    for fn in files:
        arrays, _ = load_rollout_npz(store_dir / fn)
        K = arrays["density"].shape[1]
        x = np.stack([arrays["mainline_demand"] / norm.demand_scale, arrays["ramp_inflow_vph"] / norm.inflow_scale, np.arange(K) / K], 1).astype(np.float32)
        s = torch.zeros(1, model.n_x + 1)
        preds = []
        for k in range(K):
            s = clip_state(model(s, torch.from_numpy(x[k:k + 1])), norm)
            preds.append(s[0].numpy())
        preds = np.stack(preds)                                    # (K, Nx+1)
        rho = np.maximum(norm.z_to_density(preds[:, : model.n_x].T), 0.0)
        q = np.maximum(preds[:, model.n_x] * norm.flow_scale, 0.0)
        rel_r.append(np.linalg.norm(rho - arrays["density"]) / max(np.linalg.norm(arrays["density"]), 1e-6))
        rel_q.append(np.linalg.norm(q - arrays["outflow_vph"]) / max(np.linalg.norm(arrays["outflow_vph"]), 1e-6))
    return {"rel_l2_density": float(np.mean(rel_r)), "rel_l2_flow": float(np.mean(rel_q))}


class OneStepEnsemble:
    """Plant interface of SurrogateVecEnv / eval_plant for the one-step baseline."""

    def __init__(self, models: list[OneStepModel], norm: PlantNormalisation, geometry: dict, manifest: dict | None = None) -> None:
        self.members = models
        self.norm = norm
        self.manifest = manifest or {}
        self.x_grid = np.asarray(geometry["x_grid_m"], dtype=np.float32)
        self.L = float(geometry["highway_length_m"]); self.T = float(geometry["duration_s"]); self.dt = float(geometry["dt_ctrl_s"])
        self.K = int(round(self.T / self.dt)); self.Nx = int(len(self.x_grid))
        self._state: dict[int, np.ndarray] = {}     # member -> (n, Nx+1) normalised states

    @property
    def M(self) -> int:
        return len(self.members)

    @classmethod
    def load(cls, ensemble_dir: str | Path, members: list[int] | None = None) -> "OneStepEnsemble":
        d = Path(ensemble_dir)
        man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {"members": [str(p.relative_to(d)) for p in sorted(d.glob("member_*/best.pt"))]}
        paths = [d / m for m in man["members"]]
        if members is not None:
            paths = [paths[i] for i in members]
        models, norm, geometry = [], None, None
        for p in paths:
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
            geometry = geometry or ck["config"]["data"]["geometry"]
            m = OneStepModel(len(geometry["x_grid_m"]), int(ck["config"]["model"].get("hidden", 256)), int(ck["config"]["model"].get("layers", 3)))
            m.load_state_dict(ck["model_state_dict"]); m.eval(); models.append(m)
            norm = norm or PlantNormalisation.from_dict(ck["normalization"])
        return cls(models, norm, geometry, man)

    def _inputs(self, history: np.ndarray, k: np.ndarray) -> torch.Tensor:
        n = history.shape[0]
        idx = np.arange(n)
        x = np.stack([history[idx, 0, k], history[idx, 1, k], k / self.K], axis=1).astype(np.float32)
        return torch.from_numpy(x)

    def _ensure_state(self, m: int, n: int, k: np.ndarray) -> np.ndarray:
        st = self._state.get(m)
        if st is None or st.shape[0] != n:
            st = np.zeros((n, self.Nx + 1), np.float32)
        st[k == 0] = 0.0
        self._state[m] = st
        return st

    @torch.no_grad()
    def _step_member(self, m: int, rows: np.ndarray, history: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st = self._ensure_state(m, history.shape[0], k)
        x = self._inputs(history[rows], k[rows])
        s_next = clip_state(self.members[m](torch.from_numpy(st[rows]), x), self.norm).numpy()
        st[rows] = s_next
        rho = np.maximum(self.norm.z_to_density(s_next[:, : self.Nx]), 0.0)
        q = np.maximum(s_next[:, self.Nx] * self.norm.flow_scale, 0.0)
        return rho.astype(np.float32), q.astype(np.float32)

    def predict_step(self, history: np.ndarray, k: np.ndarray, member: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = history.shape[0]
        rho = np.zeros((n, self.Nx), np.float32); q = np.zeros(n, np.float32)
        member = np.asarray(member); k = np.asarray(k)
        for m in np.unique(member):
            rows = np.where(member == m)[0]
            rho[rows], q[rows] = self._step_member(int(m), rows, history, k)
        return rho, q

    def predict_all_members_step(self, history: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = history.shape[0]
        rows = np.arange(n)
        outs = [self._step_member(m, rows, history, np.asarray(k)) for m in range(self.M)]
        return np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs])

    @torch.no_grad()
    def predict_full(self, branch: np.ndarray, members: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        members = list(range(self.M)) if members is None else members
        n = branch.shape[0]
        rho = np.zeros((len(members), n, self.Nx, self.K), np.float32); q = np.zeros((len(members), n, self.K), np.float32)
        for j, m in enumerate(members):
            s = torch.zeros(n, self.Nx + 1)
            for k in range(self.K):
                s = clip_state(self.members[m](s, self._inputs(branch, np.full(n, k))), self.norm)
                arr = s.numpy()
                rho[j, :, :, k] = np.maximum(self.norm.z_to_density(arr[:, : self.Nx]), 0.0)
                q[j, :, k] = np.maximum(arr[:, self.Nx] * self.norm.flow_scale, 0.0)
        return rho, q


def main() -> None:
    from utils.config import load_config

    ap = argparse.ArgumentParser(description="Train one one-step dynamics-model member")
    ap.add_argument("--config", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--bootstrap-seed", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    args, _ = ap.parse_known_args()
    root = Path(__file__).resolve().parent.parent.parent
    cfg = load_config(str(root / args.config))
    train_member(cfg, args.member, args.bootstrap_seed, root / args.out_dir, root, args.epochs)


if __name__ == "__main__":
    main()
