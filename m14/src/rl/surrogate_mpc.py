"""
Surrogate-MPC: gradient-based model predictive
ramp metering through the differentiable ensemble-mean DeepONet.

At step k the inflow history up to k-1 is known exactly; candidate controls
u_{k..k+H-1} (H = 20 steps = 10 min) enter through the differentiable queue
recursion (soft-min), the plant returns rho_hat and q_hat over the horizon,
and the objective is the configured reward summed over the horizon plus a
terminal queue term. 30 Adam iterations warm-started from the previous
solution shifted by one step; u_k is applied. Offset-free correction: the
previous step's prediction error (rho_meas - rho_hat) is added to the
predicted densities over the horizon (decayed geometrically).

The MPC knows the future demand profile (d, r) of the episode, i.e. it is the
perfect-forecast anticipative baseline. Runtime ~0.2-0.4 s per step on CPU,
so it also runs closed-loop inside SUMO.

Spec: mpc:<ensemble_dir>[,H=20,iters=30,lr=0.1,members=0-4,offset=1,terminal=0.5,tau=2]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rl.reward import RewardWeights


def _softmin(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    return -tau * torch.logsumexp(torch.stack([-a / tau, -b / tau]), dim=0)


class SurrogateMPC:
    def __init__(self, ensemble, weights: RewardWeights, horizon: int = 20, iters: int = 30, lr: float = 0.1,
                 discharge_vph: float = 1200.0, terminal_weight: float = 0.5, offset: bool = True, offset_decay: float = 0.9,
                 tau: float = 2.0, members: list[int] | None = None, seed: int = 0, label: str | None = None) -> None:
        self.ens = ensemble
        self.w = weights
        self.H = int(horizon); self.iters = int(iters); self.lr = float(lr)
        self.D = float(discharge_vph); self.dt = float(ensemble.dt); self.K = int(ensemble.K); self.Nx = int(ensemble.Nx)
        self.terminal_weight = float(terminal_weight); self.offset = bool(offset); self.offset_decay = float(offset_decay)
        self.tau = float(tau)
        self.members = list(range(ensemble.M)) if members is None else list(members)
        self.label = label or f"Surrogate-MPC H={self.H} it={self.iters} M={len(self.members)}"
        self.spec = {"type": "mpc", "H": self.H, "iters": self.iters, "lr": self.lr, "members": self.members,
                     "offset": self.offset, "terminal_weight": self.terminal_weight}
        self.norm = ensemble.norm
        self.tables = [ensemble.tables[m] for m in self.members]      # (K, Nx+1, 2p)
        self.models = [ensemble.members[m] for m in self.members]
        self.dx_km = float(ensemble.x_grid[1] - ensemble.x_grid[0]) / 1000.0
        torch.manual_seed(seed)
        self.reset()

    # -- episode state -----------------------------------------------------
    def reset(self, env=None) -> None:
        self.k = 0
        self.Q = 0.0
        self.d = None; self.r = None
        if env is not None:
            self.d = np.asarray(env.mainline_profile, dtype=np.float32)[: self.K]
            self.r = np.asarray(env.ramp_profile, dtype=np.float32)[: self.K]
        self.q_r_hist = np.zeros(self.K, np.float32)
        self.cum_offered = 0.0; self.cum_served = 0.0
        self.z = torch.zeros(self.H)              # logits, u = sigmoid(z); 0 -> u = 0.5
        self.rho_offset = np.zeros(self.Nx, np.float32)
        self.last_pred_rho = None
        self.wall = []

    def _profile_from_info(self, info: dict) -> None:
        if self.d is None:
            raise RuntimeError("SurrogateMPC needs the episode profile: call reset(env) with a SumoEnv in profile mode")

    # -- control -------------------------------------------------------------
    def __call__(self, obs, info=None):
        import time

        t0 = time.time()
        info = info or {}
        self._profile_from_info(info)
        k = int(info.get("k", self.k))
        if "queue_after" in info:
            self.Q = float(info["queue_after"])
        if k > 0 and "ramp_inflow_vph" in info:
            self.q_r_hist[k - 1] = float(info["ramp_inflow_vph"])
        self.cum_offered = float(info.get("cum_offered_veh", self.cum_offered))
        self.cum_served = float(info.get("cum_served_veh", self.cum_served))
        if self.offset and self.last_pred_rho is not None and "density" in info:
            self.rho_offset = (np.asarray(info["density"], np.float32) - self.last_pred_rho).astype(np.float32)
        u = self._solve(k)
        self.k = k + 1
        self.wall.append(time.time() - t0)
        return np.array([u], dtype=np.float32)

    def _solve(self, k: int) -> float:
        H = min(self.H, self.K - k)
        d_t = torch.from_numpy(self.d / self.norm.demand_scale)
        hist_prefix = torch.from_numpy(self.q_r_hist[:k] / self.norm.inflow_scale)
        r_future = torch.from_numpy(self.r[k:k + H].astype(np.float32))
        d_future = torch.from_numpy(self.d[k:k + H].astype(np.float32))
        offset = torch.from_numpy(self.rho_offset) * (self.offset_decay ** torch.arange(H, dtype=torch.float32)).unsqueeze(1)
        # GRU branch: encode the known prefix once (no grad), then per iteration
        # only the H-step horizon is run through the recurrence (6x faster than
        # re-encoding the full K-step sequence every iteration).
        prefix_states = []
        for model in self.models:
            gru = getattr(model.branch, "gru", None)
            if gru is None:
                prefix_states.append(None)
                continue
            with torch.no_grad():
                if k > 0:
                    x_prefix = torch.stack([d_t[:k], hist_prefix]).T.unsqueeze(0)     # (1, k, 2)
                    _, h0 = gru(x_prefix)
                else:
                    h0 = None
            prefix_states.append(h0)
        # warm start: shift the previous solution by one step
        z = torch.cat([self.z[1:], self.z[-1:]])[:H].clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=self.lr)
        dt_h = self.dt / 3600.0
        Qn, sig = self.w.queue_norm, self.w.sigma_ref
        for _ in range(self.iters):
            opt.zero_grad()
            u = torch.sigmoid(z)
            Q = torch.tensor(float(self.Q))
            q_r, Qs, q_refs = [], [], []
            for h in range(H):
                a = r_future[h] * dt_h
                cap = u[h] * self.D * dt_h
                rel = _softmin(Q + a, cap, self.tau * dt_h * 10.0)
                drain = torch.clamp(torch.minimum(Q * 3600.0 / self.dt, torch.clamp(torch.tensor(self.D) - r_future[h], min=0.0)), min=0.0)
                q_ref = torch.clamp(d_future[h] + r_future[h] + (drain if self.w.drain_allowance else 0.0), max=self.w.q_cap_value) \
                    if self.w.q_ref_mode == "offered" else torch.tensor(self.w.q_ref)
                Q = Q + a - rel
                q_r.append(rel / dt_h); Qs.append(Q); q_refs.append(q_ref)
            q_r = torch.stack(q_r); Qs = torch.stack(Qs); q_refs = torch.stack(q_refs)
            q_channel = torch.cat([hist_prefix, q_r / self.norm.inflow_scale, torch.zeros(self.K - k - H)])
            rho_sum = 0.0; q_sum = 0.0
            for model, table, h0 in zip(self.models, self.tables, prefix_states):
                gru = getattr(model.branch, "gru", None)
                if gru is not None:
                    x_future = torch.stack([d_t[k:k + H], q_r / self.norm.inflow_scale]).T.unsqueeze(0)   # (1, H, 2)
                    out, _ = gru(x_future, h0)
                    b = model.branch.out(out)[0]                                   # (H, p)
                else:
                    branch = torch.stack([d_t, q_channel]).unsqueeze(0)           # (1, 2, K)
                    b_all = model.branch(branch)[0]                                # (K, p) or (1, p)
                    b = b_all[k:k + H] if b_all.shape[0] > 1 else b_all.expand(H, -1)
                tau = table[k:k + H]                                             # (H, Nx+1, 2p)
                rho_z, q_hat = model.combine(b.unsqueeze(1).expand(-1, tau.shape[1], -1), tau)
                rho_sum = rho_sum + rho_z[:, : self.Nx]; q_sum = q_sum + q_hat[:, self.Nx]
            M = len(self.models)
            rho = torch.clamp(self.norm.z_to_density(rho_sum / M) + offset, min=0.0)
            q_out = torch.clamp(q_sum / M * self.norm.flow_scale, min=0.0)
            if self.w.form == "tts":
                on_road = rho.sum(dim=1) * self.dx_km
                offered = torch.cumsum((d_future + r_future) * dt_h, 0) + self.cum_offered
                served = torch.cumsum(q_out * dt_h, 0) + self.cum_served
                backlog = torch.clamp(offered - served - on_road - Qs, min=0.0)
                cost = ((on_road + Qs + backlog) * dt_h).sum() / self.w.tts_scale
                cost = cost + self.terminal_weight * (Qs[-1] + backlog[-1]) * dt_h / self.w.tts_scale
            else:
                lost = torch.clamp(q_refs - q_out, min=0.0) / q_refs
                cost = (self.w.delta * lost + self.w.beta * (Qs / Qn) ** 2 + self.w.gamma * rho.std(dim=1) / sig).sum()
                cost = cost + self.terminal_weight * self.w.beta * (Qs[-1] / Qn) ** 2
            cost.backward()
            opt.step()
        with torch.no_grad():
            self.z = torch.cat([z.detach(), self.z[H:]]) if H < self.H else z.detach().clone()
            u0 = float(torch.sigmoid(z[0]))
            # prediction of the density that the next observation will show (for the offset)
            self.last_pred_rho = torch.clamp(self.norm.z_to_density(rho_sum / M)[0], min=0.0).numpy().astype(np.float32)
        return float(np.clip(u0, 0.0, 1.0))

    # -- spec parsing ----------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: str, env=None, env_cfg: dict | None = None) -> "SurrogateMPC":
        head, _, rest = str(spec).partition(":")
        parts = [p for p in rest.split(",") if p]
        ens_dir = parts[0]
        kw = {}
        for p in parts[1:]:
            key, _, val = p.partition("=")
            kw[key.strip()] = val.strip()
        env_cfg = env_cfg or {}
        root = Path(env_cfg.get("project_root", Path(__file__).resolve().parents[2]))
        path = Path(ens_dir) if Path(ens_dir).is_absolute() else root / ens_dir
        members = None
        if "members" in kw:
            a, _, b = kw["members"].partition("-")
            members = list(range(int(a), int(b) + 1)) if b else [int(x) for x in kw["members"].split("+")]
        from surrogate.deeponet import DeepONetEnsemble

        ens = DeepONetEnsemble.load(path, members=members)
        weights = env.reward_weights if env is not None else RewardWeights.from_config(env_cfg.get("reward", {}))
        discharge = float(env.ramp_discharge_vph) if env is not None else float(env_cfg.get("ramp_discharge_vph", 1200.0))
        return cls(ens, weights, horizon=int(kw.get("H", 20)), iters=int(kw.get("iters", 30)), lr=float(kw.get("lr", 0.1)),
                   discharge_vph=discharge, terminal_weight=float(kw.get("terminal", 0.5)), offset=bool(int(kw.get("offset", 1))),
                   tau=float(kw.get("tau", 2.0)), label=f"Surrogate-MPC({path.name}, H={kw.get('H', 20)}, it={kw.get('iters', 30)})")
