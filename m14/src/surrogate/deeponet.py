"""M14 causal GRU DeepONet: boundary histories to density and exit flow.

The branch encodes [offered mainline demand / 2500, admitted ramp flow / 1600]
at 30-second intervals. The trunk encodes (x/L, interval-start time/T).
Each query combines the current-index branch features with channel-specific
trunk features. Density is z-scored; exit flow is scaled by 2500 veh/h.
The input scale 1600 is retained from training; physical M14 discharge is 1200.
"""

import torch
import torch.nn as nn
import json as _json
import math as _math
from pathlib import Path as _Path
import numpy as _np


class GRUBranch(nn.Module):
    """Unidirectional GRU read out at every step (causal by construction)."""

    def __init__(self, in_channels: int = 2, hidden: int = 128, layers: int = 2, latent_dim: int = 256) -> None:
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, latent_dim)
        self.latent_dim = int(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x.transpose(1, 2))
        return self.out(h)




class PlantTrunk(nn.Module):
    """(x/L, t/T) -> 2p latent, optionally with Fourier features of the coordinates."""

    def __init__(self, hidden: int = 512, layers: int = 3, latent_dim: int = 256, fourier_features: int = 0,
                 fourier_scale: float = 4.0) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_fourier = int(fourier_features)
        in_dim = 2
        if self.n_fourier > 0:
            gen = torch.Generator().manual_seed(0)
            self.register_buffer("B", torch.randn(2, self.n_fourier, generator=gen) * fourier_scale)
            in_dim = 2 + 2 * self.n_fourier
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        mods.append(nn.Linear(d, 2 * self.latent_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """xt: (..., 2) -> (..., 2p)"""
        if self.n_fourier > 0:
            proj = 2 * _math.pi * xt @ self.B
            xt = torch.cat([xt, torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(xt)


class PlantDeepONet(nn.Module):
    """Two-output DeepONet plant model (density and flow fields)."""

    def __init__(self, branch: nn.Module, trunk: PlantTrunk) -> None:
        super().__init__()
        if branch.latent_dim != trunk.latent_dim:
            raise ValueError("branch and trunk latent_dim must match")
        self.branch = branch
        self.trunk = trunk
        self.p = int(trunk.latent_dim)
        self.bias_rho = nn.Parameter(torch.zeros(1))
        self.bias_q = nn.Parameter(torch.zeros(1))
        self.scale = 1.0 / _math.sqrt(self.p)

    def combine(self, b: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """b: (B, Nq, p), tau: (B, Nq, 2p) -> rho_hat (B, Nq), q_hat (B, Nq)"""
        tau_rho, tau_q = tau[..., : self.p], tau[..., self.p:]
        rho = (b * tau_rho).sum(-1) * self.scale + self.bias_rho
        q = (b * tau_q).sum(-1) * self.scale + self.bias_q
        return rho, q

    def gather_branch(self, b_all: torch.Tensor, query_k: torch.Tensor) -> torch.Tensor:
        """b_all: (B, K, p); query_k: (B, Nq) long -> (B, Nq, p)"""
        idx = query_k.unsqueeze(-1).expand(-1, -1, b_all.shape[-1])
        return torch.gather(b_all, 1, idx)

    def forward(self, branch_in: torch.Tensor, query_k: torch.Tensor, query_xt: torch.Tensor):
        """
        branch_in: (B, 2, K); query_k: (B, Nq) long; query_xt: (B, Nq, 2)
        Returns (rho_hat, q_hat), each (B, Nq).
        """
        b_all = self.branch(branch_in)
        b = self.gather_branch(b_all, query_k)
        tau = self.trunk(query_xt)
        return self.combine(b, tau)

    @torch.no_grad()
    def trunk_table(self, x_norm: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """Cache tau for a grid: x_norm (Nx,), t_norm (K,) -> (K, Nx, 2p)."""
        K, Nx = t_norm.shape[0], x_norm.shape[0]
        xt = torch.stack(torch.meshgrid(t_norm, x_norm, indexing="ij"), dim=-1)   # (K, Nx, 2) as (t, x)
        xt = xt[..., [1, 0]]                                                     # -> (x, t)
        return self.trunk(xt.reshape(-1, 2)).reshape(K, Nx, 2 * self.p)


def build_plant_model(model_cfg: dict, K: int = 120) -> PlantDeepONet:
    bcfg = dict(model_cfg.get("branch", {}))
    tcfg = dict(model_cfg.get("trunk", {}))
    p = int(bcfg.get("latent_dim", tcfg.get("latent_dim", 256)))
    btype = str(bcfg.get("type", "gru"))
    if btype != "gru":
        raise ValueError(f"M14 supports the causal GRU branch, got {btype!r}")
    branch = GRUBranch(int(bcfg.get("in_channels", 2)), int(bcfg.get("hidden", 128)), int(bcfg.get("layers", 2)), p)
    trunk = PlantTrunk(int(tcfg.get("hidden_dim", 512)), int(tcfg.get("layers", 3)), p,
                       int(tcfg.get("fourier_features", 0)), float(tcfg.get("fourier_scale", 4.0)))
    return PlantDeepONet(branch, trunk)


class PlantNormalisation:
    """Scales shared by the dataset, trainer, environment and evaluation."""

    def __init__(self, density_mean: float, density_std: float, demand_scale: float = 2500.0,
                 inflow_scale: float = 1600.0, flow_scale: float = 2500.0) -> None:
        self.density_mean = float(density_mean)
        self.density_std = max(float(density_std), 1e-6)
        self.demand_scale = float(demand_scale)
        self.inflow_scale = float(inflow_scale)
        self.flow_scale = float(flow_scale)

    def to_dict(self) -> dict:
        return {"mean_density": self.density_mean, "std_density": self.density_std,
                "demand_scale": self.demand_scale, "inflow_scale": self.inflow_scale, "flow_scale": self.flow_scale}

    @classmethod
    def from_dict(cls, d: dict) -> "PlantNormalisation":
        return cls(d["mean_density"], d["std_density"], d.get("demand_scale", 2500.0),
                   d.get("inflow_scale", 1600.0), d.get("flow_scale", 2500.0))

    def branch_input(self, mainline_vph, ramp_inflow_vph) -> _np.ndarray:
        return _np.stack([_np.asarray(mainline_vph, dtype=_np.float32) / self.demand_scale,
                          _np.asarray(ramp_inflow_vph, dtype=_np.float32) / self.inflow_scale], axis=-2)

    def density_to_z(self, rho):
        return (rho - self.density_mean) / self.density_std

    def z_to_density(self, z):
        return z * self.density_std + self.density_mean


def load_plant_checkpoint(path: str | _Path, device: str = "cpu") -> tuple[PlantDeepONet, dict, PlantNormalisation]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    K = int(cfg.get("data", {}).get("K", 120))
    model = build_plant_model(cfg["model"], K=K)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, PlantNormalisation.from_dict(ckpt["normalization"])


class DeepONetEnsemble:
    """M GRU DeepONet members loaded from <dir>/manifest.json.

    predict_step(branch (n, 2, K), k (n,), member (n,)) -> rho (n, Nx), q_out (n,)
    predict_full(branch (n, 2, K)) -> rho (M, n, Nx, K), q_out (M, n, K)
    """

    def __init__(self, members: list[PlantDeepONet], norm: PlantNormalisation, config: dict,
                 x_grid_m: _np.ndarray, highway_length_m: float, duration_s: float, dt_ctrl_s: float,
                 manifest: dict | None = None, device: str = "cpu") -> None:
        self.members = members
        self.norm = norm
        self.config = config
        self.manifest = manifest or {}
        self.device = torch.device(device)
        self.x_grid = _np.asarray(x_grid_m, dtype=_np.float32)
        self.L = float(highway_length_m)
        self.T = float(duration_s)
        self.dt = float(dt_ctrl_s)
        self.K = int(round(self.T / self.dt))
        self.Nx = int(len(self.x_grid))
        t_norm = torch.arange(self.K, dtype=torch.float32) * self.dt / self.T
        x_norm = torch.cat([torch.from_numpy(self.x_grid / self.L), torch.tensor([1.0])])   # detectors + exit
        self.tables = [m.trunk_table(x_norm, t_norm).to(self.device) for m in self.members]   # (K, Nx+1, 2p)
        self.x_norm, self.t_norm = x_norm, t_norm

    @property
    def M(self) -> int:
        return len(self.members)

    @classmethod
    def load(cls, ensemble_dir: str | _Path, device: str = "cpu", members: list[int] | None = None) -> "DeepONetEnsemble":
        d = _Path(ensemble_dir)
        man_path = d / "manifest.json"
        if man_path.exists():
            manifest = _json.loads(man_path.read_text())
            paths = [d / m for m in manifest["members"]]
        else:
            paths = sorted(d.glob("member_*/best.pt"))
            manifest = {"members": [str(p.relative_to(d)) for p in paths]}
        if members is not None:
            paths = [paths[i] for i in members]
        if not paths:
            raise FileNotFoundError(f"no ensemble members under {d}")
        models, norm, cfg = [], None, None
        for p in paths:
            m, ckpt, n = load_plant_checkpoint(p, device)
            models.append(m.to(device)); norm = norm or n; cfg = cfg or ckpt["config"]
        data = cfg.get("data", {})
        geometry = manifest.get("geometry") or data.get("geometry") or {}
        x_grid = _np.asarray(geometry.get("x_grid_m", data.get("x_grid_m", (_np.arange(19) + 1) * 100.0)), dtype=_np.float32)
        return cls(models, norm, cfg, x_grid, float(geometry.get("highway_length_m", data.get("highway_length_m", 2000.0))),
                   float(geometry.get("duration_s", data.get("duration_s", 3600.0))),
                   float(geometry.get("dt_ctrl_s", data.get("dt_ctrl_s", 30.0))), manifest, device)

    @torch.no_grad()
    def branch_all(self, branch_in: torch.Tensor, member: int) -> torch.Tensor:
        return self.members[member].branch(branch_in)

    @torch.no_grad()
    def predict_step(self, branch_in: _np.ndarray, k: _np.ndarray, member: _np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
        """Physical density at the detectors and exit flow (veh/h) at step k for each row."""
        n = branch_in.shape[0]
        rho = _np.zeros((n, self.Nx), dtype=_np.float32)
        q = _np.zeros(n, dtype=_np.float32)
        x = torch.from_numpy(_np.asarray(branch_in, dtype=_np.float32)).to(self.device)
        k = _np.asarray(k, dtype=_np.int64)
        member = _np.asarray(member, dtype=_np.int64)
        for m in _np.unique(member):
            rows = _np.where(member == m)[0]
            model = self.members[int(m)]
            b_all = model.branch(x[rows])                                   # (r, K, p)
            kk = torch.from_numpy(k[rows]).to(self.device)
            b = b_all[torch.arange(len(rows)), kk]   # (r, p)
            tau = self.tables[int(m)][kk]                                    # (r, Nx+1, 2p)
            r_hat, q_hat = model.combine(b.unsqueeze(1).expand(-1, tau.shape[1], -1), tau)
            rho[rows] = self.norm.z_to_density(r_hat[:, : self.Nx]).cpu().numpy()
            q[rows] = (q_hat[:, self.Nx] * self.norm.flow_scale).cpu().numpy()
        return rho, q

    @torch.no_grad()
    def predict_all_members_step(self, branch_in: _np.ndarray, k: _np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
        """(M, n, Nx) densities and (M, n) exit flows at step k for every member."""
        outs_r, outs_q = [], []
        for m in range(self.M):
            r, q = self.predict_step(branch_in, k, _np.full(len(k), m))
            outs_r.append(r); outs_q.append(q)
        return _np.stack(outs_r), _np.stack(outs_q)

    @torch.no_grad()
    def predict_full(self, branch_in: _np.ndarray, members: list[int] | None = None) -> tuple[_np.ndarray, _np.ndarray]:
        """Full space-time fields: rho (M, n, Nx, K) veh/km, q_out (M, n, K) veh/h."""
        x = torch.from_numpy(_np.asarray(branch_in, dtype=_np.float32)).to(self.device)
        n = x.shape[0]
        members = list(range(self.M)) if members is None else members
        rho_out = _np.zeros((len(members), n, self.Nx, self.K), dtype=_np.float32)
        q_out = _np.zeros((len(members), n, self.K), dtype=_np.float32)
        for j, m in enumerate(members):
            model = self.members[m]
            b_all = model.branch(x)                                          # (n, K, p)
            tau = self.tables[m]                                             # (K, Nx+1, 2p)
            b = b_all.unsqueeze(2).expand(-1, -1, tau.shape[1], -1)          # (n, K, Nx+1, p)
            r_hat, q_hat = model.combine(b, tau.unsqueeze(0).expand(n, -1, -1, -1))
            rho_out[j] = self.norm.z_to_density(r_hat[:, :, : self.Nx]).permute(0, 2, 1).cpu().numpy()
            q_out[j] = (q_hat[:, :, self.Nx] * self.norm.flow_scale).cpu().numpy()
        return rho_out, q_out
