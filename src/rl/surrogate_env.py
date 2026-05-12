"""
Gymnasium environment backed by the trained DeepONet surrogate.

At each step, the agent provides a ramp metering action; the surrogate
predicts the resulting density field; the shaped reward (mean density +
queue + density std) is computed and returned. Designed to expose the
same observation / action / reward contract as SumoEnv so PPO training
code is environment-agnostic.

Observation (shape (N_x + 3,)):
    density[0:N_x]  — z-score normalized density at the detector grid
    demand[N_x]     — min-max normalized current mainline demand ∈ [0, 1]
                      (0.0 when min == max, matching SumoEnv)
    time[N_x+1]     — normalized time index k / T_ctrl ∈ [0, 1]
    queue[N_x+2]    — analytical on-ramp queue / queue_norm_scale

Action (shape (1,)):
    ramp metering rate ∈ [0, 1] (continuous Box)

Surrogate rollout strategy (zero-padded partial control sequences):
    At step k, branch input = [u(0),...,u(k), 0,...,0].
    Trunk queries density at (x_i, t_k) for each detector i.
    DeepONet is re-evaluated from scratch each step (not autoregressive).

See proposal.md §"Surrogate rollout strategy" and §"Analytical queue
model" for the contract and motivation.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rl.reward import RewardWeights, compute_reward
from surrogate.deeponet import BranchNet, DeepONet, TrunkNet
from utils.config import load_config


def _build_model_from_checkpoint(checkpoint: dict) -> DeepONet:
    """Reconstruct the DeepONet architecture from a saved checkpoint."""
    ckpt_config = checkpoint["config"]
    model_cfg = ckpt_config["model"]

    branch_dim = model_cfg.get("branch_input_dim", 120)
    if branch_dim == "auto":
        data_cfg = ckpt_config.get("data", {})
        duration_s = float(data_cfg.get("duration_s", 3600.0))
        dt_ctrl_s = float(data_cfg.get("dt_ctrl_s", 30.0))
        branch_dim = int(duration_s / dt_ctrl_s)
    branch = BranchNet(
        input_dim=int(branch_dim),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        output_dim=int(model_cfg.get("latent_dim", 128)),
    )
    trunk = TrunkNet(
        input_dim=int(model_cfg.get("trunk_input_dim", 2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        output_dim=int(model_cfg.get("latent_dim", 128)),
    )
    model = DeepONet(branch, trunk)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _resolve_scenario(env_config: dict) -> dict:
    """Merge env-config scenario fields with values from data.sumo_config.

    The env config may either spell out every scenario constant
    (N_x, T_ctrl, ...) or point at a SUMO config file via `sumo_config`.
    This matches how `surrogate/train.py` resolves things.
    """
    cfg = dict(env_config)
    sumo_config_path = cfg.get("sumo_config")
    if sumo_config_path:
        sumo_cfg = load_config(str(sumo_config_path))
        net_cfg = sumo_cfg["network"]
        sim_cfg = sumo_cfg["simulation"]
        det_cfg = sumo_cfg["detectors"]
        demand_cfg = sumo_cfg["demand"]

        cfg.setdefault("highway_length_m", float(net_cfg["highway_length_m"]))
        cfg.setdefault("duration_s", float(sim_cfg["duration_s"]))
        cfg.setdefault("dt_ctrl_s", float(sim_cfg["dt_ctrl_s"]))
        cfg.setdefault("N_x", int(det_cfg["n_detectors"]))
        cfg.setdefault("detector_spacing_m", float(det_cfg["spacing_m"]))
        cfg.setdefault(
            "detector_start_position_m",
            float(det_cfg.get("start_position_m", 2.0 * float(det_cfg["spacing_m"]))),
        )
        cfg.setdefault("ramp_demand_vph", float(demand_cfg["ramp_demand_vph"]))
        if not cfg.get("demand_profiles"):
            cfg["demand_profiles"] = [float(demand_cfg["mainline_demand_vph"])]
    return cfg


class SurrogateEnv(gym.Env):
    """Gymnasium environment wrapping a trained DeepONet surrogate."""

    metadata: dict = {"render_modes": []}

    def __init__(self, surrogate_checkpoint: str, config: dict) -> None:
        """
        Args:
            surrogate_checkpoint: Path to a trained DeepONet checkpoint (.pt).
            config: Environment config. Recognized keys:
                sumo_config (str): optional path to a SUMO config file to
                    auto-derive scenario constants from.
                demand_profiles (list[float]): demand values to sample
                    from at each reset.
                demand_min, demand_max (float): override demand-norm
                    bounds. Default to min/max of demand_profiles.
                ramp_demand_vph (float): max on-ramp inflow rate.
                queue_norm_scale (float): obs-side queue normalizer
                    (default 100).
                reward (dict): RewardWeights.from_config payload.
                N_x, T_ctrl, highway_length_m, duration_s, dt_ctrl_s,
                detector_spacing_m, detector_start_position_m: scenario
                    constants. Auto-filled from sumo_config if provided.
                density_mean, density_std (float): override the
                    surrogate's saved normalization. Default: use the
                    values stored in the checkpoint.
                device (str): "auto" | "cpu" | "cuda" | "mps". Default cpu.
        """
        super().__init__()
        cfg = _resolve_scenario(config or {})

        # Scenario constants
        self.N_x: int = int(cfg.get("N_x", 19))
        self.highway_length_m: float = float(cfg.get("highway_length_m", 2000.0))
        self.duration_s: float = float(cfg.get("duration_s", 3600.0))
        self.dt_ctrl_s: float = float(cfg.get("dt_ctrl_s", 30.0))
        self.T_ctrl: int = int(cfg.get("T_ctrl", self.duration_s / self.dt_ctrl_s))
        detector_spacing_m: float = float(cfg.get("detector_spacing_m", 100.0))
        detector_start_position_m: float = float(
            cfg.get("detector_start_position_m", 2.0 * detector_spacing_m)
        )

        # Demand
        demand_profiles = cfg.get("demand_profiles", [1500.0])
        if not demand_profiles:
            raise ValueError("config.demand_profiles must be non-empty")
        self.demand_profiles: list[float] = [float(d) for d in demand_profiles]
        self.demand_min: float = float(
            cfg.get("demand_min", min(self.demand_profiles))
        )
        self.demand_max: float = float(
            cfg.get("demand_max", max(self.demand_profiles))
        )

        # Queue and reward
        self.ramp_demand_vph: float = float(cfg.get("ramp_demand_vph", 800.0))
        self.queue_norm_scale: float = max(
            float(cfg.get("queue_norm_scale", 100.0)), 1e-6
        )
        self._reward_weights = RewardWeights.from_config(cfg.get("reward"))

        # Device
        device_name = cfg.get("device", "cpu")
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        # Load checkpoint and reconstruct the model
        checkpoint = torch.load(
            surrogate_checkpoint, map_location="cpu", weights_only=False
        )
        norm = checkpoint["normalization"]
        self.density_mean: float = float(
            cfg.get("density_mean", norm["mean_density"])
        )
        self.density_std: float = max(
            float(cfg.get("density_std", norm["std_density"])), 1e-6
        )

        self.model = _build_model_from_checkpoint(checkpoint).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Pre-build trunk grid: x positions match detectors.py formula.
        x_grid = np.array(
            [detector_start_position_m + i * detector_spacing_m for i in range(self.N_x)],
            dtype=np.float32,
        )
        t_grid = np.arange(self.T_ctrl, dtype=np.float32) * self.dt_ctrl_s
        self._x_grid_norm = (x_grid / self.highway_length_m).astype(np.float32)
        self._t_grid_norm = (t_grid / self.duration_s).astype(np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.N_x + 3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Episode state (set by reset())
        self._rng: np.random.Generator | None = None
        self._u_history: np.ndarray | None = None
        self._k: int = 0
        self._current_demand_vph: float = float(self.demand_profiles[0])
        self._current_density_phys: np.ndarray = np.zeros(self.N_x, dtype=np.float32)
        self._analytical_queue: float = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Sample a demand profile and reset the episode state."""
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        options = options or {}
        if "demand_vph" in options:
            self._current_demand_vph = float(options["demand_vph"])
        else:
            idx = int(self._rng.integers(0, len(self.demand_profiles)))
            self._current_demand_vph = float(self.demand_profiles[idx])

        self._u_history = np.zeros(self.T_ctrl, dtype=np.float32)
        self._k = 0
        self._analytical_queue = 0.0

        density_norm = self._predict_density(self._u_history, self._k)
        self._current_density_phys = self._denormalize_density(density_norm)
        obs = self._build_observation(density_norm)
        info = {
            "demand_vph": self._current_demand_vph,
            "analytical_queue": self._analytical_queue,
            "k": self._k,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action, query surrogate, update queue, compute reward."""
        if self._u_history is None:
            raise RuntimeError("step() called before reset()")

        u_k = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        self._u_history[self._k] = u_k

        density_norm = self._predict_density(self._u_history, self._k)
        density_phys = self._denormalize_density(density_norm)
        self._current_density_phys = density_phys

        # Analytical queue update (mirrors SumoEnv exactly).
        queue_growth = (1.0 - u_k) * self.ramp_demand_vph * self.dt_ctrl_s / 3600.0
        self._analytical_queue = max(0.0, self._analytical_queue + queue_growth)
        reward = compute_reward(
            density_phys,
            queue_length=self._analytical_queue,
            weights=self._reward_weights,
        )

        self._k += 1
        terminated = self._k >= self.T_ctrl
        truncated = False

        # Build obs for the next state. On the final step we report the
        # just-predicted density; otherwise we run one more forward pass at
        # the new time index to give the policy a fresh state to act on.
        if terminated:
            next_density_norm = density_norm
            next_k_for_time = self.T_ctrl
        else:
            next_density_norm = self._predict_density(self._u_history, self._k)
            self._current_density_phys = self._denormalize_density(next_density_norm)
            next_k_for_time = self._k
        obs = self._build_observation(next_density_norm, time_index=next_k_for_time)

        info = {
            "density_phys": density_phys,
            "mean_density": float(np.mean(density_phys)),
            "std_density": float(np.std(density_phys)),
            "u": u_k,
            "k": self._k,
            "demand_vph": self._current_demand_vph,
            "analytical_queue": self._analytical_queue,
        }
        return obs, reward, terminated, truncated, info

    def _predict_density(self, u_history: np.ndarray, k: int) -> np.ndarray:
        """Forward the surrogate at time index k, return z-normalized density."""
        branch = torch.from_numpy(u_history).to(self.device).unsqueeze(0)
        t_k_norm = float(self._t_grid_norm[min(k, self.T_ctrl - 1)])
        trunk_np = np.stack(
            [
                self._x_grid_norm,
                np.full(self.N_x, t_k_norm, dtype=np.float32),
            ],
            axis=-1,
        )
        trunk = torch.from_numpy(trunk_np).to(self.device).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(branch, trunk)
        return pred.squeeze(0).cpu().numpy().astype(np.float32)

    def _denormalize_density(self, density_norm: np.ndarray) -> np.ndarray:
        return (density_norm * self.density_std + self.density_mean).astype(np.float32)

    def _build_observation(
        self, density_norm: np.ndarray, time_index: int | None = None
    ) -> np.ndarray:
        if time_index is None:
            time_index = self._k

        span = self.demand_max - self.demand_min
        if span <= 1e-6:
            demand_norm = 0.0
        else:
            demand_norm = (self._current_demand_vph - self.demand_min) / span

        time_norm = float(time_index) / float(self.T_ctrl)
        queue_norm = float(self._analytical_queue / self.queue_norm_scale)
        return np.concatenate(
            [
                density_norm.astype(np.float32),
                np.array(
                    [demand_norm, time_norm, queue_norm],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)


def find_latest_checkpoint(runs_dir: str | Path) -> Path | None:
    """Helper: return the latest deeponet_constant_inflow_*/best.pt under runs_dir.

    Returns None if no matching checkpoint exists.
    """
    candidates = sorted(
        Path(runs_dir).glob("deeponet_constant_inflow_*/best.pt")
    )
    return candidates[-1] if candidates else None
