"""
Gymnasium environment backed by the trained DeepONet surrogate.

At each step, the agent provides a ramp metering action; the surrogate
predicts the resulting density field; the reward is computed and returned.

Observation (shape (N_x + 3,); 22 features for N_x=19; +1 ramp-arrival feature
when env.observe_ramp_demand is true, inserted after `demand`):
    density[0:N_x]  — z-score normalized density at detector locations
    demand[N_x]     — min-max normalized current mainline demand ∈ [0, 1]
    time[N_x+1]     — normalized time index k / T_ctrl ∈ [0, 1]
    queue[N_x+2]    — virtual ramp queue length normalized by queue_scale

Action (shape (1,)):
    ramp metering rate ∈ [0, 1] (continuous Box)

Virtual ramp queue:
    The surrogate predicts mainline density only, so ramp queue length is
    tracked analytically from ramp demand and metering actions:
    q_next = max(q_prev + arrivals - released, 0).
    Queue length is reported in info, included in the observation, and
    penalized in the reward.

Surrogate rollout strategy (zero-padded partial control sequences):
    At step k, branch input = [u(0),...,u(k), 0,...,0]
    Trunk queries density at all (x_i, t_k).
    DeepONet is re-evaluated from scratch each step (not autoregressive).

The current baseline DeepONet uses ramp_control only as its branch input.
Demand is still exposed in the observation for interface parity with SumoEnv,
but single-demand checkpoints are the intended Phase 1 surrogate setting.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from rl.reward import RewardWeights, reward_terms as compute_reward_terms
from sumo_env.detectors import get_x_grid
from surrogate.deeponet import BranchNet, DeepONet, TrunkNet
from surrogate.train import apply_sumo_config_defaults, resolve_branch_input_dim
from utils.config import load_config, merge_configs


def _build_model(config: dict) -> DeepONet:
    model_cfg = config["model"]
    return DeepONet(
        BranchNet(
            input_dim=resolve_branch_input_dim(config),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            output_dim=int(model_cfg.get("latent_dim", 128)),
        ),
        TrunkNet(
            input_dim=int(model_cfg.get("trunk_input_dim", 2)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            output_dim=int(model_cfg.get("latent_dim", 128)),
        ),
    )


class SurrogateEnv(gym.Env):
    """Gymnasium environment wrapping a trained DeepONet surrogate."""

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        surrogate_checkpoint: str | None = None,
        config: dict | None = None,
    ) -> None:
        """
        Args:
            surrogate_checkpoint: Path to trained DeepONet checkpoint (.pt).
            config: Environment config (N_x, T_ctrl, demand_profiles, normalization stats, etc.).
        """
        super().__init__()
        self.env_config = copy.deepcopy(config or {})
        self.project_root = Path(
            self.env_config.get("project_root", Path.cwd())
        ).resolve()

        checkpoint_path = surrogate_checkpoint or self.env_config.get(
            "surrogate_checkpoint"
        )
        if checkpoint_path is None:
            raise ValueError(
                "SurrogateEnv requires a checkpoint path, either as the "
                "surrogate_checkpoint argument or config['surrogate_checkpoint']."
            )
        self.checkpoint_path = self._resolve_path(checkpoint_path)

        self.device = self._resolve_device(
            str(self.env_config.get("device", "auto"))
        )
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"{self.checkpoint_path} does not contain 'model_state_dict'."
            )

        checkpoint_config = copy.deepcopy(checkpoint.get("config") or self.env_config)
        if "project_root" in self.env_config:
            checkpoint_config["project_root"] = str(self.project_root)
        else:
            checkpoint_config.setdefault("project_root", str(self.project_root))
        self.model_config = apply_sumo_config_defaults(checkpoint_config)
        self.branch_input_dim = resolve_branch_input_dim(self.model_config)

        self.sumo_config = self._load_sumo_config(self.env_config, self.model_config)
        sim_cfg = self.sumo_config["simulation"]
        demand_cfg = self.sumo_config["demand"]
        net_cfg = self.sumo_config["network"]

        self.dt_ctrl = int(sim_cfg["dt_ctrl_s"])
        self.duration_s = float(sim_cfg["duration_s"])
        self.warmup_s = float(sim_cfg.get("ramp_warmup_s", 0.0))
        self.T_ctrl = int(self.duration_s / self.dt_ctrl)
        if self.branch_input_dim != self.T_ctrl:
            raise ValueError(
                "SurrogateEnv currently supports DeepONet checkpoints whose "
                "branch input is exactly ramp_control with length T_ctrl. "
                f"Got branch_input_dim={self.branch_input_dim}, T_ctrl={self.T_ctrl}."
            )

        self.highway_length_m = float(net_cfg["highway_length_m"])
        self.x_grid = get_x_grid(self.sumo_config)
        self.N_x = int(self.x_grid.shape[0])
        self.t_grid = np.arange(self.T_ctrl, dtype=np.float32) * self.dt_ctrl

        self.demand_levels = [
            float(v)
            for v in self.env_config.get(
                "demand_levels", [demand_cfg["mainline_demand_vph"]]
            )
        ]
        if not self.demand_levels:
            raise ValueError("env.demand_levels must contain at least one value.")
        if len({round(v, 9) for v in self.demand_levels}) > 1:
            warnings.warn(
                "The loaded surrogate uses ramp_control only as branch input, "
                "so different demand_levels change the observation label but "
                "not the surrogate dynamics. Train a demand-conditioned "
                "surrogate before relying on multi-demand surrogate RL.",
                stacklevel=2,
            )
        self.min_demand = float(
            self.env_config.get("min_demand", min(self.demand_levels))
        )
        self.max_demand = float(
            self.env_config.get("max_demand", max(self.demand_levels))
        )

        normalization = checkpoint.get("normalization", {})
        if "mean_density" not in normalization or "std_density" not in normalization:
            if (
                "density_mean" not in self.env_config
                or "density_std" not in self.env_config
            ):
                raise KeyError(
                    "Surrogate checkpoint must contain normalization "
                    "{'mean_density', 'std_density'}, or env config must provide "
                    "density_mean and density_std."
                )
            normalization = {
                "mean_density": self.env_config["density_mean"],
                "std_density": self.env_config["density_std"],
            }
        self.density_mean = float(normalization["mean_density"])
        self.density_std = max(float(normalization["std_density"]), 1e-6)

        queue_cfg = self.env_config.get("queue", {}) or {}
        self.ramp_arrival_vph = float(
            self.env_config.get(
                "ramp_arrival_vph",
                queue_cfg.get("arrival_vph", demand_cfg["ramp_demand_vph"]),
            )
        )
        self.ramp_discharge_vph = float(
            self.env_config.get(
                "ramp_discharge_vph",
                queue_cfg.get(
                    "discharge_vph",
                    demand_cfg.get("ramp_discharge_vph", self.ramp_arrival_vph),
                ),
            )
        )
        if self.ramp_arrival_vph < 0.0:
            raise ValueError("ramp_arrival_vph must be non-negative.")
        if self.ramp_discharge_vph < 0.0:
            raise ValueError("ramp_discharge_vph must be non-negative.")
        # Per-episode ramp arrival rate (parity with SumoEnv.ramp_demand_levels).
        self.ramp_demand_levels = [
            float(v) for v in self.env_config.get("ramp_demand_levels", [self.ramp_arrival_vph])
        ]
        if not self.ramp_demand_levels or any(v < 0.0 for v in self.ramp_demand_levels):
            raise ValueError("env.ramp_demand_levels must be a non-empty list of rates >= 0")
        self.min_ramp_demand = float(min(self.ramp_demand_levels))
        self.max_ramp_demand = float(max(self.ramp_demand_levels))
        self.current_ramp_demand_vph = float(self.ramp_demand_levels[0])
        self.observe_ramp_demand = bool(self.env_config.get("observe_ramp_demand", False))
        if self.ramp_discharge_vph > max(self.ramp_demand_levels) + 1e-6 and not self.env_config.get(
            "quiet_ramp_discharge_warning", False
        ):
            warnings.warn(
                "ramp_discharge_vph is greater than the ramp arrival rate (M7 §7.10: "
                "u is the green fraction of the discharge capacity). The DeepONet "
                "was trained on ramp_control as a fraction of ramp *demand*, so the "
                "surrogate sees the metering action u directly; density predictions "
                "for u > arrival/discharge are extrapolation.",
                stacklevel=2,
            )

        reward_cfg = self.env_config.get("reward", {}) or {}
        self.reward_weights = RewardWeights.from_config(reward_cfg)
        self.queue_scale = float(
            self.env_config.get(
                "queue_scale",
                self.env_config.get(
                    "queue_norm_scale",
                    queue_cfg.get(
                        "scale",
                        reward_cfg.get(
                            "queue_scale",
                            reward_cfg.get(
                                "queue_norm",
                                max(
                                    self.ramp_arrival_vph
                                    * self.duration_s
                                    / 3600.0,
                                    1.0,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )
        if not np.isfinite(self.queue_scale):
            raise ValueError("queue_scale must be finite.")
        if self.queue_scale <= 0.0:
            raise ValueError("queue_scale must be positive.")

        self.reward_warmup_s = float(
            self.env_config.get(
                "reward_warmup_s",
                reward_cfg.get("warmup_s", 0.0),
            )
        )
        if not np.isfinite(self.reward_warmup_s):
            raise ValueError("reward_warmup_s must be finite.")
        if self.reward_warmup_s < 0.0:
            raise ValueError("reward_warmup_s must be non-negative.")

        self.model = _build_model(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.N_x + 3 + int(self.observe_ramp_demand),),
            dtype=np.float32,
        )

        self.base_seed = int(self.env_config.get("seed", sim_cfg.get("seed", 42)))
        self.dt_ctrl_s = self.dt_ctrl
        self.ramp_demand_vph = self.ramp_arrival_vph
        self.rng = np.random.default_rng(self.base_seed)
        self.episode_index = 0
        self.k = 0
        self.current_demand_vph = float(self.demand_levels[0])
        self.current_density = np.zeros(self.N_x, dtype=np.float32)
        self.current_queue_length = 0.0
        self._queue_samples: list[float] = []
        self.action_history = np.zeros(self.T_ctrl, dtype=np.float32)
        self._started = False

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Sample a demand profile and reset the episode state.

        Returns:
            observation: shape (N_x + 3,)
            info: dict
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        if "demand_vph" in options:
            demand_vph = float(options["demand_vph"])
        else:
            demand_vph = float(self.rng.choice(self.demand_levels))

        if "ramp_demand_vph" in options:
            self.current_ramp_demand_vph = float(options["ramp_demand_vph"])
        else:
            self.current_ramp_demand_vph = float(self.rng.choice(self.ramp_demand_levels))

        self.episode_index += 1
        self.k = 0
        self.current_demand_vph = demand_vph
        self.current_density = np.zeros(self.N_x, dtype=np.float32)
        self.current_queue_length = 0.0
        self._queue_samples = []
        self.action_history = np.zeros(self.T_ctrl, dtype=np.float32)
        self._started = True

        obs = self._make_observation()
        info = {
            "demand_vph": self.current_demand_vph,
            "ramp_demand_vph": self.current_ramp_demand_vph,
            "time_s": 0.0,
            "k": 0,
            "analytical_queue": 0.0,
            "backend": "surrogate",
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action, query surrogate, compute reward, advance time.

        Args:
            action: shape (1,), ramp metering rate ∈ [0, 1]

        Returns:
            observation: shape (N_x + 3,)
            reward: float
            terminated: bool (True when episode ends at T_ctrl steps)
            truncated: bool (always False in Phase 1)
            info: dict
        """
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        if self.k >= self.T_ctrl:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        ramp_rate = float(
            np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0)
        )
        query_k = self.k
        self.action_history[query_k] = ramp_rate
        queue_info = self._advance_virtual_queue(ramp_rate, query_k)
        pred_density_norm = self._predict_density_norm(query_k)
        density = pred_density_norm * self.density_std + self.density_mean
        density = self._clip_density(density).astype(np.float32, copy=False)
        density_norm = ((density - self.density_mean) / self.density_std).astype(
            np.float32,
            copy=False,
        )

        self.current_density = density
        # The DeepONet predicts density only, so no outflow measurement is
        # available here; reward_terms() requires delta == 0 in that case
        # (two-term queue + std reward on the surrogate path).
        reward_terms = self._reward_terms(
            density, self.current_queue_length, outflow_vph=None
        )
        raw_reward = float(reward_terms["reward"])
        reward_warmup_active = self._reward_warmup_active(query_k)
        reward = 0.0 if reward_warmup_active else raw_reward

        self.k += 1
        terminated = self.k >= self.T_ctrl
        truncated = False
        obs = self._make_observation()

        info = {
            "time_s": float(min(self.k, self.T_ctrl) * self.dt_ctrl),
            "query_time_s": float(self.t_grid[query_k]),
            "ramp_rate": ramp_rate,
            "density": density.copy(),
            "density_norm": density_norm.astype(np.float32, copy=True),
            "mean_density": reward_terms["mean_density"],
            "std_density": reward_terms["std_density"],
            "outflow_vph": reward_terms["outflow_vph"],
            "lost_outflow_frac": reward_terms["lost_outflow_frac"],
            "outflow_penalty": reward_terms["outflow_penalty"],
            "queue_length": float(self.current_queue_length),
            "analytical_queue": float(self.current_queue_length),
            "queue_penalty": reward_terms["queue_penalty"],
            "queue_scale": float(self.queue_scale),
            "queue_norm": self._normalize_queue(self.current_queue_length),
            "std_penalty": reward_terms["std_penalty"],
            "reward_delta": float(self.reward_weights.delta),
            "reward_beta": float(self.reward_weights.beta),
            "reward_gamma": float(self.reward_weights.gamma),
            "reward_q_ref": float(self.reward_weights.q_ref),
            "reward_sigma_ref": float(self.reward_weights.sigma_ref),
            "reward_queue_norm": float(self.reward_weights.queue_norm),
            "raw_reward": float(raw_reward),
            "reward_warmup_active": float(reward_warmup_active),
            "reward_warmup_s": float(self.reward_warmup_s),
            "episode_queue_mean": float(np.mean(self._queue_samples))
            if self._queue_samples
            else 0.0,
            "episode_queue_max": float(max(self._queue_samples))
            if self._queue_samples
            else 0.0,
            "demand_vph": self.current_demand_vph,
            "ramp_demand_vph": self.current_ramp_demand_vph,
            "k": int(self.k),
            "u": ramp_rate,
            "backend": "surrogate",
            **queue_info,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Mark the environment as inactive."""
        self._started = False

    @torch.no_grad()
    def _predict_density_norm(self, query_k: int) -> np.ndarray:
        coords = self._trunk_query(query_k)
        branch = torch.from_numpy(self.action_history).unsqueeze(0).to(self.device)
        trunk = torch.from_numpy(coords).unsqueeze(0).to(self.device)
        pred = self.model(branch, trunk).squeeze(0).detach().cpu().numpy()
        if pred.shape != (self.N_x,):
            raise RuntimeError(
                f"Surrogate predicted shape {pred.shape}; expected ({self.N_x},)."
            )
        if not np.all(np.isfinite(pred)):
            raise RuntimeError("Surrogate prediction contains NaN or Inf values.")
        return pred.astype(np.float32, copy=False)

    def _trunk_query(self, query_k: int) -> np.ndarray:
        x_norm = self.x_grid / self.highway_length_m
        t_norm = np.full(
            self.N_x,
            float(self.t_grid[query_k] / self.duration_s),
            dtype=np.float32,
        )
        return np.stack([x_norm.astype(np.float32), t_norm], axis=-1)

    def _make_observation(self) -> np.ndarray:
        density_norm = (self.current_density - self.density_mean) / self.density_std
        demand_norm = self._normalize_demand(self.current_demand_vph)
        time_norm = float(min(self.k, self.T_ctrl) / max(self.T_ctrl, 1))
        queue_norm = self._normalize_queue(self.current_queue_length)
        scalars = [demand_norm]
        if self.observe_ramp_demand:
            scalars.append(self._normalize_ramp_demand(self.current_ramp_demand_vph))
        scalars += [time_norm, queue_norm]
        return np.concatenate(
            [
                density_norm.astype(np.float32),
                np.array(scalars, dtype=np.float32),
            ]
        )

    def _normalize_ramp_demand(self, ramp_demand_vph: float) -> float:
        span = self.max_ramp_demand - self.min_ramp_demand
        if span <= 1e-6:
            return 0.0
        return float((ramp_demand_vph - self.min_ramp_demand) / span)

    def _advance_virtual_queue(self, ramp_rate: float, query_k: int) -> dict:
        queue_before = float(self.current_queue_length)
        interval_start_s = float(self.t_grid[query_k])
        arrivals = 0.0
        if interval_start_s >= self.warmup_s:
            arrivals = self.current_ramp_demand_vph * self.dt_ctrl / 3600.0

        release_capacity = ramp_rate * self.ramp_discharge_vph * self.dt_ctrl / 3600.0
        available = queue_before + arrivals
        released = min(available, release_capacity)
        self.current_queue_length = max(available - released, 0.0)
        self._queue_samples.append(self.current_queue_length)

        return {
            "queue_before": queue_before,
            "queue_after": float(self.current_queue_length),
            "ramp_arrivals": float(arrivals),
            "ramp_released": float(released),
            "ramp_release_capacity": float(release_capacity),
        }

    def _reward_warmup_active(self, query_k: int) -> bool:
        return float(self.t_grid[query_k]) < self.reward_warmup_s

    def _normalize_demand(self, demand_vph: float) -> float:
        span = self.max_demand - self.min_demand
        if span <= 1e-6:
            return 0.0
        return float((demand_vph - self.min_demand) / span)

    def _normalize_queue(self, queue_length: float) -> float:
        return float(max(queue_length, 0.0) / self.queue_scale)

    def _reward_terms(
        self, density: np.ndarray, queue_length: float, outflow_vph: float | None
    ) -> dict[str, float]:
        return compute_reward_terms(
            density, queue_length, outflow_vph, self.reward_weights
        )

    def _clip_density(self, density: np.ndarray) -> np.ndarray:
        clip_min_raw = self.env_config.get("density_clip_min", 0.0)
        clip_min = max(float(clip_min_raw), 0.0) if clip_min_raw is not None else 0.0
        clip_max = self.env_config.get("density_clip_max")
        if clip_min == 0.0 and clip_max is None:
            return np.maximum(density, 0.0)
        if clip_max is None:
            return np.maximum(density, clip_min)
        if float(clip_max) < clip_min:
            raise ValueError("density_clip_max must be greater than density_clip_min.")
        return np.clip(density, clip_min, float(clip_max))

    def _load_sumo_config(self, env_config: dict, model_config: dict) -> dict:
        if "sumo" in env_config:
            sumo_config = copy.deepcopy(env_config["sumo"])
        elif "sumo_config" in env_config:
            sumo_config = load_config(str(self._resolve_path(env_config["sumo_config"])))
        elif all(
            k in env_config for k in ("network", "simulation", "demand", "detectors")
        ):
            sumo_config = copy.deepcopy(env_config)
        else:
            data_cfg = model_config.get("data", {})
            sumo_config_path = data_cfg.get("sumo_config")
            if not sumo_config_path:
                raise KeyError(
                    "SurrogateEnv config must provide 'sumo_config', a nested "
                    "'sumo' dict, direct SUMO config keys, or use a checkpoint "
                    "whose config.data.sumo_config is available."
                )
            model_project_root = Path(
                model_config.get("project_root", self.project_root)
            ).resolve()
            sumo_path = Path(sumo_config_path)
            if not sumo_path.is_absolute():
                sumo_path = model_project_root / sumo_path
            sumo_config = load_config(str(sumo_path))

        overrides = env_config.get("sumo_overrides", {})
        if overrides:
            sumo_config = merge_configs(sumo_config, overrides)
        return sumo_config

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.project_root / p

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)


def find_latest_checkpoint(runs_dir: str | Path) -> Path | None:
    """Helper: return the latest deeponet_constant_inflow_*/best.pt under runs_dir.

    Used by scripts/eval_constant_baselines.py and tests/test_surrogate_env.py.
    (Originally added in M4, dropped in commit 1672424 while its callers kept
    importing it; restored 2026-08-27.) Returns None if no checkpoint exists.
    """
    candidates = sorted(Path(runs_dir).glob("deeponet_constant_inflow_*/best.pt"))
    return candidates[-1] if candidates else None
