"""
Gymnasium environment backed by live SUMO simulation via TraCI.

This is the direct SUMO+RL path: the policy chooses one ramp-metering action
per control interval, SUMO advances for that interval, and detector aggregates
become the next observation. It does not use the surrogate model.

Observation (shape (N_x + 2,) = (22,) for the Phase 1 configs):
    density[0:N_x]  — z-score normalized density at detector locations
    demand[N_x]     — min-max normalized current mainline demand in [0, 1]
    time[N_x+1]     — normalized control index k / T_ctrl in [0, 1]

Action (shape (1,)):
    ramp metering rate in [0, 1] (continuous Box)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import traci
except ImportError as exc:
    raise ImportError(
        "TraCI not found. Install SUMO >= 1.18 and ensure the SUMO Python "
        "bindings are on PYTHONPATH, for example: "
        "export PYTHONPATH=$SUMO_HOME/share/sumo/tools:$PYTHONPATH"
    ) from exc

from rl.reward import RewardWeights, compute_reward
from sumo_env.detectors import build_detector_file, get_detector_ids_per_lane, get_x_grid
from sumo_env.network_builder import build_network, _write_routes
from utils.config import load_config, merge_configs


class SumoEnv(gym.Env):
    """Gymnasium environment wrapping SUMO via TraCI."""

    metadata: dict = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: SUMO RL environment config. Usually this is the ``env``
                    section from configs/rl/ppo_sumo.yaml.
        """
        super().__init__()
        self.env_config = copy.deepcopy(config)
        self.project_root = Path(
            self.env_config.get("project_root", Path.cwd())
        ).resolve()
        self.sumo_config = self._load_sumo_config(self.env_config)

        sim_cfg = self.sumo_config["simulation"]
        det_cfg = self.sumo_config["detectors"]
        demand_cfg = self.sumo_config["demand"]

        self.step_len = float(sim_cfg["step_length_s"])
        self.dt_ctrl = int(sim_cfg["dt_ctrl_s"])
        self.dt_ctrl_steps = int(round(self.dt_ctrl / self.step_len))
        if not np.isclose(self.dt_ctrl_steps * self.step_len, self.dt_ctrl):
            raise ValueError("dt_ctrl_s must be an integer multiple of step_length_s")

        self.T_ctrl = int(sim_cfg["duration_s"] / self.dt_ctrl)
        self.warmup_s = float(sim_cfg.get("ramp_warmup_s", 0.0))
        self.base_seed = int(sim_cfg.get("seed", self.env_config.get("seed", 42)))
        self.sumo_binary = str(self.env_config.get("sumo_binary", sim_cfg["sumo_binary"]))
        self.ramp_demand_vph = float(demand_cfg["ramp_demand_vph"])
        self.vehicle_length_m = float(det_cfg["vehicle_length_m"])

        self.demand_levels = [
            float(v)
            for v in self.env_config.get(
                "demand_levels", [demand_cfg["mainline_demand_vph"]]
            )
        ]
        if not self.demand_levels:
            raise ValueError("env.demand_levels must contain at least one demand level")
        self.min_demand = float(min(self.demand_levels))
        self.max_demand = float(max(self.demand_levels))

        self.density_mean = float(self.env_config.get("density_mean", 0.0))
        self.density_std = max(float(self.env_config.get("density_std", 1.0)), 1e-6)

        # Shaped reward (Phase 1): -alpha*mean(density) -beta*queue -gamma*std(density).
        # See proposal.md §"Reward (Phase 1 shaped)" for term motivation and
        # the analytical queue model shared with SurrogateEnv.
        self._reward_weights = RewardWeights.from_config(
            self.env_config.get("reward")
        )
        self.queue_norm_scale = max(
            float(self.env_config.get("queue_norm_scale", 100.0)), 1e-6
        )

        self.det_ids_per_lane = get_detector_ids_per_lane(self.sumo_config)
        self.x_grid = get_x_grid(self.sumo_config)
        self.N_x = len(self.det_ids_per_lane)
        self.t_grid = np.arange(self.T_ctrl, dtype=np.float32) * self.dt_ctrl

        self.network_dir = self._resolve_path(
            self.env_config.get(
                "network_dir",
                self.sumo_config.get("output", {}).get("network_dir", "data/raw/rl_network"),
            )
        )
        self.network_dir.mkdir(parents=True, exist_ok=True)
        self.network_files = build_network(str(self.network_dir), self.sumo_config)
        self.detector_file = str((self.network_dir / "detectors.add.xml").resolve())
        build_detector_file(self.detector_file, self.sumo_config)
        self.routes_by_demand: dict[float, str] = {}
        for demand_vph in self.demand_levels:
            self._route_for_demand(demand_vph)

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.N_x + 3,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(self.base_seed)
        self.episode_index = 0
        self.k = 0
        self.current_demand_vph = float(self.demand_levels[0])
        self.current_density = np.zeros(self.N_x, dtype=np.float32)
        self._analytical_queue = 0.0
        self._started = False
        self._veh_counter = 0
        self._frac_accumulator = 0.0
        self._insert_attempts = 0
        self._insert_success = 0
        self._insert_rejected = 0
        self._teleports = 0
        self._arrived_vehicles = 0
        self._queue_samples: list[int] = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new SUMO simulation with a sampled demand profile.

        Returns:
            observation: shape (N_x + 2,) = (22,)
            info: dict
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.close()
        options = options or {}
        if "demand_vph" in options:
            demand_vph = float(options["demand_vph"])
        else:
            demand_vph = float(self.rng.choice(self.demand_levels))
        self.current_demand_vph = demand_vph

        if "sumo_seed" in options:
            sim_seed = int(options["sumo_seed"])
        elif seed is not None:
            sim_seed = int(seed)
        else:
            sim_seed = self.base_seed + self.episode_index
        route_file = self._route_for_demand(demand_vph)
        sumo_cmd = [
            self.sumo_binary,
            "--net-file",
            self.network_files["net"],
            "--route-files",
            route_file,
            "--additional-files",
            self.detector_file,
            "--step-length",
            str(self.step_len),
            "--seed",
            str(sim_seed),
            "--no-step-log",
            "--no-warnings",
            "--collision.action",
            "warn",
        ]

        traci.start(sumo_cmd)
        self._started = True
        self.episode_index += 1
        self.k = 0
        self.current_density = np.zeros(self.N_x, dtype=np.float32)
        self._analytical_queue = 0.0
        self._veh_counter = 0
        self._frac_accumulator = 0.0
        self._insert_attempts = 0
        self._insert_success = 0
        self._insert_rejected = 0
        self._teleports = 0
        self._arrived_vehicles = 0
        self._queue_samples = []

        obs = self._make_observation()
        info = {
            "demand_vph": self.current_demand_vph,
            "sumo_seed": sim_seed,
            "time_s": 0.0,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply ramp metering, advance SUMO, read detectors, compute reward.

        Args:
            action: shape (1,), ramp metering rate ∈ [0, 1]

        Returns:
            observation: shape (N_x + 2,)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        if not self._started:
            raise RuntimeError("Call reset() before step().")

        ramp_rate = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        density, speed, flow, interval_info = self._advance_control_interval(ramp_rate)
        self.current_density = density

        # Analytical queue update (mirrors SurrogateEnv for env parity; SUMO's
        # measured queue is still in interval_info / info for diagnostics).
        queue_growth = (1.0 - ramp_rate) * self.ramp_demand_vph * self.dt_ctrl / 3600.0
        self._analytical_queue = max(0.0, self._analytical_queue + queue_growth)
        reward = compute_reward(
            density,
            queue_length=self._analytical_queue,
            weights=self._reward_weights,
        )

        self.k += 1
        terminated = self.k >= self.T_ctrl
        truncated = False
        obs = self._make_observation()

        info = {
            "time_s": float(min(self.k, self.T_ctrl) * self.dt_ctrl),
            "ramp_rate": ramp_rate,
            "density": density.copy(),
            "speed": speed.copy(),
            "flow": flow.copy(),
            "mean_density": float(np.mean(density)),
            "std_density": float(np.std(density)),
            "mean_speed": float(np.mean(speed)),
            "mean_flow": float(np.mean(flow)),
            "demand_vph": self.current_demand_vph,
            "analytical_queue": self._analytical_queue,
            "arrived_vehicles": self._arrived_vehicles,
            "throughput_vph": self._throughput_vph(),
            "insert_attempts": self._insert_attempts,
            "insert_success": self._insert_success,
            "insert_rejected": self._insert_rejected,
            "teleports": self._teleports,
            **interval_info,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the TraCI connection and terminate SUMO."""
        if not self._started:
            return
        try:
            traci.close(False)
        except Exception:
            pass
        finally:
            self._started = False

    def _advance_control_interval(
        self, ramp_rate: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        sum_count = np.zeros(self.N_x, dtype=np.float64)
        sum_speed = np.zeros(self.N_x, dtype=np.float64)
        speed_count = np.zeros(self.N_x, dtype=np.int32)
        sum_occ = np.zeros(self.N_x, dtype=np.float64)
        interval_queue: list[int] = []
        interval_teleports = 0
        interval_insert_attempts = 0
        interval_insert_success = 0
        interval_insert_rejected = 0
        interval_arrived = 0

        insert_rate = ramp_rate * self.ramp_demand_vph / 3600.0

        for _ in range(self.dt_ctrl_steps):
            if traci.simulation.getTime() >= self.warmup_s:
                self._frac_accumulator += insert_rate * self.step_len
                n_insert = int(self._frac_accumulator)
                self._frac_accumulator -= n_insert
                for _ in range(n_insert):
                    self._insert_attempts += 1
                    interval_insert_attempts += 1
                    try:
                        traci.vehicle.add(
                            vehID=f"ramp_{self._veh_counter}",
                            routeID="route_ramp",
                            typeID="passenger",
                            depart=str(traci.simulation.getTime()),
                            departLane="first",
                            departPos="free",
                            departSpeed="0",
                        )
                        self._veh_counter += 1
                        self._insert_success += 1
                        interval_insert_success += 1
                    except traci.exceptions.TraCIException:
                        self._insert_rejected += 1
                        interval_insert_rejected += 1

            traci.simulationStep()

            teleports = int(traci.simulation.getStartingTeleportNumber())
            arrived = int(traci.simulation.getArrivedNumber())
            self._teleports += teleports
            self._arrived_vehicles += arrived
            interval_teleports += teleports
            interval_arrived += arrived

            queue_len = int(traci.edge.getLastStepVehicleNumber("ramp"))
            self._queue_samples.append(queue_len)
            interval_queue.append(queue_len)

            for j, lane_ids in enumerate(self.det_ids_per_lane):
                for det_id in lane_ids:
                    count = traci.inductionloop.getLastStepVehicleNumber(det_id)
                    spd_raw = traci.inductionloop.getLastStepMeanSpeed(det_id)
                    occ = traci.inductionloop.getLastStepOccupancy(det_id)

                    sum_count[j] += count
                    sum_occ[j] += occ
                    if spd_raw >= 0.0:
                        sum_speed[j] += spd_raw * count
                        speed_count[j] += count

        flow_vph = sum_count / (self.dt_ctrl_steps * self.step_len) * 3600.0
        mean_speed_mps = np.where(
            speed_count > 0,
            sum_speed / np.maximum(speed_count, 1),
            0.0,
        )
        mean_speed_kmph = mean_speed_mps * 3.6

        mean_occ_frac = sum_occ / (self.dt_ctrl_steps * 100.0)
        density_occ = mean_occ_frac * (1000.0 / self.vehicle_length_m)
        density_fd = np.where(
            mean_speed_kmph > 5.0,
            flow_vph / np.maximum(mean_speed_kmph, 1e-6),
            density_occ,
        )

        density = density_fd.astype(np.float32)
        speed = mean_speed_kmph.astype(np.float32)
        flow = flow_vph.astype(np.float32)
        info = {
            "interval_arrived": interval_arrived,
            "interval_teleports": interval_teleports,
            "interval_insert_attempts": interval_insert_attempts,
            "interval_insert_success": interval_insert_success,
            "interval_insert_rejected": interval_insert_rejected,
            "interval_queue_mean": float(np.mean(interval_queue)) if interval_queue else 0.0,
            "interval_queue_max": int(max(interval_queue)) if interval_queue else 0,
            "episode_queue_mean": float(np.mean(self._queue_samples)) if self._queue_samples else 0.0,
            "episode_queue_max": int(max(self._queue_samples)) if self._queue_samples else 0,
        }
        return density, speed, flow, info

    def _make_observation(self) -> np.ndarray:
        density_norm = (self.current_density - self.density_mean) / self.density_std
        demand_norm = self._normalize_demand(self.current_demand_vph)
        time_norm = float(min(self.k, self.T_ctrl) / max(self.T_ctrl, 1))
        queue_norm = float(self._analytical_queue / self.queue_norm_scale)
        return np.concatenate(
            [
                density_norm.astype(np.float32),
                np.array([demand_norm, time_norm, queue_norm], dtype=np.float32),
            ]
        )

    def _normalize_demand(self, demand_vph: float) -> float:
        span = self.max_demand - self.min_demand
        if span <= 1e-6:
            return 0.0
        return float((demand_vph - self.min_demand) / span)

    def _throughput_vph(self) -> float:
        elapsed_s = max(float(self.k * self.dt_ctrl), self.dt_ctrl)
        return float(self._arrived_vehicles / elapsed_s * 3600.0)

    def _route_for_demand(self, demand_vph: float) -> str:
        key = float(demand_vph)
        if key in self.routes_by_demand:
            return self.routes_by_demand[key]

        cfg_for_demand = merge_configs(
            self.sumo_config,
            {"demand": {"mainline_demand_vph": key}},
        )
        route_path = self.network_dir / f"routes_{int(round(key))}.rou.xml"
        _write_routes(route_path, cfg_for_demand)
        self.routes_by_demand[key] = str(route_path.resolve())
        return self.routes_by_demand[key]

    def _load_sumo_config(self, env_config: dict) -> dict:
        if "sumo" in env_config:
            sumo_config = copy.deepcopy(env_config["sumo"])
        elif "sumo_config" in env_config:
            sumo_config = load_config(str(self._resolve_path(env_config["sumo_config"])))
        elif all(k in env_config for k in ("network", "simulation", "demand", "detectors")):
            sumo_config = copy.deepcopy(env_config)
        else:
            raise KeyError(
                "SUMO env config must provide 'sumo_config', a nested 'sumo' dict, "
                "or direct SUMO config keys."
            )

        overrides = env_config.get("sumo_overrides", {})
        if overrides:
            sumo_config = merge_configs(sumo_config, overrides)
        return sumo_config

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.project_root / p
