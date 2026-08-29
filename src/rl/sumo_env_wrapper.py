"""
Gymnasium environment backed by live SUMO simulation via TraCI.

This is the direct SUMO+RL path: the policy chooses one ramp-metering action
per control interval, SUMO advances for that interval, and detector aggregates
become the next observation. It does not use the surrogate model.

Ramp demand is tracked as a virtual unmet-demand queue. The metering action
controls release attempts from that queue into SUMO; physical ramp-edge
occupancy is reported separately.

Observation (shape (N_x + 3,); 22 features for N_x=19; +1 ramp-arrival feature
when env.observe_ramp_demand is true, inserted after `demand`):
    density[0:N_x]  — z-score normalized density at detector locations
    demand[N_x]     — min-max normalized current mainline demand in [0, 1]
    time[N_x+1]     — normalized control index k / T_ctrl in [0, 1]
    queue[N_x+2]    — virtual ramp queue length normalized by queue_scale

Action (shape (1,)):
    ramp metering rate in [0, 1] (continuous Box)
"""

from __future__ import annotations

import copy
import warnings
import os
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

from rl.reward import RewardWeights, reward_terms as compute_reward_terms
from sumo_env.detectors import build_detector_file, get_detector_ids_per_lane, get_x_grid
from sumo_env.network_builder import build_network, _write_routes
from utils.config import load_config, merge_configs


class SumoEnv(gym.Env):
    """Gymnasium environment wrapping SUMO via TraCI."""

    metadata: dict = {"render_modes": []}

    # Each SumoEnv owns a labelled TraCI connection so that several envs can
    # be alive in one process (e.g. the PPO training env plus an EvalCallback
    # env). Every TraCI-touching method switches to its own label first.
    _instance_counter: int = 0

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

        self.duration_s = float(sim_cfg["duration_s"])
        self.T_ctrl = int(self.duration_s / self.dt_ctrl)
        self.warmup_s = float(sim_cfg.get("ramp_warmup_s", 0.0))
        self.base_seed = int(sim_cfg.get("seed", self.env_config.get("seed", 42)))
        self.sumo_binary = str(self.env_config.get("sumo_binary", sim_cfg["sumo_binary"]))
        # SUMO --max-depart-delay (seconds). Vehicles that cannot be inserted
        # within this delay are *discarded* instead of staying pending and
        # being replayed later at SUMO's slower re-insertion rate (~1550 vph
        # with departSpeed="max"), which quietly turned every post-jam
        # episode into a lower-demand scenario (progress M7 §7.6). -1 keeps
        # SUMO's default (wait forever). Discards are counted in info.
        self.max_depart_delay_s = float(sim_cfg.get("max_depart_delay_s", -1.0))
        # Extra SUMO command-line options (list of strings), e.g.
        # ["--extrapolate-departpos"] so vehicles whose depart time falls
        # between 1 s steps are placed as if inserted on time.
        self.sumo_extra_args = [str(a) for a in (sim_cfg.get("sumo_extra_args") or [])]
        if 0.0 <= self.max_depart_delay_s < self.step_len:
            warnings.warn(
                "simulation.max_depart_delay_s < step_length_s: vehicles discarded "
                "within their first step are never observed as pending, so "
                "discarded_mainline will under-count.",
                stacklevel=2,
            )
        self.ramp_demand_vph = float(demand_cfg["ramp_demand_vph"])
        # Meter saturation flow: the action u is the green fraction of this
        # discharge capacity, release = u * ramp_discharge_vph (capped by the
        # queue). Must exceed the arrival rate for a queue to be drainable;
        # default (= arrival rate) reproduces the M5–M7 behaviour where the
        # queue could never shrink. Same lookup order as SurrogateEnv.
        _queue_cfg = self.env_config.get("queue", {}) or {}
        self.ramp_discharge_vph = float(
            self.env_config.get(
                "ramp_discharge_vph",
                _queue_cfg.get(
                    "discharge_vph",
                    demand_cfg.get("ramp_discharge_vph", self.ramp_demand_vph),
                ),
            )
        )
        if not np.isfinite(self.ramp_discharge_vph) or self.ramp_discharge_vph <= 0.0:
            raise ValueError(f"ramp_discharge_vph must be positive, got {self.ramp_discharge_vph}")
        # Per-episode ramp arrival rate, sampled like demand_levels at reset()
        # (or forced via reset(options={"ramp_demand_vph": ...})).
        self.ramp_demand_levels = [
            float(v) for v in self.env_config.get("ramp_demand_levels", [self.ramp_demand_vph])
        ]
        if not self.ramp_demand_levels or any(v < 0.0 for v in self.ramp_demand_levels):
            raise ValueError("env.ramp_demand_levels must be a non-empty list of rates >= 0")
        self.current_ramp_demand_vph = float(self.ramp_demand_levels[0])
        self.min_ramp_demand = float(min(self.ramp_demand_levels))
        self.max_ramp_demand = float(max(self.ramp_demand_levels))
        # Append the (min-max normalised) ramp arrival rate to the observation.
        # Off by default so pre-§7.10 policies keep their 22-dim input; on in
        # the training configs, where ramp_demand_levels varies per episode.
        self.observe_ramp_demand = bool(self.env_config.get("observe_ramp_demand", False))
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
        reward_cfg = self.env_config.get("reward", {}) or {}
        self.reward_weights = RewardWeights.from_config(reward_cfg)
        queue_cfg = self.env_config.get("queue", {}) or {}
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
                                    self.ramp_demand_vph
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
            shape=(self.N_x + 3 + int(self.observe_ramp_demand),),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(self.base_seed)
        self.episode_index = 0
        self.k = 0
        self.current_demand_vph = float(self.demand_levels[0])
        self.current_density = np.zeros(self.N_x, dtype=np.float32)
        self._started = False
        self._veh_counter = 0
        self._ramp_arrival_accumulator = 0.0
        self._ramp_release_accumulator = 0.0
        self._virtual_queue_length = 0.0
        self._insert_attempts = 0
        self._insert_success = 0
        self._insert_rejected = 0
        self._teleports = 0
        self._arrived_vehicles = 0
        self._queue_samples: list[float] = []
        self._physical_ramp_samples: list[int] = []
        self._reset_insertion_bookkeeping()
        SumoEnv._instance_counter += 1
        self._traci_label = f"sumo_env_{SumoEnv._instance_counter}_{os.getpid()}"

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new SUMO simulation with a sampled demand profile.

        Returns:
            observation: shape (N_x + 3,)
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
        if "ramp_demand_vph" in options:
            ramp_demand_vph = float(options["ramp_demand_vph"])
        else:
            ramp_demand_vph = float(self.rng.choice(self.ramp_demand_levels))
        self.current_ramp_demand_vph = ramp_demand_vph

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
        if self.max_depart_delay_s >= 0.0:
            sumo_cmd += ["--max-depart-delay", str(self.max_depart_delay_s)]
        sumo_cmd += self.sumo_extra_args

        traci.start(sumo_cmd, label=self._traci_label)
        self._started = True
        self.episode_index += 1
        self.k = 0
        self.current_density = np.zeros(self.N_x, dtype=np.float32)
        self._veh_counter = 0
        self._ramp_arrival_accumulator = 0.0
        self._ramp_release_accumulator = 0.0
        self._virtual_queue_length = 0.0
        self._insert_attempts = 0
        self._insert_success = 0
        self._insert_rejected = 0
        self._teleports = 0
        self._arrived_vehicles = 0
        self._queue_samples = []
        self._physical_ramp_samples = []
        self._reset_insertion_bookkeeping()

        obs = self._make_observation()
        info = {
            "demand_vph": self.current_demand_vph,
            "ramp_demand_vph": self.current_ramp_demand_vph,
            "ramp_discharge_vph": self.ramp_discharge_vph,
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
            observation: shape (N_x + 3,)
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
        queue_length = float(interval_info.get("interval_queue_mean", 0.0))
        # Mainline outflow = vehicles that left the network during this
        # control interval (traci.simulation.getArrivedNumber, summed over
        # the 30 sub-steps). This is an exact count. The E1 loop flow at
        # det_18 (flow[-1]) over-counts by ~(1 + L / (v * dt)) because a
        # vehicle straddling a 1 s step boundary is counted twice (~1.2x at
        # 100 km/h, worse when slow); it is kept in info for diagnostics.
        outflow_vph = float(interval_info.get("interval_arrived", 0)) * 3600.0 / self.dt_ctrl
        det18_flow_vph = float(flow[-1])
        reward_terms = self._reward_terms(density, queue_length, outflow_vph)
        raw_reward = float(reward_terms["reward"])
        reward_warmup_active = self._reward_warmup_active()
        reward = 0.0 if reward_warmup_active else raw_reward

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
            "mean_density": reward_terms["mean_density"],
            "std_density": reward_terms["std_density"],
            "outflow_vph": reward_terms["outflow_vph"],
            "det18_flow_vph": det18_flow_vph,
            "lost_outflow_frac": reward_terms["lost_outflow_frac"],
            "outflow_penalty": reward_terms["outflow_penalty"],
            "mean_speed": float(np.mean(speed)),
            "mean_flow": float(np.mean(flow)),
            "queue_length": queue_length,
            "analytical_queue": queue_length,
            "queue_penalty": reward_terms["queue_penalty"],
            "queue_scale": float(self.queue_scale),
            "queue_norm": self._normalize_queue(queue_length),
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
            "demand_vph": self.current_demand_vph,
            "ramp_demand_vph": self.current_ramp_demand_vph,
            "ramp_discharge_vph": self.ramp_discharge_vph,
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
            traci.switch(self._traci_label)
            traci.close(False)
        except Exception:
            pass
        finally:
            self._started = False

    def _reset_insertion_bookkeeping(self) -> None:
        # Ramp vehicles handed to traci.vehicle.add() that SUMO has not yet
        # placed on the road. add() does not fail when the ramp edge is full;
        # the vehicle becomes *pending*, so the virtual queue is decremented
        # only when SUMO reports the vehicle as departed.
        self._ramp_pending_ids: set[str] = set()
        self._ramp_departed = 0
        # IDs SUMO reported pending after the previous simulation step; a
        # vehicle that leaves this set without departing was discarded by
        # --max-depart-delay.
        self._prev_pending: set[str] = set()
        self._discarded_mainline = 0
        self._discarded_ramp = 0
        self._pending_mainline_max = 0
        self._pending_ramp_max = 0

    def _advance_control_interval(
        self, ramp_rate: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        traci.switch(self._traci_label)
        sum_count = np.zeros(self.N_x, dtype=np.float64)
        sum_speed = np.zeros(self.N_x, dtype=np.float64)
        speed_count = np.zeros(self.N_x, dtype=np.int32)
        sum_occ = np.zeros(self.N_x, dtype=np.float64)
        interval_queue: list[float] = []
        interval_physical_ramp: list[int] = []
        interval_teleports = 0
        interval_insert_attempts = 0
        interval_insert_success = 0
        interval_insert_rejected = 0
        interval_arrived = 0
        interval_ramp_arrivals = 0
        interval_release_capacity = 0
        interval_ramp_departed = 0
        interval_pending_mainline_max = 0
        interval_pending_ramp_max = 0
        n_pending_main = 0
        n_pending_ramp = 0

        for _ in range(self.dt_ctrl_steps):
            if traci.simulation.getTime() >= self.warmup_s:
                arrival_rate = self.current_ramp_demand_vph / 3600.0
                self._ramp_arrival_accumulator += arrival_rate * self.step_len
                n_arrivals = int(self._ramp_arrival_accumulator)
                self._ramp_arrival_accumulator -= n_arrivals
                self._virtual_queue_length += n_arrivals
                interval_ramp_arrivals += n_arrivals

                release_rate = ramp_rate * self.ramp_discharge_vph / 3600.0
                self._ramp_release_accumulator += release_rate * self.step_len
                n_release_capacity = int(self._ramp_release_accumulator)
                self._ramp_release_accumulator -= n_release_capacity
                interval_release_capacity += n_release_capacity

                # Vehicles already handed to SUMO but still pending insertion
                # are still counted in the virtual queue; do not release them
                # a second time.
                releasable = int(self._virtual_queue_length) - len(self._ramp_pending_ids)
                n_release = min(n_release_capacity, max(releasable, 0))
                for _ in range(n_release):
                    self._insert_attempts += 1
                    interval_insert_attempts += 1
                    veh_id = f"ramp_{self._veh_counter}"
                    try:
                        traci.vehicle.add(
                            vehID=veh_id,
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
                        self._ramp_pending_ids.add(veh_id)
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

            # Insertion bookkeeping (progress M7 §7.6). Ramp vehicles leave
            # the virtual queue when SUMO actually inserts them; anything that
            # was pending and is now neither pending nor departed was
            # discarded by --max-depart-delay. Discarded ramp vehicles never
            # left the virtual queue, so the queue stays conserved.
            departed_ids = set(traci.simulation.getDepartedIDList())
            pending_ids = set(traci.simulation.getPendingVehicles())
            ramp_gone = self._ramp_pending_ids - pending_ids
            ramp_departed = ramp_gone & departed_ids
            n_ramp_departed = len(ramp_departed)
            self._ramp_pending_ids -= ramp_gone
            self._ramp_departed += n_ramp_departed
            interval_ramp_departed += n_ramp_departed
            self._discarded_ramp += len(ramp_gone) - n_ramp_departed
            self._virtual_queue_length = max(
                self._virtual_queue_length - n_ramp_departed, 0.0
            )
            self._discarded_mainline += sum(
                1
                for v in self._prev_pending - pending_ids - departed_ids
                if not v.startswith("ramp_")
            )
            self._prev_pending = pending_ids
            n_pending_ramp = len(self._ramp_pending_ids)
            n_pending_main = len(pending_ids) - sum(
                1 for v in pending_ids if v.startswith("ramp_")
            )
            self._pending_mainline_max = max(self._pending_mainline_max, n_pending_main)
            self._pending_ramp_max = max(self._pending_ramp_max, n_pending_ramp)
            interval_pending_mainline_max = max(interval_pending_mainline_max, n_pending_main)
            interval_pending_ramp_max = max(interval_pending_ramp_max, n_pending_ramp)

            # Queue sample taken after the step so that a vehicle released and
            # inserted in the same step is counted out of the queue, exactly as
            # before the departure-based accounting.
            self._queue_samples.append(float(self._virtual_queue_length))
            interval_queue.append(float(self._virtual_queue_length))

            physical_ramp_occupancy = int(traci.edge.getLastStepVehicleNumber("ramp"))
            self._physical_ramp_samples.append(physical_ramp_occupancy)
            interval_physical_ramp.append(physical_ramp_occupancy)

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
            "ramp_arrivals": interval_ramp_arrivals,
            "ramp_released": interval_ramp_departed,
            "ramp_add_calls": interval_insert_success,
            "ramp_release_capacity": interval_release_capacity,
            "ramp_departed_total": self._ramp_departed,
            "pending_mainline": n_pending_main,
            "pending_ramp": n_pending_ramp,
            "interval_pending_mainline_max": interval_pending_mainline_max,
            "interval_pending_ramp_max": interval_pending_ramp_max,
            "episode_pending_mainline_max": self._pending_mainline_max,
            "episode_pending_ramp_max": self._pending_ramp_max,
            "discarded_mainline": self._discarded_mainline,
            "discarded_ramp": self._discarded_ramp,
            "max_depart_delay_s": self.max_depart_delay_s,
            "queue_after": float(self._virtual_queue_length),
            "interval_queue_mean": float(np.mean(interval_queue)) if interval_queue else 0.0,
            "interval_queue_max": float(max(interval_queue)) if interval_queue else 0.0,
            "episode_queue_mean": float(np.mean(self._queue_samples)) if self._queue_samples else 0.0,
            "episode_queue_max": float(max(self._queue_samples)) if self._queue_samples else 0.0,
            "interval_physical_ramp_mean": float(np.mean(interval_physical_ramp)) if interval_physical_ramp else 0.0,
            "interval_physical_ramp_max": int(max(interval_physical_ramp)) if interval_physical_ramp else 0,
            "episode_physical_ramp_mean": float(np.mean(self._physical_ramp_samples)) if self._physical_ramp_samples else 0.0,
            "episode_physical_ramp_max": int(max(self._physical_ramp_samples)) if self._physical_ramp_samples else 0,
        }
        return density, speed, flow, info

    def _make_observation(self) -> np.ndarray:
        density_norm = (self.current_density - self.density_mean) / self.density_std
        demand_norm = self._normalize_demand(self.current_demand_vph)
        time_norm = float(min(self.k, self.T_ctrl) / max(self.T_ctrl, 1))
        queue_norm = self._normalize_queue(self._virtual_queue_length)
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
        if span <= 0.0:
            return 0.0
        return float((ramp_demand_vph - self.min_ramp_demand) / span)

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

    def _throughput_vph(self) -> float:
        elapsed_s = max(float(self.k * self.dt_ctrl), self.dt_ctrl)
        return float(self._arrived_vehicles / elapsed_s * 3600.0)

    def _reward_warmup_active(self) -> bool:
        return float(self.k * self.dt_ctrl) < self.reward_warmup_s

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
