"""
SurrogateVecEnv v2 (M10, draft_pipeline.md §7.1): a batched Stable-Baselines3
VecEnv whose plant is the DeepONet ensemble.

Every env slot owns a demand profile (d_k, r_k), the analytic ramp queue and
the branch history [d / 2500, q_r / 1600] (2, K). One ensemble forward per
step serves all slots. The surrogate never sees the action: u_k is converted
to released inflow through the same queue recursion as SumoEnv
(sumo_env.ramp_queue.analytic_queue_step), written into the history at index
k, and the plant returns rho_hat_k at the detectors and q_hat_out,k at the
exit. Observation, reward and info follow SumoEnv (rl.sumo_env_wrapper) so a
policy can move between the two environments unchanged.

Ensemble modes:
    sample       one member per episode, drawn at reset (default; MBPO-style)
    mean         average of the member fields
    pessimistic  reward = mean over members - kappa * std over members (MOPO-style)

Profiles: `profiles` may be a family YAML (resample every episode) or a
frozen set JSON (slot i cycles through the set, so n_envs = len(set) rolls
the whole set once per pass — the evaluation configuration).

Plant abstraction: any object with `predict_step(history, k, member)` and
`predict_all_members_step(history, k)` (DeepONetEnsemble, or the one-step
autoregressive baseline in surrogate.onestep) can be plugged in.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from rl.reward import RewardWeights, backlog_estimate, offered_q_ref, reward_terms as compute_reward_terms
from rl.sumo_env_wrapper import build_observation
from sumo_env.demand_profiles import DemandProfile, resolve_profile_source
from sumo_env.ramp_queue import analytic_queue_step, queue_override_rate

ENSEMBLE_MODES = ("sample", "mean", "pessimistic")


class SurrogateVecEnv(VecEnv):
    def __init__(self, config: dict, plant=None) -> None:
        self.env_config = copy.deepcopy(config or {})
        cfg = self.env_config
        self.project_root = Path(cfg.get("project_root", Path.cwd())).resolve()
        self.n_envs = int(cfg.get("n_envs", 16))
        self.mode = str(cfg.get("ensemble_mode", "sample"))
        if self.mode not in ENSEMBLE_MODES:
            raise ValueError(f"ensemble_mode must be one of {ENSEMBLE_MODES}")
        self.kappa = float(cfg.get("pessimistic_kappa", 1.0))
        self.log_ensemble_std = bool(cfg.get("log_ensemble_std", False))

        # ---- plant --------------------------------------------------------
        if plant is None:
            plant_type = str(cfg.get("plant_type", "deeponet"))
            ens_dir = cfg.get("ensemble_dir")
            if ens_dir is None:
                raise ValueError("SurrogateVecEnv needs env.ensemble_dir (or a plant instance)")
            ens_path = Path(ens_dir) if Path(ens_dir).is_absolute() else self.project_root / ens_dir
            if plant_type == "deeponet":
                from surrogate.deeponet import DeepONetEnsemble

                plant = DeepONetEnsemble.load(ens_path, device=str(cfg.get("device", "cpu")),
                                              members=cfg.get("members"))
            elif plant_type == "onestep":
                from surrogate.onestep import OneStepEnsemble

                plant = OneStepEnsemble.load(ens_path, members=cfg.get("members"))
            else:
                raise ValueError(f"unknown plant_type {plant_type!r}")
        self.plant = plant
        self.M = int(plant.M)
        self.K = int(plant.K)
        self.T_ctrl = self.K
        self.N_x = int(plant.Nx)
        self.dt_ctrl = float(plant.dt)
        self.duration_s = float(plant.T)
        self.x_grid = np.asarray(plant.x_grid, dtype=np.float32)
        self.dx_km = float(self.x_grid[1] - self.x_grid[0]) / 1000.0 if self.N_x > 1 else 0.1
        self.t_grid = np.arange(self.K, dtype=np.float32) * self.dt_ctrl
        self.norm = plant.norm

        # ---- scenario constants (parity with SumoEnv) -----------------------
        # Meter discharge D and the storage cap follow SumoEnv's rule: the env key
        # wins, else the scenario file's demand block (M14: v3b has D = 1200),
        # else the v2 defaults (1600, unlimited).
        _demand = {}
        if cfg.get("sumo_config"):
            try:
                from utils.config import load_config as _load_cfg
                _sc = cfg["sumo_config"]; _scp = Path(_sc) if Path(_sc).is_absolute() else self.project_root / _sc
                _demand = dict(_load_cfg(str(_scp)).get("demand", {}))
            except Exception:
                _demand = {}
        self.ramp_discharge_vph = float(cfg.get("ramp_discharge_vph", _demand.get("ramp_discharge_vph", 1600.0)))
        _qmax = cfg.get("ramp_queue_max_veh", _demand.get("ramp_queue_max_veh"))
        self.ramp_queue_max_veh = float(_qmax) if _qmax else None
        self.warmup_s = float(cfg.get("ramp_warmup_s", 0.0))
        self.density_clip = (0.0, float(cfg.get("density_clip_max", 143.0)))
        self.outflow_clip = (0.0, float(cfg.get("outflow_clip_max", 3000.0)))
        self.density_mean = float(cfg.get("density_mean", self.norm.density_mean))
        self.density_std = max(float(cfg.get("density_std", self.norm.density_std)), 1e-6)
        obs_cfg = dict(cfg.get("observation", {}) or {})
        self.lookahead_steps = int(obs_cfg.get("lookahead_steps", 0))
        self.demand_norm_vph = float(obs_cfg.get("demand_norm", 2500.0))
        self.ramp_norm_vph = float(obs_cfg.get("ramp_norm", 1000.0))
        clip = obs_cfg.get("clip", [-3.0, 25.0])
        self.obs_clip = None if clip is None else (float(clip[0]), float(clip[1]))
        self.observe_ramp_demand = bool(cfg.get("observe_ramp_demand", True))
        reward_cfg = cfg.get("reward", {}) or {}
        self.reward_weights = RewardWeights.from_config(reward_cfg)
        self.queue_scale = float(cfg.get("queue_norm_scale", cfg.get("queue_scale", obs_cfg.get("queue_norm", 100.0))))
        self.reward_warmup_s = float(cfg.get("reward_warmup_s", reward_cfg.get("warmup_s", 0.0)))
        self.symmetric_action = bool(cfg.get("symmetric_action", False))

        # ---- profiles -----------------------------------------------------
        self.profile_family, self.profile_set = resolve_profile_source(cfg.get("profiles"), self.project_root, dt_ctrl_s=self.dt_ctrl)
        if self.profile_family is None and self.profile_set is None:
            d0, r0 = float(cfg.get("demand_vph", 2000.0)), float(cfg.get("ramp_demand_vph", 800.0))
            self.profile_set = [DemandProfile.constant(d0, r0, n_blocks=self.K // 10, dt_ctrl_s=self.dt_ctrl)]
        self._set_cursor = 0

        obs_dim = self.N_x + 3 + int(self.observe_ramp_demand) + 2 * self.lookahead_steps
        observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        lo, hi = (-1.0, 1.0) if self.symmetric_action else (0.0, 1.0)
        action_space = spaces.Box(np.array([lo], np.float32), np.array([hi], np.float32), dtype=np.float32)
        super().__init__(self.n_envs, observation_space, action_space)

        self.base_seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(self.base_seed)
        self.member_rng = np.random.default_rng(self.base_seed + 12345)
        self.episode_count = 0
        # per-slot state
        self.profiles: list[DemandProfile | None] = [None] * self.n_envs
        self.d = np.zeros((self.n_envs, self.K), np.float32)
        self.r = np.zeros((self.n_envs, self.K), np.float32)
        self.history = np.zeros((self.n_envs, 2, self.K), np.float32)
        self.k = np.zeros(self.n_envs, np.int64)
        self.queue = np.zeros(self.n_envs, np.float64)
        self.member = np.zeros(self.n_envs, np.int64)
        self.density = np.zeros((self.n_envs, self.N_x), np.float32)
        self.cum_offered = np.zeros(self.n_envs, np.float64)
        self.cum_served = np.zeros(self.n_envs, np.float64)
        self.ep_return = np.zeros(self.n_envs, np.float64)
        self.queue_samples: list[list[float]] = [[] for _ in range(self.n_envs)]
        self._actions = None
        self.render_mode = None

    # ------------------------------------------------------------- VecEnv
    def seed(self, seed=None):
        if seed is not None:
            self.base_seed = int(seed)
            self.rng = np.random.default_rng(self.base_seed)
            self.member_rng = np.random.default_rng(self.base_seed + 12345)
        return [self.base_seed + i for i in range(self.n_envs)]

    def reset(self) -> np.ndarray:
        for i in range(self.n_envs):
            self._reset_slot(i)
        return self._obs_all()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32).reshape(self.n_envs, -1)[:, 0]

    def step_wait(self):
        a = self._actions
        u = (a + 1.0) / 2.0 if self.symmetric_action else a
        u = np.clip(u, 0.0, 1.0)
        n = self.n_envs
        k = self.k.copy()
        idx = np.arange(n)
        d_k = self.d[idx, k]
        r_k = self.r[idx, k]
        queue_before = self.queue.copy()
        released = np.zeros(n); arrivals = np.zeros(n)
        u_requested = u.copy()
        for i in range(n):
            arr_rate = r_k[i] if self.t_grid[k[i]] >= self.warmup_s else 0.0
            # finite ramp storage (M14): identical override to SumoEnv
            u[i] = max(u[i], queue_override_rate(queue_before[i], arr_rate, self.ramp_queue_max_veh, self.ramp_discharge_vph, self.dt_ctrl))
            self.queue[i], released[i], arrivals[i] = analytic_queue_step(queue_before[i], u[i], arr_rate,
                                                                          self.ramp_discharge_vph, self.dt_ctrl)
        q_r_vph = released * 3600.0 / self.dt_ctrl
        self.history[idx, 1, k] = q_r_vph / self.norm.inflow_scale
        # ---- plant ---------------------------------------------------------
        rho_members = q_members = None
        if self.mode == "sample" and not self.log_ensemble_std:
            rho, q_out = self.plant.predict_step(self.history, k, self.member)
        else:
            rho_members, q_members = self.plant.predict_all_members_step(self.history, k)   # (M, n, Nx), (M, n)
            if self.mode == "sample":
                rho = rho_members[self.member, idx]; q_out = q_members[self.member, idx]
            else:
                rho = rho_members.mean(0); q_out = q_members.mean(0)
        rho = np.clip(rho, *self.density_clip).astype(np.float32)
        q_out = np.clip(q_out, *self.outflow_clip).astype(np.float32)
        self.density = rho
        # ---- reward --------------------------------------------------------
        obs_list, rewards, dones, infos = [], np.zeros(n, np.float32), np.zeros(n, bool), []
        dt_h = self.dt_ctrl / 3600.0
        for i in range(n):
            q_ref_k = offered_q_ref(self.reward_weights, float(d_k[i]), float(r_k[i]), float(queue_before[i]),
                                    self.ramp_discharge_vph, self.dt_ctrl)
            terminal = bool(k[i] + 1 >= self.K)
            self.cum_offered[i] += (float(d_k[i]) + float(r_k[i])) * dt_h
            self.cum_served[i] += float(q_out[i]) * dt_h
            on_road = float(np.sum(rho[i])) * self.dx_km
            backlog = backlog_estimate(self.cum_offered[i], self.cum_served[i], on_road, float(self.queue[i]))
            terms = compute_reward_terms(rho[i], float(self.queue[i]), float(q_out[i]), self.reward_weights,
                                         q_ref=q_ref_k, terminal=terminal, backlog_veh=backlog, dt_ctrl_s=self.dt_ctrl)
            raw = float(terms["reward"])
            if self.mode == "pessimistic" and rho_members is not None:
                per_member = []
                for m in range(self.M):
                    rho_m = np.clip(rho_members[m, i], *self.density_clip); q_m = float(np.clip(q_members[m, i], *self.outflow_clip))
                    bl_m = backlog_estimate(self.cum_offered[i], self.cum_served[i] - float(q_out[i]) * dt_h + q_m * dt_h,
                                            float(np.sum(rho_m)) * self.dx_km, float(self.queue[i]))
                    per_member.append(compute_reward_terms(rho_m, float(self.queue[i]), q_m, self.reward_weights, q_ref=q_ref_k,
                                                           terminal=terminal, backlog_veh=bl_m, dt_ctrl_s=self.dt_ctrl)["reward"])
                raw = float(np.mean(per_member) - self.kappa * np.std(per_member))
            warm = float(self.t_grid[k[i]]) < self.reward_warmup_s
            reward = 0.0 if warm else raw
            self.queue_samples[i].append(float(self.queue[i]))
            self.ep_return[i] += reward
            self.k[i] = k[i] + 1
            info = {
                "time_s": float(self.k[i] * self.dt_ctrl), "k": int(self.k[i]), "query_time_s": float(self.t_grid[k[i]]),
                "ramp_rate": float(u[i]), "u": float(u[i]), "u_requested": float(u_requested[i]),
                "u_override": float(u[i] > u_requested[i] + 1e-9),
                "queue_max_veh": self.ramp_queue_max_veh if self.ramp_queue_max_veh is not None else -1.0,
                "density": rho[i].copy(),
                "mean_density": terms["mean_density"], "std_density": terms["std_density"],
                "outflow_vph": terms["outflow_vph"], "q_ref": terms["q_ref"],
                "lost_outflow_frac": terms["lost_outflow_frac"], "outflow_penalty": terms["outflow_penalty"],
                "queue_length": float(self.queue[i]), "analytical_queue": float(self.queue[i]),
                "queue_before": float(queue_before[i]), "queue_after": float(self.queue[i]),
                "queue_penalty": terms["queue_penalty"], "queue_scale": self.queue_scale,
                "queue_norm": float(self.queue[i] / self.queue_scale), "std_penalty": terms["std_penalty"],
                "terminal_penalty": terms["terminal_penalty"], "raw_reward": raw,
                "on_road_veh": terms["on_road_veh"], "backlog_veh": terms["backlog_veh"],
                "tts_step_veh_h": terms["tts_step_veh_h"], "tts_penalty": terms["tts_penalty"],
                "cum_offered_veh": float(self.cum_offered[i]), "cum_served_veh": float(self.cum_served[i]),
                "reward_form": self.reward_weights.form,
                "reward_warmup_active": float(warm), "reward_warmup_s": self.reward_warmup_s,
                "reward_delta": self.reward_weights.delta, "reward_beta": self.reward_weights.beta,
                "reward_gamma": self.reward_weights.gamma, "reward_q_ref": self.reward_weights.q_ref,
                "reward_sigma_ref": self.reward_weights.sigma_ref, "reward_queue_norm": self.reward_weights.queue_norm,
                "demand_vph": float(d_k[i]), "ramp_demand_vph": float(r_k[i]),
                "mainline_demand_vph": float(d_k[i]), "ramp_arrival_vph": float(r_k[i]),
                "ramp_discharge_vph": self.ramp_discharge_vph, "ramp_arrivals": float(arrivals[i]),
                "ramp_released": float(released[i]), "ramp_inflow_vph": float(q_r_vph[i]),
                "ramp_release_capacity": float(u[i] * self.ramp_discharge_vph * self.dt_ctrl / 3600.0),
                "pending_mainline": 0.0,
                "episode_queue_mean": float(np.mean(self.queue_samples[i])), "episode_queue_max": float(np.max(self.queue_samples[i])),
                "ensemble_member": int(self.member[i]), "ensemble_mode": self.mode, "backend": "surrogate",
                "profile_set": self.profiles[i].set_name, "profile_index": int(self.profiles[i].index),
            }
            if rho_members is not None:
                info["ensemble_std_density"] = float(rho_members[:, i].std(0).mean())
                info["ensemble_std_outflow"] = float(q_members[:, i].std(0))
            rewards[i] = reward
            if terminal:
                dones[i] = True
                info["terminal_observation"] = self._obs(i)
                info["episode_return_env"] = float(self.ep_return[i])
                self._reset_slot(i)
            infos.append(info)
            obs_list.append(self._obs(i))
        return np.stack(obs_list), rewards, dones, infos

    def close(self) -> None:
        pass

    # SB3 VecEnv abstract helpers
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        return []

    # ----------------------------------------------------------- internals
    def set_profiles(self, profiles: list[DemandProfile]) -> None:
        """Fix the profile cycle (evaluation): slot i starts at profiles[i]."""
        self.profile_set = list(profiles)
        self.profile_family = None
        self._set_cursor = 0

    def _next_profile(self) -> DemandProfile:
        if self.profile_family is not None:
            self.episode_count += 1
            return self.profile_family.sample(self.rng, "train", self.episode_count)
        p = self.profile_set[self._set_cursor % len(self.profile_set)]
        self._set_cursor += 1
        return p

    def _reset_slot(self, i: int) -> None:
        p = self._next_profile()
        self.profiles[i] = p
        self.d[i] = p.mainline_vph[: self.K]
        self.r[i] = p.ramp_vph[: self.K]
        self.history[i, 0] = self.d[i] / self.norm.demand_scale
        self.history[i, 1] = 0.0
        self.k[i] = 0
        self.queue[i] = 0.0
        self.member[i] = int(self.member_rng.integers(0, self.M)) if self.mode == "sample" else 0
        self.density[i] = 0.0
        self.cum_offered[i] = 0.0
        self.cum_served[i] = 0.0
        self.ep_return[i] = 0.0
        self.queue_samples[i] = []

    def _demand_features(self, i: int) -> list[float]:
        k = min(int(self.k[i]), self.K - 1)
        feats = [float(self.d[i, k]) / self.demand_norm_vph]
        if self.observe_ramp_demand:
            feats.append(float(self.r[i, k]) / self.ramp_norm_vph)
        if self.lookahead_steps > 0:
            idx = np.clip(np.arange(k + 1, k + 1 + self.lookahead_steps), 0, self.K - 1)
            feats.extend((self.d[i, idx] / self.demand_norm_vph).tolist())
            feats.extend((self.r[i, idx] / self.ramp_norm_vph).tolist())
        return feats

    def _obs(self, i: int) -> np.ndarray:
        return build_observation(self.density[i], self.density_mean, self.density_std, self.obs_clip,
                                 self._demand_features(i), float(min(self.k[i], self.K) / self.K),
                                 float(self.queue[i] / self.queue_scale))

    def _obs_all(self) -> np.ndarray:
        return np.stack([self._obs(i) for i in range(self.n_envs)])

    # ------------------------------------------------------ convenience
    def rollout_policy(self, policy, profiles: list[DemandProfile], deterministic: bool = True, member: int | None = None) -> list[dict]:
        """Roll `policy(obs) -> action` (raw action in this env's action space)
        over the given profiles, one per slot pass; returns per-episode metrics."""
        from sumo_env.rollout import episode_metrics

        saved = (self.profile_family, self.profile_set, self._set_cursor, self.mode)
        self.set_profiles(profiles)
        results = []
        n_pass = int(np.ceil(len(profiles) / self.n_envs))
        try:
            for _ in range(n_pass):
                obs = self.reset()
                if member is not None:
                    self.member[:] = member
                recs = [{"density": [], "outflow_vph": [], "mainline_demand": [], "ramp_arrival": [], "ramp_queue": [],
                         "action": [], "reward": [], "pending_mainline": [], "outflow_penalty": [], "queue_penalty": [], "std_penalty": []}
                        for _ in range(self.n_envs)]
                prof = [self.profiles[i] for i in range(self.n_envs)]
                for _ in range(self.K):
                    act = policy(obs)
                    obs, rew, done, infos = self.step(np.asarray(act, dtype=np.float32).reshape(self.n_envs, -1))
                    for i, info in enumerate(infos):
                        rc = recs[i]
                        rc["density"].append(info["density"]); rc["outflow_vph"].append(info["outflow_vph"])
                        rc["mainline_demand"].append(info["mainline_demand_vph"]); rc["ramp_arrival"].append(info["ramp_arrival_vph"])
                        rc["ramp_queue"].append(info["queue_after"]); rc["action"].append(info["u"]); rc["reward"].append(float(rew[i]))
                        rc["pending_mainline"].append(0.0)
                        for key in ("outflow_penalty", "queue_penalty", "std_penalty"):
                            rc[key].append(0.0 if info["reward_warmup_active"] else info[key])
                for i in range(self.n_envs):
                    arrays = {k: (np.stack(v, axis=1) if k == "density" else np.asarray(v, dtype=np.float32)) for k, v in recs[i].items()}
                    m = episode_metrics(arrays, self.dt_ctrl, float(self.x_grid[1] - self.x_grid[0]) / 1000.0)
                    m["profile_index"] = int(prof[i].index); m["profile_set"] = prof[i].set_name
                    results.append(m)
        finally:
            self.profile_family, self.profile_set, self._set_cursor, self.mode = saved
        return results[: len(profiles)]
