"""Smoke test: 2 parallel SumoEnv workers, reset + 3 control steps.

Throwaway — confirms SubprocVecEnv + worker_id isolation works on this box
before kicking off a 200k production run. Delete after use.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from utils.config import load_config

    cfg = load_config(str(project_root / "configs/rl/ppo_sumo.yaml"))
    cfg["project_root"] = str(project_root)
    env_cfg = cfg["env"]

    def make_factory(worker_id: int):
        def _init():
            from rl.sumo_env_wrapper import SumoEnv

            wc = copy.deepcopy(env_cfg)
            wc["worker_id"] = worker_id
            wc["project_root"] = cfg["project_root"]
            return Monitor(SumoEnv(wc))

        return _init

    print("[smoke] Building SubprocVecEnv(2 workers)...")
    t0 = time.perf_counter()
    vec = SubprocVecEnv([make_factory(0), make_factory(1)], start_method="spawn")
    print(f"[smoke] Built in {time.perf_counter() - t0:.1f}s. Resetting...")

    obs = vec.reset()
    print(f"[smoke] reset OK, obs.shape={obs.shape}")
    print("[smoke] Stepping 3 times...")
    for k in range(3):
        a = np.zeros((2, 1), dtype=np.float32)
        obs, rew, done, info = vec.step(a)
        print(f"  step {k}: rew={rew}, done={done}")
    vec.close()
    print("[smoke] DONE")


if __name__ == "__main__":
    main()
