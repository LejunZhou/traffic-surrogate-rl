import sys, json, time, numpy as np
from pathlib import Path
ROOT = Path(r"C:/Users/Jun18/Desktop/traffic-surrogate-rl"); sys.path.insert(0, str(ROOT / "src"))
from utils.config import load_config
from sumo_env.demand_profiles import DemandProfile
from rl.sumo_env_wrapper import SumoEnv
S = Path(r"C:/Users/Jun18/AppData/Local/Temp/v4test")
cfg = dict(load_config(str(ROOT / "configs/experiments/round0_v3b.yaml"))["env"])
cfg["project_root"] = str(ROOT); cfg["sumo_config"] = str(S / "scenario_v4_test.yaml"); cfg["network_dir"] = str(S / "network_v4")
cfg.pop("density_stats_from", None)
env = SumoEnv(cfg)
print("N_x", env.N_x, "lanes per station:", [len(l) for l in env.det_ids_per_lane][:14], "merge rule:", env.merge_station_lanes)
out = {}
for d, r, u in [(4800, 700, 1.0), (5400, 700, 1.0), (6000, 700, 1.0), (6600, 700, 1.0), (6000, 700, 0.5)]:
    t0 = time.time(); env.reset(options={"profile": DemandProfile.constant(float(d), float(r))})
    per_lane, exit_flow, tele, ins_fail, queue = [], [], 0, 0, []
    done = False
    while not done:
        obs, rew, term, trunc, info = env.step(np.array([u], dtype=np.float32))
        per_lane.append([x.copy() for x in env.last_per_lane_density]); done = term or trunc
        tele += int(info.get("interval_teleports", 0)); ins_fail += int(info.get("interval_insert_rejected", 0))
        exit_flow.append(float(info.get("interval_arrived", np.nan))); queue.append(float(info.get("queue", info.get("ramp_queue", np.nan))))
    K = len(per_lane); Nx = env.N_x
    lane0 = np.array([[st[0] for st in step] for step in per_lane])                  # (K, Nx) lane 0 (rightmost through lane; at the merge station = accel lane)
    allmean = np.array([[st.mean() for st in step] for step in per_lane])
    through = np.array([[st[1:].mean() if len(st) > 3 else st.mean() for st in step] for step in per_lane])  # merge station: drop the accel lane
    lanes_through0 = np.array([[st[1] if len(st) > 3 else st[0] for st in step] for step in per_lane])       # lane 0 of the through lanes at every station
    out[f"d{d}_r{r}_u{u}"] = {"lane0_through": lanes_through0.tolist(), "all_mean": allmean.tolist(), "through_mean": through.tolist(),
                              "exit_flow_veh_per_step": exit_flow, "teleports": tele, "insert_rejected": ins_fail,
                              "n_lanes_station": [len(st) for st in per_lane[0]], "wall_s": time.time() - t0}
    print(f"d={d} r={r} u={u}: {K} steps, teleports {tele}, insert rejected {ins_fail}, exit flow mean {3600*np.nanmean(exit_flow)/30.0:.0f} veh/h, "
          f"max through-mean density {through.max():.0f}, max lane-0 density {lanes_through0.max():.0f}, {time.time()-t0:.0f}s", flush=True)
env.close()
json.dump(out, open(S / "merge_obs_test.json", "w"))
print("saved")
