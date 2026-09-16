import sys, json, time, numpy as np
from pathlib import Path
ROOT = Path(r"C:/Users/Jun18/Desktop/traffic-surrogate-rl"); sys.path.insert(0, str(ROOT / "src"))
from utils.config import load_config
from sumo_env.demand_profiles import DemandProfile
from rl.sumo_env_wrapper import SumoEnv
S = Path(r"C:/Users/Jun18/AppData/Local/Temp/v4test"); out = {}
for tag, yml in [("imp30", "scenario_v4_imp30.yaml"), ("imp30_sg3", "scenario_v4_imp30_sg3.yaml"), ("assert2_sg3", "scenario_v4_assert2_sg3.yaml")]:
    cfg = dict(load_config(str(ROOT / "configs/experiments/round0_v3b.yaml"))["env"]); cfg["project_root"] = str(ROOT)
    cfg["sumo_config"] = str(S / yml); cfg["network_dir"] = str(S / ("network_" + tag)); cfg.pop("density_stats_from", None)
    env = SumoEnv(cfg)
    rou = open(Path(cfg["network_dir"]) / "routes.rou.xml").read(); print(tag, "vType attrs:", [l.strip() for l in rou.splitlines() if l.strip().startswith("lc")])
    for d, r, u in [(5400, 700, 1.0), (6000, 700, 1.0)]:
        t0 = time.time(); env.reset(options={"profile": DemandProfile.constant(float(d), float(r))}); per_lane, ef, tele = [], [], 0; done = False
        while not done:
            obs, rew, term, trunc, info = env.step(np.array([u], dtype=np.float32)); done = term or trunc
            per_lane.append([x.copy() for x in env.last_per_lane_density]); ef.append(float(info.get("interval_arrived", np.nan))); tele += int(info.get("interval_teleports", 0))
        L0 = np.array([[st[1] if len(st) > 3 else st[0] for st in step] for step in per_lane]); TM = np.array([[st[1:].mean() if len(st) > 3 else st.mean() for st in step] for step in per_lane])
        k = slice(60, 120); st = 11; l0 = L0[k, st].mean(); tm = TM[k, st].mean(); l12 = (3 * tm - l0) / 2
        out[f"{tag}_d{d}"] = {"lane0_through": L0.tolist(), "through_mean": TM.tolist(), "exit": ef}
        print(f"  {tag} d={d}+{r} u={u}: teleports {tele}, exit last 30 min {120*np.mean(ef[60:]):.0f} veh/h, station 11 (30-60 min): lane0 {l0:.0f}, lanes1-2 {l12:.0f}, through-mean {tm:.0f}; upstream extent lane0>40: {[100*(i+1) for i in range(19) if L0[k, i].mean() > 40][:1]}..{[100*(i+1) for i in range(19) if L0[k, i].mean() > 40][-1:]} ({time.time()-t0:.0f}s)", flush=True)
    env.close()
json.dump(out, open(S / "lc_variants2.json", "w")); print("saved")
