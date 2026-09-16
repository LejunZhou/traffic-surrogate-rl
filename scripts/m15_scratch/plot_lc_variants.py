import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = r"C:/Users/Jun18/AppData/Local/Temp/v4test"
base = json.load(open(S + "/merge_obs_test.json")); v1 = json.load(open(S + "/lc_variants.json")); v2 = json.load(open(S + "/lc_variants2.json"))
t = (np.arange(120) * 30 + 15) / 60
cases = [("LC2013 defaults", base["d6000_r700_u1.0"], base["d5400_r700_u1.0"]),
         ("keepRight 0, cooperative 1", v1["lc_noKR_only_d6000"], v1["lc_noKR_only_d5400"]),
         ("timeToImpatience 30 s", v2["imp30_d6000"], v2["imp30_d5400"]),
         ("speedGain 3, keepRight 0", v1["lc_speedgain3_noKR_d6000"], v1["lc_speedgain3_noKR_d5400"]),
         ("assertive 2, speedGain 3, keepRight 0", v2["assert2_sg3_d6000"], v2["assert2_sg3_d5400"])]
C0, C1, C2 = "#1d4ed8", "#0f766e", "#c2410c"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "legend.frameon": False})
fig, axes = plt.subplots(2, 5, figsize=(16, 5.6), dpi=150, constrained_layout=True)
for j, (name, c6, c5) in enumerate(cases):
    for i, (c, load) in enumerate([(c5, "5400 + 700"), (c6, "6000 + 700")]):
        L0 = np.array(c["lane0_through"]); TM = np.array(c["through_mean"]); L12 = (3 * TM - L0) / 2
        ef = np.array(c.get("exit_flow_veh_per_step", c.get("exit"))) * 120
        ax = axes[i, j]; st = 11
        ax.plot(t, L0[:, st], color=C0, lw=1.6, label="lane 0"); ax.plot(t, L12[:, st], color=C1, lw=1.6, label="lanes 1-2 mean"); ax.plot(t, TM[:, st], color=C2, lw=1.4, ls="--", label="through-lane mean")
        ax.set_ylim(0, 110); ax.set_title(f"{name}\n{load} veh/h, u = 1, exit {ef[60:].mean():.0f} veh/h (last 30 min)" if i == 0 else f"{load} veh/h, u = 1, exit {ef[60:].mean():.0f} veh/h", fontsize=8.5, loc="left")
        if j == 0: ax.set_ylabel("density at 1200 m (veh/km/lane)")
        if i == 1: ax.set_xlabel("time (min)")
        if i == 0 and j == 0: ax.legend(loc="upper left", fontsize=7.5)
fig.suptitle("Three-lane merge, station 11 (1200 m, just upstream of the nose): per-lane density under five LC2013 parameter sets", fontsize=9.5, x=0.01, ha="left")
fig.savefig("_progress/figures/m15/lane_change_variants.png"); print("wrote _progress/figures/m15/lane_change_variants.png")
