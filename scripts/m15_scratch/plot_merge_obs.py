import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(r"C:/Users/Jun18/AppData/Local/Temp/v4test/merge_obs_test.json"))
x_grid = np.arange(100, 2000, 100); t = (np.arange(120) * 30 + 15) / 60
C0, C1, C2 = "#1d4ed8", "#c2410c", "#52514e"   # lane 0 (through), through-lane mean, all-lane mean incl. accel lane
cases = ["d4800_r700_u1.0", "d5400_r700_u1.0", "d6000_r700_u1.0", "d6600_r700_u1.0", "d6000_r700_u0.5"]
labels = {"d4800_r700_u1.0": "4800 + 700, u = 1", "d5400_r700_u1.0": "5400 + 700, u = 1", "d6000_r700_u1.0": "6000 + 700, u = 1",
          "d6600_r700_u1.0": "6600 + 700, u = 1", "d6000_r700_u0.5": "6000 + 700, u = 0.5"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "legend.frameon": False})
fig, axes = plt.subplots(3, 5, figsize=(16, 8.4), dpi=150, constrained_layout=True)
print(f"{'case':<22s} {'station':>8s} {'onset lane0 (min)':>18s} {'onset through-mean':>19s} {'onset all-mean':>15s} {'peak lane0':>11s} {'peak through':>13s}")
for j, c in enumerate(cases):
    L0 = np.array(d[c]["lane0_through"]); TM = np.array(d[c]["through_mean"]); AM = np.array(d[c]["all_mean"])
    # row 0-1: space-time of through-lane mean and lane 0
    for i, (field, name) in enumerate([(TM, "through-lane mean"), (L0, "lane 0 (through)")]):
        ax = axes[i, j]; im = ax.imshow(field.T, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=120, extent=[0, 60, 50, 1950])
        ax.axhline(1300, color="#0b0b0b", lw=0.8, ls="--"); ax.grid(False)
        ax.set_title(f"{labels[c]}\n{name}" if i == 0 else name, fontsize=8.5, loc="left")
        if j == 0: ax.set_ylabel("x (m)")
        if j == 4: fig.colorbar(im, ax=ax, shrink=0.8, label="veh/km/lane")
    # row 2: time series at station 11 (1200 m, upstream of the nose)
    ax = axes[2, j]; st = 11
    ax.plot(t, L0[:, st], color=C0, lw=1.6, label="lane 0 (through)"); ax.plot(t, TM[:, st], color=C1, lw=1.6, label="through-lane mean")
    ax.plot(t, AM[:, st], color=C2, lw=1.2, ls=":", label="all-lane mean")
    ax.axhline(40, color="#9ca3af", lw=0.8); ax.set_ylim(0, 130); ax.set_xlabel("time (min)"); ax.set_title("station 11 (1200 m)", fontsize=8.5, loc="left")
    if j == 0: ax.set_ylabel("density (veh/km/lane)"); ax.legend(loc="upper left", fontsize=7.5)
    for st, nm in ((11, "1200 m"), (12, "1300 m"), (10, "1100 m")):
        def onset(v, thr=40.0, hold=4):
            for k in range(len(v) - hold):
                if np.all(v[k:k + hold] > thr): return t[k]
            return float("nan")
        print(f"{labels[c]:<22s} {nm:>8s} {onset(L0[:, st]):18.1f} {onset(TM[:, st]):19.1f} {onset(AM[:, st]):15.1f} {L0[:, st].max():11.0f} {TM[:, st].max():13.0f}")
fig.suptitle("Three-lane merge test (scratch v4, LC2013 defaults): is the ramp-merge shock visible in the lane-averaged density?  "
             "rows: through-lane mean vs lane 0 (space-time), station-11 traces; dashed = merge nose 1300 m", fontsize=9.5, x=0.01, ha="left")
fig.savefig("_progress/figures/m15/merge_observation_test.png"); print("wrote _progress/figures/m15/merge_observation_test.png")
