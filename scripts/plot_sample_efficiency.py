"""
Figures 1-7 of draft_pipeline.md §10 from the ledger, the evaluation JSONL
files and the study manifests (M12).

  PYTHONPATH=src python scripts/plot_sample_efficiency.py --arms runs/study/<name>/arms.json \\
      [--e1 _progress/m9_e1_surrogate_study.json] [--rounds runs/aggregation/<study>/rounds.json ...]
      [--out _progress/figures/m12]

arms.json (written by scripts/run_study.sh / by hand):
  {"arms": [
     {"name": "A1 aggregation", "kind": "curve",
      "points": [{"ee": 534, "wall_s": 1800, "seed": 0, "test": "runs/eval/x_test.jsonl", "ood": "...", "policy": "<spec>"}, ...]},
     {"name": "ALINEA", "kind": "band", "points": [{"ee": 300, "test": "...", "ood": "...", "policy": "alinea:..."}]}
  ]}
`test`/`ood` are per-episode JSONL files from scripts/eval_policy_profiles_sumo.py;
`policy` selects the rows when a file holds several policies.

Fig 1  held-out return on T vs cumulative EE (log x), one curve per arm, ALINEA
       and constant u as horizontal bands, bootstrap 95 % error bars
Fig 2  the same against wall-clock seconds
Fig 3  surrogate return-prediction error vs N_0 and data source (E1)
Fig 4  transfer gap per aggregation round: surrogate vs SUMO return per checkpoint
Fig 5  ensemble vs single-surrogate policies: breakdown rate in SUMO
Fig 6  policy structure on one peaked profile (u, Q, d, r, density heat-map)
Fig 7  OOD family results per arm

Colours: the validated reference categorical palette in fixed slot order (one
slot per arm, never cycled); ALINEA / constant references are neutral bands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# reference categorical palette (light mode), fixed slot order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ARM_SLOTS = {"A1 aggregation": 0, "B direct SUMO PPO": 1, "A2 pre-train + fine-tune": 2, "A0 zero-shot": 3,
             "Surrogate-MPC": 4, "single-surrogate": 5, "one-step model": 6, "anticipative": 7,
             # M13 round-0 budget arms (longest prefix wins)
             "A1 aggregation (N0=692, as run)": 0, "A1 aggregation (N0=240, mix1)": 2, "A1 aggregation (N0=240, mix2)": 4}
NEUTRAL = {"ALINEA": "#52514e", "PI-ALINEA": "#7a7975", "constant u": "#a3a29d"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def _rows(path: str | Path, policy: str | None = None) -> list[dict]:
    p = Path(path); p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.open() if l.strip()]
    if policy is not None:
        sel = [r for r in rows if r["policy"] == policy]
        rows = sel or rows
    return rows


def _boot_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)])
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _color(name: str) -> str:
    for key, c in NEUTRAL.items():
        if name.lower().startswith(key.lower()):
            return c
    for key in sorted(ARM_SLOTS, key=len, reverse=True):   # longest prefix wins
        if name.lower().startswith(key.lower()):
            return PALETTE[ARM_SLOTS[key]]
    return PALETTE[6]


def _point_stats(pt: dict, key: str = "test") -> tuple[float, float, float, float]:
    rows = _rows(pt.get(key, ""), pt.get("policy"))
    vals = np.array([r["return"] for r in rows], dtype=np.float64)
    m, lo, hi = _boot_ci(vals)
    bd = float(np.mean([r["breakdown"] for r in rows])) if rows else float("nan")
    return m, lo, hi, bd


def fig_curve(arms: list[dict], xkey: str, xlabel: str, out: Path, title: str, log_x: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    xs_all = []
    for arm in arms:
        pts = [p for p in arm["points"] if p.get(xkey) is not None]
        if not pts:
            continue
        stats = {}
        for p in pts:
            m, lo, hi, _ = _point_stats(p)
            if np.isfinite(m):
                stats.setdefault(float(p[xkey]), []).append((m, lo, hi))
        if not stats:
            continue
        xs = sorted(stats)
        means = np.array([np.mean([s[0] for s in stats[x]]) for x in xs])
        lows = np.array([np.mean([s[1] for s in stats[x]]) for x in xs])
        highs = np.array([np.mean([s[2] for s in stats[x]]) for x in xs])
        c = _color(arm["name"])
        if arm.get("kind") == "band":
            ax.axhspan(lows[0], highs[0], color=c, alpha=0.12, lw=0)
            ax.axhline(means[0], color=c, lw=1.2, ls="--")
            ax.text(0.99, means[0], f" {arm['name']} ({xs[0]:.0f} EE)", color=c, fontsize=8, ha="right", va="bottom", transform=ax.get_yaxis_transform())
        else:
            ax.errorbar(xs, means, yerr=[means - lows, highs - means], color=c, lw=2, marker="o", ms=5, capsize=3, label=arm["name"])
            ax.annotate(arm["name"], (xs[-1], means[-1]), xytext=(6, 0), textcoords="offset points", color=c, fontsize=8, va="center")
            xs_all += xs
    if log_x and xs_all:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("held-out return on T (mean, 95 % CI)")
    ax.set_title(title, loc="left", color=TEXT)
    if any(a.get("kind") != "band" for a in arms):
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def fig_e1(e1: dict, out: Path) -> None:
    keys = [k for k in e1 if "summary" in e1[k]]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=150)
    # (a) size curve
    size_keys = [k for k in keys if k.startswith("gru_N")] + (["gru_M5_all"] if "gru_M5_all" in keys else [])
    if size_keys:
        ns = [e1[k].get("n_train", e1[k]["summary"]["all"]["n"] if k != "gru_M5_all" else None) for k in size_keys]
        ns = [n if n is not None else max([e1[x].get("n_train", 0) for x in size_keys] + [480]) for n in ns]
        errs = [e1[k]["summary"]["all"]["return_rel_err_mean"] for k in size_keys]
        order = np.argsort(ns)
        axes[0].plot(np.array(ns)[order], np.array(errs)[order], color=PALETTE[0], lw=2, marker="o", ms=5)
        axes[0].set_xscale("log"); axes[0].set_xlabel("round-0 training rollouts N_0"); axes[0].set_ylabel("return-prediction error (rel.)")
        axes[0].axhline(0.10, color=MUTED, ls=":", lw=1); axes[0].text(min(ns), 0.10, " gate 10 %", color=MUTED, fontsize=8, va="bottom")
        axes[0].set_title("a  accuracy vs data size", loc="left")
    # (b) variants
    var_keys = [k for k in keys if not k.startswith("gru_N")]
    labels = {"gru_M5_all": "GRU branch, M=5 (mixture)", "gru_M1": "GRU, single member", "gru_random_only": "GRU, random-only data",
              "causal_conv": "causal conv branch", "mlp_padded": "padded-MLP branch", "onestep_mlp": "one-step MLP model"}
    vals = [e1[k]["summary"]["all"]["return_rel_err_mean"] for k in var_keys]
    y = np.arange(len(var_keys))
    axes[1].barh(y, vals, color=[PALETTE[0] if k == "gru_M5_all" else "#9ec5f4" for k in var_keys], height=0.6)
    axes[1].set_yticks(y); axes[1].set_yticklabels([labels.get(k, k) for k in var_keys]); axes[1].invert_yaxis()
    for yi, v in zip(y, vals):
        axes[1].text(v, yi, f" {v:.3f}", va="center", fontsize=8, color=TEXT)
    axes[1].set_xlabel("return-prediction error (rel.)"); axes[1].set_title("b  model / data-source variants", loc="left")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def fig_transfer_gap(rounds_files: list[Path], out: Path) -> None:
    """One marker per (rounds file); colour = round index (sequential ramp)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=150)
    n_rounds = max([r["round"] for rf in rounds_files for r in json.loads(Path(rf).read_text())] + [1])
    seq = plt.cm.Blues(np.linspace(0.35, 0.95, max(n_rounds, 2)))
    markers = ["o", "s", "^", "D", "v", "P"]
    lo = hi = None
    for fi, rf in enumerate(rounds_files):
        rounds = json.loads(Path(rf).read_text())
        tag = Path(rf).parent.name
        for r in rounds:
            for c in r["top"]:
                ax.scatter(c["surrogate_mean"], c["sumo_mean"], color=seq[min(r["round"], n_rounds) - 1], s=30, marker=markers[fi % len(markers)],
                           edgecolor="white", lw=0.8,
                           label=f"round {r['round']}" if c is r["top"][0] and fi == 0 and (r["round"] in (1, 2, 3, n_rounds)) else None)
                lo = c["surrogate_mean"] if lo is None else min(lo, c["surrogate_mean"], c["sumo_mean"])
                hi = c["surrogate_mean"] if hi is None else max(hi, c["surrogate_mean"], c["sumo_mean"])
        if len(rounds_files) > 1:
            ax.scatter([], [], color=MUTED, marker=markers[fi % len(markers)], s=30, label=tag)
    if lo is not None:
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("surrogate validation return"); ax.set_ylabel("SUMO validation return")
    ax.set_title("transfer gap per aggregation round", loc="left"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def fig_breakdown_rates(arms: list[dict], out: Path, key: str = "test") -> None:
    names, rates, cols = [], [], []
    for arm in arms:
        pts = arm["points"]
        if not pts:
            continue
        bds = [_point_stats(p, key)[3] for p in pts]
        bds = [b for b in bds if np.isfinite(b)]
        if not bds:
            continue
        names.append(arm["name"]); rates.append(float(np.mean(bds))); cols.append(_color(arm["name"]))
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    y = np.arange(len(names))
    ax.barh(y, rates, color=cols, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for yi, v in zip(y, rates):
        ax.text(v, yi, f" {v:.2f}", va="center", fontsize=8, color=TEXT)
    ax.set_xlabel(f"breakdown rate on {key.upper()} episodes"); ax.set_title("breakdown rate in SUMO per arm", loc="left")
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def fig_ood(arms: list[dict], out: Path) -> None:
    names, means, los, his, cols = [], [], [], [], []
    for arm in arms:
        pts = [p for p in arm["points"] if p.get("ood")]
        if not pts:
            continue
        st = [_point_stats(p, "ood") for p in pts]
        st = [s for s in st if np.isfinite(s[0])]
        if not st:
            continue
        names.append(arm["name"]); means.append(np.mean([s[0] for s in st])); los.append(np.mean([s[1] for s in st])); his.append(np.mean([s[2] for s in st]))
        cols.append(_color(arm["name"]))
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    y = np.arange(len(names)); means = np.array(means); los = np.array(los); his = np.array(his)
    ax.barh(y, means, xerr=[means - los, his - means], color=cols, height=0.6, capsize=3, error_kw={"lw": 1, "ecolor": MUTED})
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("return on the OOD set O (mean, 95 % CI)"); ax.set_title("out-of-distribution profiles", loc="left")
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def fig_policy_structure(rollouts: dict[str, Path], out: Path) -> None:
    """rollouts: {label: npz path} on the same profile."""
    from sumo_env.rollout import load_rollout_npz

    items = [(lab, *load_rollout_npz(p)) for lab, p in rollouts.items() if Path(p).exists()]
    if not items:
        return
    n = len(items)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 7.2), dpi=150, squeeze=False)
    for j, (lab, arr, meta) in enumerate(items):
        t = np.arange(arr["density"].shape[1]) * 0.5
        a0, a1, a2 = axes[0, j], axes[1, j], axes[2, j]
        a0.plot(t, arr["mainline_demand"], color=MUTED, lw=1.5, label="mainline d")
        a0.plot(t, arr["ramp_arrival"], color=MUTED, lw=1.5, ls="--", label="ramp r")
        a0.plot(t, arr["outflow_vph"], color=PALETTE[0], lw=1.2, label="outflow")
        a0.set_title(f"{lab}: return {meta['metrics']['return']:.0f}, TTS {meta['metrics']['tts_veh_h']:.0f}", loc="left", fontsize=8)
        a0.set_ylabel("veh/h"); a0.set_ylim(0, 3000)
        a1.plot(t, arr["action"], color=PALETTE[1], lw=1.5, label="u")
        a1.set_ylabel("u"); a1.set_ylim(0, 1.05)
        a1b = a1.twinx(); a1b.plot(t, arr["ramp_queue"], color=PALETTE[6], lw=1.2); a1b.set_ylabel("queue (veh)", color=PALETTE[6]); a1b.grid(False)
        im = a2.imshow(arr["density"], aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=120,
                       extent=[0, t[-1] + 0.5, 0, arr["density"].shape[0] * 100 / 1000])
        a2.set_xlabel("time (min)"); a2.set_ylabel("x (km)"); a2.grid(False)
        if j == 0:
            a0.legend(fontsize=7, loc="upper right"); a1.legend(fontsize=7, loc="upper left")
    fig.colorbar(im, ax=axes[2, :].tolist(), label="density (veh/km)", shrink=0.8)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--e1", default="_progress/m9_e1_surrogate_study.json")
    ap.add_argument("--rounds", nargs="*", default=None)
    ap.add_argument("--structure", nargs="*", default=None, help="label=path.npz pairs for figure 6")
    ap.add_argument("--out", default="_progress/figures/m12")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    if args.arms:
        arms = json.loads((PROJECT_ROOT / args.arms).read_text())["arms"]
        fig_curve(arms, "ee", "cumulative SUMO episode-equivalents (EE)", out / "fig1_return_vs_ee.png", "Fig. 1  held-out return vs SUMO budget")
        fig_curve(arms, "wall_s", "wall-clock (s)", out / "fig2_return_vs_wallclock.png", "Fig. 2  held-out return vs wall-clock")
        fig_breakdown_rates(arms, out / "fig5_breakdown_rates.png")
        fig_ood(arms, out / "fig7_ood.png")
        print(f"figs 1, 2, 5, 7 -> {out}")
    e1 = PROJECT_ROOT / args.e1 if args.e1 else None
    if e1 is not None and e1.exists():
        fig_e1(json.loads(e1.read_text()), out / "fig3_surrogate_accuracy.png"); print("fig 3 done")
    if args.rounds:
        fig_transfer_gap([PROJECT_ROOT / r for r in args.rounds], out / "fig4_transfer_gap.png"); print("fig 4 done")
    if args.structure:
        fig_policy_structure({s.split("=", 1)[0]: PROJECT_ROOT / s.split("=", 1)[1] for s in args.structure}, out / "fig6_policy_structure.png")
        print("fig 6 done")


if __name__ == "__main__":
    main()
