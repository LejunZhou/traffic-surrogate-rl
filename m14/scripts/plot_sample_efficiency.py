"""
Plot M14 controller performance and model-to-SUMO transfer from saved results.

Example: python scripts/plot_sample_efficiency.py --arms runs/study/m14/arms.json
         --rounds runs/aggregation/m14_s0/rounds.json --out runs/figures/m14

Figures show held-out return versus SUMO episodes and partial runtime
accounting, checkpoint transfer gaps, breakdown rates, and OOD performance.
The runtime figure is not elapsed wall time or normalized compute. Controller
curves use the episode JSONL results named in the generated arms manifest.
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
ARM_SLOTS = {"A1 aggregation": 0, "B direct SUMO PPO": 1, "A0 zero-shot": 3, "Surrogate-MPC": 4}
NEUTRAL = {"ALINEA": "#52514e", "PI-ALINEA": "#7a7975", "tuned constant": "#a3a29d", "fixed u=": "#a3a29d"}
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
            ax.text(0.99, means[0], f" {arm['name']}", color=c, fontsize=8, ha="right", va="bottom", transform=ax.get_yaxis_transform())
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


def comparison_points(arm: dict) -> list[dict]:
    """Select one trained controller per seed without looking at held-out results."""
    points = arm["points"]
    marker = {"A1 aggregation": "validation_selected", "B direct SUMO PPO": "budget_selected"}.get(arm["name"])
    if marker is None:
        return points
    if any(marker not in point for point in points):
        raise ValueError(f"{arm['name']} lacks {marker} metadata; rebuild arms.json with build_arms_manifest.py")
    selected = [point for point in points if point[marker]]
    if len({point.get("seed", 0) for point in selected}) != len(selected):
        raise ValueError(f"{arm['name']} has more than one selected controller per seed")
    return selected


def fig_breakdown_rates(arms: list[dict], out: Path, key: str = "test") -> None:
    names, rates, cols = [], [], []
    for arm in arms:
        pts = comparison_points(arm)
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
    ax.set_xlabel(f"breakdown rate on {key.upper()} episodes"); ax.set_title("SUMO breakdown: selected controllers\nA1: best validation; direct PPO: largest available requested budget", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def fig_ood(arms: list[dict], out: Path) -> None:
    names, means, los, his, cols = [], [], [], [], []
    for arm in arms:
        pts = [p for p in comparison_points(arm) if p.get("ood")]
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
    ax.set_xlabel("return on the OOD set O (mean, 95 % CI)"); ax.set_title("OOD: selected controllers\nA1: best validation; direct PPO: largest available requested budget", loc="left", fontsize=9)
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
    ap.add_argument("--rounds", nargs="*", default=None)
    ap.add_argument("--structure", nargs="*", default=None, help="label=path.npz pairs for figure 6")
    ap.add_argument("--out", default="runs/figures/m14")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    if args.arms:
        arms = json.loads((PROJECT_ROOT / args.arms).read_text())["arms"]
        fig_curve(arms, "ee", "cumulative SUMO episode-equivalents (EE)", out / "fig1_return_vs_ee.png", "Fig. 1  held-out return vs SUMO budget")
        fig_curve(arms, "accounted_runtime_s", "partial accounted runtime (s)", out / "fig2_return_vs_accounted_runtime.png", "Fig. 2  held-out return vs partial runtime accounting")
        fig_breakdown_rates(arms, out / "fig5_breakdown_rates.png")
        fig_ood(arms, out / "fig7_ood.png")
        print(f"figs 1, 2, 5, 7 -> {out}")
    if args.rounds:
        fig_transfer_gap([PROJECT_ROOT / r for r in args.rounds], out / "fig4_transfer_gap.png"); print("fig 4 done")
    if args.structure:
        fig_policy_structure({s.split("=", 1)[0]: PROJECT_ROOT / s.split("=", 1)[1] for s in args.structure}, out / "fig6_policy_structure.png")
        print("fig 6 done")


if __name__ == "__main__":
    main()
