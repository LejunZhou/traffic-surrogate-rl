"""
Parse SB3 PPO stdout logs and emit a per-iteration ep_rew_mean comparison.

Used by M6b to overlay M5c (surrogate-trained), M6 (SUMO 20k), and M6b
(SUMO 100k) training trajectories on the same iteration axis. This is
the test of the "M6 just needs more updates" hypothesis: if M6b's
trajectory at iter 100-209 climbs back toward M5c's, the hypothesis is
confirmed.

Reads from the ASCII task-output captures (Claude run_in_background
temp dir) rather than the project-local Tee'd logs, because PowerShell's
Tee-Object writes UTF-16 LE while the temp captures are clean ASCII.

Outputs:
- A CSV with columns: iter, m5c, m6, m6b (any column NaN if that run
  didn't reach that iter).
- A markdown summary table at sentinel iterations.
- Optional PNG plot if matplotlib is available and --plot is passed.

Usage:
  python scripts/compare_training_trajectories.py \\
      --m5c <m5c_task_output> \\
      --m6  <m6_task_output> \\
      --m6b <m6b_task_output> \\
      --out _progress/m6b_training_curves.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ITER_RE = re.compile(r"iterations\s*\|\s*(\d+)")
REW_RE = re.compile(r"ep_rew_mean\s*\|\s*(-?[\d.eE+\-]+)")


def parse_trajectory(log_path: Path) -> dict[int, float]:
    """Return {iter: ep_rew_mean} parsed from a single SB3 PPO stdout log."""
    if not log_path or not log_path.exists():
        return {}
    rewards: dict[int, float] = {}
    last_reward: float | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_rew = REW_RE.search(line)
            if m_rew:
                try:
                    last_reward = float(m_rew.group(1))
                except ValueError:
                    continue
                continue
            m_iter = ITER_RE.search(line)
            if m_iter and last_reward is not None:
                try:
                    iter_num = int(m_iter.group(1))
                except ValueError:
                    continue
                rewards[iter_num] = last_reward
                last_reward = None  # reset so we don't double-count
    return rewards


def write_csv(out_path: Path, m5c: dict[int, float], m6: dict[int, float], m6b: dict[int, float]) -> None:
    all_iters = sorted(set(m5c) | set(m6) | set(m6b))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "m5c_surrogate", "m6_sumo_20k", "m6b_sumo_100k"])
        for it in all_iters:
            w.writerow([
                it,
                f"{m5c[it]:.4f}" if it in m5c else "",
                f"{m6[it]:.4f}" if it in m6 else "",
                f"{m6b[it]:.4f}" if it in m6b else "",
            ])
    print(f"[compare] wrote {out_path}  (n_rows={len(all_iters)})")


def print_summary(m5c: dict[int, float], m6: dict[int, float], m6b: dict[int, float]) -> None:
    """Tabulate ep_rew_mean at sentinel iterations for quick at-a-glance comparison."""
    sentinel_iters = [1, 5, 10, 20, 42, 50, 100, 150, 200, 209]
    sentinel_iters = [it for it in sentinel_iters if it <= max([*m5c, *m6, *m6b, 0])]

    print()
    print(f"{'iter':>5} | {'M5c (surrogate)':>16} | {'M6 (SUMO 20k)':>14} | {'M6b (SUMO 100k)':>16}")
    print("-" * 64)
    for it in sentinel_iters:
        a = f"{m5c[it]:.2f}" if it in m5c else "-"
        b = f"{m6[it]:.2f}" if it in m6 else "-"
        c = f"{m6b[it]:.2f}" if it in m6b else "-"
        print(f"{it:>5} | {a:>16} | {b:>14} | {c:>16}")
    print()

    last_m5c = max(m5c) if m5c else None
    last_m6 = max(m6) if m6 else None
    last_m6b = max(m6b) if m6b else None
    print(f"last iters: M5c={last_m5c} ({m5c[last_m5c]:.2f}), "
          f"M6={last_m6} ({m6[last_m6]:.2f} if last_m6 else '-'), "
          f"M6b={'in progress at ' + str(last_m6b) if last_m6b is not None else 'not started'}"
          f"{'' if last_m6b is None else f' ({m6b[last_m6b]:.2f})'}")


def maybe_plot(out_path: Path | None, m5c: dict[int, float], m6: dict[int, float], m6b: dict[int, float]) -> None:
    if out_path is None:
        return
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data, color in (
        ("M5c — DeepONet surrogate (100k ts)", m5c, "tab:blue"),
        ("M6 — direct SUMO (20k ts)", m6, "tab:orange"),
        ("M6b — direct SUMO (100k ts)", m6b, "tab:green"),
    ):
        if not data:
            continue
        xs = sorted(data)
        ys = [data[x] for x in xs]
        ax.plot(xs, ys, label=label, color=color, linewidth=1.5)
    ax.set_xlabel("PPO iteration (each = 480 env timesteps, ~4 episodes)")
    ax.set_ylabel("ep_rew_mean")
    ax.set_title("PPO training trajectories — same agent, different env backends")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay PPO training trajectories")
    parser.add_argument("--m5c", type=Path, default=None, help="path to M5c SB3 stdout log")
    parser.add_argument("--m6", type=Path, default=None, help="path to M6 SB3 stdout log")
    parser.add_argument("--m6b", type=Path, default=None, help="path to M6b SB3 stdout log")
    parser.add_argument("--out", type=Path, default=Path("_progress/m6b_training_curves.csv"), help="CSV output path")
    parser.add_argument("--plot", type=Path, default=None, help="optional PNG output path")
    args = parser.parse_args()

    m5c = parse_trajectory(args.m5c) if args.m5c else {}
    m6 = parse_trajectory(args.m6) if args.m6 else {}
    m6b = parse_trajectory(args.m6b) if args.m6b else {}

    if not (m5c or m6 or m6b):
        print("[compare] no data parsed from any log. Check paths and log format.", file=sys.stderr)
        sys.exit(1)

    write_csv(args.out, m5c, m6, m6b)
    print_summary(m5c, m6, m6b)
    maybe_plot(args.plot, m5c, m6, m6b)


if __name__ == "__main__":
    main()
