#!/usr/bin/env python
"""Plot PPO reward curves from an RL run directory.

The PPO trainer writes two useful reward logs:
- progress.csv: SB3 rollout summaries, including rollout/ep_rew_mean.
- monitor.csv: raw per-episode returns from the Monitor wrapper.

By default this script plots progress.csv's rollout/ep_rew_mean because it is
the standard smoothed reward curve shown during SB3 training.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _latest_run_dir() -> Path:
    candidates: list[Path] = []
    for root in (Path("runs/rl"), Path("runs/ppo")):
        if root.exists():
            candidates.extend(
                path
                for path in root.iterdir()
                if path.is_dir()
                and ((path / "progress.csv").exists() or (path / "monitor.csv").exists())
            )
    if not candidates:
        raise FileNotFoundError(
            "No PPO run directory found under runs/rl or runs/ppo. "
            "Pass --run-dir explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_progress(
    path: Path, max_steps: float | None
) -> tuple[list[float], list[float], str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")

    y_key = "rollout/ep_rew_mean"
    if y_key not in rows[0]:
        available = ", ".join(rows[0].keys())
        raise KeyError(f"{path} has no {y_key!r} column. Available columns: {available}")

    x_key = "time/total_timesteps"
    if x_key not in rows[0]:
        x_key = "time/iterations" if "time/iterations" in rows[0] else ""

    xs: list[float] = []
    ys: list[float] = []
    for idx, row in enumerate(rows, start=1):
        raw_y = row.get(y_key, "")
        if raw_y == "":
            continue
        x_value = float(row[x_key]) if x_key else float(idx)
        if max_steps is not None and x_value > max_steps:
            continue
        xs.append(x_value)
        ys.append(float(raw_y))

    x_label = "Timesteps" if x_key == "time/total_timesteps" else "PPO iteration"
    return xs, ys, x_label


def _read_monitor(
    path: Path, window: int, max_steps: float | None
) -> tuple[list[float], list[float], list[float] | None]:
    with path.open(newline="", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows:
        raise ValueError(f"{path} has no episode rows")

    xs: list[float] = []
    ys: list[float] = []
    total_steps = 0.0
    for row in rows:
        total_steps += float(row["l"])
        if max_steps is not None and total_steps > max_steps:
            break
        xs.append(total_steps)
        ys.append(float(row["r"]))

    rolling: list[float] | None = None
    if window > 1 and len(ys) >= window:
        rolling = []
        for idx in range(len(ys)):
            start = max(0, idx + 1 - window)
            values = ys[start : idx + 1]
            rolling.append(sum(values) / len(values))
    return xs, ys, rolling


def plot_reward_curve(
    run_dir: Path,
    source: str,
    out_path: Path | None,
    monitor_window: int,
    max_steps: float | None,
) -> Path:
    progress_path = run_dir / "progress.csv"
    monitor_path = run_dir / "monitor.csv"

    if source == "auto":
        source = "progress" if progress_path.exists() else "monitor"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if source == "progress":
        xs, ys, x_label = _read_progress(progress_path, max_steps)
        ax.plot(xs, ys, linewidth=1.8, label="rollout/ep_rew_mean")
        ax.set_xlabel(x_label)
    elif source == "monitor":
        xs, ys, rolling = _read_monitor(monitor_path, monitor_window, max_steps)
        ax.plot(xs, ys, alpha=0.35, linewidth=1.0, label="episode return")
        if rolling is not None:
            ax.plot(xs, rolling, linewidth=1.8, label=f"{monitor_window}-episode mean")
        ax.set_xlabel("Timesteps")
    else:
        raise ValueError(f"Unknown source: {source}")

    if not xs:
        raise ValueError(
            f"No reward points found for source={source!r}"
            + (f" with --max-steps {max_steps:g}" if max_steps is not None else "")
        )

    ax.set_ylabel("Episode reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out_path is None:
        out_path = run_dir / f"reward_curve_{source}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a PPO reward curve")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="PPO run directory containing progress.csv or monitor.csv. Defaults to latest run.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "progress", "monitor"),
        default="auto",
        help="Which log to plot. progress uses rollout/ep_rew_mean; monitor uses episode returns.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument(
        "--max-steps",
        type=float,
        default=None,
        help="Only plot points whose timestep is at or before this value.",
    )
    parser.add_argument(
        "--monitor-window",
        type=int,
        default=10,
        help="Rolling mean window for --source monitor",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run_dir()
    out = plot_reward_curve(
        run_dir=run_dir,
        source=args.source,
        out_path=args.out,
        monitor_window=args.monitor_window,
        max_steps=args.max_steps,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
