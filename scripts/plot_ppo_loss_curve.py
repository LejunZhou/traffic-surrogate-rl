#!/usr/bin/env python
"""Plot PPO training loss curves from an SB3 progress.csv file."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = (
    "train/loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
)


def _latest_run_dir() -> Path:
    candidates: list[Path] = []
    for root in (Path("runs/rl"), Path("runs/ppo")):
        if root.exists():
            candidates.extend(
                path
                for path in root.iterdir()
                if path.is_dir() and (path / "progress.csv").exists()
            )
    if not candidates:
        raise FileNotFoundError(
            "No PPO progress.csv found under runs/rl or runs/ppo. "
            "Pass --run-dir explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_progress(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _available_train_metrics(rows: list[dict[str, str]]) -> list[str]:
    return sorted(key for key in rows[0] if key.startswith("train/"))


def _parse_metrics(raw_metrics: list[str] | None) -> list[str]:
    if not raw_metrics:
        return list(DEFAULT_METRICS)
    metrics: list[str] = []
    for raw in raw_metrics:
        metrics.extend(part.strip() for part in raw.split(",") if part.strip())
    return metrics


def _series(
    rows: list[dict[str, str]],
    metric: str,
    max_steps: float | None,
) -> tuple[list[float], list[float], str]:
    x_key = "time/total_timesteps"
    if x_key not in rows[0]:
        x_key = "time/iterations" if "time/iterations" in rows[0] else ""

    xs: list[float] = []
    ys: list[float] = []
    for idx, row in enumerate(rows, start=1):
        raw_y = row.get(metric, "")
        if raw_y == "":
            continue
        x_value = float(row[x_key]) if x_key else float(idx)
        if max_steps is not None and x_value > max_steps:
            continue
        xs.append(x_value)
        ys.append(float(raw_y))

    x_label = "Timesteps" if x_key == "time/total_timesteps" else "PPO iteration"
    return xs, ys, x_label


def _smooth(values: list[float], window: int, method: str) -> list[float]:
    if window <= 1:
        return values

    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        chunk = values[start : idx + 1]
        if method == "median":
            smoothed.append(float(statistics.median(chunk)))
        elif method == "mean":
            smoothed.append(float(sum(chunk) / len(chunk)))
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    return smoothed


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence.")
    if percentile <= 0.0:
        return min(values)
    if percentile >= 100.0:
        return max(values)

    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def plot_loss_curve(
    run_dir: Path,
    metrics: list[str],
    out_path: Path | None,
    max_steps: float | None,
    smooth_window: int,
    smooth_method: str,
    hide_raw: bool,
    y_max_percentile: float | None,
) -> Path:
    progress_path = run_dir / "progress.csv"
    rows = _read_progress(progress_path)
    available = _available_train_metrics(rows)
    selected = [metric for metric in metrics if metric in available]
    if not selected:
        raise KeyError(
            "None of the requested metrics were found. "
            f"Requested: {', '.join(metrics)}. "
            f"Available train metrics: {', '.join(available)}"
        )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_label = "Timesteps"
    plotted = 0
    plotted_values: list[float] = []
    for metric in selected:
        xs, ys, x_label = _series(rows, metric, max_steps)
        if not xs:
            continue

        if smooth_window > 1:
            smoothed = _smooth(ys, smooth_window, smooth_method)
            if not hide_raw:
                ax.plot(xs, ys, alpha=0.2, linewidth=0.9, label=f"{metric} raw")
            ax.plot(
                xs,
                smoothed,
                linewidth=1.8,
                label=f"{metric} {smooth_method}-{smooth_window}",
            )
            plotted_values.extend(smoothed)
        else:
            ax.plot(xs, ys, linewidth=1.6, label=metric)
            plotted_values.extend(ys)
        plotted += 1

    if plotted == 0:
        raise ValueError(
            "No loss points found"
            + (f" with --max-steps {max_steps:g}" if max_steps is not None else "")
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss")
    if y_max_percentile is not None:
        ax.set_ylim(top=_percentile(plotted_values, y_max_percentile))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out_path is None:
        out_path = run_dir / "ppo_training_loss.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO training loss curves")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="PPO run directory containing progress.csv. Defaults to latest run.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Metric to plot, e.g. train/loss. Can be repeated or comma-separated. "
            "Defaults to common PPO loss metrics."
        ),
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available train/* metrics in progress.csv and exit.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument(
        "--max-steps",
        type=float,
        default=None,
        help="Only plot points whose timestep is at or before this value.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Trailing smoothing window in logged PPO updates. Use values like 20 or 50.",
    )
    parser.add_argument(
        "--smooth-method",
        choices=("mean", "median"),
        default="median",
        help="Smoothing method used when --smooth-window > 1.",
    )
    parser.add_argument(
        "--hide-raw",
        action="store_true",
        help="Only draw the smoothed line when --smooth-window > 1.",
    )
    parser.add_argument(
        "--y-max-percentile",
        type=float,
        default=None,
        help="Set y-axis top to this percentile of plotted values, e.g. 95.",
    )
    args = parser.parse_args()

    if args.smooth_window < 1:
        parser.error("--smooth-window must be >= 1")
    if args.y_max_percentile is not None and not (0.0 < args.y_max_percentile <= 100.0):
        parser.error("--y-max-percentile must be in (0, 100]")

    run_dir = args.run_dir or _latest_run_dir()
    rows = _read_progress(run_dir / "progress.csv")
    if args.list_metrics:
        for metric in _available_train_metrics(rows):
            print(metric)
        return

    out = plot_loss_curve(
        run_dir=run_dir,
        metrics=_parse_metrics(args.metric),
        out_path=args.out,
        max_steps=args.max_steps,
        smooth_window=args.smooth_window,
        smooth_method=args.smooth_method,
        hide_raw=args.hide_raw,
        y_max_percentile=args.y_max_percentile,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
