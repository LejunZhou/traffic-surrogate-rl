"""
Build the paper's Table I (DeepONet accuracy), Table II (controller comparison) and the
headline TTS reductions from a finished study.

Table I   the final aggregation ensemble on the study store's validation and test splits
          (eval_val_rows.jsonl / eval_test_rows.jsonl from eval_surrogate_regimes.py), mean over
          trajectories of the relative L2 and absolute errors of density, exit flow and return.
Table II  per controller on ID (test) and OOD: full-episode TTS (tts_veh_h), mean and maximum ramp
          queue from one-second samples (episode_queue_mean; mean over episodes of episode_queue_max),
          completed trips (served_veh), and training cost = SUMO episodes and summed compute hours
          (manifest accounted_runtime_s: SUMO episode wall times from the ledgers, DeepONet member
          training, surrogate PPO process time; direct PPO's gradient updates are not included).
          Surrogate-PPO is charged the whole aggregation loop (every round run), not only the rounds
          up to the selected one. Final ID/OOD evaluation is never charged.
Headline  relative TTS reduction 1 - mean(TTS_a) / mean(TTS_b), paired over identical
          (profile, SUMO seed) episodes, 95 % bootstrap CI.

Example: python scripts/build_paper_tables.py --arms runs/study/m14/arms.json --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TABLE2_ROWS = [  # (paper label, manifest arm name, how to choose the point)
    ("Constant metering (u = 0)", "fixed u=0", "only"),
    ("Constant metering (u = 0.5)", "fixed u=0.5", "only"),
    ("Constant metering (u = 1)", "fixed u=1", "only"),
    ("ALINEA", "ALINEA", "only"),
    ("PI-ALINEA", "PI-ALINEA", "only"),
    ("SUMO-PPO", "B direct SUMO PPO", "budget_selected"),
    ("Surrogate-PPO", "A1 aggregation", "validation_selected"),
    ("Surrogate-MPC", "Surrogate-MPC", "only"),
]
TABLE1_METRICS = [  # (label, row key, scale)
    ("Density relative L2 error (%)", "rel_l2_density", 100.0),
    ("Density MAE (veh/km)", "mean_abs_err_density", 1.0),
    ("Outflow relative L2 error (%)", "rel_l2_flow", 100.0),
    ("Outflow MAE (veh/h)", "mean_abs_err_flow", 1.0),
    ("Return relative error (%)", "return_rel_err", 100.0),
    ("Return MAE", "return_err", 1.0),
]


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _key(row: dict) -> tuple:
    return (row["profile_set"], int(row["profile_index"]), int(row["sumo_seed"]))


def select_point(arm: dict, how: str, seed: int) -> dict:
    points = [p for p in arm["points"] if int(p.get("seed", 0)) == seed] or arm["points"]
    if how == "only":
        if len(points) != 1:
            raise ValueError(f"{arm['name']}: expected one point, found {len(points)}")
        return points[0]
    chosen = [p for p in points if p.get(how)]
    if len(chosen) != 1:
        raise ValueError(f"{arm['name']}: expected one point marked {how}, found {len(chosen)}")
    return chosen[0]


def loop_cost(arm: dict, seed: int) -> tuple[int, float]:
    """Surrogate-PPO is charged every round the loop ran (the selection needs all of them)."""
    points = [p for p in arm["points"] if int(p.get("seed", 0)) == seed]
    last = max(points, key=lambda p: int(p.get("round", 0)))
    return int(last["ee"]), float(last["accounted_runtime_s"])


def traffic_metrics(rows: list[dict]) -> dict:
    def mean(key):
        return float(np.mean([float(r[key]) for r in rows]))

    one_second = all("episode_queue_mean" in r and "episode_queue_max" in r for r in rows)
    return {
        "n": len(rows),
        "tts": mean("tts_veh_h"),
        "mean_queue": mean("episode_queue_mean") if one_second else float("nan"),
        "max_queue": mean("episode_queue_max") if one_second else mean("max_queue"),
        "queue_samples": "1 s" if one_second else "30 s (episode_queue_* missing)",
        "completed_trips": mean("served_veh"),
        "breakdown_rate": float(np.mean([bool(r["breakdown"]) for r in rows])),
    }


def paired_reduction(rows_a: list[dict], rows_b: list[dict], n_boot: int = 5000, seed: int = 0) -> dict:
    """1 - mean(TTS_a) / mean(TTS_b) over the episodes both controllers ran, with a paired bootstrap CI."""
    b = {_key(r): float(r["tts_veh_h"]) for r in rows_b}
    pairs = np.array([(float(r["tts_veh_h"]), b[_key(r)]) for r in rows_a if _key(r) in b])
    if len(pairs) == 0:
        raise ValueError("no common episodes")
    point = 1.0 - pairs[:, 0].mean() / pairs[:, 1].mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    boot = 1.0 - pairs[idx, 0].mean(1) / pairs[idx, 1].mean(1)
    diff = pairs[:, 1] - pairs[:, 0]
    dboot = diff[idx].mean(1)
    return {"n_pairs": int(len(pairs)), "reduction_pct": 100 * point,
            "ci95_pct": [100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5))],
            "tts_saved_veh_h": float(diff.mean()),
            "tts_saved_ci95": [float(np.percentile(dboot, 2.5)), float(np.percentile(dboot, 97.5))]}


def table1(ensemble: Path) -> dict:
    out = {"ensemble": str(ensemble), "splits": {}}
    for split in ("val", "test"):
        rows_path = ensemble / f"eval_{split}_rows.jsonl"
        if not rows_path.exists():
            raise FileNotFoundError(f"{rows_path}: run scripts/eval_surrogate_regimes.py --ensemble {ensemble} "
                                    f"--store <study store> --split {split}")
        rows = _jsonl(rows_path)
        summary_path = ensemble / f"eval_{split}.json"
        gate = json.loads(summary_path.read_text())["gate"] if summary_path.exists() else None
        out["splits"][split] = {"n": len(rows), "gate": gate,
                                **{key: float(np.nanmean([float(r[key]) for r in rows])) * scale
                                   for _, key, scale in TABLE1_METRICS}}
    return out


def table2(manifest: dict, seed: int, sets: tuple[str, ...]) -> dict:
    arms = {arm["name"]: arm for arm in manifest["arms"]}
    out, episode_rows = {}, {}
    for label, name, how in TABLE2_ROWS:
        if name not in arms:
            print(f"[tables] {label}: arm {name!r} not in the manifest, skipped")
            continue
        point = select_point(arms[name], how, seed)
        row = {"arm": name, "policy": point["policy"]}
        for set_name in sets:
            path = _path(point[set_name])
            if not path.exists():
                raise FileNotFoundError(f"{label}: {path} missing; run the final evaluation first")
            rows = _jsonl(path)
            episode_rows[(label, set_name)] = rows
            row[set_name] = traffic_metrics(rows)
        if name == "A1 aggregation":
            row["sumo_episodes"], runtime = loop_cost(arms[name], seed)
            row["selected_round"] = point.get("round")
        else:
            row["sumo_episodes"], runtime = int(point["ee"]), float(point["accounted_runtime_s"])
        row["compute_h"] = runtime / 3600.0
        out[label] = row
    return {"rows": out}, episode_rows


def headline(episode_rows: dict, sets: tuple[str, ...]) -> dict:
    out = {}
    for ref in ("PI-ALINEA", "ALINEA", "SUMO-PPO"):
        for set_name in sets:
            a, b = episode_rows.get(("Surrogate-PPO", set_name)), episode_rows.get((ref, set_name))
            if a and b:
                out[f"Surrogate-PPO vs {ref} ({set_name})"] = paired_reduction(a, b)
    return out


def _markdown(t1: dict | None, t2: dict, head: dict, sets: tuple[str, ...]) -> str:
    lines = []
    if t1:
        lines += ["## Table I: DeepONet prediction accuracy", f"ensemble `{t1['ensemble']}`", "",
                  "| Metric | Validation | Test |", "|---|---|---|"]
        for label, key, _ in TABLE1_METRICS:
            lines.append(f"| {label} | {t1['splits']['val'][key]:.2f} | {t1['splits']['test'][key]:.2f} |")
        lines.append(f"| trajectories | {t1['splits']['val']['n']} | {t1['splits']['test']['n']} |")
        lines.append("")
    set_label = {"test": "ID", "ood": "OOD"}
    cols = " | ".join(f"{set_label.get(s, s)} TTS | {set_label.get(s, s)} mean q | {set_label.get(s, s)} max q | "
                      f"{set_label.get(s, s)} trips" for s in sets)
    lines += ["## Table II: controller comparison", "",
              f"| Method | {cols} | SUMO episodes | compute (h) |",
              "|---|" + "---|" * (4 * len(sets) + 2)]
    for label, row in t2["rows"].items():
        vals = " | ".join(f"{row[s]['tts']:.2f} | {row[s]['mean_queue']:.2f} | {row[s]['max_queue']:.2f} | "
                          f"{row[s]['completed_trips']:.1f}" for s in sets)
        lines.append(f"| {label} | {vals} | {row['sumo_episodes']} | {row['compute_h']:.1f} |")
    lines += ["", "TTS in veh h; queues in vehicles (1-s samples; max = mean of per-episode maxima); "
              "compute = summed compute hours (see script docstring).", "", "## Headline TTS reductions", ""]
    for name, h in head.items():
        lines.append(f"- {name}: {h['reduction_pct']:.1f} % [{h['ci95_pct'][0]:.1f}, {h['ci95_pct'][1]:.1f}], "
                     f"{h['tts_saved_veh_h']:.2f} veh h saved per episode, n = {h['n_pairs']}")
    return "\n".join(lines) + "\n"


def _latex_rows(t2: dict, sets: tuple[str, ...]) -> str:
    out = []
    for label, row in t2["rows"].items():
        cells = [f"{row[s]['tts']:.2f} & {row[s]['mean_queue']:.2f} & {row[s]['max_queue']:.2f} & {row[s]['completed_trips']:.1f}"
                 for s in sets]
        out.append(f"{label} & " + " & ".join(cells) + f" & {row['sumo_episodes']} & {row['compute_h']:.1f} \\\\")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="runs/study/m14/arms.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--study", default="m14", help="aggregation study prefix (runs/aggregation/<study>_s<seed>)")
    ap.add_argument("--ensemble", default=None, help="Table I ensemble (default: the last aggregation round's)")
    ap.add_argument("--sets", nargs="+", default=["test", "ood"])
    ap.add_argument("--out", default="runs/study/m14/tables")
    args = ap.parse_args()
    sets = tuple(args.sets)
    manifest = json.loads(_path(args.arms).read_text())
    ensemble = args.ensemble
    if ensemble is None:
        rounds_path = PROJECT_ROOT / f"runs/aggregation/{args.study}_s{args.seed}/rounds.json"
        ensemble = json.loads(rounds_path.read_text())[-1]["ensemble_after"] if rounds_path.exists() else None
    t1 = table1(_path(ensemble)) if ensemble else None
    t2, episode_rows = table2(manifest, args.seed, sets)
    head = headline(episode_rows, sets)
    out = _path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"seed": args.seed, "arms": str(args.arms), "table1": t1, "table2": t2, "headline": head,
               "compute_definition": "summed compute: SUMO episode wall times (ledgers) + DeepONet member training "
                                     "+ surrogate PPO process time; direct-PPO gradient updates excluded"}
    (out / "tables.json").write_text(json.dumps(payload, indent=1))
    (out / "tables.md").write_text(_markdown(t1, t2, head, sets), encoding="utf-8")
    (out / "table2_rows.tex").write_text(_latex_rows(t2, sets), encoding="utf-8")
    print(_markdown(t1, t2, head, sets))
    print(f"written: {out / 'tables.json'}, tables.md, table2_rows.tex")


if __name__ == "__main__":
    main()
