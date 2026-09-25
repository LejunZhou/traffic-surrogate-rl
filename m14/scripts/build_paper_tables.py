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
          up to the selected one. Final ID/OOD evaluation is never charged. With several direct PPO
          budgets, SUMO-PPO is the largest and each smaller one gets a "SUMO-PPO (<n> ep.)" row.
Headline  relative TTS reduction 1 - mean(TTS_a) / mean(TTS_b), paired over identical
          (profile, SUMO seed) episodes, 95 % bootstrap CI.
Seeds     with --seeds s1 s2 ..., each policy seed's tables go to <out>/seed_<s>/ and <out>/tables.md
          summarises them: Tables I and II as mean ± standard deviation over seeds, and each headline
          reduction over all seeds with a hierarchical bootstrap CI (policy seeds resampled, then episodes
          within each seed), so the interval covers training randomness as well as traffic randomness.

Example: python scripts/build_paper_tables.py --arms runs/study/m14/arms.json --seed 0
         python scripts/build_paper_tables.py --arms runs/study/m14/arms.json --seeds 0 1 2
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


def _other_budgets(arm: dict, seed: int) -> list[dict]:
    """Direct SUMO-PPO points of this seed at budgets below the selected (largest) one, smallest first."""
    points = [p for p in arm["points"] if int(p.get("seed", 0)) == seed and not p.get("budget_selected")]
    return sorted(points, key=lambda p: p["nominal_ee"])


def table2(manifest: dict, seed: int, sets: tuple[str, ...]) -> dict:
    """Rows of TABLE2_ROWS; a direct SUMO-PPO arm with several budgets adds a "SUMO-PPO (<n> ep.)" row per
    smaller budget after the SUMO-PPO row (the largest budget)."""
    arms = {arm["name"]: arm for arm in manifest["arms"]}
    todo = []
    for label, name, how in TABLE2_ROWS:
        if name not in arms:
            print(f"[tables] {label}: arm {name!r} not in the manifest, skipped")
            continue
        todo.append((label, name, select_point(arms[name], how, seed)))
        if how == "budget_selected":
            todo += [(f"{label} ({p['nominal_ee']} ep.)", name, p) for p in _other_budgets(arms[name], seed)]
    out, episode_rows = {}, {}
    for label, name, point in todo:
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


def _references(labels) -> list[str]:
    """Headline comparison baselines: the tuned feedback controllers and every direct SUMO-PPO budget."""
    return ["PI-ALINEA", "ALINEA", *dict.fromkeys(l for l in labels if l.startswith("SUMO-PPO"))]


def headline(episode_rows: dict, sets: tuple[str, ...]) -> dict:
    out = {}
    for ref in _references(label for label, _ in episode_rows):
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


def _pairs(rows_a: list[dict], rows_b: list[dict]) -> np.ndarray:
    b = {_key(r): float(r["tts_veh_h"]) for r in rows_b}
    return np.array([(float(r["tts_veh_h"]), b[_key(r)]) for r in rows_a if _key(r) in b])


def seed_reduction(pairs_by_seed: list[np.ndarray], n_boot: int = 5000, seed: int = 0) -> dict:
    """1 - mean(TTS_a) / mean(TTS_b) over every seed's paired episodes; 95 % CI from a hierarchical bootstrap
    (resample policy seeds, then episodes within each resampled seed)."""
    if not pairs_by_seed or any(len(p) == 0 for p in pairs_by_seed):
        raise ValueError("every seed needs common episodes")
    allp = np.concatenate(pairs_by_seed)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, len(pairs_by_seed), len(pairs_by_seed))
        sample = np.concatenate([pairs_by_seed[j][rng.integers(0, len(pairs_by_seed[j]), len(pairs_by_seed[j]))]
                                 for j in chosen])
        boot[i] = 1.0 - sample[:, 0].mean() / sample[:, 1].mean()
    return {"n_seeds": len(pairs_by_seed), "n_pairs": int(len(allp)),
            "reduction_pct": 100 * (1.0 - allp[:, 0].mean() / allp[:, 1].mean()),
            "ci95_pct": [100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5))],
            "per_seed_pct": [100 * float(1.0 - p[:, 0].mean() / p[:, 1].mean()) for p in pairs_by_seed],
            "tts_saved_veh_h": float((allp[:, 1] - allp[:, 0]).mean())}


def _mean_sd(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "per_seed": [float(x) for x in v]}


def seed_summary(per_seed: dict, sets: tuple[str, ...]) -> dict:
    """per_seed: {seed: (table1 or None, table2, episode_rows)} -> means over seeds and seed-level headline CIs."""
    seeds = sorted(per_seed)
    t1s = [per_seed[s][0] for s in seeds if per_seed[s][0]]
    t1 = None
    if len(t1s) == len(seeds):
        t1 = {split: {key: _mean_sd([t["splits"][split][key] for t in t1s]) for _, key, _ in TABLE1_METRICS}
              for split in ("val", "test")}
        for split in ("val", "test"):
            t1[split]["n"] = [t["splits"][split]["n"] for t in t1s]
    rows = {}
    for label in per_seed[seeds[0]][1]["rows"]:
        if not all(label in per_seed[s][1]["rows"] for s in seeds):
            continue
        seed_rows = [per_seed[s][1]["rows"][label] for s in seeds]
        r = {"sumo_episodes": _mean_sd([x["sumo_episodes"] for x in seed_rows]),
             "compute_h": _mean_sd([x["compute_h"] for x in seed_rows])}
        for set_name in sets:
            r[set_name] = {m: _mean_sd([x[set_name][m] for x in seed_rows])
                           for m in ("tts", "mean_queue", "max_queue", "completed_trips", "breakdown_rate")}
        rows[label] = r
    head = {}
    for ref in _references(rows):
        for set_name in sets:
            a, b = ("Surrogate-PPO", set_name), (ref, set_name)
            if all(a in per_seed[s][2] and b in per_seed[s][2] for s in seeds):
                head[f"Surrogate-PPO vs {ref} ({set_name})"] = seed_reduction(
                    [_pairs(per_seed[s][2][a], per_seed[s][2][b]) for s in seeds])
    return {"seeds": seeds, "table1": t1, "table2": rows, "headline": head}


def _pm(x: dict, fmt: str = ".2f", sep: str = " ± ") -> str:
    varies = x["sd"] > 1e-9 * max(1.0, abs(x["mean"]))      # identical seeds give float-noise sd, not a spread
    return f"{x['mean']:{fmt}}" + (f"{sep}{x['sd']:{fmt}}" if varies else "")


def _markdown_seeds(summary: dict, sets: tuple[str, ...]) -> str:
    seeds = summary["seeds"]
    lines = [f"# Tables over policy seeds {', '.join(map(str, seeds))}", "",
             "Mean ± standard deviation over seeds (no ± = the same in every seed, e.g. the tuned baselines). "
             "Per-seed tables: `seed_<s>/tables.md`.", ""]
    if summary["table1"]:
        t1 = summary["table1"]
        lines += ["## Table I: DeepONet prediction accuracy (each seed's final ensemble)", "",
                  "| Metric | Validation | Test |", "|---|---|---|"]
        for label, key, _ in TABLE1_METRICS:
            lines.append(f"| {label} | {_pm(t1['val'][key])} | {_pm(t1['test'][key])} |")
        lines += [f"| trajectories | {'/'.join(map(str, t1['val']['n']))} | {'/'.join(map(str, t1['test']['n']))} |", ""]
    set_label = {"test": "ID", "ood": "OOD"}
    cols = " | ".join(f"{set_label.get(s, s)} TTS | {set_label.get(s, s)} mean q | {set_label.get(s, s)} max q" for s in sets)
    lines += ["## Table II: controller comparison", "", f"| Method | {cols} | SUMO episodes | compute (h) |",
              "|---|" + "---|" * (3 * len(sets) + 2)]
    for label, r in summary["table2"].items():
        vals = " | ".join(f"{_pm(r[s]['tts'])} | {_pm(r[s]['mean_queue'])} | {_pm(r[s]['max_queue'])}" for s in sets)
        lines.append(f"| {label} | {vals} | {_pm(r['sumo_episodes'], '.0f')} | {_pm(r['compute_h'], '.1f')} |")
    header = " | ".join(f"{set_label.get(s, s)} seed {sd}" for s in sets for sd in seeds)
    lines += ["", "TTS in veh h; queues in vehicles (1-s samples); compute = summed compute hours.", "",
              "### TTS per seed (veh h)", "", f"| Method | {header} |", "|---|" + "---|" * (len(sets) * len(seeds))]
    for label, r in summary["table2"].items():
        if any(r[s]["tts"]["sd"] > 1e-9 * max(1.0, abs(r[s]["tts"]["mean"])) for s in sets):
            lines.append(f"| {label} | " + " | ".join(f"{v:.2f}" for s in sets for v in r[s]["tts"]["per_seed"]) + " |")
    lines += ["", "## Headline TTS reductions over all seeds", "",
              "95 % CI: hierarchical bootstrap (policy seeds, then episodes within seeds).", ""]
    for name, h in summary["headline"].items():
        per = " / ".join(f"{v:.1f}" for v in h["per_seed_pct"])
        lines.append(f"- {name}: {h['reduction_pct']:.1f} % [{h['ci95_pct'][0]:.1f}, {h['ci95_pct'][1]:.1f}]; "
                     f"per seed {per} %; {h['tts_saved_veh_h']:.2f} veh h saved per episode; "
                     f"{h['n_seeds']} seeds, {h['n_pairs']} episode pairs")
    return "\n".join(lines) + "\n"


def _latex_rows_seeds(summary: dict, sets: tuple[str, ...]) -> str:
    out = []
    for label, r in summary["table2"].items():
        cells = [f"{_pm(r[s]['tts'], sep=' $\\pm$ ')} & {r[s]['mean_queue']['mean']:.2f} & {r[s]['max_queue']['mean']:.2f}"
                 for s in sets]
        out.append(f"{label} & " + " & ".join(cells) + f" & {r['sumo_episodes']['mean']:.0f} & {r['compute_h']['mean']:.1f} \\\\")
    return "\n".join(out) + "\n"


COMPUTE_DEFINITION = ("summed compute: SUMO episode wall times (ledgers) + DeepONet member training "
                      "+ surrogate PPO process time; direct-PPO gradient updates excluded")


def _one_seed(manifest: dict, seed: int, ensemble, sets: tuple[str, ...], arms: str, out: Path) -> tuple:
    t1 = table1(_path(ensemble)) if ensemble else None
    t2, episode_rows = table2(manifest, seed, sets)
    head = headline(episode_rows, sets)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "arms": arms, "table1": t1, "table2": t2, "headline": head,
               "compute_definition": COMPUTE_DEFINITION}
    (out / "tables.json").write_text(json.dumps(payload, indent=1))
    (out / "tables.md").write_text(_markdown(t1, t2, head, sets), encoding="utf-8")
    (out / "table2_rows.tex").write_text(_latex_rows(t2, sets), encoding="utf-8")
    return t1, t2, episode_rows, head


def _final_ensemble(study: str, seed: int):
    rounds_path = PROJECT_ROOT / f"runs/aggregation/{study}_s{seed}/rounds.json"
    return json.loads(rounds_path.read_text())[-1]["ensemble_after"] if rounds_path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="runs/study/m14/arms.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="several policy seeds: per-seed tables in <out>/seed_<s>/, the summary over seeds in <out>/tables.md")
    ap.add_argument("--study", default="m14", help="aggregation study prefix (runs/aggregation/<study>_s<seed>)")
    ap.add_argument("--ensemble", default=None, help="Table I ensemble (default: the last aggregation round's; one seed only)")
    ap.add_argument("--sets", nargs="+", default=["test", "ood"])
    ap.add_argument("--out", default="runs/study/m14/tables")
    args = ap.parse_args()
    sets = tuple(args.sets)
    manifest = json.loads(_path(args.arms).read_text())
    out = _path(args.out)
    seeds = sorted(set(args.seeds)) if args.seeds else [args.seed]
    if len(seeds) == 1:
        ensemble = args.ensemble or _final_ensemble(args.study, seeds[0])
        t1, t2, _, head = _one_seed(manifest, seeds[0], ensemble, sets, str(args.arms), out)
        print(_markdown(t1, t2, head, sets))
        print(f"written: {out / 'tables.json'}, tables.md, table2_rows.tex")
        return
    if args.ensemble:
        ap.error("--ensemble applies to one seed; with --seeds each seed uses its own final ensemble")
    per_seed = {}
    for seed in seeds:
        t1, t2, episode_rows, _ = _one_seed(manifest, seed, _final_ensemble(args.study, seed), sets, str(args.arms),
                                            out / f"seed_{seed}")
        per_seed[seed] = (t1, t2, episode_rows)
    summary = seed_summary(per_seed, sets)
    payload = {"arms": str(args.arms), **summary, "compute_definition": COMPUTE_DEFINITION}
    (out / "tables.json").write_text(json.dumps(payload, indent=1))
    (out / "tables.md").write_text(_markdown_seeds(summary, sets), encoding="utf-8")
    (out / "table2_rows.tex").write_text(_latex_rows_seeds(summary, sets), encoding="utf-8")
    print(_markdown_seeds(summary, sets))
    print(f"written: {out / 'tables.md'} (summary over seeds {seeds}) and {out}/seed_<s>/")


if __name__ == "__main__":
    main()
