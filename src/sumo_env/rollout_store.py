"""
Rollout store for the plant-model surrogate (M8/M9).

A store is a directory of rollout npz files (written by
sumo_env.rollout.save_rollout_npz) plus `index.json` (one entry per file with
its round, controller type, peak total and split) and `split_index.json`
(train / val / test file lists + train-split normalisation statistics), the
format read by surrogate.datasets.PlantRolloutDataset.

Round-0 files are split 70 / 15 / 15 stratified by (controller type, peak-total
tertile); aggregation rounds (`append_round`) go entirely to the training
split, labelled with their round number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sumo_env.rollout import load_rollout_npz


class RolloutStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.split_path = self.root / "split_index.json"
        self.entries: list[dict] = []
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                self.entries = json.load(f)["entries"]

    # -- registration ------------------------------------------------------
    def add(self, npz_path: str | Path, meta: dict, split: str | None = None) -> dict:
        p = Path(npz_path)
        rel = p.name if p.parent == self.root else str(p)
        metrics = meta.get("metrics", {})
        entry = {
            "file": rel,
            "round": int(meta.get("round", 0)),
            "controller_type": str(meta.get("controller", {}).get("type", "unknown")),
            "controller": meta.get("controller", {}),
            "profile_set": str(meta.get("profile", {}).get("set", "train")),
            "profile_index": int(meta.get("profile", {}).get("index", -1)),
            "peak_total_vph": float(meta.get("profile", {}).get("peak_total_vph", 0.0)),
            "sumo_seed": int(meta.get("sumo_seed", -1)),
            "return": float(metrics.get("return", 0.0)),
            "breakdown": bool(metrics.get("breakdown", False)),
            "split": split,
        }
        self.entries = [e for e in self.entries if e["file"] != rel] + [entry]
        return entry

    def save_index(self) -> None:
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump({"entries": self.entries}, f, indent=1)

    def files(self, split: str | None = None, rounds=None) -> list[str]:
        out = []
        for e in self.entries:
            if split is not None and e.get("split") != split:
                continue
            if rounds is not None and e["round"] not in rounds:
                continue
            out.append(e["file"])
        return out

    def __len__(self) -> int:
        return len(self.entries)

    # -- splits ------------------------------------------------------------
    def make_splits(self, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 0) -> dict:
        """Stratified split of the round-0 entries; later rounds stay train."""
        rng = np.random.default_rng(seed)
        round0 = [e for e in self.entries if e["round"] == 0]
        peaks = np.array([e["peak_total_vph"] for e in round0]) if round0 else np.zeros(0)
        edges = np.quantile(peaks, [1 / 3, 2 / 3]) if len(peaks) >= 3 else np.array([0.0, 0.0])
        strata: dict[tuple, list[dict]] = {}
        for e in round0:
            key = (e["controller_type"], int(np.searchsorted(edges, e["peak_total_vph"])))
            strata.setdefault(key, []).append(e)
        for key, group in strata.items():
            idx = rng.permutation(len(group))
            n = len(group)
            n_train = int(round(n * train_frac))
            n_val = int(round(n * val_frac))
            if n >= 3:
                n_train = max(1, min(n_train, n - 2)); n_val = max(1, min(n_val, n - n_train - 1))
            elif n == 2:
                n_train, n_val = 1, 1
            else:
                n_train, n_val = 1, 0
            for j, i in enumerate(idx):
                if j < n_train:
                    group[i]["split"] = "train"
                elif j < n_train + n_val:
                    group[i]["split"] = "val"
                else:
                    group[i]["split"] = "test"
        for e in self.entries:
            if e["round"] != 0:
                e["split"] = "train"
        self.save_index()
        return self.write_split_index()

    def append_round(self, files_and_meta: list[tuple[str | Path, dict]]) -> dict:
        for path, meta in files_and_meta:
            self.add(path, meta, split="train")
        self.save_index()
        return self.write_split_index()

    def write_split_index(self) -> dict:
        splits = {s: self.files(split=s) for s in ("train", "val", "test")}
        stats = self.density_stats(splits["train"])
        split_index = {**splits, "metadata": {**stats, "n_train": len(splits["train"]),
                                                 "n_val": len(splits["val"]), "n_test": len(splits["test"]),
                                                 "rounds": sorted({e["round"] for e in self.entries})}}
        with self.split_path.open("w", encoding="utf-8") as f:
            json.dump(split_index, f, indent=1)
        with (self.root / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(split_index["metadata"], f, indent=2)
        return split_index

    def density_stats(self, files: list[str]) -> dict:
        if not files:
            return {"mean_density": 0.0, "std_density": 1.0, "mean_outflow_vph": 0.0}
        dens, outs = [], []
        for fn in files:
            arrays, _ = load_rollout_npz(self.root / fn)
            dens.append(arrays["density"].astype(np.float32).ravel())
            outs.append(arrays["outflow_vph"].astype(np.float32).ravel())
        d = np.concatenate(dens); o = np.concatenate(outs)
        return {"mean_density": float(d.mean()), "std_density": float(d.std()),
                "mean_outflow_vph": float(o.mean())}

    def fork(self, new_root: str | Path) -> "RolloutStore":
        """New store that references this store's files (absolute paths) so an
        aggregation study can append rounds without touching the shared
        round-0 store."""
        new = RolloutStore(new_root)
        new.entries = []
        for e in self.entries:
            ent = dict(e)
            f = Path(e["file"])
            ent["file"] = str(f if f.is_absolute() else (self.root / f).resolve())
            new.entries.append(ent)
        new.save_index()
        new.write_split_index()
        return new

    def summary(self) -> dict:
        by_type: dict[str, int] = {}
        by_round: dict[int, int] = {}
        for e in self.entries:
            by_type[e["controller_type"]] = by_type.get(e["controller_type"], 0) + 1
            by_round[e["round"]] = by_round.get(e["round"], 0) + 1
        return {"n": len(self.entries), "by_controller": by_type, "by_round": by_round,
                "breakdown_rate": float(np.mean([e["breakdown"] for e in self.entries])) if self.entries else 0.0,
                "splits": {s: len(self.files(split=s)) for s in ("train", "val", "test")}}
