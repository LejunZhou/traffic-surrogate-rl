"""
Time-varying demand profiles (M8, draft_pipeline.md §4).

A *profile* is a pair of piecewise-constant rate functions on 5-minute
blocks: mainline demand d_k and ramp arrival rate r_k, expanded to the
K = 120 control steps of an episode. Profiles are drawn from a parametric
family (configs/profiles/family_v1.yaml) by a sampler that is seeded by
(set name, index), so every method that asks for "validation profile 7"
gets byte-identical inputs.

Frozen sets (configs/profiles/{val,test,ood}.json) are produced once by
`build_fixed_sets` and loaded with `load_profile_set`; the JSON stores the
12 block values, the family parameters and the SUMO seeds of each profile.

Shared by the dataset generator, SumoEnv and SurrogateVecEnv.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SET_IDS = {"train": 0, "val": 1, "test": 2, "ood": 3, "e0": 4, "tune": 5, "agg": 6}
DEFAULT_SUMO_SEEDS = {"val": [10000], "test": [100, 101, 102], "ood": [100, 101, 102]}


@dataclass
class DemandProfile:
    """One episode's demand: block rates plus their expansion to control steps."""

    mainline_blocks: np.ndarray          # (n_blocks,) veh/h
    ramp_blocks: np.ndarray              # (n_blocks,) veh/h
    block_min: float = 5.0
    dt_ctrl_s: float = 30.0
    params: dict = field(default_factory=dict)
    set_name: str = "train"
    index: int = -1
    sumo_seeds: list[int] = field(default_factory=list)

    # -- expansion -------------------------------------------------------
    @property
    def steps_per_block(self) -> int:
        return int(round(self.block_min * 60.0 / self.dt_ctrl_s))

    @property
    def K(self) -> int:
        return int(len(self.mainline_blocks) * self.steps_per_block)

    @property
    def mainline_vph(self) -> np.ndarray:
        return np.repeat(self.mainline_blocks.astype(np.float32), self.steps_per_block)

    @property
    def ramp_vph(self) -> np.ndarray:
        return np.repeat(self.ramp_blocks.astype(np.float32), self.steps_per_block)

    def mainline_flow_blocks(self) -> list[tuple[float, float, float]]:
        """(begin_s, end_s, veh/h) per block, for the SUMO route writer."""
        block_s = self.block_min * 60.0
        return [
            (i * block_s, (i + 1) * block_s, float(v))
            for i, v in enumerate(self.mainline_blocks)
        ]

    # -- descriptors used for stratification and reports ------------------
    @property
    def peak_total_vph(self) -> float:
        return float(np.max(self.mainline_blocks + self.ramp_blocks))

    @property
    def peak_mainline_vph(self) -> float:
        return float(np.max(self.mainline_blocks))

    @property
    def is_peaked(self) -> bool:
        """True when the mainline demand is not (near) constant."""
        return float(np.ptp(self.mainline_blocks)) > 150.0

    @property
    def label(self) -> str:
        m = self.params.get("mainline", {}).get("family", "?")
        r = self.params.get("ramp", {}).get("family", "?")
        return f"{self.set_name}[{self.index}] {m}/{r} peak {self.peak_total_vph:.0f}"

    # -- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "set": self.set_name,
            "index": int(self.index),
            "sumo_seeds": [int(s) for s in self.sumo_seeds],
            "block_min": float(self.block_min),
            "dt_ctrl_s": float(self.dt_ctrl_s),
            "mainline_vph_blocks": [float(v) for v in self.mainline_blocks],
            "ramp_vph_blocks": [float(v) for v in self.ramp_blocks],
            "params": _jsonable(self.params),
            "peak_total_vph": self.peak_total_vph,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DemandProfile":
        return cls(
            mainline_blocks=np.asarray(d["mainline_vph_blocks"], dtype=np.float32),
            ramp_blocks=np.asarray(d["ramp_vph_blocks"], dtype=np.float32),
            block_min=float(d.get("block_min", 5.0)),
            dt_ctrl_s=float(d.get("dt_ctrl_s", 30.0)),
            params=dict(d.get("params", {})),
            set_name=str(d.get("set", "train")),
            index=int(d.get("index", -1)),
            sumo_seeds=[int(s) for s in d.get("sumo_seeds", [])],
        )

    @classmethod
    def constant(
        cls, mainline_vph: float, ramp_vph: float, n_blocks: int = 12,
        block_min: float = 5.0, dt_ctrl_s: float = 30.0, **kw,
    ) -> "DemandProfile":
        """Constant profile (the M7 grid cells are a special case of the family)."""
        return cls(
            mainline_blocks=np.full(n_blocks, float(mainline_vph), dtype=np.float32),
            ramp_blocks=np.full(n_blocks, float(ramp_vph), dtype=np.float32),
            block_min=block_min, dt_ctrl_s=dt_ctrl_s,
            params={"mainline": {"family": "const", "d": float(mainline_vph)},
                    "ramp": {"family": "const", "r": float(ramp_vph)}},
            **kw,
        )


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _u(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return float(rng.uniform(lo, hi))


def _bump(s: np.ndarray) -> np.ndarray:
    """Raised-cosine bump on |s| < 1, zero outside."""
    out = np.zeros_like(s)
    inside = np.abs(s) < 1.0
    out[inside] = 0.5 * (1.0 + np.cos(np.pi * s[inside]))
    return out


class ProfileFamily:
    """Seeded sampler for the parametric demand-profile family."""

    def __init__(self, config: dict | str | Path, dt_ctrl_s: float = 30.0) -> None:
        self.source = None
        if not isinstance(config, dict):
            self.source = str(config)
            with Path(config).open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        self.cfg = copy.deepcopy(config)
        self.version = int(self.cfg.get("version", 1))
        self.block_min = float(self.cfg.get("block_min", 5))
        self.horizon_min = float(self.cfg.get("horizon_min", 60))
        self.n_blocks = int(round(self.horizon_min / self.block_min))
        self.dt_ctrl_s = float(dt_ctrl_s)
        # fine time grid (1 s) for block averaging of the continuous shapes
        self._t_fine = (np.arange(int(self.horizon_min * 60)) + 0.5) / 60.0  # minutes

    @classmethod
    def load(cls, path: str | Path, dt_ctrl_s: float = 30.0) -> "ProfileFamily":
        with Path(path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        fam = cls(cfg, dt_ctrl_s=dt_ctrl_s)
        fam.source = str(path)
        return fam

    # -- seeding ------------------------------------------------------------
    def rng_for(self, set_name: str, index: int) -> np.random.Generator:
        set_id = SET_IDS.get(set_name)
        if set_id is None:  # stable (not hash()-based) fallback for ad-hoc set names
            set_id = 100 + (sum(ord(c) * (i + 1) for i, c in enumerate(set_name)) % 1000)
        return np.random.default_rng(np.random.SeedSequence([self.version, int(set_id), int(index)]))

    # -- sampling -----------------------------------------------------------
    def sample(self, rng: np.random.Generator, set_name: str = "train", index: int = -1) -> DemandProfile:
        m_family = self._choose(rng, self.cfg["mainline"]["families"])
        r_family = self._choose(rng, self.cfg["ramp"]["families"])
        d_fine, m_params = self._sample_mainline(rng, m_family)
        r_fine, r_params = self._sample_ramp(rng, r_family, m_params)
        return self._finish(d_fine, r_fine, {"mainline": m_params, "ramp": r_params}, set_name, index)

    def sample_by_key(self, set_name: str, index: int) -> DemandProfile:
        return self.sample(self.rng_for(set_name, index), set_name, index)

    def sample_ood(self, rng: np.random.Generator, kind: str, set_name: str = "ood", index: int = -1) -> DemandProfile:
        ood = self.cfg["ood"]
        if kind == "double":
            c = ood["double"]
            d_base = _u(rng, c["d_base"])
            t1 = _u(rng, c["t_center_min"][0]); t2 = _u(rng, c["t_center_min"][1])
            w = _u(rng, c["half_width_min"])
            a1 = _u(rng, c["amplitude"]); a2 = _u(rng, c["amplitude"])
            d = d_base + a1 * _bump((self._t_fine - t1) / w) + a2 * _bump((self._t_fine - t2) / w)
            d = np.minimum(d, float(c["d_max"]))
            m_params = {"family": "double", "d_base": d_base, "amplitudes": [a1, a2],
                        "t_centers": [t1, t2], "half_width": w, "t_center": t1}
            r, r_params = self._sample_ramp(rng, "surge", m_params)
        elif kind == "plateau":
            c = ood["plateau"]
            d_base = _u(rng, c["d_base"]); t1 = _u(rng, c["t1_min"])
            d_high = float(c["d_high"]); L = float(c["length_min"])
            d = np.where((self._t_fine >= t1) & (self._t_fine < t1 + L), d_high, d_base)
            m_params = {"family": "plateau", "d_base": d_base, "d_high": d_high, "t1": t1,
                        "length": L, "t_center": t1 + L / 2}
            r, r_params = self._sample_ramp(rng, "surge", m_params)
        elif kind == "early_surge":
            c = ood["early_surge"]
            d, m_params = self._sample_mainline(rng, "peak")
            lead = _u(rng, c["lead_min"]); L = _u(rng, c["length_min"])
            t_r = max(0.0, float(m_params["t_center"]) - lead)
            r_base = _u(rng, c["r_base"]); A = float(c["amplitude_r"])
            r = r_base + A * ((self._t_fine >= t_r) & (self._t_fine < t_r + L))
            if "r_max" in c:                                      # family v2: never above the meter's maximum
                r = np.minimum(r, float(c["r_max"]))
            r_params = {"family": "early_surge", "r_base": r_base, "amplitude": A, "t_start": t_r, "length": L}
        else:
            raise ValueError(f"unknown OOD family {kind!r}")
        return self._finish(d, r, {"mainline": m_params, "ramp": r_params, "ood": kind}, set_name, index)

    # -- family shapes ------------------------------------------------------
    def _sample_mainline(self, rng, family: str):
        m = self.cfg["mainline"]
        t = self._t_fine
        if family == "peak":
            c = m["peak"]
            d_base = _u(rng, c["d_base"]); A = _u(rng, c["amplitude"])
            w = _u(rng, c["half_width_min"]); t_c = _u(rng, c["t_center_min"])
            t_c = min(t_c, float(c["peak_end_max_min"]) - w)      # peak over by minute 45
            d = np.minimum(d_base + A * _bump((t - t_c) / w), float(c["d_max"]))
            return d, {"family": "peak", "d_base": d_base, "amplitude": A, "t_center": t_c, "half_width": w}
        if family == "step":
            c = m["step"]
            d_base = _u(rng, c["d_base"]); d_high = _u(rng, c["d_high"])
            t1 = _u(rng, c["t1_min"]); ramp = float(c["ramp_min"])
            step_down = bool(rng.uniform() < float(c["p_step_down"]))
            if step_down:
                t2 = t1 + _u(rng, c["t2_offset_min"])
                if "step_down_end_max_min" in c:                  # family v2: the 10-min fall is over by this minute
                    t2 = max(t1 + ramp, min(t2, float(c["step_down_end_max_min"]) - ramp))
            else:
                t2 = max(t1 + ramp, float(c["plateau_end_max_min"]) - ramp)
            up = np.clip((t - t1) / ramp, 0.0, 1.0)
            down = np.clip((t - t2) / ramp, 0.0, 1.0)
            d = d_base + (d_high - d_base) * (up - down)
            return d, {"family": "step", "d_base": d_base, "d_high": d_high, "t1": t1, "t2": t2,
                       "step_down": step_down, "t_center": 0.5 * (t1 + t2)}
        if family == "const":
            c = m["const"]
            d0 = _u(rng, c["d"])
            return np.full_like(t, d0), {"family": "const", "d": d0, "t_center": 30.0}
        raise ValueError(f"unknown mainline family {family!r}")

    def _sample_ramp(self, rng, family: str, m_params: dict):
        r_cfg = self.cfg["ramp"]
        t = self._t_fine
        if family == "surge":
            c = r_cfg["surge"]
            r_base = _u(rng, c["r_base"]); A = _u(rng, c["amplitude"])
            t_r = _u(rng, c["t_start_min"]); L = _u(rng, c["length_min"])
            if "end_max_min" in c:                                # family v2: the surge is over by this minute
                t_r = max(0.0, min(t_r, float(c["end_max_min"]) - L))
            r = np.minimum(r_base + A * ((t >= t_r) & (t < t_r + L)), float(c["r_max"]))
            return r, {"family": "surge", "r_base": r_base, "amplitude": A, "t_start": t_r, "length": L}
        if family == "const":
            r0 = _u(rng, r_cfg["const"]["r"])
            return np.full_like(t, r0), {"family": "const", "r": r0}
        raise ValueError(f"unknown ramp family {family!r}")

    def _finish(self, d_fine, r_fine, params, set_name, index) -> DemandProfile:
        n_per_block = int(self.block_min * 60)
        d_blocks = d_fine.reshape(self.n_blocks, n_per_block).mean(axis=1)
        r_blocks = r_fine.reshape(self.n_blocks, n_per_block).mean(axis=1)
        return DemandProfile(
            mainline_blocks=np.round(d_blocks, 1).astype(np.float32),
            ramp_blocks=np.round(r_blocks, 1).astype(np.float32),
            block_min=self.block_min, dt_ctrl_s=self.dt_ctrl_s,
            params=params, set_name=set_name, index=int(index),
        )

    @staticmethod
    def _choose(rng: np.random.Generator, weights: dict) -> str:
        names = list(weights)
        p = np.asarray([float(weights[n]) for n in names]); p = p / p.sum()
        return str(names[int(rng.choice(len(names), p=p))])


# ── frozen sets ─────────────────────────────────────────────────────────────

def surge_overlaps_peak(p: DemandProfile) -> bool:
    """Does the ramp surge window overlap the mainline peak window?"""
    rp = p.params.get("ramp", {}); mp = p.params.get("mainline", {})
    if rp.get("family") not in ("surge", "early_surge"):
        return False
    t_r0, t_r1 = float(rp["t_start"]), float(rp["t_start"]) + float(rp["length"])
    fam = mp.get("family")
    if fam == "peak" or fam == "double":
        w = float(mp["half_width"]); tc = float(mp["t_center"])
        m0, m1 = tc - w, tc + w
    elif fam in ("step", "plateau"):
        m0 = float(mp["t1"]); m1 = float(mp.get("t2", m0 + float(mp.get("length", 30))))
    else:
        return False
    return (t_r0 < m1) and (t_r1 > m0)


def build_validation_set(family: ProfileFamily, n: int = 18, base_seed: int = 10000) -> list[DemandProfile]:
    """18 profiles stratified over peak-total tertile x surge-overlaps-peak (6 cells x 3).

    Candidates are drawn from the family with the 'val' seed stream; the
    tertile edges come from a 2000-profile sample of the training family.
    """
    ref = [family.sample_by_key("train", i) for i in range(2000)]
    edges = np.quantile([p.peak_total_vph for p in ref], [1 / 3, 2 / 3])
    per_cell = max(1, n // 6)
    cells: dict[tuple[int, bool], list[DemandProfile]] = {}
    idx = 0
    chosen: list[DemandProfile] = []
    while len(chosen) < n and idx < 20000:
        p = family.sample_by_key("val", idx); idx += 1
        tertile = int(np.searchsorted(edges, p.peak_total_vph))
        key = (tertile, surge_overlaps_peak(p))
        bucket = cells.setdefault(key, [])
        if len(bucket) < per_cell:
            bucket.append(p); chosen.append(p)
    chosen.sort(key=lambda p: (p.peak_total_vph))
    out = []
    for i, p in enumerate(chosen):
        out.append(DemandProfile(p.mainline_blocks, p.ramp_blocks, p.block_min, p.dt_ctrl_s,
                                 dict(p.params, stratum={"peak_tertile": int(np.searchsorted(edges, p.peak_total_vph)),
                                                         "surge_overlaps_peak": surge_overlaps_peak(p),
                                                         "val_draw_index": p.index}),
                                 "val", i, [base_seed + i]))
    return out


def build_test_set(family: ProfileFamily, n: int = 30, seeds=(100, 101, 102)) -> list[DemandProfile]:
    out = []
    for i in range(n):
        p = family.sample_by_key("test", i)
        p.sumo_seeds = list(seeds)
        out.append(p)
    return out


def build_ood_set(family: ProfileFamily, n: int = 12, seeds=(100, 101, 102)) -> list[DemandProfile]:
    kinds = ["double", "plateau", "early_surge"]
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        p = family.sample_ood(family.rng_for("ood", i), kind, "ood", i)
        p.sumo_seeds = list(seeds)
        out.append(p)
    return out


def save_profile_set(profiles: list[DemandProfile], path: str | Path, family_path: str, set_name: str) -> None:
    payload = {
        "set": set_name,
        "family": str(family_path),
        "n": len(profiles),
        "profiles": [p.to_dict() for p in profiles],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def load_profile_set(path: str | Path) -> list[DemandProfile]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [DemandProfile.from_dict(d) for d in payload["profiles"]]


def build_fixed_sets(family_path: str | Path, out_dir: str | Path, dt_ctrl_s: float = 30.0) -> dict[str, list[DemandProfile]]:
    family = ProfileFamily.load(family_path, dt_ctrl_s=dt_ctrl_s)
    sets = {
        "val": build_validation_set(family),
        "test": build_test_set(family),
        "ood": build_ood_set(family),
    }
    for name, profiles in sets.items():
        save_profile_set(profiles, Path(out_dir) / f"{name}.json", str(family_path), name)
    return sets


def resolve_profile_source(spec, project_root: Path | None = None, dt_ctrl_s: float = 30.0):
    """Turn an env-config `profiles` value into (family | None, fixed list | None).

    Accepted forms:
      - path to a family YAML         -> resample from the family every episode
      - path to a frozen set JSON     -> cycle / index the fixed set
      - {"family": path, "set": path} -> both (set used when an index is requested)
    """
    root = Path(project_root or Path.cwd())

    def _p(x):
        p = Path(x)
        return p if p.is_absolute() else root / p

    family = None
    fixed = None
    if isinstance(spec, dict):
        if spec.get("family"):
            family = ProfileFamily.load(_p(spec["family"]), dt_ctrl_s=dt_ctrl_s)
        if spec.get("set"):
            fixed = load_profile_set(_p(spec["set"]))
    elif spec is not None:
        p = _p(spec)
        if p.suffix.lower() == ".json":
            fixed = load_profile_set(p)
        else:
            family = ProfileFamily.load(p, dt_ctrl_s=dt_ctrl_s)
    return family, fixed


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Freeze the V / T / O profile sets")
    ap.add_argument("--family", default="configs/profiles/family_v1.yaml")
    ap.add_argument("--out-dir", default="configs/profiles")
    args = ap.parse_args()
    sets = build_fixed_sets(args.family, args.out_dir)
    for name, ps in sets.items():
        print(f"{name}: {len(ps)} profiles; peak totals "
              f"{min(p.peak_total_vph for p in ps):.0f}-{max(p.peak_total_vph for p in ps):.0f} vph")
