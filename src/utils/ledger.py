"""
SUMO episode-equivalent (EE) ledger (draft_pipeline.md §2, Appendix C).

One JSON line per SUMO rollout under runs/ledger/<study>.jsonl:

    {"study", "round", "purpose", "profile_set", "profile_id", "sumo_seed",
     "policy", "return", "breakdown", "wall_s", "ts"}

Purposes: dataset, aggregation, finetune, direct_ppo, tuning, eval_val,
eval_test, eval_ood. `budget()` sums every line except eval_test / eval_ood.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PURPOSES = ("dataset", "aggregation", "finetune", "direct_ppo", "tuning", "eval_val", "eval_test", "eval_ood")
UNCOUNTED = ("eval_test", "eval_ood")


class Ledger:
    def __init__(self, study: str, project_root: str | Path | None = None, ledger_dir: str | Path | None = None) -> None:
        root = Path(project_root or Path.cwd())
        self.study = str(study)
        self.dir = Path(ledger_dir) if ledger_dir else root / "runs" / "ledger"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.study}.jsonl"

    def log(self, round_index: int, purpose: str, profile_set: str, profile_id, sumo_seed: int,
            policy: str, ret: float, breakdown: bool, wall_s: float, **extra) -> dict:
        if purpose not in PURPOSES:
            raise ValueError(f"unknown ledger purpose {purpose!r}; expected one of {PURPOSES}")
        line = {"study": self.study, "round": int(round_index), "purpose": purpose,
                "profile_set": str(profile_set), "profile_id": profile_id, "sumo_seed": int(sumo_seed),
                "policy": str(policy), "return": float(ret), "breakdown": bool(breakdown),
                "wall_s": float(wall_s), "ts": time.time(), **extra}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        return line

    def log_many(self, rows: list[dict]) -> None:
        for r in rows:
            self.log(**r)

    def lines(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def budget(self, upto_round: int | None = None) -> int:
        n = 0
        for l in self.lines():
            if l["purpose"] in UNCOUNTED:
                continue
            if upto_round is not None and l["round"] > upto_round:
                continue
            n += 1
        return n

    def summary(self) -> dict:
        by = {}
        for l in self.lines():
            key = (l["purpose"], l["round"])
            by[key] = by.get(key, 0) + 1
        return {"study": self.study, "budget_ee": self.budget(), "total_lines": len(self.lines()),
                "by_purpose_round": {f"{p}@r{r}": n for (p, r), n in sorted(by.items())}}


def read_ledgers(ledger_dir: str | Path) -> dict[str, list[dict]]:
    out = {}
    for p in sorted(Path(ledger_dir).glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            out[p.stem] = [json.loads(l) for l in f if l.strip()]
    return out
