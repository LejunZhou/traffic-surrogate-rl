"""
Trainer for the plant-model DeepONet v2 (M9, draft_pipeline.md §5.4).

One ensemble member per process:

  PYTHONPATH=src python -m surrogate.train_plant --config configs/surrogate/plant_v2.yaml \\
      --member 0 --bootstrap-seed 0 --out-dir runs/surrogate/plant_v2_round0/member_0

Fine-tuning after an aggregation round (every member resumes from its
checkpoint, keeps its bootstrap file list and adds every rollout of the new
rounds, 20 epochs at lr 3e-4):

  ... --resume runs/surrogate/plant_v2_round0/member_0/best.pt --finetune \\
      --out-dir runs/surrogate/plant_v2_round1/member_0

Loss = weighted MSE on z-scored density queries + outflow_weight * MSE on the
exit flow; AdamW, cosine schedule with warm-up, gradient clip; best.pt on the
validation relative L2 (density + exit flow, physical units).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from surrogate.datasets import PlantRolloutDataset, plant_collate
from surrogate.deeponet import PlantNormalisation, build_plant_model
from utils.config import load_config


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _resolve(path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_store_index(store_dir: Path) -> tuple[dict, dict]:
    split = json.loads((store_dir / "split_index.json").read_text())
    index = json.loads((store_dir / "index.json").read_text())["entries"]
    return split, index


def geometry_from_sumo_config(sumo_cfg: dict) -> dict:
    from sumo_env.detectors import get_x_grid

    return {"x_grid_m": get_x_grid(sumo_cfg).tolist(), "highway_length_m": float(sumo_cfg["network"]["highway_length_m"]),
            "duration_s": float(sumo_cfg["simulation"]["duration_s"]), "dt_ctrl_s": float(sumo_cfg["simulation"]["dt_ctrl_s"]),
            "ramp_position_m": float(sumo_cfg["network"]["ramp_position_m"])}


def plant_loss(model, batch, outflow_weight: float, device):
    rho_hat, _ = model(batch["branch"].to(device), batch["query_k"].to(device), batch["query_xt"].to(device))
    w = batch["weight_rho"].to(device)
    loss_rho = (w * (rho_hat - batch["target_rho"].to(device)) ** 2).sum() / w.sum()
    _, q_hat = model(batch["branch"].to(device), batch["exit_k"].to(device), batch["exit_xt"].to(device))
    m = batch["mask_q"].to(device)
    loss_q = (m * (q_hat - batch["target_q"].to(device)) ** 2).sum() / m.sum().clamp_min(1.0)
    return loss_rho + outflow_weight * loss_q, float(loss_rho.detach()), float(loss_q.detach())


@torch.no_grad()
def evaluate_full_grid(model, dataset: PlantRolloutDataset, norm: PlantNormalisation, device, files=None) -> dict:
    """Relative L2 of density and exit flow on the full grid (physical units), per rollout mean."""
    model.eval()
    files = files or dataset.unique_files
    rel_rho, rel_q, mse_rho, abs_q_sum = [], [], [], []
    for fn in files:
        g = dataset.full_grid(fn)
        s = dataset.samples[fn]
        rho_hat, _ = model(g["branch"].to(device), g["query_k"].to(device), g["query_xt"].to(device))
        rho_hat = norm.z_to_density(rho_hat.cpu().numpy().reshape(dataset.Nx, dataset.K))
        rho_hat = np.maximum(rho_hat, 0.0)
        _, q_hat = model(g["branch"].to(device), g["exit_k"].to(device), g["exit_xt"].to(device))
        q_hat = np.maximum(q_hat.cpu().numpy().reshape(-1) * norm.flow_scale, 0.0)
        true = s["density"]; q_true = s["outflow"]
        rel_rho.append(np.linalg.norm(rho_hat - true) / max(np.linalg.norm(true), 1e-6))
        rel_q.append(np.linalg.norm(q_hat - q_true) / max(np.linalg.norm(q_true), 1e-6))
        mse_rho.append(float(np.mean((rho_hat - true) ** 2)))
        abs_q_sum.append(abs(float(q_hat.sum() - q_true.sum())) / max(abs(float(q_true.sum())), 1e-6))
    model.train()
    return {"val_rel_l2_density": float(np.mean(rel_rho)), "val_rel_l2_flow": float(np.mean(rel_q)),
            "val_mse_density": float(np.mean(mse_rho)), "val_outflow_sum_err": float(np.mean(abs_q_sum)),
            "val_rel_l2": float(np.mean(rel_rho) + np.mean(rel_q))}


def cosine_lr(step: int, total: int, warmup: int, base: float, floor: float = 0.02) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


def train_member(config: dict, member: int, bootstrap_seed: int | None, out_dir: Path, resume: Path | None = None,
                 finetune: bool = False, new_rounds: list[int] | None = None, project_root: Path | None = None,
                 epochs_override: int | None = None, max_train_files: int | None = None,
                 train_types: list[str] | None = None) -> dict:
    root = Path(project_root or config.get("project_root", Path.cwd()))
    cfg = copy.deepcopy(config)
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    ens_cfg = cfg.get("ensemble", {})
    torch.set_num_threads(int(train_cfg.get("num_threads", 2)))
    seed = int(train_cfg.get("seed", 0)) + 1000 * int(member)
    _set_seed(seed)
    device = torch.device("cpu" if str(train_cfg.get("device", "auto")) == "auto" and not torch.cuda.is_available()
                          else train_cfg.get("device", "cuda"))
    store_dir = _resolve(data_cfg["store_dir"], root)
    split, index = load_store_index(store_dir)
    sumo_cfg = load_config(str(_resolve(data_cfg["sumo_config"], root)))
    geometry = geometry_from_sumo_config(sumo_cfg)
    K = int(round(geometry["duration_s"] / geometry["dt_ctrl_s"]))
    cfg["data"]["K"] = K
    cfg["data"]["geometry"] = geometry
    n_cfg = data_cfg.get("normalisers", {})

    ckpt = None
    if resume is not None:
        ckpt = torch.load(str(resume), map_location="cpu", weights_only=False)
        norm = PlantNormalisation.from_dict(ckpt["normalization"])
        cfg["model"] = ckpt["config"]["model"]   # architecture is fixed by the checkpoint
    else:
        md = split["metadata"]
        norm = PlantNormalisation(md["mean_density"], md["std_density"], float(n_cfg.get("mainline_demand", 2500.0)),
                                  float(n_cfg.get("ramp_inflow", 1600.0)), float(n_cfg.get("flow", 2500.0)))

    # ---- files: bootstrap resample of the round-0 train split (+ new rounds on fine-tune)
    round_of = {e["file"]: int(e["round"]) for e in index}
    type_of = {e["file"]: str(e.get("controller_type", "")) for e in index}
    train_files_all = list(split["train"])
    round0_train = [f for f in train_files_all if round_of.get(f, 0) == 0]
    if train_types:
        round0_train = [f for f in round0_train if type_of.get(f) in set(train_types)]
    if max_train_files is not None:
        rng_sub = np.random.default_rng(12345)
        round0_train = [round0_train[i] for i in sorted(rng_sub.choice(len(round0_train), size=min(int(max_train_files), len(round0_train)), replace=False))]
    if ckpt is not None and ckpt.get("bootstrap_files"):
        # a forked (per-study) store references round-0 files by absolute path:
        # map the checkpoint's file names onto this store's entries by basename
        name_map = {Path(f).name: f for f in split["train"] + split["val"] + split["test"]}
        base_files = [name_map.get(Path(f).name, f) for f in ckpt["bootstrap_files"]]
    elif bool(ens_cfg.get("bootstrap", True)) and bootstrap_seed is not None:
        rng = np.random.default_rng(int(bootstrap_seed))
        base_files = [round0_train[i] for i in rng.integers(0, len(round0_train), size=len(round0_train))]
    else:
        base_files = list(round0_train)
    if finetune:
        rounds = set(new_rounds) if new_rounds is not None else {r for r in round_of.values() if r > 0}
        extra = [f for f in train_files_all if round_of.get(f, 0) in rounds]
        known = set(base_files)
        train_files = base_files + [f for f in extra if f not in known]
    else:
        train_files = base_files
    val_files = list(split["val"])
    band = data_cfg.get("band", {})
    ds_kwargs = dict(norm=norm, mode=str(data_cfg.get("mode", "causal")),
                     padded_views_per_rollout=int(data_cfg.get("padded_views_per_rollout", 8)),
                     band_rho_min=float(band.get("rho_min", 40.0)), band_x_max_m=float(band.get("x_max_m", 1400.0)),
                     band_weight=float(band.get("weight", 2.0)), highway_length_m=geometry["highway_length_m"],
                     duration_s=geometry["duration_s"])
    train_ds = PlantRolloutDataset(store_dir, train_files, n_query_points=data_cfg.get("n_query_points", 512), seed=seed, **ds_kwargs)
    val_ds = PlantRolloutDataset(store_dir, val_files, n_query_points=None, seed=seed + 1, **{**ds_kwargs, "mode": "causal"})
    loader = DataLoader(train_ds, batch_size=int(train_cfg.get("batch_size", 16)), shuffle=True, collate_fn=plant_collate,
                        num_workers=0, drop_last=False)

    model = build_plant_model(cfg["model"], K=K).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    ft = cfg.get("finetune", {})
    n_epochs = int(epochs_override or (ft.get("epochs", 20) if finetune else train_cfg.get("n_epochs", 150)))
    lr = float(ft.get("lr", 3e-4) if finetune else train_cfg.get("lr", 1e-3))
    warmup = 0 if finetune else int(train_cfg.get("warmup_epochs", 5))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 1e-6)))
    outflow_weight = float(data_cfg.get("outflow_weight", 1.0))
    clip = float(train_cfg.get("grad_clip_norm", 1.0))
    eval_every = int(train_cfg.get("eval_every", 5))

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(__import__("yaml").safe_dump(cfg, sort_keys=False))
    (out_dir / "train_files.json").write_text(json.dumps({"train": train_files, "val": val_files, "bootstrap_seed": bootstrap_seed}))
    log_path = out_dir / "metrics.csv"
    log_f = log_path.open("w", newline=""); writer = None
    best = {"metric": float("inf"), "epoch": 0}
    print(f"[plant m{member}] train views {len(train_ds)} ({len(set(train_files))} unique files), val {len(val_ds)}, "
          f"K={K} Nx={train_ds.Nx}, epochs {n_epochs}, lr {lr}, device {device}, finetune={finetune}", flush=True)
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(epoch - 1, n_epochs, warmup, lr)
        model.train()
        losses, lr_, lq_ = [], [], []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss, l_rho, l_q = plant_loss(model, batch, outflow_weight, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(float(loss.detach())); lr_.append(l_rho); lq_.append(l_q)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_loss_rho": float(np.mean(lr_)),
               "train_loss_q": float(np.mean(lq_)), "lr": opt.param_groups[0]["lr"], "wall_s": time.time() - t0}
        if epoch % eval_every == 0 or epoch == n_epochs or epoch == 1:
            row.update(evaluate_full_grid(model, val_ds, norm, device))
            metric = row["val_rel_l2"]
            if metric < best["metric"]:
                best = {"metric": metric, "epoch": epoch, **{k: v for k, v in row.items() if k.startswith("val_")}}
                _save(out_dir / "best.pt", model, cfg, norm, epoch, train_files, member, seed, best, bootstrap_seed)
            print(f"[plant m{member}] ep {epoch:3d} loss {row['train_loss']:.4f} (rho {row['train_loss_rho']:.4f} q {row['train_loss_q']:.5f}) "
                  f"val relL2 rho {row['val_rel_l2_density']:.4f} q {row['val_rel_l2_flow']:.4f} best {best['metric']:.4f}@{best['epoch']} "
                  f"[{row['wall_s']:.0f}s]", flush=True)
        if writer is None:
            writer = csv.DictWriter(log_f, fieldnames=list(row.keys()) + [k for k in ("val_rel_l2_density", "val_rel_l2_flow", "val_mse_density", "val_outflow_sum_err", "val_rel_l2") if k not in row])
            writer.writeheader()
        writer.writerow(row); log_f.flush()
    log_f.close()
    _save(out_dir / "last.pt", model, cfg, norm, n_epochs, train_files, member, seed, best, bootstrap_seed)
    (out_dir / "best.json").write_text(json.dumps(best, indent=1))
    print(f"[plant m{member}] done: best val rel-L2 {best['metric']:.4f} at epoch {best['epoch']} ({time.time() - t0:.0f} s)")
    return best


def _save(path: Path, model, cfg, norm, epoch, train_files, member, seed, best, bootstrap_seed) -> None:
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "normalization": norm.to_dict(), "epoch": int(epoch),
                "bootstrap_files": list(train_files), "member": int(member), "seed": int(seed), "best": best,
                "bootstrap_seed": bootstrap_seed}, str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one plant-model DeepONet member")
    ap.add_argument("--config", required=True)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--bootstrap-seed", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--new-rounds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--store-dir", default=None, help="override data.store_dir")
    ap.add_argument("--max-train-files", type=int, default=None, help="E1 data-size study: cap the round-0 train files")
    ap.add_argument("--train-types", nargs="*", default=None, help="E1 data-source study: keep only these controller types")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    cfg = load_config(str(_resolve(args.config, root)))
    cfg["project_root"] = str(root)
    if args.store_dir:
        cfg["data"]["store_dir"] = args.store_dir
    train_member(cfg, args.member, args.bootstrap_seed, _resolve(args.out_dir, root),
                 resume=None if args.resume is None else _resolve(args.resume, root), finetune=args.finetune,
                 new_rounds=args.new_rounds, project_root=root, epochs_override=args.epochs,
                 max_train_files=args.max_train_files, train_types=args.train_types)


if __name__ == "__main__":
    main()
