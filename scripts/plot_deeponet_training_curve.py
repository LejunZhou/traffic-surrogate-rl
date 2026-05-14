from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

run_dir = Path("runs/surrogate/deeponet_constant_inflow_20260511_220310")  # change this
df = pd.read_csv(run_dir / "metrics.csv")

plt.figure(figsize=(7, 4))
plt.semilogy(df["step"], df["train_mse"], label="train MSE")

if "val_mse" in df.columns:
    val = df.dropna(subset=["val_mse"])
    plt.semilogy(val["step"], val["val_mse"], marker="o", markersize=3, label="val MSE")

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

out = run_dir / "training_curve.png"
plt.savefig(out, dpi=300)
print(f"saved {out}")