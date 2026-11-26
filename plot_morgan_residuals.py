from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/regression/morgan_rdkit_mlp/2026-11-09_12-52-21" # the selected baseline
KEYS = ["Molecule_Name", "SMILES"]
TARGETS = [
    ("CYP1A2", "CYP1A2_pIC50_direct_inhibition"),
    ("CYP2C9", "CYP2C9_pIC50_direct_inhibition"),
    ("CYP2D6", "CYP2D6_pIC50_direct_inhibition"),
    ("CYP3A4", "CYP3A4_pIC50_direct_inhibition"),
]

#labels = pd.read_csv(ROOT / "data/regression/test.csv")
labels = pd.read_csv(ROOT / "data/regression/val.csv")
#predictions = pd.read_csv(RUN / "test_predictions.csv")
predictions = pd.read_csv(RUN / "validation_predictions.csv")
data = labels.merge(predictions, on=KEYS, suffixes=("", "_pred"), validate="one_to_one")

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, (name, target) in zip(axes.flat, TARGETS):
    pred = f"{target}_pred"
    points = data[[target, pred]].dropna()
    ax.scatter(points[target], points[target] - points[pred], s=13, alpha=0.65)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(title=name, xlabel="Observed pIC50", ylabel="Residual")
    ax.tick_params(axis="both", labelsize=14)

fig.tight_layout()
#output = ROOT / "plots/morgan_rdkit_mlp_residuals_test.png"
output = ROOT / "plots/morgan_rdkit_mlp_residuals_val.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=220)
plt.close(fig)
print(f"Saved: {output}")
