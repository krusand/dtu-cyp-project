import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, matthews_corrcoef

run_dir = Path("atom_jepa_hyperparameter_runs/tier2/classification/atom_jepa/2026-11-22_10-40-33")
test_true = pd.read_csv("data/classification/test.csv")
test_pred = pd.read_csv(run_dir / "test_predictions.csv")

# Endpoint columns are everything in test_predictions.csv except SMILES and Molecule_Name
targets = [c for c in test_pred.columns if c not in ("SMILES", "Molecule_Name")]

# 1. Per-endpoint normalized confusion matrices
fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 3.4))
if len(targets) == 1:
    axes = [axes]

for ax, t in zip(axes, targets):
    y = test_true[t].to_numpy(dtype=float)
    p = test_pred[t].to_numpy(dtype=float)
    mask = np.isfinite(y)
    y_bin = y[mask].astype(int)
    p_bin = (p[mask] >= 0.5).astype(int)

    cm = confusion_matrix(y_bin, p_bin, normalize="true")
    mcc = matthews_corrcoef(y_bin, p_bin)
    ConfusionMatrixDisplay(cm, display_labels=["Non-inh.", "Inhib."]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format=".2f"
    )
    ax.set_title(f"{t}\n(n={len(y_bin)}, MCC={mcc:.2f})")

plt.tight_layout()
plt.savefig("best_model_confusion_matrices.png", dpi=300)
plt.close()

# 2. Training loss & validation MCC curve from history.jsonl
history = pd.DataFrame([json.loads(line) for line in (run_dir / "history.jsonl").open()])

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(history["epoch"], history["train_loss"], color="tab:blue", marker="o", label="Train Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train BCE Loss", color="tab:blue")

ax2 = ax1.twinx()
ax2.plot(history["epoch"], history["macro_MCC"], color="tab:orange", marker="s", label="Val macro-MCC")
ax2.set_ylabel("Validation macro-MCC", color="tab:orange")

plt.title("Training Loss vs. Validation macro-MCC")
fig.tight_layout()
plt.savefig("best_model_learning_curve.png", dpi=300)
plt.close()