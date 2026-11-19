import json
from pathlib import Path

runs = {"regression": [], "classification": []}

for val_file in Path("atom_jepa_hyperparameter_runs").rglob("validation_metrics.json"):
    run_dir = val_file.parent
    cfg_file = run_dir / "config.json"
    if not cfg_file.exists():
        continue

    cfg = json.loads(cfg_file.read_text())
    if (
        cfg.get("model", {}).get("name") != "atom_jepa"
        or cfg.get("train", {}).get("epochs", 0) <= 2
    ):
        continue

    task = cfg["task"]
    metric = "macro_ST-RAE" if task == "regression" else "macro_MCC"
    val_score = json.loads(val_file.read_text()).get(metric)
    if val_score is None:
        continue

    test_file = run_dir / "test_metrics.json"
    test_score = json.loads(test_file.read_text()).get(metric) if test_file.exists() else None

    runs[task].append({
        "val_score": val_score,
        "test_score": test_score,
        "run_dir": str(run_dir),
        "params": {
            "encoder_lr": cfg["model"]["encoder_lr"],
            "readout_lr": cfg["model"]["readout_lr"],
            "batch_size": cfg["train"]["batch_size"],
            "dropout": cfg["readout"]["dropout"],
            "weight_decay": cfg["train"]["weight_decay"],
        },
    })

top3 = {}
for task, reverse in [("regression", False), ("classification", True)]:
    top3[task] = sorted(runs[task], key=lambda r: r["val_score"], reverse=reverse)[:3]
    print(f"\nTop 3 {task}:")
    for i, r in enumerate(top3[task], 1):
        print(f"  {i}. val={r['val_score']:.4f} | test={r['test_score']} | {r['params']} ({r['run_dir']})")

Path("top3_atom_jepa.json").write_text(json.dumps(top3, indent=2))