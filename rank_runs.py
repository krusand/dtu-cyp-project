import json
from pathlib import Path


def collect_runs():
    runs = {"regression": [], "classification": []}
    for root in ("multirun", "runs"):
        base = Path(root)
        if not base.exists():
            continue
        for val_file in base.rglob("validation_metrics.json"):
            run_dir = val_file.parent
            cfg_file = run_dir / "config.json"
            if not cfg_file.exists():
                continue

            cfg = json.loads(cfg_file.read_text())
            if cfg.get("model", {}).get("name") != "atom_jepa":
                continue
            if cfg.get("train", {}).get("epochs", 0) <= 2:
                continue

            task = cfg["task"]
            val_metrics = json.loads(val_file.read_text())
            test_file = run_dir / "test_metrics.json"
            test_metrics = json.loads(test_file.read_text()) if test_file.exists() else {}

            metric_key = "macro_ST-RAE" if task == "regression" else "macro_MCC"
            val_score = val_metrics.get(metric_key)
            if val_score is None:
                continue

            runs[task].append({
                "val_score": val_score,
                "test_score": test_metrics.get(metric_key),
                "run_dir": str(run_dir),
                "checkpoint": str(run_dir / "best.pt"),
                "params": {
                    "encoder_lr": cfg["model"]["encoder_lr"],
                    "readout_lr": cfg["model"]["readout_lr"],
                    "batch_size": cfg["train"]["batch_size"],
                    "dropout": cfg["readout"]["dropout"],
                    "weight_decay": cfg["train"]["weight_decay"],
                    "epochs": cfg["train"]["epochs"],
                },
            })
    return runs


def main():
    runs = collect_runs()
    top_results = {}

    for task, reverse in [("regression", False), ("classification", True)]:
        ranked = sorted(runs[task], key=lambda r: r["val_score"], reverse=reverse)[:3]
        top_results[task] = ranked
        metric_name = "macro_ST-RAE" if task == "regression" else "macro_MCC"

        print(f"\nTop 3 {task} runs ({metric_name}):")
        for rank, r in enumerate(ranked, 1):
            p = r["params"]
            test_str = f"{r['test_score']:.4f}" if r["test_score"] is not None else "n/a"
            print(
                f"  {rank}. val={r['val_score']:.4f} (test={test_str}) | "
                f"enc_lr={p['encoder_lr']}, head_lr={p['readout_lr']}, "
                f"bs={p['batch_size']}, dropout={p['dropout']} -> {r['run_dir']}"
            )

    Path("top3_atom_jepa.json").write_text(json.dumps(top_results, indent=2))
    print("\nSaved summary to top3_atom_jepa.json")


if __name__ == "__main__":
    main()