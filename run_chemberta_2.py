import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASK = "both"
DEVICE = "cuda"
EPOCHS = 20
SEED = 42

CONFIGS = ["B", "C", "D", "E"]
DRY_RUN = False

# The best found learning rates for each task
LEARNING_RATES = {
    "regression": (5e-6, 1e-4),
    "classification": (2e-5, 1e-3),
}

# number of layers, hidden dimension, dropout rate
READOUTS = {
    "A": (3, 512, 0.1),  # Existing reference
    "B": (2, 512, 0.1),  # Shallower
    "C": (3, 128, 0.1),  # Narrower
    "D": (3, 512, 0.3),  # Stronger dropout
    "E": (2, 128, 0.3),  # Smaller + stronger dropout
}


def main():

    tasks = (
        ["regression", "classification"]
        if TASK == "both" else [TASK]
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary_dir = ROOT / "runs" / "chemberta_readouts" / stamp
    records = []

    for task in tasks:

        encoder_lr, readout_lr = LEARNING_RATES[task]

        for name in CONFIGS:

            layers, width, dropout = READOUTS[name]

            relative_dir = (
                Path("runs") / task / "chemberta"
                / f"readout_{stamp}_{name}_seed{SEED}"
            )
            result_dir = ROOT / relative_dir

            command = [
                sys.executable,
                "-u",
                str(ROOT / "finetune.py"),
                "model=chemberta",
                "readout=mlp",
                f"task={task}",
                f"device={DEVICE}",
                f"seed={SEED}",
                "model.freeze_encoder=False",
                f"model.encoder_lr={encoder_lr}",
                f"model.readout_lr={readout_lr}",
                f"readout.num_layers={layers}",
                f"readout.hidden_dim={width}",
                f"readout.dropout={dropout}",
                "train.batch_size=16",
                f"train.epochs={EPOCHS}",
                "train.weight_decay=1e-4",
                "train.grad_clip=1.0",
                f"hydra.run.dir={relative_dir.as_posix()}",
            ]

            print(
                f"\n{task}, configuration {name}, seed {SEED}: "
                f"layers={layers}, width={width}, dropout={dropout}",
                flush=True,
            )
            print("Results:", result_dir, flush=True)

            if DRY_RUN:
                print(subprocess.list2cmdline(command))
                continue

            started = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT)

            metric = (
                "macro_ST-RAE"
                if task == "regression"
                else "macro_MCC"
            )

            score = None
            metrics_file = result_dir / "validation_metrics.json"

            if completed.returncode == 0 and metrics_file.exists():
                score = json.loads(
                    metrics_file.read_text()
                ).get(metric)

            records.append({
                "task": task,
                "configuration": name,
                "num_layers": layers,
                "hidden_dim": width,
                "dropout": dropout,
                "encoder_lr": encoder_lr,
                "readout_lr": readout_lr,
                "seed": SEED,
                "epochs": EPOCHS,
                "return_code": completed.returncode,
                "seconds": round(time.monotonic() - started, 1),
                "validation_metric": metric,
                "validation_score": score,
                "result_directory": str(result_dir),
            })

            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_file = summary_dir / "summary.csv"

            with summary_file.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=list(records[0])
                )
                writer.writeheader()
                writer.writerows(records)

            print("Summary:", summary_file, flush=True)

            if completed.returncode != 0:
                raise SystemExit(
                    "Training failed"
                )

if __name__ == "__main__":
    main()
