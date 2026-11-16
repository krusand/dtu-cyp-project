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

# Best learning rates from the first sweep.
LEARNING_RATES = {
    "regression": (5e-6, 1e-4),
    "classification": (2e-5, 1e-3),
}

# batch_size, weight_decay
TRAINING_CONFIGS = {
    "F": (8, 1e-4),
    "G": (32, 1e-4),
    "H": (16, 0.0),
    "I": (16, 1e-3),
    "J": (16, 1e-2),
}


def main():
    tasks = (
        ["regression", "classification"]
        if TASK == "both" else [TASK]
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary_dir = ROOT / "runs" / "chemberta_training" / stamp
    records = []

    print(f"Experiments: {len(tasks) * len(TRAINING_CONFIGS)}")
    print(f"Device: {DEVICE}")

    for task in tasks:
        encoder_lr, readout_lr = LEARNING_RATES[task]

        for name, (batch_size, weight_decay) in TRAINING_CONFIGS.items():
            relative_dir = (
                Path("runs") / task / "chemberta"
                / f"training_{stamp}_{name}_seed{SEED}"
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
                # Readout D from the second sweep.
                "readout.num_layers=3",
                "readout.hidden_dim=512",
                "readout.dropout=0.3",
                f"train.batch_size={batch_size}",
                f"train.weight_decay={weight_decay}",
                f"train.epochs={EPOCHS}",
                "train.grad_clip=1.0",
                f"hydra.run.dir={relative_dir.as_posix()}",
            ]

            print(
                f"\n{task}, configuration {name}, seed {SEED}: "
                f"batch_size={batch_size}, weight_decay={weight_decay}",
                flush=True,
            )
            print("Results:", result_dir, flush=True)

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
                score = json.loads(metrics_file.read_text()).get(metric)

            records.append({
                "task": task,
                "configuration": name,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "num_layers": 3,
                "hidden_dim": 512,
                "dropout": 0.3,
                "encoder_lr": encoder_lr,
                "readout_lr": readout_lr,
                "seed": SEED,
                "epochs": EPOCHS,
                "device": DEVICE,
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
                    "Training failed. Check the error above; "
                    "remaining experiments were not started."
                )


if __name__ == "__main__":
    main()