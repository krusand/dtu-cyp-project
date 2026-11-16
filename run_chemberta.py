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
BATCH_SIZE = 16
SEED = 42

SWEEP = True   # when false it runs two experiments using the original settings, when True it runs 16 experiments testing different freezing strategies and learning rates
DRY_RUN = False


# Each tuple: (freeze_encoder, encoder_lr, readout_lr)
experiments = [(False, 1e-5, 1e-3)] # runs these experiments when SWEEP=False

if SWEEP:   # when SWEEP=True
    experiments = [
        (frozen, encoder_lr, readout_lr)
        for frozen in [True, False]
        for encoder_lr in (
            [1e-5] if frozen else [5e-6, 1e-5, 2e-5]
        )
        for readout_lr in [1e-4, 1e-3]
    ]


def main():
    tasks = (
        ["regression", "classification"]
        if TASK == "both" else [TASK]
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary_dir = ROOT / "runs" / "chemberta_local" / stamp
    records = []

    for task in tasks:
        for index, (frozen, encoder_lr, readout_lr) in enumerate(experiments):

            relative_dir = (
                Path("runs") / task / "chemberta"
                / f"{stamp}_{index:02d}"
            )
            result_dir = ROOT / relative_dir

            command = [
                sys.executable,
                "-u",
                str(ROOT / "finetune.py"),
                "model=chemberta",
                f"task={task}",
                f"device={DEVICE}",
                f"seed={SEED}",
                f"train.epochs={EPOCHS}",
                f"train.batch_size={BATCH_SIZE}",
                f"model.freeze_encoder={str(frozen).lower()}",
                f"model.encoder_lr={encoder_lr}",
                f"model.readout_lr={readout_lr}",
                f"hydra.run.dir={relative_dir.as_posix()}",
            ]

            print(
                f"\n{task}: frozen={frozen}, "
                f"encoder_lr={encoder_lr}, readout_lr={readout_lr}",
                flush=True,
            )
            print("Results:", result_dir, flush=True)

            if DRY_RUN:
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
                "freeze_encoder": frozen,
                "encoder_lr": encoder_lr,
                "readout_lr": readout_lr,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "device": DEVICE,
                "seed": SEED,
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

            print("Comparison CSV:", summary_file, flush=True)

            if completed.returncode != 0:
                raise SystemExit(
                    "Training failed"
                )


if __name__ == "__main__":
    main()
