import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "runs/chemberta_hparam/tier1_bs16"


for task, metric, highest_wins in (
    ("regression", "macro_ST-RAE", False),
    ("classification", "macro_MCC", True),
):
    runs = []
    for metrics_file in (ROOT / task).glob("*/validation_metrics.json"):
        try:
            metrics = json.loads(metrics_file.read_text())
            score = metrics.get(metric)
            if score is None or not math.isfinite(float(score)):
                continue
            config_file = metrics_file.parent / "config.json"
            config = json.loads(config_file.read_text())
            runs.append((float(score), config, metrics_file.parent))
        except (OSError, ValueError, TypeError) as error:
            print(f"Skipping {metrics_file}: {error}")

    print(f"\n{task}: {len(runs)} valid completed runs")
    if not runs:
        print("No completed runs found yet.")
        continue

    best = (max if highest_wins else min)(runs, key=lambda run: run[0])
    score, config, run_dir = best
    print(f"Best {metric}: {score:.6f}")
    print(f"Run: {run_dir}")
    print(
        f"encoder_lr={config['model']['encoder_lr']}, "
        f"readout_lr={config['model']['readout_lr']}, "
        f"batch_size={config['train']['batch_size']}, "
        f"dropout={config['readout']['dropout']}"
    )
