#!/bin/bash
#BSUB -q hpc
#BSUB -J cyp_prepare
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 08:00
#BSUB -o outputs/cyp_prepare_%J.out
#BSUB -e outputs/cyp_prepare_%J.err
set -euo pipefail
pwd
uv sync --extra cpu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Download labelled training tables, generate conformers, and split train/val/test.
# Rerunning reuses existing conformer files.
uv run --no-sync python -u prepare_data.py --out data --workers 8 --seed 42 --num-conformers 50
