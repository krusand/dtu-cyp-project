#!/bin/bash
#BSUB -q hpc
#BSUB -J cyp_prepare
#BSUB -u s253844@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 04:00
#BSUB -o outputs/cyp_prepare_%J.out
#BSUB -e outputs/cyp_prepare_%J.err
set -euo pipefail
pwd
uv sync --extra cpu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Download labelled training tables, generate conformers, and split train/val/test.
# Rerunning reuses existing conformer files.
uv run --no-sync python -u prepare_data.py --out data --workers 2 --seed 42
