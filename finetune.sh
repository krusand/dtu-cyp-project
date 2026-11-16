#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_finetune
#BSUB -u s253471@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -o outputs/cyp_%J.out
#BSUB -e outputs/cyp_%J.err
set -euo pipefail

CACHE_ROOT="${BLACKHOLE:-$PWD}"
export UV_CACHE_DIR="$CACHE_ROOT/.cache/uv"
export HF_HOME="$CACHE_ROOT/.hf_cache"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

nvidia-smi

uv run --no-sync python -u finetune.py device=cuda ${ARGS:-}