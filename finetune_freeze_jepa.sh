#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cyp_freeze
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

MODEL="${MODEL:-atom_jepa}"
TASK="${TASK:-regression}"
OVERRIDES="${OVERRIDES:-}"

nvidia-smi

if [[ "$TASK" == "classification" ]]; then
    BEST_ARGS="model.encoder_lr=1e-5 model.readout_lr=1e-3 train.batch_size=16 readout.dropout=0.3"
else
    BEST_ARGS="model.encoder_lr=5e-6 model.readout_lr=1e-3 train.batch_size=16 readout.dropout=0.1"
fi

uv run --no-sync python -u finetune.py -m \
    model="$MODEL" \
    readout=mlp \
    task="$TASK" \
    seed=42 \
    device=cuda \
    model.precision=fp32 \
    train.epochs=20 \
    model.freeze_encoder=true,false \
    $BEST_ARGS \
    $OVERRIDES